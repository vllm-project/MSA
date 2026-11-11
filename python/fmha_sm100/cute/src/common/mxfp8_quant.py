# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""MXFP8 quantization of BF16 values: E4M3 data + one UE8M0 scale per 32 values.

The bytes equal FlashInfer's CuTe-DSL MXFP8 quantizer
(``flashinfer/quantization/quantization_cute_dsl_utils.py``, FlashInfer 0.7.0), i.e.
``flashinfer.mxfp8_quantize(x, is_sf_swizzled_layout=True, backend="cute-dsl")``:

- amax: ``max.bf16x2`` over the 32 sign-cleared BF16 values (exact, NaN-ignoring, so the
  reduction order does not matter);
- scale: FlashInfer rounds ``amax * RN(1/448)`` up to a power of two (UE8M0, clamped to 254, 0 for
  amax <= 0). As 448 = 1.75 * 2^8 that is the BF16 exponent less 8, plus one when the mantissa is
  above 0.75, with the exponent 255 of Inf and NaN mapping to 254: equal for every BF16 magnitude
  (checked exhaustively);
- data: FlashInfer computes ``RN_fp32(x * 2^(127 - e))``, clamps to +-448 and converts with
  ``cvt.rn.satfinite``. Here the same product is formed in packed BF16 (the inverse scale is
  a power of two or 0, so the product is exact wherever E4M3 can represent it), ``min`` with
  448 maps NaN and overflow to 448 as the fp32 clamp does, and ``satfinite`` saturates the
  negative side: equal for every BF16 value and every scale (checked exhaustively);
- scales are stored in the 128x4 swizzled (``F8_128x4``) layout.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

MXFP8_BLOCK = 32


def _asm(ret, args, body, constraints, side_effects=False):
    return llvm.inline_asm(
        ret,
        args,
        body,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def bf16x2_abs(x: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(T.i32(), [Uint32(x).ir_value(loc=loc, ip=ip)], "and.b32 $0, $1, 0x7FFF7FFF;", "=r,r")
    )


@dsl_user_op
def bf16x2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
        )
    )


@dsl_user_op
def bf16_amax_to_ue8m0(pair: Uint32, *, loc=None, ip=None) -> Uint32:
    """UE8M0 of amax / 448 rounded up, from sign-cleared BF16 bits in both halves of ``pair``."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(pair).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_up, p_special;
                .reg .b32 lo, hi, rot, amax;
                .reg .u32 exponent, mantissa;
                .reg .s32 scale;

                shl.b32 lo, $1, 16;
                shr.b32 hi, $1, 16;
                or.b32 rot, lo, hi;
                max.bf16x2 amax, $1, rot;
                and.b32 amax, amax, 0x7FFF;
                shr.u32 exponent, amax, 7;
                and.b32 mantissa, amax, 0x7F;
                sub.s32 scale, exponent, 8;
                setp.gt.u32 p_up, mantissa, 96;
                @p_up add.s32 scale, scale, 1;
                max.s32 scale, scale, 0;
                setp.eq.u32 p_special, exponent, 255;
                selp.s32 $0, 254, scale, p_special;
            }
            """,
            "=r,r",
        )
    )


@dsl_user_op
def st_global_u32(addr: Int64, value: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        side_effects=True,
    )


@dsl_user_op
def st_global_u8(addr: Int64, value: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "st.global.u8 [$0], $1;",
        "l,r",
        side_effects=True,
    )


@cute.jit
def sf_offset_128x4(row: Int32, col: Int32, padded_cols: Int32) -> Int32:
    """Byte offset of scale (row, col) in the 128x4 swizzled layout ([M/128, cols/4, 32, 4, 4])."""
    return (
        col % Int32(4)
        + (col // Int32(4)) * Int32(512)
        + (row % Int32(32)) * Int32(16)
        + ((row % Int32(128)) // Int32(32)) * Int32(4)
        + (row // Int32(128)) * (Int32(128) * padded_cols)
    )


@dsl_user_op
def ue8m0_to_inv_scale_bf16x2(e: Uint32, *, loc=None, ip=None) -> Uint32:
    """FlashInfer's ``ue8m0_to_inv_scale_fast(e)`` in both BF16 halves: 2^(127 - e), or 0 for e == 0
    and for e == 254, whose exponent is below the normal range."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(e).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero;
                .reg .u32 inv;

                setp.eq.u32 p_zero, $1, 0;
                sub.u32 inv, 254, $1;
                mul.lo.u32 inv, inv, 0x00800080;
                selp.u32 $0, 0, inv, p_zero;
            }
            """,
            "=r,r",
        )
    )


@dsl_user_op
def bf16x2x2_to_e4m3x4_scaled(
    w0: Uint32, w1: Uint32, inv2: Uint32, clamp: Uint32, *, loc=None, ip=None
) -> Uint32:
    """Four BF16 (w0 = values 0-1, w1 = values 2-3) -> four E4M3 bytes, value 0 in the low byte:
    ``RN_bf16(x * inv)``, ``min`` 448 (``clamp``), ``cvt.rn.satfinite`` (see the module docstring)."""
    return Uint32(
        _asm(
            T.i32(),
            [
                Uint32(w0).ir_value(loc=loc, ip=ip),
                Uint32(w1).ir_value(loc=loc, ip=ip),
                Uint32(inv2).ir_value(loc=loc, ip=ip),
                Uint32(clamp).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 p0, p1;
                .reg .b16 q0, q1;
                mul.rn.bf16x2 p0, $1, $3;
                mul.rn.bf16x2 p1, $2, $3;
                min.bf16x2 p0, p0, $4;
                min.bf16x2 p1, p1, $4;
                cvt.rn.satfinite.e4m3x2.bf16x2 q0, p0;
                cvt.rn.satfinite.e4m3x2.bf16x2 q1, p1;
                mov.b32 $0, {q0, q1};
            }
            """,
            "=r,r,r,r,r",
        )
    )


@dsl_user_op
def st_global_v4_u32(addr: Int64, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(a).ir_value(loc=loc, ip=ip),
            Uint32(b).ir_value(loc=loc, ip=ip),
            Uint32(c).ir_value(loc=loc, ip=ip),
            Uint32(d).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side_effects=True,
    )


@dsl_user_op
def st_global_v2_u32(addr: Int64, a: Uint32, b: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(a).ir_value(loc=loc, ip=ip),
            Uint32(b).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        side_effects=True,
    )


@cute.jit
def quantize_bf16x8_lane_of_4(w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32):
    """MXFP8 of a 32-value block spread over 4 adjacent lanes, 8 values per lane (``w0..w3``:
    BF16x2 words in column order). Returns this lane's 8 E4M3 bytes as 2 words (value 0
    lowest) and the block's UE8M0 scale. Every lane of the warp must call it."""
    local = bf16x2_max(
        bf16x2_max(bf16x2_abs(w0), bf16x2_abs(w1)), bf16x2_max(bf16x2_abs(w2), bf16x2_abs(w3))
    )
    # max.bf16x2 is exact and NaN-ignoring, so reducing across the lanes in BF16 gives the
    # same amax as FlashInfer's fp32 reduction.
    local = bf16x2_max(local, Uint32(cute.arch.shuffle_sync_bfly(local, offset=1)))
    local = bf16x2_max(local, Uint32(cute.arch.shuffle_sync_bfly(local, offset=2)))
    e = bf16_amax_to_ue8m0(local)
    inv2 = ue8m0_to_inv_scale_bf16x2(e)
    clamp = Uint32(0x43E043E0)
    return (
        bf16x2x2_to_e4m3x4_scaled(w0, w1, inv2, clamp),
        bf16x2x2_to_e4m3x4_scaled(w2, w3, inv2, clamp),
        e,
    )


@cute.jit
def transpose_e4m3_pairs_lane_of_4(q0: Uint32, q1: Uint32, lane: Int32):
    """The 4x4 transpose of 16-bit E4M3 pairs across 4 adjacent lanes (``q0``, ``q1``: two pairs
    per word, ``lane`` in [0, 4)): an exchange of halves with the neighbour lane, then of words
    with the lane two away. Every lane of the warp must call it."""
    keep_half = Uint32(0xFFFF) << Uint32((lane & Int32(1)) * Int32(16))
    take_half = keep_half ^ Uint32(0xFFFFFFFF)
    p0 = Uint32(cute.arch.shuffle_sync_bfly(q0, offset=1))
    p1 = Uint32(cute.arch.shuffle_sync_bfly(q1, offset=1))
    s0 = (q0 & keep_half) | (((p0 >> Uint32(16)) | (p0 << Uint32(16))) & take_half)
    s1 = (q1 & keep_half) | (((p1 >> Uint32(16)) | (p1 << Uint32(16))) & take_half)
    upper = Uint32(Int32(0) - ((lane >> Int32(1)) & Int32(1)))
    lower = upper ^ Uint32(0xFFFFFFFF)
    other = Uint32(cute.arch.shuffle_sync_bfly((s1 & lower) | (s0 & upper), offset=2))
    return (s0 & lower) | (other & upper), (other & lower) | (s1 & upper)


@cute.jit
def zero_padding_scales(
    base: Int64,
    rows: Int32,
    padded_cols: Int32,
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
):
    """Zero the scales of the padding rows [rows, round_up(rows, 128)) (the quantizer's padding).

    In a 128-row tile the 16 bytes at ``group * 512 + i * 16`` hold rows i, i + 32, i + 64 and
    i + 96 of a 4-column group: a chunk whose four rows are all padding is cleared with one
    16-byte store, and consecutive threads clear consecutive chunks. ``base`` must be 16-byte
    aligned."""
    tail = rows % Int32(128)
    if tail != Int32(0):
        tile = base + Int64(rows // Int32(128)) * Int64(128) * Int64(padded_cols)
        chunks = (padded_cols // Int32(4)) * Int32(32)
        c = tidx
        while c < chunks:
            i = c % Int32(32)
            addr = tile + Int64((c // Int32(32)) * Int32(512) + i * Int32(16))
            if i >= tail:
                st_global_v4_u32(addr, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
            else:
                for j in cutlass.range_constexpr(1, 4):
                    if i + Int32(32 * j) >= tail:
                        st_global_u32(addr + Int64(4 * j), Uint32(0))
            c = c + Int32(num_threads)
