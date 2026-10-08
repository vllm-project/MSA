# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""MXFP8 quantization of BF16 values: E4M3 data + one UE8M0 scale per 32 values.

The bytes equal FlashInfer's CuTe-DSL MXFP8 quantizer
(``flashinfer/quantization/quantization_cute_dsl_utils.py``, FlashInfer 0.7.0), i.e.
``flashinfer.mxfp8_quantize(x, is_sf_swizzled_layout=True, backend="cute-dsl")``:

- amax: ``max.bf16x2`` over the 32 sign-cleared BF16 values (exact, NaN-ignoring, so the
  reduction order does not matter), as Float32;
- scale: ``amax * RN(1/448)`` rounded up to a power of two (UE8M0, clamped to 254, 0 for
  amax <= 0): FlashInfer's ``float_to_ue8m0_fast`` PTX;
- data: FlashInfer computes ``RN_fp32(x * 2^(127 - e))``, clamps to +-448 and converts with
  ``cvt.rn.satfinite``. Here the same product is formed in packed BF16 (the inverse scale is
  a power of two or 0, so the product is exact wherever E4M3 can represent it), ``min`` with
  448 maps NaN and overflow to 448 as the fp32 clamp does, and ``satfinite`` saturates the
  negative side: equal for every BF16 value and every scale (checked exhaustively);
- scales are stored in the 128x4 swizzled (``F8_128x4``) layout.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

INV_E4M3_MAX = 1.0 / 448.0
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
def bf16x2_max_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    """Max of the two BF16 halves, as Float32."""
    return Float32(
        _asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 f0, f1;
                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 f0, lo;
                mov.b32 f1, hi;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
        )
    )


@dsl_user_op
def f32_to_ue8m0_ceil(value: Float32, *, loc=None, ip=None) -> Uint32:
    """UE8M0 of ``value`` rounded towards +inf, saturated to 254; 0 for value <= 0."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;
                .reg .u32 bits, exp_biased, mantissa, bump, result;

                setp.le.f32 p_zero, $1, 0f00000000;

                mov.b32 bits, $1;
                shr.b32 exp_biased, bits, 23;
                and.b32 exp_biased, exp_biased, 255;
                and.b32 mantissa, bits, 0x7FFFFF;

                setp.ne.u32 p_has_mant, mantissa, 0;
                selp.u32 bump, 1, 0, p_has_mant;
                setp.eq.u32 p_exp_zero, exp_biased, 0;
                setp.le.u32 p_tiny_sub, mantissa, 0x400000;
                and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;
                @p_tiny_sub mov.u32 bump, 0;
                add.u32 result, exp_biased, bump;

                setp.gt.u32 p_ovf, result, 254;
                selp.u32 result, 254, result, p_ovf;
                selp.u32 $0, 0, result, p_zero;
            }
            """,
            "=r,f",
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
    """FlashInfer's ``ue8m0_to_inv_scale_fast(e)`` (2^(127 - e), or 0 for e == 0 and below the
    normal range: always exactly a BF16) in both BF16 halves."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(e).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .s32 new_exp;
                .reg .b32 hi;
                .reg .pred p_zero;

                setp.eq.u32 p_zero, $1, 0;
                sub.s32 new_exp, 254, $1;
                max.s32 new_exp, new_exp, 0;
                shl.b32 hi, new_exp, 7;
                @p_zero mov.b32 hi, 0;
                prmt.b32 $0, hi, hi, 0x1010;
            }
            """,
            "=r,r",
        )
    )


@dsl_user_op
def bf16x2x2_to_e4m3x4_scaled(w0: Uint32, w1: Uint32, inv2: Uint32, *, loc=None, ip=None) -> Uint32:
    """Four BF16 (w0 = values 0-1, w1 = values 2-3) -> four E4M3 bytes, value 0 in the low byte:
    ``RN_bf16(x * inv)``, ``min`` 448, ``cvt.rn.satfinite`` (see the module docstring)."""
    return Uint32(
        _asm(
            T.i32(),
            [
                Uint32(w0).ir_value(loc=loc, ip=ip),
                Uint32(w1).ir_value(loc=loc, ip=ip),
                Uint32(inv2).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 p0, p1, c;
                .reg .b16 q0, q1;
                mov.b32 c, 0x43E043E0;
                mul.rn.bf16x2 p0, $1, $3;
                mul.rn.bf16x2 p1, $2, $3;
                min.bf16x2 p0, p0, c;
                min.bf16x2 p1, p1, c;
                cvt.rn.satfinite.e4m3x2.bf16x2 q0, p0;
                cvt.rn.satfinite.e4m3x2.bf16x2 q1, p1;
                mov.b32 $0, {q0, q1};
            }
            """,
            "=r,r,r,r",
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


@cute.jit
def quantize_store_bf16x32(w: cute.Tensor, q_addr: Int64) -> Uint32:
    """MXFP8 of one 32-value block held by one thread (``w``: 16 BF16x2 words, values in order).

    Stores the 32 E4M3 bytes at ``q_addr`` (16-byte aligned) and returns the UE8M0 scale.
    """
    a16 = [bf16x2_abs(w[i]) for i in range(16)]
    a8 = [bf16x2_max(a16[2 * i], a16[2 * i + 1]) for i in range(8)]
    a4 = [bf16x2_max(a8[2 * i], a8[2 * i + 1]) for i in range(4)]
    a2 = [bf16x2_max(a4[2 * i], a4[2 * i + 1]) for i in range(2)]
    amax = bf16x2_max_to_f32(bf16x2_max(a2[0], a2[1]))
    e = f32_to_ue8m0_ceil(amax * Float32(INV_E4M3_MAX))
    inv2 = ue8m0_to_inv_scale_bf16x2(e)
    q = [bf16x2x2_to_e4m3x4_scaled(w[2 * j], w[2 * j + 1], inv2) for j in range(8)]
    st_global_v4_u32(q_addr, q[0], q[1], q[2], q[3])
    st_global_v4_u32(q_addr + Int64(16), q[4], q[5], q[6], q[7])
    return e


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
    e = f32_to_ue8m0_ceil(bf16x2_max_to_f32(local) * Float32(INV_E4M3_MAX))
    inv2 = ue8m0_to_inv_scale_bf16x2(e)
    return (
        bf16x2x2_to_e4m3x4_scaled(w0, w1, inv2),
        bf16x2x2_to_e4m3x4_scaled(w2, w3, inv2),
        e,
    )


@cute.jit
def pack_scales_x4_lanes_by_4(e: Uint32) -> Uint32:
    """The scales of 4 consecutive blocks held by lanes L, L + 4, L + 8, L + 12 packed into lane
    L (little endian, block 0 lowest): the 4 contiguous bytes of a 4-column group in the 128x4
    layout. Every lane of the warp must call it; only lane L's result is meaningful."""
    e1 = Uint32(cute.arch.shuffle_sync_down(e, offset=4))
    e2 = Uint32(cute.arch.shuffle_sync_down(e, offset=8))
    e3 = Uint32(cute.arch.shuffle_sync_down(e, offset=12))
    return Uint32(e) | (e1 << Uint32(8)) | (e2 << Uint32(16)) | (e3 << Uint32(24))


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
