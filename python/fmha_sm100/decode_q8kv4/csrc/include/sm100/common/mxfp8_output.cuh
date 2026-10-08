// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

// MXFP8 form of the BF16 attention output: E4M3 data plus one UE8M0 scale per 32 values of a
// row, scales in FlashInfer's 128x4 swizzled layout with the padding rows zeroed. Bit for bit
// what mxfp8_quantize(out.view(tokens, -1), is_sf_swizzled_layout=True) gives for the BF16 output:
// the amax of the 32 BF16 values, amax * RN(1/448) rounded up to a power of two (UE8M0), and
// x * 2^(127 - e) clamped to 448 and converted with satfinite. The fp32 max and multiply are PTX
// without .ftz, so the decode JIT's fast-math flags do not change them.
namespace fmha_sm100::decode_q8kv4::sm100::common {

inline constexpr int kMxfp8Block = 32;
inline constexpr int kMxfp8RowTile = 128;

CUTLASS_DEVICE uint32_t bf16x2_max_abs(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a & 0x7FFF7FFFu), "r"(b & 0x7FFF7FFFu));
  return r;
}

CUTLASS_DEVICE uint32_t bf16x2_max(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

CUTLASS_DEVICE float f32_max(float a, float b) {
  float r;
  asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// The larger of the two BF16 halves as fp32.
CUTLASS_DEVICE float bf16x2_hmax_to_f32(uint32_t x) {
  return f32_max(__uint_as_float(x << 16), __uint_as_float(x & 0xFFFF0000u));
}

// UE8M0 of amax / 448: the biased exponent of amax * RN(1/448), plus one when the mantissa is
// nonzero (round up), clamped to [0, 254]; 0 for zero or negative.
CUTLASS_DEVICE uint32_t amax_to_ue8m0(float amax) {
  float value;
  asm("mul.rn.f32 %0, %1, 0f3B124925;" : "=f"(value) : "f"(amax));
  uint32_t result;
  asm("{\n"
      ".reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n"
      ".reg .u32 bits, exp_biased, mantissa, bump, res;\n"
      "setp.le.f32 p_zero, %1, 0f00000000;\n"
      "mov.b32 bits, %1;\n"
      "shr.b32 exp_biased, bits, 23;\n"
      "and.b32 exp_biased, exp_biased, 255;\n"
      "and.b32 mantissa, bits, 0x7FFFFF;\n"
      "setp.ne.u32 p_has_mant, mantissa, 0;\n"
      "selp.u32 bump, 1, 0, p_has_mant;\n"
      "setp.eq.u32 p_exp_zero, exp_biased, 0;\n"
      "setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n"
      "and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n"
      "@p_tiny_sub mov.u32 bump, 0;\n"
      "add.u32 res, exp_biased, bump;\n"
      "setp.gt.u32 p_ovf, res, 254;\n"
      "selp.u32 res, 254, res, p_ovf;\n"
      "selp.u32 %0, 0, res, p_zero;\n"
      "}\n"
      : "=r"(result)
      : "f"(value));
  return result;
}

// 2^(127 - e) as BF16 in both halves (0 for e == 0), exactly representable.
CUTLASS_DEVICE uint32_t ue8m0_to_inv_scale_bf16x2(uint32_t e) {
  uint32_t r;
  asm("{\n"
      ".reg .s32 new_exp;\n"
      ".reg .b32 hi;\n"
      ".reg .pred p_zero;\n"
      "setp.eq.u32 p_zero, %1, 0;\n"
      "sub.s32 new_exp, 254, %1;\n"
      "max.s32 new_exp, new_exp, 0;\n"
      "shl.b32 hi, new_exp, 7;\n"
      "@p_zero mov.b32 hi, 0;\n"
      "prmt.b32 %0, hi, hi, 0x1010;\n"
      "}\n"
      : "=r"(r)
      : "r"(e));
  return r;
}

// Four BF16 (w0 = values 0-1, w1 = 2-3) times the power-of-two inverse scale -> four E4M3 bytes,
// value 0 lowest. The product is exact in BF16 wherever E4M3 can represent it; min 448 maps NaN
// and overflow to 448 like an fp32 clamp, and satfinite saturates the negative side.
CUTLASS_DEVICE uint32_t bf16x4_to_e4m3x4(uint32_t w0, uint32_t w1, uint32_t inv2) {
  uint32_t r;
  asm("{\n"
      ".reg .b32 p0, p1, c;\n"
      ".reg .b16 q0, q1;\n"
      "mov.b32 c, 0x43E043E0;\n"
      "mul.rn.bf16x2 p0, %1, %3;\n"
      "mul.rn.bf16x2 p1, %2, %3;\n"
      "min.bf16x2 p0, p0, c;\n"
      "min.bf16x2 p1, p1, c;\n"
      "cvt.rn.satfinite.e4m3x2.bf16x2 q0, p0;\n"
      "cvt.rn.satfinite.e4m3x2.bf16x2 q1, p1;\n"
      "mov.b32 %0, {q0, q1};\n"
      "}\n"
      : "=r"(r)
      : "r"(w0), "r"(w1), "r"(inv2));
  return r;
}

// UE8M0 scale of a 32-value block whose BF16 values are spread over kLanes adjacent lanes, each
// holding kWords packed BF16 pairs (kLanes * kWords * 2 == 32). Every lane of the warp must call
// it; the lanes of a block get the same scale.
template <int kLanes>
CUTLASS_DEVICE float mxfp8_block_amax(float local_amax) {
  CUTLASS_PRAGMA_UNROLL
  for (int offset = 1; offset < kLanes; offset *= 2) {
    local_amax = f32_max(local_amax, __shfl_xor_sync(0xFFFFFFFFu, local_amax, offset));
  }
  return local_amax;
}

template <int kWords>
CUTLASS_DEVICE uint32_t mxfp8_block_scale(uint32_t const (&words)[kWords]) {
  constexpr int kLanes = kMxfp8Block / (2 * kWords);
  static_assert(kWords % 2 == 0 && kLanes * kWords * 2 == kMxfp8Block,
                "an MXFP8 block must be whole lanes of an even number of packed BF16 pairs");
  uint32_t local = bf16x2_max_abs(words[0], words[1]);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 2; i < kWords; i += 2) {
    local = bf16x2_max(local, bf16x2_max_abs(words[i], words[i + 1]));
  }
  return amax_to_ue8m0(mxfp8_block_amax<kLanes>(bf16x2_hmax_to_f32(local)));
}

// One BF16 value per lane, 32 lanes per block (the split-KV reduction): returns the E4M3 byte;
// `scale` receives the block's UE8M0 scale.
CUTLASS_DEVICE uint8_t mxfp8_quantize_lane(uint16_t value, uint32_t &scale) {
  float const amax = __uint_as_float(static_cast<uint32_t>(value & 0x7FFFu) << 16);
  scale = amax_to_ue8m0(mxfp8_block_amax<kMxfp8Block>(amax));
  uint32_t const pair = static_cast<uint32_t>(value) * 0x00010001u;
  return static_cast<uint8_t>(bf16x4_to_e4m3x4(pair, pair, ue8m0_to_inv_scale_bf16x2(scale)));
}

// E4M3 bytes of this lane's kWords BF16 pairs under the block's scale: kWords / 2 words.
template <int kWords>
CUTLASS_DEVICE void mxfp8_quantize_words(uint32_t const (&words)[kWords], uint32_t scale,
                                         uint32_t (&out)[kWords / 2]) {
  static_assert(kWords % 2 == 0, "E4M3 words hold four values");
  uint32_t const inv2 = ue8m0_to_inv_scale_bf16x2(scale);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < kWords / 2; ++i) {
    out[i] = bf16x4_to_e4m3x4(words[2 * i], words[2 * i + 1], inv2);
  }
}

// The four block scales of a 128-value row whose blocks start every kLanesPerBlock lanes from the
// row's first lane, packed in column order (the four contiguous bytes of the 128x4 layout) into
// that lane; other lanes get garbage. The row must lie within one warp.
template <int kLanesPerBlock>
CUTLASS_DEVICE uint32_t mxfp8_pack_row_scales(uint32_t scale) {
  static_assert(4 * kLanesPerBlock <= 32, "a row's four blocks must lie within one warp");
  uint32_t packed = scale;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 1; j < 4; ++j) {
    packed |= __shfl_down_sync(0xFFFFFFFFu, scale, j * kLanesPerBlock) << (8 * j);
  }
  return packed;
}

// Byte offset of scale (row, col_block) in the 128x4 swizzled layout with scale_cols columns
// (a multiple of 4).
CUTLASS_DEVICE int64_t mxfp8_scale_offset(int row, int col_block, int scale_cols) {
  return static_cast<int64_t>(row / kMxfp8RowTile) * kMxfp8RowTile * scale_cols +
         (col_block / 4) * 512 + (row % 32) * 16 + ((row % kMxfp8RowTile) / 32) * 4 +
         (col_block % 4);
}

// Zero the scales of the padding rows [rows, round_up(rows, 128)), spread over num_threads
// threads. In the 128x4 layout the 16 bytes at (group * 512 + i * 16) of a row tile hold rows i,
// i + 32, i + 64 and i + 96 of a 4-column group; a chunk whose four rows are all padding is
// cleared with one 16-byte store, and consecutive threads clear consecutive chunks. Needs a
// 16-byte aligned scale buffer.
CUTLASS_DEVICE void mxfp8_zero_padding_scales(uint8_t *scale, int rows, int scale_cols,
                                              int thread_idx, int num_threads) {
  int const tail = rows % kMxfp8RowTile;
  if (tail == 0) {
    return;
  }
  uint8_t *tile = scale + static_cast<int64_t>(rows / kMxfp8RowTile) * kMxfp8RowTile * scale_cols;
  int const chunks = scale_cols / 4 * 32;
  for (int c = thread_idx; c < chunks; c += num_threads) {
    int const i = c % 32;
    uint8_t *chunk = tile + (c / 32) * 512 + i * 16;
    if (i >= tail) {
      *reinterpret_cast<uint4 *>(chunk) = make_uint4(0u, 0u, 0u, 0u);
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < 4; ++j) {
        if (i + 32 * j >= tail) {
          *reinterpret_cast<uint32_t *>(chunk + 4 * j) = 0u;
        }
      }
    }
  }
}

} // namespace fmha_sm100::decode_q8kv4::sm100::common
