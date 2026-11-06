// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

// MXFP8 form of the BF16 attention output: E4M3 data plus one UE8M0 scale per 32 values of a
// row, scales in FlashInfer's 128x4 swizzled layout with the padding rows zeroed. Bit for bit
// what mxfp8_quantize(out.view(tokens, -1), is_sf_swizzled_layout=True) gives for the BF16 output:
// the amax of the 32 BF16 values, the UE8M0 of amax / 448 rounded up, and x * 2^(127 - e) clamped
// to 448 and converted with satfinite.
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

// UE8M0 of amax / 448 rounded up, from the sign-cleared BF16 bits in both halves of a pair. As
// 448 = 1.75 * 2^8, it is the BF16 exponent less 8, plus one when the mantissa is above 0.75.
// Equals FlashInfer's fp32 path (amax * RN(1/448), rounded up to a power of two, clamped to 254,
// 0 for zero) for every BF16 magnitude; the exponent 255 of Inf and NaN maps to 254.
CUTLASS_DEVICE uint32_t bf16_amax_to_ue8m0(uint32_t pair) {
  uint32_t const amax = bf16x2_max(pair, (pair >> 16) | (pair << 16)) & 0x7FFFu;
  uint32_t const exponent = amax >> 7;
  int const scale = static_cast<int>(exponent) - 8 + ((amax & 0x7Fu) > 96u ? 1 : 0);
  return exponent == 255u ? 254u : static_cast<uint32_t>(scale > 0 ? scale : 0);
}

// 2^(127 - e) as BF16 in both halves, or 0 for e == 0 and for e == 254, whose exponent is below the
// normal range: FlashInfer's ue8m0_to_inv_scale_fast.
CUTLASS_DEVICE uint32_t ue8m0_to_inv_scale_bf16x2(uint32_t e) {
  return e == 0u ? 0u : (254u - e) * 0x00800080u;
}

// Four BF16 (w0 = values 0-1, w1 = 2-3) times the power-of-two inverse scale -> four E4M3 bytes,
// value 0 lowest. The product is exact in BF16 wherever E4M3 can represent it; min 448 maps NaN
// and overflow to 448 like an fp32 clamp, and satfinite saturates the negative side.
CUTLASS_DEVICE uint32_t bf16x4_to_e4m3x4(uint32_t w0, uint32_t w1, uint32_t inv2) {
  uint32_t r;
  asm("{\n"
      ".reg .b32 p0, p1;\n"
      ".reg .b16 q0, q1;\n"
      "mul.rn.bf16x2 p0, %1, %3;\n"
      "mul.rn.bf16x2 p1, %2, %3;\n"
      "min.bf16x2 p0, p0, %4;\n"
      "min.bf16x2 p1, p1, %4;\n"
      "cvt.rn.satfinite.e4m3x2.bf16x2 q0, p0;\n"
      "cvt.rn.satfinite.e4m3x2.bf16x2 q1, p1;\n"
      "mov.b32 %0, {q0, q1};\n"
      "}\n"
      : "=r"(r)
      : "r"(w0), "r"(w1), "r"(inv2), "r"(0x43E043E0u));
  return r;
}

// UE8M0 scale of a 32-value block whose BF16 values are spread over kLanes adjacent lanes, each
// holding kWords packed BF16 pairs (kLanes * kWords * 2 == 32). Every lane of the warp must call
// it; the lanes of a block get the same scale.
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
  CUTLASS_PRAGMA_UNROLL
  for (int offset = 1; offset < kLanes; offset *= 2) {
    local = bf16x2_max(local, __shfl_xor_sync(0xFFFFFFFFu, local, offset));
  }
  return bf16_amax_to_ue8m0(local);
}

// One BF16 value per lane, 32 lanes per block (the split-KV reduction): returns the E4M3 byte;
// `scale` receives the block's UE8M0 scale.
CUTLASS_DEVICE uint8_t mxfp8_quantize_lane(uint16_t value, uint32_t &scale) {
  uint32_t const pair = static_cast<uint32_t>(value) * 0x00010001u;
  uint32_t amax = pair & 0x7FFF7FFFu;
  CUTLASS_PRAGMA_UNROLL
  for (int offset = 1; offset < kMxfp8Block; offset *= 2) {
    amax = bf16x2_max(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));
  }
  scale = bf16_amax_to_ue8m0(amax);
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

// Byte offset of scale (row, col_block) in the 128x4 swizzled layout with scale_cols columns
// (a multiple of 4).
CUTLASS_DEVICE int64_t mxfp8_scale_offset(unsigned row, unsigned col_block, unsigned scale_cols) {
  return static_cast<int64_t>(row >> 7) * kMxfp8RowTile * scale_cols + (col_block >> 2) * 512 +
         (row & 31u) * 16 + ((row & 127u) >> 5) * 4 + (col_block & 3u);
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
