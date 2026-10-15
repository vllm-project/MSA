// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "cute/arch/util.hpp"

// SM107 has no QMUL4 (mul.e4m3x4.e2m1x4); K is dequantized to FP16 and fed to HMMA.

namespace q8kv4_indexer_sm107::detail {

// FP4 x block scale in f16, with the block scale already an f16x2 pair.
CUTE_DEVICE void nvfp4x8_to_f16x2x4_scaled(uint32_t &output0, uint32_t &output1, uint32_t &output2,
                                           uint32_t &output3, uint32_t packed_fp4x8,
                                           uint32_t scale_f16x2) {
  // E2M1 x E4M3 is exact in f16 (<= 6 significant bits, 2^-10..2688): no E4M3 rounding.
  asm volatile("{\n"
               ".reg .b8 e2m1_0, e2m1_1, e2m1_2, e2m1_3;\n"
               ".reg .b32 f16_0, f16_1, f16_2, f16_3;\n"
               "mov.b32 {e2m1_0, e2m1_1, e2m1_2, e2m1_3}, %4;\n"
               "cvt.rn.f16x2.e2m1x2 f16_0, e2m1_0;\n"
               "cvt.rn.f16x2.e2m1x2 f16_1, e2m1_1;\n"
               "cvt.rn.f16x2.e2m1x2 f16_2, e2m1_2;\n"
               "cvt.rn.f16x2.e2m1x2 f16_3, e2m1_3;\n"
               "mul.rn.f16x2 %0, f16_0, %5;\n"
               "mul.rn.f16x2 %1, f16_1, %5;\n"
               "mul.rn.f16x2 %2, f16_2, %5;\n"
               "mul.rn.f16x2 %3, f16_3, %5;\n"
               "}\n"
               : "=&r"(output0), "=&r"(output1), "=&r"(output2), "=&r"(output3)
               : "r"(packed_fp4x8), "r"(scale_f16x2));
}

// {v.lo, v.lo} or {v.hi, v.hi}; ptxas folds it into the consumer's operand swizzle.
template <int Half> CUTE_DEVICE uint32_t broadcast_f16x2(uint32_t value) {
  uint32_t output;
  if constexpr (Half == 0) {
    asm("{ .reg .b16 l, h; mov.b32 {l, h}, %1; mov.b32 %0, {l, l}; }" : "=r"(output) : "r"(value));
  } else {
    asm("{ .reg .b16 l, h; mov.b32 {l, h}, %1; mov.b32 %0, {h, h}; }" : "=r"(output) : "r"(value));
  }
  return output;
}

// The 4 e2m1->f16x2 converts of nvfp4x8_to_f16x2x4_scaled, without the scale multiply.
CUTE_DEVICE void nvfp4x8_to_f16x2x4_unscaled(uint32_t &output0, uint32_t &output1,
                                             uint32_t &output2, uint32_t &output3,
                                             uint32_t packed_fp4x8) {
  asm volatile("{\n"
               ".reg .b8 e2m1_0, e2m1_1, e2m1_2, e2m1_3;\n"
               "mov.b32 {e2m1_0, e2m1_1, e2m1_2, e2m1_3}, %4;\n"
               "cvt.rn.f16x2.e2m1x2 %0, e2m1_0;\n"
               "cvt.rn.f16x2.e2m1x2 %1, e2m1_1;\n"
               "cvt.rn.f16x2.e2m1x2 %2, e2m1_2;\n"
               "cvt.rn.f16x2.e2m1x2 %3, e2m1_3;\n"
               "}\n"
               : "=&r"(output0), "=&r"(output1), "=&r"(output2), "=&r"(output3)
               : "r"(packed_fp4x8));
}

// {acc0, acc1} += scale * {p0, p1}: one sm_107 FHFMA2, scale already an f16x2 pair.
CUTE_DEVICE void fma2_f32x2_by_f16x2(float &acc0, float &acc1, uint32_t scale_f16x2, float p0,
                                     float p1) {
  asm volatile("{\n"
               ".reg .b64 acc2, p2;\n"
               "mov.b64 acc2, {%0, %1};\n"
               "mov.b64 p2, {%2, %3};\n"
               "fma.rn.f32x2.f16x2.f32x2.f32x2 acc2, %4, p2, acc2;\n"
               "mov.b64 {%0, %1}, acc2;\n"
               "}\n"
               : "+f"(acc0), "+f"(acc1)
               : "f"(p0), "f"(p1), "r"(scale_f16x2));
}

} // namespace q8kv4_indexer_sm107::detail
