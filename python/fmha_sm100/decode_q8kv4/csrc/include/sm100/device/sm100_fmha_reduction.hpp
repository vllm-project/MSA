// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "mxfp8_output.cuh"

#include <cfloat>
#include <cuda_runtime_api.h>
#include <type_traits>

namespace cutlass::fmha::kernel {

template <class T>
CUTLASS_DEVICE T warp_uniform(T a) {
  return __shfl_sync(0xffffffff, a, 0);
}

// Reduction kernel for split-KV FMHA.
// Merges partial bf16 O using float32 LSE (log-sum-exp) from multiple KV splits.
// ptr_lse stores LSE = max + log2(sum) / scale_softmax_log2.
//
// Grid:  (num_qo_heads, total_qo_len, 1)
// Block: 128 threads. Each thread handles one d element.
template <class ElementPartial,  // bf16 (workspace O type)
          class ElementOut,      // output dtype (fp8 or bf16)
          int kMaxSplits = 128>
struct Sm100FmhaReductionKernel {
  static const int MaxThreadsPerBlock = 128;

  struct Params {
    const ElementPartial* ptr_O_partial;  // [total_qo_len * num_splits, num_heads, head_dim]
    ElementOut* ptr_O;                    // [total_qo_len, num_heads, head_dim]
    const float* ptr_lse;                // [num_splits, total_qo_len, num_heads]
    const int* num_kv_splits_per_row;     // [total_qo_len]: actual split count per row
    float scale_softmax_log2;
    float inv_scale_o;
    int num_kv_splits;
    int total_qo_len;
    int num_qo_heads;
    int head_dim_vo;
    int stride_o_n;       // output: token stride
    int stride_o_h;       // output: head stride
    int stride_partial_n; // partial: token stride
    int stride_partial_h; // partial: head stride
    // Fused unpack-GQA: when ptr_O_direct != nullptr, write to unpacked layout
    // [nnz_qo, num_qo_heads_orig, head_dim_vo] instead of packed ptr_O.
    ElementOut* ptr_O_direct = nullptr;
    int num_qo_heads_orig = 0;
    int num_kv_heads = 0;
    int pack_factor = 1;
    const float *k_global_scale_ptr = nullptr;
    const float *v_global_scale_ptr = nullptr;
    // Optional MXFP8 output (see the attention kernel's o_mxfp8_ptr); BF16 ElementOut only.
    uint8_t *ptr_O_mxfp8 = nullptr;
    uint8_t *ptr_O_mxfp8_scale = nullptr;
    int mxfp8_rows = 0;
  };

  // One warp is one 32-value MXFP8 block of a (token, head) row; lane 0 writes its scale.
  CUTLASS_DEVICE static void store_mxfp8(Params const &params, ElementOut value, int token,
                                         int col, int row_values) {
    namespace mx = fmha_sm100::decode_q8kv4::sm100::common;
    static_assert(std::is_same_v<ElementOut, cutlass::bfloat16_t>, "MXFP8 output needs BF16 O");
    uint32_t scale;
    uint8_t const q = mx::mxfp8_quantize_lane(value.raw(), scale);
    params.ptr_O_mxfp8[static_cast<int64_t>(token) * row_values + col] = q;
    if ((threadIdx.x & (cutlass::NumThreadsPerWarp - 1)) == 0) {
      params.ptr_O_mxfp8_scale[mx::mxfp8_scale_offset(token, col / mx::kMxfp8Block,
                                                      row_values / mx::kMxfp8Block)] =
          static_cast<uint8_t>(scale);
    }
  }

  static dim3 get_grid_shape(Params const& params) {
    return dim3(params.num_qo_heads, params.total_qo_len, 1);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE void operator()(Params const& params, char* /* smem */) {
    int head_idx = blockIdx.x;
    int abs_row = blockIdx.y;
    int d = threadIdx.x;
    // Same folds as the attention kernel: K global scale into the LSE domain, V global scale and
    // the staging gain into the output.
    float const scale_softmax_log2 = params.scale_softmax_log2 * *params.k_global_scale_ptr;
    float const inv_scale_o = params.inv_scale_o * *params.v_global_scale_ptr;

    if (params.ptr_O_mxfp8 != nullptr && blockIdx.x == 0 && blockIdx.y == 0) {
      int const heads =
          params.pack_factor > 1 ? params.num_qo_heads_orig : params.num_qo_heads;
      fmha_sm100::decode_q8kv4::sm100::common::mxfp8_zero_padding_scales(
          params.ptr_O_mxfp8_scale, params.mxfp8_rows,
          heads * params.head_dim_vo / fmha_sm100::decode_q8kv4::sm100::common::kMxfp8Block,
          threadIdx.x, MaxThreadsPerBlock);
    }
    if (abs_row >= params.total_qo_len || d >= params.head_dim_vo) return;

    int num_splits = warp_uniform(params.num_kv_splits_per_row[abs_row]);
    int partial_off = abs_row * params.stride_partial_n +
                      head_idx * params.stride_partial_h + d;

    // Decide output base + offset. Fused unpack remaps packed (abs_row, head_idx)
    // to unpacked (actual_tok, unpacked_head); plain path uses packed strides.
    ElementOut* o_ptr_used;
    int64_t dst_off;
    int out_token;
    int out_head;
    int out_heads;
    if (params.pack_factor > 1) {
      int pf            = params.pack_factor;
      int rem_hr        = params.num_qo_heads / params.num_kv_heads;  // packed h_r
      int kv_head       = head_idx / rem_hr;
      int rhr_idx       = head_idx % rem_hr;
      int h_r_orig      = rem_hr * pf;
      int pf_idx        = abs_row % pf;
      int actual_tok    = abs_row / pf;
      int unpacked_head = kv_head * h_r_orig + rhr_idx * pf + pf_idx;
      dst_off = (int64_t)actual_tok * params.num_qo_heads_orig * params.head_dim_vo
              + (int64_t)unpacked_head * params.head_dim_vo
              + d;
      o_ptr_used = params.ptr_O_direct;
      out_token = actual_tok;
      out_head = unpacked_head;
      out_heads = params.num_qo_heads_orig;
    } else {
      dst_off = (int64_t)abs_row * params.stride_o_n
              + (int64_t)head_idx * params.stride_o_h
              + d;
      o_ptr_used = params.ptr_O;
      out_token = abs_row;
      out_head = head_idx;
      out_heads = params.num_qo_heads;
    }
    // Uniform across the block: every exit below stores, so the MXFP8 warps stay converged.
    auto store = [&](ElementOut value) {
      if (o_ptr_used != nullptr) {
        o_ptr_used[dst_off] = value;
      }
      if (params.ptr_O_mxfp8 != nullptr) {
        store_mxfp8(params, value, out_token, out_head * params.head_dim_vo + d,
                    out_heads * params.head_dim_vo);
      }
    };

    // Fast path: single split — just scale and copy, no merge needed
    if (num_splits == 1) {
      int stats_base = warp_uniform(abs_row * params.num_qo_heads + head_idx);
      float lse = params.ptr_lse[stats_base];
      if (lse == -INFINITY) {
        store(ElementOut(0.f));
      } else {
        float o_s = static_cast<float>(params.ptr_O_partial[partial_off]);
        store(ElementOut(o_s * inv_scale_o));
      }
      return;
    }

    int stats_stride = warp_uniform(params.total_qo_len * params.num_qo_heads);
    int stats_base = warp_uniform(abs_row * params.num_qo_heads + head_idx);

    float running_lse = -__FLT_MAX__;
    float running_w = 0.f;
    float running_o = 0.f;

    int partial_stride = params.total_qo_len * params.stride_partial_n;
    int stats_off = stats_base;

    // Prologue: load first split
    float lse_cur = -INFINITY, o_s_cur = 0.f;
    if (num_splits > 0) {
      lse_cur = params.ptr_lse[stats_off];
      o_s_cur = static_cast<float>(params.ptr_O_partial[partial_off]);
    }

    // Main loop: prefetch next, merge current using LSE-based weights
    // Empty splits have LSE=-INF, exp2(scale*(-INF - ref)) = 0 → natural no-op
    #pragma unroll 2
    for (int s = 0; s < num_splits - 1; ++s) {
      float lse_s = lse_cur, o_s = o_s_cur;
      stats_off += stats_stride;
      partial_off += partial_stride;
      lse_cur = params.ptr_lse[stats_off];
      o_s_cur = static_cast<float>(params.ptr_O_partial[partial_off]);

      o_s = (lse_s != -INFINITY) ? o_s : 0.f;

      if (lse_s > running_lse) {
        float rescale = exp2f(scale_softmax_log2 * (running_lse - lse_s));
        running_o = fmaf(running_o, rescale, o_s);
        running_w = fmaf(running_w, rescale, 1.f);
        running_lse = lse_s;
      } else {
        float rescale = exp2f(scale_softmax_log2 * (lse_s - running_lse));
        running_o = fmaf(o_s, rescale, running_o);
        running_w += rescale;
      }
    }

    // Epilogue: last split
    if (num_splits > 0) {
      float o_last = (lse_cur != -INFINITY) ? o_s_cur : 0.f;
      if (lse_cur > running_lse) {
        float rescale = exp2f(scale_softmax_log2 * (running_lse - lse_cur));
        running_o = fmaf(running_o, rescale, o_last);
        running_w = fmaf(running_w, rescale, 1.f);
        running_lse = lse_cur;
      } else {
        float rescale = exp2f(scale_softmax_log2 * (lse_cur - running_lse));
        running_o = fmaf(o_last, rescale, running_o);
        running_w += rescale;
      }
    }

    if (running_w == 0.f) {
      store(ElementOut(0.f));
      return;
    }

    float inv_w = warp_uniform(1.f / running_w);
    store(ElementOut(running_o * inv_w * inv_scale_o));
  }
};

}  // namespace cutlass::fmha::kernel

namespace fmha_sm100::decode_q8kv4::sm100::device {

template <typename ElementPartial, typename ElementOut>
__global__ void fmha_reduction_kernel(
    typename cutlass::fmha::kernel::Sm100FmhaReductionKernel<ElementPartial, ElementOut>::Params params) {
  cutlass::fmha::kernel::Sm100FmhaReductionKernel<ElementPartial, ElementOut> kernel;
  kernel(params, nullptr);
}

template <typename ElementPartial, typename ElementOut>
cudaError_t
launch_fmha_reduction(const ElementPartial *ptr_O_partial, ElementOut *ptr_O, const float *ptr_lse,
                      const int *num_kv_splits_per_row, float scale_softmax_log2, float inv_scale_o,
                      const float *k_global_scale, const float *v_global_scale, int num_kv_splits,
                      int total_qo_len, int num_qo_heads, int head_dim_vo, int stride_o_n,
                      int stride_o_h, int stride_partial_n, int stride_partial_h,
                      ElementOut *ptr_O_direct, int num_qo_heads_orig, int num_kv_heads,
                      int pack_factor, uint8_t *ptr_O_mxfp8, uint8_t *ptr_O_mxfp8_scale,
                      int mxfp8_rows, cudaStream_t stream) {
  if (total_qo_len <= 0) return cudaSuccess;
  using ReductionKernel =
      cutlass::fmha::kernel::Sm100FmhaReductionKernel<ElementPartial, ElementOut>;
  typename ReductionKernel::Params params{
      ptr_O_partial,     ptr_O,         ptr_lse,          num_kv_splits_per_row, scale_softmax_log2,
      inv_scale_o,       num_kv_splits, total_qo_len,     num_qo_heads,          head_dim_vo,
      stride_o_n,        stride_o_h,    stride_partial_n, stride_partial_h,      ptr_O_direct,
      num_qo_heads_orig, num_kv_heads,  pack_factor,      k_global_scale,        v_global_scale};
  params.ptr_O_mxfp8 = ptr_O_mxfp8;
  params.ptr_O_mxfp8_scale = ptr_O_mxfp8_scale;
  params.mxfp8_rows = mxfp8_rows;
  dim3 grid = ReductionKernel::get_grid_shape(params);
  dim3 block = ReductionKernel::get_block_shape();
  fmha_reduction_kernel<ElementPartial, ElementOut><<<grid, block, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace fmha_sm100::decode_q8kv4::sm100::device
