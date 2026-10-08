// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#include "sm100_fmha_reduction.hpp"

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

namespace fmha_sm100::decode_q8kv4::sm100::device {

static cudaError_t fmha_fwd_reduction_bf16_impl(
    const void *ptr_O_partial, void *ptr_O, const float *ptr_lse, const int *num_kv_splits_per_row,
    float scale_softmax_log2, float inv_scale_o, const float *k_global_scale,
    const float *v_global_scale, int num_kv_splits, int total_qo_len, int num_qo_heads,
    int head_dim_vo, int stride_o_n, int stride_o_h, int stride_partial_n, int stride_partial_h,
    void *ptr_O_direct, int num_qo_heads_orig, int num_kv_heads, int pack_factor,
    uint8_t *ptr_O_mxfp8, uint8_t *ptr_O_mxfp8_scale, int mxfp8_rows, cudaStream_t stream) {
  return launch_fmha_reduction<cutlass::bfloat16_t, cutlass::bfloat16_t>(
      static_cast<const cutlass::bfloat16_t *>(ptr_O_partial),
      static_cast<cutlass::bfloat16_t *>(ptr_O), ptr_lse, num_kv_splits_per_row, scale_softmax_log2,
      inv_scale_o, k_global_scale, v_global_scale, num_kv_splits, total_qo_len, num_qo_heads,
      head_dim_vo, stride_o_n, stride_o_h, stride_partial_n, stride_partial_h,
      static_cast<cutlass::bfloat16_t *>(ptr_O_direct), num_qo_heads_orig, num_kv_heads,
      pack_factor, ptr_O_mxfp8, ptr_O_mxfp8_scale, mxfp8_rows, stream);
}

void fmha_fwd_reduction_forward(TensorView o_partial, Optional<TensorView> o, TensorView lse,
                                TensorView num_kv_splits_per_row_tensor, double scale_softmax_log2,
                                double inv_scale_o, TensorView k_global_scale,
                                TensorView v_global_scale, int64_t num_kv_splits,
                                int64_t total_qo_len, int64_t num_qo_heads, int64_t head_dim_vo,
                                int64_t stride_o_n, int64_t stride_o_h, int64_t stride_partial_n,
                                int64_t stride_partial_h, int64_t num_qo_heads_orig,
                                int64_t num_kv_heads, int64_t pack_factor,
                                Optional<TensorView> o_mxfp8, Optional<TensorView> o_mxfp8_scale,
                                int64_t stream_ptr) {
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  // When direct unpack is enabled, route o.data_ptr() to ptr_O_direct and leave
  // packed ptr_O nullptr. The kernel branches on ptr_O_direct.
  void* o_data = o.has_value() ? o.value().data_ptr() : nullptr;
  uint8_t *mxfp8 =
      o_mxfp8.has_value() ? static_cast<uint8_t *>(o_mxfp8.value().data_ptr()) : nullptr;
  uint8_t *mxfp8_scale = o_mxfp8_scale.has_value()
                             ? static_cast<uint8_t *>(o_mxfp8_scale.value().data_ptr())
                             : nullptr;
  int const mxfp8_rows = o_mxfp8.has_value() ? static_cast<int>(o_mxfp8.value().size(0)) : 0;
  void* o_packed_ptr = (pack_factor > 1) ? nullptr : o_data;
  void* o_direct_ptr = (pack_factor > 1) ? o_data : nullptr;

  auto status = fmha_fwd_reduction_bf16_impl(
      o_partial.data_ptr(), o_packed_ptr, static_cast<const float *>(lse.data_ptr()),
      static_cast<const int *>(num_kv_splits_per_row_tensor.data_ptr()),
      static_cast<float>(scale_softmax_log2), static_cast<float>(inv_scale_o),
      static_cast<const float *>(k_global_scale.data_ptr()),
      static_cast<const float *>(v_global_scale.data_ptr()), static_cast<int>(num_kv_splits),
      static_cast<int>(total_qo_len), static_cast<int>(num_qo_heads), static_cast<int>(head_dim_vo),
      static_cast<int>(stride_o_n), static_cast<int>(stride_o_h),
      static_cast<int>(stride_partial_n), static_cast<int>(stride_partial_h), o_direct_ptr,
      static_cast<int>(num_qo_heads_orig), static_cast<int>(num_kv_heads),
      static_cast<int>(pack_factor), mxfp8, mxfp8_scale, mxfp8_rows, stream);
  if (status != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "FMHA forward decode split-KV reduction failed: " << cudaGetErrorString(status);
  }
}

}  // namespace fmha_sm100::decode_q8kv4::sm100::device

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    reduction,
    fmha_sm100::decode_q8kv4::sm100::device::
        fmha_fwd_reduction_forward);
