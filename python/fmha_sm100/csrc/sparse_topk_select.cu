// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#include "sparse_topk_select.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer;
using tvm::ffi::Optional;

template <typename T>
T* TensorDataPtr(TensorView tensor) {
  return reinterpret_cast<T*>(static_cast<char*>(tensor.data_ptr()) + tensor.byte_offset());
}

template <typename T>
const T* TensorDataPtrConst(TensorView tensor) {
  return reinterpret_cast<const T*>(static_cast<const char*>(tensor.data_ptr()) +
                                    tensor.byte_offset());
}

void sparse_topk_select_init() {
  cudaError_t status = sparse_topk::ConfigureSparseTopKSelect();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sparse_topk_select init failed: " << cudaGetErrorString(status);
}

// v2.5_oob_clamp_in_kernel:
//   Adds num_valid_pages parameter (between topk and stream_ptr).  When
//   num_valid_pages < max_k_tiles, indices >= num_valid_pages are emitted as
//   -1 and sorted to the tail by the kernel — replacing the prod wrapper's
//   torch.where + sort + torch.where post-processing (~84-101 us measured).
//
//   To disable clamping, pass num_valid_pages = max_k_tiles (or any value
//   >= max_k_tiles).
void sparse_topk_select(TensorView max_score, TensorView output_indices,
                        Optional<TensorView> maybe_workspace_buffer, int64_t topk,
                        int64_t num_valid_pages,
                        Optional<TensorView> maybe_num_valid_pages_per_token,
                        int64_t force_begin_blocks, int64_t force_end_blocks,
                        int64_t input_layout,
                        int64_t stream_ptr) {
  CHECK_INPUT(max_score);
  CHECK_CUDA(output_indices);
  CHECK_DIM(3, max_score);
  CHECK_DIM(3, output_indices);

  TVM_FFI_ICHECK(encode_dlpack_dtype(max_score.dtype()) == float32_code)
      << "max_score must be float32";
  TVM_FFI_ICHECK(encode_dlpack_dtype(output_indices.dtype()) == int32_code)
      << "output_indices must be int32";

  TVM_FFI_ICHECK(input_layout == 0 || input_layout == 1)
      << "input_layout must be 0 (HKT) or 1 (THK), got " << input_layout;
  const auto layout = input_layout == 0
      ? sparse_topk::SparseTopKInputLayout::kHKT
      : sparse_topk::SparseTopKInputLayout::kTHK;

  const int64_t total_qo_len = input_layout == 0 ? max_score.size(2) : max_score.size(0);
  const int64_t num_qo_heads = input_layout == 0 ? max_score.size(0) : max_score.size(1);
  const int64_t max_k_tiles = input_layout == 0 ? max_score.size(1) : max_score.size(2);

  TVM_FFI_ICHECK(output_indices.size(0) == total_qo_len);
  TVM_FFI_ICHECK(output_indices.size(1) == num_qo_heads);
  TVM_FFI_ICHECK(output_indices.size(2) == topk);
  TVM_FFI_ICHECK(output_indices.stride(0) >= 0 && output_indices.stride(1) >= 0 &&
                 output_indices.stride(2) >= 0)
      << "output_indices must have non-negative strides";
  TVM_FFI_ICHECK(topk == 16) << "this kernel only supports topk == 16, got " << topk;
  TVM_FFI_ICHECK(num_valid_pages > 0)
      << "num_valid_pages must be > 0, got " << num_valid_pages;

  const int32_t* num_valid_pages_per_token = nullptr;
  if (maybe_num_valid_pages_per_token.has_value()) {
    TensorView nvp = maybe_num_valid_pages_per_token.value();
    CHECK_INPUT(nvp);
    CHECK_DIM(1, nvp);
    TVM_FFI_ICHECK(encode_dlpack_dtype(nvp.dtype()) == int32_code)
        << "num_valid_pages_per_token must be int32";
    TVM_FFI_ICHECK(nvp.size(0) == total_qo_len)
        << "num_valid_pages_per_token must have shape [total_qo_len="
        << total_qo_len << "], got " << nvp.size(0);
    num_valid_pages_per_token = TensorDataPtrConst<int32_t>(nvp);
  }

  const size_t needed_workspace = sparse_topk::SparseTopKWorkspaceSize(
      static_cast<uint32_t>(total_qo_len), static_cast<uint32_t>(num_qo_heads),
      static_cast<uint32_t>(max_k_tiles), layout);
  int32_t* workspace_ptr = nullptr;
  if (needed_workspace > 0) {
    TVM_FFI_ICHECK(maybe_workspace_buffer.has_value())
        << "workspace_buffer is required for sparse_topk_select HKT layout";
    TensorView workspace_buffer = maybe_workspace_buffer.value();
    CHECK_INPUT(workspace_buffer);
    CHECK_DIM(1, workspace_buffer);
    TVM_FFI_ICHECK(encode_dlpack_dtype(workspace_buffer.dtype()) == int32_code)
        << "workspace_buffer must be int32";
    TVM_FFI_ICHECK(static_cast<size_t>(workspace_buffer.size(0)) >= needed_workspace)
        << "workspace_buffer too small: need " << needed_workspace << " int32 elements";
    workspace_ptr = TensorDataPtr<int32_t>(workspace_buffer);
  }

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  cudaError_t status = sparse_topk::SparseTopKSelect(
      TensorDataPtrConst<float>(max_score),
      TensorDataPtr<int32_t>(output_indices),
      workspace_ptr,
      static_cast<uint64_t>(output_indices.stride(0)),
      static_cast<uint64_t>(output_indices.stride(1)),
      static_cast<uint64_t>(output_indices.stride(2)),
      static_cast<uint32_t>(total_qo_len), static_cast<uint32_t>(num_qo_heads),
      static_cast<uint32_t>(max_k_tiles), static_cast<uint32_t>(num_valid_pages),
      num_valid_pages_per_token, layout,
      static_cast<uint32_t>(force_begin_blocks), static_cast<uint32_t>(force_end_blocks),
      stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sparse_topk_select failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_topk_select_init, sparse_topk_select_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_topk_select, sparse_topk_select);
