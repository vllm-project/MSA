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
                        Optional<TensorView> maybe_workspace_buffer,
                        Optional<TensorView> maybe_block_table,
                        int64_t topk,
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

  const int32_t* block_table_ptr = nullptr;
  uint64_t block_table_stride_t = 0;
  uint64_t block_table_stride_h = 0;
  uint64_t block_table_stride_k = 0;
  if (maybe_block_table.has_value()) {
    TensorView block_table = maybe_block_table.value();
    CHECK_CUDA(block_table);
    CHECK_DIM(3, block_table);
    TVM_FFI_ICHECK(encode_dlpack_dtype(block_table.dtype()) == int32_code)
        << "block_table must be int32";
    TVM_FFI_ICHECK(block_table.size(0) == total_qo_len &&
                   block_table.size(1) == num_qo_heads &&
                   block_table.size(2) == max_k_tiles)
        << "block_table must have shape [total_qo_len=" << total_qo_len
        << ", num_qo_heads=" << num_qo_heads << ", max_k_tiles=" << max_k_tiles
        << "], got [" << block_table.size(0) << ", " << block_table.size(1)
        << ", " << block_table.size(2) << "]";
    TVM_FFI_ICHECK(block_table.stride(0) >= 0 && block_table.stride(1) >= 0 &&
                   block_table.stride(2) >= 0)
        << "block_table must have non-negative strides";
    CHECK_DEVICE(block_table, max_score);
    block_table_ptr = TensorDataPtrConst<int32_t>(block_table);
    block_table_stride_t = static_cast<uint64_t>(block_table.stride(0));
    block_table_stride_h = static_cast<uint64_t>(block_table.stride(1));
    block_table_stride_k = static_cast<uint64_t>(block_table.stride(2));
  }

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
      block_table_ptr,
      static_cast<uint64_t>(output_indices.stride(0)),
      static_cast<uint64_t>(output_indices.stride(1)),
      static_cast<uint64_t>(output_indices.stride(2)),
      block_table_stride_t,
      block_table_stride_h,
      block_table_stride_k,
      static_cast<uint32_t>(total_qo_len), static_cast<uint32_t>(num_qo_heads),
      static_cast<uint32_t>(max_k_tiles), static_cast<uint32_t>(num_valid_pages),
      num_valid_pages_per_token, layout,
      static_cast<uint32_t>(force_begin_blocks), static_cast<uint32_t>(force_end_blocks),
      stream, /*enable_pdl=*/true);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sparse_topk_select failed: " << cudaGetErrorString(status);
}

void sparse_topk_select_publish(TensorView scores, TensorView num_valid_pages,
                                TensorView recv_bases, TensorView errors,
                                int64_t slot, int64_t rank, int64_t world,
                                int64_t heads_per_rank, int64_t capacity,
                                int64_t force_begin, int64_t force_end,
                                int64_t spin_limit, bool enable_pdl,
                                Optional<TensorView> maybe_generations,
                                int64_t index_scale, int64_t index_offset,
                                int64_t stream_ptr) {
  CHECK_INPUT(scores);
  CHECK_DIM(3, scores);
  CHECK_INPUT(num_valid_pages);
  CHECK_DIM(1, num_valid_pages);
  CHECK_INPUT(recv_bases);
  CHECK_DIM(1, recv_bases);
  CHECK_INPUT(errors);
  CHECK_DIM(1, errors);
  CHECK_DEVICE(scores, num_valid_pages);
  CHECK_DEVICE(scores, recv_bases);
  CHECK_DEVICE(scores, errors);
  TVM_FFI_ICHECK(encode_dlpack_dtype(scores.dtype()) == float32_code);
  TVM_FFI_ICHECK(encode_dlpack_dtype(num_valid_pages.dtype()) == int32_code);
  TVM_FFI_ICHECK(encode_dlpack_dtype(recv_bases.dtype()) == int64_code);
  TVM_FFI_ICHECK(encode_dlpack_dtype(errors.dtype()) == int32_code);
  TVM_FFI_ICHECK(world == 2 || world == 4);
  TVM_FFI_ICHECK(heads_per_rank > 0 && scores.size(1) == world * heads_per_rank);
  TVM_FFI_ICHECK(scores.size(0) == num_valid_pages.size(0));
  TVM_FFI_ICHECK(scores.size(0) <= capacity && capacity > 0);
  TVM_FFI_ICHECK(scores.size(2) > 0 && scores.size(2) < 12288);
  TVM_FFI_ICHECK(recv_bases.size(0) == world && errors.size(0) >= 1);
  TVM_FFI_ICHECK(slot >= 0 && slot < 3 && rank >= 0 && rank < world);
  TVM_FFI_ICHECK(force_begin >= 0 && force_end >= 0 && force_begin + force_end <= 16);
  TVM_FFI_ICHECK(spin_limit > 0 && spin_limit <= INT32_MAX);
  TVM_FFI_ICHECK(index_scale >= 1 && index_scale <= 2 && index_offset >= 0 &&
                 index_offset < index_scale);
  sparse_topk::SparseTopKPublish publish{
      TensorDataPtrConst<int64_t>(recv_bases), TensorDataPtr<int32_t>(errors),
      static_cast<uint32_t>(slot), static_cast<uint32_t>(rank),
      static_cast<uint32_t>(world), static_cast<uint32_t>(heads_per_rank),
      static_cast<uint32_t>(capacity), static_cast<uint32_t>(spin_limit)};
  if (maybe_generations.has_value()) {
    TensorView generations = maybe_generations.value();
    CHECK_INPUT(generations);
    CHECK_DIM(3, generations);
    CHECK_DEVICE(scores, generations);
    TVM_FFI_ICHECK(encode_dlpack_dtype(generations.dtype()) == int32_code);
    TVM_FFI_ICHECK(generations.size(0) == 3 &&
                  generations.size(1) == world * heads_per_rank &&
                  generations.size(2) == capacity);
    publish.generations =
        reinterpret_cast<uint32_t*>(TensorDataPtr<int32_t>(generations));
  }
  publish.index_scale = static_cast<uint32_t>(index_scale);
  publish.index_offset = static_cast<uint32_t>(index_offset);
  const cudaError_t status = sparse_topk::LaunchIndexerTopK<true>(
      TensorDataPtrConst<float>(scores), nullptr, nullptr,
      0, 0, 0, 0, 0, 0, scores.size(0), scores.size(1), scores.size(2), 16,
      scores.size(2) * index_scale, TensorDataPtrConst<int32_t>(num_valid_pages),
      force_begin, force_end, /*input_qo_outermost=*/true,
      reinterpret_cast<cudaStream_t>(stream_ptr), enable_pdl, publish);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sparse_topk_select_publish failed: " << cudaGetErrorString(status);
}

void sparse_topk_select_prefill(TensorView scores, TensorView num_valid_pages,
                               TensorView carrier, int64_t force_begin,
                               int64_t force_end, int64_t index_scale,
                               int64_t index_offset, int64_t stream_ptr) {
  CHECK_INPUT(scores);
  CHECK_INPUT(num_valid_pages);
  CHECK_CUDA(carrier);
  CHECK_DIM(3, scores);
  CHECK_DIM(1, num_valid_pages);
  CHECK_DIM(3, carrier);
  CHECK_DEVICE(scores, num_valid_pages);
  CHECK_DEVICE(scores, carrier);
  TVM_FFI_ICHECK(encode_dlpack_dtype(scores.dtype()) == float32_code);
  TVM_FFI_ICHECK(encode_dlpack_dtype(num_valid_pages.dtype()) == int32_code);
  TVM_FFI_ICHECK(encode_dlpack_dtype(carrier.dtype()) == int32_code);
  TVM_FFI_ICHECK(scores.size(1) > 0 && scores.size(2) > 0 && scores.size(2) < 12288);
  TVM_FFI_ICHECK(num_valid_pages.size(0) == scores.size(0));
  TVM_FFI_ICHECK(carrier.size(0) == scores.size(1) &&
                carrier.size(1) == scores.size(0) && carrier.size(2) == 33);
  TVM_FFI_ICHECK(carrier.stride(2) == 1 && carrier.stride(1) >= 33 &&
                carrier.stride(0) >= carrier.size(1) * carrier.stride(1));
  TVM_FFI_ICHECK(force_begin >= 0 && force_end >= 0 && force_begin + force_end <= 16);
  TVM_FFI_ICHECK(index_scale >= 1 && index_scale <= 2 && index_offset >= 0 &&
                 index_offset < index_scale);
  sparse_topk::SparseTopKPrefill publish{
      TensorDataPtr<int32_t>(carrier), static_cast<uint64_t>(carrier.stride(0)),
      static_cast<uint64_t>(carrier.stride(1))};
  publish.index_scale = static_cast<uint32_t>(index_scale);
  publish.index_offset = static_cast<uint32_t>(index_offset);
  const cudaError_t status = sparse_topk::LaunchIndexerTopK<true>(
      TensorDataPtrConst<float>(scores), nullptr, nullptr,
      0, 0, 0, 0, 0, 0, scores.size(0), scores.size(1), scores.size(2), 16,
      scores.size(2) * index_scale, TensorDataPtrConst<int32_t>(num_valid_pages),
      force_begin, force_end, /*input_qo_outermost=*/true,
      reinterpret_cast<cudaStream_t>(stream_ptr), /*enable_pdl=*/false, publish);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sparse_topk_select_prefill failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_topk_select_prefill, sparse_topk_select_prefill);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_topk_select_publish, sparse_topk_select_publish);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_topk_select_init, sparse_topk_select_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_topk_select, sparse_topk_select);
