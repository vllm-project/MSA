// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>

#include "q8kv4_indexer_sm107/kernel.hpp"

namespace q8kv4_indexer_sm107 {

template <class Traits> struct IndexerGemmRunner {
  static constexpr int kInlineCounterOffset = Traits::kPrepareThreads + 1;
  static constexpr int kDynamicCounterOffset = -1;
  using InlineKernel = IndexerGemmKernel<Traits, kInlineCounterOffset>;
  using DynamicKernel = IndexerGemmKernel<Traits, kDynamicCounterOffset>;
  using InlinePostscaleKernel = IndexerGemmKernel<Traits, kInlineCounterOffset, true>;
  using DynamicPostscaleKernel = IndexerGemmKernel<Traits, kDynamicCounterOffset, true>;

  static bool can_plan(IndexerGemmArguments const &arguments) {
    return arguments.page_table_ptr != nullptr && arguments.kv_lengths_ptr != nullptr &&
           arguments.scheduler_workspace_ptr != nullptr && arguments.batch > 0 &&
           arguments.max_pages > 0 && arguments.max_pages <= Traits::kMaximumPages &&
           (arguments.num_valid_pages_ptr == nullptr ||
            (arguments.query_length > 0 && arguments.query_length <= Traits::kQueryLength)) &&
           (arguments.batch <= Traits::kPrepareThreads ||
            (arguments.scheduler_temp_storage_ptr != nullptr &&
             arguments.scheduler_temp_storage_bytes > 0));
  }

  static bool can_run(IndexerGemmArguments const &arguments) {
    return arguments.q_ptr != nullptr && arguments.k_cache_ptr != nullptr &&
           arguments.page_table_ptr != nullptr && arguments.kv_lengths_ptr != nullptr &&
           arguments.scheduler_workspace_ptr != nullptr && arguments.output_ptr != nullptr &&
           arguments.batch > 0 && arguments.query_length > 0 &&
           arguments.query_length <= Traits::kQueryLength && arguments.max_pages > 0 &&
           arguments.max_pages <= Traits::kMaximumPages && arguments.physical_pages > 0 &&
           arguments.page_stride_bytes >= Traits::kPageBytes &&
           arguments.page_stride_bytes % 16 == 0 && arguments.sm_count > 0;
  }

  static cudaError_t scheduler_temp_storage_bytes(int batch, size_t &temp_storage_bytes) {
    temp_storage_bytes = 0;
    if (batch <= Traits::kPrepareThreads) {
      return cudaSuccess;
    }
    return get_indexer_gemm_scheduler_temp_storage_bytes(batch, temp_storage_bytes);
  }

  static cudaError_t encode_tma(CUtensorMap &descriptor, void *pointer,
                                uint64_t const (&dimensions)[3], uint64_t const (&strides)[2],
                                uint32_t const (&box)[3], CUtensorMapSwizzle swizzle) {
    uint32_t element_strides[3] = {1, 1, 1};
    CUresult const result = cuTensorMapEncodeTiled(
        &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, pointer, dimensions, strides, box,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
  }

  static cudaError_t to_plan_params(IndexerGemmArguments const &arguments,
                                    IndexerGemmParams &params) {
    params.page_table_ptr = arguments.page_table_ptr;
    params.kv_lengths_ptr = arguments.kv_lengths_ptr;
    params.scheduler_workspace_ptr =
        arguments.scheduler_workspace_ptr + (arguments.batch > Traits::kPrepareThreads ? 1 : 0);
    params.num_valid_pages_ptr = arguments.num_valid_pages_ptr;
    params.batch = arguments.batch;
    params.query_length = arguments.query_length;
    params.max_pages = arguments.max_pages;

    return cudaSuccess;
  }

  static cudaError_t to_underlying_arguments(IndexerGemmArguments const &arguments,
                                             IndexerGemmParams &params) {
    cudaError_t status = to_plan_params(arguments, params);
    if (status != cudaSuccess) {
      return status;
    }
    params.q_ptr = static_cast<uint8_t const *>(arguments.q_ptr);
    params.query_length = arguments.query_length;
    params.output_ptr = arguments.output_ptr;
    params.sm_count = arguments.sm_count;

    // Each vLLM page stores its packed E2M1 tokens followed by E4M3 scales;
    // both regions share the cache page stride.
    uint64_t const page_stride = static_cast<uint64_t>(arguments.page_stride_bytes);
    uint8_t const *page_base = static_cast<uint8_t const *>(arguments.k_cache_ptr);
    uint64_t packed_dimensions[3] = {static_cast<uint64_t>(Traits::kHeadDim),
                                     static_cast<uint64_t>(Traits::kHeadDim / 2),
                                     static_cast<uint64_t>(arguments.physical_pages)};
    uint64_t packed_strides[2] = {static_cast<uint64_t>(Traits::kHeadDim), page_stride};
    uint32_t packed_box[3] = {static_cast<uint32_t>(Traits::kHeadDim),
                              static_cast<uint32_t>(Traits::kHeadDim / 2), 1u};
    status = encode_tma(params.packed_k, const_cast<uint8_t *>(page_base), packed_dimensions,
                        packed_strides, packed_box, CU_TENSOR_MAP_SWIZZLE_128B);
    if (status != cudaSuccess) {
      return status;
    }

    uint64_t scale_dimensions[3] = {static_cast<uint64_t>(Traits::kScaleBytes / 8), 8u,
                                    static_cast<uint64_t>(arguments.physical_pages)};
    uint64_t scale_strides[2] = {static_cast<uint64_t>(Traits::kScaleBytes / 8), page_stride};
    uint32_t scale_box[3] = {static_cast<uint32_t>(Traits::kScaleBytes / 8), 8u, 1u};
    return encode_tma(params.k_scale, const_cast<uint8_t *>(page_base + Traits::kPackedKBytes),
                      scale_dimensions, scale_strides, scale_box, CU_TENSOR_MAP_SWIZZLE_NONE);
  }

  static cudaError_t initialize() {
    static std::once_flag attribute_once;
    static cudaError_t attribute_status = cudaSuccess;
    std::call_once(attribute_once, [] {
      attribute_status = cudaFuncSetAttribute(indexer_gemm_kernel<Traits, kInlineCounterOffset>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              InlineKernel::kSharedStorageBytes);
      if (attribute_status == cudaSuccess) {
        attribute_status = cudaFuncSetAttribute(indexer_gemm_kernel<Traits, kDynamicCounterOffset>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                DynamicKernel::kSharedStorageBytes);
      }
      if (attribute_status == cudaSuccess) {
        attribute_status =
            cudaFuncSetAttribute(indexer_gemm_kernel<Traits, kInlineCounterOffset, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 InlinePostscaleKernel::kSharedStorageBytes);
      }
      if (attribute_status == cudaSuccess) {
        attribute_status =
            cudaFuncSetAttribute(indexer_gemm_kernel<Traits, kDynamicCounterOffset, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 DynamicPostscaleKernel::kSharedStorageBytes);
      }
      // A deep page ring for three CTAs per SM needs the oversized pool where
      // the device has one; elsewhere the default carveout is kept.
      int device = 0;
      int oversized_bytes = 0;
      if (attribute_status == cudaSuccess && cudaGetDevice(&device) == cudaSuccess &&
          cudaDeviceGetAttribute(&oversized_bytes, cudaDevAttrOversizedSharedMemoryPerBlock,
                                 device) == cudaSuccess &&
          oversized_bytes > 0) {
        cudaFuncSetAttribute(indexer_gemm_kernel<Traits, kInlineCounterOffset>,
                             cudaFuncAttributeSharedMemoryMode,
                             cudaSharedMemoryModeAllowOversizedSharedMemory);
        cudaFuncSetAttribute(indexer_gemm_kernel<Traits, kDynamicCounterOffset>,
                             cudaFuncAttributeSharedMemoryMode,
                             cudaSharedMemoryModeAllowOversizedSharedMemory);
        cudaFuncSetAttribute(indexer_gemm_kernel<Traits, kInlineCounterOffset, true>,
                             cudaFuncAttributeSharedMemoryMode,
                             cudaSharedMemoryModeAllowOversizedSharedMemory);
        cudaFuncSetAttribute(indexer_gemm_kernel<Traits, kDynamicCounterOffset, true>,
                             cudaFuncAttributeSharedMemoryMode,
                             cudaSharedMemoryModeAllowOversizedSharedMemory);
        cudaGetLastError();
      }
    });
    if (attribute_status != cudaSuccess) {
      return attribute_status;
    }
    return cudaSuccess;
  }

  static cudaError_t plan(IndexerGemmParams const &plan_params, void *scheduler_temp_storage,
                          size_t scheduler_temp_storage_bytes, cudaStream_t stream) {
    IndexerGemmParams params = plan_params;
    static_assert(sizeof(typename InlineKernel::SharedStorage) <= 110592,
                  "3 CTAs/SM need <= 108 KiB of shared memory per CTA");
    cudaError_t status = initialize();
    if (status != cudaSuccess) {
      return status;
    }
    int device = 0;
    status = cudaGetDevice(&device);
    if (status == cudaSuccess) {
      status = cudaDeviceGetAttribute(&params.sm_count, cudaDevAttrMultiProcessorCount, device);
    }
    if (status != cudaSuccess) {
      return status;
    }
    params.resident_ctas = params.sm_count * resident_ctas_per_sm();
    if (params.batch <= Traits::kPrepareThreads) {
      int const prepare_threads = ((params.batch + 31) / 32) * 32;
      prepare_indexer_gemm_scheduler<Traits><<<1, prepare_threads, 0, stream>>>(params);
      status = cudaGetLastError();
    } else {
      int constexpr prepare_threads = 256;
      int const prepare_blocks = (params.batch - 1) / prepare_threads + 1;
      prepare_indexer_gemm_scheduler_counts<Traits>
          <<<prepare_blocks, prepare_threads, 0, stream>>>(params);
      status = cudaGetLastError();
      if (status == cudaSuccess) {
        int32_t *prefix = params.scheduler_workspace_ptr + 1;
        status = run_indexer_gemm_scheduler_scan(prefix, params.batch, scheduler_temp_storage,
                                                 scheduler_temp_storage_bytes, stream);
      }
    }
    return status;
  }

  // CTAs of the score kernel resident per SM (registers and shared memory
  // permitting); the grid is one wave of them.
  static int resident_ctas_per_sm() {
    static std::once_flag once;
    static int ctas = 0;
    std::call_once(once, [] {
      if (initialize() != cudaSuccess ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &ctas, indexer_gemm_kernel<Traits, kInlineCounterOffset, true>,
              InlineKernel::kThreadCount, InlineKernel::kSharedStorageBytes) != cudaSuccess ||
          ctas < 1) {
        ctas = 1;
      }
    });
    return ctas;
  }

  static cudaError_t run(IndexerGemmParams const &params, cudaStream_t stream) {
    dim3 const grid(params.sm_count * resident_ctas_per_sm(), 1, 1);
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim.x = 1;
    attributes[0].val.clusterDim.y = 1;
    attributes[0].val.clusterDim.z = 1;
    attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[1].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = grid;
    launch_config.blockDim = dim3(InlineKernel::kThreadCount, 1, 1);
    launch_config.dynamicSmemBytes = InlineKernel::kSharedStorageBytes;
    launch_config.stream = stream;
    launch_config.attrs = attributes;
    launch_config.numAttrs = 2;
    bool const postscale = params.query_length <= 4;
    cudaError_t launch_status;
    if (params.batch <= Traits::kPrepareThreads) {
      launch_status =
          postscale
              ? cudaLaunchKernelEx(&launch_config,
                                   indexer_gemm_kernel<Traits, kInlineCounterOffset, true>, params)
              : cudaLaunchKernelEx(&launch_config,
                                   indexer_gemm_kernel<Traits, kInlineCounterOffset>, params);
    } else {
      launch_status =
          postscale
              ? cudaLaunchKernelEx(&launch_config,
                                   indexer_gemm_kernel<Traits, kDynamicCounterOffset, true>, params)
              : cudaLaunchKernelEx(&launch_config,
                                   indexer_gemm_kernel<Traits, kDynamicCounterOffset>, params);
    }
    return launch_status == cudaSuccess ? cudaGetLastError() : launch_status;
  }
};

} // namespace q8kv4_indexer_sm107
