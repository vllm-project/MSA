// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include "q8kv4_indexer/collective.hpp"

namespace q8kv4_indexer {

// Pages a request scores (all but its last, local page).
template <class Traits>
CUTE_DEVICE int indexer_gemm_scored_pages(IndexerGemmParams const &params, int batch_idx) {
  int const kv_length = params.kv_lengths_ptr[batch_idx];
  int const page_count_unclamped =
      kv_length > 0 ? (kv_length + Traits::kPageTokens - 1) / Traits::kPageTokens : 0;
  int const page_count = min(page_count_unclamped, params.max_pages);
  return max(page_count - 1, 0);
}

// Pages per work tile for a launch scoring total_pages pages: the largest
// power of two up to kPagesPerWorkTile that still makes about 1.5 work tiles
// per SM, but at least kMinPagesPerWorkTile. At decode batch sizes 16-page
// tiles leave most of the persistent grid idle while a few CTAs walk their
// tiles in several load rounds; much smaller tiles than this make the CTAs
// wait on more separate page loads (measured on GB300). Each page is scored on
// its own, so the tile size changes only which CTA scores a page, not the
// scores.
template <class Traits>
CUTE_DEVICE int indexer_gemm_pages_per_work_tile(int total_pages, int sm_count) {
#ifdef Q8KV4_INDEXER_FIXED_PAGES_PER_WORK_TILE
  return Q8KV4_INDEXER_FIXED_PAGES_PER_WORK_TILE;
#else
  int const min_tiles = sm_count * 3 / 2;
  int pages = Traits::kPagesPerWorkTile;
  while (pages > Traits::kMinPagesPerWorkTile && (total_pages + pages - 1) / pages < min_tiles) {
    pages >>= 1;
  }
  return pages;
#endif
}

template <class Traits>
CUTE_DEVICE int indexer_gemm_scheduler_tile_count(int scored_pages, int pages_per_work_tile) {
  return (scored_pages + pages_per_work_tile - 1) / pages_per_work_tile;
}

// Candidate pages of each query row, its local page included: query i of
// request b sits at position kv_length - query_length + i.
template <class Traits>
CUTE_DEVICE void write_decode_num_valid_pages(IndexerGemmParams const &params, int row,
                                              int row_stride) {
  if (params.num_valid_pages_ptr == nullptr) {
    return;
  }
  int const rows = params.batch * params.query_length;
  for (; row < rows; row += row_stride) {
    int const position = params.kv_lengths_ptr[row / params.query_length] -
                         params.query_length + row % params.query_length;
    // Padded requests (kv_length 0) have negative positions and keep one page.
    int const pages = position >= 0 ? position / Traits::kPageTokens + 1 : 1;
    params.num_valid_pages_ptr[row] = min(pages, params.max_pages);
  }
}

template <class Traits>
__global__ void __launch_bounds__(Traits::kPrepareThreads)
    prepare_indexer_gemm_scheduler(const __grid_constant__ IndexerGemmParams params) {
  __shared__ int32_t warp_prefix[Traits::kPrepareWarps];
  int const thread_idx = static_cast<int>(threadIdx.x);
  write_decode_num_valid_pages<Traits>(params, thread_idx, static_cast<int>(blockDim.x));
  int const lane_idx = thread_idx & 31;
  int const warp_idx = thread_idx >> 5;
  int const active_warps = static_cast<int>(blockDim.x) >> 5;
  // The launch's total scored pages pick the work tile size.
  int const scored_pages =
      thread_idx < params.batch ? indexer_gemm_scored_pages<Traits>(params, thread_idx) : 0;
  int warp_total = scored_pages;
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_total += __shfl_xor_sync(0xffffffffu, warp_total, offset);
  }
  if (lane_idx == 0) {
    warp_prefix[warp_idx] = warp_total;
  }
  __syncthreads();
  int total_pages = 0;
  for (int w = 0; w < active_warps; ++w) {
    total_pages += warp_prefix[w];
  }
  __syncthreads();
  int const pages_per_work_tile =
      indexer_gemm_pages_per_work_tile<Traits>(total_pages, params.sm_count);
  int tile_count = indexer_gemm_scheduler_tile_count<Traits>(scored_pages, pages_per_work_tile);
  for (int offset = 1; offset < 32; offset <<= 1) {
    int const addend = __shfl_up_sync(0xffffffffu, tile_count, offset);
    if (lane_idx >= offset) {
      tile_count += addend;
    }
  }
  if (lane_idx == 31) {
    warp_prefix[warp_idx] = tile_count;
  }
  __syncthreads();
  if (warp_idx == 0) {
    int warp_count = lane_idx < active_warps ? warp_prefix[lane_idx] : 0;
    for (int offset = 1; offset < 32; offset <<= 1) {
      int const addend = __shfl_up_sync(0xffffffffu, warp_count, offset);
      if (lane_idx >= offset) {
        warp_count += addend;
      }
    }
    if (lane_idx < active_warps) {
      warp_prefix[lane_idx] = warp_count;
    }
  }
  __syncthreads();
  tile_count += warp_idx > 0 ? warp_prefix[warp_idx - 1] : 0;
  if (thread_idx == 0) {
    params.scheduler_workspace_ptr[0] = 0;
    params.scheduler_workspace_ptr[Traits::kPrepareThreads + 1] = 0;
  }
  if (thread_idx < params.batch) {
    params.scheduler_workspace_ptr[thread_idx + 1] = tile_count;
  }
}

template <class Traits>
__global__ void __launch_bounds__(256)
    prepare_indexer_gemm_scheduler_counts(const __grid_constant__ IndexerGemmParams params) {
  int const batch_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  write_decode_num_valid_pages<Traits>(params, batch_idx,
                                       static_cast<int>(gridDim.x * blockDim.x));
  if (batch_idx == 0) {
    params.scheduler_workspace_ptr[-1] = 0;
    params.scheduler_workspace_ptr[0] = 0;
  }
  if (batch_idx < params.batch) {
    // Large batches fill the grid with full-size tiles.
    params.scheduler_workspace_ptr[batch_idx + 1] = indexer_gemm_scheduler_tile_count<Traits>(
        indexer_gemm_scored_pages<Traits>(params, batch_idx), Traits::kPagesPerWorkTile);
  }
}

template <class Traits, int SchedulerCounterOffset>
__global__ void __launch_bounds__(Traits::kThreads)
    indexer_gemm_kernel(const __grid_constant__ IndexerGemmParams params) {
  using Collective = IndexerGemmCollective<Traits, SchedulerCounterOffset>;
  using SharedStorage = typename Collective::SharedStorage;
  extern __shared__ __align__(1024) uint8_t shared_memory[];
  SharedStorage &storage = *reinterpret_cast<SharedStorage *>(shared_memory);
  Collective::run(params, storage);
}

template <class Traits, int SchedulerCounterOffset> struct IndexerGemmKernel {
  using Collective = IndexerGemmCollective<Traits, SchedulerCounterOffset>;
  using SharedStorage = typename Collective::SharedStorage;
  static constexpr int kThreadCount = Traits::kThreads;
  static constexpr int kSharedStorageBytes = sizeof(SharedStorage);
};

} // namespace q8kv4_indexer
