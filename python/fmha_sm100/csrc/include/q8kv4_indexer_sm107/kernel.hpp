// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include "q8kv4_indexer_sm107/collective.hpp"

namespace q8kv4_indexer_sm107 {

template <class Traits>
CUTE_DEVICE int indexer_gemm_scored_pages(IndexerGemmParams const &params, int batch_idx) {
  int const kv_length = params.kv_lengths_ptr[batch_idx];
  int const page_count_unclamped =
      kv_length > 0 ? (kv_length + Traits::kPageTokens - 1) / Traits::kPageTokens : 0;
  int const page_count = min(page_count_unclamped, params.max_pages);
  return max(page_count - 1, 0);
}

template <class Traits>
CUTE_DEVICE int indexer_gemm_scheduler_tile_count(IndexerGemmParams const &params, int batch_idx) {
  int const scored_pages = indexer_gemm_scored_pages<Traits>(params, batch_idx);
  return (scored_pages + Traits::kBandwidthTilePages - 1) / Traits::kBandwidthTilePages;
}

// Pages per work tile for a launch, from the tile counts of every candidate
// size (tiles[t] = sum over requests of ceil(scored_pages / t)):
//  - if even the largest tile needs more than one wave of resident CTAs, the
//    launch is HBM-bound and uses kBandwidthTilePages;
//  - otherwise about sqrt(sm_count * pages) tiles: one more dispatch rank of
//    sm_count CTAs costs about as much as one more page per CTA streamed at
//    one CTA's latency-bound rate, so the two balance at sqrt(S * P) tiles;
//    never more than one wave.
// Sizes up to the page ring's depth are all efficient; beyond it the score
// warps take pages in pairs, so sizes round up to even.
template <class Traits>
CUTE_DEVICE int choose_tile_pages(int32_t const *tiles, int sm_count, int resident_ctas) {
  constexpr int kMaxTile = Traits::kPagesPerWorkTile;
  auto bucket = [](int t) {
    t += (t > Traits::kMaxPagesPerCta && (t & 1)) ? 1 : 0;
    return min(t, kMaxTile);
  };
  int const total_pages = tiles[0];
  if (total_pages == 0) {
    return 1;
  }
  if (tiles[kMaxTile - 1] > resident_ctas) {
    return min(Traits::kBandwidthTilePages, kMaxTile);
  }
  int t = bucket(
      max(static_cast<int>(ceilf(sqrtf(static_cast<float>(total_pages) / sm_count))), 1));
  while (tiles[t - 1] > resident_ctas) {
    t = bucket(t + 1);
  }
  return t;
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
  __shared__ int32_t warp_tiles[Traits::kPrepareWarps][Traits::kPagesPerWorkTile];
  __shared__ int32_t chosen_tile_pages;
  int scored_pages = 0;
  if (thread_idx < params.batch) {
    scored_pages = indexer_gemm_scored_pages<Traits>(params, thread_idx);
  }
  // Tile count of every candidate size, summed over the requests.
  CUTLASS_PRAGMA_UNROLL
  for (int t = 1; t <= Traits::kPagesPerWorkTile; ++t) {
    int tiles = (scored_pages + t - 1) / t;
    for (int offset = 16; offset > 0; offset >>= 1) {
      tiles += __shfl_xor_sync(0xffffffffu, tiles, offset);
    }
    if (lane_idx == 0) {
      warp_tiles[warp_idx][t - 1] = tiles;
    }
  }
  __syncthreads();
  if (thread_idx < Traits::kPagesPerWorkTile) {
    for (int warp = 1; warp < active_warps; ++warp) {
      warp_tiles[0][thread_idx] += warp_tiles[warp][thread_idx];
    }
  }
  __syncthreads();
  if (thread_idx == 0) {
    chosen_tile_pages =
        choose_tile_pages<Traits>(warp_tiles[0], params.sm_count, params.resident_ctas);
  }
  __syncthreads();
  int const tile_pages = chosen_tile_pages;
  int tile_count = (scored_pages + tile_pages - 1) / tile_pages;
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
    params.scheduler_workspace_ptr[Traits::kTilePagesWord] = tile_pages;
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
    params.scheduler_workspace_ptr[batch_idx + 1] =
        indexer_gemm_scheduler_tile_count<Traits>(params, batch_idx);
  }
}

template <class Traits, int SchedulerCounterOffset, bool Postscale = false>
__global__ void __launch_bounds__(Traits::kThreads, 3)
    indexer_gemm_kernel(const __grid_constant__ IndexerGemmParams params) {
  using Collective = IndexerGemmCollective<Traits, SchedulerCounterOffset, Postscale>;
  using SharedStorage = typename Collective::SharedStorage;
  extern __shared__ __align__(1024) uint8_t shared_memory[];
  SharedStorage &storage = *reinterpret_cast<SharedStorage *>(shared_memory);
  Collective::run(params, storage);
}

template <class Traits, int SchedulerCounterOffset, bool Postscale = false>
struct IndexerGemmKernel {
  using Collective = IndexerGemmCollective<Traits, SchedulerCounterOffset, Postscale>;
  using SharedStorage = typename Collective::SharedStorage;
  static constexpr int kThreadCount = Traits::kThreads;
  static constexpr int kSharedStorageBytes = sizeof(SharedStorage);
};

} // namespace q8kv4_indexer_sm107
