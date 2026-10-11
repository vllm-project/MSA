// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "cute/arch/util.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "q8kv4_indexer_sm107/nvfp4_to_e4m3.cuh"
#include "q8kv4_indexer_sm107/params.hpp"
#include "q8kv4_indexer_sm107/traits.hpp"

namespace q8kv4_indexer_sm107 {

template <class Traits, int SchedulerCounterOffset> struct IndexerGemmConfig {
  using Params = IndexerGemmParams;

  struct alignas(1024) SharedStorage {
    uint8_t pages[Traits::kMaxPagesPerCta][Traits::kPageBytes];
    // Per tile buffer: page partial maxima of every warp, folded by the tile's
    // last warp. Layout per page: [warp][query slot][group of 4 token rows].
    alignas(16) float page_partials[2][Traits::kPartialPageCapacity *
                                       Traits::kPartialPageStride];
    alignas(16) uint8_t q_tile[2][Traits::kQueryLength * Traits::kHeadDim];
    uint64_t page_barriers[Traits::kMaxPagesPerCta];
    uint64_t page_consumed_barriers[Traits::kMaxPagesPerCta];
    // Tile buffers: full when the transport warp published a tile, empty once
    // every score warp has read it, folded once its scores are stored.
    uint64_t tile_full_barriers[2];
    uint64_t tile_empty_barriers[2];
    uint64_t tile_folded_barriers[2];
    int32_t tile_done_warps[2];
    int32_t tile_valid[2];
    int32_t tile_batch_idx[2];
    int32_t tile_kv_length[2];
    int32_t tile_page_begin[2];
    int32_t tile_local_pages[2];
    int32_t work_tile_id;
    int32_t work_count;
    int32_t tile_pages;
    int32_t batch_idx;
    int32_t work_tile_idx;
  };

  CUTE_DEVICE static uint32_t smem_address(void const *pointer) {
    return cute::cast_smem_ptr_to_uint(pointer);
  }

  CUTE_DEVICE static void init_barrier(uint64_t *barrier, uint32_t count) {
    cutlass::arch::ClusterBarrier::init(barrier, count);
  }

  CUTE_DEVICE static void expect_transactions(uint64_t *barrier, uint32_t bytes) {
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barrier, bytes);
  }

  CUTE_DEVICE static void arrive(uint64_t *barrier) {
    cutlass::arch::ClusterBarrier::arrive(barrier);
  }

  CUTE_DEVICE static void wait(uint64_t *barrier, uint32_t phase) {
    cutlass::arch::ClusterBarrier::wait(barrier, phase);
  }

  // Tight poll without the nanosleep backoff: for latency-critical page waits.
  CUTE_DEVICE static void wait_poll(uint64_t *barrier, uint32_t phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, 0x989680;\n\t"
        "@!P1 bra LAB_WAIT;\n\t"
        "}"
        :
        : "r"(smem_address(barrier)), "r"(phase)
        : "memory");
  }


  CUTE_DEVICE static void convert_e4m3x4_to_f16x2(uint32_t value, uint32_t &low, uint32_t &high) {
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(low) : "h"(static_cast<uint16_t>(value)));
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                 : "=r"(high)
                 : "h"(static_cast<uint16_t>(value >> 16)));
  }

  CUTE_DEVICE static void hmma_f16(float (&accumulator)[4], uint32_t const (&matrix)[4],
                                   uint2 query) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]),
                   "+f"(accumulator[3])
                 : "r"(matrix[0]), "r"(matrix[1]), "r"(matrix[2]), "r"(matrix[3]), "r"(query.x),
                   "r"(query.y));
  }

  CUTE_DEVICE static void load_matrix_x4(uint32_t (&matrix)[4], uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(matrix[0]), "=r"(matrix[1]), "=r"(matrix[2]), "=r"(matrix[3])
                 : "r"(address));
  }

  CUTE_DEVICE static int swizzled_page_offset(int token, int quarter) {
    return (token >> 1) * 128 + ((((token & 1) * 4 + quarter) ^ ((token >> 1) & 7)) * 16);
  }

  CUTE_DEVICE static void issue_page_load(Params const &params, SharedStorage &storage, int slot,
                                          int physical_page) {
    uint64_t *barrier = storage.page_barriers + slot;
    expect_transactions(barrier, Traits::kPageBytes);
    uint64_t constexpr cache_hint_evict_first = 0x12f0000000000000ULL;
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::"
                 "complete_tx::bytes.L2::cache_hint "
                 "[%0], [%1, {%2, %3, %4}], [%5], %6;"
                 :
                 : "r"(smem_address(storage.pages[slot])),
                   "l"(reinterpret_cast<uint64_t>(&params.packed_k)), "r"(0), "r"(0),
                   "r"(physical_page), "r"(smem_address(barrier)), "l"(cache_hint_evict_first)
                 : "memory");
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::"
                 "complete_tx::bytes.L2::cache_hint "
                 "[%0], [%1, {%2, %3, %4}], [%5], %6;"
                 :
                 : "r"(smem_address(storage.pages[slot]) + Traits::kPackedKBytes),
                   "l"(reinterpret_cast<uint64_t>(&params.k_scale)), "r"(0), "r"(0),
                   "r"(physical_page), "r"(smem_address(barrier)), "l"(cache_hint_evict_first)
                 : "memory");
  }

  CUTE_DEVICE static int permute_claim(int work_tile_id, int work_count) {
    if (work_count > 1) {
      int const permutation_bits = 32 - __clz(work_count - 1);
      int const candidate =
          static_cast<int>(__brev(static_cast<uint32_t>(work_tile_id)) >> (32 - permutation_bits));
      if (candidate < work_count) {
        return candidate;
      }
    }
    return work_tile_id;
  }

  // Request of a work tile: largest b in [0, batch) with prefix[b] <= id, by a
  // 32-ary warp search. The first round's prefix entries depend only on the
  // batch, so each lane loads its entry once (prefix_lane) and a claim needs
  // no load at all for batch <= 32.
  CUTE_DEVICE static void locate(int32_t const *prefix_ptr, int batch, int lane_idx,
                                 int prefix_lane, int first_step, int id, int &batch_idx,
                                 int &work_tile_idx) {
    int lower = 0;
    int upper = batch;
    int step = first_step;
    int prefix = prefix_lane;
    int lower_prefix = 0;
    while (upper - lower > 1) {
      int const last = 31 - __clz(__ballot_sync(0xffffffffu, prefix <= id));
      lower_prefix = __shfl_sync(0xffffffffu, prefix, last);
      lower += last * step;
      upper = min(lower + step, upper);
      step = (upper - lower + 31) >> 5;
      if (upper - lower > 1) {
        int const index = lower + lane_idx * step;
        prefix = index < upper ? prefix_ptr[index] : 0x7fffffff;
      }
    }
    batch_idx = lower;
    work_tile_idx = id - lower_prefix;
  }
};

} // namespace q8kv4_indexer_sm107
