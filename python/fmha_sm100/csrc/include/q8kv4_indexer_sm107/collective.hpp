// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "q8kv4_indexer_sm107/config.hpp"

namespace q8kv4_indexer_sm107 {

template <class Traits, int SchedulerCounterOffset, bool Postscale = false>
struct IndexerGemmCollective : IndexerGemmConfig<Traits, SchedulerCounterOffset> {
  using Base = IndexerGemmConfig<Traits, SchedulerCounterOffset>;
  using Base::arrive;
  using Base::locate;
  using Base::permute_claim;
  using Base::convert_e4m3x4_to_f16x2;
  using typename Base::Params;
  using typename Base::SharedStorage;
  using Base::hmma_f16;
  using Base::init_barrier;
  using Base::issue_page_load;
  using Base::load_matrix_x4;
  using Base::smem_address;
  using Base::swizzled_page_offset;
  using Base::wait;
  using Base::wait_poll;

  struct PageFragments {
    uint32_t matrix_a[4];
    uint32_t matrix_b[4];
  };

  CUTE_DEVICE static void run(Params const &params, SharedStorage &storage) {
    int const thread_idx = static_cast<int>(threadIdx.x);
    int const warp_idx = thread_idx / cutlass::NumThreadsPerWarp;
    int const lane_idx = thread_idx % cutlass::NumThreadsPerWarp;
    constexpr int kSlots = Traits::kMaxPagesPerCta;

    // Claim before the PDL wait (see the original kernel for why).
    if (thread_idx == 0) {
      for (int slot = 0; slot < kSlots; ++slot) {
        init_barrier(storage.page_barriers + slot, 1);
        init_barrier(storage.page_consumed_barriers + slot, Traits::kScoreWarps * 32);
      }
      for (int buffer = 0; buffer < 2; ++buffer) {
        init_barrier(storage.tile_full_barriers + buffer, 1);
        init_barrier(storage.tile_empty_barriers + buffer, Traits::kScoreWarps);
        init_barrier(storage.tile_folded_barriers + buffer, 1);
        storage.tile_done_warps[buffer] = 0;
      }
    }
    cutlass::arch::fence_barrier_init();

    // Barriers and the first claim are visible to every warp. The PDL wait is
    // per warp: the transport warp runs ahead of it (below), the score warps
    // wait before they read q or store anything.
    __syncthreads();

    if (warp_idx == Traits::kScoreWarps) {
      // Transport warp: publishes tiles and streams their pages, running ahead
      // of the score warps by up to kSlots pages and one tile buffer.
      // First tile: the block index, located with the per-lane first-round
      // prefix entries, which load together with the work count.
      int32_t *const workspace = params.scheduler_workspace_ptr;
      int const first_step = (params.batch + 31) >> 5;
      int const prefix_lane =
          lane_idx * first_step < params.batch ? workspace[lane_idx * first_step] : 0x7fffffff;
      int const work_count = workspace[params.batch];
      // The inline plan picks the tile size per launch; the CUB plan (large
      // batches) keeps the full tile.
      int const tile_pages = SchedulerCounterOffset > 0 ? workspace[Traits::kTilePagesWord]
                                                        : Traits::kBandwidthTilePages;
      int work_tile_id = static_cast<int>(blockIdx.x);
      int batch_idx = 0;
      int work_tile_idx = 0;
      int counter_value = -1;
      if (work_tile_id < work_count) {
        locate(workspace, params.batch, lane_idx, prefix_lane, first_step,
               permute_claim(work_tile_id, work_count), batch_idx, work_tile_idx);
      }
      // Ring position of the next page to stage, kept as running counters so a
      // page costs an increment and a wrap instead of a division by kSlots.
      int ring_slot = 0;
      uint32_t ring_wait_phase = 1;
      bool ring_wrapped = false;
      bool waited = false;
      for (int tile_seq = 0;; ++tile_seq) {
        int const buffer = tile_seq & 1;
        bool const valid = work_tile_id < work_count;
        if (tile_seq >= 2) {
          wait(storage.tile_empty_barriers + buffer,
               static_cast<uint32_t>(((tile_seq >> 1) - 1) & 1));
        }
        int local_pages = 0;
        int page_begin = 0;
        int kv_length = 0;
        int tile_physical_page = 0;
        if (valid) {
          kv_length = __ldg(params.kv_lengths_ptr + batch_idx);
          page_begin = work_tile_idx * tile_pages;
          // Page ids do not depend on kv_length: load them alongside it.
          int32_t const *page_table =
              params.page_table_ptr + static_cast<size_t>(batch_idx) * params.max_pages;
          if (lane_idx < tile_pages && page_begin + lane_idx < params.max_pages) {
            tile_physical_page = __ldg(page_table + page_begin + lane_idx);
          }
          int const page_count =
              kv_length > 0 ? (kv_length + Traits::kPageTokens - 1) / Traits::kPageTokens : 0;
          int const scored_page_count = max(page_count - 1, 0);
          int const page_end = min(page_begin + tile_pages, params.max_pages);
          local_pages = max(min(page_end, scored_page_count) - page_begin, 0);
        }
        auto issue = [&](int local_page) {
          int const physical_page = __shfl_sync(0xffffffffu, tile_physical_page, local_page);
          if (ring_wrapped) {
            wait_poll(storage.page_consumed_barriers + ring_slot, ring_wait_phase);
          }
          if (cute::elect_one_sync()) {
            issue_page_load(params, storage, ring_slot, physical_page);
          }
          if (++ring_slot == kSlots) {
            ring_slot = 0;
            ring_wait_phase ^= 1u;
            ring_wrapped = true;
          }
        };
        int issued = 0;
        if (!waited) {
          // Before the PDL wait: only pages older than this step's first new
          // token, which the predecessor (the index-K insert) cannot be writing.
          int const first_new_page = max(kv_length - params.query_length, 0) / Traits::kPageTokens;
          int const history_pages = min(max(first_new_page - page_begin, 0), local_pages);
          for (; issued < history_pages && !ring_wrapped; ++issued) {
            issue(issued);
          }
          cudaGridDependencySynchronize();
          waited = true;
        }
        if (valid) {
          // The request's query_length queries take the last of the eight slots.
          constexpr int kChunksPerQuery = Traits::kHeadDim / 16;
          CUTLASS_PRAGMA_UNROLL
          for (int chunk = lane_idx; chunk < Traits::kQueryLength * kChunksPerQuery;
               chunk += cutlass::NumThreadsPerWarp) {
            int const slot =
                chunk / kChunksPerQuery - (Traits::kQueryLength - params.query_length);
            uint4 value = make_uint4(0u, 0u, 0u, 0u);
            if (slot >= 0) {
              size_t const query_offset =
                  (static_cast<size_t>(batch_idx) * params.query_length + slot) * Traits::kHeadDim;
              value = reinterpret_cast<uint4 const *>(params.q_ptr + query_offset)[chunk %
                                                                                  kChunksPerQuery];
            }
            reinterpret_cast<uint4 *>(storage.q_tile[buffer])[chunk] = value;
          }
        }
        if (tile_seq >= 2) {
          // The fold of this buffer's previous tile reads its metadata after
          // the page loop, so it must finish before the metadata is replaced.
          wait(storage.tile_folded_barriers + buffer,
               static_cast<uint32_t>(((tile_seq >> 1) - 1) & 1));
        }
        if (lane_idx == 0) {
          storage.tile_valid[buffer] = valid;
          storage.tile_batch_idx[buffer] = batch_idx;
          storage.tile_kv_length[buffer] = kv_length;
          storage.tile_page_begin[buffer] = page_begin;
          storage.tile_local_pages[buffer] = local_pages;
        }
        __threadfence_block();
        __syncwarp();
        if (lane_idx == 0) {
          arrive(storage.tile_full_barriers + buffer);
        }
        if (!valid) {
          break;
        }
        for (; issued < local_pages; ++issued) {
          issue(issued);
        }
        {
          int claimed = 0;
          if (lane_idx == 0) {
            claimed = atomicAdd(workspace + SchedulerCounterOffset, 1);
          }
          counter_value = __shfl_sync(0xffffffffu, claimed, 0);
          work_tile_id = static_cast<int>(gridDim.x) + counter_value;
          if (work_tile_id < work_count) {
            locate(workspace, params.batch, lane_idx, prefix_lane, first_step,
                   permute_claim(work_tile_id, work_count), batch_idx, work_tile_idx);
          }
        }
      }
      // Every CTA with a valid first tile ends with one failed counter claim, so a
      // launch takes exactly work_count counter values; the CTA that got the last
      // one resets the counter for the next launch.
      if (lane_idx == 0 && counter_value == work_count - 1) {
        params.scheduler_workspace_ptr[SchedulerCounterOffset] = 0;
      }
      cudaTriggerProgrammaticLaunchCompletion();
      return;
    }

    // Score warps: q and the step's new pages come from the predecessor, and
    // the scores buffer may still be read by the previous top-k.
    cudaGridDependencySynchronize();
    int const query_column = lane_idx >> 2;
    int const thread_group = lane_idx & 3;
    // With at most 4 queries the 8 MMA columns become 4 queries x 2 block
    // parities: column 4h+j accumulates query slot 4+j from A fragments of
    // block parity h only, so the per-block scale can multiply the MMA result
    // instead of every converted K value. Lanes whose parity differs read a
    // zero q slot (slots 0..3 are zero whenever query_length <= 4).
    constexpr bool postscale = Postscale;
    int const remap_slot = (thread_group >> 1) == (query_column >> 2)
                               ? 4 + (query_column & 3)
                               : (query_column & 3);
    int const query_base =
        (postscale ? remap_slot : query_column) * Traits::kHeadDim + 8 * thread_group;
    int const token_begin = warp_idx * 16;
    int const matrix_offset_a = swizzled_page_offset(token_begin + (lane_idx & 7), lane_idx >> 3);
    int const matrix_offset_b =
        swizzled_page_offset(token_begin + 8 + (lane_idx & 7), lane_idx >> 3);
    int const scale_offset_a =
        Traits::kPackedKBytes + (token_begin + (lane_idx >> 2)) * Traits::kScaleGroups;
    // scale_offset_b = scale_offset_a + 8 * Traits::kScaleGroups, folded into the LDS immediate.
    // scale selector inlined at the byte_perm below: thread_group >> 1 ? 0x7531u : 0x6420u

    // Ring position of the next page this warp consumes, as a running counter:
    // slot in bits 0..2, wait phase in bit 3 (flips on every wrap of the
    // 6-slot ring). Replaces a page_ticket % kSlots and / kSlots on every page.
    uint32_t ring = 0;
    auto ring_step = [&]() {
      uint32_t const next = ring + 1;
      ring = (next & 7) == kSlots ? (next & 8) ^ 8 : next;
    };
    for (int tile_seq = 0;; ++tile_seq) {
      int const buffer = tile_seq & 1;
      wait(storage.tile_full_barriers + buffer, static_cast<uint32_t>((tile_seq >> 1) & 1));
      if (!storage.tile_valid[buffer]) {
        break;
      }
      // batch_idx/kv_length/page_begin are read later, in the cold fold path,
      // to keep them out of the page loop's register budget.
      int const local_pages = storage.tile_local_pages[buffer];
      // Query fragments of this lane, (low.x, low.y, high.x, high.y) per k chunk.
      uint4 query_fragments[4];
      CUTLASS_PRAGMA_UNROLL
      for (int k_chunk = 0; k_chunk < 4; ++k_chunk) {
        int const query_offset = query_base + 32 * k_chunk;
        uint2 const query_fragment =
            *reinterpret_cast<uint2 const *>(storage.q_tile[buffer] + query_offset);
        convert_e4m3x4_to_f16x2(query_fragment.x, query_fragments[k_chunk].x,
                                query_fragments[k_chunk].z);
        convert_e4m3x4_to_f16x2(query_fragment.y, query_fragments[k_chunk].y,
                                query_fragments[k_chunk].w);
      }
      __syncwarp();
      if (cute::elect_one_sync()) {
        arrive(storage.tile_empty_barriers + buffer);
      }
      if (tile_seq >= 2) {
        // The partials of this buffer's previous tile have been folded.
        wait(storage.tile_folded_barriers + buffer,
             static_cast<uint32_t>(((tile_seq >> 1) - 1) & 1));
      }

      auto load_page = [&](auto slot_c, uint32_t phase, PageFragments &fragments) {
        // With an integral_constant slot every address folds to an immediate.
        int const slot = slot_c;
        wait_poll(storage.page_barriers + slot, phase);
        uint8_t const *packed = storage.pages[slot];
        load_matrix_x4(fragments.matrix_a, smem_address(packed + matrix_offset_a));
        load_matrix_x4(fragments.matrix_b, smem_address(packed + matrix_offset_b));
      };

      auto compute_page = [&](float *partial_row, auto slot_c, PageFragments const &fragments) {
        // Scale loads sit with their consumer so the two in-flight pages' scale
        // registers never overlap (register budget).
        int const slot = slot_c;
        uint8_t const *packed = storage.pages[slot];
        uint2 const scale_a_packed = *reinterpret_cast<uint2 const *>(packed + scale_offset_a);
        uint2 const scale_b_packed =
            *reinterpret_cast<uint2 const *>(packed + scale_offset_a + 8 * Traits::kScaleGroups);
        float accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        uint32_t scales_a[2];
        uint32_t scales_b[2];
        convert_e4m3x4_to_f16x2(__byte_perm(scale_a_packed.x, scale_a_packed.y,
                                            thread_group >> 1 ? 0x7531u : 0x6420u),
                                scales_a[0], scales_a[1]);
        convert_e4m3x4_to_f16x2(__byte_perm(scale_b_packed.x, scale_b_packed.y,
                                            thread_group >> 1 ? 0x7531u : 0x6420u),
                                scales_b[0], scales_b[1]);
        CUTLASS_PRAGMA_UNROLL
        for (int k_chunk = 0; k_chunk < 4; ++k_chunk) {
          uint32_t const scale_a =
              k_chunk & 1 ? detail::broadcast_f16x2<1>(scales_a[k_chunk >> 1])
                          : detail::broadcast_f16x2<0>(scales_a[k_chunk >> 1]);
          uint32_t const scale_b =
              k_chunk & 1 ? detail::broadcast_f16x2<1>(scales_b[k_chunk >> 1])
                          : detail::broadcast_f16x2<0>(scales_b[k_chunk >> 1]);
          uint32_t matrix_low[4];
          uint32_t matrix_high[4];
          if constexpr (postscale) {
            // Unscaled converts; the per-block scale multiplies the k_chunk's
            // MMA result (exact fp32) instead of every converted K value.
            detail::nvfp4x8_to_f16x2x4_unscaled(
                matrix_low[0], matrix_high[0], matrix_low[2], matrix_high[2],
                fragments.matrix_a[k_chunk]);
            detail::nvfp4x8_to_f16x2x4_unscaled(
                matrix_low[1], matrix_high[1], matrix_low[3], matrix_high[3],
                fragments.matrix_b[k_chunk]);
            float partial[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            hmma_f16(partial, matrix_low,
                     uint2{query_fragments[k_chunk].x, query_fragments[k_chunk].y});
            hmma_f16(partial, matrix_high,
                     uint2{query_fragments[k_chunk].z, query_fragments[k_chunk].w});
            detail::fma2_f32x2_by_f16x2(accumulator[0], accumulator[1],
                                                              scale_a, partial[0], partial[1]);
            detail::fma2_f32x2_by_f16x2(accumulator[2], accumulator[3],
                                                              scale_b, partial[2], partial[3]);
          } else {
            detail::nvfp4x8_to_f16x2x4_scaled(
                matrix_low[0], matrix_high[0], matrix_low[2], matrix_high[2],
                fragments.matrix_a[k_chunk], scale_a);
            detail::nvfp4x8_to_f16x2x4_scaled(
                matrix_low[1], matrix_high[1], matrix_low[3], matrix_high[3],
                fragments.matrix_b[k_chunk], scale_b);
            hmma_f16(accumulator, matrix_low,
                     uint2{query_fragments[k_chunk].x, query_fragments[k_chunk].y});
            hmma_f16(accumulator, matrix_high,
                     uint2{query_fragments[k_chunk].z, query_fragments[k_chunk].w});
          }
        }
        if constexpr (postscale) {
          // Lane t holds columns 4h+j of its own block parity h = t>>1; lane
          // t^2 holds the same queries of the other parity. Sum them per row.
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < 4; ++i) {
            accumulator[i] += __shfl_xor_sync(0xffffffffu, accumulator[i], 2);
          }
        }
        float maximum0 = fmaxf(accumulator[0], accumulator[2]);
        float maximum1 = fmaxf(accumulator[1], accumulator[3]);
        // One butterfly stage only (lanes 4 apart = adjacent token groups); the
        // four resulting partials per slot are folded by the tile's last warp.
        maximum0 = fmaxf(maximum0, __shfl_xor_sync(0xffffffffu, maximum0, 4));
        maximum1 = fmaxf(maximum1, __shfl_xor_sync(0xffffffffu, maximum1, 4));
        if ((lane_idx & 4) == 0) {
          if constexpr (postscale) {
            // Remapped layout: thread groups 0/1 hold the page partial maxima
            // of the real query slots 4..7; the unused slots 0..3 were zeroed
            // up front.
            if (thread_group < 2) {
              partial_row[0] = maximum0;
              partial_row[4] = maximum1;
            }
          } else {
            partial_row[0] = maximum0;
            partial_row[4] = maximum1;
          }
        }
      };

      PageFragments fragments_a;
      PageFragments fragments_b;
      // Partials row this lane writes, stepped by pages instead of recomputed
      // from local_page. Layout per page: [warp][query slot][group of 4].
      float *partial_ptr = storage.page_partials[buffer] + warp_idx * 32 +
                           ((postscale ? 4 : 0) + 2 * thread_group) * 4 + (lane_idx >> 3);
      int remaining = local_pages;

      // A single page through the generic running-counter path.
      auto single_page = [&]() {
        int const slot = ring & 7;
        load_page(slot, ring >> 3, fragments_a);
        compute_page(partial_ptr, slot, fragments_a);
        arrive(storage.page_consumed_barriers + slot);
        ring_step();
        partial_ptr += Traits::kPartialPageStride;
        --remaining;
      };

      // One pair with compile-time ring slots: every smem address folds into a
      // register+immediate and the mbarrier ops need no address arithmetic.
      auto pair_ct = [&](auto slot_a_c, auto slot_b_c, uint32_t phase_a, uint32_t phase_b) {
        constexpr int slot_a = decltype(slot_a_c)::value;
        constexpr int slot_b = decltype(slot_b_c)::value;
        load_page(slot_a_c, phase_a, fragments_a);
        load_page(slot_b_c, phase_b, fragments_b);
        compute_page(partial_ptr, slot_a_c, fragments_a);
        compute_page(partial_ptr + Traits::kPartialPageStride, slot_b_c, fragments_b);
        partial_ptr += 2 * Traits::kPartialPageStride;
        arrive(storage.page_consumed_barriers + slot_a);
        arrive(storage.page_consumed_barriers + slot_b);
      };

      // Full ring cycles (three pairs, six pages) starting at even slot S. A
      // cycle preserves the running slot, so only the phase advances.
      auto bulk_cycles = [&](auto s_c) {
        constexpr int S = decltype(s_c)::value;
        static_assert(S % 2 == 0 && S < kSlots);
        using C0 = std::integral_constant<int, (S + 0) % kSlots>;
        using C1 = std::integral_constant<int, (S + 1) % kSlots>;
        using C2 = std::integral_constant<int, (S + 2) % kSlots>;
        using C3 = std::integral_constant<int, (S + 3) % kSlots>;
        using C4 = std::integral_constant<int, (S + 4) % kSlots>;
        using C5 = std::integral_constant<int, (S + 5) % kSlots>;
        // Phase offset of each position: wraps once the raw index passes 6.
        constexpr bool w2 = S + 2 >= kSlots, w3 = S + 3 >= kSlots, w4 = S + 4 >= kSlots;
        constexpr bool w5 = S + 5 >= kSlots;
        while (remaining >= kSlots) {
          uint32_t const p = ring >> 3;
          uint32_t const np = p ^ 1u;
          pair_ct(C0{}, C1{}, p, p);
          pair_ct(C2{}, C3{}, w2 ? np : p, w3 ? np : p);
          pair_ct(C4{}, C5{}, w4 ? np : p, w5 ? np : p);
          ring = static_cast<uint32_t>(S) + 8 * np;
          remaining -= kSlots;
        }
      };

      // Pairs are ring-aligned: an odd starting slot gets one head page, the
      // bulk cycles run with compile-time slots, and the generic path finishes
      // the partial cycle (up to two pairs and a single page).
      if (remaining > 0 && (ring & 1)) {
        single_page();
      }
      switch (ring & 7) {
        case 0: bulk_cycles(std::integral_constant<int, 0>{}); break;
        case 2: bulk_cycles(std::integral_constant<int, 2>{}); break;
        case 4: bulk_cycles(std::integral_constant<int, 4>{}); break;
        default: break;
      }
      while (remaining >= 2) {
        int const slot_a = ring & 7;
        uint32_t const phase_a = ring >> 3;
        ring_step();
        int const slot_b = ring & 7;
        uint32_t const phase_b = ring >> 3;
        ring_step();
        load_page(slot_a, phase_a, fragments_a);
        load_page(slot_b, phase_b, fragments_b);
        compute_page(partial_ptr, slot_a, fragments_a);
        compute_page(partial_ptr + Traits::kPartialPageStride, slot_b, fragments_b);
        partial_ptr += 2 * Traits::kPartialPageStride;
        arrive(storage.page_consumed_barriers + slot_a);
        arrive(storage.page_consumed_barriers + slot_b);
        remaining -= 2;
      }
      if (remaining > 0) {
        single_page();
      }

      // The tile's last score warp folds the eight warps' maxima and stores the
      // scores, so no warp waits for the others at the end of a tile.
      __threadfence_block();
      __syncwarp();
      int last_warp = 0;
      if (lane_idx == 0) {
        last_warp = atomicAdd(storage.tile_done_warps + buffer, 1) == Traits::kScoreWarps - 1;
      }
      last_warp = __shfl_sync(0xffffffffu, last_warp, 0);
      if (last_warp) {
        __threadfence_block();
        int const batch_idx = storage.tile_batch_idx[buffer];
        int const kv_length = storage.tile_kv_length[buffer];
        int const page_begin = storage.tile_page_begin[buffer];
        int const local_pages = storage.tile_local_pages[buffer];
        // Lane's query row is fixed; its pages advance by 4 each step. Fold and
        // store only pages inside the query's range (as the old loop did).
        int const row = lane_idx & 7;
        int const query_position = max(kv_length - Traits::kQueryLength + row, 0);
        int const valid_pages =
            min(local_pages, max(query_position / Traits::kPageTokens - page_begin, 0));
        float const *partial = storage.page_partials[buffer] +
                               (lane_idx >> 3) * Traits::kPartialPageStride + row * 4;
        size_t const output_base =
            (static_cast<size_t>(batch_idx) * Traits::kQueryLength + row) * params.max_pages +
            page_begin;
        for (int tile_page = lane_idx >> 3; tile_page < Traits::kPagesPerWorkTile;
             tile_page += 4) {
          if (tile_page < valid_pages) {
            float maximum;
            {
              float4 const p0 = *reinterpret_cast<float4 const *>(partial);
              maximum = fmaxf(fmaxf(p0.x, p0.y), fmaxf(p0.z, p0.w));
            }
            CUTLASS_PRAGMA_UNROLL
            for (int warp = 1; warp < Traits::kScoreWarps; ++warp) {
              float4 const part = *reinterpret_cast<float4 const *>(partial + warp * 32);
              maximum =
                  fmaxf(maximum, fmaxf(fmaxf(part.x, part.y), fmaxf(part.z, part.w)));
            }
            if constexpr (Postscale) {
              // Slots 0..3 are the unused queries (q = 0); their partials are
              // never stored.
              maximum = row < 4 ? 0.0F : maximum;
            }
            params.output_ptr[output_base + tile_page] = maximum;
          }
          partial += 4 * Traits::kPartialPageStride;
        }
        __syncwarp();
        if (lane_idx == 0) {
          storage.tile_done_warps[buffer] = 0;
          arrive(storage.tile_folded_barriers + buffer);
        }
      }
    }
    cudaTriggerProgrammaticLaunchCompletion();
  }
};

} // namespace q8kv4_indexer_sm107
