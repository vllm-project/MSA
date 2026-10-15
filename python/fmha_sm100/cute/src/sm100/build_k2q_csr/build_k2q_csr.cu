// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

// CUDA C++ q2k -> k2q CSR builder.
//
// Four kernels, no memsets. q-ascending order within each CSR row is preserved
// by partitioning the queries into tiles of kQPerTile consecutive queries of
// one KV head (one CTA per tile and head); warp w of a tile's CTA owns the
// tile's w-th kQPerWarp-query slice and walks it in query order, kTopK lanes
// per query, so a row's entries ordered by tile, then warp, then warp
// instruction, then lane are ordered by (query, top-k slot).
//
//   P: prepare   -- row map (round-robin packing of rows across batches), row
//                   coords, zeroed row_counts
//   C: count     -- per tile: row histogram -> tile_counts [H, tiles, rows],
//                   and row_counts (one global atomic per touched row and tile)
//   S: schedule  -- blocks [0, H): row_counts -> row_ptr (+ the attention work
//                   list, largest first); the other blocks: exclusive scan of
//                   tile_counts over the tiles of every (head, row), in place
//   W: scatter   -- per tile: per-warp row counts (packed int16, smem) become
//                   per-warp offsets inside the tile; each warp reserves its
//                   slots with lane-ordered shared-memory atomics and writes
//                   q_idx / qsplit_idx at row_ptr + tile offset + slot, and
//                   split_counts; the tiles also fill the q_idx tail with -1.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT(x)
// q2k may be a strided outer view (e.g. transposed GQA); the int4 loads need
// only topK contiguous, int4-aligned (mult-of-4) outer strides, and 16B base.
#define CHECK_Q2K_LAYOUT(x)                                                   \
    CHECK_CUDA(x); CHECK_INT(x);                                              \
    TORCH_CHECK((x).dim() == 3,                                              \
                #x " must be rank-3 [head_kv, total_q, topK]");              \
    TORCH_CHECK((x).stride(2) == 1,                                          \
                #x " innermost (topK) dim must be contiguous (stride 1)");   \
    TORCH_CHECK((x).stride(0) % 4 == 0 && (x).stride(1) % 4 == 0,            \
                #x " head/seq strides must be int4-aligned (multiple of 4)"); \
    TORCH_CHECK((reinterpret_cast<uintptr_t>((x).data_ptr<int>()) % 16) == 0, \
                #x " base pointer must be 16-byte aligned")
#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t err__ = (expr);                                          \
        TORCH_CHECK(err__ == cudaSuccess, #expr " failed: ",                 \
                    cudaGetErrorString(err__));                              \
    } while (0)

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ void advance_batch_only(
    int const* __restrict__ cu_q, int B, int q_abs, int& bi)
{
    while (bi < B && cu_q[bi + 1] <= q_abs) ++bi;
}

// Atomic increment of a 16-bit half within a 32-bit SMEM word; returns the
// OLD 16-bit value (slot). Per-warp count must stay < 32768 so the low
// half does not carry into the high half.
//   base_int32 : int32 pointer; element i holds rows 2*i (low) and 2*i+1 (high).
__device__ __forceinline__ int atomic_inc_int16_packed(
    int* base_int32, int row)
{
    int idx = row >> 1;
    int shift = (row & 1) << 4;  // 0 or 16
    int delta = 1 << shift;
    int old = atomicAdd(&base_int32[idx], delta);
    return (old >> shift) & 0xFFFF;
}

// Read 16-bit half from packed int32 storage.
__device__ __forceinline__ int read_int16_packed(int const* base_int32, int row) {
    int v = base_int32[row >> 1];
    int shift = (row & 1) << 4;
    return (v >> shift) & 0xFFFF;
}

// Queries per tile (one CTA per tile and KV head), warps per tile CTA (fewer
// for very many rows: the scatter keeps per-warp row counts in smem), and the
// queries a warp holds in registers at a time.
constexpr int kQPerTile = 128;
constexpr int kMaxTileWarps = 8;
constexpr int kChunkQ = kQPerTile / kMaxTileWarps;

// Top-k entries of a warp's queries [q_begin, q_end) (at most kIters * kQPerIter
// of them), kQPerIter queries per iteration: lane l takes slot l % kTopK of
// query q_begin + u * kQPerIter + l / kTopK. row[u] is the entry's CSR row (-1
// for no query or an invalid / unmapped KV block), qloc[u] the query's index
// inside its batch.
template <int kTopK, int kIters>
__device__ __forceinline__ void load_warp_rows(
    int const* __restrict__ q2k,
    int const* __restrict__ cu_q,
    int const* __restrict__ row_map,
    int B, int total_rows, int max_kv_blocks,
    int q2k_head_stride, int q2k_q_stride,
    int h, int q_begin, int q_end, int lane,
    int (&row)[kIters], int (&qloc)[kIters])
{
    constexpr int kQPerIter = kWarpSize / kTopK;
    int const q_in_iter = lane / kTopK;
    int const slot_in_q = lane % kTopK;
    int const* head_q2k = q2k + (size_t)h * q2k_head_stride;
    int kvb[kIters];
    int batch[kIters];
    int bi = 0;
    #pragma unroll
    for (int u = 0; u < kIters; ++u) {
        int const q = q_begin + u * kQPerIter + q_in_iter;
        kvb[u] = -1;
        qloc[u] = 0;
        batch[u] = 0;
        if (q < q_end) {
            advance_batch_only(cu_q, B, q, bi);
            qloc[u] = q - cu_q[bi];
            batch[u] = bi;
            kvb[u] = head_q2k[(size_t)q * q2k_q_stride + slot_in_q];
        }
    }
    #pragma unroll
    for (int u = 0; u < kIters; ++u) {
        int r = -1;
        if (kvb[u] >= 0 && kvb[u] < max_kv_blocks) {
            r = row_map[(size_t)batch[u] * max_kv_blocks + kvb[u]];
        }
        row[u] = (r >= 0 && r < total_rows) ? r : -1;
    }
}

// ---------------------------------------------------------------------------
// P: row map (rows of all batches packed round-robin by KV block level), row
// coords (batch, level) and zeroed row_counts. One thread per level.
// ---------------------------------------------------------------------------
template <int kBlockK>
__global__ void k2q_prepare_kernel(
    int const* __restrict__ cu_k,
    int* __restrict__ row_map,
    int* __restrict__ row_coords,
    int* __restrict__ row_counts,
    int B,
    int max_kv_blocks,
    int num_row_counts)
{
    int const first = blockIdx.x * blockDim.x + threadIdx.x;
    int const stride = gridDim.x * blockDim.x;
    for (int i = first; i < num_row_counts; i += stride) row_counts[i] = 0;
    for (int level = first; level < max_kv_blocks; level += stride) {
        int rows_before = 0;
        for (int b = 0; b < B; ++b) {
            int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
            rows_before += (rb < level ? rb : level);
        }
        int active_before = 0;
        for (int b = 0; b < B; ++b) {
            int rb = (cu_k[b + 1] - cu_k[b] + kBlockK - 1) / kBlockK;
            if (rb > level) {
                int row_linear = rows_before + active_before;
                row_map[(size_t)b * max_kv_blocks + level] = row_linear;
                if (row_coords != nullptr) {
                    row_coords[(size_t)row_linear * 2] = b;
                    row_coords[(size_t)row_linear * 2 + 1] = level;
                }
                ++active_before;
            } else {
                row_map[(size_t)b * max_kv_blocks + level] = -1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// C: per (tile, head) CTA, the tile's entries per row (packed int16 counts in
// smem: a tile has < 32768 entries per row) -> tile_counts[h][tile][*] and
// row_counts[h][*].
// ---------------------------------------------------------------------------
template <int kTopK, int kWarps>
__global__ void __launch_bounds__(kWarps * kWarpSize) k2q_tile_count_kernel(
    int const* __restrict__ q2k,
    int const* __restrict__ cu_q,
    int const* __restrict__ row_map,
    int* __restrict__ row_counts,
    int* __restrict__ tile_counts,
    int B, int S_Q,
    int total_rows, int max_kv_blocks, int num_tiles,
    int q2k_head_stride, int q2k_q_stride)
{
    constexpr int kThreads = kWarps * kWarpSize;
    constexpr int kQPerWarp = kQPerTile / kWarps;
    constexpr int kIters = kChunkQ * kTopK / kWarpSize;
    extern __shared__ int smem_tile_count[];
    int const tile = blockIdx.x;
    int const h = blockIdx.y;
    int const tid = threadIdx.x;
    int const words = (total_rows + 1) >> 1;
    for (int i = tid; i < words; i += kThreads) smem_tile_count[i] = 0;
    __syncthreads();

    int const q_begin = tile * kQPerTile + (tid / kWarpSize) * kQPerWarp;
    int const q_end = min(q_begin + kQPerWarp, S_Q);
    for (int chunk = q_begin; chunk < q_end; chunk += kChunkQ) {
        int row[kIters];
        int qloc[kIters];
        load_warp_rows<kTopK, kIters>(q2k, cu_q, row_map, B, total_rows, max_kv_blocks,
                                      q2k_head_stride, q2k_q_stride, h, chunk,
                                      min(chunk + kChunkQ, q_end), tid % kWarpSize, row, qloc);
        #pragma unroll
        for (int u = 0; u < kIters; ++u) {
            if (row[u] >= 0) atomic_inc_int16_packed(smem_tile_count, row[u]);
        }
    }
    __syncthreads();

    int* tile_row = tile_counts + ((size_t)h * num_tiles + tile) * total_rows;
    int* head_counts = row_counts + (size_t)h * total_rows;
    for (int i = tid; i < total_rows; i += kThreads) {
        int const count = read_int16_packed(smem_tile_count, i);
        tile_row[i] = count;
        if (count > 0) atomicAdd(&head_counts[i], count);
    }
}

// ---------------------------------------------------------------------------
// W: per (tile, head) CTA. Each warp counts its entries per row (packed int16
// in its own smem slice), the CTA turns the counts into per-warp exclusive
// offsets inside the tile, and each warp walks its entries again in the same
// order, reserving slots with lane-ordered smem atomics.
// CORRECTNESS NOTE: relies on lane-ordered SMEM atomicAdd return values
// within a single warp instruction (verified on SM100; tests pass).
// ---------------------------------------------------------------------------
template <int kTopK, int kWarps>
__global__ void __launch_bounds__(kWarps * kWarpSize) k2q_tile_scatter_kernel(
    int const* __restrict__ q2k,
    int const* __restrict__ cu_q,
    int const* __restrict__ row_map,
    int const* __restrict__ row_ptr,
    int const* __restrict__ tile_prefix,
    int* __restrict__ q_idx,
    int* __restrict__ qsplit_idx,
    int* __restrict__ split_counts,
    int H, int B, int S_Q,
    int total_rows, int max_kv_blocks, int num_tiles,
    int q2k_head_stride, int q2k_q_stride)
{
    constexpr int kThreads = kWarps * kWarpSize;
    constexpr int kQPerIter = kWarpSize / kTopK;
    constexpr int kQPerWarp = kQPerTile / kWarps;
    constexpr int kIters = kChunkQ / kQPerIter;
    // One chunk: its rows stay in registers between the passes; else reloaded.
    constexpr bool kOneChunk = kQPerWarp == kChunkQ;
    extern __shared__ int smem_warp_cursor[];
    int const tile = blockIdx.x;
    int const h = blockIdx.y;
    int const tid = threadIdx.x;
    int const warp = tid / kWarpSize;
    int const lane = tid % kWarpSize;
    int const words = (total_rows + 1) >> 1;
    int* my_cursor = smem_warp_cursor + warp * words;
    for (int i = lane; i < words; i += kWarpSize) my_cursor[i] = 0;
    __syncwarp();

    int const q_begin = tile * kQPerTile + warp * kQPerWarp;
    int const q_end = min(q_begin + kQPerWarp, S_Q);
    int row[kIters];
    int qloc[kIters];
    for (int chunk = q_begin; chunk < q_end; chunk += kChunkQ) {
        load_warp_rows<kTopK, kIters>(q2k, cu_q, row_map, B, total_rows, max_kv_blocks,
                                      q2k_head_stride, q2k_q_stride, h, chunk,
                                      min(chunk + kChunkQ, q_end), lane, row, qloc);
        #pragma unroll
        for (int u = 0; u < kIters; ++u) {
            if (row[u] >= 0) atomic_inc_int16_packed(my_cursor, row[u]);
        }
    }
    __syncthreads();

    // Per row, the warps' counts -> their exclusive offsets inside the tile. A
    // word packs two rows' int16 counts, which add without carrying.
    for (int i = tid; i < words; i += kThreads) {
        int running = 0;
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            int const count = smem_warp_cursor[w * words + i];
            smem_warp_cursor[w * words + i] = running;
            running += count;
        }
    }
    __syncthreads();

    int const* head_row_ptr = row_ptr + (size_t)h * (total_rows + 1);
    int const* tile_base = tile_prefix + ((size_t)h * num_tiles + tile) * total_rows;
    int* head_q_idx = q_idx + (size_t)h * S_Q * kTopK;
    int* head_qsplit = qsplit_idx != nullptr ? qsplit_idx + (size_t)h * S_Q * kTopK : nullptr;
    int const q_in_iter = lane / kTopK;
    int const slot_in_q = lane % kTopK;
    unsigned int const group_mask = (kTopK == 32)
        ? 0xFFFFFFFFu
        : (((1u << kTopK) - 1u) << (q_in_iter * kTopK));
    unsigned int const lower_lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
    for (int chunk = q_begin; chunk < q_end; chunk += kChunkQ) {
        int const chunk_end = min(chunk + kChunkQ, q_end);
        if constexpr (!kOneChunk) {
            load_warp_rows<kTopK, kIters>(q2k, cu_q, row_map, B, total_rows, max_kv_blocks,
                                          q2k_head_stride, q2k_q_stride, h, chunk, chunk_end,
                                          lane, row, qloc);
        }
        int base[kIters];
        #pragma unroll
        for (int u = 0; u < kIters; ++u) {
            base[u] = row[u] >= 0 ? head_row_ptr[row[u]] + tile_base[row[u]] : 0;
        }
        #pragma unroll
        for (int u = 0; u < kIters; ++u) {
            bool const valid_edge = row[u] >= 0;
            unsigned int const valid_mask = __ballot_sync(0xFFFFFFFFu, valid_edge);
            int const split_slot = __popc(valid_mask & group_mask & lower_lane_mask);
            int const q = chunk + u * kQPerIter + q_in_iter;
            if (split_counts != nullptr && slot_in_q == 0 && q < chunk_end) {
                split_counts[(size_t)q * H + h] = __popc(valid_mask & group_mask);
            }
            if (valid_edge) {
                int const out_pos = base[u] + atomic_inc_int16_packed(my_cursor, row[u]);
                head_q_idx[out_pos] = qloc[u];
                if (head_qsplit != nullptr) {
                    head_qsplit[out_pos] = qloc[u] | ((split_slot & 0xFF) << 24);
                }
            }
        }
    }

    // q_idx past the head's last entry is -1; the head's tiles split the tail.
    int64_t const nnz = (int64_t)S_Q * kTopK;
    int64_t const tail_begin = head_row_ptr[total_rows];
    int64_t const per_tile = (nnz - tail_begin + num_tiles - 1) / num_tiles;
    int64_t const begin = tail_begin + (int64_t)tile * per_tile;
    int64_t const end = min(begin + per_tile, nnz);
    for (int64_t i = begin + tid; i < end; i += kThreads) head_q_idx[i] = -1;
}

// ---------------------------------------------------------------------------
// S: schedule. Blocks [0, H): one block per head, row_counts -> row_ptr.
//
// With a schedule, each such block also emits its head's attention work items:
// one per (row, chunk of <= target_q_per_cta q), in a list ordered largest
// q_count first (LPT). The attention kernels run work item i on CTA i and are
// not persistent, so the list order is the launch order: appending items with
// atomicAdd left the largest ones at random positions and the grid waited for
// them at the end. Every block histograms the item sizes of all heads
// (q_count >> lpt_shift buckets), and separately those of the heads before
// its own, which gives each bucket's offset and the block's slice of it with
// no inter-block communication: buckets descend by size, then ascend by head.
// The order inside a slice is not deterministic. No output depends on the
// order: each item writes its own partial slots. The list's unused capacity
// is zeroed.
//
// Blocks [H, ...): one thread per (head, row), the exclusive scan of
// tile_counts over the tiles, in place (-> each tile's offset in the row).
// ---------------------------------------------------------------------------
constexpr int kLptBuckets = 2048;

// Adds the work items of one (head, row) with row_count > 0 q to bucket_counts.
__device__ __forceinline__ void lpt_count_items(
    int* bucket_counts, int row_count, int target_q_per_cta, int lpt_shift)
{
    if (row_count <= target_q_per_cta) {  // one item, the common case
        atomicAdd(&bucket_counts[(row_count - 1) >> lpt_shift], 1);
        return;
    }
    int full_chunks = row_count / target_q_per_cta;
    int last_chunk = row_count - full_chunks * target_q_per_cta;
    atomicAdd(&bucket_counts[(target_q_per_cta - 1) >> lpt_shift], full_chunks);
    if (last_chunk > 0) atomicAdd(&bucket_counts[(last_chunk - 1) >> lpt_shift], 1);
}

// Exclusive scan of one value per thread across the block (warp shuffles, two
// barriers); scratch holds >= kThreads / 32 + 1 ints. Returns the thread's
// exclusive prefix and sets total to the block sum.
template <int kThreads>
__device__ __forceinline__ int block_exclusive_scan(int value, int* scratch, int& total)
{
    constexpr int kWarpsPerBlock = kThreads / kWarpSize;
    static_assert(kThreads % kWarpSize == 0 && kWarpsPerBlock <= kWarpSize, "one warp scans the warp sums");
    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int x = value;
    #pragma unroll
    for (int off = 1; off < kWarpSize; off <<= 1) {
        int y = __shfl_up_sync(0xFFFFFFFFu, x, off);
        if (lane >= off) x += y;
    }
    if (lane == kWarpSize - 1) scratch[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = lane < kWarpsPerBlock ? scratch[lane] : 0;
        int t = w;
        #pragma unroll
        for (int off = 1; off < kWarpSize; off <<= 1) {
            int y = __shfl_up_sync(0xFFFFFFFFu, t, off);
            if (lane >= off) t += y;
        }
        if (lane < kWarpsPerBlock) scratch[lane] = t - w;
        if (lane == kWarpSize - 1) scratch[kWarpsPerBlock] = t;
    }
    __syncthreads();
    total = scratch[kWarpsPerBlock];
    return scratch[warp] + x - value;
}

template <int kThreads>
__device__ __forceinline__ void row_prefix_block(
    int h, int H,
    int const* __restrict__ row_counts,
    int* __restrict__ row_ptr,
    int const* __restrict__ row_coords,
    int* __restrict__ scheduler_metadata,
    int* __restrict__ work_count,
    int total_rows,
    int target_q_per_cta,
    int work_capacity,
    int lpt_shift)
{
    static_assert(kLptBuckets % kThreads == 0, "buckets must split evenly across threads");
    int tid = threadIdx.x;
    __shared__ int scan_buf[kWarpSize + 1];
    // Items per bucket: of every head (-> bucket offsets), and of the heads
    // before this one (-> this head's next slot in each bucket).
    __shared__ int bucket_all[kLptBuckets];
    __shared__ int bucket_slot[kLptBuckets];
    bool const emit_schedule = scheduler_metadata != nullptr && work_count != nullptr;
    if (emit_schedule) {
        for (int b = tid; b < kLptBuckets; b += kThreads) {
            bucket_all[b] = 0;
            bucket_slot[b] = 0;
        }
    }

    int const* head_counts = row_counts + (size_t)h * total_rows;
    int* head_rowptr = row_ptr + (size_t)h * (total_rows + 1);
    int chunk = (total_rows + kThreads - 1) / kThreads;
    int lo = min(tid * chunk, total_rows);
    int hi = min(lo + chunk, total_rows);

    int local_sum = 0;
    for (int i = lo; i < hi; ++i) local_sum += head_counts[i];
    int head_total = 0;
    int running = block_exclusive_scan<kThreads>(local_sum, scan_buf, head_total);
    if (tid == 0) head_rowptr[0] = 0;
    for (int i = lo; i < hi; ++i) {
        running += head_counts[i];
        head_rowptr[i + 1] = running;
    }
    if (!emit_schedule) return;

    // This head's work items: rows tid, tid + kThreads, ... The loads of the
    // first kRowsPrefetched land while the buckets are counted and scanned.
    constexpr int kRowsPrefetched = 2;
    int2 const* coords2 = reinterpret_cast<int2 const*>(row_coords);
    int own_counts[kRowsPrefetched];
    int2 own_coords[kRowsPrefetched];
    #pragma unroll
    for (int u = 0; u < kRowsPrefetched; ++u) {
        int row = tid + u * kThreads;
        own_counts[u] = row < total_rows ? head_counts[row] : 0;
        own_coords[u] = row < total_rows ? coords2[row] : make_int2(0, 0);
    }

    // (head, row) pairs of every head; a batch's loads are all in flight
    // before its first atomic.
    constexpr int kPairsPerBatch = 8;
    int const all_pairs = H * total_rows;
    int const pairs_before = h * total_rows;
    for (int first = tid; first < all_pairs; first += kThreads * kPairsPerBatch) {
        int pair_counts[kPairsPerBatch];
        #pragma unroll
        for (int u = 0; u < kPairsPerBatch; ++u) {
            int pair = first + u * kThreads;
            pair_counts[u] = pair < all_pairs ? row_counts[pair] : 0;
        }
        #pragma unroll
        for (int u = 0; u < kPairsPerBatch; ++u) {
            if (pair_counts[u] > 0) {
                lpt_count_items(bucket_all, pair_counts[u], target_q_per_cta, lpt_shift);
                if (first + u * kThreads < pairs_before) {
                    lpt_count_items(bucket_slot, pair_counts[u], target_q_per_cta, lpt_shift);
                }
            }
        }
    }
    __syncthreads();

    // Exclusive scan of bucket_all in descending bucket order: thread t owns the
    // kPerThread buckets that follow those of thread t - 1.
    constexpr int kPerThread = kLptBuckets / kThreads;
    int counts[kPerThread];
    int count_sum = 0;
    #pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        counts[j] = bucket_all[kLptBuckets - 1 - (tid * kPerThread + j)];
        count_sum += counts[j];
    }
    int total_items = 0;
    int offset = block_exclusive_scan<kThreads>(count_sum, scan_buf, total_items);
    #pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        bucket_slot[kLptBuckets - 1 - (tid * kPerThread + j)] += offset;
        offset += counts[j];
    }
    if (h == 0 && tid == 0) *work_count = total_items;
    __syncthreads();

    auto place_row = [&](int row, int row_count, int2 coords) {
        int num_chunks = (row_count + target_q_per_cta - 1) / target_q_per_cta;
        for (int c = 0; c < num_chunks; ++c) {
            int q_begin = c * target_q_per_cta;
            int q_count = min(target_q_per_cta, row_count - q_begin);
            int work_idx = atomicAdd(&bucket_slot[(q_count - 1) >> lpt_shift], 1);
            if (work_idx < work_capacity) {
                int* meta = scheduler_metadata + (size_t)work_idx * 6;
                meta[0] = h;
                meta[1] = row;
                meta[2] = q_begin;
                meta[3] = q_count;
                meta[4] = coords.x;
                meta[5] = coords.y;
            }
        }
    };
    #pragma unroll
    for (int u = 0; u < kRowsPrefetched; ++u) {
        place_row(tid + u * kThreads, own_counts[u], own_coords[u]);
    }
    for (int row = tid + kRowsPrefetched * kThreads; row < total_rows; row += kThreads) {
        place_row(row, head_counts[row], coords2[row]);
    }

    // Unused capacity (never read) is zeroed, interleaved across the H blocks.
    int64_t const used = (int64_t)min(total_items, work_capacity) * 6;
    int64_t const capacity_ints = (int64_t)work_capacity * 6;
    for (int64_t i = used + (int64_t)h * kThreads + tid; i < capacity_ints; i += (int64_t)H * kThreads) {
        scheduler_metadata[i] = 0;
    }
}

template <int kThreads>
__global__ void __launch_bounds__(kThreads) k2q_schedule_kernel(
    int const* __restrict__ row_counts,
    int* __restrict__ row_ptr,
    int const* __restrict__ row_coords,
    int* __restrict__ scheduler_metadata,
    int* __restrict__ work_count,
    int* __restrict__ tile_counts,
    int H,
    int total_rows,
    int num_tiles,
    int target_q_per_cta,
    int work_capacity,
    int lpt_shift)
{
    if ((int)blockIdx.x < H) {
        row_prefix_block<kThreads>(blockIdx.x, H, row_counts, row_ptr, row_coords,
                                   scheduler_metadata, work_count, total_rows,
                                   target_q_per_cta, work_capacity, lpt_shift);
        return;
    }
    int const column = ((int)blockIdx.x - H) * kThreads + (int)threadIdx.x;
    if (column >= H * total_rows) return;
    int const h = column / total_rows;
    int const r = column - h * total_rows;
    int* tiles = tile_counts + (size_t)h * num_tiles * total_rows + r;
    constexpr int kBatch = 16;
    int running = 0;
    for (int t0 = 0; t0 < num_tiles; t0 += kBatch) {
        int counts[kBatch];
        #pragma unroll
        for (int k = 0; k < kBatch; ++k) {
            counts[k] = t0 + k < num_tiles ? tiles[(size_t)(t0 + k) * total_rows] : 0;
        }
        #pragma unroll
        for (int k = 0; k < kBatch; ++k) {
            if (t0 + k < num_tiles) {
                tiles[(size_t)(t0 + k) * total_rows] = running;
                running += counts[k];
            }
        }
    }
}

}  // anonymous namespace

// ===========================================================================
// Host orchestration
// ===========================================================================

template <int kTopK, int kBlockK>
static void launch_pipeline(
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    int total_rows,
    int max_kv_blocks,
    uintptr_t stream_ptr,
    torch::Tensor scheduler_metadata = torch::Tensor(),
    torch::Tensor work_count = torch::Tensor(),
    torch::Tensor qsplit_idx = torch::Tensor(),
    torch::Tensor split_counts = torch::Tensor(),
    int target_q_per_cta = 1,
    int work_capacity = 0,
    int max_seqlen_q = 0)
{
    (void)max_seqlen_q;
    int H = (int)q2k.size(0);
    int S_Q = (int)q2k.size(1);
    int topK = (int)q2k.size(2);
    TORCH_CHECK(topK == kTopK, "topK runtime != template kTopK");
    // Head/seq strides for strided q2k (contiguous => S_Q*kTopK, kTopK).
    int q2k_head_stride = (int)q2k.stride(0);
    int q2k_q_stride = (int)q2k.stride(1);
    int B = (int)cu_q.size(0) - 1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    bool emit_schedule = scheduler_metadata.defined();
    // Work-list size buckets: q_count >> lpt_shift < kLptBuckets.
    int lpt_shift = 0;
    while (((target_q_per_cta - 1) >> lpt_shift) >= kLptBuckets) ++lpt_shift;
    int* scheduler_metadata_ptr = emit_schedule ? scheduler_metadata.data_ptr<int>() : nullptr;
    int* work_count_ptr = emit_schedule ? work_count.data_ptr<int>() : nullptr;
    int* qsplit_idx_ptr = emit_schedule ? qsplit_idx.data_ptr<int>() : nullptr;
    int* split_counts_ptr = emit_schedule ? split_counts.data_ptr<int>() : nullptr;

    // Tile CTAs: the scatter keeps kWarps packed int16 counts per row in smem;
    // fewer warps (longer slices) for very many rows.
    int dev = q2k.get_device();
    int max_smem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    size_t const words_bytes = (size_t)((total_rows + 1) >> 1) * sizeof(int);
    int tile_warps = kMaxTileWarps;
    while (tile_warps > 1 && (size_t)tile_warps * words_bytes > (size_t)max_smem) tile_warps >>= 1;
    TORCH_CHECK((size_t)tile_warps * words_bytes <= (size_t)max_smem,
                "build_k2q_csr: too many rows (", total_rows, ") for the shared-memory row counts");
    int num_tiles = (S_Q + kQPerTile - 1) / kQPerTile;

    // One int32 workspace: row_coords [R, 2] (first: read as int2) | row_counts [H, R] |
    // tile_counts [H, tiles, R] | row_map [B, max_kv_blocks].
    size_t const n_row_coords = emit_schedule ? (size_t)total_rows * 2 : 0;
    size_t const n_row_counts = (size_t)H * total_rows;
    size_t const n_tile_counts = (size_t)H * num_tiles * total_rows;
    size_t const n_row_map = (size_t)B * max_kv_blocks;
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(q2k.device());
    auto workspace = torch::empty(
        {(int64_t)(n_row_coords + n_row_counts + n_tile_counts + n_row_map)}, opts);
    int* row_coords = emit_schedule ? workspace.data_ptr<int>() : nullptr;
    int* row_counts = workspace.data_ptr<int>() + n_row_coords;
    int* tile_counts = row_counts + n_row_counts;
    int* row_map = tile_counts + n_tile_counts;

    constexpr int kPrepareThreads = 256;
    int prepare_items = std::max(max_kv_blocks, (int)n_row_counts);
    int prepare_blocks = std::max(1, std::min((prepare_items + kPrepareThreads - 1) / kPrepareThreads, 1024));
    k2q_prepare_kernel<kBlockK><<<prepare_blocks, kPrepareThreads, 0, stream>>>(
        cu_k.data_ptr<int>(), row_map, row_coords, row_counts, B, max_kv_blocks, (int)n_row_counts);

    auto launch_tiles = [&](auto warps_const) {
        constexpr int W = decltype(warps_const)::value;
        auto count_fn = k2q_tile_count_kernel<kTopK, W>;
        auto scatter_fn = k2q_tile_scatter_kernel<kTopK, W>;
        size_t const count_smem = words_bytes;
        size_t const scatter_smem = W * words_bytes;
        if (count_smem > 48 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(
                count_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)count_smem));
        }
        if (scatter_smem > 48 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(
                scatter_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)scatter_smem));
        }
        dim3 const grid(num_tiles, H);
        count_fn<<<grid, W * kWarpSize, count_smem, stream>>>(
            q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map, row_counts, tile_counts,
            B, S_Q, total_rows, max_kv_blocks, num_tiles, q2k_head_stride, q2k_q_stride);

        constexpr int kScheduleThreads = 1024;
        int const scan_blocks = (H * total_rows + kScheduleThreads - 1) / kScheduleThreads;
        k2q_schedule_kernel<kScheduleThreads><<<H + scan_blocks, kScheduleThreads, 0, stream>>>(
            row_counts, row_ptr.data_ptr<int>(), row_coords, scheduler_metadata_ptr,
            work_count_ptr, tile_counts, H, total_rows, num_tiles, target_q_per_cta,
            work_capacity, lpt_shift);

        scatter_fn<<<grid, W * kWarpSize, scatter_smem, stream>>>(
            q2k.data_ptr<int>(), cu_q.data_ptr<int>(), row_map, row_ptr.data_ptr<int>(),
            tile_counts, q_idx.data_ptr<int>(), qsplit_idx_ptr, split_counts_ptr,
            H, B, S_Q, total_rows, max_kv_blocks, num_tiles, q2k_head_stride, q2k_q_stride);
    };

    if (tile_warps == 8) {
        launch_tiles(std::integral_constant<int, 8>{});
    } else if (tile_warps == 4) {
        launch_tiles(std::integral_constant<int, 4>{});
    } else if (tile_warps == 2) {
        launch_tiles(std::integral_constant<int, 2>{});
    } else {
        launch_tiles(std::integral_constant<int, 1>{});
    }
}

void run_build_k2q_csr(
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    int64_t topk,
    int64_t blk_kv,
    int64_t total_rows,
    int64_t max_kv_blocks,
    uintptr_t stream_ptr)
{
    CHECK_Q2K_LAYOUT(q2k);
    CHECK_INPUT(cu_q);
    CHECK_INPUT(cu_k);
    CHECK_INPUT(row_ptr);
    CHECK_INPUT(q_idx);
    TORCH_CHECK(blk_kv == 128, "build_k2q_csr only supports blk_kv == 128");
    int H = (int)q2k.size(0);
    int S_Q = (int)q2k.size(1);
    int tr = (int)total_rows;
    int mkv = (int)max_kv_blocks;
    TORCH_CHECK(tr >= 0 && mkv >= 0,
                "total_rows / max_kv_blocks must be non-negative");
    TORCH_CHECK(row_ptr.size(0) == H && row_ptr.size(1) == tr + 1,
                "row_ptr shape mismatch");
    TORCH_CHECK(q_idx.size(0) == H && q_idx.size(1) == (int64_t)S_Q * (int)topk,
                "q_idx shape mismatch");
    if (S_Q == 0 || tr == 0 || H == 0 || mkv == 0) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        CUDA_CHECK(cudaMemsetAsync(
            row_ptr.data_ptr<int>(), 0,
            (size_t)H * (tr + 1) * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(
            q_idx.data_ptr<int>(), 0xFF,
            (size_t)H * S_Q * (int)topk * sizeof(int), stream));
        return;
    }

    if (topk == 16) {
        launch_pipeline<16, 128>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
                                 stream_ptr);
    } else if (topk == 8) {
        launch_pipeline<8, 128>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
                                stream_ptr);
    } else if (topk == 32) {
        launch_pipeline<32, 128>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
                                 stream_ptr);
    } else if (topk == 4) {
        launch_pipeline<4, 128>(q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv,
                                stream_ptr);
    } else {
        TORCH_CHECK(false, "unsupported topK ", topk, " (expected 4, 8, 16, or 32)");
    }
}

void run_build_k2q_csr_with_schedule(
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    torch::Tensor scheduler_metadata,
    torch::Tensor work_count,
    torch::Tensor qsplit_idx,
    torch::Tensor split_counts,
    int64_t topk,
    int64_t blk_kv,
    int64_t total_rows,
    int64_t max_kv_blocks,
    int64_t target_q_per_cta,
    int64_t work_capacity,
    int64_t max_seqlen_q,
    uintptr_t stream_ptr)
{
    CHECK_Q2K_LAYOUT(q2k);
    CHECK_INPUT(cu_q);
    CHECK_INPUT(cu_k);
    CHECK_INPUT(row_ptr);
    CHECK_INPUT(q_idx);
    CHECK_INPUT(scheduler_metadata);
    CHECK_INPUT(work_count);
    CHECK_INPUT(qsplit_idx);
    CHECK_INPUT(split_counts);
    TORCH_CHECK(blk_kv == 128, "build_k2q_csr only supports blk_kv == 128");
    int H = (int)q2k.size(0);
    int S_Q = (int)q2k.size(1);
    int tr = (int)total_rows;
    int mkv = (int)max_kv_blocks;
    int target = (int)target_q_per_cta;
    int capacity = (int)work_capacity;
    int max_sq = (int)max_seqlen_q;
    TORCH_CHECK(tr >= 0 && mkv >= 0 && target > 0 && capacity > 0 && max_sq >= 0,
                "invalid schedule sizing arguments");
    TORCH_CHECK(row_ptr.size(0) == H && row_ptr.size(1) == tr + 1,
                "row_ptr shape mismatch");
    TORCH_CHECK(q_idx.size(0) == H && q_idx.size(1) == (int64_t)S_Q * (int)topk,
                "q_idx shape mismatch");
    TORCH_CHECK(qsplit_idx.sizes() == q_idx.sizes(), "qsplit_idx shape mismatch");
    TORCH_CHECK(scheduler_metadata.size(0) == capacity && scheduler_metadata.size(1) == 6,
                "scheduler_metadata shape mismatch");
    TORCH_CHECK(work_count.numel() == 1, "work_count must have one int32 element");
    TORCH_CHECK(split_counts.dim() == 2 && split_counts.size(0) == S_Q
                && split_counts.size(1) == H,
                "split_counts shape mismatch");
    if (S_Q == 0 || tr == 0 || H == 0 || mkv == 0) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        CUDA_CHECK(cudaMemsetAsync(
            row_ptr.data_ptr<int>(), 0,
            (size_t)H * (tr + 1) * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(
            q_idx.data_ptr<int>(), 0xFF,
            (size_t)H * S_Q * (int)topk * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(work_count.data_ptr<int>(), 0, sizeof(int), stream));
        if (split_counts.numel() > 0) {
            CUDA_CHECK(cudaMemsetAsync(
                split_counts.data_ptr<int>(), 0,
                (size_t)split_counts.numel() * sizeof(int), stream));
        }
        return;
    }

    if (topk == 16) {
        launch_pipeline<16, 128>(
            q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv, stream_ptr,
            scheduler_metadata, work_count, qsplit_idx, split_counts,
            target, capacity, max_sq);
    } else if (topk == 8) {
        launch_pipeline<8, 128>(
            q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv, stream_ptr,
            scheduler_metadata, work_count, qsplit_idx, split_counts,
            target, capacity, max_sq);
    } else if (topk == 32) {
        launch_pipeline<32, 128>(
            q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv, stream_ptr,
            scheduler_metadata, work_count, qsplit_idx, split_counts,
            target, capacity, max_sq);
    } else if (topk == 4) {
        launch_pipeline<4, 128>(
            q2k, cu_q, cu_k, row_ptr, q_idx, tr, mkv, stream_ptr,
            scheduler_metadata, work_count, qsplit_idx, split_counts,
            target, capacity, max_sq);
    } else {
        TORCH_CHECK(false, "unsupported topK ", topk, " (expected 4, 8, 16, or 32)");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_build_k2q_csr", &run_build_k2q_csr,
          "q2k -> k2q CSR build (sorted within row)",
          pybind11::arg("q2k"),
          pybind11::arg("cu_q"),
          pybind11::arg("cu_k"),
          pybind11::arg("row_ptr"),
          pybind11::arg("q_idx"),
          pybind11::arg("topk"),
          pybind11::arg("blk_kv"),
          pybind11::arg("total_rows"),
          pybind11::arg("max_kv_blocks"),
          pybind11::arg("stream_ptr"));
    m.def("run_build_k2q_csr_with_schedule", &run_build_k2q_csr_with_schedule,
          "q2k -> k2q CSR build with fused attention schedule metadata",
          pybind11::arg("q2k"),
          pybind11::arg("cu_q"),
          pybind11::arg("cu_k"),
          pybind11::arg("row_ptr"),
          pybind11::arg("q_idx"),
          pybind11::arg("scheduler_metadata"),
          pybind11::arg("work_count"),
          pybind11::arg("qsplit_idx"),
          pybind11::arg("split_counts"),
          pybind11::arg("topk"),
          pybind11::arg("blk_kv"),
          pybind11::arg("total_rows"),
          pybind11::arg("max_kv_blocks"),
          pybind11::arg("target_q_per_cta"),
          pybind11::arg("work_capacity"),
          pybind11::arg("max_seqlen_q"),
          pybind11::arg("stream_ptr"));
}
