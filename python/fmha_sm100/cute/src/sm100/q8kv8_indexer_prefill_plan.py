# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Device-side varlen task planning for the Q8KV8 prefill indexer."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cuda.bindings.driver as cuda

from src.sm100.q8kv8_indexer_prefill import Q8KV8PrefillIndexerSm100


class Q8KV8PrefillIndexerPlanReset:
    """Reset device-resident queue counters and error state."""

    num_buckets = Q8KV8PrefillIndexerSm100.num_task_buckets
    threads_per_cta = cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mTaskCounts: cute.Tensor,
        mPlanError: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(mTaskCounts, mPlanError).launch(
            grid=(1, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mTaskCounts: cute.Tensor,
        mPlanError: cute.Tensor,
    ) -> None:
        lane_idx = cute.arch.lane_idx()
        if lane_idx < cutlass.Int32(self.num_buckets):
            mTaskCounts[lane_idx] = cutlass.Int32(0)
        if lane_idx == cutlass.Int32(0):
            mPlanError[0] = cutlass.Int32(0)


class Q8KV8PrefillIndexerPlanBuild:
    """Expand device varlen metadata into page-range task buckets.

    Tasks tile Q rows ``token * num_heads + head``; a Q tile of 256 rows covers
    ``256 / num_heads`` tokens of one request. ``num_heads`` divides the tile,
    so every tile starts at a token boundary and the kernel can locate row
    ``r`` of a tile at token offset ``r // num_heads``.

    One task covers an (even, odd) pair of Q tiles of a request: the pair shares
    the odd tile's page range (the even tile's causal limit is at most one page
    shorter and is masked per row by the kernel). Descriptor word 4 packs
    ``(q_rows - 1) | (page_count - 1) << 8 | (q_rows_b - 1) << 16`` and a pair
    flag at bit 24; a request's last even tile without a partner is a
    single-tile task. Tasks go into 32 cost classes (buckets) that the score
    kernel's queue walks in descending order.
    """

    q_tile = Q8KV8PrefillIndexerSm100.q_tile
    page_size = Q8KV8PrefillIndexerSm100.k_tile
    num_buckets = Q8KV8PrefillIndexerSm100.num_task_buckets
    # Fixed per-task cost (descriptor, Q loads, pipeline refill) in units of one
    # 256-row tile x one page of MMA; it only orders the queue's claims.
    task_overhead_tile_pages = 21
    split_page_chunk = 64
    large_page_chunk = 128
    split_q_tile_threshold = 60
    descriptor_words = Q8KV8PrefillIndexerSm100.task_descriptor_words
    page_count_shift = Q8KV8PrefillIndexerSm100.task_page_count_shift
    threads_per_cta = cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mCuSeqlensQ: cute.Tensor,
        mSeqLens: cute.Tensor,
        mLengths: cute.Tensor,
        mTaskDescriptors: cute.Tensor,
        mTaskCounts: cute.Tensor,
        mPlanError: cute.Tensor,
        num_candidate_q_tiles: cutlass.Int32,
        task_capacity: cutlass.Int32,
        num_heads: cutlass.Int32,
        max_page_chunk: cutlass.Int32,
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(
            mCuSeqlensQ,
            mSeqLens,
            mLengths,
            mTaskDescriptors,
            mTaskCounts,
            mPlanError,
            task_capacity,
            num_heads,
            max_page_chunk,
        ).launch(
            grid=(num_candidate_q_tiles, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _cost_bucket(
        self,
        q_rows: cutlass.Int32,
        page_count: cutlass.Int32,
    ) -> cutlass.Int32:
        """Descending-cost claim order: MMA work in 256-row tile-pages (a partial tile costs a
        full one) plus the fixed per-task cost, eight tile-pages per class."""
        tiles = (q_rows + cutlass.Int32(self.q_tile - 1)) >> cutlass.Int32(8)
        bucket = (tiles * page_count + cutlass.Int32(self.task_overhead_tile_pages)) >> cutlass.Int32(3)
        if bucket > cutlass.Int32(self.num_buckets - 1):
            bucket = cutlass.Int32(self.num_buckets - 1)
        return bucket

    @cute.jit
    def _pack_extent(
        self,
        q_rows: cutlass.Int32,
        page_count: cutlass.Int32,
    ) -> cutlass.Int32:
        return (q_rows - cutlass.Int32(1)) | (
            (page_count - cutlass.Int32(1)) << cutlass.Int32(self.page_count_shift)
        )

    @cute.kernel
    def kernel(
        self,
        mCuSeqlensQ: cute.Tensor,
        mSeqLens: cute.Tensor,
        mLengths: cute.Tensor,
        mTaskDescriptors: cute.Tensor,
        mTaskCounts: cute.Tensor,
        mPlanError: cute.Tensor,
        task_capacity: cutlass.Int32,
        num_heads: cutlass.Int32,
        max_page_chunk: cutlass.Int32,
    ) -> None:
        lane_idx = cute.arch.lane_idx()
        candidate_idx, _, _ = cute.arch.block_idx()
        batch = cute.size(mCuSeqlensQ) - 1
        batch_idx = candidate_idx % batch
        q_tile_idx = candidate_idx // batch
        q_local_begin = q_tile_idx * cutlass.Int32(self.q_tile)

        # ``q_*`` values below count Q rows (token * num_heads + head).
        q_start = cutlass.Int32(0)
        seq_q = cutlass.Int32(0)
        seq_rows = cutlass.Int32(0)
        seq_k = cutlass.Int32(0)
        q_rows = cutlass.Int32(0)
        num_pages = cutlass.Int32(0)
        num_q_tiles = cutlass.Int32(0)
        if lane_idx == cutlass.Int32(0):
            token_start = mCuSeqlensQ[batch_idx]
            seq_q = mCuSeqlensQ[batch_idx + cutlass.Int32(1)] - token_start
            q_start = token_start * num_heads
            seq_rows = seq_q * num_heads
            seq_k = mSeqLens[batch_idx]
            if q_local_begin < seq_rows:
                q_rows = seq_rows - q_local_begin
                if q_rows > cutlass.Int32(self.q_tile):
                    q_rows = cutlass.Int32(self.q_tile)
                last_position = (
                    seq_k
                    - seq_q
                    + (q_local_begin + q_rows - cutlass.Int32(1)) // num_heads
                )
                num_pages = last_position // cutlass.Int32(self.page_size)
                if num_pages < cutlass.Int32(0):
                    num_pages = cutlass.Int32(0)
                num_q_tiles = cute.ceil_div(seq_rows, self.q_tile)

        q_start = cute.arch.shuffle_sync(q_start, 0)
        seq_q = cute.arch.shuffle_sync(seq_q, 0)
        seq_rows = cute.arch.shuffle_sync(seq_rows, 0)
        seq_k = cute.arch.shuffle_sync(seq_k, 0)
        q_rows = cute.arch.shuffle_sync(q_rows, 0)
        num_pages = cute.arch.shuffle_sync(num_pages, 0)
        num_q_tiles = cute.arch.shuffle_sync(num_q_tiles, 0)

        for row_iter in cutlass.range_constexpr(self.q_tile // cute.arch.WARP_SIZE):
            q_local = (
                q_local_begin
                + lane_idx
                + cutlass.Int32(row_iter * cute.arch.WARP_SIZE)
            )
            if q_local < seq_rows:
                query_position = seq_k - seq_q + q_local // num_heads
                local_block = query_position // cutlass.Int32(self.page_size)
                mLengths[q_start + q_local] = local_block + cutlass.Int32(1)

        # Odd tiles are carried by their even partner; an even tile without a partner stays single.
        emit_task = q_rows > cutlass.Int32(0)
        if (q_tile_idx & cutlass.Int32(1)) != cutlass.Int32(0):
            emit_task = cutlass.Boolean(False)
        rows_b = cutlass.Int32(0)
        has_b = cutlass.Int32(0)
        cost_rows = q_rows
        q_local_begin_b = q_local_begin + cutlass.Int32(self.q_tile)
        if emit_task and q_local_begin_b < seq_rows:
            has_b = cutlass.Int32(1)
            rows_b = seq_rows - q_local_begin_b
            if rows_b > cutlass.Int32(self.q_tile):
                rows_b = cutlass.Int32(self.q_tile)
            last_position_b = (
                seq_k
                - seq_q
                + (q_local_begin_b + rows_b - cutlass.Int32(1)) // num_heads
            )
            num_pages = last_position_b // cutlass.Int32(self.page_size)
            if num_pages < cutlass.Int32(0):
                num_pages = cutlass.Int32(0)
            cost_rows = q_rows + rows_b
        if lane_idx == cutlass.Int32(0) and emit_task:
            page_chunk = cutlass.Int32(self.split_page_chunk)
            if num_q_tiles >= cutlass.Int32(self.split_q_tile_threshold):
                page_chunk = cutlass.Int32(self.large_page_chunk)
            # Host-computed cap keeps the task count near the resident cluster
            # count when few Q tiles would otherwise leave most clusters idle.
            if page_chunk > max_page_chunk:
                page_chunk = max_page_chunk
            if page_chunk > num_pages:
                page_chunk = num_pages
            page_begin = cutlass.Int32(0)
            while page_begin < num_pages:
                page_count = num_pages - page_begin
                if page_count > page_chunk:
                    page_count = page_chunk
                bucket = self._cost_bucket(cost_rows, page_count)
                slot = cute.arch.atomic_add(
                    (mTaskCounts.iterator + bucket).llvm_ptr,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
                if slot < task_capacity:
                    mTaskDescriptors[bucket, slot, 0] = q_start + q_local_begin
                    mTaskDescriptors[bucket, slot, 1] = (
                        seq_k - seq_q + q_local_begin // num_heads
                    )
                    mTaskDescriptors[bucket, slot, 2] = page_begin
                    mTaskDescriptors[bucket, slot, 3] = batch_idx
                    extent = self._pack_extent(q_rows, page_count)
                    if has_b != cutlass.Int32(0):
                        extent = extent | ((rows_b - cutlass.Int32(1)) << cutlass.Int32(16)) | (
                            cutlass.Int32(1) << cutlass.Int32(24)
                        )
                    mTaskDescriptors[bucket, slot, 4] = extent
                else:
                    cute.arch.atomic_exch(
                        mPlanError.iterator.llvm_ptr,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                page_begin += page_count


class Q8KV8PrefillIndexerPlanFinalize:
    """Queue metadata: per-bucket queue offsets and the first wave of decoded descriptors.

    One warp; lane l owns bucket 31 - l (descending cost). Words [0, 32) of ``mSchedMeta`` are
    the exclusive prefix of the bucket counts in that lane order (the flat queue index of each
    bucket's first task), [32, 64) the counts, and from word 64 on the decoded descriptor
    (batch, Q row base, Q position, first page, rows A, pages, rows B, pair flag) of flat task
    c for every cluster c; pages = 0 marks an empty first-wave slot.
    """

    num_buckets = Q8KV8PrefillIndexerSm100.num_task_buckets
    page_count_shift = Q8KV8PrefillIndexerSm100.task_page_count_shift
    meta_header_words = Q8KV8PrefillIndexerSm100.sched_meta_header_words
    meta_task_words = Q8KV8PrefillIndexerSm100.sched_words
    threads_per_cta = cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mTaskDescriptors: cute.Tensor,
        mTaskCounts: cute.Tensor,
        mSchedMeta: cute.Tensor,
        num_clusters: cutlass.Int32,
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(mTaskDescriptors, mTaskCounts, mSchedMeta, num_clusters).launch(
            grid=(1, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mTaskDescriptors: cute.Tensor,
        mTaskCounts: cute.Tensor,
        mSchedMeta: cute.Tensor,
        num_clusters: cutlass.Int32,
    ) -> None:
        lane = cute.arch.lane_idx()
        cnt = mTaskCounts[cutlass.Int32(self.num_buckets - 1) - lane]
        incl = cnt
        for off in cutlass.range_constexpr(5):
            src = lane - cutlass.Int32(1 << off)
            if src < cutlass.Int32(0):
                src = cutlass.Int32(0)
            other = cute.arch.shuffle_sync(incl, src)
            if lane >= cutlass.Int32(1 << off):
                incl = incl + other
        excl = incl - cnt
        total = cute.arch.shuffle_sync(incl, cute.arch.WARP_SIZE - 1)
        mSchedMeta[lane] = excl
        mSchedMeta[cutlass.Int32(cute.arch.WARP_SIZE) + lane] = cnt
        smem = utils.SmemAllocator()
        sExcl = smem.allocate_tensor(cutlass.Int32, cute.make_layout(cute.arch.WARP_SIZE), byte_alignment=16)
        sCnt = smem.allocate_tensor(cutlass.Int32, cute.make_layout(cute.arch.WARP_SIZE), byte_alignment=16)
        sExcl[lane] = excl
        sCnt[lane] = cnt
        cute.arch.sync_warp()
        rounds = (num_clusters + cutlass.Int32(cute.arch.WARP_SIZE - 1)) // cutlass.Int32(cute.arch.WARP_SIZE)
        for round_idx in cutlass.range(rounds, unroll=1):
            c = lane + round_idx * cutlass.Int32(cute.arch.WARP_SIZE)
            if c < num_clusters:
                w0 = cutlass.Int32(0)
                w1 = cutlass.Int32(0)
                w2 = cutlass.Int32(0)
                w3 = cutlass.Int32(0)
                w4 = cutlass.Int32(0)
                w5 = cutlass.Int32(0)
                w6 = cutlass.Int32(0)
                w7 = cutlass.Int32(0)
                if c < total:
                    hit_lane = cutlass.Int32(0)
                    for i in cutlass.range_constexpr(cute.arch.WARP_SIZE):
                        if c >= sExcl[i] and c < sExcl[i] + sCnt[i]:
                            hit_lane = cutlass.Int32(i)
                    bucket = cutlass.Int32(self.num_buckets - 1) - hit_lane
                    slot = c - sExcl[hit_lane]
                    w1 = mTaskDescriptors[bucket, slot, 0]
                    w2 = mTaskDescriptors[bucket, slot, 1]
                    w3 = mTaskDescriptors[bucket, slot, 2]
                    w0 = mTaskDescriptors[bucket, slot, 3]
                    packed = mTaskDescriptors[bucket, slot, 4]
                    w4 = (packed & cutlass.Int32(0xFF)) + cutlass.Int32(1)
                    w5 = ((packed >> cutlass.Int32(self.page_count_shift)) & cutlass.Int32(0xFF)) + cutlass.Int32(1)
                    w6 = ((packed >> cutlass.Int32(16)) & cutlass.Int32(0xFF)) + cutlass.Int32(1)
                    w7 = (packed >> cutlass.Int32(24)) & cutlass.Int32(1)
                base = cutlass.Int32(self.meta_header_words) + c * cutlass.Int32(self.meta_task_words)
                mSchedMeta[base] = w0
                mSchedMeta[base + cutlass.Int32(1)] = w1
                mSchedMeta[base + cutlass.Int32(2)] = w2
                mSchedMeta[base + cutlass.Int32(3)] = w3
                mSchedMeta[base + cutlass.Int32(4)] = w4
                mSchedMeta[base + cutlass.Int32(5)] = w5
                mSchedMeta[base + cutlass.Int32(6)] = w6
                mSchedMeta[base + cutlass.Int32(7)] = w7


__all__ = [
    "Q8KV8PrefillIndexerPlanBuild",
    "Q8KV8PrefillIndexerPlanFinalize",
    "Q8KV8PrefillIndexerPlanReset",
]
