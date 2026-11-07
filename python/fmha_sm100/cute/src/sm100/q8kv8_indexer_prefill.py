# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""CuTe DSL Q8KV8 paged prefill indexer proxy scores for SM100/SM103/SM107."""

import enum

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cuda.bindings.driver as cuda


class NamedBarrierIndexerSm100(enum.IntEnum):
    """Kernel-local named barriers."""

    TmemPtr = enum.auto()
    Final = enum.auto()


class Q8KV8PrefillIndexerSm100:
    """Compute raw FP32 historical-page maxima with native FP8 UMMA.

    Q rows are ``token * num_heads + head``: all index heads share the single K head, so heads
    simply extend the Q (M) dimension of every task.

    A task is a pair of consecutive 256-row Q tiles of one request (an even tile and, when the
    request has one, its odd partner) times a chunk of that request's historical pages
    (``q8kv8_indexer_prefill_plan.py``). One persistent 2-CTA cluster runs per SM pair with
    twelve warps per CTA:

    * score warps 0-3 (group 0, tile A) and 8-11 (group 1, tile B); the Q loader (warp 4); the K
      loader (warp 5); one MMA warp per Q tile (warps 6 and 7).
    * The K ring holds 12 pages in six two-page slots shared by both MMA warps. Each MMA warp
      owns a two-stage accumulator ring in TMEM (four 128-column stages in all) that its score
      group consumes in both CTAs, so one group's stall never blocks the other tile's MMAs. An
      unpaired task keeps tile A on both MMA warps and alternates the K slots between them in
      groups of four, so every score group still writes eight consecutive pages.
    * SM107's K=64 FP8 instruction reads Q from TMEM: both tiles sit behind the accumulators
      (576 columns) and are copied from SMEM once per task. SM100/SM103 have 512 TMEM columns,
      all accumulators, and their K=32 instruction runs at full rate with both operands in SMEM:
      each Q tile keeps its SMEM stage until the task's last MMA, with two stages per tile so
      the next task's tiles can land meanwhile.
    * Each score warp reduces its 32 rows of every page to the row maxima (the reducing TMEM
      load on SM103/SM107, plain loads and FP32 CUDA Core maxima on SM100), stages them as
      8-page group tiles and writes each tile with one TMA bulk store it issues itself; groups
      cut by a row's causal limit go out as predicated register stores. Entries at or past a
      row's local page are never written.
    * Tasks are claimed from a dynamic queue over the planner's descriptors in descending cost
      order. Cluster c starts with task c, decoded by the plan; the leader CTA's Q-loader warp
      claims each later task with one atomic on the plan's work counter once the leader K
      loader is ``sched_lead_slots`` K slots from the end of the current task, resolves it to a
      descriptor and publishes the descriptor to both CTAs through a four-slot SMEM ring. The
      claim that returns the launch's last position resets the counter, so graph replays need
      no memset. A descriptor with zero pages ends the work of every warp.
    """

    supported_compute_capabilities = frozenset({(10, 0), (10, 3), (10, 7)})
    head_dim = 128
    q_tile = 256
    k_tile = 128
    cta_group_size = 2
    ctas_per_sm = 1
    # K pages resident per CTA pair, in slots of two pages (one full/empty barrier pair per slot).
    k_stages = 12
    k_pages_per_slot = 2
    # Accumulator stages in TMEM: two per MMA warp, k_tile columns each.
    acc_stages = 4
    # Warp roles: score group g owns the accumulator ring of MMA warp g.
    score_warps_per_group = 4
    score_group_warp_ids = ((0, 1, 2, 3), (8, 9, 10, 11))
    q_load_warp_id = 4
    k_load_warp_id = 5
    mma_warp_ids = (6, 7)
    threads_per_cta = 12 * cute.arch.WARP_SIZE
    # Register split of the 168 registers per thread the launch allocates: the eight score warps
    # grow to 192 while the four others drop to 64 (8 * 24 <= 4 * 104, so the increase is covered).
    num_regs_score = 192
    num_regs_other = 64
    # The planner sorts tasks into cost classes of eight tile-pages; the queue walks them in
    # descending order.
    num_task_buckets = 32
    # Descriptor ring between the scheduling warp and the task-walking warps of both CTAs.
    sched_ring_slots = 4
    sched_words = 8
    # The leader K loader triggers the scheduling of the next task this many single-tile K slots of
    # MMA work before the end of the current task's issue (a pair task's slot carries two tiles, so
    # it triggers at half that many slots; slot 0 for shorter tasks): the lookahead is bounded in
    # time, not in tasks, so the queue keeps its end balance while the descriptor still reaches
    # every consumer before it is needed.
    sched_lead_slots = 12
    # Plan metadata layout (see Q8KV8PrefillIndexerPlanFinalize): 32 bucket offsets, 32 counts,
    # then sched_words decoded words per cluster.
    sched_meta_header_words = 64
    # Scores go out in 8-page group tiles (one TMA store per 32 rows x 8 pages) assembled from
    # 4-page quads: one 16-byte row per lane.
    score_group_pages = 8
    score_quad_pages = 4
    # Task descriptor words: global Q row, Q position, first logical page, batch index, and the
    # packed extent (q_rows - 1) | (page_count - 1) << 8 | (q_rows_b - 1) << 16 | pair flag << 24.
    task_descriptor_words = 5
    task_page_count_shift = 8
    # Operand formats with a dedicated SM107 MMA instruction (K=64 per instruction). Any other
    # format, and every other architecture, goes through the generic F8F6F4 instruction at K=32.
    sm107_fp8_dtypes = (cutlass.Float8E4M3FN, cutlass.Float8E5M2)

    @classmethod
    def uses_sm107_fp8_mma(
        cls,
        compute_capability: tuple[int, int],
        io_dtype=cutlass.Float8E4M3FN,
    ) -> bool:
        return compute_capability == (10, 7) and io_dtype in cls.sm107_fp8_dtypes

    @classmethod
    def mma_inst_k(
        cls,
        compute_capability: tuple[int, int],
        io_dtype=cutlass.Float8E4M3FN,
    ) -> int:
        return 64 if cls.uses_sm107_fp8_mma(compute_capability, io_dtype) else 32

    def __init__(
        self,
        *,
        compute_capability: tuple[int, int],
        num_persistent_clusters: int,
    ) -> None:
        if compute_capability not in self.supported_compute_capabilities:
            raise ValueError(
                "compute_capability must be SM100, SM103 or SM107, "
                f"got SM{compute_capability[0]}{compute_capability[1]}"
            )
        if num_persistent_clusters <= 0:
            raise ValueError("num_persistent_clusters must be positive")
        # A pair task triggers the next claim sched_lead_slots // 2 K slots before its end; that must be at
        # least one slot or the trigger never fires.
        if self.sched_lead_slots // 2 < 1:
            raise ValueError("sched_lead_slots must be at least 2")
        self.io_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32
        self.compute_capability = compute_capability
        self.num_persistent_clusters = num_persistent_clusters
        self.use_sm107_fp8_mma = self.uses_sm107_fp8_mma(compute_capability, self.io_dtype)
        # SM103 and SM107 reduce each accumulator row inside the TMEM load.
        self.use_tmem_load_reduce = compute_capability in ((10, 3), (10, 7))
        self.k_slots = self.k_stages // self.k_pages_per_slot
        self.cluster_shape_mnk = (2, 1, 1)
        self.mma_tiler_mnk = (self.q_tile, self.k_tile, self.head_dim)
        self.mma_inst_shape_mnk = (
            self.q_tile,
            self.k_tile,
            self.mma_inst_k(compute_capability, self.io_dtype),
        )
        # Where the MMA reads its A operand (the Q tile): TMEM behind the accumulator stages on
        # SM107, SMEM elsewhere (see the class docstring).
        self.a_from_tmem = self.use_sm107_fp8_mma
        self.q_tile_stages = 1 if self.a_from_tmem else 2
        self.q_stages = 2 * self.q_tile_stages
        # An FP8 Q tile is 128 bytes per TMEM lane: head_dim // 4 columns.
        self.tmem_q_cols = self.head_dim // 4
        self.tmem_alloc_cols = self.acc_stages * self.k_tile
        if self.a_from_tmem:
            self.tmem_alloc_cols += 2 * self.tmem_q_cols
        self.num_score_warps = 2 * self.score_warps_per_group
        # Descriptor ring consumers per cluster: both CTAs' score warps and K loaders, the leader's
        # MMA warps and the peer's Q loader (the leader's Q loader feeds the ring).
        self.sched_consumer_count = (
            self.cta_group_size * self.num_score_warps
            + self.cta_group_size
            + len(self.mma_warp_ids)
            + 1
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mPageTable: cute.Tensor,
        mScores: cute.Tensor,
        mTaskDescriptors: cute.Tensor,
        mWorkCounter: cute.Tensor,
        mSchedMeta: cute.Tensor,
        num_heads: cutlass.Int32,
        stream: cuda.CUstream = None,
    ):
        """Build descriptors and launch the 2-CTA kernel."""
        if cutlass.const_expr(self.use_sm107_fp8_mma):
            op = tcgen05.SM107MmaFP8Op(
                self.io_dtype,
                self.io_dtype,
                self.acc_dtype,
                self.mma_inst_shape_mnk,
                tcgen05.CtaGroup.TWO,
                tcgen05.OperandSource.TMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        else:
            op = tcgen05.MmaF8F6F4Op(
                self.io_dtype,
                self.io_dtype,
                self.acc_dtype,
                self.mma_inst_shape_mnk,
                tcgen05.CtaGroup.TWO,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        tiled_mma = cute.make_tiled_mma(op)
        q_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler_mnk,
            self.io_dtype,
            self.q_stages,
        )
        k_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler_mnk,
            self.io_dtype,
            self.k_stages,
        )
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.select(mQ.layout, mode=[0, 2, 1]),
        )[None, None, 0]
        mK = cute.make_tensor(
            mK.iterator,
            cute.select(mK.layout, mode=[1, 2, 0]),
        )
        cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        cta_layout_vmnk = cute.tiled_divide(
            cta_layout_mnk,
            (tiled_mma.thr_id,),
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        q_tma_atom, q_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(q_smem_layout, mode=[0, 1, 2]),
            self.mma_tiler_mnk,
            tiled_mma,
            cta_layout_vmnk.shape,
        )
        k_tma_atom, k_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(k_smem_layout, mode=[0, 1, 2]),
            self.mma_tiler_mnk,
            tiled_mma,
            cta_layout_vmnk.shape,
        )
        # TMA bulk-store atoms for the score warps' staging tiles: dense 32-row x 4-page quads
        # (partial groups) and SWIZZLE_32B 32-row x 8-page group tiles.
        quad_box = (cute.arch.WARP_SIZE, self.score_quad_pages)
        quad_tma_atom, quad_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mScores,
            cute.make_layout(quad_box, stride=(self.score_quad_pages, 1)),
            quad_box,
        )
        group_box = (cute.arch.WARP_SIZE, self.score_group_pages)
        group_tma_atom, group_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mScores,
            cute.make_composed_layout(
                cute.make_swizzle(1, 4, 3),
                0,
                cute.make_layout(group_box, stride=(self.score_group_pages, 1)),
            ),
            group_box,
        )

        @cute.struct
        class SharedStorage:
            q_a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_tile_stages * 2]
            q_b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_tile_stages * 2]
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.k_slots * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.acc_stages * 2]
            sched_full_mbar: cute.struct.MemRange[cutlass.Int64, self.sched_ring_slots]
            sched_empty_mbar: cute.struct.MemRange[cutlass.Int64, self.sched_ring_slots]
            sched_go_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buffer: cutlass.Int32
            sched_ring: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.sched_ring_slots * self.sched_words], 16
            ]

        self.shared_storage = SharedStorage
        launcher = self.kernel(
            tiled_mma,
            q_tma_atom,
            q_tma_tensor,
            k_tma_atom,
            k_tma_tensor,
            cta_layout_vmnk,
            q_smem_layout,
            k_smem_layout,
            mPageTable,
            mScores,
            mTaskDescriptors,
            mWorkCounter,
            mSchedMeta,
            quad_tma_atom,
            quad_tma_tensor,
            group_tma_atom,
            group_tma_tensor,
            num_heads,
        )
        # Dynamic SMEM: the storage struct, the score staging tiles (two quad slots and two group
        # slots per score warp, 128- and 1024-byte aligned) and the 128-byte aligned Q and K stages.
        fp32_bytes = 4
        quad_staging_bytes = (
            self.num_score_warps * 2 * 2 * cute.arch.WARP_SIZE * self.score_quad_pages * fp32_bytes
        )
        group_staging_bytes = (
            self.num_score_warps * 2 * cute.arch.WARP_SIZE * self.score_group_pages * fp32_bytes
        )
        smem_bytes = (
            self.shared_storage.size_in_bytes()
            + quad_staging_bytes
            + 128
            + group_staging_bytes
            + 1024
            + cute.size_in_bytes(self.io_dtype, q_smem_layout)
            + cute.size_in_bytes(self.io_dtype, k_smem_layout)
            + 2 * 128
        )
        launcher.launch(
            grid=(self.num_persistent_clusters * self.cta_group_size, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape_mnk,
            stream=stream,
            smem=smem_bytes,
            min_blocks_per_mp=self.ctas_per_sm,
        )

    @cute.jit
    def _sched_take(self, sSchedRing, sched_full_mbar, sched_empty_mbar, state, is_leader_cta):
        """Queue consumer: wait for the next published descriptor, read it, hand the slot back.

        Returns (batch, Q row base, Q position, first page, rows A, pages, rows B, pair flag);
        pages == 0 is the end-of-work sentinel. The caller advances ``state``."""
        cute.arch.mbarrier_wait(sched_full_mbar + state.index, state.phase)
        w0 = sSchedRing[(state.index, 0)]
        w1 = sSchedRing[(state.index, 1)]
        w2 = sSchedRing[(state.index, 2)]
        w3 = sSchedRing[(state.index, 3)]
        w4 = sSchedRing[(state.index, 4)]
        w5 = sSchedRing[(state.index, 5)]
        w6 = sSchedRing[(state.index, 6)]
        w7 = sSchedRing[(state.index, 7)]
        # The leader rewrites this slot (local stores, st.async into the peer) as soon as the release
        # below lands, so the loads above must have completed first. A CTA-scoped remote arrive does
        # not order them (the peer's next descriptor then tears into these reads), and a cluster-scoped
        # release drains the warp's outstanding global stores. Making the arrive control-dependent on
        # the loaded words (their OR is never all ones) holds it until the loads have returned.
        loaded = w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            if loaded != cutlass.Int32(-1):
                if is_leader_cta:
                    cute.arch.mbarrier_arrive(sched_empty_mbar + state.index)
                else:
                    cute.arch.mbarrier_arrive(sched_empty_mbar + state.index, peer_cta_rank_in_cluster=0)
        return w0, w1, w2, w3, w4, w5, w6, w7

    @cute.jit
    def _sched_claim(self, mWorkCounter):
        """Lane 0 claims the next queue position (un-broadcast)."""
        old = cutlass.Int32(0)
        if cute.arch.lane_idx() == cutlass.Int32(0):
            old = cute.arch.atomic_add(mWorkCounter.iterator.llvm_ptr, cutlass.Int32(1))
        return old

    @cute.jit
    def _sched_reset_if_last(self, mWorkCounter, claimed, claim_limit):
        """The claim that returned the launch's last position resets the counter for the next launch.

        Every launch performs exactly one claim per task, so the claim returning total - 1 is the
        launch's last atomic on the counter."""
        if cute.arch.lane_idx() == cutlass.Int32(0):
            if claimed == claim_limit:
                cute.arch.atomic_exch(mWorkCounter.iterator.llvm_ptr, cutlass.Int32(0))

    @cute.jit
    def _sched_publish(
        self,
        flat_idx,
        total_tasks,
        bucket_excl,
        bucket_cnt,
        mTaskDescriptors,
        sSchedRing,
        sched_ring_ptr,
        sched_full_mbar,
        sched_empty_mbar,
        state,
    ):
        """Scheduler: resolve a flat queue index to its descriptor and publish it to both CTAs.

        Lane l holds the count and exclusive prefix of bucket 31 - l (descending cost); a ballot
        finds the bucket whose range holds ``flat_idx``. An index past the end publishes the
        all-zero sentinel. The caller advances ``state``."""
        lane = cute.arch.lane_idx()
        valid = flat_idx < total_tasks
        hit = cutlass.Boolean(False)
        if valid:
            hit = (flat_idx >= bucket_excl) and (flat_idx < bucket_excl + bucket_cnt)
        ballot = cute.arch.vote_ballot_sync(hit)
        match_lane = cutlass.Int32(0)
        if valid:
            match_lane = cute.arch.popc(ballot - cutlass.Int32(1))
        bucket = cutlass.Int32(self.num_task_buckets - 1) - match_lane
        slot = flat_idx - cute.arch.shuffle_sync(bucket_excl, match_lane)
        w0 = cutlass.Int32(0)
        w1 = cutlass.Int32(0)
        w2 = cutlass.Int32(0)
        w3 = cutlass.Int32(0)
        w4 = cutlass.Int32(0)
        w5 = cutlass.Int32(0)
        w6 = cutlass.Int32(0)
        w7 = cutlass.Int32(0)
        if valid:
            if lane == cutlass.Int32(0):
                w1 = mTaskDescriptors[bucket, slot, 0]
                w2 = mTaskDescriptors[bucket, slot, 1]
                w3 = mTaskDescriptors[bucket, slot, 2]
                w0 = mTaskDescriptors[bucket, slot, 3]
                packed = mTaskDescriptors[bucket, slot, 4]
                w4 = (packed & cutlass.Int32(0xFF)) + cutlass.Int32(1)
                w5 = ((packed >> cutlass.Int32(self.task_page_count_shift)) & cutlass.Int32(0xFF)) + cutlass.Int32(1)
                w6 = ((packed >> cutlass.Int32(16)) & cutlass.Int32(0xFF)) + cutlass.Int32(1)
                w7 = (packed >> cutlass.Int32(24)) & cutlass.Int32(1)
        return self._sched_publish_words(
            w0, w1, w2, w3, w4, w5, w6, w7, sSchedRing, sched_ring_ptr, sched_full_mbar, sched_empty_mbar, state
        )

    @cute.jit
    def _sched_publish_words(
        self,
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        sSchedRing,
        sched_ring_ptr,
        sched_full_mbar,
        sched_empty_mbar,
        state,
    ):
        """Scheduler: write lane 0's eight descriptor words into the ring slot of both CTAs and
        return them warp-uniform. The caller advances ``state``.

        The local copy is plain stores and an arrive; the peer's copy arms the peer's full
        barrier for the slot's transaction bytes and lands through eight asynchronous stores."""
        lane = cute.arch.lane_idx()
        cute.arch.mbarrier_wait(sched_empty_mbar + state.index, state.phase)
        if lane == cutlass.Int32(0):
            sSchedRing[(state.index, 0)] = w0
            sSchedRing[(state.index, 1)] = w1
            sSchedRing[(state.index, 2)] = w2
            sSchedRing[(state.index, 3)] = w3
            sSchedRing[(state.index, 4)] = w4
            sSchedRing[(state.index, 5)] = w5
            sSchedRing[(state.index, 6)] = w6
            sSchedRing[(state.index, 7)] = w7
            cute.arch.mbarrier_arrive_and_expect_tx(
                sched_full_mbar + state.index, cutlass.Int32(self.sched_words * 4), peer_cta_rank_in_cluster=1
            )
            slot_base = state.index * cutlass.Int32(self.sched_words)
            cute.arch.store_async_dsmem(sched_ring_ptr + slot_base, w0, sched_full_mbar + state.index, 1)
            cute.arch.store_async_dsmem(sched_ring_ptr + (slot_base + cutlass.Int32(1)), w1, sched_full_mbar + state.index, 1)
            cute.arch.store_async_dsmem(sched_ring_ptr + (slot_base + cutlass.Int32(2)), w2, sched_full_mbar + state.index, 1)
            cute.arch.store_async_dsmem(sched_ring_ptr + (slot_base + cutlass.Int32(3)), w3, sched_full_mbar + state.index, 1)
            cute.arch.store_async_dsmem(sched_ring_ptr + (slot_base + cutlass.Int32(4)), w4, sched_full_mbar + state.index, 1)
            cute.arch.store_async_dsmem(sched_ring_ptr + (slot_base + cutlass.Int32(5)), w5, sched_full_mbar + state.index, 1)
            cute.arch.store_async_dsmem(sched_ring_ptr + (slot_base + cutlass.Int32(6)), w6, sched_full_mbar + state.index, 1)
            cute.arch.store_async_dsmem(sched_ring_ptr + (slot_base + cutlass.Int32(7)), w7, sched_full_mbar + state.index, 1)
            cute.arch.mbarrier_arrive(sched_full_mbar + state.index)
        w0 = cute.arch.shuffle_sync(w0, 0)
        w1 = cute.arch.shuffle_sync(w1, 0)
        w2 = cute.arch.shuffle_sync(w2, 0)
        w3 = cute.arch.shuffle_sync(w3, 0)
        w4 = cute.arch.shuffle_sync(w4, 0)
        w5 = cute.arch.shuffle_sync(w5, 0)
        w6 = cute.arch.shuffle_sync(w6, 0)
        w7 = cute.arch.shuffle_sync(w7, 0)
        return w0, w1, w2, w3, w4, w5, w6, w7

    @cute.jit
    def _octet_pages(self, pages: cutlass.Int32) -> cutlass.Int32:
        """Unpaired task: pages of one ring's 8-page octet inside a 16-page window remainder (0..8)."""
        n = pages
        if n < cutlass.Int32(0):
            n = cutlass.Int32(0)
        if n > cutlass.Int32(self.score_group_pages):
            n = cutlass.Int32(self.score_group_pages)
        return n

    @cute.jit
    def _reduce_accumulator_row(self, tTR_rAcc: cute.Tensor) -> cutlass.Float32:
        """Reduce one 128-token accumulator row with FP32 CUDA Core max."""
        block_max_0 = tTR_rAcc[0]
        block_max_1 = tTR_rAcc[1]
        block_max_2 = tTR_rAcc[2]
        block_max_3 = tTR_rAcc[3]
        for token_group in cutlass.range_constexpr(1, self.k_tile // 4):
            token_idx = token_group * 4
            block_max_0 = cute.arch.fmax(block_max_0, tTR_rAcc[token_idx])
            block_max_1 = cute.arch.fmax(block_max_1, tTR_rAcc[token_idx + 1])
            block_max_2 = cute.arch.fmax(block_max_2, tTR_rAcc[token_idx + 2])
            block_max_3 = cute.arch.fmax(block_max_3, tTR_rAcc[token_idx + 3])
        block_max_01 = cute.arch.fmax(block_max_0, block_max_1)
        block_max_23 = cute.arch.fmax(block_max_2, block_max_3)
        return cute.arch.fmax(block_max_01, block_max_23)

    @cute.jit
    def _score_page(
        self,
        acc_pipe,
        ring_base: cutlass.Int32,
        ring_state,
        tTR_tAcc_staged: cute.Tensor,
        tmem_load_reduce_atom,
        tmem_tiled_copy,
        tTR_rAcc: cute.Tensor,
        tTR_rBlockMax: cute.Tensor,
        prev_max: cutlass.Float32,
    ) -> cutlass.Float32:
        """Score warp, one page: wait for the accumulator stage, reduce its two 64-column halves to
        the per-row maximum, release the stage. Returns the row maximum; the caller advances
        ``ring_state`` and passes this page's maximum back as ``prev_max`` for the next page."""
        stage = ring_base + ring_state.index
        if cutlass.const_expr(not self.use_tmem_load_reduce):
            # Without the reducing TMEM load the row maximum is a 128-value CUDA Core chain that
            # ptxas would sink below the next page's loads, keeping two pages' fragments live
            # (spills). Feeding the previous maximum into this wait's stage index (a test that is
            # never true for finite FP8 dot products but cannot be folded) orders the chain before
            # the next page's wait and loads. The pin is declared warp-uniform (it is: every lane
            # computes 0) so the stage index, the barrier wait and the loop stay on the uniform
            # datapath.
            pin = cutlass.Int32(0)
            if prev_max > cutlass.Float32(3.0e38):
                pin = cutlass.Int32(1)
            stage = stage + cute.arch.make_warp_uniform(pin)
        acc_pipe.sync_object_full.wait(stage, ring_state.phase)
        tTR_tAcc = tTR_tAcc_staged[(None, None, None, None, stage)]
        if cutlass.const_expr(self.use_tmem_load_reduce):
            for chunk in cutlass.range_constexpr(2):
                cute.copy_atom_call(
                    tmem_load_reduce_atom,
                    tTR_tAcc[(None, chunk, 0, 0)],
                    (tTR_rAcc[(None, chunk, 0, 0)], tTR_rBlockMax[(None, chunk)]),
                )
        else:
            cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
        cute.arch.fence_view_async_tmem_load()
        with cute.arch.elect_one():
            acc_pipe.sync_object_empty.arrive(stage, acc_pipe.consumer_mask, acc_pipe.cta_group)
        block_max = cutlass.Float32(0.0)
        if cutlass.const_expr(self.use_tmem_load_reduce):
            block_max = cute.arch.fmax(tTR_rBlockMax[(0, 0)], tTR_rBlockMax[(0, 1)])
        else:
            block_max = self._reduce_accumulator_row(tTR_rAcc)
        return block_max

    @cute.jit
    def _store_group_tma(
        self,
        sScoreGroups: cute.Tensor,
        group_tma_atom: cute.CopyAtom,
        group_tma_tensor: cute.Tensor,
        buf_warp: cutlass.Int32,
        slot: cutlass.Int32,
        group_col: cutlass.Int32,
        q_row_base: cutlass.Int32,
    ):
        """Fence this warp's staging writes for the async proxy and issue the TMA store of the
        32 x 8 group tile in ``slot`` to page column ``group_col``."""
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        gTile = cute.local_tile(
            cute.domain_offset((q_row_base, group_col), group_tma_tensor),
            (cute.arch.WARP_SIZE, self.score_group_pages),
            (0, None),
        )
        tSsG, tSgG = cpasync.tma_partition(
            group_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sScoreGroups[(None, None, None, buf_warp)], 0, 2),
            cute.group_modes(gTile, 0, 2),
        )
        with cute.arch.elect_one():
            cute.copy(group_tma_atom, tSsG[(None, slot)], tSgG[(None, 0)])
            cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def _store_quad(
        self,
        quad_max: cute.Tensor,
        sScoreQuads: cute.Tensor,
        quad_tma_atom: cute.CopyAtom,
        quad_tma_tensor: cute.Tensor,
        sScoreGroups: cute.Tensor,
        group_ok: cutlass.Boolean,
        mScores: cute.Tensor,
        buf_warp: cutlass.Int32,
        slot: cutlass.Int32,
        quad: cutlass.Int32,
        quad_len: cutlass.Int32,
        group_col: cutlass.Int32,
        warp_min_limit: cutlass.Int32,
        my_limit: cutlass.Int32,
        q_row_base: cutlass.Int32,
        row_global: cutlass.Int32,
    ):
        """Write four consecutive page maxima of this warp's 32 rows (quad ``quad`` of the 8-page
        group whose first page column is ``group_col``).

        A full group below every row's causal limit at a 16-byte aligned column (``group_ok``) is
        staged quad by quad into the 32 x 8 group tile and written by one TMA bulk store the
        caller issues after the group. Otherwise a full, aligned quad below every row's limit is
        staged as a dense 32 x 4 tile and stored by TMA on its own; any other quad goes out as
        predicated stores straight from the registers."""
        col = group_col + cutlass.Int32(self.score_quad_pages) * quad
        if group_ok:
            sTile = sScoreGroups[(None, None, slot, buf_warp)]
            sTile[(cute.arch.lane_idx(), (None, quad))].store(quad_max.load())
        else:
            tma_ok = (
                quad_len == cutlass.Int32(self.score_quad_pages)
                and (col & cutlass.Int32(3)) == cutlass.Int32(0)
                and col + cutlass.Int32(self.score_quad_pages) <= warp_min_limit
            )
            if tma_ok:
                sQuads = sScoreQuads[(None, None, None, slot, buf_warp)]
                sQuads[(cute.arch.lane_idx(), None, quad)].store(quad_max.load())
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    gQuad = cute.local_tile(
                        cute.domain_offset((q_row_base, col), quad_tma_tensor),
                        (cute.arch.WARP_SIZE, self.score_quad_pages),
                        (0, None),
                    )
                    tSsQ, tSgQ = cpasync.tma_partition(
                        quad_tma_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sQuads, 0, 2),
                        cute.group_modes(gQuad, 0, 2),
                    )
                    cute.copy(quad_tma_atom, tSsQ[(None, quad)], tSgQ[(None, 0)])
                    cute.arch.cp_async_bulk_commit_group()
            else:
                # Register path: a lane whose four pages are all valid writes them as one 16-byte
                # store (the column is a multiple of 4 whenever the quad is full); the others
                # store page by page.
                lane_full = (
                    quad_len == cutlass.Int32(self.score_quad_pages)
                    and (col & cutlass.Int32(3)) == cutlass.Int32(0)
                    and col + cutlass.Int32(self.score_quad_pages) <= my_limit
                )
                if lane_full:
                    quad_ptr = cute.make_ptr(
                        cutlass.Float32,
                        mScores.iterator.toint()
                        + (
                            cutlass.Int64(row_global) * cutlass.Int64(mScores.stride[0])
                            + cutlass.Int64(col)
                        )
                        * cutlass.Int64(4),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    cute.make_tensor(quad_ptr, cute.make_layout((self.score_quad_pages,))).store(
                        quad_max.load()
                    )
                else:
                    for j in cutlass.range_constexpr(self.score_quad_pages):
                        if cutlass.Int32(j) < quad_len and col + cutlass.Int32(j) < my_limit:
                            mScores[(row_global, col + cutlass.Int32(j))] = quad_max[j]

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        q_tma_atom: cute.CopyAtom,
        q_tma_tensor: cute.Tensor,
        k_tma_atom: cute.CopyAtom,
        k_tma_tensor: cute.Tensor,
        cta_layout_vmnk: cute.Layout,
        q_smem_layout: cute.ComposedLayout,
        k_smem_layout: cute.ComposedLayout,
        mPageTable: cute.Tensor,
        mScores: cute.Tensor,
        mTaskDescriptors: cute.Tensor,
        mWorkCounter: cute.Tensor,
        mSchedMeta: cute.Tensor,
        quad_tma_atom: cute.CopyAtom,
        quad_tma_tensor: cute.Tensor,
        group_tma_atom: cute.CopyAtom,
        group_tma_tensor: cute.Tensor,
        num_heads: cutlass.Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank)
        mma_tile_coord_v = bidx % cute.size(cta_layout_vmnk, mode=[0])
        is_leader_cta = mma_tile_coord_v == 0
        cluster_idx = bidx // cutlass.Int32(self.cta_group_size)
        is_score_warp = warp_idx <= cutlass.Int32(self.score_group_warp_ids[0][-1]) or (
            warp_idx >= cutlass.Int32(self.score_group_warp_ids[1][0])
            and warp_idx <= cutlass.Int32(self.score_group_warp_ids[1][-1])
        )

        if warp_idx == self.q_load_warp_id:
            cpasync.prefetch_descriptor(q_tma_atom)
        if warp_idx == self.k_load_warp_id:
            cpasync.prefetch_descriptor(k_tma_atom)
        if is_score_warp:
            cpasync.prefetch_descriptor(quad_tma_atom)
            cpasync.prefetch_descriptor(group_tma_atom)
        # The Q-loader warp (the leader CTA's scheduler) loads the plan's queue metadata at entry
        # so the latency hides under the barrier and TMEM setup: lane l holds the queue offset and
        # count of bucket 31 - l (descending cost); lanes 0-7 hold this cluster's first task.
        sched_cnt = cutlass.Int32(0)
        sched_excl = cutlass.Int32(0)
        sched_first_word = cutlass.Int32(0)
        if warp_idx == self.q_load_warp_id:
            sched_excl = mSchedMeta[cute.arch.lane_idx()]
            sched_cnt = mSchedMeta[cutlass.Int32(cute.arch.WARP_SIZE) + cute.arch.lane_idx()]
            sched_first_word = mSchedMeta[
                cutlass.Int32(self.sched_meta_header_words)
                + cluster_idx * cutlass.Int32(self.sched_words)
                + cutlass.min(cute.arch.lane_idx(), cutlass.Int32(self.sched_words - 1))
            ]

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # Score staging: (row, page, quad, slot, score warp) dense 32 x 4 fp32 quads (16-byte rows)
        # that each score warp fills with one 16-byte store per lane and hands to the TMA engine.
        quad_words = cute.arch.WARP_SIZE * self.score_quad_pages
        sScoreQuads = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(
                (cute.arch.WARP_SIZE, self.score_quad_pages, 2, 2, self.num_score_warps),
                stride=(self.score_quad_pages, 1, quad_words, 2 * quad_words, 4 * quad_words),
            ),
            byte_alignment=128,
        )
        # Group tiles: (row, (page in quad, quad), slot, score warp) 32 x 8 tiles, SWIZZLE_32B so
        # that the lanes' 16-byte quad stores spread over all banks; one TMA store per group.
        group_words = cute.arch.WARP_SIZE * self.score_group_pages
        sScoreGroupsRaw = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(2 * self.num_score_warps * group_words),
            byte_alignment=1024,
        )
        sScoreGroups = cute.make_tensor(
            cute.recast_ptr(sScoreGroupsRaw.iterator, swizzle_=cute.make_swizzle(1, 4, 3)),
            cute.make_layout(
                (cute.arch.WARP_SIZE, (self.score_quad_pages, 2), 2, self.num_score_warps),
                stride=(self.score_group_pages, (1, self.score_quad_pages), group_words, 2 * group_words),
            ),
        )
        sQ = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=q_smem_layout.outer,
            byte_alignment=128,
            swizzle=q_smem_layout.inner,
        )
        sK = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=k_smem_layout.outer,
            byte_alignment=128,
            swizzle=k_smem_layout.inner,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierIndexerSm100.TmemPtr),
            num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buffer,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_ids[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch="sm_107" if self.a_from_tmem else "sm_100",
        )
        thread_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        q_tma_bytes = (
            cute.size_in_bytes(self.io_dtype, cute.select(q_smem_layout, mode=[0, 1, 2]))
            * self.cta_group_size
        )
        k_tma_bytes = (
            cute.size_in_bytes(self.io_dtype, cute.select(k_smem_layout, mode=[0, 1, 2]))
            * self.cta_group_size
        )
        # Tile A's Q stages: both MMA warps release them (the second warp reads tile A for an
        # unpaired task). Tile B's stages (the upper half of sQ) have their own pipe, consumed by
        # the second MMA warp only.
        q_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_tile_stages,
            producer_group=thread_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.mma_warp_ids)),
            tx_count=q_tma_bytes,
            barrier_storage=storage.q_a_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        q_pipe_b = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_tile_stages,
            producer_group=thread_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=q_tma_bytes,
            barrier_storage=storage.q_b_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # A K slot holds two pages behind one barrier pair; both MMA warps release every slot.
        k_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_slots,
            producer_group=thread_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.mma_warp_ids)),
            tx_count=k_tma_bytes * self.k_pages_per_slot,
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # Accumulator stages: released by the owning score group's warps of both CTAs.
        acc_pipe = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stages,
            producer_group=thread_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.cta_group_size * self.score_warps_per_group
            ),
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        sSchedRing = storage.sched_ring.get_tensor(
            cute.make_layout((self.sched_ring_slots, self.sched_words), stride=(self.sched_words, 1))
        )
        sched_ring_ptr = storage.sched_ring.data_ptr()
        sched_full_mbar = storage.sched_full_mbar.data_ptr()
        sched_empty_mbar = storage.sched_empty_mbar.data_ptr()
        sched_go_mbar = storage.sched_go_mbar.data_ptr()
        if warp_idx == cutlass.Int32(0):
            with cute.arch.elect_one():
                for sched_s in cutlass.range_constexpr(self.sched_ring_slots):
                    cute.arch.mbarrier_init(sched_full_mbar + sched_s, 1)
                    cute.arch.mbarrier_init(sched_empty_mbar + sched_s, self.sched_consumer_count)
                cute.arch.mbarrier_init(sched_go_mbar, 1)
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        gK = cute.local_tile(
            k_tma_tensor,
            cute.select(self.mma_tiler_mnk, mode=[1, 2]),
            (None, 0, None),
        )
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgK = thr_mma.partition_B(gK)
        tKsK, tKgK = cpasync.tma_partition(
            k_tma_atom,
            cta_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tCgK, 0, 3),
        )
        q_mcast_mask = cpasync.create_tma_multicast_mask(
            cta_layout_vmnk,
            cta_coord_vmnk,
            mcast_mode=2,
        )
        k_mcast_mask = cpasync.create_tma_multicast_mask(
            cta_layout_vmnk,
            cta_coord_vmnk,
            mcast_mode=1,
        )
        tCrK = tiled_mma.make_fragment_B(sK)
        if cutlass.const_expr(not self.a_from_tmem):
            tCrQ = tiled_mma.make_fragment_A(sQ)
        else:
            # Placeholder for the untraced SMEM-A branches; A comes from TMEM.
            tCrQ = tCrK
        acc_shape = tiled_mma.partition_shape_C((self.q_tile, self.k_tile))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))

        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)
        tmem.allocate(self.tmem_alloc_cols)
        tmem.relinquish_alloc_permit()
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc_staged = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        if cutlass.const_expr(self.a_from_tmem):
            # Q tiles A and B behind the accumulator stages, filled by a 128x256b SMEM-to-TMEM copy.
            tQ_ptr = cute.recast_ptr(tmem_ptr + self.acc_stages * self.k_tile, dtype=self.io_dtype)
            tAtQ_shape = tiled_mma.partition_shape_A((self.q_tile, self.head_dim))
            tAtQ_frag = tiled_mma.make_fragment_A(tAtQ_shape)
            tAtQ = cute.make_tensor(tQ_ptr, tAtQ_frag.layout)
            tQ_cp_atom = cute.make_copy_atom(tcgen05.Cp128x256bOp(tcgen05.CtaGroup.TWO), self.io_dtype)
            tQ_cp_tiled = tcgen05.make_s2t_copy(tQ_cp_atom, tAtQ).get_slice(0)
            frg_sQ = tQ_cp_tiled.partition_S(sQ)
            frg_tQ = tQ_cp_tiled.partition_D(tAtQ)
            frg_sQ_desc = tcgen05.get_s2t_smem_desc_tensor(tQ_cp_atom, frg_sQ)
            tQ2_ptr = cute.recast_ptr(
                tmem_ptr + self.acc_stages * self.k_tile + self.tmem_q_cols, dtype=self.io_dtype
            )
            tAtQ2 = cute.make_tensor(tQ2_ptr, tAtQ_frag.layout)
            frg_tQ2 = tcgen05.make_s2t_copy(tQ_cp_atom, tAtQ2).get_slice(0).partition_D(tAtQ2)
        else:
            # Placeholders for the untraced TMEM-A branches; A comes from SMEM (tCrQ).
            tAtQ = tCrQ
            tAtQ2 = tCrQ
            tQ_cp_atom = k_tma_atom
            frg_sQ_desc = tCrQ
            frg_tQ = tCrQ
            frg_tQ2 = tCrQ
        # Each 128-column accumulator row is read in two 64-column halves.
        tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), self.acc_dtype)
        if cutlass.const_expr(self.use_tmem_load_reduce):
            tmem_load_reduce_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition.x64,
                    redOp=tcgen05.TmemLoadRedOp.MAX,
                ),
                self.acc_dtype,
            )
        else:
            tmem_load_reduce_atom = tmem_atom
        tmem_tiled_copy = tcgen05.make_tmem_copy(
            tmem_atom,
            tCtAcc_staged[(None, None, None, 0)],
        )
        # Score thread i of either group reads accumulator row i of this CTA's 128-row half.
        score_tid = tidx % cutlass.Int32(self.score_warps_per_group * cute.arch.WARP_SIZE)
        tmem_thr_copy = tmem_tiled_copy.get_slice(score_tid)
        tTR_tAcc_staged = tmem_thr_copy.partition_S(tCtAcc_staged)
        tCcC = thr_mma.partition_C(cute.make_identity_tensor((self.q_tile, self.k_tile)))
        tTR_cC = tmem_thr_copy.partition_D(tCcC)
        tTR_rAcc = cute.make_rmem_tensor(tTR_cC.shape, self.acc_dtype)
        tTR_rBlockMax = cute.make_rmem_tensor(cute.make_layout((1, 2), stride=(0, 1)), self.acc_dtype)

        if is_score_warp:
            cute.arch.setmaxregister_increase(self.num_regs_score)
            row_m = tTR_cC[0][0]
            # 32x32b TMEM loads: lane i of a warp owns accumulator row row_base + i.
            row_base = row_m - cute.arch.lane_idx()
            score_group = cutlass.Int32(0)
            if warp_idx >= cutlass.Int32(self.score_group_warp_ids[1][0]):
                score_group = cutlass.Int32(1)
            # Staging buffers are indexed by the warp's position among the eight score warps.
            score_warp = warp_idx - cutlass.Int32(
                self.score_group_warp_ids[1][0] - self.score_warps_per_group
            ) * score_group
            ring_base = score_group * cutlass.Int32(2)
            # This group's two-stage accumulator ring, the staging slot parity (alternating between
            # groups so a tile's bulk store may still be reading while the next group stages) and
            # the four page maxima of the quad in flight.
            ring_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            quad_slot = cutlass.Int32(0)
            quad_max = cute.make_rmem_tensor((self.score_quad_pages,), cutlass.Float32)
            sched_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.sched_ring_slots
            )
            running = cutlass.Int32(1)
            while running != cutlass.Int32(0):
                (
                    _,
                    q_global_begin,
                    q_position_begin,
                    page_begin,
                    q_rows,
                    page_count,
                    q_rows_b,
                    has_b,
                ) = self._sched_take(sSchedRing, sched_full_mbar, sched_empty_mbar, sched_state, is_leader_cta)
                sched_state.advance()
                if page_count == cutlass.Int32(0):
                    running = cutlass.Int32(0)
                # Paired task: ring 0 carries tile A's pages, ring 1 tile B's. Unpaired task: tile
                # A's pages alternate between the rings in 8-page octets (ring 0: pages 0-7, 16-23,
                # ...). Full groups run a statically unrolled body; a task's partial last group runs
                # quad by quad. (The sentinel's zero pages run neither.)
                paired = has_b != cutlass.Int32(0)
                my_q_begin = q_global_begin
                my_pos_begin = q_position_begin
                my_rows = q_rows
                if paired and score_group == cutlass.Int32(1):
                    my_q_begin = q_global_begin + cutlass.Int32(self.q_tile)
                    my_pos_begin = q_position_begin + cutlass.Int32(self.q_tile) // num_heads
                    my_rows = q_rows_b
                my_valid = row_m < my_rows
                my_block = (my_pos_begin + row_m // num_heads) >> cutlass.Int32(7)
                my_limit = cutlass.Int32(0)
                if my_valid:
                    my_limit = my_block
                n_iters = page_count
                if not paired:
                    n_iters = (page_count >> cutlass.Int32(4)) * cutlass.Int32(self.score_group_pages)
                    n_iters += self._octet_pages(
                        (page_count & cutlass.Int32(15)) - cutlass.Int32(self.score_group_pages) * score_group
                    )
                # Smallest row limit in this warp: full quads ending at or below it are stored by
                # TMA. Warp-uniform after the butterfly; told to the compiler so the quad-store
                # branch stays on the uniform datapath.
                warp_min_limit = my_limit
                for offset in cutlass.range_constexpr(5):
                    other = cute.arch.shuffle_sync_bfly(warp_min_limit, cute.arch.WARP_SIZE >> (offset + 1))
                    if other < warp_min_limit:
                        warp_min_limit = other
                warp_min_limit = cute.arch.make_warp_uniform(warp_min_limit)
                q_row_base = my_q_begin + row_base
                row_global = my_q_begin + row_m
                group_start = cutlass.Int32(0)
                # A finished group tile's TMA store is issued after the next group's first release
                # (or at the end of the task), so the fence finds the staging stores complete and
                # the issue latency falls into that page's slack.
                pend_tma = cutlass.Boolean(False)
                pend_col = cutlass.Int32(0)
                pend_slot = cutlass.Int32(0)
                prev_max = cutlass.Float32(0.0)
                for grp in cutlass.range(n_iters >> cutlass.Int32(3), unroll=1):
                    for k in cutlass.range_constexpr(self.score_group_pages):
                        block_max = self._score_page(
                            acc_pipe, ring_base, ring_state, tTR_tAcc_staged, tmem_load_reduce_atom,
                            tmem_tiled_copy, tTR_rAcc, tTR_rBlockMax, prev_max,
                        )
                        prev_max = block_max
                        ring_state.advance()
                        if cutlass.const_expr(k == 0):
                            # Every bulk store issued so far has read its staging tile (the newest
                            # is the previous group's, issued 8 pages ago) before this group's
                            # staging slot, last used two groups ago, is written at page 3; then
                            # the previous group's store goes out.
                            with cute.arch.elect_one():
                                cute.arch.cp_async_bulk_wait_group(0, read=True)
                            cute.arch.sync_warp()
                            if pend_tma:
                                self._store_group_tma(
                                    sScoreGroups, group_tma_atom, group_tma_tensor, score_warp, pend_slot, pend_col,
                                    q_row_base,
                                )
                                pend_tma = cutlass.Boolean(False)
                        quad_max[k % 4] = block_max
                        if cutlass.const_expr(k == 3):
                            # First page column of this group (paired: consecutive pages; unpaired:
                            # this ring's octet of the 16-page window) and whether the whole group
                            # lies below every row's limit at a 16-byte aligned column.
                            group_col = page_begin + group_start
                            if not paired:
                                group_col = (
                                    page_begin
                                    + cutlass.Int32(2) * group_start
                                    + cutlass.Int32(self.score_group_pages) * score_group
                                )
                            group_ok = (group_col & cutlass.Int32(3)) == cutlass.Int32(0) and group_col + cutlass.Int32(
                                self.score_group_pages
                            ) <= warp_min_limit
                        if cutlass.const_expr(k % 4 == 3):
                            self._store_quad(
                                quad_max, sScoreQuads, quad_tma_atom, quad_tma_tensor, sScoreGroups, group_ok, mScores,
                                score_warp, quad_slot, cutlass.Int32(k // 4), cutlass.Int32(self.score_quad_pages),
                                group_col, warp_min_limit, my_limit, q_row_base, row_global,
                            )
                    pend_tma = group_ok
                    pend_col = group_col
                    pend_slot = quad_slot
                    group_start += cutlass.Int32(self.score_group_pages)
                    quad_slot ^= cutlass.Int32(1)
                # Task end: drain the older bulk stores, then issue the last group's.
                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.sync_warp()
                if pend_tma:
                    self._store_group_tma(
                        sScoreGroups, group_tma_atom, group_tma_tensor, score_warp, pend_slot, pend_col, q_row_base
                    )
                n_tail = n_iters & cutlass.Int32(self.score_group_pages - 1)
                tail_col = page_begin + group_start
                if not paired:
                    tail_col = (
                        page_begin
                        + cutlass.Int32(2) * group_start
                        + cutlass.Int32(self.score_group_pages) * score_group
                    )
                for tq in cutlass.range((n_tail + cutlass.Int32(3)) >> cutlass.Int32(2), unroll=1):
                    quad_len = n_tail - cutlass.Int32(self.score_quad_pages) * tq
                    if quad_len > cutlass.Int32(self.score_quad_pages):
                        quad_len = cutlass.Int32(self.score_quad_pages)
                    for j in cutlass.range_constexpr(self.score_quad_pages):
                        if cutlass.Int32(j) < quad_len:
                            block_max = self._score_page(
                                acc_pipe, ring_base, ring_state, tTR_tAcc_staged, tmem_load_reduce_atom,
                                tmem_tiled_copy, tTR_rAcc, tTR_rBlockMax, prev_max,
                            )
                            prev_max = block_max
                            ring_state.advance()
                            quad_max[j] = block_max
                    self._store_quad(
                        quad_max, sScoreQuads, quad_tma_atom, quad_tma_tensor, sScoreGroups, cutlass.Boolean(False),
                        mScores, score_warp, quad_slot, tq, quad_len, tail_col, warp_min_limit, my_limit, q_row_base,
                        row_global,
                    )
                if n_tail > cutlass.Int32(0):
                    quad_slot ^= cutlass.Int32(1)
            # The bulk score stores must have read their staging tiles before the CTA's SMEM goes
            # away; their global writes complete with the kernel (as in CUTLASS TMA-store epilogues).
            with cute.arch.elect_one():
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_warp()
        elif warp_idx == self.q_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            q_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.q_tile_stages
            )
            q_b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.q_tile_stages
            )
            sched_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.sched_ring_slots
            )
            # Leader: scheduler state. Total task count (offset + count of the cheapest bucket, lane
            # 31) and the counter value whose claim is the launch's last one.
            sched_total = cute.arch.shuffle_sync(sched_excl, cute.arch.WARP_SIZE - 1) + cute.arch.shuffle_sync(
                sched_cnt, cute.arch.WARP_SIZE - 1
            )
            sched_limit = sched_total - cutlass.Int32(1)
            sched_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.sched_ring_slots
            )
            sched_go_phase = cutlass.Int32(0)
            sched_first = cutlass.Int32(1)
            # The task after the current one, resolved and published one iteration earlier.
            sched_next_q = cutlass.Int32(0)
            sched_next_pages = cutlass.Int32(0)
            sched_next_has_b = cutlass.Int32(0)
            running = cutlass.Int32(1)
            while running != cutlass.Int32(0):
                has_b = cutlass.Int32(0)
                q_global_begin = cutlass.Int32(0)
                page_count = cutlass.Int32(0)
                if is_leader_cta:
                    # Scheduler: the first task is this cluster's index, decoded by the plan (lanes
                    # 0-7, loaded at entry) and published here; every later task was resolved and
                    # published one iteration earlier, right after the previous task's Q was issued.
                    if sched_first == cutlass.Int32(1):
                        (_, q_global_begin, _, _, _, page_count, _, has_b) = self._sched_publish_words(
                            cute.arch.shuffle_sync(sched_first_word, 0),
                            cute.arch.shuffle_sync(sched_first_word, 1),
                            cute.arch.shuffle_sync(sched_first_word, 2),
                            cute.arch.shuffle_sync(sched_first_word, 3),
                            cute.arch.shuffle_sync(sched_first_word, 4),
                            cute.arch.shuffle_sync(sched_first_word, 5),
                            cute.arch.shuffle_sync(sched_first_word, 6),
                            cute.arch.shuffle_sync(sched_first_word, 7),
                            sSchedRing, sched_ring_ptr, sched_full_mbar, sched_empty_mbar, sched_prod,
                        )
                        sched_prod.advance()
                        sched_first = cutlass.Int32(0)
                    else:
                        q_global_begin = sched_next_q
                        page_count = sched_next_pages
                        has_b = sched_next_has_b
                else:
                    (_, q_global_begin, _, _, _, page_count, _, has_b) = self._sched_take(
                        sSchedRing, sched_full_mbar, sched_empty_mbar, sched_state, is_leader_cta
                    )
                    sched_state.advance()
                if page_count == cutlass.Int32(0):
                    running = cutlass.Int32(0)
                if page_count > cutlass.Int32(0):
                    mQ_task = cute.domain_offset((q_global_begin, 0), q_tma_tensor)
                    gQ = cute.local_tile(mQ_task, cute.slice_(self.mma_tiler_mnk, (None, 0, None)), (None, None))
                    tCgQ = thr_mma.partition_A(gQ)
                    tQsQ, tQgQ = cpasync.tma_partition(
                        q_tma_atom,
                        cta_coord_vmnk[2],
                        cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tCgQ, 0, 3),
                    )
                    q_pipe.producer_acquire(q_producer_state)
                    cute.copy(
                        q_tma_atom,
                        tQgQ[(None, 0, 0)],
                        tQsQ[(None, q_producer_state.index)],
                        tma_bar_ptr=q_pipe.producer_get_barrier(q_producer_state),
                        mcast_mask=q_mcast_mask,
                    )
                    q_producer_state.advance()
                    if has_b != cutlass.Int32(0):
                        # Paired task: the partner tile (next 256 Q rows) lands in the upper Q
                        # stages through its own pipe.
                        mQ_task_b = cute.domain_offset(
                            (q_global_begin + cutlass.Int32(self.q_tile), 0), q_tma_tensor
                        )
                        gQ_b = cute.local_tile(
                            mQ_task_b, cute.slice_(self.mma_tiler_mnk, (None, 0, None)), (None, None)
                        )
                        tCgQ_b = thr_mma.partition_A(gQ_b)
                        tQsQ_b, tQgQ_b = cpasync.tma_partition(
                            q_tma_atom,
                            cta_coord_vmnk[2],
                            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
                            cute.group_modes(sQ, 0, 3),
                            cute.group_modes(tCgQ_b, 0, 3),
                        )
                        q_pipe_b.producer_acquire(q_b_producer_state)
                        cute.copy(
                            q_tma_atom,
                            tQgQ_b[(None, 0, 0)],
                            tQsQ_b[(None, cutlass.Int32(self.q_tile_stages) + q_b_producer_state.index)],
                            tma_bar_ptr=q_pipe_b.producer_get_barrier(q_b_producer_state),
                            mcast_mask=q_mcast_mask,
                        )
                        q_b_producer_state.advance()
                if is_leader_cta:
                    if page_count > cutlass.Int32(0):
                        # Schedule the task after this one when the leader K loader is
                        # sched_lead_slots from the end of this task's issue: claim the next queue
                        # position, resolve its descriptor and publish it to both CTAs. Claiming
                        # this late keeps the queue's end balance; the lead still covers the claim,
                        # the loads and the next Q's TMA.
                        cute.arch.mbarrier_wait(sched_go_mbar, sched_go_phase)
                        sched_go_phase = sched_go_phase ^ cutlass.Int32(1)
                        sched_claimed = cute.arch.shuffle_sync(self._sched_claim(mWorkCounter), 0)
                        self._sched_reset_if_last(mWorkCounter, sched_claimed, sched_limit)
                        (_, sched_next_q, _, _, _, sched_next_pages, _, sched_next_has_b) = self._sched_publish(
                            cutlass.Int32(self.num_persistent_clusters) + sched_claimed, sched_total, sched_excl,
                            sched_cnt, mTaskDescriptors, sSchedRing, sched_ring_ptr, sched_full_mbar,
                            sched_empty_mbar, sched_prod,
                        )
                        sched_prod.advance()
            q_pipe.producer_tail(q_producer_state)
            q_pipe_b.producer_tail(q_b_producer_state)
            if is_leader_cta:
                # Ring producer tail: every consumer of both CTAs has taken the sentinel, and so
                # issued its last remote arrive on this CTA's barriers, before this CTA may exit.
                for _ in cutlass.range_constexpr(self.sched_ring_slots):
                    cute.arch.mbarrier_wait(sched_empty_mbar + sched_prod.index, sched_prod.phase)
                    sched_prod.advance()
        elif warp_idx == self.k_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            k_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.k_slots)
            sched_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.sched_ring_slots
            )
            running = cutlass.Int32(1)
            while running != cutlass.Int32(0):
                (batch_idx, _, _, page_begin, _, page_count, _, has_b) = self._sched_take(
                    sSchedRing, sched_full_mbar, sched_empty_mbar, sched_state, is_leader_cta
                )
                sched_state.advance()
                if page_count == cutlass.Int32(0):
                    running = cutlass.Int32(0)
                if page_count > cutlass.Int32(0):
                    # A K slot holds two pages (two TMA copies, one barrier armed for both); the
                    # last slot of a task may hold one. The next slot's physical page indices are
                    # fetched while this slot's TMAs are issued.
                    last_page = page_begin + page_count - cutlass.Int32(1)
                    phys_a = mPageTable[batch_idx, page_begin]
                    phys_b = mPageTable[batch_idx, cutlass.min(page_begin + cutlass.Int32(1), last_page)]
                    n_slots_task = (page_count + cutlass.Int32(1)) // cutlass.Int32(2)
                    sched_trigger_slot = n_slots_task - (cutlass.Int32(self.sched_lead_slots) >> has_b)
                    if sched_trigger_slot < cutlass.Int32(0):
                        sched_trigger_slot = cutlass.Int32(0)
                    for slot_it in cutlass.range(n_slots_task, unroll=1):
                        logical_page = page_begin + slot_it * cutlass.Int32(2)
                        n_in = cutlass.min(cutlass.Int32(2), page_begin + page_count - logical_page)
                        next_phys_a = mPageTable[batch_idx, cutlass.min(logical_page + cutlass.Int32(2), last_page)]
                        next_phys_b = mPageTable[batch_idx, cutlass.min(logical_page + cutlass.Int32(3), last_page)]
                        k_pipe.producer_acquire(k_producer_state, expected_tx=cutlass.Int32(k_tma_bytes) * n_in)
                        k_bar = k_pipe.producer_get_barrier(k_producer_state)
                        cute.copy(
                            k_tma_atom,
                            tKgK[(None, 0, phys_a)],
                            tKsK[(None, k_producer_state.index * cutlass.Int32(2))],
                            tma_bar_ptr=k_bar,
                            mcast_mask=k_mcast_mask,
                        )
                        if n_in == cutlass.Int32(2):
                            cute.copy(
                                k_tma_atom,
                                tKgK[(None, 0, phys_b)],
                                tKsK[(None, k_producer_state.index * cutlass.Int32(2) + cutlass.Int32(1))],
                                tma_bar_ptr=k_bar,
                                mcast_mask=k_mcast_mask,
                            )
                        k_producer_state.advance()
                        if is_leader_cta:
                            if slot_it == sched_trigger_slot:
                                # Let the scheduler (leader Q loader) claim and publish the next task.
                                with cute.arch.elect_one():
                                    cute.arch.mbarrier_arrive(sched_go_mbar)
                        phys_a = next_phys_a
                        phys_b = next_phys_b
            k_pipe.producer_tail(k_producer_state)
        else:
            # MMA warps. Warp 6 owns tile A (Q stage 0, or tQ on SM107) and accumulator ring 0
            # (TMEM stages 0, 1); warp 7 owns tile B (the upper Q stage, or tQ2) and ring 1 (stages
            # 2, 3). Both consume the shared K ring, each releasing every slot once. For an
            # unpaired task both warps hold tile A and the two-page K slots alternate between them
            # in groups of four (warp 6: pages 0-7, 16-23, ...; warp 7: pages 8-15, 24-31, ...), so
            # each score group's 8-page group is 8 consecutive pages and is written like a paired
            # tile. Each warp waits only on its own ring.
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            q_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.q_tile_stages)
            q_b_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.q_tile_stages)
            k_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.k_slots)
            ring = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            num_k_blocks = cute.size(tCrK, mode=[2])
            is_warp_a = warp_idx == cutlass.Int32(self.mma_warp_ids[0])
            my_ring = cutlass.Int32(1)
            if is_warp_a:
                my_ring = cutlass.Int32(0)
            ring_base_mma = my_ring * cutlass.Int32(2)
            sched_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.sched_ring_slots
            )
            if is_leader_cta:
                running = cutlass.Int32(1)
                while running != cutlass.Int32(0):
                    (_, _, _, _, _, page_count, _, has_b) = self._sched_take(
                        sSchedRing, sched_full_mbar, sched_empty_mbar, sched_state, is_leader_cta
                    )
                    sched_state.advance()
                    if page_count == cutlass.Int32(0):
                        running = cutlass.Int32(0)
                    if page_count > cutlass.Int32(0):
                        paired = has_b != cutlass.Int32(0)
                        # Both warps consume tile A's stage: warp 7 reads it for an unpaired task
                        # and takes tile B from its own pipe when paired.
                        q_pipe.consumer_wait(q_state)
                        q_stage_mine = q_state.index
                        if cutlass.const_expr(self.a_from_tmem):
                            # Q to TMEM once per task; the commit-based release frees the SMEM
                            # stage when the copy completes.
                            if is_warp_a:
                                cute.copy(tQ_cp_atom, frg_sQ_desc[(None, None, None, None, 0)], frg_tQ)
                            else:
                                if not paired:
                                    cute.copy(tQ_cp_atom, frg_sQ_desc[(None, None, None, None, 0)], frg_tQ2)
                            q_pipe.consumer_release(q_state)
                            q_state.advance()
                            if not is_warp_a:
                                if paired:
                                    q_pipe_b.consumer_wait(q_b_state)
                                    cute.copy(tQ_cp_atom, frg_sQ_desc[(None, None, None, None, 1)], frg_tQ2)
                                    q_pipe_b.consumer_release(q_b_state)
                                    q_b_state.advance()
                        else:
                            # A from SMEM: the stage this warp's MMAs read for the whole task (tile
                            # A's, or tile B's upper stage for the second warp of a paired task).
                            if not is_warp_a:
                                if paired:
                                    q_pipe_b.consumer_wait(q_b_state)
                                    q_stage_mine = cutlass.Int32(self.q_tile_stages) + q_b_state.index
                        n_slots_task = (page_count + cutlass.Int32(1)) // cutlass.Int32(2)
                        for slot_it in cutlass.range(n_slots_task, unroll=1):
                            k_pipe.consumer_wait(k_consumer_state)
                            n_in = cutlass.min(cutlass.Int32(2), page_count - slot_it * cutlass.Int32(2))
                            k_smem0 = k_consumer_state.index * cutlass.Int32(2)
                            n_mine = n_in
                            if not paired:
                                if ((slot_it >> cutlass.Int32(2)) & cutlass.Int32(1)) != my_ring:
                                    n_mine = cutlass.Int32(0)
                            if is_warp_a:
                                for j in cutlass.range_constexpr(2):
                                    if cutlass.Int32(j) < n_mine:
                                        acc_pipe.sync_object_empty.wait(ring.index, ring.phase)
                                        tCtAcc = tCtAcc_staged[(None, None, None, ring.index)]
                                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                                            if cutlass.const_expr(self.a_from_tmem):
                                                cute.gemm(
                                                    tiled_mma, tCtAcc, tAtQ[(None, None, k_block_idx)],
                                                    tCrK[(None, None, k_block_idx, k_smem0 + cutlass.Int32(j))], tCtAcc,
                                                )
                                            else:
                                                cute.gemm(
                                                    tiled_mma, tCtAcc, tCrQ[(None, None, k_block_idx, q_stage_mine)],
                                                    tCrK[(None, None, k_block_idx, k_smem0 + cutlass.Int32(j))], tCtAcc,
                                                )
                                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                        acc_pipe.sync_object_full.arrive(ring.index, acc_pipe.producer_mask, acc_pipe.cta_group)
                                        ring.advance()
                            else:
                                for j in cutlass.range_constexpr(2):
                                    if cutlass.Int32(j) < n_mine:
                                        stage = ring_base_mma + ring.index
                                        acc_pipe.sync_object_empty.wait(stage, ring.phase)
                                        tCtAcc = tCtAcc_staged[(None, None, None, stage)]
                                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                                            if cutlass.const_expr(self.a_from_tmem):
                                                cute.gemm(
                                                    tiled_mma, tCtAcc, tAtQ2[(None, None, k_block_idx)],
                                                    tCrK[(None, None, k_block_idx, k_smem0 + cutlass.Int32(j))], tCtAcc,
                                                )
                                            else:
                                                cute.gemm(
                                                    tiled_mma, tCtAcc, tCrQ[(None, None, k_block_idx, q_stage_mine)],
                                                    tCrK[(None, None, k_block_idx, k_smem0 + cutlass.Int32(j))], tCtAcc,
                                                )
                                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                        acc_pipe.sync_object_full.arrive(stage, acc_pipe.producer_mask, acc_pipe.cta_group)
                                        ring.advance()
                            k_pipe.consumer_release(k_consumer_state)
                            k_consumer_state.advance()
                        if cutlass.const_expr(not self.a_from_tmem):
                            # A from SMEM: the Q stage is free once this warp's last MMA of the
                            # task has completed (the release is a tcgen05 commit, so it waits
                            # for exactly that).
                            q_pipe.consumer_release(q_state)
                            q_state.advance()
                            if not is_warp_a:
                                if paired:
                                    q_pipe_b.consumer_release(q_b_state)
                                    q_b_state.advance()
            # Producer tail: the score warps must have released every stage of this warp's ring.
            for _ in cutlass.range_constexpr(2):
                acc_pipe.sync_object_empty.wait(ring_base_mma + ring.index, ring.phase)
                ring.advance()
        pipeline.sync(barrier_id=int(NamedBarrierIndexerSm100.Final))
        # Only the peer's SMEM/TMEM lifetime is at stake here: every remote operation has been
        # waited for through the pipelines' tails, and the kernel end publishes all global writes,
        # so the release fence of the default arrive (which drains the outstanding bulk stores)
        # is not needed.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        tmem.free(tmem_ptr)


__all__ = ["Q8KV8PrefillIndexerSm100"]
