# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Sparse forward combine (K2) kernels and launchers.

- ``combine``: the default. Runs ``SparseAttentionForwardCombine``, the TMA-ring
  combine; BF16/FP16 partials only.
- ``combine_f32f8``: runs ``SparseAttentionForwardCombineF32F8`` (the original
  combine), which also implements the FP32 and FP8 E4M3 STG.128 fake layouts.
  Callers that may write FP32 / FP8 partials pick it by partial dtype.
"""

# Modified Step 7: O_out write with SMEM fake->real column permutation.
# O_partial dim is in STG.128 fake layout; O_out dim is real layout.
import math
from dataclasses import dataclass
from typing import Type, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass import Float32, Int32, Int64, Boolean, const_expr

from src.common import mxfp8_quant as mx
from src.common import utils
from src.common.aot_cache import save_aot, try_load_aot
from src.common.cute_dsl_utils import assume_tensor_aligned, compile_with_timing, torch2cute_dtype_map
from src.common.seqlen_info import SeqlenInfo
from cutlass.cute import FastDivmodDivisor

from src.common.pack_gqa import PackGQAComb
from src.common.tma_utils import (
    stg128_fake_col_to_real_col,
    stg128_fp8_fake_col_to_real_col,
    stg128_half_fake_col_to_real_col,
)


# -----------------------------------------------------------------------------
# Original combine (combine_f32f8): FP32 / FP8 E4M3 partials
# -----------------------------------------------------------------------------


class SparseAttentionForwardCombineF32F8:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        head_dim: int,
        tile_m: int = 8,
        k_block_size: int = 64,
        topk: int = 16,
        num_threads: int = 256,
        stages: int = 4,
        use_pdl: bool = False,
        min_blocks_per_mp: int = 0,
    ):
        """
        Forward combine kernel for split attention computation.

        :param dtype: output data type
        :param dtype_partial: partial accumulation data type
        :param head_dim: head dimension
        :param tile_m: m block size
        :param k_block_size: k block size
        :param topk: exact number of split partials
        :param num_threads: number of threads
        :param varlen: whether using variable length sequences
        :param stages: number of pipeline stages
        """
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.k_block_size = k_block_size
        self.topk = topk
        self.num_threads = num_threads
        self.is_even_k = head_dim % k_block_size == 0
        self.stages = stages
        self.use_pdl = use_pdl
        self.min_blocks_per_mp = min_blocks_per_mp
        self.use_stg128_half_layout = dtype_partial in (cutlass.BFloat16, cutlass.Float16)
        self.use_stg128_fp8_layout = dtype_partial is cutlass.Float8E4M3FN

    @staticmethod
    def can_implement(
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        topk,
        num_threads,
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters."""
        if dtype not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]:
            return False
        if dtype_partial not in [
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            Float32,
        ]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if tile_m % 8 != 0:
            return False
        if topk > 256:
            return False
        if (tile_m * topk) % num_threads != 0:
            return False
        return True

    def _setup_attributes(self):
        # GMEM copy setup for O partial
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype_partial.width
        assert self.k_block_size % async_copy_elems == 0

        k_block_gmem = (
            128 if self.k_block_size % 128 == 0 else (64 if self.k_block_size % 64 == 0 else 32)
        )
        gmem_threads_per_row = k_block_gmem // async_copy_elems
        assert self.num_threads % gmem_threads_per_row == 0

        # Async copy atom for O partial load
        atom_async_copy_partial = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype_partial,
            num_bits_per_copy=universal_copy_bits,
        )
        tOpartial_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        vOpartial_layout = cute.make_layout((1, async_copy_elems))
        self.gmem_tiled_copy_O_partial = cute.make_tiled_copy_tv(
            atom_async_copy_partial, tOpartial_layout, vOpartial_layout
        )

        # GMEM copy setup for final O (use universal copy for store).
        # Keep this independent from O_partial: fp8 partial uses 16 elements
        # per 128b transaction, while bf16/fp16 O stores must remain 8-wide.
        output_copy_elems = universal_copy_bits // self.dtype.width
        assert self.k_block_size % output_copy_elems == 0
        gmem_threads_per_row_o = k_block_gmem // output_copy_elems
        assert self.num_threads % gmem_threads_per_row_o == 0
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tO_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_o, gmem_threads_per_row_o),
            order=(1, 0),
        )
        vO_layout = cute.make_layout((1, output_copy_elems))
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy,
            tO_layout,
            vO_layout,
        )
        # LSE copy setup with async copy (alignment = 1)
        lse_copy_bits = Float32.width  # 1 element per copy, width is in bits
        m_block_smem = (
            128
            if self.tile_m % 128 == 0
            else (
                64
                if self.tile_m % 64 == 0
                else (32 if self.tile_m % 32 == 0 else (16 if self.tile_m % 16 == 0 else 8))
            )
        )
        gmem_threads_per_row_lse = m_block_smem
        assert self.num_threads % gmem_threads_per_row_lse == 0

        # Async copy atom for LSE load
        atom_async_copy_lse = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            Float32,
            num_bits_per_copy=lse_copy_bits,
        )
        tLSE_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_lse, gmem_threads_per_row_lse),
            order=(1, 0),
        )
        vLSE_layout = cute.make_layout(1)
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            atom_async_copy_lse, tLSE_layout, vLSE_layout
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory
        # ///////////////////////////////////////////////////////////////////////////////

        # Shared memory to register copy for LSE
        self.smem_threads_per_col_lse = self.num_threads // m_block_smem
        assert 32 % self.smem_threads_per_col_lse == 0  # Must divide warp size

        s2r_layout_atom_lse = cute.make_ordered_layout(
            (self.smem_threads_per_col_lse, self.num_threads // self.smem_threads_per_col_lse),
            order=(0, 1),
        )
        self.s2r_tiled_copy_LSE = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32),
            s2r_layout_atom_lse,
            cute.make_layout(1),
        )

        # LSE shared memory layout with swizzling to avoid bank conflicts
        # This works for kBlockMSmem = 8, 16, 32, 64, 128, no bank conflicts
        if const_expr(m_block_smem == 8):
            smem_lse_swizzle = cute.make_swizzle(5, 0, 5)
        elif const_expr(m_block_smem == 16):
            smem_lse_swizzle = cute.make_swizzle(4, 0, 4)
        else:
            smem_lse_swizzle = cute.make_swizzle(3, 2, 3)
        lse_atom_splits = min(self.topk, 8)
        smem_layout_atom_lse = cute.make_composed_layout(
            smem_lse_swizzle,
            0,
            cute.make_ordered_layout((lse_atom_splits, m_block_smem), order=(1, 0)),
        )
        self.smem_layout_lse = cute.tile_to_shape(
            smem_layout_atom_lse, (self.topk, self.tile_m), (0, 1)
        )

        # O_partial staging layout.
        if const_expr(
            self.dtype_partial
            in [cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN]
        ):
            smem_layout_atom_o = _get_cpasync_smem_layout_atom(
                self.dtype_partial, self.k_block_size
            )
            self.smem_layout_o = cute.tile_to_shape(
                smem_layout_atom_o,
                (self.tile_m, self.k_block_size, self.stages),
                (0, 1, 2),
            )
        else:
            self.smem_layout_o = cute.make_ordered_layout(
                (self.tile_m, self.k_block_size, self.stages), order=(1, 0, 2)
            )

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor] = None,
        mLSE_temperature_partial: Optional[cute.Tensor] = None,
        mLSE_temperature: Optional[cute.Tensor] = None,
        cu_seqlens: Optional[cute.Tensor] = None,
        seqused: Optional[cute.Tensor] = None,
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        varlen_batch_idx: Optional[cute.Tensor] = None,
        semaphore_to_reset: Optional[cute.Tensor] = None,
        mSplitCounts: Optional[cute.Tensor] = None,
        mOutputScale: Optional[cute.Tensor] = None,
        qhead_per_kvhead: Int32 = Int32(1),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Type checking
        if const_expr(not (mO_partial.element_type == self.dtype_partial)):
            raise TypeError("O partial tensor must match dtype_partial")
        if const_expr(not (mO.element_type == self.dtype)):
            raise TypeError("O tensor must match dtype")
        if const_expr(mLSE_partial.element_type not in [Float32]):
            raise TypeError("LSE partial tensor must be Float32")
        if const_expr(mLSE is not None and mLSE.element_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(
            mLSE_temperature_partial is not None
            and mLSE_temperature_partial.element_type not in [Float32]
        ):
            raise TypeError("temperature LSE partial tensor must be Float32")
        if const_expr(mLSE_temperature is not None and mLSE_temperature.element_type not in [Float32]):
            raise TypeError("temperature LSE tensor must be Float32")
        if const_expr((mLSE_temperature_partial is None) != (mLSE_temperature is None)):
            raise ValueError(
                "temperature LSE partial and output tensors must either both be provided or both be None"
            )

        # Shape validation - input tensors are in user format, need to be converted to kernel format
        if const_expr(len(mO_partial.shape) not in [4, 5]):
            raise ValueError(
                "O partial tensor must have 4 or 5 dimensions: (num_splits, batch, seqlen, nheads, headdim) or (num_splits, total_q, nheads, headdim)"
            )
        if const_expr(len(mLSE_partial.shape) not in [3, 4]):
            raise ValueError(
                "LSE partial tensor must have 3 or 4 dimensions: (num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(len(mO.shape) not in [3, 4]):
            raise ValueError(
                "O tensor must have 3 or 4 dimensions: (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim)"
            )
        if const_expr(mLSE is not None and len(mLSE.shape) not in [2, 3]):
            raise ValueError(
                "LSE tensor must have 2 or 3 dimensions: (batch, seqlen, nheads) or (total_q, nheads)"
            )
        if const_expr(mLSE_temperature_partial is not None and len(mLSE_temperature_partial.shape) not in [3, 4]):
            raise ValueError(
                "temperature LSE partial tensor must have 3 or 4 dimensions: "
                "(num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(mLSE_temperature is not None and len(mLSE_temperature.shape) not in [2, 3]):
            raise ValueError(
                "temperature LSE tensor must have 2 or 3 dimensions: "
                "(batch, seqlen, nheads) or (total_q, nheads)"
            )
        if const_expr(mSplitCounts is not None):
            if const_expr(mSplitCounts.element_type not in [Int32]):
                raise TypeError("split_counts tensor must be Int32")
            if const_expr(cu_seqlens is not None):
                if const_expr(len(mSplitCounts.shape) != 2):
                    raise ValueError("varlen split_counts tensor must have shape (total_q, nheads_kv)")
            elif const_expr(len(mSplitCounts.shape) != 3):
                raise ValueError("batched split_counts tensor must have shape (batch, seqlen, nheads_kv)")
        if const_expr(mOutputScale is not None and mOutputScale.element_type not in [Float32]):
            raise TypeError("output_scale tensor must be Float32")

        mO_partial, mO = [assume_tensor_aligned(t) for t in (mO_partial, mO)]
        # (num_splits, b, seqlen, h, d) -> (seqlen, d, num_splits, h, b)
        # or (num_splits, total_q, h, d) -> (total_q, d, num_splits, h)
        O_partial_layout_transpose = (
            [2, 4, 0, 3, 1] if const_expr(cu_seqlens is None) else [1, 3, 0, 2]
        )
        # (b, seqlen, h, d) -> (seqlen, d, h, b) or (total_q, h, d) -> (total_q, d, h)
        mO_partial = cute.make_tensor(
            mO_partial.iterator, cute.select(mO_partial.layout, mode=O_partial_layout_transpose)
        )
        O_layout_transpose = [1, 3, 2, 0] if const_expr(cu_seqlens is None) else [0, 2, 1]
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        # (num_splits, b, h, seqlen) -> (seqlen, num_splits, h, b)
        # Input is pre-transposed: [topK, B, Hq, Sq] with Sq innermost for K2-friendly reads.
        # or (num_splits, total_q, h) -> (total_q, num_splits, h)
        LSE_partial_layout_transpose = [3, 0, 2, 1] if const_expr(cu_seqlens is None) else [1, 0, 2]
        mLSE_partial = cute.make_tensor(
            mLSE_partial.iterator,
            cute.select(mLSE_partial.layout, mode=LSE_partial_layout_transpose),
        )
        # (b, seqlen, h) -> (seqlen, h, b) or (total_q, h) -> (total_q, h)
        LSE_layout_transpose = [1, 2, 0] if const_expr(cu_seqlens is None) else [0, 1]
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if mLSE is not None
            else None
        )
        mLSE_temperature_partial = (
            cute.make_tensor(
                mLSE_temperature_partial.iterator,
                cute.select(mLSE_temperature_partial.layout, mode=LSE_partial_layout_transpose),
            )
            if mLSE_temperature_partial is not None
            else None
        )
        mLSE_temperature = (
            cute.make_tensor(
                mLSE_temperature.iterator,
                cute.select(mLSE_temperature.layout, mode=LSE_layout_transpose),
            )
            if mLSE_temperature is not None
            else None
        )

        # Determine if we have variable length sequences
        varlen = const_expr(cu_seqlens is not None or seqused is not None)

        self._setup_attributes()

        # Output-dtype permutation buffer for Step 7 (tile_m × k_block_size).
        # Accumulation stays fp32; the final dtype conversion happens before
        # the fake→real SMEM scatter to reduce half-output SMEM pressure.
        if const_expr(self.dtype in [cutlass.Float16, cutlass.BFloat16]):
            smem_layout_perm = cute.make_layout(
                (self.tile_m, self.k_block_size),
                stride=(self.k_block_size + 16, 1),
            )
        else:
            smem_layout_perm = cute.make_ordered_layout(
                (self.tile_m, self.k_block_size), order=(1, 0)
            )

        @cute.struct
        class SharedStorage:
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sLSETemperature: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sMaxValidSplit: cute.struct.Align[cute.struct.MemRange[Int32, self.tile_m], 128]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.dtype_partial, cute.cosize(self.smem_layout_o)], 128
            ]
            sO_perm: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_perm)], 128
            ]

        smem_size = SharedStorage.size_in_bytes()

        # Grid: (ceil(seqlen/tile_m), ceil(dim/k_block), num_head * batch)
        # Head separated from seqlen → enables future TMA (contiguous Sq tiles)
        seqlen = mO_partial.shape[0]
        num_head = mO_partial.shape[3]
        batch_size = (
            mO_partial.shape[4]
            if const_expr(cu_seqlens is None)
            else Int32(cu_seqlens.shape[0] - 1)
        )

        seqlen_divmod = FastDivmodDivisor(seqlen)
        head_divmod = FastDivmodDivisor(num_head)

        grid_dim = (
            cute.ceil_div(seqlen * num_head, self.tile_m),
            cute.ceil_div(self.head_dim, self.k_block_size),
            batch_size,
        )

        self.kernel(
            mO_partial,
            mLSE_partial,
            mO,
            mLSE,
            mLSE_temperature_partial,
            mLSE_temperature,
            cu_seqlens,
            seqused,
            num_splits_dynamic_ptr,
            varlen_batch_idx,
            semaphore_to_reset,
            mSplitCounts,
            mOutputScale,
            qhead_per_kvhead,
            SharedStorage,
            self.smem_layout_lse,
            self.smem_layout_o,
            smem_layout_perm,
            self.gmem_tiled_copy_O_partial,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_LSE,
            self.s2r_tiled_copy_LSE,
            seqlen_divmod,
            head_divmod,
            self.use_pdl,
            varlen,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def decode_flat_row_idx(
        self,
        idx: Int32,
        head_divmod: FastDivmodDivisor,
    ):
        """Decode flattened tile rows under the H_q-innermost contract."""
        q_idx_local, head_idx = divmod(idx, head_divmod)
        return q_idx_local, head_idx

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSE_temperature_partial: Optional[cute.Tensor],
        mLSE_temperature: Optional[cute.Tensor],
        cu_seqlens: Optional[cute.Tensor],
        seqused: Optional[cute.Tensor],
        num_splits_dynamic_ptr: Optional[cute.Tensor],
        varlen_batch_idx: Optional[cute.Tensor],
        semaphore_to_reset: Optional[cute.Tensor],
        mSplitCounts: Optional[cute.Tensor],
        mOutputScale: Optional[cute.Tensor],
        qhead_per_kvhead: Int32,
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout | cute.ComposedLayout,
        smem_layout_o: cute.Layout | cute.ComposedLayout,
        smem_layout_perm: cute.Layout,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        s2r_tiled_copy_LSE: cute.TiledCopy,
        seqlen_divmod: FastDivmodDivisor,
        head_divmod: FastDivmodDivisor,
        use_pdl: cutlass.Constexpr[bool],
        varlen: cutlass.Constexpr[bool],
    ):
        # Thread and block indices
        tidx, _, _ = cute.arch.thread_idx()
        m_block, k_block, maybe_virtual_batch = cute.arch.block_idx()

        batch_idx = (
            varlen_batch_idx[maybe_virtual_batch]
            if const_expr(varlen_batch_idx is not None)
            else maybe_virtual_batch
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sLSE_temperature = storage.sLSETemperature.get_tensor(smem_layout_lse)
        sMaxValidSplit = storage.sMaxValidSplit.get_tensor((self.tile_m,))
        sO = storage.sO.get_tensor(smem_layout_o)
        sO_perm_buf = storage.sO_perm.get_tensor(smem_layout_perm)

        # Handle semaphore reset — wait for dependent grids first
        if const_expr(use_pdl and semaphore_to_reset is not None):
            if (
                tidx == 0
                and m_block == cute.arch.grid_dim()[0] - 1
                and k_block == cute.arch.grid_dim()[1] - 1
                and maybe_virtual_batch == cute.arch.grid_dim()[2] - 1
            ):
                cute.arch.griddepcontrol_wait()
                semaphore_to_reset[0] = 0

        if const_expr(num_splits_dynamic_ptr is not None):
            raise ValueError("K2 combine requires compile-time exact topK")
        num_splits = Int32(self.topk)
        # Handle variable length sequences using SeqlenInfo
        seqlen_info = SeqlenInfo.create(
            batch_idx=batch_idx,
            seqlen_static=mO_partial.shape[0],
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            # Don't need to pass in tile size since we won't use offset_padded
        )
        seqlen, offset = seqlen_info.seqlen, seqlen_info.offset

        num_head = mO_partial.shape[3]
        max_idx = seqlen * num_head
        output_scale = Float32(1.0)
        if const_expr(mOutputScale is not None):
            output_scale = mOutputScale[0]

        if const_expr(not varlen) or m_block * self.tile_m < max_idx:
            # Wait for dependent grids (e.g., the main attention kernel that produces O_partial/LSE_partial)
            if const_expr(use_pdl):
                cute.arch.griddepcontrol_wait()

            # ===============================
            # Step 1: Load LSE_partial from gmem to shared memory
            # ===============================
            # `cLSE` (identity tensor for row/split coord tracking) is reused
            # later in steps 4-5, so it must be defined on both branches.
            cLSE = cute.make_identity_tensor((self.topk, self.tile_m))
            # Reshape mLSE_partial to PackGQA packed layout and delegate the
            # tile load to PackGQAComb.load_LSE. The packed form folds (H_q, Sq)
            # into one compound dim with H_q innermost (stride 1), so thread
            # rows that vary along h_pos produce one-sector coalesced reads.
            # Non-varlen path only — varlen keeps the original inline loop.
            if const_expr(not varlen):
                mLSE_partial_cur = seqlen_info.offset_batch(mLSE_partial, batch_idx, dim=3)
                # mLSE_partial_cur: (H_q, topK, Sq) — after initial transpose
                # [3,0,2,1] on [topK,B,Sq,H_q] and dropping B.
                # Reorder to (H_q, Sq, topK) then group modes 0..1 for packed dim:
                mLSE_partial_reord = cute.make_tensor(
                    mLSE_partial_cur.iterator,
                    cute.select(mLSE_partial_cur.layout, mode=[0, 2, 1]),
                )
                mLSE_partial_packed = cute.group_modes(mLSE_partial_reord, 0, 2)
                # shape ((H_q, Sq), topK) with H_q innermost.
                packgqa = PackGQAComb(
                    m_block_size=self.tile_m,
                    head_dim_padded=0,       # unused for LSE load
                    check_hdim_oob=False,    # unused for LSE load
                    qhead_per_kvhead=1,      # unused; num_heads_divmod is passed explicitly
                )
                packgqa.load_LSE(
                    mLSE_partial_packed,
                    sLSE,
                    self.topk,
                    gmem_tiled_copy_LSE,
                    tidx,
                    m_block,
                    num_splits,
                    seqlen,
                    head_divmod,
                    mSplitCounts,
                    batch_idx,
                    qhead_per_kvhead,
                )
                if const_expr(mLSE_temperature_partial is not None):
                    mLSE_temperature_partial_cur = seqlen_info.offset_batch(
                        mLSE_temperature_partial, batch_idx, dim=3)
                    mLSE_temperature_partial_reord = cute.make_tensor(
                        mLSE_temperature_partial_cur.iterator,
                        cute.select(mLSE_temperature_partial_cur.layout, mode=[0, 2, 1]),
                    )
                    mLSE_temperature_partial_packed = cute.group_modes(
                        mLSE_temperature_partial_reord, 0, 2)
                    packgqa.load_LSE(
                        mLSE_temperature_partial_packed,
                        sLSE_temperature,
                        self.topk,
                        gmem_tiled_copy_LSE,
                        tidx,
                        m_block,
                        num_splits,
                        seqlen,
                        head_divmod,
                        mSplitCounts,
                        batch_idx,
                        qhead_per_kvhead,
                    )
            else:
                # Varlen path keeps the same H_q-innermost flat-row contract:
                # after transpose [1, 0, 2], mLSE_partial_cur is
                # (q_local, split, head).
                # mSplitCounts is the authoritative valid-split count per
                # packed (q_abs, kv_head); masked splits stay at -inf and
                # therefore drop out of the final kernel LSE_out reduction.
                mLSE_partial_cur = seqlen_info.offset_batch(mLSE_partial, batch_idx, dim=3)
                mLSE_partial_copy = cute.tiled_divide(mLSE_partial_cur, (1,))
                gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_slice(tidx)
                tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE)
                tLSEsLSE_temperature = gmem_thr_copy_LSE.partition_D(sLSE_temperature)
                tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE)
                if const_expr(mLSE_temperature_partial is not None):
                    mLSE_temperature_partial_cur = seqlen_info.offset_batch(
                        mLSE_temperature_partial, batch_idx, dim=3)
                    mLSE_temperature_partial_copy = cute.tiled_divide(
                        mLSE_temperature_partial_cur, (1,))

                for m in cutlass.range(cute.size(tLSEcLSE, mode=[2]), unroll_full=True):
                    mi = tLSEcLSE[0, 0, m][1]
                    idx = m_block * self.tile_m + mi
                    if idx < max_idx:
                        m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                        row_count = (
                            mSplitCounts[offset + m_idx, head_idx // qhead_per_kvhead]
                            if const_expr(mSplitCounts is not None)
                            else num_splits
                        )
                        mLSE_partial_cur_copy = mLSE_partial_copy[None, m_idx, None, head_idx]
                        if const_expr(mLSE_temperature_partial is not None):
                            mLSE_temperature_partial_cur_copy = (
                                mLSE_temperature_partial_copy[None, m_idx, None, head_idx])
                        for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                            si = tLSEcLSE[0, s, 0][0]
                            if si < num_splits and si < row_count:
                                cute.copy(
                                    gmem_thr_copy_LSE,
                                    mLSE_partial_cur_copy[None, si],
                                    tLSEsLSE[None, s, m],
                                )
                                if const_expr(mLSE_temperature_partial is not None):
                                    cute.copy(
                                        gmem_thr_copy_LSE,
                                        mLSE_temperature_partial_cur_copy[None, si],
                                        tLSEsLSE_temperature[None, s, m],
                                    )
                            else:
                                tLSEsLSE[None, s, m].fill(-Float32.inf)
                                if const_expr(mLSE_temperature_partial is not None):
                                    tLSEsLSE_temperature[None, s, m].fill(-Float32.inf)
                    else:
                        for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                            tLSEsLSE[None, s, m].fill(-Float32.inf)
                            if const_expr(mLSE_temperature_partial is not None):
                                tLSEsLSE_temperature[None, s, m].fill(-Float32.inf)
            cute.arch.cp_async_commit_group()

            # ===============================
            # Step 2: Load O_partial for pipeline stages
            # ===============================

            gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_slice(tidx)
            cO = cute.make_identity_tensor((self.tile_m, self.k_block_size))
            tOcO = gmem_thr_copy_O_partial.partition_D(cO)
            tOsO_partial = gmem_thr_copy_O_partial.partition_D(sO)
            mO_partial_cur = seqlen_info.offset_batch(mO_partial, batch_idx, dim=4)

            # Precompute per-row values for flattened (q_local, head) tiles.
            num_rows = const_expr(cute.size(tOcO, mode=[1]))
            tOmidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOhidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOSplitCount = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOrOptr = cute.make_rmem_tensor(num_rows, cutlass.Int64)
            for m in cutlass.range(num_rows, unroll_full=True):
                mi = tOcO[0, m, 0][0]  # m coordinate in tile
                idx = m_block * self.tile_m + mi
                if idx >= max_idx:
                    tOhidx[m] = -1
                    tOmidx[m] = 0
                    tOSplitCount[m] = 0
                    tOrOptr[m] = cutlass.Int64(0)
                else:
                    tOmidx[m], tOhidx[m] = self.decode_flat_row_idx(idx, head_divmod)
                    if const_expr(mSplitCounts is None):
                        tOSplitCount[m] = num_splits
                    elif const_expr(cu_seqlens is None):
                        tOSplitCount[m] = mSplitCounts[
                            batch_idx, tOmidx[m], tOhidx[m] // qhead_per_kvhead
                        ]
                    else:
                        tOSplitCount[m] = mSplitCounts[
                            offset + tOmidx[m], tOhidx[m] // qhead_per_kvhead
                        ]
                    tOrOptr[m] = utils.elem_pointer(
                        mO_partial_cur,
                        (tOmidx[m], k_block * self.k_block_size, 0, tOhidx[m]),
                    ).toint()

            tOpO = None
            if const_expr(not self.is_even_k):
                tOpO = cute.make_rmem_tensor(cute.size(tOcO, mode=[2]), Boolean)
                for k in cutlass.range(cute.size(tOpO), unroll_full=True):
                    tOpO[k] = tOcO[0, 0, k][1] < mO_partial.shape[1] - k_block * self.k_block_size
                # if cute.arch.thread_idx()[0] == 0 and k_block == 1: cute.print_tensor(tOpO)

            load_O_partial = partial(
                self.load_O_partial,
                gmem_tiled_copy_O_partial,
                tOrOptr,
                tOsO_partial,
                tOhidx,
                tOSplitCount,
                tOpO,
                tOcO,
                mO_partial_cur.layout,
            )

            # Load first few stages of O_partial
            for stage in cutlass.range(self.stages - 1, unroll_full=True):
                if stage < num_splits:
                    load_O_partial(stage, stage)
                cute.arch.cp_async_commit_group()

            # ===============================
            # Step 3: Load and transpose LSE from smem to registers
            # ===============================

            # Wait for LSE and initial O partial stages to complete
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()
            # if cute.arch.thread_idx()[0] == 0:
            #     # cute.print_tensor(sLSE)
            #     for i in range(64):
            #         cute.printf("sLSE[%d, 0] = %f", i, sLSE[i, 0])
            # cute.arch.sync_threads()

            s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_slice(tidx)
            ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE)
            ts2rrLSE = cute.make_rmem_tensor_like(ts2rsLSE)
            cute.copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE)
            if const_expr(mLSE_temperature_partial is not None):
                ts2rsLSE_temperature = s2r_thr_copy_LSE.partition_S(sLSE_temperature)
                ts2rrLSE_temperature = cute.make_rmem_tensor_like(ts2rsLSE_temperature)
                cute.copy(
                    s2r_tiled_copy_LSE,
                    ts2rsLSE_temperature,
                    ts2rrLSE_temperature,
                )

            # ===============================
            # Step 4: Compute final LSE along split dimension
            # ===============================

            final_lse = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Float32)
            ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE)
            # We compute the max valid split for each row to short-circuit the computation later
            max_valid_split = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Int32)
            assert cute.size(ts2rrLSE, mode=[0]) == 1
            # Compute max, scales, and final LSE for each row. Invalid splits
            # have already been filled with -inf, so Step 5 can write the
            # kernel-native LSE_out directly.
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                # Find max LSE value across splits
                threads_per_col = const_expr(self.smem_threads_per_col_lse)
                lse_max = cute.arch.warp_reduction_max(
                    ts2rrLSE[None, None, m]
                    .load()
                    .reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
                    threads_in_group=threads_per_col,
                )
                # if cute.arch.thread_idx()[0] == 0: cute.printf(lse_max)
                # Find max valid split index
                max_valid_idx = -1
                for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                    if ts2rrLSE[0, s, m] != -Float32.inf:
                        max_valid_idx = ts2rcLSE[0, s, 0][0]  # Get split coordinate
                # if cute.arch.thread_idx()[0] < 32: cute.printf(max_valid_idx)
                max_valid_split[m] = cute.arch.warp_reduction_max(
                    max_valid_idx, threads_in_group=threads_per_col
                )
                # Compute exp scales and sum
                lse_max_cur = (
                    0.0 if lse_max == -Float32.inf else lse_max
                )  # In case all local LSEs are -inf
                LOG2_E = math.log2(math.e)
                lse_sum_cur = 0.0
                for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                    scale = cute.math.exp2(
                        ts2rrLSE[0, s, m] * LOG2_E - (lse_max_cur * LOG2_E), fastmath=True
                    )
                    lse_sum_cur += scale
                    ts2rrLSE[0, s, m] = scale  # Store scale for later use
                lse_sum_cur = cute.arch.warp_reduction_sum(
                    lse_sum_cur, threads_in_group=threads_per_col
                )
                # Normalize scales
                inv_sum = 0.0
                if max_valid_split[m] < 0 or lse_sum_cur == 0.0 or lse_sum_cur != lse_sum_cur:
                    final_lse[m] = -Float32.inf
                else:
                    final_lse[m] = cute.math.log(lse_sum_cur, fastmath=True) + lse_max
                    inv_sum = 1.0 / lse_sum_cur
                ts2rrLSE[None, None, m].store(ts2rrLSE[None, None, m].load() * inv_sum)
            # Store the scales exp(lse - lse_logsum) back to smem
            cute.copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE)

            if const_expr(mLSE_temperature_partial is not None):
                final_lse_temperature = cute.make_rmem_tensor(
                    cute.size(ts2rrLSE_temperature, mode=[2]), Float32)
                for m in cutlass.range(cute.size(ts2rrLSE_temperature, mode=[2]), unroll_full=True):
                    threads_per_col = const_expr(self.smem_threads_per_col_lse)
                    lse_temperature_max = cute.arch.warp_reduction_max(
                        ts2rrLSE_temperature[None, None, m]
                        .load()
                        .reduce(
                            cute.ReductionOp.MAX,
                            init_val=-Float32.inf,
                            reduction_profile=0,
                        ),
                        threads_in_group=threads_per_col,
                    )
                    lse_temperature_max_cur = (
                        0.0 if lse_temperature_max == -Float32.inf else lse_temperature_max
                    )
                    LOG2_E = math.log2(math.e)
                    lse_temperature_sum_cur = 0.0
                    for s in cutlass.range(
                        cute.size(ts2rrLSE_temperature, mode=[1]), unroll_full=True):
                        scale = cute.math.exp2(
                            ts2rrLSE_temperature[0, s, m] * LOG2_E
                            - (lse_temperature_max_cur * LOG2_E),
                            fastmath=True,
                        )
                        lse_temperature_sum_cur += scale
                    lse_temperature_sum_cur = cute.arch.warp_reduction_sum(
                        lse_temperature_sum_cur, threads_in_group=threads_per_col
                    )
                    if (
                        max_valid_split[m] < 0
                        or lse_temperature_sum_cur == 0.0
                        or lse_temperature_sum_cur != lse_temperature_sum_cur
                    ):
                        final_lse_temperature[m] = -Float32.inf
                    else:
                        final_lse_temperature[m] = (
                            cute.math.log(lse_temperature_sum_cur, fastmath=True)
                            + lse_temperature_max
                        )

            # Store max valid split to smem
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                if ts2rcLSE[0, 0, m][0] == 0:  # Only thread responsible for s=0 writes
                    mi = ts2rcLSE[0, 0, m][1]
                    if mi < self.tile_m:
                        sMaxValidSplit[mi] = max_valid_split[m]

            # ===============================
            # Step 5: Store final LSE to gmem
            # This writeback is the authoritative LSE_out returned by the
            # public Sparse Attention / Sparse Page Attention interface.
            # ===============================

            if const_expr(mLSE is not None):
                if const_expr(cu_seqlens is None):
                    mLSE_cur = mLSE[None, None, batch_idx]
                else:
                    mLSE_cur = cute.domain_offset((offset, 0), mLSE)
                if const_expr(mLSE_temperature is not None):
                    if const_expr(cu_seqlens is None):
                        mLSE_temperature_cur = mLSE_temperature[None, None, batch_idx]
                    else:
                        mLSE_temperature_cur = cute.domain_offset(
                            (offset, 0), mLSE_temperature)
                if k_block == 0:  # Only first k_block writes LSE when mLSE is provided
                    for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                        if ts2rcLSE[0, 0, m][0] == 0:  # Only thread responsible for s=0 writes
                            mi = ts2rcLSE[0, 0, m][1]
                            idx = m_block * self.tile_m + mi
                            if idx < max_idx:
                                m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                                mLSE_cur[m_idx, head_idx] = final_lse[m]
                                if const_expr(mLSE_temperature is not None):
                                    mLSE_temperature_cur[m_idx, head_idx] = (
                                        final_lse_temperature[m])

            # ===============================
            # Step 6: Read O_partial and accumulate final O
            # ===============================

            cute.arch.sync_threads()

            # Get max valid split for this thread
            thr_max_valid_split = sMaxValidSplit[tOcO[0, 0, 0][0]]
            for m in cutlass.range(1, cute.size(tOcO, mode=[1]), unroll_full=True):
                thr_max_valid_split = max(thr_max_valid_split, sMaxValidSplit[tOcO[0, m, 0][0]])

            tOrO_partial = cute.make_rmem_tensor_like(tOsO_partial[None, None, None, 0])
            tOrO = cute.make_rmem_tensor_like(tOrO_partial, Float32)
            tOrO.fill(0.0)

            stage_load = self.stages - 1
            stage_compute = 0

            # Main accumulation loop
            for s in cutlass.range(thr_max_valid_split + 1, unroll=4):
                # Get scales for this split
                scale = cute.make_rmem_tensor(num_rows, Float32)
                for m in cutlass.range(num_rows, unroll_full=True):
                    scale[m] = sLSE[s, tOcO[0, m, 0][0]]  # Get scale from smem

                # Load next stage if needed
                split_to_load = s + self.stages - 1
                if split_to_load <= thr_max_valid_split:
                    load_O_partial(split_to_load, stage_load)
                cute.arch.cp_async_commit_group()
                stage_load = 0 if stage_load == self.stages - 1 else stage_load + 1

                # Wait for the current stage to be ready
                cute.arch.cp_async_wait_group(self.stages - 1)
                # We don't need __syncthreads() because each thread is just reading its own data from smem
                # Copy from smem to registers
                cute.autovec_copy(tOsO_partial[None, None, None, stage_compute], tOrO_partial)
                stage_compute = 0 if stage_compute == self.stages - 1 else stage_compute + 1

                # Accumulate scaled partial results
                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= 0 and scale[m] > 0.0:
                        tOrO[None, m, None].store(
                            tOrO[None, m, None].load()
                            + scale[m] * tOrO_partial[None, m, None].load().to(Float32)
                        )

            # Flush any outstanding async-copy groups before the local Step-7
            # permutation buffer is read on the tail of the kernel.
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ===============================
            # Step 7: Write final O to gmem (fake→real via SMEM)
            # ===============================

            mO_cur = seqlen_info.offset_batch(mO, batch_idx, dim=3)
            if const_expr(cu_seqlens is None):
                mO_cur = mO[None, None, None, batch_idx]
            else:
                mO_cur = cute.domain_offset((offset, 0, 0), mO)
            mO_cur = utils.domain_offset_aligned((0, k_block * self.k_block_size, 0), mO_cur)
            num_vals = const_expr(cute.size(tOcO, mode=[0]))
            if const_expr(not use_pdl):
                # Direct / standalone calls don't participate in the K1->K2
                # dependency chain. Use a simple per-element real-column store
                # path here to keep mixed-shape launches stable.
                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= 0:
                        for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                            if const_expr(self.is_even_k) or tOpO[k]:
                                for v in cutlass.range(num_vals, unroll_full=True):
                                    fake_col = tOcO[v, 0, k][1]
                                    if const_expr(self.use_stg128_fp8_layout):
                                        real_col = stg128_fp8_fake_col_to_real_col(fake_col)
                                    elif const_expr(self.use_stg128_half_layout):
                                        real_col = stg128_half_fake_col_to_real_col(fake_col)
                                    else:
                                        real_col = stg128_fake_col_to_real_col(fake_col)
                                    o_val = tOrO[v, m, k]
                                    if const_expr(mOutputScale is not None):
                                        o_val = o_val * output_scale
                                    mO_cur[tOmidx[m], real_col, tOhidx[m]] = o_val.to(self.dtype)
            else:
                # 7a: fp32 accumulator -> output dtype SMEM with fake→real
                # permutation. The dedicated permutation buffer stays separate
                # from the O_partial pipeline staging buffer.
                sO_perm = sO_perm_buf

                if const_expr(self.dtype in [cutlass.BFloat16, cutlass.Float16]):
                    # O_partial uses a dtype-specific STG.128 fake layout, but
                    # sO_perm is in the final O dtype. For all supported fake
                    # layouts, adjacent fake pairs map to adjacent real columns,
                    # so write the final BF16/F16 O pair as one 32-bit SMEM store.
                    assert num_vals % 2 == 0
                    r2s_o_pair_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        cutlass.Int32,
                        num_bits_per_copy=32,
                    )
                    rO_pair_word = cute.make_rmem_tensor((1,), cutlass.Int32)
                    sO_perm_i32_base = cute.make_ptr(
                        dtype=cutlass.Int32,
                        value=sO_perm.iterator.toint(),
                        mem_space=sO_perm.iterator.memspace,
                        assumed_align=4,
                    )
                    sO_perm_i32_row_stride = Int32((self.k_block_size + 16) // 2)
                    for m in cutlass.range(num_rows, unroll_full=True):
                        row_local = tOcO[0, m, 0][0]
                        if tOhidx[m] >= 0:
                            for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                                for v_pair in cutlass.range(num_vals // 2, unroll_full=True):
                                    v = v_pair * 2
                                    fake_col = tOcO[v, 0, k][1]
                                    if const_expr(self.use_stg128_fp8_layout):
                                        real_col = stg128_fp8_fake_col_to_real_col(fake_col)
                                    elif const_expr(self.use_stg128_half_layout):
                                        real_col = stg128_half_fake_col_to_real_col(fake_col)
                                    else:
                                        real_col = stg128_fake_col_to_real_col(fake_col)
                                    o0 = tOrO[v, m, k]
                                    o1 = tOrO[v + 1, m, k]
                                    if const_expr(mOutputScale is not None):
                                        o0, o1 = cute.arch.mul_packed_f32x2(
                                            (o0, o1),
                                            (output_scale, output_scale),
                                        )
                                    rO_pair_word[0] = utils.cvt_f16x2_f32(o0, o1, self.dtype)
                                    smem_pair_ptr = cute.make_ptr(
                                        dtype=cutlass.Int32,
                                        value=(
                                            sO_perm_i32_base.toint()
                                            + Int64(
                                                row_local * sO_perm_i32_row_stride
                                                + real_col // Int32(2)
                                            )
                                            * Int64(4)
                                        ),
                                        mem_space=sO_perm.iterator.memspace,
                                        assumed_align=4,
                                    )
                                    sO_pair = cute.make_tensor(
                                        smem_pair_ptr,
                                        cute.make_layout((1,), stride=(1,)),
                                    )
                                    cute.copy(r2s_o_pair_atom, rO_pair_word, sO_pair)
                else:
                    # 7a: iterate over ALL val elements in mode[0].
                    # tOcO[v, m, k][1] gives different fake_col for each v.
                    r2s_o_scalar_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.dtype,
                        num_bits_per_copy=self.dtype.width,
                    )
                    rO_scalar = cute.make_rmem_tensor((1,), self.dtype)
                    for m in cutlass.range(num_rows, unroll_full=True):
                        row_local = tOcO[0, m, 0][0]
                        if tOhidx[m] >= 0:
                            for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                                for v in cutlass.range(num_vals, unroll_full=True):
                                    fake_col = tOcO[v, 0, k][1]
                                    if const_expr(self.use_stg128_fp8_layout):
                                        real_col = stg128_fp8_fake_col_to_real_col(fake_col)
                                    elif const_expr(self.use_stg128_half_layout):
                                        real_col = stg128_half_fake_col_to_real_col(fake_col)
                                    else:
                                        real_col = stg128_fake_col_to_real_col(fake_col)
                                    o_val = tOrO[v, m, k]
                                    if const_expr(mOutputScale is not None):
                                        o_val = o_val * output_scale
                                    rO_scalar[0] = o_val.to(self.dtype)
                                    smem_ptr = utils.elem_pointer(sO_perm, (row_local, real_col))
                                    smem_scalar_ptr = cute.make_ptr(
                                        dtype=self.dtype,
                                        value=smem_ptr.toint(),
                                        mem_space=sO_perm.iterator.memspace,
                                        assumed_align=self.dtype.width // 8,
                                    )
                                    sO_scalar = cute.make_tensor(
                                        smem_scalar_ptr,
                                        cute.make_layout((1,), stride=(1,)),
                                    )
                                    cute.copy(r2s_o_scalar_atom, rO_scalar, sO_scalar)

                cute.arch.sync_threads()

                # 7b: SMEM (real order, output dtype) → GMEM
                gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                tOcO_store = gmem_thr_copy_O.partition_D(cO)
                tOsO_store = gmem_thr_copy_O.partition_D(sO_perm)
                rO = cute.make_rmem_tensor(tOcO_store.shape, self.dtype)
                elems_per_store = const_expr(cute.size(gmem_tiled_copy_O.layout_tv_tiled[1]))
                num_store_rows = const_expr(cute.size(tOcO_store, mode=[1]))
                num_store_vals = const_expr(cute.size(tOcO_store, mode=[0]))
                tOpO_store = None
                if const_expr(not self.is_even_k):
                    tOpO_store = cute.make_rmem_tensor(cute.size(tOcO_store, mode=[2]), Boolean)
                    for k in cutlass.range(cute.size(tOpO_store), unroll_full=True):
                        tOpO_store[k] = (
                            tOcO_store[0, 0, k][1]
                            < mO_partial.shape[1] - k_block * self.k_block_size
                        )

                # Read output dtype from SMEM (now in real column order).
                for m in cutlass.range(num_store_rows, unroll_full=True):
                    for k in cutlass.range(cute.size(tOcO_store, mode=[2]), unroll_full=True):
                        if const_expr(self.is_even_k) or tOpO_store[k]:
                            cute.autovec_copy(tOsO_store[None, m, k], rO[None, m, k])

                # Write bf16 to GMEM using gmem_tiled_copy_O (same as original FA Step 7)
                for m in cutlass.range(num_store_rows, unroll_full=True):
                    row_local = tOcO_store[0, m, 0][0]
                    idx = m_block * self.tile_m + row_local
                    if idx < max_idx:
                        m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                        mO_cur_copy = cute.tiled_divide(
                            mO_cur[m_idx, None, head_idx], (elems_per_store,)
                        )
                        for k in cutlass.range(cute.size(tOcO_store, mode=[2]), unroll_full=True):
                            k_idx = tOcO_store[0, 0, k][1] // elems_per_store
                            if const_expr(self.is_even_k) or tOpO_store[k]:
                                cute.copy(gmem_thr_copy_O, rO[None, m, k], mO_cur_copy[None, k_idx])

    @cute.jit
    def load_O_partial(
        self,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        tOrOptr: cute.Tensor,
        tOsO_partial: cute.Tensor,
        tOhidx: cute.Tensor,
        tOSplitCount: cute.Tensor,
        tOpO: Optional[cute.Tensor],
        tOcO: cute.Tensor,
        mO_cur_partial_layout: cute.Layout,
        split: Int32,
        stage: Int32,
    ) -> None:
        elems_per_load = const_expr(cute.size(gmem_tiled_copy_O_partial.layout_tv_tiled[1]))
        tOsO_partial_cur = tOsO_partial[None, None, None, stage]
        for m in cutlass.range(cute.size(tOcO, [1]), unroll_full=True):
            if tOhidx[m] >= 0:
                o_gmem_ptr = cute.make_ptr(
                    tOsO_partial.element_type, tOrOptr[m], cute.AddressSpace.gmem, assumed_align=16
                )
                mO_partial_cur = cute.make_tensor(
                    o_gmem_ptr, cute.slice_(mO_cur_partial_layout, (0, None, None, 0))
                )
                mO_partial_cur_copy = cute.tiled_divide(mO_partial_cur, (elems_per_load,))
                for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                    k_idx = tOcO[0, 0, k][1] // elems_per_load
                    if split < tOSplitCount[m] and (const_expr(tOpO is None) or tOpO[k]):
                        cute.copy(
                            gmem_tiled_copy_O_partial,
                            mO_partial_cur_copy[None, k_idx, split],
                            tOsO_partial_cur[None, m, k],
                        )
                    else:
                        tOsO_partial_cur[None, m, k].fill(0)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _get_cutlass_dtype(torch_dtype: torch.dtype):
    if torch_dtype not in torch2cute_dtype_map:
        raise TypeError(f"Unsupported dtype: {torch_dtype}")
    return torch2cute_dtype_map[torch_dtype]


_combine_compile_cache = {}


def _get_cpasync_smem_layout_atom(dtype: Type[cutlass.Numeric], k_dim: int) -> cute.ComposedLayout:
    dtype_byte = const_expr(dtype.width // 8)
    bytes_per_row = const_expr(k_dim * dtype_byte)
    smem_k_block_size = (
        const_expr(
            128
            if bytes_per_row % 128 == 0
            else (64 if bytes_per_row % 64 == 0 else (32 if bytes_per_row % 32 == 0 else 16))
        )
        // dtype_byte
    )
    swizzle_bits = (
        4
        if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout(
            (8 if const_expr(k_dim % 32 == 0) else 16, smem_k_block_size),
            order=(1, 0),
        ),
    )


def combine_f32f8(
    o_partial_fake,
    lse_partial,
    o_out,
    lse_out,
    *,
    lse_temperature_partial=None,
    lse_temperature_out=None,
    cu_seqlens=None,
    seqused=None,
    split_counts=None,
    output_scale=None,
    use_pdl=False,
):
    """K2 on the original combine kernel, for FP32 / FP8 E4M3 partials.

    It also accepts BF16/FP16, but ``combine`` is the faster choice there.

    STG.128 fake-layout handling remains an internal implementation detail.
    When lse_out is provided, the kernel writes the final authoritative
    log-sum-exp for each query row/head directly into that tensor.

    Args:
        o_partial_fake:
            Batched: [num_splits, batch, Sq, head_q, dim]
            Varlen:  [num_splits, total_q, head_q, dim]
        lse_partial:
            Batched: [num_splits, batch, Sq, head_q]
            Varlen:  [num_splits, total_q, head_q]
        o_out:
            Batched: [batch, Sq, head_q, dim]
            Varlen:  [total_q, head_q, dim]
        lse_out:
            Batched: [batch, Sq, head_q]
            Varlen:  [total_q, head_q]
        lse_temperature_partial:
            Optional temperature-scaled LSE partial with the same shape as
            lse_partial.
        lse_temperature_out:
            Optional temperature-scaled final LSE with the same shape as
            lse_out.
        cu_seqlens: Optional [batch + 1] int32 for varlen-Q combine.
        seqused: Optional [batch] int32 effective lengths for combine.
        split_counts: Optional int32 rowwise valid split counts prepared from
            q2k metadata. Batched: [batch, seqlen, head_kv]. Varlen:
            [total_q, head_kv].
        output_scale: Optional fp32 tensor with at least one element. When
            provided, the final O accumulator is multiplied once before store.
        use_pdl: When True, wait on PDL dependencies from the producer K1
            kernel. When False, launch without PDL waits.
    """
    D = o_partial_fake.shape[-1]
    num_splits = o_partial_fake.shape[0]
    return_temperature_lse = (
        lse_temperature_partial is not None or lse_temperature_out is not None
    )
    if (lse_temperature_partial is None) != (lse_temperature_out is None):
        raise ValueError(
            "lse_temperature_partial and lse_temperature_out must either both be provided or both be None"
        )
    if lse_temperature_partial is not None and lse_temperature_partial.shape != lse_partial.shape:
        raise ValueError(
            "lse_temperature_partial must have the same shape as lse_partial, "
            f"got {lse_temperature_partial.shape} vs {lse_partial.shape}"
        )
    if lse_temperature_out is not None:
        if lse_out is None:
            raise ValueError("lse_temperature_out requires lse_out")
        if lse_temperature_out.shape != lse_out.shape:
            raise ValueError(
                "lse_temperature_out must have the same shape as lse_out, "
                f"got {lse_temperature_out.shape} vs {lse_out.shape}"
            )
        if lse_temperature_out.dtype != torch.float32 or lse_temperature_partial.dtype != torch.float32:
            raise TypeError("temperature LSE tensors must be torch.float32")

    partial_dtype = _get_cutlass_dtype(o_partial_fake.dtype)
    out_dtype = _get_cutlass_dtype(o_out.dtype)
    if output_scale is not None:
        if output_scale.dtype != torch.float32:
            raise TypeError(f"output_scale must be torch.float32, got {output_scale.dtype}")
        if output_scale.numel() < 1:
            raise ValueError("output_scale must contain at least one element")
        if output_scale.device != o_out.device:
            raise ValueError("output_scale must be on the same device as o_out")
        output_scale = output_scale.contiguous()
    if split_counts is not None:
        if split_counts.dtype != torch.int32:
            raise TypeError(f"split_counts must be torch.int32, got {split_counts.dtype}")
        if o_out.ndim == 4:
            if split_counts.ndim != 3:
                raise ValueError(
                    f"batched split_counts must have shape [batch, seqlen, head_kv], got {split_counts.shape}"
                )
            if split_counts.shape[:2] != o_out.shape[:2]:
                raise ValueError(
                    f"split_counts shape {split_counts.shape} must match batch/seqlen of o_out {o_out.shape}"
                )
        else:
            if cu_seqlens is None:
                raise ValueError("split_counts with varlen output requires cu_seqlens")
            if split_counts.ndim != 2:
                raise ValueError(
                    f"varlen split_counts must have shape [total_q, head_kv], got {split_counts.shape}"
                )
            if split_counts.shape[0] != o_out.shape[0]:
                raise ValueError(
                    f"split_counts total_q ({split_counts.shape[0]}) must match o_out total_q "
                    f"({o_out.shape[0]})"
                )
        if o_out.shape[-2] % split_counts.shape[-1] != 0:
            raise ValueError(
                f"o_out heads ({o_out.shape[-2]}) must be divisible by split_counts heads ({split_counts.shape[-1]})"
            )
        qheadperkv = o_out.shape[-2] // split_counts.shape[-1]
    else:
        qheadperkv = 1
    if cu_seqlens is not None:
        if cu_seqlens.dtype != torch.int32:
            raise TypeError(f"cu_seqlens must be torch.int32, got {cu_seqlens.dtype}")
        if cu_seqlens.ndim != 1:
            raise ValueError(f"cu_seqlens must be rank-1, got {cu_seqlens.shape}")
        if not cu_seqlens.is_contiguous():
            raise ValueError("cu_seqlens must be contiguous")
    if seqused is not None:
        if seqused.dtype != torch.int32:
            raise TypeError(f"seqused must be torch.int32, got {seqused.dtype}")
        if seqused.ndim != 1:
            raise ValueError(f"seqused must be rank-1, got {seqused.shape}")
        if not seqused.is_contiguous():
            raise ValueError("seqused must be contiguous")

    k_block_size = 128 if D > 64 else 64
    tile_m = 64
    has_cu_seqlens = cu_seqlens is not None
    has_seqused = seqused is not None
    has_lse = lse_out is not None
    has_split_counts = split_counts is not None
    has_output_scale = output_scale is not None
    capability = torch.cuda.get_device_capability(o_out.device)
    is_sm107 = capability == (10, 7)
    # min_blocks_per_mp > 1 makes the DSL set a preferred shared-memory
    # carveout sized against the portable 228 KB per SM. On SM107 the DSL
    # also opts the kernel into the oversized shared-memory mode, and the GPU
    # front end rejects a launch (Xid 13) whose occupancy needs more shared
    # memory than that carveout target; the 3-stage ring at 3 CTAs per SM
    # does. Rubin therefore leaves the register budget to ptxas.
    min_blocks_per_mp = 3 if has_output_scale and use_pdl and not is_sm107 else 0
    # Rubin (SM107) takes a 3-stage O_partial ring; Blackwell keeps stages=2
    # (see the occupancy note below).
    stages = 3 if is_sm107 else 2

    key = (
        "combine_f32f8",
        capability,
        stages,
        D,
        k_block_size,
        tile_m,
        num_splits,
        partial_dtype,
        out_dtype,
        has_cu_seqlens,
        has_seqused,
        has_lse,
        bool(return_temperature_lse),
        has_split_counts,
        has_output_scale,
        use_pdl,
        min_blocks_per_mp,
    )
    if key not in _combine_compile_cache:
        from src.common.aot_cache import try_load_aot, save_aot

        loaded = try_load_aot(key)
        if loaded is not None:
            _combine_compile_cache[key] = loaded
        else:
            from quack.compile_utils import make_fake_tensor

            kernel = SparseAttentionForwardCombineF32F8(
                dtype=out_dtype,
                dtype_partial=partial_dtype,
                head_dim=D,
                tile_m=tile_m,
                k_block_size=k_block_size,
                topk=num_splits,
                use_pdl=use_pdl,
                min_blocks_per_mp=min_blocks_per_mp,
                # Blackwell: stages=2 halves per-block SMEM (168 KB -> 103 KB)
                # -> 2 blocks/SM, theoretical occupancy 12.5% -> 25%. NCU DRAM
                # throughput 76.35% -> 88.64%. Runtime latency within noise
                # (kernel already at HBM bandwidth ceiling in practice) but the
                # cleaner SOL profile matters for downstream NCU comparison.
                # Rubin: stages=3 (selected above).
                stages=stages,
            )
            div = 128 // partial_dtype.width
            if has_cu_seqlens:
                total_q, nheads = (cute.sym_int64() for _ in range(2))
                mO_partial = make_fake_tensor(
                    partial_dtype, (num_splits, total_q, nheads, D), divisibility=div
                )
                mLSE_partial = make_fake_tensor(
                    Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=2
                )
                mO = make_fake_tensor(
                    out_dtype, (total_q, nheads, D), divisibility=128 // out_dtype.width
                )
                mLSE = (
                    make_fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=1)
                    if has_lse
                    else None
                )
                mLSE_temperature_partial = (
                    make_fake_tensor(
                        Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=2
                    )
                    if return_temperature_lse
                    else None
                )
                mLSE_temperature = (
                    make_fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=1)
                    if return_temperature_lse
                    else None
                )
            else:
                batch, sq, nheads = (cute.sym_int64() for _ in range(3))
                mO_partial = make_fake_tensor(
                    partial_dtype, (num_splits, batch, sq, nheads, D), divisibility=div
                )
                mLSE_partial = make_fake_tensor(
                    Float32, (num_splits, batch, sq, nheads), divisibility=1, leading_dim=3
                )
                mO = make_fake_tensor(
                    out_dtype, (batch, sq, nheads, D), divisibility=128 // out_dtype.width
                )
                mLSE = (
                    make_fake_tensor(Float32, (batch, sq, nheads), divisibility=1, leading_dim=2)
                    if has_lse
                    else None
                )
                mLSE_temperature_partial = (
                    make_fake_tensor(
                        Float32, (num_splits, batch, sq, nheads), divisibility=1, leading_dim=3
                    )
                    if return_temperature_lse
                    else None
                )
                mLSE_temperature = (
                    make_fake_tensor(Float32, (batch, sq, nheads), divisibility=1, leading_dim=2)
                    if return_temperature_lse
                    else None
                )
            if not has_split_counts:
                mSplitCounts = None
            elif has_cu_seqlens:
                total_q_ctr, nheads_kv = (cute.sym_int64() for _ in range(2))
                mSplitCounts = make_fake_tensor(
                    Int32, (total_q_ctr, nheads_kv), divisibility=1, leading_dim=1
                )
            else:
                nheads_kv = cute.sym_int64()
                mSplitCounts = make_fake_tensor(
                    Int32, (batch, sq, nheads_kv), divisibility=1, leading_dim=2
                )
            mOutputScale = (
                make_fake_tensor(Float32, (cute.sym_int64(),), divisibility=1, leading_dim=0)
                if has_output_scale
                else None
            )
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

            _combine_compile_cache[key] = cute.compile(
                kernel,
                mO_partial,
                mLSE_partial,
                mO,
                mLSE,
                mLSE_temperature_partial,
                mLSE_temperature,
                None
                if cu_seqlens is None
                else make_fake_tensor(Int32, (cute.sym_int64(),), divisibility=1, leading_dim=0),
                None
                if seqused is None
                else make_fake_tensor(Int32, (cute.sym_int64(),), divisibility=1, leading_dim=0),
                None,
                None,
                None,
                mSplitCounts,
                mOutputScale,
                Int32(qheadperkv),
                stream,
                options="--enable-tvm-ffi",
            )
            save_aot(key, _combine_compile_cache[key], sources=[__name__])

    with torch.cuda.nvtx.range("K2_Combine"):
        _combine_compile_cache[key](
            o_partial_fake,
            lse_partial,
            o_out,
            lse_out,
            lse_temperature_partial,
            lse_temperature_out,
            cu_seqlens,
            seqused,
            None,
            None,
            None,
            split_counts,
            output_scale,
            qheadperkv,
        )


# -----------------------------------------------------------------------------
# TMA-ring combine (combine, the default): BF16/FP16 partials
# -----------------------------------------------------------------------------

# One tile is 64 flat (query, head) rows of 256 B, a 16 KB chunk per split; four slots keep
# 64 KB in flight per CTA and leave room for three CTAs per SM.
TILE_M = 64
NUM_THREADS = 256
STAGES = 4
# Use a two-slot ring when the reads are not steered on SM107. Unsteered partials are
# interleaved across the memory locality domains, so half of each CTA's reads cross the
# inter-domain fabric, and on SM107 that fabric caps interleaved reads well below the local HBM
# rate. Keep in flight only what the fabric drains in one round trip: a deeper ring adds
# queueing that also delays the local half of the reads. Steered reads are domain-local and
# B200/B300 interleaved reads are HBM-bound, so both keep the four-slot ring.
STAGES_INTERLEAVED = 2
# The partial buffer is addressed as (dim, flat row, split); the TMA descriptor lowering is
# relied on for split strides below this many elements.
MAX_SPLIT_SLAB_ELEMENTS = 2**31


@dataclass(frozen=True)
class LocalityHints:
    """Steer every CTA to the memory locality domain of its SM.

    ``sm_domains`` holds the locality domain of each SM id (``%smid``) as int32, -1 where unknown.
    ``tile_counters`` holds one int32 per domain, the tile claims of a launch; they must be zero
    when the launch starts (the forward kernel zeroes them for the combine that follows it).
    ``page_bytes`` is the granularity at which the partial buffer alternates domains: page ``p``
    of every split slab lives in domain ``p % num_domains``, which a split stride that is a
    multiple of ``num_domains * page_bytes`` keeps true for every slab.
    """

    sm_domains: torch.Tensor
    tile_counters: torch.Tensor
    page_bytes: int

    @property
    def num_domains(self) -> int:
        return int(self.tile_counters.numel())


class SparseAttentionForwardCombine:
    """Merge the split partials of every (query, head) row through a TMA ring.

    The sparse forward kernels write one partial output and one partial log-sum-exp per
    (query, head, selected page) into ``[num_splits, total_q, heads, dim]`` and
    ``[num_splits, total_q, heads]`` buffers. This kernel merges the splits of every (query, head)
    row into the final output and LSE. Rows are independent, so the concatenated varlen rows form
    one flat scheduling domain: a CTA owns ``tile_m`` consecutive flat rows, which is one contiguous
    chunk of each split's slab, and streams the chunks of its splits through a ring of ``stages``
    shared-memory slots with one bulk-tensor copy per (tile, split).

    The partial output columns are in the STG.128 "fake" order the forward epilogue stores (a 4x4
    transpose of 32-bit pairs inside each 32-column group); the epilogue here undoes it with two
    XOR-shuffle stages before one coalesced 128-bit store per row.

    The split slabs of the partial buffer may be padded apart. On a GPU with several memory
    locality domains (Rubin uGPU) the caller stripes the buffer so page ``p`` of every slab lives in
    domain ``p % num_domains``, and hands the kernel the domain of every SM: a CTA then claims a tile
    whose pages live in the domain of its own SM, from per-domain counters, and takes a tile of
    another domain only once its own are gone.

    The output can also, or instead, be written as MXFP8: after the shuffles each lane holds 8
    consecutive BF16 columns of its row, so the four lanes of a pair group hold one aligned
    32-column block.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        head_dim: int,
        topk: int,
        tile_m: int = TILE_M,
        num_threads: int = NUM_THREADS,
        stages: int = STAGES,
        use_pdl: bool = False,
        min_blocks_per_mp: int = 0,
        num_domains: int = 0,
        lse_split_aligned: bool = True,
    ):
        """Forward combine kernel for split attention computation.

        :param dtype: output data type (BF16 or F16)
        :param dtype_partial: partial output data type (BF16 or F16)
        :param head_dim: head dimension; a tile row is one head's ``head_dim`` elements
        :param topk: exact number of split partials per row
        :param tile_m: flat (query, head) rows per CTA
        :param num_threads: threads per CTA
        :param stages: ring slots of ``tile_m`` partial rows each
        :param use_pdl: wait on the producer grid before reading its outputs
        :param min_blocks_per_mp: CTAs per SM ptxas must leave registers for (0 lets it choose)
        :param lse_split_aligned: the LSE partial slabs start 16-byte aligned (the flat row count is
            a multiple of 4), so one 2-D bulk-tensor copy loads a tile's LSEs of every split;
            otherwise the threads bring the tile in with 4-byte cp.async copies, since TMA needs
            a 16-byte aligned split stride and box start
        :param num_domains: memory locality domains the tiles are claimed by (0: tile = CTA index)
        """
        if dtype not in (cutlass.BFloat16, cutlass.Float16):
            raise TypeError("the combine writes BF16 or F16 outputs")
        if dtype_partial not in (cutlass.BFloat16, cutlass.Float16):
            raise TypeError("the combine reads BF16 or F16 partials")
        if not 1 <= topk <= 256:
            raise ValueError(f"topk must be in [1, 256], got {topk}")
        if num_threads % 32 != 0 or stages < 1:
            raise ValueError("num_threads must be a multiple of 32 and stages at least 1")
        if num_domains < 0 or num_domains == 1:
            raise ValueError("num_domains must be 0 (no steering) or at least 2")
        copy_bits = 128
        self.elems_per_copy = copy_bits // dtype_partial.width
        if head_dim % self.elems_per_copy != 0:
            raise ValueError("head_dim must be a multiple of one 16-byte copy")
        # Threads that share one tile row, 16 bytes of it each.
        self.threads_per_row = head_dim // self.elems_per_copy
        if num_threads % self.threads_per_row != 0:
            raise ValueError("num_threads must be a multiple of the threads per row")
        self.rows_per_pass = num_threads // self.threads_per_row
        if tile_m % self.rows_per_pass != 0:
            raise ValueError("tile_m must be a multiple of the rows one pass covers")
        # Threads that transpose one row of the LSE tile.
        if num_threads % tile_m != 0:
            raise ValueError("num_threads must be a multiple of tile_m")
        self.lse_threads_per_row = num_threads // tile_m
        if 32 % self.lse_threads_per_row != 0 or topk % self.lse_threads_per_row != 0:
            raise ValueError("the threads per LSE row must divide the warp and the split count")
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.topk = topk
        self.tile_m = tile_m
        self.num_threads = num_threads
        self.num_warps = num_threads // 32
        self.stages = stages
        self.use_pdl = use_pdl
        self.min_blocks_per_mp = min_blocks_per_mp
        self.num_domains = num_domains
        self.lse_split_aligned = lse_split_aligned

    def _setup_attributes(self):
        copy_bits = 128
        # One thread map serves the reads of a ring slot and the output stores: BF16 and F16
        # have the same width, so a thread's 16 bytes cover the same columns in both.
        thr_layout = cute.make_ordered_layout(
            (self.rows_per_pass, self.threads_per_row), order=(1, 0)
        )
        val_layout = cute.make_layout((1, self.elems_per_copy))
        self.tiled_copy_partial = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.dtype_partial, num_bits_per_copy=copy_bits
            ),
            thr_layout,
            val_layout,
        )
        self.tiled_copy_out = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=copy_bits),
            thr_layout,
            val_layout,
        )
        # LSE tile (split, row): consecutive lanes transpose the splits of one row.
        s2r_thr_layout = cute.make_ordered_layout(
            (self.lse_threads_per_row, self.tile_m), order=(0, 1)
        )
        self.s2r_tiled_copy_lse = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32),
            s2r_thr_layout,
            cute.make_layout(1),
        )
        # Unaligned LSE slabs: every thread brings topk / lse_threads_per_row splits of one row in
        # with 4-byte cp.async copies; consecutive threads take consecutive rows of a split.
        g2s_thr_layout = cute.make_ordered_layout(
            (self.tile_m, self.lse_threads_per_row), order=(0, 1)
        )
        self.g2s_tiled_copy_lse = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cpasync.CopyG2SOp(), Float32, num_bits_per_copy=Float32.width),
            g2s_thr_layout,
            cute.make_layout((1, self.topk // self.lse_threads_per_row)),
        )
        # The bulk copy writes the tile as a (row, split) box over the same bytes; the trailing
        # mode is the single stage the TMA partition idiom expects.
        self.smem_layout_lse = cute.make_ordered_layout((self.topk, self.tile_m), order=(1, 0))
        self.smem_layout_lse_tma = cute.make_ordered_layout(
            (self.tile_m, self.topk, 1), order=(0, 1, 2)
        )
        # Ring slot: row-major (row, dim); 16 threads read 16 B each of one 256 B row, which is
        # conflict-free without a swizzle. The TMA view is its (dim, row) transpose.
        self.smem_layout_o = cute.make_ordered_layout(
            (self.tile_m, self.head_dim, self.stages), order=(1, 0, 2)
        )
        self.smem_layout_o_tma = cute.make_ordered_layout(
            (self.head_dim, self.tile_m, self.stages), order=(0, 1, 2)
        )

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor] = None,
        mLSE_temperature_partial: Optional[cute.Tensor] = None,
        mLSE_temperature: Optional[cute.Tensor] = None,
        mSplitCounts: Optional[cute.Tensor] = None,
        mOutputScale: Optional[cute.Tensor] = None,
        qhead_per_kvhead: Int32 = Int32(1),
        mSmDomains: Optional[cute.Tensor] = None,
        mTileCounters: Optional[cute.Tensor] = None,
        tiles_per_page_log2: Int32 = Int32(0),
        mO_mxfp8: Optional[cute.Tensor] = None,
        mO_mxfp8_scale: Optional[cute.Tensor] = None,
        mxfp8_first_row: Int32 = Int32(0),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        if const_expr(mO_partial.element_type != self.dtype_partial):
            raise TypeError("O partial tensor must match dtype_partial")
        if const_expr(mO is not None and mO.element_type != self.dtype):
            raise TypeError("O tensor must match dtype")
        if const_expr((mO_mxfp8 is None) != (mO_mxfp8_scale is None)):
            raise ValueError("MXFP8 O data and scale tensors must both be provided or both be None")
        if const_expr(mO is None and mO_mxfp8 is None):
            raise ValueError("the combine needs O and/or its MXFP8 form")
        if const_expr(mO_mxfp8 is not None):
            if const_expr(self.dtype is not cutlass.BFloat16 or self.elems_per_copy != 8):
                raise ValueError("MXFP8 O needs a BF16 output and 8 values per thread")
            if const_expr(self.head_dim != 128):
                raise ValueError("MXFP8 O needs head_dim 128 (4 MXFP8 blocks per 16-lane row)")
            if const_expr(mO_mxfp8.element_type != cutlass.Uint8):
                raise TypeError("MXFP8 O data must be uint8")
            if const_expr(mO_mxfp8_scale.element_type != cutlass.Uint8):
                raise TypeError("MXFP8 O scale must be uint8")
            if const_expr(len(mO_mxfp8.shape) != 3):
                raise ValueError("MXFP8 O data must be (total_q, nheads, headdim)")
        if const_expr(mLSE_partial.element_type != Float32):
            raise TypeError("LSE partial tensor must be Float32")
        if const_expr(mLSE is not None and mLSE.element_type != Float32):
            raise TypeError("LSE tensor must be Float32")
        if const_expr((mLSE_temperature_partial is None) != (mLSE_temperature is None)):
            raise ValueError(
                "temperature LSE partial and output tensors must either both be provided or both be None"
            )
        if const_expr(
            mLSE_temperature_partial is not None
            and (
                mLSE_temperature_partial.element_type != Float32
                or mLSE_temperature.element_type != Float32
            )
        ):
            raise TypeError("temperature LSE tensors must be Float32")
        if const_expr(len(mO_partial.shape) != 4):
            raise ValueError("O partial tensor must be (num_splits, total_q, nheads, headdim)")
        if const_expr(len(mLSE_partial.shape) != 3):
            raise ValueError("LSE partial tensor must be (num_splits, total_q, nheads)")
        if const_expr(mO is not None and len(mO.shape) != 3):
            raise ValueError("O tensor must be (total_q, nheads, headdim)")
        if const_expr(mLSE is not None and len(mLSE.shape) != 2):
            raise ValueError("LSE tensor must be (total_q, nheads)")
        if const_expr(
            mLSE_temperature_partial is not None
            and (len(mLSE_temperature_partial.shape) != 3 or len(mLSE_temperature.shape) != 2)
        ):
            raise ValueError("temperature LSE tensors must match the LSE partial and output shapes")
        if const_expr(mSplitCounts is not None):
            if const_expr(mSplitCounts.element_type != Int32):
                raise TypeError("split_counts tensor must be Int32")
            if const_expr(len(mSplitCounts.shape) != 2):
                raise ValueError("split_counts tensor must be (total_q, nheads_kv)")
        if const_expr(mOutputScale is not None and mOutputScale.element_type != Float32):
            raise TypeError("output_scale tensor must be Float32")
        if const_expr((mSmDomains is None) != (mTileCounters is None)):
            raise ValueError("sm_domains and tile_counters must either both be provided or both be None")
        if const_expr((mTileCounters is not None) != (self.num_domains > 0)):
            raise ValueError("tile_counters are required exactly when num_domains is set")
        if const_expr(
            mTileCounters is not None
            and (mSmDomains.element_type != Int32 or mTileCounters.element_type != Int32)
        ):
            raise TypeError("sm_domains and tile_counters must be Int32")

        mO_partial = assume_tensor_aligned(mO_partial)
        if const_expr(mO is not None):
            mO = assume_tensor_aligned(mO)
        num_head = mO_partial.shape[2]
        total_flat = mO_partial.shape[1] * num_head
        self._setup_attributes()

        # (num_splits, total_q, nheads, headdim) with contiguous slabs as (headdim, flat row,
        # split): a (headdim, tile_m) box is one tile of one split. The slabs keep their own
        # stride, which may be padded.
        mO_partial_tma = cute.make_tensor(
            mO_partial.iterator,
            cute.make_layout(
                (self.head_dim, total_flat, self.topk),
                stride=(1, self.head_dim, mO_partial.stride[0]),
            ),
        )
        tma_atom_O_partial, mO_partial_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mO_partial_tma,
            cute.select(self.smem_layout_o_tma, mode=[0, 1]),
            (self.head_dim, self.tile_m),
        )
        num_m_blocks = cute.ceil_div(total_flat, self.tile_m)

        def make_lse_tma(mLSE_src: cute.Tensor):
            """The contiguous (num_splits, total_q, nheads) LSE partials as (flat row, split). With
            aligned slabs, a TMA atom whose (tile_m, num_splits) box is a CTA's whole LSE tile and
            its coordinate tensor; otherwise no atom and the plain view, which the threads copy
            with cp.async (TMA rejects a split stride or a box start that is not 16-byte aligned)."""
            mLSE_2d = cute.make_tensor(
                mLSE_src.iterator,
                cute.make_layout((total_flat, self.topk), stride=(1, total_flat)),
            )
            if const_expr(not self.lse_split_aligned):
                return None, mLSE_2d
            return cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mLSE_2d,
                cute.select(self.smem_layout_lse_tma, mode=[0, 1]),
                (self.tile_m, self.topk),
            )

        tma_atom_LSE, mLSE_partial_tma = make_lse_tma(mLSE_partial)
        tma_atom_LSE_temperature = None
        mLSE_temperature_partial_tma = None
        if const_expr(mLSE_temperature_partial is not None):
            tma_atom_LSE_temperature, mLSE_temperature_partial_tma = make_lse_tma(
                mLSE_temperature_partial
            )

        lse_tile_elems = cute.cosize(self.smem_layout_lse)

        @cute.struct
        class SharedStorage:
            sLSE: cute.struct.Align[cute.struct.MemRange[Float32, lse_tile_elems], 128]
            sLSETemperature: cute.struct.Align[
                cute.struct.MemRange[
                    Float32, lse_tile_elems if mLSE_temperature_partial is not None else 1
                ],
                128,
            ]
            sBlockMaxValidSplit: cute.struct.Align[
                cute.struct.MemRange[Int32, self.num_warps], 128
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.dtype_partial, cute.cosize(self.smem_layout_o)], 128
            ]
            sRingMbar: cute.struct.Align[cute.struct.MemRange[Int64, self.stages], 8]
            sLseMbar: cute.struct.Align[cute.struct.MemRange[Int64, 1], 8]
            sTile: cute.struct.Align[cute.struct.MemRange[Int32, 1], 8]

        grid_dim = (num_m_blocks, 1, 1)
        self.kernel(
            mO_partial_tma,
            mLSE_partial_tma,
            mLSE_temperature_partial_tma,
            mO,
            mLSE,
            mLSE_temperature,
            mSplitCounts,
            mOutputScale,
            qhead_per_kvhead,
            mSmDomains,
            mTileCounters,
            tiles_per_page_log2,
            mO_mxfp8,
            mO_mxfp8_scale,
            mxfp8_first_row,
            tma_atom_O_partial,
            tma_atom_LSE,
            tma_atom_LSE_temperature,
            SharedStorage,
            self.smem_layout_lse,
            self.smem_layout_lse_tma,
            self.smem_layout_o,
            self.smem_layout_o_tma,
            self.tiled_copy_partial,
            self.tiled_copy_out,
            self.s2r_tiled_copy_lse,
            self.g2s_tiled_copy_lse,
            FastDivmodDivisor(num_head),
            self.use_pdl,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mO_partial_tma: cute.Tensor,
        mLSE_partial_tma: cute.Tensor,
        mLSE_temperature_partial_tma: Optional[cute.Tensor],
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mLSE_temperature: Optional[cute.Tensor],
        mSplitCounts: Optional[cute.Tensor],
        mOutputScale: Optional[cute.Tensor],
        qhead_per_kvhead: Int32,
        mSmDomains: Optional[cute.Tensor],
        mTileCounters: Optional[cute.Tensor],
        tiles_per_page_log2: Int32,
        mO_mxfp8: Optional[cute.Tensor],
        mO_mxfp8_scale: Optional[cute.Tensor],
        mxfp8_first_row: Int32,
        tma_atom_O_partial: cute.CopyAtom,
        tma_atom_LSE: Optional[cute.CopyAtom],
        tma_atom_LSE_temperature: Optional[cute.CopyAtom],
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout,
        smem_layout_lse_tma: cute.Layout,
        smem_layout_o: cute.Layout,
        smem_layout_o_tma: cute.Layout,
        tiled_copy_partial: cute.TiledCopy,
        tiled_copy_out: cute.TiledCopy,
        s2r_tiled_copy_lse: cute.TiledCopy,
        g2s_tiled_copy_lse: cute.TiledCopy,
        head_divmod: FastDivmodDivisor,
        use_pdl: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        m_block, _, _ = cute.arch.block_idx()
        total_tiles, _, _ = cute.arch.grid_dim()
        has_mxfp8 = const_expr(mO_mxfp8 is not None)
        mO_shape = mO_mxfp8.shape if has_mxfp8 else mO.shape
        total_flat = mO_shape[0] * mO_shape[1]
        has_temperature = const_expr(mLSE_temperature_partial_tma is not None)
        steer_tiles = const_expr(mTileCounters is not None)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sLSE_tma = storage.sLSE.get_tensor(smem_layout_lse_tma)
        sLSE_temperature = None
        sLSE_temperature_tma = None
        if const_expr(has_temperature):
            sLSE_temperature = storage.sLSETemperature.get_tensor(smem_layout_lse)
            sLSE_temperature_tma = storage.sLSETemperature.get_tensor(smem_layout_lse_tma)
        sBlockMaxValidSplit = storage.sBlockMaxValidSplit.get_tensor((self.num_warps,))
        sO = storage.sO.get_tensor(smem_layout_o)
        sO_tma = storage.sO.get_tensor(smem_layout_o_tma)
        ring_mbar = storage.sRingMbar.data_ptr()
        lse_mbar = storage.sLseMbar.data_ptr()
        sTile = storage.sTile.get_tensor((1,))

        # Barriers are CTA-local state, and the tile claim depends on nothing the producer grid
        # writes (it zeroed the counters at its start): both happen while that grid is still
        # draining.
        if tidx == 0:
            for stage in cutlass.range(self.stages, unroll_full=True):
                cute.arch.mbarrier_init(ring_mbar + stage, 1)
            cute.arch.mbarrier_init(lse_mbar, 1 if self.lse_split_aligned else self.num_threads)
            cute.arch.mbarrier_init_fence()
            if const_expr(steer_tiles):
                sTile[0] = self.claim_tile(
                    mSmDomains, mTileCounters, total_tiles, tiles_per_page_log2
                )
        cute.arch.sync_threads()
        if const_expr(steer_tiles):
            m_block = sTile[0]
        tile_begin = m_block * self.tile_m
        if const_expr(use_pdl):
            cute.arch.griddepcontrol_wait()

        # This CTA's tile of every split, and its LSE tile(s).
        gO_partial = cute.local_tile(
            mO_partial_tma, (self.head_dim, self.tile_m), (0, None, None)
        )
        tOsO_tma, tOgO_tma = cpasync.tma_partition(
            tma_atom_O_partial,
            0,
            cute.make_layout(1),
            cute.group_modes(sO_tma, 0, 2),
            cute.group_modes(gO_partial, 0, 2),
        )
        o_tma_bytes = const_expr(
            cute.size_in_bytes(self.dtype_partial, cute.select(smem_layout_o_tma, mode=[0, 1]))
        )
        # The LSE tile: with aligned slabs one (tile_m, num_splits) TMA box per tile; otherwise
        # 4-byte cp.async copies by every thread into the same smem bytes, completing the same
        # barrier with one arrive per thread.
        if const_expr(self.lse_split_aligned):
            gLSE = cute.local_tile(mLSE_partial_tma, (self.tile_m, self.topk), (None, 0))
            tLSEsLSE_tma, tLSEgLSE_tma = cpasync.tma_partition(
                tma_atom_LSE,
                0,
                cute.make_layout(1),
                cute.group_modes(sLSE_tma, 0, 2),
                cute.group_modes(gLSE, 0, 2),
            )
            if const_expr(has_temperature):
                gLSE_temperature = cute.local_tile(
                    mLSE_temperature_partial_tma, (self.tile_m, self.topk), (None, 0)
                )
                tLSEsLSE_temperature_tma, tLSEgLSE_temperature_tma = cpasync.tma_partition(
                    tma_atom_LSE_temperature,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sLSE_temperature_tma, 0, 2),
                    cute.group_modes(gLSE_temperature, 0, 2),
                )
            lse_tma_bytes = const_expr(
                cute.size_in_bytes(Float32, cute.select(smem_layout_lse_tma, mode=[0, 1]))
            )
        else:
            g2s_thr_copy_lse = g2s_tiled_copy_lse.get_slice(tidx)
            row_in_tile = g2s_thr_copy_lse.partition_S(
                cute.make_identity_tensor((self.tile_m, self.topk))
            )[0, 0, 0][0]
            if tile_begin + row_in_tile < total_flat:
                gLSE_tile = cute.local_tile(mLSE_partial_tma, (self.tile_m, self.topk), (m_block, 0))
                cute.copy(
                    g2s_tiled_copy_lse,
                    g2s_thr_copy_lse.partition_S(gLSE_tile),
                    g2s_thr_copy_lse.partition_D(sLSE_tma[None, None, 0]),
                )
                if const_expr(has_temperature):
                    gLSE_temperature_tile = cute.local_tile(
                        mLSE_temperature_partial_tma, (self.tile_m, self.topk), (m_block, 0)
                    )
                    cute.copy(
                        g2s_tiled_copy_lse,
                        g2s_thr_copy_lse.partition_S(gLSE_temperature_tile),
                        g2s_thr_copy_lse.partition_D(sLSE_temperature_tma[None, None, 0]),
                    )
            cute.arch.cp_async_mbarrier_arrive_noinc(lse_mbar)
        primed_stages = const_expr(min(self.stages, self.topk))
        # One warp puts the LSE tile and the first ring slots in flight; they land while the
        # CTA decodes its rows.
        if warp_idx == 0:
            if const_expr(self.lse_split_aligned):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        lse_mbar, lse_tma_bytes * (2 if has_temperature else 1)
                    )
                cute.copy(
                    tma_atom_LSE,
                    tLSEgLSE_tma[None, m_block],
                    tLSEsLSE_tma[None, 0],
                    tma_bar_ptr=lse_mbar,
                )
                if const_expr(has_temperature):
                    cute.copy(
                        tma_atom_LSE_temperature,
                        tLSEgLSE_temperature_tma[None, m_block],
                        tLSEsLSE_temperature_tma[None, 0],
                        tma_bar_ptr=lse_mbar,
                    )
            for stage in cutlass.range(primed_stages, unroll_full=True):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(ring_mbar + stage, o_tma_bytes)
                cute.copy(
                    tma_atom_O_partial,
                    tOgO_tma[None, m_block, Int32(stage)],
                    tOsO_tma[None, stage],
                    tma_bar_ptr=ring_mbar + stage,
                )

        cO = cute.make_identity_tensor((self.tile_m, self.head_dim))
        thr_copy_partial = tiled_copy_partial.get_slice(tidx)
        tOcO = thr_copy_partial.partition_D(cO)
        tOsO = thr_copy_partial.partition_D(sO)
        num_rows = const_expr(cute.size(tOcO, mode=[1]))

        # The LSE tile holds stale values past a row's valid split count and past the last row:
        # fetch each row's count before the wait and mask after the transpose.
        cLSE = cute.make_identity_tensor((self.topk, self.tile_m))
        s2r_thr_copy_lse = s2r_tiled_copy_lse.get_slice(tidx)
        ts2rsLSE = s2r_thr_copy_lse.partition_S(sLSE)
        ts2rcLSE = s2r_thr_copy_lse.partition_S(cLSE)
        ts2rrLSE = cute.make_rmem_tensor_like(ts2rsLSE)
        lse_rows = const_expr(cute.size(ts2rrLSE, mode=[2]))
        lse_splits = const_expr(cute.size(ts2rrLSE, mode=[1]))
        lse_row_count = cute.make_rmem_tensor(lse_rows, Int32)
        for m in cutlass.range(lse_rows, unroll_full=True):
            row_idx = tile_begin + ts2rcLSE[0, 0, m][1]
            lse_row_count[m] = Int32(0)
            if row_idx < total_flat:
                if const_expr(mSplitCounts is not None):
                    row_q, row_h = divmod(row_idx, head_divmod)
                    lse_row_count[m] = mSplitCounts[row_q, row_h // qhead_per_kvhead]
                else:
                    lse_row_count[m] = Int32(self.topk)
        cute.arch.mbarrier_wait(lse_mbar, 0)
        cute.copy(s2r_tiled_copy_lse, ts2rsLSE, ts2rrLSE)
        if const_expr(has_temperature):
            ts2rsLSE_temperature = s2r_thr_copy_lse.partition_S(sLSE_temperature)
            ts2rrLSE_temperature = cute.make_rmem_tensor_like(ts2rsLSE_temperature)
            cute.copy(s2r_tiled_copy_lse, ts2rsLSE_temperature, ts2rrLSE_temperature)
        for m in cutlass.range(lse_rows, unroll_full=True):
            for sp in cutlass.range(lse_splits, unroll_full=True):
                if ts2rcLSE[0, sp, 0][0] >= lse_row_count[m]:
                    ts2rrLSE[0, sp, m] = -Float32.inf
                    if const_expr(has_temperature):
                        ts2rrLSE_temperature[0, sp, m] = -Float32.inf

        # Per row: the max LSE over the valid splits, the split weights exp(lse - max) normalized
        # by their sum (written back over the LSE tile), the final LSE, and the last valid split.
        final_lse = cute.make_rmem_tensor(lse_rows, Float32)
        max_valid_split = cute.make_rmem_tensor(lse_rows, Int32)
        threads_per_row = const_expr(self.lse_threads_per_row)
        LOG2_E = math.log2(math.e)
        for m in cutlass.range(lse_rows, unroll_full=True):
            lse_max = cute.arch.warp_reduction_max(
                ts2rrLSE[None, None, m]
                .load()
                .reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
                threads_in_group=threads_per_row,
            )
            max_valid_idx = -1
            for sp in cutlass.range(lse_splits, unroll_full=True):
                if ts2rrLSE[0, sp, m] != -Float32.inf:
                    max_valid_idx = ts2rcLSE[0, sp, 0][0]
            max_valid_split[m] = cute.arch.warp_reduction_max(
                max_valid_idx, threads_in_group=threads_per_row
            )
            # A row without valid splits has max -inf; its weights are exp2(-inf) = 0.
            lse_max_cur = 0.0 if lse_max == -Float32.inf else lse_max
            lse_sum_cur = 0.0
            for sp in cutlass.range(lse_splits, unroll_full=True):
                scale = cute.math.exp2(
                    ts2rrLSE[0, sp, m] * LOG2_E - (lse_max_cur * LOG2_E), fastmath=True
                )
                lse_sum_cur += scale
                ts2rrLSE[0, sp, m] = scale
            lse_sum_cur = cute.arch.warp_reduction_sum(
                lse_sum_cur, threads_in_group=threads_per_row
            )
            inv_sum = 0.0
            if max_valid_split[m] < 0 or lse_sum_cur == 0.0 or lse_sum_cur != lse_sum_cur:
                final_lse[m] = -Float32.inf
            else:
                final_lse[m] = cute.math.log(lse_sum_cur, fastmath=True) + lse_max
                inv_sum = 1.0 / lse_sum_cur
            ts2rrLSE[None, None, m].store(ts2rrLSE[None, None, m].load() * inv_sum)
        cute.copy(s2r_tiled_copy_lse, ts2rrLSE, ts2rsLSE)

        if const_expr(has_temperature):
            final_lse_temperature = cute.make_rmem_tensor(lse_rows, Float32)
            for m in cutlass.range(lse_rows, unroll_full=True):
                lse_temperature_max = cute.arch.warp_reduction_max(
                    ts2rrLSE_temperature[None, None, m]
                    .load()
                    .reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
                    threads_in_group=threads_per_row,
                )
                lse_temperature_max_cur = (
                    0.0 if lse_temperature_max == -Float32.inf else lse_temperature_max
                )
                lse_temperature_sum_cur = 0.0
                for sp in cutlass.range(lse_splits, unroll_full=True):
                    lse_temperature_sum_cur += cute.math.exp2(
                        ts2rrLSE_temperature[0, sp, m] * LOG2_E
                        - (lse_temperature_max_cur * LOG2_E),
                        fastmath=True,
                    )
                lse_temperature_sum_cur = cute.arch.warp_reduction_sum(
                    lse_temperature_sum_cur, threads_in_group=threads_per_row
                )
                if (
                    max_valid_split[m] < 0
                    or lse_temperature_sum_cur == 0.0
                    or lse_temperature_sum_cur != lse_temperature_sum_cur
                ):
                    final_lse_temperature[m] = -Float32.inf
                else:
                    final_lse_temperature[m] = (
                        cute.math.log(lse_temperature_sum_cur, fastmath=True)
                        + lse_temperature_max
                    )

        # The ring is shared by the CTA, so the split loop runs to the tile's deepest valid
        # split: one slot per warp, reduced after the barrier below.
        thread_max_valid_split = max_valid_split[0]
        for m in cutlass.range(1, lse_rows, unroll_full=True):
            thread_max_valid_split = max(thread_max_valid_split, max_valid_split[m])
        warp_max_valid_split = cute.arch.warp_reduction_max(
            thread_max_valid_split, threads_in_group=32
        )
        if tidx % 32 == 0:
            sBlockMaxValidSplit[tidx // 32] = warp_max_valid_split

        if const_expr(mLSE is not None):
            for m in cutlass.range(lse_rows, unroll_full=True):
                if ts2rcLSE[0, 0, m][0] == 0:  # the lane holding split 0 writes the row
                    row_idx = tile_begin + ts2rcLSE[0, 0, m][1]
                    if row_idx < total_flat:
                        row_q, row_h = divmod(row_idx, head_divmod)
                        mLSE[row_q, row_h] = final_lse[m]
                        if const_expr(mLSE_temperature is not None):
                            mLSE_temperature[row_q, row_h] = final_lse_temperature[m]

        cute.arch.sync_threads()
        block_max_valid_split = sBlockMaxValidSplit[0]
        for wi in cutlass.range(1, self.num_warps, unroll_full=True):
            block_max_valid_split = max(block_max_valid_split, sBlockMaxValidSplit[wi])

        # Accumulate the weighted partials in FP32. A row's weight is zero for its invalid splits
        # and for every split of a row past the last one, so stale partial rows never reach the
        # sum.
        tOrO_partial = cute.make_rmem_tensor_like(tOsO[None, None, None, 0])
        tOrO = cute.make_rmem_tensor_like(tOrO_partial, Float32)
        tOrO.fill(0.0)
        for s in cutlass.range(block_max_valid_split + 1, unroll=2):
            stage = s % Int32(self.stages)
            phase = (s // Int32(self.stages)) & Int32(1)
            scale = cute.make_rmem_tensor(num_rows, Float32)
            for m in cutlass.range(num_rows, unroll_full=True):
                scale[m] = sLSE[s, tOcO[0, m, 0][0]]
            cute.arch.mbarrier_wait(ring_mbar + stage, phase)
            cute.autovec_copy(tOsO[None, None, None, stage], tOrO_partial)
            for m in cutlass.range(num_rows, unroll_full=True):
                if scale[m] > 0.0:
                    tOrO[None, m, None].store(
                        tOrO[None, m, None].load()
                        + scale[m] * tOrO_partial[None, m, None].load().to(Float32)
                    )
            # Every thread has copied this slot to registers before it is refilled.
            cute.arch.sync_threads()
            split_to_load = s + Int32(self.stages)
            if warp_idx == 0 and split_to_load <= block_max_valid_split:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(ring_mbar + stage, o_tma_bytes)
                cute.copy(
                    tma_atom_O_partial,
                    tOgO_tma[None, m_block, split_to_load],
                    tOsO_tma[None, stage],
                    tma_bar_ptr=ring_mbar + stage,
                )

        # Primed slots the split loop did not consume still have a bulk copy in flight; wait for
        # them so none lands after the CTA has exited.
        for stage in cutlass.range(primed_stages, unroll_full=True):
            if Int32(stage) > block_max_valid_split:
                cute.arch.mbarrier_wait(ring_mbar + stage, 0)

        # The partial columns are a 4x4 transpose of 32-bit pairs inside each 32-column group:
        # undo it with two XOR-shuffle stages across the four lanes that share the group, then
        # keep the coalesced 128-bit store. No shared memory, no barrier.
        output_scale = Float32(1.0)
        if const_expr(mOutputScale is not None):
            output_scale = mOutputScale[0]
        thr_copy_out = tiled_copy_out.get_slice(tidx)
        tOcO_store = thr_copy_out.partition_D(cO)
        rO = cute.make_rmem_tensor(tOcO_store.shape, self.dtype)
        elems_per_store = const_expr(cute.size(tiled_copy_out.layout_tv_tiled[1]))
        num_vals = const_expr(cute.size(tOcO, mode=[0]))
        num_store_k = const_expr(cute.size(tOcO_store, mode=[2]))
        assert num_vals == 8 and cute.size(tOcO_store, mode=[1]) == num_rows
        lane_in_pair_group = tidx % Int32(4)
        if const_expr(has_mxfp8):
            # Per row: this lane's 8 E4M3 bytes and its block's scale.
            rQ = cute.make_rmem_tensor((num_rows, num_store_k, 2), cutlass.Uint32)
            rE = cute.make_rmem_tensor((num_rows, num_store_k), cutlass.Uint32)
            scale_base = mO_mxfp8_scale.iterator.toint()
            scale_cols = Int32(mO_shape[1]) * Int32(self.head_dim // mx.MXFP8_BLOCK)
        for m in cutlass.range(num_rows, unroll_full=True):
            for k in cutlass.range(num_store_k, unroll_full=True):
                o0, o1 = tOrO[0, m, k], tOrO[1, m, k]
                o2, o3 = tOrO[2, m, k], tOrO[3, m, k]
                o4, o5 = tOrO[4, m, k], tOrO[5, m, k]
                o6, o7 = tOrO[6, m, k], tOrO[7, m, k]
                if const_expr(mOutputScale is not None):
                    o0, o1 = cute.arch.mul_packed_f32x2((o0, o1), (output_scale, output_scale))
                    o2, o3 = cute.arch.mul_packed_f32x2((o2, o3), (output_scale, output_scale))
                    o4, o5 = cute.arch.mul_packed_f32x2((o4, o5), (output_scale, output_scale))
                    o6, o7 = cute.arch.mul_packed_f32x2((o6, o7), (output_scale, output_scale))
                word0 = utils.cvt_f16x2_f32(o0, o1, self.dtype)
                word1 = utils.cvt_f16x2_f32(o2, o3, self.dtype)
                word2 = utils.cvt_f16x2_f32(o4, o5, self.dtype)
                word3 = utils.cvt_f16x2_f32(o6, o7, self.dtype)

                # The transpose stays inside a 32-value block, so the block's amax and each
                # value's E4M3 are the same on either side of it. Pairs of E4M3 fit two to a word:
                # transposed after quantizing it takes 3 shuffles, not the 8 of BF16 words.
                if const_expr(has_mxfp8):
                    # Every lane runs the block reduction; rows past the last one hold zeros.
                    q0, q1, e = mx.quantize_bf16x8_lane_of_4(word0, word1, word2, word3)
                    rQ[m, k, 0], rQ[m, k, 1] = mx.transpose_e4m3_pairs_lane_of_4(
                        q0, q1, lane_in_pair_group
                    )
                    rE[m, k] = e
                if const_expr(mO is not None):
                    partner0 = cute.arch.shuffle_sync_bfly(word1, offset=1)
                    partner1 = cute.arch.shuffle_sync_bfly(word0, offset=1)
                    partner2 = cute.arch.shuffle_sync_bfly(word3, offset=1)
                    partner3 = cute.arch.shuffle_sync_bfly(word2, offset=1)
                    stage0, stage1 = word0, word1
                    stage2, stage3 = word2, word3
                    if lane_in_pair_group % Int32(2) != Int32(0):
                        stage0, stage2 = partner0, partner2
                    if lane_in_pair_group % Int32(2) != Int32(1):
                        stage1, stage3 = partner1, partner3

                    partner0 = cute.arch.shuffle_sync_bfly(stage2, offset=2)
                    partner1 = cute.arch.shuffle_sync_bfly(stage3, offset=2)
                    partner2 = cute.arch.shuffle_sync_bfly(stage0, offset=2)
                    partner3 = cute.arch.shuffle_sync_bfly(stage1, offset=2)
                    out0, out1 = stage0, stage1
                    out2, out3 = stage2, stage3
                    if lane_in_pair_group // Int32(2) != Int32(0):
                        out0, out1 = partner0, partner1
                    if lane_in_pair_group // Int32(2) != Int32(1):
                        out2, out3 = partner2, partner3

                    rO_pairs = cute.recast_tensor(rO[None, m, k], Int32)
                    rO_pairs[0] = out0
                    rO_pairs[1] = out1
                    rO_pairs[2] = out2
                    rO_pairs[3] = out3

        for m in cutlass.range(num_rows, unroll_full=True):
            idx = tile_begin + tOcO_store[0, m, 0][0]
            if idx < total_flat:
                row_q, row_h = divmod(idx, head_divmod)
                if const_expr(mO is not None):
                    mO_row = cute.tiled_divide(mO[row_q, row_h, None], (elems_per_store,))
                    for k in cutlass.range(num_store_k, unroll_full=True):
                        k_idx = tOcO_store[0, 0, k][1] // elems_per_store
                        cute.copy(thr_copy_out, rO[None, m, k], mO_row[None, k_idx])
                if const_expr(has_mxfp8):
                    for k in cutlass.range(num_store_k, unroll_full=True):
                        # 8 E4M3 bytes per lane: a warp writes whole 32-byte sectors.
                        mx.st_global_v2_u32(
                            utils.elem_pointer(
                                mO_mxfp8, (row_q, row_h, tOcO_store[0, 0, k][1])
                            ).toint(),
                            rQ[m, k, 0],
                            rQ[m, k, 1],
                        )
                        col = tOcO_store[0, 0, k][1]
                        if col % Int32(mx.MXFP8_BLOCK) == 0:
                            mx.st_global_u8(
                                scale_base
                                + Int64(
                                    mx.sf_offset_128x4(
                                        mxfp8_first_row + row_q,
                                        row_h * Int32(4) + col // Int32(mx.MXFP8_BLOCK),
                                        scale_cols,
                                    )
                                ),
                                rE[m, k],
                            )

        # The CTA of the last flat row clears the padding rows' scales.
        if const_expr(has_mxfp8):
            if tile_begin + self.tile_m >= total_flat:
                mx.zero_padding_scales(
                    scale_base,
                    mxfp8_first_row + Int32(mO_shape[0]),
                    scale_cols,
                    tidx,
                    self.num_threads,
                )

    @cute.jit
    def claim_tile(
        self,
        mSmDomains: cute.Tensor,
        mTileCounters: cute.Tensor,
        total_tiles: Int32,
        tiles_per_page_log2: Int32,
    ) -> Int32:
        """One tile for this CTA, from the domain of its SM while that domain has tiles left.

        Tiles are dealt to domains by page: page ``j`` (``2 ** tiles_per_page_log2`` tiles)
        belongs to domain ``j % num_domains``, so the ``k``-th tile of domain ``d`` is tile
        ``k % page`` of page ``(k // page) * num_domains + d``. Each counter hands out claims in
        order; a claim past the domain's tile count moves on to the next domain, and since the
        grid holds exactly one CTA per tile, the claims of all domains together always fit.
        """
        num_domains = const_expr(self.num_domains)
        sm_id = cute.arch.smid()
        domain = mSmDomains[sm_id]
        if domain < 0:
            domain = sm_id % Int32(num_domains)
        page_tiles = Int32(1) << tiles_per_page_log2
        full_pages = total_tiles >> tiles_per_page_log2
        tail_tiles = total_tiles & (page_tiles - 1)
        tile = Int32(-1)
        for _ in cutlass.range_constexpr(num_domains):
            if tile < 0:
                claim = cute.arch.atomic_add(mTileCounters.iterator + domain, Int32(1))
                domain_pages = full_pages // Int32(num_domains)
                if full_pages % Int32(num_domains) > domain:
                    domain_pages = domain_pages + 1
                domain_tiles = domain_pages << tiles_per_page_log2
                if full_pages % Int32(num_domains) == domain:
                    domain_tiles = domain_tiles + tail_tiles
                if claim < domain_tiles:
                    page = (claim >> tiles_per_page_log2) * Int32(num_domains) + domain
                    tile = (page << tiles_per_page_log2) + (claim & (page_tiles - 1))
                else:
                    domain = (domain + 1) % Int32(num_domains)
        return tile


def _check_cuda_aligned(tensor: torch.Tensor, *, name: str, device: torch.device) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


def _check_cuda_contiguous(tensor: torch.Tensor, *, name: str, device: torch.device) -> None:
    _check_cuda_aligned(tensor, name=name, device=device)
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def combine(
    o_partial_fake,
    lse_partial,
    o_out,
    lse_out,
    *,
    lse_temperature_partial=None,
    lse_temperature_out=None,
    split_counts=None,
    output_scale=None,
    use_pdl=False,
    locality=None,
    o_mxfp8=None,
    o_mxfp8_first_row=0,
):
    """K2: merge the sparse forward's split partials into the final output.

    Every (query, head) row is combined on its own, so the tensors use the varlen (``total_q``)
    layout whatever the batch structure. When ``lse_out`` is given, the kernel writes the final
    log-sum-exp of each row into it.

    Args:
        o_partial_fake: [num_splits, total_q, head_q, 128] BF16 or F16, columns in the forward
            epilogue's STG.128 order; the rows of a split slab are contiguous, the slabs may be
            padded apart (any 16-byte-aligned split stride).
        lse_partial: [num_splits, total_q, head_q] FP32, contiguous.
        o_out: [total_q, head_q, 128] BF16 or F16, contiguous, or None with ``o_mxfp8``.
        lse_out: [total_q, head_q] FP32, or None.
        lse_temperature_partial, lse_temperature_out: optional temperature-scaled LSE partials
            and output shaped like ``lse_partial`` and ``lse_out``; both or neither.
        split_counts: optional [total_q, head_kv] int32 valid split count per row; splits at or
            past a row's count are ignored. ``head_q`` must be a multiple of ``head_kv``.
        output_scale: optional FP32 tensor with at least one element; its first element scales
            the output once before the store.
        use_pdl: wait on the producer grid (programmatic dependent launch) before reading.
        locality: optional :class:`LocalityHints` for a partial buffer striped over the memory
            locality domains; every CTA then reads its tile from the domain of its own SM.
        o_mxfp8: optional ``(data, scale)`` receiving the output as MXFP8, bit for bit what
            ``flashinfer.mxfp8_quantize(o_out.view(total_q, head_q * 128),
            is_sf_swizzled_layout=True)`` returns: ``data`` [total_q, head_q, 128] float8_e4m3fn
            or uint8 (contiguous rows), ``scale`` a flat, 16-byte aligned uint8 buffer of UE8M0
            scales in the 128x4 swizzled layout, padding rows zeroed. A BF16 output only.
        o_mxfp8_first_row: row of ``scale`` that holds the first row of ``data``: the rows
            before it belong to an earlier kernel filling the same MXFP8 activation. The rows
            written here are the activation's last, so its padding rows are cleared here.
    """
    if o_partial_fake.ndim != 4:
        raise ValueError(
            f"o_partial_fake must be [num_splits, total_q, head_q, dim], got {tuple(o_partial_fake.shape)}"
        )
    num_splits, total_q, head_q, dim = o_partial_fake.shape
    if dim != 128:
        raise ValueError(f"the combine supports head dim 128, got {dim}")
    if not 1 <= num_splits <= 256:
        raise ValueError(f"num_splits must be in [1, 256], got {num_splits}")
    split_stride = o_partial_fake.stride(0)
    if o_partial_fake.stride()[1:] != (head_q * dim, dim, 1) or split_stride < total_q * head_q * dim:
        raise ValueError(
            "o_partial_fake rows must be contiguous inside each split slab, with a split stride "
            f"of at least {total_q * head_q * dim} elements, got strides {o_partial_fake.stride()}"
        )
    if split_stride % (128 // (8 * o_partial_fake.element_size())):
        raise ValueError("the o_partial_fake split stride must keep 16-byte alignment")
    if split_stride >= MAX_SPLIT_SLAB_ELEMENTS:
        raise ValueError(f"the o_partial_fake split stride must be below {MAX_SPLIT_SLAB_ELEMENTS}")
    if o_partial_fake.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"o_partial_fake dtype must be torch.bfloat16 or torch.float16, got {o_partial_fake.dtype}"
        )
    if o_out is None and o_mxfp8 is None:
        raise ValueError("the combine needs o_out or o_mxfp8")
    if o_out is not None:
        if o_out.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                f"o_out dtype must be torch.bfloat16 or torch.float16, got {o_out.dtype}"
            )
        if tuple(o_out.shape) != (total_q, head_q, dim):
            raise ValueError(
                f"o_out must have shape {(total_q, head_q, dim)}, got {tuple(o_out.shape)}"
            )
    o_q = o_sf = None
    if o_mxfp8 is not None:
        o_q, o_sf = o_mxfp8
        if o_out is not None and o_out.dtype != torch.bfloat16:
            raise TypeError("o_mxfp8 needs a bfloat16 o_out")
        if o_mxfp8_first_row < 0:
            raise ValueError("o_mxfp8_first_row must not be negative")
        if o_q.dtype not in (torch.uint8, torch.float8_e4m3fn) or tuple(o_q.shape) != (
            total_q,
            head_q,
            dim,
        ):
            raise ValueError(f"o_mxfp8 data must be uint8/e4m3 {(total_q, head_q, dim)}")
        if o_q.stride()[1:] != (dim, 1) or o_q.stride(0) % 16 or o_q.data_ptr() % 16:
            raise ValueError("o_mxfp8 data rows must be contiguous and 16-byte aligned")
        padded_rows = (o_mxfp8_first_row + total_q + 127) // 128 * 128
        scale_numel = padded_rows * head_q * dim // 32
        if o_sf.dtype != torch.uint8 or o_sf.ndim != 1 or o_sf.numel() < scale_numel:
            raise ValueError(f"o_mxfp8 scale must be a flat uint8 [>= {scale_numel}] tensor")
        o_q = o_q.view(torch.uint8)
    if lse_partial.dtype != torch.float32:
        raise TypeError(f"lse_partial must be torch.float32, got {lse_partial.dtype}")
    if tuple(lse_partial.shape) != (num_splits, total_q, head_q):
        raise ValueError(
            f"lse_partial must have shape {(num_splits, total_q, head_q)}, got {tuple(lse_partial.shape)}"
        )
    if lse_out is not None:
        if lse_out.dtype != torch.float32:
            raise TypeError(f"lse_out must be torch.float32, got {lse_out.dtype}")
        if tuple(lse_out.shape) != (total_q, head_q):
            raise ValueError(
                f"lse_out must have shape {(total_q, head_q)}, got {tuple(lse_out.shape)}"
            )
    if (lse_temperature_partial is None) != (lse_temperature_out is None):
        raise ValueError(
            "lse_temperature_partial and lse_temperature_out must either both be provided or both be None"
        )
    if lse_temperature_out is not None:
        if lse_out is None:
            raise ValueError("lse_temperature_out requires lse_out")
        if lse_temperature_partial.shape != lse_partial.shape:
            raise ValueError(
                "lse_temperature_partial must have the same shape as lse_partial, "
                f"got {tuple(lse_temperature_partial.shape)} vs {tuple(lse_partial.shape)}"
            )
        if lse_temperature_out.shape != lse_out.shape:
            raise ValueError(
                "lse_temperature_out must have the same shape as lse_out, "
                f"got {tuple(lse_temperature_out.shape)} vs {tuple(lse_out.shape)}"
            )
        if lse_temperature_out.dtype != torch.float32 or lse_temperature_partial.dtype != torch.float32:
            raise TypeError("temperature LSE tensors must be torch.float32")
    if split_counts is not None:
        if split_counts.dtype != torch.int32:
            raise TypeError(f"split_counts must be torch.int32, got {split_counts.dtype}")
        if split_counts.ndim != 2 or split_counts.shape[0] != total_q:
            raise ValueError(
                f"split_counts must have shape [total_q, head_kv], got {tuple(split_counts.shape)}"
            )
        if head_q % split_counts.shape[1] != 0:
            raise ValueError(
                f"head_q ({head_q}) must be divisible by split_counts heads ({split_counts.shape[1]})"
            )
    if output_scale is not None:
        if output_scale.dtype != torch.float32:
            raise TypeError(f"output_scale must be torch.float32, got {output_scale.dtype}")
        if output_scale.numel() < 1:
            raise ValueError("output_scale must contain at least one element")
    o_ref = o_out if o_out is not None else o_q
    device = o_ref.device
    if not o_ref.is_cuda:
        raise ValueError("the output must be a CUDA tensor")
    _check_cuda_aligned(o_partial_fake, name="o_partial_fake", device=device)
    for name, tensor in (
        ("lse_partial", lse_partial),
        ("o_out", o_out),
        ("lse_out", lse_out),
        ("lse_temperature_partial", lse_temperature_partial),
        ("lse_temperature_out", lse_temperature_out),
        ("split_counts", split_counts),
        ("output_scale", output_scale),
        ("o_mxfp8 scale", o_sf),
    ):
        if tensor is not None:
            _check_cuda_contiguous(tensor, name=name, device=device)
    if o_q is not None and o_q.device != device:
        raise ValueError(f"o_mxfp8 data must be on {device}, got {o_q.device}")

    partial_dtype = _get_cutlass_dtype(o_partial_fake.dtype)
    out_dtype = _get_cutlass_dtype(torch.bfloat16 if o_out is None else o_out.dtype)
    has_o = o_out is not None
    has_mxfp8 = o_mxfp8 is not None
    has_lse = lse_out is not None
    has_temperature = lse_temperature_out is not None
    has_split_counts = split_counts is not None
    has_output_scale = output_scale is not None
    qhead_per_kvhead = head_q // split_counts.shape[1] if has_split_counts else 1
    num_domains = 0
    tiles_per_page_log2 = 0
    if locality is not None:
        num_domains = locality.num_domains
        if num_domains < 2:
            raise ValueError("locality hints need at least two domains")
        for name, tensor in (
            ("locality.sm_domains", locality.sm_domains),
            ("locality.tile_counters", locality.tile_counters),
        ):
            _check_cuda_contiguous(tensor, name=name, device=device)
            if tensor.dtype != torch.int32 or tensor.ndim != 1:
                raise TypeError(f"{name} must be a one-dimensional torch.int32 tensor")
        tile_bytes = TILE_M * dim * o_partial_fake.element_size()
        tiles_per_page, page_rest = divmod(locality.page_bytes, tile_bytes)
        if page_rest or tiles_per_page & (tiles_per_page - 1):
            raise ValueError(
                f"locality.page_bytes must be a power-of-two multiple of the {tile_bytes}-byte tile"
            )
        tiles_per_page_log2 = tiles_per_page.bit_length() - 1
        if (split_stride * o_partial_fake.element_size()) % (num_domains * locality.page_bytes):
            raise ValueError(
                "the o_partial_fake split stride must be a multiple of num_domains x page_bytes "
                "so the pages of every slab fall into the same domains"
            )
    # A tensor map's split stride must be 16-byte aligned: with a head count that is a multiple
    # of 4 every split's FP32 LSE slab is, whatever total_q, and one 2-D box loads a tile's LSEs;
    # otherwise the kernel brings the tile in with 4-byte cp.async copies.
    lse_split_aligned = head_q % 4 == 0
    capability = torch.cuda.get_device_capability(device)
    # Unsteered reads on SM107 are fabric-bound; see STAGES_INTERLEAVED.
    stages = STAGES_INTERLEAVED if capability == (10, 7) and not num_domains else STAGES

    key = (
        "combine",
        capability,
        dim,
        TILE_M,
        NUM_THREADS,
        stages,
        num_splits,
        partial_dtype,
        out_dtype,
        has_lse,
        has_temperature,
        has_split_counts,
        has_output_scale,
        use_pdl,
        num_domains,
        lse_split_aligned,
    )
    if has_mxfp8:
        key = key + ("o_bf16", has_o, "o_mxfp8")
    if key not in _combine_compile_cache:
        loaded = try_load_aot(key)
        if loaded is not None:
            _combine_compile_cache[key] = loaded
        else:
            from quack.compile_utils import make_fake_tensor

            kernel = SparseAttentionForwardCombine(
                dtype=out_dtype,
                dtype_partial=partial_dtype,
                head_dim=dim,
                topk=num_splits,
                tile_m=TILE_M,
                num_threads=NUM_THREADS,
                stages=stages,
                use_pdl=use_pdl,
                num_domains=num_domains,
                lse_split_aligned=lse_split_aligned,
            )
            total_q_sym, nheads_sym = (cute.sym_int64() for _ in range(2))
            mO_partial = make_fake_tensor(
                partial_dtype,
                (num_splits, total_q_sym, nheads_sym, dim),
                divisibility=128 // partial_dtype.width,
            )
            mLSE_partial = make_fake_tensor(
                Float32, (num_splits, total_q_sym, nheads_sym), divisibility=1, leading_dim=2
            )
            mO = (
                make_fake_tensor(
                    out_dtype, (total_q_sym, nheads_sym, dim), divisibility=128 // out_dtype.width
                )
                if has_o
                else None
            )
            mO_mxfp8, mO_mxfp8_scale = (
                (
                    make_fake_tensor(
                        cutlass.Uint8, (total_q_sym, nheads_sym, dim), divisibility=16
                    ),
                    make_fake_tensor(
                        cutlass.Uint8, (cute.sym_int64(),), divisibility=16, leading_dim=0
                    ),
                )
                if has_mxfp8
                else (None, None)
            )
            mLSE = (
                make_fake_tensor(Float32, (total_q_sym, nheads_sym), divisibility=1, leading_dim=1)
                if has_lse
                else None
            )
            mLSE_temperature_partial = (
                make_fake_tensor(
                    Float32, (num_splits, total_q_sym, nheads_sym), divisibility=1, leading_dim=2
                )
                if has_temperature
                else None
            )
            mLSE_temperature = (
                make_fake_tensor(Float32, (total_q_sym, nheads_sym), divisibility=1, leading_dim=1)
                if has_temperature
                else None
            )
            mSplitCounts = (
                make_fake_tensor(
                    Int32, (cute.sym_int64(), cute.sym_int64()), divisibility=1, leading_dim=1
                )
                if has_split_counts
                else None
            )
            mOutputScale = (
                make_fake_tensor(Float32, (cute.sym_int64(),), divisibility=1, leading_dim=0)
                if has_output_scale
                else None
            )
            mSmDomains, mTileCounters = (
                (
                    make_fake_tensor(Int32, (cute.sym_int64(),), divisibility=1, leading_dim=0),
                    make_fake_tensor(Int32, (num_domains,), divisibility=1, leading_dim=0),
                )
                if num_domains
                else (None, None)
            )
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            _combine_compile_cache[key] = compile_with_timing(
                kernel,
                mO_partial,
                mLSE_partial,
                mO,
                mLSE,
                mLSE_temperature_partial,
                mLSE_temperature,
                mSplitCounts,
                mOutputScale,
                Int32(qhead_per_kvhead),
                mSmDomains,
                mTileCounters,
                Int32(tiles_per_page_log2),
                mO_mxfp8,
                mO_mxfp8_scale,
                Int32(o_mxfp8_first_row),
                stream,
                options="--enable-tvm-ffi",
            )
            save_aot(key, _combine_compile_cache[key], sources=[__name__])

    with torch.cuda.nvtx.range("K2_Combine"):
        _combine_compile_cache[key](
            o_partial_fake,
            lse_partial,
            o_out,
            lse_out,
            lse_temperature_partial,
            lse_temperature_out,
            split_counts,
            output_scale,
            qhead_per_kvhead,
            None if locality is None else locality.sm_domains,
            None if locality is None else locality.tile_counters,
            tiles_per_page_log2,
            o_q,
            o_sf,
            o_mxfp8_first_row,
        )
