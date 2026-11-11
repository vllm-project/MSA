# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Uninitialized scratch must be safe within every query's causal score prefix."""

import pytest
import torch


@pytest.mark.parametrize("tp", [2, 4])
@torch.inference_mode()
def test_sharded_score_p256_compact_columns(tp):
    from fmha_sm100.api import (
        fmha_index_score_sharded,
        fmha_index_score_sharded_plan,
    )

    torch.manual_seed(13)
    device, query_len, heads, length = torch.device("cuda"), 5, 4, 641
    blocks = (length + 127) // 128
    pages = (length + 255) // 256
    plan = fmha_index_score_sharded_plan(
        torch.tensor([query_len], dtype=torch.int32),
        torch.tensor([length], dtype=torch.int32),
        heads,
        qo_offset=torch.tensor([length - query_len], dtype=torch.int32),
        device=device,
    )
    query = torch.randn(query_len, heads, 128, device=device).to(torch.float8_e4m3fn)
    cache = torch.randn(pages, 256, 128, device=device).to(query.dtype)
    table = torch.arange(pages, device=device, dtype=torch.int32)[None]
    for rank in range(tp):
        rows = 256 // tp
        shard = cache[:, rank * rows : (rank + 1) * rows].contiguous()
        dense = torch.empty(query_len, heads, blocks, device=device)
        compact = torch.empty(query_len, heads, (blocks + 1) // 2, device=device)
        common = dict(
            q=query,
            k=shard,
            block_table=table,
            plan_info=plan,
            rank=rank,
            tp=tp,
            sm_scale=1.0,
        )
        fmha_index_score_sharded(**common, max_score=dense)

        def score_compact():
            fmha_index_score_sharded(
                **common,
                max_score=compact,
                compact=True,
                initialize=False,
            )

        offset = rank * 2 // tp
        expected = dense[..., offset::2]
        compact.fill_(1.0e9)
        score_compact()
        torch.testing.assert_close(
            compact[..., : expected.shape[-1]], expected, rtol=0, atol=0
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            score_compact()
        compact.fill_(2.0e9)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            compact[..., : expected.shape[-1]], expected, rtol=0, atol=0
        )


@pytest.mark.parametrize("tp,rank", [(2, 1), (4, 1), (4, 2), (4, 3)])
@torch.inference_mode()
def test_sharded_score_empty_rank_graph_replay(tp, rank):
    from fmha_sm100.api import (
        fmha_index_score_sharded,
        fmha_index_score_sharded_plan,
    )

    device, batch, query_len, heads = torch.device("cuda"), 32, 8, 4
    total_q = batch * query_len
    lengths = torch.full((batch,), query_len, dtype=torch.int32)
    plan = fmha_index_score_sharded_plan(
        lengths,
        lengths,
        heads,
        qo_offset=torch.zeros_like(lengths),
        device=device,
    )
    query = torch.randn(total_q, heads, 128, device=device).to(torch.float8_e4m3fn)
    cache = torch.randn(1, 256 // tp, 128, device=device).to(query.dtype)
    block_table = torch.zeros(batch, 1, dtype=torch.int32, device=device)
    scores = torch.empty(total_q, heads, 1, dtype=torch.float32, device=device)

    def score():
        fmha_index_score_sharded(
            query,
            cache,
            block_table,
            plan,
            rank=rank,
            tp=tp,
            max_score=scores,
            sm_scale=1.0,
        )

    score()
    torch.cuda.synchronize()
    assert torch.isneginf(scores).all()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        score()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.isneginf(scores).all()


@pytest.mark.parametrize(
    "query_lens,prefixes",
    [
        ([1], [0]),
        ([1], [128]),
        ([1], [256]),
        ([1], [512]),
        ([65], [0]),
        ([257, 1], [0, 128]),
        ([0, 65, 0, 3], [127, 194, 0, 127]),
    ],
)
@pytest.mark.parametrize("device_plan", [False, True])
@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("page_size", [128, 256])
@torch.inference_mode()
def test_sharded_score_missing_columns(query_lens, prefixes, device_plan, tp, page_size):
    from fmha_sm100.api import (
        _get_num_cta,
        fmha_index_score_sharded,
        fmha_index_score_sharded_plan,
        fmha_index_score_sharded_plan_device,
    )

    torch.manual_seed(12)
    device, heads, chunk = torch.device("cuda"), 4, 256
    total_q = sum(query_lens)
    lengths = [q + p for q, p in zip(query_lens, prefixes)]
    pages = (max(lengths) + page_size - 1) // page_size
    blocks = (max(lengths) + 127) // 128
    table = torch.stack([torch.randperm(pages, device=device) for _ in lengths]).int()
    table[:, 0] = table[0, 0].clone()
    query = torch.randint(-2, 3, (total_q, heads, 128), device=device).to(
        torch.float8_e4m3fn
    )
    query[..., 0] = 16
    cache = torch.randint(-2, 3, (pages, page_size, 128), device=device).to(query.dtype)
    offsets = [0]
    for length in query_lens:
        offsets.append(offsets[-1] + length)
    plans = []
    for begin in range(0, total_q, chunk):
        end = min(begin + chunk, total_q)
        local_starts = [
            min(q, max(0, begin - off)) for q, off in zip(query_lens, offsets)
        ]
        q_lens = [
            max(0, min(end, off + q) - max(begin, off))
            for q, off in zip(query_lens, offsets)
        ]
        starts = [p + s for p, s in zip(prefixes, local_starts)]
        kv_lens = [s + q for s, q in zip(starts, q_lens)]
        if device_plan:
            q_meta, kv_meta, start_meta = [
                torch.tensor(values, dtype=torch.int32, device=device)
                for values in (q_lens, kv_lens, starts)
            ]
            q_offsets, kv_offsets = [
                torch.cat((x.new_zeros(1), x.cumsum(0).int()))
                for x in (q_meta, kv_meta)
            ]
            plan = fmha_index_score_sharded_plan_device(
                q_meta,
                kv_meta,
                q_offsets,
                kv_offsets,
                start_meta,
                heads,
                total_q=end - begin,
                max_query_len=max(q_lens),
                max_seq_len=max(kv_lens),
                packed_work_range=torch.empty(
                    _get_num_cta(device), device=device, dtype=torch.int64
                ),
                packed_work_info=torch.empty(
                    heads * ((end - begin + 127) // 128 + len(q_lens) - 1),
                    device=device,
                    dtype=torch.int64,
                ),
                workspace=torch.empty(32 << 20, device=device, dtype=torch.uint8),
            )
        else:
            plan = fmha_index_score_sharded_plan(
                torch.tensor(q_lens, dtype=torch.int32),
                torch.tensor(kv_lens, dtype=torch.int32),
                heads,
                qo_offset=torch.tensor(starts, dtype=torch.int32),
                device=device,
            )
        plans.append((begin, end, plan))

    storage = torch.empty((min(chunk, total_q), heads, blocks + 1), device=device)
    positions = torch.cat(
        [torch.arange(q, device=device) + p for q, p in zip(query_lens, prefixes)]
    )
    begin, end, plan = plans[0]
    with pytest.raises(ValueError, match="query shape"):
        fmha_index_score_sharded(
            query[begin:end],
            cache[:, ::tp].contiguous(),
            table,
            {**plan, "num_kv_splits": 2},
            rank=0,
            tp=tp,
            max_score=storage[: end - begin, :, :blocks],
            sm_scale=1.0,
            initialize=False,
        )
    for poison in (float("nan"), float("inf")):
        query = (-query.float()).to(query.dtype)
        for req, length in enumerate(lengths):
            if length:
                page = table[req, (length - 1) // page_size]
                tail = (length - 1) % page_size
                cache[page, tail:] = 0
                cache[page, tail:, 0] = 256
        for rank in range(tp):
            rows = page_size // tp
            local_rows = (
                torch.arange(rank, page_size, tp, device=device)
                if page_size == 128
                else torch.arange(rank * rows, (rank + 1) * rows, device=device)
            )
            shard = (
                cache[:, rank::tp]
                if page_size == 128
                else cache[:, rank * rows : (rank + 1) * rows]
            ).contiguous()
            key_pos = (
                torch.arange(pages, device=device)[:, None] * page_size + local_rows
            ).flatten()
            reference = torch.full(
                (total_q, heads, blocks), -float("inf"), device=device
            )
            for req, q_len in enumerate(query_lens):
                if q_len == 0:
                    continue
                q = query[offsets[req] : offsets[req + 1]].float()
                keys = shard[table[req]].float().reshape(-1, 128)
                dots = torch.einsum("qhd,kd->qhk", q, keys)
                causal = positions[offsets[req] : offsets[req + 1]]
                dots.masked_fill_(
                    key_pos[None, None, :] > causal[:, None, None], -float("inf")
                )
                for block in range(blocks):
                    owned = key_pos // 128 == block
                    if owned.any():
                        reference[offsets[req] : offsets[req + 1], :, block] = (
                            dots[..., owned].amax(-1)
                        )
            for begin, end, plan in plans:
                storage.fill_(poison)
                scores = storage[: end - begin, :, :blocks]
                fmha_index_score_sharded(
                    query[begin:end],
                    shard,
                    table,
                    plan,
                    rank=rank,
                    tp=tp,
                    max_score=scores,
                    sm_scale=1.0,
                    initialize=False,
                )
                valid = positions[begin:end] // 128 + 1
                mask = (torch.arange(blocks, device=device)[None, :] < valid[:, None])[
                    :, None, :
                ].expand_as(scores)
                torch.testing.assert_close(
                    scores[mask], reference[begin:end][mask], rtol=0, atol=0
                )
                if poison != poison:
                    assert torch.isnan(storage[..., -1]).all()
                    assert torch.isnan(storage[end - begin :]).all()
                else:
                    assert torch.isposinf(storage[..., -1]).all()
                    assert torch.isposinf(storage[end - begin :]).all()
