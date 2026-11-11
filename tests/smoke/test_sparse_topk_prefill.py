# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Compact selection and reuse; publisher peer windows are on one GPU."""

import pytest
import torch

from fmha_sm100 import sparse_topk_select_prefill
from fmha_sm100.jit import get_sparse_topk_module


@pytest.mark.parametrize("index_offset", [0, 1])
def test_prefill_compact_ids_are_restored(index_offset):
    scores = torch.arange(32, device="cuda", dtype=torch.float32)[None, None]
    valid = torch.tensor([61], device="cuda", dtype=torch.int32)
    carrier = torch.empty((1, 1, 33), device="cuda", dtype=torch.int32)
    sparse_topk_select_prefill(
        scores,
        valid,
        carrier,
        force_begin_blocks=2,
        force_end_blocks=3,
        index_scale=2,
        index_offset=index_offset,
    )
    global_ids = list(range(index_offset, 61, 2))
    forced = {i for i in global_ids if i < 2 or i >= 58}
    ordinary = [i for i in global_ids if i not in forced]
    selected = forced | set(ordinary[-(16 - len(forced)) :])
    assert carrier[0, 0, 16:32].cpu().tolist() == sorted(selected)
    local_ids = [(i - index_offset) // 2 for i in sorted(selected)]
    assert torch.equal(
        carrier[0, 0, :16].cpu(), scores[0, 0, local_ids].view(torch.int32).cpu()
    )


@pytest.mark.parametrize("force_end", [0, 2])
def test_prefill_cutoff_ties_use_stable_block_order(force_end):
    pages = 64
    scores = torch.ones((1, 1, pages), device="cuda", dtype=torch.float32)
    valid = torch.tensor([pages], device="cuda", dtype=torch.int32)
    carrier = torch.empty((1, 1, 33), device="cuda", dtype=torch.int32)

    sparse_topk_select_prefill(
        scores,
        valid,
        carrier,
        force_end_blocks=force_end,
    )

    expected = list(range(16 - force_end))
    expected.extend(range(pages - force_end, pages))
    assert carrier[0, 0, 16:32].cpu().tolist() == expected


@pytest.mark.parametrize(
    "pages,mode",
    [
        (127, "unique"),
        (128, "unique"),
        (129, "unique"),
        (129, "tie"),
        (1535, "cluster"),
        (1536, "low_bits"),
        (1537, "cross_bin"),
        (1600, "unique"),
        (1600, "cluster"),
        (1600, "low_bits"),
        (1600, "cross_bin"),
        (10000, "cluster"),
        (1600, "zero"),
    ],
)
@pytest.mark.parametrize("force_begin,force_end", [(0, 0), (2, 3)])
@pytest.mark.parametrize("publish_world", [0, 2, 4])
def test_prefill_compact_selection(pages, mode, force_begin, force_end, publish_world):
    width = pages + 31
    lengths = [0, 1, 15, 16, 17, 128, pages]
    tokens, heads = len(lengths), 4
    scores = torch.empty((tokens, heads, width), device="cuda")
    valid = torch.empty(tokens, device="cuda", dtype=torch.int32)
    sentinel = 0x13131313
    storage = torch.full(
        (heads, 2 * tokens + 3, 41), sentinel, device="cuda", dtype=torch.int32
    )
    carrier = storage[:, 2 : 2 + 2 * tokens : 2, :33]
    guard = torch.ones_like(storage, dtype=torch.bool)
    guard[:, 2 : 2 + 2 * tokens : 2, :33] = False
    if publish_world:
        capacity, hpr = tokens + 2, heads // publish_world
        windows = [
            torch.zeros(
                (3, publish_world, hpr, capacity, 32), device="cuda", dtype=torch.int64
            )
            for _ in range(publish_world)
        ]
        bases = torch.tensor(
            [w.data_ptr() for w in windows], device="cuda", dtype=torch.int64
        )
        counters = [
            torch.full((3, heads, capacity), -2, device="cuda", dtype=torch.int32)
            for _ in range(publish_world)
        ]
        errors = torch.zeros(1, device="cuda", dtype=torch.int32)
        publish = get_sparse_topk_module().sparse_topk_select_publish
    for reuse in range(2):
        current_lengths = lengths if reuse == 0 else lengths[::-1]
        source = torch.full((tokens, heads, width), float("nan"))
        for token, count in enumerate(current_lengths):
            if mode == "unique":
                values = torch.arange(count, dtype=torch.float32) - count / 2
            elif mode == "cluster":
                values = (0x3F070000 + torch.arange(count, dtype=torch.int32)).view(
                    torch.float32
                )
            elif mode == "low_bits":
                # More than 128 entries share all but the final two fp32 bits.
                values = (0x3F070000 + torch.arange(count, dtype=torch.int32) % 4).view(
                    torch.float32
                )
            elif mode == "zero":
                values = torch.zeros(count)
                values[::2] = -0.0
            elif mode == "cross_bin":
                values = torch.full((count,), 0.51)
                values[-min(7, count) :] = 0.70
            else:
                values = torch.full((count,), 0.5)
            if reuse:
                values = values.flip(0)
            values[: min(force_begin, count)] = -100.0
            if force_end:
                values[max(0, count - force_end) :] = -100.0
            source[token] = torch.cat(
                (values, torch.full((width - count,), float("nan")))
            )
        scores.copy_(source)
        valid.copy_(torch.tensor(current_lengths, dtype=torch.int32))
        if publish_world:
            for rank in range(publish_world):
                publish(
                    scores,
                    valid,
                    bases,
                    errors,
                    0,
                    rank,
                    publish_world,
                    hpr,
                    capacity,
                    force_begin,
                    force_end,
                    1 << 20,
                    False,
                    counters[rank],
                    1,
                    0,
                    torch.cuda.current_stream().cuda_stream,
                )
            result = (
                torch.cat(
                    [
                        w[0, :, :, :tokens].permute(2, 0, 1, 3).flatten(1, 2)
                        for w in windows
                    ],
                    dim=1,
                )
                .to(torch.int32)
                .cpu()
            )
            generation = 0xFFFFFFFF if reuse == 0 else 1
            for owner, window in enumerate(windows):
                assert torch.all(
                    (window[0, :, :, :tokens] >> 32) & 0xFFFFFFFF == generation
                )
                assert torch.count_nonzero(window[1:]) == 0
                assert torch.count_nonzero(window[0, :, :, tokens:]) == 0
            assert errors.item() == 0
        else:
            sparse_topk_select_prefill(
                scores,
                valid,
                carrier,
                force_begin_blocks=force_begin,
                force_end_blocks=force_end,
            )
            result = carrier.cpu().permute(1, 0, 2)
            assert torch.all(storage[guard] == sentinel)
            assert torch.all(result[..., 32] == 1)
        for token, count in enumerate(current_lengths):
            forced = set(range(min(force_begin, count))) | set(
                range(max(0, count - force_end), count)
            )
            ordinary = [i for i in range(count) if i not in forced]
            quota = min(16, count) - len(forced)
            expected_scores = (
                source[token, 0, ordinary].topk(quota).values.sort().values
            )
            for output_head in range(result.shape[1]):
                head = (
                    output_head
                    if not publish_world
                    else (output_head // (publish_world * hpr)) * hpr
                    + output_head % hpr
                )
                ids = result[token, output_head, 16:32].tolist()
                live = ids[: min(16, count)]
                assert live == sorted(set(live))
                assert all(0 <= i < count for i in live)
                assert ids[min(16, count) :] == [-1] * (16 - min(16, count))
                assert forced.issubset(live)
                selected = [i for i in live if i not in forced]
                torch.testing.assert_close(
                    source[token, head, selected].sort().values,
                    expected_scores,
                    rtol=0,
                    atol=0,
                )
                expected_bits = torch.full((16,), -float("inf")).view(torch.int32)
                if live:
                    expected_bits[: len(live)] = source[token, head, live].view(
                        torch.int32
                    )
                assert torch.equal(result[token, output_head, :16], expected_bits)
