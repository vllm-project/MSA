# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Regression tests for large sparse-topk histogram boundary bins."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from fmha_sm100 import sparse_topk_select


def _clustered_scores(num_valid_pages, max_k_tiles, center=0.5):
    scores = torch.full(
        (2, 2, max_k_tiles), -float("inf"), device="cuda", dtype=torch.float32
    )
    # These 1,600 distinct fp32 values collapse into one stage-0 fp16
    # histogram bin, exercising the final-candidate path without score ties.
    cluster = center + torch.arange(
        num_valid_pages, device="cuda", dtype=torch.float32
    ) * 1e-7
    scores[..., :num_valid_pages] = cluster
    return scores


def _refined_boundary_scores(final_count, max_k_tiles, distinct=False):
    """Force stage-0/1 overflow, then a stage-2 bin of ``final_count``."""
    num_valid_pages = 3200
    scores = torch.full(
        (2, 2, max_k_tiles), -float("inf"), device="cuda", dtype=torch.float32
    )
    low = torch.tensor([0x3F060FFF], device="cuda", dtype=torch.uint32).view(
        torch.float32
    )
    scores[..., :num_valid_pages] = low
    if distinct:
        high_bits = (
            0x3F070000
            + torch.arange(final_count, device="cuda", dtype=torch.int64)
        ).to(torch.uint32)
        scores[..., :final_count] = high_bits.view(torch.float32)
    else:
        high = torch.tensor(
            [0x3F070FFF], device="cuda", dtype=torch.uint32
        ).view(torch.float32)
        scores[..., :final_count] = high
    return scores


@pytest.mark.parametrize("num_valid_pages", [512, 513, 1600])
@pytest.mark.parametrize("center", [0.5, -1.0])
def test_clustered_boundary_bin_matches_exact_topk(num_valid_pages, center):
    """Both sides of the rank/merge cutoff select the exact top-k set."""
    scores = _clustered_scores(num_valid_pages, 8192, center)

    result = sparse_topk_select(
        scores, 16, num_valid_pages=num_valid_pages, max_score_layout="THK"
    )
    expected = torch.topk(scores[..., :num_valid_pages], 16, dim=-1).indices
    expected = expected.sort(dim=-1).values.to(torch.int32)

    assert torch.equal(result, expected)


def test_large_equal_boundary_bin_returns_unique_valid_indices():
    """A large exact tie must not emit duplicates or padding indices."""
    num_valid_pages = 1600
    scores = torch.full(
        (2, 2, 8192), -float("inf"), device="cuda", dtype=torch.float32
    )
    scores[..., :num_valid_pages] = 0.5

    result = sparse_topk_select(
        scores, 16, num_valid_pages=num_valid_pages, max_score_layout="THK"
    )

    assert torch.all(result >= 0)
    assert torch.all(result < num_valid_pages)
    assert torch.all(result[..., 1:] > result[..., :-1])


@pytest.mark.parametrize("num_higher_scores", [1, 7, 15])
def test_large_boundary_bin_with_partially_filled_topk(num_higher_scores):
    """The merge emits exactly the slots left after higher bins were selected."""
    num_valid_pages = 1600
    scores = _clustered_scores(num_valid_pages, 8192)
    scores[..., :num_higher_scores] = 10.0 + torch.arange(
        num_higher_scores, device="cuda", dtype=torch.float32
    )

    result = sparse_topk_select(
        scores, 16, num_valid_pages=num_valid_pages, max_score_layout="THK"
    )
    expected = torch.topk(scores[..., :num_valid_pages], 16, dim=-1).indices
    expected = expected.sort(dim=-1).values.to(torch.int32)

    assert torch.equal(result, expected)


def test_large_equal_boundary_bin_with_forced_blocks():
    """Forced blocks and the large-bin merge jointly fill one top-k row."""
    num_valid_pages = 1600
    scores = torch.full(
        (2, 2, 8192), -float("inf"), device="cuda", dtype=torch.float32
    )
    scores[..., :num_valid_pages] = 0.5

    result = sparse_topk_select(
        scores,
        16,
        num_valid_pages=num_valid_pages,
        force_begin_blocks=1,
        force_end_blocks=1,
        max_score_layout="THK",
    )

    assert torch.all(result[..., 0] == 0)
    assert torch.all(result[..., -1] == num_valid_pages - 1)
    assert torch.all(result[..., 1:] > result[..., :-1])


def test_large_boundary_bin_preserves_high_source_indices():
    """The packed merge key retains every bit of supported source indices."""
    max_k_tiles = 10_000
    cluster_start = 8400
    scores = torch.full(
        (1, 1, max_k_tiles), -2.0, device="cuda", dtype=torch.float32
    )
    scores[..., cluster_start:] = 0.5 + torch.arange(
        max_k_tiles - cluster_start, device="cuda", dtype=torch.float32
    ) * 1e-7

    result = sparse_topk_select(
        scores, 16, num_valid_pages=max_k_tiles, max_score_layout="THK"
    )
    expected = torch.arange(
        max_k_tiles - 16, max_k_tiles, device="cuda", dtype=torch.int32
    ).view(1, 1, 16)

    assert torch.equal(result, expected)


@pytest.mark.parametrize("final_count", [384, 385, 416, 417, 511, 512, 513, 2048])
def test_refined_boundary_bin_returns_unique_top_scores(final_count):
    """A stage-2 shrink is correct on both sides of the refined cutoff."""
    scores = _refined_boundary_scores(final_count, 8192)

    result = sparse_topk_select(
        scores, 16, num_valid_pages=3200, max_score_layout="THK"
    )

    assert torch.all(result >= 0)
    assert torch.all(result < final_count)
    assert torch.all(result[..., 1:] > result[..., :-1])


def test_refined_boundary_bin_is_padding_invariant():
    """The valid prefix produces the same exact top-k with or without padding."""
    final_count = 512
    compact = _refined_boundary_scores(final_count, 3200, distinct=True)
    padded = _refined_boundary_scores(final_count, 8192, distinct=True)
    expected = torch.arange(496, 512, device="cuda", dtype=torch.int32)
    expected = expected.view(1, 1, 16).expand(2, 2, 16)

    compact_result = sparse_topk_select(
        compact, 16, num_valid_pages=3200, max_score_layout="THK"
    )
    padded_result = sparse_topk_select(
        padded, 16, num_valid_pages=3200, max_score_layout="THK"
    )

    assert torch.equal(compact_result, expected)
    assert torch.equal(padded_result, expected)


def test_per_token_valid_prefixes_select_exact_topk():
    """Tensor num_valid_pages applies a different rowEnd to every token."""
    valid_pages = torch.tensor(
        [64, 384, 417, 1600, 3200], device="cuda", dtype=torch.int32
    )
    scores = torch.full(
        (valid_pages.numel(), 2, 8192),
        -float("inf"),
        device="cuda",
        dtype=torch.float32,
    )
    for token, nvp in enumerate(valid_pages.tolist()):
        scores[token, :, :nvp] = 0.5 + torch.arange(
            nvp, device="cuda", dtype=torch.float32
        ) * 1e-7

    result = sparse_topk_select(
        scores, 16, num_valid_pages=valid_pages, max_score_layout="THK"
    )
    expected = torch.stack(
        [
            torch.arange(nvp - 16, nvp, device="cuda", dtype=torch.int32)
            for nvp in valid_pages.tolist()
        ]
    ).view(-1, 1, 16).expand(-1, 2, -1)

    assert torch.equal(result, expected)
