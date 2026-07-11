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
