# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Test forced begin/end block selection in sparse_topk_select.

Verifies that force_begin_blocks and force_end_blocks guarantee those
block indices appear in the output regardless of their scores.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch
from fmha_sm100 import sparse_topk_select


def _gather_expected_from_block_table(logical_indices, block_table):
    safe_indices = logical_indices.clamp_min(0).to(torch.long)
    gathered = torch.gather(block_table, 2, safe_indices)
    return torch.where(logical_indices >= 0, gathered, torch.full_like(gathered, -1))


def test_block_table_gather_after_sort(
    num_qo_heads=3, max_k_tiles=96, total_qo_len=7,
    topk=16, num_valid_pages=73, seed=202,
):
    """Optional block_table gathers after logical top-k indices are sorted."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")

    max_score_thk = torch.randn(
        total_qo_len, num_qo_heads, max_k_tiles, device=dev, dtype=torch.float32
    )
    max_score_thk[:, :, num_valid_pages:] = float("-inf")

    t = torch.arange(total_qo_len, device=dev, dtype=torch.int32).view(-1, 1, 1)
    h = torch.arange(num_qo_heads, device=dev, dtype=torch.int32).view(1, -1, 1)
    k = torch.arange(max_k_tiles, device=dev, dtype=torch.int32).view(1, 1, -1)
    # Deliberately not monotonic in k, proving that the final order follows the
    # sorted logical indices and then gathers values from the table.
    block_table = (100_000 * t + 1_000 * h + (max_k_tiles - 1 - k) * 7).contiguous()

    logical_thk = sparse_topk_select(
        max_score_thk, topk, num_valid_pages=num_valid_pages, max_score_layout="THK"
    )
    physical_thk = sparse_topk_select(
        max_score_thk, topk, num_valid_pages=num_valid_pages,
        max_score_layout="THK", block_table=block_table,
    )
    assert torch.equal(physical_thk, _gather_expected_from_block_table(logical_thk, block_table))
    assert not torch.equal(physical_thk, logical_thk), (
        "block_table path should emit gathered physical block IDs, not logical indices"
    )

    max_score_hkt = max_score_thk.permute(1, 2, 0).contiguous()
    logical_hkt = sparse_topk_select(max_score_hkt, topk, num_valid_pages=num_valid_pages)
    physical_hkt = sparse_topk_select(
        max_score_hkt, topk, num_valid_pages=num_valid_pages, block_table=block_table,
    )
    assert torch.equal(logical_hkt, logical_thk)
    assert torch.equal(physical_hkt, physical_thk)
    print("  [PASS] block_table gather after logical sort for THK and HKT layouts")


def test_block_table_gather_identity_path(
    num_qo_heads=2, max_k_tiles=8, total_qo_len=4,
    topk=16, num_valid_pages=5, seed=303,
):
    """Trivial max_k_tiles <= topk path must also gather through block_table."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")

    max_score = torch.randn(num_qo_heads, max_k_tiles, total_qo_len,
                            device=dev, dtype=torch.float32)
    t = torch.arange(total_qo_len, device=dev, dtype=torch.int32).view(-1, 1, 1)
    h = torch.arange(num_qo_heads, device=dev, dtype=torch.int32).view(1, -1, 1)
    k = torch.arange(max_k_tiles, device=dev, dtype=torch.int32).view(1, 1, -1)
    block_table = (10_000 * t + 100 * h + 3 * k + 1).contiguous()

    result = sparse_topk_select(
        max_score, topk, num_valid_pages=num_valid_pages, block_table=block_table,
    )
    expected = torch.full(
        (total_qo_len, num_qo_heads, topk), -1, device=dev, dtype=torch.int32
    )
    expected[:, :, :num_valid_pages] = block_table[:, :, :num_valid_pages]
    assert torch.equal(result, expected)
    print("  [PASS] block_table gather works on identity-fill path")


def test_forced_blocks(
    num_qo_heads=4, max_k_tiles=256, total_qo_len=8,
    topk=16, num_valid_pages=200,
    force_begin=3, force_end=2,
    seed=42,
):
    """Core test: forced blocks must appear in output even with worst scores."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")

    # Generate random scores, then deliberately set forced blocks to WORST scores
    max_score = torch.randn(num_qo_heads, max_k_tiles, total_qo_len,
                            device=dev, dtype=torch.float32)
    # Fill padding with -inf
    max_score[:, num_valid_pages:, :] = float('-inf')
    # Give forced blocks the WORST possible valid scores to stress-test
    max_score[:, :force_begin, :] = -1e10
    max_score[:, num_valid_pages - force_end:num_valid_pages, :] = -1e10

    # Run with forced selection
    result = sparse_topk_select(
        max_score, topk,
        num_valid_pages=num_valid_pages,
        force_begin_blocks=force_begin,
        force_end_blocks=force_end,
    )
    assert result.shape == (total_qo_len, num_qo_heads, topk)
    assert result.dtype == torch.int32

    # Verify: all forced begin indices [0, force_begin) must appear in every row
    result_cpu = result.cpu()
    for t in range(total_qo_len):
        for h in range(num_qo_heads):
            row = set(result_cpu[t, h].tolist())
            row.discard(-1)
            for idx in range(force_begin):
                assert idx in row, (
                    f"[FAIL] force_begin idx={idx} missing from row (t={t}, h={h}): {sorted(row)}"
                )
            for idx in range(num_valid_pages - force_end, num_valid_pages):
                assert idx in row, (
                    f"[FAIL] force_end idx={idx} missing from row (t={t}, h={h}): {sorted(row)}"
                )
    print(f"  [PASS] force_begin={force_begin}, force_end={force_end}, "
          f"max_k={max_k_tiles}, nvp={num_valid_pages}")


def test_forced_zero_is_noop(
    num_qo_heads=4, max_k_tiles=256, total_qo_len=8,
    topk=16, num_valid_pages=200, seed=42,
):
    """force_begin=0, force_end=0 must produce identical results to no-force."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    max_score = torch.randn(num_qo_heads, max_k_tiles, total_qo_len,
                            device=dev, dtype=torch.float32)
    max_score[:, num_valid_pages:, :] = float('-inf')

    result_noop = sparse_topk_select(max_score, topk, num_valid_pages=num_valid_pages)
    result_zero = sparse_topk_select(
        max_score, topk, num_valid_pages=num_valid_pages,
        force_begin_blocks=0, force_end_blocks=0,
    )
    assert torch.equal(result_noop, result_zero), (
        f"[FAIL] force_begin=0, force_end=0 differs from default"
    )
    print("  [PASS] force_begin=0, force_end=0 == no-force (bitwise identical)")


def test_forced_ascending_order(
    num_qo_heads=4, max_k_tiles=512, total_qo_len=4,
    topk=16, num_valid_pages=400,
    force_begin=4, force_end=3, seed=123,
):
    """Output must still be in ascending order with forced blocks."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    max_score = torch.randn(num_qo_heads, max_k_tiles, total_qo_len,
                            device=dev, dtype=torch.float32)
    max_score[:, num_valid_pages:, :] = float('-inf')
    max_score[:, :force_begin, :] = -1e10
    max_score[:, num_valid_pages - force_end:num_valid_pages, :] = -1e10

    result = sparse_topk_select(
        max_score, topk, num_valid_pages=num_valid_pages,
        force_begin_blocks=force_begin, force_end_blocks=force_end,
    )
    result_cpu = result.cpu()
    for t in range(total_qo_len):
        for h in range(num_qo_heads):
            row = result_cpu[t, h].tolist()
            valid = [x for x in row if x >= 0]
            assert valid == sorted(valid), (
                f"[FAIL] row not ascending at (t={t}, h={h}): {row}"
            )
    print(f"  [PASS] ascending order preserved with force_begin={force_begin}, force_end={force_end}")


def test_forced_with_xor_fast_path(
    num_qo_heads=4, max_k_tiles=256, total_qo_len=32,
    topk=16, num_valid_pages=224,
    force_begin=2, force_end=3, seed=77,
):
    """XorF4 transpose fast path requires qo%32==0 and K%32==0."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    assert total_qo_len % 32 == 0 and max_k_tiles % 32 == 0
    max_score = torch.randn(num_qo_heads, max_k_tiles, total_qo_len,
                            device=dev, dtype=torch.float32)
    max_score[:, num_valid_pages:, :] = float('-inf')
    max_score[:, :force_begin, :] = -1e10
    max_score[:, num_valid_pages - force_end:num_valid_pages, :] = -1e10

    result = sparse_topk_select(
        max_score, topk, num_valid_pages=num_valid_pages,
        force_begin_blocks=force_begin, force_end_blocks=force_end,
    )
    result_cpu = result.cpu()
    for t in range(total_qo_len):
        for h in range(num_qo_heads):
            row = set(result_cpu[t, h].tolist())
            row.discard(-1)
            for idx in range(force_begin):
                assert idx in row, f"[FAIL] XorF4 path: begin idx={idx} missing"
            for idx in range(num_valid_pages - force_end, num_valid_pages):
                assert idx in row, f"[FAIL] XorF4 path: end idx={idx} missing"
    print(f"  [PASS] XorF4 fast path: force_begin={force_begin}, force_end={force_end}, "
          f"qo={total_qo_len}, K={max_k_tiles}")


def test_forced_large_k(
    num_qo_heads=4, max_k_tiles=4096, total_qo_len=4,
    topk=16, num_valid_pages=4000,
    force_begin=2, force_end=4, seed=99,
):
    """Large K (> 4096 tiles) to stress histogram multi-pass path."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    max_score = torch.randn(num_qo_heads, max_k_tiles, total_qo_len,
                            device=dev, dtype=torch.float32)
    max_score[:, num_valid_pages:, :] = float('-inf')
    max_score[:, :force_begin, :] = -1e10
    max_score[:, num_valid_pages - force_end:num_valid_pages, :] = -1e10

    result = sparse_topk_select(
        max_score, topk, num_valid_pages=num_valid_pages,
        force_begin_blocks=force_begin, force_end_blocks=force_end,
    )
    result_cpu = result.cpu()
    for t in range(total_qo_len):
        for h in range(num_qo_heads):
            row = set(result_cpu[t, h].tolist())
            row.discard(-1)
            for idx in range(force_begin):
                assert idx in row, f"[FAIL] large_k: begin idx={idx} missing"
            for idx in range(num_valid_pages - force_end, num_valid_pages):
                assert idx in row, f"[FAIL] large_k: end idx={idx} missing"
    print(f"  [PASS] large K={max_k_tiles}: force_begin={force_begin}, force_end={force_end}")


if __name__ == "__main__":
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(dev)
    if not (p.major == 10 and p.minor in (0, 3)):
        print("SKIP: SM100/SM103 GPU not available")
        sys.exit(0)

    print("=== Testing forced block selection ===")
    test_block_table_gather_after_sort()
    test_block_table_gather_identity_path()
    test_forced_zero_is_noop()
    test_forced_blocks()
    test_forced_blocks(force_begin=1, force_end=0, seed=10)
    test_forced_blocks(force_begin=0, force_end=5, seed=20)
    test_forced_blocks(force_begin=8, force_end=8, seed=30)
    test_forced_ascending_order()
    test_forced_with_xor_fast_path()
    test_forced_large_k()
    print("\nAll forced-block tests PASSED!")
