#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Benchmark sparse_topk_select with and without block_table gather.

The THK layout measures IndexerTopKWithSortKernel directly.  HKT includes the
transpose stage and is useful for end-to-end API timing.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from fmha_sm100 import sparse_topk_select  # noqa: E402
from fmha_sm100.bench_utils import bench_gpu_time  # noqa: E402


def _make_scores(total_qo_len, num_qo_heads, max_k_tiles, num_valid_pages, layout, device):
    scores_thk = torch.randn(
        total_qo_len, num_qo_heads, max_k_tiles, device=device, dtype=torch.float32
    )
    scores_thk[:, :, num_valid_pages:] = float("-inf")
    if layout == "THK":
        return scores_thk.contiguous()
    return scores_thk.permute(1, 2, 0).contiguous()


def _make_block_table(total_qo_len, num_qo_heads, max_k_tiles, device):
    t = torch.arange(total_qo_len, device=device, dtype=torch.int32).view(-1, 1, 1)
    h = torch.arange(num_qo_heads, device=device, dtype=torch.int32).view(1, -1, 1)
    k = torch.arange(max_k_tiles, device=device, dtype=torch.int32).view(1, 1, -1)
    return (t * 1_000_000 + h * 10_000 + (max_k_tiles - 1 - k)).contiguous()


def _summarize(samples):
    arr = np.asarray(samples, dtype=np.float64)
    return float(np.median(arr)), float(np.std(arr))


def run_case(args, layout):
    device = f"cuda:{args.device}"
    scores = _make_scores(
        args.total_qo_len, args.num_qo_heads, args.max_k_tiles,
        args.num_valid_pages, layout, device,
    )
    block_table = _make_block_table(
        args.total_qo_len, args.num_qo_heads, args.max_k_tiles, device,
    )
    out_orig = torch.empty(
        args.total_qo_len, args.num_qo_heads, args.topk, device=device, dtype=torch.int32
    )
    out_gather = torch.empty_like(out_orig)

    def original():
        sparse_topk_select(
            scores, args.topk, num_valid_pages=args.num_valid_pages,
            output=out_orig, max_score_layout=layout,
        )

    def with_block_table():
        sparse_topk_select(
            scores, args.topk, num_valid_pages=args.num_valid_pages,
            output=out_gather, max_score_layout=layout, block_table=block_table,
        )

    # Trigger JIT and validate that gather returns block-table values for the
    # exact logical selections produced by the original path.
    original()
    with_block_table()
    safe = out_orig.clamp_min(0).to(torch.long)
    expected = torch.gather(block_table, 2, safe)
    expected = torch.where(out_orig >= 0, expected, torch.full_like(expected, -1))
    if not torch.equal(out_gather, expected):
        raise RuntimeError(f"{layout}: block_table gather correctness check failed")

    orig_samples = bench_gpu_time(
        original, dry_run_time_ms=args.dry_run_ms, repeat_time_ms=args.repeat_ms,
        cold_l2_cache=not args.warm_l2,
    )
    gather_samples = bench_gpu_time(
        with_block_table, dry_run_time_ms=args.dry_run_ms, repeat_time_ms=args.repeat_ms,
        cold_l2_cache=not args.warm_l2,
    )
    orig_ms, orig_std = _summarize(orig_samples)
    gather_ms, gather_std = _summarize(gather_samples)
    regression = (gather_ms / orig_ms - 1.0) * 100.0

    print(
        f"{layout:>3} T={args.total_qo_len} H={args.num_qo_heads} K={args.max_k_tiles} "
        f"topk={args.topk} nvp={args.num_valid_pages}: "
        f"orig={orig_ms:.4f} ms (std {orig_std:.4f}), "
        f"block_table={gather_ms:.4f} ms (std {gather_std:.4f}), "
        f"regression={regression:+.2f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layout", choices=("THK", "HKT", "both"), default="THK")
    parser.add_argument("--total-qo-len", type=int, default=1024)
    parser.add_argument("--num-qo-heads", type=int, default=8)
    parser.add_argument("--max-k-tiles", type=int, default=8192)
    parser.add_argument("--num-valid-pages", type=int, default=None)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--dry-run-ms", type=int, default=200)
    parser.add_argument("--repeat-ms", type=int, default=2000)
    parser.add_argument("--warm-l2", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.cuda.set_device(args.device)
    if args.topk != 16:
        raise SystemExit("sparse_topk_select currently supports topk=16 only")
    if args.num_valid_pages is None:
        args.num_valid_pages = args.max_k_tiles
    if not (0 < args.num_valid_pages <= args.max_k_tiles):
        raise SystemExit("--num-valid-pages must be in (0, --max-k-tiles]")

    props = torch.cuda.get_device_properties(args.device)
    print(f"device=cuda:{args.device} {props.name}")
    layouts = ("THK", "HKT") if args.layout == "both" else (args.layout,)
    for layout in layouts:
        run_case(args, layout)


if __name__ == "__main__":
    main()
