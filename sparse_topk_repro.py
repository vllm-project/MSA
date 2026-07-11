"""Standalone regression benchmark for the M3 sparse_topk_select slowdown.

Background
----------
In a steady-state decode nsys trace of MiniMax-M3, the original kernel
``flashinfer::sparse_topk::IndexerTopKWithSortKernel<16>`` (launched from
``sparse_topk_select`` in the MSA indexer decode path) shows up at ~34us typical
and up to ~82us for grid<~120,1,1>, while the same kernel in isolation runs in
~6us. This script reproduces the effect and shows it is *data-dependent* (tied
scores), NOT a memory-layout / coalescing issue.

Run from the MSA repository root:
    .venv/bin/python sparse_topk_repro.py

On an unfixed build, well-separated scores take ~6-16us while tied scores jump
to ~35us (nvp=782) / ~111us (nvp=1600).  A fixed build should keep the tied
cases close to the spread-score baseline and show no quadratic spike.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
PYTHON_ROOT = Path(os.environ.get("MSA_PYTHON_ROOT", REPO_ROOT / "python"))
sys.path.insert(0, str(PYTHON_ROOT))

from fmha_sm100 import sparse_topk_select  # noqa: E402

DEV = "cuda"
TOPK = 16
# Real decode shape from the trace: grid = total_qo_len * num_qo_heads = ~120,
# block=512.  max_k_tiles is round_up(cdiv(max_seq_len,128),128); with
# max_seq_len == max_model_len (1048576) this is 8192.
T, H, MK = 120, 1, 8192


def time_us(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000.0  # us (wall; includes launch overhead)


def call(scores, nvp):
    # Mirrors indexer_msa.py decode: THK input, per-token num_valid_pages tensor,
    # force blocks, 3D direct-gather block_table, head-major strided output view.
    out = torch.empty(H, T, TOPK, dtype=torch.int32, device=DEV).permute(1, 0, 2)
    nvpt = torch.full((T,), nvp, dtype=torch.int32, device=DEV)
    bt = (torch.arange(MK, dtype=torch.int32, device=DEV)
          .view(1, 1, MK).expand(T, H, MK).contiguous())
    return lambda: sparse_topk_select(
        max_score=scores, topk=TOPK, output=out, num_valid_pages=nvpt,
        force_begin_blocks=1, force_end_blocks=1,
        max_score_layout="THK", block_table=bt)


def make(nvp, kind):
    s = torch.empty(T, H, MK, dtype=torch.float32, device=DEV)
    s.fill_(float("-inf"))  # padding tail, exactly like the vLLM buffer
    g = torch.Generator(device=DEV).manual_seed(0)
    if kind == "spread":
        s[:, :, :nvp] = torch.randn(T, H, nvp, generator=g, device=DEV)
    elif kind == "equal":               # worst case: all valid scores identical
        s[:, :, :nvp] = 0.5
    return s


def _stress_scores(rows, max_k_tiles, nvp, kind):
    scores = torch.full(
        (rows, 1, max_k_tiles), -float("inf"), device=DEV, dtype=torch.float32
    )
    generator = torch.Generator(device=DEV).manual_seed(1234 + nvp + rows)
    if kind == "spread":
        scores[..., :nvp] = torch.randn(
            rows, 1, nvp, generator=generator, device=DEV
        )
    elif kind == "clustered":
        scores[..., :nvp] = 0.5 + torch.arange(
            nvp, device=DEV, dtype=torch.float32
        ) * 1e-7
    elif kind == "quantized":
        values = torch.randn(rows, 1, nvp, generator=generator, device=DEV)
        scores[..., :nvp] = torch.round(values * 16) / 16
    elif kind == "equal":
        scores[..., :nvp] = 0.5
    else:
        raise ValueError(f"unknown stress distribution: {kind}")
    return scores


def _refined_stress_scores(rows, max_k_tiles, final_count):
    """Force stage-0/1 overflow, then a stage-2 threshold bin."""
    scores = torch.full(
        (rows, 1, max_k_tiles), -float("inf"), device=DEV, dtype=torch.float32
    )
    low = torch.tensor([0x3F060FFF], device=DEV, dtype=torch.uint32).view(
        torch.float32
    )
    high = torch.tensor([0x3F070FFF], device=DEV, dtype=torch.uint32).view(
        torch.float32
    )
    scores[..., :3200] = low
    scores[..., :final_count] = high
    return scores


def _validate_selection(scores, output, nvp):
    assert output.dtype == torch.int32
    assert torch.all(output >= 0)
    assert torch.all(output < nvp)
    assert torch.all(output[..., 1:] > output[..., :-1])
    selected = torch.gather(scores[..., :nvp], 2, output.to(torch.long))
    threshold = torch.topk(scores[..., :nvp], TOPK, dim=-1).values[..., -1:]
    assert torch.all(selected >= threshold)


def stress(iters, soak_iters):
    rows = 120
    max_k_tiles = 8192
    output = torch.empty(rows, 1, TOPK, dtype=torch.int32, device=DEV)

    print(f"device={torch.cuda.get_device_name()} stress_iters={iters} soak_iters={soak_iters}")
    print("\n# Boundary-bin sweep (THK, grid=120, K=8192, all-equal)")
    print(f"{'nvp':>6} | {'time (us)':>10} | {'path':>12}")
    for nvp in (17, 64, 128, 256, 511, 512, 513, 514, 768, 782,
                1024, 1536, 1600, 2047, 2048, 2049, 3200, 4096, 8192):
        scores = _stress_scores(rows, max_k_tiles, nvp, "equal")
        fn = lambda: sparse_topk_select(
            scores, TOPK, num_valid_pages=nvp, output=output, max_score_layout="THK"
        )
        elapsed = time_us(fn, iters=iters, warmup=30)
        _validate_selection(scores, output, nvp)
        path = "rank" if nvp <= 416 else "merge" if nvp <= 2048 else "refine"
        print(f"{nvp:>6} | {elapsed:>10.2f} | {path:>12}")

    print("\n# Distribution sweep (THK, grid=120, K=8192, nvp=1600)")
    print(f"{'distribution':>12} | {'time (us)':>10}")
    for kind in ("spread", "clustered", "quantized", "equal"):
        scores = _stress_scores(rows, max_k_tiles, 1600, kind)
        fn = lambda: sparse_topk_select(
            scores, TOPK, num_valid_pages=1600, output=output, max_score_layout="THK"
        )
        elapsed = time_us(fn, iters=iters, warmup=30)
        _validate_selection(scores, output, 1600)
        print(f"{kind:>12} | {elapsed:>10.2f}")

    print("\n# Refinement-to-selection sweep (THK, grid=120, K=8192, nvp=3200)")
    print(f"{'finalCount':>10} | {'time (us)':>10} | {'path':>12}")
    for final_count in (256, 384, 385, 400, 416, 417, 511, 512, 513,
                        782, 1024, 1600, 2048):
        scores = _refined_stress_scores(rows, max_k_tiles, final_count)
        fn = lambda: sparse_topk_select(
            scores, TOPK, num_valid_pages=3200, output=output, max_score_layout="THK"
        )
        elapsed = time_us(fn, iters=iters, warmup=30)
        _validate_selection(scores, output, 3200)
        path = "rank" if final_count <= 384 else "merge"
        print(f"{final_count:>10} | {elapsed:>10.2f} | {path:>12}")

    print("\n# Padding sweep (stage-2 finalCount=512, nvp=3200)")
    print(f"{'max_k_tiles':>12} | {'time (us)':>10}")
    for padded_k in (3200, 4096, 6144, 8192, 10240, 12032):
        scores = _refined_stress_scores(rows, padded_k, 512)
        fn = lambda: sparse_topk_select(
            scores, TOPK, num_valid_pages=3200, output=output, max_score_layout="THK"
        )
        elapsed = time_us(fn, iters=iters, warmup=30)
        _validate_selection(scores, output, 3200)
        print(f"{padded_k:>12} | {elapsed:>10.2f}")

    print("\n# Grid sweep (THK, K=8192, nvp=1600, all-equal)")
    print(f"{'rows':>6} | {'time (us)':>10} | {'ns/row':>10}")
    for grid_rows in (1, 8, 32, 120, 256, 512, 1024):
        scores = _stress_scores(grid_rows, max_k_tiles, 1600, "equal")
        grid_output = torch.empty(
            grid_rows, 1, TOPK, dtype=torch.int32, device=DEV
        )
        fn = lambda: sparse_topk_select(
            scores,
            TOPK,
            num_valid_pages=1600,
            output=grid_output,
            max_score_layout="THK",
        )
        elapsed = time_us(fn, iters=iters, warmup=30)
        _validate_selection(scores, grid_output, 1600)
        print(f"{grid_rows:>6} | {elapsed:>10.2f} | {elapsed * 1000 / grid_rows:>10.2f}")

    print("\n# Layout/output sweep (grid=120, K=8192, nvp=1600, all-equal)")
    scores_thk = _stress_scores(rows, max_k_tiles, 1600, "equal")
    scores_hkt = scores_thk.permute(1, 2, 0).contiguous()
    strided_output = torch.empty(
        1, rows, TOPK, dtype=torch.int32, device=DEV
    ).permute(1, 0, 2)
    cases = (
        ("THK contiguous", scores_thk, output, "THK"),
        ("THK strided", scores_thk, strided_output, "THK"),
        ("HKT contiguous", scores_hkt, output, "HKT"),
    )
    for name, scores, case_output, layout in cases:
        fn = lambda: sparse_topk_select(
            scores,
            TOPK,
            num_valid_pages=1600,
            output=case_output,
            max_score_layout=layout,
        )
        elapsed = time_us(fn, iters=iters, warmup=30)
        _validate_selection(scores_thk, case_output, 1600)
        print(f"{name:>16} | {elapsed:>10.2f} us")

    print(f"\n# Soak ({soak_iters} launches, THK, grid=120, nvp=1600, all-equal)")
    fn = lambda: sparse_topk_select(
        scores_thk,
        TOPK,
        num_valid_pages=1600,
        output=output,
        max_score_layout="THK",
    )
    elapsed = time_us(fn, iters=soak_iters, warmup=30)
    _validate_selection(scores_thk, output, 1600)
    print(f"average={elapsed:.2f} us, launches={soak_iters}, status=PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-nvp",
        type=int,
        choices=(256, 512, 782, 1600, 3200),
        help="launch one case for ncu/nsys instead of running the timing table",
    )
    parser.add_argument(
        "--profile-kind", choices=("spread", "equal", "refined"), default="equal"
    )
    parser.add_argument("--profile-final-count", type=int, default=512)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--stress-iters", type=int, default=500)
    parser.add_argument("--soak-iters", type=int, default=10_000)
    args = parser.parse_args()

    if args.stress:
        stress(args.stress_iters, args.soak_iters)
        return

    if args.profile_nvp is not None:
        if args.profile_kind == "refined":
            if args.profile_nvp != 3200:
                parser.error("--profile-kind refined requires --profile-nvp 3200")
            profile_scores = _refined_stress_scores(T, MK, args.profile_final_count)
            profile_output = torch.empty(
                T, H, TOPK, dtype=torch.int32, device=DEV
            )
            fn = lambda: sparse_topk_select(
                profile_scores,
                TOPK,
                num_valid_pages=3200,
                output=profile_output,
                max_score_layout="THK",
            )
        else:
            fn = call(make(args.profile_nvp, args.profile_kind), args.profile_nvp)
        torch.cuda.nvtx.range_push(
            f"sparse_topk_{args.profile_kind}_nvp_{args.profile_nvp}"
        )
        fn()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        print(f"profile case complete: kind={args.profile_kind} nvp={args.profile_nvp}")
        return

    print(f"device={torch.cuda.get_device_name()}  T={T} H={H} MK={MK} topk={TOPK}")
    print("\n# Data-dependence: same shape/layout, only the score DISTRIBUTION changes")
    print(
        f"{'valid(nvp)':>10} | {'spread (us)':>12} | "
        f"{'all-equal (us)':>14} | {'equal/spread':>12}"
    )
    for nvp in (256, 512, 782, 1600, 3200):
        us_spread = time_us(call(make(nvp, "spread"), nvp))
        us_equal = time_us(call(make(nvp, "equal"), nvp))
        print(
            f"{nvp:>10} | {us_spread:>12.2f} | {us_equal:>14.2f} | "
            f"{us_equal / us_spread:>12.2f}x"
        )
    print("\nInterpretation:")
    print(" - Unfixed: nvp=782 ~= 35us and nvp=1600 ~= 111us (quadratic spike).")
    print(" - Fixed:   nvp=782/1600 stay near the spread baseline with no spike.")
    print(" - nvp=3200 uses the existing histogram-refinement/direct-fill path.")


if __name__ == "__main__":
    main()
