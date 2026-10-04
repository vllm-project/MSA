#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""SM100 FMHA sweep benchmark using this repository's public fmha_sm100 API.

Mirrors the section layout and table format of the reference
``minimax-inference/third_party/fmha_sm100/tests/bench_sm100_fmha.py`` so the
two outputs can be compared row-for-row. Uses the dense ``fmha_sm100`` /
``fmha_sm100_plan`` API exposed at the package root, which supports ragged,
paged (``kv_indices``), and sparse (``kv_block_indexes``) paths plus NVFP4 K/V.

Sections:
  - prefill        : ragged dense, causal, TP1/TP2/TP4 head configs
  - paged_prefill  : paged dense, causal, page-table indirection
  - sparse_prefill : sparse paged, first top-K pages (bf16/fp8/nvfp4)
  - decode         : ragged dense, batch decode, MBU metric
  - paged_decode   : paged dense, batch decode, MBU metric
  - sparse_decode  : paged dense + sparse paged paired rows, MBU metric

Decode sections use h_q=64 / h_k=4 (same as prefill TP1).
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
# NVFP4 sparse is only exposed by the cute subdir API
# (sparse_atten_nvfp4_kv_func); the root fmha_sm100 dense entry does not
# carry NVFP4 K/V scales. Add the subdir to sys.path for the nvfp4 branch.
_SPARSE_SUBDIR = REPO_ROOT / "python" / "fmha_sm100" / "cute"
sys.path.insert(0, str(_SPARSE_SUBDIR))

from fmha_sm100.bench_utils import bench_gpu_time, attention_tflops  # noqa: E402
from fmha_sm100 import fmha_sm100, fmha_sm100_plan  # noqa: E402


def _load_nvfp4_backend():
    """Lazy import of the cute NVFP4 sparse API + CSR builder."""
    from interface import sparse_atten_nvfp4_kv_func  # noqa: E402
    from sparse_index_utils import build_k2q_csr  # noqa: E402
    from quantize import quantize_kv_bf16_to_nvfp4_128x4  # noqa: E402
    return sparse_atten_nvfp4_kv_func, build_k2q_csr, quantize_kv_bf16_to_nvfp4_128x4

# ─── Hardware constants ───
PEAK_TFLOPS = {"fp8": 4500.0, "bf16": 2250.0, "nvfp4": 4500.0}
HBM_PEAK_GBS = 8000.0

GPU_ID = 0
DTYPE = "fp8"
CSV_FILE = None
DRY_RUN_MS = 200
REPEAT_MS = 2000
INCLUDE_PLAN = False


def parse_int_list(value: str):
    return [int(item) for item in value.split(",") if item.strip()]


def should_run_section(selected_sections, section_name: str):
    return "all" in selected_sections or section_name in selected_sections


# ─── Output layout ───
# Console output is fixed-width and right-aligned so columns line up; the CSV
# (machine-readable) gets plain tab-separated values without thousands seps.
_COLUMNS = [
    ("q_len", 8, ">"),
    ("kv_len", 8, ">"),
    ("dtype", 6, ">"),
    ("bsz", 5, ">"),
    ("h_q", 4, ">"),
    ("h_kv", 5, ">"),
    ("dim", 4, ">"),
    ("lat(ms)", 11, ">"),
    ("std(ms)", 9, ">"),
    ("TFLOPs", 10, ">"),
    ("GB/s", 10, ">"),
    ("MFU/MBU", 9, ">"),
    ("split%", 8, ">"),
]
_CSV_HEADER = ["q_len", "kv_len", "dtype", "batch_size", "q_head", "kv_head",
               "head_dim", "latency_ms", "std_ms", "tflops", "gbs",
               "mfu_mbu_pct", "predicted_split_speedup_pct"]
_RULE = "-" * (sum(w for _, w, _ in _COLUMNS) + len(_COLUMNS) - 1)


def _csv_write(fields):
    if CSV_FILE:
        CSV_FILE.write("\t".join(str(f) for f in fields) + "\n")
        CSV_FILE.flush()


def emit_section(title: str):
    """Console banner + column header for a section/sub-group (not in CSV)."""
    header = " ".join(f"{name:{align}{w}}" for name, w, align in _COLUMNS)
    print(f"\n{title}", flush=True)
    print(_RULE, flush=True)
    print(header, flush=True)
    print(_RULE, flush=True)


def print_row(q, k, dtype, b, h_q, h_k, d, t_ms, std_ms, tflops, gbs, metric, speedup=1.0):
    values = [
        str(q), str(k), dtype, str(b), str(h_q), str(h_k), str(d),
        f"{t_ms:.4f}", f"{std_ms:.4f}", f"{tflops:.2f}", f"{gbs:.2f}",
        f"{metric * 100:.2f}%", f"{(speedup - 1) * 100:.2f}%",
    ]
    cells = [f"{val:{align}{w}}" for val, (_, w, align) in zip(values, _COLUMNS)]
    print(" ".join(cells), flush=True)
    _csv_write([q, k, dtype, b, h_q, h_k, d,
                f"{t_ms:.4f}", f"{std_ms:.4f}", f"{tflops:.4f}", f"{gbs:.4f}",
                f"{metric * 100:.4f}", f"{(speedup - 1) * 100:.4f}"])


def progress(msg: str):
    sys.stderr.write(f"  · {msg}\n")
    sys.stderr.flush()


@contextlib.contextmanager
def _suppress_stdout():
    """Swallow noisy kernel/plan prints (e.g. 'Too huge setting to output
    maxscore!') so they do not corrupt the aligned result table."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            yield
    finally:
        captured = buf.getvalue().strip()
        if captured:
            for line in captured.splitlines():
                sys.stderr.write(f"  [kernel] {line}\n")
            sys.stderr.flush()


def compute_bytes(b, h_q, h_k, q, k, d, elem_bytes):
    return (b * q * h_q * d * elem_bytes       # Q
          + b * k * h_k * d * elem_bytes * 2   # K + V
          + b * q * h_q * d * 2)               # O (always bf16)


def compute_nvfp4_sparse_bytes(b, h_q, h_k, q, k, d, sparse_blocks):
    q_bytes = b * q * h_q * d                   # Q is FP8
    kv_data_bytes = b * k * h_k * d             # K+V packed FP4: 2 * d / 2
    kv_scale_bytes = b * k * h_k * (d // 16) * 2
    o_partial_bytes = sparse_blocks * b * q * h_q * d * 2
    return q_bytes + kv_data_bytes + kv_scale_bytes + o_partial_bytes


def bench_dense(b, h_q, h_k, q_len, k_len, d, output_mode, causal, dtype_str,
                num_kv_splits=-1, use_mbu=False):
    """Ragged dense path (no paged indirection). Port of reference bench_flashinfer."""
    torch_dtype = torch.float8_e4m3fn if dtype_str == "fp8" else torch.bfloat16
    init_dtype = torch.half if torch_dtype.itemsize == 1 else torch_dtype
    device = f"cuda:{GPU_ID}"

    q = torch.randn(b * q_len, h_q, d, dtype=init_dtype, device=device).to(torch_dtype)
    k = torch.randn(b * k_len, h_k, d, dtype=init_dtype, device=device).to(torch_dtype)
    v = torch.randn(b * k_len, h_k, d, dtype=init_dtype, device=device).to(torch_dtype)

    qo_lens = torch.full((b,), q_len, dtype=torch.int32)
    kv_lens = torch.full((b,), k_len, dtype=torch.int32)

    def make_plan():
        return fmha_sm100_plan(qo_lens, kv_lens, h_q,
            num_kv_splits=num_kv_splits,
            output_maxscore=True,
            num_kv_heads=h_k,
        )

    plan_info = make_plan()

    out = torch.empty_like(q).to(torch.bfloat16)

    # When maxscore output would exceed the int32 / 4GB limit, the kernel
    # silently disables it. We then fall back to timing the O output so the
    # row still reports a valid latency instead of zero.
    maxscore_elems = h_q * ((k_len + 127) // 128) * b * q_len
    skip_maxscore = maxscore_elems >= (1 << 31)
    want_maxscore = output_mode in ("maxscore", "full") and not skip_maxscore
    fun = lambda: fmha_sm100(q, k, v,
        plan_info=make_plan() if INCLUDE_PLAN else plan_info, out=out,
        output_maxscore=want_maxscore,
        output_o=output_mode in ("o", "full") or skip_maxscore,
    )
    _ = fun()
    measurements = bench_gpu_time(fun, dry_run_time_ms=DRY_RUN_MS, repeat_time_ms=REPEAT_MS)
    mean_ms = float(np.median(measurements))
    std_ms = float(np.std(measurements))

    tflops = attention_tflops(
        torch.full((b,), q_len), torch.full((b,), k_len), d, d, h_q, causal, mean_ms,
    )
    elem_bytes = 1 if dtype_str == "fp8" else 2
    total_bytes = compute_bytes(b, h_q, h_k, q_len, k_len, d, elem_bytes)
    gbs = total_bytes / (mean_ms * 1e-3) / 1e9
    metric = gbs / HBM_PEAK_GBS if use_mbu else tflops / PEAK_TFLOPS[dtype_str]
    return mean_ms, std_ms, tflops, gbs, metric, plan_info[3]["predicted_speedup"]


def bench_paged(b, h_q, h_k, q_len, k_len, d, output_mode, causal, dtype_str,
                page_size=128, use_mbu=False):
    """Paged dense path (page-table indirection via shuffled kv_indices).

    Port of reference bench_flashinfer_paged: HND-like K/V cache layout
    [total_pages, h_k, page_size, d], shuffled page indices, explicit causal
    qo_offset = k_len - q_len per batch.
    """
    torch_dtype = torch.float8_e4m3fn if dtype_str == "fp8" else torch.bfloat16
    init_dtype = torch.half if torch_dtype.itemsize == 1 else torch_dtype
    device = f"cuda:{GPU_ID}"

    q = torch.randn(b * q_len, h_q, d, dtype=init_dtype, device=device).to(torch_dtype)

    pages_per_seq = (k_len + page_size - 1) // page_size
    total_pages = b * pages_per_seq
    k_cache = torch.randn(total_pages, h_k, page_size, d, dtype=init_dtype, device=device).to(torch_dtype)
    v_cache = torch.randn(total_pages, h_k, page_size, d, dtype=init_dtype, device=device).to(torch_dtype)

    perm = torch.randperm(total_pages, device=device, dtype=torch.int64)
    stride = (pages_per_seq + 3) // 4 * 4
    kv_indices = torch.zeros(b, stride, dtype=torch.int32, device=device)
    for i in range(b):
        kv_indices[i, :pages_per_seq] = perm[i * pages_per_seq:(i + 1) * pages_per_seq].to(torch.int32)

    qo_lens = torch.full((b,), q_len, dtype=torch.int32)
    kv_lens = torch.full((b,), k_len, dtype=torch.int32)

    qo_offset_val = k_len - q_len if causal else 0
    qo_offset_tensor = torch.full((b,), qo_offset_val, dtype=torch.int32) if causal else None

    def make_plan():
        return fmha_sm100_plan(qo_lens, kv_lens, h_q,
            qo_offset=qo_offset_tensor,
            page_size=page_size,
            output_maxscore=True,
            num_kv_heads=h_k,
        )

    plan_info = make_plan()

    out = torch.empty_like(q).to(torch.bfloat16)

    # See bench_dense: fall back to timing O when maxscore would exceed 4GB.
    maxscore_elems = h_q * ((k_len + 127) // 128) * b * q_len
    skip_maxscore = maxscore_elems >= (1 << 31)
    want_maxscore = output_mode in ("maxscore", "full") and not skip_maxscore
    fun = lambda: fmha_sm100(q, k_cache, v_cache,
        plan_info=make_plan() if INCLUDE_PLAN else plan_info, kv_indices=kv_indices, out=out,
        output_maxscore=want_maxscore,
        output_o=output_mode in ("o", "full") or skip_maxscore,
    )
    _ = fun()
    measurements = bench_gpu_time(fun, dry_run_time_ms=DRY_RUN_MS, repeat_time_ms=REPEAT_MS)
    mean_ms = float(np.median(measurements))
    std_ms = float(np.std(measurements))

    tflops = attention_tflops(
        torch.full((b,), q_len), torch.full((b,), k_len), d, d, h_q, causal, mean_ms,
    )
    elem_bytes = 1 if dtype_str == "fp8" else 2
    total_bytes = compute_bytes(b, h_q, h_k, q_len, k_len, d, elem_bytes)
    gbs = total_bytes / (mean_ms * 1e-3) / 1e9
    metric = gbs / HBM_PEAK_GBS if use_mbu else tflops / PEAK_TFLOPS[dtype_str]
    return mean_ms, std_ms, tflops, gbs, metric, plan_info[3]["predicted_speedup"]


def _make_first_topk_q2k(b, q_len, k_len, h_k, topk, blk_kv, device):
    """Build q2k [h_k, total_q, topK] selecting the first top-K KV blocks.

    Matches the reference sparse benchmark's "first top-K pages" pattern;
    causal-safe (early blocks only).
    """
    num_kv_blocks = (k_len + blk_kv - 1) // blk_kv
    actual = min(topk, num_kv_blocks)
    total_q = b * q_len
    q2k = torch.full((h_k, total_q, topk), -1, dtype=torch.int32, device=device)
    sel = torch.arange(actual, dtype=torch.int32, device=device)
    q2k[:, :, :actual] = sel.view(1, 1, -1)
    return q2k.contiguous(), actual


def bench_sparse_nvfp4(b, h_q, h_k, q_len, k_len, d, topk, causal, use_mbu=False):
    """NVFP4 sparse prefill via the cute sparse_atten_nvfp4_kv_func.

    The root fmha_sm100 dense entry does not carry NVFP4 K/V scales, so this
    path uses the subdir API directly with flat (non-paged) varlen K/V and a
    first-top-K CSR built by build_k2q_csr.
    """
    if d != 128:
        raise ValueError("NVFP4 sparse benchmark supports head_dim=128 only")
    if q_len <= 32:
        raise ValueError("NVFP4 sparse benchmark requires q_len > 32 to enter MM-SA-Nv path")
    blk_kv = 128
    device = f"cuda:{GPU_ID}"
    sparse_atten_nvfp4_kv_func, build_k2q_csr, quantize_kv_bf16_to_nvfp4_128x4 = _load_nvfp4_backend()

    total_q = b * q_len
    total_k = b * k_len
    q = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    k_src = torch.randn(total_k, h_k, d, dtype=torch.bfloat16, device=device)
    v_src = torch.randn(total_k, h_k, d, dtype=torch.bfloat16, device=device)
    k_q, v_q = quantize_kv_bf16_to_nvfp4_128x4(k_src, v_src)

    q2k, actual_block_num = _make_first_topk_q2k(b, q_len, k_len, h_k, topk, blk_kv, device)
    cu_seqlens_q = torch.tensor([0] + list(np.cumsum([q_len] * b)), dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0] + list(np.cumsum([k_len] * b)), dtype=torch.int32, device=device)
    total_rows = b * ((k_len + blk_kv - 1) // blk_kv)

    k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
        q2k, cu_seqlens_q, cu_seqlens_k, blk_kv,
        total_k=total_k, max_seqlen_k=k_len, max_seqlen_q=q_len,
        total_rows=total_rows, qhead_per_kv=h_q // h_k, return_schedule=True,
    )
    softmax_scale = d ** -0.5

    fun = lambda: sparse_atten_nvfp4_kv_func(
        q, k_q.data, v_q.data,
        k_q.scale_128x4, v_q.scale_128x4, k_q.global_scale, v_q.global_scale,
        k2q_row_ptr, k2q_q_indices, topk,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=q_len, max_seqlen_k=k_len, blk_kv=blk_kv,
        causal=causal, softmax_scale=softmax_scale,
        partial_dtype=torch.bfloat16, return_softmax_lse=False, schedule=schedule,
    )
    _ = fun()
    measurements = bench_gpu_time(fun, dry_run_time_ms=DRY_RUN_MS, repeat_time_ms=REPEAT_MS)
    mean_ms = float(np.median(measurements))
    std_ms = float(np.std(measurements))

    effective_k_len = min(actual_block_num * blk_kv, k_len)
    tflops = attention_tflops(
        torch.full((b,), q_len), torch.full((b,), effective_k_len), d, d, h_q, False, mean_ms,
    )
    total_bytes = compute_nvfp4_sparse_bytes(b, h_q, h_k, q_len, effective_k_len, d, actual_block_num)
    gbs = total_bytes / (mean_ms * 1e-3) / 1e9
    metric = gbs / HBM_PEAK_GBS if use_mbu else tflops / PEAK_TFLOPS["nvfp4"]
    return mean_ms, std_ms, tflops, gbs, metric, 1.0


def bench_sparse(b, h_q, h_k, q_len, k_len, d, output_mode, causal, dtype_str,
                 page_size=128, topk=16, use_mbu=False):
    """Sparse paged path with the first top-K pages selected for every token.

    bf16/fp8 go through the root fmha_sm100 dense entry (kv_block_indexes);
    nvfp4 is dispatched to bench_sparse_nvfp4 (subdir API).
    """
    if topk not in (4, 8, 16, 32):
        raise ValueError(f"sparse topk must be one of 4/8/16/32, got {topk}")
    if dtype_str == "nvfp4":
        return bench_sparse_nvfp4(b, h_q, h_k, q_len, k_len, d, topk, causal, use_mbu=use_mbu)
    torch_dtype = torch.float8_e4m3fn if dtype_str == "fp8" else torch.bfloat16
    init_dtype = torch.half if torch_dtype.itemsize == 1 else torch_dtype
    device = f"cuda:{GPU_ID}"

    q = torch.randn(b * q_len, h_q, d, dtype=init_dtype, device=device).to(torch_dtype)

    pages_per_seq = (k_len + page_size - 1) // page_size
    total_pages = b * pages_per_seq
    k_cache = torch.randn(total_pages, h_k, page_size, d, dtype=init_dtype, device=device).to(torch_dtype)
    v_cache = torch.randn(total_pages, h_k, page_size, d, dtype=init_dtype, device=device).to(torch_dtype)

    perm = torch.randperm(total_pages, device=device, dtype=torch.int64)
    stride = (pages_per_seq + 3) // 4 * 4
    kv_indices = torch.zeros(b, stride, dtype=torch.int32, device=device)
    for i in range(b):
        kv_indices[i, :pages_per_seq] = perm[i * pages_per_seq:(i + 1) * pages_per_seq].to(torch.int32)

    qo_lens = torch.full((b,), q_len, dtype=torch.int32)
    kv_lens = torch.full((b,), k_len, dtype=torch.int32)

    qo_offset_val = k_len - q_len if causal else 0
    qo_offset_tensor = torch.full((b,), qo_offset_val, dtype=torch.int32) if causal else None

    kv_block_num = topk
    actual_block_num = min(topk, pages_per_seq)
    total_q = b * q_len
    kv_block_indexes = torch.full((total_q, h_k, kv_block_num), -1, device=device, dtype=torch.int32)
    selected_blocks = torch.arange(actual_block_num, device=device, dtype=torch.int32)
    kv_block_indexes[:, :, :actual_block_num] = selected_blocks.view(1, 1, -1)

    def make_plan():
        return fmha_sm100_plan(qo_lens, kv_lens, h_q,
            qo_offset=qo_offset_tensor,
            page_size=page_size,
            kv_block_num=kv_block_num,
            num_kv_heads=h_k,
        )

    plan_info = make_plan()

    out = torch.empty_like(q).to(torch.bfloat16)

    fun = lambda: fmha_sm100(
        q, k_cache, v_cache,
        plan_info=make_plan() if INCLUDE_PLAN else plan_info,
        kv_indices=kv_indices, out=out,
        kv_block_indexes=kv_block_indexes,
    )
    _ = fun()
    measurements = bench_gpu_time(fun, dry_run_time_ms=DRY_RUN_MS, repeat_time_ms=REPEAT_MS)
    mean_ms = float(np.median(measurements))
    std_ms = float(np.std(measurements))

    effective_k_len = min(actual_block_num * page_size, k_len)
    tflops = attention_tflops(
        torch.full((b,), q_len), torch.full((b,), effective_k_len), d, d, h_q, False, mean_ms,
    )
    elem_bytes = 1 if dtype_str == "fp8" else 2
    total_bytes = compute_bytes(b, h_q, h_k, q_len, effective_k_len, d, elem_bytes)
    gbs = total_bytes / (mean_ms * 1e-3) / 1e9
    metric = gbs / HBM_PEAK_GBS if use_mbu else tflops / PEAK_TFLOPS[dtype_str]
    # Sparse plan_info is a dict without the dense "predicted_speedup" split
    # heuristic; report 1.0 (no split speedup) for these rows.
    if isinstance(plan_info, dict):
        predicted_speedup = plan_info.get("predicted_speedup", 1.0)
    else:
        predicted_speedup = plan_info[3].get("predicted_speedup", 1.0)
    return mean_ms, std_ms, tflops, gbs, metric, predicted_speedup


# TP head configs: (label, h_q, h_k) — matches reference TP1/TP2/TP4.
TP_CONFIGS = {
    1: ("TP1", 64, 4),
    2: ("TP2", 32, 2),
    4: ("TP4", 16, 1),
}
# Decode head config (all decode sections), per user: 64/4.
DECODE_H_Q = 64
DECODE_H_K = 4


def run_prefill(args):
    """Ragged dense prefill, causal, TP1/TP2/TP4 head configs, q=k diagonal."""
    for tp in args.tp:
        _, h_q, h_k = TP_CONFIGS[tp]
        emit_section(f"Prefill {TP_CONFIGS[tp][0]} (ragged dense, causal, {DTYPE})")
        for s in args.seqs:
            progress(f"Prefill {TP_CONFIGS[tp][0]} q=k={s}")
            try:
                with _suppress_stdout():
                    r = bench_dense(1, h_q, h_k, s, s, args.head_dim, args.output_mode,
                                    causal=True, dtype_str=DTYPE)
                print_row(s, s, DTYPE, 1, h_q, h_k, args.head_dim, *r)
            except Exception as e:
                progress(f"ERROR: {e}")


def run_paged_prefill(args):
    """Paged dense prefill, causal, TP1 head config, q=k diagonal."""
    _, h_q, h_k = TP_CONFIGS[1]
    emit_section(f"Paged Prefill TP1 (paged dense, causal, page=128, {DTYPE})")
    for s in args.seqs:
        progress(f"Paged Prefill q=k={s}")
        try:
            with _suppress_stdout():
                r = bench_paged(1, h_q, h_k, s, s, args.head_dim, args.output_mode,
                                causal=True, dtype_str=DTYPE, page_size=args.blk_kv)
            print_row(s, s, DTYPE, 1, h_q, h_k, args.head_dim, *r)
        except Exception as e:
            progress(f"ERROR: {e}")


def run_sparse_prefill(args):
    """Sparse paged prefill, first top-K pages, TP1, q=k diagonal."""
    _, h_q, h_k = TP_CONFIGS[1]
    emit_section(f"Sparse Prefill TP1 (first top-K, page=128, topk={args.topk}, {DTYPE})")
    for s in args.seqs:
        progress(f"Sparse Prefill q=k={s} topk={args.topk}")
        try:
            with _suppress_stdout():
                r = bench_sparse(1, h_q, h_k, s, s, args.head_dim, args.output_mode,
                                 causal=True, dtype_str=DTYPE, page_size=args.blk_kv, topk=args.topk)
            print_row(s, s, DTYPE, 1, h_q, h_k, args.head_dim, *r)
        except Exception as e:
            progress(f"ERROR: {e}")


def run_decode(args):
    """Ragged dense batch decode, MBU metric, h_q=64/h_k=4. q_len=4 matches reference."""
    emit_section(f"Decode (ragged dense, MBU, h={DECODE_H_Q}/{DECODE_H_K}, {DTYPE})")
    decode_q = 4
    for b in args.decode_b:
        for k in args.decode_k:
            progress(f"Decode q={decode_q} k={k} b={b}")
            try:
                with _suppress_stdout():
                    r = bench_dense(b, DECODE_H_Q, DECODE_H_K, decode_q, k, args.head_dim,
                                    args.output_mode, causal=True, dtype_str=DTYPE, use_mbu=True)
                print_row(decode_q, k, DTYPE, b, DECODE_H_Q, DECODE_H_K, args.head_dim, *r)
            except Exception as e:
                progress(f"ERROR: {e}")


def run_paged_decode(args):
    """Paged dense batch decode, MBU metric, h_q=64/h_k=4. q_len=4 matches reference."""
    emit_section(f"Paged Decode (paged dense, MBU, page=128, h={DECODE_H_Q}/{DECODE_H_K}, {DTYPE})")
    decode_q = 4
    for b in args.decode_b:
        for k in args.decode_k:
            progress(f"Paged Decode q={decode_q} k={k} b={b}")
            try:
                with _suppress_stdout():
                    r = bench_paged(b, DECODE_H_Q, DECODE_H_K, decode_q, k, args.head_dim,
                                    args.output_mode, causal=True, dtype_str=DTYPE,
                                    page_size=args.blk_kv, use_mbu=True)
                print_row(decode_q, k, DTYPE, b, DECODE_H_Q, DECODE_H_K, args.head_dim, *r)
            except Exception as e:
                progress(f"ERROR: {e}")


def run_sparse_decode(args):
    """Sparse paged decode: paged dense + sparse paged paired rows. MBU metric."""
    emit_section(f"Sparse Decode (dense+sparse pairs, MBU, topk={args.topk}, {DTYPE})")
    sparse_decode_q = 8
    for b in args.decode_b:
        for k in args.decode_k:
            progress(f"Sparse Decode q={sparse_decode_q} k={k} b={b} topk={args.topk}")
            try:
                with _suppress_stdout():
                    r = bench_paged(b, DECODE_H_Q, DECODE_H_K, sparse_decode_q, k, args.head_dim,
                                    args.output_mode, causal=True, dtype_str=DTYPE,
                                    page_size=args.blk_kv, use_mbu=True)
                print_row(sparse_decode_q, k, DTYPE, b, DECODE_H_Q, DECODE_H_K, args.head_dim, *r)
                with _suppress_stdout():
                    r = bench_sparse(b, DECODE_H_Q, DECODE_H_K, sparse_decode_q, k, args.head_dim,
                                     args.output_mode, causal=True, dtype_str=DTYPE,
                                     page_size=args.blk_kv, topk=args.topk, use_mbu=True)
                print_row(sparse_decode_q, k, DTYPE, b, DECODE_H_Q, DECODE_H_K, args.head_dim, *r)
            except Exception as e:
                progress(f"ERROR: {e}")


SECTIONS = ["prefill", "paged_prefill", "sparse_prefill", "decode", "paged_decode", "sparse_decode"]
NVFP4_SECTIONS = {"sparse_prefill"}


def main():
    global GPU_ID, DTYPE, CSV_FILE, DRY_RUN_MS, REPEAT_MS, INCLUDE_PLAN

    parser = argparse.ArgumentParser(description="MiniMax sparse attention sweep benchmark")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_mode", choices=["full", "maxscore", "o"], default="o")
    parser.add_argument("--dtype", choices=["fp8", "bf16", "nvfp4"], default="fp8")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--sections", type=str, default="all",
        help="Comma-separated: " + ",".join(SECTIONS) + ",all")
    parser.add_argument("--seqs", type=str, default=None,
        help="Comma-separated q=k lengths for prefill sections")
    parser.add_argument("--tp", type=str, default="1,2,4",
        help="Comma-separated TP configs for prefill: 1,2,4")
    parser.add_argument("--decode-k", type=str, default=None,
        help="Comma-separated KV lengths for decode sections")
    parser.add_argument("--decode-b", type=str, default=None,
        help="Comma-separated batch sizes for decode sections")
    parser.add_argument("--topk", type=int, default=16, choices=[4, 8, 16, 32],
        help="Sparse top-K pages per token")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--blk-kv", type=int, default=128)
    parser.add_argument("--dry-run-ms", type=int, default=200)
    parser.add_argument("--repeat-ms", type=int, default=2000)
    parser.add_argument("--include-plan", action="store_true",
                        help="Include FP8/BF16 plan construction in the existing CUDA-event timed callback")
    args = parser.parse_args()

    if args.head_dim != 128:
        raise ValueError("current kernels require --head-dim 128")
    if args.blk_kv != 128:
        raise ValueError("current kernels require --blk-kv 128")

    GPU_ID = args.gpu
    DTYPE = args.dtype
    DRY_RUN_MS = args.dry_run_ms
    REPEAT_MS = args.repeat_ms
    INCLUDE_PLAN = args.include_plan
    torch.cuda.set_device(GPU_ID)

    if args.output:
        CSV_FILE = open(args.output, "w")
        CSV_FILE.write("\t".join(_CSV_HEADER) + "\n")
        CSV_FILE.flush()
        sys.stderr.write(f"Output file: {args.output}\n")

    args.seqs = parse_int_list(args.seqs) if args.seqs else [8192, 16384, 32768, 65536, 131072]
    args.tp = [int(x) for x in parse_int_list(args.tp)]
    for tp in args.tp:
        if tp not in TP_CONFIGS:
            raise ValueError(f"unknown TP config {tp}, valid: {sorted(TP_CONFIGS)}")
    args.decode_k = parse_int_list(args.decode_k) if args.decode_k else [8192, 16384, 32768, 65536]
    args.decode_b = parse_int_list(args.decode_b) if args.decode_b else [32, 64, 128]

    selected_sections = {item.strip().lower() for item in args.sections.split(",") if item.strip()}
    unknown = selected_sections - set(SECTIONS) - {"all"}
    if unknown:
        raise ValueError(f"unknown sections: {sorted(unknown)}. Valid: {SECTIONS + ['all']}")

    if DTYPE == "nvfp4":
        if "all" in selected_sections:
            selected_sections = set(NVFP4_SECTIONS)
        else:
            for section in sorted(selected_sections - NVFP4_SECTIONS - {"all"}):
                sys.stderr.write(f"  SKIP: dtype=nvfp4 only supports sparse_prefill (skipping '{section}')\n")
            selected_sections &= NVFP4_SECTIONS
        if not selected_sections:
            raise ValueError("dtype=nvfp4 supports only sparse_prefill in this benchmark")

    ordered = [s for s in SECTIONS if should_run_section(selected_sections, s)]
    print(f"MiniMax sparse attention sweep  |  GPU {GPU_ID}  |  dtype {DTYPE}  |  "
          f"output_mode {args.output_mode}", flush=True)
    print(f"sections: {', '.join(ordered)}", flush=True)

    if should_run_section(selected_sections, "prefill"):
        run_prefill(args)
    if should_run_section(selected_sections, "paged_prefill"):
        run_paged_prefill(args)
    if should_run_section(selected_sections, "sparse_prefill"):
        run_sparse_prefill(args)
    if should_run_section(selected_sections, "decode"):
        run_decode(args)
    if should_run_section(selected_sections, "paged_decode"):
        run_paged_decode(args)
    if should_run_section(selected_sections, "sparse_decode"):
        run_sparse_decode(args)

    print("", flush=True)
    if CSV_FILE:
        CSV_FILE.close()
        sys.stderr.write(f"Results saved to {args.output}\n")


if __name__ == "__main__":
    main()
