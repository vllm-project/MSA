# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""CUDA Graph decode benchmark with rotating cold-L2 input slots.

Q8KV4 (NVFP4 K) decode indexer: BatchDecodeIndexerQ8KV4Wrapper.run (scores + TopK).
MiniMax-AI/MSA nv_dev's benchmarks/inference/msa_v1/indexer/decode/q8kv4/benchmark.py
(2913dfd) with the modules it imports, folded into one script; each section names its
nv_dev source file. Port changes for fmha_sm100:
- BatchDecodeIndexerWithPagedKVCacheWrapper forwards to BatchDecodeIndexerQ8KV4Wrapper;
  K is packed into one vLLM cache tensor [pages, 128, 72] (make_k_cache) and the TopK
  output is [batch * Q, H, 16].
- --verify runs the checks of test_q8_indexer.py (query length 8).
- check_benchmark_device accepts an exclusive SM107 (Rubin) GPU.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import socket
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
# fmha_sm100 and, for --verify, the reference helpers of test_q8_indexer.py.
for path in (REPO_ROOT / "python", REPO_ROOT / "python" / "fmha_sm100" / "cute"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fmha_sm100.cute.q8_indexer_interface import BatchDecodeIndexerQ8KV4Wrapper  # noqa: E402


# ---- nv_dev benchmarks/inference/msa_v1/decode/cases.py ----

DEFAULT_SEED = 1701
BATCH_SIZES = (8, 32, 64, 128)
SEQ_LENGTHS = (1_000, 4_000, 5_000, 10_000, 50_000, 100_000, 200_000)
LOW_LATENCY_BATCH_SIZES = (1, 2, 4)
LOW_LATENCY_SEQ_LENGTHS = (1_000, 4_000, 8_000, 32_000)
Q_LEN_PER_REQ = 8

_WEIGHTS = {
    1_000: (30, 90, 180, 300),
    4_000: (40, 120, 240, 400),
    5_000: (50, 150, 300, 500),
    10_000: (140, 350, 490, 420),
    50_000: (500, 800, 500, 200),
    100_000: (1260, 980, 420, 140),
    200_000: (910, 350, 112, 28),
}


@dataclass(frozen=True)
class DecodeBenchmarkCase:
    batch_size: int
    nominal_seq_len: int
    weight: int
    q_len_per_req: int = Q_LEN_PER_REQ
    seed: int = DEFAULT_SEED

    @property
    def name(self) -> str:
        return f"b{self.batch_size}_s{self.nominal_seq_len}_q{self.q_len_per_req}"

    @property
    def batch(self) -> int:
        """Compatibility spelling used by the existing benchmark storage."""

        return self.batch_size

    @property
    def hard_gate(self) -> bool:
        return True


FULL_CASES = tuple(
    DecodeBenchmarkCase(
        batch_size=batch_size,
        nominal_seq_len=seq_len,
        weight=_WEIGHTS[seq_len][batch_index],
    )
    for seq_len in SEQ_LENGTHS
    for batch_index, batch_size in enumerate(BATCH_SIZES)
)
# Zero weight keeps this independent latency suite out of production aggregates.
LOW_LATENCY_CASES = tuple(
    DecodeBenchmarkCase(batch_size=batch_size, nominal_seq_len=seq_len, weight=0)
    for seq_len in LOW_LATENCY_SEQ_LENGTHS
    for batch_size in LOW_LATENCY_BATCH_SIZES
)
SMOKE_CASES = tuple(
    case
    for case in FULL_CASES
    if case.nominal_seq_len in (1_000, 100_000, 200_000) and case.batch_size in (8, 128)
)


def benchmark_cases(suite: str) -> tuple[DecodeBenchmarkCase, ...]:
    if suite == "smoke":
        return SMOKE_CASES
    if suite == "full":
        return FULL_CASES
    if suite == "low-latency":
        return LOW_LATENCY_CASES
    raise ValueError("suite must be 'smoke', 'full', or 'low-latency'")


def make_seq_lens(case: DecodeBenchmarkCase) -> torch.Tensor:
    """Return deterministic lengths; a singleton uses the exact nominal length."""

    generator = torch.Generator().manual_seed(
        case.seed + case.batch_size * 1009 + case.nominal_seq_len
    )
    pair_count = case.batch_size // 2
    minimum = max(1, case.nominal_seq_len // 20)
    maximum = max(minimum, case.nominal_seq_len // 4)
    deltas = torch.randint(
        minimum,
        maximum + 1,
        (pair_count,),
        generator=generator,
        dtype=torch.int64,
    )
    parts = [case.nominal_seq_len - deltas, case.nominal_seq_len + deltas]
    if case.batch_size % 2:
        # Odd batch: one request at the nominal length keeps the mean exact.
        parts.append(torch.tensor([case.nominal_seq_len], dtype=torch.int64))
    values = torch.cat(parts)
    values = values[torch.randperm(case.batch_size, generator=generator)]
    assert int(values.sum()) == case.batch_size * case.nominal_seq_len
    assert int(values.min()) >= case.q_len_per_req
    return values.to(torch.int32)


assert len(FULL_CASES) == 28
assert sum(case.weight for case in FULL_CASES) == 10_000
assert all(case.weight > 0 for case in FULL_CASES)


# ---- nv_dev benchmarks/inference/msa_v1/indexer/decode/q8kv4/cases.py ----

STANDARD_CASES = FULL_CASES
CORE_CASES = FULL_CASES
STRESS_CASES = ()
DecodeCase = DecodeBenchmarkCase


# ---- nv_dev benchmarks/inference/msa_v1/hardware.py ----

@dataclass(frozen=True)
class PeakSpec:
    hbm_tb_s: float
    dense_fp8_tflops: float
    dense_bf16_tflops: float
    source: str


_DENSE_FP8_FMAS_PER_CYCLE_PER_SM = 8192
_DENSE_BF16_FMAS_PER_CYCLE_PER_SM = 4096
_PEAKS = {
    "NVIDIA B200": PeakSpec(
        hbm_tb_s=8.0,
        dense_fp8_tflops=(_DENSE_FP8_FMAS_PER_CYCLE_PER_SM * 148 * 1.83e9 * 2 / 1.0e12),
        dense_bf16_tflops=(
            _DENSE_BF16_FMAS_PER_CYCLE_PER_SM * 148 * 1.83e9 * 2 / 1.0e12
        ),
        source="public B200/GB200 SM100 specification",
    ),
    "NVIDIA GB200": PeakSpec(
        hbm_tb_s=8.0,
        dense_fp8_tflops=(_DENSE_FP8_FMAS_PER_CYCLE_PER_SM * 148 * 1.83e9 * 2 / 1.0e12),
        dense_bf16_tflops=(
            _DENSE_BF16_FMAS_PER_CYCLE_PER_SM * 148 * 1.83e9 * 2 / 1.0e12
        ),
        source="public B200/GB200 SM100 specification",
    ),
    "NVIDIA GB300": PeakSpec(
        hbm_tb_s=8.0,
        dense_fp8_tflops=(_DENSE_FP8_FMAS_PER_CYCLE_PER_SM * 152 * 2.07e9 * 2 / 1.0e12),
        dense_bf16_tflops=(
            _DENSE_BF16_FMAS_PER_CYCLE_PER_SM * 152 * 2.07e9 * 2 / 1.0e12
        ),
        source="public SM100 rates with GB300 device SM count and maximum SM clock",
    ),
    # B300 SXM6 (air-cooled, x86 host): device attributes report 148 SMs, 2.032 GHz max SM
    # clock, HBM3e 7680-bit x 3.996 GHz DDR = 7.67 TB/s (the public 8.0 TB/s figure assumes
    # the full 8192-bit bus). Tensor rates use the public SM100 per-SM rates.
    "NVIDIA B300 SXM6 AC": PeakSpec(
        hbm_tb_s=7.67,
        dense_fp8_tflops=(
            _DENSE_FP8_FMAS_PER_CYCLE_PER_SM * 148 * 2.032e9 * 2 / 1.0e12
        ),
        dense_bf16_tflops=(
            _DENSE_BF16_FMAS_PER_CYCLE_PER_SM * 148 * 2.032e9 * 2 / 1.0e12
        ),
        source="public SM100 rates with B300 SXM6 device SM count, clock and memory bus",
    ),
    # Rubin (SM107) engineering sample reports the generic pre-release name.
    # Provisional: HBM peak is the DRAM interface peak of this sample (15360-bit
    # bus, DDR at the 3.186 GHz maximum memory clock = 3840 B/cycle = 12.23 TB/s,
    # which is also the Nsight Compute DRAM peak). The 4.995 GHz reported by
    # cudaDevAttrMemoryClockRate overstates it (19.2 TB/s) and is not used.
    # Tensor Core peak assumes 2x the SM100 per-SM FP8 rate and 1x the SM100
    # per-SM BF16 rate at 208 SMs / 2.424 GHz.
    "NVIDIA Graphics Device": PeakSpec(
        hbm_tb_s=12.23,
        dense_fp8_tflops=(
            2 * _DENSE_FP8_FMAS_PER_CYCLE_PER_SM * 208 * 2.424e9 * 2 / 1.0e12
        ),
        dense_bf16_tflops=(
            _DENSE_BF16_FMAS_PER_CYCLE_PER_SM * 208 * 2.424e9 * 2 / 1.0e12
        ),
        source="provisional SM107 estimate (ncu DRAM peak, SM100-scaled tensor rates); not a public specification",
    ),
}


def peak_spec(device_name: str) -> PeakSpec:
    try:
        return _PEAKS[device_name]
    except KeyError as error:
        raise ValueError(
            f"no public MBU/MFU peak specification is registered for {device_name!r}"
        ) from error


def roofline_metrics(
    *,
    device_name: str,
    useful_flops: int,
    logical_bytes: int,
    latency_us: float,
    compute_dtype: str = "fp8",
) -> dict[str, float | str]:
    """Return logical-work MBU/MFU diagnostics without changing E2E acceptance."""

    spec = peak_spec(device_name)
    if compute_dtype == "fp8":
        peak_tflops = spec.dense_fp8_tflops
    elif compute_dtype == "bf16":
        peak_tflops = spec.dense_bf16_tflops
    else:
        raise ValueError("compute_dtype must be 'fp8' or 'bf16'")
    observed_tflops = useful_flops / latency_us / 1.0e6
    observed_tb_s = logical_bytes / latency_us / 1.0e6
    arithmetic_intensity = useful_flops / logical_bytes
    ridge_point = peak_tflops / spec.hbm_tb_s
    bound = "memory" if arithmetic_intensity < ridge_point else "compute"
    mbu = observed_tb_s / spec.hbm_tb_s
    mfu = observed_tflops / peak_tflops
    return {
        "peak_hbm_tb_s": spec.hbm_tb_s,
        f"peak_dense_{compute_dtype}_tflops": peak_tflops,
        "peak_spec_source": spec.source,
        "logical_arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "roofline_ridge_flops_per_byte": ridge_point,
        "roofline_bound": bound,
        "logical_mbu": mbu,
        "logical_mfu": mfu,
        "roofline_efficiency": mbu if bound == "memory" else mfu,
    }


# ---- nv_dev benchmarks/inference/msa_v1/indexer/decode/metrics.py ----

MINIMUM_MBU_RATIO = 0.99
MINIMUM_Q8KV4_MULTIHEAD_MBU_RATIO = 0.90


def necessary_bytes(
    lengths, heads, page_bytes, *, query_length=8, query_element_bytes=1
):
    """Count shared K once per request; do not credit redundant kernel traffic."""
    local = (
        lengths[:, None].to(torch.int64) - query_length + torch.arange(query_length)
    ) // 128
    pages = int(local[:, -1].sum())
    scores = int(local.sum())
    selected_scores = int(local.masked_select(local >= 16).sum())
    batch = lengths.numel()
    return (
        pages * page_bytes
        + batch * query_length * heads * 128 * query_element_bytes
        + pages * 4
        + batch * 4
        + batch * query_length * heads * 4
        + heads * (scores + selected_scores) * 4
        + heads * batch * query_length * 16 * 4
    )


def compare_mbu(candidate, baseline):
    """Compare identical full protocols using aggregate bytes over aggregate time."""
    for key in ("protocol", "device", "device_uuid", "dsl_version", "precision"):
        if candidate[key] != baseline[key]:
            raise ValueError(f"baseline/candidate {key} mismatch")
    if not candidate["protocol"]["formal"]:
        raise ValueError("acceptance requires the full unfiltered manifest")
    old = {row["name"]: row for row in baseline["results"]}
    rows = candidate["results"]
    if len(rows) != 28 or len(old) != 28 or set(old) != {r["name"] for r in rows}:
        raise ValueError("acceptance requires the same 28 cases")
    heads = candidate["num_index_heads"]
    if baseline["num_index_heads"] != 1:
        raise ValueError("baseline must be the original H=1 implementation")
    totals = [0.0, 0.0, 0.0, 0.0]
    regressed = []
    for row in rows:
        previous = old[row["name"]]
        if row["weight"] != previous["weight"] or row["slots"] != previous["slots"]:
            raise ValueError("case weights and cold-cache slots must match")
        for value in (row, previous):
            if (
                not math.isfinite(value["cv"])
                or value["cv"] > 0.03
                or value["latency_us"] <= 0
            ):
                raise ValueError("invalid timing")
        w = row["weight"]
        totals[0] += w * row["necessary_bytes"]
        totals[1] += w * row["latency_us"]
        totals[2] += w * previous["necessary_bytes"]
        totals[3] += w * previous["latency_us"]
        if heads == 1 and row["latency_us"] > previous["latency_us"] * 1.05:
            regressed.append(row["name"])
    ratio = (totals[0] / totals[1]) / (totals[2] / totals[3])
    minimum_ratio = (
        MINIMUM_Q8KV4_MULTIHEAD_MBU_RATIO
        if candidate["precision"] == "q8kv4" and heads > 1
        else MINIMUM_MBU_RATIO
    )
    return {
        "passed": ratio >= minimum_ratio and not regressed,
        "minimum_weighted_effective_mbu_ratio": minimum_ratio,
        "weighted_effective_mbu_ratio": ratio,
        "h1_regressed_cases": regressed,
    }


def compare_low_latency(candidate, baseline):
    """Gate every low-latency case at the same H without a weighted aggregate."""

    for key in (
        "protocol",
        "device",
        "device_uuid",
        "dsl_version",
        "precision",
        "num_index_heads",
    ):
        if candidate[key] != baseline[key]:
            raise ValueError(f"baseline/candidate {key} mismatch")
    if (
        candidate["protocol"]["suite"] != "low-latency"
        or not candidate["protocol"]["formal"]
    ):
        raise ValueError("acceptance requires the low-latency suite")
    for key, expected_value in (
        ("warmup_replays", 5),
        ("timed_replays", 20),
        ("cuda_graph_calls", 120),
    ):
        if candidate["protocol"].get(key) != expected_value:
            raise ValueError(
                "low-latency acceptance requires the standard timing protocol"
            )
    expected = {case.name for case in LOW_LATENCY_CASES}
    mappings = []
    for payload in (candidate, baseline):
        rows = payload["results"]
        mapping = {row["name"]: row for row in rows}
        if len(rows) != len(expected) or set(mapping) != expected:
            raise ValueError("acceptance requires all 12 low-latency cases")
        for row in rows:
            if (
                not math.isfinite(row["cv"])
                or not 0 <= row["cv"] <= 0.03
                or not math.isfinite(row["latency_us"])
                or row["latency_us"] <= 0
                or row["reuse_distance_over_l2"] < 2
                or not math.isfinite(row["reuse_distance_over_l2"])
            ):
                raise ValueError("invalid timing or cold-cache coverage")
        mappings.append(mapping)
    ratios = {}
    for case in LOW_LATENCY_CASES:
        row, previous = (mapping[case.name] for mapping in mappings)
        for key in (
            "weight",
            "slots",
            "seed",
            "batch_size",
            "nominal_seq_len",
            "necessary_bytes",
        ):
            if row[key] != previous[key]:
                raise ValueError(f"case {case.name} {key} mismatch")
        if row["weight"] != 0:
            raise ValueError("low-latency cases must not carry production weights")
        ratios[case.name] = row["latency_us"] / previous["latency_us"]
    regressed = [name for name, ratio in ratios.items() if ratio > 1.05]
    return {
        "passed": not regressed,
        "maximum_latency_ratio": 1.05,
        "latency_ratios": ratios,
        "regressed_cases": regressed,
    }


# ---- nv_dev benchmarks/inference/msa_v1/indexer/decode/runtime.py ----

def _check_exclusive_rubin_gpu():
    # Port for the vllm-project/MSA fmha_sm100 indexer on Rubin (SM107): the B200/GB300
    # name and NVLink checks below do not apply; keep the exclusive-GPU check.
    hostname = socket.gethostname().split(".")[0]
    properties = torch.cuda.get_device_properties("cuda")
    context_probe = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    gpu_uuid = str(properties.uuid)
    if not gpu_uuid.startswith("GPU-"):
        gpu_uuid = "GPU-" + gpu_uuid
    processes = subprocess.check_output(
        ["nvidia-smi", "--id=" + gpu_uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    if len(processes.split()) > 1:
        raise RuntimeError("benchmark requires an exclusive GPU")
    del context_probe
    return hostname, properties


def check_benchmark_device():
    if torch.cuda.get_device_capability() == (10, 7):
        return _check_exclusive_rubin_gpu()
    hostname = socket.gethostname().split(".")[0]
    if "galaxy" in hostname.lower() or hostname in ("2u2g-gen-0176", "2u2g-gen-0300"):
        raise RuntimeError("benchmark excludes B300A and galaxy nodes")
    properties = torch.cuda.get_device_properties("cuda")
    if (
        not any(model in properties.name for model in ("B200", "GB300"))
        or "B300A" in properties.name
    ):
        raise RuntimeError(
            "benchmark requires B200/GB200 or GB300 NVL, excluding B300A"
        )
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        raise RuntimeError("benchmark requires SM100 or SM103")
    # Establish our context before counting processes; container PIDs differ from host PIDs.
    context_probe = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    gpu_uuid = str(properties.uuid)
    if not gpu_uuid.startswith("GPU-"):
        gpu_uuid = "GPU-" + gpu_uuid
    gpu_index = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + gpu_uuid,
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    rows = (line.split() for line in topology.splitlines())
    links = next(
        (row[1:] for row in rows if row and row[0] == "GPU" + gpu_index and "X" in row),
        [],
    )
    if not any(link.startswith("NV") and link[2:].isdigit() for link in links):
        raise RuntimeError("benchmark requires a verified NVLink topology")
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + gpu_uuid,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    if len(processes.split()) != 1:
        raise RuntimeError("benchmark requires an exclusive GPU")
    del context_probe
    return hostname, properties


# ---- nv_dev benchmarks/inference/msa_v1/indexer/decode/q8kv4/benchmark.py ----

PAGE_SIZE = 128
HEAD_DIM = 128
PAGE_BYTES = 128 * 64 + 128 * 8
MAX_CV = 0.03
MAX_TIMING_ATTEMPTS = 3


# Port: nv_dev wrapper interface over fmha_sm100 (was inference.msa_v1.indexer.decode.q8kv4.interface).
# K is one vLLM cache tensor [pages, 128, 72] uint8 (per page: 8192 packed NVFP4 bytes, then
# 1024 E4M3 scale bytes); q [batch, Q, H, 128] is viewed as [batch * Q, H, 128].


def make_k_cache(packed_k: torch.Tensor, k_scale: torch.Tensor) -> torch.Tensor:
    """Pack [pages, 128, 64] codes and [pages, 128, 8] E4M3 scales into the vLLM cache."""

    pages = packed_k.shape[0]
    data_bytes = PAGE_SIZE * HEAD_DIM // 2
    page_bytes = data_bytes + PAGE_SIZE * HEAD_DIM // 16
    raw = torch.empty(pages, page_bytes, dtype=torch.uint8, device=packed_k.device)
    raw[:, :data_bytes] = packed_k.reshape(pages, data_bytes)
    raw[:, data_bytes:] = k_scale.view(torch.uint8).reshape(pages, -1)
    return raw.unflatten(1, (PAGE_SIZE, page_bytes // PAGE_SIZE))


class BatchDecodeIndexerWithPagedKVCacheWrapper:
    def __init__(
        self,
        workspace_buffer: torch.Tensor | None = None,
        *,
        use_cuda_graph: bool = False,
        page_table_buffer: torch.Tensor | None = None,
        seq_lens_buffer: torch.Tensor | None = None,
    ) -> None:
        self._kwargs = dict(
            workspace_buffer=workspace_buffer,
            use_cuda_graph=use_cuda_graph,
            block_table_buffer=page_table_buffer,
            seq_lens_buffer=seq_lens_buffer,
        )
        self.wrapper: BatchDecodeIndexerQ8KV4Wrapper | None = None

    def plan(
        self,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        num_index_heads: int = 1,
        query_length: int = 8,
    ) -> None:
        if self.wrapper is None:
            self.wrapper = BatchDecodeIndexerQ8KV4Wrapper(num_heads=num_index_heads, **self._kwargs)
        self.wrapper.plan(page_table, seq_lens, query_len=query_length)

    def run(self, q: torch.Tensor, k_cache: torch.Tensor, k_scale=None, *, out=None) -> torch.Tensor:
        # k_scale is already inside k_cache (make_k_cache); the argument keeps nv_dev's call shape.
        batch, query_length, num_heads, head_dim = q.shape
        return self.wrapper.run(q.view(batch * query_length, num_heads, head_dim), k_cache, out=out)


def _make_slot(
    case: DecodeCase,
    lengths_cpu: torch.Tensor,
    device: torch.device,
    seed: int,
    num_index_heads: int = 1,
    query_length: int = 8,
):
    max_pages = (int(lengths_cpu.max()) + PAGE_SIZE - 1) // PAGE_SIZE
    physical_pages = case.batch * max_pages
    generator = torch.Generator(device=device).manual_seed(seed)
    q = (
        torch.randn(
            case.batch,
            query_length,
            HEAD_DIM,
            generator=generator,
            device=device,
        )
        * 0.5
    ).to(torch.float8_e4m3fn)
    packed_k = torch.randint(
        0,
        256,
        (physical_pages, PAGE_SIZE, HEAD_DIM // 2),
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    k_scale = (
        torch.rand(
            physical_pages,
            PAGE_SIZE,
            HEAD_DIM // 16,
            generator=generator,
            device=device,
        )
        * 0.5
        + 0.125
    ).to(torch.float8_e4m3fn)
    physical_indices = torch.arange(
        physical_pages, dtype=torch.int32, device=device
    ).reshape(case.batch, max_pages)
    page_table = torch.stack(
        [
            physical_indices[batch_index][
                torch.randperm(max_pages, generator=generator, device=device)
            ]
            for batch_index in range(case.batch)
        ]
    ).contiguous()
    seq_lens = lengths_cpu.to(device=device)
    q = q.unsqueeze(2)
    if num_index_heads > 1:
        additional = (
            torch.randn(
                case.batch,
                query_length,
                num_index_heads - 1,
                HEAD_DIM,
                generator=generator,
                device=device,
            )
            * 0.5
        ).to(q.dtype)
        q = torch.cat((q, additional), dim=2)
    output = torch.empty(
        case.batch * query_length,
        num_index_heads,
        16,
        dtype=torch.int32,
        device=device,
    )
    return q, make_k_cache(packed_k, k_scale), k_scale, page_table, seq_lens, output


def _time_graphs(
    graphs: list[torch.cuda.CUDAGraph],
    *,
    graph_calls: int,
    warmup: int,
    replays: int,
) -> tuple[list[float], float]:
    """Rotate graph banks and retry unstable measurements up to three times."""

    last_cv = float("inf")
    for _ in range(MAX_TIMING_ATTEMPTS):
        check_benchmark_device()
        for _ in range(warmup):
            for graph in graphs:
                graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(replays):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for graph in graphs:
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(
                start.elapsed_time(end) * 1000.0 / (graph_calls * len(graphs))
            )
        check_benchmark_device()
        sample_mean = statistics.fmean(samples)
        last_cv = statistics.pstdev(samples) / sample_mean
        if last_cv <= MAX_CV:
            return samples, last_cv
    raise RuntimeError(
        f"unstable timing after {MAX_TIMING_ATTEMPTS} attempts: "
        f"CV {last_cv:.4f} > {MAX_CV:.4f}"
    )


def _resolve_slots(
    *,
    requested_slots: int,
    graph_calls: int,
    read_bytes_per_call: int,
    l2_bytes: int,
) -> int:
    minimum_slots = (2 * l2_bytes) // read_bytes_per_call + 2
    if requested_slots:
        if requested_slots < minimum_slots:
            raise ValueError(
                f"--slots={requested_slots} is too small; at least {minimum_slots} "
                "disjoint slots are required"
            )
        slots = requested_slots
    elif minimum_slots <= graph_calls:
        slots = next(
            divisor
            for divisor in range(minimum_slots, graph_calls + 1)
            if graph_calls % divisor == 0
        )
    else:
        slots = ((minimum_slots + graph_calls - 1) // graph_calls) * graph_calls
    if slots <= graph_calls and graph_calls % slots:
        raise ValueError("slots must divide graph_calls when slots <= graph_calls")
    if slots > graph_calls and slots % graph_calls:
        raise ValueError(
            "slots must be a multiple of graph_calls when slots > graph_calls"
        )
    return slots


def run_case(
    case: DecodeCase,
    *,
    slots_count: int,
    graph_calls: int,
    warmup: int,
    replays: int,
    num_index_heads: int = 1,
    query_length: int = 8,
    verify: bool = False,
) -> dict[str, object]:
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    lengths_cpu = make_seq_lens(case)
    max_pages = (int(lengths_cpu.max()) + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = torch.div(
        lengths_cpu + PAGE_SIZE - 1, PAGE_SIZE, rounding_mode="floor"
    )
    scored_pages = torch.clamp(total_pages - 1, min=0)
    valid_pages = int(scored_pages.sum())
    query_positions = (
        lengths_cpu.reshape(-1, 1)
        - query_length
        + torch.arange(query_length, dtype=torch.int32).reshape(1, query_length)
    )
    query_local_pages = torch.div(query_positions, PAGE_SIZE, rounding_mode="floor")
    valid_scores = int(query_local_pages.sum())
    read_bytes_per_call = (
        valid_pages * PAGE_BYTES
        + case.batch * query_length * HEAD_DIM
        + valid_pages * 4
        + case.batch * 4
    )
    reuse_distance_bytes = (slots_count - 1) * valid_pages * PAGE_BYTES
    l2_bytes = properties.L2_cache_size
    if reuse_distance_bytes <= 2 * l2_bytes:
        raise RuntimeError(
            "slot rotation does not establish a greater-than-2x-L2 reuse distance: "
            f"reuse={reuse_distance_bytes}, L2={l2_bytes}"
        )

    slots = [
        _make_slot(
            case,
            lengths_cpu,
            device,
            seed=case.seed * 100 + slot,
            num_index_heads=num_index_heads,
            query_length=query_length,
        )
        for slot in range(slots_count)
    ]
    wrappers = [
        BatchDecodeIndexerWithPagedKVCacheWrapper(
            use_cuda_graph=True,
            page_table_buffer=torch.empty_like(slot[3]),
            seq_lens_buffer=torch.empty_like(slot[4]),
        )
        for slot in slots
    ]
    for wrapper, slot in zip(wrappers, slots, strict=True):
        wrapper.plan(
            slot[3], slot[4], num_index_heads=num_index_heads, query_length=query_length
        )
    expected_outputs = []
    for wrapper, slot in zip(wrappers, slots, strict=True):
        q, packed_k, k_scale, _, _, output = slot
        wrapper.run(q, packed_k, k_scale=k_scale, out=output)
        expected_outputs.append(output.clone())
    torch.cuda.synchronize()

    graph_count = max(1, slots_count // graph_calls)
    graphs = []
    for graph_index in range(graph_count):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for call in range(graph_calls):
                slot_index = (
                    graph_index * graph_calls + call
                    if slots_count > graph_calls
                    else call % slots_count
                )
                q, packed_k, k_scale, _, _, output = slots[slot_index]
                wrappers[slot_index].run(q, packed_k, k_scale=k_scale, out=output)
        graphs.append(graph)
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    for expected, slot in zip(expected_outputs, slots, strict=True):
        if not torch.equal(expected, slot[-1]):
            raise RuntimeError("CUDA Graph replay changed the full TopK output")

    per_call_microseconds, cv = _time_graphs(
        graphs,
        graph_calls=graph_calls,
        warmup=warmup,
        replays=replays,
    )
    latency_us = statistics.median(per_call_microseconds)
    sample_mean = statistics.fmean(per_call_microseconds)
    sample_std = statistics.pstdev(per_call_microseconds)
    flops = 2 * num_index_heads * valid_scores * PAGE_SIZE * HEAD_DIM
    output_bytes = num_index_heads * case.batch * query_length * 16 * 4
    logical_bytes = necessary_bytes(
        lengths_cpu, num_index_heads, PAGE_BYTES, query_length=query_length
    )
    if verify:
        # fmha_sm100 port: the checks of test_q8_indexer.py::test_decode_matches_reference
        # (scores vs the FP32 reference under the module's numerics rule, then the TopK
        # contract), on slot 0. The test module covers only eight queries per request.
        if query_length != 8:
            raise ValueError("--verify needs --query-length 8")
        import test_q8_indexer as T

        q, k_cache, scale, table, lengths, output = slots[0]
        pages = k_cache.shape[0]
        packed = k_cache.reshape(pages, -1)[:, : PAGE_SIZE * HEAD_DIM // 2]
        paged_k = object.__new__(T._PagedK)
        paged_k.page_pad = 0
        paged_k.exact_k = T._q8kv4_k_is_exact()
        paged_k.set_nvfp4(packed.reshape(pages, PAGE_SIZE, HEAD_DIM // 2), scale)
        q_rows = q.view(case.batch * query_length, num_index_heads, HEAD_DIM)
        inner = wrappers[0].wrapper
        scores = T._run_decode_scores(inner, q_rows, paged_k.cache)
        T._assert_decode_scores(scores, T._decode_reference(q_rows, paged_k, table), lengths)
        T._assert_topk_contract(
            scores,
            T._decode_valid_pages(lengths, num_index_heads),
            inner.run(q_rows, paged_k.cache),
        )
    return {
        **asdict(case),
        "q_len_per_req": query_length,
        "name": case.name,
        "length_min": int(lengths_cpu.min()),
        "length_max": int(lengths_cpu.max()),
        "length_mean": float(lengths_cpu.to(torch.float64).mean()),
        "valid_pages": valid_pages,
        "valid_scores": valid_scores * num_index_heads,
        "necessary_bytes": logical_bytes,
        "num_index_heads": num_index_heads,
        "max_pages": max_pages,
        "slots": slots_count,
        "graph_count": graph_count,
        "graph_calls": graph_calls,
        "replays": replays,
        "latency_us_samples": per_call_microseconds,
        "latency_us": latency_us,
        "latency_us_mean": sample_mean,
        "latency_us_std": sample_std,
        "cv": cv,
        "tflops": flops / latency_us / 1.0e6,
        "read_bytes_per_call": read_bytes_per_call,
        "output_bytes_per_call": output_bytes,
        "effective_gb_s": logical_bytes / latency_us / 1.0e3,
        "effective_tb_s": logical_bytes / latency_us / 1.0e6,
        "l2_bytes": l2_bytes,
        "reuse_distance_bytes": reuse_distance_bytes,
        "reuse_distance_over_l2": reuse_distance_bytes / l2_bytes,
        "device": properties.name,
        "e2e_scope": "fmha_sm100 BatchDecodeIndexerQ8KV4Wrapper.run",
        "correctness": "full reference passed" if verify else "TopK graph replay only",
        **roofline_metrics(
            device_name=properties.name,
            useful_flops=flops,
            logical_bytes=logical_bytes,
            latency_us=latency_us,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=("smoke", "full", "low-latency"), default="smoke"
    )
    parser.add_argument(
        "--batch",
        type=int,
        choices=tuple(sorted(set(BATCH_SIZES + LOW_LATENCY_BATCH_SIZES))),
        action="append",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        choices=tuple(sorted(set(SEQ_LENGTHS + LOW_LATENCY_SEQ_LENGTHS))),
        action="append",
    )
    parser.add_argument("--slots", type=int, default=0)
    parser.add_argument("--graph-calls", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--num-index-heads", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--query-length", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.slots < 0 or args.graph_calls < 1:
        parser.error("--slots must be non-negative and --graph-calls must be positive")
    if args.warmup < 1 or args.replays < 2:
        parser.error("--warmup must be positive and --replays must be at least two")
    check_benchmark_device()
    cases = benchmark_cases(args.suite)
    if args.batch:
        cases = tuple(case for case in cases if case.batch_size in set(args.batch))
    if args.seq_len:
        cases = tuple(
            case for case in cases if case.nominal_seq_len in set(args.seq_len)
        )
    if not cases:
        parser.error("benchmark selection is empty")
    results = []
    for case in cases:
        lengths_cpu = make_seq_lens(case)
        total_pages = torch.div(
            lengths_cpu + PAGE_SIZE - 1, PAGE_SIZE, rounding_mode="floor"
        )
        valid_pages = int(torch.clamp(total_pages - 1, min=0).sum())
        try:
            slots_count = _resolve_slots(
                requested_slots=args.slots,
                graph_calls=args.graph_calls,
                read_bytes_per_call=valid_pages * PAGE_BYTES,
                l2_bytes=torch.cuda.get_device_properties("cuda").L2_cache_size,
            )
        except ValueError as error:
            parser.error(str(error))
        result = run_case(
            case,
            slots_count=slots_count,
            graph_calls=args.graph_calls,
            warmup=args.warmup,
            replays=args.replays,
            num_index_heads=args.num_index_heads,
            query_length=args.query_length,
            verify=args.verify,
        )
        result["suite"] = args.suite
        results.append(result)
        print(
            json.dumps(result, sort_keys=True),
            flush=True,
        )
    formal = args.suite == "full" and not args.batch and not args.seq_len
    aggregate = None
    if formal:
        total_weight = sum(case.weight for case in cases)
        weighted_latency = (
            sum(
                case.weight * float(result["latency_us"])
                for case, result in zip(cases, results, strict=True)
            )
            / total_weight
        )
        aggregate = {
            "total_weight": total_weight,
            "weighted_mean_e2e_latency_us": weighted_latency,
            "weighted_effective_mbu": sum(
                case.weight * row["necessary_bytes"]
                for case, row in zip(cases, results, strict=True)
            )
            / sum(
                case.weight * row["latency_us"]
                for case, row in zip(cases, results, strict=True)
            )
            / 1e6
            / results[0]["peak_hbm_tb_s"],
        }
    payload = {
        "schema_version": 2,
        "num_index_heads": args.num_index_heads,
        "precision": "q8kv4",
        "device_uuid": str(torch.cuda.get_device_properties("cuda").uuid),
        "device": torch.cuda.get_device_properties("cuda").name,
        "dsl_version": importlib.metadata.version("nvidia-cutlass-dsl"),
        "protocol": {
            "suite": args.suite,
            "formal": formal
            or (
                args.suite == "low-latency"
                and not args.batch
                and not args.seq_len
                and args.warmup == 5
                and args.replays == 20
                and args.graph_calls == 120
            ),
            "e2e_scope": "public wrapper.run (proxy GEMM + TopK)",
            "cuda_graph_calls": args.graph_calls,
            "query_length": args.query_length,
            "warmup_replays": args.warmup,
            "timed_replays": args.replays,
            "maximum_cv": MAX_CV,
            "cold_cache_reuse_distance_over_l2": 2.0,
        },
        "aggregate": aggregate,
        "results": results,
    }
    acceptance_failed = False
    if args.baseline is not None:
        baseline_payload = json.loads(args.baseline.read_text())
        comparator = compare_low_latency if args.suite == "low-latency" else compare_mbu
        payload["acceptance"] = comparator(payload, baseline_payload)
        acceptance_failed = not payload["acceptance"]["passed"]
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({key: value for key, value in payload.items() if key != "results"})
    )
    if acceptance_failed:
        raise SystemExit("E2E performance acceptance failed")


if __name__ == "__main__":
    main()
