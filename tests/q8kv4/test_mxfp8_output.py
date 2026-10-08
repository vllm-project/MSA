# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""The decode output as MXFP8 (``out_mxfp8``) is, bit for bit, FlashInfer's MXFP8 quantization of the
BF16 output of the same plan: E4M3 data and 128x4-swizzled UE8M0 scales with the padding rows
zeroed, on every output path (stream-K fold, direct store, separate split-KV reduction)."""

import pytest
import torch

from fmha_sm100.decode_q8kv4 import BatchDecodeWithPagedKVCacheWrapper

from .cases import KV_HEADS, SM_SCALE, DecodeCase, make_inputs

flashinfer = pytest.importorskip("flashinfer")

CASES = (
    DecodeCase("b2_q4", (700, 1_100), q_len=4, seed=31),
    DecodeCase("b5_ragged_q1", (129, 2_048, 40_000, 5_000, 300), seed=32),
    DecodeCase("b40_q4_160rows", (3_000,) * 40, q_len=4, seed=33),
    DecodeCase("b130_q1", (1_000,) * 130, seed=34),
)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(-1)


def _plan(inputs, num_kv_splits):
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(
        inputs.topk_indices,
        inputs.page_table,
        inputs.seq_lens,
        q_len_per_req=inputs.case.q_len,
        num_q_heads=inputs.num_q_heads,
        num_kv_heads=KV_HEADS,
        num_kv_splits=num_kv_splits,
        sm_scale=SM_SCALE,
    )

    def run(**kwargs):
        return wrapper.run(
            inputs.q,
            (inputs.k_codes, inputs.v_codes),
            kv_cache_sf=(inputs.k_scale, inputs.v_scale_kernel),
            **kwargs,
        )

    return run


def _garbage_mxfp8(rows: int, cols: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    padded_rows = (rows + 127) // 128 * 128
    data = torch.full((rows, cols), 0x55, dtype=torch.uint8, device=device)
    scale = torch.full((padded_rows * cols // 32,), 0xAB, dtype=torch.uint8, device=device)
    return data.view(torch.float8_e4m3fn), scale


@pytest.mark.parametrize("num_kv_splits", (None, 1, 4), ids=("auto", "nosplit", "legacy4"))
@pytest.mark.parametrize("gqa", (16, 8), ids=("gqa16", "gqa8"))
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_mxfp8_output_matches_bf16_then_quantize(device, case, gqa, num_kv_splits):
    inputs = make_inputs(case, device, gqa=gqa)
    run = _plan(inputs, num_kv_splits)
    rows, cols = inputs.q.shape[0], inputs.num_q_heads * 128

    out = torch.empty((rows, inputs.num_q_heads, 128), dtype=torch.bfloat16, device=device)
    run(out=out)
    q_ref, s_ref = flashinfer.mxfp8_quantize(
        out.view(rows, cols), is_sf_swizzled_layout=True, backend="cute-dsl"
    )

    data, scale = _garbage_mxfp8(rows, cols, device)
    assert run(out_mxfp8=(data, scale)) is None
    assert torch.equal(_bits(data), _bits(q_ref))
    assert torch.equal(scale, _bits(s_ref))

    # Both outputs at once: the BF16 output is unchanged by the MXFP8 epilogue.
    both_out = torch.full_like(out, 3.0)
    both_data, both_scale = _garbage_mxfp8(rows, cols, device)
    run(out=both_out, out_mxfp8=(both_data, both_scale))
    assert torch.equal(_bits(both_out), _bits(out))
    assert torch.equal(_bits(both_data), _bits(data))
    assert torch.equal(both_scale, scale)


@pytest.mark.parametrize("num_kv_splits", (None, 4), ids=("auto", "legacy4"))
def test_mxfp8_output_cuda_graph_replay(device, num_kv_splits):
    case = CASES[0]
    inputs = make_inputs(case, device)
    run = _plan(inputs, num_kv_splits)
    rows, cols = inputs.q.shape[0], inputs.num_q_heads * 128
    data, scale = _garbage_mxfp8(rows, cols, device)
    run(out_mxfp8=(data, scale))
    expected = (data.clone(), scale.clone())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out_mxfp8=(data, scale))
    data.view(torch.uint8).fill_(0x55)
    scale.fill_(0xAB)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(data), _bits(expected[0]))
    assert torch.equal(scale, expected[1])
