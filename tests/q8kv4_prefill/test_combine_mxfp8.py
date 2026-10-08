# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""The combines' MXFP8 output is, bit for bit, FlashInfer's MXFP8 quantization of their BF16 output
(E4M3 data, 128x4-swizzled UE8M0 scales with the padding rows zeroed): the SM100 combine and the
Blackwell prefill port's combine, which run_prefill keeps with and without ``out_mxfp8``."""

import pytest
import torch

from fmha_sm100.prefill_q8kv4.interface import _sparse_stack, run_prefill

from .cases import SMOKE_CASES, make_inputs
from .runners import plan_wrapper

flashinfer = pytest.importorskip("flashinfer")

HEAD_DIM, SPLITS = 128, 16


def _combine(impl: str):
    combine = _sparse_stack()[1]
    if impl == "blackwell":
        from src.blackwell_prefill.combine import combine
    return combine


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _values(rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    """Per-32-block magnitudes over 2^-20..2^20 with NaN/Inf/0/denormals sprinkled in."""
    x = torch.randn(rows, cols, generator=gen, device="cuda")
    e = torch.randint(-20, 21, (rows, cols // 32, 1), generator=gen, device="cuda")
    x = (x.view(rows, -1, 32) * torch.pow(2.0, e.float())).view(rows, cols)
    special = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e-40, 448.0, 464.0], device="cuda"
    )
    mask = torch.rand(rows, cols, generator=gen, device="cuda") < 1e-3
    idx = torch.randint(0, len(special), (rows, cols), generator=gen, device="cuda")
    return torch.where(mask, special[idx], x).to(torch.bfloat16)


@pytest.mark.parametrize("num_tokens", [1, 129, 1000])
@pytest.mark.parametrize("store_bf16", [False, True])
@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("impl", ["sm100", "blackwell"])
def test_mxfp8_output_matches_bf16_then_quantize(device, num_tokens, store_bf16, heads, impl):
    HEADS = heads
    combine = _combine(impl)
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    first = max(1, num_tokens // 2)
    lens = [first, num_tokens - first] if num_tokens > 1 else [1]
    cu_seqlens = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32,
                              device=device)
    o_partial = _values(SPLITS * num_tokens * HEADS, HEAD_DIM, gen).view(
        SPLITS, num_tokens, HEADS, HEAD_DIM)
    lse_partial = torch.randn(SPLITS, num_tokens, HEADS, generator=gen, device=device)
    split_counts = torch.randint(0, SPLITS + 1, (num_tokens, HEADS // 16), generator=gen,
                                 device=device).to(torch.int32)
    kwargs = dict(cu_seqlens=cu_seqlens, split_counts=split_counts, use_pdl=True)

    out_ref = torch.empty((num_tokens, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
    combine(o_partial, lse_partial, out_ref, None, **kwargs)
    q_ref, s_ref = flashinfer.mxfp8_quantize(out_ref.view(num_tokens, HEADS * HEAD_DIM),
                                             is_sf_swizzled_layout=True, backend="cute-dsl")

    # Pre-filled with garbage so every byte must be written.
    padded_rows = (num_tokens + 127) // 128 * 128
    q = torch.full((num_tokens, HEADS, HEAD_DIM), 0x55, dtype=torch.uint8, device=device)
    s = torch.full((padded_rows * HEADS * HEAD_DIM // 32,), 0xAB, dtype=torch.uint8, device=device)
    out = torch.full_like(out_ref, 3.0) if store_bf16 else None
    combine(o_partial, lse_partial, out, None, o_mxfp8=(q.view(torch.float8_e4m3fn), s), **kwargs)

    assert torch.equal(q.view(num_tokens, -1), _bits(q_ref))
    assert torch.equal(s, _bits(s_ref))
    if store_bf16:
        assert torch.equal(_bits(out), _bits(out_ref))


@pytest.mark.parametrize("case", SMOKE_CASES[1:], ids=lambda case: case.name)
def test_run_prefill_mxfp8_output_is_opt_in(device, case, monkeypatch):
    """``run_prefill(out_mxfp8=...)`` writes FlashInfer's MXFP8 quantization of the BF16 output it
    computes in the same call; without ``out_mxfp8`` the call is unchanged (and with only
    ``out_mxfp8`` no BF16 output is written)."""
    inputs = make_inputs(case, device)
    state = plan_wrapper(inputs)._plan_state
    total_q, heads = inputs.q.shape[0], inputs.q.shape[1]

    def run(**kwargs):
        return run_prefill(
            inputs.q,
            (inputs.k_codes, inputs.v_codes),
            (inputs.k_scale, inputs.v_scale_kernel),
            kv_indices=state.kv_indices,
            kv_indptr=state.kv_indptr,
            cu_seqlens_q=state.cu_seqlens_q,
            cu_seqlens_k=state.cu_seqlens_k,
            k2q_row_ptr=state.k2q_row_ptr,
            schedule=state.schedule,
            sm_scale=state.sm_scale,
            topk=state.topk,
            seqused_k=state.seqused_k,
            **kwargs,
        )[0]

    def mxfp8_buffers():
        padded_rows = (total_q + 127) // 128 * 128
        q = torch.full((total_q, heads, HEAD_DIM), 0x55, dtype=torch.uint8, device=device)
        s = torch.full((padded_rows * heads * HEAD_DIM // 32,), 0xAB, dtype=torch.uint8,
                       device=device)
        return q.view(torch.float8_e4m3fn), s

    # The opt-in path keeps the Blackwell combine wherever the default path uses it.
    from fmha_sm100.sparse_fmha_adapter import _supports_blackwell_prefill

    _sparse_stack()
    import src.blackwell_prefill.combine as blackwell_combine

    calls = []
    blackwell = blackwell_combine.combine

    def spy(*args, **kwargs):
        calls.append("o_mxfp8" in kwargs)
        return blackwell(*args, **kwargs)

    monkeypatch.setattr(blackwell_combine, "combine", spy)

    default = run().clone()
    both = mxfp8_buffers()
    out = run(out=torch.empty_like(default), out_mxfp8=both)
    q_ref, s_ref = flashinfer.mxfp8_quantize(out.view(total_q, heads * HEAD_DIM),
                                             is_sf_swizzled_layout=True, backend="cute-dsl")
    assert torch.equal(_bits(both[0]).view(total_q, -1), _bits(q_ref))
    assert torch.equal(both[1], _bits(s_ref))
    torch.testing.assert_close(out, default, atol=0, rtol=0)

    only = mxfp8_buffers()
    assert run(out_mxfp8=only) is None
    assert torch.equal(_bits(only[0]), _bits(both[0]))
    assert torch.equal(only[1], both[1])
    if _supports_blackwell_prefill(inputs.q.device, topk=state.topk):
        assert calls == [False, True, True]
