# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""The combine's MXFP8 output is, bit for bit, FlashInfer's MXFP8 quantization of its BF16 output:
E4M3 data and 128x4-swizzled UE8M0 scales with the padding rows zeroed. The buffers start as
garbage, so every byte must be written; the inputs cover 2^-20..2^20 per 32-value block with NaN,
Inf, zeros and denormals mixed in."""

import pytest
import torch

from fmha_sm100.prefill_q8kv4.interface import _sparse_stack, run_prefill

from .cases import SMOKE_CASES, make_inputs
from .runners import plan_wrapper

flashinfer = pytest.importorskip("flashinfer")

HEAD_DIM, SPLITS = 128, 16
PAGE_BYTES = 2 << 20
DATA_GARBAGE, SCALE_GARBAGE = 0x55, 0xAB


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(-1)


def _values(rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(rows, cols, generator=gen, device="cuda")
    e = torch.randint(-20, 21, (rows, cols // 32, 1), generator=gen, device="cuda")
    x = (x.view(rows, -1, 32) * torch.pow(2.0, e.float())).view(rows, cols)
    special = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e-40, 448.0, 464.0], device="cuda"
    )
    mask = torch.rand(rows, cols, generator=gen, device="cuda") < 1e-3
    idx = torch.randint(0, len(special), (rows, cols), generator=gen, device="cuda")
    return torch.where(mask, special[idx], x).to(torch.bfloat16)


def _every_value_under_every_scale() -> torch.Tensor:
    """[blocks, 32] BF16 bits: each magnitude alone in a block, then every value under the largest
    amax of each UE8M0 scale. A NaN is never the amax, so it goes under every scale."""
    patterns = torch.arange(65536)
    magnitudes = patterns & 0x7FFF
    amaxes = torch.arange(0x7F81)
    scales = ((amaxes >> 7) - 8 + ((amaxes & 0x7F) > 96).long()).clamp(min=0)
    scales[0x7F80] = 254
    alone = torch.zeros((65536, 32), dtype=torch.long)
    alone[:, 0] = patterns
    blocks = [alone]
    for scale in scales.unique():
        amax = amaxes[scales == scale].max()
        values = patterns[(magnitudes <= amax) | (magnitudes > 0x7F80)]
        rest = torch.zeros(-(-values.numel() // 31) * 31, dtype=torch.long)
        rest[: values.numel()] = values
        blocks.append(torch.cat((torch.full((rest.numel() // 31, 1), amax), rest.view(-1, 31)), 1))
    blocks = torch.cat(blocks)
    return blocks.gather(1, torch.rand(blocks.shape).argsort(1))


def _partials(total_q: int, heads: int, seed: int, device: torch.device) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return dict(
        o_partial_fake=_values(SPLITS * total_q * heads, HEAD_DIM, gen).view(
            SPLITS, total_q, heads, HEAD_DIM
        ),
        lse_partial=torch.randn(SPLITS, total_q, heads, generator=gen, device=device),
        split_counts=torch.randint(
            0, SPLITS + 1, (total_q, heads // 16), generator=gen, device=device
        ).to(torch.int32),
    )


def _scale_index(rows: int, heads: int, device: torch.device) -> torch.Tensor:
    """Offsets of the ``[rows, heads, 4]`` UE8M0 scales in the 128x4 layout."""
    row = torch.arange(rows, device=device).view(-1, 1, 1)
    head = torch.arange(heads, device=device).view(1, -1, 1)
    block = torch.arange(4, device=device).view(1, 1, -1)
    return (
        (row // 128) * 128 * heads * 4
        + head * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + block
    )


def _garbage_buffers(rows: int, activation_rows: int, heads: int, device: torch.device):
    padded_rows = (activation_rows + 127) // 128 * 128
    data = torch.full((rows, heads, HEAD_DIM), DATA_GARBAGE, dtype=torch.uint8, device=device)
    scale = torch.full(
        (padded_rows * heads * HEAD_DIM // 32,), SCALE_GARBAGE, dtype=torch.uint8, device=device
    )
    return data.view(torch.float8_e4m3fn), scale


def _quantized(out: torch.Tensor, first_row: int = 0):
    """FlashInfer's MXFP8 of ``first_row`` zero rows followed by ``out``, with the scale bytes of
    the leading rows put back to the garbage only an earlier kernel may overwrite."""
    rows, heads, _ = out.shape
    head_rows = torch.zeros((first_row, heads * HEAD_DIM), dtype=out.dtype, device=out.device)
    q, s = flashinfer.mxfp8_quantize(
        torch.cat((head_rows, out.reshape(rows, heads * HEAD_DIM))),
        is_sf_swizzled_layout=True,
        backend="cute-dsl",
    )
    s = _bits(s).clone()
    s[_scale_index(first_row, heads, out.device).reshape(-1)] = SCALE_GARBAGE
    return _bits(q)[first_row * heads * HEAD_DIM :], s


@pytest.mark.parametrize("num_tokens", [1, 129, 1000])
@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("store_bf16", [False, True], ids=["mxfp8_only", "with_bf16"])
def test_mxfp8_output_matches_bf16_then_quantize(device, num_tokens, heads, store_bf16):
    combine = _sparse_stack()[1]
    partials = _partials(num_tokens, heads, num_tokens, device)
    o_partial, lse_partial = partials.pop("o_partial_fake"), partials.pop("lse_partial")
    kwargs = dict(use_pdl=True, **partials)

    out_ref = torch.empty((num_tokens, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
    combine(o_partial, lse_partial, out_ref, None, **kwargs)
    q_ref, s_ref = _quantized(out_ref)

    data, scale = _garbage_buffers(num_tokens, num_tokens, heads, device)
    out = torch.full_like(out_ref, 3.0) if store_bf16 else None
    combine(o_partial, lse_partial, out, None, o_mxfp8=(data, scale), **kwargs)

    assert torch.equal(_bits(data), q_ref)
    assert torch.equal(scale, s_ref)
    if store_bf16:
        assert torch.equal(_bits(out), _bits(out_ref))


def test_mxfp8_output_for_every_bf16_value_and_scale(device):
    """The quantization over its whole domain, in the block order of the kernel's lanes. The
    partials are in store order, so tags first find where each value of a single split lands."""
    combine = _sparse_stack()[1]
    blocks = _every_value_under_every_scale()
    heads = 16
    rows = -(-blocks.shape[0] // (heads * 4))
    flat = torch.zeros(rows * heads * HEAD_DIM, dtype=torch.long)
    flat[: blocks.numel()] = blocks.reshape(-1)
    expected = flat.to(device).to(torch.uint16).view(torch.bfloat16)
    o_partial = torch.zeros((SPLITS, rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
    lse_partial = torch.zeros((SPLITS, rows, heads), device=device)
    counts = torch.ones((rows, heads // 16), dtype=torch.int32, device=device)
    kwargs = dict(use_pdl=True, split_counts=counts)
    out = torch.empty((rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)

    position = torch.arange(expected.numel(), device=device)
    source = torch.zeros_like(position)
    for shift in (0, 8, 16):
        o_partial[0].view(-1)[:] = ((position >> shift) & 255).to(torch.bfloat16)
        combine(o_partial, lse_partial, out, None, **kwargs)
        source |= out.to(torch.float32).view(-1).long() << shift
    o_partial[0].view(-1)[source] = expected
    combine(o_partial, lse_partial, out, None, **kwargs)
    assert torch.equal(out.view(-1).nan_to_num(0.0), expected.nan_to_num(0.0))
    q_ref, s_ref = _quantized(out)

    data, scale = _garbage_buffers(rows, rows, heads, device)
    combine(o_partial, lse_partial, None, None, o_mxfp8=(data, scale), **kwargs)

    assert torch.equal(_bits(data), q_ref)
    assert torch.equal(scale, s_ref)


@pytest.mark.parametrize(
    "first_row, num_tokens",
    [(3, 129), (125, 3), (128, 5), (127, 1), (1000, 1000)],
    ids=lambda v: str(v),
)
@pytest.mark.parametrize("heads", [16, 64])
def test_mxfp8_first_row_leaves_the_earlier_rows_alone(device, first_row, num_tokens, heads):
    """Decode tokens come first in a mixed batch: the combine fills the activation's last rows and
    its padding, and every scale byte of the rows before ``first_row`` keeps its value."""
    combine = _sparse_stack()[1]
    partials = _partials(num_tokens, heads, first_row + num_tokens, device)
    o_partial, lse_partial = partials.pop("o_partial_fake"), partials.pop("lse_partial")
    kwargs = dict(use_pdl=True, **partials)

    out_ref = torch.empty((num_tokens, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
    combine(o_partial, lse_partial, out_ref, None, **kwargs)
    q_ref, s_ref = _quantized(out_ref, first_row)

    data, scale = _garbage_buffers(num_tokens, first_row + num_tokens, heads, device)
    combine(
        o_partial,
        lse_partial,
        None,
        None,
        o_mxfp8=(data, scale),
        o_mxfp8_first_row=first_row,
        **kwargs,
    )

    assert torch.equal(_bits(data), q_ref)
    assert torch.equal(scale, s_ref)


@pytest.mark.parametrize("total_q, heads", [(333, 64), (171, 16)])
def test_mxfp8_output_with_tile_claiming(device, total_q, heads):
    """With per-domain tile claiming a CTA's index says nothing about its tile: the CTA of the last
    flat row, wherever it runs, still owns the padding rows."""
    from src.sm100.fwd.combine import LocalityHints

    combine = _sparse_stack()[1]
    domains = 2
    partials = _partials(total_q, heads, total_q, device)
    contiguous, lse_partial = partials.pop("o_partial_fake"), partials.pop("lse_partial")
    stripe = domains * PAGE_BYTES
    slab_bytes = -(-total_q * heads * HEAD_DIM * 2 // stripe) * stripe
    o_partial = (
        torch.empty(SPLITS * slab_bytes // 2, dtype=torch.bfloat16, device=device)
        .as_strided(
            (SPLITS, total_q, heads, HEAD_DIM),
            (slab_bytes // 2, heads * HEAD_DIM, HEAD_DIM, 1),
        )
    )
    o_partial.copy_(contiguous)
    locality = LocalityHints(
        sm_domains=torch.arange(4096, dtype=torch.int32, device=device) % domains,
        tile_counters=torch.zeros(domains, dtype=torch.int32, device=device),
        page_bytes=PAGE_BYTES,
    )
    kwargs = dict(use_pdl=True, locality=locality, **partials)

    out_ref = torch.empty((total_q, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
    combine(o_partial, lse_partial, out_ref, None, **kwargs)
    q_ref, s_ref = _quantized(out_ref)

    locality.tile_counters.zero_()
    data, scale = _garbage_buffers(total_q, total_q, heads, device)
    combine(o_partial, lse_partial, None, None, o_mxfp8=(data, scale), **kwargs)

    assert torch.equal(_bits(data), q_ref)
    assert torch.equal(scale, s_ref)


@pytest.mark.parametrize("case", SMOKE_CASES, ids=lambda case: case.name)
def test_run_prefill_mxfp8_output(device, case):
    """``out_mxfp8`` is the MXFP8 of the BF16 output of the same forward, and asking for it does
    not change that output; with only ``out_mxfp8`` no BF16 output is written."""
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

    default = run().clone()
    q_ref, s_ref = _quantized(default)

    both = _garbage_buffers(total_q, total_q, heads, device)
    out = run(out=torch.empty_like(default), out_mxfp8=both)
    assert torch.equal(_bits(out), _bits(default))
    assert torch.equal(_bits(both[0]), q_ref)
    assert torch.equal(both[1], s_ref)

    only = _garbage_buffers(total_q, total_q, heads, device)
    assert run(out_mxfp8=only) is None
    assert torch.equal(_bits(only[0]), q_ref)
    assert torch.equal(only[1], s_ref)


def test_mxfp8_rejects_bad_buffers(device):
    combine = _sparse_stack()[1]
    total_q, heads = 3, 16
    o_partial = torch.empty((SPLITS, total_q, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
    lse_partial = torch.empty((SPLITS, total_q, heads), dtype=torch.float32, device=device)
    data, scale = _garbage_buffers(total_q, total_q, heads, device)

    with pytest.raises(ValueError, match="o_out or o_mxfp8"):
        combine(o_partial, lse_partial, None, None)
    with pytest.raises(ValueError, match="scale must be"):
        combine(o_partial, lse_partial, None, None, o_mxfp8=(data, scale[:-1]))
    with pytest.raises(ValueError, match="scale must be"):
        combine(o_partial, lse_partial, None, None, o_mxfp8=(data, scale), o_mxfp8_first_row=128)
    with pytest.raises(ValueError, match="first_row"):
        combine(o_partial, lse_partial, None, None, o_mxfp8=(data, scale), o_mxfp8_first_row=-1)
    with pytest.raises(ValueError, match="data must be"):
        combine(o_partial, lse_partial, None, None, o_mxfp8=(data[:-1], scale))
    with pytest.raises(TypeError, match="bfloat16 o_out"):
        out = torch.empty((total_q, heads, HEAD_DIM), dtype=torch.float16, device=device)
        combine(o_partial, lse_partial, out, None, o_mxfp8=(data, scale))
