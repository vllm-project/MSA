# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""fmha_sm100_plan / fmha_sm100 route NVFP4 sparse decode to the Q8KV4 kernel and keep kv_mode 3
for everything else."""

import pytest
import torch

from fmha_sm100.api import fmha_sm100, fmha_sm100_plan, nvfp4_head_slot_views
from fmha_sm100.decode_q8kv4 import BatchDecodeWithPagedKVCacheWrapper, plan_decode, run_decode
from fmha_sm100.q8kv4_decode_adapter import PLAN_KEY

from .cases import HEAD_DIM, KV_HEADS, PAGE_SIZE, SM_SCALE, SMOKE_CASES, DecodeCase, flat_page_table, global_scale, make_inputs, pack_head_slot_pages, pack_vllm_pages, unpack_views
from .conftest import run_timed
from .reference import PageDequantizer, assert_close_to_reference, sparse_decode_reference

CASE = SMOKE_CASES[0]


def _ascending(topk: torch.Tensor) -> torch.Tensor:
    """kv_mode 3 needs ascending lists; the Q8KV4 kernel accepts them too."""
    big = torch.iinfo(torch.int32).max
    lists = topk.clone()
    lists[lists < 0] = big
    lists, _ = lists.sort(dim=-1)
    lists[lists == big] = -1
    return lists.contiguous()


class ApiHarness:
    def __init__(self, device):
        self.inputs = make_inputs(CASE, device)
        self.inputs.topk_indices = _ascending(self.inputs.topk_indices)
        self.k_packed, self.v_packed = pack_head_slot_pages(self.inputs)
        self.kv_indices, self.kv_indptr = flat_page_table(self.inputs.page_table, CASE.seq_lens)
        self.unit = global_scale(1.0, device)
        self.reference = sparse_decode_reference(
            self.inputs, PageDequantizer(self.inputs.k_codes, self.inputs.k_scale), PageDequantizer(self.inputs.v_codes, self.inputs.v_scale)
        )

    def plan(self, num_q_heads=None, **kwargs):
        inputs = self.inputs
        kv = inputs.seq_lens.cpu()
        qo = torch.full_like(kv, CASE.q_len)
        return fmha_sm100_plan(qo, kv, inputs.num_q_heads if num_q_heads is None else num_q_heads,
                               num_kv_heads=KV_HEADS, causal=True, qo_offset=kv - qo, page_size=PAGE_SIZE,
                               kv_block_num=CASE.topk, **kwargs)

    def run(self, plan, *, q=None, topk=None, out=None, kv=None, label="api", **kwargs):
        inputs = self.inputs
        k, v = (self.k_packed, self.v_packed) if kv is None else kv
        def call():
            o, _ = fmha_sm100(inputs.q if q is None else q, k, v, plan,
                              kv_indices=self.kv_indices, kv_block_indexes=inputs.topk_indices if topk is None else topk,
                              out=out, sm_scale=SM_SCALE, k_scale=self.unit, v_scale=self.unit, **kwargs)
            return o
        return run_timed(label, call).clone()

    def views(self):
        """The data and E4M3 scale views of the head-slot pages, for the direct Q8KV4 API."""
        k_data, k_sf, v_data, v_sf = nvfp4_head_slot_views(self.k_packed, self.v_packed)
        return k_data, v_data, k_sf.view(torch.float8_e4m3fn), v_sf.view(torch.float8_e4m3fn)

    def direct(self, shift, topk=None):
        inputs = self.inputs
        plan = plan_decode(batch_size=CASE.batch_size, q_len_per_req=CASE.q_len, topk=CASE.topk, device=inputs.q.device,
                           num_q_heads=inputs.num_q_heads, num_kv_heads=KV_HEADS, block_scale_shift=shift)
        k_data, v_data, k_sf, v_sf = self.views()
        return run_timed(f"run_decode shift{shift}", lambda: run_decode(
            plan, inputs.q, (k_data, v_data), kv_cache_sf=(k_sf, v_sf), seq_lens=inputs.seq_lens,
            kv_indices=self.kv_indices, kv_indptr=self.kv_indptr,
            topk_indices=inputs.topk_indices if topk is None else topk, sm_scale=SM_SCALE,
            kv_global_scale=(self.unit, self.unit))).clone()


@pytest.fixture(scope="module")
def harness(device):
    return ApiHarness(device)


def test_auto_backend_plans_and_runs_the_q8kv4_kernel(harness):
    plan = harness.plan()
    assert plan[3].get(PLAN_KEY) is not None
    out = harness.run(plan, label="api auto")
    assert torch.equal(out, harness.direct(3)), "the API must run the same kernel as run_decode"
    assert_close_to_reference(out, harness.reference, label="api auto vs reference")


def test_kv_mode3_still_serves_the_same_inputs(harness):
    plan = harness.plan(decode_backend="kv_mode3")
    assert plan[3].get(PLAN_KEY) is None
    assert_close_to_reference(harness.run(plan, label="api kv_mode3"), harness.reference, label="kv_mode3 vs reference")


def test_one_plan_serves_layers_with_different_topk_lists(harness):
    plan = harness.plan()
    other = harness.inputs.topk_indices.flip(0).contiguous()  # a different valid list per row
    assert torch.equal(harness.run(plan, topk=other, label="api layer 2"), harness.direct(3, topk=other))
    assert torch.equal(harness.run(plan, label="api layer 1"), harness.direct(3))


def test_graph_replay_matches_eager(harness):
    plan = harness.plan()
    eager = harness.run(plan, label="api eager")
    out = torch.empty_like(eager)
    harness.run(plan, out=out, label="api warm")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fmha_sm100(harness.inputs.q, harness.k_packed, harness.v_packed, plan, kv_indices=harness.kv_indices,
                   kv_block_indexes=harness.inputs.topk_indices, out=out, sm_scale=SM_SCALE,
                   k_scale=harness.unit, v_scale=harness.unit)
    out.fill_(-7)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_unfit_batches_and_calls_fall_back_or_raise(harness):
    inputs = harness.inputs
    rows = inputs.q.shape[0]
    q_gqa4 = inputs.q.reshape(rows, KV_HEADS, inputs.gqa, HEAD_DIM)[:, :, :4].reshape(rows, KV_HEADS * 4, HEAD_DIM).contiguous()
    plan_gqa4 = harness.plan(num_q_heads=KV_HEADS * 4)
    assert plan_gqa4[3].get(PLAN_KEY) is None
    reference_gqa4 = sparse_decode_reference(
        harness.inputs.__class__(**{**vars(inputs), "q": q_gqa4, "gqa": 4}),
        PageDequantizer(inputs.k_codes, inputs.k_scale), PageDequantizer(inputs.v_codes, inputs.v_scale))
    assert_close_to_reference(harness.run(plan_gqa4, q=q_gqa4, label="api gqa4"), reference_gqa4, label="GQA 4 on kv_mode3")
    with pytest.raises(ValueError, match="cannot plan this batch"):
        harness.plan(num_q_heads=KV_HEADS * 4, decode_backend="q8kv4")
    assert harness.plan(kv_dtype="fp8")[3].get(PLAN_KEY) is None
    assert harness.plan(output_maxscore=True)[3].get(PLAN_KEY) is None
    plan = harness.plan()
    kv_mode3 = harness.run(harness.plan(decode_backend="kv_mode3"), label="api kv_mode3 for o_scale")
    scaled = harness.run(plan, o_scale=2.0, label="api o_scale=2")
    assert torch.allclose(scaled.float(), kv_mode3.float() * 2.0, rtol=1e-2, atol=1e-2), "o_scale must land on kv_mode 3"
    with pytest.raises(ValueError, match="cannot serve this call"):
        harness.run(harness.plan(decode_backend="q8kv4"), o_scale=2.0, label="api forced o_scale")


def test_shift_zero_through_the_api_matches_the_wrapper(harness):
    inputs = harness.inputs
    plan = harness.plan(block_scale_shift=0)
    wrapper = BatchDecodeWithPagedKVCacheWrapper()
    wrapper.plan(inputs.topk_indices, harness.kv_indices, inputs.seq_lens, q_len_per_req=CASE.q_len,
                 num_q_heads=inputs.num_q_heads, num_kv_heads=KV_HEADS, kv_indptr=harness.kv_indptr)
    k_data, v_data, k_sf, v_sf = harness.views()
    expected = run_timed("wrapper shift0", lambda: wrapper.run(inputs.q, (k_data, v_data), kv_cache_sf=(k_sf, v_sf)).clone())
    assert torch.equal(harness.run(plan, label="api shift0"), expected)


def test_nvfp4_cache_must_be_per_head_kv_slots(harness):
    """fmha_sm100 reads only per-head K/V slot pages; the side-packed layout (every head's data,
    then every head's scales) is rejected instead of misread."""
    k_side, v_side = pack_vllm_pages(harness.inputs)
    with pytest.raises(ValueError, match="per-head K/V slots"):
        fmha_sm100(harness.inputs.q, k_side, v_side, harness.plan(), kv_indices=harness.kv_indices,
                   kv_block_indexes=harness.inputs.topk_indices, sm_scale=SM_SCALE,
                   k_scale=harness.unit, v_scale=harness.unit)
    cache = harness.k_packed.as_strided((harness.k_packed.shape[0], 2 * KV_HEADS, PAGE_SIZE, 72),
                                        (harness.k_packed.stride(0), PAGE_SIZE * 72, 72, 1))
    with pytest.raises(ValueError, match="per-head K/V slots"):
        nvfp4_head_slot_views(cache[:, :KV_HEADS], cache[:, KV_HEADS:])


def test_caller_built_views_serve_both_decode_kernels(harness):
    """With k_sf / v_sf, fmha_sm100 reads the caller's data and scale views, here of the
    side-packed layout (padded page stride, storage offset), on the Q8KV4 and kv_mode 3 kernels
    alike, bitwise equal to the head-slot cache."""
    k_side, v_side = pack_vllm_pages(harness.inputs, pad_bytes=256, offset_bytes=4096)
    (k_data, k_sf), (v_data, v_sf) = unpack_views(k_side), unpack_views(v_side)
    views = {"kv": (k_data, v_data), "k_sf": k_sf.view(torch.uint8), "v_sf": v_sf.view(torch.uint8)}
    for label, plan in (("q8kv4", harness.plan()), ("kv_mode3", harness.plan(decode_backend="kv_mode3"))):
        slots = harness.run(plan, label=f"api {label} slots")
        assert torch.equal(harness.run(plan, label=f"api {label} views", **views), slots), label


def _requests(inputs, requests):
    """The inputs restricted to some requests, sharing the cache."""
    q_len = inputs.case.q_len
    rows = torch.tensor([r * q_len + t for r in requests for t in range(q_len)], device=inputs.q.device)
    index = torch.tensor(requests, device=inputs.q.device)
    case = DecodeCase(inputs.case.name, tuple(inputs.case.seq_lens[r] for r in requests), q_len=q_len,
                      topk=inputs.case.topk, seed=inputs.case.seed)
    return inputs.__class__(**{**vars(inputs), "case": case, "q": inputs.q[rows], "page_table": inputs.page_table[index],
                               "seq_lens": inputs.seq_lens[index], "topk_indices": inputs.topk_indices[rows]})


def test_forced_q8kv4_plan_is_not_bound_by_the_kv_mode3_row_limit(harness, device):
    """A decode_backend='q8kv4' plan has no kv_mode 3 work list, so a batch past that list's
    65536 query-head rows (here 1024 x 8 x 64) plans and runs; auto still builds the list first."""
    case = DecodeCase("b1024_q8", (1_000,) * 1024, q_len=8, seed=31)
    inputs = make_inputs(case, device)
    kv = inputs.seq_lens.cpu()
    qo = torch.full_like(kv, case.q_len)

    def plan(**kwargs):
        return fmha_sm100_plan(qo, kv, inputs.num_q_heads, num_kv_heads=KV_HEADS, causal=True, qo_offset=kv - qo,
                               page_size=PAGE_SIZE, kv_block_num=case.topk, **kwargs)

    with pytest.raises(AssertionError):
        plan()
    forced = plan(decode_backend="q8kv4")
    assert forced[3].get(PLAN_KEY) is not None and forced[3]["packed_work_info"] is None
    k, v = pack_head_slot_pages(inputs)
    kv_indices, _ = flat_page_table(inputs.page_table, case.seq_lens)
    unit = global_scale(1.0, device)
    out = run_timed("api forced b1024 q8", lambda: fmha_sm100(
        inputs.q, k, v, forced, kv_indices=kv_indices, kv_block_indexes=inputs.topk_indices,
        sm_scale=SM_SCALE, k_scale=unit, v_scale=unit)[0].clone())
    requests = [0, 1, 511, 1022, 1023]
    rows = torch.tensor([r * case.q_len + t for r in requests for t in range(case.q_len)], device=device)
    subset = _requests(inputs, requests)
    reference = sparse_decode_reference(
        subset, PageDequantizer(subset.k_codes, subset.k_scale), PageDequantizer(subset.v_codes, subset.v_scale))
    assert_close_to_reference(out[rows], reference, label="forced b1024 q8 vs reference")

    small = harness.plan(decode_backend="q8kv4")
    with pytest.raises(ValueError, match="no kv_mode 3 work list"):
        fmha_sm100(harness.inputs.q, harness.k_packed.view(torch.float8_e4m3fn),
                   harness.v_packed.view(torch.float8_e4m3fn), small, kv_indices=harness.kv_indices,
                   kv_block_indexes=harness.inputs.topk_indices, sm_scale=SM_SCALE)
