# M3 decode `sparse_topk_select` slow-kernel investigation

## TL;DR

In a steady-state decode nsys trace of MiniMax-M3, the kernel
`flashinfer::sparse_topk::IndexerTopKWithSortKernel<16>` (the decode top-k block
selector) runs at **~34 µs typical and up to ~82 µs** for grid `<~120,1,1>`,
whereas in isolation the exact same kernel/shape runs in **~6 µs**.

**It is *not* a memory-layout / coalescing problem.** The input read is fully
coalesced and every layout ablation is flat. The cost is **data-dependent and
algorithmic**: real M3 index logits are near-equal near the top-16 boundary, so
they collide into a single histogram bin, and the kernel's
**O(finalCount²) rank-selection** over that bin dominates.

- Well-separated scores → ~6–16 µs.
- Real / tied scores (`finalCount ≈ hundreds`) → 34 µs.
- Worst case measured (`finalCount ≈ 1600`) → **111 µs**, scaling quadratically.

---

## Where the kernel lives

| Component | Path |
|---|---|
| Kernel | `python/fmha_sm100/csrc/include/sparse_topk_select.cuh` — `IndexerTopKWithSortKernel` |
| Python API | `python/fmha_sm100/api.py` — `sparse_topk_select` |
| External vLLM caller (decode) | `vllm/models/minimax_m3/nvidia/indexer_msa.py` — `MiniMaxM3IndexerMSAImpl.forward` |

### Decode call (indexer_msa.py)

```python
sparse_topk_select(
    unified_scores[:nd],              # THK, [nd, H, max_k_tiles] contiguous fp32
    self.topk_blocks,                 # 16
    num_valid_pages=nvp[:nd],         # per-token page count (int32 tensor)
    force_begin_blocks=self.init_blocks,
    force_end_blocks=self.local_blocks,
    output=decode_topk_output,        # [nd,H,16] strided VIEW of [H,nd,16] (head-major)
    max_score_layout="THK",
    block_table=decode_block_table,   # 3D direct-gather
)
```

- `unified_scores` is `torch.full((num_tokens, H, max_k_tiles), -inf)`; the decode
  score kernel fills only the causally-valid `[:, :, :num_valid_pages]` region,
  so each row is `~num_valid_pages` real scores + a long `-inf` tail.
- `max_k_tiles = round_up(cdiv(max_seq_len, 128), 128)`. With
  `max_seq_len == max_model_len == 1_048_576`, this is **8192**. So each row is
  ~782 valid scores (≈100k ctx) + ~7410 `-inf`.

### Kernel algorithm (per CTA = one `(token, head)` row)

1. 10-bit fp16 histogram (1024 bins) → `cub::BlockScan` → find threshold bin
   whose cumulative count crosses `topk=16`.
2. Elements below the threshold bin are emitted directly; elements *in* the
   threshold bin are staged into `smemFinal.items` **iff** the bin fits in
   `kNumFinalItems = 2048`.
3. If the bin overflows 2048, refine with fp32 histogram stages 1/2/3.
4. **Otherwise (the common tie case): rank-select the staged `finalCount`
   candidates with an all-pairs O(finalCount²) loop**:

```cpp
for (int i = threadIdx.x; i < finalCount; i += kNumThreadsPerBlock) {
  int outIndex = 0;
  auto logit = smemFinal.items.logits[i];
  for (int j = 0; j < finalCount; j++) {          // <-- O(finalCount^2)
    auto otherLogit = smemFinal.items.logits[j];
    if (logit < otherLogit || (logit == otherLogit && i < j)) outIndex++;
  }
  if (outIndex + baseIdx < topK) smemOutput[outIndex + baseIdx] = ...;
}
```

`finalCount` is the number of scores sharing the top-k boundary histogram bin.
When many index logits are near-equal (real M3 case), `finalCount` is large and
this loop is the whole runtime.

---

## Evidence

### 1. Real trace (steady-state decode, `traces/m3_tp4_decode_*.nsys-rep`)

- `IndexerTopKWithSortKernel<16>`, grid `<104,1,1>` (≈120), block `512`, 37 reg/thd, 16 KB smem.
- Duration: **~34 µs typical**, min ~24 µs, **max ~82 µs**, stddev ~4.7 µs → *data-dependent*.
- The kernel **runs alone** on the GPU (verified: only 1 kernel overlaps its
  window) → *not* SM/memory contention. The preceding kernel is
  `IndexDecodeScoreKernel` (~48 µs), ending 384 ns before it.

### 2. Isolated kernel at the exact real shape (nsys)

`N=8192`, grid 120, THK, per-token `num_valid_pages`, force blocks, 3D
block_table, strided output, `-inf` padded → **avg 6365 ns (6.1–6.5 µs)**.
→ the kernel is intrinsically ~6 µs; the 34 µs is *not* baked into the shape.

### 3. Layout / config ablations — ALL FLAT (~6–16 µs), i.e. layout is not the cause

| Factor flipped | wall µs |
|---|---|
| exact vLLM (THK, strided out, 3D bt, nvp-tensor, force, `-inf` pad) | 14.7 |
| contiguous output (vs head-major strided view) | 14.0 |
| no `block_table` gather | 12.1 |
| scalar `num_valid_pages` (vs per-token tensor) | 12.1 |
| `force_begin/end = 0` | 14.1 |
| clean random (no `-inf` padding) | 14.3 |
| `max_k_tiles` 256 → 2048 | 14–15 (flat) |
| `-inf` fraction (nvp 16 → 896 @ MK=896) | 13–15 (flat) |

### 4. Data-dependence — the actual cause (`sparse_topk_repro.py`)

Same shape and layout; only the score **distribution** changes. GB300:

| valid (nvp) | spread scores (µs) | all-equal scores (µs) |
|---|---|---|
| 256 | 24 | 21 |
| 512 | 20 | 19 |
| **782** | 19 | **35**  ← matches the trace |
| **1600** | 18 | **111** ← quadratic blow-up |
| 3200 | 17 | 19  ← bin > `kNumFinalItems(2048)` → cheap direct-fill |

`35/111 ≈ (782/1600)²` confirms **O(finalCount²)**. The drop at 3200 confirms the
`kNumFinalItems` fall-through boundary. Well-separated scores never build a large
threshold bin, so they stay fast.

---

## How to reproduce

```bash
# GPU with the fmha_sm100 sparse_topk module available (SM100/Blackwell).
cd /path/to/MSA
.venv/bin/python sparse_topk_repro.py
```

Expect the `all-equal` column to spike at `nvp=782` (~35 µs) and `nvp=1600`
(~111 µs) while `spread` stays ~17–24 µs — reproducing the trace's 34 µs from a
tied-score distribution alone.

To confirm against the real workload (gold standard, optional): dump the actual
`unified_scores[:nd]` tensor from one live decode step and replay it through
`sparse_topk_select`; `finalCount` (threshold-bin size) will be large.

---

## Proposed fixes

Both target the O(finalCount²) final-candidate rank-selection.

### Option A — O(topK·finalCount) selection (exact output) — recommended

Only `topK - baseIdx ≤ 16` slots remain, so a full all-pairs rank is unnecessary.
Replace the loop with ≤16 rounds of block-wide arg-max (value, tie-broken by
index) over `smemFinal.items`, marking each selected slot consumed.

- Cost: `O(topK · finalCount)` instead of `O(finalCount²)` — ~25× less work at
  `finalCount=1600` (16·1600 vs 1600²).
- Produces **bit-identical output** (same top-16 by score, tie-break by index).
- ~25–30 lines, self-contained in the insertion-sort branch.

### Option B — lower `kIndexerNumFinalItems` (one constant)

`sparse_topk_select.cuh:809`: `kIndexerNumFinalItems = 2048 → ~512`.

- Large tie-bins (>512) now fall through to the cheap fp32 histogram refinement /
  stage-3 direct-fill instead of the O(n²) sort → caps insertion-sort work at
  `O(512²)`.
- Bonus: shrinks `smemFinal.items` (16 KB → 4 KB) → better occupancy.
- Caveat: changes the tie-break *among exactly-equal scores* (the direct-fill
  path selects by atomic order, not by lowest index). Real logits differ in low
  bits, so stages 1/2/3 separate them deterministically; only synthetic
  all-identical inputs would differ.

### Recommendation

Ship **A** (exact, no correctness surface change). Optionally add **B** for the
occupancy win and a lower worst-case ceiling.

---

## Resolution (implemented and measured)

The kernel now uses a hybrid final-candidate selector:

- `finalCount <= 512`: retain the existing all-pairs rank loop, which is faster
  for small boundary bins.
- `finalCount > 512`: pack each candidate into an exact sortable key, take each
  warp's local top-k with `cub::WarpMergeSort`, then merge at most 256 keys in
  warp 0. This preserves the existing score order and staged-position tie rule
  while bounding the large-bin path.

The sparse-topk JIT Ninja edge now also declares the `.cuh` and FFI header as
dependencies. Previously a header edit was copied into the cache but could
silently reuse the stale object file.

GB200 steady-state results from `sparse_topk_repro.py` (exact decode-style
arguments) are:

| valid (nvp) | spread (us) | all-equal before (us) | all-equal after (us) |
|---:|---:|---:|---:|
| 782 | ~16.6 | ~35.7 | ~17.1 |
| 1600 | ~16.7 | ~112.4 | ~17.7 |

Nsight Compute confirms that the quadratic instruction growth is gone. These
are single cold profiled launches, so their durations are higher than the
steady-state event timings above; the instruction counts are the important
comparison.

| case | warp instructions before | warp instructions after | ncu duration before | ncu duration after |
|---|---:|---:|---:|---:|
| equal, nvp=782 | 19.48M | 3.80M | 64.9 us | 34.9 us |
| equal, nvp=1600 | 74.01M | 3.89M | 207.4 us | 35.3 us |

The 782- and 1600-candidate fixed profiles are effectively identical. The
remaining large-bin hotspot is shared-memory traffic inside the warp merge,
not data-dependent quadratic ranking.

### PDL stress results

The final branch is based on `dev` with PDL enabled. A GB200 stress run covered
the 512/513 selector cutoff, the 2048/2049 refinement boundary, four score
distributions, grids from 1 to 1024 rows, THK/HKT layouts, strided output, and a
10,000-launch soak:

| Case | Time |
|---|---:|
| spread, nvp=1600, grid=120 | 12.38 us |
| clustered, nvp=1600, grid=120 | 17.18 us |
| all-equal, nvp=782, grid=120 | 17.25 us |
| all-equal, nvp=1600, grid=120 | 17.42 us |
| all-equal merge range, nvp=513..2048 | 17.17-17.45 us |
| THK contiguous / strided, nvp=1600 | 17.37 / 17.40 us |
| HKT end-to-end, nvp=1600 | 17.17 us |
| 10,000-launch soak | 17.38 us average |

A concurrent four-GB200 follow-up ran 5,000 launches per GPU; all four passed
at 17.36-17.40 us average.

### Validation plan for whichever fix

1. Edit the source `.cuh`, delete the JIT cache
   (`~/.cache/minfer/fmha_sm100/sparse_topk/`), let it recompile.
2. Re-run `sparse_topk_repro.py` → the `all-equal` column should collapse
   toward the `spread` column (target ≤ ~8 µs at nvp≤1600).
3. Diff kernel output vs the current build on spread / clustered / realistic
   inputs — Option A must be bit-identical; Option B identical except possibly
   the exactly-equal synthetic case.
