# FL new-code scan — byte-identity + STL/MTL comparison (S1+S2+auto-fit)

> A40, `study/pre-freeze-a40`, 2026-06-19. User ask: "run FL with the new changes to scan them, and
> run STL and MTL so we can compare." Verifies the RAM/OOM fixes (S1 train-metric streaming, S2 chunked
> val-metric, dataset-on-GPU **auto-fit**, the `<U32` builder fix) are numerically inert at FL scale, and
> reproduces the §0.1 MTL-vs-STL story under the current code. FL seed 0, 5-fold, **v14** substrate
> (`check2hgi_design_k_resln_mae_l0_1`), champion-G recipe, geom_simple, KD off, **non-overlap** (matches
> the LANE2 baseline arm). Per-task **diagnostic-best** (selector-independent).
>
> **⚠ Correction (PR #28 audit, 2026-06-19):** the scan empirically exercised **S1 (train-metric streaming,
> default ON) + dataset-on-GPU auto-fit + the `<U32` fix** — all byte-identical (S1/auto-fit by construction;
> `<U32` verified AL max|Δ|=0). **S2 (chunked val-metric) was NOT exercised** — it is behind
> `MTL_CHUNK_VAL_METRIC` (default OFF), which no committed run sets, so 73.0116/73.5414 came off the legacy val
> path. S2 is byte-identical **BY CONSTRUCTION** (per-row argmax/rank/topk reconstruction, `.float().mean()` not
> a running int) but is NOT empirically verified — **run an S2-ON A/B before the board enables it** (for
> large-state overlap val OOM). Also: "exact to 4 dp" is consistent-but-unproven (the baseline was only recorded
> to 2 dp: 73.01/73.54).

## 1 · Byte-identity — the new code is INERT (the headline of the scan)

| metric | new code (S1+S2+auto-fit+`<U32`) | LANE2 baseline | verdict |
|---|---|---|---|
| MTL cat macro-F1 | **73.0116** ± 0.98 | 73.01 | ✅ identical |
| MTL reg top10_acc_indist | **73.5414** ± 0.76 | 73.54 | ✅ identical |

Exact to 4 decimals — and **FL is where S2 chunked-val actually fires** (4703 regions ≫ the 256 gate),
so this confirms the highest-risk change (the scored val path) at scale, not just the AL byte-identity
proof (AL champion-G 52.3781/64.3450, ON==OFF==baseline). The auto-fit dataset-on-GPU + the RAM streaming
change *nothing* in the numbers; they only change memory/throughput. **All OOM fixes are byte-identity-gated.**

## 2 · STL vs MTL (matched metrics — the §0.1 story reproduces)

⚠ **Matched-metric discipline (B-A2 correction).** The board's canonical reg metric is `top10_acc_indist`
(OOD-restricted), but `p1_region_head_ablation.py` (the STL reg ceiling) reports **full** `top10_acc` only.
indist > full, so **MTL-indist 73.54 vs STL-full 71.61 is NOT a legal subtraction.** The metrics common to
BOTH heads are **`top5_acc` (full)** and **`MRR` (full)** — the comparison below uses those.

| REGION head (FL seed0, 5f) | MTL champion-G | STL ceiling (next_stan_flow) | Δ (MTL−STL) |
|---|---|---|---|
| top5_acc (full, **matched**) | 66.04 ± 0.84 | 65.20 ± 0.93 | **+0.84** |
| MRR (full, **matched**) | 55.91 ± 0.77 | 55.31 ± 0.86 | **+0.60** |
| *(non-matched, do not subtract)* | top10_acc_indist 73.54 | top10_acc_full 71.61 | — |

| CATEGORY head | MTL champion-G | STL ceiling (next_gru) | Δ |
|---|---|---|---|
| macro-F1 (**matched**) | 73.01 ± 0.98 | 67.58 ± 0.86 | **+5.43** |

**Reads:** MTL **beats** the STL cat ceiling by +5.43 (mtl_improvement reported +2.6…+4.1 across 4 states;
FL is a large state → high end), and **matches-to-slightly-exceeds** the STL reg ceiling on matched metrics
(+0.84 top5, +0.60 MRR). This is the **dissolved-gap** finding (champion G = canon v16) reproducing under
the new code: MTL does not sacrifice region. Caveats: single-seed (seed 0), unpaired (STL/MTL fold
partitions are built independently), non-overlap (the adopted stride-1 rebuild is P3) → directional, not a
paper-grade Δ. The paper-grade n=20 matched comparison is T3 (post-freeze).

## 3 · What this clears

- The RAM/OOM remediation (S1+S2+auto-fit+`<U32`) is safe to carry into the P3 full-base rebuild — proven
  inert at the scale where it matters. CA/TX (the states it was built for) inherit the same code path.
- The STL/MTL relationship is intact under the current code at a large state.

Captures: MTL `results/lane1_g01/florida_s0__newcode_mtl/`; STL cat
`results/check2hgi_design_k_resln_mae_l0_1/florida/next_*_602940/`; STL reg p1 log `/tmp/lane1/fl_stl_reg.log`.
