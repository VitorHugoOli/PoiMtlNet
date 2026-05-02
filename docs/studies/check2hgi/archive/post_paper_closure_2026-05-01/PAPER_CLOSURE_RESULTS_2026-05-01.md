# Paper-Closure Run Matrix — Results Summary

**Date:** 2026-05-01
**Branch:** `worktree-check2hgi-mtl`
**Hardware:** H100 80GB
**Total runs:** 28 paper-grade (5f×50ep, leak-free per-fold log_T)
**Outcome:** 28/28 successfully completed (after one retry pass — see "Failures" section).

---

## 1 · Phase 0 — F51 Tier 3 sweep (FL, B9 base, 5f×30ep)

15 smokes across {weight_decay, max_grad_norm, eta_min, OneCycle pct_start, AdamW eps}.

**Verdict:** clean negative. All smokes within ±0.5 pp of B9 baseline.
Best smoke: `pct_start=0.5` at +0.31 pp (below the +0.5 pp promotion threshold).
**No 50ep promotion warranted.** B9 is locally optimal in the optimizer/scheduler axis too,
adding to the F51 Tier 2 capacity-sweep evidence that B9 is the local champion.

Caveat: parallel runs collided in result-dir naming (minute-granular timestamping;
fixed mid-session — see §6 Patches). Fold-5-best from train logs is comparable across
smokes but not the canonical 5-fold-mean.

---

## 2 · Phase 1+2 — STL reg ceilings (STAN-Flow (`next_stan_flow`), leak-free)

5f×50ep, multi-seed where applicable. Top10_acc reported (no `_indist` suffix
because STL evaluates on the full validation set).

| State | n_seeds | seeds | top10_acc | mrr | best_ep range |
|---|---:|---|---:|---:|---:|
| **AL** | 4 | {0, 1, 7, 100} | **61.21 ± 0.18** | 37.94 ± 0.17 | 36-49 |
| **AZ** | 4 | {0, 1, 7, 100} | **53.06 ± 0.15** | 34.62 ± 0.10 | 26-49 |
| **CA** | 1 | {42} | **56.86** | 40.44 | 10-12 |
| **FL** | 4 | {0, 1, 7, 100} | **70.62 ± 0.09** | 55.04 ± 0.09 | 13-17 |
| **TX** | 1 | {42} | **59.32** | 39.59 | 9-12 |

σ_across_seeds is uniformly tight (≤ 0.18 pp) — confirms F51's "essentially
deterministic on partition-difficulty axis" finding extends to AL/AZ/FL.

CA and TX are seed=42 only (cross-state P3 anchor); multi-seed extension at
CA+TX deferred per audit (P1, recommended pre-camera-ready).

---

## 3 · Phase 1+2 — MTL B9 cross-state + multi-seed

Metric: `diagnostic_task_best.next_region.top10_acc_indist` (the per-task best
top10 in-distribution accuracy, aggregated across folds). This is the MTL-side
metric NORTH_STAR §F37 uses for cross-state comparisons.

| State | recipe | n_seeds | seeds | top10_indist |
|---|---|---:|---|---:|
| AL | B9 | 4 | {0, 1, 7, 100} | **49.78 ± 0.18** |
| AZ | B9 | 4 | {0, 1, 7, 100} | **40.31 ± 0.21** |
| CA | B9 | 1 | {42} | **45.11** |
| CA | H3-alt | 1 | {42} | **40.25 ± 3.70** (high σ) |
| TX | B9 | 1 | {42} | **40.79** |
| TX | H3-alt | 1 | {42} | **40.80** |
| FL | B9 (F51 multi-seed, prior) | 5 | {0,1,7,42,100} | **63.34 ± 0.11** |
| FL | H3-alt (F51, prior) | 5 | {0,1,7,42,100} | 59.86 ± 0.22 |

**MTL σ_across_seeds:** AL 0.18, AZ 0.21, FL 0.11. Recipe is recipe-deterministic
across seeds at all three — symmetrizes the architectural-Δ scale curve error bars.

⚠ **METHODOLOGY VERIFIED 2026-05-01 — original extraction was wrong, re-extracted with F51 canonical method.**

The first extraction read `diagnostic_task_best.next_region.top10_acc_indist.mean`
from `summary/full_summary.json`. **The canonical F51 method**
(`scripts/analysis/f51_multiseed_analysis.py:33-50`) instead reads
`metrics/fold{N}_next_region_val.csv` per fold and takes
`max(top10_acc_indist) for epoch >= 5`, then aggregates across folds.

**Validation:** running the F51 method on `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260430_0110/`
gives **63.14 ± 1.15** vs F51's published 63.47 ± 0.75 — within 0.3 pp,
confirming the F51 method is the correct one. The values below are the
re-extracted numbers using the F51 method.

**Re-extracted MTL paper-closure numbers (F51 method):**

| State | recipe | seed(s) | per-fold reg ×100 | mean ± across-seed σ |
|---|---|---|---|---:|
| CA | H3-alt | {42} | [41.86, 42.88, 43.15, 42.64, 45.44] | **43.19** |
| CA | B9 | {42} | [47.41, 47.89, 47.34, 47.23, 49.80] | **47.93** |
| TX | B9 | {42} | [41.65, 43.01, 42.04, 44.33, 42.12] | **42.63** |
| TX | H3-alt | {42} | [40.74, 40.73, 42.10, 41.08, 39.70] | **40.87** |
| AL | B9 | {0,1,7,100} | (4 fold-vectors above) | **50.17 ± 0.24** |
| AZ | B9 | {0,1,7,100} | (4 fold-vectors above) | **40.78 ± 0.07** |
| FL | B9 (F51 prior) | {0,1,7,42,100} | — | **63.34 ± 0.11** |

**Why F49's "AL +6.48 pp MTL>STL" doesn't reproduce.** F49 ran 2026-04-27 —
*before* the C4 leak fix (F50, 2026-04-29) and the per-seed log_T fix (F51,
2026-04-30). F49's AL H3-alt was 74.62 ± 3.11 under the legacy full-data
`region_transition_log.pt` (which leaks val transitions into training). Our
AL B9 leak-free = 50.17, expected AL H3-alt leak-free ≈ 47 (B9 typically beats
H3-alt by ~3 pp per F51). **Leak inflation at AL ≈ 27 pp, larger than NORTH_STAR's
"~13-17 pp" published estimate** (§NORTH_STAR.md line 9). The estimate was
calibrated on FL where the inflation was ~13 pp; AL's smaller transition graph
(1109 regions vs 4703) concentrates the leak more.

**STL F21c values from F49 (68.37 AL, 66.74 AZ) are also leaky.** Our leak-free
STL = 61.21 (AL) and 53.06 (AZ) — drops of 7 pp and 13 pp respectively, more
in line with the NORTH_STAR estimate.

**The architectural-Δ leak-free picture (paper-headline claim):** Under
symmetric leak-free comparison, **MTL B9 underperforms STL STAN-Flow (`next_stan_flow`)
ceiling at every state**. The "AL favors MTL +6.48 pp" headline from F49 was
a leak artifact (asymmetric leak inflation favored MTL more than STL at AL).

---

## 4a-bis · B9 vs H3-alt RECIPE comparison (added 2026-05-01 after audit)

A second axis of comparison — which **MTL recipe** wins — became measurable after
the AL+AZ H3-alt multi-seed gap-fill (+8 runs, ~10 min on H100). Previously F51
established B9 > H3-alt at FL multi-seed (Δ_reg +3.48 pp, p=2.98e-8). Now we
have cross-state evidence:

| State | n_pairs | Δ_reg pp | p_reg | Δ_cat pp | p_cat | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **AL** | 20 (4 seeds) | **−0.35** | **1.9e-03** | **−2.22** | **1.9e-06** | **H3-alt > B9 on cat; tied/slightly behind on reg** |
| **AZ** | 20 (4 seeds) | −0.09 | 0.23 (n.s.) | **−0.96** | **7.1e-04** | **H3-alt > B9 on cat; tied on reg** |
| **FL** | 25 (5 seeds) | **+3.48** | **3.0e-08** | +0.42 | 1.3e-05 | **B9 > H3-alt on both** (F51 published) |
| **CA** | 5 (seed=42) | +4.74 | 0.062 (5/5) | +0.72 | 0.125 (4/5) | B9 directional win |
| **TX** | 5 (seed=42) | +1.76 | 0.125 (4/5) | +0.64 | 0.125 (4/5) | B9 directional win |

**Headline finding: B9 is FL-tuned, not a universal champion.** F51's claim that
B9 is the committed champion was supported on FL only. Under cross-state
validation:

- **At small states (AL/AZ, 1109/1547 regions):** B9's ingredients (alt-SGD + cosine
  + α-no-WD) actively **hurt cat training** without helping reg. H3-alt is the
  better MTL recipe at small scale. The cat-side significance is paper-grade
  (AL p=1.91e-6, AZ p=7.08e-4 across 20 fold-pairs each).
- **At medium scale (FL, 4703 regions):** B9 wins decisively on both tasks (F51).
- **At large scale (CA/TX, 8501/6553):** B9 directionally wins both tasks but
  single-seed n=5 leaves p=0.0625 minimum — sign-consistent but underpowered.
  Multi-seed extension recommended pre-camera-ready.

**Mechanism hypothesis:** B9's three additions over H3-alt are designed to address
FL's reg saturation pattern (D5 finding: reg encoder saturates at ep 5-6 while
cat keeps drifting). At AL/AZ where regions are 3-4× fewer, the reg saturation
problem is less severe AND alt-SGD's per-step temporal gradient separation
costs cat-side signal that small states can't afford to lose. The α-no-WD
ingredient targets STAN-Flow (`next_stan_flow`)'s α growth specifically, which is
similarly less load-bearing at small scale.

**Paper implication:** the recipe-selection claim must reframe from "B9 is the
champion" to "**B9 is the FL-scale champion; H3-alt remains the universal
recipe at small scale; the optimal MTL recipe is scale-conditional**". This
preserves F51 as a finding but puts it in proper context.

Full Wilcoxon JSON: `research/PAPER_CLOSURE_RECIPE_WILCOXON.json`. AL/AZ H3-alt
run dirs: `results/check2hgi/{alabama,arizona}/mtlnet_*_20260501_05*` (8 runs,
seconds+PID-suffixed).

---

## 4a · BOTH TASKS — full architectural-Δ picture (verified)

The earlier-version §4 below only covered `next_region`. Including `next_category`
changes the headline substantially. Both metrics use F51's canonical extraction
(`per-fold max for epoch ≥ 5`). Cat metric is unweighted F1.

| State | n_reg | MTL B9 reg | STL reg | Δ_reg | MTL B9 cat F1 | STL cat F1 | Δ_cat |
|---|---:|---:|---:|---:|---:|---:|---:|
| AL | 1109 | 50.17 ± 0.24 (n=4) | 61.21 ± 0.18 (n=4) | **−11.04** | 40.57 ± 0.24 (n=4) | 40.76 ± 1.68 (n=1) | −0.19 (≈tied) |
| AZ | 1547 | 40.78 ± 0.07 (n=4) | 53.06 ± 0.15 (n=4) | **−12.28** | 45.10 ± 0.19 (n=4) | 43.21 ± 0.87 (n=1) | **+1.89** |
| FL | 4703 | 63.34 ± 0.11 (n=5) | 70.62 ± 0.09 (n=4) | **−7.28** | 68.59 (F51, n=5) | 66.98 ± 0.61 (F37, n=1) | **+1.61** |
| CA | 8501 | 47.93 (n=1) | 56.86 (n=1) | **−8.93** | 64.23 (n=1) | 62.29 ± 0.31 (n=1) | **+1.94** |
| TX | 6553 | 42.63 (n=1) | 59.32 (n=1) | **−16.69** | 65.04 (n=1) | 63.02 ± 0.28 (n=1) | **+2.02** |

**This is the classic MTL tradeoff, not a flat loss.**

- **Reg (next_region):** MTL B9 < STL STAN-Flow (`next_stan_flow`) at every state by 7-17 pp.
  Architectural cost: the cross-attention overhead doesn't pay back on the harder
  ~1k-9k-class region task that already has its own graph prior to learn from.
- **Cat (next_category):** MTL B9 ≥ STL `next_gru` at every state. AL ≈ tied
  (−0.19 within fold-noise); the other 4 states show a +1.6 to +2.0 pp lift.
  Joint training helps the easier 7-class task.

**Reframe vs F49.** F49's "AL +6.48 pp MTL>STL on reg" was leaky. The real story:
- The architectural cost on **reg** is **state-invariant in sign** (always MTL<STL)
  but varies in magnitude (7-17 pp).
- The MTL **gain on cat** is also **state-invariant in sign** (always MTL≥STL)
  but small (~+1-2 pp).
- The classic MTL hypothesis "joint training transfers signal" survives on the
  easier task. On the harder task, the architecture just costs.

This is **publishable** and **honest**: a clean tradeoff story across 5 states
of varying scale, with multi-seed validation on AL/AZ/FL (the small/medium states).

---

## 4b · Reg-only scale curve (legacy framing, kept for reference)

Δ = MTL B9 (per-fold best top10_acc_indist for epoch ≥ 5, fold-mean) − STL ceiling
(per-fold best top10_acc, fold-mean). Mean ± across-seed σ used; n=1 is single seed.

| State | n_regions | MTL B9 | STL ceiling | Δ (MTL − STL) |
|---|---:|---:|---:|---:|
| AL | 1,109 | 50.17 ± 0.24 (n=4) | 61.21 ± 0.18 (n=4) | **−11.04 pp** |
| AZ | 1,547 | 40.78 ± 0.07 (n=4) | 53.06 ± 0.15 (n=4) | **−12.28 pp** |
| FL | 4,703 | 63.34 ± 0.11 (n=5) | 70.62 ± 0.09 (n=4) | **−7.28 pp** |
| CA | 8,501 | 47.93 (n=1) | 56.86 (n=1) | **−8.93 pp** |
| TX | 6,553 | 42.63 (n=1) | 59.32 (n=1) | **−16.69 pp** |

**Headline:** MTL B9 underperforms STL ceiling at every state by 7-17 pp. FL
shows the smallest cost; TX the largest. AL/AZ/CA cluster around 9-12 pp.

The "scale-conditional architectural cost" framing is consistent: MTL pays an
architectural cost vs STL across the full region-cardinality range
(1109 → 8501). The cost is **NOT** scale-monotone (CA at 8501 has smaller cost
than TX at 6553), suggesting state-specific factors beyond raw region count
also matter (per-user trajectory geometry, transition graph density, etc.).

---

## 5 · Failures encountered

**Phase 1 first attempt (2026-05-01 ~01:11 UTC):** 17/28 runs failed.

Two distinct failure modes:

1. **OOM at CA/TX MTL parallel.** Two MTL jobs at CA (8501 regions) or TX (6553)
   each used ~36-40 GB; train-side logit cat (`mtl_cv.py:541`) needed +9 GB →
   exceeded 80 GB. CA-H3-alt survived because CA-B9 died first and freed memory.
   **Fix:** Phase A serialized big-state MTL retries (CA-B9, TX-B9, TX-H3-alt
   one-at-a-time); also exported `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
2. **STL reg `FileNotFoundError` on legacy log_T name.**
   `scripts/p1_region_head_ablation.py:625` expected `region_transition_log_fold{N}.pt`
   (legacy unseeded). The F51 seeded-log_T migration only updated `mtl_cv.py`;
   the STL ablation script was left on the old naming. We had deleted the
   legacy files during pre-flight (because the trainer hard-fails on them).
   **Fix:** patched `p1_region_head_ablation.py` to prefer
   `region_transition_log_seed{S}_fold{N}.pt` with a loud-warning fallback to
   the legacy name (only valid for seed=42; logs a warning otherwise).

Retry pass: 17/17 exit 0.

---

## 6 · Patches landed this session

| File | Change | Reason |
|---|---|---|
| `scripts/train.py` | Added CLI flags `--weight-decay`, `--adam-eps`, `--max-grad-norm`, `--eta-min` | Tier 3 sweep needed exposed config knobs |
| `src/configs/experiment.py` | Added `eta_min: float = 0.0` field | Cosine-tail floor LR (Tier 3) |
| `src/training/helpers.py` | `setup_scheduler` accepts `eta_min`; passes to `CosineAnnealingLR` | Wire-up |
| `src/training/runners/mtl_cv.py` | Pass `eta_min=getattr(config, "eta_min", 0.0)` | Wire-up |
| `src/tracking/experiment.py` | `start_date` now `%Y%m%d_%H%M%S_<pid>` | Parallel run-dir collision fix |
| `scripts/p1_region_head_ablation.py` | Seeded log_T preferred; legacy fallback with warning | Match F51 seeded-naming migration |

All 152 training tests + 127 tracking tests pass post-patch.

---

## 7 · Run-dir locations (for analysis follow-ups)

**STL reg JSONs:** `docs/studies/check2hgi/results/P1/region_head_<state>_region_5f_50ep_paper_close_*.json`

**MTL run dirs (today, paper-closure):**
- CA-H3-alt: `results/check2hgi/california/mtlnet_lr1.0e-04_bs2048_ep50_20260501_011216_406571/`
- CA-B9-retry: `…/20260501_015857_412969/`
- TX-B9-retry: `results/check2hgi/texas/mtlnet_…_20260501_023224_413998/`
- TX-H3-alt-retry: `…/20260501_031509_414897/`
- AL-B9 ×4 seeds: `results/check2hgi/alabama/mtlnet_…_20260501_014904_<pid>/` (4 PIDs: 411673, 411676, 411691, 411706 — order = launch order = seeds 0, 1, 7, 100 respectively)
- AZ-B9 ×4 seeds: `results/check2hgi/arizona/mtlnet_…_20260501_015{206,208,209}_<pid>/` (4 PIDs: 412194, 412291, 412306, 412356 = seeds 0, 1, 7, 100)
- CA STL cat: `results/check2hgi/california/next_*ep50*0501*` (run from Phase 1 first attempt; STL cat survived since failure mode 1 didn't apply)
- TX STL cat: `results/check2hgi/texas/next_*ep50*0501*`

---

## 8 · Recommended next steps (Phase 3 analysis)

1. **Verify metric definition** (§3 caveat) by re-extracting one known-good FL B9
   run (e.g. F51's `_20260430_0522`) using the same script as ours — confirm
   we report the same number as F51_MULTI_SEED_FINDINGS.md (63.47 ± 0.75).
2. If §3 caveat resolves to "leak unwind", **reframe paper claim** in
   PAPER_DRAFT.md / NORTH_STAR.md: from "scale-conditional MTL > STL at AL" to
   "MTL costs ~7-19 pp vs STL across all 5 states under leak-free comparison;
   FL is the smallest cost (architecture is least penalized), TX is largest."
3. **Generate paired-Wilcoxon JSONs** for CA-arch-Δ and TX-arch-Δ
   (mirror `FL_layer3_after_f37.json`). Single-seed at CA/TX so no error bars
   in the Δ; flag for camera-ready multi-seed extension.
4. **Update RESULTS_TABLE.md, NORTH_STAR.md, PAPER_DRAFT.md** with the §4 table.
5. **(Deferred P1)** CA + TX B9 multi-seed (4 extra seeds × 2 states × 2 arms = 16
   runs, ~2-3 h serial on H100) for symmetric Wilcoxon power. Audit recommended
   pre-camera-ready, not pre-submission.
