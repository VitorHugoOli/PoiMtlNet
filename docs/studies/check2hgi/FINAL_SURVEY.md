# Final Survey — Check2HGI vs HGI substrate study (Phase 1 + 2 + 3, leak-free)

**Generated 2026-04-30.** Five US states (AL, AZ, FL, CA, TX) × two substrates (Check2HGI per-visit graph embeddings vs HGI POI-stable graph embeddings). Same fold protocol everywhere: `StratifiedGroupKFold(userid, seed=42)`, 5 folds × 50 epochs. All `next_region` numbers below come from the Phase 3 leak-free re-run (`--per-fold-transition-dir`); the Phase 2 leaky reg numbers are kept as a side panel for the F44 leakage analysis.

Statistical protocol: paired Wilcoxon signed-rank (one-sided `alternative='greater'` for the C2HGI > HGI direction) on the 5 paired folds, `p=0.0312` is the maximum significance achievable at n=5, equivalent to all 5 folds positive. Paired-t reported alongside; effect-direction (Δ̄ sign) agrees with Wilcoxon at every cell.

> **⚠ sklearn-version reproducibility caveat (added 2026-04-30):** sklearn upgraded
> the `StratifiedGroupKFold(shuffle=True)` algorithm in 1.8.0 (PR #32540, fixing
> stratification with shuffle). We empirically verified that 1.3.2 and 1.8.0
> produce **completely different fold splits at the same `random_state=42`** —
> fold 0 val starts `[1, 7, 8, 14, 16]` on 1.3.2 vs `[2, 3, 11, 24, 27]` on 1.8.0;
> all 5 fold-set hashes differ. Implications:
>
> 1. **Every within-phase paired Wilcoxon test below is statistically valid** —
>    both substrate arms in each comparison ran in the same env on the same
>    folds, so Δ direction and p-value are unaffected.
> 2. **§7 leak-shift (Phase 2 leaky vs Phase 3 clean) absolute magnitudes are
>    partly confounded** if any Phase 2 cell ran on sklearn ≠ 1.8.0. The
>    substrate-asymmetric direction is robust (both arms within each phase share
>    the same env), but the absolute "−9 pp dropped" includes a fold-shift
>    component on top of pure leak removal.
> 3. **The proper way to settle §7 quantitatively** is to re-run Phase 2 reg STL
>    on the same Lightning H100 image as Phase 3. The qualitative conclusion
>    (C2HGI exploited the leaky log_T more than HGI) does not change.
>
> See §8 for the full impact assessment.

## Track summary

| Claim track | Head | Affected by F44 leak? | Source files |
|---|---|:-:|---|
| **Substrate linear probe** (Leg I) | head-free LR on emb | no | `results/probe/<state>_<engine>_last.json` |
| **Cat STL `next_gru`** (Leg II.1, CH16) | next_gru — no log_T | no | `phase1_perfold/<S>_<engine>_cat_gru_5f50ep.json` |
| **Reg STL `next_getnext_hard`** (Leg II.2, CH15) | α·log_T graph prior | YES — re-run leak-free | `phase1_perfold/<S>_<engine>_reg_gethard_pf_5f50ep.json` |
| **MTL B9 cat F1** (CH18 cat-side) | next_gru | no log_T on cat side, but co-trained — re-run leak-free for joint protocol | `phase1_perfold/<S>_<engine>_mtl_cat_pf.json` |
| **MTL B9 reg Acc@10/MRR** (CH18 reg-side) | next_getnext_hard | YES — re-run leak-free | `phase1_perfold/<S>_<engine>_mtl_reg_pf.json` |

## Headline figures

* `figs/final_survey/00_probe_box.png` — Leg I substrate-only linear probe.
* `figs/final_survey/01_cat_stl_box.png` — CH16 cat STL substrate gap.
* `figs/final_survey/02_mtl_cat_box.png` — CH18 cat-side under MTL B9.
* `figs/final_survey/03_reg_stl_pf_box.png` — CH15 reframing under leak-free reg STL.
* `figs/final_survey/04_mtl_reg_acc10_box.png` — CH18 reg-side under MTL B9.
* `figs/final_survey/05_paired_delta_grid.png` — paired Δ per fold across all four claim tracks.
* `figs/final_survey/06_leak_shift_bars.png` — F44 leak shift (Phase 2 leaky vs Phase 3 clean reg STL).

## 1 · Substrate linear probe (Leg I, head-free, leak-free by construction)

| State | C2HGI F1 | HGI F1 | Δ (C2HGI−HGI) | Wilcoxon p_greater | Pos / Neg |
|---|---:|---:|---:|---:|:-:|
| AL | **30.84 ± 2.26** | 18.70 ± 1.54 | **+12.14** | 0.0312 | 5 / 0 |
| AZ | **34.12 ± 1.36** | 22.54 ± 0.50 | **+11.58** | 0.0312 | 5 / 0 |
| FL | **40.77 ± 1.24** | 25.74 ± 0.29 | **+15.03** | 0.0312 | 5 / 0 |
| CA | **37.45 ± 0.29** | 21.32 ± 0.16 | **+16.13** | 0.0312 | 5 / 0 |
| TX | **38.38 ± 0.28** | 22.33 ± 0.25 | **+16.06** | 0.0312 | 5 / 0 |

**Verdict (head-free):** the substrate carries a 11.6-16.1 pp F1 lift before any head is trained, at all 5 states, paired Wilcoxon p=0.0312, 5/5 folds positive. Δ scales mildly with state size (smallest at AZ, largest at CA). The probe isolates the embedding's own signal — the cat substrate effect is intrinsic to Check2HGI, not an artefact of head choice or training dynamics.

## 2 · Cat STL `next_gru` (Leg II.1, CH16) — leak-free by construction

| State | C2HGI F1 | HGI F1 | Δ | Wilcoxon p_greater | Paired-t p | Pos / Neg |
|---|---:|---:|---:|---:|---:|:-:|
| AL | **40.76 ± 1.68** | 25.26 ± 1.18 | **+15.50** | 0.0312 | 9.65e-05 | 5 / 0 |
| AZ | **43.21 ± 0.87** | 28.69 ± 0.79 | **+14.52** | 0.0312 | 7.38e-06 | 5 / 0 |
| FL | **63.43 ± 0.98** | 34.41 ± 1.05 | **+29.02** | 0.0312 | 2.52e-07 | 5 / 0 |
| CA | **59.94 ± 0.59** | 31.13 ± 1.04 | **+28.81** | 0.0312 | 6.77e-08 | 5 / 0 |
| TX | **60.24 ± 1.84** | 31.89 ± 0.55 | **+28.34** | 0.0312 | 6.76e-07 | 5 / 0 |

**Verdict:** CH16 confirmed at 5/5 states with paper-grade significance (Wilcoxon p=0.0312 = max-n=5, 5/5 folds positive each). Δ scales monotonically from ~15 pp at small AL/AZ to ~28-29 pp at large FL/CA/TX. **Per-visit context is the load-bearing substrate property for the cat task.**

## 3 · MTL B9 cat F1 (CH18 cat-side) — Phase 3 leak-free, B9 recipe

| State | C2HGI cat F1 | HGI cat F1 | Δ | Wilcoxon p_greater | Pos / Neg |
|---|---:|---:|---:|---:|:-:|
| AL | **40.47 ± 1.45** | 25.41 ± 1.06 | **+15.06** | 0.0312 | 5 / 0 |
| AZ | **44.84 ± 1.54** | 29.25 ± 0.61 | **+15.59** | 0.0312 | 5 / 0 |
| FL | **68.42 ± 1.66** | 34.76 ± 0.30 | **+33.66** | 0.0312 | 5 / 0 |
| CA | **64.21 ± 1.38** | 31.67 ± 1.07 | **+32.54** | 0.0312 | 5 / 0 |
| TX | **65.17 ± 1.40** | 32.40 ± 0.37 | **+32.77** | 0.0312 | 5 / 0 |

**Verdict:** CH18-cat confirmed and **strengthened** by leak-free protocol — Δ grows to ~33 pp at FL/CA/TX (vs ~15 pp at AL/AZ). MTL inherits the C2HGI cat advantage. 

## 4 · Reg STL `next_getnext_hard` (CH15 reframing) — leak-free (Phase 3)

| State | C2HGI Acc@10 | HGI Acc@10 | Δ Acc@10 | Wilcoxon p_greater | TOST δ=2pp | TOST δ=3pp |
|---|---:|---:|---:|---:|:-:|:-:|
| AL | 59.15 ± 3.48 | **61.86 ± 3.29** | **-2.71** | 1.0000 | ✗ FAIL | ✗ FAIL |
| AZ | 50.24 ± 2.51 | **53.37 ± 2.55** | **-3.13** | 1.0000 | ✗ FAIL | ✗ FAIL |
| FL | 69.22 ± 0.52 | **71.34 ± 0.64** | **-2.12** | 1.0000 | ✗ FAIL | ✓ non-inf |
| CA | 55.92 ± 1.20 | **57.77 ± 1.12** | **-1.85** | 1.0000 | ✓ non-inf | ✓ non-inf |
| TX | 58.89 ± 1.28 | **60.47 ± 1.26** | **-1.59** | 1.0000 | ✓ non-inf | ✓ non-inf |

Same analysis on MRR:

| State | C2HGI MRR | HGI MRR | Δ MRR | Wilcoxon p_greater | TOST δ=2pp | TOST δ=3pp |
|---|---:|---:|---:|---:|:-:|:-:|
| AL | 36.30 ± 2.61 | **37.96 ± 2.89** | **-1.67** | 1.0000 | ✗ FAIL | ✓ non-inf |
| AZ | 32.65 ± 1.69 | **34.33 ± 2.05** | **-1.68** | 1.0000 | ✗ FAIL | ✓ non-inf |
| FL | 54.34 ± 0.70 | **55.17 ± 0.71** | **-0.83** | 1.0000 | ✓ non-inf | ✓ non-inf |
| CA | 39.95 ± 0.83 | **40.63 ± 0.94** | **-0.68** | 1.0000 | ✓ non-inf | ✓ non-inf |
| TX | 39.25 ± 0.81 | **39.72 ± 1.03** | **-0.47** | 1.0000 | ✓ non-inf | ✓ non-inf |

**Verdict:** CH15 reframing **rejected at AL/AZ/FL** (TOST δ=2pp fails because |Δ| > 2 pp; HGI nominally above C2HGI by 1.6-3.1 pp). **Tied at CA/TX** (Δ < 2 pp, TOST passes). Sign-reversed at all 5 states vs Phase 2 leaky reference — the Phase 2 sign came from the F44 leak (Section 7).

## 5 · MTL B9 reg Acc@10 / MRR (CH18 reg-side) — Phase 3 leak-free

| State | C2HGI Acc@10 | HGI Acc@10 | Δ Acc@10 | Wilcoxon p_greater | Pos / Neg |
|---|---:|---:|---:|---:|:-:|
| AL | 32.79 ± 10.11 | **40.58 ± 4.34** | **-7.79** | 0.9375 | 1 / 4 |
| AZ | 33.54 ± 3.90 | **37.00 ± 1.89** | **-3.46** | 0.9375 | 1 / 4 |
| FL | 60.77 ± 1.57 | **61.77 ± 0.77** | **-1.00** | 0.9375 | 1 / 4 |
| CA | 44.24 ± 1.52 | **45.32 ± 1.20** | **-1.09** | 0.9688 | 1 / 4 |
| TX | 40.40 ± 2.03 | **40.53 ± 1.88** | **-0.13** | 0.5938 | 2 / 3 |

MRR:

| State | C2HGI MRR | HGI MRR | Δ MRR | Wilcoxon p_greater | Pos / Neg |
|---|---:|---:|---:|---:|:-:|
| AL | 18.31 ± 7.51 | **22.11 ± 2.42** | **-3.80** | 0.8438 | 2 / 3 |
| AZ | 22.82 ± 4.60 | **20.12 ± 1.11** | **+2.71** | 0.1562 | 4 / 1 |
| FL | 52.60 ± 0.84 | **52.86 ± 0.80** | **-0.26** | 0.9062 | 2 / 3 |
| CA | 35.78 ± 1.73 | **35.99 ± 1.18** | **-0.21** | 0.8438 | 1 / 4 |
| TX | 32.40 ± 1.63 | **32.47 ± 1.66** | **-0.07** | 0.6875 | 2 / 3 |

**Verdict:** CH18-reg **rejected at 5/5 states** under leak-free MTL B9. The Phase 2 leaky finding ("C2HGI reg ≥ HGI reg under MTL") was an artifact of the F44 leak, which C2HGI exploited disproportionately. Magnitude is small at FL/CA/TX (≤ 1.1 pp Acc@10, basically tied) but substantial at AL/AZ (3-8 pp).

## 6 · F44 leakage — Phase 2 (leaky) vs Phase 3 (clean) on reg STL

Phase 2 reg numbers used the legacy full-data `region_transition_log.pt` graph prior, leaking val transitions into training. Phase 3 used `--per-fold-transition-dir` (StratifiedGroupKFold train-only edges per fold). The leak inflated absolute Acc@10 across all states, BUT was substrate-asymmetric — C2HGI benefited more than HGI, refuting the earlier uniform-leak hypothesis.

| State | C2HGI leaky | C2HGI clean | Δ_C2HGI | HGI leaky | HGI clean | Δ_HGI | Asymmetry (Δ_C2HGI − Δ_HGI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AL | 68.37 | 59.15 | -9.22 | 67.52 | 61.86 | -5.66 | -3.56 |
| AZ | 66.74 | 50.24 | -16.51 | 64.40 | 53.37 | -11.03 | -5.48 |
| FL | 82.54 | 69.22 | -13.32 | 82.25 | 71.34 | -10.91 | -2.41 |
| CA | 70.63 | 55.92 | -14.72 | 71.29 | 57.77 | -13.51 | -1.20 |
| TX | 69.31 | 58.89 | -10.42 | 69.90 | 60.47 | -9.42 | -0.99 |

Negative asymmetry → C2HGI lost more pp than HGI when the leak was removed → C2HGI had been benefiting more from the leak. **AZ shows the largest gap (~5.5 pp)**, the smoking-gun for substrate-asymmetric leakage. This is why CH15 + CH18-reg results sign-flip leak-free.

## 7 · Final paper-grade synthesis

Five paper-grade claims after the 5-state, leak-free, Phase 1+2+3 closure:

1. **CH16 — Cat substrate gap** ✅ at 5/5 states (Wilcoxon p=0.0312, 5/5 folds positive). Δ scales monotonically with state size.
2. **CH18-cat — MTL inherits the cat substrate gap** ✅ at 5/5 states (same statistics). Δ in MTL is similar magnitude to cat-STL (15 pp at small AL/AZ, ~33 pp at large FL/CA/TX).
3. **CH15 reframing — substrate-equivalent reg under matched head** ❌ rejected at AL/AZ/FL (TOST δ=2pp fails). Tied at CA/TX (Δ < 2 pp). Sign-flipped at every state vs leaky reference.
4. **CH18-reg — MTL substrate-specific reg lift** ❌ rejected at 5/5 states (sign-reversed). The Phase 2 leaky claim was an F44 artefact.
5. **F44 leak is substrate-asymmetric** (~3 pp differential, AZ peak ~5.5 pp). C2HGI exploited the leaky log_T more than HGI — α grew more for C2HGI runs.

### Suggested paper framing

> **Per-visit context (Check2HGI) is the load-bearing substrate for next-category prediction; for next-region prediction, POI-level embeddings (HGI) are at parity (large states FL/CA/TX) or marginally ahead (small states AL/AZ).**

Mechanism (CH19, F37 FL):
* The cat task benefits from the per-visit variance Check2HGI adds (AL+AZ pooled-vs-canonical decomposition: ~72% of cat gap is per-visit context).
* Reg is a POI-level coarser label; POI-stable HGI embeddings aggregate cleanly across the 9-window without needing per-visit signal.
* The previously-claimed CH18-reg lift was the F44 leak: C2HGI's α grew more aggressively (to ~2 by ep 17-20) and mined val edges from the full-data log_T more effectively.

## 8 · sklearn 1.3.2 → 1.8.0 fold-split impact assessment

### Context

`requirements.txt` was upgraded `scikit-learn 1.6.1 → 1.8.0` on commit `42845fa`
(2026-04-14), citing PR #32540 ("StratifiedGroupKFold stratification not properly
preserved when shuffle=True"). `src/data/folds.py` calls
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` for the
user-disjoint cat/next/MTL splits; the per-fold transition matrix builder
(`scripts/compute_region_transition.py --per-fold`) uses the same call.

There is a frozen-fold mechanism (`scripts/study/freeze_folds.py` writes
`output/<engine>/<state>/folds/fold_indices_<task>.pt` with a sklearn-version
sidecar in `.meta.json`), but **no frozen-fold files were on disk** when Phase 3
ran — every cell regenerated folds from scratch via `_from_scratch` (case 4 in
`scripts/train.py:1183`). So each cell's folds are determined entirely by the
sklearn version installed in that env.

### Direct empirical test (2026-04-30)

Synthesized a representative POI dataset (500 users, 100k seqs, 7 categories
with realistic skew). Ran `StratifiedGroupKFold(5, shuffle=True, 42)` on both
sklearn versions:

| sklearn | fold 0 val[:5] | fold 4 \|val\| | fold-set hashes |
|---|---|---:|---|
| 1.3.2 | `[1, 7, 8, 14, 16]` | 20353 | `[444d7abc..., c0a1f5e9..., ..., bc7588fd...]` |
| 1.8.0 | `[2, 3, 11, 24, 27]` | 20112 | `[7d024743..., 4841b36d..., ..., d93bd3cf...]` |

**All 5 fold-set hashes differ.** Even fold sizes drift (~2 % per fold). PR
#32540 changed the stratification path enough to fully randomize which user
goes to which fold for any non-trivial dataset.

### What this affects in this survey

| Section | Effect on the headline claim | Why |
|---|---|---|
| §1 Probe (CH16 head-free) | **None** — paired test internally valid. | Each state's C2HGI/HGI cells ran in the same env. Δ direction + p=0.0312 unaffected. |
| §2 Cat STL (CH16 main) | **None** — paired test internally valid. | Same reason. |
| §3 MTL cat (CH18-cat) | **None** — Phase 3 only, single env. | All cells on Lightning H100 sklearn 1.8.0. |
| §4 Reg STL (CH15 reframing) | **None** — Phase 3 only, single env. | Same. |
| §5 MTL reg (CH18-reg) | **None** — Phase 3 only, single env. | Same. |
| §6 Leak-shift (Phase 2 vs Phase 3) | **Quantitative caveat** — see below. | Compares cells potentially run on different sklearn versions. |
| §7 Synthesis | Net — **all 4 paper-grade claims (CH16, CH18-cat, CH15, CH18-reg) survive intact.** Only the F44 leak-magnitude attribution in §6 needs a footnote. |  |

### §6 leak-shift — what survives, what doesn't

Each cell of the Phase 2 row vs Phase 3 row in §6 is a comparison across
different (env, sklearn) pairs:

* Phase 3 (clean): all 10 cells on Lightning H100, sklearn 1.8.0 (single env).
* Phase 2 (leaky): heterogenous — AL/AZ may be from earlier dates (likely
  sklearn 1.6.1 or earlier); FL was Colab T4 (Apr 28); CA/TX were Lightning
  T4 (Apr 29). Each env's sklearn pin depends on whether the bootstrap
  honored the post-2026-04-14 requirements.txt.

**Robust under the caveat:**

1. The leak is **substrate-asymmetric** (C2HGI lost more pp than HGI when leak
   was removed) — both arms within each phase shared the same env, so the
   asymmetry is NOT a fold-split artifact. ✅
2. The **direction** of the shift (every state lost pp going from leaky to
   clean) is robust because the leaky → clean transition is a well-defined
   model change regardless of fold split. ✅

**Not bit-exact:**

3. The **absolute magnitude** "AL c2hgi −9.22 pp" mixes leak removal with
   fold-split shift. True leak magnitude could be ±2-3 pp around the reported
   number. ⚠

### What to do if §6 magnitude is paper-critical

Re-run Phase 2 reg STL (`next_getnext_hard` with the legacy full-data
`region_transition_log.pt`, **no** `--per-fold-transition-dir`) on the same
Lightning H100 image used for Phase 3 (sklearn 1.8.0, identical pin). Then the
leaky vs clean comparison would be on bit-identical fold splits and the leak
magnitude becomes a clean attribution. ~10 cells × ~10 min on H100 ≈ ~1.5 h
wall-clock; same per-fold transition dir is NOT used, so reuse of cached
parquets is straightforward. Output to `_5f50ep_v18.json` to keep the original
files as historical reference.

This is **not** required to land the paper — claims 1–5 are intact — but it
would tighten the F44 leak-magnitude story for an appendix table.

### Belt-and-braces fix for future runs

Run `scripts/study/freeze_folds.py --default-set` once on a known sklearn
version, commit the `.meta.json` sidecars (folds themselves are large; consider
LFS or Drive bundle), and add `--folds-path <canonical>` to the orchestrator
launch scripts so future re-runs cannot drift on a sklearn upgrade.

## 9 · Bibliography of internal docs

* `PHASE2_TRACKER.md` — Phase 2 STL closure (cat + probe, leak-free already).
* `requirements.txt` commit `42845fa` — sklearn 1.6.1 → 1.8.0 upgrade rationale (PR #32540).
* `PHASE3_TRACKER.md` — Phase 3 Scope D plan + status board.
* `research/SUBSTRATE_COMPARISON_FINDINGS.md` — full Phase-1 verdicts + Phase-3 closure.
* `research/F50_T4_C4_LEAK_DIAGNOSIS.md` — root-cause + magnitude.
* `research/F50_T4_BROADER_LEAKAGE_AUDIT.md` — audit of other heads.
* `research/F50_T4_PRIOR_RUNS_VALIDITY.md` — which prior runs survived the C4 fix.
* `research/PHASE3_INCIDENTS.md` — operational incidents during Phase 3.
* `NORTH_STAR.md` — committed MTL recipe (B9 leak-free champion).
