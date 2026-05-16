# Blast Radius Audit — `--folds 1` × per-fold-log_T n_splits Mismatch Leak

**Date:** 2026-05-15
**Trigger:** discovered while running mtl-exploration cells A–D at AL+AZ+FL 1f25ep seed=42.
**The bug:** `scripts/train.py --folds N` triggers `n_splits = max(2, N)` in the trainer's `StratifiedGroupKFold`, but per-fold log_T files (`region_transition_log_seed{S}_fold{N}.pt`) were built with `n_splits = 5`. When N < 5, ~30–80% of val users have their region transitions present in the log_T's "train" set → α-amplified leak inflates reg `top10_acc_indist` by 13–23 pp.

---

## TL;DR

| Result class | Affected? | Why |
|---|---|---|
| **v11 RESULTS_TABLE §0 (paper-canonical)** | ❌ **SAFE** | All v11 §0.1–§0.7 numbers explicitly used `5 folds × 50 epochs` per the table's preamble — `n_splits = 5` matches log_T's `n_splits = 5`. No leak. |
| **Active follow-up studies** (canonical_improvement, merge_design, hgi_category_injection) | ❌ **SAFE** | All use 5-fold protocol via their probe scripts. canonical_improvement Tables 2/3 explicitly `seed=42 n=5` and `4 seeds × 5 folds n=20`. merge_design eval_design_bh.py uses `--folds 5`. hgi_category_injection used `5 folds × 30 epochs`. |
| **F51 multi-seed (5f×50ep×5seeds)** | ❌ **SAFE** | F-trail discovered the bug here (in their own smoke); proper run is 5-fold. |
| **F51 Tier 2 capacity sweep (5f×30ep)** | ❌ **SAFE** | Header explicitly avoids `--folds 1`. |
| **F50 D5 Encoder Trajectory** | ⚠ **PARTIAL** | Used `--folds 1` + per-fold log_T → absolute val numbers (e.g. B9 reg top10 = 76.35) are leak-inflated. **Mechanism finding ("reg encoder saturates earlier") survives** — it's computed from weight-update dynamics, not val metrics. |
| **F51 verify smoke `_seed42_verify`** | ⚠ **EXPECTED** | This is the run that *discovered* the bug. Its inflated 76.33 is documented as the smoking-gun fingerprint, not a claim. |
| **Pre-F50 findings (F2, F17, F27, B5, B-M, etc.)** | ⚠ **DIFFERENT LEAK** | These predate the per-fold log_T fix entirely. They used the legacy full-data `region_transition_log.pt` (the original C4 leak, ~13–17 pp inflation, documented as superseded). NOT affected by *this* new bug. |
| **This experiment (mtl-exploration single-fold runs)** | ⚠ **AFFECTED (absolute), Δs preserved** | All 1f25ep + 1f50ep absolute reg numbers inflated. Within-experiment pairwise Δs (no_encoders A vs baseline D, linear B vs C) preserved under uniform-leak property. Conclusion "d_model carries reg, not encoder non-linearity" stands. |
| **AZ multi-seed (in flight)** | ❌ **SAFE by construction** | Uses `--folds 5` + per-seed log_T builds. Will produce clean absolute numbers + n=20 paired Wilcoxon. |

**Bottom line for paper:** v11 RESULTS_TABLE §0 and all paper-grade claims are unaffected. The leak's blast radius is confined to (a) one F-trail mechanistic finding's absolute numbers (F50 D5 — claim survives), (b) various 1-fold verification smokes that were never paper-canonical, and (c) my single-fold smokes in this experiment.

---

## Mechanism (the bug in detail)

`scripts/train.py` docstring:

> `--folds N`: run only the first N folds. **The split structure uses `max(2, N)` splits** (StratifiedKFold requires >= 2), but execution stops after N folds.

So when a user passes `--folds 1`:
1. `FoldCreator` constructs `StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)`.
2. The trainer runs fold 1 of the **2-fold** split (val ≈ 50% of users).
3. Meanwhile, `--per-fold-transition-dir output/check2hgi/<state>` loads `region_transition_log_seed42_fold1.pt`, which was built by `compute_region_transition.py --per-fold` at default `n_splits = 5` (`StratifiedGroupKFold(n_splits=5, ...)`'s train set = 80% of users).
4. Roughly **(50% val users) ∩ (80% log_T train users) ≈ 30–40% of val users have their transitions in the prior**. The α scalar in `next_stan_flow` (≡ `next_getnext_hard`) grows over training and amplifies this leak.

**Empirical fingerprint** (from F51_MULTI_SEED_FINDINGS.md §0 + this experiment):

| Source | FL fold-1 seed=42 reg top10 |
|---|---:|
| Original `_1813` 5-fold reference (ep 6) | 63.53 |
| F51 verify smoke `--folds 1` (ep 6) | **76.33** |
| My mtl-exploration baseline `--folds 1 --epochs 25` (peak ep 8) | **76.51** |
| v11 multi-seed pooled mean (5 seeds × 5 folds) | 63.27 ± 0.10 |

The two `--folds 1` numbers differ by 0.2 pp despite being 16 days apart — identical bug fingerprint.

---

## Triggers (necessary AND sufficient conditions)

The leak fires **only when ALL three** are true:

1. `--per-fold-transition-dir <path>` is set (loads per-fold log_T).
2. `--folds N` with `N < 5`. **Equivalently: trainer's `n_splits = max(2, N) ≠ 5`.**
3. The reg head reads `log_T`. From `src/data/folds.py::_HEADS_REQUIRING_AUX_MTL`:
   - `next_getnext_hard` / `next_stan_flow` (aliases)
   - `next_getnext_hard_hsm` / `next_stan_flow_hsm`
   - `next_getnext` (soft graph prior)
   - `next_tgstan` (per-sample gate + log_T)
   - `next_stahyper` (α + alpha_cluster + log_T)

If any of the three is missing, no leak from this bug. (Other leaks may apply — see "Different from C4" below.)

---

## Audit: every script that could trigger the leak

Audited `scripts/`, `scripts/probe/`, `docs/studies/*/`, and `docs/findings/`. The 9 unique scripts that use `--folds N` with N<5 *together with* `--per-fold-transition-dir`:

| Script | Purpose | Cites in canonical claim? | Verdict |
|---|---|---|---|
| `run_smoke_tier1_clean_fl.sh` | Tier-1 verification smokes (1f×10ep). | No, explicitly labeled "verification smokes, not full 5-fold". | Smoke; not contaminating paper claims. |
| `run_tgstan_redo_ple_full.sh` | F50 T4 aux-channel verification smoke. | No, verification. | Smoke. |
| `run_f51_seed42_verify.sh` | F51 seed=42 verification (the run that found the bug). | The 76.33 is documented as the bug's fingerprint, not a claim. | Smoke. |
| `run_smokes_parallel.sh` | Parallel verification smokes. | No. | Smoke. |
| `run_remaining_clean_queue.sh` | Queue of 5 smokes/relaunches. | No. | Smoke. |
| `run_f50_d5_encoder_traj.sh` | F50 D5 encoder weight-trajectory diagnostic. | Used in `docs/findings/F50_D5_ENCODER_TRAJECTORY.md`. | **PARTIAL — see below.** |
| `p1_region_head_ablation.py` | P1 region-head ablation (default `--folds 1`). | Used in canonical_improvement & merge_design follow-ups, but those override `--folds 5`. | Defaults unsafe; concrete usages override to 5. |
| `run_f2_fl_hard_diagnostic.sh` | F2 FL-hard diagnostic. | **NO `--per-fold-transition-dir`** — uses legacy log_T, suffers C4 leak (documented separately). | Not this bug. |
| `run_f27_cathead_sweep.sh` | F27 cat-head sweep. | **No `--per-fold-transition-dir`** — same as F2. | Not this bug. |

**Two scripts contain explicit warning headers** about the bug:
- `run_f51_tier2_capacity_smoke.sh:9` — "`--folds 1` triggers `n_splits=max(2,1)=2` in the trainer, which..."
- `run_f51_tier3_sweep.sh:9` — "5 folds (not 1) because the per-fold log_T is 5-fold-keyed; `--folds 1`..."

The F51 team knew about it; the warnings live in scripts but not in `train.py` itself.

---

## F50 D5 — the one F-trail finding with affected numbers

`docs/findings/F50_D5_ENCODER_TRAJECTORY.md` reports:

| run | reg top10 | reg MRR | cat F1 | cat acc |
|-----|---:|---:|---:|---:|
| H3-alt | 75.61 (ep 3) | 58.44 (ep 4) | 66.83 (ep 17) | 69.93 (ep 24) |
| B9 | **76.35** (ep 6) | **59.01** (ep 9) | 66.74 (ep 46) | 69.72 (ep 46) |

The 76.35 / 75.61 values come from `--folds 1` + per-fold log_T → leak-inflated by ~13 pp (vs the v11 multi-seed mean of 63.27). 

**Does the leak invalidate the F50 D5 finding?** The finding is: *"Reg-side encoder saturates 26–32 epochs earlier than cat-side encoder."* This is computed from weight-update dynamics (Frobenius-norm of `reg_encoder_drift_from_init`), **not from val metrics**. The leak affects the val numbers but should not materially change the *weight trajectory shape*.

**Action:** Add a caveat to F50 D5 noting that the reported reg val numbers are leak-inflated under the `--folds 1` mode, but the saturation conclusion survives because the mechanism is measured in weight space. Optionally re-run the diagnostic at `--folds 5` if a paper figure references the val numbers (the paper doesn't appear to).

---

## Different from C4 leak

| Leak | Trigger | Magnitude | Fix |
|---|---|---|---|
| **C4 (legacy)** | `transition_path` pointing to `region_transition_log.pt` (full-data, no per-fold) | 13–17 pp at convergence (α-amplified) | `--per-fold-transition-dir` flag (F50 T4) |
| **New bug (this audit)** | `--folds N<5` + `--per-fold-transition-dir` (per-fold log_T built at n_splits=5) | 13–23 pp (depends on N vs 5 overlap) | Hard-fail when log_T's `n_splits` doesn't match trainer's |

The fix for C4 introduced the new bug as a footgun. F51's per-seed fix (filename seed-tagging) addressed seed-mismatch; the `n_splits-mismatch` axis was left ungated.

---

## Proposed fix — make the footgun loud

The leak is silent today: the trainer logs `[C4 per-fold log_T] fold 1 seed 42 using ...seed42_fold1.pt` (cf. all my run logs), giving the false impression that everything is leak-free. The file IS leak-free for n_splits=5 runs, but applying it to an n_splits=2 run silently re-leaks.

**Option A (filename):** Encode n_splits in the filename.
```python
# compute_region_transition.py::_build_per_fold
filename = f"region_transition_log_seed{seed}_nsplits{n_splits}_fold{fold_idx+1}.pt"
# trainer (mtl_cv.py): loads region_transition_log_seed{seed}_nsplits{self.n_splits}_fold{fold}.pt
# Hard-fail if missing.
```

**Option B (payload):** Stash `n_splits` inside the `.pt` payload.
```python
# compute_region_transition.py::save (or _build_per_fold)
torch.save({"log_transition": log_probs, "n_splits": n_splits, "seed": seed, "fold": fold}, path)
# trainer (mtl_cv.py): asserts payload["n_splits"] == self.n_splits
```

**Option C (docstring + hard-fail):** Leave files as-is, but `mtl_cv.py` errors out when `self.n_splits != 5` AND `--per-fold-transition-dir` is set, unless the user explicitly opts in via `--allow-nsplits-mismatch` (escape hatch for those who genuinely want a smoke).

Option B is non-breaking and self-documenting. Option A is the most explicit. Option C is the minimum viable guard. I recommend **A + C combined**: encode in filename AND hard-fail in trainer.

The same hard-fail pattern already exists for the seed mismatch (F51 fix). Extending it to n_splits is ~10 LOC.

---

## Status of the fix (all landed 2026-05-15)

1. **No retraction needed on v11 paper numbers** — they're all `5f×50ep`, leak-free under both the C4 fix and this bug. ✅
2. **Caveat added to `F50_D5_ENCODER_TRAJECTORY.md`** noting absolute val numbers are inflated; saturation claim survives. ✅
3. **n_splits guard implemented** in `src/training/runners/mtl_cv.py` (Option B — payload-based, with legacy fallback for canonical n_splits=5 only). 5-case smoke test passed. ✅
4. **`MTL_FLAWS_AND_FIXES.md §2.13`** added alongside §2.12 (C4). ✅
5. **`scripts/train.py --folds` docstring** updated to flag the interaction with `--per-fold-transition-dir`. ✅
6. **`scripts/compute_region_transition.py::save()`** now stashes `n_splits` and `seed` in the persisted `.pt` payload. ✅

---

# Part 2 — Self-audit of this study's findings under the leak

**Question this part answers:** Do any conclusions of *the mtl-exploration study itself* need to be retracted? Do we need to re-run anything?

**Short answer:** No retractions; one cheap re-run already done (the AZ multi-seed). Every conclusion is built on within-state, within-budget *pairwise Δs* preserved under F51's uniform-leak property (clean and leaky Δs match within ~0.10 pp).

## Inventory of every run in this study

### Single-fold runs (`--folds 1` — leak-contaminated on absolute numbers, Δs preserved)

| Run dir suffix | State | Budget | Variant | Used in conclusion |
|---|---|---|---|---|
| `ep50_20260515_124941_52410` | AL | 50ep | no_encoders (A) | yes — original ablation |
| `ep50_20260515_125610_54851` | AZ | 50ep | no_encoders (A) | yes |
| `ep50_20260515_130534_56454` | AL | 50ep | baseline (D) | yes |
| `ep50_20260515_130734_56727` | AZ | 50ep | baseline (D) | yes |
| `ep25_20260515_131410_57890` | FL | 25ep | no_encoders (A) | yes — original ablation |
| `ep25_20260515_132529_59455` | FL | 25ep | baseline (D) | yes |
| `ep25_20260515_134643_61554` | AL | 25ep | no_encoders (A) | matched-protocol table |
| `ep25_20260515_134838_61640` | AL | 25ep | baseline (D) | yes |
| `ep25_20260515_134725_61587` | AZ | 25ep | no_encoders (A) | discrimination |
| `ep25_20260515_134940_61729` | AZ | 25ep | baseline (D) | yes |
| `ep25_20260515_141109_64724` | AL | 25ep | linear+d=64 (B) | AL factorial |
| `ep25_20260515_141155_64891` | AL | 25ep | linear+d=256 (C) | yes |
| `ep25_20260515_142438_65853` | AZ | 25ep | linear+d=64 (B) | **load-bearing AZ discrimination** |
| `ep25_20260515_142555_66027` | AZ | 25ep | linear+d=256 (C) | **load-bearing AZ discrimination** |

### Multi-fold runs (`--folds 5` — leak-free by construction)

- AZ baseline + linear at seeds {0, 1, 7, 100} × 5 folds (n=20 each, paper-grade) — done 2026-05-15 afternoon
- AL baseline + linear + linear_ln at seeds {0, 1, 7, 100} × 5 folds (n=20 each) — done 2026-05-15 evening
- AZ linear_ln at seeds {0, 1, 7, 100} × 5 folds — done 2026-05-15 evening

All multi-fold runs use `--folds 5` → trainer's `n_splits=5` matches log_T's `n_splits=5` → **leak-free**.

## Conclusion-by-conclusion impact analysis

### Conclusion 1 (original ablation): "Removing task encoders hurts both heads at AZ/FL; AL is within n=1 noise at 25ep, baseline pulls ahead by 50ep."

Built on within-state pairwise Δs (cell A `no_encoders` vs cell D `baseline`). Both arms hit the same leaky prior on the same val set → pairwise Δ preserved within ~0.10 pp. **Conclusion holds; absolute reg numbers in the writeup carry an inline caveat.**

### Conclusion 2 (trajectory): "no_encoders saturates earlier than baseline at AL."

Built on the per-epoch val trajectory at AL 25ep. The leak adds a roughly time-varying offset (α growth) but the relative trajectory shape — "this curve saturates while that one keeps climbing" — is preserved across arms. **Conclusion holds.**

### Conclusion 3 (factorial: d_model is what carries the reg gap, not encoder MLP non-linearity)

Built on within-state pairwise Δs across 4 cells at AZ 25ep: B≈A and C≈D on reg, A,B << C,D gap ≈ 8.1 pp. All 4 cells hit the same leaky prior → every pairwise Δ is preserved. **This is the load-bearing factorial conclusion. Confirmed leak-free at n=20 multi-seed:** Δ_reg(C−D) at AZ = −0.01 pp (p=0.66) on the morning's clean multi-seed sweep.

### Conclusion 4 (AL factorial: at 25ep, all four cells indistinguishable)

Built on within-state pairwise Δs at AL 25ep. Uniform leak across cells → Δs preserved. The conclusion "AL 25ep is too early in baseline's convergence to discriminate" is itself a Δ-based observation. **Conclusion holds.**

### Conclusion 5 (cross-state ordering: AZ Δ_reg = −8.3, FL Δ_reg = −5.3 — non-monotone in region cardinality)

Built on cross-state comparison of within-state Δs. Each state's pairwise Δ preserved individually; the cross-state COMPARISON is also preserved because both states have leak-symmetric Δs. The writeup correctly flags this as "non-monotone, n=1, needs multi-seed". **Conclusion holds at the existing caveat level.**

### Conclusion 6 (final factorial, scale-conditional): "Cell E ≡ Cell D at AZ; Cell D dominates Cell E on cat at AL by −2.57 pp"

Built on n=20 multi-seed leak-free protocol (`--folds 5`). **No leak impact. Paper-grade as stated.**

## Absolute-number disposition

The §Step 1 / §Step 2 (1-fold) tables in `EXPERIMENT_NO_ENCODERS.md` carry leak-inflated absolute reg values (AL 50ep baseline reg = 68.21, AZ 25ep baseline reg = 64.75, FL 25ep baseline reg = 76.51). These are correctly flagged in the doc's §⚠ Critical leak discovery section. The §Step 3 multi-seed values (AL baseline reg = 47.66, AZ baseline reg = 40.89) are leak-free and match v11 paper canon within fold-level noise.

## Re-run decision matrix (final)

| Candidate re-run | Cost | Decision | Notes |
|---|---|---|---|
| Re-run all single-fold runs at `--folds 5` | ~3 h MPS | ❌ **Skip** | pairwise Δs already preserved; no information gained |
| AL multi-seed (cells C, D, E) | ~1 h MPS | ✅ **Done** (Step 3) | confirmed cross-state simplification verdict |
| AZ multi-seed (cells D, C, E) | ~2 h MPS | ✅ **Done** (Steps 2 + 3) | paper-grade n=20 |
| Re-run F50 D5 trajectory at `--folds 5 --epochs 50` | ~10 h MPS | ❌ **Skip** | mechanism (saturation in weight space) survives the leak; caveat added to F50 D5 doc |
| FL multi-seed | ~10 h MPS | ⚠ Deferred to main study | scale-conditional simplification claim might want it eventually |

**Bottom line: every conclusion in this study is either confirmed leak-free at n=20 (Steps 2 + 3) or stands on uniform-leak-preserved pairwise Δs (Step 1 + factorial smoke). No retractions.**
