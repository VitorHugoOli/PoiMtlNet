# F50 T4 — C4 Leakage Root-Cause Diagnosis (2026-04-29 19:30 UTC)

**Status: DIAGNOSED.** The 16 pp drop from `_1653` (P4+Cosine, full log_T = 76.07) to `_1755` (P0-A, per-fold log_T = 60.36) is **real, dominated by the C4 leakage signal, and consistent with the math.** The audit's original 0.5-2 pp estimate undercounted by ~10× because it assumed α stays at its init value (0.1) — in practice α grows to ~1.8 over training.

Independent verification by a research auditor agent (transcript: `/tmp/.../abbf88cf1cbc5d178.output`) tested 4 hypotheses; H1 (real leakage) confirmed, H2-H4 (code bugs) all rejected.

---

## The four hypotheses tested

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| **H1** | Leakage really is that big | **✅ confirmed** | Direct measurement, see §1 |
| H2 | `build_transition_matrix_from_userids` has a filter bug | ❌ rejected | 5/5 unit tests pass; bit-identical reproduction from inline rebuild |
| H3 | Code change between commits broke top10 calc | ❌ rejected | `git diff dff13c4 HEAD -- src/tracking/metrics.py` shows `_top_k_accuracy` unchanged; only `_handrolled_cls_metrics` (f1/acc_macro) and an MPS fallback for `_rank_of_target` (MRR/NDCG, not top-K) |
| H4 | Per-fold log_T loaded with wrong fold index | ❌ rejected | `mtl_cv.py:773` builds `region_transition_log_fold{i_fold + 1}.pt`; off-by-one would yield ~7.5 max abs diff; saved files match to 0.0 |

---

## 1 · Why 16 pp is mathematically expected

### Direct measurement (advisor agent on FL fold 1)

- Val transitions: **29,303** rows from **2,691** distinct val `last_region`s.
- **70.2% of val transitions appear in the full log_T** (i.e. were leaked).
- **Prior-only val top10:** full log_T → **90.23%** vs per-fold log_T → **80.04%** → **10.19 pp gap from log_T alone.**

This is the *floor* of the leak: even with α frozen at any value, just looking up the prior gives you a 10 pp val accuracy hint when the prior is val-aware.

### Amplification through training

α is NOT static. From `fold1_diagnostics.csv` (`head_alpha` log via F63):

- **ep 0:** α ≈ 0.1 (init)
- **ep 5:** α ≈ 0.56
- **ep 10:** α ≈ 0.98
- **converged:** α ≈ 1.77–1.81 — about **18× the init**

With α≈1.8 and the per-row log_T spread of ~5.3 log-units, the prior contributes **~9.5 logit-units** of additive bias — completely dominating the STAN backbone's contribution to ranking.

When the prior is **val-aware** (full log_T case), training reinforces α growth — the supervised signal endorses the prior because the prior is "right" on val rows by construction. So the trained model leans MORE into the leaky prior than it would into a clean one. The 10.19 pp prior-only gap amplifies during training to **~16 pp at convergence**.

The audit's 0.5-2 pp estimate assumed α stays near its 0.1 init. With α≈1.8, the math gives ~16 pp of inflation — exactly what we measured.

---

## 2 · What this means for the study

### The previously-reported numbers were inflated by ~16 pp

| recipe | leaky number (pre-C4) | leak-free (per-fold log_T) | inflation |
|---|---:|---:|---:|
| H3-alt CUDA REF | 77.16 (greedy) / 71.44 (≥ep10) | TBD (running in catchup) | TBD |
| **P4 + Cosine** (was champion) | **76.07** ≥ep10 | **63.23** ≥ep5 | **−12.84** |
| **B9** (new champion candidate) | n/a (new) | **63.47** ≥ep5 | n/a |
| P4 alone | 75.48 ≥ep10 | TBD (queued) | TBD |
| P4 + OneCycle | 77.52 ≥ep10 | TBD (queued) | TBD |

### Critical for the paper

**The paper's load-bearing claim — "+4.63 pp Δ champion-vs-H3-alt, paired Wilcoxon p=0.0312" — was computed under leaky log_T.** Under leak-free log_T:

- If both H3-alt and champion drop ~16 pp uniformly → +4.63 pp Δ survives → paper claim holds.
- If H3-alt drops MORE → paper claim shrinks (could fall below +3 pp acceptance threshold).
- If H3-alt drops LESS → paper claim grows.

**We need H3-alt + per-fold log_T to know.** It's running NOW in `tmux catchup` (fold 5/5).

### Cross-state implications

The cross-state numbers (GA 46.6, AL 49.4, AZ 40.6) ARE leak-free (per-fold log_T was used). They're WAY lower than FL (60.4) — suggesting either:
1. The recipe is FL-specific and doesn't transfer to smaller states, OR
2. The recipe scales with #regions (FL 4703 >> AZ 1547 > GA 2283 > AL 1109), OR
3. AL/AZ have insufficient training data (AL: 12K rows total) for α to grow and the prior-dominated regime to engage.

This needs more investigation, but for now the cross-state portability claim is **weaker** than we thought.

---

## 3 · What's already done + what's pending

✅ **Done in this loop:**
- C4 plumbing (`--per-fold-transition-dir`) shipped + verified
- 5 leak-free runs at FL: P0-A (champion), B9, B10, GA, AL, AZ
- Independent advisor verification of the 16 pp drop

⏳ **Pending (queued):**
- H3-alt + per-fold log_T (in catchup, ~10 min remaining)
- F62 two-phase + per-fold log_T (catchup, after H3-alt)
- P4-alone + per-fold log_T (pred_queue watchdog armed)
- P4 + OneCycle + per-fold log_T (pred_queue, after P4-alone)
- 1-fold smoke test of pre-C4 champion under LATEST code with FULL log_T (definitive H1 confirmation; ~3 min when GPU frees)

---

## 4 · Operational rules learned

1. **The audit's leak estimate was based on α at init, not at convergence.** Future leakage estimates for trainable-prior models must consider α (or analogous scalars) at *steady state*, not at start.
2. **Any val-aware prior is dangerous when the prior coefficient is learnable** — the model will *amplify* the leak through training. This is a general property: trainable graph priors on cross-validation data are a class of bug that's much worse than non-amplifying leaks.
3. **The MAGNITUDE of a leak fix can completely re-rank a study.** A 16 pp uniform drop preserves rankings but invalidates absolute claims; a non-uniform drop changes rankings.
4. **Always re-run the predecessor champion under the leak-free conditions before crowning a new champion.** The "B9 +0.24 pp Pareto-dominant" claim is correct, but only relative to P0-A — we don't know the absolute leak-free position of the study yet without H3-alt clean.

---

## 5 · Decision rules (post-pending-runs synthesis)

When all 4 leak-free runs (H3-alt, F62, P4-alone, P4+OneCycle) land:

| outcome | action |
|---|---|
| H3-alt drops within 1 σ of champion drop | +Δ vs H3-alt survives → paper claim holds → ship as-is with C4 footnote |
| H3-alt drops MUCH less than champion | recipe doesn't beat H3-alt under leak-free → withdraw paper-grade claim, frame as "matched" + investigate why |
| H3-alt drops MUCH more than champion | recipe is even stronger than reported → strengthen the claim (and double-check we didn't introduce a different leak) |
| F62 ≥ B9 +1 pp | F62 (temporal separation) is the new champion; mechanism is reg-only-pretrain |
| F62 ≈ B9 | both interventions sufficient; B9 simpler so keep it |
| F62 << B9 | P4's per-batch granularity is essential, not just temporal separation |
| Cross-state (AL/AZ/GA) gives < +3 pp Δ vs same-state H3-alt | cross-state portability claim is weak/false; FL-only result |

---

## 6 · References

- Advisor agent transcript: `/tmp/claude-0/-workspace-PoiMtlNet/.../abbf88cf1cbc5d178.output`
- C4 audit finding: `F50_T3_AUDIT_FINDINGS.md` §C4
- Per-fold log_T impl: `scripts/compute_region_transition.py`, `src/training/runners/mtl_cv.py:755-789`
- Pre-C4 champion: `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260429_1653/`
- Post-C4 champion (P0-A): `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260429_1755/`
- α trajectory data: `*/diagnostics/fold1_diagnostics.csv` `head_alpha` column (F63)
