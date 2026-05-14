# check2HGI study — handover (2026-05-01)

**For the next agent / collaborator picking up this study.** Read this first; it points to everything else.

## TL;DR — where things stand

The F50/F60 series + the original MTL improvements study (architecture / optimizer / hyperparameters) are **functionally complete**. Paper closure run-matrix landed 2026-05-01 on H100. Cross-state P3 (CA + TX) closed; STL ceilings landed at all 5 states with multi-seed at AL/AZ/FL; AL/AZ MTL B9 multi-seed extension closed; F51 Tier 3 ran as Phase 0 and gave a clean negative (B9 locally optimal in optimizer/scheduler axis too).

| pillar | status |
|---|---|
| Architecture experiments (P1–P5 + Cross-Stitch + PLE + MMoE-class + head ablations) | ✅ done |
| Optimizer / training-dynamics (NashMTL family, P4 alt-SGD, B-series, F60–F65) | ✅ done |
| Hyperparameter sweeps (D-series, F-series, A-series, audit C-series) | ✅ done |
| Cross-state portability (FL/AL/AZ/GA + **CA + TX on H100, leak-free**) | ✅ done |
| **Multi-seed AL/AZ/FL (B9 + STL ceilings, both tasks)** | ✅ done 2026-05-01 |
| F51 Tier 3 optimizer/scheduler sweep | ✅ done 2026-05-01 (clean negative, B9 locked) |
| C4 graph-prior leakage audit + fix | ✅ closed (uniform leak; relative Δs preserved) |
| C-series audit hygiene (C1–C8) | ✅ all closed |
| Mechanism receipts (D5 encoder, F63 α) | ✅ both paper figures in `research/figs/` |
| Paper claim survival under leak fix | ⚠ **F49's "AL +6.48 pp MTL>STL on reg" REFUTED** under symmetric leak-free comparison; reframed as classic MTL tradeoff (see "Paper closure" §) |
| Paired Wilcoxon at all 5 states, both tasks | ✅ done 2026-05-01 (`research/PAPER_CLOSURE_WILCOXON.json`) |

## Recipe-selection reframe (added 2026-05-01 after B9-vs-H3-alt audit)

A late-session check (prompted by "have we made executions on H3-alt to compare
against B9?") surfaced a recipe-portability gap: AL/AZ had B9 multi-seed but
no leak-free H3-alt anchor. 8 H3-alt runs at AL/AZ filled the gap (~10 min on H100).

**Outcome — B9 is FL-scale-tuned, not universal:**

| State | n_pairs | Δ_reg pp | p_reg | Δ_cat pp | p_cat |
|---|---:|---:|---:|---:|---:|
| AL | 20 | **−0.35** | **1.9e-03** | **−2.22** | **1.9e-06** |
| AZ | 20 | −0.09 | 0.23 (n.s.) | **−0.96** | **7.1e-04** |
| FL | 25 | **+3.48** | **3.0e-08** | +0.42 | 1.3e-05 |
| CA | 5  | +4.74 | 0.062 (5/5) | +0.72 | 0.125 (4/5) |
| TX | 5  | +1.76 | 0.125 (4/5) | +0.64 | 0.125 (4/5) |

At AL/AZ B9 is *significantly worse on cat* than H3-alt. B9's additions (alt-SGD,
cosine, α-no-WD) only pay off at FL (F51 result). At larger scale (CA/TX) B9
directionally wins both tasks but n=5 single-seed limits inference. **Paper
recipe-selection narrative reframes: "B9 is FL-scale champion; H3-alt remains
the universal recipe at small scale; recipe is scale-conditional."** F51's
"committed champion" claim is preserved as an FL-scale finding.

Full Wilcoxon JSON: `research/PAPER_CLOSURE_RECIPE_WILCOXON.json`.
AL/AZ H3-alt run dirs: `results/check2hgi/{alabama,arizona}/mtlnet_*_20260501_05*`.

## Paper closure 2026-05-01 — the headline reframe

Cross-state P3 + STL ceilings + multi-seed at AL/AZ/FL converged on a **classic MTL tradeoff** picture, sign-consistent across all 5 states. Pre-F50 F49 numbers (which showed AL favoring MTL by +6.48 pp on reg) were a leak artifact of the legacy full-data `region_transition_log.pt`; under leak-free `--per-fold-transition-dir` with seeded naming, the AL pattern matches every other state.

| State | n_pairs | Δ_reg pp | p_reg | Δ_cat F1 pp | p_cat |
|---|---:|---:|---:|---:|---:|
| AL (4 seeds) | 20 | **−11.04** | **1.9e-06** | −0.19 | 0.76 (≈tied) |
| AZ (4 seeds) | 20 | **−12.27** | **1.9e-06** | **+1.90** | **1.9e-06** |
| FL (seed=42)  | 5 | −7.99 | 0.0625 | (n/a) | — |
| CA (seed=42)  | 5 | −8.92 | 0.0625 | +1.94 | 0.0625 |
| TX (seed=42)  | 5 | −16.69 | 0.0625 | +2.02 | 0.0625 |

**Reg:** MTL B9 < STL `next_getnext_hard` ceiling at every state by 7-17 pp.
**Cat:** MTL ≥ STL `next_gru` ceiling at every state. AL has no significant cat gain; AZ/CA/TX/FL all show ~+1.6-2.0 pp.

**Story:** the cross-attention architecture's expressiveness gets spent on cat-helps-cat (joint training transfers signal to the easier 7-class task) at the cost of the harder ~1k-9k-class region task that already has its own graph prior to learn from in `next_getnext_hard`. This is the **classic MTL tradeoff**; the F49 "architecture-dominant at AL" framing was a leak artifact.

Full results + provenance: **`PAPER_CLOSURE_RESULTS_2026-05-01.md`**. Phase plan + cancellation log: **`PAPER_CLOSURE_PHASES.md`**. Wilcoxon JSON: **`research/PAPER_CLOSURE_WILCOXON.json`**. Canonical analysis script: **`scripts/analysis/paper_closure_wilcoxon.py`** (uses F51 extraction methodology).

## Headline numbers (FL, leak-free, 5f×50ep, ≥ep5)

**Multi-seed validated (F51, 2026-04-30) — see `research/F51_MULTI_SEED_FINDINGS.md`:**

| seeds | B9 reg | H3-alt reg | Δreg | paired Wilcoxon | verdict |
|---|---:|---:|---:|---|---|
| {42, 0, 1, 7, 100} mean | **63.34 ± 0.11** | 59.86 ± 0.22 | **+3.48 ± 0.12** | **5/5 seeds at p=0.0312 each** | ✅ **DECISIVELY ROBUST** |
| pooled 5×5=25 fold-pairs | — | — | **+3.48** | **p_reg=2.98×10⁻⁸ (25/25); p_cat=1.33×10⁻⁵ (19/25)** | paper-grade on **both tasks** |

Single-seed table (kept for cross-references):

| recipe (seed=42) | reg top10 | cat F1 | Δreg vs H3-alt | paired Wilcoxon | verdict |
|---|---:|---:|---:|---|---|
| **STL F37 ceiling** (clean) | **71.12 ± 0.59** | n/a | — | — | upper bound |
| **B9 (P4 + Cosine + α-no-WD)** ⭐ | **63.47 ± 0.75** | **68.59 ± 0.79** | **+3.34** | **p=0.0312, 5/5 on BOTH tasks** | **CHAMPION** (Pareto-dominant) |
| P4-alone (constant) | 63.41 ± 0.77 | 67.82 | +3.28 | p=0.0312, 5/5 | minimal-paper-grade |
| H3-alt (clean baseline) | 60.12 ± 1.14 | 68.34 | 0 | — | anchor |

**Story:** P4 alternating-optimizer-step closes ~3.5 pp of the 7.7 pp STL→MTL gap. Mechanism is per-step temporal gradient separation, not architectural. Cross-attention mixing is dead at B9's depth=2 (3-way confirmed via P1, F52 P5, F53 cw sweep) — refined by F51 Tier 2 to **depth-conditional**: alive at depth=3 (Pareto-trade), breaks cat at depth=4. Reg encoder physically saturates at ep 5–6 while cat encoder keeps drifting through ep 38 — the temporal-dynamics bottleneck is real and substrate-agnostic.

**F51 multi-seed validation (Tier 1, 2026-04-30):** the recipe is essentially deterministic in the partition-difficulty axis (B9 absolute reg σ_across_seeds = 0.11 pp); the +3.48 pp lift is the cross-seed mean. **F51 Tier 2 (capacity sweep, 2026-04-30):** B9 is locally optimal in 5/7 capacity dimensions; no paper-grade lift available via capacity scaling. NEW finding: cat width-stability cliff — wider shared backbone breaks cat without affecting reg (P4 shields reg, cat unshielded at LR=1e-3). NEW per-seed log_T leak found and fixed mid-sweep (legacy unseeded `region_transition_log_fold{N}.pt` filename → `region_transition_log_seed{S}_fold{N}.pt`; trainer hard-fails on legacy/missing).

For all numbers: `research/F50_RESULTS_TABLE.md`.

## External resources

**Drive backup of result snapshots from other machines** (Colab T4, RunPod 4090,
A100): https://drive.google.com/drive/folders/1cka4py5MElM-mDbBW5JC8JLS36qnjmq-

Consult this when a referenced run dir (e.g. F51 multi-seed `_20260430_05XX`,
or pre-paper-closure runs from earlier sessions) no longer exists under
`results/check2hgi/<state>/`. The Drive is the backup-of-record after local
disk-cleanup passes. To reuse a run with `scripts/analysis/*.py`, download the
folder and place it under `results/check2hgi/<state>/` preserving the original
run-dir name so the metric-extraction scripts find it.

## Where to read what

### Tier 1 — start here
1. **`research/F50_T4_FINAL_SYNTHESIS.md`** — current state + pointers (canonical entry point)
2. **`research/F50_HISTORY.md`** — chronological narrative of how we got here (~2h read)
3. **`research/F50_RESULTS_TABLE.md`** — all paper-grade numbers in one place
4. **`HANDOVER.md`** (this file) — for new agents

### Tier 2 — deep dives by topic
- **F51 multi-seed validation (paper-grade headline):** `research/F51_MULTI_SEED_FINDINGS.md` (Δreg = +3.48 ± 0.12 pp across 5 seeds, pooled p < 10⁻⁷)
- **F51 Tier 2 capacity sweep:** `research/F51_TIER2_CAPACITY_FINDINGS.md` (B9 locally optimal; no capacity-scaling paper-grade lift; cat width-stability cliff finding)
- **C4 graph-prior leakage:** `research/F50_T4_C4_LEAK_DIAGNOSIS.md` (root cause), `research/F50_T4_BROADER_LEAKAGE_AUDIT.md` (which heads affected), `research/F50_T4_PRIOR_RUNS_VALIDITY.md` (which prior runs survive)
- **Mechanism narrative:** `research/F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md` (temporal-dynamics breakthrough), `research/F50_D5_ENCODER_TRAJECTORY.md` (encoder saturation receipt), `research/figs/f63_alpha_trajectory.png` (α growth figure)
- **Latest F50 follow-ups:** `research/F50_B2_F52_F65_F53_FINDINGS.md` (B2 rejected, F52 P5 paper-grade, F65 cycling NOT cause, F53 cw sweep)
- **Audit C-series fixes:** `research/F50_T3_AUDIT_FINDINGS.md`
- **Phase-3 CA/TX fallback decision tree:** `research/C05_P3_NULL_RESULT_FALLBACK.md`
- **Live tracker (richest update log):** `research/F50_T4_PRIORITIZATION.md`

### Tier 3 — historical (archived)
- `research/archive/F50/F50_PROPOSAL_AUDIT_AND_TIERED_PLAN.md` — original tiered plan
- `research/archive/F50/F50_HANDOFF_2026-04-28.md` — mid-stream handoff
- `research/archive/F50/F50_T1_RESULTS_SYNTHESIS.md` — Tier-1 sub-experiment results
- `research/archive/F50/F50_T1_1_CAT_HEAD_PATH_DECISION.md` — cat-head Path A/B decision
- `research/archive/F50/F50_T1_5_CROSSATTN_ABSORPTION.md` — original cross-attn absorption analysis
- `research/archive/F50/F50_T1_5_T2_RESULTS_SYNTHESIS.md` — T1.5 + T2 sub-results
- `research/archive/F50/F50_T3_HYPERPARAM_BRAINSTORM.md` — original B1–B10 brainstorm
- `research/archive/F50/F50_DELTA_M_FINDINGS.md` — F50 T0 Δm scoreboard
- `research/archive/F50/F50_CORRECTED_SCOREBOARD.md` — selector-corrected scoreboard
- `research/archive/F50/F50_T4_RERUN_DECISION.md` — re-run decision matrix (consolidated into synthesis)

### Project-wide
- `NORTH_STAR.md` — current champion config
- `PAPER_DRAFT.md` — paper draft
- `PAPER_PREP_TRACKER.md` — paper prep status
- `OBJECTIVES_STATUS_TABLE.md` — objectives across states
- `CONCERNS.md` — open concerns (all C01-C17 except C05/C10/C13)
- `FOLLOWUPS_TRACKER.md` — followup tracker
- `PHASE2_TRACKER.md`, `PHASE3_TRACKER.md` — phase tracking

## What's left to do

### Done 2026-05-01 (paper closure)

1. **CA + TX P3 5f×50ep MTL CH18** — ✅ **DONE on H100 80GB.** CA-B9, CA-H3-alt, TX-B9, TX-H3-alt all completed. CA-B9 leak-free reg = 47.93 (per-task best top10_indist), STL = 56.86 → Δ = −8.93. TX-B9 = 42.63, STL = 59.32 → Δ = −16.69. Run dirs in `PAPER_CLOSURE_RESULTS_2026-05-01.md §7`. Per-fold log_T rebuilt at seed=42 with seeded naming during pre-flight.

2. **STL F37 clean rerun for FL** — partially done 2026-05-01: STL reg multi-seed extension at FL covers seeds {0, 1, 7, 100} (mean = 70.62 ± 0.09); seed=42 baseline (71.12 ± 0.59) from prior c4_clean run. STL cat at FL still single-seed (F37 P1 0.6698 ± 0.0061); per-fold cat numbers are not stored in the F37 summary JSON.

3. **STL ceilings at CA + TX** — ✅ **DONE.** Both tasks, seed=42, leak-free.
   STL reg: CA 56.86 / TX 59.32. STL cat: CA 62.29 ± 0.31 / TX 63.02 ± 0.28.

4. **AL + AZ multi-seed for MTL B9 + STL reg** (P0+P1 from audit) — ✅ **DONE.** 4 seeds {0, 1, 7, 100} × 2 states × 2 arms = 16 runs. AL B9 σ_across_seeds = 0.24, AZ = 0.07 — recipe-deterministic, matching F51's FL finding.

5. **F51 Tier 3 (optimizer/scheduler hparam sweep)** — ✅ **DONE 2026-05-01 as Phase 0.** 15 smokes 5f×30ep on B9 base × {weight_decay, max_grad_norm, eta_min, OneCycle pct_start, AdamW eps}. **Clean negative**: all smokes within ±0.5 pp of B9, best `pct_start=0.5` at +0.31 pp (below threshold). B9 is locally optimal in optimizer/scheduler axis too. CLI flags `--weight-decay`, `--adam-eps`, `--max-grad-norm`, `--eta-min` patched into `scripts/train.py` and remain available for future ablations.

6. **Paired Wilcoxon at all 5 states for both tasks** — ✅ **DONE 2026-05-01.** See `research/PAPER_CLOSURE_WILCOXON.json` and `scripts/analysis/paper_closure_wilcoxon.py`. AL/AZ multi-seed Δ_reg/Δ_cat highly significant (p < 10⁻⁶ where signed-consistent). FL/CA/TX single-seed n=5 give p=0.0625 minimum (signed-consistent in all cases where data exists).

### Deferred / camera-ready (P1, not paper-blocking)

A. **CA + TX MTL B9 multi-seed** (4 extra seeds × 2 states × 2 arms = 16 runs).
   Currently single-seed at CA+TX. Audit recommended pre-camera-ready, not pre-submission.
   ETA on H100: ~3-4 h.

B. **AL + AZ + FL STL cat multi-seed.** STL cat ceilings at AL/AZ/FL are single-seed
   (F37). Symmetric multi-seed extension would close the cat-side error bars. ETA: ~30 min on H100.

C. **F51 multi-seed FL B9 per-fold archival.** F51's per-seed run dirs (`_20260430_05XX`)
   weren't kept locally; only the seed-level summary JSON was retained. To compute
   per-fold paired Wilcoxon at FL multi-seed scale (B9 vs STL), would either re-run
   F51 or restore the run dirs from offsite backup. The seed=42 paired Wilcoxon at FL
   (p=0.0625, 5/5 in sign) is locally reproducible.

D. **Patches landed this session.** See `PAPER_CLOSURE_RESULTS_2026-05-01.md §6`.
   Notably: `src/tracking/experiment.py:start_date` now seconds+PID granular (parallel
   run-dir collision fix); `scripts/p1_region_head_ablation.py` now reads seeded log_T
   with legacy fallback; 4 new CLI flags on `scripts/train.py` for Tier 3 knobs.

### Cancelled / dropped (mechanism-refuted)

- **Bi-Level GSL (T3.1)** — long-tail clustering hypothesis refuted by uniform-leak observation + D5 encoder saturation.
- **Distillation (T3.2)** — superseded by F62 two-phase REJECTED.
- **MMoE / DSelectK / Identity-backbone** — architectural bottleneck refuted; not in final scope.
- **NashMTL / GradNorm screening** — superseded by static_weight + per-head LR + P4 alt-SGD.

## How to reproduce key results

Per-fold log_T (the C4 fix — required for any clean run):
```bash
python scripts/compute_region_transition.py --state florida --per-fold
# repeat for alabama, arizona, georgia, california, texas
```

B9 champion run at FL:
```bash
bash scripts/run_p1_h3alt_f62_catchup.sh   # template — see other run_*.sh
```

The B9 recipe specifically:
```
--task mtl --task-set check2hgi_next_region --state florida --engine check2hgi
--model mtlnet_crossattn --cat-head next_gru --reg-head next_getnext_hard
--reg-head-param d_model=256 --reg-head-param num_heads=8
--reg-head-param transition_path=output/check2hgi/florida/region_transition_log.pt
--task-a-input-type checkin --task-b-input-type region
--folds 5 --epochs 50 --seed 42 --batch-size 2048
--cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3
--gradient-accumulation-steps 1
--per-fold-transition-dir output/check2hgi/florida
--no-checkpoints --no-folds-cache
--min-best-epoch 5
--mtl-loss static_weight --category-weight 0.75
--alternating-optimizer-step                 # P4
--scheduler cosine --max-lr 3e-3
--alpha-no-weight-decay                      # B9
```

P4-alone (minimal paper-grade): drop `--scheduler cosine`, `--alpha-no-weight-decay`. Keep `--alternating-optimizer-step`.

## Code interventions landed (commits)

| feature | commit | location |
|---|---|---|
| Per-fold log_T (C4 fix) | `60107eb` | `src/data/folds.py`, `--per-fold-transition-dir` |
| P4 alternating-SGD | (older) | `src/training/runners/mtl_cv.py:457` |
| B9 alpha-no-WD | (older) | `src/training/helpers.py` `setup_per_head_optimizer` |
| F61 min-best-epoch selector | (older) | `--min-best-epoch` flag |
| F62 scheduled_static step mode | (older) | `src/losses/scheduled_static/` |
| F63 α-trajectory logging | (older) | `src/training/runners/mtl_cv.py` head_alpha block |
| D5 encoder Frobenius drift | `ad7126d` | `src/training/runners/mtl_cv.py` D5 block |
| C7 aggregation_basis stamp | `83cbf68` | `src/tracking/storage.py` |
| F52 identity_cross_attn | `56f6e29` | `src/models/mtl/mtlnet_crossattn/model.py` |
| F65 joint-loader strategies | `56f6e29` | `src/utils/progress.py` `zip_longest_cycle` |
| B2/F64 reg_head warmup-decay LR | `56f6e29` | `src/training/helpers.py` `setup_scheduler` |

## Working state at handover

- Branch: `worktree-check2hgi-mtl`
- Latest commit: `87ab608` (revert skip-train-metrics; CA → A100)
- Tests: 47 training tests pass, 127 tracking tests pass, 36 integration tests pass.
- No zombie processes. GPU free.
- Repo clean (`git status` shows nothing).

## Data + results not in git

The `results/` and `output/` directories are gitignored (sizes: results 1.2 GB, output ~20 GB across all states). For the CURRENT paper run dirs, see `handover_results.tar.gz` at the repo root (71 MB, gitignored). It contains ONLY the gitignored data:

- `results_florida_clean/` — 14 paper-grade FL 5f×50ep run dirs (B9 champion + H3-alt anchor + F53 cw sweep + B2/F52/F65 follow-ups + D5 trajectory pair + PLE clean)
- `results_cross_state/` — most-recent 3 MTL runs each for AL/AZ/GA
- `README.md` — run-dir mapping + re-extraction commands

Everything else (synthesis docs, figs, scripts, analyzer JSONs) is in the git repo — the bundle's README points there. Extract with `tar xzf handover_results.tar.gz` and place alongside the cloned repo to re-run the analyzer.

## Common pitfalls for the next agent

1. **Always use `--per-fold-transition-dir`** for any `next_getnext_hard*` run. Without it you re-introduce C4 leakage and inflate by 13–17 pp.
2. **The per-fold log_T must match `--seed`** (F51 finding, 2026-04-30). Files are seed-tagged: `region_transition_log_seed{S}_fold{N}.pt`. Build with `python scripts/compute_region_transition.py --state STATE --per-fold --seed S` for each seed you plan to train at. Trainer hard-fails if the seed-tagged file is missing or if a legacy unseeded `region_transition_log_fold{N}.pt` is present (the old filename leaks ~80% of val transitions when used at any seed != the seed it was built at).
3. **`--folds 1` triggers `n_splits = max(2, 1) = 2`**, which produces a 2-fold split that does NOT match the 5-fold-keyed per-fold log_T. For smokes that need a quick signal, prefer `--folds 5 --epochs 30` over `--folds 1 --epochs 50`.
4. **Always use `--min-best-epoch 5`** for paper-grade selection. Without it the GETNext α init artifact at ep 0–2 captures the "best" epoch.
5. **Per-head LR mode** (`--cat-lr/--reg-lr/--shared-lr`) requires `mtlnet_crossattn` model. Other models will reject it.
6. **B9 requires per-head LR mode** (it relies on the alpha_no_wd group being peeled out from `reg`).
7. **`--alternating-optimizer-step` requires `--mtl-loss static_weight`** (mutual exclusion with `scheduled_static` etc.).
8. **High-cardinality reg heads OOM on 24 GB GPU** at the per-epoch train-side logit catting (`mtl_cv.py:541`). For CA (8501 regions) and TX (9870+ regions), use A100 or implement streaming train metrics.
9. **CLAUDE.md branch context:** `CLAUDE.local.md` at repo root points to `docs/studies/check2hgi/AGENT_CONTEXT.md` for active study scope.

## Contact / authority

- The user (`vho2009@hotmail.com`) is the final authority on champion freeze, paper claims, and run scheduling.
- Default behavior: confirm before any destructive action (force-push, branch-delete, etc.). The user may ask Claude Code to operate autonomously in some cases — those instructions live in CLAUDE.md / persistent memory.

---

**End of handover. Latest synthesis: `research/F50_T4_FINAL_SYNTHESIS.md`. Live tracker: `research/F50_T4_PRIORITIZATION.md`. Good luck.**
