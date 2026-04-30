# check2HGI study — handover (2026-04-30)

**For the next agent / collaborator picking up this study.** Read this first; it points to everything else.

## TL;DR — where things stand

The F50/F60 series + the original MTL improvements study (architecture / optimizer / hyperparameters) are **functionally complete**. Paper headline locked. Two follow-up runs are deferred (CA/TX on A100) but mitigation is wired and documented.

| pillar | status |
|---|---|
| Architecture experiments (P1–P5 + Cross-Stitch + PLE + MMoE-class + head ablations) | ✅ done |
| Optimizer / training-dynamics (NashMTL family, P4 alt-SGD, B-series, F60–F65) | ✅ done |
| Hyperparameter sweeps (D-series, F-series, A-series, audit C-series) | ✅ done |
| Cross-state portability (FL/AL/AZ/GA) | ✅ done |
| **Cross-state CA/TX (large states)** | ⏸ deferred to A100 (script ready) |
| C4 graph-prior leakage audit + fix | ✅ closed (uniform leak; relative Δs preserved) |
| C-series audit hygiene (C1–C8) | ✅ all closed |
| Mechanism receipts (D5 encoder, F63 α) | ✅ both paper figures in `research/figs/` |
| Paper claim survival under leak fix | ✅ 8/9 survive; 3 absolute headlines restated |

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

### Deferred (intentional, not "missing")

1. **CA + TX P3 5f×50ep MTL CH18** — 8501 / 9870+ regions. OOMed on RunPod 4090 (24 GB) at the per-epoch train-side logit catting. Targeted for **A100 (40-80 GB)** where it fits comfortably. Run command in `scripts/run_p3_ca_unblock_attempt.sh`. **Per-fold log_T must be rebuilt at the trainer's seed** (F51 finding 2026-04-30 — see Common Pitfalls #2): `python scripts/compute_region_transition.py --state california --per-fold --seed 42` (and any other seeds you sweep). The CA per-fold log_T files in `output/check2hgi/california/` were deleted during F51 disk cleanup; rebuild from scratch on the A100 box. CA fetch script: `scripts/runpod_fetch_data.sh california`.

2. **STL F37 clean rerun** for FL — current 71.12 number is from a single 5f×50ep run; cross-state STL ceilings still pending if reviewers ask.

3. **Multi-seed variance (P6)** — ✅ **DONE 2026-04-30 via F51.** 5/5 seeds {42, 0, 1, 7, 100} give Δreg = +3.48 ± 0.12 pp; pooled paired Wilcoxon p=2.98×10⁻⁸. See `research/F51_MULTI_SEED_FINDINGS.md`.

4. **F51 Tier 3 (optimizer/scheduler hparam sweep)** — design ready in `F50_NORTH_STAR_DEEP_EXPLORATION_PROMPT.md` §3 Tier 3. Not paper-blocking; only run if reviewer asks for additional architecture-via-optimizer probes. ETA ~4 h on 4090.

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
