# Session Handoff — 2026-04-24 PM (CH18 attribution chain + budget/LR investigation)

**For the next agent picking up this branch.** Supersedes no prior handoff; complements `SESSION_HANDOFF_2026-04-24.md` (AM session).

## 0 · One-minute summary

- **Afternoon spent** running experiments to attribute CH18's 12-14 pp STL-vs-MTL reg Acc@10 gap.
- **Refuted factors** (data in tracker + research notes): F38 checkpoint selection, F39 loss weight, F41 upstream MLP pre-encoder, F42 epoch budget (inverted).
- **Paper-level finding pending full validation:** **F45 (150ep + constant LR=3e-3)** broke CH18 on AL — **reg Acc@10 = 74.20 ± 2.95, EXCEEDING STL F21c ceiling (68.37)**. But cat F1 collapsed to 10.44 (majority-class baseline). Regime tradeoff, not a dead end.
- **Sweep stopped mid-AZ** (user exit). AL is complete for F44-F47; AZ only has F44 fold 1/5.

## 1 · What ran this session (chronological)

| # | Experiment | State | Status | Key result |
|---|---|---|---|---|
| F38 | diagnostic_task_best re-analysis (zero compute) | AL+AZ existing JSONs | ✅ done | **Refuted** checkpoint-selection — Δ ≤ 0.4 pp at Acc@10. `research/F38_CHECKPOINT_SELECTION.md`. |
| F39 | cat_weight sweep (0.25, 0.50) | AL only (AZ crashed SIGBUS) | ✅ AL done | **Refuted** loss weight — reg Acc@10 window 0.64 pp across cat_weight ∈ {0.25, 0.50, 0.75}. |
| F41 | STL + MTL pre-encoder (--mtl-preencoder) | AL+AZ | ✅ done | **Refuted** Fator 3a — AL 67.95 ± 2.67 vs STL 68.37 (Δ=−0.42 σ-tied); AZ 66.30 vs 66.74. `research/F41_PREENCODER_FINDINGS.md`. |
| F42 | epoch budget (AL 5f × 150ep, default schedule) | AL | ✅ done | **Refuted inversely** — more epochs HURT: reg 59.60 → 56.14, cat 42.71 → 40.68. OneCycleLR stretched. |
| F44 | 150ep + max_lr=1e-3 (gentler peak) | AL only | ✅ AL done | Flat — reg 58.82 ± 4.96, cat 40.20. No recovery. |
| F45 | 150ep + constant LR=3e-3 (no OneCycleLR) | AL only | ✅ AL done | **🎯 reg 74.20 ± 2.95 (+14 pp over B3, +6 pp over STL ceiling)**; cat collapsed to 10.44. |
| F46 | 50ep + OneCycleLR pct_start=0.1 | AL only | ✅ AL done | Flat — reg 57.73, cat 43.16 (≈ B3). Short warmup doesn't unlock reg. |
| F47 | 75ep + OneCycleLR default | AL only | ✅ AL done | Flat — reg 59.88, cat 41.81 (≈ B3). 75ep not meaningfully different from 50ep. |
| **F44-F47 AZ** | **paused mid-fold-1 of F44 AZ** | **AZ INCOMPLETE** | ⚠️ — | **Need re-launch. No data persisted (--no-checkpoints).** |

## 2 · AL master table (complete)

| Config | cat F1 | reg Acc@10 | σ reg | reg-best ep |
|---|---:|---:|---:|:-:|
| B3 50ep OneCycleLR max=3e-3 (F31) | 42.71 ± 1.37 | 59.60 ± 4.09 | 4.09 | 34..46 |
| F42 150ep OneCycleLR max=3e-3 | 40.68 ± 1.15 | 56.14 ± 4.00 | 4.00 | 20..27 (warmup) |
| F44 150ep OneCycleLR max=1e-3 | 40.20 ± 1.28 | 58.82 ± 4.96 | 4.96 | 40..62 |
| **F45 150ep constant LR=3e-3** | **10.44 ± 0.04 💀** | **74.20 ± 2.95** | **2.95** | 23..52 |
| F46 50ep OneCycleLR pct_start=0.1 | 43.16 ± 1.18 | 57.73 ± 5.15 | 5.15 | 31..45 |
| F47 75ep OneCycleLR default | 41.81 ± 2.01 | 59.88 ± 3.76 | 3.76 | 14..59 |
| STL F21c (ceiling) | n/a | 68.37 ± 2.66 | 2.66 | ~34 |

## 3 · Interpretation (refined through the session)

1. **CH18 gap is NOT structural to the MTL arch.** F41 (STL with MTL pre-encoder) matches STL puro exactly; pre-encoder isn't the culprit. F45 (constant LR) reaches reg 74 — PROVES the arch+heads are capable of exceeding STL when the training regime allows.

2. **The gap is primarily a function of the OneCycleLR schedule + loss coupling.** Under OneCycleLR with annealing, reg plateaus at ~60 regardless of budget, max_lr, or warmup fraction. Only constant LR unlocks the ceiling.

3. **The regime tradeoff:** constant LR kills cat head. Cat's 7-class classifier can't handle sustained 3e-3 from ep 1; it diverges into majority-class prediction. Cat needs gentle LR / warmup; reg benefits from sustained high LR.

4. **Paper-level implication:** CH18 reframes from "methodological limitation" to "recoverable with the right schedule". Opens a new section: **hybrid regime that preserves cat AND releases reg**.

## 4 · The hypothesis to investigate next session

**Intermediate regime — preserve cat while freeing reg.** Four designs to test (ranked by simplicity):

1. **F48-H1:** constant LR = 1e-3 (not 3e-3) for 150ep. Gentler peak; cat may survive. If cat stays at ~40 and reg rises to 65+, we have the sweet spot without any new code.
2. **F48-H2:** OneCycleLR with `pct_start=0.05` + max_lr=3e-3 @ 150ep. 7-epoch warmup then long plateau-ish behaviour (the annealing tail is much gentler than constant, but extends long).
3. **F48-H3:** Scheduled handover — cat_weight ramps 0.9 → 0.25 over training. Needs new `ScheduledStaticWeightLoss` class (~50 LOC). Cat solidifies early, reg booster late.
4. **F48-H4:** Two-stage training — Phase 1 = B3 50ep; Phase 2 = continue checkpoint for 50-100 more epochs at constant LR with cat_weight=0.1. Needs checkpoint loading infra.

**Fastest signal:** H1 is a single CLI flag change. Run AL 5f × 150ep constant LR=1e-3, ~30min. If cat F1 ≥ 35 AND reg Acc@10 ≥ 65, the regime exists; then scale to AZ + design the paper narrative around it.

## 5 · Pending work

**AZ sweep incomplete.** The AZ side of F44-F47 was interrupted mid-F44 fold 1. Need to re-launch `scripts/run_f44_f47_budget_tests.sh` (it will re-run AL unnecessarily — consider splitting or adding AZ-only flag).

**Optional cleanup:** add `--skip-state alabama` to the launcher to avoid re-running AL.

**Also pending from earlier asks:**
- **F37** STL `next_gru` cat 5f per state — script ready at `scripts/run_stl_next_gru_cat.sh`, assigned to 4050 machine.
- **F43** B3 with cat_weight=0.01 (Fator 2-residual test) — `scripts/run_f43_cat_zero.sh` ready, not launched.
- **F40** scheduled handover (follow-up paper, low prio).

## 6 · Code changes landed

| File | Change |
|---|---|
| `src/training/helpers.py` | `setup_scheduler` extended: `scheduler_type ∈ {onecycle, constant, cosine}`, optional `pct_start`. |
| `src/configs/experiment.py` | `ExperimentConfig` gained `scheduler_type` (default `"onecycle"`) and `pct_start: Optional[float]`. Back-compat. |
| `src/training/runners/mtl_cv.py` | Threads `scheduler_type` + `pct_start` to `setup_scheduler` via `getattr(config, ..., default)`. |
| `scripts/train.py` | New flags `--scheduler {onecycle,constant,cosine}` + `--pct-start FLOAT`. |
| `scripts/p1_region_head_ablation.py` | New flag `--mtl-preencoder` (+ `--preenc-hidden`, `--preenc-layers`, `--preenc-dropout`); new class `_MTLPreencoder` mirroring `MTLnet._build_encoder`. |

**All code changes are back-compat** (defaults preserve legacy behaviour).

## 7 · Scripts added

| Script | Purpose |
|---|---|
| `scripts/run_stl_next_gru_cat.sh` | STL next_gru cat 5f per state (F37, 4050-assigned). |
| `scripts/run_f39_catweight_sweep.sh` | B3 cat_weight ∈ {0.25, 0.50} on AL+AZ (F39). |
| `scripts/run_f41_stl_mtl_preencoder.sh` | STL + MTL pre-encoder on AL+AZ (F41). |
| `scripts/run_f42_epoch_sweep.sh` | B3 AL 5f × 150ep (F42). |
| `scripts/run_f43_cat_zero.sh` | B3 with cat_weight=0.01 (F43, not launched). |
| `scripts/run_f44_f47_budget_tests.sh` | F44-F47 consolidated sweep (~4.3h full run). |

## 8 · Research documents added

- `docs/studies/check2hgi/research/F38_CHECKPOINT_SELECTION.md` — Fator 2 refutation.
- `docs/studies/check2hgi/research/F41_PREENCODER_FINDINGS.md` — Fator 3a refutation.

## 9 · Files to NOT commit (transient)

- `/tmp/check2hgi_logs/*.log` — background run logs, verbose.
- `results/check2hgi/*/mtlnet_lr*_2026042*/summary/full_summary.json` — per-run summaries. Live under `results/` (repo-gitignored), content may be referenced but not committed.
- Regenerated `output/check2hgi/*/input/next_region.parquet` (added `last_region_idx` column) + `output/check2hgi/*/region_transition_log.pt` — outputs, gitignored.

## 10 · How to resume

1. Read this file + `FOLLOWUPS_TRACKER.md §F44-F47` + `research/F38/F41_FINDINGS.md`.
2. Decide: do you want F44-F47 AZ (~170 min) OR jump to F48 (intermediate regime) first?
   - If AZ first: `WORKTREE=... DATA_ROOT=... OUTPUT_DIR=... PY=... bash scripts/run_f44_f47_budget_tests.sh` (note: will re-run AL; edit out lines 77-80 to skip AL)
   - If F48 first: write `scripts/run_f48_hybrid_regime.sh` using `--scheduler constant --max-lr 1e-3` as the first hypothesis test (H1).
3. **Recommended:** F48-H1 first (single flag test, 30 min AL). If it lands anywhere between B3 and F45 with cat preserved, the narrative is set and AZ F48-H1 confirms scale-stability. AZ F44-F47 becomes nice-to-have.
