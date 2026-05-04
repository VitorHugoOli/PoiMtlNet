# Session Handoff — 2026-04-24 (post-F27)

**For the next agent that picks up this study.** Start here.

This handoff supersedes `SESSION_HANDOFF_2026-04-22.md`. Read it after § 0 if you want the B5-era operational gotchas (they all still apply).

## 0 · One-minute summary

- **Champion (committed 2026-04-24):** `mtlnet_crossattn + static_weight(category_weight=0.75) + next_gru` (task_a) + `next_getnext_hard` (task_b), d=256, 8h. See `NORTH_STAR.md`.
- **Three paper-reshaping findings landed since 2026-04-22:**
  1. **F2** (2026-04-23) — B3 Pareto-dominates the prior `GETNext-soft` north-star on FL at n=1. Diagnosed mechanism: PCGrad × hard-prior × FL-scale gradient starvation of the cat head; static_weight with cat-heavy 0.75 weighting triggers a late-stage handover that rescues cat.
  2. **F21c** (2026-04-24) — **Matched-head STL beats MTL-B3 on region by 12–14 pp Acc@10 on AL + AZ.** STL `next_getnext_hard` alone reaches 68.37 (AL) / 66.74 (AZ) vs MTL-B3 at 56.33 / 52.76. The MTL coupling does not add value on region at ablation-state scale. Paper reframing needed.
  3. **F27** (2026-04-24) — Swapping task_a head from `NextHeadMTL` (Transformer) → `next_gru` beats the default on AL (+3.43 pp cat F1 at 5f) and AZ (+2.37 pp at 5f, Wilcoxon p=0.0312 on cat F1, cat Acc@1, reg MRR; 5/5 folds positive). **But FL at 1f flipped sign** (−0.93 pp cat F1) — scale-dependence flag open.
- **Open blocker:** resolve the F27 FL scale-dependence via a **Colab run** (`notebooks/colab_f27_validation.ipynb`): FL 5f + CA 1f + TX 1f with B3+next_gru. ~10–18 h on T4. User will run this + paste results back.

## 1 · Current claim status (all metrics)

Cross-state headline table under the committed B3 (`cross-attn + static cat=0.75 + next_gru + next_getnext_hard`):

| State | Protocol | cat F1 | cat Acc@1 | reg Acc@10_indist | reg MRR_indist | JSON |
|---|---|---:|---:|---:|---:|---|
| AL | 5f × 50ep | 0.4271 ± 0.0137 | 0.4582 ± 0.0151 | 0.5960 ± 0.0409 | 0.3074 ± 0.0287 | `results/F27_validation/al_5f50ep_b3_cathead_gru.json` |
| AZ | 5f × 50ep (Wilcoxon p=0.0312) | **0.4581 ± 0.0130** | **0.4930 ± 0.0067** | 0.5382 ± 0.0311 | 0.2766 ± 0.0241 | `results/F27_validation/az_5f50ep_b3_cathead_gru.json` |
| FL | 1f × 50ep (n=1) | 0.6572 | 0.6860 | 0.6526 | 0.2956 | `results/F27_validation/fl_1f50ep_b3_cathead_gru.json` |
| CA | — | 🔴 pending | 🔴 pending | 🔴 pending | 🔴 pending | Colab F34 |
| TX | — | 🔴 pending | 🔴 pending | 🔴 pending | 🔴 pending | Colab F35 |

Matched-head STL baselines (F21c, 5-fold):

| State | STL `next_getnext_hard` Acc@10 | STL STAN Acc@10 | MTL-B3 Acc@10 | MTL vs STL hard |
|---|---:|---:|---:|---:|
| AL | **68.37 ± 2.66** | 59.20 ± 3.62 | 59.60 ± 4.09 | **−8.77 pp** |
| AZ | **66.74 ± 2.11** | 52.24 ± 2.38 | 53.82 ± 3.11 | **−12.92 pp** |
| FL | 🔴 not run | 🔴 not run | see above | — |

STL matched-head cat (current paper baseline): `NextHeadSingle` (Transformer+attention-pool), single-task `--task next`.

| State | STL cat F1 (`next_single`) |
|---|---:|
| AL | 0.3858 ± 0.0123 |
| AZ | 0.4208 ± 0.0089 |
| FL | 0.6317 (n=1) |

## 2 · What happened since 2026-04-22

Chronological, with JSONs / write-ups.

### 2026-04-22 → 04-23 session
- **F2 diagnostic** — 4-phase FL-hard probe (pcgrad baseline + α inspection, then static_weight cat ∈ {0.25, 0.5, 0.75}). Found B3 = static cat=0.75 Pareto-dominates the prior soft-probe north-star on FL at n=1. Mechanism: cat-heavy weighting triggers late-stage handover (cat best-epoch 42 vs 10 under pcgrad/equal). `research/B5_FL_TASKWEIGHT.md`. JSONs: `results/F2_fl_diagnostic/*.json`.
- **F18/F19 multi-state validation** — B3 at 5-fold on AL + AZ matches or beats prior configs.
- **F19-Wilcoxon** — B3 beats STL Check2HGI cat on AZ by +1.65 pp F1, p=0.0312 (every fold positive). `research/B3_AZ_WILCOXON_VS_STL.md`. Script: `scripts/analysis/az_b3_wilcoxon_vs_stl.py`.
- **F20 infra** — per-fold persistence in `MLHistory` so long CV runs don't lose everything on SIGKILL. 4 regression tests. `src/tracking/storage.py::save_fold_partial` + hook in `src/tracking/experiment.py::step`. Tests: `tests/test_tracking/test_fold_partial_persist.py`.
- **F17 FL 5f clean re-run** — launched, user killed (SIGTERM 143) after fold 2. Fold 1 persisted via F20 — confirmed the infrastructure works. F17 not re-launched; the FL 1-fold ablation data we have (F2 Phase B3 + F17 fold 1) is sufficient for the ablation section. FL 5-fold headline is now the Colab job.

### 2026-04-23 → 04-24 session
- **Doc cleanup** — archived pre-B3 docs under `archive/pre_b3_framing/`, `archive/research_pre_b3/`, `archive/phases_original/`. Removed the P0–P7 phase plans (they were the original roadmap, now subsumed by `NORTH_STAR.md` + `PAPER_STRUCTURE.md`). Active doc set is 9 top-level md files.
- **New** `PAPER_STRUCTURE.md` — single source of truth for paper scope, baselines, STL-matching policy, FL region Markov caveat.
- **F21c dev + runs** — extended `scripts/p1_region_head_ablation.py` to support `next_getnext_hard` (adds 3-tuple `POIDatasetWithAux` + `AuxPublishingLoader` wiring, `last_region_idx` plumbing, `transition_path` via overrides). Ran AL + AZ 5f. `research/F21C_FINDINGS.md`. JSONs: `results/B3_baselines/stl_getnext_hard_{al,az}_5f50ep.json`.
- **F27 dev** — added `--cat-head` + `--cat-head-param` CLI flags to `scripts/train.py`; extended `resolve_task_set` in `src/tasks/presets.py` with `task_a_head_factory`.
- **F27 ablation** — 7-config cat-head sweep on AZ 1f × 50ep. `next_gru` won: cat F1 +2.69, cat Acc@1 +5.24 vs default. `research/F27_CATHEAD_FINDINGS.md`. JSONs: `results/F27_cathead_sweep/*.json`.
- **F27 validation at 5-fold AZ** — confirmed the 1-fold finding with Wilcoxon p=0.0312 on cat F1 / cat Acc@1 / reg MRR. Script: `scripts/analysis/az_b3_cathead_wilcoxon.py`. JSON: `results/F27_validation/az_5f50ep_b3_cathead_gru.json`.
- **F31 AL 5f** — B3+next_gru on AL passed with bigger gains than AZ: +3.43 pp cat F1, +4.72 pp cat Acc@1, +3.27 pp reg Acc@10. MTL reg Acc@10 = 59.60 first-time-ever crosses STL STAN's 59.20 on AL. JSON: `results/F27_validation/al_5f50ep_b3_cathead_gru.json`.
- **F32 FL 1f** — B3+next_gru on FL flipped sign: −0.93 pp cat F1 vs pre-F27 mean of 0.6665. Scale-dependence flag raised. Needs 5f to resolve. JSON: `results/F27_validation/fl_1f50ep_b3_cathead_gru.json`.
- **F33/F34/F35 + Colab notebook** — tracker items and `notebooks/colab_f27_validation.ipynb` ready for user to launch.

## 3 · Active tracker items (what the next agent should know)

From `FOLLOWUPS_TRACKER.md`:

### In progress / awaiting user
- **F33** (P1) — FL 5f × 50ep B3+next_gru on Colab. **This is the binding test** for Path A (universal next_gru) vs Path B (scale-dependent cat head).
- **F34** (P1) — CA upstream pipeline + 1f × 50ep B3+next_gru on Colab.
- **F35** (P1) — TX upstream pipeline + 1f × 50ep B3+next_gru on Colab.

### Path decision (pending F33 results)
- **Path A** — Commit `next_gru` universally. Simpler narrative.
- **Path B** — Scale-dependent: `next_gru` for AL/AZ, `next_mtl` for FL/CA/TX.
- **Path C** — Already in progress via F33 (decisive FL 5-fold).

### P2 once CA/TX 1-fold lands
- F24 / F25 — CA/TX 5-fold headline runs (launched after F34/F35 confirm the config works on CA/TX data).
- F3 — AZ HGI STL cat (Objective 1 extension to n=2 states). Still pending, cheap.
- F9 — FL HGI STL cat. Pending.

### Deferred / follow-up paper
- F8 — multi-seed n=3 on headline champions (held until champion frozen).
- F27b — (archived) STL `next_getnext` soft as an additional ablation.
- F21a — (dropped by user) FL STL STAN 5f.
- F27 — cat-head exploration on MTL (done on AZ; AL/FL still could be swept at 1f if reviewer asks).

## 4 · How to pick up the work cold

1. **Read docs in this order** (all at `docs/studies/check2hgi/`):
   - `README.md` — entry point, navigation.
   - `NORTH_STAR.md` — the committed champion config + F27 scale-dependence flag.
   - `PAPER_STRUCTURE.md` — paper scope, baselines, FL region Markov caveat.
   - This file (`SESSION_HANDOFF_2026-04-24.md`) — what you just read.
   - `FOLLOWUPS_TRACKER.md` — live work queue.
   - `research/F21C_FINDINGS.md` — paper-reshaping finding.
   - `research/F27_CATHEAD_FINDINGS.md` — cat-head ablation + scale-dependence details.
   - `research/B5_FL_TASKWEIGHT.md` — F2 diagnostic.
   - `results/RESULTS_TABLE.md` — all numbers by state.

2. **Check what the user has run on Colab** (if anything):
   ```bash
   ls docs/studies/check2hgi/results/headline/ 2>/dev/null
   ls docs/studies/check2hgi/results/F27_validation/fl_5f*   # if F33 ran
   ```
   If empty, F33–F35 haven't been launched yet (or results haven't been synced back).

3. **When F33 results land** (user will paste the §⑥ summary-table output from the notebook):
   - If FL cat F1 at 5f lands within σ of the pre-F27 mean [0.6623, 0.6706] → **Path A**: commit `next_gru` universally. Update `NORTH_STAR.md` to drop the scale-dependence flag.
   - If FL cat F1 at 5f is clearly below [σ-envelope] → **Path B**: keep `next_gru` for AL/AZ, set `task_a_head_factory = None` (or explicit `next_mtl`) for FL/CA/TX in the preset. Update `PAPER_STRUCTURE.md` to document the scale-dependent champion.
   - In either case: update `OBJECTIVES_STATUS_TABLE.md` with the FL 5-fold row.

4. **Paper-reframing decision** (pending F33):
   - With F21c's matched-head STL dominating MTL on region, the paper's "MTL contribution" claim needs to shift. The three framings in `research/F21C_FINDINGS.md §Interpretation`:
     - **A** — Multi-task deployment motivation (single-model joint prediction vs two-model).
     - **B** — Matched-head ablation as the paper's main finding (honest & interesting).
     - **C** — Shift paper to graph-prior focus (STL GETNext-hard is the new SOTA).
   - Pick with the user after F33 lands.

## 5 · Operational gotchas (carried forward from 2026-04-22)

Everything from `SESSION_HANDOFF_2026-04-22.md §Read this before touching anything` still applies:

- **G1** — `--reg-head` override must apply BEFORE `FoldCreator` is constructed so the aux-side-channel branch for `next_getnext_hard` activates. Landed in `ea65fb3`.
- **G2** — PCGrad requires every `task_specific_parameters()` tensor to be in the graph. Pattern: any `nn.Parameter` on a head must be touched on every forward pass. `next_getnext_hard` fallback multiplies `alpha * 0.0` to keep it in graph.
- **G3** — `num_workers=0` is load-bearing for the B5 aux side-channel. See `src/data/aux_side_channel.py` caveats.
- **G4** — MPS sleep-induced SIGBUS on long runs. Wrap any >45 min run under `caffeinate -s`.
- **G5** — AZ fold-3 intermittent slowdown (macOS swap pressure). Kill Spotlight / idle apps on long AZ+FL runs.
- **G6** — `next_region.parquet` schema version (`last_region_idx` column) — regenerate via `scripts/regenerate_next_region.py --state X` if missing.
- **G7** — Old α-inspection checkpoints (`mtl__check2hgi_next_region_20260421_19****`) are 2f × 50ep with `--checkpoints`; champion runs are 5f × 50ep `--no-checkpoints`. Don't confuse them.

New gotcha from this session:

- **G8** — F20 per-fold persistence: fold N's artefacts only persist at end-of-fold-N (inside `history.step()`). If a process is SIGKILLed *mid-fold-N*, fold N's data is lost but folds 0..N-1 survive. This is the intended best-effort guarantee.

## 6 · Commits map (end of 2026-04-24 session)

Push is staged for origin/worktree-check2hgi-mtl. Commits in chronological order:

| Commit | Scope |
|---|---|
| `feat(tracking): per-fold partial persistence in MLHistory (F20)` | src/tracking/storage.py + experiment.py + tests |
| `feat(pipeline): extend p1 ablation for next_getnext_hard; --cat-head CLI; task_a_head_factory` | p1 script + train.py + presets.py |
| `study(b3): F2 FL-hard diagnostic + F21c matched-head STL + F27 cat-head ablation` | results JSONs + research docs |
| `study(validation): F31 AL + F32 FL 1f + F19-follow-up + F27 Wilcoxon` | validation JSONs + analysis scripts |
| `docs(study): consolidate post-B3 — archive pre-B3, new PAPER_STRUCTURE, tracker/north-star/results` | doc cleanup + new scaffolding |
| `tooling(colab): F27 scale-dependence validation notebook (FL 5f + CA/TX 1f)` | notebooks/colab_f27_validation.ipynb |
| `docs(study): session handoff for next agent (2026-04-24)` | this file |

## 7 · Don't

- **Don't launch another long FL 5-fold on M4 Pro.** F17 was user-killed; Colab (F33) is the committed path. FL 5-fold on M4 has a SIGKILL/OOM history.
- **Don't re-run F21b (STL GETNext-soft) or F21a (FL STL STAN 5f)** — user explicitly dropped / archived these.
- **Don't commit `next_gru` universally on FL** until F33 5-fold result is in hand. The F32 1-fold flipped sign; the scale-dependence flag is real until disproven.
- **Don't re-write the F21c interpretation to favour MTL.** The matched-head STL gap is real. The paper has to honestly report it.
- **Don't push to main.** All work is on `worktree-check2hgi-mtl`.

## 8 · Files most worth reading (ranked)

1. `NORTH_STAR.md` — the committed config + scale-dependence flag.
2. `PAPER_STRUCTURE.md` — scope, baselines, STL-matching policy.
3. `FOLLOWUPS_TRACKER.md` — live queue.
4. `research/F21C_FINDINGS.md` — paper-reshaping matched-head finding.
5. `research/F27_CATHEAD_FINDINGS.md` — cat-head ablation + scale-dependence.
6. `research/B5_FL_TASKWEIGHT.md` — F2 mechanism (PCGrad gradient-starvation + late-stage handover).
7. `research/B5_AZ_WILCOXON.md` — prior AZ hard-vs-soft paired test.
8. `research/B3_AZ_WILCOXON_VS_STL.md` — MTL-over-STL significance.
9. `results/RESULTS_TABLE.md` — per-state × per-method canonical table.
10. `review/2026-04-23_critical_review.md` — analytical state of the study as of 2026-04-23.
