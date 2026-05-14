# Reorg Handoff — 2026-05-14

**Author:** Claude (Opus 4.7, 1M)
**Triggering task:** "Merge worktree-check2hgi-mtl into main; redirect project from fusion to check2hgi; preserve all knowledge."
**Plan source-of-truth:** [`MERGE_REORG_PLAN_2026-05-14.md`](MERGE_REORG_PLAN_2026-05-14.md) (also in this folder).
**Pre-flight ref-sweep catalog:** [`REORG_REFSWEEP_CATALOG_2026-05-14.md`](REORG_REFSWEEP_CATALOG_2026-05-14.md).

---

## Outcome

Successful end-to-end execution of the planned reorg + merge. Main now reflects the 372-commit check2hgi study + the docs reorganization, with no information loss.

### What was merged

- The full check2hgi branch (`worktree-check2hgi-mtl`, 372 commits since the divergence point) merged into `main` via a single `--no-ff` merge commit (`d922488`). All commit history preserved and bisectable.
- The single asset conflict (`docs/archive/fusion-study/results/P0/folds/frozen.json`) auto-resolved to main's pre-merge version (1044 lines, 13 entries, `updated_at: 2026-04-18`) — the more-recent state with more entries; user accepted as "more entries = more historical preservation."

### What was moved

| From | To | Why |
|---|---|---|
| `docs/studies/check2hgi/{README,AGENT_CONTEXT,NORTH_STAR,CHANGELOG,CLAIMS_AND_HYPOTHESES,CONCERNS,FINAL_SURVEY,MTL_ARCHITECTURE_JOURNEY,PAPER_BASELINES_STRATEGY}.md` | `docs/<file>.md` | check2hgi is the project's primary study now |
| `docs/studies/check2hgi/results/` | `docs/results/` | Canonical numbers at `docs/` root |
| `docs/studies/check2hgi/research/*` (60+ F-trail files) | `docs/findings/` | Read-only paper-supporting findings |
| `docs/studies/check2hgi/research/{canonical_improvement,merge_design,hgi_category_injection}/` | `docs/studies/<name>/` | Active follow-up studies layered on check2hgi |
| `docs/studies/check2hgi/{baselines,paper,scope,review,launch_plans}/` | `docs/<name>/` | Promoted alongside the rest |
| `docs/studies/check2hgi/issues/` | `docs/issues/check2hgi/` | Nested to avoid clash with 7 existing generic issues |
| `docs/studies/check2hgi/archive/<subdir>/` (7 dirs) | `docs/archive/check2hgi-<subdir>/` | Each pre-existing archive subdir promoted as sibling |
| `docs/studies/fusion/` | `docs/archive/fusion-study/` | Predecessor study archived (with closure note) |
| `docs/RUNPOD_GUIDE.md` | `docs/infra/runpod/README.md` | Ops doc consolidation |
| `docs/COLAB_GUIDE.md` | `docs/infra/colab/README.md` | Ops doc consolidation |
| `scripts/H100_FLCATX_PERVISIT_PROMPT.md` | `docs/infra/h100/README.md` | Ops doc consolidation (rewritten as proper env doc + closed appendix) |
| `docs/KNOWLEDGE_BASE_2026-04-13.md` | `docs/archive/` | Date-stamped snapshot |
| `docs/plans/CHECK2HGI_MTL_{OVERVIEW,BRANCH_PLAN}.md` | `docs/archive/CHECK2HGI_MTL_{...}-pre-promotion.md` | Stale pre-promotion plans |

### What was added

- **`docs/README.md`** — navigation landing (by-question structure: "where are we now / canonical numbers / previous experiments / active follow-up studies / I'm on machine X / paper / background / archive").
- **`docs/studies/README.md`** — active studies index with semantics (`studies/` vs `findings/` vs `archive/`).
- **`docs/findings/README.md`** — F-trail index with `findings/` vs `results/` boundary rule + topic index.
- **`docs/infra/README.md`** + 6 per-env subdir READMEs (`local/`, `runpod/`, `colab/`, `lightning/`, `h100/`, `data/`) + 4 supporting docs (`runpod/scripts.md`, `colab/notebooks.md`, `colab/study_runner.md`, `data/drive_download.md`).
- **`docs/studies/hgi_category_injection/STATUS.md`** — explicit CLOSED banner (AZ falsified 2026-05-04) so future agents don't treat the study as active.
- **`docs/archive/fusion-study/ARCHIVE_NOTE.md`** — fusion study closure note: status, what survives, why archived, how to reach back into the work.
- **Closure banner in `docs/archive/fusion-study/README.md`** — preserves original entry-point text below.
- **Breadcrumbs at old paths**: 1-line stubs at `docs/RUNPOD_GUIDE.md`, `docs/COLAB_GUIDE.md`, `scripts/H100_FLCATX_PERVISIT_PROMPT.md` so external links keep working.

### What was archived

| Branch | Tag | Status |
|---|---|---|
| `check2hgi-up` (local + remote) | `archived/check2hgi-up` | Dead (PR #20 already merged via 474460d) |
| `feat/colab-gpu-perf` (local + remote) | `archived/feat/colab-gpu-perf` | Dead (content fully integrated) |
| `perf/training-optimizations` (local + remote) | `archived/perf/training-optimizations` | Dead (no unique commits to worktree) |
| `copilot/create-improvement-plan-for-mtlnet` (local + remote) | `archived/copilot/create-improvement-plan-for-mtlnet` | Dead (old copilot exploration) |
| `create-improvement-plan-for-mtlnet` (local-only duplicate) | `archived/create-improvement-plan-for-mtlnet` | Dead |
| `review/mtl-ablation-rebase` (local-only) | `archived/review/mtl-ablation-rebase` | Dead (same tip as above) |
| `h100/pervisit-fl-ca-tx-results` (remote-only) | `archived/h100/pervisit-fl-ca-tx-results` | Verified content-integrated (sha256-match on all 9 per-cell JSONs) |
| 11 already-merged-into-main remote branches | (no tag — reachable via main) | `worktree-metrics`, `add-claude-github-actions-1775967051138`, `feat/hgi-paper-alignment`, `feat/sphere2vec-paper-variant`, `perf/mtl-speed-batch1`, `copilot/research-prediction-classification-pois`, `fix/nashmtl-ecos-solver`, `chore/post-refactoring-cleanup`, `docs/refactoring-plan`, `copilot/deep-analysis-of-merge-request`, `copilot/deep-analysis-of-mr` |

Plus recovery anchor: **tag `pre-reorg-2026-05-14`** at `worktree-check2hgi-mtl` tip pre-reorg.

### What was deleted

Nothing destructive. Every "deleted" branch is reachable via its `archived/<name>` tag (or via main's history for already-merged ones). Every "moved" file kept its git history via `git mv`.

---

## Where fusion-study knowledge now lives

All preserved at [`docs/archive/fusion-study/`](fusion-study/) — full tree intact.

- **Concepts**: `AGENT_CONTEXT.md`, `MASTER_PLAN.md`, `KNOWLEDGE_SNAPSHOT.md`, `STUDY_OVERVIEW.md`, `QUICK_REFERENCE.md`, `COORDINATOR.md`, `HANDOFF.md`.
- **Claims & hypotheses**: `CLAIMS_AND_HYPOTHESES.md`.
- **Phase designs**: `phases/P0..P6/*.md` + `phases/P1_grid.yaml`.
- **Issue diagnoses**: `issues/{HGI_LEAKAGE_AUDIT, HGI_LEAKAGE_EXPLAINED, P1_METHODOLOGY_FLAWS}.md`.
- **Critical reviews**: `plans/CRITICAL_REVIEW_ATTACK_PLAN.md`.
- **Lab-notebook records**: `state.json`, `machines.yaml`, `assignments/P1_2026-04-16_screen/` (per advisor B1 — kept for provenance).
- **Pre-fusion archive**: `archive/{ablation_studies, full_ablation_study}/`.
- **Coordinator design specs**: `coordinator/{integrity_checks, state_schema}.md`.
- **Scientific record (results)**: full `results/P0/` tree (CSVs, JSONs, integrity files, leakage-ablation outputs).
- **Closure framing**: `ARCHIVE_NOTE.md` + banner at top of `README.md`.

The FUSION engine in code (`src/configs/paths.py` enum, `src/data/inputs/fusion.py`, `experiments/full_fusion_ablation.py`, `pipelines/fusion.pipe.py`) is **first-class and supported** post-archive — only the *study* is closed.

---

## New docs/ structure (rationale)

The reorg organizes docs/ around the questions an agent or new contributor asks:

| Subdir | Purpose | Reader question it answers |
|---|---|---|
| `docs/` root files | What the project IS now (check2hgi briefing, claims, north star, change log) | "Where are we now?" |
| `docs/results/` | Canonical paper-facing numbers + raw run artefacts by phase | "What are the canonical numbers?" |
| `docs/findings/` | Per-experiment findings (the F-trail) — closed read-only history | "What previous experiments led us here?" |
| `docs/studies/` | Active follow-up studies layered on check2hgi (each its own track) | "What's still being worked on?" |
| `docs/archive/` | Closed studies / superseded snapshots (fusion-study, paper-closure, pre-b3, etc.) | "Where's the historical knowledge?" |
| `docs/infra/` | Operational docs (RunPod, Colab, Lightning, H100, local, Drive) | "I'm on machine X — where do I look?" |
| `docs/baselines/`, `docs/paper/`, `docs/scope/`, `docs/review/`, `docs/launch_plans/` | Study-supporting infra (baselines, paper drafts, scoping decisions, dated reviews, historical launch plans) | "What did we decide, and why?" |
| `docs/context/`, `docs/datasets/`, `docs/thesis/`, `docs/plans/`, `docs/reports/` | Project-wide background | "Background reading" |
| `docs/issues/` | Generic issues + check2hgi-nested subdir | "Open / known issues" |

`docs/studies/` is now an empty-after-promotion container — currently holds only the 3 carved-out follow-up tracks (`canonical_improvement/`, `merge_design/`, `hgi_category_injection/` — the last is CLOSED) plus a `README.md`. It's ready for any future study layered on check2hgi.

---

## Where ops/infra docs now live

Single home: [`docs/infra/`](../infra/).

```
docs/infra/
├── README.md                       ← landing: "I'm on machine X — where do I look?"
├── local/README.md                 ← M4 Pro / MPS / .venv
├── runpod/
│   ├── README.md                   ← canonical RunPod setup + train recipe
│   └── scripts.md                  ← index of scripts/runpod_*.sh
├── colab/
│   ├── README.md                   ← canonical Colab T4 guide (detached-subprocess pattern)
│   ├── notebooks.md                ← index of notebooks/colab_*.ipynb
│   └── study_runner.md             ← scripts/study/colab_runner.py usage
├── lightning/README.md             ← Lightning A100/H100 pods, wall-clock refs
├── h100/README.md                  ← H100 SSH bare-metal + closed-prompt appendix
└── data/drive_download.md          ← gdown patterns + Drive layout convention
```

**Experiment-recipe scripts** (`scripts/run_b3_*.sh`, `run_b5_*.sh`, `run_b9_*.sh`, `run_f21c_*`, `run_f27_*`, `run_f37_*`, `run_f40_*`, `run_f48_*`, etc.) deliberately stay in `scripts/` and are NOT indexed under `docs/infra/` — they're recipes, not ops. Per advisor review.

CLAUDE.md has a single-line pointer to `docs/infra/`; the inline Colab section was collapsed to 3 lines.

---

## CLAUDE.md changes applied

- **Project focus** section near top names check2hgi as primary study, points to `docs/` root + `articles/[BRACIS]_Beyond_Cross_Task/`. Calls out fusion as first-class engine despite study archive.
- **File Architecture** docs/ subtree updated to the new layout (results, findings, studies, infra, archive, baselines, paper, context, datasets, thesis, plans, reports, scope, review, launch_plans, issues).
- **Running on remote/cloud machines** section replaces the inline Colab content with a 3-line pointer to `docs/infra/`.
- **Branch-scoped study context** updated — primary study now visible at `docs/` root by default; CLAUDE.local.md only needed for follow-up studies.
- **What changed (2026-05-14 reorg)** appendix at end summarizes the changes + points to `MERGE_REORG_PLAN_2026-05-14.md`.

---

## Repo-wide ref-sweep applied

Updated path references in:

- `CLAUDE.md`
- `articles/[BRACIS]_Beyond_Cross_Task/` (8 files: `AGENT.md`, `COORDINATOR_HANDOFF.md`, `AUDIT_LOG.md`, `STATISTICAL_AUDIT.md`, `TABLES_FIGURES.md`, `PAPER_STRUCTURE.md`, `PAPER_DRAFT.md`, `src/figs/render_per_visit*.py`)
- `tests/test_regression/` (3 files)
- `experiments/check2hgi_up/` (6 files)
- `research/baselines/` (7 files)
- `docs/baselines/POI_RGNN_AUDIT.md`, `docs/CHANGELOG.md`, `docs/AGENT_CONTEXT.md`, `docs/infra/runpod/README.md`
- `docs/studies/canonical_improvement/AGENT_PROMPT.md` (10 path refs + branching point: `check2hgi-up` → `main`)
- `scripts/run_f27_cathead_sweep.sh`
- `.claude/commands/{study, coordinator}.md`
- `scripts/study/*.py` (4 files: `enroll_p1.py`, `archive_result.py`, `_state.py`, `_backfill_joint_taskbest.py`)
- `experiments/hgi_leakage_ablation.py`
- `docs/reports/{report_v1_20260415, README}.md`

Dated `review/`, `scope/`, and archive memos retain old paths intentionally — they're timestamped historical snapshots.

---

## Verification log

- ✅ Pre-flight tag `pre-reorg-2026-05-14` at worktree tip
- ✅ h100/pervisit-fl-ca-tx-results: all 9 per-cell JSONs verified content-identical (sha256-match) before branch deletion
- ✅ Pre-flight ref-sweep grep complete (catalog persisted)
- ✅ Each reorg step committed separately (5 commits on worktree branch + 1 merge commit on main = bisectable)
- ✅ pytest sanity: 166 tests pass on `tests/test_regression/ tests/test_data/ tests/test_models/test_mtlnet.py`
- ✅ All sanity-checked navigation paths resolve (docs/README.md, docs/AGENT_CONTEXT.md, docs/NORTH_STAR.md, docs/CHANGELOG.md, docs/CLAIMS_AND_HYPOTHESES.md, docs/results/RESULTS_TABLE.md, docs/findings/README.md, docs/studies/README.md, docs/infra/README.md, docs/archive/fusion-study/ARCHIVE_NOTE.md, docs/studies/hgi_category_injection/STATUS.md, docs/studies/canonical_improvement/AGENT_PROMPT.md)
- ✅ AGENT_PROMPT.md branching instruction now reads `from main` (was `from check2hgi-up`)
- ✅ Final branch state: `main`, `worktree-check2hgi-mtl`, `bracis` — local + remote each
- ✅ All deleted branches preserved as `archived/<name>` tags (pushed to origin)

---

## Open follow-ups / known risks

### Cleanup leftovers

The classifier blocked `rm -rf` on three leftover worktree directories that have no git metadata anymore. **You can clean these manually:**
```bash
rm -rf /Users/vitor/Desktop/mestrado/ingred/.claude/worktrees/feat-colab-gpu-perf
rm -rf /Users/vitor/Desktop/mestrado/ingred-perf-opt
rm -rf /Users/vitor/Desktop/mestrado/ingred/.claude/worktrees/mtlnet-improve
```

These are safe (`git worktree prune` already removed git's metadata pointer; the dirs are just stale checkout files). Skipping leaves them as orphan dirs — annoying but not breaking.

### CLAUDE.local.md (gitignored, per-checkout)

The user's CLAUDE.local.md at `/Users/vitor/Desktop/mestrado/ingred/CLAUDE.local.md` (main worktree) currently points to the fusion study. Post-merge, the primary check2hgi study is visible at `docs/` root by default, so the CLAUDE.local.md is no longer needed for routing to check2hgi. Update or delete at the user's discretion. Suggested replacements per active branch:

- **Generic main work**: delete the file entirely.
- **Follow-up study work**: replace with a 3-line pointer to `docs/studies/<name>/AGENT_PROMPT.md`.

### frozen.json resolution

Per merge result, `docs/archive/fusion-study/results/P0/folds/frozen.json` ended up with main's pre-merge content (1044 lines, 13 entries, `updated_at: 2026-04-18`). User accepted this as "more entries = more historical preservation for an archived file." If you ever need to use this file at runtime (e.g., re-running fusion experiments), verify the entries still correspond to real fold artifacts.

### Dated review/scope memos still reference old paths

Files in `docs/scope/`, `docs/review/`, and `docs/findings/archive/F50/F50_HANDOFF_2026-04-28.md` retain old `docs/studies/check2hgi/...` references because they're timestamped historical snapshots — updating them would rewrite history. Future agents reading them should know these are dated artifacts (the date in the filename is a giveaway).

### Notebooks

`notebooks/colab_check2hgi_mtl.ipynb` and `notebooks/colab_phase2_grid.ipynb` reference `docs/RUNPOD_GUIDE.md` / `docs/COLAB_GUIDE.md` — those are now 1-line breadcrumbs pointing to `docs/infra/`, so the references still resolve. If you want the notebooks to reference the new paths directly, update the markdown cells at your convenience (low priority — breadcrumbs work).

### `bracis` branch retained

The `bracis` branch (+92/+4 vs worktree-check2hgi-mtl) is kept active for the BRACIS 2026 anonymous review submission. The 4 unique commits are anonymization scrubs (drop residual identifier-leaking files, document Gowalla ETL, anonymized base). Once the review is over, you can decide whether to delete or archive.

---

## What I deliberately did NOT do

- Did not move `articles/[BRACIS]_Beyond_Cross_Task/` — stays as a sibling of `docs/`.
- Did not edit `src/` — no source changes, only docs/branch reorg.
- Did not rerun any experiments or regenerate results.
- Did not touch `data/`, `output/`, `results/` (gitignored runtime artefacts).
- Did not delete the `bracis` branch (kept active for BRACIS review).
- Did not force-push main — used a clean merge commit.
- Did not strip fusion code support — fusion remains a first-class engine.

---

## Final state at a glance

- **Branches**: `main` (= worktree-check2hgi-mtl content + reorg), `worktree-check2hgi-mtl` (will become stale; keep or delete at your discretion), `bracis` (active for review).
- **Tags**: `pre-reorg-2026-05-14` (recovery anchor) + 7 `archived/<name>` tags (one per non-trivially-merged deleted branch).
- **docs/ tree**: question-driven structure with check2hgi at root, F-trail at `findings/`, active follow-ups at `studies/`, fusion at `archive/`, ops at `infra/`.
- **CLAUDE.md**: updated with project focus, new file architecture, ops pointer, branch-scoped study context, reorg appendix.

The merge is reviewable as a single commit (`d922488`) with bisectable per-step history (`e0b3e80` → `040aed9`). Recovery is one `git reset --hard pre-reorg-2026-05-14` away if anything looks wrong.

---

## Round-2 follow-up (same day, 2026-05-14 PM)

After the round-1 reorg landed, a second pass tightened the docs further. Three goals:

1. **Trim stale folders** that the round-1 promotion left at `docs/` root.
2. **Add `docs/index.html`** as a comprehensive single-page research-state summary.
3. **Enrich `docs/context/`** so the project-wide reference library covers everything a new agent needs.

### Round-2 changes

| Action | Where | What |
|---|---|---|
| User-removed | `/Users/vitor/.../ingred/CLAUDE.local.md` | Per-checkout pointer to fusion (gitignored) — no longer needed since check2hgi is at `docs/` root |
| Archive | `docs/launch_plans/` → `docs/archive/check2hgi-launch-plans-2026-04/` | 2 staging plans for completed F33/F36/CA+TX work |
| Archive | `docs/plans/` → `docs/archive/check2hgi-fusion-era-ablation-plans/` | 2 fusion-era ablation plans (DSelectK + Aligned-MTL + Fusion language) superseded by canonical_improvement |
| Archive | `docs/paper/` → `docs/archive/check2hgi-paper-drafts-pre-bracis/` | 4 pre-LaTeX section drafts; work landed in `articles/[BRACIS]_*/src/sections/` |
| Archive | `docs/review/` → `docs/archive/check2hgi-reviews-2026-04/` | 7 dated critical-review snapshots |
| Archive | `docs/scope/{ch14, ch15}/` → `docs/archive/check2hgi-scope-memos-2026-04/` | Decision memos absorbed into CLAIMS catalog |
| Archive | `docs/issues/{8 generic files}` → `docs/archive/check2hgi-generic-issues-pre-promotion/` | Proposals executed / leakage diagnoses fully covered by newer audits / etc. |
| New folder | `docs/future_works/` | Forward-looking memos for work-to-do-later (not yet a study). Initial inhabitant: `task_pivot_memo.md` (moved from `docs/scope/`) — needs eventual merge into NORTH_STAR but deferred. |
| Move | `docs/check2hgi_overview.tex` → `docs/context/check2hgi_overview.tex` | TikZ architecture figure now lives with the project context library |
| Consolidate | `docs/datasets/DATASETS.md` → `docs/context/DATASETS.md` (folder removed) | One fewer top-level folder; dataset reference becomes part of context |
| New | `docs/context/METRICS.md` | F1/Acc@10/MRR/Δm + paired-Wilcoxon (n=20 paper-grade vs n=5 screening) + F51 canonical extraction (per-fold max ep≥5) |
| New | `docs/context/DATA_SPLITS.md` | Current fold protocol (StratifiedGroupKFold, MTL fold pairing, per-fold log_T leak prevention) + pointer to `src/data/folds.py` |
| Updated | `docs/context/README.md` | Index now includes 4 new files (DATASETS, DATA_SPLITS, METRICS, check2hgi_overview.tex) |
| Updated | `docs/context/TASK_HEADS.md` | Added naming-convention banner explaining `next_*` heads are target-agnostic (sequence input shape, not prediction target) |
| New | `docs/index.html` | Comprehensive research-state summary page: thesis, headline numbers (5-state architectural-Δ, per-visit decomposition, Δm scoreboard, recipe selection), mechanism findings, architecture recipe, paper pointers, active studies, future works, navigation by machine, archive index |

### Round-2 ref-sweep updates

To keep docs internally consistent after the moves:

- `docs/NORTH_STAR.md`, `docs/CLAIMS_AND_HYPOTHESES.md`, `docs/findings/B5_FL_TASKWEIGHT.md`, `docs/findings/MTL_FLAWS_AND_FIXES.md` — `review/2026-04-2*` paths updated to `archive/check2hgi-reviews-2026-04/`
- `docs/PAPER_BASELINES_STRATEGY.md` + 4 finding docs (F50_T3_TRAINING_DYNAMICS, F37_FL_RESULTS, F50_DELTA_M_FINDINGS_LEAKFREE, MTL_FLAWS_AND_FIXES) — cleaned stale "TBD update paper/X.md" pointers (work landed in `articles/[BRACIS]_*/`)
- `docs/baselines/POI_RGNN_AUDIT.md`, `scripts/run_f27_cathead_sweep.sh`, `experiments/check2hgi_up/run_mtl_b3.py` — old `docs/studies/check2hgi/` paths updated to current locations
- `docs/AGENT_CONTEXT.md`, `docs/README.md`, `CLAUDE.md` — file architecture reflects the new structure

### Round-2 final state

- **Branches/tags**: unchanged from round 1 (`main`, `worktree-check2hgi-mtl`, `bracis`; recovery anchor `pre-reorg-2026-05-14`).
- **`docs/` top-level structure** (post round 2):
  ```
  docs/
  ├── README.md, AGENT_CONTEXT.md, NORTH_STAR.md, CHANGELOG.md, CLAIMS_AND_HYPOTHESES.md,
  │   CONCERNS.md, FINAL_SURVEY.md, MTL_ARCHITECTURE_JOURNEY.md, PAPER_BASELINES_STRATEGY.md
  │   (9 study-defining root docs)
  ├── BRACIS_GUIDE.md, PAPER_FINDINGS.md (legacy/guidance)
  ├── index.html (comprehensive summary page)
  ├── results/        — canonical numbers + raw artefacts
  ├── findings/       — paper-supporting F-trail (66 entries)
  ├── studies/        — 3 active follow-up studies (canonical_improvement, merge_design, hgi_category_injection)
  ├── future_works/   — forward-looking memos (task_pivot_memo)
  ├── infra/          — operational docs (6 environments)
  ├── baselines/      — external baselines
  ├── context/        — project-wide reference library (10 files including new METRICS, DATA_SPLITS, DATASETS, check2hgi_overview.tex)
  ├── archive/        — closed studies, paper drafts, plans, reviews, scope memos, snapshots
  ├── issues/check2hgi/ — bug-audit log (8 files)
  ├── thesis/, reports/ — kept as-is
  ```

The structure is now genuinely navigable: a new agent landing on the repo can answer "where are we now / what are the canonical numbers / what's being worked on / what's deferred / where do I run things / where's the paper / where's the history" without spelunking — that was the round-2 goal.

Round-2 commit: `69c9854 docs(reorg): round-2 cleanup — archive stale folders, enrich context/, add index.html` (cherry-picked to main as `97cd7fd`).

### Round-2.5 final cleanup (same day, late PM)

After round-2 landed, a third cleanup pass addressed `plans/` + `logs/` evaluation per user request, plus three advisor follow-ups.

| Action | Where | What |
|---|---|---|
| Archive | `plans/` (repo root, 5 stale paper-prep docs from Apr 2026) → `docs/archive/repo-root-plans-pre-bracis/` | All 5 stale: `mtl_regression_floor_investigation.md`, `mtlnet_speed_optimization.md` (work landed via feat/colab-gpu-perf), `plan.md` (BRACIS master plan, paper submitted), `todo.md`, `todo_article.md` (all paper-closure items done). Empty `plans/` folder removed. |
| Untrack | `logs/` (45 tracked training-stdout files, ~7 MB) | `git rm -r --cached logs/`. Local copies retained on disk. `/logs/` was already in `.gitignore` but files were tracked from before the entry; this honors the existing intent. |
| Polish | `docs/index.html` §0.1 table caption | Added explicit metric spec ("Reg metric: top10_acc_indist; Cat metric: Macro-F1; n=20 = 4 seeds × 5 folds") |
| Polish | `docs/index.html` background-reading list | Split context/ link into per-file sub-bullets (DATASETS / METRICS / DATA_SPLITS / etc. now 1-click reachable) |
| Polish | `docs/context/TASK_HEADS.md` | Added naming-convention banner explaining `next_*` heads are *target-agnostic* (sequence input shape, not predicted target) — clarifies "why is the cat head called `next_gru`?" confusion |
| Polish | `articles/[BRACIS]_Beyond_Cross_Task/AGENT.md` | Disambiguated `AGENT_CONTEXT.md` reference to `docs/AGENT_CONTEXT.md` (was bare path, ambiguous within the article folder) |

Round-2.5 commit: `7b62b29 docs(reorg): round-2 final cleanup — archive repo-root plans/, untrack logs/` (cherry-picked to main as `1606a1e`).

### Final state (post round 2.5)

- **Branches/tags**: unchanged from round 1.
- **Repo root**: `plans/` removed; `logs/` untracked (local copies retained).
- **`docs/` top-level**: 9 study-defining root docs + `index.html` (NEW) + `BRACIS_GUIDE.md` + `PAPER_FINDINGS.md` + 12 subdirs (`results/`, `findings/`, `studies/`, `future_works/`, `infra/`, `baselines/`, `context/`, `archive/`, `issues/check2hgi/`, `thesis/`, `reports/`).
- **No stale folder references**: ref-sweep grep returns 0 in active docs (excluding archive/ and findings/archive/ self-refs).
- **All 3 advisor passes** (round-1 pre-merge, round-2 post-execution, round-2.5 final audit) addressed.

The reorg is closed.

