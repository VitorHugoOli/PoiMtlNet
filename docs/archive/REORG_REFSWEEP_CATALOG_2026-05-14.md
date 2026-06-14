# Reorg ref-sweep catalog — Phase 3-0 pre-flight

Pre-flight grep results for the 2026-05-14 reorg. Use this list to update references after each `git mv`.

## Refs to `docs/studies/check2hgi/` (rewrite after promotion)

### High-impact (will break paper / tests / CLAUDE.md)
- `CLAUDE.md`
- `articles/[BRACIS]_Beyond_Cross_Task/` — 11 files: `AGENT.md`, `COORDINATOR_HANDOFF.md`, `AUDIT_LOG.md`, `STATISTICAL_AUDIT.md`, `TABLES_FIGURES.md`, `PAPER_STRUCTURE.md`, `PAPER_DRAFT.md`, `src/figs/render_per_visit.py`, `src/figs/render_per_visit_grouped.py`, `src/figs/render_per_visit_5state.py`, plus possibly more in `src/sections/*.tex` (re-grep in 3-3)
- `tests/test_regression/` — 3 files: `test_mtlnet_crossattn_lambda0_gradflow.py`, `test_mtl_param_partition.py`, `test_mtlnet_crossattn_partial_forward.py`
- `experiments/check2hgi_up/` — 6 files: `run_stl_eval.py`, `run_all.py`, `aggregate.py`, `run_mtl_b3.py`, `eval_anchors.py`, `run_variant.py`
- `research/baselines/` — 7 files: `README.md`, `mha_pe/etl.py`, `mha_pe/train.py`, `rehdm/train.py`, `rehdm/train_stl.py`, `rehdm/train_stl_study.py`, `rehdm/README.md`, `stan/README.md`, `stan/train.py`, `poi_rgnn/train.py`

### Internal to docs/studies/check2hgi/ itself (move with the file, then sed-update)
The promotion `git mv` preserves the file but the internal refs become wrong (because the file moved out from under its own path). Each of the following files needs internal refs updated post-move:

- `docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md`
- `docs/studies/check2hgi/CHANGELOG.md`
- `docs/studies/check2hgi/AGENT_CONTEXT.md`
- `docs/studies/check2hgi/README.md` (becomes `docs/README.md` — replaced wholesale)
- `docs/studies/check2hgi/issues/README.md`
- `docs/studies/check2hgi/research/*.md` (~40 files referencing each other) → moves to `docs/findings/`
- `docs/studies/check2hgi/research/canonical_improvement/{AGENT_PROMPT.md, INDEX.html, considerations.md, log.md}` → moves to `docs/studies/archive/canonical_improvement/`
- `docs/studies/check2hgi/research/merge_design/*` → moves to `docs/studies/merge_design/`
- `docs/studies/check2hgi/research/hgi_category_injection/INDEX.md` → moves to `docs/studies/archive/hgi_category_injection/`
- `docs/studies/check2hgi/research/archive/F50/*.md` → moves to `docs/findings/archive/F50/`
- `docs/studies/check2hgi/archive/post_paper_closure_2026-05-01/*.md` → moves to `docs/archive/check2hgi-post-paper-closure-2026-05-01/`
- `docs/studies/check2hgi/launch_plans/*.md` → moves to `docs/launch_plans/`

## Refs to `docs/studies/archive/fusion/` (will become `docs/archive/fusion-study/`)

- `CLAUDE.md`
- `experiments/hgi_leakage_ablation.py`
- `.claude/commands/study.md`, `.claude/commands/coordinator.md`
- `scripts/study/` — 4 files: `enroll_p1.py`, `archive_result.py`, `_state.py`, `_backfill_joint_taskbest.py`
- `docs/reports/report_v1_20260415.md`, `docs/reports/README.md`
- `docs/studies/check2hgi/AGENT_CONTEXT.md` (cross-references fusion docs)

**NOTE**: Many of these are Python scripts that read `docs/studies/archive/fusion/state.json` etc. at runtime. Need to verify whether they ALSO need `docs/archive/fusion-study/state.json` after the move, OR if the original path needs to stay live (e.g., as a symlink). Decide in 3-1 execution.

## Refs to `docs/RUNPOD_GUIDE.md` and `docs/COLAB_GUIDE.md` (become breadcrumbs)

- `CLAUDE.md`
- `docs/studies/check2hgi/NORTH_STAR.md` → `docs/NORTH_STAR.md`
- `docs/studies/check2hgi/launch_plans/{f33_f36_colab.md, ca_tx_upstream.md}` → `docs/launch_plans/`
- `notebooks/colab_phase2_grid.ipynb`, `notebooks/colab_check2hgi_mtl.ipynb`
- Various archive/research docs (those just stay in archive — links remain valid because of breadcrumb stubs)

## Verification status

- ✅ Pre-flight tag created: `pre-reorg-2026-05-14` at worktree-check2hgi-mtl tip
- ✅ h100/pervisit-fl-ca-tx-results integration verified — all 9 per-cell JSONs present and content-identical (sha256). Branch is safe to delete in Phase 3-7 (no cherry-pick needed).
- ✅ Ref-sweep grep complete (this doc).
