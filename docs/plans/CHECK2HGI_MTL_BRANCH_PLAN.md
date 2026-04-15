# Plan — Check2HGI MTL Branch (next-POI + Next-Region)

## Context

The project currently trains MTL jointly on (POI-category classification, next-POI-category prediction) over a fused embedding. Two things are shifting:

1. **Switching embedding engine to `check2HGI`.** Check2HGI produces **check-in-level** embeddings — the same POI visited twice yields two different vectors. That contextual encoding is wasted on a per-POI classification task, and in fact makes the "POI category" label ambiguous (which check-in's vector represents the POI?). So the category task must be removed on this track.
2. **Replacing category with a trajectory-aligned auxiliary task.** The user's proposal (`docs/issues/MTL_TASK_REPLACEMENT_PROPOSAL.md`) lists three candidates: **Next Region**, **Next Time-Gap**, **Revisit-vs-Explore**. The goal of this plan is to pick the smallest high-signal task set and add it *alongside* the existing code — legacy MTL (cat + next) must keep working, under its current tests and CLI surface.

This will live on a new branch. The new branch also gets its own study tree (`docs/studies/check2hgi/`) so the coordinator/worker skills don't confuse it with the legacy P0–P6 study; `CLAUDE.md` will be re-pointed at the new subtree on that branch.

## Task selection — decision

**Confirmed by user:** 2-task MTL — `{next_category, next_region}`. **Next Time-Gap** is scaffolded (TaskSpec registered, loader stubbed) but not activated in the initial run; it moves into Phase 3. **Revisit-vs-Explore** is skipped for the BRACIS timeline.

**Why Next-Region first:**

- **Strongest literature alignment** with hierarchical MTL for next-POI:
  - HMT-GRN (SIGIR '22) — hierarchical region MTL as the canonical auxiliary task.
  - MGCL (Frontiers 2024) — auxiliary next-region + next-category heads.
  - Bi-Level Graph Structure Learning (arXiv 2024) — ties region and POI levels.
  - "Learning Hierarchical Spatial Tasks" (TORS, 2024).
- **Direct reuse of Check2HGI assets:** the graph artifact at `IoPaths.CHECK2HGI.get_graph_data_file(state)` already contains a `placeid → region_idx` map; region embeddings already exist (`region_embeddings.parquet`).
- **Zero new loss plumbing:** classification loss, same `NextHeadMTL` transformer, same class-weight logic — just a different label column and `num_classes = n_regions`.
- **Fast to prototype:** label derivation is a single join; input X is reused from the existing `next.parquet`.

**Why defer Time-Gap to Phase 2:**

- Lit support exists (TAPT, ImNext, SGRec) but the signal complements rather than reinforces POI-category: useful, not critical for showing "check2HGI + regional MTL works".
- Requires choosing regression vs binned classification (I'd recommend binned so the loss plumbing stays identical).
- Adds a third head and stresses NashMTL's ECOS solver.

**Why skip Revisit-vs-Explore for now:** binary signal is too coarse; weakest complementarity to next-POI-category, lowest expected lift.

## Strategy: additive, registry-driven

No in-place refactor. The existing 2-task path (`MTLnet` + `mtl_cv.py` + `_run_mtl`) stays *bit-exact*. The new path lives in parallel:

- New `TaskSpec` registry + `TaskSetSpec` preset list.
- New `MTLnetGeneric` (allocates `nn.Embedding(len(task_set), ...)` sized to the active task set — legacy `MTLnet` is never instantiated by the new path, so its pinned `(2, d)` shape and RNG contract in `tests/test_regression` are preserved).
- New `mtl_cv_generic.py` runner that iterates over `task_set.names` instead of unpacking `(data_next, data_category)`.
- New `FoldCreatorGeneric`: reuses the existing StratifiedGroupKFold user-partition logic, materializes per-task tensors over the same row indices (all check2HGI tasks are check-in-level, so POI-exclusivity logic isn't needed).
- Entry point via a new `--task-set {legacy|check2hgi_next_region|check2hgi_region_timegap}` CLI flag; `--task` remains as-is.

## Files to create / modify

**New files**
- `src/tasks/registry.py` — `TaskSpec` dataclass + `TASK_SPECS` dict (`next_category`, `category`, `next_region`, `next_time_gap`).
- `src/tasks/task_sets.py` — `TaskSetSpec` + presets `LEGACY_NEXT_CATEGORY`, `CHECK2HGI_NEXT_REGION`, `CHECK2HGI_REGION_PLUS_TIMEGAP`.
- `src/data/folds_generic.py` — `FoldCreatorGeneric` + `GenericFoldResult` (dict-of-TaskFoldData).
- `src/data/inputs/next_region.py` — loader joining `next.parquet` with `placeid → region_idx` from the check2HGI graph artifact.
- `src/models/mtl/mtlnet_generic/model.py` — `MTLnetGeneric` (separate class from `MTLnet`).
- `src/training/runners/mtl_cv_generic.py` — generic N-task runner (copy of `mtl_cv.py` with named loops).
- `pipelines/create_inputs_check2hgi.pipe.py` — materializes `next_region.parquet` under `output/check2hgi/<state>/input/`.
- `tests/test_data/test_next_region_loader.py`, `tests/test_models/test_mtlnet_generic.py`, `tests/test_training/test_mtl_cv_generic.py`.
- `docs/studies/check2hgi/` tree (see below).

**Touched (additive only)**
- `src/data/folds.py` — add `TaskType.MTL_CHECK2HGI`; branch in `create_folds()` delegating to `FoldCreatorGeneric`. **Do not** edit `_create_mtl_folds`.
- `src/configs/paths.py` — add `IoPaths.get_next_region(state, engine)` + `_Check2HGIIoPath.get_region_map()`.
- `src/configs/experiment.py` — add optional `task_set: str = None` + `default_mtl_check2hgi()` factory; keep `default_mtl()` untouched.
- `scripts/train.py` — add `--task-set`; dispatch `"mtl_check2hgi"` to the generic runner. Legacy `_run_mtl` untouched.
- `CLAUDE.md` — re-point "active study" references to `docs/studies/check2hgi/`; add explicit "legacy study is frozen on this branch" note. Keep everything else.

## Data pipeline for next-region

1. Read `next.parquet` for engine=check2HGI — X features unchanged (sliding window of check-in embeddings).
2. Load the graph artifact; extract `placeid → region_idx`.
3. Map `next_placeid` → `region_idx`. Fail loud on missing placeids (log count + sample).
4. Write `output/check2hgi/<state>/input/next_region.parquet` with schema `{userid, <x_0..x_N>, region_idx}`.
5. `FoldCreatorGeneric` splits on `userid` via the same StratifiedGroupKFold call (stratified on `next_category` still — region tail classes too sparse to stratify on directly), then slices `region_idx` labels over the same row partition.

## Runner generalisation (mtl_cv_generic.py)

- Replace `(data_next, data_category)` unpacking with a `zip_longest_cycle` over `[td.train.dataloader for td in task_dataloaders.values()]` in `task_set.names` order.
- `losses = torch.stack([criteria[name](preds[i], targets[i]) for i, name in enumerate(task_set.names)])`.
- Replace every `'next'` / `'category'` string constant with `f"val_f1_{name}"` etc.
- `FoldHistory.standalone(set(task_set.names))`.
- Construct NashMTL fresh per fold with `n_tasks=len(task_set)` (already its constructor arg — no cvxpy shape rewiring needed).
- Generic N-D Pareto dominance check replaces the 2-D `_pareto_front_indices`.
- Emit `val_joint_score = mean(val_f1_*)` for the checkpoint monitor.

**Legacy bit-exact contract:** `scripts/train.py` dispatches `--task mtl` (no `--task-set`) to the legacy runner. A test asserts `_RUNNERS["mtl"] is <legacy runner>` and pins `val_f1_{next,category}` under seed 42 to the current baseline.

## Docs/studies restructure

**Legacy tree handling:** the user asked whether to physically move legacy studies to an archive on this branch. Recommendation: **do not move — freeze in place.** The user plans to keep evolving `docs/studies/*` on `main` for the legacy track; if those files are relocated on this branch, every future merge will conflict on their paths. Instead, add a one-line `docs/studies/FROZEN.md` on this branch pointing at `docs/studies/check2hgi/`, and rely on the CLAUDE.md re-point (next section) to steer subagents. Subagents read CLAUDE.md first — they will not wander into the legacy tree. If check2hgi later supersedes the main track, the rename is trivial then.

```
docs/studies/check2hgi/
  README.md                  # entry point + run commands
  QUICK_REFERENCE.md         # CLI cheat-sheet, monitors, envs
  MASTER_PLAN.md             # phases overview for this track
  CLAIMS_AND_HYPOTHESES.md   # C-claims: does next-region MTL help next-POI?
  COORDINATOR.md             # same role, rooted at this subtree
  HANDOFF.md                 # session handoff notes (stub)
  state.json                 # coordinator state (initial scaffold)
  phases/
    P0_preparation.md        # label derivation + input pipeline
    P1_baseline_legacy.md    # regression-floor gate for legacy path
    P2_next_region_mtl.md    # run the 2-task check2HGI MTL, compare baselines
    P3_region_plus_timegap.md# optional 3-task extension
    P4_ablations.md          # head choice, loss family, task-embedding init
  results/                   # placeholder, matches legacy layout
```

Each file is a stub with title, one-paragraph purpose, `Status: scaffold`, and links to sibling legacy files where useful. Legacy `docs/studies/*.md` is left untouched and frozen on this branch.

## CLAUDE.md edits (new branch only)

- Replace references to `docs/studies/` (as the active track) with `docs/studies/check2hgi/`.
- Update any listed sub-files (`MASTER_PLAN.md`, `CLAIMS_AND_HYPOTHESES.md`, `QUICK_REFERENCE.md`, `phases/P*.md`, `state.json`) to the `check2hgi/` subpath.
- Add: *"Legacy study artifacts at `docs/studies/*.md` and `docs/studies/phases/` are frozen on this branch; do not edit them. Modify `docs/studies/check2hgi/` only."*
- Leave all other sections untouched (model registry, freeze-folds workflow, test commands, etc.).

## Verification

**Legacy must stay green (gate before merge):**
- `pytest tests/test_regression -q` — unchanged assertions.
- `pytest tests/test_models/test_mtlnet.py -q` — pins task_embedding shape and forward equivalence.
- Smoke: `python scripts/train.py --state florida --engine dgi --task mtl --epochs 1 --folds 1` reproduces pre-branch `val_f1_{next,category}` at seed 42 (byte-identical).

**New path smoke:**
1. `python pipelines/create_inputs_check2hgi.pipe.py --state florida && --state alabama` — verify `next_region.parquet` exists, no NaN in `region_idx`, distribution logged.
2. `python scripts/train.py --state florida --engine check2hgi --task-set check2hgi_next_region --epochs 2 --folds 1` — verifies new runner end-to-end.
3. `pytest tests/test_training/test_mtl_cv_generic.py tests/test_models/test_mtlnet_generic.py tests/test_data/test_next_region_loader.py -q`.
4. Repeat step 2 for Alabama.

## Risks & landmines

- **NashMTL cvxpy shapes:** `n_tasks` is parameterised (good). Per-fold fresh construction is already the pattern — keep it. With N≥3, ECOS convergence gets finicky; `_solver_failures` counter logs it.
- **Task-embedding RNG contract:** legacy `MTLnet` remains untouched — its `(2, d)` embedding shape and RNG consumption are preserved because `MTLnetGeneric` is a separate class, not a subclass widening.
- **Missing `placeid → region_idx` entries:** fail loud with a count + examples; do not silently drop.
- **Stratification tail classes:** stratify on `next_category`, group on `userid`. Don't stratify on `region_idx` (too sparse).
- **`calculate_model_flops` arity:** legacy passes a 2-tuple. Generic path must pass N-tuple; wrap in the existing `FlopsMetrics(0,0)` try/except sentinel.
- **`TaskType` serialisation:** `load_folds` round-trip via `TaskType(task).value`. Adding `MTL_CHECK2HGI` is forward-compatible; asserted by a round-trip test.
- **Checkpoint monitor typos:** `ModelCheckpoint` silently no-ops on unknown keys — test that the configured monitor string is actually emitted by the generic runner.

## Critical files
- `src/training/runners/mtl_cv.py` (reference; copied, not modified)
- `src/models/mtl/mtlnet/model.py` (untouched)
- `src/data/folds.py` (add enum value + dispatch branch only)
- `scripts/train.py` (add `--task-set` flag; legacy dispatch untouched)
- `src/configs/paths.py` (add check2HGI task path helpers)
- `docs/issues/MTL_TASK_REPLACEMENT_PROPOSAL.md` (source of task candidates)
- `docs/check2hgi_overview.tex` (engine reference)
- `CLAUDE.md` (re-point to `docs/studies/check2hgi/`)

## Literature pointers

- HMT-GRN, SIGIR 2022 — hierarchical region MTL for next-POI.
- Multi-Granularity Contrastive Learning (MGCL), Frontiers 2024 — auxiliary next-region + next-category.
- Bi-Level Graph Structure Learning, arXiv 2024 — region/POI level interaction.
- "Learning Hierarchical Spatial Tasks", TORS 2024 — formal hierarchical-MTL framing.
- TAPT / ImNext / SGRec — time-interval & category-auxiliary lines (relevant for the deferred Time-Gap phase).
