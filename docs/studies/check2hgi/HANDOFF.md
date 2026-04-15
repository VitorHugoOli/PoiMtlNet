# Check2HGI Track — Session Handoff (2026-04-15)

**Snapshot of a full session of work.** Pick up here.

## Status at a glance

| Phase | Status | Notes |
|---|---|---|
| Research (datasets + metrics + enrichment verdict) | **DONE** | `docs/plans/CHECK2HGI_MTL_OVERVIEW.md` |
| Branch plan + revision notes | **DONE** | `docs/plans/CHECK2HGI_MTL_BRANCH_PLAN.md` |
| Study tree (claims, hypotheses, phases, ablations) | **DONE** | `docs/studies/check2hgi/` (this dir) |
| P-1: check2HGI embeddings (FL + AL) | **DONE** | Both states trained 500 epochs, ~20 min wall-clock |
| P0: next_region label pipeline | **DONE** | AL: 1,109 regions, FL: 4,703 regions |
| P1-a: TaskConfig / TaskSet registries | **DONE** | `src/tasks/` |
| P1-b: Parameterise MTLnet | **DONE** | Bit-exact legacy; new sequential-A path |
| P1-c: Parameterise mtl_cv.py runner | **DONE** | 60-line diff, 221 tests pass |
| P1-d: Parameterise mtl_eval.py + metric CPU fallback | **DONE** | Fixes MPS OOM at num_classes ~10^3 |
| End-to-end smoke test | **DONE** | `scripts/smoke_check2hgi_mtl.py` — 2 epochs AL |
| Full CLI wiring (`scripts/train.py --task-set`) | **PENDING** | See §Next |
| FoldCreator check2HGI MTL path | **PENDING** | Smoke test bypasses it; see §Next |
| P2: baselines (HGI single-task, check2HGI single-task, next_region single-task) | **PENDING** | Claims CH01, CH04, CH05 |
| P3: headline MTL runs FL + AL | **PENDING** | Claims CH02, CH03 |
| P4: ablations | **PENDING** | Claims CH06–CH09 |

## Commits on this branch (`worktree-check2hgi-mtl`)

```
024730c  feat(training): per-task num_classes + CPU fallback for high-cardinality metrics
a8a977e  feat(data): next-region label pipeline for check2HGI
cf18f13  feat(mtl_cv): parameterise task names with TaskSet (legacy bit-exact)
3048a69  feat(mtlnet): parameterise MTLnet with TaskSet (legacy path bit-exact)
1ae235b  feat(tasks): add TaskConfig + TaskSet presets (legacy, check2hgi_next_region)
f9b7ea8  docs(studies): scaffold docs/studies/check2hgi/ with claims + ablations
a61042f  plan(check2hgi): finalize branch plan + enable FL in embedding pipeline
6ebe6ea  docs(plans): add check2HGI MTL research overview and decisions
ad78ace  docs(plans): add check2HGI MTL branch plan (next-POI + next-region)
```

All commits sign-off'd with `Co-Authored-By: Claude`.

## Where the data lives

Main repo (`/Volumes/Vitor's SSD/ingred/`) holds all artefacts; worktree holds code:

- `output/check2hgi/alabama/embeddings.parquet` — 113,846 check-in embeddings (64-dim)
- `output/check2hgi/alabama/poi_embeddings.parquet` — 11,848 POIs
- `output/check2hgi/alabama/region_embeddings.parquet` — 1,109 regions
- `output/check2hgi/alabama/input/next.parquet` — 12,709 next-POI sequences (9×64 + category + userid)
- `output/check2hgi/alabama/input/next_region.parquet` — 12,709 rows, 576 emb cols + region_idx + userid
- `output/check2hgi/alabama/temp/checkin_graph.pt` — preprocessing artefact (placeid_to_idx, poi_to_region, …)
- Same under `florida/`. Florida has 159,175 sequences and 4,703 regions.

Region cardinality to log into claim CH04 statements when the head actually runs.

## Key numbers for the paper

- Alabama: 113,846 check-ins / 11,848 POIs / **1,109 regions** / 12,709 sequences. Majority-region frequency 2.3% (292 of 12,709).
- Florida: larger. 4,703 regions. **Majority-region frequency 22.2%** — severely imbalanced; must beat that for next_region head to be "learning" (CH04).
- Acc@1 baseline to beat:
  - AL: 2.3%
  - FL: 22.2%

## What the end-to-end smoke test shows

`scripts/smoke_check2hgi_mtl.py --state alabama --epochs 2` completes in ~10 seconds on MPS:

- Loads real check2HGI data
- Builds a 1-fold user-held-out split (StratifiedGroupKFold on userid, stratified on next_category)
- Constructs `MTLnet(task_set=CHECK2HGI_NEXT_REGION)` with runtime `n_regions=1109`
- Both heads are sequential `NextHeadMTL` transformers
- `train_model(task_set=…)` runs 2 epochs with NashMTL criterion
- Reports: next_category macro-F1 ≈ 0.07 (2 epochs only, untrained), next_region macro-F1 ≈ 0.0002 (1109-class macro-F1 is near-zero; CH05 predicts this)

**This exercise validates every piece of the plumbing.** The macro-F1 on next_region isn't the intended metric — Acc@1 + MRR are. Those are already computed (by `compute_classification_metrics`) and logged to the FoldHistory, just not surfaced in the smoke-test log. They're available via `hist.task('next_region').best.<key>`.

## Next steps (in dependency order)

### Step 1 — `scripts/train.py --task-set` wiring
The smoke test bypasses `train.py` entirely and constructs folds by hand. Real runs need:
- New `--task-set` CLI flag with choices `{legacy_category_next, check2hgi_next_region}`.
- Dispatch: if `task_set == check2hgi_next_region`, call a new `_run_mtl_check2hgi` that:
  - Builds folds via a new `FoldCreator._create_check2hgi_mtl_folds` (see Step 2).
  - Resolves `task_set.task_b.num_classes = n_regions` from the next_region.parquet load.
  - Constructs `MLHistory(tasks={"next_category", "next_region"}, …)`.
  - Uses monitor `val_accuracy_next_region` (Acc@1) OR an added `val_joint_acc1` (Option (c) from overview §2).
  - Passes `task_set=CHECK2HGI_NEXT_REGION_RESOLVED` to `train_with_cross_validation`.
- Keep `--task mtl` default routed to the legacy path (`task_set=LEGACY_CATEGORY_NEXT` implicit).

### Step 2 — `FoldCreator._create_check2hgi_mtl_folds`
- Load X + userids from next.parquet, y_cat from next.parquet's `next_category`, y_region from next_region.parquet's `region_idx`.
- Row-align on userid (cast to str).
- Single StratifiedGroupKFold on userids stratified on y_cat.
- For each fold, SAME train/val indices go to both task_a (next_category) and task_b (next_region) slots.
- Return `FoldResult` where `.category` = next_category TaskFoldData, `.next` = next_region TaskFoldData.
- Add `--task-set` selection path in `FoldCreator.create_folds` (or a new dedicated entry).

### Step 3 — Add `val_joint_acc1` metric emission
Currently mtl_cv.py emits `val_joint_score = 0.5 * (f1_val_next + f1_val_category)`. For check2HGI we want also `val_joint_acc1 = 0.5 * (acc1_next + acc1_cat)` (both heads have accuracy in their metric dict). Add it to the callback-context metrics and make it selectable as the ModelCheckpoint monitor.

### Step 4 — Run P2 baselines
After train.py is wired:
```
python scripts/train.py --state alabama --engine hgi --task next --folds 5 --epochs 50
python scripts/train.py --state alabama --engine check2hgi --task next --folds 5 --epochs 50
python scripts/train.py --state alabama --engine check2hgi --task next_region --folds 5 --epochs 50
# repeat for florida
```

(The third one needs a `TaskType.NEXT_REGION` single-task path OR reuse the MTL path with one head disabled — simpler to add `TaskType.NEXT_REGION` + `_create_single_task_next_region_folds`.)

### Step 5 — Run P3 headline
```
python scripts/train.py --state alabama --engine check2hgi --task mtl --task-set check2hgi_next_region --folds 5 --epochs 50
python scripts/train.py --state florida --engine check2hgi --task mtl --task-set check2hgi_next_region --folds 5 --epochs 50
```

Write results under `docs/studies/check2hgi/results/P3/`. Update CH02 and CH03 claim statuses.

### Step 6 — P4 ablations per `ABLATIONS.md`

## Known issues / warnings

- `enable_nested_tensor is True, but self.use_nested_tensor is False` — NextHeadMTL TransformerEncoder; harmless warning, suppressible.
- cvxpy `Problem is not DPP` — NashMTL solver setup; reported once per fresh solver; benign perf note.
- `torch.compile disabled - incompatible with PyG scatter operations` — expected; preprocessing uses PyG.

## If something breaks

- **Legacy regression:** `pytest tests/test_regression -q` must stay green. If it breaks, bisect the commit that failed; likeliest culprit is a stray diff in `mtl_cv.py` or `metrics.py` metric-key change.
- **MPS OOM at num_regions > ~256:** fix is in `src/tracking/metrics.py` `_CARDINALITY_CPU_THRESHOLD = 256`. If a future state has more regions, threshold doesn't change (it just falls back to CPU).
- **next.parquet / sequences_next.parquet userid dtype mismatch:** already handled in `src/data/inputs/next_region.py` (cast both to str). If a new state adds a third dtype (e.g. int32), update that function.
- **Corrupt checkin_graph.pt:** rerun `pipelines/embedding/check2hgi.pipe.py` for that state. Preprocessing is deterministic, but SIGTERM mid-write leaves a corrupt file.

## What wasn't done this session

- `FoldCreator` still doesn't have a check2HGI dispatch path — smoke test uses its own hand-rolled fold loading.
- `scripts/train.py` still doesn't know about `--task-set`.
- `val_joint_acc1` not yet a separate callback-context metric (only `val_joint_score` based on F1).
- No baseline runs yet — all claims are still `pending`.
- CLAUDE.md not yet updated on this branch to re-point at `docs/studies/check2hgi/` (per plan §8).

## Test command to reproduce the state

```
# From the main repo root:
cd /Volumes/Vitor\'s\ SSD/ingred
git worktree list                # confirm this branch's worktree exists
cd .claude/worktrees/check2hgi-mtl
git log --oneline -10            # confirm the 9 commits above

# Legacy regression (must be green):
PYTHONPATH=src .venv/bin/python -m pytest tests/test_regression tests/test_models tests/test_training -q

# End-to-end smoke on Alabama:
PYTHONPATH=src DATA_ROOT=/Volumes/Vitor\'s\ SSD/ingred/data OUTPUT_DIR=/Volumes/Vitor\'s\ SSD/ingred/output \
  .venv/bin/python scripts/smoke_check2hgi_mtl.py --state alabama --epochs 2
```
