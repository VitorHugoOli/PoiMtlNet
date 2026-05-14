# Phase P1 — Code parameterisation (TaskConfig)

**Gates:** none (can run in parallel with P-1).
**Exit gate:** legacy regression tests green; `--task-set legacy_category_next` byte-identical to pre-branch `--task mtl` at seed 42.

## Scope (pivoted from the original plan)

Originally: parallel `MTLnetGeneric` + `mtl_cv_generic.py` runner. Rejected because 674-line parallel copy + 143 hardcoded-string replacement is a large, blind parallel codebase.

**Adopted approach:** parameterise the existing 2-slot MTLnet + `mtl_cv.py` *in place* using `TaskConfig` / `TaskSet` (`src/tasks/`). Defaults reproduce the legacy 2-task `{category, next}` pair bit-exactly; new task sets are opt-in via a `--task-set` CLI flag.

## Deliverables

### P1-a — `src/tasks/` (complete, committed `1ae235b`)

`TaskConfig` (name, num_classes, head_factory, head_params, is_sequential, primary_metric) + `TaskSet` + presets `LEGACY_CATEGORY_NEXT`, `CHECK2HGI_NEXT_REGION`.

### P1-b — parameterise `MTLnet`

- Add `task_set: TaskSet = LEGACY_CATEGORY_NEXT` constructor arg (default preserves legacy behaviour).
- Rename internal slots from `category_encoder` / `next_encoder` to `task_a_encoder` / `task_b_encoder` **only if** the rename preserves parameter ordering and state-dict keys — otherwise keep legacy attribute names for the legacy preset and add new attributes for non-legacy presets.
- Drive `_build_category_head` / `_build_next_head` from `task_set.task_a.head_factory` / `task_set.task_b.head_factory` with default behaviour unchanged for legacy factories.
- Sequential vs flat input: if `task_set.task_a.is_sequential == task_set.task_b.is_sequential == True` (check2HGI case), both inputs are [B, T, D] — the encoder Linear stack already broadcasts correctly.

**Bit-exact contract for legacy:** with the default preset, `MTLnet(*args, task_set=LEGACY_CATEGORY_NEXT)` must produce the same `state_dict()` keys and tensor shapes as pre-change `MTLnet(*args)`. Test assertion in `tests/test_models/test_mtlnet.py`.

### P1-c — parameterise `src/training/runners/mtl_cv.py`

- Add `task_set` arg to `train_with_cross_validation` (and `train_model`) with default `LEGACY_CATEGORY_NEXT`.
- Replace every hardcoded `'category'` / `'next'` string with `task_set.task_a.name` / `task_set.task_b.name`. 143 occurrences.
- Gradient-cosine diagnostic: keep slot-indexed (`losses[0]` vs `losses[1]`) but emit `grad_cosine_{task_a.name}_vs_{task_b.name}`.
- Checkpoint monitor: use `primary_metric` from each TaskConfig to pick the per-head watched key; compute `joint_<metric>` from both.
- Fold result object: keep the two named slots `next` and `category` in legacy mode, or a pair of slots keyed by task name.

**Bit-exact contract for legacy:** `scripts/train.py --task mtl` (default task set) must reproduce seed-42 final metric scalars byte-identically vs pre-change. Test added in `tests/test_training/`.

### P1-d — `scripts/train.py`

- Add `--task-set {legacy_category_next,check2hgi_next_region}` flag.
- Add `--engine check2hgi` validation path (in `_run_mtl`).
- When non-legacy task set is chosen, route the fold creation / data loading paths through the check2HGI loaders (P0 outputs).

## Risks

- **143 string replacements is mechanical but error-prone.** Do it in small commits (per-function or per-section) to keep PR-sized diffs reviewable.
- **Scheduling & early-stopping keyed on `val_f1_category`.** Must generalise to `val_<primary_metric>_<task_b.name>` for check2HGI. Keep legacy string-literal as the default to preserve behaviour.
- **Callback monitor names (`ModelCheckpoint` etc.) silently no-op on unknown keys.** Add a positive test that the configured monitor key appears in the metric dict.

## Claims touched

None directly — but every downstream phase depends on P1 being correct.
