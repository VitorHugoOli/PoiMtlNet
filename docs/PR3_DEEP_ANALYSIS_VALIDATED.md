# Deep Analysis of PR #3 (Validated Against Current Code)

> **Date:** 2026-04-09  
> **Branch audited:** `docs/refactoring-plan`  
> **Commit audited:** `ef95bfa`  
> **Scope:** Current state of this refactor worktree  
> **Source baseline reviewed:** `docs/PR3_DEEP_ANALYSIS.md` (2026-03-29)

> **Audit-scope note:** This report is pinned to the worktree commit above.  
> If your checkout is at a different HEAD, rerun all verification commands before applying any deletion/merge step.

---

## 1. Executive Summary

The original PR3 deep analysis had useful direction, but several claims were unsafe for execution as-written.  
This validated version keeps the good parts and corrects the high-risk inaccuracies.

Main validated corrections:

1. `src/training/evaluate.py` is active and must not be deleted without caller migration.
2. Deprecated configs have mixed status:
   - `category_config.py`: currently safe to delete (no external imports).
   - `next_config.py`: still imported by active tests.
   - `MTLModelConfig`: still used in runtime code (`src/data/folds.py`).
3. Current baseline is `433 collected / 354 passed / 79 skipped` (not 411/327).
4. Logging state is mixed and needs targeted cleanup, not blanket conversion.
5. Wrapper deprecation/removal is a migration effort (14 references for canonical wrappers; 15 with broader wrapper matching), not a one-file cleanup.

---

## 2. Verified Current Snapshot

### 2.1 Test Baseline (Current)

```bash
PYTHONPATH=src pytest --co -q | tail -n 1
# 433 tests collected in 1.04s

PYTHONPATH=src pytest -q
# 354 passed, 79 skipped, 91 warnings in 105.42s

PYTHONPATH=src pytest tests/test_integration/ -q
# 12 passed, 10 warnings in 54.77s

PYTHONPATH=src pytest tests/test_regression/ -q
# 12 passed, 7 warnings in 30.37s
```

### 2.2 Phase-8 Optional Tracks Present

```bash
ls scripts/train_hydra.py dvc.yaml params.yaml \
   src/training/callbacks.py src/tracking/adapters.py \
   experiments/hydra_configs/train.yaml
# all present
```

### 2.3 Current Duplication Surface

```bash
wc -l pipelines/train/*.py scripts/train.py \
      src/training/runners/category_cv.py src/training/runners/next_cv.py \
      src/training/runners/category_trainer.py src/training/runners/next_trainer.py \
      src/training/runners/mtl_cv.py
# wrappers: 45/46/45; train.py: 323; category_cv: 111; next_cv: 177;
# category_trainer: 145; next_trainer: 187; mtl_cv: 375
```

---

## 3. Critical Corrections to `PR3_DEEP_ANALYSIS.md`

### 3.1 Evaluation Modules Are Distinct and Must Not Be Mixed

There are two different evaluation modules with different APIs and callers:

| File | Functions | Current callers |
|---|---|---|
| `src/training/evaluate.py` | `collect_predictions()`, `build_report()` | `scripts/evaluate.py` |
| `src/training/shared_evaluate.py` | `evaluate()` | `category_cv.py` and `next_cv.py` (via re-export shims) |

Evidence:

```bash
rg -n "collect_predictions|build_report" src scripts tests --glob '*.py'
rg -n "from training\.runners\.(category_eval|next_eval)|from training\.shared_evaluate import evaluate" src scripts tests --glob '*.py'
```

Required safe import rewiring before deleting shims:

```python
# src/training/runners/category_cv.py
from training.runners.category_eval import evaluate
# -> from training.shared_evaluate import evaluate

# src/training/runners/next_cv.py
from training.runners.next_eval import evaluate
# -> from training.shared_evaluate import evaluate
```

### 3.2 Deprecated Configs: Split by Actual Current Risk

Evidence:

```bash
rg -n "from configs\.category_config|CfgCategory" src tests scripts pipelines experiments --glob '*.py'
rg -n "from configs\.next_config|CfgNext" src tests scripts pipelines experiments --glob '*.py'
rg -n "MTLModelConfig" src tests scripts pipelines experiments --glob '*.py'
```

Validated status:

1. `src/configs/category_config.py`  
   - No external imports in runtime/tests/scripts/pipelines/experiments.  
   - Safe to delete now.
   - **Precondition:** update `src/configs/experiment.py` docstring text that references `CfgCategory*` before deletion to avoid a dangling reference in docs/comments.
2. `src/configs/next_config.py`  
   - Imported by multiple active tests under `tests/test_models/next/`.  
   - Not safe to delete until test migration is completed.
   - **Precondition for eventual deletion:** update `src/configs/experiment.py` `default_next()` docstring text (`"matching CfgNext* classes"`) so it does not reference removed symbols.
3. `src/configs/model.py::MTLModelConfig`  
   - Runtime dependency remains in `src/data/folds.py` (`import` + default batch size).  
   - Not safe to delete until `FoldCreator` default is migrated.
4. `src/configs/model.py` file lifecycle  
   - The file cannot be deleted after `MTLModelConfig` cleanup because `InputsConfig` is still heavily used across runtime code and tests (`rg -n "\bInputsConfig\b" src tests scripts pipelines experiments --glob '*.py'`).
5. `src/configs/model.py::ModelParameters`  
   - Currently appears to be effectively dead in runtime code (`ModelParameters` definition + one test consumer).  
   - Can be removed alongside `MTLModelConfig` during deprecated-config cleanup, leaving `InputsConfig` as the active survivor in `model.py`.

### 3.3 Logging State: Correct Counts and Targeted Scope

Raw counts:

```bash
rg -n "\bprint\(" src --glob '*.py' | wc -l
# 69
rg -l "import logging|logging\.getLogger" src --glob '*.py' | wc -l
# 6 modules
```

Logging modules (6):
- `src/data/folds.py`
- `src/data/poi_user_mapping.py`
- `src/data/sequence_poi_mapping.py`
- `src/training/callbacks.py`
- `src/tracking/display.py`
- `src/tracking/adapters.py`

Print split (important for cleanup scope):
- `31` prints are in CLI-style profiling output:
  - `src/utils/profile_reporter.py` (`27`)
  - `src/utils/profile_exporter.py` (`4`)
- `38` prints are outside those profiling reporters and should be prioritized for logging policy review.

Distribution of non-profiling prints (`38` total):
- `src/data/inputs/fusion.py`: `25`
- `src/training/runners/mtl_cv.py`: `4`
- `src/training/runners/next_trainer.py`: `3`
- `src/training/runners/category_trainer.py`: `2`
- `src/training/runners/category_cv.py`: `1`
- `src/tracking/storage.py`: `1`
- `src/data/inputs/loaders.py`: `1`
- `src/configs/embedding_fusion.py`: `1`

Phase-8 precision note:
- `3` of the runner prints above are callback-stop messages (`"Callback requested stop..."`, one per runner), added in Phase 8 for consistency with existing print-based stop reporting.

### 3.4 Wrapper Migration Effort Is Quantifiable (Pattern-Dependent)

Evidence:

```bash
rg -n "pipelines/train/(mtl|cat_head|next_head)\.pipe\.py|python pipelines/train/(mtl|cat_head|next_head)\.pipe\.py" \
   CLAUDE.md docs/DECISIONS.md docs/DATA_LEAKAGE_ANALYSIS.md docs/DATE_LEAKAGE_IMPLEMENTATION_GUIDE.md --glob '*.md'
```

Using the exact canonical-wrapper pattern above:
- `14` references across `4` docs:
  - `CLAUDE.md`: `5`
  - `docs/DECISIONS.md`: `7`
  - `docs/DATA_LEAKAGE_ANALYSIS.md`: `1`
  - `docs/DATE_LEAKAGE_IMPLEMENTATION_GUIDE.md`: `1`

If broad wrapper matching is used (all `pipelines/train/*.pipe.py` mentions):
- `15` references (adds `mtl_inductive.pipe.py` mention in `DATE_LEAKAGE_IMPLEMENTATION_GUIDE.md`).

### 3.5 Git-Tracked Junk Cleanup Is Larger Than “ignore it”

Evidence:

```bash
ls -la . | rg -n "\.DS_Store|\.coverage"
du -sh results_save
find results_save -type f | wc -l
```

Validated state:
- `.DS_Store` tracked
- `.coverage` tracked
- `processing/.DS_Store` tracked
- `results_save/` contains tracked artifacts (`11M`, `946` files)

This requires explicit tracked-file removal (for example `git rm --cached` / `git rm -r --cached`) and will create a large cleanup diff.

### 3.6 CLAUDE.md Drift Is Concrete and Countable

Evidence:

```bash
rg -n "FUSION_GUIDE\.md|REFACTORING_SUMMARY\.md|walkthrough\.md|src/model/mtlnet/|src/criterion/|src/common/ml_history/|src/common/calc_flops/|src/etl/|src/train/" CLAUDE.md
```

Current stale references detected by the command above: `14` lines, including:
- 3 non-existent docs (`FUSION_GUIDE.md`, `REFACTORING_SUMMARY.md`, `walkthrough.md`)
- multiple old module-path references from pre-migration tree.

Important structural note:
- The issue is broader than line-patching these matches. The `## File Architecture` tree section in `CLAUDE.md` is pre-Phase-5 and largely stale; this should be treated as a full rewrite task, not only grep-replace cleanup.

### 3.7 Additional Missing Risks to Track

1. `scripts/train_hydra.py` likely import bug (not just fragility)  
   - It delegates via `from scripts.train import main as train_main`, which depends on CWD/repo-root module resolution for `scripts`.  
   - Hydra typically changes CWD to a run output directory; this can break that implicit CWD-based import and raise `ModuleNotFoundError: No module named 'scripts'`.  
   - This should be treated as a latent runtime bug for hydra-enabled runs until import loading is made path-stable.
2. Known model-params serialization failure path  
   - `src/tracking/storage.py` catches exceptions in `_save_params()` and prints:
     `Error saving model parameters: ...`
   - Historical reports mention `PosixPath` serialization failures; this should be treated as an explicit cleanup item and regression-tested.
3. Evaluation naming confusion  
   - Having both `training/evaluate.py` and `training/shared_evaluate.py` is easy to misread (the original incorrect “dead code” claim came from this confusion).  
   - After shim removal, consider renaming `shared_evaluate.py` to a clearer task-specific name (for example `single_task_evaluate.py`) to reduce future review mistakes.

---

## 4. What From the Original Analysis Remains Valid

Still valid and useful:

1. Cleanup of stale root artifacts and tracked junk.
2. Elimination of one-line eval re-export shims (with explicit import rewiring first).
3. `.gitignore` expansion and lint/entrypoint ergonomics.
4. Conservative phase ordering: safe BPR first, migration BPR second, SC last.

---

## 5. Revised Prioritized Action Plan (Validated)

### Priority 1 — Safe BPR Cleanup

#### 1a. Zero-risk cleanup (docs + tracked artifacts)

1. Rewrite `CLAUDE.md` to current architecture (not just line edits), including:
   - replacing the stale file-tree section,
   - replacing old module paths,
   - removing or replacing non-existent linked docs.
2. Remove tracked junk (`.DS_Store`, `.coverage`, `processing/.DS_Store`) and expand `.gitignore`.
3. Plan and execute tracked cleanup of `results_save/` (large deletion; separate commit recommended).

#### 1b. Low-risk runtime fix (requires tests)

4. Fix model-params serialization and fallback handling in `tracking.storage`:
   - normalize non-JSON objects (for example `Path`/`PosixPath`) before JSON dump,
   - replace fallback `print()` with structured logging (`logger.warning`/`logger.error`).

### Priority 2 — BPR Refactors With Preconditions

1. Remove eval re-export shims:
   - Update imports in `category_cv.py` and `next_cv.py` to `training.shared_evaluate`.
   - Delete `category_eval.py` and `next_eval.py`.
2. Split deprecated-config migration:
   - Update `experiment.py` comments/docstrings referencing `CfgCategory*`, then delete `category_config.py`.
   - Migrate tests off `next_config.py`.
   - Update `experiment.py` `default_next()` docstring referencing `CfgNext*` before deleting `next_config.py`.
   - Remove runtime dependency on `MTLModelConfig` in `data/folds.py`.
   - Remove `ModelParameters` if no new runtime consumers are introduced.
   - Only then delete remaining deprecated config symbols.
   - Keep `src/configs/model.py` while `InputsConfig` remains in use.
3. Logging migration by intent:
   - Prioritize 38 non-profiling prints.
   - Keep/justify CLI profiling prints where stdout formatting is intentional.

### Priority 3 — Potential SC / High Coordination

1. Re-evaluate trainer/CV merge ROI:
   - Phase 8 callbacks already provide composability (early stopping/checkpoint/diagnostics hooks).
   - Remaining merge benefit is primarily code reduction; weigh against debuggability and SC risk.
2. Remove wrapper scripts only after docs/commands migration across all referenced docs.
3. Rework eval structure only after explicit caller migration (`scripts/evaluate.py` + runner imports) is complete.
4. Optional clarity refactor: rename `shared_evaluate.py` to a less ambiguous module name once imports are consolidated.

---

## 6. Guardrails for Any Follow-up Cleanup PR

Minimum acceptance gate:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src pytest tests/test_integration/ -q
PYTHONPATH=src pytest tests/test_regression/ -q
```

Deletion safety gate:

```bash
rg -n "MODULE_OR_SYMBOL_NAME" src tests scripts pipelines experiments --glob '*.py'
```

Doc-migration gate for wrapper removal:

```bash
rg -n "pipelines/train/(mtl|cat_head|next_head)\.pipe\.py|python pipelines/train/(mtl|cat_head|next_head)\.pipe\.py" \
   CLAUDE.md docs/DECISIONS.md docs/DATA_LEAKAGE_ANALYSIS.md docs/DATE_LEAKAGE_IMPLEMENTATION_GUIDE.md --glob '*.md'
```

Logging-scope gate:

```bash
rg -n "\bprint\(" src --glob '*.py' | awk -F: '{c[$1]++} END {for (f in c) printf "%4d %s\n", c[f], f}' | sort -nr
```

---

## 7. Final Assessment

The original analysis remains a good issue catalog, but it needed correction before execution.  
This updated validated report now distinguishes immediate-safe deletions from migration-required removals, quantifies migration scope, and adds missing operational risks.
