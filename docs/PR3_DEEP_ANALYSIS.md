# Deep Analysis of PR #3 — Refactoring: Improving Codebase and Data Management

> **Date:** 2026-03-29  
> **Branch:** `docs/refactoring-plan` → `main`  
> **Scope:** 206 files changed, +9,042 / −4,825 lines  
> **Phases completed:** 0 through 7 (all planned phases)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What Was Done (Phase-by-Phase)](#2-what-was-done)
3. [What to Remove / Delete](#3-what-to-remove--delete)
4. [Flows to Merge](#4-flows-to-merge)
5. [Files and Folders to Merge](#5-files-and-folders-to-merge)
6. [Flow Improvements for Execution and Maintenance](#6-flow-improvements)
7. [Prioritized Action Plan](#7-prioritized-action-plan)

---

## 1. Executive Summary

PR #3 executed a 7-phase refactoring of the MTLnet codebase — moving from a flat, duplicated structure to a modular architecture with registries, unified configuration, and proper cross-validation splits. The refactoring was thorough and well-documented, but it left behind **residual duplication**, **stale artifacts**, **intermediate abstractions that add indirection without value**, and **missing developer ergonomics**. This analysis identifies concrete follow-up actions.

### Key Metrics

| Metric | Before | After | Assessment |
|--------|--------|-------|------------|
| Source modules | 5 flat dirs | 6 logical modules | ✅ Improved |
| Config files | 4 scattered | 1 unified + 2 deprecated | 🟡 Cleanup needed |
| Code duplication (CV runners) | ~90% identical | ~65% identical | 🟡 Partially reduced |
| Training pipelines | 3 independent scripts | 3 thin wrappers + 1 CLI | 🔴 Over-layered |
| Test count | ~250 | 411 (327 passing) | ✅ Improved |
| Dead code | Moderate | Still present | 🔴 Cleanup needed |
| Deprecation shims | Many (Phase 5) | All removed (Phase 6) | ✅ Clean |
| Print statements vs logging | All print | All print | 🔴 No change |

---

## 2. What Was Done

### Phase 0 — Safety Net & Hard Decisions
- Established regression fixtures (calibrated F1 floors: 94%, 99%, 92%)
- Defined MTL split protocol with `StratifiedGroupKFold` (user isolation)
- Created `pyproject.toml` with exact-pinned dependencies
- Added CLI flags (`--state`, `--engine`) to pipelines
- Generated feasibility reports per state (4 JSON files)

### Phase 1 — Low-Risk Cleanup
- Merged duplicate `evaluate()` functions → `shared_evaluate.py`
- Removed dead `validation_model()` function
- Created `CategoryHeadMTL = CategoryHeadEnsemble` alias
- Fixed typos, removed stale requirements files
- Archived unused notebooks to `experiments/archive/`

### Phase 2 — Data Contract Hardening
- Created `src/data/schemas.py` (parquet schema validation)
- Made `FoldCreator` infer embedding dimensions from artifacts
- Retained `userid` column for user-level splits
- Built `poi_user_mapping.py` and `sequence_poi_mapping.py`
- Implemented `StratifiedGroupKFold` with split manifests

### Phase 3 — Configuration Unification
- Created `ExperimentConfig` dataclass (64 fields, 3 factory methods)
- Added `DatasetSignature` (streaming SHA-256) and `RunManifest`
- Removed all hardcoded hyperparameters from training loops
- Deprecated (but kept) old config classes

### Phase 4a — Shared Utilities & Registries
- `@register_model` / `@register_loss` decorator pattern
- Consolidated 8 category heads → `models/heads/category.py`
- Consolidated 7 next heads → `models/heads/next.py`
- Extracted `training/helpers.py` (class weights, optimizer, scheduler)
- Extracted `training/evaluate.py` (collect_predictions, build_report)

### Phase 4b — MTL Semantic Corrections
- Fixed scheduler step count (was using wrong dataloader length → crashed at step 28)
- Aligned validation to use `zip_longest_cycle()` (was truncating 45.7% of data)

### Phase 5 — Folder Tree Migration
- Moved all source files to target structure (`models/`, `losses/`, `data/`, `training/`, `tracking/`, `utils/`)
- Moved embedding trainers to `research/embeddings/`
- Added deprecation shims for backward compatibility
- Migrated test structure to match

### Phase 6 — Script Consolidation
- Created `scripts/train.py` (canonical CLI entrypoint)
- Created `scripts/evaluate.py` (checkpoint evaluation)
- Created `experiments/configs/` (declarative config constructors)
- Converted training pipelines to thin subprocess wrappers
- **Removed all deprecation shims**

### Phase 7 — Testing & Reproducibility
- Added integration tests (12 tests across 3 files)
- Upgraded regression fixtures to use shared synthetic data
- Added `random.seed()` to `FoldCreator` for determinism
- Cleaned up stale test directories

---

## 3. What to Remove / Delete

### 3.1 Deprecated Config Classes (HIGH priority)

**Files:**
- `src/configs/category_config.py` — Contains `CfgCategoryHyperparams`, `CfgCategoryModel`, `CfgCategoryTraining`
- `src/configs/next_config.py` — Contains `CfgNextHyperparams`, `CfgNextModel`, `CfgNextTraining`
- `src/configs/model.py` → `MTLModelConfig` class (marked DEPRECATED)

**Why:** These were marked for removal in Phase 5 but survived. They are not imported anywhere in `src/`, `tests/`, `pipelines/`, `scripts/`, or `experiments/`. The refactoring replaced them with `ExperimentConfig`.

**Action:** Delete the two files entirely. Remove `MTLModelConfig` from `model.py`. Verify zero import references with `grep -rn "category_config\|next_config\|MTLModelConfig" src/ tests/ pipelines/ scripts/`.

### 3.2 Dead Evaluation Functions (HIGH priority)

**Files:**
- `src/training/evaluate.py` → `collect_predictions()` and `build_report()` — **0 call sites in the codebase**
- `src/training/shared_evaluate.py` → Only used as re-export target for single-line `category_eval.py` and `next_eval.py`

**Why:** The refactoring extracted these but the runners never adopted them. The actual evaluation logic is embedded in each trainer/CV file.

**Action:** Delete `evaluate.py`. Inline `shared_evaluate.py`'s `evaluate()` into the trainers that use it, or keep it as the single shared evaluate module.

### 3.3 Single-Line Re-Export Files (MEDIUM priority)

**Files:**
- `src/training/runners/category_eval.py` — 1 line: `from training.shared_evaluate import evaluate`
- `src/training/runners/next_eval.py` — 1 line: `from training.shared_evaluate import evaluate`

**Why:** These add indirection without value. The callers can import directly from `shared_evaluate`.

**Action:** Delete both files. Update imports in `category_cv.py` and `next_cv.py` to import from `training.shared_evaluate` directly.

### 3.4 Stale Root-Level Directories (MEDIUM priority)

| Directory | Contents | Action |
|-----------|----------|--------|
| `plans/` | `plan.md`, `todo.md`, `todo_article.md` | Move to `docs/notes/` or delete |
| `processing/` | Only `.DS_Store` | **Delete entirely** |
| `results_save/` | Old model results | Add to `.gitignore`, delete from git |
| `notebooks/` | 6 Jupyter notebooks + CLAUDE.md | Move useful ones to `experiments/archive/`, rest delete |
| `art/` | `comparison_report.tex`, `resume.pdf` | Move to `articles/` or delete |

### 3.5 Files That Should Not Be in Git (LOW priority)

| File | Action |
|------|--------|
| `.DS_Store` (root) | Remove from git, add to `.gitignore` |
| `processing/.DS_Store` | Remove from git |
| `.coverage` | Remove from git, add to `.gitignore` |
| `.idea/` | Already has `.gitignore`, verify no tracked files |

### 3.6 Documentation Referenced but Missing

The `CLAUDE.md` references three files that do not exist:
- `FUSION_GUIDE.md`
- `REFACTORING_SUMMARY.md`
- `walkthrough.md`

**Action:** Either create these files or remove the references from `CLAUDE.md`.

### 3.7 Empty `experiments/` Scaffolding

- `experiments/ablations/__init__.py` — Empty directory with only `__init__.py`
- `experiments/baselines/__init__.py` — Empty directory with only `__init__.py`

**Action:** Delete until actually needed. Empty scaffolding suggests future work that hasn't materialized.

---

## 4. Flows to Merge

### 4.1 Training Pipeline Flow — THREE LAYERS → ONE (HIGH priority)

**Current flow (3 layers of indirection):**
```
pipelines/train/mtl.pipe.py          (46 lines, thin wrapper)
  → subprocess.run(scripts/train.py)  (324 lines, CLI orchestrator)
    → training.runners.mtl_cv         (344 lines, actual logic)
```

**Problem:** The pipeline files are 95% identical (only `--task` flag differs). They add a subprocess layer that makes debugging harder, loses stack traces, and prevents IDE navigation.

**Recommended merge:**

```
# Option A: Direct CLI (preferred)
python scripts/train.py --task mtl --state alabama --engine hgi

# Option B: If pipeline wrappers are needed for backward compat,
# merge all 3 into a single pipelines/train.pipe.py
```

Delete `pipelines/train/mtl.pipe.py`, `cat_head.pipe.py`, `next_head.pipe.py`. Users call `scripts/train.py` directly or use experiment configs.

### 4.2 Single-Task CV Runners — EXTRACT BASE (HIGH priority)

**Current:** `category_cv.py` (109 lines) and `next_cv.py` (175 lines) share **~65% identical code**.

**Duplicated blocks:**
1. Model creation + device placement (identical)
2. Class weight computation + loss setup (identical)
3. Optimizer + scheduler setup (identical, already in helpers.py)
4. NeuralParams logging (95% identical — only task name differs)
5. FLOPs calculation (100% identical)
6. RunManifest writing (90% identical — only dataset_paths differs)
7. MPS cache clearing (identical)

**Recommended merge:** Extract a `_run_single_task_cv()` function in `training/runners/base_cv.py`:

```python
def run_single_task_cv(
    task_name: str,
    config: ExperimentConfig,
    folds: list,
    train_fn: Callable,    # category_trainer.train or next_trainer.train
    eval_fn: Callable,     # shared evaluate function
    results_path: Path | None,
    dataset_paths: dict,
) -> MLHistory:
    # All shared logic here (model init, weights, optimizer, scheduler,
    # NeuralParams, FLOPs, training loop, manifest writing)
```

Then `category_cv.py` and `next_cv.py` become ~20-line files that configure and call the base.

### 4.3 Single-Task Trainers — EXTRACT BASE (MEDIUM priority)

**Current:** `category_trainer.py` (116 lines) and `next_trainer.py` (156 lines) share **~90% identical training loops**.

**Unique to `next_trainer.py`:**
- Gradient norm tracking (1 line)
- Early stopping with patience (8 lines)
- Per-class F1 diagnostics (6 lines)

**Recommended merge:** Create `training/runners/base_trainer.py` with hooks:

```python
def train_single_task(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    epochs, history, device,
    on_batch_end=None,    # Hook for grad norm tracking
    early_stopping=None,  # Optional EarlyStopper
    compute_diagnostics=None,  # Hook for per-class F1
) -> dict:
```

### 4.4 Evaluation Chain — SIMPLIFY (MEDIUM priority)

**Current evaluation architecture (confusing):**
```
training/evaluate.py           → collect_predictions() + build_report() [DEAD CODE]
training/shared_evaluate.py    → evaluate() [used by single-task]
training/runners/mtl_eval.py   → evaluate_model() [used by MTL]
training/runners/category_eval.py → re-export of shared_evaluate [1 line]
training/runners/next_eval.py     → re-export of shared_evaluate [1 line]
```

**Recommended merge:**
1. Delete `evaluate.py` (dead)
2. Delete `category_eval.py` and `next_eval.py` (1-line re-exports)
3. Rename `shared_evaluate.py` → `evaluate.py` (single source of truth)
4. Keep `mtl_eval.py` for MTL-specific dual-task evaluation

**Result:** 2 files instead of 5.

### 4.5 Embedding Pipelines — EXTRACT SHARED PREAMBLE (LOW priority)

**Current:** All 6 `pipelines/embedding/*.pipe.py` files share identical preamble:
```python
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent
_src = str(_root / "src")
_research = str(_root / "research")
if _src not in sys.path:
    sys.path.insert(0, _src)
if _research not in sys.path:
    sys.path.insert(0, _research)
```

**Recommended:** Create `pipelines/_bootstrap.py` with the path setup, then each pipeline does `from _bootstrap import *` or just `import _bootstrap`.

---

## 5. Files and Folders to Merge

### 5.1 Folder Structure Cleanup

| Current | Proposed | Reason |
|---------|----------|--------|
| `plans/` | → `docs/notes/` | Scattered planning docs |
| `art/` | → `articles/` or delete | Only 2 files |
| `processing/` | Delete | Only contains .DS_Store |
| `notebooks/` | → `experiments/archive/` | Notebooks are experimental |
| `docs/notes/` | Keep | Already exists, consolidate here |
| `results_save/` | → `.gitignore` | Should not be in git |

### 5.2 Config File Consolidation

| Current | Proposed | Reason |
|---------|----------|--------|
| `src/configs/category_config.py` | Delete | Deprecated, 0 imports |
| `src/configs/next_config.py` | Delete | Deprecated, 0 imports |
| `src/configs/model.py` | Remove `MTLModelConfig` class | Deprecated, replaced by `ExperimentConfig` |
| `src/configs/embedding_fusion.py` | Keep | Active, fusion-specific |
| `src/configs/globals.py` | Keep | Active (DEVICE, CATEGORIES_MAP) |
| `src/configs/experiment.py` | Keep | Canonical config |
| `src/configs/paths.py` | Keep | Path management |

### 5.3 Training Module Consolidation

| Current (11 files) | Proposed (6 files) | Change |
|--------------------|--------------------|--------|
| `runners/category_cv.py` | `runners/single_task_cv.py` | Merge category + next CV |
| `runners/next_cv.py` | ↑ merged above | — |
| `runners/mtl_cv.py` | Keep (unique dual-task logic) | — |
| `runners/category_trainer.py` | `runners/single_task_trainer.py` | Merge trainers |
| `runners/next_trainer.py` | ↑ merged above | — |
| `runners/category_eval.py` | Delete (1-line re-export) | — |
| `runners/next_eval.py` | Delete (1-line re-export) | — |
| `runners/mtl_eval.py` | Keep | — |
| `runners/mtl_validation.py` | Keep | — |
| `evaluate.py` | Delete (dead code) | — |
| `shared_evaluate.py` | Rename → `evaluate.py` | Becomes canonical |
| `helpers.py` | Keep + extend | Add more shared logic |

**Net reduction:** 11 files → 6 files (45% fewer), with significantly less duplication.

---

## 6. Flow Improvements

### 6.1 Replace Print Statements with Logging (HIGH priority)

**Current:** 40+ `print()` calls across `src/` (26 in `data/inputs/fusion.py` alone).

**Problem:** No log levels, no file output, no structured logging, can't silence during tests.

**Action:**
```python
import logging
logger = logging.getLogger(__name__)

# In each module, replace:
print(f"Processing {state}...")
# With:
logger.info("Processing %s...", state)
```

Configure in `scripts/train.py`:
```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
```

### 6.2 Add a Makefile / Task Runner (MEDIUM priority)

**Current:** No standard way to discover how to run things. Users must read docs or pipeline scripts.

**Proposed `Makefile`:**
```makefile
.PHONY: test train lint clean

test:
    python -m pytest tests/ -v --tb=short

test-fast:
    python -m pytest tests/ -v --tb=short -x --ignore=tests/test_integration

train-mtl:
    python scripts/train.py --task mtl --state florida --engine hgi

train-category:
    python scripts/train.py --task category --state florida --engine poi2hgi

lint:
    ruff check src/ tests/
    ruff format --check src/ tests/

format:
    ruff format src/ tests/

clean:
    find . -type d -name __pycache__ -exec rm -rf {} +
    rm -f .coverage
```

### 6.3 Add a Linter (MEDIUM priority)

**Current:** No linter configured. No `ruff`, `flake8`, or `black` in `pyproject.toml`.

**Recommended:** Add `ruff` (fast, comprehensive):
```toml
[tool.ruff]
target-version = "py312"
line-length = 120
select = ["E", "F", "I", "W", "UP"]

[tool.ruff.lint.isort]
known-first-party = ["configs", "models", "losses", "data", "training", "tracking", "utils"]
```

### 6.4 Simplify ExperimentConfig Factory Methods (LOW priority)

**Current:** 3 factory methods with significant overlap (~35 duplicated lines).

**Proposed:** Single `_base_config()` + task-specific overrides:
```python
@classmethod
def _base(cls, name, state, engine, **overrides):
    defaults = dict(name=name, state=state, embedding_engine=engine,
                    batch_size=2048, learning_rate=1e-4, ...)
    defaults.update(overrides)
    return cls(**defaults)

@classmethod
def default_mtl(cls, name, state, engine):
    return cls._base(name, state, engine, task="mtl", epochs=50, ...)
```

### 6.5 Use `pyproject.toml` Entry Points (LOW priority)

**Current:** Users run `python scripts/train.py`. No installed commands.

**Proposed:** Add to `pyproject.toml`:
```toml
[project.scripts]
mtlnet-train = "scripts.train:main"
mtlnet-eval = "scripts.evaluate:main"
```

After `pip install -e .`, users can run: `mtlnet-train --task mtl --state florida --engine hgi`

### 6.6 Consolidate `.gitignore` (LOW priority)

**Current `.gitignore` is minimal (6 entries). Missing:**
```gitignore
# IDE
.idea/
.vscode/
*.iml

# OS
.DS_Store
Thumbs.db

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.coverage
.pytest_cache/
htmlcov/

# Project
/results/*
/results_save/*
/data/*
/output/*
/temp/
*.pt
*.pth
```

### 6.7 Deterministic Seed Management (LOW priority)

**Current:** `ExperimentConfig.seed` is set, but seeding is scattered:
- `FoldCreator` seeds `random.seed()`
- Integration tests seed `torch`, `numpy`, `random`
- No centralized `seed_everything()` function

**Proposed:** Add `utils/seed.py`:
```python
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

Call it once at the start of `scripts/train.py`.

### 6.8 CLAUDE.md Should Reflect Current Architecture

**Current `CLAUDE.md`** references:
- `src/model/mtlnet/` → Now `src/models/mtlnet.py`
- `src/criterion/` → Now `src/losses/`
- `src/common/ml_history/` → Now `src/tracking/`
- `src/common/calc_flops/` → Now `src/utils/flops.py`
- `src/etl/` → Now `src/data/`
- `src/train/` → Now `src/training/`
- `FUSION_GUIDE.md`, `REFACTORING_SUMMARY.md`, `walkthrough.md` → Don't exist

**Action:** Rewrite `CLAUDE.md` to match the post-refactoring architecture.

---

## 7. Prioritized Action Plan

### 🔴 Priority 1 — Delete Dead Code & Fix Correctness (1 day)

| # | Action | Files Affected | Risk |
|---|--------|---------------|------|
| 1.1 | Delete `src/configs/category_config.py` | 1 file | None (0 imports) |
| 1.2 | Delete `src/configs/next_config.py` | 1 file | None (0 imports) |
| 1.3 | Remove `MTLModelConfig` from `src/configs/model.py` | 1 file | Low |
| 1.4 | Delete `src/training/evaluate.py` (dead functions) | 1 file | None (0 call sites) |
| 1.5 | Delete `src/training/runners/category_eval.py` (1-line re-export) | 1 file | Low |
| 1.6 | Delete `src/training/runners/next_eval.py` (1-line re-export) | 1 file | Low |
| 1.7 | Delete `processing/` directory (only .DS_Store) | 1 dir | None |
| 1.8 | Remove `.DS_Store` and `.coverage` from git | 3 files | None |
| 1.9 | Update `CLAUDE.md` to reflect current architecture | 1 file | None |
| 1.10 | Remove references to non-existent docs from CLAUDE.md | 1 file | None |

### 🟡 Priority 2 — Reduce Duplication (2-3 days)

| # | Action | Impact |
|---|--------|--------|
| 2.1 | Merge 3 training pipeline wrappers → delete, use `scripts/train.py` directly | -138 lines, -3 files |
| 2.2 | Extract `_run_single_task_cv()` base function in helpers or new `base_cv.py` | -80 lines duplication |
| 2.3 | Merge `category_trainer.py` + `next_trainer.py` → parameterized `single_task_trainer.py` | -70 lines, -1 file |
| 2.4 | Simplify evaluation chain (rename `shared_evaluate.py` → `evaluate.py`) | -2 files |
| 2.5 | Simplify `ExperimentConfig` factory methods with `_base()` pattern | -30 lines |

### 🟢 Priority 3 — Developer Ergonomics (1-2 days)

| # | Action | Impact |
|---|--------|--------|
| 3.1 | Replace 40+ `print()` calls with `logging` | Structured logging |
| 3.2 | Add `Makefile` with common commands | Discoverability |
| 3.3 | Expand `.gitignore` (IDE, OS, coverage, checkpoints) | Clean repo |
| 3.4 | Add `ruff` linter to `pyproject.toml` | Code quality |
| 3.5 | Add `utils/seed.py` with `seed_everything()` | Reproducibility |
| 3.6 | Create `pipelines/_bootstrap.py` for shared sys.path setup | -36 lines duplication |
| 3.7 | Delete empty `experiments/ablations/` and `experiments/baselines/` scaffolding | Clean tree |

### 🔵 Priority 4 — Nice-to-Have (Optional)

| # | Action | Impact |
|---|--------|--------|
| 4.1 | Add `pyproject.toml` entry points for CLI | `mtlnet-train` command |
| 4.2 | Move `plans/`, `art/` into `docs/notes/` or `articles/` | Fewer root dirs |
| 4.3 | Archive `notebooks/` content to `experiments/archive/` | Clean root |
| 4.4 | Extend loss registry to single-task runners (currently hardcoded `nn.CrossEntropyLoss`) | Consistency |
| 4.5 | Add pre-commit hooks (ruff + isort) | Automated quality |
| 4.6 | Write the referenced `FUSION_GUIDE.md` and `walkthrough.md` | Documentation |

---

## Appendix: Detailed File-Level Analysis

### Training Runners Duplication Matrix

| Code Block | `category_cv` | `next_cv` | `mtl_cv` | Deduplicated? |
|------------|:---:|:---:|:---:|:---:|
| Model creation + `.to(device)` | ✓ | ✓ | ✓ | ❌ |
| `compute_class_weights()` | ✓ | ✓ | ✓ | ✅ (via helpers.py) |
| `nn.CrossEntropyLoss` setup | ✓ | ✓ | ✓ | ❌ |
| `setup_optimizer()` | ✓ | ✓ | ✓ | ✅ (via helpers.py) |
| `setup_scheduler()` | ✓ | ✓ | ✓ | ✅ (via helpers.py) |
| `NeuralParams` logging | ✓ | ✓ | ✓ | ❌ |
| FLOPs calculation | ✓ | ✓ | ✓ | ❌ |
| MPS cache clearing | ✓ | ✓ | ✓ | ❌ |
| `RunManifest` writing | ✓ | ✓ | ✓ | ❌ |
| Training loop | ✗ (delegates) | ✗ (delegates) | ✓ (inline) | N/A |

### Current vs Proposed File Count

| Module | Current Files | Proposed Files | Reduction |
|--------|:---:|:---:|:---:|
| `training/runners/` | 9 | 5 | -4 |
| `training/` (top-level) | 3 | 2 | -1 |
| `configs/` | 7 | 5 | -2 |
| `pipelines/train/` | 3 | 0 | -3 |
| **Total** | **22** | **12** | **-10** |
