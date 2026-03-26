# MTLnet Refactoring — Decisions Log

> Append-only. Every executor phase reads this at startup and appends before reporting done.
> Format: `[Phase] [Item] [Decision] [Rationale] [Alternatives rejected] [Files affected]`
> The coordinator references this file by path in every phase kickoff brief.

---

## How to append

At the end of each phase, add a block:

```
### Phase N — <phase name>
Date: YYYY-MM-DD

[N.x] <item name>
Decision: <what was chosen>
Rationale: <why>
Alternatives rejected: <what else was considered>
Files affected: <list>
Verification: <command that proves it>
```

---

## Decisions

### Phase 0 — Safety Net, pyproject, and Hard Decisions
Date: 2026-03-26

---

#### [0.1] MTL split protocol — decision

**Decision:** Adopt user-level isolation via `StratifiedGroupKFold(groups=userid, y=next_category)` as described in `SPLIT_PROTOCOL.md`. User isolation is the hard invariant. Ambiguous POIs go to category training only. Residual cross-task POI overlap is quantified per fold, not eliminated.

**Rationale:** The current implementation (`create_fold.py:461-462`) creates two independent `StratifiedKFold` splits — one for next-task data (stratified on `next_category`) and one for category data (stratified on `category`). These splits are zipped together per fold. A POI can appear in category training and next-task validation (or vice versa) because the splits are independent. Since the shared MTLnet backbone sees that POI during training via one task, information leaks into the other task's validation. User-level splits are the defensible fix because: (a) POI embeddings are pre-computed and shared by construction, so POI-level isolation doesn't prevent the model from seeing the same embedding vectors; (b) the primary leakage risk is user-level behavioral pattern memorization; (c) the next-task schema drops userid during input generation (`create_fold.py:241`), so implementing the new protocol requires retaining userid through Phase 2.

**Current state (documented):**
- `src/etl/create_fold.py:34` — imports only `StratifiedKFold`, not `StratifiedGroupKFold`
- `src/etl/create_fold.py:413` — single-task: `StratifiedKFold` on labels (correct for single-task)
- `src/etl/create_fold.py:461-462` — MTL: two independent `StratifiedKFold` splits zipped together (the problematic pattern)
- `src/etl/create_fold.py:241` — `userid` dropped from next-task data before fold creation
- No `StratifiedGroupKFold` usage anywhere in the codebase

**Alternatives rejected:**
1. *Bidirectional POI isolation* — would require per-sequence POI identity checks, but next-task schema doesn't retain POI IDs after embedding lookup. Would also sacrifice user isolation for ambiguous POIs.
2. *Keep independent splits* — silently accepts cross-task leakage through the shared backbone. Not defensible for publication.
3. *Sample-level isolation* — weaker than user-level; a user's behavioral patterns could still leak across train/val.

**Files affected (Phase 2 implementation, not Phase 0):**
- `src/etl/create_fold.py` — replace MTL `StratifiedKFold` with `StratifiedGroupKFold`
- `src/etl/create_fold.py:241` — retain `userid` column
- `src/etl/mtl_input/` — retain `userid` in next-task output
- `pipelines/create_inputs.pipe.py` — pass userid through
- New: `src/etl/poi_user_mapping.py` — materialize POI→users mapping
- New: split manifest artifact

**Verification:**
```
grep -r "StratifiedKFold\|StratifiedGroupKFold" src/etl/ --include="*.py" -l
# Current output: src/etl/create_fold.py (only StratifiedKFold)
```

**Phase 1 must know:** No code changes for splits in Phase 1. The split decision is frozen. Phase 2 implements `StratifiedGroupKFold`. Until then, single-task pipelines are unaffected (they correctly use `StratifiedKFold` on labels). MTL results produced before Phase 2 carry the independent-split caveat.

---

#### [0.2] pyproject.toml

**Decision:** Created `pyproject.toml` with exact-pinned dependencies from `.venv_new` (Python 3.12.11, pip freeze snapshot 2026-03-26). Local package `graphgps` noted as manual install.

**Rationale:** No dependency manifest existed. Multiple stale requirements files (`requirements-bckp.txt`, `requirements_upgrad.txt`) were unreliable. Exact pins ensure reproducibility.

**Alternatives rejected:**
1. *Range-pinned deps (>=, <)* — less reproducible for research.
2. *poetry.lock* — adds tooling dependency; tomllib is stdlib.

**Files affected:** `pyproject.toml` (new)
**Verification:** `python -c "import tomllib; tomllib.loads(open('pyproject.toml').read())"`

---

#### [0.3] Regression fixtures

**Decision:** Three regression fixtures (category, next, MTL) in `tests/test_regression/`. Three layers each: shape/params (exact), dataloader structure (exact), F1 macro (calibrated floor). All tests force CPU for determinism.

**Calibration (5 runs, seed=42, CPU, 10 epochs, torch==2.9.1):**
- Category: F1=0.9492 ±0.0 → floor=0.94
- Next: F1=1.0000 ±0.0 → floor=0.99
- MTL category: F1=0.9286 ±0.0 → floor=0.92
- MTL next: F1=1.0000 ±0.0 → floor=0.99

CPU determinism produced std=0 across all 5 runs. Floors set 1-2% below observed mean as PyTorch version margin.

**Alternatives rejected:**
1. *MPS-based tests* — non-deterministic, would need much wider tolerance.
2. *Real data fixtures* — require data files, not portable across machines.

**Files affected:** `tests/test_regression/__init__.py`, `tests/test_regression/test_regression.py` (new)
**Verification:** `pytest tests/test_regression/ -v`

---

#### [0.4] CLI flags on pipeline scripts

**Decision:** Added `--state` and `--engine` argparse flags to `mtl.pipe.py`, `cat_head.pipe.py`, `next_head.pipe.py`. Both accept multiple values (`nargs="+"`). Defaults match the previously-hardcoded active configs.

**Rationale:** Experiments were defined by editing source code. CLI flags enable scripted sweeps without code edits.

**Alternatives rejected:**
1. *Config file (YAML/JSON)* — premature; `ExperimentConfig` planned for Phase 3.
2. *Environment variables* — less discoverable than `--help`.

**Files affected:** `pipelines/train/mtl.pipe.py`, `pipelines/train/cat_head.pipe.py`, `pipelines/train/next_head.pipe.py`
**Verification:** `python pipelines/train/mtl.pipe.py --help`

---

#### [0.5] Split feasibility report — scope and format

**Decision:** Produced a doc-only feasibility analysis (see below in Phase 0 exit report). No training code. The report identifies:
- Current split behavior (independent `StratifiedKFold`)
- Expected impact of user-isolation switch
- Data requirements for Phase 2 implementation (userid retention, POI→users mapping)
- Feasibility constraints per dataset configuration

**Rationale:** SPLIT_PROTOCOL.md Section 10 defines `FeasibilityReport` schema. Phase 0 produces the analysis; Phase 2 implements the code that generates the machine-readable artifact.

**Files affected:** `docs/DECISIONS.md` (this entry + [0.5a] below), `scripts/generate_feasibility_report.py` (new), `docs/feasibility_report_{state}.json` (new, 4 files)
**Verification:**
```
python scripts/generate_feasibility_report.py
python -c "import json; [json.load(open(f'docs/feasibility_report_{s}.json')) for s in ['alabama','florida','california','texas']]; print('OK')"
```

---

#### [0.5a] Split feasibility analysis — per-dataset results

**Method:** Simulated `StratifiedGroupKFold(n_splits=5, groups=userid, y=dominant_category, random_state=42)` on raw checkin data. Classified POIs as train-exclusive, val-exclusive, or ambiguous per fold. Val-exclusive POIs are what category validation would contain under strict user isolation.

**Default thresholds from SPLIT_PROTOCOL.md:**
- `min_category_val_fraction`: 5% of total POIs
- `min_class_count`: 5
- `min_class_fraction`: 3%

---

**Alabama** (3,855 users, 11,706 POIs, 113K checkins)

| Metric | Fold range |
|--------|-----------|
| POI overlap (ambiguous) | 41.6% – 47.0% |
| Cat-val fraction (val-exclusive) | 4.7% – 7.7% |
| Per-category min fraction | 4.7% (Entertainment) |

Assessment: **Borderline feasible under strict mode.** Some folds dip below 5% overall; Entertainment approaches 3% floor. Relaxation may be needed for 1-2 folds. Recommended: `min_category_val_fraction=0.04`, or enable `split_relaxation=True` as fallback. Low POI overlap (41-47%) means residual leakage is moderate.

---

**Florida** (21,005 users, 74,862 POIs, 1.4M checkins)

| Metric | Fold range |
|--------|-----------|
| POI overlap (ambiguous) | 54.5% – 59.9% |
| Cat-val fraction (val-exclusive) | 3.3% – 4.0% |
| Categories below 3% | Entertainment (2.9%), Food (2.4%), Nightlife (2.0%), Travel (3.0%) |

Assessment: **Fails strict mode with default thresholds.** All folds below 5% overall. Four categories below 3%. Relaxation required: moving ambiguous POIs with high val-user ratio into category validation. Recommended: `split_relaxation=True`, `min_category_val_fraction=0.03`. High POI overlap (55-60%) means significant ambiguous-POI pool available for relaxation.

---

**California** (36,970 users, 165,881 POIs, 3.2M checkins)

| Metric | Fold range |
|--------|-----------|
| POI overlap (ambiguous) | 58.3% – 62.1% |
| Cat-val fraction (val-exclusive) | 3.1% – 3.8% |
| Categories below 3% | Entertainment (2.9%), Food (2.3%), Nightlife (2.5%), Outdoors (2.5%) |

Assessment: **Fails strict mode.** Similar to Florida. Relaxation required. Large ambiguous pool (~60%) provides ample candidates. Recommended: `split_relaxation=True`, `min_category_val_fraction=0.03`.

---

**Texas** (38,454 users, 155,208 POIs, 4.0M checkins)

| Metric | Fold range |
|--------|-----------|
| POI overlap (ambiguous) | 60.2% – 69.0% |
| Cat-val fraction (val-exclusive) | 2.7% – 3.6% |
| Categories below 3% | Food (2.2%), Nightlife (2.2%) |

Assessment: **Fails strict mode.** Highest overlap of all states. Relaxation required, and even with relaxation, Food and Nightlife categories are thin. Recommended: `split_relaxation=True`, `min_category_val_fraction=0.025`. Researcher should justify that 2-3% per-category validation is acceptable for these categories.

---

**Summary and recommendations for Phase 2:**

1. **Alabama** is feasible under strict mode with threshold adjustment (`min_category_val_fraction=0.04`).
2. **Florida, California, Texas** require `split_relaxation=True`. The relaxation protocol in SPLIT_PROTOCOL.md (threshold schedule [0.8, 0.6, 0.5]) should be sufficient given the large ambiguous-POI pools (55-69%).
3. **Per-category floors** should be lowered to `min_class_fraction=0.02` for larger states, or the researcher must accept that some categories (Food, Nightlife) will have thin validation sets.
4. The high POI overlap (40-69%) is intrinsic to the data — most POIs are visited by multiple users. This is the residual cross-task leakage channel that must be quantified in the split manifest.
5. Phase 2 implementation prerequisites: (a) retain `userid` in next-task input, (b) materialize `POI→set(userids)` mapping, (c) implement the relaxation protocol, (d) generate machine-readable `FeasibilityReport` artifacts.

---

### Phase 1 — Low-Risk BPR Cleanup
Date: 2026-03-26

---

#### [1.1] paths.py — import-time side effects removed

**Decision:** Removed `FileNotFoundError` raise for missing checkins dir and `mkdir` calls for `OUTPUT_DIR`/`RESULTS_ROOT` from module-level. Moved into `IoPaths.validate()` classmethod. Also removed unused `from IPython.lib.deepreload import get_parent` and `import urllib3`.

**Rationale:** On a fresh checkout without data, `import configs.paths` raised `FileNotFoundError`, blocking all downstream imports regardless of whether the caller needed I/O. Side-effect-free imports are required for testing without data and for the `from etl.mtl_input import core` DoD test.

**Alternatives rejected:**
1. *Try/except* — silences a real misconfiguration at use time. `validate()` makes the check explicit and caller-controlled.

**Files affected:** `src/configs/paths.py`
**Verification:** `PYTHONPATH=src python -c "from configs.paths import IoPaths"`

---

#### [1.2] mtl_input/__init__.py — lazy imports

**Decision:** Replaced all eager `from .core import ...`, `from .loaders import ...`, `from .builders import ...`, `from .fusion import ...` with a module-level `__getattr__` that lazily imports on first attribute access. Added `_LAZY_IMPORTS` dict mapping each public name to its origin module + attr.

**Rationale:** Importing `from etl.mtl_input import core` previously triggered pandas, numpy, tqdm, and paths imports even if only the pure-logic `core` submodule was needed. Lazy loading eliminates that startup cost and decouples consumers of pure functions from I/O-heavy modules.

**Alternatives rejected:**
1. *Keep eager imports after paths.py fix* — still loads pandas/tqdm on every import of the package.
2. *`importlib.util.LazyLoader`* — affects the whole module, harder to read.

**Files affected:** `src/etl/mtl_input/__init__.py`
**Verification:** `PYTHONPATH=src python -c "from etl.mtl_input import core; print('OK')"`

---

#### [1.3] Shared evaluate() — merge identical implementations

**Decision:** Confirmed both `src/train/category/evaluation.py` and `src/train/next/evaluation.py` were byte-for-byte identical. Created `src/train/shared/evaluate.py` as the single canonical implementation. Replaced both originals with one-line re-export shims: `from train.shared.evaluate import evaluate  # noqa: F401`.

**Rationale:** Two identical files diverge silently under future edits. The shim preserves the existing import API for all callers without modification.

**Alternatives rejected:**
1. *Update all callers to import from shared directly* — unnecessary churn for no benefit.
2. *Merge into one of the two originals* — which one to keep is arbitrary and still leaves one dead file.

**Files affected:** `src/train/shared/__init__.py` (new), `src/train/shared/evaluate.py` (new), `src/train/category/evaluation.py` (shim), `src/train/next/evaluation.py` (shim)
**Verification:** `grep -rn "from train.shared.evaluate import" src/`

---

#### [1.4] validation_model() — dead code deletion

**Precondition grep:** `grep -rn "validation_model\b" src/` → single definition in `validation.py`, zero call sites.

**Decision (BPR* path taken: dead code):** Deleted `validation_model` and `validation_model_by_head` from `src/train/mtlnet/validation.py`. Both were defined but never called anywhere in `src/` or the pipelines. Also deleted `evaluate_model_by_head` from `src/train/mtlnet/evaluate.py` for the same reason (dead code, and was the only carrier of the `foward_method` typo other than the docstring).

**Rationale:** Dead code with a known scoping bug (`torch.no_grad()` context manager at lines 69-74 of the old `validation_model` did not cover the loop at line 76 — the loop ran without gradient isolation). Deleting is safer than fixing a function that is never invoked.

**Files affected:** `src/train/mtlnet/validation.py`, `src/train/mtlnet/evaluate.py`
**Verification:** `grep -rn "validation_model\b" src/` → definition only in deleted block (zero results post-delete)

---

#### [1.5] CategoryHeadMTL — alias to CategoryHeadEnsemble

**Decision:** Added `x = x.squeeze(1)` to `CategoryHeadEnsemble.forward()` in `category_head_enhanced.py` (safe no-op for `[B, D]` inputs with `D > 1`). Replaced `CategoryHeadMTL` class body in `category_head.py` with `CategoryHeadMTL = CategoryHeadEnsemble`.

**Rationale:** `CategoryHeadMTL` and `CategoryHeadEnsemble` were identical architectures. The only behavioral difference was that `CategoryHeadMTL.forward()` called `x.squeeze(1)` to handle `[B, 1, D]` inputs from the MTL shared backbone. Absorbing the squeeze into `CategoryHeadEnsemble` unifies the two without breaking standalone category training (where inputs are `[B, D]`).

**Alternatives rejected:**
1. *Subclass* — adds a class just to override one line.
2. *Delete CategoryHeadMTL without alias* — would require updating `mtl_poi.py` and any imports.

**Files affected:** `src/model/category/category_head_enhanced.py`, `src/model/mtlnet/category_head.py`
**Verification:** `pytest tests/test_regression/ -v` (regression suite covers MTL forward pass)

---

#### [1.6] foward_method typo — fixed

**Decision:** Fixed stale `foward_method` in `evaluate_model`'s docstring in `src/train/mtlnet/evaluate.py`. The other instances were in `evaluate_model_by_head` (dead code deleted in [1.4]).

**Rationale:** Typo in public docstring of a live function. DoD requires zero grep results.

**Files affected:** `src/train/mtlnet/evaluate.py`
**Verification:** `grep -rn "foward_method" src/` → zero source results (stale `.pyc` excluded)

---

#### [1.7] Stale requirements files — deleted

**Decision:** Deleted `requirements-bckp.txt` and `requirements_upgrad.txt`. Kept `requirements.txt` and `requirements_colab.txt`.

**Rationale:** Both deleted files were superseded by `pyproject.toml` (created in Phase 0). `requirements.txt` remains as it may be used by CI. `requirements_colab.txt` is a distinct environment file.

**Files affected:** `requirements-bckp.txt` (deleted), `requirements_upgrad.txt` (deleted)

---

#### [1.8] Legacy notebooks — archived

**Decision:** Moved `notebooks/create_inputs_hgi.py` and `notebooks/hgi_texas.py` to `experiments/archive/`. Created `experiments/archive/` directory.

**Rationale:** These are legacy scripts (not notebooks) that pre-date the modular `src/etl/mtl_input/` system. Archiving preserves history without cluttering the active notebooks directory.

**Files affected:** `notebooks/create_inputs_hgi.py` → `experiments/archive/`, `notebooks/hgi_texas.py` → `experiments/archive/`

---

#### [1.9] ANALYSIS.md / UPDATE.md — no action

**Decision:** Neither file exists at repo root. Skip.

---

#### [1.10] hmrm.py vs hmrm_new.py — resolved

**Precondition grep:** `grep -rn "from.*hmrm\|import.*hmrm" src/` → only `create_hmrm.py:5: from embeddings.hmrm.hmrm_new import HmrmBaselineNew`. `hmrm.py` had zero import references (dead code).

**Decision:** Deleted old `hmrm.py`, renamed `hmrm_new.py` → `hmrm.py`, updated `create_hmrm.py` import to `from embeddings.hmrm.hmrm import HmrmBaselineNew`.

**Rationale:** Eliminates the `_new` suffix ambiguity. The old `hmrm.py` contained an obsolete `Optimizer` class that was never imported.

**Alternatives rejected:**
1. *Keep both, add `__all__`* — leaves dead code in place.

**Files affected:** `src/embeddings/hmrm/hmrm.py` (replaced), `src/embeddings/hmrm/hmrm_new.py` (deleted), `src/embeddings/hmrm/create_hmrm.py`
**Verification:** `python -c "from embeddings.hmrm.hmrm import HmrmBaselineNew"`

