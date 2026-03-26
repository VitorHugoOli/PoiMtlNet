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
**Verification:** `PYTHONPATH=src python -c "from configs.paths import IoPaths; print('OK')"`

---

#### [1.2] mtl_input/__init__.py — lazy imports

**Decision:** Replaced all eager `from .core import ...`, `from .loaders import ...`, `from .builders import ...`, `from .fusion import ...` with a module-level `__getattr__` that lazily imports on first attribute access. Added `_LAZY_IMPORTS` dict mapping each public name to its origin module + attr.

**Rationale:** Importing `from etl.mtl_input import core` previously triggered pandas, numpy, tqdm, and paths imports even if only the pure-logic `core` submodule was needed. Lazy loading eliminates that startup cost and decouples consumers of pure functions from I/O-heavy modules.

**Alternatives rejected:**
1. *Keep eager imports after paths.py fix* — still loads pandas/tqdm on every import of the package.
2. *`importlib.util.LazyLoader`* — affects the whole module, harder to read.

**Files affected:** `src/etl/mtl_input/__init__.py`
**Verification:** `PYTHONPATH=src python -c "from etl.mtl_input import core; print('OK:', core.__name__)"`

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
**Verification:** `ls requirements*.txt` → only `requirements.txt` and `requirements_colab.txt` remain

---

#### [1.8] Legacy notebooks — archived

**Decision:** Moved `notebooks/create_inputs_hgi.py` and `notebooks/hgi_texas.py` to `experiments/archive/`. Created `experiments/archive/` directory.

**Rationale:** These are legacy scripts (not notebooks) that pre-date the modular `src/etl/mtl_input/` system. Archiving preserves history without cluttering the active notebooks directory.

**Files affected:** `notebooks/create_inputs_hgi.py` → `experiments/archive/`, `notebooks/hgi_texas.py` → `experiments/archive/`
**Verification:** `ls experiments/archive/` → `create_inputs_hgi.py  hgi_texas.py`

---

#### [1.9] ANALYSIS.md / UPDATE.md — no action

**Decision:** Neither file exists at repo root. Skip.
**Verification:** `ls {ANALYSIS,UPDATE}.md 2>/dev/null || echo "neither exists at root"` → neither exists at root

---

#### [1.10] hmrm.py vs hmrm_new.py — resolved

**Precondition grep:** `grep -rn "from.*hmrm\|import.*hmrm" src/` → only `create_hmrm.py:5: from embeddings.hmrm.hmrm_new import HmrmBaselineNew`. `hmrm.py` had zero import references (dead code).

**Decision:** Deleted old `hmrm.py`, renamed `hmrm_new.py` → `hmrm.py`, updated `create_hmrm.py` import to `from embeddings.hmrm.hmrm import HmrmBaselineNew`.

**Rationale:** Eliminates the `_new` suffix ambiguity. The old `hmrm.py` contained an obsolete `Optimizer` class that was never imported.

**Alternatives rejected:**
1. *Keep both, add `__all__`* — leaves dead code in place.

**Files affected:** `src/embeddings/hmrm/hmrm.py` (replaced), `src/embeddings/hmrm/hmrm_new.py` (deleted), `src/embeddings/hmrm/create_hmrm.py`
**Verification (import-free, numba not installed in this env):**
```
grep "from embeddings.hmrm.hmrm import" src/embeddings/hmrm/create_hmrm.py
# → from embeddings.hmrm.hmrm import HmrmBaselineNew
ls src/embeddings/hmrm/hmrm_new.py 2>/dev/null || echo "hmrm_new.py absent"
# → hmrm_new.py absent
```

---

### Phase 2 — Data Contract Hardening
Date: 2026-03-26

**Note on verification commands:** All `PYTHONPATH=src python` commands below
assume CWD is the worktree root. For commands that need real data, set
`OUTPUT_DIR` and `DATA_ROOT` to point at the main repo's `output/` and `data/`
directories (e.g. `OUTPUT_DIR=/path/to/ingred/output DATA_ROOT=/path/to/ingred/data`).

---

#### [2.1] Parquet column schemas (BPR)

**Decision:** Created `src/data/schemas.py` with parameterized column builders (`category_columns()`, `next_columns()`, `sequence_columns()`, `poi_user_mapping_columns()`) and default schemas (`CATEGORY_SCHEMA`, `NEXT_SCHEMA`, `SEQUENCE_SCHEMA`). Added `validate_dataframe()` for contract enforcement.

**Rationale:** Column structures were implicit (derived from code reading). Explicit schemas enable validation at pipeline boundaries and documentation of data contracts.

**Files affected:** `src/data/__init__.py` (new), `src/data/schemas.py` (new)
**Verification:**
```
PYTHONPATH=src python -c "from data.schemas import NEXT_SCHEMA, CATEGORY_SCHEMA; print('ok')"
# → ok
```

---

#### [2.2] Embedding-level validation for category task (SC)

**Decision:** Added guard in `generate_category_input()` that rejects check-in-level engines (`TIME2VEC`, `CHECK2HGI`). Category task requires one embedding per POI; check-in-level engines produce one per visit.

**Expected impact:** No metric impact on valid configurations. Prevents silent data corruption when misconfigured.
**Observed impact:** Confirmed: raises `ValueError` for TIME2VEC.

**Files affected:** `src/etl/mtl_input/builders.py`
**Verification:**
```
PYTHONPATH=src python -c "
from etl.mtl_input.builders import generate_category_input
from configs.paths import EmbeddingEngine
try:
    generate_category_input('x', EmbeddingEngine.TIME2VEC)
    print('FAIL')
except ValueError as e:
    print('OK:', e)
"
# → OK: Engine time2vec produces check-in-level embeddings ...
```

---

#### [2.3] FoldCreator reads dimensions from artifact (BPR*)

**Precondition:** Verified that category parquet contains numeric columns `['0'...'63']` (embedding_dim=64) and next parquet contains `['0'...'575']` (576 = 9×64). Dimensions are fully inferrable from the artifact.

**Decision:** `load_category_data()`, `load_next_data()`, and `_convert_to_tensors()` now infer `embedding_dim` from numeric columns in the artifact instead of reading `InputsConfig.EMBEDDING_DIM`. `FoldConfig.embedding_dim` is set from artifact at runtime. Also fixed `generate_next_input_from_poi()` in builders.py to infer dimension from the embeddings DataFrame.

**Rationale:** Hardcoded `EMBEDDING_DIM=64` fails silently when fusion mode produces different dimensions. Reading from artifact ensures consistency.

**Files affected:** `src/etl/create_fold.py`, `src/etl/mtl_input/builders.py`
**Verification:**
```
grep -rn "EMBEDDING_DIM" src/etl/
# → (no output — zero matches)
```

---

#### [2.4] Configurable sequence stride (BPR)

**Decision:** Added `stride` parameter to `generate_sequences()` with default `None` (falls back to `window_size`, preserving non-overlapping behavior). When set to a value < window_size, produces overlapping sequences. History always uses `window_size` items; `stride` only controls the step between starts.

**Rationale:** Enables future experiments with overlapping windows without changing the default behavior.

**Files affected:** `src/etl/mtl_input/core.py`
**Verification:**
```
PYTHONPATH=src python -c "
from etl.mtl_input.core import generate_sequences
seqs = generate_sequences(list(range(10,20)), window_size=4, stride=1)
assert all(len(s)==5 for s in seqs), 'wrong length'
print(f'stride=1: {len(seqs)} seqs, first={seqs[0]}, all len 5')
seqs2 = generate_sequences(list(range(10,20)), window_size=4)
seqs3 = generate_sequences(list(range(10,20)), window_size=4, stride=4)
assert seqs2 == seqs3, 'default != stride=window_size'
print(f'default: {len(seqs2)} seqs, matches stride=4: True')
"
# → stride=1: 9 seqs, first=[10, 11, 12, 13, 14], all len 5
# → default: 3 seqs, matches stride=4: True
```

---

#### [2.5] Retain userid as first-class column in next-task input (SC)

**Decision:** `load_next_data()` now returns `(X, y, userids, embedding_dim)` instead of dropping userid silently. `load_category_data()` now returns `(X, y, placeids, embedding_dim)`. Both ensure integer types for IDs (`userid.astype(int)` to handle string storage in parquet).

**Expected impact:** No metric impact. Enables user-level splits in [2.8].
**Observed impact:** Regression tests pass. userid is preserved as metadata array alongside feature tensors.

**Files affected:** `src/etl/create_fold.py`
**Verification:**
```
grep -rn "userid" src/etl/mtl_input/ | grep -c "userid"
# → >0 (userid referenced throughout core.py, builders.py, fusion.py)
```

---

#### [2.6] POI→users mapping artifact (BPR)

**Decision:** Created `src/data/poi_user_mapping.py` with `build_poi_user_mapping()` and `load_poi_user_mapping()`. Materializes POI→set(userids) from raw checkins as JSON artifact at `output/{engine}/{state}/input/poi_user_mapping.json`.

**Rationale:** Required by split protocol (SPLIT_PROTOCOL.md §3) to classify POIs as train-exclusive, val-exclusive, or ambiguous per fold.

**Files affected:** `src/data/poi_user_mapping.py` (new)
**Verification:**
```
PYTHONPATH=src python -c "
import json; from data.poi_user_mapping import _get_mapping_path
from configs.paths import EmbeddingEngine
p = _get_mapping_path('alabama', EmbeddingEngine.HGI); print(p)
m = json.load(open(p)); print(f'entries: {len(m)}')
"
# → .../output/hgi/alabama/input/poi_user_mapping.json
# → entries: 11706
```

---

#### [2.7] Sequence→POI mapping artifact (BPR*)

**Precondition:** Verified intermediate `sequences_next.parquet` exists with columns `[poi_0..poi_8, target_poi, userid]` — sufficient to build the mapping.

**Decision:** Created `src/data/sequence_poi_mapping.py` with `build_sequence_poi_mapping()` and `load_sequence_poi_mapping()`. Maps each next-task sample row to its set of POI IDs. Saved as JSON at `output/{engine}/{state}/input/sequence_poi_mapping.json`. Wired into `_create_mtl_folds()`: called alongside `build_poi_user_mapping()`, result passed to `_compute_overlap_seq_fraction()`.

**Rationale:** Required for computing overlap diagnostics (SPLIT_PROTOCOL.md §9) — which val sequences touch ambiguous POIs. Decouples FoldCreator from raw checkins.

**Files affected:** `src/data/sequence_poi_mapping.py` (new), `src/etl/create_fold.py` (wired in)
**Verification:**
```
PYTHONPATH=src python -c "
import json; from data.sequence_poi_mapping import _get_mapping_path
from configs.paths import EmbeddingEngine
p = _get_mapping_path('alabama', EmbeddingEngine.HGI); print(p)
m = json.load(open(p)); print(f'entries: {len(m)}')
"
# → .../output/hgi/alabama/input/sequence_poi_mapping.json
# → entries: 12699
```

---

#### [2.8] MTL split protocol implementation (SC)

**Decision:** Replaced independent `StratifiedKFold` with `StratifiedGroupKFold(groups=userid, y=next_category)` in `_create_mtl_folds()`. Implements SPLIT_PROTOCOL.md exactly:
- Step 1: User-level split on next-task data
- Step 2: POI classification using poi_users mapping
- Step 3: Category fold = train-exclusive + ambiguous → train, val-exclusive → val
- Step 4: Next fold = train-user sequences → train, val-user sequences → val

**Expected impact:** Category validation sets smaller (val-exclusive POIs only). No cross-task user leakage. Residual POI overlap quantified per fold.
**Observed impact (Alabama, HGI, seed=42):**
- Fold 0: 1297 train users, 326 val users | 5902 TE, 691 VE, 4937 AMB (42.2% overlap)
- Category val: 691 POIs (5.9% of total) — above 5% threshold
- All 5 folds healthy, overlap 37-50% consistent with feasibility analysis

**Alternatives rejected:**
1. *Keep independent StratifiedKFold* — cross-task leakage through shared backbone (frozen in [0.1])
2. *Bidirectional POI isolation* — breaks user isolation (rejected in [0.1])

**Files affected:** `src/etl/create_fold.py` (import `StratifiedGroupKFold`, rewrite `_create_mtl_folds`)
**Verification:**
```
grep -n "StratifiedGroupKFold" src/etl/create_fold.py
# → line 35: from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
# → line 547: sgkf = StratifiedGroupKFold(...)
```

---

#### [2.9] Split manifest JSON per fold (BPR)

**Decision:** `FoldCreator.save_split_manifests(output_dir)` emits `split_manifest_fold{i}.json` per fold. Each manifest contains: user assignments, POI classification (train-exclusive/val-exclusive/ambiguous), sample counts, overlap diagnostics (ambiguous POI count, fraction, affected sequence fraction), seed, split mode.

**Rationale:** SPLIT_PROTOCOL.md §9 requires a materialized split artifact for reproducibility and governance.

**Files affected:** `src/etl/create_fold.py` (added `save_split_manifests()`, `_json_default()`)
**Verification:**
```
PYTHONPATH=src python -c "
from configs.paths import EmbeddingEngine, IoPaths
p = IoPaths.get_next('alabama', EmbeddingEngine.HGI).parent
import glob; files = sorted(glob.glob(str(p / 'split_manifest_fold*.json')))
print(f'{len(files)} manifests'); [print(f'  {f}') for f in files]
"
# → 5 manifests
# →   .../split_manifest_fold0.json
# →   ... (fold1-fold4)
```

---

### Phase 3 — Configuration Unification
Date: 2026-03-26

---

#### [3.1] ExperimentConfig dataclass created

**Decision:** Created `src/configs/experiment.py` with a flat `ExperimentConfig` dataclass. One instance per experiment; different task types (MTL, category, next) get different instances with different defaults via factory classmethods (`default_mtl()`, `default_category()`, `default_next()`). `model_params` is a free-form dict (JSON-serializable) until the model registry exists in Phase 4a.

**Rationale:** A single dataclass that defines everything is simpler than nested sub-configs. The REFACTORING_PLAN §7.1 schema was followed with additions: `optimizer_eps`, `mtl_loss_params` (NashMTL kwargs). Enums stored as `.value` strings for serialization ease.

**Alternatives rejected:**
1. *Nested per-task sub-configs* — adds complexity for no benefit when each config instance is already task-specific.
2. *Hydra/YAML* — frozen in REFACTORING_PLAN "Do Never" list.

**Files affected:** `src/configs/experiment.py` (new), `tests/test_configs/test_experiment_config.py` (new)
**Verification:** `PYTHONPATH=src python -c "from configs.experiment import ExperimentConfig; c = ExperimentConfig(name='t', state='fl', embedding_engine='dgi'); c.save('/tmp/cfg.json'); c2 = ExperimentConfig.load('/tmp/cfg.json'); assert c == c2; print('ok')"`

---

#### [3.2] Hardcoded hyperparameters removed from mtl_train.py (BPR*)

**Precondition verified:** `lr=0.0001` matches `MTLModelConfig.LEARNING_RATE`; `weight_decay=5e-2` matches `CfgCategoryHyperparams.WEIGHT_DECAY` (MTL has no dedicated hyperparams class); `eps=1e-8` is PyTorch AdamW default; `max_grad_norm=1.0` matches both task configs. Precondition holds → classified as BPR.

**Decision:** `train_with_cross_validation()` now accepts `config: ExperimentConfig` instead of individual `num_classes`, `num_epochs`, `learning_rate`, `gradient_accumulation_steps` parameters. All optimizer, scheduler, and NashMTL parameters sourced from config. `train_model()` now accepts `max_grad_norm` as a parameter.

**Key disambiguation:** `config.learning_rate` (1e-4) is the optimizer LR. `config.max_lr` (1e-3 for MTL) is the OneCycleLR scheduler peak. Previously, the `learning_rate` parameter was confusingly used only for scheduler `max_lr` calculation (`learning_rate * 10`) while the actual optimizer LR was hardcoded to 0.0001.

**Files affected:** `src/train/mtlnet/mtl_train.py`, `pipelines/train/mtl.pipe.py`
**Verification:** `grep -rn "lr=0.0001\|lr=1e-4\|weight_decay=5e-2\|weight_decay=0.05" src/train/ --include="*.py"` → zero matches

---

#### [3.3] ExperimentConfig wired into category and next pipelines (BPR)

**Decision:** `run_cv()` in both `src/train/category/cross_validation.py` and `src/train/next/cross_validation.py` now accepts `config: ExperimentConfig`. Trainer functions (`src/train/category/trainer.py`, `src/train/next/trainer.py`) accept `epochs`, `max_grad_norm`, and `early_stopping_patience` as explicit parameters instead of reading from `CfgCategory*`/`CfgNext*` config classes. Pipeline files construct configs via factory classmethods.

**Key observation preserved:** Category `use_class_weights=False` (the `weight=alpha` line was commented out in the original `cross_validation.py:44`). Now driven by `config.use_class_weights`.

**Files affected:** `pipelines/train/cat_head.pipe.py`, `pipelines/train/next_head.pipe.py`, `src/train/category/cross_validation.py`, `src/train/next/cross_validation.py`, `src/train/category/trainer.py`, `src/train/next/trainer.py`
**Verification:** `pytest tests/test_regression/ -v` → 12 passed

---

#### [3.4] Old config classes deprecated (BPR)

**Decision:** Added deprecation docstrings to `src/configs/category_config.py`, `src/configs/next_config.py`, and `MTLModelConfig` in `src/configs/model.py`. Classes retained for backward compatibility — unconverted consumers (e.g., `create_fold.py` default params) continue working. Deletion deferred to Phase 5.

**Files affected:** `src/configs/category_config.py`, `src/configs/next_config.py`, `src/configs/model.py`
**Verification:**
```
grep -rn "DEPRECATED: Use ExperimentConfig" src/configs/
# → src/configs/category_config.py:1:"""DEPRECATED: Use ExperimentConfig from configs.experiment instead.
# → src/configs/model.py:38:    """DEPRECATED: Use ExperimentConfig from configs.experiment instead."""
# → src/configs/next_config.py:1:"""DEPRECATED: Use ExperimentConfig from configs.experiment instead.
```

---

#### [3.5] RunManifest and DatasetSignature added (BPR)

**Decision:** Added `DatasetSignature` (immutable file fingerprint: path, sha256, size, mtime) and `RunManifest` (write-only provenance record) to `src/configs/experiment.py`. `DatasetSignature.from_path()` computes SHA-256 in 64KB streaming chunks. `RunManifest.from_current_env()` captures torch version, device, git commit, deterministic flags, and dataset signatures. `RunManifest.write()` serializes as `manifest.json`.

**Rationale:** REFACTORING_PLAN §7.5 requires write-only provenance. The manifest never drives training — it only records what happened.

**Files affected:** `src/configs/experiment.py`, `tests/test_configs/test_experiment_config.py`
**Verification:** `pytest tests/test_configs/test_experiment_config.py -v` → 23 passed

---

#### [3.6] RunManifest serialized alongside results directories (BPR)

**Decision:** `manifest.json` is written at the end of each training loop when `results_path` is provided. Present in all three training paths: `src/train/mtlnet/mtl_train.py` (MTL), `src/train/category/cross_validation.py` (category), `src/train/next/cross_validation.py` (next). Pipeline callers pass `results_path` computed from `IoPaths.get_results_dir()`.

**Files affected:** `src/train/mtlnet/mtl_train.py`, `src/train/category/cross_validation.py`, `src/train/next/cross_validation.py`, `pipelines/train/mtl.pipe.py`, `pipelines/train/cat_head.pipe.py`, `pipelines/train/next_head.pipe.py`
**Verification:** `grep -rn "RunManifest" src/train/` → present in all 3 training loops

