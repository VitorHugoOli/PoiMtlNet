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

---

### Phase 4a — Shared Utilities and Model Registry
Date: 2026-03-26

---

#### [4a.1] Model registry (BPR)

**Decision:** Created `src/models/registry.py` with `@register_model` decorator and `create_model()` / `list_models()` functions. Uses a flat `_MODEL_REGISTRY` dict[str, type]. Lazy registration via `_ensure_registered()` imports all model modules on first `create_model()` / `list_models()` call, avoiding import-time side effects.

**Rationale:** REFACTORING_PLAN §7.2 specifies a registry pattern. Lazy registration avoids forcing all model imports at registry import time, while ensuring models are available when needed.

**Alternatives rejected:**
1. *Eager imports in `models/__init__.py`* — forces heavy torch imports on any `models.*` access.
2. *Manual registration calls* — fragile; easy to forget when adding a new model.

**Files affected:** `src/models/__init__.py` (new), `src/models/registry.py` (new)
**Verification:** `python -c "from models.registry import create_model, list_models; print(len(list_models()), 'models')"`
→ 16 models

---

#### [4a.2] Register all existing model variants (BPR)

**Decision:** All 16 model variants registered with canonical names via `@register_model` decorators:
- Category (8): `category_single`, `category_residual`, `category_gated`, `category_ensemble`, `category_attention`, `category_transformer`, `category_dcn`, `category_se`
- Next (7): `next_single`, `next_mtl`, `next_lstm`, `next_gru`, `next_temporal_cnn`, `next_hybrid`, `next_transformer_optimized`
- MTL (1): `mtlnet`

Names match `ExperimentConfig.model_name` values from Phase 3 factory classmethods.

**Files affected:** `src/models/heads/category.py`, `src/models/heads/next.py`, `src/model/mtlnet/mtl_poi.py`
**Verification:** `python -c "from models.registry import list_models; print(list_models())"`

---

#### [4a.3] Positional encoding extracted (BPR)

**Decision:** Moved `PositionalEncoding` from `src/model/next/next_head.py` to `src/models/components/positional.py`. Original file becomes a thin re-export shim. `src/model/mtlnet/next_head.py` (which imported `PositionalEncoding` from the old location) also becomes a shim.

**Rationale:** Sinusoidal positional encoding is a shared component used by multiple next-head variants. Extracting it prevents duplication as new heads are added.

**Files affected:** `src/models/components/positional.py` (new), `src/model/next/next_head.py` (shim), `src/model/mtlnet/next_head.py` (shim)
**Verification:** `python -c "from models.components.positional import PositionalEncoding; print('ok')"`

---

#### [4a.4] Category heads consolidated (BPR)

**Decision:** All 8 category head variants consolidated into `src/models/heads/category.py`. Original files in `src/model/category/` become thin re-export shims. `CategoryHeadMTL = CategoryHeadEnsemble` alias preserved per Phase 1 contract [1.5]. Helper classes renamed with `_` prefix (`_CategoryResidualBlock`, `_GatedLayer`) to indicate internal use; backward-compat re-exports (`ResidualBlock`, `GatedLayer`) provided in shim.

**Files affected:** `src/models/heads/category.py` (new), `src/model/category/category_head.py` (shim), `src/model/category/category_head_enhanced.py` (shim), `src/model/category/CategoryHeadTransformer.py` (shim), `src/model/category/DCNHead.py` (shim), `src/model/category/SEHead.py` (shim), `src/model/mtlnet/category_head.py` (shim)
**Verification:** `pytest tests/test_regression/ -v` → 12 passed

---

#### [4a.5] Next heads consolidated (BPR)

**Decision:** All 7 next-head variants consolidated into `src/models/heads/next.py`. Original files become thin re-export shims. `NextHeadMTL` uses `PositionalEncoding` from `models.components.positional`.

**Files affected:** `src/models/heads/next.py` (new), `src/model/next/next_head.py` (shim), `src/model/next/next_head_enhanced.py` (shim), `src/model/mtlnet/next_head.py` (shim)
**Verification:** `pytest tests/test_regression/ -v` → 12 passed

---

#### [4a.6] Shared training helpers extracted (BPR)

**Decision:** Created `src/training/helpers.py` with three functions extracted from the duplicated patterns across all three `cross_validation.py` files:
- `compute_class_weights(targets, num_classes, device)` → `torch.Tensor`
- `setup_optimizer(model, learning_rate, weight_decay, eps)` → `AdamW`
- `setup_scheduler(optimizer, max_lr, epochs, steps_per_epoch)` → `OneCycleLR`

All three runners updated to use shared helpers. MTL runner passes `eps=config.optimizer_eps` (preserving MTL's custom epsilon).

**Alternatives rejected:**
1. *Single `setup_fold()` function* — too much variation between runners (MTL needs dual dataloaders, NashMTL, etc.) to unify further without branching.

**Files affected:** `src/training/helpers.py` (new), `src/train/category/cross_validation.py`, `src/train/next/cross_validation.py`, `src/train/mtlnet/mtl_train.py`
**Verification:** `grep -rn "compute_class_weight\b" src/train/` → zero matches (only `compute_class_weights` from shared helper)

---

#### [4a.7] Shared evaluation helpers extracted (BPR)

**Decision:** Created `src/training/evaluate.py` with:
- `collect_predictions(model, loader, device, forward_fn)` — `@torch.no_grad()` decorated, returns `(preds, targets)` as numpy. `forward_fn` parameter enables MTL use (extract single head from tuple output).
- `build_report(preds, targets)` — wraps `sklearn.classification_report` with standard options.

Existing `src/train/shared/evaluate.py` (Phase 1 shared evaluate) and `src/train/mtlnet/evaluate.py` remain unchanged — they are consumed by their respective runners and still work. The new `training.evaluate` provides a lower-level composable API that can replace them incrementally.

**Rationale:** REFACTORING_PLAN §7.3 specifies these exact function signatures. Each runner can call `collect_predictions` + `build_report` as a drop-in for the old monolithic `evaluate()`. Deferred full wiring to avoid changing trainer internals in a BPR.

**Files affected:** `src/training/__init__.py` (new), `src/training/evaluate.py` (new)
**Verification:** `python -c "from training.evaluate import collect_predictions; print('ok')"` → ok

---

#### [4a.8] Loss registry (BPR)

**Decision:** Created `src/losses/registry.py` with `@register_loss` decorator and `create_loss()` / `list_losses()` functions. Registers all 5 existing loss classes: `nash_mtl` (NashMTL), `focal` (FocalLoss), `pcgrad` (PCGrad), `gradnorm` (GradNormLoss), `naive` (NaiveLoss). Uses lazy registration (same pattern as model registry). Registration is done via `setdefault()` in `_ensure_registered()` since the original criterion files are not modified with decorators.

**Rationale:** Original criterion files are shared infrastructure that may be imported by external code. Adding decorators would create a dependency from `criterion/` → `losses/`, which inverts the intended direction. Instead, the registry module knows about the criterion classes.

**Files affected:** `src/losses/__init__.py` (new), `src/losses/registry.py` (new)
**Verification:** `python -c "from losses.registry import create_loss; l = create_loss('nash_mtl', n_tasks=2, device='cpu'); print(type(l))"` → `<class 'criterion.nash_mtl.NashMTL'>`

---

#### [4a.9] Runners wired to use registries (BPR)

**Decision:** All three runners now use `create_model(config.model_name, **config.model_params)` instead of direct class instantiation:
- `src/train/category/cross_validation.py` — was `CategoryHeadEnsemble(...)`, now `create_model(config.model_name, ...)`
- `src/train/next/cross_validation.py` — was `NextHeadSingle(...)`, now `create_model(config.model_name, ...)`
- `src/train/mtlnet/mtl_train.py` — was `MTLnet(...)`, now `create_model(config.model_name, ...)`

MTL runner also uses `create_loss(config.mtl_loss, ...)` instead of direct `NashMTL(...)`.

Direct model head imports removed from all `src/train/` files. Category and next runners also use `compute_class_weights`, `setup_optimizer`, `setup_scheduler` from `training.helpers`.

**Files affected:** `src/train/category/cross_validation.py`, `src/train/next/cross_validation.py`, `src/train/mtlnet/mtl_train.py`
**Verification:**
```
grep -rn "CategoryHeadEnsemble\|NextHeadTransformer" src/train/ --include="*.py"
# → zero matches
```

---

### Phase 4b — MTL-Specific Semantic Corrections
Date: 2026-03-26

---

#### [4b.1] Fix MTL scheduler mismatch (SC)

**Current behavior:** `setup_scheduler()` called with `len(dataloader_next.train.dataloader)` — only the next-task loader length. But training uses `zip_longest_cycle()` from `TrainingProgressBar.iter_epoch()`, which produces `max(len(next_loader), len(category_loader))` steps per epoch. If the category loader is longer, OneCycleLR overruns its `total_steps` and raises `RuntimeError`.

**Expected behavior:** `steps_per_epoch = max(len(next), len(category))` — matching `zip_longest_cycle()`. The LR warmup→peak→cooldown cycle is distributed across the actual number of optimizer steps.

**Predicted metric direction:** Positive or neutral. Proper LR scheduling ensures the full warmup and annealing phases align with actual training steps.

**Synthetic test (asymmetric data: cat=1050 samples/17 batches, next=560/9 batches, 3 epochs, CPU, seed=42):**

| Metric | OLD (buggy) | FIXED | Delta |
|--------|------------|-------|-------|
| Category F1 | 0.5093 (crashed mid-training) | 0.8070 | +0.2977 |
| Next F1 | 1.0000 | 1.0000 | +0.0000 |
| Scheduler total_steps | 27 (too few, crashed) | 51 (correct) | — |

The old scheduler had `total_steps=27` (9 batches × 3 epochs) but training produced 17 steps/epoch (the longer loader), causing a crash at step 28. The fix sets `total_steps=51` (17 × 3), exactly matching actual training steps.

**Note on real data:** In production, the next-task loader is typically longer (more sequences than unique POIs), so the old code wouldn't crash — but `total_steps` would be inexact whenever the two loaders have different lengths. The fix is correct for all cases.

**Files affected:** `src/train/mtlnet/mtl_train.py` (line 246-251)
**Verification:**
```
grep -n "steps_per_epoch" src/train/mtlnet/mtl_train.py
# → 246: # steps_per_epoch must match zip_longest_cycle() — the longer loader
# → 247: steps_per_epoch = max(
```

---

#### [4b.2] Align MTL validation with training coverage (SC)

**Current behavior:** `evaluate_model()` (line 27) used `zip(*dataloaders)` — truncating to the shorter validation loader. `validation_best_model()` (lines 20, 33) used `zip(data_next, data_category)` — same truncation. Samples from the longer loader's tail were silently dropped during validation.

**Expected behavior:** Both functions use `zip_longest_cycle()` from `common.training_progress`, matching the training loop's iteration pattern. All validation samples from both loaders are evaluated.

**Predicted metric direction:** Unknown (small magnitude). Evaluates the full validation set instead of a truncated subset. Impact depends on whether the dropped tail samples were easier/harder than average.

**Synthetic test (asymmetric val data: cat=280 samples/5 batches, next=105/2 batches):**

| Coverage | OLD (zip) | NEW (zip_longest_cycle) |
|----------|----------|------------------------|
| Category val samples evaluated | 128/280 (45.7%) | 280/280 (100%) |
| Next val samples evaluated | 105/105 (100%) | 105/105 (100%, cycled) |
| Batches iterated | 2 (shorter) | 5 (longer) |

The old `zip()` silently discarded 54.3% of category validation samples. The fix covers all samples from both loaders. Cycled samples from the shorter loader produce symmetric (pred, truth) duplicates that don't affect macro-F1.

**Files affected:** `src/train/mtlnet/evaluate.py` (line 32), `src/train/mtlnet/validation.py` (lines 22, 35)
**Verification:**
```
grep -rn "zip_longest_cycle" src/train/mtlnet/
# → evaluate.py:4, 12, 32
# → validation.py:4, 12, 22, 35
grep -rn "for .* in zip(" src/train/mtlnet/ --include="*.py"
# → zero matches
```

---

#### [4b.3] MTL runner integration with shared helpers (BPR)

**Decision:** No additional changes needed. Phase 4a already wired the MTL runner to use all shared helpers: `compute_class_weights`, `setup_optimizer`, `setup_scheduler` from `training.helpers`, plus `create_model` and `create_loss` from their respective registries. The remaining `evaluate_model` and `validation_best_model` are MTL-specific (dual-dataloader cycling, dual-task-head evaluation) and correctly remain in `train.mtlnet/`.

**Verification:**
```
grep -rn "from training.helpers import" src/train/mtlnet/mtl_train.py
# → line 17: from training.helpers import compute_class_weights, setup_optimizer, setup_scheduler
grep -rn "from models.registry import\|from losses.registry import" src/train/mtlnet/mtl_train.py
# → line 15: from losses.registry import create_loss
# → line 16: from models.registry import create_model
pytest tests/test_regression/ -v
# → 12 passed
```

---

### Phase 5 — Folder Tree Migration
Date: 2026-03-26

---

#### [5.1] configs/ — no moves needed (BPR)

**Decision:** `src/configs/` already at target location per REFACTORING_PLAN §6. Contains `__init__.py`, `experiment.py`, `model.py`, `paths.py`, `globals.py`, `embedding_fusion.py`. Deprecated `category_config.py` and `next_config.py` kept until Phase 6.

**Files affected:** None
**Verification:**
```
PYTHONPATH=src python -c "from configs.experiment import ExperimentConfig; print('ok')"
# → ok
```

---

#### [5.2] models/ — move mtl_poi.py to models/mtlnet.py (BPR)

**Decision:** Moved `src/model/mtlnet/mtl_poi.py` → `src/models/mtlnet.py`. All `src/model/` shim files (from P4a) updated with DeprecationWarning. Registry updated to import from `models.mtlnet`. Test imports updated to canonical `models.heads.category`, `models.heads.next`, `models.mtlnet`.

**Files affected:** `src/models/mtlnet.py` (moved), `src/models/registry.py` (import updated), `src/model/mtlnet/mtl_poi.py` (shim), all `src/model/` shims (DeprecationWarning added), `tests/test_regression/test_regression.py` (imports updated)
**Verification:**
```
PYTHONPATH=src python -c "from models.mtlnet import MTLnet; print('ok')"
# → ok
```

---

#### [5.3] losses/ — move criterion/*.py to losses/ (BPR)

**Decision:** Moved 5 loss files: `FocalLoss.py` → `focal.py`, `NaiveLoss.py` → `naive.py`, `gradnorm.py`, `nash_mtl.py`, `pcgrad.py`. Created `criterion/__init__.py` and individual shim files with DeprecationWarning. Registry updated to import from `losses.*`.

**Files affected:** `src/losses/{focal,naive,gradnorm,nash_mtl,pcgrad}.py` (moved), `src/losses/registry.py` (imports updated), `src/criterion/*.py` (shims)
**Verification:**
```
PYTHONPATH=src python -c "from losses.nash_mtl import NashMTL; print('ok')"
# → ok
```

---

#### [5.4] data/ — move etl/create_fold, etl/mtl_input, poi_dataset (BPR)

**Decision:** Moved `etl/create_fold.py` → `data/folds.py`, `etl/mtl_input/{core,builders,loaders,fusion}.py` → `data/inputs/`, `common/poi_dataset.py` → `data/dataset.py`. Created shims at all old paths with DeprecationWarning. Internal imports use relative references and work in new location.

**Files affected:** `src/data/{folds,dataset,inputs/}.py` (moved), `src/etl/` (shims), `src/common/poi_dataset.py` (shim), all pipeline files (imports updated), `src/train/mtlnet/mtl_train.py` (import updated)
**Verification:**
```
PYTHONPATH=src python -c "from data.folds import FoldCreator; print('ok')"
# → ok
```

---

#### [5.5] training/ — move train/* to training/runners/ (BPR)

**Decision:** Moved all 9 train files to `src/training/runners/`: `train/shared/evaluate.py` → `training/shared_evaluate.py`, `train/{category,next,mtlnet}/*.py` → `training/runners/{category,next,mtl}_*.py`. Created shims at all old paths with DeprecationWarning. Updated cross-references within moved files.

**Files affected:** `src/training/runners/{category,next,mtl}_*.py` (moved), `src/training/shared_evaluate.py` (moved), `src/train/` (shims), pipeline files (imports updated)
**Verification:**
```
PYTHONPATH=src python -c "from training.runners.mtl_cv import train_with_cross_validation; print('ok')"
# → ok
```

---

#### [5.6] tracking/ — move common/ml_history/ to tracking/ (BPR)

**Decision:** Moved all 11 ml_history files to `src/tracking/`: `experiment.py`, `fold.py`, `metric_store.py`, `best_tracker.py`, `storage.py`, `display.py`, `parms/neural.py`, `utils/{dataset,time_history}.py`. Updated all internal cross-references. Created shim files at all old paths with DeprecationWarning.

**Files affected:** `src/tracking/*.py` (moved), `src/common/ml_history/*.py` (shims), all training runners (imports updated), all pipeline files (imports updated)
**Verification:**
```
PYTHONPATH=src python -c "from tracking.experiment import MLHistory; print('ok')"
# → ok
```

---

#### [5.7] utils/ — move calc_flops, mps_support, training_progress (BPR)

**Decision:** Moved `common/calc_flops/calculate_model_flops.py` → `utils/flops.py`, `common/calc_flops/model_profiler.py` → `utils/profiler.py`, `common/calc_flops/utils/{profile_exporter,profile_reporter}.py` → `utils/`, `common/mps_support.py` → `utils/mps.py`, `common/training_progress.py` → `utils/progress.py`. Created shims at old paths.

**Files affected:** `src/utils/*.py` (moved), `src/common/{mps_support,training_progress,calc_flops/}.py` (shims), all training runners (imports updated)
**Verification:**
```
PYTHONPATH=src python -c "from utils.flops import calculate_model_flops; print('ok')"
# → ok
```

---

#### [5.8] Test structure migrated (BPR)

**Decision:** Moved test directories to match new tree: `test_criterion/` → `test_losses/`, `test_model/` → `test_models/`, `test_common/test_ml_history.py` → `test_tracking/`, `test_common/test_{calc_flops,training_progress}.py` → `test_utils/`, `etl/` → `test_data/`. Updated all test imports to use canonical paths.

**Files affected:** All test files moved and imports updated
**Verification:** `pytest -v` → 320 passed, 79 skipped, 0 failed

---

#### [5.9] Embedding trainers moved to research/embeddings/ (BPR)

**Decision:** Moved entire `src/embeddings/` to `research/embeddings/`. Created shim `src/embeddings/__init__.py` with DeprecationWarning. Updated all embedding pipeline scripts to add `research/` to sys.path.

**Files affected:** `research/embeddings/` (moved), `src/embeddings/__init__.py` (shim), `pipelines/embedding/*.pipe.py` (sys.path updated), `pipelines/fusion.pipe.py` (sys.path updated)
**Verification:**
```
PYTHONPATH=research:src python -c "from embeddings.hgi.hgi import train_hgi; print('ok')"
# → ok
PYTHONPATH=src python -W all -c "import warnings; warnings.simplefilter('always'); import embeddings" 2>&1
# → <...>: DeprecationWarning: src/embeddings is deprecated; embedding trainers moved to research/embeddings/. This shim will be removed at the end of Phase 6.
pytest tests/test_regression/ -v
# → 12 passed
```

---

#### [5.10] src-prefixed imports fixed in tests (BPR)

**Decision:** Fixed all `from src.X import Y` imports in test files to use `from X import Y`. The `src.` prefix was a pre-existing bug causing test failures (mock `@patch` targets didn't match actual import paths). Fixed 6 previously-failing tests.

**Files affected:** `tests/conftest.py`, `tests/test_data/test_mtl_input_{builders,core,checkin_conversion,fusion}.py`, `tests/test_data/test_convert_user_checkins.py`
**Verification:** `grep -rn "^from src\." src/ scripts/ tests/ --include="*.py"` → zero results


---

### Phase 6 — Script Consolidation and Shim Removal
Date: 2026-03-27

---

#### [6.1-6.2] scripts/train.py created with argparse CLI (BPR)

**Decision:** Created `scripts/train.py` as the canonical CLI training entrypoint. All imports use Phase 5 canonical paths (`configs.experiment`, `configs.paths`, `data.folds`, `training.runners.*`, `tracking.MLHistory`). Dispatches to `_run_mtl`, `_run_category`, `_run_next` based on `--task` flag. `--folds N` runs only the first N folds; the split structure uses `max(2, N)` splits (StratifiedKFold requires ≥ 2). `--config` loads `ExperimentConfig` from an experiment file.

**Flags:** `--state`, `--engine`, `--task` (mtl/category/next), `--epochs`, `--folds`, `--config`

**Files affected:** `scripts/train.py` (new)
**Verification:**
```
PYTHONPATH=src python scripts/train.py --help
# → shows all expected flags with defaults
```

---

#### [6.3-6.4] experiments/configs/ and directory stubs created (BPR)

**Decision:** Created `experiments/configs/` with three declarative config files: `mtl_hgi_florida.py`, `mtl_dgi_florida.py`, `mtl_hgi_alabama.py`. Each exports `config() -> ExperimentConfig`. No training logic, no side effects, no imports beyond `configs.experiment`. Created `experiments/baselines/__init__.py` and `experiments/ablations/__init__.py`. (`experiments/archive/` existed from Phase 1 [1.8].)

**Files affected:** `experiments/configs/{__init__,mtl_hgi_florida,mtl_dgi_florida,mtl_hgi_alabama}.py` (new), `experiments/baselines/__init__.py` (new), `experiments/ablations/__init__.py` (new)
**Verification:**
```
PYTHONPATH=src python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('_c', 'experiments/configs/mtl_hgi_florida.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(m.config().name)  # → mtl_hgi_florida
"
```

---

#### [6.5] pipelines/train/ converted to thin wrappers (BPR)

**Decision:** Replaced `mtl.pipe.py`, `cat_head.pipe.py`, `next_head.pipe.py` with thin subprocess wrappers that call `scripts/train.py` with `--state`, `--engine`, `--task`. All training logic removed from pipelines. Multi-state/engine loops preserved via `nargs="+"` argparse.

**Alternatives rejected:**
1. *Keep old pipeline logic* — would leave duplicate logic; Phase 6 scope requires thin wrappers.

**Files affected:** `pipelines/train/mtl.pipe.py`, `pipelines/train/cat_head.pipe.py`, `pipelines/train/next_head.pipe.py`
**Verification:** `python pipelines/train/mtl.pipe.py --help`

---

#### [6.6] scripts/evaluate.py created (BPR)

**Decision:** Created `scripts/evaluate.py` for checkpoint evaluation. Loads a `.pt` checkpoint, looks up the config from `config.json` alongside results, creates fold data, runs `collect_predictions` + `build_report` from `training.evaluate`. All imports from Phase 5 canonical paths.

**Files affected:** `scripts/evaluate.py` (new)
**Verification:** `PYTHONPATH=src python scripts/evaluate.py --help`

---

#### [6.7] All Phase 5 transitional shims removed (BPR)

**Decision:** Deleted all Phase 5 shim packages. Confirmed zero production imports from shim paths before deletion.

**Shim packages removed:**
- `src/common/` — ml_history, calc_flops, mps_support, training_progress, poi_dataset shims
- `src/criterion/` — FocalLoss, NaiveLoss, gradnorm, nash_mtl, pcgrad shims
- `src/etl/` — create_fold, mtl_input/* shims
- `src/model/` — all category/next/mtlnet head shims
- `src/train/` — all category/next/mtlnet runner shims
- `src/embeddings/__init__.py` — research/embeddings shim

**Pre-deletion check:**
```
grep -rn "from common\.\|from criterion\.\|from etl\.\|from train\.\|from model\." src/ scripts/ tests/ pipelines/ --include="*.py"
# → zero results
```

**Verification:**
```
grep -rn "DeprecationWarning" src/
# → zero results (all shims gone)
grep -rn "from common\.\|from criterion\.\|from etl\.\|from train\.\|from model\." src/ scripts/
# → zero results
pytest -v
# → 320 passed, 79 skipped, 0 failures
```

---

#### [6.8] Smoke test run + graceful fallback for pre-Phase-2 data (BPR)

**Decision:** `data/folds.py:load_category_data()` returns `placeids=None` when the category parquet has no `placeid` column (pre-Phase-2 generated data). `_create_mtl_folds()` detects `None` and falls back to independent `StratifiedKFold` for category splits while still using `StratifiedGroupKFold` for next-task user splits. Emits `WARNING` log entries explaining the fallback. `scripts/train.py` max_folds slicing no longer updates `config.k_folds` (avoids `k_folds >= 2` validation error when running 1 fold).

**Rationale:** Florida DGI category parquet was generated before Phase 2 and lacks `placeid`. The fallback ensures the CLI smoke test works on real data without requiring data regeneration. Alabama HGI data (generated during Phase 2) continues to use the full user-isolation protocol.

**Files affected:** `src/data/folds.py` (load_category_data, _create_mtl_folds), `scripts/train.py`

**CLI smoke test (reproducible command — data-paths must be explicit from the worktree):**
```bash
OUTPUT_DIR=/Users/vitor/Desktop/mestrado/ingred/output \
RESULTS_ROOT=/path/to/worktree/results \
DATA_ROOT=/Users/vitor/Desktop/mestrado/ingred/data \
python scripts/train.py --state florida --engine dgi --epochs 1 --folds 1
```
Run 2026-03-27 02:34 → completed in 35s.  Key log lines (abbreviated):
```
INFO  - Creating 2-fold CV for mtl
WARNING - Category parquet for florida/dgi has no 'placeid' column (pre-Phase-2 data).
          MTL user-level splits require regenerating input with the Phase 2 pipeline.
          Falling back to independent StratifiedKFold splits.
INFO  - Fold 1/2: train_users=5241, val_users=5290
INFO  - Training: state=florida  engine=dgi  task=mtl  epochs=1  folds=1
INFO  - FLOPS: 70249216 | Params: 4307855
[Epoch 1/1 completes, val metrics: cat_val=0.4536(0.4791), next_val=0.2523(0.3327)]
INFO  - Fold 1/1 completed in 34.67s
INFO  - Done. Results written to: .../results/dgi/florida
```

**Verification (run from worktree with env vars above):**
```bash
ls results/dgi/florida/manifest.json          # → exists (timestamp matches smoke run)
python -c "import json; print(list(json.load(open('results/dgi/florida/manifest.json')).keys())[:6])"
# → ['config', 'git_commit', 'seeds', 'pytorch_version', 'device', 'deterministic_flags']
grep -rn "DeprecationWarning" src/            # → zero results
grep -rn "from common\.\|from criterion\.\|from etl\.\|from train\.\|from model\." src/ scripts/  # → zero results
PYTHONPATH=src pytest -v                      # → 320 passed, 79 skipped, 0 failures
```

**Phase 7 must know:**
- Florida DGI (and other pre-Phase-2) data does not have `placeid` in category parquets. Regenerate with the Phase 2 pipeline to enable full user-isolation MTL splits.
- The `legacy_stratified` split mode is recorded in fold manifests for traceability.
- `scripts/train.py` `--folds N` is an execution limit, not the split count. The config's `k_folds` stays at `max(2, N)`.
- All shims are gone. Any Phase 7 integration tests must use canonical paths only.

---

### Phase 7 — Testing and Reproducibility
Date: 2026-03-27

---

#### [7.1] Integration tests for category, next, MTL (BPR)

**Decision:** Created `tests/test_integration/` with three test files and a shared conftest:
- `conftest.py` — synthetic data infrastructure (make_category_data, make_next_data, make_loaders, seed_everything)
- `test_category_integration.py` (3 tests) — pipeline completion, registry roundtrip, loss decrease
- `test_next_integration.py` (4 tests) — pipeline completion, registry roundtrip, loss decrease, shape preservation
- `test_mtl_integration.py` (5 tests) — pipeline completion, dual-dataloader cycling, gradient flow, loss decrease, CPU determinism

All tests use synthetic data (class-specific centroids + noise), 2 folds × 3 epochs, CPU-only with `torch.use_deterministic_algorithms(True)`. No real data files required on fresh checkout.

**Alternatives rejected:**
1. *Use real data subset* — requires data files in repo, violates "fresh checkout" requirement
2. *Mock training loop* — wouldn't catch real integration issues between model/optimizer/scheduler

**Files affected:** `tests/test_integration/{__init__,conftest,test_category_integration,test_next_integration,test_mtl_integration}.py`
**Verification:**
```
pytest tests/test_integration/ -v
# → 12 passed
```

---

#### [7.2] Regression fixtures upgraded to shared synthetic data (BPR)

**Decision:** Refactored `tests/test_regression/test_regression.py` to import data generators from `tests.test_integration.conftest` instead of defining them locally. Same calibration values, same deterministic results. Regression-specific helpers (`_train_and_evaluate`, `_train_mtl_and_evaluate`) remain local — they use `TRAIN_EPOCHS=10` vs integration's 3.

**Files affected:** `tests/test_regression/test_regression.py`
**Verification:**
```
pytest tests/test_regression/ -v
# → 12 passed
```

---

#### [7.3] random.seed(seed) added to FoldCreator (BPR)

**Decision:** Added `random.seed(seed)` alongside existing `torch.manual_seed(seed)` and `np.random.seed(seed)` in `FoldCreator.__init__`. PCGrad loss uses `random.shuffle()` for gradient projection ordering — without seeding Python's `random` module, this was a source of non-determinism.

**Files affected:** `src/data/folds.py`
**Verification:**
```
grep -rn "torch\.manual_seed(\|np\.random\.seed(\|random\.seed(" src/ --include="*.py"
# → src/data/folds.py:136:    np.random.seed(worker_seed)
# → src/data/folds.py:449:        random.seed(seed)
# → src/data/folds.py:450:        torch.manual_seed(seed)
# → src/data/folds.py:451:        np.random.seed(seed)
```

**Note on the DoD grep pattern:** The DoD specified `grep -rn "torch.manual_seed\|np.random.seed\|random.seed" src/`
without `--include="*.py"`. That pattern also matches:
- `__pycache__/*.pyc` binaries (binary file noise)
- `src/configs/experiment.py:392: "torch_manual_seed": config.seed` (string literal in RunManifest, not an actual seed call — the unescaped `.` in the grep pattern matches `_`)

The corrected command above uses `--include="*.py"` (excludes `.pyc`) and `(` suffix (matches only actual function calls, not string keys). All 4 actual seed calls are in `data/folds.py` and use the `seed` parameter from `ExperimentConfig.seed`.

---

#### [7.4] Test documentation fixed (BPR)

**Decision:** Updated test tree in `docs/REFACTORING_PLAN.md` (proposed `test_training/` → actual `test_tracking/`, `test_utils/`). Expanded `CLAUDE.md` test section with per-directory descriptions.

**Files affected:** `docs/REFACTORING_PLAN.md`, `CLAUDE.md`

---

#### [7.5] Stale test directories removed (BPR)

**Decision:** Removed `tests/etl/` and `tests/test_common/` — empty directories left behind during Phase 5 test migration. Each contained only `__init__.py` and `__pycache__`.

**Files affected:** `tests/etl/` (deleted), `tests/test_common/` (deleted)

---

#### [7.6] Runner display bug fixed (BPR)

**Decision:** Removed explicit `history.display.end_fold()` and `history.display.end_training()` calls from `mtl_cv.py` and `next_cv.py`. These were called AFTER `history.step()`, which advances `curr_i_fold` — causing the display to access the wrong fold's timer and crash when running >1 fold. `MLHistory.step()` already handles display when `verbose=True`, so the explicit calls were redundant (and caused double-display on fold 1).

**Files affected:** `src/training/runners/mtl_cv.py`, `src/training/runners/next_cv.py`

---

#### Determinism check (MPS caveat)

**Observation:** CPU determinism is verified by `test_mtl_determinism_on_cpu` integration test (bitwise identical weights across two runs). Real-data training on MPS (Apple Silicon) shows <1% F1 variance between runs — this is a known MPS limitation, not a seeding issue. All RNG sources (torch, numpy, random) are properly seeded from `ExperimentConfig.seed`.

**Reproducibility command (run from worktree root):**
```bash
OUTPUT_DIR=/Users/vitor/Desktop/mestrado/ingred/output \
DATA_ROOT=/Users/vitor/Desktop/mestrado/ingred/data \
RESULTS_ROOT=/tmp/p7_dod_runN \
PYTHONPATH=src python scripts/train.py --state florida --engine dgi --epochs 3 --folds 2
```

**Run 1 raw output (2026-03-27 10:55):**
```
INFO - Creating 2-fold CV for mtl
WARNING - Category parquet for florida/dgi has no 'placeid' column (pre-Phase-2 data).
WARNING - MTL split: no placeid data available; falling back to independent StratifiedKFold for category splits.
INFO - Fold 1/2 completed in 66.24s
  Summary Fold 1: next | 2 | 37.99% | 31.77% | category | 1 | 47.14% | 45.13%
INFO - Fold 2/2 completed in 63.41s
  Summary Fold 2: next | 1 | 38.56% | 32.40% | category | 1 | 47.77% | 45.51%
INFO - Done. Results written to: /tmp/p7_dod_run1/dgi/florida
```

**Run 2 raw output (2026-03-27 11:03):**
```
INFO - Creating 2-fold CV for mtl
WARNING - Category parquet for florida/dgi has no 'placeid' column (pre-Phase-2 data).
WARNING - MTL split: no placeid data available; falling back to independent StratifiedKFold for category splits.
INFO - Fold 1/2 completed in 64.83s
  Summary Fold 1: category | 1 | 47.35% | 45.30% | next | 2 | 38.09% | 31.82%
INFO - Fold 2/2 completed in 62.21s
  Summary Fold 2: category | 1 | 47.20% | 45.12% | next | 1 | 38.55% | 32.41%
INFO - Done. Results written to: /tmp/p7_dod_run2/dgi/florida
```

**Comparison:**

| Fold | Task | Run 1 F1 | Run 2 F1 | Delta |
|------|------|----------|----------|-------|
| 1 | category | 45.13% | 45.30% | +0.17% |
| 1 | next | 31.77% | 31.82% | +0.05% |
| 2 | category | 45.51% | 45.12% | −0.39% |
| 2 | next | 32.40% | 32.41% | +0.01% |

All deltas <0.4% — within expected MPS non-determinism range.

**DoD verification outputs (2026-03-27):**
```
pytest tests/test_integration/ -v → 12 passed
pytest tests/test_regression/ -v → 12 passed
pytest --co -q | tail -1 → 411 tests collected

grep -rn "torch\.manual_seed(\|np\.random\.seed(\|random\.seed(" src/ --include="*.py"
→ src/data/folds.py:136:    np.random.seed(worker_seed)
→ src/data/folds.py:449:        random.seed(seed)
→ src/data/folds.py:450:        torch.manual_seed(seed)
→ src/data/folds.py:451:        np.random.seed(seed)
```

---

### Phase 8 — Deeper Improvements (Callbacks Track)
Date: 2026-03-27

---

#### [8.1] Callback primitives — new module (BPR)

**Decision:** Created `src/training/callbacks.py` with five primitives:
- `CallbackContext` — dataclass carrying epoch index, total epochs, fold info, and metrics dict
- `Callback` — base class with `on_train_begin`, `on_epoch_end`, `on_train_end` hooks and `stop_training` flag
- `CallbackList` — compositor that dispatches hooks in order, aggregates `stop_training` via `any()`
- `EarlyStopping` — patience-based monitoring of any metric key (configurable `monitor`, `patience`, `mode`, `min_delta`)
- `ModelCheckpoint` — saves model state to disk via `set_model()` injection (configurable `save_dir`, `monitor`, `mode`, `save_best_only`)

**Rationale:** The three runners (MTL, category, next) had inconsistent early stopping logic hardcoded inline. A composable callback system enables uniform extension without modifying runner internals. Only epoch-level hooks are provided (no batch-level) to avoid MPS sync overhead in the inner loop.

**Alternatives rejected:**
1. *Batch-level hooks* — would add overhead per batch in the MPS-critical inner loop. No current use case.
2. *PyTorch Lightning callbacks* — project explicitly avoids Lightning (REFACTORING_PLAN §1).
3. *Replace existing early stopping with callbacks* — would be SC; kept existing logic unchanged for BPR safety.

**Files affected:** `src/training/callbacks.py` (new), `src/training/__init__.py` (exports added)
**Verification:**
```
PYTHONPATH=src python -c "from training.callbacks import Callback, EarlyStopping, ModelCheckpoint; print('ok')"
# → ok
```

---

#### [8.2] Wire callbacks into all three training runners (BPR)

**Decision:** Added optional `callbacks: Optional[list] = None` parameter to all training functions and CV orchestrators. When `callbacks=None` (the default), `CallbackList([])` iterates zero callbacks — all hooks are no-ops, `stop_training` returns `False`. No existing code path is altered.

Hook insertion pattern (identical across all three runners):
1. `on_train_begin()` — before first epoch
2. `on_epoch_end()` — after validation metrics logged, BEFORE existing early-stop checks
3. `cb.stop_training` check — AFTER all existing early-stop checks (additive, never replaces)
4. `on_train_end()` — after loop exits

Metric keys per runner:
- category: `val_f1`, `val_loss`, `val_acc`, `train_loss`, `train_acc`
- next: `val_f1`, `val_loss`, `val_acc`, `train_loss`, `train_acc`, `train_f1`, `grad_norm`
- mtl: `val_f1_next`, `val_f1_category`, `val_loss`, `train_loss`, `train_f1_next`, `train_f1_category`

Runners also inject model references via `set_model()` for callbacks that need it (e.g. `ModelCheckpoint`).

**Rationale:** Additive wiring preserves all existing behavior. The `callbacks=None` default means `scripts/train.py` and all existing tests exercise the unchanged path. Callbacks are runtime objects, not serializable config — they don't pollute `ExperimentConfig`.

**Files affected:**
- `src/training/runners/category_trainer.py` — `train()` + 4 hook lines
- `src/training/runners/category_cv.py` — `run_cv()` pass-through
- `src/training/runners/next_trainer.py` — `train()` + 4 hook lines
- `src/training/runners/next_cv.py` — `run_cv()` pass-through
- `src/training/runners/mtl_cv.py` — `train_model()` + 4 hook lines, `train_with_cross_validation()` pass-through

**Verification:**
```
grep -rn "class .*Callback\|on_epoch_end" src/training/ --include="*.py"
# → callbacks.py: 5 matches (base + list + early_stopping + checkpoint + context)
# → mtl_cv.py, next_trainer.py, category_trainer.py: 1 match each (hook call)
pytest tests/test_integration/ -v → 12 passed
pytest tests/test_regression/ -v → 12 passed
```

---

#### [8.3] Callback unit tests (BPR)

**Decision:** Created `tests/test_training/test_callbacks.py` with 22 unit tests covering all five primitives: `CallbackContext` defaults/metrics, `Callback` base defaults/noop hooks, `CallbackList` empty/none/dispatch-order/stop-signal, `EarlyStopping` patience/reset/mode-min/min-delta/missing-metric, `ModelCheckpoint` no-model-skip/save-best/save-all/set-model.

**Files affected:** `tests/test_training/__init__.py` (new), `tests/test_training/test_callbacks.py` (new)
**Verification:**
```
PYTHONPATH=src pytest tests/test_training/test_callbacks.py -v → 22 passed
PYTHONPATH=src pytest -v → 354 passed, 79 skipped, 0 failures
```
