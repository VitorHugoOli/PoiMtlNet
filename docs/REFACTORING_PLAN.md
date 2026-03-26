# MTLnet Repository Refactoring Plan

> **What to do first:** Jump to [Section 11](#11-what-to-do-now-vs-later) for
> the prioritized action list. Then read the roadmap (Section 8) for context.

> Each change is classified as:
> - **BPR** — behavior-preserving refactor. Must pass layered safety checks.
> - **BPR\*** — conditional BPR. Verify the stated precondition first; if it
>   fails, treat as SC.
> - **SC** — semantic correction. Expected to change metrics. Document the
>   impact before execution, establish new baselines after.

---

## 1. What to Do Now vs. Later

### Do Now (Phase 0 — hard decisions and safety net)

1. **Decide the MTL split protocol** — see [SPLIT_PROTOCOL.md](SPLIT_PROTOCOL.md) **(SC)**
2. **Produce split feasibility report** — per dataset configuration **(SC)**
3. **Add 3 regression fixtures** — safety net for everything that follows **(BPR)**
4. **Add `pyproject.toml`** **(BPR)**
5. **Add simple CLI flags** (`--state`, `--engine`) to existing pipelines **(BPR)**

### Do Next (Phase 1 — low-risk BPR cleanup, hours not days)

6. Merge the two identical `evaluate()` functions (68 lines of pure duplication)
7. Fix the `validation_model()` scoping bug (`validation.py:69-76`)
8. Make `CategoryHeadMTL` an alias for `CategoryHeadEnsemble`
9. Delete dead code (unused validation functions, legacy notebooks, stale requirements)
10. Make `paths.py` side-effect free, stop eager `mtl_input` imports
11. Resolve `hmrm.py` vs `hmrm_new.py`

### Do After (Phases 2-4 — data contracts, config, registries)

12. Formalize data contracts and implement MTL split decision **(SC)**
13. Create `ExperimentConfig` + `RunManifest`
14. Build model registry
15. Extract shared training/eval helpers
16. Fix MTL scheduler mismatch and validation asymmetry **(SC)**

### Do Later (Phases 5-8 — structural and polish)

17. Folder tree migration (Phase 5 — before scripts)
18. Script consolidation and CLI entrypoints (Phase 6 — targets final paths)
19. Remove transitional re-export shims (end of Phase 6)
20. Integration tests and test migration
21. Loss registry, callback system
22. External experiment tracking (W&B / MLflow / DVC)

### Do Never

- **PyTorch Lightning** — custom MPS optimizations and NashMTL would fight it
- **Hydra** — unless sweeps become a bottleneck; dataclasses are sufficient
- **Package rename to `poimtl`** — import churn for no research benefit
- **Abstract base classes for models** — registry pattern is simpler
- **Monorepo tooling** (Bazel, Pants) — massive overkill

---

## 2. Current-State Diagnosis

~24,400 lines of Python across 151 files.

**What exists:**
- **8 embedding engines** (`src/embeddings/`) — DGI, HGI, Check2HGI, POI2HGI, Time2Vec, Space2Vec, HMRM, FUSION
- **3 model tiers** — MTLnet (joint), single-task category (5+ variants), single-task next-POI (5+ variants)
- **5 loss functions** (`src/criterion/`) — NashMTL, FocalLoss, PCGrad, GradNorm, NaiveLoss
- **3 parallel training paradigms** — MTL, category-only, next-only, each with its own trainer, CV, and evaluation
- **13 pipeline scripts** (`pipelines/`) — the real user workflow is editing and running these
- **Refactored experiment tracking** (`src/common/ml_history/`) — reusable, worth keeping

**Overall shape:** Structured by *task* (`mtlnet/`, `category/`, `next/`) at
every level — model, training, evaluation, configs, tests. This creates a 3×N
matrix where each cell is a near-duplicate of the others.

**Worth preserving:**
- `src/etl/mtl_input/core.py` — pure-logic module (no I/O, no side effects)
- `EmbeddingLevel` / `EmbeddingSpec` in `embedding_fusion.py` — POI-vs-checkin distinction
- `src/common/ml_history/` — clean, recently-refactored experiment tracking
- `FoldCreator` — unified across task types

---

## 3. Main Structural Problems

### 3.1 Pervasive Code Duplication

**Identical evaluation functions.** `src/train/category/evaluation.py` and
`src/train/next/evaluation.py` are character-for-character identical (34 lines).

**Near-duplicate cross-validation orchestrators.** `category/cross_validation.py`
(112 lines) and `next/cross_validation.py` (177 lines) follow the exact same
9-step pattern. Only differences: model class, config source, and next adds
`_extract_diagnostics()`.

**Seven evaluation/validation functions** across three files doing variations
of "iterate dataloaders, collect predictions, produce `classification_report`."

### 3.2 Configuration Scatter

| Setting | `model.py` | `category_config.py` | `next_config.py` | `mtl_train.py` (hardcoded) |
|---------|-----------|---------------------|------------------|-----------------------------|
| LR | `1e-4` | `1e-4` | `1e-4` | `0.0001` (line 259) |
| Epochs | 50 | **2** | 100 | — |
| Batch size | 2048 | 2048 | 512 | — |
| Weight decay | — | `0.05` | `0.01` | `5e-2` (line 261) |
| NUM_CLASSES | 7 | 7 | 7 | — |

`mtl_train.py:257-262` hardcodes `lr=0.0001` and `weight_decay=5e-2` despite
receiving `learning_rate` as a parameter (only used for scheduler `max_lr`).

### 3.3 No Experiment Configuration

Experiments defined by code edits, not configuration. Two experiments cannot
coexist. No reproduction without checking out the exact commit.

### 3.4 Import-Time Side Effects

- `src/configs/paths.py` raises `FileNotFoundError` at import time if
  `data/checkins` is missing.
- `src/etl/mtl_input/__init__.py` eagerly imports loader/builder modules.
- `pytest` fails before any test runs on a fresh checkout without data.

### 3.5 Data Contract Issues

- `generate_category_input()` blindly copies check-in-level embeddings for
  category, conflicting with `EmbeddingLevel.POI`.
- `FoldCreator` reads dimensions from `InputsConfig.EMBEDDING_DIM` instead of
  the actual input artifact.
- `generate_sequences()` hardcodes non-overlapping windows (stride not configurable).

### 3.6 MTL Split Protocol — Research Validity Risk

MTL fold creation zips independently stratified splits — a POI can appear in
one task's training and the other's validation. This is a data leakage issue
for the shared backbone.

**Full protocol specification:** [SPLIT_PROTOCOL.md](SPLIT_PROTOCOL.md)

**Summary:** User-level splits via `StratifiedGroupKFold(groups=userid,
y=next_category)`. User isolation is the hard invariant. Residual cross-task
POI overlap is quantified and reported per fold. Phase 0 must produce a
feasibility report before training. This is a hard gate, not a cleanup item.

### 3.7 MTL Training Loop Semantic Issues

- **Scheduler mismatch.** `OneCycleLR` uses only next-task loader length, but
  training cycles the shorter loader via `zip_longest_cycle()`.
- **Validation asymmetry.** Training cycles; validation truncates via `zip()`.

### 3.8 Model Variants Without Registry

10+ model variants selected by editing imports and class names in
`cross_validation.py`.

### 3.9 `CategoryHeadMTL` is a Duplicate

Same architecture as `CategoryHeadEnsemble`, different class name.

### 3.10 Bugs and Dead Code

- `validation_model()` scoping bug: `torch.no_grad()` at line 69 doesn't
  cover inference loop at line 76.
- `evaluate_model_by_head()` has `foward_method` [sic] typo.
- `notebooks/create_inputs_hgi.py` (18K), `hgi_texas.py` (29K) — superseded.
- `hmrm_new.py` alongside `hmrm.py` — unclear which is canonical.
- `requirements-bckp.txt`, `requirements_upgrad.txt` — stale.

### 3.11 Reproducibility Gaps

No `pyproject.toml`. Multiple stale requirements files. `MLHistory` timestamps
can collide. No run manifest. `results_save/` checked in, code writes to
`results/`.

### 3.12 Testing Gaps

No integration tests. Test documentation doesn't match actual test tree.

---

## 4. Inferred Current Workflow

```
1. EMBEDDING GENERATION
   pipelines/embedding/{engine}.pipe.py → output/{engine}/{state}/*.parquet

2. INPUT PREPARATION
   pipelines/create_inputs.pipe.py → category + next-task parquets
   Note: non-overlapping windows (step = window_size), userid dropped before split.

3. TRAINING
   pipelines/train/{mtl|cat_head|next_head}.pipe.py
   → FoldCreator → stratified 5-fold CV
   Note: MTL folds zip independently stratified splits (see SPLIT_PROTOCOL.md)
   → per fold: model → optimizer → scheduler → train → evaluate
   → results/{engine}/{state}/folds/, plots/, model/, summary/
```

---

## 5. Proposed Target Architecture

### Design Principles

1. **Shared helpers, thin task runners** — common primitives as utility
   functions; each task keeps a thin runner. No monolithic Trainer class.
2. **Config-driven experiments** — single dataclass defines everything.
3. **Registry pattern** for models and losses — select by name from config.
4. **Data contracts before training** — schemas and validation before training refactors.
5. **Embedding trainers are adjacent infrastructure** — training depends on
   "give me a tensor."
6. **Pipelines as thin orchestrators** — all logic in `src/`.
7. **BPR/SC discipline** — every change classified explicitly.

### Key Decisions

**Keep Python dataclasses** (no Hydra). 80% of the value with zero dependencies.

**Don't adopt PyTorch Lightning.** Custom MPS optimizations and NashMTL
integration would fight it.

**Shared training primitives, not a unified Trainer.** MTL, category, next
differ in meaningful ways (diagnostics, dataloader shapes, best-model
semantics, balancing logic). Shared helpers are safer than branching.

**Two config layers.** `ExperimentConfig` (canonical input) + `RunManifest`
(write-only frozen output). No third representation.

**Run manifests from day one.** See `RunManifest` dataclass in section 6.5.

---

## 6. Proposed Folder Tree

```
src/
├── configs/                            # All configuration
│   ├── __init__.py
│   ├── experiment.py                   # ExperimentConfig dataclass
│   ├── model.py                        # Architecture constants
│   ├── paths.py                        # IoPaths (lazy, side-effect free)
│   └── globals.py                      # DEVICE, CATEGORIES_MAP
│
├── models/                             # Pure nn.Module definitions
│   ├── __init__.py
│   ├── registry.py                     # @register_model + create_model()
│   ├── mtlnet.py                       # MTLnet, ResidualBlock, FiLMLayer
│   ├── heads/
│   │   ├── __init__.py
│   │   ├── category.py                 # All category heads
│   │   └── next.py                     # All next heads
│   └── components/
│       ├── __init__.py
│       └── positional.py               # Sinusoidal + Learned positional
│
├── losses/                             # Loss functions
│   ├── __init__.py
│   ├── registry.py                     # @register_loss + create_loss()
│   ├── focal.py
│   ├── nash_mtl.py
│   ├── pcgrad.py
│   ├── gradnorm.py
│   └── naive.py
│
├── data/                               # Dataset, folds, data contracts
│   ├── __init__.py
│   ├── schemas.py                      # Parquet column schemas
│   ├── dataset.py                      # POIDataset
│   ├── folds.py                        # FoldCreator (dimension-aware)
│   ├── inputs/                         # Input generation
│   │   ├── __init__.py                 # Lazy imports
│   │   ├── core.py                     # Pure logic (stride configurable)
│   │   ├── builders.py                 # Level-aware builders
│   │   ├── loaders.py
│   │   └── fusion.py
│   └── embeddings/                     # Downstream embedding integration
│       ├── __init__.py
│       ├── registry.py                 # Embedding provider registry
│       └── validation.py               # Level/dimension validation
│
├── training/                           # Shared training primitives
│   ├── __init__.py
│   ├── helpers.py                      # collect_predictions, compute_class_weights,
│   │                                   # setup_optimizer, build_report
│   ├── evaluate.py                     # Shared evaluation helpers
│   ├── callbacks.py                    # Early stopping, diagnostics
│   └── runners/                        # Thin task-specific runners
│       ├── __init__.py
│       ├── category.py                 # Category CV runner
│       ├── next.py                     # Next-POI CV runner
│       └── mtl.py                      # MTL CV runner
│
├── tracking/                           # Experiment tracking (from ml_history)
│   ├── __init__.py
│   ├── experiment.py
│   ├── fold.py
│   ├── metric_store.py
│   ├── best_tracker.py
│   ├── storage.py
│   ├── display.py
│   └── manifest.py                     # RunManifest
│
├── utils/                              # Shared utilities
│   ├── __init__.py
│   ├── flops.py
│   ├── mps.py
│   └── progress.py
│
├── common/                             # Backward-compat re-exports (transitional)
├── criterion/                          # Backward-compat re-exports (transitional)
├── etl/                                # Backward-compat re-exports (transitional)
└── train/                              # Backward-compat re-exports (transitional)

# SHIM DEPRECATION: Introduced in Phase 5, removed by end of Phase 6.
# Emit DeprecationWarning. Pytest check enforces no new imports.

research/                               # Adjacent infrastructure
├── embeddings/                         # Heavy embedding trainers
│   ├── dgi/, hgi/, check2hgi/, poi2hgi/
│   ├── time2vec/, space2vec/, hmrm/
└── notes/                              # ANALYSIS.md, UPDATE.md, etc.

experiments/                            # Experiment definitions
├── configs/                            # Declarative config constructors ONLY
│   ├── mtl_hgi_florida.py              # Must export config() -> ExperimentConfig
│   └── ...                             # No training logic, no side effects
├── baselines/
├── ablations/
└── archive/

scripts/                                # CLI entrypoints (replaces pipelines/)
├── train.py
├── generate_inputs.py
├── generate_embeddings.py
└── evaluate.py

pipelines/                              # Kept as thin wrappers during transition
tests/
├── conftest.py
├── test_models/
├── test_training/
├── test_data/
├── test_losses/
└── test_integration/

notebooks/
results/                                # gitignored
articles/
docs/
pyproject.toml
```

---

## 7. Recommended Module Boundaries

### Dependency rules

- `configs` → imported by everyone, imports nothing else from `src/`. All
  shared enums (`TaskType`, `EmbeddingEngine`, `EmbeddingLevel`) live here.
- `data` → should not import `training`, `models`, or `losses`
- `models` → should not import `training` or `data`
- `losses` → should not import `models`. Runners pass parameter views.
- `training` → imports `models`, `losses`, `data`, `tracking`
- `tracking` → should not import `training` or `models`
- `scripts` → imports `training` and `configs`, wires things together

### 7.1 ExperimentConfig (canonical input)

```python
# All shared enums live in configs/.

@dataclass
class ExperimentConfig:
    # Identification
    name: str
    state: str
    embedding_engine: EmbeddingEngine

    # Model
    model_name: str              # "mtlnet", "category_ensemble", "next_transformer"
    model_params: dict           # Passed to create_model()

    # Training
    task_type: TaskType          # MTL, CATEGORY, NEXT
    epochs: int = 50
    batch_size: int = 2048
    learning_rate: float = 1e-4
    max_lr: float = 1e-3
    weight_decay: float = 0.05
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0

    # Loss
    task_loss: str = "cross_entropy"
    mtl_loss: str = "nash_mtl"
    use_class_weights: bool = True

    # Cross-validation and split protocol
    k_folds: int = 5
    seed: int = 42
    split_relaxation: bool = False
    min_category_val_fraction: float = 0.05
    min_next_val_fraction: float = 0.05
    min_class_count: int = 5
    min_class_fraction: float = 0.03

    # Early stopping
    timeout: Optional[float] = None
    target_cutoff: Optional[float] = None
    early_stopping_patience: int = -1

    # Schema evolution
    schema_version: int = 1

    def save(self, path: Path): ...

    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig': ...
```

**Serialization:** Enums as `.value` strings, Paths as POSIX, `model_params`
must be JSON-serializable. `load()` validates `schema_version`.

**Experiment files** in `experiments/configs/` must be declarative config
constructors only — define `config() -> ExperimentConfig`. AST-based test
verifies no training/model/data imports.

### 7.2 Model Registry

```python
_REGISTRY: dict[str, type[nn.Module]] = {}

def register_model(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def create_model(name: str, **kwargs) -> nn.Module:
    return _REGISTRY[name](**kwargs)
```

### 7.3 Shared Evaluation Helpers

```python
# training/evaluate.py

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward_fn: Optional[Callable] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model over loader, return (predictions, targets) as numpy."""

def build_report(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Wrap sklearn classification_report with standard options."""
```

Each task runner calls these. MTL calls them twice. Next adds attention
extraction. No monolithic evaluate() is forced.

### 7.4 Shared Training Helpers

```python
# training/helpers.py

def compute_class_weights(loader, num_classes, device) -> torch.Tensor: ...
def setup_optimizer(model, config) -> tuple[Optimizer, Scheduler]: ...
def setup_fold(config, fold_data, history) -> dict: ...
```

Runners are ~50-80 lines each. MTL keeps `zip_longest_cycle()`, NashMTL
backward, dual-dataloader logic. Next keeps `_extract_diagnostics()`.

### 7.5 RunManifest (write-only output)

```python
@dataclass
class DatasetSignature:
    path: str           # Absolute POSIX path
    sha256: str         # SHA-256 hex digest
    size_bytes: int
    mtime: str          # ISO 8601

@dataclass
class RunManifest:
    config: ExperimentConfig
    git_commit: str
    seeds: dict[str, int]
    pytorch_version: str
    device: str
    deterministic_flags: dict[str, bool]
    timestamp: str
    dataset_signatures: dict[str, DatasetSignature]
    # Keys: "category_input", "next_input", "poi_user_mapping", "raw_checkins"
    # After Phase 2: "raw_checkins" replaced by "sequence_poi_mapping"
    split_signature: DatasetSignature
    feasibility_report_signature: DatasetSignature
    schema_version: int = 1
```

Serialized as `manifest.json`. Write-only — never drives training.

### 7.6 Boundary Summary

```
ExperimentConfig  ←  defines what to run (canonical input)
       ↓
  Task Runner     ←  thin, task-specific orchestrator
       ↓
  setup_fold()    ←  shared helper (model + optimizer + scheduler)
  create_model()  ←  builds the model by name
  create_loss()   ←  builds the loss by name
  FoldCreator     ←  creates data splits (see SPLIT_PROTOCOL.md)
  collect_predictions() + build_report()  ←  shared eval helpers
  MLHistory       ←  tracks everything
  RunManifest     ←  freezes provenance (write-only output)
```

---

## 8. High-Impact Code Refactors

### Mechanical duplication (BPR)

| # | Refactor | Files | Type | Impact |
|---|----------|-------|------|--------|
| 8.1 | Merge identical `evaluate()` | `category/evaluation.py`, `next/evaluation.py` | BPR | 68 lines eliminated |
| 8.2 | `CategoryHeadMTL` = `CategoryHeadEnsemble` | `mtlnet/category_head.py` | BPR | Stops divergence |
| 8.3 | Fix `validation_model()` scoping bug | `validation.py:58-101` | BPR* | Precondition: verify unused via grep |
| 8.4 | Extract positional encoding | `next/next_head.py` | BPR | Config parameter |
| 8.5 | Consolidate model variants | `*_enhanced.py`, etc. | BPR | Registry-selectable |
| 8.6 | Extract shared helpers | 3 `cross_validation.py` + 3 `trainer.py` | BPR | ~200 lines → shared |
| 8.7 | Remove hardcoded hyperparams | `mtl_train.py:257-283` | BPR* | Precondition: verify config matches |
| 8.8 | Clean up `validation.py` | 152 lines, 3 functions | BPR | Keep only `validation_best_model` |
| 8.9 | Resolve HMRM duality | `hmrm.py` vs `hmrm_new.py` | BPR* | Precondition: verify which is used |
| 8.10 | Delete legacy notebooks | 18K + 29K lines | BPR | Superseded by modular ETL |

### Semantic/correctness (BPR + SC)

| # | Refactor | Files | Type | Impact |
|---|----------|-------|------|--------|
| 8.11 | Make `paths.py` side-effect free | `src/configs/paths.py` | BPR | CLI-time checks |
| 8.12 | Stop eager `mtl_input` imports | `__init__.py` | BPR | `core.py` importable |
| 8.13 | Schema-driven dimensions | `FoldCreator`, `InputsConfig` | BPR* | Read from artifact |
| 8.14 | Split builders by embedding level | `builders.py` | **SC** | Reject invalid combos |
| 8.15 | Configurable sequence stride | `core.py` | BPR | Default unchanged |
| 8.16 | Fix MTL scheduler mismatch | `mtl_train.py` | **SC** | Account for cycling |
| 8.17 | Align MTL validation | `evaluate.py` | **SC** | Match training coverage |
| 8.18 | MTL split protocol | `create_fold.py` | **SC** | See SPLIT_PROTOCOL.md |

---

## 9. Refactoring Roadmap

### Phase 0: Invariants, Safety Net, Hard Decisions

1. Add `pyproject.toml` with pinned dependencies **(BPR)**
2. Decide MTL split protocol — see [SPLIT_PROTOCOL.md](SPLIT_PROTOCOL.md) **(SC)**
3. Add 3 layered regression fixtures (category, next, MTL). Run 3-5 times to
   calibrate per-fixture tolerance at 3× observed std dev. **(BPR)**
4. Add CLI flags (`--state`, `--engine`) to existing pipelines **(BPR)**
5. Produce split feasibility report per dataset configuration **(SC)**

**Dependencies:** None.

---

### Phase 1: Low-Risk BPR Cleanup

All BPR. Regression fixtures must pass after each change.

1. Make `paths.py` side-effect free (lazy validation)
2. Stop eager imports in `mtl_input/__init__.py`
3. Merge identical `evaluation.py` files into `src/train/shared/evaluate.py`
4. Fix `validation.py` scoping bug, delete unused functions **(BPR\*)**
5. Make `CategoryHeadMTL` import from `CategoryHeadEnsemble`
6. Fix typo: `foward_method` → `forward_method`
7. Delete: `requirements-bckp.txt`, `requirements_upgrad.txt`
8. Move `notebooks/create_inputs_hgi.py`, `hgi_texas.py` to `archive/`
9. Move `ANALYSIS.md`, `UPDATE.md` to `research/notes/`
10. Resolve HMRM duality **(BPR\*)**

**Dependencies:** Phase 0 regression fixtures.

---

### Phase 2: Data Contract Hardening

1. Define parquet column schemas **(BPR)**
2. Embedding-level validation: reject check-in for category **(SC)**
3. `FoldCreator` reads dimensions from artifact **(BPR\*)**
4. Configurable sequence stride **(BPR)**
5. Retain `userid` in next-task input **(SC)**
6. Materialize `POI → set(userids)` mapping **(BPR)**
7. Materialize sequence-to-POI mapping **(BPR\*)**
8. Implement MTL split decision from Phase 0 **(SC)**
9. Add explicit split manifests **(BPR)**

**Dependencies:** Phase 0 (split decision), Phase 1 (lazy imports).

---

### Phase 3: Configuration Unification (BPR / BPR\*)

1. Create `ExperimentConfig` dataclass
2. Remove hardcoded values from `mtl_train.py` **(BPR\*)**
3. Merge `category_config.py`, `next_config.py`, training parts of `model.py`
4. Add `ExperimentConfig.save()` / `.load()`
5. Add `RunManifest` (write-only JSON)
6. Serialize manifest alongside every results directory

**Dependencies:** Phase 1.

---

### Phase 4a: Shared Utilities and Model Registry (BPR)

1. `models/registry.py` with `@register_model`
2. Register all model variants
3. Positional encoding into `models/components/positional.py`
4. Consolidate all category heads into `models/heads/category.py`
5. Consolidate all next heads into `models/heads/next.py`
6. Extract shared training helpers
7. Extract shared evaluation helpers
8. `losses/registry.py` with `@register_loss`

Each task keeps its own thin runner. MTL keeps NashMTL/dual-dataloader logic.
Next keeps `_extract_diagnostics()`.

**Dependencies:** Phase 3.

---

### Phase 4b: MTL-Specific Cleanup (SC)

1. Fix scheduler mismatch
2. Align validation with training coverage
3. Integrate MTL runner with shared helpers

**Dependencies:** Phase 4a.

---

### Phase 5: Folder Tree Migration (BPR)

1. Move modules to target tree (section 6)
2. Add backward-compat re-exports with `DeprecationWarning`
3. Update all imports
4. Migrate test structure
5. Move embedding trainers to `research/embeddings/`
6. Move dormant variants to `experiments/archive/`
7. Import-chain smoke test before/during migration

**Sub-step order:** `configs/` → `models/` → `losses/` → `data/` →
`training/` → `tracking/` → `utils/`. Smoke test + pytest after each.
Items 5-6 as separate final commits.

**Shim deadline:** All shims removed by end of Phase 6.

**Dependencies:** Phases 1-4a.

---

### Phase 6: Script Consolidation (BPR)

1. `scripts/train.py` targeting final import paths
2. `argparse` CLI
3. Experiment configs in `experiments/configs/`
4. Separate `baselines/`, `ablations/`, `archive/`
5. Old pipelines as thin wrappers during transition
6. `scripts/evaluate.py` for checkpoint evaluation
7. Remove all transitional re-export shims
8. CLI smoke test (1 epoch, 1 fold)

**Dependencies:** Phase 5.

---

### Phase 7: Testing and Reproducibility (BPR)

1. Integration tests (2 folds, 3 epochs)
2. Upgrade Phase 0 fixtures to full integration tests
3. Audit seed usage reads from `ExperimentConfig.seed`
4. Fix test documentation
5. Test migration: ETL → schemas → model → trainer → CLI

**Dependencies:** Phases 5-6.

---

### Phase 8: Deeper Improvements (optional)

- Callback system (early stopping, diagnostics, checkpointing)
- Hydra (if sweeps become important)
- W&B or MLflow (if experiment volume justifies)
- DVC (if artifact management becomes painful)
- Simplify `IoPaths` (single parameterized class)

---

## 10. Migration Strategy

### Layered BPR Safety Net

MPS has non-deterministic behavior. Three layers:

1. **Deterministic unit checks** — model shapes, registry lookups, config
   round-trips. Must pass exactly.
2. **Artifact-shape checks** — fold count, CSV columns, JSON keys. Must pass
   exactly.
3. **Coarse metric comparison** — F1 within per-fixture calibrated tolerance
   (3× observed std dev from Phase 0 baseline).

### SC Rule

Document expected impact before the change. Metric delta must be justified
(valid SC may *lower* metrics — e.g., stricter splits remove leakage).
Establish new baselines after.

### BPR\* Rule

Verify precondition first. If it holds: BPR. If not: reclassify as SC.

### Validation Protocol

1. Classify: BPR, BPR\*, or SC?
2. Snapshot: run safety net, save baselines
3. Refactor: make the change
4. Compare: BPR → all 3 layers pass; SC → delta documented
5. Test: pytest with 0 new failures
6. Commit: one per logical change

### Import Compatibility

Transitional re-exports with `DeprecationWarning`. Pytest check enforces no
new imports from shim modules. Shims introduced Phase 5, removed end of Phase 6.

### File Moves, Not Copies

Always move, then update imports. Git tracks renames well.

---

## 11. Best Practices Applied

| Practice | Source | Application |
|----------|--------|-------------|
| **Config frozen with results** | Cookiecutter DS, Lightning-Hydra | `ExperimentConfig` + `RunManifest` |
| **Model registry** | timm, UvA DL Guide | `@register_model` + `create_model()` |
| **Shared helpers, thin runners** | Codex cross-review | Composable primitives, not god Trainer |
| **model / training / eval separation** | Universal consensus | Models are pure `nn.Module` |
| **Data contracts before training** | Data engineering | Schemas in Phase 2, training in Phase 4 |
| **Side-effect-free imports** | Python packaging | `paths.py` lazy, `__init__.py` lazy |
| **Run provenance** | PyTorch reproducibility docs | Manifest with hashes, seeds, device |
| **BPR/SC classification** | Codex cross-review | Every change labeled |
| **Layered safety net** | Codex cross-review | Unit → shape → coarse metrics |
| **Pre-train tests** | Eugene Yan | Shape, gradient flow, single-batch overfit |
| **Dataclass configs** | Python stdlib | Type-safe, zero dependencies |
| **Regression fixtures first** | Codex cross-review | Phase 0, not Phase 6 |
| **MTL split as hard gate** | Codex cross-review | Research validity before architecture |

---

## Sources

- Cookiecutter Data Science: https://cookiecutter-data-science.drivendata.org/
- Lightning-Hydra Template: https://github.com/ashleve/lightning-hydra-template
- UvA DL Research Projects Guide: https://uvadlc-notebooks.readthedocs.io/
- timm Model Registry: https://github.com/huggingface/pytorch-image-models
- How to Test ML Code (Eugene Yan): https://eugeneyan.com/writing/testing-ml/
- PyTorch Reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
