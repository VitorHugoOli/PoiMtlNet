# MTLnet Repository Refactoring Plan

## 1. Current-State Diagnosis

The repository implements a multi-task learning framework for POI prediction with two tasks (category classification, next-POI prediction) using hierarchical graph embeddings. It has ~24,400 lines of Python across 151 files.

**What exists:**
- **8 embedding engines** (`src/embeddings/`) — DGI, HGI, Check2HGI, POI2HGI, Time2Vec, Space2Vec, HMRM, FUSION
- **3 model tiers** — MTLnet (joint), single-task category (5+ variants), single-task next-POI (5+ variants)
- **5 loss functions** (`src/criterion/`) — NashMTL, FocalLoss, PCGrad, GradNorm, NaiveLoss
- **3 parallel training paradigms** — MTL, category-only, next-only, each with its own trainer, cross-validation, and evaluation
- **13 pipeline scripts** (`pipelines/`) — 7 embedding, 3 training, 2 data preparation, 1 fusion
- **Refactored experiment tracking** (`src/common/ml_history/`) — 10-file package with clean API
- **Test suite** — 255+ tests across 31 files
- **Academic paper** in `articles/CBIC___MTL/`

**Overall shape:** The codebase is structured by _task_ (`mtlnet/`, `category/`, `next/`) at every level — model, training, evaluation, configs, tests. This creates a 3×N matrix where each cell is a near-duplicate of the others, differing only in task-specific details.

---

## 2. Main Structural Problems

### 2.1 Pervasive Code Duplication

**Identical evaluation functions.** `src/train/category/evaluation.py` and `src/train/next/evaluation.py` are **character-for-character identical** (34 lines each). Both implement the same `evaluate()` function — collect predictions, compute `classification_report`. This is the definition of code that should be shared.

**Near-duplicate cross-validation orchestrators.** `src/train/category/cross_validation.py` (112 lines) and `src/train/next/cross_validation.py` (177 lines) follow the exact same pattern:
1. Loop over folds
2. Instantiate model (hardcoded class)
3. Compute class weights
4. Create optimizer/scheduler
5. Set model arch and params on history
6. Calculate FLOPs on first fold
7. Call `train()`
8. Call `evaluate()`
9. Store report, step history

The only differences: which model class to instantiate, which config to read, and next has `_extract_diagnostics()`.

**Three validation functions in `validation.py`.** `src/train/mtlnet/validation.py` contains three functions (`validation_best_model`, `validation_model`, `validation_model_by_head`) that are variations of the same "iterate over dataloaders, collect predictions, produce classification_report" pattern. `validation_model` (lines 58-101) has a scoping bug — the `with torch.no_grad()` block at line 69 does not cover the actual inference loop at line 76.

**Three evaluation files in MTL.** `evaluate.py` has `evaluate_model()` and `evaluate_model_by_head()` — the latter takes a `foward_method` parameter (note the typo) but does something different from the single-task `evaluate()`. None of these share code.

### 2.2 Configuration Scatter

Hyperparameters are spread across **6 files** with overlapping concerns:

| Setting | `model.py` | `category_config.py` | `next_config.py` | `mtl_train.py` (hardcoded) |
|---------|-----------|---------------------|------------------|-----------------------------|
| LR | `1e-4` | `1e-4` | `1e-4` | `0.0001` (line 259) |
| Epochs | 50 | **2** | 100 | — |
| Batch size | 2048 | 2048 | 512 | — |
| Weight decay | — | `0.05` | `0.01` | `5e-2` (line 261) |
| NUM_CLASSES | 7 | 7 | 7 | — |

The MTL trainer (`mtl_train.py:257-262`) hardcodes `lr=0.0001` and `weight_decay=5e-2` instead of reading from `MTLModelConfig`, despite receiving `learning_rate` as a parameter. The `learning_rate` parameter is only used for the scheduler's `max_lr` calculation (line 267: `max_lr=learning_rate * 10`).

`ModelParameters.INPUT_DIM` is marked deprecated but still used in `mtl_train.py:242`. Task-specific configs define their own `INPUT_DIM` separately.

### 2.3 No Experiment Configuration

There is **no single config object** that defines an experiment. To run a different experiment, you must:
1. Edit the `TRAINING_CONFIGS` list at the bottom of a pipeline script
2. Edit config dataclasses in `src/configs/` for hyperparameter changes
3. Edit the model class name hardcoded in `cross_validation.py`
4. Comment/uncomment lines (e.g., `mtl.pipe.py:89-103`, `cat_head/cross_validation.py:44` for class weights)

This means experiments are defined by _code edits_, not configuration. Two experiments cannot coexist, and there is no way to reproduce an older run without checking out the exact commit.

### 2.4 Pipeline Scripts as God Objects

The pipeline scripts in `pipelines/train/` combine:
- State/engine selection
- Fold creation
- History initialization
- Cross-validation orchestration
- Results display
- Error handling and logging

`cat_head.pipe.py` (168 lines) and `next_head.pipe.py` (171 lines) are ~80% identical in structure. `mtl.pipe.py` (125 lines) is simpler only because the cross-validation logic is pushed into `mtl_train.py`.

### 2.5 Model Variants Without Registry

There are **10+ model variants** across the codebase:

- Category: `CategoryHeadMTL`, `CategoryHeadSingle`, `CategoryHeadResidual`, `CategoryHeadGated`, `CategoryHeadEnsemble`, `CategoryHeadAttentionPooling`, `CategoryHeadTransformer`, `SEHead`, `DCNHead`
- Next: `NextHeadMTL`, `NextHeadSingle`, `NextHeadLSTM`, `NextHeadGRU`, `NextHeadTemporalCNN`, `NextHeadHybrid`, `NextHeadTransformerOptimized`

But they are selected by **editing imports and class names** in `cross_validation.py`. The enhanced models in `category_head_enhanced.py` and `next_head_enhanced.py` are imported but only one is used at a time. This makes it impossible to run the same training pipeline with different model variants without code changes.

### 2.6 `CategoryHeadMTL` is a Duplicate

`src/model/mtlnet/category_head.py:CategoryHeadMTL` and `src/model/category/category_head_enhanced.py:CategoryHeadEnsemble` are **the same architecture** with different class names. The MTL version was copied rather than imported.

### 2.7 Inconsistent Positional Encoding

- `NextHeadMTL` uses **sinusoidal** positional encoding (via `PositionalEncoding` class)
- `NextHeadSingle` uses **learned** positional embeddings + temporal decay bias

These are different architectural choices buried as implementation details, not exposed as configurable options.

### 2.8 Dead/Legacy Code

- `src/train/mtlnet/validation.py:validation_model()` has a scoping bug (lines 69-76) and is likely unused
- `src/train/mtlnet/validation.py:validation_model_by_head()` — unclear if used
- `src/train/mtlnet/evaluate.py:evaluate_model_by_head()` — takes a `foward_method` [sic] parameter, unclear usage
- `notebooks/create_inputs_hgi.py` (18K) and `notebooks/hgi_texas.py` (29K) — superseded by modular ETL
- `src/embeddings/hmrm/hmrm_new.py` alongside `hmrm.py` — unclear which is canonical
- `requirements-bckp.txt`, `requirements_upgrad.txt` — stale

### 2.9 Embedding Integration Points are Fragile

Embedding engines are integrated via `EmbeddingEngine` enum + `IoPaths` path routing, with 6 separate I/O path classes (`_DGIIoPath`, `_HGIIoPath`, etc.). Adding a new embedding requires:
1. Add to `EmbeddingEngine` enum
2. Create a new `_XxxIoPath` class
3. Register it in `IoPaths`
4. Create a pipeline script in `pipelines/embedding/`
5. Update fusion config if applicable

This is more ceremony than needed for "I have a new .parquet of embeddings."

### 2.10 Testing Gaps

Tests are thorough for models and utilities but **there are no integration tests** that run a mini training loop end-to-end (data → fold → train 2 epochs → evaluate). The test for training progress (`test_training_progress.py`) tests the progress bar, not the training.

---

## 3. Inferred Current Workflow

```
1. EMBEDDING GENERATION
   pipelines/embedding/{engine}.pipe.py
   → calls src/embeddings/{engine}/ modules
   → outputs: output/{engine}/{state}/*.parquet

2. INPUT PREPARATION
   pipelines/create_inputs.pipe.py
   → calls src/etl/mtl_input/builders.py
   → generates category input: [placeid, category, emb_0..emb_63]
   → generates next input: [emb_0..emb_575, next_category, userid]
   → outputs: output/{engine}/{state}/*.parquet

   (Optional) pipelines/fusion.pipe.py
   → concatenates multiple embedding sources
   → outputs: output/fusion/{state}/*.parquet

3. TRAINING
   pipelines/train/{mtl|cat_head|next_head}.pipe.py
   → calls src/etl/create_fold.py:FoldCreator
   → creates stratified 5-fold CV splits with class weights
   → calls src/train/{mtlnet|category|next}/cross_validation.py
     → per fold: instantiate model, optimizer, scheduler, criterion
     → call trainer.py:train() for N epochs
     → call evaluation.py:evaluate() on best model
     → store metrics in MLHistory
   → outputs: results/{engine}/{state}/folds/, plots/, model/, summary/
```

The workflow is linear and manual: edit script → run → inspect results. There is no CLI, no config files, no experiment registry.

---

## 4. Proposed Target Architecture

### Design Principles

1. **Shared abstractions for common patterns** — one evaluate function, one cross-validation loop, one model factory
2. **Config-driven experiments** — a single dataclass defines everything needed to reproduce a run
3. **Registry pattern for models and losses** — select by name from config, not by editing imports
4. **Embedding engines as plugins** — standard interface, minimal registration ceremony
5. **Pipelines as thin orchestrators** — all logic lives in `src/`, pipelines just wire things together

### Key Architectural Decisions

**Keep Python dataclasses** (no Hydra). The project is small enough that dataclass configs with CLI override via `argparse` gives 80% of the value with zero new dependencies. Hydra can be added later if sweeps become important.

**Don't adopt PyTorch Lightning**. The training loops are well-optimized for MPS with custom gradient accumulation and NashMTL integration. Lightning would add complexity without clear benefit for this specific codebase.

**Unify training into a single `Trainer` class** parameterized by task type. The three trainers share ~80% of their logic. The 20% that differs (MTL uses two dataloaders, different loss computation) can be handled by a strategy/callback pattern rather than three separate implementations.

**Keep embeddings in the repo** but behind a clean interface. They are research artifacts that belong with the project, but the training pipeline should only depend on "give me a tensor" — not on how embeddings were generated.

---

## 5. Proposed Folder Tree

```
mtlnet/
├── configs/                        # All configuration in one place
│   ├── __init__.py
│   ├── experiment.py               # ExperimentConfig: model + training + data
│   ├── model.py                    # ModelConfig variants (MTL, category, next)
│   ├── training.py                 # TrainingConfig (lr, epochs, scheduler, etc.)
│   ├── paths.py                    # IoPaths, EmbeddingEngine (kept as-is, simplified)
│   └── globals.py                  # DEVICE, CATEGORIES_MAP
│
├── models/                         # Pure nn.Module definitions
│   ├── __init__.py
│   ├── registry.py                 # @register_model + create_model()
│   ├── mtlnet.py                   # MTLnet, ResidualBlock, FiLMLayer
│   ├── heads/
│   │   ├── __init__.py
│   │   ├── category.py             # All category heads (Ensemble, Gated, SE, DCN, etc.)
│   │   └── next.py                 # All next heads (Transformer, LSTM, GRU, etc.)
│   └── components/
│       ├── __init__.py
│       └── positional.py           # PositionalEncoding, LearnedPositionalEmbedding
│
├── losses/                         # Loss functions
│   ├── __init__.py
│   ├── registry.py                 # @register_loss + create_loss()
│   ├── focal.py
│   ├── nash_mtl.py
│   ├── pcgrad.py
│   ├── gradnorm.py
│   └── naive.py
│
├── data/                           # Dataset, fold creation, data contracts
│   ├── __init__.py
│   ├── dataset.py                  # POIDataset
│   ├── folds.py                    # FoldCreator (from etl/create_fold.py)
│   ├── inputs/                     # Input generation (from etl/mtl_input/)
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── builders.py
│   │   ├── loaders.py
│   │   └── fusion.py
│   └── raw/                        # Raw data files (gitignored)
│
├── training/                       # Training and evaluation
│   ├── __init__.py
│   ├── trainer.py                  # Unified Trainer (handles single-task and MTL)
│   ├── cross_validation.py         # Unified CV runner
│   ├── evaluate.py                 # Single evaluate() function for all tasks
│   └── callbacks.py                # Early stopping, diagnostics extraction
│
├── tracking/                       # Experiment tracking (from common/ml_history/)
│   ├── __init__.py
│   ├── experiment.py
│   ├── fold.py
│   ├── metric_store.py
│   ├── best_tracker.py
│   ├── storage.py
│   └── display.py
│
├── utils/                          # Shared utilities
│   ├── __init__.py
│   ├── flops.py                    # FLOPs calculation
│   ├── mps.py                      # MPS support
│   └── progress.py                 # Training progress bar
│
├── embeddings/                     # Embedding generation (adjacent, not core)
│   ├── dgi/
│   ├── hgi/
│   ├── check2hgi/
│   ├── poi2hgi/
│   ├── time2vec/
│   ├── space2vec/
│   └── hmrm/
│
├── scripts/                        # CLI entrypoints (replaces pipelines/)
│   ├── train.py                    # python scripts/train.py --config experiments/hgi_florida.py
│   ├── generate_inputs.py          # Input generation
│   ├── generate_embeddings.py      # Embedding generation
│   └── evaluate.py                 # Standalone evaluation from checkpoint
│
├── experiments/                    # Experiment configs (Python dataclass instances)
│   ├── mtl_hgi_florida.py
│   ├── mtl_dgi_alabama.py
│   ├── baseline_category_only.py
│   └── ...
│
├── tests/
│   ├── conftest.py
│   ├── test_models/
│   ├── test_training/
│   ├── test_data/
│   ├── test_losses/
│   └── test_integration/           # End-to-end mini training tests
│
├── notebooks/
├── results/                        # Training outputs (gitignored)
├── articles/                       # Academic paper
├── docs/
└── pyproject.toml
```

---

## 6. Recommended Module Boundaries

### 6.1 Experiment Config (the "glue" object)

```python
# configs/experiment.py
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
    task_loss: str = "cross_entropy"    # or "focal"
    mtl_loss: str = "nash_mtl"          # or "pcgrad", "gradnorm", "naive"
    use_class_weights: bool = True

    # Cross-validation
    k_folds: int = 5
    seed: int = 42

    # Early stopping
    timeout: Optional[float] = None
    target_cutoff: Optional[float] = None
    early_stopping_patience: int = -1

    # Reproducibility
    def save(self, path: Path): ...

    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig': ...
```

This single object replaces the scatter across `MTLModelConfig`, `ModelParameters`, `InputsConfig`,
`CfgNextHyperparams`, `CfgNextModel`, `CfgNextTraining`, `CfgCategoryHyperparams`, `CfgCategoryModel`,
`CfgCategoryTraining`, plus hardcoded values in `mtl_train.py`.

### 6.2 Model Registry

```python
# models/registry.py
_REGISTRY: dict[str, type[nn.Module]] = {}

def register_model(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def create_model(name: str, **kwargs) -> nn.Module:
    return _REGISTRY[name](**kwargs)

def list_models() -> list[str]:
    return list(_REGISTRY.keys())
```

Each model class decorates itself: `@register_model("category_ensemble")`. The cross-validation loop
calls `create_model(config.model_name, **config.model_params)` instead of hardcoding a class.

### 6.3 Unified Evaluate Function

```python
# training/evaluate.py
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loaders: Union[DataLoader, tuple[DataLoader, DataLoader]],
    device: torch.device,
    model_state: Optional[dict] = None,
    forward_fn: Optional[Callable] = None,
) -> dict[str, dict]:
    """
    Evaluate model on one or two dataloaders.
    Returns {task_name: classification_report_dict}.
    """
```

This replaces: `category/evaluation.py:evaluate`, `next/evaluation.py:evaluate`,
`mtlnet/evaluate.py:evaluate_model`, `mtlnet/evaluate.py:evaluate_model_by_head`,
`mtlnet/validation.py:validation_best_model`, `mtlnet/validation.py:validation_model`,
`mtlnet/validation.py:validation_model_by_head` — **7 functions** collapsed into **1**.

### 6.4 Unified Trainer

```python
# training/trainer.py
class Trainer:
    def __init__(self, config: ExperimentConfig): ...

    def train_epoch(self, model, dataloaders, optimizer, scheduler, criteria, fold_history) -> dict:
        """One epoch. Handles single-task or MTL based on config.task_type."""
        if self.config.task_type == TaskType.MTL:
            return self._train_epoch_mtl(...)
        else:
            return self._train_epoch_single(...)

    def run_fold(self, fold_idx, fold_data, history) -> FoldHistory:
        """Full fold: init model, train, evaluate, return history."""
        model = create_model(self.config.model_name, **self.config.model_params)
        ...

    def run_cv(self, fold_results, history) -> MLHistory:
        """Cross-validation loop."""
        for fold_idx, fold_data in fold_results.items():
            self.run_fold(fold_idx, fold_data, history)
```

This replaces the three `cross_validation.py` files and the three `trainer.py` files.

### 6.5 Boundary Summary

```
ExperimentConfig  ←  defines what to run
       ↓
   Trainer         ←  orchestrates the run
       ↓
  create_model()   ←  builds the model by name
  create_loss()    ←  builds the loss by name
  FoldCreator      ←  creates data splits
  evaluate()       ←  computes metrics
  MLHistory        ←  tracks everything
```

Each component depends only on its interfaces, not on concrete implementations.

---

## 7. High-Impact Code Refactors

### 7.1 Merge Identical Evaluation Functions
**Files:** `src/train/category/evaluation.py`, `src/train/next/evaluation.py`
**Action:** Delete both, create single `training/evaluate.py`. Extend to handle MTL (two dataloaders) with an optional second loader parameter.
**Impact:** Eliminates 68 lines of pure duplication. Ensures any evaluation bugfix applies everywhere.

### 7.2 Remove `CategoryHeadMTL` Duplicate
**Files:** `src/model/mtlnet/category_head.py`, `src/model/category/category_head_enhanced.py`
**Action:** `CategoryHeadMTL` should import and alias `CategoryHeadEnsemble`, or both should be the same class.
**Impact:** Prevents the two implementations from diverging silently.

### 7.3 Fix `validation.py` Scoping Bug
**File:** `src/train/mtlnet/validation.py:58-101`
**Bug:** `validation_model()` declares `with torch.no_grad():` at line 69 covering only the list
initialization (lines 71-75), but the actual inference loop at line 76 runs **with gradients enabled**.
**Action:** Fix the scoping, then delete the function if unused.

### 7.4 Extract Positional Encoding into Shared Component
**Files:** `src/model/next/next_head.py:PositionalEncoding`, `src/model/next/next_head.py:NextHeadSingle`
(learned embeddings)
**Action:** Create `models/components/positional.py` with `SinusoidalPositionalEncoding` and
`LearnedPositionalEmbedding`. Each next-POI head selects which one to use via config parameter.
**Impact:** Makes the architectural choice explicit and configurable rather than buried in implementation.

### 7.5 Consolidate Model Variants into Registered Heads
**Files:** `category_head_enhanced.py` (5 classes), `next_head_enhanced.py` (5 classes),
`CategoryHeadTransformer.py`, `SEHead.py`, `DCNHead.py`
**Action:** Merge all category heads into one file `models/heads/category.py`, all next heads into
`models/heads/next.py`. Register each with `@register_model`. Delete the `_enhanced.py` naming convention.
**Impact:** Any model variant is selectable by name from config. No import-editing required.

### 7.6 Unify Cross-Validation Boilerplate
**Files:** `category/cross_validation.py`, `next/cross_validation.py`,
`mtlnet/mtl_train.py:train_with_cross_validation`
**Action:** Extract the common pattern (model init → optimizer → scheduler → class weights → FLOPs →
train → evaluate → step history) into a single `Trainer.run_fold()` method parameterized by
`ExperimentConfig`.
**Impact:** ~300 lines of near-duplicate code → ~100 lines of unified code. Adding a new model variant
requires zero changes to the training loop.

### 7.7 Remove Hardcoded Hyperparameters from `mtl_train.py`
**File:** `src/train/mtlnet/mtl_train.py:257-283`
**Problem:** `lr=0.0001`, `weight_decay=5e-2`, `eps=1e-8`, `max_norm=2.2`, `update_weights_every=4`,
`optim_niter=30` are hardcoded despite config classes existing.
**Action:** Read all values from `ExperimentConfig`. The function should receive the config, not
individual parameters.

### 7.8 Clean Up `validation.py`
**File:** `src/train/mtlnet/validation.py` (152 lines, 3 functions)
**Action:** Keep only `validation_best_model` (the one actually used in `mtl_train.py:331`). Delete
`validation_model` (buggy, likely unused) and `validation_model_by_head` (unclear usage). Then fold
the remaining function into the unified `evaluate()`.

### 7.9 Resolve HMRM Duality
**Files:** `src/embeddings/hmrm/hmrm.py`, `src/embeddings/hmrm/hmrm_new.py`
**Action:** Determine which is canonical. Delete the other. If both are needed, rename to indicate
purpose (e.g., `hmrm_v1.py`, `hmrm_v2.py`) with a note in CLAUDE.md.

### 7.10 Delete Legacy Notebook Scripts
**Files:** `notebooks/create_inputs_hgi.py` (18K), `notebooks/hgi_texas.py` (29K)
**Action:** These were superseded by the modular ETL system. Move to `archive/` or delete.

---

## 8. Refactoring Roadmap by Phases

### Phase 1: Low-Risk Structural Cleanup (1-2 days)

**What changes:**
1. Merge identical `evaluation.py` files into `src/train/shared/evaluate.py`
2. Fix `validation.py` scoping bug, delete unused validation functions
3. Make `CategoryHeadMTL` import from `CategoryHeadEnsemble` instead of duplicating
4. Delete `requirements-bckp.txt`, `requirements_upgrad.txt`
5. Move `notebooks/create_inputs_hgi.py` and `notebooks/hgi_texas.py` to `archive/`
6. Resolve HMRM duality (`hmrm.py` vs `hmrm_new.py`)
7. Fix typo: `foward_method` → `forward_method` in `evaluate.py`

**Why it matters:** Eliminates confusion and prevents divergent bug fixes. Zero behavioral change.

**Risks:** Minimal — all changes are renaming/deleting duplicates. Run the test suite after each change.

**Dependencies:** None.

**Migration:** Update imports in consumers. Run `pytest` to verify.

---

### Phase 2: Configuration Unification (2-3 days)

**What changes:**
1. Create `ExperimentConfig` dataclass that consolidates all hyperparameters
2. Remove hardcoded values from `mtl_train.py` (lr, weight_decay, NashMTL params)
3. Merge `category_config.py`, `next_config.py`, and the training parts of `model.py` into `ExperimentConfig`
4. Keep `ModelParameters` for architecture-specific constants (shared_layer_size, etc.) but reference from `ExperimentConfig.model_params`
5. Add `ExperimentConfig.save()` / `.load()` so configs are serialized alongside results
6. Deprecate `InputsConfig.TIMEOUT_TEST` / `TARGET` — move to experiment config

**Why it matters:** A researcher can look at one object to understand an experiment. Configs are saved
with results for reproducibility.

**Risks:** Medium — changes function signatures across trainers and pipelines. Must update all call sites.

**Dependencies:** Phase 1 (to avoid editing dead code).

**Migration:**
- Create `ExperimentConfig` alongside existing configs
- Update pipeline scripts one at a time to use `ExperimentConfig`
- Keep old config classes as deprecated aliases during transition
- Run the training pipeline once per task type to verify identical behavior

---

### Phase 3: Model Registry and Unified Training (3-5 days)

**What changes:**
1. Create `models/registry.py` with `@register_model` decorator
2. Register all model variants (10+ category heads, 6+ next heads, MTLnet)
3. Extract positional encoding into shared `models/components/positional.py`
4. Consolidate `category_head_enhanced.py` + `CategoryHeadTransformer.py` + `SEHead.py` + `DCNHead.py`
   into `models/heads/category.py`
5. Consolidate `next_head_enhanced.py` + `next_head.py` into `models/heads/next.py`
6. Create `Trainer` class that replaces three separate `cross_validation.py` + `trainer.py` pairs
7. Unified `evaluate()` replaces the 7 evaluation functions

**Why it matters:** Adding a new model variant becomes: (1) write the class with `@register_model`,
(2) specify the name in config. No training code changes needed. Experiment comparison becomes trivial.

**Expected benefits:**
- ~500 lines of duplicated code eliminated
- New model variants require zero changes to training pipeline
- Side-by-side comparison of model variants from config alone
- Single place to fix training bugs

**Risks:** Higher — touches the core training loop. Requires careful testing.

**Dependencies:** Phase 2 (config must be unified first).

**Migration:**
- Build `Trainer` alongside existing trainers
- Verify output equivalence on a small run (2 folds, 5 epochs)
- Switch pipelines one at a time
- Delete old trainers only after all pipelines migrate

---

### Phase 4: Pipeline and Entrypoint Cleanup (2-3 days)

**What changes:**
1. Replace `pipelines/train/{mtl,cat_head,next_head}.pipe.py` with single `scripts/train.py`
2. Add `argparse` CLI: `python scripts/train.py --experiment experiments/mtl_hgi_florida.py`
3. Create experiment definition files (Python dataclass instances)
4. Unified `scripts/generate_inputs.py` replaces `pipelines/create_inputs.pipe.py` + `pipelines/fusion.pipe.py`
5. Add `scripts/evaluate.py` for standalone evaluation from checkpoint

**Why it matters:** One entrypoint, one command. Experiments defined as files, not code edits.

**Risks:** Medium — researchers must adapt to the new CLI. Keep old pipeline scripts as deprecated wrappers during transition.

**Dependencies:** Phases 2 and 3.

**Migration:**
- Create new scripts alongside old pipelines
- Old pipelines become thin wrappers calling the new scripts
- Remove old pipelines after validating the new workflow

---

### Phase 5: Testing and Reproducibility (2-3 days)

**What changes:**
1. Add integration tests: end-to-end mini training (synthetic data, 2 folds, 3 epochs)
2. Add regression tests: known config → known metric range
3. Pin random seeds explicitly in `ExperimentConfig`
4. Serialize `ExperimentConfig` as JSON alongside every results directory
5. Add git commit hash to saved experiments
6. Create `pyproject.toml` with pinned dependencies
7. Move test structure to mirror the new source layout

**Why it matters:** Any experiment can be reproduced from its saved config + the commit hash.

**Risks:** Low — additive changes.

**Dependencies:** Phases 2-4.

---

### Phase 6: Deeper Architectural Improvements (optional, ongoing)

**What could change:**
1. Loss registry (`@register_loss`) mirroring the model registry
2. Callback system for training (early stopping, diagnostics, checkpointing)
3. Hydra integration if hyperparameter sweeps become important
4. W&B or MLflow integration for experiment tracking
5. Simplify `IoPaths` — replace 6 I/O path classes with a single parameterized class
6. `pyproject.toml` scripts section for CLI entrypoints

**When:** Only when the simpler system from Phases 1-5 proves insufficient.

---

## 9. Best Practices Applied and Why

| Practice | Source | Application |
|----------|--------|-------------|
| **Config frozen with results** | Cookiecutter Data Science, Lightning-Hydra Template | `ExperimentConfig.save()` serializes alongside results directory |
| **Model registry** | timm, UvA DL Guide | `@register_model` + `create_model()` — lightweight, no framework dependency |
| **Unified trainer** | PyTorch Lightning philosophy (not the framework) | Single `Trainer` class parameterized by task type — avoids 3× code duplication |
| **Separation: model / training / evaluation** | Universal consensus | Models are pure `nn.Module`. Training orchestrates. Evaluation is independent. |
| **Pre-train tests** | Eugene Yan "How to Test ML Code" | Output shape, gradient flow, single-batch overfit — catches real bugs cheaply |
| **Dataclass configs** | Python stdlib, appropriate for project size | Type-safe, IDE-friendly, zero dependencies. Hydra deferred until sweeps needed. |
| **CLI entrypoints** | MLOps best practices | `argparse` for now — simple, no dependencies, sufficient for reproducibility |
| **Baselines through same pipeline** | Research reproducibility guidelines | Same training loop for all models ensures fair comparison — only config differs |
| **No enterprise patterns** | Pragmatism for solo/small-team research | No dependency injection containers, no abstract factory hierarchies, no plugin systems |

---

## 10. Migration Strategy to Minimize Breakage

### Golden Rule: Never Break the Working Pipeline

At every phase, the existing `python pipelines/train/mtl.pipe.py` command must produce identical results.
New code is built alongside, validated, then switched.

### Validation Protocol

For each phase:
1. **Snapshot**: Run the current pipeline on a small config (1 state, 2 folds, 5 epochs). Save the metrics.
2. **Refactor**: Make the structural changes.
3. **Compare**: Run the same config through the refactored code. Verify metrics match within floating-point tolerance.
4. **Test**: `pytest` must pass with 0 new failures.
5. **Commit**: One commit per logical change, not per phase.

### Import Compatibility

During transition, old import paths should work via `__init__.py` re-exports:
```python
# src/train/category/evaluation.py (during transition)
from training.evaluate import evaluate  # Re-export from new location
```

Remove these shims only after all consumers are updated.

### File Moves vs. Copies

Never copy-and-modify. Always move the canonical implementation, then update imports. Git tracks
renames well when the content change is small.

---

## 11. What to Do Now vs. Later

### Do Now (Phase 1 — immediate, low risk, high clarity)

1. **Merge the two identical `evaluate()` functions** — most obvious win, zero behavioral change
2. **Fix the `validation_model()` scoping bug** — latent correctness issue in `validation.py:69-76`
3. **Make `CategoryHeadMTL` an alias** for `CategoryHeadEnsemble` — stop the silent divergence
4. **Delete dead code** — unused validation functions, legacy notebook scripts, backup requirements
5. **Resolve `hmrm.py` vs `hmrm_new.py`** — pick one, document it

### Do Next (Phases 2-3 — medium risk, highest leverage)

6. **Create `ExperimentConfig`** — single highest-leverage change for reproducibility and experiment speed
7. **Build model registry** — single highest-leverage change for experimentation flexibility
8. **Unify training into `Trainer`** — eliminates the most duplication

### Do Later (Phases 4-6 — only when the above is stable)

9. CLI entrypoints with `argparse`
10. Integration tests (end-to-end mini training)
11. Loss registry, callback system
12. External experiment tracking (W&B / MLflow)

### Do Never

- **PyTorch Lightning migration** — the custom MPS optimizations and NashMTL integration would fight it
- **Hydra** — unless hyperparameter sweeps become a bottleneck; dataclasses are sufficient
- **Abstract base classes for models** — the registry pattern is simpler and more flexible
- **Monorepo tooling** (Bazel, Pants) — massive overkill for this project size
