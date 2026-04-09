# CLAUDE.md - Project Guide

## Project Overview

**MTLnet**: Multi-task learning framework for POI (Point of Interest) prediction using hierarchical graph embeddings. Two tasks are jointly trained:
- **Category prediction**: Classify a POI's category from its embedding
- **Next-POI prediction**: Predict the next POI a user will visit given a sequence of past check-ins

### 📚 Additional Documentation

- **[FUSION_GUIDE.md](FUSION_GUIDE.md)** - Multi-embedding fusion system (Space2Vec + HGI + Time2Vec)
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - New modular architecture (completed 2026-02-03)
- **[walkthrough.md](walkthrough.md)** - Step-by-step project walkthrough

## File Architecture

```
├── data/               # Raw input data (checkins, misc)
├── output/             # Generated embeddings, model inputs, folds (per engine/state)
├── results/            # Training results (metrics, models, plots)
├── pipelines/
│   ├── embedding/      # Embedding generation (hgi.pipe.py, time2vec.pipe.py, etc.)
│   └── train/          # Training pipelines (mtl.pipe.py, cat_head.pipe.py, next_head.pipe.py)
├── src/
│   ├── model/
│   │   ├── mtlnet/     # MTL model (mtl_poi.py, category_head.py, next_head.py)
│   │   ├── category/   # Standalone category model
│   │   └── next/       # Standalone next-POI model
│   ├── train/
│   │   ├── mtlnet/     # MTL training (mtl_train.py, evaluate.py, validation.py)
│   │   ├── category/   # Category-only training
│   │   └── next/       # Next-POI-only training
│   ├── etl/            # create_input.py, create_fold.py
│   ├── criterion/      # Loss functions (NashMTL, FocalLoss, PCGrad, GradNorm, NaiveLoss)
│   ├── configs/        # model.py, paths.py
│   ├── embeddings/     # Embedding engines (dgi/, hgi/, check2hgi/, poi2hgi/, time2vec/, space2vec/, hmrm/)
│   └── common/         # Utilities (ml_history/, calc_flops/, training_progress.py)
├── notebooks/          # Analysis notebooks
└── tests/              # Unit, regression, and integration tests
    ├── test_configs/       # ExperimentConfig, model config, paths
    ├── test_data/          # ETL, fold creation, input builders
    ├── test_embeddings/    # Embedding utilities
    ├── test_integration/   # End-to-end pipeline tests (synthetic data)
    ├── test_losses/        # Loss functions (focal, nash, pcgrad, gradnorm)
    ├── test_models/        # Model heads, MTLnet, next variants
    ├── test_regression/    # Phase 0 safety net (calibrated F1 floors)
    ├── test_tracking/      # MLHistory, fold tracking
    └── test_utils/         # FLOPs, training progress
```

## MTLnet Model

Architecture in `src/model/mtlnet/`:

1. **Task-specific encoders** (`mtl_poi.py`): 2-layer MLPs (`feature_size → encoder_layer_size → shared_layer_size`), one per task
2. **FiLM modulation** (`mtl_poi.py: FiLMLayer`): Learns gamma/beta from task embeddings, applies `gamma * x + beta` to condition shared layers on task identity
3. **Shared backbone** (`mtl_poi.py: ResidualBlock`): Stack of 4 residual blocks with LayerNorm + Linear + LeakyReLU + Dropout
4. **Task heads**:
   - `CategoryHeadMTL` (`category_head.py`): Multi-path ensemble (3 parallel paths of variable depth 2-4), concatenated → Linear → LayerNorm → GELU → classifier
   - `NextHeadMTL` (`next_head.py`): Transformer encoder (4 layers, 8 heads, norm_first) with positional encoding, causal masking, and attention-based sequence pooling → classifier

Key methods: `shared_parameters()` and `task_specific_parameters()` enable gradient manipulation for MTL optimizers.

## Data Preparation

**New modular system** (see `REFACTORING_SUMMARY.md` for details):
- `src/etl/mtl_input/` - Modular input generation system
- `pipelines/create_inputs.pipe.py` - Pipeline orchestration
- `src/etl/create_input.py` - *DEPRECATED* (backward compatible)

**Sequence generation** (`src/etl/mtl_input/core.py`):
- `generate_sequences()`: Non-overlapping sliding windows of size 9 + 1 target from user check-in histories. Short sequences padded with -1.
- `create_embedding_lookup()`: POI → embedding dictionary with padding support
- `create_category_lookup()`: POI → category mapping
- Two modes: POI-level embeddings (same POI = same vector) or check-in-level embeddings (contextual per visit).

**Pipeline orchestration** (`pipelines/create_inputs.pipe.py`):
- `generate_category_input()`: Create category task inputs
- `generate_next_input_from_poi()`: Next-POI with POI-level embeddings
- `generate_next_input_from_checkins()`: Next-POI with check-in-level embeddings
- Parallel processing with configurable workers

**Output formats**:
- Category: `[placeid, category, emb_0, ..., emb_63]` → parquet
- Next: `[emb_0, ..., emb_575, next_category, userid]` → parquet
- Fusion: Variable dimensions based on concatenated embeddings

**Fold creation** (`src/etl/create_fold.py`):
- `FoldCreator`: Stratified 5-fold CV, batch size 2048
- Category data kept as 1D vectors; next data reshaped to `(samples, window=9, embedding_dim)`
- Class weights computed for weighted CrossEntropyLoss (preferred over weighted sampling)

## Training Pipeline

Key files: `src/train/mtlnet/mtl_train.py`, `pipelines/train/mtl.pipe.py`

**Training loop** (`mtl_train.py: train_with_cross_validation`):
- Per fold: initialize MTLnet, optimizer, scheduler, criteria
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.05, eps=1e-8)
- **Scheduler**: OneCycleLR (max_lr=1e-3, 50 epochs)
- **Gradient accumulation**: 2 steps
- **MTL balancing**: NashMTL (n_tasks=2, max_norm=2.2, update_weights_every=4, optim_niter=30)
- **Early stopping**: By validation F1; optional timeout and target F1 cutoffs
- Mixed batch iteration cycles the shorter dataloader to match the longer one
- FLOPs profiled on first fold

**Running the pipeline**:
```bash
python pipelines/train/mtl.pipe.py
```
Configure `state` (florida, alabama, etc.) and `embedd_engine` (EmbeddingEngine.DGI, HGI, etc.) in the script. Outputs go to `results/{engine}/{state}/` including fold metrics (CSV), classification reports (JSON), plots (PNG), and summary statistics.

## Configurations

**`src/configs/model.py`**:
- `InputsConfig`: EMBEDDING_DIM=64, SLIDE_WINDOW=9, PAD_VALUE=0
- `MTLModelConfig`: NUM_CLASSES=7, BATCH_SIZE=2048, EPOCHS=50, LR=1e-4, K_FOLDS=5
- `ModelParameters`: SHARED_LAYER_SIZE=256, NUM_HEADS=8, NUM_LAYERS=4, NUM_SHARED_LAYERS=4

**`src/configs/paths.py`**:
- `EmbeddingEngine` enum: DGI, HGI, HMRM, TIME2VEC, SPACE2VEC, CHECK2HGI, POI2HGI, **FUSION**
- `IoPaths`: Centralized path management with `get_embedd()`, `get_category()`, `get_next()`, `get_results_dir()`, and `load_*()` methods
- FUSION routing: Automatically routes FUSION engine to fusion-specific paths
- Respects `$DATA_ROOT` env var (default: `data/`)

## Embeddings

Each engine lives in `src/embeddings/<engine>/` and may have its own CLAUDE.md with detailed documentation.

| Engine | Directory | Description |
|--------|-----------|-------------|
| DGI | `dgi/` | Deep Graph Infomax - 64-dim POI embeddings from graph structure |
| HGI | `hgi/` | Hierarchical Graph Infomax - 256-dim embeddings from multi-level graphs |
| Check2HGI | `check2hgi/` | Check-in level HGI - contextual embeddings per visit |
| POI2HGI | `poi2hgi/` | POI-level HGI variant |
| Time2Vec | `time2vec/` | Temporal embeddings from check-in timestamps |
| Space2Vec | `space2vec/` | Spatial embeddings from coordinates |
| HMRM | `hmrm/` | Heterogeneous Mobility Representation Model - 107-dim |
| **FUSION** | `etl/embedding_fusion.py` | **Multi-embedding fusion - concatenates multiple embeddings (128+ dims)** |

**Note**: See `FUSION_GUIDE.md` for details on multi-embedding fusion and `REFACTORING_SUMMARY.md` for the new modular architecture.

## Loss Functions

All in `src/criterion/`:

| File | Class | Description |
|------|-------|-------------|
| `nash_mtl.py` | `NashMTL` | Primary MTL loss - Nash equilibrium gradient balancing via cvxpy/ECOS solver |
| `FocalLoss.py` | `FocalLoss` | Handles class imbalance with `(1-pt)^gamma` weighting (gamma=2.0) |
| `pcgrad.py` | `PCGrad` | Projects conflicting task gradients to reduce interference |
| `gradnorm.py` | `GradNorm` | Balances gradient magnitudes across tasks |
| `NaiveLoss.py` | `NaiveLoss` | Dynamic alpha/beta weighted sum with clamped adjustment |

## Utilities

**`src/common/ml_history/`** - Experiment tracking:
- `MLHistory`: Top-level manager, context manager + iterator over folds
- `FoldHistory` / `TaskTrainMetric`: Per-fold per-task train/val metrics (loss, accuracy, F1)
- `HistoryStorage`: Serializes to JSON, CSV, plots
- `FlopsMetrics`: FLOPs, params, memory, inference time

**`src/common/calc_flops/`** - Model profiling:
- `calculate_model_flops()`: Compute FLOPs and parameter counts
- `ModelProfiler`: Layer-wise operation profiling

**`src/common/training_progress.py`**:
- `TrainingProgressBar`: Extended tqdm with multi-dataloader support
- `zip_longest_cycle()`: Cycles shorter dataloaders to match the longest

**Other**: `poi_dataset.py` (POIDataset wrapper), `mps_support.py` (Apple Silicon MPS cache management)

## Running Pipelines

**MTL training** (main pipeline):
```bash
python pipelines/train/mtl.pipe.py
```

**Single-task training**:
```bash
python pipelines/train/cat_head.pipe.py   # Category only
python pipelines/train/next_head.pipe.py  # Next-POI only
```

**Embedding generation**:
```bash
python pipelines/embedding/hgi.pipe.py
python pipelines/embedding/time2vec.pipe.py
python pipelines/embedding/poi2hgi.pipe.py
```

**Input preparation**:
```bash
python pipelines/create_inputs.py
```

All pipelines are configured by editing variables at the top of each script (state, embedding engine, etc.).
