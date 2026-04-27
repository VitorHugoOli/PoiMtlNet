# CLAUDE.md - Project Guide

## Project Overview

**MTLnet**: Multi-task learning framework for POI (Point of Interest) prediction using hierarchical graph embeddings. Two tasks are jointly trained:
- **Category prediction**: Classify a POI's category from its embedding
- **Next-POI prediction**: Predict the next POI a user will visit given a sequence of past check-ins

## File Architecture

```
├── data/               # Raw input data (checkins, misc) — gitignored
├── output/             # Generated embeddings, model inputs, folds — gitignored
├── results/            # Training results (metrics, models, plots) — gitignored
├── pipelines/
│   ├── embedding/      # Embedding generation (hgi.pipe.py, time2vec.pipe.py, etc.)
│   └── train/          # Training pipeline wrappers (delegate to scripts/train.py)
├── scripts/
│   ├── train.py        # Canonical CLI entrypoint (--task, --state, --engine)
│   ├── evaluate.py     # Checkpoint evaluation
│   ├── train_hydra.py  # Optional Hydra entrypoint
│   └── generate_feasibility_report.py
├── src/
│   ├── configs/        # ExperimentConfig, InputsConfig, paths, globals
│   ├── data/           # Fold creation, dataset, input builders, schemas
│   │   └── inputs/     # Modular input generation (core, builders, fusion, loaders)
│   ├── losses/         # Loss functions + registry (NashMTL, Focal, PCGrad, GradNorm, Naive)
│   ├── models/         # MTLnet model, head registry
│   │   ├── heads/      # CategoryHeadMTL, NextHeadMTL
│   │   └── components/ # Positional encoding
│   ├── tracking/       # MLHistory, FoldHistory, MetricStore, BestTracker, storage
│   ├── training/       # Evaluate, helpers, callbacks, shared_evaluate
│   │   └── runners/    # CV runners + trainers (mtl, category, next)
│   └── utils/          # FLOPs, MPS support, profiler, progress bar
├── research/
│   └── embeddings/     # Embedding trainers (dgi, hgi, check2hgi, poi2hgi, time2vec, space2vec, hmrm)
├── experiments/
│   ├── configs/        # Declarative ExperimentConfig constructors
│   ├── hydra_configs/  # Hydra YAML configs
│   └── archive/        # Archived notebooks and old scripts
├── tests/
│   ├── test_configs/       # ExperimentConfig, model config, paths
│   ├── test_data/          # Fold creation, input builders
│   ├── test_embeddings/    # Embedding utilities
│   ├── test_integration/   # End-to-end pipeline tests (synthetic data)
│   ├── test_losses/        # Loss functions (focal, nash, pcgrad, gradnorm)
│   ├── test_models/        # Model heads, MTLnet, next variants
│   ├── test_regression/    # Phase 0 safety net (calibrated F1 floors)
│   ├── test_tracking/      # MLHistory, fold tracking
│   ├── test_training/      # Training runners
│   └── test_utils/         # FLOPs, training progress
└── docs/               # Analysis documents, decisions log
    ├── studies/            # Per-study namespaced subdirs (e.g. studies/fusion/)
    ├── archive/            # Outdated (pre-bugfix) analyses — historical reference only
    ├── baselines/          # External baseline references (HAVANA, POI-RGNN, PGC)
    ├── context/            # Task / embedding / architecture background
    ├── thesis/             # Paper thesis options A / B
    ├── BRACIS_GUIDE.md     # Conference submission guide
    └── PAPER_FINDINGS.md   # Legacy findings (revalidate, don't trust pre-bugfix)
```

## MTLnet Model

Architecture in `src/models/mtlnet.py`:

1. **Task-specific encoders**: 2-layer MLPs (`feature_size -> encoder_layer_size -> shared_layer_size`), one per task
2. **FiLM modulation** (`FiLMLayer`): Learns gamma/beta from task embeddings, applies `gamma * x + beta` to condition shared layers on task identity
3. **Shared backbone** (`ResidualBlock`): Stack of 4 residual blocks with LayerNorm + Linear + LeakyReLU + Dropout
4. **Task heads**:
   - `CategoryHeadMTL` (`src/models/heads/category.py`): Multi-path ensemble (3 parallel paths of variable depth 2-4), concatenated -> Linear -> LayerNorm -> GELU -> classifier
   - `NextHeadMTL` (`src/models/heads/next.py`): Transformer encoder (4 layers, 8 heads, norm_first) with positional encoding, causal masking, and attention-based sequence pooling -> classifier

Key methods: `shared_parameters()` and `task_specific_parameters()` enable gradient manipulation for MTL optimizers.

## Data Preparation

**Modular input system** in `src/data/inputs/`:
- `core.py`: `generate_sequences()` (sliding windows), `create_embedding_lookup()`, `create_category_lookup()`
- `builders.py`: High-level input builders for category and next tasks
- `fusion.py`: Multi-embedding fusion input generation
- `loaders.py`: Data loading utilities

**Sequence generation** (`src/data/inputs/core.py`):
- Non-overlapping sliding windows of size 9 + 1 target from user check-in histories
- Short sequences padded with -1
- Two modes: POI-level embeddings (same POI = same vector) or check-in-level embeddings (contextual per visit)

**Pipeline orchestration** (`pipelines/create_inputs.pipe.py`):
- `generate_category_input()`: Create category task inputs
- `generate_next_input_from_poi()`: Next-POI with POI-level embeddings
- `generate_next_input_from_checkins()`: Next-POI with check-in-level embeddings
- Parallel processing with configurable workers

**Output formats**:
- Category: `[placeid, category, emb_0, ..., emb_63]` -> parquet
- Next: `[emb_0, ..., emb_575, next_category, userid]` -> parquet
- Fusion: Variable dimensions based on concatenated embeddings

**Fold creation** (`src/data/folds.py`):
- `FoldCreator`: Stratified 5-fold CV, batch size 2048
- Category data kept as 1D vectors; next data reshaped to `(samples, window=9, embedding_dim)`
- Class weights computed for weighted CrossEntropyLoss (preferred over weighted sampling)

## Training Pipeline

Key files: `src/training/runners/mtl_cv.py`, `scripts/train.py`

**Training loop** (`mtl_cv.py`):
- Per fold: initialize MTLnet, optimizer, scheduler, criteria
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.05, eps=1e-8)
- **Scheduler**: OneCycleLR (max_lr=1e-3, 50 epochs)
- **Gradient accumulation**: 2 steps
- **MTL balancing**: NashMTL (n_tasks=2, max_norm=2.2, update_weights_every=4, optim_niter=30)
- **Early stopping**: By validation F1; optional timeout and target F1 cutoffs
- Mixed batch iteration cycles the shorter dataloader to match the longer one
- FLOPs profiled on first fold

**Running training**:
```bash
# Canonical CLI entrypoint
python scripts/train.py --task mtl --state florida --engine hgi

# Or via pipeline wrappers (legacy)
python pipelines/train/mtl.pipe.py
```

Configure `state` (florida, alabama, etc.) and engine. Outputs go to `results/{engine}/{state}/` including fold metrics (CSV), classification reports (JSON), plots (PNG), and summary statistics.

## Configurations

**`src/configs/experiment.py`**:
- `ExperimentConfig`: Unified dataclass (64 fields, 3 factory methods: `default_mtl`, `default_category`, `default_next`)
- `DatasetSignature`: Streaming SHA-256 for data integrity
- `RunManifest`: Experiment reproducibility metadata

**`src/configs/model.py`**:
- `InputsConfig`: EMBEDDING_DIM=64, SLIDE_WINDOW=9, PAD_VALUE=0

**`src/configs/paths.py`**:
- `EmbeddingEngine` enum: DGI, HGI, HMRM, TIME2VEC, SPACE2VEC, SPHERE2VEC, CHECK2HGI, POI2HGI, **FUSION**
- `IoPaths`: Centralized path management with `get_embedd()`, `get_category()`, `get_next()`, `get_results_dir()`, and `load_*()` methods
- FUSION routing: Automatically routes FUSION engine to fusion-specific paths
- Respects `$DATA_ROOT` env var (default: `data/`)

**`src/configs/globals.py`**:
- `DEVICE`: Auto-detects MPS (Apple Silicon) / CUDA / CPU
- `CATEGORIES_MAP`: Category label mapping

## Embeddings

Each engine lives in `research/embeddings/<engine>/` and may have its own CLAUDE.md.

| Engine | Directory | Description |
|--------|-----------|-------------|
| DGI | `research/embeddings/dgi/` | Deep Graph Infomax - 64-dim POI embeddings |
| HGI | `research/embeddings/hgi/` | Hierarchical Graph Infomax - 256-dim embeddings |
| Check2HGI | `research/embeddings/check2hgi/` | Check-in level HGI - contextual embeddings per visit |
| POI2HGI | `research/embeddings/poi2hgi/` | POI-level HGI variant |
| Time2Vec | `research/embeddings/time2vec/` | Temporal embeddings from check-in timestamps |
| Space2Vec | `research/embeddings/space2vec/` | Spatial embeddings from coordinates |
| Sphere2Vec | `research/embeddings/sphere2vec/` | Spherical-RBF location encoder (sphereM variant) - 64-dim POI embeddings |
| HMRM | `research/embeddings/hmrm/` | Heterogeneous Mobility Representation Model - 107-dim |
| **FUSION** | `src/data/inputs/fusion.py` | Multi-embedding fusion - concatenates multiple embeddings |

## Loss Functions

All in `src/losses/`, with `registry.py` for dynamic registration:

| File | Class | Description |
|------|-------|-------------|
| `nash_mtl.py` | `NashMTL` | Primary MTL loss - Nash equilibrium gradient balancing via cvxpy/ECOS solver |
| `focal.py` | `FocalLoss` | Handles class imbalance with `(1-pt)^gamma` weighting (gamma=2.0) |
| `pcgrad.py` | `PCGrad` | Projects conflicting task gradients to reduce interference |
| `gradnorm.py` | `GradNorm` | Balances gradient magnitudes across tasks |
| `naive.py` | `NaiveLoss` | Dynamic alpha/beta weighted sum with clamped adjustment |

## Utilities

**`src/tracking/`** - Experiment tracking:
- `MLHistory`: Top-level manager, context manager + iterator over folds
- `FoldHistory` / `MetricStore`: Per-fold per-task train/val metrics (loss, accuracy, F1)
- `storage.py`: Serializes to JSON, CSV, plots
- `best_tracker.py`: Tracks best model state per metric

**`src/utils/`** - Model profiling and helpers:
- `flops.py`: FLOPs calculation and parameter counts
- `profiler.py` / `profile_reporter.py` / `profile_exporter.py`: Layer-wise operation profiling
- `progress.py`: `TrainingProgressBar` (extended tqdm) + `zip_longest_cycle()`
- `mps.py`: Apple Silicon MPS cache management

**`src/data/dataset.py`**: `POIDataset` wrapper for PyTorch DataLoader.

## Running Pipelines

**Training** (recommended):
```bash
python scripts/train.py --task mtl --state florida --engine hgi
python scripts/train.py --task category --state florida --engine poi2hgi
python scripts/train.py --task next --state florida --engine hgi
```

**Evaluation**:
```bash
python scripts/evaluate.py --checkpoint results/hgi/florida/model.pt
```

**Embedding generation**:
```bash
python pipelines/embedding/hgi.pipe.py
python pipelines/embedding/time2vec.pipe.py
python pipelines/embedding/poi2hgi.pipe.py
```

**Input preparation**:
```bash
python pipelines/create_inputs.pipe.py
```

All pipelines are configured by editing variables at the top of each script (state, embedding engine, etc.).

## Running on Google Colab (T4)

For long runs that would saturate the local M4 Pro (notably FL 5f×50ep — ~50 min on T4 vs ~5 h on MPS), use the **Colab template + guide**:

- **Template notebook:** [`notebooks/colab_check2hgi_mtl.ipynb`](notebooks/colab_check2hgi_mtl.ipynb) — self-contained, mirrors the canonical `scripts/train.py` north-star B3 CLI from `docs/studies/check2hgi/NORTH_STAR.md`. Drop in, edit `STATE`, run cells in order.
- **Operational guide:** [`docs/COLAB_GUIDE.md`](docs/COLAB_GUIDE.md) — covers Drive layout, branch hygiene, the **detached-subprocess** launch pattern (mandatory for runs > 5 min, since MCP/cell timeouts will SIGINT a foreground `!{cmd}`), memory pitfalls (DataLoader-worker fork OOM, MRR pairwise-comparison OOM on FL's 4702 regions, between-fold CUDA fragmentation), the `feat/colab-gpu-perf` perf-pass changelog, results inspection, and a verification recipe for confirming a `git pull → relaunch` actually loaded fresh code.

The Colab perf optimisations live on **`feat/colab-gpu-perf`** (branched off `worktree-check2hgi-mtl`). All changes are quality-neutral and MPS-safe — verified via 24/24 metric+eval unit tests + AZ 5f×50ep regression (within 1 σ of NORTH_STAR.md reference) + 1f×2ep MPS smoke test on the M4 Pro.

For the *study-driven* parallel-machine workflow (consumes `state.json`-enrolled tests, packages tarballs back), use `scripts/study/colab_runner.py` + `notebooks/colab_study_runner.ipynb` instead — that's the orchestration path documented in `docs/studies/fusion/AGENT_CONTEXT.md` for the fusion track.

## Branch-scoped study context

If a file `CLAUDE.local.md` exists at the repo root, read it — it points to
the active study on this branch (e.g. `docs/studies/<study>/AGENT_CONTEXT.md`).

If `CLAUDE.local.md` does not exist, proceed without study context. Do NOT
assume any particular study is active; branches may be working on unrelated
experiments.
