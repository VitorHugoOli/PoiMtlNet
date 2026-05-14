# CLAUDE.md - Project Guide

## Project Overview

**MTLnet**: Multi-task learning framework for POI (Point of Interest) prediction using hierarchical graph embeddings. Two tasks are jointly trained:
- **Category prediction**: Classify a POI's category from its embedding
- **Next-POI prediction**: Predict the next POI a user will visit given a sequence of past check-ins

## Project focus

The project's primary study is **check2hgi** — a check-in-level Check2HGI substrate for joint POI prediction (paper at BRACIS 2026).

**Two source-of-truth folders:**
- **Science**: [`docs/`](docs/) root — `README.md` (navigation landing), `AGENT_CONTEXT.md`, `NORTH_STAR.md`, `CHANGELOG.md`, `CLAIMS_AND_HYPOTHESES.md`, `CONCERNS.md`, `FINAL_SURVEY.md`, `MTL_ARCHITECTURE_JOURNEY.md`, `PAPER_BASELINES_STRATEGY.md`. Canonical numbers: [`docs/results/RESULTS_TABLE.md §0`](docs/results/RESULTS_TABLE.md). Per-experiment findings (F-trail): [`docs/findings/`](docs/findings/).
- **Paper**: [`articles/[BRACIS]_Beyond_Cross_Task/`](articles/[BRACIS]_Beyond_Cross_Task/) — BRACIS 2026 submission working folder (`AGENT.md` first if writing prose, then `PAPER_DRAFT.md`, `PAPER_STRUCTURE.md`, `STATISTICAL_AUDIT.md`, `TABLES_FIGURES.md`, `samplepaper.tex`, `references.bib`, `AUDIT_LOG.md`).

**Active follow-up studies** (layered on check2hgi) live under [`docs/studies/`](docs/studies/):
- [`docs/studies/canonical_improvement/`](docs/studies/canonical_improvement/) — 18-experiment slate to improve canonical Check2HGI.
- [`docs/studies/merge_design/`](docs/studies/merge_design/) — Designs A-M / Levers 1-6 / Phase 11.
- [`docs/studies/hgi_category_injection/`](docs/studies/hgi_category_injection/) — CLOSED (AZ falsified 2026-05-04). Read `STATUS.md`.

The earlier **fusion** study has been archived under [`docs/archive/fusion-study/`](docs/archive/fusion-study/) — concepts, results, claim catalog, leakage diagnoses, lab-notebook records all preserved intact.

> **Important**: fusion remains a **first-class engine** in the codebase even though the fusion *study* is archived. `EmbeddingEngine.FUSION` in `src/configs/paths.py`, `src/data/inputs/fusion.py`, `experiments/full_fusion_ablation.py`, `pipelines/fusion.pipe.py` are intact and supported. "Archived" applies to the study, not the engine.

**Operational documentation** (Colab, RunPod, Lightning, H100, local, Drive) lives at [`docs/infra/`](docs/infra/) — start with [`docs/infra/README.md`](docs/infra/README.md) for the by-machine decision tree.

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
└── docs/               # Project documentation (post-2026-05-14 reorg)
    ├── README.md           # ⭐ navigation landing — read this first
    ├── AGENT_CONTEXT.md    # check2hgi study briefing
    ├── NORTH_STAR.md       # committed champion config (B9 MTL recipe)
    ├── CHANGELOG.md        # timeline of findings + lessons
    ├── CLAIMS_AND_HYPOTHESES.md  # claim catalog (paper-facing whitelist banner inside)
    ├── CONCERNS.md         # risk audit log
    ├── FINAL_SURVEY.md     # substrate-axis 5-state matrix
    ├── MTL_ARCHITECTURE_JOURNEY.md  # supplementary narrative (F-trail)
    ├── PAPER_BASELINES_STRATEGY.md  # baseline-table mapping
    ├── results/            # canonical numbers (RESULTS_TABLE.md §0) + raw artefacts by phase
    ├── findings/           # paper-supporting per-experiment findings (F-trail) — read-only
    ├── studies/            # ACTIVE follow-up studies layered on check2hgi
    ├── infra/              # ⭐ operational docs (RunPod, Colab, Lightning, H100, local, Drive)
    ├── baselines/          # external baselines (BASELINE.md overview + per-task audits)
    ├── paper/              # section drafts (methods, results, limitations, appendix)
    ├── archive/            # archived studies + snapshots (fusion-study/, check2hgi-* historical)
    ├── context/            # task / embedding / architecture / optimizer / head background
    ├── datasets/           # dataset reference
    ├── thesis/             # paper thesis options A / B
    ├── plans/              # non-archive ablation plans
    ├── reports/            # status reports
    ├── scope/              # scoping decisions
    ├── review/             # dated critical reviews
    ├── launch_plans/       # historical launch plans (durable ops recipes are in infra/)
    ├── issues/             # generic project issues + check2hgi/ subdir
    ├── BRACIS_GUIDE.md     # conference submission guide
    ├── PAPER_FINDINGS.md   # legacy findings (revalidate, don't trust pre-bugfix)
    └── check2hgi_overview.tex  # paper LaTeX figure asset
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

## Running on remote/cloud machines

All operational documentation (Colab T4, RunPod 4090, Lightning A100/H100, H100 SSH, local M4 Pro, Drive download) lives under [`docs/infra/`](docs/infra/). Start at [`docs/infra/README.md`](docs/infra/README.md) — it has a decision tree by machine type.

Quick pointers:
- **Long Colab runs** — use [`notebooks/colab_check2hgi_mtl.ipynb`](notebooks/colab_check2hgi_mtl.ipynb) (self-contained template) per [`docs/infra/colab/README.md`](docs/infra/colab/README.md). Mandatory: detached-subprocess pattern for runs > 5 min.
- **Study-driven parallel runs** (multi-test enrollment) — use `scripts/study/colab_runner.py` + `notebooks/colab_study_runner.ipynb` per [`docs/infra/colab/study_runner.md`](docs/infra/colab/study_runner.md). Default `STUDY_DIR` is `docs/archive/fusion-study/` (legacy); override to target the current study.

## Branch-scoped study context

The primary check2hgi study lives at [`docs/`](docs/) root and is loaded automatically on every branch.

Active follow-up studies live under [`docs/studies/`](docs/studies/) — currently `canonical_improvement/`, `merge_design/`, `hgi_category_injection/`. Each has its own onboarding doc (`AGENT_PROMPT.md`, `STATE.md`, or `INDEX.md`).

When a branch is dedicated to one of those follow-up studies (or a new one), create a `CLAUDE.local.md` at the repo root pointing to the study's onboarding doc. The file is gitignored and branch-local. Example:

```markdown
# Branch-active study
This branch is the **canonical_improvement** study. Read first:
- `docs/studies/canonical_improvement/AGENT_PROMPT.md`
- `docs/studies/canonical_improvement/log.md`
```

If `CLAUDE.local.md` does not exist, the primary check2hgi study at `docs/` root is the default context.

## What changed (2026-05-14 reorg)

- check2hgi promoted from `docs/studies/check2hgi/` to `docs/` root; navigation landing at `docs/README.md`.
- F-trail (60+ per-experiment findings) moved to `docs/findings/`.
- Three open research workstreams carved out as standalone studies under `docs/studies/`: `canonical_improvement/`, `merge_design/`, `hgi_category_injection/` (CLOSED, see `STATUS.md`).
- Fusion study archived to `docs/archive/fusion-study/` (intact, with closure note). FUSION engine still first-class in code.
- Ops/infra docs (RunPod, Colab, Lightning, H100, local, Drive) consolidated under `docs/infra/`.
- Old `docs/RUNPOD_GUIDE.md`, `docs/COLAB_GUIDE.md`, `scripts/H100_FLCATX_PERVISIT_PROMPT.md` left as 1-line breadcrumbs.
- Dead branches pruned; `bracis` (BRACIS review) and `worktree-check2hgi-mtl` retained.
- Full record: [`docs/archive/MERGE_REORG_PLAN_2026-05-14.md`](docs/archive/MERGE_REORG_PLAN_2026-05-14.md).
