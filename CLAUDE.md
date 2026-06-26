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

**Follow-up studies** (layered on check2hgi) live under [`docs/studies/`](docs/studies/) — the status registry is [`docs/studies/README.md`](docs/studies/README.md) + the outcomes log [`docs/studies/log.md`](docs/studies/log.md). Highlights:
- [`docs/studies/archive/mtl_improvement/`](docs/studies/archive/mtl_improvement/) — **CLOSED 2026-06-12**: the C25 class-weighting confound dissolved the "MTL sacrifices reg" gap; champion **G (= canon v16, the `train.py --task mtl` default)** matches the STL reg ceiling + beats the cat ceiling +2.6…+4.1 (4 states × 4 seeds). Read [`FINAL_SYNTHESIS.md`](docs/studies/archive/mtl_improvement/FINAL_SYNTHESIS.md) first (incl. the corrections registry).
- [`docs/studies/closing_data/`](docs/studies/closing_data/) — **SCAFFOLDED, not launched**: the experimental engine for the NEW paper (cross-study re-eval + BRACIS-suite RUN_MATRIX inventory → pre-freeze gates → recipe+substrate freeze → full base regeneration once: STL baselines re-run + champion + suite cells, ALL states × 4 seeds × 5 folds). Read `AGENT_PROMPT.md` + `PLAN.md`.
- [`docs/studies/archive/canonical_improvement/`](docs/studies/archive/canonical_improvement/) — CLOSED; 18-experiment slate to improve canonical Check2HGI.
- [`docs/studies/merge_design/`](docs/studies/merge_design/) — Designs A-M / Levers 1-6 / Phase 11.
- [`docs/studies/archive/hgi_category_injection/`](docs/studies/archive/hgi_category_injection/) — CLOSED (AZ falsified 2026-05-04). Read `STATUS.md`.

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
    ├── future_works/       # forward-looking memos (deferred work)
    ├── infra/              # ⭐ operational docs (RunPod, Colab, Lightning, H100, local, Drive)
    ├── baselines/          # external baselines (BASELINE.md overview + per-task audits)
    ├── archive/            # archived studies, paper drafts, plans, reviews, scope memos
    ├── context/            # tasks, datasets, splits, metrics, embeddings, MTL archs, optimizers, heads + check2hgi_overview.tex
    ├── thesis/             # paper thesis options A / B
    ├── reports/            # status reports
    ├── issues/             # check2hgi/ bug-audit log (8 issue files, partial-open + fixed)
    ├── index.html          # comprehensive research-state summary page
    ├── BRACIS_GUIDE.md     # conference submission guide
    └── PAPER_FINDINGS.md   # legacy findings (revalidate, don't trust pre-bugfix)
```

## MTLnet Model

Architecture in `src/models/mtlnet.py`:

1. **Task-specific encoders**: 2-layer MLPs (`feature_size -> encoder_layer_size -> shared_layer_size`), one per task
2. **FiLM modulation** (`FiLMLayer`): Learns gamma/beta from task embeddings, applies `gamma * x + beta` to condition shared layers on task identity
3. **Shared backbone** (`ResidualBlock`): Stack of 4 residual blocks with LayerNorm + Linear + LeakyReLU + Dropout
4. **Task heads**:
   - `CategoryHeadEnsemble` (`src/models/category/category_ensemble/head.py`): Multi-path ensemble (3 parallel paths of variable depth 2-4), concatenated -> Linear -> LayerNorm -> GELU -> classifier. (Other category-head variants live alongside under `src/models/category/category_*/head.py`.)
   - `NextHeadMTL` (`src/models/next/next_mtl/head.py`): Transformer encoder (4 layers, 8 heads, norm_first) with positional encoding, causal masking, and attention-based sequence pooling -> classifier. (Post-2026-05-14 head-registry restructure; previously at `src/models/heads/next.py`.)

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

> ⚠ **MTL OOM was FIXED (2026-06-19/20) — do NOT believe "large states need a bigger GPU."** Region-MTL
> (`next_region`, C=1109…8501 classes) used to OOM (FL overlap; CA/TX believed infeasible on the A40). The
> driver was the **per-epoch metric accumulating full logits O(N·C)**, not the model. Fixed by **S1**
> (streaming train-metric, `mtl_cv.py`, default-on) + **S2** (chunked val-metric, `mtl_eval.py`,
> `MTL_CHUNK_VAL_METRIC=1`) + **dataset-on-GPU auto-fit** (`folds._dataset_device`) + the `<U32` builder fix —
> all **byte-identical**. **Verified: CA/TX champion-G MTL FIT the A40** (~11–13 GB peak). Full explanation:
> [`docs/studies/pre_freeze_gates/OOM_MEMORY_FIX.md`](docs/studies/pre_freeze_gates/OOM_MEMORY_FIX.md).

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
# Canonical CLI entrypoint (smoke run — no recipe overrides, will NOT match paper numbers)
python scripts/train.py --task mtl --state florida --engine hgi

# Or via pipeline wrappers (legacy)
python pipelines/train/mtl.pipe.py
```

Configure `state` (florida, alabama, etc.) and engine. Outputs go to `results/{engine}/{state}/` including fold metrics (CSV), classification reports (JSON), plots (PNG), and summary statistics.

### Run profiler + multi-fold fan-out (`docs/studies/train_perf_multifold/`)

Three opt-in dev tools added 2026-06-26 (all default-off → bare runs are byte-identical to before):

- **Run profiler / audit** (`--profile` or `MTL_PROFILE=1`; `src/training/profiling.py`). Ephemeral (lives only
  for the process, like logs — NOT in MLHistory/results): per-fold section timing (data/forward/backward/
  train_metric/eval), throughput (batch/s, samp/s), peak GPU mem, GPU util (pynvml), torch.compile recompile/
  graph-break counts, per-fold quality, and **pain-point flags** (GPU-starved, sync/data-bound, recompile/graph
  breaks). Logged at fold/run end; `MTL_PROFILE_JSON=<path>` dumps a transient report. Use it to find bottlenecks.
- **Multi-fold fan-out** — run the 5 folds of ONE execution as separate processes sharing ONE rundir:
  - `--only-folds 2,3` (run a subset of the canonical 5-split; like `--only-fold` but several),
  - `--run-id NAME` (fix the rundir leaf so N fold-processes write into one dir; implies `--per-fold-seed`),
  - `--per-fold-seed` (reseed `seed+fold_id` → fold-k is order-independent: a fanned-out fold == its place in a
    sequential run, **proven byte-identical** even under 5-way concurrency).
  - Orchestrate: `scripts/run_folds_fanout.sh <run_id> <folds_csv> <max_parallel> -- <train.py recipe…>`
    (per-fold inductor caches; throttled). Aggregate: `scripts/aggregate_folds.py <rundir>` reads the per-fold
    artifacts (named by REAL fold id) → `fold_aggregate.json`. ⚠ in a fan-out the per-process
    `summary/full_summary.json` is unreliable (each process knows only its fold) — read `fold_aggregate.json` or
    the canonical scorer (`a40_score_matched.py`), which glob `fold*_*` by real id.
- **Quality-neutral perf** (verified within fold-std, AL A/B): removed the 3 data-dependent `.any()` graph-breaks
  in the STAN reg head ("P1"; `graph breaks 10→2`; eager byte-identical, `tests/test_models/test_stan_mask_equivalence.py`),
  cached the per-epoch OOD train-label set, and pinned CPU-resident batches (CA/TX H2D). Excluded as
  quality-risking: fused AdamW, SDPA-in-STAN, bf16-default, removing the AMP gate, `num_workers>0`.
  > ⚠ **P1 + frozen-§0.1 bit-exactness.** P1 is byte-identical in EAGER, but under `--compile` (the board protocol)
  > removing the graph break shifts the inductor reduction order → the champion moves ≤0.3 pp/fold (within fold-std,
  > mean preserved; AL 63.18/69.73 → 63.44/69.82). P1 is **default-on** (perf). **To bit-exactly reproduce a frozen
  > §0.1 COMPILED cell, set `MTL_STAN_LEGACY_MASK=1`** (restores the guarded masking). The repro driver
  > `docs/studies/train_perf_multifold/run_al_baseline.sh` pins it; **the closing_data board drivers
  > (`scripts/closing_data/board_*.sh`) should pin `MTL_STAN_LEGACY_MASK=1` too when bit-reproducing a frozen cell.**

> ⚠ **DEFAULTS & ANTI-STUMBLE (2026-06-19, read [`docs/studies/pre_freeze_gates/DEFAULTS_AND_GUARDS.md`](docs/studies/pre_freeze_gates/DEFAULTS_AND_GUARDS.md)).** The champion **recipe** is now the DEFAULT: a bare `train.py --task mtl` auto-injects **v16** via `--canon` (`DEFAULT_CANON`, `src/configs/canon.py`) — the 6 "silently-wrong-flags" below are handled by the bundle. **But four board values are deliberately NOT global defaults** (flipping them silently breaks frozen-§0.1 reproduction): **MIN_SEQUENCE_LENGTH=10**, **stride-1 (overlap)**, **`--compile`**, **`--tf32`** live ONLY in the P3 board recipe/driver, not `core.py`/`canon.py`. `train.py` now emits WARN guards (`_preflight_canon_guards`; `MTL_STRICT=1` hard-fails) for the three silent stumbles: **dev-seed 42** (paper needs `--seed {0,1,7,100}`), **champion-recipe-on-wrong-substrate**, and **torch ≠ 2.11.0+cu128**. Never flip the four board values to global defaults; see the TRAPS list in that doc.
>
> ⚠ **CANONICAL VERSIONS (read `docs/results/CANONICAL_VERSIONS.md` first).** As of **2026-06-02** there are four pinned versions (v11/v12 paper-canon + code default; v13/v14 opt-in STL bases):
> - **v11** = the BRACIS **paper canon** (FROZEN): B9/H3-alt recipe, **GCN substrate**, **log_T-KD OFF**. `docs/results/RESULTS_TABLE.md §0.1` IS v11. The on-disk `output/check2hgi/<state>/` IS the frozen v11 GCN substrate — do NOT overwrite it.
> - **v12** = the **new code default**: v11 + **log_T-KD W=0.2 ON** (scoped to MTL `check2hgi_next_region`) + **ResLN encoder** (future builds). log_T-KD is paper-grade at AL/AZ + single-seed pilot at FL/CA/TX; ResLN is **STL-only, NO MTL benefit** (the regime finding).
> - **v13** = the **recommended STL / forward-MTL base** (opt-in, NOT a default — `--engine` is always explicit): engine **`check2hgi_resln_design_b`** (ResLN + POI2Vec@pool). Best STL dual-axis engine (equalises HGI reg at AL, keeps/widens cat); **STL-only, NO MTL benefit today** — it is the strongest base for future MTL work. Needs the POI2Vec teacher; built at all five states (AL/AZ/FL/CA/TX) as of 2026-05-30. Canonical `check2hgi` engine identity unchanged → paper-safe. See CANONICAL_VERSIONS §v13.
> - **v14** = the **NEW recommended STL / forward-MTL base** (opt-in, supersedes v13 — `embedding_eval` Part-1 CLOSED 2026-06-02): engine **`check2hgi_design_k_resln_mae_l0_1`** (ResLN+mae cat lever ⊕ Delaunay-POI-GCN reg lever [design_k], orthogonal stack). Leak-free multi-seed FL: next-cat 67.36 (≈ frozen-canon ≫ HGI) + next-reg 0.7024 (closes ~69% of the canon→HGI gap; design_k(gcn) reg 0.7034 closes ~78% but is −2.5pp cat; HGI keeps a −0.36pp reg edge). **STL-only, NO MTL benefit** (v14 or dual-substrate routing pilots) — the MTL cross-attn regime is the wall. design_k was wrongly discarded by a prior AL/AZ-only study; FL re-validation overturned it. Mechanisms graduated into `Check2HGIModule` (`reg_poi_mode`); canonical `check2hgi` untouched → paper-safe. See CANONICAL_VERSIONS §v14 + `docs/studies/archive/embedding_eval/FINAL_SYNTHESIS.md`.
> - **To reproduce v11 paper numbers from v12-default code: pass `--log-t-kd-weight 0.0` and use the frozen GCN substrate.**
>
> ⚠ **For Check2HGI MTL you MUST pass the full canonical recipe — bare defaults will NOT reproduce paper numbers.** Three classes of flag are silently wrong by default and each one alone drops a head by 10–30 pp:
>
> 1. **`--mtl-loss`** defaults to `nash_mtl` (with cvxpy/ECOS solver errors mid-training). The canonical recipe is `static_weight` with `--category-weight 0.75`.
> 2. **`--cat-head` / `--reg-head`** default to the preset's choice (cat=`next_mtl` Transformer, reg=`next_gru`). NORTH_STAR B9 uses `next_gru` (cat) and `next_getnext_hard` (reg, alias for `next_stan_flow`).
> 3. **`--task-b-input-type`** defaults to `checkin`. NORTH_STAR B9 specifies **`region` embeddings** for the reg task (`docs/NORTH_STAR.md` §Champion). Running with `checkin` for task_b drops AL reg Acc@10 from ~50 % to ~28 % (verified 2026-05-14 on A40).
>
> Two **v12 defaults flipped 2026-05-30** — these are the NEW intended default, but you MUST opt out for v11 paper reproduction:
> 4. **`--log-t-kd-weight`** now defaults to **0.2** (τ=1.0) for MTL `check2hgi_next_region` (was None/off). **For the v11 paper §0.1 numbers, pass `--log-t-kd-weight 0.0`.** Category-only / non-region / non-MTL runs are unaffected (stays 0.0).
> 5. **Check2HGI encoder** now defaults to **`resln`** for future embedding builds (`scripts/canonical_improvement/regen_emb_t3.py`). The on-disk substrate is still the frozen v11 GCN artifact (not rebuilt). For a v11 GCN rebuild, pass `--encoder gcn`. ResLN is STL-only — no MTL benefit.
>
> A **third default flipped 2026-06-03** (C21 fix — see `docs/CONCERNS.md §C21`, `docs/results/CANONICAL_VERSIONS.md §selector`):
> 6. **`--checkpoint-selector`** (which scalar picks the single joint MTL checkpoint) now defaults to **`geom_simple`** = `sqrt(cat_macroF1 · reg_Acc@10)` — the headline-aligned, validated selector (recovered +5.62 pp deployable reg vs the old one). The v11 paper-canon selector was the broken `0.5*(cat_f1+reg_f1)`; pass **`--checkpoint-selector joint_f1_mean`** to reproduce v11's JOINT/deployable numbers. **§0.1 itself is UNAFFECTED** — it reports per-task *diagnostic-best* epochs, which are selector-independent. MTL-only flag.
>
> Canonical NORTH_STAR B9 invocation (paper-grade at FL/CA/TX; small-state recipe is H3-alt, see `docs/NORTH_STAR.md`). With v12 defaults this runs **v12** (canonical + log_T-KD); add `--log-t-kd-weight 0.0` for **v11** paper-canon:
> ```bash
> python scripts/train.py --task mtl --task-set check2hgi_next_region \
>     --state {state} --engine check2hgi --seed 42 \
>     --epochs 50 --folds 5 --batch-size 2048 \
>     --model mtlnet_crossattn \
>     --mtl-loss static_weight --category-weight 0.75 \
>     --scheduler cosine --max-lr 3e-3 \
>     --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
>     --alternating-optimizer-step --alpha-no-weight-decay --min-best-epoch 5 \
>     --cat-head next_gru --reg-head next_getnext_hard \
>     --task-a-input-type checkin --task-b-input-type region \
>     --per-fold-transition-dir output/check2hgi/{state}
> #   v12 default: log_T-KD W=0.2 ON. For v11 paper-canon add:  --log-t-kd-weight 0.0
> ```
>
> H3-alt (small-state recipe, AL/AZ): drop `--alternating-optimizer-step`, `--alpha-no-weight-decay`, `--min-best-epoch 5`; replace `--scheduler cosine --max-lr 3e-3` with `--scheduler constant`. Heads + input-modality flags are identical.
>
> The per-fold log_T at `--per-fold-transition-dir` MUST be the **seed-tagged** files `region_transition_log_seed{S}_fold{N}.pt`. Build them via `python scripts/compute_region_transition.py --state {state} --per-fold --seed {S}` for every seed you train at — otherwise transitions leak across the train/val split at any seed ≠ 42.
>
> ✅ **`MTL_SKIP_INERT_LOGT=1` — the champion does NOT need per-fold log_T files (opt-in, 2026-06-26).** The champion runs the reg head α-prior OFF (`freeze_alpha=True`, `alpha_init=0.0`) **and KD off**, so the per-fold log_T is **inert** (`α·log_T = 0`; with no file the head builds `log_T=zeros` — both give `logits+0`). Setting `MTL_SKIP_INERT_LOGT=1` makes `train.py` **skip the per-fold log_T load + all its leak-guards** for any provably-inert config → you can run the champion with NO `region_transition_log_*.pt` present (no regeneration needed). **Byte-identical** to loading the file (verified AL multi-seed; `docs/studies/train_perf_multifold/log.md §Phase 4b`). Default OFF; an ACTIVE prior (learnable α, α_init≠0, or ANY `--log-t/c-kd-weight`/`--cat-kd-weight`>0) is **never** skipped — the guards still fire. Use the per-fold files only when the prior is actually live.
>
> ⚠ **STALE log_T preflight (2026-05-20 lesson — mtl_protocol_fix Phase 2 P5)**: `regen_emb_t3.py` does NOT rebuild the per-fold log_T files, and `scripts/train.py` does NOT validate their freshness. **An old log_T silently survives across regens** and can inflate reg-Acc@10 by **+8 pp at STL** / **+12 pp at MTL disjoint** (FL seed=42 stale May-6 log_T case). Before any MTL/STL run that uses `--per-fold-transition-dir`, MUST verify:
>
> ```bash
> # mtime check (necessary but not sufficient)
> stat -c '%y %n' output/check2hgi/{state}/region_transition_log_seed{S}_fold*.pt
> stat -c '%y %n' output/check2hgi/{state}/input/next_region.parquet
> # If log_T mtime < next_region.parquet mtime, rebuild log_T:
> python scripts/compute_region_transition.py --state {state} --per-fold --seed {S}
> # Optional content audit: hash compare before/after rebuild
> ```
>
> Tier-6 FL-MTL sweep results at `docs/results/canonical_improvement/T6_{1,2,4}_*florida_mtl/` used stale May-6 log_T (caveat preserved by self-consistency: baseline + variants used SAME stale log_T → relative falsifications hold; ABSOLUTE Acc@10 is biased by unknown sign-and-magnitude). When citing those numbers, cross-reference [`docs/results/mtl_protocol_fix/phase1_verdict.md §Stale log_T audit`](docs/results/mtl_protocol_fix/phase1_verdict.md).
>
> ⚠ **Development-seed vs reporting-seed split**: `seed=42` is the **development seed**. All recipe choices (B9 vs H3-alt, BS, LR, scheduler) were tuned at seed=42 across canonical_improvement. **§0.1 v11 paper-canonical numbers use seeds {0, 1, 7, 100}** — NOT 42 — to avoid development-seed contamination. At small states (AL/AZ) seed=42 ≈ multi-seed; at large states seed=42 overshoots §0.1 by **+3 pp (CA) / +8 pp (TX)** of pure development-seed bias. **When reporting paper-grade numbers, always use {0,1,7,100} multi-seed, not seed=42 alone.**

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

Follow-up studies live under [`docs/studies/`](docs/studies/) — see the status registry [`docs/studies/README.md`](docs/studies/README.md). Each has its own onboarding doc (`AGENT_PROMPT.md`, `STATE.md`, `INDEX.md`, or — for closed studies — `FINAL_SYNTHESIS.md`).

When a branch is dedicated to one of those follow-up studies (or a new one), create a `CLAUDE.local.md` at the repo root pointing to the study's onboarding doc. The file is gitignored and branch-local. Example:

```markdown
# Branch-active study
This branch is the **canonical_improvement** study. Read first:
- `docs/studies/archive/canonical_improvement/AGENT_PROMPT.md`
- `docs/studies/archive/canonical_improvement/log.md`
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
