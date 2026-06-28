# SYSTEM_REFERENCE — the main parts + env of MTLnet

> Practical operator map: the champion config, the substrates/engines, every train-time
> env var, the main code parts, and the one-command board recipe. Complements `CLAUDE.md`
> (the agent guide) and `docs/results/CANONICAL_VERSIONS.md` (the version pins).
> Last updated 2026-06-26.

## 1 · The current best model — champion-G

`mtlnet_crossattn_dualtower` (model) + `next_gru` (cat head) + `next_stan_flow_dualtower`
(reg head, **prior-OFF**: `freeze_alpha=True alpha_init=0.0`). Unweighted CE, MTL loss
`static_weight` with `category-weight 0.75`, scheduler `onecycle max-lr 3e-3`
(per-head LRs cat/reg/shared = 1e-3/3e-3/1e-3 — ⚠ **but these are INERT under onecycle**:
a scalar `max-lr` broadcasts to all param groups, so every head actually peaks at 3e-3; the
per-head flags do nothing here. See [`future_works/per_head_lr_onecycle_fix.md`](future_works/per_head_lr_onecycle_fix.md)),
checkpoint selector `geom_simple`
(`√(cat_macroF1 · reg_Acc@10)`). Substrate engine `check2hgi_dk_ovl` (gated stride-1
overlap, MIN_SEQ=10); per-fold priors dir = the v14 substrate `check2hgi_design_k_resln_mae_l0_1`.
**The champion is prior-OFF + KD-OFF → the per-fold log_T is INERT** (α·log_T = 0), so it is
not needed at all (see `MTL_SKIP_INERT_LOGT`, default-on).

**Canonical board numbers** (`docs/studies/closing_data/RESULTS_BOARD.md §1` — the single source
of truth; champion-G MTL vs STL ceilings, seed 0 × 5 folds, fp32-matched scorer):

| State | regions | STL cat | MTL cat | STL reg | MTL reg | precision |
|---|---:|---:|---:|---:|---:|---|
| AL | 1109 | 55.87 | **63.56** | 69.99 | **69.81** | fp32 |
| AZ | 1547 | 57.13 | **63.39** | 59.40 | **59.34** | fp32 |
| FL | 4703 | 75.15 | **79.82** | 76.71 | **77.28** | fp32 |
| CA | 8501 | 70.26 | **77.33** | 63.48 | **65.66** | bf16 |
| TX | 6553 | 69.95 | **77.51** | 64.96 | **67.02** | fp32 |
| Istanbul | 520 | 53.20 | **59.89** | 74.80 | **74.28** | fp32 |

Story: MTL **beats the cat ceiling everywhere** (+4.7…+7.7 pp) and **beats the reg ceiling at
the large region counts** (FL/CA/TX), **matches within δ=2 pp at the small** (AL/AZ/Istanbul).

## 2 · Canonical versions, engines, substrates

Pins live in `docs/results/CANONICAL_VERSIONS.md`. Quick map:
- **v11** = BRACIS paper canon (frozen GCN substrate `check2hgi`, log_T-KD OFF). `RESULTS_TABLE.md §0.1` IS v11.
- **v12** = code default = v11 + log_T-KD W=0.2 (MTL reg only) + ResLN encoder for future builds.
- **v13 / v14** = opt-in STL / forward-MTL bases (`check2hgi_resln_design_b` / `check2hgi_design_k_resln_mae_l0_1`).
- **v16** = `DEFAULT_CANON` (`src/configs/canon.py`) — the champion-G recipe bundle a bare `train.py --task mtl` auto-injects via `--canon`.

**Engines** (`EmbeddingEngine`, `src/configs/paths.py`): DGI, HGI, HMRM, TIME2VEC, SPACE2VEC,
SPHERE2VEC, CHECK2HGI, POI2HGI, FUSION. The board champion runs on **`check2hgi_dk_ovl`** (the
gated stride-1 overlap variant). Istanbul has no `dk_ovl` dir — its overlap is baked in-place
on the `check2hgi` engine (build via `scripts/second_dataset/`).

**On-disk substrates** live under gitignored `output/<engine>/<state>/` (symlinked to `/dados`
on the a40-wk box). `--engine` is always explicit (never defaulted) — you pick the state's substrate.

## 3 · Train-time env vars

Set before `python scripts/train.py …`. **Default = unset** unless noted.

| Env var | What it does | When to use |
|---|---|---|
| **`MTL_DISABLE_AMP=1`** | Force **fp32** (no autocast). | **Required for big states** on the A40 (bf16 grad-NaN at large C, fp16 overflow). |
| `MTL_AUTOCAST_BF16=1` | bf16 autocast arm (+ `MTL_DISABLE_AMP_EVAL=1`). | Only safe at small states on the A40; the board's bf16 cells (CA). |
| **`MTL_CHUNK_VAL_METRIC=1`** | S2 chunked val-metric (chunks the [N,C] logits). | **Big-C states** (FL/CA/TX) — prevents the val-metric OOM. Board-default-on. |
| `MTL_STREAM_TRAIN_METRIC=0` | Opt OUT of the S1 streaming train-metric (default ON). | Debug only; streaming is the OOM fix for C>256. |
| **`MTL_COMPILE_DYNAMIC=1`** | One symbolic-shape compiled graph (vs recompiling per batch shape). | Board protocol (`--compile`); collapses the warmup storm. |
| `MTL_COMPILE_MODE=…` / `MTL_COMPILE_CACHE_LIMIT=N` | inductor mode / dynamo recompile-cache limit (default 64). | Tuning; raise the limit if dynamo silently falls back to eager. |
| `MTL_DATASET_GPU=1` | Force the dataset GPU-resident. | **Small states only.** NEVER for CA/TX/FL (forces ~31 GB redundant copies → OOM). |
| `MTL_RAM_HEADROOM_GB=N` | Host-RAM headroom guard (default 16). | Lower (e.g. `4`) for tiny datasets like Istanbul; negative to force after measuring. |
| **`MTL_SKIP_INERT_LOGT`** | **Skip the per-fold log_T load+guards when the prior is provably inert. DEFAULT-ON** (the champion is always inert). | `=0` opts out (legacy always-load+guard). Byte-identical; frees the champion from needing log_T files. |
| `MTL_STAN_LEGACY_MASK=1` | Restore the pre-P1 guarded `.any()` STAN masking (graph breaks 2→10). | **Rarely** — it is SLOWER and gives **no** bit-exact compiled repro (the compiled number is inductor-cache-session-governed; eager is the ground truth). Leave OFF. |
| `MTL_STAN_FP32_ATTN=1` | Run the STAN masked-softmax in fp32 under bf16/fp16 autocast. | bf16 large-state mitigation; no-op under true fp32. (Unvalidated on the real NaN states.) |
| `MTL_STRICT=1` | Fail loud on the preflight guards (dev-seed, wrong-substrate, torch version) + the non-finite step guard aborts. | Board runs; catches silent stumbles. |
| `MTL_NAN_GUARD=1` | Log the grad-norm trajectory every 100 batches. | Debugging divergence (CA ep30 collapse class). |
| `MTL_PROFILE=1` (or `--profile`) / `MTL_PROFILE_JSON=<path>` | Ephemeral run profiler: per-fold section timing, throughput, peak GPU mem, recompile/graph-break counts, pain-point flags. | Finding bottlenecks. Zero-overhead when off; NOT in MLHistory. |
| `P1_CHUNK_VAL_METRIC=1` / `P1_S2_AUTO_BUDGET_GB=N` | STL **reg** ceiling val-metric chunking / GPU-val budget. | The `p1_region_head_ablation.py` STL reg path at big C. |
| `TORCHINDUCTOR_CACHE_DIR=…` | Per-process inductor cache (avoid contention across co-scheduled procs). | Always set a distinct dir per concurrent compiled run. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduce fragmentation OOM. | Board default. |
| `REGION_EMB_ENGINE=<engine>` | Override the engine the region-embedding lookup reads. | Dual-substrate routing experiments. |
| `DATA_ROOT=<dir>` | Raw-data root (default `data/`). | Relocating data. |

**Multi-fold fan-out env** (`scripts/run_folds_fanout.sh` + `train.py --only-folds/--run-id/--per-fold-seed`):
`FANOUT_LOGDIR=<dir>` sets the per-fold log dir. The fan-out's `--per-fold-seed` reseeds `seed+fold_id`
→ an order-independent baseline, NOT bit-identical to a frozen sequential cell.

## 4 · Main code parts (train flow)

```
scripts/train.py            CLI entrypoint (--task/--state/--engine/--canon); _run_single_task (STL), MTL via cv runner
src/configs/canon.py        DEFAULT_CANON (v16) — the champion recipe bundle auto-injected by --canon
src/data/folds.py           FoldCreator; _create_check2hgi_mtl_folds (the champion fold builder, user-disjoint SGKF);
                            _resolve_task_input, _load_and_validate_check2hgi_data, _classify_pois, _resolve_per_fold_priors
src/training/runners/
  mtl_cv.py                 train_with_cross_validation + train_model (the hot loop). Helpers: _build_mtl_optimizer,
                            _build_scheduler, _build_task_criteria, _apply_stream_freezes, _log_t/c/cat_kd_loss,
                            _optimizer_micro_step (should_step body), _run_validation_epoch, guard_finite_step
  mtl_eval.py               evaluate_model + chunked/streamed metrics
src/training/profiling.py   RunProfiler (the --profile tool)
src/models/
  mtl/mtlnet_crossattn_dualtower/   the champion model
  next/next_gru/                    cat head
  next/next_stan_flow_dualtower/    reg head (α·log_T prior plumbing; the gate _LEGACY_STAN_MASK lives in next_stan/head.py)
```

The MTL path goes train.py → `train_with_cross_validation` → `train_model`. The STL ceilings use
`next_cv`/`category_cv` (NOT `train_model`). The fold split is user-disjoint StratifiedGroupKFold.

## 5 · The one-command champion (board recipe)

Canonical driver: `scripts/closing_data/board_h100_mtl.sh <state> <fp32|bf16> <cache_suffix>`. On the
a40-wk box (fp32, single A40):

```bash
export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_<state>
python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine check2hgi_dk_ovl \
  --state <state> --seed 0 --epochs 50 --folds 5 --batch-size 2048 \
  --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
  --cat-head next_gru --reg-head next_stan_flow_dualtower \
  --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
  --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
  --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
  --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
  --model mtlnet_crossattn_dualtower --compile --tf32 \
  --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/<state> --no-checkpoints
```

`--canon none` + the explicit class-weight flags are required (a bare run trips the wrong-substrate
canon-guard under `MTL_STRICT=1`). Score with `scripts/closing_data/a40_score_matched.py <rundir> --seed 0`.

## 6 · The a40-wk box constraints

- Single **NVIDIA A40, 46 GB**. **fp32 for big states** (bf16 grad-NaN at large C). Never drop batch < 2048.
- Big-state datasets stay **CPU-resident** (auto-fit); never `MTL_DATASET_GPU=1` for CA/TX/FL.
- **Single-GPU fan-out gives NO wall-clock speedup** — a fold that saturates the GPU solo (e.g. FL: 97% util,
  17.46 b/s) makes 2-parallel ~7-10% *slower* (contention + wave-structure). Fan-out's value is multi-GPU
  scaling / resumability / `--only-folds` granularity. Small states (GPU-underutilized solo) can benefit from 2-parallel.
- venv at `/home/vitor.oliveira/.venv` (NOT conda — the closing_data docs' conda note is H100-only). torch 2.11.0+cu128.
- Big data on `/dados` via `output/` + `data/` symlinks.
