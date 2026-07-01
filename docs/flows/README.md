# Execution Flows — how to run the main pipelines (CANON)

**This doc is the canonical "how to run it" reference for the current paper state.** It documents the three
main flows end-to-end — **(A) v14 embedding substrate**, **(B) gated-overlap inputs**, **(C) train** — plus the
key execution params and the operational miscellaneous (profiler, multi-fold fan-out, env vars).

> ⚠ **MAINTENANCE CONTRACT.** This doc tracks the **current canon**. When a new canon is adopted (new substrate
> version, new champion recipe, new default), **update this doc in the same change** and bump the "Canon as of"
> line + the changelog at the bottom. The numbers/recipe here must always match `docs/results/CANONICAL_VERSIONS.md`,
> `docs/NORTH_STAR.md`, and `docs/studies/closing_data/RESULTS_BOARD.md`.

**Canon as of 2026-07-01:** champion **v17** (= v16 + bs8192 + `--onecycle-per-head-lr`; `DEFAULT_CANON`). Base **G / v16** (`mtlnet_crossattn_dualtower` + `next_gru` cat + `next_stan_flow_dualtower`
reg, prior-OFF) trained on the **`check2hgi_dk_ovl`** engine (gated stride-1 overlap of the **v14** substrate
`check2hgi_design_k_resln_mae_l0_1`), **bs=2048, fp32**, OneCycle max-lr 3e-3, static_weight cw=0.75, geom_simple
selector. Paper board = `docs/studies/closing_data/RESULTS_BOARD.md §1`.
> 🏆 **CHAMPION = `--canon v17` (now the DEFAULT; AL/AZ/FL done, CA/TX running at n=20):** **bs=8192 + per-head cat-lr 1e-3** via
> `MTL_ONECYCLE_PER_HEAD_LR=1` — beats champion board-wide (AL +1.0/AZ +2.3 cat; FL +0.17cat/+0.20reg & +7% faster).
> See `docs/studies/closing_data/perhead_lr_n20.md`. Promote → update this doc.

Environment: `source /home/vitor.oliveira/.venv/bin/activate` (or `.venv/bin/python`); `export PYTHONPATH=src`;
torch **2.11.0+cu128**; A40 (large states need **fp32**). `output/` and `data/` are symlinks to `/dados/poimtlnet/`.

---

## Pipeline overview

```
raw check-ins ─▶ (A) v14 embedding substrate ─▶ (B) gated-overlap inputs ─▶ (C) train MTL ─▶ results/board
   data/          check2hgi_design_k_resln_mae_l0_1   check2hgi_dk_ovl          champion-G
                  (embeddings + region_embeddings)     (stride-1 windowing,      (cat + next-region)
                                                         symlinks v14 embeddings)
```
The v14 embeddings are built **once per state** and frozen; the overlap engine only re-windows them; training
reads the overlap engine. Do **not** rebuild/clobber the frozen v14 substrate.

---

## (A) Build the v14 embedding substrate — `check2hgi_design_k_resln_mae_l0_1`

v14 = design_k (Delaunay-POI-GCN reg lever) ⊕ ResLN+MAE cat lever. Built once per state. Canonical orchestrator:
`scripts/_v14_run/build_ge.sh` (CA/TX: `scripts/substrate_protocol_cleanup/build_v13_catx.sh`, stage D→v14).
The ordered chain (per `<State>` capitalized / `<state>` slug):

**Raw data** (`data/` → `/dados/poimtlnet/data/`): `checkins/` (Gowalla US-state + `massive_steps_istanbul/`)
+ `miscellaneous/` (shapefiles/geojson). The base build reads these.

```bash
# 0. Prereq — canonical check2hgi base (v11 GCN, 500ep): region label maps + fold groups + sequences.
#    Entrypoint pipelines/embedding/check2hgi.pipe.py (edit STATE/config at top of file, then run);
#    internally: check2hgi.create_embedding(state, args) + generate_next_input_from_checkins(state, CHECK2HGI).
#    build_ge.sh Stage A wraps this. → output/check2hgi/<state>/

# 1. Prereq — HGI Delaunay edges (spatial graph design_k imports)
PYTHONPATH=src:research .venv/bin/python research/embeddings/hgi/preprocess.py \
    --city <State> --shapefile <state_shapefile.shp>      # → output/hgi/<state>/temp/edges.csv

# 2. Prereq — POI2Vec teacher (design_k regularizes toward it, λ=0.1)
.venv/bin/python scripts/substrate_protocol_cleanup/run_poi2vec.py \
    --city <State> --epochs 100 --device cuda             # → output/hgi/<state>/poi2vec_poi_embeddings_<State>.csv

# 3. BUILD v14 (bare invocation IS the v14 default: encoder=resln, mae=0.3, Delaunay GCN, anchor=0.1)
.venv/bin/python scripts/probe/build_design_k_delaunay.py --state <state> \
    --out-suffix resln_mae_l0_1 --epochs 500 --device cuda
#   → output/check2hgi_design_k_resln_mae_l0_1/<state>/{embeddings,poi_embeddings,region_embeddings}.parquet
#                                                     /temp/{checkin_graph.pt, sequences_next.parquet}

# 4. POSTBUILD — MTL input parquets + seed-42 log_T staged into the v14 dir
bash scripts/substrate_protocol_cleanup/postbuild_design_substrate.sh \
    check2hgi_design_k_resln_mae_l0_1 <state>

# 5. (OPTIONAL — NOT needed for the prior-OFF champion; only for prior-ON KD configs or STL reg ranking)
#    Per-fold seeded log_T. The champion SKIPS this entirely (see below). Build only if you flip the prior ON.
for S in 0 1 7 100; do
  .venv/bin/python scripts/compute_region_transition.py --state <state> --per-fold --seed $S --n-splits 5
done   # → output/check2hgi/<state>/region_transition_log_seed<S>_fold{1..5}.pt (orchestrator copies into v14 dir)
```
**Train consumes:** `region_embeddings.parquet` (reg target) + `embeddings.parquet` (cat) + the overlap engine's
windowing (see B). **log_T is fully INERT for the prior-OFF champion** — `MTL_SKIP_INERT_LOGT=1` (default-on)
hits an early-return in `mtl_cv.py` **before** the `.pt` is ever loaded (proven byte-identical). The champion
command's `--per-fold-transition-dir` is therefore a **no-op kept only so flipping the prior ON works without
editing the command** — the champion runs byte-identically with it removed. Step 5 is skippable for the champion.

---

## (B) Build the gated-overlap inputs — `check2hgi_dk_ovl`

The board engine: **v14 embeddings re-windowed at stride-1 (overlapping), gated, MIN_SEQ=10**. Embeddings are
**symlinked** from v14 (NOT recomputed); only `next.parquet` / `sequences_next.parquet` / `next_region.parquet`
are rebuilt. Canonical builder:

> The windowing **logic lives in `src`** (`core.generate_sequences` + `builders._resolve_emit_tail` /
> `generate_next_input_from_checkins`, which take `stride`/`min_sequence_length`/`emit_tail`).
> `build_overlap_probe_engine.py` is only a **thin driver** that symlinks v14 embeddings and calls that src logic
> with the board values. **stride=1 / min_seq=10 are passed by the driver and deliberately NOT code defaults** —
> flipping `core.py`'s global `MIN_SEQ=5` / `stride=None` would desync the frozen v11/v14 rebuild and break
> byte-identical §0.1 reproduction (DEFAULTS_AND_GUARDS TRAP #1/#2). So they stay engine/recipe-scoped, not global.

```bash
# build (argv: state stride min_seq) — stride=1, min_seq=10 are board-recipe-only (global default is 5/non-overlap)
PYTHONPATH=src .venv/bin/python scripts/mtl_improvement/build_overlap_probe_engine.py <state> 1 10
#   → output/check2hgi_dk_ovl/<state>/{embeddings.parquet→v14, input/next.parquet, input/next_region.parquet,
#                                       temp/sequences_next.parquet, input/next_build_provenance.json}
```

**The "gate" (M1 tail-gate):** at stride=1, `emit_tail` auto-resolves to **false** (`_resolve_emit_tail`,
`src/data/inputs/builders.py:62`), dropping out-of-bounds last-POI-target tail windows. The gate **HELPS** (AL
gated 63.44/70.36 @ 96,326 rows ≫ ungated 60.99/68.15 @ 108,073) — it removes last-POI skew. The danger is a
**manual ungated rebuild left stale on disk** (trains the wrong windowing silently).

**ALWAYS verify provenance before trusting any overlap number:**
```bash
cat output/check2hgi_dk_ovl/<state>/input/next_build_provenance.json
# REQUIRED: "stride":1, "emit_tail":false, "min_sequence_length":10
# row-count tell: AL gated 96,326 (ungated 108,073); FL gated ~1.27M (ungated ~1.378M)
```
**Automated enforcement:** `src/data/folds.py:692` reads the provenance at train time; if stride==1 with
emit_tail==True or min_seq∉{None,10} it WARNs and **hard-fails under `MTL_STRICT=1`** (which the board sets),
printing the exact rebuild command. Don't use the non-board overlap builders (`build_hgi_overlap_inputs.py` =
the HGI Tbl-2 arm; `build_istanbul_stride1.py` = Istanbul base) for the champion board.

---

## (C) Train — champion-G MTL (current canon)

Run the orchestrator `scripts/closing_data/p3_board.sh` (states `alabama arizona georgia florida california texas`
× seeds `0 1 7 100`; `DRY_RUN=1` prints the exact plan). Or the single-cell command:

```bash
export PYTHONPATH=src MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_board
export MTL_DISABLE_AMP=1            # fp32. REQUIRED for large states (CA/TX); auto-fp32 also forces it for reg C>2000
                                   # (FL/CA/TX) but NOT AL/AZ (C 1109/1547 < 2000) — set it explicitly for the board

.venv/bin/python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --engine check2hgi_dk_ovl --state <state> --seed <seed> --epochs 50 --folds 5 --batch-size 2048 \
    --mtl-loss static_weight --category-weight 0.75 \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --checkpoint-selector geom_simple \
    --no-reg-class-weights --no-cat-class-weights --canon none --compile --tf32 \
    --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/<state>   # INERT no-op for the champion (prior-OFF); kept only so flipping the prior ON works without editing the command
```
Score: `scripts/closing_data/a40_score_matched.py <rundir> --seed <S>` (cat macro-F1 diag-best + reg Acc@10).

> **Champion — `--canon v17`** (now `DEFAULT_CANON`, 2026-07-01): adds `--batch-size 8192` + `--onecycle-per-head-lr`
> (→ `MTL_ONECYCLE_PER_HEAD_LR=1`, so `--cat-lr 1e-3` actually applies; without it = inert v16-uniform-3e-3).
> Beats v16 at every tested state (AL/AZ/FL n=20; CA/TX n=20 running). v16 stays reproducible via `--canon v16`. See `CANONICAL_VERSIONS §v17`.

### (C2) STL ceilings (the board's Δcat/Δreg denominators)
RESULTS_BOARD §1 reports MTL gains vs the single-task ceilings. To produce them:
```bash
# cat ceiling (STL category)
.venv/bin/python scripts/train.py --task next --state <state> --engine check2hgi_dk_ovl \
    --cat-head next_gru --seed <S> --epochs 50 --folds 5 ...     # then: scripts/closing_data/score_stl_cat_ceiling.py
# reg ceiling (STL next-region head ablation)
.venv/bin/python scripts/p1_region_head_ablation.py next_stan_flow a0 --state <state> --seed <S>
```
(See `PLAN.md §1` for the exact STL recipes; the board's `score_*` scripts aggregate them.)

### (D) Second dataset — Istanbul
Istanbul (Massive-STEPS, engine `check2hgi` set-A, stride-1 base; board row in RESULTS_BOARD §1) uses a separate
chain: `scripts/run_istanbul_champion_stride1.sh` (+ `scripts/second_dataset/`, `scripts/build_istanbul_stride1.py`).
**Do NOT** use the Gowalla `check2hgi_dk_ovl` overlap builder for Istanbul.

### Checkpoint eval
`scripts/evaluate.py --checkpoint <results/.../model.pt>` evaluates a saved checkpoint (separate from the
per-fold scoring above).

---

## Main execution params (champion-G)

| Param | Value | Notes |
|---|---|---|
| `--model` | `mtlnet_crossattn_dualtower` | cross-attention dual-tower |
| `--cat-head` / `--reg-head` | `next_gru` / `next_stan_flow_dualtower` | reg head **prior-OFF**: `freeze_alpha=True alpha_init=0.0` |
| `--mtl-loss` / `--category-weight` | `static_weight` / `0.75` | NOT nash_mtl (cvxpy errors) |
| `--scheduler` / `--max-lr` | `onecycle` / `3e-3` | per-head `--cat-lr/--reg-lr/--shared-lr` are **INERT under onecycle** unless `MTL_ONECYCLE_PER_HEAD_LR=1` |
| `--batch-size` | `2048` (canon) / `8192` (candidate) | never drop below 2048 (diverges) |
| `--epochs` / `--folds` | `50` / `5` | |
| `--checkpoint-selector` | `geom_simple` | `√(cat_F1·reg_Acc@10)`; pass `joint_f1_mean` for v11 |
| `--task-a/-b-input-type` | `checkin` / `region` | reg MUST be `region` (checkin drops reg ~50→28%) |
| `--canon none` | required | else the wrong-substrate guard hard-fails under MTL_STRICT on dk_ovl |
| precision | **fp32 for ALL states** (`MTL_DISABLE_AMP=1`) | **bf16 DROPPED board-wide for QUALITY** (~1pp cost at large C: FL bf16 78.90/76.04 vs fp32 79.82/77.28). AL/AZ bf16 is NaN-safe but gains nothing → fp32 is the settled choice everywhere, NOT just a big-state NaN dodge. (Large-C bf16 also grad-NaNs on A40-Ampere; fp16 overflows.) Sole board exception: CA §1 headline is an H100 bf16 cell. |
| seeds | `{0,1,7,100}` | NOT 42 (dev seed, overshoots §0.1 +3pp CA/+8pp TX) |

---

## Operational miscellaneous

### Run profiler / audit (`--profile` or `MTL_PROFILE=1`)
Ephemeral per-fold section timing (data/forward/backward/eval), throughput (batch/s, samp/s), peak GPU mem, util
(pynvml), torch.compile recompile/graph-break counts, and **pain-point flags** (GPU-starved, sync/data-bound,
recompile). `MTL_PROFILE_JSON=<path>` dumps a transient report. Default-off → bare runs byte-identical. Source:
`src/training/profiling.py`. Use it to find bottlenecks (it flagged `GPU-STARVED util 34%` + `GRAPH BREAKS: 10`).

### Multi-fold fan-out (5 folds as separate processes → one rundir)
```bash
scripts/run_folds_fanout.sh <run_id> <folds_csv> <max_parallel> -- <train.py recipe…>
scripts/aggregate_folds.py <rundir>     # → fold_aggregate.json (reads per-fold artifacts by REAL fold id)
```
Flags: `--only-folds 2,3` (subset), `--run-id NAME` (shared rundir leaf; implies `--per-fold-seed`),
`--per-fold-seed` (reseed `seed+fold_id` → fold-k order-independent, **proven byte-identical under 5-way
concurrency**). ⚠ In a fan-out, per-process `summary/full_summary.json` is unreliable — read `fold_aggregate.json`
or score via `a40_score_matched.py` (globs `fold*_*` by real id). Concurrent rundir naming omits the seed → score
by the **PID suffix** of the rundir, not `ls -dt | head` (mis-maps).

### Key env vars
| env | default | effect |
|---|---|---|
| `MTL_STRICT=1` | off | hard-fail the preflight/provenance/wrong-substrate guards (board sets it) |
| `MTL_DISABLE_AMP=1` | off | force fp32. **auto-fp32** also forces it for reg **C>2000** (FL/CA/TX) on Ampere — but NOT AL/AZ (C<2000), so set it explicitly for board runs |
| `MTL_SKIP_INERT_LOGT=1` | **on** | skip the per-fold log_T load + leak-guards when the prior is inert (champion) — byte-identical |
| `MTL_ONECYCLE_PER_HEAD_LR=1` | off | make `--cat-lr/--reg-lr/--shared-lr` actually apply under onecycle (per-group max_lr) |
| `MTL_COMPILE_DYNAMIC=1` | off | dynamic-shape compile (board path) |
| `MTL_CHUNK_VAL_METRIC=1` | off | chunked val-metric (large-C memory) |
| `MTL_DATASET_GPU=1` | off (**auto-fit**) | Force the per-fold dataset GPU-resident, bypassing the fit-check (`folds._dataset_device`). DEFAULT = auto-fit: pre-move to GPU only if it fits in (free VRAM − headroom), else keep CPU-resident with per-batch `.to()` (byte-identical). **Small states ONLY** (minor speed lever). ⚠ **NEVER for CA/TX/FL** — forces ~31 GB redundant per-fold copies → OOM (no CPU fallback). |
| `MTL_DATASET_CPU=1` | off | Force the dataset CPU-resident (opposite); safe escape hatch. |
| `MTL_GPU_HEADROOM_GB` | 16 | VRAM reserved for model+activations in the auto-fit check. |
| `MTL_RAM_HEADROOM_GB` | 16 | **host-RAM** reserved by the next-input build guard (CPU-side; distinct from the VRAM twin above). Big states (CA/TX/FL) tune it to avoid a build hard-fail. |
| `MTL_COMPILE_CACHE_LIMIT` | 64 | dynamo `cache_size`/`recompile` limit. The default 8 is too low for the TRAIN+EVAL+shape graph variants → **silent eager fallback** (this is the original "minutes-long fold", not a warmup). Pure safety, no numeric change. |
| `MTL_COMPILE_MODE` | unset | passes inductor `mode=` to `torch.compile` (e.g. `max-autotune`). |
| `MTL_STREAM_TRAIN_METRIC` | **on (1)** | streaming train-metric for reg C>256 (O(N·C)→O(N+C) mem); set `0` for the full-logit path (byte-identical). |
| `MTL_CHUNK_VAL_METRIC` + `MTL_S2_AUTO_BUDGET_GB` | off / 4 | chunked val-metric for large-C; the budget (GB) is the VRAM threshold above which it auto-chunks. |
| `MTL_DISABLE_AMP_EVAL=1` | off | force fp32 at **eval/val only** (train precision unchanged) — used by bf16-train cells to keep eval fp32-clean. |
| `MTL_AUTOCAST_BF16=1` | off | use **bf16** (not fp16) for autocast — the precision behind the CA §1 H100 bf16 headline cell. An explicit value wins over auto-fp32. |
| `MTL_NAN_GUARD` / `MTL_DIVERGENCE` | on / — | skip the optimizer step on non-finite grad/loss; under `MTL_STRICT=1` turns into a fail-loud abort (prevents a NaN-poisoned shared backbone at large C). |
| `MTL_STAN_LEGACY_MASK=1` | off | restores the pre-P1 guarded `.any()` STAN mask path (graph breaks 2→10, SLOWER). ⚠ Does **NOT** give bit-exact `--compile` repro — Phase-5c proved fresh-cache mask on==off (the compiled number is governed by the inductor cache/compile session, drift ≤0.3 pp within fold-std); eager is bit-exact either way (the ground truth). Leave OFF. |
| `MTL_STAN_FP32_ATTN=1` | off | fp32 masked-softmax attention under autocast — an **UNVALIDATED** candidate A40-bf16 grad-NaN mitigation; no-op under fp32. |
| `MTL_NO_TRAIN_DIAGNOSTICS=1` | off | P4 (pipeline_audit 2026-07-01): skip the batch-0 grad-cosine diagnostic (2 extra backwards/epoch + host syncs) and, for `static_weight` runs under `--compile`, keep inductor donated buffers enabled. ~9% wall at AL warm-cache; **byte-identical in eager** (parity-verified). `grad_cosine_shared` logs NaN. Use for sweeps that don't read the diagnostic. |
| `DATA_ROOT` | `data/` | relocates the data root (`src/configs/paths.py`); set on machines where check-ins aren't at the default path. |
| `TORCHINDUCTOR_CACHE_DIR` | — | per-run inductor cache (avoid cross-run compile-cache variance) |

### A40 constraints
- Large-state MTL **fits VRAM** post-2026-06-19 OOM fix (CA ~11 GB / TX ~13 GB peak); never drop batch < 2048.
- **Champion precision is fp32 across ALL states** (bf16 dropped board-wide for ~1pp quality, not merely to dodge
  the large-C grad-NaN). Big-state datasets stay CPU-resident via the `folds._dataset_device` auto-fit — never
  `MTL_DATASET_GPU=1` for CA/TX/FL (~31 GB redundant copies → OOM). The full multi-seed overlap board is H100-only (FL fp32
  ~24 min/epoch); run small states / single cells on the A40.
- Disowned (`setsid`/`nohup`) runs don't fire harness notifications → **poll actively** (read JSONs + `ps`).

---

## Other canonical flows + gotchas (pointers)

- **Version → engine map (don't-delete list).** v11 = paper §0.1 (GCN base, `check2hgi`); v14 =
  `check2hgi_design_k_resln_mae_l0_1` (substrate); **board champion engine = `check2hgi_dk_ovl`** (overlap of v14).
  `dk_ovl` **symlinks v14 embeddings** but its windowed parquets need the **v11 fold/sequence/graph backbone** —
  so **do not delete `output/check2hgi/<state>/` either** (not just v14). Full map: `closing_data/SUBSTRATE_VERSION_MAP.md`.
- **⚠ Stale log_T (C22).** `regen`/postbuild do **NOT** refresh the per-fold log_T; a stale file silently inflates
  reg ~+8 pp (STL) / ~+12 pp (MTL-disjoint). If you build log_T (step A.5) or rank STL reg: `stat` the `.pt` vs
  `next_region.parquet` mtime; if older, rerun `compute_region_transition.py`. (Moot for the prior-OFF champion — inert.)
- **v11 paper-canon repro (B9 / H3-alt).** champion-G/v16 is uniform across states, but **paper §0.1 is v11**, which
  is **state-conditional**: B9 (FL/CA/TX) vs **H3-alt (AL/AZ)** drops `--alternating-optimizer-step` /
  `--alpha-no-weight-decay` / `--min-best-epoch 5` and uses `--scheduler constant`, engine `check2hgi` (GCN) +
  `--log-t-kd-weight 0.0`. Details: `CANONICAL_VERSIONS.md §v11` / `NORTH_STAR.md`. (Note: on the H3-alt
  constant/cosine path the per-head LRs DO apply — unlike the onecycle champion.)
- **(E) STL literature baselines** (RESULTS_BOARD §1b/§4 denominators): CSLSL/CatDM cascade (`scripts/baselines/b4_cascade.py`),
  CTLE (`ctle_e2e.py`), faithful STAN, HMT-GRN, POI-RGNN, Markov-1. Run per `docs/baselines/README.md`; same seed-0×5f / `dk_ovl` / fp32 protocol.
- **Why the silently-droppable flags are non-negotiable:** `--no-{reg,cat}-class-weights` (**C25** — weighted CE
  depresses reg ~10–14 pp / cat ~3–5 pp); `geom_simple` selector (**C21** — the old `joint_f1_mean` lost ~11 pp reg
  at FL; pass `joint_f1_mean` only to reproduce v11).
- **`MTL_STRICT=1` hard-fails THREE preflights** (not just provenance): dev-seed-42 (paper needs {0,1,7,100}),
  champion-on-wrong-substrate, and torch ≠ 2.11.0+cu128 — plus the overlap-provenance gate (B).
- **Score → board:** `a40_score_matched.py` gives the per-run cat/reg; board §1 cells are the **fp32-matched
  rescore** (`r0_matched_rescore.py` → `docs/results/closing_data/*_matched_score.json`, paths in RESULTS_BOARD §3).

## Changelog
- **2026-06-30** — Created + closeout audit pass (added the env-var rows, log_T-inert framing, bf16-dropped-board-wide,
  MTL_DATASET_GPU, version map / C22 / v11-repro / baselines / C25-C21 pointers). Canon = champion-G/v16 on
  `check2hgi_dk_ovl` (v14 substrate), bs=2048 fp32. Promoted candidate `--canon v17` (opt-in; AL/AZ/FL n=20; CA/TX pending). Documented
  the per-head-LR candidate (bs=8192 + cat-lr 1e-3 via `MTL_ONECYCLE_PER_HEAD_LR`, n=20-confirmed, pending promotion).
- **2026-07-01** — **v17 PROMOTED to `DEFAULT_CANON`** (`src/configs/canon.py`; bs=8192 + `--onecycle-per-head-lr`). Bare `train.py --task mtl` now runs v17; v16 via `--canon v16`. AL/AZ/FL n=20 done, CA/TX n=20 running.
