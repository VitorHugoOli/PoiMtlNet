# train_perf_multifold — PLAN

> **Study branch:** `study/train-perf-multifold` (off `main`; PR, never merge to main).
> **Machine:** a40-wk (single A40 46 GB, fp32 board protocol).
> **Source of truth for numbers:** `docs/studies/closing_data/RESULTS_BOARD.md` (the AL targets) + this study's
> `baseline_runs/` (the same-box reference produced here). This file is the design; `log.md` is the chronology.

## 0 · Goals (user, 2026-06-26)

1. **Reproduce + validate.** Run champion-G MTL and the STL ceilings for **Alabama** on this A40, confirm the
   numbers are "very similar" to `RESULTS_BOARD.md`. (Also: bf16-on-A40 is broken — investigate.)
2. **Audit the train pipeline for speed (quality-neutral).** inputs → dataloaders → model → hparams → train →
   eval → metrics. Find where we can run faster **without changing any scored number**.
3. **Multi-fold fan-out.** Make it feasible to run the 5 folds of one execution as separate processes that all
   write into **one** shared execution rundir (so aggregation is clear), plus a `--only-folds 2,3` CLI to run a
   subset/single fold. Validate the approach is the right one.
4. **Opportunistic cleanup** while in the code: cut stale comments, remove duplication, apply best practices.

**Ordering (user):** verify AL first → audit → optimize code → multi-fold → cleanup throughout.

**Hard invariant:** every change is gated by an AL champion-G + STL A/B that must stay within fold-std of the
board (and the byte-identical changes must stay byte-identical). The frozen §0.1 reproducibility contract holds.

---

## 1 · AL reproduction runbook (Goal 1) — VALIDATED RECIPE

Driver: [`run_al_baseline.sh`](run_al_baseline.sh) (mirrors `closing_data/board_h100_mtl.sh` fp32 + `board_h100_ceiling.sh`).
Env (A40 board protocol): `PYTHONPATH=src`, `MTL_DISABLE_AMP=1` (fp32; Ampere bf16 grad-NaN), `MTL_STRICT=1`,
`MTL_CHUNK_VAL_METRIC=1`, `MTL_COMPILE_DYNAMIC=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

| cell | script | per-fold-transition-dir | A40 target (§1b same-device) | board §1 (H100) |
|---|---|---|---|---|
| champion-G MTL | `train.py --task mtl --canon none …` | **v14** `output/check2hgi_design_k_resln_mae_l0_1/alabama` | cat **63.25** / reg **69.65** | 63.56 / 69.81 |
| STL cat ceiling | `train.py --task next --model next_gru …` | — | **55.87** | 55.87 |
| STL reg ceiling | `p1_region_head_ablation.py next_stan_flow a0` | v14 | **69.99** (on disk) | 69.99 |

Score: champion-G → `closing_data/a40_score_matched.py` (per-task diagnostic-best, fold-mean; reg =
top10·(1−ood) at indist-best ep, cat = macro-F1 at f1-best ep). cat ceiling → `score_stl_cat_ceiling.py`.
**Acceptance:** champion-G beats the cat ceiling (+~7.4 pp) and matches the reg ceiling (−~0.3 pp, "within δ=2");
0 non-finite skips; late best-epochs. **Substrate + inputs + fresh seed-0 v14 & overlap log_T already on disk.**

---

## 2 · Pipeline map (for the work below) — key seams

- **Dispatch:** `train.py:2207-2212` → champion-G is `_run_mtl_check2hgi` (`:178-249`) → `mtl_cv.train_with_cross_validation`.
- **Two rundirs:** checkpoint dir `_make_run_dir` (`train.py:72`, skipped under `--no-checkpoints`); **metrics
  rundir** name from `storage._folder_name()` (`storage.py:467`) = `<model>_lr…_bs…_ep…_<start_date>` where
  `start_date = strftime + "_" + getpid()` (`experiment.py:239`).
- **Fold loop (MTL):** `mtl_cv.py:1335` `for fold_idx,(i_fold,dataloader) in enumerate(dataloaders.items())`.
  `i_fold` = real canonical fold id (dict key); `fold_idx` = enumerate position. Folds built lazily by
  `_LazyFoldMapping` (`folds.py:124`, no cache — one resident at a time, the CA/TX OOM fix).
- **Per-fold artifacts** written by `storage.py` keyed on **in-memory position** `curr_i_fold+1`
  (`save_fold_partial` `:338`, `_save_metrics` `:502`), NOT `i_fold`. The per-fold **log_T** correctly uses
  `i_fold+1` (`mtl_cv.py:1362`). ← the position/id divergence is blocker B2.
- **Aggregation** = `SummaryGenerator` over **in-memory** `history.folds` only (`storage.py:162-242`) →
  `summary/full_summary.json`. Nothing reads per-fold CSV/JSON back from disk. ← blocker for fan-out re-agg.
- **Seeding:** `seed_everything(config.seed)` once (`train.py:1973`); **no per-fold reseed anywhere** → fold-k run
  alone ≠ fold-k in a sweep (the global RNG advances across folds; loaders shuffle from it, `folds.py:277-283`).
- **CLI fold control:** `--folds N` truncates to first N **and reshapes the split** to `max(2,N)` (`train.py:1969`);
  `--only-fold k` (singular, exists) forces 5-split + `{k: fold_results[k]}` (`train.py:2143-2154`).

---

## 3 · Performance audit (Goal 2) — prioritized, quality-neutral

All items below are **byte-identical** (verified against code by two independent audit lanes) unless marked.
Gate each with an AL A/B (champion-G + reg ceiling): the scored numbers must not move.

### Tier 1 — high value, low effort, byte-identical (DO FIRST)
- **P1. Remove data-dependent `.any()` guards in the STAN reg head** — `next_stan/head.py:252-255` (×2 towers),
  `next_stan_flow_dualtower/head.py:428`, `next_stan_flow/head.py:143`. Each is a host sync **and** a
  `torch.compile` graph break on every reg forward. `masked_fill` with an all-False mask is the identity → apply
  unconditionally; replace the all-padded boolean-index fixup with a vectorized
  `padding_mask[:, -1] &= ~all_padded` (on a clone). **Byte-identical**; removes ~3 syncs + ~3 graph breaks/forward.
  Highest value on the compiled board path.
- **P2. Defer train-metric D2H to epoch-end** — `mtl_cv.py:888-889` (cat) + `:902-908` (reg S1). Today each batch
  does ~7 blocking `.cpu()` syncs that starve the GPU (util 8-25%). Accumulate the small tensors on-GPU, one
  `torch.cat(...).cpu()` at epoch end — exactly what the **val** path already does (`mtl_eval.py:215,262-265`).
  These are **diagnostic train metrics only** (not selection/early-stop/checkpoint). Byte-identical.
- **P3. Cache the per-epoch OOD train-label set** — `mtl_cv.py:1086-1089` rebuilds `set(y.unique().tolist())`
  every epoch (×50); train labels are fixed per fold. Cache on `fold_history` (same pattern as
  `_joint_lift_majority` `:1139`). Byte-identical; MED win at CA/TX.

### Tier 2 — gate diagnostics behind a flag (default off for sweeps; byte-identical to "diagnostics on" math)
- **P4.** `--no-train-diagnostics` (default off) to skip: encoder-trajectory + gate-entropy scalar reads
  (`mtl_cv.py:979-1074`), the per-epoch grad-cosine extra backward (`:770-775` → `_compute_gradient_cosine`),
  and FLOPs/fvcore profiling (`:1814`). All are reporting-only; gating removes per-epoch syncs + 2 partial
  backwards/epoch. (fvcore is not installed here, so FLOPs already no-ops — but the flag makes it explicit.)
- **P5.** `evaluate_model` `@torch.no_grad()` → `@torch.inference_mode()` (`mtl_eval.py:71`) + the train-metric
  `no_grad` block. Free; validate no inference-mode interop error in the metric reductions.

### Tier 3 — larger, still quality-neutral (needs careful A/B)
- **P6. Pin CPU-resident batches** — `folds.py:455,499,569` gate `pin_memory` on `num_workers>0` (always 0), so
  it is dead. For CPU-resident states (CA/TX, `_dataset_device→None`) the per-batch `.to(DEVICE, non_blocking=True)`
  silently runs **blocking** on pageable memory. Set `pin_memory = cuda and dataset_device is None`. Byte-identical;
  ~2× H2D bandwidth at CA/TX. **Must** gate on `dataset_device is None` (pinning an already-CUDA batch errors).
- **P7. Compile once across folds** — `create_model`+`torch.compile` are inside the fold loop (`mtl_cv.py:1560-1590`)
  so Dynamo re-traces a fresh module ×5. Reuse one compiled module, `load_state_dict` per fold + in-place copy the
  per-fold log_T buffer (don't replace the module). HIGH throughput lever for 5-fold runs, but the per-fold reg-head
  rebuild for log_T makes this delicate → **requires a byte-identical A/B**. Implement last; behind a flag if risky.

### Explicitly EXCLUDED (quality-risking — do NOT apply)
fused/foreach AdamW (accum order), `F.scaled_dot_product_attention` in `_STANAttention` (reduction order at the
±0.1 pp reg margin), bf16-autocast-by-default, removing `MTL_DISABLE_AMP`, `pack_padded_sequence` for the GRU.
`num_workers>0` is **also excluded** (no seeded generator → consumes global RNG → moves the frozen baseline; measured AL drift cat +0.92/reg +0.23).

### bf16/A40 backward-NaN (Goal 1 stretch)
No proven byte-identical fix exists; **fp32 (`MTL_DISABLE_AMP=1`) is the validated answer** and the board already
uses it for A40 large states. Two actionable items:
- **B-fix (safe, do):** launcher hardening — `p3_board.sh run_cell` sets no AMP env → falls through to the
  forbidden fp16 default on the A40. Add an Ampere guard that routes `MTL_DISABLE_AMP=1` for large states (a
  process/launcher fix, zero numerics change).
- **B-probe (A/B only, optional):** fp32 attention island in `_STANAttention` (wrap `q@k…softmax…attn@v` in
  `autocast(enabled=False)` after `.float()`). Strictly more precise (subset of the proven fp32 run), bf16-mode-only,
  but NOT byte-identical and may not fully fix it (the NaN may live in the cross-attn MHA matmuls too). Probe, don't adopt.

---

## 4 · Multi-fold fan-out (Goal 3) — design (validated as the right approach)

**Problem today:** running fold-k in its own process can't share a rundir (B1), collides on artifact names (B2),
and isn't reproducible vs a sweep (B3). Aggregation only happens in-memory in the one process that ran all folds.

**Design — opt-in "fan-out mode", default path untouched (board-safe):**

1. **`--run-id NAME` (new).** Sets the metrics rundir leaf to a fixed `NAME` (override `MLHistory.start_date` /
   `_folder_name`) so N processes pointed at the same `--run-id` write into ONE
   `results/<engine>/<state>/<NAME>/`. Fixes **B1**. The example dir
   `mtlnet_lr1.0e-04_bs2048_ep50_<ts>_<pid>` is just a `--run-id` you can also auto-generate once and reuse.
2. **Real-fold-id artifact naming.** Thread the real fold ids (`list(dataloaders.keys())`) into MLHistory; storage
   names per-fold files by `fold_ids[pos]+1` instead of `curr_i_fold+1` (fallback to old behavior when unset → a
   normal full run is byte-identical, since `fold_ids[pos]==pos`). Fixes **B2**: a subset/fan-out writes
   `fold{real_id+1}_*` with no collisions.
3. **`--only-folds LIST` (new, e.g. `2,3` or `0`).** Forces the canonical 5-split (like `--only-fold`), then
   `fold_results = {k: fold_results[k] for k in LIST}`. Mirrors the existing `--only-fold` seam (`train.py:2143`).
   Leaves the singular `--only-fold` untouched (board scripts depend on it).
4. **`--per-fold-seed` (new, opt-in; auto-on under `--run-id`).** Reseed `seed_everything(seed + i_fold)` + give the
   train loaders an explicit per-fold `generator` **before each fold is materialized**, so fold-k is a pure
   function of `(seed, fold_id, data, config)` — identical whether run alone, in a subset, or in a full sweep.
   Fixes **B3**. Default OFF → the bare board path is byte-identical to today.
5. **`scripts/aggregate_folds.py <rundir>` (new).** Reads the per-fold on-disk artifacts (`folds/fold{k}_info.json`
   + `metrics/fold{k}_*_val.csv`) present in the shared rundir and writes a unified `summary/full_summary.json`
   (replicating the `_stats` mean/std over the 3 bases) + a small `fold_manifest.json`. Makes the fan-out rundir
   look like a normal complete run. The existing scorers (`a40_score_matched.py`, `score_stl_cat_ceiling.py`)
   already read per-fold by glob → they work on the shared rundir directly once B1+B2 land.
6. **Orchestrator `run_folds_fanout.sh` / `.py` (new).** Given a recipe + `--run-id`, launch the folds (each
   `--only-folds k --run-id NAME --per-fold-seed`) with a `--max-parallel` knob (serial, or P-at-a-time — AL is
   data-bound at 8-25% util so parallel folds reclaim idle GPU; small states fit 5×, CA/TX 2-3×), then call
   `aggregate_folds.py`.

**Why this is the right approach (vs alternatives):** writing into one rundir + a disk re-aggregator keeps the
existing artifact layout, scorers, and record-comparison flow intact (no parallel result schema); per-fold seeding
makes the fan-out a *drop-in equal* of a sequential run rather than a different RNG draw; opt-in gating preserves
the frozen board. Rejected alternatives: (a) a separate parallel results schema (would fork the scorer/record
tooling); (b) in-GPU batched multi-fold (one process, all folds on device at once — blows VRAM at CA/TX and gives
no isolation/restart granularity); (c) leaving B3 unfixed (fan-out would not reproduce a sweep — unacceptable for a board).

**Validation gate (on AL, cheap):**
- `train.py --folds 5 --run-id seqA --per-fold-seed` (one process) → aggregate A.
- 5× `train.py --only-folds k --run-id fanA --per-fold-seed` → `aggregate_folds.py` → aggregate B.
- **Assert A == B byte-for-byte** (per-fold) and **A within fold-std of the board** (per-fold seeding changes the
  RNG realization, not expected performance). Also assert bare `--folds 5` (no flags) == today's baseline_runs.

---

## 5 · Cleanup (Goal 4) — behavior-preserving

Hot files are clean of commented-out code. Targets, SAFE→RISKY:
- **SAFE (do):** dedup the literal 21-member `_ALLOWED_ENGINES_FOR_C2HGI_PRESET` (train.py:2021) ≡
  `_MTL_C2HGI_ALLOWED_ENGINES` (folds.py:1219) → one constant in `configs/paths.py`; fix the wrong "orphan"
  comment in `dataset.py:22`; drop the stale `mtl_cv.py:241-247` line refs in `train.py:106`; rename
  `mtl_creterion`→`mtl_criterion` (`mtl_eval.py:76`); delete the redundant re-import `mtl_eval.py:64`; trim the
  ~45 dated "fixed 2026-.." / audit-codename narration comments to a one-line rationale (keep load-bearing
  invariants like the env-var contracts). Align the `category_trainer`/`next_trainer` signatures + drop dead defaults.
- **MED (do with one-fold byte-identity A/B):** unify the fp16/fp32 autocast hatch (4 files) into
  `helpers.build_autocast_ctx`; collapse `_run_category`/`_run_next` and `category_cv`/`next_cv` `run_cv` into one
  parametrized helper; factor the 3 DataLoader builders' shared kwargs.
- **DEFER (RISKY, only if time + strong A/B):** decompose `train_model` (~1100 lines) /
  `train_with_cross_validation` / `_parse_args`. Touch only with a multi-seed metric-parity gate.

---

## 6 · Sequence + status (CLOSED 2026-06-26)

1. ✅ Map + AL runbook + study scaffold.
2. ✅ AL baseline (`baseline_runs/`) — reproduces the board: MTL 63.18/69.73, STL 55.73/69.98 (all ≤0.14 of board).
3. ✅ Tier-1 perf P1 (STAN `.any()` → graph breaks 10→2, eager byte-identical + unit test) + P3 (OOD cache) +
   P6 (pin CPU-resident). AL A/B: 63.44/69.82 within fold-std (mean ↑); STL cat 55.73 bit-identical.
4. ✅ Multi-fold (`--only-folds`/`--run-id`/`--per-fold-seed`, `MLHistory.run_id`+`fold_label`,
   `aggregate_folds.py`, `run_folds_fanout.sh`). **5-way concurrent fan-out: 0 conflicts, all folds present by
   real id; fold-1 solo == fold-1 concurrent BYTE-IDENTICAL.** Capstone: full 50-ep fan-out (`capstone_fanout50`).
5. ➖ Tier-2/3 perf: P2a/P5 dropped (profiler shows non-bottlenecks on sensitive paths); P7 deferred (only helps
   sequential — the fan-out amortizes compile/process). Documented as future work.
6. ✅ Run profiler/audit tool (`src/training/profiling.py`, `--profile`) — user-requested; flags bottlenecks +
   speed + quality, ephemeral. SAFE cleanup: engine-allow-list dedup (paths.py SoT), mtl_eval import/typo,
   dataset.py comment. (MED refactors — unify runners/autocast hatch — left as future cleanup.)
7. ✅ Tests: 905 pass in touched modules (+ new mask test); 3 pre-existing embedding-test failures unrelated
   (fail on clean HEAD too). CLAUDE.md documents the tools. Commit + PR + notify.

### Future work (documented, not done)
- **P7 compile-once-across-folds** (biggest sequential-run lever; delicate per-fold log_T buffer swap → needs A/B).
- **bf16/A40 B-probe** (fp32 attention island in `_STANAttention`; A/B-only) + launcher auto-fp32 for large states.
- **MED cleanup**: unify `_run_category`/`_run_next` + `category_cv`/`next_cv` runners; one `build_autocast_ctx`.
- **Single-task fan-out** (`--run-id`/`--per-fold-seed` for `next_cv`/`category_cv`) — infra is generic; wire it.
- **n_regions pass-through** so a fan-out process doesn't transiently build all 5 folds for the region-count scan.
