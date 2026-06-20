# The MTL OOM problem and its fix (FL/CA/TX region-MTL) — context for any agent

> Created 2026-06-20 (`study/pre-freeze-a40`). **Read this if you see "MTL training OOMs / runs out of memory"
> or believe "CA/TX large states need a bigger GPU."** That was true with the OLD code; it is FIXED now. This
> doc is the single explanation of the problem, the root cause, the fix, and the validation — because the fix
> is otherwise scattered across commit messages (`4d2b4868`, `f272cecf`, `b5332b2e`, `8ff36dba`, `33fe18da`)
> and a stale prior belief ("large states → bigger GPU").

## The problem (what people saw)
1. **Box RAM exhaustion (~123 GB)** during MTL input build / training — killed the machine / tmux.
   **This actually happened (2026-06-20): a CA stride-1 overlap build exhausted 125 GB RAM + 8 GB swap and
   killed the tmux session.** Root cause = the `<U32` upcast × O(N) row accumulation (~440 GB at CA stride-1).
   ⚠ **Regression history:** `33fe18da` first fixed the `<U32` blowup (per-user float32 conversion); a later
   threading merge (`dade24ad`) **reverted it** back to `all_results.extend(<U32 rows>)`. It was re-fixed for
   good in `eb45c744` with the **streaming** writer (float32-immediate + O(chunk) parquet row-groups), which
   also removes the O(N) accumulation the original float32-only fix still had. **Never run a large-state
   (CA/TX/FL) overlap build with an uncapped, non-streaming builder.**
2. **A40 GPU OOM** for the **region task** (`next_region`) MTL — FL *overlap* MTL OOM'd outright; **CA/TX
   (large states, 8501/6553 regions) were believed to OOM the A40 at bs2048** (peaks ~39 GB, spikes >44 GB).
   The prior conclusion was "CA/TX region-MTL is not feasible on the A40 → use a bigger GPU / defer."

## Root cause — it was the METRIC, not the model
The dominant memory consumer was **NOT** the model or the dataset. It was the **per-epoch metric computation
accumulating the FULL logits**: O(N × C) where N = samples and **C = region count** (1109 AL … **8501 CA**).
For the region task with thousands of classes, the accumulated logit tensor is enormous. Plus a separate
input-build bug: the `<U32` 32× memory blowup (float embeddings concatenated with a string category column
upcast the whole row to 128-byte `<U32`).

## The fix (4 changes — all byte-identical; the numbers do NOT change)
| change | file | what it does |
|---|---|---|
| **S1** streaming train-metric | `src/training/runners/mtl_cv.py` | accumulate per-batch preds/targets/rank/hit on **CPU** (O(N+C)) instead of full logits on GPU (O(N·C)). **This is the actual OOM fix.** Default-ON, gated `C>256`, byte-identical. |
| **S2** chunked val-metric | `src/training/runners/mtl_eval.py` | the scored val-metric path streams in chunks (k∈{1,3,5,10}) instead of materializing full val logits. Default-OFF (`MTL_CHUNK_VAL_METRIC=1`), byte-identical. |
| **dataset-on-GPU auto-fit** | `src/data/folds.py:_dataset_device` | pre-move the dataset to GPU **if it fits** (free VRAM − 16 GB headroom, `MTL_GPU_HEADROOM_GB`), else keep it **CPU-resident** (per-batch transfer). Byte-identical (device never changes the math). Overrides: `MTL_DATASET_CPU=1` / `MTL_DATASET_GPU=1`. |
| **`<U32` builder fix (streaming)** | `src/data/inputs/builders.py` + `src/data/inputs/core.py` | convert each user's batch to float32 immediately **and stream rows to parquet in O(chunk) row-groups** (`NextInputStreamWriter`/`_SequenceStreamWriter`) → kills the 32× string-upcast RAM blowup *and* the O(N) accumulation. Byte-identical (alabama: uint32-view `array_equal`, maxdiff 0.0). |

## Validation (it works)
- **FL overlap MTL** (the case that OOM'd): now **completes** (cat 76.65 / reg 74.16).
- **AL MTL byte-identity**: bare `train.py --task mtl --state alabama --seed 0` reproduces the champion
  **52.3781 / 64.3450 exactly** (so the fixes are numerically inert).
- **CA/TX MTL on the A40 (2026-06-20)**: **FIT** — champion-G (v16, bs2048, non-overlap, fold-0):
  **CA peak 10.9 GB, TX peak 13.0 GB** (/ 46 GB) — ~3× under the old ~39 GB peak. The GPU consumer is now just
  the model+activations+per-batch (~11–13 GB); the dataset auto-fits.

## A THIRD, distinct OOM — host-RAM in CV fold construction (overlap CA/TX) — FIXED 2026-06-20
The two fixes above are about **GPU VRAM** (the metric) and the **`<U32` build**. A separate
**host-RAM** OOM appears only at **stride-1 overlap on the large states** (CA 2.93M / TX 3.83M rows)
and killed the whole box (tmux + session) twice. Audited by workflow `wjrt9xqs7` (6 agents + adversarial
+ empirical). Root cause was NOT the GPU (watchdog caught GPU at 3 MiB while host RSS hit 61 GB and
climbing): `folds._create_check2hgi_mtl_folds` built **all `n_splits` folds eagerly** into a dict and
never freed them, and each fold stored a **second** full fancy-index copy of the `[N,9,D]` slices in
`FoldData.x` (which the MTL runner never reads). At CA that is ~126 GB → OOM-kill.

**Fix (commit `8fa32344`, byte-identical):**
- **Lazy per-fold construction** (`_LazyFoldMapping`): the runner consumes folds one-at-a-time, so only
  ONE fold is resident (not all 5). 
- **Drop `FoldData.x`** on the MTL path (`None`; verified never read in training).
- **S2 auto-enable** (`mtl_eval.py`): when the full val reg-logit `[N_val,C]` would exceed a GPU budget
  (`MTL_S2_AUTO_BUDGET_GB`, default 4 GB), auto-switch to the byte-identical chunked val metric instead of
  OOMing the GPU. Non-overlap (small `N_val`) keeps the full path → frozen §0.1 untouched.
- **`_guard_mtl_check2hgi_ram`** fail-loud host-RAM guard + the RSS-watchdog harness
  (`scripts/pre_freeze_gates/overlap_fit_safe.sh`) as the safety net.

**Verified (watchdog, auto dataset mode, 2026-06-20):** CA overlap MTL **GPU 26 GB / host 49 GB**; TX
(3.83M rows) **GPU 27 GB / host 59 GB** — **both FIT** the A40 (46 GB) and the 125 GB box (was >126 GB,
box-killing, pre-fix). Byte-identity: bare AL champion-G seed 0 reproduces **52.3781 / 64.3450 exactly**.
⚠ **`MTL_DATASET_GPU=1` does NOT help this OOM** (empirically tested) — the host peak is in CPU fold
construction, *before* any `.to(device)` move. Full writeup: `WINDOWING_AUDIT.md` cross-ref + the workflow
result.

## Q1 CLOSED (2026-06-20) — CA/TX overlap MTL: use AUTO dataset-fit, NOT `MTL_DATASET_GPU=1`
**CA/TX stride-1 overlap champion-G MTL is VIABLE on the A40** (verified: TX fold-1 trains, ~160 s/epoch
on a 3.06M-row train fold; CA similar). **But the dataset-device mode matters:**
- ✅ **AUTO (the default, `folds._dataset_device`)** — for large states the per-fold dataset is ~31 GB of
  redundant copies (next.train + cat.train + the joint-train loader's 3rd copy + val tensors, ×2 towers),
  which does NOT fit on the GPU with head-room, so AUTO **keeps it CPU-resident** (TX runs at **GPU ~6 GB**,
  per-batch host→device transfer). AUTO **never OOMs** — it falls back to CPU. **This is the correct mode.**
- ❌ **`MTL_DATASET_GPU=1` (force)** — forces all those copies onto the GPU → **OOM on TX** (no CPU fallback;
  ~31 GB data + model/activations > 44 GB). Only use the force flag for SMALL states whose dataset genuinely
  fits. **Do NOT set `MTL_DATASET_GPU=1` for CA/TX.**
- AUTO's GPU peak is **occupancy-dependent** (it moves what fits at decision time) — TX has been observed at
  ~6 GB (dataset CPU) and, when more VRAM was free, higher; either way it fits because it never forces.
- **Speed:** TX overlap MTL ≈ **160 s/epoch** uncompiled (CPU-resident dataset). A full TX MTL (5f×50ep) is
  therefore ~11 h — the cost is intrinsic (8.5× overlap sequences + per-batch transfer), not a misconfig.
  `--compile`/`--tf32` give ~15% (FL-measured, `SPEED_LEVERS.md`); they are board-execution knobs, opt-in.

> TL;DR for agents: **CA/TX overlap MTL fits the A40 on the DEFAULT auto-fit (dataset CPU-resident for the
> big states). Never set `MTL_DATASET_GPU=1` for CA/TX — it OOMs. The host-RAM lazy-fold fix + auto-fit are
> what make it run; the GPU was never the constraint.**

## Implication (the takeaway for planning)
**CA/TX region-MTL no longer need a bigger GPU for VRAM — the A40 can run the whole board.** The H100 /
`closing_data` hardware is no longer forced by memory; it would only be a *speed* choice. Two standing rules
remain: (a) **never drop bs below 2048** — region-MTL *diverges* at small batch (it's a recipe break, not a
memory fix); (b) overlap (8.5× sequences) keeps the same per-batch/model footprint, and the larger dataset
just auto-fits to CPU if needed (as FL overlap demonstrated).
