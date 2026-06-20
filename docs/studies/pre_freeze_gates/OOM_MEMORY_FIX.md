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

## Implication (the takeaway for planning)
**CA/TX region-MTL no longer need a bigger GPU for VRAM — the A40 can run the whole board.** The H100 /
`closing_data` hardware is no longer forced by memory; it would only be a *speed* choice. Two standing rules
remain: (a) **never drop bs below 2048** — region-MTL *diverges* at small batch (it's a recipe break, not a
memory fix); (b) overlap (8.5× sequences) keeps the same per-batch/model footprint, and the larger dataset
just auto-fits to CPU if needed (as FL overlap demonstrated).
