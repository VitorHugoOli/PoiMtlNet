# Speed levers — why the memory fixes didn't make training faster (controlled probe)

> A40, `study/pre-freeze-a40`, 2026-06-19. Question (user): the dataset-on-GPU auto-fit + S1/S2 fixes
> were expected to speed training up, but FL-overlap MTL stayed ~50 s/epoch. Why? Controlled clean-GPU
> probe (FL overlap, champion-G, current code), 4 arms, steady-state per-epoch isolated from warmup.

## Result — the speed ladder (FL overlap MTL, per STEADY epoch, clean A40)

| arm | per-epoch (steady) | Δ vs baseline | byte-identical? |
|---|---|---|---|
| dataset **CPU**-resident (`MTL_DATASET_CPU=1`) | ~55.2 s | baseline | yes |
| dataset **GPU**-resident (`MTL_DATASET_GPU=1`) | ~54.3 s | **−1.5%** (noise) | yes |
| GPU-resident **+ TF32** (`--tf32`) | ~54.0 s | **−0.6%** (noise) | **no** (fp rounding) |
| GPU-resident + TF32 **+ `--compile`** | **~45.7 s** | **−15%** (real) | **no** (fused kernels) |

Compile also pays a **one-time warmup**: epoch 1 = 95 s (vs ~46 s steady) ≈ +49 s/fold (amortizes over a
50-epoch fold → net ~−14%; folds 2–5 may reuse the compiled graph in-process → larger).

## Diagnosis — the champion is OVERHEAD/LAUNCH-bound, not transfer- or FLOP-bound

- **Dataset-on-GPU ≈ 0%** → the loop is NOT host→device-transfer-bound. (This **refutes** the old
  `LANE2 §Perf` guess of "~28% faster from GPU-residency" — that estimate was wrong for this model.)
  The auto-fit's value is *avoiding OOM + removing manual CPU/GPU guessing*, not acceleration.
- **TF32 ≈ 0%** → NOT fp32-matmul-FLOP-bound. The matmuls are small (dim 64) and not the bottleneck.
- **`torch.compile` −15%** → kernel **fusion** helps, so a real chunk of the cost IS GPU kernel-launch
  overhead (many small ops: the GRU recurrence, cross-attn, the mixed-batch loop). But only −15%, because
  ~25% of wall-time is CPU-side (the mixed-batch Python iteration, the per-batch metric `.cpu()`
  accumulation from S1, data prep) which compile does NOT touch. GPU util sat at ~75% (free GPU) — the
  ~25% gap is that CPU/Python overhead.

## Why the fixes didn't speed things up (the answer)

1. **They were memory/OOM/correctness fixes, not compute fixes.** S1 (streaming train-metric), S2 (chunked
   val), dataset auto-fit, the `<U32` builder fix all reduce RAM/VRAM so the overlap run **fits**. Before
   them FL-overlap MTL **OOM'd — it didn't run at all**, so there was no "slow before" to speed up. Some
   even trade a little speed *for* memory (S1's per-batch `.cpu()`; S2's chunked val loop).
2. **The genuine speed knobs are deliberately OFF for byte-identity.** `--tf32` and `--compile` are
   default-off ("so paper runs match NORTH_STAR exactly"); both change the numerics. Here TF32 wouldn't
   help anyway (~0%); compile would help ~15% but breaks byte-identity → **post-freeze only**.
3. **Most of the cost is intrinsic.** Overlap is 8.5× the sequences; the per-step cost is set by the
   cross-attn dual-tower + GRU. No memory fix changes that.

## Result-neutrality check (compile+TF32 vs byte-identical baseline, 2026-06-19)

Champion-G FL non-overlap, seed 0, 5-fold, **GPU-resident + TF32 + `--compile`** vs the byte-identical
baseline (no knobs) **73.0116 / 73.5414**:

| metric | compile+TF32 | baseline | Δ |
|---|---|---|---|
| cat macro-F1 | 73.0577 ± 0.80 | 73.0116 | **+0.046 pp** |
| reg top10_acc_indist | 73.6067 ± 0.71 | 73.5414 | **+0.065 pp** |

Δ ≈ +0.05 pp on both heads — **far below** the ±0.8 pp fold std / ~±1 pp seed noise. **compile+TF32 is
result-neutral** (neither improves nor degrades — it's fp-ordering noise), and gives ~15% speed.

**✅ DECISION (user-confirmed 2026-06-19): ADOPTED for the P3 board.** Pinned as execution flags `--compile
--tf32` (+ GPU-resident auto-fit) on every P3 cell — they are execution knobs, NOT recipe identity (v16
model/heads/loss unchanged; frozen embeddings untouched, compile is training-only). Run the WHOLE board
compiled ONCE (never mix compiled + non-compiled cells); reviewers reproduce *with* them; the no-knobs
byte-identical anchors are the reference and board cells sit ~0.05 pp above. torch stays 2.11 (toolchain
blocker, §10); workers skipped (non-bottleneck + non-deterministic). Recorded in EXECUTION_PLAN §4b +
closing_data/RUN_MATRIX.md (Execution config PINNED).

## Recommendation (aligned to the DECISION above)
- **compile+TF32: ADOPTED for the P3 board** (user-confirmed 2026-06-19) — result-neutral (+0.05 pp, within
  noise), ~15% faster. Pinned as execution flags (training-only; v16 recipe + frozen embeddings unchanged);
  the whole board runs compiled ONCE; reviewers reproduce *with* them. This re-baselines the board's absolutes
  by ~0.05 pp vs the no-knobs anchors — acceptable (the board isn't built yet → built-once, no waste).
- **torch stays 2.11** (§10 NO-GO: `torch_cluster` cu128 wheel + topk re-baseline). **`num_workers` skipped**
  (non-bottleneck with GPU-resident data + non-deterministic). The remaining overhead is CPU-side (mixed-batch
  Python loop + per-batch metric) — a post-freeze target if more speed is wanted.

Captures: `/tmp/lane1/speed_{cpu,gpu,gpu_tf32,compile}.log`; methodology = cumulative tqdm elapsed at epoch
boundaries (538 batches/epoch), steady = epochs 2+ (compile warmup = epoch 1 excluded).

---

## ⚠ UPDATE 2026-06-20 — the "never mix" rule had an unhandled exception (the STL **reg** ceiling)

User-raised concern: a *systematic, one-sided* compile bias is worse for a matched MTL-vs-STL comparison
than random noise of the same size (noise averages out across seeds/folds; a systematic shift does not).
So "+0.05 pp is within noise" is **not sufficient** — what must hold is that the **delta** (MTL − ceiling)
is preserved. Audit of the board tooling against the PINNED rule ("run the WHOLE board compiled, never mix
compiled/non-compiled cells"):

- **cat** comparison — MTL (`train.py`) vs STL-cat ceiling (`train.py --task next`): both compilable ⇒ **uniform ✓**.
- **reg** comparison — MTL (`train.py`) vs STL-reg ceiling (`p1_region_head_ablation.py`): **p1 had NO
  `--compile`/`--tf32`** ⇒ MTL-compiled vs STL-**un**compiled ⇒ the rule was **silently violated for reg**.
  Worse, p1 trains **fp32** (no autocast) while the MTL trainer is **fp16-autocast**, so `--tf32` perturbs
  the STL reg matmuls *more* than the MTL ones — an asymmetric, head-specific shift on the tightest delta
  (reg Δ −0.09…−0.31).

**Fixes:**
- **(b)** `--compile`/`--tf32` added to `p1_region_head_ablation.py` (this commit) so the STL reg ceiling
  CAN share the MTL execution recipe (the board can now be genuinely uniform).
- **(a)** Until validated, the **comparison-critical cells run UNCOMPILED** (byte-identical) — do NOT rely
  on the assumed neutrality. Uniform-compiled is allowed ONLY after the delta-neutrality test below passes.

## Pending: delta-neutrality validation (before the board may flip to uniform-compiled)
Measure the **delta** (MTL − ceiling), fold-paired, focusing on **reg** (tie-break sensitive), on a SMALL
(AL) and a HUGE (TX, 3.83M) state, overlap windowing:
- `Δ_off`  = MTL_off − ceiling_off (both uncompiled)  ← trustworthy reference
- `Δ_on`   = MTL_on  − ceiling_on  (both compiled, **uniform**)  ← if ≈ `Δ_off`, uniform compile is comparison-safe
- `Δ_mixed`= MTL_on  − ceiling_off (the board's pre-fix reg asymmetry)  ← quantifies the bias removed

Pass criterion: `|Δ_on − Δ_off|` ≪ fold σ (~±0.8 pp) on reg, at BOTH states. Single seed (42) detects a
systematic shift (it appears at n=1); escalate to {0,1,7,100} only if the shift lands near the noise floor.
