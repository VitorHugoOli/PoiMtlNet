# PRECISION_LESSONS — the board's numeric-stability + schedule rules (compact)

> **One doc, three merged lessons (2026-07-08 compaction).** Full forensic writeups live in
> [`archive/lessons/`](archive/lessons/) — `CA_MTL_DIVERGENCE.md` (fp16 root cause), `TX_A40_BF16_NAN.md`
> (Ampere-bf16 root cause), `EP100_ABLATION_AND_TX_RAM.md` (schedule ablation + host-RAM mechanics). This page
> carries every **operative rule**; read the archives only for the evidence trail.

## The rules (what you must actually do)

1. **fp32 is the board invariant** (`MTL_DISABLE_AMP=1`, exported by `p3_board.sh` since PR #57; auto-fp32 covers
   train AND eval for reg C>2000 on bare runs). Never bare-fp16; bf16 only as a labeled corroboration arm.
2. **Never cite a collapse artifact.** The tell: reg best-epoch ≤ ~5, tens of thousands of skipped optimizer steps,
   or a sudden both-heads ~3% floor. Known VOID cells: CA `*_partial` fp16 (−5.23 "breach"), TX `*_bf16` (−2.37) +
   old fp16 (−2.41).
3. **50 epochs is settled** — do not "give it more epochs" because best-epochs sit late (ep43-50). That is the
   OneCycle anneal-tail, not under-training: at 100ep the peak just relocates (ep69/79) and lands marginally LOWER
   (AL cat +0.21/reg −0.39; FL cat −0.53/reg −0.18 → NULL). Frozen 50ep cells stand.
4. **Large-C on Ampere = fp32, period.** The A40 (Ampere) hits a bf16 *backward-pass* NaN at CA/TX scale that the
   H100 (Hopper) does not, under byte-identical everything — a device-class trajectory difference, not a code bug.
5. **CA/TX host-RAM:** keep the dataset CPU-resident (auto-fit; NEVER `MTL_DATASET_GPU=1` at large states) and mind
   the fold-construction RAM guard (serial, not 2-wide, at large C on the 46 GB A40).

## Why (the three findings, one paragraph each)

- **CA ep30 collapse = fp16-autocast-no-GradScaler overflow** (H100 lane, 2026-06-22/23). Champion-G MTL at CA
  collapsed both heads to ~3% at a deterministic ep30 — identically under "tf32" and "fp32" flags, because the
  trainer silently ran fp16 autocast regardless. The 8501-wide fp16 reg logits overflow (65504) → NaN →
  `clip_grad_norm_` (inf-norm → coef 0) poisons the shared backbone. The first-reported CA Δreg −5.23 "breach" was
  the pre-collapse early peak of a dying run — VOID. Fix = real fp32/bf16; the fix graduated into the #43 AMP gate +
  the PR #57 auto-fp32(train+eval). **The old "region cost grows with cardinality" narrative was this artifact.**
- **TX A40-bf16 NaN = an Ampere device-class backward instability** (A40 lane, 2026-06-24, 5-agent investigation).
  The A40 TX bf16 run NaN'd from ep33 (74,812 skipped steps; finite loss, NaN grad — NOT the fp16 forward overflow)
  while the H100 ran the byte-identical cell clean. Every transferable cause was cleared (code, recipe, substrate,
  log_T, torch). Fix = true fp32: A40 TX fp32 CLEAN, reg 67.02 beats the ceiling +2.06, matching the H100 (67.13).
  (`MTL_STAN_FP32_ATTN=1` remains an *unvalidated* bf16 mitigation candidate — see SYSTEM_REFERENCE §3.)
- **100-epoch ablation = NULL** (H100, 2026-06-24). With OneCycle, best-val lands near the low-LR anneal tail at ANY
  schedule length — late best-epochs are not "still climbing". Verified at AL (5f) and FL (fold-1).

Board-facing summaries: `RESULTS_BOARD §2` (verdicts) + §1 caveats (VOID list). Device policy + env flags:
`docs/SYSTEM_REFERENCE.md §3`, `docs/studies/pre_freeze_gates/DEFAULTS_AND_GUARDS.md`.
