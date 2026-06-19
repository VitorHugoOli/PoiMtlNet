# Lane 2 — overlapping-windows ADOPT validation (in progress)

> A40, `study/pre-freeze-a40`, 2026-06-18. The gated base-change decision (EXECUTION_PLAN §1b/§8 #1):
> validate the overlap effect at FL + small-state multi-seed BEFORE committing the full-base rebuild.
> Champion-G recipe, v14 substrate, geom_simple, KD off, **MIN_SEQUENCE_LENGTH held at 5** (isolate overlap
> vs the AL prior). Overlap engine = `check2hgi_dk_ovl` (v14 re-windowed at stride=1; log_T copied from v14,
> inert in champion G). Per-task diagnostic-best (cat=next_category macro-F1, reg=next_region Acc@10-indist).

## AL (small state) — multi-seed: overlap STRONGLY positive on BOTH heads

| seed | base cat | ovl cat | Δcat | base reg@10 | ovl reg@10 | Δreg |
|---|---|---|---|---|---|---|
| 0 | 52.38 | 61.06 | +8.69 | 64.34 | 68.12 | +3.77 |
| 1 | 53.16 | 61.22 | +8.06 | 64.54 | 67.85 | +3.31 |
| 7 | 53.66 | 61.28 | +7.61 | 64.23 | 67.86 | +3.63 |
| **mean (n=3)** | | | **+8.12** | | | **+3.57** |

Very tight across seeds. Reproduces the overlap memo's AL effect under the current champion-G recipe (memo:
AL STL cat +9.77 / MTL cat +8.9; here MTL cat +8.12). Reg lift (+3.57) is *larger* than the memo's older
AL MTL reg (+0.5/+1.0) — under champion-G's dual-tower the reg head exploits the extra supervision more.

## FL (large state) — TWO findings that gate the ADOPT decision

**(1) Saturation (from the overlap memo, STL):** FL STL cat overlap = **+1.30** (vs AL +9.77) — at FL's
~159k non-overlap windows the model is near data-saturation, so overlap adds little. The benefit is
**scale-dependent**: large at small states, marginal at large states.

**(2) Feasibility: FL overlap MTL OOMs the A40 (even alone).** Champion-G pre-moves the whole dataset to
GPU (`folds.py` `dataset_device=DEVICE` when `num_workers=0`). FL overlap is ~1.38M sequences (8.5× the
159k non-overlap; 26,950 batches/epoch vs 3,150). The pre-moved cat+reg+aux+val tensors at that scale
exhaust the 44 GB A40 (OOM in epoch 1: 44.35/44.42 GB used). This is the **known large-state constraint**
(CA/TX region-MTL also OOM the A40 at bs2048; shrinking batch diverges). **FL-overlap ≈ CA/TX sequence
scale.** So adopting overlap pushes the **dominant board states (FL/CA/TX)** beyond the A40 with the current
data path — the board rebuild at 8.5× sequences needs either a CPU-resident-dataset path (slower) or bigger
GPUs, and is plausibly an **order of magnitude more GPU-hours** (EXECUTION_PLAN §7 "the cost multiplier").

## Verdict-so-far + the decision for the user

- **Overlap is real and strong at small states** (AL multi-seed: cat +8.12, reg +3.57) — not in doubt.
- **At large scale it saturates** (FL STL cat +1.30) **and is expensive/infeasible on the A40** (FL/CA/TX OOM).
- The board is **dominated by large states** (FL/CA/TX), so the headline numbers would move little while the
  rebuild cost multiplies ~8.5× and needs new hardware. This is exactly the **"weak/costly at FL-scale →
  weigh before committing the full board"** finding the plan pre-registered → **STOP for the user.**

**Open data point:** the actual FL **MTL** overlap delta (does MTL saturate like STL's +1.30?) is unmeasured —
blocked by the A40 OOM. Obtainable only via a CPU-resident-dataset training path (a new low-risk env knob;
default off) + a multi-hour run, OR a bigger GPU. Whether to spend that is part of the user decision.

Captures: `results/lane1_g01/{alabama_s{0,1,7}__{baseline,ovl}, florida_s0__baseline}/`.

## Meticulous audit (2026-06-19) — overlap gain is SOUND, not a confound

User flagged the overlap as possibly "weird / losing the big picture." Independent audit + my measurements:

**FL overlap fold-1 (corrected): cat 77.20 (+4.19), reg 74.25 (+0.71).** (An earlier note quoted cat ~72.68 —
that was a *mid-training running-best of fold 2*, not the diagnostic-best; disregard it.) So FL overlap is
**positive on both heads**, scale-saturated on reg — matching the memo's STL pattern (AL +9.77 / FL +1.30).

**Training-length confound — REFUTED (the central worry).** Overlap runs ~8.5× more gradient updates/fold at
fixed 50 epochs, so the fear was the non-overlap baseline is under-trained. Convergence curves refute it:
AL baseline cat plateaus by epoch 20 (57.1%), reg by epoch 26-38 — flat to epoch 50; overlap reaches a
*higher* plateau *faster* (cat by ep16, reg by ep27). Both arms reach OneCycle's low-LR tail before stopping.
The captures are per-task **diagnostic-best** (peak epoch, selector-independent) = best-of-run vs best-of-run.
⇒ the +8.12 cat / +3.57 reg is a **higher ceiling from denser supervision, not extra updates**.

**Data + leak — CLEAN.** Leak re-verified empirically on the real 108k-row AL overlap data: all 5 folds
user-disjoint at seeds {0,1,7}. AL (06-03) and FL (today, fixed `<U32` builder) overlap engines are
build-consistent (same schema, MIN=5, v14-symlinked embeddings). 0.0% last_region padding is expected at
stride-1. **Same user population** (AL 1622=1622, FL 13935=13935, zero set difference) — overlap changes only
how many windows each user contributes, not which users. Power-user reweighting is **mild** (Gini +0.04,
top-decile window share +3pp) → one-line footnote, not a population swap.

**Caveats for paper-grade claims:** (a) FL reg *baseline* is mildly still rising at epoch 50 (+~1pp tail) →
train non-overlap to convergence; this would *shrink* the already-small FL reg gain, never inflate it.
(b) Folds are generated on-the-fly per arm (different windowed rows) → partitions are NOT bit-identical across
arms → use **unpaired across-seed** stats, not paired per-fold. (c) Footnote the ~3pp power-user reweighting.

**Verdict:** the "weird" feeling was a premature mid-training comparison, not a methodological flaw. Overlap is
sound and genuinely helps (scale-dependent: large at small states, saturated-but-positive at large states) —
**supports the user's ADOPT lean.** Methodological notes above travel into the freeze record.

## Perf notes (2026-06-19)
- 43 min/FL-overlap-fold = **proportional** (8.5× windows × ~5 min non-overlap fold); both run 50 epochs.
- **Dataset can go back on GPU** now the train-logit accumulation is CPU-side (`8ff36dba`): ~6.4 GB dataset +
  ~16 GB ≈ 22 GB < 44 GB → drop `MTL_DATASET_CPU` for FL-overlap → no per-batch transfer → ~28% faster,
  byte-identical. Use GPU-resident for the board where it fits; CA/TX may still need CPU-residency (bigger).
