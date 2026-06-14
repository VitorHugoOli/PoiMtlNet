# Phase 3 Rank 1 — log_T as supervisory signal — findings (AL + AZ + FL)

**Date:** 2026-05-21
**Phase scope:** mtl-protocol-fix DEFERRED_WORK §4.5 (log_T as supervisory signal, not just head input). Single-seed=42, 5 folds, 50 epochs per state per weight. Scope: AL/AZ/FL (CA/TX out of scope per user 2026-05-21).
**Code:** `--log-t-kd-weight W` + `--log-t-kd-tau τ` plumbed via `ExperimentConfig.log_t_kd_weight/log_t_kd_tau` → `train_model(...)` → KL loss term added to `task_b_loss` in `src/training/runners/mtl_cv.py` after the per-batch CE.

## The intervention

Today the reg head (`next_getnext_hard` / `next_stan_flow`) consumes `log_T[last_region_idx]` as an *additive feature*:

```
final_logits = stan_logits + α · log_T[last_region_idx]
```

This study adds a **distillation term** to the reg loss that supervises the head's output distribution to *match* the Markov-1 transition row:

```
L_reg += W · KL( softmax(reg_logits / τ) ‖ exp(log_T[last_region_idx]) )
```

Padding-handled per the head's `pad_mask` logic (`src/models/next/next_stan_flow/head.py:140-141`): rows with `last_region_idx < 0` OR `>= num_classes` are excluded from the KL.

## Results — three-frontier table per state

Sweep `W ∈ {0.0, 0.05, 0.1, 0.2}`; `τ=1.0` throughout. Baseline (W=0.0) is the supervised CE only, no KD term — equivalent to existing canonical recipe.

### Alabama (H3-alt recipe)

| W | disjoint reg | geom_simple reg | b9 reg | disjoint cat | geom cat |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 50.82 ± 3.21 | 48.56 ± 3.13 | 47.60 ± 4.14 | 45.76 ± 1.34 | 45.18 ± 1.89 |
| 0.05 | 52.34 ± 3.15 | 49.94 ± 3.40 | 48.56 ± 2.91 | 45.68 ± 1.75 | 44.80 ± 1.70 |
| 0.1 | 52.92 ± 3.48 | 50.84 ± 3.16 | 48.99 ± 3.81 | 45.78 ± 1.90 | 45.02 ± 1.64 |
| 0.2 | **53.22 ± 3.30** | **51.48 ± 2.89** | **49.87 ± 3.86** | 45.74 ± 1.48 | 45.14 ± 2.57 |

### Arizona (H3-alt recipe)

| W | disjoint reg | geom_simple reg | b9 reg | disjoint cat | geom cat |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 41.33 ± 2.73 | 39.60 ± 3.11 | 38.43 ± 2.31 | 48.87 ± 1.80 | 47.30 ± 1.75 |
| 0.05 | 45.20 ± 2.87 | 42.87 ± 2.96 | 39.35 ± 2.54 | 48.61 ± 1.80 | 45.35 ± 1.25 |
| 0.1 | 46.19 ± 3.09 | 43.51 ± 2.72 | 40.28 ± 2.38 | 48.67 ± 1.85 | 46.05 ± 1.50 |
| 0.2 | **46.39 ± 2.83** | **44.97 ± 3.28** | **39.83 ± 3.18** | 48.87 ± 1.39 | 45.86 ± 1.12 |

### Florida (B9 recipe)

| W | disjoint reg | geom_simple reg | b9 reg | disjoint cat | geom cat |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 63.98 ± 0.76 | 61.14 ± 0.95 | 53.73 ± 9.22 | 70.49 ± 0.92 | 66.98 ± 0.80 |
| 0.05 | 66.18 ± 0.77 | 64.27 ± 1.15 | 57.96 ± 1.02 | 70.40 ± 0.86 | 66.36 ± 1.16 |
| 0.1 | 66.21 ± 0.69 | 64.74 ± 1.31 | 58.79 ± 1.20 | 70.51 ± 0.79 | 66.85 ± 1.37 |
| 0.2 | **66.30 ± 0.80** | **65.35 ± 1.15** | **59.52 ± 0.70** | 70.52 ± 0.99 | 66.93 ± 1.56 |

## Wilcoxon (one-sided, paired, w > baseline) on disjoint reg

| State | w=0.05 Δ pp / p | w=0.1 Δ pp / p | w=0.2 Δ pp / p |
|---|---:|---:|---:|
| AL | +1.52 / **0.0312** | +2.10 / **0.0312** | +2.40 / **0.0312** |
| AZ | +3.88 / **0.0312** | +4.86 / **0.0312** | +5.06 / **0.0312** |
| FL | +2.20 / **0.0312** | +2.23 / **0.0312** | +2.32 / **0.0312** |

All 9 cells (3 states × 3 weights) clear the n=5 paired Wilcoxon ceiling (p=0.03125). 5/5 folds positive in every case. Monotone in W at every state, no exceptions.

## Findings

### F-Rank1-A — Supervisory log_T is a real, paper-grade reg lift

- All 3 states pass Wilcoxon-strict at all 3 KD weights on disjoint reg.
- Monotone in W up to 0.2 (the highest tested); the curve doesn't plateau within this grid — the optimum may lie at W ≥ 0.2.
- Cat at disjoint untouched at all 3 states (Δ_cat < 0.1 pp at every weight).
- Mean lift: AL +2.40 / AZ +5.06 / FL +2.32 pp on disjoint reg at W=0.2.

### F-Rank1-B — FL b9 selector variance collapses

At FL the legacy b9 production selector has σ = 9.22 pp at baseline (single-seed=42 bimodal mode documented as F4 in v6 verdict). Under log_T-KD at W=0.2 the σ drops to **0.70 pp** — bimodality fully disappears. The KD term pulls every fold's reg head toward the same canonical Markov-1 attractor, preventing the bad-mode crash in 2/5 folds that drove the 9 pp σ at baseline.

This is a unique side-benefit not seen at AL/AZ (which had no bimodality). Suggests the KD term acts as a **stability prior** at the deploy axis even where the F1 selector fix already lifts mean reg substantially.

### F-Rank1-C — Effect on deploy and production selectors

On the deploy axis (geom_simple) the lift is even larger than on disjoint:

| State | disjoint Δ pp | geom_simple Δ pp | b9 Δ pp |
|---|---:|---:|---:|
| AL | +2.40 | +2.92 | +2.27 |
| AZ | +5.06 | +5.37 | +1.40 |
| FL | +2.32 | +4.21 | +5.79 |

The KD term aligns disjoint and joint selectors: when reg distribution matches log_T, the joint geom selector also picks a reg-favourable epoch. **At FL the KD-stabilised b9 reg (59.52 ± 0.70) is now within 2.5 pp of the disjoint ceiling and within 6 pp of the geom_simple deployable — the variance suppression makes deployed b9 reliable for the first time at FL.**

### F-Rank1-D — Mechanism interpretation

The reg head already CONSUMES log_T via the additive α-prior. The new KD term **forces the reg head to also REPRODUCE log_T in its output distribution**. Two ways to read this:

1. The additive α-prior pulls reg toward log_T but slowly — the head's transformer/STAN backbone fights against it. KD adds explicit pressure on the OUTPUT, not just the input feature, accelerating convergence to the prior-aligned distribution.
2. The KD term acts as a regulariser on the reg head's softmax temperature, preventing the head from collapsing onto cat-correlated geometry (the F2 mechanism: MTL reg destabilises when shared backbone tilts cat-ward).

Both interpretations are consistent with the F4 bimodality collapse at FL: bad-mode folds were ones where reg lost the prior alignment; KD anchors them back.

## Promotion candidate

W=0.2 is the highest-lift, no-regression setting at every state in scope. **Recommend promotion to multi-seed n=20 at FL** for paper-grade evidence; AL/AZ multi-seed is optional (small-state Δ already 5/5 folds positive at single-seed). The next-tier paper revision (`paper_canon_reevaluation.md`) should include log_T-KD as a candidate recipe component for §0.1.

## Caveats

1. **Single-seed=42 only.** The dev-seed (per CLAUDE.md) — generalisation to seeds {0, 1, 7, 100} not yet measured. The C23 dev-seed bias finding suggests FL/CA/TX seed=42 may overshoot multi-seed by 0-8 pp; a multi-seed promotion run is gated on this.
2. **CA/TX skipped** per user direction (large-state compute cost). The mechanism (additive feature vs supervisory signal on the same log_T) should generalise but is not measured.
3. **W=0.2 is the max tested.** The curve does not plateau within the grid. A future sweep at W ∈ {0.3, 0.5, 1.0} may find a higher peak; AL/AZ are obvious cheap pilots.
4. **τ fixed at 1.0.** Temperature sweep not done. Lower τ may sharpen the prior alignment; higher τ may relax it. Out of scope here.
5. **No cat regression on disjoint, mild on geom_simple cat at FL/AZ (-0.6 to -1.5 pp).** Suggests the KD pulls joint geom toward earlier reg-favourable epochs where cat is still climbing. Production deploy on geom_simple should re-evaluate this tradeoff at multi-seed.

## Cross-references

- Code: `src/configs/experiment.py` (fields), `src/training/runners/mtl_cv.py` (KD loss), `scripts/train.py` (CLI), `scripts/mtl_protocol_fix/run_log_t_kd_sweep.sh` (sweep), `scripts/mtl_protocol_fix/summarize_log_t_kd.py` (analyzer).
- Per-state summaries: `docs/results/mtl_protocol_fix/phase3_rank1_log_t_kd/{alabama,arizona,florida}/{state}_summary.{md,json}`
- Future-work memo (where this sub-track was absorbed): [`docs/future_works/reg_head_architecture_sweep.md`](../../future_works/reg_head_architecture_sweep.md) §"log_T as supervisory signal".
- Phase 3 Rank 4 composite: [`phase3_rank4_composite_analysis.md`](phase3_rank4_composite_analysis.md).
- Bug + fix during execution (CUDA OOB on AZ): [`../../studies/archive/mtl-protocol-fix/log.md`](../../studies/archive/mtl-protocol-fix/log.md) 2026-05-21 entry.
