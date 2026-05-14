# B7 — ALiBi Init × GETNext Head on Alabama

**Date:** 2026-04-22. Run: `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260422_0048/`.

## Question

ALiBi-decay bias initialization reduced σ on AZ STAN d=256 by 28% at identical mean (see `STAN_FOLLOWUPS_FINDINGS.md §ALiBi`). Does the same stabilization effect carry over to the **GETNext** head on AL, where per-fold variance is the paper's biggest risk (σ≈4 pp on Acc@10)?

## Setup

Identical to PCGrad + GETNext AL baseline, with a single flag added:

```
--reg-head-param bias_init=alibi
```

- `mtlnet_crossattn` + `pcgrad` + `next_getnext` d=256, num_heads=8
- 5 folds × 50 epochs, seed=42, max_lr=0.003
- Same `log_T` transition matrix

## Results — Alabama 5-fold

| Metric | Baseline (no ALiBi) | + ALiBi | Δ (mean) | σ ratio |
|---|---:|---:|---:|---:|
| F1 | 9.10 ± 0.68 | 9.25 ± 0.79 | +0.15 | +16% |
| Acc@1_indist | 15.92 ± 1.74 | 16.83 ± 2.18 | +0.91 | +25% |
| Acc@5_indist | 43.47 ± 3.78 | 44.51 ± 3.33 | +1.04 | −12% |
| **Acc@10_indist** | **56.38 ± 4.11** | **57.46 ± 3.66** | **+1.08** | **−11%** |
| MRR_indist | 29.07 ± 2.43 | 29.86 ± 2.53 | +0.79 | +4% |

## Interpretation

- **Headline metric (Acc@10):** +1.08 pp mean, −11% σ. Both directions are helpful. Net reviewer impact is borderline — the lift is within σ but the σ itself shrinks.
- **Metric-mixed variance:** σ tightens for Acc@5 / Acc@10 (the top-k metrics that matter for paper claims) but widens for F1 / Acc@1. This is consistent with the ALiBi → STAN story: ALiBi smooths the attention bias across long-distance steps, which stabilizes top-k ranking but does not help sharpness at rank 1.
- **No regression.** ALiBi does not degrade any metric on AL.
- **Compare to AZ STAN:** the STAN-on-AZ finding was σ −28% at identical mean. On GETNext-on-AL we observe −11% σ and +1.08 pp mean. Directionally consistent (ALiBi reduces top-k variance) but weaker.

## Decision

**Keep as optional paper artefact, not default.** ALiBi + GETNext is modestly better on AL Acc@10 (+1 pp, −11% σ) but:
1. The gain is within one standard deviation — no reviewer-facing significance.
2. Adding a second head-variant column to the main results table adds clutter.
3. Multi-seed (#B3, task #137) will give a firmer σ estimate and may flip the picture.

**Recommendation:** leave `bias_init="gaussian"` as the default. Mention ALiBi + GETNext in the paper's *variance-reduction* subsection as an optional stabilization technique, not the headline config.

## Follow-ups not run

- ALiBi + GETNext on AZ (20 min). AZ variance on GETNext is already lower (σ=2.93); ALiBi's stabilizer effect should be smaller there. Low value.
- ALiBi + GETNext on FL (~6 h 5-fold). Out of scope for this pass.

## Commands

```bash
python scripts/train.py \
    --task mtl --task-set check2hgi_next_region \
    --state alabama --engine check2hgi \
    --folds 5 --epochs 50 --seed 42 \
    --task-a-input-type checkin --task-b-input-type region \
    --model mtlnet_crossattn --mtl-loss pcgrad \
    --reg-head next_getnext \
    --reg-head-param d_model=256 --reg-head-param num_heads=8 \
    --reg-head-param bias_init=alibi \
    --reg-head-param transition_path=/tmp/check2hgi_data/check2hgi/alabama/region_transition_log.pt \
    --max-lr 0.003 --gradient-accumulation-steps 1 --no-checkpoints
```

Total wall-clock: 23 min on M4 Pro MPS.
