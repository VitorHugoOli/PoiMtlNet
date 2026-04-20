# MTL with STAN region head — partial lift of the region ceiling

**Date:** 2026-04-20. Experiment ID: MTL-STAN. Branch: `worktree-check2hgi-mtl`.

## Motivation

Our prior finding ([`CONCERNS.md §C04`](../CONCERNS.md), [`issues/REGION_HEAD_MISMATCH.md`](../issues/REGION_HEAD_MISMATCH.md)) established that the MTLnet framework's **default next-head Transformer** (`next_mtl`) collapses to 7.4% Acc@10 on the 1 109-class region task, and swapping in `next_gru` lifts MTL region performance to the ~45–51% Acc@10 band. That swap left an open question: **is the remaining MTL-vs-STL region gap (−6 pp on AL) a fundamental "shared-backbone capacity ceiling", or is it the GRU head's inductive-bias ceiling?**

Once the STAN STL baseline (`next_stan`, [SOTA_STAN_BASELINE.md](SOTA_STAN_BASELINE.md)) lifted the STL ceiling from 56.94 (`next_gru`) to 59.20 Acc@10 on AL, it became easy to test the question directly: **swap STAN in as the MTL region head** and see whether the MTL region number lifts along with it.

This note records that experiment.

## Protocol

Identical to the prior `MTL cross-attn + pcgrad 5f×50ep` runs on AL (`B13` in [BASELINES_AND_BEST_MTL.md](../results/BASELINES_AND_BEST_MTL.md)), with **one** change: the region head factory is overridden via the new `--reg-head next_stan` CLI flag added to `scripts/train.py`. Everything else — task encoders, cross-attention backbone, PCGrad optimizer, OneCycleLR(max_lr=0.003), fair StratifiedGroupKFold(5), seed=42, check-in input for category stream, region embedding input for region stream, 50 epochs, batch 2048 — is bit-exact with the prior run.

```bash
python scripts/train.py \
    --task mtl --task-set check2hgi_next_region \
    --state alabama --engine check2hgi \
    --model mtlnet_crossattn --mtl-loss pcgrad \
    --reg-head next_stan \
    --task-a-input-type checkin --task-b-input-type region \
    --folds 5 --epochs 50 --seed 42 \
    --max-lr 0.003 --gradient-accumulation-steps 1 --no-checkpoints
```

## Alabama results

5-fold × 50 epoch, fair user-disjoint folds.

| Metric | `mtl cross-attn + pcgrad + GRU` (prior) | `mtl cross-attn + pcgrad + **STAN**` (new) | Δ |
|---|---:|---:|---:|
| **next_category F1** | 38.58 ± 0.98 | **39.07 ± 1.18** | +0.49 pp (within σ) |
| next_category Acc@1 | — | 40.48 ± 1.20 | — |
| **next_region Acc@10** (in-dist) | 45.09 ± 5.37 | **50.27 ± 4.47** | **+5.18 pp** |
| next_region Acc@1 | 10.06 ± 1.77 | 12.48 ± 1.44 | +2.42 pp |
| next_region Acc@5 | 32.05 ± 4.29 | 36.62 ± 4.15 | +4.57 pp |
| next_region MRR | 20.94 ± 2.52 | 24.16 ± 2.25 | +3.22 pp |
| next_region macro-F1 | 5.17 ± 1.17 | 7.00 ± 0.49 | +1.83 pp |

**Source:** `docs/studies/check2hgi/results/P8_sota/mtl_crossattn_pcgrad_al_stan_5f50ep.json`. Prior GRU-head reference: `docs/studies/check2hgi/results/P2/ablation_06_crossattn_al_5f50ep.json`.

**Verdict (AL):** Swapping STAN in as the MTL region head lifts **every region metric by 2–5 pp**, with `Acc@10_indist` the cleanest: **+5.18 pp over the GRU-head MTL** at matched everything else. Category F1 moves +0.49 pp within σ — no meaningful change. So the region head change is a **unidirectional region-side improvement**, not a joint trade-off.

## Arizona results — **lift DOES NOT replicate; result REVERSES at AZ scale**

5-fold × 50 epoch, fair user-disjoint folds, identical protocol to AL.

| Metric | `mtl cross-attn + pcgrad + GRU` (prior, az1) | `mtl cross-attn + pcgrad + **STAN**` (new) | Δ |
|---|---:|---:|---:|
| **next_category F1** | 43.13 ± 0.55 | 42.64 ± 0.26 | −0.49 pp (within σ) |
| next_category Acc@1 | 44.00 ± 0.51 | 44.07 ± 0.43 | +0.07 pp |
| next_category MRR | 65.48 ± 0.40 | 65.65 ± 0.18 | +0.17 pp |
| **next_region Acc@10** (in-dist) | **41.07 ± 3.46** | 37.47 ± 4.01 | **−3.60 pp** |
| next_region Acc@1 | 13.20 ± 1.99 | 9.79 ± 1.98 | −3.41 pp |
| next_region Acc@5 | 31.54 ± 3.57 | 26.96 ± 3.50 | −4.58 pp |
| next_region MRR | 22.49 ± 2.49 | 18.53 ± 2.54 | −3.96 pp |
| next_region macro-F1 | 5.44 ± 0.65 | 5.14 ± 0.62 | −0.30 pp |

**Source:** `docs/studies/check2hgi/results/P8_sota/mtl_crossattn_pcgrad_az_stan_5f50ep.json`. Prior GRU-head reference: `docs/studies/check2hgi/results/P2/az1_crossattn_fairlr_5f50ep.json`.

**Verdict (AZ):** MTL-STAN **regresses region by −3.60 pp Acc@10** at AZ scale, opposite direction from AL. Category F1 is unchanged within σ on both states. This is a **scale-dependent reversal**: STAN helps MTL region at AL (10 K rows) and hurts it at AZ (26 K rows).

## Combined AL + AZ picture

| State | Rows | Regions | STL `next_gru` | STL STAN | Δ_STL | MTL `cross-attn+GRU` | MTL `cross-attn+STAN` | Δ_MTL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AL | 10 K | 1 109 | 56.94 ± 4.01 | 59.20 ± 3.62 | **+2.26** | 45.09 ± 5.37 | **50.27 ± 4.47** | **+5.18** |
| AZ | 26 K | 1 540 | 48.88 ± 2.48 | 52.24 ± 2.38 | **+3.36** | 41.07 ± 3.46 | 37.47 ± 4.01 | **−3.60** |

**STL direction:** STAN > GRU on both states, margin grows with scale (+2.26 → +3.36 pp).

**MTL direction:** STAN > GRU on AL, STAN < GRU on AZ. The sign flips between 10 K and 26 K rows.

This is not what we expected. STL STAN consistently wins; MTL STAN wins only at small scale. The MTL setting adds a constraint (shared backbone + PCGrad gradient manipulation across heads) that interacts with STAN's inductive bias differently than with GRU's, and the interaction changes direction with data scale.

**Candidate explanations (to test in follow-up):**

1. **PCGrad gradient projection** penalizes the attention head's gradient direction more heavily when per-task gradients are larger (which happens at scale). The GRU head's smaller, more correlated gradient footprint survives PCGrad projection better.
2. **The cross-attention backbone's output distribution** was implicitly tuned for GRU consumption during the P2 architecture sweep. STAN's bi-layer attention expects input statistics that mildly disagree with the cross-attn block's output at larger data scale.
3. **Head parameter count** in MTL: STAN (~417 K) is comparable to GRU (~770 K) but the distribution of those parameters is very different — STAN's pairwise biases are dense per-position, GRU's recurrent weights are position-agnostic. At 26 K rows per user-disjoint split, STAN's per-position biases may overfit.
4. **Optimizer-head interaction with OneCycleLR**: `max_lr=0.003` may be tuned for recurrent heads. STAN's attention could prefer a different schedule.

All are ablatable but none are a-priori obvious.

## What this means for the CH-M4 / region-ceiling claims

Our earlier paper framing said: *"Region is capacity-ceiling-bound in all MTL architectures; the shared backbone saturates signal extraction from the 9-step region sequence."*

The MTL-STAN result is **more nuanced than expected**:

- **AL (10 K rows):** MTL→STL gap shrinks from 11.85 pp (GRU) → 8.93 pp (STAN). Head partially explains the ceiling.
- **AZ (26 K rows):** MTL→STL gap widens from 7.81 pp (GRU: 48.88−41.07) → 14.77 pp (STAN: 52.24−37.47). Head *worsens* the ceiling.

A single head-swap cannot be claimed as a universal ceiling-lift. The correct framing is:

> *"The MTL region ceiling is layered. At small scale (AL, 10 K), a stronger head (STAN) decodes more signal from the shared backbone. At moderate scale (AZ, 26 K), the MTL setup — cross-attention backbone under PCGrad gradient manipulation — interacts negatively with STAN's attention head, and the GRU head's simpler structure is preferable inside MTL. STL STAN is the correct ceiling reference across scales; MTL head choice is data-scale-dependent."*

This is an **honest scale-dependent result**, not a clean win. It belongs in the paper's Discussion/Limitations, not in the headline claims.

## What this DOES NOT change

- **CH-M1 (asymmetric MTL):** Region is still capped below STL at both scales. On AL, MTL-STAN 50.27 < STL STAN 59.20 (−8.93 pp). On AZ, MTL-STAN 37.47 < STL STAN 52.24 (−14.77 pp). Direction unchanged; magnitude scale-dependent.
- **CH-M4 (cross-attn uniquely closes cat gap):** unchanged across both states — cross-attn's category-side behaviour is head-independent. AL: 39.07 (STAN) ≈ 38.58 (GRU) ≈ STL 38.58. AZ: 42.64 (STAN) ≈ 43.13 (GRU).
- **CH-M8 (scale-dependent transfer):** the STAN result **adds a new scale-dependent axis**: MTL head compatibility reverses between 10 K and 26 K. Worth including as a dimension alongside the cat-transfer scale curve.

## Follow-up runs to consider

1. **FL MTL-STAN** — whether the scale-reversal holds at 127 K (FL) or the MTL-STAN penalty keeps growing with scale. Estimate: 12 h on M2 Pro; bundle with Phase 7 headline runs.
2. **AL/AZ MTL-STAN λ=0** — isolate architectural overhead with the new head. If λ=0 region with STAN is different from the prior GRU λ=0 baseline, we know the head-swap interacts with the backbone-only regime too.
3. **AZ MTL-STAN with static_weight instead of pcgrad** — tests the candidate explanation #1 (PCGrad penalizes attention heads).
4. **AZ MTL-STAN with dselectk backbone** — tests explanation #2 (cross-attn tuned for GRU).

## Paper implications

- Add **two** MTL-STAN rows to [BASELINES_AND_BEST_MTL.md](../results/BASELINES_AND_BEST_MTL.md) Task B tables: AL lift (+5.18 pp) and AZ regression (−3.60 pp). Honest scale-dependent pattern.
- The cross-attention design's "cat-closer" claim **strengthens** — category F1 is robust to head swap across both scales. Cat-transfer is a shared-backbone property, not a head property.
- The region-task claim gains a **nuance**: "MTL region is layered-ceiling-bound (head + backbone); head choice matters at small scale but the optimal head inside MTL is not always the STL-winner" — belongs in Discussion, not headline.
- **CH04 gate update:** still gated by STAN (STL ceiling) on both states. MTL does not exceed STL on either region task at either scale.

## References

- Luo, Liu, Liu. *STAN*, WWW 2021. [arXiv:2102.04095](https://arxiv.org/abs/2102.04095).
- Prior GRU-head MTL: `results/P2/ablation_06_crossattn_al_5f50ep.json`.
- STL STAN baseline: `research/SOTA_STAN_BASELINE.md`.
- Positioning vs HMT-GRN: `research/POSITIONING_VS_HMT_GRN.md`.
