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

## Arizona results — TBD

Running at time of writing (5f×50ep, AZ, same protocol). Will confirm whether the AL lift replicates at 2.5× more data. ETA ~30 min.

## What this means for the CH-M4 / region-ceiling claims

Our earlier papers framing said: *"Region is capacity-ceiling-bound in all MTL architectures; the shared backbone saturates signal extraction from the 9-step region sequence."*

The MTL-STAN result **partially** refutes that framing on AL:

- The `next_gru` MTL region ceiling (45.09) was lower than the STAN MTL region ceiling (50.27) by 5 pp — even though STL GRU (56.94) and STL STAN (59.20) differ by only 2.26 pp.
- That is, the **MTL→STL gap** went from 11.85 pp (STL GRU − MTL GRU) down to **8.93 pp** (STL STAN − MTL STAN) by changing just the head.
- This is strong evidence that **a non-trivial fraction of the "ceiling" we attributed to shared-backbone capacity was actually head-capacity** — STAN's bi-layer attention extracts more signal from the backbone's output than GRU does.

**Updated framing (for the paper's Discussion):** The MTL region task has *two* capacity bottlenecks: the shared-backbone's output and the head's ability to decode it. On AL, swapping the head closes about 25% of the prior MTL→STL gap without hurting the category task. The remaining gap (~9 pp) is a genuine shared-backbone dilution effect.

## What this DOES NOT change

- **CH-M1 (asymmetric MTL):** STAN-MTL region 50.27 is still below STL STAN 59.20 (−8.93 pp). Region is still capped below STL. Direction unchanged; magnitude smaller.
- **CH-M4 (cross-attn uniquely closes cat gap):** unchanged — category F1 is still at STL-parity with cross-attn (39.07 vs STL 38.58, within σ).
- **CH-M8 (scale-dependent transfer):** pending AZ + FL confirmation with the STAN head.

## Follow-up runs to schedule

1. **AZ MTL-STAN** — running, confirms AL result replicates at 26K-row scale.
2. **FL MTL-STAN** — paper-blocking if FL number changes the headline. Estimate: 12 h on M2 Pro; can be bundled with Phase 7 headline runs (add a 6th config: `mtl_crossattn_pcgrad_stan_5f50ep`).
3. **AL MTL-STAN λ=0** — isolate architectural overhead with the new head. If the overhead shrinks too, we have a cleaner decomposition story for the paper.

## Paper implications

- Add an MTL-STAN row to [BASELINES_AND_BEST_MTL.md](../results/BASELINES_AND_BEST_MTL.md) Task B AL table.
- The cross-attention design's "cat-closer" claim strengthens: it closes the cat gap to STL AND admits a stronger region head without hurting category — the shared backbone produces enough for STAN to decode 5 pp more than GRU did.
- In the Limitations/Discussion section, reframe "MTL region ceiling" as a **layered** ceiling (head + backbone) rather than a single capacity bound.

## References

- Luo, Liu, Liu. *STAN*, WWW 2021. [arXiv:2102.04095](https://arxiv.org/abs/2102.04095).
- Prior GRU-head MTL: `results/P2/ablation_06_crossattn_al_5f50ep.json`.
- STL STAN baseline: `research/SOTA_STAN_BASELINE.md`.
- Positioning vs HMT-GRN: `research/POSITIONING_VS_HMT_GRN.md`.
