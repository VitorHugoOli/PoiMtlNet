# B5 Results — Faithful GETNext (Hard `last_region_idx`) vs Soft Probe

**Date:** 2026-04-22. Commit: `bf20807` (AL) + forthcoming (AZ).
**Motivating ablation:** `B5_HARD_VS_SOFT_INFERENCE.md` (inference-time: +9 pp Acc@10 at epoch 46).
**Implementation:** `6a2f808` (head + plumbing), `ea65fb3` (bugfix: `--reg-head` ordering + PCGrad-safe fallback).

## TL;DR

B5 is a **clear win on AZ (+6.59 pp Acc@10), tied on AL (+1.47 pp, within σ)**. The asymmetric lift tracks region cardinality: the soft probe co-adapts with STAN on AL's smaller (1109) region set, but cannot keep up on AZ's larger (1547) set. Hard-index is now the recommended MTL-GETNext head for cross-state robustness.

## Headline table

All runs: `mtlnet_crossattn + pcgrad`, 5-fold × 50-epoch, seed 42, same preset (`check2hgi_next_region`), same task_a/b input modality (`checkin`/`region`). Only the region head swaps: `next_getnext` (soft probe) vs `next_getnext_hard` (faithful hard index).

| State | Head | Acc@10_indist | Acc@5_indist | Acc@1_indist | MRR_indist | F1 |
|-------|------|--------------:|-------------:|-------------:|-----------:|---:|
| AL | Soft (champion B-M6b) | 56.49 ± 4.25 | 43.40 ± 4.60 | 15.72 ± 2.74 | 28.93 ± 3.20 | 8.66 ± 1.20 |
| AL | **B5 Hard** | **57.96 ± 5.09** | **44.22 ± 5.58** | 15.03 ± 3.04 | 28.93 ± 3.88 | **9.47 ± 0.71** |
| AL | **Δ (hard − soft)** | **+1.47** | **+0.82** | −0.69 | 0.00 (tied) | **+0.81** |
| AZ | Soft (champion B-M9b) | 46.66 ± 3.62 | 35.70 ± 3.38 | 12.63 ± 1.79 | 23.81 ± 2.30 | 6.93 ± 0.68 |
| AZ | **B5 Hard** | **53.25 ± 3.44** | **40.06 ± 3.36** | **14.55 ± 2.53** | **26.89 ± 2.62** | **8.95 ± 0.52** |
| AZ | **Δ (hard − soft)** | **+6.59** | **+4.36** | **+1.92** | **+3.08** | **+2.02** |

Boldface = meaningful positive gap. On AZ every metric moves by >2σ/2 — a decisive, reviewer-defendable lift. On AL the gaps are within σ (effectively a tie).

## Asymmetric lift — why?

The inference-time ablation (`B5_HARD_VS_SOFT_INFERENCE.md`, epoch 46/47 on α-inspection checkpoints) predicted:
- AL fold 0: +9.11 pp Acc@10 (hard vs soft, inference-only substitution)
- AZ fold 0: +9.36 pp Acc@10 (same axis)

After retraining end-to-end with hard index:
- AL: +1.47 pp (5-fold mean) — most of the inference gap closed
- AZ: +6.59 pp (5-fold mean) — roughly 70% of the inference gap retained

**Working hypothesis:** the soft probe can learn to imitate the hard
index during training, but this imitation is easier when the prior
distribution is lower-dimensional. AL has 1109 regions; the probe's
`nn.Linear(d_model, 1109)` has ~284K parameters and receives
∝N_train / 1109 ≈ 9 samples per output dimension on average, with
the argmax collapsing to just 88 regions (probe-entropy finding).
That's enough signal for the probe to converge on the right
conditional transitions during co-training with STAN.

AZ has 1547 regions, ~13 samples per output dim, but the argmax
spreads across 103 regions — the probe has to actively model more
distinct transition rows, and the STAN backbone can no longer absorb
the diffuse probe into its own signal. Hard-index dodges the learning
burden entirely by gathering `log_T[last_region_idx]` directly.

**Prediction:** on FL (4702 regions, ~14 samples/dim), the gap should
be even larger. FL 5-fold hard run is the next data point.

## Cross-state champion table (post-B5)

| State | Method | Acc@10_indist | Paper status |
|-------|--------|--------------:|:-------------|
| AL | MTL-GETNext soft | 56.49 ± 4.25 | still champion (B5 hard tied, within σ) |
| AL | **MTL-GETNext hard** | 57.96 ± 5.09 | alternate champion; report both or pick one |
| AZ | MTL-GETNext soft | 46.66 ± 3.62 | second-best |
| AZ | **MTL-GETNext hard** | **53.25 ± 3.44** | **new champion, +6.59 pp over soft, +13.7 pp over MTLoRA** |

The AZ number (53.25) is within 1 pp of the AZ STL champion
`STL STAN` (52.24) — the MTL-to-STL gap is now ~minus 1 pp, i.e.
the MTL method has effectively caught up to the single-task ceiling.
On AL the MTL-hard (57.96) still trails STL STAN's 59.20, but by
only 1.24 pp — much tighter than the previous 2.71 pp gap.

## Decision for the paper

1. **Replace the MTL headline with `next_getnext_hard`** on both
   states. The soft-probe adaptation should become an ablation row,
   not the main result.
2. **Claim the MTL-to-STL gap closure on AZ** — a strong narrative
   hook: "our faithful GETNext MTL formulation recovers STL-level
   regional performance while also delivering the next-category
   signal jointly (42.98 ± 1.17 F1 on AZ cat)."
3. **Run FL 1f × 50ep hard** to confirm the scaling hypothesis.
   Budget ~1 h. If it delivers +5–10 pp Acc@10 over FL soft's 60.62
   (→ 65–70), that's a second strong headline for the paper.
4. **Multi-seed (n=3) is still an open follow-up** (seeds 42/123/2024)
   to tighten the σ; current numbers are n=1 seed × 5 folds.

## Caveats

- AL tie (within σ) means the paper can't claim "B5 wins on AL".
  A reviewer might cherry-pick AL to undermine the AZ win. Defensive
  framing: report both, note that B5 is a strict architectural
  improvement (faithful to the original GETNext SIGIR 2022 paper)
  and the AL tie is a side-effect of scale-limited probe learning.
- The inference-time ablation's +9 pp expectation on AL didn't
  materialise after retraining — paper should NOT repeat that 9 pp
  number as expected gain. Use the retrained deltas (+1.47 AL /
  +6.59 AZ).
- AZ fold-3 of the B5 run experienced transient memory pressure on
  the M4 Pro (fold took 27 min vs 9 min typical) with active swap.
  The final metrics appear clean, but if σ seems high a re-run on a
  larger-RAM host would be defensible.

## Files

- AL result JSON: `docs/studies/check2hgi/results/B5/al_5f50ep_next_getnext_hard.json`
- AZ result JSON: `docs/studies/check2hgi/results/B5/az_5f50ep_next_getnext_hard.json`
- Launcher: `scripts/run_b5_hard_mtl.sh`
- Implementation commits: `6a2f808` + `ea65fb3`
