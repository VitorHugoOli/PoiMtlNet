# v18 — proposed final settings for the full regeneration

> **Status: AWAITING AUTHOR APPROVAL.** Nothing below is adopted yet. This is the settings sheet the
> full n=20 regeneration would run under (6 datasets × 4 seeds × 5 folds × 3 model families).
>
> Every number cited here is on disk under `docs/results/closing_data/v18_sweep/`. Evidence grade is
> stated per row: **[5f]** = 5 folds with a paired test, **[1f]** = single-fold screen (direction
> only, no dispersion), **[—]** = not measured, carried from v17.
>
> Last updated 2026-08-09.

## The one change that matters: logit adjustment replaces class weighting

`--logit-adjust-tau 0.5` adds `τ·log P_train(y)` to the logits **at train time only** and *replaces*
class weighting (weighted CE is the `else` branch: `next_cv.py:123-141`, `mtl_cv.py:504-523`).
Measured at **five of six datasets**, both model families, always positive on the category axis:

| where | family | folds | Δ category | note |
|---|---|---:|---:|---|
| AL / AZ / IST | dedicated | 5 | **+2.86** | all p ≤ 0.0014 |
| AL / AZ / IST | MTL | 5 | **+3.18** | all p ≤ 0.0014 |
| **texas** | dedicated | 1 | **+3.010** vs weights-ON | **+1.453** even vs weights-OFF, the step-B winner |
| **florida** | MTL | 1 | **+1.475** | reg +0.025, **geom +1.074** |

**This dissolves the size-tiered class-weight rule.** §11.2 found class weighting had to be ON at
alabama (−1.203 if removed, p=0.004) and OFF at texas (+1.556 if removed, p=0.030) — opposite signs
at the two ends of the size range. Logit adjustment beats *both* of those tuned endpoints with **one
setting**. The tiering is now a background fact about a configuration we no longer ship.

**τ = 0.5, not 1.0.** τ=1.0 was a flat null at all three small states. Not a suspicious
non-monotonicity — at τ=1 you fully invert to the balanced posterior, which over-boosts rare classes
the model has not learned well. The same τ=0.5 optimum was found independently by T1.4 on a
different substrate (stride-9 vs v18's stride-1).

**Leak-free by construction, not by measurement.** Class counts come from the train split only; the
offset is a length-7 constant applied inside the criterion; `shared_evaluate` takes no criterion, so
evaluation logits are unadjusted. τ=0 reduces to plain CE.

## ⚠ Logit adjustment must stay OFF for the region head — measured, not assumed

Alabama dedicated region, 5 folds, τ=0 vs τ=0.5:

| metric | τ=0 | τ=0.5 | Δ | p |
|---|---:|---:|---:|---:|
| **Acc@10 — the reported metric** | **69.9956** | **68.1550** | **−1.841** | **0.0002** |
| Acc@5 | 58.6954 | 56.6129 | −2.083 | 0.0000 |
| Acc@1 | 31.1203 | 29.8787 | −1.242 | 0.0023 |
| MRR | 44.0651 | 42.4748 | −1.590 | 0.0003 |
| macro-F1 | 7.7452 | 8.1219 | **+0.377** | 0.0008 |

Exactly the predicted split, and it confirms the mechanism rather than merely rejecting the option:
logit adjustment is **Bayes-consistent for balanced error**, so macro-F1 *improves significantly* —
while every frequency-weighted metric (Acc@1/5/10, MRR) degrades significantly, because for those
the Bayes-optimal predictor is the **unadjusted** posterior. Region is reported by Acc@10, so it
must stay off. (`mtl_cv.py:477-479` already turns class-balancing off for this head for the same
reason.) It is on the MTL **category** criterion only — `mtl_cv.py:504` — so the joint model is
already correct; no code change needed.

## Family (a) — dedicated next-category

| axis | small (AL / AZ / IST) | large (FL / CA / TX) | grade |
|---|---|---|---|
| batch size | **8192** | **8192** | [5f] AL; [1f] TX (16k −0.111, 32k −0.552, **no wall saving**: 1116/1119/1117 s) |
| max_lr | AL **0.0025**, AZ **0.0005**, IST **0.0005** | **0.005** | [5f] small; [1f] TX flat 0.005–0.01 |
| epochs | **50** | **50** | [5f] / [1f] |
| class weighting | **off — superseded** | **off — superseded** | see above |
| **logit-adjust τ** | **0.5** | **0.5** | [5f] small, [1f] TX |

## Family (b) — dedicated next-region

**Never swept.** This study tuned the category recipe only; region numbers elsewhere are a read-out,
not a sweep. Everything carries from v17 unchanged.

| axis | all states | grade |
|---|---|---|
| batch size / max_lr / epochs | **v17 defaults** (max_lr 3e-3, 50 ep) | [—] not swept |
| class weighting | **off** (v17 default) | [—] pre-v18 |
| **logit-adjust τ** | **0 — OFF** | [5f] AL: −1.841 Acc@10, p=0.0002 |

## Family (c) — joint MTL

| axis | small (AL / AZ / IST) | large (FL / CA / TX) | grade |
|---|---|---|---|
| batch size | **8192** | **8192** | [5f] AL (all 6 larger arms below, 4 significant); [1f] FL **null** — span 0.078 geom |
| cat-lr | **0.001** | **0.002** ← author's call | [5f] small: null, span 0.21–0.28; [1f] FL: +0.029 geom |
| reg-lr / shared-lr | 3e-3 / 1e-3 | 3e-3 / 1e-3 | [—] unchanged |
| epochs | **50** | **50** | [5f] |
| category_weight | **0.50** ← author's call | **0.50** | [5f] geom: AL **+0.320 p=0.031**, AZ −0.003, IST −0.288 |
| cat class weighting | **off — superseded** | **off — superseded** | see above |
| **logit-adjust τ** (cat head only) | **0.5** | **0.5** | [5f] small, [1f] FL |
| reg class weighting | **off** | **off** | [—] v17 default |
| shared trunk | **keep** | **keep** | inert at AL [5f]; CA/TX only a 1-fold screen |

### Two rows are author preference rather than sweep evidence — both APPROVED 2026-08-09

Recorded so the provenance is not lost, since neither follows from a significance test.

1. **cat-lr 0.002 at large states — APPROVED.** The author's justification is correct: at florida
   0.002 has the **best geom of the whole row-3b grid**, 53.0588 vs 53.0301 (1e-3), 53.0299
   (1e-3 @ 25 ep) and 52.8571 (5e-4). The caveats stand — the margin is **+0.029 on one fold**, the
   axis is null at 5 folds at AL/AZ/IST (span 0.21–0.28 across a 4× range), and the 0.002 arm's
   *region* is 0.017 *below* the 1e-3 arm. Harmless on a flat axis; adopted on the author's call.
   ⚠ Note that row 3b was measured under **class weighting**; cat-lr has not been re-screened under
   logit adjustment. Step D pinned 0.002 in all three arms, so it does not re-test the axis.
2. **category_weight 0.50 — APPROVED.** Significantly better on geom at alabama (+0.320, p=0.031),
   a tie at arizona (−0.003), worse at istanbul (−0.288, ns). Never significantly worse. Pooled it
   is ≈ +0.01, so it redistributes ~0.15 pp from region into category rather than adding. This
   **overrides pre-registered decision rule 3** (ties break on region → 0.75); recorded as a
   deliberate geom-based override, since geom_simple is the shipped selector.

### Batch size: what was actually measured, since this was contested

- **texas [1f]** — *dedicated* model, one head, **no region output ⇒ no geom exists**. On the only
  metric it has: 8192 → 16384 → 32768 = 35.1696 → 35.0585 → 34.6175, and walls are identical
  (1116/1119/1117 s). The **argmax** column inverts this (+0.23 for 16384), which is precisely the
  spike-reading artifact `sm3` was adopted to remove — the argmax−sm3 gap grows 0.000 → 0.341 →
  0.538 with batch size.
- **alabama MTL [5f]** — geom 46.0180 at bs8192 beats all six larger-batch arms (−0.161 … −0.387),
  four of them significantly (p = 0.048 / 0.015 / 0.003 / 0.026).
- **florida MTL [1f], step D, run 2026-08-09** — the only large-state MTL batch measurement:

  | bs | cat sm3 | reg | geom (sm3) | Δ | geom (argmax) | Δ | wall |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | **8192** | 37.8597 | 77.4003 | **54.1327** | — | 54.2348 | — | 1505 s |
  | 16384 | 37.8342 | 77.4215 | 54.1219 | −0.011 | 54.2376 | +0.003 | 1456 s |
  | 32768 | 37.9049 | 77.5011 | 54.2003 | +0.068 | 54.4021 | +0.167 | 1446 s |

  **bs16384 is a dead tie on both selectors**, so that specific arm has no case here. bs32768 leads
  nominally (+0.068 / +0.167) but on one fold, and adopting it would tier batch size against 5-fold
  alabama evidence where bs32768 was significantly *worst* (−0.313 … −0.387, p = 0.003–0.026).

So bs8192 is kept everywhere: it wins where the effect is resolvable (AL) and ties where it is not
(FL), and the larger batches buy no meaningful wall time at either state.

## What this costs to regenerate

Every completed cell used class-weighted CE, so **all of them are superseded** — which matches the
author's decision to regenerate everything for a fair run. Nothing needs to be argued cell-by-cell.

## Still open — not blocking approval

- **CA never measured for logit adjustment** (arms dropped by author request). FL and TX both
  confirm transfer, so CA is expected to follow, but it is an assumption, not a measurement.
- **All sweep evidence is seed 0.** The regeneration itself supplies seeds {0,1,7,100}.
- **Trunk contribution at CA/TX unmeasured** (P4, ~22 h) — keep the trunk; do not cite "inert" as a
  general claim.
- **P1 capacity-matched region control** still held: it decides whether the CA/TX region advantage is
  multi-task sharing or just parameters.
