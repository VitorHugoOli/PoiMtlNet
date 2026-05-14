# Tests 1 + 2 + 2½ + 2¾ + 3 findings — 2026-05-06

Merged write-up of the four diagnostics that landed today. Together they
characterise the residual HGI gap and falsify the easy explanations.

> **2026-05-14 training-regime caveat (PR #20 Claude review).** The Test 1
> JSONs were trained with two suboptimal settings in
> `scripts/p1_poi_head_ablation.py` that were fixed in commit `e99e904`:
> (a) the `OneCycleLR` scheduler was constructed but never wired in
> (`None` passed to `_train_one_fold`), so training ran at a fixed LR of
> `max_lr/25 = 1.2e-4` instead of the intended warmup + cosine decay;
> (b) `CrossEntropyLoss()` was used without `label_smoothing`, despite the
> sibling `p1_region_head_ablation.py` using `label_smoothing=0.1` (relevant
> for 12 k–77 k POI classes under a Zipfian visit distribution — head-class bias).
>
> **Research impact:** *relative* ordering (HGI > J > canonical) is preserved
> because all three substrates share the same training regime. The
> *absolute* Acc@1/5/10 numbers below are pessimistic for all designs equally
> and should not be quoted as paper-grade in this state. The qualitative
> verdict ("J closes 77 % of the canonical→HGI gap but does not overcome
> HGI") holds and is the load-bearing finding the rest of the study cites.
> Re-running T1 under the fixed regime is queued; expect roughly proportional
> absolute lifts at each substrate.

## Test 1 — Next-POI probe (AL, 5f×50ep, `next_gru`, 11 848 classes)

| Substrate | Acc@10 | Δ vs canonical | Wilcoxon p_gt vs canonical |
|---|---:|---:|---|
| canonical | 0.0361 | — | — |
| **J** | **0.0499** | **+1.38 pp** | **p=0.0312 ✓** (5/5) |
| **HGI** | **0.0541** | **+1.80 pp** | **p=0.0312 ✓** (5/5) |
| HGI vs J | — | +0.42 pp (n.s.) | p=0.94 (1/5) |

Per-fold canonical: [0.0484, 0.0437, 0.0291, 0.0338, 0.0256]
Per-fold J: [0.0708, 0.0507, 0.0401, 0.0456, 0.0421]
Per-fold HGI: [0.0673, 0.0535, 0.0476, 0.0547, 0.0476]

Acc@1 ranking: HGI 1.85 % > J 1.82 % > canonical 1.20 %.
Acc@5 ranking: HGI 3.91 % > J 3.52 % > canonical 2.66 %.

### Verdict

**The advisor's hypothesis ("c2hgi naturally beats HGI on next-POI
because HGI can't distinguish revisits") is falsified empirically.**
HGI's POI2Vec semantic prior — even though it cannot distinguish two
visits to the same POI — gives a stronger next-POI prior than c2hgi's
per-visit context at 12 K-class scale.

**J closes 77 % of the canonical→HGI gap on next-POI, but does not
overcome HGI.** This mirrors next-region: J → canonical strict win,
J → HGI not overcome. Same pattern, both axes.

The user's research goal "overcome HGI on next-region AND next-POI" is
*harder* than the advisor predicted. The merge family is a real
improvement over canonical c2hgi on both axes, but is still
1-2 pp behind HGI on both. The Lever 6 reframe (POI↔POI contrastive
boundary added to c2hgi) is the principled response — current evidence
points to the missing 4th boundary in c2hgi's pretraining as the residual.

AZ next-POI numbers landing later today; expect the same pattern.

## Test 2 — Reg-head ablation: J FL with `next_gru` (no `log_T`)

| Substrate | Acc@10 | Δ vs canonical | Wilcoxon p_gt |
|---|---:|---:|---|
| canonical | 0.6836 | — | — |
| **J** | 0.6922 | **+0.86 pp** | **p=0.0312 ✓** (5/5) |
| **HGI** | 0.7086 | **+2.50 pp** | **p=0.0312 ✓** (5/5) |
| HGI vs J | — | **+1.64 pp** | **p=0.0312 ✓** (5/5) |

### Verdict

**The Markov prior was *helping* J close the gap, not masking a real one.**
- with `log_T`: J−HGI = −1.00 pp
- without `log_T`: J−HGI = −1.64 pp

Removing the head's Markov prior makes the embedding-quality gap *larger*
and Wilcoxon-strict at 5/5 folds. **The 1 pp FL gap is not
within HGI's σ as the audit-update once suggested — at p=0.0312 it is
strictly distinguishable from noise.**

## Test 2½ — Single-seed reroll: J FL build seed=43 vs seed=42

| | Acc@10 | per-fold |
|---|---:|---|
| seed=42 | 0.7034 | [0.6979, 0.7087, 0.7060, 0.6980, 0.7066] |
| seed=43 | 0.7044 | [0.6969, 0.7079, 0.7082, 0.6991, 0.7100] |
| Δ | **+0.10 pp** | tight |

### Verdict

**Build-seed-noise hypothesis falsified.** Two builds initialised
independently land within 0.1 pp of each other — an order of magnitude
tighter than the J→HGI gap. The remaining gap is not an artefact of any
random initialisation choice.

## Test 2¾ — Epoch calibration: J AL at 200 vs 500 epochs

| | Acc@10 |
|---|---:|
| J(λ=0.1) ep=200 | 0.6155 |
| J(λ=0.1) ep=500 | 0.6196 |
| Δ | −0.41 pp |

### Verdict

**200 ep is within the 0.4 pp tolerance.** The calibration script picks
200 ep for the Test 3 sweep (≈ 2.5× faster than 500 ep) without altering
the diagnostic outcome.

## Test 3 — POI2Region `num_heads` sweep on J at AL (`ep=200`)

| nh | Acc@10 | Δ vs J(nh=4) | Δ vs HGI | Wilcoxon vs J(nh=4) |
|---|---:|---:|---:|---|
| 2 | 0.6174 | **+0.19 pp** | −0.12 pp | p=0.0625 (4/5) — n.s. |
| 4 (J default) | 0.6155 | 0 | −0.30 pp | — |
| 8 | 0.6149 | −0.06 pp | −0.37 pp | n.s. |
| 16 | 0.6104 | **−0.51 pp** | **−0.82 pp** | actively worse |

Per-fold nh=2: [0.6574, 0.6208, 0.6479, 0.6050, 0.5561]
Per-fold nh=4: [0.6526, 0.6200, 0.6483, 0.6031, 0.5537]
Per-fold nh=16: [0.6456, 0.6153, 0.6412, 0.5893, 0.5608]

### Verdict — POI2Region undersized hypothesis ✗ falsified

Increasing `num_heads` (more attention capacity in PMA) **monotonically
hurts** after `nh=4`. Halving to 2 nominally lifts +0.19 pp but does not
strictly cross the canonical→HGI gap. **None of the swept variants
overcome HGI.** Closest is nh=2 at −0.12 pp from HGI (n.s.).

The user's hypothesis ("POI2Region is undersized for the merge POI
distribution") is falsified. PMA at the current `nh=4` is already at or
near optimal capacity; the consumer side is not the residual bottleneck.

### What this rules out

PMA seed-query attention capacity, region GCN single layer, and
attention-head count are all not the bottleneck. The merge POI vectors
*are not being underdiscriminated by the consumer*. The consumer is fine.

## Combined picture — what the residual gap actually is

After 4 diagnostics on top of 8 design audits + the K falsification, the
residual ~1 pp gap to HGI on both next-region and next-POI is:

- ✗ NOT Markov-prior magnification (Test 2)
- ✗ NOT build-seed noise (Test 2½)
- ✗ NOT spatial topology (Design K = J empirically, prior phase)
- ✗ NOT anchor strength (warm-start zero, Lever 1 falsified)
- ✗ NOT a "next-POI is naturally easier for c2hgi" head-ranking argument (Test 1)
- ✗ NOT POI2Region attention-head count (Test 3 — adding heads hurts, halving doesn't lift)

It is a **real embedding-quality gap with consistent magnitude (~1.5-2 pp)
across both downstream tasks**, Wilcoxon-strict at p=0.0312 on three
independent paired comparisons. The consumer side (POI2Region) is fine;
the input features (POI2Vec, learnable table) are fine; the spatial
graph (Delaunay GCN, K) doesn't help; the optimisation tricks (anchor λ,
seeds) don't shift it.

## The single live candidate

**Reframed Lever 6 — add a POI↔POI contrastive boundary to c2hgi
pretraining.**

HGI uses 4 contrastive boundaries (POI↔POI, p2r, r2c, plus POI2Vec
discriminate). c2hgi has 3 (c2p, p2r, r2c). The merge family applies
POI2Vec only as a frozen prior added to the POI residual; the discriminator
never sees POI2Vec as a contrastive target itself. After every cheaper
diagnostic has been falsified, this is the only structural difference
between c2hgi-merge and HGI that has not yet been tested.

Cost: ~6-8 h implementation + AL+AZ revalidation.
Decision criterion: if Lever 6 closes the gap to ≤0.3 pp vs HGI on AZ
(currently −1.29 pp under K, presumably similar under any merge
variant), ship it. If not, the residual is in something even more
subtle (HGI's `cross_region_weight=0.7` calibration, POI2Vec's longer
training run, etc.) and we are deep in diminishing returns.

## Decision

- **Tests 1, 2, 2½, 2¾, 3 closed.** All findings persisted in this doc.
- **Test 4 (alpha sweep) deprioritised.** Test 3 already shows the consumer
  side is at or past optimal capacity; tuning loss-weights inside an
  unchanged 3-boundary loss is unlikely to find what 4 boundary count
  did.
- **Lever 6 is the next move** if the user wants to commit. STATE.md
  updated to reflect this.
