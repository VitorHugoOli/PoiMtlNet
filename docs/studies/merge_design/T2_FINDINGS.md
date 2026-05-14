# Tests 2 + 2½ findings — 2026-05-06

Both diagnostics ran on Florida, leak-free protocol. Both **falsify** the
"residual gap is artefact" hypotheses.

## Test 2 — Reg-head ablation: J FL with `next_gru` (no log_T)

5-fold leak-free, no Markov prior added by the head:

| Substrate | Acc@10 | Δ vs canonical | Wilcoxon p_gt vs canonical |
|---|---:|---:|---|
| canonical | 0.6836 | — | — |
| **J** | **0.6922** | **+0.86 pp** | **p=0.0312 ✓** (5/5) |
| **HGI** | **0.7086** | **+2.50 pp** | **p=0.0312 ✓** (5/5) |
| HGI vs J | — | **+1.64 pp** | **p=0.0312 ✓** (5/5) |

Per-fold canonical: [0.6784, 0.6904, 0.6863, 0.6766, 0.6861]
Per-fold J: [0.6887, 0.6953, 0.6977, 0.6835, 0.6958]
Per-fold HGI: [0.7060, 0.7100, 0.7122, 0.7033, 0.7117]

### Verdict

**The Markov prior was *helping* J close the gap, not masking a real one.**
With log_T (`next_getnext_hard`), J–HGI = −1.00 pp. Without log_T
(`next_gru`), J–HGI = −1.64 pp. The embedding-quality gap to HGI is real,
structural, and **slightly larger** than the head-with-log_T showed.

The previous "1 pp at FL within HGI's σ" reading is wrong: at p=0.0312
across all three pairwise comparisons (5/5 folds same-sign), the gap is
strictly distinguishable from noise.

## Test 2½ — Single-seed reroll: J FL seed=43 vs seed=42

Build seed varied (43 vs 42); reg eval seed held at 42 for paired comparison.

| | Acc@10 | per-fold |
|---|---:|---|
| J seed=42 | 0.7034 | [0.6979, 0.7087, 0.7060, 0.6980, 0.7066] |
| J seed=43 | 0.7044 | [0.6969, 0.7079, 0.7082, 0.6991, 0.7100] |
| Δ | +0.10 pp | tight |

### Verdict

**Seed-noise hypothesis falsified.** The 1 pp gap to HGI is not within
the build-seed wobble. Two builds initialised independently land within
0.1 pp of each other, an order of magnitude tighter than the J→HGI gap.

## Combined implication

The residual gap to HGI on next-region at FL:
- is **not** Markov-prior magnification (Test 2)
- is **not** build-seed noise (Test 2½)
- is **not** spatial topology (K = J, prior phase)
- is **not** anchor strength (warm-start zero, prior phase)

It is a real **embedding-quality** difference. Tests 3 (POI2Region
hyperparams + PMA attention entropy) and 4 (boundary-weight retuning)
are now fully justified — the user's hypothesis that "maybe POI2Region
itself is undersized for the merge POI vector distribution" is the
live candidate.

## What to do next

- Run Test 2¾ (200 vs 500 ep calibration on J at AL) — currently in
  progress. Decides epoch count for the sweeps below.
- Run Test 3 — `num_heads ∈ {2, 4, 8, 16}` on J's `--attention-head`
  flag (which controls both Checkin2POI and POI2Region attention heads)
  at AL with the calibrated epoch count.
- Add a region-GCN-layer flag to POI2Region (currently hardcoded
  single-layer at `RegionEncoder.py:35`) and sweep `layers ∈ {1, 2}`.
- Log per-region PMA attention entropy after the `pyg_softmax` call at
  `RegionEncoder.py:134` — direct evidence of whether the seed query is
  collapsing onto a small set of POIs.
- Run Test 4 (alpha sweep) only if Test 3 yields no lift on any axis.
- Keep Test 1 (next-POI probe) running in parallel — it may resolve
  the next-POI half of the research question independently.
