# Re-screen of dropped candidates — L0/L1 (5-fold, isolated placeid-folds)

All variants rebuilt via `rescreen_build.sh` (OUTPUT_DIR-scratch; frozen
`output/check2hgi/` md5-verified untouched). Compared to **`check2hgi_gcn_ctrl`**
(fresh GCN wd=0, same-protocol control). Frozen `check2hgi` shown for reference —
it sits ~+0.5pp above the fresh control (epoch/code/seed offset), which is exactly
why a same-protocol control was needed.

## next-cat — L1 probe acc (mean±SD, 5 folds)
| engine | alabama | arizona |
|---|---|---|
| check2hgi (frozen ref) | 0.9853±0.0037 | 0.9833±0.0019 |
| **check2hgi_gcn_ctrl** | 0.9804±0.0033 | 0.9802±0.0019 |
| check2hgi_v3c_wd05 | 0.9800±0.0034 | 0.9802±0.0019 |
| check2hgi_t24_dropedge | 0.9797±0.0035 | 0.9798±0.0016 |
| check2hgi_t43_sidefeat | 0.9803±0.0029 | 0.9800±0.0020 |
| check2hgi_t61_p2p | 0.9800±0.0041 | 0.9801±0.0021 |
| check2hgi_gat | 0.9296±0.0033 | 0.9459±0.0036 |
| check2hgi_rgcn | 0.9579±0.0031 | 0.9828±0.0014 |

## next-reg (region embeddings) — adj_coh@10 (L0) / probe acc@10 (L1)
| engine | AL adj_coh | AL probe@10 | AZ adj_coh | AZ probe@10 |
|---|---|---|---|---|
| **check2hgi_gcn_ctrl** | 0.2172±0.014 | 0.5472±0.009 | 0.2635±0.006 | 0.4859±0.005 |
| check2hgi_v3c_wd05 | 0.2063±0.015 | 0.5476±0.013 | 0.2641±0.005 | 0.4856±0.004 |
| check2hgi_t24_dropedge | 0.2156±0.014 | 0.5473±0.008 | 0.2498±0.007 | 0.4795±0.004 |
| **check2hgi_t43_sidefeat** | **0.2909±0.011** | **0.5635±0.003** | **0.3202±0.007** | **0.4946±0.003** |
| check2hgi_t61_p2p | 0.2235±0.014 | 0.5454±0.013 | 0.2356±0.007 | 0.4843±0.005 |
| check2hgi_gat | 0.2479±0.011 | 0.5582±0.007 | 0.2634±0.005 | 0.4866±0.004 |
| **check2hgi_rgcn** | **0.3090±0.013** | **0.5770±0.008** | **0.3318±0.007** | **0.4929±0.005** |

## L2 next-reg confirmation (STL next_stan_flow, region input, 5-fold)
**Alabama Acc@10** (±SD over folds): gcn_ctrl 0.5956±0.041 · sidefeat **0.6146±0.043 (+1.9pp)** · rgcn **0.6299±0.040 (+3.4pp)**. Same ordering as L0/L1 (rgcn > sidefeat > ctrl) — direction concordant L0→L2. BUT AL fold-SD ≈4pp ⇒ gaps not yet statistically separated; FL (≈10× tighter SD) is the conclusive test (running).

## v3c at Florida (L0/L1) — falsified at all 3 states
FL next-cat L1: gcn_ctrl 0.9827 vs v3c 0.9826 (identical). FL next-reg adj_coh: ctrl 0.2093 vs v3c 0.2069 (v3c below); probe@10 0.6806 vs 0.6794. ⇒ **v3c shows no gain vs control on either axis at AL, AZ, AND FL.** Definitive.

## L2 next-reg at FLORIDA — CONCLUSIVE (tight SD, 159k pairs)
| engine | Acc@1 | Acc@10 |
|---|---|---|
| gcn_ctrl | 0.4689 | 0.7249±0.007 |
| sidefeat | 0.4689 | 0.7279±0.005 (+0.30pp) |
| rgcn | 0.4692 | 0.7316±0.006 (+0.67pp) |

At FL the AL gains **collapse into the noise band** (sidefeat +0.30pp, rgcn +0.67pp Acc@10, both within ~1 SD; Acc@1 identical to control). The AL L2 appearance (+1.9/+3.4pp) was small-N variance. **So no re-screened candidate delivers a robust L2 region win at scale.** The L0 adjacency-coherence signal (sidefeat/rgcn) did not convert to sequence-task accuracy at high N — a clean repeat of "L0 structural metric over-promises; L2 at high-N is the ranking authority."

## R-GCN leak audit (2026-06-01, dedicated auditor agent) — CONFIRMED leak
R-GCN's FL next-cat probe 0.9986 **is a real structural leak**, not substrate quality:
- **Mechanism:** `--edge-type both` adds a `same_poi` relation; R-GCN's per-relation `W_same_poi` learns a near-identity that copies the **verbatim category one-hot** (a node *input* feature; category is 100% POI-constant) across same-POI check-ins → a direct category-copy channel. Encoder-internal (reproduces with category-weight=0). Matches the documented T3.3 falsification (leak-discriminator F1 +27.85 at K=2; K=1 collapses cat −10.9 — line CLOSED).
- **Evidence:** survives held-out-POI (0.994 on unseen POIs) and control-task is at chance (0.165) and no same-POI collapse → it's category-copy (label injected as input + propagated), not POI-identity memorization. Embeddings scale-amplified ~36× (std 13.9 vs 0.39), inter-category centroid dist 85 vs 2.7. The +11pp over control is leak amplification.
- **Does it contaminate next-reg? NO.** Region embeddings are not scale-inflated (category-copy washed out by POI→Region pooling). Decisive held-out-region test (all test regions unseen): rgcn acc@10 0.5256 vs ctrl 0.4945 = **+3.1pp** (top-1 tied) — a genuine but small spatial-structure gain, not leak. (At the standard L2 next_stan_flow it was +0.67pp, within SD.)
- **Verdict: DISQUALIFY R-GCN on next-cat** (leak); its region side is clean-but-small and would re-import the cat leak if shipped.
- **Harness note (auditor):** the L1 next-cat probe's placeid-isolation makes folds *engine-identical* but does not hold out POIs at check-in granularity — add a GroupKFold-by-placeid probe variant to make any category-copying substrate's leak visible/penalized. (At POI-pooled granularity the leak is embedding-internal and shows regardless.)

## FINAL verdict (full ladder L0→L2, multi-state, controlled)
**None of the 5 re-screened dropped candidates resurrects as a robust improvement.**
- v3c (WD 5e-2): falsified — no gain vs control, 3 states, both axes.
- T4.3 sidefeat / T3.3 R-GCN: a real L0 adjacency-coherence difference + AL L2 blip, but **not separable from control at FL L2** (the high-N test). Region axis not robustly improved.
- T2.4 DropEdge, T6.1 p2p, GATv2: no gain (GAT hurts cat).
The systematic ladder confirms the original falsifications — now on the correct artifact (region embeddings), with a same-protocol control, isolated folds, and the high-N L2 check. Value: we now *know* (controlled, multi-level) rather than assume. No candidate justifies an MTL trial.
- **v3c (WD 5e-2): FALSIFIED on the region axis.** No gain over the clean control on cat OR reg at either state (reg adj_coh slightly below ctrl). The "region benefit" was an artifact of comparing to a differently-built baseline. ✗
- **T4.3 POI side-features: consistent small REGION gain** (adj_coh +0.05–0.07, probe@10 +0.9–1.6pp at both states; cat neutral). The original eval falsified it on a single-state *category* cell — but on the *region* axis it shows a real, two-state-consistent signal. → L2 confirmation warranted.
- **T3.3 R-GCN: biggest REGION gain** (adj_coh +0.07–0.09, probe@10 +0.7–3pp), without the catastrophic cat-leak in this static probe (cat 0.96–0.98). The relation-typed graph plausibly encodes region structure — but needs a leak-check (relation edges could leak adjacency) + L2. → L2 + leak audit.
- DropEdge, T6.1 p2p: no gain (≈ctrl). GATv2: cat hurt (no leak here, just worse). ✗
- **Methodology win:** the ladder on the *correct artifact* (region embeddings) with a *same-protocol control* found a region-axis signal (sidefeat, rgcn) that a single-MTL-metric eval missed — and killed v3c cleanly. These are L0/L1 proxies; **L2/L3 is the ranking authority** (see nextreg_stl.md protocol).
