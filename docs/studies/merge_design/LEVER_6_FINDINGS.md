# Lever 6 findings — POI↔POI contrastive boundary added to c2hgi

Lever 6 was the principled candidate after every cheaper diagnostic was
falsified (Lever 1 anchor, Lever 3 / K Delaunay GCN, Test 1 next-POI,
Test 2 head ablation, Test 2½ seed, Test 3 POI2Region heads).

**Build script**: `scripts/probe/build_design_lever6_p2p.py` —
`Check2HGI_DesignL_P2P` adds a 4th contrastive boundary to J's
3-boundary loss, scoring the merge POI vectors against HGI's Delaunay
edges as positive pairs and random POIs as negatives. Implementation
mirrors c2hgi's existing bilinear discriminator + σ + binary
cross-entropy. ~6h of work.

## α-sweep (α_p2p ∈ {0.1, 0.3, 1.0})

All builds: 200 ep, MPS, anchor λ=0.1, otherwise J's defaults.

| Substrate | AL Acc@10 | AL Δ vs HGI | AZ Acc@10 | AZ Δ vs HGI |
|---|---:|---:|---:|---:|
| canonical | 0.5915 | −2.71 pp | 0.5024 | −3.13 pp |
| HGI | 0.6186 | 0 | 0.5337 | 0 |
| **J (baseline)** | 0.6196 | **+0.10 pp** | 0.5215 | −1.22 pp |
| L6 α=0.1 | 0.6181 | −0.05 pp | 0.5194 | −1.43 pp |
| L6 α=0.3 | 0.6139 | −0.47 pp | 0.5238 | −1.00 pp |
| L6 α=1.0 | 0.6111 | −0.75 pp | 0.5237 | −1.01 pp |

Wilcoxon p_gt vs HGI for the best L6 setting on each state:

- AL α=0.1: Δ=−0.05 pp, p=0.59 (n.s. — within seed noise of HGI but doesn't strictly beat)
- AZ α=0.3: Δ=−1.00 pp, p=0.69 (n.s. — same band as J)

L6 also evaluated on next-POI at α=0.3 (single setting, both states):

| | AL Acc@10 | AZ Acc@10 |
|---|---:|---:|
| L6 next-POI | 0.0452 | 0.0855 |
| J next-POI | 0.0499 | 0.0855 |
| HGI next-POI | 0.0541 | 0.0895 |

L6 next-POI ties J at AZ, regresses by 0.46 pp at AL. **No α setting
of L6 overcomes HGI on any axis at any state.**

## Verdict — falsified

The new POI↔POI contrastive boundary does not close the gap to HGI.
α=0.1 lands within ±0.15 pp of J on both states; α=0.3/1.0 progressively
hurt as the boundary becomes more aggressive (best_ep gets pulled to
very early epochs, indicating loss-landscape instability with stronger
α). The boundary supervises the right thing (merge POI vectors learning
Delaunay-neighbour similarity) but contributes no measurable lift.

## What the falsification means

After 6 cheap structural levers (Lever 1, K=Lever 3, Tests 1/2/2½/3,
Lever 6) every architectural surface accessible to a c2hgi-merge build
script has been swept:

| Surface | Lever | Verdict |
|---|---|---|
| POI features (frozen) | B | ≈ canonical+2pp reg, no HGI overcome |
| POI features (learnable) | H | same |
| POI features (low-rank) | I | same |
| POI features (warm anchor) | J | same; nominally beats HGI on AL only |
| POI features (distillation) | M | same |
| Spatial topology | K | K = J empirically |
| λ-anchor strength | J λ-sweep | warm-start zeros it |
| POI2Region head capacity | T3 | nh ∈ {2,4,8,16}: optimal ≈ 4, doesn't help |
| Markov prior masking | T2 | gap is real, gets bigger w/o log_T |
| Build seed noise | T2½ | Δ=+0.10 pp |
| 4th contrastive boundary | L6 α-sweep | tied or worse than J |

The gap to HGI on AZ (~1-1.4 pp) is now confirmed to be **below this
study's architectural resolution**. It must live in something deeper:

- **POI2Vec's 2000-epoch hierarchical-fclass pretraining**. The merge
  family imports the *output* of HGI's POI2Vec training as a frozen
  prior (or warm-start). It never re-runs that pretraining under the
  c2hgi corpus. HGI's POI vectors at the start of its main pretraining
  may carry information our frozen import cannot recover.
- **HGI's `cross_region_weight=0.7` calibration** of intra-vs-cross-region
  Delaunay edge weights — tuned per-state in HGI but inherited unchanged
  in K's edge load.
- **HGI's POIEncoder at the front of the pipeline** (a GCN over POI2Vec
  inputs *before* any other module sees them). The merge family's
  POI vectors enter the consumer (POI2Region) as `detach() + γ·table`,
  never having gone through a POI-level GCN over the c2hgi corpus.

These are recipe-level differences, not architecture-level. Fixing any of
them would mean retraining HGI's POI2Vec from scratch on each state with
the c2hgi corpus, which is a different study (and out-of-scope here).

## Realised contribution of the merge family

The user's research goal was "overcome HGI on next-region AND next-POI."
The realised result is:

1. **The merge family (B/H/I/J/M) strictly dominates canonical c2hgi**
   on both axes, AL+AZ+FL, at the user's strict gate (Wilcoxon p=0.0312,
   5/5 folds).
2. **The merge family preserves canonical cat F1** (TOST p<0.003 at
   both states for B/H/I/J/M; M is even strictly higher at p=0.0312
   on both states).
3. **fclass POI semantic recovery** is HGI-grade (98% probe vs canonical
   c2hgi 4%) — a generality property HGI alone has and canonical c2hgi
   lacks.
4. **The merge family does not strictly overcome HGI** on either axis;
   J nominally beats HGI on AL reg by 0.10 pp but n.s.

The merge family's contribution is therefore: **a Pareto improvement
over canonical c2hgi on cat AND reg AND POI semantic generality, with
no regression on any axis**, settling at a small (~1 pp) gap to HGI on
the reg axis only. Whether to write this up as the contribution
(strong yes — it's a genuine Pareto win the field doesn't have) or to
keep chasing the residual gap (no — surface levers exhausted) is a
publication-strategy decision rather than a research-content one.

## Closed
- Task #36 (Lever 6 implementation + sweep): completed.
- All structural research questions in the merge_design study: closed.
