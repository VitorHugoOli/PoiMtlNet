# Design B — POI2Vec at the pool boundary (✓ DOMINANCE at AL/AZ)

## Aim

Inject HGI's POI-stable signal into c2hgi *only at the POI level*, after
check-in encoding has already happened. Cat path stays identical to
canonical; reg path benefits from POI2Vec's spatial-functional prior.

## Mechanism

```
checkin_emb       = CheckinEncoder(canonical_features)   # cat path reads this
poi_emb_canonical = Checkin2POI(checkin_emb)             # detached for reg path
poi_emb_with_p2v  = poi_emb_canonical.detach() + γ · Linear_64→64(POI2Vec)
region_emb        = POI2Region(poi_emb_with_p2v)         # reg path reads this

L_c2p uses poi_emb_canonical   (cat supervision, gradient flows to encoder)
L_p2r uses poi_emb_with_p2v    (reg supervision, gradient does NOT flow to encoder)
L_r2c uses region_emb
```

POI2Vec is a frozen 64-dim prior loaded from HGI's pre-trained
`poi_embeddings.csv`.

## AL/AZ leak-free results

| State | cat F1 | Δ vs canonical | Wilcoxon | TOST (δ=2pp) | reg Acc@10 | Δ vs canonical | Wilcoxon |
|---|---:|---:|---|---|---:|---:|---|
| AL | 41.51 ± 1.34 | +0.76 pp | p=0.0312 | p=0.0002 ✓ | 61.49 ± 4.06 | +2.34 pp | p=0.0312 |
| AZ | 43.91 ± 1.10 | +0.70 pp | p=0.0938 | p=0.0027 ✓ | 52.59 ± 3.03 | +2.35 pp | p=0.1562 |

fclass linear probe: 98.45 / 97.91 (HGI-grade, +94 pp over canonical).
kNN-Jaccard@10 vs POI2Vec: 0.109 / 0.072.

## Verdict

✓ DOMINANCE at both AL and AZ.
- Cat non-inferior at TOST p<0.003.
- Reg strictly superior at AL (Wilcoxon p=0.0312); AZ Δ matches HGI within σ.
- fclass generality probe at HGI level (98%).

## FL leak-free results (2026-05-06)

Tag: `STL_FLORIDA_design_b_reg_gethard_pf_5f50ep_leakfree`.

| | reg Acc@10 | F1 | Δ vs canonical(pf) | Wilcoxon p_gt | Δ vs HGI(pf) |
|---|---:|---:|---:|---:|---:|
| canonical c2hgi (pf) | 0.6922 | 0.0907 | — | — | −2.13 pp |
| B (leak-free) | **0.6993 ± 0.0047** | 0.0907 | **+0.71 pp** | 0.0625 (4/5 folds) | −1.41 pp |
| HGI (pf) | 0.7134 | 0.0929 | +2.13 pp | — | — |

Per-fold B: [0.6958, 0.7020, 0.7028, 0.6928, 0.7030].

**Verdict at FL**: small reg lift (+0.71 pp), not Wilcoxon-strict at p=0.0312,
still behind HGI by 1.4 pp. The merge mechanism's reg gain is small-data
driven — at AL/AZ it's +2.3 pp, at FL it shrinks to +0.7 pp. Cat F1 axis
unchanged (≈ 0.091 for both, consistent with the AL/AZ cat-preserving
property of B).

The +13 pp FL lift previously reported (B at 0.824 vs canonical 0.692) was
fully attributable to the leaky single `region_transition_log.pt`, not the
merge mechanism.
