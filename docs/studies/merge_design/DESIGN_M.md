# Design M — B + POI distillation toward POI2Vec (✓ DOMINANCE, cat strict 5/5 both states)

## Aim

Strengthen B's POI semantic recovery by adding an explicit distillation
loss from the post-pool POI embeddings to POI2Vec, on top of B's
architecture. Tests whether a soft auxiliary loss yields a tighter
generality / cat outcome than B's residual alone.

## Mechanism

```
poi_emb_with_p2v = poi_emb_canonical.detach() + γ · Linear_64→64(POI2Vec)
                                                      # B's residual

L_total = L_c2hgi + λ_d · cosine(poi_emb_with_p2v, POI2Vec)   # λ_d = 0.1
```

The distillation loss applies *after* the .detach() so the encoder is
unchanged; only the post-pool projection is supervised toward POI2Vec.

## AL/AZ leak-free results

| State | cat F1 | Δ vs canonical | Wilcoxon | TOST | reg Acc@10 | Δ vs canonical | Wilcoxon |
|---|---:|---:|---|---|---:|---:|---|
| AL | 41.31 ± 1.13 | +0.55 pp | **p=0.0312** | p=0.0001 ✓ | 61.56 ± 4.13 | +2.41 pp | p=0.0625 |
| AZ | 43.67 ± 0.78 | +0.46 pp | **p=0.0312** | p=0.0001 ✓ | 52.45 ± 3.11 | +2.21 pp | p=0.1562 |

fclass linear probe: 98.48 / 98.02. kNN-Jaccard vs POI2Vec: 0.110 / 0.072.

## Verdict

✓ DOMINANCE at both AL and AZ.
- **Cat strictly superior at BOTH states (Wilcoxon p=0.0312, 5/5 folds)**
  — only design in the family to win cat strictly on both states.
- Reg gain matches B/I (+2.2-2.4 pp); not strictly significant.
- kNN-Jaccard with POI2Vec did not lift over B (0.110 vs 0.109) — the soft
  distillation at λ_d=0.1 is too gentle to change the geometry meaningfully.

## FL leak-free results (2026-05-06)

Tag: `STL_FLORIDA_design_m_reg_gethard_pf_5f50ep_leakfree`.

| | reg Acc@10 | F1 | Δ vs canonical(pf) | Wilcoxon p_gt | Δ vs HGI(pf) |
|---|---:|---:|---:|---:|---:|
| canonical c2hgi (pf) | 0.6922 | 0.0907 | — | — | −2.13 pp |
| **M (leak-free)** | **0.7011 ± 0.0046** | 0.0905 | **+0.89 pp** | 0.0625 (4/5) | −1.23 pp |
| J (leak-free) | 0.7034 | 0.0907 | +1.12 pp | 0.0312 ✓ | −0.99 pp |
| HGI (pf) | 0.7134 | 0.0929 | +2.13 pp | — | — |

Per-fold M: [0.6938, 0.7043, 0.7042, 0.6993, 0.7037].

**Verdict at FL**: solid reg lift but n.s. at p=0.0312 (4/5 folds positive).
Sits between B (+0.71) and J (+1.12) on FL reg. Cat F1 axis unchanged.
