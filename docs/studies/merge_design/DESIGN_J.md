# Design J — H + anchor regulariser λ=0.1 (✓ DOMINANCE, beats HGI on AL reg)

## Aim

H learns its POI table from scratch and lands at HGI-grade fclass
geometry, but the table starts random. Add an anchor loss that pulls the
learnable table toward POI2Vec with weight λ=0.1, hoping to combine H's
end-to-end flexibility with B's known-good initialisation.

## Mechanism

```
poi_emb_with_table = poi_emb_canonical.detach() + γ · poi_table[i]

L_total = L_c2hgi + λ · ‖poi_table − POI2Vec‖²₂   # λ=0.1
```

`L_c2hgi` is the standard 3-boundary contrastive sum from canonical c2hgi.

## AL/AZ leak-free results

| State | cat F1 | Δ vs canonical | Wilcoxon | TOST | reg Acc@10 | Δ vs canonical | Wilcoxon | Δ vs HGI |
|---|---:|---:|---|---|---:|---:|---|---:|
| AL | 41.81 ± 1.46 | +1.05 pp | p=0.0625 | p=0.0009 ✓ | 61.95 ± 3.95 | +2.80 pp | **p=0.0312** | **+0.10 pp** |
| AZ | 43.74 ± 0.76 | +0.53 pp | p=0.1562 | p=0.0008 ✓ | 52.16 ± 2.85 | +1.91 pp | p=0.1562 | −1.22 pp |

fclass linear probe: 98.22 / 97.76.

## Verdict

✓ DOMINANCE at both AL and AZ.
- AL reg strictly superior to canonical (p=0.0312) **and the first design
  to nominally beat HGI on AL reg** (+0.10 pp).
- Cat non-inferior at both states; nominal cat improvement.
- The anchor weight λ=0.1 appears low enough to not over-constrain the
  learnable geometry.

## FL leak-free results (2026-05-06)

Tag: `STL_FLORIDA_design_j_reg_gethard_pf_5f50ep_leakfree`.

| | reg Acc@10 | F1 | Δ vs canonical(pf) | Wilcoxon p_gt | Δ vs HGI(pf) |
|---|---:|---:|---:|---:|---:|
| canonical c2hgi (pf) | 0.6922 | 0.0907 | — | — | −2.13 pp |
| **J (leak-free)** | **0.7034 ± 0.0051** | 0.0907 | **+1.12 pp** | **0.0312 ✓** (5/5) | −0.99 pp |
| B (leak-free) | 0.6993 | 0.0907 | +0.71 pp | 0.0625 (4/5) | −1.41 pp |
| HGI (pf) | 0.7134 | 0.0929 | +2.13 pp | — | — |

Per-fold J: [0.6979, 0.7087, 0.7060, 0.6980, 0.7066].

**Verdict at FL**: **strict reg superiority over canonical (Wilcoxon
p=0.0312, 5/5 folds)**. Best FL reg performer in the merge family at the
leak-free protocol. Still ~1 pp behind HGI on reg. Cat F1 axis unchanged.

Combined with AL/AZ: J is the only design with **strict reg wins at AL
(p=0.0312) AND FL (p=0.0312)**. The anchor regulariser λ=0.1 toward
POI2Vec is the architectural detail that scales further than B's frozen
residual or H's pure learnable table.
