# Design H — Learnable POI table at the pool boundary (✓ DOMINANCE)

## Aim

Strip the dependency on a pre-trained HGI POI2Vec. Replace the frozen
POI2Vec residual in Design B with a **learnable embedding table**
indexed by `placeid`, trained end-to-end inside the c2hgi loss. Tests
whether the gain in B is from POI2Vec specifically or from "any
POI-stable lookup at the pool boundary."

## Mechanism

```
poi_table[i]        = nn.Embedding(n_pois, 64)         # learned from scratch
poi_emb_with_table  = poi_emb_canonical.detach() + γ · poi_table[i]
```

Same detachment / loss routing as Design B. Param overhead = `n_pois × 64`
(roughly 758k for FL-scale POI counts).

## AL/AZ leak-free results

| State | cat F1 | Δ vs canonical | Wilcoxon | TOST | reg Acc@10 | Δ vs canonical | Wilcoxon |
|---|---:|---:|---|---|---:|---:|---|
| AL | 40.97 ± 1.20 | +0.21 pp | p=0.4062 | p=0.0024 ✓ | 62.35 ± 3.74 | +3.20 pp | p=0.0312 |
| AZ | 44.14 ± 0.64 | +0.94 pp | p=0.0312 | p<0.0001 ✓ | 52.30 ± 2.99 | +2.06 pp | p=0.1562 |

fclass linear probe: 98.43 / 98.08. kNN-Jaccard vs POI2Vec: 0.115 / 0.075.

## Verdict

✓ DOMINANCE at both AL and AZ.
- AZ cat strictly superior (p=0.0312); AL cat non-inferior.
- AL reg strictly superior (p=0.0312, +3.20 pp), nominally beats HGI.
- fclass at HGI level — the learnable table converges to a
  fclass-discriminative geometry without any POI2Vec supervision.

The H result is the most informative: the merge mechanism does **not**
require HGI's POI2Vec. A learnable per-POI lookup at the right
architectural location recovers the same generality and reg lift.

## FL leak-free results (2026-05-06)

Tag: `STL_FLORIDA_design_h_reg_gethard_pf_5f50ep_leakfree`. FL embeddings
were built fresh (warm_start=True, 500 ep, MPS, ~27 min) before the eval.

| | reg Acc@10 | F1 | Δ vs canonical(pf) | Wilcoxon p_gt | Δ vs HGI(pf) |
|---|---:|---:|---:|---:|---:|
| canonical c2hgi (pf) | 0.6922 | 0.0907 | — | — | −2.13 pp |
| **H (leak-free)** | **0.7041 ± 0.0040** | 0.0913 | **+1.20 pp** | **0.0312 ✓** (5/5) | −0.92 pp |
| J (leak-free) | 0.7034 | 0.0907 | +1.12 pp | 0.0312 ✓ | −0.99 pp |
| HGI (pf) | 0.7134 | 0.0929 | +2.13 pp | — | — |

Per-fold H: [0.6986, 0.7094, 0.7071, 0.7007, 0.7049].

**Verdict at FL**: **strict reg superiority over canonical at p=0.0312
(5/5 folds), best FL reg performer in the merge family**. Closes ~56 % of
the canonical→HGI gap (1.20 / 2.13 pp). The pure-learnable POI table at
the pool boundary is the strongest reg lever, narrowly edging out J's
anchored variant. Cat F1 unchanged.

H+J jointly pass the user's stricter Wilcoxon p=0.0312 gate at FL; the
remaining ~1 pp gap to HGI suggests the upper bound of this merge family
without further architectural change.
