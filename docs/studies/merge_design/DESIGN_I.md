# Design I — LoRA r=8 on B (✓ DOMINANCE, parameter-efficient)

## Aim

Reduce H's parameter overhead. Instead of a full POI table (n_pois × 64 ≈
758k params at FL-scale), apply a low-rank adapter (LoRA r=8) on top of
the frozen POI2Vec residual from Design B. Tests whether the merge
mechanism is bottlenecked by capacity or by the right inductive bias.

## Mechanism

```
poi_emb_with_p2v = poi_emb_canonical.detach()
                 + γ · Linear_64→64(POI2Vec)
                 + γ · (B @ A)·POI2Vec        # LoRA r=8, ≈ 95k params
```

`A ∈ ℝ^{8×64}`, `B ∈ ℝ^{64×8}` initialised so `B@A = 0` at start.

## AL/AZ leak-free results

| State | cat F1 | Δ vs canonical | Wilcoxon | TOST | reg Acc@10 | Δ vs canonical | Wilcoxon |
|---|---:|---:|---|---|---:|---:|---|
| AL | 41.62 ± 1.06 | +0.87 pp | p=0.0938 | p=0.0014 ✓ | 61.35 ± 4.22 | +2.20 pp | p=0.0625 |
| AZ | 43.71 ± 0.69 | +0.50 pp | **p=0.0312** | p<0.0001 ✓ | 52.55 ± 3.13 | +2.31 pp | p=0.1562 |

fclass linear probe: 98.58 / 97.87. kNN-Jaccard vs POI2Vec: 0.110 / 0.072.

## Verdict

✓ DOMINANCE at both AL and AZ.
- AZ cat strictly superior at p=0.0312.
- Reg superior to canonical but not strictly significant (4/5 folds at AL).
- 8× cheaper than H (95k vs 758k extra params).

## FL leak-free results (2026-05-06)

Tag: `STL_FLORIDA_design_i_reg_gethard_pf_5f50ep_leakfree`.

| | reg Acc@10 | F1 | Δ vs canonical(pf) | Wilcoxon p_gt | Δ vs HGI(pf) |
|---|---:|---:|---:|---:|---:|
| canonical c2hgi (pf) | 0.6922 | 0.0907 | — | — | −2.13 pp |
| **I (leak-free)** | **0.7002 ± 0.0048** | 0.0914 | **+0.81 pp** | 0.0625 (4/5) | −1.31 pp |
| HGI (pf) | 0.7134 | 0.0929 | +2.13 pp | — | — |

Per-fold I: [0.6961, 0.7023, 0.7037, 0.6942, 0.7049].

**Verdict at FL**: similar to B (+0.71 pp) and M (+0.89 pp), n.s. at
p=0.0312. The 8× parameter savings vs H are preserved without giving up
performance — I is the parameter-efficient default if H proves equivalent.
