# CA §4 MTL cell — champion-G, seed 0, 5f, gated overlap (bf16) — COMPLETE

**Recipe:** champion-G MTL on `check2hgi_dk_ovl`, **bf16** (large-state precision; fp16 overflow-collapsed at
ep30, bf16 trains clean to ep50), seed 0, 5 folds, `--epochs 50`, fixes #1+#3 (4.5x). `--canon none` +
`--no-{reg,cat}-class-weights`, dualtower heads (prior-OFF), `--log-t-kd-weight 0.0`, OneCycle max-lr 3e-3.
Launcher `scripts/closing_data/board_h100_mtl.sh california bf16`. Scored by `h100_score_matched.py`
(per-task diagnostic-best, fold-mean). Rundir `mtlnet_lr1.0e-04_bs2048_ep50_20260624_021104_79596`.

## Result (vs STL ceilings cat 70.26 / reg 63.48)
| task | CA MTL (n=5) | ceiling | Δ vs ceiling |
|---|---|---|---|
| **cat** macro-F1 | **77.3311** ± 0.2164 | 70.26 | **+7.07** (beats) |
| **reg** FULL top10 | **65.6634** ± 0.2613 | 63.48 | **+2.18** (BEATS) |

**CA MTL beats BOTH ceilings** (cat +7.07, reg +2.18) — the "MTL sacrifices reg" pattern is
**reversed** at CA, mirroring FL. bf16 trained healthy (best-epochs late, ep49-50, no ep30 collapse).

## Per-fold (diagnostic-best)
| fold | cat macro-F1 | cat ep | reg FULL top10 | reg ep |
|---|---|---|---|---|
| fold1 | 77.0983 | 50 | 65.4177 | 49 |
| fold2 | 77.5642 | 50 | 65.5533 | 50 |
| fold3 | 77.0474 | 50 | 65.5941 | 50 |
| fold4 | 77.5224 | 50 | 65.5814 | 50 |
| fold5 | 77.4230 | 50 | 66.1706 | 49 |

Artefact: `docs/results/closing_data/h100/california_s0_mtl/` (score JSON + per-fold CSVs).
