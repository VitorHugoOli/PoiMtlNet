# A2 — feature-concat control · running results (M4 Pro)

> Branch `study/pre-freeze-gates`. Substrates **rebuilt locally** on the M4 (user choice;
> HGI + v14 absent from this machine, not in the handoff rsync list). Machine-drift caveat
> carried but **empirically checked** against saved board numbers (RESULTS_TABLE §0.3 /
> embedding_eval FINAL_SYNTHESIS).

## Method
- Harness: `scripts/p1_region_head_ablation.py` (the validated STL substrate-comparison harness),
  extended with `--add-visit-features` (backward-compatible). Per-visit raw features built by
  `scripts/pre_freeze_gates/a2_features.py`, **alignment-validated** (reconstructed placeids ==
  authoritative `sequences_next.parquet`, row-for-row).
- Arms: **HGI**, **HGI⊕feat** (HGI ⊕ Check2HGI's exact node features = category one-hot + hour/dow
  sin/cos), **Check2HGI v14** (`check2hgi_design_k_resln_mae_l0_1`).
- Heads: `next_gru` (cat, macro-F1 at f1-best epoch) / `next_stan_flow` (reg, Acc@10 at top10-best).
- Folds: `StratifiedGroupKFold(seed)` — identical across arms at a given seed.

## Fidelity checks (rebuild vs saved board numbers)
| arm / state / task | rebuilt | saved (board) | verdict |
|---|---|---|---|
| HGI · AL · cat (5f×30ep, seed42) | 26.55 ± 1.40 | 25.3 | ✅ reproduces within seed noise |
| HGI · AL · cat (n=20, seeds 0/1/7/100) | 26.29 ± 1.04 | 25.3 | ✅ |
| v14 · AL · cat (n=20) | 50.73 ± 3.03 | v13≈51.3, c2hgi 41.4 | ✅ ≈ v13 champion |

## A2 cat results (next_gru macro-F1, f1-best epoch)
n=20 = seeds {0,1,7,100} × 5 folds, paired per (seed,fold). Gate contrast = **how much of the
Check2HGI→HGI gap the raw-feature concat closes** (magnitude, not p — n=20 is pseudoreplicated
per the board convention; effect sizes carry the verdict).

| state | n | HGI | HGI⊕feat | v11 (canon) | v14 | concat lift | gap closed vs v11 | gap closed vs v14 |
|---|---|---|---|---|---|---|---|---|
| AL | 20 | 26.29 | 28.32 | 37.36 | 50.73 | +2.02pp | **18.3%** | **8.3%** |
| AZ | 20 | 29.58 | 31.23 | 42.91 | 52.76 | +1.65pp | **12.4%** | **7.1%** |
| FL | 5 (seed0) | 36.21 | 37.04 | 69.53 | 70.45 | +0.83pp | **2.5%** | **2.4%** |

All hgifeat-vs-hgi and substrate-vs-hgifeat contrasts p<1e-3 at AL/AZ (n=20); FL n=5 p=0.0625 (the
n=5 floor — all 5 folds same direction; the 34pp effect size carries it). Pairing per (seed,fold);
p anti-conservative under pseudoreplication — **effect sizes carry the verdict**, per the board
convention. Fidelity: **HGI-cat reproduces the board** (AL 26.5 vs 25.3) and v14 FL 70.45 ≈ saved v13
70.6; the Check2HGI absolutes differ from saved board numbers by ±4–6pp **state-dependently** (v11 AL
37.4 vs 41.4 low, but v11 FL 69.5 vs 63.4 high) — a protocol/version difference, NOT under-training
(which is one-directional). This does not touch the verdict: all gate contrasts are **within-harness**
(HGI, HGI⊕feat, v11, v14 all run here on identical folds), and the gap-closure fraction is unaffected.
The concat arm has WIDER input (75 vs 64) and still closes ≤8% → the conclusion is **conservative**
wrt the feature-injection hypothesis. The gap (hence % closed) GROWS with state size (FL: 34pp gap,
2.4% closed) — feature injection explains *less* of the lift exactly where the lift is largest.

**Read:** adding Check2HGI's *exact* node features (category one-hot + hour/dow sin/cos) to HGI
lifts cat by only ~2pp and closes <10% of the gap. The hierarchical-infomax *learning* (graph
propagation of those features), not feature *access*, drives the lift. The concat arm has MORE
input width (75 vs 64) and still loses → the result is **conservative** wrt the feature-injection
hypothesis. ⇒ gate branch = **substrate claim STRENGTHENED**.

## A2 reg results (next_stan_flow Acc@10, top10-best epoch; per-fold train-only log_T held constant across arms)

| state | n | HGI | HGI⊕feat | v11 | v14 | concat lift | substrate vs HGI |
|---|---|---|---|---|---|---|---|
| AL | 20 | 63.12 | 62.99 | 59.57 | 61.90 | −0.13pp (p=0.40, NS) | v14 −1.21 / v11 −3.55 (HGI ahead) |
| AZ | 20 | 53.39 | 53.37 | 51.02 | 52.74 | −0.03pp (p=0.78, NS) | v14 −0.65 / v11 −2.37 (HGI ahead) |
| FL | 5 (seed0) | 70.28 | 70.05 | 68.89 | 70.09 | −0.23pp | v14 −0.19 (≈ HGI) / v11 −1.39 |

**Reg across AL/AZ/FL:** feature-concat is **inert everywhere** (−0.03…−0.23pp, never significant
positive). Check2HGI is at/behind HGI on region (v14 closes the reg gap to HGI at FL by design; HGI
leads at AL/AZ). ⇒ there is no Check2HGI region lift for feature injection to explain, and concat
doesn't create one. Feature injection explains **neither** head.

**Read (reg):** on the REGION task there is *no* Check2HGI lift to explain — HGI is ahead (spatial
structure wins), consistent with the board. Feature-concat is **inert** (−0.13pp, not significant) —
adding category/temporal features to HGI doesn't move region prediction. So feature injection explains
neither head: Check2HGI's edge is category (infomax learning, §cat above), HGI's edge is region (spatial).

_(AZ reg + FL seed0 + A4 in progress — interrupted by an SSD disconnect 2026-06-17, resumed.)_
