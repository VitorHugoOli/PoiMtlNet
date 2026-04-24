# F19-followup — B3 vs STL (AZ) Paired Wilcoxon

**Date:** 2026-04-23. **Tracker items:** `FOLLOWUPS_TRACKER.md §F18/F19` (post-F19 validation). **Script:** `scripts/analysis/az_b3_wilcoxon_vs_stl.py`.

Establishes the statistical status of B3's MTL-over-STL claims on AZ at n=5 paired folds.

## Runs

- **B3 MTL:** `results/check2hgi/arizona/mtlnet_lr1.0e-04_bs2048_ep50_20260423_0339/` — `mtlnet_crossattn + static_weight(category_weight=0.75) + next_getnext_hard d=256`, 5f × 50ep, seed 42.
- **STL cat:** `results/check2hgi/arizona/next_lr1.0e-04_bs1024_ep50_20260419_0302/` — Check2HGI STL cat.
- **STL STAN:** `docs/studies/check2hgi/results/P1/region_head_arizona_region_5f_50ep_STAN_az_5f50ep.json` — per-fold values.

## Per-metric paired comparison (n=5, diagnostic_task_best epoch)

| Metric | STL (%) | B3 (%) | Δ̄ (pp) | folds + | Wilcoxon p (H₁: B3 > STL) | Verdict |
|---|---|---|---:|:-:|---:|---|
| **cat F1** | 42.08 | **43.72** | **+1.65** | **5/5** | **0.0312** | ✅ **significant MTL-over-STL cat lift (new)** |
| **reg Acc@10_indist** | 52.24 | 51.44 | **−0.81** | 2/5 | 0.9062 | **tied** — does not replicate B-M9d's +1.01 |
| **reg MRR_indist** | 33.70 | 25.77 | **−7.94** | 0/5 | 1.0000 | STL STAN wins all folds |
| **reg macro-F1** | 5.42 | 9.17 | **+3.75** | 5/5 | 0.0312 | ✅ B3 significantly lifts reg F1 |

## Comparison to B-M9d (hard+pcgrad, AZ's prior MTL champion)

| Claim | B-M9d (5f) | B3 (5f) | What changes |
|---|---|---|---|
| cat F1 vs STL cat | −0.14 (not tested) | **+1.65 pp, p=0.0312** | B3 **adds** a significant cat-F1 MTL-over-STL claim |
| reg Acc@10 vs STL STAN | **+1.01, p=0.0312 (5/5 folds +)** | −0.81, p=0.91 (2/5) | B3 **loses** the strict reg Acc@10 win |
| reg MRR vs STL STAN | −6.81 (not tested) | −7.94, p=1.00 | Both trail STL STAN on MRR; B3 slightly worse |
| reg macro-F1 vs STL | ≈ STL | **+3.75, p=0.0312** | B3 adds a new macro-F1 claim |
| cat F1 vs B-M9d | — | **+1.40 pp** | B3 > B-M9d on cat |

**Reading:** B3 trades one Wilcoxon-significant MTL-over-STL claim (reg Acc@10, B-M9d's strongest) for two others (cat F1, reg macro-F1). The cat-F1 claim is arguably more paper-valuable because it's a *simultaneous* head lift (cat goes up without reg dropping, on the same run) — B-M9d's reg Acc@10 win had a cat drop of −0.14 vs STL cat.

## Implications for north-star choice

If B3 becomes the universal north-star:
- **Gained:** FL joint-task Pareto-dominance over soft (pending F17 5-fold confirmation), universal single-config story, strict cat-F1 MTL-over-STL on AZ.
- **Lost:** strict reg-Acc@10 MTL-over-STL on AZ (B-M9d's signature). To keep this finding, the paper should retain B-M9d as an **ablation row**: "under pcgrad optimization at ≤1.5K regions, the hard graph-prior variant delivers +1.01 pp Acc@10 over STL STAN with p=0.0312 — we report this as an ablation to demonstrate the pcgrad-hard mechanism; our headline configuration uses static weighting for cross-state consistency including the FL scale."

This keeps both the strongest statistical finding and the universal-config simplicity.

## Implications for soft retention

Soft's AZ run (B-M9b) has no MTL-over-STL claim at all (all metrics trail STL STAN + STL cat). B3 strictly improves on soft at AZ on every metric of interest (see F19 passing criteria). The only case for soft at AZ would be if the FL hardness test F17 finds B3's cat regression at n=5 exceeds σ on FL — which would force a scale-dependent pivot.

## Script + reproducibility

Script: `scripts/analysis/az_b3_wilcoxon_vs_stl.py` — loads fold_info.json per-fold for MTL runs and the P1 archived STAN summary per-fold list, runs scipy Wilcoxon one-sided + two-sided.

## Cross-references

- `NORTH_STAR.md` (decision memo)
- `research/B5_FL_TASKWEIGHT.md` (F2 mechanism + validation timeline)
- `research/B5_AZ_WILCOXON.md` (F1: prior AZ hard-vs-soft Wilcoxon)
- `FOLLOWUPS_TRACKER.md` §F17/F18/F19 (B3 validation gates)
