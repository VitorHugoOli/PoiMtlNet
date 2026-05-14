# Gap A — audit snapshot (2026-04-30T15:10Z)

> ## ✅ SUPERSEDED — see [`GAP_A_CLOSURE_20260430.md`](GAP_A_CLOSURE_20260430.md)
>
> This snapshot was taken mid-flight while STAN faithful CA+TX were still running. Those runs were ultimately killed (compute-infeasible at R≈8500) and the campaign was rescoped to AL/AZ/FL with REHDM-faithful + REHDM-STL completing the picture. Final closure document holds the authoritative numbers and rationale.


Mid-flight checkpoint of Gap A (CA + TX baselines closure on Lightning H100). Captured because STAN faithful CA + TX are still mid-run and the machine has had restarts; this file freezes what is reproducibly closed so far so future audits can reconstruct provenance even if the running cells are interrupted again.

## Status board (snapshot)

| Cell                                  | CA            | TX            | Commit                      |
| ------------------------------------- | ------------- | ------------- | --------------------------- |
| Floors (`compute_simple_baselines`)   | ✅            | ✅            | `37e4e1f`                   |
| Markov-K-cat (k=1,3,5,7,9)            | ✅            | ✅            | `37e4e1f`                   |
| MHA+PE faithful (5f×11ep)             | ✅            | ✅            | `37e4e1f`                   |
| POI-RGNN faithful (5f×35ep)           | ✅            | ✅            | `3e036f4` / `6a648fe`       |
| STAN STL (`stl_check2hgi`, `stl_hgi`) | ✅            | ✅            | `37e4e1f`                   |
| **STAN faithful (5f×50ep)**           | 🟡 running    | 🟡 running    | pending                     |

## Closed numbers (paper-ready)

### `next_category` (macro-F1 primary)

| Baseline                  | CA macro-F1     | TX macro-F1     |
| ------------------------- | --------------- | --------------- |
| Markov-K-cat (best K=5)   | 27.59           | 28.67           |
| MHA+PE faithful           | 29.13           | 29.91           |
| POI-RGNN faithful         | **30.71 ± 0.59**| **32.08 ± 0.70**|

POI-RGNN beats MHA+PE by +1.58 / +2.17 pp; both clear Markov-K-cat by +1.5 / +3.4 pp. Pattern holds across all 5 states (AL/AZ/FL/CA/TX) — see `baselines/next_category/comparison.md`.

### `next_region` (Acc@1 primary, partial)

| Baseline                  | CA Acc@1        | TX Acc@1        |
| ------------------------- | --------------- | --------------- |
| Markov-1-region (floor)   | 52.09 ± 0.80    | 54.94 ± 0.34    |
| STAN stl_check2hgi        | 58.82           | 61.35           |
| STAN stl_hgi              | **60.45**       | **62.70**       |
| STAN faithful             | 🔴 (running)    | 🔴 (running)    |

HGI > Check2HGI on STL stays consistent with AL/AZ/FL.

## Provenance — JSONs that back these numbers

```
docs/studies/check2hgi/results/P0/simple_baselines/california/{next_category,next_region,next_category_f1,next_category_markov_kstep}.json
docs/studies/check2hgi/results/P0/simple_baselines/texas/{next_category,next_region,next_category_f1,next_category_markov_kstep}.json
docs/studies/check2hgi/results/baselines/faithful_mha_pe_{california,texas}_5f_11ep_FAITHFUL_MHAPE_{state}_5f11ep.json
docs/studies/check2hgi/results/baselines/faithful_poi_rgnn_{california,texas}_5f_35ep_FAITHFUL_POIRGNN_{state}_5f35ep.json
docs/studies/check2hgi/results/P1/region_head_{california,texas}_region_5f_50ep_STL_{CA,TX}_{check2hgi,hgi}_stan_5f50ep.json
```

State-aggregated views (built by `scripts/_gap_a_finalize.py`):
```
docs/studies/check2hgi/baselines/next_category/results/{california,texas}.json
docs/studies/check2hgi/baselines/next_region/results/{california,texas}.json
```

## Run environment

- Lightning Studio H100 80 GB (replaced original RunPod A100 24 GB plan)
- scikit-learn 1.8.0 (PR #32540 fold-split fix in `StratifiedGroupKFold(shuffle=True)`)
- PyTorch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Raw Gowalla checkins via gdrive folder (gdown), TIGER 2022 tract shapefiles for STAN spatial-join
- Protocol: 5-fold StratifiedGroupKFold, `stratify=target_category`, `groups=userid`, `seed=42`

## Outstanding work

STAN faithful CA + TX (5f × 50ep, batch 2048): launched parallel on H100. STAN logger only emits per-fold summaries — no per-epoch progress. Datasets are 4–6× larger than AL (CA 358k rows / 8501 regions, TX 461k / 6553), so wall-clock per fold scales accordingly. Cells will land in their own commits and update `next_region/comparison.md` + this snapshot's status board.
