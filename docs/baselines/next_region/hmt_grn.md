# HMT-GRN — Hierarchical Multi-Task Graph Recurrent Network

## Source
- **Paper:** Lim, Kang, et al. *Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation.* SIGIR 2022.
- **Reference impl:** authors' release; our adaptation `scripts/baselines/b3_hmt_grn.py` (faithful-to-spirit, with a documented deviation ledger).

## Why this is a baseline (not our model)
HMT-GRN is **the sole region-native external baseline** — the only published method that targets a hierarchical *region* level directly (the field otherwise predicts next-POI and adapts the output). It is the "do we beat a published region model end-to-end?" row and the closest competitor on the region task. It is a **native end-to-end** baseline: its own architecture (shared LSTM over end-to-end-learned POI-id embeddings), with the output head adapted to next-region; it deliberately does **not** consume Check2HGI, so it is a genuinely independent system, not a substrate probe.

## What's faithful, what's adapted
- **Faithful to paper:** shared recurrent trunk + granularity heads; per-fold **train-only** region-transition prior.
- **Adapted because our task / data differ:**
  - tasks {next-region, next-POI} → **{next-category, next-region}** (our two end targets); drops Hierarchical Beam Search + graph selectivity (no next-POI head to prune);
  - geohash grid → our TIGER-tract region partition (identical label space to the champion reg task);
  - native per-user 80/20 → our user-disjoint `StratifiedGroupKFold` (bit-identical to the champion fold split);
  - gated stride-1 overlap base (`--engine check2hgi_dk_ovl`); Istanbul on the Phase-V mahalle substrate (`--engine check2hgi`).

## Variants we run
- `e2e` — native HMT-GRN-style, region head, on our board base. Comparand = **our full MTL champion** (not the STL ceiling — E2E vs frozen-ceiling would be apples-to-oranges).

## Results — board (seed 0 × 5 folds, MPS, region top10_acc)

| State | base | regions | HMT-GRN reg Acc@10 | HMT-GRN cat F1 | our MTL champion reg | verdict |
|---|---|---:|---:|---:|---:|---|
| Alabama | dk_ovl | 1,109 | 57.05 | 19.37 | ~69.8 | ✅ we lead +12.8 |
| Arizona | dk_ovl | 1,547 | 43.70 | 18.04 | ~59.3 | ✅ we lead +15.6 |
| Florida | dk_ovl | 4,703 | 63.74 | 26.87 | 77.28 | ✅ we lead +13.5 |
| California | dk_ovl | 8,501 | 49.61 | 24.01 | 65.66 | ✅ we lead +16.1 |
| Istanbul | Phase-V | 520 | 56.56 | 20.87 | ~69.8 (champ-G) | ✅ we lead; > Markov floor 52.5 |
| Texas | dk_ovl | 6,553 | _in-flight_ | _in-flight_ | 67.13 (2/5) | ⏳ pending |

**Read:** **our joint model beats the sole region-native SOTA at every measured state**, by ≈ +13–16 pp on region — even though region is precisely where a region-native model should compete. HMT-GRN clears the Markov-1 floor (so it learns), but trails our representation+MTL by a wide margin.

## Provenance & the device note (resolved)
- Per-state per-fold raw results: `results/baseline_b3_hmt_grn_style/<state>/` (gitignored); summary in `docs/results/closing_data/MACS_BOARD_RESULTS.md`; consolidated in `RESULTS_BOARD.md §4`.
- **Device audit (PR #38):** the Mac/MPS HMT-GRN numbers are **correct** — validated against deterministic CPU within **0.06 pp fold-for-fold** (AL CPU 56.99 ≈ MPS 57.05). The earlier recorded **62.37 (AL) was the anomaly** (unreproducible, +5 pp); it is *not* a device confound. The numbers above are the trustworthy ones. (Re-verifying the old 62.37 on CUDA is optional, not blocking.)
- The margin to our champion (≈ +13–16 pp) far exceeds any residual cross-device float drift, so the "we beat the region-native by a wide margin" claim is robust.
