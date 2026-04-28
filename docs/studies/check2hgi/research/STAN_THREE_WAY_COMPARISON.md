# STAN three-way comparison — Faithful vs Check2HGI vs HGI substrate

**Date:** 2026-04-26. **Scope:** AL + AZ + FL, all 5-fold × 50-epoch, single-task next-region. **Status:** ✅ all cells filled.

> **Phase-1 reframing (2026-04-27).** This comparison uses the **STAN** sequence head. Under STAN, HGI > C2HGI by +0.96 to +3.68 pp Acc@10 (table below). Under the **matched MTL reg head `next_getnext_hard`** (STAN + α·log_T graph prior), the substrate preference at AL+AZ reverses (AL tied + TOST non-inf at δ=2 pp; AZ +2.34 pp p=0.0312). Phase-1 finding: the substrate-preference flip is **head-coupled, not pure substrate** — STAN prefers HGI's POI-stable smoothness; gethard's graph prior combines productively with C2HGI's per-visit context. See `CLAIMS_AND_HYPOTHESES.md §CH15` reframing + `baselines/PHASE1_VERDICT.md §2.2`. FL gethard substrate comparison queued in `baselines/PHASE2_TRACKER.md §F36c`.

## Headline table

5-fold × 50-epoch, next_region, primary metric Acc@10:

| State | Rows | Regions | Markov-1 floor | **Faithful STAN** | STL STAN / Check2HGI | STL STAN / HGI |
|---|---:|---:|---:|---:|---:|---:|
| AL | 12 709 | 1 109 | 47.01 ± 3.55 | **34.46 ± 3.88** | 59.20 ± 3.62 | **62.88 ± 3.90** |
| AZ | 26 396 | 1 547 | 42.96 ± 2.05 | **38.96 ± 3.41** | 52.24 ± 2.38 | **54.86 ± 2.84** |
| FL | 159 175 | 4 703 | 65.05 ± ? | **65.36 ± 0.69** | 72.62 ± 0.52 | **73.58 ± 0.43** |

Bold = best Faithful (column 5) and best substrate (column 7) per row.

### Acc@1 / MRR companion (single-decimal precision)

| State | Method | Acc@1 | Acc@5 | Acc@10 | MRR | macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| AL | Faithful STAN | 7.86 ± 1.11 | 23.27 ± 2.09 | **34.46 ± 3.88** | 16.45 ± 1.47 | 0.92 |
| AL | Check2HGI substrate | 24.64 ± 1.38 | — | 59.20 ± 3.62 | 36.10 ± 1.96 | 6.34 |
| AL | HGI substrate | 27.40 ± 2.14 | 51.87 | **62.88 ± 3.90** | 39.02 ± 2.66 | 8.77 |
| AZ | Faithful STAN | 13.43 ± 2.57 | 30.49 ± 3.63 | **38.96 ± 3.41** | 21.95 ± 2.83 | 2.08 |
| AZ | Check2HGI substrate | 24.48 ± 2.29 | 43.07 | 52.24 ± 2.38 | 33.70 ± 2.36 | — |
| AZ | HGI substrate | 26.26 ± 2.38 | 45.76 | **54.86 ± 2.84** | 35.87 ± 2.47 | 7.76 |
| FL | Faithful STAN | 35.70 ± 0.67 | 56.59 ± 0.75 | **65.36 ± 0.69** | 45.57 ± 0.57 | 4.55 |
| FL | Check2HGI substrate | 46.82 ± 0.53 | 65.64 | 72.62 ± 0.52 | 55.65 ± 0.43 | 9.70 |
| FL | HGI substrate | 47.37 ± 0.52 | 66.64 | **73.58 ± 0.43** | 56.40 ± 0.40 | 10.86 |

## Pattern 1 — HGI > Check2HGI on next-region, magnitude shrinks with scale

| State | Δ Acc@10 (HGI − Check2HGI) |
|---|---:|
| AL (10 K rows) | +3.68 |
| AZ (26 K rows) | +2.62 |
| FL (159 K rows) | +0.96 |

HGI's substrate advantage on the region task is **monotone-decreasing in data scale**. At AL, HGI's POI-graph encoding gives a clean +3.68 pp edge over Check2HGI's per-checkin contextual encoding; by FL the gap narrows to +0.96 pp (still HGI-positive, still consistent across folds, but small relative to σ envelopes). Plausible mechanism: at small scale, HGI's POI-only embedding aggregated to the region is sharper than Check2HGI's contextually-smoothed region embedding, and that sharpness matters because the region head needs region-discriminative geometry; at FL scale, both substrates have enough data to converge to comparable region-level signal.

This refines the CH16-task-conditional finding from `STAN_HGI_FINDINGS.md`: **HGI > Check2HGI on next-region holds across all three states, but the magnitude is data-scale-dependent and converges as scale grows.**

## Pattern 2 — Faithful STAN's gap to substrate-bound STAN closes dramatically with scale

| State | Faithful | Best substrate (HGI) | Δ |
|---|---:|---:|---:|
| AL | 34.46 | 62.88 | **−28.42 pp** |
| AZ | 38.96 | 54.86 | −15.90 pp |
| FL | 65.36 | 73.58 | −8.22 pp |

The substrate's contribution shrinks from +28 pp on AL (10 K rows) to +8 pp on FL (159 K rows). This is consistent with the standard data-scaling story: from-scratch POI embeddings need many visits per POI to converge, and at AL scale (mean 8 visits per POI, median much lower) STAN's POI table is starved of training signal. At FL scale (mean 13 visits per POI, longer tail), the from-scratch table catches up.

## Pattern 3 — Faithful STAN crosses Markov-1 only at FL scale

| State | Faithful Acc@10 | Markov-1 Acc@10 | Δ |
|---|---:|---:|---:|
| AL | 34.46 | 47.01 | **−12.55 pp** |
| AZ | 38.96 | 42.96 | −4.00 pp |
| FL | 65.36 | 65.05 | **+0.31 pp** |

At AL scale Faithful STAN can't even beat the closed-form 1-gram Markov prior — confirming that the published architecture **needs either pre-trained substrate OR substantially more data** to deliver value on next-region. FL is the smallest of our headline states and is the threshold where Faithful STAN finally edges past Markov.

This means our paper's headline pitch — "the substrate is doing essential work" — needs a scale qualifier: **at AL/AZ scale**, the substrate is essential (faithful STAN can't beat Markov); **at FL scale**, the substrate is still helpful (+8 pp) but Faithful STAN at least clears the floor.

## Pattern 4 — Substrate-bound STAN beats Markov on every state

| State | Best substrate | Markov-1 | Δ |
|---|---:|---:|---:|
| AL | 62.88 (HGI) | 47.01 | **+15.87 pp** |
| AZ | 54.86 (HGI) | 42.96 | **+11.90 pp** |
| FL | 73.58 (HGI) | 65.05 | **+8.53 pp** |

Substrate-bound STAN comfortably beats the Markov floor at every state. The advantage shrinks at FL scale — consistent with the FL-Markov-saturated regime documented in `PAPER_STRUCTURE.md §6` — but stays positive everywhere.

## Implications for the paper

1. **Headline framing.** The clean story is *substrate contribution × data scale*. The substrate adds **+28 pp on AL → +16 pp on AZ → +8 pp on FL** Acc@10 over a faithful-STAN-from-scratch reproduction. This quantifies what our pre-trained Check2HGI/HGI embeddings buy on top of a published-architecture baseline that learns from raw inputs.
2. **HGI > Check2HGI on region (CH16-region).** Holds in all 3 states with magnitude +3.68 → +2.62 → +0.96 pp Acc@10. Direction is uniform (HGI wins all 3 cells); should be reported as a substrate-task finding alongside CH16's Check2HGI > HGI on cat (substrate preference is task-conditional).
3. **Faithful STAN at AL/AZ is below Markov-1.** Strong evidence that the published architecture needs either (a) pre-training or (b) much more data than state-level Gowalla provides. Document as a deliberate baseline-floor finding rather than a paper failure.
4. **At FL scale, Faithful STAN narrowly clears Markov (+0.31 pp).** Still well below substrate (−8 pp), but no longer below the closed-form prior. Useful as an honesty point in the paper: at the largest headline-state scale, even a from-scratch baseline reaches Markov-equivalence.
5. **Std deviations on FL are tiny** (0.4–0.7 pp on Acc@10) — the larger validation sets give cleaner numbers across the board, so FL is where comparisons get genuinely tight.

## Source JSONs (all 9 cells)

| Cell | JSON |
|---|---|
| AL Faithful | `results/baselines/faithful_stan_alabama_5f_50ep_FAITHFUL_STAN_al_5f50ep_v4.json` |
| AL Check2HGI | `results/P1/region_head_alabama_region_5f_50ep_STAN_al_5f50ep.json` |
| AL HGI | `results/P1/region_head_alabama_region_5f_50ep_STAN_HGI_al_5f50ep.json` |
| AZ Faithful | `results/baselines/faithful_stan_arizona_5f_50ep_FAITHFUL_STAN_az_5f50ep_v4.json` |
| AZ Check2HGI | `results/P1/region_head_arizona_region_5f_50ep_STAN_az_5f50ep.json` |
| AZ HGI | `results/P1/region_head_arizona_region_5f_50ep_STAN_HGI_az_5f50ep.json` |
| FL Faithful | `results/baselines/faithful_stan_florida_5f_50ep_FAITHFUL_STAN_fl_5f50ep_v4.json` |
| FL Check2HGI | `results/P1/region_head_florida_region_5f_50ep_STAN_CHECK2HGI_fl_5f50ep.json` |
| FL HGI | `results/P1/region_head_florida_region_5f_50ep_STAN_HGI_fl_5f50ep.json` |

## Architecture / protocol summary

All 9 cells share:
- 5-fold StratifiedGroupKFold(shuffle=True, seed=42), grouped by userid, stratified on target_category.
- 50 epochs, batch 2048, AdamW(lr=1e-4, wd=0.01) + OneCycleLR(max_lr=3e-3), grad-clip 1.0.
- Best-epoch selection on val Acc@10.
- Window=9, non-overlapping stride.

**Faithful STAN** (`research/baselines/stan/`): self-contained from `data/checkins/<State>.parquet` + TIGER tract shapefile. Multi-modal input (POI emb + hour-of-week emb), single-head bare attention with scalar Δt/Δd pairwise bias, matching layer with TIGER-centroid Δd bias. See `FAITHFUL_STAN_FINDINGS.md` §"Phase 2" for audit-driven architecture details.

**STL STAN on Check2HGI/HGI** (`scripts/p1_region_head_ablation.py --heads next_stan --region-emb-source <engine>`): consumes pre-trained region embeddings as the input sequence; bi-layer self-attention (in-house implementation `src/models/next/next_stan/`) with relative-position-only pairwise bias (no ΔT/ΔD because the substrate already absorbs space/time signal); linear classifier to n_regions.

Wall times on M4 Pro MPS:
- AL: Faithful 4.2 min · STL/substrate ≈ 2.5 min total (both heads pooled)
- AZ: Faithful 11 min · STL/substrate ≈ 5.5 min
- FL: Faithful **3.7 hours** · STL/substrate ≈ 21 min per substrate (~42 min total)
