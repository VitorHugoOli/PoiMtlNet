# next_region — baseline comparison

Generated from `results/<state>.json`. To refresh, regenerate the JSONs (see `../README.md` §"How to add a new baseline result") and re-run the comparison-table generator (TODO: script `scripts/build_baseline_comparison.py` not yet written).

## Cross-baseline summary — Acc@10 (mean ± σ)

| Baseline | Variant | AL | AZ | FL | CA | TX | GA |
|---|---|---:|---:|---:|---:|---:|---:|
| Markov-1-region (floor) | — | 47.01 ± 3.55 | 42.96 ± 2.05 | 65.05 ± 0.93 | 52.09 ± 0.80 | 54.94 ± 0.46 | 48.19 ± 2.18 |
| **STAN** | `faithful` | 34.46 ± 3.88 | 38.96 ± 3.41 | 65.36 ± 0.69 | ⚪† | ⚪† | 40.68 ± 1.10 |
| **STAN** | `stl_check2hgi` | 59.20 ± 3.62 | 52.24 ± 2.38 | 72.62 ± 0.52 | 58.82 ± 1.04 | 61.35 ± 0.36 | 56.35 ± 2.40 |
| **STAN** | `stl_hgi` | **62.88 ± 3.90** | **54.86 ± 2.84** | **73.58 ± 0.43** | **60.45 ± 0.97** | **62.70 ± 0.37** | **58.58 ± 1.86** |
| **ReHDM** † | `faithful` | **66.06 ± 0.98** | **54.65 ± 0.77** | 65.68 ± 0.26 | ⚪ | ⚪ | 55.82 ± 0.76 |
| **ReHDM** ‡ | `stl_check2hgi` | 26.22 ± 1.58 | 23.24 ± 1.27 | 38.74 ± 0.49 | ⚪ | ⚪ | 22.31 ± 1.31 |
| **ReHDM** ‡ | `stl_hgi` | 42.78 ± 2.82 | 34.00 ± 3.02 | **54.49 ± 0.32** | ⚪ | ⚪ | 35.07 ± 1.98 |

Bold = best variant per state-baseline. ⚪ = intentionally out of scope (STAN/REHDM faithful CA/TX shown infeasible at scale; substrate-axis covered at 5 states via STAN-STL — see `GAP_A_CLOSURE_20260430.md`). 🟡 = partial; ✅ = 5-fold/seed complete.

† **ReHDM `faithful` uses the paper's protocol** (chronological 80/10/10 + 24h sessions + 5 seeds, not 5 StratifiedGroupKFold folds). σ is inter-seed; cell-for-cell σ comparison with STAN rows is not valid. The qualitative ordering vs Markov-1 floor still holds.

‡ **ReHDM `stl_*` use study protocol** (5-fold StratifiedGroupKFold matching all other STL rows) **with the FULL hypergraph**: intra-user collaborators (same userid in train fold, naturally empty for cold-user val rows) + inter-user collaborators (shared POI in train fold, loaded from `sequences_next.parquet`). Only deviation from paper architecture is dropping time-precedence (`end(s_m) < start(target)`) since StratifiedGroupKFold is non-temporal by design. See `rehdm.md` §"Protocol & architecture choices per variant". Paper-protocol replicates archived in `results/<state>.json::baselines.rehdm._paper_protocol_stl_archive`.

## Per-baseline detail — STAN

5-fold × 50-epoch, StratifiedGroupKFold(seed=42) on userid stratified by target_category.

### `faithful` — paper-architecture from raw inputs

| State | Acc@1 | Acc@5 | Acc@10 | MRR | macro-F1 |
|---|---:|---:|---:|---:|---:|
| AL | 7.86 ± 1.11 | 23.27 ± 2.09 | 34.46 ± 3.88 | 16.45 ± 1.47 | 0.92 ± 0.32 |
| AZ | 13.43 ± 2.57 | 30.49 ± 3.63 | 38.96 ± 3.41 | 21.95 ± 2.83 | 2.08 ± 0.88 |
| FL | 35.70 ± 0.67 | 56.59 ± 0.75 | 65.36 ± 0.69 | 45.57 ± 0.57 | 4.55 ± 0.31 |
| CA | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 |
| TX | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 |
| GA | 15.19 ± 0.67 | 31.41 ± 0.92 | 40.68 ± 1.10 | 23.58 ± 0.69 | 2.45 ± 0.45 |

### `stl_check2hgi` — STAN architecture, Check2HGI substrate input

| State | Acc@1 | Acc@5 | Acc@10 | MRR | macro-F1 |
|---|---:|---:|---:|---:|---:|
| AL | 24.64 ± 1.38 | 48.19 ± 2.74 | 59.20 ± 3.62 | 36.10 ± 1.96 | 6.34 ± 0.41 |
| AZ | 24.48 ± 2.29 | 43.07 ± 2.40 | 52.24 ± 2.38 | 33.70 ± 2.36 | 5.42 ± 0.46 |
| FL | 46.82 ± 0.53 | 65.64 ± 0.62 | 72.62 ± 0.52 | 55.65 ± 0.43 | 9.70 ± 0.11 |
| CA | 31.27 ± 0.88 | 50.71 ± 1.08 | 58.82 ± 1.04 | 40.62 ± 0.91 | 8.06 ± 0.21 |
| TX | 29.04 ± 0.44 | 52.01 ± 0.53 | 61.35 ± 0.36 | 39.96 ± 0.45 | 9.29 ± 0.16 |
| GA | 27.08 ± 1.33 | 47.03 ± 2.17 | 56.35 ± 2.40 | 36.77 ± 1.66 | 5.73 ± 0.55 |

### `stl_hgi` — STAN architecture, HGI substrate input

| State | Acc@1 | Acc@5 | Acc@10 | MRR | macro-F1 |
|---|---:|---:|---:|---:|---:|
| AL | 27.40 ± 2.14 | 51.87 ± 3.96 | 62.88 ± 3.90 | 39.02 ± 2.66 | 8.77 ± 0.94 |
| AZ | 26.26 ± 2.38 | 45.76 ± 2.79 | 54.86 ± 2.84 | 35.87 ± 2.47 | 7.76 ± 0.85 |
| FL | 47.37 ± 0.52 | 66.64 ± 0.43 | 73.58 ± 0.43 | 56.40 ± 0.40 | 10.86 ± 0.21 |
| CA | 31.90 ± 0.91 | 52.20 ± 1.03 | 60.45 ± 0.97 | 41.59 ± 0.93 | 8.89 ± 0.30 |
| TX | 29.55 ± 0.43 | 53.16 ± 0.48 | 62.70 ± 0.37 | 40.75 ± 0.42 | 10.25 ± 0.21 |
| GA | 28.56 ± 1.07 | 49.34 ± 1.82 | 58.58 ± 1.86 | 38.58 ± 1.34 | 7.84 ± 0.59 |

### `faithful` — paper-architecture from raw inputs (ReHDM)

ReHDM `faithful` ingests 6 paper-defined IDs (user, POI, category, hour, day-of-week, quadkey-L10) and runs the dual-level hypergraph machinery; classifier output adapted to `n_regions`. Protocol = chronological 80/10/10 + 24h sessions + 5 seeds (paper §5.1).

§ FL run uses `batch_size=128 + lr/max_lr scaled 4×` (linear scaling rule from paper's batch=32, lr=5e-5, max_lr=5e-4). Validated on AL (Acc@10=65.85±1.53 vs ref 66.06±0.98, within 1σ) and AZ (54.94±0.12 vs ref 54.65±0.77, within 1σ) before launching FL — quality preserved, training ~3× faster. Reduced FL ETA from ~4.5h (b=32 paper-faithful) to ~92 min.

| State | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| AL | 23.73 ± 1.38 | 53.70 ± 1.31 | 66.06 ± 0.98 | 37.83 ± 1.17 |
| AZ | 19.81 ± 0.54 | 42.63 ± 0.37 | 54.65 ± 0.77 | 30.96 ± 0.36 |
| FL § | 33.95 ± 0.40 | 56.06 ± 0.36 | 65.68 ± 0.26 | 44.58 ± 0.33 |
| GA § | — | — | 55.82 ± 0.76 | 34.18 ± 0.45 |

### `stl_check2hgi` — ReHDM (full hypergraph), Check2HGI substrate input

5-fold StratifiedGroupKFold; full intra+inter hypergraph; time-precedence dropped (see footnote ‡).

| State | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| AL | 5.77 ± 1.01 | 17.92 ± 1.53 | 26.22 ± 1.58 | 12.53 ± 1.12 |
| AZ | 8.66 ± 0.78 | 17.11 ± 1.24 | 23.24 ± 1.27 | 13.58 ± 0.89 |
| FL | 24.07 ± 0.51 | 34.42 ± 0.45 | 38.74 ± 0.49 | 29.31 ± 0.50 |
| GA | 9.26 ± 0.84 | 16.66 ± 1.35 | 22.31 ± 1.31 | 13.79 ± 1.00 |

### `stl_hgi` — ReHDM (full hypergraph), HGI substrate input

| State | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| AL | 13.17 ± 1.03 | 32.44 ± 2.55 | 42.78 ± 2.82 | 22.83 ± 1.38 |
| AZ | 12.09 ± 1.71 | 26.42 ± 3.08 | 34.00 ± 3.02 | 19.53 ± 2.15 |
| FL | 32.64 ± 0.48 | 48.06 ± 0.36 | 54.49 ± 0.32 | 40.08 ± 0.28 |
| GA | 13.90 ± 1.09 | 27.27 ± 1.89 | 35.07 ± 1.98 | 21.00 ± 1.33 |

### Within-baseline pattern (ReHDM, AL Acc@10)

| Variant | Protocol | Architecture | Acc@10 |
|---|---|---|---:|
| `faithful` (raw 6 IDs) | paper | full incl. hypergraph + time-precedence | 66.06 |
| `stl_hgi` | study | full hypergraph, no time-precedence | 42.78 |
| `stl_check2hgi` | study | full hypergraph, no time-precedence | 26.22 |

ReHDM's strength on AL comes jointly from (a) the 6-ID embedding stack, (b) the dual-level hypergraph and (c) the data conditions it was designed for (warm-user chronological windows). Replacing (a) with our pre-trained substrates and switching the protocol to cold-user StratifiedGroupKFold halves the headline number even with the full hypergraph in place. This is a faithful-architecture-vs-fair-protocol trade-off: `faithful` shows what ReHDM can do under its own terms; `stl_*` shows what the same architecture buys when it has to compete on the study's terms (cold-user holdout, substrate input).

### Cross-baseline (best-of-baseline, AL Acc@10)

| Method | Variant used for column | Protocol | Acc@10 |
|---|---|---|---:|
| Markov-1-region (floor) | — | study | 47.01 |
| `next_gru` (in-house STL) | check2hgi | study | 56.94 |
| STAN | `stl_hgi` (best STAN) | study | 62.88 |
| **ReHDM** | `faithful` (best ReHDM) | **paper** | **66.06** |
| ReHDM | `stl_hgi` (best ReHDM under study protocol) | study | 42.78 |

Two honest takeaways at AL:
- Under the **paper's own protocol**, ReHDM's `faithful` variant beats the strongest STAN variant by +3.18 pp Acc@10. This is the "external published-SOTA reference" claim.
- Under the **study's protocol** (5-fold cold-user StratifiedGroupKFold) and the same architecture, ReHDM's STL variants underperform STAN's STL variants by 20–37 pp. The full hypergraph is operative (intra+inter via shared POIs) but ReHDM's theta-query pooling is a weaker readout than STAN's bi-layer last-position matching when the input is a 9-step embedding sequence rather than the paper's longer warm-user trajectories with the 6-ID stack.

## Pattern summary (Acc@10)

> ⚠ **Read the §"Substrate-head matched STL — Phase 1" section below first.** The "HGI − Check2HGI" column in this STAN-row table is **head-coupled to STAN** (which prefers POI-stable smoothness) and is **NOT the post-2026-04-27 reading** of the substrate-on-reg comparison. Under the matched MTL reg head (`next_getnext_hard` = STAN + α·log_T graph prior), Check2HGI ≥ HGI everywhere (AL TOST non-inferior at δ=2pp Acc@10; AZ +2.34 pp Acc@10 / +1.29 pp MRR, p=0.0312, 5/5 folds). CH15 has been **reframed as head-coupled** in `CLAIMS_AND_HYPOTHESES.md §CH15` and `CONCERNS.md §C16`. The STAN-row data below is preserved as the **head-sensitivity probe row** for the paper, not the headline substrate finding.

| State | Faithful − Markov | HGI − Check2HGI (STAN, head-coupled) | HGI − Faithful (substrate gap) |
|---|---:|---:|---:|
| AL | **−12.55** | +3.68 | +28.42 |
| AZ | −4.00 | +2.62 | +15.90 |
| FL | **+0.31** | +0.96 | +8.22 |

- **Faithful crosses Markov only at FL scale.** AL/AZ from-scratch STAN sits below Markov-1-region (4–13 pp), confirming the architecture needs either pre-training or much more data.
- **HGI > Check2HGI on next-region across all 3 states**, but the magnitude shrinks monotonically with scale (+3.68 → +2.62 → +0.96).
- **Substrate contribution (HGI − Faithful) shrinks with scale**, from +28 pp (AL) → +16 pp (AZ) → +8 pp (FL). Pre-trained substrates buy more at small scale.

Full deep-dive interpretation: `../../research/STAN_THREE_WAY_COMPARISON.md`.

## Substrate-head matched STL — leak-free (Phase 3, `_pf` per-fold transitions)

> **2026-04-30 update — supersedes the leaky Phase 1 numbers.** The Phase 1 GETNext-hard matched-head numbers (originally tabulated here) used a full-data `region_transition_log.pt` graph prior that leaked val transitions into the `α·log_T` term. Phase 3 re-ran every cell with `--per-fold-transition-dir` (StratifiedGroupKFold train-only edges per fold). Numbers below are leak-free. See [`../../FINAL_SURVEY.md §4 + §6`](../../FINAL_SURVEY.md) for the full statistical write-up; per-fold JSONs at `../../results/phase1_perfold/<S>_<engine>_reg_gethard_pf_5f50ep.json`.

| Substrate | Variant | AL Acc@10 | AZ Acc@10 | FL Acc@10 | CA Acc@10 | TX Acc@10 |
|---|---|---:|---:|---:|---:|---:|
| Check2HGI | matched-head `next_getnext_hard_pf` STL | 59.15 ± 3.48 | 50.24 ± 2.51 | 69.22 ± 0.52 | 55.92 ± 1.20 | 58.89 ± 1.28 |
| HGI       | matched-head `next_getnext_hard_pf` STL | **61.86 ± 3.29** | **53.37 ± 2.55** | **71.34 ± 0.64** | **57.77 ± 1.12** | **60.47 ± 1.26** |
| **Δ (C2HGI − HGI)** | Wilcoxon p_greater + TOST δ=2pp | **−2.71** (p=1.0; TOST FAIL) | **−3.13** (p=1.0; TOST FAIL) | **−2.12** (TOST FAIL δ=2pp; ✓ δ=3pp) | **−1.85** (TOST ✓ non-inf δ=2pp) | **−1.59** (TOST ✓ non-inf δ=2pp) |

**CH15 verdict (leak-free):** rejected at AL/AZ/FL (HGI nominally above by 2.1-3.1 pp); tied at CA/TX (Δ < 2 pp, TOST non-inferior at δ=2pp). The Phase 1 finding "C2HGI ≥ HGI under matched head" sign-flipped at every state once the `α·log_T` leak was removed — C2HGI had been exploiting the leaky transition prior more than HGI (substrate-asymmetric leakage, AZ peak ~5.5 pp differential). The post-fix reading is **substrate-equivalent on reg with a slight HGI tilt**, not C2HGI advantage.

> ⚪ GA was scoped to external-baseline coverage (STAN/MHA+PE/POI-RGNN/ReHDM); the Phase-1 matched-head substrate axis was not run for GA. Existing 5-state coverage is sufficient for the substrate claim.

### Phase 1 leaky reference (kept for the F44 leak shift analysis only — do NOT cite as substrate finding)

| Substrate | AL Acc@10 (leaky) | AZ (leaky) | FL (leaky) | CA (leaky) | TX (leaky) |
|---|---:|---:|---:|---:|---:|
| Check2HGI `next_getnext_hard` (leaky) | 68.37 ± 2.66 | 66.74 ± 2.11 | 82.54 | 70.63 | 69.31 |
| HGI `next_getnext_hard` (leaky)       | 67.52 ± 2.80 | 64.40 ± 2.42 | 82.25 | 71.29 | 69.90 |
| Δ (leaky)                             | +0.85 | +2.34 | +0.29 | −0.66 | −0.59 |

The leak inflated absolute Acc@10 by 9-16 pp across all states and was substrate-asymmetric (C2HGI lost more pp than HGI when the leak was removed — AL Δ_C2HGI=−9.22 vs Δ_HGI=−5.66, AZ Δ_C2HGI=−16.51 vs Δ_HGI=−11.03). See [`../../FINAL_SURVEY.md §6`](../../FINAL_SURVEY.md) for the substrate-asymmetric F44 leak diagnosis.

## MTL B3 substrate-counterfactual — Phase 1 (CH18)

| State | Substrate | cat F1 | reg Acc@10_indist | Δ_cat (C2HGI−HGI) | Δ_reg (C2HGI−HGI) |
|---|---|---:|---:|---:|---:|
| AL | C2HGI (existing) | **42.71 ± 1.37** | **59.60 ± 4.09** | — | — |
| AL | HGI (counterfactual) | 25.96 ± 1.61 | 29.95 ± 1.89 | **+16.75** | **+29.65** |
| AZ | C2HGI (existing) | **45.81 ± 1.30** | **53.82 ± 3.11** | — | — |
| AZ | HGI (counterfactual) | 28.70 ± 0.51 | 22.10 ± 1.63 | **+17.11** | **+31.72** |

MTL+HGI is **worse than STL+HGI** on reg by ~37 pp at AL — the B3 configuration does not generalise to HGI. CH18 = MTL B3 is substrate-specific. Per-fold JSONs: `../../results/phase1_perfold/{AL,AZ}_hgi_mtl_{cat,reg}.json`.
