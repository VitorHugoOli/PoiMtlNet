# next_region — baseline comparison

Generated from `results/<state>.json`. To refresh, regenerate the JSONs (see `../README.md` §"How to add a new baseline result") and re-run the comparison-table generator (TODO: script `scripts/build_baseline_comparison.py` not yet written).

## Cross-baseline summary — Acc@10 (mean ± σ)

| Baseline | Variant | AL | AZ | FL | CA | TX |
|---|---|---:|---:|---:|---:|---:|
| Markov-1-region (floor) | — | 47.01 ± 3.55 | 42.96 ± 2.05 | 65.05 ± 0.93 | 52.09 ± 0.80 | 54.94 ± 0.46 |
| **STAN** | `faithful` | 34.46 ± 3.88 | 38.96 ± 3.41 | 65.36 ± 0.69 | ⚪† | ⚪† |
| **STAN** | `stl_check2hgi` | 59.20 ± 3.62 | 52.24 ± 2.38 | 72.62 ± 0.52 | 58.82 ± 1.04 | 61.35 ± 0.36 |
| **STAN** | `stl_hgi` | **62.88 ± 3.90** | **54.86 ± 2.84** | **73.58 ± 0.43** | **60.45 ± 0.97** | **62.70 ± 0.37** |
| **ReHDM** † | `faithful` | **66.06 ± 0.98** | **54.65 ± 0.77** | 65.68 ± 0.26 | ⚪ | ⚪ |
| **ReHDM** ‡ | `stl_check2hgi` | 26.22 ± 1.58 | 23.24 ± 1.27 | 38.74 ± 0.49 | ⚪ | ⚪ |
| **ReHDM** ‡ | `stl_hgi` | 42.78 ± 2.82 | 34.00 ± 3.02 | **54.49 ± 0.32** | ⚪ | ⚪ |

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

### `stl_check2hgi` — STAN architecture, Check2HGI substrate input

| State | Acc@1 | Acc@5 | Acc@10 | MRR | macro-F1 |
|---|---:|---:|---:|---:|---:|
| AL | 24.64 ± 1.38 | 48.19 ± 2.74 | 59.20 ± 3.62 | 36.10 ± 1.96 | 6.34 ± 0.41 |
| AZ | 24.48 ± 2.29 | 43.07 ± 2.40 | 52.24 ± 2.38 | 33.70 ± 2.36 | 5.42 ± 0.46 |
| FL | 46.82 ± 0.53 | 65.64 ± 0.62 | 72.62 ± 0.52 | 55.65 ± 0.43 | 9.70 ± 0.11 |
| CA | 31.27 ± 0.88 | 50.71 ± 1.08 | 58.82 ± 1.04 | 40.62 ± 0.91 | 8.06 ± 0.21 |
| TX | 29.04 ± 0.44 | 52.01 ± 0.53 | 61.35 ± 0.36 | 39.96 ± 0.45 | 9.29 ± 0.16 |

### `stl_hgi` — STAN architecture, HGI substrate input

| State | Acc@1 | Acc@5 | Acc@10 | MRR | macro-F1 |
|---|---:|---:|---:|---:|---:|
| AL | 27.40 ± 2.14 | 51.87 ± 3.96 | 62.88 ± 3.90 | 39.02 ± 2.66 | 8.77 ± 0.94 |
| AZ | 26.26 ± 2.38 | 45.76 ± 2.79 | 54.86 ± 2.84 | 35.87 ± 2.47 | 7.76 ± 0.85 |
| FL | 47.37 ± 0.52 | 66.64 ± 0.43 | 73.58 ± 0.43 | 56.40 ± 0.40 | 10.86 ± 0.21 |
| CA | 31.90 ± 0.91 | 52.20 ± 1.03 | 60.45 ± 0.97 | 41.59 ± 0.93 | 8.89 ± 0.30 |
| TX | 29.55 ± 0.43 | 53.16 ± 0.48 | 62.70 ± 0.37 | 40.75 ± 0.42 | 10.25 ± 0.21 |

### `faithful` — paper-architecture from raw inputs (ReHDM)

ReHDM `faithful` ingests 6 paper-defined IDs (user, POI, category, hour, day-of-week, quadkey-L10) and runs the dual-level hypergraph machinery; classifier output adapted to `n_regions`. Protocol = chronological 80/10/10 + 24h sessions + 5 seeds (paper §5.1).

§ FL run uses `batch_size=128 + lr/max_lr scaled 4×` (linear scaling rule from paper's batch=32, lr=5e-5, max_lr=5e-4). Validated on AL (Acc@10=65.85±1.53 vs ref 66.06±0.98, within 1σ) and AZ (54.94±0.12 vs ref 54.65±0.77, within 1σ) before launching FL — quality preserved, training ~3× faster. Reduced FL ETA from ~4.5h (b=32 paper-faithful) to ~92 min.

| State | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| AL | 23.73 ± 1.38 | 53.70 ± 1.31 | 66.06 ± 0.98 | 37.83 ± 1.17 |
| AZ | 19.81 ± 0.54 | 42.63 ± 0.37 | 54.65 ± 0.77 | 30.96 ± 0.36 |
| FL § | 33.95 ± 0.40 | 56.06 ± 0.36 | 65.68 ± 0.26 | 44.58 ± 0.33 |

### `stl_check2hgi` — ReHDM (full hypergraph), Check2HGI substrate input

5-fold StratifiedGroupKFold; full intra+inter hypergraph; time-precedence dropped (see footnote ‡).

| State | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| AL | 5.77 ± 1.01 | 17.92 ± 1.53 | 26.22 ± 1.58 | 12.53 ± 1.12 |
| AZ | 8.66 ± 0.78 | 17.11 ± 1.24 | 23.24 ± 1.27 | 13.58 ± 0.89 |
| FL | 24.07 ± 0.51 | 34.42 ± 0.45 | 38.74 ± 0.49 | 29.31 ± 0.50 |

### `stl_hgi` — ReHDM (full hypergraph), HGI substrate input

| State | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| AL | 13.17 ± 1.03 | 32.44 ± 2.55 | 42.78 ± 2.82 | 22.83 ± 1.38 |
| AZ | 12.09 ± 1.71 | 26.42 ± 3.08 | 34.00 ± 3.02 | 19.53 ± 2.15 |
| FL | 32.64 ± 0.48 | 48.06 ± 0.36 | 54.49 ± 0.32 | 40.08 ± 0.28 |

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

## Substrate-head matched STL — Phase 1 (next_getnext_hard, MTL B3 reg head)

Quick reference for the substrate-comparison Phase-1 matched-head reg results. Authoritative source: [`../../research/SUBSTRATE_COMPARISON_FINDINGS.md`](../../research/SUBSTRATE_COMPARISON_FINDINGS.md). Per-fold JSONs: `../../results/phase1_perfold/{AL,AZ}_{check2hgi,hgi}_reg_gethard_5f50ep.json`. Paired tests + TOST: `../../results/paired_tests/{alabama,arizona}_acc10_reg_acc10.json`.

| Substrate | Variant | AL Acc@10 | AZ Acc@10 | FL | CA | TX |
|---|---|---:|---:|---:|---:|---:|
| Check2HGI | matched-head `next_getnext_hard` STL | **68.37 ± 2.66** | **66.74 ± 2.11** | 🔴 (F36c) | 🔴 | 🔴 |
| HGI | matched-head `next_getnext_hard` STL | 67.52 ± 2.80 | 64.40 ± 2.42 | 🔴 (F36c) | 🔴 | 🔴 |
| **Δ (C2HGI − HGI)** | Wilcoxon (Acc@10) + TOST δ=2pp | +0.85 (p=0.0625 marg, **TOST non-inf**) | **+2.34 (p=0.0312)** | 🔴 | 🔴 | 🔴 |

Under the matched MTL reg head (graph prior), C2HGI ≥ HGI everywhere: AL tied within σ + non-inferior at δ=2 pp; AZ significantly C2HGI. The earlier "HGI > C2HGI on reg under STAN" pattern in the table above was **head-coupled** — STAN prefers HGI's POI-stable smoothness; gethard's graph prior combines productively with C2HGI's per-visit context. See FINDINGS §2.2 for the reframing.

## MTL B3 substrate-counterfactual — Phase 1 (CH18)

| State | Substrate | cat F1 | reg Acc@10_indist | Δ_cat (C2HGI−HGI) | Δ_reg (C2HGI−HGI) |
|---|---|---:|---:|---:|---:|
| AL | C2HGI (existing) | **42.71 ± 1.37** | **59.60 ± 4.09** | — | — |
| AL | HGI (counterfactual) | 25.96 ± 1.61 | 29.95 ± 1.89 | **+16.75** | **+29.65** |
| AZ | C2HGI (existing) | **45.81 ± 1.30** | **53.82 ± 3.11** | — | — |
| AZ | HGI (counterfactual) | 28.70 ± 0.51 | 22.10 ± 1.63 | **+17.11** | **+31.72** |

MTL+HGI is **worse than STL+HGI** on reg by ~37 pp at AL — the B3 configuration does not generalise to HGI. CH18 = MTL B3 is substrate-specific. Per-fold JSONs: `../../results/phase1_perfold/{AL,AZ}_hgi_mtl_{cat,reg}.json`.
