# next_category — baseline comparison

> ⚠ **SUPERSEDED 2026-05-04.** Canonical:
> `docs/studies/check2hgi/results/RESULTS_TABLE.md §0.6`. The POI-RGNN
> numbers (33.35 / 30.71 / 32.08) below are pre-bugfix (May-2 snapshot)
> and diverge from the canonical 34.49 / 31.78 / 33.03 quoted in
> `RESULTS_TABLE.md §0.6` and inherited by the article. The cross-validation
> protocol below may also reflect an earlier protocol claim; the current
> reproduction is user-disjoint per `poi_rgnn.md` "Adapted —
> Cross-validation protocol". Trust the canonical sources, not this file.

Generated from `results/<state>.json`. To refresh, regenerate the JSONs (see `../README.md` §"How to add a new baseline result").

## Cross-baseline summary — macro-F1 (mean ± σ)

| Baseline | Variant | AL | AZ | FL | CA | TX | GA |
|---|---|---:|---:|---:|---:|---:|---:|
| Majority class (floor) | — | 7.28 ± 0.00 | 7.25 ± 0.00 | 5.66 ± 0.00 | 7.04 ± 0.00 | 6.76 ± 0.00 | 6.69 ± 0.00 |
| Markov-1-POI (floor) | last POI | 16.81 ± 1.06 | 19.48 ± 0.63 | 27.60 ± 0.32 | 24.95 ± 1.18 | 25.85 ± 0.55 | 21.36 ± 0.36 |
| Markov-9-cat (floor) | 9-cat seq, backoff | best K=5: **20.50 ± 0.67** | best K=5: **23.92 ± 2.26** | best K=3: **29.74 ± 1.19** | best K=5: **27.59 ± 0.61** | best K=5: **28.67 ± 0.66** | best K=3: **27.01 ± 1.10** |
| **MHA+PE** (Zeng 2019) | `faithful` (8-step window) | 18.95 ± 0.71 | 24.99 ± 0.85 | 32.06 ± 0.23 | 29.13 ± 0.71 | 29.91 ± 0.43 | 27.62 ± 0.97 |
| **POI-RGNN** | `faithful` (9-step window) | **23.80 ± 1.12** | **27.64 ± 2.34** | **33.35 ± 1.14** | **30.71 ± 0.82** | **32.08 ± 0.70** | **30.24 ± 0.87** |

The Markov-1-POI line conditions on the last POI ID only (paper-style 1-step floor). **Markov-K-cat is the apples-to-apples sequence floor for POI-RGNN**: both methods see the same 9-step category window, the Markov side conditioning on the last K categories with stupid backoff (K → K-1 → … → 1 → unigram). See `scripts/compute_markov_kstep_cat.py` and `next_category_markov_kstep.json` per state.

(🔴 = pending; 🟡 = partial / 1-fold; ✅ = 5-fold complete. CA + TX MHA+PE faithful + Markov-K-cat closed locally on H100 2026-04-30; POI-RGNN faithful still running.)

## Cross-baseline summary — Acc@1 (mean ± σ)

| Baseline | Variant | AL | AZ | FL | CA | TX | GA |
|---|---|---:|---:|---:|---:|---:|---:|
| Majority class (floor) | — | 34.19 | 34.01 | 24.69 | 32.72 | 30.98 | 30.56 |
| Markov-1-POI (floor) | — | 31.69 ± 0.45 | 32.58 ± 0.32 | 36.88 ± 0.44 | 34.20 ± 0.20 | 33.12 ± 0.14 | 30.43 ± 0.53 |
| best Markov-K-cat | — | k=3: 36.72 ± 1.22 | k=3: 39.26 ± 1.37 | k=3: 42.03 ± 0.64 | k=3: 39.44 ± 0.47 | k=3: 37.99 ± 0.23 | k=3: 37.55 ± 0.92 |
| **MHA+PE** | `faithful` | 39.35 ± 1.10 | 41.23 ± 1.23 | 43.83 ± 0.17 | 40.40 ± 0.40 | 39.43 ± 0.28 | 39.13 ± 0.83 |
| **POI-RGNN** | `faithful` | **39.21 ± 1.97** | **41.04 ± 1.79** | **44.07 ± 0.76** | **41.32 ± 0.52** | **40.59 ± 0.50** | **39.89 ± 0.71** |

POI-RGNN beats every floor on Acc@1 across all three states. The macro-F1 lift over the majority floor is ≥24 pp at every state — POI-RGNN learns to discriminate the minority categories that the majority floor ignores entirely.

## Per-baseline detail — MHA+PE

5-fold × 11-epoch (paper §5.4 Gowalla 'next' config), StratifiedGroupKFold(seed=42) on userid stratified by target_category. Adam(lr=7e-4, betas=(0.8, 0.9)), batch 400, early-stop patience 3.

### `faithful` — paper-architecture from raw inputs

| State | Acc@1 | Acc@3 | Acc@5 | MRR | macro-F1 | weighted-F1 | n_rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| AL | 39.35 ± 1.10 | 79.93 ± 0.39 | 92.53 ± 0.70 | 61.27 ± 0.70 | 18.95 ± 0.71 | 33.84 ± 0.66 | 12,175 |
| AZ | 41.23 ± 1.23 | 81.26 ± 0.61 | 93.07 ± 0.36 | 62.86 ± 0.76 | 24.99 ± 0.85 | 37.17 ± 1.03 | 25,394 |
| FL | 43.83 ± 0.17 | 78.91 ± 0.26 | 92.92 ± 0.14 | 63.53 ± 0.18 | 32.06 ± 0.23 | 40.85 ± 0.17 | 161,086 |
| CA | 40.40 ± 0.40 | 78.09 ± 0.17 | 92.10 ± 0.12 | 61.52 ± 0.26 | 29.13 ± 0.71 | 37.18 ± 0.51 | 367,523 |
| TX | 39.43 ± 0.28 | 78.43 ± 0.24 | 92.78 ± 0.08 | 60.97 ± 0.20 | 29.91 ± 0.43 | 36.88 ± 0.35 | 477,039 |
| GA | 39.13 ± 0.83 | 78.69 ± 0.60 | 92.13 ± 0.48 | 60.84 ± 0.52 | 27.62 ± 0.97 | 36.30 ± 0.77 | 43,941 |

## Per-baseline detail — POI-RGNN

5-fold × 35-epoch, StratifiedGroupKFold(seed=42) on userid stratified by target_category. Adam(lr=1e-3, betas=(0.8, 0.9), eps=1e-7), batch 400, ReduceLROnPlateau(patience=3, factor=0.5) on val F1, early-stop patience 10.

### `faithful` — paper-architecture from raw inputs

| State | Acc@1 | Acc@3 | Acc@5 | MRR | macro-F1 | weighted-F1 | n_rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| AL | 39.21 ± 1.97 | 80.40 ± 0.93 | 93.36 ± 0.43 | 61.43 ± 1.38 | 23.80 ± 1.12 | 36.56 ± 1.74 | 10,749 |
| AZ | 41.04 ± 1.79 | 81.84 ± 1.30 | 93.41 ± 0.59 | 62.86 ± 1.31 | 27.64 ± 2.34 | 38.09 ± 2.09 | 22,396 |
| FL | 44.07 ± 0.76 | 79.05 ± 0.68 | 93.07 ± 0.33 | 63.72 ± 0.62 | 33.35 ± 1.14 | 41.54 ± 0.93 | 142,381 |
| CA | 41.32 ± 0.52 | 78.65 ± 0.28 | 92.49 ± 0.16 | 62.21 ± 0.34 | 30.71 ± 0.82 | 38.45 ± 0.63 | 325,119 |
| TX | 40.59 ± 0.50 | 79.11 ± 0.16 | 93.18 ± 0.08 | 61.79 ± 0.32 | 32.08 ± 0.70 | 38.62 ± 0.58 | 422,534 |
| GA | 39.89 ± 0.71 | 80.21 ± 0.52 | 93.04 ± 0.14 | 61.63 ± 0.47 | 30.24 ± 0.87 | 37.86 ± 0.72 | 38,775 |

## Markov-K-cat detail (apples-to-apples sequence floor)

K-step Markov over the last K categories of the 9-step input window, stupid-backoff (K → K-1 → … → 1 → unigram). Same StratifiedGroupKFold splits and same row alignment as POI-RGNN.

### Macro-F1 (mean ± σ)

| State | k=1 | k=3 | k=5 | k=7 | k=9 |
|---|---:|---:|---:|---:|---:|
| AL | 10.01 ± 2.02 | 20.19 ± 1.01 | **20.50 ± 0.67** | 19.73 ± 0.74 | 19.27 ± 0.59 |
| AZ | 12.61 ± 0.19 | 23.79 ± 2.03 | **23.92 ± 2.26** | 22.45 ± 2.10 | 22.01 ± 2.03 |
| FL | 23.98 ± 1.22 | **29.74 ± 1.19** | 29.55 ± 1.10 | 27.63 ± 0.73 | 26.65 ± 0.66 |
| CA | 19.97 ± 1.77 | 27.06 ± 0.60 | **27.58 ± 0.61** | 25.47 ± 0.43 | 24.18 ± 0.37 |
| TX | 18.17 ± 1.38 | 27.94 ± 0.60 | **28.67 ± 0.66** | 26.26 ± 0.56 | 24.62 ± 0.43 |
| GA | 20.59 ± 0.77 | **27.01 ± 1.10** | 25.81 ± 0.84 | 23.86 ± 0.39 | 23.29 ± 0.23 |

### Acc@1 (mean ± σ)

| State | k=1 | k=3 | k=5 | k=7 | k=9 |
|---|---:|---:|---:|---:|---:|
| AL | 35.45 ± 0.80 | **36.72 ± 1.22** | 33.90 ± 0.63 | 31.95 ± 0.82 | 30.78 ± 0.52 |
| AZ | 37.25 ± 0.48 | **39.26 ± 1.37** | 35.99 ± 1.38 | 32.83 ± 1.52 | 31.86 ± 1.45 |
| FL | 37.99 ± 0.48 | **42.03 ± 0.64** | 39.90 ± 0.71 | 36.40 ± 0.56 | 34.60 ± 0.53 |
| CA | 36.44 ± 0.22 | **39.44 ± 0.47** | 38.00 ± 0.46 | 33.85 ± 0.37 | 31.48 ± 0.31 |
| TX | 34.08 ± 0.27 | **37.99 ± 0.23** | 37.15 ± 0.41 | 33.47 ± 0.40 | 30.89 ± 0.34 |
| GA | 34.87 ± 0.85 | **37.55 ± 0.92** | 34.45 ± 0.83 | 31.49 ± 0.40 | 30.60 ± 0.20 |

K=3–5 is the sweet spot — beyond that, key sparsity (7^K vs ~10K–150K windows) eats the gain even with backoff.

## Pattern summary

| State | POI-RGNN F1 | MHA+PE F1 | POI-RGNN − best Markov-K-cat | MHA+PE − best Markov-K-cat |
|---|---:|---:|---:|---:|
| AL | **23.80** | 18.95 | +3.30 | **−1.55** |
| AZ | **27.64** | 24.99 | +3.72 | +1.07 |
| FL | **33.35** | 32.06 | +3.61 | +2.32 |
| CA | **30.71** | 29.13 | +3.13 | +1.54 |
| TX | **32.08** | 29.91 | +3.41 | +1.24 |
| GA | **30.24** | 27.62 | +3.23 | +0.61 |

- **POI-RGNN beats MHA+PE on macro-F1 across all 3 states** by +2.65 (AZ) to +4.85 (AL) pp. The RNN+GNN combo's category-transition graph buys real F1 over the RNN+self-attention combo at this scale.
- **MHA+PE underperforms the count-based Markov-K-cat at AL** (−1.55 pp) and only marginally beats it at AZ (+1.07 pp). The transformer architecture without a graph signal struggles to extract minority-class structure beyond what stupid-backoff N-grams already capture.
- **At FL scale** (160K windows), MHA+PE catches up: F1 32.06 vs POI-RGNN 33.35 (gap shrinks to 1.29 pp), suggesting MHA+PE benefits more from data scale than POI-RGNN does — consistent with attention-based models being data-hungry.
- **Acc@1 is essentially tied** between the two architectures at all states (within 0.2 pp); the F1 gap is driven by minority-class recall, not top-1 dominance.

- **POI-RGNN's honest lift is +3.3–3.7 pp macro-F1** over the best Markov-K-cat floor — much smaller than the +6–8 pp vs Markov-1-POI suggested. The RNN+GNN still wins, but the win is modest and state-invariant.
- **Acc@1 lift over best Markov-K-cat is +1.8–2.5 pp** — the model's added Acc@1 over a same-context count-based floor is small, suggesting most of the dominant-class signal is recoverable from N-gram statistics.
- **Markov-K saturates at K=3–5** then degrades from sparsity, even with stupid backoff. Adding more context past K=5 hurts the count-based floor, which is exactly where the neural model's representation learning earns its keep.
- **Acc@5 plateaus around 93%** (vs ~88% top-k popular floor) — most of the cardinality is captured at top-5 even at AL scale.

## Variants we run (POI-RGNN)

Only `faithful` for now. STL substrate variants (`stl_check2hgi`, `stl_hgi`) are deferred — see `poi_rgnn.md` §"Variants we run" for context.

## Substrate-head matched STL — Phase 1 (next_gru, MTL B3 cat head)

Quick reference for the substrate-comparison Phase-1 matched-head cat results. Authoritative source: [`../../research/SUBSTRATE_COMPARISON_FINDINGS.md`](../../research/SUBSTRATE_COMPARISON_FINDINGS.md). Per-fold JSONs: `../../results/phase1_perfold/{AL,AZ}_{check2hgi,hgi}_cat_gru_5f50ep.json`. Paired tests: `../../results/paired_tests/{alabama,arizona}_cat_f1.json`.

| Substrate | Variant | AL macro-F1 | AZ macro-F1 | FL | CA | TX | GA |
|---|---|---:|---:|---:|---:|---:|---:|
| Check2HGI | matched-head `next_gru` STL | **40.76 ± 1.68** | **43.21 ± 0.87** | **63.43 ± 0.98** | **59.94 ± 0.59** | **60.24 ± 1.84** | ⚪ |
| HGI | matched-head `next_gru` STL | 25.26 ± 1.18 | 28.69 ± 0.79 | 34.41 ± 1.05 | 31.13 ± 1.04 | 31.89 ± 0.55 | ⚪ |
| **Δ (C2HGI − HGI)** | paired Wilcoxon | **+15.50 (p=0.0312)** | **+14.52 (p=0.0312)** | **+29.02 (p=0.0312)** | **+28.81 (p=0.0312)** | **+28.34 (p=0.0312)** | ⚪ |

CH16 confirmed at 5/5 states with paper-grade significance (Wilcoxon p=0.0312 = max-n=5, 5/5 folds positive each). Δ scales monotonically from ~15 pp at AL/AZ to ~28-29 pp at FL/CA/TX. ⚪ = GA was scoped to external-baseline coverage (POI-RGNN / MHA+PE / floors), not the substrate-comparison axis. Head-sensitivity probes (`next_single`, `next_lstm`) and the head-free linear probe replicate the direction with all 8 head-state probes positive at max-significance n=5 paired Wilcoxon. Authoritative aggregate: [`../../FINAL_SURVEY.md §2`](../../FINAL_SURVEY.md). See FINDINGS §5.1.
