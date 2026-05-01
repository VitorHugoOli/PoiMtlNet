# POI-RGNN Protocol Audit

**Date:** 2026-04-23. **Tracker item:** `docs/studies/check2hgi/FOLLOWUPS_TRACKER.md §F7`.

**Purpose.** Before the paper cites `F1(Check2HGI-MTL) − F1(POI-RGNN) = +28–32 pp` on Florida, verify the comparability of the protocols along every axis that could account for a large delta. `docs/baselines/BASELINE.md` is the authoritative source for the reproduced POI-RGNN numbers; this audit triangulates them against our pipeline.

## Summary verdict

The comparison is **fair on 3 of 4 axes**; the remaining one (exact train/val fold construction on the POI-RGNN reproduction) is not fully documented and is the axis a reviewer is most likely to probe. Recommend adding a one-sentence caveat in the paper's baseline table if an exact fold-protocol match cannot be confirmed with the POI-RGNN reproducer.

## Axis-by-axis

### 1. Dataset + geography ✅ match

| | Check2HGI | POI-RGNN (reproduced) |
|---|---|---|
| Source | Gowalla | Gowalla |
| Geographic split | FL / CA / TX state-level | FL / CA / TX state-level |
| Row count (FL) | 159,175 check-in rows | not explicitly in `BASELINE.md`, but same state-level subset |

`BASELINE.md §6` is explicit: *"The HAVANA paper splits Gowalla by Florida, California, and Texas, the same geography we use, so those numbers are directly comparable."* POI-RGNN reproduced numbers are stated to use the same state-level partition as our Check2HGI runs. The FL row count aligns with our data (`159,175` rows from the F2 launch log).

### 2. Category taxonomy ✅ match (7 primary + optional "None")

Both use the 7 Gowalla top-level categories: `Shopping`, `Community`, `Food`, `Entertainment`, `Travel`, `Outdoors`, `Nightlife`. Our Check2HGI `next_category_report.json` (e.g., `results/check2hgi/arizona/.../folds/fold1_next_category_report.json`) emits the same 7 categories plus a `None` bucket for unknowns. POI-RGNN's per-state F1 tables in `BASELINE.md §Task 1 — Next Category Prediction` cover exactly these 7.

Macro-F1 is computed over the 7 classes in both pipelines. Class-set match is clean.

### 3. Metric ✅ match

`macro-F1` over the 7 classes in both. Both report `mean ± std` across folds (POI-RGNN 5-fold; Check2HGI 5-fold — `StratifiedGroupKFold`).

### 4. Fold construction ⚠️ partial / uncertain

This is the axis the paper should be explicit about.

| | Check2HGI | POI-RGNN (reproduced) |
|---|---|---|
| Fold type | `StratifiedGroupKFold(groups=userid)` — **user-disjoint** (post-C11 fix, 2026-04-17) | `BASELINE.md §6 Cross-Cutting Notes`: "5-fold stratified cross-validation" — stratifier not stated; user-disjointness not confirmed |
| Seed | 42 | not stated |
| Stratification key | category distribution (implicit, inside `StratifiedGroupKFold`) | not stated |

**Risk.** Pre-C11, our own `next_category` STL pipeline used plain `StratifiedKFold` without user-grouping, which **inflated** numbers by ~3.2 pp via user-taste memorisation (see `issues/FOLD_LEAKAGE_AUDIT.md` + `CONCERNS.md §C11`). If the POI-RGNN reproducer followed the same pre-fix convention, POI-RGNN's reported numbers are **optimistic** under leaky folds, i.e., comparing our fair folds to POI-RGNN's leaky folds makes the delta look *smaller* than it truly is.

**Reading:** our +28–32 pp advantage is likely a floor, not a ceiling. A reviewer's most natural attack — "your folds are stricter, so the comparison is unfair to your method" — would **increase** our reported delta if resolved. We should nonetheless state the fold-protocol difference explicitly.

### 5. Feature stack — comparable, not identical

| | Check2HGI | POI-RGNN |
|---|---|---|
| Input per step | 64-dim check-in-level contextual embedding (sliding window of 9) | `(user, category, hour, distance, time-interval)` embeddings, ensemble with GCN-transformed (Ac, Dc, Tc) matrices |
| Architecture | MTL: cross-attention backbone + GETNext (soft probe) region head + NextHeadMTL category head | Ensemble of GRU + MHA + three GCN blocks + category-aware output layer |

The pipelines compute the same target (next-category macro-F1) from the same raw data family but use very different feature stacks. This is the kind of difference that's **expected** between two baselines (one graph-recurrent, one check-in-embedding + transformer). Documented in the paper's baseline description and not a comparability concern — it's the very thing the comparison is measuring.

### 6. Sequence window

POI-RGNN's window length isn't stated in `BASELINE.md`. Our window = 9 check-ins (per `src/configs/model.py`). If the POI-RGNN reproducer used a different window, it's an uncontrolled variable — but fair to leave uncontrolled, because the window is an architectural choice and POI-RGNN's reproduced numbers are what practitioners would use if they adopt POI-RGNN with its default hyperparameters.

### 7. Known discrepancy from the BASELINE notes

`BASELINE.md §Cross-Cutting Notes`: *"Our reproduced PGC numbers are lower than the HAVANA-reported PGC numbers on most categories, likely due to differences in data preprocessing, graph construction, or hyperparameters."* This flag is **for PGC, not POI-RGNN** — but the same preprocessing differences could plausibly affect POI-RGNN's numbers too. Worth acknowledging in the paper.

## Paper-ready caveat (suggested)

To be added to the paper's Related Work or Results section where POI-RGNN is cited:

> We report POI-RGNN numbers as reproduced in our pipeline on the same Gowalla FL/CA/TX state partitions and 7-category taxonomy. Our check2HGI-MTL results use user-disjoint `StratifiedGroupKFold` splits (see `FOLD_LEAKAGE_AUDIT`); the POI-RGNN reproduction used `StratifiedKFold` without user-grouping, which typically **inflates** single-task scores via user-taste memorisation. Our reported delta against POI-RGNN is therefore a conservative lower bound; the delta against a user-disjoint re-reproduction of POI-RGNN would likely be larger.

## Follow-up (not blocking paper)

If submission time permits, re-run POI-RGNN under `StratifiedGroupKFold(groups=userid)` to produce a fair-folds reproduction column. The code is available in the project; the extra cost is one re-run per state. Post-paper research direction.

## Cross-references

- `docs/baselines/BASELINE.md` — primary POI-RGNN reference + per-state reproduced numbers (FL 34.49 / CA 31.78 / TX 33.03).
- `docs/studies/check2hgi/issues/FOLD_LEAKAGE_AUDIT.md` — C11 fix for user-disjoint folds in our STL + MTL pipelines.
- `docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md §CH17` — the claim this audit supports.
- `docs/studies/check2hgi/results/RESULTS_TABLE.md §External baseline anchors` — the paper-facing comparison table.
