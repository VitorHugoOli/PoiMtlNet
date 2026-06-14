# Evaluation Protocol Review — Datasets, Splits, Metrics, Leakage

> Compiled 2026-06-12 from code audit (`src/data/folds.py`, `src/data/inputs/`, `src/training/runners/mtl_cv.py`, `scripts/compute_region_transition.py`), the concerns/issues registries (`docs/CONCERNS.md`, `docs/issues/check2hgi/`), and the field-norm survey in [`literature_review.md §2`](literature_review.md).

---

## 1. Protocol as implemented

- **Data**: Gowalla (SNAP; Cho et al., KDD 2011), five US-state corpora (AL ~10k, AZ ~26k, FL ~127k, CA ~230k, TX ~187k check-ins; users ≥5 check-ins). Categories merged to **7 root classes**. Regions = **TIGER census tracts** (~1.1k AL … ~8.5k CA). Note: "H3-alt" in the docs is a *training-recipe* name, not H3 cells.
- **Sequences**: non-overlapping length-9 windows per user, chronological within user, padded; target = next check-in (`src/data/inputs/core.py`).
- **Splits**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed, groups=userid)` — **user-disjoint, random (not temporal)** fold assignment (`src/data/folds.py`). Single-task path had a user-leakage bug fixed 2026-04-17 (C11).
- **Model selection**: early stopping + checkpoint selection on the **same validation fold that is reported**; no held-out test set. §0.1 reports **per-task diagnostic-best epochs** (two different checkpoints per fold); the deployable joint checkpoint uses the `geom_simple` selector (C21 fix, default since 2026-06-03).
- **Metrics**: category = macro-F1 (7 classes); region = Acc@10 (plus Acc@3/5, MRR, NDCG@10) over the full tract vocabulary (`src/tracking/metrics.py`).
- **Seeds**: development seed 42 quarantined; reporting = {0,1,7,100} × 5 folds = n=20 per state; paired Wilcoxon per state.
- **Transition prior**: per-fold, per-seed, train-edges-only `region_transition_log_seed{S}_fold{N}.pt` (C4/C19 fixes).

---

## 2. Comparison with field norms

| Dimension | Field norm (next-POI / embedding literature) | This repo | Assessment |
|---|---|---|---|
| Split | **Temporal** per user (80/10/10 chronological, first-80/last-20, or leave-last-out). No k-fold random CV anywhere in the canon | Random user-disjoint 5-fold CV | **Largest deviation.** The repo's protocol is a *cold-user* (new-user generalization) evaluation — arguably harder along the user axis, but it never tests forward-in-time generalization, and it makes every published number non-comparable |
| Test users | Seen in training (warm-user) | Unseen (user-disjoint) | Defensible, even stronger on one axis — but must be stated prominently, and it explains why faithful baselines (POI-RGNN) score below their published numbers |
| Dataset | FSQ NYC/TKY + Gowalla-CA trio; Massive-STEPS emerging | Custom Gowalla US-state cuts | No external anchor; nobody can rank this work against the literature |
| Category label space | 184–400 categories, Acc@K/MRR ranking | 7 merged classes, macro-F1 | No published comparison point; 7-class macro-F1 ceilings are not interpretable against the literature |
| Region definition | Geohash/grid/quadkey (HMT-GRN: G@2–G@6) | TIGER census tracts | No precedent in this canon; also makes region cardinality uneven across states (1.1k–8.5k) |
| Embedding substrate protocol | Pre-train on **training portion only**, freeze, feed to downstream predictors (CTLE/UniTE template) | Pre-train on the **full state corpus** (transductive), then fold downstream | **Deviation with leakage implications** — §4.1 |
| Model selection | Val split distinct from test split | Selection and reporting on the same fold | Optimistic bias; common in older literature but a real weakness — §4.2 |

---

## 3. Leakage risks already documented and fixed (credit where due)

The repo's concerns registry is unusually rigorous. Documented & resolved:

| ID | Issue | Status |
|---|---|---|
| C4 | Full-data `region_transition_log.pt` leaked ~13–27 pp into the reg prior | Fixed — per-fold per-seed train-only log_T; all §0 numbers leak-free |
| C11 | STL next-task folds lacked user grouping (≤3.2 pp inflation) | Fixed 2026-04-17, `StratifiedGroupKFold` |
| C19 | `--folds N<5` × 5-fold log_T mismatch leaked ~30% val users | Fixed 2026-05-15 |
| C21 | Joint selector `0.5·(cat_F1+reg_F1)` broken on sparse-region macro-F1 (−10.7 pp deployable reg) | Fixed 2026-06-03 (`geom_simple`); §0.1 diagnostic-best unaffected |
| C25 | MTL heads trained on class-weighted CE while metrics unweighted (−10–14 pp reg, −3–5 pp cat) | Fixed in v15/v16; **v11 paper numbers still carry it** |
| C28 | Development-seed bias (+3…+8 pp at CA/TX) and sklearn-version fold drift (±2–3 pp cross-phase) | Mitigated by reporting-seed protocol; within-phase pairs unaffected |
| — | Stale log_T survival across regens (+8–12 pp) | Documented preflight in `CLAUDE.md`; T6 sweep caveated |

Also documented honestly: the F49 λ=0 isolation is unsound under cross-attention (gradients flow through K/V even at zero loss weight), and the B9 recipe's `--category-weight 0.75` is a **silent no-op** under `--alternating-optimizer-step` (B9 is effectively 50/50 alternating single-task steps; 2026-06-12 audit note in `docs/results/CANONICAL_VERSIONS.md`). The paper must describe the mechanism accordingly.

---

## 4. Open risks (this review's findings)

### 4.1 Transductive substrate — the most material open issue

`research/embeddings/check2hgi/check2hgi.py` trains a single model on **all** check-ins of a state (best epoch by *training* loss, no validation), then embeddings for all check-ins — including those that later land in validation folds — are extracted from that model. Validation check-ins' features and sequence edges were in the training graph.

- **Internally** the comparison is approximately fair: STL/MTL arms and the HGI comparator all consume full-corpus substrates, so *relative* substrate claims (CH16/CH15) survive.
- **Two places it is not fair**: (a) versus **external baselines that never see the substrate** (STAN-faithful from raw trajectories, ReHDM-faithful) — substrate arms enjoy information external arms don't have, inflating any "ours vs theirs" rows; (b) versus the **field's pre-train-on-train-only convention** (CTLE), which means the substrate numbers are not protocol-equivalent to published embedding papers.
- **Severity is unmeasured.** User-disjoint folds blunt the user-level channel, but the GCN propagates val-user signal into POI/region structure. Nobody has quantified the inflation.
- **Recommendation**: run a **train-users-only substrate ablation** (retrain Check2HGI per fold excluding val users, ≥1 state × 1 seed) and report the delta. If small, one paragraph defuses the entire issue; if large, it must be disclosed and the headline numbers re-anchored. This is a pre-freeze gate candidate for `closing_data`.

### 4.2 Selection-on-reporting-fold

Early stopping and checkpoint selection use the same fold that produces the reported numbers, and §0.1 reports per-task *best-epoch* diagnostics (two checkpoints per fold). This is an optimistic protocol: it reports validation ceilings, not deployable performance. Mitigations: (a) always pair §0.1 with the joint-checkpoint (geom_simple) numbers and label both clearly; (b) ideally, nested selection (select epoch on 4 folds, report on the 5th) or a per-state holdout for at least one confirmation run.

### 4.3 Metric/label-space comparability

7-class macro-F1 and census-tract Acc@10 have no external anchors. Add bridging metrics (category Acc@1, region Acc@1/@5/MRR — already computed) to every paper table, and consider a geohash-grid secondary region definition at one state to connect to HMT-GRN-style results.

### 4.4 Window construction

Non-overlapping stride-9 windows discard up to 8/9 of possible training transitions and make the "next" target depend on window phase. The literature standard (all prefixes / sliding stride-1 with chronological split boundaries) extracts more supervision. Not a validity issue (consistent across arms), but a potential under-training of all arms and a reviewer question; worth one ablation or one sentence of justification.

### 4.5 Category feature circularity — checked, bounded, but worth one sentence in the paper

Check-in node features include the check-in's own category one-hot. Downstream, the *inputs* are the 9 past check-ins' embeddings and the *target* is the next check-in's category/region — so there is no direct target leakage (using past categories as features is standard, cf. GETNext's category embeddings). But it does mean the substrate-vs-HGI category gap partly measures *feature access*, not representation learning — see the Tier-1 feature-concat control in [`baseline_gap_analysis.md §2`](baseline_gap_analysis.md).

### 4.6 The paper-vs-repo inconsistency

The v11 paper narrative ("MTL pays 7–17 pp on region") is built on numbers the repo's own C25 finding shows were objective-mismatched, and the champion G result inverts the story at 4 states. Publishing the v11 narrative as-is would be publishing a claim the authors' own evidence contradicts. The restatement (or an explicit confound footnote strategy) is mandatory before submission.

---

## 5. Recommendations, prioritized

1. **Restate the paper under v16/C25-fixed numbers** (or explicitly scope it to the substrate claims and footnote the MTL confound). Blocking.
2. **Quantify the transductive-substrate effect** with a per-fold train-only substrate ablation (1 state × 1 seed minimum). Blocking for any external-baseline comparison row.
3. **Add a temporal-split bridge**: re-run champion + STL ceilings + Markov floor under a per-user chronological 80/10/10 split at 1–2 states. This single experiment connects the entire study to the field's protocol and tests warm-user/forward-time generalization the current protocol never touches.
4. **Report deployable (joint-checkpoint) numbers alongside diagnostic-best everywhere**; consider nested model selection for one confirmation pass.
5. **Add bridging metrics** (cat Acc@1; reg Acc@1/@5/MRR) to all paper tables; state the cold-user nature of the protocol prominently.
6. Disclose in the paper: alternating-step weight no-op, transductivity, sklearn fold-drift caveat, development-seed quarantine (the last two are already drafted in `STATISTICAL_AUDIT.md`).
