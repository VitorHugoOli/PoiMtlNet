# Check2HGI MTL Branch — Research Overview & Decisions

**Status:** scaffold, pre-implementation. Approval gate for `CHECK2HGI_MTL_BRANCH_PLAN.md`.
**Purpose:** consolidate dataset/metric/enrichment evidence into a decision record so implementation starts from a verified scope, not assumptions.

---

## 1. Datasets: what we use vs what the field uses

### 1.1 What the repo uses today

Source: `src/configs/paths.py:47–52`, `docs/studies/HANDOFF.md:19–29`, `docs/baselines/BASELINE.md`.

- **Gowalla LBSN check-ins**, state-split: Alabama (primary), Florida, Texas, California, Arizona, Georgia.
- Schema: `userid, placeid, datetime, category, latitude, longitude`.
- 7-category taxonomy: Community, Entertainment, Food, Nightlife, Outdoors, Shopping, Travel (imbalanced: Food ~32.5%, Shopping ~31.3%, Community ~15%, rest ≤7%).
- Input files: `data/checkins/{State}.parquet`.

**Embedding availability (as of 2026-04-14):**

| State | HGI | Check2HGI | Fusion | Sphere2Vec | Time2Vec |
|-------|:---:|:---------:|:------:|:----------:|:--------:|
| Alabama | ✓ | **✗** | ✓ | ✓ | ✓ |
| Florida | ✓ | **✗** | ✓ | ✓ | ✓ |
| Texas | ✗ | ✗ | ✗ | ✗ | ✗ |
| Arizona/CA/GA | ✗ | ✗ | ✗ | ✗ | ✗ |

> **Blocker surfaced during research:** **Check2HGI embeddings are not yet generated for any state.** The original branch plan assumed they existed. This adds a prerequisite step.

### 1.2 What the field uses

Source: web survey, `docs/issues/LITERATURE_SURVEY_POI_METRICS.md`.

- **Dominant:** Foursquare-NYC (FSQ-NYC), Foursquare-Tokyo (FSQ-TKY), Gowalla-CA. These three appear in ~50% of surveyed next-POI papers.
- Typical preprocessing: min 10 check-ins per user, min 10 per POI, 24-hour trajectory cuts, 80/10/10 time-ordered split.
- Newer cross-city benchmark: Massive-STEPS (12 cities), 2025.

### 1.3 Alignment and implications

- Gowalla state-split is **not** the mainstream benchmark, but it matches the Brazilian lineage (POI-RGNN, HAVANA, PGC) the repo already compares against. That lineage uses Gowalla-FL/CA/TX, so our current comparison surface is coherent on its own terms.
- We are **not required** to switch to FSQ-NYC/TKY for the branch plan — doing so would cost an embedding-regeneration pass across all engines and invalidate existing HGI/Sphere2Vec baselines.
- **Decision:** keep Gowalla state-split. Generate check2HGI embeddings for **Florida + Alabama only** (the two states with complete other-engine coverage). Defer Texas/AZ/CA/GA per existing P0 plan.

---

## 2. Metrics: what we use vs what the field uses

Source: `docs/issues/LITERATURE_SURVEY_POI_METRICS.md`, `docs/issues/POI_RELATED_WORK_METRICS.md`, `src/training/shared_evaluate.py:38–44`.

### 2.1 The task-framing distinction

The literature splits into **two sub-communities** that use very different metrics:

| Framing | Cardinality | Dominant metrics | Example papers |
|---|---|---|---|
| Next-POI recommendation (predict exact POI id) | 10³–10⁶ | Recall@K, NDCG@K, MRR, Acc@K | STAN, GETNext, LSTPM, Graph-Flashback |
| Next-POI category prediction (predict category) | 7 | Macro-F1, per-class F1, Accuracy | MHA+PE, iMTL, MCARNN, POI-RGNN |
| Semantic venue annotation (classify a POI) | 7–50 | Macro-F1, Accuracy | HMRM, HAVANA, PGC, TME |

Our current "next" task is actually **next-category**, not next-POI-id. Macro-F1 is therefore correctly aligned.

### 2.2 The next-region head: new metric question

Adding `next_region` as a second head changes the picture because region cardinality is **mid-to-high** (hundreds of census tracts per state — in between 7 categories and thousands of POIs).

Evidence from the hierarchical MTL lineage that's closest to our setup:

- **HMT-GRN** (SIGIR '22): reports **Acc@K + MRR** for each level of the hierarchy (region, POI).
- **MGCL** (Frontiers '24): Acc@K + MRR for next-region.
- **Bi-Level Graph Structure Learning** (arXiv '24): Acc@K + MRR.
- **"Learning Hierarchical Spatial Tasks"** (TORS '24): Acc@K + MRR.

None of these use macro-F1 as the primary metric for region — the label space is too large and sparse for class-balanced F1 to be the right summary statistic.

### 2.3 Decision

| Head | Primary metric(s) | Secondary | Rationale |
|---|---|---|---|
| `next_category` (kept) | macro-F1, per-class P/R/F1 | Accuracy, Acc@{1,3,5} (bridge) | Matches POI-RGNN / MHA+PE / HAVANA lineage. Optional Acc@K for cross-community bridge. |
| `next_region` (new) | **Acc@{1,5,10}, MRR** | Macro-F1 (for completeness) | Matches HMT-GRN / MGCL lineage. Region cardinality makes ranking metrics canonical. |

**Joint score (open design choice — flagged):**
Mixing macro-F1 (class-balanced, imbalance-robust) with Acc@1 (frequency-weighted) in a single mean is fragile — the two scales don't cancel under class skew. Three options:

- (a) **Acc@1 for both in the joint score** (per-head primary reporting unchanged). Clean, no scale mismatch; but loses the imbalance-robustness signal on the category side.
- (b) **Macro-F1 for both in the joint score** — compute macro-F1 on the region head even though ranking metrics are primary. Consistent scale; but region macro-F1 is the less-informative view.
- (c) **Report both joint scores** (`joint_acc1 = mean(acc1_cat, acc1_region)`, `joint_f1 = mean(f1_cat, f1_region)`) and use one as the checkpoint monitor — recommendation: `joint_acc1` for monitor, `joint_f1` reported alongside.

Defaulting to **(c)** in the plan; flagged for user sign-off.

This forces a small addition: `src/training/shared_evaluate.py` (or a sibling module for the generic runner) must learn to compute Acc@K and MRR from logits. These are cheap and standard — `torch.topk` + a reciprocal-rank reduction.

**Note:** we do NOT switch the next-category head's primary metric away from macro-F1. Keeping it preserves continuity with the legacy regression-floor tests and the POI-RGNN/HAVANA comparison surface.

---

## 3. Check2HGI enrichment — now vs later

Source: `docs/issues/CHECK2HGI_ENRICHMENT_PROPOSAL.md`.

The enrichment proposal has four phases:

| Phase | What | Touch surface | Expected lift (estimate) |
|---|---|---|---|
| 1 | Learnable temporal embedding (Time2Vec-like); spatial positional encoding; dwell-time / delta-t / distance features | `preprocess.py::_build_node_features` | Medium — analogous to what Time2Vec/Sphere2Vec add as standalone engines |
| 2 | New edge families (KNN spatial, temporal-window, revisit-strength) | `preprocess.py` + edge builders | Small–medium, variance-sensitive |
| 3 | Encoder-side auxiliary pretext tasks (incl. next_region at the encoder) | `Check2HGIModule.py`, `check2hgi.py` | Small–medium, couples encoder to downstream signal |
| 4 | Multi-view contrastive + hard negatives | Module + trainer | Unknown, research-grade |

### 3.1 Relationship to the MTL task switch

Phases 1–2 change **what embeddings come out** of check2HGI. Phase 3 adds encoder-side pretext tasks, one of which is *also* called "next_region" — but it operates on the encoder, not the downstream MTL head. These two "next_region" tasks are **complementary, not redundant**:

- **Head-side next_region** (this branch plan): adds a classification head to the MTL model that receives a sequence of check-in embeddings and predicts the next region. Pure downstream change.
- **Encoder-side next_region** (enrichment Phase 3): trains the check2HGI encoder itself with a next-region pretext signal, aiming for better embeddings.

### 3.2 Recommendation

**Defer all enrichment phases. Run vanilla check2HGI first.** Reasons:

1. **Prerequisite order.** We need a vanilla check2HGI baseline before we can claim any enrichment delivers lift. Without it, enrichment experiments measure "enrichment + new task" jointly and cannot be attributed.
2. **Budget.** Each enrichment phase is a research-grade change with its own ablation table. Phase 1 alone probably exceeds the MTL branch's implementation budget and would dilute the paper's story.
3. **Scope discipline.** The branch is about *task switch* (category→region in MTL). Bundling encoder changes muddies the claim. Paper readers should be able to see "2-task MTL on check2HGI with next_region" as a clean contribution.
4. **Risk isolation.** If the vanilla run doesn't show lift from next_region, we learn that before investing in encoder enrichment. If it does, enrichment becomes a sharpening step with a known-good baseline.

**Exception considered, then rejected:** we briefly considered bundling Phase 1 (temporal + spatial features) since they're cheap. Rejected because any embedding change invalidates the very comparison we're trying to make, and the regeneration pass is non-trivial.

---

## 4. Revised scope for the branch

Delta vs the original `CHECK2HGI_MTL_BRANCH_PLAN.md`:

1. **Add a pre-step P-1: "Generate check2HGI embeddings for Florida + Alabama"** using the existing `pipelines/embedding/check2hgi.pipe.py` (vanilla, no enrichment). This is the surfaced prerequisite.
2. **Add metric module for the generic runner**: compute Acc@{1,5,10} + MRR from logits for the `next_region` head. Primary metric for `next_region` is Acc@1 + MRR; macro-F1 stays secondary.
3. **Keep the 2-task decision** (`next_category` + `next_region`). Next_time_gap stays scaffolded but inactive.
4. **Freeze-in-place** for legacy `docs/studies/*` (per prior decision).
5. **Enrichment deferred** to a follow-up branch/study after the vanilla baseline is established.

### Revised phase order

| Phase | What | Artifact |
|---|---|---|
| **P-1** | Generate vanilla check2HGI for FL + AL | `output/check2hgi/{state}/{embeddings,poi_embeddings,region_embeddings}.parquet` |
| **P0** | Build data pipeline for `next_region` labels (join `placeid → region_idx`) | `output/check2hgi/{state}/input/next_region.parquet` |
| **P1** | Scaffold `TaskSpec` / `TaskSetSpec` / `MTLnetGeneric` / `mtl_cv_generic` / `FoldCreatorGeneric` | new code + tests |
| **P2** | Add Acc@K + MRR to the generic eval module | `src/training/ranking_metrics.py` (new) |
| **P3** | Run 2-task MTL `{next_category, next_region}` on FL then AL | results + claim entries |
| **P4** | Ablations: head choice, task-embedding init, loss family | tables |
| **P5 (deferred)** | Enrichment (time/space node features, encoder pretext) | out-of-scope for this branch |

### Docs/studies subtree

Create `docs/studies/check2hgi/` as planned. Include a top-level note that:
- The vanilla-check2HGI baseline is the central claim.
- Enrichment is explicitly deferred (with a pointer to this overview doc).
- Metric convention for `next_region` is Acc@K + MRR (not macro-F1), following HMT-GRN / MGCL.

---

## 5. Open questions for the user

1. **Next_region label granularity.** The check2HGI graph artifact has a `placeid → region_idx` map. Do we know the region cardinality per state? If Florida has, say, 500 regions and Alabama has 300, Acc@10 is probably saturated. We should log `n_regions` in P-1 and decide K at that point; my tentative choice is K={1, 5, 10}.
2. **Vanilla check2HGI — train from scratch or use an existing checkpoint?** The pipeline exists but hasn't been run. Is there a known-good config we should use, or should P-1 also include a brief hyperparameter sanity pass?
3. **Comparison baselines for next_region.** The literature baselines (HMT-GRN) use FSQ-NYC/TKY. Our next_region numbers won't be comparable to HMT-GRN directly (different dataset). Options: (a) present as internal comparison only (vanilla-HGI next_region head vs vanilla-check2HGI next_region head); (b) add a FSQ-NYC slice later. Recommendation: (a) for this branch; note the gap in the paper limitations section.

## 6. Sources

- `docs/issues/LITERATURE_SURVEY_POI_METRICS.md` (in-repo) — authoritative metric survey.
- `docs/issues/POI_RELATED_WORK_METRICS.md` (in-repo) — metric rationale.
- `docs/issues/CHECK2HGI_ENRICHMENT_PROPOSAL.md` (in-repo).
- `docs/issues/MTL_TASK_REPLACEMENT_PROPOSAL.md` (in-repo).
- `docs/baselines/BASELINE.md` (in-repo).
- `docs/studies/HANDOFF.md` (in-repo).
- HMT-GRN, SIGIR 2022 — https://bhooi.github.io/papers/hmt_sigir22.pdf
- MGCL, Frontiers 2024 — https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1428785/full
- Bi-Level GSL, arXiv 2024 — https://arxiv.org/html/2411.01169v1
- "Learning Hierarchical Spatial Tasks", TORS 2024 — https://dl.acm.org/doi/10.1145/3610584
- Massive-STEPS, arXiv 2025 — https://arxiv.org/html/2505.11239v1
- Foursquare dataset stats — https://sites.google.com/site/yangdingqi/home/foursquare-dataset
