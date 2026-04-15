# Check2HGI — External Benchmarks & Baseline Reference

**Purpose:** Reference for worktrees picking up baseline comparison work.
Documents which external papers are baselines for each task in the check2HGI
track, dataset requirements, re-run protocol, and what the paper tables need.

> **Important context — do not mix with the legacy study:**
> The legacy `docs/baselines/BASELINE.md` covers two different tasks:
> - *Task 1 (next-category)*: POI-RGNN, MHA+PE
> - *Task 2 (venue annotation / unknown-POI labeling)*: PGC, HAVANA
>
> **HAVANA and PGC are venue annotation baselines — they are NOT related to the
> check2HGI tasks.** The check2HGI track has no venue-annotation head; do not
> carry those numbers into this paper.

---

## 1. Check2HGI prediction tasks

| Head | Definition | Metric | Label space |
|------|-----------|--------|-------------|
| `next_category` | Predict the category of the next POI in the user's sequence | Macro-F1 | 7 classes |
| `next_region` | Predict the census tract of the next POI | Acc@{1,5,10}, MRR | ~1100–5000 classes (state-dependent) |

Both tasks operate on the same check-in-level embedding sequences (window=9, dim=64).
They are trained jointly via NashMTL (2-task MTL). Each needs its own set of external
baselines for the paper.

---

## 2. Baselines for `next_category`

These already exist in `docs/baselines/BASELINE.md` and have **reproduced results on
FL, CA, TX**. No new papers needed for this head.

| Baseline | Venue | Datasets | Results in BASELINE.md |
|----------|-------|----------|------------------------|
| **POI-RGNN** | PE-WASUN '21 / Ad Hoc Networks '22 | Gowalla global + state (FL, CA, TX re-run) | Yes — FL, CA, TX per-category F1 |
| **MHA+PE** | Zeng et al., 2019 | Gowalla global | Paper-reported only (no state re-run) |

**POI-RGNN is the primary next-category baseline.** It uses the same Gowalla state
splits (FL, CA, TX reproduced by Capanema et al.) and the same 7-category taxonomy.

For the check2HGI paper, the comparison for `next_category` is:

```
MTLnet(check2HGI) vs MTLnet(HGI) vs POI-RGNN  — all on FL, CA, TX
```

POI-RGNN results are already in `docs/baselines/BASELINE.md`. They need to be
**re-run on the check2HGI data** (same fold protocol, same states) to be a fair
comparison — the architecture is the same but the input representation changes.

---

## 3. Baselines for `next_region`

This is a **new task** with no prior baseline in the project. External papers below.

### 3.1 HMT-GRN — Primary external baseline

| Field | Detail |
|-------|--------|
| **Paper** | *Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation* |
| **Venue** | SIGIR 2022 |
| **Datasets** | Gowalla (global split), Foursquare |
| **Task** | Next POI + **next region** — explicit hierarchical MTL |
| **Metrics** | Acc@1, Acc@5, Acc@10, MRR |
| **Code** | https://github.com/poi-rec/HMT-GRN |

**Why primary:** HMT-GRN is the closest published method to what we do — it also
jointly trains a region-level head and a POI-level head on Gowalla using geographic
hierarchy. Their region definition uses grid cells (G@2..G@6, lat-lon grid) vs our
census tracts, but the prediction problem is structurally identical.

**Comparability note:** HMT-GRN reports on the global Gowalla split. To compare
directly, it needs to be re-run on our state splits (FL, CA, TX) with our
POI→region mapping. See §5 for the re-run protocol.

---

### 3.2 Graph-Flashback

| Field | Detail |
|-------|--------|
| **Paper** | *Graph-Flashback Network for Next Location Recommendation* |
| **Venue** | KDD 2022 |
| **Datasets** | Gowalla, Foursquare |
| **Task** | Next POI prediction via spatiotemporal knowledge graph |
| **Metrics** | Acc@1, Acc@5, Acc@10, MRR |
| **Code** | https://github.com/kevin-xuan/Graph-Flashback |

**Role:** Strong POI-level transition baseline. Predicts next POI (not region) but
on the same Gowalla data. Useful for showing that predicting region (coarser) is
meaningfully harder or easier than predicting POI, and as a lower bound for what a
non-hierarchical model achieves on the same data.

---

### 3.3 STAN

| Field | Detail |
|-------|--------|
| **Paper** | *STAN: Spatio-Temporal Attention Network for Next Location Recommendation* |
| **Venue** | WWW 2021 |
| **Datasets** | Gowalla, Foursquare |
| **Metrics** | Acc@K |
| **Code** | https://github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation |

**Role:** Representative spatiotemporal attention baseline pre-dating HMT-GRN. Good
for the related-work table and for showing the trajectory from plain attention to
hierarchy-aware methods. Global split only — cite paper numbers, don't re-run unless
time permits.

---

## 4. Rejected baseline

### LP-BERT (gegen07/LP-BERT) — NOT suitable

| Issue | Detail |
|-------|--------|
| Dataset | HuMob Challenge 2023 (Japanese mobility grid) — not Gowalla |
| Task | 15-day trajectory forecasting (coordinates X/Y), not next check-in |
| Metrics | GEOBLEU + DTW — incompatible with Acc@K / F1 |
| Status | Competition submission, no peer review |

Can be cited in related work as a transformer mobility model, but not in any metric
comparison table.

---

## 5. Dataset requirements & blockers

Primary states for the BRACIS paper: **Florida, California, Texas**.
Alabama = internal smoke tests only (not in paper tables).

| State | Check-ins | Shapefile | Check2HGI emb | next.parquet | next_region.parquet | Ready |
|-------|-----------|-----------|---------------|--------------|---------------------|-------|
| Florida | ✓ | ✓ TL_FL | ✓ | ✓ | ✓ | **Yes** |
| California | ✓ | ✓ TL_CA | ✗ | ✗ | ✗ | Blocked |
| Texas | ✓ | ✓ TL_TX | ✗ | ✗ | ✗ | Blocked |
| Alabama | ✓ | ✓ TL_AL | ✓ | ✓ | ✓ | Internal only |

**To unblock CA and TX:**

```bash
# 1. Enable California and Texas in pipelines/embedding/check2hgi.pipe.py STATES dict
# 2. Run embedding generation (~20 min each on MPS):
python pipelines/embedding/check2hgi.pipe.py

# 3. Build input tables for each state:
python pipelines/create_inputs_check2hgi.pipe.py --state california
python pipelines/create_inputs_check2hgi.pipe.py --state texas
```

---

## 6. Re-run protocol for external baselines

For BRACIS, baselines must be **re-run on the same split** as our model. Citing
global-Gowalla paper numbers is not sufficient for a direct comparison row.

Protocol for any external baseline (apply to HMT-GRN, Graph-Flashback):

1. Clone the baseline repo.
2. Adapt the dataset loader to read `next.parquet` and/or `next_region.parquet`.
3. Use **5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`**
   grouped by `userid` — same as our training split.
4. Run on FL, CA, TX. Report Acc@{1,5,10} and MRR for `next_region`; Macro-F1 for
   `next_category`.
5. Save results to `docs/studies/check2hgi/results/external/<baseline>/<state>/summary.json`.
6. Add evidence pointer to `CLAIMS_AND_HYPOTHESES.md`.

**HMT-GRN adaptation note:** Their region hierarchy is a lat-lon grid; ours is census
tracts. Swap their `poi_to_region` mapping with ours (from `checkin_graph.pt`) and
keep everything else. Estimated effort: 2–4h.

---

## 7. Paper table sketches

### Table A — next_category (Macro-F1)

```
Method              | FL    | CA    | TX
────────────────────|───────|───────|──────
MHA+PE (†)          | 26.9  | 26.9  | 26.9   ← global Gowalla, magnitude reference
POI-RGNN            | 34.49 | 31.78 | 33.03  ← reproduced in BASELINE.md
──────────────────────────────────────────
HGI single-task     | P2.1c | —     | —
C2HGI single-task   | P2.1d | —     | —
MTL C2HGI (ours)    | P3.1  | P3.2  | P3.3

(†) Global Gowalla — not state-split.
```

### Table B — next_region (Acc@1 / Acc@10 / MRR)

```
Method              | FL Acc@1 | FL Acc@10 | FL MRR | CA ... | TX ...
────────────────────|──────────|───────────|────────|────────|───────
Majority class      | ~2%      | ~10%      | —      | TBD    | TBD
HMT-GRN (re-run)    | TBD      | TBD       | TBD    | TBD    | TBD
Graph-Flashback (†) | TBD      | TBD       | TBD    | TBD    | TBD
─────────────────────────────────────────────────────────────────────
C2HGI single-task   | P2.4     | P2.4      | P2.4   | P2.5   | P2.6
MTL C2HGI (ours)    | P3.1     | P3.2      | P3.3   | ...    | ...
```

TBD cells filled after external re-runs complete.

---

## 8. Related claims

| Claim | What it needs |
|-------|--------------|
| CH01 | HGI single-task (P2.1a/c) vs C2HGI single-task (P2.1b/d) — Macro-F1 |
| CH02 | C2HGI single-task vs C2HGI MTL — Macro-F1 on next_category |
| CH03 | Per-head MTL ≥ single-task on both heads |
| CH04 | C2HGI single-task next_region Acc@1 vs majority-class baseline |
| CH05 | MRR surfaces differences that Macro-F1 misses on high-cardinality head |

---

*Last updated: 2026-04-15. Maintainer: check2hgi-mtl worktree.*
