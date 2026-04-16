# Phase P1 — Single-task baselines

**Goal:** establish single-task reference numbers for next-POI and next-region on both HGI and Check2HGI engines. This is the control against which every MTL result in P2+ is measured.

**Duration:** ~4h (6 runs × ~22 min on AL, ~80 min on FL at 5f × 50ep).

**Embedded claims tested:**
- CH01 — Check2HGI > HGI on single-task next-POI (headline).
- CH04 — Next-region is meaningful (beats majority baseline by > 2×).
- CH05 — Ranking metrics discriminate where macro-F1 collapses.

**Gates:** P0 complete; all code deltas from P0.3 merged; P0.4 smoke green.

---

## Experiments

| # | State | Engine | Task | Seed | Purpose |
|---|---|---|---|---|---|
| P1.1.AL.HGI | AL | HGI | next_poi | 42 | CH01 baseline — POI-level embedding |
| P1.1.AL.C2HGI | AL | Check2HGI | next_poi | 42 | CH01 test — check-in-level embedding |
| P1.1.FL.HGI | FL | HGI | next_poi | 42 | CH01 replication |
| P1.1.FL.C2HGI | FL | Check2HGI | next_poi | 42 | CH01 replication |
| P1.2.AL.C2HGI | AL | Check2HGI | next_region | 42 | CH04 + single-task floor |
| P1.2.FL.C2HGI | FL | Check2HGI | next_region | 42 | CH04 + single-task floor |

**All:** 5-fold user-held-out StratifiedGroupKFold, 50 epochs, `next_mtl` head, OneCycleLR, AdamW. Frozen folds (cached under `output/<engine>/<state>/folds/fold_indices_<task>.pt` — see fusion's fold-freeze script for parity).

**Note:** next-POI on HGI requires an analogous code path to next_poi on Check2HGI — the HGI engine's `next.parquet` uses `next_category` labels by default, so this may need a tiny loader variant. Verify during P0.3.

---

## Runbook

```bash
# P1.1 — next-POI single-task on both engines
for state in alabama florida; do
  for engine in hgi check2hgi; do
    STUDY_DIR=docs/studies/check2hgi python scripts/train.py \
      --state $state --engine $engine --task next_poi \
      --folds 5 --epochs 50 --seed 42 \
      --batch-size 4096
  done
done

# P1.2 — next-region single-task on Check2HGI only (next_region is a
# check2HGI-specific label set)
for state in alabama florida; do
  STUDY_DIR=docs/studies/check2hgi python scripts/train.py \
    --state $state --engine check2hgi --task next_region \
    --folds 5 --epochs 50 --seed 42 \
    --batch-size 4096
done
```

---

## Analysis

### CH01 — Check2HGI > HGI on single-task next-POI

Compare:
- `P1.1.AL.C2HGI.val_next_poi_acc10` vs `P1.1.AL.HGI.val_next_poi_acc10` (paired Wilcoxon, n=5 folds)
- Same on FL
- Same on MRR

**Decision:**
- If Check2HGI > HGI by > 2pp on Acc@10 AND the paired test is significant (p < 0.05): **confirm CH01**.
- If < 2pp but positive: **partial** — document as "check-in-level embeddings give a small but consistent lift."
- If Check2HGI ≤ HGI: **refute** — the thesis pivots. Either (a) check-in-level contextual information isn't the right inductive bias for next-POI, or (b) check2HGI is under-trained (see CH01 Notes + TRAINING_BUDGET_DECISION).

### CH04 — Next-region meaningfulness

From P1.2:
- `AL.val_next_region_acc1` should be ≥ `2 × 2.33% = 4.6%` (i.e. beats "always predict majority" by at least 2×).
- `FL.val_next_region_acc1` should be ≥ `2 × 22.5% = 45.0%`.

**Decision:**
- If both pass: **confirm CH04**.
- If AL passes but FL doesn't: **partial** — "next_region is learnable on low-cardinality states but saturated on FL's majority class." Implies FL results in later phases need class-weighted CE.
- If neither passes: **refute** — next_region is too noisy to be a meaningful auxiliary. CH02 (MTL lift) is likely to fail too; pause and investigate.

### CH05 — Ranking metrics vs macro-F1

On both P1.1 and P1.2 runs, report both macro-F1 and Acc@K. Expected:
- Macro-F1 for next_poi (≥10K classes) effectively zero on either engine. Report to document the fact, not as a comparison.
- Acc@10 and MRR non-trivial and discriminating.

**Decision:** **confirm CH05** by construction if `f1_macro < 0.01 AND acc10 > 0.05` on next_poi. Document as methodology justification in paper.

---

## Output artefacts

```
docs/studies/check2hgi/results/P1/
├── P1.1.AL.HGI/
│   ├── summary.json            # per-fold + aggregate
│   ├── metadata.json           # git commit, CLI, seed, config hash
│   └── per_fold/               # per-fold metric dicts
├── P1.1.AL.C2HGI/
├── P1.1.FL.HGI/
├── P1.1.FL.C2HGI/
├── P1.2.AL.C2HGI/
├── P1.2.FL.C2HGI/
└── ANALYSIS.md                 # CH01/CH04/CH05 verdicts + rationale
```

---

## Decision gate → P2

Proceed to P2 only when:

1. All 6 runs completed + archived.
2. CH01, CH04, CH05 have explicit verdicts in `ANALYSIS.md`.
3. If CH01 is negative on FL but positive on AL (or vice versa), document and pause before P2 (state-dependent behaviour may warp the MTL comparison).
