# Phase P1 — Single-task references

**Goal:** produce the single-task reference numbers for next-POI and next-region on Check2HGI. These become the **internal baselines** against which every MTL / dual-stream / cross-attention claim in P2+ is measured.

**This study is standalone — no HGI or other cross-engine comparison here.**

**Duration:** ~3h (AL + FL × 2 single-task runs × 5f × 50ep, seed 42 only; multi-seed lives in P2).

**Embedded claims tested:**
- **CH04** — Learned Check2HGI models must beat the simple-baselines floor from P0.5 by ≥ 2×.
- **CH05** — Ranking metrics (Acc@K, MRR) discriminate where macro-F1 collapses.
- **CH06** — OOD-restricted Acc@K reported alongside raw Acc@K (train-memorisation guard).

**Gates:** P0 complete; CH14 + CH15 audits resolved with acceptable verdicts; simple baselines computed.

---

## Experiments

| # | State | Task | Seed | Purpose |
|---|---|---|---|---|
| P1.1.AL | AL | next_poi single-task | 42 | Reference for CH01 pairing in P2 |
| P1.1.FL | FL | next_poi single-task | 42 | Reference for CH01 pairing in P2 |
| P1.2.AL | AL | next_region single-task | 42 | Reference for CH02 per-head pairing in P2 |
| P1.2.FL | FL | next_region single-task | 42 | Reference for CH02 per-head pairing in P2 |

All: 5-fold user-held-out StratifiedGroupKFold, 50 epochs, `next_mtl` head, OneCycleLR, AdamW, **seed 42 only** (multi-seed happens in P2 for the headline).

---

## Runbook

```bash
# P1.1 — single-task next-POI
for state in alabama florida; do
  STUDY_DIR=docs/studies/check2hgi python scripts/train.py \
    --state $state --engine check2hgi --task next_poi \
    --folds 5 --epochs 50 --seed 42 --batch-size 4096
done

# P1.2 — single-task next-region
for state in alabama florida; do
  STUDY_DIR=docs/studies/check2hgi python scripts/train.py \
    --state $state --engine check2hgi --task next_region \
    --folds 5 --epochs 50 --seed 42 --batch-size 4096
done
```

**Code-delta note:** `--task next_poi` and `--task next_region` need to be valid single-task options in `scripts/train.py::_VALID_TASKS`. The pre-P1 code-delta list (P0.3) includes this.

---

## Analysis

### CH04 floor check — learned > 2× best simple baseline

For each state × task, read the simple-baselines JSON under `results/P0/simple_baselines/<state>/<task>.json` and compare:

```
learned_acc10 / max(random_acc10, majority_acc10, markov_acc10, top_k_acc10, userhist_acc10) ≥ 2
```

**If learned is not 2× the best baseline on any state × task:** the pipeline is almost certainly broken. Pause and investigate before P2. Possible causes: incorrect poi_idx labels (should have been caught in P0.2 round-trip), wrong num_classes at head output, loss that's numerically degenerate, input embeddings all-zero, etc.

### CH05 — Ranking metrics vs macro-F1

Each P1.* run's `summary.json` contains both per-fold:
- macro-F1 (expected tiny, ~0 on next_poi with 10K+ classes, because the sparse-support denominator dominates)
- Acc@1, Acc@5, Acc@10, MRR, NDCG@5, NDCG@10 (expected meaningful)

If macro-F1 actually discriminates usefully across variants → refute CH05 (our methodology claim is wrong; reconsider paper framing).

### CH06 — OOD-restricted Acc@K reported

Each P1.* run also writes an `ood_summary.json` breaking down per-fold:
- N_total val sequences
- N_in_distribution (target POI appears in train fold)
- N_ood (target POI absent from train fold)
- raw_acc10 = correct / N_total
- ood_restricted_acc10 = correct_among_in_dist / N_in_distribution

**Implementation note:** this requires a small extension to `evaluate_model` — after computing per-sample correctness, intersect with the train-fold POI set to compute the OOD-restricted slice. Estimate ~30 LOC; add to pre-P1 code-delta list.

**Claim verdict:** CH06 confirmed when learned models STILL beat simple baselines when restricted to in-distribution POIs. This guards against the trivial "model memorises train POIs, gets 0 on OOD, raw Acc@K average is inflated by the lucky POIs that happen to be in both" artefact.

---

## Output artefacts

```
docs/studies/check2hgi/results/P1/
├── P1.1.AL/
│   ├── summary.json
│   ├── ood_summary.json
│   ├── metadata.json
│   └── per_fold/
├── P1.1.FL/
├── P1.2.AL/
├── P1.2.FL/
└── ANALYSIS.md             # CH04 floor table + CH05/CH06 verdicts
```

---

## Decision gate → P2

Proceed when:
1. All 4 runs archived under `results/P1/`.
2. **CH04 floor passes on all 4 (state × task) cells** — learned Acc@10 ≥ 2× best simple baseline. If any fails, pause.
3. CH05 and CH06 have documented verdicts.
4. `val_joint_geom_lift`-equivalent single-head metric computed and logged (for consistency with P2 monitor).

If CH04 fails on any cell, the coordinator refuses to advance phase.
