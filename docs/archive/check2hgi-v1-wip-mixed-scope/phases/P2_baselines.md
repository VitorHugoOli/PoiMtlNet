# Phase P2 — Single-task baselines

**Gates:** P-1 and P0 + P1 complete.
**Exit gate:** CH01, CH04, CH05 resolved with evidence pointers.

## Experiments (6 runs total)

| # | State | Engine | Task | Monitor | Purpose |
|---|---|---|---|---|---|
| 1 | AL | HGI | single-task next (category) | F1 | CH01 baseline |
| 2 | AL | CHECK2HGI | single-task next (category) | F1 | CH01 test |
| 3 | AL | CHECK2HGI | single-task next_region | Acc@1 | CH04 + CH05 |
| 4 | FL | HGI | single-task next (category) | F1 | CH01 replication |
| 5 | FL | CHECK2HGI | single-task next (category) | F1 | CH01 replication |
| 6 | FL | CHECK2HGI | single-task next_region | Acc@1 | CH04 replication |

All: 5-fold user-held-out CV, 50 epochs, OneCycleLR, same arch (`next_mtl`), same seed (42).

## Runbook

```bash
# CH01 baseline
python scripts/train.py --state alabama --engine hgi --task next --folds 5 --epochs 50

# CH01 test
python scripts/train.py --state alabama --engine check2hgi --task next --folds 5 --epochs 50

# CH04 + CH05
python scripts/train.py --state alabama --engine check2hgi --task next_region --folds 5 --epochs 50
# (the 'next_region' --task value is P0/P1 work — not yet implemented)
```

## Analysis

- CH01 = (2 vs 1) delta in `val_f1`. Bootstrap CI on folds.
- CH04 = compare (3) `val_accuracy` to the majority-class baseline (hand-computed from region label frequencies).
- CH05 = whether `val_mrr` delta (CHECK2HGI vs HGI on single-task next-category, from runs 1 vs 2) is statistically significant when `val_f1` delta is not.

## Output artefacts

Each run writes to `docs/studies/check2hgi/results/P2/<run_id>/summary.json` with:

```json
{
  "run_id": "...",
  "state": "alabama",
  "engine": "check2hgi",
  "task": "next",
  "folds": [{"fold": 0, "val_f1": 0.xxx, "val_accuracy": 0.xxx, ...}, ...],
  "aggregate": {"val_f1_mean": 0.xxx, "val_f1_std": 0.xxx, ...}
}
```

## Claims touched

CH01, CH04, CH05.
