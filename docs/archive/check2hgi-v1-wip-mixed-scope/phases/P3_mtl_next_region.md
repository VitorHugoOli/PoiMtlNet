# Phase P3 — Headline 2-task MTL runs

**Gates:** P2 complete.
**Exit gate:** CH02 and CH03 resolved.

## Experiments (2 headline runs + legacy reference)

| # | State | Engine | Task set | Monitor | Purpose |
|---|---|---|---|---|---|
| 1 | AL | CHECK2HGI | `check2hgi_next_region` | `joint_acc1` | CH02 + CH03 headline |
| 2 | FL | CHECK2HGI | `check2hgi_next_region` | `joint_acc1` | CH02 + CH03 replication |
| 3 | AL | HGI | `legacy_category_next` | `val_f1_category` | Legacy reference (re-runs pre-branch) |

5-fold CV, 50 epochs, NashMTL criterion, seed 42.

## Runbook

```bash
# AL headline
python scripts/train.py --state alabama --engine check2hgi --task mtl \
  --task-set check2hgi_next_region --folds 5 --epochs 50

# FL replication
python scripts/train.py --state florida --engine check2hgi --task mtl \
  --task-set check2hgi_next_region --folds 5 --epochs 50

# Legacy reference (sanity)
python scripts/train.py --state alabama --engine hgi --task mtl \
  --task-set legacy_category_next --folds 5 --epochs 50
```

## Analysis

- **CH02** (MTL lift on next-category): compare `val_f1_next_category` from P3 run 1 to `val_f1` from P2 run 2. Bootstrap CI; delta must be > CI width for `partial`/`confirmed`.
- **CH03** (no negative transfer):
  - `val_f1_next_category` (P3/1) vs `val_f1` (P2/2) → next-category head.
  - `val_accuracy_next_region` (P3/1) vs `val_accuracy` (P2/3) → next-region head.
  - Both must be ≥ single-task baseline by more than CI width. If either regresses, document and either downgrade CH03 or propose mitigation (fixed loss weighting).

## Failure modes

- **CH02 refuted on both states:** pivot paper narrative — check-in-level embeddings alone help (CH01), MTL adds no signal on this data. Still publishable; the branch remains valuable.
- **CH02 confirmed on one state, refuted on other:** expected reviewer question about dataset-specific behaviour. Honestly documented, becomes a limitations paragraph.
- **CH03 negative on one head:** first thing reviewers ask. Honestly documented; consider ablation with fixed loss weights to isolate cause.

## Claims touched

CH02, CH03. Headlines for the BRACIS paper's experimental section.
