# HGI Leakage Ablation — Florida Replication

**Date:** 2026-04-15
**Purpose:** Confirm the arm C / fclass-shuffle finding from Alabama
generalises to a larger, independently-processed state.
**Config:** Florida, HGI-only, DSelect-k(e=4,k=2) + aligned_mtl, 1 fold
(fold 0 of StratifiedGroupKFold), seed 42, 50 epochs, batch 4096,
grad_accum_steps=1, embedding_dim=64.
**Companion:** `../alabama/README.md` (full arm explanation),
`docs/studies/fusion/issues/HGI_LEAKAGE_AUDIT.md` (technical audit),
`docs/studies/fusion/issues/HGI_LEAKAGE_EXPLAINED.md` (glossary).

## Arms

Two arms only: `baseline` (current defaults) and `C_fclass_shuffle`
(`shuffle_fclass_seed=42` applied identically in Phase 3a and Phase 4,
`category` untouched). Arms A / B / A+B were run on Alabama; their
null / asymmetric results do not need cross-state replication to stand.

## Results

```
state      arm                  cat_f1     Δcat   cat_acc   next_f1     Δnxt   next_acc
----------------------------------------------------------------------------------------
alabama    baseline            0.7855   +0.00    0.8250    0.2383    +0.00    0.3029
alabama    C_fclass_shuffle    0.1437  −64.19    0.2623    0.1988    −3.95    0.2587

florida    baseline            0.7649   +0.00    0.7961    0.3627    +0.00    0.4344
florida    C_fclass_shuffle    0.1506  −61.43    0.2583    0.2982    −6.46    0.3715
```

F1 = macro F1. Δ in percentage points vs same-state baseline. Random
chance floor for macro F1 on 7 classes = 1/7 ≈ 0.1429.

## Observations

1. **Category collapses to chance on both states** (0.1437 Alabama / 0.1506
   Florida — both within ±0.01 of 1/7). The fclass→category shortcut
   dominates Category F1 regardless of dataset size.

2. **Absolute Next-POI F1 is higher on Florida** (0.3627 vs 0.2383)
   because Florida has ~85k sequences vs Alabama's ~13k — 6× more
   training signal for sequence modelling.

3. **Florida's Next-POI shuffle-drop is mildly larger** (−6.46 vs −3.95
   p.p.). Plausible reading: fclass identity provides slightly more
   contextual signal when the sequence model has more data to learn
   patterns from. Still an order of magnitude smaller than the Category
   collapse — not enough to change the verdict.

4. **Time scaling:** Alabama's two-arm replication took 21 min total;
   Florida's took 99 min (≈4.7× slower), driven by POI count
   (11 848 → 76 544, ≈6.5×). HGI training scales sub-linearly in POIs
   (message-passing on the Delaunay graph caches well) which explains
   the sub-linear wall-clock.

## Cross-state fclass→category purity (all six Gowalla states)

Confirms the shortcut is a dataset-taxonomy property, not an
Alabama/Florida coincidence:

| state | POIs | fclasses | pairs | purity_macro | purity_weighted |
|---|---:|---:|---:|---:|---:|
| Alabama | 11 848 | 284 | 284 | 1.0000 | 1.0000 |
| Arizona | 20 666 | 305 | 305 | 1.0000 | 1.0000 |
| California | 169 145 | 333 | 333 | 1.0000 | 1.0000 |
| Florida | 76 544 | 324 | 324 | 1.0000 | 1.0000 |
| Georgia | 29 667 | 313 | 313 | 1.0000 | 1.0000 |
| Texas | 160 938 | 365 | 365 | 1.0000 | 1.0000 |

Archived at `docs/studies/fusion/results/P0/leakage_ablation/fclass_purity.json`.

## Decision

The Florida replication **does not change** the conclusion or plan
described in the Alabama README (`../alabama/README.md`) and the audit
doc (`docs/studies/fusion/issues/HGI_LEAKAGE_AUDIT.md` §9). C29 is now cross-state
confirmed; the required paper-side actions remain as stated.

## Reproduce

```
.venv/bin/python scripts/hgi_leakage_ablation.py \
    --state Florida --arms baseline,C_fclass_shuffle
```

## Artifacts

- `baseline/` — embeddings, fclass .pt, MTL log, full MTL results
- `C_fclass_shuffle/` — same, with shuffled fclass
- `run_log.json` — per-arm timings, return codes, result pointers
- Backup of pre-ablation state: `output/hgi/florida.backup_pre_leakage_ablation_20260415_*`
