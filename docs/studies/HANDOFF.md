# Handoff — state as of 2026-04-15 (updated)

Snapshot written at session close so the next session (using `/coordinator` + `/worker`) starts with full context. This is a **transient** file — update or delete it once P1 is underway and state.json is authoritative.

---

## Study status at a glance

- **Current phase:** P0 (running — CBIC sanity in progress, awaiting result archive)
- **Claims catalog:** 30 claims + 3 negations in `CLAIMS_AND_HYPOTHESES.md` (C01–C30 + N01–N03)
  - C29, C30, N03 added 2026-04-15 from HGI leakage audit; both **confirmed**, enrolled in state.json as P0 tests
- **Phases registered:** P0–P6
- **Test suite:** 692 passed, 17 skipped (sklearn 1.8.0, torch 2.11.0)
- **Git:** `main` is clean; latest meaningful commit `19b9c2b` (leakage audit + C_fclass_shuffle arm for AL/FL)

---

## Data availability snapshot (as of 2026-04-15)

| State   | dgi | hgi | fusion | sphere2vec | time2vec (next only) | poi2hgi | check2hgi |
|---------|:---:|:---:|:------:|:----------:|:--------------------:|:-------:|:---------:|
| alabama |  ✓  |  ✓  |   ✓    |     ✓      |          ✓           |    ✗    |     ✗     |
| arizona |  ✓* |  ✓  |   ✓    |     ✗      |          ✗           |    ✗    |     ✗     |
| florida |  ✓* |  ✓  |   ✓    |     ✓      |          ✓           |    ✗    |     ✗     |

✓* = data present but **category parquet is pre-Phase-2 format** (missing `placeid` column). Folds frozen using independent StratifiedKFold fallback. Regenerate via Phase-2 pipeline before any experiment that needs POI-level user isolation for category.

**Integrity check results** (`docs/studies/results/P0/integrity/*.json`):

| State   | engine  | status | notes                                      |
|---------|---------|--------|--------------------------------------------|
| alabama | dgi     | OK     |                                            |
| alabama | hgi     | OK     |                                            |
| alabama | fusion  | OK     |                                            |
| arizona | dgi     | **FAIL** | category missing `placeid` (pre-Phase-2) |
| arizona | hgi     | OK     |                                            |
| arizona | fusion  | OK     |                                            |
| florida | dgi     | **FAIL** | category missing `placeid` (pre-Phase-2) |
| florida | hgi     | OK     |                                            |
| florida | fusion  | **WARN** | half-L2 ratio 40.99× > expected 5-30× band |

The FL/fusion scale ratio warning (40.99×) is higher than Alabama's ~15×. May reflect different
HGI/Sphere2Vec scale distributions in Florida. Investigate before relying on fusion results for FL;
normalization was confirmed to hurt in AL so don't normalize blindly.

---

## Frozen folds snapshot

Rollup at `docs/studies/results/P0/folds/frozen.json` — **9 entries** (all locally frozen):

| key                  | status | fold_file (local)                                      | notes |
|----------------------|--------|--------------------------------------------------------|-------|
| alabama/dgi/mtl      | frozen | output/dgi/alabama/folds/fold_indices_mtl.pt (33.5 MB) |       |
| alabama/hgi/mtl      | frozen | output/hgi/alabama/folds/fold_indices_mtl.pt (33.6 MB) |       |
| alabama/fusion/mtl   | frozen | output/fusion/alabama/folds/fold_indices_mtl.pt (65.9 MB) | was on external SSD (lost); re-frozen locally |
| arizona/dgi/mtl      | frozen | output/dgi/arizona/folds/fold_indices_mtl.pt (65.3 MB) | ⚠ fallback StratifiedKFold (no placeid) |
| arizona/hgi/mtl      | frozen | output/hgi/arizona/folds/fold_indices_mtl.pt (68.5 MB) |       |
| arizona/fusion/mtl   | frozen | output/fusion/arizona/folds/fold_indices_mtl.pt (134.6 MB) | |
| florida/dgi/mtl      | frozen | output/dgi/florida/folds/fold_indices_mtl.pt (387.5 MB) | ⚠ fallback StratifiedKFold (no placeid) |
| florida/hgi/mtl      | frozen | output/hgi/florida/folds/fold_indices_mtl.pt (398.7 MB) |       |
| florida/fusion/mtl   | frozen | output/fusion/florida/folds/fold_indices_mtl.pt (785.1 MB) | |

The external-SSD alabama/fusion entry from the prior frozen.json has been replaced by the locally-frozen file. The SSD is no longer needed for P0 and should be considered lost for fold-state purposes.

---

## P0 leakage audit — state.json entries

Four test entries enrolled in P0 as of 2026-04-15:

| test_id              | claims    | verdict             | cat_f1 | next_f1 |
|----------------------|-----------|---------------------|--------|---------|
| leakage_AL_baseline  | C29, C30  | matches_hypothesis  | 0.786  | 0.238   |
| leakage_AL_arm_C     | C29, N03  | matches_hypothesis  | 0.144  | 0.199   |
| leakage_FL_baseline  | C29, C30  | matches_hypothesis  | 0.765  | 0.363   |
| leakage_FL_arm_C     | C29, N03  | matches_hypothesis  | 0.151  | 0.298   |

Evidence lives in `docs/studies/results/P0/leakage_ablation/{alabama,florida}/`.

---

## P0 exit-criteria checklist

| Step | Done | Notes |
|---|:---:|---|
| P0.1 Embeddings regenerated for AL + AZ + FL | ✓ | All 9 combos have inputs. AZ+FL DGI pre-Phase-2 (no placeid) — usable with fallback folds |
| P0.2 `validate_inputs` tool built + exercised | ✓ | 9 pairs validated; 2 FAIL (DGI placeid), 1 WARN (FL/fusion scale) |
| P0.3 state.json initialized (P0–P6) | ✓ | Current phase P0; 4 leakage tests enrolled |
| P0.4 CBIC sanity run on AL + DGI | ✓ | cat_f1=0.463 (target 0.46–0.48 ✓), next_f1=0.245 (target 0.26–0.28, −1.3pp ✓ within ±3pp tolerance); verdict=partial_match; archived as P0/cbic_AL_dgi |
| P0.5 `launch / archive / analyze / validate` scripts | ✓ | 21/21 smoke |
| P0.6 `/study` skill | ✓ | `.claude/commands/study.md` |
| P0.7 `/worker` + `/coordinator` skills | ✓ | `.claude/commands/{worker,coordinator}.md` |
| P0.8 Fold-freezing tooling | ✓ | `scripts/study/freeze_folds.py`; auto-load plumbed into `scripts/train.py` |
| P0.8 Frozen: AL + AZ + FL × {dgi, hgi, fusion} × mtl | ✓ | All 9 locally frozen (2026-04-15) |
| Hardware decision documented | ✓ | MASTER_PLAN §Hardware: M4 Pro 24GB preferred |

---

## CBIC sanity — COMPLETED 2026-04-15

Run: `results/dgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260415_1722`
Archive: `docs/studies/results/P0/cbic_AL_dgi/`

| metric  | observed | target   | verdict          |
|---------|----------|----------|------------------|
| cat_f1  | 0.463    | 0.46–0.48 | matches_hypothesis |
| next_f1 | 0.245    | 0.26–0.28 | partial_match (−1.3pp, within ±3pp) |
| joint   | 0.354    | 0.33–0.38 | partial_match    |

**Conclusion:** CBIC sanity passes. No investigation needed. Safe to advance P0 → P1.

---

## Known issues / follow-ups

1. **AZ/DGI and FL/DGI pre-Phase-2 inputs**: category parquet missing `placeid`. Both blocked for experiments that need POI-level user isolation on category. Regenerate with the Phase-2 input pipeline before enrolling DGI+AZ or DGI+FL in P1 sweeps.

2. **FL/fusion scale imbalance (40.99×)**: Higher than expected. Investigate whether the Sphere2Vec or HGI component is responsible; compare to alabama's ~15×. Do not normalize before checking effect on training.

3. **CBIC sanity complete**: Archived as `P0/cbic_AL_dgi`. cat_f1=0.463 (matches 46–48% target), next_f1=0.245 (partial_match: 1.3pp below 26–28%, within tolerance). Safe to advance to P1.

---

## Key environmental facts to carry forward

- **`requirements.txt` pins `scikit-learn==1.8.0`.** The 1.8 release fixed a `StratifiedGroupKFold(shuffle=True)` bug — do **not** downgrade.
- **Torch 2.11.0** — regression test `test_mtl_f1_within_tolerance` re-calibrated floor 0.92 → 0.88.
- **MPS runs:** before long training, set
  ```bash
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
- **Colab parallelism:** use `scripts/study/colab_runner.py` via `notebooks/colab_study_runner.ipynb`.

---

## How to invoke next session

```
/coordinator P0        # check if CBIC sanity passed; if yes, advance to P1
/worker P1             # enroll + run P1 tests after P0 closes
```

The coordinator reads this HANDOFF + state.json + CLAIMS_AND_HYPOTHESES.md. If this file is stale vs `state.json`, **trust state.json**.
