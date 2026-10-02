# G5: HMT-GRN-style baseline at its best epoch (2026-10-01/02, M2 Pro, MPS)

**Why.** The printed HMT-GRN baseline cells (`docs/results/baselines/hmt_grn/`) report **epoch 50 with
no epoch selection**, while the dedicated cells they are compared with report each head at its
best epoch. G5 re-runs HMT-GRN with `--epoch-select per_task`: every epoch is evaluated on the
evaluation fold, and each head is read at its own best epoch, the same convention as the printed
dedicated cells.

**Run.** `scripts/baselines/b3_hmt_grn.py --state <st> --seed 0 --folds 5 --epochs 50 --engine check2hgi_dk_ovl --epoch-select per_task`,
on MPS. Per-state logs: `logs/full_<st>/` (AL/AZ/IST) and `logs/g5_fl_5f_r2/` (FL).
- **AL, AZ, IST:** 2026-10-01, commit `82dbe843`.
- **FL:** 2026-10-02, commit `04f5e12f`. No change to `b3_hmt_grn.py` since `82dbe843`.

## Result

| state | category, best epoch | region Acc@10, best epoch | cat / reg at epoch 50 (this run) | printed (epoch 50) | dedicated v18 cell (seed mean) |
|---|---|---|---|---|---|
| AL | **20.99** ± 1.05 (ep 13–29) | **63.19** ± 3.67 (ep 8–11) | 19.32 / 57.13 | 19.37 / 57.05 | 30.78 / 70.12 |
| AZ | **22.09** ± 0.73 (ep 13–17) | **52.11** ± 1.56 (ep 5–7) | 18.54 / 43.91 | 18.04 / 43.70 | 34.57 / 59.48 |
| IST | **24.85** ± 0.66 (ep 8–20) | **68.06** ± 0.79 (ep 9–12) | 19.10 / 60.44 | 19.10 / 60.42 | see `v18_results.json` |
| FL | **30.58** ± 0.30 (ep 5–9) | **72.76** ± 1.12 (ep 3–4) | 26.87 / 63.73 | 26.87 / 63.74 | 37.35 / 76.69 |

(Mean ± sd over 5 folds, seed 0. The dedicated column is `v18_results.json` `cells.<st>.stl_cat` / `stl_reg`.)

**Reading.**
- **Epoch 50 reproduces the printed cells.** Category is within 0.5 and region within 0.2, which rules
  out a device or code drift.
- **HMT peaks early and degrades.** Region peaks at epochs 3–12 and category at epochs 5–29. Reading
  epoch 50 understated HMT by:
  - category: +1.7 (AL), +3.6 (AZ), +5.8 (IST), +3.7 (FL);
  - region: +6.1 (AL), +8.2 (AZ), +7.6 (IST), +9.0 (FL).
- **The ordering against the dedicated v18 cells holds at every state checked, but the margins shrink.**
  At FL the region gap goes from about 13 to 3.9 points, and the category gap from 10.5 to 6.8.

## Folds and data

- **AL, AZ, IST:** HMT's split (`b3_hmt_grn.build_fold_split`: StratifiedGroupKFold, seed 0, int64
  userids, `next_category`, on `check2hgi_dk_ovl` rows) was asserted equal to `train.py`'s delivered
  folds 5/5 before training.
- **FL:** the same code path. Its per-fold `n_val` (254,884 ×3 / 254,883 ×2) equals the June record's,
  and fold 0's train/val sizes (1,019,534 / 254,884) equal G1 FL fold 0's frozen split. That split
  was independently reproduced from the same rows (A5-lite, `../g1_integrity_*`).
  **5/5 equality asserted on 2026-10-02 07:46.** `b3_hmt_grn.build_fold_split` was run on the
  slim copy `b3` read and compared with the `train.py` rule (`load_next_data`'s label path:
  `_map_categories`, `astype(int)`, NaN drop (0 NaN), int64 StratifiedGroupKFold, seed 0) run on
  the full `dk_ovl` file. Result:
  - train and val index shas are equal in every fold, and so are the val user sets;
  - each fold's `n_val` equals the run JSON's;
  - fold 0's `val_idx_sha256` equals G1 FL's frozen split.
- **FL inputs are a slim copy.** Only the label columns `b3` reads were copied, plus the full
  `sequences_next`, with content hashes equal on both sides. See `FL_SLIM_COPY.md`. The 576 embedding
  columns are never read by `b3`.

## Patch check (per-epoch evaluation does not change training), AL fold 0

`logs/check_al_f0_{none,none_rep,pertask,pertask_rep}/`:
- The train loss is identical at every epoch across `none` ×2 and `per_task` ×2.
- The epoch-50 values equal `none` exactly in 2 of 2 comparisons after the first.
- The first `per_task` run's epoch-50 category value differed by 0.054 pp. This was evaluation-time MPS
  numeric noise: region was identical, and it did not reproduce.

## FL run history

- **First attempt** (`logs/g5_fl_5f/`) ran in parallel with G1 jobs. It was **stopped by its watchdog
  at 02:09, in fold 4**, during a system-wide memory spike. `b3` writes only at the end, so nothing
  from it is used.
- **The re-run** (`logs/g5_fl_5f_r2/`) ran alone: 3,360 s, peak RSS 2.5 GB, rc = 0.

## Files

- `runs/<st>_b3_seed0_folds5_bestepoch.json`: per fold, each head at its best epoch, plus a `last_epoch`
  block and the full per-epoch `history`.
- `logs/`: each run's `RUN.txt` and `mem.log`, and the drivers (`g5_run.sh`, `g5_run_par.sh`).
- `G5_SUMMARY_scratch.md`: the scratch summary written on 10-01 (AL/AZ/IST).
- `FL_SLIM_COPY.md`.
- `MANIFEST.md5`.

Uncommitted. Committing needs `git add -f docs/results/closing_data/v18/g5_hmt_best_epoch/`.
