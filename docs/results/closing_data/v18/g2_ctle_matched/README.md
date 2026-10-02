# G2: frozen-CTLE control re-run on the MATCHED split, Alabama and Arizona (2026-10-02, M2 Pro)

**Why.** The June CTLE-SC cells (`docs/results/closing_data/baseline_compare/{alabama,arizona}_ctle.json`,
`MACS_BOARD_RESULTS.md`) were mismatched:
- CTLE was pre-trained on a fold split computed over the canonical `check2hgi` rows
  (`build_ctle_substrate.get_fold_indices` hard-coded `CHECK2HGI`);
- each cell was **scored** by `train.py --only-fold` on the stride-1 `check2hgi_dk_ovl` rows, whose folds differ;
- so 74–82% (AL) and 78–82% (AZ) of each scored fold's validation users were on the pre-training side.

G2 re-runs the cells with `--split-engine check2hgi_dk_ovl`, so pre-training excludes exactly the
scored fold's users.

## Result (category macro-F1, seed 0, 5 folds)

| state | June (mismatched split) | **matched split** | matched − June |
|---|---:|---:|---:|
| Alabama | 17.77 | **16.31 ± 1.02** | **−1.46 ± 0.95** (5/5 folds lower, paired t p = 0.027) |
| Arizona | 19.30 | **17.67 ± 1.13** | **−1.63 ± 1.00** (5/5 lower, t p = 0.022) |

Per fold: `G2_CTLE_matched.csv`.
- AL: 17.17 / 15.24 / 15.89 / 15.66 / 17.61 against June's 18.53 / 17.90 / 16.93 / 15.83 / 19.66.
- AZ: 17.54 / 18.01 / 17.38 / 16.16 / 19.28 against June's 18.10 / 18.92 / 19.77 / 19.08 / 20.63.

**Reading.** The mismatch favoured CTLE: removing it lowers CTLE by about 1.5 pp in every fold.
Against the representations it is compared with, CTLE's gap therefore only grows. The ordering
sentence it supports is strengthened, not reversed. Ch. 5 prints no CTLE number at AL/AZ/IST, so no
printed value changes.
- **Caution:** the `Check2HGI-SC` comparand in `MACS_BOARD_RESULTS.md` (55.59 / 56.31) is a pre-v18
  substrate value. Any CTLE vs Check2HGI Δ should use the v18 dedicated cells, not that column.

## Protocol

Per (state, fold F), driver `drivers/g2_ctle_fold.sh`, strictly serial:
1. **Build.** `scripts/baselines/build_ctle_substrate.py --state <st> --seed 0 --fold F --split-engine check2hgi_dk_ovl --stride 1 --device mps`,
   with defaults `--pretrain-epochs 10 --max-len 64 --batch-size 256 --lr 1e-3`.
   - Pre-training (MLM + masked-hour) uses only the fold's training users. Every row is then embedded
     by the frozen encoder.
   - The builder writes `next`, `sequences_next` and `next_region` (stride 1) and `CTLE_FOLD.txt`,
     which records `split_engine`, `n_val_users` and `val_users_sha256`.
2. **Assertions, fail-stop** (`drivers/g2_assert.py`, `logs/ASSERTS_g2.log`, 10/10 passed):
   - the `check2hgi_ctle` rows equal `check2hgi_dk_ovl`'s: AL 96,326 / AZ 200,895, with userids and
     labels equal in order. The builder's global `min_seq` 5 is therefore a no-op here, as its
     docstring states;
   - an independent StratifiedGroupKFold over `load_next_data(check2hgi_ctle)` int64 rows reproduces
     `CTLE_FOLD.txt`'s `val_users_sha256` for fold F, i.e. `train.py`'s scored fold. AL fold 0's sha
     (`01b7b2e1…`) also equals G1 AL fold 0's split.
3. **Score**, with the board `run_cat` recipe from `scripts/closing_data/mac_baseline_compare.py`, verbatim:
   - `train.py --task next --engine check2hgi_ctle --state <st> --seed 0 --only-fold F --cat-head next_gru --epochs 50 --batch-size 2048`;
   - no logit adjustment, `OMP_NUM_THREADS=4`, MPS;
   - macro-F1 from the single fold report (`macro avg`), as `run_cat` reads it. It equals the f1-best
     epoch of `metrics/fold1_next_val.csv` to within 1e-4 in all 10 cells, and the report support equals
     the fold's `n_val`.

**Device.** CTLE pre-training ran on **MPS** (the builder's default), at the author's request on
2026-10-02. **A June AL CTLE pre-training log shows `device=cpu`**:
`/Volumes/Vitor's SSD/ingred/output/_scratch/log_ctle_alabama_s0_f1.txt`, dated 2026-06-21, line 5:
`CTLE pretrain: vocab=11850 train_trajs=2267 … device=cpu epochs=10`.
- That is one fold-1 log dated before the board cells. **It is not proven to be the scored cell's own log.**
- In any case this re-run does not bit-reproduce June. Its purpose is the matched split, not reproduction.

**Inputs.** The canonical `output/check2hgi/{alabama,arizona}/{embeddings,region_embeddings,poi_embeddings}.parquet`,
which the builder's `_load_checkins` reads, were copied from nespedgpu md5-equal (`logs/CTLE_INPUT_MD5S_box.txt`).
The `check2hgi_dk_ovl` rows are the ones used by G1 and G5, md5-equal to the box.

**Code.** Commit `d01cd2a8` (a `--shared` clone on the external SSD). `--split-engine` landed in `82dbe843`.

**Cost.** Build 44–103 s and score 169–174 s (AL) or ~347 s (AZ) per cell, about 5 and 8 min per fold.
- Peak RSS ≤ 3 GB. At most one MPS process at a time (global lock), with a PID watchdog.
- No aborts.

## Files

- `G2_CTLE_matched.csv` and `G2_CTLE_matched_stats.json`.
- `runs/<st>_f<F>/`: `CTLE_FOLD.txt`, plus `metrics/`, `folds/` and `summary/` of the scoring run, and `SOURCE_RUNDIR.txt`.
- `logs/`: each step's `RUN.txt` and `mem.log`, a `build_summary.txt` (split, pre-training losses,
  rows) per build, and the assertion and driver logs.
- `drivers/`.
- `MANIFEST.md5`.

Scratch (not copied): `/Volumes/Vitor's SSD/ingred_g1/results/g2_ctle/`. The per-fold CTLE engine dir
was overwritten fold by fold, and the AL one was deleted after scoring (`DELETED.txt`).

Uncommitted. Committing needs `git add -f docs/results/closing_data/v18/g2_ctle_matched/`.
