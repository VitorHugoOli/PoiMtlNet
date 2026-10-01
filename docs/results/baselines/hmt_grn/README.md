# HMT-GRN-style baseline (B3): per-run records, copied for provenance

**Source.** Copied unchanged on 2026-10-01 from the external SSD:
`/Volumes/Vitor's SSD/ingred/results/baseline_b3_hmt_grn_style/<dir>/b3_seed0_folds5.json`.
Each copy is md5-equal to its source. The files were written on the M4 Pro on 2026-06-24 (03:09 to
15:13) by `scripts/baselines/b3_hmt_grn.py`. `MACS_BOARD_RESULTS.md` said they lived on the SSD and in
`~/ingred_run` on the M4. Nothing else was copied.

| dir | printed? | cat macro-F1 | reg Acc@10 | base_engine | n_val per fold |
|---|---|---:|---:|---|---|
| `alabama` | yes | 19.37 | 57.05 | check2hgi_dk_ovl | 19269 / 19265 / 19264 / 19264 / 19264 |
| `alabama_M4MPS` | no (MPS re-run of AL) | 19.34 | 56.99 | check2hgi_dk_ovl | same as `alabama` |
| `arizona` | yes | 18.04 | 43.70 | check2hgi_dk_ovl | 40179 × 5 |
| `florida` | yes | 26.87 | 63.74 | check2hgi_dk_ovl | 254884 / 254884 / 254884 / 254883 / 254883 |
| `california` | yes | 24.01 | 49.61 | check2hgi_dk_ovl | 585091 / 585092 / 585092 / 585093 / 585098 |
| `texas` | yes | 25.81 | 53.85 | check2hgi_dk_ovl | 766083 / 766083 / 766083 / 766082 / 766083 |
| `istanbul_stride1` | yes (IST 19.10 / 60.42) | 19.10 | 60.42 | check2hgi | 54333 / 54333 / 54334 / 54333 / 54333 |
| `istanbul` | no (superseded set-a) | 20.87 | 56.56 | check2hgi | 11659 / 11658 / 11665 / 11658 / 11657 |

**Read these with three cautions.**

1. **Epoch 50, no epoch selection.** `epochs = 50`, and each fold stores one value with no per-epoch
   history. A best-epoch reading needs a re-run (G5).
2. **The folds are not demonstrably the delivered folds.** `b3_hmt_grn.build_fold_split` applies the
   delivered rule (StratifiedGroupKFold, seed 0, int64 userids, `next_category`), and run today on
   today's `check2hgi_dk_ovl/alabama` rows it reproduces `train.py`'s folds 5/5 (2026-10-01,
   nespedgpu). But the June AL run's n_val [19269, 19265, 19264, 19264, 19264] is a different multiset
   from today's [19265, 19264, 19265, 19267, 19265]. So the June run saw a different partition, and it
   was not a NaN filter: `b3` asserts 0 NaN. The cause is most likely a different `dk_ovl/alabama` file at
   the time, but that is not proven. The JSONs store counts only, so the June user sets cannot be recovered.
3. **The `windowing` field is stale.** Every file says "stride-9 (current)", yet the n_val sums are the
   stride-1 row counts (AL 96,326; AZ 200,895; IST-stride1 271,666). Trust the counts, not the label.
   Arizona's five folds being exactly 40,179 each is noted, not explained.

Uncommitted. `docs/results/` is hidden by `.git/info/exclude` on the laptop, so committing needs
`git add -f docs/results/baselines/hmt_grn/`.
