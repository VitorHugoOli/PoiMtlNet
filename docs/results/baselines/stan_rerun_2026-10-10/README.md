# STAN baseline completion, nespedgpu, 2026-10-10 (provenance, uncommitted)

This was requested by knowladge and authorized by the author on 2026-10-10. It completes the STAN cells of Ch.5's region
table. It ran on nespedgpu (RTX 6000 Ada), one job at a time, after the JISA campaign ended.

- **Code:** a detached worktree `/dados/poimtlnet/wt_stan` at origin/main `93a1a6c0`. `research/baselines/stan/train.py`
  md5 `52b91f65…` (identical to the shared checkout's), `model.py` `52d6bd81…`. Python 3.12.3, torch 2.11.0+cu128,
  venv `~/.venv`.
- **Inputs:** the existing STAN ETL, read-only, no rebuild: `~/PoiMtlNet/output/baselines/stan/<state>/inputs.parquet`.
  - arizona md5 `94074f34a0c8…`, file dated 2026-06-25 22:07, before the printed JSON (22:55);
  - california `7a3812237b1d…` and texas `e29bf333031c…`, both dated 2026-07-06.
- **Runner:** `stan_rerun_2026_10_10.sh` (md5 `bd4ced32…`). `STAN_LEDGER.txt` holds every command, rc, wall time and
  per-fold metric. The temperature, load, memory and boot watch went to `/dados/poimtlnet/stan_rerun/watch.log`.

## (1) Arizona, all 5 folds rerun with the printed cell's recipe

`python -m research.baselines.stan.train --state arizona --folds 5 --epochs 200 --seed 0 --patience 20 --compile --tag v5_compiled_rerun20261010`
runs in fp32 with compile and wrote `faithful_stan_arizona_5f_200ep_v5_compiled_rerun20261010.json` (md5 `35460f6e…`).
It ran rc 0 in 493 s. The printed `faithful_stan_arizona_5f_200ep_v5_compiled.json` is untouched.

| fold | Acc@10 rerun | Acc@10 printed | best_epoch | n_epochs_ran |
|---:|---:|---:|---:|---:|
| 0 | 0.5591 | 0.5649 | 5 | 25 |
| 1 | 0.5612 | 0.5585 | 7 | 27 |
| 2 | **0.2579** | **0.2684** | 9 | 29 |
| 3 | 0.5411 | 0.5413 | 6 | 26 |
| 4 | 0.5526 | 0.5597 | 6 | 26 |
| mean | 49.44 | 49.86 | | |

**Verdict (knowladge, 2026-10-10):** fold 2 reproduces its failure. The printed cell is STAN's real behaviour on that
split, not a bad run, so **the printed cell stays at 49.86.**
- **The training curve rules out an optimisation failure.** On fold 2 the training loss falls steadily from 7.07 to 2.36
  with no NaN. Validation Acc@10 levels off near 0.25 by epoch 4, peaks at 0.2579 at epoch 9, then drops back to 0.1995
  as the model overfits; patience stopped it at epoch 29.
- **The other folds behave differently:** they reach about 0.55 by epochs 3 to 4.
- **Determinism:** the run is not seed-deterministic (`use_deterministic_algorithms(False)` and compile), but the split
  is, so fold 2 holds the same users in both runs.

**Split diagnostics: a lead, not a proof.** These were recomputed on CPU with the trainer's exact
`StratifiedGroupKFold(5, shuffle=True, random_state=0)`, stratified on `target_category` and grouped by `userid`.

| fold | n_val | val users | largest val user share | val target region unseen in train | val share in train-top-10 regions |
|---:|---:|---:|---:|---:|---:|
| 0 | 44,926 | 669 | 0.124 | 0.0024 | 0.202 |
| 1 | 44,927 | 659 | 0.100 | 0.0012 | 0.183 |
| 2 | 44,925 | 669 | 0.048 | 0.0013 | 0.196 |
| 3 | 44,925 | 670 | 0.070 | 0.0017 | 0.193 |
| 4 | 44,922 | 664 | 0.101 | 0.0022 | 0.215 |

Fold 2 is ordinary in size, in its rate of unseen target regions and in its share of the train-top-10 regions.
- **What stands out:** in folds 0, 1 and 4 the largest validation user holds 10 to 12 % of the validation rows. Prefix
  expansion lets heavy users dominate the row count. In fold 2 the heavy users sit in train instead.
- **Why it stays a lead:** fold 3, at 7 %, is normal.
- **Not pursued:** the per-user breakdown, which knowladge judged unnecessary for the deposit.

## (2) California folds 2, 3, 4 and (3) Texas fold 4

Recipe per `docs/studies/closing_data/v17_completion/stan_catx/STATUS.md` §To RESUME, using the same commands as
`run_stan_interleaved.sh` but run from the worktree:
`python research/baselines/stan/train.py --state <s> --seed 0 --epochs 200 --folds 5 --only-fold k --batch-size 2048 --amp bf16 --compile --d-model 128 --patience 10 --tag v6_p10`
- **Order:** CA2, TX4, CA3, CA4.
- **Skips:** any fold whose JSON exists is skipped. The worktree already holds CA fold 0–1 and TX fold 0–3.
- **Retries:** at most 2 attempts, and only when rc ≠ 0 or the JSON is missing.
- Results are added below as each fold lands. The ledger records best_epoch and n_epochs_ran per fold, so the
  patience-10 neutrality claim ("best_epoch ≤ 3") can be re-checked.

### Results (chain finished 2026-10-10 19:07:29; `STAN RERUN ALL DONE`; every rc 0, all on the first attempt)

Per fold, from the per-fold JSONs (`per_fold[0]`). Acc@10, Acc@5 and MRR are in %. "tree" means the JSON was already
in the worktree before this chain; "rerun" means this chain produced it.

| state | fold | Acc@10 | Acc@5 | MRR | best_epoch | n_epochs_ran | source | JSON md5 |
|---|---|---:|---:|---:|---:|---:|---|---|
| CA | 0 | 58.43 | 49.25 | 38.09 | 2 | 12 | tree | — |
| CA | 1 | 58.60 | 49.35 | 38.11 | 2 | 12 | tree | — |
| CA | 2 | 59.05 | 49.75 | 38.54 | 2 | 12 | rerun | `bc666da2…` |
| CA | 3 | 59.18 | 50.10 | 38.83 | 2 | 12 | rerun | `8cee0fcf…` |
| CA | 4 | 57.76 | 48.76 | 37.65 | 2 | 12 | rerun | `a81c403f…` |
| TX | 0 | 61.48 | 51.36 | 38.05 | 2 | 12 | tree | — |
| TX | 1 | 61.69 | 51.72 | 38.87 | 2 | 12 | tree | — |
| TX | 2 | 61.50 | 51.40 | 38.70 | 2 | 12 | tree | — |
| TX | 3 | 62.00 | 51.75 | 38.75 | 2 | 12 | tree | — |
| TX | 4 | 62.09 | 52.17 | 38.79 | 3 | 13 | rerun | `7a51bab2…` |

- **CA 5/5:** Acc@10 mean **58.61**, sd (ddof 1) 0.56.
- **TX 5/5:** Acc@10 mean **61.75**, sd (ddof 1) 0.28. The printed cell had 4/5 folds = 61.67.
- **Patience-10 neutrality:** best_epoch ≤ 3 in all 10 CA/TX folds, so training stopped 10 epochs after the best each time.
- **Wall times:** CA2 6,265 s, TX4 6,817 s, CA3 5,687 s, CA4 6,053 s.
  - These are not comparable with each other: CPU embedding builds were running on the same machine at the same time.
- **Acc@1 is the JSON key `accuracy`.** The ledger lines print it as Acc@1.
- **Not a verdict:** whether and how these numbers enter Ch. 5 is knowladge's and the author's decision. This
  directory is provenance only.
