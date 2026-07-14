# closing_data — the MobiWac paper's experimental study (BOARD COMPLETE)

> **Status (2026-07-13, PR #63): the v17 board is n=20 at all six datasets — complete.** M1-full (pre-registered
> 6-dataset Holm, ALL-REJECT @ α=0.05) and the joint-best lane are done; no verdict changed. Residual
> non-verdict work (A2 ReHDM CA/TX running + optional coverage/doc chores) is tracked in
> [`v17_completion/README.md`](v17_completion/README.md).

## The essential files (this folder)
| File | What |
|---|---|
| [`RESULTS_BOARD.md`](RESULTS_BOARD.md) | ⭐ **The board — single source of truth for every paper number** (§1 headline, §3 file-map cell→JSON, §4 baselines). |
| [`log.md`](log.md) | The append-only outcomes log (what happened, when, decided by whom). |
| [`RUN_MATRIX.md`](RUN_MATRIX.md) | The frozen recipe/scope pins (§0: engine `check2hgi_dk_ovl`, precision, scorer). |
| [`perhead_lr_n20.md`](perhead_lr_n20.md) | v17 AL/AZ/FL MTL n=20 source record (the v17 board's small/mid-state cells). |

## Subfolders
| Folder | What |
|---|---|
| [`v17_completion/`](v17_completion/README.md) | The v17 completion track (COMPLETE): [`CEILINGS_N20_FINAL.md`](v17_completion/CEILINGS_N20_FINAL.md) (the n=20 best-vs-best ceilings + Δs), [`STATISTICAL_PROTOCOL.md`](v17_completion/STATISTICAL_PROTOCOL.md) (pre-registered tests), [`stats_n20/`](v17_completion/stats_n20/RESULTS.md) (M1-full Holm output), [`PRECISION_LESSONS.md`](v17_completion/PRECISION_LESSONS.md) (operative fp16/bf16 rules), sweep/run records. |
| [`joint_best/`](joint_best/README.md) | The deployable single-checkpoint lane (J1/T6, ex-`closing_data_v2`): scoring contract [`JOINT_BEST_SCORING.md`](joint_best/JOINT_BEST_SCORING.md), results, provenance, `score_all.py`. |
| [`catx_v17_seed0_5f/`](catx_v17_seed0_5f/RESULTS.md) | CA/TX v17 seed-0 run record (the s0 arm of the n=20 cells). |
| [`archive/`](archive/README.md) | Everything spent: origin plans, gate verdicts, per-machine handoffs, run logs, **`findings/`** (W6 probe, CSLSL cascade, Istanbul baselines, faithful-STAN findings), **`lessons/`** (fp16/bf16 post-mortems). Provenance only — never current state. |

**Raw result JSONs** live under `docs/results/closing_data/` (+ `docs/results/{P0,P1,baselines,second_dataset,pre_freeze_gates}/`) —
the board's §3 maps every cell to its exact JSON.
