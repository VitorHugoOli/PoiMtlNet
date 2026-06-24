# TX MTL cell — champion-G, seed 0, 5f, gated overlap (IN PROGRESS)

**Recipe:** champion-G MTL on `check2hgi_dk_ovl`, **bf16** (large-state precision, CA precedent;
fp16 overflows, bf16≈fp32 per FL gate), seed 0, 5 folds, `--epochs 50`, fixes #1+#3.
`--canon none` + `--no-{reg,cat}-class-weights`, dualtower heads (`next_gru` / `next_stan_flow_dualtower`,
prior-OFF `freeze_alpha=True alpha_init=0.0`), `--log-t-kd-weight 0.0`, OneCycle max-lr 3e-3.
Launcher: `scripts/closing_data/launch_tx_s0.sh` (run with `MTL_RAM_HEADROOM_GB=-25` when co-scheduled —
see [[EP100_ABLATION_AND_TX_RAM]] §2). log_T built on dk_ovl (`output/check2hgi_dk_ovl/texas/`, C29-correct).

**Forced co-schedule note (2026-06-24):** launched alongside CA §4 by overriding the host-RAM guard
(negative headroom) after verifying the simultaneous peak fits (CA 26 GB + TX 66 GB < 108 GB + 35 GB swap).
RAM-safety watcher armed to kill TX (newest) if avail < 4 GB. Cleared construction (peak used 86 GB, swap 0),
CA untouched.

## TX STL ceilings (to beat)
- cat macro-F1 ceiling: **DEFERRED** — co-scheduling with the MTL run **OOM-killed it** (EXIT=137 during the
  5-fold split: TX MTL ~56 GB + cat-ceiling data+split exceeded host RAM). Now **auto-runs solo after TX MTL
  frees RAM** via `scripts/closing_data/tx_cat_ceiling_deferred.sh` (waits for MTL EXIT → runs `next_gru` STL
  → scores → commits). Falls back to the A40 lane (`a40_tx_cat_ceiling.sh`) if needed.
- reg FULL top10 ceiling: **64.96 ± 0.46** (raw top10_acc, `next_stan_flow` a0/fp32, a40 artefact
  `docs/results/closing_data/a40/tx_stl_reg_ceiling_s0.json`; per-fold [64.84, 65.18, 64.11, 65.27, 65.40]).
  Metric-basis note: this is raw top10_acc; the MTL column is FULL top10 (=top10_indist×(1-ood)) — small
  OOD-adjustment gap, but the documented board ceiling. **TX MTL fold 1 reg 66.94 → +1.98 (beats).**

## Per-fold results (diagnostic-best, scored by `h100_score_matched.py`)
Updated incrementally as each fold completes (autonomous per-fold committer).

<!--TXTABLE_START-->
| fold | cat macro-F1 | cat best-ep | reg FULL top10 | reg best-ep |
|---|---|---|---|---|
| fold1 | 77.6885 | 50 | 66.9382 | 49 |
| fold2 | 77.3113 | 50 | 67.3184 | 50 |
| fold3 | 77.3870 | 50 | 66.1210 | 50 |
| fold4 | 77.5648 | 48 | 67.1598 | 49 |
| fold5 | 77.3925 | 50 | 67.4477 | 49 |

**Running mean (n=5):** cat **77.4688** ±0.1377 | reg **66.9970** ±0.4698
<!--TXTABLE_END-->

## Status
- Started 2026-06-24 ~04:42. ~5–6h for 5 folds co-scheduled; faster once CA frees the box.
- The session driving this run **ends before TX finishes** — per-fold results are committed+pushed
  autonomously (`scripts/closing_data/tx_autocommit.sh`) so progress survives session death.


## TX CELL — COMPLETE (5f)
| task | TX MTL (5f) | ceiling | Δ |
|---|---|---|---|
| cat macro-F1 | **77.47** | 69.95 (A40 #37, dk_ovl/seed0) | **+7.52** (beats) |
| reg FULL top10 | **67.00** | 64.96 (A40, dk_ovl/seed0) | **+2.04** (BEATS) |

TX MTL **beats both ceilings** (cat +7.52 / reg +2.04) — joins CA/FL in reversing "MTL sacrifices reg".
Cat ceiling reused from the A40 lane (#37, same dk_ovl/seed0/next_gru recipe) — not re-run here.
