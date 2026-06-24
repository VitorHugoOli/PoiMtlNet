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
- cat macro-F1 ceiling: **TBD** (A40 lane `a40_tx_cat_ceiling.sh` — fill when scored)
- reg FULL top10 ceiling: **TBD**

## Per-fold results (diagnostic-best, scored by `h100_score_matched.py`)
Updated incrementally as each fold completes (autonomous per-fold committer).

<!--TXTABLE_START-->
| fold | cat macro-F1 | cat best-ep | reg FULL top10 | reg best-ep |
|---|---|---|---|---|
| _(pending — fold 1 in progress)_ | | | | |

**Running mean:** _pending._
<!--TXTABLE_END-->

## Status
- Started 2026-06-24 ~04:42. ~5–6h for 5 folds co-scheduled; faster once CA frees the box.
- The session driving this run **ends before TX finishes** — per-fold results are committed+pushed
  autonomously (`scripts/closing_data/tx_autocommit.sh`) so progress survives session death.
