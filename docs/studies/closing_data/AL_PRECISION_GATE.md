# AL precision gate — bf16 vs fp32 (champion-G MTL, gated overlap, seed 0 × 5f)

> H100 board lane (`study/board-h100`), 2026-06-23. §1 of `HANDOFF_BOARD_H100.md`. This gate doubles as the
> AL MTL cell and **BLOCKS all other MTL** until the user picks a board-wide precision. Both arms avoid the fp16
> overflow (so this is bf16-vs-fp32, NOT fp16-vs-anything). Scored fp32 by `scripts/closing_data/h100_score_matched.py`
> (per-task DIAGNOSTIC-BEST, fold-mean). Recipe = champion-G on `check2hgi_dk_ovl`, `--canon none` +
> explicit `--no-reg-class-weights --no-cat-class-weights` (see "recipe note" below).

## Result (4 dp)
| metric | bf16 (Arm X) | fp32 (Arm Y) | Δ (bf16−fp32) | STL ceiling | Δreg vs ceiling |
|---|---|---|---|---|---|
| reg FULL top10 mean | **69.6873** ± 3.3165 | **69.8067** ± 3.3898 | **−0.1194** | 69.98 | bf16 −0.29 / fp32 −0.17 |
| cat macro-F1 mean | **63.5810** ± 1.9732 | **63.5591** ± 2.0387 | **+0.0219** | — | — |

Per-fold reg (bf16 / fp32): [71.9024/72.1256, 69.0147/68.9525, 73.2312/73.5323, 70.6182/70.6752, 63.6699/63.7477]
→ per-fold Δreg = [−0.2232, +0.0622, −0.3011, −0.0570, −0.0778]
Per-fold cat (bf16 / fp32): [63.8521/63.3986, 64.8961/64.9044, 65.12/65.2468, 64.2999/64.5696, 59.7367/59.6762]
→ per-fold Δcat = [+0.4535, −0.0083, −0.1268, −0.2697, +0.0605]

Best-epochs differ between arms (diagnostic-best max-over-50-epoch selection), which is the dominant source of the
sub-0.3pp per-fold jitter.

## Anchor sanity (handoff §1)
- fp32 anchor ≈ 69.80 / ≈63.48 → **fp32 arm reg 69.8067 hits it exactly**; cat 63.5591 ≈ 63.48 (+0.08). ✓
- bf16 should land at fp32; it lands 0.12pp below on reg, 0.02pp above on cat.
- Prior fp16 anchor 69.60 / 63.44 reproduced by re-scoring the kept fp16 run → scorer path validated.

## Decision rule verdict
Rule: `|Δcat|, |Δreg| ≤ 0.05 pp` (per-fold + mean) bf16-vs-fp32 ⇒ standardize **bf16**; else **fp32**.
- Mean **|Δreg| = 0.1194 > 0.05** and several per-fold |Δreg|/|Δcat| > 0.05 ⇒ **by the letter of the rule → fp32.**
- BUT the gap is small (0.12pp reg / 0.02pp cat, both ≪ the ±3.3pp fold std) and bf16 is the fast path — decisive
  for **CA** (the ~40-min bf16 CA run is already restart-truncation-risky; fp32 at CA scale is materially slower and
  more likely to be cut short by the recurring ~1–2 h studio restarts). bf16 was already validated healthy through
  ep28 at CA. → **User call (this gate STOPs for the user).**

## Recipe note (deviation from the literal handoff command)
The handoff §1 command omits `--canon`, relying on canon-v16 auto-injection. Under `MTL_STRICT=1` that injection
trips the **wrong-substrate guard** (v16 pins `check2hgi_design_k_resln_mae_l0_1`; the board intentionally runs on
`check2hgi_dk_ovl`) → hard-fail. Fix: `--canon none` + explicit `--no-reg-class-weights --no-cat-class-weights`
(the only material champion-G flags canon adds beyond the board's explicit set; `--checkpoint-selector geom_simple`
is moot under `--no-checkpoints` + diagnostic-best scoring). `MTL_STRICT=1` still fails loud on NaN (the
`guard_finite_step` reads env independently of canon). This recipe note applies to **every** board state (AZ/FL/CA).

## Rundirs
- bf16: `results/check2hgi_dk_ovl/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260623_171050_38170` (`h100_matched_score.json`, tag `alabama_bf16`)
- fp32: `results/check2hgi_dk_ovl/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260623_171146_38758` (`h100_matched_score.json`, tag `alabama_fp32`)
- env: `MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1`, bf16 = `MTL_AUTOCAST_BF16=1 MTL_DISABLE_AMP_EVAL=1`, fp32 = `MTL_DISABLE_AMP=1`.
- Both: zero non-finite events (no `[NONFINITE-GUARD]`), 50 ep × 5 folds completed clean.

## AL CELL — COMPLETE (small states = fp32, user decision 2026-06-23)
User chose **fp32 for the small/mid states** (and a FL fp32-vs-bf16 gate, see `FL_PRECISION_GATE.md`). So the AL
MTL result is the **fp32 arm**: cat **63.5591** / reg **69.8067** (seed0, 5f, gated overlap).

| AL (seed0, 5f, gated overlap) | value | ceiling | Δ vs ceiling |
|---|---|---|---|
| MTL fp32 cat macro-F1 | **63.5591** | STL cat 55.8704 (5f, `next_gru`) | **+7.69 (beats)** |
| MTL fp32 reg FULL top10 | **69.8067** | STL reg 69.98 (REUSE, fp32) | **−0.17 (closes most of gap)** |

STL cat ceiling scored from `results/check2hgi_dk_ovl/alabama/next_lr1.0e-04_bs2048_ep50_20260622_120957_17857`
via `scripts/closing_data/score_stl_cat_ceiling.py` (the scorer reproduces the committed FL cat ceiling 75.147
exactly → validated).

**STL reg ceiling — fresh 5f artifact built 2026-06-23 (confirms the documented 69.98 scalar):** p1
`next_stan_flow` a0 (TRUE fp32) on dk_ovl, seed0, 5f → AGG Acc@10 **69.99 ± 3.56** (per-fold [72.15, 68.98,
73.44, 71.05, 64.33], best_ep [46,46,44,37,47]). Artifact `docs/results/P1/region_head_alabama_region_5f_50ep_alabama_ovl_stl_reg_s0.json`.
**69.99 ≈ the prior 69.98** → the documented scalar is now backed by an on-disk per-fold artifact. Δreg vs the
fresh ceiling = 69.8067 − 69.99 = **−0.18** (unchanged). **AL cell needs no further runs.**

## Status
**Gate verdict delivered; user chose fp32 for small/mid states. AL cell COMPLETE.** FL runs its own fp32-vs-bf16
gate (`FL_PRECISION_GATE.md`); §4 (CA) precision follows the FL gate outcome.
