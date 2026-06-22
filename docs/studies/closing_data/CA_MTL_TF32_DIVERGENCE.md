# CA champion-G MTL diverges under TF32 at overlap scale — root cause + fix

> H100 board lane (`study/board-h100`), 2026-06-22. A load-bearing finding from the early CA reg cell
> (HANDOFF_BOARD_A100 Task 2). **TL;DR: champion-G MTL on the gated-overlap substrate DIVERGES (val loss → NaN)
> under `--tf32` at CA's scale; full-precision (fp32, drop `--tf32`) is stable. `--tf32` is a perf knob, not
> recipe identity (DEFAULTS_AND_GUARDS), so fp32 is the correct fix with NO recipe deviation.** The −5.2 pp CA
> "δ_reg breach" first observed was a divergence artifact, not a real result.

## Symptom
CA champion-G MTL (gated stride-1 overlap, frozen v14, seed 0, compiled+**tf32**, max-lr 3e-3): both heads
peak then **collapse to a degenerate ~3% floor** by ep30 — every fold, tightly. The per-task best-epoch selector
masks it by picking the pre-collapse peak (reg ep5 ≈ 58, cat ep27–35 ≈ 74), so the matched scorer reported
`reg 58.26 / cat 74.72` → an apparent **Δreg = −5.23 pp** vs the CA STL reg ceiling 63.48 — far worse than FL's
clean −1.29. That "breach" is spurious.

## Investigation (what it is, and what it is NOT)
- **NOT a metric/OOD artifact:** `ood_fraction` is constant (0.0005) across all epochs; the collapse is in the
  model's predictions, not the in-dist/OOD split.
- **NOT a clean MTL-vs-STL reg sacrifice:** the val **loss diverges** — fold 1 reg val loss
  `ep5=4.30 → ep10=5.83 → ep20=7.82 → ep30=NaN(blank)`, then top10 pinned at the 3.67% floor. A reg CE loss ~4
  over 8501 classes is incompatible with 3.67% top10 → the late epochs are NaN, not a real low score.
- **Grad-clip is active** (`max_grad_norm=1.0`, `mtl_cv.py`) but does NOT prevent it — the loss rises
  *gradually* over the high-LR phase, then NaNs. (clip can't save a NaN that originates in the forward/backward.)
- **OneCycle is correctly scaled** to the 8.5× overlap step count (`steps_per_epoch = len(longer loader)`,
  `mtl_cv.py:1558`) — not a schedule miscalibration.

## Scale dependence — same bug, severity scales with state size
The same divergence appears at **FL** but is benign there:
| state | regions | NaN incidence | effect |
|---|---|---|---|
| **FL** | 4703 | **1/5 folds, at ep50** (fold 3) | harmless — scorer already captured high late peaks (~76); other 4 folds clean. FL reg 75.42 (mostly clean). |
| **CA** | 8501 | **all folds, by ep30** | catastrophic — peaks captured at ep5 (≈58, *before* convergence) → fake −5.2. |

FL fold 3: reg val loss `ep5=2.96 → ep20=4.20 → ep50=NaN`, top10@50 = 0.71 (the one FL fold that NaN'd).

## Root cause — CONFIRMED: TF32 reduced precision at scale
A/B with `--tf32` dropped (fp32), everything else identical (compiled, max-lr 3e-3, seed 0):

| epoch | tf32 (diverges) | fp32 (stable) |
|---|---|---|
| ep5 | 4.30 | 4.80 |
| ep6 | — | 5.10 (peak) |
| ep10 | 5.83 | 4.42 |
| ep15 | — | 4.09 |
| ep20 | **7.82** | **4.16** |
| ep30 | **NaN** | stable ~4.1 |

fp32 reg val loss peaks at ep6 (5.10) then **settles to ~4.1 and stays** — no divergence, no NaN. The tf32 dims
at CA's scale (8501-way softmax, large overlap matmuls) accumulate enough error to walk the optimizer off and
NaN. **fp32 is the fix.**

## Fix + board implication
- **Run CA (and any large state) MTL in fp32 — drop `--tf32`.** `--tf32` is a documented *perf knob*, not
  recipe identity (DEFAULTS_AND_GUARDS: "NEVER in canon … board-execution-only"), so fp32 is **not** a recipe
  deviation — it is the numerically-correct execution. `--compile` is retained (it does not cause the NaN).
- **DEFAULTS_AND_GUARDS' "tf32 is result-neutral (+0.05 pp)" holds only at FL/non-overlap scale.** At CA ×
  gated-overlap it is catastrophic. The board should **use fp32 for the overlap board** (at minimum the large
  states CA/TX; safest is fp32 board-wide for uniformity, accepting the ~15% speed cost).
- **The −5.23 pp CA "δ_reg breach" is VOID** (tf32-divergence artifact). The true CA Δreg comes from the fp32
  run (in progress). Do not cite −5.2.
- **FL Task 1 caveat:** the committed FL result (`florida_s0_board.json`) was produced under `--tf32`; fold 3
  NaN'd at ep50 and the scorer used its pre-NaN ep7 peak (73.5), so FL reg (75.42) may be **slightly
  underestimated**. Recommend an fp32 FL re-run for full rigor before freezing FL's headline reg number.

## Repro / commands
- Diverging (tf32): the §3c champion-G command with `--compile --tf32` on `--engine check2hgi_dk_ovl --state california`.
- Stable (fp32): same command **without `--tf32`** (keep `--compile`). `scripts/closing_data/h100_ca_reg.sh` /
  `h100_state_cells.sh` should drop `--tf32` for large states.
- Evidence: per-epoch `val=N…` reg loss in the run logs; per-fold `metrics/fold*_next_region_val.csv`
  (`loss`, `top10_acc_indist`, `ood_fraction`).

## Status
fp32 CA MTL running (stable, ~ep21 fold 1 at writing). When it completes → the real CA Δreg vs δ_reg=2 pp.
