# Pairing decomposition battery — why aligned pairing hurts AL (RESOLVED)

**Date**: 2026-07-01/02 (overnight, A40, single session/device/inductor-cache).
**Question** (user): the G0.1 advisory found `--aligned-pairing` HURTS at AL (cat −4.77,
seed-0, v16 conditions) — "should be better, this is weird"; is the implementation correct
and are we losing anything? Also: test the untested **cond_coupling × aligned-pairing** cell.
**Code re-audit first** (agent, adversarial): NO bug — the aligned path pairs the right rows,
loses no samples, keeps budgets/aux/eval identical; the advisory A/B was clean (same driver,
same recipe, only the flag differed; the init-and-dropout-MATCHED fold-0 alone showed −4.96;
5/5 folds negative, paired t ≈ −7). Full report referenced in `README.md §2`.

## Design

5 arms × seeds {0,1,7,100} × 5 folds, AL, champion **v17** recipe on `check2hgi_dk_ovl`
(fp32, bs8192, per-head LR, `--compile --tf32`, shared cache) — `run_pairing_battery.sh`;
raw per-cell results in `pairing_battery/summary.tsv`.

| arm | what it isolates |
|---|---|
| `base` | champion (independent shuffles; step sees 2×bs distinct windows) |
| `aligned` | `--aligned-pairing` (joint loader, shared perm; row i cross-reads its OWN window) |
| `derange` | **new control**: identical joint machinery/permutations/inits, task-b rolled by 1 → random-partner pairing at aligned-arm structure (`MTL_ALIGNED_DERANGE=1`) |
| `alcond` | aligned + `cond_coupling=posterior cond_dim=7 cond_inject=add` (R-CC recipe) — the untested cell |
| `cond` | conditioning WITHOUT alignment (R-CC's historical confounded form; cond-guard relaxed deliberately) |

`aligned vs derange` is an EXACTLY-matched pair (same generator, same permutations, same
fold inits, same per-step batch structure — only the pairing semantics differ).

## Results (5-fold means; paired deltas over 4 seeds, mean ± sd)

| arm | cat | reg |
|---|---|---|
| base | 64.605 ± 0.066 | 69.816 ± 0.116 |
| aligned | 61.579 ± 0.113 | 69.219 ± 0.112 |
| derange | 64.566 ± 0.036 | 69.853 ± 0.075 |
| alcond | 61.569 ± 0.092 | 69.691 ± 0.241 |
| cond | 64.524 ± 0.076 | 69.849 ± 0.185 |

| paired delta | cat | reg |
|---|---|---|
| aligned − base | **−3.025 ± 0.079** (4/4 neg) | **−0.597 ± 0.075** (4/4 neg) |
| derange − base | −0.039 ± 0.077 (null) | +0.037 ± 0.101 (null) |
| derange − aligned | **+2.986 ± 0.125** | **+0.634 ± 0.064** |
| cond − base | −0.080 ± 0.074 (null) | +0.033 ± 0.263 (null) |
| alcond − aligned | −0.010 ± 0.140 (null) | **+0.472 ± 0.352** (4/4 pos) |
| alcond − base | −3.036 ± 0.122 | −0.125 ± 0.336 |

## Verdicts

1. **The aligned deficit replicates under v17/overlap** (−3.03 cat / −0.60 reg, tiny seed
   variance) — the advisory's −4.77 (v16/v14/bs2048) was real and generalizes, smaller under
   overlap's ~7.6× more training windows.
2. **The deficit is 100% pairing SEMANTICS — mechanism resolved.** The deranged control
   recovers base exactly on BOTH heads while sharing everything with the aligned arm except
   *who* row i cross-reads. Cross-reading a random OTHER window = beneficial input-noise
   regularization; cross-reading your OWN window's other view = an overfit-friendly shortcut
   (advisory best-epochs came systematically earlier under aligned).
3. **The per-step-diversity hypothesis is REFUTED**: derange has the same halved
   distinct-window count per shared-parameter update as aligned (bs vs base's 2×bs) yet
   matches base — per-step set sharing does not matter; loader machinery differences don't
   either (also excluded by the init-matched advisory fold-0).
4. **cond_coupling de-confounded**: alone (random pairing) it is a clean null vs base —
   reproducing the historical R-CC null and confirming the model learns to ignore
   unrelated-row conditioning. WITH alignment it produces a real, consistent reg lift **over
   the aligned arm** (+0.47, 4/4 seeds — the conditioning signal is only usable when
   semantically paired: a genuine mechanism finding), but it recovers only the aligned reg
   dip and never beats base on either head (cat −3.04 / reg −0.13 vs base). **The R-CC
   closure verdict survives de-confounding**: per-sample cat→reg coupling cannot beat the
   champion even with semantic pairing.
5. **Champion default validated**: random pairing is not an accident to fix but the better
   regime at small/mid N; the deployment-style aligned input remains what validation scores,
   so reported numbers stay representative. **Binding-G0.1-grade evidence at AL** (4 seeds ×
   5 folds, paired, same session): aligned pairing is REFUTED as a champion candidate at AL
   on the v17 base. FL stays advisory-null (mechanism-consistent: 13× data → less overfit
   sensitivity). The `future_works` batch-level-pairing idea is DEMOTIVATED (any
   random-other partner ≈ base; there is no gap for a "smarter" partner to close on cat, and
   the alcond reg path is capped below base).

## Caveats

- AL only at n=4-seed grade; FL/large states not re-run (advisory FL null + mechanism
  suggests even less effect). Not the pre-registered "frozen v16 base" binding G0.1 — this is
  the v17 board base, which is what any adoption decision would target today.
- `alcond`/`cond` used `cond_signal=softmax`, `cond_detach=False` (e2e), `cond_inject=add`
  (the R-CC ccplus multiseed recipe); other cond variants untested with alignment.
- Wall-clock note: the joint (aligned/derange/alcond) arms are ~33% slower per run than the
  independent-loader arms (TensorDataset per-sample collate vs `POIDataset.__getitems__`
  batched index_select) — perf-only, but worth fixing if aligned arms are ever revisited.

---

# Addendum 2026-07-02 — the "fair re-tune for aligned" sweep (REFUTED) + advisor panel

**User hypothesis**: maybe aligned just needs its own hyperparameters (LR/epochs) — the recipe
was tuned for random pairing. **Advisor panel** (3 agents: methodology critic, trajectory
analyst, snapshot verifier) + a 6-arm × 2-seed sweep (`run_aligned_retune.sh` +
`run_aligned_retune_pass1b.sh`, `aligned_retune/summary.tsv`).

## Snapshot verification (user question: "are we scoring the pre-overfit peak?")

**YES — verified independently**: the scorer takes an unconditional argmax over all 50
per-epoch val rows per task (`a40_score_matched.py:28-38`), reproduced the battery numbers
digit-for-digit from raw CSVs; val runs every epoch; no min-epoch gate applies; discretization
loss is ≤~0.1-0.3 pp and symmetric. The −3.03 is peak-vs-peak, not a decayed checkpoint.

## Trajectory analysis (all 40 battery val curves)

- Aligned rises at the SAME slope as base (~2.95 pp/ep), starts **above base** (20/20
  folds ahead at epoch 2, per-fold leads up to +5.2 pp, mean +2.7 at ep2 — the self-read
  shortcut genuinely helps early), hovers near parity through ~ep 14, crosses below
  permanently at ep ~15 on the mean curve (per-fold final crossovers ep 8–22, median 16),
  peaks ~18 epochs earlier (ep ~23 vs ~41) at a **3.0 pp lower ceiling**, then decays −0.17
  pp/ep (base: −0.002). *(Numbers re-verified 2026-07-02 by the fact-check pass — an
  earlier phrasing "above base at ep 2–12, 10/10 folds" overstated the lead's duration.)*
- **Ceiling, not timing**: base's un-annealed mid-schedule value at ep25 (63.26) already
  exceeds aligned's all-time peak (61.49); aligned extracts **+0.02 pp** from everything
  after ep25 (base tail: +1.26).
- **Memorization**: aligned train-F1 at ep50 = 79.6 vs base 69.5 (+10.1) while val is
  7.2 pp lower — the own-window cross-read is memorized. derange ≈ base (statistical null)
  already showed the free fix is "don't self-pair".

## Sweep results (AL, seeds {0,1}, 5 folds; battery base mean 64.57, gate = cat ≥ 64.07)

| arm | cat mean | Δ vs base | Δ vs aligned (61.49) |
|---|---|---|---|
| ep25 (schedule-matched anneal) | 61.26 | −3.31 | −0.22 |
| wd 0.10 | 61.52 | −3.05 | +0.04 |
| cat-lr 5e-4 | 59.69 | −4.88 | **−1.79** (harmful) |
| combo (ep25+wd10+lr5e4) | 58.19 | −6.38 | −3.29 |
| **shared-lr 5e-4** (targets the cross-attn group — advisor-added) | 62.26 | −2.31 | **+0.78** |
| **shared_dropout 0.30** (advisor-added) | 62.16 | −2.41 | **+0.67** |

Every arm replicated across both seeds (max seed spread 0.17). **Verdict: REFUTED** — the
pre-registered win condition (cat ≥ base−0.5) is missed by ≥1.8 pp by every arm. The only
positive movers are the two arms attacking the shortcut's medium (the shared cross-attn
group), and they recover only ~0.7–0.8 of the 3.0 deficit — consistent with the trajectory
verdict that the deficit is a ceiling created by the self-pairing shortcut itself. Per the
pre-registration: **no seed extension, no alcond "prize" arm** (its cat leg is
trajectory-proven unreachable and its reg is capped at base parity — the +0.47 lift exists
only relative to aligned, which derange recovers for free). Champion random pairing stands;
mark "aligned + tuned schedule" REFUTED so future agents do not re-open it.

## Concurrency evaluation (user request: parallel cells on the A40)

Measured by running pass-1b concurrent with pass-1 (2-wide, ~23 GB/46 GB VRAM): joint-arm
wall 740 s serial → ~1250 s concurrent (+69%) ⇒ **2-wide throughput ≈ 1.18×**. Steady-state
GPU util is already ~98% at AL/bs8192/compiled, so lanes mostly contend. Verdict: marginal —
usable for bulk null-screening batteries when wall-clock matters, NOT worth it generally,
and never for precision-sensitive (<0.3 pp) paired cells (adds compiled-session jitter) or
for CA/TX (host-RAM serial-only constraint). Matches the perf-study's earlier 5-wide-slower
finding.
