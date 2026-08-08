# v18 — re-tuning sweep: test register and execution order

> **Why this exists.** The v18 category recipe was inherited from the **leaked** substrate. Both arms
> peak early and the baseline is not a ceiling by the study's own standard. This file is the running
> register of every sweep arm, its order, and its status, so the work is not lost between sessions
> and so we can return to the evaluation waves afterwards.
>
> Status legend: ⬜ queued · 🔄 running · ✅ done · ❌ failed · ⏭ skipped
>
> **All arms fp32 on both models** (user decision 2026-08-07), `--compile --tf32`, 5 folds,
> engine `check2hgi_v18`, per-command `env` (never a bare `export` — see
> [`PRECISION_CAVEAT.md`](PRECISION_CAVEAT.md)).

## Why the sweep is split by state size

Measured on v18 seed 0 — the failure mode **inverts** with data size, so one recipe cannot serve both:

| state | windows | peak−final | **train−val gap @ep50** | diagnosis |
|---|---:|---:|---:|---|
| alabama | 96 k | +4.08 | **+42.45** | severe overfit |
| arizona | 201 k | +4.29 | **+41.15** | severe overfit |
| istanbul | 272 k | +3.68 | +7.09 | mild overfit |
| florida | 1.27 M | +1.59 | +3.48 | mild |
| california | 2.93 M | +0.61 | **+0.52** | **no overfit** |
| texas | 3.83 M | +1.20 | **+0.25** | **no overfit** |

Small states memorize (AL train F1 66.4 vs val 24.0). Large states show **no train–val gap at all** —
they are capacity-limited, not overfitting. So the small-state grid goes **down** in LR and the
large-state grid goes **up**. This matches v17's own conclusion that the cat optimum is
state-size-dependent (it already tiered bs2048/bs8192 by size).

## The class-weight asymmetry (verified 2026-08-07)

| arm | class-weighted CE? | set where |
|---|---|---|
| **dedicated** (`--task next`) | **YES** | `default_next` sets `use_class_weights=True`; read at `next_cv.py:140` |
| **MTL** (`--task mtl`) | **NO** | C25 (2026-06-05): `default_mtl` sets `use_class_weights_cat=False`, `use_class_weights_reg=False` |

**`--no-cat-class-weights` is INERT on `--task next`** — it sets `use_class_weights_cat`, which only
`mtl_cv.py:497` reads. The working flag for the dedicated arm is **`--no-class-weights`**
(`train.py:499-504`, `dest=use_class_weights`).

C25's own record: *"cat: EMPIRICALLY unweighted also wins macro-F1 (+5.1 pp AL: 48.37→53.51) — the
'balancing helps macro-F1' assumption was FALSE."* That was measured on the **MTL** cat head, on the
leaked substrate. If it transfers to the dedicated arm on v18, our ceiling is **under-estimated** and
every Δcat is **too positive** — i.e. the bias runs *against* the "MTL matches" reading, not for it.

## Execution order

Sequential on the single A40 unless a row says otherwise. GPU is compute-saturated at ~98 % by one
small-state job, so 2-wide buys only ~1.18× and is used solely where a row says so.
**CPU execution is excluded**: CUDA and CPU kernels give different floating-point results, so CPU
arms would not be comparable to GPU arms — the same class of error as the fp16/fp32 leak.

## STATUS AT A GLANCE (updated 2026-08-08)

| question | answer |
|---|---|
| Dedicated (STL) sweep settled at AL/AZ/IST? | **YES — 30/30 arms done** (rows 1+2). Winner per state below. |
| Dedicated row 4 (`--no-class-weights`)? | 🔄 **queued**, runs last in queue2 |
| MTL cat-LR grid {5e-4,1e-3,2e-3} + 25-ep arm (row 3)? | **AL ✅ done · AZ 🔄 running · IST 🔄 queued** |
| MTL cat-class-weights {on,off} × category-weight {0.75,0.5} (row 7)? | **AL ✅ done (2×2 complete) · AZ 🔄 running · IST 🔄 queued** |
| P1 capacity control (after T2)? | ❌ **NOT run — deliberately HELD** by author decision 2026-08-07 → [`POSTPONED.md`](POSTPONED.md) P1 |
| CA/TX sweep (rows 5, 6, 8)? | ❌ **NOT run — POSTPONED for time** 2026-08-08 → [`POSTPONED.md`](POSTPONED.md) P6 |

### Dedicated (STL) winners — rows 1+2 complete, seed 0, fp32, sm3 selector

| state | current recipe | sm3 | **retuned winner** | sm3 | gain | med best-epoch |
|---|---|---:|---|---:|---:|---:|
| alabama | bs2048 @ 0.005 | 27.122 | **bs8192 @ 0.0025** | 27.610 | **+0.49** | 8 → 13 |
| arizona | bs8192 @ 0.005 | 31.103 | **bs8192 @ 0.0005** | 31.668 | **+0.57** | 10 → 17 |
| istanbul | bs2048 @ 0.005 | 30.998 | **bs2048 @ 0.0005** | 32.077 | **+1.08** | 6 → 16 |

Lower LR wins at all three, and the median best epoch moves from 6–10 into 13–23 — the signature v17's
76-arm sweep identified (early-peaking arms were never ceilings, 0/27). On *argmax* istanbul's old
recipe still looks best (32.722); its sm3 is the **worst** (30.998), which is the argmax-noise effect
this sweep exists to remove.

### MTL winner so far — alabama only (AZ/IST in flight)

| arm | cat sm3 | reg |
|---|---:|---:|
| cw0.75, class-weights **OFF** (current recipe) | 26.153 | 69.683 |
| **cw0.75, class-weights ON** | **27.276** | 69.696 |
| cw0.50, class-weights ON | 27.003 | 69.709 |
| cw0.50, class-weights OFF | 26.915 | 69.687 |

**The finding is the ON/OFF axis, not the loss split.** Paired per-fold at n=5:

| contrast | cat sm3 | p | reg | p |
|---|---:|---:|---:|---:|
| class-weights **ON vs OFF** (@cw0.75) | **+1.123** | **0.004** | +0.013 | 0.90 |
| cw0.50/ON vs cw0.75/ON | −0.273 | 0.62 | +0.014 | 0.90 |

Class weights ON is solid; **the 0.75-vs-0.50 choice is a coin flip at this n** and is left for AZ/IST
to break. Which of the two ON arms "wins" is selector-dependent — `cw0.50/ON` takes cat-argmax, reg
and geom_simple-argmax; `cw0.75/ON` takes cat-sm3 and geom_simple-sm3 (author observation 2026-08-08,
correcting an earlier read that looked only at category):

| arm | cat argmax | cat sm3 | reg | geom argmax | geom sm3 |
|---|---:|---:|---:|---:|---:|
| cw0.75 / OFF (current) | 27.3836 | 26.1532 | 69.6831 | 43.68 | 42.69 |
| cw0.75 / ON | 27.8604 | **27.2760** | 69.6956 | 44.07 | **43.60** |
| cw0.50 / OFF | 27.6771 | 26.9152 | 69.6873 | 43.92 | 43.31 |
| cw0.50 / ON | **27.9534** | 27.0034 | **69.7091** | **44.14** | 43.39 |

cat-LR is near-null across 4× its range (26.15–26.37 sm3). The 25-epoch arm gains cat +0.64 but costs
reg −0.79 — rejected.

Net at alabama, both arms fairly tuned and both fp32: **Δcat −0.97 → −0.33**.

## Row register

| # | stage | arm | states | grid | arms | status |
|---|---|---|---|---|---|---|
| 1 | small-state dedicated | dedicated | AL, AZ, IST | bs {2048, 8192} × max_lr {0.0005, 0.001, 0.0025, 0.005} | 24 | ✅ done |
| 2 | schedule shape | dedicated | AL, AZ, IST | epochs {15, 25} @ anchor | 6 | ✅ done |
| 3 | MTL cat-LR | MTL | AL, AZ, IST | cat-lr {5e-4, 1e-3, 2e-3} + 25-ep arm | 4/state | AL ✅ · AZ 🔄 · IST 🔄 |
| 4 | dedicated class weights | dedicated | AL | `--no-class-weights` × max_lr {0.005, 0.0025} | 2 | 🔄 queued |
| 5 | large-state dedicated | dedicated | **TX, 1 fold** | max_lr {0.005, 0.0075, 0.01} × epochs {50, 75} | 5 | ⬜ queued |
| 6 | dedicated class weights @ large | dedicated | **TX, 1 fold** | `--no-class-weights` | 1 | ⬜ queued |
| 7 | MTL knobs | MTL | AL, AZ, IST | class-weights {on,off} × cw {0.75, 0.5} | 3/state | AL ✅ · AZ 🔄 · IST 🔄 |
| 3b | MTL cat-LR @ large | MTL | **FL, 1 fold** | cat-lr {5e-4, 1e-3, 2e-3} + 25-ep | 4 | ⬜ queued |
| 8 | MTL @ large | MTL | TX/CA | carried from #7 | 2 | ⏸ **POSTPONED** (P6) |
| 9 | confirm | both | winners | best dedicated + best MTL | — | ⬜ |
| T3′ | trunk re-test on the adopted recipe | MTL | FL | `disable_cross_attn=True` | 1 | ⬜ after sweep |
| P1 | capacity-matched region control | dedicated reg | AL, CA | `d_model` 480 / 352 | 2 | ⬜ after sweep |

### 1-fold screens — scope decision 2026-08-08, and its limits

Rows 5, 6 and 3b run **one fold** to buy large-state evidence cheaply: rows 5+6 drop from ~9.6 h to
**~1.9 h**, row 3b from ~6.5 h to **~1.6 h**. This directly reduces the transfer risk recorded in
[`POSTPONED.md`](POSTPONED.md) P6 — some large-state evidence beats none.

**What one fold can and cannot do.** The effects measured at the small states were +0.49…+1.08 sm3;
TX dedicated fold-σ is ~0.30. One fold gives no dispersion, no paired test and no interval, so it can
pick a **direction** but cannot certify a winner — the same rule the region triage driver states. Arms
are compared **on the same fold**, so the contrast stays paired and fair even if fold 0 is not
representative in absolute level.

⚠ **Use `--only-fold 0`, never `--folds 1`.** `--folds N` overrides `k_folds` to `max(2,N)`, so a
1-fold run against a 5-fold-built `log_T` leaks 30–40 % of the validation transitions into the prior
and **inflates reg Acc@10 by 13–23 pp** (`mtl_cv.py:680-688`; documented in
`region_1fold_triage/run_1f.sh`). At row 3b that would manufacture the very reg number being measured.

### Inserted 2026-08-07 — trunk arms at alabama, BEFORE the MTL sweep rows

Ordering is deliberate (user decision): the trunk result can change what the MTL sweep should test,
so it runs **before** rows 3 / 7 / 8. Row 3 (MTL cat-LR grid) is deferred until these report.

| # | stage | arm | state | flag | seeds | cost | status |
|---|---|---|---|---|---|---|---|
| T1 | trunk | **A** no sharing | alabama | `--model-param disable_cross_attn=True` | 0 | 359 s | ✅ Δcat **−0.015**, Δreg **−0.138** |
| T2 | trunk | **A′** no mixing, same depth | alabama | `--model-param identity_cross_attn=True` | 0 | 414 s | ✅ Δcat **−0.154**, Δreg **−0.004** |
| T3 | trunk | **A** no sharing | florida | `--model-param disable_cross_attn=True` | 0 | ~41 min | 🔄 running |
| T4 | trunk | **A′** | florida | `identity_cross_attn=True` | 0 | ~45 min | ⏸ conditional on T3 (POSTPONED.md P5) |

**T1/T2 result: at ALABAMA the trunk is inert, not harmful.** The "trunk adds capacity that feeds
memorization" hypothesis is falsified *at alabama* — severing it neither helps nor hurts (|Δ| ≤ 0.154
on either head, against fold-σ ~1.7 cat / ~3.0 reg).

⚠ **SCOPE — do not generalise this.** This is a 5-fold measurement at ONE small state. The CA/TX/FL/
IST evidence is a **1-fold screen** which, by its own driver's rule, has power only for a
several-point effect: it refutes "the trunk carries the +2 pp" and licenses **nothing smaller**. A
5-fold trunk ablation at CA/TX does **not** exist ([`POSTPONED.md`](POSTPONED.md) P4, deferred by
decision 2026-08-07). Statements of the form "the trunk is inert at every scale" are **not
supported**; the supported statement is "inert at alabama (5-fold), and not the source of the CA/TX
advantage (1-fold screen)".

The compute observation (champion-G 1094 s vs T1 359 s at alabama) is likewise **alabama-scoped** and
is not a basis for changing the architecture until CA/TX are measured at 5 folds.

**Hypothesis under test at alabama is the OPPOSITE of the one at CA/TX.** There the question was
"does severing cost the +2 pp?" (answer: no — see
[`region_1fold_triage/FINDING.md`](region_1fold_triage/FINDING.md)). Here the question is **does
severing HELP?** At alabama the joint model is *worse* than dedicated on both heads (Δcat −0.65 /
−1.04, Δreg −0.31 / −0.33) while the dedicated arm overfits badly (train macro-F1 66.4 vs val 24.0).
If the trunk is adding capacity that feeds memorization, A and A′ should **beat** champion-G.

Falsification: if A and A′ land within fold-σ of champion-G, the trunk is simply inert at small data
too, and no trunk-side change will fix the alabama deficit.

⚠ Caveat on A′ (from the Fable review): `identity_cross_attn` zeroes the attention *output* but keeps
the attention weights (~1.05 M params) present-but-unused, so it is a control for **per-stream FFN
depth**, not for attention capacity. A vs A′ decomposes mixing vs depth; neither controls the
2.5–5.9× dual-tower region pathway.

Rows 4 and 6 test the user's hypothesis that class weights should be off on **all** arms; row 6 is
included because the class-weight effect may itself be size-dependent (macro-F1 balancing matters
more where the model memorizes).

Row 8 is **conditional** to avoid spending ~2 h on a null: TX MTL cells are the most expensive in the
matrix.

## Cost (measured, not extrapolated)

Per 5-fold run at fp32: AL dedicated ~350 s · AZ ~310 s · IST ~800 s · TX dedicated ~5 700 s ·
AL MTL ~1 340 s · TX MTL ~22 500 s. Rows 1–3 ≈ 9 h; rows 4–8 ≈ 8 h; row 9 ≈ 2 h.

## Decision rules (pre-registered, so the choice is not made after seeing the numbers)

1. **Selector.** Every arm is scored under **both** conventions: `argmax` (the board's diag-best) and
   `sm3` (raw value at the 3-epoch-smoothed argmax; see [`stage0_rescore.py`](stage0_rescore.py)).
   The winner is chosen on **sm3**, because argmax is the high-variance statistic this sweep exists
   to fix. Both are reported.
2. **Winner = per-state max of the mean over seeds**, exactly as `cat_ceiling_sweep/aggregate.py`
   does. Median best-epoch > 15 is recorded as a **sanity flag, not a selector** — v17 evidence is
   that early-peaking arms are never ceilings (0/27), but late-peaking does not guarantee a ceiling
   either (v17's low-LR arms peaked late and sat 3.2 pp below).
3. **Both arms are re-tuned.** Re-tuning only the baseline would bias Δcat against MTL.
4. ~~A retuned recipe is adopted board-wide only if it wins at both a small and a large state~~
   **SUPERSEDED 2026-08-08 (author decision, time constraint).** The CA/TX sweep is postponed, so the
   winner from AL/AZ/IST is **replicated to the large states without large-state validation**.

   ⚠ **The evidence argues against this and the risk must travel with the numbers.** The failure mode
   *inverts* with data size (see the table at the top of this file): AL/AZ overfit catastrophically
   (train−val +42 pp) while CA/TX show **no train−val gap at all** (+0.25 / +0.52) and are
   capacity-limited. The small-state fix is **lower LR**, which is the opposite of what a
   capacity-limited model needs. Replicating it to CA/TX may *reduce* their category scores.

   Two consequences to state wherever a large-state category number is reported: (a) the CA/TX
   category recipe is **transferred, not validated**; (b) if a CA/TX category number drops relative to
   the current recipe, that is the expected direction of this risk and not a new finding. The
   alternative — keep CA/TX on their existing v17 large-state tier (bs8192 @ 0.005) and retune only
   the small states — is the tiered option v17 itself used, and remains available at zero cost.

## Returning to the waves

After the sweep the evaluation waves resume with the adopted recipe:

- `docs/results/closing_data/v18/` holds **29 completed cells** (seed 0 all six states × 3 families;
  seed 1 istanbul/alabama/arizona × 3 + florida cat/reg). They are **resume-safe**: `run_wave.sh`
  skips any cell whose sidecar exists.
- ⚠ Cells produced **before** the retune used the inherited recipe and the leaked-substrate
  schedule. If the retune changes the recipe, the **category** cells must be regenerated; the
  **region** cells are unaffected (region peaks late and shows no argmax artifact — noise ratio vs
  v17 is 0.98).
- The two **void** cells regardless of the retune: florida cross-seed cat, texas s0 Δcat
  (both precision-confounded — [`PRECISION_CAVEAT.md`](PRECISION_CAVEAT.md)).
- Remaining to finish the n=10 matrix: florida joint s1, texas ×3 s1, california ×3 s1.
