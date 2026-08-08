# v18 re-tuning sweep — findings, in full, for audit

> **Written for auditing, not for reading.** Every number below is a 5-fold seed-0 mean on the v18
> engine, fp32, `--compile --tf32`. Paired tests are per-fold across the same 5 folds (n=5), so they
> are matched but underpowered — read p-values as *"is the sign stable across folds"*, not as proof.
> Raw sidecars: `docs/results/closing_data/v18_sweep/*.json` (each carries `*_per_fold`).
>
> Selector conventions: **argmax** = the board's diag-best; **sm3** = raw metric at the 3-epoch
> smoothed argmax ([`stage0_rescore.py`](stage0_rescore.py)). The pre-registered rule picks on
> **sm3**. Both are shown so a reader can check the choice is not selector-shopping.
>
> Status 2026-08-08: rows 1, 2, 3, 4, 7 complete. Row 10 running. Rows 5, 6, 3b queued. Row 8
> postponed. Row 9 pending a recipe decision.

---

## 1 · Headline: what survives three states, and what does not

| finding | states | verdict |
|---|---|---|
| **Dedicated ceiling was mistuned; lower LR wins** | AL, AZ, IST | ✅ **ROBUST** — wins at all three, +0.49 / +0.57 / +1.08 sm3 |
| **Shared trunk is inert** | AL, FL (5-fold) | ✅ robust where measured; **not** established at CA/TX |
| MTL cat-LR matters | AL, AZ, IST | ❌ **null** — spans 0.22 / 0.21 / 0.28 sm3 across 4× the LR range |
| MTL 25-epoch schedule | AL, AZ, IST | ❌ **rejected** — trades cat for reg |
| **MTL class weights ON helps** | AL only | ⚠️ **DOES NOT REPLICATE** — see §3 |
| Removing class weights from the dedicated arm | AL (5f) | ❌ **hurts** ~1 pp — see §4 |
| **…but at TEXAS it HELPS ~1.2 pp** | TX (1f) | ⚠️ **sign flips with data size** — see §7.2. AZ/IST queued to map it |
| TX dedicated LR / schedule grid | TX (1f) | ❌ **flat** — no change warranted; recipe tiers by size — see §7.1 |
| MTL batch size (16k / 32k) | AL | ❌ **keep bs8192** — larger batches cost ~1.3 reg — see §7.3 |
| cw 0.75 vs 0.50 | AL, AZ, IST | ⚠️ **tie-break reversed** — see §3.2 |

**The one-sentence summary: the dedicated baseline was genuinely mistuned and is now fixed; the MTL
side has no knob that survives three states.**

---

## 2 · Dedicated (STL) re-tune — rows 1 + 2, 30 arms, COMPLETE

Full per-state tables are in [`SWEEP_PLAN.md`](SWEEP_PLAN.md). Winners on sm3:

| state | current recipe | sm3 | **retuned winner** | sm3 | gain | median best epoch |
|---|---|---:|---|---:|---:|---|
| alabama | bs2048 @ 0.005 | 27.122 | **bs8192 @ 0.0025** | 27.610 | **+0.49** | 8 → 13 |
| arizona | bs8192 @ 0.005 | 31.103 | **bs8192 @ 0.0005** | 31.668 | **+0.57** | 10 → 17 |
| istanbul | bs2048 @ 0.005 | 30.998 | **bs2048 @ 0.0005** | 32.077 | **+1.08** | 6 → 16 |

**Why this is the most trustworthy result in the sweep:**
1. It reproduces at **all three** states, in the **same direction**, with the same mechanism.
2. The median best epoch moves out of the early-peak zone (6–10) into 13–23 — the exact signature
   v17's own 76-arm sweep used to identify a mis-tuned arm (**0 of 27** early-peaking arms was ever a
   ceiling; they averaged −2.59 pp below).
3. It is **selector-honest**: at istanbul the *old* recipe still wins on argmax (32.722 vs 32.424)
   while losing badly on sm3 (30.998 vs 32.077). That is argmax reading a 1–3-epoch spike. If we had
   judged on argmax alone we would have kept the worse recipe.

⚠️ **Not validated at a large state.** Rows 5/6 (TX, 1 fold) are queued. The failure mode inverts
with data size — AL/AZ overfit (train−val +42 pp) while CA/TX show **no train−val gap** (+0.25 /
+0.52) — so a lower LR may *hurt* the large states. See [`POSTPONED.md`](POSTPONED.md) P6.

---

## 3 · MTL knobs — rows 3 + 7, 21 arms across three states, COMPLETE

### 3.1 Full table (5-fold seed-0 means)

| state | arm | cat sm3 | cat argmax | reg | wall |
|---|---|---:|---:|---:|---:|
| alabama | cat-lr 5e-4 | 26.3148 | 27.1045 | 69.7579 | 583 s |
| alabama | cat-lr 1e-3 *(anchor)* | 26.1532 | 27.3836 | 69.6831 | 552 s |
| alabama | cat-lr 2e-3 | 26.3657 | 27.5131 | 69.6592 | 552 s |
| alabama | cat-lr 1e-3 @25 ep | 26.7901 | 27.6177 | **68.8910** | 290 s |
| alabama | **cw0.75 ON** | **27.2760** | 27.8604 | 69.6956 | 552 s |
| alabama | cw0.50 OFF | 26.9152 | 27.6771 | 69.6873 | 552 s |
| alabama | cw0.50 ON | 27.0034 | **27.9534** | **69.7091** | 552 s |
| arizona | cat-lr 5e-4 | 30.8518 | 31.5924 | 59.5326 | 1155 s |
| arizona | cat-lr 1e-3 *(anchor)* | 30.7987 | 31.6874 | 59.5411 | 1133 s |
| arizona | cat-lr 2e-3 | **31.0127** | 31.7250 | 59.5002 | 1134 s |
| arizona | cat-lr 1e-3 @25 ep | 30.9994 | 31.6297 | 59.2638 | 582 s |
| arizona | **cw0.75 ON** | 31.0282 | 31.6232 | **59.6212** | 1133 s |
| arizona | cw0.50 OFF | 30.8695 | **31.7298** | 59.3295 | 1131 s |
| arizona | cw0.50 ON | 30.9327 | 31.6155 | 59.5097 | 1131 s |
| istanbul | cat-lr 5e-4 | 32.0349 | 32.5243 | 75.4018 | 1539 s |
| istanbul | cat-lr 1e-3 *(anchor)* | 31.9130 | 32.6985 | 75.3631 | 1507 s |
| istanbul | cat-lr 2e-3 | 31.7531 | 32.7514 | 75.3422 | 1508 s |
| istanbul | cat-lr 1e-3 @25 ep | **32.2227** | 32.4091 | 75.2983 | 769 s |
| istanbul | cw0.75 ON | 31.7716 | 32.3552 | **75.4320** | 1507 s |
| istanbul | **cw0.50 OFF** | 32.0695 | **32.6726** | 75.2531 | 1506 s |
| istanbul | cw0.50 ON | 31.8189 | 32.3861 | 75.2623 | 1506 s |

**cat-LR is null.** Across a 4× range the sm3 span is 0.22 (AL), 0.21 (AZ), 0.28 (IST) — smaller than
the fold-σ. The argmax winner and the sm3 winner disagree at every state, which is what a null looks
like under a noisy selector.

**The 25-epoch arm is rejected.** It gains cat at AL (+0.64) and IST (+0.31) but costs reg
**−0.79** (AL) / −0.27 (AZ) / −0.06 (IST). Halving the schedule halves the wall time, which is its
only real attraction.

### 3.2 ⚠️ Class weights: an Alabama-only effect

**This corrects an earlier claim in this study.** After Alabama alone, the sweep reported
"class weights ON is the lever: +1.12 sm3". With arizona and istanbul measured, it does not hold.

Paired per-fold, class-weights **ON vs OFF** at cw0.75 (n=5):

| state | Δ cat sm3 | p | Δ reg | p |
|---|---:|---:|---:|---:|
| **alabama** | **+1.123** | **0.004** | +0.013 | 0.90 |
| arizona | +0.229 | 0.64 | +0.080 | 0.42 |
| istanbul | **−0.141** | 0.68 | +0.069 | 0.08 |

One significant state, one null, one null-and-negative. The three-state mean (+0.40) is carried
entirely by alabama. **Do not report "class weights help MTL" as a finding.** The defensible
statement is: *class weighting the MTL category head helps at alabama and is null at arizona and
istanbul.*

Region is unmoved everywhere (+0.013 / +0.080 / +0.069), so the knob is category-only regardless.

### 3.3 ⚠️ The tie-break reversed: region now prefers cw0.75

The pre-registered rule was: *when the two class-weights-ON arms are statistically indistinguishable,
break the tie on **region***. That rule was fixed on alabama-only data, where region gave
`cw0.50` a **+0.014 (p=0.90)** edge — i.e. a null.

With arizona and istanbul measured, `cw0.50/ON` vs `cw0.75/ON`:

| state | Δ cat sm3 | p | **Δ reg** | **p** |
|---|---:|---:|---:|---:|
| alabama | −0.273 | 0.62 | +0.014 | 0.90 |
| **arizona** | −0.095 | 0.56 | **−0.111** | **0.037** |
| **istanbul** | +0.047 | 0.69 | **−0.170** | **0.006** |

Category is a null at all three (p = 0.62 / 0.56 / 0.69), so the tie-break condition genuinely
applies. On region, **cw0.50 is significantly worse at two of three states** and null at the third.

**Applying the pre-registered rule to the full data selects `cw0.75 / class-weights ON`, not
`cw0.50`.** The rule was fixed before these numbers existed and is being applied unchanged; it is the
data that moved. Recorded here rather than silently re-deciding. **Awaiting the author's
confirmation before the adopted recipe changes.**

---

## 4 · Row 4 — removing class weights from the DEDICATED arm hurts

Hypothesis under test (author): class weights should be off on **all** arms, dedicated included.
C25's record showed unweighted cat CE winning macro-F1 by +5.1 pp — but that was measured on the
**MTL head, on the leaked substrate**.

Alabama, dedicated, bs8192, 5 folds (`--no-class-weights`; note `--no-cat-class-weights` is **inert**
on `--task next`):

| max_lr | weights ON (default) | weights OFF | Δ sm3 | p | med epoch ON → OFF |
|---|---:|---:|---:|---:|---|
| 0.005 | 27.4230 | 26.2195 | **−1.203** | 0.19 | 10 → 15 |
| 0.0025 | 27.6098 | 26.6145 | **−0.995** | 0.23 | 13 → 14 |

**Removing class weights costs ~1 pp on the dedicated arm**, consistently at both LRs. p ≈ 0.2 at
n=5, so this is directional rather than proven — but the sign is the same at both settings and the
magnitude is large relative to the retune gains it would erase.

**Consequence:** C25's unweighted-wins result does **not** transfer from the MTL head on the leaked
substrate to the dedicated arm on v18. The dedicated ceiling keeps its class weights, so the retuned
ceiling from §2 stands and Δcat is **not** pushed back down.

---

## 5 · How this changes Δcat at alabama

All fp32, both arms, sm3 selector:

| configuration | dedicated | MTL | **Δcat** |
|---|---:|---:|---:|
| both as-inherited | 27.122 | 26.153 | **−0.968** |
| both retuned (MTL = cw0.75/ON) | 27.610 | 27.276 | **−0.334** |

Fair tuning of both arms moves Δcat from −0.97 to −0.33 at alabama. This is **one state and one
seed**; it is not the pooled headline. The pooled Stage-0 result across six states remains
**Δcat +0.010, 95% CI [−0.424, +0.445], "matches" (TOST ±2 pp)**.


---

## 7 · Large states — rows 5, 6 (TX, 1 fold) and row 10 (MTL batch size, AL)

### 7.1 Row 5 — TX dedicated LR/schedule grid: FLAT

Grid went **up** (0.005 → 0.01) because TX shows no train−val gap. It found nothing.

| max_lr | epochs | argmax | sm3 | best epoch |
|---:|---:|---:|---:|---:|
| **0.005** | 50 | 34.107 | 33.613 | 16 |
| 0.0075 | 50 | 34.007 | 33.058 | 9 |
| 0.01 | 50 | 34.012 | 33.607 | **2** |
| 0.005 | 75 | 33.997 | **33.848** | 9 |
| 0.0075 | 75 | 34.077 | 33.602 | 9 |

Everything sits within **0.25 sm3 / 0.11 argmax** — smaller than the *gain* the small-state retune
produced (+0.49…+1.08). At one fold with no dispersion that is a null: the screen picks no direction.
Raising the LR does **not** help, so the "large states are capacity-limited, go up" reasoning is
**not** supported either. Both directions are flat.

The `lr 0.01` arm peaks at **epoch 2** — the pathological end of the early-peak signature; its argmax
is read off essentially a first-epoch spike and only the smoothing rescues it.

**Consequence:** the blind-transfer risk in [`POSTPONED.md`](POSTPONED.md) P6 resolves the safest
way — there is no reason to change the large-state LR at all. **Recipe tiers by state size**, exactly
as v17 did: small states take the retuned lower LR, large states keep `bs8192 @ 0.005`.
**Epochs pinned at 50** (author, 2026-08-08): 50 vs 75 is inside the margin of error.

### 7.2 Row 6 — TX class weights: the sign FLIPS versus alabama ⚠️

| state | data | train−val gap | folds | **Δ sm3 (OFF − ON)** |
|---|---|---:|---:|---:|
| **alabama** | 96 k windows | **+42 pp** (memorizing) | 5 | **−1.203** → weights **ON** better |
| **texas** | 3.83 M windows | **+0.25** (no overfit) | **1** | **+1.176** → weights **OFF** better |

Same knob, opposite sign, and the split falls exactly on the overfit/capacity divide measured in
§ *Why the sweep is split by state size*. Mechanistically coherent: macro-F1 rewards rare classes, so
where the model memorizes (AL) class weighting is a useful rebalancer, and where there is enough data
to fit rare classes anyway (TX) it distorts away from the frequency-weighted optimum.

**This partially vindicates the author's hypothesis** that class weights should be off — true at
large states, false at small ones. C25's "unweighted wins" evidently transfers at scale, not at small
data.

⚠️ **It is load-bearing and under-evidenced.** TX's Δcat was **+0.96**, the only "beats" verdict in
the category column. If the TX dedicated ceiling rises ~1.18, that becomes ≈ **−0.2** and the last
category "beats" disappears. But this is **1 fold** against alabama's 5. In its favour: +1.18 is ~4×
TX's dedicated fold-σ (~0.30). Against it: one fold has no dispersion, and row 6 ran at 75 epochs
while everything else uses 50.

### 7.3 Row 10 — MTL batch size at alabama: keep bs8192

All arms on cw0.50 + class-weights ON. Per-head LRs scaled ×1.67/2.5/3.33 (the `--max-lr` flag is
inert under `MTL_ONECYCLE_PER_HEAD_LR=1`).

| bs | lr× | cat sm3 | reg | geom |
|---:|---:|---:|---:|---:|
| **8192 (reference)** | 1.00 | 27.003 | **69.709** | **43.386** |
| 16384 | 1.67 | 26.691 | 69.570 | 43.091 |
| 16384 | 2.50 | 27.115 | 69.182 | 43.311 |
| 16384 | 3.33 | 26.477 | 69.483 | 42.892 |
| 32768 | 1.67 | **27.234** | 68.374 | 43.152 |
| 32768 | 2.50 | 26.649 | 68.470 | 42.716 |
| 32768 | 3.33 | 27.171 | 68.564 | 43.162 |

**bs8192 wins on region and on the joint selector; every larger-batch arm is worse on `geom`.** The
best category arm (bs32k ×1.67, +0.23 cat) costs **−1.34 reg**. Category is scattered with no LR
trend (noise); **region is the informative axis** and loses ~1.3 pp at bs32k across all three LRs.

Read this as **"too few optimizer steps hurts region"**, not "large batches hurt": bs32768 gives only
~118 steps over 50 epochs at alabama, and the region head peaks late (epochs 25–50), so it is exactly
what a shortened step budget damages. The two are confounded at alabama — as flagged *before* the run.

---

## 8 · Follow-ups queued (author plan 2026-08-08) — `run_next2.sh`

| step | what | scope | why | cost |
|---|---|---|---|---:|
| **A** | AZ + IST dedicated `--no-class-weights` | 5 folds, at the retuned winner LR **and** at 0.005 | AL (96 k) says ON, TX (3.8 M) says OFF. AZ (201 k) and IST (272 k) sit between, so they show whether the flip is **size-graded** or a texas peculiarity | ~37 min |
| **B** | TX class weights, folds 0/1/2, same seed, **50 ep** | 6 runs | one fold can be biased; fold 0 is re-run at 50 ep for both arms so the 3-fold mean does not mix schedules | ~1.6 h |
| **C** | TX dedicated bs {16384, 32768} | 1 fold, class-weight flag taken from **B's 3-fold mean** | batch size at a state that is not step-starved | ~40 min |
| **D** | FL MTL bs {16384, 32768} | 1 fold, at the best row-3b cat-lr | the AL batch sweep was confounded by step starvation; FL has ~4× the windows | ~50 min |

⚠️ **D may OOM at bs32768.** FL MTL at bs8192 uses **15.2 GB of 46 GB**; 4× the activations likely
exceeds the card. The driver catches OOM, logs it distinctly and continues — a failure there is
information, not a crash.

### 8.1 Row 11 — logit adjustment (queued after A–C)

`--logit-adjust-tau {0.5, 1.0}` at AL/AZ/IST, 5 folds, at each state's retuned winner LR (~49 min).

**Why this is not a repeat of row 4.** Row 4 removed class weights with **no replacement** and lost
~1 pp at alabama. Logit adjustment is the Bayes-consistent *substitute*: it shifts logits by
`τ·log(prior)` instead of reweighting the loss, and it targets macro-F1 directly. `next_cv.py:123-141`
makes the class-weighted CE the `else` branch, so passing the flag replaces class weighting on its
own — the untested cell.

Prior: T1.4 found **τ=0.5 beat balanced weighting at all four states** — but on the **stride-9**
substrate (AL 12,709 windows) versus v18's **stride-1** (96,326), so transfer is not guaranteed.

This is also the intervention the overfitting diagnosis ranked #2 of 3, and the only one of the three
expected to move the reported metric rather than merely stabilise training.

**A is deliberately first.** Two contradictory data points (AL ON, TX OFF) are not a finding; a
size-graded transition would be. The recipe decision for the large states should wait for it.

---

## 6 · What an auditor should check hardest

1. **§3.2 and §3.3 are corrections**, not fresh findings. An earlier version of this study reported
   the class-weight effect as a lever and pre-registered a tie-break on alabama-only data. Both are
   restated here against three states. Check I have not quietly kept the flattering version.
2. **n = 5, one seed, everywhere in this sweep.** Every p-value is a 5-fold paired test. Nothing here
   is n=20 evidence.
3. **The dedicated retune is the only multi-state, same-direction result.** If you discount one
   thing, discount the MTL knobs, not this.
4. **Large states are untested** (rows 5/6/3b queued, 1 fold). The size-inverted failure mode means
   the small-state recipe may not transfer — [`POSTPONED.md`](POSTPONED.md) P6.
5. **Selector honesty**: istanbul's argmax-vs-sm3 disagreement in §2 is the clearest case where the
   two conventions pick different winners. Both are reported everywhere for that reason.
