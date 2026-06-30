# Batch-size + Per-head-LR study — results at a glance

> ✅ **n=20 {0,1,7,100} LANDED 2026-06-30 → [`../closing_data/perhead_lr_n20.md`](../closing_data/perhead_lr_n20.md)
> is the current verdict.** It supersedes the seed-0 framing below — notably **FL is now a small WIN, not just
> "equal quality"**: at n=20 the bs2048 base is 79.68 (the seed-0 79.83 was a lucky fold), so bs8192+cat-lr-1e-3
> (79.85) beats it by +0.17 cat / +0.20 reg AND is ~7% faster. AL/AZ per-head deltas confirmed (+0.64/+1.52).

> One-screen visualization of the `train_perf_multifold` batch-size + per-head-LR study (2026-06-26…29).
> Full detail + provenance: [`BATCH_SIZE_SWEEP.md`](BATCH_SIZE_SWEEP.md). Numbers below are seed-0 5-fold unless
> n noted (n=20 confirmation in the closing_data file above). cat = macro-F1, reg = next-region Acc@10.

## 1 · Batch size 2048 → 8192 (LR unchanged)

| State | n | cat 2048 | cat 8192 | **Δcat** | reg 2048 | reg 8192 | Δreg |
|---|---:|---:|---:|:---:|---:|---:|:---:|
| **AL** | 20 | 63.55 | 63.90 | **+0.36** ✅ | 69.70 | 69.84 | +0.13 |
| **AZ** | 20 | 63.57 | 64.31 | **+0.75** ✅ | 59.40 | 59.58 | +0.19 |
| **FL** | 5 | 79.83 | 78.76 | **−1.07** ❌ | 77.40 | 77.42 | +0.02 |

Small states win on cat (pure gradient-noise effect); FL *regresses* on cat (before the per-head fix). reg ≈ flat everywhere.

## 2 · Per-head cat-LR 1e-3 — the additional, stacking lever (bs=8192)

| State | cat @ uniform 3e-3 | cat @ cat-lr 1e-3 | **Δcat** | note |
|---|---:|---:|:---:|---|
| **AL** | 63.97 | 64.56 | **+0.59** ✅ | cat was overdriven at 3e-3 |
| **FL** | 78.76 | **79.84** | **+1.08** ✅ | fully recovers — beats base 79.83 |

The recipe's intended `cat-lr 1e-3` had been **inert under OneCycle** (scalar `max_lr` broadcast 3e-3 to all heads);
activating it via `MTL_ONECYCLE_PER_HEAD_LR` is a real cat win everywhere.

## 3 · FL mechanism decomposition — which LR is the lever? (bs=8192)

| cell | cat / reg / shared LR | cat | recovers? | conclusion |
|---|:---:|---:|:---:|---|
| ctrl | 3 / 3 / 3 | 78.76 | — | the −1.07 regression |
| **cat_only** | **1** / 3 / 3 | **79.84** | ✅ **fully** | **cat-LR is the sole lever** |
| shared_only | 3 / 3 / **1** | 78.76 | ❌ | shared-LR irrelevant |
| reg2e3 | 3 / **2** / 3 | 78.73 | ❌ | reg-LR irrelevant → reg-capture DEAD |
| perhead | 1 / 3 / 1 | 79.72 | ✅ | cat-driven |
| perhead_reg2 | 1 / 2 / 1 | 79.75 | ✅ | cat-driven |

**Only cat-LR matters.** FL cat regression = **pure cat-LR overshoot** (cat head overdriven at 3e-3, exposed by the
larger batch). Refuted: reg-capture, shared-LR/drift, macro-F1 dilution, undertraining.

## 4 · Final recipes — n=20 CONFIRMED ([`../closing_data/perhead_lr_n20.md`](../closing_data/perhead_lr_n20.md))

| Regime | Recipe | cat | reg | speed | verdict (n=20) |
|---|---|---|---|:---:|---|
| **AL/AZ** small | bs=8192 + cat-lr 1e-3 | AL 64.54 (+0.99 vs champion) / AZ 65.84 (+2.27) | flat-up | neutral | **WIN** |
| **FL** large | bs=8192 + cat-lr 1e-3 | **79.85** vs base **79.68** (**+0.17**) | 77.42 vs 77.22 (**+0.20**) | **+7%** | **WIN — better on both heads + faster** |

> ✅ **At n=20 FL is a small WIN, not just "equal quality."** The seed-0 framing ("79.84 ≈ 79.83, equal quality")
> was superseded once n=20 landed: the seed-0 base (79.83) was a lucky fold; the **n=20 base is 79.68**, so
> bs8192+cat-lr-1e-3 (79.85) beats it +0.17 cat / +0.20 reg. (The older "reg ≫ base +1.74" draft was wrong — it
> compared against a stray 1-fold reg 75.58; the real n=20 reg deltas are the +0.20 above.)

## 5 · What also got ruled OUT (no lever beats plain bs=8192 at small states)

| lever | AL Δcat | AZ Δcat | verdict |
|---|---:|---:|---|
| epochs 65 / 75 | −0.15 / −0.33 | −0.27 / −0.25 | harmful (overtrain) |
| weight_decay 0.025 / 0.10 | −0.13 / −0.20 | −0.16 / −0.16 | wd 0.05 already optimal |
| pct_start 0.40 | +0.05 | −0.18 | neutral/noise |
| category_weight 0.70 / 0.80 | −0.15 / −0.16 | −0.17 / −0.10 | 0.75 already optimal |
| **logit-adjust τ=1.0** | **−5.25** | **−4.99** | **CRATERS macro-F1 — avoid** |
| adam-β2 0.95 / grad-clip 0.5 | −0.26 / −0.32 | −0.19 / −0.17 | mildly harmful (already stable) |

## 6 · Tooling shipped (opt-in, default-OFF byte-identical)

| flag / env | effect |
|---|---|
| `MTL_ONECYCLE_PER_HEAD_LR=1` | per-group OneCycle `max_lr` → `--cat-lr/--reg-lr/--shared-lr` actually apply |
| `--adam-beta2 <f>` | AdamW β2 override (large-batch stabilizer) |
