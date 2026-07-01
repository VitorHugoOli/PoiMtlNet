# Per-head LR is inert under OneCycle — finding + fix (SHIPPED)

**Status:** finding documented + **fix SHIPPED 2026-06-29** as opt-in `MTL_ONECYCLE_PER_HEAD_LR` (`src/training/helpers.py:347`,
per-group OneCycle `max_lr` list; default-OFF byte-identical). Used to produce the n=20 board-wide win
([`docs/studies/closing_data/perhead_lr_n20.md`](../studies/closing_data/perhead_lr_n20.md)): bs=8192 + cat-lr 1e-3
beats the champion at AL/AZ/FL. **Only the eager-byte-identical parity test for the flag-OFF path remains.**
Discovered during the `train_perf_multifold` batch-size study. Full context:
[`docs/studies/train_perf_multifold/BATCH_SIZE_SWEEP.md`](../studies/train_perf_multifold/BATCH_SIZE_SWEEP.md).

## The finding

The per-head learning-rate flags **`--cat-lr` / `--reg-lr` / `--shared-lr`** (which build a 3-group AdamW via
`setup_per_head_optimizer`) are **silently overwritten — i.e. INERT — under `--scheduler onecycle`.**

Root cause in `src/training/helpers.py:setup_scheduler`: the `onecycle` branch builds
`OneCycleLR(max_lr=<scalar --max-lr>, …)`. PyTorch's OneCycleLR broadcasts a **scalar** `max_lr` to **every**
param group, so all heads peak at the same `--max-lr` (3e-3 in the champion). The `multi_group_per_head` guard
that preserves the per-head LRs is applied only to the **`constant`** and **`cosine`** branches, never `onecycle`.

**Proof:** the run log prints `[per-head-LR] optimizer groups: [('cat',1.2e-4),('reg',1.2e-4),('shared',1.2e-4)]`
— all equal to `max_lr/div_factor` = 3e-3/25, NOT the recipe's 1e-3/3e-3/1e-3. Two FL runs differing ONLY in
`--cat-lr` (1e-3 vs 2e-3 vs 1.5e-3) came out **byte-identical** (cat 78.7648 / reg 77.4185).

## Impact

1. **The champion-G recipe's per-head LR is DECORATIVE under onecycle** — every head actually runs at the uniform
   `--max-lr 3e-3`. The §0.1 paper numbers were produced with uniform-3e-3, NOT the 1e-3/3e-3/1e-3 ratios the
   recipe string implies. (§0.1 is unaffected as a *number* — it just means those three flags were inert; the
   effective LR is the single max_lr. But the recipe documentation is misleading.)
2. **Any experiment varying `--cat-lr`/`--reg-lr`/`--shared-lr` under `--scheduler onecycle` is a NO-OP.** This
   invalidated the FL cat-lr fix; check for it before trusting any per-head-LR ablation on the onecycle path.
3. Per-head LR DOES work under `--scheduler constant` / `cosine` (the guarded branches).

## Deferred fix (do NOT ship without the guard)

To make per-head LR actually work under onecycle, pass `max_lr` to OneCycleLR as a **per-group LIST**
`[cat_max, reg_max, shared_max]` (= the group LRs from `setup_per_head_optimizer`) instead of a scalar. PyTorch
OneCycleLR accepts a list of length `len(optimizer.param_groups)` and gives each group its own peak.

**CRITICAL — must be opt-in / default-OFF.** Activating per-group max_lr changes the champion from uniform-3e-3 to
1e-3/3e-3/1e-3 → **NOT byte-identical** → breaks §0.1 reproduction. Gate behind an env flag (e.g.
`MTL_ONECYCLE_PER_HEAD_LR=1`) or an explicit CLI opt-in; default OFF preserves the frozen champion. Add a parity
test (eager byte-identical with flag OFF).

## Why this mattered for the FL large-batch regression — RESOLVED

The web-research "reg head captures the shared backbone" mechanism (agent ade63a19) was **REFUTED** by the
isolation decomposition: lowering reg-LR alone does NOT recover FL cat (reg2e3 78.73 = ctrl), and lowering
**cat-LR alone fully recovers it** (cat_only 79.84 > base). So the FL cat regression is **pure cat-LR overshoot**
(the cat head was overdriven at the uniform 3e-3 because the per-head LR was inert), NOT reg-capture. The lever is
`--cat-lr 1e-3` (with the flag ON). See `BATCH_SIZE_SWEEP.md` §"FL MECHANISM — FINAL".

## Checklist
- [x] Wire per-group `max_lr` list into the onecycle branch of `setup_scheduler`, env-gated (`MTL_ONECYCLE_PER_HEAD_LR`), default OFF. — **DONE (`helpers.py:347`)**
- [x] Verify the per-head LRs now apply (cat 1e-3 vs uniform 3e-3) — **DONE (AL on_equal==off parity; on_perhead +0.59)**
- [x] FL experiment — **DONE: cat-lr 1e-3 (not reg-lowering) is the lever; n=20 board-wide win (`closing_data/perhead_lr_n20.md`)**
- [ ] **Parity test: flag OFF → eager byte-identical** to current champion (the one remaining item).
- [ ] Promote `bs=8192 + cat-lr 1e-3` into the champion board-wide + check CA/TX.
