# Per-head LR is inert under OneCycle — finding + deferred fix

**Status:** DOCUMENTED, fix DEFERRED (2026-06-28). Discovered during the `train_perf_multifold` batch-size study
(FL cat-lr fix experiment). Full context: [`docs/studies/train_perf_multifold/BATCH_SIZE_SWEEP.md`](../studies/train_perf_multifold/BATCH_SIZE_SWEEP.md)
(§"FL CAT-LR FIX — INVALID").

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

## Why this matters for the FL large-batch regression (the original motivation)

The FL cat-craters-at-bs8192 mechanism (web-research agent ade63a19) is **reg head captures the shared backbone
once gradient noise drops**. Because cat is ALREADY at the uniform 3e-3 peak (not 1e-3), the original "raise
cat-lr" idea is moot. With per-group OneCycle max_lr available, the real FL lever to try is the **opposite**:
**LOWER the reg head's OneCycle peak** (e.g. reg_max 3e-3 → 2e-3) to reduce reg's shared-backbone dominance,
while holding cat at 3e-3 — testing whether easing reg-capture recovers FL cat without giving up the reg gain.
This is the recommended first experiment once the per-group max_lr is wired (opt-in).

## Checklist when picking this up
- [ ] Wire per-group `max_lr` list into the onecycle branch of `setup_scheduler`, env-gated (`MTL_ONECYCLE_PER_HEAD_LR`), default OFF.
- [ ] Parity test: flag OFF → eager byte-identical to current champion.
- [ ] Verify the `[per-head-LR]` log now shows distinct per-group peaks when ON.
- [ ] FL experiment: bs8192, reg_max ∈ {3e-3 (ctrl), 2.5e-3, 2e-3}, cat/shared 3e-3, seed0 5-fold → does lowering reg peak recover FL cat toward 79.83 while keeping reg ≥ 75.6?
- [ ] If it recovers, multi-seed confirm + check it doesn't regress the small states.
