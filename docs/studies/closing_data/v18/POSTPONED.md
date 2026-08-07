# v18 — postponed executions

> Work that is **justified and scoped but deliberately not run**, with the reason and the exact
> command so it can be picked up without re-deriving anything. Nothing here is abandoned; it is
> queued behind higher-priority work or behind a decision.
>
> Last updated 2026-08-07.

## P1 — Capacity-matched dedicated region control ⏸ HELD

**What.** Run the v18 dedicated-region protocol with the head widened to the joint model's region
pathway parameter count, and compare against the joint model's region score.

```bash
# california (the load-bearing state: joint reg 65.57 vs dedicated ceiling 63.45, Δ +2.12)
env MTL_CHUNK_VAL_METRIC=1 MTL_DISABLE_AMP=1 PYTHONPATH=src \
  python -u scripts/p1_region_head_ablation.py --state california --heads next_stan_flow \
  --input-type region --region-emb-source check2hgi_design_k_resln_mae_l0_1 \
  --override-hparams freeze_alpha=True alpha_init=0.0 d_model=352 \
  --engine-override check2hgi_v18 --folds 5 --epochs 50 --seed 0 --target region \
  --max-lr 0.003 --compile --tf32 --tag v18_california_reg_capmatched_s0
# alabama uses d_model=480 (widths precomputed in storyline/audit/capacity_baseline_experiment.md)
```

**Cost (measured).** Dedicated-reg walls at v18 seed 0: alabama 189 s, california 5177 s. With the
wider head ≈ **5 min (AL) / ~2.2 h (CA)** per seed.

**Why it matters.** It is the only experiment that can decide whether the surviving MTL claim — the
region advantage at CA/TX (+2.12 / +2.05) — is **multi-task sharing** or simply **parameters**. The
joint model's region pathway carries 2.5–5.9× the dedicated baseline's parameters (alabama 2,466,542
vs 417,117; california 3,420,110 vs 1,370,685) *and* a different head (`next_stan_flow_dualtower`
with `fusion_mode=aux` vs plain `next_stan_flow`). The trunk triage already showed the advantage
survives severing the trunk **and** deleting the category task
([`region_1fold_triage/FINDING.md`](region_1fold_triage/FINDING.md)), which points at capacity — but
points is not proves.

**Why HELD (user decision 2026-08-07).** The result would be a claim about **champion-G**, not about
the v18 evaluation currently in flight. It changes how the region result is *interpreted*, not what
the board must report, so it does not block the sweep or the waves. Revisit when the v18 numbers are
settled and the dissertation's region paragraph is being written.

**What it decides.** If capacity-matched dedicated reg lands within ~±0.3 pp of joint reg at CA, the
region advantage is parameters and the honest framing is *consolidation, not synergy*. If a ≥1 pp gap
persists, the joint construction genuinely adds something at large C — the one evidence-backed reason
to keep studying the joint build.

## P2 — Re-test reverse co-location KD (reg→cat) on the leak-free substrate ⏸ HELD

**What.** v18 joint recipe + `--cat-kd-weight 0.2`, at alabama and arizona, 2 seeds ≈ 1.3 h.

**Why.** Its null (AL +0.100 ± 0.282, p=0.31) was measured on a category baseline **inflated ~2× by
the leak** — a saturated target. On v18 the category head sits at ~27 with severe memorization
(train macro-F1 66.4 vs val 24.0), which is exactly the failure mode a P(cat|region) prior
regularizes. Already wired at `src/training/runners/mtl_cv.py:649-774`.

**Why HELD.** Prior expectation is still null (v18 Δcat ≈ 0 pooled with a tight CI). It is the last
untested leak-free cell in the category direction; worth closing for completeness, not for hope.

## P3 — Re-test conditional coupling (cat→reg input conditioning) on v18 ⏸ HELD

**What.** `--reg-head-param cond_coupling=posterior` on the v18 joint recipe at florida, 2 seeds
≈ 4 h; texas only if florida moves.

**Why.** The only sharing channel that ever produced a real measured transfer (FL cat +0.235 ± 0.136
4/4 seeds; reg +0.070, p=0.035), but sub-gate and measured on v14's inflated category.

**Why HELD.** Information-theoretically capped: the category posterior carries ≤ log₂7 ≈ 2.8 bits
against a 10–13-bit region target (C = 1109…8501), so the ceiling on any cat→reg transfer is coarse
re-ranking. Expected small even if real.

## P4 — Five-fold trunk ablation at california / texas ⏸ DE-PRIORITISED

**What.** A/A′ arms at CA+TX, 5 folds, seed 0 ≈ 22 h.

**Why DE-PRIORITISED.** The 1-fold triage already answered the large-effect question: severing the
trunk, and even deleting the category task entirely (`rg2`), moves region < 0.15 pp at CA/TX. A
five-fold pass would mostly re-measure a null at 22 h. The AL arms (T1/T2, run 2026-08-07) confirm
the same at small data: Δcat −0.015 / −0.154, Δreg −0.138 / −0.004.

## P5 — T2 (A′) at florida ⏸ CONDITIONAL

Run only if **T1 at florida** shows a non-null effect. T1 at FL costs ~41 min (the joint cell is
7301 s and the no-sharing arm runs ~3× faster — measured 1094 s → 359 s at alabama).

---

## Practical note worth keeping

The trunk costs **~3× the wall-clock for ~0 benefit**: at alabama the joint cell is 1094 s with the
trunk and 359 s without it, for a category difference of −0.015 pp. If the final architecture keeps
the trunk it should be for a stated reason, because it is not paying for its compute.
