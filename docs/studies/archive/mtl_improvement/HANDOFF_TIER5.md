# HANDOFF — Tier 5 (and what's actually next) — for the next agent

> ## ⛔ SUPERSEDED (2026-06-12) — read `HANDOFF_AUDIT.md` instead; do NOT start CA/TX.
> The "only work left = Tier 6 CA/TX" line below is STALE: **T6.1 CA/TX is deferred OUT of this study**
> (user decision 2026-06-12) to the upcoming **`closing-data` study** (heavy runs happen once, against
> the final frozen recipe, after all improvement studies close). The A40's entire remaining scope is the
> closure punch list in **`HANDOFF_AUDIT.md`** (P0 FL cat-transfer manifest integrity + cheap hardening).

**Date:** 2026-06-08. **Branch:** `mtl-improve` (pushed). **Champion:** **G** (unchanged) —
`mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` (aux, prior-OFF), v14 substrate, unweighted
onecycle, KD-OFF. Read order for a fresh agent: this file → `HANDOFF.md` (top banners) → `CHAMPION.md`
→ `WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md` → `log.md` (newest entries) → `INDEX.html`.

> ## ⚠ READ FIRST: Tier 5 is **CLOSED**. Do NOT re-run it.
> The original "Tier 5 = head re-ablation" is fully done. There is **no open Tier-5 experiment**. The
> live remaining work is **Tier 6 (completeness + paper)**. This file tells you (1) why Tier 5 is closed,
> (2) the exact state of the champion, (3) the only work left, and (4) the traps.

## 1. Tier 5 status — CLOSED (all three cards resolved)
| card | status | result |
|---|---|---|
| **T5.1** reg-head sweep | DONE | The reg **STAN private tower is load-bearing** — GRU/LSTM/TCN private towers all −1.8…−3.4 pp (B-A1). STAN stays. |
| **T5.2** cat-head sweep | DONE (re-run UNDER G 2026-06-09) | **`next_gru` is the multi-state cat champion** — confirmed by a clean cat-ENCODER swap under G (10 heads × AL+FL): **no head wins at both states**; every FL-beater (conv_attn +1.06, temporal_cnn +0.59, lstm +0.34) craters at AL (−1.3…−23.6). Plain CE is the loss optimum (B-A4). `T52_cathead_sweep.md`. **Bonus:** next_conv_attn = a +1.06 FL-only cat lever (scale-conditional future-work, not adopted). |
| **T5.3** HSM high-card reg head | **FALSIFIED (2026-06-08)** | Hierarchical softmax = flat softmax at FL 4.7k (73.21 vs 73.22, within σ). HSM is a speed/memory technique, not an accuracy lever. No dual-tower-HSM build motivated. `cat_transfer_and_T53.md §b`. |
| (follow-up) cat-transfer ablation | DONE (2026-06-08) | The +3 pp MTL-cat gain is **architecture-dominated** (cross-attn trunk +2.27 FL/+3.11 AL); genuine region→cat transfer only +0.89 FL/−0.71 AL. `cat_transfer_and_T53.md §a`. |

**→ Tiers 2V, 3, 4, 5 are ALL closed. The champion G is robust, multi-state, multi-seed, and
mechanistically explained.** No architecture/head/loss/optimizer/substrate/data lever beats G.

## 2. The unifying result (so you don't re-open settled questions)
The two tasks have **orthogonal gradients** on the shared trunk (cos(∇cat,∇reg) ≈ 0; FL +0.0007 / AL
+0.0026). This single fact explains the whole study and is now the paper's mechanistic spine:
- **No modern MTL optimizer helps** (Tier 4: full `src/losses` registry + loss-scale-norm all fail to
  Pareto-beat tuned `static_weight cw=0.75` — there is no gradient conflict to resolve; Kurin/Xin
  NeurIPS'22 predict exactly this at k=2 with a tuned baseline). **Do NOT re-sweep balancers.**
- **More parameter-sharing hurts** (Tier 2: MoE/cross-stitch/hard-share lose reg). **Do NOT revisit.**
- **The dual-tower wins** because it's matched to the task geometry (protect reg privately; let cat
  harvest the shared cross-attn encoder).
- **Reg-input levers transfer but don't beat the moving ceiling** (Tier 3: overlap R1, HGI routing R2 =
  rising-tide nulls). **Do NOT re-run.**
Full narrative + 3 figures: `WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md`. Claim: `CLAIMS CH31`.

## 3. The ONLY work left — Tier 6 (completeness + paper), neither changes the champion
1. **C-A / T6.1 — CA/TX completeness (heavy, the lone scale gap a reviewer will challenge).** Build the
   v14 substrate (`check2hgi_design_k_resln_mae_l0_1`) at **CA (8.5k regions)** and **TX (6.5k)** +
   seeded per-fold log_T, then run champion G + the (c)/(d) ceilings, {0,1,7,100} (1-fold acceptable if
   5-fold impractical — flag directional). Prediction (recorded): the C25 effect scales with class count
   → CA/TX should show the LARGEST G margins. The v14 build is the serial spine (~heaviest single step).
   Memo: `docs/future_works/mtl_improvement_catx_scale_conditional.md`. Driver pattern: `scripts/_v14_run/`
   build scripts + `scripts/mtl_improvement/c25_g_multistate.sh` (G) + `t2v1_ceilings_multiseed.sh` (ceilings).
2. **C-B / T6.2 — BRACIS paper-doc restatement (author decision, mostly no-GPU).** Fold the corrected
   narratives into the paper docs (`articles/[BRACIS]_Beyond_Cross_Task/` + `RESULTS_TABLE.md §0.1`):
   - the Pareto-positive headline ("a single MTL model **matches** the reg ceiling + **beats** cat +3pp", multi-state, 4-seed);
   - the **cat-gain decomposition** (architecture-dominated, not region transfer — CH30 refinement);
   - the **orthogonality + modern-optimizer-negative** as a limitations/analysis section (CH31) with the 3 figures;
   - C25 unweighting + the §0.1 continuity annotation (already staged in `PAPER_UPDATE.md`).
   Staging doc: `PAPER_UPDATE.md`. Whitelist: `CLAIMS_AND_HYPOTHESES.md`.

   (If T6.1 CA/TX changes nothing, the 4-state result is already paper-grade — CA/TX is a robustness
   appendix, not a blocker.)

## 4. Traps / discipline (carried forward — these bit us)
- **Matched-metric only (R0 bar).** G's reported reg is `top10_acc_indist`; the p1 ceiling is FULL
  `top10_acc`. Always compare on the matched metric: `full = indist·(1−ood_fraction)` (R0 method,
  validated vs B-A2; script `scripts/mtl_improvement/r0_matched_rescore.py`). The matched G−ceiling reg
  gap is tiny (−0.09…−0.31) — a lever lifting STL reg lifts the ceiling too (magnitude rule).
- **A single-state Pareto gain is NOT a champion** until confirmed at ≥1 small + GE + a large state
  (CONCERNS §C26 — the G′ over-promotion lesson). Applies to CA/TX too.
- **Gradient-surgery balancers are mis-wired under the dual-tower** (CONCERNS §C27) — if you ever sweep
  them again (you shouldn't — cos≈0), fix the task-specific-gradient handling first.
- **`--canon` contract:** every train.py invocation must pin `--canon` (default v16=G; partial flags
  merge with the bundle). For non-G arms use `--canon none` + explicit flags (the v16 bundle injects
  `--category-weight 0.75` which conflicts with non-static losses).
- **Seeded per-fold log_T mandatory + fresh** (`region_transition_log_seed{S}_fold{N}.pt`); G is prior-OFF
  so it's loaded-but-inert, but the existence/freshness/n_splits guards still hard-fail. For the overlap
  engine the v14 log_T is reused inert (point `--per-fold-transition-dir` at the v14 dir).
- **`--loss-scale-norm`** is experimental, default-OFF, FALSIFIED — do not enable.
- **setsid your background sweeps.** Plain `( … ) &` inside a tool call gets killed when the call returns
  (this cost us a re-launch of the FL T4 chain). Use `setsid` + a `.done` sentinel + a watcher.
- **Rundir capture:** PID-suffix (`results/.../mtlnet_*_${pid}`), never `ls -dt | head -1` under concurrency.

## 5. Where everything is
- Champion + reproduction: `CHAMPION.md`, `CANONICAL_VERSIONS.md §v16`.
- This session's results: `results/mtl_improvement/{R0_matched_metric_bar.json, R1_overlap_under_g.md,
  R2_dual_substrate_routing.md, T40_rlw_litmus.md, T4_full_screen.json, T4_audit_and_verdict.md,
  cat_transfer_and_T53.md}`.
- Figures: `studies/archive/mtl_improvement/figs/{grad_cosine_tasks, t4_balancer_scatter_FL,
  t4_loss_weight_trajectories_FL}.png`.
- Drivers (all `scripts/mtl_improvement/`): `r0_matched_rescore.py`, `r1_overlap_under_g.sh`,
  `r1b_shared_overlap_deconfound.sh`, `r2_dual_substrate_routing.sh`, `t40_rlw_litmus.sh`,
  `t4_full_screen.sh`, `t4_corrected_rerun.sh`, `t40a_scalenorm_wgrid.sh`, `cat_transfer_ablation.sh`,
  `t53_hsm_stl_test.sh`, `plot_grad_cosine.py`, `plot_t4_balancers.py`, `t4_agg.py`.
- Claims/concerns/timeline: `CLAIMS_AND_HYPOTHESES.md CH30/CH31`, `CONCERNS.md C25/C26/C27`, `CHANGELOG.md 2026-06-08`.
