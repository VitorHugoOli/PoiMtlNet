# C1 (3-snapshot per-task routing) — confirm-on-G G0.2 VERDICT

> **Verdict: PROMOTE → SUPPORTIVE diagnostic panel ONLY (not the primary deploy, not a headline).**
> Gate cleared (≥0.3 pp reg, multi-seed, cat not hurt, on champion G), but per **user decision
> 2026-06-17** it is adopted as a *supportive/diagnostic* `RUN_MATRIX` panel, **explicitly NOT the
> primary deploy method** — see §Methodological scope. It is a *deploy-side* result, NOT a
> single-model recipe change; it does not alter champion G's weights → it does **not** become "v17".
> Run on **Mac M2 Pro (MPS, fp32, no AMP)**, branch `study/c1-confirm-g`, 2026-06-16/17.
> Spec: [`HANDOFF_C1_M2.md`](HANDOFF_C1_M2.md); adjudication that escalated it: [`PHASE1_VERDICT.md §2`](PHASE1_VERDICT.md).

## What was tested

At inference, route **each task to its own best-epoch snapshot** (two snapshots saved from a single
champion-G training run via `--save-task-best-snapshots`) instead of selecting one joint
`geom_simple` checkpoint (the shipped deploy default). Scored on the held-out fold by an independent
code path (`scripts/route_task_best.py`) — a different path from `mtl_cv`'s training loop, so it
forecloses train-harness inflation.

- **Recipe:** champion **G** = `--canon v16` EXACTLY (`mtlnet_crossattn_dualtower` +
  `next_stan_flow_dualtower` aux/prior-OFF + `next_gru` cat, `static_weight cw=0.75`, onecycle
  max-lr 3e-3, unweighted CE, log_T-KD OFF, **v14 substrate** `check2hgi_design_k_resln_mae_l0_1`,
  default `geom_simple` selector).
- **Substrate:** v14 built locally on MPS (`scripts/probe/build_design_k_delaunay.py`, 500 ep) at
  **FL + AL**; per-fold seed-tagged log_T copied from canonical c2hgi (identical region structure,
  n_regions FL/AL match) + freshness-asserted.
- **Protocol:** states **FL + AL**, seeds **{0, 1, 7, 100}**, 5 folds → **n=20 per state**, paired
  Wilcoxon on per-(fold,seed) deltas. Reg gate metric = **Acc@10** (`top10_acc`).
- **Degenerate-snapshot guard (the AL pre-C25 failure mode):** the reg snapshot is selected on
  **Acc@10** (`top10_acc_indist`), not Acc@1 — this is already the v15 default
  (`check2hgi_next_region` task_b `primary_metric=TOP10`), confirmed empirically (reg-best epochs ≠
  joint-best epochs). No code change was needed; the guard the original study lacked is now built in.

## Result — per-task-best routing vs the single `geom_simple` checkpoint

| State | n | mean Δreg (Acc@10, pp) | Wilcoxon p (reg) | mean Δcat (F1, pp) | Wilcoxon p (cat) | gate |
|---|---|---|---|---|---|---|
| **alabama** | 20 | **+1.554** | **0.0001** | +0.109 | 0.929 (n.s.) | **PROMOTE** |
| **florida** | 20 | **+0.625** | **0.0000** | +0.187 | 0.0022 (sig. positive) | **PROMOTE** |
| **POOLED** | 40 | **+1.089** | — | +0.148 | — | **PROMOTE** |

Gate = ≥0.3 pp reg over the single `geom_simple` checkpoint, on G, multi-seed, without hurting cat.
**Both states clear it decisively; cat is never hurt (significantly *positive* at FL).**

Raw per-fold deltas: `results/check2hgi_design_k_resln_mae_l0_1/{alabama,florida}/c1_route_s*/route_fold*.json`.
Reproduce the table: `python scripts/closing_data/c1_aggregate.py "results/check2hgi_design_k_resln_mae_l0_1/*/c1_route_s*/route_fold*.json"`.

## Interpretation (settles the P1a open question)

1. **NOT subsumed by C25-fix + dual-tower + geom_simple.** A real residual deploy gain survives on
   champion G: AL +1.55 pp, FL +0.63 pp — both above the gate, multi-seed.
2. **The P1a "partly subsumed" hypothesis was also correct.** FL's gain on G (+0.63 pp) is far
   smaller than the original **pre-C25 +2.80 pp** — so most of the old 2/3-state signal *was* the
   depressed baseline + the broken selector, recovered by C25-fix + dual-tower + geom_simple. What
   remains is the genuine residual that routing adds on top of the shipped deploy default.
3. **AL — the state that FAILED in the original study (a degenerate Acc@1-selected reg snapshot) —
   now passes cleanly 5/5 every seed.** The v15 Acc@10 reg-monitor fixed that failure mode; the
   warranted "Acc@10-aligned reg-best selector + degenerate-snapshot guard" is in place.
4. **It is a deploy mode, not a recipe.** Two snapshots from ONE training run; G's weights are
   unchanged. → reported in `RUN_MATRIX` as a **supportive panel** alongside the single-checkpoint
   `geom_simple` deploy default. Does **not** become v17. **NOT the primary deploy** — see below.

## Methodological scope — why C1 is SUPPORTIVE, not the proposal (user, 2026-06-17)

The MTL thesis is "**a single model serves N tasks**": one set of weights, one forward pass, both
heads. C1 preserves the single *training run* and single *architecture* (the joint optimization,
shared backbone, and gradient interaction all genuinely happened), but at **inference** it loads two
different weight-snapshots and runs the forward twice — the reg head from the reg-best epoch, the cat
head from the cat/joint epoch. That **forfeits the single-model property** the thesis rests on (≈2×
params resident, two forward passes — effectively task-specialised models at deploy). Promoting it to
primary would quietly concede the central argument. Therefore:

- **PRIMARY / headline deploy = the single `geom_simple` checkpoint** (one model, one forward, N
  tasks) — the thesis-bearing result. Unchanged by C1.
- **C1 = a supportive *diagnostic* panel** that quantifies the **deploy-time per-task selection
  headroom** left on the table by the single-checkpoint constraint.

**Precise ordering (so the panel is not over-claimed as "the task ceiling"):**
`single joint checkpoint (geom_simple)  ≤  per-task-best snapshot (C1)  ≤  STL architectural ceiling`.
C1's reg-best is "the best reg *this jointly-trained model* reached at any epoch" — bounded above by
the separately-trained STL ceiling. So C1 measures a **deploy-time selection** gap *within* the joint
model (AL +1.55 pp, FL +0.63 pp over the single checkpoint), **not** the architectural task ceiling.
Cite it as "headroom recoverable by per-task checkpoint selection if one relaxes the single-model
constraint," never as the model's ceiling.

## Tooling added (branch `study/c1-confirm-g`)

- `scripts/closing_data/c1_prep_substrate.sh` — local postbuild (next/next_region) + seed-tagged
  log_T copy + freshness assert (adapts the remote `postbuild_design_substrate.sh`, adds seeds≠42).
- `scripts/closing_data/c1_run_g.sh` — `--canon v16` + `--save-task-best-snapshots --no-checkpoints`
  + stale-log_T preflight + seed-isolated snapshots + per-fold routing. **NOTE:** route step must
  NOT pass `--task-set` — it forces default-preset heads and the dual-tower state_dict fails to load;
  omitting it reconstructs the champion-G heads from `config.json`.
- `scripts/closing_data/c1_aggregate.py` — paired Wilcoxon + the gate verdict.

## Resolution (user, 2026-06-17)

Gate **adjudicated PROMOTE and CLOSED** with the scope above: C1 is adopted as a **supportive
diagnostic panel** in `RUN_MATRIX` (deploy-time per-task selection headroom), **NOT the primary
deploy method** — the single `geom_simple` checkpoint remains the thesis-bearing headline. Champion-G
single-model recipe unaffected; no v17. With C1 closed, **G0.1 (aligned-pairing) is the lone open
recipe-changing P0 gate** before the freeze.
