# HANDOFF — C1 (3-snapshot per-task routing) confirm-on-G · M2 Pro

> Created 2026-06-16. Machine: **Mac M2 Pro (MPS, fp32, no AMP — slower but feasible at FL+AL)**.
> C1 was **escalated to a G0.2 gate** by the user (P1a `PHASE1_VERDICT.md` had recommended
> STORY-DEPENDENT; the user wants it settled before the freeze). This handoff runs the confirm.
>
> **Read first:** `PHASE1_VERDICT.md` §2 (the ★ C1 adjudication — the full rationale), and the original
> verdict at `docs/studies/archive/substrate-protocol-cleanup/CLOSURE.md` §C1 + its `tier_c/` results.

## What C1 is

A **deploy-mode** lever: at inference, route each task to its OWN best-epoch checkpoint (two snapshots
from a single training run) instead of selecting one joint `geom_simple` checkpoint. Prototype already
exists: `--save-task-best-snapshots` (training) + `scripts/.../route_task_best.py` (scoring) — **verify
these are present on the current code and adapt if the champion-G dual-tower changed the snapshot keys.**

## Why it needs a confirm (not already answered)

The 2/3-state positive signal (AZ +2.54 p=0.031, FL +2.80 p=0.0312; AL failed on a degenerate
Acc@1-selected snapshot) was measured **pre-C25, on the single cross-attn arch, against the depressed
old baseline, on a checkpoint-selection axis the frozen `geom_simple` selector now partly addresses.**
Part of that 2.8 pp may be exactly what the C25-fix + dual-tower + geom_simple already recovered. **It is
untested on champion G.** This run decides: real residual deploy gain on G, or already-subsumed.

## Task

- Champion **G** recipe EXACTLY (`mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` + `next_gru`,
  unweighted CE, `static_weight cw=0.75`, onecycle max-lr 3e-3, v14 substrate, geom_simple selector). Use
  the canonical invocation from `NORTH_STAR.md` / `CANONICAL_VERSIONS.md`.
- States **FL + AL** (the states with v14 on this box). Train G with `--save-task-best-snapshots`.
- Compare, per (fold, seed): **per-task-best routing** vs the **single `geom_simple` checkpoint** (the
  shipped deploy default). Start seed 0 as a discriminator; if positive, expand to {0,1,7,100}.
- A degenerate-snapshot guard is warranted (the AL failure was an Acc@1-selected reg snapshot) — select
  the reg snapshot on **Acc@10**, not Acc@1.

## Gate (escalated → G0.2)

**Promote** = ≥0.3 pp reg over the single geom_simple checkpoint, on G, multi-seed, **without hurting cat**
→ adopt as a **deploy panel** (two-checkpoint routing), recorded in `RUN_MATRIX`. It is a *deploy mode*,
NOT a single-model recipe change — it does not alter champion G's weights, so it does not become "v17"; it
becomes an alternative deploy row. **Null** → confirmed dead on G (the recovery was already captured by
C25-fix + dual-tower + geom_simple); close it, freeze proceeds.

## Guardrails

MPS: fp32, no AMP, expect slow FL folds — budget accordingly. Per-fold per-seed **train-only** priors +
the stale-log_T freshness preflight (`CLAUDE.md`) before every `--per-fold-transition-dir` run. Paired
Wilcoxon, report n and p. Pin `--canon`. Branch `study/c1-confirm-g`; do not commit to `main`.

## Hand-off

Verdict → write the C1 G0.2 row result into `closing_data/PLAN.md` + `PRE_FREEZE_PROGRAM.md` gate ledger,
`STATE`/`log.md` row, and **STOP for the user** (the freeze cannot commit with this gate open).

## NOT in scope here

**P3** (full base regeneration) is **not** this machine's job: it is post-freeze, the single heaviest
spend (all states × 4 seeds × 5 folds incl. CA/TX), gated on M0 (CA/TX/GE v14 builds — H100) and on the
freeze itself. If you want this box to contribute toward P3 while idle, the only MPS-plausible item is
pre-staging the **GE** v14 substrate (M0); CA/TX v14 are H100 work. Confirm with the user first.
