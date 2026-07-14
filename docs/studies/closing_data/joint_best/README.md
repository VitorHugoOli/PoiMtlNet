# joint_best (ex-closing_data_v2) — the deployable joint-checkpoint lane (J1)

> **What this study is.** A CPU-only, no-retraining follow-up to `closing_data` that answers one question the
> MobiWac author flagged: *what does the single served checkpoint actually deliver, vs the per-task
> diagnostic-best cells reported in Table 3?* It re-reads the **same v17 / `check2hgi_dk_ovl` runs** behind
> Table 3 at the one `geom_simple`-selected epoch (the "joint-best" / deployable convention) and audits the
> result. This is the J1 lane of [`J1_JOINT_SCORE_RUNBOOK.md`](../../../articles/%5Bmobiwac%5D/J1_JOINT_SCORE_RUNBOOK.md),
> executed and closed.

## Status: ✅ CLOSED — gap closed, benign.

**The deployable single checkpoint reproduces the reported Table 3 within ≤ 0.06 pp (category) / ≤ 0.11 pp
(region) on all six datasets — largest deviations 0.051 (category) and 0.107 (region), both at AZ. No verdict
changes.** Category beats the dedicated ceiling everywhere (+5.3 … +9.4); region beats at Istanbul/FL/TX/CA and
matches (TOST, ±2 pp) at AL/AZ — identical to Table 3. The AL-region tail risk the runbook warned about did
**not** materialize (joint-best drop is −0.11 pp; the cell sits at −0.41, far from the −2 pp bound). **CA/TX are
seed-0 ×5f provisional** (n=20 blocked on the A1 GPU top-up, which is incomplete on disk — not on J1).

## Read in this order
1. [`TASKS.md`](TASKS.md) — the charter: objective + scope guardrails + task list.
2. [`JOINT_BEST_RESULTS.md`](JOINT_BEST_RESULTS.md) — **the corrected table** (diag-best vs joint-best,
   Δ vs ceilings both ways), the three parity gates, and the decision memo.
3. [`AUDIT.md`](AUDIT.md) — the independent adversarial audit (from-scratch re-derivation + tail-risk +
   sensitivity + completeness critic).
4. [`PROVENANCE.md`](PROVENANCE.md) — the 18 rundirs (state, seed, PID, path) and how the cells aggregate.
5. `data/j1_results.json` + `score_all.py` — machine-readable results + the reproducer.

## Scope (do not drift)
v17 model (v16 + bs8192 + per-head cat-lr 1e-3), `check2hgi_dk_ovl` overlap-gated substrate, fp32, `geom_simple`
selector, `min_best_epoch=0` — **the exact Table-3 configuration**. Same rundirs, same ceilings
(`CEILINGS_N20_FINAL.md`), no new training. We report nothing on a different version, substrate, or ceiling set.

## Relationship to `closing_data`
`closing_data` (the parent study, `closing-data/v17-ceilings-n20` branch) produced Table 3 as **diag-best**.
`joint_best` (formerly `closing_data_v2`) adds the **joint-best / deployable** reading of the identical runs. The convention contract
both studies share is [`JOINT_BEST_SCORING.md`](JOINT_BEST_SCORING.md). Nothing
here supersedes a `closing_data` number — it annotates the six MTL cells with their served-checkpoint value.
