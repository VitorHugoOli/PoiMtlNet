# pre_freeze_gates — substrate/base gates that must resolve before the freeze (A40)

> **Status:** SCAFFOLDED, not launched (2026-06-14). Machine: **A40**. Position: **Level 1** of
> [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md). These are cheap tests on the **substrate /
> data base** (distinct from `mtl_frontier`, which tests the MTL recipe). Each resolves into a
> `closing_data` G0.2 row before the P2 freeze.
>
> **Read first:** [`docs/research/baseline_gap_analysis.md`](../../research/baseline_gap_analysis.md) (A2),
> [`docs/research/evaluation_protocol_review.md §4.1`](../../research/evaluation_protocol_review.md) (A4),
> [`docs/future_works/overlapping_windows.md`](../../future_works/overlapping_windows.md) (validated AL memo).

## Why this study exists

Three pre-freeze questions touch the substrate or the evaluation base — not the MTL recipe — and each
can change either what the paper *claims* or what the base *is*. They must close before `closing_data`
freezes, or the RUN_MATRIX records the wrong caveats / the wrong base.

## Scope

### A2 — feature-concat control (interpretation gate; **highest priority, cheapest, decisive**)
- **Question:** is Check2HGI's +14–29 pp next-category lift the *hierarchical-infomax learning*, or just
  *feature injection*? Check2HGI's check-in nodes carry category one-hot + hour/dow sin/cos; the HGI
  comparator does not.
- **Spec:** HGI (and/or POI2Vec) embedding ⊕ the same raw per-visit features → matched heads (`next_gru`
  cat / `next_stan_flow` reg), STL, ≥3 states, same folds. No embedding retraining → cheap.
- **Gate:** interpretation. If the concat closes most of the gap → reframe the substrate claim in the new
  paper (honest, still publishable). If not → the substrate claim stands and is *strengthened*.

### A4 — transductivity bound (disclosure gate)
- **Question:** how much does training the Check2HGI substrate on the full state corpus (incl.
  validation-fold check-ins) inflate downstream numbers? (`research/embeddings/check2hgi/` trains on all
  check-ins, best epoch by training loss — transductive.)
- **Spec:** retrain Check2HGI per fold on **train users only**, 1 state (FL) × 1 seed; compare downstream
  cat/reg vs the full-corpus substrate.
- **Gate:** disclosure. Small delta → one-paragraph defusal in the paper. Large → report it + re-anchor
  the headline numbers (and motivates inductive Check2HGI as future-work).

### Overlapping-windows — ADOPT/KEEP decision (base change)
- **Status:** the effect is **already validated** at AL single-seed (see the future-work memo):
  stride-1 lifts STL cat +9.8 / STL reg +5.1 / MTL cat +8.9 but **MTL reg barely moves** — the gap
  *widens*, which *strengthens* the dual-tower thesis (rising-tide, cat; bottleneck, reg). This study
  does **not** re-derive the effect; it resolves the **decision**.
- **Decision to make pre-freeze:** keep the non-overlapping canon (default, internal consistency) **or**
  adopt stride-1/2-3 and trigger the full-base rebuild *before* the freeze. Adoption ⇒ rebuild
  `next*.parquet` + per-fold log_T + ceilings + board at all states (the memo's rebuild checklist).
- **Why it must be pre-freeze:** windowing changes every sample count, fold composition, and paired-test
  unit; doing it post-freeze invalidates every n=20 cell. **It is a base change, NOT an MTL lever** —
  do not present it as an MTL gain (rising-tide rule, [`mtl_improvement` R1 audit](../archive/mtl_improvement/)).
- **MANDATORY leak re-audit on adoption.** The window/causal-mask correctness audit
  (`head_window_batch_audit §B`, run in `substrate-protocol-cleanup` Tier D) covered the **non-overlapping**
  windows. **Overlapping windows change the leak surface** — same-user windows now share check-ins, so a
  target can land in a neighbouring window's input, and the per-fold log_T must be rebuilt for the new
  sequence set. If overlap is ADOPTED: re-run the window/mask leak audit on the new windowing **before** the
  base is rebuilt, and rebuild the seeded per-fold priors. Adoption is gated on a clean re-audit.
- **Downstream comparability.** Whatever windowing is adopted here is the regime ALL comparands inherit —
  the champion, the STL ceilings, AND the external baselines (`baseline_gap` end-to-end baselines build
  their own sequences and must use the same stride/splits). The adopt/keep verdict is an input to
  `baseline_gap`'s final-run timing — note it in the G0.2 row.

## Protocol
Substrate v14 / blessed base; matched heads; per-fold per-seed train-only priors + freshness preflight;
multi-state for A2, single-state pilot for A4; paired Wilcoxon with n and p.

## Hand-off
Each gate closes with a verdict + a G0.2 row written into [`../closing_data/PLAN.md`](../closing_data/PLAN.md)
and (for overlapping-windows adoption) a STOP for user sign-off on the rebuild. `STATE.md` + a
`docs/studies/log.md` row on close.
