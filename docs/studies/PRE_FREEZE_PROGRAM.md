# Pre-Freeze Program — the study family that launches the bases for `closing_data/`

> Created 2026-06-14. Orchestrator for the studies that must run **before** `closing_data` freezes
> the recipe/substrate and regenerates the full experimental base. Derived from
> [`docs/research/experiment_roadmap.md`](../research/experiment_roadmap.md),
> [`docs/research/mtl_frontier.md`](../research/mtl_frontier.md), and
> [`docs/research/future_work.md`](../research/future_work.md).
>
> **One rule governs the whole family:** anything that can change the recipe, the substrate, or the
> evaluation base MUST resolve **before** the `closing_data` P2 FREEZE — because the freeze is a hard
> barrier and the P3 regeneration (every state × 4 seeds × 5 folds) cannot be cheaply re-run. Promoted
> levers register into `closing_data` P0/G0.2 as gates. See [`closing_data/PLAN.md`](closing_data/PLAN.md).

---

## The dependency DAG (what blocks what)

```
LEVEL 0 — EXPLORATION & INVENTORY (must finish first; outputs influence the freeze)
  ┌─ [A40]      mtl_frontier/        R1 log_C co-location prior · R2 STEM-AFTB gating · R3 cross-task distil
  │                                  → promote-gate ≥0.3 pp either head → v17 candidate (USER sign-off)
  ├─ [reading]  closing_data P1a     cross-study re-eval → PHASE1_VERDICT.md (regime-dependence re-read)
  ├─ [A40+Mac]  baseline_gap/        triage external baselines → which become RUN_MATRIX rows/columns
  │                                  (B1 CTLE · B2 POI2Vec/skip-gram · B3 HMT-GRN-MTL · B4 cascade · B5 Flashback)
  │                                  — DECISION pre-freeze (feeds P1b); impl ∥; final runs fold into P3
  └─ [Mac, ∥]   second_dataset/      Massive-STEPS NYC ETL (cat→7-root map, coords→tracts, splits)
                                     — NO freeze dependency; prep runs concurrently with Level 0/1

LEVEL 1 — PRE-FREEZE GATES (cheap tests that can still change recipe/base; after/with Level 0)
  ├─ [A40]      pre_freeze_gates/    A2 feature-concat control · A4 transductivity bound
  ├─ [A40]      pre_freeze_gates/    overlapping-windows ADOPT/KEEP decision (base change — see memo)
  └─ [from CD]  closing_data G0.1    aligned-pairing training test (already specced in PLAN P0)
                                     ⇒ each promoted item is written into closing_data G0.2 + freeze notes

LEVEL 2 — HARD BARRIER ───────────────────────────────────────────────────────────────────────
              closing_data P2        RECIPE + SUBSTRATE FREEZE + RUN_MATRIX.md   ★ USER SIGN-OFF ★

LEVEL 3 — SINGLE HEAVY SPEND
              closing_data P3        full base regeneration (A40 unmetered + H100 6 h for CA/TX builds)

LEVEL 4 — EXTERNAL VALIDATION (may overlap P3 tail)
              second_dataset/        validation phase: champion G + STL ceilings + Markov floor on
                                     Massive-STEPS NYC/Istanbul (within-user user-grouped CV = Gowalla-parity).
                                     NOTE: this validation is NOT the temporal-split bridge — the shipped
                                     Massive-STEPS split is user-stratified RANDOM, not temporal (F1). The
                                     bridge (roadmap A5) is the separate Phase E2 chronological per-user
                                     re-split (Mac-track, parallel, no freeze dependency).
```

**Why exploration is strictly first** (the user's hierarchy): a lever promoted by `mtl_frontier` or a
gate tripped by `pre_freeze_gates`/overlapping-windows changes what the freeze pins; if the base were
regenerated first, that spend would be thrown away. Inventory (`P1a`) likewise can surface a parked
lever that becomes a gate. So Level 0 → Level 1 → freeze, no shortcuts.

**Why `second_dataset` runs in parallel and off the critical path:** its ETL touches nothing the freeze
pins (a different corpus), so the Mac builds it during Levels 0–1; only its *validation runs* depend on
the frozen champion, so they sit at Level 4. Note the validation phase is **not** the temporal-split
bridge — the shipped Massive-STEPS split is user-stratified RANDOM, not temporal (F1). The bridge
(roadmap A5) is the **Phase E2 chronological per-user re-split** built from the corpus's per-check-in
timestamps: it too is Mac-track with **no freeze dependency**, so it runs in parallel alongside the ETL
during Levels 0–1 (only its champion *runs* would join the Level-4 validation).

**Why `baseline_gap` is Level 0/1 for the *decision* but trails for the *runs*:** which external
baselines enter the final tables is a RUN_MATRIX input (`closing_data` P1b) and must be pinned at the
freeze; the baselines themselves are comparison rows that don't touch the champion recipe or substrate,
so they implement in parallel and their final runs fold into the P3 regeneration (M1 board).

**★ Comparability regime (the rule the user flagged):** a baseline is only comparable if it runs on the
**exact frozen base** the champion uses. Any base-level change pins what baselines must match —
**substrate identity** (v14 / a promoted v17), the **windowing decision** (overlapping vs non-overlapping +
stride), the **fold/seed/prior protocol**, and the **label spaces**. Substrate-column baselines (CTLE,
POI2Vec) inherit this automatically through the matched-head pipeline; **end-to-end baselines (HMT-GRN-style,
cascade, Flashback/DeepMove) build their own sequences and MUST mirror the adopted windowing/stride + splits.**
Therefore `baseline_gap` may implement/smoke-test on the current base during Level 0/1, but its **paper-grade
runs are blocked on the `pre_freeze_gates` overlapping-window ADOPT/KEEP decision + the P2 freeze**, and the
end-to-end baselines re-run if overlap is adopted. This is why the DAG draws `pre_freeze_gates(overlap) →
baseline_gap(final runs)`. The window leak-audit must also be re-run on adoption (overlapping windows change
the leak surface). External baselines on the *second dataset* are out of scope unless the user pulls one in.

---

## Machine allocation (inherits `closing_data` PLAN §Machine allocation)

| Machine | Metering | Pre-freeze role | Notes |
|---|---|---|---|
| **A40** | unmetered workhorse | `mtl_frontier` (R1–R3) **and** `pre_freeze_gates` (A2/A4/overlap) | All training-bearing exploration. Serialize R1→R2→R3 or interleave; gates are cheap and slot between waves. |
| **Mac M2 Pro** (user's local box) | local, MPS only | `second_dataset` ETL + scoring only | **No heavy CUDA training here** — ETL (parse, category map, tract spatial-join, split build), substrate prep, and scoring passes. Champion training on the new corpus waits for a CUDA box at Level 4. (Note: `docs/infra/` references an M4 Pro 32GB lane — same MPS caveats apply: no AMP, fp32, slower.) |
| **H100** | **6 h metered** | reserved for `closing_data` P3 (CA/TX v14 builds) | Do **not** spend on pre-freeze exploration — the A40 absorbs it overnight. |

---

## Gate ledger (feeds `closing_data` G0.2 — the freeze cannot commit with an open row)

| Gate | Study | Promote condition | On promote | On null |
|---|---|---|---|---|
| G0.1 aligned-pairing | closing_data P0 | ≥0.3 pp either head, multi-seed | recipe → v17 | v16 freezes; "wins without per-sample mixing" earned |
| R1 log_C co-location prior | mtl_frontier | ≥0.3 pp reg over log_T-KD-alone, multi-seed | recipe → v17 | drop / future-work |
| R2 STEM-AFTB gating | mtl_frontier | ≥0.3 pp either head over G, multi-seed | recipe → v17 | G's current sharing is the optimum (citable) |
| R3 cross-task distillation | mtl_frontier | ≥0.3 pp over R1/log_T teacher | recipe → v17 | static teacher suffices |
| R10 GRM/SSC gated read (arXiv 2602.24281, ★ user) | mtl_frontier | ≥0.3 pp either head over G, multi-seed (needs working impl) | recipe → v17 | GRM ≡ hand-built asymmetry (citable null) |
| A2 feature-concat control | pre_freeze_gates | n/a (interpretation gate) | reframes the substrate claim in the new paper | substrate claim stands as-is |
| A4 transductivity bound | pre_freeze_gates | n/a (disclosure gate) | report delta + re-anchor headline numbers | one-paragraph defusal in the paper |
| Overlapping-windows | pre_freeze_gates | adopt only if user accepts full-base rebuild + a clean leak re-audit | stride change → leak re-audit, rebuild ALL bases + priors pre-freeze, and the regime baselines must match | keep non-overlap canon (default) |
| B1–B5 baseline triage | baseline_gap | n/a (inventory decision) | chosen baselines → RUN_MATRIX rows/columns; **final runs on the frozen base** (blocked on overlap decision + freeze; end-to-end baselines mirror the adopted windowing) | baseline excluded from the matrix with a recorded reason |

> `baseline_gap`'s gate is an **inventory decision** (which external baselines the final tables carry),
> not a recipe-promotion gate: it does not change the frozen numbers, but it must resolve by P1b so the
> RUN_MATRIX and the freeze pin the right comparison set.

> A2/A4 are **interpretation/disclosure** gates, not recipe-promotion gates: they don't change the
> frozen numbers, they change what the paper can *claim* about them — but they still must resolve
> pre-freeze so the RUN_MATRIX records the right caveats.

---

## Disposition of existing `docs/future_works/` memos (pre-freeze relevance)

`closing_data` P1a formally adjudicates these; first-pass dispositions:

| Memo | Disposition |
|---|---|
| `overlapping_windows.md` (validated AL) | **pre_freeze_gates** — adopt/keep decision (base change) |
| `composite_two_substrate_engine.md` | STORY-DEPENDENT (deploy-pattern; RUN_MATRIX panel) |
| `reg_head_architecture_sweep.md`, `mtl_architecture_revisit.md` | fold into `mtl_frontier` R2/R9 scope if cheap, else future-work |
| `substrate_adaptive_mtl_balancing.md`, `part2_mtl_dual_substrate_routing.md` | regime-null under C25 — re-READ not re-run (P1a) |
| `poi_decoder_hgi_distill.md` | future-work (inductive/alignment, [`future_work.md §2–3`](../research/future_work.md)) |
| `joint_selection_and_loss_combination.md` | covered by C21 geom_simple + R4 Pareto-profiling (future) |
| `mtl_improvement_catx_scale_conditional.md`, `paper_canon_reevaluation.md`, `task_pivot_memo.md`, `head_window_batch_audit.md` | inputs to `closing_data` P1a/P1b, not new studies |

---

## Not yet homed (decisions for the user — flagged, not scoped here)

- **Inductive Check2HGI** ([`future_work.md §2`](../research/future_work.md)): structural fix for the A4 transductivity finding — post-paper future-work unless A4 comes back large.
- **Next-POI extension via region/category composition** ([`future_work.md §1`](../research/future_work.md)): the most defensible next-paper direction, but out of scope for the *closing* paper — parked.

> **Now homed (2026-06-14):** the external-baseline gap (CTLE/A3, POI2Vec columns, HMT-GRN-style MTL
> baseline, cascade, Flashback/DeepMove) — previously listed here — is owned by the new
> [`baseline_gap/`](baseline_gap/) study.
