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
  └─ [Mac, ✅]  second_dataset/      Phase E ETL ✅ COMPLETE (2026-06-15): Massive-STEPS NYC + Istanbul
                                     (FSQ→7-root map; regions = NYC TIGER tracts, IST mahalle PRIMARY +
                                     H3 secondary; both sequence protocols + leak-free per-fold priors).
                                     Mac dry-run validated pipeline + champion transfer (DIRECTIONAL only).
                                     — NO freeze dependency; Phase V (real numbers) is Level 4 below.

LEVEL 1 — PRE-FREEZE GATES (cheap tests that can still change recipe/base; after/with Level 0)
  ├─ [A40]      pre_freeze_gates/    A2 feature-concat control · A4 transductivity bound
  ├─ [A40]      pre_freeze_gates/    overlapping-windows ADOPT/KEEP decision (base change — see memo)
  └─ [from CD]  closing_data G0.1    aligned-pairing training test (already specced in PLAN P0)
                                     ⇒ each promoted item is written into closing_data G0.2 + freeze notes

LEVEL 2 — HARD BARRIER ───────────────────────────────────────────────────────────────────────
              closing_data P2        RECIPE + SUBSTRATE FREEZE + RUN_MATRIX.md   ★ USER SIGN-OFF ★

LEVEL 3 — SINGLE HEAVY SPEND
              closing_data P3        full base regeneration (A40 unmetered + H100 6 h for CA/TX builds)

LEVEL 4 — EXTERNAL VALIDATION (may overlap P3 tail)  ⛔ BLOCKED on P2 freeze + CUDA box
              second_dataset/        Phase V: champion G + per-task STL ceilings + Markov-1 floor on
                                     Massive-STEPS NYC/Istanbul, 4 seeds (within-user user-grouped CV =
                                     Gowalla-parity). ETL ✅ + Mac dry-run ✅ done (pipeline + champion
                                     transfer confirmed DIRECTIONALLY: MTL beats STL cat ceiling +9–10 pp,
                                     matches STL reg ceiling ±1 pp, clears the Markov floor). Paper numbers
                                     NEED the FROZEN substrate (P2) on a CUDA box — the Mac/ResLN-80ep dry-run
                                     substrate is throwaway. Use the corrected `--canon none` recipe
                                     (DRY_RUN_RESULTS.md), or `--canon v16` (needs the v14 substrate built).
                                     NOTE: NOT the temporal-split bridge — the shipped Massive-STEPS split is
                                     user-stratified RANDOM, not temporal (F1); the bridge (roadmap A5) is the
                                     separate Phase E2 chronological per-user re-split (Mac, no freeze dep).
```

**Why exploration is strictly first** (the user's hierarchy): a lever promoted by `mtl_frontier` or a
gate tripped by `pre_freeze_gates`/overlapping-windows changes what the freeze pins; if the base were
regenerated first, that spend would be thrown away. Inventory (`P1a`) likewise can surface a parked
lever that becomes a gate. So Level 0 → Level 1 → freeze, no shortcuts.

**Why `second_dataset` runs in parallel and off the critical path:** its ETL touches nothing the freeze
pins (a different corpus), so the Mac builds it during Levels 0–1 — **DONE (2026-06-15): Phase E ETL
complete for both NYC + Istanbul, with a Mac dry-run confirming the pipeline runs end-to-end and the
champion's behaviour transfers directionally.** Only its *validation runs* (Phase V, paper numbers)
depend on the frozen champion, so they sit at Level 4 (blocked on P2 + a CUDA box). Note the validation phase is **not** the temporal-split
bridge — the shipped Massive-STEPS split is user-stratified RANDOM, not temporal (F1). The bridge
(roadmap A5) is the **Phase E2 chronological per-user re-split** built from the corpus's per-check-in
timestamps — **✅ built 2026-06-16 (`scripts/second_dataset/build_chrono_split.py`)**; Mac-track, no
freeze dependency. Only its champion *runs* join the Level-4 validation (post-freeze, CUDA).

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
| **A40** | unmetered workhorse | `mtl_frontier` R4–R9 (RUNNING) | Currently saturated by the conditional-coupling family (R4–R9). Pre-freeze GPU gates wait for it to free. |
| **Mac M2 Pro** (user's local box) | local, MPS only | ✅ `second_dataset` Phase E + E2 DONE · ✅ **C1 confirm-on-G CLOSED (PR #26)** → now **board: build the light baseline embeddings** ([`closing_data/HANDOFF_BOARD.md`](closing_data/HANDOFF_BOARD.md)) | MPS fp32. Device rule: every paired comparison on ONE device-class — Macs build device-tolerant embeddings or own whole small states (AL/AZ), never split a comparison across MPS/CUDA. |
| **M4 Pro** (user's 2nd Mac) | local, MPS only | ✅ `pre_freeze_gates` **A2 + A4 CLOSED (PR #27)** → board: light-cell overflow per [`closing_data/HANDOFF_BOARD.md`](closing_data/HANDOFF_BOARD.md) | optional second Mac for light baseline embeddings / small-state cells (same device rule). |
| **H100** | **6 h metered** | reserved for `closing_data` P3 (CA/TX v14 builds) | Do **not** spend on pre-freeze exploration. |

---

## Gate ledger (feeds `closing_data` G0.2 — the freeze cannot commit with an open row)

> **Freeze-readiness status (2026-06-17, post PR #24/#26/#27 merge + audit).** CLOSED: **C1** (supportive panel), **A2** (claim strengthened), **A4** (defusal), **R1/R2/R3/R10** (all null — `mtl_frontier` CLOSED, champion G unchanged, nothing flows here). **OPEN — the freeze cannot commit until these resolve:** **G0.1** aligned-pairing (the lone *recipe-changing* P0 gate), **Overlapping-windows** ADOPT/KEEP (base change), **B1–B5** baseline triage (P1b RUN_MATRIX). **Cross-cutting prerequisites before P2** (audit 2026-06-17, full checklist in [`closing_data/FREEZE_READINESS.md`](closing_data/FREEZE_READINESS.md)): regenerate **ONE** canonical v14 substrate (fixed machine + build seed) + **hash all consumers** — the three studies' substrates are non-identical local rebuilds, so within-study deltas hold but the **absolutes are NOT cross-comparable** to the board; centralize the stale-`log_T` freshness preflight + rebuild the currently-stale AL `design_k` log_T.
>
> **A40 UPDATE (2026-06-19, `study/pre-freeze-a40`) — the OPEN rows above are now CLOSED.** **G0.1** → ADVISORY NULL (recipe stays **v16**). **Overlapping-windows** → ADOPT validated (Lane 2). **v14 hash-manifest** → **all 6 states** manifested (CA+TX built; **§0 STOP lifted**). **Freshness preflight** → centralized; AL not stale. **B1–B5 baselines** → 7 INCLUDE implemented + adversarially audited + enum-registered; POI2Vec resolved (GeoTreeSkipGram honest rename + faithful AAAI'17 build). New production code is **BYTE-IDENTICAL** (FL non-overlap MTL scan = 73.0116/73.5414 exact). **Re-baseline knob:** `--compile --tf32` ADOPTED + pinned for the P3 board (result-neutral +0.05 pp, ~15% faster); torch stays 2.11; workers skipped. Remaining are P2/P3, **not freeze-blocking**: the n=20 board build, B1/B2b per-fold scoring via `--only-fold`, the B2a/GeoTreeSkipGram include/exclude decision. Per-gate docs: [`pre_freeze_gates/`](pre_freeze_gates/) + [`closing_data/FREEZE_READINESS.md`](closing_data/FREEZE_READINESS.md).

| Gate | Study | Promote condition | On promote | On null |
|---|---|---|---|---|
| G0.1 aligned-pairing | closing_data P0 | ≥0.3 pp either head, multi-seed | recipe → v17 | v16 freezes; "wins without per-sample mixing" earned | **✅ ADVISORY NULL 2026-06-18 (A40, AL+FL seed0)** — FL null (cat +0.17, reg ±0.00), AL aligned HURTS cat (−4.77; random pairing is a beneficial augmentation at small data, not a confound). No positive promotion → v16 stands; X1 circularity closed on the real aligned-training counterfactual. Binding {0,1,7,100} run on the frozen base is the formal closure (expected null). Verdict: [`pre_freeze_gates/LANE1_G01_VERDICT.md`](pre_freeze_gates/LANE1_G01_VERDICT.md). |
| R1 log_C co-location prior | mtl_frontier | ≥0.3 pp reg over log_T-KD-alone, multi-seed | recipe → v17 | drop / future-work | **✅ CLOSED null 2026-06-17** — AL +0.207±0.196 (gate FAIL, p=0.008), FL null; weight non-monotonic. Real-but-sub-threshold over log_T-KD. Not v17; code behind `--log-c-kd-weight` (default off). |
| R2 STEM-AFTB gating | mtl_frontier | ≥0.3 pp either head over G, multi-seed | recipe → v17 | G's current sharing is the optimum (citable) | **✅ CLOSED null 2026-06-17** — reg clean multi-seed null all states; cat AL-only and decays with scale (FL −0.026). Sharing topology doesn't move the reg gap (cos≈0). Not v17. |
| R3 cross-task distillation | mtl_frontier | ≥0.3 pp over R1/log_T teacher | recipe → v17 | static teacher suffices | **✅ CLOSED null 2026-06-17** — CrossDistil (live-teacher warm-up + error-correction, fwd + reverse): fwd null, reverse +0.100±0.282 (p=0.31), FL null. log_T-KD saturates the output-prior family. Not v17. |
| R10 GRM/SSC gated read (arXiv 2602.24281, ★ user) | mtl_frontier | ≥0.3 pp either head over G, multi-seed (needs working impl) | recipe → v17 | GRM ≡ hand-built asymmetry (citable null) | **✅ CLOSED null 2026-06-17** — GRM impl faithful (gate live, γ 0.87→0.31) yet **≡ champion G** (the spec's own falsifier); FL +0.085±0.203 null; all 6 placements (cross-attn, hierarchical-fusion, reg-head, intra-tower P4, GRU-head P5, SSC/Soup) null. Not v17. |
| A2 feature-concat control | pre_freeze_gates | n/a (interpretation gate) | reframes the substrate claim in the new paper | substrate claim stands as-is | **✅ RESOLVED 2026-06-17 (M4) → ON NULL: claim STRENGTHENED.** HGI⊕(Check2HGI's exact node features) closes only **7–8%** of the v14 next-cat gap (12–18% of the v11 paper-canon gap) at AL+AZ (n=20); concat lift +1.6–2.0pp vs a +23–24pp gap. Reg: concat inert (−0.13pp, NS), HGI leads region anyway. ⇒ the +14–29pp cat lift is the hierarchical-infomax *learning*, not feature injection. Conservative (concat arm has wider input, 75 vs 64, and still loses). See [`pre_freeze_gates/A2_RESULTS.md`](pre_freeze_gates/A2_RESULTS.md). |
| A4 transductivity bound | pre_freeze_gates | n/a (disclosure gate) | report delta + re-anchor headline numbers | one-paragraph defusal in the paper | **✅ RESOLVED 2026-06-17 (M4) — ON NULL on BOTH axes, AL+FL seed0.** Train-only v14 rebuilt per fold. **Reg inflation ≈0 (AL −0.33pp, FL −0.12pp; full arms reproduce A2 61.90/70.09 → validated)** — low-sensitivity (log_T dominates reg) so cat measured too. **Cat (substrate-driven axis, where the +14–29pp lift lives) measured via in-coverage POI proxy: ≈0 (AL +0.29pp @66.8% cov, FL +0.00pp @86.9% cov)** → POI representations NOT transductively inflated. ⇒ one-paragraph defusal; headline numbers don't need re-anchoring. Caveats: cat is a POI-level proxy on the in-coverage subset; the contextual/cold-POI residual is bounded by inductive-Check2HGI future-work (inductive wall). See [`pre_freeze_gates/A4_RESULTS.md`](pre_freeze_gates/A4_RESULTS.md). |
| Overlapping-windows | pre_freeze_gates | adopt only if user accepts full-base rebuild + a clean leak re-audit | stride change → leak re-audit, rebuild ALL bases + priors pre-freeze, and the regime baselines must match | keep non-overlap canon (default) |
| B1–B5 baseline triage | baseline_gap | n/a (inventory decision) | chosen baselines → RUN_MATRIX rows/columns; **final runs on the frozen base** (blocked on overlap decision + freeze; end-to-end baselines mirror the adopted windowing) | baseline excluded from the matrix with a recorded reason |
| **C1 3-snapshot per-task routing** (★ ESCALATED to G0.2 by user 2026-06-16; P1a recommended STORY-DEPENDENT) — **✅ CLOSED 2026-06-17: PROMOTE as a SUPPORTIVE panel** (M2 Pro, FL+AL × {0,1,7,100}, n=20/state: AL Δreg **+1.55 pp** p=0.0001, FL Δreg **+0.63 pp** p<0.0001, cat not hurt; POOLED +1.09 pp). **NOT subsumed on G**; FL's gain ≪ pre-C25 +2.80 confirms most was recovered, a real residual remains. **User scope (2026-06-17): supportive *diagnostic* panel ONLY — NOT the primary deploy.** Per-task routing forfeits the single-model property (2 checkpoints / 2 forwards ≈ task-specialised models), which would concede the MTL thesis; the single `geom_simple` checkpoint stays the headline. C1 = deploy-time per-task *selection* headroom (single-ckpt ≤ C1 ≤ STL ceiling), not the task ceiling. Verdict: [`closing_data/C1_VERDICT.md`](closing_data/C1_VERDICT.md). | closing_data (confirm-on-G) · run on **M2 Pro** | ≥0.3 pp reg over the single `geom_simple` checkpoint, **on champion G**, multi-seed (don't hurt cat) — **MET** | adopt as a **supportive diagnostic panel** (two-checkpoint routing), NOT primary; it is a **deploy mode**, not a single-model recipe change | confirmed dead on G (the C25-fix + dual-tower + geom_simple already recovered the 2/3-state pre-C25 signal) |

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
