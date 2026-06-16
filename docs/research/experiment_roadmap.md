# Experiment Roadmap — Prioritized

> Compiled 2026-06-12. Builds on the closing_data scaffolding (`docs/studies/closing_data/PLAN.md`) — items below are tagged where they should fold into its pre-freeze gates (P0–P2) or its regeneration matrix (P3) versus standing alone. Effort estimates assume the existing A40/H100/M4 split.

---

## A. Must-have (blocking for any credible submission)

| # | Experiment | Why | Scope / cost | Fits in |
|---|---|---|---|---|
| A1 | **Restate the experimental base under v16** (champion G + STL ceilings re-run, all 5 states × 4 seeds × 5 folds, C25-fixed, geom_simple selector) | The v11 "MTL pays" narrative is confound-driven; CA/TX champion cells don't exist yet | Exactly `closing_data` M1–M4 | closing_data P3 |
| A2 | **Feature-concat control**: HGI ⊕ raw per-visit features (category one-hot + hour/dow sin/cos), matched heads, STL cat + reg, ≥3 states | Decides whether hierarchical infomax adds anything beyond feature injection; the biggest internal-logic hole in CH16 | Cheap: no embedding training, input-builder change + standard STL runs | closing_data pre-freeze gate (new) |
| A3 | **CTLE substrate baseline**: pre-train CTLE per state (train-portion-only per its protocol), freeze, feed matched heads, STL cat + reg, ≥3 states | The head-on published competitor for "contextual per-visit embeddings"; reviewers will ask first | Moderate: external codebase ([github.com/Logan-Lin/CTLE](https://github.com/Logan-Lin/CTLE)), 64-d, per-state pretraining | closing_data RUN_MATRIX (new substrate column) |
| A4 | **Transductivity bound**: retrain Check2HGI per fold on train users only, 1 state (FL) × 1 seed, compare downstream cat/reg vs full-corpus substrate | Quantifies the one open leakage channel (see [`evaluation_protocol_review.md §4.1`](evaluation_protocol_review.md)); cheap insurance against a desk-reject question | 5 substrate trainings + standard STL runs | closing_data pre-freeze gate (new) |
| A5 | **Temporal-split bridge**: a **chronological per-user re-split** (~80/10/10) with matching per-fold train-only priors. **Best done as a `second_dataset` Phase E2 addendum on Massive-STEPS** (Mac-track, no CUDA, no freeze dependency — build the split from per-check-in timestamps), which lands the bridge on a modern non-US+US corpus; the **Gowalla-side chronological re-split (FL + AL)** is the fallback. Then run champion G, STL ceilings, Markov-1, (CTLE if A3 done) | Connects to the field's universal protocol; tests forward-time generalization the CV protocol never touches | One split implementation + ~10 runs | standalone (pre-paper) |
| A6 | **Ground-up paper rewrite** (user decision 2026-06-12: the v11 BRACIS draft is dropped): new thesis on v14 substrate + champion G, fed by closing_data outputs; include the affirmative task-pairing defense (sparsity/privacy/deployment, HMT-GRN sparsity citation), the LBSN-MTL-frontier positioning ([`mtl_frontier.md §3`](mtl_frontier.md)), and full disclosures (alternating-step no-op, transductivity, cold-user protocol) | The old draft's central MTL narrative was confound-driven; the new story is the inverted (Pareto) result | Writing, gated on A1 | paper folder + closing_data P4 |

**Ordering note**: A2 and A4 are pre-freeze gates — if either falsifies the substrate story, the closing_data full regeneration (A1) would be re-running a dead narrative. Run A2/A4 first, then freeze, then A1; A3/A5 can run in parallel on the second machine.

## B. Should-have (turns a defensible paper into a strong one)

| # | Experiment | Why |
|---|---|---|
| B1 | **External MTL baseline**: HMT-GRN-style shared-LSTM + per-task heads (category+region) on the repo's data | The MTL table currently has zero external MTL rows |
| B2 | **Cascade baseline**: category→region conditional (CSLSL/CatDM pattern) vs parallel cross-attn | The dominant published alternative to parallel MTL; cheap |
| B3 | **POI2Vec / skip-gram substrate columns** (POI2Vec already in-repo as HGI input) | Completes the canonical embedding-baseline set |
| B4 | **CH19 cross-state replication** (POI-pooled counterfactual at FL + 1 more state) | The mechanism claim (~72% per-visit context) is AL-only; it is the paper's most interesting explanatory claim |
| B5 | **Sequential baselines**: Flashback (sparse-trace regime, AL/AZ) and/or DeepMove adapted to region targets | STAN alone is thin for 2026 |
| B6 | **Bridging metrics everywhere**: cat Acc@1; reg Acc@1/@5/MRR in all tables (already computed by `src/tracking/metrics.py` — reporting change only) | Reviewer calibration |
| B7 | **Nested model selection or per-state holdout** for one confirmation pass of the headline cells | Removes the selection-on-reporting-fold objection |
| B8 | **MTL-gain program, first wave** (from [`mtl_frontier.md §4`](mtl_frontier.md)): R1 log_C co-location prior (P(region\|cat) probability-chain, same per-fold infra as log_T), R2 STEM-style AFTB gating sweep, R3 live calibrated cat→reg cross-task distillation | The only lever class that has ever moved MTL reg is the prior pathway; these are its three literature-aligned extensions. R1–R3 are closing_data **pre-freeze-gate compatible** — a ≥0.3 pp win promotes the recipe to v17 before the base regeneration |
| B9 | **Second dataset validation phase**: Massive-STEPS NYC/Istanbul (champion G + STL ceilings + Markov floor, 4 seeds; within-user user-grouped CV = Gowalla-parity) — see [`future_work.md §8`](future_work.md) | Breaks Gowalla-specificity on a public, license-clean benchmark. **B9 (validation) does NOT auto-deliver A5** — the shipped Massive-STEPS split is user-stratified RANDOM, not temporal (F1). A5 is delivered by a separate **Phase E2 chronological per-user re-split** built on the same corpus from timestamps (Mac, no freeze dependency); Gowalla-side re-split is the fallback. Report gap-to-ceiling, not absolute Acc@k (F2). |

## C. Nice-to-have (strengthens, not blocking)

| # | Experiment | Why |
|---|---|---|
| C1 | Standard-benchmark port: one city from FSQ-NYC/TKY or Massive-STEPS, category+region tasks | External anchor; first step toward international-venue submission |
| C2 | Geohash-grid secondary region definition at FL (G@5-style) | Connects region results to HMT-GRN conventions; tests tract-choice sensitivity |
| C3 | Sliding-window (stride-1 / overlapping) ablation — **now user-committed** (overlapping_windows future-work memo planned) | Window-construction robustness (§4.4 of the protocol review). ⚠ Must land **pre-freeze** (it changes the evaluation base), and per the R1 rising-tide finding it will lift STL and MTL equally — see [`future_work.md §7`](future_work.md) |
| C4 | TALE / CACSR / LBSN2Vec substrate columns | Long-tail embedding baselines |
| C5 | LLM zero-shot reference row (LLM-Mob/AgentMove prompt style on 1 state) | 2025/26 reviewer expectation management |
| C6 | True PLE (inter-level gate chain) re-run | Only if a PLE claim remains in the paper |
| C7 | Ablations already largely done — keep curated: substrate axis (done), head axis (done), MTL-optimizer sweep (done, all null), sharing dose-response (done), dual-tower vs symmetric (done, G′ falsified), log_T-KD lever (done) | The repo's existing ablation coverage is a strength; the gap is *external*, not internal |

## D. Explicitly de-prioritized

- More MTL-optimizer arms (19 nulls + Kurin/Xin support; saturated). Two narrow exceptions remain citable-cheap: BayesAgg-MTL (verify the existing `src/losses/registry.py` implementation matches ICML 2024) and Smooth-Tchebycheff *only if* Pareto-front profiling reveals a non-convex front — see [`mtl_frontier.md §2.6`](mtl_frontier.md).
- More Check2HGI encoder variants for MTL (regime finding: substrate/encoder improvements wash out under the joint regime; v13/v14 are STL-only).
- ReHDM-STL resurrection (diagnosed architecture-bound; footnote only).
- Symmetric dual-tower (G′ falsified at small states 2026-06-07).

---

## E. Study-family organization (the "how/where/when" — added 2026-06-14)

The prioritized items above (the *what*) are executed by a concrete **pre-freeze study family** (the
*how/where/when*). Operational orchestrator with the full dependency DAG, gate ledger, and machine map:
**[`docs/studies/PRE_FREEZE_PROGRAM.md`](../studies/PRE_FREEZE_PROGRAM.md)**. This section is the bridge
between the two.

### Governing rule

Anything that can change the **recipe**, the **substrate**, or the **evaluation base** MUST resolve
**before** the `closing_data` **P2 FREEZE** — the freeze is a hard barrier and the P3 regeneration (every
state × 4 seeds × 5 folds) cannot be cheaply re-run. Hence the hierarchy below: exploration/research first
(it influences what the freeze pins), gates second, then the single heavy spend.

### The five levels (hierarchy + parallelism)

| Level | What | Studies | Machine | Parallel? |
|---|---|---|---|---|
| **0 — Exploration & inventory** (first; outputs influence the freeze) | MTL frontier levers; cross-study re-eval; baseline triage; 2nd-corpus ETL | `mtl_frontier`, `closing_data` P1a, `baseline_gap` (decision), `second_dataset` (ETL) | A40 + Mac + reading | yes — A40 runs frontier, Mac runs ETL, reading is free |
| **1 — Pre-freeze gates** (cheap base/recipe tests) | feature-concat, transductivity, overlapping-windows, aligned-pairing | `pre_freeze_gates`, `closing_data` G0.1 | A40 | after/with Level 0 |
| **2 — FREEZE** ★ user sign-off | recipe + substrate + `RUN_MATRIX.md` pinned | `closing_data` P2 | — | hard barrier |
| **3 — Single heavy spend** | full base regeneration (STL + champion + suite + chosen baselines) | `closing_data` P3 | A40 unmetered + H100 6 h (CA/TX) | within-level only |
| **4 — External validation** (may overlap P3 tail) | champion + ceilings + Markov on the 2nd corpus (within-user CV = Gowalla-parity); the temporal bridge (A5) is the separate Phase E2 chronological re-split | `second_dataset` Phase V (+ E2 for A5) | CUDA box (Phase V); Mac (Phase E2 split build) | yes |

### Roadmap item → owning study

| Roadmap items | Owning study | Level | Note |
|---|---|---|---|
| A1 (restate base), A6 (paper rewrite) | `closing_data` (P3 / P4) | 3 / 4 | A6 gated on A1 |
| A2 (feature-concat), A4 (transductivity), C3 (overlapping-windows) | `pre_freeze_gates` | 1 | A2/A4 interpretation+disclosure gates; C3 is a base-change ADOPT/KEEP decision |
| A3 (CTLE), B1 (HMT-GRN MTL), B2 (cascade), B3 (POI2Vec cols), B5 (Flashback/DeepMove), C4 (TALE/CACSR/LBSN2Vec), C5 (LLM row), C6 (true PLE) | **`baseline_gap`** (B1–B5 + Tier-3) | 0/1 decision → 3 runs | the audit-gap corrective — triage feeds RUN_MATRIX, runs fold into P3 |
| A5 (temporal bridge), B9 (2nd-dataset validation) | `second_dataset` | 0 (ETL) / 0 (E2 split build) / 4 (validation) | B9 (validation, within-user CV) does NOT auto-deliver A5 — the shipped split is RANDOM, not temporal (F1). A5 = a separate **Phase E2 chronological per-user re-split** on Massive-STEPS (Mac, no freeze dep), best done here on a modern non-US+US corpus; Gowalla-side re-split is the fallback. |
| B8 (MTL-gain first wave R1–R3) + R10 (★ GRM/SSC) | `mtl_frontier` | 0 | ≥0.3 pp lever → v17 pre-freeze gate (user sign-off) |
| B4 (CH19 cross-state), B6 (bridging metrics), B7 (nested selection) | `closing_data` P1b/P3 reporting | 1/3 | reporting + one mechanism replication; no new study needed |
| C1, C2 (benchmark port, geohash regions) | future-work / post-paper | — | not in the closing scope |
| D (de-prioritized) | — | — | optimizer aisle closed; substrate-for-MTL closed; G′ closed |

### Why this ordering (the user's hierarchy, restated)

A lever promoted by `mtl_frontier`, a gate tripped by `pre_freeze_gates`, a baseline added by
`baseline_gap`, or a parked lever surfaced by `closing_data` P1a each changes what the freeze pins — so all
of Level 0/1 must close before Level 2. Regenerating the base first (Level 3) would throw that spend away.
`second_dataset` is the one exception that runs concurrently throughout: its corpus touches nothing the
freeze pins, so only its *validation runs* wait (Level 4).

### Comparability regime for baselines (the binding constraint)

Baselines are only comparable if they run on the **exact frozen base** the champion uses. So any base-level
change — substrate identity (v14/v17), the **overlapping-window** decision (stride), the fold/seed/prior
protocol, the label spaces — pins what the baselines must match. Substrate-column baselines (CTLE, POI2Vec)
inherit this through the matched-head pipeline; **end-to-end baselines (HMT-GRN-style, cascade,
Flashback/DeepMove) build their own sequences and must mirror the adopted windowing/splits.** Hence
`baseline_gap` may implement/smoke-test early but its **paper-grade runs are blocked on the
overlapping-window ADOPT/KEEP decision + the P2 freeze**, and end-to-end baselines re-run if overlap is
adopted. Adopting overlapping windows additionally requires re-running the window/causal-mask **leak audit**
(the prior audit covered only non-overlapping windows). Full spec: [`docs/studies/baseline_gap/AGENT_PROMPT.md §Comparability regime`](../studies/baseline_gap/AGENT_PROMPT.md) and [`docs/studies/PRE_FREEZE_PROGRAM.md`](../studies/PRE_FREEZE_PROGRAM.md).
