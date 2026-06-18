# docs/studies/ — Active follow-up studies on check2hgi

This folder hosts ongoing research tracks layered on top of the primary check2hgi study (`docs/` root). Each subdir is a self-contained study with its own briefing, design, log, and findings.

## ⭐ Pre-freeze program (2026-06-14) — the study family launching the bases for `closing_data/`

Orchestrator + dependency DAG + machine map: **[`PRE_FREEZE_PROGRAM.md`](PRE_FREEZE_PROGRAM.md)**. Hierarchy: exploration/inventory (Level 0) → pre-freeze gates (Level 1) → `closing_data` FREEZE (Level 2) → base regeneration (Level 3) → external validation (Level 4). The three studies below are the upstream tracks; they feed `closing_data` G0.2 and must resolve before its freeze.

| Study | Status | Machine | Read first |
|---|---|---|---|
| [`mtl_frontier/`](mtl_frontier/) | **✅ CLOSED 2026-06-17 (in place; branch `study/mtl-frontier`)** — 10 lever-families: **9 nulls + 1 sub-threshold positive (cc)** + R4 (Pareto front, resolves C21) / R7 (Merge<joint-G) / R9 (optimizer aisle closed); **no v17 lever; champion G unchanged; NOTHING flows to G0.2.** The post-2022 frontier replicates-not-exceeds champion G. Headline: the cat↑/reg↓ "loss" is **reg-at-its-STL-ceiling (not lost)** + the paper-C2 class-weighting confound (see the C2 memo). | A40 | [`mtl_frontier/FINAL_SYNTHESIS.md`](mtl_frontier/FINAL_SYNTHESIS.md) ⭐ |
| [`pre_freeze_gates/`](pre_freeze_gates/) | **SCAFFOLDED 2026-06-14** — substrate/base gates: A2 feature-concat control, A4 transductivity bound, overlapping-windows adopt/keep (base change) | A40 | [`pre_freeze_gates/AGENT_PROMPT.md`](pre_freeze_gates/AGENT_PROMPT.md) |
| [`second_dataset/`](second_dataset/) | **Phase E (ETL) COMPLETE — 2 cities, 2026-06-15** — Massive-STEPS **Istanbul (PRIMARY, non-US; regions: mahalle real-admin PRIMARY + H3 secondary)** + **NYC (secondary, US, TIGER)** built on the Mac (parse + shared FSQ→7-root map + regions + BOTH protocols, leak-free; reviewed vs source+paper). ⚠ shipped split is user-stratified random over trails, NOT temporal (bridge rationale falsified); lead with within-user Gowalla-parity set. Phase V validation blocked on CUDA + frozen substrate | Mac (ETL ✓) + CUDA (validation) | [`second_dataset/PHASE_E_REPORT.md`](second_dataset/PHASE_E_REPORT.md) |
| [`baseline_gap/`](baseline_gap/) | **SCAFFOLDED 2026-06-14** — net-new external baselines (B1 CTLE, B2 POI2Vec/skip-gram, B3 HMT-GRN-style MTL, B4 cascade, B5 Flashback/DeepMove); triage → RUN_MATRIX decision pre-freeze, runs fold into P3 | A40 + Mac | [`baseline_gap/AGENT_PROMPT.md`](baseline_gap/AGENT_PROMPT.md) |

## Active / pending-decision studies

| Study | Status | Branch (post-merge) | Read first |
|---|---|---|---|
| [`closing_data/`](closing_data/) | **SCAFFOLDED (not launched; re-scoped 2026-06-12)** — the experimental engine for the NEW paper: cross-study re-eval + BRACIS-suite inventory (RUN_MATRIX) → pre-freeze gates → recipe+substrate FREEZE → full base regeneration ONCE (STL baselines re-run + champion + suite cells, ALL states × 4 seeds × 5 folds) → hand-off to the story effort. Launch pending user sign-off on the plan | (TBD) | [`closing_data/HANDOFF.md`](closing_data/HANDOFF.md) → [`closing_data/AGENT_PROMPT.md`](closing_data/AGENT_PROMPT.md) + [`closing_data/PLAN.md`](closing_data/PLAN.md) |
| [`merge_design/`](merge_design/) | **CLOSED** — Designs A-M and Levers 1-6 all falsified or saturated; all structural RQs closed; Lever 5 rescued+falsified by `substrate-protocol-cleanup` Tier B4; design_k graduated into the v14 substrate. No surviving open item. Per [`closing_data/PHASE1_VERDICT.md`](closing_data/PHASE1_VERDICT.md) §1 (cross-study table) + §2 (lever dispositions). | (no dedicated branch) | [`merge_design/STATE.md`](merge_design/STATE.md) |
| [`mtl-exploration/`](mtl-exploration/) | older exploration track (no-encoders / HGI-substrate / leak-blast audits); kept for reference | (no dedicated branch) | [`mtl-exploration/README.md`](mtl-exploration/README.md) |

> The four upstream pre-freeze studies (`mtl_frontier`, `pre_freeze_gates`, `second_dataset`, `baseline_gap`) are listed in the ⭐ Pre-freeze program table above.

## Archived studies — [`archive/`](archive/)

Fully-closed studies, moved out of the active set 2026-06-14 (see [`archive/README.md`](archive/README.md)). Their findings remain authoritative and are heavily cross-referenced; archiving is an organizational move, not a deprecation.

| Study | Closed | One-line outcome |
|---|---|---|
| [`archive/mtl_improvement/`](archive/mtl_improvement/) | 2026-06-12 | C25 confound dissolved the "MTL sacrifices reg" gap; champion **G (= canon v16)** matches the STL reg ceiling + beats cat +2.6…+4.1 (4 states × 4 seeds); gradient-orthogonal regime; no optimizer helps. Read [`FINAL_SYNTHESIS.md`](archive/mtl_improvement/FINAL_SYNTHESIS.md) |
| [`archive/embedding_eval/`](archive/embedding_eval/) | 2026-06-02 | champion **v14 = `check2hgi_design_k_resln_mae_l0_1`**; design_k re-validated at FL; NO MTL benefit (regime is the wall). Read [`FINAL_SYNTHESIS.md`](archive/embedding_eval/FINAL_SYNTHESIS.md) |
| [`archive/substrate-protocol-cleanup/`](archive/substrate-protocol-cleanup/) | 2026-05-29 | log_T-KD PROMOTED (v12 default); substrate axis NULL in MTL (regime-limited, even HGI ≈ canonical). Read [`CLOSURE.md`](archive/substrate-protocol-cleanup/CLOSURE.md) |
| [`archive/mtl-protocol-fix/`](archive/mtl-protocol-fix/) | 2026-05-24 | F1-selector fix + P4 mechanism ID (residual gap is architectural). Read [`AGENT_PROMPT.md`](archive/mtl-protocol-fix/AGENT_PROMPT.md) |
| [`archive/canonical_improvement/`](archive/canonical_improvement/) | 2026-05-19 | Tier 1-6 substrate axis exhausted at ±0.8 pp ceiling. Read [`AGENT_PROMPT.md`](archive/canonical_improvement/AGENT_PROMPT.md) |
| [`archive/hgi_category_injection/`](archive/hgi_category_injection/) | 2026-05-04 | AZ falsified; category injection on HGI POI2Vec inert. **Do NOT treat as active without an explicit re-open commit.** Read [`STATUS.md`](archive/hgi_category_injection/STATUS.md) |
| [`archive/fusion/`](archive/fusion/) | — | leftover fusion `results/` snapshot (the fusion *study* proper is at [`../archive/fusion-study/`](../archive/fusion-study/)) |

## Folder semantics — `studies/` vs `studies/archive/` vs `findings/` vs `docs/archive/`

> **`studies/`** = active or pending-decision research tracks (still being worked on, or pending a launch/re-open decision).
> **`studies/archive/`** = fully-closed studies layered on check2hgi, moved here to declutter the active set; findings stay authoritative and cross-referenced.
> **`findings/`** = closed per-experiment findings supporting the paper (the F-trail; read-only history) at [`docs/findings/`](../findings/).
> **`docs/archive/`** = repo-level archive of *non-study* material (old reorg plans, the earlier fusion-study, pre-B3 framing snapshots).

## Archive policy

When a study is fully closed and unlikely to be re-opened, `git mv docs/studies/<study>` → `docs/studies/archive/<study>/` (keep the folder name; do **not** rename with a date suffix — the cross-references rely on the name), add a row to the Archived-studies table above and a line to [`archive/README.md`](archive/README.md), and fix inbound links (`studies/<study>` → `studies/archive/<study>`).

## Where do paper-supporting findings go?

Closed per-experiment findings that support the BRACIS paper (the "F-trail") live at [`docs/findings/`](../findings/), not here.

## Adding a new study

1. Create the subdir: `docs/studies/<study_name>/`.
2. Author a study-onboarding doc at the top — `AGENT_PROMPT.md` (if agent-driven), `README.md`, or `STATE.md` (per the existing studies' conventions).
3. Update this `README.md` with a row in the Active/pending-decision studies table (or the ⭐ Pre-freeze program table if it feeds `closing_data`).
4. If the study is large enough to warrant its own branch, branch from `main` (or the latest target branch) and document the branching point in the study's onboarding doc.
5. Optional: add a `state.json` for run-tracking (see [`archive/fusion-study/state.json`](../archive/fusion-study/state.json) for the pattern).

## Layered-study pattern

Studies in this folder are *layered on check2hgi*: they assume the canonical Check2HGI substrate, the B9 MTL recipe (or its small-state H3-alt variant), and the matched-head GRU evaluation protocol. They don't redo the substrate work; they extend it.

Read [`docs/AGENT_CONTEXT.md`](../AGENT_CONTEXT.md), [`docs/NORTH_STAR.md`](../NORTH_STAR.md), and [`docs/CLAIMS_AND_HYPOTHESES.md`](../CLAIMS_AND_HYPOTHESES.md) before starting a new study — they are the project-wide scientific baseline.
