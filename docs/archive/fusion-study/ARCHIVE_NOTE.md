# Fusion Study — Archive Note

**Archived:** 2026-05-14
**Last live update:** see git log of this folder
**Reason for archive:** Superseded by the **check2hgi** study, which became the project's primary research line in Apr–May 2026 and produced the BRACIS 2026 paper (`articles/[BRACIS]_Beyond_Cross_Task/`).

## Status at archive time

The fusion study is a **closed body of work**. It was the project's master ablation effort from early Apr 2026 through ~mid-Apr 2026, organized around a claim catalog (C01..C32) executed via a multi-phase coordinator (P0..P6). Most claims were resolved (refuted, confirmed, or reframed) before the study was wound down in favor of check2hgi.

Final-state highlights:

- **P1 (arch × optimizer grid)** — completed: 180-test grid on AL+AZ. Equal-weight beats adaptive MTL methods; CGC best architecture (see `CLAIMS_AND_HYPOTHESES.md` C-series outcomes).
- **P2 (heads + MTL)** — completed: heads do not transfer between STL and MTL; small MTL benefit observed at multi-seed.
- **P3 (embedding cross)** — partial: C01 refuted strong form; C11 confirmed cross-engine; FL infeasible on MPS (state-asymmetric findings).
- **P4 (hyperparams)** — champion defaults robust at multi-seed.
- **HGI leakage diagnosis** — durable diagnosis preserved in `issues/HGI_LEAKAGE_AUDIT.md` and `issues/HGI_LEAKAGE_EXPLAINED.md`. This diagnosis directly motivated check2hgi.
- **NashMTL ECOS solver bug** — discovered and documented (see project memory `nash_mtl_solver_bug`); affected all prior NashMTL findings.

## What survives in this archive

- **Concepts**: `AGENT_CONTEXT.md`, `MASTER_PLAN.md`, `KNOWLEDGE_SNAPSHOT.md`, `STUDY_OVERVIEW.md`, `QUICK_REFERENCE.md`, `COORDINATOR.md`, `HANDOFF.md`.
- **Claims & hypotheses**: `CLAIMS_AND_HYPOTHESES.md`.
- **Phase designs**: `phases/P0_preparation.md` … `phases/P6_convergence_and_mtl_claims.md` + `phases/P1_grid.yaml`.
- **Critical reviews & plans**: `plans/CRITICAL_REVIEW_ATTACK_PLAN.md`.
- **Issue diagnoses (durable knowledge)**: `issues/HGI_LEAKAGE_AUDIT.md`, `issues/HGI_LEAKAGE_EXPLAINED.md`, `issues/P1_METHODOLOGY_FLAWS.md`.
- **Pre-fusion archive**: `archive/ablation_studies/`, `archive/full_ablation_study/`.
- **Coordinator design specs**: `coordinator/integrity_checks.md`, `coordinator/state_schema.md`.
- **Scientific record (results)**: full `results/P0/` tree — CSVs, JSONs, integrity files, leakage-ablation outputs, frozen-fold registry.
- **Lab-notebook records**: `state.json` (run timeline), `machines.yaml` (state-pinning contract), `assignments/P1_2026-04-16_screen/` (who-ran-what-where forensic record). Per advisor input these are kept as part of the "how this study was conducted" provenance.

## What does NOT survive (intentional drops)

Nothing is dropped from this archive. All files present at archive time are retained.

## Code support for the FUSION engine post-archive

The fusion study is archived; **the FUSION engine in the codebase is NOT deprecated.** Code paths remain first-class and supported:

- `src/configs/paths.py` — `EmbeddingEngine.FUSION` enum + FUSION routing in `IoPaths`.
- `src/data/inputs/fusion.py` — fusion input builder.
- `experiments/full_fusion_ablation.py` — fusion ablation harness.
- `pipelines/fusion.pipe.py` — fusion pipeline.
- `output/fusion/<state>/` — gitignored runtime artefact convention.

If a future study revives fusion (or extends it), the framework is ready.

## Why this study was archived

Key narrative (preserved here so future agents don't re-derive):

1. **Bug discoveries during the fusion study uncovered methodological holes**: the NashMTL ECOS solver bug (silent fallback to equal-weight), HGI leakage at the train/val boundary (per-fold transition matrices fixed it), and CrossEntropy weighting asymmetries. These motivated a fresh-start pass.
2. **The check-in-level Check2HGI substrate emerged as a stronger research thesis** than the multi-embedding fusion thesis. The substrate-task-asymmetry framing (per-visit context lifts cat F1 by +14–29 pp depending on state) became the BRACIS paper's headline.
3. **The fusion study's coordinator/state.json/machines.yaml infra became the template** for the check2hgi study workflow — that infra is reused, the experimental work is closed.

## How to reach back into this archive

Read in this order if you need fusion-study context:

1. `KNOWLEDGE_SNAPSHOT.md` — what we knew at study close.
2. `STUDY_OVERVIEW.md` — what the study set out to do.
3. `CLAIMS_AND_HYPOTHESES.md` — claim outcomes (status field on each).
4. `issues/HGI_LEAKAGE_*.md` — the load-bearing methodological finding.
5. `plans/CRITICAL_REVIEW_ATTACK_PLAN.md` — the original critical-review framing.
6. `phases/P0_preparation.md` … `phases/P6_*.md` — phase-by-phase design.

Pointers from the active project to here:
- `docs/CHANGELOG.md` — fusion-era entries.
- `docs/AGENT_CONTEXT.md` — historical context section.
- `CLAUDE.md` — single-line pointer in "Project focus" section.
