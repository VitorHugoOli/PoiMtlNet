# Check2HGI Study Track — Entry Point

**Status:** P-1 active (2026-04-15) — generating vanilla check2HGI embeddings for FL + AL; MTL code parameterisation in parallel.
**Scope:** a parallel study to the legacy P0–P6 track. Investigates whether check-in-level contextual embeddings (**check2HGI**) plus a trajectory-aligned auxiliary task (**next-region**) improve next-POI-category prediction over the legacy HGI + category-classification MTL setup.

> Legacy study at `docs/studies/*.md` remains active on `main`; do not edit it on this branch. All work for this track lives **only** under `docs/studies/check2hgi/`.

## Why this study exists

Two shifts motivate a new track:

1. **Embedding engine:** the legacy track runs on POI-level embeddings (HGI / fused HGI+Sphere2Vec+Time2Vec). Check2HGI produces check-in-level embeddings — the same POI visited twice yields two different vectors. That makes a per-POI category-classification task semantically ambiguous, but it's precisely the representation a *trajectory* task (next region, next time-gap) benefits from.
2. **MTL auxiliary task:** replacing POI-category classification with **next-region prediction** re-uses the check2HGI region hierarchy directly and aligns with the HMT-GRN / MGCL lineage of hierarchical next-POI MTL.

Task-selection rationale is documented in `docs/issues/MTL_TASK_REPLACEMENT_PROPOSAL.md` and consolidated in `docs/plans/CHECK2HGI_MTL_OVERVIEW.md`.

## How to navigate

```
docs/studies/check2hgi/
├── README.md                     ← you are here
├── QUICK_REFERENCE.md            ← one-page cheat sheet (CLI, monitors, envs)
├── MASTER_PLAN.md                ← phase order, dependencies, wall-clock budget
├── CLAIMS_AND_HYPOTHESES.md      ← claim catalog (CH01..CHnn) with tests & status
├── ABLATIONS.md                  ← ablation matrix for the BRACIS paper
├── COORDINATOR.md                ← coordinator agent spec for this track (pointer)
├── HANDOFF.md                    ← session-close snapshot (transient state)
├── state.json                    ← runtime state (created lazily)
└── phases/
    ├── P-1_embeddings.md         ← generate vanilla check2HGI for FL + AL
    ├── P0_inputs.md              ← next_region label derivation pipeline
    ├── P1_code.md                ← MTLnet + runner parameterisation with TaskConfig
    ├── P2_baselines.md           ← single-task + legacy-MTL baselines on check2HGI
    ├── P3_mtl_next_region.md     ← the headline 2-task MTL run on FL + AL
    └── P4_ablations.md           ← head arch × MTL optimiser × primary-metric grid
```

## Paper thesis (draft)

> **Check-in-level contextual embeddings, when paired with a trajectory-aligned auxiliary task (next-region prediction), improve next-POI-category prediction over POI-level embeddings with POI-category-classification MTL, without negative transfer on either head.**

This thesis has three falsifiable components:

1. **Embedding:** check2HGI (check-in-level) beats HGI (POI-level) on next-POI-category prediction in the single-task setting.
2. **Task:** next-region auxiliary ≥ POI-category auxiliary when stacked with next-POI-category under MTL.
3. **No regression:** per-head metrics under MTL are ≥ per-head metrics under single-task training.

## Current phase

Look at `state.json` for the authoritative answer. If absent, we're in **P-1** (embedding generation) + **P1** (code parameterisation) executing in parallel.

## Out of scope for this branch

- Check2HGI encoder enrichment (temporal/spatial node features, hard negatives, multi-view contrastive) — the four phases of `docs/issues/CHECK2HGI_ENRICHMENT_PROPOSAL.md`. Deferred per `CHECK2HGI_MTL_OVERVIEW.md §3`: we need a vanilla baseline before any enrichment lift is attributable.
- Third-task extensions (next-time-gap, revisit-vs-explore). Scaffolded in `TaskConfig` but not active in any preset.
- Texas / California / Arizona / Georgia — only FL + AL for this track.

## Deliverable

Updated `docs/PAPER_FINDINGS.md` contributions section with check2HGI track results, plus an ablation table. Evidence pointers for every claim that moves to `status: confirmed`.
