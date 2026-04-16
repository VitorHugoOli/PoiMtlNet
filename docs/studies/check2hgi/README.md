# Check2HGI Study — Entry Point

**Status:** P0 active (2026-04-15).
**Scope:** A **sibling** of `docs/studies/fusion/` that exists because check-in-level contextual embeddings (Check2HGI) and fine-grained sequence targets (next-POI + next-region) are a different scientific question from POI-level fused embeddings with a category-classification head. Fusion study stays where it is; this one runs alongside it.

> **If you're starting a fresh session:** read [`HANDOFF.md`](HANDOFF.md) first for the current data-availability snapshot and [`AGENT_CONTEXT.md`](AGENT_CONTEXT.md) for the study-scoped briefing.

## Why this study exists

Three motivations, in order:

1. **Check-in-level embeddings are wasted on category-level heads.** Check2HGI produces one vector per check-in event (~100k per state on Gowalla); two visits to the same POI get two different vectors. The category-classification task from the fusion study asks "what kind of place is this POI?" — a per-POI question that blurs the check-in granularity away. The natural downstream task pair for Check2HGI is **next-POI** (predict the exact next POI a user visits) + **next-region** (predict the region it falls in), both of which are sequence-level and both of which consume the contextual check-in embeddings at their full granularity.

2. **The next-POI literature is a different universe.** HMT-GRN (SIGIR '22), MGCL (Frontiers '24), LSTPM (AAAI '20), STAN (WWW '21), GETNext (SIGIR '22), ImNext (KBS '24), Graph-Flashback (KDD '22) all predict **exact next-POI ids** (10³–10⁶ classes) with **ranking metrics** (Acc@K / MRR / NDCG). The fusion study's baselines (HAVANA, PGC, POI-RGNN) are for POI-category classification — a different task, a different metric family, a different evaluation protocol. Mixing them confuses the contribution.

3. **Separation of concerns.** Fusion study and this one share infrastructure (the MTLnet codebase, the FoldCreator, the runner) but have different hypotheses, different baselines, different success criteria. Putting them in one study-tree mixes the claim catalog, the phase plan, and the state machine. Split cleanly.

## How to navigate

```
docs/studies/check2hgi/
├── README.md                     ← you are here
├── AGENT_CONTEXT.md              ← study-scoped briefing for subagents
├── QUICK_REFERENCE.md            ← one-page cheat sheet
├── MASTER_PLAN.md                ← phases, budget, decision gates
├── CLAIMS_AND_HYPOTHESES.md      ← authoritative claim catalog (CH01..CHnn)
├── KNOWLEDGE_SNAPSHOT.md         ← current baseline-knowledge state
├── COORDINATOR.md                ← coordinator agent spec (inherits fusion's workflow)
├── HANDOFF.md                    ← session handoff notes
├── state.json                    ← runtime state (coordinator-managed)
├── phases/
│   ├── P0_preparation.md         ← embeddings + labels + integrity + simple baselines + audits
│   ├── P1_single_task_baselines.md  ← next-POI + next-region single-task on Check2HGI (internal reference)
│   ├── P2_mtl_headline.md        ← 2-task MTL {next_POI, next_region} on Check2HGI
│   ├── P3_dual_stream.md         ← region-embedding as a second input stream
│   ├── P4_cross_attention.md     ← bidirectional cross-attention (gated on P3)
│   └── P5_ablations.md           ← heads, optimiser, sensitivity
├── coordinator/
│   ├── integrity_checks.md       ← check2HGI-specific schema + class-distribution checks
│   └── state_schema.md           ← state.json schema (shared with fusion)
├── issues/                       ← check2HGI-specific issue tracker
├── archive/
│   └── v1_wip_mixed_scope/       ← first-pass docs (scope-mixed, preserved for history)
└── results/                      ← JSON summaries archived per test
```

## Paper thesis (this study's contribution)

> **On check-in-level contextual embeddings (Check2HGI), hierarchical auxiliary supervision (next-region) improves next-POI ranking prediction over single-task training, at matched compute, with no per-head negative transfer. Characterised across two Gowalla state-level datasets (Alabama, Florida) with a decreasing-signal cardinality spectrum.**

**Standalone study** — no cross-engine comparison (no HGI, no fusion), no replication of prior-work numbers (no CBIC, no HAVANA/PGC/POI-RGNN).

Three falsifiable headline claims:

1. **MTL lift (CH01):** 2-task `{next_POI, next_region}` MTL > single-task next-POI on Check2HGI.
2. **No negative transfer (CH02):** per-head metrics under MTL ≥ single-task baselines.
3. **Dual-stream region input (CH03):** region embeddings as parallel input stream improve next-POI beyond MTL alone.

Baselines are internal (single-task Check2HGI is the reference for every MTL/dual-stream claim) + simple-baselines floor (majority, random, 1-step Markov, top-K popularity computed on our data in P0.5). External literature numbers (HMT-GRN, MGCL, etc.) go in an appendix table with scope caveat — direct numeric comparison is not valid (different datasets, different preprocessing).

## Current phase

`state.json` is authoritative. Before that file exists → we're in **P0 (Preparation)** — see [`phases/P0_preparation.md`](phases/P0_preparation.md).

## Out of scope for this study

- POI-category classification (the fusion study handles this).
- HAVANA / PGC / POI-RGNN baseline comparisons (different task).
- Expert-gating MTL architectures (CGC, MMoE, DSelect-K) — mentioned only if tight-for-time; the fusion study already covers these.
- Texas / California / Georgia states — AL + FL (+ AZ as triangulation) only.

## Workflow per phase

Same as fusion: plan → validate → execute → archive → analyze → update claims → decide. See [`COORDINATOR.md`](COORDINATOR.md).

## Deliverable

A `check2hgi_findings.md` section appended to the paper, plus an updated `docs/PAPER_FINDINGS.md` covering both tracks.

## Sibling study

`docs/studies/fusion/` — the POI-category track. Do not edit its artefacts from this branch; they belong to the fusion line of work.
