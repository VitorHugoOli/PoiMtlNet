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
│   ├── P0_preparation.md         ← embeddings + labels + integrity checks
│   ├── P1_single_task_baselines.md  ← next-POI and next-region single-task on {HGI, check2HGI}
│   ├── P2_mtl_headline.md        ← 2-task MTL {next_POI, next_region} on check2HGI
│   ├── P3_dual_stream.md         ← Option A: region-embedding as a second input stream
│   ├── P4_cross_attention.md     ← Option C: bidirectional cross-attention (gated on P3)
│   └── P5_ablations.md           ← heads, optimiser, seed sweep
├── coordinator/
│   ├── integrity_checks.md       ← check2HGI-specific schema + class-distribution checks
│   └── state_schema.md           ← state.json schema (shared with fusion)
├── issues/                       ← check2HGI-specific issue tracker
├── archive/
│   └── v1_wip_mixed_scope/       ← first-pass docs (scope-mixed, preserved for history)
└── results/                      ← JSON summaries archived per test
```

## Paper thesis (this study's contribution)

> **Check-in-level contextual embeddings (Check2HGI), combined with a hierarchical auxiliary task (next-region), improve next-POI prediction over POI-level embeddings (HGI) with single-task next-POI training, at matched compute, without negative transfer on either head.**

Three falsifiable components:

1. **Embedding:** Check2HGI > HGI on single-task next-POI (Acc@10, MRR) on AL + FL.
2. **Task:** MTL `{next_POI, next_region}` ≥ single-task next-POI at matched budget.
3. **No regression:** per-head metrics under MTL are ≥ single-task baselines.

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
