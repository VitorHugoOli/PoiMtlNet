# Check2HGI Study — Entry Point

**Status:** P0 complete, P1 ready (2026-04-16).
**Scope:** Sibling to `docs/studies/fusion/`. Investigates whether check-in-level contextual embeddings (Check2HGI) with a next-region auxiliary task improve next-POI-category prediction over single-task training.

> **If you're starting a fresh session:** read [`HANDOFF.md`](HANDOFF.md) first, then [`AGENT_CONTEXT.md`](AGENT_CONTEXT.md).

## Task pair

| Slot | Task | Classes | Primary metric |
|---|---|---|---|
| **task_a** | `next_category` | 7 | macro-F1 |
| **task_b** | `next_region` | ~1,109 (AL) / ~4,703 (FL) | Acc@K, MRR |

**Preset:** `CHECK2HGI_NEXT_REGION` — both heads sequential, shared X tensor `[B, 9, 64]`.

## Paper thesis

> **Adding next-region as a hierarchical auxiliary task to next-POI-category prediction on check-in-level embeddings (Check2HGI) improves next-category macro-F1 over single-task training, without negative transfer, on Gowalla state-level data.**

## Baselines

**For next_category (7 classes):**
- **POI-RGNN** (Capanema et al. 2019): next-category on Gowalla FL/CA/TX, macro-F1 = 31.8–34.5%
- **MHA+PE** (Zeng et al. 2019): next-category on Gowalla global, macro-F1 = 26.9%
- Our Check2HGI single-task on AL: **38.67% F1** (already above POI-RGNN range)

**For next_region (~1K classes):**
- **HMT-GRN** (SIGIR '22): hierarchical region auxiliary, GRU-based region head
- **MGCL** (Frontiers '24): multi-granularity contrastive with region + category heads
- Direct numeric comparison limited (different datasets); concept-alignment is the contribution

**Simple-baseline floor (from P0.5):**
- AL next_category: majority 34.2%, Markov 31.7%
- FL next_category: majority 24.7%, Markov 37.2%
- AL next_region: Markov Acc@10 = 21.3%
- FL next_region: Markov Acc@10 = 45.9%

## Phases

| Phase | What | Claims | Duration |
|---|---|---|---|
| **P0** | ✅ Integrity + baselines + audits | CH08, CH09 | done |
| **P1** | Region-head validation + head ablation (single-task) | CH04, CH05 | ~2h |
| **P2** | Parameterize all MTL architectures + full arch×optim ablation | CH06 | ~1 day |
| **P3** | MTL headline: champion config, multi-seed n=15 | **CH01**, CH02, CH07 | ~4h |
| **P4** | Dual-stream: region_embedding as parallel input | CH03 | ~3h |
| **P5** | Cross-attention (gated on P4) | CH10 | ~6h (gated) |

## Navigation

```
docs/studies/check2hgi/
├── README.md                     ← you are here
├── AGENT_CONTEXT.md              ← study-scoped briefing
├── QUICK_REFERENCE.md            ← one-page cheat sheet
├── MASTER_PLAN.md                ← phases, budget, gates
├── CLAIMS_AND_HYPOTHESES.md      ← claim catalog (CH01..CH11)
├── KNOWLEDGE_SNAPSHOT.md         ← baseline knowledge state
├── COORDINATOR.md                ← agent spec
├── HANDOFF.md                    ← session handoff
├── state.json                    ← runtime state
├── phases/                       ← per-phase execution plans
├── coordinator/                  ← integrity checks + state schema
├── results/                      ← archived per-test summaries
└── archive/                      ← prior scope iterations
```

## Out of scope

- POI-category classification (fusion study).
- HAVANA / PGC baselines (different task).
- Exact next-POI-id prediction (~11K classes) — was considered, doesn't match the MTLnet architecture's classification paradigm.
- Encoder enrichment (temporal/spatial features, hard negatives) — deferred.

## Sibling study

`docs/studies/fusion/` — POI-category classification on fused POI-level embeddings. Do not edit from this branch.
