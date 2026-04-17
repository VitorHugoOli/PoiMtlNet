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

> **On Check2HGI check-in-level embeddings, the two tasks `{next_category, next_region}` help each other: joint MTL training improves *both* next-category macro-F1 AND next-region Acc@10 over their respective single-task baselines, without negative transfer, on Gowalla state-level data. The mechanism is per-task input modality (check-in emb → category head, region emb → region head) coupled through a shared MTL backbone.**

The thesis is **bidirectional** — a one-sided lift (category improved, region degraded, or vice versa) fails the thesis. See CH01 + CH02 in the claim catalog.

## Baselines

**For next_category (7 classes):**
- **POI-RGNN** (Capanema et al. 2019): next-category on Gowalla FL/CA/TX, macro-F1 = 31.8–34.5%
- **MHA+PE** (Zeng et al. 2019): next-category on Gowalla global, macro-F1 = 26.9%
- Our Check2HGI single-task on AL: **38.67% F1** (already above POI-RGNN range)

**For next_region (~1K classes):**
- **HMT-GRN** (SIGIR '22): hierarchical region auxiliary, GRU-based region head
- **MGCL** (Frontiers '24): multi-granularity contrastive with region + category heads
- Direct numeric comparison limited (different datasets); concept-alignment is the contribution
- Our Check2HGI single-task on AL (region-emb input, `next_gru` default, **5f × 50ep**): **56.94% ± 4.01 Acc@10** (1.21× Markov-1-region)
- Our Check2HGI single-task on FL (region-emb input, `next_gru` default, **1f × 30ep**): **65.91% Acc@10** (only 1.013× Markov-1-region — dense-data regime, very tight margin)

**Simple-baseline floor (from P0.5, updated 2026-04-16):**
- AL next_category: majority 34.2%, Markov 31.7%
- FL next_category: majority 24.7%, Markov 37.2%
- AL next_region: **Markov-1-region Acc@10 = 47.01%** (the POI-level `markov_1step` was a degenerate baseline; see CH04 notes)
- FL next_region: **Markov-1-region Acc@10 = 65.05%**

## Phases

| Phase | What | Claims | Duration |
|---|---|---|---|
| **P0** | ✅ Integrity + baselines (with corrected region-level Markov) + audits | floor | done |
| **P1** | Region-head validation + head ablation × {check-in, region, concat} input | CH04, CH05 | ~3h |
| **P2** | Parameterize all MTL architectures + full arch×optim ablation under per-task modality | CH06 | ~1 day |
| **P3** | MTL headline: champion config, multi-seed n=15, **bidirectional per-head comparison** | **CH01, CH02**, CH07 | ~4h |
| **P4** | **Per-task input modality**: 4-way comparison {per_task, concat, shared_checkin, shared_region} | **CH03**, CH08 | ~4h |
| **P5** | Cross-attention between task-specific encoders (gated on P4) | CH09 | ~6h (gated) |
| **P6** | Check2HGI encoder enrichment (research-gated) | CH12, CH13 | ~2 days |

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
