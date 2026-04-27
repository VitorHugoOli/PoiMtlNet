# Check2HGI Study — Entry Point

**Status (2026-04-26):** **F48-H3-alt per-head LR recipe is the new champion candidate** — closes the F21c gap (CH18 promoted Tier B → Tier A). AL exceeds STL by +6.25 pp; AZ closes 75%; FL validates at scale (5-fold on all three states). Predecessor B3 (50ep + OneCycleLR) preserved as comparand. Read [`MTL_ARCHITECTURE_JOURNEY.md`](MTL_ARCHITECTURE_JOURNEY.md) for the end-to-end derivation, [`NORTH_STAR.md`](NORTH_STAR.md) for the recipe, [`research/F48_H3_PER_HEAD_LR_FINDINGS.md`](research/F48_H3_PER_HEAD_LR_FINDINGS.md) for the experimental detail.

**Status (2026-04-23, predecessor):** B3 identified as unified joint-task champion candidate via F2 diagnostic; validated 5-fold on AL + AZ + FL-1f (×2). Headline FL + CA + TX 5-fold pending. Doc cleanup complete.

**Scope:** Sibling to `docs/studies/fusion/`. Investigates whether check-in-level contextual embeddings (Check2HGI) with a next-region auxiliary task improve joint `{next_category, next_region}` prediction over HGI and over single-task training.

---

## Where to start (new agents, read in order)

1. **[`MTL_ARCHITECTURE_JOURNEY.md`](MTL_ARCHITECTURE_JOURNEY.md)** ⭐ **start here** — end-to-end narrative from initial design through F48-H3-alt; explains *why* the current recipe looks the way it does.
2. **[`NORTH_STAR.md`](NORTH_STAR.md)** — current champion candidate (F48-H3-alt per-head LR) + predecessor B3 + F27 scale-dependence flag.
3. **[`research/F48_H3_PER_HEAD_LR_FINDINGS.md`](research/F48_H3_PER_HEAD_LR_FINDINGS.md)** — H3 + H3-alt + FL scale validation (the recipe in detail).
4. **[`SESSION_HANDOFF_2026-04-26.md`](SESSION_HANDOFF_2026-04-26.md)** ⭐ **most recent** — one-minute summary of H3-alt session + pending work. Predecessor handoff at `SESSION_HANDOFF_2026-04-24.md`.
5. **[`PAPER_STRUCTURE.md`](PAPER_STRUCTURE.md)** — paper scope, baselines, STL-matching policy, FL region Markov caveat.
6. **[`FOLLOWUPS_TRACKER.md`](FOLLOWUPS_TRACKER.md)** — live work queue (F33 FL 5f + F34 CA 1f + F35 TX 1f are the next things to land).
7. **[`research/F21C_FINDINGS.md`](research/F21C_FINDINGS.md)** — paper-reshaping matched-head STL finding that triggered the CH18 attribution chain.
8. **[`research/F27_CATHEAD_FINDINGS.md`](research/F27_CATHEAD_FINDINGS.md)** — cat-head ablation + FL scale-dependence flag.
9. **[`review/2026-04-23_critical_review.md`](review/2026-04-23_critical_review.md)** — analytical state of the study as of 2026-04-23 (pre-F21c/F27).
10. **[`OBJECTIVES_STATUS_TABLE.md`](OBJECTIVES_STATUS_TABLE.md)** — one-page scorecard.
11. **[`results/RESULTS_TABLE.md`](results/RESULTS_TABLE.md)** — canonical per-state × per-method table.
12. **[`AGENT_CONTEXT.md`](AGENT_CONTEXT.md)** — long-form study context.
13. **[`SESSION_HANDOFF_2026-04-22.md`](SESSION_HANDOFF_2026-04-22.md)** — operational gotchas G1–G7 (MPS memory, caffeinate, num_workers=0, etc.). G8 in the newer handoff.

## What the paper claims (short)

- **CH16:** Check2HGI > HGI on next-category macro-F1. AL locked (+18.30 pp σ-clean). Cross-state replication pending.
- **Champion candidate (F48-H3-alt, 2026-04-26):** B3 architecture + per-head LR (`cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3`, all constant). MTL EXCEEDS matched-head STL on reg by +6.25 pp at AL; closes 75% of the gap on AZ; validates at FL scale (cat preserved, reg +6.7 pp over B3). Cat F1 stays within ~2 pp of B3 across all three states. CH18 promoted Tier B → Tier A.
- **Predecessor (B3, 2026-04-24):** `mtlnet_crossattn + static_weight(category_weight=0.75) + next_getnext_hard d=256, 8h, OneCycleLR max=3e-3, 50ep`. Beats baselines on cat F1 everywhere; beats Markov-1-region on AL+AZ (FL Markov-saturated). Strict MTL-over-STL on AZ cat F1 (+1.65 pp, p=0.0312). Preserved as the comparand against which H3-alt's contribution is measured.
- **Mechanism:** FL-hard-under-PCGrad gradient starvation + late-stage-handover rescue under unbalanced static weighting (`research/B5_FL_TASKWEIGHT.md`).

## Task pair

| Slot | Task | Classes | Primary metric |
|---|---|---|---|
| **task_a** | `next_category` | 7 | macro-F1 |
| **task_b** | `next_region` | 1,109 (AL) / 1,547 (AZ) / 4,702 (FL) / TBD (CA, TX) | Acc@10, MRR |

**Preset:** `CHECK2HGI_NEXT_REGION`.

## Baselines (see `PAPER_STRUCTURE.md §3` for detail)

- **next-cat:** POI-RGNN (external published), Markov-1-POI / Majority (simple floors), STL Check2HGI cat (matched), STL HGI cat (substrate ablation CH16).
- **next-reg:** Markov-1-region (simple floor), STL STAN (literature-aligned SOTA ceiling), STL GRU (secondary), STL GETNext-hard (matched-head — F21 pending). HMT-GRN / MGCL are concept-aligned references (different datasets, not directly comparable).

## Navigation

```
docs/studies/check2hgi/
├── README.md                              ← you are here
├── NORTH_STAR.md                          ← champion config decision
├── PAPER_STRUCTURE.md                     ← paper scope + baselines + tables
├── OBJECTIVES_STATUS_TABLE.md             ← one-page scorecard
├── FOLLOWUPS_TRACKER.md                   ← live work queue (owners, costs)
├── CLAIMS_AND_HYPOTHESES.md               ← reconciled claim catalog
├── CONCERNS.md                            ← acknowledged risks + resolutions
├── AGENT_CONTEXT.md                       ← long-form study-scoped briefing
├── SESSION_HANDOFF_2026-04-22.md          ← operational gotchas
├── issues/                                ← bug / design audits (MTL_PARAM_PARTITION_BUG, etc.)
├── research/                              ← paper-substantive notes (B3_*, B5_*, GETNEXT_*, STAN_*, POSITIONING_*, ATTRIBUTION_*, NASH_MTL_*, MTL_WITH_STAN_HEAD)
├── results/                               ← archived JSONs + tables
│   ├── RESULTS_TABLE.md                   ← per-state × per-method canonical table
│   ├── BASELINES_AND_BEST_MTL.md          ← legacy paper-comparison table (pre-B3 — kept for audit)
│   ├── B3_validation/                     ← B3 5f JSONs on AL, AZ
│   ├── B5/                                ← B5 hard-index JSONs on AL, AZ, FL-1f
│   ├── F2_fl_diagnostic/                  ← F2 4-phase JSONs on FL-1f
│   ├── P0/                                ← simple baselines (Markov k=1..9, Majority, Top-K)
│   ├── P1/                                ← STL region-head JSONs
│   ├── P1_5b/                             ← STL cat JSONs + CH16 HGI comparison
│   ├── P2/                                ← MTL arch × optim grid
│   ├── P5_bugfix/                         ← MTLoRA post-partition-bug reruns
│   └── P8_sota/                           ← MTL-STAN / TGSTAN / STA-Hyper / GETNext
├── review/                                ← dated critical reviews
└── archive/                               ← superseded docs (safe to ignore)
    ├── pre_b3_framing/                    ← MASTER_PLAN, KNOWLEDGE_SNAPSHOT, QUICK_REFERENCE, old HANDOFF, COORDINATOR, state.json, coordinator/
    ├── phases_original/                   ← original P0–P7 phase plans
    ├── research_pre_b3/                   ← pre-B3 research notes (MTL_ABLATION_PROTOCOL, SOTA_MTL_*, STRATEGIC_FRAMING, etc.)
    ├── 2026-04-20_status_reports/         ← pre-B5 status reports
    └── research_pre_b5/                   ← pre-B5 research notes (HYBRID_DECISION, EXECUTION_PLAN, CHAIN_FINDINGS)
```

## Out of scope

- POI-category classification from POI features alone (fusion study).
- HAVANA baseline (different task: semantic annotation, not sequential).
- Exact next-POI-id prediction (~11K classes) — outside MTLnet classification paradigm.
- Encoder enrichment (temporal/spatial/graph features) — deferred to follow-up paper.

## Sibling study

`docs/studies/fusion/` — POI-category classification on fused POI-level embeddings. Do not edit from this branch.
