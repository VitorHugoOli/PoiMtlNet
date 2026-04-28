# Check2HGI Study — Entry Point

**Status (2026-04-27):** **Two complementary tracks both at paper-grade** — substrate-side and architecture-side both confirm and explain the MTL story:

1. **Phase 1 substrate validation** (AL+AZ, 5-leg study): **Check2HGI > HGI on both tasks** under matched-head STL + matched MTL + linear-probe (head-invariant across 4 probes). **MTL B3 is substrate-specific**: HGI substitution breaks reg by 30 pp. Per-visit context = ~72% of cat substrate gap. See [`research/SUBSTRATE_COMPARISON_FINDINGS.md`](research/SUBSTRATE_COMPARISON_FINDINGS.md). Phase 2 (FL+CA+TX) queued in [`PHASE2_TRACKER.md`](PHASE2_TRACKER.md).

2. **F49 architecture attribution** (AL+AZ+FL n=5): **The H3-alt reg lift is architecture-dominant** (AL +6.48 ± 2.4 pp from architecture alone, ~2.7σ). **Cat-supervision transfer is null on all 3 states** (≤|0.75| pp). Refutes legacy "+14.2 pp transfer" claim by ≥9σ on FL n=5 alone. Layer 2 methodological contribution: loss-side `task_weight=0` ablation is unsound under cross-attention MTL. See [`research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`](research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md).

The two findings sit at different layers (substrate vs architecture) and are mutually reinforcing: Phase 1 says *which embedding* is necessary; F49 says *what the architecture does* with it. Together they describe the MTL win as **interactional architecture × substrate**, not transfer-driven. Champion config (F48-H3-alt per-head LR) and recipe in [`NORTH_STAR.md`](NORTH_STAR.md); end-to-end derivation in [`MTL_ARCHITECTURE_JOURNEY.md`](MTL_ARCHITECTURE_JOURNEY.md); paper deliverables in [`PAPER_PREP_TRACKER.md`](PAPER_PREP_TRACKER.md). Most recent operational state: [`SESSION_HANDOFF_2026-04-27.md`](SESSION_HANDOFF_2026-04-27.md).

**Prior status (2026-04-23):** B3 identified as unified joint-task champion candidate via F2 diagnostic; validated 5-fold on AL + AZ + FL-1f (×2). Doc cleanup complete.

**Scope:** Sibling to `docs/studies/fusion/`. Investigates whether check-in-level contextual embeddings (Check2HGI) with a next-region auxiliary task improve joint `{next_category, next_region}` prediction over HGI and over single-task training.

---

## Where to start (new agents, read in order)

1. **[`SESSION_HANDOFF_2026-04-27.md`](SESSION_HANDOFF_2026-04-27.md)** ⭐ **most recent** — Phase-1 substrate-validation outcome + F49 attribution + paper-quality findings + Phase-2 + F37 launch instructions.
2. **[`research/SUBSTRATE_COMPARISON_FINDINGS.md`](research/SUBSTRATE_COMPARISON_FINDINGS.md)** ⭐ — Phase-1 substrate-side outcome matrix (CH16 head-invariant + CH15 reframed + CH18 MTL substrate-specific + C4 per-visit mechanism). Plan in [`research/SUBSTRATE_COMPARISON_PLAN.md`](research/SUBSTRATE_COMPARISON_PLAN.md); Phase-2 work queue in [`PHASE2_TRACKER.md`](PHASE2_TRACKER.md). Per-fold data: `results/{phase1_perfold,probe,paired_tests}/`.
3. **[`research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`](research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md)** ⭐ — F49 architecture-side attribution (3-way decomposition; cat-supervision transfer null on all 3 states n=5; AL architectural +6.48 pp ~2.7σ). Layer 2 methodological contribution. Plan in [`research/F49_LAMBDA0_DECOMPOSITION_GAP.md`](research/F49_LAMBDA0_DECOMPOSITION_GAP.md).
4. **[`PAPER_PREP_TRACKER.md`](PAPER_PREP_TRACKER.md)** — paper-deliverable tracker (committable claims, headline-blockers, doc-rewrites, risk register, submission checklist).
5. **[`MTL_ARCHITECTURE_JOURNEY.md`](MTL_ARCHITECTURE_JOURNEY.md)** — end-to-end narrative from initial design through F48-H3-alt + F49.
6. **[`NORTH_STAR.md`](NORTH_STAR.md)** — committed champion (F48-H3-alt per-head LR) + Phase-1 substrate-specific addendum + F49 attribution refinement.
7. **[`research/F48_H3_PER_HEAD_LR_FINDINGS.md`](research/F48_H3_PER_HEAD_LR_FINDINGS.md)** — H3 + H3-alt + FL scale validation (the recipe in detail).
8. **[`PAPER_STRUCTURE.md`](PAPER_STRUCTURE.md)** — paper scope, baselines, STL-matching policy, FL region Markov caveat.
9. **[`FOLLOWUPS_TRACKER.md`](FOLLOWUPS_TRACKER.md)** — per-experiment live work queue (F36–F40 Phase-2; F37 STL FL F21c on 4050; F49b/F49c done).
10. **[`OBJECTIVES_STATUS_TABLE.md`](OBJECTIVES_STATUS_TABLE.md)** — one-page scorecard (v5: includes Objective 3 substrate-specific + Objective 4 architecture-dominant).
11. **[`CLAIMS_AND_HYPOTHESES.md`](CLAIMS_AND_HYPOTHESES.md)** — claim catalog: CH15 reframed (head-coupled), CH16 head-invariant, CH18 MTL substrate-specific (Tier A), CH19 transfer-null + loss-side-ablation-unsound (Tier A).
12. **[`SESSION_HANDOFF_2026-04-26.md`](SESSION_HANDOFF_2026-04-26.md)** — H3-alt-discovery session.
13. **[`research/F21C_FINDINGS.md`](research/F21C_FINDINGS.md)** — paper-reshaping matched-head STL finding (STL > MTL on reg by 12–14 pp). Phase-1 update at top.
14. **[`research/F27_CATHEAD_FINDINGS.md`](research/F27_CATHEAD_FINDINGS.md)** — cat-head ablation + FL scale-dependence flag.
15. **[`SESSION_HANDOFF_2026-04-24.md`](SESSION_HANDOFF_2026-04-24.md)** + **[`SESSION_HANDOFF_2026-04-22.md`](SESSION_HANDOFF_2026-04-22.md)** — prior handoffs (post-F27 cat-head + operational gotchas G1–G8).
16. **[`AGENT_CONTEXT.md`](AGENT_CONTEXT.md)** — long-form study context.
17. **[`results/RESULTS_TABLE.md`](results/RESULTS_TABLE.md)** — canonical per-state × per-method table.
18. **[`review/2026-04-23_critical_review.md`](review/2026-04-23_critical_review.md)** — pre-Phase-1 / pre-F49 analytical state.
19. **[`review/2026-04-27_study_overview_and_forgotten_items.md`](review/2026-04-27_study_overview_and_forgotten_items.md)** — comprehensive overview + audit of forgotten P0–P7 items.

## What the paper claims (short)

- **CH16 (cat substrate):** Check2HGI > HGI on next-category macro-F1, **head-invariant** at AL+AZ. 4-head probe + linear probe = 8 substrate-Δ measurements, all positive (+11.58 to +15.50 pp), all at maximum-significance n=5 paired Wilcoxon (p=0.0312, 5/5 folds positive each). FL/CA/TX replication queued in PHASE2_TRACKER.
- **CH15 reframed (reg substrate):** Under matched MTL reg head (`next_getnext_hard`), C2HGI ≥ HGI everywhere (AL tied within σ + TOST non-inferior; AZ +2.34 pp Acc@10 / +1.29 pp MRR, p=0.0312). The previous "HGI > C2HGI on reg under STAN" was head-coupled, not pure substrate.
- **CH18 (MTL substrate-specific, Tier A):** MTL B3 with HGI substituted for Check2HGI breaks the joint signal — cat F1 −17 pp at both states, reg Acc@10_indist −30 pp at both states. The MTL win is interactional, not substrate-agnostic.
- **CH19 (architecture-side, Tier A):** Cat-supervision transfer ≤ |0.75| pp on AL/AZ/FL n=5 — refutes legacy "+14.2 pp transfer at FL" claim by ≥9σ on FL alone. The H3-alt reg lift on AL is architecture-dominant (+6.48 pp from architecture alone). Layer 2 methodological contribution: loss-side `task_weight=0` ablation is unsound under cross-attn MTL (silenced encoder co-adapts via attention K/V); encoder-frozen isolation is the only clean architectural decomposition. Per-visit-context mechanism (Phase-1 C4): ~72% of cat substrate gap, ~28% from training signal residual.
- **Champion candidate (F48-H3-alt, 2026-04-26):** B3 architecture + per-head LR (`cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3`, all constant). MTL EXCEEDS matched-head STL on reg by +6.25 pp at AL; closes 75% of the gap on AZ; validates at FL scale (cat preserved, reg +6.7 pp over B3). F49 sharpens the *cause* to architecture, not transfer.
- **Predecessor (B3, 2026-04-24):** `mtlnet_crossattn + static_weight(category_weight=0.75) + next_getnext_hard d=256, 8h, OneCycleLR max=3e-3, 50ep`. Beats baselines on cat F1 everywhere; beats Markov-1-region on AL+AZ (FL Markov-saturated). Strict MTL-over-STL on AZ cat F1 (+1.65 pp, p=0.0312). Preserved as the comparand against which H3-alt's contribution is measured.
- **Mechanism:** FL-hard-under-PCGrad gradient starvation + late-stage-handover rescue under unbalanced static weighting (`research/B5_FL_TASKWEIGHT.md`); F49 sharpens reg attribution to **architecture × substrate interaction**, not transfer.

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
├── PAPER_PREP_TRACKER.md                  ← paper-deliverable tracker (NEW 2026-04-27)
├── PHASE2_TRACKER.md                      ← Phase-2 substrate replication (NEW 2026-04-27)
├── OBJECTIVES_STATUS_TABLE.md             ← one-page scorecard (v5)
├── FOLLOWUPS_TRACKER.md                   ← live work queue (owners, costs)
├── CLAIMS_AND_HYPOTHESES.md               ← reconciled claim catalog (CH01..CH19)
├── CONCERNS.md                            ← acknowledged risks + resolutions (C12 resolved)
├── AGENT_CONTEXT.md                       ← long-form study-scoped briefing
├── MTL_ARCHITECTURE_JOURNEY.md            ← end-to-end derivation
├── SESSION_HANDOFF_2026-04-{22..27}.md    ← chronology + operational gotchas
├── baselines/                             ← STAN, POI-RGNN, MHA+PE, REHDM faithful ports
│   ├── README.md
│   ├── next_category/{poi_rgnn,mha_pe,comparison}.md + results/{state}.json
│   └── next_region/{stan,rehdm,comparison}.md + results/{state}.json
├── issues/                                ← bug / design audits
├── research/                              ← paper-substantive notes
│   ├── SUBSTRATE_COMPARISON_PLAN.md       ← Phase-1 plan
│   ├── SUBSTRATE_COMPARISON_FINDINGS.md   ← Phase-1 verdict (NEW 2026-04-27)
│   ├── F49_LAMBDA0_DECOMPOSITION_GAP.md   ← F49 plan (NEW 2026-04-27)
│   ├── F49_LAMBDA0_DECOMPOSITION_RESULTS.md ← F49 results (NEW 2026-04-27)
│   ├── F48_H3_PER_HEAD_LR_FINDINGS.md     ← H3-alt champion derivation
│   ├── F21C_FINDINGS.md, F27_CATHEAD_FINDINGS.md, B3_*, B5_*, GETNEXT_*, STAN_*, POSITIONING_VS_HMT_GRN, ATTRIBUTION_*, NASH_MTL_*, MTL_WITH_STAN_HEAD
├── results/                               ← archived JSONs + tables
│   ├── RESULTS_TABLE.md                   ← per-state × per-method canonical table
│   ├── BASELINES_AND_BEST_MTL.md          ← legacy paper-comparison table (pre-B3 — kept for audit)
│   ├── B3_validation/, B5/, F2_fl_diagnostic/, F27_cathead_sweep/, F27_validation/, F41_preencoder/
│   ├── P0/, P1/, P1_5b/, P2/, P5_bugfix/, P8_sota/
│   ├── phase1_perfold/, probe/, paired_tests/, baselines/  ← Phase-1 substrate data (NEW)
│   └── SCALE_CURVE.md
├── review/                                ← dated critical reviews + overview
│   ├── 2026-04-23_critical_review.md
│   └── 2026-04-27_study_overview_and_forgotten_items.md  ← P0-P7 audit + forgotten items (NEW)
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
