# Check2HGI Study — Entry Point

**Status (2026-04-27):** Phase 1 of the substrate-comparison validation closed at AL + AZ. **Strong claim confirmed**: Check2HGI > HGI on **both** tasks under matched-head STL + matched MTL + linear-probe (3-leg framework, head-invariant across 4 probes). MTL B3 is **substrate-specific**: HGI substitution breaks reg by 30 pp. Per-visit context = ~72% of cat substrate gap. See [`research/SUBSTRATE_COMPARISON_FINDINGS.md`](research/SUBSTRATE_COMPARISON_FINDINGS.md) and [`SESSION_HANDOFF_2026-04-27.md`](SESSION_HANDOFF_2026-04-27.md). Phase 2 (FL + CA + TX) authorised and queued in [`PHASE2_TRACKER.md`](PHASE2_TRACKER.md).

**Prior status (2026-04-23):** B3 identified as unified joint-task champion candidate via F2 diagnostic; validated 5-fold on AL + AZ + FL-1f (×2). Doc cleanup complete.

**Scope:** Sibling to `docs/studies/fusion/`. Investigates whether check-in-level contextual embeddings (Check2HGI) with a next-region auxiliary task improve joint `{next_category, next_region}` prediction over HGI and over single-task training.

---

## Where to start (new agents, read in order)

1. **[`SESSION_HANDOFF_2026-04-27.md`](SESSION_HANDOFF_2026-04-27.md)** ⭐ **most recent** — Phase 1 substrate-validation outcome + paper-quality findings + Phase 2 launch instructions.
2. **[`research/SUBSTRATE_COMPARISON_FINDINGS.md`](research/SUBSTRATE_COMPARISON_FINDINGS.md)** — Phase-1 outcome matrix + paper-ready findings + sources appendix. Plan in [`research/SUBSTRATE_COMPARISON_PLAN.md`](research/SUBSTRATE_COMPARISON_PLAN.md); Phase-2 work queue in [`PHASE2_TRACKER.md`](PHASE2_TRACKER.md). Per-fold data: `results/{phase1_perfold,probe,paired_tests}/`.
3. **[`NORTH_STAR.md`](NORTH_STAR.md)** — committed champion MTL config + Phase-1 substrate-specific addendum.
4. **[`PAPER_STRUCTURE.md`](PAPER_STRUCTURE.md)** — paper scope, baselines, STL-matching policy, FL region Markov caveat.
5. **[`FOLLOWUPS_TRACKER.md`](FOLLOWUPS_TRACKER.md)** — live work queue (F36–F40 = Phase-2 tracker pointer).
6. **[`OBJECTIVES_STATUS_TABLE.md`](OBJECTIVES_STATUS_TABLE.md)** — one-page scorecard (v3).
7. **[`CLAIMS_AND_HYPOTHESES.md`](CLAIMS_AND_HYPOTHESES.md)** — claim catalog (CH15 reframed, CH16 head-invariant, CH18 MTL substrate-specific, CH19 mechanism).
8. **[`research/F21C_FINDINGS.md`](research/F21C_FINDINGS.md)** — paper-reshaping matched-head STL finding (STL > MTL on reg by 12–14 pp). Phase-1 update at top.
9. **[`research/F27_CATHEAD_FINDINGS.md`](research/F27_CATHEAD_FINDINGS.md)** — cat-head ablation + FL scale-dependence flag.
10. **[`SESSION_HANDOFF_2026-04-24.md`](SESSION_HANDOFF_2026-04-24.md)** + **[`SESSION_HANDOFF_2026-04-22.md`](SESSION_HANDOFF_2026-04-22.md)** — prior handoffs (post-F27 cat-head + operational gotchas G1–G8).
11. **[`AGENT_CONTEXT.md`](AGENT_CONTEXT.md)** — long-form study context.
12. **[`results/RESULTS_TABLE.md`](results/RESULTS_TABLE.md)** — canonical per-state × per-method table.
13. **[`review/2026-04-23_critical_review.md`](review/2026-04-23_critical_review.md)** — pre-Phase-1 analytical state.

## What the paper claims (short)

- **CH16 (cat substrate):** Check2HGI > HGI on next-category macro-F1, **head-invariant** at AL+AZ. 4-head probe + linear probe = 8 substrate-Δ measurements, all positive (+11.58 to +15.50 pp), all at maximum-significance n=5 paired Wilcoxon (p=0.0312, 5/5 folds positive each). FL/CA/TX replication queued in PHASE2_TRACKER.
- **CH15 reframed (reg substrate):** Under matched MTL reg head (`next_getnext_hard`), C2HGI ≥ HGI everywhere (AL tied within σ + TOST non-inferior; AZ +2.34 pp Acc@10 / +1.29 pp MRR, p=0.0312). The previous "HGI > C2HGI on reg under STAN" was head-coupled, not pure substrate.
- **CH18 (MTL substrate-specific):** MTL B3 with HGI substituted for Check2HGI breaks the joint signal — cat F1 −17 pp at both states, reg Acc@10_indist −30 pp at both states. The MTL win is interactional, not substrate-agnostic.
- **CH19 (mechanism):** Per-visit context accounts for ~72% of the cat substrate gap (matched-head STL); training signal residual = ~28%. POI-pooled C2HGI counterfactual confirms.
- **Champion (B3):** `mtlnet_crossattn + static_weight(category_weight=0.75) + next_gru (cat) + next_getnext_hard (reg) d=256, 8h`. See `NORTH_STAR.md`.

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
├── SESSION_HANDOFF_2026-04-24.md          ← post-F27 cat-head context
├── SESSION_HANDOFF_2026-04-27.md          ← Phase-1 substrate-validation handoff (latest)
├── PHASE2_TRACKER.md                      ← ⭐ FL+CA+TX substrate-comparison work queue
├── baselines/
│   ├── README.md                          ← baselines index + status board
│   ├── next_category/                     ← per-baseline docs + per-state JSONs + comparison.md
│   └── next_region/                       ← per-baseline docs + per-state JSONs + comparison.md
├── issues/                                ← bug / design audits (MTL_PARAM_PARTITION_BUG, etc.)
├── research/                              ← paper-substantive notes
│   ├── SUBSTRATE_COMPARISON_PLAN.md       ← ⭐ phase-gated 3-leg framework
│   ├── SUBSTRATE_COMPARISON_FINDINGS.md   ← ⭐ AL+AZ outcome matrix + paper-ready findings + sources appendix
│   └── (B3_*, B5_*, GETNEXT_*, STAN_*, POSITIONING_*, ATTRIBUTION_*, NASH_MTL_*, MTL_WITH_STAN_HEAD)
├── results/                               ← archived JSONs + tables
│   ├── RESULTS_TABLE.md                   ← per-state × per-method canonical table
│   ├── BASELINES_AND_BEST_MTL.md          ← legacy paper-comparison table (pre-B3 — kept for audit)
│   ├── phase1_perfold/                    ← ⭐ Phase-1 substrate-comparison per-fold JSONs (cat/reg STL + MTL counterfactual + C4 pooled)
│   ├── probe/                             ← ⭐ substrate linear-probe outputs (Leg I)
│   ├── paired_tests/                      ← ⭐ Wilcoxon + paired-t + TOST analyser outputs
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
