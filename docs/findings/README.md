# docs/findings/ — Paper-supporting per-experiment findings (F-trail)

Closed evidence supporting the check2hgi BRACIS 2026 paper. Each file documents a single experiment's findings (B-series early experiments, F-series ablations, F50-tier decompositions, B5/B7/B9/F21c/F27/F37/F40/F41/F44–F48/F49/F50/F51/...).

This is **read-only history**. Active research tracks live at [`docs/studies/`](../studies/).

## `findings/` vs `results/`

> **`findings/`** = narrative + analysis with conclusions (.md). What we learned.
> **`results/`** = canonical numerical artifacts (CSV/JSON tables, RESULTS_TABLE.md, paired-test JSONs). What the numbers were.

Some files in `findings/` have "RESULTS" in the name (e.g., `F50_RESULTS_TABLE.md`, `F49_LAMBDA0_DECOMPOSITION_RESULTS.md`) — they are still narrative findings *about* results, not the canonical numerical source. The canonical source is [`../results/RESULTS_TABLE.md §0`](../results/RESULTS_TABLE.md).

## How findings relate to the paper narrative

The full F-trail narrative — how each finding chained into the next, decision points, deferrals — lives in [`../MTL_ARCHITECTURE_JOURNEY.md`](../MTL_ARCHITECTURE_JOURNEY.md). That doc is the **supplementary-material guide** to this folder.

The headline 2-line summary of the paper-blocking findings: substrate-task-asymmetry first (per-visit context provides +14–29 pp on cat-F1 across 5 states), classic MTL tradeoff second (small cat lift via cross-attention, ~7–17 pp cost on next-region Acc@10).

## Folder layout

```
findings/
├── README.md              ← you are here
├── F##_*.md, F##_*.json   ← 60+ per-experiment findings
├── B##_*.md               ← B-series findings (B3/B5/B7/B9)
├── ATTRIBUTION_*, AZ_PERVISIT_WILCOXON.json, ARCH_DELTA_WILCOXON.json, etc.
├── archive/F50/           ← further-archived F50-tier findings
└── figs/                  ← finding figures (PNG)
```

## Index by topic (high-level)

(For the full chronological narrative, read [`../MTL_ARCHITECTURE_JOURNEY.md`](../MTL_ARCHITECTURE_JOURNEY.md) and [`../CHANGELOG.md`](../CHANGELOG.md).)

| Topic | Key files |
|---|---|
| **Substrate comparison** (canonical Check2HGI vs HGI vs alternatives) | `SUBSTRATE_COMPARISON_FINDINGS.md`, `B3_AZ_WILCOXON_VS_STL.md` |
| **B5 series** (FL hard-MTL, scaling, task-weight, macro analysis, hard-vs-soft inference, probe entropy) | `B5_*` (8 files) |
| **B7/B9** (ALiBi GetNext, STL/STAN swap, AZ/FL) | `B7_ALIBI_GETNEXT_FINDINGS.md`, `B9_STL_STAN_SWAP_AZ_FL.md` |
| **F21c** (STL GetNext hard) | `F21C_FINDINGS.md` |
| **F27** (cathead sweep + validation) | `F27_CATHEAD_FINDINGS.md` |
| **F37** (FL P1+P2) | `F37_FL_RESULTS.md` |
| **F38** (checkpoint selection) | `F38_CHECKPOINT_SELECTION.md` |
| **F40** (scheduled handover) | `F40_SCHEDULED_HANDOVER_FINDINGS.md`, `F40_F47_REPLICATION.md` |
| **F41** (pre-encoder) | `F41_PREENCODER_FINDINGS.md` |
| **F44–F48** (LR regime, warmup, per-head LR, epoch budget, cat-zero) | `F44_F48_*`, `F48_H2_*`, `F48_H3_*` |
| **F49** (lambda-0 decomposition) | `F49_LAMBDA0_DECOMPOSITION_GAP.md`, `F49_LAMBDA0_DECOMPOSITION_RESULTS.md` |
| **F50 tier** (B2/F52/F65/F53, D5 encoder trajectory, T1 results synthesis, T3 audit/dynamics, T4 leakage audits, delta-m leak-free) | `F50_*` (~15 files in this folder + further-archived in `archive/F50/`) |
| **F51** (multi-seed Wilcoxon, tier2 capacity) | `F51_MULTI_SEED_FINDINGS.md`, `F51_TIER2_CAPACITY_FINDINGS.md` |
| **Methodological side-findings** (PCGrad vs static, REHDM STL diagnosis, MTL with STAN head, MTL flaws and fixes) | `ATTRIBUTION_PCGRAD_VS_STATIC.md`, `REHDM_STL_DIAGNOSIS_20260501.md`, `MTL_WITH_STAN_HEAD.md`, `MTL_FLAWS_AND_FIXES.md` |
| **Wilcoxon JSONs** (paper-supporting paired tests) | `*_WILCOXON.json` files |
| **Design-D heterograph** | `DESIGN_D_HETEROGRAPH.md` |
| **STAN three-way** | `STAN_THREE_WAY_COMPARISON.md`, `FAITHFUL_STAN_FINDINGS.md` |
| **Positioning vs HMT-GRN** | `POSITIONING_VS_HMT_GRN.md` |

(Some filenames in the list above may be slightly inexact — read the actual `.md` to confirm before citing.)

## Reorg history

The F-trail moved here from `docs/studies/check2hgi/research/` on 2026-05-14. Full record: [`../archive/MERGE_REORG_PLAN_2026-05-14.md`](../archive/MERGE_REORG_PLAN_2026-05-14.md).
