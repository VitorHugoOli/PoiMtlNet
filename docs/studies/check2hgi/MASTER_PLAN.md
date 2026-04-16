# Check2HGI Study — Master Plan

**Goal:** validate the three Tier-A claims (CH01, CH02, CH03) with evidence strong enough for the BRACIS paper, plus enough supporting mechanism / ablation claims (Tier B–D) to defend the contribution against reviewer scrutiny.

**Target venue:** BRACIS 2026. Paper deadline context inherited from the fusion study.

## Phase overview

| Phase | Purpose | Claims addressed | Wall-clock (est.) | Requires |
|---|---|---|---|---|
| **P0** | Embeddings, labels, integrity checks, fclass-shortcut audit | CH04, CH05, CH14 | ~2h (integrity + audit) | Check2HGI embeddings for FL + AL already generated |
| **P1** | Single-task baselines: next-POI on {HGI, Check2HGI}, next-region on Check2HGI | CH01, CH04, CH05 | ~4h (AL + FL × 3 single-task runs × 5f × 50ep) | P0 complete |
| **P2** | 2-task MTL headline: `{next_poi, next_region}` on Check2HGI vs single-task next-POI | CH02, CH03 | ~5h (AL + FL × MTL × 5f × 50ep) | P1 complete; CH01 resolved |
| **P3** | Dual-stream input (region emb as parallel stream) | CH06, CH11 | ~3h (AL + FL) | P2 complete; CH02 at least `partial` |
| **P4** | Cross-attention (new architecture), **gated on P3 ≥ 2pp on FL** | CH07 | ~6h (arch implementation ~1 day + 2 runs) | P3 complete; CH06 shows ≥ 2pp FL lift |
| **P5** | Ablations: head architecture, MTL optimiser, seed variance | CH08, CH09, CH10 | ~4h | P2 complete |

**Compute budget:** ~24h wall-clock sequential (P0 through P5 assuming P4 runs). ~18h if P4 skipped. Per-fold training time on MPS with current model is ~22 min for 5f × 50ep MTL on AL; FL is ~4× larger.

## Critical-path ordering

```
P0 (integrity + audit) ─► P1 (single-task) ─► P2 (MTL headline) ─┬─► P3 (dual-stream) ─► P4 (cross-attn, gated)
                                                                 └─► P5 (ablations)
```

- **P3 and P5 are independent after P2**; can run in parallel on separate machines.
- **P4 gated on P3's CH06 result**. If CH06 shows < 2pp lift on FL, P4 is documented as "future work" and does not run.

## Phase gates

Move to the next phase only when:

- **Out of P0:** Check2HGI embeddings + next_region labels exist for both FL and AL; `coordinator/integrity_checks.md` passes; CH14 audit resolved (shortcut present or not; if present, mitigation plan documented).
- **Out of P1:** CH01, CH04, CH05 have status ∈ {confirmed, refuted, partial}; results archived under `results/P1/<test_id>/`.
- **Out of P2:** CH02, CH03 resolved. Headline paper-table entries drafted.
- **Out of P3:** CH06 and CH11 resolved. P4 gate evaluated.
- **Out of P4 (if run):** CH07 resolved.
- **Out of P5:** CH08, CH09, CH10 resolved.

## Datasets

Same as fusion:

- **Alabama** (primary, ~22 min/run for 5f × 50ep MTL).
- **Florida** (replication + high-cardinality test — ~4× the train/val set of AL).
- **Arizona** (optional triangulation for CH11 probe; already has embeddings + labels).

## Budget gates & cuts

Per the `docs/studies/check2hgi/archive/v1_wip_mixed_scope/CRITICAL_REVIEW.md` audit, the following are **deferred** to keep scope BRACIS-sized:

- Full 5×20 arch × optim grid (fusion P1 port) — we run only NashMTL (default) + 2 comparators in P5. ~6h saved.
- Expert-gating MTL architectures (CGC / MMoE / DSelect-K) — require task_set parameterisation for 4 variants; covered adequately by fusion study. ~4h saved.
- Frozen-backbone head-swap co-adaptation probe — mechanism-interesting but not paper-headline. ~2h saved.
- next_time_gap third auxiliary task — scaffolded in the `TaskConfig` registry but not active in any preset.

If the FL run shows surprising results (e.g. CH02 fails on FL but passes on AL), reserve +6h for investigation before committing to paper.

## Seeds & statistical testing

- Primary runs: seed 42.
- CH10 (seed variance) runs the P2 champion at seeds {123, 2024} additionally — n=15 paired samples total.
- Paired statistical tests (Wilcoxon signed-rank) for headline comparisons between Check2HGI and HGI engines, and between MTL and single-task. Require fold-aligned runs (frozen folds — see fusion's `P0_preparation.md §Fold freezing`).

## Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CH14 audit shows Check2HGI inherits the HGI fclass shortcut | Invalidates CH01 headline comparison | If shortcut present: run shuffle-ablation on both engines, report the ratio not the absolute number, reframe thesis around "shortcut-robust representation" |
| FL next_region NashMTL over-weights (22.5% majority region) | CH03 negative on FL | Enable `--use-class-weights` for FL runs; document as FL-specific mitigation |
| Check2HGI embeddings undertrained at 500 epochs (loss still decreasing) | CH01 confounded by training budget | Option 1 from `archive/v1_wip_mixed_scope/TRAINING_BUDGET_DECISION.md`: match FLOPs vs HGI before P1 |
| Seed variance ≥ 2pp on headline | CH02's "+≥2pp lift" claim non-statistically-significant | Pre-register: effects < 1.5σ reported as `partial` or `inconclusive`, not `confirmed` |
| n_region mismatch between train/val/test folds | Model output under-sized for val labels | Already handled in `scripts/train.py::_run_mtl_check2hgi` (takes max across all folds); verify in P0 |

## Exit criteria

Branch merge to main when:

- All Tier-A claims (CH01, CH02, CH03) resolved with evidence pointers.
- At least 2 of {CH06, CH07, CH11} in Tier C resolved.
- All Tier-D ablations (CH08, CH09, CH10) run.
- Legacy (fusion) tests remain green — verified by `pytest tests/ -q --ignore=tests/test_integration`.
- `docs/PAPER_FINDINGS.md` gets a check2HGI section drafted (sibling to the fusion section).

If Tier-A CH02 refutes, the branch still merges — the infrastructure + honest empirical result stand on their own. Paper thesis reframes around CH01 only.
