# Check2HGI Study — Master Plan

**Goal:** validate the three Tier-A claims (CH01, CH02, CH03) with evidence strong enough for the BRACIS paper, plus enough supporting mechanism / ablation claims (Tier B–D) to defend the contribution against reviewer scrutiny.

**Target venue:** BRACIS 2026. Paper deadline context inherited from the fusion study.

## Phase overview

| Phase | Purpose | Claims addressed | Wall-clock (est.) | Requires |
|---|---|---|---|---|
| **P0** | Embeddings + labels + integrity + **simple baselines** + shortcut/leakage audits | CH04, CH05, CH14, CH15 | ~3h | Check2HGI embeddings for FL + AL already generated; next_poi loader code-deltas (P0.3) |
| **P1** | Single-task references: next-POI and next-region on Check2HGI | CH04 floor check, CH05, CH06 | ~3h (AL + FL × 2 single-task runs × 5f × 50ep × 1 seed — the primary seed; multi-seed lives in P2) | P0 complete |
| **P2** | **HEADLINE:** 2-task MTL `{next_poi, next_region}` on Check2HGI vs single-task next-POI, **multi-seed** | **CH01**, CH02 | **~12h** (AL + FL × 3 seeds × 5 folds × 50ep each = n=15 paired samples; required for statistical power at 2pp effects) | P1 complete |
| **P3** | Dual-stream input (region emb as parallel stream), **multi-seed** | **CH03**, CH08 | ~9h (AL + FL × 3 seeds × 5 folds) | P2 complete; CH01 at least `partial` |
| **P4** | Cross-attention, **gated on P3 ≥ 2pp on FL** | CH07 | ~6h (new arch implementation ~1 day + 2 runs × 3 seeds) | P3 complete; CH03 shows ≥ 2pp FL lift |
| **P5** | Sensitivity: head arch, MTL optimiser (AL + FL), variance analysis | CH09, CH10, CH11 | ~5h | P2 complete |

**Compute budget:** ~38h wall-clock sequential (P0 through P5 assuming P4 runs). ~32h if P4 skipped. P2 tripled vs v1 because **multi-seed (n=15) became the default** for the headline per review-agent finding: n=5 Wilcoxon has near-zero power at 2pp effect sizes, so the whole headline comparison was statistically meaningless at single-seed.

**Compute-cost mitigations considered:**
- Dropping P5.1 (head architecture) to pay for the multi-seed bump → **kept** but moved to post-P2 after headlines land.
- Running CH10 (optimiser ablation) on AL only → **rejected** per review-agent finding #4.3 — optimiser behaviour is most likely to be imbalance-sensitive; must include FL.
- Reducing to 2 seeds (n=10) → minimum-detectable-effect drops ~30% but n=10 paired Wilcoxon still has weak power. Stay at n=15.

## Critical-path ordering

```
P0 (integrity + audit) ─► P1 (single-task) ─► P2 (MTL headline) ─┬─► P3 (dual-stream) ─► P4 (cross-attn, gated)
                                                                 └─► P5 (ablations)
```

- **P3 and P5 are independent after P2**; can run in parallel on separate machines.
- **P4 gated on P3's CH06 result**. If CH06 shows < 2pp lift on FL, P4 is documented as "future work" and does not run.

## Phase gates

Move to the next phase only when:

- **Out of P0:** Check2HGI embeddings + next_region + next_poi labels exist for both FL and AL; `coordinator/integrity_checks.md` passes; simple baselines (P0.5) computed + archived; CH14 (shortcut audit, both code inspection AND unconditional fclass-shuffle ablation on AL) + CH15 (leakage audit) resolved.
- **Out of P1:** single-task next-POI and next-region references exist; **CH04 floor check passed on both** (learned Acc@10 ≥ 2× best simple baseline); CH05, CH06 documented.
- **Out of P2:** CH01 + CH02 resolved via n=15 paired Wilcoxon. Headline paper-table entries drafted. CH11 (seed variance) computed from the n=15 runs.
- **Out of P3:** CH03, CH08 resolved. P4 gate evaluated (CH03 ≥ 2pp on FL?).
- **Out of P4 (if run):** CH07 resolved.
- **Out of P5:** CH09, CH10 resolved.

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
| CH14 audit shows Check2HGI preprocessing has a shortcut | Invalidates CH01 headline claim | P0.6 runs fclass-shuffle ablation **unconditionally** (not just if inspection suggests); if shortcut present, mitigation becomes part of the paper contribution ("shortcut-robust representation") |
| FL next_region NashMTL over-weights (22.5% majority region) | CH02 negative on FL | Enable `--use-class-weights` for FL runs; document as FL-specific mitigation |
| Check2HGI embeddings undertrained at 500 epochs (loss still decreasing) | CH01 still confounded if single-task next-POI is also poor | **No longer a blocker** — we don't compare against HGI. Train Check2HGI to a committed budget (default 500 epochs, extend if P1 single-task Acc@10 is below simple-baseline floor). Document the budget in the paper. |
| Seed variance ≥ 2pp on headline | CH01's "+≥2pp lift" claim non-significant | Multi-seed is default (n=15); CH11 reports variance bound. Effects below 1.5σ reported as `partial` or `inconclusive`. |
| n_region mismatch between train/val/test folds | Model output under-sized for val labels | Already handled in `scripts/train.py::_run_mtl_check2hgi` (takes max across all folds); verify in P0 |
| OOD POI labels in val folds mechanically = Acc@0 regardless of model | Cross-state comparisons confounded by OOD-rate, not region cardinality | Report BOTH raw and OOD-restricted Acc@K per run. CH06 formalises. |
| Transductive embedding training on 100% check-ins before user-hold-out splits | CH01 inflated by upstream leakage | CH15 audits magnitude by retraining Check2HGI with one fold's users held out, comparing Acc@10. Run once as upper bound; if gap < 1pp, declare bounded. |

## Exit criteria

Branch merge to main when:

- All Tier-A claims (CH01, CH02, CH03) resolved with evidence pointers.
- At least 2 of {CH06, CH07, CH11} in Tier C resolved.
- All Tier-D ablations (CH08, CH09, CH10) run.
- Legacy (fusion) tests remain green — verified by `pytest tests/ -q --ignore=tests/test_integration`.
- `docs/PAPER_FINDINGS.md` gets a check2HGI section drafted (sibling to the fusion section).

If Tier-A CH02 refutes, the branch still merges — the infrastructure + honest empirical result stand on their own. Paper thesis reframes around CH01 only.
