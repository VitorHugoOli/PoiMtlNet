# Check2HGI Track — Master Plan

**Goal:** validate the three Tier-A claims (CH01, CH02, CH03 — see `CLAIMS_AND_HYPOTHESES.md`) with evidence strong enough for the BRACIS paper.

## Phase overview

| Phase | Purpose | Claims addressed | Wall-clock (est.) | Parallelisable with |
|---|---|---|---|---|
| **P-1** | Generate vanilla check2HGI embeddings for FL + AL | (prerequisite) | ~10h CPU (both states) | P1 code work |
| **P0** | Build next_region label-derivation pipeline | (prerequisite) | ~1h | needs P-1 graph artifact |
| **P1** | Parameterise MTLnet + mtl_cv.py with TaskConfig | (prerequisite) | ~3–4h | P-1 |
| **P2** | Single-task baselines (next-cat, next-region) on both engines | CH01, CH04, CH05 | ~3h compute + 1h analysis | — |
| **P3** | 2-task MTL headline runs | CH02, CH03 | ~4h compute + 1h analysis | — |
| **P4** | Ablations (head arch, MTL optimiser, task embedding, monitor) | CH06, CH07, CH08, CH09 | ~8h compute + 2h analysis | — |

**Total compute budget (sequential):** ~30h wall-clock, of which ~10h is the embedding generation and ~20h is MTL training. The user has approved parallel embedding + code work, so effective wall-clock is ~20h if started now.

## Critical-path ordering

```
P-1 (embeddings) ─┐
                  ├─► P0 (labels) ─► P2 (baselines) ─► P3 (MTL) ─► P4 (ablations)
P1 (code)        ─┘
```

- **P-1 and P1 run in parallel.** They have no mutual dependency — embeddings are data; code is scaffolding.
- **P0 gates on P-1.** Label derivation needs the check2HGI graph artifact (placeid → region_idx).
- **P2/P3/P4 gate on P0 + P1.**

## Phase gates

Move to the next phase only when:

- **Out of P-1:** `embeddings.parquet`, `poi_embeddings.parquet`, `region_embeddings.parquet` exist for both FL and AL; training loss shows convergence (not just 8 epochs).
- **Out of P0:** `next_region.parquet` exists for both states; unmapped `placeid`s count is zero (or exhaustively documented); region cardinality logged and committed to `CLAIMS_AND_HYPOTHESES.md` CH04.
- **Out of P1:** legacy regression tests green (`pytest tests/test_regression -q`); `scripts/train.py --task mtl --task-set legacy_category_next` reproduces a byte-identical metric scalar at seed 42 vs the pre-branch commit.
- **Out of P2:** single-task baselines logged per-state; CH01, CH04, CH05 have `status ∈ {confirmed, refuted, partial}` with `results/` evidence pointer.
- **Out of P3:** CH02 + CH03 resolved; paper table entries drafted.
- **Out of P4:** CH06–CH09 resolved; ablation table finalised.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Check2HGI embeddings under-train (loss plateau high) | Monitor convergence daily; extend epochs if needed (500 is a heuristic from the pipeline config). Use best-loss checkpoint. |
| Region cardinality too high per-state (Acc@1 floor too low to see lift) | Report Acc@{1,5,10} + MRR; the ranking metric picks up relative improvements that Acc@1 can't. If Acc@10 is also saturated at near-zero, document as a limitation and consider region coarsening. |
| CH02 negative on both states | Paper pivots to "check-in-level embeddings alone suffice, next-region adds no lift on this data" — still publishable. |
| CH03 shows regression on one head | Report honestly; document likely cause (NashMTL alpha imbalance); consider a fixed-weight baseline. This is the reviewer's #1 concern for MTL papers. |
| Legacy regression test drift from TaskConfig refactor | Bit-exact defaults audited per CH/test; fail loud if any scalar moves. |

## Exit criteria (branch merge decision)

The branch merges into `main` when:

- All Tier-A claims have `status ∈ {confirmed, refuted, partial}` with evidence.
- At least three Tier-C ablations (CH07, CH08, CH09) are resolved.
- Legacy tests green.
- `docs/PAPER_FINDINGS.md` has a check2HGI section drafted with evidence pointers.

If any Tier-A claim refutes, the paper story adapts but the branch still merges — the value is the infrastructure + the honest result.
