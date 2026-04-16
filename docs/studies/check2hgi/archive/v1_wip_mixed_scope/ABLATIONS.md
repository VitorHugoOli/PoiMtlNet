# Check2HGI Track — Ablation Matrix

Ablations that go into the BRACIS paper's experimental tables. Each row is a single experiment; each column is a factor varied.

**Convention:** rows marked `[PAPER]` are headline table entries. Rows marked `[APPENDIX]` are supporting evidence, cited but not tabulated in the main paper. Rows marked `[AUDIT]` are sanity-checks, reported in the branch's `HANDOFF.md` but not cited externally.

---

## Table 1 — Embedding × task (the headline)

Validates **CH01** (embedding quality) and **CH02** (MTL lift) in one matrix. Cells are `next_category_macroF1` (reported) with `next_region_Acc@1 / MRR` in parentheses where applicable.

| # | Embedding | Task setup | AL result | FL result | Purpose |
|---|---|---|---|---|---|
| 1 | HGI | single-task next-category | — | — | CH01 baseline `[PAPER]` |
| 2 | HGI | 2-task MTL {next_cat, category} (legacy) | — | — | Legacy reference `[PAPER]` |
| 3 | CHECK2HGI | single-task next-category | — | — | CH01 test `[PAPER]` |
| 4 | CHECK2HGI | single-task next-region | — | — | CH04 test `[PAPER]` |
| 5 | CHECK2HGI | 2-task MTL {next_cat, next_region} | — | — | CH02 + CH03 test `[PAPER]` |

**Read-outs:**
- CH01 = row 3 vs row 1 (delta in next_cat macro-F1).
- CH02 = row 5 vs row 3 (delta in next_cat macro-F1 under MTL vs single-task).
- CH03 = row 5 per-head metrics vs row 3 + row 4 per-head metrics (no regression check).

---

## Table 2 — Head architecture ablation (next-region)

Validates **CH07**. Only the `next_region` head varies; everything else pinned at row 5 of Table 1 (`CHECK2HGI`, 2-task MTL).

| # | `next_region` head | Params | Acc@1 | Acc@5 | Acc@10 | MRR | Purpose |
|---|---|---|---|---|---|---|---|
| 1 | `next_mtl` (transformer, causal) | — | — | — | — | — | Default `[PAPER]` |
| 2 | `next_lstm` | — | — | — | — | — | Simpler seq `[PAPER]` |
| 3 | `next_gru` | — | — | — | — | — | Simpler seq `[APPENDIX]` |
| 4 | `next_transformer_relpos` | — | — | — | — | — | Relative-pos `[APPENDIX]` |
| 5 | `next_hybrid` | — | — | — | — | — | Hybrid `[APPENDIX]` |

Run on Alabama only (cheap enough to replicate on FL if a surprise appears).

---

## Table 3 — MTL optimiser ablation

Validates **CH08**. Only the loss balancer varies.

| # | MTL criterion | `next_cat` F1 | `next_region` Acc@1 | `joint_acc1` | Purpose |
|---|---|---|---|---|---|
| 1 | `nash_mtl` (default) | — | — | — | Default `[PAPER]` |
| 2 | `cagrad` | — | — | — | Gradient-surgery `[PAPER]` |
| 3 | `aligned_mtl` | — | — | — | Gradient-surgery `[APPENDIX]` |
| 4 | `naive` (equal weight) | — | — | — | Baseline `[PAPER]` |
| 5 | `pcgrad` | — | — | — | Gradient-projection `[APPENDIX]` |

Run on Alabama. If row 1 ≈ row 4 within CI, paper drops the optimiser argument and uses `naive` (cheaper).

---

## Table 4 — Micro-ablations (design-choice sanity)

Validates **CH06** and **CH09**. Each row flips one design knob relative to the default (row 5 of Table 1).

| # | Variation | Motivation | Result | Purpose |
|---|---|---|---|---|
| 1 | Default | Reference | — | `[AUDIT]` |
| 2 | Monitor on `joint_f1` instead of `joint_acc1` | CH06 | — | `[APPENDIX]` |
| 3 | `task_embedding` disabled (zero-init, frozen) | CH09 | — | `[PAPER]` |
| 4 | `num_classes=7` for next_region (cap regions to top-7 by frequency) | Control for cardinality effect | — | `[APPENDIX]` |
| 5 | Seed sweep (42, 43, 44, 45, 46) on row 5 of Table 1 | Variance baseline | — | `[AUDIT]` |

Row 4 is a *control*: if Acc@1 on a 7-class region task is similar to Acc@1 on 7-class next-category, the gain in Table 1 row 5 is the MTL signal, not cardinality effects.

Row 5 establishes the noise floor for every other comparison — if seed variance exceeds the CH02 delta, claim CH02 needs downgrading to `partial`.

---

## Reporting discipline

- Every cell includes a CI (5-fold bootstrap, 95%) or `±` std.
- A table row is not published unless the underlying run has `results/<phase>/<test_id>/summary.json` committed.
- The paper's claim pointers in `CLAIMS_AND_HYPOTHESES.md` must resolve to a specific table+row, not just "Phase P3".

---

## Table 5 — MTL architecture × optimiser grid (Phase P5)

Ported from legacy `docs/studies/phases/P1_arch_x_optimizer.md` and adapted to check2HGI's 2-task pair. Full runbook in `phases/P5_arch_optimizer_grid.md`.

**Architectures (5):** `base` (mtlnet), `cgc22`, `cgc21`, `mmoe4`, `dsk42`.

**MTL optimisers (priority-1 first, extend to 20):**

| Category | Priority-1 | Priority-2 (extend later) |
|---|---|---|
| Static | `equal_weight`, `static_weight` | `uncertainty_weighting` |
| Loss-based | `nash_mtl`, `dwa` | `famo`, `bayesagg_mtl`, `excess_mtl`, `stch` |
| Gradient-based | `cagrad`, `aligned_mtl` | `pcgrad`, `gradnorm`, `db_mtl`, `fairgrad`, `graddrop`, `mgda`, `moco`, `sdmgrad`, `imtl_h`, `imtl_l` |

Screen: 5 × 20 = 100 cells per state at 1f × 10ep. Top-10 promoted to 2f × 15ep; top-5 confirmed at 5f × 50ep. AL + AZ.

Claims addressed: **CH14** (gradient-surgery vs equal-weight on check2HGI), **CH15** (expert-gating vs FiLM), **CH16** (winner transfer vs legacy HGI track).

**Prerequisite:** `MTLnetCGC / MTLnetMMoE / MTLnetDSelectK / MTLnetPLE` need the same `task_set` parameterisation we applied to `MTLnet` in P1-b. ~150 LOC per variant — a mechanical port, but not yet done.

---

## Table 6 — Head sweep + MTL-vs-single-task (Phase P6)

Ported from legacy `docs/studies/phases/P2_heads_and_mtl.md` and adapted. Full runbook in `phases/P6_head_sweep.md`.

Both heads on check2HGI are sequential → both pull from the same pool of ~10 next-family heads (`next_mtl`, `next_lstm`, `next_gru`, `next_tcn_residual`, …). Legacy P2's 9-cat × 10-next = 90 grid becomes a symmetric 10 × 10 pool with a two-axis efficient sweep:

- P6a-A: fix task_b, vary task_a across 10 heads (10 runs)
- P6a-B: fix task_a, vary task_b across 10 heads (10 runs)
- P6a-combo: top-3 × top-3 = 9 runs
- P6b: single-task baselines for the MTL champion heads (2 runs)
- P6c: MTL vs single-task (paired t-test, the critical control for CH02)
- P6d (optional): per-head co-adaptation probe

Total ~50 runs per state. Claims addressed: **CH02** (MTL lift, headline), **CH03** (no negative transfer), **CH17** (MTL head ranking ≠ standalone), **CH18** (frozen-backbone co-adaptation), **CH19** (co-adaptation mechanism).

---

## Table 7 — Dual-stream input & cross-attention (Phase P7)

Options A and C from `CRITICAL_REVIEW.md` and `OPTION_C_SPEC.md`. Full runbook in `phases/P7_dual_stream_cross_attention.md`.

Decision-gated: P7b (cross-attention) only runs if P7a (dual-stream concat) shows ≥ 2 p.p. Acc@1 lift on next_region at Florida scale.

| # | Setup | Architecture | Input stream | Purpose |
|---|---|---|---|---|
| 1 | Baseline (vanilla P3) | MTLnet (champion) | Check-in only | Reference `[PAPER]` |
| 2 | Option A | MTLnet | Concat(check-in, region) | Region-emb as input `[PAPER]` |
| 3 | Option C (K=2) | MTLnetCrossAttn | Dual stream | Cross-attention `[PAPER]` (if gate passes) |
| 4 | Option C (K=1) | MTLnetCrossAttn | Dual stream | Depth ablation `[APPENDIX]` |
| 5 | Option C (K=3) | MTLnetCrossAttn | Dual stream | Depth ablation `[APPENDIX]` |

All run on FL + AL. Claims: **CH12** (dual-stream helps), **CH13** (cross-attention helps more), **CH20** (gain is state-dependent — predicted by the probing experiment).

---

## Not in this table (deferred)

- FSQ-NYC / TKY replication (CH10 — declared limitation).
- Encoder enrichment phases (CH11 — declared scope).
- Texas / Georgia / California states (AL + FL + AZ are the planned triple).
- 3-task extension with `next_time_gap` (scaffolded but not evaluated).
