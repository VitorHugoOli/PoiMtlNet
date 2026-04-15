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

## Not in this table (deferred)

- FSQ-NYC / TKY replication (CH10 — declared limitation).
- Encoder enrichment phases (CH11 — declared scope).
- Texas / Arizona states (out of scope for FL + AL focus).
- 3-task extension with `next_time_gap` (scaffolded but not evaluated).
