# Phase 4 — Hyperparameter sensitivity

**Goal:** validate that the champion configuration is robust to hyperparameter choices, and find the true ceiling of the framework by tuning a few high-leverage knobs.

**Duration:** ~7-10 h (AL-only for the bulk; one validation run on AZ).

**Embedded claims:**
- C15 — DSelectK is insensitive to hparams near defaults
- C16 — growing the shared backbone improves MTL transfer
- C17 — batch size and learning rate are robust

---

## Preconditions

- P3 complete, champion config confirmed across embeddings.
- No signs of fundamental issues from P1-P3.

**If champion in P1-P3 does NOT use DSelectK**, adapt this plan to the actual winning architecture (most eixos still apply).

---

## Scope of study

P4 is hyperparameter-only. We vary one axis at a time, holding the others at the champion defaults. This is a trade-off: it misses interactions, but interaction search (full grid) is too expensive.

**Champion defaults (placeholder — fill in from P2/P3 output):**
- Embedding: fusion (128D)
- Architecture: DSelectK (e=4, k=2, temp=0.5)
- Optimizer: Aligned-MTL (or winner from P1)
- Heads: cat*, next* (from P2)
- Backbone: shared_layer_size=256, num_shared_layers=4
- Training: LR=1e-4, batch=4096 (grad_accum=1), dropout=0.2, 50 epochs
- Seed: 42

---

## Axes of study

### Axis 1 — DSelectK hparams (if champion uses DSelectK)

**C15** — robustness at the champion.

| Sub-axis | Values | Runs |
|----------|--------|------|
| `num_experts` (e) | {2, 4, 6, 8} | 4 screen (1f×10ep) + top-2 at 5f×50ep |
| `num_selectors` (k) | {1, 2, 3, 4} | 4 screen + top-2 at 5f×50ep |
| `temperature` (τ) | {0.1, 0.3, 0.5, 0.7, 1.0} | 5 screen + top-2 at 5f×50ep |

**Analysis:** for each sub-axis, plot joint F1 vs value. If curve is flat within ±0.01 joint near the default, C15 confirmed.

---

### Axis 2 — Shared backbone size (highest leverage)

**C16** — does growing the backbone help?

| Sub-axis | Values | Runs |
|----------|--------|------|
| `shared_layer_size` | {128, 256, 384, 512} | 4 screen + top-2 at 5f×50ep |
| `num_shared_layers` | {2, 4, 6, 8} | 4 screen + top-2 at 5f×50ep |
| **Combined** (best size × best depth) | 1 | 1 × 5f×50ep |

**Analysis:**
- If 512-wide or 6-deep improves joint by > 0.02: **C16 confirmed**, include in paper.
- If curve is flat: refute — backbone is already large enough.

**Importance:** if C16 is confirmed, this is a new publishable finding.

---

### Axis 3 — Training schedule

**C17** — LR / batch size / dropout robustness.

| Sub-axis | Values | Runs |
|----------|--------|------|
| Learning rate | {5e-5, 1e-4, 2e-4, 5e-4} | 4 screen + top-1 at 5f×50ep |
| Batch size (grad_accum=1) | {2048, 4096, 8192, 16384} | 4 screen + top-1 at 5f×50ep |
| Dropout | {0.1, 0.2, 0.3, 0.4} | 4 screen + top-1 at 5f×50ep |

**Analysis:**
- Batch size is the critical one (it attacks the batch-size confound directly). If the champion is stable across all batch sizes: strong defence.
- LR and dropout should be mildly robust — flag if any one is critical.

---

### Axis 4 — CAGrad hyperparameter c (if champion uses CAGrad)

If Aligned-MTL wins in P1, this axis is skipped (Aligned-MTL has no hparams).

Otherwise:
- c ∈ {0.2, 0.4, 0.6, 0.8} — 4 screen + top-1 at 5f×50ep.

---

### Axis 5 — Window size (low priority, in-study addition)

The 9-step window is arbitrary from the CBIC paper. Worth testing:
- `slide_window` ∈ {5, 7, 9, 11, 15}

**Constraint:** each value requires regenerating next-task inputs from checkins (~5 min / state / value). Total: ~25 min upstream + 5 screen + top-1 at 5f×50ep = ~45 min.

**Hypothesis:** 9 is near-optimal; shorter = missing context, longer = padding waste.

---

## Test IDs

- `P4_AL_e<value>` (num_experts)
- `P4_AL_k<value>` (num_selectors)
- `P4_AL_temp<value>`
- `P4_AL_size<value>`
- `P4_AL_depth<value>`
- `P4_AL_size<s>_depth<d>` (combined)
- `P4_AL_lr<value>`, `P4_AL_bs<value>`, `P4_AL_drop<value>`
- `P4_AL_c<value>` (if applicable)
- `P4_AL_win<value>` (if applicable)

---

## Compute budget

| Axis | Screen runs | Confirm runs | Total time (AL) |
|------|-------------|--------------|-----------------|
| 1 (DSelectK) | 13 × 1min | 6 × 22min | ~2.5 h |
| 2 (Backbone) | 8 × 1min | 5 × 22min | ~2 h |
| 3 (Training) | 12 × 1min | 4 × 22min | ~1.7 h |
| 4 (CAGrad c) | 4 × 1min | 1 × 22min | ~30 min |
| 5 (Window, optional) | 5 × 1min | 1 × 22min | ~1 h (+upstream) |
| **Total** | 42 | 17 | **~7-8 h** |

---

## Analysis steps

After all axes complete:

### Identify best config per axis

For each axis, the value with highest joint F1 at 5f×50ep.

### Build a final champion

If any axis yields > 0.02 joint improvement over original defaults:
- Combine those improvements into a new "tuned champion."
- Verify with one full run.
- If it beats the original champion, it replaces it.

### Compute sensitivity summary

For the paper appendix:
- Table: axis, range of values, max Δ joint, recommended value
- Shows reviewers that we checked, and the result is robust

### C15, C16, C17 determinations

Update claim statuses based on findings.

---

## Surprises to watch for

| Symptom | Interpretation |
|---------|----------------|
| Larger backbone helps significantly | **C16 confirmed — new finding.** Consider including as a primary contribution. |
| LR 2e-4 > 1e-4 | Mild; tune the champion. |
| Batch size 2048 >> batch 8192 | Champion may be batch-size-sensitive; weakens robustness claim. |
| `num_experts=8` >> `num_experts=4` | DSelectK capacity matters more than we thought. Interesting for journal. |
| Window 15 > 9 | Longer context helps; costs more compute. Trade-off for paper. |

---

## Phase gate for P5

Proceed to P5 (or wrap up) when:
1. All axes have at least screening data.
2. No high-leverage axis has been left unexplored.
3. We have a defensible "these hparams are our choice because..." story.

If Axis 2 (backbone) reveals big improvements, consider re-running P3 with the new backbone size — expensive but important.

---

## Outputs

- `docs/studies/results/P4/` with per-axis sub-dirs
- Sensitivity curves (plots) saved per axis
- `docs/studies/results/P4/SUMMARY.md` with final tuned champion
- Paper appendix table + 1-2 sensitivity figures
- Updated claim statuses
