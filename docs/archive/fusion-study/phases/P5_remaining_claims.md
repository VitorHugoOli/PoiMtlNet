# Phase 5 — Remaining claims & supplementary analyses

**Goal:** address any claims not fully resolved by P1-P4, plus mechanistic/diagnostic analyses needed for paper figures and reviewer defence.

**Duration:** variable (2-8 h depending on scope).

**Embedded claims:**
- C18 — reproducibility across seeds (if not already done)
- C19 — scale imbalance causes source-level gradient conflict
- C20 — fusion has lower between-task gradient cosine than single-source
- C21 — MTL wall-clock / FLOPs comparison with CBIC
- Any C22+ claims that arose during P1-P4 (in-study discoveries)

---

## Preconditions

- P1-P4 completed.
- Champion identified and tuned.
- List of open questions compiled.

---

## Steps

### P5.1 — Multi-seed confirmation (C18)

If not already done in earlier phases:

Run the final tuned champion at seeds {42, 123, 2024} at 5f × 50ep on AL (and ideally AZ).

**Output:** mean joint ± std across seeds. If std < 0.01, C18 confirmed.

**Compute:** ~3 × 22min = ~1 h.

---

### P5.2 — Per-source gradient analysis (C19)

**Goal:** show empirically that on fusion, the Sphere2Vec half of the input vector receives smaller gradients than the HGI half, and that the two source-halves have lower cosine similarity than the two task losses do.

**Implementation:**

1. Add instrumentation to `scripts/train.py` (or a training wrapper):
   - On a designated fold, log per-step:
     - L2 norm of gradient w.r.t. first half of input (Sphere2Vec)
     - L2 norm of gradient w.r.t. second half (HGI)
     - Cosine between the two half-gradients
     - L2 norm of gradient w.r.t. each task's loss (for reference)
     - Cosine between the two task gradients
2. Run champion on AL fusion, 1 fold, 50 epochs, with instrumentation.
3. Parallel: same on AL HGI-only, to contrast.
4. Plot:
   - Fig A: L2 norm over epochs for each source half (should show imbalance on fusion)
   - Fig B: cosine similarity over epochs for source-halves (fusion) and task-pairs (both embeddings)

**Output:** figures + numerical summary. If fusion's source-cosine is lower than HGI's task-cosine, **C19 confirmed** and provides the paper's mechanism figure.

**Compute:** ~1-2 h including instrumentation + 2 runs.

---

### P5.3 — Task-level gradient cosine on fusion vs HGI (C20)

**Partially done by P5.2.** If the task-cosine metric shows fusion < HGI, that's C20.

No separate runs needed beyond P5.2.

---

### P5.4 — Wall-clock and FLOPs comparison (C21)

**Goal:** close the loop on CBIC's "MTL is 4× slower than single-task cumulative" finding.

**From existing runs:**
- Extract wall-clock time from P2b (single-task-cat, single-task-next) and P2a (champion MTL on same state).
- Compute ratio: MTL_time / (single_cat_time + single_next_time).

**If ratio < 2:** modern MTL is not wall-clock-penalized the way CBIC's was. A nice closing note for the paper.

**If ratio > 3:** still a trade-off; discuss honestly.

**Also report FLOPs** if `fvcore` is installed; otherwise skip.

**Compute:** ~10 min of analysis, no new runs.

---

### P5.5 — Per-category F1 analysis

For paper discussion / related-work contextualization:

Extract per-category F1 from the champion's 5-fold CV on AL and FL. Compare against HAVANA and POI-RGNN per-category numbers (from baseline docs).

**Useful to identify:**
- Which categories benefit most from our framework
- Whether gains are uniform or concentrated in specific categories (e.g., Food, Shopping)
- Whether any categories regress (important for discussion)

**Output:** table for paper: Our F1, HAVANA F1, POI-RGNN F1 per category, delta.

---

### P5.6 — Statistical robustness package

For the paper's methods/experimental-setup section, collect:

- 95% confidence intervals on champion joint F1 (via paired bootstrap or t-distribution)
- Paired t-test p-values between champion and CBIC-config
- Effect sizes (Cohen's d) for main comparisons

Mostly analytical — no new runs.

---

### P5.7 — Catch-all for in-study discoveries

If any C22+ claims were added during P1-P4 (surprising findings that warranted new tests), they live here unless already resolved. Each should have:
- A statement
- A test
- An outcome

Any unresolved claim at end of P5 → either (a) run it now, or (b) explicitly defer to journal/future work.

---

## Compute budget

| Step | Time |
|------|------|
| P5.1 multi-seed | ~1 h (if not done) |
| P5.2 instrumentation + runs | ~2 h |
| P5.4 wall-clock analysis | ~10 min |
| P5.5 per-category analysis | ~30 min |
| P5.6 stats package | ~30 min |
| P5.7 catch-all | variable |
| **Total** | **~4-6 h** |

---

## Outputs

- Figures for paper (mechanism, sensitivity curves, per-category bar chart)
- `docs/studies/results/P5/SUMMARY.md` with:
  - All resolved claim statuses
  - Summary of supplementary analyses
  - Any outstanding (deferred) claims
- Final update to `CLAIMS_AND_HYPOTHESES.md`: every claim has `status` and `evidence`

---

## Exit criterion — study is complete

Every claim in `CLAIMS_AND_HYPOTHESES.md` has:
- `status` ∈ {confirmed, refuted, partial, abandoned}
- `evidence` pointer to a results dir or explicit note "deferred"

**No claim left as `pending` at study completion.**

### P5_protocol_delta — measure GroupKFold vs KFold cost

**Claim:** N04 (provisional).

**Design:** AL/DGI, mtlnet + nash_mtl, 5f × 50ep, seed 42, two runs:
  a) `StratifiedGroupKFold(groups=userid)` — current default  
  b) `StratifiedKFold` — record-level (CBIC/HAVANA protocol)

**Expected:** record-level yields next_F1 ~0.01–0.03 higher than
user-isolated; category F1 change expected to be negligible.

**Budget:** ~30 min compute (two AL/DGI 5f × 50ep runs).

**Output:** `docs/studies/fusion/results/P5/protocol_delta/`

**Status:** planned — do NOT run until P1–P4 complete.

---

## Transition to synthesis

After P5, the study phase ends. The next step is synthesis:

1. Read all `SUMMARY.md` files from P1-P5.
2. Read `CLAIMS_AND_HYPOTHESES.md` top to bottom.
3. Write the new `docs/PAPER_FINDINGS.md` with:
   - Section per claim category (Tier A, B, C, D, E)
   - Evidence summary per claim
   - Effect size + statistical significance
   - Pointer to results dir
4. Begin paper drafting in `articles/BRACIS_2026/`.

The old `PAPER_FINDINGS.md` is archived (renamed to `PAPER_FINDINGS_pre_bugfix.md`) for historical reference.
