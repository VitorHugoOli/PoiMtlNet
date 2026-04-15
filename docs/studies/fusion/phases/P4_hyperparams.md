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

### Pre-sweep reference baseline — incidentally achieved in P0 leakage ablation (2026-04-15)

The P0 leakage-ablation runs landed an untuned *paired reference* on
both Alabama and Florida that already **beats the previous best
MTLnet + NashMTL configuration** reported in `docs/reports/report_v1_20260415.md`
§3. Worth logging as a floor before P4 starts: any champion produced by
P1–P3 must at minimum match this, or the formal ablation has selected
something worse than an incidental baseline.

**Config that produced the reference baseline (reproducible today):**

```
.venv/bin/python experiments/hgi_leakage_ablation.py --state Florida --arms baseline
.venv/bin/python experiments/hgi_leakage_ablation.py --state Alabama --arms baseline
```

- State: Florida / Alabama
- Engine: HGI-only (not fusion)
- Model: `mtlnet_dselectk`, `num_experts=4`, `num_selectors=2`, `temperature=0.5`
- MTL loss: `aligned_mtl`
- Training: 50 epochs, lr=1e-4, batch=2048, `gradient_accumulation_steps=1`, seed 42
- Folds: 1 (fold 0 of StratifiedGroupKFold `n_splits=2`; split regenerated from current inputs via `--no-folds-cache`)
- Embedding defaults: `le_lambda=1e-8`, `hard_neg_prob=0.25`, `alpha=0.5`, `poi2vec_epochs=100`, `cross_region_weight=0.7`, HGI `epoch=2000`

**Reference numbers vs the previous best (from report_v1 §3):**

| setup | state | Next-POI F1 | Category F1 |
|---|---|---:|---:|
| previous: `mtlnet` (base) + `nash_mtl` (batch=4096, norm=1.0) | Florida | 34.6% | 76.0% |
| **P0 reference: `mtlnet_dselectk` + `aligned_mtl` (batch=2048, defaults)** | **Florida** | **36.27%** | **76.49%** |
| previous best, same config | Alabama | (not logged in report_v1) | (not logged) |
| **P0 reference, same config** | **Alabama** | **23.83%** | **78.55%** |

**Delta on Florida (the state with the best SNR for Next-POI):**
**+1.67 p.p. Next-POI** and +0.49 p.p. Category, without any hyperparameter
tuning of the new config. Not a formal P1–P3 result — just an incidental
baseline from the leakage audit — but it is a published-numbers-worthy
improvement over `MTLnet + NashMTL` for free.

Artifacts: `docs/studies/results/P0/leakage_ablation/{alabama,florida}/baseline/`.

### Gate check before P4 / Axis 6 sweeps start

**Does the P1–P3 champion match or beat this reference on Next-POI F1
(same state, same fold/seed protocol)?**

- **If yes** — proceed with P4 as planned, using the P1–P3 champion as
  the center of the sweep. Report the delta as evidence that the formal
  ablation improved on the incidental baseline.
- **If no** (P1–P3 champion's Next-POI F1 is *below* the P0 reference on
  matched fold/seed) — **halt and investigate before running any
  hyperparameter sweep.** Possible causes: (i) P1–P3 tested a weaker
  architecture/optimizer combination by accident; (ii) configuration
  drift between runs; (iii) the P0 reference benefits from
  `gradient_accumulation_steps=1` (required for Aligned-MTL) that P1–P3
  didn't consistently set. In any of these cases, the hyperparameter
  sweep would be optimising the wrong center — fix the center first.
- **If P1–P3 champion uses fusion rather than HGI-only**, the
  comparison is not apples-to-apples; re-run the P0 reference arm with
  fusion for the gate check (~10 min on alabama, ~60 min on florida with
  the current driver extended to accept `--engine fusion`).

Include this gate check in any P4 preflight script alongside the data-
integrity checks.

---

### Axis 6 — POI2Vec / HGI embedding hyperparameters (added 2026-04-15, post-C29)

Added after the leakage audit. Anchor metric is **Next-POI F1**, not
Category F1 — per C29 (`CLAIMS_AND_HYPOTHESES.md` Tier H), Category F1 on
OSM-Gowalla is an fclass-identity-preservation metric and doesn't move
the representation-quality story. Report Category F1 alongside as a
fidelity sanity check, but decisions are made on Next-POI.

**Upstream motivation (from leakage ablation arms A / B):**
- Arm B (`hard_neg_prob = 0.0` in HGI hard-negative sampling) improved
  Next-POI F1 by **+2.40 p.p.** on alabama. Strongest single-knob signal
  from the 5-arm ablation — worth a proper sweep.
- Arm A (`le_lambda = 0` in POI2Vec hierarchy loss) moved Category F1
  by +1.36 p.p. but Next-POI flat (+0.06). Cosmetic for Next-POI;
  a defensible cleanup (remove dead code) but no tuning signal.
- Other POI2Vec / HGI knobs that affect the *embedding itself* have not
  been swept on the new data: `alpha`, `poi2vec_epochs`,
  `cross_region_weight`, Node2Vec `walks_per_node` / `context_size`.

**Champion defaults to vary around** (from `pipelines/embedding/hgi.pipe.py`
and the leakage-ablation driver; reproduces the Florida Next-POI F1
= 0.3627 baseline):
- `hard_neg_prob = 0.25` (HGI)
- `le_lambda = 1e-8` (POI2Vec; arm A cleared to 0 without cost)
- `alpha = 0.5` (POI2Region vs Region2City weight)
- `poi2vec_epochs = 100`
- `cross_region_weight = 0.7` (already swept on Alabama, tuned; re-confirm under new metric)
- Node2Vec `walks_per_node = 5`, `context_size = 5`, `p = q = 0.5`
- HGI `epoch = 2000`, `lr = 0.006`, `warmup_period = 40`

**Sweep plan (Florida is preferred — higher Next-POI SNR with 85k vs 13k sequences):**

| Phase | Axis | Values | Runs | Wall-time |
|---|---|---|---|---|
| 6.1 | `hard_neg_prob` | {0.0, 0.1, 0.25, 0.5} | 4 | ~3 h FL (≈50 min/arm) or ~1 h AL |
| 6.2 | `alpha` × `poi2vec_epochs` | {0.3, 0.5, 0.7} × {50, 100, 200} | 9 (fix 6.1 winner) | ~7 h FL / ~2 h AL |
| 6.3 | Node2Vec `walks_per_node` × `context_size` | {3, 5, 10} × {3, 5, 7} | 9 | ~7 h FL / ~2 h AL |
| 6.4 | `cross_region_weight` confirm | {0.5, 0.7, 0.9} | 3 | ~2.5 h FL / ~45 min AL |

**Minimum viable scope** (before journal extension): Phase 6.1 + Phase 6.2
only. On alabama that's 13 runs × ~10 min = ~2 h. On florida 13 × ~50 min
= ~11 h.

**Entry criteria:**
- P1–P3 champion established (so the MTL config is fixed around the
  sweep).
- Evaluation metric locked per C29: anchor = Next-POI F1; Category F1 as
  fidelity check only.

**Exit criteria:**
- Each phase's best config identified on Next-POI F1 (paired comparison,
  3 seeds for the top-3 candidates).
- Observed deltas > estimated 1-fold noise (±1 p.p.); otherwise mark
  current defaults as robust and move on.
- Winner applied as the new "champion embedding config" for cross-state
  replication in P3 re-runs.

**Code hooks (all already in main, default-off):**
- `le_lambda` — `research/embeddings/hgi/poi2vec.py::train_poi2vec(le_lambda=...)`
- `hard_neg_prob` — `research/embeddings/hgi/model/HGIModule.py::HierarchicalGraphInfomax(hard_neg_prob=...)`, plumbed via `args.hard_neg_prob`
- `shuffle_fclass_seed` — `research/embeddings/hgi/preprocess.py::preprocess_hgi(shuffle_fclass_seed=...)` (diagnostic only, NOT part of the sweep)
- `alpha`, `poi2vec_epochs`, `cross_region_weight`, Node2Vec params — all already CLI-threaded in `pipelines/embedding/hgi.pipe.py` CONFIG

**Driver:** `experiments/hgi_leakage_ablation.py` extended with per-arm
entries, or a dedicated `scripts/hgi_hparam_sweep.py` (to write when
phase 6.1 starts — avoid scope creep in the leakage driver).

**Output:** `docs/studies/results/P4/embedding_sweep/{alabama,florida}/`
with `<phase>_<axis>_<value>/` per-arm directories + summary JSON +
a final `README.md` reporting Next-POI F1 winners and the
fidelity-only Category F1 column.

**Reference runs for paired comparison:**
- Alabama baseline (Next-POI F1 = 0.2383): `docs/studies/results/P0/leakage_ablation/alabama/baseline/`
- Florida baseline (Next-POI F1 = 0.3627): `docs/studies/results/P0/leakage_ablation/florida/baseline/`

**Status:** `pending` — not scheduled. Run after P3 completes on clean
data, or opportunistically earlier if compute is available and the
current P1–P3 champion stabilizes.

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
