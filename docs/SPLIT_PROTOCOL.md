# MTL Split Protocol

> This document specifies the cross-validation split protocol for multitask
> learning experiments. It is referenced by [REFACTORING_PLAN.md](REFACTORING_PLAN.md)
> as a Phase 0 hard gate.

---

## 1. Problem Statement

MTL fold creation currently zips independently stratified splits for category
and next tasks. A POI can appear in the training split of one task and the
validation split of the other. Since the shared MTLnet backbone sees that POI
during training via one task, information leaks into the validation of the
other task.

---

## 2. Design Choice: User Isolation as Hard Invariant

When user isolation and bidirectional POI isolation conflict (which they do
whenever ambiguous POIs exist), **user isolation wins**. This is a **research
assumption**, not a settled fact. It is informed by the Phase 0 feasibility
report, which quantifies residual POI overlap. The report can measure exposure
but cannot prove the causal claim that user-level leakage dominates POI
overlap — proving that requires a comparative ablation study (a potential
future experiment in `experiments/ablations/`).

**Rationale:**
- POI embeddings are pre-computed and shared across all splits by construction.
- The primary leakage risk is assumed to be user-level: memorizing movement
  patterns from training and exploiting them in next-task validation.
- Bidirectional POI isolation would require per-sequence POI identity checks,
  but the next-task schema (`[emb_*, next_category, userid]`) does not retain
  sequence-level POI IDs after embedding lookup.

**Residual leakage channel:** Ambiguous POIs in category training may appear
in next-task validation (through val-user sequences). This is quantified and
reported per fold, not silently accepted. The shared backbone has seen these
exact embedding vectors during category training — this IS a real evaluation
bias, but the alternative (filtering val-user sequences) breaks user isolation.

---

## 3. Operational Algorithm

1. **Split users** into train/val folds using `StratifiedGroupKFold` on the
   next-task data: `groups=userid`, `y=next_category`. This is the single
   canonical stratification definition.

2. **Classify POIs** using a mapping `POI → set of users who visited it`
   (materialized from raw checkins — see Phase 2 subtasks):
   - **Train-exclusive POI:** all visiting users are in the train group.
   - **Val-exclusive POI:** all visiting users are in the val group.
   - **Ambiguous POI:** visited by users from both groups.

3. **Derive the category fold:**
   - Train-exclusive POIs → category training.
   - Val-exclusive POIs → category validation.
   - Ambiguous POIs → **category training only** (excluded from category
     validation).

4. **Derive the next-task fold:**
   - All sequences from train-group users → next-task training.
   - All sequences from val-group users → next-task validation.
   - No per-sequence filtering. User isolation is absolute.

5. **FoldCreator** accepts user-group fold assignments as input for both tasks
   rather than creating independent splits. `userid` must be retained as
   first-class metadata in the next-task input (currently dropped — Phase 2
   change).

---

## 4. Guarantees and Residual Leakage

**Hard invariants:**
- No user's data appears in both train and val for either task.
- No POI in category validation was seen during category training.
- User isolation is absolute: ALL of a user's data is in one split.

**NOT guaranteed (quantified residual):**
- Ambiguous POIs in category training may appear in next-task validation
  (through val-user sequences). This is the residual cross-task leakage
  channel. It is quantified per fold in the split manifest.

---

## 5. Acceptance Constraints

Separate per task, configurable via `ExperimentConfig`:

- **Category validation:** at least `min_category_val_fraction` (default 5%)
  of total samples. Each of 7 categories: at least
  `max(min_class_count, min_class_fraction × category_total)` samples
  (defaults: 5, 0.03).
- **Next-task validation:** at least `min_next_val_fraction` (default 5%)
  of total sequences. Determined by user split, not POI filtering.

Thresholds are calibration **inputs** (what the split is evaluated against).
Phase 0 may recommend adjusting them (calibration **outputs**). Researcher
freezes final values in `ExperimentConfig`. Recommendations are per dataset
configuration (state, engine, k_folds, split_relaxation — **not** per seed).

A fold violating either constraint is **invalid for the entire MTL experiment**
(both tasks).

---

## 6. Residual Leakage Evaluation

The Phase 0 feasibility report must assess the residual cross-task POI overlap
and freeze a **maximum acceptable overlap fraction** per dataset configuration
(seed-independent — a property of the data, not the random split).

**Comparison rule:** The threshold applies to the **maximum of both leakage
channels** (cat-train → next-val AND cat-val → next-train, worst-case across
directions). The feasibility decision is based on the **first valid seed**
(the one training would select). If **any fold of the selected seed** exceeds
the threshold on either channel, the configuration is flagged. Worst-case and
range across all 5 seeds are diagnostic context. High seed sensitivity
(large variance) requires justification even if the selected seed passes.

**Manual governance rules (by design):** The following are researcher judgment
informed by Phase 0 data, not automated gates with universal thresholds:
- **Seed sensitivity:** "High variance" depends on absolute overlap levels
  and dataset size. Phase 0 defines this per dataset.
- **Combined-channel exposure:** When both channels are moderately elevated
  but neither crosses the threshold, the report should assess combined impact.
- **Threshold exceedance:** Configurations exceeding their threshold, flagged
  for seed sensitivity, or identified with elevated combined exposure should
  have explicit justification. If none, exclude from MTL (single-task only).

These are deliberate governance choices. For a research repo with heterogeneous
datasets, informed judgment with full traceability is more defensible than
false-precision universal thresholds.

**Scope distinction:**
- **Threshold scope** (seed-independent): state, engine, k_folds, threshold
  settings, split_relaxation.
- **Report validity** (includes seed): measurements depend on the seed
  sequence. Changes to any parameter require re-running the report.

**Governance tradeoff:** The report is a first-class artifact, hashed in
`RunManifest`. It is **advisory, not blocking** by design: a flagged run will
succeed, but the manifest records the flag AND flagged status must appear in
MLHistory summary output (visible in result tables, not buried in metadata).
The researcher has the final say and accepts responsibility.

---

## 7. Seed Regeneration Policy

Strict mode requires **all `k_folds` folds to be valid**. If any fold
violates acceptance constraints:

1. Try the next seed: `[config.seed, config.seed+1, ..., config.seed+4]`
2. Take the **first seed** that produces all k valid folds.
3. Maximum **5 attempts**.
4. Record all attempted seeds (accepted and rejected) in the split manifest.
5. If all 5 fail, **error out**. Do not silently fall back to relaxation.

---

## 8. Relaxation Protocol (Explicit Opt-In)

Requires `split_relaxation = True` in `ExperimentConfig`.

**Execution order:** FoldCreator searches in two phases:
1. **Strict phase:** All 5 seeds in strict mode. Take first that works.
2. **Relaxed phase** (only if all strict seeds failed): Retry only seeds whose
   strict failure was **exclusively from category constraints** (not next-task).
   Same seed = same user split, so relaxation can't fix next-task failures.

**Scope:** Relaxation affects **category validation only**. Next-task is
unaffected (all val-user sequences always go to validation).

Relaxation **moves** ambiguous POIs from category training to category
validation (each POI in exactly one set — no duplication):

1. Compute `val_user_ratio` per ambiguous POI.
2. Threshold schedule: `[0.8, 0.6, 0.5]`. At each threshold, eligible POIs
   move to category validation.
3. Deterministic ordering: `(val_user_ratio desc, placeid asc)`.
4. Applied globally across all categories.
5. Stop as soon as category constraints are met.
6. Same schedule for every fold.
7. If still failing: fold is invalid, try next eligible seed.

**When relaxation is used:** Some POIs move from category training to category
validation. This introduces a new leakage direction: relaxed POIs in category
validation may appear in next-task training sequences. Results must be reported
with the relaxation qualifier.

---

## 9. Split Manifest Artifact

The fully materialized split artifact records:
- Final fold indices for both tasks (train/val sample indices)
- User-group assignments per fold
- POI classification per fold (train-exclusive, val-exclusive, ambiguous)
- Seed used (and seeds rejected, with failure mode per seed)
- Split mode per fold (strict or relaxed)
- Relaxation metadata per fold (which POIs moved, at which threshold)
- Per-fold acceptance constraint metrics for both tasks
- Per-fold cross-task POI overlap (bidirectional, both count and volume):
  - **cat-train → next-val:** ambiguous POIs in category training that appear
    in next-task validation sequences, plus fraction of affected sequences
  - **cat-val → next-train:** relaxed POIs in category validation that appear
    in next-task training sequences, plus fraction (0 under strict)

**How overlap diagnostics are computed:** Since the downstream next-task schema
does not retain POI IDs, diagnostics are computed **during FoldCreator
execution** using raw checkins and POI→users mapping, before embedding lookup.
Phase 2 should materialize a **sequence-to-POI mapping artifact** during input
generation to decouple FoldCreator from raw checkins (see Phase 2 item 7).

The `split_signature` in `RunManifest` is the SHA-256 hash of this artifact.

---

## 10. FeasibilityReport Schema

```python
@dataclass
class ThresholdSettings:
    min_category_val_fraction: float
    min_next_val_fraction: float
    min_class_count: int
    min_class_fraction: float

@dataclass
class PerChannelOverlap:
    cat_train_next_val_poi_count: int
    cat_train_next_val_seq_fraction: float
    cat_val_next_train_poi_count: int       # 0 under strict
    cat_val_next_train_seq_fraction: float  # 0 under strict

@dataclass
class SeedDiagnostic:
    seed: int
    status: str                       # "valid_strict" | "valid_relaxed" | "rejected"
    failure_mode: Optional[str]       # "category" | "next_task" | "both" | None
    per_fold_overlap: list[PerChannelOverlap]
    split_mode: str                   # "strict" | "relaxed"

@dataclass
class FeasibilityReport:
    # Scope (threshold is seed-independent)
    state: str
    engine: EmbeddingEngine
    k_folds: int
    split_relaxation: bool
    threshold_settings: ThresholdSettings

    # Frozen decision
    max_overlap_fraction: float
    decision: str                     # "approved" | "approved_with_justification" | "excluded"
    justification: Optional[str]

    # Flags
    threshold_exceeded: bool
    seed_sensitivity_flag: bool
    combined_exposure_flag: bool

    # Per-seed diagnostics
    seed_diagnostics: list[SeedDiagnostic]
    selected_seed: int
    selected_seed_worst_fold_overlap: PerChannelOverlap
    overlap_range_across_seeds: dict  # {"min": PerChannelOverlap, "max": PerChannelOverlap}

    # Decision reasons (machine-readable)
    decision_reasons: list[str]       # ["threshold_exceeded", "seed_sensitivity", ...]

    # Metadata
    timestamp: str
    schema_version: int = 1
```

Serialized as `feasibility_report.json`. Hashed in
`RunManifest.feasibility_report_signature`.