# Critical Review — Session of 2026-04-15

Honest self-audit of the code, the strategy, and the gap vs published SOTA. Three sections:

1. Code review (where the implementation is fragile, hacky, or incomplete).
2. SOTA comparison for the next-region auxiliary task.
3. **The region-embeddings question** — we have them, we're not using them.

Each item is labeled by severity: **[BLOCK]** must-fix before P2, **[PAPER]** affects paper claims, **[TECH]** technical debt worth tracking.

---

## 1. Code review

### 1.1 [BLOCK] `TaskSet` is frozen; runtime `num_classes` resolution is awkward

`TaskSet` and `TaskConfig` are `frozen=True` dataclasses. To resolve `task_b.num_classes = n_regions` at runtime, the smoke test has to reconstruct the entire `TaskSet` with the replaced `TaskConfig`. This pattern will leak into `scripts/train.py`.

```python
# Current code in smoke test:
resolved = TaskSet(
    name=CHECK2HGI_NEXT_REGION.name,
    task_a=CHECK2HGI_NEXT_REGION.task_a,
    task_b=TaskConfig(name=tb.name, num_classes=n_regions, ...),  # re-specify all fields
)
```

**Fix:** use `dataclasses.replace(preset.task_b, num_classes=n_regions)` and a thin helper:

```python
def resolve_task_set(preset: TaskSet, **overrides) -> TaskSet:
    # e.g. resolve_task_set(CHECK2HGI_NEXT_REGION, task_b_num_classes=1109)
```

**Severity:** **[BLOCK]** — not because the current code fails, but because the awkwardness will compound across the 4–5 call sites in real runs.

### 1.2 [BLOCK] `smoke_check2hgi_mtl.py` uses `num_classes=max(7, 1109)` as a shim

The `num_classes` positional arg to `train_model` is still used by `compute_class_weights` in `train_with_cross_validation` (confirmed by the advisor earlier). The smoke test passes a single `num_classes` because it doesn't go through `train_with_cross_validation` — so it never hit the bug. When we wire the real `scripts/train.py --task-set`, `compute_class_weights` must be called per task with the correct `num_classes`.

**Fix:** change `mtl_cv.py:572-576` to call `compute_class_weights(…, task_a_num_classes, …)` and `compute_class_weights(…, task_b_num_classes, …)`. Trivial once wired through.

**Severity:** **[BLOCK]** — class-weighted CE is a plausible mitigation for Florida's 22.2% majority region; if we set it up wrong, CH03 is contaminated.

### 1.3 [TECH] Runner internal variable names lie

Inside `mtl_cv.py`, variables are still called `next_loss`, `category_loss`, `next_running_loss`, etc. For the check2HGI preset, `next_loss` is actually `next_region_loss` and `category_loss` is `next_category_loss`. Metric *output strings* are parameterised but internal names aren't.

This is a deliberate tradeoff — renaming internally would have been ~100 extra lines of churn with no behaviour change. But it's a footgun for anyone reading the runner during debug. The bit-exactness audit I did was against the legacy path; the new path's internal names are misleading.

**Fix (cheap, later):** rename to `task_a_*` / `task_b_*` in a follow-up commit. Or add a header comment that spells out the mapping.

**Severity:** **[TECH]** — works but misleads.

### 1.4 [TECH] `head_factory=None` footgun

The `LEGACY_CATEGORY_NEXT` and `CHECK2HGI_NEXT_REGION` presets use `head_factory=None` to trigger the hardcoded historical defaults inside `MTLnet._build_{category,next}_head`. Users building a new preset who try `head_factory="next_mtl"` will see a non-obvious `TypeError: NextHeadMTL.__init__() missing 3 required positional arguments` — because the registry path doesn't inject `num_heads`/`seq_length`/`num_layers`.

**Fix (cheap):** change the `_build_next_head(name=...)` registry path to always inject those three args from the MTLnet constructor context, even when `overrides` is provided. Same for `_build_category_head` with `num_tokens`/`token_dim` defaults.

**Severity:** **[TECH]** — documented in the registry docstring, but still a footgun.

### 1.5 [BLOCK] Checkpoint monitor for the check2HGI track is not yet emitted

`docs/plans/CHECK2HGI_MTL_OVERVIEW.md §2` decided to use `joint_acc1 = mean(acc1_cat, acc1_region)` as the checkpoint monitor. But `mtl_cv.py:500` emits only `val_joint_score = 0.5 * (f1_val_next + f1_val_category)` — F1-based, not Acc@1-based. If `ModelCheckpoint(monitor='val_joint_acc1', …)` is configured, it will silently no-op (the key doesn't exist in `CallbackContext.metrics`).

**Fix:** add `val_joint_acc1` emission to the metrics dict in the `cb.on_epoch_end(...)` call. `val_metrics_next['accuracy']` and `val_metrics_cat['accuracy']` already exist (Acc@1).

**Severity:** **[BLOCK]** — without this, P3 runs will checkpoint against a best-F1 epoch instead of best-Acc@1, contradicting the plan's decision.

### 1.6 [TECH] No unit tests for new paths

I added a smoke test but no unit tests for:
- `TaskConfig` / `TaskSet` edge cases
- `MTLnet(task_set=...)` for the sequential-task-A forward path (the one `if self._task_a_is_sequential` branch)
- `build_next_region_frame` (the loader function in `src/data/inputs/next_region.py`)
- `compute_classification_metrics` CPU fallback at `num_classes > 256`

**Fix:** add `tests/test_tasks/test_presets.py`, `tests/test_data/test_next_region_loader.py`, and one new test in `test_mtlnet.py` that exercises the sequential-A branch.

**Severity:** **[TECH]** — the smoke test validates plumbing end-to-end, but unit tests catch regressions on single components.

### 1.7 [BLOCK] `FoldCreator` does not know about the check2HGI MTL path

The smoke test bypasses `FoldCreator._create_mtl_folds` entirely because that method enforces the legacy POI-exclusivity user-isolation protocol (train-exclusive + ambiguous POIs for the category task). For check2HGI both tasks are sequence-level and share the same X rows — the protocol is a much simpler "single StratifiedGroupKFold on userids" without any POI classification.

**Fix:** add `FoldCreator._create_check2hgi_mtl_folds` + a dispatch branch triggered by the `task_set` argument. Straightforward; smoke-test code already shows the pattern.

**Severity:** **[BLOCK]** — real runs go through `FoldCreator`; the smoke test is unrepresentative.

### 1.8 [PAPER] Legacy tests assert bit-exactness but don't assert NEW path correctness

All 221 legacy tests pass. Zero of them exercise the `task_set=CHECK2HGI_NEXT_REGION` path beyond what `tests/test_models/test_mtlnet.py` already covers (none of which use a non-legacy `task_set`). The smoke test is the only validation. Before P2 runs, we need at least one MTL-runner unit test that constructs a minimal synthetic check2HGI fold, runs 2 epochs, and asserts no runtime errors + plausible metric shapes.

**Severity:** **[PAPER]** — passing tests are implying more than they're actually testing.

---

## 2. SOTA comparison for next-region auxiliary task

### 2.1 What we're doing

Current design on this branch:

```
Input:  sequence of 9 check-in embeddings [B, 9, 64]
Shared: per-task encoder → FiLM (task-conditioned) → residual backbone
Heads:  NextHeadMTL transformer (causal + attention pool) per task
        ├── next_category head → 7-class softmax
        └── next_region   head → 1109-class (AL) / 4703-class (FL) softmax
Loss:   NashMTL-balanced sum of two cross-entropies
Metrics: macro-F1 for category, Acc@K + MRR for region
```

This is **flat classification over regions** with a shared-backbone MTL. No hierarchical beam search. No cross-granularity attention. No region-sequence input.

### 2.2 What SOTA does

| Paper | Task | Mechanism | Input granularity | Output granularity |
|---|---|---|---|---|
| **HMT-GRN** (SIGIR '22) | Next-POI recommendation | GRU over region-sequence + GRU over POI-sequence; **hierarchical beam search** narrows the POI space by top-k regions | Region + POI sequences | Both (hierarchical) |
| **MGCL** (Frontiers '24) | Next-POI recommendation | Multi-granularity **contrastive learning** at POI / region / category levels; auxiliary region + category heads | POI sequences + region + category side info | POI, region, category |
| **Bi-Level GSL** (arXiv '24) | Next-POI recommendation | Explicitly learns region-POI bi-level graph; uses **cross-level attention** | POI graph + region graph | POI |
| **Learning Hierarchical Spatial Tasks** (TORS '24) | Next-POI recommendation | Formalises the hierarchical-MTL framing; **multi-task loss at every level of the hierarchy** | POI + region + category | All three |
| **ImNext** (KBS '24) | Next-POI + time-gap | Irregular Interval Attention + MTL with next-POI + time-gap heads | POI sequence | POI + time-gap |
| **GETNext** (SIGIR '22) | Next-POI | Trajectory flow map + transformer | POI + global trajectory graph | POI |
| **SGRec** | Next-POI | Category-aware graph + auxiliary next-category | POI + category | POI + category |

**Our design is a light version.** Key things SOTA does that we don't:

1. **Hierarchical beam search.** They predict region first, then condition POI prediction on the top-k regions. We predict next-region as a flat auxiliary task only.
2. **Explicit region-sequence input.** They feed a sequence of region IDs (or region embeddings) as a separate input stream. **We feed only check-in embeddings.**
3. **Cross-granularity attention / contrastive.** They have mechanisms that let the region and POI levels exchange information in the forward pass.
4. **Region graph structure.** They use the region adjacency as an input graph edge set.

### 2.3 What our contribution can defensibly be

Two options:

**Option A — "Engineering-scoped" thesis (current plan):**
> We evaluate whether check-in-level (check2HGI) contextual embeddings + a single next-region auxiliary task improves next-POI-category prediction on Gowalla state-level data. A narrow, reproducible contribution.

- Pros: matches the code we've built. Clean falsifiable claim. Honest scope.
- Cons: not competitive with HMT-GRN-class SOTA on FSQ-NYC/TKY. A reviewer who knows the literature will ask "why not do X?"

**Option B — "Closer-to-SOTA" thesis:**
> We propose **hierarchical input representation** (check-in embeddings + region embeddings in parallel streams) + next-region auxiliary task + check2HGI contextual encoder. Evaluated on Gowalla state-level, compared to POI-RGNN / HAVANA / PGC lineage.

- Pros: more defensible against the SOTA gap. Uses an artifact (region embeddings) we already have.
- Cons: requires additional implementation (dual-stream input, new head arch). Bumps scope by ~1–2 days.

**Recommendation:** Option B, conditionally. See §3.

---

## 3. The region-embeddings question

> You asked: "on the check2hgi we have the region embedding, are we going to use that as a window to predict the next region?"

**Short answer: right now, no — and that's probably a mistake.** Read on.

### 3.1 What we have sitting unused

Check2HGI's 4-level hierarchy produces three parquets per state; we currently use only one:

| Artefact | AL shape | FL shape | Used? |
|---|---|---|---|
| `embeddings.parquet` (check-in level) | (113846, 68) | (1407034, 68) | ✓ — current input X for both heads |
| `poi_embeddings.parquet` | (11848, 65) | (76544, 65) | ✗ (HGI uses it in its track) |
| `region_embeddings.parquet` | **(1109, 65)** | **(4703, 65)** | **✗ — completely unused in our MTL path** |

Region embeddings are trained to discriminate between regions at the hierarchical MI loss level. They're explicitly optimised for region-level tasks. **And we're throwing them away.**

### 3.2 Why using them would be a natural improvement

SOTA hierarchical next-POI work (HMT-GRN, Bi-Level GSL) uses region sequences explicitly as input. Our current design assumes the check-in embeddings *implicitly encode* region context (because check2HGI was trained with a region-MI loss). That's probably partially true — but it's an untested assumption, and a dedicated region embedding stream is strictly more information than a check-in embedding that happens to know about regions.

### 3.3 Three concrete options to use region embeddings

**(A) Dual-stream input (recommended):**

At each timestep *t* in the 9-window sequence, feed both:
- the check-in embedding (current)
- the region embedding of the POI visited at *t* (new)

Concatenate at the feature dimension: input shape `[B, 9, 128]` instead of `[B, 9, 64]`. Per-task encoders already handle arbitrary feature sizes (they're nn.Linear stacks).

- Pros: minimal architecture change. Encoder first-layer width shifts from 64 to 128. FiLM and backbone unchanged. One new lookup in the data pipeline.
- Cons: assumes concatenation is the right fusion. A future ablation (FiLM-style cross-stream modulation) could do better.
- Implementation: in `build_next_region_frame` (or a new `build_dual_stream_next_input`), join the per-timestep `poi_X` placeid → region_embedding, concatenate into the per-row X.

**(B) Region-only stream for the region head:**

Two different X tensors per sample: `X_checkin` for next-category head, `X_region` for next-region head. Each head sees its own appropriate granularity. Per-task encoders already handle this — the dataloader for task_b just returns a different X.

- Pros: cleanest "use the right tool for the right task" design. Matches HMT-GRN conceptually.
- Cons: loses the MTL shared-backbone advantage — if the two heads process entirely different inputs, the shared layers don't see anything meaningful to share. This is conceptually closer to two parallel single-task models than true MTL.

**(C) Hierarchical cross-attention (ambitious, not recommended now):**

Replace NextHeadMTL's transformer with one that attends between check-in and region streams. Closest to HMT-GRN. Big architecture lift; out of scope for this branch.

### 3.4 What I'd do

- **Default plan (safe, BRACIS-compatible):** implement Option A (dual-stream concatenation). Small code lift: new input-pipeline function + slight encoder width change + ablation row showing Option A > baseline. This moves us from "Option A thesis" to "Option B thesis" in §2.3 with modest effort.
- **Add as a new claim:** CH12 — "Region embeddings as input improve next-region Acc@1 over check-in-embeddings-only." Single ablation row.
- **Keep the current check-in-only path as the A-side of that ablation.** It becomes a useful comparison: "without region embeddings in input" vs "with region embeddings in input."

### 3.5 Effort to implement

- `src/data/inputs/next_region_dual.py` — new loader that joins region embeddings into the X tensor. ~50 LOC.
- `pipelines/create_inputs_check2hgi.pipe.py` — add `--dual-stream` flag to emit the enriched parquet. ~20 LOC.
- `smoke_check2hgi_mtl.py` — add a second smoke mode that loads the dual-stream input. ~10 LOC.
- Tests: 1–2 unit tests covering the join correctness.

**Total:** ~2 hours of focused work. The payoff is a visible SOTA-alignment improvement and a new claim.

---

## TL;DR — what to do next

Before investing in P2 baseline runs:

1. Fix the five **[BLOCK]** items in §1.
2. Decide between Option A and Option B thesis framings (§2.3).
3. If Option B: implement dual-stream region-embedding input (§3.3 Option A). ~2h effort.
4. Address advisor concerns from `HANDOFF.md` §Advisor-concerns: extend check2HGI training, add class-weighted CE for region head, fix per-task `compute_class_weights`.
5. Then run P2.

The current code is **correct but thin**. It's a working implementation of the narrow-thesis version; it's one design choice away from a broader-thesis version that'd be stronger for BRACIS.
