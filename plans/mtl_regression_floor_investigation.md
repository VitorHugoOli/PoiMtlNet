# MTL regression test — `cat_f1` below floor investigation

**Author:** Claude Opus 4.6 (from the sphere2vec PR session, 2026-04-11)
**Audience:** Future Claude Code session (or human) debugging `tests/test_regression/test_regression.py::TestMTLRegression::test_mtl_f1_within_tolerance`
**Status:** **Pre-existing failure, unrelated to PR #12 (sphere2vec paper variant).** Reproduces on unmodified `main` via `git stash && pytest …`. Documented here so that it does not block `feat/sphere2vec-paper-variant` review and so that the next person picking up the regression floor has context.
**Repo root:** `/Users/vitor/Desktop/mestrado/ingred`

---

## 0. TL;DR

`tests/test_regression/test_regression.py::TestMTLRegression::test_mtl_f1_within_tolerance` fails with

```
AssertionError: MTL category F1=0.9019 below floor 0.9200
```

The floor `MTL_CAT_F1_FLOOR = 0.92` was **calibrated on 2026-03-26** at `cat_f1 = 0.9286 ± 0.0` (5 runs, CPU, seed 42, torch 2.9.1, 10 epochs). The current observed value `0.9019` is **2.67 points below the calibration mean** and **1.81 points below the floor**. Since calibration reported `std = 0` on CPU, this is a **non-noise regression** — something deterministic changed between 2026-03-26 and today.

Torch version has NOT changed (`torch==2.9.1` then and now), so the regression has to be in our own code: either `src/models/mtlnet.py`, the shared fixture generators in `tests/test_integration/conftest.py`, `src/losses/nash_mtl.py`, or a dependency chain thereof.

**Not a flake.** CPU forced, `seed_everything()` called, calibration observed `std = 0`. The `0.9019` is also a deterministic value — running the test a second time produces the same number. **This is a real behavioural drift in training, not measurement noise.**

---

## 1. Context — what the test actually does

File: `tests/test_regression/test_regression.py:316`

```python
def test_mtl_f1_within_tolerance(self):
    from models.mtlnet import MTLnet
    seed_everything()  # SEED = 42, also called by autouse setup fixture
    X_cat_train, y_cat_train = make_category_data(NUM_TRAIN // NUM_CLASSES, seed=SEED)
    X_cat_val, y_cat_val = make_category_data(NUM_VAL // NUM_CLASSES, seed=SEED + 1)
    X_next_train, y_next_train = make_next_data(NUM_TRAIN // NUM_CLASSES, seed=SEED)
    X_next_val, y_next_val = make_next_data(NUM_VAL // NUM_CLASSES, seed=SEED + 1)
    cat_train_dl, cat_val_dl = make_loaders(X_cat_train, y_cat_train, X_cat_val, y_cat_val)
    next_train_dl, next_val_dl = make_loaders(X_next_train, y_next_train, X_next_val, y_next_val)
    model = MTLnet(
        feature_size=EMBED_DIM, shared_layer_size=256,
        num_classes=NUM_CLASSES, num_heads=8, num_layers=4,
        seq_length=SEQ_LEN, num_shared_layers=4,
    )
    cat_f1, next_f1 = _train_mtl_and_evaluate(
        model, cat_train_dl, cat_val_dl, next_train_dl, next_val_dl
    )
    assert cat_f1 >= MTL_CAT_F1_FLOOR   # 0.92
    assert next_f1 >= MTL_NEXT_F1_FLOOR  # 0.99
```

`_train_mtl_and_evaluate` (lines 75–133) trains the model for 10 epochs with:
- `AdamW(lr=1e-3, weight_decay=0.01)`
- `CrossEntropyLoss()` for each task, **summed** (`loss = cat_loss + next_loss`)
- Cycles the shorter dataloader to match the longer one each epoch
- CPU-forced (`DEVICE` from `tests/test_integration/conftest.py`)

**Note: this is NOT the production MTL trainer** (`src/training/runners/mtl_cv.py`). It is a tiny synthetic-data training loop that never touches `NashMTL`, `OneCycleLR`, early stopping, or any of the production machinery. It tests only `MTLnet.forward` + basic gradient descent on synthetic data.

Synthetic data generators live in `tests/test_integration/conftest.py`:
- `make_category_data(n_per_class, seed)` — shape `(n_per_class * NUM_CLASSES, EMBED_DIM)` with per-class Gaussian clusters
- `make_next_data(n_per_class, seed)` — shape `(n, SEQ_LEN, EMBED_DIM)` with separable per-class patterns
- `make_loaders(...)` — wraps in `TensorDataset` + `DataLoader`

Constants (from `tests/test_integration/conftest.py`):
- `BATCH_SIZE`, `EMBED_DIM`, `NUM_CLASSES`, `NUM_TRAIN`, `NUM_VAL`, `SEED`, `SEQ_LEN`, `DEVICE = "cpu"`

---

## 2. Calibration evidence

From the docstring at `tests/test_regression/test_regression.py:254`:

```
# Calibration (5 runs, seed=42, CPU, 10 epochs, torch==2.9.1, 2026-03-26):
#   cat_f1: [0.9286, 0.9286, 0.9286, 0.9286, 0.9286]
#   next_f1: [1.0, 1.0, 1.0, 1.0, 1.0]
#   CPU determinism → std=0. Floors set with PyTorch version margin.
MTL_CAT_F1_FLOOR = 0.92
MTL_NEXT_F1_FLOOR = 0.99
MTL_PARAM_COUNT = 4307855
```

**Observed today** (2026-04-11, torch 2.9.1, CPU, seed 42, 10 epochs, unmodified main via `git stash`):
```
cat_f1  = 0.9019
next_f1 = ???      # test fails on the cat assertion before reaching the next one
```

**Gap**: `0.9286 → 0.9019 = -0.0267` (-2.67 pts). The `0.08` margin below the floor (0.9200 − 0.9019 = 0.0181) was not big enough to absorb the regression.

---

## 3. What has NOT regressed (rule out the easy stuff)

Already verified during the sphere2vec session:

- **Not caused by PR #12.** `git stash` on an unmodified `worktree-sphere` base (at commit `5235e3e`) reproduces the failure. This is noted both in the PR #12 description and in `/tmp/mtl_baseline_rbf.log` session artifacts.
- **Not torch version drift.** Calibration used `torch==2.9.1`; current venv `.venv_new` still has `torch==2.9.1`.
- **Not the NashMTL solver bug** fixed in PR #12. The regression test does NOT invoke `NashMTL` at all — it uses plain `CrossEntropyLoss()` summed across tasks. The solver crash only affects production training in `src/training/runners/mtl_cv.py`.
- **Not a test-side change.** `make_category_data` / `make_next_data` / `make_loaders` / `seed_everything` in `tests/test_integration/conftest.py` have been stable since Phase 7 (commit `4812a52`, which predates calibration — see §4.2).
- **Not `MTL_PARAM_COUNT` drift.** The Layer-1 test `test_mtl_shared_vs_task_params` still passes, so the model still has the exact param count (`4307855`) it had at calibration. The forward path produces different training dynamics without changing parameter shape.

---

## 4. Suspect commits (2026-03-26 → 2026-04-11)

Between the calibration date and today, the following commits touched files on the MTLnet forward or loss path. Listed in order of suspicion:

### 4.1 `63612e6` — perf(mtlnet): dedupe padding mask, cache causal mask, skip task-id gather
**Date:** 2026-04-11 (just days ago)
**Author:** VitorHugoOli
**Why suspicious:** This touches `src/models/mtlnet.py` forward path. Any change to the forward math — even one that is mathematically equivalent on paper — can produce different floating-point outputs in fp32, especially if tensors are computed in a different order or reused across calls.

**Points to check:**
- Does the new padding-mask path produce the same mask values on the synthetic `make_next_data` inputs? (Hint: `make_next_data` produces no padding tokens, so the mask should be all-attend — but if the optimization introduces a shape change or dtype change, that still affects downstream.)
- Does `task_embedding` getting skipped change the output distribution? If the task-id gather used to add a constant bias, removing it shifts the classifier's operating point.
- Is causal-mask caching keyed on the right invariants? A stale cache across forward calls would be catastrophic, but the Layer-1 shape test would still pass.

**How to verify:** `git checkout 63612e6~1 -- src/models/mtlnet.py && pytest tests/test_regression/test_regression.py::TestMTLRegression::test_mtl_f1_within_tolerance -q`. If F1 returns to `0.9286`, this is the culprit.

### 4.2 `92e8bc6` — fix(mtl): restore Nov 5 baseline by reverting category head + dropping cat class weights
**Why suspicious:** A revert of the category head that was calibrated against. The calibration snapshot was taken **after** this commit (based on date ordering — check via `git log --oneline 92e8bc6...` and `git show 92e8bc6 | grep Date`) but reverted state may have drifted behaviour relative to what was measured.

**Points to check:**
- Was calibration taken before or after this commit? If before, the current head is the "Nov 5 baseline" and the 0.9286 calibration is stale.
- Which category head is in `src/models/heads/category.py` now? Does it match what calibration measured?

**How to verify:** `git show 92e8bc6 --stat` and compare the commit date to `2026-03-26`. If the commit is *after* the calibration date, it has invalidated the floor.

### 4.3 `4812a52` — refactor: Phase 7 [7.1] add integration tests for category, next, MTL
**Why suspicious:** This is the commit that added the shared synthetic generators to `tests/test_integration/conftest.py`. If this commit is the one that introduced those generators, then the calibration comment (dated 2026-03-26) must be the original calibration against the Phase-7 generators. But if someone later touched `make_category_data` or `make_next_data` (e.g. changed the seed wiring, changed the class-separation geometry, changed `NUM_TRAIN`), the generated data would differ and the calibration would be stale.

**Points to check:**
- `git log --oneline tests/test_integration/conftest.py` — any commits after `4812a52` and `897f8f3` (the Phase 7.2 fixture migration)?
- Diff `EMBED_DIM`, `NUM_CLASSES`, `NUM_TRAIN`, `NUM_VAL`, `SEQ_LEN`, `SEED` constants against the values used at calibration time.

### 4.4 Other commits to investigate if 4.1–4.3 are clean

```
2d73fe3  perf(mtl): replace sklearn classification_report with torchmetrics on hot paths
fe04847  perf(nashmtl): keep GTG normalization on-device         # NashMTL-only, should NOT affect this test
23c01e7  perf(mtl): pre-move fold tensors to MPS, cache param groups, drop dead zero_grad
f063ac3  perf(mtl): enable MPS CPU fallback safety net at import time   # import-time side effect — CHECK
58c6757  refactor: Phase 5 [5.2] migrate models/ — move mtl_poi.py, add deprecation shims
```

`f063ac3` deserves special attention: if it calls `torch.set_default_dtype` / `torch.backends.*` / `os.environ` at import time, that could perturb fp32 arithmetic on CPU.

---

## 5. Reproduction recipe

From a clean checkout of the repo:

```bash
cd /Users/vitor/Desktop/mestrado/ingred
source .venv_new/bin/activate
# Verify torch version matches calibration
python -c "import torch; assert torch.__version__ == '2.9.1', torch.__version__"
# Run only the failing test
pytest tests/test_regression/test_regression.py::TestMTLRegression::test_mtl_f1_within_tolerance -v
```

Expected (today): `FAILED ... MTL category F1=0.9019 below floor 0.9200`.
Expected (at calibration): `PASSED`, `cat_f1 ≈ 0.9286`.

To reproduce the 2f/25e ablation baseline runs from PR #12 (useful if you want to check whether the production MTL path is *also* regressed):

```bash
python scripts/train.py --task mtl --state alabama --engine sphere2vec --folds 2 --epochs 25
```

but note that this exercises `NashMTL` + `OneCycleLR` + the real fold data, not the regression fixture.

---

## 6. Investigation plan — ordered steps

### Step 1 — Confirm the baseline commit
Find the exact commit that was `HEAD` on 2026-03-26:

```bash
git log --before='2026-03-27' --after='2026-03-25' --all --oneline
```

If that commit passes the test and produces `cat_f1 = 0.9286`, it is the known-good baseline. If not, the calibration comment is wrong and the entire floor needs to be recalibrated on a new baseline.

### Step 2 — Bisect
```bash
git bisect start
git bisect bad HEAD
git bisect good <commit-from-step-1>
git bisect run bash -c 'source .venv_new/bin/activate && pytest tests/test_regression/test_regression.py::TestMTLRegression::test_mtl_f1_within_tolerance -q'
```

Expected bisect duration: ~5 commits between 2026-03-26 and 2026-04-11 (based on `git log --oneline --since='2026-03-26'`), so ≤ 3 bisect iterations.

### Step 3 — Confirm root cause
Once bisect identifies the culprit commit, verify by:
1. `git show <culprit>` — read the diff to confirm the semantic change
2. `git checkout <culprit>~1 -- <changed-files>` — revert just those files, rerun test
3. If F1 returns to 0.9286, the revert proves causality.

### Step 4 — Decide: fix or recalibrate
Two valid responses:

**(a) The culprit is a bug** (e.g. `63612e6` introduced a mask bug, `92e8bc6` reverted to a worse head). Fix the bug: undo the regression, re-run, F1 should return to ~0.9286.

**(b) The culprit is an intentional improvement** whose cost is a small synthetic-benchmark regression (e.g. a numerically-different-but-production-better forward path). **Recalibrate**: rerun the calibration protocol on current `HEAD` and update `MTL_CAT_F1_FLOOR` in the test. Update the comment at line 254 with the new date, torch version, and measured values.

**Do NOT lower the floor without understanding why it dropped.** The calibration comment says "PyTorch version margin" — the margin was 0.0086 (0.9286 − 0.9200) and it should stay proportional to the observed std of a fresh recalibration, not stretched to absorb unexplained drift.

### Step 5 — Add a guardrail
Once fixed or recalibrated, add a sentinel test that would have caught this faster:

```python
def test_mtl_calibration_snapshot(self):
    """
    Pin a hash of (model state_dict, optimizer state, training data) after
    one epoch on synthetic data. Any drift in training dynamics invalidates
    the hash immediately, instead of waiting for the F1 floor to be breached.
    """
```

This is a higher-sensitivity early-warning for the same class of regression.

---

## 7. Why this matters (and why it is not urgent)

### Matters because
- The test is called a "regression safety net" (line 2 of the test file). A failing safety net that everyone has been ignoring is worse than no safety net — it trains people to skip regression failures.
- The 2.67-pt drop is too large to be floating-point slop on a deterministic CPU run. Something is measurably worse. On a harder, less-calibrated task (e.g. real MTL on Alabama) the same regression could be hiding 2-3 pts of cat F1.
- PR #12's 5×50 Alabama ablation showed RBF cat F1 = 14.15%. If this test's regression is real and also affects the production path, the "real" RBF cat F1 might be 2-3 pts higher, which would widen the rbf-vs-paper gap.

### Not urgent because
- It does not block PR #12 — verified pre-existing, orthogonal to sphere2vec.
- It only blocks the regression CI layer, not production training.
- The fix-or-recalibrate decision is 1–2 hours of focused work, well within one session.

---

## 8. Files / symbols / artifacts referenced

### Code
- `tests/test_regression/test_regression.py` — the failing test itself (lines 250–343)
- `tests/test_integration/conftest.py` — shared fixtures (`make_category_data`, `make_next_data`, `make_loaders`, `seed_everything`, `SEED`, `DEVICE`)
- `src/models/mtlnet.py` — the model under test
- `src/models/heads/category.py` — the head reverted in `92e8bc6`
- `src/losses/nash_mtl.py` — **not** on this test's path (test uses plain CE); mentioned here only to rule it out

### Commits (prime suspects, 2026-03-26 → 2026-04-11)
- `63612e6` — perf: dedupe padding mask, cache causal mask, skip task-id gather
- `92e8bc6` — fix: restore Nov 5 baseline by reverting category head + dropping cat class weights
- `4812a52` — refactor: Phase 7 [7.1] add integration tests
- `897f8f3` — refactor: Phase 7 [7.2] regression fixtures use shared synthetic data
- `f063ac3` — perf(mtl): enable MPS CPU fallback safety net at import time

### External
- PR #12 (sphere2vec paper variant) notes this failure in its test-plan section but does NOT attempt to fix it, because the failure is orthogonal.
- Session transcript: this plan was authored as a follow-up from the sphere2vec PR review.

---

## 9. Expected effort

| Step | Estimated time |
|---|---|
| Step 1: confirm baseline commit | 5 min |
| Step 2: git bisect run | 15–30 min (≤ 3 iterations × ~15 s each, plus bookkeeping) |
| Step 3: confirm via targeted revert | 10 min |
| Step 4: fix or recalibrate | 30–60 min (longer if the fix requires code changes to `mtlnet.py`) |
| Step 5: add snapshot guardrail | 20 min |
| **Total** | **~1.5–2 hours** |

---

## 10. Decision log

2026-04-11 (this session):
- Discovered the failure while running a broad test sweep as part of the sphere2vec PR #12 verification.
- Verified it reproduces on unmodified `main` via `git stash`, confirming it is pre-existing and not caused by the sphere2vec changes.
- Chose to document instead of fix because: (a) out of scope for PR #12, (b) needs a proper bisect that would inflate the PR's blast radius, (c) needs a decision on fix-vs-recalibrate that should be made by the repo owner, not auto-applied by the agent.
- This plan is the handoff artifact.
