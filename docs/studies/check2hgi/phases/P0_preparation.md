# Phase P0 — Preparation

**Goal:** have integrity-checked data and a runnable pipeline before any scientific test starts.

**Duration:** ~3–4h (integrity checks + code deltas + audit).

**Embedded claims tested:**
- CH04 — Next-region is meaningful (measured during integrity-baseline, refined in P1).
- CH14 — Check2HGI inheritance of HGI fclass shortcut (audit).

---

## Deliverables

1. All three states (AL, FL, AZ) pass `coordinator/integrity_checks.md` preflight PI.1 → PI.7.
2. `src/data/inputs/next_poi.py` + `IoPaths.get_next_poi` + `CHECK2HGI_NEXT_POI_REGION` preset exist in code, with unit tests.
3. `pipelines/create_inputs_check2hgi_next_poi.pipe.py` has materialised `next_poi.parquet` for AL and FL.
4. `scripts/train.py --task-set check2hgi_next_poi_region` runs end-to-end on AL (2 epochs, 1 fold, exit 0).
5. CH14 audit resolved: either (a) confirmed-by-construction (Check2HGI preprocessing has no POI2Vec dependency), or (b) arm-C ablation run + documented.

---

## Tests

### P0.1 — Integrity checks

Run the coordinator's preflight suite for each state:

```bash
STUDY_DIR=docs/studies/check2hgi /study validate --state alabama --engine check2hgi
STUDY_DIR=docs/studies/check2hgi /study validate --state florida --engine check2hgi
STUDY_DIR=docs/studies/check2hgi /study validate --state arizona --engine check2hgi
```

**Expected output:** PI.1 → PI.7 all pass, with distribution warnings at worst.

**Verdict:** per-state → state.json `p0_integrity_check2hgi_<state>` entry updated.

### P0.2 — fclass-shortcut audit (CH14)

**Step 1 (code inspection, no training):** Read `research/embeddings/check2hgi/preprocess.py` and `research/embeddings/check2hgi/check2hgi.py`. Verify whether the preprocessing chain uses POI2Vec or any other fclass-sharing mechanism.

- If **no POI2Vec dependency**, CH14 resolves to `confirmed_by_construction` (no shortcut possible). Document in state.json with the reasoning. Skip Step 2.

- If POI2Vec **is** used (unlikely per the code structure, but verify), proceed to Step 2.

**Step 2 (ablation, if needed):** Regenerate Check2HGI embeddings with `fclass_shuffle_seed=42` (or an equivalent shuffling applied to the appropriate preprocessing field). Run single-task next_poi with the shuffled embedding vs the baseline. Compare Acc@10 drop.

### P0.3 — Code deltas

**Estimated ~2–3h.** These are infrastructure, not experiments — state.json tracks them as `planned → completed` but with no `observed` metrics.

1. **`src/data/inputs/next_poi.py`** — mirror `next_region.py`. The join is:
   ```
   target_placeid (str/int from sequences_next.parquet)
   → placeid_to_idx (dict from checkin_graph.pt)
   → poi_idx (int, used as the label)
   ```
   Emit a parquet with 576 emb cols + `poi_idx` + `userid`.

2. **`src/configs/paths.py`** — add `IoPaths.get_next_poi(state, engine)` + `IoPaths.load_next_poi(state, engine)`. Guard to `engine == CHECK2HGI` only (next_poi is a check2HGI-only input because it depends on the placeid_to_idx mapping from check2HGI preprocessing).

3. **`src/tasks/presets.py`** — add:
   ```python
   CHECK2HGI_NEXT_POI_REGION = TaskSet(
       name="check2hgi_next_poi_region",
       task_a=TaskConfig(name="next_poi", num_classes=0, head_factory=None,
                         is_sequential=True, primary_metric=PrimaryMetric.ACCURACY),
       task_b=TaskConfig(name="next_region", num_classes=0, head_factory=None,
                         is_sequential=True, primary_metric=PrimaryMetric.ACCURACY),
   )
   ```
   Both slot cardinalities are placeholders resolved at runtime via `resolve_task_set`. Register the preset in `_PRESETS`.

4. **`pipelines/create_inputs_check2hgi_next_poi.pipe.py`** — CLI wrapper. Supports `--state` + `--state` repetition, defaults to `["alabama","florida"]`. Logs `n_pois`, top-5 poi_idx frequencies, majority-class fraction (for CH04 baseline computation).

5. **`src/data/folds.py::_create_check2hgi_mtl_folds`** — extend to branch on `task_set.task_a.name`:
   - If `"next_category"` (legacy preset): current behaviour (slot A loads next_category labels from `next.parquet`).
   - If `"next_poi"` (new preset): slot A loads poi_idx labels from `next_poi.parquet`.

6. **`scripts/train.py::_run_mtl_check2hgi`** — currently resolves only `task_b.num_classes` at runtime. Extend to also resolve `task_a.num_classes` from the fold tensors when the preset name is `check2hgi_next_poi_region`.

7. **Unit tests:**
   - `tests/test_data/test_next_poi_loader.py` — mirror `test_next_region_loader.py` (happy path, unmapped placeid, row-count mismatch).
   - `tests/test_tasks/test_presets.py` — extend with `CHECK2HGI_NEXT_POI_REGION` shape assertions.
   - `tests/test_training/test_mtl_cv_check2hgi.py` — add a variant that uses the new preset and asserts `val_accuracy_next_poi` + `val_accuracy_next_region` are emitted.

### P0.4 — End-to-end smoke on Alabama with the new preset

```bash
PYTHONPATH=src DATA_ROOT=/Volumes/Vitor\'s\ SSD/ingred/data \
OUTPUT_DIR=/Volumes/Vitor\'s\ SSD/ingred/output \
RESULTS_ROOT=/tmp/check2hgi_smoke \
  python scripts/train.py \
    --state alabama --engine check2hgi --task mtl \
    --task-set check2hgi_next_poi_region \
    --folds 1 --epochs 2 --gradient-accumulation-steps 1 \
    --batch-size 1024 --no-folds-cache
```

**Exit criterion:** exit 0; log shows `task_a=next_poi/<N_poi>, task_b=next_region/<N_region>`; one checkpoint written under `/tmp/check2hgi_smoke/check2hgi/alabama/checkpoints/`.

---

## Claims touched

- **CH04** — next-region meaningfulness; majority baselines logged in P0.3 pipeline output.
- **CH14** — fclass-shortcut audit. Resolved here.

Others (CH01, CH02, CH03, CH06, CH07, CH11) depend on P1+.

---

## Decision gate → P1

Proceed to P1 only when:

1. P0.1 passes on all 3 states.
2. P0.3 code deltas merged (all 7 steps); all 438+ existing tests stay green; new unit tests pass.
3. P0.4 smoke returns exit 0.
4. P0.2 (CH14 audit) resolved with a verdict (either `confirmed_by_construction` or arm-C ablation result documented).

If any step fails, pause + escalate. Do not proceed with the bug present.

---

## Output artefacts

```
docs/studies/check2hgi/results/P0/
├── integrity/
│   ├── alabama_check2hgi.json     # PI.1–PI.7 per-check results
│   ├── florida_check2hgi.json
│   └── arizona_check2hgi.json
├── fclass_audit/
│   └── ch14_audit_2026-04-XX.md    # either "not applicable" or arm-C results
└── next_poi_labels/
    ├── alabama_summary.json        # n_pois, majority fraction, class dist
    └── florida_summary.json
```
