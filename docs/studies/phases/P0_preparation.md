# Phase 0 — Preparation

**Goal:** verify new data, build observability tooling, bring the codebase to a known-good state. Nothing scientific happens in P0 — only infrastructure.

**Duration:** 2-4 h (mostly waiting on embedding regeneration).

---

## Preconditions

- User has regenerated embeddings on the fixed-label dataset.
- User has pushed the regenerated embeddings to disk (`output/{engine}/{state}/embeddings.parquet` and corresponding fusion inputs).

## Deliverables

1. `docs/studies/state.json` initialized with all phases listed as `planned` (P1-P5).
2. Integrity-check scripts under `scripts/study/`.
3. `/study` Skill functional (even if minimal).
4. Verified: **the regenerated embeddings produce sensible numbers on one sanity-check run** (see P0.4 below).

---

## Steps

### P0.1 — Wait / confirm embeddings regeneration

For each state we plan to use (minimum: AL, AZ, FL):

```bash
# Confirm files exist
ls output/dgi/<state>/embeddings.parquet
ls output/hgi/<state>/embeddings.parquet
ls output/sphere2vec/<state>/embeddings.parquet
ls output/time2vec/<state>/embeddings.parquet

# Confirm fusion inputs were regenerated (not carried over)
stat output/fusion/<state>/input/category.parquet
stat output/fusion/<state>/input/next.parquet
# timestamps should be after the bug fix
```

**Owner:** user (runs embedding pipelines).

**Blocker for:** everything downstream.

---

### P0.2 — Data integrity validation

Write `scripts/study/validate_inputs.py` that checks:

1. **Category parquet:**
   - Shape: N rows × 130 cols (placeid, category, emb_0..emb_127)
   - Unique categories ∈ {0, 1, 2, 3, 4, 5, 6}
   - No NaN/Inf in embeddings
   - L2 norm distribution: report mean, std, min, max (spot-check it looks reasonable)
   - Class distribution within 20% of expected (Alabama class distribution below for reference)

2. **Next parquet:**
   - Shape: N rows × (576 + 2) for fusion-next — nope, 9 × 128 = 1152 for fusion, 9 × 64 = 576 for single-source
   - Target label (`next_category`) ∈ {0..6}
   - userid present
   - No NaN/Inf in sequence embeddings

3. **Cross-reference:**
   - For the same state, DGI / HGI / Fusion should have the same set of POI ids for category task
   - For single-source, HGI pois should be subset of Fusion category pois
   - For Next, user-id sets should match between single-source and fusion

4. **Expected ranges (Alabama reference from literature):**
   ```
   Category counts ≈ { Food: 3800, Shopping: 3660, Community: 1750,
                       Entertainment: 760, Outdoors: 720, Travel: 710, Nightlife: 290 }
   ```
   Run with `--reference alabama` to check against this; emit warnings for >20% deviations.

**Output:** JSON report per state. Commit report to `docs/studies/results/P0/integrity/<state>.json`.

**Phase gate:** cannot proceed to P1 until all planned states pass integrity checks.

---

### P0.3 — State file initialization

Create `docs/studies/state.json` with:

```json
{
  "study_version": "1.0",
  "started": "<timestamp>",
  "current_phase": "P0",
  "phases": {
    "P0": { "status": "running" },
    "P1": { "status": "planned", "tests": {} },
    "P2": { "status": "planned", "tests": {} },
    "P3": { "status": "planned", "tests": {} },
    "P4": { "status": "planned", "tests": {} },
    "P5": { "status": "planned", "tests": {} }
  },
  "open_issues": []
}
```

Each phase's `tests` dict gets populated when the phase starts, from the corresponding phase doc's experiment list.

---

### P0.4 — Sanity run on clean data

Run ONE configuration end-to-end to confirm the pipeline works with new embeddings:

```bash
# On Alabama, the prior-best CBIC config — should roughly match published results (47-48% cat F1)
python scripts/train.py --task mtl --state alabama --engine dgi \
  --epochs 50 --folds 5 --model mtlnet --mtl-loss nash_mtl --seed 42
```

**Expected:** Cat F1 = 46-48%, Next F1 = 26-28% (CBIC published numbers, ±1-2 p.p. variance from new embedding).

**If actual deviates by > 5 p.p. from CBIC:** something is wrong — either the embedding regen has an issue, the codebase has drifted, or the labels still have a bug. **Stop and investigate.**

**Archive:** results go to `docs/studies/results/P0/sanity/AL_dgi_cbic/`.

---

### P0.5 — Build minimal observability tooling

Minimum viable scripts in `scripts/study/`:

#### `validate_inputs.py`
Runs the P0.2 checks on a given state+engine combo.

#### `archive_result.py`
Given a results directory from `scripts/train.py`, extracts `full_summary.json` and metadata, places them under `docs/studies/results/<phase>/<test_id>/`.

#### `launch_test.py`
Reads `state.json`, finds the next `planned` test, launches it, updates state to `running` with metadata.

#### `analyze_test.py`
Given a test_id, runs analysis: compares observed vs expected from the claim, updates claim status.

All scripts are thin — most of the logic lives in the phase docs and the claim catalog. These scripts just enforce the workflow.

---

### P0.6 — Build `/study` Skill

Create a Claude Code Skill (a command at `.claude/commands/study.md` or similar) that wraps the above scripts. See `COORDINATOR.md` for the action list.

**Minimum viable skill actions:**
- `status` — read state.json, print current phase + summary of pending/running/completed tests
- `next` — launch the next pending test
- `import <dir>` — archive a result from a foreign run (other machine)
- `validate` — run integrity checks on current phase

Advanced actions (can come later):
- `advance`, `analyze`, `claim`

---

### P0.8 — Freeze fold indices (methodological prerequisite)

**Why:** every test in P1–P6 is a paired comparison against other tests on the same (state, engine) pair. Paired statistical tests (Wilcoxon signed-rank, paired t) require **byte-identical train/val splits** across the models being compared (Dietterich 1998; Raschka 2018). `StratifiedGroupKFold(random_state=42)` is deterministic today but can shift silently across sklearn minor versions and across input-parquet regenerations. Freezing the fold indices once and loading them everywhere removes both risks and is a precondition for C28 (no-negative-transfer) and every paired claim in the catalog.

**References:** Dietterich 1998 (5×2cv paired-t origin), Raschka 2018 "Model Evaluation…" (arXiv:1811.12808), sklearn Common Pitfalls ("Controlling Randomness"), NeurIPS Paper Checklist ("state which factors of variability error bars capture").

**Implementation:**

1. Write `scripts/study/freeze_folds.py`:
   - For each (state, engine) in the study plan: call `FoldCreator(seed=42).create_folds(state, engine)`, then `.save(Path("output") / engine / state / "folds")`.
   - Tag the saved dict with the `DatasetSignature` (streaming SHA-256 from `src/configs/experiment.py`) of the input parquets so the cache invalidates loudly if inputs change.
   - Writes `output/{engine}/{state}/folds/fold_indices.pt` and a companion `.meta.json` with the signature.

2. Plumb `load_folds()` into `scripts/train.py`:
   - Add optional `--folds-path` flag.
   - When present, use `rebuild_dataloaders()` instead of creating folds from scratch.
   - When absent but a cached file exists at the canonical path, load it (with signature check). Otherwise fall back to on-the-fly generation (and warn).

3. Update every test config in `state.json` to use `seed: 42` (already the default but verify).

4. For **C18 multi-seed robustness** (P5.1): only `torch.manual_seed` varies across the {42, 123, 2024} runs. Folds stay frozen. This isolates model-stochasticity variance from fold-split variance — exactly what NeurIPS asks you to state explicitly.

**Freeze order:** AL first (fastest path to unblock P1), then AZ, then FL. For each state × engine in {dgi, hgi, fusion, sphere2vec, time2vec, poi2vec}.

**Output:** JSON manifest at `docs/studies/results/P0/folds/frozen.json` listing every (state, engine) that has a cached fold file, plus the input signature and fold-size summary.

**Phase gate:** any test enrolled in P1 must point at cached folds. A test that regenerates folds from scratch fails P0.8 review.

---

### P0.9 — Phase 0 exit criteria

Tick each box:

- [ ] Embeddings regenerated for at least AL + AZ (FL optional for P0 but needed by P3)
- [ ] P0.2 integrity validation passes on all target states
- [ ] P0.4 sanity run completes within expected range (or investigation concluded)
- [ ] P0.5 scripts present and runnable
- [ ] P0.6 `/study status` works and returns state.json summary
- [ ] P0.8 fold indices frozen for AL + AZ × {dgi, hgi, fusion} at minimum
- [ ] This phase doc's status in state.json is `completed`
- [ ] A git commit captures the P0 outputs

**Only after all boxes are checked:** proceed to P1.

---

## Open questions during P0

These are not blockers but should be tracked:

1. **Do we have fusion inputs for AZ?** If not, generate during P0.
2. **Does the codebase still have the T0.2 gradient-accumulation fix?** Verify `src/ablation/runner.py::_candidate_argv` still injects `--gradient-accumulation-steps 1` for `cagrad`/`aligned_mtl`/`pcgrad`. (It should — we committed this.)
3. ~~**Are the data splits stable across embedding regeneration?** If user IDs change (e.g., because a user was removed due to the label bug), stratified folds may differ. Decide whether to freeze the split mapping now.~~ **Resolved by P0.8** — folds are frozen once post-regeneration and loaded by every downstream test.
