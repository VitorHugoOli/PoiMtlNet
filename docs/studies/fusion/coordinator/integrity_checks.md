# Integrity Checks

These checks run at two points:
- **Preflight:** before launching a test, verify the inputs are sane.
- **Postflight:** after a test completes, verify the outputs are plausible and not corrupt.

Failed checks flag the test as `corrupt` in state.json and pause the coordinator.

---

## Preflight — input data checks

### PI.1 — Embedding files exist

For each engine in the test config, verify:
```
output/<engine>/<state>/embeddings.parquet         # must exist
output/<engine>/<state>/input/category.parquet     # for single-engine category tasks
output/<engine>/<state>/input/next.parquet         # for single-engine next tasks
output/fusion/<state>/input/category.parquet       # if engine=fusion
output/fusion/<state>/input/next.parquet           # if engine=fusion
```

**Fail** if any required file is missing.

### PI.2 — Schema shape

Load the parquet files and verify:

**Category (single-source DGI/HGI, 64D):**
- Columns include: `placeid`, `category`, `0`, `1`, ..., `63` (66 cols total)
- Shape: N × 66 where N > 1000 (at least)

**Category (fusion, 128D):**
- Columns include: `placeid`, `category`, `0`, `1`, ..., `127` (130 cols total)
- Shape: N × 130

**Next (single-source, 64D × 9 window):**
- Columns include: `emb_0` ... `emb_575`, `next_category`, `userid`
- Shape: M × 578 where M > 1000

**Next (fusion, 128D × 9 window):**
- Columns include: `emb_0` ... `emb_1151`, `next_category`, `userid`
- Shape: M × 1154

**Fail** on shape mismatch.

### PI.3 — Label range

Verify labels are in {0, 1, 2, 3, 4, 5, 6} (7 categories). No -1, no >6.

**Fail** on out-of-range.

### PI.4 — Label distribution (warning, not fail)

Compare observed class distribution to expected (per state). For Alabama reference:
```
{Food: ~32.5%, Shopping: ~31.3%, Community: ~15.0%,
 Entertainment: ~6.5%, Outdoors: ~6.1%, Travel: ~6.0%, Nightlife: ~2.5%}
```

**Warn** if any class deviates > 20% from expected (may indicate a new bug).

### PI.5 — Embedding value sanity

For embedding columns:
- No NaN, no Inf.
- L2 norm per row: report (mean, std, min, max). Warn if mean < 0.1 or > 100 (likely indicates scaling issue).

**Fail** on NaN/Inf.

### PI.6 — Expected scale ratios (fusion only)

On fusion inputs:
- Compute L2 norm of first 64 dims (Sphere2Vec) vs last 64 dims (HGI) for category task.
- Expected ratio: ~15:1.
- Compute same for next (HGI vs Time2Vec). Expected ratio: ~8.7:1.

**Warn** if ratio has changed substantially from expected (> 50% deviation) — may indicate an embedding regression.

### PI.7 — Cross-consistency between engines

For the same state:
- `placeid` set in category.parquet (DGI) should be equal to `placeid` set in category.parquet (HGI) should be equal to `placeid` set in category.parquet (Fusion).
- `userid` set in next.parquet should match across engines.

**Fail** on mismatch — indicates data drift.

---

## Preflight — code checks

### PC.1 — Required loss in registry

For the test's `mtl_loss`, verify it exists in `src/losses/registry.py`.

**Fail** if missing.

### PC.2 — Required architecture in registry

For the test's `model_name`, verify it exists in model registry.

### PC.3 — Gradient-accumulation sanity

If `mtl_loss` ∈ {cagrad, aligned_mtl, pcgrad}, verify the test command includes `--gradient-accumulation-steps 1`. This is injected by the ablation runner, but manual commands could forget.

### PC.4 — Git state (warn, not fail)

Warn if working tree has uncommitted changes. Record current git commit in test metadata.

---

## Postflight — output checks

### PO.1 — Results dir exists and has expected structure

Expected layout after training:
```
results/<engine>/<state>/mtlnet_lr<...>_bs<...>_ep<...>_<timestamp>/
├── summary/
│   ├── full_summary.json      # must exist
│   └── plots/                 # optional
├── folds/
│   ├── fold1/                 # per-fold outputs
│   └── ...
├── metrics/                   # per-fold train/val CSVs
├── model/                     # checkpoints
└── diagnostics/               # optional
```

**Fail** if `summary/full_summary.json` is missing.

### PO.2 — full_summary.json schema

Verify required keys:
- `model.joint_score.{mean, std, min, max}`
- `model.loss.{mean, std, ...}`
- `next.f1.{mean, std, ...}`
- `next.accuracy.{mean, ...}`
- `category.f1.{mean, std, ...}`
- `category.accuracy.{mean, ...}`

**Fail** on missing keys.

### PO.3 — Metric plausibility

- All F1 values ∈ [0, 1]
- Joint score ∈ [0, 1]
- No NaN / Inf
- **Sanity floor:** joint > 0.05 (a degenerate model still outputs ~0.1 for 7-class random, so anything below 0.05 is suspicious)

**Fail** on out-of-range.
**Warn** if joint < 0.1 (likely training problem).

### PO.4 — Per-fold metrics exist

Verify `metrics/fold{1..5}_category_val.csv` and `metrics/fold{1..5}_next_val.csv` exist for a 5-fold run. Missing folds = incomplete run.

**Fail** if any fold missing.

### PO.5 — Reproducibility metadata

Verify the run's manifest (if produced) contains:
- seed
- model config
- optimizer config
- dataset hash (if available)
- git commit (if recorded)

**Warn** on missing metadata.

### PO.6 — Result within expected range for the claim

For tests with specific claim expectations (from the phase doc), compare observed joint F1 to expected range:
- **Match:** within expected range → `matches_hypothesis`
- **Directional match:** within 1 std of expected → `partial_match`
- **Outside:** flag `surprising`; pause coordinator

### PO.7 — Duplicate detection

Check state.json: a test with the same test_id and same seed must not already exist with status `completed`. If it does, compare results:
- If new result is within 0.01 joint of old: accept (minor variance).
- If new result differs substantially: flag `conflict`; pause.

---

## Archive-time checks

When copying to `docs/studies/results/`:

### PA.1 — Metadata completeness

The metadata.json copied alongside full_summary.json must include:
- test_id
- phase
- claim_ids (list)
- config (seed, model, optimizer, heads, embedding, state, folds, epochs)
- wall_clock_seconds
- timestamp
- git_commit
- verdict (from analysis)

### PA.2 — No accidental overwrite

If `docs/studies/results/<phase>/<test_id>/` exists, check hash of existing summary. If identical: skip copy (idempotent). If different: fail with warning.

---

## Implementation stub

These checks should live in `scripts/study/validate_inputs.py` and `scripts/study/validate_outputs.py`, callable as:

```bash
python scripts/study/validate_inputs.py --state alabama --engines dgi,hgi,fusion
python scripts/study/validate_outputs.py --run-dir results/fusion/alabama/mtlnet_lr1.0e-04_bs4096_ep50_20260414_1200
```

Returns exit 0 on pass, 1 on warn, 2 on fail.

---

## Known benign exceptions

These show up in logs but are NOT failures:

- **fvcore missing** → `[INFO] FLOPS: 0 | Params: 0` — no blocker, just no FLOPs reported.
- **MPS fallback warning for Aligned-MTL's eigh** — cosmetic, correctness unaffected.
- **`enable_nested_tensor` warning** for Transformer encoder — cosmetic.

Coordinator should whitelist these so they don't fire `surprising` flags.
