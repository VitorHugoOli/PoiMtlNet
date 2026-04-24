# Integrity Checks — Check2HGI Study

Preflight + postflight checks specific to this study. The general framework (what preflight / postflight means, when checks run, how failures pause the coordinator) is inherited from `docs/studies/fusion/coordinator/integrity_checks.md`.

---

## Preflight — input data checks

### PI.1 — Check2HGI artefacts exist

For each state in the test config:
```
output/check2hgi/<state>/embeddings.parquet         # check-in-level, [N_checkins, 64] + metadata
output/check2hgi/<state>/poi_embeddings.parquet     # POI-aggregated, [N_pois, 64] + placeid
output/check2hgi/<state>/region_embeddings.parquet  # Region-aggregated, [N_regions, 64] + region_id
output/check2hgi/<state>/input/next.parquet         # Next-POI sequence X, [N_seq, 576 + next_category + userid]
output/check2hgi/<state>/input/next_region.parquet  # [N_seq, 576 + region_idx + userid]  (already exists)
output/check2hgi/<state>/input/next_poi.parquet     # [N_seq, 576 + poi_idx + userid]  (will exist after P0 code-delta task)
output/check2hgi/<state>/temp/checkin_graph.pt      # pickled dict: placeid_to_idx, poi_to_region, num_*
```

**Fail** if any required file is missing.

### PI.2 — Schema shapes

**`next.parquet`:** `N × 578` columns. First 576 are digit-named `"0".."575"` (flattened 9-window × 64-dim check-in embeddings). Then `next_category` (string) + `userid`.

**`next_region.parquet`:** `N × 578` columns. 576 numeric + `region_idx` (int64) + `userid`. Row-aligned with `next.parquet` — see `src/data/inputs/next_region.py::build_next_region_frame` for the validation logic.

**`next_poi.parquet`:** `N × 578` columns. 576 numeric + `poi_idx` (int64) + `userid`. Same row-alignment contract.

**`embeddings.parquet`:** `[userid, placeid, category, datetime, "0".."63"]`, 68 cols total.

**`poi_embeddings.parquet`:** `[placeid, "0".."63"]`, 65 cols total.

**`region_embeddings.parquet`:** `[region_id, "reg_0".."reg_63"]`, 65 cols total.

**Fail** on shape mismatch.

### PI.3 — Label range

- `region_idx` must be in `[0, num_regions)` (strict). `num_regions` read from `checkin_graph.pt["num_regions"]`.
- `poi_idx` must be in `[0, num_pois)` (strict). Read from `checkin_graph.pt["num_pois"]`.
- No `-1`, no NaN, no negative values.

**Fail** on out-of-range.

### PI.4 — Label distribution (warning, not fail)

Per-state majority-class fractions we expect:

| State | next_region majority | next_poi majority |
|---|---|---|
| Alabama | ~2.3% | (to measure in P0) |
| Florida | ~22.5% | (to measure in P0) |
| Arizona | ~6.4% | (to measure in P0) |

**Warn** if observed majority deviates > 30% from expected (suggests data-regeneration drift).

### PI.5 — Embedding value sanity

On each of `embeddings.parquet`, `poi_embeddings.parquet`, `region_embeddings.parquet`:
- No NaN, no Inf.
- L2 norm per row: report (mean, std, min, max). Warn if mean < 0.1 or > 100.

**Fail** on NaN/Inf.

### PI.6 — Cross-artefact consistency

- `len(embeddings.parquet) == checkin_graph.pt["num_checkins"]`
- `len(poi_embeddings.parquet) == checkin_graph.pt["num_pois"]`
- `len(region_embeddings.parquet) == checkin_graph.pt["num_regions"]`
- `set(placeid_to_idx.keys()) == set(embeddings.parquet["placeid"].unique())`
- `len(next.parquet) == len(next_region.parquet)` — row-alignment contract for the MTL fold creator
- `len(next.parquet) == len(next_poi.parquet)` — same contract for next_poi
- `next.parquet["userid"].astype(str) == next_region.parquet["userid"].astype(str)` row-wise (dtype-normalised; see `src/data/inputs/next_region.py` for the comment on why casting is needed)

**Fail** on any cross-artefact mismatch.

### PI.7 — Graph artefact freshness

`checkin_graph.pt` must be readable with `pickle.load` (not torch.load — it's a `pkl.dump` output, see `research/embeddings/check2hgi/preprocess.py`). Required keys: `placeid_to_idx`, `poi_to_region`, `num_checkins`, `num_pois`, `num_regions`, `edge_index`, `edge_weight`, `checkin_to_poi`, `region_adjacency`.

**Fail** if pickle load fails (corruption — observed once previously when a training run was killed mid-write).

---

## Preflight — code-side checks

### PC.1 — Required presets registered

At the moment of training launch, verify:
- `tasks.get_preset("check2hgi_next_poi_region")` returns a valid TaskSet.
- `tasks.get_preset("check2hgi_next_region")` still returns (legacy preset, preserved).
- `task_set.task_a.is_sequential is True` for the new preset.
- `task_set.task_b.is_sequential is True`.

### PC.2 — Required head in registry

- `create_model("next_mtl", ...)` resolves.

### PC.3 — Required loss in registry

- `create_loss("nash_mtl", n_tasks=2, ...)` resolves.
- If `--use-class-weights` is on, `compute_class_weights` handles absent classes (test via synthetic input; done in `tests/test_training/test_mtl_cv_check2hgi.py`).

---

## Postflight — output integrity

### PO.1 — Results parquet / JSON complete

After a run, verify:
- `results/P<phase>/<test_id>/full_summary.json` exists
- `results/P<phase>/<test_id>/metadata.json` exists (git commit, CLI command, seed, config hash)
- Per-fold summaries exist (5 for 5-fold runs)

**Fail** if any is missing.

### PO.2 — Per-fold metrics in range

- `next_poi_acc1` in `[0, 1]`
- `next_region_acc1` in `[0, 1]`
- `next_poi_mrr` in `[0, 1]`
- `joint_lift >= 0` (can exceed 1.0)

**Fail** if values are NaN or out of range.

### PO.3 — Monitor-vs-checkpoint consistency

The saved best-checkpoint's `val_joint_lift` must match the max `val_joint_lift` across epochs in the training history (within ε=1e-6). Guards against `ModelCheckpoint(monitor="...")` typo where the callback silently saves no checkpoint.

**Fail** on inconsistency.

---

## Scale-ratio check (NOT applicable)

The fusion study's `PI.6` checks the half-L2 ratio between Sphere2Vec and HGI in the fusion engine. Check2HGI is a single-engine embedding — no fusion ratio exists. This check is explicitly **skipped**.

---

## fclass-shortcut audit (one-shot, P0 only)

See CH14. The audit runs outside the regular integrity checks because it's an ablation, not a shape-validation check. Procedure:

1. Inspect `research/embeddings/check2hgi/preprocess.py` for POI2Vec dependency. If POI2Vec is not used, CH14 resolves to `confirmed_by_construction` and the arm C ablation is skipped.
2. If POI2Vec IS used: regenerate Check2HGI embeddings with an `fclass_shuffle_seed` applied during preprocessing (same mechanism as fusion's `scripts/hgi_leakage_ablation.py` arm C). Compare next_poi Acc@10 drop to the fusion study's Cat F1 drop (−60+ pp). If the drops are similar, shortcut present; if not, shortcut avoided.
