# Check2HGI Track — Quick Reference

Scannable cheat-sheet. For rationale/depth, see `README.md` and `CLAIMS_AND_HYPOTHESES.md`.

## Claims at a glance

- **CH01:** check2HGI > HGI on next-category (embedding claim).
- **CH02:** 2-task MTL {next_cat, next_region} > single-task next_cat on check2HGI (MTL lift — headline).
- **CH03:** no per-head regression under MTL (negative-transfer control).
- **CH04:** next_region is a meaningful task (not a noise-injection regulariser).
- **CH05:** ranking metrics differentiate where macro-F1 collapses.
- **CH06:** monitor choice (`joint_acc1`) stable.
- **CH07:** `next_mtl` > other seq heads on region.
- **CH08:** NashMTL vs naive vs CAGrad on this task pair.
- **CH09:** task embedding contributes.
- **CH10:** FSQ-NYC/TKY limitation (declared).
- **CH11:** enrichment deferred (declared).

Full catalog: `CLAIMS_AND_HYPOTHESES.md`.

## Presets

From `src/tasks/presets.py`:

```python
LEGACY_CATEGORY_NEXT        # (category, next)   flat/seq,  F1/F1
CHECK2HGI_NEXT_REGION       # (next_cat, region) seq/seq,   F1/Acc@1
```

## CLI — track runs

> These are planned commands; the `--task-set` flag is being added in P1.

```bash
# Generate embeddings (P-1)
python pipelines/embedding/check2hgi.pipe.py          # runs states in STATES dict

# Derive next_region labels (P0, after P-1 finishes)
python pipelines/create_inputs_check2hgi.pipe.py --state florida

# P2 — single-task baselines
python scripts/train.py --state alabama --engine check2hgi --task next --folds 5 --epochs 50

# P3 — 2-task MTL headline
python scripts/train.py --state alabama --engine check2hgi --task mtl --task-set check2hgi_next_region --folds 5 --epochs 50

# P4 — ablations: head swap, MTL optimiser, etc. (configured via ExperimentConfig overrides)
```

## Monitors

- Legacy track (unchanged): `val_f1_category`.
- Check2HGI track: `val_joint_acc1 = mean(val_accuracy_next_category, val_accuracy_next_region)`.
- Always reported alongside: `val_joint_f1`, per-head Acc@{1,5,10}, MRR.

## State paths

- Embeddings: `output/check2hgi/{state}/{embeddings,poi_embeddings,region_embeddings}.parquet`.
- Next-POI sequences (already produced by embedding pipeline): `output/check2hgi/{state}/temp/sequences_next.parquet`.
- Next-region inputs (to be produced by P0): `output/check2hgi/{state}/input/next_region.parquet`.
- Region map (in graph artifact): `output/check2hgi/{state}/temp/checkin_graph.pt` → `poi_to_region` key.

## Env vars

- `DATA_ROOT` — defaults to `<project>/data` (use main-repo data dir on this worktree).
- `OUTPUT_DIR` — defaults to `<project>/output`.
- `RESULTS_ROOT` — defaults to `<project>/results`.

## Gates between phases

- P-1 → P0: embedding convergence check + all 3 parquet files exist per state.
- P0 → P2: `next_region.parquet` with zero unmapped placeids; region cardinality logged.
- P1 → P2: `pytest tests/test_regression -q` green; legacy smoke reproduces seed-42 metric bit-exactly.
- P2 → P3: CH01, CH04, CH05 resolved.
- P3 → P4: CH02, CH03 resolved.
- P4 → merge: ablations resolved; `PAPER_FINDINGS.md` draft updated.
