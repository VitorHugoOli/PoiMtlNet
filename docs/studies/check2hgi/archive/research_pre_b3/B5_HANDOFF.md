# B5 Handoff — Hard-Index GETNext Retraining

**Target machine:** Linux 4050 (currently idle).
**Reads:** commit `6a2f808` (`feat(b5): faithful GETNext with hard last_region_idx`).
**Related docs:**
- `B5_HARD_VS_SOFT_INFERENCE.md` — inference-time evidence motivating this retraining
- `B5_IMPLEMENTATION_PLAN.md` — original design plan (now largely superseded by the actual commit; keep for history)
- `B5_PROBE_ENTROPY_FINDINGS.md` — why the soft probe is suboptimal

## TL;DR for the agent on 4050

1. Pull the latest `worktree-check2hgi-mtl` branch.
2. Regenerate `next_region.parquet` for AL + AZ (+ FL if time) — one-liner.
3. Run 5-fold × 50-epoch MTL with the new `next_getnext_hard` head on AL and AZ.
4. Optional: FL 1f × 50ep for headline-state coverage.
5. Archive the summary JSONs under `docs/studies/check2hgi/results/B5/`.
6. Push. The Mac-side agent will analyse the results.

Budget: ~15 min AL + ~30 min AZ + ~1h FL 1f on a 4050 ≈ **~1h45m total** for the core pair.

## Context: what changed in the code

Every change is additive. Existing heads / models / runners are untouched.

| File | Change |
|---|---|
| `src/data/inputs/next_region.py` | `build_next_region_frame` now emits `last_region_idx` column |
| `scripts/regenerate_next_region.py` | CLI to regenerate a state's `next_region.parquet` additively |
| `src/data/folds.py` | Added `POIDatasetWithAux` + `_create_aux_dataloader`; in `_create_check2hgi_mtl_folds`, gated branch that uses them when `task_set.task_b.head_factory == "next_getnext_hard"` |
| `src/data/aux_side_channel.py` | Thread-local aux publisher + `AuxPublishingLoader` wrapper (strips aux, publishes to thread-local, yields `(x, y)` 2-tuples so `mtl_cv.py` and `mtlnet_crossattn` forward are unchanged) |
| `src/models/next/next_getnext_hard/` | New head reading aux via `get_current_aux()`; falls back to pure STAN on missing/pad/out-of-bounds aux |
| `src/models/next/__init__.py` | Registers the new head |
| `scripts/eval_hard_vs_soft_region_idx.py` | Minor correctness fix: rows with `last_region >= num_regions` now zero-prior instead of silently clipping |

## Setup on 4050

```bash
cd /path/to/ingred/.claude/worktrees/check2hgi-mtl   # or wherever the worktree lives on 4050
git pull
```

Required env vars (adjust paths to 4050's data layout):

```bash
export DATA_ROOT=/path/to/check2hgi_data
export OUTPUT_DIR=/path/to/check2hgi_data
export PYTHONPATH=src
export PYTORCH_ENABLE_MPS_FALLBACK=1        # harmless on CUDA
export CUDA_VISIBLE_DEVICES=0                # 4050
PY=/path/to/.venv/bin/python                 # adjust
```

## 1. Regenerate parquets (additive column)

The new `last_region_idx` column needs to be added to each state's
`next_region.parquet`. Existing parquets still work with every pre-B5
code path — the column is read by name, ignored elsewhere.

```bash
$PY scripts/regenerate_next_region.py --state alabama
$PY scripts/regenerate_next_region.py --state arizona
$PY scripts/regenerate_next_region.py --state florida   # only if running FL
```

Expected output per state (AL shown):
```
[alabama] rows=12709  n_regions=1109  pad_rows (last_region=-1): 0 (0.0%)
[alabama] wrote → /path/to/check2hgi_data/check2hgi/alabama/input/next_region.parquet
```

Sanity-check the new column exists:
```bash
$PY -c "import pandas as pd; df=pd.read_parquet('$OUTPUT_DIR/check2hgi/alabama/input/next_region.parquet'); print(df.columns.tolist()[-4:]); print('last_region_idx dtype:', df.last_region_idx.dtype); print('negatives:', (df.last_region_idx<0).sum())"
```

## 2. Launch the B5 training runs

**Alabama — 5-fold × 50-epoch (headline):**

```bash
$PY -u scripts/train.py \
    --task mtl --task-set check2hgi_next_region \
    --state alabama --engine check2hgi \
    --folds 5 --epochs 50 --seed 42 \
    --task-a-input-type checkin --task-b-input-type region \
    --model mtlnet_crossattn --mtl-loss pcgrad \
    --reg-head next_getnext_hard \
    --reg-head-param d_model=256 --reg-head-param num_heads=8 \
    --reg-head-param transition_path=$OUTPUT_DIR/check2hgi/alabama/region_transition_log.pt \
    --max-lr 0.003 \
    --gradient-accumulation-steps 1 --no-checkpoints \
    > /tmp/b5_al.log 2>&1
```

**Arizona:**

```bash
$PY -u scripts/train.py \
    --task mtl --task-set check2hgi_next_region \
    --state arizona --engine check2hgi \
    --folds 5 --epochs 50 --seed 42 \
    --task-a-input-type checkin --task-b-input-type region \
    --model mtlnet_crossattn --mtl-loss pcgrad \
    --reg-head next_getnext_hard \
    --reg-head-param d_model=256 --reg-head-param num_heads=8 \
    --reg-head-param transition_path=$OUTPUT_DIR/check2hgi/arizona/region_transition_log.pt \
    --max-lr 0.003 \
    --gradient-accumulation-steps 1 --no-checkpoints \
    > /tmp/b5_az.log 2>&1
```

**Florida (optional, 1 fold × 50 epoch — paper headline state):**

```bash
$PY -u scripts/train.py \
    --task mtl --task-set check2hgi_next_region \
    --state florida --engine check2hgi \
    --folds 1 --epochs 50 --seed 42 \
    --task-a-input-type checkin --task-b-input-type region \
    --model mtlnet_crossattn --mtl-loss pcgrad \
    --reg-head next_getnext_hard \
    --reg-head-param d_model=256 --reg-head-param num_heads=8 \
    --reg-head-param transition_path=$OUTPUT_DIR/check2hgi/florida/region_transition_log.pt \
    --max-lr 0.003 \
    --gradient-accumulation-steps 1 --no-checkpoints \
    > /tmp/b5_fl.log 2>&1
```

**Baseline reference (MTL-GETNext soft probe, PCGrad, 5f × 50ep):**
- AL: Acc@10_indist 56.38 ± 4.11, MRR 29.07 ± 2.43
- AZ: Acc@10_indist 47.34 ± 2.93, MRR 24.16 ± 1.92
- FL (1f): Acc@10_indist 60.62

The B5 runs above use **identical** configs except `--reg-head next_getnext`
→ `next_getnext_hard`. Any deviation from the soft-probe numbers is the
B5 effect. Inference-time analysis predicts **+3 to +9 pp Acc@10**.

## 3. Archive the result summaries

After each run completes, copy the summary JSON to a persistent location
so the Mac-side analyst can read it:

```bash
mkdir -p docs/studies/check2hgi/results/B5
latest=$(ls -dt results/check2hgi/alabama/mtlnet_lr*_bs*_ep50_* | head -1)
cp "$latest/summary/full_summary.json" docs/studies/check2hgi/results/B5/al_5f50ep_next_getnext_hard.json

latest=$(ls -dt results/check2hgi/arizona/mtlnet_lr*_bs*_ep50_* | head -1)
cp "$latest/summary/full_summary.json" docs/studies/check2hgi/results/B5/az_5f50ep_next_getnext_hard.json

# FL if run
latest=$(ls -dt results/check2hgi/florida/mtlnet_lr*_bs*_ep50_* | head -1)
cp "$latest/summary/full_summary.json" docs/studies/check2hgi/results/B5/fl_1f50ep_next_getnext_hard.json
```

## 4. Push

```bash
git add docs/studies/check2hgi/results/B5/
git commit -m "study(b5): retrained MTL-GETNext with hard last_region_idx (AL+AZ[+FL])"
git push
```

## 5. Expected outputs & Mac-side follow-up

Each `full_summary.json` has the structure the `compute_fold_acc10.py`
helper (at `/tmp/compute_fold_acc10.py` on the Mac) already consumes:

```
  next_region.top10_acc_indist = {mean, std, min, max}
  next_region.mrr_indist       = {...}
  next_region.f1               = {...}
  ...
```

On the Mac, the analyst will compute the delta vs the soft-probe baseline
table, and write up under `docs/studies/check2hgi/research/B5_RESULTS.md`.

## Sanity checks for the 4050 agent before launching

1. **Regression test (fast, ~3s):** `$PY -m pytest tests/test_regression/test_mtl_param_partition.py tests/test_regression/test_mtlnet_crossattn_partial_forward.py -q` — should print `21 passed`.
2. **Smoke test (2-fold, 1 epoch, ~25s on CPU, ~15s on GPU):** run the same command as §2 but with `--folds 2 --epochs 1`. Should complete without error. Region accuracy will be ~0 (1 epoch isn't enough to train 1109 classes); the goal is integration, not convergence.
3. **Verify aux is flowing:** after any training run, log should contain the line `MTL_CHECK2HGI input modality: task_a=checkin ((..., 9, 64)), task_b=region ((..., 9, 64))` AND training should complete with non-trivial region numbers by epoch 50.

If the smoke test fails: do NOT launch the full 5-fold runs. Report
the error back to the Mac-side analyst (the agent may need to fix a
regression).

## What NOT to do on 4050

- **Don't modify `mtl_cv.py`, `mtlnet_crossattn/model.py`, or the existing `next_getnext` head.** B5 is designed to work without touching any of them.
- **Don't modify `next_region.parquet` schema further.** The `last_region_idx` column addition is complete.
- **Don't remove the soft `next_getnext` head.** The paper wants both rows (soft and hard) in the final table.
- **Don't launch multiple runs concurrently on the 4050.** The MTL pipeline allocates ~1-2 GB GPU and doesn't share well; sequential is faster and more stable.

## Notes on reproducibility

- Same seed (42) as the soft-probe champion run. Same hyperparams.
  The only variable is the head type.
- If you want to measure variance from MTL-optimizer choice, also run
  `--mtl-loss static_weight` as a second seed (PCGrad ≈ static per
  `ATTRIBUTION_PCGRAD_VS_STATIC.md`, but the replicate is cheap).
- Multi-seed (42, 123, 2024) is documented in
  `phases/P7_headline_states.md §11` and remains an open follow-up.
