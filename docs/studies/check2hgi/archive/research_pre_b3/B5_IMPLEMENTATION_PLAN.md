# B5 — Retraining with Hard `last_region_idx`: Implementation Plan

**Status:** NOT STARTED. Ready to launch when MPS is free (after partition bugfix rerun finishes) or on Linux 4050.
**Motivating evidence:** `B5_HARD_VS_SOFT_INFERENCE.md` — hard index beats soft probe by +20 pp Acc@10 at inference without retraining.

## Goal

Train MTL-GETNext end-to-end with the faithful hard `last_region_idx`
prior (as in GETNext's original SIGIR 2022 formulation), rather than the
soft-probe adaptation we currently use.

Target metric lifts (from `B5_HARD_VS_SOFT_INFERENCE.md §Implication`,
revised with epoch-46 evidence — earlier epoch-9 estimate was inflated):

| State | Current | Conservative B5 | Optimistic |
|---|---:|---:|---:|
| AL | 56.38 | 59 ± 3 (+3 pp) | 63 ± 3 (+7 pp) |
| AZ | 47.34 | 50 ± 2 (+3 pp) | 54 ± 2 (+7 pp) |
| FL | 60.62 | 63 (+3 pp) | 67 (+7 pp) |

## Implementation checklist (additive — safe while partition rerun is active)

### 1. Data pipeline — extend `next_region.parquet` additively

File: `src/data/inputs/next_region.py`

- [ ] In `build_next_region_frame`, after computing `region_idx`, also
  derive `last_region_idx` from `sequences_next.parquet` `poi_{0..8}`
  (last non-pad position) + `placeid_to_idx` + `poi_to_region`.
  Use `-1` sentinel for all-pad rows.
- [ ] Append `last_region_idx` column to the output DataFrame.
- [ ] Regenerate `next_region.parquet` for AL + AZ + FL via a small CLI.

**Safety:** purely additive — readers that select by column name
(`region_df["region_idx"]`) are unaffected. Verified existing code path
does this (see `folds.py::_create_check2hgi_mtl_folds` line ~793).

### 2. Dataset wrapper — carry aux column

File: `src/data/dataset.py` (NEW class, don't modify `POIDataset`)

- [ ] Add `POIDatasetWithAux(features, labels, aux)` that yields
  `(features, labels, aux)` 3-tuples from `__getitem__`.
- [ ] Keep `POIDataset` unchanged so existing runners keep working.

### 3. Fold creator — gate behind input-type flag

File: `src/data/folds.py` (minimally extend `_create_check2hgi_mtl_folds`)

- [ ] Read the new column when `task_b_head_factory == "next_getnext_hard"`.
  Pass the aux tensor to `POIDatasetWithAux` instead of `POIDataset`.
- [ ] Old code path (other heads) continues to use `POIDataset` exactly
  as today.

**Risk:** touches shared file. Acceptable if change is gated on head
name and tested to be no-op for other heads. Add a unit test.

### 4. New head

File: `src/models/next/next_getnext_hard/head.py` (NEW)

```python
@register_model("next_getnext_hard")
class NextHeadGETNextHard(NextHeadSTAN):
    def __init__(self, ..., transition_path, alpha_init=0.1):
        super().__init__(...)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        log_T = torch.load(transition_path, ...)["log_transition"].float()
        self.register_buffer("log_T", log_T[:num_classes, :num_classes])

    def forward(self, x, last_region_idx):  # NEW aux arg
        stan_logits = self.stan(x)
        # For pad rows (last_region_idx == -1), zero prior
        mask = last_region_idx < 0
        safe_idx = last_region_idx.clamp(min=0)
        prior = self.log_T[safe_idx]
        prior[mask] = 0.0
        return stan_logits + self.alpha * prior
```

### 5. Model wrapper — accept aux in forward

File: `src/models/mtl/mtlnet_crossattn_aux/model.py` (NEW class)

- [ ] Subclass `MTLnetCrossAttn` (don't modify base). Override `forward`
  to accept `(inputs, aux)` and pass `aux` to `self.next_poi`.

### 6. New runner

File: `src/training/runners/mtl_cv_aux.py` (NEW; copy of `mtl_cv.py`)

- [ ] Unpack 3-tuples in the training loop.
- [ ] Pass aux to `model(inputs, aux)`.
- [ ] Otherwise identical to `mtl_cv.py`.

### 7. Dispatcher

File: `scripts/train.py`

- [ ] Add `"mtl_aux": _run_mtl_aux` to `_RUNNERS` (additive).
- [ ] `_run_mtl_aux` dispatches to the new runner when the task_b head
  is `next_getnext_hard`.

### 8. Regression test

File: `tests/test_training/test_mtl_cv_aux.py` (NEW)

- [ ] End-to-end smoke test: 1 fold, 1 epoch, verify loss is finite and
  last_region_idx flows through to the head.

## Commands to run once implemented

```bash
# 1. Regenerate next_region.parquet additively (AL + AZ)
python pipelines/create_inputs_check2hgi.pipe.py --state alabama --refresh
python pipelines/create_inputs_check2hgi.pipe.py --state arizona --refresh

# 2. Launch B5 retraining — AL (on 4050 or M2 Pro post-rerun)
python scripts/train.py \
    --task mtl --task-set check2hgi_next_region \
    --state alabama --engine check2hgi \
    --folds 5 --epochs 50 --seed 42 \
    --task-a-input-type checkin --task-b-input-type region \
    --model mtlnet_crossattn_aux --mtl-loss pcgrad \
    --reg-head next_getnext_hard \
    --reg-head-param d_model=256 --reg-head-param num_heads=8 \
    --reg-head-param transition_path=.../region_transition_log.pt \
    --max-lr 0.003 \
    --gradient-accumulation-steps 1 --no-checkpoints

# 3. Same for AZ + FL
```

Budget: ~15 min AL + ~25 min AZ + ~1h FL 1f on MPS. Total ~2h training
once implementation lands.

## Implementation budget

~4–6 h coding (steps 1–8) + ~2 h training + ~1 h writing up. Total ~7–9 h.

## Conflict with running rerun_partition_bugfix.sh

The rerun uses `mtlnet_dselectk` + `next_gru` (default task-b head via
task_set). It does NOT use `next_getnext`, `next_getnext_hard`, or
`mtlnet_crossattn_aux`. All B5 changes are orthogonal except the
`folds.py` extension in step 3 — gate that change on
`task_b_head_factory == "next_getnext_hard"` so rerun subprocesses
that don't use that head hit the existing code path unchanged. A unit
test should pin this.

The `next_region.parquet` schema extension (step 1) is column-additive,
backward-compatible — the rerun's data loaders select by name and ignore
the new column.
