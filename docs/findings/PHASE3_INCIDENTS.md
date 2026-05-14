# Phase 3 — incident notes (2026-04-30)

Two operational issues encountered during the Phase 3 leak-free re-run on Lightning H100 80 GB. Captured here so future agents/reviewers don't re-derive them.

---

## Incident 1 — AL+AZ HGI `next.parquet` data-pipeline bug

**What happened.** The original upstream Drive folders for `output/hgi/{alabama,arizona}/input/` were missing `next_region.parquet`, so the user re-uploaded both `next.parquet` and `next_region.parquet` for those two states. Two upload-side defects in succession:

1. **First re-upload (file-identity bug).** The HGI versions of `next.parquet` and `next_region.parquet` for AL/AZ were **byte-identical** to the c2hgi versions (verified via md5sum). Both engines' MTL cells therefore loaded the same check-in-level c2hgi embeddings, producing **identical training trajectories to 17 decimal places** — exposed by an md5sum cross-check between `mtlnet_*/folds/fold1_info.json` for AL c2hgi vs AL hgi during the first MTL B9 pass. The user's pipeline that generates the per-state HGI windowed parquet appears to have re-exported the c2hgi build for AL/AZ instead of running the HGI-specific build.

2. **Second re-upload (dtype bug).** After the user re-generated and re-uploaded, the HGI files for AL+AZ were *content-correct* (md5 differs from c2hgi, embedding values look right) but their 576 embedding columns were stored as **string** (object dtype) instead of float32. The values printed as `'-0.33573684'`, `'0.5499820113'`, etc. — clearly a CSV→Parquet roundtrip in the upstream pipeline that didn't preserve numeric dtype. FL/CA/TX HGI files (which were *not* re-uploaded) had the correct float32 dtype, confirming the bug is local to the AL/AZ re-export path.

**Workaround.** We cast cols 0..575 to float32 in place via:

```python
import pandas as pd, numpy as np
for state in ['alabama', 'arizona']:
    p = f'output/hgi/{state}/input/next.parquet'
    df = pd.read_parquet(p)
    for c in [str(i) for i in range(576)]:
        df[c] = pd.to_numeric(df[c], errors='raise').astype(np.float32)
    df.to_parquet(p, index=False)
```

After casting, AL+AZ HGI cells produced expected non-identical training trajectories.

**Upstream follow-up.** The user's `pipelines/embedding/hgi.pipe.py` (or whatever generates per-state windowed `next.parquet`) needs a check that either (a) it doesn't run when the engine has already been built, or (b) writes float32 directly without round-tripping through a string-typed CSV. Pre-flight verification suggested:

```python
import pyarrow.parquet as pq
t = pq.read_table(f'output/hgi/<state>/input/next.parquet')
assert str(t.schema.field('0').type) == 'float', "embeddings must be float-typed"
```

This single-line check should be added to `setup_lightning_pod.sh`'s pre-flight verification.

---

## Incident 2 — TX MTL B9 cannot pair 2-way on H100 80 GB

**What happened.** The Phase 3 takeover dispatcher (`run_phase3_takeover_v2.sh`) packs same-state c2hgi+hgi MTL B9 cells two-at-a-time on a single GPU to halve wall-clock. AL/AZ/FL/CA all packed cleanly. **TX hgi OOM'd at fold 1 epoch 6** while sharing the GPU with TX c2hgi:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.90 GiB.
GPU 0 has a total capacity of 79.18 GiB of which 8.78 GiB is free.
Process 240769 has 40.20 GiB memory in use.
This process has 30.19 GiB memory in use.
```

**Root cause.** The OOM happens at `epoch_task_b_logits = torch.cat(all_task_b_logits)` in `src/training/runners/mtl_cv.py:513`. This aggregates per-batch validation logits across the whole val fold. For TX:

- 460,976 total rows, 5-fold StratifiedGroupKFold → ~92 K val rows per fold
- 4,702 regions
- float32

Per-fold validation logit tensor size: `92,000 × 4,702 × 4 B ≈ 1.7 GB`.

Across 50 epochs, each epoch reallocates the buffer; with PyTorch's caching allocator, fragmentation pushes the working set near 9 GB right at the `torch.cat` step. Combined with the active-fold model + activation memory (~30 GB for B9 with cross-attn), one TX cell alone uses ~40 GB. **Two TX cells co-tenant on 80 GB → ~70 GB used + ~10 GB free, which the cat allocation overruns.**

For the smaller states the validation logit tensor is much smaller:
- AL: 2,500 val × 1,109 regions × 4 B ≈ 11 MB
- FL: 32,000 val × 4,702 regions × 4 B ≈ 600 MB

so 2-way packing fits. TX is the only state that hits the ceiling.

**Recovery pattern.** `scripts/run_tx_hgi_recovery.sh` waits for the orphaned TX c2hgi python to finish (its file descriptors survive the parent bash dying because `kill -KILL` on the bash doesn't propagate by default), then launches TX hgi alone with full GPU. Solo-tenant TX hgi completes in ~44 min — same as TX c2hgi when shared, since shared-GPU contention adds ~10–15% overhead anyway.

**Operational rule for future grids on H100 80 GB.** Pair small + small (AL/AZ/FL/CA) for 2-way packing; **TX must run sequentially**. If we ever scale to multi-state larger than TX (e.g., NY = 60 K POIs), single-tenant becomes the rule, not the exception.

A cleaner long-term fix would be to stream the validation logits to disk per-batch instead of accumulating them in GPU memory, but that's a refactor of `mtl_cv.py` not in scope for Phase 3.

---

## Cross-references

- `PHASE3_TRACKER.md §9` — closure verdict + acceptance-bar table
- `research/SUBSTRATE_COMPARISON_FINDINGS.md` Phase 3 section — full cross-state tables
- `scripts/run_phase3_takeover_v2.sh` + `scripts/run_tx_hgi_recovery.sh` — orchestration
- Drive bundle: `mestrado_data/PoiMtlNet/phase3_archives/phase3_drive_bundle_2026-04-30.tar.gz`
