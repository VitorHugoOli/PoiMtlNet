# Known infrastructure issues (2026-04-18/19)

## I1 — `--no-checkpoints` flag doesn't suppress ModelCheckpoint on the check2hgi task path

**Symptom:** Nash-MTL + cross-attn AZ run (PID 45782, 2026-04-19 ~19:00) crashed at epoch end with:
```
File "src/training/callbacks.py", line 207
  torch.save(self._model.state_dict(), path)
RuntimeError: [enforce fail at inline_container.cc:672] . unexpected pos 19166976 vs 19166864
```

**Root cause:** `scripts/train.py` has `_NO_CHECKPOINTS` global set from `--no-checkpoints`. The legacy `_run_mtl` path (line 142–146) reads the flag and skips callback construction. The `_run_mtl_check2hgi` path (line 200–215) for non-legacy task_sets ignores it and always attaches a ModelCheckpoint callback via `_default_checkpoint_callbacks`.

Combined with an unstable SSD (seen in prior runs where Mac went idle), the torch.save wrote a corrupt file and raised.

**Workaround in use:** launch with `OUTPUT_DIR=/tmp/check2hgi_data` (boot volume APFS, not external SSD). All long-running experiments now use this pattern.

**Proper fix (TODO):** mirror the `_NO_CHECKPOINTS` check in `_run_mtl_check2hgi`:
```python
# In scripts/train.py around line 202:
if _NO_CHECKPOINTS:
    cbs = []
else:
    run_dir = _make_run_dir(results_path, task=f"mtl__{task_set.name}", config=config)
    cbs = _default_checkpoint_callbacks(run_dir, monitor="val_joint_acc1")
```

Non-blocking for ablation work since /tmp workaround holds.

## I2 — SSD reliability

Thunderbolt SSD has flaked on two separate multi-hour runs (2026-04-16 SIGBUS cluster, 2026-04-19 torch.save corruption). Physical reseating fixed I/O but periodic issues persist.

**Workaround:** always redirect `OUTPUT_DIR=/tmp/check2hgi_data` for long runs. Data required is ~1–2 GB per state; boot volume has 18+ GiB free.

**Risk:** any run that exhausts /tmp or that needs to write large intermediate files elsewhere still hits the SSD.

## I3 — `max_lr=0.01` run corruption on 2026-04-18

Step 7 config 2 (`max_lr=0.01`) took 635 min wall time (vs ~20 expected) and produced absurd metrics (reg A@10 = 12.33 ± 11.79). User confirmed the Mac discharged to idle mid-run. Data marked QUARANTINED in `docs/studies/check2hgi/results/P2/ablation_07_maxlr_0.01_QUARANTINED.md`.
