# Handoff — state as of 2026-04-14

Snapshot written at session close so the next session (using `/coordinator` + `/worker`) starts with full context. This is a **transient** file — update or delete it once P1 is underway and state.json is authoritative.

---

## Study status at a glance

- **Current phase:** P0 (not yet exited — see checklist below)
- **Claims catalog:** 28 claims in `CLAIMS_AND_HYPOTHESES.md` (C01–C21 original + C22–C28 added for P6)
- **Phases registered:** P0–P6 (P6 runs parallel to P5 after P4)
- **Test suite:** 651 passed, 1 skipped (sklearn 1.8.0, torch 2.11.0)
- **Git:** `main` is clean and pushed; latest commit `63af57e`

---

## Data availability snapshot (run `ls output/*/*/input/category.parquet` to refresh)

| State   | dgi | hgi | fusion | sphere2vec | time2vec (next only) | poi2hgi | check2hgi |
|---------|:---:|:---:|:------:|:----------:|:--------------------:|:-------:|:---------:|
| alabama |  ✗  |  ✓  |   ✓    |     ✓      |          ✓           |    ✗    |     ✗     |
| arizona |  ✗  |  ✗  |   ✗    |     ✗      |          ✗           |    ✗    |     ✗     |
| florida |  ✗  |  ✓  |   ✓    |     ✓      |          ✓           |    ✗    |     ✗     |

**Implications:**

- **Alabama DGI is missing** — blocks P0.4 CBIC sanity replication (expected cat F1 = 46–48%, next F1 = 26–28%). The CBIC baseline used DGI; regenerate DGI for Alabama to close P0.
- **All Arizona embeddings missing** — blocks the AZ replication leg of every phase. Minimum needed for P1 start: `{hgi, fusion}` × arizona. `sphere2vec`/`time2vec` needed before regenerating arizona/fusion inputs.
- Florida has hgi/fusion/sphere2vec/time2vec — good for P3 heavy validation once P1/P2 settle.

## Frozen folds snapshot

Rollup at `docs/studies/results/P0/folds/frozen.json` — 2 entries:

- `alabama/fusion/mtl` — frozen (65.9 MB, sklearn 1.8.0)
- `alabama/hgi/mtl`   — frozen (33.6 MB, sklearn 1.8.0)

`fold_indices_mtl.pt` is a single cache per (state, engine) that holds both category and next splits from the MTL user-isolation protocol. Paired tests (C28 etc.) need single-task and MTL runs to share the same file — don't freeze single-task caches separately unless the design intent changes.

---

## P0 exit-criteria checklist

| Step | Done | Notes |
|---|:---:|---|
| P0.1 Embeddings regenerated for AL + AZ | ⏳ | AL missing DGI; AZ entirely missing |
| P0.2 `validate_inputs` tool built + exercised | ✓ | Ran cleanly on alabama/fusion |
| P0.3 state.json initialized (P0–P6) | ✓ | Current phase P0 |
| P0.4 CBIC sanity run on AL + DGI | ⏳ | Blocked on (P0.1) |
| P0.5 `launch / archive / analyze / validate` scripts | ✓ | 21/21 smoke |
| P0.6 `/study` skill | ✓ | `.claude/commands/study.md` |
| P0.7 `/worker` + `/coordinator` skills | ✓ | `.claude/commands/{worker,coordinator}.md` |
| P0.8 Fold-freezing tooling | ✓ | `scripts/study/freeze_folds.py`; auto-load plumbed into `scripts/train.py` |
| P0.8 Frozen: AL + AZ × {dgi, hgi, fusion} × mtl | 2/6 | alabama/{hgi,fusion} done; rest blocked on embeddings |
| Hardware decision documented | ✓ | MASTER_PLAN §Hardware: M4 Pro 24GB preferred |

---

## Exact runbook when embeddings arrive

Run on the main box (`.venv/bin/python`), in this order:

```bash
# 1. Verify inputs present per engine
.venv/bin/python scripts/study/validate_inputs.py --state arizona --engine fusion
.venv/bin/python scripts/study/validate_inputs.py --state arizona --engine hgi
.venv/bin/python scripts/study/validate_inputs.py --state alabama --engine dgi   # for CBIC sanity
# Reports land under docs/studies/results/P0/integrity/<state>_<engine>.json

# 2. Freeze the P0.8 default set (skips the two already cached)
.venv/bin/python scripts/study/freeze_folds.py --default-set

# 3. CBIC sanity replication — expect cat F1 = 46-48%, next F1 = 26-28%
.venv/bin/python scripts/train.py --task mtl --state alabama --engine dgi \
    --epochs 50 --folds 5 --seed 42 --mtl-loss nash_mtl
# Deviation > 5 pp from CBIC → stop, investigate before P1.

# 4. Commit the P0 outputs and mark P0 done
.venv/bin/python scripts/study/study.py advance      # will refuse unless tests done
# Hand-edit state.json: P0.status = completed, current_phase = P1.

# 5. Launch P1 enrollment via /coordinator P1.
```

---

## Key environmental facts to carry forward

- **`requirements.txt` pins `scikit-learn==1.8.0`.** The 1.8 release fixed a `StratifiedGroupKFold(shuffle=True)` bug that silently broke stratification — do **not** downgrade. `requirements_colab.txt` pins the same version.
- **Torch 2.11.0** — regression test `test_mtl_f1_within_tolerance` was re-calibrated from floor 0.92 → 0.88 on this torch version (drifted from torch 2.9.1's 0.9286). See comment in `tests/test_regression/test_regression.py:254-262`.
- **MPS runs:** before long training, set
  ```bash
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
- **Colab parallelism:** use `scripts/study/colab_runner.py` via `notebooks/colab_study_runner.ipynb`. Colab reads committed state.json, packages results to Drive, Mac imports via `/study import`. Colab never writes state.json (avoids split-brain).

---

## How to invoke next session

```
/coordinator P0        # will see P0 not done → tell you what to run
/worker P1             # enroll + run P1 tests after P0 closes
```

The coordinator reads this HANDOFF + state.json + CLAIMS_AND_HYPOTHESES.md. If this file is stale vs `state.json`, **trust state.json** — it's updated transactionally by every `/study` action.
