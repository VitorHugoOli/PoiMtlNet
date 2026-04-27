# Colab Run Guide — for agents and humans

This guide captures the operational knowledge for running training on Google
Colab (T4 GPU). It exists because the path from "open notebook" to "valid
paper-quality result" has several non-obvious traps; the patches that landed
on the `feat/colab-gpu-perf` branch (commits `91d78f8`, `0f9bcdf`) fixed all
of them, and this doc records *why* each fix was needed so future agents
don't re-derive the diagnosis from scratch.

> **Template notebook:** [`notebooks/colab_check2hgi_mtl.ipynb`](../notebooks/colab_check2hgi_mtl.ipynb).
> Copy and edit per run. Self-contained — runs end-to-end from a fresh kernel.

---

## 1 · When to use Colab

Colab T4 is good for:

- **Long training runs that would saturate the M4 Pro for hours**, especially
  on FL (4702 regions, 159K samples). FL 5f×50ep takes ~50 min on T4 vs ~5 h
  on MPS in our setup.
- **Validation reruns** where you need a clean GPU but don't want to occupy
  the local machine.
- **Embedding generation pipelines** that benefit from fp32 throughput.

Stick to MPS (M4 Pro) when:

- The run uses operators with poor MPS coverage (rare — `PYTORCH_ENABLE_MPS_FALLBACK=1`
  generally handles it).
- You want strict run-to-run reproducibility (Colab T4 sessions are ephemeral
  — same run on the same machine is more deterministic).
- The run is small enough (AL/AZ scale): MPS at ~5 batch/s on AZ matches T4's
  startup overhead, so the wall-clock difference is small.

---

## 2 · Drive layout

Every Colab run expects this structure on Google Drive:

```
MyDrive/mestrado/PoiMtlNet/
├── data/
│   ├── checkins/                 # Alabama.parquet, Florida.parquet, …
│   └── miscellaneous/            # tl_2022_*_tract_*/  (TIGER shapefiles)
├── output/                       # Generated embeddings + model inputs
│   └── check2hgi/
│       └── florida/
│           ├── embeddings.parquet
│           ├── input/next.parquet
│           └── region_transition_log.pt    # required for next_getnext_hard
└── results/                      # Training results land here
```

`region_transition_log.pt` is **required** for the north-star B3 config because
the `next_getnext_hard` reg-head reads it during initialisation. Generate it
locally (`scripts/compute_region_transition.py --state <s>`) and rsync the
output dir to Drive before the Colab run.

---

## 3 · Branch hygiene

The check2HGI study lives on **`worktree-check2hgi-mtl`** (the long-lived
feature branch). The Colab perf optimisations live on
**`feat/colab-gpu-perf`** (branched off `worktree-check2hgi-mtl` at `c5cd8ae`).
Always `git checkout feat/colab-gpu-perf` on Colab — `main` does *not*
contain the check2HGI CLI flags (`--task-set`, `--task-a-input-type`,
`--reg-head`, `--cat-head`).

**To verify the right branch is checked out:**

```python
!grep -c task-a-input-type scripts/train.py   # must be ≥ 1
```

---

## 4 · The north-star B3 training command

This is the single canonical config from `docs/studies/check2hgi/NORTH_STAR.md`
(post-F27, committed 2026-04-24). All FL/AL/AZ comparison runs use it.

```bash
python -u scripts/train.py \
  --state <state> \
  --task mtl \
  --task-set check2hgi_next_region \
  --engine check2hgi \
  --folds 5 \
  --epochs 50 \
  --seed 42 \
  --task-a-input-type checkin \
  --task-b-input-type region \
  --model mtlnet_crossattn \
  --mtl-loss static_weight \
  --category-weight 0.75 \
  --cat-head next_gru \
  --reg-head next_getnext_hard \
  --reg-head-param d_model=256 \
  --reg-head-param num_heads=8 \
  --reg-head-param transition_path=$OUTPUT_DIR/check2hgi/<state>/region_transition_log.pt \
  --max-lr 0.003 \
  --gradient-accumulation-steps 1 \
  --no-checkpoints
```

Wall-clock on T4 (post-perf-pass): **AL ~2 min, AZ ~10 min, FL ~50 min** for 5f×50ep.

---

## 5 · The detached-subprocess pattern (DO use this)

**TL;DR:** never use `!{cmd}` for runs longer than ~5 minutes. Use
`subprocess.Popen` with `nohup` + `start_new_session=True` and `tee` to a
Drive log file.

```python
import subprocess, shlex
from pathlib import Path

LOG_FILE = Path('/content/drive/MyDrive/mestrado/PoiMtlNet/results/<run>.log')
with LOG_FILE.open('w') as lf:
    proc = subprocess.Popen(
        ['nohup'] + shlex.split(_cmd),
        stdout=lf, stderr=subprocess.STDOUT,
        cwd='/content/PoiMtlNet',
        start_new_session=True,   # setsid — detach from parent process group
    )
print(f"PID {proc.pid} (detached)")
```

This survives:
- MCP `run_code_cell` cell-timeout (the foreground `!{cmd}` path receives SIGINT
  after ~5 min on long-running cells, killing your job)
- Notebook tab close / browser refresh
- Kernel disconnect (the run keeps going; the log on Drive still receives output)

A separate "monitor" cell tails the log to inspect progress without blocking.

**Trade-off:** progress isn't streamed to the cell. You re-run a tiny monitor
cell to see the latest log lines and ETA. That's a small price for not losing
hours of GPU time to a stray cell timeout.

---

## 6 · Performance optimisations (already on `feat/colab-gpu-perf`)

These five quality-neutral optimisations were validated on AZ 5f×50ep and FL
5f×50ep (metric values byte-identical to baseline within natural CUDA
non-determinism). MPS path is preserved.

| Optimisation | Where | Why |
|---|---|---|
| `_get_num_workers() == 0` everywhere | `src/data/folds.py` | DataLoader workers fork the full in-memory tensor dataset. On FL each of 8 default workers held a 9.8 GB copy → cgroup OOM-killed before fold 1 finished. |
| Chunked `_rank_of_target` | `src/tracking/metrics.py` | Naive `(logits > target_scores)` materialises an `[N × C]` boolean tensor. On FL (127K × 4702) that's 4.46 GB peak, OOMs T4. Chunked path bounds peak at chunk × C ≈ 80 MB. |
| `cudnn.benchmark = True` (CUDA only) | `src/configs/globals.py` | Autotunes kernels for our stable shapes (bs=2048, seq=9). Small but free. |
| Vectorised OOD mask (`torch.isin`) | `src/training/runners/mtl_eval.py` | `[int(t.item()) in S for t in targets]` triggers ~31K host↔device syncs on FL val per epoch. `torch.isin` does it in one kernel. |
| Cached rank-of-target | `src/tracking/metrics.py` | MRR + NDCG@3 + NDCG@5 each rebuilt the chunked rank tensor; 3× the work. Now computed once and reused. |
| `clear_mps_cache` extended for CUDA | `src/utils/mps.py` | The function name is historical; it now also calls `torch.cuda.empty_cache()` between folds, which prevents fragmentation that pushed the 5th fold over the 15 GB ceiling. |

---

## 7 · Memory pitfalls (with diagnosis recipes)

### Pitfall A — DataLoader workers eat RAM

Symptom: `dmesg | grep oom` shows `pt_data_worker invoked oom-killer`,
`anon-rss: 9802956kB` per worker, cgroup limit hit.

Diagnosis: `_get_num_workers()` returns `min(8, cpu_count)` on CUDA (default
PyTorch behaviour). For in-memory tensor datasets there is *no I/O to overlap*,
so workers are pure overhead.

Fix: already applied on this branch. If you see it on another branch, set
`num_workers=0` explicitly.

### Pitfall B — CUDA OOM during MRR/NDCG computation

Symptom: `torch.OutOfMemoryError: ... Tried to allocate 4.46 GiB` mid-validation,
typically at `_rank_of_target` or `_mean_reciprocal_rank`.

Diagnosis: the high-cardinality reg head (FL 4702 classes) makes the `[N × C]`
pairwise comparison too big to fit on T4. The training forward pass already
consumes ~9 GB; the metric tensor is the straw.

Fix: chunked computation (already on this branch). If you ever extend the
metric set, use `_mrr_from_rank` / `_ndcg_from_rank` and pass a precomputed
rank — don't call `_rank_of_target` repeatedly.

### Pitfall C — between-fold leak on CUDA

Symptom: fold 1 trains fine, fold 2 OOMs at model construction. Logs show
*reserved memory* keeps growing.

Diagnosis: `clear_mps_cache` was MPS-only. On CUDA, the previous fold's
optimizer state stayed allocated until Python GC eventually collected it,
and `optimizer.step` on fold 2 raced against that.

Fix: `clear_mps_cache` is now device-agnostic. If your script doesn't call it,
add a `torch.cuda.empty_cache()` + `gc.collect()` between folds.

---

## 8 · Verifying the running process has the new code

Subprocess-launched runs do **not** need a kernel restart after `git pull` —
the subprocess starts a fresh Python interpreter that reads source from disk
at import time. But if you suspect the run is on stale code, this recipe
proves it definitively:

```python
import subprocess
from pathlib import Path
from datetime import datetime

PID = <your_pid>

# 1. When did the subprocess start?
r = subprocess.run(['ps', '-p', str(PID), '-o', 'lstart='],
                   capture_output=True, text=True)
print('Process started:', r.stdout.strip())

# 2. When was the source mtime?
for path in ['src/tracking/metrics.py', 'src/training/runners/mtl_eval.py']:
    p = Path(f'/content/PoiMtlNet/{path}')
    print(f'  {path} mtime: {datetime.fromtimestamp(p.stat().st_mtime)}')

# 3. Decisive proof: __pycache__ mtime is when the subprocess compiled the source.
for pyc_dir in ['src/tracking/__pycache__', 'src/training/runners/__pycache__']:
    base = Path(f'/content/PoiMtlNet/{pyc_dir}')
    for pyc in base.glob('*.cpython-*.pyc'):
        mt = datetime.fromtimestamp(pyc.stat().st_mtime)
        print(f'  {pyc.relative_to("/content/PoiMtlNet")} compiled: {mt}')
```

The `.pyc` mtime should be **after** the source mtime AND after the subprocess
start. If `.pyc` is older than the source, the subprocess is running stale
code (kill and relaunch).

---

## 9 · Inspecting results

The full per-fold + aggregate metrics live at:

```
results/check2hgi/<state>/mtlnet_lr1.0e-04_bs2048_ep50_<timestamp>/
├── summary/full_summary.json     # ← aggregated mean ± std for every metric
├── folds/foldN_info.json         # per-fold best epochs + diagnostics
├── folds/foldN_*_report.json     # sklearn-style per-class report
└── metrics/foldN_*_train.csv     # per-epoch train metric trajectory
```

Headline metrics for north-star comparison:

```python
import json
data = json.load(open('summary/full_summary.json'))
for task in ('next_category', 'next_region', 'model'):
    for metric in ('f1', 'accuracy', 'top10_acc_indist', 'mrr_indist', 'joint_score'):
        if metric in data.get(task, {}):
            stats = data[task][metric]
            print(f'{task}.{metric}: {stats["mean"]:.4f} ± {stats["std"]:.4f}')
```

The **`_indist` suffix** matters for the high-cardinality reg head: those metrics
are restricted to val samples whose target appears in the training fold (CH06
in the claim catalog). Plain `top10_acc` includes OOD samples (where the target
region is unseen at training time) and is artificially depressed.

---

## 10 · Reference benchmarks

After the perf pass, expected wall-clock on T4 for the north-star B3 config:

| State | Train samples | Region classes | 5f×50ep wall-clock |
|---|---:|---:|---:|
| AL | ~12 K | 1109 | ~10 min |
| AZ | ~28 K | 1547 | ~10 min |
| FL | ~159 K | 4702 | ~50 min |

Reference quality (from NORTH_STAR.md cross-checks):

- AL Cat F1 ≈ 0.43, Reg Acc@10_indist ≈ 0.60
- AZ Cat F1 ≈ 0.46, Reg Acc@10_indist ≈ 0.54
- FL Cat F1 ≈ 0.67, Reg Acc@10_indist ≈ 0.58 (5f mean — n=1 was 0.65)

If your run is **outside ±2 σ of these on the same seed**, something has changed
semantically — investigate before reporting.

---

## 11 · Quickstart checklist

```
[ ] Open a fresh Colab session, set Runtime → T4 GPU
[ ] Open notebooks/colab_check2hgi_mtl.ipynb
[ ] Run cell ① (Drive mount) — confirm DATA_ROOT/OUTPUT_DIR/RESULTS_ROOT exist
[ ] Run cell ② (clone + checkout feat/colab-gpu-perf)
[ ] Run cell ③ (deps + verify Device: cuda)
[ ] Run cell ④ (verify inputs for your state — embeddings, next.parquet,
                 region_transition_log.pt all present)
[ ] Edit cell ⑤ to set your STATE (florida/alabama/arizona)
[ ] Run cell ⑥ to launch detached subprocess, note the PID + log path
[ ] Run cell ⑦ periodically to monitor (re-run cell to refresh)
[ ] When done, run cell ⑧ to load full_summary.json + per-fold deltas
```

---

## 12 · Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Device: cpu` | Runtime not set to T4 | Runtime → Change runtime type → T4 GPU → restart, re-run setup |
| `mount failed` on Drive | Auth dialog wasn't accepted | `drive.mount('/content/drive', force_remount=True)` and click through |
| `No such file: region_transition_log.pt` | Drive wasn't synced from local | Generate locally, rsync to Drive |
| Job killed silently mid-run | Cell timeout on `!{cmd}` | Switch to detached subprocess (§5) |
| FLAGS NOT FOUND on `--task-set` | Wrong branch | `git checkout feat/colab-gpu-perf` |
| Long pip install on every session | Colab images don't persist | Acceptable; ~2 min one-time per session |
| `torch.isin` errors on MPS | PyTorch < 2.0 | Bump PyTorch in `requirements_colab.txt` |

---

## 13 · See also

- [`notebooks/colab_check2hgi_mtl.ipynb`](../notebooks/colab_check2hgi_mtl.ipynb) — the template this guide describes
- [`docs/studies/check2hgi/NORTH_STAR.md`](studies/check2hgi/NORTH_STAR.md) — the canonical B3 config + reference numbers
- [`scripts/study/colab_runner.py`](../scripts/study/colab_runner.py) — alternative entry point for the study-driven workflow (consumes `state.json`-enrolled tests)
- [`notebooks/colab_study_runner.ipynb`](../notebooks/colab_study_runner.ipynb) — companion notebook for the study runner
