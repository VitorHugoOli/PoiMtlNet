# docs/infra/runpod/scripts.md — RunPod scripts index

All RunPod scripts live in `scripts/` (canonical location). They're indexed here for discoverability.

## Setup + data

| Script | Purpose |
|---|---|
| `scripts/runpod_setup.sh` | One-shot env install on a fresh RunPod pod. Creates `.venv`, installs CUDA-pinned torch, project deps. ~3 min on RTX 4090 / Ubuntu 22.04 base. |
| `scripts/runpod_fetch_data.sh <state>` | Fetch the per-state dataset bundle from Google Drive via gdown. ~30 s for AL/AZ; 2.3 GB for FL. |

## Training

| Script | Purpose |
|---|---|
| `scripts/runpod_train_fl_h3alt.sh` | Train the NORTH_STAR FL H3-alt config end-to-end (5f×50ep, ~19 min on RTX 4090). |
| `scripts/runpod_train_fl_h3alt_compile.sh` | H3-alt FL with `torch.compile` enabled — perf-comparison variant. |
| `scripts/runpod_train_fl_h3alt_tf32.sh` | H3-alt FL with TF32 on Ampere — perf-comparison variant. |
| `scripts/runpod_train_fl_h3alt_bs2048.sh` | H3-alt FL at batch-size 2048 — OOM-test variant (FL 4702 regions can saturate). |

## Perf comparison

| Script | Purpose |
|---|---|
| `scripts/runpod_perf_compare.sh` | Run the multi-config perf comparison (compile/tf32/bs/baseline) on the same pod. |
| `scripts/runpod_perf_compare_resume.sh` | Resume a perf-comparison run from where it stopped (e.g., after a crash or interruption). |
| `scripts/runpod_perf_summarise.py` | Aggregate the perf-comparison JSONs into a summary table. |

## Typical session

```bash
# 1. Connect to pod, clone, branch
cd /workspace
git clone <repo-url> PoiMtlNet
cd PoiMtlNet
git checkout main                              # or worktree-check2hgi-mtl pre-merge

# 2. Bring up env
bash scripts/runpod_setup.sh                   # ~3 min
source .venv/bin/activate

# 3. Fetch data for the state you'll train
bash scripts/runpod_fetch_data.sh florida      # ~30 s

# 4. Run training in tmux (so SSH disconnect doesn't kill it)
tmux new -s mtl 'bash scripts/runpod_train_fl_h3alt.sh'

# 5. Detach (Ctrl-b d), reconnect later, sync results back
```

For the canonical end-to-end recipe see [`README.md`](README.md).
