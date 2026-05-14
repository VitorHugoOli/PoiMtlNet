# docs/infra/lightning/ — Lightning.ai pods (A100 / H100)

Lightning Studio pods with A100 (40 GB) or H100 (80 GB) for heavy parallel work — multi-state Wilcoxon at scale, Phase-3 leak-free reruns, multi-fold MTL on FL/CA/TX.

## Quick start

```bash
# 1. Spin up a Lightning Studio with ≥1× A100 (40 GB) — see https://lightning.ai
# 2. Open a terminal in the studio
# 3. Bootstrap:
git clone <repo-url> PoiMtlNet
cd PoiMtlNet
git checkout main                              # or worktree branch pre-merge
bash scripts/setup_lightning_pod.sh            # creates .venv, installs CUDA-pinned torch + deps
source .venv/bin/activate

# 4. Fetch data (Lightning has no Drive mount; use gdown — see ../data/drive_download.md)
python scripts/phase3_download_drive.py --state florida

# 5. Launch training (use the canonical CLI; for multi-state Phase-3 use the launcher scripts)
python scripts/train.py --task mtl --state florida --engine check2hgi --folds 5 --epochs 50
# or:
bash scripts/run_phase3_takeover_v2.sh         # multi-state Phase-3 launcher
```

## Pod requirements

- ≥1× A100 (40 GB). 4× A100 gives ~3× wall-clock speedup at near-flat total cost (the tensor parallelism is fold-level, not within-batch).
- Disk: ≥40 GB free (~25 GB upstream parquets across 5 states + ~15 GB run dirs).
- Linux + Python 3.12 + CUDA 12.x. PyTorch 2.8+cu128 confirmed working.
- T4 (15 GB) is **not enough** for FL multi-fold work — see archived PHASE3_TRACKER for the OOM note.

## Wall-clock reference

The Phase-3 reg STL + MTL re-run on 5 states (10 cells each) using leak-free per-fold transition matrices:

| Step | Cells | 4× A100 (40 GB) | 1× A100 (40 GB) |
|---|---|---|---|
| Per-fold transition build (CPU) | 25 matrices (5 × 5) | ~25 min total | ~25 min |
| Reg STL re-run (`_pf` suffix) | 10 cells | ~50 min | ~225 min |
| MTL B3 re-run (`_pf` suffix) | 10 cells | ~110 min | ~350 min |
| Finalize (extract + paired tests) | CPU | ~10 min | ~10 min |
| **Total** | | **~2.7 h, ~$13** | **~10 h, ~$25** |

## Scripts

Lightning-relevant scripts live in `scripts/`:

| Script | Purpose |
|---|---|
| `scripts/setup_lightning_pod.sh` | One-shot pod bootstrap: `.venv`, CUDA-pinned torch, deps. |
| `scripts/run_phase2_mtl_lightning.sh` | Phase-2 MTL launcher, tuned for Lightning multi-A100. |
| `scripts/run_phase2_tx_lightning.sh` | Phase-2 TX-specific launcher. |
| `scripts/run_phase3_takeover_v2.sh` | Phase-3 takeover launcher (per-fold leak-free reruns). |

## Detached launches

Use `tmux` or `nohup` for any run >5 min — Lightning Studio sessions can disconnect. Pattern:

```bash
tmux new -s mtl 'bash scripts/run_phase3_takeover_v2.sh 2>&1 | tee logs/phase3_$(date +%s).log'
# Detach: Ctrl-b d. Reconnect: tmux attach -t mtl
```

## Historical context

The Phase-3 leak-free reruns documented in `docs/archive/check2hgi-post-paper-closure-2026-05-01/PHASE3_LIGHTNING_HANDOFF.md` are the original source for the wall-clock numbers above. Read that file for the full Phase-3 narrative if needed.
