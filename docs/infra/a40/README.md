# docs/infra/a40/ — NVIDIA A40 SSH bare-metal

Dedicated A40 box (46 GB VRAM, SSH access, hostname `nespedgpu`). Used for sustained training runs where the A40's large VRAM and high bandwidth outperform Colab T4 / RunPod 4090.

> **Scripts moved 2026-05-16**: the parallel sweep runner, regen helpers, and audit utilities used to live in this folder. They now live under [`scripts/canonical_improvement/`](../../../scripts/canonical_improvement/) (and the unit tests under [`tests/canonical_improvement/`](../../../tests/canonical_improvement/)). This README remains for A40-specific hardware notes and conventions.

## Hardware specs

| Property | Value |
|---|---|
| GPU | NVIDIA A40 |
| VRAM | 46 GB (46 068 MiB) |
| Driver | 580.126.09 |
| CUDA (driver max) | 13.0 |
| PyTorch CUDA | 12.8 (cu128) |
| RAM | 125 GB |
| Home disk | 393 GB total, ~78 GB free (as of 2026-05-14) |
| OS | Ubuntu, kernel 6.8.0-111-generic |
| PyTorch | 2.11.0+cu128 |

## When to use A40 over other machines

- **Use A40** when: you have SSH access to `nespedgpu`, need > 24 GB VRAM (the RTX 4090 cap), want a long-lived persistent environment without per-hour billing, or want to run large batch sizes (b=4096+) without OOM.
- **Use RunPod** when: this box is occupied or unavailable and you need CUDA with SSH workflow.
- **Use Lightning** when: you want multi-GPU parallelism or ad-hoc billing without a dedicated machine.
- **Use H100** when: you need > 46 GB VRAM or maximum single-GPU throughput for paper-closure final runs.

## Quick start

```bash
# 1. Connect
ssh vitor.oliveira@nespedgpu

# 2. Navigate to repo (already cloned at ~/PoiMtlNet)
cd ~/PoiMtlNet
git pull

# 3. Activate venv
source .venv/bin/activate

# 4. Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 5. Launch in tmux (mandatory — SSH disconnects kill foreground processes)
tmux new -s mtl

# ⚠ The bare `--task mtl --engine check2hgi` invocation will SMOKE-train but will
#    NOT match paper numbers — three defaults are silently wrong. Use the full
#    canonical invocation from `docs/NORTH_STAR.md §Champion`:
python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --state florida --engine check2hgi --seed 42 \
    --epochs 50 --folds 5 --batch-size 2048 \
    --model mtlnet_crossattn \
    --mtl-loss static_weight --category-weight 0.75 \
    --scheduler cosine --max-lr 3e-3 \
    --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --alternating-optimizer-step --alpha-no-weight-decay --min-best-epoch 5 \
    --cat-head next_gru --reg-head next_getnext_hard \
    --task-a-input-type checkin --task-b-input-type region \
    --per-fold-transition-dir output/check2hgi/florida \
    2>&1 | tee logs/a40_$(date +%Y%m%d_%H%M%S).log
# Detach: Ctrl-b d    Re-attach: tmux attach -t mtl
```

> ⚠ **Verify input modality from the log** — the first line of fold 1 should be:
>   `MTL_CHECK2HGI input modality: task_a=checkin (...), task_b=region (...)`
> If task_b shows `checkin`, the run will produce a paper-grade-looking cat number
> but a reg number ~22 pp below the canonical target. Kill and relaunch.

## Conventions

- Always run inside `tmux`. SSH disconnects kill foreground processes.
- Pipe outputs to `logs/a40_<timestamp>.log` so a reconnect can `tail -f` to check progress.
- `logs/` is gitignored — safe to accumulate there.
- Data lives at `~/PoiMtlNet/data/` and `~/PoiMtlNet/output/`; results at `~/PoiMtlNet/results/`.
- No per-hour cost — but be considerate of long idle GPU holds; run `nvidia-smi` before launching to confirm no other processes are using the GPU.

## Wall-clock reference

Measured on 2026-05-14, FL check2hgi MTL (NORTH_STAR H3-alt config), b=2048, seed 42:

| Run | Measured time | Notes |
|---|---|---|
| 1 fold × 1 epoch | **6.18s** | incl. fold setup + final validation pass |
| ~17 batch/s throughput | — | steady-state training speed |
| 1 fold × 50 epochs (projected) | **~5 min** | extrapolated from 1-epoch timing |
| 5 folds × 50 epochs (projected) | **~25 min** | ~1.3× faster than RunPod RTX 4090 (19 min for 5f×50ep) |
| 5 folds × 50 epochs × 20 seeds (projected) | **~8.3 h** | for multi-seed paper runs |

Compare: RTX 4090 (RunPod) does 5f×50ep in **19 min** at b=2048.

## Batch size guidance

The A40 has 46 GB VRAM — significantly more than the RTX 4090 (24 GB). You can safely increase batch sizes:

| State | Recommended batch | Notes |
|---|---|---|
| Florida (4702 regions) | 4096 | Peak ~25–30 GB at b=4096 |
| Alabama / Arizona | 4096 | Small states — no OOM risk |
| California / Texas | 2048–4096 | Test at 4096 first |

Use `--batch-size 4096` in your `scripts/train.py` call. Fall back to 2048 if OOM.

## Troubleshooting

- **`torch.cuda.is_available() == False`** — verify driver with `nvidia-smi`. If CUDA version mismatch, reinstall the matching torch wheel.
- **OOM** — drop batch size: `--batch-size 2048`. The A40 handles FL at b=2048 well under 24 GB.
- **Another process holds the GPU** — check with `nvidia-smi`; coordinate with other users on `nespedgpu`.
- **Disk quota on /home** — home has ~78 GB free (2026-05-14). Large `output/` dirs can fill it; prune states you're not actively using.
