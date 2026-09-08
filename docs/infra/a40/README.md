# docs/infra/a40/ — `nespedgpu` SSH bare-metal

> ⚠ **The card is NOT an A40.** Measured 2026-09-02 over SSH: `nvidia-smi` reports an **NVIDIA
> RTX 6000 Ada Generation**. Same 46 068 MiB of VRAM, so every batch-size figure below still holds;
> the model name and the throughput table do not. The folder keeps its name because paths across the
> repo point at it.

Dedicated box (46 GB VRAM, SSH access, hostname `nespedgpu`). Used for sustained training runs where the A40's large VRAM and high bandwidth outperform Colab T4 / RunPod 4090.

> **Scripts moved 2026-05-16**: the parallel sweep runner, regen helpers, and audit utilities used to live in this folder. They now live under [`scripts/canonical_improvement/`](../../../scripts/canonical_improvement/) (and the unit tests under [`tests/canonical_improvement/`](../../../tests/canonical_improvement/)). This README remains for A40-specific hardware notes and conventions.

## Hardware specs

| Property | Value |
|---|---|
| GPU | **NVIDIA RTX 6000 Ada Generation** (the folder name says A40; the card is not) |
| VRAM | 46 GB (46 068 MiB) |
| Driver | 580.173.02 (measured live 2026-09-06 and again 2026-09-08; the machine updated its driver after this doc was first written) |
| CUDA (driver max) | 13.0 |
| PyTorch CUDA | 12.8 (cu128) |
| RAM | 125 GB |
| Home disk | 393 GB total, **~36 GB free (measured 2026-09-02, 91 % used)** — was ~78 GB on 2026-05-14 |
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
# ⚠ tmux lives at ~/.local/bin/tmux and is NOT on the PATH of a non-interactive `ssh host cmd`.
#   Use the full path from a script, or a login shell (`ssh host 'bash -lc "..."'`).
~/.local/bin/tmux new -s mtl

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
- **Disk quota on /home** — **~36 GB free as of 2026-09-02, 91 % used**; `output/check2hgi` alone is 21 GB. Prune states you are not actively using before starting anything that writes a substrate.
- **`ModuleNotFoundError: No module named 'configs'`** — the five `scripts/canonical_improvement/regen_emb_*.py` helpers computed their repo root with four `.parent` hops, correct while they lived in this folder and one level too high since the 2026-05-16 move. Fixed 2026-09-02. If you meet it in another script moved out of here, that is the cause.

## ⚠ Before you regenerate any substrate, read this

`scripts/canonical_improvement/regen_emb_alpha.py` writes to **`output/check2hgi/<state>/`, overwriting it**, and that directory is the frozen v11 substrate. **The dissertation does not deliver from it.** The delivered generation is **`output/check2hgi_v18/`**, built by `scripts/integrity_v2/build_study_repr.py`, whose recipe is pinned in a `V14` dict at line 77 *precisely so an upstream default cannot drift the study* — the α weights are not exposed on its CLI.

So an ablation run through `regen_emb_alpha.py` measures the **wrong generation** and destroys a frozen artifact on the way. Measured 2026-09-02, when exactly that was about to happen: the run was killed before it wrote, and the substrate was verified intact against a copy (35 files, 340 041 486 bytes, identical).

If you need to vary those weights on the delivered generation, the change belongs in `build_study_repr.py` as an explicit, declared override — not in the older helper.
