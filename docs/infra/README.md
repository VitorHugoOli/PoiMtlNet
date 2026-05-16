# docs/infra/ — Operational documentation

This is the home for everything you need to run MTLnet on a specific environment. The canonical training entrypoint (`scripts/train.py --task mtl --state X --engine Y`) is the same everywhere; what differs is how you bring up the environment, fetch data, and run long jobs without losing them to session timeouts.

## "I'm on machine X — where do I look?"

| Environment | Read | When to use |
|---|---|---|
| **Local M4 Pro / Apple Silicon (MPS)** | [`local/README.md`](local/README.md) | Iteration, small states (AL/AZ), quick smoke tests, MPS-only debugging |
| **Google Colab T4** | [`colab/README.md`](colab/README.md) | Long runs that would saturate the M4 (FL 5f×50ep), validation reruns, embedding generation |
| **RunPod (CUDA, RTX 4090)** | [`runpod/README.md`](runpod/README.md) | SSH + pip workflow on a clean Linux pod; faster than Colab T4 for sustained workloads |
| **Lightning.ai pods (A100 / H100)** | [`lightning/README.md`](lightning/README.md) | Multi-GPU heavy work (Phase-3 leak-free reruns, multi-state Wilcoxon at scale) |
| **H100 SSH bare-metal** | [`h100/README.md`](h100/README.md) | When someone hands you a dedicated H100 box (e.g., paper-closure final runs) |
| **A40 SSH bare-metal (`nespedgpu`)** | [`a40/README.md`](a40/README.md) | 46 GB VRAM persistent machine; sustained runs without per-hour billing |
| **Anywhere — fetching data from Drive** | [`data/drive_download.md`](data/drive_download.md) | Bootstrap data on a fresh machine via gdown |

## What lives where

| Type | Location | Indexed in `docs/infra/`? |
|---|---|---|
| Canonical ops guides (env setup, launch patterns) | `docs/infra/<machine>/README.md` | yes (this folder) |
| Ops scripts (`scripts/runpod_*.sh`, `scripts/setup_lightning_pod.sh`, `scripts/run_h100_*.sh`, `scripts/phase3_download_drive.py`) | `scripts/` | yes (linked from `<machine>/scripts.md` or in-line in the `<machine>/README.md`) |
| Colab notebooks (`notebooks/colab_*.ipynb`) | `notebooks/` | yes (linked from `colab/notebooks.md`) |
| **Experiment-recipe scripts** (`scripts/run_b3_*.sh`, `run_b5_*.sh`, `run_b9_*.sh`, `run_f21c_*`, `run_f27_*`, `run_f37_*`, `run_f40_*`, `run_f48_*`, etc.) | `scripts/` | **NO** — these are experiment recipes, not ops. They run via the canonical CLI; the env they need is documented here |
| Historical ops handoffs (PHASE2_LIGHTNING_HANDOFF, GAP_A_RUNPOD_HANDOFF, H100_CAMERA_READY_GAPS) | `docs/archive/check2hgi-post-paper-closure-2026-05-01/` | NO — durable infra recipes already extracted into the per-environment READMEs above |

## Conventions

- The canonical CLI is `python scripts/train.py --task X --state Y --engine Z` (see `docs/NORTH_STAR.md` for the committed champion configs). Each environment guide here only documents the *how-to-bring-up* + *how-to-launch-without-losing-it* parts; the training command itself is invariant.
- Long runs (>5 min) on Colab MUST use the detached-subprocess pattern in `colab/README.md` — foreground `!{cmd}` cells get SIGINT'd by MCP/cell timeouts and lose your work.
- Drive layout convention is shared across Colab + Lightning + remote pods: `<DRIVE_ROOT>/PoiMtlNet/{data, output, results}` + a per-state subdir scheme. See `data/drive_download.md`.

## Migration history

This folder consolidates ops docs that previously lived in:
- `docs/RUNPOD_GUIDE.md` (now `docs/infra/runpod/README.md`)
- `docs/COLAB_GUIDE.md` (now `docs/infra/colab/README.md`)
- `scripts/H100_FLCATX_PERVISIT_PROMPT.md` (now `docs/infra/h100/README.md`)
- Various PHASE2/PHASE3 handoff docs in `docs/archive/check2hgi-post-paper-closure-2026-05-01/` (durable infra recipes extracted; archive originals retained as historical context)

The migration was part of the 2026-05-14 reorg (`docs/archive/MERGE_REORG_PLAN_2026-05-14.md`).
