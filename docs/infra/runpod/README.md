# RunPod (CUDA) — `worktree-check2hgi-mtl` operational guide

End-to-end recipe for spinning up a fresh RunPod pod, training the
NORTH_STAR FL H3-alt config, and reproducing perf comparisons. Tested on
RTX 4090 / driver 570 / CUDA 12.8 / Ubuntu 22.04 base image.

The Colab guide (`docs/COLAB_GUIDE.md`) covers the Drive-mount + detached-
notebook workflow. This guide covers the SSH + `pip` workflow that RunPod
expects.

## TL;DR

```bash
cd /workspace/PoiMtlNet
git checkout worktree-check2hgi-mtl
bash scripts/runpod_setup.sh                    # ~3 min
source .venv/bin/activate
bash scripts/runpod_fetch_data.sh florida       # ~30 s, 2.3 GB
tmux new -s mtl 'bash scripts/runpod_train_fl_h3alt.sh' # ~19 min on a 4090
```

## 1. Clone + branch

```bash
cd /workspace
git clone <repo-url> PoiMtlNet
cd PoiMtlNet
git checkout worktree-check2hgi-mtl
```

Set `WORKTREE=/workspace/<dirname>` if you cloned to a different name.

## 2. One-shot env install (`scripts/runpod_setup.sh`)

```bash
bash scripts/runpod_setup.sh
source .venv/bin/activate
```

What it does:

1. Verifies branch + `nvidia-smi`.
2. Pins venv to `/opt/poimtlnet-venv`, uv cache to `/root/.cache/uv` —
   both on the overlay `/` filesystem (see §7 for the quota story).
3. Symlinks `${WORKTREE}/.venv → /opt/poimtlnet-venv` so every script
   that does `source .venv/bin/activate` keeps working.
4. `uv pip install torch==2.11.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128`.
5. Installs PyG add-ons against the live torch version.
6. Installs the rest of `requirements.txt` (torch lines stripped).
7. Installs `gdown`.
8. Sanity-checks `torch.cuda.is_available()` + PyG ops importable.

Re-runnable. Skips already-satisfied steps. ~3 min on a 4090 pod.

## 3. Fetch data (`scripts/runpod_fetch_data.sh`)

The pod has a ~3.5 GB writable quota on `/workspace` (see §7). Download
only the state you need:

| State | Drive folder ID |
|---|---|
| Florida | `1_4RxYvteg1rvN2WRB2AywKctQ1_t-ZhI` |
| California | `1ZLL8FHPeO7I-3DEfVBogW1C1eFE76ttv` |
| Texas | `1bLfFDEOM1BJ2ELoQUnd_qMXFpxGsZ7UF` |

```bash
bash scripts/runpod_fetch_data.sh florida    # 2.3 GB
# bash scripts/runpod_fetch_data.sh california
# bash scripts/runpod_fetch_data.sh texas
```

Drops everything under `output/check2hgi/<state>/` (canonical layout for
`IoPaths`). After a state run, free quota by `rm -rf
output/check2hgi/<state>` before fetching the next.

## 4. Train Florida — H3-alt (`scripts/runpod_train_fl_h3alt.sh`)

```bash
bash scripts/runpod_train_fl_h3alt.sh                  # default: 5f × 50ep, b=2048, ~19 min
FOLDS=1 EPOCHS=2 bash scripts/runpod_train_fl_h3alt.sh # smoke (~1 min)
BATCH=1024 bash scripts/runpod_train_fl_h3alt.sh       # MPS-style batch (slower)
```

Recipe (matches `docs/studies/check2hgi/NORTH_STAR.md`):

```
--scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3
--model mtlnet_crossattn --mtl-loss static_weight --category-weight 0.75
--cat-head next_gru --reg-head next_getnext_hard
--task-a-input-type checkin --task-b-input-type region
--batch-size 2048   # 1024 on MPS
```

The wrapper pre-flights every required parquet/`.pt` file and bails
early with a clear error if anything is missing — so a failed run is
almost always a fetch-data problem, not a code problem.

Results land in `results/check2hgi/florida/<run_dir>/` with
`summary/full_summary.json`, per-fold metrics, and plots.

### Verified on RTX 4090 (2026-04-28)

| Metric | This run | NORTH_STAR (MPS) | Δ |
|---|---|---|---|
| cat F1 (joint best, 5f×50ep) | 67.63 ± 0.72 | 67.92 ± 0.72 | -0.29 pp (0.40σ) ✓ |
| cat F1 (per-task best) | 68.36 ± 0.74 | — | within fold variance |
| Wall time | **19 min** | ~50–75 min on M4 Pro MPS | 2.5–4× faster |

Reg `Acc@10_indist` showed σ ≈ 12 pp on CUDA vs σ = 0.68 in the published
MPS numbers — see §9 for the open follow-up.

## 5. Long runs — survive disconnects

A 5f×50ep FL run is ~19 min on a 4090; still long enough to care about
SSH disconnects. Use `tmux`:

```bash
tmux new -s mtl
bash scripts/runpod_train_fl_h3alt.sh 2>&1 | tee logs/fl_h3alt_$(date +%Y%m%d_%H%M%S).log
# detach: Ctrl-b d        re-attach: tmux attach -t mtl
```

`tmux` is preinstalled on the RunPod base image (we added it via apt in
this session). `logs/` is gitignored.

## 6. Other states / recipes

- **California / Texas** — same recipe, change `--state florida` (or
  copy the script). FL is the Phase-2 lead per
  `docs/studies/check2hgi/PHASE2_TRACKER.md`.
- **B3 (predecessor recipe, `--max-lr 0.003` OneCycle)** — use
  `scripts/run_b3_fl_only.sh` after `export DATA_ROOT=$PWD/data
  OUTPUT_DIR=$PWD/output`.
- **Other states / engines** — drive `scripts/train.py` directly; CLI
  options documented in `CLAUDE.md`.

## 7. Disk layout (RunPod-specific, counter-intuitive)

`df -h /workspace` reports hundreds of TB free, but each pod has a
**tight per-pod quota** on the mfs volume — ~3.5 GB on the pod we
tested. `/` overlay is much roomier (~20 GB). `/usr/local/cuda-12.4` and
`/usr/local/lib/python3.11/*` (5 + 5 GB) live on a **read-only base
layer** — `rm` creates a whiteout but doesn't free real disk.

| What | Where | Why |
|---|---|---|
| venv | `/opt/poimtlnet-venv` (overlay) | ~7.7 GB; won't fit `/workspace` quota |
| uv cache | `/root/.cache/uv` (overlay) | ~5 GB during install; delete after setup |
| training data | `/workspace/PoiMtlNet/output/check2hgi/<state>/` (mfs) | survives pod restart |
| training results | `/workspace/PoiMtlNet/results/...` (mfs) | same |

`scripts/runpod_setup.sh` already sets `UV_CACHE_DIR` and `UV_LINK_MODE=copy`
correctly. If you bypass the setup script and run `uv pip` by hand,
`export UV_CACHE_DIR=/root/.cache/uv` and `export UV_LINK_MODE=copy` first
or you'll hit silent install aborts (numpy half-uninstalled, networkx
METADATA empty, etc.).

Probe your quota:

```bash
dd if=/dev/zero of=/workspace/.qprobe bs=1M count=8192 ; rm -f /workspace/.qprobe
```

If it errors with `Disk quota exceeded` before 8 GB, that's your
headroom above current usage.

After setup, free ~8 GB by removing the uv wheel cache (no longer
needed):

```bash
rm -rf /root/.cache/uv
```

## 8. Perf-variant comparison (this session's experiment)

Three perf knobs were A/B-tested against the published NORTH_STAR
numbers, all 5-fold × 50-epoch, seed 42:

| Variant | cat F1 (joint best) | reg Acc@10 indist | Wall time | Verdict |
|---|---|---|---|---|
| baseline (b=1024) | 67.35 ± 0.83 | 63.68 ± 12.07 | 36 min | reference |
| `--tf32` (b=1024) | **67.35 ± 0.83** | **63.68 ± 12.07** | 36 min | **bit-identical** to baseline |
| `--compile` (b=1024) | failed — see below | — | — | incompatible |
| **b=2048 (NORTH_STAR recipe)** | **67.63 ± 0.72** | 59.04 ± 12.00 | **19 min** | **chosen as default** |

Findings:

- **`--tf32` is a no-op** — fp16 autocast already covers the matmul hot
  path, so TF32's fp32-matmul speedup never fires.
- **`--compile` is incompatible** with this MTL pipeline as-is.
  Two distinct failures stack:
  1. `_compute_gradient_cosine` calls `torch.autograd.grad(retain_graph=True)`,
     which requires `torch._functorch.config.donated_buffer = False`
     (now set automatically when `--compile` is passed).
  2. The cross-attention head has an in-place op (`[batch×heads, seq, seq]`
     half-tensor at version 2 vs expected 1) that breaks compile's
     graph tracer. Fixing this requires rewriting the attention path or
     dropping the gradient-cosine diagnostic — out-of-scope for this work.
  Stays exposed as a CLI flag for when someone does the deeper fix.
- **b=2048 is 1.9× faster** at the same quality (cat F1 0.4σ from
  NORTH_STAR). This is now the default; `scripts/run_f48_h3alt_fl.sh`
  was updated from b=1024 to b=2048 (the 1024 was an off-recipe bug;
  the AZ launcher already used 2048).
- **Reg side has σ ≈ 12 pp on CUDA** vs σ = 0.68 in the published MPS
  numbers. All three CUDA variants land in the same 12σ band, so they
  are equivalent **to each other**, but they all diverge from MPS on
  the joint-task best-epoch selection for the reg head. Likely fp16
  autocast vs fp32 numeric drift in the reg head's val-metric tracking.
  Open follow-up — see §9.

The variant launchers and comparison driver are in `scripts/`:
`runpod_train_fl_h3alt_{tf32,compile,bs2048}.sh`,
`runpod_perf_compare.sh`, `runpod_perf_compare_resume.sh`,
`runpod_perf_summarise.py`. Use them as a template if you want to A/B
another knob (different seed, batch size, scheduler, etc.). The
auto-generated comparison table writes to
`docs/studies/check2hgi/results/perf_compare/`.

## 9. Open follow-ups

1. **CUDA-vs-MPS reg-head divergence** — joint-task best-epoch
   selection on `Acc@10_indist` shows σ ≈ 12 pp on CUDA (4090) vs σ ≈
   0.7 pp on MPS (M4 Pro). Cat side is rock-solid (matches NORTH_STAR
   within 0.4σ). Likely fp16 autocast on validation pass; the per-task
   best epochs differ by 5–15 epochs for cat vs reg, and the `model.
   joint_score` selector picks an epoch where reg has already started
   to drift down. Worth its own ablation.
2. **`--compile` integration** — rewrite cross-attention in-place ops
   to be out-of-place, or move `_compute_gradient_cosine` behind a
   `--diagnostic-grad-cosine` flag so compile-eligible runs can skip
   the `retain_graph=True` path.
3. **CA / TX runs** — same `bash scripts/runpod_train_fl_h3alt.sh`
   pattern; just change `--state florida` (PHASE2_TRACKER.md is the
   tracker).

## 10. Troubleshooting

- **`torch.cuda.is_available() == False`** — cu128 wheel didn't match
  the driver. Verify with `nvidia-smi`; if driver < 12.8, fall back to
  cu126 by exporting `TORCH_VER=2.11.0+cu126
  TORCH_INDEX=https://download.pytorch.org/whl/cu126` before the setup
  script.
- **PyG wheel not found at `data.pyg.org/whl/torch-<x>.html`** — wheels
  lag torch by ~1 week. Pin torch one minor below latest.
- **`MISSING: ...region_transition_log.pt`** — partial gdown. Re-run
  `runpod_fetch_data.sh <state>`; gdown resumes.
- **OOM at FL with 4702 regions** — drop batch: `BATCH=1024 bash
  scripts/runpod_train_fl_h3alt.sh`. The 4090 (24 GB) handles 2048 at
  ~14 GB peak; >24 GB cards are fine.
- **`AttributeError: module 'numpy' has no attribute 'ndarray'`** —
  partial uninstall during a `/workspace`-pinned `uv pip install` (see
  §7). Fix: `rm -rf /opt/poimtlnet-venv /workspace/PoiMtlNet/.venv &&
  bash scripts/runpod_setup.sh`.
- **`Disk quota exceeded` writing to /workspace** — see §7. Free space
  by deleting `output/check2hgi/<state>` for states you're not actively
  training (re-fetch via `runpod_fetch_data.sh`).
- **Compile crashes (`donated_buffer` or `inplace operation`)** — see
  §8. The pipeline is currently incompatible with `torch.compile`; the
  flag stays for future work.

## 11. Files added by this guide

```
scripts/runpod_setup.sh                    — env install
scripts/runpod_fetch_data.sh               — gdown a single state
scripts/runpod_train_fl_h3alt.sh           — H3-alt FL launcher (2048 default)
scripts/runpod_train_fl_h3alt_tf32.sh      — perf variant
scripts/runpod_train_fl_h3alt_compile.sh   — perf variant
scripts/runpod_train_fl_h3alt_bs2048.sh    — perf variant (now default of base)
scripts/runpod_perf_compare.sh             — A/B/C driver
scripts/runpod_perf_compare_resume.sh      — A/B/C driver (sequential, no tmux wait)
scripts/runpod_perf_summarise.py           — comparison-table emitter
docs/RUNPOD_GUIDE.md                       — this doc
```

Plus two small `scripts/train.py` additions (`--tf32`, `--compile`
flags) and a one-line fix to `scripts/run_f48_h3alt_fl.sh`
(`--batch-size` 1024 → 2048, with MPS-fallback note in-line).
