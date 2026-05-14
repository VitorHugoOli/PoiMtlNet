# docs/infra/local/ — Local M4 Pro / Apple Silicon (MPS)

The default development environment for this project. Use the local M4 Pro for iteration, smoke tests, AL/AZ-scale runs, and any debugging that benefits from fast turnaround.

## Quick start

```bash
cd /Users/vitor/Desktop/mestrado/ingred           # repo root
source .venv/bin/activate                          # Python 3.12, PyTorch 2.9.1+
python scripts/train.py --task mtl --state alabama --engine check2hgi --folds 1 --epochs 2  # smoke
```

## Environment

- **Python**: 3.12 (in `.venv/` at repo root)
- **PyTorch**: 2.9.1+ (with MPS backend)
- **Device**: auto-detected via `src/configs/globals.py::DEVICE` — picks MPS on Apple Silicon, CUDA on Linux+GPU, CPU otherwise.
- **MPS fallback**: set `PYTORCH_ENABLE_MPS_FALLBACK=1` if you hit operators with poor MPS coverage. The codebase already has guards in `src/utils/mps.py`.
- **Cache management**: `src/utils/mps.py::clear_mps_cache()` is called between folds to avoid memory fragmentation. Don't disable it on long runs.

## Conventions

- Data symlinks: `data/` and `output/` at repo root are symlinks to host paths (gitignored). On a fresh clone, create them yourself:
  ```bash
  ln -s /Users/vitor/Desktop/mestrado/ingred/data data
  ln -s /Users/vitor/Desktop/mestrado/ingred/output output
  ```
- Results land in `results/` (gitignored).
- Logs land in `logs/` (gitignored).
- Tests: `pytest -x` from repo root. `pytest.ini` configures `pythonpath = .`.

## When to switch off local

Switch to Colab/RunPod/Lightning/H100 when:

- The state is FL/CA/TX (4702 regions on FL → MRR pairwise OOM, between-fold CUDA fragmentation matters more on big-state runs). See `colab/README.md` for the OOM mitigations.
- You want to run multi-seed Wilcoxon (n≥10) at FL — wall-clock makes it impractical on MPS (~5 h vs ~50 min on T4).
- You're iterating on the embedding pipeline (HGI / Check2HGI training takes hours on MPS for FL).

## What MPS does NOT support well (verified)

- `torch.compile` on MPS — DO NOT enable without per-test verification (memory `feedback_torch_compile_caution`).
- `torch.cuda.amp.autocast` — wrong device. On MPS, autocast adds overhead; only enable on CUDA. Memory `feedback_mps_autocast_overhead`.

## Performance reference

The MTL FL 5f×50ep run is the wall-clock benchmark: ~5 h on M4 Pro MPS vs ~50 min on Colab T4 vs ~19 min on RunPod RTX 4090 vs ~10 min on a Lightning A100 single-GPU.
