# docs/infra/colab/notebooks.md — Colab notebooks index

All Colab notebooks live in `notebooks/`. They're indexed here for discoverability.

## Training notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/colab_check2hgi_mtl.ipynb` | **Canonical training template.** Self-contained, mirrors `scripts/train.py` north-star config from `docs/NORTH_STAR.md`. Edit `STATE`, run cells in order. Use this for any one-off training run on Colab. |
| `notebooks/colab_training.ipynb` | Older generic training notebook (pre-check2hgi). Kept for reference; prefer `colab_check2hgi_mtl.ipynb` for new work. |
| `notebooks/colab_phase2_grid.ipynb` | Phase-2 grid runner (multi-config sweep). Specific to the check2hgi study's Phase 2. |
| `notebooks/colab_f27_validation.ipynb` | F27 cat-head sweep validation runner. Specific to one finding. |

## Embedding generation

| Notebook | Purpose |
|---|---|
| `notebooks/HGI.ipynb` | Reference HGI training pipeline (paper-faithful). See `notebooks/CLAUDE.md` for the full pipeline doc. |
| `notebooks/Location_Encoder.ipynb` | Location-encoder (Sphere2Vec/Space2Vec) reference. |
| `notebooks/Time_Encoder.ipynb` | Time2Vec reference. |
| `notebooks/colab_all_states_parallel_embeddings.ipynb` | Parallel embedding generation across all states on Colab. |
| `notebooks/embedding_analysis.ipynb` | Post-hoc embedding analysis / probe utilities. |

## Study-driven runner

| Notebook | Purpose |
|---|---|
| `notebooks/colab_study_runner.ipynb` | The study-driven Colab runner — pairs with `scripts/study/colab_runner.py`. Consumes `state.json`-enrolled tests, packages tarballs back. See [`study_runner.md`](study_runner.md). |

## Misc

| Notebook | Purpose |
|---|---|
| `notebooks/playground.ipynb` | Sandbox / scratch. Don't rely on it for anything reproducible. |

## Conventions

- All notebooks expect the Drive layout described in [`README.md`](README.md) §Drive layout.
- Long-running cells (>5 min) use the **detached-subprocess pattern** (mandatory — MCP/cell timeouts will SIGINT a foreground `!{cmd}`). See [`README.md`](README.md) for the pattern.
- After reconnect, **always check Colab runtime type before running GPU work** — sessions often come back as CPU-only after disconnect (memory `feedback_check_colab_runtime_type`).
