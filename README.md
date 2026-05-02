# BRACIS 2026 — Anonymous Code Release

Companion code for the paper *Substrate Carries, Architecture Pays: Multi-Task POI
Prediction with Check-in-Level Hierarchical Embeddings*.

This repository is the anonymized snapshot used by reviewers. All author and
institutional identifiers have been scrubbed; the live development repository
will be made public after acceptance.

## What is in this repo

| Path | Purpose |
|---|---|
| `src/` | MTLnet framework — model, heads, losses, training, configs, tracking |
| `research/embeddings/` | Embedding engines: DGI, HGI, **Check2HGI**, POI2HGI, Time2Vec, Space2Vec, Sphere2Vec, HMRM |
| `research/baselines/` | External baselines: STAN, REHDM, POI-RGNN, MHA-PE |
| `pipelines/` | Embedding-generation and training pipeline wrappers |
| `scripts/` | CLI entrypoints (`train.py`, `evaluate.py`, `run_*_ablation.py`) and paper analysis scripts (`scripts/analysis/*.py`) |
| `experiments/` | Declarative experiment configs (Python and Hydra YAML) |
| `tests/` | Unit + regression test suite (~1,000 tests) |
| `articles/[BRACIS]_Beyond_Cross_Task/` | Paper sources (LNCS LaTeX + bib) and figures |
| `docs/studies/check2hgi/` | `RESULTS_TABLE.md` (canonical numerics §0) and the paper-facing claim whitelist |

## Project at a glance

**MTLnet** is a multi-task framework that jointly trains two POI-prediction tasks:
- **next-category** — predict the category of the next check-in (7 classes)
- **next-region** — predict the next region the user will visit (~1.1k–8.5k classes)

The headline finding (RESULTS_TABLE.md §0) is a substrate / architecture decomposition:
the **Check2HGI** check-in-level graph embedding is the primary driver of next-category
gains (+14 to +29 pp F1 over place-level HGI in STL), while the architectural MTL
component yields a sign-consistent classic tradeoff across five U.S. states.

## Reproducibility

### Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Dependencies are pinned in `pyproject.toml`. The NashMTL solver chain requires
`cvxpy + ecos`; both are pinned. `.venv` Python must be **3.12.x**.

### Data

This release contains code only. The paper uses **Gowalla check-ins** (public dataset;
cite via the references in the paper). The training pipeline expects per-state inputs at
`data/checkins/<State>.csv`. Two routes:

**Route A — run the included ETL** (`pipelines/etl/gowalla.pipe.py`).
Place raw Gowalla files under `data/gowalla/` and the U.S. states shapefile under
`data/miscellaneous/`, then:

```bash
python pipelines/etl/gowalla.pipe.py    # SKIP_LOCALISE=True by default
```

The pipe runs three stages:

1. **Stage 1** — load `gowalla_checkins.parquet`, label POIs against the Gowalla
   category structure JSONs.
2. **Stage 2** *(optional)* — attach `local_datetime` via timezone polygons
   (skip with `SKIP_LOCALISE = True` if you don't need wall-clock-local times).
3. **Stage 3** — spatial-join check-ins with U.S. state polygons and emit one
   CSV per state under `data/checkins/<State>.csv`.

Required raw inputs (paths in `src/configs/paths.py → Resources`):

| File | Source |
|---|---|
| `data/gowalla/gowalla_checkins.parquet` | SNAP Gowalla dump (https://snap.stanford.edu/data/loc-Gowalla.html) |
| `data/gowalla/gowalla_spots_subset{1,2}.csv` | Gowalla auxiliary POI tables |
| `data/gowalla/gowalla_category_structure.json` + `callback_categories.json` + `extra_categories.json` | Gowalla category dictionaries |
| `data/miscellaneous/tl_2022_us_state/` | Census TIGER 2022 (https://www2.census.gov/geo/tiger/TIGER2022/STATE/) |
| `data/miscellaneous/combined-shapefile-with-oceans/` *(optional)* | timezone-boundary-builder release (https://github.com/evansiroky/timezone-boundary-builder/releases) |

**Route B — provide pre-processed CSVs.** If you already have per-state Gowalla
CSVs, place them at `data/checkins/<State>.csv` and skip the ETL. The
`src/etl/gowalla/main.py` docstring documents the schema the downstream
embedding pipelines expect.

### Run the test suite

```bash
pytest tests/ -q
```

Expected result: **~980 passed, ~28 skipped, 7 known pre-existing failures**
(documented in `docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md`; none affect the
paper's claims). The 7 pre-existing failures are: an enum-vs-test drift in
`test_paths.py`, two metadata-coverage tests in
`test_research_variants_metadata.py`, an HGI hard-negative reference test, and
three GradNorm tests. The Check2HGI / HGI / MTL pipelines used for the paper all
pass under `tests/test_integration/`, `tests/test_models/`, `tests/test_training/`,
and `tests/test_regression/`.

### Reproduce the paper numbers

The canonical recipe (see `docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md` and
`articles/[BRACIS]_Beyond_Cross_Task/samplepaper.tex`) is:

```bash
# 0. (one-time) raw Gowalla → per-state CSVs — see "Data" above
python pipelines/etl/gowalla.pipe.py

# 1. Train Check2HGI / HGI substrate embeddings (per state)
python pipelines/embedding/check2hgi.pipe.py
python pipelines/embedding/hgi.pipe.py

# 2. Build per-fold leak-free transition priors
python scripts/compute_region_transition.py --state alabama

# 3. Train MTL model (canonical recipe = B9 for FL/CA, H3-alt for AL/AZ/TX)
python scripts/train.py --task mtl --state alabama --engine check2hgi

# 4. Run paired Wilcoxon Δ tests for the substrate / architecture decomposition
python scripts/analysis/paper_closure_wilcoxon.py
python scripts/analysis/gap_fill_wilcoxon.py    # multi-seed cat-Δ
python scripts/analysis/substrate_paired_test.py
python scripts/analysis/f50_delta_m_leakfree.py # joint Δm score (CH22)
```

Multi-seed runs use seeds {0, 1, 7, 42, 100}. Hardware used in the paper: NVIDIA H100
(checkpoints generated on a single H100 80GB; per-fold runs ~5–10 min each).

## License

Released for review purposes. License determination pending acceptance — the public
post-acceptance release will carry an MIT or Apache-2.0 license.

## Notes for reviewers

- `articles/[BRACIS]_Beyond_Cross_Task/samplepaper.tex` is the paper source.
- All numerical claims trace to `docs/studies/check2hgi/results/RESULTS_TABLE.md §0`
  (single source of truth) and the JSONs under `docs/studies/check2hgi/results/`
  have been removed from this snapshot to keep the release lean — see the paper for
  the per-fold tables.
- For questions during review, please use the conference review system. Author
  contact will be restored after acceptance.
