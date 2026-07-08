# MobiWac 2026 — Anonymous Code Release

Companion code for the MobiWac 2026 submission on multi-task next-POI prediction.
One model is trained jointly on two tasks over LBSN check-in sequences:

- **next-category** — predict the category of the user's next check-in (7 classes);
- **next-region** — predict the census-tract-level region of the next check-in
  (~1.1k–8.5k classes depending on the state).

The paper evaluates on five U.S. states from the Gowalla dataset (Alabama, Arizona,
Florida, California, Texas) plus Istanbul (Massive-STEPS), and compares the joint
model against dedicated single-task models ("ceilings") and six external baselines
(STAN, ReHDM, POI-RGNN, HMT-GRN, CTLE, and a two-stage cascade).

This repository is the anonymized reviewer snapshot: code only. All author and
institutional identifiers have been scrubbed; the development repository will be
made public after acceptance.

## What is in this repo

| Path | Purpose |
|---|---|
| `src/` | The MTL framework — model, task heads, losses, folds, training runners, configs, tracking |
| `research/embeddings/check2hgi/` | The check-in-level graph representation (encoder, preprocessing, trainer) |
| `research/embeddings/hgi/` | Place-level hierarchical graph infomax (region encoder + Delaunay graph + POI2Vec teacher, consumed by the substrate build) |
| `research/baselines/{stan,rehdm,poi_rgnn}/` | External baselines with their own ETL + trainers |
| `scripts/baselines/` | Remaining baselines: HMT-GRN (`b3_hmt_grn.py`), cascade (`b4_cascade.py`), CTLE (`build_ctle_substrate.py`, `ctle_e2e.py`, `ctle_lib/`) |
| `scripts/` | CLI entrypoints: `train.py`, `evaluate.py`, substrate/input builders, fold fan-out, transition priors, simple baselines |
| `scripts/closing_data/` | The paper's run recipes (`p3_board.sh`, `run_catx_v17_seed0_5f.sh`, `run_catx_v17_n20.sh`), matched scorers, and pre-registered statistical tests |
| `scripts/second_dataset/` | Istanbul (Massive-STEPS) ETL: acquisition, category mapping, parsing, graph build, splits, inputs, substrate training |
| `analysis/` | Paper analysis scripts: region non-inferiority TOST, near-miss distance analyses, shortlist compactness, co-visitation network |
| `pipelines/` | Thin pipeline wrappers (Gowalla ETL, embedding generation, input creation) |
| `tests/` | Unit + regression test suite |

## What is NOT included (and why)

- **Raw data** — Gowalla check-ins, census shapefiles, and Massive-STEPS are
  third-party datasets; download them from their original sources (Section 2).
  The category-annotated Gowalla dump cannot be redistributed here; see the
  paper's data statement for provenance.
- **Trained weights and embeddings** — several GB per state; all are fully
  regenerable from the commands below.
- **Result JSONs / per-fold metric files** — the statistical scripts in
  `scripts/closing_data/` and `analysis/` read per-fold score files that were
  produced by the training runs; they are not shipped to keep the release lean
  (and because they carry machine-specific paths). Running the recipes below
  regenerates them; the paper's tables carry the aggregated numbers.

---

## 1. Environment

Python **3.12.x**, PyTorch **2.11.0**, PyTorch-Geometric 2.7.0.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- The paper's GPU cells were run with `torch==2.11.0+cu128` (CUDA 12.8 wheels) on
  NVIDIA A40/H100; `scripts/train.py` warns when the torch build differs.
- CPU/Apple-Silicon runs work for the ETL and small smoke tests; training the
  paper cells requires a CUDA GPU (large states peak ~26–29 GB VRAM).
- Everything below assumes the repo root as working directory and
  `PYTHONPATH=src` (the scripts set it themselves when possible).

Run the test suite to validate the environment:

```bash
pytest tests/ -q
```

## 2. Download data

**Gowalla check-ins (five U.S. states).** The base check-in dump is the public
SNAP release (Cho, Myers, Leskovec, KDD 2011:
<https://snap.stanford.edu/data/loc-Gowalla.html>). The paper additionally uses a
**category-annotated** Gowalla dump (per-POI category labels); it must be obtained
per the paper's data statement. Expected raw layout (paths in
`src/configs/paths.py → Resources`):

| File | Source |
|---|---|
| `data/gowalla/gowalla_checkins.parquet` | Gowalla check-in dump (SNAP release) |
| `data/gowalla/gowalla_spots_subset{1,2}.csv` | Gowalla auxiliary POI tables (categories) |
| `data/gowalla/gowalla_category_structure.json`, `callback_categories.json`, `extra_categories.json` | Gowalla category dictionaries |
| `data/miscellaneous/tl_2022_us_state/` | U.S. states shapefile — Census TIGER 2022 (<https://www2.census.gov/geo/tiger/TIGER2022/STATE/>) |
| `data/miscellaneous/tract/tl_2022_XX_tract/` | Census-tract shapefiles, TIGER 2022, one per state FIPS code XX (<https://www2.census.gov/geo/tiger/TIGER2022/TRACT/>) — the region label space |

**Istanbul (Massive-STEPS).** Downloaded automatically from HuggingFace
(dataset id `cruiseresearchgroup/Massive-STEPS-Istanbul`):

```bash
python scripts/second_dataset/acquire.py --city istanbul
```

## 3. ETL and input creation

### 3.1 Gowalla → per-state check-in tables

```bash
python pipelines/etl/gowalla.pipe.py
```

Three stages: (1) label POIs against the Gowalla category structure;
(2) optional local-time attachment via timezone polygons (skippable);
(3) spatial-join with state polygons → one CSV per state at
`data/checkins/<State>.csv`.

### 3.2 Per-state substrate build (Gowalla states)

Each state's model inputs are built from a trained check-in-level representation
(Section 4). The chain, per state, is:

```bash
# (a) base representation + structural graph (Section 4.1) — prerequisites
# (b) the paper's representation variant: Delaunay POI-graph + teacher-anchored
python scripts/probe/build_design_k_delaunay.py --state <state> --epochs 500
# (c) re-window the sequence inputs at stride 1 with minimum sequence length 10
#     (embeddings are shared by symlink; only the windowing differs)
PYTHONPATH=src python scripts/mtl_improvement/build_overlap_probe_engine.py <state> 1 10
# (d) leak-free per-fold region-transition priors, per seed
PYTHONPATH=src python scripts/compute_region_transition.py --state <state> --per-fold --seed 0
```

Steps (b)–(d) are exactly what the board driver
`scripts/closing_data/p3_board.sh` automates (it also stages the priors and
runs the training cells of Section 5; `--dry-run` prints the full plan).

### 3.3 Istanbul ETL and inputs

Run the `scripts/second_dataset/` chain in order (all take `--city istanbul`):

```bash
python scripts/second_dataset/acquire.py            --city istanbul   # raw parquets + category tree
python scripts/second_dataset/build_category_map.py --city istanbul   # FSQ categories → the 7-class scheme
python scripts/second_dataset/parse_city.py         --city istanbul   # → data/checkins/Istanbul.csv
python scripts/second_dataset/build_h3_boroughs.py  --city istanbul   # H3-hexagon region file (tract analogue)
python scripts/second_dataset/build_graph.py        --city istanbul   # structural check-in graph
python scripts/second_dataset/build_chrono_split.py --city istanbul   # chronological per-user 80/10/10 split
python scripts/second_dataset/build_inputs.py       --city istanbul   # windowed sequences + labels
python scripts/second_dataset/phase_v_substrate.py  --city istanbul   # train the representation on the fixed graph
python scripts/build_istanbul_stride1.py                              # stride-1 / min-length-10 windowing (as 3.2c)
```

## 4. Train the representation

The check-in-level representation lives in `research/embeddings/check2hgi/`
(GCN encoder over a check-in graph with POI/region hierarchy pooling, trained
with an infomax objective; `preprocess.py` builds the graph from the per-state
CSV + tract shapefiles).

### 4.1 Base build (per state)

```bash
python pipelines/embedding/check2hgi.pipe.py     # states configured at the top of the file
python pipelines/embedding/hgi.pipe.py           # place-level HGI: also emits the Delaunay
                                                 # edge list + POI2Vec teacher used in 4.2
```

Outputs land under `output/check2hgi/<state>/` and `output/hgi/<state>/`.

### 4.2 Paper variant

The paper's substrate adds a Delaunay POI-graph GCN on the region path and a
POI2Vec-anchored learnable table (built on top of 4.1):

```bash
python scripts/probe/build_design_k_delaunay.py --state <state> --epochs 500
```

then re-window with `build_overlap_probe_engine.py <state> 1 10` (Section 3.2).

## 5. Train the models

### 5.1 Joint (multi-task) model

The exact paper recipe is in the two run scripts (batch size 8192, static loss
weighting 0.75/0.25, per-head one-cycle LRs, fp32, GRU category head + dual-tower
spatio-temporal region head, cross-attention MTL trunk):

- `scripts/closing_data/run_catx_v17_seed0_5f.sh` — California/Texas, seed 0, 5 folds;
- `scripts/closing_data/run_catx_v17_n20.sh` — California/Texas, seeds {0, 1, 7, 100} (n = 20 folds);
- `scripts/closing_data/p3_board.sh` — the same recipe driven across all states × seeds.

The core command they wrap (one state, one seed):

```bash
PYTHONPATH=src python scripts/train.py --task mtl --canon none \
    --task-set check2hgi_next_region --engine check2hgi_dk_ovl \
    --state <state> --seed <seed> --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 \
    --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --checkpoint-selector geom_simple \
    --compile --tf32 --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/<state>
```

(Environment knobs — fp32 via `MTL_DISABLE_AMP=1`, per-head one-cycle via
`MTL_ONECYCLE_PER_HEAD_LR=1`, chunked validation metrics — are set inside the
run scripts.) Istanbul uses the smaller-state variant in
`scripts/run_istanbul_champion_stride1.sh`. Per-fold parallel fan-out of one run
is available via `scripts/run_folds_fanout.sh` + `scripts/aggregate_folds.py`.

### 5.2 Dedicated single-task ceilings

The exact two-cell recipe is `scripts/closing_data/stl_ceilings.sh` (written for
Florida; set `ST=<state>` at the top). The commands it wraps:

```bash
# next-category ceiling: single-task GRU on the same inputs/folds
PYTHONPATH=src python scripts/train.py --task next --engine check2hgi_dk_ovl \
    --state <state> --seed <seed> --epochs 50 --folds 5 --batch-size 2048 \
    --model next_gru --max-lr 3e-3
# score it:
python scripts/closing_data/score_stl_cat_ceiling.py <rundir>

# next-region ceiling: single-task region head on the same inputs/folds
PYTHONPATH=src python scripts/p1_region_head_ablation.py --state <state> \
    --heads next_stan_flow --input-type region --target region \
    --engine-override check2hgi_dk_ovl \
    --region-emb-source check2hgi_design_k_resln_mae_l0_1 \
    --override-hparams freeze_alpha=True alpha_init=0.0 \
    --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/<state> \
    --folds 5 --epochs 50 --seed <seed>
```

### 5.3 Baselines (one line each)

```bash
python research/baselines/stan/train.py --state <state>                     # STAN
python research/baselines/rehdm/train.py --state <state>                    # ReHDM
python research/baselines/poi_rgnn/train.py --state <state>                 # POI-RGNN
python scripts/baselines/b3_hmt_grn.py --state <state> --seed 0             # HMT-GRN
python scripts/baselines/build_ctle_substrate.py --state <state> --seed 0 \
  && python scripts/baselines/ctle_e2e.py --state <state> --seed 0          # CTLE
python scripts/baselines/b4_cascade.py --state <state> --seed 0             # cascade (cat → region)
```

Simple non-learned reference points (majority class, Markov transition):
`scripts/compute_simple_baselines.py`, `scripts/compute_markov_kstep_cat.py`.

### 5.4 Scoring

Matched-protocol scorers (per-fold category macro-F1 at the F1-best epoch +
region Acc@10 with out-of-distribution correction, identical readout for MTL
and ceilings): `scripts/closing_data/a40_score_matched.py` and
`scripts/closing_data/h100_score_matched.py <rundir> --seed <seed>`.

## 6. Statistics and analysis

Pre-registered tests (margins and test families fixed before the final runs):

```bash
python scripts/closing_data/superiority_wilcoxon.py   # paired one-sided Wilcoxon + Holm:
                                                      # category superiority (all datasets),
                                                      # region superiority (FL/CA/TX)
python scripts/closing_data/region_match_tost.py      # TOST non-inferiority (δ = 2 pp) for the
                                                      # small-state region cells (AL/AZ/Istanbul)
python scripts/closing_data/m1_stats_n20.py           # the same families at the n=20 footing
```

These read the per-fold score files produced by Section 5 (not shipped — see
"What is NOT included").

Paper analysis scripts (in `analysis/`; repo-root autodetected, overridable via
`REPO_ROOT`):

```bash
python analysis/tost_region.py             # region TOST with per-fold pairing details
python analysis/near_miss_distance.py      # are wrong region predictions spatially near?
python analysis/near_miss_floor.py         # random-pair distance floor for the same geometry
python analysis/shortlist_compactness.py   # top-10 shortlist spatial compactness
python analysis/covisitation_network.py    # co-visitation network structure of the states
```

## License

MIT (see `LICENSE`). Released anonymously for review; the public repository will
carry full attribution after acceptance.

## Notes for reviewers

- Reproduction cost: one Gowalla state cell (joint model, 5 folds, seed 0) is
  ~1 h/fold on an A40 for the largest states, minutes/fold for small states;
  representation builds are ~10–30 min/state on one GPU.
- Multi-seed paper cells use seeds {0, 1, 7, 100} (seed 42 was the development
  seed and is deliberately excluded from reported numbers).
- For questions during review, please use the conference review system.
