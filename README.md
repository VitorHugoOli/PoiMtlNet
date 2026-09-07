# MobiWac 2026 — Code Release

Code for *"Predicting the Next Category and Region of a Visit: A Check-in-Level
Multi-Task Study on Mobility Data"* (MobiWac 2026, accepted). One model is
trained jointly on two tasks over LBSN check-in sequences:

- **next-category** — predict the category of the user's next check-in (7 classes);
- **next-region** — predict the census-tract-level region of the next check-in
  (~1.1k–8.5k classes depending on the state).

The paper evaluates on five U.S. states from the Gowalla dataset (Alabama, Arizona,
Florida, California, Texas) plus Istanbul (Massive-STEPS), and compares the joint
model against dedicated single-task models ("ceilings") and six external baselines
(STAN, ReHDM, POI-RGNN, HMT-GRN, CTLE, and a two-stage cascade).

**Authors:** Vitor H. O. Silva, Germano B. dos Santos, Fabrício A. Silva —
NESPeD-LAB, Universidade Federal de Viçosa, Florestal, MG, Brazil;
`{vitor.h.oliveira, germano.santos, fabricio.asilva}@ufv.br`.

> **Update note.** This branch previously shipped the numbers and recipe from an
> earlier substrate revision (the "v17" scripts/results below). The accepted
> paper reports a corrected representation (a check-in graph leak was found and
> fixed — see the note at the top of Section 5). This update corrects the
> reported training recipe, desanonymizes the release, and adds the two
> aggregated result files the accepted paper's own statistical test reads (see
> Section 6) — [`wilcoxon_v18.py`](research/reproducibility/mobiwac_v18/wilcoxon_v18.py)
> is runnable as shipped. It does **not** yet include a ported/sanitized copy of
> the five *pre-registered* statistical-analysis scripts (Section 6 still points
> them at the pre-correction result files) or the raw per-run regeneration
> artifacts (rundirs/logs) the two aggregates above were built from — those are
> a follow-up.

## What is in this repo

| Path | Purpose |
|---|---|
| `src/` | The MTL framework — model, task heads, losses, folds, training runners, configs, tracking |
| `research/embeddings/check2hgi/` | The check-in-level graph representation (encoder, preprocessing, trainer) |
| `research/embeddings/hgi/` | Place-level hierarchical graph infomax (region encoder + Delaunay graph + POI2Vec teacher, consumed by the substrate build) |
| `research/baselines/{stan,rehdm,poi_rgnn}/` | External baselines with their own ETL + trainers |
| `scripts/baselines/` | Remaining baselines: HMT-GRN (`b3_hmt_grn.py`), cascade (`b4_cascade.py`), CTLE (`build_ctle_substrate.py`, `ctle_e2e.py`, `ctle_lib/`) |
| `scripts/` | CLI entrypoints: `train.py`, `evaluate.py`, substrate/input builders, fold fan-out, transition priors, simple baselines |
| `scripts/closing_data/` | Matched scorers and the statistical tests (current); `run_catx_v17_seed0_5f.sh` / `run_catx_v17_n20.sh` are historical run scripts for the superseded substrate, kept for provenance — see Section 5 for the reported recipe |
| `research/reproducibility/mobiwac_v18/` | Reproduction scripts for the accepted paper's numbers: parameter-count audit (`param_counts.py`) and the paired Wilcoxon + t-test (`wilcoxon_v18.py`) — both runnable as shipped, see Section 6 |
| `docs/results/closing_data/v18/`, `docs/studies/closing_data/v18/data/` | The two aggregated per-fold/per-run result files `wilcoxon_v18.py` reads |
| `analysis_protocol/` | The analysis plan, its deviation log, the executed analysis, and the epoch-selection record (Section 6) |
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
  regenerates them; the paper's tables carry the aggregated numbers. Four
  exceptions are shipped because a claim depends on them: the four per-fold
  arrays for Istanbul's dedicated category ceiling
  (`analysis_protocol/istanbul_cat_ceiling_perfold/`) and the output of the
  registered test (`analysis_protocol/m2_prereg_output.txt`) — **both on the
  pre-correction ("v17") substrate**, per the update note above, so they
  document the superseded claim, not the paper's reported numbers — plus the
  two corrected-substrate aggregates
  (`docs/results/closing_data/v18/joint_best_perfold.json`,
  `docs/studies/closing_data/v18/data/v18_results.json`) Section 6's
  `wilcoxon_v18.py` reads.

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
  NVIDIA A40/H100; `scripts/train.py` warns when the torch build differs (hard
  guard under `MTL_STRICT=1`, which Section 5.1's joint recipe sets).
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

This chain builds the substrate used by the superseded ("v17") numbers below.
The accepted paper's substrate additionally fixes a check-in graph leak (see
the note at the top of Section 5); its build tooling is not yet included in
this branch (follow-up).

> Earlier versions of this README pointed to `scripts/closing_data/p3_board.sh`
> as the driver for steps (b)–(d) and for training. That was a documentation
> error: `p3_board.sh` is unrelated internal tooling for a different research
> board and was never part of this paper's recipe.

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

> This section builds the pre-correction substrate (the "v17" numbers) — see
> the substrate note at the top of Section 5.

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

> **Substrate note.** The accepted paper trains on `check2hgi_v18`, a corrected
> representation: the canonical check-in graph connects every consecutive visit
> in *both* directions, which lets the category head see a feature of the very
> visit it is predicting. v18 keeps only the forward (`src < tgt`) edge and adds
> four elapsed-time node features. This is not an architecture change — it is
> the same model and recipe as the substrate below, trained on a leak-fixed
> graph. It **substantially lowers the reported category numbers** (e.g.
> Alabama's dedicated category macro-F1 drops from the mid-50s to the
> high-20s/low-30s) — that drop is the corrected result, not a regression.
> Region numbers are materially unaffected (region embeddings are indexed by
> historical place, which the leak cannot reach). The tooling that builds
> `check2hgi_v18` from raw data is not yet included in this branch (follow-up);
> everything below assumes it already exists at `output/check2hgi_v18/<state>/`.
> The recipe and every flag below are verified against the code that actually
> produced the delivered numbers; this branch's surrounding codebase has not
> been synced commit-for-commit against that internal snapshot, so treat "same
> command" as verified at the recipe level, not as a byte-for-byte code match.

### 5.1 Joint (multi-task) model

The recipe reported in the accepted paper: batch size 8192, static loss
weighting with **category_weight = 0.50**, **logit adjustment (τ = 0.5) on the
category head only** — this *replaces* class-weighted CE entirely and is not
combined with it — per-head one-cycle LRs, fp32, GRU category head + dual-tower
spatio-temporal region head, cross-attention MTL trunk. The recipe is uniform
across all six datasets, including Istanbul (see the historical-scripts note
below).

The core command (one state, one seed):

```bash
PYTHONPATH=src python scripts/train.py --task mtl --canon none \
    --task-set check2hgi_next_region --engine check2hgi_v18 \
    --state <state> --seed <seed> --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.50 \
    --logit-adjust-tau 0.5 \
    --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr <cat-lr> --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --checkpoint-selector geom_simple \
    --compile --tf32 --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/<state>
```

`<cat-lr>` is **0.001** for the small states (Alabama, Arizona, Istanbul) and
**0.002** for the large states (Florida, California, Texas).

(Environment knobs — fp32 via `MTL_DISABLE_AMP=1`, per-head one-cycle via
`MTL_ONECYCLE_PER_HEAD_LR=1`, chunked validation metrics via
`MTL_CHUNK_VAL_METRIC=1`, plus `MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1` — must be
exported before the command above.) Per-fold
parallel fan-out of one run is available via `scripts/run_folds_fanout.sh` +
`scripts/aggregate_folds.py`.

**Historical scripts.** `scripts/closing_data/run_catx_v17_seed0_5f.sh` /
`run_catx_v17_n20.sh` (California/Texas) and `scripts/run_istanbul_champion_stride1.sh`
(Istanbul) implement the recipe used before this correction — an older engine
(`check2hgi_dk_ovl`, or for Istanbul a separate `check2hgi` engine with a
different model and region head entirely) and class-weighted loss instead of
logit adjustment. They are kept for provenance of the superseded numbers only;
reproduce the accepted paper's numbers with the command above.

### 5.2 Dedicated single-task ceilings

The recipe reported in the accepted paper, on the same `check2hgi_v18`
substrate as Section 5.1:

```bash
# next-category ceiling: single-task GRU on the same inputs/folds (fp32 is not
# optional — every cell in this recipe runs fp32; the large-state region head
# grad-NaNs under bf16/fp16 at this class count on an A40, and the paper
# compares all states/tasks at matched precision)
MTL_DISABLE_AMP=1 PYTHONPATH=src python scripts/train.py --task next --engine check2hgi_v18 \
    --state <state> --seed <seed> --epochs 50 --folds 5 --batch-size 8192 \
    --model next_gru --embedding-dim 64 --max-lr <cat-max-lr> --logit-adjust-tau 0.5 \
    --compile --tf32
# score it:
python scripts/closing_data/score_stl_cat_ceiling.py <rundir>

# next-region ceiling: single-task region head on the same inputs/folds
# (logit adjustment stays OFF for region — it significantly *hurts* Acc@10;
# measured at two datasets, see the paper's methodology discussion)
PYTHONPATH=src python scripts/p1_region_head_ablation.py --state <state> \
    --heads next_stan_flow --input-type region --target region \
    --engine-override check2hgi_v18 \
    --region-emb-source check2hgi_design_k_resln_mae_l0_1 \
    --override-hparams freeze_alpha=True alpha_init=0.0 \
    --folds 5 --epochs 50 --seed <seed> --max-lr 0.003 --compile --tf32
```

`<cat-max-lr>` is **0.0025** for Alabama, **0.0005** for Arizona/Istanbul, and
**0.005** for Florida/California/Texas. Note the region-ceiling command omits
`--per-fold-transition-dir`: the prior is frozen off (`freeze_alpha=True
alpha_init=0.0`), so it is inert and the directory is not needed (verified
byte-equivalent at Arizona in eager mode; under `--compile` the two arms differ
by ~4.5e-5, within known compile noise).

`scripts/closing_data/stl_ceilings.sh` implements the pre-correction recipe
(`check2hgi_dk_ovl`, batch size 2048, no logit adjustment) — kept for
provenance of the superseded numbers, not the reported recipe.

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

Matched-protocol scorers (per-fold category macro-F1 + region Acc@10 with
out-of-distribution correction, identical readout for MTL and ceilings):
`scripts/closing_data/a40_score_matched.py` and
`scripts/closing_data/h100_score_matched.py <rundir> --seed <seed>`. Two epoch
selectors are computed side by side and are **not interchangeable**:
**diag-best** (per-task best epoch — the ceilings, and Table 3's non-joint
comparisons) and **joint-best** (the single served checkpoint, selected by
`geom_simple` — Table 3's reported joint-model cells). Never compare one
against the other without saying so; see
[`analysis_protocol/JOINT_BEST_SCORING.md`](analysis_protocol/JOINT_BEST_SCORING.md)
for the convention record.

## 6. Statistics and analysis

The analysis plan, the deviation log, and the executed analysis are in
[`analysis_protocol/`](analysis_protocol/) — read its `README.md` first. It records what
was registered before the runs (superiority for next-category, non-inferiority for
next-region, assigned per task, with the two-point margin pinned), the two departures
from it, and which reported claim falls outside it.

**What was pre-registered, in one line:** next-category superiority (paired Wilcoxon,
per-fold, n=20, Holm across the six datasets) and next-region non-inferiority (TOST at a
two-point margin). Next-region *superiority* was **not** registered; those four
improvements are secondary results and are reported in their own correction family.

```bash
# the registered test at its registered footing: per-fold, n=20, Holm within the
# six-dataset next-category family (+ the four next-region cells as their own family)
python scripts/closing_data/m2_prereg_perfold.py

# the reported footing: per-seed means (n=4), paired t with the Wilcoxon alongside.
# At four pairs the exact one-sided Wilcoxon p cannot fall below 0.0625, which is why
# the t is the reported statistic (analysis_protocol/DEVIATION_LOG.md, D-1 and D-2).
python scripts/closing_data/m1_stats_n20.py
python scripts/closing_data/superiority_wilcoxon.py   # per-fold superiority, seed-0 footing

# equivalence: TOST non-inferiority at the two-point margin (AL/AZ/Istanbul)
python scripts/closing_data/region_match_tost.py

# the epoch-selection convention behind every reported joint result
python scripts/closing_data/score_joint_best.py <rundir> --seed <seed> --tag <tag>
```

These read the per-fold score files produced by Section 5 (not shipped — see
"What is NOT included"). `m2_prereg_perfold.py` aborts if any recomputed aggregate stops
matching the reported cell. **These five scripts point at the pre-correction
("v17") result files** — the statistical *methodology* is unaffected by the
Section 5 substrate correction (same tests, same registered footing), but the
scripts themselves have not yet been re-pointed at the corrected result files;
that re-pointing is a follow-up (see the update note at the top of this file).

For the accepted paper's reported test on the corrected substrate, run
[`research/reproducibility/mobiwac_v18/wilcoxon_v18.py`](research/reproducibility/mobiwac_v18/wilcoxon_v18.py):
it runs the same paired one-sided Wilcoxon (n=20, Holm-corrected) *and*
the paired one-sided t-test (n=4 per-seed means) side by side against the
corrected result files, and refuses to report anything until it reproduces the
paper's own per-cell means to within 0.005 tolerance.
[`param_counts.py`](research/reproducibility/mobiwac_v18/param_counts.py) in
the same directory reproduces the parameter-count table cited in the paper.
Both are runnable as shipped — no GPU, no raw data, no other setup:

```bash
PYTHONPATH=src python research/reproducibility/mobiwac_v18/wilcoxon_v18.py
PYTHONPATH=src python research/reproducibility/mobiwac_v18/param_counts.py
```

`wilcoxon_v18.py` reads two aggregated result files, shipped in this release:
[`docs/results/closing_data/v18/joint_best_perfold.json`](docs/results/closing_data/v18/joint_best_perfold.json)
(per-fold joint-model scores) and
[`docs/studies/closing_data/v18/data/v18_results.json`](docs/studies/closing_data/v18/data/v18_results.json)
(per-run dedicated-ceiling scores). Both had a machine-local absolute path
prefix and an internal hostname stripped before publishing — a plain string
removal, verified byte-for-byte to touch nothing else (no metric or
experimental value was altered); the sidecar-writing convention that
originally produced these two files from a Section-5 regeneration run is
still a follow-up (only the two finished aggregates are shipped, not the
per-run raw logs/rundirs they were built from).

Note: the `superiority_wilcoxon.py` and `m1_stats_n20.py` docstrings describe next-region
superiority as pre-registered. That is incorrect — `analysis_protocol/STATISTICAL_PROTOCOL.md`
is authoritative and registers non-inferiority only for that task.

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

MIT (see `LICENSE`).

## Notes

- Reproduction cost: one Gowalla state cell (joint model, 5 folds, seed 0) is
  ~1 h/fold on an A40 for the largest states, minutes/fold for small states;
  representation builds are ~10–30 min/state on one GPU.
- Multi-seed paper cells use seeds {0, 1, 7, 100} (seed 42 was the development
  seed and is deliberately excluded from reported numbers).
- Questions or issues: open a GitHub issue on this repository, or contact the
  authors directly (see above).
