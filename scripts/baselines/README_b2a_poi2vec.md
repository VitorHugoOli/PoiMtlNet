# B2a — POI2Vec (Feng et al., AAAI 2017) standalone per-POI 64-d substrate

Class-(A) SC-substrate-column baseline. Emits a probe-engine substrate that
plugs into the FROZEN matched-head MTL board (cat=`next_gru`,
reg=`next_stan_flow_dualtower` / H3-alt `next_getnext_hard`) via
`train.py --engine`, exactly like `scripts/mtl_improvement/build_overlap_probe_engine.py`.

## What it is (faithfulness)
Feng et al., "POI2Vec: Geographical Latent Representation for Predicting Future
Visitors", AAAI 2017, vol. 31 — https://ojs.aaai.org/index.php/AAAI/article/view/10500.
Core mechanism: word2vec/skip-gram over POI check-in sequences, with
**geographical influence injected through a binary tree over a recursive
rectangular partition of the map** (hierarchical-softmax routing). POIs near a
partition boundary route to BOTH children with influence weights phi (paper's
multi-leaf geographical influence). Output = a per-POI latent table (the input/
center embedding) — the standalone 64-d POI column this baseline must emit.

Implemented in `b2a_poi2vec_lib/model.py` (`build_geo_binary_tree`, `GeoPOI2Vec`).
The geo path-loss is vectorized (flat packed edges + scatter-add); ~100x faster
than a per-sample loop.

## NOT the in-repo poi2vec.py
`research/embeddings/hgi/poi2vec.py` is an **fclass-level Node2Vec teacher used
inside HGI** (multiple POIs of the same fclass share a vector); it is NOT a
standalone per-POI column. This baseline is a fresh, distinct AAAI'17 POI2Vec
that learns a per-POI 64-d vector. (Documented deviation D4.)

## Deviations (audit)
- D1: paper task = "predict future visitors"; board task = next-cat (macro-F1) +
  next-region (Acc@10 OOD-restricted) under matched champion heads. We use only
  POI2Vec's representation as a substrate column (SC protocol) — isolates the
  representation on the substrate axis.
- D2: paper optionally fuses a user latent; we export only the POI table (user
  identity enters downstream via the sequence head).
- D3: partition granularity (tree depth / theta) is a tuned HP; we use a
  recursive median split to `--max-depth` with `--min-leaf` stop + `--boundary-frac`
  multi-leaf influence.
- D4: distinct from the in-repo fclass POI2Vec (above).

## Leak-safety (HARD requirement — enforced)
POI2Vec is **pretrained on the fold's TRAIN portion only**. The fold split is
reproduced bit-identically with `StratifiedGroupKFold(n_splits, shuffle=True,
random_state=seed)` over `load_next_data(state, CHECK2HGI)` — the SAME algorithm/
groups(userid)/y(next_category)/seed as `FoldCreator._create_check2hgi_mtl_folds`
and `compute_region_transition._build_per_fold`. Skip-gram pairs are built ONLY
from check-ins of train-users; `assert val_users.isdisjoint(train_users)` runs
before training. One substrate is emitted PER (state, seed, fold) so `train.py
--folds 1` against that engine dir is fully leak-clean.
The reg ranking prior (`region_transition_log_seed{S}_fold{N}.pt`) is per-fold/
per-seed; rebuild it for the trainer's n_splits before each scored run
(the trainer hard-asserts n_splits consistency — a leak guard, seen in smoke).

## Row-alignment
The matched heads consume a CHECK-IN-LEVEL `embeddings.parquet`
(`userid,placeid,category,datetime,0..63`) in the check2hgi row order. We
LEFT-JOIN the per-POI POI2Vec table onto the frozen check2hgi embeddings frame
on `placeid` (every check-in of a POI gets that POI's vector), then run the
canonical `generate_next_input_from_checkins` + `build_next_region_for`. Asserts:
`len(embeddings)==len(check2hgi embeddings)` and `len(next)==len(next_region)==len(sequences)`.

## Run
Smoke (AL, 1 fold, 2 epochs, scratch OUTPUT_DIR, frozen output/ untouched):
```bash
bash scripts/baselines/smoke_b2a_poi2vec.sh           # DEV=cuda by default
```
Build one leak-safe per-fold substrate:
```bash
OUTPUT_DIR=/tmp/bl_b2a PYTHONPATH=src .venv/bin/python \
  scripts/baselines/build_b2a_poi2vec_substrate.py <state> --seed <S> --fold <N> \
  --epochs 5 --max-pairs 0 --device cuda
```
Then run the champion recipe with `--engine <X> --folds 1` pointed at that dir.
For P3, pass `--stride 1` to the builder (overlapping windows) and re-run n=20.

## Non-conflict / enum note (for the integrator)
Dev + smoke use the zero-enum-edit escape hatch: emit into a scratch `OUTPUT_DIR`
and run `--engine check2hgi` so the frozen allow-lists accept it. For the SCORED
P3 run, append ONE `EmbeddingEngine` member at the END of the enum (e.g.
`POI2VEC_BASELINE = "poi2vec_baseline"`) and add it to: `get_next_region`
`supported`, `folds._MTL_C2HGI_ALLOWED_ENGINES`, and
`builders._CHECKIN_LEVEL_ENGINES`. Tag the PR `[ENUM-MERGE]`; merge last. No
other `src/` edits are required.
