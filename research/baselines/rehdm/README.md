# Faithful ReHDM baseline

Self-contained reproduction of **ReHDM** (Li et al., *Beyond Individual and
Point: Next POI Recommendation via Region-aware Dynamic Hypergraph with
Dual-level Modeling*, IJCAI 2025) adapted to the next-region task on Gowalla
state-level check-ins.

## What "faithful" means here

This baseline reproduces the published architecture and preprocessing
pipeline. It does **not** consume any in-house Check2HGI / HGI substrate.
Inputs are derived from raw Gowalla check-ins and US Census TIGER tracts,
matching paper §4.1 / §5.1.

The **only** deviation from the published model is the predictor's output
domain: paper predicts next-POI (~10 K candidates); we project to
`n_regions` (≈1.1 K on AL, ≈1.5 K on AZ, ≈4.7 K on FL) so the result is
directly comparable to the check2HGI study's region-task table. Inputs and
the hypergraph machinery are unchanged.

A future variant will keep the paper's POI head and feed it the in-house
Check2HGI embeddings (single-task region head for embedding-based runs).

## Components

- `etl.py` — raw check-ins → 6 ID encodings + 24h trajectories +
  TIGER-tract region targets.
- `model.py` — `ReHDM` module: 6-ID embedding, POI-level Transformer,
  vertex→hyperedge then L−1 hyperedge↔hyperedge HG-Transformer layers with
  intra/inter edge types and gated residual; classifier projects to
  `n_regions`.
- `train.py` — repeated-seed run trainer (default 5 seeds × 50 epochs),
  AdamW + OneCycleLR, sub-hypergraph built per batch with intra-user and
  shared-POI inter-user collaborators that satisfy `end(s_m) < start(target)`.

## Outputs

```
output/baselines/rehdm/<state>/inputs.parquet      # ETL output
output/baselines/rehdm/<state>/vocab.json          # cardinalities + hypers
docs/studies/check2hgi/results/baselines/<tag>_run{i}.json
docs/studies/check2hgi/results/baselines/<tag>_summary.json
```

## CLI

```bash
PY=/Volumes/Vitor's\ SSD/ingred/.venv/bin/python
DATA_ROOT=/Volumes/Vitor's\ SSD/ingred/data
OUTPUT=/Volumes/Vitor's\ SSD/ingred/output

PYTHONPATH=. DATA_ROOT="$DATA_ROOT" OUTPUT_DIR="$OUTPUT" \
  "$PY" -m research.baselines.rehdm.etl --state alabama

PYTHONPATH=. OUTPUT_DIR="$OUTPUT" \
  "$PY" -m research.baselines.rehdm.train \
    --state alabama --folds 5 --epochs 50 \
    --batch-size 64 --max-len 20 --max-intra 4 --max-inter 4 \
    --tag REHDM_al_5seeds_50ep
```

## Faithfulness notes

| ReHDM paper | This implementation |
|---|---|
| 6 ID features `<u, p, c, h_h, t_d, r>` | identical (user/poi/category/hour/day-of-week/quadkey-L10) |
| Quadkey level 10 | configurable, default 10 |
| 24h trajectories, ≥2 check-ins | identical |
| Chronological 80/10/10, val/test ⊆ train users+POIs | identical |
| POI-level: 1× Transformer block | identical (MSA + FFN, residual + LN + dropout) |
| Trajectory-level: vertex→hyperedge then L−1 hyperedge↔hyperedge layers | identical, default L=2 |
| Edge types `r ∈ {intra, inter}` | identical (learned 2-row embedding) |
| Time-precedence filter on collaborators | enforced (`end(s_m) < start(target)`) |
| Gated residual `β h_l W + (1−β) g_l` | identical (β=0.5 default) |
| L2-normalised hidden states | identical |
| Predictor: linear → softmax over POIs | **adapted** to softmax over regions |
| Cross-entropy training | identical |

## Paper ambiguities (documented faithful guesses)

The paper does not specify the following; we picked defensible defaults:

| Item | Paper | Default here |
|---|---|---|
| Inter-user "≈" similarity | undefined | shared-POI ≥ 1, random-sample to `max_inter` (MSTHgL / DCHL convention) |
| L (number of e2e layers) | unstated | 2 (i.e. one e2e layer) |
| β (gated residual) | unstated | 0.5 |
| d_id (per-feature embedding) | unstated | 32 (⇒ d=192) |
| Optimizer / lr / batch / epochs | unstated | AdamW 1e-4 + OneCycleLR 1e-3, batch 64, 50 ep |
| Eq. 14 `Norm` | LN or L2 | L2 (consistent with Eq. 13) |
| Sub-hypergraph collaborator cap | unstated | `max_intra=4, max_inter=4` |
| Quadkey vocab encoding | "modulo grid count" | string-vocab indexing (equivalent for hashing) |

## Verified-fixed bugs (caught in audit pass)

1. Spatial join did not preserve sort order → re-sort after merge.
2. Target check-in was fed into the encoder (region label leakage) →
   encoder now only sees the first `t_len-1` check-ins; target is the
   region of position `t_len`.
3. Eval-time inter-user shuffle was non-deterministic across calls →
   stable RNG seeded with 0 during evaluation.
