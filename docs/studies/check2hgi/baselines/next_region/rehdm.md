# ReHDM

## Source
- **Paper:** Li, Gu, Yao, Zhou, Zhu, Zhao, Du. *Beyond Individual and Point: Next POI Recommendation via Region-aware Dynamic Hypergraph with Dual-level Modeling.* IJCAI 2025. [pdf](https://www.ijcai.org/proceedings/2025/0343.pdf).
- **Reference impl:** none publicly available (verified via web search 2026-04-25; closest cousins are STHGCN [`alipay/Spatio-Temporal-Hypergraph-Model`](https://github.com/alipay/Spatio-Temporal-Hypergraph-Model) and DCHL [`icmpnorequest/SIGIR2024_DCHL`](https://github.com/icmpnorequest/SIGIR2024_DCHL), neither architecturally identical).
- **Architecture (paper):** 6-ID embedding layer (`<u, p, c, hour, day, region>`, `r` from quadkey-L10) concatenated to `d=6·d_id`; one POI-Level Transformer block (MSA + FFN + LN + dropout) over the 9-checkin sequence; trajectory-level Hypergraph Transformer with vertex→hyperedge initial step (Eq. 12–13) followed by `L−1` hyperedge↔hyperedge layers (Eq. 14) carrying intra/inter edge-type messages; gated residual `β·h^(l) + (1−β)·g^(l)`; L2-normalised hidden states; linear softmax classifier over `n_pois`. Cross-entropy loss. Hypergraph H₁ is node-to-hyperedge incidence (each trajectory is a hyperedge); H₂ is hyperedge-to-hyperedge with intra-user (`r=0`) and inter-user (`r=1`) collaborator edges, time-precedence-filtered (`end(s_m) < start(target)`).

## Why this is a baseline (not our model)

ReHDM is the most recent **region-aware** published next-POI architecture (IJCAI 2025), explicitly using a coarse-grained region taxonomy (quadkey level 10) as one of its six input IDs. We use it to:

1. **External published-SOTA reference for next-region.** Faithful from-scratch reproduction (paper protocol) with the predictor's output domain swapped from `n_pois` to `n_regions`. Quantifies how strong an off-the-shelf 2025 hypergraph baseline is at our scale.
2. **STL substrate ablation.** The same architecture stripped of the 6-ID embedding stack and fed our pre-trained Check2HGI / HGI substrate as input. Isolates how much of ReHDM's strength comes from the multi-ID embedding design vs the dual-level hypergraph machinery.

## What's faithful, what's adapted

### Faithful to paper (`faithful` variant)
- **6 ID features** `<u, p, c, h_h, t_d, r>` — user, POI, category, hour-of-day (24), day-of-week (7), quadkey-L10 region — concatenated to `d=6·d_id` (paper Eq. 3). `d_id=32` ⇒ `d_model=192`.
- **Quadkey level 10 regional encoding** via Microsoft Tile Map System, base-4 string. Matches paper §4.1.
- **24-hour session boundary** + ≥2 check-ins per session + iterated ≥10-checkin filter on users/POIs. Matches paper §5.1.
- **Chronological 80/10/10 split** by trajectory start-time, with val/test ⊆ train users+POIs. Matches paper §5.1.
- **POI-Level Module:** 1× MSA + FFN block, residual + LN + dropout. Matches paper Eq. 8.
- **Trajectory-Level Module:** vertex→hyperedge step (Eq. 12–13) followed by `L−1` hyperedge↔hyperedge layers (Eq. 14) with intra/inter edge-type embedding, time-precedence-filtered collaborator pool, gated residual β·h^(l) + (1−β)·g^(l), L2/Norm + ReLU + MLP. Default `L=2` (paper does not specify; matches typical 1-e2e-layer setting).
- **Cross-entropy loss** (paper Eq. 16).

### Adapted because our task / data differ
- **Output is `n_regions` not `n_pois`.** Our table reports next-region; ReHDM's published task is next-POI. The classifier projects to `n_regions` (TIGER-tract taxonomy, same as the rest of the study) instead of `n_pois`. Necessary task adaptation.
- **Region taxonomy mismatch with paper.** Paper uses quadkey-L10 as the input region ID *and* implicitly as a POI clustering signal. We keep quadkey-L10 as an input feature (faithful) but the **target** is the TIGER tract GEOID (the rest of the study's taxonomy), so the input quadkey ID is decoupled from the target label space. Documented limitation.
- **`Norm` ambiguity in Eq. 14** is implemented as **LayerNorm**, not L2-normalize. The published paper writes `Norm(...)` without specifying; the L2 alternative produced zero-norm gradient instability on MPS in our environment.
- **Inter-user "≈" collaborator similarity** is undefined in paper §4.2. We use the standard CF proxy: shared-POI ≥ 1 between target and collaborator, then random-sample to `max_inter`. Matches MSTHgL / DCHL conventions cited by the paper.
- **Hyper-parameters** the paper does not state, set to defensible defaults: `L=2`, `β=0.5`, `d_id=32`, `max_intra=3`, `max_inter=3`, `max_inter_pool=32`. Documented in `research/baselines/rehdm/README.md`.
- **Training recipe.** AdamW (`lr=5e-5`, `weight_decay=0.01`) + OneCycleLR (`max_lr=5e-4`) over 50 epochs at batch 32, gradient clip 0.5. Lower than typical because higher LR produced NaN with the larger collaborator pool on MPS.
- **STL variants (`stl_check2hgi`, `stl_hgi`)** drop the entire 6-ID embedding stack and replace it with `Linear(emb_dim=64, d_model=192)` over the per-check-in (Check2HGI) or per-POI (HGI) substrate. Everything else (POI-Level Transformer, dual-level hypergraph, classifier) is unchanged. These variants are **not** literature-faithful by themselves; they're the substrate-as-input version of ReHDM.

## Variants we run

| Variant | Protocol | Inputs | Hypergraph | Output | Where |
|---|---|---|---|---|---|
| `faithful` | paper (chronological 80/10/10, 24h sessions, 5 seeds) | 6 IDs (raw) | intra + inter, time-precedence | `n_regions` | `research/baselines/rehdm/etl.py` + `train.py` |
| `stl_check2hgi` | study (5-fold StratifiedGroupKFold) | 9-step Check2HGI emb seq from `next.parquet` | intra (same userid) + inter (shared POI), no time-precedence | `n_regions` | `research/baselines/rehdm/train_stl_study.py` |
| `stl_hgi` | study (5-fold StratifiedGroupKFold) | 9-step HGI emb seq from `next.parquet` (region target from check2hgi) | intra (same userid) + inter (shared POI), no time-precedence | `n_regions` | `research/baselines/rehdm/train_stl_study.py` |

Paper-protocol replicates of `stl_check2hgi` / `stl_hgi` exist (`train_stl.py`) but are kept only in the per-state JSON archive, not in the comparison table — they're not protocol-comparable to the rest of the study.

## ⚠️ Protocol & architecture choices per variant

We run **two protocol regimes** to keep both the paper-faithful claim and the study-comparable claim honest:

### `faithful` — paper protocol
Reproducing the paper's claim is the whole point of `faithful`. It uses:

- **Sessions:** 24h trajectories with ≥2 check-ins (paper §5.1), not the study's 9-step windows.
- **Splits:** chronological 80/10/10 by trajectory start time, not 5-fold StratifiedGroupKFold.
- **Filtering:** iterated ≥10-checkin user/POI filter (paper §5.1).
- **Replication:** 5 *seeds* (42–46) with the same chronological split, not 5 *folds* — σ is inter-seed.
- **Architecture:** full ReHDM including the dual-level hypergraph with intra/inter edge types and time-precedence collaborator filter.

These numbers should NOT be cell-for-cell-compared with STAN / GRU σ columns in `comparison.md`. Use the qualitative pattern (ReHDM > STAN > GRU > Markov) and the absolute means, not the σ widths.

### `stl_check2hgi` / `stl_hgi` — study protocol, full hypergraph
These variants exist to be **directly comparable** to other STL rows (`next_gru`, `next_stan`, `next_transformer_relpos`, etc.) and to the future MTL model. They preserve the full ReHDM architecture; only the input substrate and the protocol change. Specifically:

- **Inputs:** 9-step pre-windowed embeddings from check2hgi/hgi `next.parquet` (matches `p1_region_head_ablation.py`). Region targets always come from check2hgi's `next_region.parquet` (TIGER-tract taxonomy is engine-invariant).
- **Splits:** 5-fold StratifiedGroupKFold (`y_cat` stratification, `userid` groups, seed=42) — identical to every other study STL.
- **Hypergraph (full, with shared-POI inter-user mining):**
    - **Intra-user (`r=0`)**: same userid in train fold. Empty for cold-user val rows by construction; the model still gets inter-user signal at val.
    - **Inter-user (`r=1`)**: train-fold rows that share at least one POI with the target (different userid). Per-position POIs come from `output/<engine>/<state>/temp/sequences_next.parquet` (`poi_0..poi_8`). This works for both warm and cold users because POI overlap doesn't require user overlap.
    - **Time-precedence dropped.** StratifiedGroupKFold is non-temporal by design — no `end(s_m) < start(target)` filter at fold-level. This is the only architectural deviation from the paper.
- **Replication:** 5 folds.

### What the two regimes tell us

`faithful` quantifies **the architecture's full strength** under its own intended setting (warm users, chronological split, 6-ID raw inputs, time-precedence). `stl_*` quantifies **what the same architecture delivers** when fed our pre-trained substrate under the study's cold-user holdout. The gap between the two (~23 pp Acc@10 at AL with HGI) is the cost of substrate-as-input plus cold-user holdout, not an architectural ablation.

## Reproduction commands

```bash
PY=/Users/vitor/Desktop/mestrado/ingred/.venv/bin/python
ENV='PYTHONPATH=. DATA_ROOT=/Users/vitor/Desktop/mestrado/ingred/data
     OUTPUT_DIR=/Users/vitor/Desktop/mestrado/ingred/output
     PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1'

# Faithful — ETL once, then train
$ENV "$PY" -m research.baselines.rehdm.etl --state alabama
$ENV "$PY" -u -m research.baselines.rehdm.train \
    --state alabama --folds 5 --epochs 50 \
    --batch-size 32 --max-len 20 --max-intra 3 --max-inter 3 \
    --tag REHDM_al_v2_5seeds_50ep
# (repeat for arizona / florida; FL ETA ~30h with current settings)

# stl_check2hgi (study protocol, full hypergraph with shared-POI inter-user)
$ENV "$PY" -u -m research.baselines.rehdm.train_stl_study \
    --state alabama --engine check2hgi --epochs 50 \
    --batch-size 256 --lr 1e-4 --max-lr 3e-3 \
    --max-intra 3 --max-inter 3 \
    --tag REHDM_STL_STUDY_v3_al_check2hgi_5f50ep

# stl_hgi (study protocol, full hypergraph)
$ENV "$PY" -u -m research.baselines.rehdm.train_stl_study \
    --state alabama --engine hgi --epochs 50 \
    --batch-size 256 --lr 1e-4 --max-lr 3e-3 \
    --max-intra 3 --max-inter 3 \
    --tag REHDM_STL_STUDY_v3_al_hgi_5f50ep
```

## Source JSONs

| Variant | State | JSON |
|---|---|---|
| `faithful` | AL | `docs/studies/check2hgi/results/baselines/REHDM_al_v2_5seeds_50ep_run{0..4}.json` |
| `faithful` | AZ | `docs/studies/check2hgi/results/baselines/REHDM_az_v2_5seeds_50ep_run{0..4}.json` |
| `faithful` | FL | pending (~30h ETA) |
| `stl_check2hgi` | AL | `docs/studies/check2hgi/results/baselines/REHDM_STL_STUDY_v3_al_check2hgi_5f50ep_fold{0..4}.json` |
| `stl_check2hgi` | AZ | pending |
| `stl_check2hgi` | FL | pending |
| `stl_hgi` | AL | `docs/studies/check2hgi/results/baselines/REHDM_STL_STUDY_v3_al_hgi_5f50ep_fold{0..4}.json` |
| `stl_hgi` | AZ | `docs/studies/check2hgi/results/baselines/REHDM_STL_STUDY_v3_az_hgi_5f50ep_fold{0..4}.json` |
| `stl_hgi` | FL | pending |
| `stl_*` paper-protocol replicates | AL/AZ | `docs/studies/check2hgi/results/baselines/REHDM_STL_*_5x50_run{0..4}.json` (archived; not used in comparison) |

Aggregated summaries at `docs/studies/check2hgi/results/baselines/REHDM_*_summary.json`.

## Implementation audit

A faithfulness audit (subagent re-read the paper and reviewed `etl.py`/`model.py`/`train.py` line-by-line) found and fixed three bugs before final results were captured:

1. `df.merge(...)` after spatial join broke time-sort, corrupting 24h session boundaries → re-sort after merge.
2. **Target leakage**: encoder saw the very check-in whose region was the prediction target → encoder now only sees the first `t_len-1` check-ins; target = region of position `t_len`.
3. Eval inter-user shuffle was non-deterministic across calls → stable `random.Random(0)` for evaluation.

A subsequent vectorisation pass (per-target attention without block-diagonal flattening, precomputed collaborator pools) gave **6× speedup** with quality preserved within 1σ (AL Acc@10 v1: 0.669 ± 0.011 → v2: 0.661 ± 0.010 — within seed noise).

Full audit log + repro discussion: `research/baselines/rehdm/README.md`.

## Cross-references

- ETL/Model/Train: `research/baselines/rehdm/{etl,model,train,etl_stl,model_stl,train_stl}.py`.
- Paper PDF: `https://www.ijcai.org/proceedings/2025/0343.pdf`.
- Aggregated metrics by state: `results/{alabama,arizona,florida}.json`.
- Cross-baseline comparison table: `comparison.md`.
