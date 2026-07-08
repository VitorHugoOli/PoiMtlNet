# Geographic near-miss distance -- cross-state results

> Consolidated summary of `near_miss_distance.py` across states. For each
> validation visit whose top-1 predicted region is wrong, the haversine distance
> (km) between the predicted and true region centroid. In-distribution and
> out-of-distribution (OOD -- true region absent from the fold's training
> vocabulary) misses are reported SEPARATELY, never pooled. Bare percentiles
> only; no service-radius / coverage-threshold overlay (venue-bridge guardrail,
> MOBILITY_PLAN.md 0/3.1). This is a property of the model's own predictions on
> the validation folds, not a measured service result.

## Provenance

- **Recipe:** champion v17 (bs8192 + per-head cat-lr, `MTL_ONECYCLE_PER_HEAD_LR=1`),
  gated stride-1 overlap engine `check2hgi_dk_ovl`, fp32, seed 0 x 5 folds,
  `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` (reg) + `next_gru` (cat),
  static_weight cw=0.75, geom_simple selector. Run on the A40, 2026-07-07.
- **Per-sample predictions:** the `MTL_DUMP_VAL_PREDS=1` dump
  (`fold{N}_reg_val_preds.parquet`), written at each reg diagnostic-best
  (`top10_acc_indist`) epoch -- the same event the reported region cell is
  selected on, so the near-miss is measured at the same checkpoint as the paper's
  Acc@10 number.
- **Region geometry:** `output/check2hgi/<state>/temp/{checkin_graph.pt, boroughs_area.csv}`
  (region_idx -> GEOID -> WKT-polygon centroid; GEOID zero-pad handled; 0 dropped
  centroids at every state).

## Caveat — Istanbul is a single-seed point estimate

Istanbul's numbers here come from ONE **seed-0 x 5-fold** v17 `check2hgi_dk_ovl`
run. The paper reports Istanbul at **n=20 (4 seeds {0,1,7,100})** for its
headline region cell, and that cell is still on the older **stride-1 GCN**
substrate (the `dk_ovl` Istanbul rebuild is the pending A40-2/H3 task) -- so the
Istanbul near-miss differs from the paper standard on BOTH seed count and
substrate. This is acceptable for a motivation-only metric, but it is a
single-seed point estimate, not the multi-seed standard. AL/AZ/FL do NOT carry
this caveat: their paper cells are themselves n=5 seed-0 provisional (P1
pending), so seed-0 near-miss matches their current headline standard; only
Istanbul's paper standard is already n=20.

**Top-up:** when the n=20 `dk_ovl` Istanbul dumps exist (A40-2/H3 rebuild, or
the P1/H100 lane, re-running v17 Istanbul at seeds 1/7/100 with
`MTL_DUMP_VAL_PREDS=1`), regenerate this metric from them and pool. Istanbul is
small (520 regions), so the top-up is A40-feasible, not strictly H100-only.

## Summary (pooled over the 5 folds, per state)

| State | regions | reg Acc@10 (this run) | in-dist miss P50 (km) | P90 (km) | mean (km) | OOD miss P50 / P90 (km) |
|---|---:|---:|---:|---:|---:|---:|
| Alabama  | 1,109 | 69.8 | **8.13** | 38.47 | 20.38 | 22.48 / 156.14 |
| Arizona  | 1,547 | 59.7 | **8.05** | 30.71 | 17.35 | 17.74 / 128.87 |
| Istanbul |   520 | 75.4 | **3.16** | 14.91 |  5.75 | 11.58 /  28.59 |
| Florida  | 4,703 | 77.4 | **7.04** | 37.58 | 20.56 | 17.01 /  53.37 |

_reg Acc@10 = mean over 5 folds of `top10_acc_indist` at the reg diagnostic-best
epoch (matches the board cell within run-to-run variation)._

## Reading

When the model's single most-likely next region is wrong, the region it did
predict sits a median of about **3 km (Istanbul) to 8 km (Alabama / Arizona)**
from the true one, with 90% of in-distribution misses within ~15 km (Istanbul) to
~30-38 km (Arizona / Alabama). The tighter Istanbul figure is consistent with its
geography: 520 dense mahalle units in one metropolitan area versus 1,109-1,547
census tracts spread across a whole US state. OOD misses (visits whose true region
was never in the training vocabulary, so no in-vocabulary correct answer exists)
are larger and are reported separately, never mixed into the in-distribution
figure.

## Health (no collapsed fold cited)

Per-fold reg diagnostic-best epochs land late at every state (no early-epoch
precision collapse): Alabama 27/31/37/43/35, Arizona 36/38/33/25/30, Istanbul
42/33/36/35/39, Florida 45/34/44/44/49. The per-fold `top10_acc_indist` tracks the
board region cell at each state (e.g. Florida mean 77.4 vs board 77.28).

## Source rundirs (on the A40; `results/` is gitignored)

| State | training rundir |
|---|---|
| Alabama  | `results/check2hgi_dk_ovl/alabama/mtlnet_lr1.0e-04_bs8192_ep50_20260707_163142_347078` |
| Arizona  | `results/check2hgi_dk_ovl/arizona/mtlnet_lr1.0e-04_bs8192_ep50_20260707_164331_349463` |
| Istanbul | `results/check2hgi_dk_ovl/istanbul/mtlnet_lr1.0e-04_bs8192_ep50_20260707_164911_350709` |
| Florida  | `results/check2hgi_dk_ovl/florida/mtlnet_lr1.0e-04_bs8192_ep50_20260707_182009_363321` |

Each holds `metrics/fold{1..5}_reg_val_preds.parquet` (the dump) + the per-fold
`fold{N}_next_region_val.csv` (from which the best-epochs above are read).

## Reproduce

Per state (`STATE` in alabama / arizona / istanbul / florida), on a CUDA box with
`torch==2.11.0+cu128`. Florida needs the host RAM free (~22 GB dataset), so run it
alone. This is the champion v17 board recipe with the opt-in dump added; the only
deviation from `scripts/closing_data/run_catx_v17_n20.sh` is dropping
`--no-checkpoints` (the dump needs a results dir) and `--profile`.

```bash
export PYTHONPATH=src
export MTL_DISABLE_AMP=1 MTL_ONECYCLE_PER_HEAD_LR=1 MTL_CHUNK_VAL_METRIC=1 \
       MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1 MTL_RAM_HEADROOM_GB=24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MTL_DUMP_VAL_PREDS=1                      # <- writes fold{N}_reg_val_preds.parquet

python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region \
    --engine check2hgi_dk_ovl --state "$STATE" --seed 0 --epochs 50 --folds 5 \
    --batch-size 8192 --mtl-loss static_weight --category-weight 0.75 \
    --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --checkpoint-selector geom_simple \
    --onecycle-per-head-lr --compile --tf32 \
    --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/"$STATE"

# then, against the rundir train.py printed (no GPU / training dep):
python "articles/[mobiwac]/analysis/near_miss_distance.py" --state "$STATE" --rundir <that rundir>
```

The per-fold log_T is inert for this config (`freeze_alpha=True`, `alpha_init=0.0`,
KD off), so `MTL_SKIP_INERT_LOGT=1` (default) skips it -- no `region_transition_log_*.pt`
files are needed (byte-identical).

## Floor: random-pair inter-centroid distance (the reference point, added 2026-07-08)

The miss distances above need a reference scale (the paper's own rule: never a
number without its reference point). `near_miss_floor.py` computes it: the
haversine distance between the centroids of two randomly drawn, distinct
candidate regions, restricted to the model's exact region vocabulary
(`region_to_idx`; all vocabulary GEOIDs matched a polygon at every state).
200,000 pairs, seed 0, CPU-seconds. This is a bare geometric scale anchor, not
a simulated random predictor (the true-region distribution is not uniform).

| State | vocab regions | random-pair P50 (km) | P90 (km) | mean (km) | model in-dist miss P50 | model is |
|---|---:|---:|---:|---:|---:|---:|
| Alabama  | 1,109 | 170.67 | 377.39 | 198.46 | 8.13 | ~21x closer |
| Arizona  | 1,547 | 120.32 | 286.93 | 134.74 | 8.05 | ~15x closer |
| Florida  | 4,703 | 241.22 | 507.89 | 262.28 | 7.04 | ~34x closer |
| Istanbul |   520 |  20.45 |  59.45 |  26.38 | 3.16 | ~6x closer |

Reading: a wrong top-1 prediction lands an order of magnitude closer to the
true region than a random candidate region would, on every measured dataset.

## Status

All four states complete (2026-07-07, A40). Florida (the stride-1 overlap large
state, ~1.3M rows, 4,703 regions) ran alone in fp32 and held convergence with no
collapsed fold, its ~7 km median in-distribution miss matching the smaller states'
~3-8 km range. Floor reference added 2026-07-08 (`near_miss_floor.py`, above).
