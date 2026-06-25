# CSLSL cascade cell (role-3 baseline) — A40, 2026-06-24/25

> **Status: AL + AZ + FL DONE (n=5 provisional, seed 0).** The CSLSL/CatDM **cascade** (directed cat→region,
> symmetric cross-attention severed) is a **dead tie** with our parallel champion-G on the joint objective
> at the small/mid (AL/AZ) AND the large (FL, 4703 regions) states → our parallel bidirectional cross-attention
> matches the dominant published multi-task alternative **at equal cost**. (FL same-device champ-G comparand
> in-flight as of 2026-06-25; FL cascade ties the §1 board champ-G FL to ±0.01.) CA/TX deferred (deadline;
> "only if cheap" per `BASELINE_A40.md`).
> Headline + board: [`RESULTS_BOARD.md §1b`](RESULTS_BOARD.md) and
> [`../../results/closing_data/MACS_BOARD_RESULTS.md`](../../results/closing_data/MACS_BOARD_RESULTS.md).

## What this baseline isolates
`scripts/baselines/b4_cascade.py` reuses the **exact champion heads on the frozen Check2HGI substrate**, with
the ONLY varying factor a directed **cat→region** cascade edge (vs our parallel bidirectional cross-attention):
- coupling pin: `cond_coupling=posterior cond_signal=softmax cond_inject=add cond_detach=True` (feed-forward;
  region stage reads the predicted-category posterior, no reverse cat←reg gradient),
- `disable_cross_attn=True` (sever the symmetric channel → a true cascade, not a coupling ablation).
Everything else is byte-identical champion-G v16: `mtlnet_crossattn_dualtower` + `next_gru`(cat) +
`next_stan_flow_dualtower`(reg, aux fusion, α·log_T off), static_weight cw=0.75, onecycle max-lr 3e-3,
geom_simple selector, checkin/region modality, log_T-KD off. Comparand = **champion-G** (NOT the STL ceiling).

## Result (seed 0 × 5f, same-device A40, true fp32)
| State | cascade cat | champ-G cat | Δcat | cascade reg | champ-G reg | Δreg | cascade joint | champ-G joint | Δjoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AL | 63.45 ±2.00 | 63.25 ±2.02 | +0.20 | 69.48 ±3.03 | 69.65 ±3.32 | −0.17 | 66.39 | 66.37 | **+0.02** |
| AZ | 63.63 ±1.34 | 63.44 ±1.33 | +0.20 | 59.18 ±1.83 | 59.36 ±1.79 | −0.18 | 61.37 | 61.36 | **+0.00** |
| FL | 79.83 ±0.49 | ⏳ A40 in-flight | — | 77.27 ±0.95 | ⏳ A40 in-flight | — | 78.54 | — | ⏳ |

cat = macro-F1; reg = FULL top10_acc = `top10_acc_indist·(1−ood_fraction)` at the diagnostic-best epoch;
joint = √(cat·reg); fold-mean ±pstd, matched scorer `scripts/closing_data/a40_score_matched.py`.

**FL (large state, 4703 regions, 1.27M rows, dk_ovl/MIN_SEQ=10, true fp32, 0 skips):** cascade cat **79.83**
/ reg **77.27** — vs the §1 board champ-G FL (H100, 79.82/77.28) → **Δcat +0.01 / Δreg −0.01**, essentially
identical (cross-device). The A40 same-device champ-G FL is **in-flight** (~1.8h); its row + Δ fill on
completion. FL canonical (`dk_ovl`, 5f, with comparand) **supersedes the M4 set-a partial**
(`baseline_compare/florida_cslsl_cascade.json`: stride-9/min_seq=5, 4-fold MPS-OOM, no comparand →
its own recommendation was "run FL CSLSL on CUDA/A40").

**Reading:** cascade ≈ parallel champion-G — Δjoint ≤ 0.02 pp, far below fold-std (~1.3–3.3 pp). The cascade
trades a hair of category (+0.20) for a hair of region (−0.17/−0.18), netting ~0. The cascade did **not** beat
champion-G (the `b4_cascade.py` docstring's anti-overclaim sanity check passes). → Our parallel joint model is
**at least as good as the cascade at equal cost**; the cross-task lift lives in the parallel coupling, and
removing it for a directed cascade neither helps nor hurts at AL/AZ.

## Provenance (reproduce)
- **Substrate**: `check2hgi_dk_ovl` (= v14 design_k embeddings re-windowed gated stride-1, MIN_SEQ=10).
  Built/rebuilt on the A40 2026-06-24 via `build_overlap_probe_engine.py {alabama,arizona} 1 10`
  (AL was rebuilt off its prior MIN_SEQ=5; AZ built fresh). Rows: AL 96,326 / AZ 200,895, pad 0.0%.
- **Per-fold seeded log_T** (leak-free, engine-aware 2026-06-23 fix): `compute_region_transition.py --state {s}
  --engine check2hgi_dk_ovl --per-fold --seed 0 --n-splits 5` → in the dk_ovl dir; verified fresher than
  `next_region.parquet`. n_regions AL 1109 / AZ 1547.
- **Precision**: `MTL_DISABLE_AMP=1` (true fp32, both train+eval autocast off; matches the AL/AZ board fp32
  decision, `AL_PRECISION_GATE.md`). `MTL_NAN_GUARD=1` → **0 non-finite skips** all 4 runs.
- **NOT set**: `MTL_STRICT` (the cascade runs auto-canon v16 → the v16-pins-v14-substrate guard would HARD-FAIL
  under STRICT on the dk_ovl engine; without STRICT it WARNs and is numerically inert). `--compile`/`--tf32`
  off on BOTH sides (true fp32, cleanest internal Δ).
- **Commands** (cascade = direct train.py = the `b4_cascade.py --engine check2hgi_dk_ovl` command, run directly
  for PID-clean rundir capture; preflight pre-validated via `--smoke` + fresh log_T):
  ```
  # cascade:
  train.py --task mtl --task-set check2hgi_next_region --engine check2hgi_dk_ovl --state {s} --seed 0 \
    --folds 5 --epochs 50 --batch-size 2048 --model mtlnet_crossattn_dualtower \
    --cat-head next_gru --reg-head next_stan_flow_dualtower --task-a-input-type checkin \
    --task-b-input-type region --per-fold-transition-dir output/check2hgi_dk_ovl/{s} \
    --checkpoint-selector geom_simple --reg-head-param cond_coupling=posterior \
    --reg-head-param cond_signal=softmax --reg-head-param cond_inject=add \
    --reg-head-param cond_detach=True --model-param disable_cross_attn=True --no-checkpoints
  # champion-G comparand = same command MINUS the 4 cond_* reg-head-params and disable_cross_attn.
  ```
- **JSONs**: `docs/results/closing_data/a40/{al,az}_cascade_s0.json` + `{al,az}_champG_a40_s0.json`.

## Cross-device corroboration (A40 vs board H100 champion-G, §1)
AL: cat 63.25 vs board 63.56 (−0.31) / reg 69.65 vs 69.81 (−0.16). AZ: cat 63.44 vs 63.39 (+0.05) /
reg 59.36 vs 59.34 (+0.02). All ≤ fold-std → the §1 board champion-G reproduces on Ampere; the same-device Δ
above is the primary (device-confound-free) cascade-vs-parallel result.

## Deviations from source papers (audit log; cf. `b4_cascade.py` D1–D5)
Pattern port, not a faithful CSLSL/CatDM re-impl (D1 heads; D2 region not POI; D3 Gowalla 7-cat + TIGER tract;
D4 over frozen Check2HGI; D5 additive zero-init posterior injection). DEFERRED: faithful CSLSL/CatDM decoders.
