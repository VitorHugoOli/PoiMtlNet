# CSLSL cascade cell (role-3 baseline) — A40, 2026-06-24/25

> **Status: AL + AZ + FL DONE (n=5 provisional, seed 0).** The CSLSL/CatDM **cascade** (directed cat→region,
> symmetric cross-attention severed) is a **dead tie** with our parallel champion-G on the joint objective
> at the small/mid (AL/AZ) AND the large (FL, 4703 regions) states → our parallel bidirectional cross-attention
> matches the dominant published multi-task alternative **at equal cost**. (FL same-device champ-G comparand
> in-flight as of 2026-06-25; FL cascade ties the §1 board champ-G FL to ±0.01.) CA/TX deferred (deadline;
> "only if cheap" per `HANDOFF_A40.md`).
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

**FL same-device champ-G comparand — IN-FLIGHT (fold 1/5 done, 2026-06-25).** Interim fold-1 cross-check
(A40 champ-G FL, same v16 recipe + per-head optimizers, true fp32): cat **79.45** (ep48) / reg **77.71** (ep47).
- vs cataloged **board champ-G FL** (H100, `BOARD_CELLS.md` fp32 per-fold[0] 79.38/77.68): Δ **+0.07 / +0.03**
  → reproduces the catalog to well within the cross-GPU ±0.3 pp tolerance (re-run is faithful).
- vs **A40 cascade FL** fold1 (79.43/77.57): Δ **+0.02 / +0.15** → same-card tie holds at fold level (per-fold
  wobble that averages toward the ±0.01 5-fold tie). Best-epochs late (47–48), matching board (47–49) + cascade (45–49).
The full 5-fold same-device Δ + the §1b FL row's A40 champ-G cells fill on completion (~2.5 h remaining).

**Reading:** cascade ≈ parallel champion-G — Δjoint ≤ 0.02 pp, far below fold-std (~1.3–3.3 pp). The cascade
trades a hair of category (+0.20) for a hair of region (−0.17/−0.18), netting ~0. The cascade did **not** beat
champion-G (the `b4_cascade.py` docstring's anti-overclaim sanity check passes). → Our parallel joint model is
**at least as good as the cascade at equal cost**; the cross-task lift lives in the parallel coupling, and
removing it for a directed cascade neither helps nor hurts at AL/AZ.

## Verification — the mechanism is ACTIVE and EXERCISED (2026-06-25 audit)

The near-identical metrics looked suspicious (could the cascade flags be silent no-ops, making the "cascade"
secretly == champion-G?). Three independent code audits + the run artifacts say **NO — the cascade is a
genuinely different, fully-exercised model.** The ±0.01–0.2 tie is a real result.

**Code audits (3 agents, independent):**
- `disable_cross_attn=True` → **ACTIVE**. `_coerce_cli_value` (`train.py:1224-1226`) maps `"True"`→ real bool;
  `create_model` passes it to `MTLnetCrossAttn.__init__` (stored `model.py:264`); forward guards the cross-attn
  loop with `if not self._disable_cross_attn:` (`mtlnet_crossattn_dualtower/model.py:64-70,149`). Runtime
  instrumentation: the 2 cross-attn blocks were called **0×** with the flag set vs **2×** without. Disabling
  cross-attn alone shifts init outputs (CAT max|Δ|=0.54, REG max|Δ|=0.086).
- `cond_coupling=posterior … cond_detach=True` → **ACTIVE + trainable**. The reg head builds a **zero-init**
  `cond_proj` (`next_stan_flow_dualtower/head.py:293-296`), injects the **live softmax cat-posterior** additively
  into the pooled region feature (`model.py:114-127`, `head.py:477-482`), detached (feed-forward). `cond_proj` is
  `requires_grad=True`, in the optimizer's reg group, and gets a nonzero gradient from step 1 (‖grad‖≈0.64) —
  zero-init does NOT pin it. No freeze/zero-multiply applies.
- Empirical instantiation: cascade has 2 extra trained tensors (`cond_proj.weight/bias`); same build path as
  `train.py`; logits diverge at init purely from the cross-attn removal (coupling adds 0 at init by design).

**Run-artifact proof (`cond_norm` = ‖cond_proj(cat_cond)‖, logged per epoch):**
| run | cond_norm ep1 → ep50 | coupling present? |
|---|---|---|
| cascade AL fold1 | 0.078 → **1.574** (max 1.62) | yes — learned |
| cascade FL fold1 | 0.291 → **4.613** (max 5.01) | yes — learned strongly (~16×) |
| cascade FL fold3 | 0.275 → **4.407** | yes |
| **champion-G (any fold)** | *column absent* | **no coupling** (cond_coupling=none) |

So in the actual scored runs the directed cat→region coupling **grew from ~0 to a large magnitude** (FL ~4.6) —
it is heavily used — and cross-attention was severed, yet the cascade still lands at champion-G's joint
performance. **Conclusion:** the cascade genuinely explores a different coupling topology (severed symmetric
channel + a strongly-learned directed edge) and **arrives at equivalent performance** → "cascade ≈ parallel"
is a robust finding, not a plumbing artifact. The symmetric bidirectional cross-attention and the directed
CSLSL-style cascade are **performance-equivalent on this substrate** (1.1k→4.7k regions).

## Provenance (reproduce)
- **Substrate**: `check2hgi_dk_ovl` (= v14 design_k embeddings re-windowed gated stride-1, MIN_SEQ=10).
  Built/rebuilt on the A40 2026-06-24 via `build_overlap_probe_engine.py {alabama,arizona} 1 10`
  (AL was rebuilt off its prior MIN_SEQ=5; AZ built fresh). Rows: AL 96,326 / AZ 200,895, pad 0.0%.
- **Per-fold seeded log_T** (leak-free, engine-aware 2026-06-23 fix): `compute_region_transition.py --state {s}
  --engine check2hgi_dk_ovl --per-fold --seed 0 --n-splits 5` → in the dk_ovl dir; verified fresher than
  `next_region.parquet`. n_regions AL 1109 / AZ 1547.
- **Precision**: `MTL_DISABLE_AMP=1` (true fp32, both train+eval autocast off; matches the AL/AZ board fp32
  decision, `BOARD_CELLS.md`). `MTL_NAN_GUARD=1` → **0 non-finite skips** all 4 runs.
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
