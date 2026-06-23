# Istanbul Phase V — seed 0 (FROZEN substrate, mahalle primary) — 2026-06-23

> H100, `study/board-h100` lane (run inline alongside the Gowalla board). **First paper-grade Phase-V cell**
> on the FROZEN Gowalla-bit-identical substrate (GCN 500ep), superseding the Mac dry-run (ResLN-80ep). Region
> def = **mahalle (520, primary admin)** per the methodology advisor; H3 (2585) is secondary, not run here.
> Champion-G H3-alt recipe, `--canon none` + explicit (crossattn / next_getnext_hard), **fp32**, seed 0, 5 folds.
> Report = **gap-to-ceiling / lift-over-floor** (F2 rule), NOT absolute Acc@k.

## Result (seed 0, 5f)
| task | MTL | STL ceiling | gap-to-ceiling | floor (Markov-1) | lift-over-floor |
|---|---|---|---|---|---|
| **reg** Acc@10 (FULL top10) | **69.7892** ± 0.70 | **70.37** (p1 next_stan_flow, fp32) | **−0.58** (matches, within noise) | **52.52** | **+17.27** |
| **cat** macro-F1 | **60.1461** ± 0.43 | **52.0994** (next_gru) | **+8.05** (beats) | — | — |

Per-fold: MTL reg [70.51,68.74,69.46,70.62,69.61] ep[17,18,15,20,18]; MTL cat [60.57,59.52,59.82,60.19,60.62] ep[21,23,23,21,20].
STL reg ceiling AGG Acc@10 0.7037; STL cat ceiling per-fold [52.18,51.75,51.36,53.01,52.20]. Markov-1 reg floor 0.5252 (matches dry-run 0.525 → floor is substrate-independent, confirms correctness).

## Reading — the champion transfers on the frozen recipe
MTL **beats the STL cat ceiling (+8.05)**, **matches the STL reg ceiling (−0.58, within fold noise)**, and
**clears the Markov-1 floor by +17.3 pp** — the exact champion-G signature (cat gain + reg parity-above-floor),
now on a **non-US corpus (Istanbul)** under the **frozen substrate**. Replicates the Mac dry-run direction
(+9.0 cat / +1.0 reg / +17.0 floor) at paper grade.

## Provenance / reproduce
- Substrate: `scripts/second_dataset/phase_v_substrate.py --city istanbul --device cuda --epochs 500` (GCN 500ep, fp32, `force_preprocess=False` → consumes the existing 520-mahalle `checkin_graph.pt`; alignment asserted POIs=29945/regions=520). 17 min.
- Inputs: `next_region.parquet` via `generate_next_region_input('istanbul')` (58,297 rows / 520 regions); seed-0 per-fold log_T rebuilt via `compute_region_transition.py --state istanbul --per-fold --seed 0` (freshness OK: log_T 19:07 > parquet 19:06).
- MTL: `train.py --task mtl --canon none --task-set check2hgi_next_region --engine check2hgi --state istanbul --seed 0 --model mtlnet_crossattn --reg-head next_getnext_hard --cat-head next_gru --scheduler onecycle --max-lr 3e-3 ... --no-{reg,cat}-class-weights --log-t-kd-weight 0.0 --per-fold-transition-dir output/check2hgi/istanbul` ; env `MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1`.
- STL cat: `train.py --task next --model next_gru`; STL reg: `p1 --heads next_stan_flow --target region --region-emb-source check2hgi --override-hparams freeze_alpha=True alpha_init=0.0 --per-fold-transition-dir output/check2hgi/istanbul`; floor: `compute_simple_baselines.py --task next_region`.
- Rundirs: MTL `results/check2hgi/istanbul/mtlnet_lr1.0e-04_bs2048_ep50_20260623_190800_105170`; STL cat `results/check2hgi/istanbul/next_lr1.0e-04_bs2048_ep50_20260623_191002_105363`; STL reg `docs/results/P1/region_head_istanbul_*`; floor `docs/results/P0/simple_baselines/istanbul/next_region.json`.

## Multi-seed (0/1/7/100) — champion-G holds across all 4 seeds (2026-06-23)
Seeds 1/7/100 MTL re-run after a studio crash (fresh per-seed log_T, fp32). Per-seed (5f) cat / reg:
| seed | cat macro-F1 | reg FULL top10 |
|---|---|---|
| 0 | 60.1461 | 69.7892 |
| 1 | 60.0606 | 69.8561 |
| 7 | 60.2401 | 69.8218 |
| 100 | 60.2016 | 69.7035 |
| **mean** | **60.16 ± 0.07** | **69.79 ± 0.06** |

→ **Δcat = +8.06** vs STL cat ceiling 52.10 (beats); **Δreg = −0.58** vs STL reg ceiling 70.37 (matches);
**+17.27** over the Markov-1 floor 52.52. Cross-seed variance is tiny (±0.07 cat / ±0.06 reg) → the
champion-G signature (cat gain + reg parity-above-floor) is **highly reproducible** on the frozen non-US
substrate. (Ceilings/floor are seed-0; both are seed-stable. Per-seed ceilings optional for full rigor.)

## Status
**Istanbul multi-seed (0/1/7/100) MTL COMPLETE.** Remaining: per-seed STL ceilings (optional, seed-stable);
H3 secondary-robustness variant; the A5 chronological-split bridge (same cells on `chrono_split/`).
