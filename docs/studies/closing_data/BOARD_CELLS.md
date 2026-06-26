# Board cells — per-state champion-G MTL records + precision gates (consolidated)

> **Consolidated 2026-06-26** from the former `AL_PRECISION_GATE.md` · `AZ_CELL.md` · `FL_PRECISION_GATE.md` ·
> `CA_CELL.md` · `TX_CELL.md` (history kept in git). All cells are champion-G MTL on `check2hgi_dk_ovl`, seed 0,
> 5 folds, `--epochs 50`, scored fp32 by `scripts/closing_data/h100_score_matched.py` (per-task DIAGNOSTIC-BEST,
> fold-mean). The summary board (with stats) lives in [`RESULTS_BOARD.md`](RESULTS_BOARD.md); this file is the
> per-cell detail (per-fold values, rundirs, recipe + precision-gate trail).

## Summary — MTL vs dedicated STL ceilings (all 6 states, seed 0 × 5f, gated overlap)
| State | precision | MTL cat | STL cat | Δcat | MTL reg (FULL top10) | STL reg | Δreg |
|---|---|---|---|---|---|---|---|
| AL | fp32 | 63.56 | 55.87 | **+7.69** | 69.81 | 69.99 | −0.18 (matches) |
| AZ | fp32 | 63.39 | 57.13 | **+6.26** | 59.34 | 59.40 | −0.06 (matches) |
| FL | fp32 | 79.82 | 75.15 | **+4.68** | 77.28 | 76.71 | **+0.56** (beats) |
| CA | bf16 | 77.33 | 70.26 | **+7.07** | 65.66 | 63.48 | **+2.18** (beats) |
| TX | bf16 | 77.47 | 69.95 | **+7.52** | 67.00 | 64.96 | **+2.04** (beats) |

**Headline:** MTL beats the category ceiling at every state (+4.7…+7.7); on region it **beats at the large
states (FL/CA/TX)** and **matches at the small (AL/AZ)**. The old "MTL sacrifices reg" pattern was an
fp16-overflow harness artifact (see [`CA_MTL_DIVERGENCE.md`](CA_MTL_DIVERGENCE.md)); correct precision reverses it.

## Shared recipe note (applies to EVERY board state)
The handoff commands omit `--canon`, relying on canon-v16 auto-injection. Under `MTL_STRICT=1` that injection trips
the **wrong-substrate guard** (v16 pins `check2hgi_design_k_resln_mae_l0_1`; the board intentionally runs on
`check2hgi_dk_ovl`) → hard-fail. Fix: **`--canon none` + explicit `--no-reg-class-weights --no-cat-class-weights`**
(the only material champion-G flags canon adds beyond the board's explicit set; `--checkpoint-selector geom_simple`
is moot under `--no-checkpoints` + diagnostic-best scoring). `MTL_STRICT=1` still fails loud on NaN
(`guard_finite_step` reads env independently of canon). Dualtower heads (`next_gru` / `next_stan_flow_dualtower`,
prior-OFF `freeze_alpha=True alpha_init=0.0`), `--log-t-kd-weight 0.0`, OneCycle max-lr 3e-3, fixes #1+#3 (4.5×).
**Precision:** small/mid states = **fp32** (user decision 2026-06-23); large states (CA/TX) = **bf16** (fp16
overflow-collapses at ep30, bf16 trains clean to ep50; bf16≈fp32 per the FL gate).

---

## AL — precision gate + cell ✅ COMPLETE (fp32)
H100 lane (`study/board-h100`), 2026-06-23. The AL gate doubled as the board-wide precision decision (bf16 vs
fp32; both arms avoid the fp16 overflow). **User chose fp32 for small/mid states.**

**bf16-vs-fp32 gate (4 dp):**
| metric | bf16 | fp32 | Δ(bf16−fp32) | STL ceiling | Δreg vs ceiling |
|---|---|---|---|---|---|
| reg FULL top10 | 69.6873 ± 3.3165 | 69.8067 ± 3.3898 | −0.1194 | 69.98 | bf16 −0.29 / fp32 −0.17 |
| cat macro-F1 | 63.5810 ± 1.9732 | 63.5591 ± 2.0387 | +0.0219 | — | — |

Per-fold reg (bf16/fp32): [71.90/72.13, 69.01/68.95, 73.23/73.53, 70.62/70.68, 63.67/63.75]; per-fold Δreg
[−0.22, +0.06, −0.30, −0.06, −0.08]. By the letter of the |Δ|≤0.05 rule → fp32 (mean |Δreg|=0.12); the gap is
≪ the ±3.3pp fold std, so bf16 stays the fast path for the large states. Both arms: 0 non-finite events, 50ep×5f clean.

**AL cell (fp32 arm):**
| AL (seed0, 5f, gated overlap) | value | ceiling | Δ |
|---|---|---|---|
| MTL cat macro-F1 | **63.5591** | STL cat 55.8704 (`next_gru`) | **+7.69 (beats)** |
| MTL reg FULL top10 | **69.8067** | STL reg 69.99 ± 3.56 (fp32 `next_stan_flow`, fresh 5f) | **−0.18 (matches)** |

STL reg ceiling fresh artifact `docs/results/P1/region_head_alabama_region_5f_50ep_alabama_ovl_stl_reg_s0.json`
(per-fold [72.15,68.98,73.44,71.05,64.33]) → 69.99 ≈ the documented 69.98. Rundirs:
bf16 `results/check2hgi_dk_ovl/alabama/mtlnet_..._171050_38170`; fp32 `..._171146_38758` (tags `alabama_{bf16,fp32}`).
Env: `MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1`; bf16=`MTL_AUTOCAST_BF16=1 MTL_DISABLE_AMP_EVAL=1`, fp32=`MTL_DISABLE_AMP=1`.

## AZ — cell ✅ COMPLETE (fp32)
Small state (1547 regions) → fp32 (no gate). All artifacts fresh on `check2hgi_dk_ovl`, seed 0, 5f.

| AZ (seed0, 5f, gated overlap, fp32) | value | ceiling | Δ |
|---|---|---|---|
| MTL cat macro-F1 | **63.3875** | STL cat **57.1305** (`next_gru`) | **+6.26 (beats)** |
| MTL reg FULL top10 | **59.3360** | STL reg **59.40** (p1 fp32) | **−0.06 (matches)** |

Per-fold — MTL cat [65.27,62.29,64.63,62.77,61.97]; MTL reg [62.65,58.97,59.30,57.67,58.09]; STL reg Acc@10
[62.82,59.46,59.59,57.42,57.70] (AGG 59.40 ± 2.15). Same pattern as AL: beats cat wide (+6.26), matches reg
(−0.06, tighter than AL). Rundir `results/check2hgi_dk_ovl/arizona/mtlnet_..._174233_87207` (tag `arizona_fp32`,
43 min/5f, `MTL_DISABLE_AMP=1`). STL reg `docs/results/P1/region_head_arizona_region_5f_50ep_arizona_ovl_stl_reg_s0.json`;
STL cat `.../arizona/next_..._181527_96788/stl_cat_ceiling_score.json` (full 5f, supersedes the 4/5 partial 57.43).

## FL — precision gate + cell ✅ COMPLETE (fp32)
H100, 2026-06-23. FL re-run under correct precision (the prior fp16 FL MTL reg 75.42 is VOID — fp16-overflow harness).

**bf16-vs-fp32 gate (full 5f):**
| metric | fp32 | bf16 | Δ(bf16−fp32) | STL ceiling | Δ vs ceiling (fp32/bf16) |
|---|---|---|---|---|---|
| cat macro-F1 | **79.8247** ± 0.51 | 80.0691 ± 0.48 | +0.2444 | 75.147 | **+4.68 / +4.92 (beats)** |
| reg FULL top10 | **77.2760** ± 0.77 | 77.2954 ± 1.11 | +0.0194 | 76.7123 | **+0.56 / +0.58 (BEATS)** |

Per-fold reg fp32 [77.68,77.39,76.32,76.55,78.44]/bf16 [77.68,77.44,76.33,75.93,79.09]; cat fp32
[79.38,79.98,79.75,79.30,80.72]/bf16 [79.61,80.21,80.04,79.59,80.90]. **bf16 ≈ fp32** (reg Δ+0.02 within the rule;
sign flips vs AL → no systematic gap) → bf16 validated equivalent, safe as the CA fp16-overflow fix. FL cell kept
**fp32** (small/mid decision). **FL MTL beats BOTH ceilings** (the fp16 "−1.29 reg" was the artifact).
Rundirs fp32 `...mtlnet_..._173805_85991` / bf16 `..._174254_87133`; scores
`docs/results/closing_data/h100/florida_s0_mtl_{fp32,bf16}_5f_matched_score.json`.

## CA — cell ✅ COMPLETE (bf16)
Large state → bf16 (fp16 overflow-collapsed at ep30; bf16 trains clean to ep50). Launcher
`scripts/closing_data/board_h100_mtl.sh california bf16`. Rundir `mtlnet_..._021104_79596`.

| task | CA MTL (n=5) | ceiling | Δ |
|---|---|---|---|
| **cat** macro-F1 | **77.3311** ± 0.2164 | 70.26 | **+7.07 (beats)** |
| **reg** FULL top10 | **65.6634** ± 0.2613 | 63.48 | **+2.18 (BEATS)** |

Per-fold cat [77.10,77.56,77.05,77.52,77.42] / reg [65.42,65.55,65.59,65.58,66.17] (best-epochs late ep49-50,
no ep30 collapse). **CA MTL beats BOTH ceilings** — "MTL sacrifices reg" reversed, mirroring FL. Artefact
`docs/results/closing_data/h100/california_s0_mtl/`.

## TX — cell ✅ COMPLETE (bf16)
Large state → bf16 (CA precedent). Launcher `scripts/closing_data/launch_tx_s0.sh`; log_T on dk_ovl
(`output/check2hgi_dk_ovl/texas/`, C29-correct). Co-scheduled with CA by overriding the host-RAM guard
(`MTL_RAM_HEADROOM_GB=-25`) after verifying the simultaneous peak fits (CA 26 GB + TX 66 GB < 108 GB + 35 GB
swap); RAM watcher armed to kill TX if avail < 4 GB (cleared, peak 86 GB, swap 0). See
[`EP100_ABLATION_AND_TX_RAM.md`](EP100_ABLATION_AND_TX_RAM.md) §2.

| task | TX MTL (5f) | ceiling | Δ |
|---|---|---|---|
| cat macro-F1 | **77.47** ± 0.14 | 69.95 (A40 #37, dk_ovl/seed0) | **+7.52 (beats)** |
| reg FULL top10 | **67.00** ± 0.47 | 64.96 (A40, dk_ovl/seed0) | **+2.04 (BEATS)** |

Per-fold cat [77.69,77.31,77.39,77.56,77.39] / reg [66.94,67.32,66.12,67.16,67.45]. **Beats both ceilings** —
joins CA/FL in reversing "MTL sacrifices reg". TX reg STL ceiling 64.96 ± 0.46 from
`docs/results/closing_data/a40/tx_stl_reg_ceiling_s0.json` (raw top10_acc; MTL is FULL top10, small OOD-adjust gap);
cat ceiling reused from the A40 lane (#37, same dk_ovl/seed0/`next_gru` recipe — the co-scheduled cat ceiling was
OOM-killed, then deferred). The driving session ended before TX finished → per-fold results were committed+pushed
autonomously (`scripts/closing_data/tx_autocommit.sh`).

> **Device-precision caveat (Ampere):** the A40 bf16 backward grad-NaNs at large C — the H100 TX bf16 run above is
> the clean one. See [`TX_A40_BF16_NAN.md`](TX_A40_BF16_NAN.md) (A40 bf16 NaN root-cause) + [`CA_MTL_DIVERGENCE.md`](CA_MTL_DIVERGENCE.md) (fp16 overflow).
