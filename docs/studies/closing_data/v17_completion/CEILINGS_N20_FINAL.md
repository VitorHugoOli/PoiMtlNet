# v17 STL ceilings — FINAL, n=20, best-vs-best (dk_ovl, 5 Gowalla states)

> **Closed 2026-07-03.** Supersedes the seed-0 provisional ceilings in `RESULTS_BOARD §1` and the (rejected) matched-knob
> H2 attempt. Category ceiling re-tuned **best-vs-best** (each arm at its own optimum); region ceiling topped up to n=20.
> Istanbul is NOT here — it is gated on the H3 `dk_ovl` substrate build (both tasks seed-0 on stride-1 GCN today).

## Final board (n=20 = 4 seeds {0,1,7,100} × 5 folds)

| State | STL cat | MTL cat (v17) | **Δcat** | STL reg | MTL reg (v17) | **Δreg** |
|---|---:|---:|---:|---:|---:|---:|
| AL | 56.82 ±0.03 | 64.54 | **+7.72** ✅beats | 70.11 | 69.80 | **−0.31** ≈matches |
| AZ | 56.24 ±0.15 | 65.83 | **+9.60** ✅beats | 59.46 | 59.56 | **+0.10** ≈matches |
| FL | 74.51 ±0.03 | 79.85 | **+5.34** ✅beats | 76.70 | 77.42 | **+0.72** ✅beats |
| CA | 70.60 ±0.07 | 77.04 | **+6.44** ✅beats | 63.49 | 65.69 | **+2.20** ✅beats |
| TX | 69.79 ±0.08 | 77.23 | **+7.44** ✅beats | 64.95 | 67.07 | **+2.12** ✅beats |

**Story:** MTL beats the dedicated **category** ceiling at every state (+5.3 … +9.6 pp); **matches** the region ceiling at
the small states (AL −0.31, AZ +0.10, within δ=2 pp) and **beats** it at the larger ones (FL +0.72, CA +2.20, TX +2.12).

> ⚠ MTL cat/reg are n=20 at AL/AZ/FL (`perhead_lr_n20.md`) but **seed-0 (n=5) at CA/TX** (`catx_v17_seed0_5f`, fp32) —
> the H1/H100 top-up ({1,7,100}) firms the large-state Δ significance. The **ceilings** here ARE n=20 at all 5 states.

## Category ceiling — recipe (best-vs-best, state-size-dependent)

The STL cat optimum depends on state size (proven by the sweep, `cat_ceiling_sweep/`):

| tier | states | recipe | why |
|---|---|---|---|
| small | AL, AZ | **bs2048 @ max_lr 0.005** | small batch + low LR peaks highest; bs8192 loses ~0.35 pp |
| large | FL, CA, TX | **bs8192 @ max_lr 0.005** | +1.7 pp over bs2048 (healthy late best-epochs vs bs2048's early-peak overfit) |

Single-task `next_gru`, `--engine check2hgi_dk_ovl`, 50 ep, 5f, OneCycle. Scored by `score_stl_cat_ceiling.py`
(macro-F1 at f1-best epoch, fold-mean). Ceiling = per-state max over the tried recipes, n=20 mean (no per-fold/seed
cherry-pick). Full LR-response curves: `cat_ceiling_sweep/sweep_results/` + `aggregate.py`.

**Rejected: the "matched-knob" H2 (STL forced to the MTL's bs8192@1e-3).** It gave AL 53.58 — *below* the STL optimum,
inflating Δcat to +10.96. That is baseline sabotage (advisor panel unanimous); kept only as a labeled iso-budget
ablation, never the headline. See `A40.md §A40-1` + the panel record.

## Region ceiling — recipe + n=20 top-up

`p1_region_head_ablation.py --heads next_stan_flow --input-type region --region-emb-source
check2hgi_design_k_resln_mae_l0_1 --engine-override check2hgi_dk_ovl --override-hparams freeze_alpha=True
alpha_init=0.0 --max-lr 0.003 --target region`, 50 ep, 5f. Prior is **OFF** (α frozen at 0) → log_T inert → the
top-up omits `--per-fold-transition-dir` (parity-validated: AL s0 no-dir = 70.00 vs board 69.99). n=20 = board seed-0 +
`{1,7,100}` (`reg_ceiling_n20/`). Reg is seed-invariant (n=20 vs seed-0 diff < 0.13 pp everywhere) — the seed-0 board
verdict was correct; this makes the paired test rigorous.

## Provenance
- cat: `cat_ceiling_sweep/{sweep.sh,aggregate.py,sweep_results/}` · driver logs `cat_ceiling_sweep/DRIVER.log`
- reg: `reg_topup/{finalize_reg.sh,DRIVER.log}` · values `docs/results/closing_data/reg_ceiling_n20/`
- MTL cat/reg: `perhead_lr_n20.md` (AL/AZ/FL n=20) · `catx_v17_seed0_5f/RESULTS.md` (CA/TX seed-0)
