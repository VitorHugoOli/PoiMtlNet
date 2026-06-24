# CSLSL / cascade (B4) — AL/AZ results (M4 Pro / MPS, seed 0 × 5 folds)

> BASELINE_M4 Task 3. The **cascade (CSLSL/CatDM pattern)** baseline (`scripts/baselines/b4_cascade.py`):
> directed cat→region coupling (cond_coupling=posterior, cond_detach, **cross-attn severed**) on the FROZEN v14
> substrate + matched champion heads — so the **only** varying factor vs champion-G is cascade-vs-parallel.
> Durable per-state JSONs: `baseline_compare/{alabama,arizona}_cslsl_cascade.json`. Rundirs (gitignored):
> `results/check2hgi_design_k_resln_mae_l0_1/{state}/mtlnet_…ep50_…/`.

## Device + base caveat (READ FIRST)
- **All cells MPS, fp32, v14 engine `check2hgi_design_k_resln_mae_l0_1`, set-a windowing** (stride=9 non-overlap,
  min_seq=5). **MPS validated == CPU** (AL cross-check below).
- **The comparand is champion-G on the SAME v14 set-a base** (the b4 command minus the 5 cascade pins, cross-attn
  ON) — run here on the same device/engine/windowing/selector for a clean cascade-vs-parallel Δ. **It is NOT the
  board dk_ovl stride-1 champion** (cat 63.6/69.8 AL etc.) — that is a different base; do not compare across.

## Cascade-vs-parallel (champion-G) — the contribution gate (Δ = parallel − cascade)
| State | CSLSL cascade cat | Champion-G (parallel) cat | **Δcat** | CSLSL cascade reg@10 | Champion-G reg@10 | **Δreg** | joint (casc/par) |
|---|---|---|---|---|---|---|---|
| Alabama | 45.93 ±2.32 | 50.97 | **+5.04** ✅beats | 63.42 ±3.19 | 63.98 | **+0.56** ✅beats | 0.540 / 0.571 |
| Arizona | 53.21 ±0.98 | 54.83 | **+1.62** ✅beats | 54.48 ±3.39 | 54.43 | **−0.05** ≈ties | 0.538 / 0.546 |

*(cat = macro-F1; reg = top10_acc_indist; both at the joint geom_simple checkpoint. n=5, seed 0 only — provisional.)*

**Read:** the **parallel symmetric cross-attention coupling (champion-G) beats or matches the directed cat→region
cascade** at both states — cat **+5.04 (AL) / +1.62 (AZ)**, reg **+0.56 (AL) / −0.05 (AZ, tie)**. This is the
expected cascade signature: severing cross-attn costs the **category** head the most (it loses the reg→cat help),
while reg still gets the cat→reg edge, so cat drops more than reg. Supports the contribution that **our parallel
cross-task coupling ≥ the published cascade (CSLSL when→what→where / CatDM) pattern** under matched substrate +
heads. Honest framing: "parallel beats the cascade on category at both states, beats/ties on region" — never "crushes."

## MPS == CPU cross-check (Alabama, the device-trust gate)
| metric | cascade MPS | cascade CPU (`INGRED_DEVICE=cpu`) | Δ (MPS − CPU) |
|---|---|---|---|
| cat macro-F1 | 45.93 | 46.56 | −0.63 |
| reg top10_acc_indist | 63.42 | 63.64 | −0.23 |

Both within fold noise (cat fold-std ±2.3pp) → **MPS is trustworthy** for this from-scratch joint MTL run
(consistent with the HMT-GRN MPS==CPU finding, PR #38). The `[M4/MPS]` label stands.

## Provenance / reproduce
- Prereqs (both states): v14 substrate `output/check2hgi_design_k_resln_mae_l0_1/<state>/` + set-a inputs
  (`input/next_region.parquet`: AL 12,709 / AZ 26,396 rows, n_regions 1109/1547) + per-fold seeded log_T
  (`region_transition_log_seed0_fold{1..5}.pt`, train-only, fresh). **AZ inputs + log_T were built this session**
  (`generate_next_input_from_checkins` default windowing + `build_next_region_for` + `compute_region_transition
  --per-fold --seed 0 --n-splits 5`); AL was already on disk.
- Cascade: `PYTHONPATH=src MTL_RAM_HEADROOM_GB=4 python scripts/baselines/b4_cascade.py --state <s> --seed 0 --folds 5 --epochs 50`.
- Parallel comparand: the same train.py invocation **minus** the 5 pins (`--reg-head-param cond_*` ×4 +
  `--model-param disable_cross_attn=True`) → champion-G matched-head on v14 set-a.
- `MTL_RAM_HEADROOM_GB=4` overrides the conservative CPU-RAM guard (datasets are ~0.1–0.3 GB; default 16 GB
  headroom false-trips on a 24 GB box). The benign `fvcore is required for FLOPs` line is non-fatal.

## Status
- ✅ **CSLSL/cascade @ AL + AZ done on the M4 [M4/MPS], with matched parallel champion-G comparand + MPS==CPU
  cross-check.** n=5 provisional (seed 0). Multi-seed {1,7,100} + FL/CA/TX cascade are post-deadline / A40 lane.
