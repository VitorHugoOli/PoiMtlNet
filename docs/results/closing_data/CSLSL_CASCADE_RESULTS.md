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

## Leak / correctness audit (2026-06-24, advisor-reviewed) — VERDICT: CLEAN
The cascade edge is **leak-free by construction**, and the comparison is **fair** (apples-to-apples information access):
- **No label access in the coupling.** `cat_cond = softmax(out_cat)` where `out_cat = category_poi(shared_cat)` is
  the model's OWN forward-pass category prediction from the input embeddings (`src/models/mtl/mtlnet_crossattn_dualtower/model.py:97-127`)
  — never a label/target. With the pinned `cond_detach=True` the reg head reads it **detached**
  (`src/models/next/next_stan_flow_dualtower/head.py:453`), so the reg loss cannot flow back into the cat head
  (feed-forward cascade). `cond_proj` is **zero-init** → the untrained cascade head ≡ champion-G. This is the
  standard CSLSL "downstream stage reads the upstream *prediction*" pattern, not label leakage.
- **Fair comparand.** The parallel champion-G also has both streams (via cross-attn); only the coupling *topology*
  differs (bidirectional vs directed cat→region). Same information access → clean cascade-vs-parallel contrast.
- **Inherited pipeline guards (verified at runtime on every cell):** b4 leak-preflight (per-fold log_T exists +
  fresh, refuses stale); per-fold log_T is **train-only** (`compute_region_transition --per-fold`,
  n_train_rows-only); folds are **user-disjoint** StratifiedGroupKFold (confirmed in each run's `users train=…/val=…`
  lines, e.g. FL fold1 train 11,139 users / val 2,796, disjoint); region partition is a fixed geographic map
  (poi_to_region), not label-derived. The AZ/FL inputs built this session reuse the same shared graph maps → no new
  leak surface.

## Florida — ATTEMPTED on M4, INCOMPLETE (MPS OOM) → run on CUDA
FL CSLSL was attempted on the M4 (v14 set-a, 159,175 rows / 4,703 regions). The cascade **trained but OOM'd on MPS
at the summary-aggregation step** — `MPS backend out of memory (… other allocations: 25.06 GiB, max 30.19 GiB)` on
the 24 GB box (swap-oversubscribed; Chrome + Android emulator co-resident). Only **4/5 folds** completed; `summary/`
was never written. Recovered from the per-fold val CSVs (geom_simple joint-best epoch):

| State | CSLSL cascade (4-fold partial) cat | reg@10 | comparand | status |
|---|---|---|---|---|
| Florida | 71.08 ±0.23 | 72.81 ±0.56 | **none** (champion-G killed at fold 1 to protect the box) | ⚠ INCOMPLETE — **no Δ** |

**Do NOT cite the FL cascade number** — it is 4-fold partial with no matched champion-G comparand, so the
cascade-vs-parallel Δ (the whole point) is not computable here. The MPS allocation hit the hardware ceiling and
risked an OOM-reboot, so both FL runs were stopped.

**Recommendation: run FL CSLSL on CUDA (A40/H100)** — its documented lane (`BASELINE_A40.md`: *"if the H100 is
saturated you may take CSLSL @ FL — you're the stable card for the longer run"*). The M4 24 GB is insufficient for a
large state under normal desktop load. (`baseline_compare/florida_cslsl_cascade.json` carries the partial numbers +
the blocker, status `INCOMPLETE_M4_MPS_OOM`.)

## Engineering knowledge — lessons settled this session (reusable)
1. **`b4_cascade.py` has no `--device` flag.** Device auto-detects (`configs.globals.DEVICE` → MPS on the M4;
   `INGRED_DEVICE=cpu` forces CPU for the cross-check). The handoff's `--device mps` is a no-op/error — omit it.
2. **CPU-RAM guard false-trips on small states.** `_guard_cpu_resident_ram` demands a 16 GB default head-room; on a
   24 GB box with desktop apps it refuses tiny (0.1–0.4 GB) AL/AZ datasets. Set **`MTL_RAM_HEADROOM_GB=4`** to pass
   (the `ram_watchdog`/OS is the real OOM backstop). Necessary for every M4 cascade/champion run.
3. **AZ v14 inputs are buildable, AL/FL were on disk.** Build set-a v14 inputs with
   `generate_next_input_from_checkins(state, EmbeddingEngine('check2hgi_design_k_resln_mae_l0_1'))` (defaults →
   stride=9/set-a, matches AL's 12,709 / AZ 26,396 rows) + `build_next_region_for(state, engine)`, then the per-fold
   log_T (`compute_region_transition --engine <v14> --per-fold --seed 0 --n-splits 5`). Verify log_T mtime >
   `next_region.parquet` (standing trap).
4. **`--only-fold` collides with the canon-injected `--folds`** (`--task mtl` auto-injects v16 canon incl. `--folds`).
   For a 1-fold device cross-check, run the **full 5-fold** on CPU and compare aggregates instead (what we did).
5. **The b4 driver does NOT set `MTL_CHUNK_VAL_METRIC=1`** (unneeded at AL/AZ's 1109/1547 regions). **At FL's 4,703
   regions on the swap-bound M4 this is the failure point:** the run trained 4 folds then **MPS-OOM'd at summary
   aggregation** (`other allocations 25 GiB` ≈ the 30 GiB MPS cap; the box was swap-oversubscribed with Chrome +
   Android emulator co-resident). Lesson: **large-state CSLSL needs CUDA** (its documented A40/H100 lane), or at
   minimum `MTL_CHUNK_VAL_METRIC=1` + a quiesced box — but the M4's 24 GB is the binding constraint, not a flag.
6. **Disk discipline.** The cascade (no `--no-checkpoints`) writes a BestTracker `checkpoints/` dir (~0.5 GB/state
   small, **2.6 GB for FL**); on the near-full SSD this tripped the disk monitor. Delete a run's `checkpoints/`
   **after** it completes (the `summary/` JSON is the durable artifact) — never mid-run (BestTracker reads it). The
   champion comparand uses `--no-checkpoints` so it never grows disk. Pass `--no-checkpoints` to the cascade too if
   only the metrics are needed.
7. **Result recovery without `summary/`.** If summary aggregation crashes (OOM), reconstruct the geom_simple result
   from `metrics/fold{N}_next_{category,region}_val.csv`: per fold pick the epoch maximising
   `sqrt(cat_f1 · reg_top10_acc_indist)`, read cat/reg there, average over folds. (Used to recover FL's 4 folds.)
