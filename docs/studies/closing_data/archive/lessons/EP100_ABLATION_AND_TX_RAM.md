# 100-epoch schedule ablation + TX host-RAM guard mechanics (2026-06-24 session)

Two reusable findings from the H100 board session, layered on the fixes #1+#3 (GPU-resident
train-metric + batched `index_select` collate, ~4.5× on wide-head CA/TX — merged in PR #33).

## 1 · 100-epoch schedule ablation — NULL at both AL and FL

champion-G MTL on `check2hgi_dk_ovl`, seed 0, 5f, identical recipe except `--epochs 50→100`
(which stretches the OneCycle warmup+anneal across 100 epochs). Diagnostic-best, fold-mean.

| state | 50ep cat | 100ep cat | 50ep reg | 100ep reg | verdict |
|---|---|---|---|---|---|
| **AL** (fp32, 5f) | 63.56 | **63.77** (+0.21) | 69.81 | **69.42** (−0.39) | **NULL** — cat flat (≪±2pp fold std), reg slightly worse |
| **FL** (fp32, fold 1 only) | 79.38 | **78.85** (−0.53) | 77.68 | **77.50** (−0.18) | **NULL** — killed at fold 1, both slightly down |

**Mechanism — the "late best-epoch" trap.** FL's 50ep best-epochs sit at **ep43-50** (cat) /
**ep43-50** (reg) — right at the schedule tail — which *looks* like "still climbing, give it more
epochs." It is not. With OneCycle, the best validation lands near the **low-LR anneal tail**
regardless of schedule length. At 100ep FL's peak just **relocates to ep69/79** (NOT ep100) and
lands marginally **lower**. AL confirms the same from the other side: AL peaks **early (ep16-41)**
and 100ep adds nothing. So a best-epoch near the schedule end is the **anneal shape**, not unsaturated
capacity — do **not** read it as "needs more epochs." The frozen **50ep** board cells stand.

## 2 · TX host-RAM guard double-counts → negative-headroom is the safe force knob

TX is the largest state (3.83M dk_ovl rows). The check2hgi-MTL dataset construction needs a
**~66 GB host-RAM peak** (`matrix × (1+n_region_towers) × 4.0`, `folds._guard_mtl_check2hgi_ram`).
This is **intrinsic + host-side** — building the master X + region tower + per-fold slices on CPU —
so `MTL_DATASET_GPU=1` does **NOT** avoid it (per-batch GPU footprint is N-independent; the peak is
the host construction, not the resident tensor).

**The guard double-counts.** It fires *inside* TX *after* `load_next_data` has already consumed
~25 GB (parquet → X matrix + page cache), so `psutil.available` has already dropped (75→50 GB), yet
the guard compares that depressed `avail` against the **total** 66 GB peak — not the **remaining**
~41 GB TX still needs. So it blocks even when the run would fit:

```
CA resident 26 GB + TX peak 66 GB = 92 GB on a 108 GB box (+35 GB swap) → fits with ~16 GB margin
```

**Force technique (verified safe here):** set **negative** `MTL_RAM_HEADROOM_GB` (e.g. `-25`) to add
slack that cancels the double-count: guard passes (`65.8 < avail(50)+25`), TX builds the remaining
~41 GB into the 50 GB available. Observed: construction peak `used=86 GB / avail=22 GB`, **swap
stayed 0**, TX reached training, **CA untouched** (0 NaN throughout). `launch_tx_s0.sh` is the
reusable launcher; pass `MTL_RAM_HEADROOM_GB=-25` when co-scheduling with a large-state run.

**Reconciling with the "never override headroom" rule** (BOARD_H100_FINDINGS §5 / memory): that rule
holds for a **genuinely full** host (the prior OOM was FL-gate + cat-ceilings all competing for real
RAM). The override is safe **only after you VERIFY** the simultaneous peak fits — measure
`psutil.available` **and** per-process RSS (`/proc/PID/status VmRSS`), compute
`Σ resident + new_construction_peak < total_RAM`, and keep a **RAM-safety watcher** that kills the
**newest** proc (TX, never CA) if `avail` drops below ~4 GB. Positive-headroom block = correct default;
negative-headroom force = a measured, watched exception, not a habit.

## 3 · TX per-fold log_T built on the dk_ovl engine (C29-correct)

The trainer **hard-fails** (`FileNotFoundError`, `mtl_cv.py`) when `--per-fold-transition-dir` is set
but the seed-tagged `region_transition_log_seed{S}_fold{N}.pt` is missing — it loads log_T **even
prior-OFF** (just multiplies by 0). TX had **0/5** seed-0 files. Built them with
`compute_region_transition.py --state texas --per-fold --seed 0 --engine check2hgi_dk_ovl` → written to
`output/check2hgi_dk_ovl/texas/`, and TX reads from there (not the design_k dir AL/FL used). This is
**stronger than AL/FL**: the prior is built on the *exact* training split (dk_ovl), so it is leak-free
*and* guard-clean (n_splits=5, fresh mtime), not merely inert. See [[c29-reg-prior-split-leak]].

## Result anchors (this session)
- **AL 100ep**: cat 63.77 / reg 69.42 (rundir `...ep100_...030914_86549`).
- **FL 100ep**: fold 1 cat 78.85 @ep69 / reg 77.50 @ep79 (killed; `...ep100_...033715_90767`).
- **CA §4** (50ep bf16): folds 1-4 cat **77.31**±0.24 / reg **65.54**±0.07 — beats ceilings
  70.26 / 63.48 by **+7.05 / +2.06**; full cell pending fold 5.
- **TX** (50ep bf16, forced): running; per-fold results appended to `BOARD_CELLS.md`.
