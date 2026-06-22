# Board lane H100 — findings & resume notes

> Branch `study/board-h100`, PR #32. Companion to `BOARD_H100_STATUS.md` (setup/decision).

## ⚠ INTERRUPTED 2026-06-22 ~05:45 UTC — GPU studio stopped
The Lightning Studio's **H100 was stopped/restarted** (SIGKILL / exit 137); the shell is now on a CPU-only
fallback VM (`torch.cuda.is_available()=False`, 4 CPU / 14 GB RAM, no `nvidia-smi`). **The persistent
filesystem survived** — rebuilt v14 substrates (FL+CA), gated-overlap engines (FL+CA), seeded log_T, and the
FL STL-ceiling results are all intact. **Resume requires restarting the H100 studio** (see Resume below).
> Operational note: stopping this studio detaches the GPU but keeps `/teamspace/studios/this_studio`.

## Results so far (seed 0, 5 folds, gated-overlap `check2hgi_dk_ovl`, compiled+tf32)

### FL — A40 (frozen substrate) vs H100 (rebuilt substrate)
| FL cell | A40 frozen | H100 rebuilt | Δ |
|---|---|---|---|
| **STL reg ceiling** (full top10, fp32) | 76.7138 | **76.7674** | **+0.05** ✅ |
| **STL cat ceiling** (macro-F1) | *(not run by A40)* | **75.46** | ref ~75.2 ✅ |
| MTL cat macro-F1 | 79.0083 | ⏳ died fold1/ep25 (best C77.84) | re-run |
| MTL reg full top10 | 75.5000 | ⏳ died fold1/ep25 (best N75.43) | re-run |

**Key validation:** despite the rebuilt substrate being byte-different from the frozen artifact
(`V14_REBUILD_H100_PROVENANCE.json`), the FL STL reg ceiling reproduces the A40's to **+0.05 pp** — the
own-states rebuild is functionally equivalent at the metric level. A40 FL device-internal Δreg (MTL−STL) =
**−1.21 pp** (within δ_reg=2pp); the H100 MTL re-run will confirm its own Δ.

## Lessons — parallelism on a GPU-rich / host-RAM-tight box (H100 80GB GPU, 108GB host)
1. **GPU memory was never the limit** (cells ~17 GB, util ~15%, transfer-bound). But the A40's "serial" rule
   still matters here for a different reason:
2. **GPU contention pushes datasets CPU-resident.** `folds._dataset_device` auto-fit decides at fold start
   using *free* VRAM; when cells share the GPU, the loser keeps its dataset on CPU → per-batch transfer
   (slower). For peak speed, run the big cells SERIALLY so each gets a GPU-resident dataset.
3. **The p1 STL-reg val-metric is the OOM hazard, not the model.** It accumulates the full val logit
   `[N_val,C]` and `torch.cat`s it (~2× = ~37 GB GPU at CA's C=8501). Options: GPU path (needs the cell ~alone
   on the GPU) or CPU path (`P1_CHUNK_VAL_METRIC=1`, frees GPU but spikes ~20 GB host — bad on this RAM-tight
   box). CA OOM'd both ways while sharing with two FL cells. **On THIS box: run CA STL-reg with the GPU val
   path (`P1_S2_AUTO_BUDGET_GB=30`) but give it the GPU to itself or pair only with one light cell.**
4. Cherry-picked the A40's p1 fix (`61a7e0fd`) onto this branch (CPU val-metric option). MTL cells use S2
   (`MTL_CHUNK_VAL_METRIC=1`) and are GPU-friendly/host-light.

## RESUME (when the H100 studio is back)
Env persists (torch 2.11.0+cu128 + pt211 pyg). Re-export the board env, then **run cells SERIALLY**:
1. **FL MTL** (re-run; died fold1): `scripts/closing_data/h100_fl_cells.sh` already ran Cells 1–2 (results on
   disk) — just re-run Cell 3, or the standalone champion-G command in `logs/h100_fl_mtl_s0.log`'s header.
   Score with `h100_score_matched.py <rundir> --seed 0` → FL Δcat / Δreg vs the ceilings above.
2. **CA**: `scripts/closing_data/h100_ca_reg.sh` (Cell A GPU-val + Cell B MTL) — run when the GPU is free of
   other big cells. CA STL-reg val-cat is ~37 GB; keep it alone or with one light cell.
3. Commit MTL + matched-score JSONs (C28); update the FL table; flag PR #32 for audit.
