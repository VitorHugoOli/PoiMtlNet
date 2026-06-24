# HANDOFF — REDUCED board INDEX (deadline-grade, ~2-day budget) · 2026-06-23

> **Mode: deadline.** MobiWac (~2026-06-25). We run the **reduced 1-seed board** (`RUN_MATRIX_REDUCE.md`): seed 0
> × 5 folds (n=5, the minimal valid paired-test config, Wilcoxon p-floor 0.0312). This supersedes the prior 4-seed
> board handoffs (deleted). Read alongside: [`RUN_MATRIX_REDUCE.md`](RUN_MATRIX_REDUCE.md),
> [`RUN_MATRIX.md §0/§0a`](RUN_MATRIX.md) (recipe + precision pin), [`CA_MTL_DIVERGENCE.md`](CA_MTL_DIVERGENCE.md)
> (the fp16 root cause), [`STATISTICAL_PROTOCOL.md`](STATISTICAL_PROTOCOL.md), [`../../../articles/[mobiwac]/PAPER_PLAN.md`](../../../articles/%5Bmobiwac%5D/PAPER_PLAN.md) (what the paper needs).

## 1 · The picture (what changed)
- **Only MTL cells are invalid** (fp16-autocast ep30 collapse + reg understatement; CA −5.23 / TX −2.41 VOID).
  STL **reg** ceilings are already fp32 (REUSE: TX 64.96 / CA 63.48 / FL 76.71 / AL 69.98); STL cat ≈
  precision-insensitive; baselines (MPS=fp32) and the Istanbul dry-run are precision-settled.
- **Re-run = MTL only**, in **bf16** (`MTL_AUTOCAST_BF16=1`) pending the precision gate (else fp32).
- The region story likely **strengthens** (FL fp32 reg 77.71 > ceiling; AL −0.18) — NOT Decision-C's fp16 "trails".

## 2 · Device allocation (whole-state-per-device → every per-state Δ is device-internal-clean)
| Device | Owns | Order | Handoff |
|---|---|---|---|
| **H100** (fast, interrupts) | **precision gate** + AL, AZ, **FL**, CA | gate@AL → AL → **AZ‖FL co-scheduled** → **CA last** | [`HANDOFF_BOARD_H100.md`](HANDOFF_BOARD_H100.md) |
| **A40** (stable) | finish POI2Vec → **TX only** | POI2Vec → **TX (~11h)** | [`HANDOFF_BOARD_A40.md`](HANDOFF_BOARD_A40.md) |
| **Macs** (M2 Pro + M4 Pro, MPS) | **CTLE-SC + HMT-GRN** (6 states, device-internal) | AL(done)→AZ→FL→Istanbul; CA/TX may need CUDA | [`HANDOFF_BOARD_MACS.md`](HANDOFF_BOARD_MACS.md) |
| **(stretch)** | Istanbul external validity | reuse the MPS dry-run as the §6.3 provisional box | — |

Fig 3 plots **Δ's** (device-internal) → the split is valid; only an absolute cross-state table needs a
device-class footnote. The A100-equivalence A/B is **skipped** (not needed for per-state Δ headlines).

## 3 · The ONE gate that blocks MTL — precision (bf16 vs fp32), H100 @ AL, FIRST
`|Δcat|,|Δreg| ≤ 0.05 pp` ⇒ bf16 board-wide; else fp32. STL ceilings / baselines / Part-1 do NOT wait. STOP +
post the table for the user before the MTL fan-out (`RUN_MATRIX.md §0a`, `HANDOFF_BOARD_H100.md §1`).

## 4 · Process — each machine: OWN branch · incremental commits · OWN PR (audited + merged by the orchestrator)
`study/board-h100`, `study/board-a40`, `study/board-m2pro`. Commit per cell + result JSON + a 1-line finding;
push as you go; never merge another lane or main. Prior fp16 PRs are mergeable as records (we reset/re-run on top).

## 5 · What the paper needs (map to cells — PAPER_PLAN.md)
- **Part 1** (Tbl 2 / Fig 4): Check2HGI vs HGI STL cat (+ reg even) on overlap; embedding-quality fig (no training).
- **Part 2** (Tbl 3 / Fig 3, the headline): MTL champion-G vs STL cat + reg ceilings, 5 states.
- **Baselines** (§5.4 / §7): one-hot, skip-gram, POI2Vec, CTLE, STAN, Markov-1 (Macs).
- **External validity** (§6.3): Istanbul/NYC — the MPS dry-run box (provisional).

## 6 · Side question — push `check2hgi_dk_ovl` to Drive? **NO.**
`dk_ovl` = v14 embeddings (symlinks) + windowed parquets that are a **deterministic regen** from v14 + v11-structure
(fixed fold split + `--stride 1`), via `build_overlap_probe_engine.py <state> 1`. **v14 and v11 are already on
Drive**, so any machine rebuilds `dk_ovl` locally in minutes — pushing 30 GB of regennable parquets wastes Drive
space and sync time. **Push to Drive instead:** the **trained baseline embeddings** (CTLE/POI2Vec/skip-gram — real
compute, NOT cheaply regennable) + per-folder manifests. (Optionally the seeded per-fold `log_T` — small, but also
regennable via `compute_region_transition.py`.) See `HANDOFF_BOARD_MACS.md §4` for the rsync + manifest + Drive steps.

## 7 · STOP conditions (shared)
- Precision-gate table → STOP for user (§3). Any OOM / log_T-freshness / leak-guard / ungated-overlap
  (`MTL_STRICT=1`) failure → STOP. An MTL cell that still NaN-collapses under bf16 → STOP (fix insufficient).
- n=5 honesty: report the TOST power statement or mark region non-inferiority "n=5 provisional".
