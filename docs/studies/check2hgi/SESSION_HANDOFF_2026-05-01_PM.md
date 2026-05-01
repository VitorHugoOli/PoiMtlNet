# Session Handoff — 2026-05-01 PM (Camera-Ready Gap-Fill on H100)

**Branch:** `worktree-check2hgi-mtl`
**Hardware:** H100 80 GB (Lightning Studio)
**Session start:** ~17:58 UTC
**Status at handoff:** Gap 2 complete; Gap 1 MTL in progress

Picks up from [`SESSION_HANDOFF_2026-05-01.md`](SESSION_HANDOFF_2026-05-01.md).
Scope defined by [`H100_CAMERA_READY_GAPS_PROMPT.md`](H100_CAMERA_READY_GAPS_PROMPT.md).

---

## TL;DR — what landed

| Gap | Description | Runs | Status |
|---|---|---|---|
| **Gap 2** | AL + AZ + FL STL cat `next_gru` multi-seed {0,1,7,100} | 12 | ✅ all exit 0 |
| **Gap 1** | CA + TX MTL multi-seed B9 vs H3-alt {0,1,7,100} | 16 | 🔄 in progress |

Launcher: `scripts/run_h100_camera_ready_gaps.sh`
Logs: `logs/h100_gaps_*.log`, `logs/h100_camera_ready_gaps.master.log`

---

## 1 · Gap 2 — STL cat `next_gru` multi-seed (DONE ✅)

### 1.1 · Per-state results

All 12 runs completed clean (exit 0). Seeds {0, 1, 7, 100}; 5f × 50ep; `--max-lr 3e-3`; `--batch-size 2048`; `--no-checkpoints`.

| State | n_regions | Seeds | Mean F1 | σ | Min F1 | Max F1 | MRR (mean) | Top3 (mean) |
|---|---:|:-:|---:|---:|---:|---:|---:|---:|
| **Alabama** | 1,109 | {0,1,7,100} | **41.35%** | 0.17% | 41.18% | 41.51% | 64.79% | 84.47% |
| **Arizona** | 1,547 | {0,1,7,100} | **43.90%** | 0.17% | 43.77% | 44.13% | 66.99% | 85.47% |
| **Florida** | 4,703 | {0,1,7,100} | **67.16%** | 0.13% | 66.98% | 67.30% | 80.99% | 90.03% |

All within-state σ < 0.2 pp F1 — extremely stable across seeds.

### 1.2 · Comparison to seed=42 anchor (prior §0.3 table)

| State | seed=42 F1 (prior) | Multi-seed mean F1 | Δ vs anchor |
|---|---:|---:|---:|
| AL | 40.76 ± 1.68 (5-fold σ) | **41.35 ± 0.17** (seed σ) | +0.59 pp |
| AZ | 43.21 ± 0.87 (5-fold σ) | **43.90 ± 0.17** (seed σ) | +0.69 pp |
| FL | 66.98 ± 0.61 (F37, 1 seed) | **67.16 ± 0.13** (seed σ) | +0.18 pp |

Anchor numbers were from a single seed=42 run; multi-seed means are consistent within <1 pp. **The cat-side narrative does not change.**

### 1.3 · Impact on RESULTS_TABLE §0.1

The STL `next_gru` F1 column in §0.1 (architectural-Δ table) can now be updated
with multi-seed means + seed σ instead of single-seed ± fold-σ values:

| State | Old STL cat F1 | New STL cat F1 (multi-seed) |
|---|---|---|
| AL | 40.76 ± 1.68 (n=1 seed) | **41.35 ± 0.17** (n=4 seeds) |
| AZ | 43.21 ± 0.87 (n=1 seed) | **43.90 ± 0.17** (n=4 seeds) |
| FL | 66.98 ± 0.61 (n=1 seed, F37) | **67.16 ± 0.13** (n=4 seeds) |

Δ_cat column values shift slightly but the sign and significance are unchanged.
Updated in RESULTS_TABLE §0.1 by this session (see git diff).

---

## 2 · Gap 1 — CA + TX MTL multi-seed B9 vs H3-alt (IN PROGRESS 🔄)

16 runs total: 2 states × 4 seeds × 2 recipes at 5f × 50ep, strictly serial
(CA MTL peaks ~49 GB VRAM; 2-way would OOM).

- CA B9 seed=0 started at 17:58 UTC, on epoch ~29–50 at last check.
- Remaining: CA B9 {1,7,100} + CA H3-alt {0,1,7,100} + TX {all 8}.

**Next action:** once CA finishes, restructure launcher to run TX 2-way parallel
(TX regions=6553 → peak ~29 GB → two TX runs ≈ 57 GB, fits in 80 GB).
This saves ~25% of the remaining TX wall time (~26 min).

---

## 3 · Parallelism lessons learned (audit record)

| Attempt | Config | Outcome |
|---|---|---|
| First launch | MAX_JOBS=2 CA MTL | OOM at ~77 GB when fold-1 cross-attn allocates +9 GB |
| Second launch | STL cat 4-way + MTL serial | STL cat ran fine; MTL slowed 3× (22s vs 8s/epoch) due to SM contention |
| Current | STL cat finished; MTL runs solo | 8s/epoch, 49 GB, healthy |
| **Planned** | CA serial → TX 2-way | Expected ~25% speedup on TX block |

Key insight: VRAM contention is not the only risk. Even 4×3 GB `next_gru`
processes saturate SMs enough to 3× the epoch time of a concurrent `mtlnet_crossattn`.

---

## 4 · What to do next (for agent picking up after Gap 1 completes)

1. **Verify all 16 MTL runs landed** — check `logs/h100_gaps_*.log` for exit 0.
2. **Extract CA/TX B9 vs H3-alt numbers** using the same F51 extraction methodology
   (`scripts/analysis/paper_closure_wilcoxon.py` or equivalent).
3. **Update §0.4 recipe-selection table** in `results/RESULTS_TABLE.md` — add
   CA/TX multi-seed rows to replace the single-seed cells.
4. **Check narrative** — does the CA/TX B9 directional story hold with 4 extra seeds?
   Hard stop if the direction flips (per H100_CAMERA_READY_GAPS_PROMPT.md §11).
5. **Commit docs** and push.

---

## 5 · Files produced this session

| File | Description |
|---|---|
| `scripts/run_h100_camera_ready_gaps.sh` | Launcher (CA serial + STL cat 4-way pool) |
| `results/check2hgi/alabama/next_lr*_175846_*` | 4 AL STL cat runs (seeds 0,1,7,100) |
| `results/check2hgi/arizona/next_lr*_175847_*` | 4 AZ STL cat runs |
| `results/check2hgi/florida/next_lr*_175852-3_*` | 4 FL STL cat runs |
| `output/check2hgi/california/region_transition_log_seed{0,1,7,100}_fold*.pt` | CA per-fold log_T |
| `output/check2hgi/texas/region_transition_log_seed{0,1,7,100}_fold*.pt` | TX per-fold log_T |
| *(pending)* `results/check2hgi/california/mtlnet_*` | CA MTL runs × 8 |
| *(pending)* `results/check2hgi/texas/mtlnet_*` | TX MTL runs × 8 |
