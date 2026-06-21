# HANDOFF — board launch · A40 · A100 · M2 Pro (2026-06-21)

> The pre-freeze gates are all closed; we are entering **P2 (freeze) → P3 (board)**. This is the turn-key
> run-spec for the three things to kick off in parallel NOW. **Read first:**
> [`../EXECUTION_PLAN.md §13`](../EXECUTION_PLAN.md) (sequence + the device rule),
> [`../pre_freeze_gates/OVERLAP_BOARD_FINDINGS.md`](../pre_freeze_gates/OVERLAP_BOARD_FINDINGS.md) +
> [`../pre_freeze_gates/DEFAULTS_AND_GUARDS.md`](../pre_freeze_gates/DEFAULTS_AND_GUARDS.md) (the board recipe +
> guards), [`STATISTICAL_PROTOCOL.md`](STATISTICAL_PROTOCOL.md) + [`RUN_MATRIX.md §2.5`](RUN_MATRIX.md).

## Governing rule (non-negotiable)
**Every PAIRED comparison runs end-to-end on ONE device-class.** Partition the board **BY STATE** across
{A40, A100, M2 Pro}; a state's full cell set (MTL + STL ceilings + its baselines) stays on ONE device. Per-state
Δ's are then device-internal and clean; the absolute cross-state table carries a device-class footnote. **Never
split one comparison across device-classes** (MPS fp32 vs CUDA tf32+compile differ ≫ the ±0.05 pp effect size).

## Process — each machine: OWN branch · incremental commits · OWN PR
**Each of the three machines works on its OWN branch and opens its OWN PR** — so each lane is independently
auditable and gets its own further instructions (same pattern as PR #26–#29). Do **NOT** share a branch (avoids
conflicts), do **NOT** commit to `main`.
- **A40** → branch `study/board-a40`
- **A100** → branch `study/board-a100`
- **M2 Pro** → branch `study/board-m2pro`

**Commit INCREMENTALLY** — per cell / per built artifact / per verdict (a small, frequent commit carrying the
result JSON + a one-line finding), never one giant end-of-run commit; this makes progress auditable mid-flight.
**Open the PR early (draft) and push as you go**; when the A/B, a de-risk cell, or an embedding-build batch
completes, flag it for audit. The orchestrator audits each PR, gives further instructions, and merges/reconciles
after — the machines do not merge each other or `main`.

## Shared CUDA board recipe (A40 + A100)
- Base = **gated stride-1 overlap, MIN_SEQ=10** engine `check2hgi_dk_ovl`, built per state via
  `python scripts/mtl_improvement/build_overlap_probe_engine.py <state> 1` (auto-gates at stride==1, min_seq=10).
- Recipe = champion **G / v16** (the `docs/NORTH_STAR.md` invocation) on `--engine check2hgi_dk_ovl`, geom_simple
  selector, matched scorer (FULL top10_acc, fp32-eval), per-fold per-seed **train-only** log_T.
- **Compile path (uniform, mandatory):** `--compile --tf32` + `MTL_COMPILE_DYNAMIC=1` + ONE shared
  `TORCHINDUCTOR_CACHE_DIR=<persistent path>`. Apply to **MTL and the STL ceilings alike** (the p1 STL-reg
  ceiling supports `--compile/--tf32`) — mixing compiled/uncompiled confounds every paired Δ at ±0.05 pp.
- **Memory:** dataset **auto-fit** (default) — **NEVER set `MTL_DATASET_GPU=1` for CA/TX** (forces ~31 GB
  redundant copies → OOM). The lazy-fold + streaming fixes (PR #29) are in place.
- **Preflight every `--per-fold-transition-dir` run:** stale-log_T freshness check (`src/data/log_t_freshness.py`).
- Drivers: `scripts/closing_data/p3_board.sh`, `scripts/pre_freeze_gates/gated_overlap_g.sh`. torch pinned 2.11.0+cu128.
- Provenance (C28): PID-suffixed rundirs, per-run seed echo, **commit the MTL + STL + baseline result JSONs**
  (closes audit F3-3 — the MTL headline artifacts were missing).
- **Per-machine branch + incremental commits + own PR** (see Process above); do **not** commit to `main`.

---

## A40
1. **A100-equivalence A/B (A40 half) — PRIORITY.** Run ONE FL champion-G cell on the overlap engine, **seed 0,
   5-fold, compiled+tf32**; record `cat macro-F1` + `reg top10_acc` to **4 dp**. The A100 runs the byte-identical
   command. Pass = |Δ| ≤ **±0.05 pp** on both heads → by-state parallelization across the two CUDA cards is clean.
2. **Early TX gated-overlap reg cell (insurance, not a stop-gate).** Build the TX overlap engine
   (`build_overlap_probe_engine.py texas 1`), then the **matched B-A2 reg**: STL `next_stan_flow` reg ceiling
   (overlap, compiled) + champion-G **MTL** reg (overlap, compiled, auto-fit), **1 seed × 5 folds**. Report
   **Δreg = MTL FULL top10 − STL ceiling** vs **δ_reg = 2 pp**. (~11 h for TX MTL; freshness preflight first;
   assert the STL ceiling is itself on overlap — the B-A2 trap.)
3. **Post-freeze:** A40 owns its assigned states for the full board (by-state partition).

## A100
1. **A100-equivalence A/B (A100 half).** Run the **byte-identical** FL command as the A40; record to 4 dp;
   confirm |Δ| ≤ ±0.05 pp **before** splitting the board.
2. **Early CA gated-overlap reg cell.** CA is the largest (8501 regions) → the **most at-risk** for the δ_reg=2 pp
   margin. Build CA overlap engine; matched B-A2 reg, 1 seed × 5 folds; report Δreg vs 2 pp.
3. **Post-freeze:** A100 owns the heavy states (per the partition) + the heavy end-to-end baselines.

## M2 Pro (MPS)
**Build the LIGHT substrate-column baseline EMBEDDINGS** on the gated-overlap base — device-tolerant *inputs* the
CUDA board's matched-head cells will consume (the actual comparison runs on CUDA, so MPS-built embeddings are
safe: small fp differences are absorbed, and the comparison is head-level on one device). For each board state ×
seed {0,1,7,100} × 5 folds, **train-only per fold, on the gated-overlap windowing** (so they match the frozen base):
- CTLE — `python scripts/baselines/build_ctle_substrate.py …`
- POI2Vec (faithful AAAI'17) — `scripts/baselines/build_poi2vec_substrate.py …`
- skip-gram (B2b) — `scripts/baselines/build_b2b_skipgram_substrate.py …`
- one-hot64 (B2c) — `scripts/baselines/build_b2c_onehot64_substrate.py …`
MPS fp32, no compile (these are builds). **Do NOT run the matched-head COMPARISON cells on MPS** — those go on
CUDA with the state's STL/MTL (device-class rule). *Optional:* if 32 GB + wall-time allow, own whole small states
AL/AZ (MTL+STL+baselines all on MPS) — confirm AL/AZ overlap-MTL fits memory first.

---

## STOP conditions
- **A100-equiv |Δ| > ±0.05 pp** → by-state partition is mandatory; cross-GPU absolutes carry a caveat. STOP for user.
- **TX/CA Δreg > 2 pp** → NOT a stop (overlap is adopted) but **flag for the reg-claim framing** — the paper
  re-scopes the reg claim honestly per `EXECUTION_PLAN §12/§13` (non-inferior where it holds; the composite-reg
  panel is a supportive fallback). Record + report.
- Any freshness-preflight / OOM / `_warn_if_ungated_overlap` (MTL_STRICT) guard failure → STOP.

## Then → P2 FREEZE → P3 board
Once the A100-equiv is clean, the early TX/CA cells are recorded, and the `RUN_MATRIX` is signed: **P2 freeze**
(recipe v16 · v14 6-state hashed · gated-overlap-MIN10 windowing · label-space · signed RUN_MATRIX) → **P3 board**
(all states × {0,1,7,100} × 5 folds, partitioned by state, CUDA cells compiled, artifacts committed) → stats per
`STATISTICAL_PROTOCOL.md` (paired Wilcoxon superiority + TOST δ_reg=2 pp non-inferiority + Holm) → L4 Phase V.
