# HANDOFF — board launch INDEX (A40 · A100 · M2 Pro) · 2026-06-21

> The pre-freeze gates are closed; we are entering **P2 (freeze) → P3 (board)**. This is the **index** — the
> shared governing rule, the per-machine process, the launch sequence, and the shared CUDA recipe. **Each machine
> works from its OWN per-machine handoff** (links in §3), and touches ONLY its own work.
>
> **Read alongside:** [`../EXECUTION_PLAN.md §12–§13`](../EXECUTION_PLAN.md) (sequence + device rule),
> [`../pre_freeze_gates/OVERLAP_BOARD_FINDINGS.md`](../pre_freeze_gates/OVERLAP_BOARD_FINDINGS.md) +
> [`../pre_freeze_gates/DEFAULTS_AND_GUARDS.md`](../pre_freeze_gates/DEFAULTS_AND_GUARDS.md) (board recipe +
> guards), [`STATISTICAL_PROTOCOL.md`](STATISTICAL_PROTOCOL.md) + [`RUN_MATRIX.md §0, §2.5`](RUN_MATRIX.md).

## 1 · Governing rule (non-negotiable)
**Every PAIRED comparison runs end-to-end on ONE device-class.** Partition the board **BY STATE** across
{A40, A100, M2 Pro (+M4 Pro)}; a state's full cell set (MTL + STL ceilings + its baselines) stays on ONE device.
Per-state Δ's (the headline: MTL-vs-STL, baseline-vs-ours) are then device-internal and clean; only the absolute
cross-state table carries a device-class footnote (small states MPS-fp32; large states CUDA-compiled-tf32).
**Never split one comparison across device-classes** (MPS fp32 vs CUDA tf32+compile differ ≫ the ±0.05 pp effect
size). The Macs' highest-value role is **building the light baseline embeddings** (device-tolerant inputs the
CUDA board consumes), NOT running the comparison.

## 2 · Process — each machine: OWN branch · incremental commits · OWN PR
Same pattern as PR #26–#29: each machine is an independently-auditable lane on its OWN branch with its OWN PR.
- **A40** → branch `study/board-a40`
- **A100** → branch `study/board-a100`
- **M2 Pro** → branch `study/board-m2pro`

Do **NOT** share a branch (avoids conflicts), do **NOT** commit to `main`, do **NOT** merge another lane.
**Commit INCREMENTALLY** — per cell / per built artifact / per verdict, each carrying the result JSON + a
one-line finding (never one giant end-of-run commit). **Open the PR early (draft) and push as you go**; when an
A/B half, a de-risk cell, or an embedding-build batch completes, flag it for audit. The orchestrator audits each
PR, gives further instructions, and merges/reconciles after — the machines do not merge each other or `main`.

## 3 · The three per-machine handoffs (work ONLY from yours)
| Machine | Branch | Scope (pre-freeze) | Handoff |
|---|---|---|---|
| **A40** (CUDA) | `study/board-a40` | A/B FL half (seed 0) · **EARLY TX** gated-overlap reg cell · then owns its by-state partition | [`HANDOFF_BOARD_A40.md`](HANDOFF_BOARD_A40.md) |
| **A100** (CUDA) | `study/board-a100` | A/B FL half (byte-identical) · **EARLY CA** gated-overlap reg cell (largest, most at-risk) · then owns the heavy states | [`HANDOFF_BOARD_A100.md`](HANDOFF_BOARD_A100.md) |
| **M2 Pro** (MPS) | `study/board-m2pro` | **BUILD the light SC baseline embeddings** (CTLE / POI2Vec / skip-gram / one-hot) on the gated-overlap base, train-only per fold, states × {0,1,7,100} × 5f · optional: own whole small state AL | [`HANDOFF_BOARD_M2PRO.md`](HANDOFF_BOARD_M2PRO.md) |

## 4 · Shared CUDA board recipe (A40 + A100)
The CUDA lanes both run this; the per-machine handoffs carry the verbatim commands + env.
- **Base** = gated stride-1 overlap, MIN_SEQ=10 engine `check2hgi_dk_ovl`, built per state via
  `python scripts/mtl_improvement/build_overlap_probe_engine.py <state> 1` (auto-gates at stride==1, min_seq=10).
- **Recipe** = champion **G / v16** (the [`../../NORTH_STAR.md`](../../NORTH_STAR.md) invocation; the exact command
  is `scripts/pre_freeze_gates/gated_overlap_g.sh` + `--compile --tf32`) on `--engine check2hgi_dk_ovl`,
  `geom_simple` selector, matched scorer (FULL `top10_acc`, fp32-eval, BOTH sides — `r0_matched_rescore.py`),
  per-fold per-seed **train-only** seeded log_T.
- **Compile path (uniform, mandatory):** `--compile --tf32` + `MTL_COMPILE_DYNAMIC=1` + ONE **per-box** persistent
  `TORCHINDUCTOR_CACHE_DIR`. Apply to **MTL and the STL ceilings alike** (the p1 STL-reg ceiling supports
  `--compile/--tf32`) — mixing compiled/uncompiled confounds every paired Δ at ±0.05 pp.
- **Precision (PINNED 2026-06-23, RUN_MATRIX §0):** every MTL cell trains with **`MTL_AUTOCAST_BF16=1`** (bf16
  autocast — NOT the trainer default fp16, which overflows CA/TX's wide reg logits → ep30 NaN collapse; see
  `CA_MTL_DIVERGENCE.md`) and evaluates in fp32 (matched scorer re-forwards fp32; add `MTL_DISABLE_AMP_EVAL=1`
  to any in-trainer val metric). bf16 is the default **pending the A40 equivalence gate** (§5 step 0.5); if it
  fails, fall back to full fp32 (`MTL_DISABLE_AMP=1`) board-wide. STL **reg** ceilings stay as-is (already fp32).
- **Memory:** dataset **auto-fit** (default) — **NEVER `MTL_DATASET_GPU=1` for CA/TX** (forces ~31 GB redundant
  copies → OOM). The lazy-fold + streaming fixes (PR #29) are in place. `MTL_CHUNK_VAL_METRIC=1` at overlap scale.
- **Preflight every `--per-fold-transition-dir` run:** stale-log_T freshness check
  (`src/data/log_t_freshness.py`). **Gate guard `MTL_STRICT=1`** so a stale ungated / min_seq≠10 overlap build
  hard-fails (`folds._warn_if_ungated_overlap`).
- **Pins:** torch **2.11.0+cu128** · the **B-A2 trap** (assert the STL ceiling is on the SAME overlap windowing) ·
  commit the MTL + STL + baseline result JSONs (C28, closes F3-3) · seeds = 1 (seed 0) for the A/B + early cells,
  {0,1,7,100} for the full board · δ_reg = 2 pp decision rule for the early reg cells.

## 5 · Launch sequence (to the freeze and the board)
0.5. **Precision-equivalence gate (A40, bf16-vs-fp32)** — PRECONDITION (RUN_MATRIX §0a). FL champion-G MTL,
   seed 0, 5f, compile+tf32 fixed: Arm X `MTL_AUTOCAST_BF16=1 MTL_DISABLE_AMP_EVAL=1` vs Arm Y
   `MTL_DISABLE_AMP=1`. `|Δcat|,|Δreg| ≤ 0.05 pp` ⇒ standardize **bf16**; else **fp32** board-wide. Re-runs the
   collapsed/at-risk MTL cells (CA all-fold, TX, FL fold-3) under the chosen precision. STOP for user with the table.
1. **A100-equivalence A/B** — one FL cell, A40 vs A100, compiled+tf32, **chosen precision**, seed 0, 5f,
   **byte-identical command** → `|Δ| ≤ ±0.05 pp` on cat macro-F1 + reg top10_acc (4 dp). Gates by-state
   parallelization across the two cards.
2. **EARLY large-state reg cells** (insurance, not a stop-gate): **TX on A40**, **CA on A100** — matched B-A2 reg
   (G-MTL FULL top10 vs STL `next_stan_flow` ceiling, BOTH on overlap), 1 seed × 5f, report **Δreg vs δ_reg=2 pp**.
   **In parallel: the M2 Pro builds the light baseline embeddings.**
3. **P2 FREEZE** (one commit): recipe v16 · substrate v14 6-state (hashed) · windowing = gated stride-1 overlap
   MIN_SEQ=10 · label-space · signed RUN_MATRIX (carrying §2.5 baseline design + STATISTICAL_PROTOCOL).
4. **P3 board** — all states × {0,1,7,100} × 5 folds, partitioned by state across A40/A100/Macs, CUDA cells
   compiled, **committing the MTL + STL + baseline artifacts** (C28). Substrate-column (run 3) + end-to-end
   native (run 4) baselines per RUN_MATRIX §2.5. The ONE sanctioned build+run path is
   `scripts/closing_data/p3_board.sh` (currently behind a launch safety-stop pending 3 P3 infra fixes — see
   DEFAULTS_AND_GUARDS).
5. **Stats** per `STATISTICAL_PROTOCOL.md` (paired Wilcoxon superiority + TOST δ_reg=2 pp non-inferiority + Holm).
6. **L4** second-dataset Phase V (mirror the overlap windowing).

## 6 · Shared STOP conditions
- **A/B `|Δ| > ±0.05 pp`** → by-state partition mandatory; cross-GPU absolutes carry a caveat. STOP for user.
- **TX/CA Δreg > 2 pp** → NOT a stop (overlap is ADOPTED) but **flag for the reg-claim framing** — the paper
  re-scopes the reg claim honestly per EXECUTION_PLAN §12/§13 (non-inferior where it holds; the composite-reg
  panel is the supportive fallback, never the headline). Record + report.
- Any freshness-preflight / OOM / `_warn_if_ungated_overlap` (`MTL_STRICT=1`) / leak-safety guard failure → STOP.
- torch ≠ 2.11.0+cu128 → STOP.
