# Gated-overlap board prep — consolidated findings (2026-06-20/21)

> Single landing page for the overlap/gate/perf work done while preparing the gated stride-1 board.
> Detailed docs: `WINDOWING_AUDIT.md` (gate), `OOM_MEMORY_FIX.md` (memory/Q1), `SPEED_LEVERS.md` (compile/Q3),
> `DEFAULTS_AND_GUARDS.md` (board values + guards). All code is on PR #29 (byte-identical + perf).

## The board windowing (now the enforced default)
**stride-1 OVERLAP, GATED (`emit_tail=False`), MIN_SEQ=10.** Built via
`build_overlap_probe_engine.py <state> 1` (auto-gates at stride==1, defaults min_seq=10). The non-overlap
frozen v11/v14 substrate stays at the global `core.py` MIN_SEQ=5 (§0.1 reproduction — never flip globally).
A train-time guard (`folds._warn_if_ungated_overlap`) WARNs / `MTL_STRICT=1` fails if an overlap engine is
ungated or min_seq≠10 — so a stale ungated build can't silently train (it bit us once: AL left ungated →
phantom −2.5pp, see `ref_ovl_emit_tail_stale`).

## M1 — the tail-gate (why gated)
At stride-1, ~window_size OOB tail windows/user all target the user's LAST POI on near-all-padding
histories — a label-distribution skew (NOT a leak; orthogonal to `STRIDE1_LEAK_REAUDIT`). The gate drops them.
Footprint **shrinks with state size**: AL 10.9% of windows / 15.1% last-POI skew → TX 5.2% / 7.5%. So the
gate helps most at small states and is ~neutral at huge (can't hurt). StratifiedGroupKFold does NOT mitigate
the skew (it balances category across folds, not the pooled target distribution).

## Q1 — CA/TX dataset-on-GPU (CLOSED)
CA/TX overlap MTL is viable on the A40 via the **default auto-fit** (large-state dataset kept CPU-resident,
TX GPU ~6 GB, ~160 s/epoch). **Never set `MTL_DATASET_GPU=1` for CA/TX** — it forces ~31 GB of redundant
per-fold copies onto the GPU → OOM (no CPU fallback). The host-RAM OOM (eager 5-fold + dead `FoldData.x`,
~126 GB) was fixed by lazy per-fold construction + dropping `FoldData.x` (byte-identical). Full TX MTL ≈ 11 h
(intrinsic to 8.5× overlap). See `OOM_MEMORY_FIX.md`.

## Q2 — does the champion thesis hold under the gated board windowing?
**FL board-grade (gated, min_seq=10, seed-42, 5-fold, matched-metric):**

| task | champion-G MTL | STL ceiling | Δ |
|---|---|---|---|
| cat macro-F1 | 78.32 ± 0.93 | 75.20 ± 0.76 | **+3.12** ✅ beats (historical +2.6…+4.1) |
| reg top10 (full) | 75.52 ± 1.07 | 76.64 | **−1.12** ⚠ |

- **cat clearly holds** (+3.12, in range). Absolutes are higher than non-overlap (overlap lifts both heads).
- **reg is a FLAG — and it's SYSTEMATIC (2026-06-21, seed-0 confirm):** historical (non-overlap) reg Δ was
  −0.35 ("matches"). Under gated overlap the reg Δ is **−1.12 (seed-42) and −1.21 (seed-0)** — *consistent
  across two seeds*, so NOT a seed artifact. **The gated overlap windowing widens the reg-match gap to a
  systematic ~−1.2 pp** (vs −0.35 non-overlap). Mechanism: overlap is a rising tide that lifts the **STL reg
  ceiling MORE than the MTL reg** (STL reg 76.6-76.7 vs MTL reg ~75.5). cat Δ is +3.12/+3.84 (beats, even
  stronger). **Implication: the central "MTL matches the reg ceiling" claim WEAKENS under the board windowing**
  (from "−0.35 matches" to "~−1.2, at the edge of matches"); cat strengthens. The full board T3 (multi-seed
  fold-paired Wilcoxon) formally settles matches-vs-below, but 2 seeds already say the effect is real. **This
  is the open paper-claim-relevant decision: is the gated-overlap reg cost (~1.2pp gap, but higher absolutes)
  acceptable for the board?** → user call (held).
- Wall times: uncompiled (seed-42) STL-cat 23m50 / STL-reg 39m41 / MTL 3h11m, total 4h15. **Compiled board
  path (seed-0, dynamic + shared-cache reuse): STL-cat 24m58 (~0%) / STL-reg 29m54 (~25%) / MTL 2h31m (~21%),
  total ~3h26 — ~19% faster, 0 warmup, quality-neutral.** Confirms Q3 positive at full 50-ep scale.

## Q3 — compile/tf32 (CLOSED)
- **Quality: neutral** (STL −0.05/+0.02; AL flat; SPEED_LEVERS A/B +0.05 pp). compile/tf32 = fp-ordering noise.
- **The "32-min compiled warmup" was a MISDIAGNOSIS** — it was torch.compile exceeding
  `cache_size_limit=8` → silent **eager fallback** (un-compiled, slow). Fixed by three levers:
  1. `MTL_COMPILE_DYNAMIC=1` (one symbolic-shape graph), 2. shared persistent `TORCHINDUCTOR_CACHE_DIR`,
  3. `cache_size_limit` 8→64 (default-on when compiling).
- **Measured (FL MTL fold-1, 8ep):** uncompiled 366 s → compile+dynamic fresh-cache 372 s (13 recompiles,
  break-even) → **cache-reuse 318 s, 0 recompiles (~13% faster, 0 warmup)**. The `requires_grad` (train↔eval)
  recompile is inherent but one-time (13 → 0 on reuse).
- **Board compiled recipe:** every compiled cell runs `--compile --tf32 MTL_COMPILE_DYNAMIC=1` + ONE shared
  `TORCHINDUCTOR_CACHE_DIR`. First cell ≈ break-even; all later cells ~13–15% faster, quality-neutral.
  Apply UNIFORMLY (p1 STL-reg ceiling supports `--compile`/`--tf32` so the comparison stays fair — never mix).

## DECISION (2026-06-21) — ADOPTED, see `BOARD_ADOPTION_DECISION.md`
All three board changes are **ADOPTED** (user call + critical advisor review = **GO-WITH-CONDITIONS**):
gated overlap base, compile path (`--compile --tf32 MTL_COMPILE_DYNAMIC=1` + shared cache), CA/TX-on-A40.

**⚠ Reframe the reg "loss" — it is NOT a claim collapse.** The paper defines "matches/Pareto-non-inferior" as
**TOST non-inferiority at δ=2 pp** (`STATISTICAL_AUDIT §0.3`), not Δ≈0. **A −1.2 pp gap (σ~1.0) still PASSES
the δ=2 pp test** → "single model non-inferior on reg (within 2pp) AND beats cat +3" survives gated overlap.
What's lost is rhetorical margin (−0.31 "visibly ties" → −1.2 "non-inferior within 2pp"). Write the reg side
as explicit TOST non-inferiority, not "ties."

**Advisor condition before the full board:** run **one TX (large-state) gated-overlap reg cell, 1 seed** — the
board commits 6 states × 4 seeds off FL-2-seed, and the mechanism warns the reg gap may be WORSE at CA/TX
(more regions, untested). Rule: |Δreg| ≤ ~1.5 pp → adopt board-wide; > 2 pp → keep non-overlap. Plus: confirm
AL under the final recipe; pin the windowing-matched ceiling (B-A2 trap); write reg as TOST. Full rationale +
caveats (compile uniformity, CA/TX phrasing, blast radius, composite fallback): `BOARD_ADOPTION_DECISION.md`.

## Open
1. **TX gated-overlap reg de-risk cell** (Condition 1) — the gate before launching the full board.
2. **PR #29** — all byte-identical + perf code (lazy-fold, S2-auto, M1 gate, guards, min_seq default, compile
   levers) + these adoption docs. Awaiting merge.


### reg-gap ATTRIBUTION — CONFIRMED windowing (2026-06-21, disentangler + 4-agent audit)
The reg Δ widening (−0.31 non-overlap → −1.2 gated) is the **GATED OVERLAP WINDOWING**, not optimizer/compile/seed.
Audit (workflow): the only optimizer mismatch is weight_decay (STL 0.01 vs MTL 0.05; max_lr matched at 3e-3),
and it's CONSTANT across windowings (present in both −0.31 and −1.2). Disentangler (FL non-overlap, seed-42,
IDENTICAL recipe, uncompiled — windowing the only flip): **Δ non-overlap = −0.28** (≈ historical multi-seed
−0.31), vs **Δ gated = −1.12**. Ruled out 3 ways: compile (gated −1.12 was UNCOMPILED; compiled reg 75.50 ≈
uncompiled 75.52), optimizer (wd constant), seed (s42 non-ovl −0.28 ≈ multi-seed −0.31; gated reproduces s0
−1.21). Mechanism: overlap lifts the STL standalone reg ceiling (+3.4) MORE than the joint MTL reg tower
(+2.5) → the matched gap widens ~0.84pp. **TRUSTWORTHY.** Net: under the board windowing cat beats +3.1/+3.8,
reg slips from "matches" to ~−1.2 below ceiling — the open board decision is whether that reg cost is acceptable.
