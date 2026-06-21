# Board adoption decision (2026-06-21) — gated overlap + compile + CA/TX-on-A40

> User decision + critical advisor review. Status of the three board changes, with the advisor's conditions.
> Grounding: `OVERLAP_BOARD_FINDINGS.md`, `SPEED_LEVERS.md`, `OOM_MEMORY_FIX.md`, NORTH_STAR, `articles/[BRACIS]_*/STATISTICAL_AUDIT.md`.

## Decisions
| change | status | notes |
|---|---|---|
| **Compile board path** (`--compile --tf32 MTL_COMPILE_DYNAMIC=1` + shared `TORCHINDUCTOR_CACHE_DIR`) | **ADOPTED** | quality-neutral, ~21% faster MTL. Apply UNIFORMLY (correctness, not preference). |
| **CA/TX trainable on the A40** (auto-fit; never `MTL_DATASET_GPU=1`) | **ADOPTED (report with caveats)** | ~11–13 GB GPU peak, dataset CPU-resident, ~11 h/TX run. |
| **Gated stride-1 overlap windowing** (min_seq=10) as board base | **ADOPTED — pending one large-state de-risk cell** | improves absolutes + cat; reg-match cost is the gate (see below). |

## Advisor verdict: **GO-WITH-CONDITIONS**
Compile + CA/TX: **GO now** (clean evidence; caveats are operational, not scientific). Gated overlap: **GO after one TX (large-state) gated-overlap reg cell** — the whole 6-state × 4-seed board is being committed off **FL-only, 2-seed** evidence on the **headline-bearing reg axis**, and the mechanism ("architectural cost grows with region cardinality") warns the gap could be WORSE at CA/TX (4703+ regions), which are **untested under overlap** and the most expensive to redo.

## ⚠ The framing correction that matters (don't over-state the reg "loss")
The session framed reg as "matches −0.31 → weakens to ~−1.2 below" — that reads like a claim collapse. **It is NOT.** The paper defines "matches/Pareto-non-inferior" via **TOST non-inferiority at δ = 2 pp** (`STATISTICAL_AUDIT §0.3`), not Δ≈0. **A −1.2 pp gap with σ~1.0 still PASSES the δ=2 pp non-inferiority test.** So under the paper's own machinery, "a single joint model is **non-inferior on region (within 2 pp)** and **beats category by +3**" survives gated overlap. What is lost is rhetorical comfort margin (−0.31 "visibly ties" → −1.2 "non-inferior within 2pp"), not the claim. **Write the reg side as explicit TOST non-inferiority from the start, not as "ties."**

Net: adopting overlap is the MORE defensible position — it removes the "you under-fed the STL baseline" attack (dense data feeds everyone) and strengthens cat + all absolutes, at the cost of a wider-but-still-non-inferior matched reg gap.

## Conditions before launching the full board (the advisor's de-risk gate)
1. **One LARGE untested state under gated overlap, 1 seed, 5-fold (TX, or CA).** Compute the matched B-A2 reg gap (G-MTL FULL top10 vs STL `next_stan_flow` ceiling, **both on overlap windowing**). **Decision rule:** |Δreg| ≤ ~1.5 pp (comfortably inside δ=2 pp TOST) → adopt overlap board-wide. > 2 pp → DO NOT adopt overlap; keep the non-overlap base (paper-grade at −0.09…−0.31). **This single run is the gate.**
2. **One SMALL state confirm (AL/AZ) under the FINAL board recipe** (champion-G v16, gated overlap) — R1 showed AL −0.34 but on the v14/seed-42 mechanism harness, not the committed config.
3. **Pin the windowing-matched ceiling in the run-spec** — assert the STL reg ceiling used in ANY overlap comparison was itself built on overlap windowing; fail loudly otherwise (this is the B-A2 trap reincarnated).
4. **Write the reg claim as TOST non-inferiority (δ=2 pp) from the start.**

## Operational caveats to enforce (not block on)
- **Compile uniformity is a CORRECTNESS requirement** — compile is fp-noise of the same magnitude (±0.05 pp) as the measured effects; mixing compiled/uncompiled cells confounds every paired Δ. Run the WHOLE board compiled; p1 STL-reg supports `--compile/--tf32`. Pin torch 2.11.0+cu128.
- **Shared cache is a perf dependency** (not correctness) — a wiped/per-node `TORCHINDUCTOR_CACHE_DIR` gives fresh-cache break-even, not ~20%. Pin one persistent path.
- **CA/TX claim phrasing:** "trainable on a single A40 (≈11–13 GB peak via CPU-resident dataset auto-fit; ~11 h/run for TX), provided `MTL_DATASET_GPU=1` is never set and the lazy-fold host-RAM fix is in place." Auto-fit is occupancy-dependent (run on a clean GPU).
- **Blast radius:** adopting overlap re-windows ~7 external E2E baselines + rebuilds all per-fold log_T (`RUN_MATRIX §4`). §0.1 restatement was already required (frozen-recipe board), so overlap doesn't create that obligation — it changes the numbers inside it. The frozen non-overlap v11/v14 substrate is protected (separate overlap engines; `core.py` MIN_SEQ stays 5; train-time guard).
- **Reg fallback (insurance, not plan):** the 2-model composite reg edge collapsed to +0.53 pp at FL non-overlap; under overlap it may re-open some edge → a weaker "two models for parity" framing if CA/TX blows the δ=2 pp margin.

## Bottom line
Adopt **compile** and **CA/TX-on-A40** now. Adopt **gated overlap** too — but run **one TX gated-overlap reg cell first** (Condition 1). The reg claim survives at −1.2 *because the paper defines "matches" as δ=2 pp TOST*; the TX cell answers whether the big states stay inside that 2 pp margin. Cheap insurance (~one large-state run) against baking a claim-level mistake into a multi-day, ~7-baseline board regeneration.
