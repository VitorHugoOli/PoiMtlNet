# mtl_frontier — STATE

**Status:** FIRST WAVE + R10 + follow-ups + conditional coupling + **R-CC+** DONE (2026-06-17). **R4–R9 OPEN.**
· **Machine:** A40 · **Created:** 2026-06-14 · **Branch:** `study/mtl-frontier`
**Onboarding:** [`AGENT_PROMPT.md`](AGENT_PROMPT.md) · **Mechanism narrative + every number:** [`FINDINGS.md`](FINDINGS.md)
**Continue the study (R4–R9):** [`HANDOFF.md`](HANDOFF.md) ⭐ · **Family DAG:** [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md)

## Headline (2026-06-17)
**Champion G is UNCHANGED — no v17 promotion.** 9 lever-families tested: **8 nulls + 1 genuine
sub-threshold positive** (conditional coupling: FL cat **+0.235** / reg +0.070, audit-confirmed
deterministic, below the 0.3 gate). **R-CC+ (2026-06-17) fully mapped the conditional-coupling family
along signal × injection × output-side axes — no variant exceeds the original additive-posterior cc;
the +0.235 cap is the regime (weak 2.8-bit auxiliary), not a fixable knob.** The cos≈0 / data-rich / weak-7-class-auxiliary regime caps every mechanism. The one
direction that produced real (capped) transfer is **input-side conditional coupling** (cat output → reg
input feature, iMTL/GETNext) — the recommended R4+ direction is to push *that* family, not more
output-prior / sharing-gate variants. Audit (user-flagged "seed-0 always best"): champion G is fully
DETERMINISTIC; the seed-0 pattern is real seed variance (G seed-0 is the weakest seed), not a bug.

## Level / blocking
- Level 0 (exploration). Blocks: `closing_data` P2 FREEZE (a promoted lever → v17 → re-pin before freeze).
- Runs in parallel with `second_dataset` (Mac) and `closing_data` P1a (reading).
- **Nothing flows to `closing_data` G0.2** (no promoted lever). The aligned-pairing pre-freeze gate (G0.1)
  is a separate `mtl_improvement` inheritance, untouched here.

## First-wave queue
| ID | Lever | State | Verdict |
|---|---|---|---|
| R1 | log_C co-location prior + probability-chain | **CLOSED 2026-06-15** | **NULL** — real but sub-threshold (AL multi-seed Δreg **+0.207±0.196**, p=0.008 15/20 pairs, gate ≥0.3 FAIL; FL seed0 +0.171 / cat −0.27; W non-monotonic, peaks at 0.2). Not v17. See `FINDINGS.md §R1`. |
| R2 | STEM-AFTB gating sweep | **CLOSED 2026-06-15** | **NULL** (not v17) — reg clean multi-seed null at all states; cat lift is AL-only & **decays with scale** (AL +0.636 / AZ +0.173 / GE +0.158 / **FL −0.026**); user-approved multi-state confirm → does NOT generalize. Citable STEM-AFTB dose-response. See `FINDINGS.md §R2`. |
| R3 | live cross-task distillation | **CLOSED 2026-06-15** | **NULL** — CrossDistil (warm-up+error-correct+reverse) doesn't beat G+log_T-KD. Fwd refinements don't rescue R1 (AL cat −0.18); reverse reg→cat AL cat +0.45 seed0 → multi-seed **+0.100±0.282** (p=0.31, one seed −0.34); FL null. log_T-KD saturates the output-prior family. See `FINDINGS.md §R3`. |

| **R10** ★ | Memory-Caching / GRM gated read | **CLOSED 2026-06-15** | **NULL** — learned input-conditioned generalization of R2's binary AFTB. FL cat +0.324 seed0 → multi-seed **+0.085±0.203** (2 seeds neg) = noise. Trained-γ swings 0.87→0.31 yet ≡ G → "nothing to gate" (cos≈0). SSC not pursued. `--model-param crossattn_grm=True`. §R10. |

**Follow-ups (user ideas, advisor-structured) — CLOSED 2026-06-15, three more nulls:** (1) R10 GRM at
AZ/GE seed0 + AL multi-seed → null everywhere; (2) **aux_gated** (input-dependent β in reg head): gate
moves (γ 0.12→0.47) but AL cat +0.48 / **FL cat −0.85 crater** → re-confirms champion β→0 / aux>gated;
(3) **best-stack** → SUB-additive (AL cat +0.456 < best component aftb_late +0.636). §Follow-up.

**★ CONDITIONAL COUPLING (cat output → reg input feature, iMTL/GETNext) — CLOSED 2026-06-16: the FIRST
genuine positive (sub-threshold).** `cond_coupling=posterior` (feed `softmax(cat_logits)` as a reg input
feature, train+inference). FL cat **+0.235** (4/4 seeds positive [+0.45,+0.07,+0.21,+0.21]) + reg
**+0.070** — audit-confirmed deterministic, matched-baseline, NOT an artifact. **Below the 0.3 v17 gate**
but the study's only real Pareto-positive. Richer 256-dim `features` variant HURTS FL (−0.31) → cap is
the regime (weak 7-class prior), not a fixable HP. Code: `cond_coupling`/`cond_detach` (G unchanged). §Conditional coupling.

**AUDIT (user-flagged "seed-0 always best") — RESOLVED 2026-06-16:** champion G is fully DETERMINISTIC
(2 distinct runs/seed bit-identical). The earlier "non-determinism 0.38/0.11" was **code drift** vs the
old `mtl_improvement` R0 numbers (different code state) — within mtl_frontier all comparisons are valid.
seed-0 pattern = real seed variance (G seed-0 genuinely weakest). Reused baselines were correct. **No
bug; nulls unchanged; conditional coupling confirmed real.** §AUDIT. **Lesson: run matched same-batch
baselines (harmless here only because G is deterministic); don't compare absolute numbers across the
mtl_improvement↔current code drift (~0.1–0.4 cat).**

**★ R-CC+ (conditional-coupling family extension) — CLOSED 2026-06-17: NULL (the cc cap is the regime,
not the injection knob).** Pushed cc along 3 orthogonal axes — signal {calibrated-τ, discrete-argmax,
top-k}, injection {FiLM, input-side concat-into-sequence}, output-side {learned cat→region logit prior}.
**Every variant ties or underperforms the original additive-posterior `cc_e2e`**; FL multi-seed: calib
+0.237 / argmax +0.214 / **cc_e2e +0.235** (all 4/4 seeds +, reg +0.066…+0.070 p<0.05) / **concat +0.033
washes out** (2 seeds neg) / logitp +0.016 null; richer 256-dim `features` already shown to HURT (−0.31).
**Nothing clears 0.3** → the sub-threshold bound is the **regime** (data-rich main + weak 7-class ~2.8-bit
auxiliary), not a fixable HP. Independent advisor audit: code correct, leak-free, G bit-identical (5/5
claims PASS). New code `cond_signal`/`cond_temp`/`cond_topk`/`cond_inject`/`cond_logit_prior` (G unchanged).
No v17. See `FINDINGS.md §R-CC+`.

**★ R4 (Pareto-front profiling, PaLoRA-style) — DONE 2026-06-17 (paper-narrative; resolves C21).** Profiled
the cat↔reg front on the FROZEN champion G (FL multi-seed): **(1) the scalarization/mixture-weight axis is
a near-corner** — champion cw=0.75 is Pareto-dominant (lowering cw 0.75→0.55 buys +0.05pp reg for −0.87pp
cat; raising to 0.85 loses on both), tasks weakly coupled (the falsifier → publishable regime datum);
**(2) the deployment-epoch axis carries the real, STABLE trade** — 12–16 Pareto epochs/run, geom_simple
deploys a consistent ep18–20 point every seed. **C21 resolved**: the selector saga is an epoch-deployment
choice on a real, well-localised epoch-front, not a weight/architecture problem. **PaLoRA-proper NOT built**
(justified: shared-trunk adapter mixture can't move the private-tower reg → would reproduce the
near-collapsed weight-front). No code changes; champion G untouched; paper-narrative (no promote gate).
See `FINDINGS.md §R4`.

**STUDY STATUS (2026-06-17): 8 NULLS + 1 sub-threshold positive + R4 (paper-narrative front).** No v17
promotion; champion G stands. **R5/R9/R6/R7/R8 OPEN** (running autonomously per user: stop only on a
≥0.3 multi-seed promote or a blocker). See `FINDINGS.md §SYNTHESIS` + `§R-CC+` + `§R4`; `HANDOFF.md`.

## Promote-gate convention
≥0.3 pp either head, multi-seed {0,1,7,100} → STOP for user (recipe → v17) → register in `closing_data` G0.2.
Null → log here + `../log.md` row; do not silently fold into the freeze.

## Decisions log
- 2026-06-14 — scaffolded from `docs/research/mtl_frontier.md` §4 (R1–R9). Optimizer aisle declared closed
  (19-arm null + Kurin/Xin/Mueller); only R9 residual sanity arms remain citable-cheap.
- 2026-06-15 — **R1 launched + CLOSED NULL.** Built train-only per-fold/seed `P(region|cat)`
  (`compute_region_colocation.py`) + ESMM KD coupling on top of log_T-KD (`--log-c-kd-weight`,
  default off; G unchanged). Screen AL+FL seed0 → AL +0.331 (promote-eligible) / FL +0.171 null →
  multi-seed AL {0,1,7,100} = **+0.207±0.196 (gate ≥0.3 FAIL)**, Wilcoxon p=0.008; weight sweep
  non-monotonic (peaks W=0.2, craters at 0.6). Real-but-small incremental signal over log_T-KD;
  weak-7-class-auxiliary + spatial overlap with log_T. Proceeding to **R2 (STEM-AFTB gating sweep)**.
- 2026-06-15 — **R2 launched + CLOSED NULL.** Built directional per-layer AFTB gates
  (`detach_ab`/`detach_ba` + `aftb_spec`, champion G unchanged). AL seed0 → all 5 configs cross gate;
  AL multi-seed → reg null (best +0.173 p=0.009 sub-threshold), cat AL-only high-var lift. User-approved
  multi-state confirm (aftb_late, AZ/GE seed0 + FL {1,7,100}): **cat decays with scale, AL-only**
  (AL +0.64 / AZ +0.17 / GE +0.16 / FL −0.03) → does NOT generalize → NOT v17. Inverse-G′. Citable
  STEM-AFTB dose-response (cross-task gradient is small-state harmful noise; reg unaffected — sharing
  topology doesn't move the reg gap). Proceeding to **R3 (live cross-task distillation)**.
- 2026-06-15 — **R3 launched + CLOSED NULL.** Built CrossDistil: warm-up gating + error-correction on
  the fwd cat→reg co-loc KD + a new reverse reg→cat arm (`log_C_rev`=P(cat\|region), `--cat-kd-weight`).
  Screen vs G+log_T-KD: fwd refinements don't rescue R1 (AL cat −0.18); reverse AL cat +0.45 seed0 →
  multi-seed +0.100±0.282 (p=0.31, seed1 −0.34) → noise; FL null. log_T-KD saturates the output-prior
  family. **First wave (R1/R2/R3) = three nulls, all the same regime** (small-state-only, FL-null,
  reg-immovable). User decision "full R3 then R10" → proceeding to **R10 (GRM/SSC gated read)**.
- 2026-06-15 — **R10 launched + CLOSED NULL.** GRM-gated cross-attn read; FL seed0 +0.324 → multi-seed
  +0.085±0.203 noise; trained-γ diagnostic (0.87→0.31, ≡ G) proves the gate is live but cos≈0 gives
  nothing to gate. Independent audit (16-property) + efficiency: G+GRM = +5.6% params, ~equal speed,
  same accuracy → G dominates.
- 2026-06-15 — **Follow-ups (3 user ideas) CLOSED, all null.** R10-other-states (null everywhere),
  aux_gated (input-dependent β: FL cat −0.85 crater, re-confirms champion aux>gated), best-stack
  (sub-additive). Advisor recommended **conditional coupling** as the higher-merit next bet.
- 2026-06-16 — **CONDITIONAL COUPLING launched + CLOSED: first genuine positive (sub-threshold).**
  cat posterior → reg input feature (iMTL). FL cat +0.235 / reg +0.070, 4/4 seeds positive. Richer
  256-dim features HURT FL. Below 0.3 gate; no v17 promotion.
- 2026-06-16 — **AUDIT (user-flagged seed-0 pattern) RESOLVED.** Champion G deterministic; seed-0 is
  the genuinely-weakest seed; earlier "non-determinism" was code drift vs old R0; cc result confirmed
  real. Methodology lesson recorded. **R4–R9 handed off via `HANDOFF.md`.**
- 2026-06-17 — **R-CC+ launched + CLOSED NULL (cc family fully mapped).** Implemented 3 orthogonal cc
  axes (champion G bit-identical, zero-init ≡ G, smoke-verified + advisor-audited 5/5 PASS, no GT leak):
  `cond_signal`{softmax,calibrated,argmax,topk} · `cond_inject`{add,film,concat_seq,none} ·
  `cond_logit_prior`. Seed-0 screen (AL+FL, fresh matched G): FL additive family (e2e/calib/argmax/topk)
  ties +0.42…+0.45; FiLM +0.243 & input-side concat +0.259 worse; output-side logitp +0.016 null.
  FL multi-seed {0,1,7,100} (matched in-batch G): cc_e2e **+0.235** (reproduced exactly → determinism +
  no-drift reconfirmed), calib +0.237, argmax +0.214 (all 4/4 seeds +, reg +0.066…0.070 p<0.05),
  **concat washes out +0.033** (2 seeds neg, advisor-requested input-side control → confirms additive is
  the family optimum, not init-confound). **None clears 0.3** → sub-threshold cap = regime (weak 2.8-bit
  auxiliary), not a knob. richer `features` already HURT (prior cc work). No v17; champion G stands.
  **Pausing here for R4–R9 realignment per user.** Untested future lever: cross-attn cat↔reg coupling
  (advisor idea; expected sub-threshold). See `FINDINGS.md §R-CC+`.
- 2026-06-17 — **R4 (Pareto-front profiling) DONE — paper-narrative, resolves C21.** Scalarization
  `--category-weight` sweep on frozen champion G (FL, multi-seed {0,1,7,100}): weight-front is a
  **near-corner** (champion cw=0.75 Pareto-dominant; lowering→0.55 = +0.05pp reg/−0.87pp cat; raising→0.85
  dominated) → tasks weakly coupled on the loss-weight axis (falsifier datum). Champion **epoch-trajectory
  front** is the real, **stable** C21 locus (12–16 Pareto epochs, geom_simple ep18–20 every seed).
  **PaLoRA-proper declined** (mechanistic: shared-trunk adapter mixture can't move the private-tower reg
  → reproduces the near-collapsed weight-front). No model code; champion untouched. → R5.
