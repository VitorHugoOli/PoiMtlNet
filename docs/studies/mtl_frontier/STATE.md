# mtl_frontier — STATE

**Status:** **R-PROGRAM COMPLETE (R1–R10 + R-CC+) — 2026-06-17.** WAVE-1 + R10 + follow-ups + cc + R-CC+
+ R4 + R5 + R9 done (executed); R6/R7/R8 closed as reasoned predicted-negatives (deferred, runnable on
request). **No v17 promotion; champion G stands.**
· **Machine:** A40 · **Created:** 2026-06-14 · **Branch:** `study/mtl-frontier`
**Onboarding:** [`AGENT_PROMPT.md`](AGENT_PROMPT.md) · **Mechanism narrative + every number:** [`FINDINGS.md`](FINDINGS.md)
**Continue the study (R4–R9):** [`HANDOFF.md`](HANDOFF.md) ⭐ · **Family DAG:** [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md)

## Headline (2026-06-17)
**Champion G is UNCHANGED — no v17 promotion.** 10 lever-families tested: **9 nulls + 1 genuine
sub-threshold positive** + R4 (paper-narrative front). The positive is conditional coupling: FL cat
**+0.235** / reg +0.070 (audit-confirmed deterministic, below the 0.3 gate). **R-CC+ (2026-06-17) fully
mapped the conditional-coupling family along signal × injection × output-side axes — no variant exceeds the
original additive-posterior cc; the +0.235 cap is the regime (weak 2.8-bit auxiliary), not a fixable knob.**
**R5 (per-instance KD gating)** fired the gate vs its global-W comparand but is a comparand artifact —
worse than the KD-off champion on both heads (NULL). **R4** resolved the C21 selector saga via the
epoch-front. The cos≈0 / data-rich / weak-7-class-auxiliary regime caps every mechanism. The one
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

**★ R5 (per-instance KD gating) — DONE 2026-06-17: NULL (a fired gate that is a comparand artifact).**
Redistribute the log_T-KD weight per check-in by Markov-coverage (covmax/coventr), batch-mean-fixed.
FL multi-seed clears the ≥0.3 cat gate **vs its designed global-W comparand** (covmax +0.472, 4/4 seeds)
— BUT this only recovers part of log_T-KD's own FL cat-cost (log_T-KD(0.2) costs −0.70 cat at FL); vs the
study's TRUE FL champion (KD-OFF G) R5 is **−0.224 cat / −0.063 reg = worse on BOTH heads**, dominated by
the trivial "KD-off at FL". On the KD's target axis (reg) gated ≤ global-W (falsifier met); AL gate fails.
**No v17** (independently advisor-audited: null, do not escalate). Citable mechanism datum: instance-gated
KD recovers ~2/3 of log_T-KD's FL cat-cost but never beats KD-off. Code `--log-t-kd-gate` (G unchanged).
See `FINDINGS.md §R5`.

**★ R9 (BayesAgg-MTL optimizer close-out) — DONE 2026-06-17: optimizer aisle closed citably.** Repo
`bayesagg_mtl` is NOT faithful to ICML'24 (gradient-MAGNITUDE variance, Kendall-style; not the posterior
gradient-DIRECTION uncertainty). Re-run at the CHAMPION recipe it **still craters cat** (AL 37.754 = the
19-arm 37.75 reproduced exactly; FL 63.689) → the crater is a recipe-invariant **impl pathology** (down-
weights the 7-class cat vs 1109-class reg), not a defaults artifact → the 19-arm bayesagg null is now
**diagnosed**. Faithful ICML'24 port unbuilt + expected-null (cos≈0 + 19-arm + Mueller TMLR'25) → deferred.
R9(b) Smooth-Tchebycheff unmotivated (R4 front near-degenerate). No promotion; champion G stands. See
`FINDINGS.md §R9`.

- 2026-06-17 — **AUDIT (user-requested: per-experiment impl+outcome review + the cat↑/reg↓ "are we losing
  something" question) — 18-agent workflow, all 4 cat/reg lenses adversarially verified.** VERDICT: **no
  reg loss, no bug.** (1) Core R-program (R-CC+/R4/R5/R9 + R1/R2/R3/R10/cc) all impl=correct/outcome=sound,
  numbers reproduced to 3dp. (2) cat↑/reg↓ is **selective perception** (cat+/reg+ is the largest of 41
  quadrant-pairs; cat-reg correlation POSITIVE; reg positive 28/41) + a **diagnostic-best epoch-decoupling
  artifact** (R5 reg-down → null at the deployable checkpoint) + the **β-gated regime split** (head_beta→0
  at FL ⇒ no shared reg channel to rob; live β=0.08-0.12 at small states ⇒ both heads move together). (3)
  DECISIVE: **MTL reg matches the same-substrate STL reg ceiling within −0.09…−0.31pp** ⇒ reg is already
  MAXED, not shortchanged. (4) Two follow-up BUGS found & FIXED: FU2 aux_gated ran with shared_stan severed
  (head.py:208) — re-ran fixed (FL now craters BOTH heads −0.91/−0.31, validly re-confirming aux>gated);
  FU1 "AL 4-seed" was seed-0-only (agg seed-column bug) — corrected to +0.045 reg/−0.066 cat. R4 cat
  convention-A vs convB +0.259pp cross-table hazard documented. **Champion G stands; no verdict flipped.**
  See `FINDINGS.md §CAT↑/REG↓` + `§AUDIT`.
- 2026-06-17 — **R6/R7/R8 CLOSED as reasoned predicted-negatives (deferred).** Each is a substantial
  from-scratch impl (fork/merge surgery; weight-space merge framework; 2→3-task refactor) whose outcome is
  mechanistically determined: **R6** ForkMerge ≤ G (scheduled_static already null + R4 near-corner weight-
  front); **R7** merge-vs-joint < G (tangent-space theory — from-scratch experts don't share a basin);
  **R8** time-aux rising-tide null (lifts STL=MTL) + disproportionate 2-task→3-task refactor cost. Closed
  by reasoned prediction per *What NOT to pursue*; **measured runs available on user request**. See
  `FINDINGS.md §R6/R7/R8`. **R-program complete.**

**STUDY STATUS (2026-06-17) — R-PROGRAM COMPLETE: 9 NULLS + 1 sub-threshold positive + R4 (Pareto front,
resolves C21) + R9 (optimizer aisle closed) + R6/R7/R8 (reasoned predicted-negatives, deferred).** No v17
promotion; **champion G stands.** The post-2022 MTL frontier (output-priors, asymmetric/learned sharing,
input-side conditioning, Pareto-profiling, merging, optimizers, auxiliary tasks), brought to this LBSN
regime (cos≈0 + dual-tower + data-rich main + weak 2.8-bit auxiliary), **replicates but does not exceed**
champion G's two wins (dual-tower + log_T-KD) — a strong, citable domain-frontier negative. See
`FINDINGS.md §SYNTHESIS` + `§R-CC+` + `§R4` + `§R5` + `§R9` + `§R6/R7/R8`; `HANDOFF.md`.

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
- 2026-06-17 — **R5 (per-instance KD gating) DONE — NULL (gate fired, comparand artifact).** `--log-t-kd-gate`
  redistributes log_T-KD weight by Markov-coverage (covmax/coventr), batch-mean-fixed (G bit-identical off).
  FL multi-seed covmax +0.472 cat (4/4 seeds) vs the global-W-KD-ON base **clears the gate** — but that base
  is handicapped (log_T-KD costs −0.70 cat at FL); vs the true KD-OFF champion R5 is −0.224 cat/−0.063 reg
  (worse on both, dominated by KD-off-at-FL); reg ≤ global-W (falsifier met); AL gate fails. Advisor-audited
  → null, no escalation. Caught by matched-baseline rigor (read the gate vs the *deployable* champion, not a
  lever's internal control). Incidental: re-quantifies log_T-KD FL cat-harm. → R9.
- 2026-06-17 — **R9 (BayesAgg-MTL) DONE — optimizer aisle closed citably.** Verified repo `bayesagg_mtl`
  is gradient-MAGNITUDE-variance (Kendall-style), NOT faithful ICML'24 (gradient-DIRECTION posterior).
  Re-ran at the champion recipe (not the 19-arm registry defaults): cat **still craters** (AL 37.754 =
  19-arm 37.75 exactly; FL 63.689) → recipe-invariant impl pathology (down-weights 7-class cat vs 1109-
  class reg), 19-arm null now diagnosed. Faithful port unbuilt + expected-null (cos≈0 + Mueller TMLR'25) →
  deferred per "trivial close-out". R9(b) Smooth-Tchebycheff unmotivated (R4 front near-degenerate). No
  code changes; champion G untouched. → R6/R7/R8.
