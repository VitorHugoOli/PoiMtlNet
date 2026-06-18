# FINAL SYNTHESIS — `mtl_frontier` (CLOSED 2026-06-17)

> **Read this first for the closed study.** Then `FINDINGS.md` (per-lever mechanism + every number),
> `STATE.md` (queue + decisions log), `HANDOFF.md` (entry point + reusable code). The cross-study one-liner
> is in `docs/studies/log.md`. The paper-facing finding is `articles/[BRACIS]_Beyond_Cross_Task/MEMO_2026-06-17_catreg_regime_and_C2.md`.

## 1. Outcome in one paragraph

`mtl_frontier` brought the **post-2022 MTL frontier** — output-level priors, asymmetric/learned sharing,
input-side conditioning, Pareto-front profiling, model-merging, residual optimizers, auxiliary tasks — to
this repo's measured LBSN regime (**cos(∇cat,∇reg)≈0** on the shared trunk, **dual-tower** with a private
reg tower, **data-rich** next-region main task, **weak 7-class ~2.8-bit** category auxiliary). Across **10
lever-families** (R1, R2, R3, R10, 3 follow-ups, conditional-coupling, R-CC+, **R5**) plus the non-gated
methodology items **R4** (Pareto front), **R7** (merge-vs-joint), **R9** (optimizer close-out), the result
is **9 nulls** (R1, R2, R3, R10, FU1, FU2, FU3, R-CC+, R5) **+ 1 genuine sub-threshold positive (conditional
coupling, FL cat +0.235 / reg +0.070)**; **R4**
resolved the C21 selector saga (publish the epoch-front), **R7** measured Merge<joint-G, **R9** closed the
optimizer aisle. **No lever clears the ≥0.3 multi-seed promote gate; champion G is unchanged; nothing flows
to `closing_data` G0.2.** The frontier **replicates but does not exceed** champion G's two wins (dual-tower +
log_T-KD) — a strong, citable **domain-frontier negative**. The study's most consequential output is the
**cat↑/reg↓ resolution** (§4): the apparent "MTL sacrifices reg" is **not a loss** — reg is at its STL
ceiling, insulated by a β→0 shared channel; the paper's C2 "−7…−17pp reg cost" is **~half a class-weighting
confound + ~half config**, not a representational tradeoff (§4 + the C2 memo).

## 2. The regime — settled, do NOT re-litigate

- **cos(∇cat,∇reg) ≈ 0** on the shared trunk ⇒ no first-order cross-task transfer through the gradient;
  re-gating it (R2/R10/aux_gated/SSC) is empty.
- **reg is a private STAN tower at its single-task ceiling** (champion G reg = STL reg within −0.09…−0.31pp,
  4 states × 4 seeds, `R0_matched_metric_bar.json`); its learned shared-fusion **β→0 at FL** (insulated).
  No reg headroom to gain and no shared channel to disturb at scale.
- **cat harvests the shared cross-attn trunk** (the easy 7-class task); it is movable, and joint co-training
  lifts it **+2.6…+4.1pp over the STL cat ceiling**.
- **β is gated by state scale**: ~0 at FL, live (+0.078 GE … +0.12 AL) at small states — where the shared
  channel is live, levers move **both heads together** (R1 AL cat+0.20/reg+0.21); at FL only cat moves.
- **log_T-KD saturates the output-prior family** (R1/R3 add nothing over it) and is **cat-harmful at FL**
  (small-state-only). **The optimizer aisle is closed** (19-arm null + Kurin/Xin + Mueller TMLR'25 + R9).

## 3. The lever roster (what was tested, the verdict)

| lever | family | verdict |
|---|---|---|
| R1 log_C co-location KD | output-prior | NULL (AL reg +0.207 sub-threshold; FL null) |
| R2 STEM-AFTB gating | asymmetric sharing | NULL (AL-only cat lift, decays to FL −0.03; reg null) |
| R3 CrossDistil | output-prior | NULL (log_T-KD saturates the family) |
| R10 GRM gated read | asymmetric sharing | NULL (GRM≡G; cos≈0 nothing to gate; re-audited 2026-06-17, §5) |
| FU1 GRM other-states | asymmetric sharing | NULL everywhere (AL 4-seed corrected +0.045/−0.066) |
| FU2 aux_gated | input-dep fusion | NULL+harmful (**bug-fixed 2026-06-17**: FL both heads −0.91/−0.31) |
| FU3 best-stack | stacking | NULL (sub-additive) |
| **cc** conditional coupling | **input-side conditioning** | **sub-threshold POSITIVE** (FL cat +0.235 / reg +0.070, 4/4 seeds) |
| **R-CC+** cc family map | input-side conditioning | NULL (signal×injection×output-side; nothing beats additive cc; cap = 2.8-bit aux) |
| **R4** Pareto front | profiling | DONE — weight-front near-corner; epoch-front resolves C21; PaLoRA declined (reg-privacy) |
| **R5** per-instance KD gating | output-prior | NULL (fired gate = comparand artifact; worse than KD-off champion on both heads) |
| **R7** merge-vs-joint | merging | DONE — **Merge < joint G** (ensemble loses the joint cat lift; reg unchanged) |
| **R9** BayesAgg-MTL | optimizer | DONE — impl unfaithful + craters at champion recipe; optimizer aisle closed |
| R6 ForkMerge / R8 time-aux | scheduling / aux-task | deferred reasoned-negatives (see §7) |

## 4. ★ The cat↑/reg↓ resolution — reg is at its ceiling, not lost (the headline finding)

A 4-lens, adversarially-verified audit (`FINDINGS.md §CAT↑/REG↓`, §AUDIT) answers the question "cat improves
while reg degrades — are we losing something?": **No.**
1. **Selective perception** — across 41 lever-deltas the largest quadrant is cat+/**reg+** (19); the cat–reg
   correlation is *positive*; reg is positive in 28/41. The impression came from R5 (wrong comparand) + R10
   (seed noise) only.
2. **Measurement artifact** — cat & reg peak ~26 epochs apart; at the deployable checkpoint R5 is
   cat-up/reg-**flat**, not reg-down.
3. **Mechanism** — the **β-gated regime split** (β→0 insulates reg at FL; live β at small states moves both
   heads). cat harvests the shared trunk; reg is a saturated private tower.
4. **No bug; reg not under-trained** — **MTL reg matches the same-substrate STL ceiling within −0.09…−0.31pp.**

**Paper consequence (the C2 memo).** The BRACIS draft's C2 ("classic MTL tradeoff: −7…−17pp next-region
cost") is **not a representational tradeoff**: a controlled A/B shows **~half is the C25 class-weighting
confound** (unweighting alone recovers +3.15pp at FL, halving the gap) and **the rest closes under the
confound-free champion G** (v14 substrate + dual-tower) → **reg-parity + a +2.6…+4.1pp cat lift**. Recommended
C2 reframing → *"Pareto gain on the easy task at parity on the hard one."* Full memo + honest caveats
(confound+config not a flag-flip; CA/TX unmeasured; cat lift partly head-config):
`articles/[BRACIS]_Beyond_Cross_Task/MEMO_2026-06-17_catreg_regime_and_C2.md`. This is known-pending project
work (`closing_data` re-runs §0 on the confound-free champion; `CONCERNS §C25`: "only paper-doc restatement
remains").

## 5. R10 re-evaluation (the "most promising" lever) — done correctly, placed correctly

A dedicated re-audit (13-agent workflow, 2026-06-17, all 3 verifications hold) re-examined R10's
implementation, usage, evaluation, and **placement**:
- **Implementation: correct + faithful.** The GRM gate `γ=σ(W·masked-mean_seq(query))` on each tower's
  cross-attn read is the paper's primitive; manual forward == model bit-exact; champion G (grm=False)
  bit-identical; gradient flows; gate is **live** (trained γ swings 0.874→0.31) yet metrics ≡ G. The +263,168
  shared-only params == exactly 2 blocks × 2 gates × (256²+256). No artifact.
- **Evaluation: correct.** FL seed0 cat +0.324 → multi-seed {0,1,7,100} **+0.085±0.203 (2/4 cat seeds
  negative, Wilcoxon p=0.31), reg −0.027 (p=0.82)** = null; reproduced bit-exact. Multi-seeded at FL (the
  tight headline state) per the promote-gate protocol. GRM fires at eval (no `self.training` guard).
- **Placement: on-point + fully mapped.** R10 was applied at the spec's **named primary point** (the
  dual-tower cross-attn read). The other placements were evaluated and are **not worth running**:

| placement | channel | faithful | status | why not run |
|---|---|---|---|---|
| **P1 cross-attn read** (done) | cross-task | yes | DONE/null | cos≈0 → nothing to gate (the genuine GRM≡G null) |
| P2 hierarchical fusion ("on the layers") | substrate | yes | regime-bounded | STL-axis, rising-tide-bounded + reg-at-ceiling; VERY HIGH cost (substrate rebuild) |
| P3 reg-head fusion gate | intra-reg | yes | = FU2 `aux_gated` | already run, **harmful** (FL −0.91/−0.31) |
| **P4 "in a TOWER"** (intra-STAN gate, `priv_grm`) | intra-reg | yes | **MEASURED null** (2026-06-17) | FL multi-seed **Δcat +0.075±0.167 (p=0.31, 2/4 seeds neg), Δreg −0.033** — washes out to ≈ R10's cross-attn +0.085; reg tower at its STL ceiling |
| **P5 "as a HEAD"** (GRU state gate, `grm_state`) | intra-cat | partial | **MEASURED sub-gate** (2026-06-17) | FL seed0 cat +0.229 (<0.3, no multi-seed); AL +0.80 cat / **−0.23 reg** = seed0-flare-with-reg-cost; length-9 kills growing-memory |
| P6 SSC / Memory-Soup | cross-task | yes | covered | same cross-attn read P1 nulled; trained-γ "nothing to gate" closes the family |

**Answer to "should it be a head or in a tower?" — MEASURED 2026-06-17 (the user's two hypotheses, now
empirical).** Implemented both as default-off flags (champion G bit-identical, 5/5 identity tests pass):
**P4 "in a tower"** = GRM gated read inside the private reg STAN tower (`--reg-head-param priv_grm=True`);
**P5 "as a head"** = GRM on the cat GRU's last hidden (`--cat-head-param grm_state=True`). Both are
genuinely *different intra-task* channels the cos≈0 null does not directly cover — and both reproduce the
**same seed0-flare-that-washes-out** signature as the cross-attn placement: **P4 FL seed0 cat +0.324
(identical to R10's cross-attn +0.324) → multi-seed +0.075 NULL** (2/4 seeds negative, p=0.31, reg −0.033);
**P5 FL seed0 +0.229 sub-gate** (AL +0.80 cat but −0.23 reg = a cat-only flare with a reg cost). The
intra-reg placement is bounded by the reg tower being **at its STL ceiling** (P4 + the morally-equivalent
FU2 `aux_gated`, harmful), and the intra-cat by the **length-9 window** (no growing-memory) + cat already
lifted. **So no placement — cross-attn (P1), hierarchical-fusion (P2), reg-head (P3=FU2), intra-tower (P4),
GRU-head (P5), SSC/Soup (P6) — beats champion G. R10 is fully mapped and closed.** Artifacts:
`docs/results/mtl_frontier/{r10_placement_results.json, r10_p4_fl_multiseed_results.json}`; drivers
`scripts/mtl_frontier/{r10_placement_screen.sh, r10_p4_fl_multiseed.sh}`; code `grm_read` (NextHeadSTAN) /
`priv_grm` (dualtower) / `grm_state` (next_gru), all default-off.

## 6. Corrections & retractions registry (cite the RIGHT claim)

| superseded | correct | source |
|---|---|---|
| FU2 aux_gated "γ opens the shared pathway; FL cat −0.85" | **shared pathway was SEVERED (`head.py:208` bug)**; fixed + re-ran → FL both heads −0.91/−0.31 | §Follow-up, §AUDIT |
| FU1 "AL 4-seed +0.141/+0.178" | seed-0-only (agg bug); **true 4-seed +0.045 reg / −0.066 cat** | §Follow-up, §AUDIT |
| R5 "fired the 0.3 gate → promote" | comparand artifact vs handicapped global-W-KD base; **worse than KD-off champion on both heads** | §R5, §AUDIT |
| paper C2 "−7…−17pp irreducible reg cost / classic tradeoff" | **~half C25 confound + ~half config**; confound-free champion = reg-parity + cat lift | §4, C2 memo |
| R4 cat numbers cross-compared to cc/R5 | **convention A vs B** (+0.259pp cat offset) — do not cross-compare | §R4 |

## 7. Deferred / expected-null future levers (consolidated ledger — none owed before closure)

All regime-bounded expected-nulls, correctly deferred (each with mechanism + falsifier in `FINDINGS.md`):
- **R10 P2** hierarchical GRM/Memory-Soup fusion across Check2HGI levels — substrate-axis, rising-tide +
  reg-at-ceiling bounded; STL-first; very-high cost.
- **R10 P4/P5** intra-tower (STAN) / GRU-head GRM gates — the user's head/tower hypotheses; regime-bounded
  (reg-at-ceiling / length-9-window); FU2 ≈ measured-harmful. **LOW cost if a measured confirmation is wanted.**
- **R10 SSC top-k router / Memory-Soup** — same cross-task channel as the nulled GRM read.
- **R-CC+ cross-attn cat↔reg coupling** (cat penultimate as K/V queried by reg) — bounded by the 2.8-bit
  aux cap + `features` already-hurts.
- **Faithful ICML'24 BayesAgg** (gradient-direction posterior) — the one untested optimizer arm; expected
  null by 19-arm + Mueller + cos≈0.
- **R6 ForkMerge** (≤ G: scheduled_static-null + R4 near-corner) / **R8 next-visit-time aux** (rising-tide
  null + 2→3-task refactor) — measured negatives available on request.

## 8. What flows onward

- **To `closing_data` G0.2:** **NOTHING** (no promoted lever; champion G = canon v16 unchanged). The
  no-promotion outcome is faithfully propagated (`closing_data/PLAN.md` G0.2 remains the empty placeholder).
- **To the paper:** the **C2 reframing memo** (`articles/[BRACIS]_…/MEMO_2026-06-17_catreg_regime_and_C2.md`)
  — advisory; the actual C2 restatement + §0 re-baseline on the confound-free champion is `closing_data` +
  paper-team work.

## 9. Process lessons

1. **Read the promote-gate against the *deployable champion*, not a lever's internal control** (R5: a fired
   gate that was a comparand artifact).
2. **Matched same-batch baselines + multi-seed before any claim** — every single-seed flare (R3/R10/cc-AL/R5)
   washed out or inverted multi-seed.
3. **Assert the mechanism fires (C28)** AND that it's bit-identical when off — but a *live* gate that ≡ G
   (R10 trained-γ) is the real regime result, not a bug.
4. **Audit inherited code** — the two bugs (FU2 severed pathway, FU1 seed mislabel) were in inherited
   follow-ups, found only by the per-experiment re-audit.
5. **Reconcile across the doc trail** — the cat/reg "loss" was the paper's stale, C25-confounded C2; the
   confound-free champion tells a stronger story.

## 10. What shipped in code (all on `study/mtl-frontier`, champion G defaults unchanged)

`cond_coupling`/`cond_signal`/`cond_temp`/`cond_topk`/`cond_inject`/`cond_logit_prior` (reg head); the
`aux_gated` fusion + **the `head.py:208` shared-tower fix**; directional AFTB (`aftb_spec`/`detach_ab/ba`);
`crossattn_grm` + `_masked_mean_seq`; log_C-KD + reverse cat-KD + warmup/ec; `--log-t-kd-gate`
(coverage_max/entropy); R4/R5/R7/R9 drivers + aggregators. All flags default-off → champion G bit-identical
(5/5 `test_mtlnet_crossattn_identity` pass after every change).

## 11. Reading map

- Mechanism + every number per lever → `FINDINGS.md` (§R1…§R9, §CAT↑/REG↓, §AUDIT, §SYNTHESIS).
- Queue + chronological decisions → `STATE.md`.
- Entry point + reusable infra + champion-G invocation → `HANDOFF.md`.
- Paper-facing C2 finding → `articles/[BRACIS]_Beyond_Cross_Task/MEMO_2026-06-17_catreg_regime_and_C2.md`.
- The regime this builds on → `docs/studies/archive/mtl_improvement/FINAL_SYNTHESIS.md` (champion G).
- Cross-study one-liner → `docs/studies/log.md`.
