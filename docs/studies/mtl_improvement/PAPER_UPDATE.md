# ⭐ PAPER UPDATE (2026-06-12) — code-audit layer: claims on HOLD pending X-series + literature positioning

> **Three paper-facing items from the 2026-06-12 deep code audit (`CODE_AUDIT_2026-06-12.md`; probes
> = `HANDOFF_AUDIT.md §X-SERIES`). G's headline numbers are NOT invalidated — but hold these claims:**
> 1. **"K/V mixing is dead" / "cat gain is architecture, not transfer" — HOLD until X1.** Cross-attn
>    trained on randomly-paired windows (two independent shuffled loaders) and evaluated aligned —
>    per-sample cross-modal transfer was never trainable, which *itself* predicts exactly those two
>    findings. X1 (aligned-training run) decides whether they are intrinsic or artifacts.
> 2. **"KD adds nothing on the dual-tower" — RETRACTED (dead codepath; the kd arms were no-ops).**
>    KD-on-G runs for the first time in X2. Do not cite the old verdict.
> 3. **Literature positioning for the MTL-negative (write it this way):** with a 7-class (~2.8-bit)
>    auxiliary, cos≈0, and a data-rich tuned main task, parity-not-improvement is the EXPECTED
>    outcome (Du et al. arXiv:1812.02224; Bingel & Søgaard EACL'17; Kurin & Xin NeurIPS'22). Frame
>    the negative as the **weak-auxiliary regime**, not "category auxiliaries don't work" (the
>    positive literature uses 180-300+ class vocabularies). Phrase orthogonality as a
>    **first-order average statement** (Fifty et al. NeurIPS'21 lookahead-affinity is the reviewer
>    counter; a per-module/lookahead measurement is the optional hardening). **Pre-empt MCARNN
>    (IJCAI'18)** — the one published same-architecture MTL>STL ablation in this task pair: 2018
>    low-capacity RNN, single-run no-variance, rich latent-topic aux, and its own λ-sweep shows the
>    main task improving as the aux weight shrinks. Also disclose: the +3pp cat comparison spans
>    head-config differences (STL ceiling: 2-layer GRU, dropout 0.3, logit-adjust τ=0.5; G's cat
>    head: 4-layer, dropout 0.1, plain CE) — absorbed by the "architecture-dominated" framing but
>    must be stated. Citations + mechanism future-work list: `CODE_AUDIT_2026-06-12.md` Part 2 +
>    INDEX `#T7-FW`.

# ⭐ PAPER UPDATE (2026-06-10) — the MECHANISM + the analysis/limitations section (Tiers 3/4/5 closed)

> This is the newest layer. The G Pareto-positive headline (below, 2026-06-06/07) is unchanged; this
> section adds the paper-relevant findings from the Tier-3/4/5 close-out. Source docs:
> `WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md`, `results/mtl_improvement/{T4_audit_and_verdict.md,
> orthogonality_intrinsic_test.md, cat_transfer_and_T53.md, T52_cathead_sweep.md}`; claims `CLAIMS CH31`.

**1. The unifying mechanism — the two tasks are gradient-ORTHOGONAL (tested intrinsic).**
cos(∇L_cat, ∇L_reg) on the shared trunk ≈ 0 (FL +0.0007 / AL +0.0026). TESTED, not asserted: in a
fully-shared model where reg's shared gradient is *larger* than cat's (ratio 1.26 AL / 1.78 FL), cos is
still ≈0 — so it's intrinsic to the task pair, not induced by the dual-tower. This single fact is the
paper's mechanistic spine: it explains why the MTL win is **architectural/representational, not
gradient-cooperation**, why more parameter-sharing *hurts* (induces conflict that isn't there), and why
the dual-tower wins (it exploits the orthogonality).

**2. Analysis / limitations section — no modern MTL optimizer helps (a paper-grade NEGATIVE).** The full
`src/losses` balancer registry (GradNorm/PCGrad/CAGrad/Nash/Aligned-MTL/DWA/DB-MTL/FAMO/UW/RLW…) +
loss-scale normalization + a static-weight fairness sweep all FAIL to Pareto-beat tuned
`static_weight cw=0.75`. This is the *expected* k=2-with-tuned-baseline result (Kurin NeurIPS'22,
Xin NeurIPS'22): at gradient cos≈0 there is no conflict for a balancer to resolve.
⚠ *Paper wording (2026-06-12 precision)*: state it as a **convergent-evidence negative** (defaults
screen + tuned-static fairness sweep + RLW litmus + cos≈0 mechanism), NOT as "every method
individually tuned" — and disclose that the gradient-surgery family is inapplicable to the
dual-tower as wired (collapses to equal-weighting, which was itself screened and lost). See the
evidence-strength banner in `results/mtl_improvement/T4_audit_and_verdict.md`.
Figures: `figs/{grad_cosine_tasks, t4_balancer_scatter_FL, t4_loss_weight_trajectories_FL}.png`.

**3. State the cat result precisely (decomposition).** MTL category beats single-task by +3 pp, but
the gain is **architecture-dominated** (the cross-attn encoder: +2.13 FL / +3.22 AL, 4-seed), with only
a small genuine region→category **transfer (+1.08 FL / −0.67 AL)**. Frame as "the joint cross-attn
architecture is a better category *encoder*, with a small region transfer at scale" — NOT "the region
task teaches category."

**4. Completeness (no champion change).** The reg-input axis (overlap data-scale, HGI substrate
routing), the loss/optimization axis, and the head axis (cat-encoder sweep under G → `next_gru` is the
unique multi-state choice; HSM reg head = flat softmax) are all EXHAUSTED — none beats G. A bonus
scale-conditional finding (a conv-attention cat head wins +1.06 at FL only, craters small states) is
logged as future-work, not adopted. Only Tier-6 completeness (CA/TX build) + this restatement remain.

---

> ⛔⛔ **2026-06-05 SUPERSEDING BANNER — READ BEFORE USING ANYTHING BELOW.** The Tier-2 close-out's **R1 ("no single-model MTL architecture closes the MTL→STL reg gap; ship the composite")** is **SUBSTANTIALLY INVALIDATED**: the MTL→STL reg gap it studied was largely a **class-weighting CONFOUND**, not architecture. The MTL reg head trained on class-BALANCED CE (`default_mtl use_class_weights=True`, `experiment.py:364` → `mtl_cv.py:1283-1291`) while the STL ceiling used UNWEIGHTED CE — and class-balancing optimizes macro accuracy AWAY from the top-K (Acc@10) metric. **Verified:** with `--no-class-weights`, MTL reg reaches/exceeds the STL ceiling (T2P.0 AL 64.81 ≥ 62.88, vs the buggy 52.90). So the dose-response, the "irreducibly architectural" claim, and "ship the composite" were all measuring a confounded gap. The architecture search attacked an artifact. **R1 is UNDER RE-VALIDATION** (regime finding + §0.1 + composite advantage being re-tested under unweighted reg CE at AL/GE/FL). **R2 (onecycle recipe) is unaffected** (recipe-level, both arms class-weighted → common-mode). Do NOT cite R1 or any absolute MTL-reg number until the re-baseline lands. Full trail: `log.md` 2026-06-05 ROOT-CAUSE entry + `HANDOFF.md` §top.

# ⭐ PAPER UPDATE (2026-06-05) — the reg narrative is REFRAMED (the "unweighting" finding)

> The earlier "Tier 2 close-out" (R1/R2 below, 2026-06-04) is **superseded** by this section. The whole "MTL sacrifices reg → ship the composite + architecture-negative" line was an **objective-mismatch confound**: the MTL reg head trained on **class-weighted CE** while the STL ceiling + the reported **Acc@10** metric are unweighted. Fixed (both heads unweighted, per-task) and re-validated multi-seed. **Source-of-truth numbers + full trail: `CONCERNS.md §C25`, `log.md` 2026-06-05, `HANDOFF.md §top`.**

## The single sentence
**A single jointly-trained MTL model MATCHES the single-task STL reg ceiling (Pareto-non-inferior) and substantially BEATS the cat ceiling (+3 pp) — once (i) the reg loss matches the reported metric (unweighted CE for top-K Acc@10), (ii) the reg task gets a private un-diluted pathway fused additively (`aux`), and (iii) the biased α·log_T prior is turned off. The previously-reported −7…−17 pp 'MTL sacrifices reg' tension was a class-weighting artifact (C25); once fixed, the MTL tradeoff is Pareto-POSITIVE (matches reg + beats cat), not the classic sacrifice.**

> ⚠ **REG CLAIM CORRECTED 2026-06-07 (B-A2, the critique's checkpoint re-eval).** Earlier drafts of this doc said G "beats both ceilings." The independent re-eval found the reg "beat" compared G's *in-distribution* Acc@10 to the (c) ceiling's *full* `top10_acc` (the p1 harness has no indist split); on a **matched** metric G is ~0.35 pp BELOW the (c) reg ceiling (FL: 72.93 vs 73.31). **Honest claim: reg = "matches" (Pareto-non-inferior, within ~0.4 pp); cat = "beats by +3 pp" (exact).** Trail: `log.md`/`INDEX.html #T2V-3` 2026-06-07. (Every "beats both ceilings" phrasing below should read "matches reg + beats cat".)

> ⛔ **G′ (the cat-private both-private dual-tower, FL cat 74.77) is NOT a paper claim — DO NOT CITE.** It is an FL-only experimental dead-end: the cat gain craters −3.6…−15.3 pp at AL/AZ/GE (AL/AZ below the STL cat ceiling), and a rescue screen found no recoverable config (CLOSED 2026-06-07; `INDEX.html #T2V-5` / `CHAMPION.md §G′`). The paper champion is **G** with the **cat-SHARED `next_gru`** head. Cite G's cat (AL 52.91 / AZ 54.48 / GE 61.43 / FL 73.16), never G′'s 74.77.

> ⭐ **Champion (G) — `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` (`raw_embed_dim=64 fusion_mode=aux freeze_alpha=True alpha_init=0.0`), v14 substrate, unweighted onecycle KD-OFF. CONFIRMED MULTI-STATE @ 4 seeds {0,1,7,100} (2026-06-06) — MATCHES the STL reg ceiling + BEATS the STL cat ceiling at ALL 4 available states** *(table corrected in place 2026-06-12 to the R0 matched-metric values — `results/mtl_improvement/R0_matched_metric_bar.json`; the original Δreg column here was the indist-vs-full artifact, +1.59/+0.64/+0.92/+0.26)*:
>
> | state | G reg (full, matched) | (c) STL reg (full) | Δreg (matched) | G cat | (c) STL cat | Δcat |
> |---|---|---|---|---|---|---|
> | AL | 62.57±0.10 | 62.67±0.13 | **−0.09** (matches) | 52.91±0.27 | 50.35 | **+2.56** |
> | AZ | 54.68±0.24 | 54.80±0.22 | **−0.12** (matches) | 54.48±0.74 | 50.39 | **+4.08** |
> | GE | 58.35±0.04 | 58.44±0.06 | **−0.09** (matches) | 61.43±0.26 | 57.50 | **+3.93** |
> | FL | 72.97±0.06 | 73.27±0.06 | **−0.31** (matches) | 73.16±0.04 | 69.96 | **+3.20** |
>
> Composite, matched metric (R0): the (d) FL composite reg-full is **73.49** → G is **−0.53** on reg alone; the composite is dominated only on the **joint** reading (G: reg ≈ −0.5, cat **+3.2**, at ~half the deploy footprint) — do NOT write "strictly dominated" / "ties the composite" (those were indist-vs-full artifacts). CA/TX deferred (no v14 substrate). Trail: `log.md` 2026-06-06 + R0 2026-06-08; drivers `c25_combos_{screen,promote}.sh` + `c25_g_multistate.sh`, re-score `r0_matched_rescore.py`.

## Re-validated headline numbers (multi-seed {0,1,7,100}, unweighted real-joint, AL/GE/FL)
| state | MTL reg (v14) | STL ceiling (c) | composite (d) | MTL cat (v14) | STL cat ceiling |
|---|---|---|---|---|---|
| AL | 64.52 | 62.88 (**+1.6**) | 63.58 (**+0.9, beats it**) | 53.38 | 49.97 (**+3.4**) |
| GE | 57.84 | 58.45 (−0.6) | 58.76 (−0.9) | 61.37 | 58.12 (**+3.2**) |
| FL | 71.55 | 73.31 (−1.8*) | 73.62 (−2.1) | 71.89 | 69.97 (**+1.9**) |

\* the FL row above is the **base_a cross-attn baseline** (71.55). The −1.8pp residual is the **α·log_T prior** + shared-pathway dilution, and is now **OVER-closed**: dual-tower gated 73.06 (−0.25), and the **(G) `dual aux + prior-OFF` champion BEATS the ceiling at 73.57 (+0.26)** and matches the composite (see the champion callout above). Not a fundamental MTL limit — it was the prior + the gated-fusion's competition between the private and shared pathways; `aux` fusion (additive, non-diluting) + prior-OFF resolves both.

## What changes for the paper (claim-by-claim)
- **Regime finding (CH28) — OVERTURNED.** Old: "STL substrate gains wash out under the cross-attn MTL regime." New: **Δreg(v14−canonical) = +1.92/+1.49/+0.81 (AL/GE/FL), σ~0.1 — the substrate gain TRANSFERS to MTL** (partial — smaller than the STL Δ — but positive and significant).
- **Composite advantage (CH25) — DISSOLVED** *(verb corrected 2026-06-12 per R0 matched metric)*. Old: "composite STL-cat ⊕ STL-HGI-reg = +7-12pp over MTL@disjoint; deploy 2 models." New: the +7–12pp composite reg advantage collapses to **+0.5pp at FL (73.49 vs G 72.97, matched full metric)** while G wins cat by +3.2 at roughly half the deploy footprint — the composite is dominated on the **joint** reading, NOT "strictly" (it keeps a ~0.5pp reg-only edge at FL; the earlier "TIES at 73.57 vs 73.62" was the indist-vs-full artifact).
- **Tier-2 architecture — POSITIVE, and now BEATS the ceiling.** Old: "no architecture closes the gap; dual-tower loses; irreducibly architectural." New: the **dual-tower CLOSES the gap** (gated FL +1.51 vs base_a, −0.25 vs ceiling) and the **(G) `aux`-fusion + prior-OFF variant BEATS it** (FL 73.57, +0.26 over ceiling, 4-seed). Orderings flipped under the fix (`dual_aux_off > dual_privonly_off > dual_gated > prior_off > crossstitch > base_a ≈ hardshare`). The class-weighted "dual-tower loses" was the confound interacting non-uniformly with its private reg tower. **Mechanism settled:** MoE expert-capacity (mmoe/cgc) and SwiGLU backbone-quality were both NULL on reg — the gap is the α·log_T prior + shared-pathway dilution (resolved by `aux` additive fusion + prior-OFF), NOT architecture capacity/quality.
- **§0.1 absolute MTL reg — re-stated +10-13pp** (canonical v11 GCN substrate: AL 62.60 / GE 56.34 / FL 70.74, multi-seed, vs old ~50/42/61). The "MTL reg ≪ STL" architectural-Δ column flips to "MTL reg ≈ STL ceiling."
- **MTL cat — re-stated UP** (exceeds the STL cat ceiling at all states). Caveat: stacks with the deployable-recipe correction (don't double-count); use unweighted MTL cat vs unweighted STL cat for apples-to-apples.
- **R2 (onecycle small-state recipe) — UNAFFECTED** (recipe-level, common-mode).
- **Per-task class-weighting — a genuine recipe WIN** to adopt (cat-unweighted +3-5pp macro-F1 is not just a bug fix).

## Open before the paper freezes
FL-B9 follow-up for exact §0.1-table recipe continuity (re-validation used onecycle); the Acc@1→Acc@10 reg checkpoint-monitor fix (deployable numbers); the dual-tower + prior-OFF combo (full FL closure confirmation); pin a new `CANONICAL_VERSIONS` version for the unweighted recipe; promote to `NORTH_STAR.md`. **Frozen (c)/(d) STL ceilings are UNAFFECTED** (unweighted p1, valid comparands).

---

# Tier 2 close-out — paper-facing update (2026-06-04) — ⚠ R1 SUPERSEDED, see banner above

The MTL Improvement track's **Tier 2 (architecture)** is COMPLETE. Two paper-grade results, both
pointing the same way, plus one consequential recipe finding. This doc is what the BRACIS paper should
absorb; numbers + provenance below. (Author: implementing agent, branch `mtl-improve`.)

---

## R1 — Architecture: a clean, multi-seed-hardened NEGATIVE

**No single-model MTL architecture closes the MTL→STL region-prediction gap.** The reg-private
dual-tower — the track's centerpiece, motivated by the §6.4 "missing private backbone" decomposition —
**loses** to the matched shared-backbone baseline, and the loss is structural, not a tuning artifact.

**Evidence (all v14, onecycle, KD-OFF, seeded per-fold log_T, 5f×50ep, frozen-fold paired):**
1. **Sharing dose-response** (reg@10 disjoint, vs matched base_a): the dual-tower is the WORST point on a
   monotonic 5-point curve at all three states —
   `CrossStitch ≥ base_a ≈ hard-share ≫ dual-tower-gated > dual-tower-private`.
   This is the OPPOSITE of the §6.4 hypothesis: **more cross-task sharing helps reg; isolating the reg
   head hurts it.**
2. **FL multi-seed** ({0,1,7}): dual-tower 59.03±0.15 vs base_a 62.38 = **−3.35** (tight σ) → the negative
   holds at the large state, multi-seed.
3. **Mechanism (3 cells):** the MTL→STL reg gap is NOT multi-task interference (cat-weight=0 barely moves
   reg), NOT the α·log_T prior, NOT weight decay (prior-OFF+wd0.01 matched to the STL ceiling barely
   moves reg). Even reg-only, recipe-matched to the STL ceiling, the cross-attn-MTL reg head sits
   −5.8/−10.5 below it. **The residual is the joint cross-attn MTL pathway itself** (a sharper P4).

**Deployable conclusion: the two-model composite (STL c2hgi-cat ⊕ STL HGI-reg, routed by task) is the
answer for the reg axis; the MTL reg gap is irreducibly the joint training pathway, not a fixable head
or substrate.** The one architecture that *doesn't* lose reg is **CrossStitch** — a real but small
multi-seed partial (+1.0pp reg at AL/AZ, σ-tight; cat mixed, AL −1.4), still 5–10pp below the STL
ceiling → not a gap-closer.

**Off-limits / confirmed-dead** (do not re-run): the dual-tower (this work), thin residual-skip (§6.2),
MMoE/CGC/PLE (§6.3), substrate/routing-in-MTL (regime finding). T2.0 hard-share ≈ base_a (hard ≈ soft
sharing). The architecture axis is exhausted.

---

## R2 — Recipe: a scale-conditional optimizer finding (onecycle), with a §0.1 implication

A byproduct of the per-arch LR sweep: **`onecycle` (aggressive schedule, NO alternating-optimizer-step)
dominates the blessed small-state recipe H3-alt and beats B9 at small states.** Genuinely new — the
recipe-selection study (`RESULTS_TABLE §0.4`) compared H3-alt vs B9 as a binary and pinned B9's
small-state rejection on alt-SGD; the "aggressive-schedule-minus-alt-opt" cell was never tested at AL/AZ.

**Mechanism (recipe × state matrix, baseline head):** at small states the aggressive schedule helps reg
AND no-alt-opt helps both reg and cat (B9's alt-SGD tanks small-state cat ~−7pp). **alt-opt flips sign by
scale** — it hurts small-state cat, is neutral/helpful at large scale. So:
- **AL/AZ: onecycle dominates** (v14 multi-seed: +6–9pp reg / +1–2pp cat vs H3-alt).
- **FL/CA: keep B9** (onecycle does not dominate — FL reg-tie + B9 wins cat; CA B9 +2.5 reg / onecycle
  +2.0 cat, a trade).

**Paper-substrate (v11) confirmation** (AL/AZ {0,1,7,100}, diagnostic-best, vs §0.1's B9):
| | onecyc reg | Δ vs B9 | arch-Δ reg (was) | onecyc cat | Δ vs B9 | arch-Δ cat (was) |
|---|---|---|---|---|---|---|
| AL | 53.15 | +2.98 | −8.06 (−11.04) | 47.93 | +7.36 | +6.58 (−0.78) |
| AZ | 41.54 | +0.76 | −11.52 (−12.28) | 49.79 | +4.69 | +5.89 (+1.20) |

### §0.1 implication — READ THE NUANCE before editing the submission
§0.1's architectural-Δ table reports **MTL B9 at all states**, but the *shipped* small-state recipe is
**H3-alt** (§0.4) — so §0.1's small-state cat numbers (e.g. AL B9 cat 40.57, the "−0.78 vs STL") are a
**B9-recipe artifact** (B9 tanks small-state cat). Therefore:
- The large **cat** change under onecycle (AL −0.78 → +6.58) is **mostly fixing §0.1's B9-vs-H3-alt
  inconsistency**, not a pure onecycle effect: vs the shipped H3-alt (cat ~46.78), onecycle cat (~47.9)
  is only ~+1pp. The honest statement: **the deployable small-state cat is positive vs STL; §0.1's
  −0.78 was the B9 recipe, not the deployable one.**
- The **reg** change under onecycle is **modest on the paper substrate** (AL +2.98, AZ +0.76) → the reg
  arch-deficit shrinks only a little (AL −11→−8, AZ −12.3→−11.5). The big +6-9pp reg gain is a v14 (new
  base) phenomenon; on v11 it is smaller.

**Recommendation to the paper:** (i) adopt onecycle as the small-state recipe (it dominates H3-alt and
beats B9); (ii) re-state the small-state §0.1 architectural-Δ **under the deployable recipe** so the cat
axis isn't understated by B9; (iii) keep the framing honest — the reg deficit shrinks modestly, the cat
"flip" is largely a recipe-consistency fix. **This reshapes a central §0.1 small-state claim → it needs
author sign-off, not a silent rewrite.** Numbers are in hand; the annotation is in `RESULTS_TABLE` +
`T21_recipe_matrix.md`.

---

## Provenance
Study log: `docs/studies/mtl_improvement/log.md` (2026-06-04 entries). Design + capstone review:
`T2.1_DUALTOWER_DESIGN.md`, log capstone entry. Results: `docs/results/mtl_improvement/T21_*.txt` +
`T21_recipe_matrix.md`. Code: `src/models/{next/next_stan_flow_dualtower,mtl/mtlnet_crossattn_dualtower}`,
`scripts/mtl_improvement/t21_*`. Frozen (c)/(d) ceilings UNTOUCHED; `t14_freeze_sanity.py` GREEN.
