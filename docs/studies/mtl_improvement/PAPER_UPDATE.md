> ⛔⛔ **2026-06-05 SUPERSEDING BANNER — READ BEFORE USING ANYTHING BELOW.** The Tier-2 close-out's **R1 ("no single-model MTL architecture closes the MTL→STL reg gap; ship the composite")** is **SUBSTANTIALLY INVALIDATED**: the MTL→STL reg gap it studied was largely a **class-weighting CONFOUND**, not architecture. The MTL reg head trained on class-BALANCED CE (`default_mtl use_class_weights=True`, `experiment.py:364` → `mtl_cv.py:1283-1291`) while the STL ceiling used UNWEIGHTED CE — and class-balancing optimizes macro accuracy AWAY from the top-K (Acc@10) metric. **Verified:** with `--no-class-weights`, MTL reg reaches/exceeds the STL ceiling (T2P.0 AL 64.81 ≥ 62.88, vs the buggy 52.90). So the dose-response, the "irreducibly architectural" claim, and "ship the composite" were all measuring a confounded gap. The architecture search attacked an artifact. **R1 is UNDER RE-VALIDATION** (regime finding + §0.1 + composite advantage being re-tested under unweighted reg CE at AL/GE/FL). **R2 (onecycle recipe) is unaffected** (recipe-level, both arms class-weighted → common-mode). Do NOT cite R1 or any absolute MTL-reg number until the re-baseline lands. Full trail: `log.md` 2026-06-05 ROOT-CAUSE entry + `HANDOFF.md` §top.

# ⭐ PAPER UPDATE (2026-06-05) — the reg narrative is REFRAMED (the "unweighting" finding)

> The earlier "Tier 2 close-out" (R1/R2 below, 2026-06-04) is **superseded** by this section. The whole "MTL sacrifices reg → ship the composite + architecture-negative" line was an **objective-mismatch confound**: the MTL reg head trained on **class-weighted CE** while the STL ceiling + the reported **Acc@10** metric are unweighted. Fixed (both heads unweighted, per-task) and re-validated multi-seed. **Source-of-truth numbers + full trail: `CONCERNS.md §C25`, `log.md` 2026-06-05, `HANDOFF.md §top`.**

## The single sentence
**A single jointly-trained MTL model matches/beats the STL reg ceiling AND the 2-model composite once the reg loss matches the reported metric (unweighted CE for top-K Acc@10, not class-balanced CE); the substrate gain transfers to MTL; and cat improves too.**

## Re-validated headline numbers (multi-seed {0,1,7,100}, unweighted real-joint, AL/GE/FL)
| state | MTL reg (v14) | STL ceiling (c) | composite (d) | MTL cat (v14) | STL cat ceiling |
|---|---|---|---|---|---|
| AL | 64.52 | 62.88 (**+1.6**) | 63.58 (**+0.9, beats it**) | 53.38 | 49.97 (**+3.4**) |
| GE | 57.84 | 58.45 (−0.6) | 58.76 (−0.9) | 61.37 | 58.12 (**+3.2**) |
| FL | 71.55 | 73.31 (−1.8*) | 73.62 (−2.1) | 71.89 | 69.97 (**+1.9**) |

\* the FL −1.8pp residual is the **α·log_T prior** (prior-OFF closes ~80%) and is **closed by the dual-tower** (FL dual-tower 73.06, −0.25 vs ceiling) — not a fundamental MTL limit.

## What changes for the paper (claim-by-claim)
- **Regime finding (CH28) — OVERTURNED.** Old: "STL substrate gains wash out under the cross-attn MTL regime." New: **Δreg(v14−canonical) = +1.92/+1.49/+0.81 (AL/GE/FL), σ~0.1 — the substrate gain TRANSFERS to MTL** (partial — smaller than the STL Δ — but positive and significant).
- **Composite advantage (CH25) — DISSOLVED.** Old: "composite STL-cat ⊕ STL-HGI-reg = +7-12pp over MTL@disjoint; deploy 2 models." New: a **single MTL model ≥ the composite** (AL +0.94; GE/FL within 1-2pp). The 2-model deploy is no longer justified on reg.
- **Tier-2 architecture — POSITIVE, not negative.** Old: "no architecture closes the gap; dual-tower loses; irreducibly architectural." New: the **dual-tower CLOSES the gap** (FL +1.51 vs base_a, −0.25 vs ceiling) — the orderings FLIPPED under the fix (`dual_gated > prior_off > crossstitch > base_a ≈ hardshare`). The class-weighted "dual-tower loses" was the confound interacting non-uniformly with its private reg tower.
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
