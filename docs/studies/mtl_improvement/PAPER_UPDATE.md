> ⛔⛔ **2026-06-05 SUPERSEDING BANNER — READ BEFORE USING ANYTHING BELOW.** The Tier-2 close-out's **R1 ("no single-model MTL architecture closes the MTL→STL reg gap; ship the composite")** is **SUBSTANTIALLY INVALIDATED**: the MTL→STL reg gap it studied was largely a **class-weighting CONFOUND**, not architecture. The MTL reg head trained on class-BALANCED CE (`default_mtl use_class_weights=True`, `experiment.py:364` → `mtl_cv.py:1283-1291`) while the STL ceiling used UNWEIGHTED CE — and class-balancing optimizes macro accuracy AWAY from the top-K (Acc@10) metric. **Verified:** with `--no-class-weights`, MTL reg reaches/exceeds the STL ceiling (T2P.0 AL 64.81 ≥ 62.88, vs the buggy 52.90). So the dose-response, the "irreducibly architectural" claim, and "ship the composite" were all measuring a confounded gap. The architecture search attacked an artifact. **R1 is UNDER RE-VALIDATION** (regime finding + §0.1 + composite advantage being re-tested under unweighted reg CE at AL/GE/FL). **R2 (onecycle recipe) is unaffected** (recipe-level, both arms class-weighted → common-mode). Do NOT cite R1 or any absolute MTL-reg number until the re-baseline lands. Full trail: `log.md` 2026-06-05 ROOT-CAUSE entry + `HANDOFF.md` §top.

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
