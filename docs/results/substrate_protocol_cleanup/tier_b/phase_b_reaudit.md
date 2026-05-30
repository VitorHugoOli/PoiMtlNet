# Tier B RE-AUDIT — is the Designs-B/J/Lever-5 reg NULL real or a measurement artefact?

**Date**: 2026-05-29
**Trigger**: user challenge to the Tier B Wave-1 verdict (`phase_b1b2b4_verdict.md`). The shape — reg DEAD-FLAT across 4 mechanistically-distinct substrates (|Δreg| ≤ 0.38 pp, all p ≥ 0.44) + a uniform ~−2 pp next-cat regression — is suspicious: a substrate that lifts STL-reg should move MTL-reg in the same direction while keeping cat roughly flat.
**Scope**: AL, seed=42, 5 folds, H3-alt, `--no-checkpoints`. Three diagnostics D1 (α-dominance, GPU), D2 (STL reproduction, from existing artefacts), D3 (cat-path build scope, code+data). No FL/CA/TX, no Tier A1/C2/C3 touched.

**Verdict in one line: the reg NULL is REAL but its framing was wrong on two counts. The reg-side mechanism is ANCHOR-DOMINATED JOINT TRAINING — under the B9/H3-alt joint config in 50 epochs, the substrate-carrying `stan_logits` branch contributes essentially nothing to MTL reg beyond the α·log_T prior; an α=0 ablation (an out-of-training config) reveals no hidden substrate gain (Δ−0.08, p=0.56) — NOT "substrate fails to transfer." The ~−2 pp cat regression is a BUILD-SCOPE CONFOUND (every design build re-trains a fresh-init CheckinEncoder, so the cat-path checkin vectors drift wholesale), NOT a design mechanism. The "log_T-anchor masking" hypothesis is FALSIFIED. No design warrants promotion. [Hedged per the 2026-05-29 independent verification (see log.md): the stronger reading "the MTL reg encoder is reg-INERT / α=0 floors at chance" overstated the chance level (top10 chance ≈0.9% for 1109 regions, not 5.5%) and cross-attributed the STL no-prior +1.64pp; corrected below.]**

---

## D1 — α-dominance test (KEY). Disable the α·log_T prior (`freeze_alpha=true, alpha_init=0.0`) and re-measure.

Re-ran Design B MTL and canonical-c2hgi MTL at AL with the reg head's α frozen to 0 (so `final_logits = stan_logits` alone — the substrate-carrying encoder branch, no transition prior). Matched control. Same H3-alt recipe, seed=42, 5 folds. RAW per-fold Wilcoxon (scipy exact).

Run dirs: `reaudit_d1/{d1_design_b_a0,d1_canonical_a0}/.../mtlnet_*/`.

| Cell | reg disjoint top10 (joint-best-epoch readout, per-fold) | cat disjoint F1 mean |
|---|---|---:|
| **α=0** design_b | **5.46** [5.03, 8.65, 6.24, 2.64, 4.76] | 10.69 |
| **α=0** canonical | **5.55** [5.03, 6.50, 5.96, 4.76, 5.49] | 10.49 |

> **Note (verification, 2026-05-29):** the α=0 "5.46/5.55" figures are **joint-best-epoch** readouts that land near each fold's noisy reg-top10 peak; they are NOT a converged level. The **all-epoch mean top10 is ~1.1%** (design_b 1.13%, canonical ~1.1%), vs **pure top10 chance ≈ 0.9%** (10/1109 regions at AL). Only 7/250 epoch-points exceed 3.7%. So at α=0 the encoder branch sits **near floor (~1.2× chance)** — it does not acquire region structure — but the earlier "~5.5% ≈ chance" was a best-epoch readout and overstated the chance level by ~6×.
| α=0.1 design_b (orig Tier B) | 50.44 [52.76, 51.30, 52.45, 51.77, 43.92] | 43.59 |
| α=0.1 canonical (orig Tier B) | 50.82 [53.16, 51.10, 53.02, 51.53, 45.30] | 45.76 |

Wilcoxon design_b > canonical, disjoint reg:
- **α=0: meanΔ = −0.08 pp, perfoldΔ [0.00, +2.15, +0.28, −2.11, −0.73], W=5.0, p_gt = 0.5625** → no substrate advantage revealed.
- α=0.1: meanΔ = −0.38 pp, W=3.0, p_gt = 0.906 (reproduces the original Tier B null).

Δcat (design_b − canonical), disjoint: **α=0 → +0.19 pp**; α=0.1 → −2.17 pp.

**Two decisive readings:**

1. **Sanity-gate / the masking hypothesis is FALSIFIED.** With α=0 the reg head sits at **near-floor** — all-epoch mean top10 ~1.1% (vs ~0.9% pure chance for 1109 regions), best-epoch noise peaking ~5–8%, only 7/250 epoch-points above 3.7%, in *both* cells. So under this joint config in 50 epochs the α·log_T prior supplies essentially **all** of the usable reg signal; the `stan_logits` (substrate) branch contributes nothing beyond the anchor. Removing the anchor does NOT uncover a hidden substrate gain (Δ−0.08, p=0.56) — there is no gain to uncover. The anchor was not *masking* a substrate advantage. **Hedge (verification 2026-05-29):** α=0 is an out-of-training config, so "near-floor in 50 epochs" supports "the substrate branch is undertrained / contributes nothing beyond the anchor under this regime" but is weaker than an absolute "the encoder can never learn region under MTL." The defensible claim is regime-and-config-scoped.

2. **The cat side is clean once you remove the substrate-driven readout coupling.** At α=0 (where reg is a degenerate floor and exerts no useful joint pressure) the design_b − canonical cat gap is +0.19 pp — i.e. the cat task itself trains fine (acc ≈ 26 %); the −2.17 pp cat gap at α=0.1 is NOT from the cat head failing. See D3 for its true cause.

---

## D2 — STL reproduction under the current post-bugfix pipeline (from EXISTING artefacts; no new GPU)

The merge_design Design-B STL "dominance" used the `reg_gethard` head (= `next_stan_flow`, α·log_T prior present, alpha_init=0.1) — i.e. the SAME prior as MTL. Re-extracted RAW per-fold `top10_acc` from the current `docs/results/P1/region_head_{alabama,arizona}_..._STL_..._{design_b,check2hgi}_reg_gethard_pf_5f50ep.json`:

| State | design_b STL reg | canonical STL reg | meanΔ | Wilcoxon p_gt |
|---|---:|---:|---:|---:|
| AL | 61.49 [65.26,61.41,65.26,59.95,55.57] | 59.15 [58.91,60.00,64.40,57.46,55.00] | **+2.34 pp** | **0.03125** |
| AZ | 52.59 [52.71,53.80,55.28,47.41,53.76] | 50.24 [47.84,49.31,50.19,49.39,54.47] | +2.35 pp | 0.15625 |

**The STL Design-B reg dominance REPRODUCES exactly under the current pipeline (AL +2.34 pp, Wilcoxon-strict p=0.0312), WITH the α·log_T prior present.** It is not a pre-bugfix artefact.

**Correction (verification 2026-05-29) — the "+1.64 pp without prior" was misattributed.** merge_design **Test 2** is **Florida / Design J**, head `next_gru`, NO log_T — and there **+1.64 pp is the HGI−J gap**, while **J − canonical without prior was +0.86 pp (p=0.0312)**. There is **no STL design_b-AL no-prior artefact on disk**. So the prior-removal evidence (substrate gain survives without log_T) is real but comes from a **different state + design** (FL/J) than the AL/design_b α=0 MTL cell.

**STL ↔ MTL contrast (corrected):** at STL the encoder branch demonstrably learns region and the substrate helps (AL design_b **+2.34 pp WITH prior**, verified); a separate FL no-log_T ablation shows the encoder-branch substrate advantage persists without the prior (J−canonical **+0.86 pp**; HGI−J widens to +1.64 pp). Under MTL the encoder branch is near-floor at α=0 so the substrate cannot express itself. The difference is the training regime — but note the no-prior STL arm crosses state/design from the MTL α=0 cell, so the contrast is **directionally clean, not single-cell apples-to-apples**.

---

## D3 — cat-path build scope (the −2 pp cat). Code + data, no GPU.

`DESIGN_B.md` claims "cat path stays identical to canonical." That is true **only of the gradient-flow architecture** (L_c2p uses the non-detached canonical poi pool; POI2Vec enters detached on the reg path). It is FALSE of the **stored cat-input weights**: `scripts/probe/build_design_{b,j,l}.py` each instantiate a **fresh-init `CheckinEncoder` and re-train it for 500 epochs**, then write `embeddings.parquet` (the cat/checkin input, cols 0–575) from that independent run. So the cat input is a *different training trajectory* than canonical c2hgi, not canonical weights.

Quantified at AL (cols 0–575, checkin level, vs canonical `embeddings.parquet`):

| Design | shape | maxabs Δ | meanabs Δ | frac cells changed |
|---|---|---:|---:|---:|
| design_b | (113846, 64) | 6.12 | 1.22 | 100 % |
| design_j | (113846, 64) | 6.58 | 1.22 | 100 % |
| design_l | (113846, 64) | 5.84 | 1.19 | 100 % |

**The cat-path checkin embeddings moved wholesale (100 % of cells, meanabs ≈ 1.2) across all three designs** — not because of any design mechanism (J/L don't even touch the cat path), but because of shared build-time encoder re-init/re-training drift. The canonical_baseline MTL cell uses the *actual* canonical embeddings, so the Tier B cat comparison carries an **uncontrolled confound**: the uniform ~−2.4 pp cat drop is consistent with re-trained-encoder drift, NOT a substrate property. A region-only-perturbing build (same canonical checkin embeddings, only `region_embeddings.parquet` swapped) would be expected to give the user's predicted "cat flat" — the current builds cannot test that because they regenerate the cat input.

---

## Synthesis — which hypothesis held

| Hypothesis | Status | Evidence |
|---|---|---|
| **log_T-anchor masking** (α·log_T swamps a real substrate reg gain in MTL) | **FALSIFIED** | D1 α=0: reg near-floor (all-epoch mean ~1.1% vs ~0.9% chance); no substrate advantage revealed (Δ−0.08, p=0.56). No gain to mask. |
| **anchor-dominated joint training** (under the B9/H3-alt config in 50 ep, the MTL encoder branch contributes ~nothing beyond the α·log_T anchor; α=0 OOD ablation reveals no hidden gain) | **HELD (regime-scoped)** | D1 α=0 near-floor in MTL vs D2 healthy STL encoder branch (AL +2.34 pp with prior; FL no-prior arm J−canon +0.86 pp). Same head/prior/log_T — the regime differs; no-prior STL arm is cross-state/design, so directional not single-cell. |
| **build-scope cat bug** (−2 pp cat is re-trained-encoder drift, not design) | **HELD** | D3: cat input 100 % changed, meanabs ≈ 1.2, across all 3 designs incl. J/L which don't touch cat; D1 α=0 cat Δ = +0.19 pp. |
| **genuine substrate null** ("substrate fails to transfer") | **REJECTED as the framing** | The substrate DOES transfer at STL (D2). It is the MTL regime that nullifies it, via the anchor-dominated reg head. |

---

## Corrected Tier B framing

- **Accurate claim (hedged per verification)**: "Under the canonical MTL recipe (B9/H3-alt, cat-weight 0.75, 50 epochs), the reg head is dominated by its α·log_T transition anchor — the substrate-carrying encoder branch contributes essentially nothing beyond the prior, and an α=0 ablation (an out-of-training config) leaves reg near-floor (all-epoch mean ~1.1% vs ~0.9% chance) for both design and canonical, revealing no hidden substrate gain (Δ−0.08, p=0.56). Therefore Designs B/J/Lever-5's STL reg advantage (real and reproduced, AL +2.34 pp Wilcoxon-strict WITH prior) cannot move MTL reg under this regime. Tier B measured **'no reg gain BEYOND the canonical log_T anchor under the joint config'**, not 'the substrate fails to transfer.'"
- **Verification caveats (2026-05-29)**: avoid the absolute "MTL reg encoder is reg-INERT / α=0 floors at chance" — α=0 is OOD, the chance level is ~0.9% (not 5.5%), and the STL no-prior arm (+0.86/+1.64 pp) is FL/J, a different state+design than the AL/design_b α=0 cell. The mechanism is supported in direction; state it regime-scoped, not as an architectural law.
- **Both fronts agree** (see `phase_b_two_front.md`): reg is NULL on BOTH the disjoint oracle front AND the deployable geom_simple/joint front (all |Δ|≤1.22 pp, every p≥0.21); no design is null-on-one-front-but-live-on-the-other. The no-promotion verdict holds on the shipped front too.
- **NOT accurate**: "the substrate's STL reg advantage is washed out by the shared backbone *and accompanied by a cat-side cost the STL evaluation never exposed*" — the cat cost is a build-scope confound (re-trained encoder), not a substrate property; and "doesn't transfer" overstates a regime-specific nullification of a real STL effect.
- **Promotion**: unchanged — **no design promoted.** The reg NULL stands for the canonical anchor-dominated MTL reg head, and the cat gate fails only on a confounded comparison. None of B/J/L is a "free upgrade" to the architectural champion.

## Does any design warrant a re-run?

A **clean re-run is well-motivated but LOW expected value** and is NOT launched here:
1. Region-only build (reuse canonical `embeddings.parquet` byte-identical; swap only `region_embeddings.parquet`) to remove the D3 cat confound. This would likely recover "cat flat," but reg would STILL be flat under MTL because of the anchor (D1). So it changes the cat verdict's *attribution*, not the promotion outcome.
2. The only path to MTL reg movement from the substrate would require a reg head whose substrate branch actually learns region under MTL (e.g. α-warmup / encoder-branch pre-train), which is a Tier-C head-mechanism question, not a substrate-swap question — and the §4.2 deploy composite already delivers +7–12 pp reg, dwarfing any sub-1-pp substrate effect.

**Recommendation: keep all four designs NOT PROMOTED; correct the framing (below); do not spend further GPU.** AZ confirmation not required — D1 falsified masking and localised the mechanism to the MTL reg-head architecture, which is state-independent.

---

## Artefacts

- D1 run dirs: `docs/results/substrate_protocol_cleanup/tier_b/reaudit_d1/{d1_design_b_a0,d1_canonical_a0}/...`
- D1 launcher: `scripts/substrate_protocol_cleanup/run_tier_b_reaudit_d1.sh` (detached megascript, DONE marker)
- D2: existing `docs/results/P1/region_head_{alabama,arizona}_region_5f_50ep_STL_*_{design_b,check2hgi}_reg_gethard_pf_5f50ep.json` + merge_design `STATE.md` Test 2
- D3: `scripts/probe/build_design_{b,j,l}*.py` (re-trained CheckinEncoder + `embeddings.parquet` write); cat-divergence quantified inline above
- Head mechanism: `src/models/next/next_stan_flow/head.py` L95-148 (`freeze_alpha` buffer path; `final_logits = stan_logits + α·log_T[last_region_idx]`)
