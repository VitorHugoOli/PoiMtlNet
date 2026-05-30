# Tier B Wave 1 verdict — Designs B / J / Lever 5 (KL distill) MTL under F1 at AL/AZ

> **⚠ RE-AUDIT ADDENDUM (2026-05-29) — framing corrected, numbers below unchanged.**
> A re-audit (`phase_b_reaudit.md`) of the reg NULL + uniform −2 pp cat shape found two of the explanations in this doc are wrong, though the **promotion outcome (no design promoted) stands**:
> - **§1/§2 reg NULL is REAL but the mechanism is anchor-dominance, not "substrate fails to transfer."** D1 (α frozen to 0, AL) shows that WITHOUT the α·log_T prior reg sits at **near-floor** (all-epoch mean top10 ~1.1% vs ~0.9% pure chance for 1109 regions; best-epoch noise peaks ~5–8%) for BOTH design and canonical — so under the B9/H3-alt joint config in 50 epochs the substrate-carrying `stan_logits` branch contributes **essentially nothing beyond the anchor**. Reg is an α·log_T-anchor readout; the substrate can't move it under this regime. The "log_T-anchor masking" idea is FALSIFIED (removing the anchor reveals NO hidden gain, Δ−0.08 pp, p=0.56). Meanwhile the STL substrate advantage is REAL and reproduces (AL +2.34 pp, Wilcoxon p=0.0312, **with** the prior; a separate FL no-prior ablation shows J−canonical +0.86 pp). So Tier B measured *"no reg gain BEYOND the canonical log_T anchor under the joint config,"* not *"the substrate doesn't transfer."* **[Hedged per verification 2026-05-29: the chance level is ~0.9% not 5.5% — the "5.5%" was a best-epoch readout; "reg-inert" overstates an α=0 OOD result; and the no-prior +1.64 pp was HGI−J at FL, not design_b — corrected to J−canon +0.86 pp. Two-front check (`phase_b_two_front.md`): reg NULL holds on BOTH disjoint and joint/geom fronts.]**
> - **§3/§4 uniform ~−2.4 pp cat is a BUILD-SCOPE CONFOUND, not a substrate side-effect.** `build_design_{b,j,l}.py` each re-train a fresh-init CheckinEncoder for 500 ep and rewrite `embeddings.parquet` (the cat input), so the cat-path checkin vectors drift wholesale (100 % of cols changed, meanabs ≈ 1.2 vs canonical) — even for J/L which don't touch the cat path by design. At α=0 (reg degenerate, no readout coupling) design_b−canonical cat Δ = +0.19 pp, i.e. the cat task trains fine; the −2.17 pp is re-trained-encoder drift vs the canonical baseline's actual canonical embeddings.
> See `phase_b_reaudit.md` for D1/D2/D3 detail. The §1–§5 tables/numbers below are retained verbatim.

**Date**: 2026-05-28
**Phase**: Tier B Wave 1 (B1 = Design B, B2 = Design J, B4 = Lever 5).
**Scope**: AL + AZ, seed=42, 5 folds, H3-alt small-state recipe, `--no-checkpoints`.
**Baseline**: the `canonical_baseline` MTL cell at the SAME state (same seed=42, same H3-alt recipe, same folds) — NOT the separate phase1v3 JSON.
**Provenance**: post-allowlist-fix clean rerun. The first Tier B attempt failed on two engine-allowlist gates (`scripts/train.py:1548`, `src/data/folds.py:873`); both fixed. The design engines reuse canonical c2hgi sequences/folds/log_T verbatim — only the substrate embeddings differ.

**Verdict in one line: ALL THREE DESIGNS FAIL THE PROMOTION GATE at both states. No design promoted. No reg lift; uniform ~−2.4 pp cat regression. No B3 winner-substrate stack required.**

---

## §1 Three-frontier tables (per state)

Frontiers per fold from the val metric CSVs: disjoint reg = max-over-epochs `top10_acc_indist`; disjoint cat = max-over-epochs `f1`; geom_simple = epoch maximising `sqrt(cat_f1 · reg_top10)` over shared epochs. Means over 5 folds (seed=42). STL reg ceiling = `next_stan_flow` matched-head, `RESULTS_TABLE.md §0.1 v11` L70-71.

### Alabama

| Substrate | best joint reg (geom_simple) | best disjoint reg | best disjoint cat F1 | STL reg ceiling |
|---|---:|---:|---:|---:|
| canonical c2hgi (control) | 48.56 | 50.82 | 45.76 | 61.21 |
| Design B | 48.68 | 50.44 | 43.59 | 61.21 |
| Design J | 47.34 | 50.60 | 43.71 | 61.21 |
| Lever 5 (KL distill) | 47.83 | 50.54 | 43.27 | 61.21 |

### Arizona

| Substrate | best joint reg (geom_simple) | best disjoint reg | best disjoint cat F1 | STL reg ceiling |
|---|---:|---:|---:|---:|
| canonical c2hgi (control) | 39.60 | 41.33 | 48.87 | 53.06 |
| Design B | 39.11 | 41.35 | 46.45 | 53.06 |
| Design J | 39.43 | 41.30 | 46.21 | 53.06 |
| Lever 5 (KL distill) | 40.23 | 41.34 | 46.46 | 53.06 |

Every design sits within ±0.5 pp of canonical on disjoint reg at both states and 10–12 pp below the STL reg ceiling — i.e. the substrate swap does NOT move the MTL→STL reg gap.

---

## §2 Wilcoxon (5-fold paired, one-sided design > canonical, RAW per-fold values) on disjoint reg top10_acc

RAW per-fold `top10_acc_indist` (no rounding); paired by fold; `scipy.stats.wilcoxon(deltas, alternative="greater")`. Per the Tier-A1 scipy-dispatch lesson, raw values are used (5-fold has no ties → exact branch).

### Alabama (baseline disjoint reg = 50.82)

| Design | per-fold Δ pp (f1..f5) | mean Δ | folds + | W | p (1-sided) | Δcat pp | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| Design B | −0.41, +0.20, −0.57, +0.24, −1.38 | −0.38 | 2/5 | 3.0 | 0.9062 | −2.17 | FALSIFIED |
| Design J | −0.04, +0.12, −0.20, +0.12, −1.10 | −0.22 | 2/5 | 5.0 | 0.7812 | −2.05 | FALSIFIED |
| Lever 5 | +0.16, +0.16, 0.00, −0.37, −1.38 | −0.28 | 2/5 | 3.0 | 0.8125 | −2.49 | FALSIFIED |

### Arizona (baseline disjoint reg = 41.33)

| Design | per-fold Δ pp (f1..f5) | mean Δ | folds + | W | p (1-sided) | Δcat pp | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| Design B | −0.06, +0.19, +0.02, 0.00, −0.02 | +0.03 | 2/5 | 6.0 | 0.4375 | −2.41 | NULL |
| Design J | +0.04, +0.12, −0.06, −0.10, −0.12 | −0.02 | 2/5 | 6.0 | 0.6875 | −2.66 | FALSIFIED |
| Lever 5 | −0.14, −0.06, −0.02, +0.02, +0.25 | +0.01 | 2/5 | 6.0 | 0.6875 | −2.41 | NULL |

No design reaches p ≤ 0.05 at either state (every p ≥ 0.44). Δreg is within fold noise (|mean Δ| ≤ 0.38 pp). **The disjoint-reg promotion criterion fails outright for all three designs.**

NULL = not significant, mean Δreg ≥ 0 (AZ B/L, where reg is flat-to-trivially-positive but p ≫ 0.05). FALSIFIED = not significant with mean Δreg < 0 (all AL designs + AZ J). Either way: not promoted.

---

## §3 Δcat F1 (design − canonical) at disjoint

| Design | AL Δcat pp | AZ Δcat pp |
|---|---:|---:|
| Design B | −2.17 | −2.41 |
| Design J | −2.05 | −2.66 |
| Lever 5 | −2.49 | −2.41 |

**Every design also independently fails the Δcat ≥ −0.5 pp non-inferiority gate** — a uniform ~−2.4 pp cat regression at both states. So even the two AZ NULL cells (B, L) where reg is flat-positive would still be rejected on the cat gate. The promotion gate requires BOTH `p ≤ 0.05` AND `Δcat ≥ −0.5 pp`; none satisfies either condition.

---

## §4 C18 leak-probe per design (F_TIER_A1_LEAK_AUDIT reasoning)

The leak signatures from `F_TIER_A1_LEAK_AUDIT.md` / C18: a *clean* lift is reg-side with roughly-flat cat; a *leak* is a large disjoint reg jump with NO geom_simple lift and/or cat anti-correlation (a label-shortcut leak lifts cat alongside reg, per the T3.1 GAT catastrophic pattern).

- **Design B** — disjoint reg flat (Δ −0.38 AL / +0.03 AZ), geom reg flat (+0.12 / −0.49), cat DROPS (−2.17 / −2.41). There is no reg jump to explain and cat moves DOWN, not up. This is the OPPOSITE of a label-shortcut leak. NO leak signature. (Moot — design is not promoted.)
- **Design J** — identical shape: reg flat, geom reg flat-to-negative, cat drops ~−2.4 pp. NO leak signature.
- **Lever 5** — reg flat (Δ −0.28 AL / +0.01 AZ), geom reg mixed (−0.72 AL / +0.63 AZ), cat drops ~−2.4 pp. NO leak signature.

**Leak-probe conclusion**: none of the three is leak-shaped. A leak would have produced a large disjoint-reg jump and/or a cat *lift*; instead all three show flat reg and a uniform cat *regression*. The uniform ~−2.4 pp cat drop across three mechanistically-distinct designs at both states is most consistent with a shared property of the design `embeddings.parquet` checkin-level vectors (the cat task input differs from canonical, while the reg task region embeddings net out flat) — a substrate side-effect, NOT a leak and NOT a lift. No dedicated leak audit is warranted (nothing was promoted; nothing is leak-shaped).

---

## §5 Decision per design

| Design | AL | AZ | Combined verdict |
|---|---|---|---|
| Design B (B1) | FALSIFIED (p=0.91, Δreg −0.38, Δcat −2.17) | NULL (p=0.44, Δreg +0.03, Δcat −2.41) | **NOT PROMOTED** — no reg lift; cat-gate fails |
| Design J (B2) | FALSIFIED (p=0.78, Δreg −0.22, Δcat −2.05) | FALSIFIED (p=0.69, Δreg −0.02, Δcat −2.66) | **NOT PROMOTED** — no reg lift; cat-gate fails |
| Lever 5 (B4) | FALSIFIED (p=0.81, Δreg −0.28, Δcat −2.49) | NULL (p=0.69, Δreg +0.01, Δcat −2.41) | **NOT PROMOTED** — no reg lift; cat-gate fails |

No design promoted to multi-seed (A-tier), on EITHER the disjoint or joint/geom front (`phase_b_two_front.md`). **[RE-AUDIT-CORRECTED 2026-05-29]** The merge_design STL dominance of Designs B/J does NOT *show up* under MTL+F1 — but the corrected mechanism is **anchor dominance**, not generic washout: under the joint config the MTL reg head is dominated by its α·log_T transition anchor (D1: α=0 → reg near-floor, all-epoch mean ~1.1% vs ~0.9% chance), so the substrate-carrying encoder branch contributes ~nothing beyond the prior and cannot express its (real, reproduced) STL advantage. (Hedged: α=0 is an OOD config; this is a regime-scoped statement, not "the encoder can never learn region under MTL.") The accompanying ~−2.4 pp cat drop is a **build-scope confound** (every design build re-trains the CheckinEncoder, drifting the cat input), NOT a substrate-exposed cat cost. See `phase_b_reaudit.md`. ~~The substrate's STL reg advantage is washed out by the shared MTL backbone and accompanied by a cat-side cost the STL evaluation never exposed.~~ (original wording struck — overstated on both counts.)

**Consequence for B3 (Lever 4)**: per `INDEX.md` §B3, Lever 4 is applied to (a) canonical c2hgi (control) and (b) the Wave-1 winner. With NO Wave-1 winner, only the canonical+Lever4 control (`check2hgi_lever4_canonical`) is run; the (b) winner-stack is skipped.

---

## §4.2 Composite cross-reference

A PROMOTE in Tier B would have been a "free upgrade" to whatever the architectural champion is (per `INDEX.md` §B framing line 49 + §4.2) — NOT a project headline. The project's strongest reg lift remains the §4.2 deploy composite (STL c2hgi cat + STL HGI reg), which already gives **+7 to +12 pp** disjoint reg vs MTL at every state (`phase3_rank4_composite_analysis.md`). Tier B Wave 1 produced no upgrade to layer onto the architectural champion: all three substrate variants are reg-flat and cat-regressing under MTL+F1. Even had one promoted, its expected magnitude (the merge_design STL lift was ~sub-1 pp at AL/AZ) would have been far below the composite headline and must never be cited as the project's reg story.

---

## Artefacts

- Per-cell run dirs: `docs/results/substrate_protocol_cleanup/tier_b/{canonical_baseline,design_b,design_j,design_l}/{alabama,arizona}/seed42/mtlnet_*/`
- Analysis script (RAW-value Wilcoxon vs canonical_baseline cell): `scripts/substrate_protocol_cleanup/analyze_tier_b_wave1.py`
- Raw extraction JSON: regenerate via `.venv/bin/python scripts/substrate_protocol_cleanup/analyze_tier_b_wave1.py`
