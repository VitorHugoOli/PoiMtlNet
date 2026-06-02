# Tier B TWO-FRONT analysis — Designs B / J / Lever-5 vs canonical, DISJOINT and JOINT(geom_simple), reg AND cat

**Date**: 2026-05-29
**Type**: Pure re-analysis of EXISTING per-fold val CSVs. NO GPU, no retraining.
**Scope**: AL + AZ, seed=42, 5 folds, H3-alt small-state recipe. Cells: `canonical_baseline`, `design_b`, `design_j`, `design_l` (α=0.1 canonical recipe) + the α=0 re-audit cells (`reaudit_d1/d1_design_b_a0`, `d1_canonical_a0`, AL only).
**Trigger**: the existing Tier B verdict + re-audit reported mostly DISJOINT reg. The user wants BOTH fronts (disjoint + joint) for reg AND cat, per cell, per state, to expose anything single-front reporting hid.

## Definitions

- **DISJOINT** = each head at its OWN best val epoch (oracle upper bound on what the artefact contains). reg = max-over-epochs `top10_acc_indist`; cat = max-over-epochs `f1`. Two checkpoints.
- **JOINT (geom_simple)** = both heads read off the SINGLE epoch maximising `sqrt(cat_f1 · reg_top10_indist)` over shared epochs. One deployable checkpoint. Matches the canonical `joint_geom_simple` selector (`scripts/canonical_improvement/analyze_t64_selectors.py` L78-85) and the Tier A1 geom_simple definition exactly.
- **Wilcoxon**: RAW per-fold values (no rounding), paired by fold, one-sided `design > canonical` for reg; scipy auto-exact (5 folds, no ties). Cat p shown one-sided `design>canon` for symmetry (the gate uses Δcat ≥ −0.5, not a p-test). Analyser: `scripts/substrate_protocol_cleanup/analyze_tier_b_two_front.py`.

**Cross-check PASSED**: every DISJOINT number reproduces `phase_b1b2b4_verdict.md` to the reported decimal (Design B AL reg Δ = −0.38, p = 0.9062, Δcat = −2.17; AZ B Δreg +0.03, p 0.4375; J/L identical), and JOINT reg means match its §1 frontier table (AL canon 48.56 / B 48.68; AZ L 40.23). The α=0 cell reproduces `phase_b_reaudit.md` D1 (Δreg −0.08, p 0.5625; Δcat +0.19). No discrepancy.

## Canonical baseline reference means (per-fold, seed=42)

| Cell | reg DISJOINT | reg JOINT | cat DISJOINT | cat JOINT |
|---|---:|---:|---:|---:|
| canonical AL (α=0.1) | 50.82 | 48.56 | 45.76 | 45.18 |
| canonical AZ (α=0.1) | 41.33 | 39.60 | 48.87 | 47.30 |
| canonical AL (α=0)  | 5.55 | 4.99 | 10.49 | 8.13 |

## Two-front table (Δ = design − canonical, pp; p = one-sided design>canon, raw, exact)

| Design | State | reg DISJOINT Δ (p) | reg JOINT/geom Δ (p) | cat DISJOINT Δ | cat JOINT/geom Δ |
|---|---|---:|---:|---:|---:|
| **Design B** | AL | −0.38 (0.906) | +0.12 (0.406) | −2.17 | −2.47 |
| **Design B** | AZ | +0.03 (0.438) | −0.49 (0.969) | −2.41 | −1.45 |
| **Design J** | AL | −0.22 (0.781) | −1.22 (0.969) | −2.05 | −1.78 |
| **Design J** | AZ | −0.02 (0.688) | −0.17 (0.844) | −2.66 | −1.89 |
| **Lever 5** | AL | −0.28 (0.812) | −0.72 (0.906) | −2.49 | −2.08 |
| **Lever 5** | AZ | +0.01 (0.688) | +0.63 (0.219) | −2.41 | −2.22 |
| **Design B α=0** | AL | −0.08 (0.562) | −0.26 (0.781) | +0.19 | +0.81 |

Per-fold reg Δ (for the record):
- B AL disjoint `[−0.41,+0.20,−0.57,+0.24,−1.38]`; joint `[+1.34,−1.14,−0.49,+1.14,−0.24]`
- B AZ disjoint `[−0.06,+0.19,+0.02,0.00,−0.02]`; joint `[−0.48,−0.63,−1.20,+0.04,−0.15]`
- L AZ joint `[+2.81,+0.29,−0.64,+0.89,−0.17]` (the lone >0 mean-Δ joint cell; still p=0.219, fold 1 dominated)
- B α=0 AL disjoint `[0.00,+2.15,+0.28,−2.11,−0.73]`; joint `[−1.30,+4.91,−0.97,−3.21,−0.73]`

## Interpretation per design

- **Design B** — reg NULL on BOTH fronts (disjoint −0.38 p0.91 / joint +0.12 p0.41 at AL; +0.03 / −0.49 at AZ). The tiny positive joint Δ at AL (+0.12) is noise (p=0.41, 2/5 folds positive), not a hidden gain. cat REGRESSES on both fronts at both states (−1.5 to −2.5 pp). No front rescues it.
- **Design J** — reg NULL→slightly NEGATIVE on both fronts; the joint front is WORSE than disjoint at AL (−1.22, p=0.97). cat regresses ~−1.8 to −2.7 pp on both fronts. No front helps.
- **Lever 5** — reg NULL on disjoint; on joint, AZ shows the largest positive mean Δ in the whole slate (+0.63) but it is NOT significant (p=0.219, driven by fold-1 +2.81) and AL joint is −0.72. cat regresses ~−2.1 to −2.5 pp on both fronts. No promotable signal.

**Does the front choice change any verdict?** NO. On reg, both fronts are inside fold noise (|Δ| ≤ 1.22 pp, every p ≥ 0.21) at both states — the disjoint NULL and the joint NULL agree. On cat, both fronts show the same uniform ~−2 pp regression. The joint front does NOT reveal a design that is null-on-disjoint-but-positive-on-joint (or vice versa): the one mean-positive joint cell (Lever-5 AZ) is non-significant and its AL counterpart is negative. **Single-front reporting hid nothing material here** — which is itself the finding: the substrate swap is inert on every front.

## α=0 two-front numbers (the log_T prior's contribution per front)

Both reg fronts collapse to floor (~5 % top10 = chance for 1109 AL regions) once α·log_T is frozen to 0: canonical reg DISJOINT 5.55 / JOINT 4.99; Design B 5.46 / 4.73. Design B − canonical at α=0 is reg DISJOINT −0.08 (p0.56) / reg JOINT −0.26 (p0.78) — **no substrate advantage on either front**. So on BOTH fronts the entire ~50/40 pp reg signal at α=0.1 is supplied by the log_T anchor, not the substrate-carrying encoder branch (which is reg-inert under MTL). The α=0 cat Δ is mildly POSITIVE on both fronts (DISJOINT +0.19, JOINT +0.81), confirming the cat task itself trains fine and the α=0.1 cat regression is the build-scope CheckinEncoder-drift confound (re-audit D3), independent of which front you read.

## Summary — does the two-front view change the Tier B promotion verdict?

**No. The verdict is unchanged on BOTH fronts: no design promoted.** The promotion gate (disjoint reg p ≤ 0.05 AND Δcat ≥ −0.5) fails on every cell — and adding the joint/geom front does not flip it: joint reg is also non-significant everywhere (every p ≥ 0.21) and joint cat carries the same ~−2 pp regression. There is no design that is null-on-disjoint-but-significant-on-joint, so the deployable (joint) framing yields the same NOT-PROMOTED outcome as the oracle (disjoint) framing.

**Corrected Tier B claim, both fronts explicit:** *Under the canonical MTL recipe at AL/AZ, Designs B/J/Lever-5 produce NO significant reg lift over canonical c2hgi on EITHER the disjoint (per-task-best-epoch) OR the joint (geom_simple, deployable) front (all reg p ≥ 0.21, |Δreg| ≤ 1.22 pp), and regress cat by ~−2 pp on both fronts. The α=0 control shows reg collapses to chance on both fronts without the log_T anchor (Δ vs canon −0.08/−0.26, n.s.), confirming the MTL reg head is a pure log_T-anchor readout whose substrate-carrying encoder branch is reg-inert regardless of selector; the cat regression is the build-scope encoder-drift confound (positive Δcat at α=0 on both fronts), not a substrate property.* No design warrants multi-seed promotion on any front.

## Artefacts

- Analyser (this doc): `scripts/substrate_protocol_cleanup/analyze_tier_b_two_front.py` (RAW-value Wilcoxon, both fronts, both tasks; reuses the wave-1 disjoint+geom extraction and the canonical geom_simple definition).
- Cells: `docs/results/substrate_protocol_cleanup/tier_b/{canonical_baseline,design_b,design_j,design_l}/{alabama,arizona}/seed42/mtlnet_*/metrics/`
- α=0: `docs/results/substrate_protocol_cleanup/tier_b/reaudit_d1/{d1_design_b_a0,d1_canonical_a0}/.../alabama/mtlnet_*/metrics/`
- Reconciles with: `phase_b1b2b4_verdict.md` (disjoint), `phase_b_reaudit.md` D1 (α=0), `tier_a1/phase_a1_verdict.md` (geom_simple method).
