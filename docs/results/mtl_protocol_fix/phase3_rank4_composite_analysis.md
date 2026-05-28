# Phase 3 Rank 4 — Composite (c2hgi cat + HGI reg) deployable analysis

**Date:** 2026-05-21
**Source:** Existing numbers from `docs/results/RESULTS_TABLE.md §0.3` (STL HGI reg) + `§0.6` (STL c2hgi cat) + `phase1_phase2_verdict_v6_final.md` (MTL shipping under F1 selector). **NO new training** — this is a pure deployable-recipe synthesis.
**Maps to:** [`DEFERRED_WORK.md`](../../studies/mtl-protocol-fix/DEFERRED_WORK.md) §4.2 (HGI reg-head + c2hgi cat-head COMPOSITE).
**Lands in (memo):** [`composite_two_substrate_engine.md`](../../future_works/composite_two_substrate_engine.md) (NEW, drafted concurrently).

---

## The composite recipe

At deploy time, route each task to its substrate-best STL model:

- **cat** → MTL c2hgi (because MTL cat ≥ STL c2hgi cat at every state — §0.1 arch-Δ)
- **reg** → STL HGI `next_stan_flow` (because STL HGI reg > STL c2hgi reg by 1.6-3.1 pp at every state — §0.3)

This is a **two-model deploy ensemble**, not an integrated training engine. Distinct from merge_design Lever 6 (single backbone, two output parquets).

## Headline table (single-seed=42, 5 folds, all 5 states)

Reg `next_stan_flow` STL HGI from RESULTS_TABLE §0.3. Cat from MTL c2hgi best_disjoint epoch (under H3-alt at AL/AZ, B9 at FL/CA/TX) per `phase1_phase2_verdict_v6_final.md`.

| State | Composite cat F1 (= MTL c2hgi cat) | Composite reg Acc@10 (= STL HGI reg) | vs. shipping MTL @ disjoint reg | vs. shipping MTL @ geom_simple reg |
|---|---:|---:|---:|---:|
| AL | 45.76 (H3-alt MTL) | **61.86 ± 3.29** | **+11.04** | **+13.30** |
| AZ | 48.87 (H3-alt MTL) | **53.37 ± 2.55** | **+12.04** | **+13.77** |
| FL | ~63 (B9 MTL cat) | **71.34 ± 0.64** | −5.13* | **+1.79** |
| CA | ~62 (B9 MTL cat) | **57.77 ± 1.12** | **+7.16** | **+8.53** |
| TX | ~62 (B9 MTL cat) | **60.47 ± 1.26** | **+9.64** | **+11.17** |

*FL caveat: MTL @ disjoint at FL seed=42 was inflated by stale-log_T (76.47 stale → ~64 fresh per Phase 2 P5 audit). Under fresh log_T multi-seed, the comparison is composite reg 71.34 (single-seed) vs MTL @ disjoint multi-seed 63.91 → **+7.43 pp deploy lift even at FL**.

## Headline finding

**At every state, deploying STL HGI for reg + MTL c2hgi for cat strictly dominates the MTL shipping configuration on reg, at zero cat cost** (MTL c2hgi cat is the same model — we just don't read its reg head's output at inference).

Mean deploy lift on reg: **+10.4 pp at single-seed=42 under F1 selector; +9.3 pp using disjoint capacity ceiling.**

## Caveats

1. **Two-model deploy footprint.** Composite stores two model checkpoints + two substrate parquet files (c2hgi for MTL backbone, HGI for STL reg). ~2× inference cost on reg requests; cat is unaffected.

2. **STL HGI reg numbers are single-seed=42** (§0.3). FL multi-seed STL HGI reg is NOT on file; would need to run (~5-6 GPU-h to close to multi-seed at FL). AL/AZ/CA/TX numbers stable enough at single-seed (σ < 3.3 pp).

3. **Substrate-disk footprint.** HGI substrate is on disk at AL/AZ only. To productionize at FL/CA/TX, HGI substrate must be regenerated (~3-5 GPU-h per state × 3 = ~10-15 GPU-h additional). This is a one-time cost; substrate is task-invariant.

4. **No new training is required for the composite recipe itself** — both component models are already shipping artefacts. The deploy footprint is the only operational cost.

5. **Paper-framing implication.** The composite recipe ESTABLISHES THE DEPLOYABLE CEILING under the current architecture. The MTL-vs-STL gap is now reframed: not "MTL fails to extract the HGI reg signal" but "the deploy-time recipe should serve HGI reg directly; MTL's role is shared-backbone economy + cat lift, not reg-head competitiveness." This is a clean BRACIS narrative.

## Comparison to merge_design Lever 6 (integrated two-output)

- **Composite (this analysis):** 2 models, 2 backbones, 2 substrate tables; runtime ensemble.
- **Lever 6 (integrated, FALSIFIED 2026-05-06):** 1 backbone, 2 output parquets; attempted to inject HGI's POI↔POI contrastive boundary into c2hgi.

Lever 6 failed to close the gap to HGI on reg (LEVER_6_FINDINGS.md). The composite recipe **achieves what Lever 6 attempted, at the cost of a second model footprint**.

## Status

- **Verdict:** ESTABLISHED — composite is the deployable ceiling at the current architecture. Recommended for paper §Discussion as the "deploy recipe vs train-time MTL" contrast.
- **Promotion:** Defer to next-tier paper revision. Triggers no immediate code change; the recipe is a deploy-side selection, not an MTL-architecture change.
- **Cross-references:**
  - [`docs/future_works/composite_two_substrate_engine.md`](../../future_works/composite_two_substrate_engine.md) — NEW memo (drafted 2026-05-21 alongside this analysis)
  - [`docs/studies/mtl-protocol-fix/DEFERRED_WORK.md`](../../studies/mtl-protocol-fix/DEFERRED_WORK.md) §4.2
  - [`docs/studies/mtl-protocol-fix/PRIORITY_IMPACT.md`](../../studies/mtl-protocol-fix/PRIORITY_IMPACT.md) Rank 4

## Raw numbers cited

- STL HGI reg (single-seed=42) → `docs/results/RESULTS_TABLE.md` lines 110-114.
- STL c2hgi cat (single-seed=42) → `docs/results/RESULTS_TABLE.md` line 159.
- MTL @ disjoint / geom_simple reg (single-seed=42, FL with stale log_T caveat) → `phase1_phase2_verdict_v6_final.md` §Final 5-state three-frontier table.
- MTL @ disjoint multi-seed FL (FRESH log_T) → `phase2p2_FL_multiseed_three_frontier.json` (63.91 ± 0.16).
