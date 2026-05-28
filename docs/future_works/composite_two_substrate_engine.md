# Future Work — Composite two-substrate deploy engine (§4.2)

**Date drafted:** 2026-05-21
**Updated:** 2026-05-28 — AL/AZ scoring is **DONE** (Phase 3 Rank 4, [`phase3_rank4_composite_analysis.md`](../results/mtl_protocol_fix/phase3_rank4_composite_analysis.md)). What remains is paper-side framing + FL/CA/TX productionisation (HGI substrate regen at FL/CA/TX missing). The FL/CA/TX productionisation is **explicitly held** until [`docs/studies/mtl_improvement/`](../studies/mtl_improvement/) lands a new champion architecture (the composite ceiling is the upper bound any winner has to beat or match). The relationship to the parallel [`docs/studies/substrate-protocol-cleanup/`](../studies/substrate-protocol-cleanup/) study: substrate-protocol-cleanup may produce a substrate winner (§4.7 Designs J/B under F1) that, IF promoted, *changes the c2hgi cat checkpoint* the composite shipped — re-score becomes free analysis (no retrain). Watch its Tier B for promotion.
**Source:** [`docs/studies/mtl-protocol-fix/DEFERRED_WORK.md`](../studies/mtl-protocol-fix/DEFERRED_WORK.md) §4.2 ("HGI reg-head ensemble"). NEW memo flagged for drafting in DEFERRED_WORK.md §"NEW MEMOS TO DRAFT WHEN PICKED UP".
**Sequencing:** independent of any other future_works memo; the inputs (STL c2hgi cat checkpoints + STL HGI reg checkpoints) are already shipping artefacts at AL/AZ. Pick up whenever the deploy-side framing becomes load-bearing for the paper or operations.
**Distinct from:** [`docs/studies/merge_design/`](../studies/merge_design/) Lever 6 — Lever 6 was an *integrated single-backbone* attempt (FALSIFIED 2026-05-06). This memo's framing is *deploy-time composite*, two separate trained models routed by task.

## What's deferred

Productionise the deployable composite recipe **STL c2hgi (cat) + STL HGI (reg)**: at inference, route cat requests to the c2hgi `next_gru` checkpoint and reg requests to the HGI `next_stan_flow` checkpoint. Both are already trained shipping artefacts; the deploy lift is real (see Phase 3 Rank 4 analysis below).

## What the analysis already shows

Mean deploy lift on reg under this composite (single-seed=42 5-fold):

| State | Composite reg Acc@10 (= STL HGI reg) | vs MTL @ disjoint | vs MTL @ geom_simple |
|---|---:|---:|---:|
| AL | 61.86 ± 3.29 | **+11.04** | **+13.30** |
| AZ | 53.37 ± 2.55 | **+12.04** | **+13.77** |
| FL | 71.34 ± 0.64 | +7.43† | **+1.79** |
| CA | 57.77 ± 1.12 | **+7.16** | **+8.53** |
| TX | 60.47 ± 1.26 | **+9.64** | **+11.17** |

†FL MTL@disjoint comparison uses fresh-log_T multi-seed 63.91 (single-seed=42 stale was 76.47, see C22).

Full analysis: [`docs/results/mtl_protocol_fix/phase3_rank4_composite_analysis.md`](../results/mtl_protocol_fix/phase3_rank4_composite_analysis.md).

**Headline:** at every state, the composite strictly dominates MTL on reg at zero cat cost (MTL c2hgi cat is the same checkpoint).

## Why deferred

1. **Two-model deploy footprint** — composite stores two checkpoints + two substrate parquet files. Operational cost: ~2× reg inference; cat unchanged. Acceptable but a non-default deploy choice.
2. **HGI substrate on disk only at AL/AZ.** Productionising FL/CA/TX requires HGI substrate regen (~3-5 GPU-h per state). One-time cost, but not yet paid.
3. **Paper framing pending** — the composite recipe is a deploy-side reframe of the MTL-vs-STL gap and needs careful §Discussion wording. Sequenced AFTER `mtl_architecture_revisit.md` lands (in case the architectural-revisit closes the gap without needing a composite).

## Acceptance criterion

When picked up:

1. **Substrate regen** at FL/CA/TX (if FL/CA/TX deploy is in scope). AL/AZ unchanged.
2. **Composite scoring script** — load c2hgi cat checkpoint + HGI reg checkpoint, score on shared fold splits, emit a single JSON per state with both head metrics + n_params + flops.
3. **Three-frontier-plus table** — composite added as a 4th column alongside MTL@disjoint / MTL@geom_simple / STL ceiling.
4. **Paper-grade table** — multi-seed (4 seeds × 5 folds) HGI STL reg at FL (the one missing multi-seed cell from §0.3) IF FL is in deploy scope.

## Cost (estimated)

- AL/AZ scoring script + composite table: ~2-3 GPU-h.
- HGI substrate regen at FL: ~3-5 GPU-h.
- (Optional) HGI substrate regen at CA/TX: ~6-10 GPU-h.
- Multi-seed STL HGI reg at FL: ~5-6 GPU-h.
- **Total: 5-25 GPU-h depending on state coverage.** (~3 days calendar.)

## Live docs the work would touch

- `src/training/inference/composite.py` — NEW module (composite scorer)
- `scripts/composite_score.py` — NEW CLI entry point
- `docs/results/RESULTS_TABLE.md` — new §0.8 "Composite deploy recipe"
- `docs/NORTH_STAR.md` — possible deploy-recipe addition
- `docs/CLAIMS_AND_HYPOTHESES.md` — new claim CH24 "composite > MTL on reg"
- `articles/[BRACIS]_Beyond_Cross_Task/PAPER_DRAFT.md` §Discussion — deploy reframe paragraph

## Risks

1. **Paper-grade defense.** Reviewers may push back on "two-model deploy" as a non-architectural contribution. Defense: composite IS the deployable substrate-capacity ceiling; MTL's role is shared-backbone economy + cat lift, NOT reg-head competitiveness. This is a clean MTL-as-tool framing.
2. **STL HGI reg multi-seed missing at FL.** §0.3 reports single-seed=42 only. Multi-seed σ may be larger than the σ=0.64 single-seed at FL (the C22 stale-log_T audit suggests STL HGI reg single-seed was potentially affected by stale log_T as well; need to verify).
3. **Lever 6 falsification.** Lever 6 of merge_design attempted the *integrated* version of this composite and failed. The deploy composite sidesteps the integration problem entirely — but a reviewer who treats Lever 6 as proof that "the gap can't be closed" may also reject the composite. Defense: Lever 6 attempted to inject HGI's signal INTO c2hgi at the boundary level; the composite uses both substrates' fully-converged STL representations at deploy time.

## Pointers

- Phase 3 Rank 4 analysis: [`../results/mtl_protocol_fix/phase3_rank4_composite_analysis.md`](../results/mtl_protocol_fix/phase3_rank4_composite_analysis.md)
- §0.3 STL HGI reg numbers: [`../results/RESULTS_TABLE.md`](../results/RESULTS_TABLE.md) lines 110-114
- §0.6 STL c2hgi cat numbers: same file line 159
- MTL fresh-log_T baseline: [`../results/mtl_protocol_fix/phase2p5_FL_stale_vs_fresh.md`](../results/mtl_protocol_fix/phase2p5_FL_stale_vs_fresh.md)
- Lever 6 falsification (integrated framing, distinct from this composite): [`../studies/merge_design/LEVER_6_FINDINGS.md`](../studies/merge_design/LEVER_6_FINDINGS.md)
