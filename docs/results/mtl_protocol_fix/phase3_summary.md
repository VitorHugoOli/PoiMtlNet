# Phase 3 — Post-closure deferred-work execution summary

**Date:** 2026-05-21
**Scope:** Execution pass on `docs/studies/archive/mtl-protocol-fix/DEFERRED_WORK.md` items per the user-confirmed Option A ("polish the closed study"). Tier 1 items run; Tier 2/3 deferred to next-tier studies.
**States:** AL/AZ + FL (CA/TX skipped per user 2026-05-21 — closure of substrate + MTL means large-state compute is not worth it).
**Seeds:** single seed=42 throughout. Multi-seed promotion gated on next-tier work.

## Aggregate verdicts

| Rank | Item | Verdict | Δ disjoint reg (AL/AZ/FL) | Cat impact | Artefact |
|---|---|---|---|---|---|
| 1 | §4.5 log_T as **supervisory signal** | **PROMOTED** | +2.40 / +5.06 / +2.32 pp @ W=0.2 (all 9 cells Wilcoxon-strict p=0.0312) | flat at disjoint | [`phase3_rank1_findings.md`](phase3_rank1_findings.md) |
| 2 | §4.6 **class-balanced reg sampler** | **FALSIFIED** | −30.46 / −18.49 / (skipped) pp (all 10 cells p=1.0000) | flat at disjoint | [`phase3_rank2_findings.md`](phase3_rank2_findings.md) |
| 3 | §4.1 per-task best-epoch shipping | **DROPPED** by user 2026-05-21 | — | — | sub-track in [`mtl_architecture_revisit.md`](../../future_works/mtl_architecture_revisit.md) |
| 4 | §4.2 **composite** (HGI reg + c2hgi cat) | **ESTABLISHED** | +11.04 / +12.04 / +7.43 pp vs MTL@disjoint (zero retrain) | matches MTL cat | [`phase3_rank4_composite_analysis.md`](phase3_rank4_composite_analysis.md) + [`composite_two_substrate_engine.md`](../../future_works/composite_two_substrate_engine.md) |
| 5-8 | rest | DEFERRED | — | — | see [`PRIORITY_IMPACT.md`](../../studies/archive/mtl-protocol-fix/PRIORITY_IMPACT.md) Tier 2/3 |

## Headline finding — the residual MTL-vs-STL reg gap is architectural

Three independent strands now converge:

1. **P4 frozen-cat** (Phase 2, mtl-protocol-fix v6) — MTL reg peaks at ep 2-4 even when cat is fully frozen from epoch 0 ⇒ cat-task gradient interference is NOT the bottleneck.
2. **Rank 2 balanced sampler** (THIS PASS, FALSIFIED) — Re-weighting the reg dataloader regresses reg by 18-30 pp ⇒ long-tail under-sampling is NOT the bottleneck (the existing weighted-CE is enough; layering a sampler creates dual-prior conflict).
3. **canonical_improvement Tier 6** (closed 2026-05-19) — Substrate axis exhausted at ±0.8 pp ⇒ the gap is NOT substrate either.

By elimination, the residual is the MTL **shared-backbone architecture** itself. This makes `mtl_architecture_revisit.md` the unambiguous next-tier study.

## Headline finding — Rank 1 (log_T-KD) is a deployable, Wilcoxon-strict reg lift TODAY

3 states × 3 KD weights = 9 cells, all Wilcoxon-strict at p=0.0312 (5/5 folds positive in every cell). Cat untouched at disjoint at every state.

Mean Δ at W=0.2 on disjoint reg: **AL +2.40 / AZ +5.06 / FL +2.32 pp**. On deploy (geom_simple): **AL +2.92 / AZ +5.37 / FL +4.21 pp**. On legacy b9: AL +2.27 / AZ +1.40 / **FL +5.79 pp + σ collapse 9.22 → 0.70** (the F4 bimodality fully disappears).

Mechanism: the head already consumes `log_T` as an additive prior; the KD term forces the head's **output distribution** to also match `log_T` — a second pressure that accelerates prior-alignment and stabilises the deployable selector.

Recommend FL multi-seed promotion (n=20) in the next-tier paper revision.

## Headline finding — Rank 4 (composite) is a deployable substrate-capacity recipe TODAY

Composite = **STL c2hgi for cat + STL HGI for reg, routed by task at deploy time**. Both component models are already shipping artefacts. Pure inference-side recipe, zero retraining.

Reg lift vs MTL @ disjoint: AL +11 / AZ +12 / CA +7 / TX +10 / FL +7-12 (single-seed)*. Cat untouched (composite cat = MTL c2hgi cat).

This IS the substrate-capacity ceiling at the current architecture. Distinct from merge_design Lever 6 (integrated single-backbone, FALSIFIED 2026-05-06). New future-work memo: [`composite_two_substrate_engine.md`](../../future_works/composite_two_substrate_engine.md).

*FL composite under stale-log_T was 71.34 single-seed; multi-seed STL HGI reg at FL is not on file (5-6 GPU-h to close).

## Headline finding — Rank 2 (class-balanced sampler) FALSIFIED — important null result

Adding `WeightedRandomSampler` to the reg dataloader (alongside the existing weighted-CE) **regresses disjoint reg by 18-30 pp at AL/AZ**, p=1.0000 in both. FL skipped after AL (−30 pp leaves no plausible recovery).

This closes the "long-tail rare-class under-sampling drives reg destabilisation" hypothesis. Combined with P4 (frozen-cat falsification), it leaves only the architecture as the residual mechanism — which is precisely what `mtl_architecture_revisit.md` will test.

## What this pass did NOT cover (deferred for next-tier)

Per [`PRIORITY_IMPACT.md`](../../studies/archive/mtl-protocol-fix/PRIORITY_IMPACT.md):
- Rank 5 — §4.4-partial freeze-reg-after-peak (P4 already says cat isn't the bottleneck; symmetric reg test untested)
- Rank 6 — §4.7 Designs J/B re-eval under F1 (40-60 GPU-h, substrate parquets need rebuild)
- Rank 7 — merge_design Levers 4/5 (substrate axis, Lever 6 already FALSIFIED in same gap)
- Rank 8 — §4.8 POI decoder with HGI-emb target (new memo [`poi_decoder_hgi_distill.md`](../../future_works/poi_decoder_hgi_distill.md) drafted)

These are speculative or compute-heavy with low EV; defer to the next-tier studies.

## Pointers

- Findings docs: [`phase3_rank1_findings.md`](phase3_rank1_findings.md), [`phase3_rank2_findings.md`](phase3_rank2_findings.md), [`phase3_rank4_composite_analysis.md`](phase3_rank4_composite_analysis.md)
- Per-state raw summaries: `docs/results/mtl_protocol_fix/phase3_rank{1_log_t_kd, 2_balanced_sampler}/{alabama,arizona,florida}/{state}_summary.{md,json}`
- Ranking + execution decisions: [`docs/studies/archive/mtl-protocol-fix/PRIORITY_IMPACT.md`](../../studies/archive/mtl-protocol-fix/PRIORITY_IMPACT.md)
- Inventory of deferred items: [`docs/studies/archive/mtl-protocol-fix/DEFERRED_WORK.md`](../../studies/archive/mtl-protocol-fix/DEFERRED_WORK.md)
- Execution log: [`docs/studies/archive/mtl-protocol-fix/log.md`](../../studies/archive/mtl-protocol-fix/log.md) 2026-05-21 entries
- Code: `src/configs/experiment.py`, `src/training/runners/mtl_cv.py`, `scripts/train.py`, `src/data/folds.py`, `scripts/mtl_protocol_fix/`
