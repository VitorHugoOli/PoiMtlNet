# User Considerations — mtl-protocol-fix study

> Captured 2026-05-20 from the deep-dive post-closure review of canonical_improvement (the user's verbatim questions and concerns, condensed only where syntax required).

## Conceptual questions (the study must address these explicitly)

### Q1 — STL substrate paradox

> "OK if the check2hgi is a load-bearing of the next-reg, can you explain me why once both is executed in a same STL head for next-reg the HGI still wins?"

**The framing**: check2hgi vs HGI on STL next-reg shows HGI ahead by 1.6-3.1 pp (`RESULTS_TABLE.md §0.3`). If check2hgi were load-bearing for both heads, this shouldn't happen.

**Resolution (Study Q1, must be explicit in the paper §Discussion)**:
- Check2HGI is load-bearing for **next-cat**: per-visit contextual embedding is what category prediction needs (+15 pp small-states, +28-29 pp large states on cat STL F1).
- HGI is marginally load-bearing for **next-reg STL**: POI-stable spatial manifold (Delaunay POI-POI graph) is what region prediction needs (+1.6-3.1 pp).
- The two substrates encode different inductive biases. Tier 4.4 (Delaunay edges in c2hgi) and T5.2a (Node2Vec POI-POI 4th boundary) both failed to inject HGI's spatial signal into c2hgi without breaking cat — confirming the two are mechanism-incompatible at the c2hgi substrate level.
- **Critical**: §0.3 STL reg gap is **small (1.6-3.1 pp)**. §0.1 MTL reg gap to matched-head STL is **large (7-17 pp)**. The MTL-vs-STL component is what this study targets; the STL substrate-Δ is a separate, smaller axis.

### Q2 — Beyond protocol/selector

> "About protocol/selector have we explore if had other possible problem beyond Protocol/selector?"

**Coverage matrix** (study must validate / extend):

| Axis | Status pre-study | This study's role |
|---|---|---|
| Substrate axis | Exhausted (canonical_improvement Tier 1-6) | Out of scope; closed |
| Protocol/selector | Partial (C21 at FL single-seed only) | **IN SCOPE — Phase 1** |
| MTL loss balancing | Not under leak-free per-fold log_T | Out of scope → `substrate_adaptive_mtl_balancing.md` future work |
| MTL architecture | Very-simple-variants only (legacy) | Out of scope → `mtl_architecture_revisit.md` future work |
| Head architecture | Grandfathered from pre-leak-fix P1 | Out of scope → `head_window_batch_audit.md` future work |
| Window construction + masks | Never audited | Out of scope → `head_window_batch_audit.md` future work |
| Batch class-balance | Never tested | Out of scope → `head_window_batch_audit.md` future work |

**The study's residual-gap characterisation** (Phase 3) explicitly quantifies how much of the MTL-vs-STL reg gap survives F1, which feeds into deciding which of the future-work tracks above to launch next.

### Q3 — Three-frontier MTL evaluation

> "Also for the MTL we should produce two protocols/selectors the result the best disjoint and the best joint (I believe in the canonical experiment we have produced a formula for the joint) for a same model so we can compare a MTL model in three frontiers."

**Adopted as the study's primary methodological deliverable**:

For each MTL model trained, report three numbers per state per task:
1. **MTL @ best joint** — single-checkpoint deployable number, under a principled joint selector (`joint_geom_simple = sqrt(cat_f1 × reg_top10_indist)` OR `joint_geom_lift` already coded at `src/training/runners/mtl_cv.py:710`).
2. **MTL @ best disjoint** — substrate-capacity ceiling (cat from cat-best epoch, reg from reg-best epoch).
3. **STL matched-head ceiling** — single-task best.

This three-frontier reporting becomes the new paper-grade protocol; replaces the implicit single-selector reporting that obscured C21.

### Q4 — STL head metric correctness

> "How the STL heads is working they should been get the best results during the metrics"

**Confirmation gate** (Phase 0 — pre-flight): verify STL `next_gru` (cat) and `next_stan_flow` (reg) at FL/AL/AZ pick per-task best epoch (no joint constraint, since STL is single-task). Compare extracted numbers to `RESULTS_TABLE.md §0.1` STL columns to confirm STL ceiling is already at best-epoch.

If STL is also reported at a constrained selector (would be surprising), the same F1 fix logic applies to STL as well.

## Inherited from canonical_improvement closure

- **The substrate axis is closed.** Do not re-open Tier 1-6 mechanisms. Read `docs/studies/canonical_improvement/log.md` 2026-05-19 final entry first; falsified-history is off-limits.
- **The pre-flight leak-probe protocol still applies** to any new training-touching code change (T1.1 protocol at `scripts/canonical_improvement/ijm_leak_probe.py`). No substrate change is expected in this study, but if any code path is added (e.g. a new selector that needs new metrics computation), confirm leak floor is unchanged.

## Inherited from mtl-exploration

- Predecessor F1/F2/F3 memo at [`../mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md`](../mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md). F2/F3 portions are split into the future-work memos under `docs/future_works/`. **F1 is this study's Phase 1.**
- The `EXPERIMENT_HGI_SUBSTRATE.md` finding (MTL+HGI ≡ STL+HGI on both heads) is a critical reference — if c2hgi+v3c+T3.2 reg under per-task disjoint matches HGI reg, the reg "gap" might already be closed in capacity, only obscured by the selector.

## Pointers

- C21 source-of-truth: `docs/CONCERNS.md` C21
- CH23-A/B paper claim status: `docs/CLAIMS_AND_HYPOTHESES.md`
- Predecessor study closure log: `docs/studies/canonical_improvement/log.md` 2026-05-19 final entry
- Predecessor F1/F2/F3 memo: `docs/studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md`
- Selector bug site: `src/training/runners/mtl_cv.py:679`
- Already-coded alternative selector: `src/training/runners/mtl_cv.py:710` (`joint_geom_lift`)
- Analysis tool (zero retraining): `scripts/canonical_improvement/analyze_t64_selectors.py`
- Matched-protocol source-of-truth: `docs/results/canonical_improvement/T6_4_dual_selector_final.{json,md}`
