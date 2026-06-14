# Defensible Future Work

> Compiled 2026-06-12. Ordered by defensibility given the evidence base; each item states what it builds on and what would have to be true for it to pay off.

## 1. Next-POI extension via the region/category heads (most defensible)

HMT-GRN (SIGIR 2022) showed region distributions can power **hierarchical beam search** over P(region, POI): predict region, then rank POIs within predicted regions. This repo already has trained region heads and category heads — the natural extension is *category- and region-conditioned next-POI prediction*: P(POI | history) ≈ P(region|h) · P(category|h) · P(POI | region, category, h). This (a) directly answers the "amputated next-POI" objection by showing the two tasks compose into the canonical task, (b) inherits HMT-GRN's sparsity argument, and (c) gives the first protocol-bridgeable comparison against the next-POI literature. Prerequisite: the temporal-split bridge (roadmap A5), since next-POI numbers are only meaningful under field protocol.

## 2. Inductive Check2HGI (fixes transductivity + enables deployment claims)

The current substrate cannot embed a check-in that arrives after training ([`evaluation_protocol_review.md §4.1`](evaluation_protocol_review.md)). An inductive variant — encode a new check-in from its features + frozen POI/region tables, GraphSAGE-style or via the attention-pooling path — would (a) eliminate the leakage question structurally, (b) make production-inference claims legitimate, and (c) be a genuine methodological delta over both HGI (transductive) and CTLE (inductive but graph-free). The roadmap's A4 ablation tells you how much headroom this matters for.

## 3. Embedding alignment toward next-POI

If the substrate is to serve next-POI later, the embedding objective can be aligned now: add a 4th boundary (check-in ↔ next-POI/next-region contrastive term) so the infomax objective is predictive rather than purely structural. The repo's v13/v14 levers (POI2Vec@pool, Delaunay-POI-GCN) are early instances of this on the region path; the established "regime finding" (substrate improvements wash out under MTL joint training) warns that such levers should be validated at STL first and may only pay off via the composite/routing deployment pattern (`docs/future_works/composite_two_substrate_engine.md`).

## 4. Stronger spatial-temporal modeling — targeted, not exploratory

The internal evidence says the wall is the joint-training regime, not the heads. Two directions remain live: (a) **prior-pathway work** — log_T-KD is the one confirmed MTL-reg lever; richer transition priors (time-conditioned log_T, distance-decayed transitions à la Flashback) extend a proven channel; (b) **target-time conditioning** (ROTAN, KDD 2024 showed large gains from injecting *when* the next event happens) — absent from the current heads and cheap to test.

## 5. External validation

In order of value: (i) one standard-benchmark city (FSQ-NYC/TKY or Massive-STEPS) with category+region targets — external anchor; (ii) a second LBSN source (Foursquare or Weeplaces US cuts) to test Gowalla-specificity; (iii) cross-state transfer (train FL → test TX substrate) to probe whether the substrate learns transferable structure or per-state memorization. The user-disjoint protocol already supports a cold-user story; cross-region transfer would complete it.

## 6. The category+region framing as a privacy-preserving prediction setting

If the task pairing is to be the lasting contribution rather than a workaround, formalize it: venue-anonymous mobility prediction (no POI IDs at inference) with utility benchmarks against POI-level systems degraded to tract/category outputs. This converts the "amputated" criticism into a feature and has no direct published competitor ([`literature_review.md §4`](literature_review.md), negative search result).

## 7. Committed change (user, 2026-06-12): overlapping windows — guidance

A future-work memo `overlapping_windows.md` is planned. Three evidence-grounded cautions to bake into it:

1. **The rising-tide rule applies.** Denser supervision via overlap was already tested as an MTL-gap lever (R1 in `docs/studies/archive/mtl_improvement/`, see `docs/results/mtl_improvement/R1_R2_audit.md`): it lifts the STL ceiling by the same amount it lifts MTL. Expect overlapping windows to raise *absolute* numbers and training-data density, **not** to change MTL-vs-STL deltas. Frame the change as base-strengthening, not as an MTL lever.
2. **It changes the evaluation base, so it must precede the freeze.** Window construction alters every sample count, fold composition, and paired-test unit. It belongs in `closing_data` pre-freeze decisions (P2 gate), never as a post-hoc change — otherwise every n=20 cell regenerated in P3 is invalidated.
3. **Leakage shape stays safe but statistics change.** With user-disjoint folds there is no cross-user window-overlap leakage; however, overlapping windows from the same user are correlated samples — fold-level paired statistics remain valid, but any per-window significance computation would overstate effective n. Also re-derive the per-fold log_T after the change (stale-artifact lesson, `docs/CONCERNS.md`).

Side benefit worth claiming: stride-1 (or all-prefixes) windowing moves the protocol *closer* to the literature standard (STAN-style prefix samples), removing one reviewer objection ([`evaluation_protocol_review.md §4.4`](evaluation_protocol_review.md)).

## 8. Committed change (user, 2026-06-12): second dataset — recommendation

Goal: break Gowalla-specificity and ideally buy a protocol bridge at the same time. Verified options (details in [`mtl_frontier.md` survey side-questions](mtl_frontier.md) / [`references.md`](references.md)):

| Candidate | Categories | Coords (tract-mappable) | Recency | Splits shipped | Notes |
|---|---|---|---|---|---|
| **Massive-STEPS NYC** ([arXiv:2505.11239](https://arxiv.org/abs/2505.11239), [GitHub](https://github.com/cruiseresearchgroup/Massive-STEPS)) | ✅ fine-grained Foursquare | ✅ (US city → TIGER tracts) | 2017–18 | ✅ train/val/test | 6,929 users / 272k check-ins; Apache-2.0/CC-BY; 14 more cities for cross-city work (non-US → use H3 cells) |
| FSQ TSMC-2014 NYC/TKY ([source](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)) | ✅ (~250 fine; roots via external taxonomy join) | ✅ NYC; TKY needs H3/grid | 2012–13 | ❌ (community-standard 80/10/10) | The canonical trio anchor; maximum literature comparability; old data |
| Weeplaces | ⚠ partial coverage | ✅ | 2003–11 | ❌ | Category gaps unquantified — verify raw CSV before committing; weakest option |

**Recommendation: Massive-STEPS NYC as the primary second dataset.** It is the only option that simultaneously (a) is a different LBSN source than Gowalla, (b) ships per-check-in fine-grained categories *and* coordinates (so both the 7-class rollup and TIGER-tract regions reproduce), (c) ships **temporal train/val/test splits — so running it doubles as the temporal-split protocol bridge** (roadmap A5) for free, and (d) is modern (2017–18), pre-empting the "2012-era data" critique that Massive-STEPS itself was built to make. Add **FSQ-TKY (TSMC-2014)** as an optional third corpus only if a non-US robustness point is wanted (regions = H3 cells there, which also tests tract-choice sensitivity).

Two design notes: (i) build a documented Foursquare→7-root category mapping so the category task is commensurable with the Gowalla states (or, better, report both 7-class macro-F1 and fine-grained Acc@k); (ii) scope the second dataset as a **validation phase, not a full matrix** — champion G + STL ceilings + Markov floor + (CTLE if available), one city, 4 seeds — not the full closing_data regeneration.

## 9. Not recommended as future work

- Scaling the MTL-optimizer search (saturated; Kurin/Xin predict the null).
- LLM-based prediction as a core contribution (the frontier moves monthly; this repo's comparative advantage is controlled substrate/regime analysis, not LLM engineering — keep LLMs as a reference row only).
- Further symmetric-towers architecture variants (G′ falsified).
