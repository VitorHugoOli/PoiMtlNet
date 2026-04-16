# Claims and Hypotheses Catalog — Check2HGI Study

This is the **authoritative list** of every claim/hypothesis this study intends to validate. Same shape as the fusion study's catalog, but scoped to **next-POI + next-region prediction on check-in-level contextual embeddings**.

Each entry has:
- **ID** (`CH01`, `CH02`, …) for stable referencing (note: `CH` prefix distinguishes from fusion's `C##` — they live in the same repo).
- **Statement** — what we want to show.
- **Source** — why we care.
- **Test** — which experiment(s) validate it.
- **Phase** — where the test lives.
- **Status** — `pending` / `running` / `confirmed` / `refuted` / `partial` / `abandoned` / `declared`.
- **Evidence** — pointer to `docs/studies/check2hgi/results/.../` once tested.
- **Notes** — caveats, surprises, follow-ups.

**Rule:** no claim enters the paper without `status ∈ {confirmed, partial}` and an evidence pointer.

---

## Tier A — Headline paper thesis

Three claims; the paper lives or dies by them. Every other tier either supports interpretation (B, D) or constrains scope (E).

### CH01 — Check2HGI (check-in-level) beats HGI (POI-level) on single-task next-POI

**Statement:** In the single-task setting, training a `next_mtl` transformer head on a 9-window of Check2HGI check-in-level embeddings achieves higher Acc@10 and MRR on next-POI prediction than the same head trained on HGI POI-level embeddings, at matched 5-fold user-held-out CV, matched 50 epochs, matched batch, on Alabama and Florida.

**Source:** Paper thesis — contextual check-in vectors should encode trajectory phase (morning commute vs evening leisure vs weekend vs weekday) that POI-level vectors blur out.

**Test:** P1.1 — run single-task next-POI on {HGI, Check2HGI} × {AL, FL}. Report Acc@{1,5,10}, MRR, NDCG@{5,10}. Report bootstrapped 95% CI across folds.

**Phase:** P1.

**Status:** `pending`.

**Notes:** Negative result is still publishable — "contextual embeddings help only on high-cardinality sequence tasks, not on per-POI classification" (see also CH02). If CH01 refutes AND CH02 confirms, the paper reframes around the MTL contribution. Training budget must be matched to avoid the confound identified by the critical-review agent (see `docs/studies/check2hgi/archive/v1_wip_mixed_scope/TRAINING_BUDGET_DECISION.md`).

---

### CH02 — Next-region auxiliary task improves next-POI under MTL on Check2HGI (headline)

**Statement:** On Check2HGI embeddings, the 2-task MTL `{next_poi, next_region}` raises next-POI Acc@10 over the single-task next-POI baseline by a margin exceeding the 5-fold CI, on AL and FL. MTL balancer: NashMTL with `val_joint_lift` monitor.

**Source:** HMT-GRN (SIGIR '22), MGCL (Frontiers '24), Bi-Level GSL (arXiv '24), Learning Hierarchical Spatial Tasks (TORS '24) all report lift from hierarchical region-level auxiliaries on next-POI tasks. We test whether the same mechanism transfers to Check2HGI's contextual embeddings.

**Test:** P2.1 — compare (a) single-task next-POI on Check2HGI (the CH01 winner) vs (b) 2-task MTL `{next_poi, next_region}` on Check2HGI. 5-fold CV, 50 epochs, NashMTL.

**Phase:** P2.

**Status:** `pending`.

**Notes:** This is the paper's headline comparison. If CH02 is negative on both states, the narrative shrinks to CH01 alone and the paper becomes "check-in-level embeddings help next-POI; auxiliary tasks don't add signal on this data." Still publishable, weaker contribution.

---

### CH03 — No per-head negative transfer under MTL

**Statement:** Under the 2-task MTL from CH02, each head's validation metrics are ≥ its single-task baseline by a margin larger than the 5-fold CI:
- next_POI under MTL ≥ next_POI single-task (Acc@10 and MRR)
- next_region under MTL ≥ next_region single-task (Acc@10 and MRR)

**Source:** Critical-reviewer concern — without this control, a composite "joint_lift goes up" claim is confounded with one head improving at the other's expense.

**Test:** P2.2 — single-task next-region head must also be run (P1.2), then compared against its MTL counterpart.

**Phase:** P2.

**Status:** `pending`.

**Notes:** The fusion study's analogous claim is C28. A negative result on either head is the #1 reviewer ask; plan for it by (a) reporting it honestly, (b) proposing a mitigation (class-weighted CE, especially on Florida where next_region has a 22.5% majority class), and (c) showing whether mitigation rescues the head.

---

## Tier B — Methodology & supporting claims

### CH04 — Next-region is a meaningful task (not a noise-injection regulariser)

**Statement:** Single-task next-region prediction on Check2HGI embeddings achieves Acc@1 at least 2× the majority-class baseline on both states (AL majority ~2.3% → learned ≥ 4.6%; FL majority ~22.5% → learned ≥ 45%). Acc@10 ≥ Acc@1 by a meaningful margin.

**Source:** Sanity / anti-debugging. If next-region is trivial to saturate at majority-class, any lift in CH02 could be a regularisation artefact (noise injection), not genuine task signal.

**Test:** P1.2 — single-task next-region on Check2HGI. Report Acc@{1,5,10}, MRR, majority-class baseline.

**Phase:** P1.

**Status:** `pending`.

**Notes:** From the 2026-04-15 probing experiment (linear-probe-from-embedding → region), we already know the signal is non-trivial on AL (3.4× majority) and inconclusive on FL under linear probing (1.04× — but a transformer sequence model should do much better).

---

### CH05 — Ranking metrics discriminate where macro-F1 collapses

**Statement:** On both heads (next_poi and next_region), Acc@10 and MRR differentiate Check2HGI from HGI with a statistically significant margin, while macro-F1 on either head is effectively zero for both engines (the label space is too large and sparse for class-balanced F1 to separate).

**Source:** Methodology / justification. HMT-GRN and MGCL use ranking metrics precisely because macro-F1 is uninformative at ~10³–10⁵ classes with sparse support. Our `compute_classification_metrics` already has the hand-rolled high-cardinality path (see `src/tracking/metrics.py::_handrolled_cls_metrics`).

**Test:** P1.1 output — both macro-F1 and Acc@K/MRR reported per engine. Differences compared.

**Phase:** P1.

**Status:** `pending`.

**Notes:** Paper methods section — explicitly documents why ranking metrics are primary and macro-F1 is reported only for completeness.

---

## Tier C — Region-input mechanism (gated)

### CH06 — Region embeddings as input improve next-POI Acc@10

**Statement:** Using dual-stream input (concatenating the region embedding of each check-in's POI to the check-in embedding per timestep, doubling the per-timestep feature width from 64 to 128) improves next-POI Acc@10 on Check2HGI over the check-in-only input baseline, on at least one of Florida or Alabama.

**Source:** Probe experiment (AL/AZ/FL linear-probe of check-in emb → region showed 3.4× / 1.7× / 1.04× majority lift respectively). The FL result suggests that at high region cardinality, check-in embeddings lose region signal — dual-stream input should help most on FL.

**Test:** P3.1 — same P2-champion backbone + optimiser, two input variants (check-in-only vs check-in⊕region-emb). 5-fold × 50ep.

**Phase:** P3.

**Status:** `pending`.

**Notes:** See `docs/studies/check2hgi/archive/v1_wip_mixed_scope/CRITICAL_REVIEW.md §3.6` for the empirical probe that motivates this claim.

---

### CH07 — Bidirectional cross-attention > concat (gated on CH06)

**Statement:** At matched parameter budget, a new `MTLnetCrossAttn` architecture with two cross-attention layers between the check-in stream and the region-embedding stream achieves higher Acc@10 on next_POI than the concat baseline (P3.1 winner), on FL.

**Source:** HMT-GRN / Bi-Level GSL use explicit cross-level attention, not just concatenation. Tests whether bidirectional information flow between streams captures structure that static concatenation misses.

**Test:** P4.1 — `MTLnetCrossAttn` vs concat baseline, same data, matched parameter count.

**Phase:** P4 (only runs if CH06 shows ≥ 2pp Acc@10 lift on FL in P3).

**Status:** `pending (gated)`.

**Notes:** Full architecture spec was in `archive/v1_wip_mixed_scope/OPTION_C_SPEC.md`. Separate model class; legacy MTLnet untouched.

---

### CH11 — Region-input gain is state-dependent

**Statement:** The Δ Acc@10 from adding region embeddings as input (CH06 or CH07, whichever is tested) is larger on Florida (4,703 regions, 22.5% majority) than on Alabama (1,109 regions, 2.3% majority) by at least 2×.

**Source:** Probe-experiment trend — linear-recovery lift of region from check-in embedding shrinks monotonically with region cardinality. The full transformer's lift should follow.

**Test:** P3 comparison across both states.

**Phase:** P3.

**Status:** `pending`.

**Notes:** CH11 is paper-valuable even if CH06 / CH07 are partial — it characterises *when* hierarchical-input-representation matters.

---

## Tier D — Ablations (light, not a full legacy port)

### CH08 — `next_mtl` head outperforms simpler sequence heads on next-POI

**Statement:** Among sequential head variants in the registry (`next_mtl`, `next_lstm`, `next_gru`, `next_tcn_residual`, `next_temporal_cnn`), `next_mtl` (4-layer transformer + causal + attention pool) achieves the highest Acc@10 on next_POI head at matched capacity.

**Source:** Prior single-task experiments on the fusion track (C08 analogue) showed transformer heads dominate on high-cardinality sequence prediction; test whether this replicates on Check2HGI.

**Test:** P5.1 — swap `next_poi` head across 5 candidates; keep everything else at the P2 champion. Alabama only.

**Phase:** P5.

**Status:** `pending`.

**Notes:** Cheap ablation (same fold data, same backbone, different head). If `next_mtl` is not the winner, use the winner as the P2 champion retroactively.

---

### CH09 — NashMTL vs equal_weight vs CAGrad on this task pair

**Statement:** On Check2HGI `{next_poi, next_region}`, NashMTL achieves `val_joint_lift` ≥ equal_weight and ≥ CAGrad by a margin larger than the 5-fold CI.

**Source:** Fusion track's C02/C03 analogue. Next_POI and next_region have very different label-space magnitudes (ratio ~10× in cardinality, larger in loss magnitude) — gradient-surgery optimisers may help. Or equal_weight may suffice (the Xin et al. finding on single-source embeddings).

**Test:** P5.2 — same fold data, same arch, 3 MTL criteria: `nash_mtl`, `equal_weight`, `cagrad`.

**Phase:** P5.

**Status:** `pending`.

**Notes:** If equal_weight is within CI, drop NashMTL (cheaper + fewer dependencies — cvxpy/ECOS solver warnings).

---

### CH10 — Seed variance bounds

**Statement:** The 5-fold standard deviation of Acc@10 for next_POI on Check2HGI + MTL + P2 champion is < 2 pp across seeds {42, 123, 2024}.

**Source:** Methodology — a paired-test claim of "+3 pp from MTL" is meaningless if seed variance is ±4 pp. Pre-register variance floor.

**Test:** P5.3 — replicate P2 champion across 3 seeds on Alabama.

**Phase:** P5.

**Status:** `pending`.

**Notes:** If seed variance is high, downgrade "confirmed" thresholds across the catalog and publish CIs explicitly. This is the bound referenced in CH02 and CH06 "margin larger than CI" language.

---

## Tier E — Limitations & declared scope

### CH12 — Gowalla state-level results do not transfer to FSQ-NYC/TKY

**Statement:** Because Check2HGI relies on census-tract region assignment (US-only in the current pipeline), results on Gowalla-AL/FL are not directly comparable to reported HMT-GRN / MGCL / GETNext numbers on FSQ-NYC/TKY or Gowalla-global. Our numbers sit in the Gowalla-state sibling lineage.

**Source:** External-validity disclosure required by any reviewer who knows the next-POI literature.

**Test:** N/A — declared scope limitation, documented in paper threats-to-validity.

**Phase:** —.

**Status:** `declared`.

**Notes:** Follow-up work (future paper): port the region-definition pipeline to FSQ-NYC/TKY using administrative units or neighbourhood polygons.

---

### CH13 — Encoder enrichment (temporal/spatial features, hard negatives, multi-view contrastive) is deferred

**Statement:** This paper reports results on vanilla Check2HGI (fixed 4D temporal features, user-sequence edges, random negatives, 3-boundary MI loss at default α). The four phases of `docs/issues/CHECK2HGI_ENRICHMENT_PROPOSAL.md` are identified as future work.

**Source:** Scope discipline — attributing any lift requires a vanilla baseline first.

**Test:** N/A — declared scope.

**Phase:** —.

**Status:** `declared`.

**Notes:** Bundle enrichment into a follow-up paper; do not mix with the MTL-task-pair contribution.

---

### CH14 — Check2HGI's relationship to the HGI fclass-identity shortcut (audit)

**Statement:** Check2HGI embeddings are evaluated for inheritance of the fclass → category 1:1 determinism shortcut that main's commit `19b9c2b` identified in HGI preprocessing. Specifically: under an fclass-shuffle ablation (arm C analogue, if applicable to check2HGI preprocessing), next-POI Acc@10 drop is **smaller** than next-category F1 drop — demonstrating that check2HGI's representation does not collapse to the fclass shortcut.

**Source:** Reviewer finding from main (C29 in fusion study): HGI + POI2Vec pipeline on Gowalla preserves a 1:1 fclass→category mapping via POI2Vec's fclass-level embedding sharing, which makes POI-category classification trivially solvable. Check2HGI's preprocessing uses a different POI2Vec path (or none) — we must verify empirically.

**Test:** P0.2 — inspect Check2HGI preprocessing for the shortcut; if present, replicate the arm C fclass-shuffle ablation on Check2HGI single-task next-POI and next-region (not next-category, which is not our task). Compare the Δ Acc@10 to the fusion study's arm-C results.

**Phase:** P0.

**Status:** `pending`.

**Notes:** If the shortcut IS present in Check2HGI and it affects next-POI prediction (not just next-category), our headline comparison may inherit a confound. This claim is the first thing to check in P0.

---

## Summary dashboard

| ID | Tier | Phase | Status | Decides |
|----|------|-------|--------|---------|
| CH01 | A | P1 | pending | Embedding-quality claim (single-task next-POI) |
| CH02 | A | P2 | pending | MTL lift (headline) |
| CH03 | A | P2 | pending | No per-head negative transfer |
| CH04 | B | P1 | pending | Next-region is meaningful |
| CH05 | B | P1 | pending | Ranking metrics > macro-F1 methodology |
| CH06 | C | P3 | pending | Region emb as input helps |
| CH07 | C | P4 (gated) | pending | Cross-attention > concat |
| CH08 | D | P5 | pending | Head architecture ablation |
| CH09 | D | P5 | pending | MTL optimiser ablation |
| CH10 | D | P5 | pending | Seed variance bound |
| CH11 | C | P3 | pending | State-dependent gain |
| CH12 | E | — | declared | External-validity limit |
| CH13 | E | — | declared | Enrichment out of scope |
| CH14 | E | P0 | pending | Fclass shortcut audit |

**Deferred (not in this study; captured in `docs/studies/check2hgi/archive/v1_wip_mixed_scope/` for reference):**
- Next_time_gap auxiliary task.
- Expert-gating MTL architectures (CGC/MMoE/DSelect-K).
- Full P5-legacy arch×optim grid (5×20).
- Frozen-backbone head-swap co-adaptation probe.
