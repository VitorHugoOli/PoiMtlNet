# Claims and Hypotheses Catalog — Check2HGI Study

This is the **authoritative list** of every claim/hypothesis this study intends to validate. Same shape as the fusion study's catalog, but scoped to **next-POI + next-region prediction on check-in-level contextual embeddings** — a **standalone** study with no cross-engine comparisons (no HGI, no fusion) and no replication of prior work (no CBIC baselines).

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

Three claims. The paper lives or dies by them. Every other tier either supports interpretation (B, C), characterises sensitivity (D), or constrains scope (E).

### CH01 — Next-region auxiliary task improves next-POI prediction under MTL on Check2HGI (HEADLINE)

**Statement:** On Check2HGI check-in-level embeddings, the 2-task MTL `{next_poi, next_region}` raises next-POI Acc@10 over the single-task next-POI baseline by a margin exceeding the per-seed × per-fold CI, on Alabama and Florida. Comparison uses **3 seeds × 5 folds** (n=15 paired samples) via Wilcoxon signed-rank, α=0.05. Monitor: `val_joint_geom_lift` (geometric mean of per-head lifts over majority baseline).

**Source:** HMT-GRN (SIGIR '22), MGCL (Frontiers '24), Bi-Level GSL (arXiv '24), Learning Hierarchical Spatial Tasks (TORS '24) all report lift from hierarchical region-level auxiliaries on next-POI tasks. This study tests whether the same mechanism transfers to Check2HGI's contextual check-in-level embeddings on Gowalla state-level data.

**Test:** P2.1 — compare (a) single-task next-POI on Check2HGI (from P1) vs (b) 2-task MTL `{next_poi, next_region}` on Check2HGI. 3 seeds × 5-fold user-held-out StratifiedGroupKFold, 50 epochs, NashMTL.

**Phase:** P2.

**Status:** `pending`.

**Notes:** Per-head tasks and the comparison run on frozen folds so the Wilcoxon pair is valid. If CH01 refutes on both states, the paper reframes around CH03 (dual-stream input helps next-POI even without MTL) or CH04/CH05 (methodology). If it refutes on only one state, flag state-dependent behaviour and investigate before publication. See MASTER_PLAN §Exit criteria for the "minimum viable paper" positioning.

---

### CH02 — No per-head negative transfer under MTL

**Statement:** Under the 2-task MTL from CH01, each head's validation metrics are ≥ its single-task baseline by a margin larger than seed × fold CI:
- next_POI under MTL ≥ next_POI single-task (Acc@10 and MRR)
- next_region under MTL ≥ next_region single-task (Acc@10 and MRR)

**Source:** Critical-reviewer concern — without this control, a composite "joint metric goes up" claim is confounded with one head improving at the other's expense.

**Test:** P2.2 — single-task next-region head must also be run (P1.2), then compared against its MTL counterpart with the same 3 × 5 pairing as CH01.

**Phase:** P2.

**Status:** `pending`.

**Notes:** Florida has a 22.5% majority next-region class; `--use-class-weights` is the pre-registered mitigation. If CH02 still fails on FL with class weights, document as a data-specific limitation, not a method failure. This is the #1 reviewer ask for any MTL paper; plan for the negative result.

---

### CH03 — Region embeddings as an explicit input stream improve next-POI over check-in-only input

**Statement:** Feeding the region embedding of each timestep's POI as a parallel input stream (dual-stream concat, `[B, 9, 128]` instead of `[B, 9, 64]`) improves next-POI Acc@10 over the MTL baseline from CH01, on at least one of Alabama or Florida, at matched training budget.

**Source:** HMT-GRN / Bi-Level GSL use region-level features as explicit model input, not just as auxiliary supervision. Tests whether giving the encoder direct region information (beyond what the shared backbone learns from the next-region auxiliary head) helps the POI task.

**Test:** P3.1 — same P2-champion backbone + optimiser, two input variants (check-in-only vs check-in⊕region-emb). 3 seeds × 5 folds × 50 epochs.

**Phase:** P3.

**Status:** `pending`.

**Notes:** CH03 is independent of CH01 — dual-stream can help even if MTL doesn't, and MTL can help even if dual-stream doesn't. The paper's strongest positioning has both confirmed.

---

## Tier B — Methodology & supporting claims

### CH04 — Check2HGI learned models beat simple baselines by ≥ 2× on both heads

**Statement:** On both next-POI and next-region tasks, every Check2HGI-based learned model (single-task or MTL) achieves Acc@10 at least 2× the strongest non-learned baseline (majority-class, random, 1-step Markov, user-history top-K) on both Alabama and Florida.

**Source:** Pipeline-correctness floor. If a learned model doesn't beat 1-step Markov on its own data, something is fundamentally broken before any paper claim is made. This is the "known-good reference" equivalent to fusion's P0.4 CBIC replication — except instead of replicating an external number, we verify our learned model actually learns something over trivial baselines on our own data.

**Test:** P0.5 (baselines) + P1.1/P1.2 (single-task) + P2.1 (MTL). Every row's Acc@10 compared against the best simple baseline in the same `results/P0/simple_baselines/<state>/` summary.

**Phase:** P0 + P1 + P2 (ongoing check).

**Status:** `pending`.

**Notes:** Simple baselines are computed once in P0.5 and referenced throughout. The paper's main experimental table starts with a simple-baselines row so readers see the floor explicitly.

---

### CH05 — Ranking metrics discriminate where macro-F1 collapses

**Statement:** On both heads, Acc@{1, 5, 10} and MRR differentiate model variants with margins that match the practical-significance threshold set in MASTER_PLAN, while macro-F1 on either head is effectively zero-variance across variants (label spaces of 10³–10⁵ classes with sparse support).

**Source:** Methodology justification. HMT-GRN, MGCL and the broader next-POI literature use ranking metrics precisely because macro-F1 is uninformative at high cardinality. This claim empirically documents that on our specific data.

**Test:** Every P1 / P2 run reports both macro-F1 and Acc@K / MRR. Differences compared.

**Phase:** P1.

**Status:** `pending`.

**Notes:** Paper methods section — explicitly documents why ranking metrics are primary and macro-F1 is reported only for completeness. Saves us from a reviewer asking "why not F1?".

---

### CH06 — Check2HGI learned models beat simple baselines on OOD-restricted Acc@K (guards against train-memorisation artefact)

**Statement:** When Acc@K is restricted to sequences whose target POI also appears in the current training fold ("in-distribution" POIs), Check2HGI models still beat simple baselines (CH04 standard). Equivalently: the learned Acc@K on the in-distribution subset is not a trivial consequence of memorising training POIs.

**Source:** Critical-review concern: StratifiedGroupKFold holds users out, but next-POI labels are POI indices. Some val users visit POIs that no training user visited — for those sequences, Acc@K is mechanically 0 regardless of model quality. Report both raw Acc@K (for completeness) and OOD-restricted Acc@K (for the defensible comparison).

**Test:** Every P1 / P2 / P3 / P4 run reports two Acc@K numbers: raw and OOD-restricted. `coordinator/integrity_checks.md` PO.4 asserts OOD-restricted > raw.

**Phase:** P1 onwards.

**Status:** `pending`.

**Notes:** AL and FL have different OOD rates (sparse vs dense coverage). Cross-state comparisons use OOD-restricted numbers to avoid confounding. Raw Acc@K remains in the appendix paper table.

---

## Tier C — Region-input mechanism

### CH07 — Bidirectional cross-attention between check-in and region streams improves over concat (gated on CH03)

**Statement:** A new `MTLnetCrossAttn` architecture with two cross-attention layers between the check-in stream and the region-embedding stream achieves higher next-POI Acc@10 than the concat baseline (P3.1 winner) at matched parameter budget, on Florida.

**Source:** HMT-GRN / Bi-Level GSL use explicit cross-level attention, not just concatenation. Tests whether bidirectional information flow between streams captures structure that static concat misses.

**Test:** P4.1 — `MTLnetCrossAttn` vs concat baseline, same data, matched parameter count, 3 seeds × 5 folds.

**Phase:** P4 (only runs if CH03 shows ≥ 2pp Acc@10 lift on FL in P3).

**Status:** `pending (gated)`.

**Notes:** Full architecture spec in `archive/v1_wip_mixed_scope/OPTION_C_SPEC.md`. Separate model class; legacy MTLnet untouched.

---

### CH08 — Region-input gain is state-dependent (AL → FL spectrum)

**Statement:** The Δ next-POI Acc@10 from adding the region embedding stream (CH03 or CH07, whichever is tested) differs between Alabama (1,109 regions, 2.3% majority next-region) and Florida (4,703 regions, 22.5% majority next-region) by a measurable margin.

**Source:** The preliminary linear-probe (AL/AZ/FL check-in-emb → region recovery) showed decreasing signal as region cardinality grows. If transformers extract more than linear probes can, the state-dependence may flatten — that's a finding in itself.

**Test:** P3 cross-state comparison.

**Phase:** P3.

**Status:** `pending`.

**Notes:** CH08 is paper-valuable even if CH03 and CH07 are partial — characterises *when* hierarchical input representation matters.

---

## Tier D — Sensitivity ablations (Alabama primary, FL spot-check)

### CH09 — `next_mtl` head outperforms simpler sequence heads on next-POI

**Statement:** Among sequential head variants in the registry (`next_mtl`, `next_lstm`, `next_gru`, `next_tcn_residual`, `next_temporal_cnn`), `next_mtl` (4-layer transformer + causal + attention pool) achieves the highest next-POI Acc@10 at matched capacity.

**Source:** Prior experiments on high-cardinality sequence prediction showed transformer heads dominate; test whether this replicates on Check2HGI.

**Test:** P5.1 — swap both heads across 5 candidates (both slots sequential); keep everything else at the P2 champion. Alabama only.

**Phase:** P5.

**Status:** `pending`.

**Notes:** Cheap ablation (same fold data, same backbone, different head). If `next_mtl` is not the winner, use the winner as the P2 champion retroactively.

---

### CH10 — MTL optimiser: NashMTL vs equal_weight vs CAGrad

**Statement:** On Check2HGI `{next_poi, next_region}`, NashMTL achieves `val_joint_geom_lift` ≥ equal_weight and ≥ CAGrad by a margin larger than seed × fold CI.

**Source:** Next_POI and next_region have very different label-space magnitudes — gradient-surgery optimisers may help. Or equal_weight may suffice (Xin-et-al. finding).

**Test:** P5.2 — same fold data, same arch, 3 MTL criteria. Run on both AL and FL (per reviewer suggestion — optimiser behaviour is most likely to be imbalance-sensitive).

**Phase:** P5.

**Status:** `pending`.

**Notes:** If equal_weight is within CI, drop NashMTL (cheaper + fewer dependencies — cvxpy/ECOS solver warnings).

---

### CH11 — Seed variance bound

**Statement:** The 3-seed × 5-fold standard deviation of next-POI Acc@10 on the P2 champion is < 2 pp. (Required for CH01's "≥ 2pp margin" claims to be statistically meaningful.)

**Source:** Methodology — a paired-test claim of "+3 pp from MTL" is meaningless if seed variance is ±4 pp. Pre-register the variance bound.

**Test:** CH01 and CH02 are already multi-seed (3 × 5 = n=15), so this variance is computed as a by-product. No additional runs needed beyond re-analysing those results.

**Phase:** P2 (analysis); reported explicitly in P5.3 summary.

**Status:** `pending`.

**Notes:** If observed seed variance > 2 pp, downgrade "confirmed" thresholds in CH01/CH03/CH07 accordingly and report CIs explicitly in the paper table.

---

## Tier E — Limitations & declared scope

### CH12 — Gowalla state-level results do not transfer to FSQ-NYC/TKY

**Statement:** Results on Gowalla-AL/FL are not directly comparable to reported HMT-GRN / MGCL / GETNext numbers on FSQ-NYC/TKY or Gowalla-global. External literature numbers appear only in an appendix table with a scope caveat; the paper's headline comparisons are internal (simple baselines → single-task Check2HGI → MTL → dual-stream → cross-attn).

**Source:** External-validity disclosure. Check2HGI relies on census-tract region assignment (US-only), and our preprocessing differs from those papers'.

**Test:** N/A — declared scope limitation, documented in paper threats-to-validity.

**Phase:** —.

**Status:** `declared`.

**Notes:** Follow-up work: port the region-definition pipeline to FSQ-NYC/TKY using administrative units or neighbourhood polygons.

---

### CH13 — Encoder enrichment (temporal/spatial features, hard negatives, multi-view contrastive) is deferred

**Statement:** This paper reports results on vanilla Check2HGI (fixed 4D temporal features, user-sequence edges, random negatives, 3-boundary MI loss at default α). The four phases of `docs/issues/CHECK2HGI_ENRICHMENT_PROPOSAL.md` are identified as future work.

**Source:** Scope discipline — attributing any lift requires a vanilla baseline first.

**Test:** N/A — declared scope.

**Phase:** —.

**Status:** `declared`.

**Notes:** Bundle enrichment into a follow-up paper; do not mix with the MTL-task-pair contribution.

---

### CH14 — Check2HGI preprocessing shortcut audit

**Statement:** Check2HGI's preprocessing pipeline does not contain a task-level shortcut that trivially solves next-POI or next-region (analogous to the fclass→category 1:1 determinism that fusion's C29 identified in HGI/POI2Vec, but audited here with respect to our own tasks).

**Source:** Fusion-study audit (C29) raised the possibility that shared Gowalla input features (fclass, category, raw POI ids) could create shortcuts. We verify that Check2HGI's different preprocessing chain doesn't reproduce an equivalent shortcut for our tasks.

**Test:** P0.6 — (a) code inspection of `research/embeddings/check2hgi/preprocess.py` for any fclass / category / POI-identity channel that could be exploited for next-POI or next-region; (b) **unconditionally** (per review-agent recommendation) run an fclass-shuffle ablation on next-POI on Alabama, single-task, 1 fold, 10 epochs — compare Acc@10 drop.

**Phase:** P0.

**Status:** `pending`.

**Notes:** If shortcut is found: document as CH14 `refuted_by_construction` + mitigation plan. If no shortcut: document the negative result (also publishable — proves the representation is doing real work). The unconditional shuffle test is cheap (~30 min) and catches shortcuts the inspection might miss.

---

### CH15 — Upstream (transductive) embedding leakage

**Statement:** Check2HGI is trained on 100% of Gowalla check-ins for all users before the downstream task uses user-held-out folds. Validation-fold users' trajectories shape the embeddings they're then evaluated on. We audit the magnitude of this effect by retraining Check2HGI with one fold's users held out and comparing downstream next-POI Acc@10.

**Source:** Fusion study's C30 flagged this for HGI. The transductive-training effect is potentially larger for next-POI prediction than for category classification, because the label IS a POI index whose embedding was directly shaped by the sequences being predicted.

**Test:** P0.7 — retrain Check2HGI on 4-of-5 train folds only (one audit fold; not full 5× re-training), compute next-POI Acc@10 on the held-out fold, compare to the standard result on the same fold.

**Phase:** P0.

**Status:** `pending` (expensive — ~20 min extra per held-out fold; run once as an upper-bound on the effect).

**Notes:** If the gap is small (< 1 pp), declare the effect as bounded. If large, this becomes a paper limitation section — it's a Gowalla-specific transductive-learning phenomenon, not a method failure.

---

## Summary dashboard

| ID | Tier | Phase | Status | Decides |
|----|------|-------|--------|---------|
| **CH01** | A | P2 | pending | MTL lift on next-POI (HEADLINE) |
| **CH02** | A | P2 | pending | No per-head negative transfer |
| **CH03** | A | P3 | pending | Dual-stream region-input helps |
| CH04 | B | P0.5, P1, P2 | pending | Learned models beat simple baselines |
| CH05 | B | P1 | pending | Ranking metrics > macro-F1 methodology |
| CH06 | B | P1+ | pending | OOD-restricted Acc@K (train-memorisation guard) |
| CH07 | C | P4 (gated) | pending | Cross-attention > concat |
| CH08 | C | P3 | pending | State-dependent gain |
| CH09 | D | P5 | pending | Head architecture ablation |
| CH10 | D | P5 | pending | MTL optimiser ablation (AL + FL) |
| CH11 | D | P2 analysis | pending | Seed variance bound |
| CH12 | E | — | declared | Gowalla state-level ≠ FSQ-NYC/TKY |
| CH13 | E | — | declared | Encoder enrichment deferred |
| CH14 | E | P0 | pending | Preprocessing shortcut audit |
| CH15 | E | P0 | pending | Transductive embedding leakage audit |

**Deferred** (not in this study; kept in `archive/v1_wip_mixed_scope/` for reference):
- Next_time_gap third auxiliary task.
- Expert-gating MTL architectures (CGC/MMoE/DSelect-K) — covered by fusion study.
- Frozen-backbone head-swap co-adaptation probe.
- Full P5-legacy arch×optim grid (5×20).

**Minimum viable paper** (if Tier-A CH01 refutes): reframe around CH03 (dual-stream helps without MTL) and CH04/CH05 (methodology contribution) — still a BRACIS-sized paper, weaker contribution.
