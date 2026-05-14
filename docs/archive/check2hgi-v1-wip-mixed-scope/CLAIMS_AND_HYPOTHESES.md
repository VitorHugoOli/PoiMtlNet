# Check2HGI Claims & Hypotheses Catalog

**Rule:** no claim moves into the BRACIS paper without `status ∈ {confirmed, partial}` and a pointer to evidence under `docs/studies/check2hgi/results/`.

**ID scheme:** `CH01..CHnn` (distinct from legacy `C01..Cnn`).

Each entry: **Statement → Source → Test → Phase → Status → Evidence → Notes**.

---

## Tier A — Headline paper thesis

These are the three claims the paper makes; every other claim either supports them or constrains their interpretation.

### CH01 — Check2HGI (check-in-level) beats HGI (POI-level) on next-POI-category

**Statement:** In a single-task setting (no MTL), predicting next-POI-category from a sliding window of check2HGI check-in-level embeddings achieves higher macro-F1 than the same task using HGI POI-level embeddings, at matched architecture, 5-fold CV, matched training budget, on Alabama and Florida.

**Source:** Paper thesis. Check-in-level contextual embeddings should encode trajectory phase (morning commute vs evening leisure) that POI-level vectors blur together.

**Test:** P2.1 — single-task `next` head on HGI vs CHECK2HGI. Same architecture (`next_mtl`), same 5-fold user-held-out split, same 50-epoch OneCycleLR. Report macro-F1 with bootstrapped CIs.

**Phase:** P2.

**Status:** `pending`.

**Notes:** Negative result is itself publishable ("contextual embeddings don't help on 7-class category prediction, only emerge on high-cardinality tasks like next-region"). If CH01 is negative but CH04 is positive, the paper framing shifts from "better embedding" to "better task".

---

### CH02 — Next-region auxiliary task improves next-POI-category under MTL

**Statement:** Adding next-region as an auxiliary task (2-task MTL with NashMTL balancing) raises next-POI-category macro-F1 over the single-task baseline at matched total training budget, on check2HGI embeddings, on AL and FL.

**Source:** Paper thesis. HMT-GRN (SIGIR 22), MGCL (Frontiers 24), Bi-Level GSL (arXiv 24), TORS 24 all report lift from hierarchical region-level auxiliaries on next-POI tasks.

**Test:** P3.1 — same check2HGI embeddings, same `next_mtl` head architecture, compare: (a) single-task next-POI-category; (b) 2-task MTL {next_category, next_region} with NashMTL. 5-fold CV, 50 epochs, matched batch.

**Phase:** P3.

**Status:** `pending`.

**Notes:** This is the headline comparison of the paper. If CH02 is negative on check2HGI but CH01 is positive, the narrative becomes "check2HGI helps but MTL doesn't add signal" — publishable but weaker. If both positive, clean composite story. If CH02 negative on AL but positive on FL (or vice-versa), expect reviewer questions about dataset-specific behaviour.

---

### CH03 — No negative transfer: per-head metrics under MTL ≥ single-task baselines

**Statement:** Under the 2-task MTL from CH02, per-head validation metrics (`next_category` macro-F1, `next_region` Acc@1 and MRR) are ≥ their respective single-task baselines by a margin larger than the 5-fold CV confidence interval.

**Source:** Critical-reviewer concern. Without this control, a composite "joint score goes up" claim is confounded with the possibility that one head improves at the other's expense.

**Test:** P3.2 — compare:
- next_category MTL vs next_category single-task (macro-F1)
- next_region MTL vs next_region single-task (Acc@1 + MRR)

using the same 5-fold split.

**Phase:** P3.

**Status:** `pending`.

**Notes:** The legacy track's analogous claim was C28. Expected outcome is ≥ ties on both heads; any significant regression on one head is the first thing a reviewer will ask about.

---

## Tier B — Supporting & methodology

### CH04 — Next-region is a meaningful task on check2HGI (not a free-ride regulariser)

**Statement:** Single-task next-region prediction on check2HGI embeddings achieves Acc@1 substantially above the majority-class baseline (i.e., the head learns to discriminate regions, not just output the most popular one), and Acc@10 ≥ Acc@1 + reasonable margin on both states.

**Source:** Sanity / anti-debugging. If the next-region head is easy to saturate at majority-class, then CH02's apparent MTL lift could be a regularisation artefact (noise injection from uninformative labels), not genuine task signal.

**Test:** P2.3 — single-task next-region on check2HGI. Report Acc@{1, 5, 10}, MRR, macro-F1, majority-class baseline.

**Phase:** P2.

**Status:** `pending`.

**Notes:** Alabama has ~1109 regions; majority-class Acc@1 ≤ 5% expected. Florida similar. If single-task Acc@1 is barely above majority-class, next-region is not a useful task on its own and CH02 needs a different explanation.

---

### CH05 — Ranking metrics surface differences that macro-F1 conflates on high-cardinality heads

**Statement:** On the next-region head (∼10³ classes), Acc@K and MRR rank check2HGI- vs HGI-embedding configurations in a way that macro-F1 does not — specifically, configurations indistinguishable by macro-F1 show a measurable MRR delta.

**Source:** Methodology / limitations. HMT-GRN and MGCL use ranking metrics precisely because class-balanced F1 collapses on high-cardinality, sparse-support labels.

**Test:** P2.2 — on the next-region head specifically, compute macro-F1 and MRR for HGI vs CHECK2HGI at matched arch and budget; check whether MRR delta is statistically significant when macro-F1 delta is not (or vice-versa).

**Phase:** P2.

**Status:** `pending`.

**Notes:** Paper limitations section — documents WHY the next-region head is monitored via Acc@1 + MRR and not macro-F1 (per the overview-doc §2 decision). One negative example would suffice.

---

### CH06 — Joint-score monitor `joint_acc1` is robust under head-metric heterogeneity

**Statement:** Checkpoint selection via `joint_acc1 = mean(acc1_cat, acc1_region)` selects a checkpoint whose `joint_f1 = mean(f1_cat, f1_region)` is within a tight tolerance of the best checkpoint selected by `joint_f1`. I.e., the monitor choice does not materially change which checkpoint wins.

**Source:** Overview-doc §2 open design decision. We chose `joint_acc1` as primary monitor to avoid mixing F1 + Acc@1 scales; this claim validates that choice empirically.

**Test:** P4.1 — for each completed run, log both `joint_acc1` and `joint_f1` curves; measure the epoch-distance between the `argmax` of each and the metric loss at the other's argmax.

**Phase:** P4.

**Status:** `pending`.

**Notes:** If the two monitors pick wildly different checkpoints, either pick one and defend it (stricter) or report both in the paper table (safer, more cluttered). Low-risk claim; just need the data.

---

## Tier C — Ablations

These are the questions a reviewer WILL ask. Each has a concrete ablation row in `ABLATIONS.md`.

### CH07 — Head architecture: NextHeadMTL (transformer) vs simpler sequence heads

**Statement:** On the next-region head, `next_mtl` (transformer, causal + attention pool) achieves higher Acc@1 and MRR than `next_lstm` and `next_gru` at matched parameter budget.

**Source:** Prior MTLnet ablations showed transformer heads dominated on next-category; claim is that the lift replicates at high-cardinality.

**Test:** P4.2 — swap `next_region` head across `next_mtl`, `next_lstm`, `next_gru`, `next_transformer_relpos`, `next_hybrid` (if parameter-matched variants exist). Compare Acc@1 + MRR on AL.

**Phase:** P4.

**Status:** `pending`.

**Notes:** Cheapest ablation; shared fold data, only the head changes.

---

### CH08 — MTL optimiser: NashMTL vs CAGrad / Aligned-MTL / naive equal-weight

**Statement:** On check2HGI 2-task MTL, NashMTL achieves joint_acc1 ≥ CAGrad and ≥ naive equal-weight by a margin exceeding the 5-fold CI.

**Source:** Legacy C02 / prior fusion-track finding that gradient-surgery optimisers help on scale-imbalanced task pairs; this claim tests whether the same holds on check2HGI's next-category + next-region pair.

**Test:** P4.3 — same fold data, vary MTL criterion across {nash_mtl, cagrad, aligned_mtl, naive}. 5-fold CV each.

**Phase:** P4.

**Status:** `pending`.

**Notes:** If NashMTL and naive are within CI on check2HGI, the paper drops the optimiser-choice argument and just uses whichever is faster. Parallels the pre-bug finding from the legacy track.

---

### CH09 — Task embedding contributes vs FiLM-only conditioning

**Statement:** Removing the per-task embedding (using only FiLM + shared backbone) drops joint_acc1 on the 2-task check2HGI MTL, vs keeping the task embedding.

**Source:** MTLnet design question; task_embedding is a learnable (2, D) table that conditions FiLM. Zeroing it reduces the task-conditioning signal.

**Test:** P4.4 — `TaskConfig(task_embedding=False, …)` variant; otherwise identical config.

**Phase:** P4.

**Status:** `pending`.

**Notes:** Small-diff ablation; good for a model-design paragraph.

---

## Tier D — Limitations & external-validity flags

These are claims we expect to *refute* or *flag* — paper limitations, not contributions.

### CH10 — Results do not transcribe to FSQ-NYC/TKY

**Statement:** Because check2HGI relies on census-tract region assignment (US-only in the current pipeline), results on Gowalla-AL/FL are not directly transferable to FSQ-NYC/TKY without a region-definition port (neighbourhoods or administrative units).

**Source:** External validity honesty. The mainstream benchmark is FSQ-NYC/TKY; our Gowalla state-level work is a sibling lineage (POI-RGNN / HAVANA / PGC), not the same benchmark.

**Test:** N/A — declared limitation, documented in the paper's threats-to-validity section.

**Phase:** P4 (documentation only).

**Status:** `declared`.

**Notes:** Required disclosure; reviewers otherwise treat the missing benchmark as a weakness.

---

### CH11 — Enrichment is deferred; vanilla check2HGI is the baseline the paper reports

**Statement:** The paper reports results on vanilla check2HGI (fixed 4D temporal features, user-sequence edges only). Enrichment (learnable temporal, spatial positional encoding, KNN spatial edges, hard negatives) is identified as future work.

**Source:** Scope discipline (`CHECK2HGI_MTL_OVERVIEW.md §3`).

**Test:** N/A — declared scope.

**Phase:** —.

**Status:** `declared`.

**Notes:** Protects the causal attribution of any lift. If we bundled Phase-1 enrichment into the baseline, reviewers couldn't separate "better encoder features" from "better task signal".

---

## Tier E — Ported legacy ablations (P5, P6)

Mirror claims from the legacy `docs/studies/phases/P1_arch_x_optimizer.md` + `P2_heads_and_mtl.md` applied to the check2HGI track. Let us compare check2HGI directly with the legacy HGI / fusion findings.

### CH14 — Gradient-surgery optimisers beat equal-weight on check2HGI

**Statement:** On check2HGI `{next_category, next_region}`, `cagrad` / `aligned_mtl` achieve joint_acc1 ≥ 2 p.p. higher than `equal_weight` / `static_weight` at matched `gradient_accumulation_steps=1, batch_size=4096`, on AL and AZ.

**Source:** Legacy C02 mirror. Check2HGI's task-scale imbalance (7-class vs 1109-class labels) is much larger than the legacy fusion pair's — gradient-surgery advantages could be amplified.

**Test:** P5 confirmation stage (5-fold × 50-epoch top-5 comparison).

**Phase:** P5.

**Status:** `pending`.

**Notes:** If `equal_weight` wins here, a Xin-et-al.-style finding extends to check2HGI — publishable and counter-intuitive.

---

### CH15 — Expert-gating architectures beat FiLM-only on check2HGI

**Statement:** `mtlnet_cgc`, `mtlnet_mmoe`, `mtlnet_dselectk` achieve higher mean joint_acc1 than `mtlnet` (FiLM-only) when averaged across optimisers.

**Source:** Legacy C05 mirror.

**Test:** P5 post-analysis.

**Phase:** P5.

**Status:** `pending` (blocked on TaskSet parameterisation of the expert-gating variants).

---

### CH16 — Winning (arch, optim) pair on check2HGI differs from HGI

**Statement:** The P5 champion on check2HGI is not the same (arch, optim) pair as the legacy study's winner on HGI / fusion. Dataset-dependent architecture-optimiser interaction.

**Source:** Legacy C04 mirror.

**Test:** P5 vs legacy results cross-ref.

**Phase:** P5.

**Status:** `pending`.

---

### CH17 — Head rankings in MTL differ from standalone rankings

**Statement:** The ranking of task_a heads when trained in the 2-task MTL differs (by Spearman ρ < 0.7) from the ranking of the same heads trained as single-task next-category models.

**Source:** Legacy C08 mirror.

**Test:** P6a vs P6b ranking comparison.

**Phase:** P6.

**Status:** `pending`.

---

### CH18 — Frozen-backbone head-swap matches MTL rankings better than standalone

**Statement:** When the MTL backbone from the P5/P6 champion is frozen and alternative heads are fine-tuned on it, the resulting head ranking correlates more strongly with the MTL-end-to-end ranking (Spearman ρ > 0.8) than with the standalone ranking.

**Source:** Legacy C09 mirror. Tests the co-adaptation-to-backbone hypothesis.

**Test:** P6d probe.

**Phase:** P6 (optional).

**Status:** `pending`.

---

### CH19 — Head co-adaptation mechanism (narrative claim)

**Statement:** In MTL on check2HGI, each head adapts to the backbone representation more than to the raw embedding — i.e., the backbone effectively compiles a task-specialised feature for each head, and heads are selected for "how well they read that compiled feature," not "how well they solve the task in isolation."

**Source:** Narrative framing of CH17 + CH18 for the paper.

**Test:** P6d + qualitative analysis of backbone activations per-task.

**Phase:** P6.

**Status:** `pending`.

---

## Tier F — Option A / Option C (Phase P7)

### CH12 — Region embeddings as input improve next-region Acc@1

**Statement:** Using dual-stream input (concat check-in emb + region emb per timestep) instead of check-in only increases `val_acc1_next_region` by ≥ 2 p.p. on Florida at 5f × 50ep, champion backbone + optimiser.

**Source:** Probe experiment (`scripts/exp_embedding_region_info.py`) + literature (HMT-GRN, Bi-Level GSL).

**Test:** P7a ablation.

**Phase:** P7.

**Status:** `pending`.

**Notes:** Probe predicts this will be larger on FL (4703 regions, probe at 23.5% ≈ majority) than on AL (1109 regions, probe at 7.9% ≫ majority).

---

### CH13 — Bidirectional cross-attention between streams improves over concat

**Statement:** At equal parameter budget, `MTLnetCrossAttn` (K=2 cross-attention layers) achieves joint_acc1 > Option A (dual-stream concat) on FL.

**Source:** HMT-GRN's hierarchical-attention design as a template.

**Test:** P7b ablation.

**Phase:** P7 (gated on CH12 succeeding).

**Status:** `pending`.

---

### CH20 — Region-input gain is state-dependent

**Statement:** The Δ Acc@1 from adding dual-stream region input (Option A or C vs vanilla) is larger on Florida (4703 regions) than Alabama (1109 regions) by at least 2×.

**Source:** Probing experiment showed check-in emb is 3× above majority on AL but barely above majority on FL under LR; the gap at the MTL level should track this pattern.

**Test:** P7 comparison across states.

**Phase:** P7.

**Status:** `pending`.

**Notes:** CH20 is paper-valuable even if CH12/CH13 are partial — it tells readers *when* the design choice matters.

---

## Summary dashboard

| ID | Tier | Phase | Status | Decides |
|----|------|-------|--------|---------|
| CH01 | A | P2 | pending | Embedding-quality claim |
| CH02 | A | P3, P6 | pending | MTL lift claim (headline) |
| CH03 | A | P3, P6 | pending | No negative transfer |
| CH04 | B | P2 | pending | Next-region is meaningful |
| CH05 | B | P2 | pending | Ranking-metric methodology |
| CH06 | B | P4 | pending | Monitor-choice robustness |
| CH07 | C | P4 | pending | Head architecture ablation |
| CH08 | C | P4 | pending | MTL optimiser ablation |
| CH09 | C | P4 | pending | Task-embedding ablation |
| CH10 | D | — | declared | External-validity limit |
| CH11 | D | — | declared | Enrichment out of scope |
| CH12 | F | P7 | pending | Dual-stream region input helps |
| CH13 | F | P7 | pending | Cross-attention > concat |
| CH14 | E | P5 | pending | Gradient-surgery optimisers (legacy mirror) |
| CH15 | E | P5 | pending | Expert-gating > FiLM (legacy mirror) |
| CH16 | E | P5 | pending | Winner differs across engines |
| CH17 | E | P6 | pending | MTL head ranking ≠ standalone |
| CH18 | E | P6 | pending | Frozen-backbone-swap matches MTL ranking |
| CH19 | E | P6 | pending | Co-adaptation mechanism (narrative) |
| CH20 | F | P7 | pending | Gain is state-dependent (from probe) |
