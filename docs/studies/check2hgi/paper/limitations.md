# Limitations section — draft v1 (revised after F37 FL closing 2026-04-28)

**Date:** 2026-04-28
**Target:** Submission paper §6 (Limitations / Threats to validity). ~1 page.

---

## 6 Limitations

### 6.1 Scale-conditional architectural lift on `next_region` (AL-only)

While the H3-alt recipe achieves an architecturally-dominant reg lift on Alabama (+6.48 pp from cross-attention with frozen-random cat features, paired Wilcoxon p=0.0312, 5/5 folds; full MTL +6.25 pp above STL `next_getnext_hard`), this advantage **does not generalise to FL scale**. Matched-head STL `next_getnext_hard` exceeds MTL H3-alt at Florida by 8.78 pp (5-fold paired Wilcoxon p=0.0312, 5/5 folds negative). The F49 architectural decomposition reveals that frozen-cat performance on FL collapses to 16.16 pp below STL (paired p=0.0312), with per-fold variance σ ≈ 12 pp consistent with α-growth failing when cat features are random at 4,702-region scale.

**Cardinality-conditional pattern across our 3 states:** the architectural Δ from `(frozen-cat λ=0 − STL F21c)` is **+6.48 (AL, 1,109 regions) → −6.02 (AZ, 1,547) → −16.16 pp (FL, 4,702)** — a monotonic decrease as region cardinality grows. We do not yet have a CA or TX measurement (P3 in PAPER_PREP_TRACKER, gated on upstream pipelines), so we cannot distinguish whether (i) FL is a single outlier or (ii) the architectural cost is a smooth function of cardinality. The headline contribution at FL is the **substrate-side cat advantage** (CH16 + CH18-substrate); the architecture-side reg lift is AL-only. The paper reports the per-state pattern explicitly rather than papering it over.

### 6.1b Phase-1 substrate validation runs on 2 dev states only

Phase-1 substrate validation (CH16 head-invariance, CH18-substrate counterfactual, CH19 per-visit-context mechanism) runs on AL (10K check-ins, 1,109 regions) and AZ (26K check-ins, 1,547 regions). FL+CA+TX are headline states. Phase-2 substrate-grid replication at FL is queued (F36) and CA+TX are gated on upstream pipelines (P3). The cat-side MTL > STL relation has been verified at FL (+0.94 pp F37 P1 2026-04-28); the head-invariance and counterfactual-breaks-reg findings are queued. If Phase-1 substrate findings fail to replicate at headline scale, the paper retreats to "AL+AZ-only substrate findings" with the FL/CA/TX H3-alt rows as scale-validation of the joint pipeline alone.

### 6.2 FL frozen-cat reg-path instability

The F49 encoder-frozen variant on FL n=5 shows reg Acc@10 σ = 12.03 (vs σ = 1.40 for loss-side λ=0 on the same fold split). Per-fold reg-best epochs collapse to early values {2, 14, 9, 4, 2}, suggesting that when cat features are random, the α (graph prior) growth mechanism either fails to engage or saturates prematurely. Two interpretations:

- **(A) Random-cat features are below a capacity threshold at FL scale (4,702 regions)**, so the architecture cannot exploit them; the loss-side variant works because slow co-adaptation eventually produces useful features.
- **(B) Selection-noise** in the per-task-best-epoch criterion dominates at FL: the reg head finds spurious early peaks before α stabilises.

We do not yet distinguish (A) and (B). A seed-sweep (~30 min, deferred to camera-ready as P11) would help.

**Status as of F37 closing (2026-04-28):** Layer 3 of the F49 decomposition is now closed. The FL absolute architectural Δ vs STL F21c is **−16.16 pp paired Wilcoxon p=0.0312, 5/5 folds negative** — a sign-unambiguous result despite the high σ on the architectural cell itself. Per-fold deltas are {−32.22, −31.05, −4.01, −5.42, −8.08} pp; the magnitude has σ ≈ 12 pp (driven by the frozen-cat instability above), but the direction across all 5 paired folds is consistent. Multi-seed replication (P6/P11) would tighten the magnitude estimate; the qualitative claim "architecture costs reg at FL scale" is robust.

### 6.3 CA + TX gated on upstream pipelines

Check2HGI embeddings, region transition matrices, and input parquets are not yet generated for CA or TX. P3 (~37h Colab) is the gating dependency. The headline paper currently spans AL+AZ+FL; CA+TX rows are placeholders.

If the paper is submitted before P3 completes, we frame as "3-state US replication study, with CA and TX deferred to extended/journal version."

### 6.4 CH15 was head-coupled, not pure substrate

Earlier framings reported CH15 ("HGI > Check2HGI on `next_region` under STL STAN") as a substrate finding. Phase-1 Leg II showed this was a STAN-head preference for POI-stable smoothness, not a pure substrate effect: under matched MTL reg head (`next_getnext_hard`), Check2HGI ≥ HGI everywhere (AL TOST non-inferior, AZ +2.34 pp p=0.0312). We preserve the STAN-head data as a head-sensitivity row (Table A.X) alongside the matched-head row.

This reframe is documented (CH15-revised, see appendix `SCOPE_DECISIONS.md`) but introduces a residual asymmetry: "meaningful Check2HGI advantage is on the cat-input side" (CH16 +11–15 pp), not the reg-input side (∓1 pp under matched head, head-coupled under non-matched).

### 6.5 Single-seed (seed=42) per cell

All cells in Tables 1–4 use `seed=42`. We did not run a multi-seed sensitivity test (deferred — P6 in PAPER_PREP_TRACKER, ~3h MPS for AL+AZ {0, 7, 100}). σ values reported are over the 5 stratified-group folds, not over seed perturbations. A reviewer concerned about "are these results within fold-σ but outside seed-σ?" should consult the appendix where we report seed-{0, 7, 100} multi-fold-mean variance once it lands.

### 6.6 Loss-side ablation methodology (F49 §3.7)

We argue (and demonstrate with regression tests) that loss-side `task_weight=0` ablation is unsound under cross-attention MTL. The encoder-frozen variant we propose has its own caveats:

- **Block-internal `ffn_a` and `ln_a*` are not frozen.** They live in `shared_parameters()` and continue to train under L_reg. A "totally frozen cat-side block" variant requires autograd-detach on the cat stream output before cross-attention reads it (deferred — P9, ~2h dev + ~1h compute).
- **AdamW silent-decay bug.** `weight_decay=0.05` decays frozen weights unless explicitly filtered from optimiser groups; we filter and document with a regression test, but the bug demonstrates how subtle MTL ablation soundness can be.

### 6.7 No POI-granularity claim

We deliberately do not report results at the POI-id ranking granularity (~tens of thousands of classes per state). Pilot experiments showed long-tail sparsity dominates user-grouped 5-fold macro-F1; this granularity is also not the primary granularity of comparable prior work (POI-RGNN, MHA+PE, HMT-GRN). Readers seeking POI-id ranking should consult [follow-up paper / extended version, TBD].

### 6.8 What we cannot conclude

From the current evidence, we **cannot** claim:
- Multi-seed σ stability (single-seed runs).
- Cross-dataset generalisation outside US Foursquare (no Brightkite / Gowalla / Yelp replications; HMT-GRN/MGCL comparisons are concept-aligned, not on the same data).
- That `next_getnext_hard` (graph-prior reg head) is uniquely Pareto-optimal — alternative reg heads (TGSTAN, STA-Hyper) are deferred to follow-up work.
- Generalisation beyond cross-attention MTL — our methodological note (§A) flags applicability to MulT/InvPT/etc., but we have not empirically replicated F49 attribution on those architectures.
