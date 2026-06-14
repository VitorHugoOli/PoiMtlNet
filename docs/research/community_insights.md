# Community-Worthy Insights from the Studies Program

> Compiled 2026-06-12 from a full sweep of `docs/studies/` (canonical_improvement, mtl_improvement, embedding_eval, mtl-protocol-fix, substrate-protocol-cleanup, merge_design, hgi_category_injection) and `docs/archive/fusion-study/`. Purpose: the studies program ran 60+ controlled experiments, most of which produced **negative or methodological results that never make it into papers** — yet several are exactly the findings other groups in POI prediction / graph embeddings / MTL would pay to know. This document catalogs them as candidate community contributions (paper sections, a companion "lessons" paper, or workshop notes), each with evidence strength.
>
> Evidence-strength legend: **A** = multi-state × multi-seed, paired test, p reported; **B** = multi-seed single/two-state, or single-seed multi-state; **C** = pilot/single-seed, directional.

---

## 1. Top insights, ranked by community value

### Tier 1 — paper-grade, domain-general

| # | Insight | Type | Strength | Evidence |
|---|---|---|---|---|
| 1 | **Class-weighted CE training vs unweighted top-K metrics silently costs 10–14 pp** (and scales with class count). The entire "MTL sacrifices region prediction" narrative was this confound. Audit loss↔metric pairing *before* architecture work. | METHODOLOGICAL | A (4 states × 4 seeds, confound isolated and reversed) | `docs/studies/archive/mtl_improvement/FINAL_SYNTHESIS.md` §1–2 |
| 2 | **At cos(∇A,∇B) ≈ 0, no MTL optimizer helps — exploit orthogonality, don't fight it.** ~19 optimizer arms null vs tuned static weighting; the win came from asymmetric architecture (private tower for the fragile task + read-only harvest of the shared encoder by the other). Independently aligned with Kurin/Xin NeurIPS'22 and Mueller TMLR'25, and convergent with industrial patterns (STEM's AFTB). | NEGATIVE + POSITIVE | A (gradient cosine over 4 states × 4 seeds × ~3.8k epoch-fold points; 19-arm sweep) | `mtl_improvement/WHY_ORTHOGONAL_AND_NO_MODERN_OPTIMIZERS.md`, `docs/results/mtl_improvement/T4_audit_and_verdict.md` |
| 3 | **Small-state screening produces false negatives for scale-conditional effects — large-state replication is mandatory before discarding a hypothesis.** Four independent instances: design_k (Delaunay POI-GCN) discarded at AL/AZ, later shown to close 54–78% of the substrate gap at FL; AdamW WD=5e-2 (noise at AL/AZ, +0.56 pp reg at FL, p=0.031); ResLN (clearer at FL); 3-snapshot routing (+2.80 pp FL, fails AL). | METHODOLOGICAL | A–B (leak-free multi-seed re-validations) | `embedding_eval/FINAL_SYNTHESIS.md` (design_k reopened), `canonical_improvement/log.md` |
| 4 | **The joint-training regime, not the substrate, is the MTL bottleneck: even the STL-winning substrate's advantage vanishes under joint training.** HGI's +2.1 pp STL reg edge → +0.51 pp NS (p=0.41) under MTL; four substrate designs null in MTL while the *same* designs lift STL +2.3 pp; dual-substrate routing also null. "Rising-tide rule": every reg-input lever lifts the STL ceiling by the same amount it lifts MTL. | NEGATIVE (5 independent falsifications) | A–B | `substrate-protocol-cleanup/CLOSURE.md`, `tier_b_fl/hgi_mtl_fl.md`, `mtl_improvement` R1/R2 audits |
| 5 | **Graph attention (GATv2) over temporally-decayed user-sequence edges learns a label-copying shortcut** — cat F1 jumped to ~99% with leak-probe +11.3 pp while reg stayed flat; the leak emerges *during training* (random-init probe showed GAT less leaky than GCN before training). Architecture papers should report leak probes alongside headline metrics whenever attention is added over behavior graphs. | NEGATIVE + METHODOLOGICAL | B (diagnosed with probes, mechanism isolated) | `canonical_improvement/log.md` (T3.1) |

### Tier 2 — solid, useful to practitioners in this subfield

| # | Insight | Type | Strength | Evidence |
|---|---|---|---|---|
| 6 | **Hard-index transition priors beat learned soft probes** (+3–9 pp Acc@10 near convergence): indexing log_T by the *observed* last region outperforms a learned soft probe over regions. | POSITIVE | B (B5 ablation) | `docs/PAPER_BASELINES_STRATEGY.md` (STAN-Flow), B5 ablation records |
| 7 | **A static Markov prior distilled as KD (log_T-KD, W=0.2) lifts MTL reg +2–5 pp at small states** — the prior pathway is the one channel that moves MTL reg when trunk-level levers are dead. | POSITIVE | A at AL/AZ (n=20, p=9.5e-07); C at FL/CA/TX (seed-42 pilot) | `substrate-protocol-cleanup/CLOSURE.md`, `tier_a1/phase_a1_verdict.md` |
| 8 | **Class-balanced batch sampling is catastrophic for top-K metrics** (−18 to −30 pp reg): undersampling dominant classes optimizes away from top-K mass. Companion to insight #1. | NEGATIVE | B | `mtl-protocol-fix` Phase 3 |
| 9 | **Joint-checkpoint selection by averaging per-task F1s fails when one task is high-cardinality**: macro-F1 over ~4.7k sparse regions is rare-class noise; the F1-mean selector threw away ~10.7 pp of deployable reg. Geometric mean of each task's *primary* metric fixed it (+5.6 pp recovered). | METHODOLOGICAL | B (FL multi-seed) | `docs/CONCERNS.md §C21`, `mtl-protocol-fix` |
| 10 | **Optimizer choice is embedding-dependent in fusion settings**: with multi-source fusion (15× embedding-norm imbalance), gradient-surgery optimizers (CAGrad/Aligned-MTL) are essential (25% joint-score gap); on single-source, equal-weight suffices. Testing an MTL optimizer on one embedding and transferring the conclusion is unsound. | POSITIVE + METHODOLOGICAL | B (4 archs × 5 optimizers grid) | `docs/archive/fusion-study/FINAL_ANALYSIS.md` §1, §5 |
| 11 | **Late residual injection at the post-pool boundary transfers a foreign embedding's signal without harming the host's other axis** (POI2Vec into the reg path with a detached cat path: zero cat regression across 5 designs), whereas naive late-fusion concat broke both axes (−9/−10 pp). | POSITIVE | B (AL/AZ; FL partial) | `merge_design/STATE.md` |
| 12 | **Frozen-cat / λ=0 isolation is unsound under cross-attention** — gradients reach the "disabled" encoder through K/V projections, so loss-side ablations overstate architectural costs. Clean isolation requires freezing, which has its own instability. | METHODOLOGICAL | B | `docs/CONCERNS.md §C12`, F49 records |
| 13 | **Single geometry metrics are not substrate oracles**: region-silhouette predicts the spatial axis of downstream gains, but a substrate can win on an orthogonal axis (fclass mix) while silhouette *drops*. Multi-metric geometry baterries are needed for embedding evaluation. | METHODOLOGICAL | B (5 substrates × 3 geometry metrics at FL) | `embedding_eval/FINAL_SYNTHESIS.md` (metric-specificity correction) |
| 14 | **The +3 pp MTL category gain decomposes into architecture (+2.3–3.2 pp) vs genuine cross-task transfer (+0.9 FL / −0.7 AL)** — measured by reg-weight-0 ablation. MTL papers claiming task-transfer gains should run this decomposition; most of the gain may be "the joint architecture is just a better encoder". | METHODOLOGICAL | A (4 seeds, 2 states) | `docs/results/mtl_improvement/cat_transfer_and_T53.md` |
| 15 | **Stale derived artifacts silently survive regeneration**: a transition matrix older than its source data inflated reg by +8–12 pp across an entire experiment tier; relative comparisons survived (same-stale-everywhere) but absolute numbers were biased. Derived-artifact freshness checks (mtime/hash preflight) are load-bearing infrastructure. | METHODOLOGICAL | B (FL case study, audited) | `docs/results/mtl_protocol_fix/phase1_verdict.md` |

### Tier 3 — smaller but citable

- **ResidualLN GCN encoder** is a scale-agnostic category micro-improvement (+1.2–1.7 pp, 3 states, 5 seeds, p=0.031) — STL-only (`canonical_improvement/log.md` T3.2).
- **DropEdge gains at short budgets are budget-rescue artifacts** — they invert at full budget (`canonical_improvement/log.md` T2.4 stacking probe).
- **Boundary-weight (α) sweeps in hierarchical-infomax objectives are inert** (≤0.3 pp) — the boundary *architecture*, not its loss weight, is load-bearing (`canonical_improvement/log.md` T1.3).
- **Injecting explicit category features into an embedding that already encodes category structure is inert** — HGI's POI2Vec behaves as an fclass lookup; 6 injection variants all null (`hgi_category_injection/STATUS.md`).
- **Curriculum/freezing schedules don't rescue joint training**: freeze-reg-after-peak null for all N; frozen-cat does not recover MTL reg (`substrate-protocol-cleanup` C2; `mtl-protocol-fix` P4).
- **Development-seed contamination is real and scale-dependent**: seed-42 (the tuning seed) overshoots multi-seed means by +3 pp (CA) to +8 pp (TX) — reporting seeds must be disjoint from the development seed (`docs/results/CANONICAL_VERSIONS.md`).
- **sklearn version changes silently reshuffle StratifiedGroupKFold folds** (PR #32540): within-phase paired tests survive; cross-phase absolute comparisons carry ±2–3 pp fold-shift noise (`STATISTICAL_AUDIT.md`).

---

## 2. Cross-study contradictions worth publishing as a meta-lesson

The program's own corrections are unusually instructive:

1. **design_k**: falsified at AL/AZ → overturned at FL (54–78% gap closure). *Lesson: scale-conditional effects + small-state screening = systematic false negatives.*
2. **"HGI's reg advantage will carry into MTL"**: assumed → falsified (p=0.41 NS). *Lesson: STL rankings don't survive joint-training regimes.*
3. **"MTL sacrifices region"** (the original paper thesis): falsified as a confound (C25). *Lesson: objective↔metric audits precede architecture conclusions.*
4. **v13 "best dual-axis"**: qualified — its geometry metric moved the *wrong* way while downstream improved, exposing the metric-axis specificity issue (#13).

Together these justify the program's defining methodological stance: multi-state replication, leak probes, paired statistics, and a corrections registry. **A short "lessons from 60+ controlled experiments on embeddings and MTL for POI prediction" paper (workshop or national-venue companion) is a viable publication on its own** — negative results of this density and rigor are rare in this subfield.

---

## 3. Suggested vehicles

| Vehicle | Content | Effort |
|---|---|---|
| New paper §Discussion/§Lessons | Insights 1, 2, 4, 14 (they *are* the paper's mechanism story) | None — already planned |
| Companion workshop paper ("what didn't work and why") | Insights 3, 5, 8, 9, 10, 15 + §2 meta-lessons | Low — material exists in study logs |
| Methods note / blog-style tech report | Insight 15 (artifact freshness), 9 (selector design), C28 (seed protocol) | Low |
| Reproducibility appendix of the new paper | C28 sklearn fold-drift, dev-seed quarantine | Trivial |

> Note: before externalizing any insight, re-verify its evidence strength against the current canonical version (some Tier-2/3 items are B/C strength and would need one confirmation run under the frozen v16+ recipe — candidates for `closing_data` P1a cross-study re-eval, which is exactly the phase designed to do this).
