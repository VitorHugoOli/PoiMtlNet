# mtl_frontier — the MTL exploration frontier (A40, runs FIRST)

> **Status:** SCAFFOLDED, not launched (2026-06-14). Machine: **A40 (unmetered)**. Position in the
> family: **Level 0** of [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md) — exploration precedes
> the freeze. A promoted lever becomes a `closing_data` pre-freeze gate (G0.2).
>
> **Read first:** [`docs/research/mtl_frontier.md`](../../research/mtl_frontier.md) (the full frontier
> survey + the ranked R1–R9 program) and [`../archive/mtl_improvement/FINAL_SYNTHESIS.md`](../archive/mtl_improvement/FINAL_SYNTHESIS.md)
> (champion G, the orthogonal-gradient regime, the 19-arm optimizer null). This study does NOT redo the
> optimizer aisle — that is closed.

## Why this study exists

The repo's measured regime — 2 tasks, shared-trunk gradient cos ≈ 0, all ~19 classical MTL optimizers
null, wins only from architectural asymmetry (champion G's dual-tower) + an output-level KD prior
(log_T-KD) — matches the post-2022 MTL consensus (Kurin/Xin NeurIPS'22; Mueller TMLR'25: gradient
conflict doesn't predict transfer). Therefore **genuine new transfer must enter at the output level or
through asymmetric read-only sharing, not through the trunk.** This study tests exactly those channels.

## Scope — first wave (the three highest-yield, gate-eligible)

| ID | Experiment | Mechanism / why it can work here | Promote-gate |
|---|---|---|---|
| **R1** | **log_C co-location prior + probability-chain coupling** | Build a train-only region×category matrix P(region\|cat) — same per-fold/per-seed infrastructure as log_T (`scripts/compute_region_transition.py` pattern) — and couple ESMM-style: prior(reg)=Σ_c P(reg\|c)·P̂(c). Extends the **only** lever class that has ever moved MTL reg. | ≥0.3 pp reg over log_T-KD-alone, multi-seed |
| **R2** | **STEM-style AFTB gating sweep** | Parameterize champion G's dual-tower sharing as explicit all-forward / task-specific-backward gates (per-layer stop-grad masks); sweep which layers cat may read vs own. Turns the hand-built asymmetry into a measured dose-response; citable against STEM (AAAI 2024). | ≥0.3 pp either head over G, multi-seed |
| **R3** | **Live cross-task distillation (CrossDistil)** | Calibrated cat-head posterior as a *dynamic* teacher for the reg head (and/or reverse), warm-up-gated. Generalizes log_T-KD from a static Markov teacher to a learned one. | ≥0.3 pp over R1 / log_T teacher |

Later waves (only if first wave motivates): R4 Pareto-front profiling (PaLoRA-style — resolves the C21
selector class permanently), R5 per-instance KD gating, R6 ForkMerge, R7 merge-vs-joint (ZipIt/SIMO),
R8 next-visit-time auxiliary (paired STL control — rising-tide caution), R9 residual optimizer sanity
(BayesAgg-MTL — verify the existing `src/losses/registry.py` impl matches ICML'24; Smooth-Tchebycheff
only if R4 shows a non-convex front). Full rationale + citations: [`mtl_frontier.md §4`](../../research/mtl_frontier.md).

### R10 — Memory-Caching / GRM gating at the LAYER level (arXiv 2602.24281 — ★ user-requested)

> **The user's explicit framing: explore this paper's mechanism _on the layers, not on the transformers_** — i.e. apply its primitive to an existing layer's state, not as a transformer-block swap. Second-wave architectural exploration (adjacent to R2); promote-eligible only once a working impl exists.

- **Paper.** *Memory Caching: RNNs with Growing Memory* — Behrouz, Li, Deng, Zhong, Razaviyayn, Mirrokni (Google Research, Titans/Atlas lineage), arXiv:2602.24281, Feb 2026; OpenReview `R3EJ2IjgOI`. **No code release** → any adoption is a from-scratch reimplementation of the primitives. Headline contribution (growing memory for long-context recall) targets a problem this project does **not** have — check-in windows are short, fixed length 9 — so it is the **sub-mechanisms**, not the headline, that transplant.
- **The transplantable primitive ("on the layers").** Snapshot a layer's state at segment boundaries → at read time, aggregate the live state with the cached snapshots via one of: **(i) Residual Memory** (sum), **(ii) Gated Residual Memory (GRM)** — input-dependent gates `γ = ⟨u, MeanPool(segment)⟩ ∈ [0,1]`, **(iii) Memory Soup** (weight-average the cached layer params; matters for non-linear memory), **(iv) Sparse Selective Caching (SSC)** — a top-k router over cached states. The paper's own claim is "applicable to any recurrent update rule — add caching, don't redesign the architecture."
- **Two application points in THIS stack** (test the dual-tower one first — it is the on-point fit):
  1. **MTL dual-tower gated read (primary, aligns with the regime finding).** Treat the cat-tower and reg-tower representations as the "memories"; let an **input-dependent GRM gate** decide how much each task reads from a shared cached representation, and/or an **SSC top-k router** = a tiny mixture-of-task-experts at the read step. This is a *continuous, input-dependent* generalization of R2's binary STEM-AFTB forward/backward masks — opportunistic transfer where it helps, isolation (cos≈0) where it doesn't. **Run R2 first**; R10 is "what if the AFTB gate were learned and input-conditioned?"
  2. **Check2HGI hierarchical fusion (the literal "on the layers", speculative).** Replace the fixed sums/skip-connections across the check-in→POI→region→city encoder levels with **GRM-gated or Memory-Soup fusion** — a learned, input-dependent multi-resolution readout over the hierarchy levels. The paper never touches GNNs, so this is the more extrapolative of the two; scope it only if (1) shows signal. Note: changing the substrate encoder is STL-axis work that, per the regime finding, may not express under MTL — validate at STL first (coordinate with `pre_freeze_gates`/`embedding_eval` conventions).
- **Promote-gate (if a working impl is reached):** ≥0.3 pp either head over champion G, AL+FL seed 0 → multi-seed {0,1,7,100}. **Falsifier / stop conditions:** the gate cannot fire if (a) the reimplemented primitive does not actually fire (assert the gates/router are non-trivial — C28 dead-codepath rule), or (b) it merely reproduces R2/G within noise (then GRM ≡ the hand-built asymmetry, a citable null). Because window length is 9, do **not** invest in the segment-caching/long-context machinery itself — only the gated-read primitive.

## Protocol (match the family so results are freeze-eligible)

- Substrate **v14** (`check2hgi_design_k_resln_mae_l0_1`) or the blessed base at run time; champion **G**
  baseline (`mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` + `next_gru`, unweighted CE).
- Screen at AL+FL seed 0 first (cheap discriminator + the scale-conditional check). **Any positive →
  multi-seed {0,1,7,100} before claiming** — the small-state-false-negative lesson
  ([`community_insights.md` #3](../../research/community_insights.md)) cuts both ways.
- Leak hygiene: per-fold per-seed train-only priors; freshness preflight (`CLAUDE.md` stale-log_T rule);
  paired Wilcoxon, report n and p.

## Hand-off

Each R-lever closes with a one-paragraph verdict + the gate decision. Promoted levers → write a gate row
into [`../closing_data/PLAN.md`](../closing_data/PLAN.md) G0.2 and STOP for user sign-off (recipe → v17).
Nulls → `STATE.md` + a `docs/studies/log.md` row. Mechanism narrative → `FINDINGS.md` here.
