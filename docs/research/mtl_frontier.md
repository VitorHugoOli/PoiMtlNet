# The MTL Frontier (2023–2026) — and How to Produce a Real MTL Gain in This Study

> Compiled 2026-06-12 from a targeted web survey of the post-2022 multi-task-learning literature, read against this repo's established regime findings (`docs/studies/archive/mtl_improvement/FINAL_SYNTHESIS.md`). Two questions: **(1)** what is the current MTL frontier this research could explore; **(2)** which mechanisms can plausibly produce *genuine* positive transfer in this project's measured regime — 2 tasks, shared-trunk gradient cosine ≈ 0, all ~19 classical optimizers null, wins only from architectural asymmetry (dual-tower) and an output-level KD prior (log_T-KD).

---

## 1. Where the field stands: the post-optimizer consensus

The Kurin/Xin NeurIPS 2022 result ("scalarization suffices") has **not** been overturned by any 2024–2026 paper on performance grounds. The field's responses split four ways:

| Thread | Representative work | Status |
|---|---|---|
| Cheap loss-history weighting | FAMO (NeurIPS 2023, [arXiv:2306.03792](https://arxiv.org/abs/2306.03792)) | Still a weighting scheme; expected null at K=2 with a tuned static weight |
| Scalarization-at-scale | Royer et al. (NeurIPS 2023, [arXiv:2310.08910](https://arxiv.org/pdf/2310.08910)); AutoScale ([arXiv:2508.13979](https://arxiv.org/pdf/2508.13979)) | Optimizers demoted to *weight-pickers*; effort goes into the weight search |
| Theoretical rebuttal | Hu et al. (NeurIPS 2023, [arXiv:2308.13985](https://arxiv.org/abs/2308.13985)) | Linear scalarization can't trace non-convex Pareto *fronts* — a coverage argument, not a single-point performance argument |
| Why conflict metrics were never predictive | **Mueller et al., TMLR 2025 ([arXiv:2408.14677](https://arxiv.org/pdf/2408.14677))** | MTL's generalization gap vs STL appears *early, at matched training loss*; gradient conflict correlates with optimization friction but **does not predict transfer** |

**Mueller et al. is the single most important external citation for this project**: it independently explains the repo's own trajectory — why 19 optimizer arms were null, why cos(∇cat,∇reg)≈0 told us nothing about transfer, and why the gap turned out to be representational ("the shared-backbone pathway is the wall"). Cite it next to Kurin/Xin in the new paper's related work. Consensus anchor for the narrative: the *Multitask Learning 1997–2024* retrospective (HDSR 2025, [link](https://hdsr.mitpress.mit.edu/pub/7fcc3jhv)).

**Implication**: this repo's null results are now the *expected* outcome in the general literature. The frontier moved to (a) architecture/modularity, (b) output-level coupling, (c) merging/Pareto-profiling, (d) data/task selection — and the repo's two wins already live in (a) and (b).

---

## 2. Frontier map, with relevance verdicts for this regime

### 2.1 Asymmetric / modular architectures — *the institutionalized form of the dual-tower win*

- **STEM** (AAAI 2024, Tencent, [arXiv:2308.13537](https://arxiv.org/abs/2308.13537), [code](https://github.com/LiangcaiSu/STEM)): negative transfer lives in the shared *embedding* layer, not just the towers; fix = task-specific embeddings + **All-Forward, Task-specific-Backward (AFTB) gating** — a task may *read* other tasks' representations in the forward pass, but gradients flow only to its own. **This is precisely the repo's "private reg tower + cat harvests the shared encoder" finding, published as an industrial pattern.** Highest-relevance architectural citation and the most direct next experiment (§4, R2).
- **AdaTT** (KDD 2023, Meta, [arXiv:2304.04959](https://arxiv.org/abs/2304.04959)): learned per-level task-pair fusion units — a parametric superset of the hand-built dual-tower asymmetry; one ablation candidate.
- **HoME** (KDD 2025, Kuaishou, [arXiv:2408.05430](https://arxiv.org/abs/2408.05430)): diagnoses **expert collapse** and **expert degradation** in MMoE/PLE trunks. Relevant as a diagnostics manual: the repo's MMoE/CGC nulls were never autopsied for expert health.
- **PEFT-MTL** (Polytropon/C-Poly [arXiv:2312.03248](https://arxiv.org/html/2312.03248), MTLoRA CVPR 2024 [arXiv:2403.20320](https://arxiv.org/abs/2403.20320)): LoRA-per-task over a **frozen shared trunk** — cheap to pilot on the frozen Check2HGI substrate.
- **AITM** (KDD 2021, [arXiv:2105.08489](https://arxiv.org/abs/2105.08489)): learned transfer module along a sequentially-dependent task chain — the canonical conditional-chain (cat→reg) tower design.

**Verdict**: high relevance. The field converged on *separate what conflicts, share selectively, make sharing asymmetric (forward-yes / backward-no)* — convergent, citable validation of champion G's design logic.

### 2.2 Output-level (prediction-level) coupling — *the family of the log_T-KD win, and the highest-yield direction*

- **ESMM** (SIGIR 2018, [arXiv:1804.07931](https://arxiv.org/pdf/1804.07931)): tasks coupled by a probability-chain *identity* in label space (pCTCVR = pCTR·pCVR). Template question for this project: is there an algebraic identity tying the two heads? There is — **P(region) = Σ_cat P(region|cat)·P(cat)** through a region×category co-location matrix (§4, R1).
- **CrossDistil** (AAAI 2022, [arXiv:2202.09852](https://arxiv.org/abs/2202.09852)): one task's *calibrated predictions* as supervision for another, with error-correction for the synchronous teacher. **log_T-KD is a static-teacher instance of exactly this**; the frontier version replaces the Markov teacher with the live calibrated category head.
- **Taskology** (CVPR 2020, [arXiv:2005.07289](https://arxiv.org/pdf/2005.07289)): *analytic consistency losses between outputs of separately-trained networks — no shared trunk required.* Maximally compatible with the regime finding that representation-level sharing is the wall.
- **MTPSL** (CVPR 2022, [arXiv:2111.14893](https://arxiv.org/pdf/2111.14893)): learned label-space mappings between tasks as auxiliary supervision.

**Verdict**: highest expected yield. The only lever class that has *ever* moved MTL reg in this repo (prior-pathway) is this family; the literature offers three concrete unexplored extensions of it.

### 2.3 Model merging / task arithmetic — *test with care, expectations bounded by theory*

Task vectors (ICLR 2023, [arXiv:2212.04089](https://arxiv.org/abs/2212.04089)), TIES (NeurIPS 2023), DARE (ICML 2024), AdaMerging (ICLR 2024, [arXiv:2310.02575](https://arxiv.org/abs/2310.02575)). Key theory: **weight disentanglement is an emergent property of pre-training** (Ortiz-Jimenez et al., NeurIPS 2023 oral, [arXiv:2305.12827](https://arxiv.org/abs/2305.12827)) — from-scratch STL experts don't share a basin, so naive merging is expected to fail here. The applicable variants: **ZipIt!** (ICLR 2024, [arXiv:2305.03053](https://arxiv.org/pdf/2305.03053)) merges without shared fine-tuning init via feature correlation, with *partial-depth* merging (share early, privatize late — the merging-side mirror of the dual-tower); **SIMO** ([arXiv:2504.11268](https://arxiv.org/abs/2504.11268), 2025) targets single-input multi-output topologies like this one. Evidence that merging beats joint training exists ([arXiv:2410.10801](https://arxiv.org/html/2410.10801v1), [arXiv:2410.15035](https://arxiv.org/pdf/2410.15035)) but is conditional on conflict/imbalance being the joint-training failure mode — which, post-C25, this project does *not* have.

**Verdict**: medium relevance; one honest merge-vs-joint experiment (warm-started shared trunk → two STL specialists → ZipIt partial merge vs champion G) would be novel in the LBSN space, with modest expected gain.

### 2.4 Pareto-front profiling — *a publishable upgrade over point selection*

PHN (ICLR 2021), Pareto Manifold Learning (ICML 2023, [arXiv:2210.09759](https://arxiv.org/abs/2210.09759)), and especially **PaLoRA** (ICLR 2025, [arXiv:2407.08056](https://arxiv.org/pdf/2407.08056)): frozen/shared base + one small LoRA adapter per task; the convex mixture of adapters parameterizes the whole cat↔reg trade-off curve at inference. **This directly addresses the project's checkpoint-selector saga (C21/geom_simple)**: instead of selecting one joint epoch by a scalar, publish the front. At K=2 this is cheap and would be a first in the POI-MTL literature.

### 2.5 Task grouping & auxiliary learning

TAG (NeurIPS 2021, [arXiv:2109.04617](https://arxiv.org/abs/2109.04617)) — its *lookahead affinity* (effect of a task-i update on task-j loss) is signed and directional, unlike cosine, and would make a better diagnostic for the paper's mechanism section. **ForkMerge** (NeurIPS 2023, [arXiv:2301.12618](https://arxiv.org/pdf/2301.12618)) — fork into branches with different task-weight vectors, select/merge by *validation* error; explicitly rejects gradient-conflict explanations; optimizer-agnostic and works at K=2. AuxiLearn / Auto-λ formalize the "treat cat as auxiliary to reg (or vice versa)" reframing. Per-instance task-relatedness via data attribution ([arXiv:2505.21438](https://arxiv.org/html/2505.21438v1), 2025) opens the one genuinely unexplored axis here: **per-sample transfer gating** (§4, R6).

### 2.5b Layer-level gated memory — Memory Caching / GRM (arXiv 2602.24281, user-flagged)

*Memory Caching: RNNs with Growing Memory* (Behrouz et al., Google Research, Feb 2026, [arXiv:2602.24281](https://arxiv.org/abs/2602.24281); no code). Its headline — growing memory for long-context recall — does **not** apply here (fixed length-9 windows). What transplants is the **read-time aggregation primitive applied at the layer level (not as a transformer-block swap)**: snapshot a layer's state and aggregate via **Gated Residual Memory** (input-dependent gates `γ=⟨u, MeanPool⟩∈[0,1]`), **Memory Soup** (weight-average cached layer params), or **Sparse Selective Caching** (top-k router over states). Two fits in this project, both pursued in the `mtl_frontier` study (R10): (a) a GRM-gated / SSC-routed read **between the dual towers** — a continuous, input-conditioned generalization of R2's binary STEM-AFTB masks (the on-point fit for the cos≈0 / asymmetric-sharing regime); (b) GRM/Memory-Soup **fusion across the Check2HGI hierarchy levels** (check-in→POI→region→city), replacing fixed sums — the literal "on the layers" reading, speculative, STL-first. Honest scoping: from-scratch reimplementation; run R2 first; gate = ≥0.3 pp over G, multi-seed.

### 2.6 Residual optimizer tests (only two non-vacuous ones left)

- **BayesAgg-MTL** (ICML 2024, [PMLR](https://proceedings.mlr.press/v235/achituve24a.html)) — weights by per-dimension gradient *uncertainty*, not conflict; the only optimizer family whose mechanism isn't vacuous at cos≈0. (Note: `bayesagg_mtl` already exists in `src/losses/registry.py` — verify the implementation matches the ICML 2024 method and whether it was part of the 19-arm null sweep before re-running.)
- **Smooth Tchebycheff scalarization** (ICML 2024, [arXiv:2402.19078](https://arxiv.org/pdf/2402.19078)) — worth one run only if the measured cat↔reg front turns out non-convex (which §2.4 would reveal).

Everything else in the optimizer aisle is closed for this project.

---

## 3. Is this project "a new frontier of MTL in LBSN"? — honest assessment

**Yes, with precision about what the frontier claim is.** The published LBSN-MTL literature (MCARNN 2018, iMTL 2020, HMT-GRN 2022, CSLSL 2024 — see [`literature_review.md §4`](literature_review.md)) is, by general-MTL standards, architecturally early-2018: hard sharing or cascades, uniform/static weights, no gradient diagnostics, no optimizer evaluation, no Pareto analysis, no STL-ceiling controls, and auxiliary-task framing throughout. **No published LBSN work has run the post-2022 MTL research program** (optimizer sweeps with nulls reported, gradient-geometry measurement, asymmetric privatization with stop-gradient harvesting, Pareto-non-inferiority against *tuned* STL ceilings, output-level distillation priors). This repo has — and the industrial RecSys frontier (STEM, AdaTT, HoME) independently converged on the same two winning mechanisms, which is corroboration, not anticipation: none of those works touches spatio-temporal/LBSN tasks.

So the defensible frontier claim is: **"the first rigorous MTL regime study in LBSN POI prediction"** — bringing the modern MTL toolkit to this domain and showing which of its lessons replicate (scalarization suffices; asymmetric sharing pays; output-level priors are the transfer channel). What remains *not* claimable: any individual mechanism as new-to-ML (each has a 2018–2024 antecedent). This is a *domain-frontier* contribution, not a *methods-frontier* contribution — and for the LBSN audience (BRACIS included) that is a legitimate, strong framing.

---

## 4. How to produce a REAL MTL gain in the current study — ranked program

Grounded in the repo's own causal picture: the shared-backbone joint regime suppresses reg-side substrate/encoder gains (rising-tide rule, anchor dominance: `docs/studies/archive/substrate-protocol-cleanup/CLOSURE.md`); genuine region→cat transfer is small (+0.93 pp FL, −0.67 AL); the only proven MTL-reg lever is the prior pathway (log_T-KD, +2–5 pp AL/AZ, n=20, p=9.5e-07). Therefore: transfer must enter at the **output level** or through **asymmetric read-only sharing** — not through the trunk.

| Rank | Experiment | Mechanism & why it can work *here* | Cost | Falsifier |
|---|---|---|---|---|
| **R1** | **log_C co-location prior + probability-chain coupling**: build a train-only region×category matrix P(region\|cat) (the exact analog of log_T, same per-fold per-seed infrastructure), then (a) KD-distill cat-head-weighted region prior into the reg head: prior(reg) = Σ_c P(reg\|c)·P̂(c); (b) optionally the reverse P(cat\|last-region) for the cat head | Extends the **only proven lever class**; ESMM-style label-space identity; touches no shared trunk | Low — reuses `compute_region_transition.py` patterns | If Δreg ≤ log_T-KD alone (redundant with Markov-1) |
| **R2** | **STEM-style AFTB formalization**: parameterize the dual-tower's sharing as explicit all-forward/task-specific-backward gates (per-layer stop-gradient masks), sweep which layers cat may read vs own | Institutionalizes the architecture win; converts a hand-built design into a measured dose-response; directly citable against STEM | Low-medium | If the current G configuration is already the optimum of the sweep |
| **R3** | **Live cross-task distillation (CrossDistil)**: calibrated cat-head posterior as a *dynamic* teacher term for the reg head (and/or reverse), with warm-up gating to avoid early-epoch noise | Generalizes log_T-KD from static Markov teacher to learned teacher; the literature's named version of this repo's mechanism | Low | If calibrated-teacher ≤ static log_T teacher everywhere |
| **R4** | **Pareto-front profiling (PaLoRA-style)**: frozen trunk + per-task LoRA adapters; publish the cat↔reg front instead of one geom_simple point | Replaces point-selection with curve-selection; resolves the C21 selector class of problems permanently; novel in POI-MTL | Medium | Front collapses to a point (tasks fully decoupled) — itself a publishable regime datum |
| **R5** | **Per-instance KD gating**: modulate the R1/R3 prior weight per check-in (e.g., by Markov-coverage of the last region, user-trajectory entropy, or sequence length) | The 2025 per-sample-transfer axis; plausible because Markov-1 binds at dense states (FL ≥85% coverage) but not sparse ones — a global W is provably suboptimal | Low | Gated ≤ global-W on all states |
| **R6** | **ForkMerge-style weight forking**: periodic forks with different (w_cat, w_reg), select/merge on val | Validation-driven, conflict-agnostic, K=2-friendly | Medium | Merged ≤ champion G |
| **R7** | **Merge-vs-joint (ZipIt/SIMO)**: warm-start shared trunk → fine-tune STL specialists → partial-depth merge | Honest test of the merging frontier; expectation bounded by tangent-space theory (from-scratch caveat) | Medium | Merge < G (likely; still a citable negative for LBSN) |
| **R8** | **Auxiliary third task — next-visit time**: GETNext/Where-and-When pattern (time loss as auxiliary) | The literature's most-used auxiliary; but the rising-tide rule predicts it lifts STL too — run with paired STL control | Low | Lifts STL and MTL equally (rising tide) |
| R9 | Residual optimizer sanity arms: BayesAgg-MTL (verify registry implementation), Smooth-Tchebycheff (only if R4 shows a non-convex front) | Closes the optimizer aisle citably | Trivial | Expected null |

**Sequencing note**: R1+R3 (prior-pathway extensions) and R2 (AFTB sweep) are pre-freeze-gate compatible with `closing_data` — if any promotes the recipe (≥0.3 pp either head, the existing G0.1 gate convention), it becomes v17 *before* the full base regeneration, exactly as the closing_data plan anticipates. R4 (front profiling) is paper-narrative work and can run on the frozen champion afterward.

**What NOT to pursue** (closed by repo evidence + literature): more gradient-surgery/balancing arms beyond R9; symmetric dual-towers (G′ falsified at small states); substrate improvements aimed at MTL (regime finding — five independent falsifications); curriculum/freezing schedules (C2 null, P4 null).
