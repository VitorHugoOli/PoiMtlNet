# Paper Draft — Title, Abstract, and Writing Targets

> ⚠ **DEPRECATED 2026-05-01 — superseded by `articles/[BRACIS]_Beyond_Cross_Task/PAPER_DRAFT.md`.**
>
> The BRACIS 2026 submission scratch now lives in the article folder, with the leak-free closure (`PAPER_CLOSURE_RESULTS_2026-05-01.md` + `FINAL_SURVEY.md` + `CLAIMS_AND_HYPOTHESES.md §CH22 reframe`) absorbed into the story.
>
> **Headline reframe.** The committed title and 130-word abstract below are anchored to the F48-H3-alt "Per-Head LR" finding and the F49 "+6.48 pp MTL > STL on AL" attribution. Both are now stale: F49's AL win was a leak artefact (asymmetric C4 leak inflated MTL more than STL — see `PAPER_CLOSURE_RESULTS_2026-05-01.md §3, §4a`). Under leak-free measurement, MTL trails STL on `next_region` at every state by 7–17 pp; the honest story is the **scale-sensitive classic MTL tradeoff** (substrate carries cat, architecture costs reg, cost shrinks with data scale on the small-to-medium regime).
>
> **Article-side working title (locked-default 2026-05-01):** *Beyond Cross-Task Transfer: Check-In-Level Embeddings and the Scale-Sensitive Multi-Task Tradeoff in POI Prediction*. See `articles/[BRACIS]_Beyond_Cross_Task/PAPER_DRAFT.md §0` for the live decision register and `articles/[BRACIS]_Beyond_Cross_Task/AGENT.md` for sub-agent guidance.
>
> **What this file is now.** Audit / historical record of the pre-leak-free framing, the F-trail derivation, and the `BRACIS_GUIDE.md §10` title/abstract style references. Useful for: tracing how the title evolved; auditing the F49 → F50 reframe; sourcing the target-phrase library in §4 (most still apply, modulo the legacy "transfer null" framing). **Do not edit for the BRACIS submission** — edit the article-side file instead.

---

**Created:** 2026-04-28. **Owner:** This file held the committed title and abstract for the BRACIS 2026 submission, plus per-section drafting targets, decision flags, and target phrases distilled from study findings. **Deprecated 2026-05-01** — see banner above.

**Read alongside:**
- [`PAPER_STRUCTURE.md`](PAPER_STRUCTURE.md) — table layout, baseline policy, scope decisions
- [`PAPER_PREP_TRACKER.md`](PAPER_PREP_TRACKER.md) — TODO list, paper-blocking experiments, risk register
- [`NORTH_STAR.md`](NORTH_STAR.md) — committed champion config (F48-H3-alt)
- [`CLAIMS_AND_HYPOTHESES.md`](CLAIMS_AND_HYPOTHESES.md) — claim catalog (CH16–CH21)
- [`MTL_ARCHITECTURE_JOURNEY.md`](MTL_ARCHITECTURE_JOURNEY.md) — F-number trail and design derivation
- [`../../BRACIS_GUIDE.md`](../../BRACIS_GUIDE.md) §10 — empirical patterns from 13 sampled BRACIS abstracts (the basis for title/abstract decisions below)

---

## 1. Title (committed 2026-04-28)

> **Beyond Cross-Task Transfer: Per-Head Learning Rates and Check-In-Level Embeddings for Multi-Task POI Prediction**

14 words, single colon, BRACIS-shape. Sits in pattern A (method-acronym + colon) of `BRACIS_GUIDE.md` §10.3, balanced toward pattern B (concept-for-task) with the subtitle.

### Why this title

- **"Beyond Cross-Task Transfer"** sets up the abstract's surprise: the conventional MTL-as-transfer framing is empirically null on this problem. This is the paper's most novel contribution and the lead reviewer hook.
- **"Per-Head Learning Rates and Check-In-Level Embeddings"** names both technical contributions in the subtitle. A reviewer scanning the proceedings can reproduce the recipe from the title alone.
- **"for Multi-Task POI Prediction"** anchors the domain, matching BRACIS topical fit (Machine Learning, Deep Learning, Graph Neural Networks).
- No math symbols, no undefined acronyms, no polemical "Not Transfer" register.

### Alternates considered (preserved for context, do not delete)

| Option | Trade-off |
|---|---|
| *When Multi-Task Learning Beats Single-Task POI Prediction: Per-Head Learning Rates and Check-In-Level Embeddings* | Defensive ("beats STL" framing); doesn't carry the transfer-null finding. Was the user's first preference. |
| *Per-Head Learning Rates and Check-In-Level Embeddings for Multi-Task POI Prediction* | BRACIS-classical concept-for-task; safer but loses the surprise hook. Backup option if reviewers push back on "Beyond" framing. |
| *Decomposing the Multi-Task Win in POI Prediction: Architecture, Substrate, and the Limits of Transfer* | Foregrounds the attribution analysis; de-emphasises the practical recipe. Best for a methodology-leaning audience. |
| *Architecture and Substrate, Not Transfer: A Multi-Task Approach to POI Category and Region Prediction* | Bold thesis-forward; "Not Transfer" reads polemical for BRACIS. |
| *Reconciling Joint Training in Multi-Task POI Prediction: Per-Head Learning Rates over Contextual Embeddings* | Borrows the "Embracing/Reconciling" pattern from the BRACIS 2023 best paper; less concrete than committed. |

---

## 2. Abstract (committed 2026-04-28)

### 2.1 Short-form for the LNCS PDF (130 words; fits LNCS 70–150 cap)

> Multi-task learning (MTL) for point-of-interest (POI) prediction typically uses POI-level graph embeddings shared across task heads, assuming joint training transfers signal between tasks. We empirically show this transfer is essentially null on joint next-category and next-region prediction, yet a single MTL model still substantially surpasses single-task and published baselines once two coupled design choices are paired. We propose check-in-level contextual embeddings (Check2HGI), which provide per-visit context, combined with a cross-attention backbone trained under per-head learning rates and no annealing. Under 5-fold user-disjoint cross-validation on multiple U.S.-state Gowalla splits, the proposal yields macro-F1 gains of up to 33 percentage points on next-category and outperforms strong neural single-task baselines on next-region Acc@10. An attribution analysis decomposing substrate, architecture, and joint training attributes the gain to architecture–substrate interaction, not transfer, and exposes a methodological pitfall in cross-attention task-weight-zero ablations.

**Word count:** 130 / 150.

### 2.2 Why this abstract works

Follows BRACIS abstract structure observed in 13 sampled accepted papers (`BRACIS_GUIDE.md` §10.2):
1. **Domain framing** (sentence 1): MTL for POI typically uses shared embeddings, assumes transfer.
2. **Gap statement** (sentence 2): empirically that transfer is null — yet method still works.
3. **"We propose"** (sentence 3): names Check2HGI + per-head LR mechanism in plain prose.
4. **Evaluation protocol + specific number** (sentence 4): 5-fold user-disjoint CV, multiple splits, "up to 33 pp".
5. **Attribution + methodological note** (sentence 5): closing differentiator — architecture × substrate, plus the cross-attn ablation pitfall.

### 2.3 Open decisions on the abstract (track at submission time)

| ID | Decision | Trigger | Default |
|---|---|---|---|
| **D1** | "Up to 33 percentage points" vs. "double-digit gains across splits, exceeding 30 pp on the largest" | Want to emphasise consistency | Use "up to 33 pp" — accurate, hedged with "up to", leaves room for variation |
| **D2** | "Multiple U.S.-state Gowalla splits" assumes Phase 2 (FL+CA+TX) lands | Phase-2 status at submission | If incomplete, change to "U.S.-state Gowalla splits" — defensible at AL+AZ+FL |
| **D3** | Reg-side baselines named (STAN, ReHDM) or kept qualitative | Word budget | Keep qualitative; named in body and tables |
| **D4** | Coin a paper-internal MTL-method acronym (e.g., MTLnet-PHL)? | Branding consistency | Default no; Check2HGI is the named noun |
| **D5** | Long-form abstract for JEMS3 short-form field if it accepts >150 chars | Check JEMS3 form on submission day | Reuse the 130-word version unless JEMS3 explicitly invites long prose |

### 2.4 Long-form abstract (DRAFT, for JEMS3 if applicable)

Not yet revised. The 280-word long-form discussed 2026-04-27 needs the same fixes as the short:
- Drop the "+14 pp / ≥9σ" line (legacy internal claim, not external; cf. CLAIMS_AND_HYPOTHESES §CH20 framing).
- Unify reg-side baseline framing as qualitative rather than named.

To be drafted as needed.

---

## 3. Section-by-section writing targets

Following `BRACIS_GUIDE.md` §7 page-budget guidance (15 pages including refs and appendices). Quoted page allotments are **suggested**, not binding.

### 3.1 Introduction (~1.5 pages)

**Target opening sentences (3 for context):**
- POI prediction supports location-based services, urban mobility analysis, and recommendation pipelines.
- A common approach is multi-task learning (MTL), jointly predicting POI category and region from check-in trajectories.
- The conventional rationale is **cross-task transfer over a shared embedding** — that joint training shares signal between heads.

**Pivot to gap (3 sentences):**
- We re-examine this assumption with a controlled architectural decomposition and find that conventional task-supervision transfer is **essentially null** on the joint next-category / next-region problem.
- Yet a single-model MTL still substantially surpasses both single-task and published baselines once the substrate and the optimiser regime are paired.
- This motivates a **re-attribution**: substrate and architecture, jointly, are the locus of the multi-task win — not transfer.

**Contribution bullets (3–4 bullets, each one sentence):**
- **(C1) Substrate finding.** Replacing POI-stable graph embeddings with check-in-level contextual embeddings (Check2HGI) yields head-invariant macro-F1 gains on next-category, with a POI-pooled counterfactual attributing the dominant fraction of the gap to per-visit context rather than training signal.
- **(C2) Architectural finding.** Per-head learning rates without annealing on cross-attention MTL exceed matched-head single-task ceilings on next-region while preserving category F1; three orthogonal scheduling controls (loss-side ramp, gentle constant LR, warmup-then-plateau) each fail.
- **(C3) Attribution analysis.** A three-way decomposition (encoder-frozen / loss-side / full MTL) shows category-supervision transfer ≤ |0.75| pp per state, isolating an **architecture–substrate interaction** as the locus of the joint lift.
- **(C4) Methodological note.** Under cross-attention, loss-side `task_weight=0` ablations silently train the silenced encoder via attention K/V projections; encoder-frozen isolation is the only clean architectural decomposition. This caveat applies beyond POI to any cross-task MTL with attention-based interaction (MulT, InvPT, HMT-GRN).

### 3.2 Related Work (~1.5 pages)

**External baselines to cite and position:**
- POI category prediction: Capanema et al. (POI-RGNN), Zeng et al. (MHA+PE).
- POI region prediction: Luo et al. (STAN, WWW'21), Lim et al. (HMT-GRN), MGCL.
- MTL gradient methods: NashMTL (Navon et al.), PCGrad (Yu et al.), GradNorm (Chen et al.), CAGrad (Liu et al.), Aligned-MTL, FAMO. **Position**: we use `static_weight`, not gradient surgery — the contribution is *not* a new MTL optimiser.
- Hierarchical graph embeddings: HGI (Hierarchical Graph Infomax), POI2HGI lineage, Check2HGI provenance.
- Cross-attention multi-task models: MulT, InvPT — relevant because the methodological note (C4) generalises.

**Phrase to land in this section:**

> "Our work re-examines a default assumption in MTL-for-POI: that joint training transfers signal between tasks. We find this assumption empirically vacuous in the cross-attention regime, motivating a different attribution."

### 3.3 Method (~3 pages)

Sub-sections in narrative order:
- **3.3.1 Notation and task setup.** Next-category (7-class), next-region (~4.7K-class for the largest split), sequence length 9 check-ins, user-disjoint StratifiedGroupKFold.
- **3.3.2 Substrate.** HGI vs. Check2HGI — per-place vs. per-visit. One paragraph on why per-visit context matters at the cat task (mechanism finding).
- **3.3.3 MTL backbone.** Cross-attention architecture, per-task input modality (check-in stream → cat head, region stream → reg head), head registry choices.
- **3.3.4 Optimiser regime.** Per-head learning rates with no annealing — derivation from F44–F48 chain (in supplementary; main text gives the *what*, not the *F-number trail*).
- **3.3.5 Head choices.** `next_gru` (cat) and STAN-Flow (`next_stan_flow`) (reg, STAN + α·log_T graph prior), motivated as matched-head pairs.

**Don't write in main text:** the F-number experiment trail. Belongs in supplementary materials or appendix as an attribution narrative.

### 3.4 Experimental Setup (~2 pages)

- Datasets: Gowalla US-state splits — list per `PAPER_STRUCTURE.md` §3 with check-in counts, user counts, region counts, category distribution.
- Folds: 5-fold StratifiedGroupKFold, user-disjoint, seed 42.
- Baselines: per `PAPER_STRUCTURE.md` §3.1 (cat: Majority, Markov-1-POI, POI-RGNN, MHA+PE) and §3.2 (reg: Majority, Top-K, Markov-{1..9}-region, STL GRU, STAN, ReHDM).
- Metrics: macro-F1 (cat primary), Acc@1/5/10 + MRR (reg primary).
- Statistical tests: paired Wilcoxon signed-rank (per fold), TOST non-inferiority where applicable.
- Hardware and reproducibility note (anonymised code link via Anonymous GitHub).

### 3.5 Results & Analysis (~4 pages)

Tables (per `PAPER_STRUCTURE.md` §5):
- **T1.** Per-state headline: per-state block with all baselines + STL + MTL champion.
- **T2.** Cross-state summary: best-of-each per state.
- **T3.** Substrate Δ (Check2HGI vs. HGI under matched head): 8 cells minimum for AL+AZ; expand to FL/CA/TX as Phase 2 lands.
- **T4.** Three-way decomposition (encoder-frozen / loss-side / full MTL) by state.

Figures (TBD; `PAPER_PREP_TRACKER.md` §3.2 still flags figure list as TODO):
- **F1.** Per-visit-context mechanism (POI-pooled counterfactual), bar chart.
- **F2.** Three-way attribution chain — schematic + per-state numbers.
- **F3.** Per-head LR ablation: champion vs. three negative controls (F40 loss-side, F48-H1 gentle constant, F48-H2 warmup-then-plateau).

### 3.6 Methodological Appendix or Limitations (~1.5 pages)

- **The cross-attn `task_weight=0` ablation pitfall.** Formal claim, gradient-flow argument, regression test pointers (`tests/test_regression/test_mtlnet_crossattn_lambda0_gradflow.py`). Frame as a generalisable methodology note, not a bug fix. (See `research/F49_LAMBDA0_DECOMPOSITION_GAP.md`.)
- **Limitations.** AL/AZ are dev-scale (10K–26K rows) used as ablation beds; FL Markov-saturation on Acc@10 (per `CONCERNS.md` §C02); CA/TX scale dependency if Phase 2 partial at submission.

### 3.7 Discussion & Conclusion (~1.5 pages)

- Re-attribution narrative: **substrate + architecture > transfer**.
- Practical recipe summary (one paragraph, citation-ready for follow-up work).
- Future directions: NS-2 hybrid cross-attn cat + dselectk reg; multi-seed n=3 robustness; encoder enrichment (P6 research track).

### 3.8 References (~0.5 pages, 25–30 refs max)

Per `BRACIS_GUIDE.md` §7. Tight bibliography; prefer recent (2020+) for related work, classic for foundations.

---

## 4. Hooks / target phrases (ready to drop into prose)

Strong distillations from study findings, suitable verbatim or near-verbatim:

- *"Conventional MTL framing assumes cross-task transfer over a shared embedding; our attribution analysis shows this transfer is bounded by ≤ |0.75| pp per state."*
- *"The MTL win is interactional — substrate and architecture together — not transferable in the conventional sense."*
- *"Per-head learning rates without annealing isolate the unique design satisfying joint Pareto: three orthogonal scheduling controls (loss-side ramping, gentle constant LR, warmup-then-plateau) each fail."*
- *"Under cross-attention, a loss-side `task_weight=0` ablation silently trains the silenced encoder via attention K/V projections; encoder-frozen isolation is the only clean architectural decomposition."*
- *"Per-visit context, not training signal, accounts for approximately 72% of the substrate gain on next-category."*
- *"The category head and region head occupy disjoint optimal learning-rate regimes; under a single shared schedule, one head necessarily collapses."*

---

## 5. Cross-references to authoritative artefacts

- **Title rationale and BRACIS norms:** `../../BRACIS_GUIDE.md` §10
- **Champion config (F48-H3-alt):** `NORTH_STAR.md`
- **Claim catalog (CH16–CH21):** `CLAIMS_AND_HYPOTHESES.md`
- **Paper-blocking experiments and risk register:** `PAPER_PREP_TRACKER.md`
- **Table layout, baseline policy, scope:** `PAPER_STRUCTURE.md`
- **Phase-2 launch state:** `PHASE2_TRACKER.md`
- **Most recent operational state:** `SESSION_HANDOFF_2026-04-27.md`
- **F49 attribution analysis:** `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`
- **Cross-attn λ=0 methodological note:** `research/F49_LAMBDA0_DECOMPOSITION_GAP.md`
- **Substrate comparison (Phase-1):** `research/SUBSTRATE_COMPARISON_FINDINGS.md`
- **MTL architecture journey (F-trail):** `MTL_ARCHITECTURE_JOURNEY.md`

---

## 6. Status

- Title: **committed**
- Short-form abstract (130w): **committed** (open decisions in §2.3)
- Long-form abstract: **DRAFT pending** (only if JEMS3 form invites it)
- Section-by-section targets: **outline only**, prose drafting not started
- Figures: **TODO** (PAPER_PREP_TRACKER P10 / §3.2)
- Methodological appendix structure: **decision pending** (`PAPER_PREP_TRACKER` D7 — stand-alone vs. merged with Limitations)
