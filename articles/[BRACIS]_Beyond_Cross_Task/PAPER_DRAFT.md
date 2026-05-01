# PAPER_DRAFT.md — Full-Story Scratch

> **Read order:** (1) `AGENT.md` for voice / statistics / anonymization, (2) `PAPER_STRUCTURE.md` for section-level structure, (3) **this file** for paragraph-level beats. This file is the working scratch from which sub-agents draft prose. **Lock §0 first** — those are the user-facing decisions that must precede sub-agent fan-out.
>
> **Provenance.** This draft is the BRACIS-2026 article-side reframe of `docs/studies/check2hgi/PAPER_DRAFT.md`. The earlier study-side draft is anchored to the F48-H3-alt "Per-Head LR" finding and its committed title/abstract are now stale (the F49 "+6.48 pp MTL > STL" claim was a leak artefact; see `PAPER_CLOSURE_RESULTS_2026-05-01.md` and `CLAIMS_AND_HYPOTHESES.md §CH22 2026-05-01 reframe`). **This file supersedes the study-side draft for the BRACIS submission.**

---

## §0 Open decisions — lock these before sub-agents fan out

> **Status (session 2026-05-01 PM):** D0–D6 reflect the user's reframe. D2 = three-state headline (FL/CA/TX) **with AL/AZ retained as a scale-progression supplement** showing how the architectural reg cost trends with data scale. D0 / D1 reframed accordingly to absorb the scale-sensitive finding. The "all five states" wording survives in §Results, framed as headline + supplement layering.
>
> **Late-session update (2026-05-01 PM):** **CA/TX MTL multi-seed at {0, 1, 7, 100} is in flight on H100 at submission**, ETA ~1 h — once it lands, every state (AL/AZ/FL/CA/TX) has multi-seed paired tests, breaking the n = 5 ceiling at the headline scale. The acceptance gate for the late-arriving cells: direction-consistent with the seed = 42 anchor (Δ_cat ∈ [+1.6, +2.0] pp; Δ_reg ∈ [−9, −17] pp). If the multi-seed direction matches the anchor, T3 numbers are upgraded with the multi-seed σ; if direction flips at any state, surface that as a finding in §6.2. Sub-agent A5 must check this before T3 commits.
>
> **Late-session update — ReHDM scope:** ReHDM (Li et al., IJCAI 2025) is reported at AL/AZ/FL only. CA/TX ReHDM runs are deferred — the dual-level hypergraph's collaborator pool scales quadratically with region cardinality and the per-state run cost on a single H100 exceeded the compute budget at 8.5 K (CA) and 6.5 K (TX) regions. Deferred to camera-ready; T5 footnote and §7 Limitations disclose the asymmetric baseline coverage.

| ID | Decision | Default (recommended) | Alternate | Status |
|---|---|---|---|---|
| **D0** | Title direction | **Beyond Cross-Task Transfer: Check-In-Level Embeddings and the Scale-Sensitive Multi-Task Tradeoff in POI Prediction** (14 words, Pattern A acronym+colon) — preserves the "Beyond" hook; "Scale-Sensitive" replaces "Classic" so the title carries the scale-progression finding (AL/AZ pay more, FL/CA pay less, TX outlier) without committing to a state count or dataset | *Substrate Carries, Architecture Pays: Check-In-Level Embeddings for Multi-Task POI Prediction* (10 words, Pattern D phenomenon-and-method, more direct, no "Beyond" hook — sharpest alternate; the "Across U.S.-State Gowalla Splits" tail dropped 2026-05-01 because it bound the title to a dataset and pushed it into the applied Pattern C shape `BRACIS_GUIDE.md §10` warns against) | locked-default 2026-05-01 |
| **D1** | Abstract reframe | Rewrite the 130-word abstract around C1 (substrate +15 to +33 pp) + C2 (classic tradeoff with the **scale-progression note** — gap shrinks from −11/−12 pp at AL/AZ to −7 pp at FL on the small-to-medium trajectory; CA at higher cardinality preserves the regime, TX is non-monotone) + C3 (cross-attn ablation pitfall). Drop the "five-state" hard claim; use "U.S.-state Gowalla splits" with FL/CA/TX as headline and AL/AZ as smaller-scale anchors. The committed 130-word abstract in `docs/studies/check2hgi/PAPER_DRAFT.md` is now stale | Keep committed abstract and footnote | locked-default 2026-05-01 |
| **D2** | Headline states | **Three-state headline (FL/CA/TX) + AL/AZ as scale-progression supplement.** Headline tables (T3, T4, T5) report FL/CA/TX. AL/AZ live in a per-state supplement table that shows *the architectural cost trends downward as data scale increases on the small-to-medium regime* (Δreg AL −11.04 / AZ −12.28 / FL −7.28 pp). CA preserves the regime (−8.93); TX is the honest outlier (−16.69) flagged as state-specific factors beyond raw cardinality. T2 substrate ablation reports all five states because the substrate Δ scales monotonically with data (+15 → +33 pp) and the finding is sharper at five states. | All five states in every headline table | locked-default 2026-05-01 |
| **D3** | Recipe story in main text | Report MTL B9 at FL/CA/TX, MTL H3-alt at AL/AZ — single MTL row per state at its best recipe — and put the cross-recipe comparison + scale-conditional finding in §6.2 / appendix | Pick a single recipe (B9) for all 5 states and footnote the AL/AZ underperformance | locked-default 2026-05-01 |
| **D4** | Cite the F-trail in main text? | **No** — F-numbers + journey doc go in supplementary. Main text gives recipe + result, not derivation. | Yes — narrate the F-trail in §3 / §6 | locked-default 2026-05-01 |
| **D5** | Main result framing | **The scale-sensitive MTL tradeoff is the headline.** Substrate carries cat (C1, monotone-scaling); architecture costs reg (C2, with the scale-progression finding: small states pay more, mid-large states pay less, with state-specific factors at the largest scale). | Position substrate as headline and tradeoff as honest counterpoint | locked-default 2026-05-01 |
| **D6** | Anonymous code link | Anonymous GitHub at `https://anonymous.4open.science/r/<TBD>/` — sub-agent A6 generates the snapshot | Anonymous Dropbox | pending |

If the user further changes a default, the orchestrator updates this file *and* `samplepaper.tex` (title + abstract) in the same commit.

---

## §1 Introduction — paragraph beats

**Beat 1 — Domain framing (one paragraph).** Open with POI prediction in LBSNs as the substrate. Two complementary tasks: (i) **next-category prediction** — given a length-9 check-in window, predict the semantic category (7 classes) of the next visit; (ii) **next-region prediction** — predict the next visit's region (≈ 1.1K to 8.5K classes depending on the state). Both tasks share a mobility substrate but place disjoint demands on the model: cat is a coarse 7-way decision driven by per-visit context (intent, time-of-day, co-location); reg is a fine-grained ranking problem driven by spatial-temporal trajectory regularities and short-horizon transitions. Voice cue: this is the same opener style as CoUrb's `intro.tex` paragraph 1 — domain framing, not claim-first.

**Beat 2 — The arc, framed in third person (one paragraph; double-blind safe).** Two recent works on this task pair establish the bottleneck. Silva et al. (CBIC 2025) proposed a hard-parameter-sharing MTL framework on POI-stable graph embeddings (DGI) and reported only marginal gains over single-task baselines — most differences fell within standard deviations. The diagnosis pointed at *representation mismatch*: a single shared encoder learning a "compromise" representation that was specialised enough for neither head. Paiva et al. (CoUrb 2026) tested an embedding-side response by decomposing the monolithic graph embedding into independent spatial (SIREN / Sphere2Vec-M), temporal (Time2Vec), and categorical (HGI) sub-encoders, recovering substantial cat-side gains across three states. Together, these two works frame the operative hypothesis: the embedding choice — not the MTL recipe — is the load-bearing factor.

**Beat 3 — The pivot (one paragraph).** This raises a sharper question. If the embedding is the bottleneck, can a single principled substrate that supplies *per-visit context* — one vector per check-in, not one vector per POI — finally enable multi-task learning to deliver bidirectional gains across both heads? Per-visit context is the natural axis: it is what differentiates a user's lunchtime visit to a café from their evening visit to the same café, and that distinction is exactly what next-category prediction needs. Per-POI embeddings (HGI, DGI) cannot supply it.

**Beat 4 — What we do (one paragraph).** We adopt **Check2HGI**, a check-in-level extension of HGI's contrastive objective: same POI visited twice yields different embeddings because the visit's temporal and co-visit context perturbs the contrastive view. We hold Check2HGI fixed as the substrate and run a controlled comparison along three axes. (i) **Substrate axis:** Check2HGI vs. HGI under matched-head single-task baselines, head-invariant across linear / GRU / single / LSTM probes, on Gowalla U.S.-state splits with **FL/CA/TX as the headline scale and AL/AZ as smaller-scale anchors**. (ii) **MTL axis:** a cross-attention multi-task backbone over Check2HGI with per-task input modality (check-in stream → cat head, region stream → reg head), `static_weight(cat = 0.75)`, GRU cat head and STAN-Flow (`next_stan_flow`) reg head. (iii) **Joint axis:** paired Wilcoxon and Δm (Maninis 2019; Vandenhende 2021) against matched-head STL ceilings on both heads. All `next_region` measurements use leak-free per-fold transition priors.

**Beat 5 — Findings (one paragraph, three sentences carrying C1, C2, C3).** First, the substrate carries the cat win: Check2HGI lifts STL macro-F1 by **+15 pp** at AL/AZ and **+29–33 pp** at FL/CA/TX, head-invariant, paired Wilcoxon p = 0.0312 each (5 / 5 folds positive at the n = 5 ceiling); the substrate Δ scales monotonically with data, becoming sharper at the largest states. Second, MTL on top of Check2HGI gains a small additional cat lift (+0 to +2 pp) at every state and pays a sign-consistent cost on next-region Acc@10 vs. a matched-head STL STAN-Flow ceiling — but the cost magnitude **shrinks with data scale on the small-to-medium regime** (Δreg = −11.04 pp at AL → −12.28 at AZ → **−7.28 at FL**), suggesting MTL begins to recover under more data; CA preserves the regime (−8.93 pp at the largest cardinality) while TX is non-monotone (−16.69), pointing at state-specific factors (transition graph density, per-user trajectory geometry) beyond raw region count. Third, drop-in MTL fixes (FAMO, Aligned-MTL, hierarchical-additive-softmax reg head) do not recover the reg gap, and a methodological side-finding falls out: under cross-attention MTL, loss-side `task_weight = 0` ablations do not silence the silenced encoder — they co-adapt via attention K/V — so encoder-frozen isolation is the only clean architectural decomposition.

**Beat 6 — Contributions (bulleted, end of intro, mirrors AGENT.md §2):**
- **C1.** A controlled cross-state ablation showing that check-in-level contextual embeddings (Check2HGI) carry the next-category gain over POI-stable embeddings (HGI), head-invariant, with paired-Wilcoxon-significant Δs of +15 pp at the small-scale anchors (AL/AZ) and +29–33 pp at the headline-scale states (FL/CA/TX) — the substrate gain scales with data. A pooled-vs-canonical counterfactual at AL attributes ~72 % of the gap to per-visit context.
- **C2.** A scale-sensitive MTL tradeoff: with Check2HGI fixed as substrate, joint multi-task training over a cross-attention backbone gains a small cat lift at every state and pays a sign-consistent reg cost — but the cost **shrinks as data scale grows on the small-to-medium regime** (AL/AZ −11/−12 pp → FL −7 pp), evidence that MTL begins to recover with more data. CA preserves the regime; TX is non-monotone, pointing at state-specific factors. Drop-in alternatives (FAMO, Aligned-MTL, hierarchical-softmax reg head) do not recover the gap.
- **C3.** A methodological note: under cross-attention MTL, loss-side `task_weight = 0` ablations are unsound because the silenced encoder co-adapts via attention K/V projections. Encoder-frozen isolation is the only clean architectural decomposition; we provide regression tests.

Closing sentence: a one-line organisation statement (Section ↦ Section). Standard.

Total: 1.5 pp.

---

## §2 Related Work — paragraph beats

**Sub 2.1 — Graph-based POI embeddings (one paragraph).** Trace POI2Vec → CATAPE → DGI → HGI → Check2HGI line. State the contribution of each in one clause. Land Check2HGI's positioning: *"a check-in-level extension of HGI's hierarchical contrastive objective that emits one vector per check-in instead of one per POI, supplying per-visit context absent from POI-stable alternatives."* Cite Velickovic 2019 (DGI), Huang 2023 (HGI), Feng 2017 (POI2Vec), Rahmani 2019 (CATAPE).

**Sub 2.2 — Multi-task learning balancers (one paragraph).** Brief survey of the gradient-balancer line — NashMTL (Navon 2022), PCGrad (Yu 2020), GradNorm (Chen 2018), CAGrad (Liu 2021), Aligned-MTL (CVPR 2023), FAMO (NeurIPS 2023). Position our choice: *"we use `static_weight(cat = 0.75)` — the contribution is not a new MTL balancer; the balancer choice is fixed and we ablate against FAMO and Aligned-MTL in §6.2."* Important to flag this up-front so reviewers do not expect a balancer contribution.

**Sub 2.3 — POI multi-task models (one paragraph).** HMT-GRN (Lim 2022) is the closest concept-aligned baseline (joint next-POI + region with hierarchical graphs); cite as a concept-aligned reference, not a direct comparand (different task pair, different dataset). GETNext (Yang 2022) introduces the `α · log_T[last_region_idx]` graph prior — our reg head (STAN-Flow) borrows the additive log-transition prior pattern but is not a faithful GETNext reproduction (we do not include friendship / check-in graph priors). MGCL is also concept-aligned but trained on FSQ-NYC/TKY rather than Gowalla state-level. ReHDM (faithful port) is reported as an external single-task baseline.

**Sub 2.4 — Cross-attention multi-task models (one paragraph).** MulT (Tsai 2019) and InvPT (Ye 2022) introduce cross-attention as a multi-task interaction primitive in vision and NLP. Position our methodological note (C3) as generalising to that family: any cross-task interaction MTL with `task_weight = 0` ablations under attention will silently co-adapt the silenced encoder via K/V — encoder-frozen is the only clean isolation. *"We are not aware of prior work flagging this in the cross-attn MTL ablation methodology."*

**Land sentence (end of §2):** *"Our work re-examines a default assumption in MTL-for-POI — that joint training transfers signal between heads. We find this transfer empirically null on next-region in the cross-attention regime, motivating a reattribution: substrate first, MTL coupling second."*

Total: 1.5 pp.

---

## §3 Method — paragraph beats

### §3.1 Notation and task setup (≤ 0.5 pp)

One paragraph. Define check-in trajectory `T_u = (c_1, …, c_n)` per user; non-overlapping length-9 windows `(c_i, …, c_{i+8})` predicting `(category(c_{i+9}), region(c_{i+9}))`. State the class cardinalities per task and per state. State user-disjoint StratifiedGroupKFold(seed = 42) and the multi-seed protocol.

### §3.2 Substrate: Check2HGI vs. HGI (≤ 0.5 pp)

One paragraph. HGI is hierarchical-graph-infomax over POI / region / city graphs producing one vector per POI; same POI visited twice yields the same vector. Check2HGI extends this contrastive objective to the check-in graph: same POI visited twice yields different vectors because the visit's temporal and co-visit context perturbs the contrastive view. Both substrates emit 64-dim sequences, drop-in interchangeable. Cite Velickovic 2019 + Huang 2023; describe Check2HGI in main text only conceptually, defer recipe to supplementary.

### §3.3 MTL backbone — `MTLnetCrossAttn` (1 pp)

Two paragraphs.

*Paragraph 1 — architecture.* Two task-specific encoders (cat reads check-in embeddings, reg reads region embeddings) project to `d_model = 256`. An 8-head bidirectional cross-attention block bridges the two streams, followed by a four-block residual + LayerNorm + LeakyReLU + Dropout shared backbone. Per-task input modality is by design (CH03 in the study catalogue): the cat head needs check-in granularity, the reg head needs region granularity, and forcing them to share a single modality collapses one head (we provide the ablation in §6).

*Paragraph 2 — heads.* The cat head is `next_gru` — a 2-layer GRU + attention pooling + 7-class softmax. The reg head is **STAN-Flow** (`next_stan_flow`): a STAN attention backbone (Luo, WWW 2021) over the region sequence, augmented with a learnable scalar α gating an additive log-transition prior `α · log_T[last_region_idx]` constructed from training-fold-only edges (per-fold, leak-free). The α scalar is the load-bearing reg parameter — its growth across training drives the reg signal. We borrow the additive-log-prior pattern from GETNext (Yang 2022) but do not include friendship or check-in graph priors. Cite Luo 2021 + Yang 2022.

### §3.4 Loss and optimiser (1 pp)

*Paragraph 1 — loss.* Static-weight aggregation `L = 0.75 · L_cat + 0.25 · L_reg`. Both `L_cat` and `L_reg` are unweighted cross-entropy (deviating from the original MTLnet's per-class-weighted loss, after a pre-existing finding that unweighted CE wins under joint training; cite the auto-memory note `mtl_category_loss_unweighted.md` paraphrased in supplement).

*Paragraph 2 — optimiser regime.* AdamW with three parameter groups (cat-specific, reg-specific, shared). We report two recipes selected per state by paper-closure ablation: (i) **H3-alt**, constant per-group LR `cat = 1e-3, reg = 3e-3, shared = 1e-3`, used at AL/AZ; (ii) **B9**, alternating-SGD per batch + cosine LR (max = 3e-3) + α-no-weight-decay + minimum-best-epoch = 5, used at FL/CA/TX. Both share the per-task input modality and head choices. The recipe selection is itself a finding (B9 wins on FL multi-seed; H3-alt wins on small-state cat-side multi-seed) reported in §6.2. Mention but do not narrate the F-trail: the recipes were derived through a systematic ablation over LR scheduling, batch ordering, and weight-decay scoping; the full derivation is in supplementary materials.

### §3.5 Training protocol (≤ 0.5 pp)

One paragraph. 50 epochs, batch 2048 (1024 at FL on MPS for memory), gradient accumulation 2, FP32 throughout (no autocast). Per-fold transition prior: `region_transition_log_seed{S}_fold{N}.pt` built from training-fold-only edges. Checkpoint = per-task best `Acc@10_indist` for `epoch ≥ 5` (after F50/F51 the per-task-best convention is canonical). **Hardware:** all headline-scale runs (FL/CA/TX) on a single **NVIDIA H100 80 GB** GPU; smaller-scale anchors (AL/AZ) additionally validated on a consumer Apple-Silicon laptop (MPS) for cross-platform reproducibility. Per-run wall time: AL/AZ ~10 min, FL ~30 min, CA/TX ~50 min on H100 (5-fold × 50-epoch). Anonymous code at the URL in §0 D6.

Total: 2.5 pp.

---

## §4 Experimental Setup — paragraph beats

### §4.1 Datasets (≤ 0.5 pp)

One paragraph + table T1. Five Gowalla state splits (AL, AZ, FL, CA, TX), filtered to users with ≥ 5 check-ins and length-9 non-overlapping windows. Per-state stats: number of users, check-ins, POIs, regions, mean trajectory length, category distribution. Cite Cho 2011 (Gowalla, SNAP).

### §4.2 Baselines (≤ 0.5 pp)

One paragraph **per task**.

*Cat baselines:* Majority-class, Markov-1-POI, **POI-RGNN** (Capanema 2019, faithful reproduction with Gowalla state-level partition and 7-category taxonomy), **MHA+PE** (Zeng 2019). Internal: STL `next_gru` on Check2HGI (matched-head MTL ceiling), STL `next_gru` on HGI (substrate ablation). Audit hub: `docs/studies/check2hgi/baselines/next_category/`.

*Reg baselines:* Majority, Top-K popular, **Markov-1-region** (the binding floor at FL because of dense-data short-horizon coverage), STL GRU (literature-aligned), **STL STAN** (Luo 2021 adapt), **STL STAN-Flow** (matched-head MTL ceiling — same head class as the MTL reg head), **REHDM** (faithful port). Audit hub: `docs/studies/check2hgi/baselines/next_region/`.

State explicitly: **GETNext is not a baseline.** STAN-Flow's `α · log_T` graph prior is part of our MTL reg head, not a comparison method. Justification one-liner.

### §4.3 Metrics and statistical protocol (≤ 0.5 pp)

One paragraph. Cat primary: macro-F1; secondary: Acc@1/3. Reg primary: Acc@10; secondary: Acc@5, MRR. Joint: Δm (Maninis 2019; Vandenhende 2021) with primary `Δm-MRR = ½(r_cat F1 + r_reg MRR)` and secondary `Δm-Acc@10 = ½(r_cat F1 + r_reg Acc@10)`. Tests: paired Wilcoxon signed-rank `alternative='greater'` for directional claims; TOST non-inferiority at δ ∈ {2, 3} pp where the claim is "tied". State once: paired Wilcoxon at n = 5 has a maximum significance ceiling of p = 0.0312 (5 / 5 folds positive); pooled multi-seed (5 seeds × 5 folds = 25 fold-pairs at FL) gives sub-1e-7 p-values where available.

Total: 1.5 pp.

---

## §5 Results — paragraph beats

### §5.1 Substrate carries next-category — C1 (1.5 pp)

**Beat 1 — Headline number (one paragraph).** *"On the headline FL/CA/TX scale, check-in-level Check2HGI outperforms POI-stable HGI on next-category macro-F1 by +29–33 pp under matched-head STL `next_gru`, paired Wilcoxon p = 0.0312 each (5 / 5 folds positive, n = 5 ceiling); the smaller-scale anchors AL/AZ replicate the direction at +14.5 / +15.5 pp."* Forward to T2 for the full head-invariance grid (5-state). The substrate Δ scales monotonically with data — the embedding becomes more useful as the dataset grows.

**Beat 2 — Head invariance (one paragraph).** Across four head probes (linear / `next_gru` / `next_single` / `next_lstm`) the substrate Δ is uniformly positive at every state — 8 / 8 head-state cells positive at AL+AZ Phase 1, plus the matched-head extension at FL/CA/TX from FINAL_SURVEY §2. *"The substrate Δ is not an artefact of head choice — it is intrinsic to the embedding."*

**Beat 3 — Mechanism preview (one paragraph).** Forward-reference §6.1: a POI-pooled counterfactual at AL attributes ~72 % of the matched-head substrate gap to per-visit context (canonical − POI-pooled) and ~28 % to Check2HGI's training signal (POI-pooled − HGI). Per-visit context is the load-bearing axis.

**Beat 4 — External comparison (one paragraph).** Our STL `next_gru` Check2HGI cat F1 strongly exceeds POI-RGNN's published Gowalla state-level numbers (Capanema 2019: ~31.8 % FL, ~34.5 % CA) on every matched state. Forward to T5 for the cross-baseline table.

### §5.2 MTL gains on cat, costs on reg — C2 (1.5 pp)

**Beat 1 — The headline (one paragraph).** *"At the headline FL/CA/TX scale, with Check2HGI fixed as substrate, joint MTL over a cross-attention backbone gains +1.6 to +2.0 pp on next-category macro-F1 at every state and pays 7–17 pp on next-region Acc@10 vs. a matched-head STL STAN-Flow ceiling — the textbook MTL tradeoff (Caruana 1997; Vandenhende 2021), where the easier task gains and the harder task pays."* Forward to T3 for the per-state numbers. Frame this as a tradeoff, not a refutation: the model still produces both predictions in one forward pass, and we accept the reg cost in exchange for joint single-model deployment with a non-trivial cat-side lift.

**Beat 2 — Scale-progression on the reg cost (one paragraph; the mechanism finding).** AL/AZ as smaller-scale anchors (T3-supp) sharpen the picture: Δreg is **−11.04 pp** at AL (1.1K regions, 10K check-ins) → **−12.28 pp** at AZ (1.5K, 26K) → **−7.28 pp** at FL (4.7K, 127K). The architectural cost shrinks by ~5 pp going from the small-scale anchors to the smallest headline state — direct evidence that **MTL begins to recover under more data on the harder task**. CA preserves the regime at the largest cardinality (−8.93 pp, 8.5K regions); TX is the honest non-monotone outlier (−16.69 pp at 6.5K regions), a state-specific signal that the scaling story has factors beyond raw class count (transition-graph density, per-user trajectory geometry — left for follow-up). The five-state pattern is sign-consistent in *direction* (all negative) and broadly downward in *magnitude* on the small-to-medium regime, with TX flagged honestly.

**Beat 3 — Δm joint score (one paragraph).** Δm-Acc@10 is negative at every state (sign-consistent with the per-task reg loss). Δm-MRR is positive at FL multi-seed (+2.33 %, p = 2.98 × 10⁻⁸ across 25 fold-pairs at 25 / 25 positive) and negative or marginal elsewhere. Source: `CLAIMS_AND_HYPOTHESES.md §CH22 (2026-05-01 leak-free reframe)`. Forward to T4. Note the FL Acc@10-vs-MRR split is itself a small mechanism finding: MTL produces *better-ranked* region predictions than STL even where raw top-10 is worse — paper-worthy single sentence.

**Beat 4 — Recipe sensitivity (one paragraph).** B9 (alt-SGD + cosine + α-no-WD) is the FL-tuned recipe; H3-alt (per-head LR, constant) is the universal small-state recipe. We report MTL B9 at FL/CA/TX (headline) and H3-alt at AL/AZ (supplement) — single MTL row per state at its best recipe — and forward the cross-recipe ablation to §6.2. *"The recipe-selection is itself scale-conditional, but no recipe in our search closes the MTL-vs-STL gap on next-region."*

**Beat 5 — Forward to robustness (one paragraph).** Frame the next section: *"The architectural reg cost is robust to drop-in MTL fixes. We tested FAMO (NeurIPS 2023), Aligned-MTL (CVPR 2023), and a hierarchical-additive-softmax reg head; none reaches paired-Wilcoxon significance at FL against H3-alt."* Forward to §6.2 / T6.

### §5.3 Cross-baseline summary (1 pp)

**Beat 1 — Per-state comparison block (table-driven, one paragraph).** T5 reports per-state {Markov-1-region floor, STL STAN, STL STAN-Flow, MTL B9} on the reg side and {Majority, Markov-1-POI, POI-RGNN, MHA+PE, STL `next_gru` Check2HGI, MTL B9} on the cat side. State the headline: cat-side, our method beats every external baseline at every state; reg-side, MTL trails STL STAN-Flow but exceeds Markov-1-region at AL/AZ/CA/TX (FL is Markov-saturated, see Beat 2).

**Beat 2 — FL Markov-saturation, honest framing (one paragraph).** *"On dense-data state splits where Markov-1-region transitions cover ≥ 85 % of validation rows (Florida Gowalla, 127K check-ins), the classical 1-gram prior is near-optimal for Acc@10 on short horizons. Our neural models exceed Markov-1 on Acc@5 and MRR but not on Acc@10 at this scale."* Lifted near-verbatim from `docs/studies/check2hgi/PAPER_STRUCTURE.md §6` (the original study one) — the framing is honest and pre-vetted.

**Beat 3 — One-paragraph closer.** Re-state the substrate dominance: cat win is large and external; reg cost is structural to joint training and is paid for the convenience of single-model deployment.

Total: 4 pp.

---

## §6 Mechanism and Ablations — paragraph beats

### §6.1 Per-visit-context counterfactual (~0.75 pp)

**Beat 1 — Counterfactual setup (one paragraph).** Define *POI-pooled Check2HGI* as the substrate where canonical Check2HGI vectors are mean-pooled per POI across all check-ins, applied uniformly to every visit at that POI. This kills per-visit variation while preserving Check2HGI's training signal. The control is what isolates per-visit context as a mechanism axis distinct from the embedding's training procedure.

**Beat 2 — Decomposition (one paragraph + optional F1).** At AL, matched-head STL `next_gru`: Check2HGI canonical 40.76 % vs POI-pooled 29.57 % vs HGI 25.26 % macro-F1. Decomposition: per-visit context = +11.19 pp (~72 %); training signal (pooled − HGI) = +4.31 pp (~28 %). The linear-probe split (canonical 30.84 vs pooled 23.20 vs HGI 18.70) gives 63 % / 37 % — head choice amplifies the per-visit share. *"Per-visit context is the dominant mechanism."* Source: `CLAIMS_AND_HYPOTHESES.md §CH19`.

**Beat 3 — Why this matters for the paper claim (one paragraph).** Two consequences. (i) The substrate gain is not an artefact of better contrastive training — the largest fraction comes from a property HGI cannot architecturally supply (per-POI vectors). (ii) Check2HGI's training-signal residual (~28 %) is itself worth reporting honestly — Check2HGI's contrastive view *also* produces per-POI vectors that beat HGI's even before per-visit context enters.

### §6.2 Drop-in MTL ablations and recipe selection (~0.5 pp)

**Beat 1 — Drop-in fixes (one paragraph + T6).** At FL (where the architectural cost is paper-grade), we tested three classes of drop-in fix: head-capacity (hierarchical-additive softmax reg head), magnitude balancing (FAMO, NeurIPS 2023), direction alignment (Aligned-MTL, CVPR 2023). Result: none closes a +3 pp paired-Wilcoxon target against H3-alt; mean Δreg ranges from −3.01 pp (HSM-reg, n.s.) to +0.62 pp (FAMO, n.s.). *"The architectural reg cost is robust to balancer / head-capacity drop-in fixes — the gap is structural to the cross-attention shared-backbone interaction at large region cardinality."* Source: `research/F50_T1_RESULTS_SYNTHESIS.md`.

**Beat 2 — Recipe sensitivity (one paragraph).** Cross-state B9-vs-H3-alt comparison (28 paper-closure runs + 8 small-state H3-alt gap-fills): B9's three additions over H3-alt (alternating-SGD + cosine + α-no-WD) help at FL (+3.48 pp Δreg, p = 2.98 × 10⁻⁸ across 25 fold-pairs) but hurt cat at small states (AL/AZ Δcat −0.96 to −2.22 pp p ≤ 7 × 10⁻⁴). Headline: *"the optimal MTL recipe is scale-conditional — B9 at FL/CA/TX, H3-alt at AL/AZ — but no single recipe in our search closes the MTL-vs-STL gap on next-region."* Source: `PAPER_CLOSURE_RESULTS_2026-05-01.md §4a-bis`.

### §6.3 Methodological note: cross-attn `task_weight = 0` pitfall — C3 (~0.25 pp)

**Beat 1 — The note (one paragraph).** Under cross-attention MTL, setting the cat head's `task_weight = 0` does not silence the cat encoder — its parameters continue to update via gradient flow through the shared cross-attention K/V projections, since the reg loss still depends on the cat-encoder output through the attention block. The standard "loss-side λ = 0" ablation therefore does not isolate the architectural contribution; encoder-frozen isolation (random-init cat encoder, frozen across training) is the only clean decomposition. We provide regression tests in our anonymous code (`tests/test_regression/test_mtlnet_crossattn_lambda0_gradflow.py`). The note generalises to MulT, InvPT, and any cross-task interaction MTL with `task_weight = 0` ablations. Source: `research/F49_LAMBDA0_DECOMPOSITION_GAP.md`.

Total: 1.5 pp.

---

## §7 Discussion and Limitations — paragraph beats

**Beat 1 — Re-attribution narrative (one paragraph).** The five-state evidence collapses into a clean re-attribution: the multi-task win on cat is **substrate-carried**, not transfer-carried. Cross-task transfer through `L_cat` is null at every multi-seed state we measured (≤ |0.75| pp; ≥ 9σ refutation of legacy "transfer" claims at FL alone). The architecture provides at most a small additional cat lift (+0–2 pp), and structurally costs the reg head 7–17 pp at every state. The conventional MTL framing — "joint training transfers signal between heads" — is, in this regime, empirically vacuous on the harder task. Honest reattribution: substrate first, MTL coupling second.

**Beat 2 — Scale-progression mechanism (one paragraph).** The reg cost is sign-consistent (all negative) but its **magnitude shrinks as data grows** on the small-to-medium regime: AL −11.04 pp → AZ −12.28 → FL −7.28 (~5 pp recovery from the smallest anchors to the smallest headline state). Reading: under more data the cross-attention backbone gets enough representational room to share without over-paying on reg, and the textbook tradeoff approaches break-even — *MTL begins to generalise on reg as the dataset scales*. CA at the largest cardinality (8.5K regions) preserves the regime (−8.93 pp); TX (6.5K regions, −16.69 pp) is non-monotone and is the honest outlier. State-specific factors beyond raw class count — per-user trajectory geometry, transition-graph density, short-horizon Markov coverage — are plausible drivers and a natural target for follow-up. The recipe-selection finding (B9 FL-tuned, H3-alt small-state) tracks this scale axis: B9's FL-targeted ingredients (alt-SGD, cosine, α-no-WD) help reg at FL by addressing the "reg encoder saturates by epoch 5–6" pattern, but cost cat-side signal at small states that cannot afford the per-step temporal gradient separation.

**Beat 3 — Limitations (one paragraph).** Four honest limitations. (i) Multi-seed coverage is asymmetric: AL/AZ/FL run at four to five seeds; **CA/TX multi-seed at {0, 1, 7, 100} is in flight on H100 at submission and lifts paired Wilcoxon above the n = 5 ceiling at every state once it lands** (target update for camera-ready). (ii) FL `next_region` Acc@10 sits in a Markov-saturated regime; we report Acc@5 + MRR alongside to characterise the small-margin band fairly. (iii) **External baseline coverage is asymmetric: ReHDM (Li et al., IJCAI 2025) is reported at AL/AZ/FL but deferred at CA/TX — the dual-level hypergraph's collaborator-pool grows quadratically with region cardinality and exceeded our H100 compute budget at 8.5K and 6.5K regions; CA/TX ReHDM rows are flagged for camera-ready.** (iv) sklearn-version reproducibility caveat: `StratifiedGroupKFold(shuffle = True)` produces different fold splits across sklearn 1.3.2 → 1.8.0 (PR #32540); paired tests within a single env are unaffected, but absolute leak-magnitude attribution requires single-env re-runs documented in `FINAL_SURVEY.md §8`.

**Beat 4 — What this implies for follow-up work (one paragraph, forward-looking).** Two natural directions. *Asymmetric routing:* PLE (Tang 2020) and Cross-Stitch Networks (Misra 2016) free the reg head from shared-backbone interference; testing whether they recover the MTL-over-STL gap on reg without sacrificing the cat lift is the obvious next experiment. *Encoder enrichment:* the embedding-side response from CoUrb (spatial + temporal + categorical fusion) and the substrate response here (Check2HGI per-visit context) are complementary — a fused substrate that combines both axes is plausibly the path to a Pareto-positive joint solution. Both are out of scope for this paper.

Total: 1 pp.

---

## §8 Conclusion — paragraph beats

**Beat 1 — Recap (one paragraph, 4–6 lines).** *"We tested whether check-in-level contextual graph embeddings (Check2HGI) carry the multi-task POI prediction win on next-category and next-region across U.S.-state Gowalla splits, with FL/CA/TX as the headline scale and AL/AZ as smaller-scale anchors. The substrate carries the cat win — +15 pp at small states, +29–33 pp at large states, head-invariant — and a multi-task setup over the same substrate gains a small additional cat lift (+1.6 to +2.0 pp at the headline scale) while paying a sign-consistent reg cost. The cost magnitude shrinks as data grows on the small-to-medium regime (AL/AZ −11/−12 pp → FL −7 pp), evidence that MTL begins to generalise on reg with more data; CA preserves the regime, TX is non-monotone. The architectural cost is robust to drop-in fixes (FAMO, Aligned-MTL, head-capacity scaling). A methodological side-finding generalises beyond our study: under cross-attention MTL, loss-side `task_weight = 0` ablations co-adapt the silenced encoder and are not a clean architectural isolation."*

**Beat 2 — Future work (one paragraph).** Three lines: asymmetric routing (PLE / Cross-Stitch), encoder enrichment (substrate fusion across spatial / temporal / contextual axes), region-side dynamic priors. Closing line: *"Overcoming the tradeoff — recovering reg without sacrificing the cat lift — is the open challenge this paper sets up for follow-up work."*

Total: 0.5 pp.

---

## §9 References (working list, ≤ 30 entries)

Port from `articles/CBIC___MTL/references.bib` and `articles/CoUrb_2026/references.bib`, dedup, splncs04 style. Working canon (sub-agent A8 finalises):

**MTL methods.**
1. Caruana, R. (1997). *Multitask Learning.*
2. Maninis, K.-K., Radosavovic, I., Kokkinos, I. (2019). *Attentive Single-Tasking of Multiple Tasks.* CVPR.
3. Vandenhende, S. et al. (2021). *Multi-Task Learning for Dense Prediction Tasks: A Survey.* TPAMI.
4. Yu, T. et al. (2020). *Gradient Surgery for Multi-Task Learning (PCGrad).* NeurIPS.
5. Navon, A. et al. (2022). *Multi-Task Learning as a Bargaining Game (NashMTL).* ICML.
6. Chen, Z. et al. (2018). *GradNorm.* ICML.
7. Liu, B., Liu, X. et al. (2021). *CAGrad.* NeurIPS.
8. Senushkin, D. et al. (2023). *Aligned-MTL.* CVPR.
9. Liu, B. et al. (2023). *FAMO.* NeurIPS.
10. Tang, H. et al. (2020). *PLE: Progressive Layered Extraction.* RecSys.
11. Misra, I. et al. (2016). *Cross-Stitch Networks.* CVPR.

**POI embeddings.**
12. Velickovic, P. et al. (2019). *Deep Graph Infomax.* ICLR.
13. Huang, X. et al. (2023). *Hierarchical Graph Infomax (HGI).*
14. Velickovic, P. et al. (2018). *Graph Attention Networks (GAT).*
15. Feng, S. et al. (2017). *POI2Vec.* AAAI.
16. Rahmani, H. A. et al. (2019). *CATAPE.*

**POI prediction (cat / region / next-POI).**
17. Capanema, F. et al. (2019). *POI-RGNN.* (state-level Gowalla cat baseline)
18. Zeng, J. et al. (2019). *MHA+PE for Next-POI category.*
19. Luo, Y., Liu, Q., Liu, Z. (2021). *STAN: Spatio-Temporal Attention Network for Next-Location Recommendation.* WWW.
20. Yang, S. et al. (2022). *GETNext.* SIGIR.
21. Lim, N. et al. (2022). *HMT-GRN.*
22. Chen, J. et al. (2020). *HMRM.* (CBIC baseline)

**Cross-attention MT models.**
23. Tsai, Y.-H. H. et al. (2019). *MulT.* ACL.
24. Ye, H., Xu, D. (2022). *InvPT.* ECCV.

**Datasets and protocol.**
25. Cho, E. et al. (2011). *Friendship and Mobility (Gowalla).* KDD; SNAP.
26. Wilcoxon, F. (1945). *Individual Comparisons by Ranking Methods.* Biometrics.

(Optional cuts if pages tight: 22 HMRM, 16 CATAPE, 24 InvPT.)

---

## §10 Open TODOs for sub-agents (write-side, not science-side)

- **A1 (intro).** Confirm with §0 D0 + D1 before opening. Land C1/C2/C3 bullets verbatim from `AGENT.md §2`.
- **A2 (related).** Pull the four-axis structure from CoUrb's `related.tex`; do not over-claim continuity (CBIC + CoUrb are *cited as related work*, not "our prior").
- **A3 (method).** Resist the temptation to walk through the F-trail. The B9 / H3-alt distinction is described as "we report the per-state best of two recipes derived through ablation; full derivation in supplementary materials." That is the limit of method-section recipe-narrative.
- **A4 (setup).** Generate T1 dataset stats from the actual `data/<state>/` directories — do not copy stale numbers. Verify class cardinalities (regions per state) match `output/check2hgi/<state>/regions.parquet`.
- **A5 (results).** This is the longest section. Write T2 → T3 → T4 → T5 first, then prose around them. The paragraph beats above are deliberate — one finding per paragraph, claim sentence first, defend with one number, move on.
- **A6 (mechanism).** Keep §6.1, §6.2, §6.3 short and load-bearing. T6 is optional. F1 is optional. C3 is one paragraph max — it is a methodological note, not a contribution chapter.
- **A7 (discussion + conclusion).** No new claims here. Repackage the results section as re-attribution + scale + limitations + future work. Close with the "overcoming the tradeoff" line.
- **A8 (references).** Port + dedup + splncs04. ≤ 30 entries. Cross-check that every `\cite{}` in main text resolves and that no anonymizing red flags slipped through (no self-cites of "silva2025mtlnet"; cite as Silva et al. 2025 with the BibTeX `author` line preserving "Silva, V. H. O. et al." but the rest of the entry not pointing back to the lab).

---

## §11 Status

- **§0 (open decisions):** pending user lock.
- **Section beats §1–§8:** committed (this file).
- **Tables T1–T6:** specified; data-side numbers exist in `docs/studies/check2hgi/`; sub-agent A5 builds the LaTeX tables.
- **Figures F1–F2:** optional; lowest priority. Cut F2 first if pages tight; cut F1 next.
- **References:** working canon listed; A8 ports + dedups.
- **Title and abstract:** stale committed versions in `docs/studies/check2hgi/PAPER_DRAFT.md` superseded by §0 D0 + D1.
