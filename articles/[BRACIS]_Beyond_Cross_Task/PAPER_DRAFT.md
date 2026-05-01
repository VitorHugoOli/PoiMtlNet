# PAPER_DRAFT.md — Full-Story Scratch

> **Read order:** (1) `AGENT.md` for voice / statistics / anonymization, (2) `PAPER_STRUCTURE.md` for section-level structure, (3) **this file** for paragraph-level beats. This file is the working scratch from which sub-agents draft prose. **Lock §0 first** — those are the user-facing decisions that must precede sub-agent fan-out.
>
> **Provenance.** This draft is the BRACIS-2026 article-side reframe of `docs/studies/check2hgi/PAPER_DRAFT.md`. The earlier study-side draft is anchored to the F48-H3-alt "Per-Head LR" finding and its committed title/abstract are now stale. **The sole canonical numerical source for paper tables is `docs/studies/check2hgi/results/RESULTS_TABLE.md §0` (v7, 2026-05-01 PM, multi-seed STL ceiling).** `PAPER_CLOSURE_RESULTS_2026-05-01.md` and `CLAIMS_AND_HYPOTHESES.md §CH22 2026-05-01 reframe` are background provenance only. This file supersedes the study-side draft for the BRACIS submission.

---

## §0 Open decisions — lock these before sub-agents fan out

> **Status (post-Codex reframe 2026-05-01 PM, v3):** D0–D6 reflect the **substrate task-asymmetry** reframe. The earlier "scale-sensitive" emphasis was demoted after a Codex audit (see §0.1 audit log) caught two load-bearing issues: (a) the cat-Δ numbers in the article-side draft were stale relative to the v7 RESULTS_TABLE.md (multi-seed STL ceiling; AL flipped from −0.19 to −0.78); (b) the "+33 pp substrate" claim conflated STL substrate-only (+28-29 pp) with the MTL counterfactual (+33 pp). The reframed lead story is the substrate's task-asymmetric value (per-visit context lifts cat by +14 to +29 pp at every state, paired Wilcoxon p = 0.0312; on reg HGI is nominally ahead by 1.6-3.1 pp, TOST tied at CA/TX) — sharper, paper-grade significant at every state, and explained by CH19 (~72 % per-visit-context mechanism at AL).
>
> **Coverage state at submission (paper-side limitations, not workflow notes):**
> - **TX MTL multi-seed:** TX is seed = 42 single-seed at submission; multi-seed extension at {0, 1, 7, 100} is a camera-ready audit item. T3/T4 TX cells sit at the n = 5 paired-Wilcoxon ceiling (p_min = 0.0625 two-sided). The CA recipe-selection axis went multi-seed in v8 (paper-grade significant on both tasks), but the CA MTL-vs-STL §0.1 axis remains single-seed and is also a camera-ready audit item. Disclosed in §7.
> - **ReHDM scope:** reported at AL/AZ/FL only. CA/TX deferred — the dual-level hypergraph's collaborator pool scales quadratically with region cardinality and the per-state run exceeded our compute budget at 8.5 K (CA) and 6.5 K (TX) regions on a single H100. Disclosed in T5 footnote and §7.
> - **AL/AZ/FL cat-Δ Wilcoxon (v8 update, landed 2026-05-01):** AL p = **0.036** (n = 20 multi-seed; small-significantly negative, Δ = −0.78 pp); AZ p **< 1e-04** (n = 20; significantly positive, Δ = +1.20 pp); FL p = **0.0625** (n = 5 ceiling; sign-consistent positive, Δ = **+1.52 pp**). Wilcoxon JSON: `docs/studies/check2hgi/research/GAP_FILL_WILCOXON.json`. The "pending re-run" framing is now resolved.

### §0.1 Codex audit log (post-commit `7a60e1c`, accepted 2026-05-01 PM)

External critic flagged six issues; all accepted with edits:

1. **Stale cat-Δ numbers.** Article used PAPER_CLOSURE_RESULTS_2026-05-01 §4a (v6) values (AL −0.19, AZ +1.89, FL +1.61); canonical is `docs/studies/check2hgi/results/RESULTS_TABLE.md §0.1` (v7, 2026-05-01 PM, multi-seed STL): **AL −0.78, AZ +1.20, FL +1.43, CA +1.94, TX +2.02**. Math: STL `next_gru` was refreshed from multi-seed runs {0, 1, 7, 100}; MTL B9 cat unchanged. AL p-value pending re-Wilcoxon against the multi-seed STL ceiling (the v6 p = 0.76 was against single-seed STL).
2. **+33 pp substrate conflation.** Abstract said "29 to 33 pp at headline scale". Wrong: the **substrate-only matched-head STL** Δ is +28.3 to +29.0 pp (FINAL_SURVEY §2); the +33 pp belongs to the MTL counterfactual (FINAL_SURVEY §3, MTL Check2HGI − MTL HGI). These are different stories. Corrected to "+14 to +29 pp" everywhere.
3. **Scale-sensitive title overclaim.** TX (−16.69 pp on reg) breaks the "cost shrinks with data" pattern; the latest data does not support scale-sensitivity as a title-bearing claim. Demoted to descriptive secondary in §5.2 / §7.
4. **Substrate task-asymmetry is the cleanest story.** Cat: Check2HGI > HGI by +14.5 to +29 pp at every state, paired Wilcoxon p = 0.0312 each. Reg: HGI nominally ≥ Check2HGI by 1.6-3.1 pp, TOST δ=2pp passes at CA/TX. Promoted to lead.
5. **CH19 mechanism (~72 % per-visit) is a top-tier asset.** F1 (per-visit bar) promoted from "cut first" to required figure; F2 (scale scatter) demoted to optional.
6. **CH03 per-task-modality** is `partial` status (AL-dev only). Method §3.3 framing softened to "an architectural choice; full ablation deferred to supplement".
7. **C3 cross-attn λ = 0 note** demoted from contribution bullet to methodological note in §6.3 (still in the paper, not contribution-prominent).

| ID | Decision | Default (recommended) | Alternate | Status |
|---|---|---|---|---|
| **D0** | Title direction | **Substrate Carries, Architecture Pays: Check-In-Level Embeddings for Multi-Task POI Prediction** (10 words, Pattern D phenomenon-and-method) — promoted from alternate to default after Codex audit; carries the substrate task-asymmetry as the lead reading without committing to the noisy scale curve | *Beyond Cross-Task Transfer: A Task-Asymmetric Substrate for Multi-Task POI Category and Region Prediction* (14 words, Pattern A acronym+colon) — preserves "Beyond" hook with task-asymmetry framing; alternate if reviewers prefer the colon-shape | locked-default 2026-05-01 (v3) |
| **D1** | Abstract reframe | Rewrite the 130-word abstract around C1-substrate-asymmetry (per-visit context lifts cat by +14 to +29 pp at every state, p = 0.0312 each; on reg HGI is nominally ahead by 1.6-3.1 pp under matched-head STL — TOST tied at CA/TX) + C2 (classic MTL tradeoff: cat gains 0 to +2 pp; reg pays 8 to 17 pp, sign-consistent at every state) + C1-mechanism (~72 % per-visit at AL). Drop "scale-sensitive" framing; drop "+33 pp" — that conflated STL substrate with MTL counterfactual. C3 (cross-attn λ = 0 note) NOT in abstract — moved to §6.3 only | Keep prior abstract and footnote — NOT recommended | locked-default 2026-05-01 (v3) |
| **D2** | Headline states | **Three-state headline (FL/CA/TX) + AL/AZ as smaller-scale anchors.** Headline tables (T3, T4) report FL/CA/TX; AL/AZ supplement T3-supp (smaller-scale ablation evidence, NOT a "scale-curve mechanism"). T2 substrate ablation (two-panel, cat+reg) reports all five states — the substrate task-asymmetry is paper-grade significant at every one of the five states and is the strongest survival of the closure. | All five states in every headline table | locked-default 2026-05-01 |
| **D3** | Recipe story in main text | Report MTL B9 at FL/CA/TX, MTL H3-alt at AL/AZ — single MTL row per state at its best recipe — cross-recipe comparison + scale-conditional finding in §6.2 / appendix. Recipe-selection is robustness evidence, NOT headline | Pick a single recipe (B9) for all 5 states and footnote AL/AZ underperformance | locked-default 2026-05-01 |
| **D4** | Cite the F-trail in main text? | **No** — F-numbers + journey doc go in supplementary. Main text gives recipe + result, not derivation. | Yes — narrate the F-trail in §3 / §6 | locked-default 2026-05-01 |
| **D5** | Main result framing | **Substrate task-asymmetry is the headline (C1).** Per-visit context is the load-bearing substrate property for cat (5/5 states paper-grade); on reg the substrate is at parity (CA/TX) or marginally HGI-favoring (AL/AZ/FL). The MTL tradeoff (C2) is the honest cost of joint single-model deployment. CH19 (~72 % per-visit at AL) explains the mechanism. | "Scale-sensitive MTL tradeoff" — DEMOTED post-Codex audit. Reg cost is non-monotone (TX outlier); does not carry a title-bearing claim | locked-default 2026-05-01 (v3) |
| **D6** | Anonymous code link | Anonymous GitHub at `https://anonymous.4open.science/r/<TBD>/` — sub-agent A6 generates the snapshot | Anonymous Dropbox | pending |

If the user further changes a default, the orchestrator updates this file *and* `samplepaper.tex` (title + abstract) in the same commit.

---

## §1 Introduction — paragraph beats

**Beat 1 — Domain framing (one paragraph).** Open with POI prediction in LBSNs as the substrate. Two complementary tasks: (i) **next-category prediction** — given a length-9 check-in window, predict the semantic category (7 classes) of the next visit; (ii) **next-region prediction** — predict the next visit's region (≈ 1.1K to 8.5K classes depending on the state). Both tasks share a mobility substrate but place disjoint demands on the model: cat is a coarse 7-way decision driven by per-visit context (intent, time-of-day, co-location); reg is a fine-grained ranking problem driven by spatial-temporal trajectory regularities and short-horizon transitions. Voice cue: this is the same opener style as CoUrb's `intro.tex` paragraph 1 — domain framing, not claim-first.

**Beat 2 — The arc, framed in third person (one paragraph; double-blind safe).** Two recent works on this task pair establish the bottleneck. Silva et al. (CBIC 2025) proposed a hard-parameter-sharing MTL framework on POI-stable graph embeddings (DGI) and reported only marginal gains over single-task baselines — most differences fell within standard deviations. The diagnosis pointed at *representation mismatch*: a single shared encoder learning a "compromise" representation that was specialised enough for neither head. Paiva et al. (CoUrb 2026) tested an embedding-side response by decomposing the monolithic graph embedding into independent spatial (SIREN / Sphere2Vec-M), temporal (Time2Vec), and categorical (HGI) sub-encoders, recovering substantial cat-side gains across three states. Together, these two works frame the operative hypothesis: the embedding choice — not the MTL recipe — is the load-bearing factor.

**Beat 3 — The pivot (one paragraph).** Both prior works treat the embedding as a *what* axis (which features to encode: graph topology in CBIC; spatial + temporal + categorical sub-encoders in CoUrb). We test an orthogonal *granularity* axis instead: the embedding's emission rate. POI-stable embeddings (HGI, DGI) emit one vector per place — same POI visited twice yields the same vector. Check-in-level contextual embeddings emit one vector per visit, capturing the user's intent shift across two visits to the same café (lunchtime vs. evening). The natural question is whether per-visit context — a different bottleneck axis from per-modality decomposition — finally enables multi-task learning to deliver bidirectional gains.

**Beat 4 — What we do (one paragraph).** We adopt **Check2HGI**, a check-in-level extension of HGI's contrastive objective: same POI visited twice yields different embeddings because the visit's temporal and co-visit context perturbs the contrastive view. We hold Check2HGI fixed as the substrate and run a controlled comparison along three axes. (i) **Substrate axis:** Check2HGI vs. HGI under matched-head single-task baselines, head-invariant across linear / GRU / single / LSTM probes, on Gowalla U.S.-state splits with **FL/CA/TX as the headline scale and AL/AZ as smaller-scale anchors**. (ii) **MTL axis:** a cross-attention multi-task backbone over Check2HGI with per-task input modality (check-in stream → cat head, region stream → reg head), `static_weight(cat = 0.75)`, GRU cat head and STAN-Flow (`next_stan_flow`) reg head. (iii) **Joint axis:** paired Wilcoxon and Δm (Maninis 2019; Vandenhende 2021) against matched-head STL ceilings on both heads. All `next_region` measurements use leak-free per-fold transition priors.

**Beat 5 — Findings (one paragraph, three sentences carrying C1, C1-mechanism, C2).** First, **the substrate is task-asymmetric**: per-visit context (Check2HGI) lifts single-task next-category macro-F1 by **+15.5 pp at AL, +14.5 at AZ, +29.0 at FL, +28.8 at CA, +28.3 at TX** under matched-head STL — paired Wilcoxon p = 0.0312 each (the maximum at n = 5 folds), all 5/5 folds positive, head-invariant across linear / GRU / single / LSTM probes at AL+AZ — while on the harder next-region task POI-stable HGI is **nominally ahead by 1.6 to 3.1 pp** under matched-head STL STAN-Flow, with TOST non-inferiority at δ = 2 pp passing at CA/TX (statistically tied) and δ = 3 pp passing at FL. Second, a pooled-vs-canonical counterfactual at Alabama attributes **~72 % of the cat substrate gap to per-visit context** and ~28 % to Check2HGI's training signal (single-state mechanism evidence) — explaining the asymmetry: per-visit variance is what cat needs; per-POI pooling smooths it away for reg. Third, with Check2HGI fixed as substrate, joint MTL over a cross-attention backbone adds a small additional cat lift at four of five states (AZ +1.20 pp p < 1e-4; FL +1.52 pp p = 0.0625 n = 5 ceiling; CA +1.94 pp; TX +2.02 pp) while AL is small-significantly negative (Δ = −0.78 pp, p = 0.036 n = 20; magnitude < 2 % relative). On next-region MTL pays a sign-consistent 8 to 17 pp cost on Acc@10 at every state — the textbook MTL tradeoff (cat mostly gains, reg pays); drop-in MTL fixes (FAMO, Aligned-MTL, hierarchical-additive-softmax reg head) do not recover the reg gap, and a methodological side-finding falls out (cross-attention `task_weight = 0` ablations co-adapt via K/V; encoder-frozen isolation is required).

**Beat 6 — Contributions (bulleted, end of intro, two contributions plus a methodological note):**
- **C1.** A controlled five-state ablation showing that check-in-level contextual embeddings (Check2HGI) provide a **task-asymmetric substrate** for joint POI category and region prediction. On next-category, per-visit context lifts matched-head STL macro-F1 by +14.5 to +29.0 pp at every state, paired Wilcoxon p = 0.0312, head-invariant. On next-region, per-place embeddings (HGI) are nominally ahead by 1.6 to 3.1 pp under matched-head STAN-Flow (TOST non-inferiority δ = 2 pp passes at CA/TX). A pooled-vs-canonical counterfactual at AL attributes ~72 % of the cat substrate gap to per-visit context.
- **C2.** A direct measurement of the **classic multi-task tradeoff** under the load-bearing substrate. With Check2HGI fixed, joint MTL over a cross-attention backbone gains on next-category at four of the five states (AZ +1.20 pp, p < 1e-04 across n = 20 multi-seed fold-pairs; FL +1.52 pp, p = 0.0625 at the n = 5 ceiling; CA +1.94 pp; TX +2.02 pp) while AL is small-significantly negative (Δ = −0.78 pp, paired Wilcoxon p = 0.036 across n = 20 multi-seed fold-pairs; magnitude small at < 2 % relative on a 41 % F1 scale). On the harder next-region task it pays a sign-consistent 8 to 17 pp cost on Acc@10 vs. a matched-head STAN-Flow STL ceiling at every one of the five states. The reg-cost magnitude varies across states (FL smallest at −7.99; TX largest at −16.69 — non-monotone outlier) — we report the broadly-downward pattern descriptively, not as an inferential scaling claim. Drop-in alternatives (FAMO, Aligned-MTL, hierarchical-softmax reg head) do not recover the reg gap.
- **Methodological note.** Under cross-attention MTL, loss-side `task_weight = 0` ablations are unsound — the silenced encoder co-adapts via attention K/V projections; encoder-frozen isolation is the only clean architectural decomposition. We provide regression tests. (This note is a §6.3 sub-section, not a top-level contribution; it generalises beyond our study to cross-task attention MTL methodology.)

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

*Paragraph 1 — architecture.* Two task-specific encoders (cat reads check-in embeddings, reg reads region embeddings) project to `d_model = 256`. An 8-head bidirectional cross-attention block bridges the two streams, followed by a four-block residual + LayerNorm + LeakyReLU + Dropout shared backbone. The per-task input modality (check-in stream into cat head; region stream into reg head) is an **architectural choice** — single-modality variants collapsed one head in our development experiments at AL; we report the ablation in supplementary materials and treat the choice as a design decision rather than a paper claim.

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

### §5.1 Substrate is task-asymmetric — C1 (1.5 pp)

**Beat 1 — Headline (one paragraph; the substrate task-asymmetry).** *"Under matched-head single-task baselines, the substrate is sharply task-asymmetric. On next-category, check-in-level Check2HGI outperforms POI-stable HGI by **+15.5 pp at AL, +14.5 at AZ, +29.0 at FL, +28.8 at CA, +28.3 at TX** (paired Wilcoxon p = 0.0312 each — the maximum significance at n = 5 folds — all 5/5 folds positive; head-invariant at AL+AZ across linear / GRU / single / LSTM probes, matched-head replicated at FL/CA/TX). On next-region, the same matched-head ablation under STL STAN-Flow shows POI-stable HGI nominally ahead by 1.6 to 3.1 pp; TOST non-inferiority at δ = 2 pp passes at CA/TX (statistically tied) and δ = 3 pp passes at FL (within 3 pp of HGI), but the cat-side picture is the load-bearing finding."* Forward to T2 (two-panel) for the full grid.

**Beat 2 — Head invariance and external comparison (one paragraph).** Head invariance: across four head probes (linear / `next_gru` / `next_single` / `next_lstm`) the cat substrate Δ is uniformly positive at AL+AZ — 8 / 8 head-state cells positive at AL+AZ Phase 1, plus the matched-head replication at FL/CA/TX (`RESULTS_TABLE.md §0.3`). External: our STL `next_gru` Check2HGI cat F1 (60–67 % at FL/CA/TX) exceeds our faithful POI-RGNN reproduction by ≥ 28 pp at every matched state (faithful POI-RGNN: FL 34.49, CA 31.78, TX 33.03 per `RESULTS_TABLE.md §0.6`). The POI-RGNN published evaluation reported a 31.8–34.5 pp range across states under non-user-disjoint folds; our reproduction at user-disjoint folds is the comparison we report, and the gap remains a conservative lower bound vs. the published configuration. Forward to T5.

**Beat 3 — Mechanism preview (one paragraph).** *"Why is the substrate task-asymmetric? Per-visit variance is what cat needs (intent shift across two visits to the same place); per-POI pooling smooths it away for reg (the prediction target is region cardinality, not the per-visit identity). A POI-pooled counterfactual at AL attributes ~72 % of the matched-head cat substrate gap to per-visit context (canonical − POI-pooled) and ~28 % to Check2HGI's training signal (POI-pooled − HGI). Per-visit context is the load-bearing axis on the cat side; on the reg side it is irrelevant."* Forward to §6.1 + F1.

### §5.2 MTL trades cat gains for reg cost — C2 (1.5 pp)

**Beat 1 — The classic MTL tradeoff (one paragraph; the headline).** *"With Check2HGI fixed as substrate, joint MTL over a cross-attention backbone lifts next-category macro-F1 at four of the five states (AZ +1.20 pp p < 1e-4; FL +1.52 pp p = 0.0625 at n = 5 ceiling; CA +1.94 pp; TX +2.02 pp) while AL is small-significantly negative (Δ = −0.78 pp, paired Wilcoxon p = 0.036 across n = 20 fold-pairs; magnitude small at < 2 % relative on a 41 % F1 scale). On next-region MTL pays 8 to 17 pp on Acc@10 vs. a matched-head STAN-Flow STL ceiling at every one of the five states — the textbook MTL tradeoff (Caruana 1997; Vandenhende 2022), easier task mostly gains, harder task pays."* n = 20 paired Δs at AL/AZ (4 seeds × 5 folds, multi-seed pooled paired Wilcoxon at p = 1.9 × 10⁻⁶ on reg); n = 5 single-seed at FL/CA/TX with p = 0.0625 at the n = 5 ceiling. TX multi-seed extension is a camera-ready audit item; CA recipe-selection axis went multi-seed in v8 (paper-grade significant on both tasks; see §6.2). Forward to T3. Frame as a tradeoff, not a refutation: the model produces both predictions in one forward pass, and we accept the reg cost in exchange for joint single-model deployment with the cat-side lift at most states.

**Beat 2 — Magnitude varies non-monotonically (one paragraph; honest scale comment).** AL/AZ as smaller-scale anchors (T3-supp) extend the picture: Δreg is **−11.04** at AL (1.1 K regions, 10 K check-ins) → **−12.27** at AZ (1.5 K, 26 K) → **−7.99** at FL (4.7 K, 127 K) → **−8.92** at CA (8.5 K, 230 K) → **−16.69** at TX (6.5 K, 187 K). The pattern is broadly downward on the AL → AZ → FL trajectory and CA preserves it, but **TX breaks monotonicity** — state-specific factors (transition-graph density, per-user trajectory geometry, region-cardinality vs check-in-density coupling) plausibly explain TX's outlier behaviour but are not directly testable at our state count. *"We report this pattern descriptively; the cost varies non-monotonically across states and we leave the scale observation as honest secondary description rather than an inferential claim — quantifying it would require within-state density ablation, deferred to follow-up."*

**Beat 3 — Δm joint score (one paragraph).** Δm-Acc@10 is negative at every state (sign-consistent with the per-task reg loss). Δm-MRR is positive at FL multi-seed (+2.33 %, p = 2.98 × 10⁻⁸ across 25 fold-pairs at 25 / 25 positive) and negative or marginal elsewhere. Source: `CLAIMS_AND_HYPOTHESES.md §CH22 (2026-05-01 leak-free reframe)`. Forward to T4. Note the FL Acc@10-vs-MRR split is itself a small mechanism finding: MTL produces *better-ranked* region predictions than STL even where raw top-K is worse — paper-worthy single sentence.

**Beat 4 — Recipe sensitivity (one paragraph).** B9 (alt-SGD + cosine + α-no-WD) is the FL-scale recipe; H3-alt (per-head LR, constant) is the small-state recipe. We report MTL B9 at FL/CA/TX (headline) and H3-alt at AL/AZ (supplement) — single MTL row per state at its best recipe — and forward the cross-recipe ablation to §6.2. The recipe-selection axis is now paper-grade significant at three states: B9 > H3-alt at FL (Δ_reg +3.48 pp, p = 3 × 10⁻⁸, n = 25) and at CA (Δ_reg +4.18 pp, Δ_cat +0.51 pp, both p < 1e-04, n = 20 multi-seed v8); H3-alt > B9 on cat at AL/AZ (Δ_cat −2.22 / −0.96 pp, both p < 1e-3, n = 20). TX recipe direction matches FL/CA but stays at the n = 5 single-seed ceiling. *"The recipe-selection is itself scale-conditional, but no recipe in our search closes the MTL-vs-STL gap on next-region."*

**Beat 5 — Forward to robustness (one paragraph).** Frame the next section: *"The architectural reg cost is robust to drop-in MTL fixes. We tested FAMO (NeurIPS 2023), Aligned-MTL (CVPR 2023), and a hierarchical-additive-softmax reg head; none reaches paired-Wilcoxon significance at FL against H3-alt."* Forward to §6.2.

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

**Beat 2 — Recipe sensitivity (one paragraph).** Cross-state B9-vs-H3-alt comparison (28 paper-closure runs + 8 small-state H3-alt gap-fills): B9's three additions over H3-alt (alternating-SGD + cosine + α-no-WD) help at FL (+3.48 pp Δreg, p = 2.98 × 10⁻⁸ across 25 fold-pairs) but hurt cat at small states (AL/AZ Δcat −0.96 to −2.22 pp p ≤ 7 × 10⁻⁴). Headline: *"the optimal MTL recipe is scale-conditional — B9 at FL/CA/TX, H3-alt at AL/AZ — but no single recipe in our search closes the MTL-vs-STL gap on next-region."* Source: `RESULTS_TABLE.md §0.4` (v7).

### §6.3 Methodological note: cross-attn `task_weight = 0` pitfall — C3 (~0.25 pp)

**Beat 1 — The note (one paragraph).** Under cross-attention MTL, setting the cat head's `task_weight = 0` does not silence the cat encoder — its parameters continue to update via gradient flow through the shared cross-attention K/V projections, since the reg loss still depends on the cat-encoder output through the attention block. The standard "loss-side λ = 0" ablation therefore does not isolate the architectural contribution; encoder-frozen isolation (random-init cat encoder, frozen across training) is the only clean decomposition. We provide regression tests in our anonymous code (`tests/test_regression/test_mtlnet_crossattn_lambda0_gradflow.py`). The note generalises to MulT, InvPT, and any cross-task interaction MTL with `task_weight = 0` ablations. Source: `research/F49_LAMBDA0_DECOMPOSITION_GAP.md`.

Total: 1.5 pp.

---

## §7 Discussion and Limitations — paragraph beats

**Beat 1 — Substrate task-asymmetry, not cross-task transfer (one paragraph).** The five-state evidence collapses into a clean reading: the multi-task picture is dominated by the **substrate's task-asymmetric value**, not by cross-task transfer through joint training. Per-visit context (Check2HGI) is the load-bearing substrate property for next-category prediction (paper-grade significant at every state, +14.5 to +29 pp under matched-head STL); on next-region the same substrate ties or marginally trails per-place embeddings (HGI nominally ahead by 1.6 to 3.1 pp; TOST tied at CA/TX), because pooling to region cardinality smooths the per-visit variance away. The MTL architecture, on top of the substrate, adds a small cat lift at four of five states (AZ/FL/CA/TX +1.20 to +2.02 pp) but is small-significantly negative at AL (Δ = −0.78 pp, p = 0.036, n = 20) and pays a structural reg cost of 8 to 17 pp at every state — the textbook tradeoff. The conventional MTL framing — *"joint training transfers signal between heads to lift both"* — is in this regime empirically vacuous on the harder task. Honest reattribution: **substrate first (asymmetric), MTL coupling second (tradeoff-paying).**

**Beat 2 — Cost magnitude varies; the scale story is descriptive, not inferential (one paragraph).** The reg cost magnitude varies across states (−7.99 pp at FL → −16.69 pp at TX) but its sign does not. The pattern is broadly downward on the AL → AZ → FL trajectory (−11.04 → −12.27 → −7.99) and CA preserves the regime (−8.92, 8.5 K regions); **TX (−16.69 pp at 6.5 K regions) breaks monotonicity**, suggesting state-specific factors beyond raw class count — per-user trajectory geometry, transition-graph density, short-horizon Markov coverage are plausible drivers but not directly testable at our state count. We report this pattern descriptively rather than as an inferential claim — quantifying scale dependence formally would require within-state density ablation, deferred to follow-up. The recipe-selection finding (B9 FL-tuned, H3-alt small-state) tracks the same axis at the optimiser level: B9's FL-targeted ingredients (alt-SGD, cosine, α-no-WD) help reg at FL by addressing the "reg encoder saturates by epoch 5–6" pattern, but cost cat-side signal at small states that cannot afford the per-step temporal gradient separation.

**Beat 3 — Limitations (one paragraph).** Five honest limitations. (i) Multi-seed coverage is asymmetric: AL/AZ/FL run at four to five seeds; the CA recipe-selection axis is multi-seed (v8); **TX is seed = 42 single-seed at submission**, with multi-seed extension a camera-ready audit item. The CA MTL-vs-STL §0.1 axis is also still single-seed pending camera-ready. AL/AZ/FL cat-Δ Wilcoxon p-values against the v8 multi-seed STL ceiling have landed (AL p = 0.036 small-significantly negative; AZ p < 1e-04; FL p = 0.0625 at the n = 5 ceiling); see `GAP_FILL_WILCOXON.json`. (ii) FL `next_region` Acc@10 sits in a Markov-saturated regime; we report Acc@5 + MRR alongside to characterise the small-margin band fairly. (iii) **External baseline coverage is asymmetric: ReHDM (Li et al., IJCAI 2025) is reported at AL/AZ/FL but deferred at CA/TX — the dual-level hypergraph's collaborator-pool grows quadratically with region cardinality and exceeded our compute budget at 8.5K and 6.5K regions on a single H100; CA/TX ReHDM rows are a camera-ready audit item.** (iv) **POI-RGNN reproduction caveat:** our faithful POI-RGNN reproduction uses user-disjoint folds; the published numbers used non-user-disjoint folds, which tends to inflate absolute scores. The +28 pp Check2HGI−POI-RGNN gap we report is a conservative lower bound against the published configuration. (v) sklearn-version reproducibility caveat: `StratifiedGroupKFold(shuffle = True)` produces different fold splits across sklearn 1.3.2 → 1.8.0 (PR #32540); paired tests within a single env are unaffected, but absolute leak-magnitude attribution requires single-env re-runs documented in `FINAL_SURVEY.md §8`.

**Beat 4 — What this implies for follow-up work (one paragraph, forward-looking).** Two natural directions. *Asymmetric routing:* PLE (Tang 2020) and Cross-Stitch Networks (Misra 2016) free the reg head from shared-backbone interference; testing whether they recover the MTL-over-STL gap on reg without sacrificing the cat lift is the obvious next experiment. *Encoder enrichment:* the embedding-side response from CoUrb (spatial + temporal + categorical fusion) and the substrate response here (Check2HGI per-visit context) are complementary — a fused substrate that combines both axes is plausibly the path to a Pareto-positive joint solution. Both are out of scope for this paper.

Total: 1 pp.

---

## §8 Conclusion — paragraph beats

**Beat 1 — Recap (one paragraph, 4–6 lines).** *"We tested whether check-in-level contextual graph embeddings (Check2HGI) carry the multi-task POI prediction win on next-category and next-region across U.S.-state Gowalla splits. The substrate is **task-asymmetric**: per-visit context lifts single-task next-category macro-F1 by +14.5 to +29 pp at every one of five states (paired Wilcoxon p = 0.0312 each — the n = 5 maximum), head-invariant; on next-region, the same substrate ties or marginally trails per-place embeddings under matched-head single-task ceilings (HGI nominally ahead by 1.6 to 3.1 pp; TOST tied at CA/TX). A pooled-vs-canonical counterfactual at Alabama attributes ~72 % of the cat substrate gap to per-visit context. With Check2HGI fixed, joint multi-task learning over a cross-attention backbone adds a small cat lift at four of five states (AZ +1.20 / FL +1.52 / CA +1.94 / TX +2.02 pp) while AL is small-significantly negative (Δ = −0.78 pp, p = 0.036, n = 20), and pays a sign-consistent 8 to 17 pp cost on reg Acc@10 at every state — the textbook MTL tradeoff. The reg cost is robust to drop-in fixes (FAMO, Aligned-MTL, head-capacity scaling); a methodological side-finding generalises beyond our study (cross-attention `task_weight = 0` ablations co-adapt via K/V; encoder-frozen isolation is required)."*

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
