# PAPER_STRUCTURE.md — Section-Level North Star

> **Read this alongside `AGENT.md` (operational rules + voice + statistics) and `PAPER_DRAFT.md` (paragraph-level beats).** This file commits the section/sub-section structure, the table-and-figure inventory, the claim-to-evidence map, and the sub-agent fan-out plan. It is *what* the paper says where; `PAPER_DRAFT.md` is *how* each part says it.

Conference: **BRACIS 2026 Main Track** (Springer LNAI, double-blind, 15-page hard cap including references and appendices). Template: `samplepaper.tex` in this folder.

> **Locked 2026-05-02 v4+v11 (post-Codex audit + v9/v10/v11 closures, per `PAPER_DRAFT.md §0` — FL §0.1 closed at n=20 in v11; all five states paper-grade on §0.1):** the headline is **substrate task-asymmetry** — per-visit context (Check2HGI) lifts cat by +14.5 to +29 pp at every state under matched-head STL (paired Wilcoxon p = 0.0312 each, head-invariant at the AL/AZ ablation scale and matched-head replicated at FL/CA/TX), while on reg HGI is nominally ahead by 1.6 to 3.1 pp (TOST tied at CA/TX). The **MTL tradeoff** (cat lift at four of five states: AZ +1.20 p < 1e-4, FL +1.40 p = 2e-06 (n=20), CA +1.68 p = 2e-06, TX +1.89 p = 2e-06; AL **small-significantly negative** at Δ = −0.78 pp, p = 0.036 n=20; reg loss 7 to 17 pp sign-consistent at every state) is the secondary finding. The reg-cost magnitude varies non-monotonically (TX outlier at −16.59 pp); the scale-progression is reported descriptively, not as an inferential claim. The earlier "Scale-Sensitive" title was demoted after Codex caught (a) stale cat numbers, (b) the +33 pp substrate/MTL-counterfactual conflation, (c) TX breaking scale monotonicity. **Headline tables T3/T4/T5 report FL/CA/TX; AL/AZ live in T3-supp; T2 (two-panel substrate cat+reg) reports all five states.**

---

## 1 · Page budget and section map

| § | Section | Pages | Carries which contribution? | Key tables/figures |
|---:|---|---:|---|---|
| 1 | Introduction | 1.5 | Frames C1, C2 + methodological note | — |
| 2 | Related Work | 1.5 | Positions our work | — |
| 3 | Method | 2.5 | Substrate + MTL backbone | (Fig. arch optional) |
| 4 | Experimental Setup | 1.5 | Datasets, protocol | T1 (datasets) |
| 5 | Results | 4 | C1, C2 | T2, T3, T4, T5 |
| 6 | Mechanism & Ablations | 1.5 | C1 mechanism, C2 robustness, methodological note | **F1 required**, T6 inline-prose |
| 7 | Discussion & Limitations | 1 | Re-attribution narrative | — |
| 8 | Conclusion | 0.5 | Future work | — |
| 9 | References | 0.75 | 25–30 refs, splncs04 | — |
|   | (Appendix, optional) | ≤ 0.5 | Leak-magnitude, raw 28-run grid | — |

Total: **14.25 + 0.75 ref ≈ 15** pages.

---

## 2 · Section-by-section contract

Each subsection below states (a) the headline sentence the section must land, (b) the sub-sections, (c) the claims it touches, (d) the artefacts it cites.

### §1 Introduction (1.5 pp)

Headline sentence to land: *"Across five U.S.-state Gowalla splits, check-in-level contextual graph embeddings provide a task-asymmetric substrate for joint POI prediction — they lift next-category by +14 to +29 pp at every state under matched-head single-task baselines, but on next-region they tie or marginally trail per-place embeddings. Multi-task learning over this substrate then gains a small additional cat lift at four of five states and pays a sign-consistent cost on the harder next-region task — the textbook MTL tradeoff."*

Sub-structure (one paragraph each):

1. **Domain framing.** POI category and POI region prediction are jointly useful for location-based services and urban mobility. They share a mobility substrate but place disjoint demands on the model.
2. **Prior trajectory framed in third person.** Two prior papers establish the bottleneck. (i) An MTL approach over POI-stable embeddings (Silva et al., CBIC 2025) reported only marginal gains over single-task baselines; the diagnosis pointed at representation mismatch. (ii) An embedding-decomposition follow-up (Paiva et al., CoUrb 2026) replaced the monolithic graph embedding with task-aware spatial / temporal / categorical encoders and recovered large category-side gains, suggesting the embedding choice — not the MTL recipe — was the load-bearing factor.
3. **Pivot.** The natural next question is whether a single principled substrate that supplies *per-visit context* (as opposed to per-POI stability) finally enables MTL to deliver bidirectional gains.
4. **What we do.** We adopt **Check2HGI**, a check-in-level contextual graph embedding from the same hierarchical-graph-infomax line as HGI but emitting one vector per check-in (per-visit context) rather than per POI. We measure Check2HGI vs. HGI under matched-head single-task baselines, then run a cross-attention MTL configuration over Check2HGI for joint next-category / next-region prediction across five Gowalla state splits with leak-free per-fold transition priors.
5. **Findings.** (i) The substrate is task-asymmetric: per-visit context lifts cat by +15.5 pp at AL, +14.5 at AZ, and +28.3 to +29.0 pp at FL/CA/TX under matched-head STL (paired Wilcoxon p = 0.0312 each — the n = 5 maximum, head-invariant at AL+AZ); on reg HGI is nominally ahead by 1.6 to 3.1 pp under matched-head STL (TOST tied at CA/TX). (ii) MTL on top of Check2HGI adds a small additional cat lift at four of five states (AZ +1.20 p < 1e-4 / FL +1.40 p = 2e-06 (n=20) / CA +1.68 p = 2e-06 / TX +1.89 p = 2e-06) while AL is small-significantly negative (Δ = −0.78 pp, p = 0.036 n = 20), and pays a 7 to 17 pp sign-consistent cost on next-region Acc@10 vs. a matched-head STAN-Flow STL ceiling at every state. (iii) Drop-in MTL fixes (FAMO, Aligned-MTL, head-capacity scaling) do not recover the reg gap.
6. **Contribution bullets (numbered C1/C2/C3 to mirror the rest of the paper).**

Citations: Silva et al. 2025 (CBIC), Paiva et al. 2026 (CoUrb), Caruana 1997 (MTL), Velickovic 2019 (DGI), Huang 2023 (HGI), POI-RGNN, HMT-GRN.

### §2 Related Work (1.5 pp)

Sub-sections — the four-axis structure CoUrb already uses translates well:

1. **Graph-based POI embeddings.** POI2Vec, CATAPE, DGI, HGI, Check2HGI line. Position Check2HGI as a check-in-level extension of HGI (per-visit, not per-POI).
2. **Multi-task learning balancers.** NashMTL, PCGrad, GradNorm, CAGrad, FAMO (NeurIPS 2023), Aligned-MTL (CVPR 2023). Position our choice as `static_weight(cat = 0.75)` — we do not propose a new balancer.
3. **POI MTL prior.** HMT-GRN (joint next-POI + region), GETNext (graph-prior trajectory flow), MGCL, ReHDM. Position our reg head (STAN-Flow = STAN-attention + α · log_T region trajectory prior, after the GETNext-hard pattern but not a faithful GETNext reproduction).
4. **Cross-attention multi-task models.** MulT, InvPT — relevant because the methodological note (C3 cross-attn λ = 0 pitfall) generalises to that family.

Land the sentence: *"Our work re-examines a default assumption in MTL-for-POI — that joint training transfers signal between heads to lift both. We find that, with the load-bearing substrate fixed, joint training delivers a small additional cat lift but pays a sign-consistent reg cost — the textbook multi-task tradeoff — motivating a reattribution: substrate first, MTL coupling second."* (Register matches the abstract's "textbook tradeoff" framing; do not sharpen to "empirically null/vacuous" outside §7 Beat 1, where the discussion can carry the stronger formulation.)

### §3 Method (2.5 pp)

#### 3.1 Notation and task setup (0.5 pp)

- Tasks: next-category (7 classes, macro-F1 primary), next-region (~1.1K AL / ~1.5K AZ / ~4.7K FL / ~8.5K CA / ~6.5K TX classes, Acc@10 primary, MRR + Acc@5 secondary). Sequence length 9 check-ins, predict the 10th's category and region. User-disjoint StratifiedGroupKFold, seed 42 anchor + multi-seed where available.

#### 3.2 Substrate: Check2HGI (per-visit) vs. HGI (per-POI) (0.5 pp)

- One paragraph: HGI is hierarchical-graph-infomax over POI / region / city graphs producing one vector per POI; same POI visited twice yields the same vector. Check2HGI extends HGI's contrastive objective to the check-in graph: same POI visited twice yields *different* vectors because the visit's temporal and co-visit context perturbs the embedding. The embedding's downstream use is identical (drop-in 64-dim sequence input) — this is the controlled comparison the paper exploits.
- Cite: Velickovic 2019 (DGI), Huang 2023 (HGI). Check2HGI itself is described in supplement; in main text it is *"a check-in-level extension of HGI's contrastive objective"*.

#### 3.3 MTL backbone (1 pp)

- **Cross-attention backbone (`MTLnetCrossAttn`).** Two task-specific encoders (cat-side reads check-in embedding sequence, reg-side reads region embedding sequence), 8-head 256-dim cross-attention block bridging the two, residual + LayerNorm + LeakyReLU + Dropout shared backbone. The per-task input modality is an **architectural choice** rather than a paper-claim — the underlying ablation ran at AL during development; the full multi-state ablation is supplementary.
- **Cat head** = `next_gru` (2-layer GRU + attention pooling + 7-class softmax). Justified empirically — replaces the original `NextHeadMTL` Transformer.
- **Reg head** = STAN-Flow (`next_stan_flow`) — STAN attention backbone (Luo, WWW 2021) + a learnable scalar α gating a precomputed log-transition prior `log_T[last_region_idx]` constructed from training-fold-only edges (after GETNext-hard pattern but not a faithful GETNext reproduction; we do **not** include friendship/check-in graph priors). The α scalar is the load-bearing parameter — its growth across training drives the reg signal.
- **Loss:** static-weight `λ_cat · L_cat + λ_reg · L_reg` with `λ_cat = 0.75, λ_reg = 0.25`.
- **Optimiser regime:** AdamW. We report two recipes — H3-alt (universal small-state) and B9 (FL-tuned). Headline tables use B9 at FL/CA/TX, H3-alt at AL/AZ; the recipe-selection result is itself a finding (scale-conditional, see §6.2). For the abstract / headline we report a single MTL row per state at its best recipe; the cross-recipe comparison goes in §6.
- **Leak-free protocol.** Per-fold log_T built from training-fold-only transitions, seed-tagged file. Cite the protocol explicitly because legacy POI-MTL papers in this line have leaky reg priors (the F44 leak documented in `research/F50_T4_C4_LEAK_DIAGNOSIS.md`).

#### 3.4 Training (0.5 pp)

- 50 epochs, batch 2048 (1024 at FL for memory), three-group AdamW with `cat_lr = 1e-3 / reg_lr = 3e-3 / shared_lr = 1e-3` (H3-alt) or alternating-SGD + cosine + α-no-WD (B9). Per-fold transition prior, gradient accumulation 2, no autocast on MPS, FP32 on H100/T4. Checkpoint = per-task best `Acc@10_indist` for `epoch ≥ 5`. Five-fold mean ± std reported, with paired Wilcoxon across folds.

### §4 Experimental Setup (1.5 pp)

#### 4.1 Datasets (0.5 pp)

- Five Gowalla state splits: AL, AZ, FL, CA, TX. Filtered to users with ≥ 5 check-ins; non-overlapping length-9 windows; user-disjoint folds. Table T1 reports per-state {users, check-ins, POIs, regions, mean trajectory length, category distribution}.
- Cite the public Gowalla source (Cho et al. 2011, SNAP).

#### 4.2 Baselines (0.5 pp)

- **Cat:** Majority-class, Markov-1-POI, **POI-RGNN** (Capanema et al. 2022, faithful port), **MHA+PE** (Zeng 2019). Internal: STL `next_gru` on Check2HGI (matched-head); STL `next_gru` on HGI (substrate ablation, CH16).
- **Reg:** Majority, Top-K popular, **Markov-1-region** (Floor; binds at FL on Acc@10), STL GRU, **STL STAN** (Luo, WWW 2021 adapt), **STL STAN-Flow (`next_stan_flow`)** (matched-head MTL reg head ceiling), **REHDM** (faithful port).
- Audit hub: `docs/studies/check2hgi/baselines/{next_category,next_region}/comparison.md`.

#### 4.3 Metrics and statistics (0.5 pp)

- Cat: macro-F1 primary, Acc@1/3 secondary.
- Reg: Acc@10 primary, MRR + Acc@5 secondary.
- Joint: **Δm** (Maninis 2019; Vandenhende 2022) — primary `Δm-MRR = ½(r_cat F1 + r_reg MRR)`, secondary `Δm-Acc@10 = ½(r_cat F1 + r_reg Acc@10)`.
- Tests: paired Wilcoxon signed-rank, `alternative='greater'` for directional claims; TOST non-inferiority at δ ∈ {2, 3} pp where the claim is "tied".
- State *once* the n = 5 single-seed ceiling: paired Wilcoxon p = 0.0312 = max significance for n = 5 (5/5 folds positive).

### §5 Results (4 pp)

#### 5.1 Substrate carries next-category — C1 (1.5 pp)

- T2 (substrate, two-panel cat + reg) — five-state matched-head STL ablation. Cat panel headline (`RESULTS_TABLE.md §0.3` / `FINAL_SURVEY.md §2`): `next_gru` STL Δ = **+15.5 / +14.5 / +29.0 / +28.8 / +28.3 pp** at AL/AZ/FL/CA/TX (NOT +33 — that was the MTL counterfactual; corrected post-Codex audit). Reg panel headline (`RESULTS_TABLE.md §0.3` / `FINAL_SURVEY.md §4`): HGI nominally ahead by 1.6 to 3.1 pp under matched-head STAN-Flow; TOST δ=2pp passes at CA/TX (tied), δ=3pp passes at FL. Wilcoxon p = 0.0312 each cat cell. *"The substrate carries the cat win at every state; on reg the substrate is at parity (CA/TX) or marginally HGI-favoring (AL/AZ/FL)."*
- One paragraph on **mechanism** (preview of §6.1): per-visit context accounts for ~72 % of the cat gap (POI-pooled counterfactual at AL); training signal accounts for ~28 %.
- One paragraph on **external comparison**: STL `next_gru` Check2HGI cat F1 exceeds POI-RGNN's published numbers and beats MHA+PE — table T5.

#### 5.2 MTL trades cat gains for reg cost — C2 (1.5 pp)

- T3 — five-state MTL vs STL ceilings on **both** tasks. **Source:** `RESULTS_TABLE.md §0.1` (v11, 2026-05-02) + `GAP_FILL_WILCOXON.json` + `ARCH_DELTA_WILCOXON.json`.
  - cat: Δ ∈ [−0.78, +1.89] pp; positive at four of five states (AZ +1.20 p < 1e-4 / FL +1.40 p = 2e-06 (n=20) / CA +1.68 p = 2e-06 / TX +1.89 p = 2e-06), **small-significantly negative at AL** (Δ = −0.78 pp, paired Wilcoxon p = 0.036 across n = 20 multi-seed fold-pairs; magnitude < 2 % relative).
  - reg Acc@10: Δ ∈ [−7.34, −16.59] pp (FL **−7.34** from paired Δ; CA −9.50; TX −16.59; AL −11.04; AZ −12.27), sign-consistent across all five states (always negative).
- One paragraph framing this **as the classic MTL tradeoff**: easier task gains, harder task pays. Cite Caruana 1997, Maninis 2019, Vandenhende 2022.
- T4 — Δm joint score — primary Δm-MRR is positive at FL multi-seed (+2.33 %, p = 2.98e-8 across 25 fold-pairs) and negative at AL/AZ/CA/TX (single-seed ceiling p ∈ {0.0625, 0.1250}). Δm-Acc@10 is negative at all 5 states. The metric ratifies the per-task picture; the FL MRR-vs-Acc@10 split is itself a small mechanism finding (better-ranked but not-better-top-K). **Source:** `RESULTS_TABLE.md §0.2` (v11, leak-free CH22 2026-05-01).
- One paragraph: the MTL gain on cat is **bounded**; the substrate, not the architecture, is the load-bearing factor for cat. Likewise the architectural cost on reg is **structural**: see §6 for ablations refusing to recover.

#### 5.3 Five-state cross-baseline summary (1 pp)

- T5 — per-state block: Markov-1-region floor / STL STAN / STL STAN-Flow (`next_stan_flow`) / MTL B9 on the reg side; Majority / Markov-1-POI / POI-RGNN / MHA+PE / STL `next_gru` Check2HGI / MTL B9 on the cat side. Numbers come from `baselines/*/results/<state>.json` + paper-closure tables.
- FL Markov-saturation note: at FL Markov-1-region binds at ~65 % Acc@10 because of dense-data short-horizon coverage. Honest framing — cite `PAPER_STRUCTURE.md §6` (the original study one) and reframe in §7.

### §6 Mechanism and Ablations (1.5 pp)

#### 6.1 Per-visit-context counterfactual (0.75 pp)

- Pooled-vs-canonical counterfactual at AL: POI-mean-pool the canonical Check2HGI vectors, train STL `next_gru` cat. Result: linear-probe ~63 %, matched-head ~72 % of the cat gap is per-visit context; ~28 % is Check2HGI's training signal. **F1 (required) bar chart** (post-Codex audit promotion — only visual mechanism anchor for the substrate task-asymmetry). Source: `CLAIMS_AND_HYPOTHESES.md §CH19`.

#### 6.2 Drop-in MTL ablations — does anything close the reg gap? (0.5 pp)

- T6 — **FAMO** (NeurIPS 2023), **Aligned-MTL** (CVPR 2023), **hierarchical-additive softmax reg head** at FL only — none reaches paired-Wilcoxon significance against H3-alt's reg ceiling. The architectural cost is robust to balancer / head-capacity drop-in fixes. Source: `research/F50_T1_RESULTS_SYNTHESIS.md`.
- One sentence on **recipe sensitivity**: H3-alt is universal at small states, B9 is FL-tuned, neither closes the MTL-vs-STL gap on next-region. The recipe-selection finding is itself reported (in T6 footnote or appendix table); it does not change the headline.

#### 6.3 Methodological note — the cross-attn `task_weight = 0` pitfall (0.25 pp)

- Short paragraph framed as a *methodological observation* (C3): under cross-attention, setting `task_weight = 0` does not silence the silenced encoder — it co-adapts via attention K/V. Encoder-frozen isolation is the only clean architectural decomposition. Applies to MulT, InvPT, and any cross-task interaction MTL with task-weight-zero ablations. Cite `research/F49_LAMBDA0_DECOMPOSITION_GAP.md`.

### §7 Discussion and Limitations (1 pp)

- One paragraph **re-attributing**: the multi-task picture is dominated by the substrate's task-asymmetric value, not by cross-task transfer. The architecture lifts cat marginally (4 of 5 states) and costs reg structurally at every state.
- One paragraph **on cost variation**: the reg cost magnitude varies across states (8–17 pp) but its sign does not. The pattern is broadly downward on the AL → AZ → FL trajectory, but TX (−16.59 pp at 6.5 K regions) breaks monotonicity — state-specific factors plausibly explain it. We report this descriptively, not as an inferential scaling claim.
- **Limitations.** (i) ReHDM (Li et al., IJCAI 2025) is reported at AL/AZ/FL only; CA/TX are deferred for compute reasons (dual-level hypergraph collaborator pool scales quadratically with region cardinality). (ii) FL `next_region` Acc@10 sits in a Markov-saturated regime; we report Acc@5 + MRR alongside to honestly characterise the small-margin band. (iii) sklearn-version reproducibility caveat for fold splits — paired tests within an env are unaffected but absolute leak-magnitude footnoting requires single-env re-runs (see `FINAL_SURVEY.md §8`).

### §8 Conclusion and Future Work (0.5 pp)

- 4–6 lines summarising the substrate + tradeoff finding.
- Future work — overcoming the tradeoff:
  - Asymmetric routing (PLE, Cross-Stitch) to free reg from shared-backbone interference.
  - Encoder enrichment (temporal + spatial + Check2HGI fusion in the spirit of CoUrb) — does the substrate gap close further when reg gets a richer input?
  - Region-side dynamic priors (per-fold log_T already gives a hint; richer priors might close the architectural cost).

### §9 References (0.75 pp)

25–30 BibTeX entries, splncs04 style. **`references.bib` exists in this folder** (28 verified entries, every entry carries a `% verified:` line per `AGENT.md §9`; ported from `articles/CBIC___MTL/references.bib` and `articles/CoUrb_2026/references.bib` with two corrections — POI-RGNN replaced from the wrong CBIC entry to Capanema 2022 PMC; HMRM authors corrected from CBIC's typos to Meng Chen et al.). Sub-agent A8 polishes for splncs04 compliance and ensures every `\cite{}` in main text resolves.

---

## 3 · Tables and figures inventory

### Required tables

| ID | Caption (working) | Source artefact | Section |
|---|---|---|---|
| **T1** | Dataset statistics for the headline three (FL/CA/TX) and the smaller-scale anchors (AL/AZ). One block per state with users, check-ins, POIs, regions, mean trajectory length. | Computed from `data/checkins/<state>.parquet` + `output/check2hgi/<state>/regions.parquet` | §4.1 |
| **T2** | Substrate ablation, two-panel: cat (Check2HGI vs HGI macro-F1, matched-head STL `next_gru`, four-head invariance grid for AL+AZ) and reg (matched-head STL `next_stan_flow` Acc@10 with TOST). All five states. Paired Wilcoxon p = 0.0312 each cat cell; substrate Δ on cat = +15.5 / +14.5 / +29.0 / +28.8 / +28.3 pp (AL/AZ/FL/CA/TX) — broadly scales with data but not strictly monotone (CA +28.81 < FL +29.02). | `RESULTS_TABLE.md §0.3` (v11); `FINAL_SURVEY.md §1, §2, §4` | §5.1 |
| **T3** | **Headline.** MTL vs STL ceilings on both tasks across **FL/CA/TX**. cat F1 (vs STL `next_gru`), reg Acc@10 + MRR (vs STL `next_stan_flow`). | **`RESULTS_TABLE.md §0.1` (v11, 2026-05-02)** | §5.2 |
| **T3-supp** | **Smaller-scale anchors supplement.** Same metrics for **AL/AZ**, included to show the cost-magnitude pattern across the five states. Δreg = −11.04 / −12.27 pp at AL/AZ vs FL's −7.34, CA's −9.50, TX's −16.59 pp. **Reported descriptively; not an inferential scale claim** — TX breaks monotonicity. | **`RESULTS_TABLE.md §0.1` (v11)** | §5.2 |
| **T4** | Δm joint score (cat F1 + reg MRR) and Δm-Acc@10, paired Wilcoxon. **FL multi-seed (n = 25 fold-pairs) is the strongest joint-score cell; AL/AZ/CA/TX remain at the single-seed ceiling on this axis.** | **`RESULTS_TABLE.md §0.2` (v11, leak-free CH22 2026-05-01)** | §5.2 |
| **T5** | External baselines: cat (POI-RGNN, MHA+PE) and reg (Markov-1-region, STL STAN, ReHDM) **per headline state (FL/CA/TX)**, with our STL ceiling and MTL row. ReHDM at AL/AZ/FL only (CA/TX deferred — limitation in §7). AL/AZ in T5-supp if pages allow. | **`RESULTS_TABLE.md §0.5–0.6` (v11)** + `baselines/next_*/results/<state>.json` | §5.3 |

### Required figure (post-Codex audit)

| ID | Caption (working) | Source artefact | Section |
|---|---|---|---|
| **F1** | Per-visit-context mechanism at AL: POI-pooled counterfactual. Bars: linear-probe Δ split into per-visit (~63 %) + training-signal (~37 %); matched-head split ~72 / 28. **F1 is the only visual mechanism anchor for the substrate task-asymmetry; cannot be cut.** | `CLAIMS_AND_HYPOTHESES.md §CH19` (whitelisted) | §6.1 |

### Optional / cut-first

| ID | Caption (working) | Source artefact | Cut order |
|---|---|---|:-:|
| **T6** | Drop-in MTL ablations at FL: FAMO, Aligned-MTL, HSM-reg-head. **Reduced to one inline sentence in §6.2** — see TABLES_FIGURES.md §2.3. | `research/F50_T1_RESULTS_SYNTHESIS.md` (already inline) | (cut to prose) |
| **F2** | Architectural cost vs region cardinality across the five states. Demoted from required to optional — TX breaks the scale pattern, so the figure no longer carries a title-anchoring claim. | `RESULTS_TABLE.md §0.1` reg col | cut 1st |
| **F-arch** | Architecture schematic — standard for BRACIS methods papers but not load-bearing. | new figure | cut 2nd |

**Cut order if pages tight: F2 → F-arch → T5-supp. F1 is non-negotiable.**

---

## 4 · Claim-to-evidence map

| Claim | Statement (1 line) | Evidence (study artefact + numbers) | Section |
|---|---|---|---|
| **C1-cat** | Check2HGI > HGI cat F1, matched-head STL, 5 states, paired Wilcoxon p = 0.0312 each. Head-invariant at AL/AZ. | **`RESULTS_TABLE.md §0.3`** (v10); cat STL `next_gru` Δ = +15.50 / +14.52 / +29.02 / +28.81 / +28.34 pp at AL/AZ/FL/CA/TX. | §5.1, T2 panel (a) |
| **C1-reg** | HGI ≥ Check2HGI under matched-head STL `next_stan_flow` reg by 1.6 to 3.1 pp; TOST δ=2pp passes at CA/TX (tied), δ=3pp passes at FL. | **`RESULTS_TABLE.md §0.3`** + `FINAL_SURVEY.md §4`. | §5.1, T2 panel (b) |
| **C1-mechanism** | At AL, ~72 % of the matched-head cat substrate gap is per-visit context (POI-pooled counterfactual). Single-state evidence — qualifier required everywhere. | `CLAIMS_AND_HYPOTHESES.md §CH19` (whitelisted). | §6.1, F1 |
| **C2-cat** | MTL vs STL on cat F1: positive at four of five states (AZ +1.20 p < 1e-4 n=20; FL +1.40 p = 2e-06 (n=20) ceiling; CA +1.68 p = 2e-06 n=20; TX +1.89 p = 2e-06 n=20); AL **small-significantly negative** (Δ = −0.78 pp, paired Wilcoxon p = 0.036 across n=20 multi-seed fold-pairs; magnitude < 2 % relative). | **`RESULTS_TABLE.md §0.1` (v11, 2026-05-02)** + `GAP_FILL_WILCOXON.json` + `ARCH_DELTA_WILCOXON.json`. | §5.2, T3 |
| **C2-reg** | MTL < STL reg Acc@10, sign-consistent (≤ 0) at 5 states; AL −11.04, AZ −12.27, FL **−7.34**, CA −9.50, TX −16.59 pp. | **`RESULTS_TABLE.md §0.1`** (v11, paired Δs). | §5.2, T3 |
| **C2-Δm** | Joint Δm-Acc@10 negative at 5 states; Δm-MRR positive at FL multi-seed only (+2.33 %, p = 2.98e-8 across 25 fold-pairs). | **`RESULTS_TABLE.md §0.2`** (v11, leak-free CH22). | §5.2, T4 |
| **C2-robustness** | At FL, drop-in MTL fixes (FAMO, Aligned-MTL, HSM) do not reach paired-Wilcoxon significance vs H3-alt. | `research/F50_T1_RESULTS_SYNTHESIS.md`. | §6.2 (inline prose) |
| **C2-recipe** | B9 is FL/CA/TX paper-grade; H3-alt is small-state universal (AL/AZ paper-grade). Recipe-selection is scale-conditional. | **`RESULTS_TABLE.md §0.4` (v11)** + `GAP_FILL_WILCOXON.json` (CA/TX n=20). | §6.2 footnote / appendix |
| **Methodological note** | Cross-attn `task_weight = 0` co-adapts via K/V; encoder-frozen isolation is the only clean decomposition. | `research/F49_LAMBDA0_DECOMPOSITION_GAP.md`. | §6.3 (side note, NOT contribution-prominent) |

Open: external baseline comparison (POI-RGNN published numbers) needs the cat-side state-level audit to be cross-checked against our reproductions in `baselines/next_category/results/<state>.json` — sub-agent §5.3 should not rely on the absolute POI-RGNN numbers without verification.

---

## 5 · Sub-agent fan-out plan

When the user fan-outs the writing across sub-agents, propose the following slicing (each sub-agent reads `AGENT.md` first, then the assigned section's beats from `PAPER_DRAFT.md`, then writes one `.tex` file).

| Sub-agent | File to write | Inputs (read) | Outputs |
|---|---|---|---|
| A1 — Intro | `sections/intro.tex` | `PAPER_DRAFT.md §1`, CBIC + CoUrb intros (style only), `BRACIS_GUIDE.md §10` | 1.5 pp prose + contribution bullets |
| A2 — Related Work | `sections/related.tex` | `PAPER_DRAFT.md §2`, CoUrb related-work section (style + four-axis structure) | 1.5 pp prose, 12–18 cites |
| A3 — Method | `sections/method.tex` | `PAPER_DRAFT.md §3`, `MTL_ARCHITECTURE_JOURNEY.md` (for context only — do **not** narrate F-numbers in main text) | 2.5 pp prose, 1 optional architecture figure |
| A4 — Experimental Setup | `sections/setup.tex` | `PAPER_DRAFT.md §4`, dataset stats from `data/` and `output/<engine>/<state>/` | 1.5 pp prose, T1 |
| A5 — Results | `sections/results.tex` | `PAPER_DRAFT.md §5`, **`docs/studies/check2hgi/results/RESULTS_TABLE.md §0` (v11) — sole canonical numerical source**, `FINAL_SURVEY.md` (substrate panel context only) | 4 pp prose + T2 + T3 + T4 + T5 |
| A6 — Mechanism & Ablations | `sections/mechanism.tex` | `PAPER_DRAFT.md §6`, `CLAIMS_AND_HYPOTHESES.md §CH19` (whitelisted), `research/F49_LAMBDA0_DECOMPOSITION_GAP.md` | 1.5 pp prose, **F1 required**, T6 inline prose |
| A7 — Discussion & Conclusion | `sections/discussion.tex` + `sections/conclusion.tex` | `PAPER_DRAFT.md §7-§8` | 1 + 0.5 pp |
| A8 — References | `references.bib` | Port + dedup from CBIC + CoUrb `.bib`, splncs04 style | ≤ 30 entries |

Each sub-agent's deliverable is a self-contained `.tex` file that compiles when included by `samplepaper.tex`. They **must not** edit `samplepaper.tex` directly — only the orchestrator (the user or a coordinator agent) updates the include skeleton.

---

## 6 · What this doc is NOT

- Not the operational guide. See `AGENT.md`.
- Not the paragraph-level scratch. See `PAPER_DRAFT.md`.
- Not the live results table. **The sole canonical numerical source for paper tables is `docs/studies/check2hgi/results/RESULTS_TABLE.md §0` (v11).** The earlier `PAPER_CLOSURE_RESULTS_2026-05-01.md` was moved to `docs/studies/check2hgi/archive/post_paper_closure_2026-05-01/` in the 2026-05-01 study cleanup and is background provenance only; numbers there have been superseded by later `RESULTS_TABLE` upgrades, including v8's cat-Δ Wilcoxon, v10's CA/TX §0.1 n=20 closure, and v11's FL §0.1 n=20 closure (the last headline asymmetry).
- Not the claim catalogue. See `docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md`.

This file commits *structure*; the cited files hold the *evidence*.
