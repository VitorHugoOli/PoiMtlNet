# RELATED_WORK_TRIAGE.md — evaluation of externally shared refs vs the paper (2026-07-06)

> **What this is.** A citation triage of (a) the four PDFs shared at `~/Desktop/mestrado/artigos/refs/` and
> (b) the 24-entry `new_references.bib` in the same folder, judged against the near-final MobiWac draft
> (`src/main.tex`, 10-page budget FULL [superseded 2026-07-19: 8-page EDAS budget, no fee], 38 bib keys / 31 rendered). Method: one deep-read agent per PDF
> (full text + web venue verification), each verdict then adversarially re-checked against the text by an
> independent verifier; 4 cluster audits over the bib; plus a web gap scan (12+ searches, full 2024+
> citation sweeps of CTLE / HMT-GRN / CSLSL on Semantic Scholar) hunting for closer competitors than the
> ones already defused. Every quote and venue below was verified against the primary source.

## 1 · Bottom line

**No must-cite was found anywhere — the paper's novelty claims hold.** Nothing in the four PDFs, the 24
bib entries, or the open web predicts next-category and next-region as **co-equal end targets** on
check-in data. DRRGNN, KGTB, and the CSLSL cascade (all already cited and correctly distinguished in
§2.2/§2.3) remain the nearest neighbors.

| Item | Verdict | One-line reason |
|---|---|---|
| POI-rec survey (TKDE 2025) | **optional** | Field map of exact-place rec, the task we drop; silent on all our axes (that silence is *ammo*, §5) |
| CoMaPOI (SIGIR 2025) | **optional** | LLM multi-agent **next-place**; no category/region targets, no MTL, no learned representation |
| ROS (ACL 2026) | **optional** | LLM generative **next-place**; region/category are internal SID tokens + RL reward shaping, never end targets |
| FisherGT / HAMURE (CIKM'26 sub) | **no** | Anonymous, under review, no public preprint → uncitable; also static region-rep (no prediction, no check-ins) |
| `new_references.bib` (24 entries) | **0 cite / 4 maybe / 20 skip** | Mostly next-place recommenders in families already covered by run baselines (ReHDM) or existing defusals |
| Gap scan (beyond both sets) | **2 should + 5 optional** | MobiWac'22 handover cite (venue-local, intro) + STCCR (CTLE successor, §2.1 defusal) — see §4 |

The two genuinely actionable items came from the **gap scan**, not from the shared refs.

## 2 · The four PDFs, in detail

### 2.1 "A Survey on POI Recommendation: Models, Architectures, and Security" — Zhang et al., **IEEE TKDE 2025** (vol. 37, no. 6, DOI 10.1109/TKDE.2025.3551292; the PDF is arXiv 2410.02191v2) — OPTIONAL

Broad survey of exact-place POI recommendation: models (latent factor → neural → SSL → generative/LLM),
architectures (centralized/decentralized/federated), security (poisoning, privacy). ~85–90-work mapping
table. **Zero overlap with our axes**: its sub-task taxonomy is entirely place-centric (next-category /
next-region never appear as end targets), DGI/HGI/CTLE are absent, and MTL appears only via TLR-M
(queue-time auxiliary). It threatens nothing and covers nothing we lack that a MobiWac reviewer would
expect.

- **If cited** (cheapest possible LLM-era modernization, one clause in §2.2's opening sentence):
  *"…recurrent and attention models such as ST-RNN, DeepMove, Flashback, STAN, and GETNext; a recent
  survey covers this line through the large-language-model era \cite{zhang2025poisurvey}."*
- **Adversarial-check correction:** the "our coverage stops at 2022" rationale is overstated — §2.2/§2.3
  already carry 2025 cites (sun2025kgtb, wang2025hamtl, li2025rehdm); only the recurrent/attention
  exemplar list stops at 2022. This weakens the case for adding it. Default: leave out; keep for the
  response letter (§5).

### 2.2 "CoMaPOI: A Collaborative Multi-Agent Framework for Next POI Prediction" — Zhong et al., **SIGIR 2025** (DOI 10.1145/3726302.3729930, pp. 1768–1778) — OPTIONAL

Three LoRA-fine-tuned LLM agents (Profiler / Forecaster / Predictor) that verbalize trajectories and
constrain the candidate POI set (250 → 25) before predicting the **exact next place**. Category is only
an input attribute; no region; multi-agent decomposition is not multi-task learning; no learned
check-in representation (off-the-shelf sentence embeddings for RAG). Different paradigm, same
deliberately-dropped target. Note: its "CA" dataset is the **Gowalla California/Nevada slice** (Yuan et
al. 2013), so raw-source overlap with our CA exists, but preprocessing and task differ entirely.

- **No citation job**: defuses nothing, no reviewer can call it a missing baseline. Citing it would
  *invite* "then compare against LLM baselines". Its inference stack is **three H800 80 GB GPUs** for a
  3-LLM pipeline — the systems-venue answer if that invitation ever arrives (§5).
- Also quietly validating: a SIGIR 2025 full paper still evaluates on a single fixed split, no folds/seeds/
  significance tests — our Wilcoxon/TOST protocol spends page budget where that field doesn't.

### 2.3 "Reasoning Over Space (ROS)" — Lv et al., **ACL 2026 main** (Long Papers pp. 7322–7336; arXiv 2601.04562v2; XJTU + Amap/Alibaba) — OPTIONAL, adjacent

Generative LLM recommender: each POI becomes a compositional discrete token (S2 hex-cell geospatial
prefix + RQ-VAE semantic tokens from category text), a 3-stage mobility chain-of-thought, GRPO RL with
hierarchically weighted rewards (coarse-region match earns partial credit). Target: **exact next POI**
("correct iff p_{n+1} exactly matches"). Region and category live *inside* the token vocabulary and the
reward, never as evaluated outputs. This is precisely the "coarse signals in service of next-place"
pattern our §2.2 already defuses with KGTB — ROS **reinforces** the "region as co-equal end target is
underexplored" claim rather than threatening it.

- **If cited** (only if we want a second, LLM-era defusal next to KGTB in §2.2): *"The same pattern
  extends to language-model recommenders: a recent generative system discretizes coarse-to-fine locality
  into hierarchical spatial tokens and rewards coarse-region correctness during training, but only in
  service of its exact next-place target \cite{lv2026ros}."* Default: leave out — the sun2025kgtb clause
  already occupies that slot.
- **Best takes in the whole batch** came from this paper (§5: near-miss distance metric, coarse-level
  partial-credit accuracy).

### 2.4 "Fisher Graph Transformers + HAMURE" — **anonymous CIKM 2026 submission, under review** — NO (cannot cite)

Graph-transformer attention with a Fisher information-geometric (anisotropic, elliptical) metric;
HAMURE = urban **region representation** model evaluated on land-use clustering, crime prediction, and
population-density estimation over NYC / Chicago / Singapore (census tracts: 2,230 / 793 / 316), with
**HGI as a baseline**. No check-ins, no trajectories, no next-anything, no multi-task prediction.
Verified: no author list, placeholder DOI/ISBN, and **no arXiv or other public version exists** → a
citation is impossible regardless of relevance, and relevance is low anyway (static region-level
socioeconomic probes vs our sequential per-visit prediction).

- **Observation for the authors:** the PDF's running header reads **"Santos et al."** on five pages —
  this looks like it comes from Germano's (or an adjacent) group. If so, worth knowing that in its
  tables **HGI beats the newer region embedders (RegionDCL, CityFM, HAFusion) in most cells** and even
  beats HAMURE on Chicago crime — third-party corroboration that building Part 1 on the HGI lineage was
  the right substrate bet (usable in a response letter, not in prose).
- Its census-tract partitioning is also a precedent for our region-unit choice if a reviewer questions it.

## 3 · The 24-entry `new_references.bib` audit

**Verdict: nothing to import.** 0 cite / 4 conditional maybes / 20 skips. The file is also **not safe to
merge as-is**: three key collisions (`Lai_2024` ×2, `Wang_2023` ×2, `Liu_2023` ×2) would silently drop
entries under BibTeX.

| Cluster | Entries | Verdict | Why |
|---|---|---|---|
| Hypergraph next-POI | Yan_2023 (STHGCN), Lai_2023 (MSTHN), Lai_2024 (DCHL), Lai_2024 (ASTHL), An_2024 (MvStHgL), Yangyang_2022 | 1 maybe, 5 skip | Entire family = hypergraph **next-place**; already covered in the strongest way possible — ReHDM (li2025rehdm), a region-aware member of this family, is an actually-run baseline. Three of six are the same first-author team. Yangyang_2022 is a low-credibility venue and would *weaken* an all-verified bib. |
| KG / context next-POI | Zhang_2024, Chen_2025 (STKG+prompt), Liu_2023 (Mandari), Wang_2023 (CCDSA), Wang_2023 (zone-enhanced ToP), Sun_2024 (privacy) | 6 skip | KG-flavored next-place; the KG defusal is already done by a strictly closer neighbor (KGTB predicts category+region, albeit instrumentally). The two verified threats dissolved: CCDSA uses category only as a preference signal; ToP uses zones only as auxiliary embeddings, never predicted outputs. Privacy is out of scope. |
| LLM / trajectory | Feng_2024, Beneduce_2025, Solatorio_2023 (GeoFormer), Cheng_2025 (POI-Enhancer), Liu_2022 (CSTRM) | 2 maybe, 3 skip | Adds at most ONE conditional job: if the authors decide the paper needs a one-sentence LLM acknowledgment, use **Beneduce_2025** (§2.2, LLM next-location) and/or **Cheng_2025** (§2.1, LLM semantic enrichment of *place-level* embeddings — orthogonal axis, needs POI text Gowalla lacks). Default: none — a MobiWac networking reviewer does not expect LLM-recommender coverage. |
| Urban region rep + non-LBSN MTL | Huang_2022, Luo_2023, Luo_2022, Wang_2025 (CaLLiPer), Liu_2023 (ships), Wang_2022 (AIS), Meng_2023 (lane-change) | 1 maybe, 6 skip | The urban-rep entries sit on the "*which* features a place vector encodes" axis that §2.1 explicitly closes with DGI→HGI; none is per-visit. CaLLiPer (CEUS 2025) is the only one with stature — still no defusal job. The ship/car MTL entries would **dilute** §2.3, whose argument is specific to the category/region pair on check-in data. |

**Standout worth knowing even if uncited:** Yan_2023 = **STHGCN** (SIGIR'23, Ant Group), the seminal
hypergraph next-POI model and ancestor of ReHDM's family. If a RecSys-literate reviewer asks "why not
STHGCN?", the answer is: ReHDM (IJCAI 2025) is the newer region-aware descendant of that family and is
already run as a baseline. Optionally make the lineage explicit with one clause when introducing ReHDM
in §5.4.

## 4 · Gap scan — what is actually missing (and what is not)

The scan searched for (1) any 2024–2026 work predicting next-category + next-region jointly as end
targets, (2) post-CTLE check-in-level representation work, (3) venue-local MobiWac literature, (4)
2025–2026 multi-task next-POI successors, (5) census-tract prediction from mobility traces. Result: **no
competitor on the exact task pair** (full 2024+ citation sweeps: CTLE 69 citing papers, HMT-GRN 69,
CSLSL 22 — nothing closer than what §2.2 already defuses). Aggregate OD-flow work on tracts (e.g. Nature
Comms 2025) is not per-user next-visit; no cite owed.

### 4.1 SHOULD add (both web-verified)

**(a) Venue-local anticipatory-services cite — intro ¶1–2.** The bib currently contains **zero MobiWac
papers** although the intro's whole motivation is anticipatory service adaptation. MobiWac 2022 has
exactly that paper from the networking side:

> Vielhaus, Busch, Geuer, Palaios, Rischke, Külzer, Latzko, Fitzek, *"Handover Predictions as an Enabler
> for Anticipatory Service Adaptations in Next-Generation Cellular Networks"*, MobiWac '22, pp. 19–27,
> DOI 10.1145/3551660.3560913.

One sentence in the intro grounds the mobility-management bridge in the venue's own lineage and defuses
"what does this have to do with mobility management?" at near-zero cost. PC members likely know it.

```bibtex
@inproceedings{vielhaus2022handover,
  author    = {Vielhaus, Christian L. and Busch, Johannes V. S. and Geuer, Philipp and
               Palaios, Alexandros and Rischke, Justus and K{\"u}lzer, Daniel F. and
               Latzko, Vincent and Fitzek, Frank H. P.},
  title     = {Handover Predictions as an Enabler for Anticipatory Service Adaptations
               in Next-Generation Cellular Networks},
  booktitle = {Proceedings of the 20th ACM International Symposium on Mobility
               Management and Wireless Access (MobiWac '22)},
  year      = {2022},
  pages     = {19--27},
  doi       = {10.1145/3551660.3560913}
}
```

**(b) STCCR — one-clause defusal in §2.1.** §2.1 claims "CTLE is the closest example" and cites nothing
in that family after 2021. The most likely attack from a reviewer who knows CTLE is its authors' own
successor: **STCCR** (Gong, Wan, Guo, Li, Lin, Zheng, Wang, Zhou, Lin — Wan/Guo/Y.Lin/Yf.Lin overlap
with the CTLE author list), *"Spatial-Temporal Cross-View Contrastive Pre-training for Check-in Sequence
Representation Learning"*, **IEEE TKDE 2024**, DOI 10.1109/TKDE.2024.3434565 (AAAI-23 predecessor
exists). It pre-trains a representation of a whole check-in **sequence** (spatial-topic + temporal-
intention contrastive views), not a per-visit vector inside a hierarchical graph — so it does not
displace the CTLE contrast; acknowledging it shows the 2024 state of the line was checked.

```bibtex
@article{gong2024stccr,
  author  = {Gong, Letian and Wan, Huaiyu and Guo, Shengnan and Li, Xiucheng and
             Lin, Yan and Zheng, Erwen and Wang, Tianyi and Zhou, Zeyu and Lin, Youfang},
  title   = {Spatial-Temporal Cross-View Contrastive Pre-training for Check-in
             Sequence Representation Learning},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2024},
  doi     = {10.1109/TKDE.2024.3434565}
}
```

Per the folder convention (bib entries carry primary-source verification quotes in comments), both
should get a quote-comment at add time, like the 2026-07-06 defusal entries did.

### 4.2 OPTIONAL (add only if a slot opens; none changes a verdict)

- **ROTAN (KDD 2024)** or **REPLAY (IEEE TMC 2024, Flashback authors)** — freshness cite for §2.2's
  recurrent/attention exemplar list (ends 2022). Optics only.
- **Hong, Martin, Raubal (Transp. Res. C 2023)** — transformer jointly predicting next location + travel
  mode on GNSS data; the best-known "location + semantic second target" instance outside LBSN, for §2.3.
- **Geo-Tokenizer (ECML/PKDD 2023)** — contextual location embeddings with hierarchical multi-scale
  grids; sits between CTLE and Check2HGI conceptually.
- **"Into the Unknown" (SIGSPATIAL 2025)** — inductive spatial-semantic location embeddings for
  predicting mobility beyond visited places; the inductive answer to exactly the transductivity caveat
  §5.2 admits. Natural discussion-section cite if space appears.
- **TraXion (arXiv 2605.06906, preprint)** — 2026 visit-tuple pre-training; newest per-visit-adjacent
  work, but a preprint; watch it for the thesis.
- **MobTCast (NeurIPS 2021)** — checked and rejected: category/geography as auxiliary tasks for
  next-place, functionally identical to the already-cited SGRec/MCMG/HMT-GRN defusals.

## 5 · Takes worth stealing (independent of any citation)

**Rebuttal kit (response letter, not prose):**
1. *"Why no LLM baselines?"* — CoMaPOI's own inference stack is three H800 80 GB GPUs for a 3-LLM
   pipeline per prediction vs our single small forward pass; and the TKDE 2025 survey itself rates
   generative/LLM recommenders low on inference efficiency and training scalability ("longer inference
   times", "prohibitively expensive"). A latency-sensitive mobility-management setting is the one place
   this answer is airtight.
2. *"Is the task pair really underexplored?"* — the field's own comprehensive 2025 survey taxonomy
   contains no category- or region-as-end-target track, no per-visit representation line (DGI/HGI/CTLE
   absent), and no MTL-for-mobility discussion beyond a queue-time auxiliary. External corroboration,
   stated as such.
3. *"Why census tracts?"* — the CIKM'26 region-representation submission (and the urban-computing line
   generally) partitions by census tracts too (2,230 NYC / 793 Chicago); standard, defensible unit.
4. *"Why build on HGI?"* — in that same submission's tables HGI still beats RegionDCL, CityFM, and
   HAFusion in most cells; the substrate bet is corroborated by 2026 third-party numbers.

**Analysis ideas (cheap, high leverage for camera-ready or thesis):**
5. **Geographic near-miss metric** (from ROS): for wrong next-region predictions, report the distance
   (or tract adjacency) between predicted and true region as a CDF or P50/P90. If misses are near, a
   near-miss tract is still actionable for resource placement — this is the single best idea in the
   batch for the MobiWac audience, and it converts region errors into a mobility-management argument.
6. **Coarse-level partial credit** (from ROS's hierarchical reward): report next-region accuracy at a
   coarser aggregation (county / S2 parent) alongside exact-tract Acc@10 — quantifies "how wrong is
   wrong".
7. **Candidate-space framing** (from CoMaPOI): even the newest LLM systems dedicate an entire agent to
   shrinking the exact-POI candidate set before predicting; independently supports dropping next-place.
   Their hit-rate lower-bound analysis also formalizes our future-work gesture (next-region as a learned
   candidate constrainer for a downstream next-place stage) — thesis material.
8. **Attention-geography diagnostic** (from FisherGT §5.5): Moran's I over attention weights, validated
   with paired Wilcoxon — adaptable to our cross-attention trunk (is region-task attention spatially
   structured? does it sharpen with region count?). Interpretability appendix material.
9. **Ablation pattern**: FisherGT holds the model fixed and swaps only the attention geometry — same
   isolate-one-factor logic as our CTLE feature-concat control; good precedent if the control is challenged.

## 6 · Recommended actions, ranked

1. **Add the MobiWac'22 handover cite** (intro, 1 sentence + bib). Venue-local, defuses the
   "why here?" question. ~2 lines of budget.
2. **Add the STCCR clause** (§2.1, 1 clause + bib). Closes the only real coverage seam a
   knowledgeable reviewer would poke (post-2021 CTLE lineage).
3. **Decide once on the LLM question**: default NO LLM cites; if the authors want one acknowledgment,
   the pair is zhang2025poisurvey (§2.2 clause) or Beneduce_2025 — never both, never more.
4. **Do not import `new_references.bib`** (0 must-cites; 3 duplicate-key defects would break the bib).
5. **Keep the rebuttal kit** (§5 items 1–4) with the review materials.
6. **Consider the near-miss distance CDF** (§5 item 5) as the one analysis add worth GPU-free effort
   (it re-reads existing prediction JSONs) if any figure space ever opens.

## 7 · The survey-absence question: "if the field's survey never mentions our tasks, are we wrong — or novel?"

Asked by the author 2026-07-06; answered with a dedicated survey re-read + web evidence sweep. **Verdict:
scoping artifact, not a red flag — and the absence doubles as (weak) positive evidence of the gap.**

**Why the absence proves nothing against us.** The TKDE 2025 survey never taxonomizes *tasks* at all: its
axes are models / architectures / security, its §II fixes exactly three task definitions (POI rec, next-POI
rec, spatial-item rec), and it has **no evaluation-metrics section whatsoever** (zero occurrences of NDCG,
HR@k, recall, F1 in 2,319 lines). A document with no slot for *any* task variation cannot be read as
adjudicating whether a task is worth doing. In its universe, category and region appear only as side
information (HKGNN), latent clusters (TPM "topic-regions"), representation enrichments (GeoMF), and
deployment scopes (DCPR) — features that serve exact-POI ranking, never outputs.

**Where our tasks DO live (verified sources).** Both halves of our pair are established end tasks in the
*neighboring* community:
- **Luca et al., ACM Computing Surveys 55(1) 2021** (the mobility community's canonical survey) defines
  next-location prediction over "a spatial tessellation G of the geographic space (generally an i×j grid)"
  — region/tile-level next-location IS that community's standard formulation; ours instantiates it on
  census tracts.
- **HuMob Challenge, ACM SIGSPATIAL 2023/2024** (YJMob100K, Nature Scientific Data 2024): the flagship
  benchmark's target is a 500 m grid cell — region-level next-location as an active, benchmarked end task.
- **Category/activity lineage unbroken 2013→present:** Ye et al. SDM 2013 (next activity category, then
  place given category), MCARNN IJCAI 2018 (joint activity+location, both evaluated), iMTL IJCAI 2020.
- **Networking lineage:** next-cell and handover prediction (native at MobiWac; e.g. Vielhaus et al.
  MobiWac 2022) has made coarse location the natural target at networking venues for decades.

So: two communities work on "where next" and barely cite each other. The RecSys/IR side (the survey's
universe) ranks venues; the mobility side predicts regions/activities. Our paper deliberately sits at the
boundary — community-A data (LBSN check-ins) and machinery (graph embeddings), community-B tasks and
metrics, submitted at a community-B venue. A task cannot simultaneously be "judged worthless by the field"
and be the standard formulation of the neighboring field.

**The steelman, and what it does and does not answer.** The strongest version of the worry: "RecSys has
had these labels since 2013 and *every time* kept them auxiliary — revealed preference that the pair is
not worth predicting on its own." Answer: that preference is endogenous to the community's benchmark
objective (papers compete on HR/NDCG over the POI vocabulary; within that objective, auxiliary use is the
local optimum); no evidence exists that anyone evaluated the pair as an end task and *rejected* it —
absence of a slot, not a negative result. Meanwhile HMT-GRN itself validates region prediction as
valuable (its multi-task User-Region matrices are its own sparsity fix). **Two residual risks, named
honestly:**
- **R1 — "predict the POI and derive both."** Partially answered in-paper (intro hardness argument, §2.2
  actionability, the cascade tie). But note: no baseline in the paper actually *derives* category+region
  from a predicted venue — faithful STAN was retargeted to rank regions directly, and the cascade tests
  coupling topology inside our model. An IR-trained reviewer can still ask for the literal derive
  baseline; the current answer is an argument (label-space size, sparsity, cold POIs, low published
  exact-POI Acc@1) plus adjacent evidence, not a head-to-head. See `MOBILITY_PLAN.md §5` for a cheap
  rebuttal-experiment option.
- **R2 — no third-party champion for the *pairing* yet.** The application pull (anticipatory services
  need what-kind + where-roughly) is argued in our motivation, not demonstrated by external demand. The
  venue choice and the venue-local handover cite (§4.1a) are the mitigation.

**Novel in three precise senses** (all matching what the paper already claims — its calibration is
correct): (1) *task framing* — the pair as co-equal end targets over a fixed fine-grained administrative
partition, next-place dropped; every near-neighbor stops short (HMT-GRN: regions as beam-search aids;
MCMG/SGRec/KGTB: auxiliary channels; MCARNN/iMTL/HAMTL: exact place kept; CSLSL/CatDM: category
instrumental in a chain; DRRGNN: per-person regions, not a fixed partition). (2) *method combination* —
per-visit vectors inside hierarchical graph infomax, with CTLE run as the control isolating exactly that.
(3) *the empirical account* — the substrate lift, the MTL-vs-dedicated accounting, the region gain
scaling with region count, the near-zero gradient-correlation mechanism. **Not novel and correctly never
claimed:** per-visit vectors per se, region targets per se, category prediction per se, hard sharing.

---
*Produced 2026-07-06 by a multi-agent triage (4 deep-reads + 4 adversarial verifications + 4 bib-cluster
audits + 1 web gap scan; 13 agents), extended same day with the survey-absence analysis (§7; survey
re-read + verified mobility-side evidence sweep). Verdicts spot-verified by hand against the PDF texts
and the live web (venues, DOIs, author lists). Source PDFs and bib: `~/Desktop/mestrado/artigos/refs/`.*
