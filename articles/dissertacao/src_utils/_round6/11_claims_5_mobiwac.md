# 11_claims_5_mobiwac.md — citation claim-support audit, Chapter 5, MobiWac 2026

**Unit:** `src/chapters/5_mobiwac.tex`  
**Run:** 2026-07-28, round 6, as the per-chapter pass the author asked for by name (COD-008 decision).  
**Errata regime for this unit:** UNDER REVIEW: a correction is applied to the dissertation AND to articles/[mobiwac]/src/ so the two texts stay identical, then named in that article's own errata record rather than in Appendix B (author instruction, 2026-07-27).

## 1 · Counts (every citation in the unit, not a sample)

- `\cite` commands in the unit: **56**, on **43** source lines, carrying **60** key instances (a multi-key `\cite` counts once per key). Every one was audited.
- Distinct bibliography keys used: **33**.
- Verdicts: **SUPPORTED** 57, **PARTIAL** 3.

Comments were stripped before counting, so a key that appears only inside a `%` comment is not counted; every counted site renders.

## 2 · Every citation, with its verdict

Verdict scale: SUPPORTED, the citing sentence's attribution is present in or a fair paraphrase of the
source; PARTIAL, part is supported and part is not, or the sentence is stronger than the source;
NOT-SUPPORTED, the attribution is absent from or contradicted by the source; UNVERIFIABLE, the source
of record does not carry enough to decide and the attribution is not implausible.

| # | Site (file:line) | Key | Verdict | Evidence quoted from the source (under 20 words) |
|---|---|---|---|---|
| 1 | `5_mobiwac.tex:40` | `bastug2014edge` | SUPPORTED | "peak traffic demands can be substantially reduced by proactively serving predictable user demands via caching at base stations" |
| 2 | `5_mobiwac.tex:40` | `cho2011gowalla` | SUPPORTED | "data from two online location-based social networks, we aim to understand what basic laws govern human motion" |
| 3 | `5_mobiwac.tex:40` | `moura2025mobilityaware` | SUPPORTED | "leverage a large dataset of Foursquare check-ins... Mobility-aware systems play a crucial role in leveraging such insights to design adaptive, data-dr" |
| 4 | `5_mobiwac.tex:40` | `song2010limits` | SUPPORTED | "there was 93% predictability across the whole user base" |
| 5 | `5_mobiwac.tex:40` | `vielhaus2022handover` | SUPPORTED | "Predicting handovers in high mobility scenarios enables networks and applications to adapt ahead of time to improve the Quality of Service" |
| 6 | `5_mobiwac.tex:44` | `caruana1997multitask` | PARTIAL | "what is learned for each task can help other tasks be learned better" |
| 7 | `5_mobiwac.tex:45` | `silva2025mtlnet` | SUPPORTED | "the multi-task learning approach did not consistently yield substantial improvements over the single-task baselines" |
| 8 | `5_mobiwac.tex:53` | `huang2023hgi` | SUPPORTED | "the mutual information among the POI - region - city hierarchy is leveraged as the objective" |
| 9 | `5_mobiwac.tex:53` | `velickovic2019deep` | SUPPORTED | "DGI relies on maximizing mutual information between patch representations and corresponding high-level summaries of graphs" |
| 10 | `5_mobiwac.tex:55` | `wongso2025massivesteps` | SUPPORTED | "a large-scale, publicly available benchmark dataset built upon the Semantic Trails dataset" |
| 11 | `5_mobiwac.tex:96` | `silva2025mtlnet` | SUPPORTED | "a joint MTL architecture that shares lower-level embeddings and sequence encoders while maintaining task-specific heads" |
| 12 | `5_mobiwac.tex:103` | `paiva2026stmtlnet` | SUPPORTED | "supera o baseline em todas as 21 combinações de categoria e estado para classificação" |
| 13 | `5_mobiwac.tex:112` | `velickovic2019deep` | SUPPORTED | "maximizing mutual information between patch representations and corresponding high-level summaries of graphs" |
| 14 | `5_mobiwac.tex:116` | `huang2023hgi` | SUPPORTED | "the mutual information among the POI - region - city hierarchy is leveraged as the objective" |
| 15 | `5_mobiwac.tex:120` | `lin2021ctle` | SUPPORTED | "calculates a location's representation vector with consideration of its specific contextual neighbors in trajectories" |
| 16 | `5_mobiwac.tex:138` | `feng2018deepmove` | SUPPORTED | "DeepMove, an attentional recurrent network for mobility prediction from lengthy and sparse trajectories" |
| 17 | `5_mobiwac.tex:139` | `luo2021stan` | SUPPORTED | "Spatio-Temporal Attention Network for Next Location Recommendation" |
| 18 | `5_mobiwac.tex:139` | `yang2022getnext` | SUPPORTED | "GETNext incorporates the global transition patterns, user's general preference, spatio-temporal context ... into a transformer model" |
| 19 | `5_mobiwac.tex:144` | `luca2021mobilitysurvey` | SUPPORTED | "guide to the leading deep learning solutions to next-location prediction, crowd flow prediction, trajectory generation, and flow generation" |
| 20 | `5_mobiwac.tex:146` | `silva2025mtlnet` | SUPPORTED | "the multi-task learning approach did not consistently yield substantial improvements over the single-task baselines across both tasks" |
| 21 | `5_mobiwac.tex:153` | `Lim2022` | PARTIAL | "learning different User-Region matrices of lower sparsities in a multi-task setting" |
| 22 | `5_mobiwac.tex:153` | `sun2024mcmg` | SUPPORTED | "local multi-channel (i.e., region, category, and POI channels) encoder" |
| 23 | `5_mobiwac.tex:158` | `zhu2022drrgnn` | SUPPORTED | "developing models that can answer... (2) Which region will be the next AR, and (3) Why do people make this regional mobility" |
| 24 | `5_mobiwac.tex:160` | `sun2025kgtb` | SUPPORTED | "introduces multiple behavior-specific prediction tasks for LLM fine-tuning, e.g., POI, category, and region visit behaviors" |
| 25 | `5_mobiwac.tex:166` | `Liao2018` | SUPPORTED | "Multi-task Context Aware Recurrent Neural Network to leverage the spatial activity topic for activity and location prediction" |
| 26 | `5_mobiwac.tex:168` | `wang2025hamtl` | SUPPORTED | "Hierarchy Aware-based Multi-task Learning for User Location Prediction" |
| 27 | `5_mobiwac.tex:171` | `ye2013nextmove` | SUPPORTED | "predict the category of user activity at the next step and then predict the most likely location given the estimated category distribution" |
| 28 | `5_mobiwac.tex:172` | `huang2024cslsl` | SUPPORTED | "explicitly model the "when → what → where", a.k.a. "time → activity → location" decision logic" |
| 29 | `5_mobiwac.tex:174` | `yu2020catdm` | SUPPORTED | "incorporates POI category and geographical influence to reduce search space" |
| 30 | `5_mobiwac.tex:181` | `caruana1997multitask` | PARTIAL | "learning tasks in parallel while using a shared representation" |
| 31 | `5_mobiwac.tex:183` | `nash` | SUPPORTED | "combine per-task gradients into a joint update direction using a particular heuristic" |
| 32 | `5_mobiwac.tex:183` | `yu2020pcgrad` | SUPPORTED | "propose a form of gradient surgery that projects a task's gradient onto the normal plane" |
| 33 | `5_mobiwac.tex:184` | `xin2022domtl` | SUPPORTED | "MTO methods do not yield any performance improvements beyond what is achievable via traditional optimization approaches" |
| 34 | `5_mobiwac.tex:258` | `silva2019urbancomputing` | SUPPORTED | "a survey of recent urban computing studies that make use of LBSN data" |
| 35 | `5_mobiwac.tex:260` | `moura2025mobilityaware` | SUPPORTED | "Key points of interest (especially transportation hubs and cultural landmarks) serve as essential connectors shaping network flow" |
| 36 | `5_mobiwac.tex:280` | `huang2023hgi` | SUPPORTED | "row-wise shuffling of the POI graph's feature matrix Xp ... to form a corrupted graph" |
| 37 | `5_mobiwac.tex:280` | `velickovic2019deep` | SUPPORTED | "maximizing mutual information between patch representations and corresponding high-level summaries" |
| 38 | `5_mobiwac.tex:300` | `caruana1997multitask` | SUPPORTED | "learning tasks in parallel while using a shared representation; what is learned for each task can help other tasks" |
| 39 | `5_mobiwac.tex:327` | `cho2011gowalla` | SUPPORTED | "data from two online location-based social networks" |
| 40 | `5_mobiwac.tex:328` | `wongso2025massivesteps` | SUPPORTED | "Massive-STEPS spans 15 geographically and culturally diverse cities" |
| 41 | `5_mobiwac.tex:388` | `holm1979` | SUPPORTED | "a simple and widely applicable multiple test procedure of the sequentially rejective type" |
| 42 | `5_mobiwac.tex:388` | `lakens2017tost` | SUPPORTED | "an upper and lower equivalence bound is specified based on the smallest effect size of interest" |
| 43 | `5_mobiwac.tex:392` | `Lim2022` | SUPPORTED | "Hierarchical Multi-Task Graph Recurrent Network (HMT-GRN) approach" |
| 44 | `5_mobiwac.tex:392` | `capanema2023poirgnn` | SUPPORTED | "Combining recurrent and Graph Neural Networks to predict the next place's category" |
| 45 | `5_mobiwac.tex:392` | `li2025rehdm` | SUPPORTED | "ReHDM utilizes regional encoding to mine the potential spatial relationships among POIs with coarse-grained geographical information" |
| 46 | `5_mobiwac.tex:392` | `luo2021stan` | SUPPORTED | "STAN explicitly exploits relative spatiotemporal information of all the check-ins with self-attention layers along the trajectory" |
| 47 | `5_mobiwac.tex:394` | `lin2021ctle` | SUPPORTED | "calculates a location's representation vector with consideration of its specific contextual neighbors in trajectories" |
| 48 | `5_mobiwac.tex:396` | `huang2023hgi` | SUPPORTED | "aggregate POI embeddings and generate region raw embeddings" |
| 49 | `5_mobiwac.tex:396` | `huang2024cslsl` | SUPPORTED | "explicitly model the “ when → what → where ”, a.k.a. “ time → activity → location ” decision logic" |
| 50 | `5_mobiwac.tex:396` | `ye2013nextmove` | SUPPORTED | "predict the category of user activity at the next step and then predict the most likely location given the estimated category distribution" |
| 51 | `5_mobiwac.tex:396` | `yu2020catdm` | SUPPORTED | "incorporates POI category and geographical influence to reduce search space" |
| 52 | `5_mobiwac.tex:404` | `huang2023hgi` | SUPPORTED | "aggregate POI embeddings and generate region raw embeddings" |
| 53 | `5_mobiwac.tex:409` | `lin2021ctle` | SUPPORTED | "calculates a location's representation vector with consideration of its specific contextual neighbors in trajectories" |
| 54 | `5_mobiwac.tex:578` | `caruana1997multitask` | SUPPORTED | "improves generalization by using the domain information contained in the training signals of related tasks" |
| 55 | `5_mobiwac.tex:664` | `Lim2022` | SUPPORTED | "learning different User-Region matrices of lower sparsities in a multi-task setting" |
| 56 | `5_mobiwac.tex:664` | `luo2021stan` | SUPPORTED | "STAN explicitly exploits relative spatiotemporal information of all the check-ins with self-attention layers" |
| 57 | `5_mobiwac.tex:665` | `li2025rehdm` | SUPPORTED | "ReHDM utilizes regional encoding to mine the potential spatial relationships among POIs" |
| 58 | `5_mobiwac.tex:666` | `capanema2023poirgnn` | SUPPORTED | "Combining recurrent and Graph Neural Networks to predict the next place's category" |
| 59 | `5_mobiwac.tex:750` | `huang2024cslsl` | SUPPORTED | "utilizes a causal structure based on multi-task learning to explicitly model the "when -> what -> where" ... decision logic" |
| 60 | `5_mobiwac.tex:810` | `moura2025mobilityaware` | SUPPORTED | "One potential research direction is the integration of the analyzed metrics with machine learning algorithms" |
## 3 · Failures and partials in this unit, in detail

### `5_mobiwac.tex:44` — `caruana1997multitask` — **PARTIAL**

**Citing sentence.** Sharing one representation across tasks has a cost: in multi-task learning (MTL), one model does several jobs at once by sharing most of its parts, so the shared parameters can converge to a compromise optimal for neither task, helping one while hurting the other~\cite{caruana1997multitask}.

**Reference resolved.** DOI 10.1023/A:1007379606734. Source of record: Crossref REST; OpenAlex API; PDF in repo: 10.1023_A_1007379606734.pdf. Record reads: Multitask Learning | Machine Learning | 1997 | type journal-article.

**Located passage.** "what is learned for each task can help other tasks be learned better"

**Why.** The compromise-optimal-for-neither mechanism is the negative-transfer reading of shared representations. Caruana 1997 is the origin of the shared-representation idea and the paper does discuss when MTL helps, but the abstract states the positive direction. The chapter itself corrected an adjacent claim in the same passage in round 4 (comment at :46-50).

**Recommended disposition.** Under review, so a change propagates to articles/[mobiwac]/src/ and to that article's errata rather than Appendix B. Lowest-cost repair: cite a work whose stated finding is negative transfer. standley2020tasks says "often leads to inferior overall performance as task objectives can compete" and is already in the bibliography, cited for exactly this at 2_fundamentals.tex:310.

### `5_mobiwac.tex:153` — `Lim2022` — **PARTIAL**

**Citing sentence.** The field increasingly models several granularities at once; in those systems, category and region are auxiliary signals that help a primary next-place task (MCMG \cite{sun2024mcmg}, HMT-GRN \cite{Lim2022}).

**Reference resolved.** DOI 10.1145/3477495.3531989. Source of record: Crossref REST; OpenAlex API. Record reads: Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation | Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval | 2022 | type proceedings-article.

**Located passage.** "learning different User-Region matrices of lower sparsities in a multi-task setting"

**Why.** In HMT-GRN region IS a multi-task target, used to alleviate User-POI sparsity and then searched hierarchically toward the next POI. So "auxiliary signals that help a primary next-place task" is right about the ROLE (next POI is the end target) and understates that region is a trained target. The chapter's own next sentences make exactly this distinction, so the paragraph as a whole is accurate.

**Recommended disposition.** Leave; the following sentences carry the distinction. If tightened, say the coarse target is trained but subordinate.

### `5_mobiwac.tex:181` — `caruana1997multitask` — **PARTIAL**

**Citing sentence.** On optimization, we are conservative by design: joint training with a fixed loss weighting is standard practice \cite{caruana1997multitask}, not itself our contribution.

**Reference resolved.** DOI 10.1023/A:1007379606734. Source of record: Crossref REST; OpenAlex API; PDF in repo: 10.1023_A_1007379606734.pdf. Record reads: Multitask Learning | Machine Learning | 1997 | type journal-article.

**Located passage.** "learning tasks in parallel while using a shared representation"

**Why.** Joint training with a shared representation is supported. "with a fixed loss weighting is standard practice" is a claim about current practice; the sentence's own next clause cites xin2022domtl and kurin2022scalarization, which do establish that a fixed or uniform weighting is the baseline to beat.

**Recommended disposition.** Leave, or move the fixed-weighting clause onto kurin2022scalarization / xin2022domtl, which are cited two lines later.

## 4 · Source ledger for this unit

Every distinct key cited in this unit, the identifier it resolved by, and where I opened it this session.

| Key | Identifier | Opened at | Record as returned |
|---|---|---|---|
| `Liao2018` | DOI 10.24963/ijcai.2018/477 | Crossref REST; OpenAlex API | Predicting Activity and Location with Multi-task Context Aware Recurrent Neural Network \| Proceedings of the Twenty-Seventh International Joint Conference on Artificial  |
| `Lim2022` | DOI 10.1145/3477495.3531989 | Crossref REST; OpenAlex API | Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation \| Proceedings of the 45th International ACM SIGIR Conference on Research and Development in I |
| `bastug2014edge` | DOI 10.1109/MCOM.2014.6871674 | Crossref REST; OpenAlex API | Living on the edge: The role of proactive caching in 5G wireless networks \| IEEE Communications Magazine \| 2014 \| type journal-article |
| `capanema2023poirgnn` | DOI 10.1016/j.adhoc.2022.103016 | Crossref REST; OpenAlex API | Combining recurrent and Graph Neural Networks to predict the next place’s category \| Ad Hoc Networks \| 2023 \| type journal-article |
| `caruana1997multitask` | DOI 10.1023/A:1007379606734 | Crossref REST; OpenAlex API; PDF in repo: 10.1023_A_1007379606734.pdf | Multitask Learning \| Machine Learning \| 1997 \| type journal-article |
| `cho2011gowalla` | DOI 10.1145/2020408.2020579 | Crossref REST; OpenAlex API | Friendship and mobility \| Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining \| 2011 \| type proceedings-article |
| `feng2018deepmove` | DOI 10.1145/3178876.3186058 | Crossref REST; OpenAlex API | DeepMove \| Proceedings of the 2018 World Wide Web Conference on World Wide Web - WWW '18 \| 2018 \| type proceedings-article |
| `holm1979` | no identifier in the bib entry | OpenAlex API | A Simple Sequentially Rejective Multiple Test Procedure \| Scandinavian Journal of Statistics \| 1979 \| type article |
| `huang2023hgi` | DOI 10.1016/j.isprsjprs.2022.11.021 | Crossref REST; OpenAlex API; Semantic Scholar API; PDF in repo: Learning urban region representations with POIs and hierarchical graph infomax.pdf | Learning urban region representations with POIs and hierarchical graph infomax \| ISPRS Journal of Photogrammetry and Remote Sensing \| 2023 \| type journal-article |
| `huang2024cslsl` | DOI 10.1140/epjds/s13688-024-00460-7 | Crossref REST; OpenAlex API | Human mobility prediction with causal and spatial-constrained multi-task network \| EPJ Data Science \| 2024 \| type journal-article |
| `lakens2017tost` | DOI 10.1177/1948550617697177 | Crossref REST; OpenAlex API | Equivalence Tests \| Social Psychological and Personality Science \| 2017 \| type journal-article |
| `li2025rehdm` | DOI 10.24963/ijcai.2025/343 | Crossref REST; OpenAlex API | Beyond Individual and Point: Next POI Recommendation via Region-aware Dynamic Hypergraph with Dual-level Modeling \| Proceedings of the Thirty-Fourth International Joint  |
| `lin2021ctle` | DOI 10.1609/aaai.v35i5.16548 | Crossref REST; OpenAlex API | Pre-training Context and Time Aware Location Embeddings from Spatial-Temporal Trajectories for User Next Location Prediction \| Proceedings of the AAAI Conference on Arti |
| `luca2021mobilitysurvey` | DOI 10.1145/3485125 | Crossref REST; OpenAlex API | A Survey on Deep Learning for Human Mobility \| ACM Computing Surveys \| 2021 \| type journal-article |
| `luo2021stan` | DOI 10.1145/3442381.3449998 | Crossref REST; OpenAlex API | STAN: Spatio-Temporal Attention Network for Next Location Recommendation \| Proceedings of the Web Conference 2021 \| 2021 \| type proceedings-article |
| `moura2025mobilityaware` | DOI 10.1109/MSWiM67937.2025.11308734 | Crossref REST; OpenAlex API | On the Design of Mobility-Aware Systems: A Tourist’s Perspective \| 2025 International Conference on Modeling, Analysis and Simulation of Wireless and Mobile Systems (MSW |
| `nash` | no identifier in the bib entry | arXiv API; OpenAlex API | Multi-Task Learning as a Bargaining Game \| arXiv preprint \| 2022 \| type posted-content |
| `paiva2026stmtlnet` | DOI 10.5753/courb.2026.22960 | Crossref REST; OpenAlex API | ST-MTLNet: Representações Espaço-Temporais de Pontos de Interesse para Aprendizado Multitarefa \| Anais do X Workshop de Computação Urbana (CoUrb 2026) \| 2026 \| type pr |
| `silva2019urbancomputing` | DOI 10.1145/3301284 | Crossref REST; OpenAlex API | Urban Computing Leveraging Location-Based Social Network Data \| ACM Computing Surveys \| 2019 \| type journal-article |
| `silva2025mtlnet` | DOI 10.21528/CBIC2025-1191324 | Crossref REST; OpenAlex API | An Investigation into Multi-Task Learning for Point-of-Interest Category Classification and Next-POI Prediction \| Anais do XVII Congresso Brasileiro de Inteligência Comp |
| `song2010limits` | DOI 10.1126/science.1177170 | Crossref REST; OpenAlex API; PDF in repo: 201002-19_Science-Predictability.pdf | Limits of Predictability in Human Mobility \| Science \| 2010 \| type journal-article |
| `sun2024mcmg` | DOI 10.1145/3592789 | Crossref REST; OpenAlex API | A Multi-channel Next POI Recommendation Framework with Multi-granularity Check-in Signals \| ACM Transactions on Information Systems \| 2023 \| type journal-article |
| `sun2025kgtb` | DOI 10.48550/arXiv.2509.12350 | arXiv API; OpenAlex API | Knowledge Graph Tokenization for Behavior-Aware Generative Next POI Recommendation \| arXiv preprint \| 2025 \| type posted-content |
| `velickovic2019deep` | no identifier in the bib entry | OpenAlex API | Deep Graph Infomax \| Apollo (University of Cambridge) \| 2018 \| type conference-paper |
| `vielhaus2022handover` | DOI 10.1145/3551660.3560913 | Crossref REST; OpenAlex API | Handover Predictions as an Enabler for Anticipatory Service Adaptations in Next-Generation Cellular Networks \| Proceedings of the 20th ACM International Symposium on Mob |
| `wang2025hamtl` | DOI 10.1007/s11227-025-07643-7 | Crossref REST; OpenAlex API | Hierarchy aware-based multi-task learning for user location prediction \| The Journal of Supercomputing \| 2025 \| type journal-article |
| `wongso2025massivesteps` | no identifier in the bib entry | arXiv API; OpenAlex API | Massive-STEPS: Massive Semantic Trajectories for Understanding POI Check-ins -- Dataset and Benchmarks \| arXiv preprint \| 2025 \| type posted-content |
| `xin2022domtl` | no identifier in the bib entry | arXiv API; OpenAlex API | Do Current Multi-Task Optimization Methods in Deep Learning Even Help? \| arXiv preprint \| 2022 \| type posted-content |
| `yang2022getnext` | DOI 10.1145/3477495.3531983 | Crossref REST; OpenAlex API | GETNext \| Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval \| 2022 \| type proceedings-article |
| `ye2013nextmove` | DOI 10.1137/1.9781611972832.19 | Crossref REST; OpenAlex API | What's Your Next Move: User Activity Prediction in Location-based Social Networks \| Proceedings of the 2013 SIAM International Conference on Data Mining \| 2013 \| type  |
| `yu2020catdm` | DOI 10.1145/3366423.3380202 | Crossref REST; OpenAlex API | A Category-Aware Deep Model for Successive POI Recommendation on Sparse Check-in Data \| Proceedings of The Web Conference 2020 \| 2020 \| type proceedings-article |
| `yu2020pcgrad` | no identifier in the bib entry | arXiv API; OpenAlex API | Gradient Surgery for Multi-Task Learning \| arXiv preprint \| 2020 \| type posted-content |
| `zhu2022drrgnn` | DOI 10.1145/3529091 | Crossref REST; OpenAlex API | Predicting a Person’s Next Activity Region with a Dynamic Region-Relation-Aware Graph Neural Network \| ACM Transactions on Knowledge Discovery from Data \| 2022 \| type  |

## 5 · Errata regime for anything changed here

Chapter 5 is **under review**. Per the author's instruction of 2026-07-27, a correction here is
applied to the dissertation AND to the submitted source at `articles/[mobiwac]/src/` so the two
texts stay identical, then named in that article's own errata record rather than in Appendix B. All
three PARTIAL sites below were matched against the submitted source
(`articles/[mobiwac]/src/sections/01_introduction.tex` and `02_related.tex`) and are present there
verbatim, so any change is a two-file change.

## 6 · Two sites a naive check inverts

**`5_mobiwac.tex:96`, `silva2025mtlnet`.** An abstract-only check reports this as reversed: the CBIC
abstract says the model "shares lower-level embeddings and sequence encoders while maintaining
task-specific heads", while the chapter says "task-specific encoders feed shared layers". Both are
true of the same architecture, and the chapter's version is the one the CBIC **method section** of
record states: inputs "are first processed by separate, task-specific encoders", then FiLM
conditioning on a learnable task embedding, then shared residual layers, then task-specific heads,
with Nash-MTL aggregating gradients (`articles/CBIC___MTL/sections/method.tex`). **SUPPORTED.**

**`apx_b_errata.tex:220`** (in the appendices unit) is the same class of false positive in the other
direction: an errata row cites the source that contradicts the text being corrected. Recorded there.

## 7 · What I could not confirm in this chapter

- `capanema2023poirgnn` and `wang2025hamtl` return no abstract at Crossref, OpenAlex or Semantic
  Scholar, and their Elsevier and Springer landing pages are outside the network allowlist. Both are
  used as identity-of-baseline or pattern-continuation pointers that the resolved record and title
  support; no mechanism claim rests on either.
- `:409`, `lin2021ctle`: the negative clause ("the category vocabulary never enters its training") is
  consistent with the abstract, which names only locations and temporal information as inputs, but an
  abstract cannot prove an absence. Narrow `[VERIFY]`.
- `:750`, `huang2024cslsl`: the cited internal comparison (the chain against a shared-trunk parallel
  variant on CSLSL's own benchmarks) is an ablation that the abstract does not carry. Narrow
  `[VERIFY]`; the sentence attributes it to the paper explicitly, so it needs a page.
- The comparative results claims at `:664-666` were **not** audited here. What a citation must
  support at those sites is the baseline's identity, which each record does. The comparison itself
  is a number claim under AGENT_GUARDRAILS section 2, whose single source of truth for this chapter
  is `RESULTS_BOARD.md`. That is a numbers audit, not this one, and I did not perform it.
