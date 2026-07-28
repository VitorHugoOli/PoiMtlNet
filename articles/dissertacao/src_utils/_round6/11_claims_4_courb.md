# 11_claims_4_courb.md — citation claim-support audit, Chapter 4, CoUrb 2026

**Unit:** `src/chapters/4_courb.tex`  
**Run:** 2026-07-28, round 6, as the per-chapter pass the author asked for by name (COD-008 decision).  
**Errata regime for this unit:** PUBLISHED: a correction to reproduced prose is applied in the dissertation and listed in Appendix B; the published article record is not edited.

## 1 · Counts (every citation in the unit, not a sample)

- `\cite` commands in the unit: **50**, on **32** source lines, carrying **53** key instances (a multi-key `\cite` counts once per key). Every one was audited.
- Distinct bibliography keys used: **28**.
- Verdicts: **SUPPORTED** 47, **PARTIAL** 4, **NOT-SUPPORTED** 2.

Comments were stripped before counting, so a key that appears only inside a `%` comment is not counted; every counted site renders.

## 2 · Every citation, with its verdict

Verdict scale: SUPPORTED, the citing sentence's attribution is present in or a fair paraphrase of the
source; PARTIAL, part is supported and part is not, or the sentence is stronger than the source;
NOT-SUPPORTED, the attribution is absent from or contradicted by the source; UNVERIFIABLE, the source
of record does not carry enough to decide and the attribution is not implausible.

| # | Site (file:line) | Key | Verdict | Evidence quoted from the source (under 20 words) |
|---|---|---|---|---|
| 1 | `4_courb.tex:18` | `paiva2026stmtlnet` | SUPPORTED | "title=ST-MTLNet: Representações Espaço-Temporais de Pontos de Interesse para Aprendizado Multitarefa; venue=Anais do X Workshop de Computação Urbana (" |
| 2 | `4_courb.tex:25` | `cho2011gowalla` | SUPPORTED | "data from two online location-based social networks" |
| 3 | `4_courb.tex:25` | `jure2014snap` | SUPPORTED | "A collection of more than 50 large network datasets" |
| 4 | `4_courb.tex:32` | `caruana1997multitask` | SUPPORTED | "improves generalization by using the domain information contained in the training signals of related tasks" |
| 5 | `4_courb.tex:34` | `silva2025mtlnet` | SUPPORTED | "shares lower-level embeddings and sequence encoders while maintaining task-specific heads" |
| 6 | `4_courb.tex:38` | `wu2024torchspatial` | SUPPORTED | "a unified location encoding framework that consolidates 15 commonly recognized location encoders" |
| 7 | `4_courb.tex:40` | `jure2014snap` | SUPPORTED | "A collection of more than 50 large network datasets" |
| 8 | `4_courb.tex:60` | `feng2017poi2vec` | SUPPORTED | "we propose a new latent representation model POI2Vec that is able to incorporate the geographical influence" |
| 9 | `4_courb.tex:60` | `rahmani2019category` | PARTIAL | "previous studies fail to capture crucial information about POIs such as categorical information" |
| 10 | `4_courb.tex:62` | `huang2023hgi` | SUPPORTED | "aggregate POI embeddings and generate region raw embeddings" |
| 11 | `4_courb.tex:62` | `velickovic2019deep` | SUPPORTED | "DGI relies on maximizing mutual information between patch representations and corresponding high-level summaries of graphs" |
| 12 | `4_courb.tex:68` | `wu2024torchspatial` | SUPPORTED | "a unified location encoding framework that consolidates 15 commonly recognized location encoders, ensuring scalability and reproducibility" |
| 13 | `4_courb.tex:70` | `mai2023sphere2vecgeneralpurposelocationrepresentation` | SUPPORTED | "propose a multi-scale location encoder called Sphere2Vec which can preserve spherical distances when encoding point coordinates on a spherical surface" |
| 14 | `4_courb.tex:70` | `russwurm2024geographiclocationencodingspherical` | SUPPORTED | "combines spherical harmonic basis functions... with sinusoidal representation networks (SirenNets)... for globally distributed geographic data" |
| 15 | `4_courb.tex:70` | `sitzmann2020implicit` | SUPPORTED | "leverage periodic activation functions for implicit neural representations...ideally suited for representing complex natural signals" |
| 16 | `4_courb.tex:76` | `sun2020go` | SUPPORTED | "a nonlocal network for long-term preference modeling and a geo-dilated RNN for short-term preference learning" |
| 17 | `4_courb.tex:76` | `sun2024transtarec` | SUPPORTED | "fuse user preference and temporal influence... unification with user preference and sequential dynamics" |
| 18 | `4_courb.tex:78` | `kazemi2019time2vec` | SUPPORTED | "model-agnostic vector representation for time, called Time2Vec" |
| 19 | `4_courb.tex:82` | `Halder2022` | SUPPORTED | "propose a multi-task, multi-head attention transformer model...recommends the next POIs...and predicts queuing time...simultaneously" |
| 20 | `4_courb.tex:82` | `Liao2018` | SUPPORTED | "integrate the sequential dependency and temporal regularity of spatial activity topics" |
| 21 | `4_courb.tex:82` | `Xia2020` | PARTIAL | "exploits a structure of generative adversarial networks (GAN) simultaneously considering temporal check-ins and geographical locations" |
| 22 | `4_courb.tex:82` | `caruana1997multitask` | SUPPORTED | "Multitask Learning is an approach to inductive transfer that improves generalization by using the domain information" |
| 23 | `4_courb.tex:84` | `Lim2022` | SUPPORTED | "learning different User-Region matrices of lower sparsities in a multi-task setting" |
| 24 | `4_courb.tex:84` | `Xu2023` | SUPPORTED | "utilizes the predefined category hierarchy to regularize the relatedness among categories" |
| 25 | `4_courb.tex:89` | `silva2025mtlnet` | SUPPORTED | "We propose a joint MTL architecture that shares lower-level embeddings and sequence encoders while maintaining task-specific heads" |
| 26 | `4_courb.tex:96` | `kazemi2019time2vec` | SUPPORTED | "model-agnostic vector representation for time, called Time2Vec, that can be easily imported into many existing and future architectures" |
| 27 | `4_courb.tex:96` | `silva2025mtlnet` | SUPPORTED | "a joint MTL architecture that shares lower-level embeddings and sequence encoders" |
| 28 | `4_courb.tex:96` | `wu2024torchspatial` | SUPPORTED | "a learning framework and benchmark for location (point) encoding, which is one of the most fundamental data types" |
| 29 | `4_courb.tex:105` | `silva2025mtlnet` | SUPPORTED | "We propose a joint MTL architecture that shares lower-level embeddings and sequence encoders while maintaining task-specific heads" |
| 30 | `4_courb.tex:109` | `caruana1997multitask` | SUPPORTED | "learning tasks in parallel while using a shared representation" |
| 31 | `4_courb.tex:109` | `perez2018film` | SUPPORTED | "FiLM layers influence neural network computation via a simple, feature-wise affine transformation based on conditioning information" |
| 32 | `4_courb.tex:109` | `silva2025mtlnet` | SUPPORTED | "We propose a joint MTL architecture that shares lower-level embeddings and sequence encoders while maintaining task-specific heads" |
| 33 | `4_courb.tex:116` | `baxter2000model` | SUPPORTED | "the learner can search for a hypothesis space that contains good solutions to many of the problems" |
| 34 | `4_courb.tex:120` | `nash` | SUPPORTED | "viewing the gradients combination step as a bargaining game, where tasks negotiate to reach an agreement on a joint direction" |
| 35 | `4_courb.tex:124` | `silva2025mtlnet` | SUPPORTED | "We propose a joint MTL architecture that shares lower-level embeddings and sequence encoders while maintaining task-specific heads" |
| 36 | `4_courb.tex:124` | `silva2025mtlnet` | SUPPORTED | "a joint MTL architecture that shares lower-level embeddings and sequence encoders" |
| 37 | `4_courb.tex:124` | `velickovic2019deep` | SUPPORTED | "learning node representations within graph-structured data in an unsupervised manner" |
| 38 | `4_courb.tex:134` | `mai2023sphere2vecgeneralpurposelocationrepresentation` | SUPPORTED | "propose a multi-scale location encoder called Sphere2Vec which can preserve spherical distances when encoding point coordinates on a spherical surface" |
| 39 | `4_courb.tex:134` | `russwurm2024geographiclocationencodingspherical` | PARTIAL | "both spherical harmonics and sinusoidal representation networks are competitive on their own but set state-of-the-art performances when combined" |
| 40 | `4_courb.tex:134` | `wu2024torchspatial` | SUPPORTED | "TorchSpatial contains three key components: 1) a unified location encoding framework that consolidates 15 commonly recognized location encoders" |
| 41 | `4_courb.tex:153` | `russwurm2024geographiclocationencodingspherical` | PARTIAL | "sinusoidal representation networks (SirenNets) that can be interpreted as learned Double Fourier Sphere embedding" |
| 42 | `4_courb.tex:157` | `mai2023sphere2vecgeneralpurposelocationrepresentation` | SUPPORTED | "we propose a multi-scale location encoder called Sphere2Vec which can preserve spherical distances when encoding point coordinates on a spherical surf" |
| 43 | `4_courb.tex:161` | `sun2020go` | NOT-SUPPORTED | "a nonlocal network for long-term preference modeling and a geo-dilated RNN for short-term preference learning" |
| 44 | `4_courb.tex:163` | `kazemi2019time2vec` | SUPPORTED | "model-agnostic vector representation for time, called Time2Vec, that can be easily imported into many existing" |
| 45 | `4_courb.tex:176` | `huang2023hgi` | SUPPORTED | "the mutual information among the POI - region - city hierarchy is leveraged as the objective" |
| 46 | `4_courb.tex:208` | `grover2016node2vec` | SUPPORTED | "design a biased random walk procedure, which efficiently explores diverse neighborhoods" |
| 47 | `4_courb.tex:208` | `mikolov2013negsampling` | SUPPORTED | "a simple alternative to the hierarchical softmax called negative sampling" |
| 48 | `4_courb.tex:208` | `mikolov2013word2vec` | SUPPORTED | "two novel model architectures for computing continuous vector representations of words" |
| 49 | `4_courb.tex:219` | `belkin2003laplacian` | NOT-SUPPORTED | "a geometrically motivated algorithm for representing the high-dimensional data ... nonlinear dimensionality reduction" |
| 50 | `4_courb.tex:223` | `huang2023hgi` | SUPPORTED | "Learning urban region representations with POIs and hierarchical graph infomax" |
| 51 | `4_courb.tex:248` | `silva2025mtlnet` | SUPPORTED | "a joint MTL architecture that shares lower-level embeddings and sequence encoders" |
| 52 | `4_courb.tex:277` | `cho2011gowalla` | SUPPORTED | "data from two online location-based social networks" |
| 53 | `4_courb.tex:277` | `jure2014snap` | SUPPORTED | "A collection of more than 50 large network datasets" |
## 3 · Failures and partials in this unit, in detail

### `4_courb.tex:161` — `sun2020go` — **NOT-SUPPORTED**

**Citing sentence.** Human mobility patterns exhibit cyclical regularities, such as meal times and weekly movements, which carry discriminative information about the functional nature of the visited POIs \cite{sun2020go}.

**Reference resolved.** DOI 10.1609/aaai.v34i01.5353. Source of record: Crossref REST; OpenAlex API. Record reads: Where to Go Next: Modeling Long- and Short-Term User Preferences for Point-of-Interest Recommendation | Proceedings of the AAAI Conference on Artificial Intelligence | 2020 | type journal-article.

**Located passage.** "a nonlocal network for long-term preference modeling and a geo-dilated RNN for short-term preference learning"

**Why.** LSTPM (AAAI 2020) models long- and short-term user preference for next-POI recommendation. The citing sentence claims that cyclical regularities such as meal times and weekly movements carry discriminative information about the FUNCTIONAL NATURE of visited POIs. That is a claim about temporal signal predicting place semantics, which this paper does not make.

**Recommended disposition.** Published CoUrb prose. The chapter cites kazemi2019time2vec and Xu2023 elsewhere; neither states this either. Either narrow the sentence to the temporal regularity of visits (which sun2020go and cho2011gowalla both support) or find a source for the semantics half.

### `4_courb.tex:219` — `belkin2003laplacian` — **NOT-SUPPORTED**

**Citing sentence.** The implementation incorporates a hierarchical regularization term \cite{belkin2003laplacian} between category and \textit{fclass}: $\mathcal{L}_{\text{hier}} = \sum_{(c,s) \in \mathcal{H}} \left\| \mathbf{e}_s - \mathbf{e}_c \right\|_2^2$, in which $\mathcal{H}$ contains the (category, \textit{fclass}) pairs.

**Reference resolved.** DOI 10.1162/089976603321780317. Source of record: Crossref REST; OpenAlex API. Record reads: Laplacian Eigenmaps for Dimensionality Reduction and Data Representation | Neural Computation | 2003 | type journal-article.

**Located passage.** "a geometrically motivated algorithm for representing the high-dimensional data ... nonlinear dimensionality reduction"

**Why.** Laplacian eigenmaps is a manifold dimensionality-reduction method. The cited object is an L2 penalty pulling a subcategory embedding toward its parent category embedding, that is, a hierarchical regularizer over a known label tree. The connection is at best thematic (graph Laplacian smoothness) and the sentence attributes the term itself.

**Recommended disposition.** Published CoUrb prose. Xu2023, already in the bibliography, regularizes category relatedness with a predefined category hierarchy, which is what this term does. Swap the key, or drop the citation and present the term as the implementation's own.

### `4_courb.tex:60` — `rahmani2019category` — **PARTIAL**

**Citing sentence.** CATAPE (\textit{Category-Aware Location Embedding}) \cite{rahmani2019category} extends this idea by incorporating categorical and sequential information, capturing the geographic influence between POIs based on the temporal sequence of user visits.

**Reference resolved.** DOI 10.1145/3341981.3344240. Source of record: Crossref REST; OpenAlex API. Record reads: Category-Aware Location Embedding for Point-of-Interest Recommendation | Proceedings of the 2019 ACM SIGIR International Conference on Theory of Information Retrieval | 2019 | type proceedings-article.

**Located passage.** "previous studies fail to capture crucial information about POIs such as categorical information"

**Why.** Crossref returns a truncated abstract for this SIGIR ICTIR paper (four sentences). The categorical half is supported. The sequential/temporal-visit-order half is not visible in what the source of record returns.

**Recommended disposition.** Open as [VERIFY] on the sequential clause only. Published CoUrb prose, low exposure.

### `4_courb.tex:82` — `Xia2020` — **PARTIAL**

**Citing sentence.** MTPR \cite{Xia2020} jointly models location and temporal context through geographic LSTMs and adversarial learning.

**Reference resolved.** DOI 10.3390/app10196664. Source of record: Crossref REST; OpenAlex API. Record reads: MTPR: A Multi-Task Learning Based POI Recommendation Considering Temporal Check-Ins and Geographical Locations | Applied Sciences | 2020 | type journal-article.

**Located passage.** "exploits a structure of generative adversarial networks (GAN) simultaneously considering temporal check-ins and geographical locations"

**Why.** Same work, same defect class as 3_cbic.tex:125: adversarial learning and joint temporal-geographic modeling are supported; "geographic LSTMs" is not in the abstract.

**Recommended disposition.** Published CoUrb prose. Same disposition as the Ch.3 site; fix both or neither, so the chapters do not describe one system two ways.

### `4_courb.tex:134` — `russwurm2024geographiclocationencodingspherical` — **PARTIAL**

**Citing sentence.** This chapter selects two \textit{encoders} that represent distinct spatial encoding paradigms: SIREN \cite{russwurm2024geographiclocationencodingspherical}, which models continuous functions through sinusoidal activations with controllable frequencies, and Sphere2Vec-M \cite{mai2023sphere2vecgeneralpurposelocationrepresentation}, which operates directly on spherical coordinates preserving geodesic distance properties.

**Reference resolved.** arXiv:2310.06743. Source of record: arXiv API; OpenAlex API. Record reads: Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks | Published as a conference paper at ICLR 2024 | 2023 | type posted-content.

**Located passage.** "both spherical harmonics and sinusoidal representation networks are competitive on their own but set state-of-the-art performances when combined"

**Why.** The paper proposes spherical harmonics COMBINED with SirenNets. The chapter uses the name SIREN for the sinusoidal-network half and describes only that half. By the paper's own words the halves are separable and each is competitive alone, so naming one is defensible, but the chapter never says which component of the cited work it instantiated.

**Recommended disposition.** Published CoUrb prose. One clause would close it: state that the encoder used is the sinusoidal-representation-network component of that work. This also affects 4_courb.tex:153.

### `4_courb.tex:153` — `russwurm2024geographiclocationencodingspherical` — **PARTIAL**

**Citing sentence.** The SIREN model (\textit{Sinusoidal Representation Networks}) \cite{russwurm2024geographiclocationencodingspherical} models a continuous function $f_\theta : \mathbb{R}^2 \rightarrow \mathbb{R}^{64}$ that directly maps normalized geographic coordinates into a vector \textit{embedding}.

**Reference resolved.** arXiv:2310.06743. Source of record: arXiv API; OpenAlex API. Record reads: Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks | Published as a conference paper at ICLR 2024 | 2023 | type posted-content.

**Located passage.** "sinusoidal representation networks (SirenNets) that can be interpreted as learned Double Fourier Sphere embedding"

**Why.** Same issue as :134. The R^2 -> R^64 map and the 64-dimensional output are this chapter's own configuration, not the paper's claim.

**Recommended disposition.** As :134.

## 4 · Source ledger for this unit

Every distinct key cited in this unit, the identifier it resolved by, and where I opened it this session.

| Key | Identifier | Opened at | Record as returned |
|---|---|---|---|
| `Halder2022` | DOI 10.1007/s10618-022-00865-w | Crossref REST; OpenAlex API | POI recommendation with queuing time and user interest awareness \| Data Mining and Knowledge Discovery \| 2022 \| type journal-article |
| `Liao2018` | DOI 10.24963/ijcai.2018/477 | Crossref REST; OpenAlex API | Predicting Activity and Location with Multi-task Context Aware Recurrent Neural Network \| Proceedings of the Twenty-Seventh International Joint Conference on Artificial  |
| `Lim2022` | DOI 10.1145/3477495.3531989 | Crossref REST; OpenAlex API | Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation \| Proceedings of the 45th International ACM SIGIR Conference on Research and Development in I |
| `Xia2020` | DOI 10.3390/app10196664 | Crossref REST; OpenAlex API | MTPR: A Multi-Task Learning Based POI Recommendation Considering Temporal Check-Ins and Geographical Locations \| Applied Sciences \| 2020 \| type journal-article |
| `Xu2023` | DOI 10.1145/3582553 | Crossref REST; OpenAlex API | TME: Tree-guided Multi-task Embedding Learning towards Semantic Venue Annotation \| ACM Transactions on Information Systems \| 2023 \| type journal-article |
| `baxter2000model` | DOI 10.1613/jair.731 | Crossref REST; OpenAlex API | A Model of Inductive Bias Learning \| Journal of Artificial Intelligence Research \| 2000 \| type journal-article |
| `belkin2003laplacian` | DOI 10.1162/089976603321780317 | Crossref REST; OpenAlex API | Laplacian Eigenmaps for Dimensionality Reduction and Data Representation \| Neural Computation \| 2003 \| type journal-article |
| `caruana1997multitask` | DOI 10.1023/A:1007379606734 | Crossref REST; OpenAlex API; PDF in repo: 10.1023_A_1007379606734.pdf | Multitask Learning \| Machine Learning \| 1997 \| type journal-article |
| `cho2011gowalla` | DOI 10.1145/2020408.2020579 | Crossref REST; OpenAlex API | Friendship and mobility \| Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining \| 2011 \| type proceedings-article |
| `feng2017poi2vec` | DOI 10.1609/aaai.v31i1.10500 | Crossref REST; OpenAlex API | POI2Vec: Geographical Latent Representation for Predicting Future Visitors \| Proceedings of the AAAI Conference on Artificial Intelligence \| 2017 \| type journal-articl |
| `grover2016node2vec` | DOI 10.1145/2939672.2939754 | Crossref REST; OpenAlex API | node2vec \| Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining \| 2016 \| type proceedings-article |
| `huang2023hgi` | DOI 10.1016/j.isprsjprs.2022.11.021 | Crossref REST; OpenAlex API; Semantic Scholar API; PDF in repo: Learning urban region representations with POIs and hierarchical graph infomax.pdf | Learning urban region representations with POIs and hierarchical graph infomax \| ISPRS Journal of Photogrammetry and Remote Sensing \| 2023 \| type journal-article |
| `jure2014snap` | no identifier in the bib entry | OpenAlex API | {SNAP Datasets}: {Stanford} Large Network Dataset Collection \| (no venue in record) \| 2014 \| type article |
| `kazemi2019time2vec` | arXiv:1907.05321 | arXiv API; OpenAlex API | Time2Vec: Learning a Vector Representation of Time \| arXiv preprint \| 2019 \| type posted-content |
| `mai2023sphere2vecgeneralpurposelocationrepresentation` | arXiv:2306.17624 | arXiv API; OpenAlex API | Sphere2Vec: A General-Purpose Location Representation Learning over a Spherical Surface for Large-Scale Geospatial Predictions \| ISPRS Journal of Photogrammetry and Remo |
| `mikolov2013negsampling` | arXiv:1310.4546 | arXiv API; OpenAlex API | Distributed Representations of Words and Phrases and their Compositionality \| arXiv preprint \| 2013 \| type posted-content |
| `mikolov2013word2vec` | arXiv:1301.3781 | arXiv API; OpenAlex API | Efficient Estimation of Word Representations in Vector Space \| arXiv preprint \| 2013 \| type posted-content |
| `nash` | no identifier in the bib entry | arXiv API; OpenAlex API | Multi-Task Learning as a Bargaining Game \| arXiv preprint \| 2022 \| type posted-content |
| `paiva2026stmtlnet` | DOI 10.5753/courb.2026.22960 | Crossref REST; OpenAlex API | ST-MTLNet: Representações Espaço-Temporais de Pontos de Interesse para Aprendizado Multitarefa \| Anais do X Workshop de Computação Urbana (CoUrb 2026) \| 2026 \| type pr |
| `perez2018film` | DOI 10.1609/aaai.v32i1.11671 | Crossref REST; OpenAlex API | FiLM: Visual Reasoning with a General Conditioning Layer \| Proceedings of the AAAI Conference on Artificial Intelligence \| 2018 \| type journal-article |
| `rahmani2019category` | DOI 10.1145/3341981.3344240 | Crossref REST; OpenAlex API | Category-Aware Location Embedding for Point-of-Interest Recommendation \| Proceedings of the 2019 ACM SIGIR International Conference on Theory of Information Retrieval \| |
| `russwurm2024geographiclocationencodingspherical` | arXiv:2310.06743 | arXiv API; OpenAlex API | Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks \| Published as a conference paper at ICLR 2024 \| 2023 \| type posted-conten |
| `silva2025mtlnet` | DOI 10.21528/CBIC2025-1191324 | Crossref REST; OpenAlex API | An Investigation into Multi-Task Learning for Point-of-Interest Category Classification and Next-POI Prediction \| Anais do XVII Congresso Brasileiro de Inteligência Comp |
| `sitzmann2020implicit` | arXiv:2006.09661 | arXiv API; OpenAlex API | Implicit Neural Representations with Periodic Activation Functions \| arXiv preprint \| 2020 \| type posted-content |
| `sun2020go` | DOI 10.1609/aaai.v34i01.5353 | Crossref REST; OpenAlex API | Where to Go Next: Modeling Long- and Short-Term User Preferences for Point-of-Interest Recommendation \| Proceedings of the AAAI Conference on Artificial Intelligence \|  |
| `sun2024transtarec` | DOI 10.1109/ICCEA62105.2024.10603711 | Crossref REST; OpenAlex API | TransTARec: Time-Adaptive Translating Embedding Model for Next POI Recommendation \| 2024 5th International Conference on Computer Engineering and Application (ICCEA) \|  |
| `velickovic2019deep` | no identifier in the bib entry | OpenAlex API | Deep Graph Infomax \| Apollo (University of Cambridge) \| 2018 \| type conference-paper |
| `wu2024torchspatial` | arXiv:2406.15658 | arXiv API; OpenAlex API | TorchSpatial: A Location Encoding Framework and Benchmark for Spatial Representation Learning \| arXiv preprint \| 2024 \| type posted-content |

## 5 · Provenance of every failure in this chapter

All six non-SUPPORTED sites are the English donor text of the published CoUrb 2026 article, matched
as exact string prefixes against `articles/CoUrb_2026/src_en/sections/related.tex` and
`.../metodology.tex`. The version of record is the Portuguese text; I spot-checked the PT source at
`articles/CoUrb_2026/src/sections/related.tex` for the POI2Vec and CATAPE sentences and the claims
map one to one ("adapta a arquitetura Word2Vec ... por meio de uma estrutura de arvore binaria
geografica"; "estende essa ideia incorporando informacoes categoricas e sequenciais"), so the
findings are properties of the published article and not of the translation.

One point worth the author's attention: the PT introduction at `articles/CoUrb_2026/src/sections/
intro.tex` says the two spatial encoders were "originalmente validadas em tarefas geoespaciais de
sensoriamento remoto e ecologia \cite{wu2024torchspatial}" — the same "originally validated"
attribution to TorchSpatial that Chapter 1 carries at `1_introduction.tex:50` and that is flagged
PARTIAL there. The two sites are the same claim and should take the same disposition.

## 6 · What I could not confirm in this chapter

- `:60`, `rahmani2019category`: Crossref returns a truncated abstract (four sentences) for this
  ICTIR paper. The categorical half of the citing sentence is supported; the sequential half is not
  visible in what the source of record returns. Open as `[VERIFY]` on that clause only.
- `:60`, `feng2017poi2vec`: the Word2Vec lineage and the geographic binary tree are POI2Vec's
  construction but are not in the abstract. The load-bearing half (geographic influence in the
  embedding) is verbatim.
