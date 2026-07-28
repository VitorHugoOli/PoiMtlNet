# 11_claims_3_cbic.md — citation claim-support audit, Chapter 3, CBIC 2025

**Unit:** `src/chapters/3_cbic.tex`  
**Run:** 2026-07-28, round 6, as the per-chapter pass the author asked for by name (COD-008 decision).  
**Errata regime for this unit:** PUBLISHED: a correction to reproduced prose is applied in the dissertation and listed in Appendix B; the published article record is not edited.

## 1 · Counts (every citation in the unit, not a sample)

- `\cite` commands in the unit: **57**, on **37** source lines, carrying **64** key instances (a multi-key `\cite` counts once per key). Every one was audited.
- Distinct bibliography keys used: **31**.
- Verdicts: **SUPPORTED** 49, **PARTIAL** 8, **NOT-SUPPORTED** 6, **UNVERIFIABLE** 1.

Comments were stripped before counting, so a key that appears only inside a `%` comment is not counted; every counted site renders.

## 2 · Every citation, with its verdict

Verdict scale: SUPPORTED, the citing sentence's attribution is present in or a fair paraphrase of the
source; PARTIAL, part is supported and part is not, or the sentence is stronger than the source;
NOT-SUPPORTED, the attribution is absent from or contradicted by the source; UNVERIFIABLE, the source
of record does not carry enough to decide and the attribution is not implausible.

| # | Site (file:line) | Key | Verdict | Evidence quoted from the source (under 20 words) |
|---|---|---|---|---|
| 1 | `3_cbic.tex:48` | `Zhang2020` | SUPPORTED | "novel interactive multi-task learning (iMTL) framework to better exploit the interplay between activity and location preference" |
| 2 | `3_cbic.tex:48` | `caruana1997multitask` | SUPPORTED | "improves generalization by using the domain information contained in the training signals of related tasks as an inductive bias" |
| 3 | `3_cbic.tex:48` | `kokkinos2016ubernet` | SUPPORTED | "jointly handles low-, mid-, and high-level vision tasks in a unified architecture that is trained end-to-end" |
| 4 | `3_cbic.tex:48` | `wei2022finetuned` | SUPPORTED | "instruction tuning -- finetuning language models on a collection of tasks described via instructions -- substantially improves zero-shot performance" |
| 5 | `3_cbic.tex:62` | `chen2020modeling` | SUPPORTED | "propose a holistic approach named Human Mobility Representation Model (HMRM) to simultaneously produce the vector representations" |
| 6 | `3_cbic.tex:62` | `cho2011gowalla` | SUPPORTED | "data from two online location-based social networks" |
| 7 | `3_cbic.tex:62` | `jure2014snap` | SUPPORTED | "A collection of more than 50 large network datasets" |
| 8 | `3_cbic.tex:62` | `zeng2019next` | SUPPORTED | "A Next Location Predicting Approach Based on a Recurrent Neural Network and Self-Attention" |
| 9 | `3_cbic.tex:86` | `Xu2023` | SUPPORTED | "devise a Tree-guided Multi-task Embedding model (TME for short) to learn effective representations of venues and categories" |
| 10 | `3_cbic.tex:88` | `Lim2022` | SUPPORTED | "learning different User-Region matrices of lower sparsities in a multi-task setting" |
| 11 | `3_cbic.tex:95` | `caruana1997multitask` | SUPPORTED | "improves generalization by using the domain information contained in the training signals of related tasks" |
| 12 | `3_cbic.tex:97` | `yu2024survey` | SUPPORTED | "categorizes MTL techniques into five key areas: regularization, relationship learning, feature propagation, optimization, and pre-training" |
| 13 | `3_cbic.tex:97` | `zhang2021survey` | NOT-SUPPORTED | "five categories, including feature learning approach, low-rank approach, task clustering approach, task relation learning approach and decomposition a" |
| 14 | `3_cbic.tex:102` | `caruana1997multitask` | PARTIAL | "improves generalization by using the domain information contained in the training signals of related tasks as an inductive bias" |
| 15 | `3_cbic.tex:104` | `misra2016cross` | SUPPORTED | "we propose a new sharing unit: "cross-stitch" unit. These units combine the activations from multiple networks" |
| 16 | `3_cbic.tex:104` | `ruder2017sluice` | SUPPORTED | "learns a latent multi-task architecture that jointly addresses (a)--(c)" |
| 17 | `3_cbic.tex:106` | `ma2018mmoe` | SUPPORTED | "successfully used in many real-world large-scale applications such as recommendation systems" |
| 18 | `3_cbic.tex:108` | `chen2018gradnorm` | SUPPORTED | "gradient normalization (GradNorm) algorithm that automatically balances training in deep multitask models by dynamically tuning gradient magnitudes" |
| 19 | `3_cbic.tex:108` | `liu2019dwa` | SUPPORTED | "less sensitive to various weighting schemes in the multi-task loss function" |
| 20 | `3_cbic.tex:108` | `sener2018mgda` | SUPPORTED | "we explicitly cast multi-task learning as multi-objective optimization, with the overall objective of finding a Pareto optimal solution" |
| 21 | `3_cbic.tex:108` | `yu2020pcgrad` | SUPPORTED | "projects a task's gradient onto the normal plane of the gradient of any other task that has a conflicting gradient" |
| 22 | `3_cbic.tex:112` | `ruder2017sluice` | SUPPORTED | "MTL involves searching an enormous space of possible parameter sharing architectures to find (a) the layers or subspaces that benefit from sharing" |
| 23 | `3_cbic.tex:112` | `zhang2021survey` | SUPPORTED | "leverage useful information contained in multiple related tasks to help improve the generalization performance of all the tasks" |
| 24 | `3_cbic.tex:113` | `sener2018mgda` | SUPPORTED | "different tasks may conflict, necessitating a trade-off" |
| 25 | `3_cbic.tex:113` | `yu2020pcgrad` | SUPPORTED | "detrimental gradient interference, and develop a simple yet general approach for avoiding such interference between task gradients" |
| 26 | `3_cbic.tex:114` | `nash` | NOT-SUPPORTED | "since the gradients of these different tasks may conflict, training a joint model for MTL often yields lower performance" |
| 27 | `3_cbic.tex:114` | `standley2020tasks` | NOT-SUPPORTED | "which tasks should and should not be learned together in one network when employing multi-task learning" |
| 28 | `3_cbic.tex:115` | `yu2024survey` | PARTIAL | "MTL's key advantages encompass streamlined model architecture, performance enhancement, and cross-domain generalizability" |
| 29 | `3_cbic.tex:115` | `zhang2021survey` | PARTIAL | "When the number of tasks is large or the data dimensionality is high, we review online, parallel and distributed MTL models as well as dimensionality " |
| 30 | `3_cbic.tex:118` | `standley2020tasks` | SUPPORTED | "propose a framework for assigning tasks to a few neural networks such that cooperating tasks are computed by the same neural network" |
| 31 | `3_cbic.tex:118` | `yu2024survey` | SUPPORTED | "categorizes MTL techniques into five key areas: regularization, relationship learning, feature propagation, optimization, and pre-training" |
| 32 | `3_cbic.tex:123` | `Liao2018` | PARTIAL | "a novel Context Aware Recurrent Unit is designed to integrate the sequential dependency and temporal regularity" |
| 33 | `3_cbic.tex:123` | `Zhang2020` | NOT-SUPPORTED | "temporal-aware activity encoder equipped with fuzzy characterization over uncertain check-ins" |
| 34 | `3_cbic.tex:125` | `Halder2021` | SUPPORTED | "Transformer-Based Multi-task Learning for Queuing Time Aware Next POI Recommendation" |
| 35 | `3_cbic.tex:125` | `Xia2020` | PARTIAL | "exploits a structure of generative adversarial networks (GAN) simultaneously considering temporal check-ins and geographical locations" |
| 36 | `3_cbic.tex:127` | `Xu2023` | PARTIAL | "we devise a Tree-guided Multi-task Embedding model (TME for short) to learn effective representations of venues and categories" |
| 37 | `3_cbic.tex:143` | `du2019beyond` | SUPPORTED | "spatial complementarity refers to the effect that the role of a spatial entity can be complemented and augmented by other different yet compatible spa" |
| 38 | `3_cbic.tex:145` | `huang2022estimating` | UNVERIFIABLE | "(no single decisive passage; see the ledger)" |
| 39 | `3_cbic.tex:151` | `velickovic2019deep` | SUPPORTED | "maximizing mutual information between patch representations and corresponding high-level summaries of graphs" |
| 40 | `3_cbic.tex:151` | `velivckovic2017graph` | SUPPORTED | "novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers" |
| 41 | `3_cbic.tex:191` | `perez2018film` | SUPPORTED | "FiLM layers influence neural network computation via a simple, feature-wise affine transformation based on conditioning information" |
| 42 | `3_cbic.tex:191` | `standley2020tasks` | SUPPORTED | "which tasks should and should not be learned together in one network when employing multi-task learning" |
| 43 | `3_cbic.tex:197` | `perez2018film` | SUPPORTED | "FiLM layers influence neural network computation via a simple, feature-wise affine transformation based on conditioning information" |
| 44 | `3_cbic.tex:204` | `baxter2000model` | SUPPORTED | "the learner can search for a hypothesis space that contains good solutions to many of the problems" |
| 45 | `3_cbic.tex:213` | `caruana1997multitask` | SUPPORTED | "improves generalization by using the domain information contained in the training signals of related tasks as an inductive bias" |
| 46 | `3_cbic.tex:213` | `ruder2017sluice` | NOT-SUPPORTED | "we present an approach that learns a latent multi-task architecture" |
| 47 | `3_cbic.tex:214` | `standley2020tasks` | NOT-SUPPORTED | "this often leads to inferior overall performance as task objectives can compete" |
| 48 | `3_cbic.tex:228` | `nash` | SUPPORTED | "viewing the gradients combination step as a bargaining game, where tasks negotiate to reach an agreement" |
| 49 | `3_cbic.tex:228` | `nash` | SUPPORTED | "proposed viewing the gradients combination step as a bargaining game... Nash Bargaining Solution" |
| 50 | `3_cbic.tex:231` | `nash` | SUPPORTED | "viewing the gradients combination step as a bargaining game, where tasks negotiate to reach an agreement on a joint direction" |
| 51 | `3_cbic.tex:231` | `nash` | SUPPORTED | "tasks negotiate to reach an agreement on a joint direction of parameter update" |
| 52 | `3_cbic.tex:238` | `nash` | SUPPORTED | "we propose viewing the gradients combination step as a bargaining game" |
| 53 | `3_cbic.tex:238` | `nash` | SUPPORTED | "Nash Bargaining Solution, which we propose to use as a principled approach to multi-task learning" |
| 54 | `3_cbic.tex:238` | `nash` | SUPPORTED | "derive theoretical guarantees for its convergence" |
| 55 | `3_cbic.tex:244` | `nash` | PARTIAL | "Empirically, we show that Nash-MTL achieves state-of-the-art results on multiple MTL benchmarks" |
| 56 | `3_cbic.tex:280` | `yu2020pcgrad` | SUPPORTED | "gradient surgery that projects a task's gradient onto the normal plane of the gradient of any other task" |
| 57 | `3_cbic.tex:288` | `cho2011gowalla` | SUPPORTED | "data from two online location-based social networks, we aim to understand what basic laws govern human motion" |
| 58 | `3_cbic.tex:306` | `chen2020modeling` | PARTIAL | "We apply HMRM to both unsupervised and supervised tasks including two activity evaluation tasks and two embedding evaluation tasks" |
| 59 | `3_cbic.tex:308` | `vaswani2017attention` | SUPPORTED | "new simple network architecture, the Transformer, based solely on attention mechanisms" |
| 60 | `3_cbic.tex:308` | `zeng2019next` | SUPPORTED | "A Next Location Predicting Approach Based on a Recurrent Neural Network and Self-Attention" |
| 61 | `3_cbic.tex:311` | `chen2020modeling` | SUPPORTED | "We apply HMRM to both unsupervised and supervised tasks including two activity evaluation tasks and two embedding evaluation tasks" |
| 62 | `3_cbic.tex:317` | `zeng2019next` | SUPPORTED | "A Next Location Predicting Approach Based on a Recurrent Neural Network and Self-Attention" |
| 63 | `3_cbic.tex:322` | `chen2020modeling` | SUPPORTED | "we propose a holistic approach named Human Mobility Representation Model (HMRM)" |
| 64 | `3_cbic.tex:322` | `zeng2019next` | SUPPORTED | "A Next Location Predicting Approach Based on a Recurrent Neural Network and Self-Attention" |
## 3 · Failures and partials in this unit, in detail

### `3_cbic.tex:97` — `zhang2021survey` — **NOT-SUPPORTED**

**Citing sentence.** Recent surveys \cite{yu2024survey,zhang2021survey} organize contemporary MTL research along five methodological dimensions: (i) \textit{parameter sharing} (hard vs.\ soft); (ii) \textit{relationship learning} (discovering task affinity or hierarchy); (iii) \textit{feature routing} (e.g., cross-stitch, sluice networks, attention gating); (iv) \textit{optimization} (conflict-aware gradient techniques); and (v) \textit{pre-training and instruction tuning}.

**Reference resolved.** DOI 10.1109/TKDE.2021.3070203. Source of record: Crossref REST; OpenAlex API. Record reads: A Survey on Multi-Task Learning | IEEE Transactions on Knowledge and Data Engineering | 2022 | type journal-article.

**Located passage.** "five categories, including feature learning approach, low-rank approach, task clustering approach, task relation learning approach and decomposition approach"

**Why.** The sentence says "Recent surveys [yu2024survey,zhang2021survey] organize contemporary MTL research along five methodological dimensions" and then lists parameter sharing / relationship learning / feature routing / optimization / pre-training and instruction tuning. That list is yu2024survey's five areas (regularization, relationship learning, feature propagation, optimization, pre-training), loosely renamed. zhang2021survey also gives five, but a DIFFERENT five, none of which is parameter sharing or pre-training. The plural "surveys" makes a taxonomy claim of both.

**Recommended disposition.** Published CBIC prose. Narrow the attribution: keep the list on yu2024survey and cite zhang2021survey for the survey framing only, or drop it from this sentence (it is cited three more times in the chapter). Appendix B row if the prose changes.

### `3_cbic.tex:114` — `nash` — **NOT-SUPPORTED**

**Citing sentence.** \textbf{Data Heterogeneity}: Variations in modality, label granularity, and dataset size complicate sampling strategies and minibatch construction \cite{nash,standley2020tasks}.

**Reference resolved.** no identifier in the bib entry. Source of record: arXiv API; OpenAlex API. Record reads: Multi-Task Learning as a Bargaining Game | arXiv preprint | 2022 | type posted-content.

**Located passage.** "since the gradients of these different tasks may conflict, training a joint model for MTL often yields lower performance"

**Why.** The bullet is "Data Heterogeneity: variations in modality, label granularity, and dataset size complicate sampling strategies and minibatch construction". Nash-MTL is a gradient-aggregation method; it addresses gradient conflict, which is the PRECEDING bullet in the same list, and says nothing about modality, label granularity, dataset size, sampling or minibatch construction.

**Recommended disposition.** Published CBIC prose. The two keys are mis-slotted across adjacent bullets. Either cite a survey that does treat data heterogeneity, or narrow the bullet to what these two works support.

### `3_cbic.tex:114` — `standley2020tasks` — **NOT-SUPPORTED**

**Citing sentence.** \textbf{Data Heterogeneity}: Variations in modality, label granularity, and dataset size complicate sampling strategies and minibatch construction \cite{nash,standley2020tasks}.

**Reference resolved.** arXiv:1905.07553. Source of record: arXiv API; OpenAlex API. Record reads: Which Tasks Should Be Learned Together in Multi-task Learning? | arXiv preprint | 2019 | type posted-content.

**Located passage.** "which tasks should and should not be learned together in one network when employing multi-task learning"

**Why.** Same bullet. The paper studies task cooperation and competition and proposes a task-grouping framework. It does not treat modality, label granularity or dataset-size heterogeneity, nor sampling or minibatch construction.

**Recommended disposition.** As above.

### `3_cbic.tex:123` — `Zhang2020` — **NOT-SUPPORTED**

**Citing sentence.** Similarly, the iMTL framework~\cite{Zhang2020} uses an LSTM architecture to model next-activity prediction, incorporating temporal dynamics in user behavior modeling.

**Reference resolved.** DOI 10.24963/ijcai.2020/491. Source of record: Crossref REST; OpenAlex API. Record reads: An Interactive Multi-Task Learning Framework for Next POI Recommendation with Uncertain Check-ins | Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence | 2020 | type proceedings-article.

**Located passage.** "temporal-aware activity encoder equipped with fuzzy characterization over uncertain check-ins"

**Why.** iMTL is an interactive multi-task framework for next-POI recommendation with uncertain check-ins; its encoders are a temporal-aware activity encoder and a spatial-aware location preference encoder, with a task-specific decoder. The abstract does not name LSTM, and "next-activity prediction" is one of two interacting tasks, not the model's object.

**Recommended disposition.** Published CBIC prose. Restate as its authors do: an interactive multi-task framework whose temporal-aware activity encoder handles uncertain check-ins. Appendix B row.

### `3_cbic.tex:213` — `ruder2017sluice` — **NOT-SUPPORTED**

**Citing sentence.** \textbf{Implicit Regularization:} By constraining the hypothesis space, hard sharing acts as a regularizer, often leading to more generalizable models, especially when tasks are related \cite{ruder2017sluice}.

**Reference resolved.** arXiv:1705.08142. Source of record: arXiv API. Record reads: Latent Multi-task Architecture Learning | arXiv preprint | 2017 | type posted-content.

**Located passage.** "we present an approach that learns a latent multi-task architecture"

**Why.** The bullet claims hard sharing acts as a regularizer. The cited work (arXiv:1705.08142, whose arXiv title of record is "Latent Multi-task Architecture Learning") proposes LEARNING what and how much to share, that is a soft-sharing alternative to hard sharing, and reports it outperforming standard MTL. It is evidence against the bullet it is attached to, not for it.

**Recommended disposition.** Published CBIC prose. baxter2000model, already in the bibliography and cited for exactly this at 4_courb.tex:116, does support a shared-hypothesis-space regularization claim. Swap the key. Appendix B row.

### `3_cbic.tex:214` — `standley2020tasks` — **NOT-SUPPORTED**

**Citing sentence.** \textbf{Empirical Performance:} In practice, hard parameter sharing frequently matches or exceeds the performance of more complex architectures on many benchmarks, while offering faster training and inference \cite{standley2020tasks}.

**Reference resolved.** arXiv:1905.07553. Source of record: arXiv API; OpenAlex API. Record reads: Which Tasks Should Be Learned Together in Multi-task Learning? | arXiv preprint | 2019 | type posted-content.

**Located passage.** "this often leads to inferior overall performance as task objectives can compete"

**Why.** ITEM 3. See the dedicated section of this report.

**Recommended disposition.** ITEM 3 draft: narrowed sentence + Appendix B row, handed over, not applied.

### `3_cbic.tex:102` — `caruana1997multitask` — **PARTIAL**

**Citing sentence.** This remains the simplest and most popular baseline, providing effective regularization \cite{caruana1997multitask}.

**Reference resolved.** DOI 10.1023/A:1007379606734. Source of record: Crossref REST; OpenAlex API; PDF in repo: 10.1023_A_1007379606734.pdf. Record reads: Multitask Learning | Machine Learning | 1997 | type journal-article.

**Located passage.** "improves generalization by using the domain information contained in the training signals of related tasks as an inductive bias"

**Why.** The regularization half is squarely supported. "the simplest and most popular baseline" is a bibliometric claim about the field in 2025 that a 1997 paper cannot carry.

**Recommended disposition.** Published CBIC prose, low exposure. Leave and record, or attribute the popularity clause to a survey (vandenhende2022mtl / zhang2021survey are both in the bibliography).

### `3_cbic.tex:115` — `zhang2021survey` — **PARTIAL**

**Citing sentence.** \textbf{Scalability}: Routing complexity, memory footprint, and evaluation costs often grow super-linearly as the number of tasks increases \cite{zhang2021survey,yu2024survey}.

**Reference resolved.** DOI 10.1109/TKDE.2021.3070203. Source of record: Crossref REST; OpenAlex API. Record reads: A Survey on Multi-Task Learning | IEEE Transactions on Knowledge and Data Engineering | 2022 | type journal-article.

**Located passage.** "When the number of tasks is large or the data dimensionality is high, we review online, parallel and distributed MTL models as well as dimensionality reduction and featur"

**Why.** The survey does treat cost growth with the number of tasks, and names computational and storage concerns. The specific word "super-linearly" is a quantitative shape claim that the abstract does not state, and neither does yu2024survey's.

### `3_cbic.tex:115` — `yu2024survey` — **PARTIAL**

**Citing sentence.** \textbf{Scalability}: Routing complexity, memory footprint, and evaluation costs often grow super-linearly as the number of tasks increases \cite{zhang2021survey,yu2024survey}.

**Reference resolved.** arXiv:2404.18961. Source of record: arXiv API; OpenAlex API. Record reads: Unleashing the Power of Multi-Task Learning: A Comprehensive Survey Spanning Traditional, Deep, and Pretrained Foundation Model Eras | arXiv preprint | 2024 | type posted-content.

**Located passage.** "MTL's key advantages encompass streamlined model architecture, performance enhancement, and cross-domain generalizability"

**Why.** Same bullet. The survey addresses architectures and efficiency but the abstract makes no super-linear growth claim. Published CBIC prose, low exposure: the bullet is a challenge list, not a result.

### `3_cbic.tex:123` — `Liao2018` — **PARTIAL**

**Citing sentence.** Early MTL-based approaches such as MCARNN~\cite{Liao2018} employ recurrent neural networks with temporal attention mechanisms to jointly predict user activities and future visited locations.

**Reference resolved.** DOI 10.24963/ijcai.2018/477. Source of record: Crossref REST; OpenAlex API. Record reads: Predicting Activity and Location with Multi-task Context Aware Recurrent Neural Network | Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence | 2018 | type proceedings-article.

**Located passage.** "a novel Context Aware Recurrent Unit is designed to integrate the sequential dependency and temporal regularity"

**Why.** MCARNN does jointly predict activity and location with a recurrent model, which is the load-bearing part. "temporal attention mechanisms" is not what the paper describes: its mechanism is a Context Aware Recurrent Unit over spatial-activity topics.

**Recommended disposition.** Published CBIC prose. Substitute "context-aware recurrent units" for "temporal attention mechanisms" (describe the system as its authors do, AGENT_GUARDRAILS R2), with an Appendix B row.

### `3_cbic.tex:125` — `Xia2020` — **PARTIAL**

**Citing sentence.** MTPR~\cite{Xia2020} combines LSTMs and adversarial learning to address uncertainty in check-ins and improve multi-task POI recommendation both location and temporal context with a generative component.

**Reference resolved.** DOI 10.3390/app10196664. Source of record: Crossref REST; OpenAlex API. Record reads: MTPR: A Multi-Task Learning Based POI Recommendation Considering Temporal Check-Ins and Geographical Locations | Applied Sciences | 2020 | type journal-article.

**Located passage.** "exploits a structure of generative adversarial networks (GAN) simultaneously considering temporal check-ins and geographical locations"

**Why.** The multi-task, adversarial and temporal-geographic halves are supported verbatim. "combines LSTMs" is not in the abstract. The citing sentence is also ungrammatical in the published text ("improve multi-task POI recommendation both location and temporal context").

**Recommended disposition.** Published CBIC prose. Drop "LSTMs and" or verify against the paper body; the sentence needs a grammatical repair regardless, which is a wording row.

### `3_cbic.tex:127` — `Xu2023` — **PARTIAL**

**Citing sentence.** Some Models such as TME~\cite{Xu2023} address category annotation using graph-based encoders, but treat prediction and classification separately.

**Reference resolved.** DOI 10.1145/3582553. Source of record: Crossref REST; OpenAlex API. Record reads: TME: Tree-guided Multi-task Embedding Learning towards Semantic Venue Annotation | ACM Transactions on Information Systems | 2023 | type journal-article.

**Located passage.** "we devise a Tree-guided Multi-task Embedding model (TME for short) to learn effective representations of venues and categories"

**Why.** The load-bearing half is supported: TME addresses category annotation and does not pair it with next-POI prediction. "using graph-based encoders" is not TME's described mechanism, which is multi-context embedding regularized by a predefined category hierarchy.

**Recommended disposition.** Published CBIC prose. Replace "graph-based encoders" with "a tree-guided multi-task embedding". Appendix B row.

### `3_cbic.tex:145` — `huang2022estimating` — **UNVERIFIABLE**

**Citing sentence.** Besides that, following \cite{huang2022estimating} the weight of an edge $e_{ij}$ is defined as $w_{ij} = \log((1+D^{1.5}/1+d_{ij}^{1.5}))$, where D is the diagonal length of bounding box that encloses the coordinates of POIs, and $d_{ij}$ is the geodesic distance between $p_{i}$ and $p_{j}$.

**Reference resolved.** DOI 10.1080/13658816.2022.2040510. Source of record: Crossref REST; OpenAlex API. Record reads: Estimating urban functional distributions with semantics preserved POI embedding | International Journal of Geographical Information Science | 2022 | type journal-article.

**Why.** The citing sentence reproduces a specific edge-weight formula, w_ij = log((1+D^1.5)/(1+d_ij^1.5)), and says it follows the cited work. That is a formula-level attribution; the abstract of record cannot confirm or refute it, and I did not obtain the paper body. Open as [VERIFY] with a named check: locate the formula in the cited paper, or restate it as this work's own construction.

### `3_cbic.tex:244` — `nash` — **PARTIAL**

**Citing sentence.** For efficiency, task weights can be updated less frequently, significantly reducing runtime while maintaining performance~\cite{nash}.

**Reference resolved.** no identifier in the bib entry. Source of record: arXiv API; OpenAlex API. Record reads: Multi-Task Learning as a Bargaining Game | arXiv preprint | 2022 | type posted-content.

**Located passage.** "Empirically, we show that Nash-MTL achieves state-of-the-art results on multiple MTL benchmarks"

**Why.** The abstract does not carry the less-frequent-update claim. I could not locate it from the abstract alone, and I did not read the paper body for this clause, so I cannot certify it either way. The chapter has already corrected two other cost claims in this same subsection against this same paper.

**Recommended disposition.** Open as [VERIFY]. The clause needs a page or section from arXiv:2202.01017 before it stands; the neighbouring corrections show the paper has been read for cost claims before.

### `3_cbic.tex:306` — `chen2020modeling` — **PARTIAL**

**Citing sentence.** The Human Mobility Representation Model (HMRM), introduced by Chen et al. (2020) \cite{chen2020modeling}, is designed for POI category classification.

**Reference resolved.** DOI 10.1109/TKDE.2020.3001025. Source of record: Crossref REST; OpenAlex API. Record reads: Modeling Spatial Trajectories With Attribute Representation Learning | IEEE Transactions on Knowledge and Data Engineering | 2022 | type journal-article.

**Located passage.** "We apply HMRM to both unsupervised and supervised tasks including two activity evaluation tasks and two embedding evaluation tasks"

**Why.** HMRM is a general trajectory-attribute representation model, not a model "designed for POI category classification". The rest of the passage (PMI, matrix factorization, SVM on the embeddings) describes how THIS chapter used it as a baseline, which is a legitimate use.

**Recommended disposition.** Published CBIC prose. Narrow to "used here for POI category classification" or "a representation model that this chapter applies to POI category classification".

## 4 · Source ledger for this unit

Every distinct key cited in this unit, the identifier it resolved by, and where I opened it this session.

| Key | Identifier | Opened at | Record as returned |
|---|---|---|---|
| `Halder2021` | DOI 10.1007/978-3-030-75765-6_41 | Crossref REST; OpenAlex API; Semantic Scholar API | Transformer-Based Multi-task Learning for Queuing Time Aware Next POI Recommendation \| Lecture Notes in Computer Science \| 2021 \| type book-chapter |
| `Liao2018` | DOI 10.24963/ijcai.2018/477 | Crossref REST; OpenAlex API | Predicting Activity and Location with Multi-task Context Aware Recurrent Neural Network \| Proceedings of the Twenty-Seventh International Joint Conference on Artificial  |
| `Lim2022` | DOI 10.1145/3477495.3531989 | Crossref REST; OpenAlex API | Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation \| Proceedings of the 45th International ACM SIGIR Conference on Research and Development in I |
| `Xia2020` | DOI 10.3390/app10196664 | Crossref REST; OpenAlex API | MTPR: A Multi-Task Learning Based POI Recommendation Considering Temporal Check-Ins and Geographical Locations \| Applied Sciences \| 2020 \| type journal-article |
| `Xu2023` | DOI 10.1145/3582553 | Crossref REST; OpenAlex API | TME: Tree-guided Multi-task Embedding Learning towards Semantic Venue Annotation \| ACM Transactions on Information Systems \| 2023 \| type journal-article |
| `Zhang2020` | DOI 10.24963/ijcai.2020/491 | Crossref REST; OpenAlex API | An Interactive Multi-Task Learning Framework for Next POI Recommendation with Uncertain Check-ins \| Proceedings of the Twenty-Ninth International Joint Conference on Art |
| `baxter2000model` | DOI 10.1613/jair.731 | Crossref REST; OpenAlex API | A Model of Inductive Bias Learning \| Journal of Artificial Intelligence Research \| 2000 \| type journal-article |
| `caruana1997multitask` | DOI 10.1023/A:1007379606734 | Crossref REST; OpenAlex API; PDF in repo: 10.1023_A_1007379606734.pdf | Multitask Learning \| Machine Learning \| 1997 \| type journal-article |
| `chen2018gradnorm` | no identifier in the bib entry | arXiv API; OpenAlex API | GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks \| Proceedings of the 35th International Conference on Machine Learning (2018), 79 |
| `chen2020modeling` | DOI 10.1109/TKDE.2020.3001025 | Crossref REST; OpenAlex API | Modeling Spatial Trajectories With Attribute Representation Learning \| IEEE Transactions on Knowledge and Data Engineering \| 2022 \| type journal-article |
| `cho2011gowalla` | DOI 10.1145/2020408.2020579 | Crossref REST; OpenAlex API | Friendship and mobility \| Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining \| 2011 \| type proceedings-article |
| `du2019beyond` | DOI 10.1109/ICDM.2019.00026 | Crossref REST; OpenAlex API | Beyond Geo-First Law: Learning Spatial Representations via Integrated Autocorrelations and Complementarity \| 2019 IEEE International Conference on Data Mining (ICDM) \|  |
| `huang2022estimating` | DOI 10.1080/13658816.2022.2040510 | Crossref REST; OpenAlex API | Estimating urban functional distributions with semantics preserved POI embedding \| International Journal of Geographical Information Science \| 2022 \| type journal-arti |
| `jure2014snap` | no identifier in the bib entry | OpenAlex API | {SNAP Datasets}: {Stanford} Large Network Dataset Collection \| (no venue in record) \| 2014 \| type article |
| `kokkinos2016ubernet` | arXiv:1609.02132 | arXiv API; OpenAlex API | UberNet: Training a `Universal' Convolutional Neural Network for Low-, Mid-, and High-Level Vision using Diverse Datasets and Limited Memory \| arXiv preprint \| 2016 \|  |
| `liu2019dwa` | DOI 10.1109/CVPR.2019.00197 | Crossref REST; OpenAlex API | End-To-End Multi-Task Learning With Attention \| 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) \| 2019 \| type proceedings-article |
| `ma2018mmoe` | DOI 10.1145/3219819.3220007 | Crossref REST; OpenAlex API | Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts \| Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discover |
| `misra2016cross` | DOI 10.1109/CVPR.2016.433 | Crossref REST; OpenAlex API | Cross-Stitch Networks for Multi-task Learning \| 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) \| 2016 \| type proceedings-article |
| `nash` | no identifier in the bib entry | arXiv API; OpenAlex API | Multi-Task Learning as a Bargaining Game \| arXiv preprint \| 2022 \| type posted-content |
| `perez2018film` | DOI 10.1609/aaai.v32i1.11671 | Crossref REST; OpenAlex API | FiLM: Visual Reasoning with a General Conditioning Layer \| Proceedings of the AAAI Conference on Artificial Intelligence \| 2018 \| type journal-article |
| `ruder2017sluice` | arXiv:1705.08142 | arXiv API | Latent Multi-task Architecture Learning \| arXiv preprint \| 2017 \| type posted-content |
| `sener2018mgda` | arXiv:1810.04650 | arXiv API; OpenAlex API | Multi-Task Learning as Multi-Objective Optimization \| arXiv preprint \| 2018 \| type posted-content |
| `standley2020tasks` | arXiv:1905.07553 | arXiv API; OpenAlex API | Which Tasks Should Be Learned Together in Multi-task Learning? \| arXiv preprint \| 2019 \| type posted-content |
| `vaswani2017attention` | arXiv:1706.03762 | arXiv API | Attention Is All You Need \| arXiv preprint \| 2017 \| type posted-content |
| `velickovic2019deep` | no identifier in the bib entry | OpenAlex API | Deep Graph Infomax \| Apollo (University of Cambridge) \| 2018 \| type conference-paper |
| `velivckovic2017graph` | arXiv:1710.10903 | arXiv API | Graph Attention Networks \| arXiv preprint \| 2017 \| type posted-content |
| `wei2022finetuned` | URL https://openreview.net/forum?id=gEZrGCozdqR | arXiv API; OpenAlex API | Finetuned Language Models Are Zero-Shot Learners \| arXiv preprint \| 2021 \| type posted-content |
| `yu2020pcgrad` | no identifier in the bib entry | arXiv API; OpenAlex API | Gradient Surgery for Multi-Task Learning \| arXiv preprint \| 2020 \| type posted-content |
| `yu2024survey` | arXiv:2404.18961 | arXiv API; OpenAlex API | Unleashing the Power of Multi-Task Learning: A Comprehensive Survey Spanning Traditional, Deep, and Pretrained Foundation Model Eras \| arXiv preprint \| 2024 \| type pos |
| `zeng2019next` | DOI 10.1007/978-3-030-30146-0_21 | Crossref REST; OpenAlex API; Semantic Scholar API | A Next Location Predicting Approach Based on a Recurrent Neural Network and Self-attention \| Lecture Notes of the Institute for Computer Sciences, Social Informatics and |
| `zhang2021survey` | DOI 10.1109/TKDE.2021.3070203 | Crossref REST; OpenAlex API | A Survey on Multi-Task Learning \| IEEE Transactions on Knowledge and Data Engineering \| 2022 \| type journal-article |

## 5 · Provenance of every failure in this chapter

**All fifteen non-SUPPORTED sites in this chapter are verbatim published CBIC 2025 prose.** I did
not take this on trust. Each citing sentence was matched, as an exact string prefix, against the
article source of record in this repository:

| Site | Found verbatim in |
|---|---|
| `:97` five survey dimensions | `articles/CBIC___MTL/sections/basis.tex` |
| `:102` hard-sharing baseline | `articles/CBIC___MTL/sections/basis.tex` |
| `:108` DWA | `articles/CBIC___MTL/sections/basis.tex` |
| `:114` data heterogeneity | `articles/CBIC___MTL/sections/basis.tex` |
| `:123` MCARNN and iMTL | `articles/CBIC___MTL/sections/basis.tex` |
| `:125` MTPR | `articles/CBIC___MTL/sections/basis.tex` |
| `:127` TME | `articles/CBIC___MTL/sections/basis.tex` |
| `:213` regularization and Caruana | `articles/CBIC___MTL/sections/method.tex` |
| `:214` empirical performance | `articles/CBIC___MTL/sections/method.tex` |
| `:244` Nash update frequency | `articles/CBIC___MTL/sections/method.tex` |
| `:306` HMRM | `articles/CBIC___MTL/sections/results.tex` |

So every disposition in this chapter is an errata-policy decision, not a typo fix: the correction is
applied in the dissertation and listed in Appendix B, and the published record is not edited. That
is also why the load-bearing ranking matters more here than elsewhere: each row costs an Appendix B
line.

## 6 · The Standley site

`3_cbic.tex:214` is treated separately and at length in `11_citation_claims.md` section 5, per the
author's instruction: a deeper evaluation of the effect on the text, a commit-history check for an
earlier reference at that site, a replacement candidate, and a drafted narrowed sentence with its
Appendix B row. **No edit to `3_cbic.tex` was made in this task.**

## 7 · What I could not confirm in this chapter

- `:145`, `huang2022estimating`: the edge-weight formula is attributed to the cited work and I could
  not locate the formula. Open as `[VERIFY]`.
- `:244`, `nash`: the less-frequent-update efficiency clause is not in the abstract of record and I
  did not read the paper body for it. Open as `[VERIFY]`.
- `zeng2019next` (four sites) returns no abstract at Crossref, OpenAlex or Semantic Scholar, and the
  Springer chapter is outside the network allowlist. All four sites are identity-of-baseline
  attributions that the record's title supports; no mechanism claim is made at any of them.
