# Literature Review — LBSN Representation Learning, Next-POI / Next-Category / Next-Region Prediction, and MTL for Mobility

> **Status**: compiled 2026-06-12 from a targeted web survey (ACM, IEEE, Springer, arXiv, OpenReview, official proceedings) run for the critical research assessment of this repository. Companion documents: [`project_positioning.md`](project_positioning.md), [`baseline_gap_analysis.md`](baseline_gap_analysis.md), [`evaluation_protocol_review.md`](evaluation_protocol_review.md). All references consolidated in [`references.md`](references.md).
>
> **Terminology used throughout**: *next-POI prediction* = predict the venue ID of the next check-in. *Next-category prediction* = predict the category of the next check-in. *Next-region prediction* = predict the spatial cell (geohash / grid / census tract) of the next check-in. *Embedding (substrate) contribution* ≠ *architecture contribution*; *empirical improvement* ≠ *conceptual novelty*.

---

## 1. Next-POI recommendation — method families

### 1.1 RNN era (2016–2020)

| Model | Citation | Key idea | Datasets | Metrics |
|---|---|---|---|---|
| ST-RNN | Liu et al., AAAI 2016 ([link](https://ojs.aaai.org/index.php/AAAI/article/view/9971)) | RNN with time-interval- and distance-specific transition matrices | Gowalla, GTD | Recall@k, F1, MAP |
| DeepMove | Feng et al., WWW 2018 ([link](https://dl.acm.org/doi/10.1145/3178876.3186058)) | Multi-modal embedding GRU + historical attention | Foursquare-NY, CDR | top-1 Acc |
| Flashback | Yang et al., IJCAI 2020 ([link](https://www.ijcai.org/proceedings/2020/302)) | Spatiotemporal-weighted aggregation of past RNN hidden states (sparse traces) | Gowalla, Foursquare global | Acc@5/10, MRR |
| LSTPM | Sun et al., AAAI 2020 ([link](https://ojs.aaai.org/index.php/AAAI/article/view/5353)) | Non-local long-term + geo-dilated short-term RNN | FSQ-NYC, Gowalla | Recall@k, NDCG@k |

### 1.2 Attention / transformer era (2020–2024)

| Model | Citation | Key idea | Notes |
|---|---|---|---|
| GeoSAN | KDD 2020 (title to be verified on DBLP) | Self-attention + hierarchical gridding/quadkey geography encoder | Sampled-negative evaluation, now deprecated |
| STAN | Luo et al., WWW 2021 ([link](https://dl.acm.org/doi/10.1145/3442381.3449998)) | Bi-layer attention with explicit spatio-temporal interval matrices | **Reimplemented faithfully in this repo** (`src/models/next/next_stan/`) |
| GETNext | Yang et al., SIGIR 2022 ([link](https://dl.acm.org/doi/10.1145/3477495.3531983)) | Trajectory-flow-map GCN prior + transformer; **defined the standard NYC/TKY/CA benchmark**; auxiliary time + category losses (L = L_poi + 10·L_time + L_cat) | NYC Acc@1 0.244 / Acc@10 0.614. The repo's `next_stan_flow` head borrows its α·log_T prior pattern (loosely, not a reproduction) |
| ROTAN | Feng et al., KDD 2024 ([link](https://dl.acm.org/doi/10.1145/3637528.3671809)) | Time2Rotation; injects target time | NYC Acc@1 0.311 — strongest non-LLM model on the trio |
| CLSPRec | CIKM 2023 ([link](https://dl.acm.org/doi/10.1145/3583780.3614813)); CLLP, SIGIR 2024 ([link](https://dl.acm.org/doi/10.1145/3626772.3657730)) | Contrastive long/short-term preference learning | — |

### 1.3 Graph-based (2022–2025)

| Model | Citation | Key idea |
|---|---|---|
| Graph-Flashback | Rao et al., KDD 2022 ([link](https://dl.acm.org/doi/10.1145/3534678.3539383)) | ST knowledge graph → learned POI transition graph refines Flashback hidden states |
| HMT-GRN | Lim et al., SIGIR 2022 ([link](https://dl.acm.org/doi/10.1145/3477495.3531989)); TORS 2023 extension ([link](https://dl.acm.org/doi/10.1145/3610584)) | **Multi-task next-POI + next-region at geohash precisions G@2…G@6**; hierarchical beam search; equal-weight CE. The canonical next-region precedent — see §4 |
| DRAN | SIGIR 2022 ([link](https://dl.acm.org/doi/10.1145/3477495.3532012)) | Disentangled GCN (transition vs distance factors) |
| SNPM | AAAI 2023 ([link](https://ojs.aaai.org/index.php/AAAI/article/view/25608)) | Dynamic neighbor graph (RotatE + Eigenmap) |
| AGRAN | SIGIR 2023 ([link](https://dl.acm.org/doi/10.1145/3539618.3591634)) | Learns the POI graph structure instead of using a static graph |
| 2024–25 frontier | Bi-level graph structure learning ([arXiv:2411.01169](https://arxiv.org/abs/2411.01169)); context-adaptive GNN ([arXiv:2506.10329](https://arxiv.org/abs/2506.10329)); disentangled graph debiasing, SIGIR 2025 ([link](https://dl.acm.org/doi/10.1145/3726302.3729952)) | Graph-structure learning + debiasing |

### 1.4 Hypergraph

| Model | Citation | Key idea |
|---|---|---|
| STHGCN | Yan et al., SIGIR 2023 ([link](https://dl.acm.org/doi/10.1145/3539618.3591770)) | Trajectories as hyperedges; hypergraph transformer; NYC Acc@1 0.273 |
| DCHL | SIGIR 2024 ([link](https://dl.acm.org/doi/10.1145/3626772.3657726)) | Multi-view hypergraphs + cross-view contrastive |
| MSHL | AAAI 2026 ([link](https://ojs.aaai.org/index.php/AAAI/article/view/38552/42514)) | Scenario-aware hyperedges |
| ReHDM | Li et al., IJCAI 2025 | Region-aware dual-level hypergraph. **Reimplemented faithfully in this repo** (`docs/baselines/next_region/comparison.md`) |

### 1.5 Diffusion / LLM era (2023–2026)

| Model | Citation | Note |
|---|---|---|
| Diff-POI | TOIS 2023 ([link](https://dl.acm.org/doi/10.1145/3624475)) | Diffusion sampling of spatial preference |
| LLM4POI | SIGIR 2024 ([link](https://dl.acm.org/doi/10.1145/3626772.3657840)) | LoRA-SFT Llama-2-7B; NYC Acc@1 0.337 |
| AgentMove | NAACL 2025 ([link](https://aclanthology.org/2025.naacl-long.61/)) | Agentic zero-shot decomposition |
| GNPR-SID | KDD 2025 ([link](https://dl.acm.org/doi/10.1145/3711896.3736981)) | Semantic-ID generative rec; NYC Acc@1 0.362 — current published SOTA on the trio |
| Refine-POI | 2025 ([arXiv:2506.21599](https://arxiv.org/abs/2506.21599)) | RFT LLM; NYC Acc@1 0.347 |

**SOTA trajectory on FSQ-NYC Acc@1**: GETNext 0.244 (2022) → STHGCN 0.273 (2023) → ROTAN 0.311 (2024) → LLM4POI 0.337 (2024) → GNPR-SID 0.362 (2025). The frontier has moved to fine-tuned/generative LLMs and semantic-ID models.

---

## 2. The standard evaluation protocol in this literature

This section is the yardstick against which the repo's protocol is judged in [`evaluation_protocol_review.md`](evaluation_protocol_review.md).

- **Datasets**: the canonical trio since GETNext — **Foursquare-NYC** (~1.1k users / ~5k POIs / ~100k check-ins), **Foursquare-TKY** (~2.3k / ~7.8k / ~361k), **Gowalla-CA** (~4k / ~9.7k / ~250k). Older strand: global Gowalla/Foursquare snapshots (Flashback, HMT-GRN), Brightkite, Weeplaces. New benchmark: **Massive-STEPS** (2025, [arXiv:2505.11239](https://arxiv.org/abs/2505.11239)), 12 cities, explicitly motivated by over-reliance on 2012–13 NYC/TKY data.
- **Filtering**: drop users and POIs with <10 check-ins; remove test users/POIs unseen in training; 24-hour trajectory windows; discard length-1 trajectories.
- **Splits — always temporal, never random**: per-user chronological 80/10/10 (GETNext lineage), per-user first-80%/last-20% (Flashback/HMT-GRN lineage), or leave-last-out (STAN). **No paper in this canon uses k-fold random CV.**
- **Metrics**: Acc@k (k ∈ {1,5,10,20}) + MRR over the full POI vocabulary; sampled-negative ranking is deprecated; macro-F1 is essentially absent.
- **Methodological critique to cite**: Luca et al., "Trajectory test-train overlap in next-location prediction datasets", *Machine Learning* 2023 ([link](https://link.springer.com/article/10.1007/s10994-023-06386-x)) — overlapping train/test trajectories inflate accuracy.

---

## 3. Next-category / activity prediction

Category-as-headline is rare; the lineage is:

- **Ye, Zhu & Cheng, SDM 2013** ([PDF](https://www1.se.cuhk.edu.hk/~hcheng/paper/sdm2013.pdf)) — mixed HMM predicts next activity category, then location given category. The founding precedent for category-first prediction.
- **LBPR**, He et al., IJCAI 2017 ([link](https://www.ijcai.org/proceedings/2017/255)) — listwise BPR over next-category, then category-filtered POI ranking.
- **MCARNN**, Liao et al., IJCAI 2018 ([link](https://www.ijcai.org/proceedings/2018/477)) — **both next activity and next location are headline tasks**; shared context-aware recurrent unit + task-specific GRUs. The closest pre-2020 "shared backbone + two heads" precedent.
- **CatDM**, WWW 2020 ([link](https://dl.acm.org/doi/10.1145/3366423.3380202)) — category predicted first as a candidate-filtering stage (cascade, not parallel MTL).
- **iMTL**, Zhang et al., IJCAI 2020 ([link](https://www.ijcai.org/proceedings/2020/491)) — activity prediction is co-headline (first results table); see §4.
- **CHA**, ACM TOIT 2021 ([link](https://dl.acm.org/doi/fullHtml/10.1145/3464300)); category-aware GRU, IJIS 2021 ([link](https://onlinelibrary.wiley.com/doi/abs/10.1002/int.22412)).
- Transportation strand (purpose ≈ category, no spatial IDs): mode+purpose MTL, *Travel Behaviour and Society* 2024 ([link](https://www.sciencedirect.com/science/article/pii/S2214367X23000765)); *Mathematics* 13:1528, 2025 ([link](https://doi.org/10.3390/math13091528)).

**Norms**: Foursquare city datasets with 184–400 categories; metrics are Acc@K / Rec@K / MAP@K / MRR as ranking. **Macro-F1 over a merged 7-class taxonomy (this repo) has no published comparison point** — defensible for class imbalance, but non-comparable.

---

## 4. Multi-task learning for mobility — the closest competitors

These are the papers this project must position against. Detailed mechanics (verified from full texts):

### iMTL (IJCAI 2020) — [link](https://www.ijcai.org/proceedings/2020/491), [code](https://github.com/iMTL2020/iMTL)
3 tasks: next activity/category (fuzzy over uncertain check-ins), auxiliary POI-type (binary), next POI. Two-channel LSTM encoder + *interactive* decoder cascade (location decoder consumes predicted activity). Loss: tuned static scalarization (λ = 0.4/0.3/0.3), BPR losses. Datasets: Foursquare Charlotte/Calgary/Phoenix (184–251 categories), 80/10/10. Ablations show MTL + interaction beats parallel STL.

### HMT-GRN (SIGIR 2022) — [link](https://dl.acm.org/doi/10.1145/3477495.3531989), [code](https://github.com/poi-rec/HMT-GRN)
**The canonical next-region precedent.** 6 tasks: next-POI (explicit "main task") + next-region at geohash precisions G@2…G@6 (1,251 km → 0.61 km cells). One shared embedding layer + shared LSTM/GRN hidden state; per-task softmax heads; **equal-weight CE** ("to not bias to any task"). Regions exist to power hierarchical beam search and to relieve sparsity (user-POI matrix 99.8% sparse vs 97–98% at G@2). Region accuracy is shown ≫ POI accuracy but **region performance is never a deliverable** — regions are auxiliary tools. Global Gowalla + Foursquare, per-user first-80%/last-20% split, Acc@{1,5,10,20} + MRR.

### CSLSL (arXiv 2022 → EPJ Data Science 2024) — [link](https://link.springer.com/article/10.1140/epjds/s13688-024-00460-7), [code](https://github.com/urbanmobility/CSLSL)
Cascaded time → activity(category) → location ("when→what→where" causal chain) + spatial-consistency auxiliary loss. Location headline; category instrumental. FSQ NYC/TKY + Gowalla Dallas, Recall@{1,5,10}.

### Other relevant MTL works
- **MobTCast** (NeurIPS 2021, [link](https://arxiv.org/abs/2110.01401)) — auxiliary coordinate-forecasting branch + consistency loss.
- **MCLP** (KDD 2024, [link](https://dl.acm.org/doi/10.1145/3637528.3671916)) — topic-model category preference as context, not a supervised co-task.
- **HAMTL** (*J. Supercomputing* 2025, [link](https://link.springer.com/article/10.1007/s11227-025-07643-7)) — jointly predicts next location + its category with a hierarchy-aware decoder. Closest recent two-headline-task paper; spatial task is location-level, not region.
- **MMPAN** (ESWA 2024, [link](https://www.sciencedirect.com/science/article/abs/pii/S0957417424030562)) — POI + grid-cell + category all supervised, POI main.
- **KGTB** (arXiv 2025, [link](https://arxiv.org/abs/2509.12350)) — LLM generative next-POI with **category + region as auxiliary behaviors** ("users first pick category by intent, then region by geography"). **The exact category+region pairing — but explicitly subordinate to next-POI.** The single most important paper to cite and distinguish.
- **DRRGNN** (TKDD 2022, [link](https://dl.acm.org/doi/10.1145/3529091)) — next activity-*region* prediction as headline (data-driven regions), no category co-task.
- **Where and When** (IJCAI 2025, [PDF](https://www.ijcai.org/proceedings/2025/0390.pdf)) — the 2025 MTL frontier pairs POI with *time*, not category/region.

**Pattern across all competitors**: hard parameter sharing or cascades; uniform/static loss weights (nobody uses GradNorm/PCGrad/Nash-MTL-class optimizers); MTL beats STL ablations *when the auxiliary task is a denser/easier signal serving a sparse main task*.

**Key negative search result**: after targeted searching, **no published work was found where next-category + next-region (without next-POI) is the headline MTL formulation**. The pairing exists only as auxiliaries to next-POI (KGTB, HMT-GRN, MMPAN). See [`project_positioning.md §3`](project_positioning.md) for what this does and does not buy.

---

## 5. General MTL architectures and optimization

Architectures: shared-bottom (Caruana 1997, [link](https://link.springer.com/article/10.1023/A:1007379606734)); Cross-stitch (CVPR 2016, [link](https://arxiv.org/abs/1604.03539)); MMoE (KDD 2018, [link](https://dl.acm.org/doi/10.1145/3219819.3220007)); PLE/CGC (RecSys 2020 best paper, [link](https://dl.acm.org/doi/10.1145/3383313.3412236)); DSelect-k (NeurIPS 2021, [link](https://arxiv.org/abs/2106.03760)); MTAN cross-task attention + DWA (CVPR 2019, [link](https://arxiv.org/abs/1803.10704)); FiLM conditioning (AAAI 2018, [link](https://arxiv.org/abs/1709.07871)); MulT cross-modal transformer (ACL 2019, [arXiv:1906.00295](https://arxiv.org/abs/1906.00295)) — the repo's `mtlnet_crossattn` is MulT-adapted.

Optimization: uncertainty weighting (CVPR 2018, [link](https://arxiv.org/abs/1705.07115)); GradNorm (ICML 2018, [link](https://arxiv.org/abs/1711.02257)); PCGrad (NeurIPS 2020, [link](https://arxiv.org/abs/2001.06782)); CAGrad (NeurIPS 2021, [link](https://arxiv.org/abs/2110.14048)); Nash-MTL (ICML 2022, [link](https://arxiv.org/abs/2202.01017)); Aligned-MTL (CVPR 2023, [link](https://arxiv.org/abs/2305.19000)).

**Directly load-bearing for this project**:
- Xin et al., "Do Current Multi-Task Optimization Methods in Deep Learning Even Help?", NeurIPS 2022 ([link](https://arxiv.org/abs/2209.11379)) and Kurin et al., "In Defense of the Unitary Scalarization", NeurIPS 2022 ([link](https://arxiv.org/abs/2201.04122)) — both find specialized MTL optimizers give no gains over well-tuned scalarization. **The repo's empirical trajectory (NashMTL/PCGrad/19-arm registry → static_weight champion; pooled gradient cosine ≈ 0) replicates this finding in a new domain and should cite both.**
- Standley et al., "Which Tasks Should Be Learned Together?", ICML 2020 ([link](https://arxiv.org/abs/1905.07553)) — task-grouping analysis.

---

## 6. Pre-trained location / check-in embeddings — the direct novelty competitors for Check2HGI

| Method | Unit | Contextual? | Mechanism | Objective | Venue |
|---|---|---|---|---|---|
| POI2Vec ([AAAI 2017](https://ojs.aaai.org/index.php/AAAI/article/view/10500)) | POI | static | word2vec + binary spatial tree | skip-gram | AAAI 2017 |
| Geo-Teaser ([WWW 2017](https://link.springer.com/chapter/10.1007/978-981-13-1349-3_4)) | POI(+day-state) | static | skip-gram + geo ranking | skip-gram | WWW 2017 |
| CAPE ([IJCAI 2018](https://www.ijcai.org/proceedings/2018/458)) | POI | static | check-in context + text | skip-gram | IJCAI 2018 |
| TALE ([TKDE 2022](https://ieeexplore.ieee.org/document/9351627/)) | location | static (time-aware training) | CBOW + temporal-tree softmax | masked CBOW | TKDE 2022 |
| **CTLE** ([AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16548), [code](https://github.com/Logan-Lin/CTLE)) | **visit (check-in)** | **contextual** | bidirectional transformer over trajectory + continuous temporal encoding | masked location + masked hour | AAAI 2021 |
| Hier ([SIGSPATIAL 2020](https://arxiv.org/pdf/2002.02058)) | place | static | multi-scale grid hierarchy | skip-gram | SIGSPATIAL 2020 |
| CASTLE ([ISPRS Archives 2023](https://ui.adsabs.harvard.edu/abs/2023ISPAr48W2...15C/abstract)) | visit | contextual | BART-style transformer | denoising | ISPRS 2023 |
| Geo-Tokenizer ([ECML PKDD 2023](https://arxiv.org/abs/2310.01252)) | visit | contextual | hierarchical grid tokens | hierarchical AR masking | ECML PKDD 2023 |
| CACSR ([AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25546)) | sequence | n/a | adversarial contrastive | InfoNCE | AAAI 2023 |
| STCCR ([TKDE 2024](https://arxiv.org/abs/2407.15899)) | sequence | n/a | cross-view contrastive | InfoNCE | TKDE 2024 |
| LBSN2Vec ([WWW 2019](https://dl.acm.org/doi/fullHtml/10.1145/3308558.3313635)) / LBSN2Vec++ ([TKDE 2020](https://ieeexplore.ieee.org/abstract/document/9099985)) | node | static | hypergraph random walk | skip-gram-style | WWW 2019 / TKDE |
| Space2Vec ([ICLR 2020](https://arxiv.org/abs/2003.00824)), Sphere2Vec ([ISPRS 2023](https://arxiv.org/abs/2306.17624)), Time2Vec ([arXiv 2019](https://arxiv.org/abs/1907.05321)) | coordinates/time | static | sinusoidal encoders | various | — |
| HMRM ([TKDE, DOI 10.1109/TKDE.2020.3001025](https://ieeexplore.ieee.org/document/9112685/)) | location/user/time/activity | static | collective matrix factorization | MF | TKDE |

**Correction for this repo's docs**: HMRM = "**Human** Mobility Representation Model" (Chen et al., "Modeling Spatial Trajectories With Attribute Representation Learning", TKDE) — not "Heterogeneous" as `CLAUDE.md` and `docs/context/EMBEDDINGS.md` currently expand it.

**CTLE is the head-on competitor.** Verified from the full paper: it produces per-visit contextual embeddings (same location → different vectors in different trajectory contexts), pre-trains with masked-location + masked-hour objectives, freezes the embedding, and feeds it to third-party downstream predictors (ST-RNN, ERPP, ST-LSTM) on two mobile-signaling datasets with **temporal splits**, reporting Accuracy / macro-Recall / **macro-F1**. Its baseline suite (one-hot, skip-gram, POI2Vec, Geo-Teaser, TALE, Hier) is the de-facto canon for location-embedding papers. CTLE is currently **absent from this repo**.

**Standard protocol for embedding-substrate papers** (CTLE/UniTE template): pre-train on the *training portion only* → freeze → plug into multiple established downstream predictors → temporal split → Acc/macro-F1. The repo's transductive full-corpus substrate training deviates from this (see [`evaluation_protocol_review.md §4`](evaluation_protocol_review.md)).

Survey/benchmark authority: **UniTE** (TKDE 2025, [link](https://arxiv.org/abs/2407.12550)) — unified pipeline for pre-trained trajectory embeddings; reviewers may point to it for standardized implementations.

---

## 7. The HGI lineage and its citation neighborhood

- **HGI**: Huang, Zhang, Mai, Guo, Cui, "Learning urban region representations with POIs and hierarchical graph infomax", *ISPRS JPRS* 196:134–145, 2023 ([link](https://www.sciencedirect.com/science/article/abs/pii/S0924271622003148), [code](https://github.com/RightBank/HGI)). Embeds a **static POI → region → city hierarchy** (no check-ins, no users, no time): GCN POI encoder over a spatial-proximity graph, attentive region aggregation, DGI-style MI maximization across the hierarchy. Output: **region embeddings** for urban-analytics tasks (function inference, population, housing price). No sequential or mobility task.
- **DGI**: Veličković et al., ICLR 2019 ([link](https://arxiv.org/abs/1809.10341)).
- **Citation scan (96 citing papers, 2023–2026, via Semantic Scholar)**: all citers are urban-region / land-use / geospatial-foundation-model works (ReCP AAAI 2024, VecCity PVLDB 2025, CityFM CIKM 2023, ReFound KDD 2024, HyperRegion TMC 2025, …). **None extends HGI to check-in granularity, recommendation, or next-location tasks.** The only check-in/MTL-adjacent citer is ST-MTLNet (CoUrb 2026 — this project's own lineage). Direct searches for "Check2HGI" / "hierarchical graph infomax check-in" return nothing. **No scooping detected as of 2026-06-12.**

---

## 8. Region representation learning (bearing on "next-region")

ReMVC (TKDE 2022, [link](https://ieeexplore.ieee.org/document/9973276/)); MGFN (IJCAI 2022, [link](https://www.ijcai.org/proceedings/2022/321)); ReCP (AAAI 2024, [link](https://arxiv.org/abs/2312.09681)); VecCity benchmark (PVLDB 2025, [link](https://arxiv.org/pdf/2411.00874)).

**Key observation**: region-embedding papers evaluate on land-use / popularity / price — **"next-region prediction" is not an established benchmark task in either the region-embedding or the location-embedding literature**. The nearest analogue is coarse next-location prediction over grid/base-station cells (CTLE's mobile-signaling setup) and HMT-GRN's auxiliary geohash tasks. Census-tract regions (this repo, TIGER tracts) have no precedent in this canon at all; geohash/quadkey/grid dominate.

---

## 9. Surveys worth citing

1. Luca et al., "A Survey on Deep Learning for Human Mobility", ACM CSUR 2021 ([link](https://dl.acm.org/doi/10.1145/3485125)).
2. Islam et al., Neurocomputing 2022 ([link](https://www.sciencedirect.com/science/article/abs/pii/S0925231221016106)).
3. Zhang et al., "A Survey on POI Recommendation: Models, Architectures, and Security", TKDE 2025 ([link](https://arxiv.org/abs/2410.02191)) — best single up-to-date survey.
4. GNN-based next-POI survey, *J. Reliable Intelligent Environments* 2024 ([link](https://link.springer.com/article/10.1007/s40860-024-00233-z)).
5. UniTE (trajectory-embedding pretraining survey + pipeline), TKDE 2025 ([link](https://arxiv.org/abs/2407.12550)).
6. Luca et al., trajectory test-train overlap critique, *Machine Learning* 2023 ([link](https://link.springer.com/article/10.1007/s10994-023-06386-x)).

---

## 10. Cross-cutting observations for this project

1. **The substrate-novelty intersection is vacant** (contextual check-in unit × hierarchical graph infomax) — but both axes are individually occupied (CTLE; HGI), so the claim must be the combination, defended against "why not CTLE?".
2. **The task pairing (category+region headline, no POI) is unpublished as a framing** — but published as auxiliaries (KGTB, HMT-GRN). The defense must be affirmative (sparsity, privacy, deployment sufficiency), not just "nobody did it".
3. **The repo's null result on MTL optimizers + gradient orthogonality replicates Kurin/Xin (NeurIPS 2022) in a new domain** — a citable strength.
4. **Protocol divergence is the biggest external-validity exposure**: random (user-disjoint) 5-fold CV vs the field's universal temporal splits; macro-F1 over 7 merged categories vs Acc@K over hundreds of categories; census tracts vs geohash. Every cross-paper numeric comparison is invalid; only within-protocol relative claims survive.
5. **No published competitor evaluates with macro-F1**; bridging metrics (Acc@1/@5 for category) would help reviewers calibrate.
