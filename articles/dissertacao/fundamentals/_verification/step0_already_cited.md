# STEP 0 — De-duplicated already-cited reference set (project ground truth)

_Union of the three paper bibliographies (CBIC 46 + CoUrb 33 + MobiWac 46 = 125 raw entries) collapsed by DOI and normalized title to **92 distinct works**. This is both the starting set and the "do-not-rediscover" list for STEP 1. Themes assigned from the repo `science.md §6` author-reviewed digest. The frontier set (`science/new_references.bib`, 24 entries) is listed separately at the end: it is already proposed, not yet in a paper .bib, and carries three key collisions to rename before use._


## Theme A · POI-prediction tasks & next-POI sequence models  (40 works)

| BibTeX key(s) | Year | Title | DOI |
|---|---|---|---|
| song2010limits | 2010 | Limits of Predictability in Human Mobility | 10.1126/science.1177170 |
| ye2013nextmove | 2013 | What's Your Next Move: User Activity Prediction in Location-based Social Networks | 10.1137/1.9781611972832.19 |
| 10.1145/2661829.2662002 | 2014 | Exploiting Geographical Neighborhood Characteristics for Location Recommendation | 10.1145/2661829.2662002 |
| lipton2015learning | 2015 | Learning to Diagnose with LSTM Recurrent Neural Networks | — |
| liu2016strnn | 2016 | Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts | — |
| he2017lbpr | 2017 | Category-aware Next Point-of-Interest Recommendation via Listwise Bayesian Personalized … | 10.24963/ijcai.2017/255 |
| Liao2018 | 2018 | Predicting Activity and Location with Multi-task Context Aware Recurrent Neural Network | 10.24963/ijcai.2018/477 |
| feng2018deepmove | 2018 | DeepMove: Predicting Human Mobility with Attentional Recurrent Networks | 10.1145/3178876.3186058 |
| zeng2019mhape / zeng2019next | 2019 | A Next Location Predicting Approach Based on a Recurrent Neural Network and Self-Attention | 10.1007/978-3-030-30146-0_21 |
| rahmani2019category | 2019 | Category-aware location embedding for point-of-interest recommendation | — |
| du2019beyond | 2019 | Beyond geo-first law: Learning spatial representations via integrated autocorrelations a… | — |
| capanema2019identificacao | 2019 | Identificação e Classificação de Pontos de Interesse Individuais com Base em Dados Espar… | — |
| silva2019urbancomputing | 2019 | Urban Computing Leveraging Location-Based Social Network Data: A Survey | 10.1145/3301284 |
| Zhang2020 / zhang2020interactive | 2020 | An Interactive Multi-Task Learning Framework for Next POI Recommendation with Uncertain … | 10.24963/ijcai.2020/491 |
| Xia2020 / xia2020mtpr | 2020 | MTPR: A Multi-Task Learning Based POI Recommendation Considering Temporal Check-Ins and … | 10.3390/app10196664 |
| yang2020flashback | 2020 | Location Prediction over Sparse User Mobility Traces Using RNNs: Flashback in Hidden Sta… | 10.24963/ijcai.2020/302 |
| yu2020catdm | 2020 | A Category-Aware Deep Model for Successive POI Recommendation on Sparse Check-in Data | 10.1145/3366423.3380202 |
| Halder2021 | 2021 | Transformer-Based Multi-task Learning for Queuing Time Aware Next POI Recommendation | — |
| lin2021ctle | 2021 | Pre-training Context and Time Aware Location Embeddings from Spatial-Temporal Trajectori… | — |
| luo2021stan | 2021 | STAN: Spatio-Temporal Attention Network for Next Location Recommendation | 10.1145/3442381.3449998 |
| li2021sgrec | 2021 | Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph A… | 10.24963/ijcai.2021/206 |
| luca2021mobilitysurvey | 2021 | A Survey on Deep Learning for Human Mobility | 10.1145/3485125 |
| Lim2022 / lim2022hierarchical / lim2022hmtgrn | 2022 | Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation | 10.1145/3477495.3531989 |
| Halder2022 | 2022 | POI Recommendation with Queuing Time and User Interest Awareness | 10.1007/s10618-022-00865-w |
| yang2022getnext | 2022 | GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation | 10.1145/3477495.3531983 |
| zhu2022drrgnn | 2022 | Predicting a Person's Next Activity Region with a Dynamic Region-Relation-Aware Graph Ne… | 10.1145/3529091 |
| chen2020hmrm / chen2020modeling | 2022 | Modeling Spatial Trajectories with Attribute Representation Learning | 10.1109/tkde.2020.3001025 |
| wei2022finetuned | 2022 | Finetuned Language Models are Zero-Shot Learners | — |
| Xu2023 | 2023 | TME: Tree-guided Multi-task Embedding Learning towards Semantic Venue Annotation | 10.1145/3582553 |
| capanema2023poirgnn | 2023 | Combining recurrent and Graph Neural Networks to predict the next place's category | 10.1016/j.adhoc.2022.103016 |
| he2024imnext | 2024 | ImNext: Irregular Interval Attention and Multi-Task Learning for Next POI Recommendation | — |
| wu2024mrp-llm | 2024 | MRP-LLM: Multitask Reflective Large Language Models for Privacy-Preserving Next POI Reco… | — |
| dos2024havana | 2024 | HAVANA: Hybrid Attentional Graph Convolutional Network Semantic Venue Annotation Model | — |
| sun2024transtarec | 2024 | Transtarec: Time-adaptive translating embedding model for next poi recommendation | — |
| sun2024mcmg | 2024 | A Multi-channel Next POI Recommendation Framework with Multi-granularity Check-in Signals | 10.1145/3592789 |
| huang2024cslsl | 2024 | Human Mobility Prediction with Causal and Spatial-constrained Multi-task Network | 10.1140/epjds/s13688-024-00460-7 |
| sun2025kgtb | 2025 | Knowledge Graph Tokenization for Behavior-Aware Generative Next POI Recommendation | 10.48550/arxiv.2509.12350 |
| li2025rehdm | 2025 | Beyond Individual and Point: Next POI Recommendation via Region-aware Dynamic Hypergraph… | 10.24963/ijcai.2025/343 |
| moura2025mobilityaware | 2025 | On the Design of Mobility-Aware Systems: A Tourist's Perspective | 10.1109/mswim67937.2025.11308734 |
| oh2025fmintent | 2025 | FM-Intent: Predicting User Session Intent with Hierarchical Multi-Task Learning | — |

## Theme B · Representations for mobility  (18 works)

| BibTeX key(s) | Year | Title | DOI |
|---|---|---|---|
| belkin2003laplacian | 2003 | Laplacian eigenmaps for dimensionality reduction and data representation | — |
| grover2016node2vec | 2016 | node2vec: Scalable feature learning for networks | — |
| he2016resnet | 2016 | Deep Residual Learning for Image Recognition | 10.1109/cvpr.2016.90 |
| velivckovic2017graph | 2017 | Graph attention networks | — |
| vaswani2017attention | 2017 | Attention Is All You Need | — |
| church2017word2vec | 2017 | Word2Vec | — |
| feng2017poi2vec | 2017 | POI2Vec: Geographical Latent Representation for Predicting Future Visitors | — |
| perez2018film | 2018 | FiLM: Visual Reasoning with a General Conditioning Layer | — |
| velickovic2019deep / velickovic2019dgi / velivckovic2018deep | 2018 | Deep graph infomax | — |
| kazemi2019time2vec | 2019 | Time2vec: Learning a vector representation of time | — |
| sun2020go | 2020 | Where to go next: Modeling long-and short-term user preferences for point-of-interest re… | — |
| mai2020multiscalerepresentationlearningspatial | 2020 | Multi-Scale Representation Learning for Spatial Feature Distributions using Grid Cells | — |
| sitzmann2020implicit | 2020 | Implicit neural representations with periodic activation functions | — |
| huang2022estimating | 2022 | Estimating urban functional distributions with semantics preserved POI embedding | — |
| huang2023hgi / huang2023learning | 2023 | Learning Urban Region Representations with POIs and Hierarchical Graph Infomax | 10.1016/j.isprsjprs.2022.11.021 |
| mai2023sphere2vecgeneralpurposelocationrepresentation | 2023 | Sphere2Vec: A General-Purpose Location Representation Learning over a Spherical Surface … | — |
| rußwurm2024geographiclocationencodingspherical | 2024 | Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Netw… | — |
| wu2024torchspatial | 2024 | Torchspatial: A location encoding framework and benchmark for spatial representation lea… | — |

## Theme C · Multi-task learning  (26 works)

| BibTeX key(s) | Year | Title | DOI |
|---|---|---|---|
| caruana1997multitask | 1997 | Multitask Learning | 10.1023/a:1007379606734 |
| baxter2000model | 2000 | A Model of Inductive Bias Learning | — |
| misra2016cross | 2016 | Cross-Stitch Networks for Multi-Task Learning | 10.1109/cvpr.2016.434 |
| kokkinos2016ubernet | 2016 | UberNet: Training a `Universal' Convolutional Neural Network for Low-, Mid-, and High-Le… | — |
| ruder2017sluice | 2017 | Sluice Networks: Learning What to Share Between Loosely Related Tasks | — |
| ma2018mmoe | 2018 | Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts | 10.1145/3219819.3220007 |
| thung2018brief | 2018 | A brief review on multi-task learning | — |
| chen2018gradnorm | 2018 | GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks | — |
| sener2018mgda | 2018 | Multi-Task Learning as Multi-Objective Optimization | — |
| yu2019mmoe | 2019 | Multi-Gate Mixture-of-Experts for Multi-Task Learning in Recommendations | — |
| liu2019dwa | 2019 | End-to-End Multi-Task Learning with Attention | — |
| standley2020tasks | 2020 | Which Tasks Should Be Learned Together in Multi-Task Learning? | — |
| yu2020pcgrad | 2020 | Gradient Surgery for Multi-Task Learning | — |
| islam2022survey | 2022 | A Survey on Deep Learning Based Point-of-Interest (POI) Recommendations | 10.1016/j.neucom.2021.05.114 |
| zhang2021survey | 2022 | A Survey on Multi-Task Learning | 10.1109/tkde.2021.3072953 |
| nash / navon2022nashmtl | 2022 | Multi-task learning as a bargaining game | — |
| vandenhende2022mtl | 2022 | Multi-Task Learning for Dense Prediction Tasks: A Survey | 10.1109/tpami.2021.3054719 |
| xin2022domtl | 2022 | Do Current Multi-Task Optimization Methods in Deep Learning Even Help? | — |
| kurin2022defense | 2022 | In Defense of the Unitary Scalarization for Deep Multi-Task Learning | — |
| senushkin2023aligned | 2023 | Independent Component Alignment for Multi-Task Learning | — |
| liu2023famo | 2023 | FAMO: Fast Adaptive Multitask Optimization | — |
| huang2024mt-net | 2024 | Learning Time Slot Preferences via Mobility Tree for Next POI Recommendation | — |
| yu2024survey | 2024 | Unleashing the Power of Multi-Task Learning: A Comprehensive Survey Spanning Traditional… | — |
| wang2025hamtl | 2025 | Hierarchy Aware-based Multi-task Learning for User Location Prediction | 10.1007/s11227-025-07643-7 |
| silva2025mtlnet | 2025 | An Investigation into Multi-Task Learning for Point-of-Interest Category Classification … | — |
| paiva2026courb | 2026 | ST-MTLNet: Representa\cc\~oes Espa\cco-Temporais de Pontos de Interesse para Aprendizado… | — |

## Theme D · Datasets, metrics & evaluation  (8 works)

| BibTeX key(s) | Year | Title | DOI |
|---|---|---|---|
| holm1979 | 1979 | A simple sequentially rejective multiple test procedure | — |
| Cho2011 / cho2011friendship / cho2011gowalla | 2011 | Friendship and Mobility: User Movement in Location-Based Social Networks | 10.1145/2020408.2020579 |
| SNAP2014 / jure2014snap | 2014 | SNAP Datasets: Stanford Large Network Dataset Collection | — |
| bastug2014edge | 2014 | Living on the Edge: The Role of Proactive Caching in 5G Wireless Networks | 10.1109/mcom.2014.6871674 |
| lakens2017tost | 2017 | Equivalence tests: A practical primer for t tests, correlations, and meta-analyses | — |
| vielhaus2022handover | 2022 | Handover Predictions as an Enabler for Anticipatory Service Adaptations in Next-Generati… | 10.1145/3551660.3560913 |
| cho_gowalla_2023 | 2023 | gowalla\_data | 10.6084/m9.figshare.22126586 |
| wongso2025massivesteps | 2025 | Massive-STEPS: Massive Semantic Trajectories for Understanding POI Check-ins --- Dataset… | — |

## Key-spelling variants collapsed (one work, multiple keys)

| Work | Keys collapsed | Canonical DOI |
|---|---|---|
| Modeling Spatial Trajectories with Attribute Represen… | `chen2020hmrm` / `chen2020modeling` | 10.1109/tkde.2020.3001025 |
| Friendship and Mobility: User Movement in Location-Ba… | `Cho2011` / `cho2011friendship` / `cho2011gowalla` | 10.1145/2020408.2020579 |
| Learning Urban Region Representations with POIs and H… | `huang2023hgi` / `huang2023learning` | 10.1016/j.isprsjprs.2022.11.021 |
| Hierarchical Multi-Task Graph Recurrent Network for N… | `Lim2022` / `lim2022hierarchical` / `lim2022hmtgrn` | 10.1145/3477495.3531989 |
| Multi-task learning as a bargaining game | `nash` / `navon2022nashmtl` | — |
| SNAP Datasets: Stanford Large Network Dataset Collect… | `SNAP2014` / `jure2014snap` | — |
| Deep graph infomax | `velickovic2019deep` / `velickovic2019dgi` / `velivckovic2018deep` | — |
| MTPR: A Multi-Task Learning Based POI Recommendation … | `Xia2020` / `xia2020mtpr` | 10.3390/app10196664 |
| A Next Location Predicting Approach Based on a Recurr… | `zeng2019mhape` / `zeng2019next` | 10.1007/978-3-030-30146-0_21 |
| An Interactive Multi-Task Learning Framework for Next… | `Zhang2020` / `zhang2020interactive` | 10.24963/ijcai.2020/491 |

## Frontier set — already proposed in `science/new_references.bib` (do NOT re-discover)

_24 entries from the prior frontier survey. **Three key collisions** (each name used twice for different papers) must be renamed before citing: `Wang_2023`, `Liu_2023`, `Lai_2024`._

| Key | Year | Title | DOI |
|---|---|---|---|
| `An_2024` | 2024 | MvStHgL: Multi-View Hypergraph Learning with Spatial-Temporal Periodic Interes… | 10.1145/3664651 |
| `Beneduce_2025` | 2025 | Large Language Models are Zero-Shot Next Location Predictors | 10.1109/access.2025.3565297 |
| `Chen_2025` | 2025 | Next-POI Recommendation via Spatial-Temporal Knowledge Graph Contrastive Learn… | 10.1109/tkde.2025.3545958 |
| `Cheng_2025` | 2025 | POI-Enhancer: An LLM-based Semantic Enhancement Framework for POI Representati… | 10.1609/aaai.v39i11.33252 |
| `Feng_2024` | 2024 | Where to Move Next: Zero-shot Generalization of LLMs for Next POI Recommendation | 10.1109/cai59869.2024.00277 |
| `Huang_2022` | 2022 | Estimating urban functional distributions with semantics preserved POI embedding | 10.1080/13658816.2022.2040510 |
| `Lai_2023` | 2023 | Multi-view Spatial-Temporal Enhanced Hypergraph Network for Next POI Recommend… | 10.1007/978-3-031-30672-3_16 |
| `Lai_2024`  ⚠collision | 2024 | Disentangled Contrastive Hypergraph Learning for Next POI Recommendation | 10.1145/3626772.3657726 |
| `Lai_2024`  ⚠collision | 2024 | Adaptive Spatial-Temporal Hypergraph Fusion Learning for Next POI Recommendation | 10.1109/icassp48485.2024.10447357 |
| `Liu_2022` | 2022 | CSTRM: Contrastive Self-Supervised Trajectory Representation Model for traject… | 10.1016/j.comcom.2022.01.001 |
| `Liu_2023`  ⚠collision | 2023 | A multi-task deep learning model integrating ship trajectory and collision ris… | 10.1016/j.oceaneng.2023.115870 |
| `Liu_2023`  ⚠collision | 2023 | Mandari: Multi-Modal Temporal Knowledge Graph-aware Sub-graph Embedding for Ne… | 10.1109/icme55011.2023.00264 |
| `Luo_2022` | 2022 | Urban Region Profiling via Multi-Graph Representation Learning | 10.1145/3511808.3557720 |
| `Luo_2023` | 2023 | Urban Functional Zone Classification Based on POI Data and Machine Learning | 10.3390/su15054631 |
| `Meng_2023` | 2023 | Lane-changing trajectory prediction based on multi-task learning | 10.1093/tse/tdac073 |
| `Solatorio_2023` | 2023 | GeoFormer: Predicting Human Mobility using Generative Pre-trained Transformer … | 10.1145/3615894.3628499 |
| `Sun_2024` | 2024 | Towards privacy-preserving category-aware POI recommendation over encrypted LB… | 10.1016/j.ins.2024.120253 |
| `Wang_2022` | 2022 | A multi-task learning-based framework for global maritime trajectory and desti… | 10.1016/j.martra.2022.100072 |
| `Wang_2023`  ⚠collision | 2023 | Context-and category-aware double self-attention model for next POI recommenda… | 10.1007/s10489-022-04396-1 |
| `Wang_2023`  ⚠collision | 2023 | Zone-Enhanced Spatio-Temporal Representation Learning for Urban POI Recommenda… | 10.1109/tkde.2023.3243239 |
| `Wang_2025` | 2025 | Multi-modal contrastive learning of urban space representations from POI data | 10.1016/j.compenvurbsys.2025.102299 |
| `Yan_2023` | 2023 | Spatio-Temporal Hypergraph Learning for Next POI Recommendation | 10.1145/3539618.3591770 |
| `Yangyang_2022` | 2022 | POI Recommendation System using Hypergraph Embedding and Logical Matrix Factor… | 10.36548/jaicn.2022.1.003 |
| `Zhang_2024` | 2024 | Hyper-relational knowledge graph neural network for next POI recommendation | 10.1007/s11280-024-01279-y |