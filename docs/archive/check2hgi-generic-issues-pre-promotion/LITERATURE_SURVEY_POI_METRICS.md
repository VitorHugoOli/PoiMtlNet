# Literature Survey: Evaluation Metrics for POI Tasks

## Context of This Project

This project (MTLnet) addresses **two tasks** jointly via multi-task learning:

1. **POI Category Classification** ("category") — Given a POI's embedding (derived from check-in data, spatial/temporal features), classify it into one of 7 semantic categories (Community, Entertainment, Food, Nightlife, Outdoors, Shopping, Travel). This is a **static classification** problem (one embedding → one label).

2. **Next-POI Category Prediction** ("next") — Given a user's sequence of past check-ins (window of 9), predict the **category** of the next POI the user will visit. This is a **sequential prediction** problem.

The project currently uses: **Macro-average F1-score** (primary), along with per-class Precision, Recall, and F1, plus overall Accuracy.

---

## TASK 1: POI Category Classification / Semantic Venue Annotation

This task is sometimes called "semantic venue annotation," "POI type inference," or "land use classification." The related works span from traditional ML to deep graph-based methods.

### Key Papers and Their Metrics

| # | Paper | Year | Venue | Primary Metrics | Notes |
|---|-------|------|-------|-----------------|-------|
| 1 | **HMRM** — Chen et al., "Modeling Spatial Trajectories with Attribute Representation Learning" | 2020 | IEEE TKDE | **Macro F1, Accuracy** | Uses PMI + matrix factorization → SVM. Reports per-class F1. |
| 2 | **TME** — Xu et al., "Tree-guided Multi-task Embedding Learning towards Semantic Venue Annotation" | 2023 | ACM TOIS | **Macro F1, Micro F1, Accuracy** | Graph-based encoder with tree-structured categories. Reports Acc@1 as "accuracy." |
| 3 | **HAVANA** — dos Santos et al., "Hybrid Attentional Graph Convolutional Network Semantic Venue Annotation" | 2024 | BRACIS (Springer) | **Macro F1, Weighted F1, Accuracy** | GAT + ARMA. Per-class precision/recall/F1 reported. |
| 4 | **POI-RGNN** — Capanema et al., "Combining Recurrent and Graph Neural Networks to Predict the Next Place's Category" | 2019 | SBRC | **Macro F1, per-class F1, Accuracy** | RNN + GNN hybrid. Brazilian research group (same lineage as this project). |
| 5 | **Du et al.** — "Beyond Geo-First Law: Learning Spatial Representations via Integrated Autocorrelations" | 2019 | ICDM | **Accuracy, Macro F1** | Spatial representation learning for POI type classification. |
| 6 | **Huang et al.** — "Estimating Urban Functional Distributions with Semantics Preserved POI Embedding" | 2022 | IJGIS | **Accuracy, Macro F1, NMI (Normalized Mutual Information)** | POI embedding for urban function classification. |
| 7 | **Place2Vec** — Yan et al., "From ITDL to Place2Vec" | 2017 | SIGSPATIAL | **Accuracy** | Word2Vec-inspired POI embedding for type classification. |
| 8 | **POI2Vec** — Feng et al., "POI2Vec: Geographical Latent Representation for Predicting Future Visitors" | 2017 | AAAI | **Accuracy, AUC** | Hierarchical softmax POI representation. |
| 9 | **Zhai et al.** — "Beyond Word2Vec: An Approach for Urban Functional Region Extraction and Identification by Combining Place2Vec and POIs" | 2019 | Computers, Env. & Urban Systems | **Overall Accuracy (OA), Kappa coefficient** | Urban zone-level POI classification. |

### Summary for Task 1 (Category Classification)

| Metric | Frequency of Use | Notes |
|--------|-----------------|-------|
| **Macro F1-score** | ★★★★★ (Most common) | Standard primary metric; handles class imbalance |
| **Accuracy** | ★★★★☆ (Very common) | Always reported alongside F1 |
| **Per-class Precision/Recall/F1** | ★★★★☆ | Reported in detailed tables |
| **Weighted F1** | ★★☆☆☆ | Sometimes alongside macro F1 |
| **Micro F1** | ★★☆☆☆ | Less common, equivalent to accuracy for single-label |
| **Kappa coefficient** | ★☆☆☆☆ | Used in GIS/urban planning contexts |
| **AUC** | ★☆☆☆☆ | Rare for multi-class POI classification |

---

## TASK 2: Next-POI Prediction / Next-POI Recommendation

This task has **two distinct sub-communities** that use **very different metrics** depending on whether they predict the **exact next POI** (recommendation/ranking) or the **category of the next POI** (classification).

### A) Next-POI Recommendation (Predicting the exact POI — ranking problem)

| # | Paper | Year | Venue | Primary Metrics |
|---|-------|------|-------|-----------------|
| 1 | **STRNN** — Liu et al., "Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts" | 2016 | AAAI | **Acc@1, Acc@5, Acc@10** (Top-k accuracy) |
| 2 | **DeepMove** — Feng et al., "DeepMove: Predicting Human Mobility with Attentional Recurrent Networks" | 2018 | WWW | **Acc@1, Acc@5, Acc@10** |
| 3 | **LSTPM** — Sun et al., "Where to Go Next: Modeling Long- and Short-Term User Preferences for POI Recommendation" | 2020 | AAAI | **Recall@5, Recall@10, Recall@20, NDCG@5, NDCG@10, NDCG@20** |
| 4 | **STAN** — Luo et al., "STAN: Spatio-Temporal Attention Network for Next Location Recommendation" | 2021 | WWW | **Acc@1, Acc@5, Acc@10, MRR** |
| 5 | **GETNext** — Yang et al., "GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation" | 2022 | SIGIR | **Recall@1, Recall@5, Recall@10, NDCG@5, NDCG@10, MRR** |
| 6 | **HMT-GRN** — Lim et al., "Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation" | 2022 | SIGIR | **Recall@5, Recall@10, NDCG@5, NDCG@10, MRR** |
| 7 | **CARA** — Manotumruksa et al., "A Contextual Attention Recurrent Architecture for Context-Aware Venue Recommendation" | 2018 | SIGIR | **Recall@5, Recall@10, NDCG@5, NDCG@10** |
| 8 | **PLSPL** — Wu et al., "Personalized Long- and Short-term Preference Learning for Next POI Recommendation" | 2020 | IEEE TKDE | **Recall@5, Recall@10, Recall@20, NDCG@5, NDCG@10** |
| 9 | **Graph-Flashback** — Rao et al., "Graph-Flashback Network for Next Location Recommendation" | 2022 | KDD | **Acc@1, Acc@5, Acc@10, MRR** |
| 10 | **ImNext** — He et al., "Irregular Interval Attention and Multi-Task Learning for Next POI Recommendation" | 2024 | KBS | **Recall@5, Recall@10, Recall@20, NDCG@5, NDCG@10, NDCG@20** |
| 11 | **MRP-LLM** — Wu et al., "Multitask Reflective LLM for Privacy-Preserving Next POI Recommendation" | 2024 | arXiv | **Recall@5, Recall@10, Recall@20, NDCG@5, NDCG@10, NDCG@20** |
| 12 | **TLR-M** — Huang et al., "Learning Time Slot Preferences via Mobility Tree for Next POI Recommendation" | 2024 | arXiv | **Recall@5, Recall@10, Recall@20, NDCG@5, NDCG@10, NDCG@20** |

### B) Next-POI Category Prediction (Predicting the category — classification problem)

| # | Paper | Year | Venue | Primary Metrics |
|---|-------|------|-------|-----------------|
| 1 | **MHA+PE** — Zeng et al., "A Next Location Predicting Approach Based on RNN and Self-Attention" | 2019 | CollaborateCom | **Macro F1, per-class F1, Accuracy** |
| 2 | **MCARNN** — Liao et al., "Predicting Activity and Location with Multi-task Context Aware RNN" | 2018 | IJCAI | **Accuracy, Macro F1** |
| 3 | **iMTL** — Zhang et al., "An Interactive Multi-Task Learning Framework for Next POI with Uncertain Check-ins" | 2020 | IJCAI | **Accuracy, Macro F1** |
| 4 | **MTPR** — Xia et al., "Multi-Task Learning Based POI Recommendation Considering Temporal Check-Ins" | 2020 | Applied Sciences | **Accuracy, Recall@k (k=5,10), NDCG@k** |
| 5 | **TLR-M** — Halder et al., "Transformer-Based Multi-task Learning for Queuing Time Aware Next POI Recommendation" | 2021 | PAKDD | **Acc@1, Acc@5, NDCG@5, MRR** |
| 6 | **POI-RGNN** — Capanema et al. | 2019 | SBRC | **Macro F1, per-class F1, Accuracy** |

### Summary for Task 2 (Next-POI Prediction)

| Metric | Frequency | Sub-community | Notes |
|--------|-----------|---------------|-------|
| **Recall@k (k=5,10,20)** | ★★★★★ | Recommendation/Ranking | THE dominant metric for next-POI recommendation |
| **NDCG@k (k=5,10,20)** | ★★★★★ | Recommendation/Ranking | Almost always paired with Recall@k |
| **Acc@k / Hit@k (k=1,5,10)** | ★★★★☆ | Both | Acc@1 ≡ Top-1 accuracy. Very common, especially in older papers |
| **MRR (Mean Reciprocal Rank)** | ★★★☆☆ | Recommendation/Ranking | Growing in popularity |
| **Macro F1-score** | ★★★☆☆ | Classification | Used when predicting category (not exact POI) |
| **Accuracy** | ★★★☆☆ | Classification | Always reported with F1 |
| **Per-class Precision/Recall/F1** | ★★☆☆☆ | Classification | Detailed breakdown tables |
| **MAP (Mean Average Precision)** | ★☆☆☆☆ | Recommendation | Rare, older papers |

---

## Key Insight: Metric Choice Depends on Task Framing

There is a **critical distinction** in the literature:

1. **If you predict the exact next POI** (among thousands of candidates) → This is a **ranking/retrieval** problem → Metrics are **Recall@k, NDCG@k, MRR, Acc@k (Hit Rate)**. These are the dominant metrics in the mainstream next-POI recommendation community (SIGIR, KDD, WWW).

2. **If you predict the next POI's category** (among 7-10 categories) → This is a **multi-class classification** problem → Metrics are **Macro F1, Accuracy, per-class Precision/Recall/F1**. This is what this project does.

**This project's current metric choice (Macro F1 + per-class P/R/F1) is well-aligned** with the papers that also predict the next category (MCARNN, iMTL, MHA+PE, POI-RGNN). However, the broader next-POI recommendation community predominantly uses Recall@k and NDCG@k because they frame it as a ranking task.

---

## Recommendations

1. **For Category Classification**: The use of **Macro F1** as primary metric is the standard and correct choice. This is consistently used across TME, HMRM, HAVANA, and POI-RGNN.

2. **For Next-POI Category Prediction**: The use of **Macro F1** is appropriate since the project predicts the category (not the exact POI). However, also reporting **Acc@1** (which equals standard accuracy for single-label) connects with the broader recommendation literature. Consider adding **Weighted F1** as well to show robustness to class imbalance.

3. **If expanding to predicting the exact next POI**: Would need to switch to **Recall@k, NDCG@k, and MRR** to be comparable with the mainstream recommendation literature (GETNext, STAN, LSTPM, etc.).

4. **Cross-community bridge**: A few papers like MTPR and TLR-M report both classification metrics (F1) and ranking metrics (Recall@k, NDCG@k) because they handle both sub-tasks. This dual-reporting approach could strengthen the paper's positioning.
