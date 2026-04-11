# Related Work and Metrics for POI Category Tasks

Date: 2026-04-11

This note records the project context and a literature scan for the two tasks used in this repository:

- **Next task:** predict the category of the next POI visited from a user's recent check-in history.
- **Category task:** classify the category of an unknown/unlabeled POI from its POI-level embedding.

The main metric question is whether the results should be displayed as top-k recommendation metrics, F1/precision/recall, accuracy, or ranking metrics such as MRR/NDCG/MAP.

## Project Context

The repository is not a standard "recommend the next POI ID" benchmark. It is closer to **classification over POI categories**.

Current implementation:

- `src/data/inputs/core.py` builds fixed-length sequences for next-POI prediction and maps the target POI to its category label.
- `src/data/inputs/builders.py` separates POI-level embeddings from check-in-level embeddings. Category classification explicitly requires POI-level embeddings, because the same POI should have one stable representation.
- `src/models/mtlnet.py` trains a shared multi-task model with two heads: one for POI category classification and one for next-category prediction.
- `src/training/runners/_single_task_train.py` monitors validation macro F1 and accuracy.
- `src/training/evaluate.py` and `src/training/shared_evaluate.py` use `sklearn.classification_report`, which reports per-class precision, recall, F1-score, support, macro average, weighted average, and accuracy.
- `articles/CBIC___MTL/sections/results.tex` and the result tables report per-class F1, precision, and recall for both tasks.

So the current project display style is defensible: **per-class precision/recall/F1 plus macro-F1 and accuracy**. Top-k metrics are common in next-POI ID recommendation, but less directly aligned with a 7-class category classifier unless the model is intentionally evaluated as a ranked category recommender.

## Terminology Map

The literature uses several names for nearby tasks:

| Literature term | Usually predicts | Most common metric family | Fit to this project |
|---|---:|---|---|
| Next location prediction | exact next location/venue/POI ID | Recall@K, F1@K, MAP, AUC, Acc@K | Related, but target is more fine-grained than category |
| Next POI recommendation | ranked POI list | Recall@K, Hit/Acc@K, NDCG@K, MRR | Related; metrics are often top-k |
| Next category prediction | POI category of the next visit | Precision@K, Recall@K, F1@K, or classification F1 | Closest to the project's next task |
| POI category classification / semantic annotation | category label for an unlabeled POI | accuracy, precision, recall, macro/micro F1 | Closest to the project's category task |
| POI category embedding | category/category-hierarchy representation | downstream task metrics | Useful for embeddings, not always a direct classifier |

## Metric Pattern

**Ranking-oriented next-POI papers** usually report whether the true next POI appears high in a ranked list:

- `Recall@K`, `HitRate@K`, or `Acc@K`: did the true POI appear in the top K?
- `NDCG@K`: did the true POI appear near the top of the top-K list?
- `MRR`: reciprocal rank of the true POI.
- `MAP` and `AUC`: older global ranking metrics.

**Category classification papers** usually report class metrics:

- Accuracy.
- Precision, recall, and F1-score.
- Macro-F1 is especially important when categories are imbalanced, which is true for the Gowalla/Foursquare-style high-level category mapping used here.

For this project, the most robust presentation is:

1. Primary: macro-F1, plus per-class precision/recall/F1.
2. Secondary: accuracy, because it is easy to interpret but can be inflated by dominant categories.
3. Optional if needed for comparison with recommender papers: top-k category accuracy or `Recall@K` over categories, especially `K=1`, `K=3`, and `K=5`. With only 7 classes, large K values should be interpreted carefully.

## Related Work: Next POI / Next Category Prediction

| Year | Paper | Task emphasis | Metrics used/displayed | Relevance to this project |
|---:|---|---|---|---|
| 2010 | FPMC: Factorizing Personalized Markov Chains for next-basket/sequential recommendation | Early sequential recommendation baseline later reused for POI transitions | Usually top-k/ranking metrics in later POI adaptations | Useful historical baseline family, but not category-specific |
| 2013 | Cheng et al., "Where You Like to Go Next: Successive Point-of-Interest Recommendation" | Successive POI recommendation with geographical influence | Top-k POI recommendation style metrics in later comparisons | One of the older POI-specific baselines behind ST-RNN and later work |
| 2015 | PRME: Personalized Ranking Metric Embedding | Sequential POI recommendation with metric embeddings | Ranking metrics in ST-RNN comparisons | Embedding baseline, not category classifier |
| 2016 | Liu et al., "Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts" / ST-RNN | Next location prediction from spatial-temporal histories | `Recall@K` and `F1-score@K` for `K={1,5,10}`, plus `MAP` and `AUC` | Strong old baseline. Shows early next-location work used both top-k and F1@K. |
| 2018 | Feng et al., "DeepMove: Predicting Human Mobility with Attentional Recurrent Networks" | Human mobility / next location prediction with attention and periodicity | Top-1 and top-5 prediction accuracy | Important neural baseline; metric is top-k accuracy, not macro-F1 |
| 2019 | Zeng et al., "A Next Location Predicting Approach Based on a Recurrent Neural Network and Self-Attention" / MHA+PE | Recurrent next-location prediction with multi-head attention and positional encoding | In this repository's comparison, displayed as per-class precision, recall, and F1 for next-category prediction | Direct local baseline for the current next task |
| 2019/2020 | Zhao et al., "Where to Go Next: A Spatio-Temporal Gated Network for Next POI Recommendation" / STGN, NeuNext | Next POI recommendation with spatio-temporal gates and context prediction | Accuracy/`Acc@K` and `MAP` | Useful for sequence modeling; still POI-ID ranking oriented |
| 2020 | Sun et al., "Where to Go Next: Modeling Long- and Short-Term User Preferences for POI Recommendation" / LSTPM | Next-POI recommendation with long/short-term preferences | `Recall@K` and `NDCG@K`, with `K={1,5,10}` | Modern benchmark style: ranking metrics dominate |
| 2021 | Luo et al., "STAN: Spatio-Temporal Attention Network for Next Location Recommendation" | Attention over spatio-temporal trajectory context | Commonly compared using top-k ranking metrics such as hit/recall and NDCG | Good related baseline when comparing attention mechanisms |
| 2022 | Yang et al., "GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation" | Graph-enhanced Transformer using global trajectory flow maps and category embeddings | Later papers benchmark it with `Acc@K`/top-k and `MRR` | Strong transformer/GNN baseline, especially for graph-enhanced embeddings |
| 2024 | Jia et al., "Learning Hierarchy-Enhanced POI Category Representations Using Disentangled Mobility Sequences" / SD-CEM | POI category representations; includes masked category prediction and next category prediction as pretext tasks | Downstream task metrics across multiple tasks; exact task table should be checked before direct metric comparison | Highly relevant because it explicitly optimizes a next-category signal and category hierarchy |
| 2025 | Wang and Jiang, "Enhancing Next POI Recommendation via Multi-Graph Modeling and Multi-Granularity Contrastive Learning" / MGMCL | Multi-graph and multi-contrastive next-POI recommendation | `Acc@K` and `MRR`; reports repeated-run averages | State-of-the-art style: top-k accuracy plus rank quality |
| 2026 | Liu and Fu, "Language-Guided Spatio-Temporal Context Learning for Next POI Recommendation" / LSCNP | BERT-guided spatio-temporal context, graphs, contrastive learning | `HR@5`, `HR@10`, `HR@20`, and `MRR` | Current SOTA-style paper; metrics are ranking-oriented |

## Related Work: Unknown POI / Category Classification

| Year | Paper | Task emphasis | Metrics used/displayed | Relevance to this project |
|---:|---|---|---|---|
| 2015 | Wu et al., "Semantic Annotation of Mobility Data Using Social Media" | Semantic annotation of places/mobility traces | Accuracy-style semantic annotation metrics | Older semantic labeling direction, related to unknown-POI category assignment |
| 2020 | Chen et al., "Modeling Spatial Trajectories with Attribute Representation Learning" / HMRM | POI/category representation from mobility trajectories | The project baseline table reports precision, recall, and F1 by category | Directly relevant category-classification baseline in this repo |
| 2021 | Liu et al., "An Attention-Based Category-Aware GRU Model for the Next POI Recommendation" / ATCA-GRU | Predicts next check-in category before recommending POIs | `Precision@K`, `Recall@K`, and `F1-score@K` over top categories | Very close to the next-category framing, but displayed as top-k category recommendation |
| 2024 | Jia et al., SD-CEM | Hierarchy-enhanced POI category representations | Evaluated through multiple downstream tasks, including category/next-category prediction tasks | Relevant for adding hierarchy-aware category embeddings |
| 2024 | Zhang et al., "Exploring Urban Semantics: A Multimodal Model for POI Semantic Annotation with Street View Images and Place Names" | POI semantic annotation from multimodal signals | Metric table was not extracted in this scan; keep as a candidate rather than a metric anchor | Relevant if the project later includes street-view/name/text features |

## Direct Source Notes

- ST-RNN states that `Recall@K` and `F1-score@K` are used with `K=1,5,10`, and also reports `MAP` and `AUC`: [AAAI PDF](https://cdn.aaai.org/ojs/9971/9971-13-13499-1-2-20201228.pdf).
- DeepMove ranks candidate locations and checks top-1/top-5 prediction accuracy: [ResearchGate full text / ACM DOI page mirror](https://www.researchgate.net/publication/324509265_DeepMove_Predicting_Human_Mobility_with_Attentional_Recurrent_Networks).
- LSTPM explicitly adopts `Recall@K` and `NDCG@K`, with `K={1,5,10}`: [ResearchGate full text](https://www.researchgate.net/publication/341907286_Where_to_Go_Next_Modeling_Long-_and_Short-Term_User_Preferences_for_Point-of-Interest_Recommendation).
- MGMCL explicitly adopts `Acc@K` and `MRR`: [Springer open-access article](https://link.springer.com/article/10.1007/s44443-025-00325-7).
- LSCNP explicitly adopts `HR@5`, `HR@10`, `HR@20`, and `MRR`: [MDPI open-access article](https://www.mdpi.com/2220-9964/15/1/28).
- SD-CEM explicitly studies hierarchy-enhanced POI category representations and optimizes masked category prediction plus next category prediction: [IJCAI page](https://www.ijcai.org/proceedings/2024/0231), [PDF](https://www.ijcai.org/proceedings/2024/0231.pdf).
- The current project article/table style reports per-class F1, precision, and recall for both tasks: `articles/CBIC___MTL/sections/results.tex`, `articles/CBIC___MTL/tables/next_result.tex`, and `articles/CBIC___MTL/tables/category_result.tex`.

Rows without a direct source note above should be treated as context/citation candidates. Before using them as metric claims in the article text, re-open the original paper and verify the exact evaluation table.

## Recommendation for This Project

Use a **classification-first** result layout:

| Task | Primary metrics | Secondary metrics | Optional literature-bridge metrics |
|---|---|---|---|
| Next POI category prediction | macro-F1; per-class precision/recall/F1 | accuracy; confusion matrix | top-k category accuracy / Recall@K for `K=1,3,5` |
| Unknown POI category classification | macro-F1; per-class precision/recall/F1 | accuracy; weighted-F1 | none required unless comparing to category-recommendation papers |

Rationale:

- The target label is category, not POI ID.
- The class distribution is imbalanced; macro-F1 is more informative than accuracy.
- Per-class tables match the current CBIC article and expose category-specific failures such as Nightlife/Outdoors.
- Top-k metrics should be added only as an auxiliary bridge to next-POI recommendation literature. They should not replace macro-F1 for the current task.

## Citation Candidates for a Paper Section

Historical and baseline sequence models:

- Rendle et al. 2010, FPMC.
- Cheng et al. 2013, successive POI recommendation.
- Feng et al. 2015, PRME.
- Liu et al. 2016, ST-RNN.
- Feng et al. 2018, DeepMove.

Category/next-category closest works:

- Chen et al. 2020, HMRM / spatial trajectory attribute representation.
- Liu et al. 2021, ATCA-GRU category-aware next POI recommendation.
- Capanema et al. 2023, combining recurrent and graph neural networks to predict next place category.
- Jia et al. 2024, SD-CEM hierarchy-enhanced POI category representations.

State-of-the-art next-POI recommendation:

- Sun et al. 2020, LSTPM.
- Luo et al. 2021, STAN.
- Yang et al. 2022, GETNext.
- Wang and Jiang 2025, MGMCL.
- Liu and Fu 2026, LSCNP.
