# Baselines

This document describes the external baselines used to evaluate MTLnet.
For each baseline, paper-reported numbers and our reproduced results are shown where available.

> **Dataset note:**  
> The RGNN / MHA+PE paper (Capanema et al., 2022) reports results on the **full global Gowalla** dataset — not split by US state — so those numbers are **not directly comparable** with our state-level experiments.  
> The HAVANA paper (Santos et al.) splits Gowalla by **Florida, California, and Texas**, the same geography we use, so those numbers **are directly comparable**.

---

## Task 1 — Next Category Prediction

Predicts the **category** of the next POI a user will visit, given a window of recent check-ins.  
Evaluation: macro-averaged F1 over 7 categories, 5-fold cross-validation.

---

### MHA+PE

| Attribute | Detail |
|-----------|--------|
| **Source** | Zeng et al., 2019 — referenced as baseline in Capanema et al. (2022) |
| **PDF** | Referenced in `Combining_Recurrent_and_Graph_Neural_Networks_to_Predict_the_Next_Place's_Category.pdf` |
| **Task** | Next-category prediction from check-in sequence |
| **Reason for inclusion** | Canonical attention-based sequential baseline; primary comparison point in the RGNN paper |

**Architecture:** Multi-Head Attention (MHA) over the check-in sequence with Positional Encoding (PE) to preserve visit order. Adapted from the NLP transformer for POI recommendation; captures correlations between non-adjacent visits under different attention heads.

#### Paper-reported results — Gowalla (global, not state-split)

| Category | Precision (%) | Recall (%) | F1 (%) |
|----------|--------------|-----------|--------|
| Shopping | 42.2 ± 1.3 | 37.3 ± 4.7 | 39.5 ± 2.2 |
| Community | 38.3 ± 1.5 | 29.6 ± 4.1 | 33.2 ± 2.1 |
| Food | 39.2 ± 0.6 | 70.4 ± 2.5 | 50.3 ± 0.4 |
| Entertainment | 42.9 ± 2.0 | 12.4 ± 0.5 | 19.2 ± 0.7 |
| Travel | 43.6 ± 4.8 | 3.7 ± 1.0 | 6.8 ± 1.7 |
| Outdoors | 43.9 ± 1.5 | 14.3 ± 2.3 | 21.5 ± 2.5 |
| Nightlife | 45.4 ± 2.8 | 10.8 ± 1.1 | 17.5 ± 1.3 |
| **Macro F1** | | | **~26.9** |

**Our reproduced results:** Not available.

---

### POI-RGNN

| Attribute | Detail |
|-----------|--------|
| **Source** | Capanema et al. — *"Combining Recurrent and Graph Neural Networks to Predict the Next Place's Category"*, preprint submitted to Ad Hoc Networks (extended journal version, 2022) |
| **Short paper** | Capanema et al. — PE-WASUN '21, ACM (18th Symposium on Performance Evaluation of Wireless Ad Hoc, Sensor, & Ubiquitous Networks, Alicante, November 2021) |
| **PDF** | `Combining_Recurrent_and_Graph_Neural_Networks_to_Predict_the_Next_Place's_Category.pdf` |
| **Task** | Next-category prediction from check-in sequence + user mobility graph |
| **Reason for inclusion** | Direct predecessor work; shares the same Gowalla/Foursquare dataset family and 7-category taxonomy |

**Architecture:**
- **Recurrent component**: GRU over a sequence of (user, category, hour, distance, time-interval) embeddings. A Multi-Head Attention sublayer re-weights the GRU hidden states. The sequential embedding includes a learnable product term `V₁ × dist_prev × dur_prev`.
- **Graph component**: Three independent 2-layer GCN blocks applied to the adjacency matrix `Ac`, distance matrix `Dc`, and duration matrix `Tc`; outputs are `Hd`, `Ht`, `Hdt = GCN(V₂ × Dc × Tc)`.
- **RGNN Ensemble layer** (journal extension): Three intermediate softmax predictions (recurrent, graph, combined) are entropy-weighted and summed; a trained variable subtracts the current-category embedding to discourage revisits.
- **Category-Aware output**: `Y = YRNN·w₁ + YGNN·w₂ + YRGNN·w₃ − V₃ · Yc`.

#### Paper-reported results — Gowalla (global, not state-split)

| Category | Precision (%) | Recall (%) | F1 (%) |
|----------|--------------|-----------|--------|
| Shopping | 42.0 ± 0.4 | 49.7 ± 1.4 | 45.5 ± 0.4 |
| Community | 39.8 ± 0.9 | 32.4 ± 0.9 | 35.7 ± 0.5 |
| Food | 42.8 ± 0.3 | 60.8 ± 1.3 | 50.3 ± 0.5 |
| Entertainment | 44.5 ± 2.0 | 15.2 ± 0.4 | 22.7 ± 0.6 |
| Travel | 37.8 ± 2.1 | 12.3 ± 0.6 | 18.5 ± 0.6 |
| Outdoors | 41.7 ± 2.3 | 18.9 ± 1.2 | 26.0 ± 1.3 |
| Nightlife | 44.1 ± 1.3 | 20.4 ± 1.1 | 27.9 ± 0.8 |
| **Macro F1** | | | **~32.4** |

#### Our reproduced results — per US state (mean ± std, 5 folds)

**California:**

| Category | Precision (%) | Recall (%) | F1 (%) |
|----------|--------------|-----------|--------|
| Shopping | 43.58 ± 0.99 | 43.79 ± 2.86 | 43.65 ± 1.74 |
| Community | 41.35 ± 0.90 | 31.34 ± 2.19 | 35.61 ± 1.51 |
| Food | 43.27 ± 0.61 | 65.96 ± 2.52 | 52.23 ± 0.67 |
| Entertainment | 35.76 ± 3.88 | 9.20 ± 1.17 | 14.56 ± 1.49 |
| Travel | 48.54 ± 1.17 | 40.03 ± 0.88 | 43.86 ± 0.68 |
| Outdoors | 35.12 ± 1.53 | 14.20 ± 2.68 | 20.14 ± 2.97 |
| Nightlife | 35.05 ± 2.39 | 7.58 ± 1.48 | 12.41 ± 2.09 |
| **Macro F1** | | | **31.78** |

**Florida:**

| Category | Precision (%) | Recall (%) | F1 (%) |
|----------|--------------|-----------|--------|
| Shopping | 41.56 ± 0.71 | 53.59 ± 3.17 | 46.78 ± 1.39 |
| Community | 38.74 ± 1.21 | 27.44 ± 2.32 | 32.06 ± 1.69 |
| Food | 40.43 ± 0.24 | 45.62 ± 2.13 | 42.84 ± 0.89 |
| Entertainment | 40.03 ± 1.34 | 16.58 ± 1.10 | 23.42 ± 1.12 |
| Travel | 55.77 ± 1.26 | 71.25 ± 0.75 | 62.56 ± 1.01 |
| Outdoors | 37.55 ± 2.44 | 13.04 ± 3.01 | 19.15 ± 3.38 |
| Nightlife | 37.14 ± 2.08 | 9.17 ± 2.42 | 14.59 ± 3.30 |
| **Macro F1** | | | **34.49** |

**Texas:**

| Category | Precision (%) | Recall (%) | F1 (%) |
|----------|--------------|-----------|--------|
| Shopping | 43.10 ± 0.64 | 49.28 ± 2.45 | 45.96 ± 1.31 |
| Community | 40.28 ± 0.62 | 32.96 ± 1.70 | 36.23 ± 1.11 |
| Food | 42.46 ± 0.33 | 61.41 ± 1.99 | 50.19 ± 0.72 |
| Entertainment | 42.28 ± 1.42 | 19.76 ± 1.68 | 26.91 ± 1.73 |
| Travel | 37.34 ± 3.04 | 15.52 ± 1.65 | 21.90 ± 2.06 |
| Outdoors | 37.23 ± 1.65 | 17.44 ± 1.67 | 23.71 ± 1.69 |
| Nightlife | 38.68 ± 1.46 | 20.00 ± 2.41 | 26.29 ± 2.22 |
| **Macro F1** | | | **33.03** |

---

## Task 2 — POI Category Labeling

Classifies the **category** of a POI from its intrinsic features (spatial, graph-structural, or embedding-based), without relying on user visit sequences.  
Evaluation: macro-averaged F1 over 7 categories, 5-fold cross-validation.

---

### PGC (Prediction of General Categories)

| Attribute | Detail |
|-----------|--------|
| **Source** | Capanema et al. — *"Combining Recurrent and Graph Neural Networks to Predict the Next Place's Category"*, preprint submitted to Ad Hoc Networks (2022) |
| **PDF** | `Combining_Recurrent_and_Graph_Neural_Networks_to_Predict_the_Next_Place's_Category.pdf` |
| **Task** | POI category labeling from raw GPS traces (no check-in labels required at inference) |
| **Reason for inclusion** | Closest direct competitor for the category-labeling task; uses ARMA GNNs on the same mobility graph structure used by the project |

**Architecture:**
- Builds a **mobility graph** from raw GPS data under two perspectives: *individual* (user-specific transitions) and *collective* (population-level co-occurrence), split further into weekday/weekend sub-graphs.
- **ARMA GNN layers** (Autoregressive Moving Average graph convolution) applied to each graph view; dropout 0.3 between layers.
- Node embeddings from all views are aggregated and passed through a linear classifier.
- Uses **transfer learning**: trains the graph encoder on a labelled source dataset then fine-tunes on target data where POI labels may be sparse or absent.

#### Paper-reported results — Gowalla (global, not state-split)

| Category | Precision (%) | Recall (%) | F1 (%) |
|----------|--------------|-----------|--------|
| Shopping | 43.9 ± 1.5 | 40.7 ± 2.0 | 42.2 ± 1.7 |
| Community | 53.4 ± 1.0 | 60.1 ± 1.6 | 56.5 ± 0.7 |
| Food | 51.1 ± 1.4 | 70.5 ± 1.0 | 59.2 ± 1.0 |
| Entertainment | 82.2 ± 4.0 | 47.4 ± 3.4 | 60.0 ± 1.8 |
| Travel | 77.5 ± 0.9 | 60.3 ± 2.2 | 67.8 ± 1.5 |
| Outdoors | 25.6 ± 41.9 | 2.3 ± 3.7 | 4.3 ± 1.1 |
| Nightlife | 57.5 ± 2.0 | 52.0 ± 4.9 | 54.6 ± 3.1 |
| **Macro F1** | | | **~49.2** |

#### Paper-reported results — state-split Gowalla (from HAVANA paper, directly comparable)

| Category | Florida F1 (%) | California F1 (%) | Texas F1 (%) |
|----------|---------------|------------------|-------------|
| Shopping | 44.47 ± 5.2 | 17.68 ± 3.3 | 24.78 ± 12.6 |
| Community | 11.92 ± 7.6 | 19.08 ± 1.4 | 15.00 ± 4.3 |
| Food | 62.99 ± 0.8 | 59.99 ± 0.6 | 63.78 ± 2.2 |
| Entertainment | 71.69 ± 1.9 | 50.44 ± 8.1 | 55.79 ± 8.5 |
| Travel | 70.53 ± 1.0 | 60.44 ± 1.9 | 61.23 ± 1.1 |
| Outdoors | 52.58 ± 1.6 | 32.08 ± 7.9 | 39.91 ± 5.9 |
| Nightlife | 38.20 ± 4.0 | 18.28 ± 4.3 | 63.07 ± 2.9 |
| **Macro F1** | **~50.3** | **~36.9** | **~46.2** |

#### Our reproduced results — per US state (mean ± std, 5 folds)

**California:**

| Category | Precision (%) | Recall (%) | F1 (%) |
|----------|--------------|-----------|--------|
| Shopping | 47.35 ± 5.54 | 11.23 ± 2.01 | 18.07 ± 2.76 |
| Community | 63.31 ± 7.69 | 5.06 ± 2.32 | 9.26 ± 3.99 |
| Food | 41.91 ± 0.30 | 92.98 ± 1.04 | 57.77 ± 0.22 |
| Entertainment | 76.07 ± 4.52 | 29.32 ± 7.90 | 41.40 ± 7.87 |
| Travel | 86.19 ± 1.76 | 56.09 ± 0.65 | 67.95 ± 0.94 |
| Outdoors | 71.79 ± 7.11 | 13.67 ± 2.69 | 22.91 ± 4.17 |
| Nightlife | 45.24 ± 2.72 | 11.43 ± 3.06 | 18.12 ± 3.97 |
| **Macro F1** | | | **33.64** |

**Florida:**

| Category | Precision (%) | Recall (%) | F1 (%) |
|----------|--------------|-----------|--------|
| Shopping | 39.60 ± 2.92 | 25.64 ± 5.58 | 30.65 ± 3.11 |
| Community | 46.71 ± 4.95 | 2.74 ± 0.83 | 5.16 ± 1.49 |
| Food | 40.29 ± 1.37 | 79.12 ± 3.07 | 53.33 ± 0.60 |
| Entertainment | 77.26 ± 2.59 | 43.43 ± 3.29 | 55.47 ± 2.47 |
| Travel | 84.01 ± 1.30 | 77.99 ± 1.48 | 80.88 ± 1.11 |
| Outdoors | 75.15 ± 4.04 | 32.35 ± 1.55 | 45.20 ± 1.90 |
| Nightlife | 49.51 ± 5.21 | 8.78 ± 2.47 | 14.83 ± 3.67 |
| **Macro F1** | | | **40.79** |

**Texas:**

| Category | Precision (%) | Recall (%) | F1 (%) |
|----------|--------------|-----------|--------|
| Shopping | 40.77 ± 2.81 | 40.43 ± 3.60 | 40.36 ± 1.17 |
| Community | 58.52 ± 5.81 | 6.23 ± 0.45 | 11.25 ± 0.80 |
| Food | 46.33 ± 0.72 | 79.08 ± 3.05 | 58.39 ± 0.44 |
| Entertainment | 74.42 ± 5.41 | 38.87 ± 3.55 | 50.76 ± 1.80 |
| Travel | 70.81 ± 2.66 | 32.38 ± 1.71 | 44.36 ± 1.27 |
| Outdoors | 78.17 ± 6.80 | 18.12 ± 1.87 | 29.28 ± 2.27 |
| Nightlife | 67.23 ± 3.15 | 47.11 ± 3.01 | 55.30 ± 2.02 |
| **Macro F1** | | | **41.39** |

---

### HAVANA

| Attribute | Detail |
|-----------|--------|
| **Source** | Santos et al. — *"HAVANA: Hybrid Attentional Graph Convolutional Network Semantic Venue Annotation Model"* |
| **PDF** | `HAVANA.pdf` |
| **Task** | Semantic venue annotation — assigning category labels to POIs using spatial and spectral graph convolution |
| **Reason for inclusion** | Representative hybrid GNN approach for POI labeling; tested on the same Gowalla state splits (FL, CA, TX); strongest reported baseline for the labeling task |

**Architecture:**
- **Hybrid convolution block**: combines a *spatial* branch (Graph Attention Network — GAT) and a *spectral* branch (ARMA graph convolution).
- Outputs of both branches are fused via **self-attention** with a two-layer feedforward network and residual connection.
- **Multi-view learning**: integrates individual mobility graphs and collective mobility graphs for richer node representations.
- Final node embeddings are classified into venue categories.

#### Paper-reported results — state-split Gowalla (directly comparable)

**Florida:**

| Category | HAVANA F1 (%) | PGC-NN F1 (%) | STPA F1 (%) | k-FN F1 (%) |
|----------|--------------|--------------|------------|------------|
| Shopping | 58.30 ± 1.0 | 44.47 ± 5.2 | 53.83 ± 3.6 | 23.28 ± 0.2 |
| Community | 30.20 ± 2.0 | 11.92 ± 7.6 | 42.94 ± 13.1 | 13.95 ± 0.4 |
| Food | 70.47 ± 0.6 | 62.99 ± 0.8 | 41.92 ± 1.1 | 28.94 ± 0.3 |
| Entertainment | 81.93 ± 0.9 | 71.69 ± 1.9 | 30.50 ± 8.6 | 9.45 ± 0.2 |
| Travel | 77.47 ± 0.9 | 70.53 ± 1.0 | 33.64 ± 7.2 | 11.27 ± 0.4 |
| Outdoors | 61.48 ± 1.6 | 52.58 ± 1.6 | 34.45 ± 8.3 | 6.27 ± 0.3 |
| Nightlife | 60.71 ± 2.0 | 38.20 ± 4.0 | 23.54 ± 7.7 | 5.71 ± 0.4 |
| **Macro F1** | **~62.9** | **~50.3** | **~37.3** | **~14.1** |

**California:**

| Category | HAVANA F1 (%) | PGC-NN F1 (%) | STPA F1 (%) | k-FN F1 (%) |
|----------|--------------|--------------|------------|------------|
| Shopping | 32.98 ± 0.8 | 17.68 ± 3.3 | 51.07 ± 7.1 | 21.69 ± 0.2 |
| Community | 23.63 ± 1.7 | 19.08 ± 1.4 | 43.32 ± 9.5 | 14.77 ± 0.1 |
| Food | 62.23 ± 0.1 | 59.99 ± 0.6 | 44.60 ± 4.1 | 34.13 ± 0.1 |
| Entertainment | 57.55 ± 1.0 | 50.44 ± 8.1 | 26.92 ± 8.4 | 6.08 ± 0.1 |
| Travel | 63.93 ± 0.8 | 60.44 ± 1.9 | 25.53 ± 10.9 | 9.85 ± 0.2 |
| Outdoors | 39.32 ± 1.5 | 32.08 ± 7.9 | 36.39 ± 9.7 | 7.72 ± 0.1 |
| Nightlife | 48.89 ± 0.6 | 18.28 ± 4.3 | 19.24 ± 5.6 | 5.28 ± 0.3 |
| **Macro F1** | **~46.9** | **~36.9** | **~35.3** | **~14.2** |

**Texas:**

| Category | HAVANA F1 (%) | PGC-NN F1 (%) | STPA F1 (%) | k-FN F1 (%) |
|----------|--------------|--------------|------------|------------|
| Shopping | 50.77 ± 0.9 | 24.78 ± 12.6 | 50.51 ± 3.8 | 22.73 ± 0.1 |
| Community | 29.46 ± 1.7 | 15.00 ± 4.3 | 49.58 ± 6.7 | 17.39 ± 0.2 |
| Food | 72.56 ± 0.4 | 63.78 ± 2.2 | 43.58 ± 4.8 | 32.60 ± 0.1 |
| Entertainment | 67.02 ± 0.5 | 55.79 ± 8.5 | 30.42 ± 7.7 | 8.36 ± 0.1 |
| Travel | 66.49 ± 1.3 | 61.23 ± 1.1 | 24.03 ± 10.0 | 5.71 ± 0.2 |
| Outdoors | 54.08 ± 0.8 | 39.91 ± 5.9 | 31.94 ± 9.0 | 5.42 ± 0.2 |
| Nightlife | 77.95 ± 0.7 | 63.07 ± 2.9 | 24.38 ± 5.0 | 7.40 ± 0.2 |
| **Macro F1** | **~59.8** | **~46.2** | **~36.3** | **~14.2** |

#### Our reproduced results — per US state (mean ± std, 5 folds)

**Florida:**

| Category | Precision (%) | Recall (%) | F1 (%) |
|----------|--------------|-----------|--------|
| Shopping | 46.38 ± 0.91 | 61.73 ± 2.49 | 52.93 ± 0.58 |
| Community | 69.71 ± 2.44 | 11.47 ± 1.21 | 19.66 ± 1.73 |
| Food | 55.56 ± 0.89 | 70.99 ± 1.55 | 62.31 ± 0.29 |
| Entertainment | 80.53 ± 2.39 | 66.04 ± 1.50 | 72.52 ± 0.17 |
| Travel | 85.40 ± 2.39 | 60.25 ± 1.67 | 70.61 ± 0.73 |
| Outdoors | 75.51 ± 3.50 | 36.62 ± 2.10 | 49.22 ± 1.33 |
| Nightlife | 66.45 ± 2.20 | 43.29 ± 1.81 | 52.37 ± 1.08 |
| **Macro F1** | | | **54.23** |

---

## Notes

- All runs use 5-fold stratified cross-validation; reproduced results are mean ± std over folds.
- RGNN and PGC are from the same research group and paper family (Capanema et al.); PGC addresses labeling, RGNN addresses sequence prediction.
- The HAVANA paper is the primary source for state-level (FL, CA, TX) PGC numbers — it independently reproduced PGC-NN on the same data splits.
- Our reproduced PGC numbers are lower than the HAVANA-reported PGC numbers on most categories, likely due to differences in data preprocessing, graph construction, or hyperparameters.
- MHA+PE lacks reproduced results; it serves as an architectural reference point and lower bound for the next-prediction task.
