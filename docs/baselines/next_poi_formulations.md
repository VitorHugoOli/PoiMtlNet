# next_poi_formulations.md — per-paper task-formulation cards

> Companion to [`RELATED_WORK_TRIAGE.md`](RELATED_WORK_TRIAGE.md) (read that first for the synthesis).
> One card per paper: the formal task definition **as quoted from the primary source**, what the output
> layer actually predicts, input representation, vocabulary/metrics, and the role of category/region.
> All quotes were extracted from primary sources and survived 3-vote adversarial verification
> (2026-07-16) unless marked otherwise. Use these cards when writing related-work prose or answering
> reviewer questions about task framing.

---

## FPMC — Rendle & Freudenthaler, WWW 2010

- **Paper:** *Factorizing Personalized Markov Chains for Next-Basket Recommendation.*
  [PDF](https://www.ismll.uni-hildesheim.de/pub/pdfs/RendleFreudenthaler2010-FPMC.pdf)
- **Formal task:** *"the item recommendation task can be formalized in creating a personal ranking
  <_{u,t} ⊂ I² over all pairs of items for user u for his t-th basket. With this ranking, we can
  recommend the user the top n items."* Items I = {i₁,…,i_|I|} are individual items.
- **Output:** score x̂_{u,t,i} per item, pairwise ranking (S-BPR) over the factorized personalized
  Markov tensor A ∈ [0,1]^{|U|×|I|×|I|}.
- **Metrics:** HLU, Precision/Recall/F-measure@5, AUC — on item rank.
- **Category/region:** absent entirely (not even auxiliary).
- **Caveat:** original domain is retail next-basket; later next-POI work adopted the formulation with
  POIs as items. The "thousands of venues" framing belongs to its successors.

## ST-RNN — Liu et al., AAAI 2016

- **Paper:** *Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts.*
  [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/9971)
- **Formal task:** *"Let P be a set of users and Q be a set of locations… the task is to predict where
  a user will go next at a specific time t."*
- **Output:** score o_{u,t,v} = (h + p_u)ᵀ q_v per candidate location v ∈ Q, trained with BPR.
- **Input:** (location, timestamp) pairs through an RNN with time/distance-bin transition matrices.
- **Metrics:** Recall@k, F1@k, MAP, AUC on Gowalla. Baselines (TOP, MF, MC, TF, FPMC, FPMC-LR, PRME,
  RNN) all rank location IDs.
- **Category/region:** absent in the LBSN task. (A secondary experiment on the Global Terrorism
  Database predicts province/state — not a check-in/mobility setting; do not cite as region-target
  precedent.)

## DeepMove — Feng et al., WWW 2018

- **Paper:** *DeepMove: Predicting Human Mobility with Attentional Recurrent Networks.*
  [PDF](https://vonfeng.github.io/files/WWW2018_DeepMove.pdf)
- **Formal task (Problem 1):** *"the mobility prediction is simplified to predict the next location
  identification l"* — a spatiotemporal point is (time, location-ID).
- **Output:** *"soft-max layer with negative sampling"*; multi-class cross-entropy;
  *"the size of the location candidate set can also be up to ten thousand."*
- **Input:** dense embeddings of (location, time-of-day, user); "region" mentioned once, informally,
  as a possible input feature — never formalized, never a target.
- **Vocab / metrics:** 43,379 / 31,522 / 17,785 locations (its three datasets); top-1/top-5 accuracy.
- **Category/region:** absent; §6 declares incorporating semantic *"point of interests"* as future work.

## Flashback — Yang et al., IJCAI 2020

- **Paper:** *Location Prediction over Sparse User Mobility Traces Using RNNs: Flashback in Hidden
  States.* [IJCAI](https://www.ijcai.org/proceedings/2020/302) ·
  [code](https://github.com/eXascaleInfolab/Flashback_code)
- **Formal task:** given POI sequence {…, p_{i-1}, p_i}, weight past hidden states by spatiotemporal
  relevance (ΔT, ΔD) and predict p_{i+1} (Figure 3 labels the output as p_{i+1}).
- **Output (code-level evidence):** `self.fc = nn.Linear(2*hidden_size, input_size)` where
  `input_size` = number of distinct locations — discrete classification over the full location
  vocabulary.
- **Input:** POI ID + Δtime/Δdistance + user embedding. **No category or region feature anywhere**
  (only decorative icons in a concept figure).
- **Vocab / metrics:** 121,851 POIs (Gowalla), ~69,005 (Foursquare); Acc@1/5/10, MRR.

## LSTPM — Sun et al., AAAI 2020

- **Paper:** *Where to Go Next: Modeling Long- and Short-Term User Preferences for Point-of-Interest
  Recommendation.* [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/5353)
- **Formal task (§3):** given geo-coded POI set L = {l₁,…,l_|L|} and the user's trajectories,
  *"the goal is to recommend the top-N preferable POIs to user u at the next timestamp t."*
- **Output:** p = softmax(W_p(s ⊕ h)), W_p ∈ ℝ^{|L|×2d} — probability over every venue identity.
- **Vocab / metrics:** 9,296 POIs (FSQ), 40,868 (Gowalla); Recall@K, NDCG@K, K ∈ {1,5,10}.
- **Category/region:** unused — and the paper *removes* the category context from the TMCA baseline
  *"for fairness because no other methods make use of it"*. Direct evidence that category is treated
  as optional side information in this lineage.

## STAN — Luo et al., WWW 2021 — 📁 repo-audited

- **Paper:** *STAN: Spatio-Temporal Attention Network for Next Location Recommendation.*
  [arXiv:2102.04095](https://arxiv.org/abs/2102.04095)
- **Formal task:** next location recommendation — layer 2 ("matching") ranks **candidate POIs** via
  attention from candidate-POI embeddings to trajectory states.
- **Input:** multimodal embedding e_loc + e_user + e_time(hour-of-week) + pairwise Δt/Δd bias.
- **Category/region:** input side only.
- **Note:** this card comes from the repo's own two-agent faithfulness audit, not the web rounds —
  full architecture/audit detail in [`next_region/stan.md`](next_region/stan.md).

## GETNext — Yang et al., SIGIR 2022

- **Paper:** *GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation.*
  [arXiv:2303.04741](https://arxiv.org/abs/2303.04741)
- **Formal task:** given historic + current trajectories of a user, *"predict the most likely future
  POIs q_{m+1},…,q_{m+k}"* over P = {p₁,…,p_N}.
- **Output:** MLP head in ℝ^{1×N} — one score per POI in the fixed vocabulary.
- **The category head is auxiliary — the field's clearest statement:** *"The category head is employed
  to regulate next POI prediction as forecasting the next POI category is easier than exact POI
  prediction."* Loss: L = L_poi + 10·L_time + L_cat; evaluation stays on POI-ID.
- **Sparsity numbers (citable motivation for category-level targets):** NYC ≈ 227k check-ins over
  ~38k POIs (~6 check-ins/POI) vs ~400 categories (~570 check-ins/category).
- **Metrics:** Acc@1/5/10/20 + MRR, uniformly for GETNext and all baselines (MF, FPMC, LSTM, PRME,
  ST-RNN, STGN, STGCN, PLSPL, STAN).

## HMT-GRN — Lim et al., SIGIR 2022 — 📁 repo-audited

- **Paper:** *Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation.*
- **Formal task:** multi-task learning over next-POI + next-region at several geohash granularities;
  Hierarchical Beam Search over the region hierarchy prunes the POI candidate space. The **evaluated
  end target is next-POI**; regions are auxiliary tasks against sparsity.
- **Why it matters to us:** the sole *region-native* published architecture — our region-adapted E2E
  variant is the paper's primary external baseline; results + deviation ledger in
  [`next_region/hmt_grn.md`](next_region/hmt_grn.md).

## Graph-Flashback — Rao et al., KDD 2022

- **Paper:** *Graph-Flashback Network for Next Location Recommendation.*
  [PDF](https://www.atailab.cn/seminar2022Fall/pdf/2022_KDD_Graph-Flashback%20Network%20for%20Next%20Location%20Recommendation.pdf) ·
  [code](https://github.com/kevin-xuan/Graph-Flashback)
- **Formal task (Definition 4):** *"the next POI recommendation problem aims to output the top-k POIs
  that user u_i will be most likely to visit next."* The spatiotemporal knowledge graph (Def. 3) is
  input-side representation enrichment.
- **Output (Eq. 13-14):** ŷ_t^u = W_f[ĥ_t ‖ e^u], W_f ∈ ℝ^{|L|×2d}; softmax cross-entropy over L.
- **Vocab / metrics:** 106,994 (Gowalla) / 68,879 (FSQ) POIs; Acc@1/5/10, MRR.
- **Category/region:** category appears exactly once — in the conclusion, as **future work**
  (*"we will consider other side information (e.g., POI categories…)"*). Not even auxiliary.

## SNPM — AAAI 2023

- **Paper:** *[Similar-Neighborhood-based next-POI model]* —
  [AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/view/25608/25380)
- **Formal definitions:** Definition 1: *"A check-in is represented by c = (u, ℓ, t)"*; Definition 3:
  *"recommend a list of top-ranked POIs"* — each POI ℓᵢ with its own (lat, lon).
- **Vocab / metrics:** 106,994 POIs (Gowalla), 68,879 (Foursquare); Acc@k, MRR
  (rank_i = index of the true POI in the ordered list).

## STHGCN — Yan et al., SIGIR 2023

- **Paper:** *Spatio-Temporal Hypergraph Learning for Next POI Recommendation.*
  [code](https://github.com/alipay/Spatio-Temporal-Hypergraph-Model)
- **Formal task (§3.1):** *"the task is to let the model rank the visiting confidence among all POIs
  ŷ_i (i = 0, 1, …, |P|) and predict the POIs that the user incline to visit next."*
- **Input:** check-in tuple q = ⟨u, p, c, g, t⟩ — category c is an input attribute; the hypergraph has
  Category **nodes** ("Belonging" edges) for representation building only.
- **Output (code-level evidence):** `self.linear = nn.Linear(self.checkin_embed_size, self.num_poi)` +
  `nn.CrossEntropyLoss()` on POI logits. **No category head, no auxiliary category loss anywhere.**
- **Vocab / metrics:** FSQ NYC/TKY/CA; categories in data: 318/290/296 (input-only); Acc@1/5/10, MRR.

## Multi-channel framework — arXiv 2209.00472 (TOIS-track)

- **Paper:** *A Multi-Channel Next POI Recommendation Framework with Multi-Granularity Check-in
  Signals.* [arXiv](https://arxiv.org/pdf/2209.00472)
- **Formal task:** *"predict POI l_{t_{k+1}}"*; check-in (u, l, t, r, c, g) ties POI l to region r,
  category c, coords g — the tuple itself shows POI ≠ region ≠ category.
- **Category/region role:** **auxiliary channels** — region- and category-level signals densify sparse
  POI transitions; final evaluation on POI-ID. The closest mainstream use of *region* as a modeling
  signal, still subordinated.
- **Vocab / metrics:** 3,013–56,252 POIs; HR@5/10, NDCG@5/10.

## LoTNext — arXiv 2410.14970 (2024)

- **Paper:** *[Long-tail next-POI transformer]* — [arXiv](https://arxiv.org/html/2410.14970)
- **Formal task:** *"our goal is to predict a list of top POIs that the user u is likely to visit next,
  which can be taken as a typical sequence classification task over |P| POI candidates."*
- **Output:** logits ℝ^{n×|P|} with long-tail logit adjustment; softmax over all candidates.
- **Metrics:** Acc@1/5/10 + MRR.
- **Note:** venue not independently verified in our rounds; cite by arXiv ID.

## UniMove — Han et al., SIGSPATIAL 2025

- **Paper:** *UniMove: A Unified Model for Multi-city Human Mobility Prediction.*
  [arXiv:2508.06986](https://arxiv.org/abs/2508.06986)
- **Formal task (§3.1-3.2):** *"A location is a grid area within a city, uniquely identified by a
  specific location ID"* (500 m × 500 m cells); predict loc_{H+1} = F(S_{1:H}).
- **Output:** probability over the city's grid-cell vocabulary (Location Tower × Trajectory Tower);
  cross-entropy on cell IDs. Per-city vocabularies: 5,451 (Shanghai), 2,055 (Nanchang), 166 (Lhasa).
- **Category/region role:** 28 POI-derived features + popularity rank characterize each *cell* as
  input; never a target.
- **Why it matters:** the one structural outlier — venue → grid-cell relaxation in a 2025
  foundation-model-style work. Still discrete-location-ID logic, NOT semantic category or
  administrative region. (Two fine-grained output-layer claims were refuted in verification; treat
  architecture details beyond the above as uncertain. Proceedings status in ACM DL unconfirmed.)

## Zone relaxation — Amichi et al., SIGSPATIAL 2021

- **Paper:** [PDF](https://www.cs.bu.edu/faculty/crovella/paper-archive/sigspatial21-mobility-prediction.pdf)
- **What it does:** for **exploration events only** (visits to never-seen places), predictors *"adapt
  their prediction to forecast a zone (large cells) instead of a location"* — grid cells 800 m–4 km.
- **The citable motivation:** *"the prediction accuracy of the MC Location-Predictor in forecasting
  explorations is equal to zero."*
- **Why it matters:** the verified precedent for coarsening the spatial target when exact-place is
  unlearnable — conditional (exploration only), spatial-grid (not administrative regions), but the
  same direction as our next-region choice.

## Survey — Zhang et al., IEEE TKDE 2025 (arXiv 2410.02191)

- **Paper:** *A Survey on Point-of-Interest Recommendation: Models, Architectures, and Security.*
- **Task definition:** *"Given a user's check-in list C^u, the next POI recommendation task revolves
  around predicting the subsequent POI… recommending the immediate next POI."* POI = *"a particular
  location or site… restaurants, hotels"* with own coordinates.
- **Taxonomy:** ~60 models under "Next POI Recommendation"; **no parallel task category for
  next-category or next-region** (absence-of-evidence — medium confidence, single survey).
- Also triaged as a citation candidate in
  [`articles/[mobiwac]/RELATED_WORK_TRIAGE.md §2.1`](../../articles/[mobiwac]/RELATED_WORK_TRIAGE.md).
