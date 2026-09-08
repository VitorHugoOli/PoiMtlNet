# RELATED_WORK_TRIAGE.md — how the literature defines "next-POI prediction" (task-formulation triage)

> **What this is.** The consolidated answer to "*is 'given a sequence of check-ins, predict the next POI'
> the canonical task in the literature — and what exactly is a POI there?*". Produced from **two
> adversarially-verified deep-research rounds (2026-07-16)**: round 1 (20 sources fetched, 25 claims
> 3-vote verified, 18 confirmed) established the canonical formulation; round 2 (24 sources, 23/25
> confirmed) audited the seminal lineage paper-by-paper (FPMC → UniMove). Per-paper detail cards with
> exact quotes live in [`next_poi_formulations.md`](next_poi_formulations.md).
>
> **Not the same doc as** [`articles/[mobiwac]/RELATED_WORK_TRIAGE.md`](../../articles/[mobiwac]/RELATED_WORK_TRIAGE.md)
> — that one is a *citation* triage of shared refs against the MobiWac draft. This one is the
> *task-formulation* reference for the whole baselines folder.

## 1 · Bottom line

1. **Yes — categorically.** "Given a user's past check-in sequence, predict the exact next POI" is THE
   dominant, canonical framing. Every one of the 14 works examined (2010–2025) defines the task this way;
   no counter-example was found.
2. **"POI" = a unique venue identity** — one discrete ID from a fixed vocabulary P = {p₁,…,p_N}
   (N ≈ 3,000–122,000 depending on dataset), each with its own coordinates. Category and coordinates are
   *attributes of* the POI, never the prediction target.
3. **The task is ranking/retrieval, not few-class classification.** Standard evaluation is Acc@k / HR@k /
   Recall@k / NDCG@k (k = 1, 5, 10, 20) + MRR over the position of the true POI in a ranked candidate list
   drawn from the full vocabulary.
4. **Category/region appear only as (a) input features or (b) auxiliary losses** in service of the
   POI-ID objective — never as end targets in this lineage. Several papers make the subordination
   explicit (GETNext, LSTPM, Graph-Flashback — quotes in §4).
5. **Our two tasks (next-category over 7 classes; next-region over census tracts) are therefore a
   deliberate departure from the mainstream**, not a variant of it. The nearest published neighbors
   (§5) all keep exact-place as the end target.

## 2 · The canonical formulation

Formally (composited from GETNext/SIGIR'22, SNPM/AAAI'23, and the TKDE'25 survey — exact quotes in the
detail cards):

- **Check-in** = tuple *(user u, POI ℓ, timestamp t)*; richer variants attach category/region/coords as
  attributes: *p = (id, loc, cate)* (ICDE'24), *q = ⟨u, p, c, g, t⟩* (STHGCN). "Next check-in" and
  "next POI" are used interchangeably — the predicted unit is the venue ID of the next check-in.
- **Task**: given the user's check-in sequence, output a **ranked list over all N POIs**; the model's
  output layer is literally a score vector of size N (e.g. GETNext's MLP head in ℝ^{1×N}; Flashback's
  `nn.Linear(2·hidden, n_locations)`; STHGCN's `nn.Linear(embed, num_poi)`).
- **Loss**: softmax cross-entropy over the vocabulary (deep models) or pairwise ranking (BPR family:
  FPMC, ST-RNN).
- **Metrics**: Acc@k/Recall@k/HR@k + MRR (occasionally NDCG@k) on the rank of the true venue.

## 3 · Master table — 14 works, one line each

Rows marked 🔎 were verified from primary sources in the deep-research rounds; rows marked 📁 come from
this repo's own audited baseline docs (linked).

| # | Work (venue, year) | Predicts (output layer) | Vocab size | Metrics | Category/region role |
|---|---|---|---|---|---|
| 🔎 1 | **FPMC** (WWW 2010) | item ID, pairwise ranking (S-BPR) | retail items (pre-POI; formulation later adapted) | HLU, P/R/F@5, AUC | absent entirely |
| 🔎 2 | **ST-RNN** (AAAI 2016) | location ID, BPR score per candidate | Gowalla locations | Recall@k, F1@k, MAP, AUC | absent (region only in a non-LBSN side dataset) |
| 🔎 3 | **DeepMove** (WWW 2018) | location ID, softmax + neg. sampling | 17,785–43,379 | top-1/top-5 acc | absent; POI semantics = declared future work |
| 🔎 4 | **Flashback** (IJCAI 2020) | POI ID, FC layer sized to vocab | 121,851 (Gow) / ~69k (FSQ) | Acc@1/5/10, MRR | absent entirely |
| 🔎 5 | **LSTPM** (AAAI 2020) | POI ID, softmax over \|L\| | 9,296 (FSQ) / 40,868 (Gow) | Recall@K, NDCG@K | absent; *removes* category from a baseline "for fairness" |
| 📁 6 | **STAN** (WWW 2021) | POI ID, candidate-matching attention layer | FSQ/Gowalla scale | Acc@k | input embedding only ([`next_region/stan.md`](next_region/stan.md)) |
| 🔎 7 | **GETNext** (SIGIR 2022) | POI ID, MLP head ℝ^{1×N} | ~38k POIs (NYC) | Acc@1/5/10/20, MRR | **auxiliary loss** (L = L_poi + 10·L_time + L_cat) |
| 📁 8 | **HMT-GRN** (SIGIR 2022) | POI ID via beam search over region hierarchy | FSQ/Gowalla scale | Acc@k | **auxiliary multi-task heads** (geohash regions) serving next-POI ([`next_region/hmt_grn.md`](next_region/hmt_grn.md)) |
| 🔎 9 | **Graph-Flashback** (KDD 2022) | POI ID, W_f ∈ ℝ^{\|L\|×2d} | 106,994 / 68,879 | Acc@1/5/10, MRR | not even auxiliary — listed as future work |
| 🔎 10 | **SNPM** (AAAI 2023) | POI ID, ranked list | 106,994 / 68,879 | Acc@k, MRR | input attribute |
| 🔎 11 | **STHGCN** (SIGIR 2023) | POI ID, `Linear(embed, num_poi)` | FSQ-NYC/TKY/CA (cats: 290–318, input-only) | Acc@1/5/10, MRR | input feature (hypergraph node type) |
| 🔎 12 | **Multi-channel** (arXiv 2209.00472, TOIS-track) | POI ID | 3,013–56,252 | HR@5/10, NDCG@5/10 | **auxiliary region + category channels** (densify sparse POI signal) |
| 🔎 13 | **LoTNext** (arXiv 2410.14970, 2024) | POI ID, logits ℝ^{n×\|P\|} (long-tail adjusted) | FSQ/Gowalla scale | Acc@1/5/10, MRR | auxiliary/input |
| 🔎 14 | **UniMove** (SIGSPATIAL 2025) | **grid-cell ID** (500 m), per-city vocab | 166–5,451 cells/city | Acc@k | POI stats = input features of the cell |

**Read:** 14/14 predict a discrete *location identity* (venue ID, or grid-cell ID for UniMove). 0/14
predict category or region as the end target. The only structural outlier is UniMove — coarser spatial
unit, same "discrete-ID from fixed vocabulary" logic.

## 4 · How category/region appear in the mainstream (the three roles)

1. **Input attribute** (most common): category/coords ride along in the check-in tuple and enrich the
   representation (STHGCN's hypergraph "Belonging" edges; STAN's multimodal embedding; SNPM, LoTNext).
2. **Auxiliary loss / auxiliary channel**: a secondary head regularizes the POI-ID objective.
   - GETNext (the citable sparsity argument): *"The category head is employed to regulate next POI
     prediction as forecasting the next POI category is easier than exact POI prediction"* — NYC has
     ~6 check-ins/POI vs ~570 check-ins/category. The evaluated target stays POI-ID.
   - Multi-channel (2209.00472): adds region- and category-prediction channels *because* raw POI
     transitions are sparse; final metric still HR/NDCG on POI-ID.
   - HMT-GRN: predicts region at several geohash granularities as auxiliary tasks + beam search; the
     paper's goal and evaluation remain next-POI.
3. **Explicitly absent**: Flashback (no category anywhere), DeepMove and Graph-Flashback (category
   deferred to future work), LSTPM (strips category from the TMCA baseline *"for fairness because no
   other methods make use of it"* — direct evidence the community treats category as optional side
   information, not target).

## 5 · Nearest neighbors — where category/region get closer to being targets

None of these makes category/region a *co-equal end target*; each is already positioned in the MobiWac
paper (§2.2/§2.3) or run as a baseline in this repo:

- **Cascade lineage (category as intermediate stage)**: Ye et al. 2013 → LBPR (2017) → **CatDM**
  (WWW 2020) → **CSLSL** (2024): predict category (or time→what) first, then rank POIs *conditioned on*
  it. Category is scaffolding for the place ranking, never the reported outcome. Our cascade ablation:
  [`cslsl_cascade.md`](cslsl_cascade.md).
- **HMT-GRN** (SIGIR 2022): the sole region-*native* published architecture (multi-granularity region
  heads), but regions serve next-POI. Our region-native E2E adaptation beats it at all 6 datasets
  ([`next_region/hmt_grn.md`](next_region/hmt_grn.md)).
- **Novelty defusals** (already cited in the paper): DRRGNN, KGTB (predicts category+region
  *instrumentally* in service of next-place), HAMTL. See
  [`articles/[mobiwac]/RELATED_WORK_TRIAGE.md`](../../articles/[mobiwac]/RELATED_WORK_TRIAGE.md).
- **Zone relaxation** (Amichi et al., SIGSPATIAL 2021): the one verified precedent for *coarsening the
  spatial target* — predict a grid zone instead of the exact place, but **only for exploration events**
  (new places), motivated by a measured 0% Markov accuracy on exploration. Conditional relaxation, grid
  cells (not administrative regions), and still location-identity logic.
- **UniMove** (SIGSPATIAL 2025): whole task on 500 m grid-cell IDs — evidence that 2024-25
  "foundation-model" mobility work relaxes *venue* granularity, but to a spatial ID, not to a semantic
  category or an administrative region.
- **"Next activity" lineage** (MCARNN, iMTL): the field's older term for category-level prediction —
  the closest historical precedent for category-as-target, predating the current next-POI-ID wave.

## 6 · Implications for our papers

**Supports (novelty):**
- The claim *"to our knowledge, the first to treat fine-grained region as an end target of equal
  standing"* survived two adversarial sweeps (round 1 + the MobiWac citation triage) — no
  counter-example in 40+ primary sources.
- GETNext's own sparsity argument (§4) is a *mainstream, citable* motivation for category/region
  targets: the field already concedes exact-POI is data-starved (~6 check-ins/POI).
- Amichi's 0%-on-exploration result + UniMove's grid pivot show the field itself relaxing spatial
  granularity when exact-place is not learnable/needed.

**Constrains (honesty):**
- Never compare our Acc@10-on-regions against published Acc@10-on-POIs — different vocabularies,
  different tasks. Baseline tables must re-run methods on *our* targets (which is exactly what the
  faithful-STAN / HMT-GRN / ReHDM lanes do).
- Say explicitly (once, early) that we do **not** predict the exact next place — the GLOSSARY's
  three-target rule (next category / next region / next place). Blurring them was a BRACIS-reject factor.
- "Next check-in's category" ≡ "next POI's category" (the category is the visited POI's attribute) —
  interchangeable; but neither equals "next POI".

## 7 · Verification provenance & open questions

- Round 1 (canonical formulation): workflow `wf_914db79f`, 2026-07-16 — 5 search angles, 20 sources,
  85 claims extracted, 25 verified 3-vote adversarial, **18 confirmed / 7 refuted**.
- Round 2 (seminal lineage): workflow `wf_82eb612b`, 2026-07-16 — 6 angles, 24 sources, 88 claims,
  **23 confirmed / 2 refuted** (both refuted claims were fine-grained UniMove output-layer details;
  residual uncertainty flagged in the UniMove card).
- Caveats: FPMC's original domain is retail next-basket (its formulation was adapted to POI by later
  work); LoTNext's venue was not independently verified (cite as arXiv 2410.14970); UniMove's
  SIGSPATIAL 2025 proceedings status not yet confirmed in ACM DL.
- Open: (a) quantify the share of 2018-2025 papers using category/region as primary vs auxiliary vs
  absent; (b) trace who first adapted FPMC to geographic check-ins; (c) whether grid/H3 discretization
  is an emerging foundation-model trend (UniMove n=1 so far).

## 8 · Where to go next

- Per-paper cards with exact quotes + code-level evidence: [`next_poi_formulations.md`](next_poi_formulations.md).
- Our run-baseline protocol + numbers: [`README.md`](README.md) (status board), `next_category/`,
  `next_region/`.
- Paper-facing positioning: [`articles/[mobiwac]/PAPER_PLAN.md`](../../articles/[mobiwac]/PAPER_PLAN.md) §2,
  [`articles/[mobiwac]/GLOSSARY.md`](../../articles/[mobiwac]/GLOSSARY.md) (three-target rule).
