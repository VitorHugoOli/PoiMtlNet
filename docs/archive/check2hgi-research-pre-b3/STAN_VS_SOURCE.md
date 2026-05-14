# STAN — adapted numbers vs source article

**Source:** Luo, Liu, Liu. *STAN: Spatio-Temporal Attention Network for Next Location Recommendation.* The Web Conference (WWW) 2021. [arXiv:2102.04095](https://arxiv.org/abs/2102.04095) · [Repo](https://github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation).

## Why a direct numerical comparison is not apples-to-apples

| Axis | STAN paper | Our adaptation |
|---|---|---|
| **Target label** | exact next POI (thousands of classes) | next region (~1.1K / 1.5K / 4.7K classes on AL/AZ/FL) |
| **Pairwise bias source** | raw `Δt_ij` (minutes) + great-circle `Δd_ij` (km), linearly interpolated in learned embedding tables | learnable bias per (head, i, j) relative-position cell; ΔT/ΔD are *already internalised* by check2HGI's contextual embeddings |
| **Dataset** | Foursquare-NYC (~1 K users, ~5 K POIs), Foursquare-TKY (~2.3 K users, ~7.8 K POIs), Gowalla (full global) | Gowalla US-state splits (AL 10 K rows / 1.1 K regions; AZ 26 K / 1.5 K; FL 127 K / 4.7 K) |
| **Input modality** | POI-ID embedding + time-slot + geo-coordinate via linear interpolation | Check2HGI check-in-level contextual embeddings (per-visit, 64-d) |
| **Training protocol** | 80/10/10 user-random split (inductive by user) | StratifiedGroupKFold on userid (5f or 1f), no val-train user overlap |
| **Metrics reported** | Recall@K, NDCG@K (ranking metrics over POI catalog) | Acc@K, MRR (ranking summaries over region catalog); macro-F1 for sanity |

Because task, label space, input modality, and split protocol all differ, our numbers and the STAN paper's numbers **cannot be directly compared as "reproduction or deviation"**. The paper only shares the **architecture family** (bi-layer self-attention with pairwise bias).

## What IS comparable — the regime

We can sanity-check that our adapted STAN lands in the **competitive attention-baseline ballpark** for its task family. Rough anchor points from the literature for next-location-ranking on similar-sized catalogs:

| Anchor | Source | Acc@1 | Acc@10 | Notes |
|---|---|---:|---:|---|
| STAN (FSQ-NYC, POI target, ~5 K classes) | Luo WWW'21 | ~24–26% | ~55–60% | as reported |
| STAN (FSQ-TKY, POI target, ~7.8 K classes) | Luo WWW'21 | ~21–24% | ~48–55% | as reported |
| GETNext (FSQ-NYC, POI target) | Yang SIGIR'22 | ~28% | ~58% | as reported, stronger than STAN |
| **Our STAN-adapt on AL region (1.1 K classes)** | this study | **24.64 ± 1.38** | **59.20 ± 3.62** | 5f fair |
| **Our STAN-adapt on AZ region (1.5 K classes)** | this study | **24.48 ± 2.29** | **52.24 ± 2.38** | 5f fair |

Our numbers are within the STAN/GETNext regime on comparable label-space sizes. We do NOT claim this is a reproduction — the tasks differ — only that the adapted architecture achieves competitive performance on our substrate.

## What's missing from our adaptation vs the source

If we wanted a faithful STAN reproduction (separate paper, out of scope here), the deltas would be:

1. **Raw ΔT and ΔD features per pair.** STAN looks up time-interval and geo-distance in learned embedding tables with linear interpolation. Our `next_region.parquet` contains check2HGI embeddings but no raw timestamps or lat/lon. Adding them requires extending the preprocessing pipeline (~400 LOC in `pipelines/create_inputs_check2hgi.pipe.py` + schema change in `next_region.py`).
2. **Personalized item frequency (PIF) module.** STAN's second attention layer uses PIF — a per-user visit frequency prior over the POI catalog — at inference time to re-rank candidates. For a region target this becomes per-user region visit frequency. Not yet implemented; could be a 50-LOC addition.
3. **Candidate scoring via inner product.** STAN scores candidates by `context · candidate_embedding`. Our head uses a flat linear classifier over `n_regions` logits. The inner-product variant could help at tail regions but requires exposing the region embedding table to the head.

## Bottom-line framing for the paper

> "Our region head `next_stan` is an **architectural adaptation** of Luo et al. (WWW'21) — bi-layer self-attention with a fully-learnable pairwise bias — consuming the check-in-level contextual embeddings produced by the check2HGI encoder. Since check2HGI already encodes per-check-in spatio-temporal context, the pairwise bias in our adaptation is indexed by relative trajectory position rather than raw ΔT/ΔD (which the substrate has internalised at encode time). On our next-region task, `next_stan` achieves Acc@10 of 59.2% / 52.2% on AL / AZ respectively (5-fold fair user-disjoint splits, 1.1K / 1.5K region classes), lifting the GRU-family ceiling by +2.3 pp / +3.4 pp. Direct numerical comparison with the STAN paper's next-POI numbers is not appropriate — the label space, pairwise-bias feature source, and dataset all differ — but our adapted architecture operates in the same competitive-attention-baseline regime as reported STAN numbers at comparable class cardinalities."

## References

- Luo, Liu, Liu. *STAN*, WWW 2021. [arXiv:2102.04095](https://arxiv.org/abs/2102.04095) · [Repo](https://github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation).
- Our adapted implementation: `src/models/next/next_stan/head.py`.
- Architecture description + adaptation note: `src/models/next/next_stan/README.md`.
- Our experimental results: `research/SOTA_STAN_BASELINE.md`, `research/MTL_WITH_STAN_HEAD.md`, `results/RESULTS_TABLE.md`.
- Critical review with SOTA survey + improvement ideas: `research/STAN_CRITICAL_REVIEW.md`.
