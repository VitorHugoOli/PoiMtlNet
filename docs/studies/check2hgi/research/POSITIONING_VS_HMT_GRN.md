# Positioning — Check2HGI MTL vs HMT-GRN

**Date:** 2026-04-20. Written for the BRACIS 2026 related-work section.

## Why this doc exists

HMT-GRN (Lim et al., SIGIR 2022) is the canonical hierarchical-MTL reference for next-POI recommendation and an obvious reviewer touchstone for our work. This document enumerates the architectural and experimental differences so the paper can cite HMT-GRN as **prior art whose design space ours complements and contrasts with**, not as a head-to-head baseline we need to beat numerically.

## What HMT-GRN is

> Lim, Hooi, Ng, Goh, Weng, Tan. *Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation.* Proc. 45th ACM SIGIR, 2022, pp. 1133–1143. DOI 10.1145/3477495.3531989. Repo: [poi-rec/HMT-GRN](https://github.com/poi-rec/HMT-GRN).

### Architecture

HMT-GRN's encoder is a Graph Recurrent Network (GRN): a custom LSTM/GRU cell augmented at each step with **spatial** and **temporal** multi-head attention over precomputed POI-POI graphs (a spatial graph encoding geographic proximity, a temporal graph encoding time-slot co-visitation). Neighbor embeddings from the two attention modules are fused into the recurrent state per step.

On top of the GRN encoder, HMT-GRN attaches **five parallel prediction heads**, one per geohash precision level (levels 2–6: `≈2500 km` → `≈0.6 km` cells). Each head is trained with its own cross-entropy loss; the total objective sums all five losses plus the POI-level loss. Inference uses **Hierarchical Beam Search (HBS)**: the coarser-level prediction narrows the candidate set for the finer level. A **Selectivity Layer** separately predicts whether the next POI is a revisit or an exploration (binary), and that prediction gates the ranked candidates from HBS.

### Tasks

All six heads predict geographic targets — geohashes of monotonically decreasing cell size plus the exact POI. It is a **purely geographic hierarchy** over a single "next-POI" event.

### Datasets and metrics

Foursquare (NYC, TKY) and Gowalla (full global). Primary metrics: Recall@{5, 10}, NDCG@{5, 10}, MRR — i.e., POI-ranking metrics in a next-POI recommendation framing.

## What our Check2HGI study is

### Architecture

Two-task MTL over `{next_category, next_region}` on check2HGI **check-in-level contextual** embeddings. The MTLnet framework is a **backbone + plug-in heads** decomposition: task-specific 2-layer MLP encoders → FiLM/CGC/MMoE/DSelectK/PLE shared backbone → per-task heads chosen from a registry (`next_gru`, `next_tcn_residual`, `next_transformer_relpos`, `next_mtl`, `next_stan`, and `category_*`). The headline MTL architecture is **cross-attention** (`mtlnet_crossattn`): two per-task streams with distinct input modalities exchange context through a cross-attention layer before each head.

No beam search is used — our region label is a single argmax target with Acc@K / MRR reported as ranking summaries of the logit distribution.

### Tasks

A **semantic + geographic** hierarchy:
- `next_category` (7 classes, macro-F1) — what kind of place will the user visit next?
- `next_region` (~1.1K–4.7K classes, Acc@10 / MRR) — which region?

These are **different kinds** of supervision (category = functional semantics, region = geographic bucket), not two resolutions of the same thing.

### Datasets and metrics

US state-split Gowalla (Alabama, Arizona, Florida) at `10K / 26K / 127K` training rows respectively — the same split used by HAVANA (Santos et al., BRACIS 2024) so the category-side numbers are directly comparable with the BRACIS baselines track. Primary metrics: **macro-F1** for category (per the semantic-annotation lineage — HAVANA, PGC-NN, POI-RGNN), **Acc@10 / MRR** for region (per the POI-ranking lineage — GETNext, STAN).

## Axis-by-axis contrast

| Axis | HMT-GRN (SIGIR'22) | Check2HGI MTL (this paper) |
|---|---|---|
| **Substrate** | POI-level embeddings (one vector per place) | **Check-in-level** contextual embeddings (same POI at different visits → different vectors) |
| **Task hierarchy** | Purely geographic (5 geohash scales + POI ID) | **Semantic + geographic** (category + region) |
| **Auxiliary signal** | Coarser geography | **Functional semantics** (7 POI categories) |
| **Encoder** | Graph Recurrent Network (GRN): RNN + spatial-/temporal-GAT | MLP encoder → CGC / MMoE / DSelectK / PLE / cross-attention shared backbone |
| **MTL architecture space explored** | One design (GRN + multi-head) | **6 MTL architectures** × 4 gradient-manipulation losses (CGC, MMoE, DSelectK, PLE, MTLoRA, cross-attention × NashMTL, PCGrad, GradNorm, static) |
| **Per-task input modality** | Shared encoder for all heads | **Per-task modality** explored: check-in emb for category head, region emb for region head (CH03) |
| **Inference** | Hierarchical Beam Search + Selectivity Layer | Single-pass argmax (logits ranked for Acc@K summary) |
| **Scale analysis** | Not reported | **Scale curve** over AL (10K) → AZ (26K) → FL (127K) data rows |
| **Datasets** | Global Gowalla, FSQ-NYC, FSQ-TKY | US state-split Gowalla (FL/CA/TX headline; AL/AZ dev) |
| **Primary metrics** | Recall@K, NDCG@K, MRR (ranking) | macro-F1 (category) + Acc@10 / MRR (region) |
| **External anchor** | — | HAVANA, PGC-NN, POI-RGNN for category at matched state splits |
| **Ablation on MTL gating** | Not ablated | Full grid: NashMTL / PCGrad / GradNorm / static; λ=0 isolation of architectural overhead; cross-attn vs DSelectK hybrid tested and rejected |
| **Null findings reported** | — | **5 rejected hypotheses** documented (Nash-MTL ≠ PCGrad, loss-balancing null, hybrid null, GRU hd=384 null at FL, etc.) |

## What we borrow from HMT-GRN and what we genuinely add

**Borrowed framing.** That a coarser hierarchical auxiliary task — region at roughly neighbourhood granularity — can regularize next-POI predictions. HMT-GRN's SIGIR'22 result established this empirically for geohash levels over Foursquare/Gowalla; our paper extends the idea to the check-in-level substrate where no prior MTL study exists.

**Borrowed baseline choice.** Per `docs/studies/check2hgi/CONCERNS.md §C06`, `next_gru` (our region-head champion) is the literature-aligned choice because HMT-GRN's GRN is a GRU + graph message passing. We adopt the GRU family deliberately to keep comparability; the graph signal HMT-GRN injects per step is already baked into check2HGI's input embeddings at encode time (the check2HGI encoder is a hierarchical-graph-contrastive learner — see `docs/check2hgi_overview.tex`).

**Genuinely new in our work.**
1. **Check-in-level MTL** — HMT-GRN has no notion of per-visit contextual embeddings. The same POI visited twice at different times yields two different vectors in our substrate; this changes what "sharing a backbone" means. We are the first to test MTL on this substrate.
2. **Semantic + geographic hierarchy** — the category axis is orthogonal to HMT-GRN's geohash stack. Category asks "what kind of place?", geohash asks "where (coarser)?". Our experiments show the semantic axis produces non-trivial transfer from region → category at FL scale (CH-M8: +14.2 pp transfer at FL vs +0.14 pp at AL), a scale-dependent transfer effect HMT-GRN's pure-geographic stack cannot express.
3. **Cross-attention between distinct input modalities** — our headline MTL architecture pipes check-in-level emb (category stream) and region-level emb (region stream) through cross-attention. HMT-GRN's shared-encoder design architecturally cannot do per-task-modality exchange. On FL this lifts category F1 by +3.29 pp over single-task (`results/P2/fl_crossattn_fairlr_1f50ep.json`).
4. **Scale curve with null-point honesty** — the claim "MTL helps category" is not universal; it holds at FL (+1.61 pp) and is null at AL (10K). We report both rather than only the winning scale.
5. **Decomposition of architectural overhead vs transfer** — the λ=0 category-weight isolation (cross-attention + static_weight, 0 category-weight) separates the `+Δ` from "sharing a backbone costs regions capacity" (architectural overhead) vs "category supervision transfers to region" (signal transfer). At FL: overhead = 25 pp, transfer = +14.2 pp; net regression = 10.7 pp. HMT-GRN reports aggregate lift only; we show the decomposition.

## Why this is publishable work given HMT-GRN exists

The paper's contribution is **not** "beat HMT-GRN on Foursquare-NYC Recall@10." A reviewer asking that should get the following answer, which we'll place in the paper's Limitations / Related Work boundary:

> "HMT-GRN establishes that a hierarchical geographic auxiliary task (geohash at multiple scales) helps a POI-ranking objective on a global-graph substrate. Our paper asks a different question on a different substrate: **when the embedding is per-visit contextual (check2HGI) rather than per-POI (HMT-GRN's input), and the auxiliary task is functional-semantic (category) rather than geographic, does MTL still help — and how does the answer change with data scale and architectural choice?** We find: (i) the semantic auxiliary transfers at scale (FL, 127K rows) but not at small scale (AL, 10K), (ii) only **cross-attention** MTL architectures close the category gap to single-task at AL and exceed it at FL, and (iii) the region-task side is capacity-ceiling-bound in all architectures tested, decomposable into `~25 pp` architectural overhead plus `~14 pp` category→region transfer at FL scale. These results are complementary to HMT-GRN's: they describe what happens in the regime HMT-GRN does not address (per-visit encoding, semantic auxiliary, per-task input modality, MTL-architecture ablation)."

## One-sentence elevator

> HMT-GRN showed hierarchical geographic MTL helps next-POI on POI-level embeddings; we characterise when and how semantic-auxiliary MTL helps next-region on **check-in-level contextual** embeddings across three data scales, six MTL architectures, and the λ=0 isolation of architectural overhead vs transfer.

## References

- Lim et al., *HMT-GRN*, SIGIR 2022. [PDF](https://bhooi.github.io/papers/hmt_sigir22.pdf) · [Repo](https://github.com/poi-rec/HMT-GRN)
- Luo, Liu, Liu, *STAN*, WWW 2021. [arXiv:2102.04095](https://arxiv.org/abs/2102.04095) · [Repo](https://github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation)
- Santos et al., *HAVANA*, BRACIS 2024. See `docs/baselines/BASELINE.md` for reproduced numbers at matched state splits.
- `docs/check2hgi_overview.tex` — our substrate's encoder.
- `docs/studies/check2hgi/CONCERNS.md §C06` — rationale for `next_gru` as region champion.
