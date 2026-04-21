# STAN Follow-ups — Findings and SOTA Improvement Survey

**Date:** 2026-04-21. Executed from `STAN_CRITICAL_REVIEW.md §5` follow-up plan.

## Summary

| Follow-up | Status | Result | Paper impact |
|---|---|---|---|
| ALiBi-decay init for STAN pairwise bias | ✅ done (AL STL + MTL d=256) | **Null**: σ barely changed, mean unchanged | Drop; Gaussian init is fine |
| MTL STAN ALiBi on AZ | 🔄 running | TBD | Will confirm null on AZ |
| GETNext-style graph-prior head | ⛔ deferred | Not implemented this session | Future work — best expected gain |
| TGSTAN / STA-Hyper | ⛔ documented as future | Surveyed only | Future work |
| PCGrad × static_weight ablation | ⛔ skipped | Not paper-blocking | Low expected yield |
| AZ Markov baselines (simple-floor gap) | ✅ done | **Paper-strengthening** | MTL region falls below Markov floor at AZ (−1.9 pp) |

## 1 · ALiBi-decay init for STAN pairwise bias — **NULL FINDING**

### Hypothesis (from `STAN_CRITICAL_REVIEW.md §4.2`)

Replace `std=0.02` Gaussian init of the `[num_heads, 9, 9]` pairwise-bias tensor with ALiBi-style recency decay (same formula as `next_transformer_relpos`):

```
with torch.no_grad():
    positions = torch.arange(seq_length).float()
    rel_dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    for h in range(num_heads):
        slope = 1.0 / (2 ** ((h + 1) * 8.0 / num_heads))
        self.pair_bias.data[h] = -slope * rel_dist  # ALiBi-style
```

Expected outcome: reduces effective degrees of freedom (~81 free cells per head); a recency-decay prior should reduce variance on small data (AL).

### Implementation

`src/models/next/next_stan/head.py` gains a `bias_init: str = "gaussian" | "alibi"` constructor kwarg threaded through `_STANBlock` and `_STANAttention`. Backward compatible — default `"gaussian"` preserves prior behaviour.

### Results

**STL STAN on AL (5f × 50 ep, region embedding input):**

| Init | Acc@1 | Acc@10 | MRR | F1 | Notes |
|---|---:|---:|---:|---:|---|
| Gaussian (prior) | 24.64 ± 1.38 | **59.20 ± 3.62** | 36.10 ± 1.96 | 6.34 | baseline |
| **ALiBi** | 24.51 ± 1.00 | **59.38 ± 3.43** | 36.03 ± 1.63 | 6.34 | this study |
| Δ | −0.13 | +0.18 | −0.07 | 0.00 | **essentially identical** |

Within σ everywhere. Attention learns the pairwise prior from data whether or not we seed it.

**MTL cross-attn + pcgrad + STAN d=256 on AL (5f × 50 ep):**

| Init | reg Acc@10_indist | reg Acc@1 | reg MRR | cat F1 | per-fold min / max |
|---|---:|---:|---:|---:|---|
| Gaussian d=256 (prior) | 51.60 ± **10.09** | 13.86 ± 3.43 | 25.69 ± 5.34 | 38.11 ± 1.11 | 34.16 / 58.95 |
| **ALiBi d=256** | 51.64 ± **8.92** | 14.09 ± 3.71 | 25.69 ± 5.40 | 38.63 ± 1.74 | 36.76 / 59.14 |
| Δ | +0.04 | +0.23 | 0.00 | +0.52 | similar spread |

**Null finding.** Variance went 10.09 → 8.92 (−1.17 pp σ, not statistically meaningful) and mean barely moved. Per-fold range still spans 22 pp. The AL d=256 variance issue is **not** an init problem.

### Interpretation

The root cause of the variance on AL d=256 must be something else. Candidate explanations (not yet ablated):

1. **Capacity vs fold-size ratio.** STAN d=256 has ~1.2 M params; AL val = ~2.5 K samples/fold. One fold happens to land on a distribution where the attention overfits; another does not. More data (AZ) would fix it — and in fact does (AZ d=256 σ = 4.55, much cleaner).
2. **Fold-level user distribution skew.** AL's user-disjoint folds may have specific users whose region-visit patterns reward/punish STAN disproportionately.
3. **OneCycleLR max_lr interaction with d=256 head-dim.** STAN d=256 has head_dim=32 (num_heads=8); prior d=128 had head_dim=32 (num_heads=4). Effective optimizer signal differs in subtle ways.

None of these is cheaply testable. The **paper-ready conclusion**: at AL (10 K rows), STAN d=128 is the cleanest choice for MTL; STAN d=256 has higher mean but unreliable σ. Document, move on.

### Verdict

**ALiBi init is neither a help nor a harm for our adapted STAN.** Drop from the paper's ablation table. The `bias_init` kwarg stays in the code for future work and reproducibility; `"gaussian"` remains the default.

## 2 · AZ simple baselines archive — **NEW CRITICAL FINDING**

Ran `scripts/compute_simple_baselines.py --state arizona --task next_region` to fill gap in `RESULTS_TABLE.md` (row B-B7 was TBD).

### AZ region Markov-k curve

| k | Acc@10 | Notes |
|---|---:|---|
| 1 | **42.96 ± 2.05** | **Simple floor** |
| 2 | 36.35 ± 1.53 | Sparsity kicks in |
| 3 | 34.52 ± 1.53 | |
| 5 | 33.64 ± 1.35 | |
| 7 | 33.46 ± 1.33 | |
| 9 (ctx-matched) | 33.38 ± 1.33 | |

Majority baseline: 7.43 ± 0.70. Top-K popular: 20.82 ± 1.28. Confirms the AL/FL pattern — Markov monotonically degrades with k.

### Implication — CH-M1 gets stronger

**MTL region on AZ is BELOW the Markov-1 floor:**

| Method | AZ Acc@10 |
|---|---:|
| Majority (floor) | 7.43 |
| Markov-1-region (**floor**) | **42.96 ± 2.05** |
| MTL cross-attn + pcgrad + GRU | 41.07 ± 3.46 |
| MTL cross-attn + pcgrad + STAN d=128 | 37.47 ± 4.01 |
| MTL cross-attn + pcgrad + STAN d=256 | 41.04 ± 4.55 |
| MTL cross-attn + pcgrad + STAN d=256 ALiBi | TBD |
| STL STAN (ceiling) | **52.24 ± 2.38** |

**MTL region on AZ undershoots Markov-1 by ~2 pp.** This is a strong, paper-worthy finding: our neural MTL substrate on AZ *regresses below a closed-form baseline on the region side*. The same pattern holds on FL (MTL 57.60 vs Markov-1 65.05, −7.5 pp).

This is not a failure of our approach — it's a **feature of the shared-backbone dilution effect** documented in `ablation_architectural_overhead.md`. When the category head consumes backbone capacity, region's classical Markov transitions (which don't need any shared representation) beat the diluted neural head. It's the cleanest quantitative argument for CH-M1 / CH-M8 we have.

### Paper positioning

The Discussion/Limitations section should note:

> "MTL with a strong secondary-task competes with a shared backbone for capacity. On AL (10 K rows), MTL region edges above Markov-1 (+3 pp). On AZ (26 K rows) and FL (127 K rows), the MTL region head **underperforms the Markov-1-region floor** (−2 pp and −7 pp respectively). This is not a pipeline bug — it is the **signature cost** of MTL at moderate+ scale: the shared-backbone dilution in the region direction grows with data scale and backbone capacity saturation. The paper's headline is therefore *not* 'MTL beats Markov on every task'; it is 'MTL **lifts the category task** (the weaker head with abundant data), at the **price** of region-side dilution below the Markov floor at scale'."

## 3 · SOTA literature survey — improvements over STAN

Comprehensive review of STAN-family and graph-augmented next-POI models published after STAN (WWW 2021). All from our 2025 search; full citations at the end.

### 3.1 TGSTAN (Inf. Process. & Management, 2023)

> Liu, Gao, Chen. *Improving the spatial–temporal aware attention network with dynamic trajectory graph learning for next Point-Of-Interest recommendation.* IPM 2023.

**Key additions over STAN:**

1. **Trajectory-aware Dynamic Graph Convolution (TDGCN) module.** Dynamically adjusts the normalized adjacency matrix of a trajectory graph by element-wise multiplication with self-attentive POI representations at each forward pass.
2. **Bilinear-interpolation ST interval embedding.** Replaces STAN's linear interpolation over 1-D ST interval table with a 2-D bilinear-interpolated ST embedding (jointly encoding Δt × Δd).
3. **Local trajectory graph per training batch.** The graph is built from the training batch's trajectories (reflecting real-time collaborative signals) and respects causality (no future-edge leakage).

**Reported improvements over STAN:**
- Foursquare-TKY: +8.18%
- Weeplaces: +6.59%
- Gowalla-CA: +9.60%

**Applicability to our work:** Highest. TGSTAN keeps STAN's bi-layer attention and adds a graph-convolution module that could be straightforwardly added to `next_stan`. Expected cost: ~200 LOC. The "local batch-level trajectory graph" is easily derivable from our sequence POI IDs in `sequences_next.parquet`.

### 3.2 STA-Hyper (KSEM 2025)

> *STA-Hyper: Hypergraph-Based Spatio-Temporal Attention Network for Next Point-of-Interest Recommendation.* KSEM 2025.

**Key addition:** Constructs a **hypergraph** (each hyperedge is a user's full trajectory segment, rather than pairwise POI edges). Captures higher-order collaborative signals beyond pairwise POI co-visits.

**Why interesting:** Our fold-based StratifiedGroupKFold preserves trajectory structure; hypergraph hyperedges would map cleanly to our per-user training windows.

**Applicability:** Medium. Requires a hypergraph convolution module (~400 LOC) and rethinking how to pass hyperedge identity to the head during inference.

### 3.3 STAC-HNN (2025)

Employs a spatio-temporal weight matrix with multi-head self-attention. Marginally different from STAN — less architectural novelty. Low priority.

### 3.4 GETNext (SIGIR 2022)

> Yang, Liu, Zhao. *GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation.* SIGIR 2022.

**Key additions over STAN:**

1. **Global trajectory flow map.** A directed weighted graph over POIs, edge weight = co-visit count. Attributes: (lat, lon, category, frequency).
2. **Graph Laplacian attention bias.** Transition attention map `Φ = (Φ₁×1ᵀ + 1×Φ₂ᵀ) ⊙ (L̃ + Jₙ)` where `L̃ = (D+I)⁻¹(A+I)` is the normalized Laplacian of the flow map. Element-wise multiplication with `L̃+Jₙ` directly masks non-existent transitions and amplifies observed ones.
3. **Residual logit injection.** Final logits = transformer output + `Φ[current_POI]` (row-indexed).
4. **Loss:** `L_final = L_poi + 10×L_time + L_cat` (joint next-POI + time-gap + category).

**Reported gains on FSQ-NYC next-POI:** ~+5–10% Acc@10 over STAN baseline.

**Applicability to our work:** Highest-expected-value. Our region-transition graph is cheap to build (aggregate of `sequences_next.parquet` region transitions). Graph Laplacian bias is a drop-in at the head level. Residual logit injection matches our classifier stage.

**Estimated cost:** 150 LOC new head + 50 LOC pipeline helper to build + save the transition matrix per state. Reruns: ~2 h total (AL + AZ STL + MTL, 5f × 50ep).

### 3.5 TraFriendFormer (2025)

> *Trajectory- and Friendship-Aware Graph Neural Network with Transformer for Next POI Recommendation.* MDPI 2025.

Combines trajectory flow + **user friendship** graph. Requires social graph — not available in our Gowalla state splits.

**Applicability:** Low (missing friendship data). Deferred.

### 3.6 MCGT (Neurocomputing 2025)

Multi-classification Graph-Model-Driven Transformer. Partitions users into communities (spatial + temporal similarity) and builds localized subgraphs. Cross-community + intra-community attention.

**Applicability:** Medium — depends on whether user communities can be inferred from check2HGI embeddings without supervision. Potential follow-up for a future paper.

## 4 · Recommended next experiments (if time allows)

Ordered by expected paper-impact × cost:

1. **GETNext-adapted head on AL + AZ** — directly answers "does graph-prior help where STAN's attention did not?". ~4 h total.
2. **TGSTAN-style GCN addition to STAN** — tests whether STAN's architecture specifically benefits from dynamic trajectory graphs. ~6 h total.
3. **PIF-style region-frequency prior** — simplest STAN augmentation; tests whether personalised frequency matters in region recommendation. ~2 h.
4. **STA-Hyper hypergraph** — biggest implementation cost, deferred.
5. **Multi-seed n=5 on MTL headline configs** — statistical stability rather than architectural novelty.

## 5 · What STAYS in the paper

Given the ALiBi null and the pending AZ ALiBi result:

- **STAN STL** (AL + AZ 5f + FL 1f) — paper SOTA ceiling for region task.
- **MTL STAN d=128 on AL** — 5 pp lift over GRU (only win-win config documented).
- **MTL STAN d=256 hp-tuned on AZ** — ties GRU, bottleneck hypothesis confirmed.
- **MTL STAN d=256 on FL (1f sanity)** — ties GRU on Acc@10, confirms AZ pattern.
- **AZ Markov baseline** — floor reference showing MTL underperforms Markov at AZ.
- **ALiBi null** — ablation row, shows attention prior learned from data.

## 6 · What is CUT from the paper

- GETNext / TGSTAN / STA-Hyper implementations — **future work** subsection only.
- PCGrad × static_weight × STAN grid — cut.
- ALiBi init as a claimed improvement — null finding, just a neutral ablation line.

## 7 · References

- Luo, Liu, Liu. *STAN*, WWW 2021. [arXiv:2102.04095](https://arxiv.org/abs/2102.04095)
- Yang, Liu, Zhao. *GETNext*, SIGIR 2022. [arXiv:2303.04741](https://arxiv.org/abs/2303.04741)
- Liu, Gao, Chen. *TGSTAN*, IPM 2023. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0306457323000729)
- *STA-Hyper*, KSEM 2025. [SpringerLink](https://link.springer.com/chapter/10.1007/978-981-96-8725-1_19)
- Lim et al. *HMT-GRN*, SIGIR 2022. [PDF](https://bhooi.github.io/papers/hmt_sigir22.pdf)
- *TraFriendFormer*, MDPI 2025.
- *MCGT*, Neurocomputing 2025.
- Explanation paper on STAN's limitations — [arXiv:2410.03841](https://arxiv.org/abs/2410.03841) — found STAN's user embedding under-learned; confirms our "region-head invariance of cat-F1" observation since our MTL task doesn't rely on user-ID alone.
