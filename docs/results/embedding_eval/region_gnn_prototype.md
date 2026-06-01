# Preliminary prototype — adjacency-aware next-region head

Tests whether a head that uses the region-adjacency graph unlocks the spatial
structure that L0 adj_coh detects. Proxy: propagate region embeddings k hops over
the symmetric-normalized geographic adjacency graph (GCN^k·E) before the 1-step
transition probe. k=0 = current linear probe; k>0 = adjacency-aware. FL, 5-fold.
Script: `scripts/embedding_eval/region_gnn_probe.py`.

| engine | k=0 Acc@10 | k=1 | k=2 |
|---|---|---|---|
| gcn_ctrl | 0.6851±.003 | 0.7073 | **0.7126** |
| sidefeat | 0.6886±.004 | 0.7091 | **0.7130** |
(Acc@1 slightly *drops* with k — propagation blurs the exact target but improves top-10 recall, consistent with transitions going to spatially-nearby regions.)

## Findings
1. **The adjacency-aware head concept WORKS.** k=2 graph propagation lifts next-region Acc@10 by **~+2.7pp** over the plain probe (0.685→0.713), for both engines — ~7 SD, clearly real. A head that consumes the region graph extracts spatial structure the plain transition probe (and the embedding alone) misses. This is a genuine, exploitable architectural lever.
2. **But it does NOT favor the high-adj_coh substrate.** sidefeat's k=0 edge (+0.35pp) **washes out by k=2** (+0.04pp): once the head injects adjacency from the graph directly, the embedding's *own* encoded adjacency (adj_coh) becomes redundant. ⇒ **the lever is the HEAD (give it the region graph), not the SUBSTRATE (chase high adj_coh).** A higher-adj_coh embedding adds little when the head already has the adjacency.

## Implication
- The promising direction for a future MTL region improvement is an **adjacency-aware head** (region-graph GNN / graph-propagated logits), which helps *any* substrate equally — NOT pursuing a higher-adj_coh substrate (sidefeat's edge is real but redundant under such a head).
- **Caveat (must confirm next):** this is a 1-step transition proxy. The deployed head (`next_stan_flow` + **log_T region-transition prior**) already encodes *observed* transitions, which overlap heavily with geographic adjacency — so the +2.7pp may be partly redundant with log_T in the full pipeline.

## DECISIVE TEST — does graph propagation survive on top of next_stan_flow (log_T)?
Built GCN²-propagated region embeddings (`output/check2hgi_gprop/`) and ran the DEPLOYED head `next_stan_flow` (which has the log_T transition prior) on them, FL 5-fold:
| setup | Acc@10 |
|---|---|
| next_gru (no prior), plain | 0.6822 |
| next_gru + GCN² | 0.7126 (**+3.0pp**) |
| next_stan_flow (+log_T), plain | 0.7249±.007 |
| **next_stan_flow + GCN²** | **0.7305±.007 (+0.56pp, ~0.8 SD — NOT significant)** |

**Verdict: the adjacency lever is REDUNDANT with the log_T prior.** Graph propagation gives a real +3.0pp when the head has NO transition prior (next_gru), but **collapses to +0.56pp (within noise) once the head has log_T** (next_stan_flow). The log_T region-transition prior (learned from *observed* transitions) already captures the spatial structure that geographic-adjacency propagation provides. ⇒ On the deployed pipeline there is **no free region lift** from either route — a higher-adj_coh substrate (sidefeat, +0.30pp) OR an adjacency-aware head (GCN², +0.56pp) — both are subsumed by log_T. The "adj_coh potential" thread is real geometry but **already exploited** by the deployed head.
- **Residual untested upside (smaller, honest):** regimes where log_T is weak — small states with sparse transition counts, cold-start regions, or MTL where cross-task gradients perturb the prior. Not pursued here; FL (dense log_T) shows full redundancy.
