# Design K — J + Delaunay POI-POI GCN (root-cause fix for HGI gap)

## Aim

The audit (`AUDIT_HGI_GAP.md`) identified HGI's load-bearing structural piece
that none of the merge family (B/H/I/J/M) inherit: a **Delaunay
triangulation of POI coordinates** with weighted edges
`log((1+D^1.5)/(1+dist^1.5)) × {1.0 intra-region, 0.7 cross-region}`. HGI
runs a GCN over those edges at the POI level *before* aggregating to
regions. The c2hgi family runs no POI-POI graph convolution at all — POIs
only see each other through the region-level PMA pooling.

Design K closes that loop: take Design J (learnable POI table + λ-anchor
to POI2Vec) and insert HGI's Delaunay GCN between Checkin2POI and
POI2Region.

## Mechanism

```
checkin_emb        = CheckinEncoder(canonical_features)            # cat path
poi_emb_canonical  = Checkin2POI(checkin_emb)                      # detached for reg
poi_residual       = γ · poi_table[i]
poi_pre_gcn        = poi_emb_canonical.detach() + poi_residual
poi_emb_for_reg    = PReLU(GCNConv(poi_pre_gcn,                    # NEW
                                   delaunay_edge_index,
                                   delaunay_edge_weight))
region_emb         = POI2Region(poi_emb_for_reg, ...)              # reg path

L_total = L_c2hgi + λ · ‖poi_table − POI2Vec‖²                     # same anchor as J
```

Delaunay edges are loaded from HGI's existing `output/hgi/<state>/temp/edges.csv`
(symmetric, weighted), with placeids remapped from HGI's row index space
to c2hgi's POI index space via the shared `placeid` column. AL: 71 k
symmetric edges from 35.5 k raw triangulation pairs.

The GCN layer adds one PReLU-activated GCNConv at the POI level. This is
literally the layer HGI's `POIEncoder` runs on its POI2Vec inputs; we
splice it in here on top of c2hgi's per-visit aggregated POI embeddings.

## What this tests

The audit's hypothesis: the residual ~1 pp gap to HGI on reg at FL is
**structural**, not feature-side. POI2Vec gets us 50-56 % of the way
because it injects fclass-clustered semantics into the POI representation.
The remaining gap is the spatial *topology* — which POIs are physically
near which — and a GCN over Delaunay edges is the cheapest way to
represent that without rewriting the whole engine.

If K closes the FL gap to ≤0.3 pp vs HGI on Acc@10, the audit's
root-cause diagnosis is confirmed and Design K becomes the recommended
ship configuration. If K matches J/H (~+1 pp vs canonical, still −1 pp vs
HGI), then the spatial topology is not the active piece and a different
architectural lever is needed.

## Implementation

- Builder: `scripts/probe/build_design_k_delaunay.py`
- Model class: `Check2HGI_DesignK` (subclasses `Check2HGI`, adds
  `poi_table`, `gamma`, `poi_gcn`, `poi_gcn_act` modules)
- Edge loader: `load_delaunay_edges(state, placeid_to_idx, num_pois)`
  reads HGI's `edges.csv` + `pois.csv` and remaps to c2hgi indices,
  symmetrises, returns `(edge_index, edge_weight)`
- Output dir: `output/check2hgi_design_k_l<λ>/<state>/` with
  `embeddings.parquet`, `poi_embeddings.parquet`, `region_embeddings.parquet`
- Substrate name in p1 ablation: `check2hgi_design_k_l0_5` (matches via
  the `startswith("check2hgi_design_k")` branch added in
  `scripts/p1_region_head_ablation.py`)
- Cost: AL has 11 848 POIs, builds at ~8 s/iter on MPS (vs J's ~5.6 s/iter
  — extra GCN + Delaunay edges). 500 epochs ≈ 70 min on AL.

## AL/AZ leak-free results (2026-05-06)

Tag: `STL_<STATE>_design_k_l0_5_reg_gethard_pf_5f50ep`.

| State | reg Acc@10 | Δ vs canonical | Wilcoxon p_gt | Δ vs HGI | Δ vs J (λ=0.1) |
|---|---:|---:|---|---:|---:|
| AL | 0.6193 ± ? | **+2.78 pp** | **p=0.0312 ✓** (5/5) | +0.07 pp (n.s.) | **−0.02 pp** |
| AZ | 0.5209 ± ? | +1.85 pp | p=0.16 (3/5) | **−1.29 pp** | **−0.06 pp** |

Per-fold K AL: [0.6609, 0.6220, 0.6515, 0.6015, 0.5608]
Per-fold K AZ: [0.5244, 0.5359, 0.5423, 0.4690, 0.5327]

## Verdict: ✗ Delaunay GCN does NOT close the HGI gap

K matches J exactly at both states (AL Δ=−0.02 pp, AZ Δ=−0.06 pp from
J(λ=0.1)). Adding HGI's spatial topology as a POI-level GCN layer
contributes **zero empirical lift** over the existing J anchor mechanism.
At AZ K is still 1.29 pp below HGI — the same gap J has.

**The audit's "structural residual" hypothesis is falsified.** The
~1 pp residual to HGI on AZ (and presumably on FL) is *not* spatial-graph
information. Candidates that remain:

1. **POI2Vec's hierarchical fclass L2 regulariser** during HGI's
   pretraining (`research/embeddings/hgi/poi2vec.py`). The merge family
   uses POI2Vec only as a *frozen prior* — it never re-trains POI2Vec
   end-to-end with c2hgi's losses. HGI's POI representation may benefit
   from joint training of the fclass-clustering objective alongside the
   contrastive boundaries.
2. **HGI's larger contrastive boundary count** (4 boundaries vs c2hgi's
   3): HGI has POI↔POI as a contrastive boundary, not just a graph for
   message passing. The merge family lacks a POI↔POI contrastive
   objective entirely.
3. **HGI's `cross_region_weight=0.7` edge weighting** in Delaunay edges
   (per `embeddings/hgi/CLAUDE.md`). K's GCN uses HGI's edges *with*
   their weights, but the GCN may be ignoring the structure under the
   c2hgi loss landscape.

## FL: skipped

Given AL+AZ show K ≡ J empirically, FL would only confirm the same
pattern at a 10× larger build cost. No new information; cancelled.

## What this rules out

The user's research question — "how to fill the gap and overcome HGI on
next-region?" — has now eliminated:

- Lever 1 (λ-anchor sweep on J): inactive due to warm-start (DESIGN_J
  warm_start=True keeps anchor loss ≈ 0, regardless of λ).
- Lever 3 (Delaunay POI-POI edges): K = J empirically.

The remaining cheap candidate (per AUDIT_HGI_GAP.md):

- **Lever 5: distribution-level distill** in M (replace cosine with KL on
  top-k softmax over neighbours). Could be tested at AL+AZ in ~3 h on MPS.
- **Lever 4: POI2Vec at the p2r boundary** (currently only at the pool
  boundary). Cheap structural extension.

The expensive but principled candidate:

- **Lever 6: two-output engine** with HGI-grade reg-side training
  (separate `region_embeddings.parquet` trained with POI↔POI contrastive
  on top of POI2Vec, while `embeddings.parquet` keeps canonical c2hgi
  losses). This is the path that *can* close the gap, because it would
  replicate HGI's actual training recipe on the c2hgi POI structure.
