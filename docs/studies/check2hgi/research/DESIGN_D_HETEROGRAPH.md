# Design D — Heterogeneous check-in + POI graph

**Generated 2026-05-04** as a follow-up to `MERGE_DESIGN_NOTES.md`. Captures the
"can't we put POIs in the graph?" idea: instead of late-fusing two engines or
swapping POI features in a shared encoder, **make POIs and check-ins both
first-class nodes in one graph** and let typed message passing do the merging.

---

## Why the current graphs miss this

Both engines collapse one half of the structure:

- **Check2HGI graph**: only check-in nodes. POIs exist in the data
  (`checkin_to_poi`) but only as *aggregation targets* for `Checkin2POI`'s
  attention pool. There is **no POI-POI edge**, no message propagation between
  POIs. POI semantics enter only via the loss boundary, never via the
  forward pass on a non-checked-in POI.

- **HGI graph**: only POI nodes (Delaunay). Check-ins exist in the data
  (training-set categories aggregate into POI features upstream) but visit
  trajectories never appear as edges. The model is *structurally blind* to
  who visited what when.

These are two halves of one bipartite reality: users move through space,
visiting POIs in temporal sequences. Joining them into one graph keeps
**both** signals on the wire.

---

## The graph

Two node types, four edge types:

```
Nodes
─────
  CHECKIN[N_checkins]   features = canonical 11-dim (cat_onehot + temporal)
  POI[N_pois]           features = POI2Vec(64)         (HGI's pre-trained input)

Edges
─────
  (CHECKIN -- sequence  --> CHECKIN)    user trajectory, weight = exp(-Δt / τ)
  (CHECKIN -- visits    --> POI)        membership; weight = 1
  (POI     -- visits_T  --> CHECKIN)    reverse of "visits" for bidir flow
  (POI     -- spatial   --> POI)        Delaunay; weight = log(D^1.5/d^1.5) × {1, w_r}

(Optional, redundant with sequence+visits but cheaper at message-passing time:
  (POI     -- co_visit  --> POI)        accumulates user co-visit counts)
```

The two node types live in **different feature spaces** (11-dim behavioural
vs 64-dim semantic). A shared encoder is impossible. The natural fit is
PyG's heterogeneous machinery: typed projections per node type, typed
convolutions per edge type, then message aggregation across types.

---

## Encoder

PyG `HeteroConv` over the edge type set. One layer per type, stacked 2-3
deep. Each edge type uses GCN/GAT internally; the `HeteroConv` wrapper sums
contributions across edge types per destination node.

```
Layer 1
  msg_seq      = GCN_seq(CHECKIN.x, edge_index_seq)            → CHECKIN_h1
  msg_visits   = GAT_visits(POI.x, edge_index_visits.flip())   → CHECKIN_h1
  CHECKIN_h1   = PReLU(msg_seq + msg_visits)

  msg_spatial  = GCN_spatial(POI.x, edge_index_spatial)        → POI_h1
  msg_visits_T = GAT_visits_T(CHECKIN.x, edge_index_visits)    → POI_h1
  POI_h1       = PReLU(msg_spatial + msg_visits_T)

Layer 2 (analogous, taking *_h1 as input)
  CHECKIN_h2 = ...
  POI_h2     = ...

Output:
  checkin_emb = CHECKIN_h2          ← cat path consumer (next.parquet)
  poi_emb     = POI_h2              ← cat-POI lookup
  region_emb  = POI2Region(poi_emb, poi_to_region, region_adjacency)
```

Critically, every check-in node receives messages from **both** its
trajectory neighbours **and** its POI's semantic embedding. Every POI node
receives messages from its spatial neighbours **and** every check-in that
visited it. The two graphs are no longer disjoint — they're two coupled
substructures of one heterogeneous space.

---

## Loss — three boundaries, four contrastive heads

Keep the contrastive supervision the user-facing tasks have already
validated, expanded so each tower gets its own signal:

```
L_total =
  α_c2c   · L_checkin↔checkin       (per-visit info; behaviour view)
  + α_c2p · L_checkin↔POI            (membership; bridges both views)
  + α_p2p · L_POI↔POI                (spatial; semantics view)
  + α_p2r · L_POI↔region             (HGI's contrastive boundary)
  + α_r2c · L_region↔city            (HGI's top boundary)
```

Defaults: `α_c2c = α_p2p = 0.15`, `α_c2p = 0.3`, `α_p2r = 0.25`, `α_r2c = 0.15`.
Tunable per state.

The c2c boundary preserves the per-visit variance that drives cat F1; the
p2p boundary preserves the POI-stable signal that drives reg Acc@10; the
c2p boundary forces them into a coherent joint space. Unlike the failed
POI2Vec-input augmentation, these signals reach the **typed encoder
parameters separately** — there is no shared encoder forced to compromise.

---

## Why this should escape the cat-vs-reg trade-off

The dominant failure of every shared-encoder design we've tried is that
gradient from POI-stable contrastive boundaries pulls the encoder toward
POI-mean discriminative features, flattening the per-visit variance. Two
properties of Design D break that:

1. **Per-type encoder weights**: `GCN_seq`, `GAT_visits`, `GCN_spatial`,
   `GAT_visits_T` are independent parameter sets. Gradient from `L_p2p`
   only updates `GCN_spatial` and `POI.x` projection — **never** the
   sequence or visits encoder weights that produce the cat-relevant
   `CHECKIN_h*`.

2. **POI2Vec is a fixed input on POI nodes only**: the per-visit variance
   on check-in nodes never sees POI2Vec features directly, only through
   the visits edge. The visits-edge GAT can learn to *attenuate* POI
   semantic signal when it's noise for cat (per-visit feedback), or
   amplify it when it's signal (per-visit category prediction at a chain
   restaurant).

If the trade-off persists despite both, we'll know the issue is deeper
(e.g., dataset size limits, the contrastive objective itself), not
representation choice.

---

## Sanity checks built into the design

- **Set `α_c2c = 1, others = 0`**: should recover canonical c2hgi cat
  performance (modulo the HeteroConv overhead). If it doesn't, the
  encoder typing is broken.
- **Set `α_p2p = α_p2r = α_r2c = 1, c2c/c2p = 0`**: should recover HGI
  reg performance. If it doesn't, the spatial topology is broken.
- **Detach poi_emb before region readout**: ablation — cuts behavioural
  back-propagation through the region boundary. Quantifies how much of
  the reg lift comes from check-in feedback into POIs.

---

## Implementation cost (honest)

- **PyG HeteroData wrapper**: ~50 lines. Build once during preprocess.
- **Encoder**: ~150 lines (HeteroConv stack with 4 typed convs per layer
  × 2 layers). PyG provides `to_hetero` autoconverter as fallback.
- **Loss with 3-5 contrastive boundaries**: ~80 lines, mostly bilinear
  ops we already have in HGI/C2HGI modules.
- **Output schema**: same as canonical c2hgi (`embeddings.parquet`,
  `poi_embeddings.parquet`, `region_embeddings.parquet`). The
  downstream pipeline doesn't change.
- **Training time on AL**: full-batch HeteroConv with ~100K check-ins +
  12K POIs, 2 layers, 64 hidden. Ballpark 2-5× canonical c2hgi (10-25 min
  on M4 Pro CPU vs ~5 min canonical).
- **Total**: ~3-5 days of careful work + ~1 day for AL+AZ ablation.

This is the most code of any design (A: ~30 min, B: ~1 day, C: ~3 days,
D: ~5 days). It's also the only one that **structurally** carries both
signals into the encoder rather than reconciling them post-hoc.

---

## Where Design D sits on the design tree

| Design | Mechanism | Cat preserved | Reg lift | Cost |
|---|---|---|---|---|
| A | downstream concat | likely | likely partial | ~30 min |
| B | POI2Vec at POI-pool boundary, cat path detached | by construction | bounded | ~1 day |
| C | two parallel encoders + cross-attn readout | yes (separate towers) | yes (semantics tower) | ~3 days |
| D | one heterograph, typed encoders, multi-boundary loss | by typed-weight separation | by spatial+POI2Vec topology | ~5 days |

Design D is the **only one** that produces a single coherent embedding
space where check-ins and POIs talk to each other through edges, not just
through aggregation operators or feature concat. For "general-purpose
embedding usable across tasks beyond cat/reg," that coherence is what we
actually want.

---

## Recommended path

Don't skip A → D directly. The right ladder:

1. **A first** (half-day): if late concat works, the simpler answer wins,
   stop here.
2. **D second, not B/C** (a week): if A fails, **skip to D**. B is a
   half-measure (cat preserved by construction, but reg lift bounded by
   how late POI2Vec enters); C is a half-measure (two towers but no edge
   between them — still post-hoc reconciliation). D is the principled
   merge. The cost gap between B/C and D is ~2 days; the design payoff is
   much larger.

Both A and D are falsifiable on AL+AZ with paired Wilcoxon (n=5, p=0.0312
floor) before investing in the 5-state replication.
