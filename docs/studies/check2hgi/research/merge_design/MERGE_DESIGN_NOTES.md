# Merging HGI + Check2HGI — design notes

**Generated 2026-05-04** after the postpool falsification + POI2Vec-augmentation diagnostic at AL+AZ.

## What we know empirically (AL+AZ, leak-free per-fold log_T)

| Substrate | AL cat F1 | AL reg Acc@10 | AZ cat F1 | AZ reg Acc@10 |
|---|---:|---:|---:|---:|
| canonical c2hgi | **40.76** | 59.15 | **43.21** | 50.24 |
| c2hgi + POI2Vec input (probe) | 31.47 | **61.38** | 35.58 | **52.31** |
| HGI | 25.26 | 61.86 | 28.69 | 53.37 |
| c2hgi POI-mean → region (postpool) | n/a | 57.73 | n/a | 48.95 |

Three pinned facts:

1. **Canonical c2hgi crushes HGI on cat by ~15 pp** at both states (CH16, paper-grade at 5 states with Wilcoxon p=0.0312 each).
2. **HGI marginally beats canonical c2hgi on reg by ~2-3 pp** at both states (CH15 reframed leak-free).
3. **POI2Vec input augmentation lifts c2hgi reg by ~+2 pp** (closes 66-82% of the gap to HGI at AL+AZ) **but tanks cat by ~7-9 pp** (5/5 folds, Wilcoxon p=0.0312 vs canonical).

The two engines have opposing inductive biases. Naive merging breaks both.

---

## What each engine actually represents (read from source)

### Check2HGI — per-visit, behaviour-driven

- **Nodes**: every check-in (~100K-500K per state). Each node is a *single* visit.
- **Input features per node**: `[category_onehot(7) + temporal(4)] = 11-dim` (no POI identity).
- **Graph topology**: user trajectory edges. `(check-in_t, check-in_{t+1})` for each user, weighted by `exp(-time_gap / temporal_decay)` (default 3600 s).
- **Hierarchy**: 4-level. Check-in → POI (attention pool over check-ins of that POI) → Region (PMA + GCN) → City (area-weighted).
- **Contrastive boundaries**: 3 (c2p, p2r, r2c).
- **What it captures**: temporal context, sequence patterns, behavioural co-occurrence. Visiting the same POI at different times yields different embeddings.

### HGI — POI-stable, semantics-driven

- **Nodes**: every POI (~10K-30K per state). One node per place.
- **Input features per node**: 64-dim **POI2Vec** (fclass-level Node2Vec over the POI Delaunay graph + hierarchical category-fclass L2 regulariser). POIs sharing fclass share embeddings.
- **Graph topology**: spatial Delaunay triangulation. Edges weighted `log((1+D^1.5)/(1+dist^1.5)) × {1 intra-region, 0.7 cross-region}`.
- **Hierarchy**: 3-level. POI → Region (PMA + GCN) → City.
- **Contrastive boundaries**: 2 (p2r, r2c).
- **What it captures**: spatial-functional similarity, fclass clustering, POI identity. Same POI always emits same vector; spatial neighbourhood transmits semantics through Delaunay edges.

### Cross-comparison

| Axis | Check2HGI | HGI |
|---|---|---|
| Node level | check-in (visit) | POI (place) |
| Temporal signal | yes (sequence + decay) | no |
| POI identity | learned end-to-end from cat+temporal | injected as POI2Vec(fclass) prior |
| Graph topology | user trajectories | Delaunay spatial + cross-region penalty |
| Granularity | per-visit | per-place |
| Best at | **next-category** (per-visit cues drive task) | **next-region** (POI-stable cues aggregate cleanly) |

The reason POI2Vec-augmentation hurt cat: it injects POI identity into the input, but the encoder is **shared** between the cat-relevant and reg-relevant signal paths. Backprop from the POI-stable contrastive boundaries pulls per-check-in features toward POI-mean-discriminative directions, flattening the per-visit variance that drives cat. The reg lift is real but the cat collapse is the textbook signature of "shared encoder forced to compromise."

---

## Three merge designs, ordered by ambition

### Design A — Late fusion at the input layer (cheapest)

**Architecture**: keep both engines as-is, build embeddings independently, concatenate at the downstream model's input.

```
input to next_gru / next_stan_flow
= [c2hgi_check_in_emb(64), hgi_poi_emb_for_this_check_in(64)]   # 128-dim per step
```

For cat: model can learn to weight per-visit features. For reg: model can lean on POI-stable HGI vectors. The gradient never crosses between engines.

**Pros**
- Zero engine-level work. Both substrates already exist.
- Falsifiable in a day: train one cat STL + one reg STL with concat input on AL+AZ.
- Cleanly compatible with existing fusion infrastructure (`src/data/inputs/fusion.py` already concatenates engines).

**Cons**
- 128-dim downstream model (2× params at the input projection layer).
- No representation transfer between the two — they don't enrich each other.
- Cat F1 ceiling is bounded by how well the head can ignore the HGI columns.

**Risk it fails**: if the head can't suppress the HGI columns when computing cat (overfit-style interference), cat may regress mildly. Likely small, but possible.

### Design B — Two-tower Check2HGI with POI2Vec only at the POI boundary

**Architecture**: keep canonical c2hgi check-in features (cat path stays pure), but **inject POI2Vec at the POI level** as a side-input to `Checkin2POI`'s output before `POI2Region`.

```
checkin_emb = CheckinEncoder(canonical_features, edge_index, edge_weight)
                                                                                ← cat path reads checkin_emb
poi_emb_canonical  = Checkin2POI(checkin_emb, checkin_to_poi)
poi_emb_with_p2v   = poi_emb_canonical + γ · Linear_64→64(POI2Vec_for_poi)
region_emb = POI2Region(poi_emb_with_p2v, poi_to_region, region_adjacency)      ← reg path reads region_emb

Loss:
    L_c2p with poi_emb_canonical (canonical c2hgi loss; cat path supervision)
    L_p2r with poi_emb_with_p2v  (region path benefits from POI2Vec prior)
    L_r2c with region_emb        (unchanged)
```

POI2Vec is a fixed-frozen input (loaded from HGI's pre-trained `poi_embeddings.csv`). The check-in encoder receives gradient only from `L_c2p`, **never** from `L_p2r / L_r2c` (use `.detach()` on `poi_emb_canonical` before the residual).

**Pros**
- Cat path is identical to canonical c2hgi by construction. No risk of cat regression.
- Reg path gets the POI-stable signal without the encoder being pulled toward it.
- Single model, single training run. Output: `embeddings.parquet` (cat-grade) + `region_embeddings.parquet` (reg-improved).

**Cons**
- The cat side is *exactly* canonical — no synergy gain on cat (HGI features can't lift cat in this design).
- Region path gain bounded by how much POI2Vec adds beyond canonical c2hgi POI vectors. The diagnostic suggests +2 pp at AL+AZ; unclear if that extends.
- Adds engine-level complexity: a new `EmbeddingEngine.CHECK2HGI_DR` (dual-readout) value, new training loop, new output schema.

**Risk it fails**: the POI2Vec injection at the POI boundary may be too late — by the time signal reaches POI2Region, c2hgi's per-visit attention has already locked the POI vectors into a per-visit-noise representation. The +2 pp probe gain came from injecting at the *check-in* level, not the POI level.

### Design C — Bi-graph C2HGI with two encoders + cross-attention readout (ambitious)

**Architecture**: train two parallel encoders on two graph topologies, joined at the region level via cross-attention.

```
Tower 1 (behaviour): Check2HGI's check-in graph + canonical features
                     CheckinEncoder → Checkin2POI → poi_emb_behavior

Tower 2 (semantics): HGI's Delaunay POI graph + POI2Vec input
                     POIEncoder → poi_emb_semantic

Cross-attention readout (per POI):
    poi_emb_combined = MultiheadAttention(query=poi_emb_behavior,
                                          key=poi_emb_semantic,
                                          value=poi_emb_semantic) + poi_emb_behavior

Region level:
    region_emb = POI2Region(poi_emb_combined, poi_to_region, region_adjacency)

Loss (4 boundaries):
    L_c2p (per-visit context — behaviour tower only)
    L_p2r_behavior + L_p2r_semantic + L_p2r_combined (multi-view contrastive)
    L_r2c
```

The region readout has access to both per-visit and POI-stable views. Two separate output parquets:
- `embeddings.parquet` from the behaviour tower (cat consumer)
- `region_embeddings.parquet` from the combined readout (reg consumer)

**Pros**
- Faithful to the empirical finding that the two tasks want different inductive biases.
- Each tower keeps its own contrastive supervision; no compromise on either side.
- Cross-attention is the principled merge — query "what does this POI look like behaviourally?" against "what does this POI look like spatially?" and let the model pick.
- Generalises beyond cat+reg: any downstream task can pull either tower or the combined readout.

**Cons**
- Real engine-level intervention. ~3-5× the code of design B.
- Two training runs (POI2Vec for HGI tower, c2hgi-style for behaviour tower) plus the joint phase.
- Risk of co-adaptation pathologies (per F49 lambda-0 decomposition: cross-attn can let one tower silently piggyback the other's signal).

**Risk it fails**: cross-attention adds capacity but no specific inductive bias for "use behaviour for cat, semantics for reg." The model may learn to ignore one tower entirely, or worse, average them in a way that helps neither.

---

## Recommendation — run A first, then escalate

Design A is **a half-day experiment that decides the rest of the path**:

1. Build a per-step input by concatenating `c2hgi_checkin_emb(64)` and `hgi_poi_emb(64)` per check-in (look up by `placeid`). Output 128-dim per-step parquet.
2. Run `next_gru` (cat) and `next_getnext_hard` (reg) STL × 5f×50ep × seed 42 × {AL, AZ}.
3. Compare to canonical c2hgi cat baseline + HGI reg baseline.

Three outcomes:

- **Cat ≥ canonical c2hgi AND reg ≥ HGI**: design A wins. Ship it. Designs B/C unnecessary.
- **Cat ≥ canonical c2hgi AND reg < HGI**: late fusion preserves cat but reg lift is tower-deep, not concat-deep. Move to design B.
- **Cat < canonical c2hgi**: late fusion contaminates cat. Move to design C.

The cost of design A is one input-builder script + four training runs. ~30 min total wall-clock on M4 Pro.

## Design-space critique (advisor agent, 2026-05-04)

A–D cluster on **architectural fusion** and miss two families:

- **Optimization-side**: gradient surgery (PCGrad/CAGrad/GradNorm) on a single
  shared encoder addresses the same gradient-interference failure as D, far
  cheaper.
- **Objective-side**: task-specific projector heads with stop-gradient between
  reg head and encoder — directly tests whether the cat-vs-reg trade-off is
  fixable without graph surgery.

Adding three cheaper alternatives to the ladder, ordered by cost:

### Design E — Per-task projector heads on canonical c2hgi+POI2Vec concat (≈2 h)
Take the failed POI2Vec-augmented c2hgi probe. Add **two small projector
heads** (Linear → ReLU → Linear) — one for cat, one for reg — and apply
**stop-gradient on the reg projector** before it touches the shared encoder.
Cat path's gradient flows back as normal; reg path can only update its own
projector. Falsifies "shared encoder collapse is unavoidable."

### Design F — PCGrad on POI2Vec-augmented c2hgi (≈1 day)
Same architecture as the failed probe, but apply PCGrad gradient projection
between `L_c2p` and `L_p2r`. If the trade-off is gradient-interference (as
the advisor argues), PCGrad should mostly close it without any structural
change. Existing infrastructure (`src/losses/pcgrad.py`).

### Design G — Adapter modules on frozen canonical encoder (≈1 day)
Freeze canonical c2hgi encoder. Add two small per-task FiLM/LoRA layers, fed
POI2Vec on the reg branch only. Tests whether per-task feature modulation
(rather than per-task encoder weights as in D) is sufficient.

### Updated ladder

```
A → E → F → (A wins? stop. F wins? stop. else G → D)
```

E is now the cheapest entry. If E preserves cat AND closes ≥50% of reg gap,
the answer is "shared encoder + projectors are sufficient" and B/C/D are
research debt. Run E in parallel with A.

---

## Generality probes (added 2026-05-04)

The original AL+AZ cat+reg evaluation tests two tasks that share the same
trajectory-input failure surface. **A design that wins both is not
necessarily general-purpose.** Adding three cheap probes that run in
seconds–minutes per substrate:

1. **Linear probe on POI fclass** — logistic regression on frozen POI
   embeddings → fclass label. Tests semantic structure independent of
   trajectory. HGI's POI2Vec is fclass-clustered by construction; canonical
   c2hgi POI embeddings should land somewhere weaker. A merged design must
   match or beat HGI here.
2. **kNN-overlap with POI2Vec** — Jaccard@10 of nearest-POI sets between
   each design's POI embeddings and HGI's POI2Vec. Catches the "encoder
   memorized a task-discriminative noise direction" pathology that
   downstream F1 won't see.
3. **Region category-mix regression** — linear regression from region
   embeddings to per-region category-fraction vector. Tests aggregation
   quality independent of next-region prediction.

These run on the existing parquet artifacts; no new training needed.
Generator: `scripts/probe/generality_probes.py` (TBD).

## Updated success criteria (dominance)

Paired Wilcoxon at p=0.0312 ranks within-task but is too lenient for
selecting a general-purpose embedding. Replace with **pre-registered
dominance**:

> A design **wins** iff it is (a) non-inferior on cat F1 vs canonical c2hgi
> at TOST δ=2 pp **and** (b) strictly superior on reg Acc@10 vs canonical
> c2hgi at Wilcoxon p<0.05 **and** (c) non-inferior on the three
> generality probes vs the better of {canonical c2hgi, HGI} at TOST
> δ=2 pp on each.

Designs failing (a) regress on cat — disqualified regardless of reg lift.
Designs failing (c) overfit the joint cat+reg objective — disqualified
even if (a) and (b) hold.

---

## Design D — heterograph (separate doc)

Promoted to its own doc: **`DESIGN_D_HETEROGRAPH.md`** in this directory.

Short version: instead of late-fusing two engines or swapping POI features
in a shared encoder, make POIs and check-ins **both first-class node types
in one heterogeneous graph**. Edges: check-in→check-in (sequence),
check-in→POI (visits, bidir), POI→POI (Delaunay). Typed encoder weights
per edge type prevent the cat-vs-reg gradient interference that killed the
shared-encoder probe.

Cost: ~5 days. Only design that produces a single coherent embedding space
where check-ins and POIs talk through edges rather than aggregation
operators or feature concat.

The recommended ladder is **A → D** (skip B/C as half-measures) if A
fails to preserve cat while lifting reg.

## Beyond cat+reg

A merged embedding aimed at general-purpose use (the user's stated goal) needs to satisfy more than the two tasks here:

- **Per-place static lookup** (POI category prediction, POI ranking outside trajectory context) — needs POI-stable view, satisfied by HGI tower.
- **Per-visit dynamic lookup** (sequential next-POI, dwell-time prediction, spatio-temporal anomaly) — needs per-visit view, satisfied by Check2HGI tower.
- **Region-level aggregation** (region popularity, region category mix, regional flow) — needs region view, satisfied by HGI's POI2Region or Check2HGI's enriched POI2Region.
- **Fclass / category labelling** — POI2Vec's hierarchical L2 loss already enforces fclass-clustered embeddings; preserving this in a merged engine is a free side-benefit.

Design C is the only one that keeps all three views as separable named outputs. If the goal is reusable across tasks, design C is the principled endpoint, with A as a falsifier of whether the simpler answer is sufficient.

## Results tracker

Per-design audits with leak-free numbers live in
[`INDEX.md`](INDEX.md) (this folder). One file per design
(A, B, D, E, H, I, J, M) with aim, mechanism, AL/AZ leak-free table, and
FL leak-free status.

The full AL/AZ paired-test JSON is at
`docs/studies/check2hgi/results/paired_tests/design_audit_al_az.json`
(regenerate via `scripts/probe/finalize_design_ijm.py --designs b d e h i j m --states alabama arizona`).

### AL/AZ leak-free baselines + generality probes (2026-05-04)

| Substrate | AL cat F1 | AL reg Acc@10 | AZ cat F1 | AZ reg Acc@10 | fclass (AL/AZ) | kNN vs POI2Vec |
|---|---:|---:|---:|---:|---:|---:|
| canonical c2hgi | 40.76 ± 1.68 | 59.15 ± 3.48 | 43.21 ± 0.87 | 50.24 ± 2.51 | 4.26 / 4.31 | 0.006 / 0.004 |
| HGI | 25.26 ± 1.18 | 61.86 ± 3.29 | 28.69 ± 0.79 | 53.37 ± 2.55 | 98.34 / 97.92 | 1.000 (self) |
| c2hgi + POI2Vec input (probe) | 31.47 ± 0.56 | 61.38 ± 4.17 | 35.58 ± 1.26 | 52.31 ± 2.98 | 91.58 / 93.15 | 0.097 / 0.065 |

**Striking finding (generality):** Canonical c2hgi POI embeddings carry
~zero fclass-discriminative signal (probe ≈ 4 % vs HGI 98 %). c2hgi POI
vectors encode trajectory-level signal but lack POI semantic identity.
Any merge that recovers cat F1 without lifting fclass probe is task-overfit.

### FL status — leak-free reruns in progress

The 2026-05-05 FL block in earlier revisions of this doc was built on top
of the leaky single `region_transition_log.pt` (built from all rows
including val/test). A linear-probe diagnostic confirmed an inflation of
~13 pp on `next_getnext_hard` Acc@10 driven by the GETNext graph prior. All
FL design rows on that path (B/I/J/M ~82.2-82.4 vs canonical 82.45) are
invalid as comparisons.

Leak-free FL canonical (3-fold checkpoint): a10 = 0.6960. Per-fold
log_T tensors are now in
`output/check2hgi/florida/region_transition_log_seed42_fold{1..5}.pt`.
FL leak-free reg reruns are launched against these per-fold tensors;
results land in `STL_FLORIDA_design_<x>_reg_gethard_pf_5f50ep_leakfree`
JSONs and propagate into the per-design docs above.

FL cat (`next_gru` 5f×50ep) has not been rerun yet under the leak-free
protocol. Per the AL/AZ pattern, cat is preserved by B/H/I/J/M within
±1 pp of canonical; the dominance prediction at FL is non-inferiority
pending verification.

**Lesson:** always train baselines and treatments with identical
protocol *at the time of comparison*; do not reuse historical baselines
across protocol changes.
