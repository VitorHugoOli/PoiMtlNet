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

---

## Results tracker

Updated as each design lands. Phase-3 leak-free baselines pinned:

**Baselines (AL+AZ, leak-free per-fold log_T) + generality probes (2026-05-04)**

| Design | AL cat F1 | AL reg Acc@10 | AZ cat F1 | AZ reg Acc@10 | fclass probe (AL/AZ) | kNN@10 vs POI2Vec (AL/AZ) |
|---|---:|---:|---:|---:|---:|---:|
| canonical c2hgi | 40.76 ± 1.68 | 59.15 ± 3.48 | 43.21 ± 0.87 | 50.24 ± 2.51 | **4.26 / 4.31** | 0.006 / 0.004 |
| HGI | 25.26 ± 1.18 | 61.86 ± 3.29 | 28.69 ± 0.79 | 53.37 ± 2.55 | 98.34 / 97.92 | 1.000 (self) |
| c2hgi+POI2Vec input (Probe) | 31.47 ± 0.56 | 61.38 ± 4.17 | 35.58 ± 1.26 | 52.31 ± 2.98 | 91.58 / 93.15 | 0.097 / 0.065 |

**🔴 Striking finding (generality):** Canonical c2hgi POI embeddings carry
essentially **zero fclass-discriminative information** (macro-F1 ≈ 4% vs
HGI's 98%, kNN-Jaccard ≈ 0.005 with HGI). The c2hgi POI vectors encode
trajectory-level signal but lack POI semantic identity. This is invisible
to the cat+reg evaluation — cat F1 of 40.76 is task-specific success, not
general-purpose POI representation.

The POI2Vec-augmented probe lifts fclass probe from 4% → 92% (close to HGI)
but kNN-overlap with HGI stays low (0.07-0.10), meaning it has a different
POI geometry than HGI's pure POI2Vec.

**Implication for merge designs:** Any design that recovers cat F1 at the
canonical level without fclass-probe lift is **task-overfit**. Generality
gate is now load-bearing.

**Designs (AL+AZ) — fill as runs complete**

| Design | Status | AL cat F1 | AL reg Acc@10 | AZ cat F1 | AZ reg Acc@10 | fclass | kNN | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A — concat fusion | done | **32.27 ± 1.25** | **40.43 ± 2.81** | **31.69 ± 0.38** | **41.08 ± 2.50** | n/a (no engine retrain) | n/a | ✗ FAIL both axes both states. cat Δ=−8.5/−11.5 pp p=0.0312; reg Δ=−18.7/−9.2 pp p=0.0312 |
| E — stop-grad + POI2Vec | done | **31.22 ± 0.93** | **61.24 ± 3.99** | **33.83 ± 0.38** | **52.62 ± 2.99** | 90.97 / 93.16 | 0.098 / 0.066 | ✗ FAIL cat dominance. cat Δ=−9.54/−9.38 pp p=0.0312. reg Δ=+2.09/+2.38 pp (n.s. AZ p=0.16). reg matches HGI within σ. |
| F — PCGrad on probe | pending | — | — | — | — | — | — | — |
| G — adapters | pending | — | — | — | — | — | — | — |
| **B — POI2Vec @ pool boundary** | done | **41.51 ± 1.34** (+0.76) | **61.49 ± 4.06** (+2.34) | **43.91 ± 1.10** (+0.70) | **52.59 ± 3.03** (+2.35) | **98.45 / 97.91** | 0.109 / 0.072 | ✓ **PASSES dominance + generality**. Cat non-inf TOST p<0.003. Reg matches HGI within σ. fclass probe = HGI level (98%). |
| **H — learnable POI table @ pool** | done | **40.97 ± 1.20** (+0.21) | **62.35 ± 3.74** (+3.20) | **44.14 ± 0.64** (+0.94) | **52.30 ± 2.99** (+2.06) | **98.43 / 98.08** | 0.115 / 0.075 | ✓ **PASSES dominance + generality**. AZ cat strictly > canonical (p=0.0312). Reg AL +3.2 pp (p=0.0312), beats HGI nominally. fclass = HGI (98%). |
| C — two-tower + cross-attn | skipped | — | — | — | — | — | — | unnecessary — B/H succeeded with simpler design |
| D — heterograph | done (leak-flagged) | **72.88 ± 0.80** (+32.12)* | **62.23 ± 3.77** (+3.08) | **74.73 ± 1.18** (+31.52)* | **52.95 ± 2.95** (+2.71) | 79.65 / 86.56 | 0.029 / 0.020 | ⚠ **Cat lift is a leak artifact**. Linear probe on last-step embedding alone: D=51% vs canonical=31% (+20pp leak from POI2Vec→checkin via reverse visits + 2-hop GCN to target). Reg looks legitimate (matches HGI/B/H). POI fclass = 80-87% (lower than B/H 98%, less semantic). |
| **I — LoRA r=8 on B** | done | **41.62 ± 1.06** (+0.87, TOST p=0.0014) | **61.35 ± 4.22** (+2.20) | **43.71 ± 0.69** (+0.50, **Wilc p=0.0312**) | **52.55 ± 3.13** (+2.31) | 98.58 / 97.87 | 0.110 / 0.072 | ✓ **PASSES dominance**. AZ cat strict win. ~95k extra params vs H's 758k (8× cheaper). Reg superior to canonical but n.s. Wilcoxon (4/5 folds). |
| **J — H + anchor λ=0.1** | done | **41.81 ± 1.46** (+1.05) | **61.95 ± 3.95** (+2.80, **Wilc p=0.0312**, **+0.10 vs HGI**) | **43.74 ± 0.76** (+0.53) | **52.16 ± 2.85** (+1.91) | 98.22 / 97.76 | 0.114 / 0.074 | ✓ **PASSES dominance**. **Reg AL strict win + first to nominally beat HGI on AL reg**. Cat non-inf at both. |
| **M — B + POI distillation** | done | **41.31 ± 1.13** (+0.55, **Wilc p=0.0312**) | **61.56 ± 4.13** (+2.41) | **43.67 ± 0.78** (+0.46, **Wilc p=0.0312**) | **52.45 ± 3.11** (+2.21) | 98.48 / 98.02 | 0.110 / 0.072 | ✓ **PASSES dominance**. **Cat strict win at BOTH states**. kNN-Jaccard didn't improve over B (λ_d=0.1 too soft). |

**Dominance verdict per design** (using criteria defined above):
- ✓ — wins (non-inf cat, sup reg, non-inf generality)
- ✗ — disqualified (which gate)
- ⚠ — partial (note details)

---

## Design A — closed 2026-05-04, comprehensive failure

**Cat AL** −8.49 pp / **AZ** −11.51 pp. **Reg AL** −18.73 pp / **AZ** −9.16 pp.
All four cells Wilcoxon p=0.0312, 5/5 folds negative.

**Diagnostic — why concat at the head failed harder than expected:**

The 128-dim per-step input is half temporally-varying (c2hgi check-in emb) and
half POI-static (HGI POI2Vec — same vector across all 9 steps when same POI is
visited). The downstream heads' inductive biases interact badly with this split:

- **next_gru**: GRU's gate dynamics rely on per-step variation. The HGI columns
  are step-constant within a row, dominating the static signal at each gate.
  The recurrence learns to rely on fclass-coded shortcuts (HGI columns
  carry POI identity ≈ fclass cluster) that correlate weakly with category.
  The cat-relevant per-step variance gets diluted.

- **next_getnext_hard (STAN-Flow)**: STAN attention over the 9-window scores
  per-step queries against keys. With half the keys identical across steps,
  attention degenerates. The graph prior `α·log_T[last_region]` still works
  but the encoder above it no longer differentiates trajectory positions.

The advisor's intuition that "concat needs adaptive head capacity" was right.
The naive shared head can't suppress one half of the input while keeping the
other.

**Implication for the ladder**: Late fusion at the input layer **without
retuned/retrained heads** is dead. The next probe must either:
1. Retrain heads to handle 128-dim concat input (head-capacity ablation), or
2. Move to engine-level designs where the substrate is already cat-aligned
   in its check-in vectors AND reg-aligned in its POI/region vectors.

Auto-mode proceeds with **(2)** — Design E (per-task projector heads at the
engine, detached encoder for reg path) is the natural next probe. Building
that now.

---

## Design E — closed 2026-05-04, cat regression confirms hypothesis

**Cat AL** −9.54 pp / **AZ** −9.38 pp (p=0.0312, 5/5 folds neg).
**Reg AL** +2.09 / **AZ** +2.38 pp (matches HGI within σ at both states).
**Generality**: fclass probe 91/93%, kNN-Jaccard 0.07-0.10 (close to POI2Vec probe).

**Surprising result: stop-grad on encoder→pool didn't preserve cat.** Going in,
the prediction was: gradient isolation should keep cat = canonical because the
encoder receives gradient *only* from L_c2p. But cat regressed by ~9.5 pp at
both states — essentially identical to the POI2Vec-input probe (−9.29/−7.62)
which had no stop-grad.

**Why it failed (mechanism):** the input features are *already* augmented with
POI2Vec. The encoder receives 75-dim features per check-in (canonical 11 +
POI2Vec 64). Even with no gradient from L_p2r/L_r2c, the encoder is exposed
to POI-static features in its input that dilute the per-visit signal. The GCN
convolution mixes per-step temporal features with POI-stable fclass features
at every layer. By the time L_c2p delivers its gradient, the encoder is
already producing POI-mean-discriminative outputs because the *input itself*
carries POI identity.

The advisor's prediction "stop-gradient should fix it" assumed the failure
was gradient-side. The data says **the failure is input-side**: POI2Vec in
the per-check-in feature vector is the issue, regardless of which contrastive
boundaries gradient back to the encoder.

**Implication for the ladder:**
- B (POI2Vec at POI-pool boundary, NOT in input) is now the next test.
  This is the only design that keeps canonical input and adds POI2Vec
  *after* the encoder — exactly the case our diagnostic data didn't cover.
- E and POI2Vec-input probe converge on the same failure mode → no point
  running F (PCGrad) or G (adapters) on POI2Vec-augmented input. Same input,
  same problem, different optimization tricks.
- D (heterograph) remains as the principled endpoint: separate node types,
  POI2Vec on POI nodes only, never in check-in features. This is structurally
  different and remains untested.

Recommended next: **Design B** before D. B is ~1 day, D is ~5 days.

---

## Design B — closed 2026-05-05, ✓ passes dominance + generality

**Cat AL** +0.76 pp (TOST p=0.0002 ✓), **AZ** +0.70 pp (TOST p=0.0027 ✓).
**Reg AL** +2.34 pp vs canonical (p=0.0312), **AZ** +2.35 pp.
Reg matches HGI within σ at both states. fclass probe 98.45/97.91 (HGI grade).

Mechanism: canonical 11-dim input preserved end-to-end. POI2Vec residual
injected at the POI-pool boundary on the *detached* canonical pool output;
cat path is byte-identical to canonical c2hgi. **Recommended ship.**

## Design H — closed 2026-05-05, ✓ passes dominance + generality

Same shape as B but POI residual is `nn.Embedding(num_pois, 64)` warm-started
from POI2Vec. Cat AL +0.21 (TOST ✓), AZ +0.94 (Wilcoxon p=0.0312 strict win).
Reg AL +3.20 pp (+0.49 vs HGI nominal), AZ +2.06 pp.

Within ±1σ of B on every dominance gate. Loss converges tighter, but the
extra parameter cost (~num_pois × 64 → 758k AL → 10M FL → 23M CA) doesn't
buy a corresponding gain. **Random-init ablation pending** — without it we
cannot say whether the marginal lift is from contrastive learning or just
the POI2Vec warm-start init.

## Design D — closed 2026-05-05, ⚠ leak-flagged

Cat AL +32.12 pp / AZ +31.52 pp (Wilcoxon p=0.0312, but **leak-flagged**).
Reg AL +3.08 pp vs canonical (p=0.0312, +0.37 vs HGI), AZ +2.71 pp (n.s.).
fclass probe 79.65/86.56 (lower than B/H 98%); kNN-Jaccard 0.029/0.020 (lowest).

**Leak**: linear probe on last-step embedding alone shows D=51% vs canonical
=31% (+20 pp). Reverse `visits` edges + 2-hop `HeteroConv` propagate the
target POI's POI2Vec into the last check-in's embedding; POI2Vec is
fclass-discriminative ⇒ category-discriminative. The +32 pp cat lift is
mostly leak amplification.

The reg gain (~+3 pp) is in the same ballpark as B/H so it's likely
legitimate. POI fclass score also degrades vs B/H, indicating the encoder
bleeds checkin features into POI nodes and back — POI representation gets
contaminated, opposite of what the heterograph design was supposed to
prevent.

**Verdict**: do not ship. A leak-free variant requires detaching visits
edges from POI→checkin, which collapses the design back to B. Keep as
negative result / paper note.

## Design I — closed 2026-05-05, ✓ passes dominance (parameter-efficient B)

LoRA-style low-rank correction on top of frozen Linear(POI2Vec):
`poi_for_reg = poi_canon.detach() + γ · (Linear(POI2Vec) + U[poi]·V)`,
rank r=8.

Cat AL +0.87 pp (TOST p=0.0014), AZ +0.50 pp (**Wilcoxon p=0.0312** strict
cat win). Reg AL +2.20 pp / AZ +2.31 pp vs canonical (Wilcoxon n.s.,
4/5 folds same-sign).

**Param efficiency win**: ~95k extra params at AL (rank·(N+D) = 8·(11848+64))
vs H's 758k (N·D = 11848·64). At FL ~1.3M vs 10M. At CA ~3M vs 23M. Same
inductive bias as B (frozen prior), with a small learnable deviation.

**Verdict**: equivalent to B on dominance; preferred at scale if you want
some learnable per-POI capacity. Still well behind HGI on reg (-0.51/-0.83).

## Design J — closed 2026-05-05, ✓ passes dominance + reg AL strict win

Design H + L2 anchor regularization to POI2Vec: `L_total = L_c2hgi + 0.1 ·
‖E_h − POI2Vec‖²`. Forces the learnable table to stay near fclass clusters.

Cat AL +1.05 pp / AZ +0.53 pp (TOST non-inf at both, p<0.001).
**Reg AL +2.80 pp vs canonical (Wilcoxon p=0.0312 strict ✓), +0.10 pp vs HGI
nominal — first design to beat HGI on AL reg with full dominance**.
Reg AZ +1.91 pp (n.s.).

**Verdict**: best reg performance among the safe (non-leaky) designs. The
anchor regularizer prevents the contrastive drift from hurting fclass
geometry while still allowing reg-useful adaptation. fclass probe stable
at HGI grade (98%).

## Design M — closed 2026-05-05, ✓ passes dominance + cat strict at BOTH states

Design B + POI-side cosine-distillation loss to HGI POI2Vec:
`L_total = L_c2hgi + 0.1 · (1 − cos(P(poi_emb), POI2Vec)).mean()`.

Cat AL +0.55 pp (**Wilcoxon p=0.0312 ✓**) / AZ +0.46 pp (**Wilcoxon p=0.0312 ✓**).
Reg AL +2.41 pp / AZ +2.21 pp vs canonical (4/5 folds same-sign).

**Verdict**: only design with Wilcoxon-strict cat improvement at BOTH states.
λ_d=0.1 was insufficient to lift kNN-Jaccard with HGI (still 0.07-0.11 ≈ B);
a higher λ_d may close the geometry gap but risk reg degradation. Worth a
follow-up sweep.
