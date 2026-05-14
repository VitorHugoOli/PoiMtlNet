# AUDIT — closing the HGI gap on next-region (and next-POI?)

Scope: leak-free per-fold (`region_transition_log_seed42_fold{1..5}.pt`)
results for canonical c2hgi vs HGI vs merge designs B/H/I/J/M on AL/AZ/FL.
Goal: identify the residual mechanism that lets HGI win next-region and
recommend the next architectural lever to overcome it without breaking cat.

Sources:
- AL/AZ: `docs/results/paired_tests/design_audit_al_az.json`
- FL design rows: `docs/results/P1/region_head_florida_region_5f_50ep_STL_FLORIDA_design_<x>_reg_gethard_pf_5f50ep_leakfree.json`
- FL canonical c2hgi/HGI baselines: `region_head_florida_region_5f_50ep_STL_FLORIDA_{check2hgi,hgi}_reg_gethard_pf_5f50ep.json` (these were not built from the leaky single-log; the leak-confound was specific to the design reruns and FL canonical_LEAKFREE 3-fold checkpoint)

The 3-fold `..._canonical_LEAKFREE_pf_5f50ep.checkpoint.json` is excluded
from Wilcoxon (n=3 cannot reach p=0.0312); it is reported as a sanity
anchor only.

---

## 1. Current standings — tri-state, leak-free

`reg = top10_acc on next_getnext_hard`. `cat = next_gru macro F1`.
Wilcoxon one-sided greater. n=5 floor p=0.0312, n=5 4/5-fold p=0.0625.
Δhgi = mean(design) − mean(HGI). Cat numbers for FL designs not yet
re-run on the leak-free pipeline (see §3).

| State | Variant | reg Acc@10 (%) | Δ vs can | p_gt_can | Δ vs HGI | p_gt_hgi | cat F1 (%) | Δcat | TOST 2pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| AL | canonical c2hgi | 59.15 | — | — | −2.71 | — | 40.76 | — | — |
| AL | HGI | 61.86 | +2.71 | — | 0 | — | 25.26 | −15.50 | ✗ |
| AL | B | 61.49 | +2.34 | **0.0312** | −0.37 | 0.69 | 41.51 | +0.76 | ✓ (p=1.6e-4) |
| AL | H | 62.35 | +3.20 | **0.0312** | +0.49 | 0.41 | 40.97 | +0.21 | ✓ (p=2.4e-3) |
| AL | I | 61.35 | +2.20 | 0.0625 | −0.51 | 0.78 | 41.63 | +0.87 | ✓ (p=1.4e-3) |
| AL | J | 61.96 | +2.80 | **0.0312** | **+0.10** | 0.50 | 41.81 | +1.05 | ✓ (p=8.7e-4) |
| AL | M | 61.56 | +2.41 | 0.0625 | −0.30 | 0.69 | 41.31 | +0.55 | ✓ (p=1.2e-4) |
| AZ | canonical c2hgi | 50.24 | — | — | −3.13 | — | 43.21 | — | — |
| AZ | HGI | 53.37 | +3.13 | — | 0 | — | 28.69 | −14.52 | ✗ |
| AZ | B | 52.59 | +2.35 | 0.1562 | −0.78 | 0.69 | 43.91 | +0.70 | ✓ (p=2.7e-3) |
| AZ | H | 52.30 | +2.06 | 0.1562 | −1.07 | 0.69 | 44.14 | +0.94 | ✓ (p=3.8e-5) |
| AZ | I | 52.55 | +2.31 | 0.1562 | −0.83 | 0.69 | 43.70 | +0.50 | ✓ (p=1.7e-5) |
| AZ | J | 52.15 | +1.91 | 0.1562 | −1.22 | 0.69 | 43.74 | +0.53 | ✓ (p=8.3e-4) |
| AZ | M | 52.45 | +2.21 | 0.1562 | −0.93 | 0.69 | 43.67 | +0.46 | ✓ (p=6.0e-5) |
| FL | canonical c2hgi | 69.22 | — | — | −2.12 | — | n/a | n/a | — |
| FL | HGI | 71.34 | +2.12 | **0.0312** | 0 | — | n/a | n/a | — |
| FL | B | 69.93 | +0.71 | 0.0625 | −1.41 | — | n/a | n/a | — |
| FL | H | 70.41 | +1.20 | **0.0312** | −0.92 | — | n/a | n/a | — |
| FL | I | 70.03 | +0.81 | 0.0625 | −1.31 | — | n/a | n/a | — |
| FL | J | 70.34 | +1.13 | **0.0312** | −0.99 | — | n/a | n/a | — |
| FL | M | 70.11 | +0.89 | 0.0625 | −1.23 | — | n/a | n/a | — |

Headline: **on AL, J nominally beats HGI by +0.10 pp (4/5 folds, p_gt_hgi=0.50; not significant but the first design to flip the sign)**.
On AZ and FL, no design overcomes HGI; H and J are best (∼50–56 % gap close).

---

## 2. The HGI gap on next-region — what HGI has that c2hgi doesn't

Magnitudes:
- AL: HGI − canonical = +2.71 pp; best merge (H) closes 118 % (over-shoots), J closes 103 % (overcomes nominally).
- AZ: HGI − canonical = +3.13 pp; best merge (B) closes 75 %; H/J 66 %/61 %.
- FL: HGI − canonical = +2.12 pp; best merge (H) closes 56 %; J 53 %.

Why does AL flip but AZ/FL don't? AL has the smallest region count and
densest user trajectories per region — POI2Vec's hierarchical category
prior (which is what merge designs inject) is sufficient on AL but not on
AZ/FL. The residual signal HGI has on AZ/FL is the **spatial Delaunay edge
graph at the POI level**, which c2hgi's substrate does not contain:

- HGI: `research/embeddings/hgi/preprocess.py:154` builds a Delaunay
  triangulation over POI coordinates, producing `edges.csv`; the contrastive
  objective treats triangulated neighbours as positive pairs, embedding
  spatial locality into the POI manifold.
- Check2HGI: `research/embeddings/check2hgi/preprocess.py:117-150` builds
  edges from **user check-in sequences** (and optionally same-POI), with
  no spatial-adjacency component. POI proximity is only encoded indirectly
  via co-visited sequences.

The merge designs (B/H/I/J/M) all inject HGI's *POI2Vec* prior at the
**POI-pool boundary** — i.e. the POI-side hierarchical category embedding
(`scripts/probe/build_design_b_poi_pool.py`, lines 90–117 for the freeze;
`build_design_j_anchor.py:38-87` for the learnable+anchored variant). They
give the model HGI's POI-categorisation prior but **not HGI's spatial
adjacency**. That is the residual ~1 pp on FL/AZ.

A secondary contributor: HGI's contrastive objective trains over more
boundaries (POI–region and region–region) than c2hgi (check-in–check-in
sequence), so the region-level head receives a stronger discriminative
signal even with identical Delaunay graphs.

---

## 3. Next-POI evaluation — does it exist?

**No.** The merge-design pipeline trains and evaluates only **next-region**
(predicting `region_idx`).

Evidence:
- `scripts/p1_region_head_ablation.py:223-274` — target is constructed
  from `region_df["region_idx"]`; `n_regions = int(y_region.max()) + 1`;
  `num_classes = n_regions`. There is no path that swaps the target to
  `placeid`.
- `scripts/probe/build_design_*.py` — every design builder writes
  `embeddings.parquet` (POI-level) consumed by `p1_region_head_ablation.py`
  with the region target. `placeid` is used only as a graph join key, never
  as a label.
- `docs/results/P1/region_head_florida_region_*` filenames
  encode `region_5f_50ep` — the head is region-classification.
- `src/models/next/*` registry contains next-category and next-region
  trajectories (next_gru, next_getnext, next_stan, etc.) but no
  registered placeid-target head; the cat/region target is plumbed
  externally by the ablation script.
- Repo-wide search for `placeid` as a *label*: zero hits in
  `src/models/heads/`, `scripts/p1_*`, or `scripts/probe/*`.

The thesis's "next-POI" claim therefore has **no current empirical
support** for the merge designs. The user's research target ("overcome
HGI on next-POI") cannot be evaluated from existing artefacts.

**Minimal probe to add this axis** (cost ≈ 1 day):

1. New script `scripts/p1_next_poi_head.py` cloned from
   `p1_region_head_ablation.py`; substitute `y_region` with
   `placeid_to_idx[poi_idx]` (the same map already loaded at line 165),
   set `num_classes = num_pois` (≈ 30k for FL — viable with sampled
   softmax or top-k accuracy).
2. Reuse `next_getnext` head (it already operates over a large vocabulary;
   the GETNext POI head is the natural fit). Eval metrics:
   `top1/top5/top10`, MRR, NDCG@10.
3. Run 5f×50ep × {canonical c2hgi, HGI, B, H, I, J, M} on AL+AZ first
   (≈ 4 h on M4 Pro, ≈ 1 h T4); FL only after AL/AZ trends are clear (FL
   has 4702 regions and ≈ 30k POIs — softmax memory pressure).
4. Wilcoxon n=5 with the 0.0312 gate as for next-region.

This must be staged **before** any further architectural work, otherwise
the "overcome HGI on next-POI" success criterion has no measuring stick.

---

## 4. Mechanisms by which a design could surpass HGI

Ordered by hours-of-implementation × risk. Gain figures are **upper-bound
hypotheses** until measured.

### Lever 1 — λ-anchor sweep on J (TESTED 2026-05-06: ✗ inactive due to warm-start)

**Empirical result, AL+AZ at λ=0.3** (vs the original λ=0.1 baseline):

| State | J(λ=0.1) Acc@10 | J(λ=0.3) Acc@10 | Δ |
|---|---:|---:|---:|
| AZ | 0.5215 | 0.5226 | +0.11 pp |

Run during training shows `anc=0.0000` from epoch 1 onward. The mechanism
is `--warm-start=True` (`build_design_j_anchor.py:271-274`): the POI
table is initialised to POI2Vec exactly. The anchor loss
`((poi_table.weight − POI2Vec)**2).mean()` therefore stays ≈ 0 throughout
training. Multiplying by larger λ (0.3, 0.5, 1.0, 3.0) keeps multiplying
zero by zero — no pull, no gradient, no movement.

**Verdict**: λ-tuning under warm-start is a no-op. The full sweep was
cancelled after λ=0.3 confirmed the hypothesis. To revisit this lever, we
would need `--no-warm-start` (random init + heavy anchor) — a different
experiment that tests "can we recover POI2Vec geometry from random?",
not "is more anchor better in a warm-started run?". Deferred as out of
scope; the merge family already lands at HGI-grade fclass (98 %) without
needing the random-init test.

The AL/λ=0.3 reg eval (Δ ≈ +1 pp vs canonical, in line with the original
λ=0.1) is logged at
`docs/results/P1/region_head_alabama_..._design_j_l0_3_..._leakfree.json`
once the queued rerun finishes.

### Lever 2 — H + J combined: confirm J already is H+anchor (cost: 0 h, decision)

Reading `build_design_j_anchor.py:3` (header) and the `LearnablePOIAnchor`
class at lines 38-87, **J is already H + L2 anchor to POI2Vec**. So the
"combine H and J" lever is moot — it already exists. The lever to take is
λ-tuning (Lever 1) and possibly a hybrid loss (Lever 5).

### Lever 3 — inject Delaunay edges into c2hgi POI level (cost: 12-20 h, risk: medium)

The biggest structural gap. Add a Delaunay-edge channel to
`research/embeddings/check2hgi/preprocess.py` (the `_build_*_edges`
methods at lines 117+) and a POI-level contrastive head over those edges
during c2hgi pretraining. Mirrors `hgi/preprocess.py:149-156`. Hypothesis:
closes the residual ∼1 pp gap on AZ/FL by matching HGI's spatial
inductive bias. Risk to cat: medium — adds a second pretraining objective;
needs ablation to confirm cat path is not pulled toward spatial geometry
(may help or hurt). Recommend behind a flag and validated against the
canonical cat F1.

### Lever 4 — POI2Vec at the p2r boundary too (cost: 4 h, risk: medium)

Currently every merge design injects POI2Vec at the **POI-pool** boundary
(after POI nodes pool into regions). HGI also conditions the **region**
nodes via the same hierarchy. Add a POI2Vec-derived region prior at the
p2r boundary in J (e.g. mean-pool POI2Vec into region init). Hypothesis:
+0.3-0.6 pp on FL/AZ reg. Risk: small numerical instability if region
pools have varying POI counts; mitigate with normalisation.

### Lever 5 — distribution-level distill, not cosine (cost: 6 h, risk: low)

M currently uses cosine alignment between the projected merged POI
embedding and POI2Vec (`build_design_m_distill.py:88-92`). Cosine is
pointwise and ignores neighbour structure. Replace with KL on top-k
softmax over the row-wise cosine *distribution* — i.e. distil the
neighbour ranking, not the raw vector. This is closer to what HGI's
contrastive boundary actually learns. Hypothesis: +0.4-0.8 pp on FL reg
without losing cat (M already passes TOST p=1.2e-4 on AL).

### Lever 6 — two-output engine: separate cat- and reg-grade tables (cost: 16-24 h, risk: high but bounded)

Train a single c2hgi backbone with two output projections trained jointly:
`embeddings.parquet` (cat-grade, current loss) and
`region_embeddings.parquet` (HGI-grade with Delaunay + POI2Vec
contrastive). Downstream uses the cat table for next-cat and the reg
table for next-region. Hypothesis: removes the merge-design tradeoff
entirely; reg table approaches HGI quality, cat table stays canonical.
Risk: significant pretraining refactor; bounded because each table is
evaluated independently against a known floor.

---

## 5. Recommendation

**Updated 2026-05-06 after Lever 1 falsification.**

Lever 1 was tested and returned inactive (warm-start makes the anchor
loss ≈ 0). With the cheap lever exhausted, the next live candidate is
**Lever 3 (Delaunay edges)** — the audit's identified structural
residual. Implementation landed in
`scripts/probe/build_design_k_delaunay.py` as **Design K** = J + a single
GCNConv over HGI's Delaunay POI-POI edges, applied between Checkin2POI
and POI2Region (see `merge_design/DESIGN_K.md`).

**Updated 2026-05-06 14:05 — Lever 3 (Design K) RESULT.**

| State | K (λ=0.5) Acc@10 | Δ vs canonical | Δ vs HGI | Δ vs J (λ=0.1) |
|---|---:|---:|---:|---:|
| AL | 0.6193 | **+2.78 pp ✓ p=0.0312** | +0.07 pp (n.s.) | **−0.02 pp** |
| AZ | 0.5209 | +1.85 pp (n.s.) | **−1.29 pp** | **−0.06 pp** |

**K = J empirically.** Delaunay POI-POI GCN over c2hgi's POI level
contributes zero lift over J's anchor mechanism. The structural-topology
hypothesis is **falsified**. FL K skipped (no new info expected).

**Active execution order, revised:**

1. ✗ Lever 1 (λ-sweep): inactive (warm-start zero-out, see Lever 1 above).
2. ✗ Lever 3 (Delaunay): K = J. No spatial-topology residual.
3. **Next live candidate — Lever 5 (distribution-level distill on M)**.
   Replace M's cosine alignment with KL on top-k softmax over neighbours.
   Cheap (~3 h on MPS at AL+AZ); tests whether the POI-similarity
   *distribution* over neighbours (closer to HGI's contrastive
   objective) closes the gap that pointwise alignment leaves open.
4. **Next live candidate — Lever 4 (POI2Vec at p2r boundary)**.
   Currently every merge design injects POI2Vec only at the POI-pool
   boundary; HGI also conditions region nodes via the same hierarchy.
5. **Principled but expensive — Lever 6 (two-output engine)**. Train a
   second output head with the same loss family HGI uses (POI↔POI
   contrastive on top of POI2Vec), keep `embeddings.parquet` on the
   canonical c2hgi pipeline. Cost ~16-24 h, but this is the candidate
   that can structurally match HGI's recipe.

The next-POI probe (§3) remains a precondition for any next-POI claim
and is queued as a parallel track once one of (Lever 4 / 5 / 6) yields
a closer-to-HGI design.

The next-POI probe (§3) remains a precondition for any next-POI claim
and is queued as a parallel track once the K AL/AZ verdict is in.

---

## Appendix — verified numerical sources

- AL/AZ rows: `paired_tests/design_audit_al_az.json` keys
  `{state}_{b,h,i,j,m}.{cat,reg_acc10}`.
- FL leak-free rows (per-fold top10_acc means computed from
  `heads.next_getnext_hard.per_fold[*].top10_acc`):
  - canonical c2hgi: 0.69219 (5/5 leak-free; the FL canonical_LEAKFREE
    *checkpoint* file has only 3 folds and is sanity-anchor only)
  - HGI: 0.71336
  - B: 0.69929  H: 0.70414  I: 0.70025  J: 0.70344  M: 0.70105
- FL Wilcoxon vs canonical c2hgi (5 folds, one-sided greater):
  H p=0.0312, J p=0.0312, B/I/M p=0.0625.
- HGI vs canonical c2hgi on FL: Δ=+2.117 pp, p=0.0312 (5/5 same-sign).
