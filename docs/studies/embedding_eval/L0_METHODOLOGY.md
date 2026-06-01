# L0 methodology — task-specific validity of train-free geometry (2026-06-01)

Audit motivated by the v13 inversion: **v13 (resln_design_b) has the WORST region
adj_coh (0.231) yet the BEST family L2-region (0.7309)**. If L0 were a valid region
ranker this could not happen. The audit (advisor + code review of `geometry.py`,
`region_eval.py`) settles whether our L0 metrics are used correctly, **per task**.

## Bottom line
L0 must be **task-specific**:
- **next-cat (static-attribute task): L0 is a legitimate, near-sufficient RANKER.** The
  target (a POI's own category) is a static property carried in the embedding geometry,
  so own-label separability metrics (kNN-LOO, silhouette, centroid-sep) map monotonically
  to L2-cat. This is why cat L0 tracks L2 (Check2HGI family ≫ HGI at L0 AND L2). ✔ keep.
- **next-reg (transition/sequence task): NO static-geometry metric can RANK substrates.**
  The target (next region given 9-step history) is **not** a static property of the current
  region vector — it lives in the **transition operator (log_T)**, not the embedding
  geometry. adj_coh measures *geographic adjacency*, which is (a) not what design_b
  optimizes (design_b adds **fclass** similarity, a different axis the head exploits) and
  (b) **already supplied to the L2 head by log_T** (redundant). Hence adj_coh ANTI-ranked
  v13. ⇒ **adj_coh is DEMOTED to a diagnostic — never a region ranker.** Region comparisons
  must start at **L2** (`next_stan_flow` + log_T, 5-fold, multi-seed).

## Per-metric verdict
| metric (file) | measures | verdict |
|---|---|---|
| kNN-LOO (`geometry.py:20`) | local own-label coherence | **sound — RANKS next-cat.** Meaningless for reg. |
| Silhouette (`geometry.py:129`) | global own-label cluster separation | **sound for cat.** Not a reg signal. |
| Centroid sep-ratio (`geometry.py:100`) | own-label cohesion / inter-centroid sim | **sound for cat.** Task-mismatched for reg. |
| Linear CKA (`geometry.py:146`) | engine-vs-engine representational similarity | **diagnostic only** ("same space?"), never a quality/rank signal — and reads low across scale/dim by construction (mean-center only). |
| Adjacency-coherence (`region_eval.py:72`) | fraction of region kNN that are geo-adjacent | **TASK-MISMATCHED.** Arithmetically sound, wrong axis; log_T-redundant; anti-ranked v13. **Diagnostic only.** |

The single real error-class is conceptual: every metric except CKA is an *own-label
static-separability* metric — exactly right for next-cat, structurally wrong for next-reg.

## Why the asymmetry is fundamental (not a bug)
- **next-cat** ≈ "is the item's own label recoverable from its geometry?" — the same quantity
  the L2 head reads. Short, monotone L0→L2 map. L0 is near-sufficient.
- **next-reg** ≈ "where do visitors go next?" — a property of the *dynamics*, not any single
  region vector. No static geometry encodes it. The closest static proxy (adj_coh) is doubly
  defeated: partial-geographic AND redundant with log_T. The GCN² prototype confirms it:
  adjacency propagation gives +3.0pp under `next_gru` (no prior) but **+0.56pp NS under
  `next_stan_flow` (+log_T)** — once log_T is present, adjacency geometry adds nothing.

## Corrected per-task L0 protocol
| task | nature | valid L0? | protocol |
|---|---|---|---|
| next-cat | static attribute | **YES — ranks** | kNN-LOO + silhouette + centroid-sep on `embeddings.parquet`, category labels (POI-pooled OK). May crown the cat axis. |
| next-reg | transition | **NO static ranker** | adj_coh / own-label region metrics = **diagnostic flags only** (present/absent, geo-coherence). Ranking starts at **L2** (`next_stan_flow`+log_T, 5-fold, multi-seed). Optional fclass-coherence diagnostic when the mechanism is fclass-based. |
| next-poi | high-card/future | n/a | check-in-granularity recoverability diagnostic only. |

**Cross-engine L0 comparison is valid for next-cat, NOT for next-reg.**

## Prior conclusions re-stated
1. "resln+sidefeat best region substrate / v14 supersedes v13" — **RETRACTED** (already, in
   FINAL_SYNTHESIS §RESOLVED). Canonical adj_coh-misuse; L2 inverted it.
2. "adj_coh reproduces HGI's region win" (README §7) — **coincidental, not validating.** HGI
   is high on both; the *family-internal* ranking (v13 low adj_coh, best L2) shows adj_coh has
   no ranking authority. Re-state as a diagnostic agreement, not a screen.
3. adj_coh listed as a peer L0 metric — **re-label "region diagnostic (non-ranking)."**

**Unchanged / sound:** all next-cat L0 conclusions (Check2HGI family ≫ HGI; v13/resln top);
GCN²-redundant-with-log_T; "HGI next-reg gap is OPEN, no substrate closes it under L2."

## EMPIRICAL TEST — region-label separability (user proposal, 2026-06-01)
Q: instead of clustering the 7 categories, cluster the k regions (use `poi_to_region`,
the task's own label space). Computed kNN10 region-acc + region-silhouette on POI
embeddings (shared placeid→region map; FL):
| engine | kNN10 reg-Acc | region-silhouette | L2-reg Acc@10 |
|---|---|---|---|
| check2hgi (canonical) | 0.079 | −0.649 | 0.7274 |
| check2hgi_resln | 0.080 | −0.641 | 0.7275 |
| v13 (resln_design_b) | 0.059 | −0.665 | 0.7309 |
| **hgi** | 0.078 | **−0.460** | **0.7362** |

**Two-part verdict:**
1. **For the cross-substrate / HGI-gap question this metric WORKS** where category-L0 and
   adj_coh did not: HGI's region-silhouette (−0.46) is markedly above the whole Check2HGI
   family (~−0.65), **concordant with HGI's next-reg win.** It localises HGI's advantage to
   **POI-level spatial/region cohesion** (its Delaunay graph), the axis the user-sequence
   Check2HGI graph lacks. → adopt region-silhouette as the **next-reg cross-substrate L0
   diagnostic**.
2. **Within the Check2HGI family it still anti-ranks v13** (Spearman −0.60: v13 lowest
   reg-Acc/silhouette yet best L2). design_b's fclass similarity pulls same-function POIs
   together across regions → lowers region cohesion but helps the transition head. So the
   within-family micro-gain (+0.35pp) is NOT region-geometry — confirming the static-L0
   ceiling: it captures the coarse self-stay/cohesion axis (where HGI wins) but not the
   fine transition axis (where design_b wins).

**Actionable lead:** HGI's next-reg edge = POI-level spatial cohesion (Delaunay). The lever
to close the gap is **spatial/Delaunay POI-POI edges** (T6.1 p2p / design_k = J+Delaunay),
NOT fclass/sidefeat — those raise different axes that don't lift region cohesion.
