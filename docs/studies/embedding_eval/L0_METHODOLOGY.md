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
