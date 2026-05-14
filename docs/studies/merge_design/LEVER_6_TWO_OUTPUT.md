# Lever 6 — REFRAMED: POI↔POI contrastive boundary on c2hgi (proposal)

> **Reframed 2026-05-06** per [ADVISOR_REFRAMING.md](ADVISOR_REFRAMING.md).
> The original "two-output engine" framing below was a half-measure
> (parallel HGI grafted onto c2hgi backbone, not a true merge). The
> principled merge is **add a POI↔POI contrastive boundary to the
> existing 3-boundary c2hgi loss**, scoring the merge POI vectors
> against POI2Vec / Delaunay neighbour pairs, while keeping a single
> output head. This gives the existing POI representation HGI's missing
> supervision signal without forking the architecture.
>
> Concretely:
>
> ```
> # Existing c2hgi 3-boundary loss
> L_c2hgi = α_c2p · L_c2p + α_p2r · L_p2r + α_r2c · L_r2c
>
> # NEW 4th boundary
> L_p2p   = contrastive( poi_emb_for_reg, positive=Delaunay-neighbour POIs,
>                        negative=random non-Delaunay POIs )
>
> L_total = L_c2hgi + α_p2p · L_p2p + λ · ‖poi_table − POI2Vec‖²
> ```
>
> POIs are positives if they share a Delaunay edge (i.e. spatially
> adjacent under HGI's triangulation). This re-uses the same edge map
> Design K already loads.
>
> Cost: ~6-8 h (smaller than the original 16-24 h two-output design,
> because we're adding one term to one existing loss, not designing a
> new head + parallel training).
>
> Only run if Tests 1-4 from STATE.md do not close the gap. The reframe
> is documented; the original two-output proposal is preserved below
> for archaeology.

---

## Original proposal (deprioritised) — Two-output engine

## Aim

Replicate HGI's actual training recipe on the c2hgi POI structure as a
**second output head**, while keeping the canonical c2hgi pipeline for
the first output. Single backbone, two output parquets:

- `embeddings.parquet` — cat-grade, byte-identical to canonical c2hgi.
- `region_embeddings.parquet` — reg-grade, trained with HGI's recipe
  (POI↔POI contrastive on top of POI2Vec, Delaunay edges, hierarchical
  fclass L2 regulariser).

## Why

K (Lever 3) and the warm-start finding (Lever 1) together rule out
both the *features* and the *graph topology* as the residual gap to
HGI. What's left is the **training recipe**:

- HGI has 4 contrastive boundaries (POI↔POI, p2r, r2c, plus the
  POI2Vec discriminate). c2hgi has 3 (c2p, p2r, r2c). The merge family
  has the same 3 as c2hgi, with POI2Vec as a frozen prior added to the
  POI residual — never as a contrastive target itself.
- HGI's POI2Vec is trained with a hierarchical fclass L2 regulariser
  during pretraining. The merge family imports its output as a frozen
  prior; never re-trains it under c2hgi's losses.

Lever 6 gives the reg-side output a full HGI-style 4-boundary loss
while leaving the cat-side output on canonical c2hgi. The two heads
share the c2hgi backbone but have different post-pool projections and
different loss families.

## Architecture sketch

```
checkin_emb              = CheckinEncoder(canonical_features)        # shared backbone
poi_emb_cat (.detach)    = Checkin2POI(checkin_emb)                  # cat head
poi_emb_reg              = Checkin2POI(checkin_emb).detach() + γ · poi_table
                                                                      # reg head start
poi_emb_reg              = POIEncoder(poi_emb_reg, delaunay_edges)   # HGI-style POI GCN
region_emb_reg           = POI2Region(poi_emb_reg, …)                # standard pool

L_total = L_c2hgi(cat-side, 3 boundaries on poi_emb_cat / region_emb_cat)
        + L_hgi(reg-side, 4 boundaries on poi_emb_reg / region_emb_reg
                          incl. POI↔POI contrastive on Delaunay edges
                          and POI2Vec discriminate)
```

The cat side outputs `embeddings.parquet`. The reg side outputs
`region_embeddings.parquet`. Both come from the same training run.

## Hypothesis

This is the **principled** answer to the user's research goal:
"overcome HGI on next-region without breaking next-cat." It replicates
HGI's training recipe on the c2hgi POI structure, so the reg head
should approach HGI's reg quality. Cat is byte-identical to canonical
because the cat-side path has no new losses.

Expected: reg Acc@10 within ±0.3 pp of HGI at FL (currently merge family
is −0.9 to −1.0 pp). Cat F1 within ±0.3 pp of canonical.

## Cost

~16-24 h. New `Check2HGI_TwoOutput` model class, new build script,
needs careful loss balancing between cat-side and reg-side losses.
`alpha_cat` and `alpha_reg` weights need to be tuned (start at 0.5/0.5,
sweep on AL).

## Risk

- Loss balancing: medium. Two contrastive loss families with different
  scales; bad balance can starve one head.
- Generality probe shift: low. fclass should be ≥98 % (matches HGI).
- Effort: this is the largest single change in the study; ~1-2 days of
  implementation + AL+AZ revalidation.

## Decision criterion

This is the candidate that *should* work if features-only and
topology-only didn't. If Lever 6 fails to close the gap, the residual
is in something even more subtle (e.g. HGI's `cross_region_weight=0.7`
calibration, HGI's longer training run with different LR schedule),
and we'd be in deep diminishing-returns territory.

## Why this isn't already done

Lever 6 is the principled answer but it's also the largest engineering
change. We do Levers 4 and 5 first because they're cheap and might be
sufficient. Only if both fail do we commit to the bigger lift.

## Pre-conditions

Before starting Lever 6:

1. Apply SPEEDUP_AUDIT.md Tier 1 wins so the new training loop runs
   at ~6 s/iter instead of 8 s/iter — this matters because Lever 6's
   loss is heavier and we'll otherwise regress on iteration time.
2. Stand up the next-POI probe (per AUDIT_HGI_GAP.md §3) so we have a
   baseline to measure against on the next-POI axis as well as
   next-region.
3. Confirm Lever 5 (and possibly 4) don't already solve it. If they do,
   Lever 6 is unnecessary.
