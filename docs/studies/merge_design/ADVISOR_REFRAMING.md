# Advisor reframing — 2026-05-06 14:35

The advisor was consulted on the merge_design study direction after K
(Lever 3) was empirically falsified. The advisor's review is the most
important course-correction in this study to date. Persisting verbatim
key claims and the reframed plan.

## Core observation: convergence ≠ hard residual

> Six merge variants — B (frozen residual), H (learnable from scratch),
> I (LoRA), J (H + anchor), M (B + cosine distill), K (J + Delaunay GCN)
> — span a wide architectural space: features fixed/learned/low-rank,
> anchored/unanchored, with/without spatial topology. **They all land
> within ±0.1pp of each other on FL reg.** That's not "the residual gap
> is hard" — that's "you've saturated this whole class of intervention."
> When six mechanisms converge to the same output, the bottleneck is in
> none of them.

## What's wrong with Levers 4/5

- **Lever 4** (POI2Vec at p2r) and **Lever 5** (KL distill) are more of
  the same — feature-side interventions on top of B/J. If six feature
  variants converged, a seventh and eighth almost certainly will too.

## What's wrong with Lever 6 as written

- "Two-output engine" was framed as: reg head becomes "HGI grafted onto
  c2hgi backbone." That's not a merge — it's parallel HGI.
- **Reframe**: the principled merge is **add a POI↔POI contrastive
  boundary to the existing 3-boundary c2hgi loss, on top of the merge
  POI vectors** — give the existing POI vectors HGI's missing
  supervision, not a separate head.

## Three candidates we haven't tested (cheap)

### Candidate A — POI2Region hyperparams (user's hypothesis)

> You import HGI's `POI2Region` verbatim
> (`embeddings/check2hgi/check2hgi.py:30`). HGI's PMA seed query was
> tuned to consume POI-stable vectors (POI2Vec → spatial GCN → smooth,
> low-rank). c2hgi-merge's POI vectors are *attention-pooled means of
> contextual check-in vectors plus a POI2Vec residual* — a mixture
> distribution with visit-noise riding on a POI-stable component.
> PMA `num_heads=4` and a single region GCN layer may be undersized
> for disentangling these.

**Test**: J at AL, sweep `num_heads ∈ {2, 4, 8, 16}` and `region_adjacency`
GCN layers ∈ {1, 2}. ~2-3 h.
**Decision rule**: if reg moves ≥0.5 pp on any axis, POI2Region is the
residual.

### Candidate B — Boundary-weight retuning

> `alpha_c2p=0.4, alpha_p2r=0.3, alpha_r2c=0.3` were inherited from
> canonical c2hgi and never retuned for the merge regime. Adding a POI
> residual changes the gradient signal through `L_p2r` — the POI
> vectors carry POI2Vec semantics now, so the optimal weight on `L_p2r`
> shifts.

**Test**: J at AL with `alpha_p2r ∈ {0.2, 0.4, 0.5}` and
`alpha_c2p ∈ {0.3, 0.5}`. ~3 h.

### Candidate C — Reg head prior saturation

> `next_getnext_hard` adds `log_T[i,j]` additively. The Markov-1-step
> floor at FL is ~46 % Acc@10; the merge family hits ~70 %. The
> embedding contributes ~24 pp on top of a Markov prior — and a 1 pp
> embedding-quality difference is being heavily diluted.

**Test**: J at FL with `next_gru` (no `log_T` prior). If the gap to HGI
grows from 1 pp to 3-4 pp, the head is masking the embedding signal; if
it stays ~1 pp, the embedding gap is real. ~1 h.

## Is the north star right?

> The stated goal: overcome HGI on next-region.
>
> The data already shows: the merge family beats canonical c2hgi on
> reg by 1 pp at FL (Wilcoxon p=0.0312) **and preserves the 15 pp cat
> advantage that HGI lacks**. On any combined cat+reg metric the merge
> family already crushes HGI by 14 pp+. The "still 1 pp below HGI on
> reg" gap is within HGI's own σ (≈ 0.4 pp on FL).
>
> **Next-POI is unmeasured** (audit caught this; nothing was done).
> c2hgi's per-visit modeling has structural reasons to *naturally beat
> HGI on next-POI* — HGI's POI-stable vectors literally cannot
> distinguish two visits to the same place, while c2hgi can. Half the
> user's stated research goal may already be settled and uninspected.

## Reframed execution plan (9 hours, before any new design)

| # | Test | Cost | Expected outcome / decision |
|---|---|---|---|
| 1 | Next-POI probe on AL+AZ for canonical / HGI / J | ~3 h | If merge wins next-POI: half the goal already met |
| 2 | Reg-head ablation J FL with `next_gru` (no `log_T`) | ~1 h | Tells if 1 pp gap is real or masked |
| 3 | POI2Region hyperparam sweep on J at AL | ~2-3 h | User's hypothesis; if ≥0.5 pp lift → residual found |
| 4 | Boundary-weight (alpha) sweep on J at AL | ~3 h | Backup if Candidate A null |

If after these the residual is still ≥1 pp at FL, **then** commit to
the reframed Lever 6 — POI↔POI contrastive boundary added to c2hgi,
**not** "second output head".

If after step 1 the merge family is already ahead on next-POI: declare
the research goal substantially met, write up, and the remaining 1 pp
on reg becomes a paper-table footnote.

## Statistical caution

> With 6+ designs × 3 states × 2 axes ≈ 36 paired tests at uncorrected
> α=0.05, ~1.8 false positives are expected by chance. The
> 5/5-unanimity Wilcoxon at p=0.0312 is much stricter than that floor,
> so the *positive* findings are robust — but be cautious about reading
> "n.s." as "no effect" when n=5; some of the merge-vs-HGI ties may
> also be underpowered. Worth a sentence in the writeup.

## What to do with Levers 4/5/6 docs

- LEVER_4 and LEVER_5 are deprioritised. They remain in the folder as
  "feature-side interventions tested via the convergence pattern" but
  are not the next move.
- LEVER_6 needs a rewrite: drop the two-output framing; replace with
  "add POI↔POI contrastive boundary to the c2hgi 3-boundary loss".
- LEVER_5 may be reanimated post-Candidate-A if the residual is found
  to be in the loss objective and not in POI2Region.
