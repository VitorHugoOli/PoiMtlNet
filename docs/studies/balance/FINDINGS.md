# Why more input features do not become more embedding

_Internal scientific record. Alabama, the six honest arms of the integrity study, evaluated on the
repository's own ladder plus two additions._

## What was measured, and with which instrument

Three rungs, kept separate because they disagree:

| rung | instrument | trained? |
|---|---|---|
| L0 | `scripts/embedding_eval/geometry.py`: kNN leave-one-out, cosine silhouette, centroid separability | no |
| L1 | per-factor multinomial logistic probe on frozen vectors, user-disjoint splits | yes, a linear probe |
| L2 | `next_gru` at the dedicated recipe, 5 folds, seed 0 | yes, the full model |

Two measurements were added because the ladder does not have them: subspace overlap between factors,
and the participation ratio of the covariance spectrum (an effective-dimension count).

## Which cheap instrument actually predicts the model

| arm | L0 kNN-LOO | L0 silhouette | L1 category probe | L2 macro-F1 |
|---|---:|---:|---:|---:|
| forward-only | 0.9164 | 0.2517 | 0.988 | 27.5127 |
| canonical + elapsed | 0.8898 | 0.2078 | 0.988 | 28.3461 |
| canonical + region | 0.7763 | −0.0225 | 0.910 | 26.9601 |
| canonical + place | 0.7810 | −0.0184 | 0.733 | 25.65 |

Rank correlation against L2, over these four arms: L0 kNN-LOO +0.600, L0 silhouette +0.600, L1
category probe +0.949.

**The train-free geometry does not rank this arm family.** It inverts both adjacent pairs: it puts
forward-only above the elapsed-time arm, where L2 has elapsed ahead by 0.83 points, and it puts place
above region, where L2 has region ahead by 1.31. The first inversion involves the only arm that
improves on the baseline, so it is not a rounding detail.

This does not contradict `L0_METHODOLOGY.md`, which certified L0 as a ranker for next-category across
ENGINES. It bounds that certification: the arms here differ only in a few input columns of one engine,
their kNN spread is 0.776 to 0.916 against an L2 spread of 2.7 points, and at that resolution the
geometry cannot separate them. Cross-engine differences are far larger, which is the regime the
archived study validated.

**The L1 linear probe does rank them**, reproducing the L2 order exactly. It is the cheap stand-in
worth using: no sequence model, no folds of `next_gru`, seconds rather than GPU-minutes. Where this
study needs a verdict without the dedicated model, L1 is the instrument, and L0 is reported alongside
as a diagnostic rather than as a ranking.

## The mechanism: the representation trades, it does not accumulate

Per-factor linear recoverability (L1), all on the same rows and splits:

| arm | category | place | region | time gap | effective rank |
|---|---:|---:|---:|---:|---:|
| forward-only | 0.988 | 0.176 | 0.067 | 0.265 | 8.43 |
| + elapsed | 0.988 | 0.170 | 0.082 | 0.669 | 10.25 |
| + region | 0.910 | 0.786 | 0.964 | 0.175 | 8.43 |
| + place | 0.733 | 0.865 | 0.531 | 0.191 | 19.45 |
| + place + elapsed | 0.745 | 0.874 | 0.540 | 0.342 | 22.08 |

Adding place identity raises place recoverability from 0.176 to 0.865 and drops category from 0.988 to
0.733. Adding region raises region from 0.067 to 0.964 and drops category to 0.910. The representation
does not add the new factor to what it already had; it substitutes.

Elapsed time is the exception that identifies the rule. It reaches 0.669 recoverability while leaving
category at 0.988, and it is the only arm that improves L2.

## Capacity was never the shortage

The forward-only arm uses 8.43 of 64 available dimensions by participation ratio, thirteen percent. The
place arm uses 19.45, thirty percent. Place therefore had more than forty unused dimensions available
and overwrote category anyway.

This rules out the whole capacity family of repairs. A wider embedding has nothing to do with room it
already declines to use, and the fix has to be a constraint that forbids the trade rather than more
space in which to make it.

## Status

Open: the joint place-and-elapsed arm has no L2 score yet, and the v18 comparison is pending. v18
ships only a windowed table, so its per-visit vectors were recovered from the last slot of each window,
which excludes the first eight visits of every user; that subset is stated wherever the v18 numbers
appear.
