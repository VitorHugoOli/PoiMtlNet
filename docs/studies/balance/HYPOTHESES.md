# Hypotheses for "more features, no improvement", ranked against the evidence

_Internal scientific record. Each hypothesis is stated so that it can fail, and each is scored against
what has already been measured rather than against plausibility._

## The measurement the hypotheses must explain

At Alabama, per-factor linear recoverability of frozen check-in vectors:

| arm | category | place | region | time gap | effective rank / 64 |
|---|---:|---:|---:|---:|---:|
| forward-only | 0.988 | 0.176 | 0.067 | 0.265 | 8.43 |
| + elapsed time | 0.988 | 0.170 | 0.082 | 0.669 | 10.25 |
| + region identity | 0.910 | 0.786 | 0.964 | 0.175 | 8.43 |
| + place identity | 0.733 | 0.865 | 0.531 | 0.191 | 19.45 |

Two facts any explanation has to fit. Adding a factor raises that factor and lowers the category, so
the representation substitutes rather than accumulates. And it does this while using between thirteen
and thirty percent of the available directions, so it is not out of room.

Five explanations are already refuted and are not re-proposed here: under-training, lossy compression
in the export, insufficient encoder depth, insufficient width, and insufficient training data.

## What the literature calls this

Two established phenomena match the two halves of the measurement.

**Feature suppression.** Robinson et al., "Can contrastive learning avoid shortcut solutions?"
(NeurIPS 2021, arXiv:2106.11230) study exactly the situation where several predictive features are
available to a contrastive encoder. Their finding is that the objective uses a subset and discards the
rest, and that interventions improving one feature commonly degrade another. Their remedy, implicit
feature modification, perturbs the discrimination task so that the easy feature stops sufficing,
rather than modifying the encoder. That is the substitution half of our table, and it says the lever
belongs on the objective, not on capacity.

**Dimensional collapse.** The embedding occupying a small subspace of the available dimensions is
surveyed in the SSL cookbook (arXiv:2304.12210) and treated directly by whitening and
covariance-regularisation work (arXiv:2408.07519, arXiv:2402.09586). That is the 8.43-of-64 half.

The two are related but not the same, and they imply different fixes, which is why the study treats
them as separate hypotheses rather than one.

## Ranked hypotheses

### H1 — The objective suppresses features, and the fix is a decodability constraint

**Claim.** The hierarchical infomax objective has no term requiring the category to remain present, so
when a stronger discriminative signal arrives the category is overwritten. Adding an explicit linear
decodability floor for the visit's own category should stop the trade.

**Evidence for.** It is the only hypothesis that predicts the sign pattern of the whole table,
including why elapsed time is the exception: a time gap is a poor discriminator (measured lift over
chance 1.5x against 38x for region and 148x for place), so it does not compete for the objective's
attention and nothing is traded for it.

**Evidence against.** None yet, which is a reason for suspicion rather than confidence.

**Status: implemented** as the category-preservation term in `research/balance/balance_terms.py`.

**The circularity that limits it.** The term optimises a linear category decoder and our L1 instrument
is a linear category probe. A rise in category recoverability under this term is therefore close to
tautological and is NOT evidence. The informative outcomes are whether the other factors survive at
the same time, which is what balance means, and whether the downstream score moves, which the term
does not optimise. Stated here and repeated in the results.

### H2 — The embedding is dimensionally collapsed, and the fix is a variance-covariance penalty

**Claim.** With 8.43 effective dimensions of 64, a new factor has no free directions to occupy and is
written on top of an existing one. A variance hinge plus off-diagonal covariance penalty should open
directions and let factors coexist.

**Evidence for.** The effective rank is measured and low. The mechanism is well documented in SSL.

**Evidence against, and it is substantial.** The place arm reached effective rank 19.45, more than
double the baseline, and it is the WORST arm downstream at 25.65. So a higher effective rank is not
sufficient for a better representation in this setting, and may be orthogonal to it. Five capacity-side
explanations have already been refuted, and this is arguably a sixth.

**Status: implemented**, with the prediction on record that it will raise effective rank without
raising the downstream score. If that prediction holds, the anti-collapse family is closed.

### H3 — The consumer cannot absorb a richer embedding

**Claim.** The downstream head barely grows with the embedding: at the default hidden size, doubling
the embedding from 64 to 128 adds 49,152 parameters, 7.6 percent, because the recurrent and classifier
weights dominate and do not scale with the input width. If the consumer has no capacity for extra
information, no improvement in the representation can show up in the score.

**Evidence for.** The parameter arithmetic is exact and verified by construction.

**Test.** The per-width tuning grid, three learning rates by three hidden sizes at each of dim 32, 64
and 128, comparing best against best. This hypothesis was raised by the author after observing that
the earlier width arms varied the artifact but not the head.

**Status: under test.** Note that H3 competes with H1 and H2: if tuning the consumer recovers the
gain, the representation was never the bottleneck.

### H4 — The pretext and downstream tasks are misaligned, and no input change can fix it

**Claim.** The objective reconstructs place-level category composition and discriminates shuffled
nodes. Neither rewards predicting the NEXT category. Under this reading the plateau is a property of
the objective and the only real fix is a predictive pretext term.

**Evidence for.** The scaling curve is flat over an eight-fold range while the pretext loss RISES with
data, so scale makes the pretext task harder without making the representation more useful. That is
the signature of a misaligned objective rather than an underfit one.

**Evidence against.** It is the least falsifiable of the four as stated, and it subsumes H1: a
decodability floor is a weak form of the same idea.

**Status: not implemented.** The natural instantiation is a next-category prediction term over the
forward edge, which the forward-only graph already provides. It should be attempted only after H1 and
H3 report, since it is a larger change to a frozen recipe.

### Rejected without implementation

**Richer inputs.** Every added input factor so far has cost category recoverability. The
documented next-steps idea of a spatio-temporal hypergraph fusion inherits this risk directly: it adds
coordinate and periodic-time encoders to the same objective, and on this evidence the objective would
trade the category away for them. The measurement here is a concrete, cheap pre-test for that
much larger design, and it belongs in the record before that work is scheduled. Note also that the
repository's own wall states that substrate gains wash out in the joint regime, so the value of any
substrate change is representation quality and generality, not the joint score.

## Order of execution

1. The width grid, because it tests H3 and H3 competes with everything else.
2. The two implemented terms, separately and together, each judged on all factors at once and on the
   downstream score, never on the category axis alone.
3. H4 only if the first two fail, and framed as a change to the objective rather than to the inputs.
