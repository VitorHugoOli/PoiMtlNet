# Audit prompt — the balance study and the integrity study that preceded it

Give this file, unedited, to a fresh agent with repository read access and GPU access to the remote
host. It is written to be adversarial: the work below was produced by an agent that made several
errors it caught only after a reviewer forced the check, so the prior probability that something here
is still wrong is high, not low.

## Your role

You are auditing, not extending. Do not propose new experiments, new architectures, or improvements
until every item in the checklist has a verdict. Your deliverable is a findings list where each entry
is CONFIRMED, REFUTED, or UNVERIFIABLE with the command or file that settles it. An item you could not
settle is reported as unsettled; it is never rounded to "looks fine."

Two standing rules, both of which the original agent violated at least once:

1. **Never accept a number from prose.** Every value in the reports must be traced to a JSON written
   by a scorer, and the JSON must be traced to a run directory. If prose and JSON disagree, the JSON
   wins and the prose is a finding.
2. **Never compare across instruments.** A value from the one-fold study probe may not be differenced
   against a value from the five-fold dedicated model, and a train-free geometry value may not be
   differenced against either. The original agent broke this rule and had to strip the comparisons
   from three documents.

## What was claimed

Read these first, in order:

- `docs/studies/balance/FINDINGS.md` — the balance study: which cheap instrument predicts the model,
  the substitution mechanism, the effective-rank result.
- `articles/dissertacao/science/checkin_repr_scaling_and_capacity.md` — scaling, depth, width, region.
- `articles/dissertacao/science/consecutive_link_causal_audit.md` — the leak audit that started it.
- `docs/studies/archive/embedding_eval/L0_METHODOLOGY.md` — the repository's own rule for which
  evaluation rung may rank which task. It is load-bearing for the balance study.

Headline claims to audit:

| # | Claim | Where |
|---|---|---|
| C1 | The L1 linear probe tracks the dedicated model at rank correlation 0.949, and the train-free L0 geometry does not (0.600, two inversions) | FINDINGS §2 |
| C2 | Adding place identity moves category recoverability 0.988 to 0.733 while place goes 0.176 to 0.865: the representation substitutes rather than accumulates | FINDINGS §3 |
| C3 | The forward-only arm uses 8.43 of 64 dimensions by participation ratio, so capacity was never the shortage | FINDINGS §4 |
| C4 | Depth is not the constraint: one layer equals three within 0.014 macro-F1 | scaling doc |
| C5 | The scaling curve is flat: 0.54 points across an eight-fold range of training users, smaller than the 0.78 pooled fold standard deviation | scaling doc |
| C6 | Region identity harms, and the earlier contrary reading was retracted because the arms contained a zero-width region block | scaling doc |
| C7 | The width sweep is confounded, because the head fixes hidden_dim 256 and a max_lr tuned at dim 64 | this session, superseded by the width grid if it has landed |

## Checklist

### A. Does the code do what the prose says?

1. `scripts/balance/diag_budget.py` claims to use the repository's own L0 implementation. Confirm it
   imports `embedding_eval.geometry` and does not reimplement kNN or silhouette.
2. The B1 probe claims user-disjoint splits. Read `user_split` and confirm no user appears on both
   sides. Construct a case with one user and check the function's behaviour rather than assuming it.
3. `research/balance/balance_terms.py` claims that zero weights reproduce the frozen recipe exactly.
   Verify `balance_loss(z)` returns a tensor equal to zero, not merely small.
4. The variance-covariance term claims to detect collapse. Feed it a synthetic rank-3 matrix and an
   isotropic one and confirm the hinge separates them.
5. `--model-param hidden_dim` is claimed to reach the head. Confirm by parameter count, not by reading
   argument-parser code.

### B. Are the numbers traceable?

6. Every value in FINDINGS §2 and §3 must appear in `docs/results/balance/alabama_budget.json`. Check
   each one. The original agent verified this once; verify it again after any edit.
7. Every value in the scaling document must appear in a `stl_cat_ceiling_score.json` under
   `docs/results/check2hgi_integrity_v2/alabama/`. Check the depth, width, scaling and region rows.
8. For any arm, confirm that `build.json` records an `in_channels` consistent with the arm's stated
   composition. The zero-width region bug was invisible in the score and visible only here.

### C. Is the reasoning sound?

9. C1 rests on a rank correlation over FOUR arms. Four points is a weak basis for a correlation claim.
   Decide whether the claim is stated with appropriate hedging, and whether the two inversions are
   reported as prominently as the agreement.
10. C2 is the study's central mechanism claim. Ask whether an alternative explanation survives: could
    the category drop be caused by the probe's regularisation rather than by the representation? Test
    by refitting the category probe with a different regularisation strength on the same vectors.
11. C3 uses participation ratio as "dimensions used." Ask whether a representation could be useful
    while concentrated, which would make a low participation ratio uninformative rather than damning.
12. The category-preservation term optimises a linear category decoder, and the evaluation includes a
    linear category probe. Confirm the report states this circularity and does not present a category
    recoverability gain as evidence on its own.

### D. What is missing?

13. Only Alabama has the full arm set. Arizona was requested by the author. Report which claims have
    been checked at a second dataset and which have not.
14. The v18 comparison uses per-visit vectors recovered from the last slot of each window, which drops
    the first eight visits of every user. Assess whether that subset biases the comparison, since users
    with short histories are excluded entirely.
15. The joint place-and-elapsed arm has L1 numbers but, at the time of writing, no downstream score.

## Then, and only then

If the checklist leaves the mechanism standing, the open question is whether either implemented term
actually helps. The prediction on record is that the anti-collapse term will raise effective rank
without raising the downstream score, because five capacity-side explanations have already been
refuted. Design the test so that prediction can fail visibly.
