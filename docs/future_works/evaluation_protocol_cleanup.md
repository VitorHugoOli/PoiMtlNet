# Evaluation protocol cleanup — protect the reporting fold from checkpoint selection

**Date drafted:** 2026-07-24
**Protocol review:** 2026-07-24

## Decision

Keep the five user-disjoint outer folds. They are not the error.

For the final, confirmatory evaluation, use **5-fold outer
`StratifiedGroupKFold` + one user-disjoint inner validation split inside each
outer development set**. Use the inner split for checkpoint/epoch selection,
then evaluate the outer fold exactly once. When compute permits, refit on the
complete outer development set for the selected number of epochs before that
one-shot evaluation.

This is the best cost/rigour trade-off for this codebase **provided that the
model recipes and hyperparameters are frozen before the confirmatory run**.
Use full nested CV only when the final evaluation itself still selects among
architectures, heads, learning rates, batch sizes, loss weights, or other
configurations.

The resulting default proportions should be approximately:

| Role | Share of all users | Purpose |
|---|---:|---|
| outer test | 20% | one-shot reporting only |
| inner validation | 10% | checkpoint/epoch selection only |
| inner training | 70% | fit the checkpoint-selection model |

One deterministic way to obtain the 10% inner set is to run an 8-way
`StratifiedGroupKFold` on the 80% outer-development block and use one of its
folds as validation: `12.5% × 80% = 10%` of the full data. The outer-split seed,
inner-split seed, and model-initialisation seed must be separate fields.

### Why the current five-fold idea is still valuable

The five outer models test every eligible user out of sample exactly once.
This is stronger and more data-efficient than a single 20% holdout. The
problem is only that the current runner looks at that held-out user's fold
after every epoch and reports the maximum found there.

The five fold scores are **not five independent experiments**, however. Their
training sets overlap heavily and the five test folds are complementary pieces
of the same dataset. Likewise, four initialisation seeds over the same five
frozen folds produce 20 fitted models, not 20 independent statistical
replicates. Do not use a naive fold-level standard error or pool
`seed × fold` as inferential `n=20`.

### What “five folds” means in the current code

The implementation does **not** divide the dataset into five disjoint
mini-datasets and then split each mini-dataset into train/validation.
`StratifiedGroupKFold.split()` returns the complement of each held-out block
as `train_idx`.

For a simplified ten-row dataset, assuming one distinct user per row:

```text
A = rows 1–2     B = rows 3–4     C = rows 5–6
D = rows 7–8     E = rows 9–10

execution 1: train = B+C+D+E; val = A
execution 2: train = A+C+D+E; val = B
execution 3: train = A+B+D+E; val = C
execution 4: train = A+B+C+E; val = D
execution 5: train = A+B+C+D; val = E
```

Each execution creates a fresh model, so weights and reported results do not
flow from one fold to another. Nevertheless, executions 1 and 2 share
`C+D+E`: 60% of the complete dataset, or 75% of each execution's training
set. The executions are operationally separate but their performance
estimates are statistically correlated.

Current code evidence:

- `src/data/folds.py::_create_check2hgi_mtl_folds` calls
  `StratifiedGroupKFold(...).split(X, y_cat, groups=userids)` and stores the
  returned `(train_idx, val_idx)` pair.
- `src/training/runners/mtl_cv.py::train_with_cross_validation` creates a new
  model inside the fold loop.
- Validation runs every epoch, updates the best-model tracker, and the final
  report evaluates that selected state on the same validation loader.

Thus standard five-fold averaging is useful, but the current `val` has two
incompatible roles: checkpoint selector and final reporting set.

### Split seed and model seed are different experimental factors

The code currently has two behaviours:

1. If a valid frozen fold cache exists, `scripts/train.py::_resolve_folds`
   loads it, so all model seeds use the same user partition.
2. If the cache is absent or stale, `_resolve_folds` constructs
   `FoldCreator(seed=config.seed)`. The requested model seed then also changes
   the user partition.

At the 2026-07-24 audit, this workspace did not contain canonical
`fold_indices_*.pt` caches for the principal AL/AZ/FL/CA Check2HGI inputs.
Fresh local runs on those inputs would therefore take the second path unless
an explicit `--folds-path` is supplied. Remote paper runs may have used frozen
artefacts, but their equality must be established from manifests/digests
rather than inferred from the intended protocol.

The corrected protocol must expose and record separate values:

```text
split_seed = 42             # immutable outer and inner user partitions
model_seed = 0, 1, 7, 100   # initialization/training randomness
```

All paired model arms must use the same frozen outer/inner indices. A run must
fail closed, rather than silently regenerate partitions from `model_seed`, if
confirmatory frozen folds are unavailable.

## What's deferred

Refactor the training/evaluation codepath so the fold used for **final reporting** is not also
used to **choose the best checkpoint / epoch**.

Today the MTL runner uses a 2-way fold split only (`train`, `val`) and reuses the same held-out
fold both to select the checkpoint and to produce the reported score. The same protocol must be
audited and aligned across the dedicated runners as well, so the codebase has one explicit and
documented evaluation contract.

Protocol targets:

1. **Frozen recipes / checkpoint selection only — recommended default:** outer 5-fold test +
   one inner user-disjoint validation split.
2. **Configuration search remains inside final evaluation — required:** full nested CV, with all
   candidate selection repeated independently inside every outer fold.
3. **Locked epoch — valid but secondary:** a fixed epoch selected without inspecting any outer
   test result. This retains more training data, but is only clean if the rule was genuinely
   locked in advance; it cannot be declared retroactively after reading the current curves.

## Evidence from this project

The final overlap-board datasets are large in rows/windows, while the split
unit remains the user:

| Dataset | Users | Prediction windows | Regions |
|---|---:|---:|---:|
| Alabama | 3,858 | 96,326 | 1,109 |
| Arizona | 7,869 | 200,895 | 1,547 |
| Florida | 21,052 | 1,274,418 | 4,703 |
| California | 37,090 | 2,925,466 | 8,501 |
| Texas | 38,644 | 3,830,414 | 6,553 |
| Istanbul | 23,694 | 271,666 | 520 |

Source: `docs/studies/second_dataset/STATS_T1.md` and the dissertation's
Chapter 5 dataset table. Not every raw user necessarily contributes an
eligible window, so manifests must also record the eligible-user count.

Even the older, smaller non-overlap Alabama input currently materialised in
`output/check2hgi/alabama/input/next.parquet` contains 12,709 rows from 1,622
users. A simulated 70/10/20 user-disjoint split on that input leaves, on
average, approximately 8,896 train rows / 1,138 users, 1,271 inner-validation
rows / 160 users, and 2,542 outer-test rows / 324 users. Thus the proposed
inner holdout is viable on the smallest state; full inner CV would spend much
more compute without creating more independent data.

The code audit confirms the present coupling:

- `src/data/folds.py` creates only `train_indices` and `val_indices`.
- `src/training/runners/mtl_cv.py` evaluates `*.val.dataloader` every epoch,
  records the best state from those values, then sends the same loaders to
  `validation_best_model`.
- `docs/context/METRICS.md` currently defines the per-fold result as the
  maximum over epochs `>= 5`; this is selection on the reporting observations,
  not a test estimate.
- `articles/dissertacao/src_utils/dissertation_review.md` identifies the same
  issue as REV-003 and separately identifies the `seed × fold` inferential-unit
  problem as REV-014.

## Exact recommended workflow

For every dataset, model family, outer fold, and model-initialisation seed:

1. **Create one immutable outer partition.** Use five user-disjoint,
   category-stratified folds. Reuse the same user partition across MTL/STL
   arms and across initialisation seeds. Record a dedicated `split_seed`;
   never derive it from `model_seed`.
2. **Seal the outer test fold.** The training loop, scheduler, early stopping,
   checkpoint selector, hyperparameter selector, and human operator must not
   read its labels or metrics.
3. **Split the outer development users 70/10.** Create a deterministic inner
   validation set shared by both MTL tasks and by all paired comparator arms.
4. **Fit and select internally.** Select the epoch/checkpoint from inner
   validation only. For MTL, use the frozen joint selector; record per-task
   inner metrics as diagnostics.
5. **Preferred refit.** Reinitialise the model and train on the complete 80%
   outer-development block for the selected number of epochs. Recompute every
   learned train-only artefact, including class weights and transition priors,
   from this block. This estimates the intended procedure without discarding
   10% of the data at final fit time.
6. **Test once.** Load the refitted model and evaluate the outer test loader a
   single time. Save predictions, targets, user IDs, fold ID, initialisation
   seed, selected epoch, inner-selection metric, and outer-test metrics.
7. **Never promote an outer result back into selection.** If a configuration
   changes after viewing outer results, the run becomes exploratory and needs
   a new untouched confirmatory dataset or a nested selection analysis.

If refitting doubles the cost beyond the available budget, evaluating the
inner-selected checkpoint directly on the outer test is still statistically
clean. It must then be described as performance of a procedure trained on
approximately 70% of the users, rather than the preferred 80%-refit procedure.

### When full nested CV is required

The inexpensive protocol above is sufficient for comparing **already frozen**
training procedures. It is insufficient for a per-dataset "ceiling" selected
from several learning rates, batch sizes, heads, or loss weights during the
same final study.

For such a ceiling, each outer fold must independently:

1. run the complete candidate search using only inner folds;
2. select the candidate from mean inner-validation performance;
3. refit that candidate on the complete outer-development set; and
4. evaluate it once on the outer test fold.

The candidate set and tie-breaking rule must be frozen first. Selecting one
global winner from all five outer results and then reporting those same five
results is not nested evaluation.

Given the millions of windows in the large states, the recommended research
strategy is therefore:

- use legacy results only to define a small, frozen shortlist;
- use the 70/10/20 outer-test protocol for the confirmatory comparison;
- reserve full nested CV for claims that explicitly depend on per-dataset
  configuration search; and
- confirm the locked winner once on a dataset not used during the long
  development history. Re-running old states with corrected splitting removes
  checkpoint bias, but it does not make those heavily inspected states
  pristine external confirmation.

## Reporting and statistical analysis

### Point estimates

Save row-level out-of-fold predictions. For each initialisation seed,
concatenate the five outer-test prediction files and compute the metric once
over all out-of-fold observations:

- global macro-F1 for `next_category`;
- global Acc@10 and MRR for `next_region`.

This is preferable to an unweighted average of five fold metrics. In
particular, the average of fold macro-F1 values is not generally equal to
macro-F1 over all out-of-fold predictions. Keep per-fold values as
heterogeneity diagnostics, not as five independent trials.

Across four initialisation seeds, report the mean and SD of the four pooled
out-of-fold estimates. Phrase the design as **four initialisations over five
fixed outer folds (20 fits)**. If a seed-level paired test is used, its
inferential sample size is four, not twenty.

### If Wilcoxon over initialization randomness is required

For one dataset, one independent Wilcoxon pair can be defined as one complete
five-fold experiment under one independently chosen `model_seed`:

```text
model_seed 0  -> pool five outer folds -> Δ_0 = metric(MTL) - metric(STL)
model_seed 1  -> pool five outer folds -> Δ_1
...
model_seed R  -> pool five outer folds -> Δ_R
```

Run the paired Wilcoxon on `[Δ_0, ..., Δ_R]`; its inferential sample size is
the number of model seeds. Ten to twenty seeds provide a usable exact test,
subject to an effect-size/power calculation and the multiplicity family.
They require 50–100 outer-fold fits per model, so this is not the default
recommendation for California and Texas.

This test answers a narrow but valid question: whether the paired performance
difference is robust over model initialization/training randomness,
conditional on the fixed dataset and split. It does not turn the fixed users
or one geographic domain into new independent samples.

### Uncertainty and paired comparison

The most useful cost-aware statistical artefact is a paired, user-clustered
bootstrap over the saved outer-test predictions:

1. sample users with replacement within a dataset;
2. carry all windows and all seed predictions for each selected user;
3. recompute each model's global metric and the paired difference;
4. average the paired difference across the four fixed initialisation seeds;
5. repeat at least 5,000 times and report a 95% interval.

This interval quantifies user-sampling uncertainty conditional on the frozen
outer partitions and the four fitted seeds. Report seed SD separately because
the user bootstrap does not estimate model-initialisation uncertainty or the
full variability of drawing new training sets.

For a frequentist superiority p-value without new fits, use a paired
user-cluster randomization test: independently swap the two models' complete
prediction bundles for each user, retaining all that user's rows and all four
seeds together, then recompute the global metric difference. Use at least
10,000 permutations. The bootstrap supplies the effect interval; the
randomization supplies the p-value. Both are conditional on the fitted
models, so retain the four per-seed effects and their SD as the separate
optimization-stability analysis.

For non-inferiority/equivalence, compare the paired user-bootstrap interval
with the pre-specified practical margin. For superiority families, apply the
pre-specified multiplicity correction to the cluster-randomization p-values.
Do not infer equivalence from a non-significant superiority test.

If split sensitivity itself is a research question, repeat the complete
outer protocol with a few independently generated **group partitions**.
Predictions for the same user across repeats remain correlated; resample the
user as a cluster and never count `repeat × seed × fold` cells as independent
observations.

### What the five folds do and do not prove

- They do ensure that all eligible users receive out-of-sample predictions.
- They reduce dependence on one lucky train/test split.
- They do not constitute five independent datasets or five independent
  replications.
- More initialization seeds measure optimization randomness, not uncertainty
  over users or geographic domains.
- Replication across genuinely distinct states/cities is stronger evidence
  for cross-domain generalisation than manufacturing a very small p-value by
  pooling correlated fold scores.

## Cost-aware recommendation for California and Texas

Do **not** spend days creating ten or twenty seeds merely to recover a
Wilcoxon label. The recommended confirmatory package for the two largest
states is:

1. five frozen outer user folds;
2. four paired model seeds, kept as four complete five-fold experiments;
3. inner validation with **no refit** for the default budget path, so each
   outer fold still requires one training run;
4. pooled outer out-of-fold predictions saved with user IDs for every model
   and seed;
5. paired user-cluster bootstrap interval plus paired user-cluster
   randomization p-value;
6. four seed-level paired effects reported in full, without claiming
   seed-level `n=20`; and
7. replication of the direction across states/cities as the evidence for
   geographic transfer.

The no-refit path is not computationally larger than the present runner in
number of trained models. Per epoch it trains on about 70% and validates on
about 10%, instead of training on 80% and evaluating 20%; the sealed 20% test
is evaluated once after selection. It estimates the explicitly declared
70%-training procedure.

An alternative for an already locked recipe is to choose a fixed epoch rule
on development states, freeze it before California/Texas, train each outer
model on the full 80% outer-development block, and test once. This avoids both
the inner-validation holdout and the refit, but it is valid only if the epoch
rule is fixed without consulting California/Texas outer results.

Two literature-backed alternatives exist but are secondary here:

- the Nadeau–Bengio corrected resampled test or a correlated Bayesian t-test
  explicitly models correlation from overlapping training sets and can
  analyse fold-level differences without pretending independence;
- Dietterich's 5×2cv test requires only ten paired outer fits, but trains on
  50% splits, changes the target learning-curve regime, and is unattractive
  for sparse high-cardinality region prediction.

If a venue insists specifically on a Wilcoxon p-value whose pairs are model
initializations, additional model seeds are unavoidable. There is no valid
transformation that turns the existing four seeds × five correlated folds
into twenty independent Wilcoxon pairs.

## Literature basis

- Cawley and Talbot (2010), [On Over-fitting in Model Selection and Subsequent
  Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html):
  optimisation of a noisy selection criterion can create bias comparable to
  the differences between algorithms.
- Varma and Simon (2006), [Bias in error estimation when using cross-validation
  for model selection](https://doi.org/10.1186/1471-2105-7-91): the same CV
  result used for tuning is biased; nested CV substantially reduces the bias.
- Bengio and Grandvalet (2004), [No Unbiased Estimator of the Variance of
  K-Fold Cross-Validation](https://www.jmlr.org/papers/v5/grandvalet04a.html):
  overlapping CV training sets induce correlations that make naive variance
  estimates too small.
- Nadeau and Bengio (2003), [Inference for the Generalization
  Error](https://doi.org/10.1023/A:1024068626366): comparison tests must account
  for training-set variability, not only test examples.
- Tsamardinos et al. (2018), [Bootstrapping the out-of-sample predictions for
  efficient and accurate cross-validation](https://doi.org/10.1007/s10994-018-5714-4):
  pooled out-of-sample predictions support performance intervals and
  bias-aware comparison without treating folds as independent datasets.
- Corani et al. (2017), [Statistical comparison of classifiers through
  Bayesian hierarchical modelling](https://www.jmlr.org/papers/v18/16-305.html):
  correlated Bayesian tests explicitly model the dependence created by
  overlapping cross-validation training sets.
- Dietterich (1998), [Approximate Statistical Tests for Comparing Supervised
  Classification Learning Algorithms](https://doi.org/10.1162/089976698300017197):
  the 5×2cv test trades training-set size for a low-compute algorithm
  comparison with better type-I control than naive fold tests.

## Why deferred

1. **This is a protocol correction, not a one-line fix.** It touches fold creation, runner
   control-flow, history/report storage, and the interpretation of existing results.
2. **It is expensive to adopt retroactively.** Once the code path changes, any paper/dissertation
   number that depends on the old selection rule needs to be clearly labeled as legacy or
   regenerated under the new protocol.
3. **The current text already discloses the existing convention**, so the immediate need is
   documentation honesty; the codebase change should be done carefully instead of as a rushed
   pre-defense patch.

## Acceptance criterion

When picked up:

1. `src/data/folds.py` exposes an explicit protocol with separate roles for:
   - outer reporting fold
   - inner validation data used for checkpoint/epoch selection
   - training data
2. `src/training/runners/mtl_cv.py`, `category_cv.py`, and `next_cv.py` no longer report metrics
   from the same observations that decided the selected checkpoint.
3. `scripts/train.py` and run manifests record which evaluation protocol was used
   (`legacy_same_fold_selection` vs `outer_test_inner_val`, or equivalent), plus
   distinct `split_seed`, `inner_split_seed`, and `model_seed` fields.
4. Stored fold artefacts make the selection path auditable: selected epoch, inner-val metric, and
   final outer-fold metric are saved distinctly. Row-level outer-test predictions include user IDs.
5. At least one smoke experiment per task verifies the new path end-to-end and updates the
   documentation that describes the protocol.
6. No runner executes the outer-test loader inside an epoch loop or before the final checkpoint has
   been irreversibly selected.
7. Analysis computes pooled out-of-fold metrics and does not treat fold or `seed × fold` cells as
   independent inferential observations.
8. Confirmatory execution fails if the frozen split artefact is absent/stale; it never silently
   regenerates folds from `model_seed`.

## Live docs the work would touch

- `docs/CONCERNS.md` — close or downgrade the protocol-leakage concern
- `docs/CLAIMS_AND_HYPOTHESES.md` / dissertation review notes — align the wording of the
  limitation with the implementation
- `docs/results/RESULTS_TABLE.md` and paper/dissertation text — mark legacy numbers and, if
  rerun, cite the corrected protocol

## Pointers

- `src/data/folds.py`
- `src/training/runners/mtl_cv.py`
- `src/training/runners/category_cv.py`
- `src/training/runners/next_cv.py`
- `articles/dissertacao/src_utils/dissertation_review.md` (review item on held-out-fold reuse)
