# Scientific explanation of the pairing result — article-ready framing

**Purpose**: the mechanistic explanation of why sample-aligned cross-task pairing HURTS
(AL cat −3.0) while random pairing (the champion default) wins, grounded in established
literature, with a paper-ready paragraph + citation set. Evidence base:
[`PAIRING_BATTERY.md`](PAIRING_BATTERY.md) (battery + re-tune sweep + advisor panel).
**Fact-checked 2026-07-02**: every empirical number below re-verified against the raw
per-epoch logs by an adversarial fact-checker; citations audited (all real, none
mis-attributed); reviewer-lens pass applied. Remaining pre-submission hardening items in §6.

## 1 · The phenomenon, precisely

In the MulT-style bidirectional cross-attention backbone, row *i* of the category stream
attends to row *i* of the region stream (and vice versa) within each batch. The **pairing
policy** decides what row *i* of the other stream contains:

- **aligned**: the SAME user-window's other view (its region-embedding sequence) — the
  deployment condition;
- **default (champion)**: a random other window (independent per-task shuffles) — the
  cross-read is re-sampled every epoch.

Measured (AL, v17, 4 seeds × 5 folds, peak-vs-peak): aligned −3.03 cat / −0.60 reg. A
machinery-matched *deranged* control (identical permutations/inits, partner rotated by one
position WITHIN each shuffled batch — i.e., every row cross-reads a random other window) is
**statistically null vs base** (cat −0.04 ± 0.08, reg +0.04 ± 0.10) — the harm is purely
*who* is read, not how. Trajectory diagnostics (battery val curves): aligned starts ABOVE
base — every fold shows an early lead (20/20 folds at epoch 2, per-fold leads up to
+5.2 pp; mean-curve lead +2.7 pp at epoch 2) — hovers near parity through ~ep 14, crosses
below permanently at ep ~15 on the mean curve (per-fold final crossovers ep 8–22, median
16), and peaks ~18 epochs earlier (ep ~23 vs ~41) at a **3.0 pp lower ceiling**. At ep 50,
aligned's TRAIN F1 is 79.6 vs base 69.5 (+10.1) while its val is 7.2 pp LOWER.
Hyperparameter re-tuning (schedule, wd, cat/shared LR, cross-attn dropout; 2 seeds ×
5 folds, max seed spread 0.17) recovers ≤0.8 of the 3.0. At FL (13× more windows) the
effect was null in the advisory A/B (single seed, v16 conditions; not re-run in the
battery). The trained champion is empirically insensitive to per-sample cross-stream
content at eval (FL roll probe, champion-G v16, cat-F1 Δ −0.004 — note the probe rolls the
unshuffled, user-contiguous val stream by 1, which mostly substitutes the same user's
adjacent window; a cross-user permutation probe at AL is the outstanding strengthener, §6).

## 2 · Mechanism: an instance-specific shortcut channel vs. structured input noise

**Aligned pairing opens a shortcut channel.** The region view of the same window is a
second, highly informative, *instance-specific* encoding of the very example being
classified. The cross-attention channel then lets the network lower training loss by
co-adapting the two views of each *training instance* — instance memorization — rather than
by extracting population-level sequence structure. Every signature matches:

- *Shortcut-first learning*: the early-epoch advantage (all 20 folds ahead at epoch 2, by
  up to +5.2 pp) followed by a permanent crossover at ep ~15 is the canonical shortcut
  trajectory — easy features are learned first and then starve the harder, generalizable
  ones (simplicity bias / gradient starvation).
- *Memorization*: the train–val divergence (train +10.1 pp over base at −7.2 pp val) is the
  standard capacity-finds-instance-solutions signature; consistently, the effect was absent
  at 13× the data (advisory FL) and resists regularization-style fixes (wd, dropout, LR) —
  the redundancy is in the *input construction*, not in excess parameters.
- *Ceiling, not schedule*: base's un-annealed ep-25 value already exceeds aligned's
  all-time peak; aligned extracts +0.02 pp from its entire second half of training.

**Random pairing turns the same channel into a regularizer.** With partner content drawn
independently of the target every epoch, the partner content is statistically independent
of the target — the channel carries no label information — and its sample-to-sample
variability acts as structured input noise on the shared blocks. Training with corrupted
inputs is a classical regularizer (Bishop (1995) formalizes the small-additive-noise limit
as Tikhonov regularization; the resampled-partner perturbation here is large and
structured, so we invoke the principle, not the equivalence). The trained network is
empirically insensitive to the channel's per-sample content (probe above), so the
cross-attention pathway functions during training as a stochastic regularizer rather than
as an inference-time information route — which is also why forcing alignment "loses
nothing": there is no per-sample cross-stream information being used that alignment would
improve.

**Why validation is not biased by this choice**: validation (and deployment) inputs are
row-aligned for ALL variants, and all reported numbers are per-task best-epoch peaks. The
comparison is apples-to-apples on the deployment distribution; only the *training*
distribution differs. Note "aligned at eval only" IS the champion configuration — and the
random-trained model outscores the aligned-trained model under that aligned evaluation.

## 3 · Literature grounding (all citations verified real; use distinct bib keys for the two Zhangs)

| Claim | Reference |
|---|---|
| Networks preferentially exploit predictive shortcuts | Geirhos et al., "Shortcut Learning in Deep Neural Networks", *Nature Machine Intelligence* 2, 665–673 (2020) |
| Simple/shortcut features are learned first and starve harder features | Shah et al., "The Pitfalls of Simplicity Bias in Neural Networks", NeurIPS 2020; Pezeshki et al., "Gradient Starvation: A Learning Proclivity in Neural Networks", NeurIPS 2021 |
| Deep nets readily memorize instance-specific solutions at small N | Zhang et al. (Chiyuan), "Understanding Deep Learning Requires Rethinking Generalization", ICLR 2017 [`zhang2017rethinking`]; Arpit et al., "A Closer Look at Memorization in Deep Networks", ICML 2017 |
| Training with corrupted/noisy inputs regularizes (small-additive-noise limit ≈ Tikhonov) | Bishop, "Training with Noise is Equivalent to Tikhonov Regularization", *Neural Computation* 7(1), 108–116 (1995) |
| Injecting content from unrelated training samples during training regularizes (nearest augmentation analog; NB mixup also interpolates the LABELS — our scheme leaves labels untouched) | Zhang et al. (Hongyi), "mixup: Beyond Empirical Risk Minimization", ICLR 2018 [`zhang2018mixup`] |
| Preventing co-adaptation improves generalization | Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 15 (2014) |
| Cross-modal transformers pair modalities of the same instance (the convention our result qualifies; NB MulT's "unaligned" refers to TEMPORAL alignment within an instance — modalities are still paired per instance) | Tsai et al., "Multimodal Transformer for Unaligned Multimodal Language Sequences", ACL 2019 |

**The positioning claim for the article**: same-instance pairing is the correct convention
when streams are complementary *modalities of one prediction* (MulT's setting). Our result
identifies a boundary condition: when the streams feed **separate task objectives** over
views derivable from the **same underlying sequence**, same-instance pairing creates an
instance-redundancy shortcut that memorizes at small N — and marginal (random) pairing is
the better training regime, acting as implicit augmentation on the cross-task channel.

## 4 · Paper-ready paragraph (lift/adapt into the design-justification or ablation section)

> A natural design question is whether the two task streams should be fed *sample-aligned*
> inputs — the same user-window's check-in and region views — within each training step,
> mirroring the inference condition. We tested this directly: with one shared permutation
> aligning the streams row-for-row, category macro-F1 drops by 3.0 pp at Alabama (4 seeds ×
> 5 folds, paired; regions −0.6 pp), while a *deranged* control — identical loader
> machinery and permutations with the partner stream rotated by one position within the
> shared shuffled permutation, so each row cross-reads a random other window — is
> statistically indistinguishable from the baseline (|Δ| ≤ 0.04 pp), isolating pairing
> semantics as the cause. The training signatures are those of shortcut learning [Geirhos
> et al., 2020] followed by instance memorization [Zhang et al., 2017; Arpit et al., 2017]:
> under aligned pairing, cross-attention supplies a second, instance-specific view of each
> training example, which the network exploits early (aligned is ahead of the baseline in
> every fold within the first epochs, by up to +5 pp, before crossing below permanently
> around epoch 15) and then memorizes — training F1 ends 10 pp above the baseline while
> validation F1 is 7 pp lower.
>
> Hyperparameter re-tuning for the aligned regime (schedule horizon, weight decay,
> per-group learning rates, attention dropout) reduces the gap by at most roughly a quarter
> (≤0.8 pp of the 3.0 pp), and the deficit was not observed at a state with roughly 13× the
> training windows (single-seed check) — consistent with an overfitting mechanism. Under
> the default independent shuffling, the partner content is re-sampled every epoch
> independently of the target, so the cross-task channel carries no label information and
> its variability acts as structured input noise — a classical regularizer [Bishop, 1995] —
> and the trained model is insensitive to its per-sample content (rolling the partner
> stream at evaluation changes category F1 by −0.004 at Florida). Validation inputs remain
> aligned in all variants, so the comparison is made on the deployment distribution; the
> choice concerns the training distribution only. We therefore train with independently
> shuffled task streams.

Suggested figure (compelling, data already on disk): mean validation cat-F1 vs epoch for
base vs aligned (± fold band) at AL — shows the early aligned lead, the crossover at
ep ~15, and the ceiling gap; trajectories in
`results/check2hgi_dk_ovl/alabama/<rundir>/metrics/fold*_next_category_val.csv` (rundirs in
`pairing_battery/summary.tsv`).

## 5 · Anticipated reviewer questions (and the answers we hold)

1. *"Is this just a bad implementation of alignment?"* — No: adversarial code re-audit
   found none; the deranged control shares 100% of the machinery and is statistically null
   vs baseline.
2. *"Did you tune for the aligned regime?"* — Yes, six arms including schedule-matched
   OneCycle horizon and attacks on the shared cross-attention group (its LR and dropout);
   best recovery 0.8 of 3.0 (2 seeds × 5 folds; every arm replicated across both seeds,
   max seed spread 0.17; PAIRING_BATTERY.md addendum).
3. *"Is the deficit a checkpoint/selection artifact?"* — No: all numbers are per-task
   best-epoch peaks (argmax over per-epoch validation, reproduced from raw logs);
   validation runs every epoch.
4. *"Does the model need alignment to use cross-task conditioning?"* — Per-sample
   conditioning (cat→reg posterior injection) is only usable when aligned (+0.47 reg over
   the aligned arm, 4/4 seeds) but does not recover the random-pairing baseline (paired
   means: reg −0.13 ± 0.34, cat −3.04; 1/4 seeds nominally above on reg) — the cross-task
   benefit in this regime is parameter-level, not per-sample content-level.
5. *"Would this hold at scale?"* — The effect is a small-N overfitting phenomenon: it was
   absent at Florida, which has ~13× the windows (evidence grade: advisory single-seed
   under v16; a 2-seed v17 FL aligned arm would harden this before submission).
6. *"Is Alabama idiosyncratic, or is this small-N?"* — Open at n=1 harmful dataset. The
   decisive cheap test: subsample Florida to Alabama's window count and rerun
   base-vs-aligned (2 seeds × 5 folds); a reappearing deficit makes N the causal moderator
   rather than state identity. A second small state (AZ) is the fallback. Not yet run.
7. *"You train with random partners but deploy aligned — is that not a train/serve
   mismatch?"* — Evaluation is aligned for every variant, so "aligned at eval only" IS the
   champion configuration; the trained model is insensitive to per-sample partner content
   (probe), and empirically the random-trained model outscores the aligned-trained model
   under aligned evaluation — the mismatch costs nothing measurable.

## 6 · Pre-submission hardening items (optional, cheap, prioritized)

1. **Cross-user permutation probe at AL** (eval-only; strengthens the invariance claim —
   the existing FL roll-by-1 probe mostly substitutes same-user adjacent windows because
   val rows are user-contiguous): randomly permute the partner stream at eval on an AL
   champion run. ~15 min.
2. **FL-subsample arm** (converts the size-scoping from correlational to causal): FL
   subsampled to AL's window count, base-vs-aligned, 2 seeds × 5 folds. ~2-3 h.
3. **FL 2-seed v17 aligned arm** (hardens the "null at 13× data" claim beyond the
   single-seed v16 advisory). ~2×2.5 h (FL is slow on the A40).
4. **AZ base-vs-aligned** (second small state; replicates the small-N harm). ~1 h.

## 7 · "Is it still valid MTL?" — the validity argument (user question, 2026-07-02)

**The concern**: if the trained model ignores per-sample cross-stream content and the reg
head is dominated by a private tower, is this genuinely one multi-task model where tasks
benefit from each other — or two separate models sharing a file?

**The test that decides it**: MTL is *invalid* (mere parameter bookkeeping) iff the jointly
trained model is no better than dedicated single-task models trained on the same substrate.
That comparison is exactly what the board measures, and it comes out in MTL's favor:

- **Category: the joint model BEATS the dedicated STL category ceiling at every state**
  (+3…+7.7 pp across the six-state board) — the definitional evidence of positive transfer.
- **Region: the joint model BEATS the dedicated STL region ceiling at the largest states**
  (CA +2.2, TX +2.1) and is non-inferior within the cardinality-dependent margin at the
  smaller ones (the Decision-C scoping) — while spending ONE training run and deploying ONE
  model for both predictions.

**What kind of MTL benefit is it?** The pairing program lets the paper say something most
MTL papers cannot — *which* transfer channel carries the benefit:

- **Per-sample content-level exchange: null.** The trained model is insensitive to
  cross-stream content (probe); forcing semantic pairing helps nothing (G0.1 battery);
  per-sample conditioning cannot beat the champion even de-confounded.
- **Parameter/representation-level transfer: alive and measured.** The shared blocks are
  optimized under BOTH task losses (gradients measured near-orthogonal — the tasks do not
  fight, no balancer needed), the second stream shapes the shared representation as an
  inductive bias, and its stochastic content regularizes the category pathway. This is
  Caruana's original mechanism — the auxiliary task as an inductive bias on a shared
  representation — not an exotic loophole: MTL's standard definition is joint optimization
  of multiple objectives over shared parameters (Caruana 1997; Ruder 2017), and it does NOT
  require per-sample information exchange. Regularization-style benefit from a co-trained
  task is a canonical, respected MTL mode (auxiliary-task learning: Jaderberg et al. 2017;
  Liebel & Körner 2018).

**The calibrated claim for the article** (this is what the evidence licenses):
"one jointly-trained model serves both tasks at-or-above the dedicated single-task
ceilings; the benefit arrives through shared-representation dynamics (inductive-bias +
regularization), not through per-sample cross-task content — which we demonstrate by
direct intervention on the pairing distribution." Do NOT claim per-sample knowledge
exchange ("the model uses task B's prediction to inform task A's") — that channel is
measured dead in this regime. The architecture's drift toward task-privacy (dual reg
tower) is not an embarrassment to hide but the regime finding itself: for this task pair,
light parameter sharing + joint optimization captures all the available synergy — hard
content coupling adds nothing and self-paired content actively hurts. This is precisely
the "beyond cross-task" thesis.

**Additional citations for this section** (verified real): Caruana, "Multitask Learning",
*Machine Learning* 28, 41–75 (1997); Ruder, "An Overview of Multi-Task Learning in Deep
Neural Networks", arXiv:1706.05098 (2017); Standley et al., "Which Tasks Should Be Learned
Together in Multi-task Learning?", ICML 2020 (task synergy is an empirical property of the
pair, not a given); Jaderberg et al., "Reinforcement Learning with Unsupervised Auxiliary
Tasks", ICLR 2017; Liebel & Körner, "Auxiliary Tasks in Multi-task Learning",
arXiv:1805.06334 (2018).
