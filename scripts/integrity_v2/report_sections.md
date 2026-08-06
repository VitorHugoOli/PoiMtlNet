# The decision

The consecutive-visit link is a genuine leak for next-category prediction, and it is large. The
reported check-in-level representation lets the category head see a feature of the visit it is being
asked to predict. This is not a marginal contamination to be noted and moved past: at Alabama,
removing the target's own contribution costs the dedicated category model 34.71 macro-F1 points, and
cutting the user's path at the target, which is the graph an honest prediction-time protocol would
have, costs 35.43 points. Both intervals exclude the two-point margin by a wide distance.

The number that settles it is not the size of the drop but where the drop lands. Under the strict
readout the model reaches 0.2583 macro-F1, and a probe given nothing but the nine observed category
labels reaches 0.2669. The check-in-level representation, once the target's own features are
withheld, carries essentially nothing beyond the observed label sequence. What the reported
configuration adds over that benchmark is the target.

Two further comparisons close the alternative explanations.

The whole-dataset question is separately clean. Training the representation on every user, then
predicting for users the representation never saw, costs 0.33 points with an interval of -0.11 to
+1.56. That is inside the two-point margin and inside the build-to-build spread, which a second
representation seed puts at 0.05 points. So the transductive concern that the earlier audit raised
does not, by itself, inflate the reported number. When the training-user restriction and the causal
graph are applied together, the loss is 35.33 points, indistinguishable from the causal graph alone.
All of the effect is the edge; none of it is the users.

The controls hold. A destruction applied to a visit outside every slot's receptive field moves no
vector and costs exactly zero points, so the pipeline is not merely reacting to being perturbed. The
same destruction applied to a legitimately observed visit in the middle of the window costs 0.87
points, which is the scale of losing one available visit. The target's 34.71 points is forty times
that. A graph-attention encoder, used as a positive control, decodes the target category at 0.6256
against the reported encoder's 0.4345, confirming the instrument responds to a representation that
carries more of the target rather than saturating.

# What this study does not close

It is one fold, seed 0, at two datasets. The conclusion it supports is about the existence and the
approximate size of the effect, not about its variance across folds. The effect is roughly seventeen
times the two-point margin, so fold-to-fold variation of the size seen elsewhere in this project
cannot plausibly reverse the sign, but a precise magnitude would need the full five-fold, four-seed
protocol.

The strict readout is not a pure subtraction. Cutting the path at the target removes the leak, and it
also removes some legitimate structure, because the last three slots are recomputed with fewer
neighbours and a different degree normalization. The observed-visit placebo bounds that cost at 0.87
points, small against 35, but not zero. The honest statement is therefore that the strict number is a
lower bound on what a leak-free check-in-level representation could achieve, not an estimate of it.

Next region is not settled here, and the reason is structural rather than a matter of effort. The
region stream consumes region embeddings indexed by the historical places, so the target's region
vector is not an input and a region embedding is a per-region constant for the whole dataset. The
backward check-in edge cannot reach that stream. Region exposure, if any exists, would have to run
through whole-dataset representation training or through cross-attention from the category tower,
which are different channels needing different instruments.

The joint model is not measured here. Every number in this report comes from a dedicated
single-task category head trained on the representation under test. The joint model adds
cross-attention between towers, and whether the category tower's access to the target propagates
into the region tower is exactly the question this design cannot answer.

Three channels remain open and are not touched by this study: the place-level pooling that
aggregates a place's visits across the whole dataset at representation-training time; the masked
reconstruction objective, which by construction uses a place's later visits as targets; and the
region-transition prior, whose per-fold construction is a separate audit.

# What follows for the dissertation

The reported next-category numbers for the check-in-level representation cannot be presented as
prediction-time performance. Either the representation is rebuilt so that a visit's vector depends
only on visits at or before it, and the numbers are regenerated, or the reported numbers are
explicitly labelled as measured under a protocol in which the target visit is present in the graph.

The second option is weak, because the strict result sits below the label-only benchmark, which means
the contribution the dissertation attributes to the check-in-level representation for this task is
not demonstrated once the target is withheld.

The recommended path is the first: make the graph causal at construction, rebuild, and re-run. The
change is small and local; the exactness result in section 1 means it does not disturb the held-out
protocol, and the study-only builder in this study already saves the encoder needed to verify it.