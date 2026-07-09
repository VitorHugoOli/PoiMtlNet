Quote: "Across six datasets, five U.S. states and one non-U.S. city"
Comment: We Use two datasets(Gowalla and Massive steps), in the gowalla we extract the five U.S and in the massive steps
we extracted the non-U.S. My point is can we call this state extracted form the gowalla as a dataset is the correct
name ?

Qoute: "and a service that can anticipate
the next move can prepare ahead of time instead of reacting
after the fact."
Comment: Shouldn't we have a ref for this ?

Qoute: "Four of the six datasets are measured at
four seeds over five folds on both arms (the joint model
and the dedicated one); California and Texas are single-
seed and provisional."
Comment: Do we need to say this here ?

Qoute: "Overlapping windows give the category task more examples
to learn from..."
Comment: We also use overleap in the next-region, may should be: "Overlapping windows give the tasks more examples
to learn from..."

Qoute: "Third, the one component that could pass visit-to-
visit information is a region-transition prior: a table of how
often visits move from one region to the next, used to bias
region scores. An earlier version that used the whole dataset
inflated region accuracy by 13 to 27 points across states, which
is why any such table is built per fold, from training data
only, seed by seed. The joint and dedicated models reported
here run with this prior disabled entirely; it survives only
in the HMT-GRN baseline (Section V-D), with the per-fold
training-only build."
Comment: We are not using the region-transition prior, and we say this in the text, but my point is should we relly let
this in thext to say that is disable ? Cause leting this can rais a question: Why they were using it and turn off? Do
they develop on the reson of this ?

Qoute: "The place-level baseline representation is
pre-trained once on the whole dataset, exactly like ours;"
Comment: We are using the poi2vec, for the place-level. I may be wrong, but I am not remember of seen any comment to the
poi2vec, and the explanation why we inject it ? Should we develop this a little bit more ? I have a felling that is
loose on the text.

Qoute: "As in Section III, a mobility-
aware service acts on which region will be busy, not on a
single rank position. "
Comment: We are in section III, in this phrase. I think this phrase is a bit lost and wihout context. We are talinking
about messurements and from nothing we talk abou the `mobility-
aware service acts`

Qoute:"The HMT-GRN comparison is therefore a
region-native model, not a component-complete reproduction."
Comment: I belive that you are talking that the HMT-GRN in the way that we did can't be used as a head in the mtl,
right ? But this phrase is clear for a non contributor in the project ?

Qoute: "The second
role is a representation control, run end to end on Florida and
repeated in frozen form at Alabama, Arizona, and Istanbul,
to attribute the category gain to our specific design and not
to contextualization in general or to extra features: CTLE
[10], the closest prior contextual check-in embedding, in both
its end-to-end and frozen forms, and a feature-concatenation
control that appends raw per-visit features to a standard place
embedding under the same model."
Comment: This phrase is confusing, first you are writing about a representeation control(which one ?) then you turn to
take about the CTLE. I belive that this phrase desiver a re-write or a better explanation.


Qoute: "The same geometry
does not separate regions, which is exactly the point: the representation improves category separability but carries no
spatial structure, so its benefit is category-only and sets up
the joint study in Section VI-B."
Comment: Saying that our representation not carries aptial stucture is wrong , no ? We use this representation to train the next-region.

Qoute: "A controlled test isolates the
source: we freeze the region pathway at the start of training
so it cannot learn, and therefore cannot teach the category
task, yet the full category lift survives at Alabama, Arizona,
and Florida (within 0.3 of the joint model and far above the
dedicated single-task model). We therefore read the category
gain as a stronger shared trunk, obtained without a second
model to serve (one model, one forward pass), rather than
the region task teaching the category one; we report this as a
finding, not a hypothesis."
Comment: We need to take a lot of care with this part since can raised many questions like: "so why use a shared-trunk ?","If the results keep positive maybe the dedicated model is undertunned ?","And region loose something when not use the shared trunk ?"

Qoute: "Second, our representation
is trained once over all places. One slice of its behavior, visits
to places never seen during training, is the single effect we
cannot fully isolate. The same property sets the deployment
path: a served system would refresh the precomputed represen-
tation on a schedule, as embedding infrastructures commonly
do; a planned follow-up study trains the representation on each
fold’s training places only and extends it to embed unseen
visits directly, closing the remaining gap"
Comment: Eval if we can improve this explanation. And if is necessary.

Others:
- Are we explaining and do we need to explain that from the check2hgi is genrate two embedding that we feed diferrently for the category and region ?
- Do we explain about the the joint loss metric that we use in the gradient losses ?



---

Feel free to disagree and bring your points. The intution here is to bring the best of the papaer
