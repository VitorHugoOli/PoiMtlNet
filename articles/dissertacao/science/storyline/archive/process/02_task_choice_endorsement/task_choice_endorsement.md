# Endorsing the task choice in the storyline

> **What this file is.** The author raised a concern: the dissertation *changes the tasks it
> predicts*, and the storyline must endorse that choice well, because the honest motivation ("these
> tasks are more useful in the literature and more convergent with next-POI prediction") is currently
> under-argued and defensively phrased. This file (a) states the task choice precisely and verifies
> it against source, (b) gives the positive, service-first argumentation the frame should carry, (c)
> surfaces a load-bearing subtlety the first review missed — the task *pair* itself evolved — and
> shows how to turn it from a hidden confound into a strength, and (d) lists the verified anchors and
> the sign-off flags.
>
> **Fail-closed status.** An OpenAlex sweep for a cleaner external anchor (next-category /
> next-region as end-targets) returned only noise; it produced **no new citable reference**, so no new
> citation is proposed. Everything below is built on sources already in the corpus and verified this
> session (CBIC `sections/intro.tex`+`method.tex`; CoUrb `sections/intro.tex`+`metodology.tex`;
> MobiWac `sections/02_related.tex`+`03_problem.tex`; drafted Fundamentals `2.1`). New connective
> sentences are marked **[NEEDS SIGN-OFF]**.

---

## 1. What actually changed — verified across all three papers

The first review treated "the exact next place is not predicted" as the whole of the task-scope
story, and filed it as a clean, closed thread. That was an under-reading. Read against source, the
task *pair* evolved across the arc:

| Paper | Task 1 | Task 2 | Pair character | Source (verified this session) |
|---|---|---|---|---|
| **CBIC** | POI category classification (**static**, non-sequential) | next category (sequential) | one static + one sequential | `CBIC___MTL/sections/intro.tex` L38–42, `method.tex` L36–54 |
| **CoUrb** | POI category classification (**static**) | next category (sequential) | **same pair as CBIC** | `CoUrb_2026/src_en/sections/intro.tex` L4–5 |
| **MobiWac** | next category (sequential) | **next region** (sequential) | two sequential "next-X" tasks | `[mobiwac]/src/sections/03_problem.tex`; `02_related.tex` |

Two facts are now firsthand-verified and load-bearing:

1. **The constant is next category; the second task changed.** Across the arc, next category is
   predicted in all three papers. The companion task went from *static category classification*
   (CBIC, CoUrb) to *next region* (MobiWac). The word "region" appears in CoUrb only as the HGI
   embedding hierarchy (the POI–region and region–city infomax losses) and in geographic
   descriptions — **never as a prediction task**. Next region as a *task* is genuinely new to
   MobiWac.
2. **CBIC blamed the null partly on the static-vs-sequential dissimilarity of its own task pair.**
   CBIC's introduction asks, verbatim, "can a single, shared representation effectively serve two
   tasks with such distinct underlying characteristics?" and names the risk that "forcing a shared
   encoder to learn features for both a static and a sequential task could result in negative
   transfer." That is the dissimilarity of *category-classification (static) + next-category
   (sequential)*.

Put together: the CBIC → MobiWac reversal ("MTL finally helps") changed **three** things at once —
the representation (place-level → check-in-level), the sharing topology (hard sharing + FiLM →
cross-attention two-stream + private spatial path), **and** the task pair (one static + one
sequential → two sequential). The arc's headline credits the first two. The third is currently
invisible in the frame. This is the side of the story the concern correctly sensed was being lost.

---

## 2. Why the tasks are the right object — the positive argument (service-first)

The author's honest motivation is the correct spine, and it is *stronger* than the defensive phrasing
currently in MobiWac §3 ("not to make the task easier"). State it positively, and state it early.
The material is verified and already in the corpus; the work is assembly and placement, not new
claims.

**2.1 The two tasks are what a mobility-aware service can act on.** A service does not need the exact
next venue to be useful. It needs to know *what kind of place* the user is heading to (to prepare
content, offers, or capacity) and *which part of the city* (to place or provision resources). MobiWac
§1/§3 already frame category as intent ("what the user wants") and region as location ("where to
prepare"). This is the "useful in the literature" half of the author's motivation, and it is the
honest reason the tasks were chosen. *Frame move:* lift this into the Introduction's stakes paragraph
(recommendation G-9 in the pass-1 review), so the reader knows in Chapter 1 what the predictions buy.

**2.2 The two tasks are the coarse, learnable properties of the next visit — convergent with
next-POI, not a retreat from it.** Next place is one specific realization; category and region are
two orthogonal *coordinates* of that same next visit (its semantic type and its spatial cell). This
is the "convergent with next-POI" half of the motivation, and it is verifiable: the field's canonical
next-place systems *already compute* both signals internally — HMT-GRN predicts region to constrain a
beam search over places, and CatDM predicts category to prune the place candidate set
(`Lim2022`, `yu2020catdm`, both in MobiWac's bib and in the drafted §2.1). The dissertation's move is
to promote those two internal signals from *means* (intermediate steps toward a place) to *ends*
(the prediction targets themselves) — a stance a smaller body of work already takes for each target
alone (`zhu2022drrgnn` for region, `capanema2023poirgnn` for category). *Frame move:* the drafted
§2.1 already makes this "means → ends" argument well; the Introduction should echo its one-sentence
form so the task choice reads as a deliberate position, not an omission.

**2.3 Region over a map partition is the standard mobility formulation, not an invention.** MobiWac
§2.2 states, verifiably, that predicting over a partition of the map "is also the standard formulation
in the human-mobility literature, with a grid cell as the target [luca2021mobilitysurvey]; our
next-region task substitutes official neighborhood-scale units for grid cells." This pre-empts the
"why census tracts / mahalle?" question: the dissertation uses administratively meaningful units
instead of arbitrary grid squares, but the *task shape* (predict the next spatial cell) is
canonical. *Frame move:* keep this sentence in §2.1/§2.4 and reference it once in the Introduction.

**2.4 The choice does not make the problem easier — it makes it harder.** This is the rebuttal to the
sharpest suspicion ("you switched to easier tasks to manufacture a win"), and it is quantitative and
verified: next region has, depending on the dataset, **from a few hundred to several thousand
classes** (520 for Istanbul up to 8,501 for California), whereas the *dropped* static task had **seven**.
The task set got harder on the second task, not easier. Next place (the hardest, tens of thousands of
candidates) was dropped, but it was never in the MTL pair to begin with — CBIC and CoUrb never
predicted next place either. *Frame move:* state the class-count contrast plainly wherever the task
choice is defended; it is the single most disarming fact available.

---

## 3. The confound, and how to dissolve it honestly (the load-bearing move)

**The suspicion, stated plainly.** MTL did not help when the pair was static + sequential (CBIC).
MTL helped when the pair was sequential + sequential (MobiWac). CBIC itself blamed the static/
sequential dissimilarity. So a skeptic — a banca member is the relevant one — can argue: *the
reversal is because you made the two tasks more similar, not because you fixed the representation.*
If the frame credits the win to representation alone while quietly swapping the task pair, that reads
as a moved goalpost, and it is the kind of thing an examiner enjoys finding.

This must not be smoothed over. But it does not weaken the thesis — handled correctly, it deepens it.
There are three honest responses, and they compound.

**3.1 The strongest response: the task change is a *corollary* of the representation thesis, not an
independent knob. [NEEDS SIGN-OFF]**
The central idea of the arc is that a place has no single visit-independent identity — the same
coffee shop is a weekday lunch stop and a Saturday-night spot, and one per-place vector cannot be
right for both. Static category classification asks for one label per POI from stable,
visit-independent features, which is the least natural task to pose on top of a per-visit
representation. **[Correction, pass-2 critic — POI/mobility expert.]** An earlier draft of this file
said the static task becomes "incoherent" under a check-in representation; that overreaches. The task
does not become impossible: one can *pool* the visit vectors of a POI into a single POI vector and
classify that, which is exactly what CoUrb's own POI Encoder does (it generates the embedding per
category and remaps it to each POI). So the honest claim is weaker but still load-bearing: under a
per-visit representation the *sequential* category task (what kind of place comes next) is the natural
fit, and the static per-POI task requires an extra pooling step that discards the per-visit signal the
representation was built to carry. The coherent, natural pair under a per-visit representation is
therefore two next-visit properties — next category and its spatial companion, next region. **So the
task pair did not change independently of the
representation; it changed because the representation changed.** The task refinement is the
representation thesis applied a second time — to the definition of the problem rather than to the
encoder. Framed this way, the two-sequential-task pair is not a convenient choice that happened to
help MTL; it is the task pair the representation *forces*. This is the connective claim that turns
the confound into the arc's most intellectually satisfying beat, and it needs author sign-off before
it enters the text (it is a new framing, strongly supported by MobiWac §2.1 but not verbatim in any
source).

**3.2 The controlled evidence for representation-dominance lives in CoUrb, and is untouched by the
task switch.** CoUrb holds the task pair *fixed* (the original static + sequential pair, identical to
CBIC) and changes *only* the input representation — and category performance rises sharply. That is a
true controlled comparison: same tasks, same architecture, representation varied. So
"representation is the dominant factor" is established **on the original, dissimilar pair**, before
any task change occurs. The task switch happens later (MobiWac) and cannot retroactively explain
CoUrb's result. *This is why CoUrb is the load-bearing control of the whole dissertation*, and it is
another reason to elevate its role in the frame (pass-1 recommendation G-13). The honest logline is:
representation-dominance is proven under the hard (dissimilar) pair by CoUrb; the joint *win* is then
delivered under the pair the check-in representation naturally induces by MobiWac.

**3.3 The mechanism is measured, not assumed: the two final tasks do not conflict.** MobiWac reports
that the two tasks' training gradients are near-orthogonal on the shared trunk, so there is no
*directional* conflict for a gradient balancer to resolve — which is *why* balancers (PCGrad,
Nash-MTL) do not beat a tuned fixed weighting on this pair. **The number must travel with its source's
scope** (MobiWac `02_related.tex` L89–94): the cosine similarity "averages +0.001 across training
(four seeds each on three of our six datasets, per-dataset means within ±0.003)," it was "measured
during development on the same joint architecture (on an earlier preparation of the data)," and the
source states it is "a finding for this pair of tasks, not a general rule." Two precisions the critic
pass (MTL expert) added: (i) cosine captures *directional* conflict only — magnitude imbalance is not
measured by it, and Adam already partially normalizes that — so the honest phrasing is "no directional
conflict," not "no conflict"; (ii) near-orthogonality is evidence that negative transfer is *absent*,
but it is equally evidence against gradient-level *positive* transfer, so this supports "sharing
stopped hurting," not "the tasks teach each other." With that scope attached, this is quantitative
evidence that the two sequential tasks coexist without destructive interference, and it is a finding in its
own right (pass-1 recommendation G-10 / D-WORTH-3). It supports "sharing stops hurting" with a
measured mechanism rather than an assertion.

**3.4 What honesty still requires the frame to concede.** Even with 3.1–3.3, the dissertation does
**not** run a single controlled ablation that holds the task pair fixed while swapping to the
check-in representation *and* producing the joint win (CoUrb fixes the pair but reports category
gains, not the joint-beats-both result; MobiWac produces the joint win but on the new pair). So the
decomposition of "representation" from "task-homogeneity" in the *final win* rests on the conceptual
argument (3.1) plus the CoUrb control (3.2), not on one clean experiment. The frame should **say
this** — as a scope statement in Chapter 2 or a limitation in Chapter 6 — and can point to the
controlled ablation (check-in representation on the original static+sequential pair) as future work.
Conceding it costs nothing and removes the examiner's opening; hiding it is the only way it becomes
dangerous.

---

## 4. The endorsement, assembled (what the frame should say, in order)

A reader should meet the task choice as a *position the dissertation argues*, not a scope note. The
honest, verified sequence:

1. **Stakes (Ch.1):** a mobility-aware service acts on *what kind of place* and *which part of the
   city*; it does not need the exact venue. (§2.1 already; lift to Intro.)
2. **Convergence (Ch.1/§2.1):** category and region are two coordinates of the next visit that the
   field's next-place systems already compute internally (HMT-GRN, CatDM); the dissertation promotes
   them from means to co-equal ends (`Lim2022`, `yu2020catdm`, `zhu2022drrgnn`,
   `capanema2023poirgnn`). (§2.1 already; echo in Intro.)
3. **Not easier (§2.1/§3):** region spans hundreds to thousands of classes vs seven for the dropped
   static task; the task set got harder, and next place was never in the MTL pair. (Verified.)
4. **Standard formulation (§2.1/§2.4):** predicting over a map partition is canonical; the
   dissertation substitutes administrative units for grid cells (`luca2021mobilitysurvey`). (§2.2
   MobiWac already.)
5. **The corollary (Ch.1 arc ¶ and/or Ch.5 recap) [NEEDS SIGN-OFF]:** the task refinement follows
   from the representation thesis — a per-visit representation makes the static task incoherent, so
   the coherent pair is two next-visit properties. This is the beat that pre-empts the confound.
6. **The concession (Ch.2 scope or Ch.6 limitation):** representation and task-homogeneity are not
   separated by a single controlled ablation in the final win; CoUrb is the control that isolates
   representation on the fixed pair, and a fixed-pair ablation under the check-in representation is
   future work.

---

## 5. Verified anchors and flags

**Citable, already in the corpus, verified this session** (no new references introduced):

| Key | Supports | Where verified |
|---|---|---|
| `luca2021mobilitysurvey` | map-partition prediction is the standard mobility formulation; DL-for-mobility spans several tasks | MobiWac `02_related.tex` L45; §2.1 draft |
| `Lim2022` (HMT-GRN) | region predicted as a *means* (beam-search constraint toward place) | §2.1 draft; MobiWac bib |
| `yu2020catdm` (CatDM) | category predicted as a *means* (candidate pruning toward place) | §2.1 draft; MobiWac bib |
| `zhu2022drrgnn` | next region as an *end* target in its own right | §2.1 draft; MobiWac bib |
| `capanema2023poirgnn` | next category as an *end* target | §2.1 draft (errata-corrected key) |
| `silva2019urbancomputing`, `song2010limits`, `cho2011gowalla` | LBSN stakes + 93% predictability ceiling | §2.1 draft (song verified firsthand) |

**Sign-off flags:**
- **[NEEDS SIGN-OFF]** — §3.1 / §4-step-5: "the task refinement is a corollary of the representation
  thesis (a per-visit representation makes static category classification incoherent)." New connective
  framing. Route through AGENT_GUARDRAILS §3 (C2) + personas 07 (claim honesty) + 14 (adversarial
  advisor).
- **[NEEDS SIGN-OFF]** — §4-step-2: "category and region are two coordinates of the next visit …
  promoted from means to ends." The means→ends framing is in §2.1; the "two coordinates of one visit"
  phrasing is new connective language.
- **No [VERIFY] external flags** — the OpenAlex sweep produced no citable anchor; nothing external is
  asserted. The class-count figures (7 vs 520–8,501) must re-verify against the MobiWac source of
  truth at adaptation (N1), as with any number entering frame prose.

**Fail-closed note.** The most attractive possible move here would be to cite a survey that explicitly
canonizes "next category and next region" as the two standard coarse mobility tasks. I searched for
one and did not find a clean, openable anchor. I am therefore **not** asserting that such a consensus
exists; the argument above rests only on the verified means→ends and standard-partition claims. If
the author knows of such a survey, it can be added after opening and verifying it.
