# Second-pass audit: sides of the story that need more attention

> **Why this file exists.** The author's task-change concern exposed a systematic error in the first
> review: I graded the *discipline* of a scope choice (was it stated cleanly, held consistently?)
> instead of whether the choice was *justified* and whether it hid a confound. This file re-runs the
> arc with that corrected lens and reports every side I under-weighted, each with its firsthand
> verification status. Two of these are corrections to my own pass-1 claims. New connective framings
> are **[NEEDS SIGN-OFF]**; unverified claims are **[VERIFY]**.

---

## The error type, named

Pass 1 filed "next place is not predicted" as *thread T8 — opened → closed, disciplined throughout*.
The discipline was real, but the *justification* was thin and the *task-pair evolution* underneath it
was invisible. Grading discipline over justification is the mistake; below is every place it recurs.

---

## UW-1 · The task pair evolved, and the arc credits only two of three changes (MAJOR)

Fully developed in `02_task_choice_endorsement/`. In brief: the CBIC→MobiWac reversal changed the
representation, the sharing topology, **and** the task pair (static+sequential → sequential+
sequential), and CBIC itself blamed the static/sequential dissimilarity for the null. The frame
currently narrates two of the three changes. **Verified firsthand** (CBIC `intro.tex` L38–44,
`method.tex` L36–54; CoUrb `intro.tex` L4–5; MobiWac `03_problem.tex`). The honest resolution
(the task refinement is a *corollary* of the representation thesis, plus CoUrb as the fixed-pair
control) is in file 02. **This is the side most needing attention**, and it is the one the author
sensed.

## UW-2 · CoUrb never tests "does MTL help" — it only tests "does the representation help" (MAJOR)

**Verified firsthand** (CoUrb `metodology.tex` L36–38, `results.tex`, `conclusion.tex` L3): CoUrb's
*only* baseline is MTLNet. Every CoUrb comparison is ST-MTLNet (decomposed-representation MTL) vs
MTLNet (place-embedding MTL) — **both multi-task**. CoUrb runs no single-task model.

Consequence for the arc: the central research question ("does MTL help this task pair?") is answered
by CBIC (no, for that configuration) and by MobiWac (yes, for the check-in configuration). **CoUrb is
silent on it.** CoUrb answers a *different*, subordinate question ("is the representation the lever?")
and answers it cleanly. This is actually a *strength* for honesty — CoUrb is the controlled
representation experiment (architecture fixed, single-task question held constant, only the input
varied) — but it creates a specific frame risk: a reader carried by momentum may conclude "CoUrb
showed MTL works," which CoUrb did not show. The Ch.4 preface/recap must state CoUrb's question
precisely: *it isolates the representation effect; it does not revisit the MTL-vs-single-task
verdict.* **[NEEDS SIGN-OFF]** on the framing sentence that draws this boundary.

This also sharpens the logline (pass-1 lens B): the three clauses are "MTL does not beat single-task
on a place embedding" (CBIC) → "the representation, not the architecture, is why" (CoUrb, on the
representation axis only) → "with a check-in representation and redesigned sharing, MTL finally beats
single-task" (MobiWac). CoUrb's clause is about the *diagnosis*, not about MTL beating anything.

## UW-3 · CORRECTION to pass 1 — **RESOLVED 2026-07-23, verified firsthand**

> **Resolution.** The author provided the CoUrb codebase (`/Users/vitor/Desktop/mestrado/temp/
> tarik-new`). Verified firsthand: `PoiMtlNet_Novo/src/etl/mtl/create_fold.py` L190–199 reads
> `userid` and then **drops the column**; folds are built with plain
> `StratifiedKFold(n_splits, shuffle=True)` on sample rows stratified by class (L225–228), and
> `src/etl/next/fold.py` L19+L34 does the same; a repo-wide grep finds **no group-aware splitter**
> in project code. A user's windows can therefore span train and test. **The original claim was
> correct: CoUrb's protocol is sample-stratified, weaker than Ch.5's user-disjoint split.** The
> protocol beat is restored in NORTH_STAR §4/§6 and GLOSSARY with this file/line evidence. The
> retraction below is kept for the record of the fail-closed process.

### The original retraction (historical record)

Pass 1 asserted, in lens B and recommendation G-13, that CoUrb's protocol is "sample-stratified, not
user-disjoint (weaker than MobiWac)." **I cannot verify this firsthand.** CoUrb's `results.tex`
reports only "mean and standard deviation over 5 folds"; it does not state whether folds are
user-disjoint or sample-stratified. The dissertation's `../../../../../../../docs/context/DATA_SPLITS.md` documents a
`StratifiedGroupKFold` user-disjoint protocol but does **not** attribute protocols per paper, and
MobiWac (`05_setup.tex` L28) is explicitly "split by user with stratified five-fold cross-validation."

So the protocol *difference* between CoUrb and MobiWac is **[VERIFY]**, not established. Two honest
paths: (a) the author confirms CoUrb's actual split from the CoUrb codebase / `slides/judge_feedback`
and the frame states it; or (b) the frame does not assert a protocol difference. Any recommendation
that leaned on "CoUrb's weaker protocol as an arc strength" (pass-1 G-13) is **suspended** until this
is verified — do not build a narrative beat on an unverified protocol gap. This correction is itself
an example of UW-0 (I graded my own summary's fluency instead of re-verifying the source).

## UW-4 · "Does MTL help?" is answered on two different task pairs — the comparison is not like-for-like (MODERATE)

Following from UW-1 and UW-2: CBIC's "no" is on {category classification, next category}; MobiWac's
"yes" is on {next category, next region}. The dissertation's headline answer to its own research
question therefore rests on a *no* and a *yes* measured on **different task pairs**. This is defensible
(the pair changed for a principled reason — file 02 §3.1), but the frame must not present it as "the
same experiment, opposite result." The honest statement is "naive MTL did not help the original pair;
a check-in representation with redesigned sharing helps the pair that representation induces." **[NEEDS
SIGN-OFF]** on the exact wording of the arc's answer, because it is the single sentence a banca will
quote back.

## UW-5 · The negative-transfer thread is opened, measured, and never explicitly closed (MODERATE)

Pass 1 flagged this (T5 / D-WORTH-1), but under-weighted how much verified evidence exists to close
it. MobiWac measures the shared-trunk gradients as near-orthogonal — a direct answer to CBIC's
negative-transfer worry. **The number carries scope that must travel with it** (MobiWac
`02_related.tex` L89–94): cosine "+0.001 across training, four seeds each on three of our six
datasets, per-dataset means within ±0.003," "measured during development … on an earlier preparation
of the data," and "a finding for this pair of tasks, not a general rule." Two precisions from the
pass-2 critic (MTL expert): cosine measures *directional* conflict only (so say "no directional
conflict"), and near-orthogonality is evidence negative transfer is *absent* but also evidence against
gradient-level *positive* transfer — so it supports "sharing stopped hurting," not "the tasks teach
each other." Closing this explicitly ("the negative transfer CBIC feared is absent once the
representation carries the visit; on the datasets measured, the tasks' gradients are near-orthogonal")
is high-value and grounded, with the scope attached. **Verified firsthand** (MobiWac `02_related.tex`).
The connective sentence is **[NEEDS SIGN-OFF]**.

## UW-6 · The cascade-vs-parallel choice is a defended strength the frame under-sells (MINOR)

The field predicts category/region as *steps toward* a place (the cascade: Ye2013 → CatDM → CSLSL).
MobiWac drops the cascade and predicts the two as co-equal ends, and — verified — **tests that choice
directly** ("since the cascade is the pattern the field uses, we test the choice directly",
`02_related.tex` L80). This is a genuine methodological contribution (it does not just assert the
parallel framing, it validates it), and the arc currently treats it as a design note. Worth a clause
in the Introduction or §2.1. **Verified firsthand.** Low new-claim risk (it restates what the paper
does).

## UW-7 · Istanbul is generalization evidence, not just a second dataset (MINOR)

Pass 1 filed Istanbul as T9 (external validity, closed). Under-weighted: Istanbul is the *only*
non-US, non-Gowalla dataset in the whole arc (CBIC and CoUrb are Gowalla-only US states). It is the
arc's one piece of evidence that the result is not a Gowalla artifact. The frame should let it carry
that weight — "the joint result holds on a different continent, a different data source, and a
different administrative unit (mahalle)" — rather than listing it as dataset six. **Verified**
(MobiWac datasets; NORTH_STAR). Low risk.

---

## Ranked: how much more attention each side needs

| # | Side | Severity | Verified? | Needs |
|---|---|---|---|---|
| UW-1 | task pair evolved; only 2 of 3 changes credited | **major** | firsthand ✓ | the corollary framing (file 02 §3.1) [SIGN-OFF] |
| UW-2 | CoUrb tests representation, not MTL-vs-STL | **major** | firsthand ✓ | precise Ch.4 preface boundary [SIGN-OFF] |
| UW-4 | RQ answered on two different pairs | moderate | firsthand ✓ | exact wording of the arc's answer [SIGN-OFF] |
| UW-5 | negative-transfer opened, measured, not closed | moderate | firsthand ✓ | one closing sentence [SIGN-OFF] |
| UW-3 | pass-1 "CoUrb weaker protocol" unverified | correction | **NOT verified** | author verifies split, or drop the beat [VERIFY] |
| UW-6 | cascade-vs-parallel is a tested choice | minor | firsthand ✓ | one clause; low risk |
| UW-7 | Istanbul = generalization evidence | minor | firsthand ✓ | reframe; low risk |

**The through-line of this audit:** the arc's honesty is not at risk from any single fact — every
fact is defensible — but from *how many things changed at once between the null and the win*.
Representation, sharing, and task pair all moved. The dissertation's credibility depends on the frame
saying so plainly and showing that the task change follows *from* the representation thesis rather
than sitting beside it as an unacknowledged second cause. That is the highest-value honesty move in
the whole storyline, and it is also, handled well, the most intellectually satisfying beat available.
