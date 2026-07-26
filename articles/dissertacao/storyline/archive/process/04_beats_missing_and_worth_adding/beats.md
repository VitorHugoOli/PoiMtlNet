# Missing and worth-adding beats (lens 4)

> Part of the storyline review. Extracted verbatim from the consolidated `STORY_REVIEW.md`
> (artifact of this project), unchanged except for the pass-2 note below. All result-claims trace to
> sources fixed in the project instructions; new frame claims are [NEEDS SIGN-OFF]; unverified
> external claims are [VERIFY].

---

## D. Missing or worth-adding beats (lens 4)

Split into MISSING (the story breaks, or reads as stapled, without it) and WORTH-ADDING (enrichment).
Each names the load-bearing bridge it repairs.

### MISSING — the story is materially weaker without these

**D-MISSING-1 · The pivotal jump: "why check-in level, specifically."**
This is the single most important missing beat. The arc's logic is: CoUrb says "enrich the
representation" → MobiWac answers "go *below the place*, to the check-in." But "enrich" admits many
answers (more encoders, better graphs, sequence models, larger embeddings). Nothing in the frame yet
argues *why the specific move is downward in granularity* rather than sideways in richness. The
mechanism that justifies it exists (a place has different functions on different visits, so no
per-place vector can be right for both), but it is currently used to justify the *representation*
in the abstract, not the *specific descent to the check-in level*. Without one explicit bridging
argument — "enriching the input helped (CoUrb); but every place-level scheme, however enriched,
still assigns one vector per place; the only way past that ceiling is to represent the visit itself"
— the reader experiences Check2HGI as a rabbit from a hat. **The story breaks here** in the sense
that the arc's turn from diagnosis to resolution is the least motivated of all its joints.
*Fix:* one bridging paragraph, in the Introduction arc and/or the Ch.5 recap subsection. **[NEEDS
SIGN-OFF]** — the sentence "every place-level representation, however enriched, shares the
per-place-vector ceiling" is a new connective claim; it is strongly supported by MobiWac §2.1 but is
not verbatim in any source.

**D-MISSING-2 · Why representation before architecture (the CBIC three-door choice).**
CBIC ends with three hypotheses and points first at the *architecture* door. The dissertation walks
the *representation* door. A reader who read CBIC carefully will ask why. The honest answer is
available and good: CoUrb tested the representation door by holding the architecture fixed and saw a
large effect, which is stronger evidence for representation than anything the architecture door
produced; and MobiWac ultimately walked the architecture door too (it redesigned sharing). So the
frame can say "we tested the cheapest, most-controlled hypothesis first (change only the input), it
paid off, and the final model returned to the architecture question once the representation was
right." Without this, thread T7 leaves two named hypotheses visibly hanging. *Fix:* two or three
sentences in the Introduction arc paragraph or the Ch.4 preface. **[NEEDS SIGN-OFF]** (a new
connective claim about *why* the order of investigation).

**D-MISSING-3 · The stakes, early and concrete.**
The reader is given a concrete reason to care — but only inside MobiWac (§1 p1: caching content where
a user is heading, provisioning capacity before demand arrives; §7: the California shortlist of ten
regions out of 8,501 contains the true region 65.69% of the time). In a ~100-page document, that
stakes-setting has to happen in Chapter 1, or the reader spends the CBIC null with no felt reason to
care whether MTL works. The material exists and is already quantified and sourced; it just has to be
lifted into the frame's opening. *Fix:* fold the MobiWac §1-p1 / §3 motivation into the Introduction
context funnel. Low new-claim risk (it is reused, sourced material), but the §7 shortlist number, if
promoted to Chapter 1, must carry its convention (single-seed, four datasets) and re-verify against
the MobiWac source of truth. Mark the specific number **[VERIFY at adaptation]**.

**D-MISSING-4 · The mechanism, shown at the top, not just in Ch.5.**
Covered in A.3-4 and T3. The "same coffee shop, two visits, identical vector" mechanism is the
intellectual heart of the arc and currently lives in Chapter 5's related work plus one §2.5 sentence.
It should be shown once, concretely, in the Introduction (as the reason the journey turns) and
echoed at the Conclusion (as the thing the resolution fixed). This is placement, not new content, so
new-claim risk is low — but promoting it to the Introduction means stating it *before* Ch.5 proves
it, so it must be framed as the hypothesis the dissertation will test, not as an established fact, to
stay honest to the time-capsule rule.

### WORTH-ADDING — enrichment, the arc survives without them

**D-WORTH-1 · Name the negative-transfer reversal explicitly.**
CBIC observes negative transfer (sharing hurts one task). MobiWac shows sharing helping. The arc
implies the reversal but never says, in one sentence at the payoff, "the negative transfer the first
study saw is absent once the representation carries the visit, and here is the evidence." Closing
T5 out loud is satisfying and honest. Low cost; **[NEEDS SIGN-OFF]** as a connective claim.

**D-WORTH-2 · Open the region-scaling thread before Ch.5 delivers it.**
The "region gain grows with region count" finding (closed → never-opened, C.2) is a strong result
arriving unpromised. One clause in §2.4 or the Introduction that flags "whether the joint benefit
depends on how finely the map is partitioned" turns a bonus into an answered question.

**D-WORTH-3 · Surface the gradient-orthogonality mechanism in §2.3.**
MobiWac's finding that loss/gradient balancers do not beat a tuned fixed weighting *because the two
tasks' gradients show no directional conflict* is an honest, self-contained result. (The number and
its scope: cosine "+0.001, four seeds each on three of six datasets, measured during development on an
earlier data preparation, a finding for this pair, not a general rule" — MobiWac `02_related.tex`
L89–94; the scope must travel with the number wherever it is used, per the pass-2 MTL-expert critic.)
§2.3 currently reviews balancers (GradNorm, PCGrad, Nash-MTL, CAGrad, FAMO, Aligned-MTL) as
background. One forward-pointing sentence — "whether these balancers help this task pair is an
empirical question Chapter 5 answers" — converts a catalog into a thread. It also pre-empts the
obvious banca question "why didn't you use [balancer X]?".

**D-WORTH-4 · Make the §2.5 hinge pose clause 3 as a question, not a result.**
Covered in C.3. Keeps the hinge a set of three questions and avoids spoiling the payoff two chapters
early. Pure craft; no claim change.

**D-WORTH-5 · A one-row-per-paper "what changed / what it forced" bridge table in the Introduction.**
The excellence rubric (persona 17, dim. 2) and the compilation-thesis literature (§E.3) both prize
explicit "Chapter N showed X, which forced Y" connective tissue. A compact table — paper → what it
changed → what result → what question it forced next — in the Introduction or at the head of Ch.3
would make the arc's logic visible at a glance and directly answers the banca's "convince me this is
one dissertation, not stapled papers" (Q19). The model-lineage table already exists for the *models*;
this is its argument-level twin. Medium cost; high unity leverage.

---
