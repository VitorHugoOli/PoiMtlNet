# Narrative review of the CBIC → CoUrb → MobiWac dissertation arc

> **Scope.** A storytelling/craft review of the unified three-paper arc, run against the intended
> spine (`NORTH_STAR.md` §1–§3, §6), the drafted Fundamentals (`fundamentals/`), and the three
> papers as actually written. Read-only: I propose narrative moves, never apply them and never draft
> replacement chapter prose. No fact, number, citation, or claim is fabricated; every result I name
> traces to the sources fixed in the project instructions (the published CBIC/CoUrb tables + audited
> errata, and the MobiWac whitelist), and any number that enters chapter prose must still be
> re-verified at adaptation against its single source of truth (AGENT_GUARDRAILS §2 N1). New
> connective sentences are marked **[NEEDS SIGN-OFF]**; external claims I did not open this session
> are marked **[VERIFY]**. Where a cleaner story would require a claim past its statistical test, I
> say so and stop.

> **Draft state (governs how to read this).** Ch.2 Fundamentals is drafted (2.1–2.5 `.tex` +
> assembled `fundamentals.tex`); §2.5 "Relevance" is real prose. Ch.1 Introduction and Ch.6
> Conclusion are **spine-only** (NORTH_STAR §6; no draft files exist). The three paper-chapters
> exist only as their standalone published/submitted sources; the mandatory recap subsections and
> time-capsule prefaces are **planned, not written**. This is therefore a **blueprint review**: the
> cohesion findings are about whether the plan welds, and the craft findings bite on real prose only
> where prose exists (Ch.2 and the three papers' own intros/conclusions). Re-run on the compiled v1
> for the craft lenses to bite on the frame.

> **Complement, not duplicate.** I have read personas 17 (excellence), 01 (cold reader), 15
> (readability), 12 (banca). This review does not re-score their dimensions. It does what they do
> not: reconstruct the arc from the artifacts, diff it against the intended spine, run a thread
> ledger, and pressure-test the two load-bearing bridges (why representation dominates; why
> check-in-level specifically). Where I trip over a defect they own, it is one line, out of scope.

---

## A. The reconstructed arc, and the diff against the intended spine (lens 1)

### A.1 The story as it actually reads right now (one paragraph)

Location-based social networks record where people go as check-ins, and a service that anticipates
the next move can prepare ahead. Two coarse questions are enough for that: what type of place comes
next (the next category) and which part of the city (the next region); the exact next place is not
predicted. The natural engineering wish is one model for both, so the dissertation asks whether
multi-task learning helps this task pair and what the answer depends on. The first study (CBIC)
builds the first joint model on a place-level graph embedding with hard parameter sharing, and finds
an honest null: the joint model does not consistently beat two dedicated single-task models, and it
costs more to train. CBIC closes by naming three candidate explanations, one of which is that the
shared representation may not be rich enough. The second study (CoUrb) holds the architecture fixed
and replaces the single place-level input with decomposed spatial, temporal, and categorical
encoders; the category score rises sharply, which is read as evidence that the representation, not
the sharing architecture, is the lever. The third study (MobiWac) builds a representation at the
check-in level (Check2HGI), so each visit carries its own vector rather than each place carrying one
fixed vector, and pairs it with a redesigned joint model (a cross-attention trunk that exchanges
semantic context, plus a private spatial path for region). On that combination one joint model
finally outperforms both dedicated models: the next category on every dataset, and the next region
at four of six, with statistical non-inferiority (TOST, two-point margin) at the other two. The
payoff is stated as a corrected view, not a triumph: a published null result, its diagnosis, and its
resolution.

### A.2 Beat-by-beat map (delivered)

| Beat | Where it lives now | State |
|---|---|---|
| Context funnel (LBSN → anticipate next move → mobility-aware services) | MobiWac §1 p1 (strong); Ch.2 opener (weaker) | Exists in MobiWac's own intro; **not yet owned by a general Introduction** |
| The two tasks kept distinct; next place excluded | MobiWac §1 p2, §3; Ch.2 §2.1; GLOSSARY | Delivered and disciplined |
| The tension (sharing is not free; negative transfer) | CBIC §1; MobiWac §1 p3; Ch.2 §2.3, §2.5 | Delivered |
| Research question, bold inline | NORTH_STAR §1 only | **Spine-only** (Ch.1 undrafted) |
| The journey as the contribution (null → diagnosis → resolution) | NORTH_STAR §2 honest-arc ¶; Ch.2 §2.5 hinge | **Spine-only in the frame**; each paper tells only its own leg |
| CBIC leg: first joint model, honest null, three hypotheses | CBIC §1 + conclusion | Delivered in the source paper |
| CoUrb leg: hold architecture, enrich input, category rises | CoUrb §1 + conclusion | Delivered in the source paper; **cites MTLnet by name** (native bridge) |
| MobiWac leg: check-in representation + redesigned sharing → joint win | MobiWac §1, §2, §4, §6 | Delivered in the source paper |
| The mechanism (same place, different visit, same vector) | MobiWac §2.1; Ch.2 §2.2 map + §2.5 ("weekday lunch vs Saturday night") | Present, but **buried in Ch.5 related work / stated once in §2.5** |
| Objectives 1:1 with chapters | NORTH_STAR §6.1 | Spine-only |
| Recap subsections (Ch.4 recaps MTLnet, Ch.5 recaps both) | Planned (Viegas device) | **Not written** |
| Time-capsule prefaces (venue/status/what-later-revises) | Planned | **Not written** |
| Conclusion answering the question + limitations + future work | NORTH_STAR §6.4 | Spine-only |

### A.3 The diff — where the delivered arc drifts from the intended spine

The intended spine (NORTH_STAR §6) is sound and, where drafted, faithfully executed. The drift is
almost entirely **the drift of an unwritten frame**: the spine promises the connective tissue, but
the connective tissue is exactly the part that does not exist yet. Four specific drifts, in order of
consequence:

1. **The logline compresses a two-factor result into one factor.** The spine's headline is "the
   representation is the dominant factor." That is well-earned by CoUrb (which changed *only* the
   input and saw category rise). But the *resolution* — MobiWac — changed **both** the representation
   *and* the sharing topology (cross-attention two-stream + private spatial path replaced
   hard-sharing + FiLM), and its own text says sharing "helps instead of hurting" once the
   representation changes, with the private spatial path doing real work on region. So the delivered
   evidence is "representation dominates, *and* converting that into a joint win also required
   redesigning how the two tasks share." The spine knows this (it says "a check-in-level
   representation **and** the right sharing topology"), but the one-line logline does not, and an
   undrafted Introduction is where that flattening will happen if it is going to. This is the single
   highest-leverage narrative risk in the arc. (Detailed in F1 and D-MISSING-1.)

2. **CBIC opened three doors; the arc walks through one without saying why.** CBIC's conclusion lists
   three co-equal hypotheses for the null — subtle negative transfer, representation mismatch, and
   architectural restrictiveness — and its own future-work paragraph points *first* at the
   architecture door (soft sharing, Mixture-of-Experts). The dissertation instead walks the
   representation door first (CoUrb). That is a legitimate and, in hindsight, correct choice, but the
   spine's phrase "closes hypothesizing that the shared representation may not be rich enough — the
   thread the rest pulls" quietly promotes one of three hypotheses to *the* thread. Nowhere yet does
   the frame say *why representation before architecture*. That "why" is a missing beat, not a
   falsehood. (Detailed in D-MISSING-2.)

3. **The "cost" thread is opened and never closed as opened.** The intended intro (NORTH_STAR §6.1,
   beats 1–2) wishes for "one model … instead of one dedicated model per task" and frames MTL as
   promising "shared structure and lower cost." CBIC then reports the joint model cost *more*
   (convergence time, MFLOPs). MobiWac's joint model is *larger than the two dedicated models
   combined* (~4.2M vs 1.1M params at Alabama; the paper is scrupulous that the benefit is
   *operational* — one artifact, one forward pass — not arithmetic). So across the arc the "lower
   cost" wish is never delivered as compute savings; it is *redefined* to operational simplicity. The
   spine does not currently narrate that redefinition. If the Introduction promises lower cost and
   the Conclusion delivers "one deployable artifact (that costs more compute)," a banca member will
   read a quietly moved goalpost unless the frame owns the redefinition explicitly. (Detailed in F3
   and D-MISSING-3.)

4. **The mechanism is present but demoted.** The spine (§6.2 Ch.2 beat) wants the reader shown *why*
   a place-level vector is the limit — "the same POI, different visit, same vector." That mechanism
   exists in the corpus (MobiWac §2.1: "two visits to the same coffee shop look identical to the
   model"; Ch.2 §2.5: "cannot tell a weekday lunch from a Saturday night out"). But it currently
   lives in Chapter 5's related-work and in one synthesis sentence in §2.5. For a mechanism that is
   the pivot of the entire dissertation, it is under-placed: the reader should meet it in the
   Introduction, as the reason the journey turns. (Detailed in D-MISSING-4.)

None of these four is a contradiction of the spine; three of the four are *the spine's own nuances
that the one-line version drops*, and the fourth is a placement problem. The finding is that the arc
is intellectually complete and honest, and its risks are all concentrated in the frame chapters that
have not been written — which is exactly where a coletânea's unity is won or lost.

---

## B. The logline, and the per-chapter earn-its-clause verdict (lens 2)

### B.1 The logline

Stated in one sentence, problem → journey → payoff, within the whitelist:

> **A single model that predicts both what kind of place a person will visit next and where should be
> possible, yet a naive joint model on a place-level embedding does not beat two dedicated models
> (CBIC); the reason is the representation, not the sharing architecture (CoUrb); and once each visit
> carries its own vector and the two tasks share through cross-attention rather than a common trunk,
> one model finally outperforms both dedicated models — category everywhere, region at four of six
> datasets and non-inferior at the other two (MobiWac).**

That sentence is honest (verbs bound to their tests, AZ/AL not upgraded, the two-factor resolution
preserved) and it is a genuine problem→journey→payoff. It is long because the honest version *is*
long; a shorter version that keeps only "the representation is the bottleneck" is the tempting
flattening flagged in A.3-1. The recommendation table (G) proposes the frame keep the two-clause
resolution ("a representation built for visits, shared the right way"), not the one-clause one.

### B.2 Does each chapter earn its clause?

- **CBIC earns its clause — as the setup, and it is the arc's structural anchor.** Its clause is "a
  naive joint model does not beat two dedicated models." It delivers exactly that, and — this is the
  quiet strength of the whole dissertation — it delivers it as a *confirmed hypothesis*, not a
  disappointment: CBIC's introduction *predicts* the null ("the central hypothesis of this study is
  that a standard hard parameter-sharing MTL architecture will face significant limitations")
  before it reports it. A predicted null is the strongest possible foundation for a null→resolution
  arc (see §E.4: Lovitts and Mullins & Kiley, firsthand from the internal excellence doc, prize a
  null handled with a diagnosed mechanism and critical self-assessment). The clause is earned. The one risk is that CBIC's
  own framing attributes the null substantially to *task dissimilarity* ("static vs sequential"),
  which is a *different* diagnosis from the one the arc ultimately backs (representation richness).
  The frame must not let CBIC's task-dissimilarity language read as the arc's final word. (F4.)

- **CoUrb earns its clause, and it is the pivot — but it is also the weakest-owned clause.** Its
  clause is "the representation is the lever, not the architecture." It delivers a sharp category
  gain from an input-only change, which is the cleanest single piece of evidence in the whole arc
  for representation-dominance, because it is a true controlled comparison (same architecture, only
  the input changed). Three things weaken how the clause lands, none fatal: (a) CoUrb is
  second-authored (Vitor 2nd author/presenter), so the contribution note is load-bearing — the arc
  leans hardest on the paper the candidate did not lead; (b) **[VERIFY — corrected pass 2]** an
  earlier draft of this review asserted CoUrb used a weaker sample-stratified (not user-disjoint)
  protocol; that is **not** firsthand-verified (CoUrb's text reports only "5 folds"; the per-paper
  split is not stated), so no protocol difference should be claimed until the author confirms it from
  the CoUrb codebase — see `storyline/08_underweighted_sides/` UW-3; (c) CoUrb
  changes *three* things at once (space + time + category encoders) and does not isolate them, so the
  clause it earns is "an enriched, decomposed representation helps," not "here is which axis of
  enrichment mattered." That is fine for the arc, but the frame should state the claim at the
  granularity the evidence supports.

- **MobiWac earns its clause and delivers the payoff — provided the payoff is stated as two-factor.**
  Its clause is "check-in-level representation + redesigned sharing → one model beats both." It
  delivers, and its claim discipline is the strongest in the corpus. The single narrative caution is
  the one in A.3-1: MobiWac's win is the joint effect of a new representation *and* a new sharing
  topology, and its own §2.1/§6 are careful about this. If the frame credits the win to
  representation alone, it *contradicts Chapter 5's own text* — a rare case where overclaiming the
  arc would also be internally inconsistent. Stated as two-factor, the clause is fully earned.

**Verdict:** all three chapters earn their clause. The logline moves forward at every step, with no
dead chapter. The one clause at risk of being *under-delivered* by the frame is CoUrb's (its role as
the controlled pivot is the most likely thing for an undrafted Introduction to under-sell), and the
one clause at risk of being *over-delivered* is MobiWac's (representation-only framing). Both risks
live in the frame, not the papers.

---

## C. The cohesion audit and the thread ledger (lens 3)

### C.1 Seam-by-seam audit

**The general Introduction's arc narrative — does it own the through-line?**
Cannot fully audit (undrafted), but the *plan* (NORTH_STAR §6.1, beat 4, the "honest-arc paragraph")
does own it, and owns it well: it commits to naming the negative result as a finding, the diagnosis
as the turning point, and the final model as the payoff. The risk is not the plan; it is that beat 4
is one paragraph among eight, and the through-line needs to be *the spine of the whole introduction*,
not a single paragraph inside it. The examiner-research calibration (§E.3, firsthand from the
internal excellence doc) is blunt on this: strong publication-based theses carry "linking material
between publications to contextualise and integrate each submission," and stapled compilation with no
thesis-level claim above the papers is the most-cited failure mode. Recommendation G-1 makes the arc
paragraph structural rather than one beat.

**The mandatory bridging subsections (Ch.4 recaps MTLnet, Ch.5 recaps both) — present, carrying?**
**Not written.** This is the largest single cohesion gap, and it is the documented failure mode of
the whole format ("stapled papers"). Right now the CBIC→CoUrb bridge is *native and strong* — CoUrb's
own introduction cites MTLnet by name as the baseline it improves ("MTLNet, proposed in
[silva2025mtlnet] … the question arises of whether decomposing the input …"), so the reader who
arrives at Ch.4 is carried by the paper's own words. The CoUrb→MobiWac bridge is *weaker natively*:
MobiWac §2 cites `silva2025mtlnet` ("our earlier work established this two-task setup and observed
negative transfer") but does **not** mention ST-MTLNet / the CoUrb representation finding at all — so
the "representation is the lever" pivot that MobiWac is supposed to answer is invisible in MobiWac's
own text. Without the planned recap subsection in Ch.5, a reader has no in-text bridge from CoUrb's
diagnosis to MobiWac's resolution. This is the seam most likely to show. (G-2.)

**The time-capsule prefaces (venue/status/what-later-revises) — present, keeping superseded claims
from reading as current?** **Not written.** They are essential here in a way they are not in most
coletâneas, because this arc deliberately contains superseded conclusions: CBIC's "MTL does not
help" and CoUrb's protocol are *meant* to be read as of-their-time and later revised. Without the
prefaces, a banca member reading Ch.3 cold will encounter "MTL does not deliver consistent gains" as
if it were the dissertation's position. The plan (NORTH_STAR §3 time-capsule rule, §6 prefaces) is
correct; it just has to be executed, and it is load-bearing, not decorative. (G-3.)

**The intro–conclusion loop.** Audited as a thread ledger below. The plan closes the loop (the
Conclusion beats in §6.4 map onto the Introduction beats in §6.1), but two threads the corpus opens
are currently at risk of being dropped, and one payoff risks arriving unopened.

### C.2 THREAD LEDGER (opened → closed / opened → dropped / closed → never-opened)

Threads the Introduction/Fundamentals open (per the spine + drafted §2.5) and whether the planned
Conclusion pays them off:

| # | Thread opened | Opened where | Paid off? | State |
|---|---|---|---|---|
| T1 | Does MTL help this task pair, and what does the answer depend on? | Intro §6.1 b3 (RQ); §2.5 clause set | Conclusion §6.4 "consolidated answer" | **opened → closed** (the spine's central loop; sound) |
| T2 | Is the representation the lever, not the architecture? | §2.5 clause 2; Intro arc ¶ | §6.4 representation-dominant answer | **opened → closed** (but see F1: must stay two-factor) |
| T3 | Why a place-level vector is the limit (the mechanism) | §2.2, §2.5 (once) | Not in the §6.4 beats | **opened → at risk of dropped** — mechanism never restated at the payoff (D-MISSING-4) |
| T4 | "One model instead of one per task," framed as lower cost | Intro §6.1 b1–2 | §6.4 "one forward pass, two predictions" | **opened → closed by redefinition** — cost becomes operational, not compute; redefinition currently unnarrated (F3) |
| T5 | Negative transfer as the risk MTL runs | CBIC §1; §2.3; §2.5 | §6.4 mentions region matches/wins | **opened → partially closed** — the arc shows sharing stops hurting, but never explicitly says "the negative transfer CBIC saw is gone, and here is why" (D-WORTH-1) |
| T6 | Why check-in level *specifically* (vs any richer input) | §2.2 spine, §2.5 | Implicit in Ch.5 | **opened → weakly closed** — the jump from "enrich the representation" to "go below the place" is asserted more than motivated (D-MISSING-1's twin, the pivotal jump) |
| T7 | The three CBIC hypotheses (negative transfer / representation / architecture) | CBIC conclusion | Only representation pursued | **opened → two dropped by design** — legitimate, but the frame should say the architecture door was also opened (by CoUrb holding architecture fixed, then MobiWac redesigning it) rather than leave two hypotheses hanging (D-MISSING-2) |
| T8 | Scope: next place is NOT predicted | §1.4; §2.1; MobiWac §3 | §6.2 limitation + §6.3 future work | **opened → closed** (disciplined throughout) |
| T9 | External validity beyond the US (Istanbul) | MobiWac §1, §5; §2.4 | §6.2 "single-city non-US coverage" | **opened → closed** |
| T10 | Stakes: what a mobility-aware service does with the predictions | MobiWac §1 p1, §3, §7 | §6.4 final remarks | **opened → closed in Ch.5, not yet in the frame** — the "why care" is currently strongest inside one paper (D-MISSING-3 / stakes) |

Also the reverse check — **closed → never-opened** (payoffs that arrive without a promise):

- **The region-scaling finding** (region gain grows with region count; California largest) is a real
  and interesting result delivered in MobiWac §1/§6. It is currently *not opened* by the Fundamentals
  or the planned Introduction as a question the dissertation will answer. It risks arriving in Ch.5
  as an unpromised bonus. Worth opening a thread for it in §2.4/§2.5 or the Intro (D-WORTH-2).
- **The gradient-cosine ≈ 0 mechanism test** (why balancers do not help: the two tasks' gradients are
  near-orthogonal, so there is no conflict to resolve) is a genuinely elegant sub-finding in MobiWac
  §2. It closes a thread (T5, negative transfer) that the frame barely opens. Surfacing it in §2.3
  would convert a buried result into a visible answer (D-WORTH-3).

### C.3 Does Ch.2's §2.5 hinge set up exactly the three questions Ch.3/4/5 answer?

**Yes — this is the strongest single piece of connective tissue that exists in drafted prose.** The
§2.5 hinge paragraph is well built: its three clauses map cleanly onto Ch.3 (does naive MTL help?),
Ch.4 (is the representation the lever?), Ch.5 (what does a check-in representation unlock?), and it
is disciplined about verbs (it explicitly binds "outperforms" to paired tests and does not upgrade
AZ/AL). Two refinements would make it load the arc even better, both low-cost:
(1) clause 3 currently front-loads the *result* ("outperforms … everywhere it is tested …"); it could
instead pose the *question* the way clauses 1–2 do and let Ch.5 deliver the result, keeping the hinge
a set of questions rather than a spoiler (D-WORTH-4);
(2) the mechanism sentence ("cannot tell a weekday lunch from a Saturday night out") is the best
single sentence in the drafted frame — it should be echoed in the Introduction, not spent only here
(D-MISSING-4). Net: §2.5 does its job. The gap is above it (the Introduction) and after it (the
Conclusion), not in it.

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
tasks' gradients are near-orthogonal* (cosine ≈ +0.001) is an elegant, honest, self-contained
result. §2.3 currently reviews balancers (GradNorm, PCGrad, Nash-MTL, CAGrad, FAMO, Aligned-MTL) as
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

## E. Craft, pacing, enjoyment (lens 5)

### E.1 The momentum map (where curiosity is created vs satisfied)

Reading the arc as currently constituted (paper intros/conclusions + drafted Ch.2 + spine):

- **Ch.1 (planned):** curiosity *creation* is strong in the raw material — MobiWac's opening
  (anticipate the next move, prepare ahead) is a genuine hook. Risk: if the Introduction opens on
  MTL machinery rather than on the mobility stakes, it creates curiosity about the wrong thing.
- **Ch.2 Fundamentals:** paced well for a thin chapter. §2.2 (the representation spine) and §2.5 (the
  synthesis) are the momentum peaks; §2.3/§2.4 are necessarily catalog-like but kept short. §2.5's
  hinge is the best-crafted transition in the drafted document — it visibly hands the reader to the
  papers. **This chapter does its job of building authority by the end of the literature review**,
  which the examiner research (§E.2) identifies as decisive.
- **Ch.3 CBIC:** this is the emotional *frustration* beat (the honest null), and it works because
  CBIC predicted its own null — the reader feels a hypothesis confirmed, not a failure. Momentum
  risk: without a time-capsule preface the reader may read the null as the dissertation's verdict and
  disengage ("so MTL doesn't work, why am I reading four more chapters?"). The preface is what keeps
  the frustration *productive*.
- **Ch.4 CoUrb:** the *insight* beat — the turn. This is where the reader should feel the arc pivot
  ("it was the representation all along"). The native CBIC→CoUrb bridge helps. The momentum risk is
  that CoUrb's *own* framing is modest (it presents itself as an input-engineering study on three
  states, not as the diagnostic turning point of a thesis); the frame's preface/recap has to
  *elevate* CoUrb's role, or the pivot reads flatter than it is.
- **Ch.5 MobiWac:** the *payoff*. Strongest-written of the three papers, and the "aha" lands in its
  §1 contributions and §6. Momentum risk: MobiWac is dense (six datasets, TOST, region scaling,
  shortlist analysis), and the payoff can get muffled under the machinery. The frame's job is to make
  sure the reader arrives already knowing the one question Ch.5 answers, so the density reads as
  thoroughness rather than as noise.
- **Ch.6 (planned):** the resolution restated at thesis level. If it delivers the §6.4 beats, the
  loop closes. Risk: the true "aha" (a null became a method) must be *stated as such* here, or the
  arc's emotional shape is left implicit.

**The emotional shape (setup → frustration → insight → payoff) is present and, unusually, honest** —
this dissertation has a real dramatic arc that most theses have to manufacture. The risk is not that
the shape is missing; it is that the frame chapters (which carry the shape) are undrafted, so the
shape currently lives only in the reader's ability to infer it across three separately-written
papers. **The single biggest craft win available is to let the frame narrate the emotional arc the
evidence already has.**

### E.2 Where it stalls or sags

- **§2.3 and §2.4** are the natural sag (catalog of balancers, list of metrics). They are correctly
  kept thin; the fix is not to expand them but to thread one forward-pointing sentence each (D-WORTH-3
  and D-WORTH-2) so even the catalog sections pull toward the payoff.
- **The CoUrb→MobiWac seam** is the structural sag: it is the one chapter transition with no strong
  native bridge (C.1). Reader momentum will dip crossing from Ch.4 to Ch.5 unless the recap
  subsection carries them.

### E.3 The one-voice seam verdict

Cannot be fully judged until the frame is drafted and the CoUrb translation exists, but the risk
profile is clear and specific:

- **Terminology is already well-governed** by GLOSSARY + WRITING_LAW, so the *lexical* seam (the
  usual giveaway) is defended: "next category / next region / next place," "check-in," "place
  embedding (HGI)," "the joint model" are enforced repo-wide. This is ahead of most coletâneas.
- **The three papers have visibly different registers**, and this is the real seam risk: CBIC's prose
  is the most conventional ("This dichotomy raises a critical question…"); CoUrb's is an
  input-engineering study translated from Portuguese; MobiWac's is the most disciplined and plain
  (its GLOSSARY is stricter). Read back-to-back, MobiWac will sound like a different, more careful
  author than CBIC. That is partly unavoidable (they *were* written at different times), but the
  frame chapters set the dominant voice, so if Ch.1/2/6 are written in MobiWac's register, the reader
  hears one authorial voice framing three dated artifacts — which is exactly the right effect for a
  time-capsule coletânea. **Recommendation: write the frame in MobiWac's voice** (plainest, most
  disciplined), and let the paper prefaces explicitly mark the older papers as of-their-time, so
  register drift reads as *chronology*, not as inconsistency.
- **The CoUrb translation is the sharpest single-voice risk**: a translated-from-PT chapter dropped
  among English originals is where the seam shows most. The translation-fidelity gate
  (AGENT_GUARDRAILS L5) governs claim drift; the *readability* editor (persona 15) owns the voice
  seam. This review's contribution is only to flag that CoUrb is the chapter to watch.

**Overall craft read:** where prose exists, it is good — MobiWac is genuinely well-written, §2.5 is
well-built, and the dissertation has an authentic dramatic arc that is rare and valuable. The
enjoyment risk is entirely about the *undrafted frame*: a banca member wants to keep reading a
null→diagnosis→resolution story *if the frame tells them that is what they are reading*. Right now the
story is enjoyable to someone who already knows the arc (the author) and would read as three good but
separate papers to someone who does not (a cold banca member) until the frame is written.

---

## F. Honesty under narrative pressure (lens 6)

Every place where a cleaner or more dramatic story would tempt a violation. For each: the temptation,
the truth, and the ruling. **Truth wins in all of them; where a stronger story needs a stronger
claim, I stop.**

**F1 · The one-factor logline (the biggest temptation in the whole arc).**
*Temptation:* "The representation is the bottleneck" is a cleaner, more quotable thesis than "the
representation is the dominant factor, and converting that into a joint win also required redesigning
how the tasks share." The one-factor version is what a punchy Introduction and a punchy Conclusion
both want.
*Truth:* CoUrb isolates the representation effect cleanly (input-only change). But MobiWac's *win* —
the payoff clause — changed both the representation and the sharing topology, and MobiWac's own text
says so (§2.1: sharing "helps instead of hurting" *on the new representation*; §4.2: the private
spatial path is what keeps region competitive). Crediting the joint win to representation alone
**contradicts Chapter 5's own text**.
*Ruling:* the frame may say "the representation is the dominant factor" as the *diagnosis* (CoUrb
earns it), but the *resolution* must be stated as two-factor. Keep the spine's own phrasing ("a
check-in-level representation **and** the right sharing topology"). Do not let the logline drop the
second factor. This is honesty *and* internal consistency — they point the same way.

**F2 · CBIC's null read as current.**
*Temptation:* the arc is more dramatic if "MTL does not help" lands hard in Ch.3. *Truth:* it is a
conclusion "of the time, for that configuration" (place-level embedding, hard sharing), later shown
configuration-specific. *Ruling:* the time-capsule preface is mandatory (WRITING_LAW §3; NORTH_STAR
§3). The drama is legitimate *only* if the preface time-indexes it. Never let a superseded claim read
as the project's position. (Also: CBIC's Nash-MTL "consistently better" predates the solver-bug
discovery — do not amplify it in the frame; NORTH_STAR §4.)

**F3 · The "lower cost" promise vs the larger joint model.**
*Temptation:* MTL's textbook selling point is efficiency; the Introduction wants to promise "one
model, lower cost." *Truth:* CBIC's joint model cost *more* (time, MFLOPs); MobiWac's joint model is
*larger than the two dedicated models combined* (~4.2M vs 1.1M params at Alabama). The honest benefit
is operational (one artifact, one forward pass), which MobiWac §4 states carefully.
*Ruling:* the frame must **not** promise compute savings. If the Introduction raises cost as
motivation, the Conclusion must close it as "operational simplicity, at higher compute," not as
"cheaper." This is thread T4; narrate the redefinition, do not hide it. A banca member will compute
the parameter ratio.

**F4 · CBIC's task-dissimilarity diagnosis vs the arc's representation diagnosis.**
*Temptation:* to make CBIC's null point cleanly at the representation (so CoUrb is its direct answer),
one could soft-pedal CBIC's own stated diagnosis. *Truth:* CBIC attributes the null substantially to
*task dissimilarity* (static vs sequential) and lists representation as one of three hypotheses. The
arc's final position is representation-richness. *Ruling:* do not retrofit CBIC's conclusion. The
honest bridge is "CBIC named three candidate causes; this dissertation tested the representation one
first (CoUrb) and it paid off" — which is true and is also a better story (it shows the research
reasoning). Reframing CBIC's emphasis after the fact would be a silent correction (AGENT_GUARDRAILS
§7); if any CBIC conclusion sentence is adjusted in the re-typeset chapter, it goes in the Appendix B
errata list, not silently.

**F5 · CoUrb's win-count and gain numbers.**
*Temptation:* use the published CoUrb numbers ("16/21", "+20–24 pp") because they are slightly larger
/ rounder. *Truth:* the internal audit recounted **15/21 strict wins + 1 technical tie** and
**+20.2…+22.0 pp**; the deck was corrected, the .tex was not. *Ruling:* the chapter uses the audited
numbers (NORTH_STAR §4; N1). This is a number-integrity flag, not strictly narrative, but it becomes
a narrative flag the moment the frame *summarizes* CoUrb's result — the summary must use the audited
figures. Any CoUrb number promoted into Ch.1/2/6 is **[VERIFY at adaptation]** against
`slides/judge_feedback.md`.

**F6 · The region verbs (the standing MobiWac law).**
*Temptation:* "one model beats both dedicated models" is cleaner than "beats on category everywhere,
beats on region at four of six, matches at the other two." *Truth:* region at AL/AZ is
non-inferiority (TOST, ±2 pp), and AZ is 0.00 — never upgraded. *Ruling:* the whitelist governs
(WRITING_LAW §3; PAPER_PLAN §3). The frame's summary sentence must carry the four-of-six split and
the "matches" verb for AL/AZ. §2.5 already does this correctly — the risk is only that a punchy
Conclusion sentence drops the qualifier. Never "outperforms region everywhere," never "beats,"
never upgrade AZ.

**F7 · The stakes numbers, if promoted to the frame.**
*Temptation:* the "ten regions contain the true region 65.69% of the time, 500× better than random"
line is a fantastic hook for Chapter 1. *Truth:* it is a single-seed, four-dataset motivation sketch
(MobiWac §7), explicitly "motivation, not a measured service result." *Ruling:* it may be used as
motivation, but if promoted to the Introduction it must carry its convention (single seed, the
specific datasets) and the "not a measured service result" hedge, exactly as §7 does. Do not let it
harden into a headline capability. **[VERIFY at adaptation]** + convention required.

**F8 · Fake cohesion from templated bridges.**
*Temptation:* the fastest way to make three papers "read as one" is to bolt identical transition
sentences between chapters ("Building on the previous chapter, we now…"). *Truth:* that is the
documented fake-cohesion failure mode (AGENT_GUARDRAILS §7; WRITING_LAW §4.4). *Ruling:* the recap
subsections and prefaces must carry *real* content (what the prior chapter established, what it
forced), not template scaffolding. Cohesion comes from the argument, not from transition words. Vary
the bridge shapes; the excellence rubric penalizes discourse-skeleton reuse across a 100-page
document.

**No honesty violation is present in the drafted prose** (§2.5 is clean, disciplined, and correctly
hedged). All eight flags are *forward risks* the frame drafting will run into. The arc does not need a
single upgraded claim to be compelling — its honesty *is* its drama.

---

## G. Ranked recommendations (lens 7)

Ranked by narrative leverage per hour. In a coletânea the frame chapters (1, 2, 6) dominate leverage,
and the ranking reflects that: the top six all live in the frame. "Cost" is drafting effort. **[NEEDS
SIGN-OFF]** = a new connective/frame claim requiring author approval (AGENT_GUARDRAILS C2) before it
enters the text; route those through personas 07 + 14. **[VERIFY]** = a number to re-check against
its source of truth at adaptation.

| # | Move | Type | Where (file / beat) | Why it strengthens the story | Cost | Flag |
|---|---|---|---|---|---|---|
| 1 | Make the honest arc the **structural spine** of the Introduction, not one paragraph among eight — open on stakes, state the RQ, then narrate null→diagnosis→resolution as the through-line | REFRAME | Ch.1 (NORTH_STAR §6.1 b1–4) | Fixes the top failure mode (stapled papers); the intro is where a coletânea's unity is won (§E.2, persona 17 dim.2) | subsection | [NEEDS SIGN-OFF] (arc sentences) |
| 2 | Write the **Ch.5 recap subsection** bridging CoUrb→MobiWac ("The MTLnet framework and the representation finding") — the one seam with no native bridge | ADD | Ch.5 related-work head (Viegas device) | Welds the weakest structural seam (C.1, E.2); without it the pivot→payoff jump is invisible in Ch.5's own text | subsection | [NEEDS SIGN-OFF] |
| 3 | Write the **three time-capsule prefaces** (venue/status/what-later-revises), one italic paragraph each | ADD | head of Ch.3, Ch.4, Ch.5 | Keeps CBIC's null and CoUrb's protocol from reading as current (F2); makes superseded conclusions read as chronology, not contradiction | paragraph ×3 | — (status wording only) |
| 4 | Add the **"why check-in level specifically" bridging paragraph** (enrich → but every place-level scheme shares the per-place-vector ceiling → represent the visit) | ADD | Ch.1 arc ¶ and/or Ch.5 recap | Motivates the pivotal jump — the least-motivated joint in the arc (D-MISSING-1) | paragraph | [NEEDS SIGN-OFF] |
| 5 | State the **resolution as two-factor** (representation + sharing topology) everywhere the payoff is summarized | REFRAME | Ch.1 arc ¶, Ch.6 §6.4, logline | Prevents the one-factor flattening that would contradict Ch.5 (F1) — honesty and consistency aligned | sentences | — (enforces existing whitelist) |
| 6 | **Show the mechanism at the top** (same place, two visits, one vector) as the hypothesis the arc will test; echo it at the Conclusion as what the resolution fixed | ADD/MOVE | Ch.1 arc ¶; Ch.6 §6.4; echo of §2.5 line | Promotes the intellectual heart of the arc from Ch.5 related-work to the frame (D-MISSING-4, T3) | paragraph | low risk (frame as hypothesis, not fact) |
| 7 | Add a compact **"what each paper changed / what it forced" bridge table** in the Introduction | ADD | Ch.1 (argument twin of the model-lineage table) | Makes the arc's logic visible at a glance; directly answers banca Q19 "convince me this is one dissertation" (D-WORTH-5) | table + para | [NEEDS SIGN-OFF] (connective claims) |
| 8 | Add the **"why representation before architecture"** reasoning (CBIC's three doors; cheapest/most-controlled first) | ADD | Ch.1 arc ¶ or Ch.4 preface | Closes thread T7; shows research reasoning instead of leaving two hypotheses hanging (D-MISSING-2) | 2–3 sentences | [NEEDS SIGN-OFF] |
| 9 | Lift the **stakes** (mobility-aware service, the shortlist number) into the Introduction context funnel | MOVE | Ch.1 §6.1 b1 (from MobiWac §1/§3/§7) | Gives the reader a felt reason to care through the CBIC null (D-MISSING-3, T10) | paragraph | [VERIFY] number + convention |
| 10 | Thread **one forward-pointing sentence** into §2.3 (balancers → Ch.5 answers) and §2.4 (region scaling → Ch.5 answers) | ADD | Ch.2 §2.3, §2.4 | Converts the two catalog/sag sections into threads; pre-empts "why not balancer X?" (D-WORTH-2/3) | 1 sentence ×2 | low risk |
| 11 | **Name the negative-transfer reversal** out loud at the payoff | ADD | Ch.6 §6.4 | Closes T5 explicitly; satisfying and honest (D-WORTH-1) | sentence | [NEEDS SIGN-OFF] |
| 12 | Rephrase **§2.5 hinge clause 3 as a question**, not a result | REFRAME | Ch.2 §2.5 last ¶ | Keeps the hinge three questions; avoids spoiling the payoff two chapters early (D-WORTH-4, C.3) | sentence | — |
| 13 | Elevate **CoUrb's role** in its preface/recap (the controlled *representation* pivot — architecture fixed, input varied, second-authored) and state its question precisely: it isolates the representation effect, it does **not** revisit MTL-vs-single-task (UW-2) | REFRAME | Ch.4 preface + Ch.4 recap in Ch.5 | Prevents the arc's most under-owned clause from reading flat, and stops a reader concluding "CoUrb showed MTL works" | paragraph | [VERIFY] CoUrb numbers; protocol-difference claim **suspended** (UW-3) |
| 14 | Write the frame in **MobiWac's register**; mark older papers as of-their-time | REFRAME | Ch.1/2/6 voice | Makes register drift read as chronology, not inconsistency (E.3 one-voice seam) | style choice | — (persona 15 owns) |

**Leverage note:** moves 1–3 are the unity backbone — if only three things are done, do these. Moves
4–6 are the intellectual-honesty backbone (they make the arc land *correctly*). Moves 7–14 are
enrichment and craft. The model-lineage table already exists and is good; move 7 is its argument-level
complement.

---

## H. PROTECT LIST — what already works; do not dilute

1. **The honest arc itself.** Null → diagnosis → resolution is a real, rare dramatic structure. Do not
   sand it into a conventional "we propose X and it works" story to look tidier. The null is the
   foundation, not an embarrassment (WRITING_LAW §3; persona 17 dim.7 calls it the dissertation's
   natural superpower).
2. **CBIC as a predicted null.** CBIC's introduction *hypothesizes* the limitation before reporting
   it. This is what makes the null land as a finding. Preserve that framing when re-typesetting; do
   not rewrite CBIC's intro to sound surprised by its own result.
3. **The §2.5 Relevance hinge.** The best-built connective prose in the drafted document. Its
   three-clause structure and its verb discipline (bound to tests, AZ not upgraded) are exactly right.
   Refine clause 3 (move 12) but do not restructure it.
4. **The native CBIC→CoUrb bridge.** CoUrb's own introduction cites MTLnet by name as the baseline it
   improves. This is free, real cohesion — keep it visible; the recap subsection should *complement*
   it, not replace it.
5. **The claim discipline in MobiWac and §2.5.** The four-of-six / matches-at-two / never-upgrade-AZ
   wording, the operational-not-arithmetic cost framing, the "motivation not a measured service"
   hedges. This is the honesty that makes the arc defensible. Every recommendation above is
   constrained to preserve it.
6. **The scope discipline** (next place is not predicted, stated once early and held). Do not blur the
   three tasks for a cleaner sentence.
7. **The model-lineage table.** DGI → HGI → MTLnet → ST-MTLNet → Check2HGI → joint model, with verbs
   disciplined. Keep it; move 7 adds an argument-level twin, it does not replace this.
8. **The mechanism sentence** ("cannot tell a weekday lunch from a Saturday night out", §2.5). The
   single most vivid line in the drafted frame. Reuse it in the Introduction (move 6) — do not
   rewrite it into something blander.

---

## I. Closing — the three questions, answered directly

**Is it cohesive?** The *argument* is cohesive — genuinely one investigation, not three topics: a
single research question, a settled chronological-is-intellectual order, one native inter-paper bridge
already in the text, and a Fundamentals chapter whose §2.5 hinge sets up exactly the three questions
the papers answer. But the *document* is not yet cohesive, because the connective tissue that carries
the cohesion — the Introduction's arc narrative, the two recap subsections, the three time-capsule
prefaces — is planned and unwritten. Cohesion here is a drafting task, not a rethinking task: the
plan welds; it just has not been welded yet. The one place the plan itself risks incoherence is the
logline's temptation to credit the payoff to representation alone, which would contradict Chapter 5's
own two-factor account — keep the resolution two-factor and the argument stays sound.

**Is it enjoyable?** For a reader who already knows the arc, yes — it has an authentic
setup→frustration→insight→payoff shape most theses have to fake, and where prose exists (MobiWac,
§2.5) it is good. For a cold banca member, not yet, because the frame that would tell them "you are
reading a null-that-became-a-method" is the undrafted part; today they would meet three well-written
but separately-voiced papers and have to infer the drama themselves. The enjoyment is latent in the
evidence and will be released the moment the frame narrates it.

**Is it well-crafted?** The drafted prose is well-crafted and disciplined; the governance
(GLOSSARY, WRITING_LAW, the claim whitelist) is unusually strong and already defends the lexical seam
that usually betrays a compilation thesis. The craft risk is entirely in the unwritten frame and in
one translated chapter (CoUrb), both known and gated.

**The single highest-leverage narrative move:** write the Introduction so the honest arc is its
structural spine (recommendation 1) — open on the mobility stakes, pose the research question, then
narrate null → diagnosis → resolution as the through-line, stating the resolution as two-factor
(a representation built for visits, *and* a sharing topology that lets the tasks help each other) and
showing the mechanism (same place, different visit, same vector) as the hypothesis the journey tests.
Everything else in this review is in service of that one move: the recaps and prefaces protect it, the
bridge table makes it visible, and the honesty flags keep it true. The arc is already there in the
evidence; the Introduction is where the reader is told what they are about to read.

---

### Source ledger (what each result-claim in this review traces to)

All result-claims trace to sources fixed in the project instructions; no number is computed here, and
any number promoted into chapter prose must re-verify at adaptation (N1). Key traces:

- CBIC null, three hypotheses, task-dissimilarity emphasis, cost-more, Nash-MTL caveat →
  `articles/CBIC___MTL/sections/intro.tex`, `conclusion.tex`; NORTH_STAR §2, §4.
- CoUrb input-only change (only baseline is MTLNet — no single-task comparison, UW-2), category gain,
  16/21 vs audited 15+1 & +20.2–22.0 pp; protocol split **not** firsthand-verified (UW-3) — earlier "sample-stratified"
  split, second authorship → `articles/CoUrb_2026/src_en/sections/intro.tex`, `conclusion.tex`;
  NORTH_STAR §2, §4; audited numbers per `slides/judge_feedback.md` (not re-opened here — **[VERIFY at
  adaptation]**).
- MobiWac two-factor method (Check2HGI + cross-attention + private spatial path), region 4/6 + TOST
  ±2pp, AZ 0.00 never upgraded, gradient cosine ≈ +0.001, larger-than-combined params (4.2M vs 1.1M),
  shortlist 65.69% single-seed motivation, no ST-MTLNet mention in §2 →
  `articles/[mobiwac]/src/sections/01_introduction.tex`, `02_related.tex`, `03_problem.tex`,
  `04_method.tex`, `07_discussion.tex`, `08_conclusion.tex`.
- §2.5 hinge, mechanism sentence, model-lineage table, draft state →
  `../articles/dissertacao/science/fundamentals/2.5_relevance/2.5_relevance.tex`, `model_lineage_table.md`,
  `fundamentals.tex`; intended spine → `NORTH_STAR.md` §1–§3, §6.
- Honesty bounds → `WRITING_LAW.md` §3, §5; `AGENT_GUARDRAILS.md` §1–§3, §7; `GLOSSARY.md`.
- Excellence/coletânea calibration → `docs/research/dissertation_excellence_2026-07-20.md` (opened
  this session, firsthand) + `exemples/viegas/VIEGAS_ANALYSIS.md`; external searches below.

### External calibration note (§E references) — provenance corrected

**Provenance honesty (fail-closed).** Two web searches were run this session
("framing a negative result as a contribution"; "compilation-thesis coherence / stapled-thesis").
Both returned **titles and URLs only; no page body was retrieved and no page was opened**. They
therefore provide **no firsthand external grounding**, and I do not cite them as support. Every
load-bearing external claim in §E below is re-anchored on the **internal excellence doc**
(`docs/research/dissertation_excellence_2026-07-20.md`), which *was* opened firsthand this session and
already contains the relevant examiner-research findings with their identifiers. Convention claims
that appear in neither the internal doc nor an opened page are marked **[VERIFY]** (general domain
knowledge, to confirm against a source before any of it enters chapter prose).

- **E.3 (compilation-thesis unity) — firsthand-grounded, internal doc.** The internal excellence doc
  records (firsthand from the examiner-research literature): stapled compilation is "the most cited
  PBT failure mode" (ANTI_PATTERNS #1: "papers bound without linking material … no thesis-level claim
  above the papers"); and strong publication-based theses have "a substantial introduction with
  literature review, **linking material between publications to contextualise and integrate each
  submission**, and a concluding synthesis" (Sharmini & Kumar 2018). This is what ranks the recap
  subsections and the arc-spine Introduction at the top (moves 1–2); it needs no web source. The
  university-guideline phrasings I drew from the search titles ("narrative overview"; Toronto/ANU
  compilation guidelines) are **[VERIFY]** — I did not open those pages.
- **E.4 (negative result as contribution) — firsthand-grounded, internal doc.** The internal doc
  records (firsthand): Lovitts — otherwise-good dissertations remain acceptable "when experiment(s) do
  not work out and students get null or negative results," and what distinguishes outstanding is *what
  the author does with them*; Mullins & Kiley — examiners prize "how they recognise and deal with
  contradictions" and "critical assessment of their own work"; and the ML-specific form (CS_SIGNALS):
  "a falsified hypothesis with a diagnosed mechanism … is excellence evidence; an unexplained dip is
  not." This fully supports protecting CBIC's predicted-null framing (protect-list 2) and
  time-indexing rather than apologizing for it (move 3) — CBIC predicts its null, CoUrb diagnoses,
  MobiWac resolves. The narrower writing-craft convention "prime the reader in the introduction / frame
  the null as challenging an expectation" is **[VERIFY]** (general knowledge; not in the internal doc
  verbatim and not opened this session) — the recommendation stands without it on the Lovitts/M&K
  grounding above.

