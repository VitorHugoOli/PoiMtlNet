# Reconstructed arc, diff, and logline (lenses 1–2)

> Part of the storyline review. Extracted verbatim from the consolidated `STORY_REVIEW.md`
> (artifact of this project), unchanged except for the pass-2 note below. All result-claims trace to
> sources fixed in the project instructions; new frame claims are [NEEDS SIGN-OFF]; unverified
> external claims are [VERIFY].
>
> **Pass-2 revision (see `08_underweighted_sides/` UW-2, UW-4).** The logline's middle clause is
> sharpened: CoUrb does **not** show "MTL works" — its only baseline is MTLNet (both multi-task), so
> it isolates the *representation* effect and is silent on MTL-vs-single-task. And the research
> question is answered on **two different task pairs** (CBIC's static+sequential vs MobiWac's
> two-sequential). Read this section together with `02_task_choice_endorsement/`.
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
  leans hardest on the paper the candidate did not lead; (b) CoUrb's protocol is sample-stratified,
  *not* user-disjoint (weaker than MobiWac), which the frame must flag as a limitation of the
  evidence, not hide (the spine already commits to this, and it *strengthens* the arc — the honest
  read is "even under a weaker protocol the representation effect was already visible"); (c) CoUrb
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
