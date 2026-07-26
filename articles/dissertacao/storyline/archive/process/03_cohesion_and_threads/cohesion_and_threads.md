# Cohesion audit and thread ledger (lens 3)

> Part of the storyline review. Extracted verbatim from the consolidated `STORY_REVIEW.md`
> (artifact of this project), unchanged except for the pass-2 note below. All result-claims trace to
> sources fixed in the project instructions; new frame claims are [NEEDS SIGN-OFF]; unverified
> external claims are [VERIFY].
>
> **Pass-2 revision.** Thread T8 ("next place not predicted") was under-graded here as a clean closed
> thread; it is re-opened as a *justification* question in `02_task_choice_endorsement/`. A new thread
> (the task pair itself evolved) is added in `08_underweighted_sides/` UW-1.
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
