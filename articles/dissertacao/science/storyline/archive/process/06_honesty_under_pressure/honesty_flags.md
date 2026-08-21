# Honesty under narrative pressure (lens 6)

> Part of the storyline review. Extracted verbatim from the consolidated `STORY_REVIEW.md`
> (artifact of this project), unchanged except for the pass-2 note below. All result-claims trace to
> sources fixed in the project instructions; new frame claims are [NEEDS SIGN-OFF]; unverified
> external claims are [VERIFY].
>
> **Pass-2 addition.** A ninth honesty flag belongs here: the research-question answer rests on two
> different task pairs (`08_underweighted_sides/` UW-4), and the frame must not present it as "same
> experiment, opposite result." See file 02 §3 for the honest resolution.
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
