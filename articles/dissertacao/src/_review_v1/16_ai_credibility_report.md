# 16 · AI-credibility reviewer report — the external-perception simulation

> Persona: `reviewers/16_ai_credibility.md`. Runs AFTER persona 03 (style gate). Two readers in
> one report: (1) the **screener** (a 2026-grade detector pass, Pangram-class, with the
> hybrid-text windowing caveat) and (2) the **suspicious expert** (a well-read CS examiner
> keying on gestalt). Evidence base: `docs/research/ai_detection_landscape_2026-07-20.md` +
> `docs/research/ai_writing_evidence_2026-07-18.md`, refreshed this session (§7). Read-only.
> Does NOT re-run 03's counted sweeps (banned words, -ly density) — references 03's report.
> Scope: full defense build (`src/main_defense.pdf`, 87 pp) + sources (`src/chapters/*.tex`,
> `src/0_main.tex`). Snapshot 2026-07-23.
>
> **Mission framing (from the persona header):** AI use here is disclosed and legitimate. The
> job is not camouflage; it is that the text earns full credibility anyway. "There is no problem
> in using AI, but the text needs to be great." Nothing below recommends evading detection of
> disclosed use.

---

## VERDICT (per channel)

**SCREENER RISK: MEDIUM** (windowing caveat stated). The frame chapters (1, 2, 6, appendices,
front matter) are disclosed *substantive* AI drafting from author-approved outlines, which is the
mode NeurIPS 2026 verified a Pangram-class detector *does* flag (as opposed to light copy-editing,
which it does not); so a hybrid-document scan would likely place frame stretches in the elevated
range. That number would not be a truth: on hybrid text, detector scores are window-size artifacts
(NeurIPS measured 42.7% of position papers "high-AI" at 250–350-word windows, 12.7% at ~100-word
windows), so any single score is unstable by measurement, not by opinion. Two facts hold the risk
at MEDIUM rather than HIGH: the prose is lexically rich and syntactically varied, so it does *not*
additionally trip the L2-simplicity false-positive channel that flags Brazilian-authored English
(Liang 61.3%→11.6% when vocabulary was enriched); and the provenance shield (§5) is strong enough
to convert any flag from an integrity question into a documented-process one. The three re-typeset
article chapters are human-published text — a flag there is a pure false positive on peer-reviewed
prose.

**EXPERT-SUSPICION RISK: LOW.** The strongest human tell is *absence* of concrete, first-person
research detail, and this document is saturated with the opposite: the capacity-matched baseline
triple (56.16 / 56.82 / 64.54 macro-F1), the 4.2M-vs-0.6M parameter budget, the +0.001 gradient
cosine over four seeds, the partial fifteen-of-twenty California run, the freeze control at three
named datasets, and a task-pair confound the author admits against his own result — material a
generator never produces because it does not run the experiments. Burstiness is high (03: frame CV
49–57%; Abstract CV 41%, sentences 19–83 words), section openers vary, copulas are plain ("is,"
not "serves as"), and every verdict verb is bound to a named test. The residual gestalt tells are
localized and context-exempt (two conventional bold-label lists; scoped negative parallelism), not
a pervasive machine evenness.

---

## TOP 3 FINDINGS

1. **[MEDIUM · credibility shield · front matter] No up-front disclosure line — the AI-use
   statement lives only in Appendix C (p. 87, the last page).** The 2026 evidence (arXiv 2601.09620,
   "Full Disclosure, Less Trust?") establishes the layered *detail-on-demand* pattern as the design
   that minimizes the disclosure trust penalty: a one-line statement up front (near-costless to
   trust) plus the full appendix behind it (~2/3 of readers want the detail, and it is there for
   them). The dissertation has the excellent appendix but not the one-liner, so a reader meets the
   disclosure only if they reach the final appendix. Direction (additive, not applied): add a single
   front-matter disclosure sentence (folha-de-rosto footnote, preface, or a pointer where the
   Resumo/Abstract sits) that names the tool and points to Appendix C. This also serves CAPES /
   CNPq Portaria 2.664/2026 visibility. Placement mechanics cross-ref persona 13 (compliance).

2. **[MEDIUM · over-correction guard · Ch3/Ch4] Do not sterilize the re-typeset published chapters
   when acting on 03's -ly-density finding.** Persona 03 correctly reports Ch3 at 1.83% and Ch4 at
   1.24% -ly density, over the ≈0.8% band. But those two chapters are *human-published, peer-reviewed*
   text (CBIC 2025, CoUrb 2026). Aggressively scrubbing manner adverbs out of genuinely-human prose
   to hit a band does two kinds of harm this persona is charged to prevent: it risks altering
   published wording (an errata-integrity issue), and blanket tell-scrubbing is the documented
   "defensive writing" failure mode that reads as its *own* red flag (Wikipedia WP:AISIGNS; the
   L2-harm literature). Direction: trim only the decorative manner adverbs 03 named
   (effectively/frequently/consistently/largely), keep the functional ones, and stop well short of
   flattening — a few decorative -ly adverbs in real human published text are not a credibility risk.

3. **[LOW–MEDIUM · human channel · document-wide watch] Negative parallelism ("X, not Y" /
   "rather than") is the single tell a 2026 CS examiner is most primed to see — currently in spec,
   keep it from creeping.** 03 counts 27 "X, not Y" + 28 "rather than", concentrated in Ch5 (21, its
   own audited spec) and used ~1× per frame chapter as a scoped honesty device ("the input
   representation, not the sharing architecture"). That density is defensible today. The finding is
   not a fix; it is a guard: this construction is #5 on the current human-tell catalog and the target
   of new public de-slop tooling (Peter Yang's `/no-ai-slop`, July 2026), so any future AI-assisted
   edit wave that adds more will cross from "load-bearing emphasis" into "tic." Freeze the count; do
   not let edit passes raise it.

---

## 1 · GESTALT PASS (human channel)

Read as a suspicious, LLM-fluent CS examiner (Russell et al. ACL 2025: this population detects at
~92%, keying on formality / originality / clarity — "too clean, too even" — plus lexical tells).
Chapter openings, section transitions, and one full results discussion per frame chapter (Ch6 §6.2,
the consolidated answer) were read on the built PDF for rhythm and on source for quoting.

**What an expert would NOT flag (the gestalt is human):**
- **No frictionless evenness.** The arc has real friction: a published *null* result carried as the
  foundation, a diagnosis that overturns it, an admitted confound (§6.3.6) that weakens the author's
  own attribution. Generators smooth toward success; this text argues against itself and wins anyway.
- **Openers vary** (03 confirmed Ch2's five sections open five different ways; I re-confirmed on the
  PDF). No outline-shaped "This section discusses…" skeleton repeating across sections.
- **Copulas are plain.** "The starting point is the one-hot identifier"; "A place embedding … shares
  one property." No systematic copula-avoidance ("serves as / functions as / boasts"), the #6 tell.
- **Closers do not restate.** §2.1 closes on a forward hook, §2.5 on "these three questions in turn,"
  Ch6 sections on distinct concrete statements. No appended wrap-up sentences (the #4 tell).

**The residual gestalt tells (localized, context-exempt — logged, not alarming):**
- **Two bold-header-colon description lists**: §1.6 Contributions (Theoretical / Software / Empirical
  / Practical, p. 16) and Appendix C's scope enumeration (Drafting / Editing and review / Formatting
  / Code, p. 87). The bold-label-colon vertical list is tell #3 ("bullet-itis where prose belongs").
  Here the tell fires on *form* but the *context exempts it*: a contributions taxonomy is a sanctioned
  dissertation convention (WRITING_LAW §5, the Viegas pattern), and a disclosure enumeration is
  clarity-first by design. An examiner reading a dissertation expects a contributions taxonomy. The
  content inside each label is concrete and specific, not filler. Net: these are the two spots the eye
  *pattern-matches* to an AI shape, but the residual suspicion is low. Noted so the author knows where
  the eye catches; no change required unless he wants to lower the visual signal (running prose would).
- **One business idiom**: "move the needle" — Ch6 §6.1 (L40), "A change of input, with no change of
  architecture, moved the needle farther than any architectural variation tried before it." A single
  instance, a motion/business-metaphor idiom that reads slightly off in a defense register. It is
  03/idiom-law scope (the counted idiom sweep), not this persona's counted call — handoff below.
  (Verified: "needle" occurs once in the frame chapters, Ch6 only; not in Ch1.)

## 2 · SPECIFICITY AUDIT (the highest-yield check)

The persona's core thesis: the strongest human signal is the presence of concrete research detail and
reflective, first-person methodological voice that a generator cannot fabricate because it did not run
the work. **This audit PASSES, and it is the dissertation's single largest credibility asset.** Every
frame section that *should* carry lived detail does.

| Frame location | Lived-research detail present (verified in source/PDF) | Verdict |
|---|---|---|
| Ch1 §1.2 arc | The three candidate explanations of the CBIC null, named and carried forward as the arc's engine; the honest "the promise, however, is not automatic." | Grounded |
| Ch2 §2.5 relevance | The three-gap → Ch3/4/5 mapping (a synthesis only the author of the arc can write), not a literature restatement. | Grounded |
| Ch6 §6.2 answer | Capacity-baseline triple 56.16 / 56.82 / 64.54; 4.2M vs 0.6M params at Alabama; freeze control at AL/AZ/FL; +0.001 gradient cosine over 4 seeds; partial 15-of-20 California run "at the time of writing." | **Exemplary** |
| Ch6 §6.3.6 limitation | The task-pair confound stated *against* the author's own result — "no single controlled ablation separates the representation-and-topology change from the task-pair change." | **Exemplary** |
| Ch6 §6.5 remarks | "The negative result was not an obstacle … worked through, it was the contribution's first half." | Reflective voice |

**No frame section reads as generic filler that should have carried detail but did not.** The
literature-background sections (§2.1–§2.3) are appropriately less first-person, but they take
positions rather than cataloging ("This lineage is background for the present work, not its target";
"a balancer earns its place only by outperforming it") — 17 corroborates. The specificity is placed
exactly where it defeats the human tell: in the arc (Ch1), the synthesis (Ch2.5), and the results
discussion (Ch6).

*One small additive opportunity (optional, not a gap):* §1.6's Practical/Empirical items describe the
validation abstractly; the concrete headline (5.3–9.4 macro-F1) lives in the Abstract and Ch6 but not
in the contributions list. Injecting it there would add one more anchor of specificity to the most-read
page. Low priority — the number is already carried nearby.

## 3 · RHYTHM / VARIANCE PASS (residual after 03)

Variance compression is the deepest measured tell (Claude-family revision reduced variance in ~78% of
stylometric features). 03 already reported per-chapter sentence-length CV; I do not re-run that sweep. I
add the one surface 03 did not measure — **the Abstract**, the single most externally-scrutinized
paragraph in the document:

- Abstract (EN, `0_main.tex` L240–271): **9 sentences, word counts [19, 53, 45, 27, 42, 44, 41, 83, 29],
  mean 42.6, CV 41%.** The 83-word results sentence (packed with the six-dataset numbers) sits between a
  19-word opener and a 29-word thesis close. This is healthy burstiness, not the flat ~20-word uniformity
  a detector reads as low-perplexity. The high mean (42.6) reflects the numeric-dense results sentence,
  within academic-abstract norms; the *variance* is what protects it, and the variance is good.
- Frame chapters (03's figures, cited not re-derived): Ch1 CV 49%, Ch2 54%, Ch6 57% — all high. The
  frame prose is the *opposite* of variance-compressed. An edit pass that only smooths would regress this;
  it has not happened yet (03's read-aloud found the author's voice intact — concessive clauses, mid-
  paragraph result openers, varied length).
- The one compressed chapter is Ch3 (CV 43%), the re-typeset CBIC paper — human-published text. See the
  over-correction guard (Top Finding 2): its compression is a property of the *original human* prose, and
  is not a credibility problem.

## 4 · DETECTOR SIMULATION (screener channel)

**No local detector was run, by design.** A Pangram-class screener is proprietary/API-gated and not
available in this environment; the only open-weights option (RoBERTa-family) is exactly the tool the
evidence shows misclassifies 30–69% of *human* text as AI (Booth audit) and would produce noise, not a
committee-relevant signal. Running it would violate the persona's own rule that a score is never a
verdict. The screener channel is therefore a **qualitative estimate**, stated as such.

Estimate, by the two-channel logic:
- **L2-simplicity false-positive channel: LOW.** The detector-bias literature (Liang; Pindrop/Authors
  Guild ACL 2026) flags *structurally simple, low-lexical-richness* L2 prose. This dissertation's prose is
  the opposite — rich technical vocabulary, varied and complex syntax, high burstiness. The single
  intervention proven to drop L2 false-positives (enrich the vocabulary: 61.3%→11.6%) is already the
  document's baseline state. A Brazilian-author FPR spike is unlikely here.
- **Substantive-generation channel: MEDIUM.** The frame chapters *are* disclosed generation, the mode a
  strong detector flags. A hybrid-document scan would probably light up frame stretches. But (a) the score
  is a window-size artifact, not a measurement, and must be reported to any committee with that caveat; and
  (b) the shield (§5) makes the flag a process question, not an integrity one. The re-typeset chapters are
  human text where a flag is a false positive.

**Reporting rule for the author:** if anyone ever produces a detector score on this document, it is — on
hybrid text, by NeurIPS's own calibration — unstable by measurement. The correct response is never to
argue the number; it is to present the provenance (§5). That is the officially-recognized corroboration
path (NeurIPS 2026 appeal protocol), and this author has the evidence to walk it.

## 5 · PROVENANCE-SHIELD STATUS TABLE (process, not prose — the real defense)

| Shield element | Status | Evidence |
|---|---|---|
| Git AI/author commit discipline (GUARDRAILS §5) | **PRESENT (strong)** | `git log` shows clean `draft(ai):` vs `edit(author):` labels across the assembly (phases 0b–6) and the CoUrb work; commit `b642d1ce` ("audit of the Opus readability pass") corroborates the appendix's Opus claim. Disclosure is reconstructible from history, not remembered. |
| Layered disclosure (short front + full appendix) | **PARTIAL** | Appendix C (p. 87) is present and well-drafted; the up-front one-liner is **absent** (Top Finding 1). The detail-on-demand pattern is only half-built. |
| Task-precise wording (generation vs editing) | **PRESENT (exemplary)** | Appendix C discloses frame chapters as "drafted by the assistant" (generation named as generation — the honest, higher-penalty framing, correct because true) and paper chapters as "re-typeset reproductions" + a fidelity-checked translation (editing named as editing). This is exactly the EMNLP-2024-informed distinction. **Protect it — do not soften "drafted" to "edited"** (false, and exposure-after-nondisclosure is the worst outcome, Schilke & Reimann). |
| PT-BR thinking trail | **PRESENT (rich)** | `storyline/01…07` (PT structure), three `AVAL_NECESSARIA_ptBR.md` audit docs, `ch1_beat_budget.md`, `capacity_baseline_experiment.md`. This is the near-unforgeable authorship evidence no generator produces incidentally. **Recommend preserving it past the defense** (do not delete the storyline/ tree). |
| Per-chapter pre-AI / post-AI / final checkpoints (NeurIPS format) | **ADEQUATE via git** | For AI-drafted frame chapters the "pre-AI" artifact is the author-approved outline (storyline/ + beat budgets), then the `draft(ai)` commit, then author edit/approval commits — a defensible three-point chain. Optional hardening: snapshot the explicit pre/post/final triple per chapter if belt-and-suspenders is wanted for a formal appeal. |
| Oral defensibility of any passage | **SUPPORTED (soft)** | The specificity audit (§2) is itself the evidence: the numbers and controls are the author's own experiments. Direct oral-readiness is persona 12's scope. |

## 6 · OVER-CORRECTION GUARD

Flag any place where tell-scrubbing has produced defensive, sterile, or vocabulary-flattened text (a
documented failure mode that harms L2 authors specifically and reads as its own red flag).

- **No over-correction detected in the frame.** The frame prose retains transitions, real hedges, and the
  author's register; 03's read-aloud confirms the voice is intact. The process is demonstrably aware of the
  risk — 03 §9 explicitly protects load-bearing CS vocabulary (framework, robust, baseline) and warns
  against sterilization.
- **The forward risk is on the re-typeset chapters** (Top Finding 2): if 03's -ly-band fix is applied
  bluntly to Ch3/Ch4, it would push *human-published* prose toward the defensive-sterile pattern and risk
  altering published wording. Trim decoration, keep function, stop early.
- **`co-equal` ×3 (03's finding)**: replacing it is fine (genuine awkwardness), but the guard is not to
  flatten "co-equal ends" into something bland — keep the meaning (neither target subordinate).

## 7 · REFRESH PASS (bounded web check, 2026-07-24 — proposed updates for author sign-off)

The two evidence files are 4–6 days old; a bounded pass confirms **nothing fundamental has moved** on the
stylometric-tells side. Genuinely-new, datable items worth folding into the evidence files (never
auto-applied):

1. **Vrije Universiteit Brussel study (peer-reviewed, June 2026)** — 4 detectors (Pangram, GPTZero,
   Turnitin, Copyleaks) × 160 academic papers >4,000 words, evenly split human-ESL / AI / hybrid /
   humanized. Only Pangram detected reliably (97.5% fully-AI, 95% humanized); the other three scored
   ~0% on fully-AI. This is *new peer-reviewed corroboration on LONG academic text* (the dissertation's
   exact genre) of both the legacy-detector collapse and Pangram dominance, and it tested ESL-human text
   directly. → add to `ai_detection_landscape` detector-landscape section.
2. **Peter Yang `/no-ai-slop` (open-source Claude skill, 22 Jul 2026, ~1k GitHub stars in a day)** —
   targets 20+ patterns including *binary contrasts* (= negative parallelism), fake-profound closers, and
   throat-clearing openers. Reinforces the Top-Finding-3 watch and is a new public human-tell catalog
   analog alongside Wikipedia WP:AISIGNS. → add to the human-perception catalog list.
3. **Substack platform-wide Pangram deployment (21–22 Jul 2026)**, scoring posts human / AI-assisted /
   AI-generated over 100 words. Context only: the Pangram-as-default-screener trend keeps spreading beyond
   venues/admissions; does not change a dissertation's threat model. → optional context note.
4. **Wikipedia WP:AISIGNS current state**: Grok-specific "underscore"/"causal/empirical/correlate"
   overuse persists into 2026; copula-avoidance and rule-of-three items stable; no new *structural* tell
   beyond the July-2026 baseline. Confirms the evidence file is current. No change needed.

## 8 · WHAT READS CREDIBLY HUMAN (protect it — do not push toward sterility)

- **The specificity (§2) is the crown jewel.** The capacity-baseline numbers, the freeze control, the
  +0.001 cosine, the admitted confound. This is what LLM filler cannot contain. Never trade it away in a
  trim.
- **The honest-arc reflective voice** — "the negative result … was the contribution's first half"; "a
  finding for this pair of tasks rather than a general rule." This is the subjective research voice the
  Witch-Hunt reviewers found *missing* in AI-suspected text. Keep it.
- **Burstiness and varied openers** (frame CV 49–57%; Abstract CV 41%). Protect against any smoothing pass.
- **Task-precise disclosure wording** (generation disclosed as generation). Correct and honest; protect it
  verbatim (§5).
- **The PT-BR trail + git discipline** — the actual shield. Preserve, do not prune.
- **The per-venue notation-dialect seam (Ch3 MTL/Single, Ch4 MTLNet/baseline, Ch5 Joint/Dedicated).**
  Persona 15 flags this as a *readability* defect and may be right to harmonize it. But from THIS channel it
  is a **credibility asset**: three visibly different provenances is what genuinely-different human papers
  look like, and a uniform machine-smoothed document would not have it. Conscious trade-off for the author:
  if harmonizing per 15, keep some human texture per chapter — do not flatten all three into one seamless
  machine voice, which would trade a readability gain for a small credibility loss.

## 9 · RANKED FINDINGS (channel · severity · location · direction — never applied)

1. **[credibility · MEDIUM · front matter]** No up-front layered disclosure line; disclosure only in
   Appendix C p. 87. → add a one-line front-matter disclosure pointing to Appendix C (detail-on-demand
   lowers the trust penalty; serves CAPES/CNPq visibility). Cross-ref 13 for placement mechanics.
2. **[over-correction guard · MEDIUM · Ch3/Ch4]** 03's -ly-band fix, if applied bluntly to the
   human-published chapters, risks sterilization + errata drift. → trim only named decorative adverbs; keep
   functional ones; stop short of flattening.
3. **[human channel · LOW–MEDIUM · document-wide]** Negative parallelism is the most examiner-primed 2026
   tell; currently in spec. → guard, do not add; freeze the count across future edit waves.
4. **[credibility asset trade-off · LOW · Ch3/4/5]** The notation-dialect seam (15's readability defect) is
   a human-provenance signal from this channel. → if harmonizing, retain per-chapter texture; do not
   machine-smooth to one voice.
5. **[human channel · LOW/NIT · Ch6 §6.1 L40]** "move the needle" business idiom (single instance) reads
   off-register. → 03/idiom-law counted scope (handoff); reword to "moved the results / mattered more than."
6. **[human channel · NIT/watch · §1.6, Appendix C]** Two bold-header-colon description lists are the eye-
   catch AI-shape spots, but sit in conventional dissertation contexts with specific content. → note only;
   optional conversion to running prose if the author wants to lower the visual signal.

## OUT-OF-SCOPE HANDOFFS (one line each)

- **Persona 03 (style/idiom counted gate):** "move the needle" (Ch6 §6.1 L40, single instance) is a
  motion/business idiom not on the current sweep — add to the idiom count.
- **Persona 13 (UFV compliance):** the *placement* of a front-matter disclosure line (Top Finding 1) and
  whether CAPES/CNPq require it in a specific front-matter location is compliance's call; I own only the
  credibility rationale.
- **Persona 15 (readability):** already owns the notation-dialect seam as a readability defect; §8 adds the
  credibility counter-weight for the author to balance.

## OPEN QUESTIONS (author only)

1. **Front-matter disclosure line** — do you want to add the one-liner (Top Finding 1), and where (folha-de-
   rosto footnote / preface / near the Abstract)? This is the single highest-value credibility edit.
2. **Detector-score posture** — if the banca or CAPES ever runs a detector, are you prepared to present the
   provenance (git trail + storyline/ PT-BR outlines + per-chapter checkpoints) rather than argue the score?
   The material exists; the question is whether to assemble it into a one-page "authorship evidence" packet
   pre-emptively.
3. **PT-BR trail retention** — confirm the `storyline/` tree and `AVAL_NECESSARIA_ptBR.md` docs are kept
   (not cleaned up) through and past the defense; they are your strongest authorship evidence.

---
_End of report. Screener risk MEDIUM (windowing caveat); expert-suspicion risk LOW. The text largely earns
its credibility; the one structural gap is the missing up-front disclosure line, and the one process risk is
over-scrubbing the human-published chapters. Re-run after the disclosure line lands and after any heavy edit
wave — tells creep back through AI-assisted rewriting._
