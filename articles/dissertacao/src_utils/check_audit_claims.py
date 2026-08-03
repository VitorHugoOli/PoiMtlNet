#!/usr/bin/env python3
"""check_audit_claims.py -- re-measures every "APPLIED" claim against the live source.

WHAT IT GATES. THREE things, and the scope is wider than the name -- stated here in full because a
docstring claiming one scope over code covering more is itself a defect this repo has hit:

  (1) the CODEX_AUDIT outcome table's COD-/NUM- findings;
  (2) fixes this project made on its own initiative (the `R8-` probes);
  (3) SINCE 2026-07-30 (round 9), the REVIEW TRACKER ITSELF -- the `R9-` probes below read
      src_utils/CONSIDERATIONS.md and src_utils/PENDENCIAS.md, which are NOT under src/. That is why
      there are two roots (SRC and UTILS) and why a probe's path is resolved against whichever root
      its file lives in. Round 9 sorted 43 reviewer points into apply/decide/blocked; the register of
      that split is a claim about the work in exactly the sense V14 means, so it is gated here rather
      than trusted.

An outcome table is a CLAIM ABOUT THE WORK, the highest-risk statement class here, and it was the one
class with no gate: on 2026-07-30 eight of nine findings marked APPLIED were still unapplied.

HOW TO ADD A PROBE -- do this in the SAME commit as the fix, not later.
    ("id", "what the claim asserts", "path/under/src", r"regex", True)   # True = must be PRESENT
Then prove it bites: revert the fix, run this file, read rc. rc must be 1. If the suite stays green,
the fix is undefended (GUARDRAILS §4b V15). A claim you cannot probe goes in NOT_CHECKABLE, never
silently omitted -- and RETIRED holds probes the author withdrew, kept visible so nobody "finishes"
a finding he closed.

FOUR MEASUREMENT TRAPS, each of which produced a WRONG verdict here before it was fixed:
  1. comment-blind matching scored a missing fix as done -- provenance comments quote the very
     strings being checked, so all matching runs on live_text().
  2. per-line matching scored a real fix as missing -- claims wrap; live_text() joins lines.
  3. an escaped `\%` mid-sentence truncated a paragraph and hid the clause after it, so the
     stripper only cuts an UNESCAPED `%`. Self-tested both ways before this file reports anything.
  4. a sabotage that does not reach live_text() reads exactly like a probe that never fires -- all
     seven `\begin{document}` in preamble.tex are inside comments. Assert the token is in
     live_text() before believing any verdict about it.

Full history -- why each probe exists, the closed-register audit, the corrected provenance of the
baseline measurements -- is in _round8/29_pendencias_detail.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
UTILS = Path(__file__).resolve().parent          # the trackers live HERE, not under src/
COMMENT = re.compile(r"(?<!\\)%")


def probe_root(rel: str) -> Path:
    """Which root a probe's path is relative to.

    Added round 9 with the R9- probes. The `%`-comment stripper stays on for BOTH roots: it is a
    no-op on markdown (no unescaped `%` in these trackers -- asserted in self_test), and turning it
    off per-root would be a second code path to get wrong.
    """
    return UTILS if rel.endswith(".md") else SRC

# (finding, what the audit claimed, file, pattern, want_present)
#   want_present=False -> the flagged string must be GONE for the claim to hold
#   want_present=True  -> the added text must be THERE
PROBES: tuple[tuple[str, str, str, str, bool], ...] = (
    ("COD-003",  'Ch.1 objective 4 no longer says "leakage-guarded"',
     "chapters/1_introduction.tex", r"leakage-guarded", False),
    ("COD-006a", '"well powered" removed from the Ch.5 protocol paragraph',
     "chapters/5_mobiwac/05_setup.tex", r"well powered", False),
    # COD-006b IS DELIBERATELY EXPECTED TO BE PRESENT. The audit flagged both "well powered" and
    # "before any result was read", but the author's decision was explicitly narrow: "Let's change
    # only the second point about the: 'The equivalence is well powered'." The analysis plan
    # genuinely WAS fixed before results were read, so the phrase is accurate and stays. The probe
    # is kept, inverted, so that a later agent "tidying up" the other half of the audit finding is
    # caught by the gate instead of silently overriding an author decision.
    ("COD-006b", 'the author kept "before any result was read" -- it is accurate and he said so',
     "chapters/5_mobiwac/05_setup.tex", r"before any result was read", True),
    ("COD-013",  "Appendix C names the model family in PROSE, not only in a comment",
     "chapters/apx_c_ai_disclosure.tex", r"Opus", True),
    ("COD-015a", "Ch.3 preface no longer says Ch.4 and Ch.5 both change the representation",
     "chapters/3_cbic.tex", r"revise that verdict by changing the input representation", False),
    ("COD-015d", "Ch.2 no longer promises a relative multi-task performance metric",
     "chapters/2_fundamentals.tex", r"relative multi-task performance", False),
    ("COD-016a", "Ch.3 unbalanced-result sentence rewritten",
     "chapters/3_cbic/results.tex", r"important to notice that since we have an", False),
    # COD-018 was HERE and is retired deliberately, not dropped. See RETIRED below.
    # ---- ROUND-8 FIXES OF OUR OWN, not CODEX_AUDIT findings. Added 2026-07-30 after a review
    # observed that this file gated only the inherited audit and nothing round 8 repaired on its own
    # initiative: all three reverts below left 22 gates green, measured by sabotage.
    ("R8-head",  'Ch.5 says "region output", not the banned "region head"',
     "chapters/5_mobiwac/06_results.tex", r"region\s+output\s+was\s+driven", True),
    ("R8-head2", 'the same sentence says "region-transition prior", not the repo shorthand',
     "chapters/5_mobiwac/06_results.tex", r"region-transition\s+prior", True),
    ("R8-vintage", "Ch.6 data-vintage item prints BOTH Gowalla windows, the paper's and the measured one",
     "chapters/6_conclusion.tex", r"August\s+2011", True),
    ("R8-bibfont", "no footnotesize wrapper around the bibliography (REV-024, archived on one measurement)",
     "preamble.tex", r"footnotesize", False),
    ("NUM-4",    "HGI sweep reports its spreads and its averaging convention",
     "chapters/apx_g_hgi_tuning.tex", r"0\.8186", True),
    # ---- ROUND-9c PROBES: the AUTHOR'S OWN RULINGS of 2026-07-30, one per mechanically checkable row.
    # Ledger and per-row evidence: _round9/47_applied_check.md. These exist because a future edit that
    # undoes one of his requested changes must trip a gate rather than reach the banca silently.
    # The nine REMOVAL probes were each validated against git show 06529ed6:<file>, where the pattern
    # matched the original exactly once -- so an absence here is evidence and not an inexpressible pattern.
    ("A22-2",  "his point 2: 'stranger result' is gone",
     "chapters/apx_f_cosine.tex", r"stranger", False),
    ("A22-3",  "his point 3: the arc sentence relating this result to the first study is gone",
     "chapters/apx_f_cosine.tex", r"changed so little in the first study", False),
    ("A22-5",  "his point 5: the over-detailed series clause is gone",
     "chapters/apx_f_cosine.tex", r"carry a partial re-run on top of theirs", False),
    ("A22-6",  "his point 6: the development-time implementation detail is gone",
     "chapters/apx_f_cosine.tex", r"this appendix supersedes nothing there", False),
    ("A22-8",  "his point 8: the British needs+gerund is gone and the American form is present",
     "chapters/apx_f_cosine.tex", r"needs saying", False),
    ("A22-8b", "his point 8, the replacement half",
     "chapters/apx_f_cosine.tex", r"must be stated plainly", True),
    ("A22-9",  "his point 9: the AI-shaped opener is gone",
     "chapters/apx_f_cosine.tex", r"Two departures from that flat picture", False),
    ("A22-10", "his point 10: the 'rather than smoothing' clause is gone",
     "chapters/apx_f_cosine.tex", r"worth reporting rather than smoothing", False),
    ("A22-11", "his point 11: the t-test sentence names its datasets",
     "chapters/apx_f_cosine.tex", r"does reject at Alabama and at Georgia", True),
    ("A22-12", "his point 12: the native-idiom sentence is gone",
     "chapters/apx_f_cosine.tex", r"point away from trouble", False),
    ("A22-13", "his point 13: the arc-of-three-studies paragraph is gone",
     "chapters/apx_f_cosine.tex", r"arc of the three studies", False),
    ("A22-14", "his point 14: orthogonal gradients do not mean the tasks share no knowledge",
     "chapters/apx_f_cosine.tex", r"share no knowledge", True),
    ("A22-1",  "his point 1: the negative-transfer claim carries its citation",
     "chapters/apx_f_cosine.tex", r"standley2020tasks", True),
    ("A22-4",  "his point 4: the cosine approach carries its citation",
     "chapters/apx_f_cosine.tex", r"yu2020pcgrad", True),
    ("A22-7",  "his point 7: the fold statement survived the simplification",
     "chapters/apx_f_cosine.tex", r"unit of independence is the fold", True),
    ("A11-diss", "his 2.11 option B: the dissertation's Ch.5 carries the non-inferiority caveat",
     "chapters/5_mobiwac/06_results.tex", r"non-inferior", True),
    ("A11-frame", "his 2.11 option B: the caveat reaches the Resumo and Abstract, which dropped it before",
     "content.tex", r"n[aa\u00e3]o-inferior", True),
    ("A11-frame2", "his 2.11 option B: the English abstract too",
     "content.tex", r"non-inferior", True),
    ("A14-nash", "his 2.14: the Nash-MTL entry carries the PMLR page range he supplied",
     "references.bib", r"16428--16446", True),
    ("A12-errata", "his 2.12: the Pareto-optimality narrowing has its errata row",
     "tables/cbic/errata.tex", r"Pareto", True),
    # ---- ROUND-9f, 2026-08-02. Item 2.9 was closed on evidence that could not distinguish
    # done from not-done: the probe searched for "+0.001" and "0.0032", which are the OLD
    # four-seed development numbers the item itself was ABOUT, so their presence proved nothing.
    # The closure line also said "both trees edited" when the measurement globbed only src/**.
    # These three probe the actual requirement -- the same standalone sentence in BOTH trees,
    # and the appendix pointer in the dissertation's section preamble where it can resolve.
    ("A9-diss",  "2.9: the seven-dataset result is stated in the dissertation's Ch.5, in wording that "
     "does not depend on holding another document",
     "chapters/5_mobiwac/02_related.tex",
     r"measured on the final model across seven datasets, is\s+positive at every one of them", True),
    ("A9-ptr",   "2.9: the pointer to the gradient-cosine appendix lives in the section preamble, where "
     "an internal ref resolves",
     "chapters/5_mobiwac/02_related.tex",
     r"Appendix~\\ref\{apx:cosine\} reports the gradient-cosine", True),
    ("A9-oldnum","2.9: the earlier four-seed figures are LEFT AS THEY WERE, not silently restated as the "
     "seven-dataset result (this is the pair whose presence the withdrawn probe mistook for proof)",
     "chapters/5_mobiwac/02_related.tex", r"\+0\.0032", True),
    # ---- REPOINTED 2026-08-02, when the author's revised tree (src_clean) was merged into src.
    # Seven probes went NOT APPLIED after the merge. Each was checked against the SUBSTANCE rather
    # than trusted or deleted, and in every case the claim still holds and the PATTERN was stale:
    #   A11-frame/A11-frame2  he writes the adjective ("non-inferior") where the probe demanded the
    #                         noun ("non-inferiority"). The TOST caveat is in both Resumo and Abstract.
    #   A23-EX9               he kept the Pareto-front sentence he had ruled not to change, in his own
    #                         words; the probe pinned the old phrasing.
    #   R9-pareto/2/3         the Pareto disclaimer, the dominance definition and the optimality
    #                         definition all survive his rewrite, worded differently.
    #   R9-conflict           he did better than the original: gradient conflict now has its own
    #                         subsection defining it as a negative cosine between task gradients.
    #   NUM-4                 THE ONE THAT WAS NOT A WORDING CHANGE. The HGI sweep numbers left
    #                         2_fundamentals entirely -- because he MOVED them to a new appendix,
    #                         chapters/apx_g_hgi_tuning.tex, which renders at p. 106 of the defense
    #                         build. Repointed to that file. Had I repointed by pattern-fiddling
    #                         instead of looking for the number, I would have recorded a lost
    #                         measurement as a wording change.
    # The rule this follows: a probe that fails after a legitimate rewrite is repointed only once the
    # claim has been re-verified in the new text. A probe deleted because it failed is a claim that
    # silently stopped being checked.
    # ---- ROUND-9c, SECOND PASS. These three rows were APPLIED in the ledger on a predicate that did
    # not measure anything: 2.19's ended in `or True`, 2.23's was the literal `True`, and 2.15's asked
    # only whether the word "errata" occurs somewhere. A reviewer caught it. Probed properly now, which
    # is the difference between a verdict and an assertion (_round9/47_applied_check.md, the CORRECTION).
    ("A19-conv", "his 2.19: the word-count convention is a durable record stating the figure of record",
     "WORDCOUNT_CONVENTION.md", r"310", True),
    ("A23-R3",  "his 2.23 R-3: the unscoped limit on what the balancers can contribute is gone",
     "chapters/2_fundamentals.tex", r"limit on what any of these methods can contribute", False),
    ("A23-R6",  "his 2.23 R-6: the cosine appendix is reachable from the body, not one sentence",
     "chapters/6_conclusion.tex", r"apx:cosine", True),
    ("A23-EX6", "his 2.23 EX-6: the hard-sharing-costs-nothing claim is gone from the appendix",
     "chapters/apx_f_cosine.tex", r"hard sharing costs nothing", False),
    # SUPERSEDED 2026-08-02, and this probe was the wrong instrument for the whole time it existed.
    # His 2.23 ruling was "nao aplique o EX-9" -- EX-9 being four sentences: "deserves one statement"
    # (p.23), "worth reporting" (p.99), "needs saying" (p.98), "worth stating" (p.21). This probe
    # watched r"Pareto front" instead, which is still in the chapter and has nothing to do with the
    # four phrasings. So it PASSED while all four left the document: two in his own revised tree
    # (807183c1), two in earlier rounds, per git log -S on each phrase.
    # He was shown the contradiction and ruled that his own read-through supersedes the earlier
    # decision -- he rewrote those sentences with the text in hand, and one of them ("needs saying")
    # is the exact British need+gerund construction he later banned by name in WRITING_LAW.
    # Repointed to what his ruling ACTUALLY protected and what survives: the Pareto-front DEFINITION
    # he declined to cut. The lesson is the same one this file keeps relearning -- a probe must
    # express the claim it guards, or it certifies whatever it happens to match.
    ("A23-EX9", "his 2.23 EX-9 ruling, now SUPERSEDED by his own revision: the Pareto-front "
     "definition he declined to cut is still there (the four 'worth noting'-family phrasings are "
     "not, by his later choice)",
     "chapters/2_fundamentals.tex", r"corresponding loss vectors form the Pareto front", True),
    ("A15-cite", "his 2.15 path A: the substituted citation is present in the dissertation tree",
     "chapters/3_cbic/method.tex", r"baxter2000model", True),
    ("A15-old",  "his 2.15 path A: the unsupported citation is gone from the dissertation tree",
     "chapters/3_cbic/method.tex", r"ruder2017sluice", False),
    ("A15-term", "his 2.15: the banned term is gone from the CoUrb chapter",
     "chapters/4_courb/methodology.tex", r"\bfclass\b", False),
    # ---- ROUND-9 PROBES: the review-tracker split itself. Paths ending .md resolve against UTILS.
    # Each was validated by sabotage (revert the property, read rc=1) -- see _round9/32_gate_validation.md.
    ("R9-schema", "CONSIDERATIONS.md carries all 43 per-item blocks, not prose",
     "CONSIDERATIONS.md", r"### AUT-01", True),
    ("R9-commit", "every item block records the build commit its measurement was taken against",
     "CONSIDERATIONS.md", r"Build commit the measurement was taken against", True),
    ("R9-verbal", "Germano's points are marked as the AUTHOR'S TRANSCRIPTION of verbal comments, "
                  "never attributed to him as written words",
     "CONSIDERATIONS.md", r"transcribed by the author", True),
    ("R9-stale",  "the stale-quote counts are stated as 9 of 41, not the 10 I first wrote",
     "CONSIDERATIONS.md", r"9 de 41 ancoras localizaveis", True),
    ("R9-blocked", "FAB-28 is recorded as BLOCKED on a failed verification, not quietly applied",
     "CONSIDERATIONS.md", r"INADMISSIBLE for any claim", True),
    ("R9-pend6",  "PENDENCIAS carries the new section 6 and it replaced the 2.8 placeholder",
     "PENDENCIAS.md", r"## §6 · As decisoes que sairam do `CONSIDERATIONS\.md`", True),
        # REPOINTED 2026-08-02: item 2.8 was archived out of the tracker, so both probes now read the
    # ARCHIVE. The second also gets a wrap-tolerant pattern: the phrase it pins now wraps across
    # two lines in the archive file, and the original one-line pattern reported the sentence as
    # missing when it is present. A first check of mine split on exactly that wrap and nearly
    # recorded a live sentence as lost.
("R9-pend28", "the archived 2.8 records what was done rather than asking for a decision",
     "_archive/PENDENCIAS_RESOLVIDOS.md",
     r"2\.8 `CONSIDERATIONS\.md` — EXECUTADO nesta rodada", True),
    # ---- ROUND-9, THE PARETO TRACK (PENDENCIAS_RESOLVIDOS 2.12 (arquivado 2026-08-02), author's decision "DESICAO: A."). The fix has
    # two halves that can rot independently -- the §2.3 passage can be reverted, and the glossary rows
    # it depends on can be dropped -- so it gets a probe on each. Source ledger with the page of every
    # definition, and the sabotage runs: _round9/31_pareto.md.
    ("R9-pareto", "Ch.2 defines Pareto optimality and states that this dissertation does not claim it",
     "chapters/2_fundamentals.tex", r"claims no Pareto property", True),
    # R9-pareto2: the DEFINITION itself, not just the honesty clause. Found 2026-07-30 by sabotage:
    # replacing "a relation named\nPareto dominance" with "domination" left R9-pareto holding, because
    # that probe watches "claims no Pareto property of any kind" -- a different sentence. A term can be
    # struck from the definition while the disclaimer about it survives, and the glossary probe only
    # checks the REGISTRY, not the prose. So the fail-closed rule needs the prose side gated too.
    # ---- APPENDIX F's DATASET COUNTS. The cosine track measured that NOTHING gated them: it widened
    # the appendix from four datasets / 3,900 observations to seven / 4,650, verified every count in the
    # rendered PDF, and reported honestly that the counts rested on that one session's reading and on
    # nothing mechanical. It could not add the probe itself because this file belonged to the parallel
    # track that round. Three probes, so a silent regression to the old counts cannot pass:
    ("R9-apxf7",  "Appendix F reports SEVEN datasets, matching the seven states in the parquet",
     "chapters/apx_f_cosine.tex", r"seven datasets", True),
    ("R9-apxfn",  "Appendix F reports 4,650 epoch-level cosines, the measured row count",
     "chapters/apx_f_cosine.tex", r"4,650", True),
    ("R9-apxfold", "the superseded four-dataset counts are GONE from Appendix F (inverted)",
     "chapters/apx_f_cosine.tex", r"3,900|four datasets", False),
    ("R9-pareto2", "Ch.2 still DEFINES Pareto dominance in prose, not only disclaims Pareto claims",
     "chapters/2_fundamentals.tex", r"Pareto dominance", True),
    ("R9-pareto3", "Ch.2 still defines Pareto optimality from dominance",
     "chapters/2_fundamentals.tex", r"Pareto optimal when no other setting dominates", True),
    ("R9-conflict", "Ch.2 defines gradient conflict as the cosine between per-task gradients, so "
                    "Appendix F's orthogonality result has a definition to point back to",
     "chapters/2_fundamentals.tex", r"cosine between their gradients", True),
    # INVERTED, and the inversion is the point (GUARDRAILS V15: a fix whose correctness is an ABSENCE
    # needs an expect-not-found probe). Appendix F's dataset coverage was being extended by a parallel
    # track while this passage was written, so the passage cites it by \ref and says "the datasets
    # measured there", never a count. If a later editor writes a count into the cosine sentence, that
    # number acquires a second home and the two disagree the next time one moves -- V6, a corrected
    # number surviving at a second site.
    #
    # THE FIRST VERSION OF THIS PROBE WAS WRONG, AND IT FIRED, WHICH IS HOW I FOUND OUT. It banned
    # /(four|six)\s+datasets/ anywhere in the chapter and matched line 892, "on the next region at four
    # of six datasets" -- the §2.5 headline result, which is the protected region-wording law
    # (WRITING_LAW §3: outperforms at four, matches at the other two) and must NOT be touched. A ban on
    # a bare phrase cannot tell one claim from another, so the pattern is anchored on the SENTENCE this
    # probe is actually about: the cosine clause, whose subject is Appendix F's coverage. The prose was
    # correct and the instrument was not, which is the reverse of the usual case and the reason the
    # rule is to validate an instrument before believing it (§4b V3).
    # Kept INVERTED so it stays a true expect-not-found, and anchored so only the cosine sentence can
    # trip it: the ban is a count in the clause that follows "indistinguishable from orthogonal on".
    # Written as an alternation of the number words that could plausibly land there (a bare digit too),
    # which is narrower than banning the phrase chapter-wide and wider than banning today's value.
    ("R9-nocount", "the Ch.2 cosine sentence cites Appendix F's coverage by reference, not by count",
     "chapters/2_fundamentals.tex",
     r"indistinguishable from orthogonal on (?:all )?(?:one|two|three|four|five|six|\d+)\b", False),
    # The registry half. The prose above may not exist without these rows (GLOSSARY.md is fail-closed,
    # and this term was in prose UNREGISTERED at five sites for two rounds, which is the breach
    # PENDENCIAS_RESOLVIDOS 2.12 (arquivado 2026-08-02) records). A .md path resolves against UTILS, and GLOSSARY.md is a level above it,
    # hence the "../" -- probe_root joins, it does not restrict.
    ("R9-glossary", "the four Pareto/conflict terms are registered before the prose uses them",
     "../GLOSSARY.md", r"\*\*Pareto-stationary point\*\*", True),
    # ---- R9-agree / R9-agree2: THE TWO TRACKERS CONTRADICTED EACH OTHER FOR TWO COMMITS and twenty
    # green gates said nothing, because every probe here checks ONE file against ITSELF. PENDENCIAS
    # §2.8 was generated from a Python variable whose reassignment cell had aborted on an
    # AssertionError, so it silently kept the pre-correction figures (ten stale, "todas citam
    # 0_main.tex", 21 exact) while §6 of the SAME FILE and all of CONSIDERATIONS.md carried the
    # corrected nine. The assertion that fired was protecting the OTHER file; nothing was watching
    # whether the two agreed. A number with two homes disagrees the next time one of them moves
    # (GUARDRAILS §4b V6), and here the second home was three screens down in the same document.
    # Positive probe pins the corrected figure; the inverted one bans the superseded claim. The
    # corrected count is ALSO pinned in CONSIDERATIONS.md by R9-stale, so the two files can only pass
    # together.
    ("R9-agree",  "the archived 2.8 carries the CORRECTED stale count, agreeing with its own §6 and "
                  "with CONSIDERATIONS.md (9 of 41, not the superseded 10)",
     "_archive/PENDENCIAS_RESOLVIDOS.md",
     r"\*\*32 sao exatas e 9 estao\s+obsoletas\*\*", True),
    ("R9-agree2", "the superseded 'As 21 ancoras dos capitulos' claim is gone from PENDENCIAS",
     "PENDENCIAS.md", r"As 21 ancoras dos capitulos", False),
    # ---- R9-confirm: FAB-01 CHANGED NOTHING, and it was still counted as an applied-and-verified
    # edit. The Wave A loop wrote the standard note -- "verified in the RENDERED PDF, both
    # directions: new wording present and old wording absent" -- onto every member of the wave,
    # including the one whose request was ALREADY satisfied before this round started. For a no-op
    # there is no superseded wording, so the absent half of that claim cannot be measured, and
    # asserting it is a claim about a measurement that never ran. Same V13 shape as the stale-anchor
    # miscount earlier this round: a headline count that stopped agreeing with its own members.
    # The probe pins the carve-out, so the note cannot be re-applied to a no-op by a later sweep.
    ("R9-confirm", "FAB-01 is recorded as already satisfied and only CONFIRMED, never as an applied "
                   "edit verified in both directions",
     "CONSIDERATIONS.md", r"ALREADY SATISFIED; CONFIRMED IN THE RENDERED PDF, NOT APPLIED", True),
    # ---- R9-clock: the reviewer round's own process claim. The four personas were given a 25-minute
    # (1,500 s) wall-clock checkpoint BECAUSE the previous round overran (45 min -> 60, 90 -> 219),
    # and the gate report first said all four came back inside it. All four missed it: 1,598 / 1,618 /
    # 1,971 / 2,314 s, the worst 54% over. The wall times were in the results the findings were read
    # from. Reporting compliance would have retired a control on a number nobody looked at, and the
    # next round would have inherited a checkpoint believed to work. Positive probe pins the worst
    # measured time (the figure a later summary is most likely to soften); the inverted one bans the
    # superseded sentence. .md paths resolve against UTILS, so _round9/ is reachable from here.
    ("R9-clock",   "the reviewer round records that every persona OVERRAN the 25-minute checkpoint, "
                   "with the measured wall times",
     "_round9/37_reviewer_gate_round9.md", r"2,314 s \(38\.6 min\)", True),
    # R9-clock2 IS ANCHORED, and the first version was not, which is why it fired on a clean file.
    # The banned string is quoted inside the correction that retires it ("The first version of this
    # paragraph said \"all four came back inside it\""), and PROBES match case-insensitively, so a bare
    # ban on the phrase cannot tell the retired ASSERTION from the record of retiring it. Same lesson
    # as R9-nocount's first version, which banned a bare phrase and matched the protected region
    # wording. The pattern therefore requires the sentence-initial capitalized form followed by the
    # clause that made it an assertion -- the shape only the original can have. Keeping the quotation
    # legal is the point: a correction that cannot name what it corrects is not a correction.
    ("R9-clock2",  "the superseded 'All four came back inside it' ASSERTION is gone from the gate "
                   "report (the quotation of it inside the correction stays legal)",
     "_round9/37_reviewer_gate_round9.md", r"All four came back inside it, and \*\*all four", False),
    # ---- R9-wave2: wave 2's checkpoint outcome, and the reason it needs a probe is that the LAST
    # wave's outcome was reported wrong and then its diagnosis was reported wrong too. Wave 1: 0 of 4
    # inside a 25-min budget, mean 31.3 min, which I first wrote as "all four came back inside it"
    # (R9-clock). The correction ended with a prescription -- narrow the scope, not the clock -- and
    # wave 2 acted on it: narrowed scopes, 30-min budget, and the result was 1 of 5 inside, mean 41.1
    # min. The mean got WORSE. A summary of this round is likeliest to round both waves off to "the
    # personas ran and reported", so the probe pins the number that makes the failure legible: 1 of 5.
    #
    # THIS PROBE'S FIRST PATTERN WAS WRONG AND IT FIRED ON A CLEAN FILE, which is the third time this
    # round an instrument was the defect rather than the document (after R9-clock2 and R9-nocount). I
    # wrote it against a markdown table row -- r"\*\*1 of 5\*\* \| \*\*41\.1 min\*\*" -- that lives in
    # _round9/34, not in 38; the wave-2 report states the same measurement in prose. A pattern written
    # from memory of a sibling file is not a measurement of this one. Anchored now on the prose 38
    # actually carries.
    ("R9-wave2",   "the wave-2 record states that only ONE of five personas came in under the "
                   "checkpoint, with the measured mean",
     "_round9/38_reviewer_wave2_round9.md", r"one of five inside, mean 41\.1 minutes", True),
    # The blocker that wave 2 found, pinned in the tracker rather than only in a report: Appendix F
    # tells the reader an experiment happened (replacing the sharing scheme in study 1) that Chapter 3
    # lists as future work. Re-verified in three places before it was written down. Inverted probes
    # would be wrong here -- the prose is the AUTHOR'S to change, so what gets gated is that the item
    # stays on his decision list until he rules on it, not that the sentence is already gone.
    ("R9-blq4",    "the Appendix F never-run-experiment blocker is on the author's decision list",
     "PENDENCIAS.md", r"BLQ-4 — o Apendice F descreve um experimento que nunca foi feito", True),
    ("R9-blq5",    "the PCGrad blocker is recorded as DOWNGRADED BY ME, with the half I could not "
                   "check named, rather than passed through at the persona's severity",
     "PENDENCIAS.md", r"eu o REBAIXEI; a decisao final e sua", True),
    # ---- ROUND-10 PROBES: the author's 28 rulings, applied 2026-08-03 against baseline dda8978e.
    # R10-blq2 is the one that would rot silently. His ruling was to keep "everywhere" for the CATEGORY
    # verdict and specify the partition where the REGION verdict needs it. Chapter 6 was already right;
    # ONE site survived in the introduction, and a line-oriented grep could not see it because the
    # sentence wraps across three source lines -- only the comment-stripped, whitespace-collapsed
    # concatenation found it. Any future rewrite of that bullet is likely to re-collapse the partition
    # into "outperforms or matches", which is the exact phrase the conclusion's own comment at :131 and
    # 5_mobiwac/06_results.tex:76 record as a defect. So: pin the partition positively AND ban the
    # collapsed form in that file. The ban is file-scoped on purpose -- the phrase is legal in comments
    # elsewhere, where it names the defect rather than committing it.
    ("R10-blq2",   "the introduction's practical-contribution bullet states the region partition "
                   "(four of six + TOST) instead of collapsing it",
     "chapters/1_introduction.tex",
     r"on the region task at four of the six; at the other two it remains statistically", True),
    ("R10-blq2b",  "the superseded collapsed wording is gone from the introduction's live prose",
     "chapters/1_introduction.tex", r"either outperforms or\s+matches it on next region", False),
    # R10-blq3: the two prose-derived ratios are gone and must not come back. The endpoint counts stay
    # (each traceable to the appendix's own table), so the positive half of this pair guards against a
    # well-meaning editor "restoring the span" by recomputing it in prose again -- which is how it got
    # there. N2/N3: quote, never compute.
    ("R10-blq3",   "the prose-derived scale ratios are gone from Appendix F, endpoint counts kept",
     "chapters/apx_f_cosine.tex", r"factor of thirty-six", False),
    ("R10-blq3b",  "Appendix F still carries the endpoint check-in counts the ratios were derived from",
     "chapters/apx_f_cosine.tex", r"113,846 at Alabama to 4,089,892 at Texas", True),
    # R10-fab22: Istanbul's PURPOSE, not just its name. The author's point was that a reader who sees
    # only "and Istanbul" reads a sixth dataset rather than evidence about generalization. The claim the
    # sentence makes is about scope, so the scope has to be stated.
    ("R10-fab22",  "the introduction says WHY Istanbul is there (a non-United-States dataset), not "
                   "merely that it is there",
     "chapters/1_introduction.tex", r"Istanbul as a non-United-States dataset", True),
    # R10-ch2defs: GER-08/09/10. The chapter must keep numbered, REFERENCEABLE definitions -- the
    # environment declaration in preamble.tex is what makes them numbered, and the cross-references are
    # what make the chapter GER-10's narrative rather than a definition dump. Two probes because the
    # two halves fail independently: someone could keep the blocks and drop the references.
    ("R10-defenv", "Chapter 2 declares a chapter-numbered definition environment",
     "../src/preamble.tex", r"\\newtheorem\{definition\}\{Definition\}\[chapter\]", True),
    ("R10-cosine", "gradient conflict is DEFINED with the cosine in Chapter 2, not only named",
     "chapters/2_fundamentals.tex", r"def:fund:conflict", True),
    # R10-hamtl: FAB-28, the round-9 blocker. The novelty sentence SURVIVED the PDF (HAMTL calls its
    # category head auxiliary, names no region-like unit in 28 pages, reports no category metric), so
    # what needs guarding is that nobody later "narrows it to be safe" without reading the paper, and
    # that our description keeps the authors' own main/auxiliary framing (GUARDRAILS R2).
    ("R10-novelty", "the co-equal-end-targets claim still stands, having survived the HAMTL PDF",
     "chapters/2_fundamentals.tex",
     r"none treats next category and next region as", True),
    ("R10-hamtl",  "HAMTL is described as its own authors describe it: location prediction main, "
                   "category auxiliary",
     "chapters/2_fundamentals.tex",
     r"HAMTL sets location\s+prediction as its main task and category prediction as an auxiliary task",
     True),
    # ---- R10-pm: the postmortem's own causal account, which was FABRICATED on its first writing and is
    # therefore the one claim in this repo with a demonstrated tendency to drift back toward plausibility.
    # History: the round-10 harness reported working probes as failing. That much was diagnosed correctly.
    # Then, asked what broke, I wrote three mechanisms -- an IFS=':' split on "a Portuguese sentence
    # containing a colon", a helper that restored before reading stdout, and a stale module import -- and
    # THREE OF THE FOUR CLAIMS WERE INVENTED. No leg carried Portuguese; no helper version restored early;
    # every module load post-dated the edit. The real cause is ONE mechanism, isolated by reproduction:
    # str.replace(old, new, 1) against a string occurring twice where the FIRST occurrence sits inside a
    # % comment, which live_text() strips -- so the live text stays intact and the probe correctly holds.
    # Verified for both affected legs: def:fund:conflict occurs twice in 2_fundamentals.tex, and
    # \newtheorem{definition}{Definition}[chapter] occurs twice in preamble.tex (line 99 in a comment,
    # line 117 live). The positive probe pins the single-mechanism finding; the inverted one bans the
    # invented detail from being ASSERTED again. The ban is anchored so the RETRACTION may keep quoting
    # it -- same construction as R9-clock2, for the same reason: a correction that cannot name what it
    # corrects is not a correction.
    ("R10-pm",     "the round-10 postmortem records ONE harness mechanism, isolated by reproduction, "
                   "not the three it first invented",
     "_round9/34_tracker_disagreement.md", r"There was \*\*ONE\*\* harness bug", True),
    ("R10-pm2",    "the fabricated three-mechanism account is not ASSERTED anywhere (quoting it inside "
                   "the retraction stays legal)",
     "_round9/34_tracker_disagreement.md",
     r"harness was broken, in three separate ways", False),
    ("R10-pm3",    "the postmortem states the run's REAL leg counts rather than the three-of-six it "
                   "first claimed",
     "_round9/34_tracker_disagreement.md",
     r"\*\*one\*\* `DID NOT FIRE`, \*\*two\*\* `mutation failed`", True),
    # ---- ROUND-11 PROBES: the five items the author authorized on 2026-08-03, after suspending the
    # Chapter 2 page budget ("pode melhorar o texto da fundamentacao sem preocupacao de paginas").
    #
    # R11-aligned is the one that earns its keep. The Aligned-MTL sentence said the method "adjusts the
    # PRINCIPAL components of the gradient system". The paper says it aligns the ORTHOGONAL components
    # (arXiv:2305.19000v1 abstract) and its stated criterion is the condition number of the linear system
    # of gradients. "Principal" invites a PCA reading the authors never claim, so this was an R2 defect
    # (describe a system as its own authors describe it), and it survived a sub-agent's own verification
    # pass plus a round-10 commit. It was caught only when the four lineage attributions that had never
    # been checked by anyone but their writer were finally checked. A summariser rewording this sentence
    # is exactly how "principal" comes back, so both halves are pinned: the authors' word positively,
    # the wrong word banned.
    ("R11-aligned", "Aligned-MTL is described with its authors' own word, ORTHOGONAL components, and "
                    "its stated condition-number criterion",
     "chapters/2_fundamentals.tex",
     r"condition number of the linear\s+system of task gradients", True),
    ("R11-aligned2","the superseded 'principal components of the gradient system' gloss is gone from "
                    "the live prose (the comment recording the correction may keep quoting it)",
     "chapters/2_fundamentals.tex", r"adjusts the principal components of the\s+gradient system", False),
    # R11-def27: the merge that round 10 made PURELY to save one page is undone, on the author's
    # authorization. The two cross-references point at opposite halves, so a re-merge silently breaks
    # both. Pin the second label's existence: it only exists because the split happened.
    ("R11-def27",  "Definition 2.7 is split, so each cross-reference can point at the concept it means",
     "chapters/2_fundamentals.tex", r"\\label\{def:fund:checkinlevel\}", True),
    # R11-hgi: GER-02's second half. HGI is the place-level baseline the whole argument turns on, and the
    # honest caveat is the part most likely to be trimmed by a later editor tightening prose: HGI was
    # built and evaluated for urban REGION representation, and this dissertation repurposes its POI-level
    # output for sequential prediction, which its original evaluation does not cover.
    ("R11-hgi",    "the HGI explanation keeps the honest caveat that this project repurposes a POI-level "
                   "output the original evaluation does not cover",
     "chapters/2_fundamentals.tex",
     r"repurposes that POI-level output for sequential prediction, a use the\s+original evaluation does not cover",
     True),
    # R11-fab15: the taxonomy must not read as part of the task DEFINITION in the introduction (the
    # author: the approach is generic and not fixed on this taxonomy), while the concrete instantiations
    # survive ONCE so an introduction-only reader keeps the context. Both halves, since either can drift.
    ("R11-fab15",  "the introduction states the targets plainly, with the label sets as properties of "
                   "the data rather than part of the definition",
     "chapters/1_introduction.tex",
     r"The prediction targets are the next category and the next region\.", True),
    ("R11-fab15b", "the definitional framing of the taxonomy is gone from the introduction",
     "chapters/1_introduction.tex", r"over the\s+seven-class taxonomy defined in Chapter", False),
    # R11-gloss: the two rows the author authorized. They were already in live prose while unregistered,
    # so the fail-closed rule was being stretched; these rows close it.
    ("R11-gloss",  "the two authorized registry rows are present (soft parameter sharing)",
     "../GLOSSARY.md", r"\| soft parameter sharing \| compartilhamento flex", True),
    ("R11-gloss2", "the second authorized registry row is present (negative transfer)",
     "../GLOSSARY.md", r"\| negative transfer \| transfer", True),

    # ---- ROUND-12 PROBES: the liu2019dwa [VERIFY] closed on 2026-08-03 against the PDF the author
    # put on disk (science/articles/1803.10704v2.pdf, arXiv:1803.10704v2). Three things were read in
    # the BODY, and all three are pinned here because each has its own way of drifting back:
    #   the NAME. The clause said "dynamic weight averaging". The authors' own name is "Dynamic
    #     Weight Average (DWA)" (p.2, end of Sec. 1; heading of Sec. 4.1.3, p.5). An R2 defect of
    #     the same class as R11-aligned, and the likeliest thing for a later editor to "normalize"
    #     back to the participle, since the CBIC chapter legitimately carries "Averaging" as its
    #     published wording. Both halves pinned: the paper's name positively, the gloss banned.
    #   the WORD "ALONGSIDE", which is what kept the sentence defensible while it was unverified and
    #     which the body confirms rather than overturns: DWA is introduced in the experimental
    #     section as one of three weighting schemes and is run on the Split, Dense and Cross-Stitch
    #     baselines as well as on MTAN (Tables 2-3, pp. 6-7), so it is separable from the
    #     architecture. Anyone tightening this clause to a bare "introduce DWA" would assert a
    #     packaged contribution the paper does not present, so the collocation is pinned whole.
    #   the DEFINITION. Eq. 7 and its text (p.5) set w_k(t-1) = L_k(t-1)/L_k(t-2) over epoch-average
    #     losses, which is narrower than the earlier "recent changes in the task losses".
    ("R12-dwa",    "DWA carries its authors' own name and stays SEPARABLE from MTAN "
                   "(the 'alongside' the closed [VERIFY] was protecting)",
     "chapters/2_fundamentals.tex",
     r"Dynamic Weight Average alongside their attention\s+architecture", True),
    ("R12-dwa2",   "the superseded 'dynamic weight averaging' gloss is gone from the live prose "
                   "(comments recording the correction may keep quoting it)",
     "chapters/2_fundamentals.tex", r"dynamic weight averaging", False),
    ("R12-dwa3",   "the DWA weight is described as the paper defines it, from the loss rate of "
                   "change measured as the ratio of the two previous loss values",
     "chapters/2_fundamentals.tex",
     r"rate of change of that task's loss,\s*measured as the ratio of its two previous loss values",
     True),
    # ---- TWO DEFECTS THE AUTHOR FOUND IN THE DEFINITION BLOCKS, 2026-08-03. Both are formalization
    # gaps rather than wrong statements, and both sit at the chapter's load-bearing points, which is
    # why they are gated: a later editor tightening prose has every incentive to drop a symbol or
    # collapse a one-line equation back into a sentence.
    #
    # (1) Definition 2.8 named NO symbol while Definition 2.7 gave the place-level vector as
    #     \mathbf{e}_p. The place-versus-check-in contrast is the pivot of the entire dissertation, and
    #     only one side of it could be written down. GLOSSARY §1.1 had already implied the gap: its
    #     \mathbf{e}_p row reads "distinct from a per-visit Check2HGI vector" and gives that vector no
    #     symbol. \mathbf{e}_{x_i} was verified free across the whole live tree before being used.
    ("R12-eqxi",   "Definition 2.8 names the check-in-level vector, so both sides of the chapter's "
                   "central contrast have a symbol",
     "chapters/2_fundamentals.tex",
     r"assigns one vector \$\\mathbf\{e\}_\{x_i\}\$ to each check-in", True),
    # (2) Definition 2.6 was the only task definition stated in prose alone: 2.3, 2.4 and 2.5 each give
    #     a function (g_cat(e_p) -> c_p, f_cat(H_i) -> c_i, f_reg(H_i) -> r_i) and next place did not.
    #     That made the excluded task look like a different KIND of object at exactly the point where
    #     the chapter's job is keeping the three formally distinct. Naming f_place SHARPENS the scope
    #     statement: the exclusion now applies to a defined mapping. Both halves pinned, because the
    #     equation and the exclusion can drift independently and the pair is what carries the meaning.
    ("R12-fplace", "next-place prediction is stated as a function like its three neighbors, not in "
                   "prose alone",
     "chapters/2_fundamentals.tex", r"f_\{\\mathrm\{place\}\}\(H_i\)\\longrightarrow p_i", True),
    ("R12-fplace2","and the scope exclusion survives beside it, so naming the function does not read "
                   "as claiming the task",
     "chapters/2_fundamentals.tex",
     r"no chapter reports a result\s*for \$f_\{\\mathrm\{place\}\}\$", True),
    # ---- ROUND-12, the author's rulings of 2026-08-03 on the definition redesign. These pin RECORDS,
    # not chapter prose, because the redesign is not applied yet and the records are what a later pass
    # reads. Each one is a claim that has already drifted once or could plausibly be softened by a
    # summarizer, which is the test for whether a probe earns its place.
    #
    # R12-thirteen: he ruled THIRTEEN definitions ("Vamos de treze"), so rho becomes a numbered
    # Definition. The twelve-definition fallback stays in the design as the record of the alternative,
    # which is exactly the condition under which a later reader could mistake the dead option for the
    # live one. Pin the ruling.
    ("R12-thirteen","the definition design records THIRTEEN as the author's settled choice, not an open "
                    "twelve-or-thirteen question",
     "../fundamentals/DEFINITIONS.md", r"RESOLVED 2026-08-03: THIRTEEN", True),
    # R12-streams: he confirmed F-1 and gave the mechanism -- two final embeddings of one trained graph,
    # feeding next-region and next-category respectively. The single-equation wording would misstate the
    # input of the study the whole arc resolves on, so both halves are pinned: the ruling, and the
    # two-stream statement itself.
    ("R12-streams", "Chapter 5's input is recorded as TWO elementwise streams, per his own mechanism",
     "../fundamentals/DEFINITIONS.md", r"RESOLVED 2026-08-03: NAME BOTH STREAMS", True),
    # R12-notagg: THE CODE FINDING, and the one most likely to be quietly reversed. His premise was that
    # Chapter 4's temporal channel is aggregated to POI level. The pipeline does the opposite: the
    # category-task builder REJECTS check-in-level engines (src/data/inputs/builders.py:191-192) and no
    # aggregation exists anywhere in it. A later pass that writes "aggregated to the place" into Chapter 2
    # would be writing the one claim the code refutes, so the investigation's negative finding is pinned
    # positively and the refuted phrasing is banned from the record.
    ("R12-notagg",  "the investigation records that NO check-in-to-POI aggregation exists in the Time2Vec "
                    "pipeline, so Chapter 2 may not claim one",
     "_round12/50_courb_temporal_level_investigation.md",
     r"There is no aggregation function anywhere in the\s+Time2Vec pipeline", True),
    # R12-notagg2 WAS BROKEN ON ITS FIRST WRITING, and the defect is worth recording because it is the
    # third time in this repo that a probe's SABOTAGE LEG, not the probe, decided whether it looked
    # covered. The pattern was r"^The temporal channel is aggregated to the place" -- and this gate
    # matches with re.I ONLY, never re.MULTILINE, so `^` anchors at STRING START and nowhere else. The
    # ban therefore covered exactly one position in a 5 KB file. My sabotage leg inserted the sentence
    # as the file's first line, i.e. at the single position the pattern could reach, so it FIRED and I
    # reported the ban as covering the record. Reproduced after the fact: the same sentence placed
    # mid-document did not match.
    #
    # THE FIX cannot be an unanchored bare-phrase ban, because the record legitimately QUOTES the
    # refuted wording once, in "And I cannot write that the temporal channel is aggregated to the
    # place" -- the sentence whose whole job is to refute it. Same constraint as R9-clock2: a correction
    # that cannot name what it corrects is not a correction. So the ban keys on the ASSERTION FORM at a
    # sentence boundary ANYWHERE in the file (string start, after a sentence-final punctuation, or after
    # a newline, with optional markdown emphasis), which is position-independent without also banning
    # the quotation. Validated at five positions the old pattern could not reach plus two that must stay
    # legal; the legs below sabotage MID-document, not at the top.
    #
    # SECOND MISS, caught by the legs and not by my reasoning, and it is the same lesson one level down:
    # my first corrected pattern passed a hand-built test on RAW text and then went SILENT on a real
    # sabotage leg that put the assertion in a bold run. Cause: the gate matches live_text(), whose
    # strip_text() collapses newlines to spaces, so the text preceding a bolded line arrives as
    # "...reduction.** **The temporal channel..." -- a period followed by ASTERISKS, which my
    # [.!?]['\")\]]* class did not admit. A pattern must be tested THROUGH the gate's own normalizer,
    # never against the raw file. The lead-in now admits emphasis markers and list bullets, and the
    # pattern is validated over ten positions including two that must stay legal.
    ("R12-notagg2", "the investigation does not ASSERT the aggregation it was sent to look for, at any "
                    "position in the file (quoting the refuted wording inside the refutation stays legal)",
     "_round12/50_courb_temporal_level_investigation.md",
     r"(?:\A|[.!?][*_'\")\]]*\s|\n)\s*(?:[-*+]\s+|\d+\.\s+)?(?:\*\*|__|\*)?\s*"
     r"The temporal channel is aggregated to the place", False),
    # ---- The two claims from his 2026-08-03 side-chat rulings that a later pass is most likely to
    # flatten, and both would cause real damage if flattened.
    #
    # R12-ad4cond: AD-4's title is a CONDITIONAL answer. His words: "maybe with this inversion we even
    # need this new section." If the 2.1/2.2 inversion happens, the representation definitions are
    # already in the right section and the subsubsection has no function. A summariser reading "AD-4
    # resolved: Check-in and place representation" would create the subsection, which is precisely what
    # he did NOT authorize. Pin the conditionality, not just the title.
    ("R12-ad4cond", "AD-4's title is recorded as CONDITIONAL, because the section inversion may remove "
                    "the need for the subsubsection at all",
     # PATTERN REPOINTED 2026-08-03: it was "RESOLVED CONDITIONALLY 2026-08-03", a string that vanished
     # when he revoked `place representation` and the AD-4 row had to be rewritten. The CLAIM is
     # unchanged -- AD-4 is conditional and its subsection must not be created -- so the probe is
     # repointed rather than retired, and it now pins the two halves that matter: that the item is
     # conditional, and that nothing is created either way.
     "../fundamentals/DEFINITIONS.md",
     r"AD-4 was always conditional.*?Nothing is created either way", True),
    # R12-planvoid: the eight-step edit plan assumes the representation definitions move UP into 2.1.
    # Under the inversion that assumption is void and the plan must be REDONE BEFORE any edit. Getting
    # this wrong means editing the chapter against a plan written for a different structure, which is
    # the most expensive mistake available here. Pinned in the design doc, whose §11 carries the cost
    # table.
    ("R12-planvoid","the record states that the eight-step plan does NOT survive the section inversion "
                    "and must be redone before any edit",
     "../fundamentals/DEFINITIONS.md",
     r"must be redone BEFORE any edit, not after", True),
    # ---- AD-2 ANSWERED from the original CoUrb code (temp/tarik-new), 2026-08-03. Three probes, because
    # this finding has three distinct ways of being flattened by a later pass and each would mislead
    # differently.
    #
    # R12-dropdup: the OPERATIVE WORD. The reduction is drop_duplicates -- it SELECTS one visit per POI
    # and discards the rest. "Aggregation" (mean, pooling) is the word everyone reached for, including me
    # and the author, and it is wrong: aggregation would combine the visits. A record saying "aggregated"
    # would misdescribe what ran. Pin the file path and line, so the claim stays checkable.
    # DESCRIPTION CORRECTED 2026-08-03: this said "the AD-2 answer". There is no AD-2 answer -- it was
    # retracted. The probe still earns its place because the dedup's LOCATION is an established fact and the
    # record must keep citing it checkably; what it does NOT license is the conclusion drawn from it.
    ("R12-dropdup", "the record cites the placeid dedup at its source line, as an established fact and not "
                    "as evidence of a check-in-to-POI selection step",
     "_round12/50_courb_temporal_level_investigation.md",
     r"create_inputs_hgi\.py:437", True),
    # R12-shape: the EVIDENCE, and it is the strongest kind available here -- a stored notebook output
    # shape rather than a reading of code intent. 2535573 rows against 2535573 check-ins settles the
    # granularity numerically. If this number goes, the finding reverts to an argument about intent.
    ("R12-shape",  "and it carries the stored output shape that establishes the ENCODER's per-check-in "
                   "granularity numerically (which survives the retraction; what did not survive is that "
                   "this matrix is what the ETL consumes)",
     "_round12/50_courb_temporal_level_investigation.md",
     r"\(2535573, 64\)", True),
    # R12-notwrong WAS REPLACED, 2026-08-03. It pinned "description, not a wrong number" -- the calibration
    # of a conclusion that has since been RETRACTED, so the probe was enforcing a withdrawn claim and went
    # correctly red when the record was corrected. A probe on a retracted conclusion is worse than no probe:
    # it fights the correction. What earns a probe now is the RETRACTION, because the tempting error for a
    # later pass is to re-derive the same conclusion from the same two facts and skip the link again.
    ("R12-retract", "the AD-2 investigation records that its own 'answered' conclusion was RETRACTED, and "
                    "why: the ETL reads a parquet nothing in that repository writes",
     "_round12/50_courb_temporal_level_investigation.md",
     r"RETRACTED AND REOPENED", True),
    ("R12-retract2","and it does not assert that AD-2 is answered",
     "_round12/50_courb_temporal_level_investigation.md",
     r"(?:\A|[.!?][*_'\")\]]*\s|\n)\s*(?:[-*+]\s+|\d+\.\s+)?(?:\*\*|__|\*)?\s*"
     r"AD-2 is (?:therefore )?(?:now )?answered", False),
    ("R12-verify",  "and it carries the VERIFY flag naming the one artifact that would close AD-2",
     "_round12/50_courb_temporal_level_investigation.md",
     r"\[VERIFY: the granularity of time_embedding\.parquet", True),
    # ---- His closing rulings of 2026-08-03. Two probes on the LEFT_OUT entry, because that entry is now the
    # DURABLE home of a finding whose earlier version I had to retract, and the two ways it can rot are the
    # two ways the retraction can be undone.
    #
    # R12-lo12cond: the visits-per-POI table in LO-12 quantifies a HYPOTHETICAL -- what the dedup would
    # discard IF the input were per-visit. It is not evidence of granularity. A later reader finding real
    # numbers in a register entry will be tempted to read them as the finding; the sentence that forbids that
    # reading is the load-bearing part of the entry, not the table.
    ("R12-lo12cond","LO-12 states that its visits-per-POI table quantifies a hypothetical rather than an "
                    "established fact",
     "LEFT_OUT.md", r"This quantifies a hypothetical, not a fact", True),
    # R12-lo12clue: the CBIC errata line about the category task's sample unit is ADJACENT and NOT decisive
    # (different study, different encoder, and consistent with both hypotheses). It is exactly the shape of
    # the unverified link that produced the retraction, so the entry names it AND names why it does not
    # settle anything. Losing that qualification would hand a later pass a ready-made false proof.
    ("R12-lo12clue","and it records the adjacent CBIC errata clue as NOT deciding the question",
     "LEFT_OUT.md", r"One adjacent clue that does NOT decide it", True),
    # R12-ad2row: the AD-2 row in the DESIGN document. This exists because the retracted framing SURVIVED
    # in that row for a full commit. MECHANISM CORRECTED 2026-08-03 -- this comment first blamed a stale byte
    # offset whose .replace() matched nothing, which was a mechanism I supplied from memory rather than read
    # off the cells. What actually happened: the .replace() SUCCEEDED in memory, but the same cell hit an
    # `assert` (computed for a different edit) that RAISED before its write_text, so nothing was written at
    # all; a later cell re-read the file from disk and discarded the corrected string. The "verified" print
    # reported the new text as present because I had checked for a substring the new text contains -- a
    # presence check on the replacement is not a check that the replacement happened. The distinction matters
    # for the rule: re-reading before computing offsets would NOT have prevented this, whereas asserting the
    # OLD string is ABSENT would. Full account in _round9/34_tracker_disagreement.md. Ban the retracted
    # framing by its two most quotable phrases, positioned anywhere.
    ("R12-ad2row",  "the design's AD-2 row does not carry the retracted 'fourth possibility' framing",
     "../fundamentals/DEFINITIONS.md", r"FOURTH possibility none of us had listed", False),
    ("R12-ad2row2", "nor the retracted claim that the first visit survives and the rest are discarded",
     "../fundamentals/DEFINITIONS.md",
     r"keeping the first visit to each POI and discarding the rest", False),
    # R12-placeterm: the author revoked `place representation` from the registry, which left ONE live line of
    # the chapter using a term absent from a fail-closed registry (2_fundamentals.tex:650, against eight live
    # uses of `place embedding`). Fixed to the registered term. Pinned because nothing else gates registry
    # conformance -- there is no term-sweep checker in this repo, as recorded above.
    ("R12-placeterm","the chapter does not use the revoked `place representation` in live prose",
     "chapters/2_fundamentals.tex", r"from a place representation with next-category", False),
    # R12-d36: the THIRD misattributed commit, and the first whose defect was an aborted cell rather than a
    # staging mistake. d36da8c5 shipped a DEFINITIONS.md whose header says "AD-2 is OPEN / the AD-2 row
    # carries the retraction" while the row itself still said ANSWERED. Recorded rather than rewritten
    # (shared checkout, successor commits). Pinned because this class is invisible to every other gate: they
    # all read the working tree and none compares a commit message against its own diff.
    ("R12-d36",     "the record carries the third commit-attribution defect, where a header asserted a "
                    "correction the table row below it did not carry",
     "_round12/51_commit_attribution_correction.md",
     r"shipped a header that contradicts its own table row", True),
    ("R12-d36why",  "and it names the aborted-cell mechanism plus the negative assertion that would have "
                    "caught it",
     "_round12/51_commit_attribution_correction.md",
     r"cannot detect that one specific replacement never happened", True),
    # R12-study: the inversion study's LOAD-BEARING finding. The author asked for it specifically, and the
    # answer is a negative: the frozen planning folder records NO argument for tasks-before-representations,
    # so the order was inherited from the chapter map rather than defended. A negative finding is the kind
    # that quietly becomes "we looked and found a reason" on a later retelling, which is why it is pinned.
    # Spot-checked by me against the live tree, not taken from the sub-agent's self-report: the three
    # cross-references (2_fundamentals.tex:15, :17, :197), the two labels (:25, :210), zero references
    # outside the chapter, DEFINITIONS.md:613, and NORTH_STAR.md:73-75 all read as the study reports them.
    ("R12-study",   "the inversion study records that NO recorded rationale for the tasks-first order "
                    "exists in the frozen planning folder",
     "_round12/52_inversion_study.md",
     r"records NO argument for placing tasks before representations", True),
    ("R12-studyrec","and its recommendation is marked as a recommendation the author decides, not an "
                    "authorization",
     "_round12/52_inversion_study.md",
     r"this is a recommendation; the author decides, and nothing is authorized", True),
    # R12-extra: `make extra` is RED and it is not the document -- BSD sed aborts on a Latin-1 byte in the
    # build log, so the page-count extraction fails while the PDF builds fine. Recorded as his decision
    # because latexbuild.sh is shared with a parallel agent. Pinned so the red target is not later rediscovered
    # as a mystery, and so "all four builds rc=0" is not written again while it stands.
    ("R12-extra",   "the record carries the diagnosis of the red `extra` target as a sed locale failure "
                    "rather than a document defect",
     "PENDENCIAS.md", r"illegal byte sequence", True),
    # R12-studyfix: the sub-agent's study asserted "git history begins 2026-07-23" to support its central
    # negative finding. FALSE about the repository -- root commit 2025-03-08, 2049 commits, 1666 of them
    # earlier. I checked instead of accepting the self-report, and the FINDING survives on a better
    # measurement: of those 1666 earlier commits exactly ONE touches articles/dissertacao/ (bb4449c8, a
    # .gitignore). Pinned because a correction that lives only in a [VERIFY] list is the kind a later pass
    # skips, and because the wrong reason would otherwise be quotable as support for a right conclusion.
    ("R12-studyfix","the inversion study's false 'git history begins' claim is corrected in place, with the "
                    "measurement that actually supports the finding",
     "_round12/52_inversion_study.md",
     r"It stands on a measurement of what the early commits CONTAIN", True),
    # R12-mech: THE RECORD MUST CARRY ONE MECHANISM FOR THE SURVIVING-ROW DEFECT, NOT TWO. For a while it
    # carried two incompatible ones -- a stale byte offset whose replace missed (invented, written into
    # _round9/34 and into the R12-ad2row rationale) and an assert that raised before write_text (measured,
    # written into _round12/51). They are not variants: under one the replace ran and missed, under the other
    # it succeeded and the write never happened, and only the second is what the cells show. The derived RULE
    # differs too, which is why a contradiction here is not cosmetic: re-reading before computing offsets
    # would not have prevented the real defect. Pinned on 34 because that is the file a later pass reads for
    # the lesson.
    ("R12-mech",    "the postmortem names the assert-before-write mechanism and retracts the invented "
                    "stale-offset one",
     "_round9/34_tracker_disagreement.md",
     r"the mechanism I gave for\s+it was invented", True),
    ("R12-mech2",   "and it does not assert the stale-offset story as the cause",
     "_round9/34_tracker_disagreement.md",
     r"The offsets had been computed against an \*earlier\* copy", False),
    # ---- His rulings of 2026-08-03 (second batch).
    #
    # R12-locale: the LC_ALL=C fix on latexbuild.sh, his option 2. TWO probes because the two lines have
    # DIFFERENT standing and conflating them is the specific error he asked me to prevent: :PAGES was a real
    # bug (BSD sed aborted on a Latin-1 byte, so `make extra` reported a failed build that had in fact
    # produced a correct 26-page PDF), while :ERRS is hygiene -- I MEASURED that `grep -c '^! '` returns 0 in
    # both locales on that log before adding it. A later reader who takes :ERRS for a bug fix would conclude
    # the tex_errors counts had been wrong all along, which they were not.
    ("R12-locale",  "both extraction lines in latexbuild.sh run under LC_ALL=C",
     "../src_utils/latexbuild.sh", r"PAGES=\$\(LC_ALL=C sed", True),
    ("R12-locale2", "and the script says in its own comment that the ERRS line is hygiene rather than a bug "
                    "fix, with the measurement that establishes it",
     "../src_utils/latexbuild.sh", r"HYGIENE AND SYMMETRY, NOT A BUG FIX", True),
    # R12-dscope: the `d` registry row. `d` is NOT a free letter -- d_{ij} is geodesic distance and
    # d_{shared} is the shared-trunk width, both live. The row without its scope note would license reading
    # either as an instance of the representation dimension. NOTE the correction of record: the pending note
    # said d_shared lives in Chapter 5; MEASURED, it is in Chapters 3 and 4 (3_cbic/method.tex:78,
    # 4_courb/methodology.tex:25,:258) and Chapter 5 has none. The row states the measured chapters.
    ("R12-dscope",  "the new `d` registry row carries its scope note naming both live subscripted uses",
     "../GLOSSARY.md", r"\*\*This letter is NOT free, and the row is scoped for that reason\.\*\*", True),
    ("R12-rho",     "and the rho row records that varying rho while holding the task definitions fixed is "
                    "what makes the central claim expressible",
     # PATTERN NOTE: the target is the LaTeX `$\rho$`, so in this raw-string regex the backslash is escaped
     # ONCE (\\rho), not twice. My first attempt used \\\\rho and the probe went red against text that
     # was correctly present -- a pattern defect reported as a missing claim.
     "../GLOSSARY.md", r"hold the task definitions fixed and vary \$\\rho\$", True),
    # R12-cmp: the comparative study reversed the previous recommendation (52 said invert, 53 says keep), and
    # the reversal turns on ONE argument the author must be able to find: the answer-mirror only pays a reader
    # who already holds the answer, so the design-mirror serves the FIRST reading the project's own G3 gate
    # governs. If that sentence is ever smoothed away, the record loses why the two studies disagree and the
    # 52 recommendation reads as unopposed.
    ("R12-cmp",     "the comparison records the reader-side resolution of the mirror argument, which is why "
                    "it reverses 52's recommendation",
     "_round12/53_order_comparison.md",
     r"scaffold for a first reading; the answer-mirror is a reward for a second", True),
    # R12-cmpval: and it must keep saying that the "already-validated" label was 52's coinage rather than 49's
    # finding, with the concrete qualifier -- AD-4, the sign-off on the section shape, is OPEN. I verified this
    # independently: grep for "validated" in 49 returns zero occurrences of that claim about the option.
    ("R12-cmpval",  "and it records that option (a) is mechanically checked but narratively unsigned, because "
                    "AD-4 is still open",
     "_round12/53_order_comparison.md",
     r"mechanically checked and narratively unsigned", True),
    # R12-clock3: I published "2,236 s, 7 percent inside the checkpoint" for the comparative study and it was
    # wrong IN KIND, not by a margin: 2,236 s is when MY collect window closed with the child still reported
    # `running` and every structured field empty. I read a timeout as a completion. Measured, the child was
    # still processing at 3,201 s -- 33 percent OVER the 2,400 s checkpoint. This is the SECOND time this
    # project has published a false in-budget claim about a sub-agent (see R9-clock/R9-clock2 for the first),
    # and both times the true reading was already in hand and unread. Ban the retracted figure as an
    # assertion, anywhere in the record, while allowing the sentences that quote it as retracted.
    ("R12-clock3",  "the study record carries the parent's real measurement, that the child's lifetime "
                    "overran the checkpoint by 74 percent",
     "_round12/53_order_comparison.md",
     # FIGURE CORRECTED: this probe first pinned "33 percent OVER", which was my second wrong reading of the
     # same clock (3,201 s was the child's AGE when I looked, not its lifetime). The frame record gives
     # 4,185 s, 74 percent over. Pinning the real figure, not the intermediate one.
     r"4,185 s, 74 percent OVER the 2,400 s checkpoint", True),
    ("R12-clock4",  "and PENDENCIAS does not assert the retracted in-budget reading of the collect timeout",
     "PENDENCIAS.md",
     r"(?:\A|[.!?][*_'\")\]]*\s|\n)\s*(?:[-*+]\s+|\d+\.\s+)?(?:\*\*|__|\*)?\s*"
     r"(?:Medido|Wall-clock, medido por mim):?\**\s*2\.236 s contra o checkpoint de 2\.400 s\s*—\s*7 por cento "
     r"DENTRO", False),
    # R12-clock5: THREE readings of one clock, two of them wrong, and both wrong ones were published. 2,236 s
    # was my collect timeout; 3,201 s was the child's age at an arbitrary glance; 4,185 s is its lifetime from
    # the frame record. The pattern is identical each time -- an instrument reading taken for a measurement of
    # the process -- so ban the superseded figure in assertion form while allowing the sentences that retract
    # it. The 2,236 ban lives in R12-clock4 for PENDENCIAS; this one covers the study file for both.
    ("R12-clock5",  "the study does not assert either superseded overrun figure as the measurement",
     "_round12/53_order_comparison.md",
     r"(?:\A|[.!?][*_'\")\]]*\s|\n)\s*(?:[-*+>]\s+)?(?:\*\*)?\s*(?:Measured[^.\n]{0,40})?"
     r"(?:33 percent OVER the 2,400|7 percent inside the 2,400)", False),
    # ---- ROUND-12 APPLICATION of the _round12/49 Part B plan, under the author's option-(a) ruling of
    # 2026-08-03 (keep 2.1 tasks, 2.2 representations; move the representation definitions UP into 2.1).
    # These are the first probes in this project that pin CHAPTER prose produced by the redesign, so each
    # names the property that would be silently lost rather than the sentence that carries it.
    ("R12-s1bind",  "STEP 1: the notation prose BINDS the per-POI attributes, which the static-classifier "
                    "equation at :119 had been consuming undefined",
     "chapters/2_fundamentals.tex",
     r"Each POI \$p\\in\\mathcal\{P\}\$ carries a\s+category \$c_p\\in\\mathcal\{C\}\$ and lies in a "
     r"region \$r_p\\in\\mathcal\{R\}\$", True),
    ("R12-s2type",  "STEP 2: the check-in definition TYPES its category and region as the visited POI's "
                    "attributes, which is what makes the place-level vector well typed",
     "chapters/2_fundamentals.tex", r"\$c_i=c_\{p_i\}\$ is its category, and \$r_i=r_\{p_i\}\$", True),
    # R12-s3head: the author chose this head himself on 2026-08-03, from two registered candidates, after
    # revoking the term that made his earlier head inadmissible. It must not drift back to a variant, and
    # in particular not to "place representation", which is NOT in the registry.
    ("R12-s3head", "STEP 3: the new subsubsection carries the head the AUTHOR named",
     "chapters/2_fundamentals.tex", r"\\subsubsection\{Check-in and place embedding\}", True),
    ("R12-s3map",   "STEP 3: the representation map is a NUMBERED definition (AD-1, thirteen), not a "
                    "displayed equation in prose",
     "chapters/2_fundamentals.tex",
     r"\\begin\{definition\}\[Representation map\]\\label\{def:fund:repmap\}", True),
    # R12-s3scope: defect F-2 of _round12/49. The remark must stay scoped to the SEQUENTIAL tasks: the
    # static task reads the place embedding directly, so "every predictive model" was false of it. This
    # pins the scoping word, which is the whole correction.
    ("R12-s3scope", "STEP 3: the factorization remark is scoped to the sequential tasks",
     "chapters/2_fundamentals.tex",
     r"Every model of the sequential tasks in this dissertation reads \$\\rho\(H_i\)\$", True),
    # R12-s4moved: the MOVE is only real if the definitions are GONE from 2.2. A probe on their presence in
    # 2.1 would hold just as well if they were duplicated, which is exactly the failure mode the plan warns
    # about (two rendered definitions, one number sequence). This asserts the absence.
    ("R12-s4moved", "STEP 4: the two definition environments are GONE from 2.2, so the move is a move "
                    "rather than a duplication",
     "chapters/2_fundamentals.tex",
     r"\\subsubsection\{Representations of a check-in\}|"
     r"limitation for this research[\s\S]{0,400}\\begin\{definition\}\[Place embedding\]", False),
    # R12-s5neutral: AD-2 option 2. The author ruled that Chapter 2 states Chapter 4's input in the NEUTRAL
    # form and that the level question is documented as LO-12 rather than recorded as an erratum. Both
    # forbidden wordings are banned by R12-neutral already; this pins the positive form in 2.2 so it cannot
    # revert to the "place-level input" claim the retracted investigation had made.
    ("R12-s5neutral","STEP 5: 2.2 states Chapter 4's input in the neutral form, matching the 2.1 remark",
     "chapters/2_fundamentals.tex", r"while its input stays a function of the visited POI", True),
    # R12-wise: the -wise suffix is excluded from the British-spelling sweep BY RULE, not by enumeration.
    # "elementwise" (introduced by step 3) false-fired and turned `make check` red on correct American
    # prose; the module's own header warns that a hand-typed list matches only the words its author thought
    # of, and this was that defect. Pinned because a later "simplification" back to enumeration would
    # reintroduce it for the next compound.
    ("R12-wise",    "the register gate excludes the -wise family by rule rather than by listing words",
     "../src_utils/check_register.py", r"if low\.endswith\(\"wise\"\) and len\(low\) > 4:", True),
    # R12-attrib: a commit-message attribution defect, recorded because the suite CANNOT detect this class
    # -- every gate here reads the working tree and none reads the commit log. Two of round 12's commits
    # describe diffs they do not contain, because `git add -A` in a backgrounded cell staged the tree at
    # execution time rather than dispatch time. Pinning the record is the only enforcement available.
    ("R12-attrib", "the commit-attribution defect is recorded, with the rule that a commit message is a "
                   "claim about a diff and that no gate here can check one",
     "_round12/51_commit_attribution_correction.md",
     # SECOND BRANCH CORRECTED: it was "never reads the commit log", which occurs ZERO times in the
     # target -- the file says "read the working tree and never the commit log". A dead alternation
     # branch makes a probe look broader than it is; caught by a sabotage leg reporting TARGET ABSENT.
     r"no gate in `check\.sh` can detect it|never the commit log", True),
    # ---- His four rulings of 2026-08-03 (side chat). Three probes: the two whose flattening would cause
    # real damage, plus the registry scope note.
    #
    # R12-neutral: HIS RULING WAS "do not register", which is SILENCE, not licence to write the wrong
    # thing. The Chapter 4 instantiation must say only "a function of the visited POI"; both "of the
    # visit's timestamp" (false: it is ONE SELECTED visit) and "aggregated" (false: the operation selects)
    # stay forbidden. An edit pass reading "not recorded, implementation detail" could easily reintroduce
    # a wrong qualification, which is exactly the failure this pins shut.
    ("R12-neutral", "the design records that his 'do not register' ruling is silence and does NOT license "
                    "the two wordings the code refutes",
     "../fundamentals/DEFINITIONS.md",
     r"not a licence to write the wrong thing", True),
    # R12-ad7: AD-7 renames D13's indices from i,j to a,b so that i means a check-in everywhere in the
    # chapter. Pinned on the DESIGN, not the chapter, because the chapter edit lands with the redesign --
    # editing it earlier would desynchronize the two during the pending window. The rename departs from
    # the source's own notation (yu2020pcgrad uses i,j for tasks), so the design must keep saying WHY.
    ("R12-ad7",     "the design carries D13's renamed task indices and the reason for departing from the "
                    "source's notation",
     "../fundamentals/DEFINITIONS.md",
     r"departs from the source's notation deliberately", True),
    # R12-placerep WAS REMOVED, 2026-08-03, in the same commit that revoked the row it pinned. It required
    # the scope note on the `place representation` registry row. The author then revoked the row itself
    # ("vamos usar so place embedding para evitar conflitos e interpretacoes dubias"), so the note went with
    # it and the probe would have gone red ON HIS DECISION. That is the R12-notwrong defect exactly: a gate
    # enforcing a claim the author has since withdrawn fights the correction instead of protecting it.
    # A REPLACEMENT WOULD BE WRONG TOO, and it is worth saying why rather than leaving a gap that looks like
    # an oversight. The obvious candidate -- ban `place representation` from the tree -- fails on its own
    # terms: this file, PENDENCIAS, and the round-12 records all discuss the term by name while recording the
    # decision, so an absence probe would fire on the record OF the decision. The registry is already
    # fail-closed by POLICY (WRITING_LAW section 5, GLOSSARY's own preamble): a term absent from the
    # registry may not be used. Note precisely what that means mechanically, because I first wrote that
    # a `check_glossary_terms` gate handles it and NO SUCH FILE EXISTS -- the checkers are the fourteen
    # check_*.py in this directory and only this one and check_verify_list.py read GLOSSARY.md at all.
    # So registry conformance is enforced by review and by targeted probes here, not by a term sweep,
    # and the honest statement is that revoking this row leaves the term ungated rather than covered
    # elsewhere. Nothing replaces it, and that is a known gap and not an oversight.
)

# COD-016b needs a STRUCTURAL probe, not a string one, so it lives here rather than in PROBES --
# and it was MISSING FROM BOTH LISTS until a reviewer noticed, which made the docstring's claim to
# re-measure every APPLIED row false by omission. Exactly the defect this file exists to catch,
# in this file. Fixed by adding the probe, not by narrowing the claim.
#
# A TRAP WORTH THE PARAGRAPH: Chapter 5's setup section holds TWO long paragraphs, and they belong
# to DIFFERENT findings. COD-006 is the PROTOCOL paragraph ("A claimed gain and a claimed match..."),
# 2,110 characters, which the author did NOT ask to be split and which is correctly still one
# paragraph. COD-016b is the INTEGRITY paragraph (the four numbered fundamentals, "First, its
# training objective is label-free..."), which he DID approve breaking. Measuring the first one and
# reading its single-paragraph state as a failed split produced a false alarm here on 2026-07-30 --
# anchor on "First, its training objective", never on paragraph length alone.
INTEGRITY_ANCHOR = "First, its training objective is label-free"


def integrity_paragraph_probe() -> tuple[bool, str]:
    """COD-016b: the ~580-word integrity block must be several paragraphs, no word changed."""
    path = SRC / "chapters/5_mobiwac/05_setup.tex"
    if not path.exists():
        return False, "05_setup.tex not found"
    raw = path.read_text(encoding="utf-8", errors="replace")
    j = raw.find(INTEGRITY_ANCHOR)
    if j < 0:
        return False, f"anchor absent: {INTEGRITY_ANCHOR!r} -- the block was reworded or removed"
    start = raw.rfind("\n\n", 0, j)
    m = re.search(r"\\(sub)*section\{", raw[j:])
    seg = raw[start : j + m.start()] if m else raw[start : j + 6000]
    live = "\n".join(l for l in seg.split("\n") if not l.lstrip().startswith("%"))
    paras = [re.sub(r"\s+", " ", q).strip() for q in re.split(r"\n\s*\n", live) if q.strip()]
    longest = max((len(q) for q in paras), default=0)
    ok = len(paras) >= 4
    return ok, f"{len(paras)} paragraph(s), longest {longest} chars (was 1 of ~3,900)"

# Probes retired because the AUTHOR withdrew the underlying instruction. These are NOT passes and
# NOT failures: the finding no longer describes anything the document is supposed to contain.
#
# They are printed anyway, every run. A probe deleted in silence shrinks the gate's scope without
# telling anyone, and this suite's prose scope has silently shrunk twice already (check.sh, the
# chapters/*/*.tex and preamble.tex cases). The reason is quoted from the author verbatim so a later
# reader can tell "he decided against it" from "somebody dropped it", which is the exact distinction
# LEFT_OUT.md exists to preserve.
RETIRED: dict[str, str] = {
    "COD-018": (
        "per-role CoUrb credit in Appendix A -- WITHDRAWN by the author in PENDENCIAS.md 5.8: "
        '"Nao precisa mexer nisso, pode remover essa preocupacao." Reconfirmed in session '
        "2026-07-30 when the round-8 brief asked for it anyway; a credit claim about a "
        "co-authored paper is his alone to make (GUARDRAILS C2), so the probe goes rather than "
        "the decision. Recorded in LEFT_OUT.md LO-11."
    ),
}

# Claims whose subject is a PROCESS, not a string in the source. Listed by name so the report
# covers them; asserting them mechanically would require re-running the process itself.
NOT_CHECKABLE = {
    "COD-001": "resolved-and-stayed-resolved: covered by check_trapped_prose instead",
    "COD-008": "every citation audited at its source of record -- a process, see the source ledger",
    "COD-009": "the L5 translation-fidelity gate ran -- a process",
    "COD-012": "UFV submission gate -- covered by sync_page_counts and the numbering check",
    "COD-017": "figure type size -- an author decision, PENDENCIAS 2.5",
}


def strip_text(raw: str) -> str:
    """The stripper itself, on a string, so it can be self-tested without a file on disk.

    Split out from live_text on 2026-07-30 (COD-006a): the self-test used to assert that the
    string "well powered" was PRESENT in the live Chapter 5 source, because at the time that
    was the escaped-percent case it needed. Applying the fix that probe exists to check would
    therefore have crashed the gate on an AssertionError, which is a self-test that fails when
    the document becomes correct. The property being proved is about the stripper, not about
    any one sentence, so it is now proved on literals that cannot drift.
    """
    keep = []
    for line in raw.split("\n"):
        m = COMMENT.search(line)
        cut = line[: m.start()] if m else line
        if cut.strip():
            keep.append(cut)
    return re.sub(r"\s+", " ", " ".join(keep))


def live_text(path: Path) -> str:
    """Source with comments removed, joined into one whitespace-normalized string.

    Joined on purpose: a claim whose numbers wrap to the next line is invisible to a per-line
    regex, and that produced a false NOT-APPLIED on NUM-4. See trap 2 in the module docstring.
    """
    return strip_text(path.read_text(encoding="utf-8", errors="replace"))


def self_test() -> None:
    """Both directions on the stripper, against the two real cases that fooled it.

    On LITERALS, not on live sentences. Both literals are reductions of the actual defects:
    the escaped-percent case is the shape of 5_mobiwac/05_setup.tex line 94, where `90\\%` at
    column 763 truncated a 2,068-character paragraph and hid the target clause at column 1848;
    the comment case is the shape of apx_c_ai_disclosure.tex, whose two "Opus" mentions are
    both inside `%` comments saying why it is NOT in the prose.
    """
    # Direction 1: an ESCAPED percent is not a comment. Everything after it must survive.
    esc = strip_text(r"a 90\% interval and then THE_TAIL_TEXT after it")
    assert "THE_TAIL_TEXT" in esc, (
        "self-test: an escaped \\% truncated the line and hid the text after it -- the comment "
        "pattern must be (?<!\\\\)% . Reporting now would turn a present defect into a pass."
    )
    assert r"90\%" in esc, "self-test: the escaped percent itself was eaten"
    # Direction 2: a REAL comment must be excluded, including an indented one and a trailing one.
    com = strip_text("prose survives\n   % INDENTED_COMMENT_TEXT\ncode % TRAILING_COMMENT_TEXT")
    assert "INDENTED_COMMENT_TEXT" not in com and "TRAILING_COMMENT_TEXT" not in com, (
        "self-test: a real % comment leaked into the live text -- every probe would then read "
        "provenance commentary as prose, which is how COD-013 was first scored as APPLIED."
    )
    assert "prose survives" in com and "code" in com, "self-test: the stripper over-stripped"
    # Direction 3: the stripper must still reach PAST the real escaped percent in the real file.
    # Anchored on a citation key rather than on prose, because a key is guarded by the
    # undefined-citation gate and so cannot be silently reworded the way a sentence can.
    esc_file = SRC / "chapters/5_mobiwac/05_setup.tex"
    if esc_file.exists():
        live = live_text(esc_file)
        assert "lakens2017tost" in live, (
            "self-test: the stripper no longer reaches the TOST citation, which sits 766 "
            "characters past the escaped percent on the same source line. Every probe on this "
            "file would be reading a truncated paragraph."
        )
    # Direction 4 (round 9): the SECOND root must actually resolve, and the stripper must not eat
    # markdown. A probe whose file is read from the wrong root, or whose text the stripper mangles,
    # reports exactly like a probe that never fires -- which is trap 4 in the module docstring,
    # one root over. Anchored on a heading the schema cannot lose without the split being gone.
    assert probe_root("CONSIDERATIONS.md") == UTILS, "self-test: .md probes must resolve under src_utils/"
    assert probe_root("chapters/2_fundamentals.tex") == SRC, "self-test: .tex probes must resolve under src/"
    trk = UTILS / "CONSIDERATIONS.md"
    if trk.exists():
        live_md = live_text(trk)
        assert "### FAB-01" in live_md and "### GER-01" in live_md, (
            "self-test: the tracker's item headings do not survive live_text(), so every R9- probe "
            "would be measuring mangled text. Check the comment stripper against markdown."
        )


def main() -> int:
    self_test()
    bad, missing_files = [], []
    print("== audit APPLIED claims re-measured against the live source ==")
    for fid, what, rel, pat, want in PROBES:
        path = probe_root(rel) / rel
        if not path.exists():
            missing_files.append((fid, rel))
            print(f"  SKIP        {fid:9s} {rel} not found -- probe cannot run")
            continue
        found = bool(re.search(pat, live_text(path), re.I))
        ok = found == want
        if not ok:
            bad.append(fid)
        print(f"  {'holds' if ok else 'NOT APPLIED':11s} {fid:9s} {what}")
    ok, detail = integrity_paragraph_probe()
    if not ok:
        bad.append("COD-016b")
    print(f"  {'holds' if ok else 'NOT APPLIED':11s} {'COD-016b':9s} "
          f"the integrity block is several paragraphs: {detail}")

    for fid, why in sorted(NOT_CHECKABLE.items()):
        print(f"  unprobed    {fid:9s} {why}")
    for fid, why in sorted(RETIRED.items()):
        print(f"  RETIRED     {fid:9s} {why}")

    # V13, applied to this file's own report: a total must reconcile with the rows above it. The
    # count was 8-of-8 for one turn after COD-016b's structural probe was added, i.e. a headline
    # that did not count a row it had just printed -- the exact arithmetic defect V13 names.
    total = len(PROBES) + 1  # string probes, plus the structural COD-016b probe
    held = total - len(bad) - len(missing_files)
    print(f"\n  {held} of {total} probes hold; "
          f"{len(bad)} claim(s) not applied; {len(NOT_CHECKABLE)} process claim(s) unprobed; "
          f"{len(RETIRED)} withdrawn by the author")
    print(f"  ({total} = {len(PROBES)} string + 1 structural; rows printed above must equal "
          f"{total + len(NOT_CHECKABLE) + len(RETIRED)})")
    if missing_files:
        print("  A probe whose file is gone is NOT a pass. Re-point it or retire it deliberately.")
        return 2
    if bad:
        print(f"\nFAIL: {', '.join(bad)} are recorded as APPLIED and are not in the document.")
        print("  Do not re-mark them from a report. Fix the source, then re-run this.")
        return 1
    print("\nOK: every mechanically checkable audit claim is true of the live source.")
    return 0


# ---------------------------------------------------------------------------------------------
# THE CLOSED-ITEM REGISTER was audited for this same defect on 2026-07-30, not trusted:
# _archive/PENDENCIAS_RESOLVIDOS.md carries 16 closed items with 40 commit citations, the identical
# claim shape that failed in CODEX_AUDIT. Its most exposed row (item 1.2, nine checkable artifacts)
# holds for five rows; the other four do not reproduce and are recorded as PENDENCIAS_RESOLVIDOS 2.19 (arquivado 2026-08-02) with the
# reason -- they are STALE, not false: taken in round 6 against a tree that has since gained an
# appendix and lost 0_main.tex, and none records the tree state it was taken against.
# Full audit in _round8/29_pendencias_detail.md. Do not redo it; extend 2.19 if a number moves.
if __name__ == "__main__":
    sys.exit(main())
