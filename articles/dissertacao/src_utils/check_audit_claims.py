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
     "chapters/2_fundamentals.tex", r"0\.8186", True),
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
    ("R9-pend28", "the old 2.8 no longer asks for a decision -- it records what was done",
     "PENDENCIAS.md", r"2\.8 `CONSIDERATIONS\.md` — EXECUTADO nesta rodada", True),
    # ---- ROUND-9, THE PARETO TRACK (PENDENCIAS 2.12, author's decision "DESICAO: A."). The fix has
    # two halves that can rot independently -- the §2.3 passage can be reverted, and the glossary rows
    # it depends on can be dropped -- so it gets a probe on each. Source ledger with the page of every
    # definition, and the sabotage runs: _round9/31_pareto.md.
    ("R9-pareto", "Ch.2 defines Pareto optimality and states that this dissertation does not claim it",
     "chapters/2_fundamentals.tex", r"claims no Pareto property of any kind", True),
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
     "chapters/2_fundamentals.tex", r"a relation named\s+Pareto dominance", True),
    ("R9-pareto3", "Ch.2 still defines Pareto optimality from dominance",
     "chapters/2_fundamentals.tex", r"no other dominates is Pareto optimal", True),
    ("R9-conflict", "Ch.2 defines gradient conflict as the cosine between per-task gradients, so "
                    "Appendix F's orthogonality result has a definition to point back to",
     "chapters/2_fundamentals.tex", r"cosine of the angle between two tasks", True),
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
    # PENDENCIAS 2.12 records). A .md path resolves against UTILS, and GLOSSARY.md is a level above it,
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
    ("R9-agree",  "PENDENCIAS 2.8 carries the CORRECTED stale count, agreeing with its own §6 and "
                  "with CONSIDERATIONS.md (9 of 41, not the superseded 10)",
     "PENDENCIAS.md", r"\*\*32 sao exatas e 9 estao obsoletas\*\*", True),
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
# holds for five rows; the other four do not reproduce and are recorded as PENDENCIAS 2.19 with the
# reason -- they are STALE, not false: taken in round 6 against a tree that has since gained an
# appendix and lost 0_main.tex, and none records the tree state it was taken against.
# Full audit in _round8/29_pendencias_detail.md. Do not redo it; extend 2.19 if a number moves.
if __name__ == "__main__":
    sys.exit(main())
