#!/usr/bin/env python3
"""check_audit_claims.py -- every "APPLIED" claim in an audit is re-measured against the live source.

WHY THIS EXISTS
===============
On 2026-07-28 the round-6 outcome table in CODEX_AUDIT.md was written with sixteen rows reading
**APPLIED**. On 2026-07-30 the author read PENDENCIAS.md and found that many of those fixes were
never in the document. Re-measured here: of the nine instructions he had given, EIGHT were still
unapplied, and five of the eight sat under a row asserting they were done. COD-006's row said
'"before any result was read" and "well powered" removed' -- both strings were still in
5_mobiwac/05_setup.tex, in both the dissertation and the submitted-paper tree.

The cause is not that anyone lied. Round 6 ran eight parallel tracks; a track reported what it
INTENDED, the outcome table recorded the report, and nothing ever re-read the source. An audit
outcome table is a CLAIM ABOUT THE WORK, which is the highest-risk statement class in this
repository (AGENT_GUARDRAILS §4b), and it was the one class with no gate.

So each finding here carries a MACHINE-CHECKABLE probe: a string that must be absent because it was
removed, or present because it was added. If a probe cannot be written, the finding is listed as
NOT MECHANICALLY CHECKABLE rather than assumed -- an unreported gap is how this survived.

THREE TRAPS THIS FILE HIT WHILE BEING WRITTEN, all in the "measure the source" step, all cheap to
repeat and expensive to notice:

  1. A COMMENT-BLIND grep counts provenance comments as prose. Appendix C mentions "Opus" twice --
     both inside `%` comments explaining why it is NOT named in the text. A plain `grep -c opus`
     therefore reported the fix as DONE when the reader sees nothing.
  2. A LINE-BASED probe misses a claim that wraps. NUM-4's numbers sit two lines below the sentence
     that introduces them, so a per-line regex found nothing and reported a correctly-applied fix
     as missing. Comments must be stripped and the file joined into one string.
  3. AN ESCAPED PERCENT IS NOT A COMMENT. `90\\%` inside a sentence truncated a 2,068-character
     paragraph at column 766, hiding "well powered" at column 1848 -- so the fixed stripper
     reported the defect as absent. The comment pattern must be `(?<!\\)%`.
SCOPE, WIDENED 2026-07-30, and stated here because a docstring claiming one scope over code covering
two is itself a defect this repository has hit. This file started as a gate on CODEX_AUDIT's outcome
table and now also gates FIXES THIS PROJECT MADE ON ITS OWN INITIATIVE -- the `R8-` probes. The reason
is measured, not precautionary: a review pointed out that nothing gated round 8's own repairs, and
reverting each of the three left all 22 gates green.
  R8-head / R8-head2  the Ch.5 glossary violation ("region head" -> "region output", and the repo
                      shorthand -> "region-transition prior"), fixed in 48c4d01d
  R8-vintage          the Ch.6 data-vintage item printing BOTH Gowalla windows, the paper's stated one
                      and the measured span of the files actually used
  R8-bibfont          an INVERTED probe: the \footnotesize bibliography wrapper must stay ABSENT.
                      REV-024 was archived as closed this session on a ONE-TIME measurement, which is
                      the very defect written up as PENDENCIAS 2.19 -- a measurement without its tree
                      state can only be re-taken, never re-checked. This probe converts it into
                      something re-checkable on every run.
Adding a probe here is now part of applying a fix, not a later tidy-up: if reverting the edit leaves
the suite green, the fix is undefended.

ONE TRAP WHEN VALIDATING AN INVERTED PROBE, hit while validating R8-bibfont. My sabotage inserted the
banned token near `\begin{document}` in preamble.tex -- and all SEVEN occurrences of that anchor in
that file are inside `%` comments, so live_text() stripped the sabotage and the probe correctly
reported holds. It read exactly like a probe that does not fire. Insert the sabotage into the first
LIVE line instead, and assert the token is present in live_text() before believing the verdict.

The stripper self-tests both directions (escaped % survives, real comment excluded) before this file
reports anything, because a stripper that silently over-strips turns every probe into a false pass.

HOW TO VALIDATE THIS GATE, because my first two attempts were both invalid and both looked like the
gate failing to fire. Sabotage must reintroduce the defect THE WAY IT ORIGINALLY EXISTED:

  1. WRONG -- copy src_utils and src to a temp tree and sed there. SRC resolves from __file__, so a
     copied checker reads the copied src; that part is fine. What broke it is (2).
  2. WRONG -- `sed 's/a user-disjoint statistical protocol/a leakage-guarded .../'`. That string does
     not exist on any single line: objective 4 wraps, with "user-disjoint" ending one line and
     "statistical protocol" opening the next. The sed matched only the PROVENANCE COMMENTS (which
     quote the phrase unwrapped), so the live prose was untouched and the gate correctly reported
     holds -- while I read it as the gate being blind.
  3. RIGHT -- replace across the wrap, in place, then restore:
       s.replace("in a user-disjoint\n        statistical protocol",
                 "in a leakage-guarded\n        statistical protocol", 1)
     with an assert that the substitution changed the text. Measured: rc=1 with the defect, naming
     COD-003; rc=0 after restoring; `git diff` empty, so the file came back byte-identical.

The irony is the point: the wrap that made my sabotage silently no-op is the SAME wrap that made a
per-line probe score NUM-4's real fix as missing (trap 2 above). A test that cannot fail is worth
nothing, and "the sabotage did not apply" and "the gate did not fire" look identical from the outside.
Always assert that the sabotage changed something before believing what the gate says about it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
COMMENT = re.compile(r"(?<!\\)%")

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


def main() -> int:
    self_test()
    bad, missing_files = [], []
    print("== audit APPLIED claims re-measured against the live source ==")
    for fid, what, rel, pat, want in PROBES:
        path = SRC / rel
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
# THE CLOSED-ITEM REGISTER, checked for the same defect. _archive/PENDENCIAS_RESOLVIDOS.md carries
# 16 closed items with 40 commit citations -- the identical claim shape that failed in CODEX_AUDIT,
# so it was audited on 2026-07-30 rather than trusted. Its most exposed row is item 1.2, "the
# author's decisions that were applied", nine rows each naming a checkable artifact.
#
# FIVE OF THE NINE ROWS WERE PROBED, and this comment first said "ALL NINE HOLD" -- a batch claim of
# exactly the kind V13 names, written in the file that exists to stop batch claims. Corrected
# 2026-07-30 after review. What was actually measured:
#   HOLDS  LEFT_OUT.md carries 11 LO- entries (the row claimed 8, so it grew)
#   HOLDS  apx_b_static_scope.tex exists
#   HOLDS  main_ppgc.tex is 2 live lines, as claimed
#   HOLDS  the chapter split is exactly 18 per-section files
#   HOLDS  the static-scope section is reachable through ONE \input -- but see below
# The static-scope row needed a second look rather than a verdict: grepping src/*.tex found ZERO,
# because the section moved into the supplementary volume with the errata appendix. It is included at
# apx_b_errata.tex:448 and renders on 4 pages of main_extra.pdf. The claim holds; the probe was
# looking in the volume the section had left.
#
# FOUR ROWS WERE NEVER MEASURED HERE and must not be read as holding: the margin/geometry row, the
# comment-volume row (1,217 of 1,269), the front-matter-placeholder row, and the Resumo/Abstract row
# (500->310, 423->271). Re-probed on 2026-07-30, NONE of the three checkable ones reproduces from the
# live tree with a direct instrument: comment lines measure 3,614 across 59 .tex files against a
# claimed 1,269; the preamble carries 14 bracketed placeholders against a claimed 3; and the geometry
# and linespread are not in preamble.tex at all, so that row's instrument is not the one I reached for.
# THE RESUMO ROW IS ACTIVELY CONTRADICTED, and by three different numbers: the row says 310/271, a
# round-8 track measured 312/277, and my own instrument here gives 345/307. Three instruments, three
# answers, so the honest state is UNREPRODUCIBLE pending one agreed convention -- not "holds".
#
# WHY THE FOUR DO NOT REPRODUCE is almost certainly benign: they were measured in round 6 against a
# tree that has since gained an appendix, lost 0_main.tex to the preamble/content split, and moved two
# appendices into a second volume. A count taken then is not wrong; it is stale, and the row does not
# carry the tree state it was taken against. That is the defect worth recording -- a measurement
# without its tree state cannot be re-checked, only re-taken. PENDENCIAS 2.19 hands the convention
# question to the author rather than my picking a word-count instrument for the deposit.
#
# Not added as live probes here: these are claims about REPO STRUCTURE, not document strings, and
# most are already covered (check_tex_root, the extra-volume xref gate, sync_deliverables). Recorded
# so the next agent does not have to re-audit the register to learn it was audited.
# ---------------------------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())
