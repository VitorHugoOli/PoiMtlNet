#!/usr/bin/env python3
"""check_audit_claims.py -- re-measures every "APPLIED" claim against the live source.

WHAT IT GATES. Two things, and the scope is wider than the name: (1) the CODEX_AUDIT outcome table's
COD-/NUM- findings, and (2) fixes this project made on its own initiative (the `R8-` probes). An
outcome table is a CLAIM ABOUT THE WORK, the highest-risk statement class here, and it was the one
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
# THE CLOSED-ITEM REGISTER was audited for this same defect on 2026-07-30, not trusted:
# _archive/PENDENCIAS_RESOLVIDOS.md carries 16 closed items with 40 commit citations, the identical
# claim shape that failed in CODEX_AUDIT. Its most exposed row (item 1.2, nine checkable artifacts)
# holds for five rows; the other four do not reproduce and are recorded as PENDENCIAS 2.19 with the
# reason -- they are STALE, not false: taken in round 6 against a tree that has since gained an
# appendix and lost 0_main.tex, and none records the tree state it was taken against.
# Full audit in _round8/29_pendencias_detail.md. Do not redo it; extend 2.19 if a number moves.
if __name__ == "__main__":
    sys.exit(main())
