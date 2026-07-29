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
The stripper self-tests both directions (escaped % survives, real comment excluded) before this file
reports anything, because a stripper that silently over-strips turns every probe into a false pass.
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
    ("COD-006b", '"before any result was read" removed from the same paragraph',
     "chapters/5_mobiwac/05_setup.tex", r"before any result was read", False),
    ("COD-013",  "Appendix C names the model family in PROSE, not only in a comment",
     "chapters/apx_c_ai_disclosure.tex", r"Opus", True),
    ("COD-015a", "Ch.3 preface no longer says Ch.4 and Ch.5 both change the representation",
     "chapters/3_cbic.tex", r"revise that verdict by changing the input representation", False),
    ("COD-015d", "Ch.2 no longer promises a relative multi-task performance metric",
     "chapters/2_fundamentals.tex", r"relative multi-task performance", False),
    ("COD-016a", "Ch.3 unbalanced-result sentence rewritten",
     "chapters/3_cbic/results.tex", r"important to notice that since we have an", False),
    ("COD-018",  "Appendix A carries the author's per-role CoUrb credit",
     "chapters/apx_a_contributions.tex",
     r"undergraduate|implementation support|wrote the multi-task", True),
    ("NUM-4",    "HGI sweep reports its spreads and its averaging convention",
     "chapters/2_fundamentals.tex", r"0\.8186", True),
)

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
    for fid, why in sorted(NOT_CHECKABLE.items()):
        print(f"  unprobed    {fid:9s} {why}")

    print(f"\n  {len(PROBES) - len(bad) - len(missing_files)} of {len(PROBES)} probes hold; "
          f"{len(bad)} claim(s) not applied; {len(NOT_CHECKABLE)} process claim(s) unprobed")
    if missing_files:
        print("  A probe whose file is gone is NOT a pass. Re-point it or retire it deliberately.")
        return 2
    if bad:
        print(f"\nFAIL: {', '.join(bad)} are recorded as APPLIED and are not in the document.")
        print("  Do not re-mark them from a report. Fix the source, then re-run this.")
        return 1
    print("\nOK: every mechanically checkable audit claim is true of the live source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
