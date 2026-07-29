#!/usr/bin/env python3
"""count_errata_rows.py -- measure the Appendix B errata tables and reconcile the count the
appendix CLAIMS against the rows the tables actually hold.

WHY THIS EXISTS, and it is a defect of mine from round 8 rather than an inherited one.
=====================================================================================
AGENT_GUARDRAILS 4b V1 says a number about the work carries the command that produced it. I
obeyed the letter of that rule and broke it in substance: after adding a row to Table B.2 I
wrote the new counts into the reconciliation comment of apx_b_errata.tex and pasted a python
heredoc beside them as "the command that produced it". The counts were right (they came from a
different, correct measurement) and THE PASTED COMMAND COULD NOT PRODUCE THEM.

It used LaTeX control words as ordinary Python double-quoted literals. In Python, "\\begin" is
'\\x08egin' and "\\addlinespace" is '\\x07ddlinespace', because \\b is backspace and \\a is bell. So
the longtable test `if "\\begin{longtable}" in t` was false for every real longtable, both
longtables fell through to the \\midrule branch the comment's own CAVEAT paragraph warns about,
and the addlinespace filter never matched either. Run verbatim it prints

    B.1 12, B.2 15, B.3 5, B.4 22, TOTAL 54

against the EXPECT block sitting two lines below it, which said 8, 14, 4, 18, 44. A reader
checking my work would have found the tool disagreeing with the claim and had no way to tell
which was wrong. This is V3 exactly: a clean-looking instrument nobody had interrogated.

The lesson is not "escape your backslashes". It is that a command pasted into a LaTeX comment
passes through two escaping layers before anyone runs it, so the paste is the wrong medium for
anything with a backslash in it. A committed script has one layer, gets exercised, and can carry
its own self-test. So this file is the command; the appendix comment names it.

WHAT IT CHECKS
==============
The appendix states its own itemized row counts in prose ("B.1 CBIC content 8, B.2 CBIC wording
14, ... Total 44"). This script parses THAT claim, measures the four tables, and compares. No
expected value is hardcoded here: a constant in this file would be a second place for the number
to go stale, which is failure class T6. The claim in the document is the expectation, and this
script is what makes it a checked one.

Exit 0 when the claim matches the tables, 1 when it does not, 2 when the claim or a table cannot
be found at all (a probe whose target is gone is not a pass).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_audit_claims import live_text, strip_text          # noqa: E402

DISS = Path(__file__).resolve().parent.parent
TABLES = DISS / "src/tables"
APPENDIX = DISS / "src/chapters/apx_b_errata.tex"

# label -> table source. Order is the order the appendix states them in.
TABLE_FILES = (
    ("B.1", "cbic/errata.tex"),
    ("B.2", "cbic/errata_wording.tex"),
    ("B.3", "courb/errata.tex"),
    ("B.4", "frame/bib_errata.tex"),
)

# The claim, as the appendix writes it. Deliberately tolerant of the descriptive words between
# the label and the number ("B.1 CBIC content 8") so the prose can be reworded without breaking
# the guard, but NOT tolerant about the digits.
CLAIM_ROW = re.compile(r"\bB\.(?P<n>[1-4])\b[^,.]{0,40}?(?P<count>\d{1,3})\b")
CLAIM_TOTAL = re.compile(r"\bTotal\s+(?P<total>\d{1,3})\b", re.I)


def count_rows(tex: str) -> tuple[int, str]:
    r"""Data rows in one errata table, comments already stripped.

    Two containers, and the difference is the whole reason this function exists. A `table`
    float has one \midrule, so the body is what lies between it and \bottomrule. A `longtable`
    repeats its header on every page, which means it carries THREE \midrule and the first one
    belongs to \endfirsthead; splitting on it lands inside the header block and counts header
    material as data. For a longtable the body therefore starts at \endlastfoot.

    Measured cost of getting this wrong: B.1 reads 4 instead of 8, and B.4 reads 22 instead
    of 18 once the addlinespace separators are miscounted with it.
    """
    if r"\begin{longtable}" in tex:
        kind = "longtable"
        if r"\endlastfoot" not in tex:
            raise ValueError(r"longtable with no \endlastfoot: body boundary is undefined")
        body = tex.split(r"\endlastfoot", 1)[1].split(r"\end{longtable}", 1)[0]
    else:
        kind = "table"
        body = tex.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    rows = [r for r in (x.strip() for x in body.split(r"\\"))
            if r and r.replace(r"\addlinespace", "").strip()]
    return len(rows), kind


def self_test() -> None:
    r"""Prove the longtable branch FIRES, and prove the naive \midrule split gets it wrong.

    The second assertion is the important one and it is not decoration: it is the caveat from
    the appendix comment turned into a test. If a future edit "simplifies" count_rows by
    dropping the \endlastfoot branch, this fails instead of quietly reporting 4 for B.1.
    """
    lt_fixture = "\n".join([
        r"\begin{longtable}{ll}", r"\toprule", r"HEADER A & HEADER B \\", r"\midrule",
        r"\endfirsthead", r"\toprule", r"HEADER A & HEADER B \\", r"\midrule", r"\endhead",
        r"\midrule", r"\multicolumn{2}{r}{continued} \\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot",
        r"data one L & data one R \\", r"\addlinespace", r"data two L & data two R \\",
        r"\end{longtable}",
    ])
    n, kind = count_rows(lt_fixture)
    assert kind == "longtable", f"self-test: longtable not recognized, got {kind!r}"
    assert n == 2, f"self-test: longtable body should hold 2 data rows, got {n}"

    naive = lt_fixture.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    naive_n = len([r for r in (x.strip() for x in naive.split(r"\\")) if r])
    assert naive_n != 2, (
        "self-test: the naive \\midrule split was expected to MISCOUNT this longtable, and it "
        "agreed instead. Either the fixture stopped exercising the repeated header or the "
        "branch under test is no longer doing anything."
    )

    tbl_fixture = "\n".join([
        r"\begin{table}", r"\toprule", r"H & H \\", r"\midrule",
        r"only row L & only row R \\", r"\bottomrule", r"\end{table}",
    ])
    n2, kind2 = count_rows(tbl_fixture)
    assert (n2, kind2) == (1, "table"), f"self-test: float table gave {(n2, kind2)}"

    # And the escaping defect this file was written to replace: a control word must survive as
    # itself. If someone re-introduces a non-raw literal, this catches it here rather than in
    # a number the author reads six weeks from now.
    assert strip_text(r"x \addlinespace y").count(r"\addlinespace") == 1, \
        "self-test: \\addlinespace did not survive as a literal (non-raw string somewhere?)"


def main() -> int:
    self_test()
    print("== Appendix B errata rows: the appendix's own claim vs the tables ==")

    if not APPENDIX.exists():
        print(f"  MISSING {APPENDIX} -- the claim cannot be read")
        return 2
    # RAW text, NOT live_text. The reconciliation claim lives inside a `%` provenance comment, so
    # the comment stripper deletes exactly the sentence being checked and the gate reported "claim
    # not found" (rc=2) on a tree where the claim was present and correct. The stripper is right
    # for PROSE probes and wrong here: this guard's subject is a note to future maintainers, which
    # is comment-only by design. The tables below are still read through live_text, because there
    # the commented-out rows must not be counted.
    claim_text = APPENDIX.read_text(encoding="utf-8", errors="replace")
    claim_text = re.sub(r"\s+", " ", claim_text)
    # Narrow to the reconciliation sentence, so a stray "B.2" elsewhere cannot be misread.
    # The window runs from the "forward:" colon to the "Total NN." that closes the claim. It is NOT
    # bounded by "the next period": the labels themselves contain periods ("B.1"), so a [^.] window
    # stopped after the first label and only B.1 was ever parsed, which made every other row read
    # MISMATCH with "claimed --" on a tree whose claim was correct. That false alarm is the same
    # shape as the defect this script replaces: a guard that looks like it is measuring something.
    anchor = re.search(r"Itemized rows, RE-MEASURED.*?forward:(?P<body>.*?Total\s+\d{1,3}\.)",
                       claim_text, re.S)
    if not anchor:
        print("  The itemized-rows claim was not found in the appendix. Cannot reconcile.")
        return 2
    body = anchor.group("body")
    claimed = {f"B.{m.group('n')}": int(m.group("count")) for m in CLAIM_ROW.finditer(body)}
    total_m = CLAIM_TOTAL.search(body)

    measured: dict[str, int] = {}
    for label, rel in TABLE_FILES:
        path = TABLES / rel
        if not path.exists():
            print(f"  MISSING {rel} -- probe cannot run")
            return 2
        n, kind = count_rows(live_text(path))
        measured[label] = n
        state = "ok" if claimed.get(label) == n else "MISMATCH"
        print(f"  {state:9s} {label}  {rel:26s} measured {n:3d}  claimed "
              f"{claimed.get(label, '--')!s:>3s}  ({kind})")

    bad = [k for k in measured if claimed.get(k) != measured[k]]
    tot_measured = sum(measured.values())
    tot_claimed = int(total_m.group("total")) if total_m else None
    tot_state = "ok" if tot_claimed == tot_measured else "MISMATCH"
    print(f"  {tot_state:9s} TOTAL {'':30s} measured {tot_measured:3d}  claimed "
          f"{tot_claimed if tot_claimed is not None else '--'}")

    if bad or tot_claimed != tot_measured:
        print("\nFAIL: the appendix states row counts that its own tables do not hold.")
        print("  Fix the claim in src/chapters/apx_b_errata.tex, or the table, then re-run.")
        return 1
    print("\nOK: every itemized row count in Appendix B matches the table it describes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())