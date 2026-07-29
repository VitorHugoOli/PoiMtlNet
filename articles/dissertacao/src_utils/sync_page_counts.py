#!/usr/bin/env python3
"""Sync the recorded page counts in the governance files to the measured build.

WHY THIS EXISTS
---------------
The page counts in CLAUDE.md, PLAN.md, PENDENCIAS.md and codex_reviewer.md are PRESENT-TENSE claims
about what is on disk, and they have now drifted three times (87/83 -> 89/84 -> 103/98 -> 104/99).
Each drift was found by review, never by the author of the edit, because the cell that changes the
document is never the cell that updates the record.

The page-drift note in codex_reviewer.md is load-bearing: it tells a reader that every file:line in
that review has moved, and by roughly how much. A wrong count there silently misleads.

Run after any build that changes the page count:
    python3 src_utils/sync_page_counts.py            # report only
    python3 src_utils/sync_page_counts.py --write    # apply

Exits 1 when a recorded count disagrees with the build, so it can gate a commit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (file, regex with ONE capture group for the number, which build the number refers to)
#
# THE THIRD FIELD IS THE INTERNAL KEY, NOT DOC TEXT. It was "final" until 2026-07-29 and is now
# "academico", matching measured() below and the renamed build (LATEX_UPGRADE.md §4 A-1). The
# REGEXES are deliberately NOT renamed with it: they match what the author-facing documents
# actually say, and those documents still say `make final` / `main_final.pdf`. Renaming a
# pattern to match a document that has not changed is how this tool went silent on 2026-07-28 --
# it printed nothing and exited 0 while four page-count claims went unchecked. If a document is
# later reworded to `academico`, change its pattern HERE in the same commit and confirm this
# script still reports it (an UNMATCHED line is the loud failure; there is no silent skip).
CLAIMS = [
    # CLAUDE.md §1 was rewritten 2026-07-28 to describe three builds; these patterns follow it.
    ("CLAUDE.md",                r"`build/main\.pdf` \(\*\*(\d+) pp\*\*\)", "defense"),
    ("CLAUDE.md",                r"`build/main_final\.pdf` \(\*\*(\d+) pp\*\*", "academico"),
    ("CLAUDE.md",                r"`build/main_ppgc\.pdf` \(\*\*(\d+) pp\*\*", "ppgc"),
    ("PLAN.md",                  r"defense \*\*(\d+) pp\*\*", "defense"),
    ("PLAN.md",                  r"final AcademicoPG \*\*(\d+) pp\*\*", "academico"),

    # PENDENCIAS was rewritten as a three-part tracker on 2026-07-28 and its build state is now a
    # table row per target, not a "X/Y" pair in prose. The old patterns SKIPped silently after that,
    # which is the same failure this whole tool exists to prevent: a page-count claim nobody checks.
    ("src_utils/PENDENCIAS.md",  r"`make defense` -> `main\.pdf` \| \*\*(\d+)\*\*", "defense"),
    ("src_utils/PENDENCIAS.md",  r"`make final` -> `main_final\.pdf` \| \*\*(\d+)\*\*", "academico"),
    ("src_utils/PENDENCIAS.md",  r"`make ppgc` -> `main_ppgc\.pdf` \| \*\*(\d+)\*\*", "ppgc"),
    ("src_utils/codex_reviewer.md", r"The builds on disk are \*\*(\d+)/\d+ pages\*\*", "defense"),
    ("src_utils/codex_reviewer.md", r"The builds on disk are \*\*\d+/(\d+) pages\*\*", "academico"),
]


def measured() -> dict[str, int]:
    out = {}
    # ppgc added 2026-07-28: three targets now, and a claim about a target this tool does not
    # measure would raise a KeyError rather than being silently skipped, which is the right failure.
    # main_final -> main_academico on 2026-07-29 (LATEX_UPGRADE.md §4 A-1). The stem is
    # hardcoded here and in five other tools; a missed one does not error, it silently reports
    # the page count of a log that is no longer written.
    for stem, key in (("main", "defense"), ("main_academico", "academico"), ("main_ppgc", "ppgc")):
        log = ROOT / "src" / "build" / f"{stem}.log"
        if not log.exists():
            sys.exit(f"no {log.relative_to(ROOT)} -- build first, this script reads the real log")
        hits = re.findall(r"Output written on \S+ \((\d+) pages", log.read_text(errors="replace"))
        if not hits:
            sys.exit(f"{log.relative_to(ROOT)} has no page count -- the build did not finish")
        out[key] = int(hits[-1])
    return out


def main(write: bool) -> int:
    truth = measured()
    print("measured from the build logs: " +
          ", ".join(f"{k} {v} pp" for k, v in truth.items()))
    stale = 0
    for rel, pattern, which in CLAIMS:
        path = ROOT / rel
        if not path.exists():
            print(f"  SKIP  {rel} (missing)")
            continue
        text = path.read_text()
        m = re.search(pattern, text)
        if not m:
            # NOT a silent skip. A pattern that stops matching is a page-count claim that
            # nobody is checking any more -- the exact thing this tool exists to prevent.
            # PENDENCIAS.md drifted this way on 2026-07-28 when it was restructured.
            print(f"  UNMATCHED {rel}: pattern no longer matches -- {pattern}")
            print(f"            the claim it guarded is now unchecked. Fix the pattern or "
                  f"drop the row from CLAIMS.")
            stale += 1
            continue
        recorded = int(m.group(1))
        if recorded == truth[which]:
            continue
        stale += 1
        print(f"  STALE {rel}: records {recorded} for the {which} build, measured {truth[which]}")
        if write:
            lo, hi = m.span(1)
            path.write_text(text[:lo] + str(truth[which]) + text[hi:])
            print(f"        -> updated")
    if not stale:
        print("all recorded page counts agree with the build")
        return 0
    if write:
        print(f"{stale} claim(s) updated")
        return 0
    print(f"{stale} claim(s) stale; re-run with --write to fix")
    return 1


if __name__ == "__main__":
    sys.exit(main("--write" in sys.argv))
