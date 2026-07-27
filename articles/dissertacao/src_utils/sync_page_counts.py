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
CLAIMS = [
    ("CLAUDE.md",                r"\(\*\*(\d+) pp\*\*, full front matter", "defense"),
    ("CLAUDE.md",                r"\(\*\*(\d+) pp\*\*, AcademicoPG body-only", "final"),
    ("PLAN.md",                  r"defense \*\*(\d+) pp\*\*", "defense"),
    ("PLAN.md",                  r"final AcademicoPG \*\*(\d+) pp\*\*", "final"),
    ("src_utils/PENDENCIAS.md",  r"\*\*Build:\*\* defesa (\d+) pp", "defense"),
    ("src_utils/PENDENCIAS.md",  r"\*\*Build:\*\* defesa \d+ pp, final (\d+) pp", "final"),
    ("src_utils/PENDENCIAS.md",  r"O que esta em disco e \*\*(\d+)/\d+\*\*", "defense"),
    ("src_utils/PENDENCIAS.md",  r"O que esta em disco e \*\*\d+/(\d+)\*\*", "final"),
    ("src_utils/codex_reviewer.md", r"The builds on disk are \*\*(\d+)/\d+ pages\*\*", "defense"),
    ("src_utils/codex_reviewer.md", r"The builds on disk are \*\*\d+/(\d+) pages\*\*", "final"),
]


def measured() -> dict[str, int]:
    out = {}
    for stem, key in (("main", "defense"), ("main_final", "final")):
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
    print(f"measured from the build logs: defense {truth['defense']} pp, final {truth['final']} pp")
    stale = 0
    for rel, pattern, which in CLAIMS:
        path = ROOT / rel
        if not path.exists():
            print(f"  SKIP  {rel} (missing)")
            continue
        text = path.read_text()
        m = re.search(pattern, text)
        if not m:
            print(f"  SKIP  {rel}: pattern not found -- {pattern}")
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
