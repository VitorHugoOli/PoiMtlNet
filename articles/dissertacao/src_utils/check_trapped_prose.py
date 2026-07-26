#!/usr/bin/env python3
"""Detect prose accidentally trapped inside a LaTeX comment line.

WHY: this failure is SILENT. The build succeeds, no warning is emitted, and the reader gets a
sentence that stops mid-clause or a disclosure that never appears. It has bitten this document
five times, three of them from edits that were themselves repairs:

  apx_a_contributions.tex   a sentence FRAGMENT appended after a comment's terminal period
  4_courb.tex:187           half a PUBLISHED methodology sentence, dropping three method facts
  5_mobiwac.tex:385         four sentences including a required three-limits disclosure
  4_courb.tex:311 and :362  two PUBLISHED results passages, about 270 words

THE DISCRIMINATOR IS LINE GEOMETRY, NOT CONTENT. Three content-based attempts all failed:
vocabulary filters either flooded (40 false positives) or silenced a real case; requiring the tail
to be a complete sentence missed the apx_a case, whose trapped text is a FRAGMENT continuing on the
next body line; and treating parentheses as a ledger tell missed the 4_courb:187 case, whose tail
ends "...distinct counties (GEOIDs)."

What every real case shares is geometry: a hand-written comment block wraps at a consistent width,
so the line carrying appended body text is markedly longer than its neighbours in the same block.
Confirmed across the five cases: damaged lines run 141 to 283 characters against block medians near
90. That is the test, plus the only confirmation that matters, absence from the rendered PDF.

Threshold rationale: OVERSHOOT is the block median plus 40 characters, and a block must hold at
least three comment lines to have a measurable wrap width at all (a one- or two-line note is written
to whatever length it needs, which produced six false positives on section-header ledger notes). Real cases overshoot by 50 to
190. A hand-wrapped comment that merely runs a little long does not reach it, and the one line in
this repository that did (2_fundamentals.tex, 141 characters) was reflowed at the source rather than
tuned around.

VALIDATION, exercised in both directions against build-fresh PDFs:
  repaired tree      -> 0 findings
  /tmp/tp2 (rebuilt) -> catches 5_mobiwac.tex:385, 4_courb.tex:311, 4_courb.tex:362
  /tmp/tp4 (rebuilt) -> catches apx_a_contributions.tex and 4_courb.tex:187
Every one of the five historical cases is covered by a run recorded in this repository's history.

Exit 1 if any suspect is found, so this can gate a build.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
PDF = SRC / "dissertacao.pdf"
OVERSHOOT = 40          # characters beyond the block median that mark an appended tail
MIN_TAIL_WORDS = 8


def rendered_text() -> str:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(PDF))
    raw = "\n".join(doc[i].get_textpage().get_text_range() for i in range(len(doc)))
    return re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "", raw.replace("\r", " ")))


def probe_of(text: str, n: int = 7) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z\-']*", text)
    return " ".join(words[:n]) if len(words) >= n else ""


def block_median(lines: list[str], i: int) -> int:
    lo = i
    while lo > 0 and lines[lo - 1].strip().startswith("%"):
        lo -= 1
    hi = i
    while hi + 1 < len(lines) and lines[hi + 1].strip().startswith("%"):
        hi += 1
    others = sorted(len(lines[k].rstrip()) for k in range(lo, hi + 1) if k != i)
    # A block of fewer than three comment lines has no reliable wrap width: a one- or two-line
    # note is written to whatever length it needs, so "longer than its neighbours" is meaningless
    # there and produced six false positives on section-header ledger notes. Require a real block.
    if len(others) < 3:
        return 0
    return others[len(others) // 2]


def suspects_in(path: Path, pdf: str) -> list[tuple[int, str, str, int, int]]:
    lines = path.read_text(encoding="utf8", errors="replace").split("\n")
    out = []
    for i, line in enumerate(lines):
        if not line.strip().startswith("%"):
            continue
        width, median = len(line.rstrip()), block_median(lines, i)
        if median == 0 or width < median + OVERSHOOT:
            continue                                   # in line with its own block
        m = re.search(r"[.!?]\s+([A-Za-z][^%]*)$", line.strip())
        if not m:
            continue
        tail = m.group(1).strip()
        if len(tail.split()) < MIN_TAIL_WORDS:
            continue
        probe = probe_of(tail)
        if probe and probe not in pdf:                 # the reader never saw it
            out.append((i + 1, tail[:110], probe, width, median))
    return out


def main() -> int:
    if not PDF.exists():
        print(f"FAIL: {PDF} not found; build first")
        return 2
    pdf = rendered_text()
    total = 0
    for tex in sorted(list((SRC / "chapters").glob("*.tex")) + [SRC / "0_main.tex"]):
        for lineno, tail, probe, width, median in suspects_in(tex, pdf):
            total += 1
            print(f"TRAPPED PROSE {tex.name}:{lineno}")
            print(f"  line is {width} chars against a block median of {median}, "
                  f"and its tail is absent from the PDF: '{probe}...'")
            print(f"  tail: {tail}")
    print(f"trapped-prose suspects: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
