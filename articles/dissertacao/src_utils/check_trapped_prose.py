#!/usr/bin/env python3
"""Detect prose accidentally trapped inside a LaTeX comment line.

WHY: this failure is SILENT. The build succeeds, no warning is emitted, and the reader gets a
sentence that stops mid-clause or a disclosure that never appears. It has now bitten this document
EIGHT times, every one of them from an edit that inserted an audit comment:

  apx_a_contributions.tex   a sentence FRAGMENT appended after a comment's terminal period
  4_courb.tex:187           half a PUBLISHED methodology sentence, dropping three method facts
  5_mobiwac.tex:385         four sentences including a required three-limits disclosure
  4_courb.tex:311 and :362  two PUBLISHED results passages, about 270 words
  2_fundamentals.tex:366    ", Nash-MTL treats"                       (3 words)
  6_conclusion.tex:105      "a capacity-matched dedicated baseline..." (7 words)
  apx_b_errata.tex:188      "The emphasis convention of the..."        (9 words)

THE ROOT CAUSE IS ALWAYS THE SAME: a comment block written without a trailing newline, so the body
line that followed got pulled onto the last comment line.

WHAT DOES *NOT* WORK AS A DISCRIMINATOR, learned by shipping each one and being caught:
  - vocabulary filters: ledger commentary and trapped prose are both English about the same
    subject. Filters either flooded (40 false positives) or silenced a real case.
  - "the tail must be a complete sentence": misses fragments (apx_a, and ", Nash-MTL treats").
  - "the tail must not contain parentheses": misses "...distinct counties (GEOIDs)."
  - LINE LENGTH / block-median overshoot: misses SHORT tails. The three cases above are 38, 124
    and 88 characters, at or below their blocks' own median width. This is the trap: tuning a
    length threshold on the long cases makes the checker blind to the short ones, and a
    three-word tail breaks a sentence just as thoroughly as a 270-word one.

THE TEST THAT ACTUALLY WORKS is the definition of the bug, with no proxy in between: take the words
that follow the LAST `%` on a comment line, and ask whether they appear in the rendered PDF. Body
text reaches the PDF; a comment never does. To keep ledger commentary from firing, require the
candidate to look like a continuation of the surrounding document rather than a note about it:
the words must run on into the NEXT source line, which is what makes it a torn sentence rather
than a self-contained remark.

VALIDATION, both directions, against build-fresh PDFs:
  repaired tree      -> 0 findings
  /tmp/tp2 (rebuilt) -> catches 5_mobiwac:385, 4_courb:311, 4_courb:362
  /tmp/tp4 (rebuilt) -> catches apx_a_contributions:57, 4_courb:198
  the three short cases above -> caught (they are why this version exists)

Exit 1 if any suspect is found, so this can gate a build.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
PDF = SRC / "dissertacao.pdf"
MIN_TAIL_WORDS = 2          # ", Nash-MTL treats" is three; do not raise this


def rendered_text() -> str:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(PDF))
    raw = "\n".join(doc[i].get_textpage().get_text_range() for i in range(len(doc)))
    return re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "", raw.replace("\r", " ")))


def words(text: str, n: int) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']*", text)[:n]


def suspects_in(path: Path, pdf: str) -> list[tuple[int, str, str]]:
    lines = path.read_text(encoding="utf8", errors="replace").split("\n")
    out = []
    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line.startswith("%"):
            continue
        # The candidate is whatever follows the last sentence terminator or closing bracket. The
        # separator may be whitespace OR nothing at all: the real 2_fundamentals case read
        # "% citation protocol)., Nash-MTL treats", where the appended text begins with a comma
        # directly against the period. Requiring whitespace here missed it.
        m = re.search(r"[.!?\]]\s*([A-Za-z,][^%]*)$", line)
        if not m:
            continue
        tail = m.group(1).strip().lstrip(",").strip()
        if len(words(tail, 6)) < MIN_TAIL_WORDS:
            continue
        # CONTINUATION TEST: real trapped prose is a torn fragment, so the NEXT source line is body
        # text that continues the same sentence. Two things follow, and both are required:
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if not nxt or nxt.startswith("%") or nxt.startswith("\\") or nxt.startswith("}"):
            continue
        #  (a) if the tail closes its own sentence it MIGHT be a self-contained ledger remark --
        #      but the real 4_courb:187 case also ended in a period ("...distinct counties
        #      (GEOIDs)."), so a closed sentence cannot be rejected outright. What separates them
        #      is whether the tail reads as an instruction to the author or as document prose:
        #      ledger remarks address the reader of the source ("Revert if you prefer...",
        #      "Author may reword", "see T4:8-17"), and cite files, line numbers, or finding ids.
        if re.search(r"[.!?][\]\)\"']*$", tail):
            LEDGER = re.compile(
                r"[\w/]+\.(tex|py|md|bib|json|csv|log|pdf)"     # file names
                r"|:\d+\b"                                      # line refs
                r"|\b(REV|NEW|COD|MJ|F)-?\d+\b"                 # finding ids
                r"|\b(author|revert|prefer|reword|persona|claim unchanged|see|source|verified"
                r"|unchanged|left as|deliberately|note for)\b",  # author-directed vocabulary
                re.I)
            if LEDGER.search(tail):
                continue
        #  (b) the RENDER TEST: the tail's own words, joined to the line that follows, must be
        #      ABSENT from the rendered PDF. This is the definition of the bug.
        #      NOTE: do NOT try "skip if the next line renders on its own" as a shortcut. In a real
        #      tear the rest of that line still renders perfectly well (only the torn fragment is
        #      missing), so that test silently suppressed four of the six real fixtures.
        joined = " ".join(words(tail + " " + nxt, 8))
        if not joined or joined.lower() in pdf.lower():
            continue
        #  (c) LAST GUARD, for the one false-positive shape seen in practice: a comment that merely
        #      PRECEDES a paragraph. There the tail is a complete ledger sentence and the following
        #      paragraph opens a new sentence of its own, so the words of the next line appear in
        #      the PDF immediately after a sentence boundary rather than continuing the tail.
        #      A genuine tear has the tail and its continuation adjacent in the SOURCE sentence, so
        #      the next line does not begin a new sentence -- it begins mid-clause, lowercase or
        #      with a connective. (apx_a_contributions.tex:92 was this shape.)
        if re.match(r"[A-Z]", nxt) and re.search(r"[.!?][\]\)\"']*$", tail):
            first_words = " ".join(words(nxt, 8))
            if first_words and first_words.lower() in pdf.lower():
                continue
        out.append((i + 1, tail[:100], joined))
    return out


def main() -> int:
    if not PDF.exists():
        print(f"FAIL: {PDF} not found; build first")
        return 2
    pdf = rendered_text()
    total = 0
    for tex in sorted(list((SRC / "chapters").glob("*.tex")) + [SRC / "0_main.tex"]):
        for lineno, tail, joined in suspects_in(tex, pdf):
            total += 1
            print(f"TRAPPED PROSE {tex.name}:{lineno}")
            print(f"  runs on into the next line, and the joined text is absent from the PDF:")
            print(f"    '{joined}...'")
            print(f"  comment tail: {tail}")
    print(f"trapped-prose suspects: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
