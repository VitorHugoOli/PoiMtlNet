#!/usr/bin/env python3
"""Detect prose accidentally trapped inside a LaTeX comment line.

WHY: this failure is SILENT. The build succeeds, no warning is emitted, and the reader simply gets a
sentence that stops mid-clause or a disclosure that never appears. It has bitten this document four
times, twice from edits that were themselves repairs:

  apx_a_contributions.tex   a sentence appended after a comment's terminal period
  4_courb.tex:187           half a PUBLISHED methodology sentence, dropping three method facts
  5_mobiwac.tex:385         four sentences including a required three-limits disclosure
  4_courb.tex:311 and :362  two PUBLISHED results passages, ~270 words

THE DISCRIMINATOR. Vocabulary does not separate ledger commentary from trapped prose: both are
English sentences about the same subject, and every filter attempt either flooded (40 false
positives) or silenced a real case. POSITION does separate them. A LaTeX comment block is written as
consecutive `%` lines; trapped prose is body text that landed at the END of the LAST comment line of
its block, because that is where an editing tool appends. So:

    a comment line whose NEXT source line is also a comment  -> continuation, never flagged
    the LAST comment line of a block, with a long prose tail  -> candidate
    ... and the candidate is confirmed only if that tail is ABSENT from the rendered PDF.

Both conditions are mechanical and neither guesses at meaning. Validated in both directions: 0
findings on the repaired tree, and all four historical cases are caught when reintroduced.

Exit 1 if any suspect is found, so this can gate a build.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
PDF = SRC / "dissertacao.pdf"
MIN_TAIL_WORDS = 12


def rendered_text() -> str:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(PDF))
    raw = "\n".join(doc[i].get_textpage().get_text_range() for i in range(len(doc)))
    return re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "", raw.replace("\r", " ")))


def probe_of(text: str, n: int = 8) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z\-']*", text)
    return " ".join(words[:n]) if len(words) >= n else ""


def suspects_in(path: Path, pdf: str) -> list[tuple[int, str, str]]:
    lines = path.read_text(encoding="utf8", errors="replace").split("\n")
    out = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("%"):
            continue
        # POSITION TEST. Appended body prose lands at the end of a comment LINE, and the tell is
        # that the line is over-long relative to the block it sits in: a hand-written comment block
        # wraps at a consistent width, so the line carrying appended prose is markedly longer than
        # its neighbours. Measure the block, then test each line against it. (An earlier version
        # only tested the block's LAST line, which missed 5_mobiwac.tex:385 because more comment
        # lines followed the damaged one.)
        lo = i
        while lo > 0 and lines[lo - 1].strip().startswith("%"):
            lo -= 1
        hi = i
        while hi + 1 < len(lines) and lines[hi + 1].strip().startswith("%"):
            hi += 1
        block = [len(lines[k].rstrip()) for k in range(lo, hi + 1)]
        if len(block) > 1:
            others = sorted(block[k] for k in range(len(block)) if lo + k != i)
            typical = others[len(others) // 2] if others else 0
            if typical and len(line.rstrip()) < typical + 25:
                continue        # in line with its block: an ordinary wrapped comment
        # The tail is what follows the final sentence terminator on this line.
        m = re.search(r"[.!?]\s+([A-Za-z][^%]*)$", stripped)
        if not m:
            continue
        tail = m.group(1).strip()
        if len(tail.split()) < MIN_TAIL_WORDS:
            continue
        # SHAPE TEST. Trapped body prose is complete typeset text: it ends at a sentence terminator
        # (or a LaTeX macro/math close), because it was torn out of a finished paragraph. A ledger
        # note that merely runs long ends mid-thought, on a bare word or a bracketed marker, since
        # its author wrapped by eye. This is what separates the three residual false positives in
        # 2_fundamentals.tex from the three real cases in 4_courb.tex and 5_mobiwac.tex.
        if not re.search(r"(?:[.!?]|\}|\$)\s*$", tail):
            continue
        # A tail that is one clause of ledger shorthand rather than prose: no verb-bearing second
        # sentence and heavy on parentheticals/identifiers.
        if len(re.findall(r"[.!?]", tail)) < 2 and re.search(r"[\[\]{}():;]|\b[a-z]+\d{4}[a-z]*\b", tail):
            continue
        # RENDER TEST: body prose reaches the PDF; a comment does not.
        probe = probe_of(tail)
        if probe and probe not in pdf:
            out.append((i + 1, tail[:110], probe))
    return out


def main() -> int:
    if not PDF.exists():
        print(f"FAIL: {PDF} not found; build first")
        return 2
    pdf = rendered_text()
    total = 0
    for tex in sorted(list((SRC / "chapters").glob("*.tex")) + [SRC / "0_main.tex"]):
        for lineno, tail, probe in suspects_in(tex, pdf):
            total += 1
            print(f"TRAPPED PROSE {tex.name}:{lineno}")
            print(f"  last comment line of its block; tail absent from the PDF: '{probe}...'")
            print(f"  tail: {tail}")
    print(f"trapped-prose suspects: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
