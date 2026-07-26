#!/usr/bin/env python3
"""Detect prose accidentally trapped inside a LaTeX comment line.

WHY: this failure has bitten this document twice and it is SILENT. The build succeeds, no warning is
emitted, and the reader simply gets a sentence that stops mid-clause.

  apx_a_contributions.tex   a sentence was appended after a comment's terminal period
  4_courb.tex:187           half a PUBLISHED methodology sentence was appended to a comment tail,
                            dropping three method facts from the rendered document

TWO EARLIER CHECKS FAILED, and the reason matters:
  * a source-only heuristic flagged every multi-line comment (the trapped text sits after a period,
    which is exactly what a legitimate comment continuation looks like);
  * a "are these words in the PDF?" check flagged every comment that discusses repo files, because
    ledger vocabulary is legitimately absent from the rendered document.

So this checker tests the thing that actually defines the bug: does the RENDERED sentence run to a
terminator? Trapped prose leaves the preceding body line ending without sentence-final punctuation,
and the rendered text then jumps to the next paragraph. That is checkable without guessing.

METHOD
  For every comment line whose tail looks like body prose, look at the nearest preceding
  NON-comment source line. If that line ends mid-sentence (no terminal ., :, ;, }, or math/macro
  close) AND the rendered PDF contains that line's closing words followed by something other than
  the tail's opening words, the tail is trapped.

Exit 1 if any suspect survives, so this can gate a build.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
PDF = SRC / "dissertacao.pdf"
TERMINAL = (".", ":", ";", "?", "!", "}", "]", "$", "\\\\", "%")


def rendered_text() -> str:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(PDF))
    raw = "\n".join(doc[i].get_textpage().get_text_range() for i in range(len(doc)))
    return re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "", raw.replace("\r", " ")))


def words_of(text: str, n: int, tail: bool = False) -> str:
    w = re.findall(r"[A-Za-z][A-Za-z\-']*", text)
    if len(w) < n:
        return ""
    return " ".join(w[-n:] if tail else w[:n])


def suspects_in(path: Path, pdf: str) -> list[tuple[int, str, str]]:
    lines = path.read_text(encoding="utf8", errors="replace").split("\n")
    out = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("%"):
            continue
        m = re.search(r"[.!?]\s+([A-Za-z][^%]{30,})$", stripped)
        if not m:
            continue
        tail = m.group(1).strip()
        if len(re.findall(r"[A-Za-z][A-Za-z\-]{3,}", tail)) < 6:
            continue
        # nearest preceding non-comment, non-blank source line
        prev = ""
        for j in range(i - 1, -1, -1):
            cand = lines[j].strip()
            if cand and not cand.startswith("%"):
                prev = cand
                break
        if not prev or prev.endswith(TERMINAL):
            continue  # the body sentence closed properly: the comment tail is just a comment
        anchor = words_of(prev, 6, tail=True)
        follow = words_of(tail, 4)
        if not anchor or anchor not in pdf:
            continue
        seg = pdf[pdf.index(anchor) + len(anchor):pdf.index(anchor) + len(anchor) + 120]
        if follow and follow.split()[0] not in seg:
            out.append((i + 1, tail[:110], f"body line {j + 1} ends mid-sentence: ...{anchor}"))
    return out


def main() -> int:
    if not PDF.exists():
        print(f"FAIL: {PDF} not found; build first")
        return 2
    pdf = rendered_text()
    total = 0
    for tex in sorted(list((SRC / "chapters").glob("*.tex")) + [SRC / "0_main.tex"]):
        for lineno, tail, why in suspects_in(tex, pdf):
            total += 1
            print(f"TRAPPED PROSE {tex.name}:{lineno}\n  {why}\n  commented tail: {tail}")
    print(f"trapped-prose suspects: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
