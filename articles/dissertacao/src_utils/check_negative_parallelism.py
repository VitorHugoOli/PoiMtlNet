#!/usr/bin/env python3
"""check_negative_parallelism.py -- hold the negative-parallelism density under its ceiling.

WHY THIS EXISTS. On 2026-07-20 the AI-credibility persona measured the `rather than` /
`X, not Y` / `instead of` family, called it "the single tell a 2026 CS examiner is most
primed to see", judged the density then defensible, and issued an explicit instruction:
"Freeze the count; do not let edit passes raise it."

On 2026-07-28 the same persona re-ran and found the count had gone from 67 to 79
document-wide, with this round's new prose accounting for the whole increase. Its verdict
on why is the reason this file exists:

    "A guard that lives only in a previous round's review report is a guard nobody is
    checking."

That is exactly right. An instruction in a review report is read once, by whoever asked
for that report. A guard in `check.sh` is read by every commit. So the instruction is
moved here, where it runs.

WHAT IT MEASURES. Density per 1,000 prose words, not an absolute count -- the document
grows, and a frozen absolute count would fire on honest growth and would let the density
rise inside a shrinking section. Comment lines are stripped first: this repository's
provenance comments quote the constructions they discuss, and counting them would make the
gate fire on its own documentation.

THE CEILING. 3.60 per 1,000, set from the measured state after the 2026-07-28 reduction
(3.07/1k document-wide: `rather than` 69, `, not` 38, `instead of` 9, over 37,780 prose
words). That leaves real headroom for ordinary drafting while catching a pass that adds
these constructions the way this round's did (the new prose alone ran at 5.35/1k).

RAISING THE CEILING IS A DECISION, NOT A FIX. If this gate fires, the first question is
whether the new sentences need the construction to scope a claim -- `X, not Y` is a
legitimate honesty device in this document and appears in the writing law as one. If they
do, the sentences stay and the author raises the ceiling here with a comment saying why.
If they do not, rewrite the sentences. Do not silently bump the number.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

CEILING_PER_1K = 3.60

PATTERNS = {
    "rather than": re.compile(r"\brather than\b"),
    ", not":       re.compile(r", not\b"),
    "instead of":  re.compile(r"\binstead of\b"),
    "not ... but": re.compile(r"\bnot only\b[^.]{0,80}\bbut\b"),
}


def strip_comments(text: str) -> str:
    """Drop comment lines and inline comment tails, honouring escaped percent signs."""
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("%"):
            continue
        buf = []
        i = 0
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            buf.append(line[i])
            i += 1
        out.append("".join(buf))
    return "\n".join(out)


def files() -> list[Path]:
    # chapters/*/*.tex included: the paper chapters are split per section (2026-07-28).
    return (sorted(SRC.glob("chapters/*.tex")) + sorted(SRC.glob("chapters/*/*.tex"))
            + [SRC / "0_main.tex"])


def measure(paths: list[Path]) -> tuple[dict[str, int], int, dict[str, dict[str, int]]]:
    counts = {k: 0 for k in PATTERNS}
    per_file: dict[str, dict[str, int]] = {}
    words = 0
    for path in paths:
        if not path.exists():
            continue
        text = strip_comments(path.read_text(encoding="utf-8"))
        words += len(re.findall(r"[A-Za-z][A-Za-z'-]+", text))
        local = {}
        for name, pat in PATTERNS.items():
            n = len(pat.findall(text))
            counts[name] += n
            if n:
                local[name] = n
        if local:
            per_file[str(path.relative_to(SRC))] = local
    return counts, words, per_file


def self_test() -> None:
    """Both directions: a dense sample must fail, a clean one must pass."""
    dense = "We measured this rather than assuming it. It is a fact, not a guess. " \
            "We chose the first instead of the second. " * 1
    clean = "We measured this. It is a fact. We chose the first option. " * 12
    for label, sample, want_over in (("dense", dense, True), ("clean", clean, False)):
        n = sum(len(p.findall(sample)) for p in PATTERNS.values())
        w = len(re.findall(r"[A-Za-z][A-Za-z'-]+", sample))
        dens = n / w * 1000
        over = dens > CEILING_PER_1K
        assert over == want_over, f"self-test {label}: density {dens:.2f}, over={over}"
    # a construction inside a comment must not count
    assert sum(len(p.findall(strip_comments("% rather than in a comment\nplain text")))
               for p in PATTERNS.values()) == 0, "self-test: comment stripping"


def main() -> int:
    self_test()
    counts, words, per_file = measure(files())
    total = sum(counts.values())
    density = total / words * 1000 if words else 0.0
    detail = ", ".join(f"{k} {v}" for k, v in counts.items() if v)
    print(f"negative parallelism: {total} instances / {words} prose words = "
          f"{density:.2f} per 1k (ceiling {CEILING_PER_1K:.2f}) [{detail}]")
    if density > CEILING_PER_1K:
        print("FAIL: density above the ceiling. Densest files:")
        ranked = sorted(per_file.items(), key=lambda kv: -sum(kv[1].values()))
        for name, local in ranked[:6]:
            print(f"    {name}: {local}")
        print("  Rewrite the sentences that do not need the construction to scope a claim.")
        print("  If they all do, raise CEILING_PER_1K here with a comment saying why.")
        return 1
    print("OK (self-test passed in both directions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
