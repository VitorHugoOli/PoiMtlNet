#!/usr/bin/env python3
"""check_verify_list.py -- run every command documented in VERIFY_LIST.md and PENDENCIAS.md,
and compare what it returns against what the document says it should return.

WHY THIS EXISTS. On 2026-07-28 three separate commands written into the author-facing verification
documents did not answer the question they were annotated with:

  * a count of \path{} entries annotated "# 13" returned 15, because it did not strip the appendix's
    own provenance comments, which mention \path{} while explaining the count;
  * a "four of six" sweep promised 3 prose hits and returned 4, one an indented comment;
  * a "three of our six" sweep promised ZERO prose hits and returned 5, every one an audit comment.

Each was individually plausible. What made them survive is that nothing ran them. A verification
document nobody executes is a claim, not a check -- the same failure this repository has now recorded
for its build reporter, four checker globs, and a page-count syncer.

WHAT IT CHECKS. For each fenced bash block carrying a machine-checkable EXPECT annotation (see below),
the block is run from its declared working directory and its output compared against the expectation.
Blocks without an EXPECT annotation are still RUN, and reported as executed-not-asserted, so the count
of what was actually verified is never overstated.

THE ANNOTATION. Add a comment line inside the bash block:

    # EXPECT: lines=3
    # EXPECT: contains=RC=0
    # EXPECT: equals=13

Supported: lines=N (exactly N non-empty output lines), contains=STR, equals=STR (whole stripped
output). Multiple EXPECT lines all have to hold. A block with no annotation is executed only.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DISS = Path(__file__).resolve().parent.parent
ROOT = DISS.parent.parent
DOCS = [DISS / "src_utils" / "_round6" / "VERIFY_LIST.md",
        DISS / "src_utils" / "PENDENCIAS.md"]

EXPECT = re.compile(r'#\s*EXPECT:\s*(lines|contains|equals)=(.*)$')


def blocks(doc: Path):
    text = doc.read_text(encoding="utf-8")
    for m in re.finditer(r'```bash\n(.*?)```', text, re.S):
        code = "\n".join(re.sub(r'^> ?', '', l) for l in m.group(1).split("\n"))
        yield code


def expectations(code: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).strip()) for m in
            (EXPECT.search(l) for l in code.split("\n")) if m]


def cwd_for(code: str) -> Path:
    """A block that cd's into articles/dissertacao needs the repo root; everything else runs
    from articles/dissertacao, which is what the document's own header declares."""
    first = next((l for l in code.split("\n") if l.strip()), "")
    if first.strip().startswith("cd articles/"):
        return ROOT
    return DISS


def main() -> int:
    ran = asserted = failed = 0
    for doc in DOCS:
        if not doc.exists():
            continue
        for code in blocks(doc):
            body = [l for l in code.split("\n")
                    if l.strip() and not l.strip().startswith("#")]
            if len(body) == 1 and body[0].strip().startswith("cd "):
                continue                       # the working-directory note, not a check
            exps = expectations(code)
            proc = subprocess.run(["bash", "-c", code], capture_output=True,
                                  cwd=cwd_for(code))
            out = proc.stdout.decode("utf-8", "replace").strip()
            lines = [l for l in out.split("\n") if l.strip()]
            ran += 1
            head = body[0][:58] if body else "(empty)"
            if not exps:
                print(f"  ran      {doc.name}: {head}  (no EXPECT annotation)")
                continue
            asserted += 1
            problems = []
            if proc.returncode != 0:
                problems.append(f"exit {proc.returncode}")
            for kind, want in exps:
                if kind == "lines" and len(lines) != int(want):
                    problems.append(f"lines={len(lines)}, expected {want}")
                elif kind == "contains" and want not in out:
                    problems.append(f"output does not contain {want!r}")
                elif kind == "equals" and out != want:
                    problems.append(f"output {out[:40]!r} != {want!r}")
            if problems:
                failed += 1
                print(f"  FAIL     {doc.name}: {head}")
                for p in problems:
                    print(f"           {p}")
            else:
                print(f"  verified {doc.name}: {head}")
    print(f"\n{ran} documented command(s) executed; {asserted} carried a machine-checkable "
          f"expectation; {failed} failed.")
    if ran - asserted:
        print(f"{ran - asserted} were executed but NOT asserted against their prose expectation "
              f"(prose too discursive to encode, or the check is a human judgment). "
              f"Do not describe those as verified.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
