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

RECURSION. This gate is invoked from check.sh, and two documented blocks invoke check.sh (or
`make check`) themselves -- which is correct for the author, since checking that the gate suite passes
is one of the things the list tells him to do. Running them from inside the harness re-enters check.sh,
which re-enters the harness, and the run does not terminate. I built exactly that cycle on 2026-07-28
and it hung `make check` for ten minutes before I noticed.

So the harness SKIPS any block that invokes `check.sh` or `make check`, and reports them as skipped
with the reason rather than silently -- an unreported skip counted as a pass is the defect this file
was written to answer. Those two blocks are exercised every time check.sh itself runs, which is the
only place they can be exercised without recursion.

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
    skipped: list[tuple[str, str]] = []
    build_blocks: list[tuple[str, str, bool]] = []
    for doc in DOCS:
        if not doc.exists():
            continue
        for code in blocks(doc):
            body = [l for l in code.split("\n")
                    if l.strip() and not l.strip().startswith("#")]
            # ORDER MATTERS. The recursion test runs FIRST: blocks 3 and 17 of VERIFY_LIST are
            # single-line `cd ... && make check` commands, so the working-directory test below would
            # swallow them as "notes" and skip them silently -- which is how the first version of this
            # guard reported 0 skipped while still recursing. An unreported skip is the failure mode
            # this whole file exists to answer, so it must not be reachable for these.
            if "check.sh" in code or "make check" in code:
                # See RECURSION in the module docstring.
                skipped.append((doc.name, body[0][:58] if body else "(empty)"))
                continue
            if ("make defense" in code or "make final" in code or "make ppgc" in code
                    or "pdflatex" in code):
                # A three-target build takes ~4 minutes and check.sh is a lint gate that people run
                # constantly. Running it here took make check from 4 seconds to 297. The build is
                # already verified by build.sh, which is the tool for it; what matters HERE is that
                # the documented command is well formed and its paths resolve, so that is what is
                # checked. Reported, not silent.
                first = body[0] if body else ""
                probe = re.sub(r'&&\s*\(?cd src.*$', '', first).strip().rstrip("&").strip()
                ok = True
                if probe:
                    r = subprocess.run(["bash", "-c", probe + " && echo REACHED"],
                                       capture_output=True, cwd=cwd_for(code))
                    ok = b"REACHED" in r.stdout
                build_blocks.append((doc.name, first[:58], ok))
                continue
            if len(body) == 1 and body[0].strip().startswith("cd ") and "&&" not in body[0]:
                continue                       # the bare working-directory note, not a check
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
    for name, head in skipped:
        print(f"{len(skipped)} skipped to avoid recursion (they invoke this gate's own caller): "
              f"{name}: {head}" if skipped.index((name, head)) == 0 else
              f"    also: {name}: {head}")
    if skipped:
        print("    Those run every time check.sh runs, which is the only place they can run "
              "without re-entering this harness.")
    for name, head, ok in build_blocks:
        print(f"  {'cd-ok   ' if ok else 'CD-FAIL '} {name}: {head}  "
              f"(build block: cwd checked, build NOT run here -- use build.sh)")
    if any(not ok for _, _, ok in build_blocks):
        print("    A build block whose own cd does not resolve cannot work for the author.")
        return 1
    if ran - asserted:
        print(f"{ran - asserted} were executed but NOT asserted against their prose expectation "
              f"(prose too discursive to encode, or the check is a human judgment). "
              f"Do not describe those as verified.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
