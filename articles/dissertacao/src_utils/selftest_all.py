#!/usr/bin/env python3
"""selftest_all.py -- prove every checker in src_utils/ in BOTH directions, or report that it is
unproven. Run with `make selftest`.

WHY THIS EXISTS, AND WHY IT IS NOT PART OF `make check`
=======================================================
Twenty-one tools live in src_utils/. Each was written to catch a defect, and until 2026-07-30 none
had ever been audited as code. Two rounds of the author asking "why is this slow" turned up, in
plain output, a gate that rebuilt the whole document on every invocation and a mutation guard that
let seven of eight dangerous command shapes through. Both had been in place for weeks, and the gate
suite reported RC=0 the entire time.

The structural problem was never the individual bug. It was that a checker which has never been seen
to FIRE is a checker nobody knows works. A guard validated only against a clean tree proves exactly
nothing: it would pass if its regex matched nothing at all, if its file list were empty, if it had
been commented out. That is the failure class this runner closes.

It is deliberately NOT wired into `make check`. A lint gate people run constantly must stay at
seconds, and a gate suite that runs its own test suite on every invocation is the same
work-inside-work mistake being fixed here. `make check` is the gate; `make selftest` is the proof
that the gate can fail.

THE CONTRACT FOR EACH CHECKER
=============================
For checker X, two fixture trees under _fixtures/X/:
    dirty/   contains the defect X exists to catch. X MUST exit nonzero here, and its output MUST
             name the offending file -- an exit code alone does not prove it found the right thing.
    clean/   the same tree with the defect removed. X MUST exit 0 here.
A checker with only one side is reported as HALF-PROVEN. A checker with no fixtures is reported as
UNPROVEN. Both are printed, both are counted, and the runner exits nonzero if any checker in the
REQUIRED set is not fully proven -- because an unreported gap is precisely how the two big defects
survived. Silence is what we are engineering out.

Fixtures are TINY hand-written files, never copies of the real tree: a fixture that is a copy stops
being a fixture the moment the real tree changes, which is failure class T6 (self-describing values
that drift).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

UTILS = Path(__file__).resolve().parent
FIXTURES = UTILS / "_fixtures"

# The checkers that MUST be fully proven for this runner to pass. Start with the ones whose
# guarantees are load-bearing and add as fixtures are written; a name here with no fixture is a
# LOUD failure, which is the point -- the list is the commitment, the fixtures are the evidence.
REQUIRED = (
    "check_trapped_prose",
    "check_torn_sentences",
    "check_doubled_macro",
)

# Every checker in the tree, so the report covers the whole surface rather than only what is easy.
ALL_CHECKERS = (
    "check_comment_hygiene", "check_doubled_macro", "check_extra_xrefs", "check_meta_claims",
    "check_negative_parallelism", "check_tex_root", "check_torn_sentences", "check_tracker_refs",
    "check_trapped_prose", "check_verify_list", "check_wordcount_claims", "sweep_guard",
    "sync_page_counts", "sync_deliverables",
)


def run_in(checker: str, tree: Path) -> tuple[int, str]:
    """Run one checker against a fixture tree. Returns (rc, combined output).

    THE CHECKERS RESOLVE PATHS FROM __file__, NOT FROM cwd. Nearly every one computes
    `SRC = Path(__file__).parent.parent / "src"`, so running the real script with cwd set to a
    fixture would still sweep the REAL dissertation -- the fixture would be ignored and every
    result would be meaningless while looking perfectly plausible. That is exactly the shape of
    silent failure this runner exists to catch, so it must not commit it.

    The script is therefore COPIED into <tree>/src_utils/ and run from there, which makes its
    __file__-relative arithmetic land inside the fixture. A fixture tree must contain the src/
    subtree the checker expects; conftest-style shape is documented in each fixture's README.
    """
    tree_utils = tree / "src_utils"
    tree_utils.mkdir(parents=True, exist_ok=True)
    target = tree_utils / f"{checker}.py"
    target.write_bytes((UTILS / f"{checker}.py").read_bytes())
    proc = subprocess.run(
        [sys.executable, str(target)],
        cwd=str(tree), capture_output=True, text=True, timeout=120,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def prove(checker: str) -> dict:
    """Both directions for one checker. Never raises: a broken fixture is a REPORTED result."""
    base = FIXTURES / checker
    out: dict[str, object] = {"checker": checker, "dirty": None, "clean": None, "verdict": "UNPROVEN"}
    if not base.is_dir():
        return out

    dirty, clean = base / "dirty", base / "clean"
    if dirty.is_dir():
        try:
            rc, text = run_in(checker, dirty)
            # An exit code is not enough. The checker must NAME something in the fixture, otherwise
            # a tool that crashes for an unrelated reason would count as "fired correctly".
            named = any(p.name in text for p in dirty.rglob("*.tex")) or \
                    any(p.name in text for p in dirty.rglob("*.md"))
            out["dirty"] = {"rc": rc, "fired": rc != 0, "named_the_file": named,
                            "first_line": text.strip().split("\n")[0][:100] if text.strip() else ""}
        except Exception as exc:                                    # noqa: BLE001
            out["dirty"] = {"rc": None, "fired": False, "named_the_file": False,
                            "first_line": f"fixture raised: {exc!r}"[:100]}
    if clean.is_dir():
        try:
            rc, text = run_in(checker, clean)
            out["clean"] = {"rc": rc, "silent": rc == 0,
                            "first_line": text.strip().split("\n")[0][:100] if text.strip() else ""}
        except Exception as exc:                                    # noqa: BLE001
            out["clean"] = {"rc": None, "silent": False, "first_line": f"fixture raised: {exc!r}"[:100]}

    d, c = out["dirty"], out["clean"]
    if d and c:
        ok = d["fired"] and d["named_the_file"] and c["silent"]
        out["verdict"] = "PROVEN" if ok else "FAILED"
    elif d or c:
        out["verdict"] = "HALF-PROVEN"
    return out


def main() -> int:
    results = [prove(c) for c in ALL_CHECKERS]
    by_verdict: dict[str, list[str]] = {}
    for r in results:
        by_verdict.setdefault(str(r["verdict"]), []).append(str(r["checker"]))

    print(f"== selftest_all: {len(ALL_CHECKERS)} checkers, fixtures in "
          f"{FIXTURES.relative_to(UTILS.parent)} ==")
    for r in results:
        v = str(r["verdict"])
        line = f"  {v:12s} {r['checker']}"
        d, c = r["dirty"], r["clean"]
        if v == "FAILED":
            if d and not d["fired"]:
                line += "  -- did NOT fire on the defect (exit 0 where it must be nonzero)"
            elif d and not d["named_the_file"]:
                line += "  -- fired but named no fixture file; it may be failing for another reason"
            elif c and not c["silent"]:
                line += f"  -- fired on the CLEAN fixture too (rc={c['rc']}): false positive"
        elif v == "HALF-PROVEN":
            line += "  -- only " + ("dirty/" if d else "clean/") + " exists; one side proves nothing"
        print(line)

    unproven = by_verdict.get("UNPROVEN", []) + by_verdict.get("HALF-PROVEN", [])
    failed = by_verdict.get("FAILED", [])
    proven = by_verdict.get("PROVEN", [])

    print(f"\n  PROVEN {len(proven)} | FAILED {len(failed)} | "
          f"UNPROVEN or HALF {len(unproven)} of {len(ALL_CHECKERS)}")
    if unproven:
        print("  Unproven checkers are NOT known to work. Each one's guarantee is currently an")
        print("  assumption: it would report OK if its pattern matched nothing, if its file list")
        print("  were empty, or if it had been commented out. Write a fixture pair before trusting one.")

    missing_required = [c for c in REQUIRED if c in unproven or c in failed]
    if failed:
        print(f"\nFAIL: {len(failed)} checker(s) did not hold in both directions: {', '.join(failed)}")
        return 1
    if missing_required:
        print(f"\nFAIL: required checker(s) not fully proven: {', '.join(missing_required)}")
        print("  These are in REQUIRED because their guarantees are load-bearing. Add the fixture")
        print("  pair, or remove the name and say in PENDENCIAS that the guarantee is unproven.")
        return 1
    print("\nOK: every required checker fires on its defect and stays silent on the clean fixture.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
