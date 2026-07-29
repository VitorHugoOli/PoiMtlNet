#!/usr/bin/env python3
"""check_tracker_refs.py -- every live PENDENCIAS section citation must resolve.

WHY THIS EXISTS. On 2026-07-29 PENDENCIAS.md was renumbered (2.3-2.7 -> 2.4-2.8) to insert a new
item. Four source comments cited "PENDENCIAS 2.2" for the static-scope section. Three were
repointed; the fourth -- apx_b_static_scope.tex's own header, the file the citation is ABOUT -- was
missed, and the commit message claimed all four had been done. The grep that found four was run,
its output was read, and the fix list was written with three entries.

Nothing caught it, because a stale section number is not a LaTeX error, not an undefined reference,
and not a broken link. It renders fine and points somewhere wrong. That is the same silent class as
AGENT_HANDOFF §2.6 (records that drift from the thing they describe): the citation and its target
drift apart and only a reader who follows the pointer notices.

WHAT IT CHECKS. Every `PENDENCIAS <n>.<m>` citation in the live tree resolves to a heading that
exists in PENDENCIAS.md. A citation may be deliberately historical -- the tracker is rewritten every
round and a quote can outlive its coordinate -- so a citation is EXEMPT when its own text says so,
by carrying "was <n>.<m>" or naming the date of the tracker it refers to. That keeps the gate from
punishing an honest historical reference while still catching a silently orphaned one.

DELIBERATELY NOT CHECKED: citations inside dated audit records (_round*/, _review_v*/, _archive/,
_gates/, _specialists*/). Those describe a tracker as it stood and must not be rewritten.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DISS = Path(__file__).resolve().parent.parent
TRACKER = DISS / "src_utils" / "PENDENCIAS.md"
SKIP = ("_round", "_review_v", "_archive", "_gates", "_specialists", "/build/", "__pycache__")

# The citation may or may not carry a section symbol, and BOTH spellings are in the live tree.
# Found 2026-07-29 (round 8 audit): the original pattern was `PENDENCIAS\s+(\d+)\.(\d+)`, which
# requires whitespace immediately before the digits and therefore could not see `PENDENCIAS §2.9`.
# Two live citations use that spelling -- src/chapters/apx_f_cosine.tex:316 and
# src_utils/check_verify_list.py:128 -- so this gate reported "every live citation resolves" while
# never having looked at them. Reproduced before the fix: a file citing `PENDENCIAS §9.9` (no such
# heading) gave RC=0; the same file with `PENDENCIAS 9.9` gave RC=1. That is T1/T2 -- a guard whose
# pattern is a list of the spellings that existed when it was written. The `§` may also arrive as
# `\S` from LaTeX source or as the word "section".
CITE = re.compile(r"PENDENCIAS\s*(?:§|\\S|[Ss]ection|[Ss]ec\.?|item)?\s*(\d+)\.(\d+)")
HEADING = re.compile(r"^#{2,4}\s+(\d+)\.(\d+)\b", re.M)
# An exemption must be adjacent to the citation, not anywhere in the file: a "was 2.2" on line 400
# does not license a bare "PENDENCIAS 2.2" on line 3. 90 chars is about one wrapped line.
EXEMPT = re.compile(r"was\s+\d+\.\d+|tracker (?:of|was)|no longer resolves|de que data|daquela data")


def sections() -> set[tuple[str, str]]:
    if not TRACKER.exists():
        return set()
    return {(a, b) for a, b in HEADING.findall(TRACKER.read_text(encoding="utf-8"))}


def scan() -> list[str]:
    live = sections()
    if not live:
        return [f"{TRACKER.name}: no numbered section headings found -- "
                f"if the format changed, update check_tracker_refs.py or this gate is blind"]
    problems = []
    for path in sorted(DISS.rglob("*")):
        if not path.is_file() or path.suffix not in (".tex", ".md", ".sh", ".py"):
            continue
        rel = str(path.relative_to(DISS))
        if any(s in rel for s in SKIP) or rel.endswith("check_tracker_refs.py"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in CITE.finditer(text):
            key = (m.group(1), m.group(2))
            if key in live:
                continue
            window = text[m.start(): m.start() + 90]
            if EXEMPT.search(window):
                continue          # honest historical citation, says so on its own line
            line = text[:m.start()].count("\n") + 1
            problems.append(f"{rel}:{line}: cites PENDENCIAS {key[0]}.{key[1]}, "
                            f"which is not a heading in PENDENCIAS.md")
    return problems


def self_test() -> None:
    live = sections()
    assert live, "self-test: the tracker must expose numbered headings"
    # a bare citation of a nonexistent section is a finding; the same with "was N.M" is not
    assert not EXEMPT.search("PENDENCIAS 2.2: deixe isso facil"), \
        "self-test: a bare citation must NOT be exempt"
    assert EXEMPT.search("PENDENCIAS 2.4, was 2.2 before the renumber"), \
        "self-test: a citation carrying 'was N.M' must be exempt"
    # EVERY SPELLING IN THE LIVE TREE MUST BE VISIBLE TO CITE. A guard written as the set of
    # spellings that existed on the day it was written expires without announcing it, and this one
    # had: `PENDENCIAS §2.9` was invisible for as long as the gate existed. Each variant below is
    # asserted to yield the SAME coordinate, so adding a spelling to the tree without adding it
    # here fails the self-test instead of passing the gate.
    for variant in ("PENDENCIAS 2.4", "PENDENCIAS §2.4", "PENDENCIAS \\S2.4",
                    "PENDENCIAS section 2.4", "PENDENCIAS Sec. 2.4"):
        assert CITE.findall(variant) == [("2", "4")], \
            f"self-test: CITE cannot see the spelling {variant!r} -- it would go unchecked"
    # ...and a bare number that is not a tracker citation must not be swept in.
    assert not CITE.findall("see Table 2.4 of the manual"), \
        "self-test: CITE must not match a coordinate that does not name the tracker"


def nesting_problems() -> list[str]:
    """A "### N.M" heading must sit under its own "## §N", not under some other section's.

    FOUND 2026-07-30 BY THE AUTHOR, not by this gate. Items 2.9 and 2.10 were appended at the end of
    the file, which by then was inside "§5 - raised from CODEX_AUDIT", so a reader following the
    headings found a GPU-disk item and a checker-coverage item filed under an audit they have
    nothing to do with. Every citation still resolved -- scan() was green throughout -- because a
    citation resolves on the NUMBER and this defect is about POSITION.

    That is the same shape as the round-6 failure this whole round is about: the check that existed
    was green, and the thing that was wrong was not the thing it checked.
    """
    out, current = [], None
    for line in TRACKER.read_text(encoding="utf-8").split("\n"):
        m_sec = re.match(r"^## §(\d+)\b", line)
        if m_sec:
            current = m_sec.group(1)
            continue
        m_item = re.match(r"^### (\d+)\.(\d+)\b(.*)", line)
        if m_item and current is not None and m_item.group(1) != current:
            out.append(
                f"MISFILED  item {m_item.group(1)}.{m_item.group(2)} sits inside §{current}: "
                f"{m_item.group(3).strip()[:56]}"
            )
    return out


def main() -> int:
    self_test()
    problems = scan()
    misfiled = nesting_problems()
    for m in misfiled:
        print(m)
    if misfiled:
        print(f"\nFAIL: {len(misfiled)} item(s) filed under the wrong section. A reader navigating "
              f"by heading will not find them, and every citation still resolves, so nothing else "
              f"catches this. Move them under their own §N.")
        return 1
    for p in problems:
        print(p)
    if problems:
        print(f"\nFAIL: {len(problems)} tracker citation(s) point at a section that no longer "
              f"exists. Repoint them, or mark the citation historical by naming the old number "
              f"('PENDENCIAS 2.4, was 2.2') so a reader knows the coordinate moved.")
        return 1
    n = len(sections())
    print(f"OK: every live PENDENCIAS citation resolves ({n} numbered sections in the tracker; "
          f"self-test passed in both directions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
