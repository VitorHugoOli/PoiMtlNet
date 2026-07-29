#!/usr/bin/env python3
"""mkformat.py -- precompile this document's preamble into a pdflatex format dump.

WHY. Measured on this machine (2026-07-29, src/ at 0bfc9e5e, see src_utils/_round7/
20_build_speed.md for the table): one pdflatex pass over main.tex costs ~32 s, and the same
preamble with an EMPTY body costs ~28 s. The preamble -- abntex2 + memoir + newtxmath +
hyperref + abntex2cite -- is ~87% of every pass, and the document body is about four seconds
of it. A pass that loads a mylatexformat dump of that preamble instead of re-reading it runs
in ~4 s.

WHAT THIS SCRIPT DOES. It derives THREE files into build/fmt/ and never asks a human to keep
anything in sync:

  build/fmt/_pre.tex      main.tex's switch block + 0_main.tex's PREAMBLE region + \\endofdump
  build/fmt/_body.tex     0_main.tex's BODY region, from \\begin{document} to \\end{document}
  build/fmt/_run_<t>.tex  three lines per target: set the two switches, \\input the body

  build/fmt/mainpre.fmt   the dump itself
  build/fmt/fmt.key       the staleness key of the sources the dump was built from

THE SPLIT IS MECHANICAL, WHICH IS THE POINT. A hand-made split is how the first attempt at
this left a stray \\begin{document} in the body driver: the driver then re-read the whole of
0_main.tex, \\documentclass ran a second time, and the pass logged
"! LaTeX Error: Can be used only in preamble." while STILL emitting a PDF -- failure mode
science/AGENT_HANDOFF.md §2.3b exactly (a PDF existing is not evidence the source is
correct). Here the boundary is found by locating \\begin{document} in 0_main.tex, and the two
halves are complementary byte ranges of one file, so neither can contain the other's
commands. verify_format.py asserts `grep -c '^! '` is zero on the accelerated log rather
than trusting that a PDF appeared.

THE STALENESS GUARD IS THE OTHER HALF. A format dump is silently wrong when the preamble
moves under it: the build succeeds and the PDF is stale, which is the worst failure shape
this repository has (§2.3b again, and §2.4b). The key below covers, in order:

  1. the byte content of main.tex's switch region  (\\newif ... \\finalbuildfirstpage)
  2. the byte content of 0_main.tex's preamble region (everything before \\begin{document})
  3. the byte offset of that \\begin{document}      (so a moving boundary moves the key)
  4. the full byte content of abntex2-UFV.sty      (a local class-level style file)
  5. this script's own byte content                (a changed generator invalidates the dump)
  6. pdflatex's version banner
  7. every FILE THE DUMP ITSELF LOADED: path, size and mtime, read out of mainpre.log

Item 7 is what makes the guard closed rather than merely likely: the first six cover this
repository, and item 7 covers the ~90 class, package and font-map files in the TeX trees
that the preamble pulls in. Touch any of them and the key moves. `make fast` rebuilds the
dump when the key moves, so a stale-format build cannot be requested.

Deliberately NOT in the key: 0_main.tex's body region, chapters/, tables/, figures/ and
references.bib. Those are read at run time on every pass, so a prose edit must NOT cost a
28 s re-dump -- that would defeat the accelerator on the one edit the author makes most.

USAGE (from src/, with src_utils/texenv.sh sourced):
    python3 ../src_utils/mkformat.py --emit         # write the derived .tex files only
    python3 ../src_utils/mkformat.py --status       # exit 0 fresh, 1 stale/absent, and say why
    python3 ../src_utils/mkformat.py --build        # emit, then dump the format if stale
    python3 ../src_utils/mkformat.py --build --force
Run --selftest to exercise the split and the key on synthetic trees (both directions).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile

# The two switch declarations live in main.tex, and they must reach the dump BEFORE
# 0_main.tex's preamble: abntex2cite is loaded inside that preamble, and dumping without the
# switch block first dies with "Emergency stop" inside it. Anchored by PHRASE, not by line
# number (BRIEF.md: a third of the previous round's line coordinates went stale within one
# commit).
SWITCH_FIRST = r"\newif\ifdefensebuild"
SWITCH_LAST = r"\newcommand{\finalbuildfirstpage}"
BEGIN_DOC = r"\begin{document}"

# The three targets, and the switch state each one needs. These mirror Makefile's three
# recipes: `defense` leaves both defaults, `academico` clears \ifdefensebuild (which
# main_academico.tex does with \ACADEMICOBUILD), `ppgc` sets \ifapprovalsheet via
# main_ppgc.tex. Because \newif has already run inside the format, a run-time driver sets the
# booleans directly and ONE dump serves all three -- which is also why renaming the switch
# macro \FINALBUILD -> \ACADEMICOBUILD on 2026-07-29 (LATEX_UPGRADE.md §4 A-3) did not touch
# the drivers: they set \defensebuildfalse itself and never mention either macro.
TARGETS = {
    "defense": ("main", [r"\defensebuildtrue", r"\approvalsheetfalse"]),
    "academico": ("main_academico", [r"\defensebuildfalse", r"\approvalsheetfalse"]),
    "ppgc": ("main_ppgc", [r"\defensebuildtrue", r"\approvalsheettrue"]),
}


def read(path: str) -> str:
    with open(path, encoding="utf8") as fh:
        return fh.read()


def find_live(text: str, needle: str, start: int = 0) -> int:
    """Offset of the first occurrence of `needle` that is NOT inside a % comment, or -1.

    THIS FUNCTION IS THE WHOLE SAFETY MARGIN OF THE SPLIT, and it exists because the
    comment-blind version was actively dangerous rather than merely wrong. main.tex's header
    quotes the switch pattern inside a provenance comment:

        %       pdflatex "\\newif\\ifdefensebuild\\defensebuildfalse\\input{main.tex}"

    A plain str.find() for the switch anchor lands THERE, 34 lines above the real
    declaration, and slicing from the match offset drops the leading `%` -- so that quoted
    command becomes LIVE code in the dump and recursively \\input{main.tex} inside the format.
    Measured before the fix: the extracted switch region was 2,809 bytes for a region that is
    130. 0_main.tex has the same trap for \\begin{document} ("... would be too late").
    AGENT_GUARDRAILS §4b V4, which is a rule because it caused three defects in one day.
    """
    pos = start
    while True:
        i = text.find(needle, pos)
        if i < 0:
            return -1
        line_start = text.rfind("\n", 0, i) + 1
        if not re.search(r"(?<!\\)%", text[line_start:i]):
            return i
        pos = i + 1


def switch_region(main_tex: str) -> str:
    """main.tex's two \\newif blocks plus \\finalbuildfirstpage, verbatim.

    Fails loudly rather than guessing: an unfound anchor means main.tex was restructured,
    and a silently empty switch region would produce a dump that dies inside abntex2cite.
    """
    i = find_live(main_tex, SWITCH_FIRST)
    if i < 0:
        raise SystemExit(f"mkformat: live anchor not found in main.tex: {SWITCH_FIRST}")
    j = find_live(main_tex, SWITCH_LAST, i)
    if j < 0:
        raise SystemExit(f"mkformat: live anchor not found in main.tex: {SWITCH_LAST}")
    end = main_tex.find("\n", j)
    if end < 0:
        end = len(main_tex)
    return main_tex[i:end + 1]


def split_body(zero_main: str) -> tuple[str, str, int]:
    """Return (preamble region, body region, offset of \\begin{document}).

    The split point is the FIRST \\begin{document} that is not inside a comment. 0_main.tex
    mentions the string in a provenance comment above the real one ("\\begin{document} would
    be too late"), so a naive find() lands 114 lines early and cuts the preamble in half.
    AGENT_GUARDRAILS §4b V4: strip comments before matching, and filter the FILE.
    """
    i = find_live(zero_main, BEGIN_DOC)
    if i < 0:
        raise SystemExit(f"mkformat: no uncommented {BEGIN_DOC} found in 0_main.tex")
    return zero_main[:i], zero_main[i:], i


def loaded_files(dump_log: str) -> list[str]:
    """Every file the dump run opened, from mainpre.log's parenthesised paths.

    pdflatex writes "(<path>" for each file it reads. This is the input to key item 7, so it
    is deliberately over-inclusive: a path that no longer exists is kept in the key as
    "missing", which moves the key rather than silently dropping a member.
    """
    if not os.path.exists(dump_log):
        return []
    raw = open(dump_log, encoding="utf8", errors="replace").read()
    hits = re.findall(r"\((/[^\s()]+\.(?:sty|cls|clo|ltx|def|fd|cfg|tex|sto|map))", raw)
    return sorted(set(hits))


def compute_key(src: str, dump_log: str) -> tuple[str, dict]:
    """The staleness key. Returns (hex digest, the parts that went into it)."""
    main_tex = read(os.path.join(src, "main.tex"))
    zero_main = read(os.path.join(src, "0_main.tex"))
    pre, _body, off = split_body(zero_main)
    parts: dict[str, object] = {}
    h = hashlib.sha256()

    def feed(label: str, data: bytes) -> None:
        h.update(label.encode() + b"\0" + data + b"\0")
        parts[label] = hashlib.sha256(data).hexdigest()[:16]

    feed("main.tex:switches", switch_region(main_tex).encode())
    feed("0_main.tex:preamble", pre.encode())
    feed("0_main.tex:begindoc_offset", str(off).encode())
    sty = os.path.join(src, "abntex2-UFV.sty")
    feed("abntex2-UFV.sty", open(sty, "rb").read() if os.path.exists(sty) else b"MISSING")
    feed("mkformat.py", open(os.path.abspath(__file__), "rb").read())
    try:
        ver = subprocess.run(["pdflatex", "--version"], capture_output=True,
                             text=True, timeout=60).stdout.splitlines()[0]
    except Exception as exc:                      # noqa: BLE001 - reported, never swallowed
        ver = f"UNPROBED:{exc.__class__.__name__}"
    feed("pdflatex.version", ver.encode())

    # Item 7: the TeX-tree files the dump loaded. Absent on the first ever build (no log
    # yet), which is correct -- there is no dump to be stale.
    stat_lines = []
    for p in loaded_files(dump_log):
        try:
            st = os.stat(p)
            stat_lines.append(f"{p}\t{st.st_size}\t{int(st.st_mtime)}")
        except OSError:
            stat_lines.append(f"{p}\tmissing")
    feed("texfiles", "\n".join(stat_lines).encode())
    parts["texfiles:count"] = len(stat_lines)
    return h.hexdigest(), parts


def emit(src: str, fmtdir: str) -> dict:
    """Write _pre.tex, _body.tex and the three _run_<t>.tex drivers. Returns a small report."""
    os.makedirs(fmtdir, exist_ok=True)
    main_tex = read(os.path.join(src, "main.tex"))
    zero_main = read(os.path.join(src, "0_main.tex"))
    sw = switch_region(main_tex)
    pre, body, off = split_body(zero_main)

    header = ("%% GENERATED by src_utils/mkformat.py -- do not edit, do not commit.\n"
              "%% Derived from main.tex + 0_main.tex. Regenerated whenever fmt.key moves.\n")
    with open(os.path.join(fmtdir, "_pre.tex"), "w", encoding="utf8") as fh:
        fh.write(header)
        fh.write("%% part 1/2: main.tex's switch block (must precede the preamble;\n"
                 "%% dumping without it dies with Emergency stop inside abntex2cite).\n")
        fh.write(sw)
        fh.write("\n%% part 2/2: 0_main.tex, everything before \\begin{document}.\n")
        fh.write(pre)
        fh.write("\n\\endofdump\n")
    with open(os.path.join(fmtdir, "_body.tex"), "w", encoding="utf8") as fh:
        fh.write(header)
        fh.write("%% 0_main.tex from \\begin{document} to \\end{document}. Contains NO\n"
                 "%% preamble-only command by construction: it is the complementary byte\n"
                 "%% range of _pre.tex, cut at the same offset.\n")
        fh.write(body)
    for target, (stem, switches) in TARGETS.items():
        with open(os.path.join(fmtdir, f"_run_{target}.tex"), "w", encoding="utf8") as fh:
            fh.write(header)
            fh.write("%% Run-time driver. \\endofdump is \\relax once the format is loaded; it\n"
                     "%% is on line 3 so mylatexformat's preamble scanner stops HERE instead of\n"
                     "%% skipping this whole file looking for a \\begin{document} it has not got.\n")
            fh.write("\\endofdump\n")
            for s in switches:
                fh.write(s + "\n")
            fh.write("\\input{build/fmt/_body.tex}\n")
    return {"switch_bytes": len(sw), "preamble_bytes": len(pre),
            "body_bytes": len(body), "begindoc_offset": off,
            "targets": sorted(TARGETS)}


def status(src: str, fmtdir: str) -> tuple[bool, str]:
    fmt = os.path.join(fmtdir, "mainpre.fmt")
    keyfile = os.path.join(fmtdir, "fmt.key")
    if not os.path.exists(fmt):
        return False, "no format dump present"
    if not os.path.exists(keyfile):
        return False, "format dump present but fmt.key missing (provenance unknown)"
    want, parts = compute_key(src, os.path.join(fmtdir, "mainpre.log"))
    have = read(keyfile).strip().split()[0]
    if have != want:
        old = {}
        for line in read(keyfile).splitlines()[1:]:
            if "\t" in line:
                k, v = line.split("\t", 1)
                old[k] = v
        moved = [k for k, v in parts.items() if str(old.get(k)) != str(v)]
        return False, "stale: " + (", ".join(moved) if moved else "key mismatch")
    return True, "fresh"


def write_key(src: str, fmtdir: str) -> str:
    key, parts = compute_key(src, os.path.join(fmtdir, "mainpre.log"))
    with open(os.path.join(fmtdir, "fmt.key"), "w", encoding="utf8") as fh:
        fh.write(key + "\n")
        for k, v in parts.items():
            fh.write(f"{k}\t{v}\n")
    return key


def build(src: str, fmtdir: str, force: bool = False) -> int:
    emit(src, fmtdir)
    ok, why = status(src, fmtdir)
    if ok and not force:
        print(f"format: {why}, no rebuild needed")
        return 0
    print(f"format: rebuilding dump ({why})")
    cmd = ["pdflatex", "-interaction=nonstopmode", "-ini",
           f"-output-directory={fmtdir}", "-jobname=mainpre",
           "&pdflatex mylatexformat.ltx", "build/fmt/_pre.tex"]
    proc = subprocess.run(cmd, cwd=src, capture_output=True, text=True)
    log = os.path.join(fmtdir, "mainpre.log")
    errs = []
    if os.path.exists(log):
        errs = re.findall(r"^! .*", open(log, encoding="utf8", errors="replace").read(), re.M)
    if not os.path.exists(os.path.join(fmtdir, "mainpre.fmt")) or errs:
        print(f"format: DUMP FAILED (rc={proc.returncode}, tex_errors={len(errs)})")
        for e in errs[:5]:
            print("    " + e.strip()[:110])
        if not errs:
            print("    " + (proc.stdout or "")[-400:])
        return 1
    # The key is written AFTER a successful dump, and includes the file list this dump
    # actually loaded, so the first build's key is complete rather than provisional.
    key = write_key(src, fmtdir)
    print(f"format: built {fmtdir}/mainpre.fmt  key={key[:16]}  tex_errors=0")
    return 0


def selftest() -> int:
    """Both directions, per BRIEF.md: the split must reject a defective tree and accept a good one."""
    fails = []
    checks = 0
    # Fixture 1: the comment trap, exactly as it appears in this tree. A commented MENTION of
    # \begin{document} sits above the real one, so a comment-blind cut lands 2 lines early and
    # leaves \documentclass in the body half.
    good_zero = ("% \\begin{document} would be too late, says a comment\n"
                 "\\documentclass{abntex2}\n\\usepackage{graphicx}\n"
                 "\\begin{document}\nbody text\n\\end{document}\n")
    pre, body, off = split_body(good_zero)
    checks += 4
    if "\\documentclass" in body or "\\usepackage" in body:
        fails.append("split leaked a preamble-only command into the body half")
    if not body.startswith(BEGIN_DOC):
        fails.append("body half does not start at \\begin{document}")
    if find_live(pre, BEGIN_DOC) >= 0:
        fails.append("split leaked a LIVE \\begin{document} into the preamble half")
    if off == good_zero.find(BEGIN_DOC):
        # A comment-blind find() returns 2 here; the live matcher must return the later offset.
        fails.append("split cut at the COMMENTED mention (the comment trap is not closed)")

    # Fixture 2: the switch-region trap, which is the more dangerous half -- main.tex quotes
    # the switch pattern in a comment, and slicing from that offset makes an \input{main.tex}
    # live inside the dump.
    checks += 2
    tricky_main = ('%       pdflatex "\\newif\\ifdefensebuild\\defensebuildfalse\\input{main.tex}"\n'
                   "% ... prose ...\n"
                   "\\newif\\ifdefensebuild\n\\ifdefined\\FINALBUILD\\fi\n"
                   "\\newif\\ifapprovalsheet\n"
                   "\\newcommand{\\finalbuildfirstpage}{8}\n\\input{0_main.tex}\n")
    sw = switch_region(tricky_main)
    if "\\input{main.tex}" in sw:
        fails.append("switch_region matched the COMMENTED quote and would recurse into main.tex")
    if "\\input{0_main.tex}" in sw:
        fails.append("switch_region ran past \\finalbuildfirstpage into the \\input")

    # Both directions: a tree without a live anchor must FAIL LOUDLY, not return empty.
    checks += 2
    try:
        split_body("% \\begin{document}\n\\documentclass{a}\n")
        fails.append("split accepted a file with no uncommented \\begin{document}")
    except SystemExit:
        pass
    try:
        switch_region("\\documentclass{abntex2}\n")
        fails.append("switch_region accepted a main.tex without the anchors")
    except SystemExit:
        pass

    # The key must MOVE when the preamble moves and HOLD when only the body moves.
    with tempfile.TemporaryDirectory() as td:
        def write_tree(preamble_extra: str, body_extra: str) -> None:
            with open(os.path.join(td, "main.tex"), "w") as fh:
                fh.write("\\newif\\ifdefensebuild\n\\newif\\ifapprovalsheet\n"
                         "\\newcommand{\\finalbuildfirstpage}{8}\n\\input{0_main.tex}\n")
            with open(os.path.join(td, "0_main.tex"), "w") as fh:
                fh.write(f"\\documentclass{{article}}{preamble_extra}\n"
                         f"\\begin{{document}}\n{body_extra}\n\\end{{document}}\n")
            with open(os.path.join(td, "abntex2-UFV.sty"), "w") as fh:
                fh.write("% stub\n")
        write_tree("", "one")
        k0, _ = compute_key(td, os.path.join(td, "nolog.log"))
        write_tree("", "two -- a prose edit")
        k1, _ = compute_key(td, os.path.join(td, "nolog.log"))
        if k0 != k1:
            fails.append("key moved on a BODY-only edit (that would re-dump on every prose edit)")
        write_tree("\\usepackage{xcolor}", "one")
        k2, _ = compute_key(td, os.path.join(td, "nolog.log"))
        if k0 == k2:
            fails.append("key did NOT move on a PREAMBLE edit (a stale format would be used)")
        checks += 2

    for f in fails:
        print("FAIL: " + f)
    print(f"mkformat selftest: {checks - len(fails)}/{checks} checks pass")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="the LaTeX source dir (default: cwd)")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    src = os.path.abspath(a.src or os.getcwd())
    if not os.path.exists(os.path.join(src, "main.tex")):
        raise SystemExit(f"mkformat: no main.tex in {src} (pass --src)")
    fmtdir = os.path.join(src, "build", "fmt")
    if a.emit:
        rep = emit(src, fmtdir)
        print("format sources emitted: " + " ".join(f"{k}={v}" for k, v in rep.items()))
        return 0
    if a.status:
        ok, why = status(src, fmtdir)
        print(f"format: {why}")
        return 0 if ok else 1
    if a.build:
        return build(src, fmtdir, force=a.force)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
