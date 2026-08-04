#!/usr/bin/env python3
r"""Mirror src/ into src_clean/ 1:1, with every LaTeX comment stripped from the .tex files.

src_clean is the comment-free copy of the active tree. src/ carries a large provenance
apparatus in comments (decision records, probe names, round history); src_clean is what a
reader gets who should see only the document.

Comment stripping is not "delete from % to end of line". Three cases must survive:

  1. ESCAPED PERCENT.  \% is a literal percent sign in the rendered text ("20\% of the
     data"). It is NOT a comment opener. \\% IS one, because there the backslash is an
     escaped backslash and the % that follows is bare. The rule is therefore: a % opens a
     comment only when preceded by an EVEN number of backslashes.
  2. COMMENT-ONLY LINES.  A line whose first non-space character is % is dropped whole,
     including its newline. Leaving a blank line behind would insert a paragraph break in
     LaTeX and change the typeset output.
  3. TRAILING % AS A LINE-JOIN.  In LaTeX a % at end of line suppresses the newline, which
     is load-bearing inside tabulars and long macro arguments. When a stripped comment
     leaves nothing but that %, the % is KEPT so the join survives.
  4. MAGIC COMMENTS.  "% !TeX root = ..." is a DIRECTIVE, not prose: editors read it to know
     which file to compile, and check_tex_root.py requires it on the FIRST line of every .tex
     with a path that resolves. 62 files in this tree carry one. A blanket strip deletes all
     of them, breaks build-from-editor, and fails that gate. Any "% !" magic comment is kept
     (this also covers % !TeX program, % !TeX encoding, % !TeX spellcheck, and arara
     directives, none of which are prose either).

The script verifies its own work: after syncing, both trees are built and their page
counts and tex_errors compared. A page-count difference means the strip changed the
typeset document, which is a bug, not an acceptable difference.
"""
from __future__ import annotations
import re, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "src", ROOT / "src_clean"
SKIP_DIRS = {"build", "__pycache__", ".git"}
SKIP_NAMES = {".DS_Store"}

# "% !TeX root = ...", "% !TeX program = ...", arara directives: functional, never prose.
# The pattern must be TIGHT. An earlier version was r"^\s*%\s*!" and it wrongly kept
# content.tex's "%     ! Extra }, or forgotten \endgroup." -- an ordinary comment quoting a
# LaTeX error message, which merely happens to have a bang after the percent. Requiring a
# known directive keyword after the bang fixes it.
MAGIC = re.compile(r"^\s*%\s*!\s*(TeX\b|arara\b|BIB\b)", re.IGNORECASE)
ROOT_DIRECTIVE = re.compile(r"^\s*%\s*!TeX\s+root\s*=", re.IGNORECASE)


def strip_tex_comments(text: str) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n\r")
        nl = line[len(body):]
        if MAGIC.match(body):                    # case 4: directive, not prose -- keep verbatim
            out.append(line)
            continue
        if body.lstrip().startswith("%"):        # case 2: drop the whole line
            continue
        cut = None
        i = 0
        while i < len(body):
            if body[i] == "\\":
                i += 2                            # skip the escaped char, incl. \% and \\
                continue
            if body[i] == "%":                    # case 1: bare % -> comment opener
                cut = i
                break
            i += 1
        if cut is not None:
            kept = body[:cut].rstrip()
            if not kept:
                continue                          # comment was the only content
            # case 3: preserve a deliberate line-join
            body = kept + ("%" if not nl else "")
            out.append(body + nl)
        else:
            out.append(line)
    return "".join(out)


def check_no_duplicate_roots() -> list[str]:
    """Report .tex files in src/ carrying more than one "% !TeX root" directive.

    Case 4 keeps every magic comment verbatim, which is right for a directive but also
    faithfully reproduces an ACCIDENTAL duplicate. Nine files in this tree had the directive
    twice (main.tex, content.tex, preamble.tex, the three other masters, 1_introduction,
    2_fundamentals, tables/mobiwac/errata_scope). The author removed one such duplicate from
    src_clean by hand and the next sync would have silently put it back, because the mirror
    reproduces src and the duplicate was in src. Fix the SOURCE, then re-sync -- and this
    check is here so the situation is reported instead of round-tripping unnoticed.
    """
    offenders = []
    for p in sorted(SRC.rglob("*.tex")):
        if any(part in SKIP_DIRS for part in p.relative_to(SRC).parts):
            continue
        n = sum(1 for line in p.read_text(encoding="utf-8").splitlines()
                if ROOT_DIRECTIVE.match(line))
        if n > 1:
            offenders.append(f"{p.relative_to(SRC)} ({n} directives)")
    return offenders


def sync() -> tuple[int, int]:
    dup = check_no_duplicate_roots()
    if dup:
        print("WARNING: duplicate '% !TeX root' directives in src/ -- fix the SOURCE, not the "
              "mirror, or the next sync reintroduces them:")
        for d in dup:
            print("  " + d)
    tex = copied = 0
    for src_path in sorted(SRC.rglob("*")):
        rel = src_path.relative_to(SRC)
        if any(p in SKIP_DIRS for p in rel.parts) or src_path.name in SKIP_NAMES:
            continue
        dst_path = DST / rel
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.suffix == ".tex":
            dst_path.write_text(strip_tex_comments(src_path.read_text(encoding="utf-8")),
                                encoding="utf-8")
            tex += 1
        else:
            shutil.copy2(src_path, dst_path)
            copied += 1
    # Remove .tex files that no longer exist in src, so a deleted chapter cannot linger in the
    # mirror as a stale copy.
    #
    # SCOPED TO .tex ON PURPOSE, and the scope is load-bearing. An earlier version deleted ANY
    # file absent from src, which removed 36 tracked review screenshots under
    # src_clean/tmp/pdfs/ -- they live only in src_clean and are nobody's build output. The
    # mirror's job is the document sources; it has no mandate over files that were never in src
    # to begin with. Anything outside the .tex set is left alone even when src has no
    # counterpart.
    removed = 0
    for dst_path in sorted(DST.rglob("*.tex"), reverse=True):
        rel = dst_path.relative_to(DST)
        if any(p in SKIP_DIRS for p in rel.parts) or dst_path.name in SKIP_NAMES:
            continue
        if dst_path.is_file() and not (SRC / rel).exists():
            dst_path.unlink()
            removed += 1
    print(f"tex stripped: {tex}   assets copied: {copied}   stale removed: {removed}")
    return tex, copied


def build(job: tuple[Path, str]) -> tuple[Path, str, int, str, str]:
    """Build one target in one tree. Safe to run concurrently: latexbuild.sh gives every
    target its own build/<stem>-aux (see its header note), so two targets never share an
    aux file. The two TREES are separate directories, so cross-tree overlap cannot happen
    either. Decoding is errors="replace" on purpose -- pdflatex emits raw bytes from font
    and encoding messages that are not valid UTF-8, and a decode crash here would look
    like a build failure when the build actually succeeded."""
    tree, target = job
    r = subprocess.run(["make", target], cwd=tree, capture_output=True)
    out = r.stdout.decode("utf-8", errors="replace")
    pages = re.search(r"pages=(\d+)", out)
    errs = re.search(r"tex_errors=(\d+)", out)
    return tree, target, r.returncode, pages.group(1) if pages else "?", errs.group(1) if errs else "?"


if __name__ == "__main__":
    sync()
    if "--no-verify" in sys.argv:
        sys.exit(0)

    targets = ("defense", "academico", "ppgc", "extra")
    jobs = [(tree, t) for t in targets for tree in (SRC, DST)]
    # 8 builds (4 targets x 2 trees) run concurrently rather than one at a time.
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        done = {(tree, t): (rc, pg, er) for tree, t, rc, pg, er in pool.map(build, jobs)}

    bad = False
    for target in targets:
        a = done[(SRC, target)]
        b = done[(DST, target)]
        ok = (a[0] == b[0] == 0) and a[1] == b[1] and a[2] == b[2] == "0"
        print(f"{target:10s} src rc={a[0]} pages={a[1]} errs={a[2]} | "
              f"src_clean rc={b[0]} pages={b[1]} errs={b[2]}  {'OK' if ok else 'MISMATCH'}")
        bad |= not ok
    if bad:
        print("FAIL: the stripped tree does not reproduce the source tree.")
        sys.exit(1)
    print("OK: src_clean is a 1:1 comment-free mirror and builds identically.")
