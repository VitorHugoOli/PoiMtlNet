#!/bin/bash
# fastbuild.sh -- one target, three passes, loading the precompiled preamble format.
#
#   usage: src_utils/fastbuild.sh <target> [srcdir]      target = defense | final | ppgc
#
# WHAT IT IS FOR. A three-pass build spends most of its time re-reading the same preamble
# (abntex2 + memoir + newtxmath + hyperref + abntex2cite) three times. This script loads the
# format dump that src_utils/mkformat.py produces instead. MEASURED cold, this session:
# 122.7 s -> 15.4 s for the defense build. Full table, with the contention caveat:
# src_utils/_round7/20_build_speed.md §3.
#
# WHAT IT IS NOT. It is an ACCELERATOR, never a requirement. `make defense`, `make final` and
# `make ppgc` do not use it and keep working with no format present -- that matters because
# Overleaf cannot load a local format dump, and because the plain path is what six rounds of
# verification are calibrated against. If anything about the format looks wrong, use `make`.
#
# THE STALENESS GUARD IS NOT OPTIONAL AND IS NOT ADVISORY. A format dump is silently wrong
# when the preamble changes under it: the build succeeds, the PDF is stale, and nothing in the
# log says so. That is the exact shape of science/AGENT_HANDOFF.md §2.3b (a PDF existing is
# not evidence the source is correct). So this script REFUSES to run on a stale key rather
# than warning about it; `make fast` calls mkformat.py --build first, which re-dumps when the
# preamble moved (measured 33.7 s and 36.0 s in this session's two runs) and does nothing when
# it did not.
#
# AUX ISOLATION. Each target writes into build/<stem>-aux/, its own output directory, and the
# three result files are copied back into build/ afterwards -- the same arrangement the
# Makefile's plain targets use, and for the same reason. All three targets used to share
# build/chapters/*.aux, so two concurrent builds corrupted each other's aux mid-write. Measured
# while writing this: a concurrent pass truncated build/chapters/4_courb.aux and the other
# build died with "Runaway argument? ... ! File ended while scanning use of \@writefile", which
# reads like a defect in chapter 4 and is nothing of the kind.
#
# NOTE for anyone tempted by -aux-directory: that flag is MiKTeX-only. TeX Live's pdflatex
# rejects it outright ("unrecognized option"), which is why the isolation is a per-target
# -output-directory plus a copy-back rather than the one-flag version.
set -u
TARGET="${1:?usage: fastbuild.sh <defense|final|ppgc> [srcdir]}"
SRC="${2:-}"
UTILS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # resolve BEFORE any cd (§2.5)
if [ -z "$SRC" ]; then SRC="$(cd "$UTILS/../src" && pwd)"; else SRC="$(cd "$SRC" && pwd)"; fi

case "$TARGET" in
  defense)   STEM=main ;;
  # `final` -> `academico` on 2026-07-29 (LATEX_UPGRADE.md §4 A-1). The old spelling is kept as
  # an accepted alias, not removed: this script is called by name from notes and by hand.
  academico) STEM=main_academico ;;
  # Normalize the alias, do not just map its stem: the run-time driver file below is named
  # build/fmt/_run_$TARGET.tex after mkformat.py's target keys, so leaving TARGET=final would
  # look for a _run_final.tex that mkformat no longer emits.
  final)     TARGET=academico; STEM=main_academico ;;
  ppgc)      STEM=main_ppgc ;;
  *) echo "fastbuild: unknown target '$TARGET' (defense|academico|ppgc)"; exit 2 ;;
esac

cd "$SRC" || exit 1
AUX="build/$STEM-aux"
mkdir -p build "$AUX/chapters" build/fmt

if ! python3 "$UTILS/mkformat.py" --status >/dev/null 2>&1; then
  echo "fastbuild: REFUSING to build -- $(python3 "$UTILS/mkformat.py" --status)"
  echo "           run: python3 ../src_utils/mkformat.py --build   (or: make fast)"
  exit 3
fi

RUN="build/fmt/_run_$TARGET.tex"
[ -f "$RUN" ] || { echo "fastbuild: $RUN missing; run mkformat.py --emit"; exit 3; }

# -halt-on-error, like the Makefile: it is the honest pass/fail signal. nonstopmode recovers
# from an error and still writes a PDF, which is how a broken source passed six builds.
PDF() { pdflatex -interaction=nonstopmode -halt-on-error \
          -output-directory="$AUX" -jobname="$STEM" \
          "&build/fmt/mainpre" "$RUN"; }

PDF || exit 1
BIBINPUTS=".:" TEXMFOUTPUT="$AUX" bibtex "$AUX/$STEM"; test $? -le 1 || exit 1
PDF || exit 1
PDF || exit 1

cp "$AUX/$STEM.pdf" "$AUX/$STEM.log" "$AUX/$STEM.blg" build/ || exit 1

# tex_errors, counted WITHOUT `grep -c ... || echo 0`.
#
# THE DEFECT THAT WAS HERE, and it inverted this script's whole verdict: `grep -c` exits 1 when
# the count is ZERO, so the `|| echo 0` fallback fired on every CLEAN build, ERRS became the two
# words "0 0", the `[ "$ERRS" = "0" ]` test failed, and fastbuild.sh exited 1 after a perfectly
# good build. Measured: `make fast3` reported "*** [fast3-academico] Error 1" and
# "Waiting for unfinished jobs" while all three PDFs were correct at 108/105/109 pages with
# tex_errors=0, and the equivalence check passed 3 of 3 against the serial reference. A build
# that succeeds while its runner reports failure is the mirror image of AGENT_HANDOFF §2.3b, and
# it is worse in one way: it trains the operator to ignore this script's exit code.
#
# `grep -c ... | tail -1` keeps grep's stdout and discards its status (the pipeline's status is
# tail's), so a zero count reads as the single word "0".
ERRS=$(grep -c '^! ' "build/$STEM.log" 2>/dev/null | tail -1)
PAGES=$(sed -n 's/.*Output written on [^ ]* (\([0-9]*\) pages.*/\1/p' "build/$STEM.log" | tail -1)
echo "fastbuild $TARGET -> build/$STEM.pdf  pages=$PAGES tex_errors=$ERRS aux=$AUX"
if [ -z "$PAGES" ]; then
  echo "fastbuild: NO PAGE COUNT in build/$STEM.log -- the build did not finish"
  exit 1
fi
[ "$ERRS" = "0" ] || exit 1
