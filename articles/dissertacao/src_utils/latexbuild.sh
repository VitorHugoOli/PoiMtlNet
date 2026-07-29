#!/usr/bin/env bash
# latexbuild.sh <stem> <entry.tex> -- one plain (non-accelerated) build: three pdflatex passes
# with bibtex between the first and second, into an isolated aux tree, results published to build/.
#
# WHY IT EXISTS. The Makefile had this same eight-line recipe copy-pasted four times, differing only
# in the stem and the entry file. Four copies is four places for a fix to be applied three times.
# fastbuild.sh is the format-accelerated sibling; this is the plain path, and the two are deliberately
# symmetric so a reader who knows one knows the other.
#
# THREE THINGS HERE ARE LOAD-BEARING, each learned by breaking it:
#
#   1. ISOLATED AUX TREE (build/<stem>-aux/). Every target used to write the same
#      build/chapters/*.aux, so two concurrent builds left each other reading a truncated aux. The
#      victim died with "Runaway argument? ... ! File ended while scanning use of \@writefile",
#      which reads like a defect in whichever chapter it names and is nothing of the kind.
#
#   2. `test $? -le 1` AFTER BIBTEX, not `|| true`. bibtex exits 1 on warnings and 2+ on real
#      errors; `|| true` swallowed both, and four undefined citations shipped in two PDFs that way
#      (science/AGENT_HANDOFF.md §2.4).
#
#   3. ATOMIC PUBLISH. `cp` into build/ is not atomic, so a reader (sync_page_counts.py, check.sh,
#      another build) that opens build/<stem>.log mid-copy sees a TRUNCATED file. Observed at 12,288
#      bytes and at 0 bytes against a complete 35,586-byte aux log, while the page-count tool
#      reported "the build did not finish" about a build that had finished. Copy to a temp name in
#      the same directory, then rename: rename(2) is atomic within one filesystem.
#
# ONLY .pdf/.log/.blg are published. The .aux/.toc/.lof/.lot deliberately are NOT: an aux file
# sitting in a directory bibtex also searches is how the citations in (2) shipped.
set -euo pipefail

STEM="${1:?usage: latexbuild.sh <stem> <entry.tex>}"
ENTRY="${2:?usage: latexbuild.sh <stem> <entry.tex>}"
AUX="build/${STEM}-aux"
FLAGS=(-interaction=nonstopmode -halt-on-error -output-directory="$AUX")

mkdir -p build "$AUX/chapters"

pdflatex "${FLAGS[@]}" "$ENTRY"
BIBINPUTS=.: TEXMFOUTPUT="$AUX" bibtex "$AUX/$STEM" || test $? -le 1
pdflatex "${FLAGS[@]}" "$ENTRY"
pdflatex "${FLAGS[@]}" "$ENTRY"

for ext in pdf log blg; do
  [ -f "$AUX/$STEM.$ext" ] || continue
  cp "$AUX/$STEM.$ext" "build/.$STEM.$ext.$$"
  mv -f "build/.$STEM.$ext.$$" "build/$STEM.$ext"
done

PAGES=$(sed -n 's/.*Output written on [^ ]* (\([0-9]*\) pages.*/\1/p' "build/$STEM.log" | tail -1)
ERRS=$(grep -c '^! ' "build/$STEM.log" || true)
if [ -z "$PAGES" ]; then
  echo "latexbuild: NO PAGE COUNT in build/$STEM.log -- the build did not finish"
  exit 1
fi
echo "latexbuild $STEM -> build/$STEM.pdf  pages=$PAGES tex_errors=$ERRS  aux=$AUX"
