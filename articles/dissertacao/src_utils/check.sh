#!/bin/bash
# Lint hook (TEMPLATE.md §2 item 10): the cheap half of gates G2/G3.
# Usage: make check  (or ../src_utils/check.sh from the src root). This script lives in
# src_utils/ (a SIBLING of src/); the LaTeX source it lints is in the sibling src/. Resolves
# that path from this script's location, so it works from any cwd. Exits nonzero on any finding.
FAIL=0
SRCROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"
cd "$SRCROOT"
CH=chapters/*.tex

echo "== em-dashes (WRITING_LAW §1: none anywhere in prose) =="
EMDASH=$(printf '\xe2\x80\x94')
if grep -n "$EMDASH" $CH 0_main.tex 2>/dev/null | grep -v '^[^:]*:[0-9]*: *%'; then FAIL=1; else echo OK; fi

echo "== 'this paper' / 'this article' inside chapters =="
if grep -niE 'this (paper|article)' $CH | grep -v '^[^:]*:[0-9]*: *%'; then FAIL=1; else echo OK; fi

echo "== contractions =="
if grep -nE "\b(don't|doesn't|isn't|aren't|won't|can't|couldn't|wouldn't|shouldn't|it's|we're|they're|there's|hasn't|haven't|didn't|wasn't|weren't)\b" $CH | grep -v '^[^:]*:[0-9]*: *%'; then FAIL=1; else echo OK; fi

echo "== WRITING_LAW §4 banned words (prose lines only; apx_b quotes published text and is exempt) =="
if grep -nwiE 'delve|delves|intricate|showcase|showcases|underscores?|pivotal|leverages?|leveraging|seamless|seamlessly|testament|moreover|furthermore' $CH | grep -v '^[^:]*:[0-9]*: *%' | grep -v '^chapters/apx_b_errata'; then FAIL=1; else echo OK; fi

echo "== banned verdict verbs (beats/wins/ties as result verbs; crude sweep, review hits) =="
grep -nwiE 'beats?|wins?|Pareto' $CH | grep -v '^[^:]*:[0-9]*: *%' || echo OK

echo "== repo codenames =="
if grep -nwE 'B9|v1[1-7]|champion-G|H3-alt|dk_ovl|log_T|substrate' $CH | grep -v '^[^:]*:[0-9]*: *%'; then FAIL=1; else echo OK; fi

echo "== unresolved \\ref/\\cite (needs a compiled .log) =="
# NOTE 2026-07-26: this check used a line-anchored grep. LaTeX WRAPS its warnings at 79 columns, so
# a wrapped "Citation `key' on page N undefined" was invisible to it and four undefined citations
# shipped in both PDFs. The log is flattened before matching, and the .blg is read too, because a
# BibTeX error (e.g. a bare at-sign inside a % comment in references.bib) never reaches the .log.
LOG=build/main.log
BLG=build/main.blg
if [ -f "$LOG" ]; then
  if tr -d '\n' < "$LOG" | grep -oE "(Reference|Citation) \`[^']+' on page [0-9]+ undefined"; then FAIL=1; else echo OK; fi
else echo "SKIP: no $LOG"; fi
if [ -f "$BLG" ]; then
  if grep -iE "error|didn't find|I was expecting" "$BLG"; then FAIL=1; else echo "OK (bibtex)"; fi
else echo "SKIP: no $BLG"; fi

echo "== prose trapped inside a % comment (silent: builds clean, reader sees a broken sentence) =="
# Has happened twice: apx_a_contributions.tex, and 4_courb.tex:187 where half a PUBLISHED
# methodology sentence was appended to a comment tail and three method facts stopped rendering.
DETECTOR=""
for cand in "src_utils/check_trapped_prose.py" "../src_utils/check_trapped_prose.py"; do
  if [ -f "$cand" ]; then DETECTOR="$cand"; break; fi
done
if [ -z "$DETECTOR" ]; then
  echo "SKIP: check_trapped_prose.py not found"
elif ! python3 "$DETECTOR"; then
  FAIL=1
fi

exit $FAIL
