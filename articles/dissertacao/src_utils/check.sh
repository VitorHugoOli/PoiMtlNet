#!/bin/bash
# Lint hook (TEMPLATE.md §2 item 10): the cheap half of gates G2/G3.
# Usage: make check  (or ../src_utils/check.sh from the src root). This script lives in
# src_utils/ (a SIBLING of src/); the LaTeX source it lints is in the sibling src/. Resolves
# that path from this script's location, so it works from any cwd. Exits nonzero on any finding.
FAIL=0
SRCROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"
UTILS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # absolute: this script cds into src/
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
  # Matching is done in Python with errors='replace', NOT with tr|grep. pdflatex writes raw bytes
  # from font and PDF metadata into the log (there is an invalid UTF-8 continuation byte at offset
  # ~75.6k in a normal build of this document), and `tr` aborts on it with "Illegal byte sequence"
  # under any UTF-8 locale -- which is the default here. Everything after that byte was therefore
  # never examined, and the abort went to stderr while the pipeline still reported success.
  # Verified by injecting a synthetic undefined-citation warning after the bad byte: found under
  # LC_ALL=C, silently missed under en_US.UTF-8. This is the very check that exists because four
  # undefined citations shipped in both PDFs.
  if python3 - "$LOG" <<'PYEOF'
import re, sys
raw = open(sys.argv[1], encoding='utf8', errors='replace').read()
flat = raw.replace("\n", "")                      # LaTeX wraps warnings at 79 columns
hits = sorted(set(re.findall(r"(?:Reference|Citation) `[^']+' on page \d+ undefined", flat)))
for h in hits:
    print(h)
sys.exit(1 if hits else 0)
PYEOF
  then echo OK; else FAIL=1; fi
else echo "SKIP: no $LOG"; fi
if [ -f "$BLG" ]; then
  if grep -iE "error|didn't find|I was expecting" "$BLG"; then FAIL=1; else echo "OK (bibtex)"; fi
else echo "SKIP: no $BLG"; fi

echo "== sweep-guard self-tests (a no-op substitution must not look like a result) =="
# Twice this project drew a conclusion from a parameter sweep whose arms never applied: once a doubled
# backslash made the target unmatchable, once a bad escape killed the substitution inside a heredoc.
# Both printed identical results across arms, which read as evidence. These tests pin both cases.
if ! python3 "$UTILS/sweep_guard.py" >/dev/null 2>&1; then
  echo "  FAIL: sweep_guard self-tests do not pass -- the guard itself is broken"
  python3 "$UTILS/sweep_guard.py" 2>&1 | tail -5
  FAIL=1
else
  echo "OK (4 self-tests)"
fi

echo "== recorded page counts vs the measured build =="
# These are PRESENT-TENSE claims about what is on disk (CLAUDE.md, PLAN.md, PENDENCIAS.md,
# codex_reviewer.md). They have drifted three times, always caught by review rather than by the
# edit that caused it. The codex page-drift note is load-bearing: it tells a reader how far every
# file:line in that review has moved.
if ! python3 "$UTILS/sync_page_counts.py"; then
  echo "  -> run: python3 src_utils/sync_page_counts.py --write"
  FAIL=1
fi

echo "== prose trapped inside a % comment (silent: builds clean, reader sees a broken sentence) =="
# Has happened twice: apx_a_contributions.tex, and 4_courb.tex:187 where half a PUBLISHED
# methodology sentence was appended to a comment tail and three method facts stopped rendering.
DETECTOR=""; FIXTURES=""
for d in "src_utils" "../src_utils"; do
  if [ -f "$d/check_trapped_prose.py" ]; then
    DETECTOR="$d/check_trapped_prose.py"; FIXTURES="$d/test_trapped_prose.py"; break
  fi
done
if [ -z "$DETECTOR" ]; then
  echo "SKIP: check_trapped_prose.py not found"
else
  # Run the detector's OWN fixtures first. The checker has been wrong four times, each time by
  # being tuned on the cases in front of it, so a green document means nothing unless the checker
  # still catches every defect that has actually shipped.
  if [ -f "$FIXTURES" ]; then
    # Capture the OUTPUT and the EXIT STATUS separately. Do NOT pipe into tail here: without
    # `set -o pipefail` a pipeline reports the status of its LAST command, so `python3 ... | tail -1`
    # always returns 0 and the failure branch below becomes unreachable. That defect shipped once,
    # in a commit that advertised this very block as a gate.
    FIXTURE_OUT="$(python3 "$FIXTURES" 2>&1)"
    FIXTURE_RC=$?
    echo "$FIXTURE_OUT" | tail -1
    if [ "$FIXTURE_RC" -ne 0 ]; then
      echo "  ^ detector fixtures FAILED — the checker itself is broken, its result is not evidence"
      echo "$FIXTURE_OUT" | grep '^FAIL' | sed 's/^/    /'
      FAIL=1
    fi
  fi
  if ! python3 "$DETECTOR"; then FAIL=1; fi
fi

exit $FAIL
