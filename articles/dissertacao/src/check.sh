#!/bin/bash
# Lint hook (TEMPLATE.md §2 item 10): the cheap half of gates G2/G3.
# Usage: ./check.sh   (run from src/; exits nonzero on any finding)
FAIL=0
CH=chapters/*.tex

echo "== em-dashes (WRITING_LAW §1: none anywhere in prose) =="
if grep -n $'\u2014' $CH 0_main.tex 2>/dev/null | grep -v '^\s*%'; then FAIL=1; else echo OK; fi

echo "== 'this paper' / 'this article' inside chapters =="
if grep -niE 'this (paper|article)' $CH | grep -v '^[^:]*:[0-9]*: *%'; then FAIL=1; else echo OK; fi

echo "== contractions =="
if grep -nE "\b(don't|doesn't|isn't|aren't|won't|can't|couldn't|wouldn't|shouldn't|it's|we're|they're|there's|hasn't|haven't|didn't|wasn't|weren't)\b" $CH | grep -v '^[^:]*:[0-9]*: *%'; then FAIL=1; else echo OK; fi

echo "== WRITING_LAW §4 banned words (prose lines only) =="
if grep -nwiE 'delve|delves|intricate|showcase|showcases|underscores?|pivotal|leverages?|leveraging|seamless|seamlessly|testament|moreover|furthermore' $CH | grep -v '^[^:]*:[0-9]*: *%'; then FAIL=1; else echo OK; fi

echo "== banned verdict verbs (beats/wins/ties as result verbs; crude sweep, review hits) =="
grep -nwiE 'beats?|wins?|Pareto' $CH | grep -v '^[^:]*:[0-9]*: *%' || echo OK

echo "== repo codenames =="
if grep -nwE 'B9|v1[1-7]|champion-G|H3-alt|dk_ovl|log_T|substrate' $CH | grep -v '^[^:]*:[0-9]*: *%'; then FAIL=1; else echo OK; fi

echo "== unresolved \\ref/\\cite (needs a compiled .log) =="
LOG=main_defense.log
if [ -f "$LOG" ]; then
  if grep -E 'Reference .* undefined|Citation .* undefined' "$LOG"; then FAIL=1; else echo OK; fi
else echo "SKIP: no $LOG"; fi

exit $FAIL
