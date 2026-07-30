# 34_tracker_disagreement.md — the two trackers contradicted each other, and no gate saw it

Round 9, 2026-07-30. Found by an independent reviewer, not by me and not by the suite.

## The defect

`PENDENCIAS.md` §2.8 asserted **ten** stale anchors, "**todas** citam `0_main.tex`", and "As 21
ancoras dos capitulos 1, 2 e 6 conferem todas". Three screens below, §6 of the **same file** asserted
**nine**, and so did every statement in `CONSIDERATIONS.md`. Both cannot be true. The corrected
measurement is nine: 32 exact, 4 changed, 5 gone of 41 locatable anchors, and 20 of Fabrício's
anchors sit outside `0_main.tex`.

## The mechanism, which is the part worth keeping

I corrected the count in the right place and published the wrong one anyway, and the reason is
mechanical rather than careless.

The cell that was supposed to rebuild §2.8 from the corrected figures **aborted on an
`AssertionError` before it reached the reassignment.** The assertion was mine and it was doing its
job, but it was guarding a claim in `CONSIDERATIONS.md`, and it fired on a line-wrap false negative,
so the cell died at line 5 of 30. The variable holding §2.8's text kept its previous value: the
pre-correction text. Two cells later I wrote that variable to disk and read back a plausible
byte count, which is exactly the shape of confirmation that V8 warns about ("shape is satisfied by an
empty result" -- here, by a stale one).

So the failure is not "I forgot to fix it". It is: **an assertion protecting file A aborted the cell
that was rewriting file B, and the write of B still happened, from stale state, in a later cell.**
The kernel keeps variables across cells; a failed cell leaves the previous values in place and
nothing marks them as stale.

Two defenses, and the second is the general one:

1. **Re-read the block from disk immediately before writing it,** and diff the replacement against
   `git HEAD` rather than against whatever is in memory. This is the same rule the clobber of
   `c94d1f19` produced two hours earlier; that repair recorded it and this defect still slipped
   through, because I applied it to the file I had clobbered and not to the file I had regenerated.
2. **A number with two homes needs a gate that compares the homes.** Every probe in
   `check_audit_claims.py` checked one file against itself, so twenty-plus green gates were all
   correct and all blind: the two figures lived in different files (and then in different sections of
   one file), and nothing asserted they agreed.

## The gate

`R9-agree` pins the corrected figure in `PENDENCIAS.md` §2.8 positively; `R9-agree2` bans the
superseded 21-anchor sentence (inverted, because the fix's correctness is an absence).
`R9-stale` already pins the same corrected count in `CONSIDERATIONS.md`, so **the two files can only
pass together** -- which is the property that was missing.

Validated by sabotage, each leg flagging only its own row, restored byte-identical:

| leg | mutation | result |
|---|---|---|
| `R9-agree`  | 2.8's count reverted to "31 sao exatas e 10 estao obsoletas" | NOT APPLIED, rc nonzero |
| `R9-agree2` | the "As 21 ancoras dos capitulos" sentence reintroduced | NOT APPLIED, rc nonzero |

Clean run: 26 of 26 probes hold, rows reconcile (26 + 5 unprobed + 1 retired = 32 printed).

## What this says about the round's other numbers

Every figure in these two trackers is generated from one measurement pass, so the same class of
defect could have hit any of them. I re-grepped both files for the superseded values after the fix:
`10 estao obsoletas`, `As dez obsoletas`, `As 21 ancoras`, `31 sao exatas` now appear **nowhere**
except inside the two paragraphs that explicitly record the correction. The corrected figures appear
in three places and agree.


---

## Second instance, same shape: a wave count that stopped counting edits

Found by the reviewer, not by me and not by the suite.

**The defect.** Wave A had 8 members and I reported 8 items "applied and verified in the
rendered PDF, both directions". One member, **FAB-01**, changed nothing: it asked for the advisor line
in English, the line was already there, and the diff against `c39b7b77` contains no advisor line added
or removed. The loop that stamped the verification note walked `sorted(wave_a)` and wrote the same
sentence onto all 8 blocks, so FAB-01 carried a claim that the old wording was **absent** from the
render. There is no old wording for a no-op. That half of the claim could not have been measured, and
it was not.

**The mechanism, which is the same one as above.** Both defects are a headline number that stopped
agreeing with the members it summarizes:

| | the number | what it actually counted |
|---|---|---|
| earlier | ten stale anchors, "todas citam `0_main.tex`" | nine did; the tenth was in a different file |
| here | 8 items applied and verified both directions | 7 were; the eighth changed nothing |

In both cases the per-item data was correct and sitting in the same structure the summary was
generated from. What failed was that the summary was written from the **container** (`len(wave_a)`,
the count of stale rows) instead of from the **property being claimed** (items that carry an edit;
anchors that cite the deleted file). A loop over a set applies one predicate to every member, which is
precisely wrong when the members differ in the property the sentence asserts.

**The rule this yields, and it is narrower and more useful than "check your counts":** before writing
a count into prose, re-derive it from the predicate in the sentence, not from the collection the items
came from. If the sentence says "verified in both directions", count the items for which both
directions exist.

**The gate.** `R9-confirm` pins FAB-01's carve-out text, so a later sweep cannot restore the
applied-and-verified note to a no-op. Validated by sabotage: replacing the carve-out with the standard
note turns the gate red (rc 0 -> 1) and flags only that row; restored byte-identical; clean run is
27 of 27.

**Scope check on the same class.** The other Wave A members were each re-read: all 7 carry a real
source change in the diff and a two-directional render assertion. Waves B and C claim nothing applied.
