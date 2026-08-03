# 51_commit_attribution_correction.md — two commit messages describe diffs they do not contain

Round 12, 2026-08-03. Found by re-reading my own commits after a backgrounded cell landed, not by the gate
suite, which cannot see this class at all.

## What is wrong

| commit | its message describes | its diff actually contains |
|---|---|---|
| `2f2021c9` | the AD-4 conditional closure only, and states "the header said five of six settled and now says five and one" | that, **plus** the AD-2 `ANSWERED 2026-08-03` row, AD-7, the §3 correction about the selected visit, and 75 new lines of `50_courb_temporal_level_investigation.md`. The `DEFINITIONS.md` it committed says **"All six gating decisions are now settled"**, contradicting its own message. |
| `6aca55e7` | the whole AD-2 finding, including the §3 correction and the investigation file | `PENDENCIAS.md`, `_round9/34`, `check_audit_claims.py`, the PDF. **Neither `DEFINITIONS.md` nor `50_`** — those had already landed in the previous commit. |

## What is NOT wrong, checked before writing any of this

- **The probe counts in both messages are correct.** I first read "100 of 100" against 99 probe rows and
  nearly reported a false discrepancy. The gate reports `len(PROBES) + 1` (the integrity-paragraph probe):
  99 + 1 = 100 at `2f2021c9`, 102 + 1 = 103 at `6aca55e7`. Verified by loading each commit's gate module.
- **The content.** `HEAD` carries every claim, the working tree is clean, all four builds and both suites
  are rc=0, and every probe holds. Nothing needs redoing.

## The cause

I dispatched a cell ending in `git add -A && git commit`, it was backgrounded, and I continued editing the
same files while it ran. **`git add -A` stages the tree at the moment the cell reaches that line, not at
dispatch.** So the commit boundary was set by timing and each message describes the work I had in mind
rather than the diff that landed.

## Why history is not being rewritten

`6aca55e7` already builds on `2f2021c9`, and a parallel agent shares this checkout. Rewriting shared
history to fix a message would risk their work to tidy mine. The correction lives here, where it is
auditable.

## The rule

**A commit message is a claim about a diff, and it is subject to the same evidence standard as any other
claim in this repo.** Two consequences:

1. **Never leave `git add -A && git commit` in a cell that may be backgrounded while editing continues.**
   Stage explicit pathspecs for the work being described, or make the commit the final action and wait for
   it before touching the tree again.
2. **`git add -A` is the wrong verb for an auditable commit.** It stages whatever happens to be dirty,
   including a parallel agent's in-flight files — which is how an earlier round clobbered another track's
   note. Explicit paths make the message and the diff the same object.

Why this matters here specifically: the whole point of recording the build commit against every measurement
is that a reading can be re-taken at the tree it was taken against. A message that misattributes its own
diff breaks exactly that, and no gate in `check.sh` can detect it, because the gates read the working tree
and never the commit log.
