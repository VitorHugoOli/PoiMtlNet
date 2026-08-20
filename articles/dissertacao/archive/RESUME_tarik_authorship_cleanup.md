# Resume note: Tarik authorship + cross-volume appendix cleanup

**Session:** `c462f106-439a-4dcd-9062-56a8b800687e` ("shared trunk")
**Status when checked (2026-08-06):** `completed`, 50 messages. It is NOT running, so there was
nothing to stop. Resuming means sending that session a new message; a fresh session would lose the
two decisions recorded below.

## Where it stopped

The last action was a `bash` step, "Re-verifying current state of authorship sentences after live
edits", which re-grepped:

- `articles/dissertacao/src/chapters/1_introduction.tex` for `Tarik` (plus lines 295-320 for context)
- `articles/dissertacao/src/chapters/4_courb.tex` for `Tarik`

So the edits had been started and it was checking what remained. **Re-run that grep first**: the tree
may have been edited live since, and the session's own view of it is stale.

## The two decisions it had just obtained (honor these; do not re-ask)

1. **Do not touch the bibliography.** The CoUrb entry's author field (Tarik S. Paiva, Vitor H. O.
   Silva, Germano B. dos Santos, Fabrício A. Silva) is the Crossref-verified record and stays exactly
   as it is, including where it prints in the back-matter References. Only the *prose* changes. The
   specific sentence to remove and re-prose is:

   > "Tarik S.\ Paiva is the first author. This dissertation's author is the second author,
   > contributed the MTLnet baseline, and presented the work at the event."

2. **Remove all cross-volume pointers**, not only broken ones. This includes the literal phrases of
   the form "Appendix B of the supplementary volume", which were a deliberate earlier fix (a `\ref`
   cannot span two separate LaTeX documents) and are guarded by a dedicated check. That guard will
   need attention once the pointers are gone.

## Remaining work

- Finish the authorship re-prose in `1_introduction.tex` and `4_courb.tex`.
- Remove the cross-volume appendix pointers tree-wide.
- Update or retire the check that guards the "Appendix X of the supplementary volume" phrasing,
  since its subject is being removed.
- Full build validation: `make check`, `make selftest`, and confirm no gate probe breaks.

## Interaction with the Check2HGI integrity study

None. The two touch disjoint paths:

| this cleanup | the integrity study |
|---|---|
| `articles/dissertacao/src/chapters/*.tex`, the bibliography, build checks | `scripts/integrity_v2/`, `docs/results/check2hgi_integrity_v2/`, `results/check2hgi_integrity_v2/`, and one new report in `articles/dissertacao/science/` |

The integrity study's compute runs on the remote GPU host and writes nothing under
`articles/dissertacao/src/`, so the two can proceed in either order or concurrently. The only shared
directory is `articles/dissertacao/science/`, where the study adds a new file rather than editing an
existing one.
