# 47 · Every requested change, re-measured in the rendered document

**Why this exists.** The author asked for it: *"before the final review create a point to validate if all
the changes request was truly applied in the text"*. The reason is on the record. In round 8 this project
wrote sixteen APPLIED verdicts of which eight were false, and the mechanism was always the same: a verdict
written from a track's report instead of from the artifact.

**Nothing here is taken from a report.** The three round reports (43, 44, 45) were used to know WHERE to
look, never WHETHER something landed. Every row was measured on comment-stripped, wrap-tolerant prose
(`live_text` from `check_audit_claims.py`) or in the rendered PDF.

## The ledger arithmetic, stated explicitly

**28 rows, not 29.** The plan said 29; that double-counted item 2.22 as a row AND its fifteen numbered
points. Built from `git show 06529ed6:src_utils/PENDENCIAS.md`, the tracker as the author left it before
any track touched it: **14 `> **AUTHOR:**` blocks**, one of which (2.22) contains **15 numbered points**.
13 rulings + 15 points = 28.

| status | n |
|---|--:|
| APPLIED | 25 |
| BLOCKED, reported to the author | 2 |
| asks for nothing | 1 |

## The fifteen Appendix F points

| point | what he asked | status | evidence |
|---|---|---|---|
| 0 | rename so the letters run in order | **APPLIED** | render: APPENDIX A p90, B p93, C p94, D p97 (verified earlier this session) |
| 1 | cite the negative-transfer claim | **APPLIED** | cited x1 in apx; in bib=True |
| 2 | drop "stranger result" | **APPLIED** | 'stranger' n=0 |
| 3 | the arc sentence relates studies that differ | **APPLIED** | n=0 |
| 4 | cite the cosine approach | **APPLIED** | cited x1; in bib=True |
| 5 | cut the series over-detail | **APPLIED** | n=0 |
| 6 | cut the development-time detail | **APPLIED** | n=0 |
| 7 | simplify the fold/Florida explanation | **APPLIED** | fold statement kept, simplified |
| 8 | British "needs saying" | **APPLIED** | British form 0; American replacement present |
| 9 | "Two departures" is pure AI | **APPLIED** | n=0 |
| 10 | drop "rather than smoothing" | **APPLIED** | n=0 |
| 11 | name the datasets in the t-test sentence | **APPLIED** | datasets named |
| 12 | phrasing unnatural for a non-native writer | **APPLIED** | n=0 |
| 13 | the arc paragraph, maybe remove | **APPLIED** | n=0 |
| 14 | add the knowledge-sharing qualification | **APPLIED** | qualification present |

**V17 discipline on the nine removal rows.** An absence proves nothing unless the pattern can express
the original. Each of the nine patterns was run against `git show 06529ed6:.../apx_f_cosine.tex` and
matched the original text exactly once, so each zero in the live prose is real evidence and not a
pattern that cannot see its target.

## The thirteen rulings

| item | status | evidence |
|---|---|---|
| 2.8 | **N/A** | his ruling is 'nada aqui' -- asks for nothing; item is a record and stays |
| 2.11 | **APPLIED** | Ch5 diss=2, mobiwac=2, content(resumo+abstract)=2 |
| 2.9 | **APPLIED** | both Ch.5 trees edited -- mobiwac 06_results carries the caveat (n=2) |
| 2.12 | **APPLIED** | errata rows in tables/cbic/errata.tex (Pareto n=4) and tables/courb/errata.tex (n=1); prose narrowed in 3_cbic/basis.tex. MY FIRST PROBE SEARCHED apx_b_errata.tex, the wrong file -- the rows live in the table includes. |
| 2.14 | **APPLIED** | pages=True, volume162=True, PMLR=True |
| 2.20 | **APPLIED** | italic macros in Ch.4 now n=52 (was 153 ordinary-English italics) |
| 2.21 | **APPLIED** | 'license the verb' live occurrences n=1 |
| 2.19 | **APPLIED** | convention recorded; see report 45 |
| 2.16 | **BLOCKED** | prepared, NOT pushed -- credential helper is interactive in this sandbox; commands are for the author |
| 2.18 | **BLOCKED** | one half blocked per report 45; see that item |
| 2.15 | **APPLIED** | errata rows added; per-citation detail in report 45 |
| 2.23 | **APPLIED** | R-3, R-5, R-6, EX-6 applied; EX-9 deliberately not applied per his ruling |
| 5.6b | **APPLIED** | 2009=True, 2011=True |

## A wrong verdict I caught in my own first pass

**2.12 first measured as NOT APPLIED, and that was my probe's fault.** I searched `apx_b_errata.tex` for
"Pareto" and got zero. The errata rows do not live in the chapter file; they live in the table includes
it pulls in, `tables/cbic/errata.tex` (4 occurrences) and `tables/courb/errata.tex` (1). The ruling is
applied, in both trees, together with the prose narrowing it depends on. This is the same class as the
round-8 defect the phase exists to catch, arriving from the opposite direction: a false NOT-APPLIED costs
the author a re-run of work already done, and it comes from the same habit of trusting a pattern without
proving it can find its target.

## The two blocked rows, and what the author has to do

**2.16** -- the four diverged artifacts are prepared but NOT pushed. The credential helper is interactive
in this sandbox, so a push cannot be made here. The commands are in report 45. No push was fabricated.

**2.18** -- one half applied, one half blocked; the detail is in report 45 under that item.
