# REVIEWER v1 (corrected) — adversarial holistic review of the corrections-round-2 build

> **Operator:** Dissertation Reviewer (adversarial read-only review operator).
> **Build under review:** `articles/dissertacao/src/dissertacao.pdf` (defense, 87 pp) and
> `articles/dissertacao/src/build/main_final.pdf` (AcademicoPG, 83 pp), both built 2026-07-23 23:20
> from source last touched 2026-07-23 23:17. **Run date:** 2026-07-24.
> **Mode:** holistic pass, sampling the persona suite (04 concordance, 05 citation, 06 number,
> 07 claim/honesty, 13 UFV, 18 visual) against the round-2 change set, plus a full regression sweep.
> **Charge:** did corrections round 2 introduce any regression (broken cross-ref, new inconsistency
> from the heading/title/B.1/CBIC edits, src-restructure fallout), and are the previously-queued
> cross-chapter seams better or unchanged?
> **Protocol:** read-only; fail-closed; every finding carries a verbatim quote + file:line or PDF
> page; nothing from memory; numbers traced to the chapters' own sources of truth, quoted not
> computed. This report was written incrementally.

---

## 0 · Fail-closed caveats (what this run could and could not verify)

- **No LaTeX toolchain in the review sandbox.** `pdflatex`/`latexmk`/`bibtex` are absent, so I
  **could not independently recompile** either build. All build-health statements below
  (page counts, "0 undefined refs/cites") are verified against the **committed** `build/main.log`
  and `build/main_final.log` and by rendering the committed `dissertacao.pdf` with pypdfium2 —
  not by re-running the build. If the author wants a build-integrity guarantee, `make check` +
  a clean compile must be re-run on a TeX host. Flagged UNVERIFIED, not asserted.
- **The one writable file this run produced is this report.** Everything else was read-only.
- The `[VERIFY]`/`[NEEDS SIGN-OFF]` LaTeX comments throughout the source are author-action markers,
  not defects; I confirmed each is a comment (invisible in the render), and treat the underlying
  items as the author's open queue, not review findings.

---

## 1 · Verdict

**APPROVED WITH CORRECTIONS.** Round 2 did its job: both original blockers (open title, CBIC
dataset placeholders) are resolved in the render, the B.1 misattribution fix is faithful and
correctly mirrored, and the source restructure produced no broken cross-reference or path fallout.
The science is untouched and the arc still reads as one investigation.

Two **MAJOR** items are round-2 *fallout* — both are documentation/concordance seams created by
the asymmetric way the corrections landed, neither is a fabricated or wrong number, neither touches
the experiments:

1. **F1 (regression).** The CBIC dataset counts were filled in Chapter 3 but the Appendix B
   erratum row that tracked them was **not** updated: Appendix B still tells the reader the counts
   are unfilled placeholders "pending recomputation." The document now contradicts itself about
   whether the counts exist.
2. **F2 (new seam).** Filling Chapter 3's Florida counts made a previously-hidden cross-chapter
   number divergence **visible**: Florida is 1,407,034 check-ins in Chapters 3 and 5 but 990,518
   in Chapter 4, with no reader-facing reconciliation anywhere. Each number is faithful to its own
   chapter's source; the defect is that the coletânea visibly disagrees with itself on the size of
   the same state's data and never says why.

Two **MINOR** items are pre-existing seams that round 2 did not touch (F3, F4). No blocker remains
that is a review finding; the remaining blockers are author sign-off items already in the queue.

---

## 2 · Round-2 change set: regression audit (each change verified in render + source)

### 2.1 Title set at all echo points — VERIFIED CLEAN, blocker resolved

The working title *"From Representations to a Single Joint Model: Multi-Task Learning for
Point-of-Interest Category and Region Prediction"* is live at every echo point, and the old
`[TITLE — open decision]` placeholder is gone from the render.

| Echo point | Source | Render |
|---|---|---|
| `\titulo` (folha de rosto) | `0_main.tex:115` | PDF p.1, renders as 2 lines, no defect |
| Resumo catalog header | `0_main.tex:174` | PDF p.3 |
| Abstract catalog header | `0_main.tex:236` | PDF p.4 |
| hyperref `pdftitle={\@title}` | `0_main.tex:132` (inherits `\titulo`) | metadata inherits, no separate string to drift |

The three alternates are parked as comments (`0_main.tex:109-114`); the selection note at
`0_main.tex:95-104` correctly records this as a "for now" choice pending the advisor. No orphaned
placeholder anywhere. **BL-2 resolved.** (Author action remaining: the final advisor call — not a
review finding.)

### 2.2 Chapter 3/4/5 headings shortened — VERIFIED CLEAN, no cross-ref leak

Each article chapter now carries a short one/two-line heading, with the full published title moved
into the chapter preface:

- Ch.3 `chapters/3_cbic.tex:11` — `\chapter[Multi-Task Learning for POI Category and Next-POI
  Prediction]{...same...}`; full title + DOI 10.21528/CBIC2025-1191324 in the preface (L15-18).
- Ch.4 `chapters/4_courb.tex:9` — `\chapter[ST-MTLNet: Spatio-Temporal POI Representations]{...for
  Multi-Task Learning}`; full PT title + DOI 10.5753/courb.2026.22960 + pages 323-336 + Tarik/Vitor
  authorship note in the preface (L13).
- Ch.5 `chapters/5_mobiwac.tex:15` — `\chapter[A Check-in-Level Multi-Task Study of Next Category
  and Region]{...same...}`; full title + EDAS #1571313639 + "submitted ... under review" in the
  preface (L22-24).

**Regression checks passed:**
- **TOC (PDF p.9-10):** every chapter title now fits one line — Ch.3 → p.25, Ch.4 → p.41, Ch.5 →
  p.56. The header-padding + TOC-wrap defect (persona-18 MJ-12) is resolved.
- **Running headers (PDF p.26, p.42):** now single-line ("Chapter 3. Multi-Task Learning for POI
  Category and Next-POI Prediction"; "Chapter 4. ST-MTLNet: Spatio-Temporal POI Representations").
- **No cross-reference leak.** The document uses **only numeric `\ref`** — a full sweep found
  **zero** `\nameref`/`\autoref`/`\Cref`/`\titleref` in any chapter or `0_main.tex`. A shortened
  heading therefore cannot surface a truncated title inside any cross-reference. The Appendix B
  section titles embed a live `\ref` (`apx_b_errata.tex:38` `Article 1 (Chapter~\ref{ch:cbic}...)`)
  which resolves at the final pass (build log: 0 undefined refs), so the TOC line 95 renders
  "Chapter 3", not a raw label.

### 2.3 B.1 CBIC misattribution fix — VERIFIED FAITHFUL, mirrored, and documented

This was the highest-risk edit to re-audit: a correction that could easily re-introduce a softened
form of the very error it removed. It does not.

**The corrected sites (dissertation Ch.5):**
- Intro, `chapters/5_mobiwac.tex:45`: "...work reported no consistent multi-task advantage for the
  paired category tasks and attributed it, in part, to this effect~\cite{silva2025mtlnet}..."
- Related-work recap, `chapters/5_mobiwac.tex:145-148`: "Our earlier work~\cite{silva2025mtlnet}
  paired static category classification with next-category prediction and found no consistent
  multi-task gain; this chapter introduces the next-region task..."

**Faithfulness check against the CBIC record (the Ch.3 source of truth).** The concern was whether
"attributed it, in part, to this effect [negative transfer]" smuggles back the killed "CBIC
*observed* negative transfer" claim. It does not, because CBIC's own conclusion **does** attribute
its null, in part, to negative transfer — as a *hypothesis*, not an observation:
- `chapters/3_cbic.tex:372`: "We hypothesize three primary factors contributed to this outcome:
  [1] **Subtle Negative Transfer due to Task Dissimilarity** ... The shared encoder may have been
  forced to learn a 'compromise' representation that was not specialized enough for either task..."

So Ch.5's "attributed ... in part" is faithful to a hypothesized cause, and the "this chapter
introduces the next-region task" clause correctly restores that CBIC had **no** region task
(`3_cbic.tex` pairs *static POI category classification* with *next-POI/next-category prediction*;
model named "our MTL model"). The two original errors (region task; observed-vs-hypothesized) are
both gone and neither returns. **B.1 fix sound.**

**Mirror + documentation, all three consistent:**
- Version of record: `articles/[mobiwac]/src/sections/01_introduction.tex:17` (same corrected
  sentence, "this paper") and `.../02_related.tex:48-49`.
- `articles/[mobiwac]/ERRATA.md:23-35` logs both errors and the correction, "applied both here and
  in the version of record."
- Dissertation Appendix B §B.3, `chapters/apx_b_errata.tex:176-192`, states the correction, names
  both errors, and confirms it "was applied both here and in the version of record."

### 2.4 CBIC Florida counts filled — VERIFIED in render (see F1, F2 for the fallout)

`chapters/3_cbic.tex:238` now renders real numbers (PDF p.35): "This subset comprises 21{,}052
users and 76{,}544 unique Points-of-Interest (POIs) across 1{,}407{,}034 check-ins; after
discarding users with fewer than five visits, as done for the next-POI task, 13{,}935 users and
76{,}266 POIs across 1{,}392{,}262 check-ins remain." The both-bases wording removes the need for a
basis choice and matches `cbic_recompute_result.md`. The `[VERIFY]` markers (L239-248) are LaTeX
comments (invisible). **BL-1 resolved in render** — subject to the author confirming the recompute
basis (an author action already queued), and to F1/F2 below.

### 2.5 src/ restructure (non-LaTeX → src_utils/, output → build/) — VERIFIED CLEAN

- Entry chain intact: `main.tex:34` `\input{0_main.tex}`; `0_main.tex:340-345` `\include` the six
  chapters; `:357-359` the three appendices; `:353` `\bibliography{references}`. All resolve.
- All `\includegraphics` use paths relative to `src/` that survived the move (`figures/...`,
  `figures/courb/...`, `figures/mobiwac/...`) — 8 figures, no broken path.
- `Makefile` + `src_utils/check.sh` resolve paths from the src root (`check.sh` computes
  `SRCROOT` from its own location), so the move did not break the lint hook.
- Build currency: newest source `3_cbic.tex` 2026-07-23 23:17; PDFs built 23:20 — the render is
  newer than every source. The git "M" on `src/dissertacao.pdf` is the rebuilt binary differing
  from the committed blob, not a stale-build signal.

---

## 3 · Findings (ranked)

### F1 — MAJOR (regression). Appendix B still calls the CBIC counts unfilled placeholders

Round 2 filled the counts in Chapter 3 (§2.4 above) but left the Appendix B erratum row that
tracked them unchanged. The two sites now contradict each other:

- **Chapter 3 (filled), `chapters/3_cbic.tex:238`:** "This subset comprises 21{,}052 users and
  76{,}544 unique Points-of-Interest (POIs) across 1{,}407{,}034 check-ins..."
- **Appendix B (still says pending), `chapters/apx_b_errata.tex:76-80`**, Table B.1 (CBIC content
  errata), last row:
  > "Unfilled dataset placeholders ($N_{\text{users}}$, $N_{\text{poi}}$, $N_{\text{checkins}}$) in
  > the results section. & *Pending.* Not invented: the chapter renders visible placeholders; the
  > values await recomputation by a repository-committed script over the article's data pipeline,
  > with author approval."

Both render (Appendix B is on PDF p.83). A banca reader who reaches Appendix B is told the chapter
shows placeholders and awaits recomputation — after having read the filled numbers on p.35. This is
exactly the "silent correction / missing errata trail" failure the errata policy exists to prevent,
inverted: the chapter was corrected and the errata ledger was not caught up.

**This is round-2 fallout, not pre-existing:** before round 2 the placeholder row was *accurate*
(the chapter did render placeholders). The CBIC-fill commit (a9ba3929) updated `3_cbic.tex` and
`cbic_recompute_result.md` but not `apx_b_errata.tex`.

**Author action (not a reviewer edit).** Options, for the author to rule on:
(a) if the recompute basis is confirmed, rewrite the row to record it as a *resolved* erratum
("the published article left the three dataset statistics unfilled; the chapter states the
recomputed Florida corpus, 21,052 users / 76,544 POIs / 1,407,034 check-ins, and the
after-filter counts, recomputed from the sanctioned per-state ETL output"); or
(b) if the basis is still unconfirmed, reword to match the true current state ("the chapter states
recomputed counts pending the author's confirmation of the recompute basis") so ledger and chapter
agree. Either way, Appendix B must stop asserting the chapter renders placeholders.

### F2 — MAJOR (new visible seam). Florida check-in count disagrees across chapters, unreconciled

Filling Chapter 3 exposed a cross-chapter number divergence that the placeholder previously hid.
For the **same U.S. state (Florida) of the same Gowalla dataset**, three chapters now render:

| Chapter | Florida check-ins | Florida POIs | Florida users | Source of truth |
|---|---|---|---|---|
| Ch.3 (CBIC), `3_cbic.tex:238` (PDF p.35) | **1,407,034** | 76,544 | 21,052 | per-state ETL, both-bases |
| Ch.5 (MobiWac), `5_mobiwac.tex:324` (Table, PDF p.63) | **1,407,034** | 76,544 | 21,052 | RESULTS_BOARD / ETL |
| Ch.4 (CoUrb), `4_courb.tex:237` (Table 5, PDF p.51) | **990,518** | 65,009 | 20,301 | CoUrb published table |

Ch.3 and Ch.5 **agree to the digit** (a *positive* concordance outcome — the CBIC fill was drawn
from the same ETL the MobiWac chapter uses, so the two now match where before Ch.3 was blank).
Ch.4 differs because CoUrb applied its own task-specific filtering (valid category label,
window size 9, embedding coverage), which drops rows and POIs — this is explained in
`cbic_recompute_result.md` and in a LaTeX comment, but **only there**. A reader of the PDF sees
1,407,034 on p.35 and p.63, then 990,518 on p.51, with **no sentence anywhere reconciling them**
(doc-wide search for a reader-facing "different filter / different pipeline / counts differ"
reconciliation returned nothing in rendered prose).

**Severity rationale.** This is not a fabricated or wrong number — each value is faithful to its
own chapter's published source, and per-chapter fidelity is exactly what the coletânea errata
policy requires (Ch.4 must reproduce CoUrb's published table; Ch.3 must state its own corpus).
The defect is **concordance (persona 04)**: a coletânea that visibly reports two different sizes
for the same state's data, with no note, reads as an internal inconsistency to a banca even though
both are individually correct. A single sentence resolves it.

**Author action (not a reviewer edit).** A one-line reconciliation where the second-rendered
Florida count first appears (a footnote or parenthetical on Ch.4's Table 5, or a sentence in the
frame), stating that the article chapters reproduce each source paper's own preprocessing so the
same state's totals differ across chapters (Ch.4's CoUrb pipeline filters to task-usable check-ins;
Ch.3/Ch.5 report the fuller corpus). Do not alter either published table value. This is a
whitelist/errata decision because it touches a published table's surrounding prose.

### F3 — MINOR (pre-existing, unchanged by round 2). "MTLnet" is named in the frame but never in Chapter 3

The frame and the later chapters name the first model "MTLnet" and point to Chapter 3 as its
origin, but **Chapter 3 never uses the name** (0 occurrences of "MTLnet"/"MTLNet" in `3_cbic.tex`;
it says "our MTL model", "our proposed model", "the MTL framework").

- Names it and points back: `1_introduction.tex:102` ("MTLnet, from standard components"),
  `1_introduction.tex:236` ("The MTLnet framework (Chapter~\ref{ch:cbic})"),
  `5_mobiwac.tex:93` ("Chapter~\ref{ch:cbic} introduced MTLnet, the first..."),
  `6_conclusion.tex:25` ("Chapter~\ref{ch:cbic} contributed MTLnet, the first joint model").
- Chapter 3 itself: no name. A reader who follows "MTLnet, introduced in Chapter 3" to the chapter
  finds no such name defined there.

This is the queued concordance seam MJ-18 (persona 04, round 1). Round 2 did not touch it.
**Author action:** either introduce the name once in Ch.3 (e.g., in the preface or §3.3.2, "we
refer to this framework as MTLnet"), or soften the frame's back-pointers to "the framework of
Chapter 3". A Ch.3-preface introduction is the lighter touch and preserves the published body.

### F4 — MINOR (casing, low priority). Chapter 4 heading uses "MTLnet" against its own declared rule

Chapter 4 explicitly declares that it preserves the published paper's "MTLNet" casing:
`4_courb.tex:83` — "the published paper typesets the name as MTLNet, and this chapter preserves
that form." The chapter body then uses "MTLNet" 46 times, consistent with that rule. But two of its
own **section headings / running text** break it in the other direction:

- `4_courb.tex:80` `\subsection{The MTLnet framework}` (lowercase n) — renders in the TOC as
  subsection 4.2.5 "The MTLnet framework" (PDF p.10).
- `4_courb.tex:116` `\subsection{Baseline: MTLNet with DGI}` (capital N) — TOC 4.3.1.

So within one chapter the heading at L80 uses the dissertation-canonical "MTLnet" while the chapter
declares it preserves "MTLNet", and the heading at L116 uses "MTLNet". This is cosmetic and does not
affect a claim or number, but it is a visible inconsistency in the TOC. **Author action:** pick one
form for Chapter 4's headings consistent with the L83 preservation note (most likely "MTLNet" to
match the chapter's declared rule), or, if the frame convention wins, restate L83. Not blocking.

---

## 4 · Previously-queued seams: better or unchanged? (the charge's second question)

Re-derived against the current build (not echoed from the round-1 consolidated report).

| Queued item (round 1) | Round-2 status | Evidence |
|---|---|---|
| **BL-1** CBIC dataset placeholders render on p.35 (BLOCKER) | **BETTER — resolved in render** | `3_cbic.tex:238` filled; PDF p.35 shows real numbers. Subject to author basis-confirm + F1. |
| **BL-2** Title renders as `[TITLE — open decision]` (BLOCKER) | **BETTER — resolved** | `0_main.tex:115/174/236`; PDF p.1/3/4 clean. |
| **MJ-12** header padding + TOC wrap (persona 18) | **BETTER — resolved** | short headings `3_cbic.tex:11` etc.; TOC one line/chapter p.9-10; running headers one line p.26/p.42. |
| **B.1** CBIC misattribution in Ch.5 (persona 10 BLOCKER / 14) | **BETTER — resolved + mirrored + documented** | `5_mobiwac.tex:45,145`; `[mobiwac]/src` intro L17 + related L48; ERRATA L23; AppB §B.3. Faithful (§2.3). |
| **N-2 / MJ-1** Alabama 64.51 vs 64.54 blur | **BETTER — reconciled to 64.51** | `5_mobiwac.tex:485` Table 3 AL Joint = 64.51; `6_conclusion.tex:78` = 64.51; dedicated 56.82 / capacity 56.16 consistent. |
| **MJ-18** "MTLnet" named in frame, absent in Ch.3 | **UNCHANGED** | F3. Frame names it (`1_introduction.tex:102`, `5_mobiwac.tex:93`, `6_conclusion.tex:25`); `3_cbic.tex` = 0 occurrences. |
| **MJ-5** data vintage "2009 and 2010" vs 2009-2011 | **UNCHANGED in prose; basis recorded** | `6_conclusion.tex:114` still "collected in 2009 and 2010"; `cbic_recompute_result.md` records the measured span 2009-2011. Author decision in `DECISOES_PENDENTES_ptBR.md`. |
| Florida count divergence across chapters | **WORSE (now visible)** | F2. Was hidden while Ch.3 was a placeholder; the fill exposed 1,407,034 (Ch.3/Ch.5) vs 990,518 (Ch.4) with no reader-facing note. |

Net: five queued items improved (two blockers cleared, three seams reconciled), two are unchanged
minors (F3, MJ-5), and one seam became visible and needs a one-line reconciliation (F2). No queued
item regressed in the sense of a correct thing becoming wrong; the only new exposure (F2) is a
concordance-visibility problem, not a number error.

---

## 5 · Gate-lens summary (personas sampled this run)

| Lens | This run's read | Notes |
|---|---|---|
| **Concordance (04)** | **SEAMS — 1 new (F2), 1 stale-ledger (F1), 2 unchanged minors (F3/F4)** | arc threads cleanly (null → diagnosis → resolution); prefaces present with venue/status/what-later-chapters-revise; time-capsule framing intact (`3_cbic.tex:21`, `4_courb.tex:13`, `5_mobiwac.tex:22`). |
| **Number (06)** | **No fabrication, no wrong number found in the change set** | AL 64.51 reconciled; CBIC fill matches `cbic_recompute_result.md`; Ch.3↔Ch.5 Florida agree to the digit. F2 is a *cross-source divergence with each side faithful*, not a mismatch to source. Full numeral re-extraction was NOT run this pass (holistic sample); the round-1 exhaustive N4 pass stands. |
| **Claim/honesty (07)** | **No unlicensed claim introduced by round 2** | B.1 verbs bound to tests preserved ("no consistent multi-task advantage", "outperforms ... at four of the six", "matches ... within two points"); AZ never upgraded (`5_mobiwac.tex:485` AZ region carries the ≈ non-inferiority marker, not ↑). Region-verdict scopes ("four of six", "at AL/AZ") intact in the Ch.5 preface (L27-30). |
| **Citation (05)** | **No citation touched destructively by round 2** | `references.bib` mtime predates the chapter edits; `silva2025mtlnet` key stable across the B.1 edit; build log 0 undefined citations. A fresh full existence-recheck was not part of this regression pass (round-1 R3 PASS stands). |
| **UFV compliance (13)** | **defense 87pp / final 83pp; title no longer the sole non-compliance** | build log confirms page counts; folha de rosto + Resumo/Abstract + approval-sheet placeholder present; the round-1 "defense non-compliant only on the title placeholder" condition is now cleared. Toolchain re-compile UNVERIFIED (§0). |
| **Visual (18)** | **header/TOC defect resolved** | p.9-10, p.26, p.42 confirm single-line headings/headers. Figure-asset items (Fig 2 PT labels, Fig 3 color-only encoding) are unchanged asset work, outside the round-2 text change set. |

---

## 6 · Load-bearing devices confirmed intact (so future editors know what not to cut)

- **Scope statement present and singular:** "We do not predict the exact next place"
  (`5_mobiwac.tex:211`, and the Ch.5 problem section), plus the frame's §1.4 scope. Not duplicated,
  not dropped.
- **Time-capsule prefaces** on all three article chapters, each naming venue + status + what later
  chapters revise (`3_cbic.tex:14-28`, `4_courb.tex:12-14`, `5_mobiwac.tex:18-35`).
- **CoUrb ownership note** intact: `4_courb.tex:13` states Tarik S. Paiva is first author, Vitor is
  second author and presenter and first author of the MTLnet baseline.
- **CoUrb weaker-protocol caution** intact: `4_courb.tex:13` "stratifies the cross-validation split
  by sample rather than by user, a weaker protocol ... the conclusions reported here are those of
  the time"; and the mandated no-revisit floor sentence "This chapter isolates the representation
  effect ... it does not revisit the MTL-versus-single-task question, which Chapter 5 reopens."
- **Verb–test binding** intact across the B.1 edit and the Ch.5 preface (outperforms / matches /
  never-upgrade-AZ).
- **Errata trail** intact and bidirectional for B.1 (chapter ↔ Appendix B §B.3 ↔ version-of-record
  ERRATA). **F1 is the one place the errata trail is currently out of sync** (CBIC counts).
- **Lint surface clean in rendered prose:** 0 em-dashes, 0 contractions, 0 repo codenames outside
  LaTeX comments (verified by a comment-stripping sweep; `check.sh` enforces the same and excludes
  comment lines by design).

---

## 7 · Author to-do produced by this run (all author-owned; reviewer proposes, does not edit)

1. **F1 (MAJOR):** update Appendix B Table B.1's CBIC dataset-placeholder row so it no longer tells
   the reader the counts are unfilled/pending — record it as a resolved (or basis-pending) erratum
   consistent with the filled Chapter 3. `apx_b_errata.tex:76-80`.
2. **F2 (MAJOR):** add a one-line reader-facing reconciliation of the Florida count divergence
   (1,407,034 in Ch.3/Ch.5 vs 990,518 in Ch.4) at the point the differing value first appears;
   do not alter either published table value. Whitelist/errata decision.
3. **F3 (MINOR):** name "MTLnet" once inside Chapter 3, or soften the frame's back-pointers.
4. **F4 (MINOR):** make Chapter 4's two model-name headings (`4_courb.tex:80,116`) consistent with
   its own L83 preservation note.
5. **Carry-over (author actions, not new findings):** confirm the CBIC recompute basis; the Tier-2
   `[NEEDS SIGN-OFF]` content; MJ-5 vintage wording; and a clean `make check` + recompile on a TeX
   host (this sandbox has no LaTeX — build health is verified only against the committed logs, §0).

---

*End of report. Read-only run; the only file written was this report. Self-reported success is not
trusted — the author audits independently.*
