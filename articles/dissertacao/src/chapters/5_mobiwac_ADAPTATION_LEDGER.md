# Chapter 5 (MobiWac) — adaptation ledger

> **Source (version of record):** `articles/[mobiwac]/src/` (main.tex + sections/01–08 +
> tables/tbl1–tbl3 + figs/fig1–fig4), the current working build per NORTH_STAR §5.4
> ("the last one in the src"). Re-typeset performed 2026-07-23. **Status wording used
> throughout: "submitted to MobiWac 2026, under review" (EDAS #1571313639).**
> Re-sync obligation: the author refines `[mobiwac]/src/` in parallel; this chapter must be
> re-synced against it before the final gate pass (single-source rule, NORTH_STAR §5.4).
>
> This ledger lists EVERY departure from the source text (feeds Appendix B; the
> reconciliation gate is "fixes applied == fixes listed"). ERRATA.md for this paper records
> no known citation/number defects, so no errata corrections were applied — every entry
> below is a re-typeset transformation, not a content fix.

## A. New-to-chapter text (not in the submitted paper) — each marked [NEEDS SIGN-OFF] in a LaTeX comment

| # | Where | What | Basis |
|---|---|---|---|
| A1 | Chapter opening | The italic time-capsule preface (`chapterpreface`): venue, "submitted, under review" status, EDAS number, the arc-resolution sentence (claim wording bound to the whitelist: category at all six datasets with the scope named; region outperforms at four of six; matches within two points at the other two; no numerals), and the reformatting/errata sentence. | NORTH_STAR §3 (time-capsule rule) + §4 Ch.5 notes; PAPER_PLAN §3 whitelist |
| A2 | §5.2.1 "The MTLnet framework and the representation diagnosis" | New recap subsection (first subsection of Background and Related Work) covering BOTH the Ch.3 artifact (MTLnet, `\cite{silva2025mtlnet}`, with the CBIC null result time-indexed as "the conclusion of the time for that configuration") AND the Ch.4 finding (ST-MTLNet, `\cite{paiva2026courb}`, representation-not-architecture diagnosis, scoped to "each of the three states studied"). No numbers quoted. | NORTH_STAR §3 bridging devices ("Ch.5 gets both"); GLOSSARY model-lineage table |
| A3 | §5.6.1 | **Figure 6 (fig3_embquality) RESTORED.** It was cut from the 8-page submitted build (author decision 2026-07-09; its four numbers are fully stated in the §6.1 prose, which is unchanged). Restored because the dissertation has no page limit; caption verbatim from `figs/fig3_embquality.tex`; PDF regenerated from the paper's own script (numbers untouched). Remove the block to match the submitted article exactly. | fig wrapper comment in `[mobiwac]/src/main.tex`; NORTH_STAR §4 Ch.5 (dissertation gains space) |

The OPTIONAL restorations of NORTH_STAR §4 Ch.5 (expanded A4 leak-audit prose, fuller
statistical-protocol detail, the fp16→fp32 harness lesson) were **NOT** performed in this v1:
the chapter is a faithful re-typeset. §5.5.2 keeps the paper's compressed leak-audit text
unchanged.

## B. Mechanical re-typeset transformations (rule-mandated; prose meaning unchanged)

| # | Transformation | Instances |
|---|---|---|
| B1 | "this paper" → "this chapter" | 3 prose instances: 01_introduction "The rest of the paper presents" → "The remainder of this chapter presents"; 02_related "this paper adds the check-in-level" → "this chapter adds…"; 07_discussion "the paper's claims are the prediction results themselves" → "this chapter's claims are…". Post-sweep grep: zero residual "this paper/this article/this work/the paper" in prose. |
| B2 | Section levels | Paper `\section` → `\section` under the `\chapter`; `\subsection` unchanged. No section was split, merged, reordered, or retitled. |
| B3 | Labels renumbered, chapter-prefixed | `sec:*` → `sec:mobiwac:*`, `fig:*` → `fig:mobiwac:*`, `tab:*` → `tab:mobiwac:*`, `eq:loss` → `eq:mobiwac:loss`, `fn:code` → `fn:mobiwac:code`. One label RENAMED beyond the prefix: `tab:substrate` → `tab:mobiwac:representation` (the repo word "substrate" is banned from dissertation files; label only, caption/prose untouched). |
| B4 | `Fig.~\ref` → `Figure~\ref` | All instances (dissertation style; IEEE abbreviation removed). |
| B5 | `\cite` keys kept AS-IS from `[mobiwac]/src/references.bib` | 35 unique keys; no new keys invented. The two dissertation-internal works are cited by the paper's own keys (`silva2025mtlnet`, `paiva2026courb`); the Phase-4 bib merge maps them (note: the dissertation seed bib currently holds `paiva2026stmtlnet` for the same DOI — merge point, not a chapter edit). Until Phase 4 lands the donor entries, most Ch.5 citations render as `??` — expected, not a defect of this chapter. |
| B6 | Per-section plan-comment headers stripped | The leading `% Section N: … Plan: PAPER_PLAN …` comment blocks were removed; all IN-BODY provenance comments (hidden data-provenance notes, the joint-best convention note, the AZ-ceiling sensitivity note, table provenance blocks, never-cite reminders) were PRESERVED verbatim inside the chapter file. |
| B7 | Chapter title | From the paper title, with "Article 3:" prefix per the dissertation chapter map; stub's escaped `\:` fixed to a plain colon. |

## C. Tables (script-generated sources copied, never retyped; every numeric cell byte-identical to `[mobiwac]/src/tables/`)

| # | Table | Departures from source |
|---|---|---|
| C1 | Table `tab:mobiwac:datasets` (tbl1) | `table*` → `table` (one-column layout); `\hline` set → booktabs (`\toprule/\midrule/\bottomrule`); tabular wrapped in `adjustbox{max width=\textwidth}`. Caption verbatim, above the table (ABNT). |
| C2 | Table `tab:mobiwac:representation` (tbl2) | `\hline` → booktabs. Caption + coincidence footnote verbatim. |
| C3 | Table `tab:mobiwac:results` (tbl3) | `table*` → `table`; `\hline` → booktabs + `\cmidrule` pair under the two group headers; vertical rules removed from the column spec (booktabs convention); the `\multicolumn{11}` footnote row moved OUT of the tabular into a `\footnotesize` block below it (text verbatim); tabular wrapped in `adjustbox{max width=\textwidth}`. Bolding, `\sd`, `↑`/`≈` markers, dagger/ddagger disclosures all verbatim. |

Lead takeaway sentences before each table are the paper's own sentences (§5.1 for Table 1,
§6.1 opening for Table 2, §6.2 for Table 3) — no new lead sentences were written.

## D. Figures (assets in `figures/mobiwac/`; captions verbatim from `[mobiwac]/src/main.tex` / the fig wrappers)

| # | Figure | What was done |
|---|---|---|
| D1 | Figure `fig:mobiwac:dataflow` (fig1, TikZ) | RECOMPILED standalone (not stretched): natural size 5.43 × 2.47 in, within the ~6.3 in text width; fonts switched from the paper build's Computer Modern to `newtxtext/newtxmath` to match the dissertation body (in-figure text near body size). Placement: end of Section 5.2 (the paper floats it as a two-column `figure*` next to §2). Caption verbatim. |
| D2 | Figure `fig:mobiwac:model` (fig2, TikZ) | Same treatment; natural size 3.94 × 2.73 in. Placement after §5.4 prose. Caption verbatim. |
| D3 | Figure `fig:mobiwac:embquality` (fig3) | RESTORED (ledger item A3). Regenerated from `fig3_embquality.py`: figsize 3.3 → 5.2 in, fonts 8 → 10 pt, bar labels 6.5 → 8.5 pt. **Data constants untouched.** Patched script copied as `figures/mobiwac/fig3_embquality_diss.py`. |
| D4 | Figure `fig:mobiwac:deltas` (fig4) | Regenerated from `fig4_deltas.py`: figsize 3.3 → 5.2 in, fonts 8 → 10 pt, value labels 6 → 8 pt. **Data constants untouched** (joint-best deltas: Istanbul +8.58/+0.19, AL +7.69/−0.41, AZ +9.35/0.00, FL +5.33/+0.71, TX +7.45/+2.11, CA +6.45/+2.20). The paper wrapper's `width=0.79\columnwidth` scaling dropped; included at natural size. Patched script copied as `figures/mobiwac/fig4_deltas_diss.py`. Caption verbatim. |

## E. Elements of the paper NOT reproduced in the chapter

| # | Element | Why |
|---|---|---|
| E1 | Paper abstract + IEEE keywords | The coletânea chapter reproduces the body; the abstract's claims live in the frame chapters (Germano/Viegas pattern). No chapter-level abstract exists in the skeleton. |
| E2 | Title block / author block / IEEE `\maketitle` | Frame front matter owns authorship; the preface states venue and status. |
| E3 | Paper bibliography commands (`\bibliographystyle{IEEEtran}` etc.) | Single global dissertation bibliography (decision #5). |

## F. Claim-discipline conformance (checked after assembly)

- Region verbs bound to tests: "outperforms" only at Istanbul/FL/TX/CA; "matches"/"non-inferior
  (TOST, ±2 pp)" at AL/AZ; the Arizona sentence ("a match, not a gain") preserved verbatim —
  AZ never upgraded. Scaling claim scoped to the five U.S. states (paper sentence preserved).
  Cascade framed as the paper frames it (same combined score; "a defense of the parallel
  design, not a claim that we outperform the cascade").
- Never-cite lists: no STAN v4-collapse numbers, no ReHDM v2 row, no VOID fp16/bf16 cells
  anywhere in the chapter (the provenance comments carrying the never-cite reminders were kept).
- No repo codenames in prose; `src/check.sh` codename sweep passes (the one `substrate` label
  hit was pre-empted by B3).
- No em-dashes, no contractions (check.sh: OK on both).
- Numbers: 100% of numerals are byte-copies from the source sections/tables/scripts; **no
  number was computed, rounded, or aggregated by the agent** (N2). No new numbers exist outside
  the source text, so no separate numbers ledger is needed beyond this file: every value traces
  to `articles/[mobiwac]/src/` (whose own source of truth is
  `docs/studies/closing_data/RESULTS_BOARD.md` per AGENT_GUARDRAILS N1).

## G. Open items / [VERIFY] register

- **[NEEDS SIGN-OFF] A1, A2, A3** (preface, recap subsection, Figure 6 restoration) — author
  approval required; each is marked with a `% [NEEDS SIGN-OFF: …]` comment at its site in
  `5_mobiwac.tex`.
- **Phase-4 bib merge dependency:** the chapter's 35 cite keys resolve only after the MobiWac
  donor bib is merged into `src/references.bib`; the duplicate CoUrb keys
  (`paiva2026courb` [paper key] vs `paiva2026stmtlnet` [seed key]) must be unified at merge
  time. Until then the compiled chapter shows `??` for the unmerged keys.
- **Re-sync before final gate:** `[mobiwac]/src/` is being refined in parallel (NORTH_STAR
  §5.4); diff and re-run this re-typeset before the final gate pass.
- No [VERIFY] flags on content: no unsourced number or claim was introduced.
