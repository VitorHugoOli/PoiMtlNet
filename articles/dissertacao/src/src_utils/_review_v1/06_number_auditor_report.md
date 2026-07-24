# 06 · NUMBER AUDITOR REPORT — dissertation v1 (assembled, `src/`)

> Persona 06 (number auditor, G2 fact gate, rules N1–N5). Fresh-eyes, read-only.
> Auditor drafted none of the text under review. This file is the only file written.
> Scope: ALL SIX chapters (`src/chapters/1_introduction.tex` … `6_conclusion.tex`) +
> appendices (`apx_a`, `apx_b`, `apx_c`) + front matter (`src/0_main.tex`).
> Protocol: `reviewers/README.md` Common protocol; AGENT_GUARDRAILS §2 (N1–N5).
> The prior N4 report (`src/_gates/N4_R3_REPORT.md`) is RE-DERIVED here, not trusted.
>
> **STATUS: COMPLETE** — 2026-07-23. Exhaustive extraction + trace done for all ten files.

## VERDICT: **GATE FAIL (conditional)** — 1 BLOCKER (declared, pre-existing), 0 fabrications, 0 MISMATCH

The document's numbers are in excellent shape. Every result table is byte-faithful to its source
of record; every prose numeral, caption, footnote, abstract, and Resumo value traces to a
committed source; there is **no MISMATCH anywhere**, the **never-cite sweep is clean**, and the
**Abstract↔Resumo numeric parity is exact**. The gate fails on a single item — the three
**declared, unfilled dataset placeholders in Ch.3** (`3_cbic.tex:235`), which are orphan numerals
by definition and which the chapter's own ledger marks as blocking handoff. No value was
fabricated; the failure is that the gate cannot record PASS while three numerals have no source.
Two MAJOR items (a convention-naming gap between Ch.5's 64.51 and Ch.6's 64.54; a declared-partial
California value at n=15/20) are correct-but-incomplete, not mismatches. This independently
reproduces the prior N4 verdict (FAIL-conditional, same three-item shape) with fresh extraction.

### Per-chapter verdicts

| Chapter / part | Verdict | Basis |
|---|---|---|
| Ch.1 Introduction | **PASS** | 14 numerals all traced; DOIs/years/ordinal (23rd) correct |
| Ch.2 Fundamentals | **PASS** | 5 numerals (93% ×2, Acc@10 ×3), convention values, coherent |
| Ch.3 CBIC | **FAIL** | tables byte-identical + 84 prose numerals verbatim, BUT 3 orphan placeholders (N-1) |
| Ch.4 CoUrb | **PASS** | tables byte-identical; 99 prose numerals traced (0.02 pp sourced to judge_feedback) |
| Ch.5 MobiWac | **PASS** | tables byte-identical; 139 prose numerals traced; never-cite clean |
| Ch.6 Conclusion | **PASS (with MAJOR N-2/N-3)** | headline + capacity numbers all sourced; convention-naming gap + partial CA value to finalize |
| apx_a Contributions | **PASS** | dates owned by CLAUDE.md §1 |
| apx_b Errata | **PASS** | restatements self-consistent with sanctioned errata records |
| apx_c AI disclosure | **PASS** | CNPq Portaria 2.664/2026 |
| Front matter (0_main) | **PASS** | Resumo↔Abstract numeric parity exact; title still a placeholder (out-of-scope) |
| **Document** | **GATE FAIL (conditional on N-1)** | one declared BLOCKER; resolve N-1 (+ N-3 before banca) → re-run → PASS |

---

## Method

Scripted numeral extraction over the ten `.tex` files in scope: LaTeX comments stripped
(unescaped `%` to EOL), arguments of `\cite/\ref/\label/\url/\includegraphics/\input/\bibliography`
masked, every remaining numeral+unit captured with file:line context. Table cells verified
separately as ordered numeral sequences against the source-of-record files. Front matter
(Resumo/Abstract) extracted the same way. No sampling (gate day = exhaustive).

Sources of truth (per `reviewers/README.md §Sources`):
- Ch.5 (MobiWac): `docs/studies/closing_data/RESULTS_BOARD.md` §1 + §3 file map to JSONs.
- Ch.3 (CBIC): published tables in `articles/CBIC___MTL/`.
- Ch.4 (CoUrb): published tables in `articles/CoUrb_2026/` + `slides/judge_feedback.md`.
- Leak audit: `docs/studies/pre_freeze_gates/A4_RESULTS.md`.
- Venue/status/date: `articles/dissertacao/CLAUDE.md §1`.
- Never-cite: STAN v4-collapse, ReHDM v2 row, VOID fp16/bf16, `docs/PAPER_FINDINGS.md` flags.

---

## Part A — numeral extraction counts — COMPLETE

Exhaustive scripted extraction (digit numerals + a spelled-out-number sweep). Counts below are
digit-numeral totals; the spelled-number sweep (Part C) is additional and caught the CA
"fifteen of twenty" value a digit-only pass misses.

| File | Digit numerals | Classification |
|---|---|---|
| 1_introduction.tex | 14 | all traced (predictability 93; CBIC/CoUrb DOIs+years; 23rd MobiWac; 64-d) |
| 2_fundamentals.tex | 5 | all traced (93% ×2; Acc@10 "10" ×3) — convention/reference values |
| 3_cbic.tex | ~360 | table cells ordered-identical to source (Part B); 84 prose numerals all verbatim; **3 declared placeholders (BLOCKER, pre-existing)** |
| 4_courb.tex | ~366 | table cells ordered-identical; 99 prose numerals traced (0.02 pp + DOI frag explained) |
| 5_mobiwac.tex | 371 | table cells ordered-identical; 139 prose numerals traced (EDAS#/year owned by CLAUDE.md §1) |
| 6_conclusion.tex | 19 | all traced to capacity source + CoUrb ERRATA + board; 2 declared roundings; 1 declared-partial (CA n=15) |
| apx_a_contributions.tex | 4 | dates (BRACIS rejection, owned by CLAUDE.md §1) |
| apx_b_errata.tex | 60 | errata/bib restatements; sampled DOIs/pages match sanctioned records (Part C) |
| apx_c_ai_disclosure.tex | 3 | CNPq Portaria 2.664/2026 |
| 0_main.tex (front matter) | 23 | Resumo/Abstract data numerals = {5.3, 9.4} + n=20 convention; parity holds (Part D) |
| **Total** | **~1,225** | **0 MISMATCH; 3 ORPHAN-by-design placeholders (declared BLOCKER)** |

Digit-numeral counts track the prior N4 report within extraction-boundary noise (Ch.3/4 differ
by 1–5 tokens from how math-mode subscripts and DOI fragments are split; every such token was
individually resolved, none is a data cell). No numeral is left unaccounted for.

## Part B — table verification (cell-by-cell, ordered) — COMPLETE

Nine result tables traced to their sources of record. Method: extract ordered numeral
sequence from each chapter table environment and from its source file, strip layout/macro
tokens (`\multicolumn`, `\cmidrule`, `\multirow`, `\providecommand{\sd}`), diff the two
sequences, then independently re-verify the emphasis sets (`\textbf` = best; `\ul`/`\underline`
= second-best) as numeric-payload multisets.

| Chapter table | Source of record | Data cells | Emphasis | Verdict |
|---|---|---|---|---|
| tab:cbic:category | `CBIC___MTL/tables/category_result.tex` | ordered-identical | bold 21/21 | ✅ |
| tab:cbic:next | `.../next_result.tex` | ordered-identical | bold 21/21; **2nd-best 15/15** (`\ul`→`\underline` macro swap, same payloads) | ✅ |
| tab:cbic:convergence | `.../converge_result.tex` | ordered-identical | — | ✅ |
| tab:courb:dataset | `CoUrb_2026/src_en/resultados/tabela_dataset.tex` | ordered-identical | — | ✅ |
| tab:courb:category | `.../tabela_comparativa_f1_category.tex` | ordered-identical | bold 22/22 | ✅ |
| tab:courb:next | `.../tabela_comparativa_f1_next.tex` | ordered-identical | bold 22/22 | ✅ |
| tab:mobiwac:datasets | `[mobiwac]/src/tables/tbl1_datasets.tex` | ordered-identical | — | ✅ |
| tab:mobiwac:representation | `.../tbl2_substrate.tex` | ordered-identical | — | ✅ |
| tab:mobiwac:results | `.../tbl3_results.tex` | ordered-identical | bold 12/12 | ✅ |

The only sequence differences were traced and are NOT data:
- CBIC three tables: a stray `5` = the chapter rewrote the source's terse captions into fuller
  ones stating the fold convention ("over the 5 folds" / "5-fold cross-validation"). The
  numeric data cells are untouched. (Caption authorship, N5-relevant — see Part D.)
- mobiwac representation/results: source's `\providecommand{\sd}[1]{...#1}` macro-definition
  tokens (`1,1`), which the chapter defines in the preamble instead; and the chapter's
  `\cmidrule(lr){3-6}\cmidrule(lr){7-11}` vs the source's `\multicolumn{11}{...2\tabcolsep}`
  header-layout tokens. Layout only.

**No data cell differs from its source of record in any of the nine tables.** This
independently reproduces the prior N4 report's table verdict.

## Part C — prose numeral trace — COMPLETE

Method: extract prose numerals (outside table environments, true line numbers, identifier args
masked) per chapter; verify each appears verbatim in the chapter's source of record. Digit
regex supplemented by a spelled-out-number sweep (one…twenty, hundred/thousand/million) filtered
to experimental-unit nouns — this caught the "fifteen of twenty repetitions" CA value that a
digit-only sweep misses.

**Paper chapters (re-typeset published text) — all prose numerals trace:**
- **Ch.3 (CBIC): 84 prose numerals, 84 found verbatim** in `CBIC___MTL/sections/*.tex` +
  tables + `ERRATA.md`. 0 unexplained.
- **Ch.4 (CoUrb): 99 prose numerals, 97 found verbatim** in `CoUrb_2026/src_en/` sources. The
  two not found are both benign: (a) `2026.22960` = the DOI fragment `10.5753/courb.2026.22960`,
  matching CLAUDE.md §1 (owner of DOI facts); (b) `0.02` percentage points (FL-Outdoors technical
  tie) — quoted from `slides/judge_feedback.md §2` ("baseline 21,61 vs Sphere 21,59 — baseline
  ainda vence por 0,02 pp") and declared in the Ch.4 ledger A1 + ERRATA #1; the two cells 21.61
  and 21.59 are both present in the source table. Quoted, not computed (N2 OK).
- **Ch.5 (MobiWac): 139 prose numerals, 136 found verbatim** in `[mobiwac]/src/sections|tables|figs`.
  The three not found: `2026` and `1571313639` (EDAS #, owned by CLAUDE.md §1, confirmed);
  `003` = a regex-boundary artifact of `$\pm0.003$`, which appears verbatim in `02_related.tex:93`.

**Frame chapters + front matter (author-drafted, highest N-audit risk) — headline numbers traced
to the MobiWac board:**
- **Ch.6 capacity paragraph (L74–79)** traces cell-for-cell to
  `storyline/audit/capacity_baseline_experiment.md §5`:
  4.2M = 4,197,621 (§5.1); 0.6M = 644,359 (§5.1, declared rounding); 56.16 ±1.88 = best arm
  n=20 (§5.3); 56.82 = dedicated ceiling at its own width, n=20 (§5.3); 64.54 = joint v17 n=20
  (§5.2/§5.3); "fifteen of twenty repetitions" CA partial = "seeds {0,1,7} = n=15 of 20" (§5.4),
  "same direction" verbatim. **NOTE**: `56.16` exists in NO file under `docs/`; its only source of
  truth is this `storyline/audit/` file (job `d38a1382`, `al_capmatch_summary.json`). See finding
  N-2 on the convention distinction between Ch.6's 64.54 and Ch.5 Table 3's 64.51.
- **Ch.6 other numerals**: +0.001 cosine (L88, four seeds, three of six datasets) traces to
  `02_related.tex`; 2009/2010 Gowalla vintage traces to DATASETS/CoUrb conclusion; 20.2–22.0
  (L39) = CoUrb ERRATA #2 audited range; 5.3–9.4 (L51) = the headline (see Part D / N5).
  limitation~1…6 are `enumerate` counters, not data.
- **Ch.1**: 93 (song2010limits), CBIC/CoUrb DOIs + years, 23rd ACM MobiWac, 64-d embedding —
  all trace (93% is the DRAFT_LEDGER-verified predictability bound; 64-d is the CBIC/CoUrb input).
- **Ch.2**: 93%, Acc@10 ("10"), seven categories, six datasets / five states / two check-in
  datasets / one city — all convention constants, coherent with Ch.5 (see Part D).
- **apx_a**: BRACIS rejection "June 8, 2026" — owned by CLAUDE.md §1 (BRACIS containment C4).
- **apx_c**: CNPq Portaria 2.664/2026 — AGENT_GUARDRAILS §6.

## Part D — cross-checks (abstract ↔ body ↔ Resumo; captions ↔ tables; convention N5) — COMPLETE

**D.1 Abstract ↔ Resumo ↔ body numeric parity.** The Resumo (PT) and Abstract (EN) in
`0_main.tex` carry exactly one pair of data numerals: the headline range **5.3 to 9.4**
(PT "5,3 a 9,4" / EN "5.3 to 9.4" — correct decimal-separator localization), plus the
convention triple "twenty repetitions (four random initializations, five folds)" =
"vinte repetições (quatro inicializações aleatórias, cinco partições)". Both match Ch.6 L49–53
and NORTH_STAR §2. The claim structure maps 1:1: "outperforms at four of six … matches
(TOST ±2 pp / margem de dois pontos) at the other two" in both. **Parity holds.**
(The stray `5.8` in both blocks is the document pointer "NORTH_STAR §5.8" inside the
`[TITLE --- open decision]` placeholder — not a data numeral. See out-of-scope: title unresolved.)

**D.2 Headline range recomputed from source.** The deployable (joint-best) category deltas in
`joint_best/JOINT_BEST_RESULTS.md` are AL +7.69, AZ +9.35, FL +5.33, TX +7.45, CA +6.45,
Istanbul +8.58. min = 5.33 → 5.3; max = 9.35 → 9.4. The Ch.6/Abstract headline **"5.3 to 9.4"**
is the min/max of this set rounded, and Ch.5 body states the same range unrounded (+5.33 to
+9.35, L533). Internally consistent. The CoUrb headline **"20.2 to 22.0"** (Ch.6 L39) is the
min/max of the per-state best-of-two-encoder means FL +20.24 / CA +20.91 / TX +21.98
(`slides/judge_feedback.md §11`), with 21.98 → 22.0 the ERRATA #2 declared rounding. Correct.

**D.3 Captions ↔ table contents.** All nine result-table captions verified (Part B). The CBIC
captions were rewritten by the chapter to state the fold convention ("over the 5 folds"), which
introduces the caption `5` token; data cells untouched. Ch.5 Table 3 caption correctly names its
emphasis convention (bold = statistically supported improvement over dedicated; ↑ = supported
region improvement; ≈ = TOST ±2 pp non-inferior match). Ch.5 Table 2 caption correctly carries
the matched-recipe **seed-0 × 5-fold** convention and does not blur with Table 3's n=20.

**D.4 Convention constants — cross-chapter consistency (N5).** Verified coherent everywhere:
- "twenty repetitions = four seeds × five folds" (n=20): stated together in Ch.1, Ch.6, front
  matter; Ch.5 uses the operational "four seeds / five folds". Ch.3 (CBIC) correctly uses its
  own "5-fold" protocol (predates the n=20 convention; time-indexed).
- seven categories / six datasets / five US states + Istanbul / two check-in dataset sources:
  coherent in every chapter that names them (no count drift).
- "four of six" region result: identical wording in Ch.1, Ch.2, Ch.5 (×2), Ch.6, front matter.
- TOST two-point margin at **Alabama and Arizona**: consistent; AZ is never upgraded from a
  match to a beat (Ch.5 results comments enforce it; Ch.6 restates it as a match).

**D.5 The one genuine convention subtlety (see finding N-2).** The AL joint-category cell is
reported as **64.51** in Ch.5 Table 3 (deployable / joint-best checkpoint) and **64.54** in the
Ch.6 capacity paragraph (diagnostic-best, n=20). Both are individually correct and individually
sourced (`JOINT_BEST_RESULTS.md` L32: `AL 56.82 | 64.54 diag | 64.51 deploy | −0.04`). Ch.5 **does**
name its convention in rendered prose (`5_mobiwac.tex:515–518`); the gap is at the Ch.6 site, which
switches to the diagnostic-best value without signalling it, leaving a reader with an unexplained
0.03 discrepancy against Table 5.3. This is the "joint-best vs diagnostic-best distinction must
never blur" risk (N5) — not a MISMATCH, but a convention-naming gap at Ch.6. Reported as MAJOR N-2.

## Part E — never-cite sweep — COMPLETE (CLEAN)

Swept every chapter's prose (comments stripped) for each absolute never-cite value
(README §Sources + Ch.5 results comment L461–463):

| Never-cite value | Meaning | Hits in prose |
|---|---|---|
| 34.46 / 38.96 | STAN v4-collapse (AL/AZ) | **0** |
| 62.37 | HMT-GRN AL outlier | **0** |
| 66.06 / 65.68 | ReHDM v2 row (a, c) | **0** |
| 54.65 | ReHDM v2 row (b) | 1 — **FALSE POSITIVE, disambiguated below** |
| fp16 / bf16 VOID cells | — | **0** (none of those values present) |

**The single 54.65 hit is not a violation.** It occurs at `5_mobiwac.tex:402` as the **Istanbul
next-category macro-F1 at the check-in level** in Table 2 (`tab:mobiwac:representation`), traced
byte-identical to `tbl2_substrate.tex:24` and its provenance comment (Check2HGI-SC 54.65 ±0.56).
This is a different physical quantity that coincidentally shares the numeral with the ReHDM v2
region-Acc@10 value. Confirmed the chapter's actual ReHDM row (Table 3) uses the sanctioned **v4**
values (Ist 69.33 / AL 65.38 / AZ 53.00, per `rehdm.md` and Ch.5 comment L455), and the v2 triple
66.06 / 54.65 / 65.68 as a *set* appears nowhere. **Never-cite sweep passes.**

## Findings

### Top 3 (most valuable)
1. **N-1 [BLOCKER]** — three orphan dataset numerals in Ch.3 (declared placeholders, gate cannot pass).
2. **N-2 [MAJOR]** — AL joint-category cell differs between Ch.5 Table 3 (64.51) and Ch.6 (64.54) under two unnamed conventions (joint-best vs diagnostic-best).
3. **N-3 [MAJOR]** — Ch.6 California capacity value is a declared partial (n=15/20); must be finalized to n=20 before the final gate.

---

**[N-1] BLOCKER (declared, pre-existing) — Ch.3 dataset placeholders are orphan numerals.**
`3_cbic.tex:235`: "This subset comprises a total of `[$N_{\text{users}}$; VERIFY: recompute per
ERRATA.md]` users, `[$N_{\text{poi}}$; ...]` unique Points-of-Interest (POIs), and
`[$N_{\text{checkins}}$; ...]` check-ins." Three numerals have no source value — by the persona's
definition an orphan numeral is a BLOCKER, and the chapter's own ledger agrees: `3_cbic_ADAPTATION_LEDGER.md`
B1 marks it "**AUTHOR ACTION REQUIRED before handoff**" and D.1 marks it "**[VERIFY — blocks handoff]**".
The values were *not* invented (correct fail-closed behaviour); the sanctioned path (a repo-committed
recompute over the CBIC-era Florida pipeline — Gowalla FL, users with <5 visits dropped —
author-approved) has not been run. *Direction:* run the sanctioned script and author-approve, or
ship the advisor draft with visible placeholders only if the author explicitly accepts that. The
gate cannot record PASS while three orphan numerals stand. Reproduces prior N4-1.

**[N-2] MAJOR (BLOCKER-adjacent) — Ch.6 quotes the AL joint cell under a different, unnamed
selection convention than Ch.5.** The Alabama joint next-category value is **64.51** in Ch.5
Table 3 (`tab:mobiwac:results`, `5_mobiwac.tex:479`) and **64.54** in the Ch.6 capacity paragraph
(`6_conclusion.tex:78`, "64.54 for the joint model"). Both trace correctly to
`joint_best/JOINT_BEST_RESULTS.md` L32 (`AL | 56.82 | 64.54 diag | 64.51 deploy | −0.04`), so this
is **not a MISMATCH** — they are two legitimate summaries of the same model under two selection
rules the source keeps as distinct columns: *deployable* (one saved artifact per fold, epoch
chosen by the joint validation score) = 64.51, and *diagnostic-best* (each task read at its own
best epoch) = 64.54. **Ch.5 names its convention in rendered prose** (`5_mobiwac.tex:515–518`:
"every reported model is one saved artifact per fold, read at its validation-selected epoch … the
joint model at the epoch selected by its joint validation score"). The defect is at the **Ch.6**
site only: the capacity paragraph switches to the diagnostic-best 64.54 (the reference its own
source `capacity_baseline_experiment.md §5.2–5.3` uses, and internally consistent there with
56.16 / 56.82 on the same basis) **without signalling the switch**, so a reader who remembers
Table 5.3's AL joint = 64.51 meets "64.54 for the joint model" with an unexplained 0.03 gap. It is
BLOCKER-adjacent under the contract's "blurred convention" / "same fact quoted twice" cross-checks,
but ranked MAJOR because (i) both values are source-true, (ii) the capacity comparison is
internally convention-consistent, and (iii) the paragraph's conclusion (the +8 pp gap the wider
dedicated model fails to close) holds identically at 64.51 or 64.54. *Direction:* at
`6_conclusion.tex:78` name the convention (e.g. "64.54 for the joint model read at each task's best
epoch, the reference used by the capacity experiment; Table 5.3's deployable-checkpoint value is
64.51"), or switch the comparison to 64.51 for cross-table consistency. Which convention Ch.6
should present is also a claim decision (persona 07).

**[N-3] MAJOR (declared-partial) — Ch.6 California capacity value is n=15/20, not final.**
`6_conclusion.tex:78–80`: "A partial California run, fifteen of twenty repetitions at the time of
writing, shows the same direction." Matches the source (`capacity_baseline_experiment.md §5.4`:
job `4cff4b00`, seeds {0,1,7} = n=15/20, 68.35 ±0.53, "same direction"), and the prose honestly
discloses partiality and states no point value — so it is not an orphan or mismatch. But §5.4's
own contract says "the final value is the best arm's n=20 mean once seed 100 and the second arm
land." *Direction:* replace with the final n=20 California verdict when job `4cff4b00` completes,
then re-run this numeral check. Reproduces prior N4-2.

**[N-4] NOTE — declared-rounding inventory (all sanctioned, none agent-computed).**
- 5.33 → 5.3 and 9.35 → 9.4 (headline; Ch.6:51, Abstract `0_main.tex`, Resumo; source
  `JOINT_BEST_RESULTS.md`, min/max of the deployable deltas; NORTH_STAR §2 records "5.3…9.4").
- 21.98 → 22.0 (Ch.6:39; CoUrb ERRATA #2).
- 644,359 → 0.6M and 4,197,621 → 4.2M (Ch.6:74–75; `capacity_baseline_experiment.md §5.1` gives
  the full-precision values; the chapter quotes the ~M rounding). Each rounding exists in — or is
  the declared min/max over — its source; the chapters quote, they do not compute.

**[N-5] NOTE — CBIC captions rewritten (not reproduced verbatim).** The three CBIC tables' captions
were rewritten by the chapter to add the fold convention ("over the 5 folds" / "5-fold
cross-validation"), which is why a `5` token appears in the chapter tables that is absent from the
source captions. The numeric *data cells* are byte-identical. This is a legitimate N5 improvement
(the published captions did not state the convention), not a data change — recorded for completeness.

**[N-6] NOTE — 34.97 s and 2.3× in Ch.3 are sanctioned reconciliations.** `3_cbic.tex:349`:
"80.88 s … about 2.3 times the cumulative 34.97 s". The published paper said "almost four times";
CBIC `ERRATA.md` line 32 sanctions the correction ("80.88 s / 34.97 s = 2.3x — reconcile to the
table"). 34.97 = 16.26 + 18.71 (the two single-task times from the convergence table); the
reconciliation lives in the committed errata, not in agent arithmetic. N2-compliant.

## What holds (do not touch)

- **Every result table is byte-faithful to its source of record.** All nine tables (CBIC ×3,
  CoUrb ×3, MobiWac ×3) have data cells ordered-identical and emphasis sets (bold = best,
  `\ul`/`\underline` = second-best) identical to their published/board sources. This includes the
  CBIC next-table second-best set (15/15 cells, verified through the `\ul`→`\underline` macro swap)
  and the MobiWac Table 3 bold set (12/12).
- **Zero MISMATCH anywhere.** Not one numeral in prose, tables, captions, footnotes, abstract,
  or Resumo disagrees with its source. The only sequence differences found were layout/macro
  tokens (`\providecommand{\sd}`, `\cmidrule`, `\multicolumn`) and a rewritten-caption convention
  token — all individually traced.
- **The headline result is sound and consistently reported.** "5.3 to 9.4" (category, all six
  datasets) and "four of six, TOST ±2 pp at the other two" (region) are the correct min/max /
  count over the deployable joint-best deltas, stated identically across Ch.1, Ch.5 body, Ch.6,
  Abstract, Resumo, and NORTH_STAR §2. AZ is never upgraded from a match to a beat.
- **Abstract ↔ Resumo numeric parity is exact** (claim-parity includes numbers): same range, same
  n=20 convention, same hedges, correct decimal-separator localization (5,3 / 5.3).
- **Never-cite sweep is clean.** No STAN v4-collapse (34.46/38.96), no HMT-GRN AL 62.37, no
  ReHDM v2 row (66.06/54.65/65.68 as a set), no fp16/bf16 VOID cells. The lone 54.65 hit is a
  distinct, correctly-sourced quantity (Istanbul category F1).
- **Convention constants are internally coherent** across all six chapters + front matter
  (seven categories, six datasets = five US states + Istanbul, two check-in dataset sources,
  n=20 = four seeds × five folds). Ch.3's own 5-fold convention is correctly time-indexed and
  not conflated with the later n=20.
- **The capacity-experiment numbers trace cell-for-cell** to `capacity_baseline_experiment.md §5`
  (4.2M, 0.6M, 56.16, 56.82, 64.54, n=15 CA partial) — every value present and exact.
- **Declared roundings and reconciliations are all sanctioned** in committed sources
  (NORTH_STAR §2, CBIC/CoUrb ERRATA), never produced by agent arithmetic in prose (N2 holds).

## Could-not-verify (fail-closed)

- **The three Ch.3 dataset statistics (N-1)** cannot be verified because no value exists — they
  are unfilled placeholders awaiting the sanctioned recompute. This is *reported as a blocker*,
  not smoothed over. Missing input: the author-approved output of the CBIC-era Florida recompute
  script.
- **The final California capacity value (N-3)** cannot be verified as final because the run
  (`4cff4b00`) had reached only n=15/20 at the time of writing. The current text correctly
  discloses this and states no point value; the *final* n=20 number is what is missing.
- **apx_b DOI/venue/page restatements**: verified as *self-consistent with the sanctioned errata
  records* (CBIC/CoUrb `ERRATA.md`, `BIB_MERGE_REPORT.md`) — e.g. TKDE 34(4):1902–1914 2022,
  CVPR.2016.**433**, TKDE.2021.3070203, pp. 323–336, arXiv:1905.07553. Re-resolving those DOIs
  against Crossref/OpenAlex is persona 05's scope (the prior R3 pass did so and found no defect);
  as number auditor I confirm only that the appendix restates them without internal contradiction.

## Out-of-scope handoffs (one line each)
- **Dissertation title is still `[TITLE --- open decision NORTH_STAR §5.8]`** in the folha de
  rosto, Resumo, and Abstract (`0_main.tex`) — a front-matter completeness item for the author /
  concordance, not a numeral defect. (This is the source of the stray "5.8" pointer.)
- Approval-sheet placeholder in the front matter (`0_main.tex:161`) — expected pre-defense; format/persona 13.
- The 64.51/64.54 convention wording (N-2) also touches claim-honesty (persona 07): the choice of
  which convention Ch.6 presents is a claim decision, not only a number-naming one.

## Open questions for the author
1. **N-1**: run the sanctioned Florida recompute now and fill the three Ch.3 statistics, or ship
   the advisor draft with visible placeholders (the gate stays FAIL until they are filled)?
2. **N-2**: name the convention at `6_conclusion.tex:78` (64.54 = diagnostic-best) alongside a
   pointer to Table 5.3's deployable 64.51, or switch Ch.6 to 64.51 for cross-table consistency?
3. **N-3**: hold the final gate for the n=20 California capacity value, or keep the honestly-hedged
   partial for the advisor draft and finalize before the banca build?
