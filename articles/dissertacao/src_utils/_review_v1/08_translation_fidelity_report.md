# 08 · Translation Fidelity Checker — L5 gate report (CoUrb PT → EN)

**Scope:** Chapter 4 only — `src/chapters/4_courb.tex` vs the published Portuguese paper of
record `articles/CoUrb_2026/src/` (DOI 10.5753/courb.2026.22960, Anais do CoUrb 2026, pp. 323–336).
Intermediate EN translation `articles/CoUrb_2026/src_en/` used to classify each PT→chapter
difference as translation-artifact vs sanctioned-adaptation.
**Reviewer:** fresh eyes (did not draft this chapter). Read-only. **Date:** 2026-07-23.

---

## (1) VERDICT: **L5 PASS**

Every claim-bearing sentence in the English chapter maps 1:1 to the published Portuguese text in
quantifier, hedge, tense, negation, and scope. Every numeric value in the two result tables and
the dataset table is byte-identical to the published tables after locale normalization
(machine-checked: 63/63 category cells, 63/63 next-POI cells, 9/9 dataset counts — see §5). The
three departures from the published numbers/verbs are the **documented, audited errata**, applied
under the settled errata policy and listed in Appendix B — not silent fixes and not silent
reproductions. No claim was strengthened, weakened, or scope-shifted; nothing in the PT paper was
silently dropped, and nothing was added beyond the sanctioned frame devices (preface, MTLnet
recap subsection, protocol-honesty sentence, table lead-in sentences), each declared in the
chapter's `ADAPTATION_LEDGER.md` and/or Appendix B.

---

## TOP 3 FINDINGS

1. **[PASS — confirmation]** All three published CoUrb errata are correctly handled everywhere
   they occur: the category-gain range `20–24 pp` → audited `20.2 to 22.0 pp` with the
   best-of-two-encoders disclosure (chapter L33, L252, L343), and the sequential-task win count
   `16/21` → audited `15/21 + 1 technical tie` with the 0.02 pp / within-1σ explanation (L295,
   L343). No stray `16`, `76`, or `24` claim survives in the chapter. All three are itemized in
   Appendix B Table B.3. **This is the core of the L5 gate and it holds.**

2. **[MINOR — terminology landing, sanctioned]** The chapter keeps the paper's own task names
   *POI Category Classification* and *Next-POI Prediction* rather than the GLOSSARY canonical
   *category classification* / *next-category prediction*. This is **correct, not a drift**:
   GLOSSARY §1 rules "the chapter keeps the paper's usage and the frame uses this registry; the
   per-paper mapping in §2 bridges them." The chapter is internally clear that the predicted
   label is the category (intro list item: "predict the category of the next POI"). Flagged only
   so the author knows it was checked and is intentional — do **not** "fix" it to the canonical
   names inside Ch.4.

3. **[MINOR — coordination flag for persona 07, out of L5 scope]** The verb "outperforms" appears
   throughout the reproduced results prose resting on **mean-F1 comparison, not a paired
   significance test** (the CoUrb study reports mean ± std over 5 folds; no Wilcoxon/TOST). As a
   *translation* this is faithful — PT "supera"/"vence" → "outperforms" is equal-strength (ledger
   A3, "claim strength unchanged"). The note is cross-chapter: Ch.5 uses "outperforms" as a
   test-bound verb, so a reader may back-read significance onto Ch.4. The preface time-index +
   "only baseline / does not revisit MTL-vs-STL" sentence mitigates but does not explicitly say
   "no significance testing was performed." Author/persona-07 decision, not an L5 blocker.

---

## (2) DRIFT TABLE

Claim-bearing passages, PT-of-record vs EN chapter. Classification: **none** = faithful 1:1;
**number (sanctioned erratum)** = documented audited correction, not a fidelity failure.

| # | PT (published `src/`, verbatim) | EN chapter (`4_courb.tex`, verbatim) | Class |
|---|---|---|---|
| D1 | "ganhos médios de 20 a 24 pontos percentuais" (intro, results, conclusion) | "average gains per state of 20.2 to 22.0 percentage points, considering the better of the two spatial encoders in each combination" (L33, L252, L343) | **number — sanctioned erratum #2** (ledger A2, Appx B). Audit: FL +20.24 / CA +20.91 / TX +21.98; 22.0 = sanctioned rounding of 21.98. The "better of the two encoders" qualifier is the disclosure the audit requires. Not silent. |
| D2 | "os modelos espaço-temporais superam o MTLNet original em 16 das 21 combinações avaliadas" (results); "vence em 16 das 21" (conclusion) | "the spatio-temporal models outperform the original MTLNet in 15 of the 21 evaluated combinations, with one additional technical tie in *Outdoors* in Florida, where the *baseline* mean exceeds the best variant by 0.02 percentage points, a gap within one standard deviation" (L295); "in 15 of the 21 … with one additional technical tie" (L343) | **number — sanctioned erratum #1** (ledger A1, Appx B). Audit recount: 15 strict wins + 1 tie (FL Outdoors, baseline 21.61 vs Sphere2Vec-M 21.59). Not silent. |
| D3 | "o modelo proposto vence na maioria dos cenários" (intro) | "the proposed model outperforms the *baseline* in most scenarios" (L33) | **none** — banned verb "wins/vence" → "outperforms"; scope word "most scenarios / na maioria dos cenários" preserved exactly (ledger A3). Equal strength. |
| D4 | "ganhos particularmente **expressivos** em … *Nightlife* e *Travel*" (results) | "gains are particularly **substantial** in categories such as *Nightlife* and *Travel*" (L252) | **none** — "expressivo" is a false friend (means substantial-in-magnitude, not "expressive"); the correct EN sense was chosen (fixed at the `src_en` red-team stage). Faithful. |
| D5 | "a abordagem modular seja consistentemente superior ao *baseline*" (intro) | "the modular approach is consistently superior to the *baseline*" (L33) | **none** — 1:1. |
| D6 | "não há um único *encoder* espacial universalmente superior" (conclusion) | "there is no single universally superior spatial *encoder*" (L343) | **none** — 1:1. |
| D7 | "a diferença na dimensionalidade de entrada **pode influenciar parte** dos ganhos observados … permitiria validar de forma mais precisa" (methodology) | "the difference in input dimensionality **may influence part** of the observed gains … would allow validating more precisely" (§4.3 Embedding Integration) | **none** — hedge ("pode/may", "parte/part", conditional "permitiria/would allow") preserved 1:1. |
| D8 | "o *baseline* mantém vantagem em alguns casos, especialmente em *Travel* na Flórida e na Califórnia, além de … *Entertainment*, *Nightlife* e *Outdoors*" (results) | "the *baseline* maintains an advantage in some cases, especially in *Travel* in Florida and California, in addition to specific categories in California, such as *Entertainment*, *Nightlife*, and *Outdoors*" (L296) | **none** — 1:1 incl. all scope qualifiers. |
| D9 | Scope phrases throughout: "nos três estados", "três estados avaliados" | "in the three evaluated states" / "three states" (×5, no occurrence of "across the datasets" / "everywhere" / "six datasets") | **none** — scope never widened. Machine-checked. |

**Sanctioned frame additions** (present in chapter, absent from published PT — declared in ledger
§C and Appendix B, visibly frame material, no new results/contribution claims):
- **Preface** (L12): translated-reproduction statement + DOI + pages 323–336 + `\cite{paiva2026stmtlnet}`; Vitor's contribution note (2nd author, presenter, author of baseline MTLNet); sample-stratified-split caveat; time-index sentence; the "only baseline / does not revisit MTL-vs-STL, which Ch.5 reopens" floor sentence.
- **§4.2.5 "The MTLnet framework"** (L79–84): Ch.3 artifact recap + naming-variant note + time-indexed CBIC null recap.
- **Protocol-honesty sentence** in Experimental Setup (L224): "The split is stratified by sample, not by user, so the *check-ins* of one user may appear in both training and validation; Chapter 5 adopts a stricter user-disjoint protocol."
- **Three table lead-in sentences** (dataset/category/next-POI) + one Figure 4.2 caption reading-instruction (B4). Each reads its table/figure without introducing a new number or a new results claim; verified faithful to the cells. The dataset lead-in ("Texas concentrates the largest volume … Florida provides the smallest") is true from Table 4.1 (TX 3,355,419; FL 990,518).

---

## (3) TERMINOLOGY-LANDING REPORT

| PT term (published) | Chapter EN | GLOSSARY canonical | Landing verdict |
|---|---|---|---|
| Classificação de Categoria de POI | POI Category Classification | category classification | **OK — paper usage kept in-chapter** (GLOSSARY §1/§2 per-paper mapping; frame bridges). |
| Predição do Próximo POI (categoria) | Next-POI Prediction | next-category prediction | **OK — paper usage kept**; chapter defines it as "predict the category of the next POI". Frame owns the bridge sentence. |
| check-in / check-ins | check-in / check-ins (italic) | check-in (never "event") | **OK.** |
| POI / Ponto de Interesse | POI / Point of Interest | POI / place (never "venue") | **OK** — "place" also used; no "venue". |
| baseline | *baseline* (italic, paper voice) | dedicated single-task model / baseline | **OK in-chapter** (reproduced-paper voice; the frame reserves "dedicated single-task model"). |
| supera / vence | outperforms | verb bound to test in frame | **Faithful translation** (equal strength); cross-chapter verb-law note → finding #3 / persona 07. |
| representações desacopladas | decoupled representations | — | **OK — consistent.** |
| MTLNet (paper spelling) | MTLNet ×45 in body; MTLnet ×4 in the recap subsection with explicit note "the published paper typesets the name as MTLNet, and this chapter preserves that form" | MTLnet (glossary) / MTLNet (paper) | **OK — the naming-variant note (L82) is exactly the GLOSSARY-sanctioned bridge.** Body preserves the paper's MTLNet; the frame recap uses MTLnet and reconciles the two. |

No ad-hoc translation created a synonym pair with another chapter within Ch.4's own text.

---

## (4) ERRATA-POLICY CHECK — **PASS**

Policy (NORTH_STAR §5.7, decision #7): fix silently in the re-typeset chapter + one frame
sentence + list every departure in Appendix B; published records not edited.

- **Not silently reproduced:** the erroneous `16/21` and `20–24 pp` do not appear as chapter
  claims (grep: zero surviving `16`/`76`/`24` result claims).
- **Not silently fixed:** all three corrections are itemized in **Appendix B Table B.3**
  (`apx_b_errata.tex`), matching the audit source `slides/judge_feedback.md` §2 and
  `articles/CoUrb_2026/ERRATA.md`.
- **Preservation done right:** the published bold on the FL-Outdoors baseline cell in the
  next-POI table is kept exactly (`\textbf{21.61 ± 0.99}`); the technical-tie reading lives in
  prose only (ledger A4). This is the correct "reproduce the table, correct the prose" split.
- The published paper's **abstract** (which carried the uncorrected "76%" and "20–24 pp") is
  dropped per coletânea convention (ledger B7); the erroneous abstract numbers therefore do not
  leak into the chapter, and the corrected values appear in intro/results/conclusion. Clean.

---

## (5) SECTIONS VERIFIED CLEAN (coverage)

| Section | Grain | Result |
|---|---|---|
| Preface (frame) | sentence | All mandated elements present and correct (DOI, pages, authorship, split caveat, time-index, floor sentence). |
| §4.1 Introduction | sentence | 1:1 with `src/sections/intro.tex`; erratum #2 + verb sub applied. |
| §4.2 Related Work (4 axes) | paragraph | 1:1 with `src/sections/related.tex`; all cited systems described as in PT. |
| §4.2.5 MTLnet recap | sentence | Sanctioned frame addition; time-indexed, no new result claim. |
| §4.3 Methodology (baseline, data prep, spatial/SIREN/Sphere2Vec-M, temporal/Time2Vec, categorical/POI Encoder/HGI, integration) | paragraph | 1:1 with `src/sections/metodology.tex`; every parameter identical (64/192/256, L_h=9, 1728/576, τ=0.15, 10/70 km, α=0.5, S=16, 10 km–10,000 km, 7 classes, 8 heads/4 layers, 2/3/4-layer MLPs). |
| §4.4 Results (setup, category, next-POI) | sentence + cell | Prose 1:1; both errata applied; split-honesty sentence added; **tables cell-identical** (machine-checked). |
| §4.5 Conclusion | sentence | 1:1 with `src/sections/conclusion.tex`; both errata applied; limitations (Travel, no ablation, Gowalla 2009–2010) 1:1. |

**Machine checks run (fail-closed):**
- Category table: 63 mean±std cells PT vs chapter → **IDENTICAL**.
- Next-POI table: 63 cells → **IDENTICAL**.
- Dataset table: 9 counts {20301, 36106, 37522, 65009, 135570, 148314, 990518, 2535573, 3355419} → **IDENTICAL** (locale normalized).
- Scope-phrase sweep: "three states" only; no "datasets/everywhere/six" widening.
- Banned verb sweep: zero "wins/win/beats" in prose.

---

## WHAT HOLDS / WHAT READS WELL (do not touch)

- The **errata handling is exemplary**: corrected in prose, preserved in the table cell, disclosed
  in Appendix B, traced in the ledger. This is precisely the "not silent either way" standard.
- The **preface** carries the full reproduction statement, authorship note, protocol caveat,
  time-index, and the Ch.5-reopening floor sentence in four clean sentences — the most
  fidelity-sensitive paragraph in the chapter, and it is correct.
- **Number fidelity is total** — no fabricated, dropped, or corrupted cell; the hardest failure
  class for translated result tables is fully clean.
- The **false-friend trap** ("expressivo" → "substantial", not "expressive") was caught upstream
  and is right in the chapter.

---

## OUT-OF-SCOPE HANDOFFS (one line each)

- **→ persona 05 (citation auditor):** bib-key renames between PT and chapter for the *same works*
  — `huang2023learning`→`huang2023hgi` (HGI), `cho2011friendship`→`cho2011gowalla` and
  `10.1145/2661829.2662002`→`liu2014geographical` (Gowalla), and the sanctioned
  `church2017word2vec`→`mikolov2013word2vec` (Appx B bib row 6). Confirm the global bib resolves
  each renamed key to the identical work.
- **→ persona 07 (claim & honesty):** the mean-F1-based "outperforms" verb across the reproduced
  results (finding #3) — decide whether the preface needs one sentence stating no significance
  test was applied in this study, to prevent cross-chapter back-reading from Ch.5's test-bound
  "outperforms".
- **→ persona 04 (concordance) / frame:** ensure Ch.1/Ch.2 state once that CoUrb's "Next-POI
  Prediction" = the canonical "next-category prediction" (GLOSSARY §2 mapping); the bridge is a
  frame duty, not a Ch.4 edit.
- **→ persona 18 (visual):** the chapter title uses `\:` (renders as a thin space, not a colon)
  and the wording "Point-of-Interest Representations" vs the paper's EN "Representations of Points
  of Interest" — both are pre-existing author `[VERIFY]` flags in the ledger; heading/render, not
  claim fidelity.

## OPEN QUESTIONS (author only)

1. Chapter-heading title form + the `\:` thin-space (ledger §E [VERIFY] flags) — confirm the
   intended EN title wording and whether a literal colon is wanted (global to Ch.3/4/5 stubs).
2. Whether to add the Nash-MTL solver-bug caveat to Ch.4 (ledger §E marks it NOT added, to avoid
   a new claim; a preface sentence is the place if wanted).
