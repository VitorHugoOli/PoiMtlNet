# 13 · UFV Compliance Report — dissertation v1

> Persona 13 (UFV compliance checker). Read-only audit against `UFV_COMPLIANCE.md` (the LAW,
> re-verified 2026-07-18) + `TEMPLATE.md`. Scope: BOTH builds — `src/main_defense.pdf` (87 pp)
> and `src/main_final.pdf` (83 pp); LaTeX in `src/0_main.tex` + `main_defense.tex` + `main_final.tex`.
> **STATUS: IN PROGRESS** — written incrementally; do not treat as final until the Verdict block says FINAL.

Audit started: (session in progress). Today's date for calendar math: **2026-07-23**.

---

## Working notes (raw findings, promoted to the contract below as verified)

### MEASURED (PyMuPDF, both builds) — formatting law (Manual 04_2026 §7-§8)

| Item | Rule | DEFENSE measured | FINAL measured | Verdict |
|---|---|---|---|---|
| Paper | A4 21x29.7cm | 595.3x841.9pt = 21.00x29.70cm, ALL 87pp | idem, ALL 83pp | PASS |
| Font (body) | Arial or Times 12 | TeX Gyre Termes (newtx Times-equiv), 12.0pt body | idem | PASS |
| Left margin | 3cm | text x0 = 85.0pt = 3.00cm (every prose page) | idem | PASS |
| Right margin | 2cm | text x1 = 538.3pt = 2.00cm (nominal); 2 overfull hbox exceptions below | idem | PASS w/ 2 exceptions |
| Top margin | 3cm | body block top at 3cm; running header at 2.15cm (head margin, by design) | idem | PASS |
| Bottom margin | 2cm | last-line BASELINE at y=785.2pt = exactly 2.00cm; only descenders extend ~3pt (universal) | idem | PASS |
| Line spacing | 1.5 | 17.93pt baseline-to-baseline for 12pt = 1.49x font size = ABNT/Word "1,5 linhas" (abnTeX2 \OnehalfSpacing) | idem | PASS |
| Page numbers | top-right, arabic, start first body page | top-right x=528pt, arabic; see numbering map | idem | PASS (see below) |

**Page-numbering map (the pre-textual-counted-not-numbered rule, Manual §7):**
- DEFENSE: pre-textual pp 1-11 carry NO printed number (cover, approval, resumo, abstract, LoF,
  LoT, abbrev, contents). First BODY page (Introduction) is pdf-p13 and prints **"12"** (contents
  span pdf-p9-11 -> first numbered body = 12). Wait: printed numbers appear from pdf-p12 onward =
  page "12". So 11 pre-textual pages counted, body starts numbered 12. MATCHES the manual example
  (10 pre-textual -> first body 11 -> here 11 pre-textual -> first body 12). PASS.
- FINAL: strips cover/approval/resumo/abstract; opens at List of Figures -> LoT -> abbrev ->
  Contents -> body. First body page prints **"11"** via \finalbuildfirstpage=11. This offset is
  correctly flagged in main_final.tex as [VERIFY: tune against the AcademicoPG RASCUNHO PDF] --
  it is a post-defense step, cannot be finalized now. PASS-with-known-open-item.

**Overfull hbox (right-margin) exceptions — from build log + pixel-confirmed + margin-overlay:**
- pdf-p20 (Ch.2 Fundamentals): Table 1 (model-lineage) overflows right margin by **29.76pt (1.05cm)**
  -> table content sits ~1.0cm from paper edge instead of 2.0cm. Log lines 214-232. Visible in
  margin overlay: the rules + "Reference"/"Chapter 4/5" column cross the 2cm line. MINOR (table
  slightly over margin; not a reader-misleading defect, but a measurable margin breach).
- pdf-p49 (Ch.4): inline math E_POI_category overflows by **15.14pt (0.53cm)**. Log lines 173-174.
  MINOR.
- These are the ONLY two horizontal overflows in the whole document (log: exactly 2 Overfull hbox).
- 30 Overfull vbox all identical 14.49998pt = global \checkandfixthelayout/\OnehalfSpacing
  bookkeeping constant (noted in 0_main.tex header), NOT per-page content spill; printed text
  respects the 2cm bottom line (baselines measured). Benign.
- 11 Underfull hbox = cosmetic (loose lines), not compliance.

### STRUCTURE (coletanea)
- TOC order (defense): 1 Introduction -> 2 Fundamentals -> 3 Article 1 (CBIC) -> 4 Article 2
  (CoUrb) -> 5 Article 3 (MobiWac) -> 6 Conclusion -> References (single global list) ->
  Appendix A (Other Scientific Contributions) / B (Errata) / C (AI-Use Disclosure).
- Mandatory triplet (i) Introducao Geral / (ii) Artigos / (iii) Conclusao Geral: PRESENT + ORDERED.
- AI-Use Disclosure = Appendix C (settled placement, survives both builds). PRESENT.
- [TO VERIFY] a standalone Fundamentals (Ch.2) chapter inserted between Introduction and the
  articles is NOT one of the three mandated coletanea elements. Common in BR coletaneas and the
  Germano precedent should be checked -- flagged below.
- Article-status labels: [PENDING read of chapter prefaces].

---

### FINAL BUILD (AcademicoPG) — additional measured facts
- Strip verification (text probe, all pages): "Magister Scientiae", "Resumo", "Abstract",
  "Palavras-chave", "Keywords", "Orientador: Fabricio", "VITOR HUGO OLIVEIRA SILVA", "Approval
  sheet" -> ALL PRESENT in defense, ALL ABSENT in final. Clean two-build split. PASS (§1 build 2).
- Opens at List of Figures -> List of Tables -> abbreviations -> Contents -> body. PASS.
- Funding sentence (CAPES 001/FAPEMIG/CNPq): NOT present anywhere in either PDF (no agradecimentos
  in source) -> zero duplication risk with the system-inserted sentence (§2 item, checklist 7). PASS.
- pdftitle metadata (both builds) = "[TITLE - open decision NORTH_STAR §5.8]" -> carries the same
  placeholder; fix when the title is decided (metadata, minor).

### BUILD HEALTH (both logs)
- 0 undefined references, 0 undefined citations, 0 "Rerun to get cross-references right". Clean.
- bibtex .blg: 0 substantive warnings.

### PROCESS / precedent
- Germano precedent (exemples/germano, defended 2024, same advisor, EN body): body order is
  Introduction -> HAVANA (article) -> "Urban Region Representation Learning: A Comprehensive
  Review" (NON-article chapter) -> Tackling Spatial Heterophily (article) -> Conclusion. Direct
  in-program precedent that a standalone non-article chapter in the coletanea body is admissible.
- Art. 22 calendar (today 2026-07-23): defense Aug 18 -> text filed by Jul 29 (6 days); Aug 29 ->
  by Aug 9 (17 days). Text-lock window Jul 29-Aug 9. Feasible, tight, no slack for a re-freeze.

---

## Verdict — FINAL

### (1) VERDICT PER BUILD

**DEFENSE build (`main_defense.pdf`, 87 pp): NON-COMPLIANT AS BUILT** — one BLOCKER (undecided
title prints as a literal placeholder on the folha de rosto), plus two MINOR right-margin
overflows. Every *measured* formatting rule (A4, Times 12, 3/3/2/2 cm margins, 1.5 spacing,
top-right arabic numbering starting on the first body page, pre-textual counted-not-numbered) and
every structural rule (coletanea triplet present + ordered, article statuses correctly labeled,
Resumo+Abstract present, AI-use disclosure present) PASSES. Insert the title and fix the two
overflows and this build is COMPLIANT.

**FINAL AcademicoPG build (`main_final.pdf`, 83 pp): COMPLIANT WITH CONDITIONS** — front-matter
strip, body-only ordering, and numbering-start mechanism are all correct. Conditions: the same
Table 1 margin overflow (MINOR), the pdftitle-metadata placeholder (MINOR), and the
`\finalbuildfirstpage=11` offset which is correctly flagged in-source as [VERIFY against the
RASCUNHO PDF] and can only be tuned post-defense. No blocker in the visible content (the title
placeholder is stripped from this build).

### TOP 3 FINDINGS
1. **[BLOCKER, defense]** Title placeholder printed on the folha de rosto (see B1).
2. **[MINOR, both]** Table 1 (model lineage, Ch.2) overflows the right margin by 1.05 cm (see M1).
3. **[ADVISORY]** Fundamentals chapter admissibility — confirm with the Comissao (see A1).

### RANKED FINDINGS (rule -> quote/measurement -> location -> direction)

**B1 · BLOCKER (defense build) · Undecided title prints as a literal placeholder**
- Rule: the defense PDF that ships to the secretariat/banca is a conventional full PDF whose folha
  de rosto must carry the real title; the title must match the folha de rosto exactly
  (UFV_COMPLIANCE §1 build 1; CLAUDE.md §2 open decision "must match the folha de rosto exactly").
- Measurement/quote: folha de rosto (defense pdf-p1) renders `[TITLE - OPEN DECISION NORTH_STAR
  §5.8]`; identical placeholder in the Resumo header (pdf-p3), the Abstract header (pdf-p4), and
  the pdftitle metadata of BOTH builds.
- Location: `src/0_main.tex` `\titulo{...[TITLE --- open decision...]...}` and the two `\textbf{[TITLE
  --- open decision NORTH\_STAR §5.8]}` lines in the Resumo/Abstract blocks.
- Direction: this is a tracked open decision (CLAUDE.md §2; "latest sensible moment: before the
  defense-build front matter ~ Jul 23" = today). A working title already sits in the source
  comments ("From Representations to a Single Joint Model: Multi-Task Learning for
  Point-of-Interest Category and Region Prediction"). Replace the three placeholder strings (and
  confirm pdftitle inherits) before the defense build ships. Not a formatting defect; a
  content-decision gap that blocks shipping.

**M1 · MINOR (both builds) · Table 1 (model lineage) overflows the right margin**
- Rule: right margin 2 cm (Manual §7); text must not cross x = 538.6 pt.
- Measurement: Overfull \hbox 29.76 pt (1.05 cm) at `2_fundamentals.tex` lines 214-232; pixel-
  confirmed (table rules + "Reference"/"Chapter 4/5" column reach x = 567 pt = 0.97 cm from the
  paper edge) and margin-overlay-confirmed. Location: defense pdf-p20 / final pdf-p18.
- Direction: this is the model-lineage table the brief mandates for Ch.2. Constrain the table
  width to the 2 cm line, e.g. fix the "What it added" column with a sized `p{...}` /
  tabularx, or wrap in `\resizebox{\linewidth}{!}{...}` / `adjustbox` (already in the preamble).

**M2 · MINOR (defense) · Inline math overflows the right margin**
- Rule: right margin 2 cm.
- Measurement: Overfull \hbox 15.14 pt (0.53 cm) at `4_courb.tex` (Categorical Encoder, POI
  Encoder), `E_{POI\_category} \in \mathbb{R}^{64}` runs past the line. Location: defense pdf-p49.
- Direction: allow a line break before the `\in` / restructure the sentence so the display does
  not sit at line end, or set the subscript with `\mathit` to shave width.

**A1 · ADVISORY (structure) · Standalone Fundamentals chapter not explicitly named in the norms**
- Rule: coletanea body = (i) Introducao Geral, (ii) Artigo(s), (iii) Conclusao Geral (Normas §2.3,
  §2.6; UFV_COMPLIANCE §3). The three mandated elements are all present and correctly ordered. A
  non-article "Fundamentals" chapter (Ch.2) sits between the Introduction and the first article;
  the norms text names the triplet + optional CLOSING sections + appendices, not opening
  non-article body chapters.
- Support: direct in-program precedent — Germano (defended 2024, same advisor, EN) placed a
  standalone "Comprehensive Review" chapter inside the coletanea body. §2.6 free formatting adds
  latitude.
- Direction: keep the chapter; raise it in the same advisor/Comissao sign-off bundle as the
  frame-language and CoUrb-inclusion questions (CLAUDE.md §2 open item 3). Fail-closed flag, not a
  violation; it does not disturb the mandated triplet.

**N1 · NIT (both) · Separate ABNT capa absent**
- The defense build's first page is the folha de rosto; there is no distinct capa. The Manual's
  "cover page" is satisfied by the folha de rosto and the ficha/capa are BBT/system elements for
  the final deposit, so this is almost certainly fine — but confirm the secretariat does not
  require a separate capa in the pre-defense build. (Ficha catalografica correctly absent from
  both builds: it is BBT-generated from the RASCUNHO after the text is finalized, §4.)

### (2) PROCESS-PREREQUISITES STATUS TABLE

| # | Prerequisite | Status | Date / note |
|---|---|---|---|
| Art.21 §1 | >=1 published/accepted/submitted article of the research | SUBSTANCE DONE | CBIC published, DOI 10.21528/CBIC2025-1191324 verified (CLAUDE.md §1); CoUrb published as backup |
| Art.21 §1 | comprovante filed with secretariat (ppgcc@ufv.br) | PENDING | author/secretariat action; not verifiable in the PDF |
| Art.21 §1 | operative Qualis/resolucao bar cleared | AT-RISK / OPEN | CBIC=B4 per repo research; whether B4 clears the internal resolution is the open question for the secretariat (UFV_COMPLIANCE §7 open item a) |
| Art.22 | text to secretariat >=20 days pre-defense | AT-RISK (tight) | defense Aug 18-29 -> file Jul 29-Aug 9; 6-17 days from today; no slack for a re-freeze |
| Defense | anti-plagiarism certificate | PENDING | mandatory ("a defesa nao sera aprovada" without it); not verifiable in-session |
| Defense | wet-signature items (termo de assentimento, BBT authorization) | PENDING | plan physical signatures around the defense; not verifiable in-session |
| AI disclosure | note present at settled placement | DONE | Appendix C (AI-Use Disclosure), one page, survives both builds |
| Format | defense build formatting law | DONE (pending B1) | all measured rules pass; title B1 blocks shipping |
| Format | final build body-only + strip | DONE | verified; numbering offset pending RASCUNHO (post-defense) |

### (3) ITEMS NOT VERIFIABLE IN-SESSION (flag for the author)
- Comprovante actually filed with the secretariat; operative Art.21 quality bar confirmed.
- Anti-plagiarism certificate obtained.
- Wet-signature items (termo de assentimento, BBT authorization term) tracked/signed.
- `\finalbuildfirstpage` offset (=11) tuned against the AcademicoPG RASCUNHO PDF — a post-defense
  step; the mechanism is correct, only the constant is unverified. Already flagged [VERIFY] in
  `main_final.tex`.
- Whether the secretariat requires a distinct ABNT capa in the defense build (N1).
- Advisor/Comissao sign-off on: EN frame language, CoUrb inclusion, and the Fundamentals chapter
  (A1) — the unregulated-area bundle.

### WHAT HOLDS / WHAT READS WELL (do not touch)
- A4, Times-equivalent 12 pt body, 3 cm left/top + 2 cm right/bottom margins, and 1.5 line spacing
  are all measured-correct across every body page of both builds. The `0_main.tex` header note about
  re-deriving the text block under 1.5 spacing worked: the bottom-line baseline lands at exactly
  2.00 cm.
- Page numbering is exactly to spec: pre-textual pages counted-but-unnumbered, arabic top-right,
  starting on the first body page (defense: body starts "12" after 11 pre-textual pages; final:
  "11").
- The two-build split is clean and correct: the final build strips cover/approval/resumo/abstract/
  keywords and opens at the lists, and no funding sentence is duplicated.
- Article-status labels are exactly right and honest: CBIC "published", CoUrb "published"
  (translated reproduction, DOI stated), MobiWac "submitted ... under review at the time of writing
  (EDAS #1571313639)". BRACIS correctly contained in Appendix A as an unpublished earlier
  iteration, not a chapter.
- Build is clean: zero undefined refs/citations.

### OUT-OF-SCOPE HANDOFFS (one line each; not this persona's call)
- Em-dashes appear inside the title/approval PLACEHOLDER strings and the Resumo/Abstract header
  placeholder -> WRITING_LAW no-em-dash rule; persona 03. (They vanish when the placeholders are
  replaced.)
- Resumo/Abstract claim + number parity (5.3-9.4 macro-F1, TOST, n=20) -> personas 04/06/07.
- Whether the abbreviations-list membership is complete/correct vs. prose usage -> persona 04/18.

---
_Report FINAL. Full working measurements above the verdict block. Read-only: no source file was
edited by this persona; the only file written is this report._
