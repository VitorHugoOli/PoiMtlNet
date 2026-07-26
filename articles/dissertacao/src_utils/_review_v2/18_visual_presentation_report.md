# 18 · Visual & presentation reviewer — the rendered-pages pass

**Build audited:** `src/dissertacao.pdf` (94 pp, written 2026-07-25 23:43:53), pages rendered to
images and measured; `src/build/main_final.pdf` (89 pp) checked for the same defects.
**Date:** 2026-07-26. **Persona:** `reviewers/18_visual_presentation.md`. Read-only.

**Method.** Every font size below is **measured**, not estimated: per-character nominal point sizes
read from the PDF text layer via `FPDFText_GetFontSize`, and for bitmap figures, glyph-band pixel
heights measured on the source image and converted through the `\includegraphics` scale factor that
the LaTeX log records. Body-text baseline established over eight prose pages (14, 15, 21, 22, 36,
62, 74, 75): **11.96 pt**, modal at 17,297 of 17,747 glyphs. The WRITING_LAW §5 floor for in-figure
text is ~80% of body = **9.57 pt**.

## Verdict

**NEEDS A VISUAL PASS.**

The four layout claims I was asked to verify split three-to-one:

| Claim under test | Measured result |
|---|---|
| The two overfull boxes are gone | **CONFIRMED** — 0 overfull hboxes in the current build log |
| The two previously shrunken Chapter 5 tables now render at a legible size | **CONFIRMED** — both at 11.96 pt, full body size |
| No table renders before the heading that introduces it | **CONFIRMED** — all 15 tables and 7 figures render at or after first reference |
| No floats-only page | **FAILS** — page 71 is floats only |
| No float splits a sentence | **FAILS** — Table 10 + Figure 7 split the §5.6.2 results paragraph across pp. 70→72 |

Against those, two defects the round did not touch are more serious than anything it introduced: a
**Portuguese-labeled figure in an English chapter**, and **in-figure text at 33–51% of body size**
in the two Chapter 4 bitmap figures. Both are pre-existing; both are the kind of thing a banca
notices in the first thirty seconds of paging through.

**1 BLOCKER, 3 MAJOR, 4 MODERATE, 3 MINOR.**

## Top 3 findings

1. **V-01 (BLOCKER)** — Figure 2 (p. 48) carries Portuguese labels in the English-frame chapter.
2. **V-02 (MAJOR)** — Chapter 4's two bitmap figures render text at 3.96–6.04 pt against 11.96 pt body.
3. **V-03 (MAJOR)** — four `(??)` citation markers are visible on pp. 21, 45, 49, 50.

---

## Verification of the round's layout claims

### Overfull boxes: gone (confirmed)

`src/build/main.log` (defense, 2026-07-25 23:42:43) and `main_final.log` (23:43:53) both report
**zero** `Overfull \hbox`. The two that the stale root-level `src/main.log` (15:41) still shows —
29.76 pt at lines 220–238 and 15.14 pt at lines 175–176 — are from the pre-correction build and no
longer reproduce.

I also checked the pages for margin bleed directly rather than trusting the log: no text extends
past the text block on any page.

Residual, not a defect: 13 `Underfull \hbox` warnings, all in the reference list (badness 1270–10000,
lines 353–502 of `main.bbl`), which is normal for a `\raggedright` bibliography.

### The two shrunken Chapter 5 tables: fixed (confirmed, with measurements)

The round split Table 10 into two tabulars and restructured Table 8's headers. Both worked:

| Table | Page | Body text size, measured | Previous (per source comments) | Against 11.96 pt body |
|---|---|---|---|---|
| Table 8 (datasets) | 65 | **11.96 pt** | 8.13 pt | **100%** |
| Table 9 (representation) | 69 | **11.96 pt** | (not shrunk) | **100%** |
| Table 10 (results) | 71 | **11.96 pt** | 8.00 pt | **100%** |

Sub- and superscripts inside those tables measure 8.97 pt (the `\sd{}` standard deviations) and
8.77 pt (the ↑/≈/†/‡ markers), which is correct typographic behavior for `\scriptsize` annotations,
not shrinkage — the digits they attach to are full size. Table footnotes measure 9.96 pt
(`\footnotesize`), also correct.

This is a real improvement and the measurement confirms the source comments' predicted outcome
(they estimated ~10.1 pt for Table 8; it came out better, at full size).

### No table renders before its introducing heading: confirmed

I mapped every numbered float to its render page and to the page of its first prose reference,
excluding the front-matter lists:

| Float | Renders | First referenced | Δ |
|---|---:|---:|---:|
| Table 1 | 22 | 21 | +1 |
| Table 2 | 38 | 38 | 0 |
| Table 3 | 39 | 38 | +1 |
| Table 4 | 40 | 40 | 0 |
| Table 5 | 53 | 53 | 0 |
| Table 6 | 54 | 54 | 0 |
| Table 7 | 56 | 55 | +1 |
| Table 8 | 65 | 62 | +3 |
| Table 9 | 69 | 59 | +10 |
| Table 10 | 71 | 59 | +12 |
| Tables 11–15 | 88–93 | 87–92 | +1 to +2 |
| Figure 1 | 35 | 33 | +2 |
| Figure 2 | 48 | 47 | +1 |
| Figure 3 | 53 | 54 | −1 |
| Figure 4 | 62 | 59 | +3 |
| Figure 5 | 64 | 59 | +5 |
| Figure 6 | 70 | 69 | +1 |
| Figure 7 | 71 | 59 | +12 |

**No float precedes its first reference** except Figure 3 (−1), which renders on p. 53 and is first
referenced on p. 54 — one page early, and its introducing section heading (4.4 Results, p. 53) is on
the same page, so the reader meets it in context. Acceptable.

The large positive deltas for Table 9, Table 10 and Figure 7 are artifacts of the Chapter 5
contributions list on p. 59 forward-referencing them; each has a *local* reference on the facing or
adjacent page (Table 9 → p. 68; Table 10 and Figure 7 → p. 70). That is the placement that matters
and it is within the ~1-page rule. **Claim confirmed.**

---

## Ranked findings

### V-01 · BLOCKER · Figure 2 is labeled in Portuguese inside an English chapter

**Page 48.** The figure's own boxes read, verbatim from the source image:

- `Encoder Espacial` / `Encoder Temporal` / `Encoder Categórico` (the three section headings)
- `Coordenadas (lat, lon)`
- `Timestamps (hora, dia)`
- `Categorias (POI graph)`

The caption beneath is in English ("Architecture based on MTLNet (1). The spatial, temporal, and
categorical *encoders* are trained in a decoupled manner…"), so the caption and the figure disagree
on language *on the same page*.

Chapter 4 is a declared translated reproduction — its preface says so — and the translation reached
every sentence of the body and the caption, but not the figure bitmap
(`src/figures/courb/arquitetura_modelo.png`). A banca reading an English-frame dissertation meets an
untranslated diagram at p. 48. This is the single most visible presentation defect in the document.

*Direction:* regenerate the diagram with English labels (Spatial Encoder / Temporal Encoder /
Categorical Encoder; Coordinates (lat, lon); Timestamps (hour, day); Categories (POI graph)). If the
original must be preserved for fidelity to the published paper, the caption should say so
explicitly — but WRITING_LAW §5's "self-contained caption" rule and the English frame both argue for
regeneration. Author's call; either way it needs a decision, not silence.

### V-02 · MAJOR · Chapter 4's bitmap figures render text far below the legibility floor

Measured on the source images, converted through the scale factors in `src/build/main.log`:

**Figure 2** (`arquitetura_modelo.png`, 1102 px natural width → 455.0 pt requested, scale 0.4113):

| Label | glyph band | on page | effective nominal | % of body |
|---|---:|---:|---:|---:|
| "Location Encoder" (box text) | 9 px | 3.72 pt | **~5.44 pt** | **45%** |
| "Encoder Espacial" (heading) | 10 px | 4.13 pt | **~6.04 pt** | **51%** |

**Figure 3** (`distribuicao_estados.png`, 5389 px → 455.0 pt, scale 0.3505):

| Label | glyph band | on page | effective nominal | % of body |
|---|---:|---:|---:|---:|
| "Longitude" axis title | 32 px | 2.70 pt | **~3.96 pt** | **33%** |
| x-axis tick label ("−80.150") | 44 px | 3.71 pt | **~5.44 pt** | **45%** |

Against the ~80% floor (9.57 pt), all four are failures, and Figure 3's axis titles at **33%** are
below the threshold at which the label is readable at print size at all. I confirmed this visually
on a 4× render: the tick labels on p. 53 are legible only when magnified.

The two figures are also **bitmaps** where the pipeline elsewhere produces vector output — Chapter
5's Figures 4–7 are PDFs (`fig1_dataflow.pdf`, `fig2_model.pdf`, `fig3_embquality.pdf`,
`fig4_deltas.pdf`), Chapter 4's are PNGs. `distribuicao_estados.png` at least carries 300 dpi
(5389 px), so it downscales cleanly; `arquitetura_modelo.png` is 1102 px for a 455 pt target,
i.e. ~0.996 px/pt natural — effectively screen resolution, and it will show softness in print.

*Direction:* regenerate both at 1-column width with in-figure text set near body size — the planned
regeneration step this persona exists to verify. For Figure 3, the three-panel layout is what forces
the shrink; two rows of panels, or one panel per row, would let the axis text breathe. Combine with
V-01 for Figure 2.

### V-03 · MAJOR · Four unresolved citation markers are visible on the page

Rendered `(??)` at:

- **p. 21** — "…again to honor the spherical domain that flat sine-and-cosine features distort (??)."
- **p. 45** — "…applied to the geographic context (??), models continuous functions…"
- **p. 49** — "…distinct spatial encoding paradigms: SIREN (??), which models…"
- **p. 50** — "The SIREN model (Sinusoidal Representation Networks) (??) models a continuous function…"

Final build: same defect at pp. 16, 40, 44, 45.

The cause is a BibTeX parse failure that dropped the entry (persona 05 owns the diagnosis and the
fix). I report it here because it is a **visible page defect**: `(??)` in a defense build is the
Viegas precedent's documented defect class, and it is the first thing a reader's eye catches.

### V-04 · MODERATE · Page 71 is a floats-only page

Page 71 contains, in order: Table 10's caption, Table 10's two tabulars, the table footnote,
Figure 7, and Figure 7's caption. **No body prose.** I verified by extracting the full page text —
every line belongs to a float or its caption.

The consequence is V-05: the paragraph that reads these two floats is severed.

*Direction:* the two floats are jointly ~0.9 of a text block. Letting Table 10 take p. 71 and
Figure 7 float to the top of p. 72 would keep prose on both pages. Alternatively `[htbp]` → `[tbp]`
on Figure 7. Cheap to test.

### V-05 · MODERATE · A float pair splits the §5.6.2 results paragraph

The reading experience across pp. 70–72:

- **p. 70** ends mid-argument: "…and each gain is significant after a Holm correction across the six
  datasets (paired *t*, corrected *p* < 0.001); the four next-region gains hold under their own Holm
  correction as well (corrected *p* < 0.001)."
- **p. 71** is entirely floats.
- **p. 72** resumes: "The registered Wilcoxon test on the individual fold differences agrees at
  every dataset…"

So the sentence pair that establishes the headline statistical result is interrupted by a full page
of floats. The reader must hold the Holm result across a page turn to reach the Wilcoxon
confirmation. Both halves are on facing-away pages (70 verso, 72 recto in a duplex print), so they
are never visible together.

*Direction:* same fix as V-04.

### V-06 · MODERATE · Figures 4 and 5 carry 6.97 pt in-figure text

Measured directly from the PDF text layer (these are vector figures, so the sizes are exact, not
estimated):

| Figure | Page | In-figure text | % of body | Content at that size |
|---|---:|---:|---:|---|
| Figure 4 (dataflow) | 62 | **6.97 pt** (350 glyphs) | **58%** | "raw check-in sequence", "four-level graph", "edges: consecutive visits by a user", "features: category, hour, weekday", "trained with no task labels (infomax)", "per-visit vectors (one per check-in)", "sliding windows of 9 visits" |
| Figure 5 (joint model) | 64 | **6.97 pt** (312 glyphs) | **58%** | "check-in window (semantic stream)", "semantic encoder (private)", "shared trunk (bidirectional cross-attention)", "streams exchange information", "region output shared + private spatial path" |

Below the 80% floor, though comfortably above Chapter 4's figures and legible at print size (I
checked on a 3× render — these read fine). The content is not decorative: Figure 4's small text
carries the graph's edge and feature definitions, which is the figure's substance.

These are vector PDFs, so the fix is a font-size parameter in the generating script, not a
regeneration from scratch.

*Direction:* raise in-figure text to ≥9.6 pt. Figure 5 has whitespace to absorb it; Figure 4 is
denser and may need the caption to take over one or two of the annotation strings.

### V-07 · MODERATE · Appendix B carries a float too large for its page

`src/build/main.log`:

> "LaTeX Warning: Float too large for page by 21.55853pt on input line 504."

Line 504 of `apx_b_errata.tex`. The float still places (it lands on pp. 92–93) but LaTeX is
reporting it does not fit its allotted area, which is the warning that precedes a badly-broken table
in a later edit. It is the only float warning in the build.

*Direction:* worth resolving before the banca build so the appendix is not one edit away from
breaking.

### V-08 · MINOR · Page 42 is a four-line chapter-end runt

Page 42 carries 339 characters: the tail of Chapter 3's future-work paragraph, then the chapter
ends. Ink span 92 pt on a ~640 pt text block.

This is normal book typesetting (a chapter ends where it ends) and I would not touch it. Recorded so
it is not mistaken for a float-pressure artifact — it is not; there are no floats on the page.

### V-09 · MINOR · Front-matter sparse pages

Pages 2 (approval-sheet placeholder, 109 chars), 4 (Resumo overflow: three keyword lines, 79 chars),
8 (List of Tables overflow, 607 chars), 9 (abbreviations, 529 chars), 84 (the "Appendix" divider,
8 chars).

All structural. Page 4 is the only one worth a glance: the Resumo runs to p. 3 and pushes three
keyword lines onto p. 4, leaving a nearly blank page. Tightening the Resumo by three lines would
pull the keywords back. Cosmetic, and the Resumo's length is claim-parity-constrained (it mirrors
the Abstract), so this may not be movable. **Note:** I initially read the extracted text as showing
`ponto de interesseprevisão da próxima categoria` run together; rendering p. 4 at 3× confirms the
keywords are correctly on separate lines. Text-extraction artifact, not a defect.

### V-10 · MINOR · Figure 6 uses color + value labels, but not hatch

Page 70. Two series (Check2HGI navy, HGI grey) distinguished by color and by a legend, with the
numeric value printed above each bar (0.57, 0.98, 0.00, 0.78). WRITING_LAW §5 asks for color + hatch
dual encoding. Navy-vs-grey survives grayscale conversion at adequate contrast, and the printed
values make the figure readable even if the fill were lost entirely, so the substance of the rule is
met by a different mechanism. Recorded, not pressed.

Figure 7 (p. 71) uses navy/red with a grey ±2 pp band and printed signed values on every bar — same
verdict, and the printed values make it fully readable in grayscale.

---

## Per-chapter consistency matrix

| Chapter | Figure style | Table style | Caption style | Verdict |
|---|---|---|---|---|
| 2 (Fundamentals) | none | booktabs, tabularx, no vertical rules | above table, 11.96 pt | **consistent** |
| 3 (CBIC) | 1 bitmap, pastel boxes, English | booktabs | above | **consistent** in style; figure is bitmap |
| 4 (CoUrb) | 2 bitmaps: 1 pastel diagram (**Portuguese**), 1 matplotlib scatter | booktabs | above | **DRIFTING** — see V-01, V-02 |
| 5 (MobiWac) | 4 vector PDFs, consistent palette (navy = joint/ours throughout), thin-stroke diagrams | booktabs, `\sd{}` convention | above | **consistent, and the best in the document** |
| Appendices | none | booktabs | above | **consistent** |

**Caption placement:** captions above tables and below figures at all 22 floats. Verified on the
rendered pages, not from source. ABNT-compliant and internally uniform.

**Color threading:** the joint model is navy in both Figure 6 and Figure 7; the comparison arm is
grey in Figure 6 and red in Figure 7. Minor inconsistency in the *comparand* color, though the two
figures encode different quantities (absolute scores vs signed deltas), so a shared comparand color
would arguably mislead. Acceptable.

**The visible seam** is Chapter 4: two bitmap figures in a document whose other seven figures are
vector, one of them in the wrong language. Chapters 3 and 5 read as one document; Chapter 4's
figures do not.

---

## The two builds

- **Defense build** (94 pp): front matter renders correctly — cover (p. 1), approval-sheet
  placeholder (p. 2), Resumo (pp. 3–4), Abstract (p. 5), List of Figures (p. 6), List of Tables
  (pp. 7–8), abbreviations (p. 9), Contents (pp. 10–12), body from p. 13.
- **Final AcademicoPG build** (89 pp): body-only, starts at the lists. The four `(??)` renders
  appear at pp. 16, 40, 44, 45. Page numbering present and consistent.

Both builds carry the identical defect set. Compliance measurement belongs to persona 13; I confirm
only that both *look* right structurally.

---

## Best pages — the bar for the rest

1. **p. 71** — as a *table*, this is the best thing in the document. Two blocks, booktabs, the
   Dataset and Regions columns repeated so each block reads alone, ↑/≈ markers carrying the
   statistical verdict, unbolded matched cells, and a footnote that discloses every partial-fold and
   single-seed cell. It renders at full body size. (Its *placement* is V-04/V-05; the artifact
   itself is excellent.)
2. **p. 64** — Figure 5. A clean box-and-arrow diagram whose dashed private-path annotation carries
   real information, above a caption that names every element and ends with the thesis in six words
   ("One model, one forward pass, two predictions"). This is what WRITING_LAW §5's self-containment
   rule looks like when it is satisfied.
3. **p. 62** — Figure 4. Four-level graph, extract arrows, and the window strip, all in one
   left-to-right read. Only the 6.97 pt annotation text keeps it off the top spot.
4. **p. 70** — Figure 6 with the §5.6.2 opening. The bar chart's printed values make it readable in
   grayscale, and the prose beneath opens with the result rather than a throat-clear.
5. **p. 53** — Table 5's three-column layout is exemplary: four columns, booktabs, right-aligned
   figures, no rules. (The figure below it is V-02.)

## Open questions for the author

1. **V-01:** regenerate Figure 2 in English, or keep the published Portuguese original and say so in
   the caption? This is a fidelity-versus-frame-consistency decision only the author can make.
2. **V-02:** is regeneration of the two Chapter 4 figures in scope before the advisor build, or is
   this a banca-build item?
3. **V-07:** the oversized Appendix B float — leave (it places) or fix?

## Out-of-scope handoffs

- Persona 05: V-03's root cause is the BibTeX parse error at `references.bib:831`.
- Persona 13: page-number placement and margin compliance are measurement, not appearance.
- Persona 06/07: caption *content* truthfulness — I checked only that captions exist, sit on the
  correct side of their float, and name their elements.
