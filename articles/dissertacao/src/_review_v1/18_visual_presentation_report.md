# Reviewer 18 · Visual & Presentation — report (v1 defense build)

> Persona: the rendered-pages pass. Read-only. Input: `main_defense.pdf` (87 pp) rendered
> page by page as images, plus `main_final.pdf` front-matter check and the build log.
> Common protocol: reviewers/README.md. Sources: WRITING_LAW §5, VIEGAS_ANALYSIS §2–§3.
> Started: session in progress. This file is written incrementally.

## Status: COMPLETE

## Verdict: NEEDS A VISUAL PASS

The book reads as one document at the structural level: booktabs throughout, captions
placed correctly (above tables, below figures), self-contained captions, uniform mean +/- std,
clean cross-references, both build front matters correct. It is legible and coherent. It is
not yet print-ready: four MAJOR visual defects would be caught by a careful examiner and one
(Portuguese labels inside an English-frame figure) is a visible contradiction of the frame-
language decision. None rises to "not presentable" — all are localized and fixable without
touching the science. Fix the MAJORs, then this is a clean defense build.

## Method

`main_defense.pdf` (87 pp) rendered page by page at 120 dpi (survey contact sheets) and the
float-bearing pages re-rendered at 200 dpi; every figure additionally rendered to grayscale to
test print-safety. `main_final.pdf` (83 pp) front matter checked. Build log parsed for
overfull/underfull boxes and mapped to pages. Rule-edge overflow on Table 1 measured in pixels
against the text-block width. Evidence images live in `_review_v1/hi/` and `_review_v1/sheets/`.

## Top 3 findings

1. **[MAJOR] Figure 2 (p.46) carries Portuguese labels in an English-frame chapter** — the
   outer group boxes read "Encoder Espacial", "Encoder Temporal", "Encoder Categórico",
   "Coordenadas (lat, lon)", "Timestamps (hora, dia)", "Categorias (POI graph)". The frame is
   English (CLAUDE.md decision). This is the single most visible cross-chapter drift.
2. **[MAJOR] Figure 3 (p.51) distinguishes Food from Shopping by color only (red vs orange)** —
   fails grayscale print (the two collapse to near-identical gray) and is weak even in color;
   the co-location pattern the figure exists to show becomes unreadable. Violates WRITING_LAW §5
   (grayscale-safe dual encoding; no color-only distinctions).
3. **[MAJOR] Chapter-opening titles stretch to 3-4 justified lines with mid-word hyphen breaks**
   (Ch.3 p.25 is 4 lines: "Multi-/Task", "Ca-/tegory", "Pre-/diction"). This is the exact Viegas
   defect VIEGAS_ANALYSIS §3 says not to repeat, and worse (4 lines + hyphenation). Same long
   titles drive the two-line running headers and all 30 overfull-vbox warnings.

## Findings (ranked)

### MAJOR

**M1. Portuguese labels inside Figure 2, p.46 (CoUrb architecture).**
Quote (in-figure text): "Encoder Espacial", "Encoder Temporal", "Encoder Categórico",
"Coordenadas (lat, lon)", "Timestamps (hora, dia)", "Categorias (POI graph)". The inner boxes
are English ("Location Encoder", "Category Encoder", "Shared Layers Module", "Category Output").
The chapter frame and every other figure are English. A banca reading an English dissertation
hits Portuguese diagram labels in one figure only.
Direction: regenerate Figure 2 with English labels (Spatial Encoder / Temporal Encoder /
Categorical Encoder; Coordinates (lat, lon); Timestamps (hour, day); Categories (POI graph)).
If the source diagram is not regenerable, redraw to match the MobiWac diagram vocabulary (see M4).

**M2. Figure 3, p.51 (spatial distribution) uses a color-only, grayscale-unsafe encoding.**
Caption: "Spatial distribution of POIs of the Food (red) and Shopping (orange) categories".
Red vs orange is a weak contrast in color and near-indistinguishable in grayscale (verified on
`_review_v1/hi/p051_gray.png`: both classes render as the same mid-gray dot). The figure's whole
point (Food/Shopping co-location) is lost in print. In-figure axis ticks, panel titles, and the
legend are also far below body size.
Direction: re-encode the two categories with distinct marker shapes AND a high-contrast,
grayscale-safe color pair (e.g. filled circle vs open triangle; dark blue vs light gray, matching
the MobiWac bar-chart palette on pp.67-68). Enlarge in-figure text to near body size.

**M3. Stretched, hyphen-broken chapter-opening titles.**
Evidence `_review_v1/hi/openers_compare.png`. Ch.3 (p.25) title sets on 4 justified lines with
mid-word hyphenation ("into Multi-/Task", "Point-of-Interest Ca-/tegory", "Next-POI Pre-/diction");
Ch.4 (p.41) is 3 lines with large inter-word gaps on line 2 ("Point-of-Interest    Representations
   for"); Ch.5 (p.56) is 3 lines. Ch.6 "Conclusion" (1 line) shows how clean the rest look.
Direction: set chapter titles ragged-right (no justification) and suppress hyphenation in the
title font, and/or supply a short title via `\chapter[short]{full}`. A short title also fixes M5
and the running-header overflow (M6) in one move.

**M4. Cross-chapter figure-style drift (diagram vocabulary).**
Ch.3 Fig 1 (p.33) and Ch.4 Fig 2 (p.46) are saturated red/green/orange flowchart boxes (a
"different paper" look, and different from each other). Ch.5 Figs 4-5 (pp.60,62) are a muted
pastel, thin-rule, italic-annotation vocabulary that is visibly more refined and consistent.
The three chapters' architecture diagrams do not read as one hand.
Direction: adopt the Ch.5 diagram vocabulary as the house style and redraw Fig 1 and Fig 2 to
match (box fill, rule weight, font, arrow style). Keep an "adapted from [CBIC/CoUrb]" note if the
originals are preserved for provenance. This is the highest-leverage single consistency fix.

**M5. Table 1 (p.20, lineage) overflows the right text margin by ~29.8 pt (~1 cm).**
Build log: "Overfull \hbox (29.76408pt too wide) in paragraph at lines 214-232" (the `tabular`).
Measured on the render: text block right edge x=1494 px; Table 1 rules extend to x=1577 px = 83 px
= 29.9 pt past the margin (`_review_v1/hi/t1_overflow_crop.png` shows "Reference"/"Chapter 4/5"
protruding). It looks flush only because the 2 cm right margin absorbs it; the table rule clearly
overshoots the body text and the header rule.
Direction: narrow the middle "What it added" column (it is the widest), e.g. wrap it in a fixed
`p{width}` column or shorten the phrasings, so the table fits `\textwidth`. Do not scale the whole
table down (would shrink the text below body size).

### MINOR

**m6. Two-line running headers on Ch.3 and Ch.5 overflow the header box (30 x 14.5 pt vbox).**
All 30 "Overfull \vbox (14.49998pt too high) while \output is active" warnings fall on exactly
pp.25-39 (Ch.3) and pp.56-70 (Ch.5) — the two chapters whose long titles wrap the running header
to two lines (`_review_v1/hi/header_compare.png`). Ch.2/Ch.4/Ch.6 (one-line headers) produce zero.
The header does not visibly break on the page (the overflow is absorbed at shipout), so this is
log hygiene plus a small risk of a 14.5 pt vertical shift; still worth clearing.
Direction: same short-title fix as M3 collapses the header to one line and removes all 30 warnings;
alternatively increase `\headheight`.

**m7. Italic "one-hot" bleeds ~15 pt into the right margin, p.49.**
Build log: "Overfull \hbox (15.13911pt too wide) ... lines 173-174". On the render the italic
"one-hot" at a line end crosses the text-block edge (`_review_v1/hi/p049_rightedge.png`). Does not
reach the page edge; visible only on close inspection.
Direction: rephrase the line or allow a discretionary hyphen so the italic token breaks cleanly.

**m8. p.67 is a float-only page (Table 9 + Figure 6 + Table 10 stacked).**
Build log: "Text page 67 contains only floats." p.66 is full body text ending "Figure 6 shows the
same separability contrast graphically"; the three floats then collide onto p.67. All three sit
within ~1 page of their first reference (Table 9 and Fig 6 referenced on p.66, Table 10 on p.68),
so placement is acceptable, but a page with zero body text between two full text pages reads as a
gap.
Direction: let one float (e.g. Table 9, the smallest) float earlier or nudge with `[tbp]` so p.67
carries at least a few body lines; low priority.

**m9. In-figure text below body size in the CBIC/CoUrb figures (Fig 1 p.33, Fig 2 p.46, Fig 3 p.51).**
Box labels, axis ticks, and legends are visibly smaller than ~80% of the 12 pt body (WRITING_LAW §5
floor). The MobiWac figures (pp.60,62,67,68) meet the bar. Bundled with the redraws in M1/M2/M4.

### NIT

**n10. Title placeholders still present, pp.1, 3, 4.** "[TITLE — OPEN DECISION NORTH_STAR §5.8]"
on the cover and above the Resumo/Abstract. This is a known open decision (CLAUDE.md §2 open #1),
not a defect, but it MUST be resolved before the banca build front matter ships (target ~Jul 23).
Flagged so it is not forgotten.

**n11. Approval-sheet placeholder, p.2.** "[Approval sheet placeholder — PPG signature-page model
is inserted here for the defense; signed version replaces it afterward]" — expected for the pre-
defense build; confirm the real signature page is inserted for the deposit.

## Per-chapter consistency matrix

| Chapter | Figure style | Table style | Caption style |
|---|---|---|---|
| Ch.2 Fundamentals | (only Table 1) | booktabs, caption above (OK); **T1 overflows margin, M5** | consistent |
| Ch.3 CBIC | **saturated flowchart, small in-fig text (drift, M4/m9)** | booktabs, caption above, bold = better-of-two (rule in caption) | consistent |
| Ch.4 CoUrb | **saturated flowchart + Portuguese labels (M1); Fig 3 color-only (M2)** | booktabs, caption above, bold = best-of-three (rule in caption) | consistent |
| Ch.5 MobiWac | muted/refined, English, grayscale-safe (the bar) | booktabs, caption above, bold = statistical significance + ↑/≈ legend | consistent |
| Ch.6 + apx | (errata tables only) | booktabs two-column defect/correction tables, caption above | consistent |

Reading: **caption style CONSISTENT** across the book; **table style CONSISTENT** (booktabs, no
vertical rules, caption-above everywhere) with legitimate per-chapter bold-rule differences, each
declared in its own caption (MobiWac's significance-bold is a required honesty device, not drift);
**figure style DRIFTS** — Ch.3/Ch.4 diagrams are a saturated, mutually-inconsistent flowchart look
(Ch.4 additionally Portuguese), while Ch.5 is a refined English house style. Fixing M1/M2/M4 closes
the only real consistency gap.

## Best pages (the bar for the rest)

- **p.67-68 (MobiWac results).** Table 9 and Table 10: booktabs, caption above, mean with subscript
  fold-sd, bold marking statistical significance, ↑/≈ region symbols with a footnote legend, a
  clarifying footnote on the 26.56 coincidence. Figure 6 and Figure 7 bar charts: dark-vs-light
  encoding PLUS a numeric value label on every bar, so they survive grayscale; the ±2 pp band is
  a shaded region, not a color. This is exactly the presentation law realized.
- **p.20 Table 1 (lineage), content.** The DGI -> HGI -> MTLnet -> ST-MTLNet -> Check2HGI -> joint
  model progression with a "What it added" column and per-row chapter/reference pointers is the
  clearest single object in the book (fix only its width, M5).
- **p.60, p.62 (MobiWac diagrams).** Self-contained, English, muted palette, italic side-notes; the
  "one model, one forward pass, two predictions" framing lands visually. Adopt as house style.
- **p.36 (Table 2) and p.52 (Table 6).** Clean multi-block results tables, fit the block, uniform
  formatting, lead sentence before each.

## What holds / reads well (do not touch)

- Booktabs discipline is total: no vertical rules anywhere; captions above every table and below
  every figure (fixes the Viegas inconsistency, as WRITING_LAW §5 intends).
- Captions are self-contained and interpretive (2-4 sentences, elements named, reading
  instructions and legends included).
- No undefined or multiply-defined references in the whole build.
- Both build front matters are correct: defense build (cover -> Resumo -> Abstract -> LoF -> LoT ->
  abbreviations -> Contents) and final build (lists -> sumário -> body starting p.11), page numbers
  top-right arabic throughout.
- Only 2 overfull hboxes in an 87-page book, both localized (M5, m7); the 30 vboxes share one root
  cause (m6). This is a tidy log for a three-venue re-typeset.

## Open questions for the author (only you can answer)

1. Are Figures 1, 2, 3 (CBIC/CoUrb) regenerable from source (matplotlib/draw.io/TikZ), or are they
   fixed bitmaps? The M1/M2/M4 fixes are cheap if regenerable, a redraw job if not. This decides
   the effort, not the finding.
2. Do you want to keep the "Article N:" prefix in chapter titles? It is a style choice (out of my
   scope), but dropping it or supplying a short title would shorten the titles and resolve M3 + m6
   at once.
3. Bolding rules differ per chapter (better-of-two / best-of-three / statistical significance). Each
   is declared in its own caption, so I read it as intentional and correct, not drift. Confirm you
   want them to stay per-chapter rather than harmonized. (Content truth is 06/07's call, not mine.)

## Out-of-scope handoffs (one line each)

- The "Article 1/2/3:" title phrasing and the bold-rule semantics are content/style decisions for
  03/06/07, not visual — flagged above only where they drive a layout symptom.
- Caption CONTENT truthfulness (e.g. whether "red"/"orange" in the Fig 3 caption still matches the
  figure after a re-encode) must be re-checked by 06/07 after any figure regeneration.
