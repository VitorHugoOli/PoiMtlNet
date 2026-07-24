# L3 / L4 / STYLE / BUILD gate report — assembled dissertation v1

Fresh-eyes audit (AGENT_GUARDRAILS L6). Auditor did not write any of the text under audit.
Date: 2026-07-23 (evening). Tree audited: `articles/dissertacao/src/` at the state of
2026-07-23 ~16:58 (chapters mtime 16:15–16:52). Both PDFs rebuilt from source in a clean
sandbox copy; rendered pages inspected visually; all greps run on prose lines
(comment lines excluded). Method notes at the end.

## Verdicts

| Audit | Verdict | Blocking findings |
|---|---|---|
| A. L3 cross-chapter duplication | **FAIL (2 MAJOR)** | frame repeats paper/frame phrasing outside sanctioned recaps (A-1, A-2) |
| B. L4 cross-ref lint | **PASS** | 0 undefined `\ref`/`\cite`; 137 sec/ch pointers + all tab/fig pointers semantically checked, no mismatches; 1 MINOR (unreferenced restored figure) |
| C. WRITING_LAW §7 checklist | **FAIL (1 MAJOR)** | banned idioms in Conclusion frame prose (C-1); everything else passes or is MINOR/NIT |
| D. Two-build check (UFV §1–§2) | **CONDITIONAL FAIL** | bottom margin of the text block measures ~1.5 cm against the 2 cm spec on full pages (D-1); everything else passes |

**Top 3 findings:** D-1 (bottom margin), C-1 (Conclusion idioms: "the win lives in the
shared trunk", "buys nothing"), A-1/A-2 (near-verbatim frame duplication).

---

## A. L3 cross-chapter duplication sweep

Method: sentence-level 8-gram overlap across the nine chapter files (comments stripped,
LaTeX de-texed). Sanctioned and NOT flagged: Ch.4 §4.2.5 "The MTLnet framework" recap
(4_courb.tex:79–84), Ch.5 §5.2.1 recap (5_mobiwac.tex:80–101), the three chapter
prefaces, Ch.1 §1.5 organization bullets, and Appendix B errata quoting chapter text
(quotes are its function).

### A-1. MAJOR — Introduction repeats a MobiWac results-discussion sentence nearly verbatim
- `chapters/1_introduction.tex:76`: "concrete: one artifact to train, version, and
  deploy, and one forward pass whose single set of inputs produces both answers at once,
  instead of two dedicated single-task models running side by side."
- `chapters/5_mobiwac.tex:256` (paper chapter, Discussion): "than arithmetic: one
  artifact to train, version, and deploy, and one forward pass whose one set of inputs
  produces both answers at once."
- Why it matters: the frame repeating paper-chapter wording is exactly the L3 defect
  class; a banca reader hits the same coined phrase twice ~50 pages apart.
- Direction (not applied): reword the Introduction instance (the paper chapter is the
  version of record and cannot move).

### A-2. MAJOR — Introduction repeats the Fundamentals "weekday lunch" argument sentence
- `chapters/1_introduction.tex:116–117`: "Any place-level embedding assigns a place the
  same vector on every visit; a representation that cannot tell a weekday lunch from a
  Saturday night out at the same place is working against both tasks at once."
- `chapters/2_fundamentals.tex:502–504`: "…any place embedding assigns a place the same
  vector on every visit. A representation that cannot tell a weekday lunch from a
  Saturday night out is working against both tasks at once…"
- Both are frame chapters; neither is a sanctioned recap. The image is memorable, which
  makes the repetition more visible, not less.
- Direction: keep it in ONE chapter (Fundamentals §2.5 is the natural home, where it
  hinges to Ch.5) and let the other allude ("a representation blind to the visit context,
  as Chapter 2 argues, works against both tasks").

### A-3. MINOR — Introduction §1.2 and Conclusion §6.1–§6.2 share several verdict formulas
8 sentence pairs share 8-grams, e.g. "did not consistently outperform … and it cost more
to train" (1_introduction.tex:101 ↔ 6_conclusion.tex:28), the six-dataset verdict
sentence (1_introduction.tex:125ff ↔ 6_conclusion.tex:50–53), and the research-question
restatement (intro:143 ↔ conclusion:14). Partial defense: the verbs-bound-to-tests law
forces identical verdict wording, and an intro-arc/conclusion echo is a normal
dissertation shape. Still, the *connective* prose around the verdicts is also near-identical
in places. Direction: vary the surrounding sentences in Ch.6 (the verdict clauses may stay
identical by law).

### A-4. MINOR — "The exact next place is not predicted anywhere in this work" appears verbatim twice
`chapters/1_introduction.tex:168` and `chapters/6_conclusion.tex:113` (plus paraphrases at
2_fundamentals.tex:56 and :496). WRITING_LAW §7 asks for the delimitation "once, early".
The Conclusion limitations-list entry arguably needs it, but the verbatim repetition is
avoidable. Direction: in Ch.6 rephrase ("the conclusions concern the next category and
next region targets only" already follows and suffices).

### A-5. INFO — CBIC ↔ CoUrb methodology overlap (not flagged)
5 sentence pairs (e.g. task definitions, FiLM equation prose, "ordered chronologically per
user; users with fewer than five visits are discarded" at 3_cbic.tex:149 ↔
4_courb.tex:123). Both are published papers reproduced under §2.6.4; overlap is in the
sources themselves. No action; recorded for completeness.

---

## B. L4 cross-reference lint

- Compile: **0 undefined references, 0 undefined citations** in both `main_defense.log`
  and `main_final.log` (grep for "Reference … undefined" / "Citation … undefined": empty).
- Inventory: 94 labels, 178 `\ref` instances. All resolve.
- Semantic check: every sec/ch/apx pointer (137) was mapped to its target heading and read
  with its citing sentence; every tab/fig pointer was checked for word/target agreement
  (Table→tab:, Figure→fig:). **No mismatched pointer found** (the Viegas defect class is
  absent). Spot-verified deeper: the "Section 5.2.2" pointer at 5_mobiwac.tex:226 correctly
  points at the subsection where Check2HGI-vs-prior-work is discussed; the two Ch.4 preface
  pointers to Ch.3/Ch.5 are correct; the Ch.2 lineage-table pointer is correct.

### B-1. MINOR — restored figure is never referenced in prose
`fig:mobiwac:embquality` (5_mobiwac.tex:430, the class-separability figure restored from
the 8-page cut, [NEEDS SIGN-OFF] block at :417) is the only float with no `\ref` anywhere.
Its four numbers are stated in prose at 5_mobiwac.tex:370, but the figure itself is never
called out, so it floats unanchored (and triggers the "Text page 67 contains only floats"
build warning). Direction: add "(Figure~\ref{fig:mobiwac:embquality})" to the silhouette
sentence at :370, or drop the figure to match the submitted article.

---

## C. WRITING_LAW §7 checklist (prose lines only)

### C-1. MAJOR — banned idioms in Conclusion frame prose
- `chapters/6_conclusion.tex:80`: "Parameter count alone, without the second task's
  training signal, **buys** nothing here" — money-metaphor verb, WRITING_LAW §4 idiom rule
  names "buys" explicitly.
- `chapters/6_conclusion.tex:82`: "…the sentence Chapter 5 already states as its finding,
  that **the win lives in** the shared trunk" — phrasal metaphor of the banned family
  ("the win lives in"); also "win" as a result noun brushes the banned verdict-verb family
  (§3: never "beats"/"wins").
- `chapters/6_conclusion.tex:126`: "a possible contributor to the size of **the win**" —
  same result-noun concern.
- These are frame prose (full-force zone, §6). Direction: "adds nothing here" / "the gain
  originates in the shared trunk" / "the size of the gain".

### C-2. MINOR — "frozen" outside the glossed-weights exception
`chapters/6_conclusion.tex:69`: "with the region pathway frozen". The law allows "frozen"
only for weights, glossed; the gloss lives in Ch.5 (:355 "frozen weights (no fine-tuning)"
and :564 "We freeze the region pathway…"), not in Ch.6. Direction: "with the region
pathway fixed during training" or repeat the short gloss.

### C-3. MINOR — "not only … but" template in the CoUrb translation
`chapters/4_courb.tex:345`: "the main gain of this chapter lies not only in the choice of
a specific spatial architecture, but in the very adoption of decoupled representations…"
(also a softer instance at :213). Banned template (§4) — but this is translated published
prose, where fidelity beats style (L5). Direction: author decision; if the PT original has
"não apenas … mas", keep it and let Appendix B's wording-table note it.

### C-4. MINOR — placeholder em-dashes will print if placeholders survive
Chapter prose has **zero** em-dashes (both `---` and U+2014 greps clean). But
`0_main.tex:104/111/112/113/153/163/225` use `---` inside the [TITLE], [Banca member],
[defense date], and [Approval sheet] placeholders, and these DO render as em-dashes in the
built PDFs today. Direction: when the real title/banca/date land, confirm no em-dash
remains; consider switching the placeholder separator to a colon now to remove the trap.

### C-5. NIT — "venue" occurrences (5) are all publication-venue sense
`1_introduction.tex:216` and four in `apx_b_errata.tex` (:239–:251). The GLOSSARY ban
targets "venue" as a word for a *place/POI*; none of these is that sense. No action
required; noted because the grep will keep hitting them.

### C-6. NIT — "at the event" for a workshop
`chapters/1_introduction.tex:205`: "presented the paper at the event" — "event" is banned
as a synonym for check-in; this is the conference sense, so legal, but cheap to harden:
"at the workshop".

### C-7. NIT — reproduced-paper tics (time-capsule text, no action)
`3_cbic.tex:290` "notably" and the CBIC chapter's higher -ly density (1.35% vs 0.35–0.49%
in the frame) are properties of the published text, which §6 says to preserve. Appendix B
already documents the wording substitutions that WERE made (leverage/underscore/moreover →
plain forms). Consistent and correct; recorded so the next auditor does not reopen it.

### Checklist items that PASS (evidence)
- **Canonical names:** no "venue"-as-place, no "event"-as-check-in; next category / next
  region / next place kept distinct (Ch.1 §1.1 and Ch.2 §2.1 both delimit); no bare
  "baseline" for the dedicated model in the frame (all 11 frame hits are compound:
  "MTLnet baseline", "fixed-weight baseline", "capacity-matched … baseline", or refer to
  reference floors); "the joint model" / "dedicated single-task model" used consistently.
- **Repo codenames: zero** in prose. B9/v11–v17/champion-G/H3-alt/dk_ovl/log_T/substrate/
  engine/board/recipe: all hits are in comments or the quoted BRACIS manuscript title in
  Appendix A ("Substrate Carries, Architecture Pays…"), which is a factual title and exempt.
  The repo id `mtlnet_crossattn_dualtower` appears nowhere.
- **AI-tell sweep:** banned words zero in prose (all grep hits are Appendix B's
  published-vs-chapter wording table, i.e., quotes of the *published* text); no
  moreover/furthermore/firstly openers; intensifier scan of frame chapters: zero
  very/highly/extremely/dramatically; -ly density 0.35–0.49% in frame (in band);
  one two-adverb sentence ("daily … occasionally", 2_fundamentals.tex, content-bearing,
  acceptable); chapter openers use three distinct shapes (scene-setting / purpose
  statement / question restatement).
- **Em-dash = 0 and contractions = 0** in all chapter prose (strict grep, comment-excluded).
- **Verbs bound to tests:** every frame "outperforms" is scoped to the category task or
  to the four named region datasets; "matches" always rides with TOST/two-point margin
  (2_fundamentals.tex:444–451 defines the binding explicitly; 6_conclusion.tex:50–53 and
  the Abstract both keep it); **AZ is never upgraded** (5_mobiwac.tex:553 reports the
  zero-centered interval as a match; 6_conclusion.tex:70 lists Arizona only in the freeze
  control); no "beats"/"wins"/"ties"/"Pareto" as verdict verbs in the frame (the two
  "Pareto" hits are Ch.3's published method text describing MGDA/Nash-MTL, legal).
  Residual: C-1's "the win" noun.
- **Universals scoped:** "everywhere" appears twice, both immediately scoped
  ("everywhere it is tested" 2_fundamentals.tex:530 after the six datasets are named;
  "on the category task everywhere" 6_conclusion.tex:64 two sentences after the
  six-dataset enumeration at :50–53). Borderline but within the law's letter.
- **Time-indexing:** CBIC preface (3_cbic.tex, "conclusions of the time, for the
  configuration studied here", Nash-MTL caution), CoUrb preface (protocol caveat,
  "conclusions … of the time"), Ch.2 :374 ledger and Ch.4 :84 body repeat it. Present
  and consistent.
- **Resumo ↔ Abstract claim parity: PASS.** Clause-by-clause read of both (rendered
  pages def_02/def_03): same claim sequence (LBSN framing → two tasks → MTL promise →
  negative-transfer hedge → three-study arc → null result → representation diagnosis →
  check-in level + cross-attention → protocol (users disjoint, 20 repetitions = 4×5,
  paired tests, non-inferiority, leakage audit) → 5.3–9.4 macro-F1 at all six → region
  4/6 + TOST two-point at the other two → conditional-answer close). Numbers match
  (5,3–9,4 ↔ 5.3–9.4; vinte/quatro/cinco ↔ twenty/four/five); hedges match (supera /
  equipara-se ↔ outperforms / matches); keywords mirror 1:1 in order, one per line,
  lowercase. No claim present in one and absent in the other.
- **"this paper" residue: zero.** "Dataset N" prose: zero. Lead takeaway sentences
  precede results tables; no literal "Read this as:" tag (5_mobiwac.tex:584 "We read
  this as a defense of…" is a normal sentence, not the banned tag).
- **Caption placement:** all `table`/`longtable` environments have `\caption` above the
  tabular; all `figure` environments have `\caption` below the graphic; no vertical
  rules in any tabular spec (booktabs only). Verified by position scan over every float
  in the nine files.

---

## D. Two-build check (UFV_COMPLIANCE §1–§2)

Build: clean copy, TeX Live 2026 basic + the repo usermode tree. **Both PDFs build with
zero errors** and zero undefined refs/cites. Defense: **87 pages**; final: **83 pages**.

Reproduction note (README_SRC gap): in a fresh environment the build died fatally
("Font t1xtt at 657 not found") until `TEXMFVAR` was pointed at the usermode tree's
`.texmf-var` (where the updmap-generated `pdftex.map` includes the txfonts map). On the
author's machine the default var tree evidently works, but README_SRC.md only mentions
`TEXMFHOME`; one sentence ("if t1xtt errors appear, also export
`TEXMFVAR=$TEXMFHOME/.texmf-var` or run `updmap-user`") would make the recipe portable.

### Structure — PASS
- **Defense build order** (rendered pages 0–11 inspected): folha de rosto (author, title
  placeholder, preambulo, Florestal-MG/2026) → approval-sheet placeholder page → Resumo
  (PT, UFV catalog header) → Abstract (EN, catalog header) → List of Figures → List of
  Tables → siglas → Contents (3 pp) → body Chapter 1. Matches UFV §1 build 1. (No cover
  page beyond the folha de rosto; the Germano/abnTeX2-UFV flow starts at the folha de
  rosto — flagging for author awareness only, since the secretariat model may add a capa.)
- **Final build order** (pages 0–9 inspected): starts directly at List of Figures → List
  of Tables → siglas → Contents → body. No Resumo/Abstract/folha de rosto in the PDF.
  Matches UFV §1 build 2 (system generates the pre-textuals).
- **Pre-body pages unnumbered — verified visually/pixel-scan:** defense pages 1–11 and
  final pages 1–7 have zero ink in the top-right corner; the first numbered page is the
  first body page in both builds.
- **Page numbers:** top-right, arabic. Defense: first body page prints **12** after 11
  counted-not-numbered pre-textual pages — consistent with the manual's counting rule.
  Final: first body page prints **11** via `\finalbuildfirstpage{11}` — carries its own
  [VERIFY] flag in `main_final.tex` to be tuned against the AcademicoPG RASCUNHO PDF
  (correct process; leave the flag standing).
- **A4:** page size 595.28×841.89 pt = 21.0×29.7 cm in both PDFs. ✓
- **1.5 spacing:** measured baseline-to-baseline ≈ 0.64 cm on body pages ✓ (spot-read of a
  rendered page shows clearly one-and-a-half spacing; `\OnehalfSpacing` in force).
- **Font:** embedded fonts are TeX Gyre Termes (+X variants), NewTX math, and
  TimesNewRomanPSMT only — Times everywhere including headings (the qhv/Heros request is
  remapped; no Computer Modern, no Type 3 in the text). ✓
  NIT (D-3): the regenerated figures embed DejaVu Sans/DejaVu Serif (matplotlib defaults),
  so in-figure text is not Times. Not a manual §8 violation (the rule governs the text),
  but the visual-presentation reviewer (18) may want matched fonts.

### D-1. MAJOR (verify before ship) — bottom margin of the text block ≈ 1.5 cm, spec says 2 cm
Measured on rendered pages at 200 dpi (ink bounding box), across all body pages of both
builds: left = 3.00 cm ✓, right = 1.96–2.02 cm ✓, header top line = 1.9 cm with body text
starting at 3.1 cm ✓ (header sits in the margin band like the page number, standard), but
the **lowest text baseline reaches ≈ 1.52 cm from the bottom edge on every full page**
(min over 70 body pages: 1.52 cm; typical full page: 1.52–1.56 cm). The manual §7 spec is
bottom = 2 cm. Cause hypothesis: `abntex2-UFV.sty` sets the block with
`\setulmarginsandblock{3cm}{2cm}{*}` + `\checkandfixthelayout`, but `\OnehalfSpacing` is
applied at `\begin{document}` (0_main.tex), *after* the layout is fixed; memoir's
`\checkandfixthelayout` rounds `\textheight` to an integer number of *single-spaced*
lines, and the 1.5-spaced grid then overshoots the nominal block by roughly one line
(~0.45 cm). Germano's tree presumably shipped the same geometry and passed, so this may
be tolerated in practice — but it is a measurable deviation from the written spec on
every page. Direction: re-run `\checkandfixthelayout` after setting the spacing, or set
`\textheight` explicitly so the last baseline sits ≥ 2 cm from the edge; then re-measure.

### D-2. NIT — "Text page 67 contains only floats" (defense build)
Consequence of the unanchored restored figure (B-1). Fixes itself with B-1.

### Open placeholders visible in both builds (expected, tracked elsewhere)
Title ("[TITLE — open decision NORTH_STAR §5.8]" on folha de rosto + both catalog
headers), banca members, defense date, approval sheet. These are known open decisions
(CLAUDE.md §2); listed here only so the pre-ship checklist has them.

---

## What holds / what reads well (do not touch)
- The cross-reference discipline is excellent: 178 pointers, zero dangling, zero
  semantically wrong — rare at this document size.
- The verbs-bound-to-tests law is enforced *everywhere it matters*, including the
  Abstract/Resumo pair, Table 10's caption conventions, and the AZ zero-centered interval
  (reported as a match, never upgraded).
- Resumo ↔ Abstract parity is genuinely clause-for-clause; the pair reads as one text in
  two languages.
- The chapter prefaces (time capsules) do exactly what NORTH_STAR §3 asks: venue, status,
  contribution note, and what later chapters revise — and the CoUrb preface's protocol
  caveat (sample-stratified vs user-disjoint) is stated plainly in the body too
  (4_courb.tex:224).
- Caption/table hygiene (captions above tables, below figures, booktabs-only) is
  uniform across all nine files — the Viegas inconsistency is fixed as planned.
- Frame -ly density, intensifier count, and paragraph-shape variety are all inside the
  §4 bands; the AI-tell surface of the frame is low.

## Out-of-scope handoffs
- **To 06 (number auditor):** the Conclusion's capacity-matched baseline numbers
  (56.16 / 56.82 / 64.54 at Alabama, "fifteen of twenty repetitions" at California,
  6_conclusion.tex:73–83) are frame-level analysis numbers not present in any paper
  chapter — they need a source-of-truth trace (RESULTS_BOARD or a study file).
- **To 08 (translation fidelity):** C-3's "not only … but" and the courb translation's
  `\textit{}` loanword styling (encoders/embeddings/folds/dataset italics) are fidelity
  vs. style calls.
- **To 18 (visual presentation):** D-3 figure fonts (DejaVu vs Times); Figure 3's legend
  overlapping the Texas panel's x-axis labels in the rendered PDF (fig3, page ~51 defense
  build) is worth a look.
- **To 13 (UFV compliance) / author:** D-1 margin decision (fix vs. document Germano
  precedent); whether the secretariat expects a capa page before the folha de rosto in
  the defense PDF; README_SRC TEXMFVAR note.
- **To author:** B-1 sign-off block (restored figure) still carries [NEEDS SIGN-OFF];
  the Ch.5 preface and Abstract/Resumo drafts likewise. These are tracked in-source and
  not re-flagged as defects.

## Method notes
- Duplication: 8-gram sentence overlap on de-texed, comment-stripped chapter text;
  sanctioned zones excluded by location.
- Cross-refs: label/ref inventory by regex; every sec/ch pointer mapped to its target
  heading text; tab/fig pointers checked for preceding-word agreement; compile logs
  grepped for undefined refs/cites.
- Style greps: comment lines excluded via leading-% filter; contraction/em-dash/banned-word
  patterns per WRITING_LAW §1/§2/§4; -ly density computed on de-texed word counts.
- Build: clean tree copy; TEXMFHOME + TEXMFVAR at the repo usermode tree; pdflatex ×3 +
  bibtex per mode; pages rendered with pypdfium2 at 150–200 dpi; margins measured from
  ink bounding boxes at 200 dpi (±0.05 cm); fonts read from embedded BaseFont records.
