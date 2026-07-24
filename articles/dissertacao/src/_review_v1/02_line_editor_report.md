# Line-editor report — dissertation defense build v1

**Persona:** 02 · Line editor (sentence-level mechanics only; read-only, corrected text supplied never applied).
**Scope:** full defense build — front matter (`0_main.tex`), chapters 1–6, appendices A–C. Sources are the `.tex` files (not the PDF).
**Method:** automated zero-checks + whole-document mechanical scans (preposition transfer, agreement, article use, hyphenation, number style, dangling modifiers, doubled words) + close read of every source line. Comment lines (`%`) excluded from all fault counts.

---

## Zero-checks (mandatory — counts)

| Check | Count | Verdict |
|---|---|---|
| Em-dash `---` (LaTeX) | **0** | PASS |
| Em-dash `—` (unicode) | **0** | PASS |
| Contractions (English prose) | **0** | PASS |
| `can not` (wrongly split) | **0** | PASS |
| `cannot` (correct form) | 10 | OK — used, never split |

No en-dash-as-parenthetical-dash and no single-hyphen dash in rendered prose either (the one `--` hit is a math subscript, `\text{region--city}`, 4_courb.tex:197). Sentence-initial adverbials consistently carry their comma. Participial openers all attach to the correct subject (no danglers). No misattached numeric comparisons. Portuguese Resumo mechanically clean (foreign terms correctly `\emph{}`-italicized).

---

## Top 3 findings

- ★1 — Preposition **"at [a dataset / set of datasets / state]"** for reported performance: non-idiomatic and internally inconsistent (MAJOR, systematic across frame chapters).
- ★2 — Chapter 4 title uses the math-mode command **`\:`** where a plain colon is intended (MAJOR; BLOCKER if it breaks the build — unverified in this environment).
- ★3 — **Number style**: the same quantity rendered two ways ("93 percent" vs "93\%"; "90 percent" vs "90\%") across and within chapters (MAJOR consistency).

---

## Ranked findings

### ★1 · MAJOR — "at" as the reported-performance preposition (frame chapters + new frame text)
This is the persona's signature target (the systematic "at a dataset" transfer). English idiom reports model performance **on** a dataset/benchmark or **across** datasets, not **at** one; "at" for a named place-dataset reads as the Portuguese *em*-transfer. The objective fault is that the document uses **both** forms for the **same** construction, so the reader cannot tell whether a difference is intended.

Author's own idiomatic form (keep these): 5_mobiwac.tex:59 "on every dataset", :65 "on all six datasets", 1_introduction.tex:130 / 5_mobiwac.tex:54 "across the six datasets".

Non-idiomatic instances in frame prose / front matter / **new-to-chapter** frame text:
- `0_main.tex:265` (Abstract) — "…the dedicated models on the next-category task **at all six datasets** studied…" → *on all six datasets*.
- `0_main.tex:267` (Abstract) — "on the next-region task, **outperforms at four of six datasets**…" → *outperforms on four of six datasets*.
- `1_introduction.tex:114` — "category performance rose sharply **at every state tested**" → *on every state tested* (or *across the states tested*).
- `1_introduction.tex:131` — "on the category task **at all six**, and on the region task **at four of six**" → *on all six … on four of six*.
- `2_fundamentals.tex:534` — "…everywhere it is tested and on the next region **at four of six datasets**" → *on four of six datasets*.
- `6_conclusion.tex:50–51` — "on the category task **at all six** datasets … and on the region task **at four of six**" → *on all six … on four of six*.
- `6_conclusion.tex:53` — "within a two-point margin (TOST) **at Alabama and Arizona**" → *on Alabama and Arizona*.
- `5_mobiwac.tex:26–29` (NEW `[NEEDS SIGN-OFF]` preface) — "**at all six datasets** studied … the dedicated region model **at four of the six** … (…points) **at the other two**" → *on all six … on four of the six … on the other two*.
- `5_mobiwac.tex:95` (NEW recap subsection) — "the category score rose **at each of the three states** studied" → *on each of the three states*.

Direction: standardize to **on** (single dataset / quantified set) or **across** (the whole set), matching the forms already in the text. Do **not** touch the correct idiom "at all" (2_fundamentals.tex:430 "learnable at all", :528 "helps at all") or positional/temporal "at" ("at each layer", "at its best epoch", "at least ten visits", "at random", "at most"). The MobiWac **body** prose (e.g. :373 "at Alabama, Arizona, and Florida", :533 "smallest at Florida", :610 "at California") carries the same pattern heavily; those are re-typeset article text, so they route through the Chapter 5 errata/paper law rather than direct edit, but the same standardization should apply so the frame and the chapter read alike.

### ★2 · MAJOR (BLOCKER if it breaks the build) — Chapter 4 title uses `\:` for the colon
- `4_courb.tex:8` — `\chapter{Article 2: ST-MTLNet\: Spatio-Temporal Point-of-Interest Representations for Multi-Task Learning}`

`\:` is a **math-mode** medium-space command. In a text-mode chapter title it is at best a stray thin space, and with `amsmath` loaded (it is, `0_main.tex`) it can raise `\: allowed only in math mode`. The intended character is a plain colon — the identical name is punctuated with a plain `:` in this chapter's own preface (`4_courb.tex:12`, "ST-MTLNet: Representações…") and in the Chapter 3 and Chapter 5 titles (`3_cbic.tex:8`, `5_mobiwac.tex:14`). This is the **only** occurrence of `\:` in the build.

Correction: `\chapter{Article 2: ST-MTLNet: Spatio-Temporal Point-of-Interest Representations for Multi-Task Learning}` (plain colon). A chapter title also feeds the ToC, PDF bookmarks, and running head, so a stray space or build error propagates. Render-confirm was not possible in this environment (no `pdflatex`); hand the build check to persona 18.

### ★3 · MAJOR — Number style: same quantity rendered two ways
The persona hunts number-style inconsistency for the same quantity kind. Two clean same-quantity splits:
- **Song predictability bound (93):** `1_introduction.tex:38` "about **93 percent**" vs `2_fundamentals.tex:35` and `:429` "**93\%**". Same figure, two renderings, adjacent chapters.
- **90% confidence interval:** `5_mobiwac.tex:367` "**90 percent**" vs `5_mobiwac.tex:349, 553, 558, 562, 597` "**90\%**". Same concept, both forms, same chapter.

Broader mixing (context): Chapter 5 prose uses the word "percent" 7× and the symbol `\%` 6×; Chapters 2 and 4 use `\%`. WRITING_LAW §1 mandates only "digits for data quantities" and does not fix word-vs-symbol, so this is **consistency, not a law violation** — but the 93 and 90 cases are exactly what a careful examiner flags. Direction: choose one rendering per quantity kind (the symbol `\%` is the majority form) and apply document-wide.

### 4 · MINOR — "aggregate tasks gradients" (attributive-noun number)
- `3_cbic.tex:227` — "its ability to **aggregate tasks gradients** in a balanced and principled manner".

The attributive noun should be singular: **"task gradients"** (Chapter 4 writes it correctly at `4_courb.tex:82`, "Nash-MTL balancing the **task gradients**"). Correction: "aggregate task gradients". Article chapter → correct in the re-typeset text and list in Appendix B (mechanical).

### 5 · MINOR — "spatial-temporal" vs "spatio-temporal" inside Chapter 2
- `2_fundamentals.tex:64` — "HST-LSTM folds **spatial-temporal** transition context" vs the same chapter's `:68` "**spatio-temporal** correlations" and `:105` "**spatio-temporal**".

The standard adjective (and the chapter's own dominant form) is **"spatio-temporal"**. Correction: `:64` → "spatio-temporal transition context". (Chapter 3 already uses "spatio-temporal" at `:102`, `:106`.)

### 6 · NIT (batched — article chapters, errata-bound)
- **Ch.4 acronym expanded two ways:** title `4_courb.tex:8` "**Spatio**-Temporal …" vs body `:31` "ST-MTLNet (**Spatial**-Temporal MTLNet)" and `:215` "Spatial-Temporal". The acronym S-T is spelled out two ways in one chapter; pick one expansion (it is the model's proper name, so the body form "Spatial-Temporal MTLNet" is likely the intended one — then fix the title). Article chapter → errata.
- **"a MTL" article error:** `3_cbic.tex:111` "using **a MTL** approach". "MTL" opens with a vowel **sound** (/ɛm/), so it takes "an" — as the same chapter does correctly at `:48` and `:349` ("an MTL"). Correction: "an MTL approach". Article chapter → errata.

---

## What holds / what reads well (do not touch)

- **All five mandatory zero-checks pass** (em-dash, unicode em-dash, contractions, split "can not", correct "cannot"). This is a disciplined manuscript.
- **Article use is otherwise clean:** "a POI" (consonant /p/), "an LSTM/MLP/F1" (vowel sounds) all correct; only the single "a MTL" slips.
- **Hyphenation of compound adjectives is correct and consistent:** "two-point margin" / "two points below", "nine-visit windows" / "windows of nine visits", "five-fold cross-validation" / "five folds", "user-disjoint", "first-order", "top-ten" — adjective forms hyphenated, noun forms open, throughout. No drift.
- **"U.S." punctuation consistent** (19 prose uses, no bare "US", no missing dot).
- **Sentence-initial adverbials carry their comma** ("In contrast,", "Therefore,", "However,") per WRITING_LAW §1; relative pronouns written out; American spelling throughout.
- **No dangling participles, no comma splices, no doubled words** in prose (the one "spherical spherical" hit is a `.bib`-note comment, not rendered).
- **Subject–verb agreement holds**, including the deceptive "The analysis of these data **is** relevant" (`4_courb.tex:18` — subject is "analysis", singular; correct).
- **Burstiness preserved** — sentence lengths vary, sections open differently; no mechanical smoothing tell at the sentence level.

---

## Out-of-scope handoffs (noted once, not mine to rule)

- Front-matter **title placeholder** `[TITLE --- open decision NORTH_STAR §5.8]` still live in `0_main.tex` (folha de rosto, Resumo header, Abstract header) → author / persona 13 (compliance), not a mechanical fault.
- CBIC **unfilled dataset statistics** `[$N_{\text{users}}$; VERIFY…]` at `3_cbic.tex:235` → number auditor (06) / author.
- Scattered `[NEEDS SIGN-OFF]` / `[VERIFY]` / `[GATE FIX]` author markers in sources → change gate (14) / author.
- Whether the `\:` title actually errors the LaTeX build → visual/presentation reviewer (18) on the rendered pages.

---

## Open questions for the author

1. **"at [dataset]" — intended convention or transfer?** If you deliberately adopted "at [state]" as a compact per-dataset locative, say so and I will not touch it — but then the many "on/across [dataset]" instances should flip to match. My recommendation is the reverse (standardize to "on"/"across"), because that is the field idiom for performance-on-data and is already your majority form.
2. **Chapter 4 acronym:** does ST-MTLNet expand to "**Spatio**-Temporal" (title) or "**Spatial**-Temporal" (body/model name)? Fix the other to match the one you consider the proper name.

---

## Verdict

**Minor pass needed.** Mechanically the manuscript is in strong shape: all five zero-checks pass, article use and hyphenation are consistent, and there are no danglers, comma splices, or agreement errors of the kind an examiner pounces on. Two systematic items should be fixed before the advisor handoff — the "at [dataset]" preposition inconsistency (frame + new frame text) and the same-quantity number-style split — plus the Chapter 4 title `\:`, which needs a build-render confirmation because a broken **chapter title** would be conspicuous. The remaining items are local and low-risk.

_Zero-check counts: em-dash 0 · unicode em-dash 0 · contractions 0 · split "can not" 0 · correct "cannot" 10._
