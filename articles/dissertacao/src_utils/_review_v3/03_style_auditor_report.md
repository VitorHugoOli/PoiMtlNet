# 03 · Style auditor: the G3 style gate (v3)

**Build audited:** `src/dissertacao.pdf`, **103 pages**, rebuilt 2026-07-27 19:06, against
`src/0_main.tex` + `src/chapters/*.tex` + `src/tables/**/*.tex` at 2026-07-27 (5_mobiwac.tex 18:56,
the newest chapter file). **Date:** 2026-07-27. **Persona:** `reviewers/03_style_auditor.md`.
Read-only, fresh eyes, nothing edited. This gate's output is quantitative: every metric below
carries its number.

## SUPERSEDES the v2 report (`../_review_v2/03_style_auditor_report.md`, dated 2026-07-26)

**What changed since v2.** v2 audited a **94-page** build; this one is **103 pages**. Two whole
appendices that v2 never saw are now in the document and are audited here for the first time:
**Appendix D** (*A Label-History Benchmark for the Next-Category Task*, `apx_d_ceiling.tex`, 1,162
words) and **Appendix E** (*Data Ethics and Governance*, `apx_e_ethics.tex`, 857 words). The tables
were extracted out of the chapters into `src/tables/**` (16 files) after v2 ran, so v2's counts did
not include table body text; **this audit includes it**. Chapter 6's §6.2 second control was
rewritten from the interim "partial California run, fifteen of twenty repetitions" to a completed
run. Appendix A §A.2 was removed and the BRACIS entry was dropped from the abbreviation list.

**What v2 got wrong or missed, stated plainly.**

1. **v2 never applied the Chapter 5 glossary law.** The persona brief, line 27, makes
   `articles/[mobiwac]/GLOSSARY.md` binding for Chapter 5 and says it *wins* for that chapter.
   v2 contains **zero** references to that file. Its §3 jargon table (26 rows, each with a
   keep/gloss/avoid/never verdict) and §4 "words to avoid or always explain" were therefore never
   linted. **§1 of this report is that lint, row by row.** It found two live `never`-verdict
   violations that v2 passed over (S3-02), plus the `sits`/`lies` verb split that the paper
   glossary §8 rules on (S3-07).
2. **v2's "GATE PASS" rested on a defect-free front matter that is not defect-free.** Four
   sentences in the Resumo/Abstract pair are torn: a clause is missing and the remainder renders
   starting in lowercase, on rendered pages **3 and 4**. `check_trapped_prose.py`'s docstring
   enumerates nine prior instances of a related failure mode; these four are a **tenth through
   thirteenth** of a different shape that the detector cannot see (no comment is involved). v2
   reported "Contractions in prose 0 / repo codenames 0" for `0_main.tex` and did not read the
   abstract as rendered prose. **S3-01, BLOCKER.**
3. **v2 counted `\emph{seed}` in Chapter 5 but never counted italics as a register signal.** No v2
   report line audits `\textit`/`\emph` usage as a register question. Chapter 4 italicizes ordinary
   English 155 times, including the same word both ways in the same paragraph. **S3-03.**
4. **v2's density table is superseded, not contradicted.** Recomputed over a corpus that now
   includes table text and two new appendices: Ch.3 1.47% (v2: 1.69%), Ch.4 1.04% (v2: 1.31%),
   Ch.5 0.42% (v2: 0.46%). The v2 numbers were right for the corpus v2 measured. The **shape** of
   the finding is unchanged: only the two reproduced paper chapters exceed the band.
5. **v2's proposed law updates are still open and are re-proposed here** (parallel-list semicolon
   exemption; "fitted models" into GLOSSARY §4/§6). Neither has landed; "fitted models" is now at
   four sites in text including the Abstract and Resumo.

**Also: one violation the author found himself and fixed before this run.** `arm` (paper glossary
§3, verdict *never*, "clinical-trial word, foreign to this audience") had been introduced into
Chapter 5's limitation sentence by this round's edits. Confirmed removed: `\barms?\b` returns
**0 prose hits document-wide**. It is not re-reported. It was used as calibration for §1.

---

## Verdict

**GATE FAIL.**

One BLOCKER: four torn sentences in the Resumo/Abstract pair (S3-01). Under the gate's own
definition a rendering defect in the two paragraphs the banca and the AcademicoPG reviewer read
first is not a style nit, and the Resumo/Abstract are a mandated claim-parity pair, so the same
defect is present twice in two languages.

Everything else is MAJOR or below. The em-dash count in prose is 0, contractions are 0, repo
codenames are 0, `mtlnet_crossattn_dualtower` is 0, `substrate`/`recipe`/`arm`/`checkpoint` are 0,
banned-word hits in frame prose are 0. Variance is not compressed: Chapter 6 and the front matter
carry the highest sentence-length dispersion in the document (CV 0.667 and 0.659), and the new
Appendix D reaches 0.567, above every reproduced paper chapter.

## Top 3 findings

1. **S3-01 · BLOCKER.** Four torn sentences in the Resumo and Abstract, rendered pages 3 and 4.
   Each drops a clause and restarts mid-sentence in lowercase.
2. **S3-02 · MAJOR.** The paper glossary's `never` verdict violated twice in Chapter 5: result-**"cell"**
   used in our own prose at `5_mobiwac.tex:588` (p. 71) and `:689` (p. 72).
3. **S3-03 · MAJOR.** Chapter 4 italicizes ordinary English words 155 times (`encoder` ×14,
   `baseline` ×16, `embedding` ×18) and is internally inconsistent: the same words appear plain
   in the same chapter. The advisor's "the way some terms are inserted sounds strange" has a
   typographic component, and this is it.

---

# 1 · The Chapter 5 glossary lint (`articles/[mobiwac]/GLOSSARY.md` §3, row by row)

The brief's instruction: judge each hit against the table's own verdict column, and distinguish a
term describing **another paper's method** (usually legitimate) from the same word in **our own
claims** (not). Counts are prose only: LaTeX comments excluded, math replaced by a token,
`5_mobiwac.tex` plus its four table files included.

| # | Row (jargon) | Verdict in the table | Hits in Ch.5 | Assessment |
|---|---|---|---:|---|
| 1 | embedding / representation | gloss once, then "representation" | 20 | **PASS.** Glossed at `:110-116` before use; "representation" is the running word. |
| 2 | per-visit / contextual | keep; this is our key idea | 18 | **PASS.** Load-bearing throughout. |
| 3 | substrate | avoid (repo word) | **0** | **PASS.** |
| 4 | graph | keep, gloss once | 22 | **PASS.** Glossed at `:280` ("a graph with four levels"). |
| 5 | infomax objective | gloss; skip the formula | 7 | **PASS.** Glossed at `:280` ("each vector learns to match its real neighborhood and reject a shuffled one"). No formula. |
| 6 | hierarchical | keep | 9 | **PASS.** |
| 7 | multi-task / parameter sharing | keep, gloss "sharing" once | 12 | **PASS.** Sharing mechanism spelled out at `:292-293`. |
| 8 | single-task ceiling | keep; "dedicated model" | 0 bare | **PASS** (see row 19). |
| 9 | negative transfer | gloss; use the plain phrase | 0 in Ch.5 | **PASS.** Glossed in the frame (`0_main.tex:300`, `3_cbic.tex:58`). |
| 10 | non-inferiority (TOST) | gloss once with the 2-point margin | 14 | **PASS.** Full form at `:68`; margin named every time. |
| 11 | Markov / transition baseline | keep, gloss once | 7 | **PASS.** Glossed at `:392` ("predicts the category that most often follows the recent ones"). |
| 12 | overlapping (stride-1) windows | "sliding windows" is the ONE name; the definitional site says "overlapping sliding windows … one starting at each visit" ONCE | 2 | **PASS, exactly as ruled.** `:342` reads "time-ordered overlapping sliding windows of nine visits, one starting at each visit". Zero later uses of "overlapping windows" as a second name. The `non-overlapping` contrast survives only in Ch.3/Ch.4 where it names those chapters' own protocol. |
| 13 | transductive | gloss only where the leak discussion needs it | 1 | **PASS, and tight.** Single use, inside the leak discussion (`:346`), with its own gloss attached ("what a graph fitted over all users passes to the test side…"). |
| 14 | ablation | keep, gloss once | 3 | **PASS.** |
| 15 | seed | "seed" IS the term; banned compounds "multi-seed run", "single-seed states", "seed by seed" | 15 | **PASS on the compounds** (all three at 0) but see **S3-05** for arrival order. |
| 16 | fold | covered by the 5-fold CV gloss | 41 | **PASS.** |
| 17 | **arm** | **never** (clinical-trial word) | **0** | **PASS.** The author's own fix, verified, not re-reported. |
| 18 | **cell** (a table result) | **never in prose**; grid sense written **"grid cell"** in full | 7 | **FAIL ×2 → S3-02.** Three are correct "grid cell" in full (`:144`, `:145`, `:155`); two are the deliberate networking disambiguation the glossary itself sanctions (`:264` "not a radio cell", `:266` "Cell association and handover", both p. 62); **two are the banned result sense in our own prose** (`:588`, `:689`). |
| 19 | ceiling | keep **only** with the gloss "the level a joint model is expected at best to match" | 6 | **PASS in Ch.5, with a frame-order caveat → S3-06.** The gloss is present at `:470` ("the level a joint model is usually expected, at best, to match"); the five later uses are all "dedicated … ceiling", the licensed compound. |
| 20 | checkpoint | prefer "saved model"; gloss if kept | **0** | **PASS.** `:472` reads "one saved artifact per fold". |
| 21 | frozen (folds/weights) | "fixed" for folds; "frozen" only for weights, glossed | 3 | **PASS.** All three are weights (`:394` "with frozen weights (no fine-tuning)", glossed in place; `:409`; `:590` "freezing the region stream"). Folds are "fixed" everywhere. |
| 22 | recipe | avoid; ML-blog register | **0** | **PASS.** |
| 23 | lift (noun) | noun is slang; the verb sparingly | 1 | **PASS.** Verb, once, at `:767`. |
| 24 | epoch | keep, gloss once | 8 | **PASS.** Glossed at first Ch.5 use, `:409` ("an epoch is one pass over the training data"). The parenthesis falls inside an already-crowded sentence (see S3-04). |
| 25 | end-to-end | gloss once or avoid | 1 | **PASS.** `:392` glosses it in place ("trained as one piece from the raw check-ins"). |
| 26 | from raw / from-raw | never the bare ellipsis | 0 bare | **PASS.** `:688` reads "from the raw check-ins", the full form. |

**Paper glossary §4 (words to avoid or always explain), separately:**

| Rule | Result |
|---|---|
| "activity"/"area" for the two tasks | **PASS, and the distinction the brief asked for holds.** All 9 document-wide "activity" hits describe **other** systems' own targets: MCARNN (`2_fundamentals.tex:351`, `5_mobiwac.tex:166`), CSLSL's causal chain (`:353`), DRRGNN's "next activity region" (`5_mobiwac.tex:158`), iMTL (`3_cbic.tex:123`), MTPR (`4_courb.tex:82`), and "activity-region prediction" as the name of a research line (`2_fundamentals.tex:93`). **Zero** describe our tasks. "area" appears twice, neither as a task name. |
| "Pareto" / "Pareto-dominate" | **PASS.** 3 hits, all `3_cbic.tex:95-108`, all naming MGDA's own property ("Pareto-optimal descent directions"). Never a verdict about our result. |
| architecture names glossed, not just named | **PASS.** cross-attention glossed at `5_mobiwac.tex:53` and `:290`; the trunk introduced once as "a shared cross-attention stack (the trunk)". |
| "SOTA" | **PASS, 0.** "state of the art" spelled out. |
| **"margin" reserved for the TOST margin** | **MINOR → S3-08.** 16 of 24 uses are the TOST margin. Six are a *different* named margin with its own definition (the screening margin, Appendix D `:71` "the screening margin of three points") and read cleanly. One is `4_courb.tex:325` "outperforming DGI by a wide margin" in reproduced CoUrb prose, which is the exact phrase the paper glossary §4 lists for renaming to "gap". |
| "arm(s)" and result-"cell(s)" | arms **0**; cells **FAIL ×2** (S3-02). |
| every number carries its reference point | existence verified; content is persona 06/07's gate. |
| every table has a lead takeaway sentence, never the literal "Read this as:" | **PASS.** Verified on all four Ch.5 tables; `Read this as` 0 document-wide. |
| self-praising "audited"/"rigorous"/"principled" about our own method | **1 borderline**, `3_cbic.tex:280`: Nash-MTL aggregates gradients "in a balanced and **principled** manner". Reproduced CBIC prose, describing the cited optimizer rather than our own method. Recorded, no action recommended. |

---

# 2 · The advisor's charge, operationalized

> "soa um pouco estranho o jeito que alguns termos sao inseridos"

Five measurable failure shapes, per chapter. Counts are prose only.

| Chapter | words | terms used **once** doc-wide (named methods) | other technical tokens used once | gloss **pile-ups** (≥2 parenthetical definitions in one sentence) | non-statistical semicolon braids |
|---|---:|---:|---:|---:|---:|
| Front matter | 1,115 | 0 | 1 | 0 | 0 |
| Ch.1 Introduction | 1,863 | 0 | 1 | 0 | 0 |
| **Ch.2 Fundamentals** | 4,190 | **14** | **9** | 0 | **5** |
| Ch.3 CBIC | 5,115 | 3 | 15 | 1 | 2 |
| Ch.4 CoUrb | 4,891 | 6 | 11 | 2 | 0 |
| **Ch.5 MobiWac** | 7,814 | 2 | 2 | **6** | 0 |
| Ch.6 Conclusion | 1,643 | 0 | 0 | 0 | 0 |
| Apx A | 325 | 0 | 0 | 1 | 0 |
| Apx B Errata | 3,417 | 0 | 3 | 0 | 0 |
| Apx C AI | 370 | 0 | 2 | 1 | 1 |
| Apx D Benchmark | 1,162 | 0 | 0 | 0 | 0 |
| Apx E Ethics | 857 | 0 | 1 | 0 | 0 |

**Reading.** The advisor's instinct localizes to **two chapters, for two different reasons**:

- **Chapter 2 is where terms arrive and never return.** Twenty-five named methods appear exactly
  once in the entire 103 pages, and **fourteen of them are introduced in Chapter 2**:
  `uncertainty weighting` (`:316`), `CAGrad` (`:322`), `Aligned-MTL` (`:327`), `FAMO` (`:329`),
  `random loss weighting` (`:331`), progressive layered extraction / PLE (`:298`), `DSelect-k`
  (`:300`), the graph convolutional network (`:144`), `GraphSAGE` (`:147`), `Deep InfoMax` (`:154`),
  `DeepWalk` (`:139`), `Space2Vec` (`:204`), `GeoSAN` (`:76`), `HST-LSTM` (`:70`). Two neighbours are
  near-singletons: `GradNorm` ×2 and `cross-stitch` ×5, every instance a survey mention, never a
  method the dissertation uses. This is a legitimate survey shape, since the fundamentals
  chapter's job is to name the field, and the persona's remit is not to ban it. What makes it *sound* strange is the
  **delivery rate**: `2_fundamentals.tex:315-335` introduces nine balancers in twenty-one lines,
  each in one clause, each with a citation, five of them carrying a technical noun that also appears
  exactly once in the document (`homoscedastic`, `condition number`, `principal components`,
  `scalarization`, and, three paragraphs earlier, `spectral rule` and `aggregator functions`).
  See **S3-10**.
- **Chapter 5 is where glosses stack.** Six sentences carry two or three parenthetical definitions
  each. Two are severe (**S3-04**).

**Definition-order audit, all 96 registry and lineage terms.** Ten terms are first used before
their definition arrives; of those, **seven are correct by design** (the frame states a claim, the
paper chapter defines the machinery: `macro-F1`, `Massive-STEPS`, `mahalle`, `census tract`,
`fold`, `Check2HGI`, `PCGrad`). **Three are real** and are S3-05/S3-06/S3-09. No term is defined
twice in conflicting words; the closest case is `census tract`, glossed compatibly at
`1_introduction.tex:55`, `2_fundamentals.tex:54`, `5_mobiwac.tex:256` and `:334`: four glosses of
one term, none contradicting, which is didactic repetition rather than a double definition.

---

# 3 · Ranked findings

### S3-01 · **BLOCKER** · Four torn sentences in the Resumo and Abstract (rendered pp. 3 and 4)

`0_main.tex:208-209` and `:237-238` (Resumo, p. 3); `0_main.tex:300-301` and `:326-327`
(Abstract, p. 4). In each case a sentence ends, its successor's opening clause is **absent from
the PDF**, and the remainder renders starting with a lowercase word.

As rendered on p. 4:

> "…so one model could learn them together through multi-task learning (MTL). **sharing parameters
> between tasks can hurt one of them**, a failure mode known as negative transfer…"

> "…within a two-point Acc@10 margin (TOST), at the other two. **condition is the finding:** whether
> multi-task learning helps point-of-interest prediction depends on the input representation…"

And on p. 3, the same two defects in Portuguese:

> "…por meio de aprendizado multitarefa (MTL). **entre tarefas pode prejudicar uma delas**, um modo
> de falha conhecido como transferência negativa…"

> "…de Acc@10 (TOST), nos outros dois. **condicional, e a condição é o achado:** se o aprendizado
> multitarefa ajuda…"

The intact source text exists elsewhere in the document, which is how the missing words can be
identified without guessing: `1_introduction.tex:83` reads "**Sharing parameters** between tasks can
hurt one of them, a failure mode known as negative transfer." (renders correctly, p. 13), and
`6_conclusion.tex:83` reads "The dissertation's answer **is conditional, and the** condition is the
finding." (renders correctly, p. 76). So the Abstract is missing a sentence-initial capital
"Sharing" and roughly "The dissertation's answer is conditional, and the"; the Resumo is missing
roughly "Compartilhar parâmetros" and "A resposta desta dissertação é".

*Why the repository's own detector does not see it.* `src_utils/check_trapped_prose.py` looks for
prose trapped **after the last `%` on a comment line**. These four are a different shape: the text
is not trapped in a comment at all, it is simply **absent**, and the line above is ordinary body
text. I re-ran the repository lint's logic and confirm it reports 0 for these; the defect class the
docstring enumerates (nine historical instances) does not cover it. A detector that would catch it:
*a body line beginning with a lowercase word whose preceding non-blank body line ends in a sentence
terminator*. Applied document-wide, that rule returns **exactly these four lines and nothing else**
(zero false positives across 12 chapter files and 16 table files). I offer it as a proposed
addition to `check.sh`, not as an edit.

*Independently confirmed.* After completing this measurement I found that persona 17's report in
this same round (`_review_v3/17_resumo_abstract_assessment.md`, §2, written 2026-07-27 19:39)
reaches the same conclusion from a different direction: it read the two blocks as rendered pages and
identifies the same four sites, the same two donor sentences, and files them as two BLOCKERs. Two
gates converging on one defect from a font-span reading and from a source-level sweep is stronger
evidence than either alone, and it also means the finding needs one fix, not two. Where the two
reports differ in framing: persona 17 counts two BLOCKERs (one per language pair of sentences), this
one counts four torn sentences. The underlying defect set is identical. Persona 17 additionally
measures a claim-parity consequence I did not: the English word "conditional" is now absent from the
Abstract entirely, so the dissertation's central hedge survives only in the Portuguese.

*Severity.* The abstract is the first prose a banca member and the AcademicoPG reviewer read, and
Appendix C asserts the author read and approved every word. Two of the four defects also break the
mandated Resumo↔Abstract claim parity in the same place, so the pair is symmetric in its damage.

*Direction:* author restores the four clauses from the two intact sites named above, then re-reads
the pair side by side, since `0_main.tex`'s six `[NEEDS SIGN-OFF]` markers are already an
audit-as-a-pair item (PENDENCIAS §3.1).

---

### S3-02 · **MAJOR** · Result-"cell" in our own prose, twice (paper glossary §3, verdict *never*)

`articles/[mobiwac]/GLOSSARY.md` §3: *cell (a table result) → "result / entry"; **never in prose**;
this audience reads "cell" as a radio cell.* Chapter 5's own §5.1 makes the collision explicit ("A
census tract is a neighborhood, not a radio cell", p. 62), which is precisely why the word must not
then be reused for a table entry eight pages later.

> `5_mobiwac.tex:588` (p. 71): "…which it matched to within $0.3$, and not to the joint **cells**
> reported here."

> `5_mobiwac.tex:689` (p. 72): "The ReHDM reference runs under its own published protocol, so its
> **cell** is not measured on our windows or folds at all."

Both are **our own claims** about our own table, which is the distinction the brief asked me to
draw. For contrast, the three "grid cell" uses at `:144`, `:145`, `:155` are correct: each writes
the grid sense in full, exactly as the ruling requires, and two of them describe **other** work's
target ("a grid cell as the target", "substitutes official neighborhood-scale units for grid
cells"). The two radio-cell uses at `:264`/`:266` are the sanctioned disambiguation.

*Direction:* "the joint entries reported here"; "so its result is not measured on our windows or
folds at all". Both are one-word substitutions with no claim content.

---

### S3-03 · **MAJOR** · Chapter 4 italicizes ordinary English 155 times, inconsistently

`\textit`/`\emph` inventory by chapter: Ch.1 6, Ch.2 6, Ch.3 23, **Ch.4 155**, Ch.5 10, Ch.6 0,
Apx B 12. Chapter 4's set is not terminology-at-first-use; it is ordinary running vocabulary:
`embedding` ×18, `baseline` ×16, `encoders` ×15, `encoder` ×14, `embeddings` ×12, `check-ins` ×8,
plus the seven category names, `timestamp`, `head`, `framework`, `benchmark`, `pipeline`.

It is also internally inconsistent. The same word appears **both** ways in the same chapter:
`encoder` italic 14 / plain 9, `encoders` italic 15 / plain 7, `baseline` italic 16 / plain 4,
`embedding` italic 18 / plain 2, `framework` italic 1 / plain 1. The clearest case is **one
paragraph on rendered p. 43** (`4_courb.tex:40`), which contains both forms 336 characters apart:

> "…with average gains per state of 20.2 to 22.0 percentage points, considering the better of the two
> spatial **encoders** in each combination. In Next-POI Prediction, the proposed model outperforms
> the *baseline* in most scenarios, with emphasis on categories such as *Food*. The comparison
> between SIREN and Sphere2Vec-M also shows that, although the modular approach is consistently
> superior to the *baseline*, the most suitable spatial *encoder* depends on the geographic
> distribution of the POIs in each territory."

Roman "encoders" and italic "*encoder*" in adjacent sentences of one paragraph. A reader looking for
the reason a word is italicized will not find one, which is exactly the effect the advisor
described.

This is inherited from the published Portuguese paper, where italicizing English loanwords in a PT
text is standard practice and correct. In an **English** chapter the same markup no longer marks a
loanword; it reads as emphasis on a word that carries none, and it is a plausible referent for
"the way some terms are inserted sounds a little strange". Note the v2 translation-fidelity report
(`08_translation_fidelity_report.md:443`) recorded the italics as *preserved deliberately* for
`embedding`, so this is a decision to revisit with a reason, not an unnoticed slip.

*Direction:* author's ruling, and it is one decision, not 155: either (a) drop italics on all words
that are ordinary English in an English chapter and keep them only for the seven category labels and
first-use expansions such as *Spatial-Temporal MTLNet*, logging the departure in Appendix B beside
the existing MTLnet-spelling row; or (b) keep them as published and add one sentence to the
chapter's preface stating that the typography follows the Portuguese original. Option (a) also
resolves the inconsistency; option (b) does not.

---

### S3-04 · **MAJOR** · Gloss pile-ups: two sentences carry three definitions each

Six Chapter 5 sentences carry two or more parenthetical glosses. Two are severe.

**(a) `5_mobiwac.tex:409` (rendered p. 68), three glosses in one sentence:**

> "…the per-visit vectors' silhouette score by category **(how tight and well separated the seven
> labeled groups are, on a $-1$ to $1$ scale)** is about $0.57$ against about $0.00$ for the place
> embedding, and their nearest-neighbor category purity **(the share of nearest neighbors with the
> vector's own category)** is about $0.98$ against about $0.78$ **(both averaged over the five U.S.
> states)**."

Three parentheses, two of them definitions of terms used only here (`silhouette` ×2, `category
purity` ×1 document-wide), one a scope note, and four numbers, in one sentence. The paragraph it
opens then runs to `:413` and also absorbs the `epoch` gloss (row 24 above) and the CTLE
comparison. This is the single densest arrival site in the document.

**(b) `5_mobiwac.tex:206-211` (rendered p. 61), a 40-word parenthesis inside a claim:**

> "…the cosine similarity between the next-category and next-region updates on the shared trunk
> averages $+0.001$ across training **(four seeds each on four Gowalla states: Alabama, Arizona and
> Florida, which are three of the five United States datasets reported here, and Georgia, which
> this dissertation does not otherwise use, per-dataset means within $\pm0.003$)**."

The parenthesis carries a dataset list, a cross-reference, a disclosure about a dataset used
nowhere else, and a second statistic. Every element is honest and load-bearing, which is why this is
a MAJOR and not a cut: the fix is to promote the disclosure to its own sentence, not to delete it.

The other four (`:24-30`, `:289`, `:778`, `tables/mobiwac/results.tex:8`) each carry two, and each
reads acceptably.

*Direction:* split (a) into two sentences at "and their nearest-neighbor category purity"; in (b),
close the parenthesis after "four Gowalla states" and make the Georgia disclosure its own sentence.
No number and no claim moves in either.

---

### S3-05 · **MAJOR** · `seed` is used four chapters before it is defined, and "fitted models" is still not in the registry

`seed` is defined at `5_mobiwac.tex:388` (p. 66): "A *seed* is one complete repetition of the
five-fold experiment, over the same folds, with a different random initialization." Correct, and
exactly the registry's wording. But **first use is `1_introduction.tex:243` (p. 16)**, fifty pages
earlier: "twenty fitted models per configuration (four **seeds** over one fixed set of five
folds), paired significance tests on the four per-**seed** means".

The registry (GLOSSARY §2, `seed` row) bans bare "seed" in the abstract and requires "random
initialization" there. The **Abstract and Resumo comply** (`0_main.tex:320` "four random
initializations"; `:230` "quatro inicializações aleatórias"). Chapter 1's contributions list and
Chapter 6 §6.1 (`:72-73`) do not: both use bare "seeds" and "per-seed means" with no gloss anywhere
before p. 66. The rule as written covers the abstract; the *spirit* of "define once at first use"
is what is broken, and Chapter 1 is a frame chapter where WRITING_LAW is in full force.

Compounding it, **"fitted models"**, the countable unit that carries the n=20 arithmetic, is
used four times (`0_main.tex:320`, `1_introduction.tex:243`, `6_conclusion.tex:72`, `:135`),
including in both the Abstract and the Resumo (as "modelos ajustados"), and is **still absent from
GLOSSARY §4 and §6**. v2 flagged this and recommended the author approve the entry; PENDENCIAS
§3.1 records the question as open. Under the fail-closed maintenance rule the term entered text
before entering the registry, twice over, in the two paragraphs that are hardest to change late.

*Direction:* one clause at `1_introduction.tex:243` ("four random initializations, each a complete
repetition of the five-fold experiment") removes the forward dependency without touching Ch.5's
definition; and the GLOSSARY §4/§6 entries for "fitted models" / "modelos ajustados" need the
author's approval before the defense build.

---

### S3-06 · MINOR · "ceiling" arrives in Chapter 2 in three different senses before Appendix D disambiguates it

Appendix D exists to remove exactly one confusion: two quantities "have both been called the
ceiling" (`apx_d_ceiling.tex:30-31`), and the appendix names each and forbids "ceiling" for the
label-history benchmark. GLOSSARY keeps "ceiling" correct for one quantity only, the dedicated
single-task model's score.

Document order of the word:

| Location | Page | Sense | Verdict |
|---|---:|---|---|
| `2_fundamentals.tex:454` | 23 | Song et al.'s 93% predictability, **explicitly denied** as a ceiling | correct, and doing useful work |
| `2_fundamentals.tex:456` | 23 | "The **operative ceiling** for the two tasks studied here is the dedicated single-task model" | **the licensed sense** |
| `2_fundamentals.tex:582` | 24 | "the floors and the single-task ceiling" | licensed |
| `5_mobiwac.tex:470` | 69 | glossed in full | licensed, gloss present |
| `6_conclusion.tex:237` | 78 | "**the seven-class ceiling**" | **fourth sense**: the taxonomy's granularity |
| `apx_d_ceiling.tex:31` | 98 | the retired sense, named in order to forbid it | correct |
| `apx_d_ceiling.tex:100` | 100 | "Calling it a ceiling … would assert more than the measurement supports" | correct |

Three senses are handled, one is not: `6_conclusion.tex:237` "finer-grained taxonomies would test
them below the seven-class ceiling" uses the word for a label-space granularity, which is neither
the dedicated-model score nor anything the registry sanctions, in a Future Work sentence a banca
member reads late and fast.

Also note the appendix's own closing paragraph (`:114-118`, p. 100) discloses that the released
filenames keep the retired word (`autocorrelation_ceiling.py`). That is the honest thing to do and
should stay.

*Direction:* `6_conclusion.tex:237` → "below the seven-class taxonomy" or "at finer category
granularity". One phrase, no claim content.

---

### S3-07 · MINOR · `sits` / `lies`: one residual, in Chapter 5

Paper glossary §8: *"the interval sits above / below" → **lies** above / below (one verb
everywhere; the draft cycles sits/lies)*. The document is now at **7 `lies` / 1 `sits`**:

> `5_mobiwac.tex:346` (p. 65): "…the residual variant that the encoder we ship descends from
> **sits** at the same level, $0.4197$ and $0.4182$…"

The same sentence also contains the one remaining `ship` metaphor in Chapter 5 prose ("the encoder
we **ship**"), which paper glossary §8 lists explicitly ("ships with the released code" → "is
included in the released code"). Both are in the four-grounds paragraph, the longest in the
chapter. A second `shipped` appears at `apx_d_ceiling.tex:104` ("under the shipped representation").

*Direction:* "lies at the same level"; "the encoder we release descends from" / "the released
encoder". Note the four-grounds paragraph is on the do-not-touch list below; these are two-word
substitutions inside it, not a restructuring.

---

### S3-08 · MINOR · "by a wide margin" survives in reproduced CoUrb prose

`4_courb.tex:325` (p. 54): "…while in California the same model reaches 51.28 $\pm$ 0.57,
outperforming DGI **by a wide margin**." Paper glossary §4 names this exact phrase for renaming
(margin is reserved for the TOST margin; a representation difference is a **gap**), and Chapter 5
follows the rule correctly at `:404` ("by a wide **gap** on every dataset").

Two reasons this is MINOR and not MAJOR: it is reproduced published text under the errata policy,
and the paper glossary binds Chapter 5, not Chapter 4. It is reported because the *inconsistency*
is visible across two chapters of one document, and because it stacks an intensifier onto a number
that already carries its own spread.

*Direction:* "outperforming DGI by a wide gap", or let the numbers carry it and drop the phrase.
Either way an Appendix B wording row, alongside the three already there.

---

### S3-09 · MINOR · Three terms arrive with no gloss anywhere: `user-disjoint`, `joint-best`, `cross-attention`

Not undefined *terms*, since each is explained by its surrounding machinery eventually, but each
**arrives** in a position where the reader meets it cold:

- **`user-disjoint`** first appears at `1_introduction.tex:159` (p. 14) inside an objective:
  "a leakage-guarded statistical protocol, the user-disjoint cross-validation with paired
  significance and non-inferiority testing of Chapter 5". The property that makes it matter
  ("a test user's visits never appear in training") is not stated until `5_mobiwac.tex:344`,
  p. 64. Eight uses, no gloss before p. 64. GLOSSARY §3 has a one-clause gloss ready to reuse.
- **`joint-best`** appears twice only, both in the front matter (`0_main.tex:235` Resumo,
  `:324` Abstract), italicized in the Portuguese, and **carries the interval** "5.3 to 9.4 macro-F1
  points under a joint-best selection". Its definition is in Chapter 5 §5.4 (`:473-474`) under
  different words ("the validation-selected epoch"), and the exact hyphenated term never reappears.
  A reader of the abstract alone meets a selection convention they cannot resolve, in the sentence
  that carries the headline number.
- **`cross-attention`** first appears at `0_main.tex:316` (Abstract, p. 4) and
  `1_introduction.tex:125` (p. 13); the gloss lands at `5_mobiwac.tex:53` (p. 58). WRITING_LAW §1
  explicitly relaxes ML vocabulary for the frame chapters *once defined*, so this is the mildest of
  the three, but Ch.2 §2.3's one mention (`:304`) would be the natural place and passes it by.

*Direction:* author's call, and probably one clause each. `joint-best` is the one that matters:
either gloss it in four words in the abstract ("read at the one validation-selected model per
fold") or use the plain phrase there and keep the hyphenated term for Chapter 5.

---

### S3-10 · NIT · Chapter 2's nine-balancers-in-twenty-one-lines paragraph

`2_fundamentals.tex:315-335` (rendered pp. 21-22) introduces uncertainty weighting, GradNorm,
dynamic weight averaging, PCGrad, CAGrad, Nash-MTL, Aligned-MTL, FAMO, and random loss weighting.
Doc-wide counts for the nine: 1, 2, 2, 4, 1, 23, 1, 1, 1. **Five of the nine appear exactly
once in the whole document** and only Nash-MTL is a name the dissertation actually uses. Four
technical nouns inside the same span also appear exactly once (`homoscedastic`, `condition number`,
`principal components`, `scalarization`). Each clause is correct, cited, and describes the method as
its own authors do (I checked this against the paper glossary §9.3 rule, and CAGrad's description
was visibly corrected in this round per the source ledger at `:344-347`).

This is what a fundamentals chapter is for and the persona's remit does not include thinning a
survey. It is recorded because it is the highest-density arrival site outside Chapter 5, and because
if the advisor's marked passages are in Chapter 2, this paragraph and the §2.2 encoder paragraph
(`:197-206`: Time2Vec, SIREN, Space2Vec, Sphere2Vec in ten lines) are where the measurement points.

*Direction:* none required. If the author wants one change here, the highest-value one is a
half-sentence at `:315` telling the reader these are being named as a family rather than used
("the family below is named for orientation; Chapters 3 and 4 use only Nash-MTL"), which converts
eight one-shot names from insertions into a catalog the reader knows not to memorize. This is also
the cheapest defense at the defense: a banca member who asks "why is CAGrad here?" gets answered by
that half-sentence rather than by the candidate.

---

### S3-11 · NIT · Five parallel-list semicolon braids in Chapter 2, unchanged since v2

`2_fundamentals.tex:12` (4 semicolons, the chapter roadmap), `:74`, `:143`, `:331`, `:587`. Same
five sentences v2 reported; the author has not ruled. Chapter 5's four braids are all statistical
notation and exempt. Ch.3's two are reproduced survey lists. The v2 assessment stands: these are
legitimate parallel-clause lists that cross a mechanical threshold, and the law needs the exemption
or two of the five need splitting. Re-proposed below.

### S3-12 · NIT · Metaphor budget: `carry/carries` over in three places; two live metaphors in Ch.2

Budget ≤3 per chapter. Counts: Ch.2 **7**, Apx B **6**, Apx D **5**, Apx E **5**, Ch.4 5, Ch.5 3.
As in v2, nearly all are the literal sense ("every number carries its reference point", "a vector
that stays the same across visits carries nothing about the visit"). Over budget on tokens,
compliant on the offense.

Two metaphors in Chapter 2 do read as inserted: **"the hinge of the representation argument"**
(`:86`, p. 18) and **"Here the representation is the lever"** (`:564`, p. 24, with a second lever at
`:594`, p. 25). Both are the author's own structural signposting rather than AI decoration. "Hinge"
promises a return that §2.2 delivers at `:186-195`, and "lever" is picked up by Ch.6. Neither is on
any ban list. Recorded for the author's ear, not as a violation. `pays off` at `:563` is the one I
would change ("That stance only pays off if the model can represent a visit well enough to serve
both"), because it is a money metaphor in the §8 family, and "only holds if" does the same work.

---

# 4 · The counted report

### Hard bans

| Check | Count | Status |
|---|---:|---|
| Em-dash in prose (`---` or U+2014) | **0** | **PASS**. Four `---` exist, all inside front-matter *placeholders* (`0_main.tex:144,145,146,186`: banca-member and defense-date fields awaiting the advisor). Only one renders (p. 2, the approval-sheet placeholder) and it is not prose. |
| Contractions in prose | **0** | **PASS** |
| Repo codenames (B9, v11–v17, champion-G, H3-alt, dk_ovl, log_T, substrate, engine, board, recipe, frozen-for-folds) | **0** | **PASS**, each checked individually. "region-transition prior" stands where `log_T` would (`5_mobiwac.tex:346`); "fixed at its initial values" where "frozen" would (`:579`). The four `engine*` hits are `engineering`/`embedding-engine suite` in Apx A's software description, not the banned sense. |
| `mtlnet_crossattn_dualtower` | **0** | **PASS** |
| `arm(s)` | **0** | **PASS** (author's fix, verified) |
| `substrate` / `recipe` / `checkpoint` | **0 / 0 / 0** | **PASS** |
| `beats` / `wins` / `ties` / `Pareto` as verdicts | **0** | **PASS**. `ties` ×1 is the verb "ties this structure to" (`5_mobiwac.tex:259`); the two `wins` are Appendix B **quoting** the published CoUrb wording it corrects; `Pareto` ×3 all name MGDA's own property. |
| `SOTA` / `SOAT` | **0** | **PASS** |
| Literal "Read this as:" | **0** | **PASS** |
| **Torn sentences (new check)** | **4** | **FAIL** (S3-01) |

### Banned words and templates

**Frame chapters (1, 2, 6) and the new Appendices D and E: ZERO hits.** 71 banned entries swept
case-insensitively over prose including captions and table bodies; 53 return zero document-wide.
Everything found lies in reproduced paper text or in Appendix B's quotations of the wording being
corrected:

| Word | n | Where | Disposition |
|---|---:|---|---|
| `robust` | 5 | all `3_cbic.tex` | reproduced CBIC; load-bearing ("robust feature extractors", "a robust evaluation"). WRITING_LAW §4.6 declines to over-ban. |
| `leverag*` | 4 | all `tables/cbic/errata_wording.tex` | **exempt.** The errata table quotes the published phrases it replaces |
| `crucial` | 3 | `3_cbic.tex:212,303,322` | reproduced CBIC |
| `surpass*` | 3 | `3_cbic.tex:317` + 2 errata quotations | reproduced / exempt |
| `enhance*` | 3 | `3_cbic.tex:95,308`; `5_mobiwac.tex:52` | reproduced CBIC ×2; the third is the **C2-mandated** lead-in "We propose two enhancements" (paper glossary §9.1 requires this exact wording) |
| `highlight` | 2 | `3_cbic.tex:68`, `4_courb.tex:347` | reproduced paper prose |
| `landscape` | 2 | `3_cbic.tex:317,322` | reproduced CBIC ("a competitive performance landscape") |
| `Moreover` / `Furthermore` / `underscore` | 2 / 2 / 1 | all `errata_wording.tex` | **exempt.** Quotations of corrected wording |
| `Additionally` | 2 | `3_cbic.tex:166`; `apx_e_ethics.tex:46` | reproduced CBIC; the Apx E one is mid-sentence ("is additionally access-gated"), not the banned sentence-initial connective |
| `notably` / `crucially` / `comprehensive` / `serves as` / "it is important to notice" | 1 each | all `3_cbic.tex` | reproduced CBIC |
| `genuine` | 1 | `5_mobiwac.tex:346` | **load-bearing**: "the genuine category history of the input window" distinguishes real labels from encoder-derived ones. This is the GLOSSARY's own definition wording for the label-history benchmark. Keep. |
| `navigate` | 1 | `1_introduction.tex:41` | "a navigation or transit service", the literal noun, not the metaphor |

Zero hits for: delve, intricate, nuanced, showcase, boasts, pivotal, vital, meticulous, thoughtful,
judicious, realm, tapestry, interplay, harness, unlock, foster, garner, embark, seamless, holistic,
innovative, groundbreaking, unprecedented, remarkable, testament, stands as, "in conclusion",
valuable insights, advancements, emphasize, "align with", decorative "key X", bolstered, vibrant,
enduring, commendable, exceptional, invaluable, noteworthy, adept, versatile, paradigm shift,
myriad, cutting-edge, game-changer, deep dive, multifaceted, "not only X but also Y",
"plays a crucial role", "in today's world", Firstly/Secondly/Thirdly, "a wide array of".

### Density metrics

| chapter | words | -ly | **-ly %** | intensifiers | semicolons | "X, not Y" | "rather than" | carry* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Front matter | 1,115 | 3 | 0.27 | 1 | 7 | 1 | 0 | 0 |
| Ch.1 Introduction | 1,863 | 10 | 0.54 | 1 | 6 | 1 | 2 | 1 |
| Ch.2 Fundamentals | 4,190 | 20 | 0.48 | 3 | 23 | 2 | 16 | 7 |
| Ch.3 CBIC | 5,115 | 75 | **1.47** | 14 | 18 | 1 | 5 | 0 |
| Ch.4 CoUrb | 4,891 | 51 | **1.04** | 3 | 6 | 2 | 2 | 5 |
| Ch.5 MobiWac | 7,814 | 33 | 0.42 | 5 | 78 | **23** | 17 | 3 |
| Ch.6 Conclusion | 1,643 | 8 | 0.49 | 1 | 8 | 1 | 5 | 0 |
| Apx A | 325 | 2 | 0.62 | 0 | 1 | 0 | 3 | 0 |
| Apx B Errata | 3,417 | 22 | 0.64 | 4 | 30 | 7 | 15 | 6 |
| Apx C AI | 370 | 0 | 0.00 | 0 | 5 | 0 | 1 | 0 |
| **Apx D Benchmark** (new) | 1,162 | 4 | 0.34 | 1 | 2 | 1 | 1 | 5 |
| **Apx E Ethics** (new) | 857 | 4 | 0.47 | 0 | 4 | 2 | 0 | 5 |
| **Document** | **32,762** | **232** | **0.71** | **33** | **188** | **41** | **67** | **32** |

Band is ≈0.8% max. Every frame chapter and both new appendices are inside it; only the two
reproduced paper chapters exceed it, for the reason v2 gave (the errata policy targets
claim-strength words, not register). Two -ly adverbs in one sentence: **0** in frame prose.

Chapter 5's 23 "X, not Y" is the highest count in the document and is the one density figure worth
the author's eye: the paper glossary §7 audited 21 at submission and names five as
ledger-mandated keeps. I verified all five survive verbatim ("a match, not a gain"; "a defense of
the parallel design, not a claim that we outperform the cascade"; "a region-native model, not a
reproduction"; "a neighborhood, not a radio cell"; "motivation, not a measured service result").
The count rose by two, not fell, so the honesty device is intact and no decorative instance was
added in bulk.

### Variance and burstiness (WRITING_LAW §4.3)

| chapter | sentences | mean | sd | **CV** | min | max |
|---|---:|---:|---:|---:|---:|---:|
| Front matter | 28 | 39.8 | 26.2 | **0.659** | 3 | 110 |
| Ch.1 Introduction | 65 | 28.7 | 13.7 | 0.477 | 6 | 65 |
| Ch.2 Fundamentals | 158 | 25.8 | 14.5 | 0.560 | 4 | 80 |
| Ch.3 CBIC | 219 | 22.3 | 9.4 | 0.421 | 3 | 57 |
| Ch.4 CoUrb | 177 | 26.3 | 12.5 | 0.474 | 4 | 70 |
| Ch.5 MobiWac | 293 | 26.4 | 13.5 | 0.513 | 4 | 76 |
| **Ch.6 Conclusion** | 56 | 29.2 | 19.5 | **0.667** | 3 | 111 |
| Apx A | 11 | 29.5 | 12.7 | 0.429 | 5 | 51 |
| Apx B Errata | 69 | 26.8 | 13.3 | 0.497 | 5 | 55 |
| Apx C AI | 13 | 28.5 | 16.6 | 0.585 | 10 | 78 |
| **Apx D Benchmark** (new) | 59 | 19.3 | 11.0 | **0.567** | 4 | 51 |
| **Apx E Ethics** (new) | 37 | 23.2 | 11.5 | 0.497 | 4 | 51 |

**No variance compression.** The two most heavily edited units, Chapter 6 and the front matter, are
the burstiest in the document (0.667, 0.659). This is the same finding v2 reported, and it survives the
corpus change. The new Appendix D, written entirely this round, comes in at CV 0.567 with a mean
sentence length of 19.3 words, the shortest in the document: it is the *least* smoothed new prose,
not the most. Chapter 6 runs a 3-word sentence against a 111-word one. Read-aloud spot check on
`apx_d_ceiling.tex:94-101`: "The gap is not by itself evidence of a leak." (9 words) against a
34-word sentence, then "The benchmark is also not an upper bound." (8 words). Not one weight.

### Discourse-skeleton variety (§4.4)

**Chapter/appendix openers, all twelve, no template:** Ch.1 opens on a definition of LBSNs; Ch.2 on
a purpose statement; Ch.3 on a paradigm definition (reproduced); Ch.4 on the dataset (reproduced);
Ch.5 on the problem; Ch.6 on the research question; Apx A on scope; Apx B on what the chapters
reproduce; Apx C on the disclosure; Apx D on a question; Apx E on the data. Ten distinct moves.

One repeated move worth naming: **"This X answers one question"** appears three times
(`1_introduction.tex:90`, `6_conclusion.tex:14`, `apx_d_ceiling.tex:16`). The first two are a
deliberate frame bracket (open the question, close it) and read as structure. The third, in the new
appendix, makes it a tic rather than a bracket. Similarly Appendix D opens three of its eleven
paragraphs with a bare numeral phrase: "Two quantities are involved…", "Two readings follow…",
"Two coverage limits apply." Recorded as **NIT**; one of the three would vary usefully.

**Section-ending check:** no section closes by restating itself. `2_fundamentals.tex:602-603` ends
on a forward hinge; `apx_d_ceiling.tex:114-118` ends on a provenance note; `apx_e_ethics.tex:90-98`
ends on a comparable precedent. Zero "In summary" / "To summarize" / "In conclusion" openers in
frame prose.

**Sentence openers:** no two-word opener exceeds 3 occurrences in any chapter, and no chapter opens
more than 2% of its sentences the same way.

### Term-registry lint (L2)

| Concept | Name used | Alternatives found |
|---|---|---|
| the "what" task | next category / next-category prediction | none |
| the "where" task | next region / next-region prediction | "area" **0** as a task name |
| the exact-place task | next place | kept distinct; "We do not predict the exact next place" once, early (`5_mobiwac.tex:256`) |
| one visit | check-in | "event" **0** (the single `event` hit is `1_introduction.tex:209`, "presented the paper at the event") |
| a place | place / POI | "venue" **0** in the task sense (6 hits, all publication-venue metadata in Apx B and the organization section) |
| our representation | check-in-level representation (Check2HGI) | "substrate" **0** |
| place-level baseline | place embedding (HGI) | bare "the baseline" not used as the referent in the frame; Ch.4 uses italic *baseline* as its own referent (S3-03) |
| one model, both tasks | the joint model | bare "MTLnet" before introduction: **none** |
| one task, one model | dedicated single-task model | bare "baseline" alone: none in the frame |
| the shared middle | the shared trunk | "exchange stack" **0**, "backbone" **0** |
| repetition unit | seed | "run"/"multi-seed run"/"single-seed"/"seed by seed" **0**; but arrival order fails, S3-05 |
| model-name spellings | MTLnet (66), ST-MTLNet (19) | `MTLNet` appears **only** where the registry sanctions it: inside "ST-MTLNet", as the published expansion *Spatial-Temporal MTLNet*, and in Apx B's row documenting the spelling departure. **No stray `MTLNet`, `MTL-Net`, or `MtlNet`.** |

**Two registry gaps, both fail-closed:** `fitted models` (S3-05) and, newly, the Ch.4 chapter title
uses *Spatio-Temporal* (`4_courb.tex:14`, chapter opening p. 42, running head pp. 43-56, TOC p. 9)
while the in-text expansion
at `:38` and `:248` uses *Spatial-Temporal*, which is the form GLOSSARY §2 registers as published.
Document-wide: `spatio-temporal` 6, `spatial-temporal` 3. The two in `:38`/`:248` are the registered
name and are correct; the **title** is the odd one out. One-line NIT, worth a look because it is the
chapter title and the running head.

### Structure / presentation spot-checks (§5)

| Rule | Result |
|---|---|
| Every results table has a lead takeaway sentence | **yes**, verified on all four Ch.5 tables, both Ch.3 result tables, both Ch.4 result tables, and Apx D's benchmark table |
| Literal "Read this as:" | **0** |
| Captions above tables / below figures | **yes** on every float, verified on rendered pages |
| Metrics defined defensively at first use | **yes.** macro-F1 with plain reading and floor (`5_mobiwac.tex:383`); Acc@10 with boundary behavior (`:385`); Apx D adds the scale convention ("macro-F1 on a zero-to-one scale… one point is 0.01") at `:27`, which is a genuine improvement over v2's build |
| Hygiene sentences at leakage-sensitive steps | **present** at splitting (`:344`), representation training (`:346`), region-transition prior (`:346`), baselines (`:394`). Content is persona 07/09's gate |
| Section purpose statements, varying shape | **yes** |

---

# 5 · Proposed law updates (author approval; never applied)

1. **NEW: add a torn-sentence check to `check.sh`.** The rule that catches S3-01 with zero false
   positives on this document: *flag any body line whose first word is lowercase when the preceding
   non-blank body line ends in `.`, `!`, or `?`*. This is a different defect class from
   `check_trapped_prose.py`'s (no comment is involved; the words are absent, not hidden), so it
   belongs as a second check rather than a change to that detector's threshold. Its docstring's own
   warning applies: do not tune it on the long cases.
2. **Re-proposed from v2, still open: parallel-list semicolons.** WRITING_LAW §4's braid rule reads
   as absolute with only CI notation exempt; five legitimate parallel-clause lists in Ch.2 cross it.
   Propose: *a three-or-more-item parallel list whose items are clauses, closing with "and", is
   exempt, provided the items are genuinely parallel.*
3. **Re-proposed from v2, still open and now more urgent: "fitted models" into GLOSSARY.** §4
   carries the "n = 20 (fitted models)" row; the countable noun phrase needs a §2/§3 entry and a §6
   PT equivalent ("modelos ajustados"), since it is now in both the Abstract and the Resumo.
4. **NEW: a §2 note on italics in re-typeset chapters.** WRITING_LAW §6 says the MobiWac chapter
   must not be re-technicalized, but nothing governs *typography* inherited from a Portuguese
   source. Propose one line: *in an English chapter translated from Portuguese, italics are kept
   only for terms at first use and for label values, not for English loanwords that were
   italicized because the source was Portuguese.* This is what S3-03 needs to be decidable rather
   than a matter of taste.
5. **No new AI-tells found in the wild.** I looked for 2026-vintage tells beyond the law's tables
   (nominal-style creep, uniform impersonal register, shrinking vocabulary, over-hedging stacks)
   across the two new appendices specifically, since they are the newest AI-assisted prose. Nothing
   worth adding. Appendix D's register is unusually concrete and its sentences are the shortest in
   the document.

---

# 6 · What is legal and load-bearing (do not push toward sterility)

A later pass must not "fix" any of these:

- **The four-grounds paragraph** (`5_mobiwac.tex:346`). It is one dense paragraph because the four
  grounds are one argument, and its length is the honesty. S3-07 asks for two two-word
  substitutions inside it, nothing more.
- **`genuine` at `:346`.** On the ban list, and here it is the registry's own defining word for the
  label-history benchmark. Deleting it would blur the distinction the appendix exists to protect.
- **"We propose two enhancements"** (`5_mobiwac.tex:52`). `enhance*` is banned; this instance is
  mandated verbatim by the paper glossary's C2 ruling. Do not "improve" it.
- **Chapter 5's 78 semicolons.** The majority are statistical notation
  (`(each entry: point estimate; interval)`). Removing them would make the interval reporting worse.
- **The 23 "X, not Y" constructions in Chapter 5**, and specifically the five ledger-mandated ones.
  All five verified present verbatim.
- **`robust`, `framework`, `baseline`, `comprehensive`** where they appear. Load-bearing CS words;
  WRITING_LAW §4.6 exists for exactly this.
- **Chapter 6's 111-word sentence and its 3-word neighbours**, and the front matter's 110-word span.
  The dispersion is the point.
- **Appendix D's short, flat sentences.** "The gap is not by itself evidence of a leak." "The
  benchmark is also not an upper bound." These do more than a paragraph of hedging would, and they
  are the strongest new prose in the document.
- **Appendix D §D.4's disclosure that the released filenames keep the retired word.** Awkward, and
  correct.
- **The em-dashes in the front-matter placeholders.** They mark fields awaiting the advisor
  conversation. They are not prose and must survive until those fields are filled.
- **`5_mobiwac.tex:264` "A census tract is a neighborhood, not a radio cell."** One of the five
  mandated keeps, and the reason S3-02 matters.

---

# 7 · Out-of-scope handoffs (one line each)

- **Persona 05 (citations):** nothing new. No `(??)` renders remain; the four v2 reported are gone.
- **Persona 06 (numbers):** `6_conclusion.tex:118`'s 56.16 still carries no spread; the source
  comment at `:156-158` records this as knowingly left outside that correction's scope.
- **Persona 07 (claims):** I verified hygiene sentences *exist*; their content is yours. Also:
  `apx_d_ceiling.tex:105` states "196 of the 29,816 places carry more than one category", a number
  in new prose that postdates your last pass.
- **Persona 08 (translation fidelity):** S3-03's italics decision touches your `:443` ruling that
  `embedding`'s italics were preserved deliberately; the two findings should be reconciled by one
  author decision, not two.
- **Persona 13 (UFV compliance):** the four front-matter em-dash placeholders are still unfilled
  (banca members, defense date, approval sheet).
- **Persona 18 (visual):** Ch.4 Figure 2's Portuguese in-figure labels remain (PENDENCIAS §3.2).

---

## Method note (so this audit is reproducible)

Prose was extracted from all 12 chapter files plus 16 table files by stripping LaTeX comments,
replacing math with a token, replacing `\cite`/`\ref` with tokens, and keeping the text arguments of
`\textbf`/`\emph`/`\textit`/`\caption`/`\section`-family commands; tabular, equation, and verbatim
environments were excluded from sentence statistics but table *captions and notes* were included,
which is the main reason the counts differ from v2. Rendered page numbers were resolved by matching
the normalized sentence against the extracted text of the 103-page `dissertacao.pdf`, so every page
number in this report was located in the PDF rather than inferred from the source. Every count is a
regex sweep over that corpus; every quotation was copied from the source file or the rendered page,
not retyped. Where a number could not be located in a committed file it is not asserted.
