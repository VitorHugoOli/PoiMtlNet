Qoute: \begin{abstract}
fabricio.asilva: mais contexto e motivação do que resultados

Qoute: ", and what sharing a single model costs."
fabricio.asilva: não entendi essa parte "what sharing...." o que quer dizer?

vitor: De fato nosso abstract está muito voltado para resultados, precisamos focar mais no contexto e na motivação.

Qoute: \section{Problem and Tasks}
fabricio.asilva: nessa seção tem que descrever o problema (independente se vai usar Gowalla, Istanbul, etc). Não é para
descrever detalhes da solução (isso fica para a próxima seção).

Qoute: We order each user's check-ins in time, form windows of nine consecutive visits, and from each window predict two
properties of the next visit.
fabricio.asilva: essa primeira frase não está falando do problema, mas de parte da solução. na descrição do problema tem
que descrever o que está sendo resolvido, e não como

Qoute: On category it is $+0.29$ at Alabama, $+0.27$ at Arizona, and $+0.00$ at Florida;
fabricio.asilva: nill ?? none??

Qoute: Four of the six datasets are measured at $n{=}20$ on both arms:
fabricio.asilva: heads ??

Qoute: \label{sec:results-external} Read this as: the finding holds on a non-U.S. city under the same representation,
overlapping-window protocol, and training setup as the U.S.
fabricio.asilva: estranho começar a seção com "Read this as"....melhorar isso (tem em outro lugar no texto)

Qoute: One model is enough. .....
fabricio.asilva: ??????, está muito brusco, temos que ter uma abordagem mais convidativa

Qoute: user will do next and where. The code for the model, representation, baselines, and statistical tests is
available at \url{https://github.com/VitorHugoOli/PoiMtlNet/tree/mobiwac}; both data sources are already public: the
category-annotated Gowalla dump (\url{https://figshare.com/articles/dataset/gowalla_data/22126586}) and the
Massive-STEPS collection~\cite{wongso2025massivesteps}.
fabricio.asilva: colocar na introdução, e os links como footnote

---

# Part 2 — Direct text edits by the advisor (NOT applied; documented for audit)

**Provenance.** Advisor's edited copy: `/Users/vitor/Downloads/_MOBIWAC__26_/` (Overleaf download, files dated
2026-07-10). Baseline: repo `articles/[mobiwac]/src/` at commit `5e288e19` (working tree clean on `src/`).
Diffed file-by-file with `diff -u`. **None of these edits have been applied to the repo.** Each item below is
BEFORE (repo) → AFTER (advisor). An auditor should judge each against `GLOSSARY.md`, `PAPER_PLAN.md §3`
(claim discipline), and the decisions ledger in `articles/[mobiwac]/CLAUDE.md §3`, then mark ACCEPT / REJECT / REWORD.

**Files touched by the advisor:** `main.tex`, `sections/01_introduction.tex`, `sections/04_method.tex`,
`sections/05_setup.tex`, `sections/06_results.tex`.
**Files verified identical:** `sections/02_related.tex`, `03_problem.tex`, `07_discussion.tex`,
`08_conclusion.tex`, all of `tables/`, all of `figs/`, `references.bib`.

## main.tex (abstract + keywords)

- **E1 (abstract, wording):** "These are **normally** handled by separate models" → "These are **usually** handled by separate models".
- **E2 (abstract, wording):** "whether one model can learn **both**, and what sharing a single model costs" → "whether one model can learn **both tasks**, and what". (Note: the PDF comment above questions the same sentence's "what sharing... costs" clause; this edit does not resolve that comment.)
- **E3 (abstract, typography):** `$+28$ to $+40$` → `+28 to +40` and `$+5$ to $+9$` → `+5 to +9` (math mode removed from the abstract deltas). AUDIT NOTE: changes minus-sign/spacing rendering; the rest of the paper keeps math mode for signed deltas — consistency question.
- **E4 (abstract, commas):** comma added after introductory phrases: "Across the five U.S. states**,** the gain..." and "On the non-U.S. city**,** the result holds".
- **E5 (keywords, reorder):** "mobility data" moved from last to first: `next-category prediction, next-region prediction, multi-task learning, check-in-level representation, location-based social networks, mobility data` → `mobility data, next-category prediction, next-region prediction, multi-task learning, check-in-level representation, location-based social networks`.

## sections/01_introduction.tex

- **E6 (¶2, wording):** "the useful question is where sharing helps, **where** it costs, and how" → "...where sharing helps, **what** it costs, and how".
- **E7 (¶3, claim strength):** "We introduce two **changes**." → "We introduce two **novelties**." AUDIT NOTE: strengthens the novelty claim; check against the novelty-defusal discipline (CLAUDE.md §3 cascade-framing ruling, PAPER_PLAN claim whitelist).
- **E8 (¶3, tense + scope cut):** "We **evaluate on** two deliberately different settings: five U.S. states of different sizes **(Gowalla)** and one international city (Istanbul)." → "We **evaluated in** two deliberately different settings: five U.S. states of different sizes and one international city (Istanbul)." AUDIT NOTE: (a) drops the "(Gowalla)" dataset attribution from the intro; (b) shifts to past tense — the paper's voice elsewhere is present tense.
- **E9 (contributions lead-in):** "Our contributions are the following." → "Our contributions are summarized as follows:".
- **E10 (contribution 1, cross-ref):** "(Table~\ref{tab:substrate})" → "(see results in Section~\ref{sec:results}, Table~\ref{tab:substrate})". Label `sec:results` exists (06_results.tex:11) — compiles.
- **E11 (contribution 2, comma):** "In one forward pass**,** it outperforms".
- **E12 (contribution 2, cross-ref):** "(Table~\ref{tab:results})" → "(see results in Section~\ref{sec:results}, Table~\ref{tab:results})".
- **E13 (contribution 3, scope wording):** "across five **states** and one international city" → "across five **U.S. states** and one international city".
- **E14 (contribution 3, cross-ref):** "(Fig.~\ref{fig:deltas})" → "(**see in** Section~\ref{sec:results}, Fig.~\ref{fig:deltas})". AUDIT NOTE: "see in Section" is ungrammatical as written; if accepted, reword to "see Section" or match E10/E12's "see results in Section".

## sections/04_method.tex

- **E15 (cost paragraph, cross-ref removed):** "larger than either dedicated single-task model **of Table~\ref{tab:results}**, and a forward pass costs more" → "larger than either dedicated single-task model, and a forward pass costs more". AUDIT NOTE: removes a forward reference from §4 to Table III; check nothing else anchors "dedicated single-task model" to the table at first mention.

## sections/05_setup.tex

- **E16 (Splitting ¶, typography):** "from about 25 **percent** of visits in Florida to 34 **percent** in Alabama" → "from about 25**\%** of visits in Florida to 34**\%** in Alabama". AUDIT NOTE: pure "percent"→"\%" swap (both occurrences). GLOSSARY prefers plain words for this audience; check paper-wide percent-vs-\% consistency before deciding.
- **E17 (Baselines ¶, structure):** the single long baselines paragraph split into **three paragraphs**, one per role ("The first is... / The second is... / The third compares..."). No wording changed in roles 1–2 (trailing whitespace added at split points).
- **E18 (Baselines ¶, cross-ref removed):** final sentence "The full training recipe for every model ships with the released code **(Section~\ref{sec:conclusion})**." → same sentence **without** the Section reference. AUDIT NOTE: related to the advisor's PDF comment above (move the code/data URLs to the introduction with footnotes); if that restructuring is accepted, this pointer needs re-anchoring, not just deletion.

## sections/06_results.tex

- **E19 (§6.1 like-for-like ¶, punctuation restructure):** the standalone parenthetical sentence "(The check-in-level column therefore keeps one fixed recipe, intentionally not the per-dataset-tuned ceiling of Table~\ref{tab:results}.)" is folded into the preceding sentence: "...so the margin isolates the representation (the check-in-level column therefore keeps one fixed recipe, intentionally not the per-dataset-tuned ceiling of Table~\ref{tab:results})."
- **E20 (§6.1, paragraph break):** new paragraph started at "A feature-concat control (the place embedding joined with raw per-visit features, same model)..." (was mid-paragraph after the CTLE sentences).
- **E21 (§6.2, paragraph break):** blank line inserted before "Table~\ref{tab:results} reports the single joint model..." (splits it from the scale-calibration sentences).
- **E22 (§6.2, paragraph break):** blank line inserted before "Figure~\ref{fig:deltas} plots the signed gains..." (after the commented-out block).
- **E23 (§6.2, comma):** "Across the five U.S. states**,** the region gain rises with the number of regions".
- **E24 (§6.2, grammar fix):** "On region, all four are **tested equivalences**" → "all four are **tested for equivalence**". AUDIT NOTE: the repo text is a genuine grammar slip; advisor's fix looks correct.
- **E25 (§6.3, comma):** "On principle**,** we prefer coupling the tasks in parallel".
- **E26 (§6.4, commas):** "On Istanbul**,** the joint model outperforms..." and "on category**,** the joint model is far above the Markov floor". (Note: the advisor's PDF comment above separately objects to this subsection's "Read this as:" opener — the text edit does NOT change that opener; both issues are open.)

## Completeness verification (final pass, 2026-07-10)

- **Recursive tree diff** (`diff -qr`, both directions): only the 5 files above differ; no file exists in the
  advisor's copy that is absent from the repo (the reverse "Only in" hits are repo build artifacts: .aux/.bbl/.blg/.log).
  `README.md`, `.gitignore`, `IEEEtran.cls`, `IEEEtran.bst`, `references.bib`, all `tables/`, all `figs/` — identical.
- **Word-level diff** (`git diff --word-diff`, whitespace-insensitive) on the 5 changed files: every changed token
  maps to exactly one of E1–E26; no additional sub-edit hides inside the long re-wrapped lines.
- **Blank-line audit:** 2 added blank lines in `05_setup.tex` (= the E17 three-paragraph split) and 3 in
  `06_results.tex` (= E20/E21/E22); none elsewhere. The advisor added **no LaTeX `%` comments** in any file.
- **PDF annotations:** the advisor's `main.pdf` carries **zero** annotations (pypdf sweep) — Part 1's quotes come
  from the Overleaf review pane, not the PDF. If more comments exist on Overleaf beyond the seven transcribed in
  Part 1, they live only there; the local artifact contains nothing further to extract.

## Not changed by the advisor (explicitly verified)

- All numbers, tables, figures, and the bibliography are untouched — every edit is prose/typography/structure.
- The "Read this as:" opener (§6.4), the "One model is enough" discussion opener, the Problem-section framing, and the conclusion's URL paragraph — all flagged in the PDF comments above — are **unchanged in the text**: the advisor left those as comments only, expecting us to do the rewrite.
