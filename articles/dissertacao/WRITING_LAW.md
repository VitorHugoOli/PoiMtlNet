# WRITING_LAW.md — the writing law for the dissertation (v1, 2026-07-18)

> **Scope.** Every sentence written in this dissertation — by the author or by an agent — obeys
> this file. It inherits the MobiWac [`GLOSSARY.md`](../%5Bmobiwac%5D/GLOSSARY.md) (the paper's
> writing law, battle-tested through two review cycles) and adapts it to a **dissertation**:
> different audience (a computing banca, not networking reviewers), different length (didactic
> register allowed), same honesty discipline. Where the two files conflict for dissertation
> prose, THIS file wins; for the MobiWac chapter's re-typeset prose, the paper GLOSSARY wins.
> Process rules (how agents verify citations/numbers) are in
> [`AGENT_GUARDRAILS.md`](AGENT_GUARDRAILS.md) — this file is about the words on the page.

---

## 1 · Register: dissertation ≠ paper

- **Audience:** a CS/ML banca + future students. Unlike MobiWac's networking audience, standard
  ML vocabulary (embedding, transformer, cross-attention, macro-F1) is fine **once defined**.
  The MobiWac plain-word substitutions (§3 of the paper GLOSSARY) are therefore relaxed for the
  frame chapters — but every term still gets ONE definition at first use (Fundamentals is where
  most of them live) and is then used consistently.
- **Didactic room:** the dissertation may explain (worked examples, notation tables, "Relevance"
  subsections) where the paper had to compress. Viegas patterns to use: definition → citation →
  concrete example rhythm; sections open with a 1–3 sentence purpose statement; background always
  tied to its downstream use ("this assumption matters for Chapter 5's protocol because…").
- **Register bar (inherited, unchanged):** clear, simple academic English that a Brazilian
  non-native English writer would defend aloud. Prefer common, precise words and direct
  constructions. Do not replace an established technical term with an inaccurate plain word, but
  do explain the term at first use. Both failure poles are banned: AI-inflated vocabulary AND
  native-literary idiom (phrasal-verb metaphors, money/motion metaphors). The test: *would the
  author say it at the defense, would the community write it, and would a qualified reader
  understand it on the first reading?* Safe verbs: use, cost, show, obtain, reach, remain, include,
  provide, predict, measure, train, keep.
- **American English throughout, with ONE deliberate exception.** Use American spelling,
  vocabulary, and usage consistently.
- **Terminal punctuation goes OUTSIDE the closing quotation mark, per ABNT NBR 10520:2023.**
  Author's ruling of 2026-08-02, PENDENCIAS 2.24. American style puts the period inside; this
  document is deposited under ABNT, whose citation standard places the closing punctuation after
  the quotation mark, and the deposit norm wins over the house language convention. The document
  already follows ABNT against American practice elsewhere: table captions go above the table.
  Write `... shared information".` and never `... shared information."`
  This is not a style preference to be tidied by a later pass. The 14 sites that carry it are
  almost all errata tables, where the quoted string IS the evidence for a correction: moving a
  period inside such a quotation alters the quotation, which is a worse fault than a punctuation
  convention a reader may not share. Measured 2026-08-02: 6 in `tables/cbic/errata.tex`, 3 in
  `tables/courb/errata.tex`, 2 each in `tables/cbic/errata_wording.tex` and
  `chapters/3_cbic/method.tex`, 1 in `chapters/apx_b_errata.tex`.
- **Clarity takes priority over variation or elegance:** a reader must not need to
  re-read a sentence or paragraph to recover its intended meaning or logical connection. If a
  passage needs a second reading, revise it by reducing clause load, naming the referent, making
  the logical link explicit, or splitting it. No contractions; "cannot" is correct. Digits for
  data quantities ("8,501 regions"); words for small counts ("two tasks, seven categories").
  Comma after sentence-initial adverbial phrases. Write relative pronouns ("the head **that** we
  do not predict").

- **No British English, and the ban is wider than a spelling list.** The author read Appendix F on
  2026-07-30 and found *"feature needs saying plainly"* in the deposited prose (PENDENCIAS_RESOLVIDOS 2.22 (arquivado 2026-07-30)
  point 8). Read the instance carefully, because it is the reason this rule is shaped the way it is:
  **it is not a misspelling.** Every word is spelled identically in both dialects. It is the British
  `need`+gerund construction, where American English writes "needs to be said plainly". He also
  asked whether the governing documents banned it; they did not. Measured before this rule was
  written: `grep -cin "british"` over this file and `AGENT_GUARDRAILS.md` returned 0 and 0. The
  previous line said "American English throughout" and named no British form, so an agent had
  nothing to check against. Both halves are now named.

  **Spellings.** American in every case: `-ize`/`-ization` not `-ise`/`-isation`, `-yze` not `-yse`,
  `-or` not `-our` (behavior, color, neighbor, labor), `-er` not `-re` (center, meter, fiber),
  single `l` before a vowel suffix (traveled, modeled, labeled, signaled) and double `l` where
  American doubles it (skillful, fulfill, installment), `-se` for the noun (defense, offense,
  license, pretense) and `practice` for both noun and verb, `while` not `whilst`, `among` not
  `amongst`, `toward`/`forward`/`afterward` without the `-s`, `learned`/`spelled`/`burned` not
  `learnt`/`spelt`/`burnt`, `program` not `programme`, `catalog` not `catalogue`, `gray` not `grey`,
  `judgment`/`acknowledgment` without the medial `e`, `aging` not `ageing`, `percent` not `per
  cent`, `skeptic` not `sceptic`, `inquiry` not `enquiry`, `oriented` not `orientated`, `focused`
  not `focussed`, no `oe`/`ae` digraph. **Words that end in `-ise` in both dialects are not
  British** and must not be "corrected": surprise, comprise, exercise, advertise, supervise, revise,
  devise, improvise, compromise, franchise, arise, precise, premise, and their prefixed and
  inflected forms.

  **Constructions**, which no spelling list reaches: `need`/`want` + gerund ("needs saying",
  "wants doing"), `different to` or `different than` for `different from`, the bare institution
  noun ("in hospital", "at university", "in future" for "in the future"), `have got` for `have` or
  `must`, `shall` for the future or for obligation, `at the weekend`, `was sat`/`was stood` for the
  progressive, a collective noun with a plural verb ("the committee have"), and `providing that`
  for `provided that`.

  Two carve-outs, both narrow. A **verbatim quotation** of published wording keeps its source's
  spelling: correcting inside `` ``...'' `` would falsify the quotation. A British form in
  **reproduced published prose of one of the three papers** is an errata decision, not a spelling
  fix, and goes to the author with its cost under the errata regime (NORTH_STAR §5.7) rather than
  being changed quietly. Gate 25 (`src_utils/check_register.py`) enforces the mechanical half and
  holds the author-owned hits open by name.

- **No phrasing a non-native writer would not produce, and this is the harder rule.** The author's
  second complaint, from the same reading (PENDENCIAS_RESOLVIDOS 2.22 (arquivado 2026-07-30) points 9 and 12): prose that is correct,
  even well written, and still forces a non-native reader to read it twice. His two instances, both
  from Appendix F, and his own words on each:
  - *"Two departures from that flat picture appear"* — **"pure A.I, we can be more simple"**.
  - *"Both point away from trouble in any case. A positive cosine is mild cooperation, not conflict,
    and the decline stays inside the margin throughout while moving toward zero rather than away
    from it."* — **"well written, but is not natural for a non native writer in english, and force a
    non native read more than once to understand"**.

  "Avoid awkward phrasing" would be unenforceable, so the ban names **shapes**:
  - **Inverted or delayed subjects.** A named subject held away from its verb by a modifier chain,
    with an intransitive verb of appearance at the end of the clause ("Two departures from that flat
    picture *appear*"), and the cleft that does the same job ("What carries that diagnosis is…").
    Name the subject and let it act: "The figure shows two departures."
  - **An abstract noun as the subject where a person or a thing would do.** A number does not move,
    prefer, or point: "the decline stays inside the margin *while moving toward zero rather than
    away from it*" asks the reader to animate a statistic. Say what was measured: "the mean falls
    toward zero and stays inside the margin."
  - **Chained qualification inside one sentence.** Three or more qualifying connectives (while,
    whereas, though, rather than, instead of, throughout, nevertheless) in one sentence, each
    narrowing the last. Split it, or drop the qualifications the claim does not need. Honesty does
    not require them in a single breath; §3 requires them *present*, not *stacked*.
  - **Idiom that is native literary register rather than academic register.** "point away from
    trouble", "in any case", "at any rate", "by the same token", "not least", "if anything". This
    extends §4's idiom rule from phrasal-verb metaphors to literary connective idiom, which is the
    class that survived the earlier sweeps.

  **The test is his, and it has two halves.** Would a Brazilian non-native writer of academic
  English produce this sentence? And can a non-native reader take it in on **one** reading? A no to
  either sends the sentence back. Technical difficulty is not a defense: keep the technical term,
  define it once, and simplify everything around it.

  **What is gated and what is judged.** Gate 25 catches the four shapes above, and that is the
  mechanical half only. The judgment half is the readability editor's first-read method
  (`reviewers/15_readability_editor.md`, review method and lens 2), whose verdict is PASS only with
  zero passages that need a second reading; `AGENT_GUARDRAILS.md` §5 makes that verdict part of G3.
  This rule deliberately does not restate that method, and **a green gate is not a first-read PASS**.
- **No em-dash anywhere.** Use commas, parentheses, semicolons, or two sentences. (Also an AI
  tell; also the MobiWac rule.)
- **No process narration, and this is a hard ban.** The prose states what is true of the work, never
  how the work came to be done or what the writing went through. Four sub-classes, each with a real
  instance from this repository that a reader would have received:
  - **Infrastructure.** *"the machine that would have run them was out of disk"* (Appendix F, deleted
    2026-07-30 at the author's instruction). A lab machine's free space is not a fact about mobility
    data. Also banned: GPU model, queue state, wall-clock caps, checkpoint sizes, run directories.
  - **The document's own version history.** *"California, Texas, and Istanbul were absent from an
    earlier version of this appendix"* (same paragraph). The reader of the deposited document never
    saw an earlier version and cannot act on the fact that one existed. Errata belong in the errata
    appendix, where the regime is deliberate and the reader is told why.
  - **Scheduling and provenance of the agent's own effort.** *"were measured afterward"*, *"at the
    time of writing"*, *"we then ran"*. When a measurement was taken is a fact about the project, not
    about the result. If a date genuinely qualifies a claim (a dataset vintage, a snapshot of a live
    resource), the date goes in and the narrative does not.
  - **Self-reference to the writing.** *"this appendix originally reported"*, *"as noted above"*,
    *"the boundary the paragraph above draws"*. The last of these also violates §4's ban on restating
    a section, and it appeared in the same deleted paragraph.

  **Why the ban is absolute rather than a matter of taste.** Process narration reads as an excuse, it
  dates the document the moment the circumstance changes, and it is unverifiable by the reader: a
  banca member cannot check that a machine was full. It is also the single easiest AI tell to spot,
  because a human author simply omits it. **The test:** if the sentence would be false or pointless
  once the circumstance changes, or if it explains why something is missing rather than stating what
  is present, cut it and put the reason in a source comment or a round report.

  **Where the material goes instead.** A limitation the reader must know ("this appendix covers one
  architecture family") is a LIMITATION and stays, stated as a property of the evidence. The reason
  the limitation exists goes in the provenance comment. Nothing is lost by the cut; measured before
  deleting the paragraph above, its only fact (the dataset coverage) was already stated three other
  times including in the figure caption.

## 2 · Canonical names (repo-wide; unchanged from the paper GLOSSARY §1)

> This section is the core rules only — the FULL expanded registry (model lineage, protocol
> terms, metrics, acronyms, PT equivalents, per-paper task mapping) is
> [`GLOSSARY.md`](GLOSSARY.md), and its fail-closed maintenance rule governs new terms.

| Concept | Use | Never |
|---|---|---|
| The "what" task | **next category** / category prediction | activity, POI classification |
| The "where" task | **next region** / region prediction | area, "next-POI" for region |
| The exact-place task | **next place** (we do NOT predict it; say so once, early) | conflating it with the other two |
| A place | place / point of interest (POI) | venue |
| One visit | check-in | event |
| Our representation | **check-in-level representation (Check2HGI)** | "substrate" (repo word) |
| Place-level baseline | place embedding (HGI) | "the baseline" alone |
| One model, both tasks | the joint model / single multitask model | bare "MTLnet" before it is introduced |
| One task, one model | dedicated single-task model | "baseline" alone |
| Repetition unit | **seed** = one complete repetition of the five-fold experiment with a different random initialization (define once, then "seed") | "run", "multi-seed run", bare "seed" in the abstract (say "random initialization") |

- Hyphenate the compound adjective ("next-category prediction"), leave the bare task name open
  ("the next category").
- **No repo codenames in dissertation prose**: B9, v11–v17, champion-G, H3-alt, dk_ovl, log_T
  (write "region-transition prior"), C25/CH16/F-numbers, "engine", "board", "recipe" (write
  "training configuration"), "frozen" (write "fixed", except frozen weights, glossed).
- One name per concept for the whole document; synonym-cycling is both imprecise and an AI tell.
  If a paragraph repeats a term 4+ times, restructure the sentences, never rotate synonyms.
- Model names across chapters: **MTLnet** (CBIC/CoUrb architecture), **ST-MTLNet** (CoUrb input
  variant), **Check2HGI** (representation), and the MobiWac joint model described per its paper.
  The Fundamentals chapter carries a small lineage table so the names never blur.

## 3 · Honesty rules (non-negotiable; violations are bugs, not style)

- **Every number carries its reference point** (majority-class floor, Markov floor, dedicated
  ceiling) and its convention (which metric, which selection rule, n=how many). Never a naked
  percentage.
- **Verbs bound to tests** (MobiWac law, applies wherever those results appear): "outperforms"
  only with paired superiority (Istanbul/FL/TX/CA region; category everywhere); "matches" /
  "statistically non-inferior within a two-point margin (TOST)" at AL/AZ; **never upgrade AZ
  (0.00)**; never "ties", "Pareto", "outperforms region everywhere", never "beats"/"wins".
- **Time-indexed claims** (this arc's rule): CBIC's "MTL does not help" and CoUrb's protocol
  are presented as conclusions *of the time, for that configuration*. Superseded numbers never
  read as current. Corrections are stated as corrections ("later shown to be
  configuration-specific"), with the correcting chapter named.
- **Uncertainty is stated, not implied**: fold-std or CI wherever a mean appears in a claim;
  "significant" only with the test named.
- Scope every universal: "at all six datasets" only right after the six are enumerated; the
  region-count scaling claim is scoped to the five U.S. states; bare "everywhere" never.
- Limitations are concrete ("3,150 rows", "2009–2010 Gowalla") and split Viegas-style:
  design-time scope/assumptions in §1.4 vs. evaluation-time limitations in §6.2.
- Failures and negative results are findings, not embarrassments — the CBIC null result is the
  arc's foundation; write it with the same care as the wins.

## 4 · AI-tell law (2026 state; inherits GLOSSARY §7 wholesale, plus the updates below)

**Inherited bans (see the paper GLOSSARY §7 for the full tables — they apply verbatim):** the
banned-word list (delve, intricate, showcase, underscore, pivotal, leverage, seamless, testament,
moreover-family openers, "it is worth noting", …), banned templates ("not only X but also Y",
"plays a crucial role", "in today's world", Firstly/Secondly scaffolds, participial significance
tails), and the density rules (≤1 intensifier per claim; -ly adverb density ≈0.8% max, never two
in one sentence; no semicolon braids; no rule-of-three cascades; vary paragraph openings; never
end a section by restating it).

**2025–2026 updates (from the current detection literature; sources in
[`AGENT_GUARDRAILS.md`](AGENT_GUARDRAILS.md) §8). The numbered items below are cited elsewhere
as §4.1–§4.6:**

1. **The list is versioned and rotting.** Word lists decay as models and authors adapt (the
   classic tells are already declining in 2024–25 corpora). Re-audit with fresh eyes per pass;
   do not treat the ban table as complete protection.
2. **Distributional tells matter more than tokens.** Watch for: nominal style (noun/determiner
   density creeping up, adjectives/adverbs vanishing), uniformly formal impersonal register, and
   shrinking vocabulary. Spot-check: read one page aloud; if every sentence has the same weight,
   rewrite.
3. **Variance compression is the deepest tell.** LLM revision homogenizes sentence and paragraph
   statistics (measured: Claude-family revision reduced variance in ~78% of stylometric
   features). The law: **preserve burstiness** — keep short sentences next to long ones, let some
   paragraphs open with a result, keep the author's own phrasings when editing; an editing pass
   that only smooths is a regression.
4. **Discourse-skeleton reuse.** Models recycle the same section shapes (identical opening move,
   same transition sentences, same wrap-up) — across a 100-page document this is glaring. Chapter
   openers must not follow one template; results discussions must vary their move order; the
   per-section "purpose statement" rule (§1) is about content, not a fixed sentence shape.
5. **Model-specific tics ("aidiolects").** For Claude-family drafting specifically: watch
   "genuine/genuinely", "comprehensive", "robust" (decorative), "crucially", "notably", and
   over-hedging stacks ("it is important to note, however, that…"). One decorative instance is
   noise; density convicts.
6. **Do not over-ban:** robust, novel, framework, comprehensive, baseline are normal CS words
   when load-bearing. The offense is decoration and stacking.

**Idiom rule (inherited GLOSSARY §8):** no phrasal-metaphor idioms ("edges past", "buys",
"ships", "lands", "folds in", "clears it by"); metaphor budget for "carry/carries" ≤3 per
chapter; "deliberately X" → "X by design"; "sits above" → "lies above" (one verb everywhere).

## 5 · Structure and presentation rules (Viegas-derived; details in `exemples/viegas/VIEGAS_ANALYSIS.md`)

- **Abstract formula:** problem → barrier → named contribution → concrete capabilities →
  validation design → ONE headline number → closing thesis-verb restatement. UFV catalog header;
  keywords per system rules (one per line, lowercase except proper nouns).
- **Introduction devices:** funnel (3 paragraphs max to the gap); the research question bold
  inline, once; objectives 1:1 with chapters; the coletânea "magic sentence" in the organization
  section with per-chapter venue+status bullets; contributions taxonomy (Theoretical / Software /
  Empirical / Practical) with section cross-refs.
- **Every chapter Introduction ends with a roadmap paragraph**; every section opens with a
  purpose statement (varying shape, per §4.4).
- **Tables:** booktabs only (no vertical rules); captions **above** tables (ABNT; fix Viegas's
  inconsistency); mean ± std with per-block bolded best values; every results table introduced by
  a lead takeaway sentence (a normal sentence, never a literal "Read this as:" tag).
- **Figures:** captions below, 2–4 self-contained interpretive sentences (name every element,
  include reading instructions and color/symbol legends); a notation/legend figure before any
  figure that uses custom notation; color + hatch dual encoding (grayscale-safe); in-figure text
  near body size.
- **Metrics defined defensively:** formula + one-sentence plain reading + boundary/degenerate
  behavior (the Viegas KIS pattern) — apply to macro-F1, Acc@10, the OOD-discounted region
  metric, and the checkpoint-selection rule.
- **Reproducibility blocks:** each experimental section opens with hardware, seeds, folds,
  versions; the code repository footnoted at first mention of each artifact; infrastructure
  software (PyTorch etc.) cited formally in the bibliography.
- **Hygiene sentences:** one explicit in-line sentence per leakage-sensitive step ("X was
  computed on the training folds only…") — the pattern MobiWac §5.2 and the A4 audit already use.
- **Anti-patterns from the example (do NOT imitate):** leftover "this paper"/"the article is
  organized as" inside chapters; "Dataset N" instead of names in results prose; unresolved
  citation keys; inconsistent caption placement; cross-ref typos; 3-line stretched chapter
  titles.

## 6 · Language mechanics per part

- **Frame chapters (1, 2, 6):** English, this file in full force.
- **Paper chapters (3–5):** the source paper's prose is the base; edits obey this file; the
  MobiWac chapter additionally obeys the paper GLOSSARY (it is stricter about plain words —
  do not "re-technicalize" its prose when re-typesetting).
- **Portuguese surfaces** (Resumo; AcademicoPG fields; folha de rosto boilerplate): formal
  PT-BR, no anglicisms where a standard PT term exists (aprendizado multitarefa, ponto de
  interesse, representação em nível de check-in); the Resumo mirrors the Abstract content
  exactly (same claims, same numbers, same hedges — audit them as a pair).
- **The CoUrb chapter** follows open decision #3 (NORTH_STAR §5): if kept PT, this file's
  honesty and structure rules still apply; if translated, translation is a *faithful* re-typeset
  (asserted claims may not drift in strength — see AGENT_GUARDRAILS §4 gate).

## 7 · Consistency checklist (run before every advisor handoff; the gates in AGENT_GUARDRAILS automate part)

- [ ] Canonical names only (§2); zero repo codenames; zero synonym-cycling.
- [ ] Every acronym expanded at first use; acronym count minimal; List of Abbreviations complete.
- [ ] next category / next region / next place kept distinct; "we do not predict the exact next
      place" stated once, early.
- [ ] Region wording law intact (outperforms 4 / matches AL–AZ / never upgrade AZ); scaling claim
      scoped; every verdict verb bound to its test.
- [ ] Time-index framing on CBIC/CoUrb conclusions; no superseded number reads as current.
- [ ] Every number has reference point + convention; every mean has its spread; "significant"
      only with a test.
- [ ] AI-tell sweep: banned words/templates at zero; intensifiers ≤1 per claim; -ly density in
      band; no semicolon braids; paragraph shapes vary; chapter openers not templated.
- [ ] Language pass: simple, direct American English that the author would defend aloud; American
      spelling and usage are consistent; no passage requires a second reading for its intended
      meaning or logical connection.
- [ ] Register sweep (§1, gate 25 `check_register.py`): zero British spellings and zero British
      constructions outside a verbatim quotation; any hit in reproduced published prose is with the
      author as an errata decision, not changed quietly.
- [ ] Hard-phrasing sweep (§1): no delayed or inverted subject, no abstract noun as agent of a
      motion verb, no chained qualification, no literary connective idiom. The gate covers these
      four shapes; the first-read verdict is persona 15's and is required separately.
- [ ] Idiom sweep: no phrasal metaphors; em-dash count = 0; contractions = 0.
- [ ] Tables captioned above with lead sentences; figures self-contained; metrics defined with
      boundary behavior; hygiene sentences present at every leakage-sensitive step.
- [ ] Resumo ↔ Abstract claim-parity audit passed.
- [ ] "this paper"→"this chapter" sweep in re-typeset chapters; no "Dataset N" prose; no
      unresolved \ref/\cite.
