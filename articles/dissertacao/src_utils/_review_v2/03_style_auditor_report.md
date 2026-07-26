# 03 · Style auditor — the G3 style gate

**Build audited:** `src/chapters/*.tex` + `0_main.tex` at 2026-07-25 23:43, against the rendered
94-page PDF. **Date:** 2026-07-26. **Persona:** `reviewers/03_style_auditor.md`. Read-only, fresh
eyes. This gate's output is quantitative: every metric below carries its number.

**Special charge this round:** WRITING_LAW §4.3 — "an editing pass that only smooths is a
regression." The corrected sentences were written by an assistant, so they are the text most at risk
of variance compression. I measured for it specifically rather than reading for it.

## Verdict

**GATE PASS.**

Zero em-dashes, zero contractions in prose, zero repo codenames, zero registry violations. The
banned-word table is at zero in frame prose; the fourteen hits all sit in reproduced paper text or
in Appendix B quotations of wording being corrected. Adverb density is inside the band in every
frame chapter. Most importantly for this round: **the corrected sentences did not compress
variance** — the chapters carrying the heaviest edits have the *highest* sentence-length dispersion
in the document, not the lowest.

Two MODERATE and five MINOR findings, none gate-blocking.

## Top 3 findings

1. **S-01 (MODERATE)** — semicolon braids: nine sentences carry two or more semicolons; five are in
   Chapter 2 and are structural list-in-a-sentence constructions.
2. **S-02 (MODERATE)** — Chapter 3's adverb density is 1.69%, more than double the ~0.8% band.
3. **S-03 (MINOR)** — three sentences exceed 70 words, all in frame chapters.

---

## The counted report

### Hard bans (WRITING_LAW §1, §4)

| Check | Count | Status |
|---|---:|---|
| **Em-dashes** (any file, any position) | **0** | **PASS** |
| **Contractions in prose** | **0** | **PASS** — one hit at `2_fundamentals.tex:175` is inside a LaTeX comment quoting a source document ("that's tuned for dense"), which does not render |
| **Repo codenames** in prose (B9, v11–v17, champion-G, H3-alt, dk_ovl, log_T, substrate, engine, board, recipe, frozen) | **0** | **PASS** — verified individually. "region-transition prior" is used where `log_T` would be (`5_mobiwac.tex:367`); "fixed at its initial values" where "frozen" would be (`:659`), with the source comment at `:689-690` recording the substitution as deliberate |
| **`mtlnet_crossattn_dualtower`** (GLOSSARY: "NEVER appears in text") | **0** | **PASS** |
| Banned verbs: `beats` / `wins` / `ties` / `Pareto` (as verdicts) | **0** | **PASS** — `Pareto` appears once at `3_cbic.tex:108` as the cited technique's own term ("MGDA finds Pareto-optimal descent directions"), which is the method's name, not a verdict |

### Banned words and templates

Fourteen hits total, **zero in frame-chapter prose**. Full disposition:

| Word | Hits | Where | Disposition |
|---|---:|---|---|
| `Furthermore` | 2 | `apx_b_errata.tex:202,204` | **exempt** — Appendix B quotes the published CBIC wording it is documenting the replacement of |
| `Moreover` | 2 | `apx_b_errata.tex:197,198` | **exempt** — same |
| `leverag*` | 3 | `apx_b_errata.tex:191,193,195` | **exempt** — same |
| `underscore` | 1 | `apx_b_errata.tex:200` | **exempt** — same |
| `not only` | 2 | `4_courb.tex:227,377` | reproduced paper prose (translated CoUrb); `:377` is "there is no single universally superior spatial encoder. SIREN presents better…" where "not only" is part of a comparative clause, not the banned "not only X but also Y" template |
| `comprehensive` | 1 | `3_cbic.tex:297` | reproduced CBIC prose ("For a comprehensive evaluation, we selected two state-of-the-art approaches"). Load-bearing, not decorative |
| `crucially` | 1 | `3_cbic.tex:377` | reproduced CBIC prose |
| `notably` | 1 | `3_cbic.tex:340` | reproduced CBIC prose |
| `landscape` | 2 | `3_cbic.tex:340,377` | reproduced CBIC prose ("a competitive performance landscape") |
| `Additionally` | 1 | `3_cbic.tex:162` | reproduced CBIC prose, inside a mathematical definition |
| `state-of-the-art` | 1 | `3_cbic.tex:297` | reproduced CBIC prose. WRITING_LAW §4.6 explicitly declines to over-ban working CS words |

**The frame chapters (1, 2, 6) and the abstract carry zero banned-word hits.** Chapters 3 and 4 are
reproduced published text where the errata policy governs what may be changed; Appendix B's hits are
quotations of the text being corrected, which is the only honest way to document a substitution.

Zero hits for: delve, intricate, showcase, pivotal, seamless, testament, "it is worth noting",
"plays a crucial role", "in today's world", Firstly/Secondly/Thirdly, paradigm shift, meticulous,
nuanced, holistic, myriad, tapestry, navigate, harness, cutting-edge, game-changer, deep dive,
multifaceted, intricacies, genuine/genuinely.

### Density metrics

**-ly adverb density** (band ≈0.8% max; excluding only/early/family/apply/supply/reply/likely):

| Chapter | words | -ly | density | verdict |
|---|---:|---:|---:|---|
| 1 Introduction | 1,443 | 8 | **0.55%** | in band |
| 2 Fundamentals | 3,761 | 20 | **0.53%** | in band |
| 3 CBIC | 3,675 | 62 | **1.69%** | **over** — see S-02 |
| 4 CoUrb | 4,052 | 53 | **1.31%** | over (reproduced text) |
| 5 MobiWac | 5,846 | 27 | **0.46%** | in band |
| 6 Conclusion | 1,255 | 7 | **0.56%** | in band |
| Appendix A | 494 | 2 | 0.40% | in band |
| Appendix B | 1,316 | 8 | 0.61% | in band |

Chapter 5's 0.46% is the lowest in the document, and its -ly inventory is almost entirely
load-bearing: `statistically` ×3, `exactly` ×3, `entirely` ×3, `jointly` ×2, `usually` ×2. Those are
precision words, not decoration.

**Intensifiers** (very, quite, extremely, highly, significantly, substantially, considerably,
remarkably, notably, greatly, strongly, clearly, essentially, particularly, especially):

| Chapter | count | per-claim |
|---|---:|---|
| 1, 2 | **0**, **0** | — |
| 5 MobiWac | **1** | ≤1 |
| 6 Conclusion | **1** | ≤1 |
| 3 CBIC | 12 | reproduced text |
| 4 CoUrb | 3 | reproduced text |

Frame prose is essentially intensifier-free. This is unusually disciplined.

**Semicolons:** 5 / 23 / 11 / 6 / 62 / 4 (Ch.1–6). Chapter 5's 62 is high but the majority are
inside statistical parentheticals (`(each entry: point estimate; interval)`) and citation lists,
which the law exempts.

**Em-dash: 0. Contractions: 0.** Both mandated at zero, both met.

### Variance and burstiness — the §4.3 check

This is the measurement the round required. Sentence-length distribution per chapter:

| Chapter | n | mean | sd | **CV** | min | max |
|---|---:|---:|---:|---:|---:|---:|
| 1 Introduction | 53 | 28.5 | 14.1 | **0.495** | 6 | 65 |
| 2 Fundamentals | 153 | 26.2 | 14.9 | **0.568** | 4 | 75 |
| 3 CBIC | 171 | 22.9 | 9.5 | **0.414** | 4 | 71 |
| 4 CoUrb | 170 | 25.3 | 10.7 | **0.424** | 5 | 65 |
| 5 MobiWac | 225 | 27.7 | 13.8 | **0.497** | 4 | 75 |
| 6 Conclusion | 43 | 30.6 | 19.6 | **0.640** | 6 | 110 |

**Reading:** the two chapters this round edited most heavily — Chapter 5 (the four grounds, the
freeze control, the third limitation, the results split) and Chapter 6 (the qualified verdict, the
completed California run) — show CV 0.497 and **0.640**, the highest dispersion in the document.
The reproduced paper chapters, which the round barely touched, sit lowest at 0.414 and 0.424.

If the edit pass had smoothed, the edited chapters would be the *flattest*. They are the burstiest.
Chapter 6 in particular runs a 6-word sentence against a 110-word one. **No variance compression
detected.** The round passes §4.3.

Spot-check by reading, per the persona's procedure: `6_conclusion.tex:92-119`, the capacity-matched
paragraph written this round, alternates a 12-word sentence ("Two controls separate this claim from
wishful attribution.") against a 47-word one, and closes on a short observational sentence. It does
not have one sentence weight.

### Sentence openers (discourse-skeleton variety, §4.4)

Top repeated two-word openers:

- Ch.1: "The task" ×3, "The two" ×2, "The exact" ×2, "This dissertation" ×2 — 53 sentences
- Ch.2: "The dissertation" ×3, "This section" ×2, "In the" ×2, "Multi-task learning" ×2 — 153 sentences
- Ch.5: "The first" ×3, "We therefore" ×3, "The joint" ×3, "One model" ×3 — 225 sentences
- Ch.6: "Chapter contributed" ×3 (i.e. `Chapter~\ref{...} contributed`), "With a" ×2 — 43 sentences

No opener exceeds 3 occurrences in any chapter, and no chapter opens more than 2% of its sentences
the same way. Chapter 6's "Chapter X contributed" ×3 is a deliberate parallel across the three
contribution paragraphs, which is a rhetorical structure rather than a template tic.

**Chapter openers** are not templated: Ch.1 opens on a definition, Ch.2 on a roadmap sentence, Ch.5
on the problem, Ch.6 on the research question. Four different moves.

**Section-ending check:** I looked for sections that close by restating themselves. Chapter 2 §2.5
ends on a forward hinge ("The chapters that follow answer these three questions in turn",
`:621-622`), not a restatement — and the source ledger at `:635` records "Section does not restate
2.1-2.4 (no 'in summary')" as an explicit design constraint. Verified: no "In summary", "To
summarize", or "In conclusion" openers anywhere in frame prose.

---

## Ranked findings

### S-01 · MODERATE · Nine semicolon braids

A prose sentence with two or more semicolons is two sentences (WRITING_LAW §4, CI notation exempt).
Nine occurrences:

**Chapter 2 (five)** — all are list-in-a-sentence constructions:

> "It defines the prediction tasks and keeps them distinct (Section ); follows the line of
> representations for mobility from one-hot identifiers to the check-in level (Section ); reviews
> multi-task learning and the conditions under w[hich]…"
> — `2_fundamentals.tex`, the chapter roadmap

> "STAN models spatio-temporal correlations between non-adjacent visits and shows that visits far
> apart in a sequence still carry signal ; GeoSAN encodes geography through a hierarchical grid and
> a self-attention layer ; and GETNext…"

> "The graph convolutional network combines a node's features with those of its neighbors through a
> localized spectral rule ; graph attention networks weight each neighbor by learned attention
> rather than by fixed degree ; and GraphS[AGE]…"

> "Random loss weighting is a competitive baseline ; a controlled study finds that current MTL
> optimizers often do not outperform a well-tuned fixed-weight baseline ; and a direct defense of
> plain loss summation shows…"

> "The field has a place-level representation but not a check-in-level one built for these targets;
> it has multi-task learning but almost no treatment of the next region as an end target; and it has
> strong evaluation practice that mo[st]…"

**Chapter 5 (two)** — both are CI parentheticals, **exempt**:

> "…pass with 90 % confidence intervals well inside two points (each entry: point estimate;
> interval): Alabama (…) and Arizona (…)"

**Chapter 3 (two)** — reproduced paper text (the five-dimension survey list and the optimizer list).

*Assessment.* The Chapter 2 five are a recognizable and legitimate academic construction: a
three-item parallel list where each item is a clause. They are not the "semicolon braid" the law
targets (a run-on stitched from unrelated statements) — each is a deliberate parallel with an "and"
before the final item. But the law's threshold is mechanical and these cross it. The last one is the
strongest of the five and I would not touch it; it is the chapter's synthesis sentence and the
parallelism is the point.

*Direction:* author's ruling. Either accept parallel-list semicolons as an exemption (and record it
in WRITING_LAW, see "Proposed law updates"), or break the two weakest (the GCN one and the STAN one)
into two sentences each.

### S-02 · MODERATE · Chapter 3's adverb density is 1.69%, double the band

62 -ly adverbs in 3,675 words. This is reproduced CBIC prose, so the errata policy governs: the
round's own Appendix B Table 12 shows the substitutions applied were targeted at *claim-strength*
words (`significantly`, `statistically`) rather than at density. Those four substitutions are
correct and were the right priority.

The residual density is a property of the published paper, not of this round's work. Chapter 4 shows
the same pattern (1.31%) for the same reason.

*Direction:* no action recommended. Recorded so the number is on the table if the author decides to
extend the errata to register. Flagging it *without* recommending a fix is deliberate: rewriting a
published paper's prose for density would be a larger departure from the source than the errata
policy contemplates, and Appendix B would have to grow accordingly.

### S-03 · MINOR · Three sentences exceed 70 words

Max sentence lengths: Ch.2 75 words, Ch.5 75 words, Ch.6 110 words, Ch.3 71 words.

The Chapter 6 110-word sentence is the outlier by a wide margin. Long sentences are not banned — and
burstiness *requires* some — but 110 words asks a reader to hold too much. Persona 15 owns the
reader-experience judgment; I record the measurement.

### S-04 · MINOR · Term-registry lint: clean, with one observation

Every recurring concept carries exactly one name. Verified against GLOSSARY §1–§4:

| Concept | Name used | Alternatives found |
|---|---|---|
| the "what" task | next category / next-category prediction | none |
| the "where" task | next region / next-region prediction | none |
| the exact-place task | next place | none — and "We do not predict the exact next place" appears once, early (`5_mobiwac.tex:226`) |
| one visit | check-in | "event" **0 hits** |
| a place | place / POI | "venue" **0 hits** |
| our representation | check-in-level representation (Check2HGI) | "substrate" **0 hits** |
| place-level baseline | place embedding (HGI) | bare "the baseline" not used as the referent |
| one model, both tasks | the joint model | bare "MTLnet" before introduction: none |
| one task, one model | dedicated single-task model | bare "baseline" alone: none |
| repetition unit | seed, defined at `5_mobiwac.tex:394` ("A \emph{seed} is one complete repetition of the five-fold experiment, over the same folds, with a different random initialization") | "run" / "multi-seed run" **0 hits**; the abstract correctly says "random initializations" not "seeds" (`0_main.tex:286`) |

**Observation, not a violation:** the round introduced "fitted models" as a countable unit ("twenty
fitted models per configuration"). The source comment at `0_main.tex:233` records this honestly:
"'fitted models' ('modelos ajustados') NAO consta do GLOSSARY §6: entrada proposta". Per the
fail-closed maintenance rule, the term should land in GLOSSARY §4 and §6 before it lands in text. It
is already in text at four sites. The term itself is well-chosen and unambiguous; this is a
bookkeeping gap, and the drafting pass flagged it rather than hiding it. **Recommend the author
approve the GLOSSARY entry.**

### S-05 · NIT · "carry/carries" metaphor budget

Budget is ≤3 per chapter. Counts: Ch.2 4, Ch.5 6. Both over.

Inspection: nearly all are the literal sense ("each visit's own category enters as a node feature…
what that feature can carry between visits", "every number carries its reference point", "the
sniffed rows carry…"). The idiom rule targets the decorative metaphorical use ("this result carries
weight"), which I found zero of. **Over budget on the token count, compliant on the offense.**

### S-06 · NIT · Structure spot-checks (WRITING_LAW §5)

| Rule | Result |
|---|---|
| Every results table has a lead takeaway sentence | **yes** — Table 9 (`5_mobiwac.tex:409`), Table 10 (`:490` "One model outperforms or matches the dedicated…"), Table 8 (`:304`). No literal "Read this as:" tag anywhere |
| Captions above tables / below figures | **yes**, all 22 floats (verified on rendered pages) |
| Metrics defined defensively at first use | **yes** — macro-F1 with its plain reading and its floor (`:389`); Acc@10 with its boundary behavior, "a visit whose true region is absent from that fold's training data counts as an error" (`:391`) |
| Hygiene sentences present at leakage-sensitive steps | **yes** — existence verified at splitting (`:365`), representation training (`:367`), region-transition prior (`:367`), baseline representations (`:367`). Content is persona 07/09's call |
| Section purpose statements, varying shape | **yes** |

### S-07 · NIT · Two `(??)` renders are a style-visible defect

Not my gate (persona 05 owns citations, 18 owns pages), but `(??)` on four rendered pages is the
kind of thing the G3 reader notices. Cross-referenced.

---

## Proposed law updates (for author approval — never applied)

1. **Parallel-list semicolons.** WRITING_LAW §4's semicolon-braid rule currently reads as an
   absolute (2+ semicolons = two sentences), with only CI notation exempt. Five legitimate
   constructions in Chapter 2 cross it. Propose adding: *a three-or-more-item parallel list whose
   items are clauses, closing with "and", is exempt* — with the caveat that the items must be
   genuinely parallel.
2. **"Fitted models" into GLOSSARY.** §4 already carries the "n = 20 (fitted models) and n = 4
   (inferential unit)" row; the countable noun phrase "fitted models" should be listed as the
   canonical term in §2/§3 and get a PT equivalent ("modelos ajustados") in §6, since it now appears
   in the Resumo.
3. **No new tells found in the wild.** I looked for 2026-vintage tells beyond the law's tables
   (nominal-style creep, uniform impersonal register, shrinking vocabulary) and found none worth
   adding. The document's register is varied and concrete.

---

## What is legal and load-bearing — do not push toward sterility

The following are normal, correct, working CS prose and must not be "fixed" by a later pass:

- **`robust`, `framework`, `baseline`, `novel`** where they appear — all load-bearing, none
  decorative. WRITING_LAW §4.6 exists for exactly this.
- **Chapter 5's 62 semicolons** — the majority are statistical notation. Removing them would make
  the interval reporting worse.
- **The long sentences in Chapter 5's four-grounds paragraph** (`:367`). It is a single dense
  paragraph because the four grounds are one argument; breaking it into four paragraphs would lose
  the "we bound what we can measure and name what we cannot" structure that makes it honest.
- **Chapter 6's 110-word sentence and its 6-word neighbors.** The dispersion is the point.
- **The plain verbs.** `use`, `cost`, `show`, `obtain`, `reach`, `remain`, `include`, `provide`,
  `predict`, `measure`, `train`, `keep` dominate the frame chapters. This is the register the law
  asks for and the document achieves it.
- **Chapter 5's "We report this attribution as a finding, not a hypothesis."** (`:668`) — 11 words,
  flat, declarative, and it does more work than a paragraph of hedging would.

## Out-of-scope handoffs

- Persona 15: the 110-word sentence in Chapter 6; whether Chapter 2's parallel-list sentences read
  well as opposed to merely being legal.
- Persona 05: the four `(??)` renders.
- Persona 07: I checked that hygiene sentences *exist*; their content is the claim gate's.
