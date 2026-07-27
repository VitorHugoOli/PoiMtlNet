# Persona 17 · Excellence assessor: the Resumo/Abstract pair

> **Scope.** The Resumo (rendered p. 3) and the Abstract (rendered p. 4) of
> `src/dissertacao.pdf`, defense build rebuilt 2026-07-27, 103 pp, against the five exemplar
> dissertations in `exemples/`. Nothing else in the document was reviewed. Read-only: no file
> was edited, no build was run.
> **Source of the text under review:** `src/0_main.tex:193-282` (resumo environment) and
> `src/0_main.tex:285-357` (abstract environment). Everything quoted below was extracted from the
> committed PDF and cross-checked against the committed source.
> **Verdict: BELOW the exemplar standard as currently rendered, ABOVE it in substance.** Two
> BLOCKERs, both from the same round-5 edit, are the whole of the gap. Details in §2.

---

## 0 · Two corrections to the charge, filed first (fail-closed)

**(a) There is no `checkpoint` in the Resumo.** The charge states the Resumo italicizes four
English terms: *check-ins*, *embedding*, *joint-best*, and *checkpoint*. Measured on the rendered
page by font span, the Resumo carries **three** italic terms and the Abstract carries **one**:

| Block | Italic spans measured (font `TeXGyreTermesX-Italic`, 11.96 pt) |
|---|---|
| Resumo, p. 3 | `check-ins`, `embedding`, `joint-best` |
| Abstract, p. 4 | `check-ins` |

The string `checkpoint` occurs **zero** times on pp. 3–4, zero times in `0_main.tex`, and zero
times in either rendered build (103-page defense, 99-page final). Its only occurrence anywhere in
the LaTeX source is inside a comment at `chapters/5_mobiwac.tex:795`, which does not render. The
fourth term does not exist and needs no defense; the asymmetry between the two blocks does, and is
finding MINOR-1.

**(b) The word counts I measured differ from 529/452, because the basis differs.** On whitespace
tokens of the rendered text: Resumo **body 484 words**, Abstract **body 407 words**. Adding the
UFV catalog header and the keyword block gives 536 and 451, consistent with the 529/452 recorded
at `src_utils/_archive/PENDENCIAS_RESOLVIDOS.md:235-236` to within PT hyphenated-token counting.
I report body-only throughout, because that is what is comparable across the five exemplars. The
compression itself is confirmed: the page did close, p. 4 is no longer nearly blank, and the
keyword block sits with its Resumo on p. 3.

---

## 1 · The measured comparison

All ten texts extracted from the committed PDFs and normalized identically: ligatures expanded,
end-of-line hyphenation repaired, UFV catalog header and keyword block excluded, so the counts are
body prose only. "Sentence units" counts terminal-punctuation-delimited units as a reader meets
them, which is why our two blocks show ten units for eight authored sentences (see BLOCKER-1).
"Quantitative facts" counts numerals plus spelled-out quantities, excluding indefinite articles.

| Text | Words | Sentence units | Mean sent. len | SD | Longest | Quant. facts | States a limitation |
|---|---:|---:|---:|---:|---:|---:|---|
| **OURS Resumo (PT)** | **484** | **10** | **48.4** | **23.5** | **111** | **21** | Partial |
| **OURS Abstract (EN)** | **407** | **10** | **40.7** | **21.7** | **99** | **20** | Partial |
| Germano 2024 Abstract | 214 | 10 | 21.4 | 4.2 | 29 | 0 | No |
| Viegas 2026 Abstract | 200 | 8 | 25.0 | 6.7 | 33 | 3 | No |
| Canesche 2021 Resumo | 257 | 11 | 23.4 | 7.0 | 36 | 5 | No |
| Canesche 2021 Abstract | 243 | 12 | 20.2 | 8.3 | 33 | 5 | No |
| Passe 2020 Resumo | 229 | 10 | 22.9 | 7.4 | 35 | 1 | No |
| Passe 2020 Abstract | 193 | 10 | 19.3 | 7.6 | 36 | 1 | No |
| Dorigueto 2021 Resumo | 282 | 8 | 35.2 | 15.4 | 68 | 3 | No |
| Dorigueto 2021 Abstract | 225 | 6 | 37.5 | 15.8 | 61 | 3 | No |

**Exemplar envelope:** words 193 to 282 (median 227); mean sentence length 19.3 to 37.5; longest
sentence 29 to 68; quantitative facts 0 to 5. **Our Resumo is 2.1x the exemplar median length and
carries four times the quantitative load of the most numeric exemplar.** Its longest sentence
(111 words) is 1.6x the longest sentence in any of the ten texts.

Three observations the table does not show on its own.

1. **`UFV_COMPLIANCE.md` §2 imposes no word limit**, and I found none in the local norm
   extraction. So length is not a compliance question. It is a standards-against-peers question,
   and against peers we are the outlier.
2. **Three of the five exemplars carry a Resumo; two do not.** Canesche (p. 7), Passe (p. 7) and
   Dorigueto (p. 6) render both languages, as the table above shows, so rendering the pair is the
   majority precedent and ours follows it. The two that lack one are **Viegas (p. 5) and Germano
   (p. 2)**, which render the Abstract only; neither PDF contains the string `RESUMO` or
   `Palavras-chave` on any page. Both are pre-deposit builds, and
   `exemples/viegas/VIEGAS_ANALYSIS.md:95-96` already records this and warns against copying it.
   That matters here because those two are the closest precedents on other axes, Viegas
   structurally and Germano by program and advisor, so our pair rendering both languages is a
   point over them specifically rather than over the exemplar set as a whole.
3. **The limitation column is "No" for all five exemplars.** Ours is "Partial": both blocks
   disclose the resampling scope in the protocol clause ("um único conjunto fixo de cinco
   partições" / "one fixed set of five folds"). That is a real honesty device none of the
   exemplars attempts, and it is on the protect list.

---

## 2 · The two BLOCKERs

Both come from the same edit: the round-5 compression that took the pair from 565/485 to 529/452.
`src_utils/_archive/PENDENCIAS_RESOLVIDOS.md:236` records what was cut as "sairam as duas glosas
parenteticas da selecao \emph{joint-best} e a frase de motivacao foi comprimida." The compression removed
the **heads of two sentences in each language** and left the tails rendering as lowercase
fragments. This is not the trapped-prose bug that `check_trapped_prose.py` gates: the words are
not hidden inside a comment, they were deleted outright, so that checker cannot see this and
`check.sh` has no lint that would.

### BLOCKER-1 · Four sentences render decapitated on the first two pages a banca reads

**Resumo, `0_main.tex:208-209`, rendered p. 3 line 10:**

> "…por meio de aprendizado multitarefa (MTL). **entre tarefas pode prejudicar uma delas**, um modo
> de falha conhecido como transferência negativa, …"

The subject is gone. What renders is not merely miscapitalized, it is ungrammatical: "entre
tarefas pode prejudicar uma delas" has no subject in Portuguese, so the PT reader is never told
*what* can hurt one of the tasks.

**Resumo, `0_main.tex:237-238`, rendered p. 3 line 36:**

> "…margem de dois pontos de Acc@10 (TOST), nos outros dois. **condicional, e a condição é o
> achado**: se o aprendizado multitarefa ajuda a previsão sobre pontos de interesse depende…"

**Abstract, `0_main.tex:300-301`, rendered p. 4 line 9:**

> "…together through multi-task learning (MTL). **sharing parameters** between tasks can hurt one
> of them, a failure mode known as negative transfer, …"

**Abstract, `0_main.tex:326-327`, rendered p. 4 line 31:**

> "…within a two-point Acc@10 margin (TOST), at the other two. **condition is the finding**:
> whether multi-task learning helps point-of-interest prediction depends…"

**Severity BLOCKER.** These are pages 3 and 4. A reader who opens the defense PDF meets an
ungrammatical Portuguese clause in the fourth sentence of the document. `AGENT_GUARDRAILS`
§6/D1 and Appendix C both assert the author read and approved every word; four broken sentence
heads on the two highest-visibility pages is the single hardest place in the document to say that
out loud. Persona 16's round-2 report made exactly this argument about one lowercase sentence
opener on p. 72 (`_review_v2/16_ai_credibility_report.md:372`); this is the same defect class,
four instances, in the front matter, and it was introduced *after* that report.

**Repair is verbatim-available in the repo, so no new prose is needed and no claim moves.** The
canonical heads exist:

| Site | Restore from | Canonical text at that source |
|---|---|---|
| `0_main.tex:209` | `chapters/1_introduction.tex:83` | "Sharing parameters between tasks can hurt one of them, a failure mode known as negative transfer." |
| `0_main.tex:301` | same | same |
| `0_main.tex:238` | `chapters/6_conclusion.tex:82-83` | "The dissertation's answer is conditional, and the condition is the finding." |
| `0_main.tex:327` | same | same |

Cost of the repair as modelled (not applied): Resumo 484 → 489 words, Abstract 407 → 414, ten
sentence units in each become ten complete sentences, zero lowercase openers. **Five words in PT
and seven in EN.** The page-fill measurement says this is safe: p. 3 has 49.9 pt of clear space
below its last baseline and p. 4 has 127.9 pt, so restoring twelve words cannot re-open the
near-blank-page defect that round 5 closed. I did not build to confirm; treat the page-count
consequence as `[VERIFY]` on the author's next compile.

### BLOCKER-2 · The claim-parity pair is broken in both directions, at the thesis's central hedge

I ran 40 claim tokens in both directions: every number, every verdict verb, every hedge, every
test name, every scope enumeration, every named system. **38 match. Two break, and they break
opposite ways**, because the round-5 deletion cut a different number of words in each language.

| # | Claim token | Portuguese | English | Consequence |
|---|---|---|---|---|
| 1 | **The answer is conditional** (the dissertation's central hedge) | `condicional` present, 1x | `conditional` **absent, 0x** | The EN reader is told "condition is the finding" without ever being told the answer *is* conditional. The thesis's own qualifier is missing from the English record. |
| 2 | **Negative transfer's mechanism** (what can hurt a task) | subject **absent**; no `compartilhar`/`Compartilhar` anywhere in the Resumo | `sharing parameters between tasks` present | The PT reader learns the failure mode's *name* but not its *cause*; the EN reader learns both. |

**Severity BLOCKER**, on the charge's own terms: the pair is a declared claim-parity pair
(`WRITING_LAW.md` §6; `GLOSSARY.md:116`; `src_utils/PENDENCIAS.md:384-385` restates it as the rule
that cannot be worked around), and a claim present in one language and absent in the other is
exactly what parity forbids. Break #1 is the more serious of the two: `NORTH_STAR` and
`6_conclusion.tex:82-83` make "the answer is conditional" the dissertation's answer to its own
research question. Repairing BLOCKER-1 repairs both breaks at the same stroke; they are one
defect seen from two angles, and I report them separately only because the parity audit is the
gate that catches the asymmetry.

**What parity holds on, verified token by token, so the author knows what not to re-check:** the
check-in definition; both task names; the MTL expansion; `transferência negativa`/`negative
transfer`; the open-question framing; three studies; the coletânea-in-chronological-order framing;
the negative-result/diagnosis/resolution arc; `MTLnet`; the place-level embedding gloss; hard
sharing; `não superou de forma consistente`/`did not consistently outperform`; the cost clause;
`resultado nulo`/`null result`; architecture held fixed; the decomposed encoders; `subiu de forma
acentuada`/`rose sharply`; the time-indexed bottleneck hedge; the check-in-level move; the
per-visit-vector gloss; the cross-attention trunk; the task-pair-changed disclosure; user-disjoint
CV; `vinte modelos ajustados`/`twenty fitted models`; four initializations over one fixed set of
five folds; the n = 4 inferential unit; the non-inferiority test; the leakage audit;
`supera`/`outperforms`; the six-dataset scope enumeration; **5,3–9,4 / 5.3–9.4 macro-F1**;
`joint-best`; four of six; `equipara-se estatisticamente`/`statistically matches`; the two-point
Acc@10 TOST margin; `nos outros dois`/`at the other two`; `a condição é o achado`/`condition is
the finding`; and the depends-on close. Numerals are identical modulo decimal convention
(`5,3`/`5.3`, `9,4`/`9.4`, `10`), and the spelled-quantity inventories match one-for-one at 16
items each.

---

## 3 · Insertion quality and register: the advisor's specific complaint

The advisor's words: *"soa um pouco estranho o jeito que alguns termos sao inseridos."* Not which
terms, but how they arrive. I tested each of the four terms named in the charge against three
questions: does it arrive with a gloss the surrounding register can absorb, is it introduced at or
before first use, and is it used again.

### MAJOR-1 · `joint-best` is a document hapax: it exists only in the Resumo and the Abstract

Measured across all 103 pages of the defense build, `joint-best` (and the variants `joint best`,
`jointbest`) appears on **exactly two pages: 3 and 4**. In the 99-page final AcademicoPG build it
appears on **zero pages**, because that build has no front matter. So the term is introduced in
italics, in a foreign-term slot, in the single most-read paragraph of the document, and is then
never used, never defined, and never repeated anywhere in the body.

The body does carry the convention; it just says it in other words, at `p. 69`:

> "Throughout, every reported model is one saved artifact per fold, read at its validation-selected
> epoch: each dedicated model at its task's best epoch, and the joint model at the epoch selected
> by its joint validation score (the geometric mean of the two task metrics), with both tasks read
> from that one model."

`GLOSSARY.md:83` registers "joint-best convention" as a term, so the word is licensed. What is not
licensed is the arrival: the reader meets `sob uma seleção joint-best` / `under a joint-best
selection` on p. 3 with no gloss, and the document never returns to it. This is precisely the
failure the advisor described, and it is the strongest instance of it in the pair. Round 5 made it
worse rather than better: `PENDENCIAS_RESOLVIDOS.md:236` records that what the compression cut was
"as duas glosas parenteticas da selecao \emph{joint-best}", that is, **the two parenthetical glosses
that were this term's only support were the words removed.** The term stayed and its explanation
left.

*Direction (author's call, three options, no wording proposed as final):* (i) restore a short gloss
at the p. 3/p. 4 site, in parity, at a cost of roughly eight words per language; (ii) drop the
term and keep the plain description, since the range 5,3–9,4 needs a stated convention but not
necessarily this name; (iii) keep the term and add it to the body at its p. 69 definition so the
front matter is forward-referencing something that exists. Option (ii) is the cheapest and the
only one that also reduces length. Option (iii) is the only one that makes the italics earn their
place. Whichever is chosen, it must be applied to **both** blocks: right now the Resumo sets
`joint-best` in italics and the Abstract sets it roman, which is MINOR-1.

### MINOR-1 · The emphasis convention differs between the two halves of a parity pair

| Term | Resumo (PT) | Abstract (EN) | Assessment |
|---|---|---|---|
| `check-ins` | italic | italic | **Correct in PT, unearned in EN.** In Portuguese, italicizing a foreign loanword at first use is what a Brazilian committee expects. In English, `check-ins` is an English word inside English prose; the italic marks nothing. None of the four exemplar abstracts italicizes its own-language technical terms (Germano writes "Points of Interests (POIs)" roman; Canesche writes "(CGRAs)" roman). |
| `embedding` | italic, inside a gloss | **roman**, bare ("a place-level embedding as input") | **PT arrival earned, protect it** (see below). The EN side uses the same term unmarked, which is correct for English prose but makes this the second term whose emphasis differs across the pair. |
| `joint-best` | italic | **roman** | Inconsistent across the pair. Same term, same claim, two conventions. |
| `checkpoint` | absent | absent | Does not exist (§0a). |

The EN italic on `check-ins` is a NIT on its own; it becomes a MINOR because in a declared parity
pair the emphasis convention is itself a claim about which words are foreign, and that claim
cannot be true in both languages at once.

### What the `embedding` gloss gets right, and why I am defending it

> "um modelo conjunto com uma representação (*embedding*) em nível de POI como entrada"

This is the model for how a term should arrive, and it should not be touched. `GLOSSARY.md:104`
registers exactly this PT form for `place embedding`, and `GLOSSARY.md:117-118` states the policy
the sentence follows: "Embedding may stay as a loanword in PT (standard in the BR community) with
representação as the running word." The Resumo does precisely that: it introduces the loanword once,
parenthetically, subordinate to the Portuguese head noun, and then uses `representação` for the
rest of the block, four times, never returning to the loanword. A Brazilian committee reads this
as a Portuguese sentence that acknowledges an English term of art, not as English dropped into
Portuguese. Compare the `joint-best` arrival above, which does the opposite. If the author wants a
pattern to apply elsewhere in the pair, this is it.

### MAJOR-2 · The reserved term `next place` is used, on p. 3 and p. 4, for the category task

Both blocks read:

> "a categoria do **próximo lugar visitado** e a região onde ela ocorrerá"
> "the category of the **next place visited** and the region where it will happen"

`GLOSSARY.md` reserves **next place** for the exact-POI task the dissertation does not study, and
the document defends that reservation three times: `1_introduction.tex:57-59` ("The exact *next
place* task … is a third and different problem; this dissertation does not address it"),
`2_fundamentals.tex:62`, and `3_cbic.tex:30-31` ("the dissertation reserves *next place* for the
exact-POI task, which is not studied here"). `WRITING_LAW.md` §2 lists conflating the three tasks
as the one thing never to do, and §7's checklist requires the non-prediction to be stated "once,
early."

On the rendered page, "early" is p. 3, and what p. 3 says is that the dissertation predicts the
category of the *next place*. The abstract also never states that the exact next place is not
predicted; that disclaimer first reaches the reader at `1_introduction.tex:172`, rendered p. 14.
A committee member who reads the Resumo and skips to the results has been given the collision and
not the correction. The body's own fix at `1_introduction.tex:53` shows the safe phrasing exists:
"the category of the next visited place", same meaning, no reserved term.

*Direction:* change `próximo lugar visitado` / `next place visited` to the body's own
`próximo lugar a ser visitado` / `next visited place` at both sites, and consider adding the
non-prediction clause in parity if the length budget allows after §4's cuts. Zero-cost on
substance, and it removes the one place in the pair where the canonical-name law is broken.

### MINOR-2 · `trace` / `histórico` is used but is not in the registry

`the two tasks read the same trace` / `as duas tarefas leem o mesmo histórico`. `GLOSSARY.md`
contains no entry for `trace`, `histórico`, or `trajetória`, and the registry's own rule
(`GLOSSARY.md:124-125`) is fail-closed: a term not registered "does not exist for this
dissertation." The body uses `traces` freely (first prose use p. 12, then pp. 14, 17, 20), so the
EN side is at least consistent with the body. The PT `histórico` is the weaker half: it is not the
standard PT rendering of a check-in trace, it is not registered, and a Brazilian reader meets it
as a slightly vague noun where `histórico de check-ins` or `trajetória` would be exact. Two
options: register the pair in `GLOSSARY.md` §6, or say `o mesmo histórico de check-ins` in PT and
register that. Either satisfies the rule; neither costs a claim.

---

## 4 · Does the pair do the job for a reader who reads nothing else?

**It does the hardest part of that job better than any exemplar, and misses three cheap things.**

Against the abstract formula the document set itself (`WRITING_LAW.md` §5, from
`VIEGAS_ANALYSIS.md:56-59`): problem → barrier → **named contribution** → capabilities →
validation design → ONE headline number → thesis-verb close. Measured presence:

| Move | Present | Evidence |
|---|---|---|
| Problem | Yes | "Antecipar duas propriedades da próxima visita…" |
| Barrier | Yes, damaged | `transferência negativa`/`negative transfer` present; its subject is BLOCKER-1/BLOCKER-2 |
| **Named contribution** | **No** | See MAJOR-3 |
| Capabilities | Yes | the three-study arc, 25 words in EN, 30 in PT |
| Validation design | Yes, over-delivered | see MAJOR-4 |
| Headline number | Yes | 5,3–9,4 macro-F1 points, with its convention named |
| Thesis-verb close | Yes, damaged | the depends-on close, decapitated per BLOCKER-1 |

### MAJOR-3 · What is missing: the dissertation never names what it built

Neither block names **Check2HGI**, which `GLOSSARY.md:47` calls "Ch.5 (the centerpiece)" and which
the title's own promise ("From Representations to…") points at. Neither names **ST-MTLNet**.
Neither names **Gowalla** or **Massive-STEPS**. `MTLnet` is named, and it is the model the
dissertation reports a null result for; so the one system a reader can name from the front matter
is the one that did not work.

The exemplars all do this and it is the cheapest excellence move available here. Dorigueto names
LapsusVGI and LapsusTerrae; Canesche names CGRA and the Zigzag traversal; Passe names Node-RED,
RISCVerilog, READY and PLAIN; Viegas names Causal-Nest, the Knowledge Integrity Score, the Sachs
dataset and CatBoost; Germano names HAVANA and HAMURE with both acronyms expanded. Our pair
describes its contribution in full and correct prose ("moveu a representação para o nível do
check-in, dando a cada visita seu próprio vetor") and then declines to say what that thing is
called. A reader who reads only p. 3 cannot cite this work by the name of its artifact, and a CTD
or award committee reading a products list would find no handle.

*Direction:* name `Check2HGI` at the third-study sentence in both blocks, apposed to the
description already there, and name the two data sources in the results sentence where "cinco
estados dos Estados Unidos e Istambul" already stands. Cost: roughly six words per language, and
it converts the vaguest sentence in the pair into the most citable one. This is a delivery
upgrade, not a scope change: no claim is added, no verb is strengthened.

### MAJOR-4 · What is surplus: the abstract carries the full statistical protocol

The single result sentence is **111 words in PT and 99 in EN**, with 13 and 12 commas and two
parenthetical inserts each. Its neighbours in the same block run 19 to 54 words. It carries four
protocol facts (n = 20 fitted models; four initializations over one fixed fold set; paired tests
on four per-initialization means; non-inferiority testing) plus a leakage audit, before it reaches
a single result.

**This class is already logged and already deferred, and I am not re-opening it.** `CODEX_AUDIT.md`
COD-016 confirms it, `PENDENCIAS.md:100` carries it, and the author's recorded decision at
`PENDENCIAS.md:391-392` is to point out only the most critical items. Two refinements are worth
having on the record anyway, because both are new measurements:

1. **The recorded figure needs correcting.** COD-016 records 114 words for "the abstract's result
   sentence." Measured on the current post-compression text, macros stripped: **EN 99 words,
   PT 111 (112 in source tokens)**. The 114 predates round 5.
2. **The Portuguese one is the longer of the two and has never been separately measured.** Every
   prior report measured the Abstract. At 111 words the PT sentence is the longest single sentence
   in any of the ten texts compared here, by 43 words.

*Direction, when the author takes the deferred language pass:* the protocol clause is where the
length is, and it is also where the honesty is, so cut with care. The defensible split is to keep
in the abstract what a verdict cannot be read without (user-disjoint cross-validation, the n, the
TOST margin with its metric) and let the rest live in the chapter, where it already does. The
other cut candidate is the task-pair-change sentence (38 words PT, 28 EN): it is meta-commentary
about how the collection was assembled rather than about the research result, and §1.5 or the Ch.3
preface can carry it. **Both cuts touch honesty devices, so both need persona 14 and the honesty
gate before application**. `AGENT_GUARDRAILS` treats a lost disclosure as a claim change, and the
task-pair disclosure in particular is the sentence that keeps the CBIC/CoUrb task pair from
reading as the MobiWac one.

### What is neither missing nor surplus, and reads well

The three-study arc is the best thing in the pair. "Esta dissertação responde a essa pergunta ao
longo de três estudos … um resultado negativo, seu diagnóstico e sua resolução" does in 30 words
what the persona brief calls the dissertation's natural superpower, and it does it as a promise the
next four sentences then keep, one study each, in parallel construction, with the diagnosis
("a representação de entrada, e não a arquitetura de compartilhamento, era o gargalo") stated as
the finding it is. No exemplar abstract attempts a narrative of its own error. This is above the
standard and must survive any trim.

---

## 5 · Register mechanics, swept

Clean, and worth recording as clean so no one re-audits it: **em-dashes 0** in both blocks
(en-dashes 0 as well), **contractions 0**, **`WRITING_LAW` §4 banned tokens 0** (no delve,
intricate, showcase, underscore, pivotal, leverage, seamless, testament, moreover, furthermore,
crucial, notably, comprehensive, robust, genuine, "it is worth noting", "not only"). Adverb
density is inside the §4 band: four `-ly` adverbs in 407 EN words (0.98%, and `consistently`,
`only`, `sharply`, `statistically` are all load-bearing rather than decorative), one `-mente` in
484 PT words (0.21%).

One distributional note against §4.4, discourse-skeleton reuse: sentence openers run
"O primeiro estudo… / O segundo estudo… / O terceiro estudo…" and "The first study… / The second
study… / The third study…". Deliberate parallelism serving the three-study arc, and I read it as
earned rather than templated, but it is the one place in the pair where three consecutive
sentences share a shape. If the author wants variance, vary the **third** one, since that study is
the resolution and currently reads as merely the next item in a list.

---

## 6 · Award lens, restricted to this pair

Could this Resumo and Abstract carry a CTD-style screen? **Yes, after BLOCKER-1 and MAJOR-3.**
The screen's question is whether a committee can extract problem, contribution, evidence and
significance from one page. Problem and evidence are already stronger here than in any exemplar:
ours is the only one of the ten that binds a verdict verb to a named test, scopes a universal to
an enumerated set, and time-indexes a superseded conclusion. Contribution is the gap, and it is
the gap precisely because the artifact is unnamed (MAJOR-3): a products list needs a noun. As
rendered, the four broken sentence heads would end the screen before the substance was reached,
which is why BLOCKER-1 outranks everything else in this report despite being a twelve-word fix.

---

## 7 · The protect list

Do not let any repair or trim touch these.

1. **The verdict-verb discipline.** `supera` / `equipara-se estatisticamente` and `outperforms` /
   `statistically matches`, with the two-point Acc@10 TOST margin named in full, and four-of-six
   scoped explicitly. Better than all five exemplars and the reason the pair survives a hostile
   read.
2. **The three-study arc sentence and the four study sentences that keep its promise** (§4).
3. **The time-indexed hedge**: "era o gargalo naquele estágio da pesquisa" / "was the bottleneck at
   that stage of the research." Four words doing the work of a paragraph.
4. **The `(embedding)` gloss** and its follow-through on `representação` (§3). It is the pair's own
   model for how a foreign term should arrive.
5. **The resampling-scope disclosure** "um único conjunto fixo de cinco partições" / "one fixed set
   of five folds", and the n = 4 inferential unit alongside the 20 fits. This is the honesty that
   `_review_v2` fought for; it is also 30 words, and it will look like the obvious cut when the
   length pass comes. It is not.
6. **The headline number with its convention attached**, whatever happens to the word
   `joint-best` (§3, MAJOR-1): the range 5,3–9,4 may not appear without a stated selection basis.
7. **Both blocks rendering, in both languages**, against the Viegas and Germano precedent of
   Abstract-only.

---

## 8 · Ranked findings

| # | Severity | Finding | Location |
|---|---|---|---|
| 1 | **BLOCKER** | Four sentences render decapitated; the PT instance at p. 3 line 10 is ungrammatical | `0_main.tex:208-209, 237-238, 300-301, 326-327`; rendered p. 3 lines 10 and 36, p. 4 lines 9 and 31 |
| 2 | **BLOCKER** | Claim parity broken in both directions: `conditional` absent from EN; the negative-transfer subject absent from PT | same sites; audit in §2 |
| 3 | MAJOR | `joint-best` occurs only on pp. 3–4 of 103, nowhere in the body, and round 5 deleted the glosses that supported it | rendered pp. 3, 4; body convention at p. 69; cut recorded at `PENDENCIAS_RESOLVIDOS.md:236` |
| 4 | MAJOR | The reserved term `next place` is used for the category task, before the non-prediction disclaimer reaches the reader | rendered pp. 3, 4; reservation at `1_introduction.tex:57-59`, `3_cbic.tex:30-31` |
| 5 | MAJOR | No artifact is named: `Check2HGI`, `ST-MTLNet`, `Gowalla`, `Massive-STEPS` all absent; only `MTLnet`, the null-result model, is named | both blocks |
| 6 | MAJOR (deferred class) | The result sentence is 111 words PT / 99 EN; prior record of 114 predates round 5, and the PT sentence had never been measured | `0_main.tex:229-237`, `:319-326`; COD-016 |
| 7 | MINOR | Emphasis convention differs across the parity pair: `joint-best` italic in PT, roman in EN; `check-ins` italicized in EN where the italic marks nothing | rendered pp. 3, 4 |
| 8 | MINOR | `trace` / `histórico` used but unregistered, against the fail-closed registry rule; `histórico` is the weaker PT choice | both blocks; `GLOSSARY.md` §6, `:124-125` |
| 9 | NIT | Three consecutive sentences open on the same frame ("O primeiro/segundo/terceiro estudo") | both blocks |

---

## 9 · Verdict

**Substance: above the exemplar standard, clearly.** On the dimensions this persona scores,
problem framing and empirical rigor are delivered in this pair better than in any of the five
exemplars: it is the only one of the ten texts that names a statistical test, attaches a
non-inferiority margin to a metric, enumerates the scope of a universal claim, and reports its own
null result as a finding rather than omitting it. Germano's abstract, the closest precedent, same
program and same advisor, carries zero numbers and opens on a grammatical error. Ours carries
twenty and a conditional thesis. Critical self-assessment, dimension 7 of the rubric, is
**OUTSTANDING** in this pair and it is the reason to protect §4's arc sentence above everything
else.

**Execution: below the standard, and the gap is narrow and mechanical.** Four broken sentence
heads on pp. 3 and 4, one of them ungrammatical in the language the committee will read first, and
a parity break at the dissertation's central hedge. Every exemplar's abstract is grammatically
clean; ours is not. Nothing about this is a writing problem, a length problem, or a judgment
problem: it is twelve words that a compression pass removed and no gate could see, and the
replacement text already exists verbatim in `1_introduction.tex` and `6_conclusion.tex`.

**The single highest-leverage investment remaining in this scope** is the twelve-word restoration
in §2. It clears both BLOCKERs, restores parity to 40 of 40, and costs less than an hour including
a rebuild. Second is naming `Check2HGI` and the two datasets (§4, MAJOR-3), roughly six words per
language, which converts the pair from a correct description of a contribution into a citable one.
Third is deciding `joint-best`'s fate (§3, MAJOR-1), which is the one finding here that speaks
directly to what the advisor raised. Everything else on the list can wait for the deferred
language pass, and the length question can wait indefinitely: no norm requires the cut, and the
substance currently justifies the words.

---

## 10 · Open questions only the author can answer

1. **`joint-best`**: restore a gloss, drop the term, or add it to the body? Option (ii) is the only
   one that also shortens; option (iii) is the only one that makes the italics legitimate.
2. **`modelos ajustados`** as the PT equivalent of "fitted models" is still not in `GLOSSARY.md` §6
   (`PENDENCIAS.md:387-389` asks the same question and is unanswered). It is in the rendered
   Resumo now, so the registry is behind the text. Confirm the term or replace it.
3. **The non-prediction clause** ("the exact next place is not predicted"): add it to the pair in
   parity, or accept that its first appearance is p. 14? Adding it costs about eight words per
   language and closes MAJOR-4's exposure.
4. **`trace`/`histórico`**: register the pair, or change the PT to `histórico de check-ins` and
   register that?
5. Whether the six `[NEEDS SIGN-OFF]` markers on this pair (`PENDENCIAS.md:372`) can be cleared
   before the repairs above land. They cannot honestly be cleared while BLOCKER-1 stands, because
   Appendix C's claim covers these two pages too.

## 11 · Out-of-scope handoffs (one line each, not pursued)

- `check.sh` has no lint for a sentence that begins lowercase after a full stop.
  `check_trapped_prose.py` is scoped to comment-swallowed prose and structurally cannot see a
  plain deletion, which is why four instances shipped through gate day. A one-line rendered-text
  regex over the built PDF would have caught all four. Persona 19 / tooling scope.
- `CLAUDE.md` §1 still describes the state as "2026-07-24 — v1 ASSEMBLED + corrections round 2"
  while the builds under review are from 2026-07-27 round 5. Persona 13/18 scope.

---

### Method and traceability

Text extracted from the committed PDFs with PyMuPDF 1.28.0 (MuPDF 1.29.0); italic detection by
font-name inspection of rendered spans, not by source markup; hyphenation artifacts repaired
before counting and the repair verified against the source; all ten texts normalized by one
function so the table's columns are comparable. Every quoted string was re-checked verbatim
against the extraction before being written here. No number in this report was carried over from a
prior review without re-measurement, and where a prior figure disagreed with mine (COD-016's 114
words) I report both and name the basis. Two consequences I could not verify read-only and flag as
`[VERIFY]`: the page count after restoring twelve words, and whether `529`/`452` were counted on
the basis I reconstructed in §0b.
