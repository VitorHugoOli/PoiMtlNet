# 17 · The Resumo and the Abstract, reviewed after the cut

**Track.** The author ruled that both blocks be cut and largely rewritten (*"No review acima ei
decidir por cotar e refazer boa parte do abstract/resumo"*). This is the fresh-eyes check on whether
that landed. Six personas over those two blocks only: **01** cold reader, **03** style auditor,
**06** number auditor, **07** claim and honesty auditor, **15** readability, **16** AI credibility.
Every finding below is tagged with the persona that produced it.

**Scope.** `src/0_main.tex` Resumo prose block (lines 229-254 today) and Abstract prose block
(315-336), as **rendered**. Nothing else in the document was reviewed. Read-only: no file under
`src/` was touched.

**What I measured on, and how.** The convention is the precedent report's
(`src_utils/_review_v3/17_resumo_abstract_assessment.md`) as re-stated by this round's report §1:
counts from the **rendered page**, UFV catalog header and keyword block stripped, hyphenated
compounds one word, sentence enders excluding decimals, initials and `Sec./Fig./cf./e.g./i.e./et
al.`. I ran the round's own instrument, `src_utils/_round6/_measure_abs.py`, unmodified.
**Convention unchanged from the precedent, so the numbers are comparable.** One difference of basis
worth naming: the precedent measured whitespace tokens and reported 484/407; this round's
`_measure_abs.py` counts letter/digit tokens. Both are stated; I use the round-6 instrument
throughout so my before/after is comparable with the report I am checking.

**The build I measured.** Two independent artifacts, so no finding rests on a stale PDF:

| artifact | pages | Resumo | Abstract | provenance |
|---|---:|---|---|---|
| `src/dissertacao.pdf` (worktree) | 108 | p.2 | p.3 | the committed reading copy |
| `build/main.pdf` built by me from `git archive 01915ba7` | **108** | p.2 | p.3 | isolated tree, `make defense` exit 0 |
| `build/main_ppgc.pdf` (worktree) | 109 | p.3 | p.4 | approval sheet shifts by one |

`0_main.tex` in the worktree is **byte-identical** to `01915ba7` (`diff` clean), and the only change
to it since the cut commit `40ed8e7b` is the approval-sheet `\ifapprovalsheet` guard from
`7a91b720`: the prose is the cut prose, unmodified. The three paper chapters were split into
per-section files at `4e84cf7a` while I worked; that does not touch either block, and every body
coordinate I cite below is re-resolved against the **new** layout with today's line number.

---

## Verdict

**The cut landed. Both blocks are now defensible on the page a committee reads first, and the
specific defects the author was told about are real and closed.** The rewrite fixed the four
decapitated sentences of round 5, added the next-place exclusion that the pair had never carried,
brought both mean sentence lengths inside the defended envelope, and reclaimed the orphan keyword
page. The nineteen floor claims are present in both languages, the verdict verbs are bound to their
tests in both, Arizona is not upgraded, and the law sweeps are clean at zero.

**Gate verdicts by persona:**

| persona | verdict | basis |
|---|---|---|
| 03 style auditor | **GATE PASS** | em-dash 0, contractions 0, banned tokens 0, codenames 0, idiom 0, adverb density in band, both languages |
| 06 number auditor | **GATE PASS** | every numeral traces to `stats_n20/RESULTS.md` rev 4 or the results table; PT/EN identical modulo decimal comma |
| 07 claim and honesty | **PASS WITH FINDINGS** | verbs bound, floor intact; two honesty devices thinned in ways the report does not name (M-1, M-2) |
| 01 cold reader | followed it | one stumble worth logging (M-4) |
| 15 readability | 8.0 overall | one sentence at 58 words carries the whole protocol (m-2) |
| 16 AI credibility | screener LOW / expert LOW | the arc and the null result are the human signal; no gestalt tells |

**Nothing here is a BLOCKER.** Two MAJORs are claim-scope questions the cut created or inherited
and did not disclose; the rest are MINOR. The pair is in materially better shape than the version
the precedent report reviewed.

**Top three findings:**

1. **M-1** (persona 07) — the time-index hedge left the pair. `naquele estágio da pesquisa` / `at
   that stage of the research` was in the pre-cut text and is gone; both the body sites that carry
   the same claim still have it. WRITING_LAW §3 makes the time-indexing of the CoUrb conclusion
   non-negotiable, and the report's §4.2 loss table does not list this among its cut claims.
2. **M-2** (persona 07 + 06) — "em todos os estados testados" / "in every state tested" is an
   unscoped universal. Chapter 4 tested **three** states; the pair's next sentence says **five**,
   and nothing between them tells the reader the two counts belong to different studies. Chapter 6
   says "across the three states tested" at the same claim.
3. **M-3** (persona 06) — `_check_pair_parity.py`, the gate this round leaves behind as its
   evidence, **fails on its own documented invocation**: its defaults are `PT_PAGE=3 EN_PAGE=4`,
   the current defense build has the pair on pp. 2 and 3, and run as documented it reports **19 of
   19 floor claims missing** while exiting 0. A future reader who runs it as its docstring says
   gets a total false alarm.

---

## 1 · Claim parity, claim by claim (persona 07)

The report states 19 floor claims present in both languages, zero failures. **I re-derived it
rather than trusting it, and it holds.** Run against the fresh `dissertacao.pdf` with the pages the
pair actually occupies:

```
floor failures: 0 []
rendered sentence counts: PT 11 / EN 11
rendered word counts:     PT 310 / EN 271
```

The nineteen are: research question open · answer conditional · representation dominates · input
over architecture · joint model outperforms · next category · next region · next place excluded ·
5.3 to 9.4 · joint-best convention · four of six · TOST two-point margin · Gowalla five U.S. states
· Istanbul Massive-STEPS · user-disjoint CV · MTL expanded · check-in level · n = 20 fitted models ·
equipara-se/matches.

Nineteen strings is not nineteen claims, so I ran three checks the string list does not cover.

**(a) Sentence-for-sentence correspondence: verified, 11 to 11, no drift.** Each PT sentence carries
the same claim as its EN twin, in the same order, with PT running 1.04 to 1.29 times longer per
sentence, which is normal expansion:

| # | PT w | EN w | ratio | the claim |
|---|---:|---:|---:|---|
| 1 | 17 | 15 | 1.13 | check-in defined as a visit by a user to a POI |
| 2 | 39 | 37 | 1.05 | both tasks; **the next-place exclusion**; same trace; MTL expanded |
| 3 | 27 | 26 | 1.04 | sharing can hurt a task; the question and its depends-on clause |
| 4 | 6 | 5 | 1.20 | three studies answer it |
| 5 | 30 | 24 | 1.25 | study 1: MTLnet, place-level embedding, hard sharing, no consistent gain |
| 6 | 34 | 30 | 1.13 | study 2: input replaced, category rose, representation was the bottleneck |
| 7 | 31 | 24 | 1.29 | study 3: check-in level, one vector per visit, cross-attention trunk |
| 8 | 17 | 16 | 1.06 | the six datasets and their two sources |
| 9 | 58 | 50 | 1.16 | the protocol and the category result with range and convention |
| 10 | 28 | 25 | 1.12 | the region result: four of six, TOST at the other two |
| 11 | 23 | 21 | 1.10 | the answer is conditional, and on what |

**(b) Hedge and modal parity: 16 of 17 tokens match.** `poderia`/`could`, `pode`/`can`,
`permanecia em aberto`/`was unresolved`, `consistentemente`/`consistently`,
`acentuadamente`/`sharply`, `logo`/`so`, `portanto`/`therefore`, `mas não`/`though not`,
`depende`/`depends`, `era`/`was`, and every scope quantifier (`nos seis`/`at all six`, `quatro
deles`/`four of them`, `outros dois`/`other two`) are present on both sides. The one asymmetry is
lexical, not semantic: PT `não superou` renders in EN as `did not consistently outperform`, which is
the same hedge in a different word order. **Not a finding.**

**(c) Emphasis parity: still broken, and still in the same direction the precedent flagged.**
Measured by font-name inspection of rendered spans, not by source markup:

| term | Resumo p.2 | Abstract p.3 |
|---|---|---|
| `check-ins` | italic | italic |
| `embedding` | italic | absent (EN says "a place-level embedding", roman) |
| `joint-best` | **italic** | **roman** |

Italic character totals: PT 28, EN 9. This is precedent MINOR-1, unchanged by the cut, and it is
**m-1** below.

---

## 2 · The hedging, both languages (persona 07)

**`supera` / `outperforms` is licensed everywhere it appears.** Traced to
`docs/studies/closing_data/v17_completion/stats_n20/RESULTS.md` **rev 4, 2026-07-13**, whose header
states A1 landed the CA and TX n = 20 runs so the six-dataset family Holm runs: all six category
cells reject at α = 0.05 (Holm-adjusted p ≤ 8.9e-07). I anchored on the **revision header**, not the
first matching line, which is the trap that produced the superseded-section error this round's
report §7 documents against itself. The four region `outperforms` cells are the four `$^{\uparrow}$`
rows of `src/tables/mobiwac/results.tex` (Istanbul, FL, TX, CA).

**`equipara-se estatisticamente` / `statistically matches` appears only at the two TOST cells**, and
both blocks name the test, the margin and the metric in full: "não-inferioridade dentro de uma
margem de dois pontos de Acc@10 (TOST)" / "non-inferiority within a two-point Acc@10 margin (TOST)".
That satisfies GLOSSARY §4's requirement that the full form appear at least once.

**Arizona is not upgraded.** `RESULTS.md` §1 records AZ region Δ +0.10, 90% CI (+0.001, +0.206), with
the explicit instruction not to upgrade to "beats". The pair puts AZ inside "nos outros dois" /
"at the other two", the non-inferiority group. No `beats`, `wins`, `ties`, `Pareto`, `vence`,
`empata` or `bate` in either block (swept, zero).

**A non-inferior result never reads as a win in either language.** Sentence 10 separates the two
verdicts with an explicit `e` / `and`: "supera em quatro deles **e** equipara-se estatisticamente
... nos outros dois". The scope precedes the universal in both.

**One residue, inherited rather than introduced (m-3).** Ch.5 §5.5 (`chapters/5_mobiwac/05_setup.tex`
today, anchor "did not cover next-region superiority") states that the four next-region gains "are
secondary results outside" the written analysis plan, which registered non-inferiority only for that
task. The pair says `supera` / `outperforms` for those four with no such qualification. This is
consistent with Ch.1 and Ch.6, which do not carry it either, so the pair mirrors the body correctly
and the question belongs to whoever owns the frame. Flagged, not charged against the cut.

---

## 3 · The next-place exclusion (persona 07)

**The report claims the pair GAINED this in the rewrite. Verified true, on both the rendered page
and the pre-cut source.**

Pre-cut text (`git show 40ed8e7b^:articles/dissertacao/src/0_main.tex`, prose extracted): the phrase
that carried this slot was "a categoria do próximo lugar a ser visitado e a região onde ela
ocorrerá" / "the category of the next visited place and the region where it will happen". Searched
the pre-cut prose for the exclusion in every form: `exato`, `exact next place`, `não o ponto de
interesse`, `does not predict`. **Zero hits in either language.** The precedent report's MAJOR-2 had
removed the reserved-term collision, and the removal left the exclusion itself unstated.

Current text, sentence 2 of both blocks, positively phrased:

> "sua categoria e sua região, **mas não o ponto de interesse exato**"
> "its category and its region, **though not the exact next place**"

This is the earliest place it can go, and it now precedes rather than follows the results, so a
committee member who reads only this page is told what is not predicted. The wording tracks the
body's own claim at `chapters/1_introduction.tex:172` ("The exact next place is not predicted
anywhere in this work") and `chapters/2_fundamentals.tex:62` ("It does not predict the exact next
place; that target is named only to hold it apart"). The reserved term `next place` is used in EN,
which is correct here because it is used **to delimit**, exactly as GLOSSARY §1 licenses; the PT
side avoids the term and says `ponto de interesse exato`, which is also correct and is why the
collision the precedent flagged does not return.

**One asymmetry, noted and not charged.** EN uses the reserved English term; PT uses a description.
Both are licensed, and forcing `próximo lugar` into the PT would reintroduce the collision the
precedent fought. Leave it.

---

## 4 · Publication status (persona 07)

**Neither block mentions publication status, venue, DOI or review state.** Swept both prose ranges
for `submitted`, `under review`, `published`, `CBIC`, `CoUrb`, `MobiWac`, `DOI`: zero hits. The
requirement is therefore satisfied vacuously, and the report says so.

I checked that the status lives where the report says it does, because a vacuous pass is only safe
if the claim is made correctly somewhere. `chapters/1_introduction.tex:118` reads "submitted to
MobiWac 2026 and under review"; `chapters/5_mobiwac.tex:24` reads "under review at the time of
writing (EDAS #1571313639)". Both are the licensed wording. **No finding.**

The report's reasoning for the absence is sound and worth preserving: an abstract that names three
venues and their review states spends words on bibliography rather than claims.

---

## 5 · The nine Portuguese terms registered on 2026-07-28 (personas 03 + 07)

GLOSSARY §6 gained nine rows at `01915ba7` because the Resumo already used them, which the
fail-closed rule does not permit. **Verified each against the rendered Resumo. Eight of nine are
used in their registered form; one is a fair composition; none is contradicted.**

| # | registered PT form | as rendered in the Resumo | verdict |
|---|---|---|---|
| 1 | vinte modelos ajustados (por configuração) | "vinte modelos ajustados por configuração" | **exact** |
| 2 | usuários disjuntos entre treino e teste | identical | **exact** |
| 3 | partição (uma de cinco); *as cinco* partições fixas | "cinco partições fixas" | **consistent** (the *as* is a definite article the sentence does not need) |
| 4 | média por inicialização; *plural* médias por inicialização | "as quatro médias por inicialização" | **exact**, plural form, as registered |
| 5 | seleção *joint-best* | "sob uma seleção *joint-best*", italic | **exact**, italics as the registry's convention specifies |
| 6 | tronco de atenção cruzada | identical | **exact** |
| 7 | codificadores decompostos (espacial, temporal, categórico) | "codificadores espaciais, temporais e categóricos decompostos" | **consistent**: the same three modifiers, inflected to plural and reordered to attach to the noun. The registry gives the term in citation form; PT syntax requires this order. |
| 8 | compartilhamento rígido de parâmetros | identical | **exact** |
| 9 | topologia de compartilhamento | identical | **exact** |

**And the tenth-term question, which is the one the brief asked.** I swept 43 technical noun phrases
actually rendered in the Resumo against every PT cell in §6. Most misses are ordinary Portuguese
(`arquitetura`, `vetor`, `conjuntos de dados`), proper nouns (`Gowalla`, `Istambul`,
`Massive-STEPS`), or metric names registered in §4 in English and used identically in PT (`macro-F1`,
`Acc@10`, `TOST`). Three deserve the author's eye, and one is a live inconsistency with the report
itself:

- **`histórico`** is the one that matters, and it is **m-4** below. Not in §6, not in §4, not
  anywhere in GLOSSARY or WRITING_LAW (grep, zero hits). The precedent report raised it as MINOR-2
  and it survived the cut untouched in both languages. The EN `trace` at least matches the body,
  which uses `traces` freely; the PT `histórico` is the weaker half.
- **`modelos ajustados`**: **now registered**, row 19 of §6, as of `01915ba7`. The round-6 report's
  `[VERIFY]` flag 2 says it is *still not* in §6 and asks the glossary owner to register it. **That
  flag is stale and can be closed** — the registration landed in the same commit the brief measures
  the state at. This is **m-5**, filed against the report rather than the document.
- **`modelo conjunto`**: §2 registers **the joint model** in English and §6 has no PT row for it. The
  Resumo uses `modelo conjunto` three times. Compositional from a registered English name, so it is
  the same class as the four the report's flag 3 already lists, and the strict reading is that §6
  should grow one more row.

---

## 6 · The keyword blocks (persona 03, UFV_COMPLIANCE §2)

Read verbatim off the rendered pages, not the source:

| | PT (p.2) | EN (p.3) |
|---|---|---|
| 1 | aprendizado multitarefa | multi-task learning |
| 2 | ponto de interesse | point of interest |
| 3 | previsão da próxima categoria | next-category prediction |
| 4 | previsão da próxima região | next-region prediction |
| 5 | representação em nível de check-in | check-in-level representation |

**Compliant on every clause of §2, and mirrored.** One term per line (five lines, five terms, each
its own rendered line). Lowercase throughout; no proper noun appears, so the exception is not
exercised. No terminal punctuation on any line, and none between terms; the only colon is on the
`Palavras-chave:` / `Keywords:` label, which is the label's own punctuation and matches the PPG
model. PT mirrors EN one-for-one in the same order, and every pair is a GLOSSARY §6 row (rows 1, 3,
5, 7, 9 of the table above). **No finding.**

The report says it did not touch these blocks because they were already compliant. Confirmed.

---

## 7 · The catalog headers (persona 06)

**Both echo the title verbatim.** Compared programmatically against `\titulo`:

```
titulo       : From Representations to a Single Joint Model: Multi-Task Learning for
               Point-of-Interest Category and Region Prediction
Resumo hdr   : identical  (string equality True)
Abstract hdr : identical  (string equality True)
```

On the rendered page both wrap as `From Re-presentations` / `From Rep-resentations`, which is
pdfium's soft hyphen at a justified line break, not a defect: the same string renders in the folha de
rosto uppercased and unbroken. The `agosto de 2026` / `August 2026` and `Orientador:` / `Advisor:`
fields differ by language, which is correct per the PPG Resumo/Abstract model. **No finding.**

---

## 8 · The near-blank page, verified on the raster (persona 06 + 18-adjacent)

**Gone. Confirmed on rendered pixels, not the text layer** — which is the specific failure this
brief warns about.

I rasterized every page of both defense builds at 150 dpi and measured ink coverage and vertical ink
span per page, then flagged any page under 40 text-layer words **or** under 1 percent ink:

| build | pages | sparse pages found |
|---|---:|---|
| defense (`dissertacao.pdf`) | 108 | p.87 (1 word, 0.174% ink) = the "Appendix" divider; p.108 (31 words, 0.728%) = the last page of Appendix E |
| ppgc (`main_ppgc.pdf`) | 109 | p.2 (17 words, 0.777%) = the approval-sheet placeholder; p.88 and p.109, the same two as above |

**No page carries a keyword block alone, and no near-blank page exists anywhere in the front
matter.** The three sparse pages that remain are a part divider, a normal short final page of an
appendix, and the intentional approval-sheet placeholder. On the pair's own pages the raster gives
Resumo 6.844% ink over 65.7% of page height and Abstract 5.828% over 59.1%: both a single full block
with about a third of the page as trailing whitespace, keyword block included. I read both rendered
images directly to confirm structure (heading, catalog header, one prose block, five keywords) with
no orphan and no widow.

**One correction to the report's page arithmetic, which does not affect the finding.** Report §3
says "the defense build goes from 105 pp to 104 pp". That was true in its isolated tree. The
document is **108 pp** now, and the brief already attributes that to unrelated causes. I verified
independently by building `01915ba7` in a clean `git archive` tree: **108 pages, `make defense` exit
0**. So the 104 in the report is a measurement of a tree that no longer exists, correctly labelled
as such, and the orphan-page fix survives at 108. Nothing to fix; recorded so the next reader does
not treat 104 as current.

---

## 9 · Numbers (persona 06)

Every numeral in both blocks, traced. **Nothing is computed in the pair; every figure is quoted.**

| numeral, as rendered | PT form | source of record | verdict |
|---|---|---|---|
| 5.3 to 9.4 macro-F1 points | 5,3 a 9,4 | `stats_n20/RESULTS.md` rev 4 §1: FL +5.34 low, AZ +9.40 high; AL +7.73, Istanbul +8.59, CA +6.45, TX +7.45 all interior | **traces**; range is correct on rev 4's own values |
| all six / nos seis (category) | nos seis | same, six of six reject, Holm-adj ≤ 8.9e-07 | **traces** |
| four of them / quatro deles (region) | quatro deles | `tables/mobiwac/results.tex`: four `$^{\uparrow}$` vs two `$^{\approx}$` | **traces** |
| two-point Acc@10 margin (TOST) | dois pontos de Acc@10 | δ_reg = 2 pp pre-registered; GLOSSARY §4 | **traces** |
| twenty fitted models | vinte modelos ajustados | GLOSSARY §4: 4 seeds × 5 folds = 20 | **traces** |
| four random initializations | quatro inicializações aleatórias | same | **traces** |
| five fixed folds | cinco partições fixas | same; one fixed partition across arms | **traces** |
| four initialization means | quatro médias por inicialização | GLOSSARY §4, n = 4 inferential unit | **traces** |
| six datasets / seis conjuntos | seis conjuntos de dados | results table, six rows per block | **traces** |
| five states of the United States | cinco estados | GLOSSARY §3 Gowalla: AL, AZ, FL, CA, TX | **traces** |

**Decimal convention:** PT uses the comma (5,3 / 9,4), EN the point (5.3 / 9.4). Correct for both
languages and the only cross-language difference in the numeric inventory. Values are identical to
the digit.

**Convention naming (N5):** the category range carries its metric (macro-F1), its selection rule
(joint-best), its resampling design (twenty fitted models, four initializations, five fixed folds)
and its inferential unit (paired tests on the four initialization means). The region verdict carries
its metric (Acc@10), its margin (two points) and its test (TOST). **This is better than the body
does in some places** and is the strongest thing in the pair.

**Never-cite sweep:** STAN v4-collapse values, the ReHDM v2 row, VOID fp16/bf16 cells. Zero hits in
either block; no baseline number appears in the pair at all.

**One number is absent that the body carries, and it is M-2 below**: the pair's sentence 6 asserts a
rise "in every state tested" with no count, where Chapter 6 says "20.2 to 22.0 percentage points
across the three states tested".

---

## 10 · Findings

### M-1 · MAJOR (persona 07) · The time-index hedge left the pair, and the loss table does not list it

**Anchor phrase:** "logo o gargalo era a representação, e não a arquitetura" / "so the bottleneck was
the representation, not the architecture".
**Location:** `src/0_main.tex:241` (PT, "logo o gargalo era a representação") and `:325-326` (EN),
today; rendered p.2 sentence 6 and p.3 sentence 6.

**Measured.** I extracted the pre-cut prose from `40ed8e7b^` and diffed device-by-device against the
current rendered text. The pre-cut pair read "era o gargalo **naquele estágio da pesquisa**" / "was
the bottleneck **at that stage of the research**". Both phrases are **absent** from the current pair
(string search, zero hits, both languages). I then swept for any substitute time marker in the pair:
`at that stage`, `of the time`, `for that configuration`, `at the time`, `naquele`, `na época`,
`daquele` — **zero hits in both blocks.** The only remaining time marker is the past tense of the
copula (`era` / `was`).

The same claim keeps its hedge at both body sites: `chapters/1_introduction.tex:114-116` reads "The
diagnosis followed: **at that stage of the research**, the input representation, not the sharing
architecture, was the bottleneck"; `chapters/6_conclusion.tex:53-55` reads "the input representation,
not the sharing scheme, was the binding constraint **at that stage of the research**".

The report's §4.2 table lists what was cut and where each cut claim survives. **The time-index hedge
is not among its rows.** (Two further arithmetic notes on that table, filed as m-6: its prose says
"Six claims left the pair" while the table carries **eight** content rows, and the `0_main.tex`
comment at :220 says "**Two** claims were cut as NOT on the floor". Three counts of one thing.)

**Conclusion.** WRITING_LAW §3 makes this non-negotiable: "CBIC's *MTL does not help* and CoUrb's
protocol are presented as conclusions *of the time, for that configuration*." The precedent report
put this hedge on its protect list as item 3, "four words doing the work of a paragraph". The past
tense does part of the job, and no reader will take "was the bottleneck" as a present-tense claim
about the state of the art. But the pair now asserts the diagnosis without the qualifier the law
requires and the body supplies, on the page most likely to be read alone, and the loss was not
declared. Persona 07's evidence-guard rule treats a lost honesty device as a claim change.

**Closes when** either the hedge returns to both blocks in parity (about four words per language,
the body's own wording is available verbatim), or the author rules that the past tense discharges
§3's requirement in an abstract and the ruling is recorded in the block comment beside the existing
`[NEEDS SIGN-OFF]` note.

---

### M-2 · MAJOR (personas 07 + 06) · "em todos os estados testados" is an unscoped universal, and the adjacent sentence supplies the wrong number

**Anchor phrase:** "o desempenho de categoria subiu acentuadamente em todos os estados testados" /
"category performance rose sharply in every state tested".
**Location:** `src/0_main.tex:240-241` (PT, the phrase spans the line break), `:325` (EN);
rendered p.2 and p.3 sentence 6.

**Measured.** Chapter 4 tested **three** states. `chapters/4_courb/intro.tex:27`: "in the states of
Florida, California, and Texas". `chapters/4_courb/results.tex:63`: same three, "whose statistics are
presented in Table 4.1". `chapters/4_courb/results.tex:92`: "In the three evaluated states ... in all
21 category-state combinations" (3 states × 7 categories = 21). Chapter 6 states the claim with its
count: `chapters/6_conclusion.tex:46-47`, "raised category macro-F1 by 20.2 to 22.0 percentage points
across the three states tested".

The pair never says three. Searched: `três` appears once in the Resumo, in "Três estudos respondem a
essa pergunta" (sentence 4, the three studies), and `three` appears in EN only in "Three studies
answer that question". **Two sentences after the unscoped universal, sentence 8 says "cinco estados
dos Estados Unidos" / "five states of the United States".** So a reader meets "every state tested"
and then, as the next quantity in the block, the number five.

**Conclusion.** WRITING_LAW §3: "Scope every universal: *at all six datasets* only right after the
six are enumerated ... bare *everywhere* never." "Every state tested" is the bare form with a
different noun, and the nearest available antecedent is the wrong study's five. The claim itself is
true and licensed; what is missing is its scope. This one is inherited rather than created by the cut
(`1_introduction.tex:114` has the same bare "at every state tested"), but the pair is the only place
where a wrong count stands two sentences away, and the frame's own conclusion chapter shows the
scoped form.

**Closes when** the pair says three, in parity, at that sentence ("nos três estados testados" / "in
the three states tested", two words per language, Chapter 6's own wording), or the author rules the
scope is discharged by the chapter and records it.

---

### M-3 · MAJOR (persona 06, tooling) · The parity gate fails on its own documented invocation and still exits 0

**Anchor phrase:** `PAIR_PDF=<path to a FRESH main.pdf> PT_PAGE=3 EN_PAGE=4 python3
_check_pair_parity.py`
**Location:** `src_utils/_round6/_check_pair_parity.py:4` (usage line) and `:35-36` (the defaults
`PT_PAGE 3`, `EN_PAGE 4`), today.

**Measured.** Three runs:

| invocation | result |
|---|---|
| documented usage against the current `dissertacao.pdf` (defaults 3/4) | **19 of 19 floor claims reported MISSING**; "PT 271 / EN 400" words; exit 0 |
| bare default (`build/main.pdf`, defaults 3/4) | same 19 failures, exit 0 |
| against the `main.pdf` I built fresh from `01915ba7` | same 19 failures, exit 0 |
| pages corrected to the pair's real location (2/3) | **0 failures, PT 310 / EN 271, 11/11 sentences** |

The cause is not staleness. It is that the pair moved from pp. 3-4 to pp. 2-3 when the approval sheet
became conditional at `7a91b720`, one commit before the state under review, and the checker's
defaults were not updated. Its docstring anticipates a stale PDF and promises a loud
`ZeroDivisionError`; what actually happens on a **fresh** PDF is worse than staleness — the checker
reads the Abstract page as PT and the List of Figures as EN (hence "EN 400 words", which is the List
of Figures), reports total parity failure, and **exits 0 anyway**, so a wrapper that checks the exit
status sees success.

**Conclusion.** This is the "gate that has never fired" bias in AGENT_GUARDRAILS §7, in its inverted
form: a gate that fires on everything is as uninformative as one that fires on nothing, and this one
does it while reporting exit 0. The report's own validation of this tool (§9) exercised only the
stale-PDF direction. The pair itself is fine; the instrument left behind to prove it is not.

**Closes when** the checker resolves the pair's pages by searching the text layer for the `Resumo`
and `Abstract` headings instead of taking page numbers from defaults, and exits nonzero on any floor
failure. Both are small changes and neither touches the document.

---

### M-4 · MINOR (persona 01, cold reader) · "the same trace" arrives before anything has established that a trace exists

**Anchor phrase:** "as duas tarefas leem o mesmo histórico" / "both tasks read the same trace".
**Location:** `src/0_main.tex:232` (PT), `:318` (EN); sentence 2 of both blocks.

**Measured, as a first-pass stumble.** Sentence 1 introduces a check-in as "visits by a user to a
point of interest". Sentence 2 then refers to "the same trace" with the definite article, and no
trace, history or sequence has been mentioned. I had to infer that a trace is a sequence of the
check-ins just defined. The inference is available but it is an inference, in the second sentence of
the document, and the PT `histórico` is vaguer than the EN `trace`.

**Conclusion.** The fix is one word, not a restructure: `o mesmo histórico de check-ins` / `the same
check-in history` would make sentence 1 do the work it is already positioned to do. This also closes
the registry gap in m-4 below, since `histórico de check-ins` is the composition the precedent report
proposed.

**Closes when** the phrase names what the trace is made of, in parity, or the author accepts the
inference as within a specialist reader's reach.

---

### m-1 · MINOR (persona 03) · The emphasis convention differs across a declared parity pair

**Anchor phrase:** `joint-best`.
**Location:** rendered p.2 (italic, `TeXGyreTermesX-Italic`) and p.3 (roman,
`TeXGyreTermesX-Regular`); source `0_main.tex:250` uses `\emph{joint-best}`, `:333` does not.

**Measured** by font-name inspection of rendered spans: Resumo italic spans are `check-ins`,
`embedding`, `joint-best` (28 italic characters); Abstract has one, `check-ins` (9 characters).

**Conclusion.** Precedent MINOR-1, unchanged by the cut. In a declared claim-parity pair the
emphasis convention is itself a claim about which words are foreign, and it cannot be true in both
languages at once. Defensible as it stands — italicizing a retained English term of art is BR
convention, and GLOSSARY §6 explicitly says the Resumo italicizes `seleção joint-best` — but then
the EN italic on `check-ins`, an English word in English prose, is the one that marks nothing.

**Closes when** the author rules one way: either drop the EN italic on `check-ins`, or accept the
asymmetry as the PT-convention consequence it is and record that in GLOSSARY §6 beside the
`joint-best` row.

---

### m-2 · MINOR (persona 15, readability) · One sentence carries the entire protocol at 58 words

**Anchor phrase:** "Sob validação cruzada com usuários disjuntos entre treino e teste" / "Under
user-disjoint cross-validation".
**Location:** `0_main.tex:245-250` (PT), `:329-333` (EN); sentence 9 of both.

**Measured.** Sentence-length distribution of the rendered blocks, not just the mean:

| | PT | EN |
|---|---|---|
| per-sentence words | 17, 39, 27, 6, 30, 34, 31, 17, **58**, 28, 23 | 15, 37, 26, 5, 24, 30, 24, 16, **48**, 25, 21 |
| sum / block total | 310 / 310 | 271 / 271 |
| mean | 28.2 | 24.6 |
| SD (population) | 12.9 | 10.9 |
| longest | 58 | 48 |
| shortest | 6 | 5 |

Per-sentence counts are on the **same** convention as the block totals, which is why each column
sums to its block total exactly. That reconciliation is not free: `_measure_abs.py` protects decimals
with U+2024 inside its sentence splitter, so counting a returned sentence directly splits `5.3` into
two tokens. Restoring the character before counting is what makes the two levels agree, and it is
the difference behind the fourth item of m-5.

Against the eight exemplar blocks I re-measured on the same instrument: means 19.5 to 37.5, longest
single sentence 29 to 69. **Both our means and both our longest sentences are inside the defended
envelope** — the round-5 pair was at 50.0/42.3 with a 111-word sentence, outside it on both axes.
Sentence 9 carries five protocol facts (split axis, twenty fits, four initializations, five fixed
folds, paired tests on four means) before it reaches its verdict, its range and its selection
convention.

**Conclusion.** This is the price of WRITING_LAW §3's convention requirement, and it buys real
honesty: the pair is the only one of the ten blocks I measured that names its statistical convention
at all. Distribution is healthy — a 6-word sentence sits beside a 58-word one, so burstiness is
preserved and the block does not read as machine-levelled. **Not a defect. Do not "fix" it by
levelling.** If the author wants the last twenty words, the report's advice is right: the protocol
clause is the only honest source, and it is also where the honesty is, so a cut there needs
persona 14.

**Readability scores** (persona 15's contract): readability 8, flow 9, clarity 8, conciseness 7,
consistency 9, **overall 8.0**. Flow is the strength: sentences 4 through 7 set up "three studies"
and then deliver one per sentence in parallel, and the parallelism is earned rather than templated.
Conciseness is the weak axis and it is a deliberate trade.

---

### m-3 · MINOR (persona 07) · Region superiority is a secondary result in Ch.5 and the pair does not say so

**Anchor phrase:** "did not cover next-region superiority, so the four next-region gains ... are
secondary results outside it".
**Location:** `chapters/5_mobiwac/05_setup.tex:76` today (was in `5_mobiwac.tex` before the split).

**Measured.** Ch.5 states that the written analysis plan assigned superiority to next-category and
non-inferiority to next-region only, so the four region gains sit outside the registered plan.
`stats_n20/RESULTS.md` §1b carries the same correction dated 2026-07-27: "The registered primary test
for every reg cell, CA/TX included, is TOST non-inferiority at δ_reg = 2 pp", with the region
superiority statistics "post-hoc and reported as such". The pair says `supera` / `outperforms` at
four region cells with no qualification. **Neither does Ch.1 (`:132`) nor Ch.6 (`:21`, `:75`).**

**Conclusion.** The pair mirrors the body faithfully, which is its job, so this is not a defect of
the cut. It is a frame-wide question: the scoping sentence exists in Ch.5 and nowhere in the frame or
the front matter. Out of this track's scope to rule on; raised because the front matter is where the
claim is loudest.

**Closes when** whoever owns the frame decides whether the secondary-result footing belongs in Ch.1
and Ch.6; the pair should follow whatever they decide, not lead it.

---

### m-4 · MINOR (persona 03) · `histórico` / `trace` is used in both blocks and registered in neither

**Anchor phrase:** "leem o mesmo histórico" / "read the same trace".
**Location:** `0_main.tex:232`, `:318`.

**Measured.** `grep -n -i "trace\|histórico\|trajet"` over `GLOSSARY.md` and `WRITING_LAW.md`:
**zero hits in both files.** GLOSSARY's own rule is fail-closed: a term not in the registry "may not
be used in dissertation prose". The body uses `traces` freely, so EN is consistent with the body;
PT `histórico` has no registered counterpart and no body precedent.

**Conclusion.** Precedent MINOR-2, unchanged by the cut, and it is the one registry gap in the pair
that names a **concept** rather than composing registered material. Same fix as M-4: `histórico de
check-ins` would both register cleanly and remove the cold reader's stumble.

**Closes when** §6 gains the row, or the phrase changes to registered material.

---

### m-5 · MINOR (persona 06, filed against the report) · Three stale or inconsistent statements in `15_resumo_abstract.md`

The document is fine in all three cases; the record is not, and this round's records will be read
by the next pass.

1. **`[VERIFY]` flag 2 is closed and reads as open.** It says "*Modelos ajustados* for *fitted
   models* is still not in `GLOSSARY.md` §6" and hands it to the glossary owner. It **is** in §6,
   row 19 ("n = 20 (fitted models) | vinte modelos ajustados (por configuração)"), added at
   `01915ba7` — the commit the brief measures the state at. The `0_main.tex` comment at :227-228
   carries the same stale claim ("still NOT in §6: proposed, pending the author's approval").
2. **The §4.2 loss table's count disagrees with its own prose and with the source comment.** Prose:
   "Six claims left the pair". Table: **eight** content rows. `0_main.tex:220`: "**Two** claims were
   cut as NOT on the floor". All three describe the same cut.
3. **Report §3's "105 pp to 104 pp" is a measurement of a tree that no longer exists.** Correctly
   labelled as isolated, and the brief already says the document is 108 pp for unrelated reasons. I
   rebuilt `01915ba7` in a clean tree and got **108 pages, exit 0**. Recorded so nobody reads 104 as
   current.
4. **The §2 per-sentence parity table is on a different basis from the same report's block totals.**
   Its columns sum to **PT 311 / EN 276**, against the block totals of **310 / 271** stated three
   paragraphs above. Two causes, both mechanical: `5.3` and `9.4` count as two tokens each when a
   returned sentence is counted without restoring the decimal separator the splitter protects, and
   `multi-task` counts as two where the line break falls inside it. The claim the table makes,
   eleven sentences in one-to-one correspondence, is **correct** and I re-derived it; only the word
   figures are on a second convention while the report states one.

**Closes when** the flag is struck, the count reconciled to one number, the page figure annotated
with its tree, and the per-sentence table recomputed on the block convention (or its basis named).

---

## 11 · What holds, and must not be touched (all six personas)

1. **The verdict-verb discipline, in both languages.** `supera` / `equipara-se estatisticamente` and
   `outperforms` / `statistically matches`, each bound to its test, with the TOST margin named with
   its metric and four-of-six scoped explicitly. Arizona not upgraded. This is the single best thing
   on these two pages and the reason the pair survives a hostile read.
2. **The next-place exclusion in sentence 2 of both blocks.** It is new, it is positive rather than
   negative in construction, and it precedes the results. The pair gained a floor claim in a pass
   whose brief was to cut. Protect the position as much as the wording: it works because it arrives
   before the reader has formed an expectation.
3. **The eleven-to-eleven sentence correspondence.** Any future edit to one block is an edit to both.
   The pair is now short enough that the correspondence is legible to a human reader, which is a
   maintenance property worth more than a few words.
4. **The three-study arc, sentences 4 through 7.** "Três estudos respondem a essa pergunta" followed
   by one sentence per study, negative result then diagnosis then resolution, in parallel
   construction. Persona 16's specificity audit: this is the strongest human signal in the document.
   An abstract that opens its own record with a null result is not something a language model
   produces from a thin prompt, and it is not something a polished-but-empty manuscript contains.
   The report cut the explicit words "a negative result, its diagnosis, and its resolution" and the
   three-beat survives structurally. It survives *well*; leave it.
5. **The headline number with its full convention attached** (metric, selection rule, resampling
   design, inferential unit). Better than any of the ten blocks I measured. The range 5,3-9,4 may
   never appear without its selection basis.
6. **Both blocks rendering, in both languages**, against the Viegas and Germano precedent of
   Abstract-only.
7. **The `(embedding)` gloss in the Resumo** and the follow-through on `representação` afterwards.
   Still the pair's own model for how a foreign term should arrive.
8. **The register mechanics.** Em-dash 0, contraction 0, banned tokens 0, codenames 0, phrasal-idiom
   0, in both languages. Adverbs: EN four `-ly` in 271 words (1.48%), never two in one sentence,
   every one load-bearing (`consistently`, `only`, `sharply`, `statistically`); PT three `-mente` in
   310 words (0.97%). Recorded as clean so nobody re-audits it.

**Persona 16, both channels LOW.** Screener risk LOW, with the standing caveat that hybrid-text
detector scores are windowing artifacts and no local detector was run: the prose is neither
flattened toward L2 simplicity nor inflated, sentence-length SD is 12.8 and 11.4 across eleven
sentences, and the vocabulary is domain-specific rather than elevated. Expert-suspicion risk LOW:
no outline shape, no bullet-itis, no wrap-up sentence, no significance trailer, no copula avoidance,
no negative parallelism stack, and the one structural repetition (three sentences opening on "O
primeiro / O segundo / O terceiro") is a promise being kept rather than a template. The concrete
research detail an expert looks for is present in quantity — a named architecture, a named
representation level, six datasets by source, a resampling design, a named test with its margin.
**No over-correction damage:** the cut removed words, not variance.

---

## 12 · Open questions only the author can answer

1. **M-1**: does the past tense discharge WRITING_LAW §3's time-indexing requirement in an abstract,
   or should "naquele estágio da pesquisa" / "at that stage of the research" return to both blocks?
   Four words per language, wording available verbatim from Ch.1 and Ch.6.
2. **M-2**: "nos três estados testados" / "in the three states tested", or is the scope discharged by
   Chapter 4?
3. **m-1**: drop the EN italic on `check-ins`, or record the asymmetry as intentional PT convention?
4. **m-4**: register `histórico`/`trace` in §6, or change to `histórico de check-ins` (which also
   closes M-4)?
5. **m-3**: does the region-superiority secondary footing belong in the frame? The pair should follow
   that decision, not anticipate it.
6. Can the `[NEEDS SIGN-OFF]` markers on this pair be cleared? Nothing in my measurements blocks
   them: there is no BLOCKER on these two pages, unlike the state the precedent report reviewed.

---

## 13 · Out-of-scope handoffs (one line each, not pursued)

- `check.sh` gained the new `chapters/*/*.tex` glob at `4e84cf7a`, but the front matter's prose lives
  in `0_main.tex`, which is a third path again — worth confirming each prose gate reaches it.
- The report's own tooling handoff stands, with one number moved: it hands off "the FINAL count is
  100", measured in its isolated tree. The worktree's `build/main_final.pdf` is **105 pages**
  (measured), which matches the brief's own figure. Whoever re-runs `sync_page_counts.py` should use
  105.
- `apx_b_static_scope.tex` renders (p.98, confirmed in the text layer), and nothing in the front
  matter points at it; the pair's sentence 6 is the claim it qualifies. Frame scope, not this track.

---

### Method and traceability

Text and images from the committed PDFs via pypdfium2; word and sentence counts from
`src_utils/_round6/_measure_abs.py`, unmodified, so they are directly comparable with the report
under review; italic detection by font-name inspection of rendered spans, not source markup;
hyphenation normalized (pdfium's U+FFFE at justified breaks) before every phrase match, which is the
error class the round-6 report documents against itself. Blank-page detection on rasterized pixels at
150 dpi (ink fraction and vertical ink span per page), not on the text layer. The exemplar envelope
re-measured from the five PDFs in `exemples/` rather than carried over: my figures reproduce this
round's to the word on all eight blocks. Pre-cut text extracted from `40ed8e7b^` and diffed
device-by-device against the rendered current text. Statistical claims traced to
`docs/studies/closing_data/v17_completion/stats_n20/RESULTS.md` **rev 4** — anchored on the revision
header, not the first matching line, because that file retains its superseded revisions inline. Body
coordinates re-resolved after the `4e84cf7a` chapter split by grepping both `chapters/*.tex` and
`chapters/*/*.tex`; every line number in this report was measured 2026-07-28 and every finding is
anchored on a phrase that will outlive it. No number was carried from a prior review without
re-measurement, and where a prior figure disagreed with mine (104/100 pages) I report both and name
the tree.

**Two things I could not verify, stated fail-closed:**

1. **Whether a Pangram-class detector would score these two pages as machine-written.** No detector
   is available in this environment; persona 16's screener verdict is a qualitative estimate from the
   L2-simplicity and uniformity angles, and hybrid-text scores are unstable by measurement. UNVERIFIED
   — blocked on detector access.
2. **Whether the author considers the past tense sufficient time-indexing** (M-1). That is a ruling,
   not a measurement, and it is his to make.
