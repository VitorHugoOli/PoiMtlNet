# Persona 04 · Concordance Checker — Cross-Chapter Consistency Report (v1)

> **Status: IN PROGRESS** (writing incrementally; a restart must not lose this).
> Scope: the FULL assembled document as a system — all six chapters + appendices + front
> matter. Checks: terminology/notation concordance, promises-vs-delivery, definitions,
> time-capsule integrity, duplication (L3), cross-references (L4), cross-chapter numbers,
> transitions/seams, Abstract↔Resumo parity.
> Read-only. I judge whether the document AGREES WITH ITSELF, not whether a claim is true
> (persona 07) or a number correct against external sources (persona 06).
> Builds under review: `src/main_defense.pdf` (87pp), `src/main_final.pdf` (83pp).
> Sources: `src/chapters/*.tex` + `src/0_main.tex`.

---

## VERDICT: **SEAMS NEED WORK**

The three chapters are individually sound and the arc (null -> diagnosis -> resolution) is
threaded cleanly: the research question is stated identically in Ch.1 and Ch.6, the region
result wording is locked across all six surfaces (four of six + non-inferior TOST at AL/AZ,
AZ never upgraded), the CoUrb audited numbers (20.2-22.0 pp; 15/21 + 1 tie) agree
chapter-to-appendix to the digit, and the L3 cross-chapter duplication sweep is clean (every
overlap is a sanctioned recap, errata quote, or the deliberately restated research question).
No blocker, and no result is misreported.

What holds the verdict back from "coherent" is a small cluster of **naming/number seams between
the frame and the chapter of record** — most seriously, the central artifact of the whole
document, **MTLnet, is named in the Abstract, Resumo, and Chapters 1, 2, 4, 5, and 6 as "the
model introduced in Chapter 3," yet its own Chapter 3 never uses the name once** — plus one
cross-chapter number that disagrees (Alabama joint category macro-F1: 64.51 in Ch.5's main
table vs 64.54 in Ch.6). These are exactly the seams a coletanea is most exposed to: parts
written apart that no longer point at each other correctly. All are fixable with small,
localized edits; none requires reopening a chapter's science.

## TOP 3 FINDINGS

1. **[MAJOR] "MTLnet" is introduced everywhere except its own chapter.** The name is used as
   the Ch.3 artifact by the Abstract (L251), Resumo (L192), Ch.1 (x4, incl. the Software
   contribution "The MTLnet framework (Chapter~\ref{ch:cbic})"), Ch.2 (lineage table + prose),
   Ch.4 (30+x), Ch.5 (L87), and Ch.6 (L25) — but Chapter 3 names it **zero** times (grep), in
   body and preface alike. A banca member directed to Ch.3 to meet "MTLnet" does not find it.
2. **[MAJOR] Cross-chapter number disagreement — Alabama joint category macro-F1.** Ch.5
   Table `tab:mobiwac:results` (L479) reports the AL joint value as **64.51**; Ch.6 §6.2 (L78)
   reports the same quantity as **64.54**. The dedicated value (56.82) agrees to the digit in
   both; only the joint value differs, and no erratum or "different run" note reconciles it.
3. **[MINOR] The de-duplicated "weekday/Saturday" image was not fully removed.** The L3-fix
   comment at Ch.2 L506 states the image "now appears only in Ch.1," but Ch.2 §2.2 (L162-163)
   still carries it ("visited on a weekday morning or a Saturday night, by a commuter or a
   tourist"), so the motif appears in both Ch.1 (L120) and Ch.2 after a fix that assumed it was
   gone from Ch.2.

---

## Working notes (raw, being refined into findings)

CONFIRMED (evidence captured):
- F-A [MAJOR]: "MTLnet" naming. Ch.3 (re-typeset CBIC) has ZERO "MTLnet"/"MTLNet" in prose
  (grep); published CBIC source only has it in a figure filename `mtlnet_poi.drawio.png` + repo
  `PoiMtlNet`, never as a prose model name. Ch.3 calls it "our MTL model"/"the proposed MTL
  framework". BUT Ch.4 L12+L82, Ch.5 L87, Ch.6 L25, Ch.2 lineage table + prose all name
  "MTLnet ... introduced in Chapter 3". Ch.4 L82 asserts "the published paper typesets the name
  as MTLNet, and this chapter preserves that form" — false vs the reproduced Ch.3.
- F-B [MAJOR]: 64.51 (Ch.5 tab:mobiwac:results, AL joint category) vs 64.54 (Ch.6 L77 capacity
  baseline para). Same dataset/model/task/metric. Hand exact value to persona 06.
- F-C [MINOR]: casing MTLnet (Ch.1/2/5/6) vs MTLNet (Ch.4, 30+x, explicitly sanctioned in-text).
  GLOSSARY canonical = MTLnet. Lineage table row = "MTLnet".
- F-D [MINOR]: "seed" (defined-term, GLOSSARY = one full 5-fold repetition) first USED Ch.1 L243
  "four seeds, five folds", DEFINED only in Ch.5 L349. Abstract correctly avoids it ("random
  initializations"). Ch.6 L88 also uses pre-definition.

PASSES (praise / what holds):
- L4 machine check: all \ref targets resolve to a real \label (built label/ref sets diffed).
  No dangling refs. Orphan (unreferenced) labels exist (eq:cbic:nbs, eq:courb:*, eq:mobiwac:loss,
  conclusion subsecs) — harmless.
- Region result wording locked across Ch.1/2/5/6/Abstract/Resumo: "four of six + non-inferior
  (TOST, 2pp) at AL/AZ"; AZ never upgraded; named set {Ist,FL,TX,CA} identical.
- CoUrb gain 20.2-22.0pp identical Ch.4/Ch.6/AppxB; "15/21 + 1 tie" identical Ch.4/AppxB.
- Category gain 5.3-9.4 (frame) consistent w/ Ch.5 +5.33..+9.35 under rounding.
- Abstract<->Resumo structural parity holds (same beats, same numbers, keywords mirror).
- Massive-STEPS spelling uniform; census tract / mahalle consistent; 520 Istanbul regions
  consistent within Ch.5.

---

## FINDINGS (ranked)

Locations are `file:line` in `src/` (chapters under `src/chapters/`, front matter in
`src/0_main.tex`). "Suggested direction" is never applied (read-only persona).

---

### 1 · [MAJOR] Terminology — "MTLnet" named as the Chapter 3 artifact everywhere but in Chapter 3

**Both sites quoted.** The frame and the two later paper chapters treat "MTLnet" as a proper
name introduced by Chapter 3:
- Abstract, `0_main.tex:251`: *"The first study built **MTLnet**, a joint model with a
  place-level embedding as input and hard parameter sharing"* (Resumo mirror, `0_main.tex:192`).
- Ch.1 `1_introduction.tex:102`: *"built the first joint model of this research, **MTLnet**"*;
  and the Software contribution, `1_introduction.tex:236`: *"The **MTLnet framework**
  (Chapter~\ref{ch:cbic})"*.
- Ch.2 lineage table, `2_fundamentals.tex:224`: row *"**MTLnet** ... First joint model ...
  \cite{silva2025mtlnet}"*, and §2.3 prose `2_fundamentals.tex:374`: *"Its own starting model,
  **MTLnet**, applies hard parameter sharing"*.
- Ch.4 preface, `4_courb.tex:12`: *"the baseline model **MTLNet**, introduced in
  Chapter~\ref{ch:cbic}"*, and `4_courb.tex:82`: *"the joint architecture introduced in
  Chapter~\ref{ch:cbic} \cite{silva2025mtlnet}; **the published paper typesets the name as
  MTLNet**, and this chapter preserves that form."*
- Ch.5 `5_mobiwac.tex:87`: *"Chapter~\ref{ch:cbic} introduced **MTLnet**, the first joint
  model"*; Ch.6 `6_conclusion.tex:25`: *"Chapter~\ref{ch:cbic} contributed **MTLnet**"*.

**The contradicting site:** Chapter 3, the chapter of record, calls the model *"our MTL model"*,
*"our proposed MTL model"*, *"the proposed MTL architecture"*, and *"the MTL framework"*
throughout (e.g. `3_cbic.tex:44,111,120,247,354`) and **never once writes "MTLnet"/"MTLNet"** in
prose, preface, caption, or heading (verified: `grep -c` = 0). This is faithful to the published
CBIC article, whose source contains the string only in a figure filename
(`imgs/mtlnet_poi.drawio.png`) and the repo URL (`PoiMtlNet`), never as a prose model name.

**Why it is a concordance defect, not a style nit.** (a) The reader is explicitly sent to Ch.3
to meet the name ("introduced in Chapter 3", "contributed MTLnet") and does not find it.
(b) Ch.4's assertion *"the published paper typesets the name as MTLNet"* is contradicted by the
reproduced Ch.3 in the same document — an internal self-contradiction about a verifiable fact.
(c) GLOSSARY §2 and WRITING_LAW §2 both make MTLnet the one canonical name and instruct that the
Ch.2 lineage table exists "so the names never blur"; the blur is precisely that the naming
chapter is silent.

**Suggested direction (author's call):** the cleanest fix is one sentence in the Ch.3 preface or
§3.3 naming the architecture ("we refer to this joint architecture as MTLnet"), which also
retro-justifies every "introduced in Chapter 3" downstream and makes Ch.4's "typesets the name
as MTLNet" true of the dissertation. Hand the underlying question — did the *published* CBIC
paper actually name it MTLNet — to persona 05 (Ch.4:82 makes a checkable claim about the source
of record). Note the casing split as part of the same fix (Finding 3).

---

### 2 · [MAJOR] Number disagreement across chapters — Alabama joint next-category macro-F1

- Ch.5 main results table `tab:mobiwac:results`, `5_mobiwac.tex:479`:
  `AL ... 56.82 (Dedicated) ... **64.51** (Joint)`.
- Ch.6 §6.2 capacity-baseline paragraph, `6_conclusion.tex:77-78`: *"its best configuration
  reaches 56.16 macro-F1, against 56.82 for the dedicated model at its own tuned width and
  **64.54** for the joint model."*

Same dataset (Alabama), same model (joint), same task (next category), same metric (macro-F1).
The dedicated value 56.82 matches to the digit across both; only the joint value differs (64.51
vs 64.54, a 0.03 gap). Ch.5 also states the largest category gain is +9.35 at Arizona and gives
AL dedicated 56.82 / joint 64.51 -> +7.69; Ch.6's 64.54 would make it +7.72. No erratum, seed
note, or convention difference reconciles the two.

**My finding is the DISAGREEMENT; the correct value is persona 06's call.** The Ch.5 table is
the source of record per README §Sources (RESULTS_BOARD -> the chapter table), so 64.51 is the
more likely correct value and Ch.6:78 the likely typo, but I do not adjudicate. Hand both exact
values + locations to persona 06.

**Suggested direction:** reconcile Ch.6:78 to the Ch.5 table value once 06 confirms.

---

### 3 · [MINOR] Notation — model-name casing: MTLnet vs MTLNet

Canonical form (GLOSSARY §2, lineage table row, WRITING_LAW §2) is **MTLnet** (lower-case n).
Used correctly in Ch.1 (x4), Ch.2 prose+table, Ch.5 (L80/87/93), Ch.6 (x2), and both Abstract
and Resumo. Chapter 4, however, uses **MTLNet** (capital N) 30+ times and explicitly sanctions
it at `4_courb.tex:82` (*"the published paper typesets the name as MTLNet, and this chapter
preserves that form"*), and Ch.2 quotes the CoUrb form once at `2_fundamentals.tex:224` inside
the ST-MTLNet row. So the document runs two casings for one artifact.

This is defensible IF the Ch.4:82 rationale is accepted (preserve the CoUrb paper's own
typesetting) — but then Ch.2 §2.2 (`2_fundamentals.tex:224`) writes "MTLnet" and "MTLNet" in
the *same paragraph* (the ST-MTLNet sentence: *"MTLnet, replaces the place-embedding input"* in
the What-it-added column while the model column reads MTLNet), which reads as an inconsistency
rather than a deliberate per-paper form. Decide once: either (a) MTLnet everywhere and Ch.4:82's
"preserves that form" sentence is dropped, or (b) keep MTLNet only inside the Ch.4 reproduction
and ensure no frame sentence mixes the two. Tied to Finding 1 — fix together.

---

### 4 · [MINOR] Duplication residue — the "weekday/Saturday" motif not fully de-duplicated

The gate L3-fix comment at `2_fundamentals.tex:506-509` states: *"The 'weekday lunch / Saturday
night out' image now appears **only in Ch.1**"* and reports that the §2.5 hinge was reworded to
avoid the duplication. But the image was removed from §2.5 only, not from §2.2: `2_fundamentals.tex:162-163`
still reads *"the same whether the place is visited on a weekday morning or a Saturday night, by
a commuter or a tourist"*, and Ch.1 `1_introduction.tex:120` carries *"cannot tell a weekday
lunch from a Saturday night out at the same place."* Two chapters, same motif — after a fix that
believed it had localized the motif to one.

Not a blocker (the §2.2 instance is a legitimate first technical statement of the static-vector
limitation, and Ch.1's is the signed-off mechanism beat), but the fix comment's own premise is
now false, which will mislead the next editor. **Suggested direction:** either accept the motif
in both places and correct the L506 comment to say so, or vary the §2.2 image (morning/night is
already a second variant — commuter/tourist — so the repetition is mild). Author's call on
whether it reads as a deliberate motif.

---

### 5 · [MINOR] Definition-before-use — "seed" used in the frame before it is defined in Ch.5

"seed" is a GLOSSARY defined term (one complete repetition of the five-fold experiment). It is
formally defined only in Ch.5, `5_mobiwac.tex:349` (*"A seed is one complete repetition of the
five-fold experiment ..."*), but it is USED earlier in reader order: Ch.1 §1.6/contributions
`1_introduction.tex:243` (*"twenty repetitions per configuration (four **seeds**, five folds)"*)
and Ch.6 `6_conclusion.tex:88` (*"twenty repetitions per configuration (four **seeds**, five
folds)"*). The Abstract/Resumo correctly avoid the bare term (they say "four random
initializations", per GLOSSARY's abstract rule), so the discipline is understood — it just is
not carried into the two frame body chapters that precede the definition.

Mild: "four seeds, five folds" is self-explaining in context. But Ch.2 §2.4 is where the
protocol vocabulary is set up, and it never glosses "seed" (it says "seeds" only in the
NORTH_STAR beat, not in the rendered §2.4 prose — the §2.4 text uses "repetitions"). **Suggested
direction:** add a one-clause gloss at the first frame use (Ch.1:243: "four seeds — four random
initializations of the five-fold experiment — ...") or in the Ch.2 §2.4 protocol paragraph, so
the term is defined at first use per WRITING_LAW §2. Persona 15/03 may also flag this; it is
listed here as a cross-chapter definition-ordering issue.

---

### 6 · [MINOR] Notation — lineage table mixes citation styles for same-status artifacts

`tab:fund:lineage` (`2_fundamentals.tex`) gives its Reference column as `\cite{...}` for DGI,
HGI, and **MTLnet** (`\cite{silva2025mtlnet}`), but as `Chapter~\ref{...}` for **ST-MTLNet**
(Ch.4), **Check2HGI** (Ch.5), and the **Joint model** (Ch.5). MTLnet and ST-MTLNet are both
"an artifact of a paper that is also a dissertation chapter," so a reader sees MTLnet pointed to
by a bibliography key while its sibling ST-MTLNet is pointed to by a chapter number, with no
stated reason for the split. (The MobiWac artifacts have no citeable published record — they are
"submitted, under review" — so Chapter-ref is correct for them; the inconsistency is specifically
MTLnet-via-cite vs ST-MTLNet-via-Chapter-ref, both published.)

**Suggested direction:** either give both published-chapter artifacts a Chapter-ref (and let the
bib key ride the chapter), or give both a `\cite`, so the column reads under one rule. Minor,
but the lineage table is the one place the law says the names/pointers "never blur." Tied to
Findings 1 and 3.

---

### 7 · [NIT] Gowalla vintage — a provenance nuance that does not surface in prose (handoff to 06)

Ch.6 §6.2 (`6_conclusion.tex:108`) and Ch.4 (`4_courb.tex:349`) both date Gowalla to
**2009-2010** ("collected in 2009 and 2010"; "between February 2009 and October 2010"), and
these two rendered statements AGREE. However, Ch.5 carries a hidden provenance comment
(`5_mobiwac.tex` datasets block) noting the actual figshare dump the ETL consumes spans
**2009-01-21 .. 2011-08-16** ("collected 2009 to 2011"), with cho2011 (Feb 2009-Oct 2010) cited
only as the LBSN reference. This never reaches rendered prose, so there is **no cross-chapter
prose disagreement** for me to flag as a concordance defect — but if Ch.5 ever states its own
2009-2011 range in prose, it will disagree with Ch.4/Ch.6's 2009-2010. Flagged to persona 06 as
a latent number issue, not an active concordance break. No action required from a concordance
standpoint today.

---

### 8 · [NIT] Task-name bridge ("next-POI prediction" = next category) present but late

WRITING_LAW/GLOSSARY require the per-paper task mapping (the older papers' "Next-POI Prediction"
= canonical **next category**) to be stated where the reader needs it. Ch.1 §1.1
(`1_introduction.tex:53`) defines next category as "the category of the next visited place" and
the arc paragraph explains the task-pair evolution, and Ch.2 §2.1 keeps the three targets
formally distinct — so the concept is bridged. BUT the *lexical* bridge is implicit: Ch.3 and
Ch.4 use "Next-POI Prediction" 17 and 16 times respectively (their published term), and no frame
sentence says in so many words "what these chapters call next-POI prediction is the next-category
task of this dissertation." Ch.4's preface and recap lean on "MTLNet ... POI Category
Classification and Next-POI Prediction" without the one-line mapping. A CS banca will follow it;
a careful reader may briefly wonder whether "Next-POI Prediction" in Ch.3/4 is the next-place
task Ch.1/Ch.2 explicitly exclude. **Suggested direction:** one bridging clause in the Ch.3
and/or Ch.4 preface ("the task this article calls next-POI prediction is the next-category task
of this dissertation; it predicts the next POI's category, not the exact next place"). Low
severity because §1.1/§2.1 carry the conceptual distinction; listed for completeness.

---

## Duplication report (L3)

Method: comment-stripped, LaTeX-stripped 12-gram sweep across all six chapters + three
appendices, coalesced into shared runs >= 14 words. **Result: clean — every cross-chapter
overlap is sanctioned.**

| Passage pair | Longest shared run | Verdict |
|---|---|---|
| Ch.4 <-> Appx B | "20.2 to 22.0 percentage points, considering the better of the two spatial encoders..." (19w); "Outdoors in Florida where the baseline mean exceeds the best variant by 0.02 pp" (16w) | **Sanctioned** — Appx B errata table quotes the corrected CoUrb values verbatim; identity is required (the erratum IS the quote). |
| Ch.1 <-> Ch.6 | "whether multi-task learning helps point-of-interest prediction for the next category and next region tasks and..." (18w) | **Sanctioned** — the research question, deliberately restated Intro->Conclusion (NORTH_STAR spine; the arc's bookend). |
| Ch.3 <-> Appx B | "about 2.3 times the cumulative 34.97 s of the individual single-task models" (15w) | **Sanctioned** — Appx B errata quotes the corrected Ch.3 wall-time sentence. |
| Ch.1 <-> Ch.3 | "Congresso Brasileiro de Inteligencia Computacional (CBIC 2025) DOI 10.21528/CBIC2025-1191324 with..." (14w) | **Sanctioned** — venue+DOI string, necessarily identical in the organization bullet and the Ch.3 preface. |

No unsanctioned near-duplicate prose across chapters. The mandated recaps are present exactly
where NORTH_STAR §3 places them and nowhere spurious: Ch.4 §4.2.5 "The MTLnet framework"
(`sec:courb:mtlnet-recap`) recaps the Ch.3 artifact; Ch.5 §5.2.1 "The MTLnet framework and the
representation diagnosis" (`sec:mobiwac:related-recap`) recaps BOTH the Ch.3 artifact and the
Ch.4 finding. Both are the sanctioned "The MTLnet framework" pattern; neither repeats the source
paper beyond the recap.

---

## Cross-reference lint (L4)

Machine check: every `\ref` target set diffed against the `\label` definition set across all
nine files. **All 90+ `\ref`/`\autoref` targets resolve to a defined label; no dangling
references.** Spot-checks of semantic targeting (the Viegas precedent shipped refs that compiled
but pointed at the wrong float):

| Cross-reference | From | Resolves to | Semantically correct? |
|---|---|---|---|
| "cascade architectures reviewed in Chapter~\ref{ch:mobiwac}" | Ch.6:151 | Ch.5 | **Yes** — Ch.5 §5.2.4 + §5.6 discuss the cascade (CSLSL/CatDM) at length (7 hits). |
| "Section~\ref{sec:intro:arc} explains why the final study replaced [category classification]" | Ch.1:63 | Ch.1 §1.2 | **Yes** — §1.2 L127 gives the "less natural fit under a per-visit representation" reason. |
| "Chapters~\ref{ch:cbic} and~\ref{ch:courb}" (FiLM conditions on task identity) | Ch.2:188 | Ch.3, Ch.4 | **Yes** — both use FiLM on task identity (Ch.3 §3.3.2, Ch.4 eq. film). |
| "Table~\ref{tab:mobiwac:results}" (the main result) | 8 sites | Ch.5 Table 3 | **Yes** — all point at the joint-vs-dedicated table. |
| Ch.2 lineage "ST-MTLNet ... Chapter~\ref{ch:courb}" / "Check2HGI ... Chapter~\ref{ch:mobiwac}" | Ch.2 table | Ch.4 / Ch.5 | **Yes.** |

Figure/table numbers in prose match the floats (all figures/tables are `\ref`'d, none by hard
number). One observation, not a defect: several equation labels are orphans (never `\ref`'d —
`eq:cbic:nbs`, `eq:courb:concat`, `eq:courb:film`, `eq:courb:time2vec`, `eq:mobiwac:loss`) and
several conclusion-subsection labels are unreferenced; harmless (numbered display equations need
no back-reference), listed only so persona 18/03 need not re-derive it.

---

## Numbers appearing in more than one chapter (concordance view; exact values -> persona 06)

| Quantity | Sites | Agree? |
|---|---|---|
| Research question wording | Ch.1:89-92, Ch.6:22-24, Abstract, Resumo | **Identical.** |
| Category gain 5.3-9.4 macro-F1 | Ch.1:132, Ch.2:530, Ch.5 (+5.33..+9.35), Ch.6:52, Abstract/Resumo | **Consistent** (frame rounds Ch.5's +5.33..+9.35). |
| Region result: 4 of 6 + non-inferior TOST(2pp) at AL/AZ; set {Ist,FL,TX,CA} | Ch.1:135, Ch.2:534, Ch.5, Ch.6:54, Abstract/Resumo | **Identical; AZ never upgraded** anywhere. |
| CoUrb category gain 20.2-22.0 pp | Ch.4:31,304,347, Ch.6:71, Appx B | **Identical.** |
| CoUrb win count 15/21 + 1 tie | Ch.4:295,347, Appx B | **Identical** (published 16/21 correctly relegated to the erratum). |
| Check2HGI vs place-level +28..+40 macro-F1 | Ch.5:33 (+27.63..+39.62), Ch.5 intro "+28 to +40" | Consistent within Ch.5 (rounding). |
| Capacity baseline AL: dedicated 56.82 | Ch.5:479, Ch.6:77 | **Agree.** |
| **Capacity baseline AL: JOINT category macro-F1** | **Ch.5:479 = 64.51; Ch.6:78 = 64.54** | **DISAGREE — Finding 2.** |
| Gowalla vintage 2009-2010 (rendered prose) | Ch.4:349, Ch.6:108 | **Agree** (Ch.5 2009-2011 is comment-only; Finding 7). |
| Parameter counts ~4.2M vs 1.1M (AL) | Ch.1:beat-guard region (scope), Ch.5:257, Ch.6:76 | **Consistent** (Ch.5 4.2M/1.1M; Ch.6 "about 4.2 million ... 0.6 million at its published width" — 0.6M is the dedicated *category* model, not the 1.1M *pair*; NOT a contradiction, but persona 06 should confirm the 0.6M vs 1.1M framing reads cleanly). |

---

## Transitions and seams (the arc)

- **Ch.1 -> Ch.2:** clean. Ch.1 §1.5 announces Ch.2 as "consolidates the background the three
  articles share"; Ch.2 delivers exactly that and closes §2.5 with the three-clause "pressing
  need" hinge pre-motivating Ch.3/4/5. The hinge's three clauses map 1:1 to the three chapters
  and to the four objectives.
- **Ch.2 -> Ch.3 -> Ch.4 -> Ch.5:** the null -> diagnosis -> resolution arc is stated
  consistently in every preface and in Ch.6. Time-capsule prefaces are present on all three
  paper chapters (Ch.3:10, Ch.4:10, Ch.5:20), each naming venue + status + what later chapters
  revise. Ch.4's mandated one-sentence floor ("isolates the representation effect ... does not
  revisit the MTL-versus-single-task question, which Chapter 5 reopens") is present verbatim
  (`4_courb.tex:12`).
- **Corrections cross-referenced both directions:** Ch.3 preface points forward ("Chapters 4
  and 5 revise that verdict"); Ch.4 and Ch.5 point back (recap subsections + "conclusion of the
  time"). Ch.5:141 explicitly reverses the Ch.3 negative-transfer observation on the new
  representation. The Nash-MTL of-the-time caution appears in Ch.3 preface, Ch.4:82, and Ch.6
  reading — consistent.
- **Status wording:** "published" for CBIC/CoUrb, "submitted, under review" for MobiWac —
  consistent across Abstract, Ch.1 bullets, all three prefaces, and Ch.6. No chapter reads
  MobiWac as published (the Ch.5 "published" hits are all about prior work, not its own status).
- **One seam to watch (Finding 1):** the MTLnet naming gap sits on the Ch.2->Ch.3 and
  Ch.3->Ch.4 seams — the reader crosses into Ch.3 expecting the named artifact and the name is
  absent, then crosses into Ch.4 which asserts the name was in the published paper.

---

## Abstract <-> Resumo structural parity (I own the structural half; values -> 06/07)

Structural parity **holds**. Same seven-beat structure, same paragraph order (LBSN/check-in ->
two tasks -> MTL + negative transfer -> three-study arc -> study 1 null -> study 2 diagnosis ->
study 3 resolution + protocol + headline number -> conditional-answer thesis). Numbers mirror:
"twenty repetitions (four random initializations, five folds)" = "vinte repetições (quatro
inicializações aleatórias, cinco partições)"; "5.3 to 9.4 macro-F1 points" = "5,3 a 9,4 pontos
de macro-F1"; "non-inferiority within a two-point margin (TOST)" = "não-inferioridade dentro de
uma margem de dois pontos (TOST)"; verbs bound to tests in both ("outperforms/matches" =
"supera/equipara-se"). Keywords mirror one-to-one (5 each, same order). Both correctly say
"random initializations"/"inicializações aleatórias" rather than the bare term "seed". Hand the
value-by-value and claim-strength halves to personas 06 and 07 as the contract directs; the
structural pair is sound.

---

## What holds / what reads well (do NOT touch)

1. **The arc is genuinely one investigation.** Research question identical at both bookends;
   every preface time-indexes its chapter; the correction trail is explicit and bidirectional.
   This is the hardest thing for a coletanea to achieve and it is achieved.
2. **The region result wording is locked to the digit and the test** across six surfaces
   (Abstract, Resumo, Ch.1, Ch.2, Ch.5, Ch.6): four of six, non-inferior TOST(2pp) at AL/AZ, AZ
   never upgraded, named set {Istanbul, FL, TX, CA} identical everywhere. Exemplary discipline.
3. **CoUrb's audited numbers are consistent chapter-to-appendix** (15/21 + 1 tie; 20.2-22.0 pp),
   with the superseded published values (16/21; 20-24 pp) correctly quarantined in Appendix B and
   never readable as current.
4. **L3 duplication is clean** — no padding, mandated recaps exactly where required.
5. **L4 cross-references all resolve and (on sample) target correctly** — no Viegas-style
   wrong-target refs.
6. **The lineage table exists and is consistent with chapter usage** for artifact identity and
   introduction chapter (the two residual issues — MTLnet-via-cite and the casing — are Findings
   1/3/6, narrow and fixable without touching the table's structure).

---

## Out-of-scope handoffs (one line each; not my verdict to make)

- **-> Persona 06 (numbers):** AL joint category 64.51 (Ch.5) vs 64.54 (Ch.6) — pick the correct
  value; Gowalla 2009-2010 (prose) vs 2009-2011 (Ch.5 provenance comment) latent issue; confirm
  the AL "0.6 million" (Ch.6) vs "1.1 million pair" (Ch.5) framing reads without contradiction.
- **-> Persona 05 (citations):** Ch.4:82 asserts "the published [CBIC] paper typesets the name
  as MTLNet" — verify against the source of record (my read of the CBIC source shows the string
  only in a figure filename + repo URL, not prose).
- **-> Persona 07 (claims/honesty):** Abstract/Resumo value-and-claim-strength parity (I
  confirmed structure only).
- **-> Persona 03/15 (style/readability):** "seed" defined-term first-use ordering (Finding 5);
  the weekday/Saturday motif density (Finding 4) if the author keeps it in both places.

---

## Open questions only the author can answer

1. **MTLnet naming (Finding 1):** do you want the name introduced in Ch.3 (one sentence), or do
   you intend Ch.3 to stay name-free (faithful to the published paper) and have the frame
   introduce the label? If the latter, Ch.4:82's "the published paper typesets the name as
   MTLNet" must be softened, because the reproduced Ch.3 does not show that name.
2. **Casing (Finding 3):** MTLnet everywhere, or MTLnet in the frame + MTLNet preserved only
   inside the Ch.4 reproduction? Either is defensible; the document currently does both without
   a stated rule at the one place they collide (Ch.2:224).
3. **Weekday/Saturday motif (Finding 4):** deliberate motif in both Ch.1 and Ch.2, or de-dupe to
   one? If kept, the L506 fix-comment should be corrected so the next editor is not misled.
