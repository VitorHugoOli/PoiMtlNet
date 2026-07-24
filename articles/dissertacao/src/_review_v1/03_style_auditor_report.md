# 03 · Style Auditor Report — G3 Style Gate (WRITING_LAW enforcement)

> Persona: `reviewers/03_style_auditor.md`. Scope: ALL SIX chapters
> (`src/chapters/1_introduction.tex` … `6_conclusion.tex`) + appendices
> (`apx_a`, `apx_b`, `apx_c`) + front matter (`src/0_main.tex`).
> Read-only. Enforces `WRITING_LAW.md` (+ inherited MobiWac GLOSSARY §7–§8 ban tables),
> `GLOSSARY.md` term registry. This gate's output is quantitative.
> Method: LaTeX comments stripped before prose sweeps; math/commands/`\cite`/`\ref`/tabular
> content stripped for word-count and -ly density; every hit carries `file:line`.
> All counts are a dated snapshot (2026-07-23); a re-audit replaces numbers, never rules.

---

## VERDICT (per-chapter + document)

**DOCUMENT: GATE FAIL.** Three independent triggers, none of which a mandated exemption
covers: (1) a hard-banned word in the Chapter 2 climax sentence (`unlocks`); (2) a
hard-ban cluster surviving in the re-typeset Chapter 3 prose (`crucial`/`Crucially` ×3,
`enhances` ×2, `surpassing`, `landscape` ×2, sentence-initial `Additionally`); (3) visible
unresolved editorial placeholders in rendered Chapter 3 body text
(`[$N_{\text{users}}$; VERIFY: recompute per ERRATA.md]`). The two re-typeset published
chapters also run over the -ly adverb band (Ch3 1.83%, Ch4 1.24%, band ≈0.8%).

| Chapter | Verdict | Why |
|---|---|---|
| Ch1 Introduction | **PASS** | Zero hard-ban, zero templates, -ly 0.64% (in band), healthy burstiness. |
| Ch2 Fundamentals | **FAIL** | `unlocks` (hard-ban, §2.5 hinge sentence); `co-equal` ×3 (idiom on the replacement list); metaphorical `carry` ×7 (budget ≤3). |
| Ch3 CBIC | **FAIL** | Hard-ban cluster (≥8 hits); participial significance tail; **visible `[VERIFY]` placeholders (BLOCKER)**; -ly 1.83%; `significant(ly)` ×12. |
| Ch4 CoUrb | **CONDITIONAL PASS** | No prose hard-ban words, no templates, no codenames; but -ly density 1.24% over band (MAJOR reservation). |
| Ch5 MobiWac | **PASS** | Hard triggers clean (`enhancements` L46 is the C2-mandated exemption; `frozen weights` glossed); "X, not Y" ×21 within its audited spec; -ly 0.46%. |
| Ch6 Conclusion | **PASS (1 MINOR)** | `frozen` L69 is a pathway, not weights → should be "fixed". -ly 0.48%. |
| Appendices + Front | **PASS (2 notes)** | AppA `substrate` is inside the rejected paper's verbatim title (keep); AppB banned words are all in the errata "published wording" column (documented corrections). Front-matter `[TITLE / banca / date]` placeholders are the documented open decisions — must be filled before the banca build. |

**Auto-fail triggers (document-wide): em-dash = 0, contractions = 0, booster-stacking
(≥2/sentence) = 0.** These are clean and confirmed on the raw files (comments/captions/tables
included). The failure is banned-vocabulary and one presentation blocker, not punctuation.

---

## TOP 3 FINDINGS

1. **[BLOCKER · Ch3 L235] Unresolved `[VERIFY: recompute per ERRATA.md]` placeholders render
   in the body.** The Gowalla-Florida dataset sentence ships three literal bracketed
   placeholders: `"This subset comprises a total of [$N_{\text{users}}$; VERIFY: recompute per
   ERRATA.md] users, [$N_{\text{poi}}$; VERIFY: ...] unique Points-of-Interest (POIs), and
   [$N_{\text{checkins}}$; VERIFY: ...] check-ins."` A reader of the built PDF sees the scaffolding.
   Presentation blocker (the number itself is persona 06's recompute; the *visible placeholder*
   is a style/presentation defect equivalent to an unresolved `\ref`). Must not reach the advisor.

2. **[MAJOR · Ch2 L532] Hard-banned `unlocks` in the frame's climax sentence.** `"It finally
   asks what a representation built for check-ins unlocks for a redesigned joint model"` — the
   §2.5 Relevance hinge, one of the most-read sentences of the fundamentals chapter, uses a
   verb on the leverage/harness/unlock ban list. Direction: "enables", "makes possible", or
   "what a redesigned joint model can do with".

3. **[MAJOR · Ch3, Ch4] The two re-typeset published chapters carry a distributional AI-tell the
   frame chapters do not: -ly adverb density 1.83% (Ch3) and 1.24% (Ch4) against a ≈0.8% band,
   plus a hard-ban cluster in Ch3 that the AppB conformance pass missed** (`crucial`/`Crucially`
   ×3, `enhances` ×2, `surpassing`, `landscape` ×2). AppB documents fixing `leverage`,
   `Moreover`, `underscore`, `Furthermore` in the same file, so the CBIC writing-rule pass was
   started but left incomplete. Ch3 also has the tightest sentence-length variance (CV 43%, 71%
   of sentences mid-length), consistent with the same signature.

---

## 1 · COUNTED WORD / TEMPLATE SWEEP (case-insensitive, whole text, comments stripped)

### 1a · Hard-ban words in the dissertation's OWN prose (exemptions already removed)

| Word (rule) | Loc | Quote (trimmed) | Note |
|---|---|---|---|
| `unlocks` (leverage/harness/unlock) | Ch2 L532 | "…built for check-ins **unlocks** for a redesigned joint model" | Frame §2.5 hinge. Not exempt. |
| `crucial` (pivotal/crucial/vital) | Ch3 L185 | "This is **crucial** for deployment on resource-constrained edge devices" | Reproduced CBIC prose. |
| `crucial` | Ch3 L244 | "Class-wise metrics are **crucial** because category frequencies are highly skewed" | |
| `Crucially` | Ch3 L327 | "**Crucially,** many of these observed differences in F1-scores are minor" | Sentence-initial. |
| `enhances` (enhance-family) | Ch3 L72 | "an inductive transfer mechanism that **enhances** generalization" | Describing MTL generally → "improves". |
| `enhances` | Ch3 L249 | "This model **enhances** a recurrent neural network with a Multi-Head Attention" | Describing MHA+PE → "augments/extends". |
| `surpassing` (surpasses-family) | Ch3 L290 | "significantly **surpassing** MHA+PE in these categories" | → "exceeding"/"outperforming". |
| `landscape` (realm/landscape) | Ch3 L290 | "indicate a competitive performance **landscape**" | → "…the results are competitive". |
| `landscape` | Ch3 L327 | "presents a more competitive **landscape** when compared against" | |
| `Additionally,` (sentence-initial) | Ch3 L139 | "…random shuffling method. **Additionally,** $\vec{s}$ is the global graph embedding" | AppB fixed the twin `Moreover`; this one survived. |

**Exempt hits verified and NOT counted above (evidence checked):**
- **AppB errata table (L108–123, 239–251):** `leverage` ×4, `Moreover` ×2, `Furthermore` ×2,
  `underscore` ×1, `surpass`-in-`standley` venue, `venue` ×4 — all sit in the LEFT "Published
  wording" column documenting corrections already applied, or in venue-erratum prose. Legitimate.
- **Ch5 L46 `enhancements`:** `"We propose two enhancements."` — the C2-mandated contributions
  lead-in, explicitly exempted by MobiWac GLOSSARY §7. Keep verbatim.
- **Front L59 `abnt-emphasize=bf`:** LaTeX package option, not prose.

### 1b · Auto-fail punctuation/format triggers

| Trigger | Count (prose) | Count (raw, incl. comments/tables/captions) |
|---|---|---|
| Em-dash `—` (U+2014) | **0** | **0** |
| LaTeX `---` ligature | 7 | all inside front-matter `[…]` placeholders (title/banca/date) |
| En-dash between words | — | **0** |
| Contractions | **0** | **0** |
| Booster stacking (≥2 intensifiers / sentence) | **0** | — |

### 1c · Template sweep

| Template | Hits | Verdict |
|---|---|---|
| Sentence-initial `Moreover`/`Furthermore`/`Additionally` | 1 (Ch3 L139 `Additionally`) | FAIL — see 1a |
| Participial significance tail | 1 (Ch3 L102: "…, **demonstrating the value of** attention-based models…") | MINOR — promote to a sentence with evidence or cut |
| `not only … but also` | 0 | pass |
| `plays a __ role` | 0 | pass |
| `in today's world` | 0 | pass |
| `Firstly/Secondly/Finally` scaffold | 0 | pass |
| `a wide array of` | 0 | pass |
| `let us / let's examine` (reader-facing meta) | 0 (2 false positives: verb "lets" in Ch5 L239, Front L243) | pass |
| Literal `Read this as:` tag | 0 (Ch5 L586 "We read this as a defense…" is ordinary prose) | pass |

---

## 2 · DENSITY METRICS (numbers)

### 2a · -ly adverb density (band ≈0.8% max; never two -ly in one sentence)

| Chapter | words | -ly | density | status |
|---|---:|---:|---:|---|
| Ch1 Introduction | 1,714 | 11 | **0.64%** | in band |
| Ch2 Fundamentals | 3,611 | 18 | **0.50%** | in band |
| Ch3 CBIC | 4,199 | 77 | **1.83%** | **OVER — 2.3× band** |
| Ch4 CoUrb | 4,268 | 53 | **1.24%** | **OVER — 1.5× band** |
| Ch5 MobiWac | 5,661 | 26 | **0.46%** | in band |
| Ch6 Conclusion | 1,238 | 6 | **0.48%** | in band |
| AppB errata | 1,183 | 9 | 0.76% | in band |

The split is clean: the four freshly-written frame/paper-native chapters (Ch1, Ch2, Ch5, Ch6)
are all comfortably in band; the two re-typeset published chapters (Ch3, Ch4) are the only ones
over. Ch3's top -ly contributors: `simultaneously` ×7, `jointly` ×5, `finally` ×4,
`significantly` ×4, `effectively`/`frequently`/`consistently`/`highly` ×3 each. Many are
functional (jointly, statistically, simultaneously) and legal; the decorative ones
(effectively, frequently, consistently, largely) are the trim targets. No sentence contains two
-ly adverbs (checked).

### 2b · Intensifier / booster counts (≤1 per claim; "significant" only with a test)

| Chapter | n | breakdown |
|---|---:|---|
| Ch3 CBIC | 11 | significantly ×4, highly ×3, very ×1, substantially ×1, strongly ×1, notably ×1 |
| Ch5 MobiWac | 7 | far ×3, entirely ×3, highly ×1 |
| Ch2 Fundamentals | 4 | far ×2, entirely ×1, widely ×1 |
| Ch1 / Ch4 / Front | 1 each | sharply / very / sharply |
| Ch6 + appendices | 0 | — |

No sentence stacks ≥2 boosters. Ch3's `significant(ly)` appears **12×** (L42, 44, 50, 67, 186,
214, 252, 290, 327, 330, 349, 358) — a density flag on its own; several attach to comparison
claims (L252 "significantly outperform HMRM", L290 "significantly surpassing MHA+PE") whose
test-binding is persona 07's call, but the sheer repetition is a style concern here. `far` in
Ch2/Ch5 is used non-decoratively ("far from random", "far below the check-in-level
representation's 75.15") and reads acceptably.

### 2c · "X, not Y" negative-parallelism (honesty device AND a known LLM fingerprint)

Total **27** (`, not <lowercase>`), plus **28** `rather than`. Distribution: **Ch5 21**, Ch1 1,
Ch2 1, Ch4 1, Ch6 1, AppB 1, Front 1. The Ch5 concentration matches the MobiWac paper's own
audited ~21 count and is within its documented spec. **Mandated verbatim keeps present (4 of 5):**
"a neighborhood, not a radio cell" (L213); "not a reproduction of the complete published system"
(L353); "not a claim that we outperform the cascade" (L587); "motivation, not a measured service
result" (L612); plus "a finding, not a hypothesis" (L568). **The keep "a match, not a gain" was
not found verbatim** — Ch5 phrases the Arizona/Alabama case as "matches … within two points" /
"remains non-inferior". The honesty content (non-inferiority, not a win) appears preserved, but
confirming that is persona 07's binding check — see out-of-scope handoffs. The frame's own uses
(Ch1 L115, Ch2 L287, Ch6 L42, Front L257) all scope the central thesis ("the input
representation, not the sharing architecture") — load-bearing, keep.

### 2d · Semicolon braids (a 2-semicolon prose sentence is two sentences; CI/stat notation exempt)

Sentence-level, non-numeric: **0 true prose braids.** The apparent hits are all exempt —
enumerated list colons `(i)…(ii)…(iii)` (Ch3 L74, L85), statistical-interval listings
(Ch5 L301, L554, L559), and one serial-semicolon list with internal commas (AppC L54: "checked
citations…; a separate style gate; and the author's own reading") which is a legitimate
three-item construction. No rewrite required.

### 2e · Metaphorical `carry/carries` budget (≤3 per chapter)

| Chapter | metaphorical carries | status |
|---|---:|---|
| Ch2 Fundamentals | **7** | **OVER** (L29, L69, L133, L190, L200, L227, L503) |
| Ch4 CoUrb | 3 metaphor (L89, L154, L291) + 2 "carried out" (L33, L169, phrasal verb = performed, exempt) | at budget |
| Ch5 MobiWac | 3 | at budget |
| Ch1 / AppA / AppB | 1 / 1 / 3 | in budget |

Ch2 is over: "each visit **carries** its own vector" recurs as the key-idea phrasing (L200,
L227) and mirrors the GLOSSARY gloss, so keep one or two; the others (L29 "carry geographic and
temporal detail", L190 "context these encoders carry", L503 "carries nothing") can become
holds / encodes / has.

---

## 3 · IDIOM SWEEP (phrasal-metaphor idioms at zero; register test on suspects)

| Idiom | Hits | Verdict |
|---|---|---|
| `co-equal` | **3** (Ch2 L91, L335, L498) | **FAIL** — on the MobiWac §8 replacement list ("rephrase around *equal standing*"; bare co-equal collides with the disclosed 0.75/0.25 loss weights). Repeated in the frame. |
| `sits` (on/above) | 1 (AppB L68 "the label … **sits on** the Dataset subsection") | MINOR — law mandates "lies", one verb everywhere; errata-appendix prose describing a fixed bug. |
| edges past / buys / ships / lands / folds in / clears by / comes out ahead / line up / settle on / staging / recent trail | **0** | pass — the MobiWac idiom cleanup held across the re-typeset chapters. |

No money/motion phrasal metaphors survive in the frame or Ch5. This sweep is otherwise clean —
the earlier campaign's idiom discipline transferred well.

---

## 4 · TERM-REGISTRY LINT (L2 — one name per concept; codenames zero; fail-closed on unlisted)

### 4a · Repo codenames in prose (must be ZERO)

| Codename | Hits | Verdict |
|---|---|---|
| `frozen` | 3 | Ch5 L355 "with **frozen weights** (no fine-tuning)" + L370 "With frozen weights" = the sanctioned *frozen-weights* exemption, glossed at first use — **legal**. Ch6 L69 "with the region pathway **frozen**" is a pathway, not weights → **MINOR**, should be "fixed". |
| `substrate` | 1 | AppA L27 — inside the **verbatim italic title** of the rejected BRACIS submission (*"Substrate Carries, Architecture Pays…"*). Title of record; keep as-is. Not a prose concept-use. |
| B9 / v11–v17 / champion-G / H3-alt / dk_ovl / log_T / engine / board / recipe / C2HGI / dualtower | **0** | clean |

Region-transition prior is written out (never `log_T`); "the joint model" is used throughout
(never `mtlnet_crossattn_dualtower`). The codename discipline is essentially intact — one
`frozen` misuse in Ch6.

### 4b · Banned task/place/visit synonyms

| Banned term | Hits | Verdict |
|---|---|---|
| `activity` (for the task) | 9 | **All exempt** — every instance describes another system's own term (MCARNN, iMTL, DRRGNN, CSLSL cascade). Law: "activity appears only when describing other papers." OK. |
| `area` (for region) | 1 | Ch1 L43 "planning resources **by area**" — generic geographic sense in a downstream-uses list, not the next-region task. NIT: "by region"/"geographically" would remove the collision. |
| `venue` | 5 | **All exempt** — publication-venue sense (Ch1 L220 "identifying its venue"; AppB "venue corrected to…"), never place/POI. OK. |
| `event` | 1 | Ch1 L209 "presented the paper at the **event**" = the conference, not a check-in. Exempt. |
| `cell` | 8 | **All legitimate senses** — "grid cell" (mandated full form), "radio cell" (the contrast), "recurrent cell" (LSTM), errata "cell values" (table cells). OK. |
| `zone` | 1 | Ch3 L120 "leisure **zones** are often near residential districts" — reproduced CBIC example of spatial co-location, not the region task. NIT. |
| `run` (for seed) | 12 | Mostly the verb "run" (execute a model) or the idiom "runs to" — legal. Two noun uses read loosely: Ch5 L413 "two independent **runs**" (two model executions, not seeds) and Ch6 L79 "a partial California **run**, fifteen of twenty repetitions" (the "repetitions" gloss saves it). NIT: prefer "training runs"/"the California experiment". |

### 4c · The "Next-POI" label — registry-bridge gap [MAJOR]

The task label **"Next-POI Prediction"** appears **35×** in reproduced Chapters 3 and 4 prose
(defined in Ch3 L35 as "Predicting the **category** of the next POI"). The frame chapters use
only the canonical "next category" in prose ("next-POI" appears in Ch2/Ch5 only inside LaTeX
comments, never rendered). GLOSSARY §1–§2 require the frame to state the per-paper bridge once
(CBIC "next-POI category prediction" = canonical **next-category prediction**). **That bridge is
absent** — no sentence in Ch2, Ch1, or the Ch3/Ch4 prefaces tells the reader that CBIC's
"Next-POI Prediction" is this dissertation's "next-category" task. The collision is real: a
reader who has just been told (Ch1, Ch2, Ch5) that "the exact **next place** is not predicted"
then meets a chapter whose central task is called "**Next-POI** Prediction". The label reads as
the very task the dissertation disclaims. Direction: add one bridging sentence to the Ch3 (and
Ch4) preface — "The task this article calls *Next-POI Prediction* is next-**category**
prediction in the dissertation's terminology (Chapter 2); it is not the exact-next-place task."
Registry violation, fail-closed.

---

## 5 · DISTRIBUTIONAL PASS (variance compression is the deepest tell)

**Sentence-length burstiness** (coefficient of variation of sentence word-counts; higher = more
human variance):

| Chapter | n sents | mean | std | CV% | short<12 / mid / long>28 |
|---|---:|---:|---:|---:|---|
| Ch1 | 65 | 26.4 | 12.9 | **49%** | 14 / 43 / 43 |
| Ch2 | 157 | 23.0 | 12.4 | **54%** | 22 / 52 / 27 |
| Ch3 | 207 | 20.3 | 8.7 | **43%** | 15 / 71 / 14 |
| Ch4 | 175 | 24.4 | 10.8 | 44% | 9 / 59 / 32 |
| Ch5 | 237 | 23.8 | 12.6 | 53% | 16 / 50 / 34 |
| Ch6 | 49 | 25.1 | 14.2 | **57%** | 18 / 47 / 35 |

Ch3 shows the most compressed distribution (CV 43%, 71% of sentences mid-length, only 14%
long) — the same chapter that fails 1a and 2a. This is the mildest of the three signals (43% is
low, not pathological), but it points the same direction: the CBIC chapter reads as the most
uniform. The frame chapters (Ch2, Ch6 especially) show good burstiness.

**Section openers (frame):** varied — Ch2's five sections open five different ways (a definition,
a consequence, a definition, a trust claim, a "these are one argument" synthesis); Ch1 and Ch6
likewise. No single opener template. The apparent duplicates in the Ch5 extraction are
section→subsection pairs sharing the first content sentence (a `\section` immediately followed by
its first `\subsection`), which is normal.

**Section closers (frame):** none ends by restating itself. Ch2 §2.1 closes on a forward hook
("the setting the following sections build the tools for"), §2.4 on the verb-binding rule, §2.5
on "these three questions in turn". Ch6 sections close on distinct concrete statements. Clean.

**Read-aloud spot check (one page/chapter):** the frame prose carries the author's voice
(concessive clauses, mid-paragraph result openers, varied length). No monotone smoothing
detected in Ch1/Ch2/Ch5/Ch6. Ch3 is the flattest to the ear, consistent with the metrics.

---

## 6 · STRUCTURE / PRESENTATION SPOT-CHECKS (Viegas-derived §5)

| Check | Result |
|---|---|
| Table captions **above** the tabular | **PASS** — every table (Ch2 lineage, Ch3 ×3, Ch4 ×2, Ch5 ×2) places `\caption` before `\begin{tabular}`. |
| Figure captions **below** the graphic | **PASS** — every figure places `\caption` after `\includegraphics` (Ch3, Ch4, Ch5 ×4). |
| Table lead-takeaway sentence, no literal "Read this as:" | **PASS** — the literal tag is at zero; Ch4/Ch5 tables are introduced by prose lead sentences (per the Ch4 ledger B4). |
| Metrics defined defensively at first use | **PASS** — macro-F1 (Ch2 L431, Ch5), Acc@10 (Ch2 L411), TOST/OOD-discounted defined; boundary behavior stated. |
| Model-lineage table present (DGI→…→joint model) | **PASS** — Table `tab:fund:lineage` (Ch2 L207), names taken from GLOSSARY. |
| "next category / region / place" kept distinct; disclaimer once, early | **PASS** — "the exact next place is not predicted" stated in Ch1 L172 and Ch2 L56, reinforced Ch5 L205, Ch6 L115. Consistent. |
| Acronyms expanded at first use | **PASS** — LBSN (Ch2 L27), POI (Ch1 L35), MTL (Ch1 L70), DGI/HGI/FiLM (Ch2 L150/154/187), TOST (Ch1 L132), Acc@10 (Ch2 L411) all expanded. STL and GRU (listed in the GLOSSARY §5 acronym set) are **used 0×** in the whole document — no unexpanded-acronym risk; drop them from the List of Abbreviations if they stay unused. |
| `this paper` / "is organized as" leftover in re-typeset chapters | **PASS on "this paper"** (zero); the roadmap sentences correctly say "this chapter" (Ch3 L53, Ch4 L44). Prefaces correctly say "This chapter reproduces the article…". |
| `Dataset N` prose anti-pattern | **PASS** — zero; datasets named (Florida, California, Istanbul…). |
| Unresolved `\ref`/`\cite` (`??`) | none literal in source. |
| **Visible editorial placeholders in body** | **FAIL** — Ch3 L235 `[VERIFY: recompute per ERRATA.md]` ×3 (see Top Finding 1); front-matter `[TITLE / Banca member / defense date]` placeholders (0_main L112–121, 161, 171, 233) are documented open decisions but MUST be resolved before the banca build. |

---

## 7 · RANKED FINDINGS (severity · location · rule · direction)

1. **BLOCKER — Ch3 L235.** Rendered `[VERIFY: recompute per ERRATA.md]` placeholders in the
   dataset sentence. Rule: WRITING_LAW §5 anti-patterns (no leftover scaffolding in body).
   Direction: fill the three counts (persona 06 recompute) or, if not yet available, remove the
   sentence rather than ship the brackets. Blocks the advisor handoff.
2. **MAJOR — Ch2 L532.** `unlocks` (hard-ban, leverage/harness/unlock family) in the §2.5
   climax. Direction: "enables" / "makes possible".
3. **MAJOR — Ch3 (8+ hits).** `crucial` L185/L244, `Crucially` L327, `enhances` L72/L249,
   `surpassing` L290, `landscape` L290/L327, `Additionally,` L139. Rule: AI-tell ban table.
   The AppB pass fixed the sibling words in this file; finish it. Direction: important/central
   (sparingly) or state the fact; improves/augments; exceeds/outperforms; "the results are
   competitive"; delete the sentence-initial connective.
4. **MAJOR — Ch3 1.83% / Ch4 1.24% -ly density.** Rule: §4 density (band ≈0.8%). Direction: cut
   decorative manner adverbs (effectively, frequently, consistently, largely, particularly) in
   the two re-typeset chapters; keep the functional ones (jointly, statistically, simultaneously).
5. **MAJOR — Ch2/Ch3/Ch4 registry bridge.** "Next-POI Prediction" (35×) is never bridged to
   canonical "next-category" and collides with the disclaimed "next place". Direction: one
   bridging sentence in the Ch3 and Ch4 prefaces (see §4c).
6. **MAJOR — Ch2 `co-equal` ×3 (L91, L335, L498).** Rule: MobiWac §8 replacement list.
   Direction: rephrase around "equal standing" / "neither target is subordinate to the other".
7. **MINOR — Ch2 metaphorical `carry` ×7 (budget ≤3).** Direction: keep "each visit carries its
   own vector" (key phrase); convert L29/L190/L503 to holds/encodes/has.
8. **MINOR — Ch3 `significant(ly)` ×12.** Density flag (binding to a test is persona 07's).
   Direction: let the ± spread carry the size; reserve "significant" for test-backed claims.
9. **MINOR — Ch3 L102 participial significance tail** ("…, demonstrating the value of…").
   Direction: promote to a sentence with the evidence, or cut.
10. **MINOR — Ch6 L69 `frozen`** ("region pathway frozen"). Rule: "frozen"→"fixed" except
    glossed weights. Direction: "with the region pathway fixed".
11. **MINOR — AppB L68 `sits on`.** Rule: "lies", one verb everywhere. Direction: "lies on".
12. **NIT — Ch1 L43 `by area`; Ch3 L120 `zones`; Ch5 L413 / Ch6 L79 noun `run(s)`.** Prefer
    "by region"/"geographically"; leave the reproduced `zones` or gloss; "training runs".
13. **NIT — Front matter placeholders.** `[TITLE]`, `[Banca member 1/2]`, `[defense date]` —
    documented open decisions; fill before the banca build (0_main L112–121, 161, 171, 233).

---

## 8 · PROPOSED LAW UPDATES (for author approval — never self-applied)

1. **Add a coletânea-specific rule: reproduced-chapter task labels must carry a one-line
   terminology bridge to the registry at first use in the chapter preface.** The current law
   (GLOSSARY §2) assumes the bridge is stated "once" but does not fix *where*; §4c shows the
   result is that no chapter states it at all. Proposed text: "Where a re-typeset article keeps
   its published task name (e.g. CBIC 'Next-POI Prediction'), its preface states the mapping to
   the canonical registry term once, before the term recurs."
2. **Add `unlock(s)` explicitly to the visible ban table.** It is a leverage/harness-family verb
   but is not spelled out; it slipped into the Ch2 climax. One line in the §7 table.
3. **Promote the -ly density check to a per-chapter reported metric with a hard ceiling for
   re-typeset chapters.** The band is stated but only the MobiWac chapter was ever audited; Ch3
   and Ch4 crossed it undetected. Proposed: "-ly density is reported per chapter at every G3
   pass; >1.0% blocks the gate for any chapter."
4. **Clarify the `frozen` exemption wording** to "frozen *weights only*, glossed; a frozen
   *pathway/branch/layer* is 'fixed'." Ch6 L69 shows the current phrasing is ambiguous.

---

## 9 · WHAT HOLDS / READS WELL (do not touch — avoid pushing the text toward sterility)

- **Punctuation and register discipline is genuinely clean:** em-dash 0, contractions 0, no
  booster stacking, no phrasal-metaphor idioms (edges past / buys / ships / lands / folds in all
  at zero across the re-typeset chapters). The earlier campaign's idiom cleanup transferred.
- **Ch5 (MobiWac) is the strongest chapter stylistically** — in-band -ly (0.46%), the "X, not Y"
  device used as a scoped honesty tool with its mandated keeps intact, the region wording bound
  to its tests, and healthy burstiness (CV 53%). Leave its prose alone.
- **Load-bearing CS vocabulary is used correctly and must NOT be banned:** `framework` (13×,
  all "MTL framework"/"unified framework"), `robust` (5× in Ch3, all technical — robust
  evaluation, robust feature representation), `baseline`, `novel` (1×), `comprehensive` (1×) are
  legitimate. The offense in this document is decoration/stacking and the specific banned tokens,
  not these working words. Do not sterilize.
- **The honesty scaffolding is present and precise:** the "we do not predict the exact next
  place" disclaimer is stated early and consistently; the CBIC/CoUrb conclusions are explicitly
  time-indexed ("the conclusions of the time, for the configuration studied here", Ch3 preface);
  the Nash-MTL benefit is not amplified. This is exactly the register the law asks for.
- **Structure/presentation is compliant:** caption placement correct throughout, lineage table
  present, metrics defined defensively, no "Dataset N", no "this paper" leftovers, roadmap uses
  "this chapter". The Viegas patterns landed.

---

## OUT-OF-SCOPE HANDOFFS (one line each; not this gate's call)

- **Persona 06 (numbers):** Ch3 L235 three dataset counts are unfilled — recompute and fill (I
  flag only the *visible placeholder*, not the value).
- **Persona 07 (claims/honesty):** the mandated keep "a match, not a gain" is absent verbatim;
  Ch5 rephrases the AL/AZ case as "matches … within two points" — confirm the
  non-inferiority-not-superiority content is intact. Also Ch3 L252/L290 "significantly
  outperform/surpassing" — confirm test-binding.
- **Persona 04 (concordance):** STL and GRU are in the GLOSSARY acronym set but used 0× —
  concordance should decide whether they stay in the List of Abbreviations.

---
_End of report. Verdict: **GATE FAIL** (document). Re-run this gate after the fixes land — the
banned-word tables rot, and words creep back through AI-assisted rewrites._
