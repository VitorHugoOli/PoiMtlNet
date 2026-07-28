# 17 · Style gate (G3) + readability + AI-credibility — round 6

**Personas 03 (style auditor, gate G3) → 15 (readability editor) → 16 (AI-credibility), in that order.**
Written 2026-07-28. Read-only pass; no document file was edited.

**Verdict.** **G3 (persona 03): GATE PASS.** Every fail condition the persona defines is at zero in
the round-6 prose: banned words 0 in new frame prose (1 hit, `stands as`, is discussed below and is
not on the law's list as a template), em-dash 0 in prose, contractions 0, British spellings 0, repo
codenames 0 new. Personas 15 and 16 return **PASS WITH FINDINGS**: nothing here endangers the gate,
but four things a careful examiner will notice are named below, and one of them (STY-01) is a
fail-closed registry violation the law says blocks the passage until the author registers the terms.

**Read AIC-01 first.** It is the only finding where this round crossed an instruction that was already
written down: the v1 run of persona 16 told the author to freeze the negative-parallelism count, and
this round raised `rather than` from 67 to 79 document-wide. A guard that lives only in a previous
round's review report is a guard nobody is checking.

---

## 0 · What I measured, and on what

Built all three targets myself from a clean copy of `src/` at `4e84cf7a`, in an isolated tree:

| build | pages | tex_errors | overfull hbox | overfull vbox | Float too large | undef ref | undef cite |
|---|---|---|---|---|---|---|---|
| `main.pdf` (defense) | **108** | 0 | 0 | 0 | 0 | 0 | 0 |
| `main_final.pdf` (final) | **105** | 0 | 0 | 0 | 0 | 0 | 0 |
| `main_ppgc.pdf` (ppgc) | **109** | 0 | 0 | 0 | 0 | 0 | 0 |

Matches the state I was given. Per the guardrails' "trusting the tolerant tool" rule I took
`tex_errors=0` from the logs rather than inferring correctness from a PDF existing.

**The mid-session split (`4e84cf7a`) — I verified the render-neutrality claim rather than accepting
it, and one half of the claim as handed to me is false.** I was told the three builds are
"byte-identical before and after … same SHA". Measured on my own pre-split and post-split builds:

| build | PDF file size | PDF sha256 (16) pre → post | text-layer sha256 (16) pre → post |
|---|---|---|---|
| defense | 1,339,651 = 1,339,651 | `0d82fe5c…` → `cea2ae65…` **differ** | `bbc0d8b4…` → `bbc0d8b4…` **match** |
| final | 1,333,480 = 1,333,480 | `36f2860b…` → `4355635d…` **differ** | `f1969684…` → `f1969684…` **match** |
| ppgc | 1,340,420 = 1,340,420 | `b29755a8…` → `e911b305…` **differ** | `c6eb1e84…` → `c6eb1e84…` **match** |

**The PDFs are not byte-identical and their SHAs do not match** (pdflatex writes a creation
timestamp and a document ID, so two builds of identical source never hash equal). What *is* identical
is the extracted text layer, byte for byte, in all three targets: 277,820 / 272,757 / 277,949
characters, same hash. Equal file sizes are consistent with, but are not evidence of, identical
content.

So the conclusion I needed holds — the split changed nothing a reader sees, every page number below
is valid, and this report's findings survive the refactor — but it holds on the text-layer
measurement, not on a SHA comparison that in fact fails. Every source coordinate below was
re-resolved against the post-split files today.

**The audit corpus.** Baseline `9893a2c1` (the last commit before this round's prose landed).
Comparing sentence sets, comment-stripped and de-TeXed: **108 new or rewritten sentences, 3,364
rendering words** in chapters and appendices, plus **18 sentences / 751 words** of new errata-table
prose (which the reader does read, so I swept it too). That is close to the ~1,400 words the brief
names; the difference is that my count includes the errata-table cells and the two protocol
paragraphs in full rather than only their new clauses.

**One provenance correction to my own first pass.** My initial diff-based extraction flagged
`To ensure a robust evaluation of our models` and `Class-wise metrics are crucial because` (now
`chapters/3_cbic/results.tex:30`) as new. They are not: both are in the baseline, verbatim, in the
published CBIC text. The paragraph was *re-wrapped* this round when four protocol sentences were
inserted into it, which made a line-level diff report the whole paragraph as changed. `robust` and
`crucial` are therefore reproduced published prose, outside this round's scope and outside the
errata policy's reach. I moved to a sentence-set comparison after catching this, and every count in
this report is on that footing.

---

## 1 · Re-resolved coordinates (today, post-split)

Cite the phrase; these line numbers are good as of 2026-07-28 and will drift.

| ID | passage | file | line |
|---|---|---|---|
| R1 | Resumo, rebuilt (PT) | `0_main.tex` | 229 |
| R2 | Abstract, rebuilt (EN) | `0_main.tex` | 315 |
| C2a | Ch.2 HGI sweep re-anchored | `chapters/2_fundamentals.tex` | 171 |
| C2b | Ch.2 Check2HGI loss, three equations | `chapters/2_fundamentals.tex` | 241 |
| C2c | Ch.2 joint model's descent from MTLnet | `chapters/2_fundamentals.tex` | 322 |
| C3a | Ch.3 Standley footnote | `chapters/3_cbic/method.tex` | 94 |
| C3b | Ch.3 protocol additions | `chapters/3_cbic/results.tex` | 30 |
| C4a | Ch.4 preface | `chapters/4_courb.tex` | 18 |
| C4b | Ch.4 Nash-MTL narrowed | `chapters/4_courb/methodology.tex` | 36 |
| C4c | Ch.4 protocol additions | `chapters/4_courb/results.tex` | 14 |
| C5a | Ch.5 balancer-screen scope | `chapters/5_mobiwac/02_related.tex` | 112 |
| C5b | Ch.5 trunk attribution softened | `chapters/5_mobiwac/07_discussion.tex` | 16 |
| C5c | Ch.5 parity limitation | `chapters/5_mobiwac/07_discussion.tex` | 74 |
| A1 | Appendix A §A.2 reproducibility | `chapters/apx_a_contributions.tex` | 97 |
| A2 | Appendix B additions paragraph | `chapters/apx_b_errata.tex` | 153 |
| A3 | Appendix B §B.5 static scope (new file) | `chapters/apx_b_static_scope.tex` | 7 |
| A4 | Appendix C disclosure, trimmed | `chapters/apx_c_ai_disclosure.tex` | 81 |

---

## 2 · Persona 03 — the counted report (G3)

This gate's output is quantitative. Corpus = the 3,364 new rendering words unless stated.

### 2.1 Hard fail conditions — all at zero

| Check | Count | Evidence |
|---|---|---|
| Banned words / templates (law §4 + MobiWac GLOSSARY §7 tables, 63 patterns) | **1** | `stands as`, `apx_b_static_scope.tex:32`. See STY-04. |
| Em-dash **in prose** | **0** | 4 source instances exist, all in front-matter placeholders (`0_main.tex:144,145,146,190`: banca members, defense date, approval-sheet text). |
| Em-dash **rendered** | 2 (defense/final), 3 (ppgc) | defense pp. 83, 86: one is a source title (`Massive-STEPS: … — Dataset and Benchmarks`, `references.bib:1181`), the other is inserted by `abntex2cite.sty:294-295` (`\def\UrlLeft{<}` block sets a ` --- ` separator for the mastersthesis entry). ppgc adds p. 2, the approval-sheet placeholder. **None is authored prose**; the law's target is met. |
| Contractions | **0** | whole document, comment-stripped |
| British spellings | **0** | whole document (19 patterns) |
| Repo codenames new this round | **0** | Surviving hits are all pre-existing and legal in context: `frozen` ×3 (glossed as frozen weights / frozen pathway), `engine` ×1 (`apx_a:40`, "embedding-engine suite"). No new instance. |
| Registry violations (fail-closed L2) | **7** | STY-01. This is the one place the law is not satisfied. |

### 2.2 Density metrics

| Metric | Audit corpus | Whole doc (live) | Whole doc (base `9893a2c1`) | Law |
|---|---|---|---|---|
| `-ly` adverbs | 16 / **0.48 %** | 250 / 0.71 % | 236 / 0.71 % | band ≈0.8 % max — **inside** |
| two `-ly` in one sentence | **1** | — | — | never — see STY-05 |
| Intensifiers / boosters | **2** (`sharply`, `far`) | — | — | ≤1 per claim — **met** |
| `significant*` without a named test | **0** | — | — | — |
| Semicolon braids (≥2 semicolons) | **2** | 18 | 18 | a 2-semicolon sentence is 2 sentences — see RDB-02 |
| `X, not Y` | 3 (0.89/1k) | 42 (1.19/1k) | 41 (1.23/1k) | honesty device; density **fell** |
| `rather than` | 14 (**4.16/1k**) | 79 (2.23/1k) | 67 (2.01/1k) | see AIC-01 |
| Negative-parallelism family (`, not` + `rather than` + `instead of` + `not…but`) | **5.35/1k** | 3.73/1k | 3.57/1k | see AIC-01 |
| Rule-of-three / enumerated announcements | 4 new | 13 | 9 | see AIC-02 |

### 2.3 Distributional pass (variance, the deepest tell)

Sentence-length statistics, computed on de-TeXed sentences:

| corpus | n | mean | sd | cv | min | max | short <12w | long >35w |
|---|---|---|---|---|---|---|---|---|
| This round's new prose | 106 | 32.1 | 23.3 | **0.73** | 5 | 153 | 10 | 33 |
| Whole document (live) | 1305 | 28.9 | 25.9 | 0.89 | 3 | 341 | 158 | 329 |
| Whole document (base) | 1225 | 29.2 | 26.4 | 0.90 | 3 | 341 | 149 | 314 |

The new prose is **slightly flatter and longer** than the document it joins (cv 0.73 vs 0.89, mean
32.1 vs 28.9) but is not variance-compressed: it keeps 5-word sentences next to 90-word ones, and
the document-level statistics barely moved (cv 0.90 → 0.89). **This is not the failure mode the law
warns about.** Per-block, two blocks are flat enough to name:

- `apx_c_ai_disclosure.tex` new prose: **cv 0.37**, the lowest substantive block (see AIC-03).
- `chapters/4_courb/results.tex` protocol paragraph: cv 0.35 on 4 sentences (mean 35.2 words).

Sentence openers: no repeated 2-word opener above 3 instances across the corpus; per-block anaphora
is clean except `Only the` ×2 in `apx_a` and `Every` ×3 in `apx_c` (AIC-03).

**Section endings:** none of the new blocks ends by restating itself. `§B.5` ends on where the
measurement lives; `§A.2` ends on a scope limit; `apx_c` ends on the author's responsibility; the
Ch.2 lineage passage ends on what the fact licenses. This rule is well observed.

---

## 3 · Findings

### STY-01 · MAJOR — seven terms are in the prose that the fail-closed registry does not hold, and the chapter comment says this blocks the passage

**Anchor.** `is the logistic function` — `chapters/2_fundamentals.tex:255`; `bilinear
discriminator` at `:250`; and `logo o gargalo era a representação` — `0_main.tex:241`, `e testes
pareados sobre as quatro médias` — `0_main.tex:247`.

**Measured.** GLOSSARY §1's rule is "a term not in this registry may not be used." Accent-insensitive
containment against the whole registry file at HEAD:

*English, new to the document this round (live count / base count):* `bilinear discriminator` 1/0,
`logistic function` 1/0, `fine class` 3/0, `early stopping` 4/0, `the shared middle` 1/0. Not
registered.

*Portuguese, in the rebuilt Resumo:* `gargalo` 1/0 and `testes pareados` 1/0. Not registered — and
these two are **not** covered by commit `01915ba7`, which registered nine PT terms this same
afternoon and mentions neither (I read the commit body and diffed §6). The other seven PT phrases I
checked are covered by that commit, including in inflected form.

The Ch.2 source comment at `:307-314` states the position itself: "this passage uses two terms the
registry does not hold … the rule says the entry lands before the term does. Until they are
registered this paragraph is blocked on that approval." That is the drafting agent's own reading of
the law, and it is correct.

**Conclusion.** The registry is behind the prose in seven places, two of them in the Portuguese
front matter, where nobody flagged them. `early stopping` and `fine class` are ordinary vocabulary
that need translation-pair rows at most; `bilinear discriminator`, `logistic function` and `the
shared middle` are naming decisions (see STY-02, STY-03); `gargalo` and `testes pareados` need §6
rows whose English counterparts are `bottleneck` — itself unregistered, though used 4× in the body —
and `paired superiority test`, which is registered.

**Closes when** the author approves registry rows for the seven terms (or replaces the term in
prose), and `bottleneck` gets an English row so the PT pair anchors on something.

### STY-02 · MINOR — `the shared middle` is a second name, used once, for a thing the registry already names

**Anchor.** `overrides exactly one component, the shared middle` — `chapters/2_fundamentals.tex:326`,
renders on defense **p. 20**.

**Measured.** `shared middle` appears **1×** in the document (0 at base). `shared trunk` appears
**10×**. GLOSSARY §2 registers **the shared trunk** and defines it using the words "The shared
middle of the joint model" — i.e. "shared middle" is the registry's *gloss*, not its term. On the
render, `the shared middle` (p. 20) precedes the first body use of `shared trunk` (p. 58) by 38
pages, so a reader meets the non-canonical name first and the canonical one much later.

**Conclusion.** Synonym-cycling under WRITING_LAW §2, in the one direction that costs most: the
frame teaches the reader a name the rest of the document does not use.

**Closes when** the sentence uses the registered name ("it overrides exactly one component, the
shared trunk") or the registry gains `the shared middle` as an explicit alias.

### STY-03 · MINOR — the same function is named two ways in the document, and the new instance is the departure

**Anchor.** `is the logistic function` — `chapters/2_fundamentals.tex:255` (defense p. 19) against
`denotes the sigmoid function` — `chapters/4_courb/methodology.tex:114` (defense p. 49).

**Measured.** `logistic` 3 instances, `sigmoid` 1. The Ch.4 instance is **reproduced published text**
(present verbatim at base, inside the translated CoUrb methodology), so it cannot be changed under
the errata policy without a ledger entry. The Check2HGI source document says "σ é a função sigmoid"
(`docs/context/check2hgi_overview.tex:222`); the new Ch.2 sentence renders it as "logistic
function". Both names are correct mathematics; the document now uses both, 30 pages apart, for the
same σ.

**Conclusion.** One-name-per-concept is broken across the frame/paper seam. The frame is the
changeable side.

**Closes when** the author picks one (I would keep `logistic function` in the frame and add a
one-clause gloss "(also called the sigmoid)" at its first use, which also serves the Ch.4 reader),
or accepts the divergence and records it as a paper-chapter departure.

### STY-04 · MINOR — `stands as` is a copula-avoidance phrase on the inherited ban list

**Anchor.** `stands as published` — `chapters/apx_b_static_scope.tex:32`, defense **p. 99**.

**Measured.** The MobiWac GLOSSARY §7 banned-word table lists "a testament to, **stands as**, serves
as → shows, is". This is the only hit in 3,364 new words; the other 62 patterns are at zero, and
`serves as`/`acts as`/`functions as` are at zero.

**Conclusion.** A single instance is noise, not density, and the sentence it sits in is doing honest
work ("Every claim Chapter 4 makes about the sequential task … stands as published"). But it is a
listed phrase, and the persona's contract is to report hits, not to grade them.

**Closes when** the phrase becomes `remains as published` / `is unaffected`, or the author rules the
listed phrase acceptable here and the ruling is recorded.

### STY-05 · NIT — two `-ly` adverbs in one sentence, in the Standley footnote

**Anchor.** `argues the other way empirically, reporting that joint training … costly at training
time` — `chapters/3_cbic/method.tex:91`.

**Measured.** `empirically` and `costly` in one sentence. Law: "never two -ly adverbs in one
sentence." (`costly` is an adjective, so this is arguably a false positive of the rule as written —
which is itself worth telling the author, because the rule's mechanical form will keep firing here.)

**Closes when** the sentence splits at "and then argues", or the law's wording is narrowed to manner
adverbs.

### RDB-01 · MAJOR (readability) — the three-equation passage is the hardest page in the document, and its own scale convention is stated 4 pages before the metric is defined

**Anchor.** `The training objective makes that extension concrete` —
`chapters/2_fundamentals.tex:241`, renders **pp. 19–20**; and `the category F1 rose monotonically` —
`:173`, renders **p. 18**.

**Measured.** The equation passage: 506 words, 12 sentences, **42.2 words/sentence**, Flesch–Kincaid
**18.8** against the document's 16.0 — the densest block in the round by a wide margin.

Against that, the symbol discipline is genuinely good: I checked all twelve symbols that appear in
Eqs. 2.1–2.3 (`L`, `L_c2p`, `L_p2r`, `L_r2c`, `L_*`, `D`, `e_1`, `e_2`, `W`, `σ`, `e^+`, `e^-`) and
**every one is glossed in the surrounding prose**, the subscript shorthand is mapped to the three
boundaries in words, and the `*` in `L_*` is explained as "a boundary's term". A reader from outside
the project can follow the equations.

Two things that reader will trip on, both outside the equations:

1. **`0.7388 ± 0.0205` on p. 18 arrives with "on a zero-to-one scale", but macro-F1 is not defined
   until p. 22**, and the document's dominant convention is out of 100 (GLOSSARY §4: "Out of 100";
   Ch.5 reports 55.87, 75.15). So the first F1 number a reader meets is on the minority scale, four
   pages before the metric exists in the text, and it is called `category F1` — a phrase used
   **once** in the whole document, where `category macro-F1` is used 8×. The clause is scrupulous
   about its own provenance (the source records "Cat F1" without naming the averaging convention,
   per the `[VERIFY]` at `:188-198`) and that scruple is right; the cost is a reader who cannot tell
   whether 0.7388 is comparable to 55.87.
2. **The passage at `:322` uses five pieces of Ch.5 vocabulary 43 pages before Ch.5 defines them.**
   Measured first-body-use pages: `cross-attention blocks` p. 20, `task stream` p. 20, `feed-forward
   weights` p. 20, `private spatial path` p. 20, `the region sequence` p. 20 — against their
   definitions: the cross-attention stack p. 58, the private spatial path ("a small branch inside
   the one model") p. 63, the spatial stream p. 63. The p. 20 passage also precedes Ch.2's own MTL
   section (p. 21), where cross-attention is placed in the architecture spectrum.

**Conclusion.** The equations themselves are readable. The two framing problems are: a number on an
undeclared-until-later scale, and a paragraph that spends Ch.5's vocabulary before either chapter has
issued it.

**Closes when** (a) the sweep clause says which averaging convention it is or drops the two values
and stays qualitative (the `[VERIFY]` flag already proposes exactly this), and states the scale
relation to the document's out-of-100 convention; and (b) the lineage paragraph either moves after
Ch.2 §2.3 or adds a half-sentence gloss at "cross-attention blocks" ("blocks in which each task's
stream attends to the other's features; Chapter 5 §5.4.2 gives the construction").

### RDB-02 · MAJOR (readability) — the Appendix B additions paragraphs are a 120-word and an 89-word list inside single sentences

**Anchor.** `The others state protocol facts the published paper left implicit` —
`chapters/apx_b_errata.tex:158`, defense **p. 94**; and `Three state protocol facts the published
paper left implicit` — `:261`, defense **p. 96**.

**Measured.** 120 words with **3 semicolons**, and 89 words with **2 semicolons**. These are the only
two semicolon braids in the round's new prose, and they are the two longest new sentences in the
document (the next longest is 90). Whole-document braid count is unchanged at 18, so this is not a
new document-level pattern — it is two sentences carrying a four-item and a three-item list without
a list.

On the page they read as a wall: p. 94's first paragraph runs 14 lines with no sentence boundary
until line 11.

**Conclusion.** The content is exactly right for an errata appendix (each item names a protocol fact
and its consequence). The delivery contradicts the law's own rule that a two-semicolon sentence is
two sentences, and it is the least scannable prose in a document whose appendices exist to be
scanned.

**Closes when** each becomes a short lead sentence plus an `enumerate` or `description` list — which
is also what §A.2 two pages earlier already does for the same kind of content, so the fix makes the
appendices consistent rather than introducing a new device.

### RDB-03 · MAJOR (readability + presentation) — 13 new file paths render inside angle brackets and 4 of them break mid-token

**Anchor.** `\path{docs/studies/closing_data/v17_completion/STATISTICAL_PROTOCOL.md}` —
`chapters/apx_a_contributions.tex:116`, renders defense **p. 89**.

**Measured, on the render, not the source.** `abntex2cite.sty:294-295` sets `\def\UrlLeft{<}` and
`\def\UrlRight{>}`, and `url.sty:205` defines `\path` as a URL command, so **every `\path{...}`
prints wrapped in `<...>`**. On p. 89 that is **13 angle-bracket tokens**, all of them local
repository paths, not URLs. Document-wide: 29 angle-bracket tokens, of which 12 are genuine URLs (the
ABNT convention, correct) and **17 are local paths** — 13 of them new this round (`\path` count in
`apx_a` went 0 → 13; the 4 pre-existing ones are in Appendices D and E).

Worse, `xurl` breaks these long paths **inside tokens**. Measured from the text layer, 5 paths break
across a line and **4 of those breaks fall mid-word**:

- `scripts/closing_data/score_joint_b` | `est.py`
- `implemented in <script` | `s/closing_data/superiority_wilcoxon.py`
- `docs/studies/closing_data/v1` | `7_completion/stats_n20`
- `scripts/embedding_eval/autocorrela` | `tion_ceiling.py`

**Conclusion.** A reader who wants to type one of these paths cannot read it off the page reliably,
and `<…>` signals "this is a URL" for something that is a path in a repository. This is the section
whose whole purpose is that a reader can find the file. It is the most consequential presentation
defect in the round, and it is invisible in the source.

**Closes when** the paths use `\texttt{}` with explicit break hints, or a `\pathnobrackets` macro
that keeps the monospace and drops `\UrlLeft`/`\UrlRight`, and the four mid-token breaks are
re-checked **on the render** afterwards.

### RDB-04 · MINOR (readability) — Appendix A now ends on a 93-word page, and the near-blank page the round set out to remove has moved rather than gone

**Anchor.** `Two qualifications belong with this list` — `chapters/apx_a_contributions.tex:143`,
defense **p. 90**.

**Measured.** Median page fill in the defense build is **390.5 words**. Page 90 carries **93 words**
and nothing else. I built the baseline (`9893a2c1`) to check whether this is new: at base, Appendix A
occupied **one page** (p. 89, 331 words) and there was no p. 90; live it occupies **three** (374,
396, 93), because §A.2 is new. The round's Resumo fix did remove the old near-blank p. 4 (base p. 4
carried 19 words; live has no such page), so that fix worked — but the document still has a
near-empty page, now at the end of Appendix A.

**Conclusion.** Not a gate item and not a claim defect; a presentation cost of the new section that
a committee member flipping the appendices will see.

**Closes when** §A.2's last paragraph is tightened by ~30 words, or the two qualifications fold into
the preceding list as a final `\item`, pulling the page back.

### AIC-01 · MAJOR (AI-credibility, expert channel) — `rather than` runs at twice the document's rate in the new prose

**Anchor.** `for two reasons that we can state rather than assume` —
`chapters/5_mobiwac/07_discussion.tex:74`, defense **p. 74**.

**Measured.** `rather than`: **14 instances in 3,364 words = 4.16 per 1,000**, against 2.23/1k
document-wide and 2.01/1k at base. The whole negative-parallelism family (`X, not Y` + `rather than`
+ `instead of` + `not … but`) runs **5.35/1k in the new prose against 3.57/1k at base** — a 50 %
increase in the construction the AI-detection literature calls negative parallelism and this
project's own law lists as a known LLM fingerprint.

Distribution: `apx_b_static_scope.tex` 3, `chapters/3_cbic/*` 4, `apx_b_errata.tex` 3,
`apx_a_contributions.tex` 2, `apx_c` 1, `07_discussion` 1. In `§B.5` the three land within 30 lines:
"a deterministic mapping **rather than** a learned semantic inference" … "a visit **rather than** a
place" … "We state this here **rather than** in the chapter".

Note that `X, not Y` itself went **down** (1.23 → 1.19 per 1k), so the drafting did honor the
"keep it only where it scopes a claim" rule for that form. The pressure moved into `rather than`,
which no sweep was counting.

**This round crossed a standing guard, and the guard was issued by this same persona.** The v1 run of
persona 16 (`src_utils/_review_v1/16_ai_credibility_report.md:72-79`) raised this exact construction
as its item 3, called it "the single tell a 2026 CS examiner is most primed to see", judged the
density then defensible, and closed with an explicit instruction: **"Freeze the count; do not let
edit passes raise it."** Its summary line (`:266-267`) repeats it: "guard, do not add; freeze the
count across future edit waves."

On my own instrument and my own baseline, this round raised it: **`rather than` 67 → 79 (+12)
document-wide**, and 14 of the 79 live instances are inside this round's new sentences, so the new
prose accounts for the whole increase. `X, not Y` moved 41 → 42 (+1). (I deliberately do **not**
compare against the prior report's absolute counts of 27 and 28: those were measured on the v1 build,
a materially shorter document, on an instrument whose convention I cannot reproduce. The trend above
is measured on one instrument across one baseline.)

**Conclusion.** Each instance is individually defensible — most are genuine scope contrasts, which is
the honest use. Density is what convicts, this is the one distributional measure in the round that
moved materially in the wrong direction, and it moved against a written instruction not to move it.
That the guard was crossed silently is the finding; the twelve added instances are the symptom. Three
of the fourteen new ones are decorative and would read better as direct statements ("The protocol
described is that chapter's, and Chapter 2 scopes it there rather than to the whole collection" →
"…and Chapter 2 scopes it to that chapter"; "quoted rather than computed" is fine as a rule
statement; "state rather than assume" is doing work).

**Closes when** the document-wide count returns to approximately its pre-round level (≈67, i.e. about
8 of the 14 new instances rewritten as direct statements) — **not** by substituting a synonym, which
would be synonym-cycling. I also propose adding `rather than` to WRITING_LAW §4's density list with a
per-1,000-word band, since the existing list counts only `X, not Y` and therefore could not see this;
a guard that lives only in a review report from a previous round is a guard that gets crossed.

### AIC-02 · MINOR (AI-credibility, expert channel) — the "N qualifications … The first / The second / The third" scaffold is now a house template, and three of its instances sit within ten pages

**Anchor.** `Three qualifications matter, and none of them is a softening` —
`chapters/apx_b_static_scope.tex:26`, defense **p. 99**.

**Measured.** The `N <plural noun>` announcement opener: **13 instances live, 9 at base, 4 new this
round** (`Two properties` in the Abstract, `Two qualifications` in §A.2, `Three qualifications` in
§B.5, `Three rules` in Appendix C). Of the 13, **3 are followed by an explicit First/Second/Third
scaffold**, and all three of those are new or newly adjacent: the Abstract (p. 3), §B.5 (p. 99), and
Ch.5's `Two facts` (p. 65, pre-existing).

In the appendices the openers now cluster: p. 90 `Two qualifications`, p. 94 `Two corrections`, p. 99
`Three qualifications`, p. 102 `Three rules`, p. 106 `Two qualifications`, p. 106 `Two facts`. Six
paragraph openings of one shape in seventeen pages.

The §B.5 sentence stacks three separate tells in eleven words: the count-announcement, the
`matters, and` hinge, and the `none of them is` negative. It is then followed by exactly
`The first is that…` / `The second is that…` / `The third is that…` — the outline-shaped uniformity
persona 16 exists to catch.

**Conclusion.** Discourse-skeleton reuse (law §4.4). Not a fail condition — the device is a real
academic move and each instance is honest — but it is the strongest "nobody owned this prose" signal
in the round, precisely because it is legal under every word list.

**Closes when** at least the two new adjacent instances (§B.5 and Appendix C) drop the count from
the opener and let the paragraph open on its content — e.g. §B.5: "The sequential task is
unaffected." then "This is specific to Chapter 4's representation…" then "Chapter 5 does not inherit
the problem…". The information is identical; the shape stops repeating.

### AIC-03 · MINOR (AI-credibility, both channels) — the Appendix C disclosure is the flattest and most anaphoric paragraph in the round, on the one page a suspicious reader will read closest

**Anchor.** `Three rules governed the drafting` — `chapters/apx_c_ai_disclosure.tex:81`, defense
**p. 102**.

**Measured.** New prose in that file: 9 sentences, mean 16.1 words, **cv 0.37** (lowest substantive
block; corpus 0.73, document 0.89), Flesch–Kincaid 10.1. First words: `This, Assistant-run, Their,
Three, Every, Every, Every, Passages, The` — **three consecutive sentences open with `Every`**, in
strict parallel (`Every reference was checked…` / `Every number was traced…` / `Every verdict verb
was bound…`), all passive, all 12–17 words.

Also: the paragraph announces "Three rules" and then delivers **five** sentences before the
author-responsibility close, the fourth of which ("Passages that could not be verified were
flagged…") is a fourth rule in rule shape. The source comment at `:44-48` says this is deliberate —
the fail-closed rule was "the one substantive thing the chain named" and was promoted out of it —
but on the page it reads as a count that does not match its list.

**Conclusion.** This is the passage whose credibility matters most: it is the AI-use disclosure, and
the trimming this round genuinely improved it (the removed "passed an eighteen-reviewer panel" claim
was false, and cutting it was right — the panel's own record shows two gate FAILs). What remains is
tell-shaped in the two ways the evidence base flags for *both* channels: low sentence-length variance
raises detector risk, and triple anaphora with uniform passives raises expert suspicion.

The fix is **additive, not subtractive** (persona 16's standing rule): the paragraph is abstract
where the repo record could make it concrete. The project's honest-arc material is exactly what a
generator never has — a false claim about this document's own review was found and removed this
round; a Standley citation was corrected against its source; a `[VERIFY]` flag stands open on the
HGI averaging convention. One sentence naming one such correction would do more for credibility than
any rewording.

**Closes when** the three `Every` sentences vary in shape and length (at least one active, at least
one short), the count matches the list (either "Four rules" or drop the number), and one concrete
instance of a caught-and-corrected defect is named.

### AIC-04 · MINOR (AI-credibility, expert channel) — the same protocol facts are restated at up to seven sites, one of them a 53-word verbatim run across two chapters

**Anchor.** `The folds are formed by a stratified splitter over the samples rather than over the
users` — `chapters/3_cbic/results.tex:30` (p. 36) against `The split is stratified by sample, not by
user` — `chapters/4_courb/results.tex:14` (p. 52).

**Measured.** The two new protocol paragraphs share **96 words in three verbatim runs of ≥8 words**
(20, 23 and 53 words), which is **62 % of the Ch.4 paragraph**. The 53-word run is identical to the
character.

Counted on the render as **distinct statement sites** (my first pass said 8 for the seed row; that was
a two-alternative regex counting one sentence twice, since "pins a single random seed" and "one
repetition of the experiment" co-occur in the same sentence at every site. Corrected here, and the
split-axis row was de-duplicated the same way, from 12 raw pattern hits to 7 sites):

| fact | distinct statement sites | pages |
|---|---|---|
| split stratified by sample, not by user | **7** | 23, 36, 42, 52, 88, 94, 96 |
| single seed / one repetition | **4** | 36, 52, 94, 96 |
| no early stopping, own-best epoch | **4** | 37, 52, 94, 96 |
| Ch.5 repeats at four initializations | **5** | 3, 37, 52, 58, 89 |

**Conclusion.** Every instance is defensible in isolation: two chapters need the disclosure, the
appendix must ledger it, and Appendix A must state the contrasting protocol. But a 53-word verbatim
run across two chapters, plus a near-verbatim restatement in the appendix, is the "long-form
repetition" failure the guardrails' L3 cross-chapter duplication check exists to catch, and it is
exactly what a reviewer reading straight through will read as machine-assembled. Appendix B is
already aware of the duplication and *documents* it ("the last of them is worded exactly as in
Chapter 4, whose study runs the same code") — which is honest, and also an admission that the same
sentence is in the document twice.

**Closes when** the author decides which site owns the full statement and the other cites it. My
suggestion: Ch.3 carries the full four facts (it comes first), Ch.4 carries the split axis and the
seed in its own words plus "under the protocol described in Chapter 3 §3.4.1", and the Appendix B
paragraphs point at both rather than re-listing. That removes ~90 duplicate words and the appearance
of paste.

### VER-01 · MINOR — the split's "same SHA" claim does not hold as stated; the render-parity conclusion it supports does

**Anchor.** The structural-change notification for commit `4e84cf7a`: "all three builds' full text is
byte-identical before and after (defense 108 pp, final 105 pp, ppgc 109 pp, **same SHA**)."

**Measured.** Built the tree before and after the split and hashed both. The three PDF pairs have
**identical file sizes** and **different sha256** (`0d82fe5c` → `cea2ae65`, `36f2860b` → `4355635d`,
`b29755a8` → `e911b305`). This is expected: pdflatex embeds a creation timestamp and a document ID, so
two builds of byte-identical source cannot hash equal. The **text layers** do match byte for byte and
by hash (`bbc0d8b4`, `f1969684`, `c6eb1e84`, identical pre and post).

**Conclusion.** The substantive claim is true and I relied on it: the split is render-neutral, so page
numbers and findings carry across it. But "same SHA" is not the measurement that shows it, and equal
file sizes are consistent with identical content without being evidence of it. Worth recording because
this repository's standing failure mode is exactly this shape — a plausible self-claim resting on an
instrument that cannot support it.

**Closes when** the claim is restated as what was measured (identical extracted text layer, or
identical page images), which is both true and stronger than a SHA comparison would have been.

### UNV-01 · MINOR (partially verified) — Appendix C's own word-count claim: the delta reproduces exactly, the absolute pair is off by a constant 5

**Anchor.** The source comment at `chapters/apx_c_ai_disclosure.tex:47-48`: "Word count 374 → 303,
measured on this file with LaTeX comments stripped and escaped percent signs preserved".

**Measured.** I counted the live file and its base (`9893a2c1`) version on four plausible readings of
that convention (labels and chapter heading stripped; heading kept; whitespace tokens; whitespace
tokens with heading). Results, live / base / delta:

| convention | live | base | delta |
|---|---|---|---|
| strip `\label` + `\chapter` | 298 | 369 | **71** |
| include the heading | 301 | 372 | **71** |
| whitespace tokens | 299 | 370 | **71** |
| whitespace tokens + heading | 300 | 371 | **71** |

The claimed delta is 374 − 303 = **71**. So the *reduction* reproduces exactly on every convention I
tried, and my absolute counts sit a constant **5 tokens below** the claim — consistent with the claim
counting five tokens I am dropping (most plausibly the two `\ref` expansions, the
`\textsuperscript{o}`, and the chapter title as two words). The claim is therefore substantively
sound, not reproducible to the digit from its stated convention.

**Separately verified, and this is the part that matters:** the false claim is gone. No
"eighteen-reviewer" or "passed a panel" text exists in the source or on rendered p. 102. I checked
the panel record the comment cites (`_review_v1/CONSOLIDATED_REVIEW_REPORT.md`) is what the comment
says it is — two gate FAILs — so removing the claim was correct and the comment's justification holds.

**Closes when** the comment's convention names what it counts (`\ref` expansions and the heading), or
the two absolute numbers are dropped and only the delta kept — they are internal bookkeeping and
nothing in the document depends on them.

---

## 4 · The Portuguese Resumo, judged on its own terms

WRITING_LAW's register rules were written for the English frame; §6 gives Portuguese only three
lines. So I judged the Resumo as Portuguese academic prose and report separately.

**Measured on the rendered page (defense p. 2), stated convention: catalog header and keyword block
excluded, hyphenated compounds one word.**

| | Resumo | Abstract |
|---|---|---|
| words | **308** | **270** |
| sentences | **11** | **11** |
| mean | 28.0 | 24.5 |

Both are inside the 195–282/6–12 envelope the round targeted for sentence count, and slightly above
it for Resumo words. The two blocks pair **1:1, sentence for sentence** — I checked all eleven pairs.

**Claim parity: passes.** Every claim-bearing token matches across the pair: `TOST`, the two-point
margin, `Acc@10`, the joint-best convention, superiority verb count (2 each: `supera`/`outperforms`),
match verb count (1 each: `equipara-se`/`statistically matches`), "nos seis"/"at all six", "quatro
deles"/"four of them", the next-place exclusion, and the conditional answer. The **only** divergence
is the decimal separator (`5,3 a 9,4` vs `5.3 to 9.4`), which is correct in each language and is
what parity should look like.

**What is right about it as Portuguese.**

- Verb choices are the registered ones and correctly bound to their tests: `supera` for the paired
  superiority result, `equipara-se estatisticamente, com não-inferioridade dentro de uma margem de
  dois pontos` for the TOST result. Arizona is not upgraded. This is the honesty law surviving
  translation, which is the hard part.
- Register is formal PT-BR without anglicism where a standard term exists: `aprendizado multitarefa`,
  `ponto de interesse`, `validação cruzada`, `partições fixas`, `inicializações aleatórias`,
  `topologia de compartilhamento`. Loanwords are held to eleven tokens, all of them terms of art
  (`check-ins`, `embedding` in parentheses as a gloss, `joint-best` italicized per BR convention,
  `macro-F1`, `Acc@10`, `TOST`, `MTL`, `MTLnet`, `POI`, `Massive-STEPS`). That is the right call:
  calquing `joint-best` would make the Resumo disagree with the tables.
- Decimal commas throughout (`5,3`, `9,4`), correct for PT.
- `mas não o ponto de interesse exato` states the next-place exclusion positively and avoids the
  reserved term `próximo lugar visitado`. Good; the round-5 finding is still honored.

**Three things I would raise with a Brazilian reader in mind.**

1. **`sobre` is doing two different jobs in one sentence** (`0_main.tex:246-247`, p. 2): "quatro
   inicializações aleatórias **sobre** cinco partições fixas, e testes pareados **sobre** as quatro
   médias por inicialização". The first `sobre` means *across/over* a set of folds; the second means
   *on* a set of values. A third `sobre` closes the block ("construída **sobre** ela"). Three
   `sobre` with two senses in 57 words is the kind of thing that reads as translated-from-English
   rather than written in Portuguese. `em cinco partições fixas` for the first would fix it.
2. **Sentence 9 is 57 words** — the longest in either block, and 29 words longer than the Resumo's
   mean. In Portuguese, with its longer words, that lands heavier than the 47-word English
   counterpart. Splitting after `por inicialização` would cost nothing.
3. **Serial comma before `e`** appears 6 times (`…, e não a arquitetura`; `…, e Istambul`;
   `…, e testes pareados`). This is acceptable in modern PT-BR when the coordinated members are long,
   and two of the six are genuinely disambiguating. But `do Gowalla, e Istambul, do Massive-STEPS` is
   the one place it reads oddly, because the commas around `do Gowalla` already make the reader pause
   — the sentence has three commas in twelve words. `cinco estados dos Estados Unidos (Gowalla) e
   Istambul (Massive-STEPS)` is cleaner and keeps the parenthetical convention the Resumo already
   uses for `(embedding)`.

**On the law itself:** WRITING_LAW §6 tells an agent that the Resumo must be formal PT-BR and mirror
the Abstract, and it gives the term table. It says nothing about PT-specific mechanics — decimal
comma, serial-comma practice, preposition repetition, sentence length in a language with longer
words. Those are the only three defects I found, and none was catchable from the law as written. I
propose §6 gain four lines covering them, so the next agent drafting Portuguese has a standard to
obey instead of English rules applied by analogy.

---

## 5 · What holds, and should not be touched

- **The claim-parity pair is the best thing in this round.** Eleven sentences, 1:1, every hedge and
  verb and scope token matching, cut in one pass so they cannot drift. It reads as one author in two
  languages.
- **The equation symbol discipline.** All twelve symbols glossed, the subscript shorthand mapped to
  the hierarchy in words, `e^-` described honestly as the batch-permutation path with a source note
  saying the equations do not depend on it. A reader from outside can follow Eqs. 2.1–2.3.
- **Ch.4's Nash-MTL narrowing** (`chapters/4_courb/methodology.tex:36`). "Away from a
  Pareto-stationary point, meaning a point at which some convex combination of the task gradients is
  zero, and under the method's assumption that the gradients are linearly independent there, that
  direction is a descent direction for every task" — a conditional guarantee stated as a conditional,
  with the condition glossed inline for a reader who does not know the term. This is what the rest of
  the document should sound like when it touches a cited method's guarantee.
- **Ch.5's attribution softening** (`07_discussion.tex:16`). "Which part of the joint architecture
  produces the category gain is not settled by the controls reported here" — an honest negative,
  short, placed immediately after the positive claim rather than buried in limitations.
- **The Ch.5 parity limitation** (`07_discussion.tex:74`) is over-long at 88 words but its content is
  the most credible writing in the round: it names the direction of its own bias, says the residual
  favors the comparator, and then refuses the easy conclusion ("It does not follow that the bias
  cancels exactly."). Do not smooth that last sentence away.
- **§B.5's opening** is exemplary honest scoping: it states the identity between input and label,
  quantifies it (284–365 fine classes, all mapping to exactly one of seven), and calls the result "a
  deterministic mapping rather than a learned semantic inference" without softening. Only its
  three-part scaffold (AIC-02) and one banned phrase (STY-04) need attention; the substance should
  not be diluted.
- **Appendix C's cut of the false panel claim.** Removing an assertion that the document "passed an
  eighteen-reviewer panel" whose own record shows two gate FAILs is the single most credibility-positive
  edit in the round. Whatever happens to AIC-03, that cut stays cut.

---

## 6 · Proposed law updates (for author approval, not applied)

1. **WRITING_LAW §4 density list: add `rather than` with a per-1,000-word band.** The list counts
   `X, not Y` and therefore could not see AIC-01, where the same rhetorical move moved into a phrase
   nobody was counting. Suggested band: ≤2.5/1k, the document's current rate.
2. **WRITING_LAW §4.4: name the `N <plural noun>` + `The first/second/third` scaffold explicitly.**
   It is now at 13 instances document-wide and is the round's strongest gestalt tell; it is legal
   under every existing rule.
3. **WRITING_LAW §6: add PT-specific mechanics** (decimal comma; serial-comma practice; do not repeat
   a preposition in two senses in one sentence; sentence-length guidance calibrated for PT). Section
   6 currently gives an agent no Portuguese standard beyond the term table.
4. **GLOSSARY §6: register `gargalo` and `testes pareados`, and give `bottleneck` an English row** so
   the PT pair anchors (STY-01).
5. **The `-ly` rule's mechanical form catches adjectives** (`costly`, STY-05). Narrow it to manner
   adverbs or the gate will keep firing on false positives.

---

## 7 · Three of my own measurements were wrong before this report was written

Recorded because this project's standing failure mode is a plausible claim about itself that nobody
checked, and a reviewer is not exempt from it.

1. **I flagged reproduced published prose as new.** A line-level diff reported the whole Ch.3 protocol
   paragraph as changed, because inserting four sentences re-wrapped it. `robust` and `crucial` are
   baseline CBIC text, not round-6 additions. Had I not re-checked against the baseline file, this
   report would have opened with two banned-word hits against published prose the errata policy
   protects. Fixed by switching to sentence-set comparison; §0 records it.
2. **I counted one sentence twice and called it eight repetitions.** The protocol-repetition table's
   seed row used `pins a single random seed|one repetition of the experiment`; both clauses live in the
   same sentence at all four sites, so the alternation doubled every hit. The split-axis row had the
   same defect (12 raw hits → 7 sites). Corrected in AIC-04.
3. **I filed the Appendix C word count as UNVERIFIED when the substantive part verifies.** I could not
   reproduce 374 and 303 as absolutes, concluded "not reproducible" and stopped. Re-measuring on four
   readings of the stated convention showed the **delta is exactly 71 on every one of them** — the
   claim's substance holds and only the offset differs. "I could not reproduce it" and "it is wrong"
   are different findings, and the first does not license the second.

The instrument note that follows from these: on a document whose paragraphs are single long source
lines, line-level diffs are not evidence of what changed, and multi-alternative regexes are not
counts. Both are worth writing into the next round's brief.

---

## 8 · Out-of-scope handoffs (one line each, other personas own them)

- Persona 06/07: the `[VERIFY]` at `chapters/2_fundamentals.tex:188-198` on the swept "Cat F1"
  averaging convention is still open and is load-bearing for the p. 18 number (see RDB-01).
- Persona 19: `\path` under `abntex2cite`'s `\UrlLeft`/`\UrlRight` prints `<…>` for local paths and
  breaks them mid-token; this is a source/build-engineering fix (RDB-03).
- Persona 19 / gate maintenance: the split at `4e84cf7a` briefly made 55 % of the prose invisible to
  every `chapters/*.tex` glob while the gates reported clean. The commit fixed the globs. I did not
  re-validate the fixed checkers in both directions, which the guardrails' "gate that has never
  fired" rule requires; that validation is owed and is not mine.
- Persona 18: defense p. 90 carries 93 words against a 390.5-word median (RDB-04).
