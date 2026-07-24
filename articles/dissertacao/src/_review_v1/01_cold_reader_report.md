# Cold Reader Report — main_defense.pdf (v1)

> Persona 01 (cold reader). First-pass comprehension only. No project docs, planning files,
> NORTH_STAR, glossary, or paper folders read — inputs were the persona, the Common protocol,
> and the 87-page document alone. Read-only; no science judgments, no style-law enforcement,
> no grammar lists. Findings = quote + location + why it broke + one-line direction (not applied).
> Document read start to finish once (text layer for all 87 pp; pp.33,35,51,60,62,67,68 also
> rendered as images to check figures/tables), then one organizing pass.

---

## 1. Overall verdict (scope: first-pass comprehension)

**The document is comprehensible and the argument-thread holds from start to finish. I was not
lost in any chapter.** The three-study arc (negative result → representation diagnosis →
resolution) is set up in the abstract and Chapter 1, threaded through a genuinely useful
Fundamentals chapter, and consolidated honestly in Chapter 6; I could state at the end of each
chapter what it claimed and why. The coletânea machinery works: each article chapter opens with a
preface that told me its venue, its date/status, and which later chapters revise it, so I always
knew whether I was reading a "conclusion of the time" or the final position.

Friction is real but concentrated in two buckets, not spread through the argument:
(a) **unfinished surface items that are known-open** — the placeholder title, and three literal
`[VERIFY: recompute …]` dataset numbers printed in the Chapter 3 results; and
(b) **one recurring terminology collision inherited from the reproduced papers** — the term
"Next-POI Prediction" (Chapters 3–4) reads as the one task the dissertation repeatedly says it
does *not* do (next place), when it actually means next *category*.

Neither bucket broke my understanding of the science; both would trip a banca reader on first
contact, and the first bucket is not shippable as-is. Fix those and this reads as one document,
not three stapled papers.

## 2. Top 3 findings (marked)

**① [TOP] "Next-POI Prediction" collides with the frame's core distinction (comprehension).**
Chapters 1–2 work hard to separate *next place* (named, explicitly NOT predicted) from *next
category* (predicted). Then Chapters 3 and 4 title-case "Next-POI Prediction" throughout, which
reads as next-*place*; §3.2.1 (p.27) even defines it as predicting "which specific location a user
is likely to visit next," directly contradicting the same term's definition one page earlier
(p.26: "the category of the next POI"). A reader who believed the abstract stumbles: "didn't they
say they don't predict the exact place?" Recurs 4+ times (pp.26, 27, 41). The prefaces do not warn
that the reproduced term = the frame's "next category." *Direction:* one sentence in the Ch.3 and
Ch.4 prefaces mapping "Next-POI Prediction (as used in this article) = next-category in the frame's
terms"; and disambiguate the two definitions at p.27. (= findings [3-1]/[3-2]/[4-1].)

**② [TOP] Dataset size is three printed `VERIFY` placeholders in the Ch.3 results (p.35).**
"This subset comprises a total of [N_users; VERIFY: recompute per ERRATA.md] users,
[N_poi; VERIFY: recompute per ERRATA.md] unique Points-of-Interest, and [N_checkins; VERIFY:
recompute per ERRATA.md] check-ins." A first reader cannot learn how big the Florida dataset is —
the basic fact the whole chapter rests on — and sees the project's internal review tooling in the
built PDF. It IS disclosed as deliberate ("Pending. Not invented") in errata Table 11 (p.83), but
that disclosure sits 48 pages after the damage. *Direction:* fill the three numbers (or, if truly
pending, render a neutral "(counts pending recomputation)" rather than the raw VERIFY strings)
before any external eyes. BLOCKER if it reaches the banca. (= finding [3-3].)

**③ [TOP] The dissertation has no title (front matter).** Title page (p.1), Resumo (p.3), and
Abstract (p.4) all read `[TITLE — OPEN DECISION NORTH_STAR §5.8]`. As a cold reader I never learn
what the work is called; the abstract's first line carries a bracketed editorial note. Clearly a
known open decision, and it does not impede understanding the argument — but it is the first thing
a banca sees and is unmistakably unfinished. *Direction:* resolve the open title decision before
the defense build ships. (= finding [FM-1].)

## 3. Ranked findings (remaining, most valuable first)

Severity per Common-protocol §5. Each: quote/location + why it broke + one-line direction.

1. **[4-2] MTLnet vs MTLNet — two spellings, late disambiguation.** MINOR. Frame (Ch.1/2) and
   Ch.5 write "MTLnet"; Ch.4 writes "MTLNet" everywhere from its title (p.41) on. The note
   explaining the difference ("the published paper typesets the name as MTLNet") does not arrive
   until §4.2.5 (p.44). *Direction:* move that one-line note into the p.41 preface.
2. **[2-1] The 93% predictability figure lands three times before it is qualified.** MINOR.
   pp.12, 17, 23. Only at §2.4 (p.23) is it bounded ("not a ceiling on seven-class macro-F1 or
   region ranking"). I felt the repetition as "haven't I read this?" before the qualification
   resolved it. *Direction:* qualify at first substantive use (p.17), or trim one instance.
3. **[1-1] Four-task bookkeeping right after the abstract sells "two."** MINOR. §1.1/§1.4 ask the
   reader to track next place (not done), next category (done), next region (done), and a *fourth*
   "category classification" that "also appears" and was later dropped. Handled explicitly, but the
   task count is briefly muddy on first contact. *Direction:* at first mention, one clause —
   "a static, non-sequential fourth task used only in the first two studies."
4. **[4-3] A headline gain the eye cannot verify from its own table.** MINOR. "average gains per
   state are 20.2 to 22.0 percentage points" (pp.51–52, and again p.62/Ch.6), but Table 6 has no
   per-state average row, so I cannot check it on first read. *Direction:* add a per-state mean
   row, or state "averaged over the seven categories."
5. **[5-1] Gradient-cosine sentence with stacked caveats.** MINOR. p.59 §5.2.4: "Measured during
   development on the same joint architecture (on an earlier preparation of the data), the cosine
   similarity … averages +0.001 … (four seeds each on three of our six datasets, per-dataset means
   within ±0.003)." Two nested parentheticals plus "an earlier preparation of the data" made me
   stop to ask whether this is on the final data and whether that matters. *Direction:* split into
   two sentences; state why the earlier-data measurement still stands.
6. **[3-4] Ch.3 re-teaches MTL from scratch (§3.2.2) right after Ch.2 taught it.** MINOR /
   structural. Expected in a faithfully reproduced paper and partly covered by the preface, but as
   a continuous reader I felt the repetition (hard/soft sharing, MoE, GradNorm/PCGrad/DWA all
   re-defined). *Direction:* none required if fidelity is the priority; note only.
7. **[5-2] Table 10 "Dedicated" header names two different models.** NIT. p.67: one "Dedicated"
   under Next-category (the dedicated category model), one under Next-region (the dedicated region
   model). Caption resolves it; first scan read them as one. *Direction:* optional subscripts.
8. **[FIG-1] Figure 3 scatter tick labels near-unreadable at print size.** NIT. p.51. The
   co-location message survives via the caption, but the axis numbers are too small to use.
   *Direction:* larger tick fonts or drop the numeric ticks.

## 4. What holds / what reads well (do NOT "improve" these)

- **The abstract (pp.3–4).** A single clean arc — negative → diagnosis → resolution — every
  sentence resolving. The PT/EN pair is parallel. This sells the whole dissertation in one page.
- **§2.4's binding of verbs to tests.** "outperforms" ← paired superiority test; "matches" ← TOST
  non-inferiority within a stated margin. Stated once, plainly, and it made every later results
  claim legible. Best single paragraph in the frame for a skeptical reader.
- **All of Chapter 5.** Dense, but every term is defined before use; the leakage audit (three
  grounds, p.63–64), the superiority-vs-TOST pre-assignment, and the freeze/capacity controls are
  laid out so a first reader can follow the reasoning. The CTLE-vs-Check2HGI contrast (sequence
  model vs network model) is crisp and memorable.
- **Chapter 6 consolidation.** Reads as genuine synthesis, not a summary of unread things; §6.2's
  conditional yes/no lands the arc.
- **The article-chapter prefaces and Appendix B (errata).** The prefaces did their orienting job
  every time. Appendix B retroactively answered several frictions I had accumulated (the CBIC
  bolding convention, the MTLnet spelling, the p.35 placeholders) — reassuring, not padding.
- **Figures 4 and 5** (the two Ch.5 schematics) are interpretable from the caption alone.

## 5. Out-of-scope handoffs (one line each; not comprehension findings)

- **→ persona 06 (number auditor): cross-page value mismatch.** p.73 §6.2 says the Alabama joint
  model reaches "64.54"; Table 10 (p.67) gives AL Joint category 64.51 ±0.09. 64.54 vs 64.51.
  Caught only because §6.2 restates a table value. Author to rule.

## 6. Open questions (only the author can answer)

- Is the placeholder title (finding ③) blocked on a decision that will land before the defense
  build, or does it need escalation now?
- Are the p.35 dataset counts (finding ②) genuinely pending a recompute script, or can the
  published CBIC values be quoted directly in the interim?
- Is preserving "Next-POI Prediction" verbatim in Chapters 3–4 a hard fidelity requirement (finding
  ①)? If so, a preface bridge sentence is the whole fix; if not, the term itself could be aligned.

---

## Appendix — Running stumble log (raw, in reading order)

### Front matter (pp.1–11)
- **[FM-1] Title is a literal placeholder.** Title page (p.1), Resumo (p.3), Abstract (p.4)
  all read `[TITLE — OPEN DECISION NORTH_STAR §5.8]`. As a cold reader I never learn what the
  dissertation is *called*. Running head/footers throughout say "Chapter N. …" so the doc is
  navigable, but the missing title on the abstract line is jarring. (Known open decision; flag
  so it is not shipped to the banca this way.) — MINOR (BLOCKER only if it reaches the banca).
- Resumo/Abstract (pp.3–4): parallel PT/EN, dense but each sentence resolves. The abstract is a
  single long arc (negative → diagnosis → resolution). Reads well; no stumble.
- Lists of figures/tables (pp.5–7), acronyms (p.8), contents (pp.9–11): complete and navigable.

### Ch.1 Introduction (pp.12–16)
- Reads effortlessly. The three-study arc is set up cleanly; scope ("exact next place is NOT
  predicted") is stated early and repeatedly.
- **[1-1] The "fourth task" bookkeeping.** p.12–13 §1.1 and §1.4: the reader must track four
  tasks — next place (named, not done), next category (done), next region (done), and a
  *fourth*, "category classification," that "also appears" in the first two studies and was
  later dropped. It is handled explicitly (and §1.2/§2.1 revisit it), but on first contact the
  count of "how many tasks am I tracking?" is briefly muddy right after the abstract sold "two."
  — MINOR; direction: none needed, or one clause naming it "a static, non-sequential fourth task
  used only in the first two studies."

### Ch.2 Fundamentals (pp.17–24)
- Strong, linear, didactic. 2.2's one-hot → DGI → HGI → check-in lineage and Table 1 read
  effortlessly. 2.3 MTL taxonomy (hard/soft, negative transfer, balancers, routing) is clear.
- **[2-1] The 93% predictability figure lands three times.** p.12 (§1.1), p.17 (§2.1), p.23
  (§2.4). §2.4 is where it is finally qualified ("not a ceiling on seven-class macro-F1 or region
  ranking"). A cold reader feels the repetition before reaching the qualification. — MINOR.
- What reads well: §2.4 binding of verbs to tests ("outperforms" ← paired test; "matches" ← TOST)
  is unusually clear and I will remember it going into the results chapters.

### Ch.3 Article 1 (CBIC 2025) (pp.25–40)
- Preface (p.25) works: tells me venue, that it is a "conclusion of the time," and that Ch.4/5
  revise it, including the Nash-MTL caveat. This is the coletánea preface doing its job.
- **[3-1] "Next-POI Prediction" collides head-on with the frame's careful vocabulary.** Ch.1/2
  spent real effort separating *next place* (named, NOT predicted) from *next category*. Then
  Ch.3's title and headers say "Next-POI Prediction" everywhere, which reads as next-*place*. On
  p.26 (line 776) it is defined as "Predicting the category of the next POI" — i.e. next category.
  A cold reader who believed the abstract ("the exact next place is not predicted") stumbles hard:
  "wait, didn't they say they don't do this?" The preface does not warn that the reproduced
  article's term "Next-POI Prediction" = the frame's "next category." — MAJOR; direction: add one
  sentence to the p.25 preface mapping the old term to the frame term.
- **[3-2] Same term, two different definitions, adjacent pages (in-article contradiction).**
  p.26 (line 776): "Next-POI Prediction: Predicting the **category** of the next POI a user will
  visit." p.27 §3.2.1 (line 827): "Next-POI Prediction, in contrast, aims to predict which
  **specific location** a user is likely to visit next." Same bold term, category on one page and
  specific place on the next. On first pass I could not tell whether the paper predicts the
  category or the venue. (Inherited from the published paper; §3.2.1 is arguably describing the
  literature's task, but nothing signals that.) — MAJOR; direction: distinguish "the general
  next-POI task (place)" from "our next-POI *category* task" at first use in §3.2.1.
- **[3-3] Raw editorial placeholders in the results text.** p.35 §3.4.1 (line 1137): "comprises a
  total of [N_users; VERIFY: recompute per ERRATA.md] users, [N_poi; VERIFY: recompute per
  ERRATA.md] unique Points-of-Interest (POIs), and [N_checkins; VERIFY: recompute per
  ERRATA.md] check-ins." The dataset size — a basic fact the whole chapter rests on — is missing
  and shows the internal VERIFY tooling in the built PDF. A cold reader cannot learn how big
  Florida is. — BLOCKER if it reaches the banca; at minimum MAJOR. Direction: fill the three
  numbers before any external eyes.
- **[3-4] The chapter opens by re-teaching MTL from scratch.** p.25–29 §3.2.2 re-defines hard/soft
  sharing, MoE, negative transfer, GradNorm/PCGrad/DWA — all already taught in §2.3. Expected in a
  reproduced paper, and the preface sort of covers it, but as a continuous reader I felt "I just
  read this in Ch.2." — MINOR (structural to the coletánea format; note only).
- What reads well: §3.3.3 Nash-MTL bargaining-game explanation is self-contained and clear even
  without the equations rendering perfectly in the text layer.

### Ch.4 Article 2 (CoUrb 2026) (pp.41–55)
- Preface (p.41) works well: translated reproduction, Tarik first author, and — crucially — it
  warns me up front that the split is by-sample not user-disjoint ("a weaker protocol"), and that
  this chapter does NOT revisit MTL-vs-single. Good expectation-setting.
- **[4-1] "Next-POI Prediction" = next category, third and fourth times.** p.41 (line 1361):
  "Next-POI Prediction: predict the category of the next POI." Same collision as [3-1]/[3-2],
  now in Ch.4. Reinforces that this is a *document-wide* hazard: two of three article chapters
  use a term ("Next-POI Prediction") that reads as the one task the dissertation says it does not
  do (next place). — folds into top finding [3-1].
- **[4-2] MTLnet vs MTLNet capitalization.** Frame (Ch.1/2) and Ch.5 write "MTLnet"; Ch.4 writes
  "MTLNet" everywhere (title p.41, preface, intro). The disambiguation note ("the published paper
  typesets the name as MTLNet, and this chapter preserves that form") does not arrive until §4.2.5
  (p.44), several pages after the reader first trips on the two spellings. — MINOR; direction:
  move the one-line note into the p.41 preface.
- **[4-3] A worked "gain" figure the eye cannot verify from the table.** p.51–52: "average gains
  per state are 20.2 to 22.0 percentage points." Table 6 gives per-cell values but no per-state
  average row, so I cannot check the 20.2–22.0 claim against the table on first read. — MINOR;
  direction: add a per-state mean row, or say "computed over the 7 categories."
- What reads well: §4.4.2/4.4.3 walk through Tables 6–7 concretely (naming Nightlife, Travel,
  Food winners), and the Travel-category exception is honestly flagged in both results and
  conclusion. The by-sample caveat is repeated at the table (p.51), not just the preface.

### Ch.5 Article 3 (MobiWac 2026) (pp.56–71) — reading so far pp.56–62
- Preface (p.56) works: "submitted, under review (EDAS #...)", states the resolution cleanly, and
  promises errata for every departure. Status wording is consistent ("under review").
- **Relief: Ch.5 returns to the frame's vocabulary.** "next category" / "next region", and "we do
  not predict the exact next place" restated (p.56, p.60). After Ch.3/4's "Next-POI Prediction",
  this chapter reads as continuous with Ch.1/2. This makes the Ch.3/4 term collision [3-1] feel
  even more like the odd-one-out to fix.
- **[5-1] Dense gradient-cosine sentence with stacked caveats.** p.59 §5.2.4 (lines 2070–2074):
  "Measured during development on the same joint architecture (on an earlier preparation of the
  data), the cosine similarity ... averages +0.001 across training (four seeds each on three of
  our six datasets, per-dataset means within ±0.003)." Two nested parentheticals plus "an earlier
  preparation of the data" made me stop: is this measured on the final data or not, and does that
  matter? — MINOR; direction: split into two sentences and state why the earlier-data measurement
  still stands.
- What reads well: §5.1 intro and the three contribution bullets (p.57) are crisp; the CTLE-vs-
  Check2HGI distinction (sequence model vs network model) is a clean, memorable contrast.
- Ch.5 results (pp.63–71) read very well. §5.5.2/5.5.3 (windows, splitting, integrity, metrics,
  tests) is dense but every term is defined before use; the leakage audit (three grounds) and the
  superiority-vs-TOST assignment are laid out so a first reader can follow the logic. Tables 8–10
  and Figs 6–7 are interpretable from their captions (rendered-page check confirms this).
- **[5-2] Table 10 "Dedicated" header means two different models.** p.67: "Dedicated" appears once
  under Next-category and once under Next-region, denoting the dedicated *category* model and the
  dedicated *region* model respectively. The caption's "improvement over the dedicated model"
  resolves it, but on first scan I read one "Dedicated" model. — NIT.

### Ch.6 Conclusion (pp.72–75)
- Reads as genuine consolidation, not a summary of unread things. §6.2 "consolidated answer"
  (conditional yes/no) lands the arc cleanly. The two controls (freeze; capacity-matched baseline)
  are clearly flagged as frame-level additions run after Ch.5 submission.
- §6.3 six numbered limitations and §6.4 future-work-per-limitation are easy to follow.
- **[OOS-1 | out-of-scope, number-auditor handoff] Cross-page number mismatch.** p.73 §6.2 (line
  2580): the joint model at Alabama is "64.54 for the joint model"; Table 10 (p.67) gives AL Joint
  category = 64.51 ±0.09. 64.54 vs 64.51. I only caught it because §6.2 restates a Table-10 value;
  a true first-pass reader likely would not. Flagging for persona 06 (number consistency), not a
  comprehension finding. Author to rule.

### Appendices (pp.81–87)
- App.A (BRACIS) reads cleanly and explains the rejected iteration and why no result from it is
  cited, pre-empting a "what about BRACIS?" question. The phrase "Substrate Carries, Architecture
  Pays" appears only as the rejected paper's actual (quoted) title. No stumble.
- App.B errata (pp.83–86) is the most reassuring back matter: it retroactively answered several
  "why is this like this?" frictions — the CBIC bolding convention (why HMRM is never bold), the
  MTLnet/MTLNet spelling, and crucially the p.35 placeholders ("Pending. Not invented"). A reader
  who reaches B.1 learns the placeholders are deliberate. That does not remove the p.35 gap (most
  readers hit p.35 long before p.83) but downgrades it from "looks like a bug" to "disclosed item."
- App.C AI-use disclosure reads clearly and is thorough.

### Figure rendered-page check (pp.33,35,51,60,62,67,68)
- Figs 1,3,4,5,6,7 all interpretable from caption + panel. Fig 4/5 schematics read cleanly; Fig 6/7
  bar charts and the Table 10 ↑/≈ markers read well.
- **[FIG-1] Fig 3 (p.51) scatter tick labels near-unreadable at print size**; the co-location point
  survives via the caption, but the axis numbers are too small to use. — NIT.
- p.35 confirms the three `[N_… ; VERIFY: recompute per ERRATA.md]` placeholders print verbatim in
  the built PDF (see top finding [3-3]).
