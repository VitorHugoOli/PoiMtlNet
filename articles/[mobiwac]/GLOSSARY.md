# MobiWac 2026: Glossary and Plain-Word Guide (working draft v3)

> **Why this file exists.** Two of the three BRACIS reviewers rejected the previous paper mainly because it was
> hard to follow: too many acronyms and too much machine-learning jargon, used without explanation. MobiWac
> readers know networking, not embeddings, so the paper must use simple words by default and spend its few
> technical terms carefully.
>
> **v1 change.** Trimmed to only the terms the [`PAPER_PLAN.md`](PAPER_PLAN.md) actually uses. Task names are
> **category** and **region** (the literature terms in CBIC/CoUrb and the field), not activity/area. Unused
> networking acronyms (AP, MEC, 5G) and unused ML jargon were removed. Vitor will revise; it leans toward
> simpler, push back where a term earns its place.
>
> **v2 change (2026-07-10).** Added §7 (AI-text tells, web-sourced) and §8 (plain international English, the
> non-native register rule); added audience-collision rows to §3 (seed, arm, cell, ceiling, checkpoint, recipe)
> and the "margin" reservation to §4. Both new sections carry an audit of the current draft.
>
> **v3 change (2026-07-10, PM).** Added §9 (patterns from the Germano/Fabrício reviews, each rule vetted by an
> in-field adversarial review). Rulings: C1 — the literal "Read this as:" tag is banned, the lead takeaway
> sentence stays; C2 — the contributions lead-in is "we propose two enhancements". §4/§6/§7/§8 updated to match.

---

## 1 · Canonical names (commit to these; use them consistently)

| Concept | Use this | Technical term (first use only) | Never use |
|---|---|---|---|
| The "what" task | **next category** / **category prediction** | next-category | "activity", "POI classification", "next-POI category" |
| The "where" task | **next region** / **region prediction** | next-region | "area", "location", "next-POI" |
| A place | **place** / **point of interest (POI)** | POI | "venue" |
| One region unit | **region** (a **census tract**; a **mahalle** in Istanbul) | — | "cell", "zone" |
| One visit | **check-in** | — | "event" |
| Our representation | **check-in-level representation** (Check2HGI) | Check2HGI | "substrate" (repo word) |
| Place-level baseline | **place embedding (HGI)** | HGI | "the baseline" alone |
| One model, both tasks | **the joint model** / **single multi-task model** | MTL model | naming "MTLnet" before introducing it |
| One task, one model | **dedicated model** / **single-task model** | STL | "baseline" alone |
| The shared middle of the model | **the shared trunk**, introduced once as "a shared cross-attention stack (the trunk)" | cross-attention | "exchange stack" (§5.4 has one today), "backbone". Germano #31 ruled KEEP "trunk"; do not reopen. "Trunk" is also correct for other systems' shared layers (CSLSL's parallel variant) |

> **Naming rule (from the literature audit):** use **category** and **region**, never **activity** or **area**.
> "Activity" appears only when describing other papers (MCARNN, iMTL) where the field's older term was
> "next activity". "Region" is the canonical spatial term; name the unit (census tract; mahalle).

> **Keep three targets distinct:** *next category*, *next region*, and *next place* (the exact POI). The last
> submission blurred them. We predict the first two and not the exact next place; say so once, early.

> **Honesty rule (region wording, renumbered 2026-07-08 to the v17 board):** the region result is
> **mixed**: the joint model **outperforms** at **Istanbul, Florida, Texas, and California** (the 90% CI of the
> paired difference lies entirely above zero; CA/TX on the provisional seed-0 fold footing) and
> **matches** (non-inferior, TOST) at **Alabama and Arizona**. **NEVER upgrade Arizona to a gain** (+0.10, CI
> grazes zero, supplementary p=0.049 — the stats doc forbids it). CA/TX joint cells are single-seed
> provisional until A1 lands. The scaling claim ("region gain rises with region count") is scoped to the
> **five U.S. states**; Istanbul (fewest regions) is positive and sits off that trend line — never state the
> monotone claim across all six. **The superiority verb is "outperforms"** (never "beats" / "wins");
> "matches" / "non-inferior (TOST)" is the equivalence verb. Keep each verb bound to its test. The full form **"statistically
> non-inferior within a two-point margin (TOST)"** must appear **at least once in the paper** as a formal claim.
> Elsewhere — including the **abstract** and repeated
> instances — the **short forms are fine and preferred for readability**: "matches it (statistically, within two
> points)" in prose, or "(TOST, $\pm2$ pp)" after the term is defined. Never "ties" or "Pareto-dominates",
> and never "outperforms region everywhere".

> **Hyphenation:** hyphenate the compound adjective before a noun ("next-category prediction"); leave the bare
> task name open ("the next category"). Same for next-region and next-place.

---

## 2 · Acronyms (expand on first use, then short form; keep the count low)

| Acronym | Expansion | Plain meaning | First-use plan |
|---|---|---|---|
| **LBSN** | Location-Based Social Network | An app where people share the places they visit. | Expand in the abstract/intro. |
| **POI** | Point of Interest | A place a person can visit. | Expand in §1. |
| **MTL** | Multi-Task Learning | Training one model to do several jobs at once. | Expand in §1; then "the joint model" mostly. |
| **STL** | Single-Task Learning | One model per job (the usual way). | Expand once as "dedicated / single-task model". |
| **macro-F1** | balanced F1 score | A score that counts every category equally, so rare ones matter. | Gloss once. |
| **Acc@10** | accuracy at top 10 | How often the true region is among the model's top ten guesses. | Define once. |
| **CV** | cross-validation | The way we split the data to test fairly. | "5-fold cross-validation" once, then CV. |

**Introduce once, then mostly avoid (method internals):**

| Acronym | Expansion | Use note |
|---|---|---|
| **HGI** | Hierarchical Graph Infomax | The standard place embedding we build on. Expand plus one-line gloss (§2.1). |
| **DGI** | Deep Graph Infomax | HGI's ancestor. Mention once in the §2.1 primer. |
| **Check2HGI** | (our method's name) | Introduce once as "our check-in-level representation (Check2HGI)", then prefer the plain phrase. |
| **CTLE / STAN** | (baseline names) | Name once each, in related work and the baseline list. |

**Do not use in the paper (repo-internal):** `C2HGI` (write "Check2HGI"), recipe and version codenames (B9,
v11–v16, champion-G, H3-alt; write "our model"), `log_T` (write "region-transition prior"), internal finding IDs
(C25, CH16, CH19, F-numbers).

---

## 3 · Jargon to plain substitution (keep / gloss / avoid)

| Jargon | Say instead | Verdict |
|---|---|---|
| embedding / representation | **a vector that summarizes a place or visit** | gloss once, then "representation" |
| per-visit / contextual | **each visit gets its own vector** (not one fixed vector per place) | keep; this is our key idea |
| substrate | **the representation** | avoid (repo word) |
| graph | **a network of places linked by similarity, time, and distance** | keep, gloss once |
| infomax objective | **trains the vectors to tell real neighborhoods from shuffled ones** | gloss; skip the formula unless space allows |
| hierarchical | **organized as check-in → place → region → city** (four levels; ours adds the check-in level) | keep |
| multi-task / parameter sharing | **one model doing two jobs and sharing most of its parts** | keep, gloss "sharing" once |
| single-task ceiling | **the best a dedicated, one-job model reaches** | keep; "dedicated model" |
| negative transfer | **sharing hurts one task** | gloss; use the plain phrase |
| non-inferiority (TOST) | **statistically no worse than, within a stated margin** | gloss once with the 2-point margin |
| Markov / transition baseline | **a simple "what usually follows what" baseline** | keep, gloss once |
| overlapping (stride-1) windows | **overlapping windows of recent visits** | gloss once (nine visits plus the next as target) |
| transductive | **the representation was trained seeing all places** | gloss only where the leak discussion needs it |
| ablation | **a controlled test that removes one piece** | keep, gloss once |
| seed | **a repetition of the full experiment, differing only in random initialization** | gloss once in §5; then "seed" is fine in the stats. In the **abstract and intro** say **"random initialization"**, never bare "seed" ("training run" would be false: one seed = five fold-runs) |
| fold | **one of the five data splits** | covered by the "5-fold cross-validation" gloss (§2); then "fold" is fine |
| arm (joint vs. dedicated) | **both models / the joint and the dedicated model** | never; clinical-trial word, foreign to this audience |
| cell (a table result) | **result / entry** | never in prose; this audience reads "cell" as a radio cell. When the grid sense is meant, write **"grid cell"** in full (§2.2 "auxiliary coarse cell" today) |
| ceiling | **the dedicated single-task score** | keep only with the gloss "the level a joint model is expected at best to match" |
| checkpoint | **the saved model (one epoch's weights)** | prefer "saved model"; gloss if kept |
| frozen (folds / weights) | **fixed** | say "the same five fixed folds"; "frozen" only for weights, glossed once |
| recipe | **training configuration** (then "configuration") | avoid; ML-blog register |
| lift (noun) | **gain** | the noun is data-science slang; the verb "lifts" sparingly |
| epoch | **one training pass over the data** | keep, gloss once (first use is §6.1 today, unglossed) |
| end-to-end (training) | **trained whole, together with the task model** | gloss once or avoid; this audience owns "end-to-end" (a network principle) |
| from raw / from-raw | **trained from the raw check-ins** | never the bare ellipsis |

---

## 4 · Words to avoid or always explain

- **"activity" / "area"** for the two tasks: use **category** / **region**.
- **"Pareto", "Pareto-dominate"**: avoid; and recall we cannot even claim it. Say "better on one task without
  being worse on the other" only if true.
- **architecture names** ("cross-attention", "FiLM", "residual block", "transformer"): describe what they do in
  plain words; name one only if it is truly load-bearing.
- **our internal research words** ("substrate", "engine", "head", "regime", "frontier", "orthogonal gradients",
  "gate", "board", "freeze", "lane"): jargon in the paper. Say "representation", "model", "output", "setting".
  "Audit" as a self-adjective ("audited recipe") is banned self-praise (§7); the noun is fine for the §5.2 leak
  measurement. Standard terms stay standard when describing OTHER systems (HMT-GRN's next-place head, CSLSL's
  shared trunk).
- **recipe / version codenames** (B9, v11–v16, champion-G): invisible to the reader; say "our model".
- **"SOTA"**: write "state of the art" (never "SOAT").
- **"margin"**: reserved for the **TOST two-point margin**. The Part-1 representation difference is a **gap**
  (or "improvement"); never one word for both meanings. Five non-TOST uses to rename (2026-07-10): "by a wide
  margin" (abstract + §6.1), "The margin is large" (§1), "the margin isolates the representation" and
  "with margins of $+37.8$..." (§6.1).
- **"deliberately X"**: see §8.
- **"arm(s)"** and result-**"cell(s)"**: never (§3).
- **undefined metrics**: never give a number without its reference point (majority-class or Markov floor).
- **dense tables with no lead sentence**: every table and results subsection opens with a lead takeaway
  sentence, written as a normal sentence; **never the literal tag "Read this as:"** (C1 ruling, 2026-07-10).

---

## 5 · Numbers in plain language (say it once, this way)

| We report | Say it as | Reference point |
|---|---|---|
| macro-F1 (category) | "out of 100, higher is better; counts each of the 7 categories equally" | majority-class floor; HGI baseline |
| Acc@10 (region) | "how often the true next region is in the model's top 10" | Markov-1 floor |
| Δ (delta) | "the gain (or cost) versus the dedicated model" | always paired, with the margin |
| non-inferior (2-point margin) | "no worse than the dedicated model by more than 2 points, with statistical support" | state the margin every time |

---

## 6 · Consistency checklist (before submission)

- [ ] Every acronym expanded on first use; acronym count as low as possible.
- [ ] **category / region** used throughout; never "activity" / "area".
- [ ] **next category / next region / next place** kept distinct; "we do not predict the exact next place" stated once.
- [ ] Region wording (v17 board): "outperforms" at Istanbul/FL/TX/CA (CA/TX provisional) and "matches /
  non-inferior within two points" at AL/AZ — never upgrade AZ; the scaling claim scoped to the five U.S.
  states; the formal "statistically non-inferior within a two-point margin (TOST)" appears at least once;
  never "ties", "Pareto", or "outperforms region everywhere". The superiority verb is
  "outperforms", never "beats" / "wins".
- [ ] No recipe or version codenames anywhere.
- [ ] No bare "substrate / engine / head / cross-attention"; replaced or glossed.
- [ ] Every table has a lead takeaway sentence (a normal sentence; the literal "Read this as:" tag at zero).
- [ ] "state of the art", never "SOTA".
- [ ] American English throughout (behavior, modeling, neighbor, favor).
- [ ] No em-dash ("—"); use commas, parentheses, semicolons, or short sentences.
- [ ] AI-tell sweep (§7): banned words and templates at zero; "X, not Y" kept only where it scopes a claim;
  no stacked intensifiers; -ly adverb density held near 0.8% and never two in one sentence; no semicolon
  braids; no synonym-cycling.
- [ ] Idiom sweep (§8, figure captions included): no phrasal-metaphor idioms ("edges past", "buys", "ships",
  "trail", "staging", "folds in", "clears it by"); "deliberately" at zero; metaphorical "carries" ≤ 3;
  noun-"lift" at zero; "frozen folds" → "fixed folds"; "checkpoint" and "epoch" glossed at first use.
- [ ] "seed" glossed at first use; abstract and intro say "random initialization"; "arm(s)" and
  result-"cell(s)" nowhere.
- [ ] "margin" only for the TOST margin; the Part-1 difference is a "gap".
- [ ] One name for the shared component: "the shared trunk", glossed once as a cross-attention stack;
  "exchange stack" at zero (§5.4 has one today).
- [ ] §9 sweep: standard-register glosses; no ellipsis in claims; decision-critical constants carry provenance;
  digits for data quantities; commas after sentence-initial adverbials; relative pronouns written; one
  self-delta sentence in §2; the §5.2 evidence floor respected (§9.4).

---

## 7 · Do not read as machine-written (AI-text tells)

> Reviewers pattern-match for AI text by default now; CS abstracts are the most LLM-exposed corpus (~17.5%
> LLM-modified by 2024). One hit is noise; **density convicts**. (Sources: WikiProject AI Cleanup; Kobak et
> al. 2025; Liang et al. 2024; Juzek & Ward 2024; Sage editor guidance.)
>
> **Audit 2026-07-10: the draft has ZERO hits on the word/template lists and zero em-dashes.** The live risks
> are the density patterns below. All counts in this file are a dated snapshot: a re-audit replaces the
> numbers, never the rules. Re-run after every edit pass; these words creep in through AI-assisted rewrites.

**Banned words (never write → write instead):**

| Never write | Write instead |
|---|---|
| delve / delving into | examine, study |
| intricate, nuanced | complex, fine-grained; or name the specific difficulty |
| showcase, boasts | show, has, achieves |
| underscore, highlight (the importance of) | show, indicate; or state the fact directly |
| pivotal, crucial, vital | important, central (sparingly) |
| meticulous(ly), thoughtful(ly), judicious(ly) | describe the actual procedure |
| faithful, audited, rigorous, principled, careful (about our own method) | certify with the act, keeping the evidential content: "re-implemented from the authors' code under our user-disjoint protocol", "one fixed configuration, released with the code" — never bare deletion (§9.4) |
| realm, landscape, tapestry, interplay | field, literature, interaction |
| leverage, harness, unlock, foster, garner, embark | use, apply, enable, obtain, begin |
| seamless(ly), holistic, innovative, groundbreaking, unprecedented, remarkable | delete the praise; give the number |
| a testament to, stands as, serves as | shows, is |
| moreover, furthermore, additionally, notably (sentence-initial) | "also", or no connective |
| it is important / worth noting that | "Note that", or just state it |
| in conclusion | delete (the heading already says it) |
| valuable insights, advancements, surpasses | findings, advances, exceeds / outperforms |

**Banned templates:** "not only X but also Y" (split it); "this paper delves into / embarks on" (studies,
presents); "plays a crucial role in" (name the mechanism); "in today's ... world" (open with the problem);
"Firstly / Secondly / Finally" scaffolds; "a wide array of"; reader-facing meta ("let's examine"); participial
significance tails (", highlighting the importance of ..."; cut it, or promote it to a sentence with evidence).

**Density patterns (the actual current risk; audited counts in parentheses):**

- **"X, not Y" contrast** (21 in the draft, plus 10 "rather than"): our honesty device, and also a known LLM
  fingerprint ("negative parallelism"). Keep it ONLY where it scopes a claim; five are ledger-mandated and stay
  ("a match, not a gain"; "a defense of the parallel design, not a claim that we outperform the cascade"; "a
  region-native model, not a reproduction"; "a neighborhood, not a radio cell"; "motivation, not a measured
  service result"). Rewrite the decorative ones as direct statements.
- **Rule-of-three lists and scaffolds**: frequent triads read as generated, and three "First/Second/Third"
  enumerations sit in adjacent sections today (§5.2, §5.4, §7). Vary list lengths and paragraph shapes; keep
  only the item holding the evidence.
- **Booster stacking** (today: "far" ×6, "by a wide margin" ×2, "sharply" ×1): let the number carry the size;
  at most one intensifier per claim. "Significant(ly)" only with a statistical test attached.
- **-ly adverb density** (audited: 45 true adverbs in ~5,500 words, ≈0.8%, fine today): heavy -ly use is an AI
  tell, especially manner adverbs decorating verbs ("carefully designed", "seamlessly integrates"). Not a ban:
  keep density near the current level, prefer the functional ones (jointly, statistically, directly), cut the
  decorative ones (perfectly, sharply, plainly), and never two -ly adverbs in one sentence.
- **Semicolon braids** (13-14 per section in §5-§6): a sentence that needs two semicolons is two sentences.
- **Synonym-cycling**: rotating synonyms for one technical concept reads as AI and harms precision; repeat the
  exact term (§1; and the "margin" rule in §4).
- **Uniform paragraphs / wrap-ups**: vary paragraph openings (some open with a result, not a topic sentence);
  never end a section by restating it.

**Do not over-ban:** robust, novel, framework, comprehensive, baseline, state of the art are normal CS words
when load-bearing; the tell is decorative use and stacking. "Outperforms" stays; it is the mandated
superiority verb (§1).

---

## 8 · Plain international English (the non-native register rule)

> The paper must read like careful academic English its Brazilian authors would naturally write and defend
> aloud at the podium. TWO registers betray machine drafting: (a) AI-inflated vocabulary (§7) and (b)
> **native-literary idiom** — phrasal-verb metaphors, money/motion metaphors, inverted syntax — that a
> non-native author would not produce. Germano's review flagged exactly this ("swap poetic phrasing"; #31:
> "trunk ... still weird"). **The test: if you would not say it at the talk, do not write it.**
>
> Safe default verbs (Portuguese cognates or basic English): use, cost, show, obtain, reach, remain, include,
> provide, predict, measure, train, keep.

**Instances in the current draft (audited 2026-07-10) → replacement:**

| In the draft | Say instead |
|---|---|
| kept deliberately plain / deliberately conservative / deliberately different | kept simple by design / conservative by design / two settings chosen to differ |
| we state it plainly | we state it directly (or delete) |
| edges past | is slightly above |
| comes out (slightly) ahead (abstract + intro) | is (slightly) ahead |
| what the single model buys | what the single model provides |
| ships with the released code | is included in the released code |
| the harder task to pay for the easier one | the harder task's training signal to be what improves the easier one (NOT "one task's gain costs the other" — that inverts the claim the freeze control tests) |
| recent trail / check-in trail | recent visits / check-in sequence |
| staging content / content staging | caching content ahead of time ("proactive caching" is the cited paper's own term, bastug2014edge) |
| before it lands / lands the true region | before it arrives / includes the true region |
| gets there differently | but through a different construction |
| stop short of our pairing | none studies our exact pairing |
| the U.S. picture repeats | the U.S. result repeats |
| verdict (outside a test's name) | claim, conclusion ("no reported claim depends on...") |
| co-equal | rephrase around **equal standing**: "neither target is subordinate to the other" (bare "equal" collides with the disclosed 0.75/0.25 loss weights) |
| an order nothing in the tasks dictates | an order the tasks do not define |
| where it costs ("where sharing helps, where it costs") | what it costs — or name the quantity: "the trade-off between shared training and per-task accuracy" |
| on deployment grounds | because a service acts on ... (spell the reason) |
| ordinary (fixed-weight training) | standard |
| folds in the representation advantage | also includes the representation advantage (after §5 defines "fold" as a data split, "folds in" misparses) |
| clears it by 12 to 23 points | exceeds it by 12 to 23 points |
| the baselines line up the same way | the baselines show the same ordering |
| settle on a compromise | converge to a compromise |
| the interval sits above / below zero | **lies** above / below zero (one verb everywhere; the draft cycles sits/lies) |
| One captures intent, the other geography | one captures intent, the other captures geography (spell out the verb) |

- **Metaphor budget:** metaphorical **carry/carries** appears 9 times in prose ("carries the cross-attention
  stack", "gains carry 90% intervals", "carries the semantic context" — that last one is Germano-settled §7
  wording; keep it). Keep 2-3 overall; prefer has, holds, includes, uses.
- The bar is double: **would the authors say it, AND would the community write it.** Both of this note's
  earlier protected examples ("sharing is not free", "One model is enough") failed the second test with real
  readers (2026-07-10 reviews): plain-but-blunt fails like fancy-but-foreign. The target is the band between,
  standard academic English with short glosses (§9.1).

---

## 9 · Patterns from the co-author reviews (2026-07-10, vetted in-field)

> Distilled from Germano's PDF annotations and Fabrício's Overleaf review, each rule vetted by an in-field
> adversarial review. Their clarity instincts were right for this venue about 80% of the time; their evidence
> cuts were wrong. So: adopt the clarity, never the evidence cuts (§9.4).

**9.1 Register (plain ≠ informal; the target is standard academic):**

- Gloss with standard words, never folksy: "test" not "ask" (as the hypothesis verb), "build" not "form",
  "transforms" not "turns", "the place category" not "the kind of place", "standard" not "ordinary".
- No clipped declarative openers ("One model is enough."). Open sections with a complete sentence whose
  **main clause is still the claim** (no throat-clearing either).
- Contributions lead-in: **"we propose two enhancements"** (C2 ruling; never "changes", never "novelties").
  Compensate the modest noun inside the bullets: the novelty statement ("the first, to our knowledge, to treat
  fine-grained region as an end target of equal standing") belongs there, and §2.2 supports it.
- "on X grounds" → "because ...". Cooking metaphors out: "ingredient" → component ("recipe" already in §3).

**9.2 Clarity mechanics (every reviewer "não entendi" traced to one of these):**

- No elliptical constructions in claims: name the quantity ("the trade-off between shared training and
  per-task accuracy", not "what sharing costs"); repeat the noun ("at small region counts ... at large region
  counts"); resolve pairs explicitly ("the first task ... the second task"). Fix per instance; parallel
  structure itself is not banned.
- No dangling deictics ("planning capacity there"): name the referent.
- **Decision-critical constants declare provenance at first mention** (the 0.75/0.25 loss weights, the 2-pp
  TOST margin, the 9-visit window): chosen in advance vs. tuned, and on what. §2.3 already says the weighting
  was TUNED, so §4.2 must match ("tuned once on validation during development, held fixed across all six
  datasets"). Everything else gets one blanket sentence ("all other hyperparameters were fixed during
  development; the full configuration is released with the code"). The tuning asymmetry favors us: the
  dedicated ceilings are per-dataset tuned, the joint model is not — say so.
- Statistics: **cite Holm and TOST** (one bib line each); do **not** gloss paired t / Wilcoxon — over-glossing
  textbook tests reads as unfamiliarity with our own tools. TOST keeps its existing short gloss.
- **Digits for data quantities** ("520 to 8,501 regions (Table I)"); words only for small counts (two tasks,
  seven categories). IEEE style; spelled-out large numbers read as evasive.
- Comma after a sentence-initial adverbial phrase ("Across the five U.S. states, ...").
- Write the relative pronoun ("the head **that** we do not predict"). Zero contractions; **"cannot" is correct
  formal English** — never "fix" it.
- One coined term paper-wide (§7 synonym rule), BUT restructure any paragraph repeating it 4+ times
  ("stream" ×6 drew "too much repetition"): fix with sentence structure, never with synonyms.
- Standalone "+0.00" → "no change (0.00)". Inside interval notation ("+0.00 to +0.21") the number stays.
- Section titles plain and descriptive: "Metrics and Statistical Tests", not "Metrics, superiority versus
  non-inferiority" (and not bare "Metrics", which under-describes).

**9.3 Rhetoric and citation posture:**

- Self-citation (single-blind: the reviewer sees our names anyway): **one** self-positioning sentence in §2
  stating the delta over our prior work ("[7] established the two-task setup and observed negative transfer;
  this paper adds the check-in-level representation and shows the transfer reverses on it"); everywhere else
  cite jointly [6,7]; never the sentence subject in the intro. An unstated delta reads as "incremental over
  the authors' own prior work".
- Novelty defusals: the intro states the contribution positively once (its "not new on its own" aside is cut);
  §2's named-system comparisons (DRRGNN, KGTB, HAMTL, the cascade lineage) are ledger-mandated positioning and
  STAY.
- **Cost claims name both artifacts and the direction in one clause**: "the joint model (4.2 million parameters
  at Alabama) against the two dedicated models combined (1.1 million)". (Germano misread the current sentence —
  proof the rule is needed.)
- Describe cited systems as their authors describe them (DGI is a general graph self-supervised method, not a
  "place embedding model"). One accuracy check per citation.
- Universal quantifiers → named scope ("on Istanbul and the five U.S. states"); "at every dataset" only right
  after the six are enumerated; bare "everywhere" never.
- Problem section describes the problem only, zero solution mechanics (no window sizes, no models). Abstract
  leads with context and motivation, results compressed but never numberless.
- §4.2 shows the training loss as one displayed equation ($L = 0.75\,L_{cat} + 0.25\,L_{reg}$, unweighted
  cross-entropy); the infomax objective stays a citation.

**9.4 Evidence guard (where the co-author instinct was WRONG; never follow):**

- **NEVER cut the §5.2 leak audit to "assume no leakage"**: the previous submission was rejected partly on a
  leakage accusation, and this venue penalizes real traces without a leakage protocol. Compress for clarity
  (target ~60% length, plain sentences, one measured number per ground), never below the floor: the three
  grounds + the audit numbers + the per-fold prior construction.
- **NEVER delete a fairness signal without an act-replacement** (§7 self-certifying row): a baseline-heavy
  paper whose baseline descriptions carry no provenance reads as "probably crippled baselines".
- **NEVER weaken a measured finding into speculation** (the balancer confirmation stays a finding, not "a
  hypothesis"); never delete true numbers (the "+9"); never adopt puffery from review margins ("achieves
  superior performance" is §7-banned).
