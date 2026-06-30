# Review — Germano Barcelos (markup on `_MOBIWAC_.pdf`)

Internal co-author review pass extracted from the PDF highlight/annotation layer of
`_MOBIWAC_.pdf` (paper: *"Predicting the Next Category and Region of a Visit: A Check-in-Level
Multi-Task Study on Mobility Data"*). **70 highlight comments**, all by **Germano Barcelos**.

Each entry shows the **exact highlighted span** (recovered via word-level bounding boxes) and
the reviewer's **verbatim** comment (mostly Portuguese), plus an English **Action** gloss.

Recurring themes: (1) swap poetic phrasing for standard ML jargon; (2) soften/justify novelty
claims; (3) move the MTL definition earlier and keep Related Work about *others'* work; (4) add
citations to strong claims; (5) check the references aren't outdated.

> **Co-author response (added 2026-06-27, advisor-reviewed).** A multi-agent pass evaluated all 70 comments
> against the current draft and our prior decisions, with two advisors checking for missing interpretations and
> conflicts. Verdicts: **Accept 29, Partial 29, Reject 12.** Each comment below carries a **Response** (verdict +
> rationale) and the **Edit** to make. Rejections push back honestly where a comment is taste-only, factually
> wrong, or conflicts with a decision already made (e.g. the softened abstract, the plain-words audience).

---

## ⭐ Final decision pass (v2 — author-reviewed, 2026-06-28)

> **What this pass is.** The 2026-06-27 multi-agent pass (the "Response/Edit" blocks below) answered all 70
> comments but **(a) overlooked several `GLOSSARY.md` rules** (most importantly the no-em-dash rule, which its own
> proposed edits then re-introduced, and the "name CTLE once, in Related Work" rule the author flagged at #29), and
> **(b) pre-dated the author's inline considerations** (the `AUTHOUR/Authour/Authors` lines added later). This
> section is the **authoritative ruling**: it re-checks every author-annotated item against `GLOSSARY.md` + the
> `CLAUDE.md §3` decisions ledger, records the final verdict, and routes the broad-scope items to
> [`review/`](review/). The v1 Response/Edit blocks below stay as **provenance** — where this section and a v1 block
> disagree, **this section wins**.
>
> **Nothing here is applied to the paper yet** (per the author's instruction). This is the worklist; the prose pass
> and the [`review/`](review/) investigations come after.

### Cross-cutting rulings (apply to every edit before it reaches `src/`)

- **CC1 — No-em-dash sweep (GLOSSARY law).** The v1 edits repeatedly use the dash-as-parenthetical construction
  (`--`) the glossary bans — at **#29, #61, #62, #64, #66**. Every one must be rewritten with commas / parentheses
  / semicolons before it ships. **Also a LIVE violation already in the paper:** `06_results.tex:57-58` uses a real
  em-dash (`---so called...alone---and`). Fix it in the same sweep. (Tracked as the em-dash arm of OP5.)
- **CC2 — CTLE acronym discipline (author #29; GLOSSARY §2: "CTLE/STAN: name once each, in related work").**
  Never name CTLE (or cite it) in the **Introduction**. The intro refers generically ("sequence-model
  embeddings"); CTLE is named + cited only in §2.1 (where it already is) and the baseline list. This overrides the
  v1 #29 edit, which injected `CTLE~\cite{lin2021ctle}` into §1.
- **CC3 — Superiority verb = author override of the ledger (author #8, #10, #35).** The `CLAUDE.md §3` ledger says
  "do NOT swap 'beats' to 'outperforms'." The **author has now overridden that.** Final ruling: adopt **one**
  superiority verb **globally** — recommend **"outperforms"** (do not also introduce "exceeds in performance" from
  #10; one verb only). Keep **"matches" / "non-inferior (TOST)"** for the equivalence cells (do NOT change those).
  This requires a coordinated sweep: replace "beats" → "outperforms" in abstract + §1 + §6 + §7 + §8 + the Table III
  caption, and update the §5.3 definition sentence so the verb still maps to the paired-Wilcoxon superiority test.
  **➜ Update `CLAUDE.md §3` and `GLOSSARY.md` once applied.** (Author: confirm the verb is "outperforms" and that
  the scope is global, not abstract-only — abstract-only would re-fragment the vocabulary, which is the one outcome
  to avoid.)
- **CC4 — "coarser" overuse (author #18, #48, and the #55.1 thread).** "coarse/coarser" appears **8×** (abstract +
  5 sections). Reduce to **one glossed use**, then drop the word. On the author's "simpler instead of coarser?"
  (#18): **recommend against "simpler"** — it amplifies exactly the "you just made the problem easier" attack that
  #48 rebuts. Use **"coarser (less detailed)"** once at first use; elsewhere say "higher-level" / "lower-resolution"
  / name the targets directly. Keep the #48 *rebuttal* (it is load-bearing) but reword it so it does not pile the
  word back on.
- **CC5 — "we do not predict the exact next place" stated once (GLOSSARY: "say so once, early"; author #21, #55.1).**
  The scope statement appears **3×**: §1 intro (`01_introduction.tex:26`), §2 related ("We set aside the exact next
  place", `02_related.tex:44`), §3 problem (`03_problem.tex:16`). Keep **exactly one** — recommend the **§3
  Problem** instance (the formal task-definition home) and delete the intro (#21, author agrees) and the related-work
  (#47) instances. (The "Predicting the exact next place ... is hard" *motivation* line at `01_introduction.tex:22`
  is a different sentence — keep it.)

### Index of changed verdicts and author decisions

| # | v1 verdict | Author note? | FINAL | One-line reason |
|---|---|---|---|---|
| 8 | Partial (keep "beats") | "Replace to outperforms" | **Accept author** | CC3 — outperforms globally |
| 10 | Partial (keep "beats") | "Exceeds in performance" | **Accept author (verb=outperforms)** | CC3 — one verb only, not two |
| 14 | Partial (rewrite) | "keep the current" | **Accept author** | current line is honest + glossary-clean |
| 15 | Partial (reword) | "[REJECT] keep as it is" | **Accept author** | keep original hook |
| 18 | Partial (gloss) | "simpler instead?" | **Partial (push back on 'simpler')** | CC4 — "simpler" feeds the #48 attack |
| 21 | Partial (fold in) | "We can remove it" | **Accept author (remove)** | CC5 — keep the §3 instance only |
| 29 | Partial (add CTLE clause) | "not following glossary" | **Revised** | CC2 — keep intent, drop CTLE from intro |
| 31 | Partial (keep "trunk") | "still weird" | **Partial + flag** | drop "semantic" modifier; trunk-wording consistency pass |
| 37 | Accept (expand pp) | "pp is common, no?" | **Keep v1 (expand once)** | cheap insurance for networking readers |
| 38 | Accept (gloss) | offers alt definition | **Keep v1 — correct the author** | raw identifiers = bare place ID, not features |
| 46 | Partial (light copyedit) | "split into new para" | **Accept author** | split for clarity, keep all framing |
| 48 | Accept (add rebuttal) | "watch 'coarser'" | **Keep rebuttal + CC4** | reword to not repeat the word |
| 50 | Partial (reframe) | "search web/lit" | **Interim + route INV1** | soften to "to our knowledge" until verified |
| 54 | Reject (keep line) | "keep, but surface?" | **Keep line; goal already met** | abstract/intro already foreground check-ins |
| 55.1 | (author item) | repetition x3 | **CC5** | keep one scope statement |
| 57 | Partial (no cite now) | "keep pending, I'll fix" | **Pending (author-owned)** | track in LOG/INV6 |
| 59 | Reject (keep) | "add a gloss" | **Accept author (add gloss)** | gloss "radio cell"; merge with #60 |
| 60 | Partial (split) | "confusing, eval" | **Accept (split + #59 gloss)** | one merged rewrite |
| 64 | Accept (gloss) | "did we explain MTL?" | **Keep v1 + route INV5** | line 35 expands but never glosses MTL |
| 66.1 | (author item) | Acc@10 / stratification | **Soft-spot + route INV3** | metric-alignment argument is imprecise |
| 67 | Partial (relocate) | "comprove in codebase" | **Keep + route INV2** | +5% must be measured before it ships |
| 69 | Accept (lift numbers) | methodology/axis/name | **Accept prose + route INV4** | figure design needs the .py |

### Per-item final rulings (annotated / changed items)

**#8, #35 — FINAL: Accept the author override (CC3).** Use **"outperforms"** for category superiority, applied
globally and consistently, with "matches/non-inferior (TOST)" untouched. Keep the v1 anti-triple-`beats`
restructuring of the abstract, just with the new verb. Update the ledger.

**#10 — FINAL: Accept the spirit, but use the CC3 verb, not a second one.** The author wrote "exceeds in
performance" here and "outperforms" at #8; introducing both re-creates the fragmentation we are trying to remove.
Pick **one** ("outperforms" recommended). Keep the v1 noun fix from #11 ("where there are many regions", not "region
space is large" / not "region area").

**#14 — FINAL: Accept the author — keep the current Istanbul abstract sentence.** It is honest and glossary-clean
("beats on category and stays within two points on region"). Reject Germano's "superior performance on both"
(region is a TOST *match*, not a beat — claiming superiority is the overclaim a reviewer flags). Under CC3 the verb
becomes "outperforms on category".

**#15 — FINAL: Accept the author — keep the original hook.** No edit. (The v1 reword was a taste call; the author
declined it.)

**#18 — FINAL: Partial, pushing back on "simpler" (CC4).** Replace the first "coarser" with **"coarser (less
detailed)"** as the single glossed use, then remove the word elsewhere. Do **not** switch to "simpler": it makes the
targets sound like a shortcut and strengthens the #48 "you simplified the problem" line of attack rather than
defusing it. (Author: if you still prefer "simpler", flag it and we apply it everywhere — but the recommendation is
"coarser (less detailed)" once + the #48 rebuttal.)

**#21 + #55.1 — FINAL: Accept the author — remove, keeping one canonical scope statement (CC5).** Delete the intro
instance (#21) and the related-work "We set aside the exact next place" (#47); keep the **§3 Problem** instance as
the single statement of scope. This resolves the "said it 3× already" concern (#55.1) in one move.

**#29 — FINAL: Revised per the author + CC2.** Keep the differentiation the v1 edit was after (per-visit context is
not ours), but **without naming CTLE in the intro**. Suggested intro clause: *"Per-visit context is not new on its
own; what is new is obtaining it from inside the hierarchical-graph representation, which we test directly
(Section II)."* The CTLE name + cite + the direct comparison stay in §2.1 and §6.1, where they already live. This is
the glossary-compliant version of the v1 intent.

**#31 — FINAL: Partial + flag for a consistency pass.** Drop the "semantic trunk" compound (it reads poetic) and the
"semantic" modifier on "trunk"; carry the semantic-vs-spatial contrast with the surrounding words, e.g. *"a shared
trunk that both tasks use, with a private spatial path for the region task."* Keep the bare word "trunk" (the body
uses "shared trunk" throughout; "backbone" is worse — it collides with "network backbone"). Because the author is
still unsatisfied, add a **trunk-wording consistency check** to OP4 (make intro/§4/§6/§7 agree on exactly one phrasing).

**#37 — FINAL: Keep v1 (expand once).** Yes, "pp" is standard in stats/ML, but the MobiWac audience is networking,
where it is less universal, so a single first-use expansion ("percentage points (pp)") at `01_introduction.tex:71`
is cheap insurance; use "pp" thereafter. The softened abstract stays in plain "points" (no "pp").

**#38 — FINAL: Keep the v1 gloss; the author's alternative is incorrect.** "raw identifiers" here means **the bare
place ID** (an arbitrary index that carries no inherent meaning), which is the standard motivation for learning
embeddings (reason about places by learned geometry instead of meaningless IDs). It does **not** mean "raw category,
location and time" — those are *features* the graph encodes, not identifiers, so using them as the gloss would be
wrong and would confuse the reader. Keep: *"...rather than by raw identifiers, the arbitrary ID numbers that carry no
meaning on their own."*

**#46 — FINAL: Accept the author's structural fix.** Split the dense combination-novelty paragraph in §2.1 into two:
(1) the *"which features vs how often a vector is produced"* contrast; (2) *"the novelty is the combination"* + the
direct CTLE comparison commitment. **Keep all the content and the novelty scoping** (do not weaken it — it is the
BRACIS defense); the improvement is readability, not deletion. Fold in the #40 (`emit`→`produce`) and #45
(name the antecedents DGI/HGI) copyedits.

**#48 — FINAL: Keep the rebuttal, reword per CC4.** The "coarser is not trivial: region spans hundreds to several
thousand classes; we choose these targets because a service acts on them, not to make the task easier" rebuttal is
the strongest answer in the batch — keep it. But reword so it does not add three more "coarser"s (CC4): e.g.
*"These targets are not trivial: region alone spans hundreds to several thousand classes. We choose them because they
are what a mobility-aware service can act on, not to make the task easier."*

**#50 — FINAL: Interim wording now, verify later (➜ INV1).** Keep the v1 reframing ("fine-grained region as an end
target, rather than an auxiliary coarse cell, has little precedent"), but until the literature is actually checked,
soften the novelty claim to **"to our knowledge, ... is underexplored."** Web/literature verification is **INV1** in
[`review/review_v2.md`](review/review_v2.md).

**#54 — FINAL: Keep the §3 line; the author's goal is already met.** "We work with check-in sequences" correctly
anchors the windowing sentence that follows, so it stays in §3. The author's aim (surface check-in-sequence work to
attract that audience) is **already served** by the abstract ("one check-in at a time") and §1 ("each check-in is a
small record..."), which foreground the check-in framing up front. No move needed; optional one-phrase strengthening
in §1 only if desired.

**#57 — FINAL: Pending, author-owned (➜ INV6).** No prose change now. The motivation sentence stays as motivation;
if a real mobility-aware-service reference is found it attaches to the **concrete examples** (content staging /
capacity planning), never the framing sentence, with a `%verified` note. Tracked in [`review/LOG.md`](review/LOG.md).

**#59 + #60 — FINAL: Accept the author — keep "radio cell" but gloss it, and split the run-on.** One merged rewrite,
e.g.: *"These examples are sized to what we actually measure. A census tract is a neighborhood, not a radio cell (the
small coverage area a single base station serves); even the ten most likely tracts are far too coarse to drive cell
association or handover. We therefore keep the motivation at the regional level: demand and load anticipation,
content staging, and capacity planning."* The gloss answers the author's #59 ("why are we even mentioning a radio
cell?") for both audiences; the split answers #60.

**#64 + #26 — FINAL: Keep the v1 glosses, but they must actually *define* MTL (➜ INV5), and apply CC1.** The author
is right that MTL is not currently explained: `01_introduction.tex:35` only *expands the acronym*
("multi-task learning (MTL)") with no plain-words gloss of what MTL is. So #26 must add a real gloss ("training one
model to do several jobs at once, sharing most of its parts") and #64 glosses "shared trunk" / "hard parameter
sharing" at first use — **with commas/parens, not the `--` the v1 edit used (CC1).** Whether the definition then
lands cleanly and exactly once is verified by **INV5** (folds into the OP3 gloss audit).

**#66.1 — FINAL: Real methodological soft spot — soften the prose, route the verification (➜ INV3).** The author is
right: **Acc@10 (region) is instance-weighted, not class-balanced**, so the §4.2 symmetry argument ("our metric
counts every class equally, so we don't reweight") does not transfer to region — and, more sharply, **unweighted CE
on *category* does not actually align with macro-F1** (which *is* class-balanced), so the current justification is
imprecise and a reviewer could press it. Until INV3 settles it in the codebase, **recommend replacing the
metric-alignment argument with the empirically-true one** ("a fixed unweighted sum balanced the two tasks best in our
experiments"), and add a sentence on the category class distribution + that folds are stratified. INV3 (class
distribution, stratification, what Acc@10 weights, whether class-weighting category helps macro-F1) is in
[`review/review_v2.md`](review/review_v2.md).

**#67 — FINAL: Keep the relocation; the +5% number must be measured before it ships (➜ INV2).** The v1 move (cost
paragraph after the single-model paragraph) is right and the +5% is a genuine edge-audience selling point, but it is
currently an unverified figure. **INV2** measures params + FLOPs of the joint `mtlnet_crossattn_dualtower` vs a
dedicated single-task model (the repo has `src/utils/flops.py`). Mark "+5%" provisional in [`review/LOG.md`](review/LOG.md)
until INV2 returns a number; the "two answers at the price of one" line depends on it.

**#69 — FINAL: Accept the prose/caption fixes now; route the figure-design questions (➜ INV4).** Lift the numbers
into the caption (v1) **and** fix the methodology framing the author flagged: make it explicit that **silhouette**
(separation of the seven *ground-truth-labeled* groups) and **nearest-neighbor category purity** (a kNN-style
separability measure) are two **separability metrics, not discovered clusters** — never imply k-means. On "is
'Category separability' the clearest name?" and "do the two panels belong on one Y axis?": these need the actual
plot, so they go to **INV4** (inspect + possibly regenerate `figs/fig3_embquality.py`). Until INV4, the prose stays
metric-accurate and clustering-free.

### The broad "other points" (bottom of this file) → [`review/`](review/)

The author's `Other points to review` list (orthogonal transfer ↔ no gradient balancers; repeated abbreviation
expansions; un-glossed technical concepts; cross-chapter cohesion / repeated concepts; a systematic English pass) is
**not** per-comment work — it needs codebase/doc scrapes and section-spanning sweeps. It is catalogued, scoped, and
sequenced in **[`review/review_v2.md`](review/review_v2.md)** (tasks OP1–OP5 + INV1–INV6), with the attack plan in
**[`review/PLAN.md`](review/PLAN.md)** and the running cross-reference ledger in **[`review/LOG.md`](review/LOG.md)**.

### Evidence-backed update (agents + advisors + critics, 2026-06-28)

> A workflow then **investigated each author note against its real source** (codebase / docs / web / paper), an
> adversarial **advisor** checked every verdict against `GLOSSARY.md` + `PAPER_PLAN.md §3`, and two completeness
> **critics** audited the rulings above. **Where this subsection differs from the solo rulings above, this wins.**
> Two facts changed a verdict materially: **#67 (the "+5%" is false)** and **#66.1 (the loss justification is
> backwards)**. Two author notes the solo pass had wrong-footed are corrected: **#38** is now backed by code, and
> **CC3** is downgraded from FINAL to PENDING.

**Corrections to the cross-cutting rulings**

- **CC1 (+):** the dash sweep must cover **every *accepted* edit's replacement text, not only the v1 originals** —
  #64's and **#66's** accepted edits still contain `--`. (Add an explicit #66 line: rewrite "deployment property we
  care about `--` a service..." with a comma.) Live paper violation stands: `06_results.tex:57-58`.
- **CC3 → DOWNGRADED to PENDING author confirmation (do NOT file as FINAL).** The critics are right on two counts.
  (a) The author wrote *two* verbs (#8 "outperforms", #10 "exceeds in performance") and **#35 has no author note**,
  so the override is not yet one clean instruction. (b) The sweep was under-scoped: the verb family is
  **{beats, beat, wins, win, "spatial win", "spatial gain"}**, and the sentence that actually binds the verb to the
  test is `05_setup.tex:68` ("Where the joint model is **meant to win**, we test superiority") — it says *win*, not
  *beats*. Table III's caption carries **both** "Bold marks a **win**" and "$\uparrow$ a **beat**". A "beats →
  outperforms" swap that ignores "win" leaves *three* superiority words (worse than today). **Exclude
  `02_related.tex:73`** ("such balancers rarely **beat** a tuned fixed weighting" — generic, about other methods).
  **Preserve the abstract anti-triple** (do not mint three verbatim "outperforms"). **Ordering:** GLOSSARY §1/§6 +
  *both* ledgers (repo-root `/CLAUDE.md` and `[mobiwac]/CLAUDE.md §3`) change **atomically with/before** the prose,
  not after. ➜ *Confirm: single verb = "outperforms"? scope = global?*
  **✅ APPLIED 2026-06-28 — author confirmed "outperforms", global.** Law swept (`GLOSSARY.md` honesty rule + §6
  checklist; `[mobiwac]/CLAUDE.md §3` Verdict-verb row + §2 narrative) and all prose swept (abstract, §1, the §5.3
  test-binding sentence, §6 ×12, §7, §8, Table III caption); the generic `02_related:73` "beat" → "improve on"; the
  abstract anti-triple is preserved. Remaining beats/wins as a verdict verb in rendered text: **0**. See LOG R1.
- **CC4 → arbitration (it contradicted #1).** "Coarse" is **kept in the abstract** (#1 rejected removal as
  load-bearing), so CC4 cannot also demand "one glossed use at first use" (first use *is* the abstract). Realistic
  budget: **at most 2–3 total** — the abstract's "coarse" (#1, the canonical instance) + one glossed "coarser (less
  detailed)" in §1 (#18). Reword the rest off the word: #48 rebuttal → "these targets are **not trivial**…";
  #59/#60 → "the ten most likely tracts are **too broad**…". Bring **#2's** accepted abstract rewrite (it adds
  "coarse") under this budget too.
- **Index additions:** **#26** (define MTL — author reopened a v1 Accept) and **#47** (the §2 gap statement + the
  CC5 deletion of `02_related.tex:44`) are decision-bearing and get their own rows. *CC5 check passed:* the §2 scope
  line `:44` ("We set aside the exact next place") is a separate sentence from the §2 gap statement `:40-42`, so
  deleting `:44` does not strand the gap; and POI's first expansion at `01_introduction.tex:22` sits inside the
  **kept** motivation sentence, so the #21/CC5 deletions are safe.

**Evidence-backed per-item verdicts** (file:line = where the evidence lives)

| # | Author | Evidence-backed finding | FINAL |
|---|---|---|---|
| 38 | wrong | Check2HGI node features = category one-hot + temporal sin/cos, keyed by `placeid`; category/time/distance are graph FEATURES (inputs); the embedding replaces the bare place ID (`research/embeddings/check2hgi/README.md`, `04_method.tex:22-23`) | keep v1 substance, **author incorrect**; name the place ID concretely, **drop "one-hot"** jargon, use short "POI" (fixes an acronym-once breach at `02_related.tex:11`), and disambiguate against the feature reading |
| 50 | partly | coarse-cell-as-main-target IS precedented (TrajLearn, ACM TSAS 2025; geohash/grid next-cell; cellular next-cell); the **owned** combination (fine admin unit + co-equal with category + no next-POI) is underexplored (INV1 web, done) | use the **method contrast** (auxiliary coarse cell vs co-equal end target); soften any precedence claim to "to our knowledge … underexplored"; **do NOT add TrajLearn to the bib** (locked at 26); note "underexplored" is outside `PAPER_PLAN §3` CAN-say — add it there with the hedge or keep only the contrast |
| 64/26 | correct | no plain-words MTL definition exists; `01:35` only expands the acronym; "shared trunk" never glossed (4 reuses) | add an MTL gloss at `01:35` + a "shared trunk" gloss at `01:46`, **exactly once**; don't touch `04_method:65`; commas not dashes (CC1) |
| 66.1 | correct | region Acc@10 is **instance-level / frequency-weighted** (`mtl_eval.py:57-61`), so the §4.2 "macro-F1 counts every class equally → unweighted" logic is **backwards for category** (macro-F1 is balanced; unweighted CE is the *misaligned* choice there). BUT both heads ARE unweighted in the board recipe (`--no-{reg,cat}-class-weights`), so the *factual* claim stands; only the *reason* is wrong. Honest defense is **empirical** (C25: class-weighting tested, hurt both — region −10..14pp, category macro-F1 +5.14pp weighted→unweighted at AL, `docs/CONCERNS.md:590`) | **author correct**; replace the metric-alignment sentence with the per-task + empirical version, kept **qualitative** (C25 ran on champion-G non-overlap, not the board cells — don't print the pp); add the stratification sentence (StratifiedGroupKFold, user-grouped, category-stratified); a per-state category-distribution table is optional + currently unsourced (light groupby) → INV3 |
| 67 | wrong | "+5%" is **false and live in the paper**: vs ONE dedicated model the joint is **+38% to +90% params / +22% to +74% FLOPs**; the "+5%-scale" only holds vs TWO models, where the joint is *cheaper* (~−28% FLOPs, −17..20% params) (INV2, CPU param count) | **author right to doubt it**; **Option A**: drop the magnitude, keep "cheaper than two dedicated models"; change "two answers at the price of **one**" → "well under the price of **two**"; fix `04_method:5` header, `PAPER_PLAN:393/745`, and the v1 #67 rationale (it called +5% "a selling point" — falsified) |
| 69 | partly | no clustering bug — Fig 3 is a grouped bar chart of **precomputed** silhouette + kNN-LOO purity on **ground-truth** labels (`fig3_embquality.py:38-41`, `geometry.py:46-94,169`); prose already correct (`06:31,33`) | **cosmetic only**: relabel the y-axis (two metrics on one 0–1 scale), rename the kNN tick "kNN purity (k=10)", add a "no clustering" caption clause; **regenerate the PDF**; **hold the numeric caption lift inside INV4** (numbers must match the .py) |
| 37 | partly | "pp" collides with "pp."=pages, used **18× in this paper's own `main.bbl`**; bare-and-undefined is the weakest option | gloss "pp" once at `01:71` **or** write "percentage points"/"points" throughout (medium-confidence polish, not a correctness bug); leave the non-rendered `tbl3:9` comment |
| 31 | partly | only one "semantic trunk" compound exists (`01:46`); 4/5 shared-layer sites already say "shared trunk"; grounded by the `caruana1997` gloss | single intro edit → "a shared trunk that carries the semantic context"; **satisfies the OP4 trunk pass**; do NOT touch `05_setup:89` "HMT-GRN-style trunk" |
| 8/10/35 | (CC3) | "win/wins" family + `05_setup:68` "meant to win" + Table III dual legend must move together | **PENDING author confirm** (see CC3) |
| 54 | n/a | check-in framing already foregrounded (`main.tex:56` "one check-in at a time", `01:13`); §3 line correctly stays | keep the §3 line; optional one-phrase echo of "check-in sequences" earlier; the author's goal is already met |

**Critic-driven fixes to the taste rulings**

- **#14:** no contradiction — keep the current Istanbul sentence; it is a **CC3 exception** unless/until CC3 is
  confirmed global (then the verb applies there too).
- **#15:** the author's [REJECT] keeps a sentence the reviewer found *confusing* ("o que quis dizer aqui?"), so
  "keep" ships a flagged line. Honor the keep on content, but **route a minimal clarity check to OP3/OP5** — don't
  close it as done.
- **#29:** confirm the revised intro clause uses a **semicolon, not `--`** (CC1) and introduces **neither bare "HGI"
  nor un-glossed "infomax"** in the intro (per #30).
- **#46:** accept the split, but **carry the v1 guard**: it must not read verbatim with the intro (#29) and Method
  (#61) novelty sentences.
- **#48:** keep the rebuttal, reworded off "coarser" (CC4).
- **#59/#60:** keep the radio-cell gloss + split; note "radio cell" is the **networking** cell (allowed), distinct
  from the GLOSSARY-banned region-"cell"; **route the §3-vs-§5.3 "tracts can't drive handover" redundancy to OP4**.

---

## Germano-lens proactive audit, §4–§8 (agents + advisors, 2026-06-29)

> **Why this exists.** Germano scrutinized the Abstract/Intro densely (35 comments) but thinned out later: §4 Method
> got 7, §6 Results got 2, and **§5 Setup, §7 Discussion, §8 Conclusion got ZERO**. We distilled his 70 comments
> into a **12-point pitfall lens** (G1 poetic phrasing · G2 un-glossed term · G3 referential ambiguity · G4 novelty
> over-claim · G5 uncited claim · G6 section discipline · G7 redundancy · G8 confusing/run-on · G9 misplaced · G10
> methodology/fairness · G11 factual · G12 grammar/connective) and ran one auditor per chapter (§4→§8 + its
> floats), each finding adversarially verified against GLOSSARY + claim-discipline. Only **verified** findings are
> listed; false positives the advisor caught are noted at the end. CC3 ("outperforms") is already applied, so the
> verb is not re-flagged.

### ⭐ Two cross-chapter sweeps — accepted Germano fixes were applied only at the flagged line

- **CC6 — "spatial gain" / "spatial benefit" → "the gain on region" (paper-wide).** Germano's #13 ("spatial gain?
  rephrase") was **accepted** but only fixed where he flagged it; the coined phrase still appears at **abstract
  `main.tex:70`, §6 `06:51`, §7 `07:17` ("the spatial benefit of sharing"), §8 `08:20`.** Apply #13 everywhere.
- **CC7 — "region space" / "space of regions" → "number of regions" / "many regions" (paper-wide).** Germano's #11
  was **accepted** but the term still appears at **§1 `01:65`, §5 `05:18`, §6 `06:51/63/67`, §7 `07:14/16-17`, §8
  `08:14`.** Apply #11 everywhere. *(Process lesson: every accepted Germano edit needs a paper-wide sweep, not a
  single-line patch.)*
- **CC3 residue:** `07:18` now reads **"meets or outperforms"** — "meets" is an unbound third verb; → **"matches or
  outperforms"** (completes CC3; "meets" ≈ "matches" but is not the bound verb).

### High-severity (a reviewer would likely object)

- **§6 `06:53` "majority-class floor of about 7%" — NOT an error (audit over-flag; corrected by the final advisor).**
  Computed: majority class Food ≈27.8% → its F1 = 2·0.278/1.278 = 0.435, so macro-F1 of a majority-class predictor =
  0.435/7 ≈ **6.2% ≈ "about 7%"**. The paper is correct, and the §5.3 floor definition agrees. The audit's "far below
  7%" reasoning was wrong (it ignored the majority class's own F1), and the proposed "one in seven" (14%) fix would
  inject an *accuracy* chance rate as a *macro-F1* reference (a metric-mix). **No fix needed; optional precision tweak
  "about 7%" → "about 6%".** (This is the one audit item that, if applied, would have made the paper worse.)
- **§4 `04:61-64` (G11) = the #66.1 finding, independently re-confirmed:** the macro-F1 "metric-alignment"
  justification for unweighted CE is backwards. Replace with the per-task + empirical (qualitative) version.
- **§6 `06:40` (G10) CTLE fairness:** the decisive "CTLE 29.69–33.45 vs our 75.15" single-state claim states no
  matched protocol. **Fix:** add "under the same overlapping windowing and the same single-task head" (verified true
  against the JSON, so no overclaim).
- **§5 `05:53` (G2) S-1:** "region-transition prior" (and "spatial path") appears naked in the leak paragraph —
  gloss it once ("a table, built from training data, of how often visits move from one region to the next, used to
  bias the region scores").
- **§5 tbl1 caption (G5/G11) S-8:** "the standard next-POI sparsity measure" is an uncited "field-norm" claim, and
  density vs sparsity are conflated (column says Density, caption says sparsity). Fix the wording, drop/cite "standard".
- **§7 `07:31` (G1) D-1:** "We are honest about three limits" reopens the over-apologetic register Germano killed at
  #61. → "Three limits qualify these results." (keep all three limits).

### Per-chapter verified findings (medium/low)

**§4 Method**
| ID | sev | pitfall | loc | fix |
|---|---|---|---|---|
| A-M2 | low | G2 | Fig.2 caption `main.tex:101` | gloss "cross-attention" once in the caption (body already glossed) |
| A-M4 | med | G2/G11 | Fig.2 caption `main.tex:102-104` | "GRU head / dual-tower / private spatial **tower**" are unglossed + mismatch the body's "private spatial **path**" → rewrite in the body's plain vocabulary |
| A-M5 | low | G2 | Fig.2 caption `main.tex:104` | drop "raw" in "raw region window" (or say "the sequence of region identifiers") |
| A-M6 | low | G7 | `04:33-39` | combination-novelty stated in §1/§2/§4 → keep the full statement in one home, point from the others (coordinate with #46/#61) |
| A-M7 | low | G5 | `04:36-37` | "earlier contextual embeddings ... do not provide" → cite at the claim or point to §2.1 |
| A-M9 | low | G3 | `04:66-67` | "any gain" has a vague antecedent → "any improvement of the joint model over the dedicated single-task models" |
| A-M10 | low | G11 | `04:26-28` | "tell a true local structure from a random one" is imprecise/redundant → tighten or delete |
| A-M-miss | — | G11 | Fig.2 caption `main.tex:101` | caption calls the cross-attention stack "the only shared parameters", but the body also has private encoders with params → check the precision |

**§5 Setup** (zero prior comments — richest)
| ID | sev | pitfall | loc | fix |
|---|---|---|---|---|
| A-S3 | med | G8 | `05:72-75` | ~45-word run-on (deployment-margin) → split; drop the extra "coarse" |
| A-S4 | med | G1 | `05:94-95` | "we note in passing ... future headroom" (the STAN aside) → neutralize/defer. **This is the author's "other point" sentence** ("We never feed STAN our representation") |
| A-S5 | med | G5 | `05:54-55` | "inflated region accuracy by 13 to 27 points" → add "across states" + a pointer |
| A-S6 | med | G8/G3 | `05:56-58` | dense closing sentence ("the one residual ... finds none") → unpack, tie to the cold-place limitation |
| A-S7 | med | G10 | `05:47-48` | category leak "the large majority" is generous for AL (66.8%) → surface the coverage fraction (≈67% AL to 87% FL) + label it a POI-proxy |
| A-S9 | med | G7 | `05:16,22-23` + tbl1 caption | scale-spread rationale stated 3× → say once |
| A-S10 | med | G8 | `05:29-31` | the "gate" mechanism is hard to follow → state the rule plainly once |
| A-S12 | low | G11 | `05:27-28` | min-sequence/stride unstated → add "one window starting at each visit; at least ten visits per user" (makes the Tbl 1 Windows column reproducible) |
| A-S-miss | low | G7/data | tbl1 | "integrity of the representation" repeated (title+topic sentence); §5.4 Baselines is a dense block of long sentences; **TX Avg-len 105.8 vs CA 14.9 looks anomalous — worth a data sanity check** |

**§6 Results**
| ID | sev | pitfall | loc | fix |
|---|---|---|---|---|
| A-R2 | med | G1 | `06:35` | "the representation sharpens the semantic question and stays neutral" → state mechanically (improves category separability, carries no spatial structure) |
| A-R3 | med | G1 | `06:77` | "We state the support honestly." → delete, lead with the evidence |
| A-R5 | low | G2 | `06:101` | "stronger shared trunk" reused as the headline explanation → make the object concrete ("the shared layers trained better") |
| A-R9 | low | G7 | `06:49,102` | single-model property stated twice in §6.2 → drop the "single forward pass" at the line-49 lead, keep it at 102 |
| A-R10 | med | G11/CC1 | `06:57-58` | "ceilings---so called..." → **live em-dash (CC1)** + rephrase the appositive with a cite |
| A-R11 | med | G8 | `06:63` | ~70-word run-on (six numbers + TOST) → split at the TOST clause |
| A-R-miss | low | G7 | tbl2 footnote vs `06:26` | "64 to 90 percent" appears in both prose and the Tbl 2 footnote (body/caption repeat) |

**§7 Discussion**
| ID | sev | pitfall | loc | fix |
|---|---|---|---|---|
| A-D2 | low | G2 | `07:11` | "protects the region signal" → "keeps next-region accuracy from degrading" |
| A-D3 | med | G1 | `07:17` | "the spatial benefit of sharing" → "the gain on next-region from sharing" (CC6) |
| A-D5 | low | G8 | `07:16-18` | run-on chaining three claims → split; fix "meets or outperforms" → "matches or outperforms" (CC3 residue) |
| A-D7 | low | G7 | `07:34-35` | small-state match stated twice in §7 (also line 14) → drop the repeat |
| A-D-miss | low | G1 | `07:24` | "it is only motivation, since we measure no service" — second over-apologetic instance |

**§8 Conclusion**
| ID | sev | pitfall | loc | fix |
|---|---|---|---|---|
| A-C2 | low | G1 | `08:17` | "What we take away ... a clear reading" → "In summary, ..." |
| A-C5 | med | G8 | `08:17-20` | five-clause run-on → split into two sentences |
| A-C7 | med | G7 | `08:20` | scaling claim stated 3× (§1/§7/§8) → compress §8 |
| A-C8 | low | sync | `08:2-5` | header **comment** still says "beats"/"spatial gain"/"region space" (non-rendered) → sync to CC3/CC6/CC7 |
| A-C-miss | low | G12 | `08:19` | "anticipate both what a user will do and where" drops "next" (others say "what a user will do **next**") |

### False positives the advisor correctly dropped (for trust)

"infomax" is glossed in §2.1 (not undefined in §4); **"Read this as:"** is GLOSSARY-mandated, not a coinage;
**"mahalle"** is already glossed in §1/§3/§5; **"spatial task"** and **"integrity of the representation"** are
settled/author-chosen vocabulary; the §4 "private per-task encoder" wording is the accepted #63 decision. These were
proposed and rejected on verification.

> **Routing:** CC6/CC7 + the §4–§8 G7 redundancies feed **OP4** (cohesion); the run-ons + connectives feed **OP5**
> (English); the gloss items feed **OP3**; the §4 loss item is a correctness fix for the prose pass. None changes a
> verdict in the results.

### Final advisor pass (2 advisors over the whole review, 2026-06-29)

> Two independent advisors (completeness + correctness) audited everything above against the paper, GLOSSARY, and
> claim discipline. They **corrected one of my own errors**, found **two new HIGH items**, and surfaced **two
> whole-paper reviewer risks** nobody was tracking. Net: the review is safe to start applying once these are folded
> in (and the 7%-floor "fix" dropped — done above).

**New HIGH items (were untracked):**
- **§6 `06:113-115` cascade self-contradiction (G11 + claim discipline).** The paragraph says our model "reaches the
  same combined accuracy ... (the difference on the joint score is about zero)" (a **tie**) and then "the chained
  alternative **does not match our gains**" — which contradicts the tie and reads as a win-claim, breaking
  `PAPER_PLAN §3` ("never claim a win over the cascade"; this is the prior submission's exact failure mode). **Fix:**
  "... not a claim that we outperform the cascade: the chained alternative reaches the same combined accuracy at the
  same cost, so the simpler parallel structure loses nothing." (drop "does not match our gains").
- **`tbl1_datasets.tex:21` "next-POI" in a rendered caption (G2/glossary).** GLOSSARY bans "next-POI". The S-8 fix
  already touches this caption; also replace "the standard next-POI sparsity measure" → "the standard sparsity
  measure for next-place recommendation" (and fix the density-vs-sparsity wording per S-8).

**Whole-paper claim-discipline risks (highest-read spots; track these):**
- **Istanbul substrate caveat is dropped where it matters most.** §6.2/§6.3 honestly say Istanbul is measured on a
  *different* representation/windowing ("cross-setting check, not like-for-like"), but the **abstract (`main.tex:71`)
  and conclusion say "the result holds" / "the picture repeats" with no caveat.** After venue fit, this is the most
  likely reviewer pushback. Add a half-clause caveat in the abstract/conclusion or soften "holds" → "is consistent".
- **The "gain grows with region count" scaling claim drops its confound hedge everywhere except §6.2.** §6.2 hedges
  it ("we read the trend across six points rather than a precise law"; region-count and corpus-size co-vary, Istanbul
  on a different substrate), but the abstract, §1 contribution, §7, and §8 state it cleanly. Carry a one-clause hedge
  into at least one of those, or a reviewer calls it a confounded n=6 claim.
- **Venue-fit register (cumulative).** The service disclaimer recurs (§3 ×2, §5.3, §7 ×2, the radio-cell scoping);
  the more it apologizes, the more it invites "then why MobiWac?". Consolidate to one confident scoping in §3 (folds
  into OP4); this is the D-1/D-miss over-apologetic thread seen whole.

**Corrections to my own catalog (the advisors caught these):**
- **§6 7%-floor is NOT an error** — corrected above; drop the proposed fix.
- **CC3 was incomplete:** `figs/` was never swept. `fig4_deltas.tex:10` "the largest **win**" → "gain" is now
  **fixed** (CC3 truly complete: 0 verdict beats/wins in rendered text incl. figs; remaining "win" hits are TikZ
  node names for the *window* node in fig1, not prose).
- **CC7 list extended:** add the two highest-read misses — **abstract `main.tex:69`** and **Table 3 caption
  `tbl3_results.tex:21`** ("region space is large"). These should be done in the SAME pass as the body, never half.

**Apply-safety (for whoever runs the prose pass):**
- **Treat CC3 as already applied.** Ignore every v1 "keep/normalize/restore 'beats'" instruction (#8/#10/#14/#35) —
  applying them verbatim would *revert* CC3. Take only the non-verb parts (e.g. #9 "dedicated single-task model", #11
  "many regions"). The §7:18 "meets" → "matches" residue still needs doing.
- **Never paste the v1 edit text for #61/#64/#66 verbatim** — they contain `--`; rewrite with commas/parens (CC1).
- **#50:** contrast-only wording + "to our knowledge" hedge, **no TrajLearn bib entry** (locked at 26); update
  `PAPER_PLAN §3` if "underexplored" ships.

---

## Abstract (p1)

**1.**
> **Highlight:** "coarse"
> **Germano:** retirar
> **Action:** Remove the word "coarse".

> **Response — ❌ REJECT:** Keeping 'coarse' — it's the load-bearing scoping term, not filler, and it pre-empts the 'you
> just simplified the problem' attack (your own #48). The real risk is a connotation misread (coarse =
> crude/low-quality)
> for a non-ML reader, which the intro's '(less detailed)' gloss (#18) guards against; the abstract pairs it with the
> two
> named questions so it can't read as 'low-quality predictions'. No change to the word.
>
> **Edit:** none (rejected)

**2.**
> **Highlight:** "city)." (…the next region (which part of the **city**))
> **Germano:** two questions of what? / Frase sem sentido
> **Action:** The sentence is unclear — next category/region *of what?* (of the next visit).

> **Response — ✅ ACCEPT:** Fair — the two questions aren't anchored to anything. Binding them to 'the next visit' fixes
> the 'of what?' ambiguity and also resolves #3 (suffice) and #4 (the same of-what concern on the category clause) in
> one
> rewrite.
>
> **Edit:** Replace 'Two coarse questions usually suffice: the next category (the kind of place) and the next region (
> which part of the city).' with 'Two coarse questions about the next visit are usually enough: its category (the kind
> of
> place) and its region (which part of the city).'

**3.**
> **Highlight:** "suffice:"
> **Germano:** suffice????
> **Action:** Avoid "suffice" — reword.

> **Response — ✅ ACCEPT:** Agreed, 'suffice' reads oddly; 'are usually enough' is plainer for the networking audience.
> Folded into the #2 rewrite. Your deeper worry (is two questions really enough?) is the paper's thesis, argued in §3
> and
> §5.3, not the abstract.
>
> **Edit:** Within the #2 rewrite, 'usually suffice:' becomes 'are usually enough:'.

**4.**
> **Highlight:** "the next category (the kind of place) and the next region"
> **Germano:** next category and the next region of what?
> **Action:** Specify "of the next visit".

> **Response — ✅ ACCEPT:** Same valid point as #2, on the category clause. The #2 rewrite makes both targets explicit
> properties of the next visit ('its category ... and its region'), answering 'of what?' once for both — no separate
> edit.
>
> **Edit:** Covered by the #2 rewrite.

**5.**
> **Highlight:** "what sharing a single representation costs."
> **Germano:** estranho
> **Action:** Awkward phrasing — rephrase.

> **Response — 🟡 PARTIAL:** The 'cost of sharing' is the paper's central research question, so I won't cut the idea, but
> the gerund subject reads heavy. Smoothing keeps the meaning. Note: #27 must stay consistent with this — it will NOT
> say 'without paying that cost' (that would contradict this line and overclaim).
>
> **Edit:** Replace '... so we ask whether one model can learn both, and what sharing a single representation costs.'
> with '... so we ask whether one model can learn both, and what it costs to share one representation.'

**6.**
> **Highlight:** "instead of giving every place one fixed vector."
> **Germano:** retirar. O leitor pode ficar perdido aqui
> **Action:** Remove — too much detail for the abstract; reader can get lost.

> **Response — ❌ REJECT:** Keeping it — this is the one plain-language line that tells a non-ML reader what '
> check-in-level representation' means: per-visit context versus one fixed vector per place. That contrast is the core
> idea and the paper's stated novelty, so for the MobiWac audience it clarifies rather than confuses. Cutting it leaves
> the term undefined exactly where it first appears.
>
> **Edit:** none (rejected)

**7.**
> **Highlight:** "five U.S. states,"
> **Germano:** which ones?
> **Action:** Name the states (or defer to Setup).

> **Response — ❌ REJECT:** Enumerating the five states in the abstract is clutter abstracts conventionally avoid;
> they're named and tabulated in §5.1 / Table I. Your 'which ones?' is answered there, where it belongs.
>
> **Edit:** none (rejected)

**8.**
> **Highlight:** "beats" (…this lifts… / it **beats** a dedicated category model)
> **Germano:** outperforms
> **Action:** Use "outperforms" instead of "beats".

> **Response — 🟡 PARTIAL:** I'm rejecting the swap to 'outperforms', Germano. 'Beats/matches' is the paper's DEFINED
> verdict vocabulary — §5.3 sets up 'which a beat and which a match is a result' (paired Wilcoxon for beats, TOST for
> matches) and all eight sections plus the Table III caption use it. Swapping only the abstract introduces a third
> superiority verb (the body also has 'wins') decoupled from the test that defines it. Your real point — three
> verbatim '
> beats' in the abstract (#25) — is valid, so I'll vary sentence structure to break the triple while keeping the defined
> verb.
>
> **Edit:** Keep 'beats' as the defined verdict verb. Restructure the abstract's region/Istanbul sentences to avoid
> three verbatim 'beats' (e.g., '...it beats a dedicated single-task category model (+4.7 to +7.7 macro-F1); on region
> it
> again beats the single-task model where regions are many, and matches it — statistically, within two points — where
> they
> are few.'). Do NOT introduce 'outperforms'.
>
> AUTHOUR: Replace to outpeforms


**9.**
> **Highlight:** "dedicated category model"
> **Germano:** what is a dedicated model? Try to use more classical jargons
> **Action:** Define / replace "dedicated model" with standard terminology (single-task baseline).

> **Response — ✅ ACCEPT:** Good catch — 'dedicated model' is vague (dedicated to what?). But the paper's anchor name
> is 'dedicated single-task model' (§5.3) and Table III's column header is literally 'Dedicated', so the fix is to
> EXPAND
> on first use, not strip 'dedicated' (which would orphan the table). I'll write 'dedicated single-task category model'
> at
> first mention and keep 'dedicated model' as the established shorthand thereafter.
>
> **Edit:** First use (abstract): 'dedicated category model' -> 'dedicated single-task category model'; thereafter '
> dedicated model' as shorthand. Do NOT rename to bare 'single-task model'. Apply the same first-use expansion in the
> intro contribution bullets; leave Table III's 'Dedicated' column and the body shorthand intact.

**10.**
> **Highlight:** "beats" (…on region it **beats** the dedicated model)
> **Germano:** exceeds in performance
> **Action:** Use "exceeds in performance" instead of "beats".

> **Response — 🟡 PARTIAL:** Accept the spirit but keep 'beats', not 'outperforms' (it's the §5.3-defined verb; see
> #8). 'Beats' is honest here because region genuinely BEATS at the large counts (paired Wilcoxon superiority at
> FL/TX/CA), distinct from the small-count MATCH. I'll only fix the imprecise noun phrase 'region space is large' (see
> #11).
>
> **Edit:** Replace 'on region it beats the dedicated model where the region space is large' with 'on region it beats
> the single-task model where there are many regions' (the 'many regions' part resolves #11).
>
> AUTHOUR: Exceeds in performance

**11.**
> **Highlight:** "region space"
> **Germano:** region area
> **Action:** Prefer "region area".

> **Response — 🟡 PARTIAL:** Pushing back on 'region area' — it's technically wrong here: it reads as geographic area (
> km^2), but the paper means the number of region classes (cardinality, ~500 in Istanbul up to ~8,500 in California),
> the
> opposite kind of 'size'. The clean fix is 'many regions', which also matches the next sentence ('grows with the number
> of regions').
>
> **Edit:** Replace 'where the region space is large' with 'where there are many regions' (folded into the #10 edit).

**12.**
> **Highlight:** "matches it (statistically, within two points) where it is small."
> **Germano:** não entendi. O que seria *statistically, within two points*? Fez teste de hipótese?
> **Action:** Clarify "statistically, within two points" — state explicitly that a TOST hypothesis test was run.

> **Response — ❌ REJECT:** This conflicts with a prior decision: the abstract was deliberately softened ('matches it,
> statistically, within two points') with the TOST machinery deferred. Your underlying question — did we run a
> hypothesis
> test? — is fully answered: §5.3 names a 'two one-sided tests (TOST)' non-inferiority procedure (confirmed in
> 05_setup.tex L72) at a ±2 pp margin, applied per-cell in §6.2 with 90% CIs. Keeping the acronym out of the abstract;
> pointing you to §5.3/§6.2.
>
> **Edit:** none (rejected)

**13.**
> **Highlight:** "spatial gain"
> **Germano:** spatial gain?
> **Action:** Define / rephrase "spatial gain".

> **Response — ✅ ACCEPT:** Agreed, 'spatial gain' is a coined phrase that's easy to misread. Plain 'the gain on region'
> is clearer for both audiences and ties directly to the region task. No information lost.
>
> **Edit:** Replace 'The spatial gain grows with the number of regions.' with 'The gain on region grows with the number
> of regions.'

**14.**
> **Highlight:** "result holds:" (On a non-U.S. city (Istanbul) the **result holds**…)
> **Germano:** Our model also demonstrate superior performance on a non-U.S city (Istanbul) when comparing the
> next-category and next-region.
> **Action:** Suggested rewrite of the Istanbul sentence (use Germano's phrasing as a base).

> **Response — 🟡 PARTIAL:** I must reject your suggested wording — it claims 'superior performance' on BOTH tasks, which
> is factually wrong. Istanbul has ~500 mahalle (small count), so region is a statistical MATCH (TOST, within two
> points),
> NOT a beat; claiming superiority re-triggers exactly the overclaim a reviewer would flag. I accept that 'the result
> holds' is vague and make it explicit while keeping the beats-on-category / matches-on-region split, and keep the
> defined
> verb 'beats' (not 'outperforms', per #8).
>
> **Edit:** Replace 'On a non-U.S. city (Istanbul) the result holds: it beats on category and stays within two points on
> region.' with 'On a non-U.S. city (Istanbul, where regions are the mahalle) the same pattern holds: it beats on
> category
> and statistically matches the single-task model on region (within two points).'
>
> AUTHOUR: keep the current
---

## Introduction (p1)

**15.**
> **Highlight:** "because a service that can anticipate the next move can prepare ahead of time instead of reacting
> after the fact."
> **Germano:** confuso. o que quis dizer aqui?
> **Action:** Confusing — clarify the intended meaning.

> **Response — 🟡 PARTIAL:** The proactive-vs-reactive framing is exactly what a wireless reviewer cares about, so the
> intent is right, but you're correct that 'prepare ahead of time instead of reacting after the fact' is abstract (
> prepare
> what? react to what?). The next sentence already grounds it, so I'll tighten the hook by naming what gets prepared
> rather than cut the idea.
>
> **Edit:** Reword to: '...because anticipating the next move lets a mobility-aware service provision for demand in
> advance rather than react once it has already arrived.'
>
> **[REJECT BY AUTHOUR] keep as it is**

**16.**
> **Highlight:** "Predicting the exact next place, a single point of interest (POI), is hard,"
> **Germano:** predicting the next place exactly is hard.
> **Action:** Reword to "predicting the next place exactly is hard."

> **Response — 🟡 PARTIAL:** Happy to adopt your adverb order, but the load-bearing element is the inline POI definition
> on first use, which the non-ML audience needs and your literal rewrite drops. So I take the reorder and keep the
> appositive that defines POI.
>
> **Edit:** Reword to: 'Predicting the next place exactly, a single point of interest (POI), is hard,' (keep the
> appositive).

**17.**
> **Highlight:** "and it is often more than a service needs."
> **Germano:** retirar. Senão colocar uma citação
> **Action:** Remove, or add a citation.

> **Response — ❌ REJECT:** This clause is load-bearing motivation — the reason we predict coarser targets, which you
> yourself probe in #48. It's a design rationale grounded by the paper's own use cases (content staging and capacity
> planning need the region, not the exact POI), so an external citation wouldn't fit and removal would weaken the
> central
> why-coarser argument. (Consistent with #57: cite concrete application examples, never the framing sentence.)
>
> **Edit:** none (rejected)

**18.**
> **Highlight:** "coarser"
> **Germano:** o que é coarser questions?
> **Action:** Define "coarser questions".

> **Response — 🟡 PARTIAL:** The contrast with 'the exact next place' already implies coarser = lower-resolution, so it
> isn't broken — but a one-word plain gloss on first use is cheap, matches how the rest of the paper glosses terms for
> the
> non-ML reader, and doubles as the connotation guard for #1 (coarse != low-quality).
>
> **Edit:** Reword to: 'Two coarser (less detailed) questions are easier to answer and usually enough:'.
>
> AUTHOUR: Maybe simpler insteat fo coarser ?

**19.**
> **Highlight:** "part of the city"
> **Germano:** region
> **Action:** Use "region".

> **Response — ❌ REJECT:** 'part of the city' IS the plain-language gloss of 'region' here ('the next region, the part
> of the city a user is heading to'). Replacing the gloss with 'region' makes it circular ('the next region, the
> region')
> and deletes exactly the plain-words definition the clarity panel asked for. The jargon swap hurts the actual MobiWac
> reader.
>
> **Edit:** none (rejected)

**20.**
> **Highlight:** "(we use the census tract, or the mahalle in Istanbul)."
> **Germano:** sem necessidade de colocar aqui essa parte. Você está definindo o problema somente
> **Action:** Unnecessary here — this is just problem definition; cut from the intro.

> **Response — 🟡 PARTIAL:** I won't fully cut this — it grounds 'region' for the networking reader and, via the mahalle,
> previews the Istanbul external-validity result. But you're right the full operational definition belongs in Problem
> and
> Tasks (confirmed present in §3), so I trim to keep just the grounding and the international preview.
>
> **Edit:** Reword to: '(a neighborhood-scale unit, the census tract in the U.S. and the \emph{mahalle} in Istanbul)'
> and let the detailed operational definition live in Section~\ref{sec:problem}.

**21.**
> **Highlight:** "We do not predict the exact next place."
> **Germano:** retirar
> **Action:** Remove.

> **Response — 🟡 PARTIAL:** Don't drop it entirely — an explicit 'we do not predict the exact next place' is a
> deliberate scope boundary that heads off the 'why not predict the exact POI?' reaction (a BRACIS sore point), and it's
> also stated in §3. But as a bare standalone sentence it reads choppy, so I'll fold it into the flow.
>
> **Edit:** Merge into the preceding clause: '...the \emph{mahalle} in Istanbul); we do not target the exact POI.'
>
> AUTHOUR: We can remove it.

**22.**
> **Highlight:** "The" (**The** two questions pair naturally…)
> **Germano:** These
> **Action:** Replace "The" with "These".

> **Response — ✅ ACCEPT:** Agreed, 'These' is more precise — it points back to the two questions just named. Harmless,
> correct micro-fix.
>
> **Edit:** Reword to: 'These two questions pair naturally,'.

**23.**
> **Highlight:** "and they are more learnable than"
> **Germano:** não entendi
> **Action:** Unclear — rephrase.

> **Response — ✅ ACCEPT:** Fair, 'more learnable' is borderline jargon. The point is that one-of-seven categories or
> one-of-N regions is easier to predict accurately than one-in-thousands POIs, so I'll say that plainly, which also
> helps
> the networking reader.
>
> **Edit:** Reword to: 'and they are easier to predict accurately than a single POI while still being useful.'

**24.**
> **Highlight:** "Our earlier work [3] found that joint training can help one task while quietly hurting the other,
> because the shared part is pulled toward a compromise that suits neither perfectly [4]."
> **Germano:** Aqui você afirma que nosso trabalho identificou isso, mas não foi. A gente só chegou nessa conclusão para
> essas duas tarefas, mas não foi a gente quem identificou isso. … pode parafrasear o pareto. Otimalidade de pareto que
> discute esse ponto.
> **Action:** Don't credit *us* with discovering the trade-off — we only observed it for these two tasks. Reframe via *
*Pareto optimality** (paraphrase that concept).

> **Response — ✅ ACCEPT:** Right to attribute the phenomenon — negative transfer is a known MTL property, not our
> discovery. But I'm dropping the word 'Pareto': Caruana 1997 is the canonical MTL-helps / hard-parameter-sharing
> reference, not a Pareto / multi-objective source, and there is no multi-objective citation in references.bib to carry
> it (an ML reviewer would catch 'Pareto [Caruana 1997]'). I'll credit the general trade-off to the MTL literature and
> reserve our earlier work for the specific observation.
>
> **Edit:** Reword to: 'joint training can help one task while hurting the other, a well-known trade-off in multi-task
> learning where the shared parameters settle on a compromise that suits neither task perfectly~
> \cite{caruana1997multitask}; our earlier work~\cite{silva2025mtlnet} observed exactly this for next-category and
> next-region.' (drop 'Pareto').

**25.**
> **Highlight:** "useful question"
> **Germano:** repetindo muito *useful question*.
> **Action:** "useful question" repeated too often — vary the wording.

> **Response — ✅ ACCEPT:** Agreed, 'useful question' appears in para 1 and again in para 3. Vary the second one. (This
> and #8's restructuring together kill the abstract's repetition complaint.)
>
> **Edit:** In para 3 reword 'So the useful question is not whether...' to 'So the question that matters is not
> whether...'.

**26.**
> **Highlight:** "multi-task learning (MTL)"
> **Germano:** apresenta o que é multi-task learning primeiro.
> **Action:** Introduce/define MTL **before** using it.

> **Response — ✅ ACCEPT:** Agreed, and defining MTL in plain words on first use serves the networking audience
> directly (it also feeds #64, where 'shared trunk' is used before MTL is explained). Introduce and gloss it at the top
> of
> the paragraph, then use the bare acronym afterward.
>
> **Edit:** Change the paragraph opener to: 'Sharing one representation across the two tasks, an approach known as
> multi-task learning (MTL), is not free.' and drop the parenthetical from the later sentence so it reads '...is not
> whether MTL is good in general...'.

**27.**
> **Highlight:** "what to do about it."
> **Germano:** não entendi direito. Fazer o que aqui?
> **Action:** Unclear — clarify what "what to do about it" refers to.

> **Response — ✅ ACCEPT:** Agreed the phrase is vague, but I will NOT write 'without paying that cost' — that
> overclaims (our region result is non-inferiority, a bounded cost <=2 pp, not zero) and contradicts the line two
> sentences up (#5: 'what it costs to share one representation'). The honest concrete version names the goal: keep the
> gains while keeping the cost small.
>
> **Edit:** Edit line 36: '...where sharing helps, where it costs, and what to do about it.' -> '...where sharing helps,
> where it costs, and how to share so the gains hold and the cost stays small.'

**28.**
> **Highlight:** "We make two changes"
> **Germano:** we make? *We propose two changes* é melhor.
> **Action:** Prefer "We propose two changes".

> **Response — ✅ ACCEPT:** Agreed, minor register fix. 'We introduce two changes' reads better than 'We make'
> alongside '...and measure each one carefully'. Adopting it.
>
> **Edit:** Edit line 38: 'We make two changes and measure each one carefully.' -> 'We introduce two changes and measure
> each one carefully.'

**29.**
> **Highlight:** "instead of one fixed vector per place, where two visits to the same coffee shop look identical, each
> check-in gets its own vector that carries its context (the time, the nearby places, and the recent trail)."
> **Germano:** isso aqui me parece muito natural. Qual artigo que faz diferente disso? Não me parece uma contribuição
> nossa.
> **Action:** Feels too natural to claim as a contribution — which prior work does it differently? Justify or soften.

> **Response — 🟡 PARTIAL:** Fair challenge, and it matches our novelty discipline: per-visit context is not ours (CTLE
> already provides it). I'll add ONE light clause in the intro and keep the full scoping in Related Work (#46) and
> Method (#61/#62) so the same sentence doesn't appear three times. Audience caveat: CTLE is never expanded in the
> paper,
> so in the intro I'll present it as 'sequence-model embeddings such as CTLE [cite]' rather than a bare acronym.
>
> **Edit:** After '...from the place down to the individual visit (Fig.~\ref{fig:dataflow}).' add a brief,
> differentiated clause: 'Per-visit context is not new on its own — sequence-model embeddings such as CTLE~
> \cite{lin2021ctle} provide it — but we obtain it from inside the hierarchical-graph-infomax representation and test
> that
> choice against CTLE directly.' Keep the fuller treatment in Related Work/Method; do not duplicate verbatim.
>
> AUTHOUR: The rephrase is good, but is not following our glossary.

**30.**
> **Highlight:** "This extends hierarchical graph representations of places [1],"
> **Germano:** representar assim estende o trabalho de hierarchical graph representation?
> **Action:** Verify/justify the claim that this *extends* the hierarchical graph representation work.

> **Response — 🟡 PARTIAL:** The 'extends' claim is accurate and verified — Method states a four-level graph (
> check-in/place/region/city) and Fig 1 draws it, so 'a fourth, check-in level' is correct, not off-by-one. I'll make
> the
> hierarchy-extension explicit. But I'll keep the intro ACRONYM-FREE: 'HGI' isn't expanded until Related Work, so
> injecting bare 'HGI' into the intro would hit the networking reader with undefined jargon.
>
> **Edit:** Edit line 42: '...from the place down to the individual visit' -> '...by adding a fourth, check-in level
> beneath the hierarchy of nested place graphs, from the place down to the individual visit'. Do NOT use the bare 'HGI'
> acronym in the intro (first expanded in Related Work).

**31.**
> **Highlight:** "semantic trunk"
> **Germano:** semantic trunk? use classical jargon
> **Action:** Replace "semantic trunk" with standard terminology (shared layers / backbone).

> **Response — 🟡 PARTIAL:** Half-agree: I'll drop the 'semantic trunk' compound that reads poetic, but I'm keeping the
> word 'trunk' — Method, results, and discussion all say 'shared trunk', so switching the intro to 'shared layers' would
> create an intro/body mismatch. I keep the semantic-vs-spatial contrast (it's load-bearing: what is shared vs. what
> stays
> private). Note 'backbone' would be worse for our audience — it collides with 'network backbone'.
>
> **Edit:** Edit line 46: 'with a shared semantic trunk and a private spatial path for the region task' -> 'with a
> shared trunk that carries the semantic context and a private spatial path for the region task'. Keep 'trunk'; the
> hard-parameter-sharing gloss lands once at first use (#64).
>
> AUTHOUR: E need to think more about this one is still weird

**32.**
> **Highlight:** "international city"
> **Germano:** African city / western cities versus eastern cities
> **Action:** Consider framing external validity as Western vs. Eastern cities.

> **Response — ❌ REJECT:** Istanbul is one city, so it can only serve as an external-validity check (do findings survive
> outside the U.S.?), not a 'Western vs Eastern' cultural contrast. Claiming a cultural axis from n=1 overclaims and
> would
> invite a reviewer to demand a balanced multi-city cultural study we don't have ('African city' also doesn't apply —
> Istanbul isn't African). Keeping the honest 'international / non-U.S. city' framing.
>
> **Edit:** none (rejected)

**33.**
> **Highlight:** "(a balanced score that counts every category equally)"
> **Germano:** retirar
> **Action:** Remove the parenthetical.

> **Response — ❌ REJECT:** Keeping the gloss — this is a deliberate clarity decision for our actual audience. MobiWac
> reviewers are wireless-networking people, and the clarity panel specifically asked that ML metrics be glossed in plain
> words. Removing '(a balanced score that counts every category equally)' serves an ML reader at the expense of the
> reader
> we're writing for.
>
> **Edit:** none (rejected)

**34.**
> **Highlight:** "+28 to +40"
> **Germano:** huge. É esse número mesmo?
> **Action:** Double-check the +28 to +40 figure — it looks huge.

> **Response — ✅ ACCEPT:** Checked and the number holds: AL +29.3, AZ +27.6, FL +39.6, CA +38.0, TX +37.5 (Table II),
> and the abstract already hedges with 'about +28 to +40', so AZ's +27.6 rounds in honestly. It's large because a single
> fixed vector per place floors category macro-F1 in the high-20s/30s — co-located, multi-activity places collapse into
> one representation — and per-visit context restores the separation. I'll add one clause saying why and point it at the
> Fig 3 evidence so the claim is visibly backed.
>
> **Edit:** Keep 'about $+28$ to $+40$'. Append to the first contribution bullet: '...this margin is large because a
> single fixed vector per place cannot separate the categories of places that host several kinds of activity, which
> per-visit context recovers (Section~\ref{sec:results}, Fig.~\ref{fig:embquality}).'

**35.**
> **Highlight:** "beats" (…it **beats** a dedicated category model at every state)
> **Germano:** outperforms
> **Action:** Use "outperforms".

> **Response — 🟡 PARTIAL:** Same call as #8: keeping 'beats' (the §5.3-defined verdict verb used across all sections),
> NOT swapping to 'outperforms' — that would fragment the vocabulary. Category genuinely beats at every state, so the
> verb
> is honest. The noun I'll normalize via the naming sweep: first use becomes 'dedicated single-task category model',
> then 'dedicated model' as shorthand (#9).
>
> **Edit:** Line 63: keep 'beats'; normalize the noun on first use to 'dedicated single-task category model' (then '
> dedicated model'). Do NOT change 'beats' to 'outperforms'.
---

## Contributions (p2)

**36.**
> **Highlight:** "An empirical account,"
> **Germano:** empirical account?
> **Action:** Rephrase "empirical account".

> **Response — ✅ ACCEPT:** Agree it reads a bit literary. 'An empirical analysis' is clearer and standard, and loses
> nothing. Swapping it.
>
> **Edit:** Edit line 68: 'An empirical account, across five states...' -> 'An empirical analysis, across five
> states...'.

**37.**
> **Highlight:** "±2 pp)"
> **Germano:** pp?
> **Action:** Spell out "pp" (percentage points) on first use.

> **Response — ✅ ACCEPT:** Right, 'pp' is unglossed on first use. I'll expand it to 'percentage points (pp)' at intro
> line 72 and use 'pp' thereafter. Convention note: the softened abstract deliberately stays in plain 'points' (no '
> pp');
> the body uses 'percentage points (pp)' for the TOST margin and metric deltas — so I'm fixing the first-use expansion,
> not converting the abstract.
>
> **Edit:** Edit intro line 72: '(TOST, $\pm2$~pp)' -> '(TOST, $\pm2$ percentage points (pp))'; use 'pp' after. Leave
> the abstract's 'within two points' as the plain-language form.
>
> AUTHOUR: i belive the pp is a common abbreviation in the field, what do you think ?
---

## Related Work (p2)

**38.**
> **Highlight:** "by raw identifiers."
> **Germano:** what is raw identifiers?
> **Action:** Define "raw identifiers".

> **Response — ✅ ACCEPT:** Fair point — for the networking audience 'raw identifiers' is undefined jargon for the bare
> place ID, and a one-clause gloss helps both audiences without bloating the sentence.
>
> **Edit:** Edit to: '...so that a model can reason about places by their geometry rather than by raw identifiers, the
> arbitrary ID numbers that carry no meaning on their own.'
>
> AUTHOUR: Raw identifiers would be it's raw category, location and time, no ?

**39.**
> **Highlight:** "Deep Graph Infomax (DGI) [5] learns such vectors by training them to tell a real network of places,
> linked by similarity, time, and distance, from a shuffled copy of that network."
> **Germano:** DGI is not a work for places. It is a technique for self-supervised learning on graphs
> **Action:** Correct the description — DGI is a self-supervised graph learning technique, not a place-specific method.

> **Response — ✅ ACCEPT:** Correct and important. DGI is a task-agnostic self-supervised method for node representations
> on graphs; the current sentence reads as if DGI were place-specific, which a reviewer could rightly flag. Make
> explicit
> that DGI is the general technique and the place network is our application of it.
>
> **Edit:** Edit to: 'Deep Graph Infomax (DGI)~\cite{velickovic2019dgi}, a general self-supervised method for learning
> node representations on graphs, is applied here to a network of places: it learns such vectors by training them to
> tell
> the real place network, linked by similarity, time, and distance, from a shuffled copy.'

**40.**
> **Highlight:** "Both methods emit one vector per place,"
> **Germano:** what is to emit a vector? Do you mean to yield?
> **Action:** Replace "emit" a vector (→ produce/yield).

> **Response — ✅ ACCEPT:** Agreed, 'emit' is mildly odd; 'produce' is plainer and standard. The three occurrences are
> 02_related L18 and L28 and the Method novelty sentence (04_method L35, 'emitting'). I'll change all three — including
> the Method one, which #61's merge must also carry as 'producing', so we don't end up with 'produced' and 'emitting'
> for
> the same action.
>
> **Edit:** Replace 'emit'/'emits'/'emitting' with 'produce'/'produces'/'producing' at 02_related L18, 02_related L28,
> and 04_method L35. Coordinate with #61 so the merged novelty sentence reads 'producing a vector per check-in'.

**41.**
> **Highlight:** "but it gets there a different way."
> **Germano:** but in a different way
> **Action:** Reword to "but in a different way".

> **Response — 🟡 PARTIAL:** Minor register tweak, not a real defect. 'Gets there a different way' is clear plain English
> the MobiWac reader parses fine, but I'll tighten it slightly since your version reads marginally cleaner. I would not
> call the original wrong.
>
> **Edit:** Optional tightening: '...also works at the check-in level, but it gets there differently.'

**42.**
> **Highlight:** "now extended down to a fourth, check-in level (Fig. 1),"
> **Germano:** não entendi. Não sei se é a melhor posição aqui para explicar isso
> **Action:** Unclear and possibly the wrong place to explain this — consider moving/rewriting.

> **Response — 🟡 PARTIAL:** The clarity complaint is fair, the relocation suggestion is not. The phrase is dense because
> it assumes the reader has counted place/region/city as three levels — and the four-level structure is verified (Method
> L20, Fig 1) — so I'll spell it out, but this embeddings subsection is the right place to explain the HGI lineage (HGI
> is
> already expanded here), so I keep it here.
>
> **Edit:** Reword in place to: '...trains it with the same infomax objective, now extended one level deeper, from
> individual places down to individual check-ins, a fourth level beneath place, region, and city (Fig.~\ref{fig:
> dataflow}), rather than switching to a sequence model.'

**43.**
> **Highlight:** "rather than switching to a sequence model."
> **Germano:** como assim sequence model?
> **Action:** Clarify what "sequence model" means here.

> **Response — 🟡 PARTIAL:** The term itself is standard, but you're right the contrast floats: the 'sequence model' we
> contrast against is CTLE, named two sentences earlier. Pinning it to CTLE removes the ambiguity for both audiences
> without adding a definition.
>
> **Edit:** Edit to: '...rather than switching to a sequence model, as CTLE does.'

**44.**
> **Highlight:** "Put another way,"
> **Germano:** Put another way?????
> **Action:** Drop "Put another way".

> **Response — 🟡 PARTIAL:** Low stakes. 'Put another way' is a harmless reader aid signaling a plain-language
> restatement, which the non-expert audience benefits from, so I won't cut the restatement. But the connective can be
> swapped to something less conversational since it bothers you.
>
> **Edit:** Replace 'Put another way,' with 'In short,' (or drop the connective and keep the sentence).

**45.**
> **Highlight:** "earlier representations"
> **Germano:** which earlier representations?
> **Action:** Specify *which* earlier representations.

> **Response — ✅ ACCEPT:** Specifying the antecedent answers your question and fixes a subtle inaccuracy: DGI and HGI
> changed which features are encoded, whereas CTLE already changed how often a vector is produced (per visit), so
> lumping
> all three is imprecise. Naming them sharpens the contrast (and this occurrence is one of the emit->produce fixes,
> #40).
>
> **Edit:** Edit to: '...earlier place embeddings (DGI, HGI) changed \emph{which} features a place vector encodes; CTLE
> and we change \emph{how often} a vector is produced, once per place or once per visit.'

**46.**
> **Highlight:** "representations changed which features a place vector encodes; we change how often the representation
> emits a vector, once per place or once per visit. The novelty is this specific combination, per-visit context inside a
> hierarchical graph-infomax representation, and we compare against CTLE directly to show that the combination, not
> per-visit context on its own, is the source of the gain."
> **Germano:** redigir outro parágrafo
> **Action:** Rewrite this paragraph.

> **Response — 🟡 PARTIAL:** A wholesale rewrite isn't warranted — this paragraph does exactly what the BRACIS
> post-mortem requires: it scopes novelty to the specific combination and commits to the CTLE comparison, so weakening
> it
> re-triggers the novelty attack. I'll apply only light copyedits (emit->produce #40, antecedent fix #45) and keep the
> combination-novelty framing, differentiated from the intro pointer (#29) and the fuller Method statement (#61/#62) so
> it
> doesn't read verbatim three times.
>
> **Edit:** Keep the combination-novelty framing; apply only the copyedits from #40/#45. Do not rewrite the paragraph
> wholesale; ensure it is not a verbatim duplicate of the intro (#29) or Method (#61) versions.
>
> AUTHORS: The ideia here is to split and write a new paragraph with the other changes proposed, not rewrite, so we can
> be more clear, what do you think?

**47.**
> **Highlight:** "We deliberately predict two coarser properties of the next visit instead: its category, what kind of
> place it is, and its region, where it falls, at the granularity of a census tract."
> **Germano:** não tem necessidade de falar o que a gente faz aqui. é só falar o que está sendo feito por outros
> trabalhos e apresentar o gap
> **Action:** In Related Work describe only *others'* work and present the gap — don't describe our method here.

> **Response — 🟡 PARTIAL:** Half right. The method specifics (census-tract granularity) belong in Problem/Tasks (
> confirmed present in §3), so I'll defer them. But the contrast sentence itself is the gap statement — 'others predict
> the exact place; we deliberately do not' is exactly how Related Work presents a gap — so I won't strip it.
>
> **Edit:** Trim to a gap statement and defer the granularity: 'In contrast, we target two coarser properties of the
> next visit, its category and its region, rather than the exact place.' (Census-tract definition stays in §3.)

**48.**
> **Highlight:** "Coarser targets are easier to get right and still useful to act on, and they line up with two kinds of
> preparation a mobility-aware service can make."
> **Germano:** então a gente simplifica o problema? isso não é um drawback?
> **Action:** Address the implied criticism — does coarsening simplify the problem (a drawback)? Frame as a deliberate,
> justified choice.

> **Response — ✅ ACCEPT:** This is the strongest comment in your batch, and a reviewer will ask the same thing: did we
> just make the problem easier? We rebut head-on — coarser is not trivial here (region still spans hundreds to ~8,500
> classes), and the choice is operationally motivated (a service acts on category and region, not on an exact POI), not
> score inflation.
>
> **Edit:** Add justification: 'Coarser does not mean trivial: the region target alone spans hundreds to several
> thousand classes. We choose these targets because they are what a mobility-aware service can actually act on, not to
> make the problem easier, and they line up with two kinds of preparation such a service can make.'
>
> Authour: Take care with the over use of the word "coarser"

**49.**
> **Highlight:** "The category task appears in our own earlier line of"
> **Germano:** retirar / se aparece em outro trabalho por que não colocar aqui?
> **Action:** Remove — or if it appears in other work, cite it here.

> **Response — 🟡 PARTIAL:** Your either/or is already half-satisfied — the sentence carries the citation (
> \cite{silva2025mtlnet}, our earlier MTLnet work), so 'remove' would erase honest lineage and the 'cite it' branch is
> met. I'll keep the self-citation but tighten the phrasing so it doesn't read as a stranded fragment.
>
> **Edit:** Keep the citation; integrate: 'The category task itself follows our earlier line of work~
> \cite{silva2025mtlnet}.'

**50.**
> **Highlight:** "We also note that next region over census tracts has little precedent:"
> **Germano:** predict over census tracts is a novelty?
> **Action:** Is census-tract prediction really novel? Justify the claim or cite prior work.

> **Response — 🟡 PARTIAL:** Fair challenge, accepting the scoping per our novelty discipline. The honest claim is not
> that the census-tract unit is novel — it's that treating fine-grained region as a co-equal END target (rather than an
> auxiliary coarse cell, as HMT-GRN does) is underexplored. I'll reword so we don't overclaim the unit, but won't drop
> the
> scoping the comparison rests on.
>
> **Edit:** Reframe to: 'Fine-grained region as an end target, rather than an auxiliary coarse cell, has little
> precedent: HMT-GRN predicts a coarse geohash cell only as an aid, so we define our region unit carefully when we state
> the tasks.'
>
> Authour: Search on the web/literature to be more sure about his.

**51.**
> **Highlight:** "uses category in the same spirit,"
> **Germano:** in the same spirit?
> **Action:** Reword the vague "in the same spirit".

> **Response — ✅ ACCEPT:** Agreed, 'in the same spirit' is vague, and the precise meaning is already in the next
> clause ('to filter candidate places'), so name it: category is used instrumentally, as in the CSLSL cascade. Clean
> fix.
>
> **Edit:** Edit to: 'CatDM~\cite{yu2020catdm} likewise uses category instrumentally, to filter candidate places.'

**52.**
> **Highlight:** "balancer" (yet another gradient **balancer**…)
> **Germano:** regularizer
> **Action:** Terminology — consider "regularizer".

> **Response — ❌ REJECT:** Technically wrong, Germano. PCGrad, GradNorm, and Nash-MTL are gradient-balancing MTL
> optimizers — they manipulate per-task gradients/loss weights, not regularizers (which add a penalty term). Renaming
> them 'regularizers' is a substantive error a reviewer would catch. The first-use term 'gradient balancer' is already
> glossed at 02_related L70, so the clarity need is met; at most an optional tighten to 'gradient-balancing method',
> never 'regularizer'.
>
> **Edit:** None (optional: 'gradient balancer' -> 'gradient-balancing method' for precision; do NOT use 'regularizer').

**53.**
> **Highlight:** "balancers" (such **balancers** rarely beat…)
> **Germano:** regularizers
> **Action:** Terminology — "regularizers".

> **Response — ❌ REJECT:** Same issue as #52: 'such balancers rarely beat a well-tuned fixed weighting' is correct —
> these methods balance gradients, they don't regularize, and the line is anaphoric to 'gradient balancer' just above (
> L70). 'Regularizers' would mislabel the entire MTL-optimizer literature being cited (xin2022domtl, kurin2022defense).
> Keep 'balancers'.
>
> **Edit:** None (optional: 'such balancers' -> 'such gradient-balancing methods'; do NOT use 'regularizers').
---

## Problem & Tasks (p2–3)

**54.**
> **Highlight:** "We work with check-in sequences."
> **Germano:** deveria vir na introdução
> **Action:** This belongs in the Introduction.

> **Response — ❌ REJECT:** Keeping the line — a Problem and Tasks section conventionally re-anchors its data substrate,
> and 'We work with check-in sequences' sets up the very next sentence (ordering check-ins, forming nine-visit windows);
> cut it and the windowing starts abruptly. I take your redundancy point, though: if it reads as a restatement of the
> intro, the low-churn option is to fold it into the windowing sentence rather than stand alone. Leaving placement as-is
> but flagging that merge as available.
>
> **Edit:** None (optional: merge into the next sentence — 'For each user we order their check-in sequence in time and
> form short windows of nine consecutive visits.').
>
> Authour: We must keep the line, but the reviewer here was talking about metion that we are working with checking
> sequences, may this worth to be metion so we can caputre the attention of works that are intresting on it. what do you
> think?

**55.**
> **Highlight:** "The first is its category, the kind of place, one of seven fixed labels: Community, Entertainment,
> Food, Nightlife, Outdoors, Shopping, and Travel."
> **Germano:** o outro dataset tem essas mesmas categorias?
> **Action:** Confirm the other dataset (Istanbul) uses these same categories.

> **Response — ✅ ACCEPT:** Good question, and the answer is yes: Istanbul (Massive-STEPS) is collapsed to the same seven
> root categories, so the category task is identical across all datasets. Right now the reader has to assume that, so
> I'll
> state it explicitly.
>
> **Edit:** After '...Shopping, and Travel.' add: 'Istanbul uses these same seven categories via the Massive-STEPS
> seven-root mapping, so the category label set is identical across all datasets.'

**55.1[AUTHORS]**
>
> "We do not predict the exact next place; these two coarser properties are easier to learn and, for most uses, enough."
>
>We have said this same phrase at least three times until here. Should we keep this?
>

**56.**
> **Highlight:** "Region is a large one, ranging from about five hundred classes (Istanbul) to about eight thousand five
> hundred (California),"
> **Germano:** classes for region???
> **Action:** "classes" for regions is confusing — clarify the region-as-classification framing.

> **Response — ✅ ACCEPT:** Fair — 'classes' reads oddly attached to region until the reader registers that we frame
> next-region as classification, where each candidate census tract is one class. §3 half-says this ('a fine-grained
> choice
> over many candidate regions') but 'classes' lands first, so I'll make the region-as-classification framing explicit at
> first mention. This also helps the non-ML reader.
>
> **Edit:** Reword to: 'Region is a large set: each candidate census tract is one class, ranging from about five
> hundred (Istanbul) to about eight thousand five hundred (California),' then keep the existing 'fine-grained choice
> over
> many candidate regions' clause.

**57.**
> **Highlight:** "The motivation is practical, and the two predictions line up with two kinds of preparation a
> mobility-aware service makes."
> **Germano:** afirmação bem forte sem nenhuma citação
> **Action:** Strong claim with no citation — add a reference.

> **Response — 🟡 PARTIAL:** This is motivational framing, and §3 already self-disclaims it ('the application is the
> reason the predictions matter, not a result we claim'), so it doesn't strictly need a citation. Per the rule we apply
> at
> #17, we cite concrete application examples, never the framing sentence — but references.bib has NO mobility-aware
> content-staging/capacity-planning entry, and I will not fabricate one. So: leave the sentence as motivation now;
> attach
> a single verified reference to the concrete examples only IF a real one is sourced and added with a %verified
> provenance
> note.
>
> **Edit:** No cite added now (no verified mobility-aware-service reference exists in references.bib). If one is
> sourced, attach it to the 'content staging / capacity planning' examples only (not the framing sentence), with a
> provenance note; otherwise leave as motivation, consistent with #17.
>
> Authour: This is a problem the worth to keeping pending, I will solve it soon.

**58.**
> **Highlight:** "in hand" (With both **in hand** a service can act…)
> **Germano:** , (insert a comma)
> **Action:** Punctuation fix — add a comma.

> **Response — ✅ ACCEPT:** Correct — 'With both in hand' is an introductory phrase and needs a trailing comma. Trivial,
> uncontroversial fix.
>
> **Edit:** Change to 'With both in hand, a service can act ahead of regional demand...'

**59.**
> **Highlight:** "a census tract is a neighborhood, not a radio cell,"
> **Germano:** radio cellll???
> **Action:** "radio cell" feels out of place — reconsider.

> **Response — ❌ REJECT:** Keeping 'radio cell'. This paper targets a wireless/MobiWac audience, and the radio cell is
> exactly the spatial unit those reviewers think in — explicitly saying a census tract is NOT a radio cell is the
> deliberate scoping that stops a networking reviewer from expecting handover or cell-association results. It reads as
> stray jargon to an ML eye, but for the actual reader it's audience-serving and load-bearing.
>
> **Edit:** none (rejected)
>
> Authour: We should at least add a gloss for explain why we are refering to radio cell, so the reader can understand
> the point we are trying to make.

**60.**
> **Highlight:** "and even the ten most likely tracts are far too coarse to drive cell association or handover, so we
> keep the motivation at the level of regional demand and load anticipation, content staging, and capacity planning."
> **Germano:** deslocado do texto
> **Action:** Out of place — relocate or trim.

> **Response — 🟡 PARTIAL:** The scoping content stays — it's load-bearing per the section plan, right-sizing the
> motivation to tract granularity (demand/load anticipation, content staging, capacity planning) and ruling out
> handover/cell-association, which preempts the wireless reviewer's overclaim question. It only feels 'out of place'
> through an ML lens that doesn't value the cell-scoping. I'll grant the flow point: it's a single run-on, so I'll split
> it while preserving every clause.
>
> **Edit:** Split for flow, keep all content: 'These examples are sized to what we actually measure. A census tract is a
> neighborhood, not a radio cell; even the ten most likely tracts are far too coarse to drive cell association or
> handover. We therefore keep the motivation at the regional level: demand and load anticipation, content staging, and
> capacity planning.'
>
> Authour: I confess that this phrase is a bit confuse for me, worth a evalutaion
---

## Method (p3–4)

**61.**
> **Highlight:** "The pieces here are individually standard. Per-visit context, hierarchical place graphs, and the
> infomax objective all exist in prior work."
> **Germano:** sem necessidade
> **Action:** Unnecessary — cut.

> **Response — 🟡 PARTIAL:** I won't cut the concession — 'the pieces are individually standard' is exactly the
> BRACIS-defense that disarms the 'not novel' attack, and deleting it makes the combination claim float. But two
> standalone meta-sentences are heavy, so I'll compress into one. And I'll apply emit->produce here (#40): the merged
> sentence must say 'producing a vector per check-in', not 'emitting', or Related Work and Method disagree on the same
> action.
>
> **Edit:** Merge 04_method L33-39 into: 'Each ingredient is individually standard -- per-visit context, hierarchical
> place graphs, and the infomax objective all appear in prior work -- but the novelty is their combination: producing a
> vector per check-in from inside a hierarchical-graph-infomax representation, which earlier contextual embeddings built
> from sequence models (CTLE) do not provide. We show later that this combination, not extra supervision, is what makes
> the category task far easier to learn.'

**62.**
> **Highlight:** "Our one honest novelty is the combination: emitting a vector per check-in from inside a
> hierarchical-graph-infomax representation, a combination that earlier contextual embeddings built from sequence models
> do not provide, and we show later that this combination, not extra supervision, is what makes the category task far
> easier to learn."
> **Germano:** não tem necessidade. Pensa que você está estendendo um artigo. Não precisa de *honest novelty*
> **Action:** You're extending a paper — drop the "honest novelty" framing.

> **Response — 🟡 PARTIAL:** Agreed on tone, not on substance: 'our one honest novelty' reads over-apologetic, so I'll
> cut it to a plain 'the novelty is their combination'. But I'm keeping the scoping itself (combination, not bare
> per-visit context; the CTLE/sequence-model contrast; 'not extra supervision') because that honest scope is what
> protected us at BRACIS — dropping it re-opens the wound. 'Emitting' becomes 'producing' (#40).
>
> **Edit:** Within the #61 merge, 'Our one honest novelty is the combination: emitting...' -> 'the novelty is their
> combination: producing...'; retain the CTLE/sequence-model contrast and the 'not extra supervision' clause verbatim.

**63.**
> **Highlight:** "private per-task encoders"
> **Germano:** o que é private per task encoder?
> **Action:** Define "private per-task encoder".

> **Response — ✅ ACCEPT:** Fair — for the MobiWac/mobile-systems audience 'private' can read as a private network, so a
> one-clause gloss on first use helps both audiences. Define it inline rather than assume the reader knows the MTL
> idiom.
>
> **Edit:** Line 46-47: 'pass through private per-task encoders into a shared trunk' -> 'pass through private per-task
> encoders (a small input network dedicated to each task, with no weights shared between them) into a shared trunk'.

**64.**
> **Highlight:** "shared trunk,"
> **Germano:** shared trunk? você não explicou nada de MTL
> **Action:** "shared trunk" used before any MTL explanation — introduce MTL / hard-parameter-sharing first.

> **Response — ✅ ACCEPT:** Correct — 'shared trunk' is used before hard parameter sharing is introduced (the citation
> lands two paragraphs later). Glossing it at first use names the standard MTL setup for the networking reader and
> removes
> the forward reference; this pairs with defining MTL in the intro (#26). Then trim the later repeat so the definition
> lands exactly once.
>
> **Edit:** Line 47: 'into a shared trunk, where the two streams exchange information' -> 'into a shared trunk -- a
> stack of layers used by both tasks, the standard multi-task setup known as hard parameter sharing~
> \cite{caruana1997multitask} -- where the two streams exchange information'; then at line 65 trim 'This is ordinary
> hard
> parameter sharing~\cite{caruana1997multitask}' to 'kept deliberately plain...' so the definition is not repeated.
>
> Authour: We need to eval if we have explain what is MTL in some part of the code, I belive we haven't explain it well.

**65.**
> **Highlight:** "We want to be precise about what that private path is, because it is easy to misread."
> **Germano:** Sem necessidade dessa frase
> **Action:** Unnecessary sentence — cut.

> **Response — ✅ ACCEPT:** Agreed, this is throat-clearing — the actual content is in the next sentence; the lead-in
> announces an intention without adding information. Cutting it and going straight to the definition is tighter and
> loses
> nothing; the single-model clarification it precedes stays fully intact.
>
> **Edit:** Delete the sentence and start the paragraph at the content: 'The private spatial path is a task-specific
> branch inside the one model, not a second model, not a second representation, and not a route that swaps models per
> task.'

**66.**
> **Highlight:** "swaps models per task. The whole system is a single model: one set of shared parameters, one forward
> pass, two predictions. This is the deployment property we care about, since a service that wants both answers runs one
> model once, not two models in sequence."
> **Germano:** não entendi
> **Action:** Unclear — rephrase.

> **Response — 🟡 PARTIAL:** Keeping this — the single-model property (one set of shared parameters, one forward pass,
> two predictions) is the paper's primary deployment thesis and can't be softened. Your 'nao entendi' is likely 'two
> models in sequence', which is imprecise; the real contrast is two separate single-task models. I'll rephrase for
> clarity
> while preserving the claim.
>
> **Edit:** Replace 04_method L55-58 with: 'The whole system is a single model: one set of shared parameters and one
> forward pass produce both predictions. This is the deployment property we care about -- a service that wants both
> answers runs a single model once, instead of two separate dedicated single-task models.'


**66.1**
> "Our category metric, macro-F1, already counts every class equally, so re-weighting the loss to favor rare classes
> would optimize a different objective than the one we report; we keep the objective and the metric aligned and let the
> equal-weight sum balance the two tasks.""
>
>
> Authour: And the ACC@10 for the region not count every class equally ?; Also about the distribution of the classes is
> important to eval in the code and write about the stratification for category;

**67.**
> **Highlight:** "The cost of serving both tasks this way is small. One joint model has about five percent more
> parameters and compute than a single dedicated model, far less than the two separate models the usual setup would
> deploy, so two answers come at roughly the price of one."
> **Germano:** tentar reescrever essa subseção. Me pareceu bem deslocado
> **Action:** Rewrite this subsection — reads as misplaced.

> **Response — 🟡 PARTIAL:** I won't rewrite the subsection, but the +5% figure is a genuine selling point for a
> MobiWac/edge audience and backs the single-model thesis, so it stays. The 'misplaced' feel is real — it reads orphaned
> after the loss paragraph — so I'll move it to sit right after the single-model paragraph (#66), and normalize 'single
> dedicated model' to 'dedicated single-task model' (#9).
>
> **Edit:** Relocate the cost paragraph to immediately follow the single-model paragraph (#66) and edit: 'One joint
> model has about five percent more parameters and compute than a single dedicated model' -> '...than a dedicated
> single-task model'.
>
> Authour: Search and comprove it in the codebase and with experiments to be assure!
---

## Results (p7)

**68.**
> **Highlight:** "Check-in level | Place level" (Table II — Check2HGI vs HGI columns)
> **Germano:** é uma comapração quase injusta. Porque a ideia aqui é bem diferente não?! São dois targets completamente
> diferentes. São duas modelagens diferentes
> **Action:** Comparison may be almost unfair — the two are very different ideas/targets/models. Justify comparability
> or reframe.

> **Response — 🟡 PARTIAL:** Rejecting the substance — this is a controlled comparison by construction, not unfair: both
> columns predict the SAME next-category macro-F1, through the SAME matched single-task head, on the SAME folds and
> windowing; only the input representation changes (vector-per-visit vs vector-per-place HGI). The targets and model are
> not different — that's what makes the margin attributable to the representation. But your misreading signals a
> reviewer
> could misread too, so I'll add one explicit comparability sentence to §6.1.
>
> **Edit:** Add to §6.1 after the comparison sentence: 'This is a controlled, like-for-like comparison: the target (
> next-category macro-F1), the single-task head, and the cross-validation folds are identical across the two columns;
> only
> the input representation changes (a vector per visit versus a vector per place), so the margin isolates the
> representation rather than a difference in task or model.'

**69.**
> **Highlight:** "Category separability of the two representations. The check-in-level representation cleanly separates
> the seven categories where the place embedding does not, which is exactly why it helps the category task and is
> neutral
> on region." (Fig. 3 caption)
> **Germano:** how many clusters? kNN accuracy em category separability?
> **Action:** State the number of clusters; clarify why kNN accuracy measures "category separability".

> **Response — ✅ ACCEPT:** Fair question, and worth fixing. There are seven groups, not discovered clusters — points are
> colored by the seven ground-truth category labels, not a k-means k. Separability is quantified two ways in §6.1:
> silhouette-by-category (0.56 vs 0.00) and nearest-neighbor category purity (0.98 vs 0.78), the latter being the
> kNN-style measure you're asking about. The caption asserts separability without the numbers; I'll lift them in so the
> figure is self-contained.
>
> **Edit:** Replace the Fig. 3 caption with: 'Category separability of the two representations (per-visit and per-place
> vectors projected to two dimensions, colored by the seven categories). The check-in-level representation cleanly
> separates the categories where the place embedding does not: silhouette-by-category $\approx 0.56$
> versus $\approx 0.00$, and nearest-neighbor category purity (the share of each vector's nearest neighbors that share
> its
> category, a kNN-style separability measure) $\approx 0.98$ versus $\approx 0.78$. This is exactly why it helps the
> category task and is neutral on region.' (Match the projection-method wording to whatever fig3_embquality.py renders.)
>
>
> Author: We need to explain this better. Fisrt we need an explanation for the methodology since knn is not a clustering
> and in the way that is been presented it's seems like a clustering; Then, `Category separability` is this the correct
> and most clear name ?; This two image in the same Y axis make sense and is it the best way to present this?(question
> to eval and treat it carrefully).
---

## References (p9)

**70.**
> **Highlight:** "Joint Conference on Artificial Intelligence (IJCAI), pp. 3081–3089, 2025." (Ref [26])
> **Germano:** Estou com medo das referências estarem ultrapassadas
> **Action:** Worried the references may be outdated — review for recency.

> **Response — ✅ ACCEPT:** Good instinct to spot-check, and the recency pass is done — but the references aren't
> outdated. I counted the actual citations: 26 unique \cite keys across all sections/tables/figures (references.bib
> holds
> 33 entries; 7 are uncited), so the paper cites 26 references, all web-verified with %verified provenance notes. The
> span
> you flagged is in fact the NEWEST entry — ReHDM (li2025rehdm, IJCAI 2025), our next-region reference — alongside
> Massive-STEPS (2025), MCMG (TOIS 2024), and CSLSL (2024). No edit needed; the 26-vs-27 ambiguity resolves to 26
> actually-cited.
>
> **Edit:** No paper edit. Recency review complete: 26 cited references (grep-verified; 33 bib entries, 7 uncited), all
> web-verified and current; newest = ReHDM (IJCAI 2025) = the highlighted entry. Reference list stands.
---

Other points to review:

- comentamos sobre o fato de a transferencia ser ortogonal ? E por isso não usamos o gradient balancers ?
- Revisar explicações de abrevisacoes repetidas no texto(e.x POI)
- Revisar conceitos técnicos sem gloss ou explicações formais
- Revisar coesão entre chapters, varios capítulos estão explicando mesmo conceitos
- We need to do a systematic review of the english in the paper and the phrases in the paper. some phrases are very confusining, other is missing connectives like the: `We never  feed STAN our representation;` that seems that is missing a `with` 

----

*All 70 highlight comments, in reading order. Highlighted spans recovered via word-level
bounding boxes (PyMuPDF) so each maps to exactly the text under the highlight. Verbatim
Portuguese preserved with English action glosses.*
