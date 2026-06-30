# review_v2 — the broad "other points" (task catalog)

> **What this is.** The author's *broad-scope* review items, the ones that are **not** a single per-comment prose
> tweak. They come from two places: the `Other points to review` list at the bottom of
> [`../REVIEW_GERMANO.md`](../REVIEW_GERMANO.md), and the inline `Authour` considerations that need investigation
> (codebase scrape, web/literature, a figure rebuild, an experiment) before any prose can be written.
>
> **Two task families:**
> - **OP — section-spanning text sweeps** (one or more dynamic workflows over `src/sections/`): consistency,
>   redundancy, gloss coverage, English. They *find and propose*; they do not edit the paper.
> - **INV — investigations** (Explore / general-purpose / web agents over `src/`, `research/`, `docs/`, and the
>   literature) that produce a *fact* the paper needs (a measured number, a class distribution, a precedent check).
>
> **Status legend:** `TODO` (not started) · `READY` (scoped, can launch) · `RUNNING` · `BLOCKED` · `DONE`.
> Nothing here has been launched yet — this is the worklist. Sequencing + workflow shapes are in
> [`PLAN.md`](PLAN.md); cross-references and decisions land in [`LOG.md`](LOG.md).

---

## OP — section-spanning text sweeps

### OP1 — Orthogonal transfer ↔ why no gradient balancers
- **Source:** Other points #1 — *"comentamos sobre o fato de a transferencia ser ortogonal? E por isso não usamos o gradient balancers?"*
- **Question:** Does the paper say (a) that the cross-task transfer is **orthogonal / not antagonistic** (the easy
  and hard tasks do not fight over the shared trunk), and (b) that this is **why** we use a plain unweighted fixed
  weighting instead of a gradient balancer (PCGrad/GradNorm/Nash-MTL)? Right now §2.3 says only that balancers
  "rarely beat a well-tuned fixed weighting at two tasks", which is the *literature* reason, not *our* finding.
- **Scope:** mixed — a doc/codebase scrape for the actual finding, then a small text decision.
- **Attack:** INV-style scrape of `docs/` (the repo's "MTL regime finding" / orthogonal-gradients result — see the
  root memory `mtl_regime_finding.md` and `docs/studies/.../FINAL_SYNTHESIS.md`) to confirm what we can honestly
  claim; then decide whether to add one sentence to §2.3 or §6.2. **Honesty guard:** the glossary bans "orthogonal
  gradients" as a repo word and bans "Pareto" — any added sentence must be in plain words and must match what the
  data actually shows (the W6 encoder-isolation probe says category is a stronger encoder, not transfer).
- **Output:** a yes/no on whether we make the claim, the exact sentence if yes, and the source.
- **Depends on / cross-ref:** overlaps INV3 (loss justification) and the §6.2 "stronger trunk, not transfer" finding.
- **Status:** TODO

### OP2 — Repeated abbreviation expansions
- **Source:** Other points #2 — *"Revisar explicações de abreviações repetidas no texto (e.x POI)"*
- **Confirmed evidence:** **POI** is expanded ("point of interest (POI)") **3×** — `01_introduction.tex:22`,
  `02_related.tex:11`, `04_method.tex:14`. Expand **once** (first use, intro), then "POI". Same check for **LBSN,
  MTL, CV, HGI, DGI, Acc@10, macro-F1, TOST, pp**.
- **Scope:** pure text sweep across all 8 sections + abstract + tables + figure captions.
- **Attack:** WF-CONSISTENCY (see PLAN). For each acronym: list every occurrence in reading order, mark the
  intended first-use site (per GLOSSARY §2), flag every *re-expansion* and every *use-before-expansion*.
- **Output:** a per-acronym occurrence map + the exact deletions/edits (e.g. "drop '(POI)' at related:11 and
  method:14").
- **Cross-ref:** GLOSSARY §2 / §6 checklist. Coordinate with OP3 (gloss coverage) — same readers.
- **Status:** TODO

### OP3 — Technical concepts without a gloss or formal explanation
- **Source:** Other points #3 — *"Revisar conceitos técnicos sem gloss ou explicações formais"*
- **Scope:** every technical term the networking audience may not know, checked for a one-time plain gloss at first
  use. Seeds: **MTL / hard parameter sharing** (author #64 — currently expanded but never *defined*), **infomax
  objective**, **silhouette score**, **nearest-neighbor purity**, **macro-F1**, **Acc@10**, **transductive**,
  **stratified k-fold CV**, **region-transition prior**, **cross-attention**, **dual-tower head**, **TOST /
  non-inferiority**, **Wilcoxon**.
- **Attack:** WF-CONSISTENCY (gloss-coverage pass). For each term: first-use location; is it glossed; is the gloss
  GLOSSARY-compliant (plain words, no repo jargon); is any term glossed *more than once* (that's an OP4 redundancy).
- **Output:** a "term → first use → glossed? → fix" table.
- **Absorbs:** **INV5** (verify MTL is actually defined) lives here.
- **Cross-ref:** GLOSSARY §3 (jargon→plain table) is the rubric. Author #63/#64 already cover two of these.
- **Status:** TODO

### OP4 — Cross-chapter cohesion / the same concept explained in several places
- **Source:** Other points #4 — *"Revisar coesão entre chapters, varios capítulos estão explicando mesmo conceitos"*
- **Confirmed redundancy seeds:**
  - "we do not predict the exact next place" ×3 → **already ruled** (CC5: keep one). Verify it's actually one after
    the prose pass.
  - "one fixed vector per place, two visits to the same coffee shop/café look identical" — appears in abstract, §1
    (`:39-41`), §2.1 (`:18-19`), §4.1 (`:14-15`). The café image is repeated; decide where it lives once.
  - the single-model / "one model, one forward pass, two predictions" property — §1, §4.2, §6.2, §7. Likely
    intentional (it's the thesis) but check for verbatim repetition vs. deliberate restatement.
  - the +28..+40 representation margin and the per-visit-context share — abstract, §1 contribution, §6.1, Tbl 2.
  - the "stronger trunk, not transfer" reading — §6.2 + §7.
  - "trunk" wording consistency (author #31) — **already satisfied** once the single `01:46` "semantic trunk" edit
    lands (4/5 sites already say "shared trunk"); just verify, and do NOT touch `05_setup:89` "HMT-GRN-style trunk".
  - **superiority-verb family consistency (from CC3/AN8_10)** — `{beats, beat, wins, win, "spatial win", "spatial
    gain"}` must end up as ONE verb; `05_setup:68` ("meant to win") binds the verb to the test and Table III's caption
    carries both "win" and "beat". This is the OP4 enforcement arm of the CC3 sweep.
  - **§3-vs-§5.3 "tracts can't drive handover" redundancy (from #59/#60)** — the tract-granularity scoping appears in
    both §3 motivation and §5.3 (the TOST-margin deployment rationale). Pick one home.
- **Attack:** WF-CONSISTENCY (redundancy matrix): cluster sentences by concept across sections, mark
  {canonical-home, acceptable-restatement, cut}. **Distinguish** harmful duplication from deliberate thesis
  reinforcement — do not flatten the spine.
- **Output:** a concept→locations matrix with a keep/cut recommendation per location.
- **Cross-ref:** CC4 (coarser), CC5 (scope statement), author #31 (trunk).
- **Status:** TODO

### OP5 — Systematic English + phrasing pass
- **Source:** Other points #5 — *"systematic review of the english ... some phrases are very confusing, others
  missing connectives, e.g. `We never feed STAN our representation;` seems to be missing a `with`."*
- **Scope:** grammar, missing connectives/prepositions, confusing phrases, American English (GLOSSARY §6),
  **and the no-em-dash rule (CC1)** — including the **live `---` at `06_results.tex:57-58`** and any `--` the prose
  pass introduces. Note the cited example: "We never feed STAN our representation" (`05_setup.tex:93`) — "feed X Y"
  is grammatical, but the sentence is worth a clarity check.
- **Attack:** WF-ENGLISH — per-section language reviewers, each flagged phrase adversarially re-checked (is it
  actually wrong, or is the reviewer over-correcting?), then a consolidated edit list. **Run LAST**, after OP1–OP4
  and the per-comment prose edits land, so we don't polish sentences that are about to change.
- **Output:** a consolidated, line-anchored edit list (American-English + connectives + em-dash + clarity).
- **Cross-ref:** CC1; GLOSSARY §6 checklist.
- **Status:** TODO (run last)

---

## INV — investigations (produce a fact the paper needs)

### INV1 — Is "next-region over census tracts" really low-precedent? (web / literature)
- **Source:** author #50 — *"Search on the web/literature to be more sure about this."*
- **Question:** Verify the claim that fine-grained region (census-tract-scale) as a **co-equal end target** (not an
  auxiliary coarse cell, as HMT-GRN uses geohash) has little precedent. Find any counter-examples; pin down the
  honest wording.
- **Attack:** the `deep-research` skill (fan-out web search + adversarial verify), scoped to next-region / region
  prediction / hierarchical POI prediction / census-tract or administrative-unit prediction.
- **Output:** a short cited finding → either "claim stands" or "soften to: «to our knowledge»" or "cite paper X".
- **Blocks:** the final §2.2 wording (#50). Interim wording already set in REVIEW_GERMANO (#50 FINAL).
- **Status:** ✅ DONE (2026-06-28). Coarse-cell-as-main-target IS precedented (TrajLearn ACM TSAS 2025; geohash/grid
  next-cell; cellular next-cell). The **owned** combination (fine administrative unit + co-equal with category + no
  next-POI) is what is underexplored. Action: keep the **method contrast** (auxiliary coarse cell vs co-equal end
  target), soften any precedence claim to "to our knowledge … underexplored"; **do NOT add a TrajLearn bib entry**
  (bib locked at 26); "underexplored" is outside `PAPER_PLAN §3` CAN-say, so either add it there with the hedge or
  keep only the contrast.

### INV2 — Verify "+5% more parameters and compute" (codebase + experiment)
- **Source:** author #67 — *"Search and comprove it in the codebase and with experiments to be assure!"*
- **Question:** Is the joint model really ≈+5% params **and** compute vs a dedicated single-task model? Get both
  numbers (params and FLOPs), for the **paper's** model (`mtlnet_crossattn_dualtower` + `next_gru` cat head +
  `next_stan_flow_dualtower` reg head) vs one dedicated single-task model.
- **Attack:** Explore/general-purpose agent over `src/models/` + `src/utils/flops.py` (FLOPs + param counts), the
  registry (`src/models/registry.py`), and the board recipe. Compute joint-vs-STL for params and FLOPs; reconcile
  "params" vs "compute" (they may differ — the claim says both).
- **Output:** measured params Δ% and FLOPs Δ%, the exact comparison basis, and the corrected sentence for §4.2.
- **Blocks:** the §4.2 cost paragraph (#67). Mark "+5%" provisional in LOG until this returns.
- **Status:** ✅ DONE (2026-06-28). **+5% is FALSE.** vs ONE dedicated model the joint is **+38% to +90% params /
  +22% to +74% FLOPs** (it carries both heads + cross-attention + the dual-tower spatial path); the "+5%-scale" only
  describes the gap to TWO separate models, where the joint is actually *cheaper* (~−28% FLOPs / −17..20% params).
  Action = **Option A**: drop the magnitude, keep "cheaper than two dedicated models", change "two answers at the
  price of **one**" → "well under the price of **two**". Ripples: `04_method:5` header comment, `PAPER_PLAN:393/745`,
  and the v1 #67 rationale that called +5% "a selling point" (falsified).

### INV3 — Class distribution, stratification, and the loss/metric justification (codebase)
- **Source:** author #66.1 — *"the ACC@10 for the region not count every class equally?; the distribution of the
  classes is important to eval in the code and write about the stratification for category."*
- **Questions:** (1) the **7-category** class distribution per state (how imbalanced?); (2) is the category task
  **stratified** in the folds (`src/data/folds.py` — StratifiedGroupKFold) and on what key; (3) what does **Acc@10**
  actually weight (instance-level → frequency-weighted, *not* class-balanced); (4) does class-weighting the
  **category** loss help or hurt **macro-F1** in our runs (the §4.2 "keep objective and metric aligned" argument is
  imprecise — macro-F1 is balanced but unweighted CE is not, so the honest justification may be empirical, not
  metric-alignment).
- **Attack:** Explore/general-purpose over `src/data/folds.py`, `src/data/inputs/`, `src/losses/`, plus the repo
  memory/findings on class-weighting (root memory `mtl_category_loss_unweighted.md`, the C25 class-weighting
  confound in the mtl_improvement study).
- **Output:** the class-distribution numbers + a one-paragraph factual basis to **rewrite the §4.2 loss
  justification** honestly, + a sentence for §5.1/§5.2 on stratification.
- **Blocks:** §4.2 loss paragraph (#66.1) and possibly OP1 (orthogonality framing).
- **Status:** ✅ MOSTLY DONE (2026-06-28). (1) region Acc@10 is **instance-level / frequency-weighted**
  (`mtl_eval.py:57-61`, `hit.float().mean()`). (2) The §4.2 metric-alignment reason is **backwards for category**
  (macro-F1 is class-balanced, so unweighted CE is the *misaligned* choice there; it is aligned only for region). (3)
  Both heads ARE unweighted in the board recipe (`--no-{reg,cat}-class-weights`), so the *factual* "unweighted" claim
  stands; only the *reason* is wrong. (4) Honest defense is **empirical** (C25, `docs/CONCERNS.md:590`:
  class-weighting tested, hurt both — region −10..14pp, category macro-F1 +5.14pp at AL weighted→unweighted), kept
  **qualitative** (C25 ran on champion-G non-overlap, not the board cells). (5) Folds: StratifiedGroupKFold,
  user-grouped, **category-stratified** (`folds.py`); region not the strat key. **Replacement §4.2 prose + a
  stratification sentence are drafted in the workflow output.** STILL OPEN (optional): a printed **per-state
  7-category distribution table** — none exists in docs; needs a light groupby on the category column (currently
  unsourced).

### INV4 — Figure 3 design, methodology, and naming
- **Source:** author #69 — *"knn is not a clustering … `Category separability` is this the correct/clearest name? …
  two images on the same Y axis make sense / is it the best way?"*
- **Questions:** (1) does `figs/fig3_embquality.py` actually plot what the caption/text claims (silhouette +
  nearest-neighbor purity, on ground-truth labels, **not** k-means clusters)? (2) is a **shared Y axis** for the two
  panels appropriate/misleading? (3) is **"Category separability"** the clearest title, or is there a better one
  ("class separability of the representations")?
- **Attack:** focused agent — Read `figs/fig3_embquality.py` + render `fig3_embquality.pdf`; propose a corrected
  figure (axis/labels/title) and the matching caption + §6.1 prose (clustering-free).
- **Output:** verdict on the current figure + a concrete regeneration spec (the `.py` is a committed input, so any
  change re-renders the PDF). Prose/caption fixes already specified in REVIEW_GERMANO (#69 FINAL).
- **Note:** this is the one INV that may **edit a committed artifact** (the figure) — but only when the author
  greenlights it; for now it produces a spec.
- **Status:** ✅ INSPECTED (2026-06-28) — **cosmetic only, no clustering bug.** `fig3_embquality.py:38-41` is a
  grouped bar chart of two **precomputed** constants per representation (cosine silhouette on the ground-truth
  category partition, `geometry.py:169`; leave-one-out cosine kNN purity, `geometry.py:46-94`); it loads no
  embeddings and runs no k-means. Prose already frames them correctly (`06:31,33`). Fixes: relabel the shared y-axis
  (two different metrics on one 0–1 scale), rename the kNN tick to "kNN purity (k=10)", add a caption clause that
  both are computed on the true labels (no clustering). The plotted constants must NOT change. **Regenerate the
  committed `fig3_embquality.pdf`** after the label edits (deferred to Phase 3, on author go).

### INV5 — Is MTL actually defined plainly? (text)
- **Source:** author #64 — *"We need to eval if we have explain what is MTL in some part of the code, I believe we
  haven't explain it well."* (Cross-check #26.)
- **Resolution:** **folded into OP3** (gloss-coverage). **✅ ANSWERED (2026-06-28, via AN64):** no plain-words MTL
  definition exists anywhere; `01_introduction.tex:35` only expands the acronym, and "shared trunk" (`01:46`, reused
  4×) is never glossed. Concrete edits: add an MTL gloss at `01:35` + a "shared trunk" gloss at `01:46`, **exactly
  once each**, do NOT touch `04_method:65`. Apply in Phase 3.
- **Status:** ✅ DONE (edits specified; apply in the prose pass)

### INV6 — Mobility-aware-service citation (author-owned, pending)
- **Source:** author #57 — *"This is a problem worth keeping pending, I will solve it soon."*
- **Action:** the author will source a real reference for the content-staging / capacity-planning examples. When
  found, attach it to the **concrete examples** only (not the framing sentence), with a `%verified` provenance note.
- **Attack:** none from us unless asked; track only.
- **Status:** PENDING (author)

---

## Quick map: author note → task

| REVIEW_GERMANO item / list | Routed to |
|---|---|
| Other points #1 (orthogonal / no balancers) | OP1 |
| Other points #2 (repeated abbreviations) | OP2 |
| Other points #3 (un-glossed concepts) | OP3 |
| Other points #4 (cross-chapter cohesion) | OP4 |
| Other points #5 (English / connectives / em-dash) | OP5 (+ CC1) |
| #50 (census-tract precedent) | INV1 |
| #67 (+5% params/compute) | INV2 |
| #66.1 (Acc@10 / stratification / loss) | INV3 |
| #69 (Fig 3 design) | INV4 |
| #64 / #26 (MTL defined?) | INV5 → OP3 |
| #57 (motivation citation) | INV6 (author) |
