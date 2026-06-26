# MobiWac 2026: Paper Plan (working draft v1)

> **What this is.** The storytelling plan, the section blueprint, and draft paragraphs for our MobiWac 2026
> submission. It is a plan, not the paper. Vitor will review and revise; every draft paragraph is a starting
> point in his voice, not final text. Companion file: [`GLOSSARY.md`](GLOSSARY.md).
>
> **v1 changes (from Vitor's review):** tasks are named **next-category** and **next-region** (the literature
> terms, matching CBIC/CoUrb and the field, web-checked), not activity/area. The mobility-management angle is kept
> as a **storytelling and motivation thread** (intro, problem, and discussion) to highlight what the tasks are
> for, but it is never a **claim, contribution, or measured result** (we have not surveyed that literature and we
> run no network experiment). Citations are attached to the draft claims. A real dataset table is added. The
> windowing note moved into the data section. The no-leakage of the representation is stated as standard protocol.
> Punctuation avoids the em-dash, per Vitor's style.
>
> **v2 changes:** the contributions now claim a **novel check-in-level representation** and a **novel multi-task
> architecture**, scoped honestly (contextual embeddings like CTLE exist, so the representation's novelty is the
> specific combination; the architecture is novel for these tasks, not a new machine-learning mechanism). The weak
> third contribution (an "honest correction") was dropped and replaced by a true one: the cross-scale and
> cross-city account of when joint training helps and when it costs. **All references to the rejected prior
> submission were removed from the paper prose** (it is not published, so the paper cannot lean on it); the §9
> traceability is internal only.
>
> **v3 changes (2026-06-24 board results: the story got much stronger):** the new board confirms the single model
> **beats the dedicated category ceiling at every state (+4.7 to +7.7)** AND **beats the dedicated region ceiling
> at the large states** (FL, CA, TX, all 5 folds) while **matching within two points at the small**
> (AL, AZ, Istanbul). The largest region state (CA, 8,501 regions) beating **retires the old "cost grows with
> region count" framing**; that earlier cost was a precision artifact (fp16). The region claim flips from
> "non-inferior" to "**beats at the large states, matches at the small**", which **reopens the regular track**
> (the poster cut, `PAPER_PLAN_POSTER.md`, is now the deadline fallback, not the recommendation). Baselines locked
> (§7 + `BASELINE_HANDOFF.md`): category = POI-RGNN + Markov-9-cat; region = ReHDM + STAN-`stl_hgi` (Markov-1
> region dropped); representation control = CTLE (FL) + feature-concat; MTL-design comparator = CSLSL cascade. (Region externals later settled: HMT-GRN primary, STAN faithful + ReHDM secondary.)
> All Part-2 numbers are n=5 (seed 0) provisional; the board is complete (TX closed at 5 folds, region +2.06).

---

## 0 · Status, decisions, and the venue

**Three decisions, as revised:**

| Decision    | Choice                                                                                                                                                                                                                                                                                                                                                                                                      |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Framing** | A **multi-task POI-prediction paper on location-based social networks**, written for a mobility-aware audience. Mobile and urban services (anticipating where and what a user will do next) are the **motivation**, used through the storytelling (intro, problem, discussion) to show what the tasks are for. They are never a claim or a contribution, and the paper makes no network-system measurement. |
| **Story**   | Two parts. Part 1: a check-in-level representation makes next-category prediction far easier. Part 2: a single model that predicts next-category and next-region together beats a dedicated model on category at every state (+4.7 to +7.7), and on region beats the dedicated model at the large region counts (FL, CA, TX) while matching it within two points at the small (AL, AZ, Istanbul). One model wins both tasks, and the region win grows with scale (CA, the largest, beats by +2.18). Provisional (n=5, seed 0).                                                                          |
| **Scope**   | Five Gowalla states (AL, AZ, FL, CA, TX) plus Istanbul (Massive-STEPS) for external validity.                                                                                                                                                                                                                                                                                                               |

**The venue, honestly.** MobiWac is a wireless mobility-management and network-access symposium. We verified the
2026 call and the 2020 to 2025 proceedings. A pure recommendation paper risks a quick reject, but our topic is
not homeless: the call lists *"AI-based mobility management"*, *"Mobility models, control and management"*, and
*"Social mobile networks and applications"*, and there is precedent (the 2020 Best Paper was a location-based
mobility study; a 2021 finalist was a neural mobility-prediction model). Because we keep the network angle as
motivation rather than as a measured contribution, the paper rests on this mobility-prediction precedent and on
the strength of the empirical study, not on a system claim. That is the honest trade we are making, and the now
much stronger result (one model beating both dedicated models, the region win largest at scale) is what carries
it: with the largest region state measured and winning, the headline is no longer a forecast, so the **regular
track is the target** and the poster cut is the deadline fallback (§10).

**Format (verified):** IEEE two-column template (not ACM; ACM is poster-track only); 8 pages for a regular paper
(10 with a fee); single-blind, so author names stay; EDAS; Paris, 26 to 30 October 2026. **Open action:**
reconfirm the submission deadline on the MobiWac site (our notes say about 25 June 2026, unverified, and it may
force the poster track or a later cycle).

---

## 1 · The thesis and the title

**Thesis.** *A check-in-level representation, where each visit is described in its own context, makes the
next-category task far more learnable than a fixed per-place embedding; and a single model that learns
next-category and next-region together beats a dedicated model on category at every state and beats it on region
where the region space is large, while matching it where the region space is small. One model wins both tasks,
and the spatial win is strongest at scale.*

**Working title (leads with the science, keeps a mobility-aware nod):**
*Predicting the Next Category and Region of a Visit: A Check-in-Level Multi-Task Study on Mobility Data.*

Alternatives:

- *One Model, Both Tasks: A Check-in-Level Multi-Task Model that Beats a Dedicated Next-Category Model and Matches
  or Beats on Next-Region.* (leads with the finding; honest about region, which beats at the large states and is
  non-inferior at the small. Never let a title imply region is beaten everywhere.)
- *Anticipating the Next Category and Region of a Visit: A Check-in-Level Multi-Task Model for Mobility-Aware
  Services.* (leans into the usage and motivation framing; acceptable as long as we never claim a measured
  service result.)
- *Where Sharing Wins: A Check-in-Level Multi-Task Study of Next-Category and Next-Region Prediction.* (the old
  "where it costs" framing is now weaker; the win, not the cost, is the story.)

Avoid a title led by the model name or by an architecture term; a mobility audience prefers a title that names
the finding.

---

## 2 · The storytelling spine

We tell it as question, evidence, consequence, in two parts, with one figure and one table each. The reader
should never have to assemble the story from dense tables (Reviewer 2's main complaint last time).

**The hook (motivation).** Location-based social networks record where people go and what they do there, one
check-in at a time. If we can anticipate the next move, mobile and urban services can act ahead of it. Two
coarse questions are enough for most of that: the next *category* (the kind of place) and the next *region*
(which part of the city). They are easier to learn than the exact next place, and they pair naturally, one for
intent and one for
geography. The open question is whether one model should learn both, and whether sharing helps or hurts.

**Part 1: the representation makes the category task learnable (frozen result).** Most place embeddings give
every place one fixed vector, so two visits to the same café look identical. We instead describe each *visit* in
its own context (time, nearby places, recent trail). This lifts next-category macro-F1 by a large, consistent
margin over the standard hierarchical-graph embedding, about +28 to +40 points across all five states, and we show
why: most of the gain (about 64 to 90 percent) is the per-visit context itself, not extra training signal. On
next-region the two representations are even, so the benefit is uneven across the two tasks, and that asymmetry
sets up Part 2.

**Part 2: one model, two tasks, and it wins both (the scientific core).** We train one model that predicts both
tasks at once (one model, one forward pass, two answers), and we compare it against strong dedicated single-task
models. On **category** the joint model beats the dedicated ceiling at every state, by about +4.7 to +7.7 points.
Our reading is that the shared trunk is simply a **stronger category encoder**, obtained at no extra deployment
cost (one model, one forward pass), rather than the region task teaching category. The encoder-isolation probe
backs this as a **finding**: freezing the region pathway at start leaves the full category lift intact at AL, AZ,
and FL (within 0.3 of the joint model, far above the dedicated ceiling), so the gain is a stronger shared trunk,
not region-to-category transfer. On **region** the joint model **beats** the dedicated ceiling
where the region space is large (FL +0.57, CA +2.18, TX +2.06, all 5 folds) and **matches** it
within a two-point margin where the region space is small (AL −0.18, AZ −0.06, Istanbul −0.52; each a **tested
equivalence**: paired TOST non-inferior at δ=2 pp, all three 90% CIs within ±0.7 pp — `STATISTICAL_PROTOCOL.md §3.4`). The cost we once
expected at scale does not appear: the largest region space (CA, 8,501 regions) is exactly where the joint model
wins most. We report the region win as an empirical finding ordered by region count, and we control for the
region-count versus data-density confound rather than claim a law.

A reader steeped in multi-task learning expects the harder task to pay for the easier one. Here it does not: once
the model and the loss are set up cleanly (an unweighted loss and a private spatial path), the spatial task is
matched at small scale and beaten at large scale, while the semantic task is beaten everywhere. We say plainly
that these are seed-0, five-fold results with a multi-seed top-up still to come, and we never claim more than that.

**The takeaway.** One shared model is the right tool for **both** tasks. The semantic side can be shared cheaply,
and at scale the shared model helps the spatial side too; the spatial win is largest where the region space is
largest. For a mobility-aware service this reads as a simple design hint, one model can anticipate both what and
where, but we present it as an observation, not as a measured system result.

---

## 3 · Claim discipline (read before writing any number)

This is the section that keeps every number in the paper honest and defensible.

**We CAN say (frozen, paper-grade):**

- The check-in-level representation (Check2HGI) beats HGI on next-category macro-F1 by **+29.31 / +27.63 /
  +39.62 / +37.95 / +37.47** (AL/AZ/FL/CA/TX), paired Wilcoxon p = 0.031 (5/5 folds) each, **under the paper's
  overlap windowing** (re-score COMPLETE for all 5 Gowalla states; engine `check2hgi_dk_ovl`, seed 0 × 5f). Two
  bands: about +28 to +29 at the small states, about +37 to +40 at the large ones; the HGI place-level category is
  a consistent ~0.46 to 0.52 times the Check2HGI check-in-level category. (These supersede the older non-overlap
  measurement, which was +15 to +29; the whole paper is now one windowing.) JSONs
  `docs/results/closing_data/baseline_compare/{state}_hgi_ovl_cat.json`.
- About 64 to 72 percent (small states) and 89 to 90 percent (large states) of that category gain is the
  per-visit context, measured against a per-place-averaged version of the same embedding.
- On next-region the two representations are within about 1.6 to 3.1 points (HGI slightly ahead), so the
  representation's benefit is category-only.
- (Optional, text only.) Our models clear a simple Markov-1 transition floor by a wide margin on region. We
  **dropped Markov-1 from the region table** (we already sit above HMT-GRN, a faithful STAN, and a ReHDM reference,
  the stronger comparison), but a one-clause mention of the floor is available if a reviewer wants it. Do not attach
  the +28 to +40 figure here; that is the category-representation margin, a different axis.

**We CAN say, but must mark PROVISIONAL (reduced board: seed 0 × 5 folds, n=5; the {1,7,100} top-up to n=20 is
post-deadline):**

The single joint model versus the dedicated single-task ceilings, under the adopted overlapping-window board
(fp32-matched scorer):
- **Category: beats the ceiling at every state**, Δ = AL +7.69 / AZ +6.26 / FL +4.68 / CA +7.07 / TX +7.56 /
  Istanbul +6.69.
- **Region: beats at the large region counts, matches at the small.** The beat rests on **FL +0.57 (4,703
  regions), CA +2.18 (8,501 regions), and TX +2.06 (6,553 regions), all 5 folds**. Matches within two points: AL −0.18, AZ −0.06, Istanbul −0.52 (board, n=20; paired-TOST arm at s0 Δ=−0.50),
  **each a tested equivalence** — paired TOST non-inferior at δ=2 pp (AL p=7e-5, 90% CI (−0.46,+0.09); AZ p=1.5e-4,
  CI (−0.41,+0.29); Istanbul p=2e-5, CI (−0.65,−0.35)); all CIs sit well inside ±2 pp (`STATISTICAL_PROTOCOL.md §3.4`).
- **CA, the largest region state, is measured and beats**, and that single cell retires the earlier "cost grows with
  region count" reading. The earlier large region cost (and the TX −2.4 figure) was a precision artifact (fp16
  autocast / Ampere bf16), not a real trade-off.

**Caveats that travel with these numbers:** n=5 (seed 0 only); the small-state region matches are now backed by a
**paired TOST** (δ=2 pp, §3.4) rather than only an "n=5 provisional" label — the per-fold region σ is small
(0.16–0.37 pp) so the equivalence is well-powered (≈1.0 to declare a true match; ≥95% to reject a true 2-pp gap).
TX is now closed at 5 clean folds (fp32 single-device, region +2.06); never cite the void fp16/bf16 collapse JSONs.

**We must NOT say:**

- "beats region everywhere" or "Pareto-dominates everywhere". At the small region counts (AL, AZ, Istanbul) the
  region result is a **match** (slightly negative), not a beat. The honest claim is: **beats category everywhere;
  beats region at the large states; matches region within a two-point margin at the small.**
- "ties" on region. Say "statistically non-inferior within a two-point margin" (TOST) for the small-state region
  cells, and "beats" (superiority, paired Wilcoxon) only for the large-state region cells that clear the ceiling.
- "cost grows with region count" / the cardinality-cost framing / the TX −2.4 figure. These are superseded; the
  earlier cost was an fp16 precision artifact, and CA (the largest) beats.
- Anything that headlines a two-model composite, per-task routing, or two-substrate setup. They break the
  single-model property, which is the point, so they are supportive evidence at most.
- The old region numbers (7 to 17) or any pre-2026-05 region numbers; they were leak-inflated or confounded.
- "trivial" or "padding" for the dropped overlap windows. They are valid examples; the gate removes an
  over-representation of each user's last place.
- "we beat STAN-on-our-representation" (the `stl_hgi` variant). It is **not a paper baseline**, and at AL it scores
  above our model (70.35 vs 69.81). The faithful STAN (its own embeddings, from raw) is the baseline, and we are
  above it. The substrate-bound variant is at most a future-headroom signal, never a beat.
- "we beat the cascade (CSLSL)". It is a **dead tie at equal cost** (Δjoint ≤ 0.02). Frame it as a defense (a
  cheaper cascade would not have matched our lift either), never a win.

**The single-model property is the primary thesis:** one model, one forward pass, two predictions.

---

## 4 · Section-by-section plan with draft paragraphs

> Voice rules for the draft prose: American English; plain, motivation-first, first-person plural ("we");
> simple words by default, technical terms only where they earn it, glossed lightly on first use; effects stated
> directly but softly; avoid the em-dash, prefer commas, parentheses, semicolons, and short sentences. Numbers
> in ⟨angle brackets⟩ are placeholders. Citation keys in [square brackets] are from the existing CBIC/CoUrb
> bibliographies; "NEW" marks a reference we must add (see §8).

### Abstract (about 180 words): draft

> Location-based social networks record where people go and what they do, one check-in at a time, and if we can
> anticipate the next visit then mobile and urban services can act ahead of demand. Two questions are usually
> enough: the next category (the kind of place) and the next region (which part of the city). Usually they are
> handled by
> separate models, so we ask whether one model can learn both, and what it costs to share one representation. We
> first build a check-in-level representation that describes each visit in its own context, instead of giving every
> place one fixed vector. Across five U.S. states, this representation improves next-category prediction by a wide
> margin over a standard place embedding [huang2023hgi], and most of the gain comes from the per-visit
> context. We then train one model for next-category and next-region together. On every state we measure, it beats
> a dedicated category model (by about +4.7 to +7.7 macro-F1), and on region it beats the dedicated model where the
> region space is large and is statistically non-inferior, within a two-point margin (TOST), where the region space
> is small. One model wins
> both tasks, and the spatial win grows with scale. On a non-U.S. city (Istanbul) the result is consistent: it
> beats on category and stays within two points on region.

*Notes:* motivation first; no model name in the first sentence; no system number, because the system reading is
no longer a contribution; the closing line states the finding, not an application.

### §1 Introduction (about 1 page): draft + checklist

> **¶1 (hook).** Location-based social networks, such as Gowalla [Cho2011], let people share the places they
> visit. Each check-in is a small record of human movement, and together they let us study how people move
> through a city. A natural and useful question is what a person will do next, because a system that can
> anticipate it can prepare ahead of time instead of reacting. For a mobility-aware service this could mean
> staging the right content in the area the user is heading to, or planning capacity there ahead of demand, and it
> also helps recommendation and urban analysis.
>
> **¶2 (the two tasks).** Predicting the exact next place is hard, and often it is more than a service needs. Two
> coarser questions are easier and usually enough: the next *category* (the kind of place, for example food or
> shopping) and the next *region* (a census tract). They also pair naturally, one for intent and one
> for geography. The question we study is whether a single model should learn both at once.
>
> **¶3 (the tension, from our own line of work).** Sharing one representation across tasks is not free. Our
> earlier work [silva2025mtlnet] found that joint training can help one task while quietly hurting the other,
> because the shared part is pulled toward a compromise that suits neither perfectly [caruana1997multitask]. So
> the useful question is not whether multi-task learning is good in general, but where sharing helps, where it
> costs, and what to do about it.
>
> **¶4 (what we do).** We make two changes and we measure each one carefully. First, a check-in-level
> representation: instead of one fixed vector per place, each check-in gets its own vector that carries its context
> (time, nearby places, recent trail). This builds on hierarchical graph representations of places
> [huang2023hgi] and on the infomax idea behind them [velickovic2019deep]. Second, a single model that
> predicts the next category and the next region in one forward pass. We evaluate on two very different datasets,
> five U.S. states of different sizes and one international city, and we chose these on purpose, so we can see whether the
> findings hold across small and large, U.S. and non-U.S. settings.
>
> **¶5 (contributions).** Our contributions are:
> 1. a novel check-in-level representation that extends hierarchical graph infomax from the place to the
>    individual visit, so each check-in is described in its own context. It improves next-category prediction by a
>    large and consistent margin over a standard place embedding (about +28 to +40 macro-F1; measured under the
>    paper's windowing at all five states, Alabama, Arizona, Florida, California, and Texas), and a
>    controlled test shows the gain comes from the per-visit context, not from extra supervision;
> 2. a multi-task design for joint next-category and next-region prediction that shares a semantic context across
>    the two tasks while keeping a private path for the spatial one, and the finding that this design lets a single
>    model, in one forward pass, beat a dedicated category model at every state (by about +4.7 to +7.7 macro-F1) and
>    beat or match the dedicated region model (beating it where the region space is large, and matching it,
>    statistically non-inferior within a two-point margin (TOST), where it is small);
> 3. an empirical account, across five states of different sizes (with a check on one international city), of how
>    the joint gain scales with the region space: the category win holds at every state, and the region result
>    improves monotonically with scale, staying statistically non-inferior within a two-point margin (TOST) at the
>    small region counts and beating the dedicated model at the large (the largest state, California, shows the
>    largest region win). Provisional at n=5 (seed 0).
>
> **¶6 (roadmap).** The rest of the paper covers background and related work (§2), the problem and the two tasks
> (§3), our method (§4), the experimental setup (§5), results (§6), discussion and limitations (§7), and
> conclusions (§8).

*Internal notes (do not appear in the paper):* contributions stated positively (not "we do not propose a new
optimizer"); the tension named once and resolved later (the structural-bottleneck question); plain language
throughout.

*Novelty scope (be honest, this is load-bearing; the reviewer panel pushed hard here):* contextual per-visit
embeddings already exist (CTLE, CASTLE), so contribution 1's novelty is the **specific combination** (check-in
level + hierarchical graph + infomax), the intersection none of them occupies, not the idea of a per-visit
vector. The one true distinction for C1: per-visit emission **in the hierarchical-graph-infomax family**, which
sequence- and language-model-based contextual embeddings (CTLE) do not provide. **Contribution 2's novelty is
empirical, not architectural.** Shared-trunk-plus-task-towers is standard hard-parameter sharing
(`caruana1997multitask`). Do **NOT** write "prior LBSN multi-task work shares the whole representation": that is
**false** (HMT-GRN is hierarchical with granularity-specific recurrent nets, and uses region as a beam-search
filter to improve next-POI). The defensible C2 line: prior multi-granularity MTL keeps **next-POI** as the target
and treats category/region as **auxiliaries**; we drop next-POI and make category and region **co-equal end
targets in one forward pass**, and we find that a private spatial path plus an unweighted loss lets the single
model **match or beat** the dedicated region model (beating it at the large region counts). Frame C2 as that
finding, and concede the multi-granularity-MTL space (HMT-GRN, SGRec, MCMG, iMTL, MCARNN) in one sentence. The category gain is **architecture-driven** (the shared trunk is a stronger category
encoder at no extra deployment cost, one model and one forward pass), not "the region task teaches category";
word it that way everywhere. To substantiate "novel", the CTLE baseline (§5.4) and the HGI ablation must both be
in the results, scored leak-clean. Contribution 3 is provisional at n=5 (the board is complete; the largest
state, CA, beats on region); if it ever reads as overlapping with 2, fold it
into 2 and ship two contributions.

### §2 Background and Related Work (about 1.25 pages): skeleton + fragments

R3 asked for more background on the embeddings and short descriptions of the competitors. We give a short
pedagogical primer and a tightly scoped related-work, organized so the reader can place our novelty.

**§2.1 From place embeddings to per-visit embeddings (the primer R3 wanted).**
> A place embedding turns a point of interest into a vector so a model can reason about it. Deep Graph Infomax
> [velickovic2019deep] learns such vectors by contrasting real graph neighborhoods against shuffled ones.
> Hierarchical Graph Infomax [huang2023hgi] adds structure, place to region to city, so the vectors respect
> geography. Both give one vector per place, so two visits to the same place look identical. Contextual check-in
> embeddings do give a vector per visit: CTLE [NEW] is the closest example, learning per-visit vectors with a
> masked-language-model objective over a user's check-in sequence. Our representation, Check2HGI, also works at
> the check-in level, but it gets there a different way: it keeps the hierarchical place-to-region-to-city graph
> and trains it with an infomax objective, rather than a sequence model. That specific combination, per-visit
> context inside a hierarchical-graph-infomax representation, is what is new; we compare against CTLE directly to
> show the difference is the source of the gain.

One small figure: DGI to HGI to Check2HGI, showing the added check-in level.

**§2.2 Predicting the next category and the next region.** One short paragraph each. Next-place recommenders
(ST-RNN [NEW], DeepMove [NEW], Flashback [NEW], STAN [NEW], GETNext [NEW]) and why we predict the coarser
category and region instead. The category task in our own line [silva2025mtlnet]. Next-region as a hierarchical
aid in HMT-GRN [Lim2022]. The field increasingly models several granularities together: place-level,
category-level, and region-level sequences, where category and region are predicted as auxiliary signals to help
the next-place task (SGRec [NEW], MCMG [NEW], HMT-GRN [Lim2022]). Position it plainly: prior work treats category
or region as helpers for next-place prediction, while we study the pair as the object itself. Be explicit, per
the literature (web-checked: the canonical terms are "next category" and "next region", and "region" is the
coarse spatial unit, not "area"), that next-region over census tracts has little precedent (HMT-GRN uses geohash
cells as an auxiliary), so we define our region unit clearly.

**§2.3 Multi-task learning for mobility, briefly.** MCARNN [Liao2018], iMTL [Zhang2020], HMT-GRN [Lim2022] in two
or three sentences. The closest multi-task alternative we compare against is the **cascade**: CSLSL [NEW] predicts
in a chain (when, then what, then where), with location as the headline and category as an instrumental step; the
CatDM line uses category similarly, to filter candidate places. We differ in two ways: we predict category and
region **in parallel, as co-equal end targets** in one model and one forward pass, and we **drop next-place**
entirely, so neither task is instrumental to a third. Our one-line distinction from the balancer line: we ask
where sharing helps and where it costs and give a clear account, rather than proposing yet another balancer; keep
the optimizer list (PCGrad [yu2020pcgrad], GradNorm [chen2018gradnorm], Nash-MTL [nash]) to one sentence, and note
the finding that such methods rarely beat a tuned fixed weighting at two tasks [xin2022domtl].

### §3 Problem and tasks (about 0.4 page): draft

This was the "system model" section. We shorten it to a problem setting plus a light motivation, with no system
claim.

> We work with check-in sequences. For each user we order their check-ins in time and form short windows of nine
> visits, and from a window we predict two properties of the next visit: its category, one of seven
> classes (Community, Entertainment, Food, Nightlife, Outdoors, Shopping, Travel), and its region, the census
> tract it falls in (the mahalle for Istanbul). Category is a small, fixed label set; region is a large one, from
> about five hundred (Istanbul) to about eight thousand five hundred (California) classes depending on the dataset. The motivation is practical, and the
> two predictions line up with two kinds of preparation a mobility-aware service makes. The category says what
> kind of place comes next, which hints at what content or service the user will want; the region says where it
> is, which says where to get ready. With both in hand a service can act ahead of demand, for example to pre-load
> content, prepare a connection, or plan capacity. So we study how well one model can anticipate both. We do not
> build or evaluate such a service here; the application is the reason the prediction matters, not a result we
> claim.

*Why this section exists:* it defines the two tasks precisely and gives the mobility-aware motivation, while
making clear that the network application is context, not a contribution. This is the honest version of the
earlier "system model" section.

⚠ **Granularity caution (the venue reviewer caught this):** a **census tract is not a radio cell**, and **Acc@10
(the right area in the top ten)** is far too coarse to drive cell or AP selection, handover preparation, or
"setting aside a connection." So pick motivation examples that **match the granularity we actually have**:
regional **demand and load anticipation**, **content staging** at the right edge area, and **capacity planning**
read naturally at tract level; **handover / cell-association / connection-setup do not** and should be dropped or
softened in §1 and here, or a reviewer will say the motivation is mismatched to the metric. Keep the motivation,
right-size the examples.

### §4 Method (about 1.5 pages): plan and the figures R3 asked for

**§4.1 The check-in-level representation (Check2HGI).** Plain mechanism first, one formal piece after.

- Prose: build a graph with four levels (check-in, place, region, city); edges encode category similarity, time,
  and distance; train it with the infomax objective (real versus shuffled neighborhoods); read out one vector per
  check-in.
- ⭐ **Figure 1 (top priority): the input-graph data flow.** Raw check-in trail, then graph construction,
  then per-visit representation, then windows, then two task outputs. The graph and the model are hard to picture
  from text alone, so this is the single most important figure.
- One honest sentence on novelty: the parts are standard; the contribution is the per-visit combination and what
  it buys, not a new mechanism.

**§4.2 The single model.** Plain description of one model with a shared part and two outputs, where the region
output keeps its own path. **State this once, explicitly, so the private path is not misread as a second model:**
the private spatial path is a task-specific branch inside the single model, sharing the trunk and one forward
pass; it is not a separate model, a second representation, or a routed path. State the deployment property
clearly: one model, one forward pass, two predictions.
Give the loss in one line (a fixed-weight sum, both outputs on plain unweighted cross-entropy), and explain why:
class-weighting one output while the metric is unweighted is an objective-metric mismatch we avoid, so both
outputs use plain unweighted cross-entropy. Report the parameter and compute cost (about +5 percent, one model
rather than two), so the cost of joint training is on the table. One architecture figure (reuse and clean the
existing one).

### §5 Experimental Setup (about 1.1 pages): plan

**§5.1 Data.** The statistics table (Table 1, below). Five Gowalla states [Cho2011, SNAP2014] and Istanbul from
Massive-STEPS [NEW]. State the small-versus-large design on purpose. Seven categories named.

**§5.2 Windows, splitting, and the integrity of the representation.** This subsection folds in the old §4.3
windowing note and states the no-leakage argument as part of the protocol. Write it as standard methodology, not
as a reply to anyone; it also closes the leak concern from the previous review, but the prose should never say so.

> *Windows.* For each user we sort check-ins in time and form overlapping windows of nine visits plus the next
> visit as the target. Overlapping windows give the category task more examples to learn from. We then apply a
> simple gate that removes a small over-representation of each user's last place (at stride one, many windows end
> on the same final visit); the gate de-skews that target, and it keeps one full-context prediction of the last
> place. These are valid examples, not padding.
>
> *Splitting.* We split by user with stratified five-fold cross-validation, so all of a user's windows fall in
> the same fold. Overlapping windows therefore cannot leak across the split, because a test user's visits never
> appear in training.
>
> *Integrity of the representation.* We train the representation once on the whole dataset and then feed it to
> both the single-task and the joint models, so it is worth being clear that it carries no information about the
> test visits. It does not, for three reasons. First, the representation is trained without any task label. Its
> objective only contrasts real graph neighborhoods against
> shuffled ones, and it never sees the next-category or next-region target, so no label can travel from test to
> train through it. Second, the only thing it could carry from the test side is graph structure (which places sit
> near which, in space, time, and category), because the graph is built over all places. We measured that
> exposure directly: we rebuilt the representation using only each fold's training users and re-ran both models,
> and both tasks moved by under a third of a point (region: AL ⟨−0.33⟩, FL ⟨−0.12⟩; category: AL ⟨+0.29⟩, FL
> ⟨+0.00⟩ on the places that carry its gain). Third, the one component that really passes visit-to-visit
> information, the region-transition prior used by the spatial model, is built per fold from training data only,
> seed by seed; an earlier version that used the whole dataset inflated region accuracy by 13 to 27 points, so we
> removed it. Every learned baseline representation is pre-trained the same way, so all inputs are held to the
> same standard. The one residual we cannot fully isolate, visits to places never seen in training, is the scope
> of a planned training-only variant; the evidence in hand bounds any inflation to under a point on the
> measurable majority, and finds none.

⚠ **HARD BLOCKER (cannot ship bracketed): source the four leak-Δ numbers** (AL −0.33 / FL −0.12 region; AL +0.29 /
FL +0.00 category; plus the in-coverage share) from the **train-users-only rebuild audit** (rebuild Check2HGI per
fold excluding val users, re-run both heads, report the delta; see `docs/research/evaluation_protocol_review.md §4.1`).
This is the load-bearing answer to the topological-leak rejection; §5.2 cannot ship with placeholders. The "13 to 27
points" transition-prior inflation IS sourced (`AGENT_CONTEXT.md`: full-data `region_transition_log.pt` leaks ~13-27
pp); the §3 "old region numbers (7 to 17)" note is a different quantity (the old MTL region cost), not the leak.
Mirror the one-sentence residual caveat into §7.

**§5.3 Metrics, and superiority versus non-inferiority.** Category: macro-F1 over the seven classes, so rare
categories count. Region: Acc@10 (the true region is among the top ten). Reference point for category is the
majority-class floor; for region, the dedicated single-task ceiling is the comparison anchor (we dropped the
Markov-1 floor row, §7).

⚠ **Statistical protocol (this is setup, not a result):** we test **superiority** with a paired Wilcoxon
signed-rank test and **non-inferiority** with a TOST at a two-point margin. Which cells are a beat (superiority)
versus a match (non-inferiority) is a result, reported in §6.2, not here. Justify the two-point non-inferiority
margin a-priori, the right way (the panel will reject a margin chosen to fit the result): (a) state a
**deployment-grounded** rationale for "two points of Acc@10 is negligible" *before* any appeal to whether the
claim survives, and never write "the margin under which the claim holds"; (b) report the **relative** effect and
a random-top-10 floor, so "negligible over 1k to 8.5k region classes" is shown, not asserted; (c) report a
**TOST-power** statement (the power to reject a true two-point gap at the board's variance), since an underpowered
non-inferiority test is not evidence of equivalence; (d) re-justify the margin for this multi-task-versus-single-
task axis rather than reusing the substrate-axis margin.

⚠ **The n=5 reporting reality (Decision, 2026-06-26):** all current numbers are seed 0 over five folds (n=5), so
each per-state one-sided Wilcoxon sits at the **exact n=5 floor, p=0.031, with all five folds in the winning
direction**; a family-wise Holm correction across the six category cells (or the three region-beat cells) does
**not** reach 0.05 at n=5 (the floor times the family size). We therefore (i) report the per-state effect with its
"5/5 folds positive, p=0.031" line, (ii) carry the **pooled** across-state evidence as the inferential backbone
(category: 30 of 30 fold-pairs favor the joint model, p≈9e-10; region beats, FL/CA/TX: 15 of 15, p≈3e-5), and
(iii) state plainly that we do **not** claim per-cell Holm-corrected significance at n=5 (the family does not clear
0.05 at this sample size), and leave a multi-seed confirmation (seeds {1,7,100} → n=20) to future work. Do not
promise an "in-progress" top-up in the submission. (Reproduce: `scripts/closing_data/superiority_wilcoxon.py`.)

**§5.4 Baselines (locked; see §7 and `BASELINE_HANDOFF.md`).** Three roles, kept separate in the writing: (1)
**task SOTA**, faithful author implementations on our data: next-category **POI-RGNN + Markov-9-cat**;
next-region **STAN (faithful, from raw) + ReHDM + HMT-GRN**. ⚠ **Region-baseline footing + STAN audit (2026-06-26,
see `STAN_REFOOTING_HANDOFF.md`):** keep all region externals on ONE footing for the matched columns.
**STAN** is run **faithfully (its own embeddings + STAN-native prefix-expansion sequences, from raw)** on our data,
seed-0 user-disjoint folds, and region targets (the literature norm; feeding STAN a pretrained embedding, the old
`stl_hgi` variant, is non-standard, so it is at most a labeled ablation). The converged, audited run (PR #53)
**clears the Markov floor and lands below our joint model at every state**: AL 60.72, AZ 49.86, Istanbul 61.86 (FL
in-flight, CA/TX optional). The earlier AL 34.46 was an under-trained artifact (a slow-backward bug + a STAN-derived
head, now fixed and audited) and is superseded. **HMT-GRN** is board-matched (seed 0, stride-1) at **all 6 states** (TX closed PR #38: reg 53.85 /
cat 25.81). **ReHDM-faithful** is reported as a **published-method reference under its own protocol** (chronological
80/10/10 + 5 seeds), never a paired/matched cell. STAN on our Check2HGI is dropped. **Comparability hierarchy:**
HMT-GRN (region-native, board-matched, multi-task — **HMT-GRN-*style***: own end-to-end LSTM trunk + region-transition
prior from raw, graph module + hierarchical beam search dropped; *not* a strict reproduction) is the **primary**
region-native comparison; **STAN (faithful, from raw) and ReHDM (own protocol) are secondary references, each labeled.** STAN, though built for fine next-POI, is a competitive coarse-region baseline once trained properly (it clears the Markov floor), and the joint model beats it at every state. (2) **representation** (FL only): **CTLE**, the closest prior contextual
embedding, presented fairly (its end-to-end form alongside the frozen one), plus a **feature-concat control**
(HGI ⊕ raw per-visit features), so the category gain is attributed to the hierarchy, not to any contextualization
nor to feature injection. (3) **multi-task comparator**: the **CSLSL cascade** (the dominant published
alternative to parallel joint training) is the dedicated MTL-design comparator (parallel vs cascade). HMT-GRN is
*also* multi-task, but its comparison is already captured as the **primary region-native external** (role 1), so we
do not re-cast it here. The **dedicated single-task ceilings** are the comparison anchors for the joint model, and the HGI representation is the
per-visit ablation (Part 1).

### §6 Results (about 1.5 pages): two results, two reads

Each result gets a one-sentence "read this as" lead (fixes Reviewer 2's uninterpreted tables).

**§6.1 The representation makes the category task learnable (Part 1).** Table (Tbl 2): Check2HGI versus HGI
category macro-F1, scored under the overlap board so it is on the same windowing as Part 2 (all five Gowalla
states, +27.6 to +39.6 macro-F1), plus
the per-visit-context share. ⭐ **Category embedding-quality figure (Fig 3):** show one or two **category**-
separability metrics where Check2HGI clearly wins (silhouette about 0.56 versus HGI 0.00; kNN-by-category about
0.98 versus 0.78), not five small plots. **Do not plot the region geometry:** on region both embeddings sit at
the floor and HGI scores spuriously higher, which would mislead a reader (the region signal comes from the
sequence model and the transition prior, not the static embedding's geometry). One sentence turns that absence
into evidence: the embedding cleanly separates category but not region, which is exactly why the representation
helps category and is neutral on region. Read: per-visit context, not extra training, is what makes the category
task learnable.

**§6.2 One model, two tasks (Part 2).** The central table: joint model versus the dedicated ceilings, per state,
both tasks, marking **superiority** where the joint model beats and **non-inferiority** (within two points) where
it matches. ⭐ **Figure 4: the signed category and region deltas across states, ordered by region count**, showing
the category delta positive everywhere and the region delta positive at the large states and within the margin at
the small (no cardinality cost: the largest state, CA, has the largest region win). **Significance, stated
honestly (see §5.3):** every fold favors the joint model (each state 5/5 folds, one-sided p=0.031, the n=5 floor);
the pooled evidence is decisive (category 30/30 fold-pairs, p≈9e-10; region beats 15/15, p≈3e-5); we do not claim
per-cell Holm-corrected significance at n=5, and leave a multi-seed confirmation to future work. The encoder-isolation probe (the
region stream frozen at start) keeps the **full** category lift at Alabama, Arizona, and Florida (within 0.3 of
the joint model, far above the dedicated ceiling), so we report the category win as **a stronger shared encoder,
not region-to-category transfer**, as a finding, not a hypothesis. **The region gain is monotone with scale:** the
region delta rises strictly with region count (Istanbul −0.52, AL −0.18, AZ −0.06, FL +0.57, TX +2.06, CA +2.18),
so the joint model helps the spatial task more, not less, as the region space grows. **Against the externals
(prose, not only the table):** the joint region head is above the primary region-native baseline (HMT-GRN) at all
six states, above a faithful STAN (AL, AZ, Istanbul) and a ReHDM reference, and clears the Markov-1 floor by a wide
margin; on category it is far above POI-RGNN and the Markov-9-cat floor. (Never list the substrate-bound STAN, which
is above us at AL and is not a baseline.) Also report the cascade result honestly: our parallel model **ties** the
published cascade (CSLSL) at **equal cost** (Δjoint about 0), which rules out that a cheaper cascade would have
matched our lift, a defense, not a "we beat the cascade" claim. Read: one model beats the dedicated category model
at every state, beats the dedicated region model where the region space is large and is non-inferior (TOST) where
it is small, and beats the external baselines on both tasks; the region win grows with scale. Mark Gowalla cells
n=5 (seed 0) provisional, Istanbul n=20.

**§6.3 External validity (Istanbul).** One small row or figure: the finding replicates on a non-U.S. city
(category beats, about +6.7 macro-F1; region matches, about −0.5), reported as gap-to-ceiling or lift, never
absolute Acc@10, because the region counts differ. Note that Istanbul is measured on the earlier
graph-convolution (GCN) representation (the overlapping-window board representation was not built for it), so it
is a cross-representation external check; mark it provisional. The category headline (+6.7) is the four-seed mean
versus the single-seed dedicated ceiling; the clean seed-0 fold-paired gain is +6.5 (5/5 folds). Istanbul baselines
are in (PR #51/#53): category Markov-9-cat 24.55 and POI-RGNN 30.12 (both far below our 53.20/59.89); region
faithful STAN 61.86 and HMT-GRN 60.4 (both below our 74.28). ReHDM-Istanbul is deferred (footnote).

### §7 Discussion and Limitations (about 0.5 page): draft

> Our results show that one model can serve both tasks: the shared semantic context lifts the category task
> sharply, and a private spatial path inside the same model keeps the region task competitive, matching a
> dedicated model at the small region counts and beating it at the large. The design reading is that you do not
> need two models: one model, with a shared trunk and a private spatial path, does both, and the spatial gain
> grows with scale. For a mobility-aware service this is a useful hint: a single model can
> anticipate both what and where. We are honest about the limits. First, these numbers are from a single seed
> over five folds, so they are provisional until a multi-seed run, and at the smallest
> region counts the region result is statistically non-inferior within a two-point margin (a match, not a beat),
> while at the large counts it is a beat. Second, our representation is transductive, so it
> is trained seeing all places, and a small part of its behavior, visits to places never seen in training, is the
> one thing we cannot fully isolate; a planned training-only variant is meant to close that gap. Third, although
> the motivation is a mobility-aware service, we do not evaluate one; tying these predictions to a concrete
> service, with its own baselines and costs, is the natural next step.

*Usage illustration (storytelling, not a claim).* The discussion can include a short, concrete illustration of
how a mobility-aware service would use these predictions, for example reading the next-region top-k as a set of
areas to prepare in advance (an anticipatory prefetch set), with the next category as a hint at what to prepare.
Frame it as motivation and future work, with no measured number. Keep it out of the contributions and the
results.

### §8 Conclusion (about 0.3 page): draft

> We asked whether one model can predict both the category and the region of a user's next visit. A check-in-level
> representation makes the category task far more learnable, and a single model beats a dedicated category model at
> every state while, on region, beating the dedicated model where the region space is large and matching it where
> it is small. What we take away is not just a model but a clear reading: one model with a shared semantic context
> and a private spatial path can anticipate both what and where, and it does so without giving up the spatial task,
> which is where the gain grows with scale.

---

## 5 · Table 1: Dataset statistics (real numbers, ordered by region count)

> Source: `docs/studies/second_dataset/STATS_T1.md`. **Windows column is now the OVERLAP board** (gated stride-1,
> MIN_SEQ=10, emit_tail=False — the `check2hgi_dk_ovl` windowing), recomputed 2026-06-25 and **validated** against
> the one on-disk dk_ovl parquet (AL recompute = 96,326 = on-disk exactly; FL = 1,274,418 = the handoff value).
> Max/avg sequence length and sparsity derived from the per-user check-in parquet. Region counts cross-checked
> against `RESULTS_TABLE.md §0.1`. Gowalla rows are raw == substrate; **Istanbul row is the mahalle substrate**
> (post null-coord drop: 462,615 ck / 23,694 users / 29,816 POIs), not the 544,471 raw corpus — so its windows,
> avg-seq, and sparsity are on the data actually trained.

| Dataset  | Source        | Check-ins |  Users |    POIs |       Regions | Categories | Windows (9+1, overlap) | Max seq | Avg seq | Sparsity |
|----------|---------------|----------:|-------:|--------:|--------------:|-----------:|-----------------------:|--------:|--------:|---------:|
| AL       | Gowalla       |   113,846 |  3,858 |  11,848 |         1,109 |          7 |                 96,326 |   3,835 |    29.5 |   0.9975 |
| AZ       | Gowalla       |   236,450 |  7,869 |  20,666 |         1,547 |          7 |                200,895 |   5,589 |    30.1 |   0.9985 |
| FL       | Gowalla       | 1,407,034 | 21,052 |  76,544 |         4,703 |          7 |              1,274,418 |  16,679 |    66.8 |   0.9991 |
| TX       | Gowalla       | 4,089,892 | 38,644 | 160,938 |         6,553 |          7 |              3,830,414 |  42,300 |   105.8 |   0.9993 |
| CA       | Gowalla       | 3,171,380 | 37,090 | 169,145 |         8,501 |          7 |              2,925,466 |  14,855 |    85.5 |   0.9995 |
| Istanbul | Massive-STEPS |  462,615† | 23,694 |  29,816 | 520 (mahalle) |          7 |                270,217 |     817 |    19.5 |   0.9993 |

† Istanbul = mahalle substrate (post null-coord drop); raw corpus is 544,471 check-ins / 23,700 users.
Sparsity = 1 − check-ins / (users × POIs); avg/max seq = per-user check-in count (verified on the parquet, not
divided). NYC (272,368 check-ins; 6,929 users; 1,912 tracts) is available in the same pipeline if a second
external city is wanted; we lead with Istanbul.

---

## 6 · Figures and tables plan (one finding per figure; legible in two columns)

| #     | Type    | Content                                                                                                      | Answers                                            |
|-------|---------|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------|
| Fig 1 | diagram | Input-graph data flow: check-in trail, four-level graph, per-visit representation, windows, two outputs      | R3 (could not picture graph or model)              |
| Fig 2 | diagram | The single model: two input streams, private encoders, a shared bidirectional cross-attention stack, then a GRU category head and a dual-tower region head with a private spatial path; one forward, two outputs | R3 (structural bottleneck) |
| Fig 3 | panel   | **Category** embedding quality only (silhouette ~0.56 vs HGI 0.00; kNN-by-category ~0.98 vs 0.78), on the design_k board substrate (not v11). Do NOT plot region geometry (both at the floor, HGI spuriously higher = misleading). Part-1 "why"; one sentence: separates category, not region (the asymmetry) | Part 1 mechanism; Vitor's caution |
| Fig 4 | plot    | Signed category and region deltas across states, ordered by region count, with the two-point margin. Shows category positive everywhere and region positive at the large states / within-margin at the small (the largest state, CA, has the largest region win). All cells n=5 (seed 0) provisional | R2 (interpretation); the core finding (one model wins both) |
| Tbl 1 | stats   | Dataset statistics (above)                                                                                   | Vitor's request; R3                                |
| Tbl 2 | result  | Representation: Check2HGI versus HGI category macro-F1 + per-visit share, **on the overlap board** (one windowing for the whole paper); margin +27.6 to +39.6 (AL +29.31 / AZ +27.63 / FL +39.62 / CA +37.95 / TX +37.47), all 5 states scored | Part 1 |
| Tbl 3 | result  | Joint versus dedicated ceilings, both tasks, per state, marking superiority (beats) or non-inferiority (matches) per cell | Part 2 (provisional, n=5)                          |

> Keep one headline figure (Fig 4, the signed deltas) carrying the finding. Captions state the conclusion, not "see table".

---

## 7 · Baselines (LOCKED 2026-06-24, full execution plan in [`BASELINE_HANDOFF.md`](BASELINE_HANDOFF.md))

Three roles, kept separate in the writing (see the handoff §1):

**Role 1: task SOTA (faithful author implementations, our data):**
- [ ] next-category: **POI-RGNN** + **Markov-9-cat** floor.
- [ ] next-region: **HMT-GRN** (primary) + **STAN** (faithful, from raw) + **ReHDM**.
  ⚠ **Footing + STAN audit (2026-06-26, `STAN_REFOOTING_HANDOFF.md`):** each faithful baseline keeps its OWN native
  pipeline; the shared protocol is only data + seed-0 user-disjoint folds + region target + Acc@k (NOT our windowing).
  **STAN** is run **faithfully (own embeddings + STAN-native prefix-expansion, from raw)**; the converged, audited run
  (PR #53) **clears the Markov floor and is below our joint model**: AL 60.72 / AZ 49.86 / Istanbul 61.86 (FL
  in-flight, CA/TX optional). The earlier AL 34.46 was an under-trained artifact (slow-backward bug + STAN-derived
  head), now fixed + audited and superseded. Feeding STAN a pretrained embedding (`stl_hgi`) is non-standard, so it is
  at most a labeled ablation. **HMT-GRN** (HMT-GRN-style, board-matched at all 6 states incl. TX 53.85) is the
  **primary** region external. **ReHDM-faithful** (AL/AZ/FL; CA/TX footnoted infeasible) is a **published-method reference under its own
  protocol** (chronological split, 5 seeds), NOT a paired cell. STAN-on-our-Check2HGI (the Istanbul stop-gap) is
  dropped. **Markov-1 region DROPPED (2026-06-24)**; keep at most a one-clause text mention.
- [ ] our STL ceilings + MTL champion (the comparison anchors).

**Role 2: representation novelty (FL only, small validation block):**
- [ ] **CTLE** (the closest prior contextual embedding): fix the frozen run + add **CTLE-E2E**, present fairly vs
  Check2HGI-SC under matched head.
- [ ] **feature-concat control** (HGI ⊕ raw per-visit features → same head): closes "is it just feature injection?".

**Role 3: multi-task comparator:**
- [ ] **CSLSL cascade** (`b4_cascade.py`) is the dedicated MTL-design comparator (parallel vs cascade, the §1b tie).
  HMT-GRN (multi-task) is already the **primary region-native external** (role 1), not re-listed here.

**Dropped:** CTLE as a SOTA row / full SC ladder; MHA+PE; STAN-faithful as a headline; SC-region (quarantined).
Tables stay slim (one category table, one region table, one small FL validation block).

---

## 8 · References and reusable assets from the prior (unpublished) draft

> The earlier BRACIS submission was **rejected, so it is unpublished**. We own its material and can reuse any of
> it freely (no self-citation, no self-plagiarism). Reuse the **substance and assets**, never the prose (it is
> too technical and was part of why the paper did not land). All paths below are under
> `articles/[BRACIS]_Beyond_Cross_Task/`.

### Bibliography (the single biggest reuse)
Base the MobiWac bibliography on the **audited, error-corrected** bib at `src/references.bib`, and
**de-anonymize it** (MobiWac is single-blind, so author names stay).

Already verified there (reuse the key): DGI `velickovic2019dgi`; HGI `huang2023hgi`; STAN `luo2021stan`;
GETNext `yang2022getnext`; HMT-GRN `lim2022hmtgrn` (kept as a commented entry, restore it); ReHDM `li2025rehdm`;
POI-RGNN `capanema2023poirgnn`; HMRM `chen2020hmrm`; MHA+PE `zeng2019mhape`; Gowalla `cho2011gowalla`; MTL
`caruana1997multitask` plus the survey `vandenhende2022mtl`; MTL optimizers `senushkin2023aligned`,
`liu2023famo`; our published self-cites `silva2025mtlnet` (CBIC) and `paiva2026courb` (the CoUrb / ST-MTLNet
record we needed).

**Carry its three corrections (do not reintroduce the errors):** (1) CBIC's POI-RGNN reference was wrong
(`capanema2019identificacao` is a different paper) → use `capanema2023poirgnn`; (2) HMRM had author-name errors,
fixed in `chen2020hmrm`; (3) GAT was cited via arXiv → use the ICLR canonical.

**Still genuinely new (add; URLs in `docs/research/references.md`):** CTLE, Massive-STEPS, Kurin et al. 2022,
DeepMove, Flashback, ST-RNN, SGRec, MCMG; and port from CoUrb: Xin `xin2022domtl`, POI2Vec `feng2017poi2vec`,
skip-gram.

**Claims in the draft that need a citation:** the "shared part is pulled toward a compromise" sentence
(`caruana1997multitask`); the infomax primer (`velickovic2019dgi`, `huang2023hgi`); the "rarely beats a tuned
fixed weighting at two tasks" sentence (`xin2022domtl`, and Kurin NEW). Verify the Kurin/Xin direction before
citing.

### Figures, tables, and protocol (reuse with adaptation)
- **`src/figs/per-visit.png`: RECREATE, do not reuse the file.** Its layout is a good template for our Part 1
  figure (HGI vs POI-pooled vs canonical Check2HGI next-category macro-F1 across the five states, with the
  per-visit-context share annotated), but it plots the **old v11 Check2HGI** (the frozen GCN substrate), which is
  superseded by the current substrate and the overlapping-window board, so the numbers and the underlying
  embedding are both out of date. Build a new figure from current results; reuse only the layout, not the file.
- **`src/figs/arch.png`: REDRAW, do not reuse as-is.** It shows the *old fully-shared* architecture (cross
  attention, then one shared backbone, then two heads). Contribution 2 is the **private spatial path**, which
  this figure does not show, so reusing it would misrepresent the new model. Keep its clean style, redraw the
  private region path.
- **`src/tables/{datasets,substrate,mtl_vs_stl,external}.tex`: reuse the table scaffolding,** refill with current
  numbers (the BRACIS numbers are superseded by the overlap board).
- **`STATISTICAL_AUDIT.md`: reuse the protocol** (user-split cross-validation, per-fold train-only transition
  prior, paired Wilcoxon for superiority, TOST at a 2-point margin for non-inferiority). It is the same protocol
  we use; rewrite the description plainly.
- **`sections/related.tex`: reuse the substance** (which papers, how positioned), rewrite the prose.

**Do not reuse:** the draft prose (superseded "large trade-off" framing, and the writing itself); the Springer
LLNCS template (MobiWac is IEEE); the old result numbers as final.

### Validated prose fragments to carry (an agent mined the prose; this is the audited result)

> Most of the BRACIS prose is superseded. These fragments survive an audit against the current science. **Two
> audit corrections first** (the mining agent got these wrong):
> - `silva2025mtlnet` (CBIC) and `paiva2026courb` (CoUrb) are our **published predecessors, not the rejected
>   submission**, so we cite them freely (MobiWac is single-blind). The BRACIS draft's "do not self-cite /
>   anonymity" note was a double-blind artifact and does not apply here. The "two prior works frame the
>   bottleneck" intro framing is ours to use.
> - Everywhere the BRACIS prose says **"substrate"**, write **"representation"** (our glossary bans the repo
>   word). Flip every **"non-overlapping"** window to overlapping. Re-pull every number from the v14 board.

**Carry (still true, high value):**
- **Methodology, for §5 (the crown jewels):** user-disjoint `StratifiedGroupKFold` folds reused across all
  conditions so paired tests are well defined; the **matched-head-ceiling** justification ("the same head classes
  appear on both sides, so any multi-task vs single-task difference is the sharing, not head capacity"; this
  answers "is it just head capacity?"); the transductive-exposure disclosure; the per-fold train-only seed-tagged
  transition prior; and the n=5 single-seed ceiling (p=0.0312) vs n=20 multi-seed {0,1,7,100} reporting protocol.
- **Graph-construction source for §4.1** (verify against the current v14 representation before use): the
  Check-in to POI to Region to City hierarchy; visit nodes carrying category plus a sin/cos timestamp;
  temporal-decay edges between adjacent visits; the per-boundary contrastive loss (bilinear-discriminator binary
  cross-entropy against shuffled negatives); and the frozen two-product output (a per-check-in and a per-region
  64-dimensional vector).
- **Framing for §1 and §2:** the "granularity axis" pivot (prior work fixed one vector per place and changed
  *which* features were encoded; we change *how often* the representation emits a vector), and the café-at-8am
  vs-10pm example for why per-visit context matters.
- **Thesis one-liners for §6 and §7:** "the picture is dominated by the representation's asymmetric value across
  the two tasks, not by cross-task transfer"; and "any multi-task system here works on a representation that
  already encodes the easier task more clearly than the harder one."

**Do not carry (superseded or traps):** the "−7 to −17" trade-off magnitude; any "Pareto / ties / matches
everywhere" wording; all v11 numbers; the cross-attention backbone (its equation, the old architecture figure,
and the lambda=0 / key-value co-adaptation note), since our architecture is the private spatial path (a branch
inside one model); the
non-overlapping-window under-supervision limitation; the recipe-selection story; mobility-management as anything
but motivation; and the entire `PAPER_DRAFT.md` (process scratch, not prose). Run an em-dash pass on every
carried fragment.

---

## 9 · Reviewer-feedback traceability (internal checklist only)

> Internal only. This maps the feedback from the earlier rejected submission to concrete fixes, so we are sure we
> address it. It never appears in the paper, and the paper never refers to a prior submission or to its reviewers.

| Earlier concern                                            | Where fixed                                                                                                                                                          |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| R1: topological leak (the embedding sees test transitions) | §5.2 "why the representation is not a data leak": no labels in the representation; measured ≈0 transductive effect; per-fold train-only transition prior; user split |
| R1: loses to Markov                                        | moot under the new result: the joint model is above the primary region external HMT-GRN at all six states and a faithful STAN trained from raw (AL 60.72 / AZ 49.86 / Istanbul 61.86), exceeds a ReHDM reference run under its own protocol, and every region model clears a Markov-1 floor by a wide margin (one-clause text mention; Markov-1 region row dropped). Do NOT cite STAN-on-our-representation: it is not a baseline and at AL sits above us (70.35 vs 69.81). |
| R1: marginal gains while scaling cost                      | Part 2: category +4.7 to +7.7 and region beats at scale, in one model; §4.2 cost accounting (+5 percent, one model vs two). No longer marginal, no longer a cost.    |
| R2: hard to follow, one spine                              | §2 storytelling spine: one question, two parts, one figure each                                                                                                      |
| R2: novelty not positioned                                 | §2.2 / §2.3 positive one-line distinctions; §1 ¶5 positive contributions                                                                                             |
| R2: uninterpreted tables                                   | §6 "read this as" lead per table; Fig 3 as the single headline                                                                                                       |
| R3: thin DGI/HGI background                                | §2.1 primer plus Fig 1                                                                                                                                               |
| R3: unclear graph/model                                    | Fig 1 (input graph) and Fig 2 (model)                                                                                                                                |
| R3: rule out a structural bottleneck                       | §4.2 private spatial path; the region win grows with scale (no shared-backbone bottleneck); the encoder-isolation probe (DONE: region stream frozen keeps the full category lift at AL/AZ/FL) backs "category win = a stronger encoder, not transfer"            |
| R3: LaTeX glitches                                         | mechanical: IEEE template, proofread pass                                                                                                                            |

---

## 10 · Open dependencies and decisions

0. **BLOCKER 0 (before any further writing): reconfirm the deadline.** Today is 2026-06-26; the §0 note "~25 June
   2026" is unverified and may already be **past**. Reconfirm immediately on the MobiWac / EDAS site. If the regular
   deadline is missed, switch to `PAPER_PLAN_POSTER.md` or the next cycle before investing in the eight skeleton
   sections. This gates everything below.

1. **The board is COMPLETE (2026-06-24).** All 6 states are 5f-settled; **TX is closed** (fp32 single-device,
   +2.06 reg, device-mix resolved); Istanbul is stride-1 n=20; CSLSL is done (a tie); feature-concat and HMT-GRN
   are done. Part 2 numbers are **n=5 (seed 0) provisional** for the 5 Gowalla states ({1,7,100}→n=20 is
   post-deadline); Istanbul is n=20. **Freeze-readiness verdict: CLOSEABLE-WITH-CAVEATS on the regular track.**
   **The earlier submission blockers are now CLOSED** (2026-06-26): Tbl 1 overlap windows + TOST (PR #49); Tbl 2
   overlap re-score, all 5 Gowalla states (PR #50/#52); FL CTLE-E2E (PR #50; CTLE-SC stays 2/5, text-only per
   decision); W6 encoder-isolation probe (PR #48, trunk-not-transfer); region externals HMT-GRN (6 states, PR #38),
   faithful STAN AL/AZ/Istanbul (PR #53, converged + below ours), ReHDM (current numbers). **What remains is prose +
   a few small data items, not missing science:**
   - **Write the paper.** The `src/sections/0{1..8}.tex` are still skeletons; the draft prose lives in §4 here. This
     is now the biggest task: turn the plan into the IEEE paper.
   - **Source the §5.2 leak Δ numbers** (still ⟨placeholders⟩) from the transductive-exposure audit before they go in.
   - **Faithful STAN FL** finish (in-flight); CA/TX optional (HMT-GRN carries CA/TX; the AL/AZ/Istanbul pattern holds).
     **ReHDM** keep the current numbers (CA/TX/Istanbul footnoted), update post-deadline.
   - (Deadline is now Blocker 0 above.)
2. **One windowing for the whole paper (Tbl 2) — DONE.** Part 1 is re-scored under the overlap board at all 5 Gowalla
   states (AL/AZ/FL/CA/TX; margins +27.6 to +39.6), so Tbl 2 and Tbl 3 share one windowing. (Istanbul has no HGI
   build → footnote.) The embedding-geometry figure is windowing-robust.
3. **Regular paper versus poster.** The freeze-readiness audit (2026-06-25) supports **REGULAR**: three backed
   pillars (a clustering-validated representation win with a feature-concat control; one model beating the
   category ceiling at all six states and the region ceiling at the three large states; beating every task-SOTA
   baseline plus a cascade tie that closes the obvious reviewer attack). The open items are recomputes and prose,
   not missing science, so they do not argue down to poster. **Aim regular, conditional on closing the three
   blockers in item 1.** The poster cut (`PAPER_PLAN_POSTER.md`) stays as the deadline fallback only if those
   blockers cannot land. Reconfirm the deadline first.
4. **External validity (Istanbul).** Measured at 4 seeds, but on the v11 / GCN substrate (the overlap-board
   substrate was not built for it), so it is a cross-substrate external check (category beats, region matches);
   report as gap-to-ceiling / lift, not absolute. NYC is available as a second city.
5. **The dataset table.** Refill the window counts under overlapping windows, and compute max/avg sequence
   length and sparsity from the parquet.
6. **Citations.** Add the eight-plus new bib entries (§8). Verify the Kurin/Xin direction.
7. **Naming locked:** next-category and next-region (literature terms). Region unit stated explicitly (census
   tract; mahalle for Istanbul), since next-region over tracts has little precedent.

---

## 11 · Page budget (8 pages, IEEE two-column)

> ⚠ **The earlier budget summed to exactly 8.0 with NO float allocation.** IEEE two-column floats (Figs 1-4 +
> Tables 1-3) consume ~1.5-2 pages, so running text must be capped at ~6.4 pages. The revised budget below carries an
> explicit floats row, raises Results, and trims Background and Method.

| Section                   | Pages |
|---------------------------|-------|
| Abstract + §1             | ~1.0  |
| §2 Background and Related | ~1.0  |
| §3 Problem and tasks      | ~0.4  |
| §4 Method                 | ~1.3  |
| §5 Setup                  | ~1.0  |
| §6 Results                | ~2.0  |
| §7 Discussion             | ~0.4  |
| §8 Conclusion             | ~0.3  |
| References                | ~0.35 |
| **Figures + Tables (floats)** | **~1.6** |

> If still over budget: cut the optimizer sentence, then collapse the §7 usage illustration to one clause, then fold
> §3 into the tail of §1. Never cut the §5.2 leak explanation; it answers the rejection. If the squeeze persists,
> decide the 10-page (fee) variant up front rather than at typesetting.
