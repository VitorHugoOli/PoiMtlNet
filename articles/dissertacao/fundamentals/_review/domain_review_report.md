# Domain Review Report — Chapter 2 (Fundamentals)

**Panel:** persona 10 (MTL expert) + persona 11 (POI / mobility expert), run over the five
section drafts (2.1–2.5), the model-lineage table, and the chapter wrapper.
**Mode:** read-only, fresh eyes. Findings judge DOMAIN CORRECTNESS only (a separate fact gate
covers citation existence/attribute integrity). Every finding carries a verbatim quote + `file:line`
and, where a claim was checked against a source, the source of record.
**Date:** 2026-07-21.

## Overall verdicts

- **MTL content (persona 10): sound-with-corrections.** The hard/soft/structured-sharing
  spectrum is characterized correctly; MMoE, PLE, DSelect-k, and cross-stitch are each described
  as their authors describe them; the balancer family (uncertainty weighting, GradNorm, DWA,
  PCGrad, CAGrad, Nash-MTL, Aligned-MTL, FAMO) is characterized correctly; the multi-objective
  framing (sener2018mgda) is faithful to the source; the scalarization-skeptic position and the
  time-indexed MTLnet null result are stated with the right scope. Corrections needed: one wrong
  attribution (negative transfer), one missing canonical anchor (Kurin), and one architecture-
  lineage claim to verify (PLE → joint model).
- **POI / mobility content (persona 11): sound-with-corrections.** The three tasks are kept
  distinct and correctly defined; the next-place sequence-model lineage (ST-RNN, DeepMove,
  HST-LSTM, Flashback, STAN, GeoSAN, GETNext) is characterized correctly and correctly framed as
  next-PLACE work; the representation lineage (one-hot → word2vec → DeepWalk/node2vec →
  GCN/GAT/GraphSAGE → MINE/DIM → DGI → HGI → check-in level) is correct and in the right order;
  the "a place-level vector is static across visits" claim (CTLE) is sound; datasets and metrics
  are correctly described. **One BLOCKER** (an encoder-attribution error that contradicts the
  arc and the chapter's own lineage table) must be fixed before the advisor sees the chapter.

## Top 3 findings

1. **[BLOCKER, 2.2]** The prose says Chapters 3 *and* 4 use the spatial/temporal encoders to inject
   context. MTLnet (Ch.3) uses none of them; adding them is precisely CoUrb's (Ch.4) contribution.
   This inverts the dissertation's central arc and contradicts the chapter's own lineage table.
2. **[SHOULD-FIX, 2.3]** Negative transfer is attributed to `Zhang2020`, which resolves in every
   project bib to the iMTL next-POI *recommender* (Zhang et al., IJCAI 2020), not the
   negative-transfer literature.
3. **[SHOULD-FIX, 2.2]** FiLM is described as the mechanism that injects the encoders' side
   information; in MTLnet/ST-MTLNet FiLM conditions on *task identity* (γ_t/β_t per task) and the
   context enters by *concatenation*. This contradicts the project's own GLOSSARY definition of FiLM.

---

## Findings (ranked, with evidence)

### BLOCKER

**F1 — [POI persona, lens: problem-formulation / arc consistency] Section 2.2: spatial/temporal
encoders misattributed to Chapter 3 (MTLnet).**

> `2.2_representations_for_mobility.tex:74` — "The models in Chapters~3 and~4 use FiLM and these
> encoders to inject spatial and temporal context into a place-level representation."

"these encoders" refers to Time2Vec, SIREN, Space2Vec, and Sphere2Vec (lines 59–71). Source of
record (CoUrb methodology, `articles/CoUrb_2026/src_en/sections/metodology.tex:1`): MTLnet
"uses a single vector E_DGI ∈ R^64 as input for both tasks ... it does not incorporate continuous
geographic coordinates or temporal visitation patterns." CoUrb's own stated contribution
(`metodology.tex:3`) is "to decompose the unimodal DGI representation into three independent
64-dimensional components ... a continuous spatial encoder, a temporal encoder (Time2Vec), and a
hierarchical categorical encoder (HGI)." The CBIC method (`articles/CBIC___MTL/sections/method.tex:37,62`)
confirms MTLnet's input is only the 64-d DGI embedding plus task-specific MLP encoders + FiLM.
So Chapter 3 uses **none** of the named context encoders; only Chapter 4 does. Two aggravating
points: (a) this contradicts the chapter's own lineage table, which correctly says MTLnet is
"place embedding, FiLM conditioning, and hard parameter sharing" and that ST-MTLNet is the one
that "replaces the place-embedding input with decomposed spatial, temporal, and categorical
encoders" (`model_lineage_table.tex:18,20`); (b) Space2Vec (`mai2020...`) is named in the draft but
is used in **neither** chapter (it is commented out in CoUrb, `metodology.tex:66`). Attributing
MTLnet with the very context it lacked erases the arc's turning point (representation, not
architecture, is the bottleneck).
*Suggested direction (not applied):* attribute FiLM to both Ch.3 and Ch.4 (correct), but the
spatial/temporal encoders (Time2Vec + the SIREN-variant / Sphere2Vec-M) to Chapter 4 only; present
Space2Vec as background context that is named but not adopted, or drop it.

### SHOULD-FIX

**F2 — [MTL persona, lens 3: negative transfer] Section 2.3: negative transfer attributed to the
wrong paper.**

> `2.3_multi_task_learning.tex:30` — "joint training can leave a task worse off than its
> single-task model, a failure called negative transfer \cite{Zhang2020}."

`Zhang2020` resolves in every project bibliography to the iMTL paper: Zhang, Lu et al., "An
Interactive Multi-Task Learning Framework for Next POI Recommendation with Uncertain Check-ins,"
IJCAI 2020 (`CBIC___MTL/references.bib:422`; `[mobiwac]/src/references.bib:258`). That is a next-POI
*recommender*, not a study that defines or surveys negative transfer. In the single global
dissertation bibliography the sentence will render "a failure called negative transfer [iMTL
next-POI recommender]," which an MTL examiner reads as a miscredit. The draft's own ledger even
labels it "[iMTL author work]" (`2.3...tex:82`). The anchor the field expects here is the
negative-transfer survey (Zhang et al., arXiv:2009.00909) or Standley et al. (arXiv:1905.07553,
already cited in the next sentence as `standley2020tasks`).
*Suggested direction:* cite the negative-transfer survey (2009.00909) for the definition, or fold
the definition onto the already-cited `standley2020tasks`; reserve `Zhang2020`/iMTL for the mobility
MTL paragraph (lines 62–68) where it belongs.

**F3 — [MTL persona, lens 4/capacity + mechanism] Section 2.2: FiLM's mechanism inverted.**

> `2.2_representations_for_mobility.tex:71` — "Feature-wise linear modulation (FiLM) is the
> mechanism that lets such side information act on a shared network: it conditions a layer by
> scaling and shifting its features per input."

The general FiLM description (scale + shift per input) is correct for Perez et al. 2018. The
*role* assigned to it here is wrong for this dissertation's models. In MTLnet/ST-MTLNet, FiLM is
conditioned on a **task-identifier embedding**: "A unique, learnable task embedding e_t for each
task t is used to generate a scaling vector γ_t and a shifting vector β_t"
(`CBIC___MTL/sections/method.tex:68`; identical in `CoUrb .../metodology.tex:15`). The spatial/
temporal/categorical context enters by **concatenation** at the input
(`CoUrb .../metodology.tex:143,147`: "E_input = [E_HGI ‖ E_loc ‖ E_time] ∈ R^192 ... delegating to
MTLNet the task of learning how to combine these dimensions through its shared layers and FiLM
modulation"). The project GLOSSARY agrees: FiLM = "per-task learned γ/β scaling that conditions
shared layers on task identity" (`GLOSSARY.md §2`). The draft credits FiLM with injecting the
context, when concatenation does that and FiLM modulates by task.
*Suggested direction:* describe FiLM as per-task modulation of the shared pathway (conditioned on
task identity), and state that the spatial/temporal/categorical context is injected by
concatenation at the input, not by FiLM.

**F4 — [MTL persona, lens 1: scalarization-first skepticism] Section 2.3: the primary
scalarization-skeptic anchor is missing.**

> `2.3_multi_task_learning.tex:53` — "A recurring result tempers the family. Random loss weighting
> is a competitive baseline \cite{lin2022rlw}, and a controlled study finds that current MTL
> optimizers often fail to beat a well-tuned fixed-weight baseline \cite{xin2022domtl}."

The cautious position is stated and is genuinely supported by Xin (2209.11379) and RLW
(lin2022rlw). But the single most-expected anchor for "a tuned fixed-weight scalarization matches
specialized MTL optimizers" — Kurin et al., "In Defense of the Unitary Scalarization"
(arXiv:2201.04122) — is absent, as is the theory follow-up (Hu et al., 2308.13985) and Royer et al.
(2310.08910). For a fundamentals chapter that stakes the dissertation's own MTL position on this
claim, an MTL examiner will expect Kurin by name. This is a missing anchor, not padding: the claim
is already made; it lacks its canonical citation.
*Suggested direction:* add Kurin (2201.04122) as the primary anchor for the fixed-weight-baseline
claim (positioned beside xin2022domtl); optionally Hu/Royer.

**F5 — [POI persona, lens 4: floors and ceilings] Section 2.4: the 93% predictability ceiling is
a next-place, cell-level bound applied to the category/region tasks.**

> `2.4_datasets_and_evaluation.tex:45` — "At the other end lies the predictability ceiling: an
> entropy analysis of human movement reports a potential predictability of about 93\%, the upper
> level against which a model's accuracy should be judged \cite{song2010limits}."

Song et al. 2010 (Science) derive Π_max ≈ 93% for the maximum predictability of an individual's
**next location** (cell-tower resolution, hourly). Two mismatches when it is used here as the
ceiling for this dissertation's tasks: (a) it is a next-*place* bound at coarse spatial/temporal
resolution, not a bound for next-category (7 classes) or next-region (census tract / mahalle) —
those tasks have different predictability ceilings (region is coarser, category is a different
label space); (b) it is an *accuracy/predictability* bound, but the primary category metric here is
macro-F1, not accuracy (`2.4...tex:25`), so "the upper level against which a model's accuracy should
be judged" does not map onto the headline category metric at all. In §2.1 the same number is fine
(it is introduced generically as predictability of "where an individual goes next," `2.1...tex:16`);
the problem is the §2.4 repurposing as the operative ceiling for the two studied tasks.
*Suggested direction:* keep 93% as context for the next-place literature, but do not present it as
the ceiling for the category/region tasks; the dedicated single-task model (already named as the
"practical ceiling," `2.4...tex:48`) is the operative reference for these tasks.

**F6 — [MTL persona, lens: architecture lineage] Section 2.3: verify/soften the "joint model
descends from PLE" claim.**

> `2.3_multi_task_learning.tex:24` — "Progressive layered extraction sharpens this by separating
> shared experts from task-specific ones in layered gates, the structured-sharing topology the
> later joint model descends from \cite{tang2020ple}."

The PLE description itself is correct (Tang et al. 2020: shared vs task-specific experts, layered
gates). The lineage claim is architecturally imprecise. Verified against the Ch.5 source: the joint
model is "the shared trunk, a cross-attention stack of two blocks"
(`articles/[mobiwac]/src/sections/04_method.tex:30`), with a private spatial path — a
cross-attention design with no expert-gating. PLE is specifically a gated mixture-of-experts. The
joint model therefore does not "descend from" PLE in any architectural sense; the shared connection
is conceptual (shared plus task-specific components), not a PLE/MoE inheritance. As written, an MTL
examiner reads the joint model as a PLE variant, which it is not.
*Suggested direction:* reword to the structured-sharing *principle* (shared plus task-specific
components) that the joint model adopts, rather than "the structured-sharing topology the later joint
model descends from"; PLE stays as one instance of learned/structured sharing, not as the joint
model's ancestor.

### NOTE

**F7 — [POI persona, must-cite canon] Section 2.1: FPMC absent from the next-place lineage.**

The next-place lineage (`2.1...tex:40–55`) runs ST-RNN → DeepMove → HST-LSTM → Flashback → STAN →
GeoSAN → GETNext. It omits FPMC (Rendle et al. 2010, factorizing personalized Markov chains), the
field's classical foundational next-basket/next-POI model, while including less-canonical entries
(HST-LSTM, GeoSAN). For a representative lineage in a thin chapter this is defensible, but an
examiner may note the absence of the classical starting point. Not a correctness error.
*Suggested direction:* consider one clause placing FPMC at the head of the sequential line, or note
the lineage begins at the deep-learning era by design.

**F8 — [POI persona, lens 8: metric conventions] Section 2.4: OOD/unseen-region handling not
foreshadowed.**

§2.4 defines plain Acc@10 as the region metric (`2.4...tex:30`). Per the GLOSSARY the joint model's
headline region metric is OOD-discounted Acc@10 ("regions unseen in training count as misses"),
defined in Ch.5. Deferring the full definition to Ch.5 is consistent with the GLOSSARY, but a reader
of Ch.2 alone will take plain Acc@10 as the region metric. This is a NOTE, not a defect.
*Suggested direction:* one clause foreshadowing that unseen regions are counted as misses in the
reported region metric (full definition in Ch.5).

**F9 — [POI persona, critique canon] Section 2.5: the evaluation-rigor claim lacks its anchors.**

> `2.5_relevance.tex:41` — "and it has strong evaluation practice that mobility studies do not
> always apply."

The chapter *applies* the field's evaluation-critique lessons (user-disjoint splits, trivial floors,
verb–test binding, no cross-cardinality comparison), which is the substance that matters. But the
claim that mobility studies do not always apply strong evaluation practice is asserted without the
canon that established it (Dacrema et al. 1907.06902; Sánchez & Bellogín, doi:10.1145/3510409; POI
Pitfalls 2507.13725). Optional for a thin synthesis section (§2.5 by design adds no new citations),
but the claim currently stands unanchored.
*Suggested direction:* if any anchor is added, place it at the first mention of evaluation rigor in
§2.4 (not §2.5, which is citation-free by design).

---

## What holds / what reads well (do not touch)

**MTL (persona 10) — credibility signals present:**
- Hard vs soft parameter sharing defined correctly (`2.3...tex:16–19`); cross-stitch, MMoE, PLE,
  DSelect-k each described as their authors describe them (`2.3...tex:20–28`).
- The balancer family is accurate per method: uncertainty weighting (homoscedastic), GradNorm
  (gradient-magnitude), DWA (loss-rate), PCGrad (projection off conflict), CAGrad (conflict-averse,
  convergence), Nash-MTL (bargaining game), Aligned-MTL (principal-component alignment / condition
  number), FAMO (O(1) loss-decrease) — `2.3...tex:40–53`.
- Multi-objective framing (sener2018mgda) is a faithful paraphrase of the source's own motivation
  ("weighted sum valid only when tasks do not conflict"), `2.3...tex:32–35`.
- The scalarization-skeptic position is stated with the right scope and not overclaimed; the MTLnet
  null result is time-indexed ("a result that holds for that configuration"), `2.3...tex:69–72`.
- No result verb is upgraded; no balancer benefit is amplified (honors the Nash-MTL containment).

**POI / mobility (persona 11) — credibility signals present:**
- The three tasks are kept distinct and correctly defined, and "we do not predict the exact next
  place" is stated once, early (`2.1...tex:24–38`); the cardinality contrast is given
  (`2.1...tex:33–36`).
- The next-place lineage is correctly framed as next-PLACE work ("every model named in this
  paragraph predicts the exact next place," `2.1...tex:54`); ST-RNN, DeepMove, HST-LSTM, Flashback,
  STAN, GeoSAN, GETNext each described accurately.
- The representation lineage is in the correct order and each step is described correctly: one-hot →
  skip-gram → DeepWalk → node2vec → GCN → GAT → GraphSAGE → MINE → DIM → DGI → HGI → check-in level
  (`2.2...tex:14–44,77–82`); GraphSAGE correctly flagged inductive.
- The "a place embedding assigns each place a single fixed vector ... static across visits" claim is
  sound and correctly anchored to CTLE (`2.2...tex:46–55`).
- CSLSL's causal chain "time then activity then location" (`2.3...tex:64–66`) matches the published
  "when → what → where" logic (verified against Huang et al., EPJ Data Science 2024).
- Datasets (Gowalla five U.S. states, Foursquare NYC/Tokyo, Massive-STEPS/Istanbul) and metrics
  (macro-F1 as mean per-class F1 with the majority-class floor; Acc@10; paired Wilcoxon bound to
  "outperforms"; TOST bound to "matches") are correctly described (`2.4...tex:14–68`).
- The lineage table matches the GLOSSARY exactly.

## Unstated defenses (facts the repo holds, the text does not carry — for the author, not defects here)

- The scalarization-skeptic position would be stronger with the near-zero gradient-cosine
  measurement (scoped) that the repo holds; §2.3 states the position from literature only.
- §2.4 names user-disjoint CV but does not carry the "overlap cannot leak because a test user's
  visits never appear in training" hygiene sentence, nor the transductivity-audit result — both may
  be intended for Ch.5, but if the frame claims the protocol is leakage-guarded, one hygiene clause
  belongs here.

## Out-of-scope handoffs (not domain findings; routed to the fact/style gates)

- The `Zhang2020` key collision (F2) and the encoder-vs-chapter attribution (F1) also touch the
  citation fact gate (persona 05, R3 claim-support) — flagged here for the domain impact.
- No prose/style/grammar judgments made (Common protocol §7).
