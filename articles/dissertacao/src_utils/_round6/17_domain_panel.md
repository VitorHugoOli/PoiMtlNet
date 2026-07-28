# 17 · Domain panel — personas 09 (stats & leakage), 10 (MTL), 11 (POI/mobility)

**Written 2026-07-28** against commit `4e84cf7a` (the per-section chapter split), on a build I
produced myself: `make defense` → `build/main.pdf` **108 pp**, `make final` → `build/main_final.pdf`
**105 pp**, `tex_errors=0` on both logs, `Float too large` = 0. Every coordinate below was
re-resolved by phrase against the post-split files and carries today's line number.

Read first, in order: `reviewers/README.md`, `reviewers/{09,10,11}_*.md`, `AGENT_GUARDRAILS.md`
§1–§3 + §7, `WRITING_LAW.md`, `GLOSSARY.md`, `NORTH_STAR.md` §1–§4, `_round6/ANCHORS.md`.

---

## Verdicts

| Persona | Scope | Verdict |
|---|---|---|
| **09** stats & leakage skeptic | the new Appendix B static-task scope section, protocol detail in Ch.3/Ch.4 | **exposed** — on one claim, `apx_b_static_scope.tex:34-38`. Everything else in my remit survives with corrections. |
| **10** MTL expert | Ch.2 fundamentals + Check2HGI equations, Ch.5 balancer screen and trunk attribution, the arc | **sound-with-corrections** — the balancer and attribution work is now among the best-scoped prose in the document; the arc's causal claim is not. |
| **11** POI/mobility expert | tasks, representations, datasets, regions, taxonomy | **sound-with-corrections** — dataset and region treatment would pass this literature; two representation claims would not. |

**The single weakest methodological sentence in the document** (persona 09's required nomination),
`src/chapters/apx_b_static_scope.tex:34-38`, rendered on **p. 99**:

> "The earlier chapter builds a node feature from the average of a place's neighbors' one-hot
> categories, excluding the place itself, so a place's own label never enters its own
> representation."

The premise is true and the conclusion does not follow. I measured it. Details in **D-01**.

---

## The three findings that matter

### D-01 · BLOCKER · The Appendix B exculpation of Chapter 3 is refuted by measurement

**Anchor phrase.** "so a place's own label never enters its own representation" —
`src/chapters/apx_b_static_scope.tex:34-38`, rendered **p. 99**. Provenance for the claim is the
comment block at the `\input` site, `src/chapters/apx_b_errata.tex:397-400`, which cites
`research/embeddings/dgi/preprocess.py:114-130` and concludes: "This is why the author was right to
separate the two chapters: the defect is CoUrb's, not CBIC's."

**What I measured.** Two steps, and the second is the one nobody took.

*Step 1 — the input feature is as described.* `dgi/preprocess.py:115` builds the one-hot matrix,
`:121-130` replaces each row with `embedding_array.iloc[neighbors].mean(axis=0)` over the Delaunay
neighbours, and the node itself is not in `neighbors` (isolated nodes get a zero vector, `:128`).
`dgi/dgi.py:56` passes exactly that matrix as `x`. So node *i*'s own one-hot is absent from *X_i*.
The appendix is right this far, and so is the Ch.3 footnote at `3_cbic/method.tex:22`.

*Step 2 — the embedding is not the input feature.* The task never sees *X*. It sees
`model.poi_encoder(data.x, data.edge_index)` (`dgi/dgi.py:79`), one `GATConv` layer
(`dgi/model/DGIModule.py:12`, `:53`). One hop of message passing sets

    h_i = combination over j in N(i) (and i, via the default self-loop) of X_j

and *i* is a neighbour of its own neighbours. So *X_j* for every *j* in *N(i)* contains *i*'s own
one-hot with weight 1/deg(j). **Excluding a node's label from its own input feature does not
exclude it from its own embedding; one hop puts it back.**

I quantified the channel on the real graphs (POI table via `groupby(placeid).last()` as
`preprocess.py:75` does, Delaunay with the same qhull options as `:145`, mean degree 6.0):

| | Alabama | Florida | Arizona |
|---|---:|---:|---:|
| analytic weight of node *i*'s own one-hot inside *h_i* | 0.166 | 0.166 | 0.166 |
| macro-F1, 1-hop input feature *X* alone | 0.2515 | 0.3082 | 0.2732 |
| macro-F1, one hop of aggregation (the encoder's receptive field) | **0.6407** | **0.6530** | **0.6411** |
| macro-F1, same hop with *i*'s own one-hot surgically removed and renormalized | **0.2500** | **0.3145** | — |
| majority-class floor / chance | 0.0695 / 0.1429 | 0.0690 / 0.1429 | 0.0724 / 0.1429 |

Probe: multinomial logistic regression, 5-fold stratified, `random_state=0`, uniform attention as a
stand-in for learned attention. Reproduction ledger: `_round6/_domain_ledger.json`.

**What I conclude.** Deleting *only* the own-label channel and keeping all neighbourhood homophily
takes macro-F1 from 0.641 to 0.250 at Alabama and 0.653 to 0.315 at Florida — back to the level of
the raw input feature. The own-label channel, not spatial homophily, carries essentially the whole
difference. The mechanism in Ch.3 is therefore weaker than Ch.4's exact `fclass` lookup but it is
**the same kind of thing**, not a different kind, and the appendix asserts the strong form of the
distinction ("never enters") in a section whose stated purpose is a public statement about a
co-authored published result. For calibration only, and not as a like-for-like comparison: the
published Ch.3 static-task per-class F1 at Florida runs 29.94 to 62.51
(`src/tables/cbic/category.tex`), which is the range my 1-hop probe reaches on the own-label
channel alone.

Two things I am **not** saying, because persona 09's hard limit is to separate them. This is not
classical label leakage: no held-out `category` value reaches DGI training (the objective is
contrastive, `DGIModule.py:81-87`; `data.y` is carried in the pickle and never read during
training). And the Ch.3 static task is not thereby a deterministic mapping — the identity is
partial, weight ~0.17 diluted through attention, not the exact function Ch.4 has. What is wrong is
the *text's* categorical claim and the "the defect is CoUrb's, not CBIC's" conclusion drawn from it.

**What would close it.** Replace the second qualification with what is measurable. The honest form
is roughly: Ch.4's place embedding is an exact lookup on a label-determining class, so its static
accuracy measures a deterministic mapping; Ch.3's node feature excludes a place's own label, but one
graph-convolution hop re-admits it at a diluted weight, so Ch.3's static task is partially, not
fully, insulated — the two chapters differ in degree. Then either quantify the degree or say it was
not quantified. **Do not ship the current sentence.** It is the one sentence in the document that
an examiner can refute with thirty lines of code, and it is in the section that exists precisely to
be scrupulous.

---

### D-02 · BLOCKER · The arc's quantified diagnosis rests on the task the appendix disqualifies

**Anchor phrase.** "raised category macro-F1 by 20.2 to 22.0 percentage points across the three
states tested" — `src/chapters/6_conclusion.tex:46`. Sibling: "Category performance rose sharply at
every state tested" — `src/chapters/1_introduction.tex:114`.

**What I measured.** Chapter 4 reports two tasks and the 20.2–22.0 figure belongs to exactly one of
them. `4_courb/intro.tex:27`: "**In categorical classification**, ST-MTLNet outperforms MTLnet in
all evaluated category-state combinations, with average gains per state of 20.2 to 22.0 percentage
points"; same figure, same scoping, at `4_courb/results.tex:92` and `4_courb/conclusion.tex:14`.
"Categorical classification" is the **static** task — `4_courb/intro.tex:15` defines it as
classifying a POI's category "from its features". The sequential task's result is separate and
weaker: 15 of 21 combinations plus one tie (`4_courb/results.tex:111`).

Then, worse, the two arms of that static comparison do not use the same representation family.
`4_courb/methodology.tex:100` — baseline MTLnet input is `E_DGI ∈ R^64` from DGI, the
neighbour-mean node feature of D-01. `4_courb/methodology.tex:104` — the proposed input is
`E_cat = [E_HGI ‖ E_loc ‖ E_time] ∈ R^192`, and `E_HGI` is the HGI output whose node features are
the POI2Vec **fclass** table (`hgi/hgi.py:207` Phase 3b–3d → `:243` Phase 4
`preprocess_hgi(poi_emb_path=...)` → `hgi/preprocess.py:331` `node_features =
self._load_poi_embeddings(...)`; the lookup itself at `hgi/poi2vec.py:487`). So on the static task
the proposed arm is handed the exact label-determining channel and the baseline arm is not.

I verified the determinism myself across all five reported Gowalla states: `spot` takes 284 (AL),
305 (AZ), 324 (FL), 365 (TX), 333 (CA) values, seven categories, **zero** values spanning more than
one category. This reproduces the appendix's own numbers exactly.

**What I conclude.** Appendix B's first qualification says "the sequential task is unaffected …
Every claim Chapter 4 makes about the sequential task … stands as published"
(`apx_b_static_scope.tex:29-33`). That is correct and I did not find a hole in it. But the frame
chapters do not quote the sequential number. They quote the static one, unlabelled, as *the
diagnosis of the whole arc* — and by the appendix's own reasoning that number substantially
measures the introduction of a deterministic input-to-label channel on one side of the comparison.
The appendix therefore does not protect the arc as the frame currently states it, and the two texts
are three pages apart in the same PDF.

`6_conclusion.tex:48-53` already discloses the 64-versus-192 width asymmetry. The
representation-family asymmetry is the larger of the two and is disclosed nowhere.

**What would close it.** Two text changes, no new experiments. (a) In `6_conclusion.tex:46` and
`1_introduction.tex:114`, say which of the two category tasks the number belongs to, and add the
pointer to Appendix B that Ch.4's preface already carries. (b) If the arc's diagnosis is to be
carried by a number at all, carry it with the **sequential** result, which the appendix leaves
standing; if the static number stays, it needs the appendix's caveat attached at the point of use,
not three pages later. An examiner who reads Appendix B and then re-reads Ch.6 will ask this
question, and right now the document does not answer it.

---

### D-03 · MAJOR · The Check2HGI equations are the canonical objective, not the objective that produced Chapter 5's results

**Anchor phrase.** "The training objective makes that extension concrete" —
`src/chapters/2_fundamentals.tex:241`; equations (2.1)–(2.3), rendered **p. 19**, verified in the
PDF text layer (not inferred from source): `L = 0.4 L_c2p + 0.3 L_p2r + 0.3 L_r2c`,
`D(e1,e2) = σ(e1ᵀ W e2)`, `L_* = −log D(e⁺,e⁺) − log(1 − D(e⁺,e⁻))`.

**What I measured.** *Transcription fidelity: clean.* All three match
`docs/context/check2hgi_overview.tex:215`, `:220`, `:227` symbol for symbol, and the weights match
the shipped defaults (`Check2HGIModule.py:51-53`; assembled at `:1193-1197`; per-boundary term at
`:1159`, `:1184`, `:1189`; discriminator `discriminate()` at `:1003-1018`, and the r2c boundary uses
`discriminate_global()` at `:1035-1036`, which contracts against a single summary vector rather
than a paired embedding). Persona 10 finds no fabrication here, and the drafting comment's own
faithfulness note about `e⁻` is accurate.

*Scope: not clean.* The chapter's sentence says the total objective **is** a fixed-weight sum of one
term per boundary. For the representation that produced Chapter 5's numbers, it is not. The reported
substrate is v14 = `check2hgi_design_k_resln_mae_l0_1`
(`docs/studies/closing_data/archive/provenance/SUBSTRATE_VERSION_MAP.md:19`, "the single reported
Check2HGI substrate"; the board trains on `dk_ovl`, which is v14 embeddings re-windowed, `:21-23`),
and v14 is built with `--reg-poi-mode delaunay_gcn --encoder resln --mae-poi-lambda 0.3
--anchor-lambda 0.1` (`research/embeddings/check2hgi/check2hgi.py:477-478`). Both auxiliaries are
0.0 by default and non-zero in the shipped build, which is exactly why Ch.5 names them:
`5_mobiwac/04_method.tex:22` reads "Two small label-free auxiliary terms are added (weights 0.3 and
0.1)". So the document now states two different objectives for one artifact, twenty pages apart,
and the Ch.2 one is presented as complete. The drafting comment at `2_fundamentals.tex:293-299`
saw this and left `[VERIFY: whether Ch.5's two auxiliary terms should be named here as well]`. They
should; the run configuration is in the file cited above.

Two consequences a domain reader will care about, which the flag did not reach:

1. **`--mae-poi-lambda 0.3` reconstructs each POI's mean category one-hot**
   (`check2hgi/preprocess.py:461-464`, `:485-491`). Its own docstring argues this adds no new label
   information because the check-in category one-hot is already an input (`:466-471`) — a fair
   argument, but it is an argument about a *category-derived* auxiliary objective, and Ch.5's
   "label-free" is doing work here. `5_mobiwac/05_setup.tex:33` correctly says the fourth ground
   measures what the category input feature can carry between visits, so the disclosure exists; it
   simply does not extend to the auxiliary that reconstructs a category distribution.
2. **`--anchor-lambda 0.1` anchors the reg-path POI table to POI2Vec**
   (`Check2HGIModule.py:167`, `:474`; loader `check2hgi/reg_poi_aug.py:27-35` reads
   `output/hgi/<state>/poi2vec_poi_embeddings_<State>.csv`) — that is the **fclass-level** table of
   D-01/D-02, entering the shipped check-in-level representation as an anchor target. Ch.5 calls it
   "an anchor to a place embedding pre-trained, label-free, on the same data"
   (`04_method.tex:22`), which is true and complete as far as it goes, but the appendix's third
   qualification (`apx_b_static_scope.tex:41-45`, "Chapter 5 does not inherit the problem") is
   stated without reference to it. Ch.5's targets are *sequential*, so I do not claim the
   determinism transfers; I claim the appendix asserts non-inheritance and the anchor is the one
   place a reader could contest it, and the text does not engage it.

**What would close it.** In `2_fundamentals.tex`, one clause: state that Equations (2.1)–(2.3) are
the three-boundary objective and that the configuration used in Chapter 5 adds the two auxiliary
terms that chapter names, with a forward pointer. In `apx_b_static_scope.tex:41-45`, either say the
anchor was considered and why it does not carry the identity, or drop the third qualification's
claim of non-inheritance to a narrower statement about the input being a visit.

*On placement, since the task asked:* keeping the equations in Ch.2 is defensible and I would keep
them there. They land where the reader has just met DGI's local-global objective (`:157-160`) and
HGI's hierarchical one (`:163-166`), so the fourth boundary is a term added to an objective already
in hand; and Ch.5 is under review, so an equation there is a two-file change to submitted text. The
symbols are all defined at first use (`e1`, `e2`, `W`, `σ`, `e⁺`, `e⁻`, the three subscripts), which
is more than the source document does. **A reader could not implement from them**, and should not
expect to: absent are the four encoders, the attention aggregation at the POI and region levels, the
temporal edge decay `w_ij = exp(−Δt/τ)`, `τ = 3600 s`, and the negative-sampling scheme — all in
`docs/context/check2hgi_overview.tex:130-160`. If the intent is implementability, that is an
appendix, not three equations; if the intent is to show what the fourth level costs, which is what
the sentence claims, the three equations do that.

---

## Persona 09 — remaining findings

### D-04 · MINOR · Istanbul is not a sixth clean case, and nothing says so
`apx_b_static_scope.tex:18` scopes the determinism measurement to "the five Gowalla state subsets",
which is correct and matches my re-derivation. For completeness I ran Istanbul too: at POI level
after the same `mode_or_first` dedup, `spot` takes 575 values and **11 of them span more than one
category** (at raw check-in level, 580 values, zero spanning). So the property is a Gowalla-taxonomy
property, not a dataset-independent one. Nothing in the document is wrong; the scope line is doing
real work and a future reader may not notice that it is. Worth one parenthetical if the section
survives.

### D-05 · What holds, and must not be edited away
Persona 09's job includes protecting existing defenses. These are present, correctly scoped, and I
tried to break each one:

- **The overlap-cannot-leak argument is stated as the reason, not asserted.**
  `5_mobiwac/05_setup.tex:28` — "so all of a user's windows fall in the same fold and overlap cannot
  leak: a test user's visits never appear in training." This is the sanctioned form.
- **The four-ground integrity paragraph** (`05_setup.tex:33`) is the strongest passage in the
  document. It bounds rather than closes, states the 67–87 percent in-coverage caveat and names the
  unseen-places residual as the part it cannot reach, discloses that the per-fold transition prior
  followed a 13-to-27-point inflation, and says the joint and dedicated models do not use that prior.
- **The tuning-leakage consequence is stated where the absolutes are read**
  (`07_discussion.tex:24`): epoch selection consults the evaluation fold, "so every absolute score
  reported here is optimistic", with the two verifiable mitigations and an explicit "It does not
  follow that the bias cancels exactly." The round-6 narrowing of "identically" is right, and the
  reasoning in the comment (a word that strengthens a mitigation understates a limitation) is the
  correct reading of WRITING_LAW §3.
- **The fixed-partition caveat** is now in the chapter that reports the intervals, not only in Ch.1.
- **Pre-registration** (`05_setup.tex:43`) names the plan, that it assigned tests per task and not
  per dataset, that next-region superiority was **outside** it so those four gains are secondary,
  and the Wilcoxon-to-paired-*t* departure with its reason (exact one-sided *p* floor 0.0625 at four
  seeds). The equivalence-power sentence is a design statement, not post-hoc power.
- **Ch.3/Ch.4 protocol honesty.** `3_cbic/results.tex:30` and `4_courb/results.tex:14` both now
  state the weaker split axis, the single seed, and the per-task-best checkpoint rule, and both point
  at Ch.5's stricter protocol. `2_fundamentals.tex:464-470` scopes the split axis **and** the
  significance tests to Ch.5 alone. This is the arc's honesty story and it is carried in text.

---

## Persona 10 — remaining findings

### D-06 · MINOR · The balancer screen: the round-6 scoping is right, and one residue survives
`5_mobiwac/02_related.tex:111-117` now reads "of nineteen loss and gradient balancers screened at
their default configurations at a single seed on two datasets, Alabama and Florida … none improved
on a tuned fixed task weighting across both tasks and both datasets", then names both exceptions
with what each gives up. I traced all of it to
`docs/results/mtl_improvement/T4_audit_and_verdict.md`: the defaults/seed-0/AL+FL scope at `:8-10`
and again at `:111-112`; nineteen arms; nash_mtl AL cat 54.25 (+0.68) and scale_norm AL cat 53.76
(+0.19) at `:191-196`, with scale_norm's FL region collapse to 35.47. **Nothing is overstated, and
this is the field's expected k=2 result** (Kurin, Xin), which the chapter cites. It engages the
skeptic literature rather than only pro-balancer work, which is what my prior asks for.

The residue the source record names and the prose does not: three of the nineteen — CAGrad, PCGrad,
Aligned-MTL — were **never validly tested** under this architecture. The private region tower sits
outside `shared_parameters()` (verified: `mtlnet_crossattn/model.py:554` yields only the
cross-attention blocks and the two final LayerNorms; `:578` puts `next_encoder`/`next_poi` in
`reg_specific_parameters`), so a method that operates on shared parameters leaves >80 percent of the
region pathway at unit weight and collapses to approximately equal weighting
(`T4_audit_and_verdict.md:23-30`). "At their default configurations" answers the
DWA/GradNorm/FairGrad objection; it does not answer this one, because a wiring result is invariant to
configuration. The agent comment at `02_related.tex:125-130` says exactly this and records that the
author's instruction governs. I agree with the comment: the minimal repair is to stop naming PCGrad
in the citing sentence, or to add "three of which reduce to equal weighting under this
architecture". As it stands, an MTL examiner who knows the gradient-surgery family will ask how
PCGrad was wired, and the answer is in the repo, not the document. Author's call; flagging that the
question is live.

**On the softening at `07_discussion.tex:12-18`, since the task asked whether it went far enough:
it is right, and I would not move it.** The sentence now states the result and says the locus is
unsettled. I read the disconfirming arm myself:
`docs/findings/archive/F50/F50_T1_5_CROSSATTN_ABSORPTION.md` — Florida, 5 folds × 50 epochs,
category macro-F1 68.36 ± 0.74 with cross-attention against 68.32 ± 0.67 without, Δ −0.04 ± 0.13,
paired Wilcoxon W+ = 5, p = 0.6250 (`:19-20`, `:37`), and the record's own one-paragraph summary
calls the null "misleading" and a "hidden compensation effect" (`:229`). `06_results.tex:204-213`
reports both directions: it declines to name the trunk **and** declines to present the ablation as
evidence against it, and it states the two limits (the ablation ran on a configuration whose region
head used a transition prior the reported models do not use; the record reads the null as
compensation). That is the correct handling of a disconfirming internal result — neither buried nor
inflated. Going further would overstate the ablation.

### D-07 · MAJOR · The arc's causal claim changes more than one variable at a time, and only one confound is named
The task asked persona 10 to say plainly whether "representation dominates architecture" is
supported or imposed. **It is partly imposed, and the document is one disclosure short of being able
to say so itself.**

What the chapters actually vary between CBIC and CoUrb: the input representation (DGI 64-d →
decomposed 192-d), the **width** (disclosed, `6_conclusion.tex:48-53`), the **representation
family** on the static task (DGI → HGI-with-fclass; **not disclosed anywhere** — D-02), and, across
the full arc to MobiWac, the task pair, the sharing topology, the split protocol, the seed count and
the statistical treatment. Of those, `6_conclusion.tex:212-223` names the task-pair confound
explicitly and well: "No single controlled ablation separates the representation-and-topology change
from the task-pair change in the final result", with Ch.4 named as the fixed-pair control for the
diagnosis. That paragraph is the reason my verdict is "sound-with-corrections" rather than
"at-risk" — the document already knows the arc is not a clean single-variable experiment.

Two things it does not carry. First, D-02: the fixed-pair control's headline number is itself
confounded by a representation-family change on the affected task, so the control is weaker than the
limitation paragraph implies. Second, the arc says architecture *changes* did not move the needle,
and the strongest internal evidence for the sharing topology's contribution is the disconfirming
F50 arm at Florida (D-06) plus F49's finding that the cross-attention lift at Alabama is
"purely architectural, not transfer" (`docs/findings/F49_LAMBDA0_DECOMPOSITION_RESULTS.md:9`) and
state-dependent (`:12`). Ch.5 handles this locally and honestly; the frame's "the representation is
the dominant factor" does not inherit that nuance. **What would close it:** in the Ch.6 limitation
list, extend the task-pair confound item to name the representation-family change on Ch.4's static
task, and soften the frame's causal verb from dominance to what the evidence licenses — that the
representation change produced the larger measured gain in the settings tested, with the confounds
named. No experiment is required.

### D-08 · What holds (MTL credibility signals present)
The capacity disclosure is at the point of the claim (`04_method.tex:34`: 4.2 M against 1.1 M at
Alabama, 5.2 against 2.0 at California) and is not dressed up — "What the single model provides is
operational rather than arithmetic". The joint-best convention is defined where the selector is
(`07_discussion.tex:24`, geometric-mean joint validation score, with the dedicated models at their
own task's best epoch, and the asymmetry named rather than smoothed). The CBIC null is time-indexed
in Ch.2 (`2_fundamentals.tex:436-440`, "a result that holds for that configuration"), in Ch.6
(`:39-41`), and in the chapter preface. The Nash-MTL claims are corrected rather than reproduced in
three places, each against arXiv:2202.01017 read this round, and each correction runs *against* the
chapter's interest — including the CAGrad description at `2_fundamentals.tex:409-412`, which I
re-checked against the maximin objective and is now right. The gradient-cosine mechanism claim
(`02_related.tex:156-164`) states its scope in text — development-time, earlier data preparation,
four Gowalla states of which three are reported, four seeds, per-dataset means within ±0.003 — and
I traced it: pooled +0.0008 over 16 runs, n = 3,797 epoch-fold points, per-state FL +0.0007 /
AL +0.0032 / AZ −0.0005 / GE −0.0004 (`T4_audit_and_verdict.md:47-49`). AL's +0.0032 sits 0.0022
from the stated +0.001, inside the stated band. It is never "not shown".

---

## Persona 11 — remaining findings

### D-09 · MINOR · The taxonomy's fine level is invisible in the frame, and it is now load-bearing
`2_fundamentals.tex:47-56` defines the three targets and the seven top-level categories, and
`:55-56` names static category classification as a fourth, non-sequential task — correctly, and with
the "we do not predict the exact next place" statement placed early (`:65-67`), which is the
glossary's law and the field's. What the frame never mentions is that every Gowalla place also
carries a **fine class** (the `spot` column; 284–365 values per state), that it is a strict
refinement of the seven labels, and that two of the three chapters' representations are built on it.
After Appendix B, that fact is load-bearing for how a reader should interpret Ch.3's and Ch.4's
category numbers, and it appears for the first time on p. 99. One sentence in §2.2, where the
place-embedding line is drawn, would make the appendix legible to a reader who meets it cold.

### D-10 · What holds (POI/mobility credibility signals present)
This is the part of the document I expected to attack and could not.

- **Region definitions** are named at first use and never blurred: census tract for the Gowalla
  states, *mahalle* for Istanbul, glossed as a municipal neighborhood (`03_problem.tex:12`,
  `05_setup.tex:20`, `tables/mobiwac/datasets.tex` caption, `GLOSSARY.md:22`, `:61`). The
  "a census tract is a neighborhood, not a radio cell" sentence (`03_problem.tex:16-20`) and the
  explicit exclusion of cell association and handover is the right scoping for a MobiWac audience
  and the kind of thing this literature rewards.
- **Label-space cardinality is tabled per dataset** (520 to 8,501 regions,
  `tables/mobiwac/datasets.tex`) and stated in prose (`03_problem.tex:12`), and no
  cross-cardinality Acc@K comparison is implied anywhere I could find.
- **Floors are present and protocol-matched**: majority-class for category with the metric
  justified (`05_setup.tex:38`, `2_fundamentals.tex:451-457`), and a first-order region-transition
  Markov floor "under the same sliding-window protocol and fold splits as our models"
  (`05_setup.tex:47`).
- **OOD handling is defined**, in the metric sentence rather than a footnote: a visit whose true
  region is absent from that fold's training data counts as an error (`05_setup.tex:38`), with the
  same convention foreshadowed in Ch.2.
- **Baseline provenance sentences are present for every baseline** and each asymmetry is disclosed
  at the point of comparison — POI-RGNN re-implemented from published architecture; HMT-GRN
  deliberately reduced to a region-native model with its graph components and hierarchical beam
  search dropped, and stated as "not a reproduction of the complete published system"; STAN
  re-implemented under one fixed configuration and adapted to region only, with the reason it was
  not adapted to category; ReHDM under its own published protocol (`05_setup.tex:47`). This is
  better than most published work in this line.
- **Dataset vintage and single-non-U.S.-city limits** are stated where generalization is claimed,
  and Istanbul is framed as an external check on the finding rather than on absolute numbers.
  Massive-STEPS is cited for the field's own over-reliance critique (`2_fundamentals.tex:446-449`).
- **Windows are fully disclosed** — stride 1, overlapping, nine visits, min ten visits per user,
  padded end-of-history duplicates dropped, and the horizon characterized (median 0.4 h at Florida
  to 5.5 h at Istanbul; 5 to 27 percent of targets over three days out, `05_setup.tex:25`).

One gap that is a genuine gap rather than a defect: **revisitation**. Persona 11's lens 4 asks what
fraction of the region and category numbers is repeat behaviour, and I could not find a
repeat-versus-explore statement anywhere in the document. The Markov floors bound it indirectly.
This is future work the text could name in one clause, not a correction.

---

## UNVERIFIED — blocked, and stated rather than smoothed

1. **The exact attention weights in the DGI hop (D-01).** My probe used uniform attention;
   `torch_geometric` is not installed in this sandbox and I did not train a GAT. Learned attention
   can concentrate or dilute the own-label channel relative to my 0.166 analytic weight. The
   direction of the finding does not depend on it — the channel exists and the surgical-removal
   contrast isolates it — but the magnitude on the shipped embedding is unmeasured. Closing it means
   one training run with and without the own-label channel.
2. **Whether the appendix's "recorded in the repository released with this work" is true.** The
   determinism measurement exists at
   `docs/archive/fusion-study/results/P0/leakage_ablation/fclass_purity.json`. Whether that path is
   inside the released `github.com/VitorHugoOli/PoiMtlNet/tree/mobiwac` tree I cannot check from
   here (no network). If it is not, that sentence is a promise the release does not keep.
3. **Which auxiliary-term configuration produced each Ch.5 dataset's numbers.** I established that
   the reported substrate is v14 and that v14's build command sets both auxiliaries, but I did not
   verify per-dataset run configurations (Alabama's `dk_ovl` was built at MIN_SEQ=5 against the
   board's 10, per `SUBSTRATE_VERSION_MAP.md:47` — outside my remit, flagged for whoever owns it).
4. **Whether Ch.4's published "generated per category and remapped to each POI"
   (`4_courb/methodology.tex:120`) means per category or per fine class.** The code does per fine
   class (`poi2vec.py:487`). The published sentence reads per category. I did not resolve which the
   published experiments ran; both readings make D-02 worse rather than better, so the finding does
   not turn on it.

---

## Out-of-scope handoffs (one line each, not pursued)

- The gate-glob defect the author reported (`check.sh:12` now `chapters/*.tex chapters/*/*.tex`) is
  fixed in `4e84cf7a`; I confirmed the pattern but did not exercise the checkers in both directions,
  which AGENT_GUARDRAILS §7 requires of a new gate.
- `2_fundamentals.tex:270-274` blocks its own paragraph on two unregistered GLOSSARY terms,
  "bilinear discriminator" and "logistic function". I confirmed both are absent from `GLOSSARY.md`.
  Registry work, not domain review.
- Appendix B's additions count for Article 2 rises from eight to nine (`apx_b_errata.tex:48-51`); a
  number auditor should confirm the rendered paragraph agrees.

## Open questions only the author can answer

1. **D-01/D-02 are one decision, not two.** Does the static-task scope section ship at all before
   the advisor conversation? If it ships, the second qualification must change (D-01) and the frame
   must stop quoting the static number unlabelled (D-02). If it is suppressed by commenting the
   `\input`, D-01 disappears — but D-02 does not, because the frame's number is confounded whether
   or not the appendix says so.
2. Does the arc's diagnosis get re-anchored to Ch.4's **sequential** result, which Appendix B leaves
   standing, or does the static number stay with its caveat attached at the point of use?
3. PCGrad: drop the name from the citing sentence, or add the wiring qualification? (D-06)
