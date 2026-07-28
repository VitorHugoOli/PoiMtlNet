# GLOSSARY.md — the dissertation's term registry (v1, 2026-07-18)

> **What this is.** The single registry of every technical term, name, and symbol the
> dissertation uses — [`WRITING_LAW.md`](WRITING_LAW.md) states the *rules* (one name per
> concept, gloss at first use); THIS file holds the *names themselves*. It is the consistency
> artifact AGENT_GUARDRAILS L2 requires: agents lint their prose against this registry instead
> of trusting memory of earlier chapters.
>
> **Maintenance rule (fail-closed):** a term not in this registry may not be used in
> dissertation prose. Agents PROPOSE new entries in their handoff notes; the author approves;
> the entry lands here BEFORE the term lands in text. When a definition here conflicts with a
> source paper's local usage, the chapter keeps the paper's usage and the frame uses this
> registry (the per-paper mapping in §2 bridges them).

---

## 1 · The tasks (keep the three distinct; say "we do not predict the exact next place" once, early)

| Term | Definition (gloss at first use) | Notes |
|---|---|---|
| **next-category prediction** (next category) | Given a user's recent check-in sequence, predict the **category** of the next visited POI (7 top-level classes). | Present in all three papers. Hyphenate as compound adjective only. |
| **next-region prediction** (next region) | Given the recent sequence, predict the **region** of the next check-in among the dataset's regions (a census tract in the U.S. states; a mahalle in Istanbul). | Introduced in MobiWac (Ch.5). Never "area", never "cell". |
| **next-place prediction** (next place) | Predicting the exact next POI. **Out of scope for this dissertation** — named only to delimit. | Never conflate with the two above. |
| **category classification** | Static task: classify a POI's category from its representation (no sequence). | CBIC/CoUrb task A. |

**Per-paper task mapping (the bridge the frame must state once):**

| Paper | Paper's own term | Canonical term here |
|---|---|---|
| CBIC | "next-POI prediction" / "next-POI category prediction" (task B) | **next-category prediction** (the label is the next POI's category) |
| CBIC | "POI category classification" (task A) | **category classification** |
| CoUrb | "previsão do próximo POI" (categoria) | **next-category prediction** |
| MobiWac | "next category" + "next region" | identical — no mapping needed |

## 2 · Model and artifact lineage (the table Ch.2 renders; one name per artifact, everywhere)

| Name | What it is | Introduced | Dissertation chapter |
|---|---|---|---|
| **DGI** | Deep Graph Infomax — self-supervised graph representation learning; here: 64-d POI embeddings from a Delaunay POI graph (one-hot category features, distance-decay edge weights). | Veličković et al.; used as input in CBIC | Ch.2 (defined), Ch.3 (used) |
| **HGI** | Hierarchical Graph Infomax — hierarchical place→region→city embeddings; the standard place-level baseline representation. | prior work; baseline in MobiWac Part 1 | Ch.2, Ch.5 |
| **MTLnet** | The joint architecture of the CBIC paper: task-specific MLP encoders → FiLM modulation → shared residual trunk → task heads (category ensemble; Transformer next-head). **Canonical spelling: `MTLnet`, lowercase `n`** (author decision 2026-07-26). The published CoUrb paper typesets it `MTLNet`; Ch.4 was normalized to `MTLnet` and the departure is listed in Appendix B. Never `MTLNet`, `MTL-Net`, or `MtlNet` in dissertation prose. | CBIC (Vitor) | Ch.3 (introduced), recapped Ch.4/Ch.5 related work |
| **FiLM** | Feature-wise Linear Modulation — per-task learned γ/β scaling that conditions shared layers on task identity. | Perez et al.; used in MTLnet | Ch.2 or Ch.3 (gloss once) |
| **Nash-MTL** | Gradient-balancing MTL optimizer (Nash bargaining over task gradients); used in CBIC/CoUrb. | Navon et al. | Ch.3/Ch.4; the frame does NOT amplify its benefit claims (NORTH_STAR §4 caveat) |
| **ST-MTLNet** | CoUrb's variant: MTLnet unchanged, input replaced by concatenated spatial + temporal + categorical encoders (64-d each → 192-d). **Canonical spelling: `ST-MTLNet`, capital `N`** — a separate registered name, not the MTLnet rule with a prefix; it is the published title of the CoUrb paper and keeps that form. The expansion is *Spatial-Temporal MTLNet*, also as published. | CoUrb (Tarik 1st author) | Ch.4 |
| **SIREN / Sphere2Vec-M** | The two spatial encoders compared in CoUrb (sinusoidal MLP on normalized coordinates; multi-scale spherical encoder). | CoUrb | Ch.4 only |
| **Time2Vec** | The temporal encoder (hour-of-day + day-of-week) in CoUrb. | Kazemi et al.; CoUrb | Ch.4 only |
| **Check2HGI** | **Our check-in-level representation**: extends the place→region→city hierarchy with a fourth check-in level, trained without task labels (hierarchical graph infomax); yields one vector per **visit** (semantic set) plus region vectors (spatial set). | MobiWac (Vitor) | Ch.5 (the centerpiece); gloss = "each visit gets its own vector" |
| **the joint model** (MobiWac) | One multi-task model, one forward pass, two predictions: private per-task encoders → shared cross-attention trunk → category output (trunk) + region output (trunk + private spatial path). | MobiWac | Ch.5. In prose: "the joint model"; repo id `mtlnet_crossattn_dualtower` NEVER appears in text |
| **dedicated single-task model** | The comparison arm: one model trained for one task (the "ceiling" the joint model is measured against). | all papers | everywhere; never bare "baseline" |
| **the shared trunk** | The shared middle of the joint model, introduced once as "a shared cross-attention stack (the trunk)". | MobiWac GLOSSARY ruling | Ch.5 + frame |
| **label-history benchmark** (next category) | The best macro-F1 reached by four specified predictors that read only the genuine category history of the input window, with no representation read: persistence, and a balanced logistic model on the last category, on window category counts, and on positional one-hots. A property of the label sequence, NOT of any encoder. Measured: FL 0.3617, AL 0.2800, AZ 0.3232, CA 0.3242, IST 0.3016 (Appendix D). **It is NOT an upper bound:** the four predictors are a specified set, and a better predictor of the same restricted information could exceed them. Say so once, at first use. | this document, 2026-07-26; renamed 2026-07-27 | Ch.5 + Appendix D. Two quantities that must NOT be swapped: this one, and the **clean reference encoder** below, which is what the screen actually gates on |
| ~~**label-only ceiling**~~ | **RETIRED 2026-07-27** (author decision). Superseded by **label-history benchmark**, above. The quantity is unchanged; only its name is. It was never an upper bound, so "ceiling" overstated it. Do not reintroduce "ceiling", "autocorrelation ceiling", or "what the past itself allows" for this quantity. | — | The word *ceiling* stays correct for the **dedicated single-task model** row of this table, a different quantity: the score of a real trained model |
| **clean reference encoder** | The encoder a screened candidate is compared against in the leak-sniff screen (at Florida, our own graph encoder at 0.4090 standardized / 0.4074 raw). The screen's verdicts are relative to it, with a three-point margin. It scores ABOVE the label-history benchmark and is not the same quantity. | MobiWac screening record | Ch.5 + Appendix D; never call this "the ceiling" |

## 3 · Data and protocol terms

| Term | Definition | Notes |
|---|---|---|
| **check-in** | One visit record (user, POI, timestamp) from an LBSN. | never "event" |
| **POI / place** | Point of Interest — a place a person can visit. | expand POI at first use; never "venue" |
| **region** | The spatial prediction unit: a census tract (U.S.); a mahalle (Istanbul). | name the unit at first use |
| **Gowalla** | Public LBSN dataset (2009–2010 check-ins); we use five U.S. states: AL, AZ, FL, CA, TX. | vintage stated in limitations |
| **Istanbul (Massive-STEPS)** | The non-U.S. dataset (Ch.5): Istanbul check-ins from the Massive-STEPS benchmark, mapped to the 7-category taxonomy. | |
| **the 7-category taxonomy** | Community, Entertainment, Food, Nightlife, Outdoors, Shopping, Travel. | identical across chapters |
| **fold** | One of the five data splits of cross-validation. | gloss via "5-fold cross-validation" once |
| **seed** | One complete repetition of the five-fold experiment, same folds, different random initialization. | abstract/intro say "random initialization"; banned: "run", "multi-seed run" |
| **user-disjoint split** | CV where a user's data never spans train and test (StratifiedGroupKFold by user) — Ch.5's protocol. | Ch.4 used sample-stratified splits — VERIFIED firsthand 2026-07-23 from the CoUrb codebase (plain StratifiedKFold, userid dropped; NORTH_STAR §4 Ch.4 has file/line); the frame states the difference plainly. Ch.3's split: same pipeline family, but verify from the CBIC codebase before asserting it in prose |
| **sliding windows** | Overlapping windows of the last 9 visits, one starting at each visit (stride 1) — Ch.5. | Ch.3/Ch.4 used non-overlapping windows; "sliding" is the ONE name (MobiWac ruling) |
| **transductive** | The representation was trained seeing all places (not new-place-generalizing). | gloss only where the leak discussion needs it |
| **leakage audit** | The controlled test (A4) verifying the transductive representation does not leak labels across the user-disjoint split (null: ≤0.33 pp region, ≤0.29 pp category). | Ch.5 §5.2 material |
| **bilinear discriminator** | The scoring function inside an infomax objective: it maps a pair of embeddings to a compatibility score through a learned weight matrix, $\mathcal{D}(\mathbf{e}_1,\mathbf{e}_2)=\sigma(\mathbf{e}_1^{\top}\mathbf{W}\mathbf{e}_2)$. Bilinear because it is linear in each argument separately. | Ch.2's Check2HGI equations. Registered 2026-07-28 to clear the fail-closed block the drafting agent recorded at `2_fundamentals.tex:307-314`. Ch.3 already uses the bare word "discriminator" in published prose, so only the modifier is new |
| **logistic function** | $\sigma(z)=1/(1+e^{-z})$, which maps a real score to the interval between 0 and 1. | Ch.2, naming $\sigma$ in the discriminator equation. Say "logistic function", never "sigmoid", in prose |
| **fine class** | The fine-grained class label attached to each place by the source data, one level below the 7-category taxonomy (Airport, Coffee Shop, Seafood): 284 to 365 distinct values per state, each mapping to exactly one top-level category. | Appendix B §B.5 only, where the static-task scope needs it. In code this column is `spot`, renamed `fclass` at `hgi/preprocess.py:62`; NEVER write `fclass` in prose |
| **early stopping** | Halting training when a validation metric stops improving, rather than running the configured epoch budget to the end. | Used only to say the published chapters did NOT use it: they run the full budget and read each task at its own best validation epoch. Gloss at first use |
| **bottleneck** | The one factor limiting performance at a given stage: the input representation in Ch.4's diagnosis, not the sharing scheme. | Registered 2026-07-28 because the PT Resumo needs an anchor for *gargalo*. Never a metaphor for anything else |

## 4 · Metrics, statistics, conventions

| Term | Definition (defensive form: formula + plain reading + boundary) | Notes |
|---|---|---|
| **macro-F1** | Mean of per-category F1; counts every category equally, so rare ones matter. Out of 100. | category metric everywhere |
| **Acc@10** | Share of test check-ins whose true next region is among the model's top-10 predictions. | region metric (Ch.5) |
| **OOD-discounted Acc@10** | Acc@10 on in-distribution samples × (1 − out-of-distribution fraction) — regions unseen in training count as misses. | the "full" region metric of the MobiWac cells; define once in Ch.5 |
| **paired superiority test** | Paired *t* on the four per-seed means (inferential unit: n = 4, three degrees of freedom), which is the reported convention; the registered per-fold Wilcoxon signed-rank over the 20 fitted models (n = 20) is reported alongside it and agrees. Verdict verb: **"outperforms"**. | verbs bound to tests (WRITING_LAW §3). At n = 4 the exact one-sided Wilcoxon floors at 0.0625 whatever the effect size, which is why the *t* carries the verdict; both footings and the departure are logged in `docs/studies/closing_data/log.md` D-1 to D-3 |
| **TOST non-inferiority** | Two one-sided tests within a **two-point margin**; verdict verb: **"matches"** ("statistically non-inferior within a two-point margin (TOST)" appears in full at least once). | "margin" is reserved for this; representation differences are "gaps" |
| **Holm correction** | Multiplicity control across the six per-dataset tests. | cite, don't gloss textbook tests |
| **n = 20** (fitted models) and **n = 4** (inferential unit) | 4 seeds × 5 folds per cell = 20 fitted models. All four seeds reuse the same fixed fold partition, and the reported tests pair the four per-seed means, so the inferential unit is n = 4. | state the arithmetic once (Ch.5 protocol); never write "n = 20 paired repetitions" for the reported test. Design: `docs/studies/closing_data/v17_completion/STATISTICAL_PROTOCOL.md`:18-19 (seeds {0,1,7,100} × 5 folds), :187-190 (one fixed fold partition across arms); executed footings: `.../stats_n20/RESULTS.md`:65-67 |
| **joint-best convention** | Both tasks read at the one saved model per fold (the validation-selected epoch). Ch.5's reported convention. | never mix with per-task diagnostic-best without saying so |
| **the floors** | Reference points: majority-class floor (category), Markov-1 floor (region). | every number carries its reference point |

## 5 · Acronym list (seeds the List of Abbreviations; expand each at first use)

LBSN · POI · MTL · STL · CV · DGI · HGI · Check2HGI · FiLM · GRU · OOD ·
TOST · CI (confidence interval) · SGKF (only if the protocol section needs it) ·
CBIC · CoUrb · SBRC · MobiWac · BRACIS (appendix only) · UFV · PPGCC

Keep the count low; method internals (STAN, CTLE, HMT-GRN, ReHDM, POI-RGNN — baseline names)
are named once each in Ch.5's baseline list and the List of Abbreviations only if used ≥3×.

## 6 · Portuguese equivalents (Resumo + AcademicoPG fields + folha de rosto)

| EN (body) | PT (Resumo/system) |
|---|---|
| multi-task learning | aprendizado multitarefa |
| single-task / dedicated model | modelo dedicado (tarefa única) |
| point of interest (POI) | ponto de interesse (POI) |
| check-in / visit | check-in / visita |
| check-in-level representation | representação em nível de check-in |
| place embedding | representação (embedding) em nível de POI |
| next-category prediction | previsão da próxima categoria |
| next-region prediction | previsão da próxima região |
| census tract | setor censitário |
| sliding windows | janelas deslizantes |
| cross-validation (5-fold) | validação cruzada (5 partições) |
| random initialization (seed) | inicialização aleatória (semente) |
| non-inferiority (TOST, 2-point margin) | não-inferioridade estatística (TOST, margem de 2 pontos) |
| outperforms / matches | supera / equipara-se (estatisticamente) |
| leakage audit | auditoria de vazamento |
| shared trunk | tronco compartilhado |
| n = 20 (fitted models) | vinte modelos ajustados (por configuração) |
| user-disjoint split | usuários disjuntos entre treino e teste |
| fold (one of five) | partição (uma de cinco); *as cinco* partições fixas |
| per-seed mean(s) | média por inicialização; *plural* médias por inicialização |
| joint-best convention | seleção *joint-best* |
| cross-attention trunk | tronco de atenção cruzada |
| decomposed encoders (spatial, temporal, categorical) | codificadores decompostos (espacial, temporal, categórico) |
| hard parameter sharing | compartilhamento rígido de parâmetros |
| sharing topology | topologia de compartilhamento |
| bottleneck | gargalo |
| paired superiority test | teste pareado; *plural* testes pareados |

The Resumo mirrors the Abstract claim-for-claim (WRITING_LAW §6); this table keeps the pair
translation-stable. "Embedding" may stay as a loanword in PT (standard in the BR community)
with "representação" as the running word.

**The last nine rows were added 2026-07-28**, after the Resumo was cut and rebuilt. Each was already
in use in the Resumo and none was registered, which the fail-closed rule (§1) does not permit. Before
adding each row I confirmed its English counterpart is itself registered above: `n = 20 (fitted
models)` §4, `user-disjoint split` §3, `fold` and `seed` §3, `joint-best convention` §4,
`cross-attention trunk` in **the joint model** §2, `the shared trunk` §2. Three had no English row of
their own and are ordinary MTL vocabulary rather than names this project coined, so they are
registered here as translation pairs only: decomposed encoders, hard parameter sharing, sharing
topology.

Two notes on the choices, since a later translator will second-guess them:

- **`seleção joint-best` keeps the English term of art.** It names a selection rule this document
  defines (§4, "joint-best convention"), it appears in the results tables in English, and inventing a
  Portuguese calque would make the Resumo and the tables disagree. The Resumo italicizes it, which is
  the BR convention for a retained foreign term.
- **`partição` for `fold`, and `as cinco partições fixas`.** The existing row translates "5-fold
  cross-validation" as "validação cruzada (5 partições)", so `partição` was already the registered
  word for the unit; these rows make the singular and the "fixed across seeds" sense explicit,
  because the Resumo needs to say that the five are the *same* five at every seed.

## 7 · Banned and repo-internal words (pointer)

The ban lists live in [`WRITING_LAW.md`](WRITING_LAW.md) §2 (repo codenames: B9, v11–v17,
champion-G, dk_ovl, log_T, "substrate", "engine", "board", "recipe"…) and §4 (AI-tells). This
registry deliberately contains NO entry for them: if a term is not here and not proposable
under the maintenance rule, it does not exist for this dissertation.
