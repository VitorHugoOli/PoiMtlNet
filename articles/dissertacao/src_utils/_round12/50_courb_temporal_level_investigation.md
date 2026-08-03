# 50_courb_temporal_level_investigation.md — where does the temporal channel live in Chapter 4?

Round 12, 2026-08-03. Baseline commit `6e43663b`. The author asked for this specifically: "vale um estudo
mais aprofundado no codigo e no docs/, porque o chap. 4 o embedding temporal produz um checking embedding
mas tranasformamos em poi embedding, no final e um poi embedding que entra no modelo."

**The investigation found no such transformation, and the answer inverts the question.** That is worth
saying plainly, because he asked exactly the right question and the evidence does not go where he
expected.

## The tension is INSIDE the published chapter, not just between prose and code

Two sentences of `chapters/4_courb/methodology.tex`, both quoted verbatim:

- **:93, the category task.** "each POI is represented by the embedding resulting from the concatenation
  of the three components, $\mathbf{E}_{cat} = [\mathbf{E}_{HGI} \| \mathbf{E}_{loc} \| \mathbf{E}_{time}] \in
  \mathbb{R}^{192}$, forming pairs $(\mathbf{E}_{cat}, c)$ where $c$ is the real category of the POI"
  For this to be well defined, $\mathbf{E}_{time}$ must have ONE value per POI.
- **:153, the temporal component.** "The final output is the embedding $\mathbf{E}_{time} \in \mathbb{R}^{64}$,
  which represents the timestamp of each check-in."
  One value per CHECK-IN. A POI visited at 08:00 and at 20:00 has two different vectors.

**These cannot both be true as written.** Either the temporal channel is aggregated to the POI, in which
case :153 should say so, or the category task cannot consume it, in which case :93 should say so.

## The code answers it, and it answers the second way

Four independent places, all read this session:

| where | what it says |
|---|---|
| `src/data/inputs/builders.py:40` | `EmbeddingEngine.TIME2VEC` is a member of `_CHECKIN_LEVEL_ENGINES` |
| `src/data/inputs/builders.py:191-192` | the category-task builder raises: "Rejects check-in-level engines (Time2Vec, Check2HGI) -- category task requires one embedding per POI" |
| `research/embeddings/time2vec/README.md:66-69` | "Time2Vec is check-in level, which means each check-in has its own embedding (varies with time of day), so a single placeid can have many different embeddings. The category task in MTLnet expects one embedding per POI and explicitly rejects Time2Vec in `src/data/inputs/builders.py`." Output stated as `N_checkins x (6 metadata + 64 dims)` |
| `docs/context/EMBEDDINGS.md:83,:103-104` | "Type: Check-in-level (one embedding per visit event)" and "Check-in-level embeddings mean the 9-step next-task window becomes a true spatio-temporal trajectory, not just a POI sequence" |

`src/data/inputs/fusion.py` carries two alignment paths, `align_poi_level` merging on `placeid` (:49) and
`align_checkin_level` merging on the composite key `(userid, placeid, datetime)` (:122), and the comment
at :44-45 assigns Time2Vec to the check-in path. **There is no aggregation function anywhere in the
Time2Vec pipeline: no `groupby`, no mean, no per-POI reduction.**

## What this entitles, and what it does not

**Entitled.** The CURRENT pipeline cannot feed the temporal channel to the category task, and :93 as
written is inconsistent with :153 and with the code.

**NOT entitled, and this is the important half.** I cannot establish what the PUBLISHED CoUrb RUN did.
The chapter is a version of record from experiments that ran earlier; the code on disk is today's. Three
possibilities stay open, and only the author can close them, because distinguishing them needs the run
artifacts rather than the source tree:

1. the published run aggregated $\mathbf{E}_{time}$ to POI level, and that step has since been removed;
2. the published run's category task ran WITHOUT the temporal channel, and :93 overstates its input;
3. the published run predates the guard and fed check-in-level vectors to the category task, which would
   make the published category numbers describe a different input than :93 claims.

Under `AGENT_GUARDRAILS` §1 this is a `[VERIFY]`: the inconsistency is measured, the resolution is not
mine. **Chapter 4 is a version of record and was not edited.** If the resolution turns out to be (2) or
(3), it is an errata matter under `NORTH_STAR` §5.7, and the errata appendix does not currently carry it
(checked: `apx_b_errata.tex` mentions the category table lead and one typographical re-typeset of this
same methodology passage at :311-318, but nothing about the temporal channel's level).

## The consequence for Chapter 2, which is what the decision was for

**My option 3 was wrong, not right.** I had offered "keep place-level, the vector is still indexed by the
visited POI" as defensible; `EMBEDDINGS.md:103-104` says the next-task window is explicitly NOT a POI
sequence.

**And I cannot write that the temporal channel is aggregated to the place**, which is the wording the
author's premise would have produced. That is the one claim the code refutes.

What Chapter 2 CAN say, and stay true under all three possibilities, is that Chapter 4 replaces the
monolithic place vector with a DECOMPOSED representation whose components are learned by separate
encoders, without asserting a single level for the result. The level question belongs to Chapter 4 and is
now open there.

---

## CLOSED, 2026-08-03: the author pointed at the original code and the answer is a FOURTH possibility

He said: "Calma analise o codigo original do chap. 4: /Users/vitor/Desktop/mestrado/temp/tarik-new, antes
de decidirmos." That repository is the CoUrb-era code, and it settles what this document said only the run
artifacts could settle. **None of the three possibilities above was correct.**

### What the original code does

**1. The temporal embedding is per CHECK-IN, and the notebook's own stored outputs prove it numerically
rather than by reading intent.** `Time_Encoder.ipynb`, California:

| cell | stored output | what it means |
|---|---|---|
| 2 | `N checkins (antes de filtrar): 2535573` | the input is the check-in table |
| 3 | `(2535573, 2)` | two features per check-in |
| 13 | `time_embeds_sin shape: (2535573, 64)` | **one 64-d row per check-in** |

Cell 3 shows the features are `t_hour = hour/24` and `t_dow = dow/7`, computed per check-in, so two visits
to the same POI at different times produce **different** vectors. Cell 14 assembles those rows into a frame
keyed by `placeid` and writes it to a path whose own name states the level,
`time_encoder_embeddings_sin_CHECKIN_{estado}.csv`; cell 15 copies it to
`data/output/{estado}/time_embedding_novo.csv`, which is what the ETL reads.

**2. The category-task input then reduces check-in level to POI level by DISCARDING ROWS, not by
aggregating them.** `PoiMtlNet_Novo/src/etl/create_inputs_hgi.py:437`, verbatim:

    time_df = time_emb[["placeid"] + num_cols_time].drop_duplicates("placeid")

With one row per check-in keyed by `placeid`, `drop_duplicates("placeid")` **keeps the first occurrence of
each POI and throws every other visit away.** The three components are then inner-joined on `placeid`
(:441-443) and the category attached per `placeid` (:448), which produces the $(\mathbf{E}_{cat}, c)$ pairs
of `methodology.tex:93`. `process_state`'s default is `cat_embeddings=("poi","loc","time")`, so the temporal
channel IS in the category input.

### The answer, and why it was not among the three

**There is a check-in-to-POI reduction, and it is `drop_duplicates`, not an aggregation.** The author's
instinct that something converts the level was right; the operation is not a mean or a pooling. It selects
one arbitrary visit per POI. So the temporal channel reaching the category task carries the timestamp of a
single visit, not a summary of that POI's visits, and the variation the encoder was built to capture is
discarded for that task.

Against the three possibilities this document left open:

| possibility | verdict |
|---|---|
| (1) an aggregation later removed | **closest but wrong in the operative word.** A level reduction exists and is still in the original code. It is not an aggregation. |
| (2) the category task ran without the temporal channel | **refuted.** `cat_embeddings` defaults to include `time`. |
| (3) check-in-level vectors fed to the category task | **refuted for the category task.** They are reduced to one row per `placeid` first. |

The true answer is a fourth one nobody had listed, which is the argument for having stopped here rather
than picking from three.

### What this means for `methodology.tex:93` and `:153`

Both sentences are now explicable, and neither is false in the way a reader would first suspect. `:153` is
correct: the encoder does produce one vector per check-in. `:93` is correct that a POI-level 192-d vector
is paired with the POI's category. **What the published text never states is the step between them**, and
that step is lossy in a way a reader would want to know about: one visit per POI survives and the rest are
dropped.

That is a gap in the description, not a wrong number: the pairs the category task trained on are exactly
what `:93` says they are. Whether the chapter should record the selection step is an **errata** question
under `NORTH_STAR` §5.7, and `apx_b_errata.tex` does not carry it today. **Chapter 4 was not edited.**

### What Chapter 2 may now say

The neutral wording this document recommended is still the safe one, and it is now also the accurate one:
Chapter 4 replaces the monolithic place vector with a **decomposed representation whose components are
learned by separate encoders**. Chapter 2 may add that the components are combined **at the place level for
the static task**, which is true and sourced. It must not say the temporal channel is *aggregated* to the
place, because the operation discards rather than combines.
