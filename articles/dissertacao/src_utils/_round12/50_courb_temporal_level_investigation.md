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
