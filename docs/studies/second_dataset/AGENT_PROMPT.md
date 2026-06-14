# second_dataset — second corpus (Massive-STEPS NYC) · Mac track, parallel

> **Status:** SCAFFOLDED, not launched (2026-06-14). Machine: **Mac M2 Pro (user's local box) for ETL + scoring only**
> — no heavy CUDA training here. Position: **Level 0 (ETL, parallel)** + **Level 4 (validation)** of
> [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md). Off the freeze critical path: a different corpus
> touches nothing the freeze pins, so ETL proceeds concurrently; only the validation *runs* depend on the
> frozen champion.
>
> **Read first:** [`docs/research/future_work.md §8`](../../research/future_work.md) (dataset
> recommendation + scoping) and [`docs/research/evaluation_protocol_review.md`](../../research/evaluation_protocol_review.md)
> (the protocol-bridge motivation).

## Why this study exists

The whole project rests on a single source (Gowalla). A second LBSN corpus breaks Gowalla-specificity and
— if it ships temporal splits — delivers the **temporal-split protocol bridge** (roadmap A5) for free,
answering the field's "you used random CV, everyone uses temporal splits" objection in one move.

## Dataset (recommended; confirm with user before ETL)

**Massive-STEPS NYC** ([arXiv:2505.11239](https://arxiv.org/abs/2505.11239),
[GitHub](https://github.com/cruiseresearchgroup/Massive-STEPS), HF per-city). Chosen because it is the
only candidate that simultaneously: (a) is a different source than Gowalla; (b) ships **per-check-in
fine-grained categories AND coordinates** (so both the 7-root category task and TIGER-tract regions
reproduce); (c) ships **temporal train/val/test splits** (= the bridge); (d) is modern (2017–18),
pre-empting the "2012-era data" critique. NYC is the only US city (tract-mappable); the other 14 cities
are cross-city future-work with H3 cells. License Apache-2.0 / CC-BY. Alternatives if rejected:
FSQ-TKY 2014 (max literature comparability, no shipped split, H3 regions) — see `future_work.md §8`.

## Scope

### Phase E (ETL — Mac, parallel with Levels 0–1; **no freeze dependency**)
1. Acquire NYC subset; parse to the repo's check-in schema (userid, placeid, datetime, category, lat, lon).
2. **Category map → 7 roots:** build + document a Foursquare→7-root mapping commensurable with the Gowalla
   states (and/or keep fine-grained for an Acc@k bridge metric). This mapping is a load-bearing, reviewable
   artifact — version it.
3. **Regions:** point-in-polygon join NYC venue coords → TIGER census tracts (mirror the Gowalla region
   definition). Record tract cardinality.
4. Build folds + the shipped temporal split; build the substrate inputs; per-fold train-only priors.
5. Output: a documented `second_dataset` corpus + a stats table (users / check-ins / POIs / tracts /
   categories) paralleling RESULTS_TABLE T1.

### Phase V (validation — Level 4, needs a CUDA box + the FROZEN champion)
Champion **G** + per-task **STL ceilings** + **Markov-1 floor** on Massive-STEPS NYC, 4 seeds. **Scoped as
a validation phase, NOT the full closing_data matrix** — one city, headline cells only. Run under the
shipped temporal split → doubles as the temporal bridge; A5 then shrinks to a Gowalla-side chronological
re-split only.

## Hand-off
Phase E artifacts + the category-map doc land here. Phase V results → a `docs/results/` validation table,
cross-referenced from the new paper's external-validity section. `STATE.md` + `docs/studies/log.md` rows.
