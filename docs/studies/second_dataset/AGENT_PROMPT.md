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

The whole project rests on a single source (Gowalla). A second LBSN corpus breaks Gowalla-specificity, and
on a corpus with per-check-in timestamps it can also deliver the **temporal-split protocol bridge**
(roadmap A5), answering the field's "you used random CV, everyone uses temporal splits" objection.

> ⚠ **Correction (F1, ETL 2026-06-15):** Massive-STEPS does **NOT** ship a temporal split — the shipped
> split is user-stratified **RANDOM** over short trails (median ~2 check-ins), not chronological, and the
> trails don't fit the window-9 protocol (see [`PHASE_E_REPORT.md §F1–F3`](PHASE_E_REPORT.md)). So the
> bridge is **not free**. It **is** still achievable here by building our OWN **chronological per-user
> split** from the per-check-in timestamps — the scoped **Phase E2** item below (Mac-track, no CUDA, no
> freeze dependency). Doing it on Massive-STEPS is now the **recommended** route to A5 (modern non-US+US
> corpus); a Gowalla-side chronological re-split is the fallback.

## Dataset (recommended; confirm with user before ETL)

**Massive-STEPS NYC** ([arXiv:2505.11239](https://arxiv.org/abs/2505.11239),
[GitHub](https://github.com/cruiseresearchgroup/Massive-STEPS), HF per-city). Chosen because it is the
only candidate that simultaneously: (a) is a different source than Gowalla; (b) ships **per-check-in
fine-grained categories AND coordinates** (so both the 7-root category task and TIGER-tract regions
reproduce); (c) carries per-check-in **timestamps** from which we can build our own **chronological
per-user split** (the Phase E2 bridge — the *shipped* split is RANDOM, not temporal; see F1); (d) is
modern (2017–18), pre-empting the "2012-era data" critique. NYC is the only US city (tract-mappable); the
other 14 cities are cross-city future-work with H3 cells. License Apache-2.0 / CC-BY. Alternatives if
rejected: FSQ-TKY 2014 (max literature comparability, no shipped split, H3 regions) — see `future_work.md §8`.

## Scope

### Phase E (ETL — Mac, parallel with Levels 0–1; **no freeze dependency**)
1. Acquire NYC subset; parse to the repo's check-in schema (userid, placeid, datetime, category, lat, lon).
2. **Category map → 7 roots:** build + document a Foursquare→7-root mapping commensurable with the Gowalla
   states (and/or keep fine-grained for an Acc@k bridge metric). This mapping is a load-bearing, reviewable
   artifact — version it.
3. **Regions:** point-in-polygon join NYC venue coords → TIGER census tracts (mirror the Gowalla region
   definition). Record tract cardinality.
4. Build folds + the shipped split (note: it is user-stratified RANDOM, **not** temporal — F1; the
   chronological bridge split is the separate Phase E2 item below); build the substrate inputs; per-fold
   train-only priors.
5. Output: a documented `second_dataset` corpus + a stats table (users / check-ins / POIs / tracts /
   categories) paralleling RESULTS_TABLE T1.

### Phase E2 (chronological per-user re-split — Mac, parallel; **no CUDA, no freeze dependency**) — delivers roadmap A5
This is the **temporal-split protocol bridge** (A5), and it is the corrective to F1 (the *shipped* split is
RANDOM, not temporal). It is pure ETL on the existing parsed corpus, so it runs on the Mac alongside Phase E
with no dependency on the freeze.
1. Order each user's check-ins by **timestamp** (local civil time is fine for per-user chronological
   ordering) and cut a **chronological 80/10/10-ish per-user split** (earliest → train, middle → val,
   latest → test), mirroring the field's universal temporal protocol.
2. Build the window-9 inputs over this split and **matching per-fold train-only priors** (log_T / region
   transition built from the train portion only — no future leakage; honor the CLAUDE.md stale-log_T rule).
3. Output a documented chronological-split artifact per city (NYC + Istanbul), parallel to the within-user
   CV set, ready for the Phase V champion/ceiling/floor runs to deliver A5.
A Gowalla-side chronological re-split remains the **fallback** if the Massive-STEPS route is dropped.

### Phase V (validation — Level 4, needs a CUDA box + the FROZEN champion)
Champion **G** + per-task **STL ceilings** + **Markov-1 floor** on Massive-STEPS NYC/Istanbul, 4 seeds.
**Scoped as a validation phase, NOT the full closing_data matrix** — headline cells only. Lead set =
within-user user-grouped CV (Gowalla-parity). The **temporal bridge (A5)** is delivered by re-running the
same headline cells on the **Phase E2 chronological per-user split** — NOT by the shipped split (which is
RANDOM, not temporal; F1).

> **F2 reporting rule (mandatory):** region cardinalities differ across corpora (NYC 1,912 TIGER tracts;
> Istanbul 520 OSM mahalle [primary] + ~2,585 H3 cells [sensitivity]). Phase V MUST report
> **gap-to-ceiling / lift-over-floor (vs STL ceilings + Markov-1 floor in the same table), NOT absolute
> Acc@k** — the cross-corpus comparison is only apples-to-apples in that framing.

## Hand-off
Phase E artifacts + the category-map doc land here. Phase V results → a `docs/results/` validation table,
cross-referenced from the new paper's external-validity section. `STATE.md` + `docs/studies/log.md` rows.
