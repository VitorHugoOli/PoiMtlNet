# CBIC Florida dataset recompute — result (2026-07-24, corrections round 2)

> Fills the [VERIFY] placeholders at `src/chapters/3_cbic.tex` (the CBIC dataset sentence:
> "This subset comprises a total of N_users users, N_poi unique POIs, and N_checkins check-ins").
> Computed from the SANCTIONED per-state ETL output the author pointed to
> (`data/checkins/` -> `data/checkins_by_state/Florida.parquet`, 1,407,034 rows), NOT a fresh
> spatial join. Fail-closed: numbers are reported, the author confirms the basis before they go
> into Ch.3. Still marked [VERIFY] until that confirmation.

## The two candidate bases

The CBIC method (and CoUrb, verbatim) build the modelling data by ordering each user's check-ins
chronologically and **discarding users with fewer than five visits**. That gives two natural
counts, and the CBIC prose does not state which one its "subset comprises" sentence refers to:

| Basis | Users | Unique POIs | Check-ins |
|---|---|---|---|
| **(a) Raw Florida subset** (all rows) | **21,052** | **76,544** | **1,407,034** |
| **(b) After the <5-visit-per-user filter** (users with >=5 check-ins) | **13,935** | **76,266** | **1,392,262** |

The filter drops 7,117 users (33.8%) but only ~15k check-ins (1.0%) and 278 POIs — i.e. it
removes many one-off users who contribute almost no data, which is why the check-in and POI
totals barely move.

## Cross-check against CoUrb's published Florida row (CROSS-CHECK ONLY, not a source)

CoUrb 2026 (`src_en/resultados/tabela_dataset.tex`, Table "tab:dataset") reports for Florida:
**990,518 check-ins / 65,009 POIs / 20,301 users**.

These are LOWER than both bases above, and that is expected: CoUrb's pipeline filters further than
the raw <5-visit rule — it keeps only check-ins usable by its category-classification + windowed
Next-POI tasks (valid category label, windowing of size 9, embedding coverage), which drops
additional rows and POIs. The CoUrb user count (20,301) sits between the raw (21,052) and the
<5-visit-filtered (13,935) counts, confirming CoUrb applies a DIFFERENT (task-specific) filter,
not the plain <5-visit one. Per the CBIC ERRATA, CoUrb's row is a sanity cross-check, never the
source; the agreement is order-of-magnitude and directionally consistent (same state, ~1-2.5M
check-ins, tens of thousands of POIs), which is all the cross-check is meant to establish.

## Recommendation (author confirms)

- If the CBIC "subset comprises" sentence is meant to describe **the data actually fed to the
  model** (the usual meaning, and the way CoUrb's own table is built), use **basis (b), the
  <5-visit-filtered counts: 13,935 users / 76,266 POIs / 1,392,262 check-ins.**
- If it is meant to describe **the raw Florida slice before filtering**, use **basis (a):
  21,052 / 76,544 / 1,407,034.**
- The safest wording states both: "a Florida subset of 21,052 users and 76,544 POIs across
  1,407,034 check-ins; after discarding users with fewer than five visits, 13,935 users and
  76,266 POIs across 1,392,262 check-ins remain" — this is fully faithful and needs no choice.

## Bonus finding (feeds Tier-3 MJ-5): data vintage

The Florida check-ins span **2009-2011** (2009: 40,304; 2010: 769,792; 2011: 596,938), NOT
"2009 and 2010". This confirms the review-suite flag MJ-5: Ch.6's "2009 and 2010" is incomplete
and should read 2009-2011 (or "2009 to 2011"). Author decision in the pt_BR decisions doc.

## 7-class taxonomy check

The Florida data carries exactly the 7 top-level categories the dissertation uses: Community,
Entertainment, Food, Nightlife, Outdoors, Shopping, Travel. No taxonomy drift.

## Provenance

- Source: `/Users/vitor/Desktop/mestrado/data/checkins_by_state/Florida.parquet` (the target of
  the `data/checkins` symlink; the committed per-state ETL output, mtime 2025-04-14).
- Columns used: userid, placeid, datetime, category. Computed with pandas/pyarrow this session.
- No fresh spatial join was run (the author confirmed the per-state split already exists); the
  redundant ETL sub-agent was stopped.
