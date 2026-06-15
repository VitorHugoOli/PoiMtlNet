# Foursquare(v1) → Gowalla-7-root category map — Massive-STEPS (all cities)

**Version:** `v1-2026-06-15` · **Study:** second_dataset (Phase E2) · **Status:** built, awaiting review

> Load-bearing, reviewable artifact. **Shared across all Massive-STEPS cities** (they use
> the same FSQ v1 taxonomy). Source of truth = the crosswalk in
> `scripts/second_dataset/build_category_map.py` (`FSQ_ROOT_TO_GOWALLA`). Per-city generated
> artifacts: `data/massive_steps_<city>/category_map.{csv,json}` (gitignored data).
> Regenerate: `python scripts/second_dataset/build_category_map.py --city <city>`.
>
> **Leaf-id coverage: 100%** at both built cities — NYC 585/585, Istanbul 580/580.
> The crosswalk is identical; only the resulting *distribution* differs by city (NYC vs
> Istanbul, see [`STATS_T1.md`](STATS_T1.md) — e.g. Istanbul Outdoors 26.8% vs NYC 12.9%,
> reflecting Istanbul's universities / neighborhoods / cafés). The distribution table
> below is **NYC**; Istanbul's is in STATS_T1.

## Goal

Make the NYC category task **commensurable** with the Gowalla states, whose POIs
carry one of 7 *root* categories (`src/configs/globals.py CATEGORIES_MAP`:
Community, Entertainment, Food, Nightlife, Outdoors, Shopping, Travel). NYC ships
**585 fine-grained Foursquare v1** category leaves; we collapse them to the same 7
roots. The fine FSQ label is **preserved** in the corpus for a future Acc@k bridge
metric on native FSQ categories.

## Method (principled + fully reproducible)

1. **Leaf → FSQ root.** Each of the 585 FSQ leaf ids → its FSQ v1 *top-level* root
   (10 roots) via the canonical FSQ v1 taxonomy tree (`fsq_v1_tree.json`, 937
   leaves). **Coverage of NYC leaf-ids = 585/585 (100%).** Tree source:
   `github:MettiHoof/3circles/.../4sq_categories.json` (a standard mirror of the
   FSQ v1 `venues/categories` tree; verified to contain all NYC leaf ids).
2. **FSQ root → Gowalla root** via the 10→7 crosswalk below.
3. We follow **FSQ's own hierarchy consistently** rather than replicating
   Gowalla's per-leaf idiosyncrasies. This keeps the map objective (no 585 manual
   judgement calls) and reproducible. Known philosophical divergences are listed
   below; a reviewer can override individual leaves in the CSV if Gowalla-exact
   alignment is preferred.

## The crosswalk (10 FSQ roots → 7 Gowalla roots)

| FSQ v1 top-level root | Gowalla root | Justification |
|---|---|---|
| Arts & Entertainment | **Entertainment** | direct |
| Event | **Entertainment** | concerts/parties → Gowalla maps "Concert","Party" → Entertainment |
| Food | **Food** | direct |
| Nightlife Spot | **Nightlife** | direct (Gowalla: Bar/Brewery → Nightlife) |
| Outdoors & Recreation | **Outdoors** | direct (Gowalla: "Plaza / Square","Other - Parks" → Outdoors) |
| Shop & Service | **Shopping** | direct (Gowalla: stores + services → Shopping) |
| Travel & Transport | **Travel** | direct (Gowalla: hotels/airports/transit → Travel) |
| College & University | **Community** | cf. Gowalla "Other - College & Education" → Community; campus spots sit under Gowalla's Community root |
| Professional & Other Places | **Community** | offices, medical, government, spiritual, generic buildings; Gowalla has no "work" root, Community is the closest commensurable bucket |
| Residence | **Community** | homes / apartments / condos |

All 7 Gowalla roots are populated; no leaf falls through to `None`.

## Resulting distribution (check-in-weighted, n=272,368)

| Gowalla root | check-ins | share | # FSQ leaves |
|---|---:|---:|---:|
| Food | 75,831 | 27.8% | 144 |
| Community | 42,677 | 15.7% | 117 |
| Shopping | 37,332 | 13.7% | 127 |
| Outdoors | 35,055 | 12.9% | 69 |
| Travel | 33,149 | 12.2% | 44 |
| Nightlife | 28,627 | 10.5% | 24 |
| Entertainment | 19,697 | 7.2% | 60 |

## Known divergences from Gowalla's idiosyncratic grouping

A reviewer wanting Gowalla-*exact* alignment (rather than FSQ-consistent) should
weigh these. Default = follow FSQ.

1. **Gym / Fitness / Yoga Studio → Outdoors** (FSQ "Outdoors & Recreation").
   Gowalla idiosyncratically files fitness under **Shopping/Services**. This is
   the single largest divergence — ~7k NYC check-ins ("Gym" 4,110 + "Gym /
   Fitness Center" 3,083). Moving these to Shopping would shift Outdoors 12.9%→10.4%
   and Shopping 13.7%→16.2%.
2. **Neighborhood / Plaza / Scenic Lookout → Outdoors** (FSQ "Outdoors &
   Recreation"). Matches Gowalla's "Plaza / Square" → Outdoors.
3. **Office / Building / Coworking → Community** (FSQ "Professional & Other
   Places"). No Gowalla "work" root exists; Community is the nearest bucket.

## Caveats

- The map keys on `venue_category_id` (the stable FSQ id), not the display name,
  so renamed leaves remain correctly mapped.
- Coordinates absent (7.3% of check-ins, see `STATE.md`) do not affect the
  category map; those rows still carry a valid 7-root category.
- The fine FSQ category (`venue_category` / `venue_category_id`) is retained in
  the corpus → an Acc@k bridge metric on the 585-way native labels is available
  to Phase V without re-derivation.
