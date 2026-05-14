# POI Benchmark Datasets

Reference catalog for datasets suitable for next-POI prediction and POI category labeling. Each dataset is evaluated for compatibility with the MTLnet pipeline, which requires: **category labels mappable to the 7-class Foursquare taxonomy**, temporal check-in sequences per user, and at least one geographic split suitable for state/city-level analysis.

---

## Quick Verdict

| Dataset | Category labels | Sequences | Public | ETL | Priority |
|---------|----------------|-----------|--------|-----|----------|
| Gowalla (state-split) | 7 super-categories | Yes | Yes (SNAP) | Implemented | — |
| **Foursquare NYC** | 9 top-level FSQ cats | Yes | Yes | **Planned** | 1 |
| **Foursquare Tokyo** | 9 top-level FSQ cats | Yes | Yes | **Planned** | 2 |
| **Massive-STEPS** | FSQ v2 hierarchy | Yes | Yes (HuggingFace) | **Planned** | 3 |
| Brightkite | Sparse / undocumented | Yes | Yes (SNAP) | Document-only | — |
| Foursquare GSCD | FSQ hierarchy | Yes | Limited | Document-only | — |
| Yelp Open Dataset | 80+ business cats | Partial | Yes (non-commercial) | Document-only | — |

---

## Category Mapping to 7-Class Taxonomy

The pipeline uses this fixed taxonomy (`src/configs/globals.py`):

```
Community, Entertainment, Food, Nightlife, Outdoors, Shopping, Travel
```

### Foursquare TIST2015 (NYC + Tokyo)

The raw `venue_category_name` field contains Foursquare top-level categories:

| Raw venue_category_name | Mapped super-category |
|-------------------------|-----------------------|
| Arts & Entertainment | Entertainment |
| College & University | Community |
| Food | Food |
| Nightlife Spot | Nightlife |
| Outdoors & Recreation | Outdoors |
| Professional & Other Places | Community |
| Residence | Community |
| Shop & Service | Shopping |
| Travel & Transport | Travel |
| Event | **dropped** (ambiguous, <0.5% of check-ins) |

### Massive-STEPS

Massive-STEPS uses Foursquare API v2 categories (hierarchical). Top-level parent categories are the same 9 buckets above; subcategories (e.g. "Italian Restaurant" → Food) are resolved via the bundled `categories.csv` hierarchy file:

| Top-level parent | Mapped super-category |
|------------------|-----------------------|
| Arts & Entertainment | Entertainment |
| College & University | Community |
| Food | Food |
| Nightlife Spot | Nightlife |
| Outdoors & Recreation | Outdoors |
| Professional & Other Places | Community |
| Residence | Community |
| Shop & Service | Shopping |
| Travel & Transport | Travel |

---

## Dataset Profiles

---

### Foursquare NYC (Yang TIST2015)

| Attribute | Value |
|-----------|-------|
| **Abbreviations** | FSQ-NYC, Foursquare-NYC |
| **Source** | Dingqi Yang — https://sites.google.com/site/yangdingqi/home/foursquare-dataset |
| **Kaggle mirror** | `chetanism/foursquare-nyc-and-tokyo-checkin-dataset` |
| **Paper** | Yang et al., *"Participatory Cultural Mapping Based on Collective Behavior Data in Location Based Social Networks"*, ACM TIST 2015 |
| **Users** | ~18,200 |
| **POIs (venues)** | ~9,000–10,000 |
| **Check-ins** | ~600,000 |
| **Date range** | April 2012 – February 2013 (11 months) |
| **Geography** | New York City, USA |
| **Category taxonomy** | 9 top-level Foursquare categories → 7-class mapping (see table above) |
| **License** | Academic research use; cite Yang et al. TIST 2015 |
| **Usage in papers (2020–2024)** | Very high — standard benchmark in the vast majority of next-POI papers |
| **ETL status** | `src/etl/foursquare/` |
| **Output** | `data/checkins/New York City.parquet` |

**Raw file format** (tab-separated, no header):
```
user_id  venue_id  venue_category_id  venue_category_name  latitude  longitude  timezone_offset  utc_time
```
- `venue_id`: alphanumeric string (e.g. `4b058f49f964a520b04e23e3`) → integer-indexed
- `timezone_offset`: integer minutes from UTC (e.g. `-300` for EST)
- `utc_time`: `Fri Apr 03 18:00:09 +0000 2012`

**Download steps:**
1. Go to https://sites.google.com/site/yangdingqi/home/foursquare-dataset
2. Download *"NYC check-in dataset"* → `dataset_TSMC2014_NYC.txt`
3. Place at `data/raw/foursquare/dataset_TSMC2014_NYC.txt`

**Known caveats:**
- Older dataset (2012–2013); temporal patterns may not reflect current behaviour
- Single-city coverage limits geographic generalization claims
- ~0.5% of check-ins have `Event` category — dropped in ETL

---

### Foursquare Tokyo (Yang TIST2015)

| Attribute | Value |
|-----------|-------|
| **Abbreviations** | FSQ-TKY, Foursquare-Tokyo |
| **Source** | Same as NYC above |
| **Kaggle mirror** | `chetanism/foursquare-nyc-and-tokyo-checkin-dataset` |
| **Users** | ~11,900 |
| **POIs (venues)** | ~7,000–8,000 |
| **Check-ins** | ~500,000 |
| **Date range** | April 2012 – February 2013 (11 months) |
| **Geography** | Tokyo, Japan |
| **Category taxonomy** | Same 9-category mapping as NYC |
| **License** | Academic research use; cite Yang et al. TIST 2015 |
| **Usage in papers (2020–2024)** | Very high — almost always paired with FSQ-NYC |
| **ETL status** | `src/etl/foursquare/` (same pipeline as NYC, `city="tokyo"`) |
| **Output** | `data/checkins/Tokyo.parquet` |

**Raw file format:** identical to NYC (`dataset_TSMC2014_TKY.txt`).

**Download steps:**
1. Same URL as NYC
2. Download *"Tokyo check-in dataset"* → `dataset_TSMC2014_TKY.txt`
3. Place at `data/raw/foursquare/dataset_TSMC2014_TKY.txt`

**Known caveats:**
- Same temporal limitation as NYC (2012–2013)
- Slightly smaller than NYC; good for cross-city comparison
- Japanese venue names may appear in `venue_category_name` — covered by the same top-level mapping

---

### Massive-STEPS (2025)

| Attribute | Value |
|-----------|-------|
| **Abbreviations** | Massive-STEPS, STEPS |
| **Source** | GitHub — https://github.com/cruiseresearchgroup/Massive-STEPS |
| **HuggingFace** | `w11wo/Massive-STEPS-*` (per-city datasets) |
| **Paper** | *"Massive-STEPS: Massive Semantic Trajectories for Understanding POI Check-ins"*, arXiv 2505.11239 |
| **Cities** | 15: Bandung, Beijing, Istanbul, Jakarta, Kuwait City, Melbourne, Moscow, New York, Palembang, Petaling Jaya, São Paulo, Shanghai, Sydney, Tangerang, Tokyo |
| **Check-ins per city** | 40,000–300,000+ (varies significantly) |
| **POIs per city** | New York ~49,000; Jakarta ~76,000+ |
| **Date range** | 2017–2018 (24 months, more recent than legacy datasets) |
| **Geography** | 15 major cities across Asia, Europe, Americas, Oceania |
| **Category taxonomy** | Foursquare API v2 hierarchy (3 levels); subcategories resolved to top-level via `categories.csv` |
| **License** | Open-source; academic research |
| **Usage in papers (2020–2024)** | Emerging — published 2025, early adoption only |
| **ETL status** | `src/etl/massive_steps/` |
| **Output** | `data/checkins/{CityName}.parquet` per city |

**Raw file format** (per-city CSV):
```
user_id, venue_id, local_datetime, venue_category_id, venue_category_name, latitude, longitude
```

**Download steps (HuggingFace):**
```python
from huggingface_hub import hf_hub_download
# Example for New York:
path = hf_hub_download(
    repo_id="w11wo/Massive-STEPS-New-York",
    filename="checkins.csv",
    repo_type="dataset"
)
```
Or clone the full GitHub repo:
```bash
git clone https://github.com/cruiseresearchgroup/Massive-STEPS
```
Place per-city files at `data/raw/massive_steps/{city_slug}.csv` and the hierarchy at `data/raw/massive_steps/categories.csv`.

**Known caveats:**
- Very new dataset — limited published baseline comparisons as of 2025
- Long-tailed trajectory distribution: most users have fewer than 10 check-ins
- Category distributions vary widely across cities (São Paulo vs. Tokyo have very different Food/Nightlife ratios)
- No timezone offset column — requires shapefile-based localization (same approach as Gowalla)

---

### Gowalla (state-split US) — reference

| Attribute | Value |
|-----------|-------|
| **Source** | Stanford SNAP — https://snap.stanford.edu/data/loc-gowalla.html |
| **Users** | ~196,000 (full); subset per state |
| **Check-ins** | 6.4M (full); 500K–1M per state |
| **Date range** | Feb 2009 – Oct 2010 |
| **Geography** | US states — Florida, California, Texas (primary splits) |
| **Category taxonomy** | 7 super-categories (directly from Foursquare hierarchy) |
| **ETL status** | Implemented — `src/etl/gowalla/` |
| **Output** | `data/checkins/{State}.parquet` |

This is the primary dataset. State-level splits (FL, CA, TX) align with HAVANA and PGC baseline papers.

---

### Brightkite — document only

| Attribute | Value |
|-----------|-------|
| **Source** | Stanford SNAP — https://snap.stanford.edu/data/loc-brightkite.html |
| **Users** | 58,228 |
| **Check-ins** | 4,491,143 |
| **Date range** | April 2008 – October 2010 |
| **Geography** | Global (North America + Europe heavy) |
| **Category taxonomy** | Location type labels — poorly documented; no clear top-level hierarchy |
| **License** | Academic use via SNAP |

**Why document-only:** Brightkite's category annotation is sparse and undocumented at the top level — there is no published, canonical mapping from its venue types to the 7-class Foursquare taxonomy. Additionally, ~94% of accounts were trial users with minimal activity, creating severe sparsity. Usage in recent papers (2022–2024) has declined sharply in favour of FSQ-NYC/TKY + Gowalla.

---

### Foursquare Global Scale Check-in Dataset (GSCD) — document only

| Attribute | Value |
|-----------|-------|
| **Source** | Dingqi Yang (research group; limited public access) |
| **Users** | 266,909 |
| **Check-ins** | 33,278,683 |
| **Date range** | April 2012 – September 2013 |
| **Geography** | 415 cities across 77 countries |
| **Category taxonomy** | Foursquare → Schema.org mapping |
| **License** | Research sharing agreement required |

**Why document-only:** ~44% of check-ins (14M+) are flagged as erroneous in the original paper. Public access requires contacting the authors. Substantial cleaning is required before use.

---

### Yelp Open Dataset — document only

| Attribute | Value |
|-----------|-------|
| **Source** | https://www.yelp.com/dataset |
| **Businesses** | 144,000+ |
| **Users** | 1.2M+ |
| **Reviews** | 6M+ |
| **Date range** | Rolling; updated annually |
| **Geography** | Multi-city USA + some international |
| **Category taxonomy** | 80+ top-level business categories |
| **License** | Non-commercial academic use only |

**Why document-only:** Yelp is primarily a review/rating dataset. Check-in sequences are not available with per-visit timestamps in the standard release — only aggregate check-in counts per business. The non-commercial license limits research publication rights. Better suited for POI recommendation (rating prediction) than check-in sequence modelling.

---

## Comparison with Existing Baselines

The project's primary baselines (HAVANA, POI-RGNN, PGC) all use Gowalla on either global or US state splits:

| Baseline | Dataset | Macro F1 (Category) |
|----------|---------|---------------------|
| MHA+PE | Gowalla global | ~26.9% |
| POI-RGNN | Gowalla global | ~32.4% |
| PGC | Gowalla FL/CA/TX | 50.3% / 36.9% / 46.2% |
| HAVANA | Gowalla FL/CA/TX | 62.9% / 46.9% / 59.8% |

The FSQ-NYC and FSQ-TKY datasets are the **standard benchmarks in the majority of recent papers** (GETNext, KGQAN, LLM4POI, etc.) and enable direct comparison with a wider body of literature. Massive-STEPS is the natural next-generation benchmark for future-proofing claims.

---

## ETL Output Schema

All datasets produce a unified parquet at `data/checkins/{Name}.parquet`:

| Column | Type | Notes |
|--------|------|-------|
| `userid` | int64 | Original integer or index-mapped from string ID |
| `placeid` | int64 | Index-mapped from alphanumeric venue_id |
| `datetime` | datetime64[ns, UTC] | UTC timestamp |
| `latitude` | float64 | |
| `longitude` | float64 | |
| `category` | str | One of 7 super-categories |
| `venue_category_name` | str | Original fine-grained category (for debugging / future re-mapping) |
| `local_datetime` | datetime64 | Timezone-local; derived from offset field or shapefile sjoin |
| `city_name` | str | e.g. `"New York City"`, `"Tokyo"` |

The `state_name` / `country_name` fields from the Gowalla pipeline are absent here — they are metadata unused by the downstream `src/data/inputs/` pipeline.
