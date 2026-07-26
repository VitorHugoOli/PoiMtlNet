# Dataset licensing: findings note (REV-026, ethics gap)

**Prepared** 2026-07-25, references track. **Scope:** evidence only. No dissertation prose was
written and no chapter was edited for this item. Every line below was opened this session; where
a source could not be opened, the line says so.

**What this note is for.** The dissertation currently renders zero sentences on data licensing.
Before any such sentence is written, the two facts below have to be settled, and one of them
carries a discrepancy the author should rule on first.

---

## 1 · Gowalla (Chapters 3, 4, 5)

### 1.1 The source the Chapter 5 pipeline actually consumes

The hidden comment at `src/chapters/5_mobiwac.tex:299-303` records the provenance. It reads, in
part, that the Gowalla source is "the public CC0 category-annotated dump,
figshare.com/articles/dataset/gowalla_data/22126586 (gowalla_checkins + gowalla_spots_subset1/2,
the exact files the ETL consumes; 36,001,959 check-ins)", that the date range was measured on the
parquet on 2026-07-09 as 2009-01-21 to 2011-08-16, and that "The SNAP/cho2011 dump (Feb 2009-Oct
2010) is NOT the data source; cho2011gowalla is cited as the LBSN reference only." The same
Figshare URL is already printed to the reader in the Chapter 5 footnote at
`src/chapters/5_mobiwac.tex:55`.

The comment's file claim checks out against the code: `src/etl/gowalla/main.py:22-24` names
`gowalla_checkins.parquet`, `gowalla_spots_subset1.csv` and `gowalla_spots_subset2.csv`.

### 1.2 What the Figshare record says (VERIFIED)

| Field | Value |
|---|---|
| Record | Gowalla_data, https://figshare.com/articles/dataset/gowalla_data/22126586 |
| DOI | 10.6084/m9.figshare.22126586.v2 |
| Licence | **CC0**, identifier `CC0`, URL https://creativecommons.org/publicdomain/zero/1.0/ |
| Depositor | "Yang" (single name; no ORCID on the record) |
| Published | 2023-02-20 |
| Files | `gowalla_checkins.csv`, `gowalla_spots_subset1.csv`, `gowalla_spots_subset2.csv`, `gowalla_category_structure.json`, `gowalla_friendship.csv`, `gowalla_userinfo.csv` |
| Record's own description | states the data was "downloaded from http://www.yongliu.org/datasets/" and reports 36,001,959 check-ins by 319,063 users over 2,844,076 locations |

Opened at https://api.figshare.com/v2/articles/22126586 (Figshare public API, the record of
record for its own deposits). The CC0 label and the check-in count in the LaTeX comment are
therefore both **confirmed**, and the three consumed filenames appear verbatim in the deposit.

### 1.3 What remains unverifiable, and why it matters

The CC0 label is an assertion made by a **third-party depositor**, not by the party that
collected the data. The record itself says the files were downloaded from another site, and that
upstream site (`www.yongliu.org/datasets/`) returned HTTP 403 to this session, so **its terms
could not be read**. The depositor is identified only as "Yang". Nothing on the Figshare record
establishes that the depositor held the rights to place the data under CC0.

The honest statement available today: *the copy consumed is distributed on Figshare under CC0
(DOI 10.6084/m9.figshare.22126586.v2), a licence applied by the depositor of that copy.* Anything
stronger, in particular any claim that the Gowalla data as such is CC0, is not supported by what
could be opened.

### 1.4 The SNAP discrepancy (REPORTED, not resolved)

`docs/context/DATASETS.md:187-199` records the Gowalla source as **Stanford SNAP**
(https://snap.stanford.edu/data/loc-gowalla.html) and carries **no License row at all**, while
the Chapter 5 ETL consumes the Figshare dump. These are two different artifacts, not two links
to one artifact:

| | SNAP `loc-gowalla` | Figshare 22126586 |
|---|---|---|
| Files | `loc-gowalla_edges.txt.gz`, `loc-gowalla_totalCheckins.txt.gz` | `gowalla_checkins.csv`, `gowalla_spots_subset1/2.csv`, + 3 more |
| Check-ins | 6,442,890 | 36,001,959 (per the record's description) |
| Period stated | Feb 2009 to Oct 2010 | (not stated on the record; the repo measured 2009-01-21 to 2011-08-16 on the parquet) |
| Categories | none | 7 top-level categories (the category annotation the work depends on) |
| Licence | **none stated on the page** (searched for licence, terms, copyright: absent) | CC0 |

SNAP page opened 2026-07-25 at https://snap.stanford.edu/data/loc-gowalla.html. It publishes no
licence or terms-of-use statement of any kind.

Two consequences for the author to rule on:

1. `docs/context/DATASETS.md` documents a source the Chapter 5 pipeline does not read. The
   Chapter 5 comment already flags this ("the SNAP/cho2011 dump is NOT the data source"), so the
   defect is in DATASETS.md, not in the chapter. It is a repo-documentation fix, outside this
   track's edit scope.
2. The bibliography cites `cho2011gowalla` and `jure2014snap` for Gowalla. Per the Chapter 5
   comment those are the LBSN and SNAP references, not the provenance of the consumed files. Any
   licensing sentence must therefore name the Figshare deposit, not SNAP, or it will describe the
   wrong artifact.

---

## 2 · Massive-STEPS / Istanbul (Chapter 5)

### 2.1 What the repo currently records

`docs/context/DATASETS.md:153` gives the License row as "Open-source; academic research". That is
a characterization, not an identified licence: it names no licence, no version, and no URL.

### 2.2 What the dataset records say (VERIFIED)

| Where | Identifier | Licence |
|---|---|---|
| Hugging Face dataset (the distribution) | https://huggingface.co/datasets/CRUISEResearchGroup/Massive-STEPS-Istanbul | **Apache-2.0** (`license: apache-2.0` in the dataset card front matter; repository tag `license:apache-2.0`) |
| GitHub repository | https://github.com/cruiseresearchgroup/Massive-STEPS | **Apache-2.0** (SPDX `Apache-2.0`, `LICENSE` at repository root) |
| Paper | arXiv:2505.11239, Wongso, Xue, Salim | (no licence claim in the record) |

Opened via the Hugging Face datasets API, the raw dataset card, and the GitHub licence API. Note
that `docs/context/DATASETS.md:145` points at `w11wo/Massive-STEPS-*`; that namespace now
redirects to `CRUISEResearchGroup/Massive-STEPS-Istanbul`, which is the same record.

So the licence **is** identifiable, and "Apache-2.0" replaces the current characterization. Two
qualifications belong with it:

1. Apache-2.0 is a software licence applied here to a data release. That is what the distributors
   published; it is not this note's place to interpret its effect on data.
2. Massive-STEPS is derived material. Its README states the dataset "is derived from the
   [Semantic Trails Dataset] and [Foursquare Open Source Places]". Both upstreams were checked:
   Semantic Trails is on Figshare under **CC0** (DOI 10.6084/m9.figshare.7429076.v2), and
   Foursquare Open Source Places is distributed under **apache-2.0** on Hugging Face. The chain
   is therefore consistent, with no upstream term more restrictive than the Massive-STEPS
   licence surfacing in what could be opened.

### 2.3 What remains unverifiable

The Foursquare Open Source Places **product terms** (as distinct from the Hugging Face licence
tag) were not opened; only the Hugging Face record's tag was read. If a licensing sentence needs
to speak to the Foursquare terms specifically, that check is still outstanding.

---

## 3 · Summary for the author

| Dataset | Source actually consumed | Licence verified this session | Identifier / URL | Unverifiable |
|---|---|---|---|---|
| Gowalla (Ch. 3/4/5) | Figshare deposit `gowalla_data` | CC0, applied by the depositor of that copy | 10.6084/m9.figshare.22126586.v2 ; creativecommons.org/publicdomain/zero/1.0/ | Whether the depositor held the rights to apply it; the upstream terms at yongliu.org (403 to this session) |
| Gowalla as documented in DATASETS.md | Stanford SNAP `loc-gowalla` (NOT what the ETL reads) | none: the page states no licence | snap.stanford.edu/data/loc-gowalla.html | n/a; the absence is the finding |
| Massive-STEPS / Istanbul (Ch. 5) | Hugging Face `CRUISEResearchGroup/Massive-STEPS-Istanbul` | Apache-2.0 | huggingface.co/datasets/CRUISEResearchGroup/Massive-STEPS-Istanbul ; github.com/cruiseresearchgroup/Massive-STEPS | Foursquare Open Source Places product terms (only the HF licence tag was read) |

**Open decisions this note does not take:**

1. Whether to correct `docs/context/DATASETS.md` (Gowalla source row plus a missing License row;
   Massive-STEPS License row "Open-source; academic research" to Apache-2.0). Outside this
   track's edit scope.
2. Where a licensing sentence should live (Chapter 5 setup, or a general ethics or data statement
   in the frame), and how it should hedge the Gowalla case. No such prose was drafted.

**Machine-readable evidence:** `src_utils/item4_licence_evidence.json` (same session, same URLs).
