# Category (class) distribution — per dataset

**Date:** 2026-06-30 · **Study:** closing_data · **Status:** computed, factual

Documents how imbalanced the 7-class category task is, per dataset, for the MobiWac
paper. Read/compute only; no training.

## Source

- **File (per state):** `output/check2hgi/<state>/input/next.parquet`
- **Column read:** `next_category` (single-column read via pyarrow).
- **Why this column:** this is the **category-task label** the model actually learns
  (the task_a / CATEGORY slot in the MTL setup). It is **check-in/task-sample weighted**
  — one row per next-POI task sample, so a category's share is the fraction of task
  samples whose target POI carries that root category. This matches the requested
  "count each category over all task samples" definition and is the distribution the
  category macro-F1 is measured against.
- **Labels:** stored as the 7 Gowalla root-category name strings (no `None`/8th class).
  The 7 roots and their integer encoding are `src/configs/globals.py CATEGORIES_MAP`:
  `0 Community, 1 Entertainment, 2 Food, 3 Nightlife, 4 Outdoors, 5 Shopping, 6 Travel`
  (id 7 = `None`, not present in any of these task files).
- **Datasets:** the 5 Gowalla states (Alabama, Arizona, Florida, Texas, California)
  + Istanbul (Massive-STEPS; FSQ-v1 leaves collapsed to the same 7 Gowalla roots via
  `docs/studies/second_dataset/category_map.md`). All six come from the **same board**
  (`output/check2hgi/<state>/`), so they are computed identically and are commensurable.

> Note: the per-POI `category.parquet` (one row per place, column `category`) gives a
> *POI-weighted* distribution — a different quantity. We deliberately report the
> **check-in/task-sample weighted** `next_category` distribution because that is what the
> category head is trained and evaluated on.

## Per-state 7-category distribution (count + %)

Sorted by share within each state. `n` = number of next-POI task samples.

### Alabama — n = 12,709
| Category | Count | % |
|---|---:|---:|
| Food | 4,345 | 34.19% |
| Shopping | 3,288 | 25.87% |
| Community | 2,193 | 17.26% |
| Travel | 881 | 6.93% |
| Entertainment | 820 | 6.45% |
| Outdoors | 729 | 5.74% |
| Nightlife | 453 | 3.56% |

### Arizona — n = 26,396
| Category | Count | % |
|---|---:|---:|
| Food | 8,977 | 34.01% |
| Shopping | 6,944 | 26.31% |
| Community | 3,295 | 12.48% |
| Travel | 2,869 | 10.87% |
| Entertainment | 1,621 | 6.14% |
| Outdoors | 1,437 | 5.44% |
| Nightlife | 1,253 | 4.75% |

### Florida — n = 159,175
| Category | Count | % |
|---|---:|---:|
| Food | 39,296 | 24.69% |
| Entertainment | 34,062 | 21.40% |
| Shopping | 33,509 | 21.05% |
| Travel | 17,920 | 11.26% |
| Community | 15,723 | 9.88% |
| Outdoors | 11,617 | 7.30% |
| Nightlife | 7,048 | 4.43% |

### Texas — n = 460,976
| Category | Count | % |
|---|---:|---:|
| Food | 142,804 | 30.98% |
| Shopping | 110,090 | 23.88% |
| Community | 70,084 | 15.20% |
| Entertainment | 50,092 | 10.87% |
| Travel | 30,743 | 6.67% |
| Nightlife | 29,576 | 6.42% |
| Outdoors | 27,587 | 5.98% |

### California — n = 358,302
| Category | Count | % |
|---|---:|---:|
| Food | 117,246 | 32.72% |
| Shopping | 82,100 | 22.91% |
| Travel | 44,998 | 12.56% |
| Community | 43,216 | 12.06% |
| Entertainment | 27,034 | 7.55% |
| Outdoors | 26,421 | 7.37% |
| Nightlife | 17,287 | 4.82% |

### Istanbul — n = 58,075
| Category | Count | % |
|---|---:|---:|
| Food | 19,401 | 33.41% |
| Outdoors | 13,806 | 23.77% |
| Community | 9,743 | 16.78% |
| Shopping | 4,519 | 7.78% |
| Travel | 3,963 | 6.82% |
| Nightlife | 3,728 | 6.42% |
| Entertainment | 2,915 | 5.02% |

## Imbalance summary + macro-F1 floor

The **majority class is `Food` in every dataset.** Normalized entropy = Shannon entropy
in bits ÷ log2(7); 1.0 = uniform. Imbalance ratio = majority count ÷ minority count.

The **macro-F1 of a constant majority-class predictor over 7 classes** is
`floor = (2p / (1 + p)) / 7`, where `p` is the majority prevalence (the majority class
gets F1 = 2p/(1+p), the other 6 get F1 = 0; macro-averaged over 7).

| State | Majority | p (prevalence) | Norm. entropy | Imbalance (max/min) | Macro-F1 floor |
|---|---|---:|---:|---:|---:|
| Alabama | Food | 0.3419 | 0.855 | 9.59 | **7.28%** |
| Arizona | Food | 0.3401 | 0.870 | 7.16 | **7.25%** |
| Florida | Food | 0.2469 | 0.929 | 5.58 | **5.66%** |
| Texas | Food | 0.3098 | 0.903 | 5.18 | **6.76%** |
| California | Food | 0.3272 | 0.900 | 6.78 | **7.04%** |
| Istanbul | Food | 0.3341 | 0.882 | 6.66 | **7.15%** |

- **Gowalla floors:** min 5.66% (FL), max 7.28% (AL); **mean 6.80%**. Four of the five
  states sit at 6.76–7.28%; Florida is the single low outlier (most balanced majority,
  p = 0.247) at 5.66%.
- **Istanbul floor:** 7.15%.
- **All-6 mean floor:** 6.86%.

### Verdict: "about 7%" vs "about 6%"

**"About 7%" is the better single figure.** Five of the six datasets (AL 7.28, AZ 7.25,
CA 7.04, IST 7.15, TX 6.76) round to ~7%; the Gowalla mean (6.80%) and all-6 mean (6.86%)
both round to ~7%. Only Florida (5.66%) is closer to 6%. The paper's §6.2 wording
"a majority-class floor of about 7%" is accurate for the cross-state picture; the one
caveat worth noting is that the most balanced state (Florida) has a floor near 5.7%.
A defensible alternative is "≈6–7%" to bracket the Florida outlier.

## Fold stratification (confirmed in code)

`src/data/folds.py`: folds are **user-disjoint stratified 5-fold CV via
`StratifiedGroupKFold`** (`n_splits=5, shuffle=True, random_state=seed`), **stratified on
the `next_category` label and grouped by `userid`** (e.g. lines 975–977 for single-task
NEXT; 1062–1069 for the POI-protocol category step; 1404–1416 for the MTL joint split).
So both tasks' folds preserve the category-label distribution above while keeping each
user wholly within one fold.

## Reproduce

```python
import pyarrow.parquet as pq
col = pq.read_table("output/check2hgi/<state>/input/next.parquet",
                    columns=['next_category']).column(0).to_pylist()
# then value_counts; floor = (2*p/(1+p))/7 with p = majority share
```
