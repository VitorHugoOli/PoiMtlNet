# STEP 2 — Gap analysis: foundational set vs. already-cited

_Diff of the STEP-1 verified foundational set against the STEP-0 already-cited ground truth (92 distinct works + 24 frontier). Three outputs: (1) seminal works the fundamentals chapter should cite but no paper cites yet; (2) recent works that keep the chapter current in 2026; (3) integrity fixes the verification pass caught in the existing bibliographies. All new works are DOI/arXiv-verified and carried in `new_references_ch2.bib`._

## 2.1  Summary

| Theme | Already-cited (distinct) | New verified | Coverage verdict |
|---|---|---|---|
| A · POI-prediction tasks & sequence models | 40 | 3 | Strong. Task formulations and next-place lineage well covered; only classic anchors missing. |
| B · Representations for mobility | 18 | 6 | Spine complete after adding the graph-rep foundations and the MI-maximization bridge. |
| C · Multi-task learning | 26 | 7 | Strong after filling four documented gaps; two DOI errors to fix. |
| D · Datasets, metrics & evaluation | 8 | 8 | Was the thinnest theme; now anchored. Some metrics remain textbook-cited. |

## 2.2  Seminal works the chapter should add (missing anchors)

These are canonical works a computing banca expects to see defined, absent from all three paper bibs:

| Theme | Work | Why it is needed | Identifier |
|---|---|---|---|
| B | GCN (Kipf & Welling 2017) | The graph-convolution primitive HGI's encoder is built on; 2.2 defines it before DGI. | arXiv:1609.02907 |
| B | GraphSAGE (Hamilton et al. 2017) | Inductive neighbor-aggregation; situates why place graphs generalize to unseen nodes. | arXiv:1706.02216 |
| B | DeepWalk (Perozzi et al. 2014) | The random-walk ancestor of node2vec; completes the graph-embedding genealogy. | DOI 10.1145/2623330.2623732 |
| B | word2vec / skip-gram (Mikolov et al. 2013) | The actual skip-gram source (the cited `church2017word2vec` is a commentary, not the method). | arXiv:1301.3781 |
| B | MINE (Belghazi et al. 2018) | Neural mutual-information estimation: the objective DGI/HGI/Check2HGI maximize. | arXiv:1801.04062 |
| B | Deep InfoMax / DIM (Hjelm et al. 2019) | The conceptual bridge from MINE to DGI (local/global MI in representations). | arXiv:1808.06670 |
| A | LSTM (Hochreiter & Schmidhuber 1997) | The recurrent unit under ST-RNN/DeepMove/Flashback/HST-LSTM; 2.1 needs the primitive. | DOI 10.1162/neco.1997.9.8.1735 |
| C | Uncertainty weighting (Kendall & Gal 2018) | Canonical loss balancer; documented gap. Cite the CVPR 2018 version of record (preprint arXiv:1705.07115). | DOI 10.1109/CVPR.2018.00781 |
| C | PLE / CGC (Tang et al. 2020) | The structured-sharing model the joint model's topology descends from; documented gap. | DOI 10.1145/3383313.3412236 |
| C | CAGrad (Liu et al. 2021) | Named in the balancer set (2.3) with no bib entry; documented gap. | arXiv:2110.14048 |
| C | DSelect-k (Hazimeh et al. 2021) | Differentiable expert selection; completes the MMoE->PLE->DSelect-k routing arc; documented gap. | arXiv:2106.03760 |
| D | Gowalla origin — Cho et al. 2011 | Dataset of record; cite the KDD origin (already cited, DOI now confirmed 10.1145/2020408.2020579). | DOI 10.1145/2020408.2020579 |
| D | Foursquare TSMC2014 (Yang et al. 2015) | The NYC/Tokyo check-in benchmark of record (distinct from the Cultural-Mapping paper). | DOI 10.1109/TSMC.2014.2327053 |
| D | Holm 1979 / Lakens 2017 TOST | Correction + non-inferiority tests the papers already use but did not resolve a DOI for. | 10.2307/4615733 / 10.1177/1948550617697177 |
| D | macro-F1 anchor (Sokolova & Lapalme 2009) | A citable source for macro-averaged F1, the primary category metric. | DOI 10.1016/j.ipm.2009.03.002 |

## 2.3  Recent works that keep the chapter current (2020-2026)

Beyond the classics, these recent verified works were confirmed. Most are already cited (the three papers are current); the frontier survey (`new_references.bib`, 24 entries) supplies the rest. New additions here:

- **GeoSAN (Lian et al. 2020, KDD)** — geography-aware self-attention for next-place; rounds out the attention lineage (STAN/GETNext already cited).
- **RLW (Lin et al. 2022, TMLR)** — random-weighting litmus test; strengthens the honest claim that adaptive balancers rarely beat tuned fixed weights.
- **Massive-STEPS (2025, arXiv:2505.11239)** — already cited; the Istanbul benchmark, kept as preprint-status.
- The frontier set already carries the 2023-2026 hypergraph / contrastive / LLM-for-POI / region-representation state of the art (do not re-discover; three keys need renaming first).

## 2.4  Integrity fixes caught during verification (act before citing)

The fail-closed pass surfaced errors in the **existing** bibliographies. These are corrections, not new references:

| Key (existing) | Problem | Fix |
|---|---|---|
| `misra2016cross` (Cross-Stitch) | Listed DOI `10.1109/cvpr.2016.434` resolves to a *different* paper (Song et al., Lifted Structured Feature Embedding). | Correct DOI is **`10.1109/CVPR.2016.433`**. |
| `zhang2021survey` (Zhang & Yang MTL survey) | Listed DOI `10.1109/tkde.2021.3072953` does **not resolve**. | Correct TKDE DOI is **`10.1109/TKDE.2021.3070203`**. |
| `church2017word2vec` | Resolves to Church's *commentary* column "Word2Vec" (Nat. Lang. Eng.), not the method. | Cite **`mikolov2013word2vec`** (arXiv:1301.3781) for the skip-gram method; keep Church only if the commentary is actually wanted. |
| `yu2019mmoe` | No distinct 2019 "Multi-Gate MoE for Recommendation" work with a resolvable ID; duplicates `ma2018mmoe` (MMoE, KDD 2018). | **Author confirm or drop** — likely a phantom/duplicate key. |
| `velickovic2019dgi` / `velickovic2019deep` / `velivckovic2018deep` | Three keys, one paper (DGI). | Consolidate to one key. |
| `nash` / `navon2022nashmtl` | Duplicate keys for Nash-MTL; one contains a slash. | Consolidate; clean the slash before BibTeX compile. |
| Foursquare origin | Briefing conflated TSMC2014 (dataset origin, IEEE TSMC) with "Participatory Cultural Mapping" (ACM TIST 2016). | Cite **`yang2015tsmc`** as the NYC/Tokyo benchmark; keep the two distinct. |
| `ruder2017sluice` | Same work as "Latent Multi-Task Architecture Learning" (AAAI 2019); year/title drift. | Note the rename; confirm intended venue/year. |

## 2.5  Thin-coverage flags

- **No prior MTL-mobility work predicts next-region as an *end target*.** The cascade line (ye2013 -> LBPR -> CatDM -> CSLSL) treats category/activity as an *intermediate* toward next *place*; the parallel line (MCARNN, iMTL) pairs category with next *place*. This is a genuine literature gap, not a search failure, and it directly supports the dissertation's novelty (joint next-category + next-region). HAMTL (Wang 2025) is the only possible exception, but its target could not be confirmed (gated abstract) and is [VERIFY]-flagged.
- **Acc@k / top-k accuracy** has no single origin DOI; it is ranking/recsys convention. Cite conventionally or via a specific next-POI paper, do not invent an origin.
- **Majority-class baseline** is textbook; no citable origin.
- **HGI abstract is unreadable** (closed access, no OA copy anywhere). Its claim is title/venue-grounded only. Because HGI is the direct base of Check2HGI, obtaining the publisher PDF to lock the claim is the single most valuable follow-up. [VERIFY]
- **2024-2026 dynamic/contextual POI-representation contrasts** to "per-place vectors are static across visits" were out of scope for this batch; a dedicated search would further strengthen 2.2 if desired.