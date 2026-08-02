# ERRATA — CBIC 2025 (this folder's article)

> **Article:** "An Investigation into Multi-Task Learning for Point-of-Interest Category
> Classification and Next-POI Prediction." CBIC 2025 (Congresso Brasileiro de Inteligencia
> Computacional), DOI 10.21528/CBIC2025-1191324. Published — the record of the published paper
> is fixed and is NOT edited here.
>
> **Purpose of this file.** These are defects known in the published article. Per the dissertation
> errata policy (NORTH_STAR §4, decision #7), they are corrected **silently in the re-typeset
> dissertation chapter** (Ch.3) and listed in the dissertation's Appendix B. This file is the
> living record for this article folder; add future points as they are found.

## Citation errata (fix in the dissertation bib during adaptation)

| # | Entry / place | Defect | Fix (verified this session unless noted) |
|---|---|---|---|
| 1 | POI-RGNN citation (basis.tex) | Cites the wrong paper for the Capanema POI-RGNN model. | Use `capanema2023poirgnn` (the correct Capanema POI-RGNN paper). **[VERIFY at adaptation]**: the exact record could not be resolved via OpenAlex this session (likely an SBC/SBRC venue); confirm authors/venue/year/DOI from the source of record before inserting. |
| 2 | `chen2020modeling` (HMRM) | Author names garbled and given/family swapped: "Chen, Min and Zhao, Yanmin and Liu, Yanchi and Yu, Xun and Zheng, Kai". Also typed `@inproceedings` with a journal in `booktitle`; DOI missing. | Authors → **Meng Chen, Yan Zhao, Yang Liu, Xiaohui Yu, Kai Zheng**. Type → `@article`, `journal = {IEEE Transactions on Knowledge and Data Engineering}`, `volume=34, number=10, pages=4829--4841, year=2020`. DOI → **10.1109/tkde.2020.3001025**. (Title "Modeling Spatial Trajectories with Attribute Representation Learning" is correct.) Verified against OpenAlex get_work this session. |
| 3 | `velivckovic2017graph` (GAT) | Cited as an arXiv preprint (arXiv:1710.10903, 2017). | Cite the **ICLR 2018** version of record: `booktitle = {Proc. ICLR}, year = {2018}` (arXiv id may stay as a note). |
| 4 | `misra2016cross` (Cross-Stitch) | DOI resolves to a different paper. | DOI → **10.1109/CVPR.2016.433** (verified). |
| 5 | `zhang2021survey` (Zhang & Yang MTL survey) | Listed DOI does not resolve. | DOI → **10.1109/TKDE.2021.3070203** (verified). |
| 6 | `church2017word2vec` | Resolves to a commentary column, not the skip-gram method. | For the word2vec/skip-gram method cite `mikolov2013word2vec` (arXiv:1301.3781). |
| 7 | `yu2019mmoe` | No distinct resolvable 2019 work; likely a duplicate of `ma2018mmoe`. | **Confirm the intended work or drop** the key. |
| 8 | DGI key variants | The same paper appears under `velickovic2019dgi` / `velickovic2019deep` / `velivckovic2018deep`. | Consolidate to a single key in the global dissertation bib. |
| 9 | Nash-MTL key | One key spelling contains a slash. | Consolidate; remove the slash before compiling. |

## Non-citation errata (already recorded in NORTH_STAR §4 — repeated here for this folder)

- Unfilled dataset placeholders in `results.tex` (`N_users`, `N_poi`, `N_checkins`). Recompute via a
  repo-committed script over the CBIC-era Florida pipeline (author-approved); CoUrb's published FL row is a
  cross-check only, not a source.
- Prose "almost four times more wall time" vs table 80.88 s / 34.97 s = 2.3x — reconcile to the table.
- Prose MFLOPs "roughly double" contradicts the table — reconcile to the table.
- Broken cross-ref label `sec:method:single_task_heads` on the Dataset subsection.
- Typo "spatio-tegm mporal" in `basis.tex`.
- Claim discipline: Nash-MTL "consistently better" predates the solver-bug discovery; do not amplify. "MTL does
  not help" is time-indexed (present as the conclusion of the time, for that configuration).
