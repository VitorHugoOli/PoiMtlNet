# STEP 3c — Reviewer-pass report (existence + claim-support on 100% of new entries)

_Two independent checks on all 24 new references: (1) the agent's own STEP-1 verification (source ledger, below), and (2) a fresh-eyes reviewer (separate context, actor-critic critic role) that re-resolved every identifier from scratch and re-graded every attributed claim. This satisfies the fail-closed review gate (AGENT_GUARDRAILS: >=20% sample; here 100% of new entries)._

## Headline verdict (independent reviewer)

- **Existence: 24 / 24 confirmed.** Every identifier re-resolved to the correct work. **Zero fabricated, zero misresolved DOIs, zero wrong-author/wrong-venue, zero fabricated DOIs.**
- **Claim support: 15 SUPPORTED · 5 PLAUSIBLE-UNCONFIRMED · 4 existence-only (no attributed sentence) · 0 UNSUPPORTED.**
- **No claim overreached or contradicted its source.** The 5 PLAUSIBLE-UNCONFIRMED are all works whose abstract is license-gated on OpenAlex and was unreadable through every open route this session (Unpaywall / Semantic Scholar / PMC / publisher); their titles and venues are consistent with the attributed claim but the exact wording was not confirmed against an opened abstract. These carry [VERIFY] flags, not fabricated support.

## [VERIFY] flags — author must confirm before binding these sentences

| Key | Why flagged | Action taken this session |
|---|---|---|
| `wilcoxon1945` | Origin of the signed-rank test; the paired-comparison wording not confirmed vs opened text (abstract null). Textbook-standard. | Left as-is; note in map that 'across folds' is the dissertation's application, not the 1945 source. |
| `kohavi1995crossval` | 'Stratified k-fold recommended' is the paper's known conclusion but not confirmed vs opened text; **BibTeX had no resolvable identifier**. | Added Zenodo re-deposit DOI `10.5281/zenodo.19712698` with an honest provenance note in the .bib. |
| `sokolova2009measures` | Macro-F1 averaging definition consistent with the title but not confirmed vs opened abstract. | Left as-is; author confirms the macro-averaging definition on the page before binding. |
| `pedregosa2011sklearn` | **Attribution nuance:** StratifiedGroupKFold was added ~v0.24/2020, not described in the cited 2011 paper. | **Fixed:** section-map row now cites the 2011 paper as library foundation + the scikit-learn API docs for the specific splitter; .bib comment updated. |
| `yang2015tsmc` | The 'origin of the Foursquare NYC/Tokyo benchmark' statement is widely known but not confirmed vs opened abstract. | Left as-is; author confirms dataset provenance from the paper's data-release page. |

## Attribute notes (not errors; caught by the reviewer, worth a glance)

- `perozzi2014deepwalk`, `leskovec2016snap`: OpenAlex truncates the title field ('DeepWalk', 'SNAP'); full titles are correct in the .bib.
- `belghazi2018mine`: arXiv title prefixes 'MINE:'; secondary-author order (Hjelm/Courville) matches the published ICML order, not the arXiv metadata.
- `kipf2017gcn`, `hjelm2019dim`, `lin2022rlw`: BibTeX year is the publication venue year (ICLR 2017 / ICLR 2019 / TMLR 2022), correctly distinct from the earlier arXiv-submission year.
- `perozzi2014deepwalk`: the arXiv preprint (1403.6652) carries the identical KDD DOI — independent cross-confirmation of the identifier.
- `kendall2018uncertainty`: CVPR 2018 record confirmed via OpenAlex get_work (pp. 7482-7491, 2850 cites); preprint arXiv:1705.07115. Both identifiers now shown in the .bib comment.

## Integrity fixes in the EXISTING project bibs (from STEP 2, re-stated here for the audit trail)

These are corrections to works already cited by the three papers, surfaced during verification:

- `misra2016cross` (Cross-Stitch): listed DOI resolves to a different paper -> correct is **`10.1109/CVPR.2016.433`**.
- `zhang2021survey` (Zhang & Yang MTL survey): listed DOI does not resolve -> correct is **`10.1109/TKDE.2021.3070203`**.
- `church2017word2vec`: resolves to a commentary, not the skip-gram method -> cite **`mikolov2013word2vec`** for the method.
- `yu2019mmoe`: no distinct resolvable work; likely a duplicate of `ma2018mmoe` -> **author confirm or drop**.
- DGI triple-key and Nash-MTL double-key: **consolidate** before compile (one contains a slash).

---


_One row per new reference: BibTeX key -> identifier -> where the record was opened (STEP 1) -> the claim it supports. This is the agent's own ledger; the independent reviewer_report.md re-checks it. `_theme` B/A/C/D._

| Key | Theme | Identifier | Record opened (this session) | Claim it supports |
|---|---|---|---|---|
| `hochreiter1997lstm` | A | DOI 10.1162/neco.1997.9.8.1735 | Publisher abstract (returned by fulltext fetch service) | Long Short-Term Memory (LSTM) enables recurrent networks to learn dependencies over very long time lags by enforcing con |
| `kong2018hstlstm` | A | DOI 10.24963/ijcai.2018/324 | Full text PDF (ijcai.org/proceedings/2018/0324.pdf) ope | Location prediction in a weak-real-time setting (next minutes/hours) is addressed by combining spatial-temporal influenc |
| `lian2020geosan` | A | DOI 10.1145/3394486.3403252 | Publisher abstract (returned by fulltext fetch service) | Sequential location recommendation should make effective use of geographical information; GeoSAN uses a self-attention n |
| `mikolov2013word2vec` | B | arXiv:1301.3781 | arXiv abstract | Two efficient neural architectures (CBOW and skip-gram) can learn high-quality continuous vector representations of word |
| `perozzi2014deepwalk` | B | DOI 10.1145/2623330.2623732 | arXiv abstract | DeepWalk learns latent vertex representations by applying language-modeling (skip-gram) techniques to sequences of verti |
| `kipf2017gcn` | B | arXiv:1609.02907 | arXiv abstract | A scalable graph convolutional network, motivated by a localized first-order approximation of spectral graph convolution |
| `hamilton2017graphsage` | B | arXiv:1706.02216 | arXiv abstract | GraphSAGE is an inductive framework that generates node embeddings by learning aggregator functions over a node's local  |
| `belghazi2018mine` | B | arXiv:1801.04062 | arXiv abstract | Mutual information between high-dimensional continuous variables can be estimated by gradient descent over neural networ |
| `hjelm2019dim` | B | arXiv:1808.06670 | arXiv abstract | Deep InfoMax learns unsupervised representations by maximizing mutual information between an input and an encoder's outp |
| `ruder2017mtloverview` | C | arXiv:1706.05098 | arXiv abstract | This survey organizes deep MTL around two mechanisms, hard and soft parameter sharing, and offers guidance on when auxil |
| `tang2020ple` | C | DOI 10.1145/3383313.3412236 | OpenAlex record + abstract (Crossref/Semantic Scholar v | Progressive Layered Extraction (and its Customized Gate Control building block) explicitly separates shared from task-sp |
| `hazimeh2021dselectk` | C | arXiv:2106.03760 | arXiv abstract | DSelect-k is a differentiable, sparse expert-selection gate for mixture-of-experts that lets a model choose exactly k ex |
| `kendall2018uncertainty` | C | arXiv:1705.07115 | arXiv abstract | Task losses can be weighted automatically using each task's homoscedastic (task-dependent) uncertainty, removing the nee |
| `liu2021cagrad` | C | arXiv:2110.14048 | arXiv abstract | Conflict-Averse Gradient descent seeks an update direction that minimizes conflict by staying close to the average gradi |
| `lin2022rlw` | C | arXiv:2111.10603 | arXiv abstract | Training an MTL model with randomly sampled loss/gradient weights (Random Weighting) is a simple baseline that the paper |
| `he2016lbpr` | C | DOI 10.1609/aaai.v30i1.9994 | OpenAlex record + downloaded PDF (articles/10.1609_aaai | Next-POI recommendation is modeled by fusing a personalized Markov chain with users' latent behavior patterns via a thir |
| `yang2015tsmc` | D | DOI 10.1109/tsmc.2014.2327053 | OpenAlex record (title/authors/venue/year/DOI confirmed | User activity preference can be modeled by leveraging user spatial-temporal check-in behaviour in location-based social  |
| `yang2016cultural` | D | DOI 10.1145/2814575 | OpenAlex record (title/authors/venue/year/DOI confirmed | Collective check-in behaviour data in LBSNs supports participatory cultural mapping |
| `leskovec2016snap` | D | DOI 10.1145/2898361 | OpenAlex record (title/authors/venue/year/DOI confirmed | SNAP is a general-purpose, high-performance system for analysis and manipulation of large networks |
| `sokolova2009measures` | D | DOI 10.1016/j.ipm.2009.03.002 | OpenAlex record (title/authors/venue/year/DOI confirmed | Classification performance measures (including precision, recall, F-measure, and their macro/micro averages) have distin |
| `gambs2012mmc` | D | DOI 10.1145/2181196.2181199 | OpenAlex record (title/authors/year/DOI confirmed; abst | The next location a user visits can be predicted using mobility Markov chains built from their movement history |
| `kohavi1995crossval` | D | DOI 10.5281/zenodo.19712698 | OpenAlex record + readable abstract | Cross-validation and bootstrap are compared for accuracy estimation and model selection; stratified k-fold is recommende |
| `pedregosa2011sklearn` | D | DOI 10.5555/1953048.2078195 | OpenAlex record (JMLR) + arXiv 1201.0490 abstract (read | scikit-learn integrates a wide range of machine-learning algorithms behind a consistent Python API aimed at non-speciali |
| `wilcoxon1945` | D | DOI 10.2307/3001968 | OpenAlex record (title/authors/venue/year/DOI confirmed | Paired samples can be compared by ranking their differences rather than assuming normality |