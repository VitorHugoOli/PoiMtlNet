# 11_claims_appendices.md — citation claim-support audit, Appendices A to E

**Unit:** `src/chapters/apx_a..apx_e`  
**Run:** 2026-07-28, round 6, as the per-chapter pass the author asked for by name (COD-008 decision).  
**Errata regime for this unit:** author's own text; Appendix B is the errata register itself.

## 1 · Counts (every citation in the unit, not a sample)

- `\cite` commands in the unit: **7**, on **7** source lines, carrying **9** key instances (a multi-key `\cite` counts once per key). Every one was audited.
- Distinct bibliography keys used: **9**.
- Verdicts: **SUPPORTED** 9.

Comments were stripped before counting, so a key that appears only inside a `%` comment is not counted; every counted site renders.

## 2 · Every citation, with its verdict

Verdict scale: SUPPORTED, the citing sentence's attribution is present in or a fair paraphrase of the
source; PARTIAL, part is supported and part is not, or the sentence is stronger than the source;
NOT-SUPPORTED, the attribution is absent from or contradicted by the source; UNVERIFIABLE, the source
of record does not carry enough to decide and the attribution is not implausible.

| # | Site (file:line) | Key | Verdict | Evidence quoted from the source (under 20 words) |
|---|---|---|---|---|
| 1 | `apx_b_errata.tex:220` | `silva2025mtlnet` | SUPPORTED | "focusing on two complementary tasks: POI Category Classification and Next-POI Prediction" |
| 2 | `apx_d_ceiling.tex:55` | `kohavi1995crossval` | SUPPORTED | "the best method to use for model selection is ten-fold stratified cross validation" |
| 3 | `apx_d_ceiling.tex:55` | `pedregosa2011sklearn` | SUPPORTED | "Scikit-learn is a Python module integrating a wide range of state-of-the-art machine learning algorithms" |
| 4 | `apx_d_ceiling.tex:56` | `sokolova2009measures` | SUPPORTED | "the measure invariance taxonomy with respect to all relevant label distribution changes" |
| 5 | `apx_e_ethics.tex:36` | `cho2011gowalla` | SUPPORTED | "data from two online location-based social networks" |
| 6 | `apx_e_ethics.tex:36` | `jure2014snap` | SUPPORTED | "A collection of more than 50 large network datasets" |
| 7 | `apx_e_ethics.tex:40` | `wongso2025massivesteps` | SUPPORTED | "large-scale, publicly available benchmark dataset ... spans 15 geographically and culturally diverse cities" |
| 8 | `apx_e_ethics.tex:61` | `luca2021mobilitysurvey` | SUPPORTED | "deep learning ... human mobility" |
| 9 | `apx_e_ethics.tex:91` | `santos2024urban` | SUPPORTED | "it is fundamental to anonymize the locations with an appropriate method" |
## 3 · Failures and partials in this unit, in detail

None. Every citation in this unit is SUPPORTED.

## 4 · Source ledger for this unit

Every distinct key cited in this unit, the identifier it resolved by, and where I opened it this session.

| Key | Identifier | Opened at | Record as returned |
|---|---|---|---|
| `cho2011gowalla` | DOI 10.1145/2020408.2020579 | Crossref REST; OpenAlex API | Friendship and mobility \| Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining \| 2011 \| type proceedings-article |
| `jure2014snap` | no identifier in the bib entry | OpenAlex API | {SNAP Datasets}: {Stanford} Large Network Dataset Collection \| (no venue in record) \| 2014 \| type article |
| `kohavi1995crossval` | no identifier in the bib entry | OpenAlex API | A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection \| (no venue in record) \| 1995 \| type article |
| `luca2021mobilitysurvey` | DOI 10.1145/3485125 | Crossref REST; OpenAlex API | A Survey on Deep Learning for Human Mobility \| ACM Computing Surveys \| 2021 \| type journal-article |
| `pedregosa2011sklearn` | no identifier in the bib entry | arXiv API; OpenAlex API; PDF in repo: Pedregosa2011_ScikitLearn.pdf | Scikit-learn: Machine Learning in Python \| Journal of Machine Learning Research (2011) \| 2012 \| type posted-content |
| `santos2024urban` | no identifier in the bib entry | NOT RESOLVED at any source of record | None \| (no venue in record) \| None \| type None |
| `silva2025mtlnet` | DOI 10.21528/CBIC2025-1191324 | Crossref REST; OpenAlex API | An Investigation into Multi-Task Learning for Point-of-Interest Category Classification and Next-POI Prediction \| Anais do XVII Congresso Brasileiro de Inteligência Comp |
| `sokolova2009measures` | DOI 10.1016/j.ipm.2009.03.002 | Crossref REST; OpenAlex API; PDF in repo: sokolova2009.pdf | A systematic analysis of performance measures for classification tasks \| Information Processing &amp; Management \| 2009 \| type journal-article |
| `wongso2025massivesteps` | no identifier in the bib entry | arXiv API; OpenAlex API | Massive-STEPS: Massive Semantic Trajectories for Understanding POI Check-ins -- Dataset and Benchmarks \| arXiv preprint \| 2025 \| type posted-content |

## 5 · The one site a naive check inverts

**`apx_b_errata.tex:220`, `silva2025mtlnet`.** A checker reading this sentence as the dissertation's
own claim about the CBIC work reports NOT-SUPPORTED, because the CBIC abstract says the work studied
"POI Category Classification and Next-POI Prediction" while the sentence says next-category and
next-region with negative transfer between them. That is the point of the row: it records that the
**submitted MobiWac manuscript** described the CBIC work that way, that the description was
inaccurate, and that it was corrected. The cited abstract is the evidence FOR the erratum. This is a
systematic false-positive class for errata registers and is recorded here so it is not re-raised.

## 6 · The one entry with no external source of record

**`santos2024urban`** resolves at no external source: no DOI, absent from Crossref, arXiv and
OpenAlex (title search returns unrelated works). It is a UFV master's dissertation. I verified it
against the document itself, which is in the repository at
`articles/dissertacao/exemples/germano/Dissertação_Mestrado___Germano.pdf`:

- Title page: "GERMANO BARCELOS DOS SANTOS", "URBAN REGION REPRESENTATION LEARNING: A POSITIONAL AND
  STRUCTURAL GRAPH APPROACH", Federal University of Viçosa, "Orientador: Fabrício Aguiar Silva",
  2024. Every bib field checks.
- Section 2.6, "Ethical Statement", states the location-privacy position and says which fields were
  left unmasked: the study "used Gowalla anonymized user identifier information, but we maintained
  the location without masking the latitude and longitude of a collected GPS point".
- Searched the full extracted text (244,987 characters) for "research ethics", "Comitê", "CEP",
  "CAAE", "institutional review" and "IRB": **zero occurrences of each**, which is the negative half
  of the citing sentence at `apx_e_ethics.tex:91`.

Every clause of that sentence checks, including the negative one. The entry should keep a bib comment
recording that it has no external identifier by nature, so a future existence-checker does not read
the absence as a defect.

## 7 · What I could not confirm in this unit

Nothing outstanding. All nine key instances are SUPPORTED.
