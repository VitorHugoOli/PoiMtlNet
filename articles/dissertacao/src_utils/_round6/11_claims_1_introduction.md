# 11_claims_1_introduction.md — citation claim-support audit, Chapter 1, Introduction

**Unit:** `src/chapters/1_introduction.tex`  
**Run:** 2026-07-28, round 6, as the per-chapter pass the author asked for by name (COD-008 decision).  
**Errata regime for this unit:** frame chapter: author's own text, no errata mechanism; claim changes are [NEEDS SIGN-OFF]-class.

## 1 · Counts (every citation in the unit, not a sample)

- `\cite` commands in the unit: **8**, on **8** source lines, carrying **9** key instances (a multi-key `\cite` counts once per key). Every one was audited.
- Distinct bibliography keys used: **9**.
- Verdicts: **SUPPORTED** 8, **PARTIAL** 1.

Comments were stripped before counting, so a key that appears only inside a `%` comment is not counted; every counted site renders.

## 2 · Every citation, with its verdict

Verdict scale: SUPPORTED, the citing sentence's attribution is present in or a fair paraphrase of the
source; PARTIAL, part is supported and part is not, or the sentence is stronger than the source;
NOT-SUPPORTED, the attribution is absent from or contradicted by the source; UNVERIFIABLE, the source
of record does not carry enough to decide and the attribution is not implausible.

| # | Site (file:line) | Key | Verdict | Evidence quoted from the source (under 20 words) |
|---|---|---|---|---|
| 1 | `1_introduction.tex:38` | `song2010limits` | SUPPORTED | "there was 93% predictability across the whole user base" |
| 2 | `1_introduction.tex:45` | `luca2021mobilitysurvey` | SUPPORTED | "its impact on several aspects of our society, such as disease spreading, urban planning, well-being, pollution" |
| 3 | `1_introduction.tex:46` | `Xu2023` | SUPPORTED | "categories (e.g., Bar and Museum ) are vital to the task, as they often serve as excellent semantic characterization of the venues" |
| 4 | `1_introduction.tex:50` | `mai2023sphere2vecgeneralpurposelocationrepresentation` | SUPPORTED | "fine-grained species recognition, Flickr image recognition, and remote sensing image classification" |
| 5 | `1_introduction.tex:50` | `wu2024torchspatial` | PARTIAL | "a learning framework and benchmark for location (point) encoding" |
| 6 | `1_introduction.tex:70` | `caruana1997multitask` | SUPPORTED | "improves generalization by using the domain information contained in the training signals of related tasks" |
| 7 | `1_introduction.tex:72` | `kokkinos2016ubernet` | SUPPORTED | "jointly handles low-, mid-, and high-level vision tasks in a unified architecture" |
| 8 | `1_introduction.tex:73` | `lipton2015learning` | SUPPORTED | "multilabel classification of diagnoses, training a model to classify 128 diagnoses given 13 frequently but irregularly sampled clinical measurements" |
| 9 | `1_introduction.tex:75` | `wei2022finetuned` | SUPPORTED | "finetuning language models on a collection of tasks described via instructions" |
## 3 · Failures and partials in this unit, in detail

### `1_introduction.tex:50` — `wu2024torchspatial` — **PARTIAL**

**Citing sentence.** The field also imports its tools: part of the representation machinery used in this research, the spatial location encoders of the second study, was first validated on geospatial tasks such as species recognition and remote sensing classification~\cite{mai2023sphere2vecgeneralpurposelocationrepresentation,wu2024torchspatial}.

**Reference resolved.** arXiv:2406.15658. Source of record: arXiv API; OpenAlex API. Record reads: TorchSpatial: A Location Encoding Framework and Benchmark for Spatial Representation Learning | arXiv preprint | 2024 | type posted-content.

**Located passage.** "a learning framework and benchmark for location (point) encoding"

**Why.** TorchSpatial is a framework and benchmark that consolidates 15 existing location encoders and supplies LocBench (7 geo-aware image classification and 10 regression datasets). It is where the encoders are benchmarked on geospatial tasks, not where they were "first validated". Sphere2Vec, the co-cited entry, does carry the original validation.

**Recommended disposition.** Narrow "first validated" to "validated": the clause is true of both works under the weaker verb, and Sphere2Vec alone supports the stronger one.

## 4 · Source ledger for this unit

Every distinct key cited in this unit, the identifier it resolved by, and where I opened it this session.

| Key | Identifier | Opened at | Record as returned |
|---|---|---|---|
| `Xu2023` | DOI 10.1145/3582553 | Crossref REST; OpenAlex API | TME: Tree-guided Multi-task Embedding Learning towards Semantic Venue Annotation \| ACM Transactions on Information Systems \| 2023 \| type journal-article |
| `caruana1997multitask` | DOI 10.1023/A:1007379606734 | Crossref REST; OpenAlex API; PDF in repo: 10.1023_A_1007379606734.pdf | Multitask Learning \| Machine Learning \| 1997 \| type journal-article |
| `kokkinos2016ubernet` | arXiv:1609.02132 | arXiv API; OpenAlex API | UberNet: Training a `Universal' Convolutional Neural Network for Low-, Mid-, and High-Level Vision using Diverse Datasets and Limited Memory \| arXiv preprint \| 2016 \|  |
| `lipton2015learning` | arXiv:1511.03677 | arXiv API; OpenAlex API | Learning to Diagnose with LSTM Recurrent Neural Networks \| arXiv preprint \| 2015 \| type posted-content |
| `luca2021mobilitysurvey` | DOI 10.1145/3485125 | Crossref REST; OpenAlex API | A Survey on Deep Learning for Human Mobility \| ACM Computing Surveys \| 2021 \| type journal-article |
| `mai2023sphere2vecgeneralpurposelocationrepresentation` | arXiv:2306.17624 | arXiv API; OpenAlex API | Sphere2Vec: A General-Purpose Location Representation Learning over a Spherical Surface for Large-Scale Geospatial Predictions \| ISPRS Journal of Photogrammetry and Remo |
| `song2010limits` | DOI 10.1126/science.1177170 | Crossref REST; OpenAlex API; PDF in repo: 201002-19_Science-Predictability.pdf | Limits of Predictability in Human Mobility \| Science \| 2010 \| type journal-article |
| `wei2022finetuned` | URL https://openreview.net/forum?id=gEZrGCozdqR | arXiv API; OpenAlex API | Finetuned Language Models Are Zero-Shot Learners \| arXiv preprint \| 2021 \| type posted-content |
| `wu2024torchspatial` | arXiv:2406.15658 | arXiv API; OpenAlex API | TorchSpatial: A Location Encoding Framework and Benchmark for Spatial Representation Learning \| arXiv preprint \| 2024 \| type posted-content |

## 5 · What I could not confirm in this chapter

Nothing outstanding beyond the single PARTIAL above. The `song2010limits` figure at `:38` was
checked against the paper itself and not only its abstract: `science/articles/201002-19_Science-
Predictability.pdf` states "a potential 93% average predictability in user mobility" in its summary
and gives Pmax approximately 0.93 in the body, which is what the chapter's "about 93 percent"
reports, with the chapter's own hedge ("potential predictability") matching the source's.
