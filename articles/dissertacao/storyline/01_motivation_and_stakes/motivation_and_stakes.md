# Motivation and stakes — the settled content

> What the frame chapters SAY about why this work matters, with each claim's license.
> Rendered in: Ch.1 §1.1 (`drafts/1_introduction.tex`). Process trail: `archive/process/09`.

## The stakes (Ch.1 opening funnel)

1. **The data**: check-ins from location-based social networks; "at large scale" (NO quantity —
   "billions" was removed as unledgered; do not reintroduce a number without an opened source).
2. **The regularity**: potential predictability of an individual's next location "about
   93 percent" — `song2010limits` (Science 327:1018, opened firsthand). Always "potential
   predictability" (upper bound), never achieved accuracy.
3. **What a service does with the two predictions**: recommender ranks candidates of the right
   type (category); navigation/transit prepares for where the user heads (region); platform
   allocates attention and resources by area.

## Beyond-mobility breadth (author-approved Item 11; all anchors verified by opened abstracts)

- Human mobility informs urban planning, disease spreading, pollution analysis →
  `luca2021mobilitysurvey` (abstract lists exactly these).
- Place categories = the semantic characterization location-based services rely on → `Xu2023`
  (abstract: categories "serve as excellent semantic characterization of the venues").
  NOT "urban planning" — that framing was CBIC's, not Xu's.
- Encoder provenance: "part of the representation machinery ... the spatial location encoders of
  the second study, was first validated on geospatial tasks such as species recognition and
  remote sensing classification" → `mai2023sphere2vec` + `wu2024torchspatial`. Scoped to the
  CoUrb encoders only — never "the representation machinery" unqualified.
- MTL breadth: vision (`kokkinos2016ubernet`), clinical multilabel diagnosis
  (`lipton2015learning`), instruction-tuned language models (`wei2022finetuned` = FLAN — the
  wording must say instruction tuning across tasks, not classic NLP MTL).

## The engineering wish (the tension beat)

Operational simplicity ONLY: one artifact to train, version, deploy; one forward pass, both
answers. **Never "lower cost"** (F3 guard: the joint model is larger and cost more to train —
disclosed, not hidden). The threat: negative transfer. The open question at research start:
does joint training help this pair, and what does the answer depend on.

## Sources of truth

Citation ledger with verification records: `drafts/1_citations.md`. Verified breadth analysis:
`archive/process/09_application_scope_breadth/`.