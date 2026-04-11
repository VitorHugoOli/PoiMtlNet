# Research Variants

This directory tracks experimental variants independently from production-style
pipeline code.

Structure:

- `research/losses/<variant>/`: MTL weighting/scalarization variants.
- `research/mtl/<variant>/`: MTL backbone/architecture variants.
- `research/next/<variant>/`: next-task head variants.
- `research/category/<variant>/`: category-task head variants.
- `research/embeddings/<variant>/`: embedding variants (existing layout).

Each `<variant>/` folder contains:

- `metadata.yaml`: lightweight machine-readable metadata.
- `README.md`: why this variant exists and where it comes from.
- A small Python wrapper module (`loss.py`, `model.py`, or `head.py`) that
  points to the runtime implementation used by current training code.
