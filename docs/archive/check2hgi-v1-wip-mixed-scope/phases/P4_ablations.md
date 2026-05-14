# Phase P4 — Ablations

**Gates:** P3 complete (CH02 and CH03 resolved — ablations only make sense on top of a positive headline).
**Exit gate:** Tables 2, 3, 4 in `ABLATIONS.md` populated; CH06, CH07, CH08, CH09 resolved.

See `ABLATIONS.md` for the full matrix.

## Experiments

### Head architecture (Table 2 → CH07)

Vary `next_region` head across `next_mtl` (default), `next_lstm`, `next_gru`, `next_transformer_relpos`, `next_hybrid`. Alabama only, 5-fold CV, everything else pinned at the P3 default.

5 runs.

### MTL optimiser (Table 3 → CH08)

Vary MTL criterion across `nash_mtl` (default), `cagrad`, `aligned_mtl`, `naive`, `pcgrad`. Alabama only.

5 runs.

### Micro-ablations (Table 4 → CH06, CH09)

- Row 1 (default): already run as P3.
- Row 2: flip monitor to `joint_f1`. Same data, just different checkpoint-select.
- Row 3: `task_embedding=False` — zero-initialise and freeze `MTLnet.task_embedding` (CH09).
- Row 4: `num_classes=7` for next_region (cap region labels to top-7 by frequency). Control for cardinality.
- Row 5: seed sweep 42/43/44/45/46 on default. Establishes variance floor.

≈ 8 runs.

## Order

1. Seed sweep (row 5) — needed first so we know the noise floor for every other comparison.
2. Head architecture — cheapest, most likely reviewer ask.
3. MTL optimiser — moderate cost.
4. Micro-ablations (task_embedding, monitor choice, cardinality cap) — last.

## Claims touched

CH06, CH07, CH08, CH09. Populate Tables 2, 3, 4 in `ABLATIONS.md`.
