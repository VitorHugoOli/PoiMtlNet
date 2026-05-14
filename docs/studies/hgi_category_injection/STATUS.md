# hgi_category_injection — STATUS

## Status: CLOSED (AZ falsified, 2026-05-04)

Six variants tested on Arizona (linear probes + MTLnet + north-star). **Category injection on HGI's POI2Vec does NOT lift either load-bearing metric** (next-CAT F1, next-REG Acc@10) above noise. Best variant (D_orth) is "doesn't hurt" but does not survive the merge_design study's strict gate (paired Wilcoxon p=0.0312 at n=5).

See [`INDEX.md`](INDEX.md) for the full variant table, results, and interpretation.

## Why this folder is in `studies/` and not `archive/`

Per user decision (2026-05-14 reorg): kept under `studies/` pending decision to revisit on FL/CA/TX. The directional reads at AZ are noise-level; multi-state replication MIGHT change the verdict, especially if the per-visit substrate gap at FL/CA/TX (~89-90%) changes the relative weight of the `category` feature in the embedding space.

**Do NOT treat this study as active without an explicit re-open commit.** If you decide not to re-open it, `git mv docs/studies/hgi_category_injection docs/archive/check2hgi-hgi-category-injection-closed-2026-05-04` and remove it from `docs/studies/README.md` Active studies table.

## What was falsified

| Variant | Approach | Cat F1 | Reg Acc@10 | Verdict |
|---|---|---|---|---|
| canonical (fclass, 100ep) | baseline | — | — | reference ceiling |
| category-only (cat-as-fclass, 100ep) | drop POI2Vec to 7 cat buckets | -57 pp on fclass probe | — | confirms "POI2Vec ≈ fclass lookup" |
| baseline (fclass, matched ep) | matched-epoch reference | — | — | reference floor for A/B/C |
| A — separate cat table + λ | fix indexing bug + raise hierarchical λ | within ±1.5 pp on cat | within ±0.5 pp on reg | noise |
| B — concat + PCA | dual POI2Vec, concat → 128 → PCA → 64 | within ±1.5 pp on cat | within ±0.5 pp on reg | noise |
| C — additive joint skip-gram | single skip-gram, two embedding tables, additive POI emb | within ±1.5 pp on cat | within ±0.5 pp on reg | noise |
| A_sum (advisor follow-up, λ-sweep) | A with summed losses, λ ∈ {0.01, 0.1, 1.0, 10.0} | clear regression at λ=10 | — | falsifies large-λ branch |
| C_balanced (advisor follow-up) | C with γ tuned by gradient norm | within noise | — | noise |
| D_orth (advisor follow-up) | orthogonal projection (cos = 2e-9 vs canonical) | +0.36 pp on cat | +0.10 pp on reg | "doesn't hurt" — best variant; but no Wilcoxon at n=5 |

## Realised contributions (despite the falsification)

1. **Sharper mechanism diagnosis**: confirms merge_design's hypothesis that "HGI's POI2Vec is essentially a fclass lookup table" — dropping fclass→category collapses the fclass linear probe from 71% → 13%; category injection in any form doesn't push above baseline.
2. **D_orth is the only "doesn't hurt" variant** — geometrically clean orthogonal projection, neutral on cat, nominal lift on reg. Worth referencing if a future variant (multi-state replication, larger λ space) wants a starting point.
3. **The fclass-as-supervision tautology lives in HGI's published code**: `EmbeddingModel(vocab_size=305, embed_size=64)` is keyed on fclass, but `category` IDs ∈ [0..6] index the same table — "category 0" was effectively fclass-0's embedding. Documented for future agents.

## What's NOT shown

- Multi-seed / multi-state replication. All numbers are AZ × seed=42 × 30-epoch protocol. The strict-Wilcoxon gate at n=5 is `p=0.0312`; my best results are at `p=0.0625` floor.
- The c2hgi family's published HGI cat F1 (28.69%) doesn't reproduce in my pipeline (18.73%). The 10-pp gap is a protocol artefact (different input pipelines for the `next_gru` head); within-substrate variant comparisons are unaffected.

## Re-open criteria

Re-open this study if EITHER:
1. A new theoretical motivation emerges (e.g., a different way to inject category that wasn't in the 6 tested variants).
2. A multi-state replication on FL/CA/TX shows a directionally different read than AZ (particularly if D_orth or one of the noise-level variants becomes positive).

Otherwise, follow the archive policy in [`../README.md`](../README.md) §Archive policy.
