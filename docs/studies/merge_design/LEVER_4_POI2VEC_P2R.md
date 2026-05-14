# Lever 4 — POI2Vec at the p2r boundary (proposal)

## Aim

Inject POI2Vec at a **second** boundary — the POI→Region pooling — in
addition to the existing pool-boundary injection that B/H/I/J/M all use.

## Why

Currently every merge design adds POI2Vec only at the POI level
(`poi_emb_canonical.detach() + γ · POI2Vec`). HGI's hierarchy then
pools those POI vectors into regions via `POI2Region` (PMA + region
GCN). Region embeddings inherit POI2Vec semantics indirectly through
PMA aggregation.

HGI builds region embeddings the same way but starts from POI2Vec at
the POI level *plus* the contrastive `L_p2r` loss anchored on POI2Vec
similarity. The merge family has `L_p2r` but the POI input is a
detached canonical POI vector with a residual — the *discriminator*
sees POI2Vec only via the residual term.

Lever 4 makes the region-side discriminator see POI2Vec directly:
augment `L_p2r` with a region-prior term derived from per-region
mean-pooled POI2Vec.

```
region_prior[r] = mean({ POI2Vec[i] for i in pois_of_region[r] })

L_p2r = L_p2r_canonical + α · cos(region_emb[r], region_prior[r])
```

`α` ≈ 0.1; tunable.

## Hypothesis

The region path currently learns POI2Vec semantics indirectly via the
detached residual; making it direct shortens the credit-assignment
path and may give a small lift (+0.3-0.6 pp) on AZ/FL reg without
disturbing the cat path.

## Cost

~4 h. Add a new design (call it "N" or "P", non-clashing with K/L/M).
Mostly a 5-line change in the loss aggregation.

## Risk

- Region collapse: low-medium. If `α` is too high, region embeddings
  collapse to mean-POI2Vec and lose canonical c2hgi region structure.
  Sweep `α ∈ {0.03, 0.1, 0.3}`.
- Cat regression: zero (cat path is already detached and not touched).

## Decision criterion

Same as Lever 5 — if AZ closes to within 0.3 pp of HGI without cat
regression, ship and run FL.
