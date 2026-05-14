# Lever 5 — Distribution-level POI distillation (proposal)

## Aim

Test whether the residual ~1 pp gap to HGI on AZ/FL reg is closed by
matching the **neighbour-similarity distribution** of POI2Vec, instead
of the pointwise vector alignment that Design M currently uses.

## Why

Design M's distillation loss (`build_design_m_distill.py:88-92`):

```
L_distill = (1 − cos(P(poi_emb), POI2Vec)).mean()
```

This is a *pointwise* alignment — pull each POI's vector toward its
POI2Vec target. It successfully transfers fclass-cluster identity (probe
→ 98 %) but doesn't transfer *neighbour structure*. HGI's contrastive
boundary `L_p2p` is implicitly distributional: it scores POI-POI
similarity rankings, not vector means.

Lever 5 replaces the pointwise loss with a KL on top-k softmax over the
similarity *distribution* among neighbours:

```
S_merge[i, :] = top-k cosine(merge_poi_emb[i], merge_poi_emb[other])    # ranked over k POIs
S_p2v[i, :]   = top-k cosine(POI2Vec[i],       POI2Vec[other])
L_distill     = KL( softmax(S_merge / τ) ‖ softmax(S_p2v / τ) ).mean()
```

`τ` ≈ 1.0 is a temperature hyper-parameter. `k` ≈ 10-20 is the top-k
neighbour count. The sampling is over a fixed neighbour set per POI
(precomputed once from POI2Vec), so the loss is cheap.

## Hypothesis

The gap to HGI is in *which POIs are similar to which*, not in the raw
POI vectors. KL-on-distributions transfers ranking; pointwise cosine
does not. Expected: closes ≥0.5 pp of the AZ residual (currently
−1.29 pp vs HGI under K/J).

## Cost

~3 h on MPS for AL+AZ build × eval. New script:
`scripts/probe/build_design_l_distkl.py` (or substitute "L" with a
non-clashing letter — "N" is free; ensure compatibility with the
`startswith(\"check2hgi_design_…\")` branch in
`scripts/p1_region_head_ablation.py`).

## Risk

- Cat regression: low. The distillation is on `poi_table` (the residual
  branch), not on `checkin_emb`. Cat path is detached.
- Optimisation instability: low. KL on a fixed-target distribution is
  well-behaved as long as `τ > 0.5`.
- Generality probe drift: medium. The fclass linear probe could go up
  or down — neighbour-ranking distillation may push the geometry away
  from HGI's vector space (lower fclass) while improving rank quality
  (higher reg). Need to track both.

## Decision criterion

If Lever 5 at AL+AZ shows reg Acc@10 within ≤0.3 pp of HGI on AZ
(currently 53.37) and ≥+1.5 pp over canonical, ship it. Then run FL
to confirm.
