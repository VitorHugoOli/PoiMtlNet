# HGI MTL+F1 at Florida (B9) — the MISSING three-way ceiling

> **For the completion agent**: this file is ADDITIVE and self-contained. The
> parent should integrate the three-way table below into the tier_b_fl
> CLOSURE/log. This agent did NOT touch CLOSURE.md / log.md / CHANGELOG while
> the FL design/STL completion agent was active. Numbers produced 2026-05-29.

## What this fills

HGI MTL+F1 had **never** been evaluated anywhere. FL had MTL canonical
(~63.98 disjoint reg) + designs B/J/L (≈ canonical disjoint, null) but **no HGI
MTL**. This run provides the MTL ceiling: does HGI itself — the substrate that
beats canonical at STL reg (HGI ~70.9 % vs canonical ~68.4 % no-prior) — carry
that advantage into joint training at FL?

Recipe: NORTH_STAR **B9**, FL, seed=42, 5-fold, `--engine hgi`,
`static_weight` (cat-weight 0.75), cosine max-lr 3e-3, cat-head `next_gru`,
reg-head `next_getnext_hard`, task_a=checkin / task_b=region, per-fold
seed-42 log_T (cp'd from canonical, region label space identical), `--no-checkpoints`.

## Build cost (for the record)

| Stage | Cost |
|---|---|
| HGI Phase-4 preprocess (FL, reuse existing temp + poi2vec) | 26.3 s |
| HGI Phase-5 train (2000 epochs, CPU, dim=64) | **378.7 s (0.105 h)**, loss 8.52 → 0.377 |
| HGI MTL+F1 (5-fold B9, A40) | **15.01 min (0.25 GPU-h)**, ~170 s/fold |
| **Total** | **≈ 0.36 GPU-h** (well under the 6 GPU-h ceiling) |

Embeddings non-degenerate: POI 76544×64 (74313 unique rows, fclass-level
sharing expected), region 4703×64 (all unique), no NaN. next_region.parquet
(159175×579) + next.parquet (159175×578) row-aligned to canonical c2hgi
(region_idx / last_region_idx byte-identical; cat labels reused verbatim).

## HGI MTL FL — RAW per-fold

| Fold | disjoint reg | joint reg | disjoint cat (F1) | joint cat (F1) |
|---|---|---|---|---|
| 1 | 63.99 | 61.59 | 34.66 | 34.64 |
| 2 | 65.46 | 65.08 | 34.82 | 33.95 |
| 3 | 64.58 | 62.52 | 35.69 | 34.63 |
| 4 | 64.31 | 64.21 | 34.67 | 33.79 |
| 5 | 64.11 | 62.42 | 34.35 | 32.94 |
| **mean** | **64.49** | **63.16** | **34.84** | **33.99** |
| std | 0.55 | 1.46 | 0.50 | 0.65 |

Reg = `top10_acc_indist` (%), cat = macro-F1 (%); disjoint = each head's own
best-val epoch; joint = single shared epoch maximising sqrt(cat_F1 · reg_top10).

**Cat is catastrophic, as expected** (HGI POI-static fclass-level embeddings ≈
324 distinct vectors → ~34.8 % F1 vs canonical ~70.5 %). This was anticipated
(merge_design: HGI cat ≪ c2hgi); the point of this run is the **reg ceiling**.

## THE CEILING QUESTION — does HGI beat canonical in MTL reg at FL?

Wilcoxon one-sided HGI > canonical, RAW per-fold (paired, n=5):

| Front | HGI mean | canon mean | Δ mean | per-fold Δ | W | p (HGI>canon) |
|---|---|---|---|---|---|---|
| **disjoint reg** | 64.49 | 63.98 | **+0.51** | [-0.31, +2.73, -0.17, +0.11, +0.21] | 9 | **0.406 (NS)** |
| **joint reg** | 63.16 | 61.14 | **+2.02** | [-0.73, +5.24, +0.98, +2.77, +1.82] | 14 | **0.0625 (marginal)** |
| disjoint cat | 34.84 | 70.49 | −35.66 | — | 0 | 1.0 (HGI far worse) |
| joint cat | 33.99 | 66.98 | −32.99 | — | 0 | 1.0 (HGI far worse) |

**Answer: NO — HGI does NOT meaningfully beat canonical in MTL reg at FL.**
- On the **disjoint** front HGI ≈ canonical (+0.51 pp, p=0.41, not significant).
- On the **joint** front HGI is +2.02 pp (p=0.0625, marginal, not significant
  at α=0.05) — and this "edge" is partly because HGI's joint reg is dragged down
  *less* (HGI cat collapses, so the geom-mean-selected epoch is essentially the
  reg-best epoch; the joint penalty that hurts canonical's reg doesn't bite HGI
  because HGI has no cat to trade off). The disjoint front — the clean reg-vs-reg
  comparison — is the honest read, and there HGI = canonical.

## Three-way cross-reference (FL MTL, seed=42, B9, RAW per-fold means)

| Substrate | disjoint reg | joint reg | disjoint cat | joint cat |
|---|---|---|---|---|
| **canonical c2hgi** | 63.98 | 61.14 | 70.49 | 66.98 |
| Design B (POI2Vec @ pool) | 63.82 | 57.65 | 68.61 | 67.33 |
| Design J (H + anchor λ0.1) | 64.06 | 57.52 | 68.80 | 68.01 |
| Lever 5 / Design L (KL distill) | 63.97 | 57.67 | 68.71 | 67.68 |
| **HGI (this work)** | **64.49** | **63.16** | **34.84** | **33.99** |

(Design + canonical cells from the existing tier_b_fl runs; HGI is this agent's.
Design joint-reg ≈ 57.5-57.7 are BELOW canonical 61.14 — the designs trade reg
on the joint front; HGI does not, but only because HGI carries no cat.)

## Implication for the design verdict + "MTL regime is the bottleneck"

1. **MTL flattens EVERYONE on disjoint reg.** At STL, HGI reg (~70.9 %) clearly
   beats canonical (~68.4 % no-prior, merge_design). Under B9 MTL at FL, HGI's
   disjoint reg drops to 64.49 % — statistically indistinguishable from
   canonical's 63.98 % and from designs B/J/L (63.8-64.1 %). **The ~2.5 pp
   STL HGI-over-canonical reg advantage VANISHES under joint training.** Even
   the substrate that demonstrably wins at STL cannot carry that win into MTL.
   This is the strongest direct evidence yet that **the MTL regime — not the
   substrate — is the reg bottleneck at FL**: the joint-training optimisation /
   shared-backbone / cat-coupling collapses all substrates to the same reg band.

2. **The designs are not "failing to carry HGI's advantage" — there is no
   advantage left to carry under MTL.** A plausible prior worry was "designs
   fail because they don't import HGI's STL edge into MTL." This run falsifies
   that framing: HGI ITSELF has no MTL reg edge to import. The designs land in
   the same disjoint-reg band as both canonical and HGI; they are null vs
   canonical for the same reason HGI is — the regime, not the embedding.

3. **HGI is not a viable MTL substrate** despite the marginal joint-reg number:
   its cat collapses (~35 % F1, −36 pp), so any cat-bearing joint objective is
   wrecked. The marginal joint-reg "win" is an artefact of HGI having no cat to
   trade against. canonical remains the only substrate that holds BOTH heads.

## Reconciliation of units (honest)
STL reg ~70 % vs MTL reg ~64 % is a **regime difference**, not a contradiction:
STL trains a single next_region head end-to-end; MTL B9 shares a backbone with
the cat head under static_weight 0.75 and the alternating-step schedule, which
costs ~6 pp of reg across all substrates. All numbers here are the SAME
two-front protocol as the design/canonical cells (`analyze_tier_b_fl.py`), so
the three-way table is internally comparable.

## Artefacts
- Embeddings: `output/hgi/florida/{embeddings,region_embeddings}.parquet`
- Inputs: `output/hgi/florida/input/{next,next_region}.parquet` (+ cp'd seed-42 log_T fold1-5, touched after parquets, C22-clean)
- Run: `docs/results/substrate_protocol_cleanup/tier_b_fl/mtl_hgi/florida/seed42/mtlnet_lr1.0e-04_bs2048_ep50_20260529_113917_151084/`
- Analysis JSON: `docs/results/substrate_protocol_cleanup/tier_b_fl/mtl_hgi/hgi_mtl_fl_analysis.json`
- Build scripts: `scripts/substrate_protocol_cleanup/build_hgi_fl_{phase4_probe,train}.py`, `scripts/substrate_protocol_cleanup/build_hgi_next_cat.py`, `scripts/probe/build_hgi_next_region.py`, analyzer `scripts/substrate_protocol_cleanup/analyze_hgi_mtl_fl.py`

> Note for parent: STL-reg HGI under the current pipeline (Stage D, optional)
> was NOT re-run — budget reserved, and merge_design already documents STL HGI
> ~70.9 %. The STL-vs-MTL gap is the regime delta discussed above.
