# Future Work — POI decoder with HGI embedding as distillation target (§4.8)

**Date drafted:** 2026-05-21
**Source:** [`docs/studies/archive/mtl-protocol-fix/DEFERRED_WORK.md`](../studies/archive/mtl-protocol-fix/DEFERRED_WORK.md) §4.8. NEW memo flagged for drafting in DEFERRED_WORK.md §"NEW MEMOS TO DRAFT WHEN PICKED UP".
**Sequencing:** speculative re-open of the canonical_improvement substrate axis (declared exhausted 2026-05-19 at ±0.8 pp ceiling). Pick up ONLY if a substrate-axis hook is needed AND `mtl_architecture_revisit.md` + `composite_two_substrate_engine.md` both leave a residual reg gap that the substrate side could plausibly close.

## What's deferred

A new substrate variant that augments Check2HGI's training loss with a **decoder branch reconstructing the HGI POI embedding** from the c2hgi POI embedding (distillation framing). The same hard-rule-compatible scope as canonical_improvement Tier 4 (POI feature reconstruction), but **the decoder target is HGI's POI embedding, not raw POI features.**

```
L_total = L_c2hgi_3boundary + γ · ‖Dec(poi_emb_c2hgi) − poi_emb_HGI.detach()‖²
```

`Dec` is a small MLP (~2-3 layers). `γ` sweepable in `{0.05, 0.1, 0.3}`. Cat path stays detached (cat is consumed pre-decoder).

## Why this is different from Tier 4 (canonical_improvement) and Lever 6 (merge_design)

| Approach | Decoder target | Where it acts | Verdict |
|---|---|---|---|
| Tier 4 (canonical_improvement) | Raw POI features (fclass, coords, etc.) | New 4th c2hgi boundary | Falsified — features were already in c2hgi pipeline |
| Lever 6 (merge_design) | Delaunay POI-POI similarity ranking | 4th contrastive boundary | Falsified — boundary contributes 0 reg lift |
| **§4.8 (this memo)** | **HGI's learned POI embedding** | Auxiliary decoder loss on c2hgi POI vectors | **Untested** |

The mechanism hypothesis: HGI's POI embedding encodes 2000-epoch hierarchical-fclass + Delaunay-POI-POI contrastive supervision. Distilling that into c2hgi's POI vectors **transfers HGI's geometry without breaking c2hgi's 3-boundary loss**. If HGI's spatial inductive bias is what gives HGI's STL reg head its +1.6-3.1 pp advantage, this is the mechanism that should transfer it.

## Why deferred

1. **Substrate axis closure.** canonical_improvement Tier 1-6 declared the substrate axis exhausted at ±0.8 pp. Re-opening requires a high-confidence new mechanism; this one is speculative.
2. **Composite alternative.** [`composite_two_substrate_engine.md`](composite_two_substrate_engine.md) achieves the same goal (use HGI's POI semantics for reg) without re-opening the substrate axis. If composite is acceptable for the paper, §4.8 is unnecessary.
3. **Cost.** ~6-10 GPU-h to build + run a sweep at AL/AZ + FL. Not trivial.

## Acceptance criterion

When picked up:

1. **Build script**: `scripts/probe/build_design_t4_hgi_decoder.py` mirroring `build_design_lever6_p2p.py` but with the HGI-embedding-decoder loss.
2. **Sweep** γ ∈ {0.05, 0.1, 0.3} at AL + AZ (small states, cheap pilot).
3. **Promotion gate**: Wilcoxon p ≤ 0.05 on reg vs canonical at AL/AZ on STL reg `next_stan_flow`, no Δ_cat regression > 0.5 pp.
4. **If AL/AZ passes**: extend to FL (~5-6 GPU-h additional).

## Cost (estimated)

- Build script + sweep at AL/AZ: ~4-6 GPU-h.
- FL extension if AL/AZ shows lift: ~3-5 GPU-h.
- **Total: 4-11 GPU-h depending on FL inclusion.**

## Live docs the work would touch

- `scripts/probe/build_design_t4_hgi_decoder.py` — NEW build script
- `docs/studies/merge_design/` — possibly add as Design O / new lever
- `docs/results/RESULTS_TABLE.md` — substrate-axis table extension if winner
- `docs/CLAIMS_AND_HYPOTHESES.md` CH15 — substrate-axis residue claim

## Risks

1. **HGI embedding load.** HGI POI embedding is 256-dim; c2hgi POI is 64-dim. Decoder MLP must lift 64→256. Optimisation is well-posed (autoencoder-style), but at FL with ~143k POIs the decoder may need careful regularisation to avoid overfitting the target.
2. **Substrate-axis falsified-history.** canonical_improvement Tier 4 family (POI feature recon, T5.2a Node2Vec injection) all failed for reasons compatible with §4.8 (concat-style leak, fclass-recovery without reg lift). A future agent must justify why this attempt differs.
3. **Composite preempts.** If composite (§4.2) becomes the paper recipe, §4.8's lift is redundant. Don't run unless composite is rejected for some paper-side reason.

## Pointers

- Tier 4 falsified history: [`../studies/archive/canonical_improvement/INDEX.html`](../studies/archive/canonical_improvement/INDEX.html) §Tier 4
- Lever 6 falsified history: [`../studies/merge_design/LEVER_6_FINDINGS.md`](../studies/merge_design/LEVER_6_FINDINGS.md)
- Composite alternative: [`composite_two_substrate_engine.md`](composite_two_substrate_engine.md)
- HGI POI embedding: `output/hgi/{state}/poi_embeddings.parquet` (256-dim)
- c2hgi POI embedding: `output/check2hgi/{state}/poi_embeddings.parquet` (64-dim)
- Build script template to mirror: `scripts/probe/build_design_lever6_p2p.py`
