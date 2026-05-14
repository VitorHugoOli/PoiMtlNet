# F41 — Experiment D: STL + MTL Pre-encoder Ablation

**Date:** 2026-04-24. **Tracker:** `FOLLOWUPS_TRACKER.md §F41`. **Script:** `scripts/run_f41_stl_mtl_preencoder.sh` + `scripts/p1_region_head_ablation.py --mtl-preencoder`. **Cost:** ~10 min MPS (AL 2.5min + AZ 5min).

## Question

CH18 (F21c) showed STL `next_getnext_hard` beats MTL-B3 on reg Acc@10 by 12–14 pp at AL+AZ. F38 refuted checkpoint-selection (Fator 2) and F39 refuted loss-weight (Fator 1). The remaining structural candidate was Fator 3: the MTL pipeline alters the head's input distribution via (a) an MLP pre-encoder (64→256) and (b) cross-attention blocks. F41 tests (a) alone.

## Method

Extended `scripts/p1_region_head_ablation.py` with a `--mtl-preencoder` flag. When set, wraps the head with `_MTLPreencoder` — a Linear+ReLU+LayerNorm+Dropout stack mirroring `MTLnet._build_encoder` (`in_size=64, hidden_size=256, out_size=256, num_layers=2, dropout=0.1`). The head then sees `[B, 9, 256]` input, matching what the MTL head receives from `next_encoder`. No cross-attention is added.

Run identical to F21c otherwise: 5f × 50ep, seed 42, AdamW(lr=1e-4, wd=0.01), OneCycleLR(max_lr=3e-3), `next_getnext_hard(d_model=256, num_heads=8)`, input_type=region, transition path per state.

## Results

| State | Variant | Acc@10 | Acc@5 | MRR | best_ep range |
|---|---|---:|---:|---:|:-:|
| **AL** | STL F21c (puro, no pre-encoder) | **68.37 ± 2.66** | ~52.19 | ~41.17 | ~34 |
| | **F41 STL + MTL pre-encoder** | **67.95 ± 2.67** | 56.41 | 40.63 | 26–34 |
| | Δ | **−0.42 pp** | +4.22 | −0.54 | — |
| | σ-overlap? | ✓ | ✓ | ✓ | |
| **AZ** | STL F21c (puro) | **66.74 ± 2.11** | ~50.95 | ~41.15 | — |
| | **F41 STL + MTL pre-encoder** | **66.30 ± 2.31** | 55.24 | 40.99 | 26–34 |
| | Δ | **−0.44 pp** | +4.29 | −0.16 | — |
| | σ-overlap? | ✓ | ✓ | ✓ | |

## Verdict

**Fator 3a (upstream MLP pre-encoder) is REFUTED.** Adding the MTLnet pre-encoder stack to the STL path does NOT depress reg Acc@10 — within σ at both AL and AZ, the difference is ≤ 0.5 pp. The head tolerates the upstream MLP (64→256) without measurable degradation; STL's inductive gradient on the reg task survives the additional parameters.

Secondary observation: Acc@5 and F1 actually rose slightly under the pre-encoder variant on both states, suggesting the extra capacity helps fine-grained ranking. Head architecture is adaptable to the input dimension change.

## Implication for CH18 attribution

After F38 (Fator 2 refuted), F39 (Fator 1 refuted), F42 (Fator 5 / epoch budget refuted inversely), and now F41 (Fator 3a refuted), the remaining candidates for the 12–14 pp gap are:

1. **Fator 3b — Cross-attention blocks.** The MTL pipeline has 2 cross-attention blocks between the pre-encoder and the head (`MTLnetCrossAttn._CrossAttnBlock`). In MTL, the reg head's input is the cross-attn output (reg stream bidirectionally attending to cat stream); in STL+pre-encoder (F41), the head sees only the pre-encoder output. The extra cross-attn processing could:
   - Smooth the STAN backbone input, dampening the graph-prior's effective magnitude;
   - Pollute region-relevant features with category-relevant ones through cross-task attention.

2. **Fator 2-residual — Joint-training gradient pollution.** In MTL, the cross-attn params see gradient signal from BOTH cat loss and reg loss. The cat-loss-side gradient could bias the cross-attn params toward features that help cat at reg's expense, even though the absolute loss weight (F39-refuted) isn't the issue.

## Next experiments

1. **F43 — B3 with `category_weight=0.01` (reg-dominant loss).** Tests Fator 2-residual. If reg Acc@10 rises toward STL F21c (66–68), joint-training signal is the culprit; cross-attn architecture itself is fine. If reg Acc@10 stays at ~56, cross-attn architecture is the culprit regardless of cat-loss contribution. Cost: 1 AL run, ~10 min.

2. **F41-D2 (deferred, gated on F43 outcome).** STL + pre-encoder + self-attention blocks matching cross-attn dimensions (Q=K=V=same stream). Isolates the extra transformer capacity from the cross-stream information transfer. Implementation cost: ~2h dev.

## Files

- Launcher: `scripts/run_f41_stl_mtl_preencoder.sh`
- Code: `scripts/p1_region_head_ablation.py::_MTLPreencoder` (new class, 2026-04-24)
- Source JSONs (AL): `results/check2hgi/alabama/region_head_alabama_region_5f_50ep_stl_gethard_preenc.json`
- Source JSONs (AZ): `results/check2hgi/arizona/region_head_arizona_region_5f_50ep_stl_gethard_preenc.json`

## Cross-references

- `research/F21C_FINDINGS.md` — original CH18 gap measurement
- `research/F38_CHECKPOINT_SELECTION.md` — Fator 2 refutation
- `CLAIMS_AND_HYPOTHESES.md §CH18` — canonical claim
- `FOLLOWUPS_TRACKER.md §F41, §F43` — experiment chain
