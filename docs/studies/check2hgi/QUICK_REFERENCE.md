# Check2HGI Study — Quick Reference

**Task pair:** `{next_category (7 cls, macro-F1), next_region (~1K cls, Acc@K/MRR)}`
**Preset:** `CHECK2HGI_NEXT_REGION`
**Thesis:** Adding next-region auxiliary to next-category on Check2HGI improves macro-F1 without negative transfer.

## Phases

| Phase | What | Key claim |
|---|---|---|
| P0 ✅ | Integrity + simple baselines + audits | CH04 floor |
| **P1** | Region-head validation + head ablation (5 heads, single-task) | CH04, CH05 |
| **P2** | Parameterise all MTL archs + full arch×optim ablation (screen→promote→confirm) | CH06 |
| **P3** | MTL headline with P2 champion, multi-seed n=15 | **CH01**, CH02, CH07 |
| **P4** | Dual-stream region_embedding input | CH03, CH08 |
| P5 | Cross-attention (gated on P4) | CH09 |
| **P6** | Check2HGI encoder enrichment (literature research → implement → ablation) | CH12, CH13 |

## Claims

| ID | Statement |
|---|---|
| **CH01** | MTL {next_cat, next_region} > single-task next_cat (HEADLINE) |
| **CH02** | No per-head negative transfer |
| **CH03** | Dual-stream region-emb input helps |
| CH04 | Region head validates (beats simple baselines ≥ 2×) |
| CH05 | Head choice matters for region (GRU vs transformer?) |
| CH06 | Champion MTL arch × optim identified |
| CH07 | Seed variance < 2pp |
| CH08 | Region-input gain is state-dependent |
| CH09 | Cross-attention > concat (gated) |
| CH10 | Gowalla ≠ FSQ-NYC/TKY (declared) |
| CH11 | Enrichment is a research track (P6) |
| CH12 | Temporal enrichment (Time2Vec-like) improves F1 |
| CH13 | Spatial enrichment (Sphere2Vec-like) improves F1 |

## Baselines

**Next-category:** POI-RGNN (FL: 31.8–34.5% F1), MHA+PE (26.9% F1). Our single-task AL: **38.67%**.
**Next-region:** HMT-GRN, MGCL (concept-aligned, different datasets).
**Simple floor:** AL majority 34.2%, Markov 31.7% (next-cat). AL Markov Acc@10 21.3% (next-region).

## Joint monitor

`val_joint_geom_lift = sqrt((acc1_cat / majority_cat) × (acc1_region / majority_region))` — geometric mean of per-head lifts. Scale-coherent across 7-class vs ~1K-class heads.

## CLI

```bash
# Single-task next-category
python scripts/train.py --state alabama --engine check2hgi --task next --folds 5 --epochs 50

# MTL {next_category, next_region}
python scripts/train.py --state alabama --engine check2hgi --task mtl \
  --task-set check2hgi_next_region --folds 5 --epochs 50 \
  --gradient-accumulation-steps 1

# With class weights for FL
python scripts/train.py --state florida --engine check2hgi --task mtl \
  --task-set check2hgi_next_region --folds 5 --epochs 50 \
  --gradient-accumulation-steps 1 --use-class-weights
```

## Key dataset numbers

| State | next_cat majority | next_region majority | Regions |
|---|---|---|---|
| AL | 34.2% (Food) | 2.3% | 1,109 |
| FL | 24.7% (Food) | 22.5% | 4,703 |
