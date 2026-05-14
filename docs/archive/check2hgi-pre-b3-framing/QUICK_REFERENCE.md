# Check2HGI Study — Quick Reference

**Task pair:** `{next_category (7 cls, macro-F1), next_region (~1K cls, Acc@K/MRR)}`
**Preset:** `CHECK2HGI_NEXT_REGION`
**Thesis (bidirectional):** MTL `{next_category, next_region}` must improve **both** heads over single-task — via per-task input modality (check-in emb → category head, region emb → region head) through a shared MTL backbone.

## Phases

| Phase | What | Key claim |
|---|---|---|
| P0 ✅ | Integrity + simple baselines (**region-level Markov**) + audits | floor |
| **P1** | Region-head ablation × input type {check-in, region, concat} | CH04 (retired), CH05 |
| **P2** | Parameterise all MTL archs (TaskSet port) + full arch×optim ablation | CH06 |
| **P3** | MTL headline: champion + multi-seed n=15, **bidirectional per-head comparison** | **CH01**, **CH02**, CH07 |
| **P4** | **Per-task input modality**: 4-way {per_task, concat, shared_checkin, shared_region} | **CH03**, CH08 |
| P5 | Cross-attention between task-specific encoders (gated on P4) | CH09 |
| **P6** | Check2HGI encoder enrichment (literature research → implement → ablation) | CH12, CH13 |

## Claims

| ID | Statement |
|---|---|
| **CH01** | MTL improves **both** next-category F1 AND next-region Acc@10 over single-task, on AL and FL (HEADLINE, bidirectional) |
| **CH02** | No per-head negative transfer (statistical test on both heads, α=0.05) |
| **CH03** | **Per-task input modality** (check-in → cat, region → region) > shared / concat (ARCH CHOICE) |
| CH04 | Region head vs Markov-1-region — retired as gate, reported as context (AL 1.16× Markov) |
| CH05 | Head choice matters for region — `next_gru` wins, transformer collapses (confirmed) |
| CH06 | Champion MTL arch × optim identified |
| CH07 | Seed variance < 2pp |
| CH08 | MTL gain is state-dependent |
| CH09 | Cross-attention between task-specific encoders > per-task modality (gated) |
| CH10 | Gowalla ≠ FSQ-NYC/TKY (declared) |
| CH11 | Enrichment is a research track (P6) |
| CH12 | Temporal enrichment (Time2Vec-like) improves F1 |
| CH13 | Spatial enrichment (Sphere2Vec-like) improves F1 |

## Baselines

**Next-category:** POI-RGNN (FL: 31.8–34.5% F1), MHA+PE (26.9% F1). Our single-task AL: **38.67%**.
**Next-region:** HMT-GRN, MGCL (concept-aligned, different datasets). Our single-task AL (region-emb input, `next_gru` default, 5f×50ep): **56.94% ± 4.01 Acc@10** (1.21× Markov). FL (1f×30ep): **65.91% Acc@10** (only 1.013× Markov — dense-data regime).
**Simple floor (updated 2026-04-16 — use region-level Markov):**
- next-cat: AL majority 34.2%, Markov 31.7% / FL majority 24.7%, Markov 37.2%
- next-region: AL **Markov-1-region 47.01%** Acc@10 / FL **Markov-1-region 65.05%** Acc@10

The old POI-level `markov_1step` (21.3% AL / 45.9% FL) had ~50% fallback rate to top-k popularity and underestimated the floor by 2×. Paper reports the region-level version.

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
