# Check2HGI Study — Quick Reference

**Thesis:** On Check2HGI check-in-level embeddings, MTL `{next_poi, next_region}` improves next-POI prediction over single-task training, without negative transfer. Standalone study — no cross-engine or prior-work comparisons. Baselines are internal + simple-baselines floor (majority/random/Markov/top-K on our data).

## Task pair

| Slot | Name | Cardinality | Primary metrics | Secondary |
|---|---|---|---|---|
| task_a | `next_poi` | ~10K (AL) / ~80K (FL) | Acc@{1,5,10}, MRR, NDCG@{5,10} | macro-F1 (noisy at this scale) |
| task_b | `next_region` | ~1.1K (AL) / ~4.7K (FL) / ~1.5K (AZ) | Acc@{1,5,10}, MRR | macro-F1 |

**Joint monitor:** `val_joint_geom_lift = sqrt((acc1_poi/majority_poi) * (acc1_region/majority_region))` — **geometric** mean of per-head lifts. Fixed on 2026-04-15 from the arithmetic-mean version (which was dominated by the POI head when cardinalities differ by orders of magnitude). Checkpoint selection uses this. Reported alongside: `val_joint_arith_lift` (old formula, for cross-ref), per-head raw Acc@K, and **OOD-restricted Acc@K** (sequences where target POI appears in train fold).

## Phases

| Phase | Duration | Claims addressed | Gate to next |
|---|---|---|---|
| P0 | ~3h | CH04 (beats baselines), CH05, CH14 (shortcut audit), CH15 (leakage audit) | integrity green + labels + simple baselines computed + audits resolved |
| P1 | ~3h | CH04 floor check, CH05, CH06 (OOD) | single-task references exist for P2 pairing |
| P2 | ~12h | **CH01 (MTL lift, HEADLINE), CH02 (no negative transfer)** | CH01 resolved — 3 seeds × 5 folds |
| P3 | ~9h | CH03 (dual-stream helps), CH08 (state-dependent gain) | CH03 resolved |
| P4 | ~6h | CH07 (cross-attn > concat) | only runs if CH03 shows ≥ 2pp on FL |
| P5 | ~4h | CH09 (head choice), CH10 (optimiser on AL+FL), CH11 (seed variance) | — |

**P2 is ~3× longer than v1** because multi-seed (n=15) became the default for CH01/CH02 per review-agent recommendation — n=5 Wilcoxon couldn't detect 2pp effects with any power.

Full details in `MASTER_PLAN.md`.

## Claims at a glance

- **CH01** MTL {next_POI, next_region} > single-task next_POI on Check2HGI (HEADLINE)
- **CH02** no per-head negative transfer
- **CH03** dual-stream region-emb input improves next-POI
- **CH04** learned models beat simple baselines (majority/random/Markov/top-K) by ≥ 2×
- **CH05** ranking metrics discriminate where macro-F1 collapses
- **CH06** OOD-restricted Acc@K still beats baselines (train-memorisation guard)
- **CH07** bidirectional cross-attention > concat (gated on CH03)
- **CH08** region-input gain is state-dependent
- **CH09** head architecture: next_mtl vs seq baselines
- **CH10** MTL optimiser: NashMTL vs equal_weight vs CAGrad (AL + FL)
- **CH11** seed variance bound ≤ 2pp
- **CH12** limitation: Gowalla state-level ≠ FSQ-NYC/TKY
- **CH13** limitation: encoder enrichment out of scope
- **CH14** preprocessing shortcut audit (standalone; not HGI-comparison)
- **CH15** transductive embedding-leakage audit

Full text + test specs in `CLAIMS_AND_HYPOTHESES.md`.

## Baselines (three tiers)

### 1. Simple baselines on our data (P0.5) — the floor
- Majority-class POI (always predict most frequent)
- Random POI (uniform over n_pois)
- 1-step Markov: P(next_poi | current_poi) learned from train folds
- Top-K popular: Acc@K equals cumulative frequency of top-K POIs
- User-history top-K: per-user most-visited POIs
- Same 5 baselines for next_region.

### 2. Internal baselines (P1–P4) — the contribution chain
- Check2HGI + single-task next-POI (P1)
- Check2HGI + MTL `{next_POI, next_region}` (P2)
- + dual-stream region input (P3)
- + cross-attention (P4, gated)

### 3. External-literature baselines (appendix only)
HMT-GRN (SIGIR '22), MGCL (Frontiers '24), Bi-Level GSL, LSTPM, STAN, GETNext, Graph-Flashback, ImNext. Reported on FSQ-NYC/TKY / Gowalla-global — **not directly comparable** to our Gowalla state-level numbers. Appendix table with scope caveat (CH12).

## Datasets

Gowalla state-split:

| State | Check-ins | POIs | Regions | Seq rows |
|---|---|---|---|---|
| Alabama | 113,846 | 11,848 | 1,109 | 12,709 |
| Florida | 1,407,034 | 76,544 | 4,703 | 159,175 |
| Arizona (triangulation) | ~120K | ~10K | 1,547 | 26,396 |

## Artefact paths

- Embeddings: `output/check2hgi/{state}/{embeddings,poi_embeddings,region_embeddings}.parquet`.
- Next-POI sequence X: `output/check2hgi/{state}/input/next.parquet` (576 cols + `target_poi` raw placeid + `userid`).
- Next-region labels: `output/check2hgi/{state}/input/next_region.parquet` (X + `region_idx` + `userid`).
- Preprocessing artefact: `output/check2hgi/{state}/temp/checkin_graph.pt` (pickle: `placeid_to_idx`, `poi_to_region`).

## CLI

```bash
# Single-task next-POI (once P1 loaders + script flag are wired)
STUDY_DIR=docs/studies/check2hgi python scripts/train.py \
  --state alabama --engine check2hgi --task next_poi \
  --folds 5 --epochs 50

# 2-task MTL headline
STUDY_DIR=docs/studies/check2hgi python scripts/train.py \
  --state alabama --engine check2hgi --task mtl \
  --task-set check2hgi_next_poi_region \
  --folds 5 --epochs 50 --gradient-accumulation-steps 1
```

## Env vars

- `DATA_ROOT` — default `<project>/data` (use main repo's data dir on this worktree).
- `OUTPUT_DIR` — default `<project>/output`.
- `STUDY_DIR` — default `docs/studies/fusion`. **Always set to `docs/studies/check2hgi` on this branch.**
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` before long runs.

## Related studies

`docs/studies/fusion/` — sibling study on POI-category classification. Different task, different baselines, different state machine. Do not cross-modify.
