# Check2HGI Study — Quick Reference

**Thesis:** Check-in-level embeddings (Check2HGI) + next-region auxiliary task improve next-POI prediction over POI-level embeddings (HGI) with single-task training, at matched compute, without negative transfer.

## Task pair

| Slot | Name | Cardinality | Primary metrics | Secondary |
|---|---|---|---|---|
| task_a | `next_poi` | ~10K (AL) / ~80K (FL) | Acc@{1,5,10}, MRR, NDCG@{5,10} | macro-F1 (noisy at this scale) |
| task_b | `next_region` | ~1.1K (AL) / ~4.7K (FL) / ~1.5K (AZ) | Acc@{1,5,10}, MRR | macro-F1 |

**Joint monitor:** `val_joint_lift = mean(acc1_poi / majority_poi, acc1_region / majority_region)` — scale-coherent across the two heads. Checkpoint selection uses this.

## Phases

| Phase | Duration | Claims addressed | Gate to next |
|---|---|---|---|
| P0 | ~1h | CH04 (next-region meaningful), CH05 (ranking metrics > F1) | integrity green + labels generated |
| P1 | ~4h | CH01 (check2HGI > HGI single-task) | CH01 resolved ∈ {confirmed, refuted, partial} |
| P2 | ~5h | CH02 (MTL lift), CH03 (no negative transfer) | CH02 resolved |
| P3 | ~3h | CH06 (dual-stream helps), CH11 (state-dependent gain) | CH06 resolved |
| P4 | ~6h | CH07 (cross-attn > concat) | only runs if CH06 shows ≥ 2pp on FL |
| P5 | ~4h | CH08 (head choice), CH09 (optimiser), CH10 (seed variance) | — |

Full details in `MASTER_PLAN.md`.

## Claims at a glance

- **CH01** check2HGI > HGI single-task next-POI
- **CH02** MTL {next_POI, next_region} > single-task next_POI (headline)
- **CH03** no per-head negative transfer under MTL
- **CH04** next-region meaningful (beats majority by > 2×)
- **CH05** ranking metrics discriminate where macro-F1 collapses
- **CH06** dual-stream region-emb input improves next-POI Acc@10
- **CH07** bidirectional cross-attention > concat (gated on CH06)
- **CH08** next-POI head architecture: next_mtl vs seq baselines
- **CH09** MTL optimiser: NashMTL vs equal_weight vs CAGrad
- **CH10** seed variance ≤ the 2pp effects we call "decisive"
- **CH11** region-input gain is state-dependent (larger on FL)
- **CH12** limitation: Gowalla state-level ≠ FSQ-NYC/TKY
- **CH13** limitation: encoder enrichment out of scope
- **CH14** check2HGI inheritance of HGI fclass shortcut — audit

Full text + test specs in `CLAIMS_AND_HYPOTHESES.md`.

## Baselines

**Ranking baselines from next-POI literature:** HMT-GRN (SIGIR '22), MGCL (Frontiers '24), Bi-Level GSL, LSTPM, STAN, GETNext, Graph-Flashback, ImNext. Reported numbers are on FSQ-NYC/TKY or Gowalla-global; our state-level runs are **not** directly comparable. Declared in CH12.

**Internal baselines (primary comparison):**
- HGI (POI-level) + single-task next-POI
- Check2HGI + single-task next-POI
- Check2HGI + MTL {next_POI, next_region}
- (optional) Check2HGI + MTL + dual-stream input
- (optional) Check2HGI + MTL + cross-attention

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
