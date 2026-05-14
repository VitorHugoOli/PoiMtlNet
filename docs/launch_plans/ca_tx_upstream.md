# N8 — CA + TX upstream pipeline handoff

**Date:** 2026-04-28
**Goal:** Stage the CA + TX upstream embedding/input pipeline plan so the user can launch in the morning. P3 in PAPER_PREP_TRACKER (~37h total Colab).

---

## Why this is paper-blocking

The headline paper covers **FL + CA + TX** as US-state replication. CA and TX have **no Check2HGI embeddings yet**. Without them, the paper either:
- (a) Submits as **AL+AZ+FL only** (3-state paper, fallback per risk register C13), OR
- (b) Waits for full CA+TX pipeline before submission.

P3 closes option (b).

---

## Pipeline stages per state (CA, TX)

Each state requires **5 sequential pipeline stages** before MTL training can begin. Total ~12h Colab T4 per state per stage chain.

### Stage 1 — Check2HGI embedding generation (~3-4h Colab T4)
Generates `output/check2hgi/{ca,tx}/check_embeddings.parquet` + `region_embeddings.parquet`.
```bash
python pipelines/embedding/check2hgi.pipe.py --state california --epochs 200
python pipelines/embedding/check2hgi.pipe.py --state texas --epochs 200
```
Inputs needed:
- `data/checkins/California.parquet` (~500K check-ins)
- `data/checkins/Texas.parquet` (~700K check-ins)
- `data/miscellaneous/tl_2022_06_tract_CA/` (TIGER census tracts)
- `data/miscellaneous/tl_2022_48_tract_TX/`

### Stage 2 — Region transition matrix (~30 min Colab T4)
```bash
python pipelines/region_transition.py --state california
python pipelines/region_transition.py --state texas
```
Output: `output/check2hgi/{ca,tx}/region_transition.npz`. Used by `next_getnext_hard` reg head.

### Stage 3 — Input creation (~1h Colab T4)
```bash
python pipelines/create_inputs_check2hgi.pipe.py --state california --window 9
python pipelines/create_inputs_check2hgi.pipe.py --state texas --window 9
```
Outputs:
- `output/check2hgi/{ca,tx}/category_inputs.parquet`
- `output/check2hgi/{ca,tx}/next_inputs.parquet`

### Stage 4 — F34 / F35: MTL B3 + `next_gru` 1-fold sanity (~6h Colab T4 per state)
```bash
python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --model mtlnet_crossattn --mtl-loss static_weight --category-weight 0.75 \
    --cat-head next_gru --reg-head next_getnext_hard \
    --folds 1 --epochs 50 --batch-size 1024 \
    --max-lr 3e-3 --scheduler one_cycle \
    --state california \
    --output-dir results/check2hgi/california/f34_b3_gru_1f50ep

# same for texas → F35
```
Acceptance: cat F1 ≥ 0.5 AND reg Acc@10 ≥ 0.5 (sanity floors). Wide envelope; just confirms pipeline works end-to-end.

### Stage 5 — F24 / F25: H3-alt 5-fold headline runs (~12h Colab T4 per state)
```bash
python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --model mtlnet_crossattn --mtl-loss static_weight --category-weight 0.75 \
    --cat-head next_gru --reg-head next_getnext_hard \
    --folds 5 --epochs 50 --batch-size 1024 \
    --scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --state california \
    --output-dir results/check2hgi/california/f24_h3alt_5f50ep

# same for texas → F25
```
This is the **headline cell** per state. Output goes into the cross-state summary table in PAPER_STRUCTURE.md.

---

## Total cost summary

| Stage | Per state | Both states |
|-------|-----------|-------------|
| 1 — Embedding gen | ~4h | ~8h |
| 2 — Region transition | ~30 min | ~1h |
| 3 — Input creation | ~1h | ~2h |
| 4 — F34/F35 1f sanity | ~6h | ~12h |
| 5 — F24/F25 5f headline | ~12h | ~24h |
| **Total** | **~24h** | **~47h** |

(Higher than the PAPER_PREP_TRACKER's "~37h total" — that estimate may have assumed Stage 1+2+3 already done. Verify before committing.)

---

## Critical-path ordering

1. CA can be parallel to TX from Stage 1 onwards (different machines/notebooks).
2. Within a state: stages MUST run sequentially (each consumes prior output).
3. F33 (FL decisive test) gates whether to use `next_gru` or `next_mtl` for cat head on CA/TX. **Run F33 first, then start Stage 1 for CA/TX with the chosen cat head.**

---

## Phase-2 substrate-comparison grid (gated on Stage 5)

After F24/F25 land, can also run **Phase-2 substrate grid** at CA/TX (same 4-cell grid as F36 for FL: probe + cat STL × 2 + reg STL × 2 + MTL+HGI counterfactual). Adds ~5h × 2 = ~10h Colab T4.

Verdict on whether to run Phase-2 at CA/TX:
- **Required by acceptance** (PHASE2_TRACKER.md §6): CH16 cross-state at ≥2 of {FL, CA, TX} cat F1 paired Wilcoxon p<0.05.
- If FL Phase-2 (F36) confirms CH16 → CA Phase-2 is sufficient (≥2 of 3 = FL + CA, drop TX) → saves ~5h.
- **Recommend running CA Phase-2; defer TX Phase-2 unless reviewer asks.**

---

## Risk register

| Risk | Mitigation |
|------|------------|
| CA/TX checkins parquet not yet on disk | Verify before launching; if missing, source from Foursquare collection |
| Embedding training instability at scale | Stage 1 may need `--early-stopping-patience 20` if loss plateaus |
| FL frozen-cat reg-path instability (F49 caveat) → may replicate at CA/TX | Document if observed; doesn't block headline |
| Colab session 12h limit | Split Stage 5 across two sessions per state via per-fold checkpointing |
| SSD storage pressure (CA+TX add ~10 GB combined) | Boot-volume staging same as F49c protocol if SSD fills |

---

## Pre-launch checklist

- [ ] Verify CA + TX raw checkins on disk
- [ ] Verify TIGER census-tract files for CA + TX on disk
- [ ] Confirm F33 (Path A vs Path B) outcome → set cat-head accordingly
- [ ] Read `docs/COLAB_GUIDE.md` (detached-subprocess pattern)
- [ ] Bring up two Colab tabs (one per state) for parallel Stage 1
