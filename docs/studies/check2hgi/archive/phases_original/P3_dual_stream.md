# Phase P3 — Dual-stream input (region embeddings as explicit parallel stream)

> **⚠️ STALE (2026-04-16).** The "dual-stream concat" design this file describes has been superseded by **per-task input modality** (check-in → category_encoder, region → next_encoder) — P1 evidence showed concat is strictly worse than region-only for the region head. Phase numbering also shifted: this is now master-plan **P4**; the concat variant is one of 4 ablation arms, not the headline. Authoritative plan: `docs/studies/check2hgi/MASTER_PLAN.md §P4`. Claims: `CLAIMS_AND_HYPOTHESES.md` (CH03 per-task modality; CH08 state-dependent gain). Read for historical framing only.

**Goal:** test whether feeding region embeddings alongside check-in embeddings at the input layer improves next-POI prediction. This is "Option A" from the prior critical-review work.

**Duration:** ~3h (2 states × 1 config × 5f × 50ep — same as P2 + a parquet materialisation step).

**Embedded claims tested:**
- **CH03** — Dual-stream region embeddings as input improve next-POI Acc@10 (Tier-A, independent of MTL).
- CH08 — Region-input gain is state-dependent (larger on FL than AL).

**Gates:** P2 complete. Note: P3 is run even if CH01 (MTL) refutes — CH03 is orthogonal (tests whether region as *input* helps, regardless of whether region as *auxiliary task* helps). If both CH01 and CH03 refute, the paper reframes around methodology-only.

**Multi-seed:** 3 seeds × 5 folds × 2 states = 30 runs per variant. Matches the P2 n=15 paired-samples protocol.

---

## What dual-stream means

At each of the 9 timesteps in the sequence window, the per-timestep input becomes:

```
x_t = concat(
    checkin_emb[t],                                           # [64] from check2hgi embeddings
    region_emb[ poi_to_region[ checkin_to_poi[t] ] ]          # [64] from region_embeddings.parquet
)                                                              # → [128] per timestep
```

Input shape `[B, 9, 128]`. MTLnet's encoder first-layer Linear adapts automatically; FiLM and shared backbone are dim-agnostic. No architecture change — only data-pipeline change.

---

## Experiments

| # | State | Input | Optimiser | Purpose |
|---|---|---|---|---|
| P3.1.AL.dual | AL | dual-stream (128-dim per timestep) | NashMTL | CH03 on AL |
| P3.1.FL.dual | FL | dual-stream | NashMTL | CH03 on FL + CH08 |

Comparison baselines: P2.1.AL and P2.1.FL (check-in-only, 64-dim per timestep).

---

## Prerequisites (code deltas)

1. **`src/data/inputs/next_poi_region_dual.py`** — new loader. Reuses `sequences_next.parquet` + `checkin_graph.pt` but joins region_embeddings per timestep. Output parquet: `output/check2hgi/<state>/input/next_poi_dual.parquet` with columns `0..1151 + poi_idx + userid` (1152 = 9 × 128).
2. **`pipelines/create_inputs_check2hgi_dual.pipe.py`** — wrapper.
3. **Option A preset** — a new TaskSet `CHECK2HGI_NEXT_POI_REGION_DUAL` that adds `head_params={"dual_stream": True}` or similar marker, OR a CLI flag `--dual-stream`. The cleaner path: CLI flag that routes the loader from the dispatcher.
4. **`FoldCreator._create_check2hgi_mtl_folds`** — accepts a `dual_stream: bool` flag; loads from `next_poi_dual.parquet` when True.
5. **Unit tests:** `tests/test_data/test_next_poi_region_dual_loader.py`.

Estimate: ~3h code work.

---

## Analysis

### CH03 — Dual-stream helps (Tier-A)

Per state:
- `dual_stream_next_poi_acc10` vs `check_in_only_next_poi_acc10` (from P2)

**Decision:**
- `confirm CH03` if dual > check-in-only by ≥ 2pp on at least one state, Wilcoxon p < 0.05 on 15 paired samples.
- `partial` if 1–2pp positive or p < 0.10.
- `refute` if dual ≤ check-in-only on both states.

### CH08 — State-dependent gain

Compare:
- `Δ_AL = dual_AL_acc10 − check_in_only_AL_acc10`
- `Δ_FL = dual_FL_acc10 − check_in_only_FL_acc10`

**Decision:**
- `confirm CH08` if `|Δ_FL| ≥ 2 × |Δ_AL|` (FL gain is at least twice AL's).
- `partial` if same sign but smaller ratio.
- `refute` if AL gain ≥ FL gain.

This claim is built on the AL/AZ/FL linear-probe result (3.4× / 1.7× / 1.04× majority lift for check-in-emb → region linear recovery). If transformers extract signal linear probes miss, the state-dependent pattern may flatten — that's a finding in itself.

---

## Decision gate → P4

**P4 runs if and only if CH03 confirms with ≥ 2pp Acc@10 lift on FL specifically** (the state where the probe predicted the biggest gap).

If CH03 shows < 2pp on FL: document P4 (cross-attention) as future work, skip to P5.

---

## Output

```
docs/studies/check2hgi/results/P3/
├── P3.1.AL.dual/
├── P3.1.FL.dual/
└── ANALYSIS.md
```
