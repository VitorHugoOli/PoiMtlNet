# Phase P3 — Dual-stream input (region embeddings as explicit parallel stream)

**Goal:** test whether feeding region embeddings alongside check-in embeddings at the input layer improves next-POI prediction. This is "Option A" from the prior critical-review work.

**Duration:** ~3h (2 states × 1 config × 5f × 50ep — same as P2 + a parquet materialisation step).

**Embedded claims tested:**
- CH06 — Region embeddings as input improve next-POI Acc@10.
- CH11 — Region-input gain is state-dependent (larger on FL than AL).

**Gates:** P2 complete; CH02 at least `partial`.

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
| P3.1.AL.dual | AL | dual-stream (128-dim per timestep) | NashMTL | CH06 on AL |
| P3.1.FL.dual | FL | dual-stream | NashMTL | CH06 on FL + CH11 |

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

### CH06 — Dual-stream helps

Per state:
- `dual_stream_next_poi_acc10` vs `check_in_only_next_poi_acc10` (from P2)

**Decision:**
- `confirm CH06` if dual > check-in-only by ≥ 2pp on at least one state, paired-test p < 0.05.
- `partial` if < 2pp but positive.
- `refute` if dual ≤ check-in-only on both states.

### CH11 — State-dependent gain

Compare:
- `Δ_AL = dual_AL_acc10 − check_in_only_AL_acc10`
- `Δ_FL = dual_FL_acc10 − check_in_only_FL_acc10`

**Decision:**
- `confirm CH11` if `|Δ_FL| ≥ 2 × |Δ_AL|` (FL gain is at least twice AL's).
- `partial` if same sign but smaller ratio.
- `refute` if AL gain ≥ FL gain.

This claim is built on the AL/AZ/FL linear-probe result (3.4× / 1.7× / 1.04× majority lift for check-in-emb → region linear recovery). If transformers extract signal linear probes miss, the state-dependent pattern may flatten — that's a finding in itself.

---

## Decision gate → P4

**P4 runs if and only if CH06 confirms with ≥ 2pp Acc@10 lift on FL specifically** (the state where the probe predicted the biggest gap).

If CH06 shows < 2pp on FL: document P4 (cross-attention) as future work, skip to P5.

---

## Output

```
docs/studies/check2hgi/results/P3/
├── P3.1.AL.dual/
├── P3.1.FL.dual/
└── ANALYSIS.md
```
