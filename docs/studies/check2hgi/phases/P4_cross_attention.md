# Phase P4 — Cross-attention (Option C, gated)

> **⚠️ STALE (2026-04-16).** Phase numbering shifted: this is now master-plan **P5** (gated on P4's per-task-modality outcome, not on a "dual-stream" CH03 threshold). The substantive idea — bidirectional cross-attention between the two task-specific encoders — survives but is reframed around per-task input modality rather than shared-input dual-stream. Authoritative plan: `docs/studies/check2hgi/MASTER_PLAN.md §P5`. Claims: `CLAIMS_AND_HYPOTHESES.md` (CH09). Read for historical framing only.

**Goal:** test whether bidirectional cross-attention between the check-in stream and the region stream improves over naive concatenation (P3). This is "Option C" from the prior critical-review work.

**Duration:** ~1 day implementation + ~6h training (2 states × 3 K values × 5f × 50ep if full, or a single K=2 run if tight).

**Embedded claim:** CH07.

**Gates:** P3 complete with CH03 confirming ≥ 2pp Acc@10 lift on Florida. If P3 fails the gate, this phase is documented as future work and **does not run**.

---

## What Option C means

Architecture: `MTLnetCrossAttn` (new class, separate from `MTLnet` — legacy untouched).

```
x_checkin  [B, 9, 64]  → EncoderA → h_c [B, 9, D]
x_region   [B, 9, 64]  → EncoderB → h_r [B, 9, D]       D = shared_layer_size (256)

For ℓ = 1..K cross-attention layers:
    h_c ← h_c + MultiHeadAttn(Q=h_c, K=h_r, V=h_r) + FFN(h_c)
    h_r ← h_r + MultiHeadAttn(Q=h_r, K=h_c, V=h_c) + FFN(h_r)

next_poi head    ← NextHeadMTL(h_c)  → [B, n_poi]
next_region head ← NextHeadMTL(h_r)  → [B, n_region]
```

2 heads preserved. Loss: same NashMTL over two CEs. Parameter budget: ~1.3× baseline MTLnet.

---

## Experiments

| # | K (cross-attn layers) | State | Purpose |
|---|---|---|---|
| P4.1.FL.K2 | 2 | FL | Headline — CH07 |
| P4.1.AL.K2 | 2 | AL | CH07 replication |
| P4.2.FL.K1 (optional) | 1 | FL | Depth ablation (shallower) |
| P4.2.FL.K3 (optional) | 3 | FL | Depth ablation (deeper) |

If compute is tight, run only K2 on both states.

---

## Prerequisites (code deltas)

1. **`src/models/mtl/mtlnet_cross_attn/model.py`** — new registered model `mtlnet_cross_attn`. Separate class from `MTLnet`. ~200 LOC.
2. **`src/models/mtl/_components.py`** — `CrossAttentionBlock(nn.Module)`. ~50 LOC.
3. **`FoldCreator`** — already handles the dual-stream input from P3; reused here.
4. **New preset `CHECK2HGI_CROSS_ATTN`** — task-set with both heads sequential + model_name="mtlnet_cross_attn".
5. **`scripts/train.py`** — route the new preset / model_name through.
6. **Tests:** `tests/test_models/test_mtlnet_cross_attn.py` (forward, param budget, backward).

Total: ~1 day implementation.

---

## Risks

- **Parameter/data ratio** — with ~2M params total on Alabama's 12k sequences, over-fitting is plausible. Monitor val-train gap.
- **Attention collapse** — cross-attention on small datasets can collapse to trivial patterns. If training is unstable, plan a warm-start: initialise from the P2 MTLnet champion weights (shared backbone) + fresh cross-attention layers with identity init.
- **If Option A (P3) showed no gain, Option C is very unlikely to gain either.** That's why the gate exists.

---

## Analysis

### CH07 — Cross-attention > concat

Paired comparison: `mtlnet_cross_attn_next_poi_acc10` vs P3's `dual_concat_next_poi_acc10`, FL primary.

**Decision:**
- `confirm CH07` if cross-attn > concat by ≥ 1pp on FL, p < 0.05.
- `partial` if trend positive but < 1pp.
- `refute` if cross-attn ≤ concat.

Lower threshold than P3's 2pp because cross-attention is a strictly more expressive family over dual-stream concat — any significant gain is meaningful.

---

## Paper table row

Headline experimental table adds:

| Setup | AL next_poi Acc@10 | FL next_poi Acc@10 | AL next_poi MRR | FL next_poi MRR |
|---|---|---|---|---|
| HGI single-task | … | … | … | … |
| Check2HGI single-task | … | … | … | … |
| Check2HGI MTL (P2) | … | … | … | … |
| Check2HGI MTL + dual input (P3) | … | … | … | … |
| **Check2HGI MTL + cross-attn (P4)** | … | … | … | … |

Bolded row = Pareto winner.

---

## Output

```
docs/studies/check2hgi/results/P4/
├── P4.1.AL.K2/
├── P4.1.FL.K2/
├── P4.2.FL.K1/   (optional)
├── P4.2.FL.K3/   (optional)
└── ANALYSIS.md
```
