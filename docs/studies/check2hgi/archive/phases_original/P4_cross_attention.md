# Phase P4 — Cross-attention (Option C, gated)

> **✅ IMPLEMENTED + TESTED ON AL (2026-04-18).** `MTLnetCrossAttn` built, registered as `mtlnet_crossattn`, and ablated against 6 other intervention families (see `FINAL_ABLATION_SUMMARY.md`). Key finding: cross-attention **closes the category gap to STL exactly** (cat F1 = 38.58 ± 0.98 vs STL 38.58 ± 1.23) while the **region gap persists** (45.09 vs 56.94 = −11.85 pp).
>
> This is the only architecture in our ablation that reaches STL-parity on the weaker head (category). It supports the paper's task-asymmetric framing (CH-M1 ... CH-M4 in `FINAL_ABLATION_SUMMARY.md`).
>
> **Remaining open questions (next steps logged below).** Phase transitions from "design + run" to "replicate + extend".

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

---

## ✅ Status 2026-04-18 — Implementation + AL ablation complete

**Implementation:** `src/models/mtl/mtlnet_crossattn/model.py` (registered as `mtlnet_crossattn`). 2 blocks × 4 heads, per-task FFN, bidirectional cross-attention between task encoder outputs. No parameter sharing across tasks.

**AL result** (5f × 50ep, seed 42, per-task modality, user-disjoint folds, GRU region head):

| | MTL cross-attn | STL (fair) | Δ |
|---|---:|---:|---:|
| cat macro-F1 | **38.58 ± 0.98** | 38.58 ± 1.23 | **0.00 pp (matches)** |
| reg Acc@10 | 45.09 ± 5.37 | 56.94 ± 4.01 (GRU) | −11.85 pp |
| reg MRR | 20.94 | 34.57 | −13.63 pp |
| Δm | −15.05% | 0 (ref) | |

Result file: `docs/studies/check2hgi/results/P2/ablation_06_crossattn_al_5f50ep.json`. Full ablation landscape in `FINAL_ABLATION_SUMMARY.md`.

**Paper-grade finding:** Cross-attention is the **first and only architecture in our 7-family ablation that reaches STL-parity on the weaker head (category)**. This asymmetry — helps weak head, doesn't help strong head — is the paper's central mechanistic claim (CH-M1).

## 🟡 Next steps (saved here 2026-04-18)

Ordered by expected value × cost:

### NS-1 — FL replication of cross-attention (high priority, ~2h compute)

**Question:** does the "cross-attn matches STL on cat" property hold at scale?

**Why it matters:** AL is our dev state (10 K rows). FL (127 K rows) is the headline. A single AL data point is noisy-bounded (cat std 0.98 on cross-attn). Need FL replication to include in the paper's main table.

**Command sketch:**
```bash
PYTHONPATH=src DATA_ROOT=... OUTPUT_DIR=/tmp/check2hgi_data \
  python -u scripts/train.py --task mtl --task-set check2hgi_next_region \
    --state florida --engine check2hgi --folds 1 --epochs 50 --seed 42 \
    --task-a-input-type checkin --task-b-input-type region \
    --model mtlnet_crossattn --mtl-loss pcgrad \
    --gradient-accumulation-steps 1 --no-checkpoints
```

**Expected wall time:** ~1.5–2 h (1 fold FL; 5 folds would be 8–10 h).

**Decision rule:**
- If FL cross-attn cat F1 ≥ FL STL fair cat 63.17: CH-M4 (cross-attn uniquely closes weaker-head gap) generalises. Strong paper claim.
- If FL cross-attn cat F1 < FL STL: AL was lucky; CH-M4 downgrades to AL-only. Paper adds a caveat.

### NS-2 — Hybrid (cross-attn cat branch + dselectk reg branch) (medium priority, ~3 h)

**Question:** can we combine the best of both — cross-attn's cat lift AND dselectk's reg capacity?

**Design sketch:** pass enc_cat and enc_next through 2 cross-attn blocks → then route the resulting representations through a DSelectK mixer → task heads. Effectively splits the MTL backbone into an "early content exchange" stage (cross-attn) and a "late per-task routing" stage (DSelectK).

**Expected lift:**
- cat: near STL-ceiling 38.58 (from cross-attn component)
- reg: approach dselectk+MTLoRA r=8 ceiling 50.72 (from DSelectK + MTLoRA component)
- Overall Δm: −12 to −13%, better than either component alone

**Implementation:** 1 new model class `MTLnetHybrid` inheriting from MTLnetCrossAttn and adding a DSelectK layer after the cross-attention stack. ~150 LOC + tests. Run AL 5f × 50ep ~25 min.

**Decision rule:**
- Hybrid reg Acc@10 > 51% AND cat F1 > 38 → new champion, promote to FL/CA/TX.
- Hybrid ≤ component baselines → no combination benefit; paper keeps cross-attn as the cat-closer and dselectk+MTLoRA as the reg-closest.

### NS-3 — Multi-seed cross-attention on AL (low priority, ~1.5 h)

**Question:** is the cat F1 match to STL within σ robust across seeds, or was seed 42 lucky?

**Runs:** cross-attn AL 5f × 50ep × {seed=42, 123, 2024}.

**Decision rule:** median across seeds confirms or downgrades CH-M4.

### NS-4 — Cross-attention block-count sweep (optional, ~2 h)

**Question:** does more cross-attention capacity (3-4 blocks) help, or does 2 saturate?

**Runs:** num_crossattn_blocks ∈ {1, 2, 3, 4} × AL 5f × 50ep.

**Decision rule:** monotonic improvement → scale to 4 in headline. Plateau at 2 → use 2 in paper.

### NS-5 — CA/TX replication for headline table (post-paper-draft)

Once AL results stabilise across seeds, replicate winning configs on CA and TX at 5f × 50ep × 3 seeds per state. Pure compute (no new code). ~20 h total.

---

## Recommended execution order

1. **NS-1** (FL replication) — validates the headline cat-closing claim at scale. Do first.
2. **NS-2** (Hybrid) — could push reg Acc@10 to the 51+% range, completing the "paper improves both tasks" story.
3. **NS-3** (multi-seed) — in parallel with NS-1/NS-2 if MPS has headroom.
4. **NS-4** — optional sensitivity analysis.
5. **NS-5** — gated on NS-1..3 outcomes; scaled up after draft is ready.
