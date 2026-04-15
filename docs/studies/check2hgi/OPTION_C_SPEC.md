# Option C — Hierarchical cross-attention (detailed spec)

Answers the three questions from your review:

1. **Do we keep the 2 heads?** → **Yes.** Both `next_category` and `next_region` heads stay; they read from task-specific post-attention streams.
2. **How does MTL work?** → Same loss topology (NashMTL over two cross-entropies). The *sharing* moves from a single post-encoder backbone into cross-attention between two parallel streams.
3. **Closer to SOTA than A/B?** → **Yes.** It replicates the information-flow pattern of HMT-GRN / Bi-Level GSL (cross-level attention) while staying inside our 2-task MTL framing.

---

## 1. Inputs

Dual-stream, one stream per granularity (same 9-window).

```
x_checkin : [B, 9, 64]   sequence of check2HGI check-in embeddings (as today)
x_region  : [B, 9, 64]   sequence of region embeddings, via
                         checkin_t → poi_t → region_emb[poi_to_region[poi_t]]
```

The region sequence is cheap to build — lookup table of size `n_regions × 64`, one gather per position.

## 2. Architecture

```
                 ┌──────────────────────────┐
                 │   Per-stream encoders    │
                 │  (nn.Linear stack, as    │
                 │  today but two copies)   │
                 └──────────────────────────┘
    x_checkin ──►  EncoderA ──► h_c  [B, 9, D]   (D = shared_layer_size, e.g. 256)
    x_region  ──►  EncoderB ──► h_r  [B, 9, D]

                 ┌──────────────────────────────────────────┐
                 │   K cross-attention layers (new block)    │
                 │   (K = 2 in the default)                  │
                 └──────────────────────────────────────────┘
    For ℓ = 1..K:
      h_c ← h_c + MultiHeadAttn(Q=h_c, K=h_r, V=h_r) + FFN(h_c)
      h_r ← h_r + MultiHeadAttn(Q=h_r, K=h_c, V=h_c) + FFN(h_r)
    Residual + LayerNorm as usual (norm_first=True).

                 ┌──────────────────────────┐
                 │    Task heads (2 heads)  │
                 └──────────────────────────┘
    Task A (next_category): NextHeadMTL on h_c  → [B, 7]
    Task B (next_region)  : NextHeadMTL on h_r  → [B, n_regions]
```

Every layer is standard `nn.TransformerEncoderLayer`-class blocks reused; the novel glue is the cross-attention pair.

## 3. Loss / MTL semantics

- Per-head cross-entropy (same as today).
- `losses = torch.stack([region_loss, category_loss])` (same slot ordering).
- MTL criterion: `NashMTL(n_tasks=2)` — unchanged.

The "multi-task" sharing moves:

| Variant | Where sharing happens |
|---|---|
| Current (`MTLnet` shared-backbone) | Per-task encoders → **FiLM** → **single shared residual backbone** → per-task heads |
| Option A (dual-stream concat) | Per-task encoders receive a 128-dim input (check-in ⊕ region) → **shared backbone** → per-task heads |
| **Option C (cross-attention)** | Per-task encoders → **cross-attention block (bidirectional between streams)** → per-task heads |

Each variant is still a 2-task MTL with NashMTL. Option C's cross-attention is strictly more expressive than A's concatenation (concatenation is a special case where attention weights collapse to identity).

## 4. What changes in the codebase

| File | Change | Lines (est.) |
|---|---|---|
| `src/models/mtl/mtlnet_c2hgi/model.py` (new) | `MTLnetCrossAttn` registered as `"mtlnet_c2hgi"`. Separate class — legacy `MTLnet` untouched. | ~200 |
| `src/models/mtl/_components.py` | Add `CrossAttentionBlock(nn.Module)` — Q from one stream, K/V from the other, FFN + LayerNorm | ~50 |
| `src/data/inputs/next_region_dual.py` (new) | Build dual-stream input parquet with region embeddings joined per position | ~100 |
| `pipelines/create_inputs_check2hgi.pipe.py` | Add `--dual-stream` flag | ~20 |
| `src/tasks/presets.py` | Add `CHECK2HGI_DUAL_STREAM_NEXT_REGION` preset — `is_dual_stream=True` flag on TaskConfig | ~30 |
| `src/training/runners/mtl_cv.py` | Handle `is_dual_stream` — the forward pass receives two tensors per task slot, not one | ~20 |
| `scripts/train.py` | Route `--task-set check2hgi_dual_stream_next_region` to the new model | ~15 |

**Total:** ~430 LOC. ~1 day of focused work.

## 5. Preserves legacy?

Yes. `MTLnetCrossAttn` is a separate registered model class. Legacy MTLnet + `LEGACY_CATEGORY_NEXT` preset + existing tests are untouched. Activation is opt-in via the new CLI flag.

## 6. Expected BRACIS impact

Claims added (per `CLAIMS_AND_HYPOTHESES.md`):

- **CH12 — Cross-attention between check-in and region streams improves next-region Acc@1 and next-category F1 vs flat MTLnet on check2HGI.** (Ablation row.)
- **CH13 — Dual-stream input (region embeddings as explicit input) improves over check-in-only input at equal architecture.** (Ablation row — compares Option A and current.)

A paper table row comparing:
  1. Current: `MTLnet` + check-in only (baseline)
  2. Option A: `MTLnet` + concat(check-in, region)
  3. Option C: `MTLnetCrossAttn` + (check-in stream, region stream)

Makes the contribution story richer: "we studied three increasing levels of region-information integration."

## 7. Risks

- **Parameter budget.** Cross-attention adds ~`2 × K × 4 × D² ≈ 500k` params on top of the current ~1.5M. Alabama has 12k training sequences — param/data ratio gets uncomfortable. Risk of overfitting. Monitor val F1 vs train F1 gap.
- **Convergence.** Cross-attention can collapse to trivial attention patterns on small datasets. Plan a warm-start ablation (start from current MTLnet weights, add cross-attn layers with identity init) if training is unstable.
- **Inference cost.** Roughly 2× FLOPs of current MTLnet. Still fast; not a deployment concern for a research paper.
- **If Option A shows nothing, Option C is unlikely to help.** Both test "is more region info useful?". Recommendation: Option A first (cheap, ~2h), then C if A shows lift.

## 8. Decision point — when to commit to Option C

After running Option A:

- **Option A adds > 2% Acc@1 on next_region:** signal that region info helps → proceed to Option C for the paper.
- **Option A adds 0–2%:** marginal; Option C unlikely to do better unless the cross-attention captures non-concatenable structure (unlikely on 12k-sequence dataset).
- **Option A adds nothing:** check-in embeddings already encode region. Neither A nor C worth pursuing; paper thesis stays on the narrow "embedding + auxiliary task" story.

## 9. In the experiment spectrum

| Granularity of region info in the model | Inference time | Training time | Params |
|---|---|---|---|
| Current (implicit via check-in emb) | 1× | 1× | 1× |
| Option A (concat input) | 1.05× | 1.05× | 1.02× |
| **Option C (cross-attention)** | ~2× | ~2× | ~1.3× |

---

## TL;DR

**Option C = dual-stream input + bidirectional cross-attention between streams + keep the 2 heads + same NashMTL loss.** It's the SOTA-aligned variant of the plan. ~1 day of work. Recommended gated on Option A showing lift first.
