# F62 — T5.2b Masked POI feature reconstruction (implementation notes)

**Status**: implementation landed; awaiting AL+AZ multi-seed run.
**Date**: 2026-05-17
**Branch**: `worktree-agent-a3bb7f7bb217a001f`
**Spec**: `docs/studies/canonical_improvement/INDEX.html` §T5.2b (lines ~1700-1730)
**Pair**: T4.1 (falsified, AL+AZ 2026-05) — see "paired-falsification value" below

---

## TL;DR

T5.2b adds an auxiliary **POI-level masked-reconstruction head** to Check2HGI:
mask 15 % of POIs (zero their pooled embedding) and have a tiny MLP decoder
reconstruct each masked POI's **feature aggregate** (mean check-in category
one-hot, or log visit count, or both) from the masked POI's **Delaunay POI-POI
neighbours' pooled embeddings**. Auxiliary SCE loss added with coefficient
λ_mae_poi (default 0.3 when enabled). Default OFF — canonical reproduces
bit-for-bit.

---

## T5.2b vs T4.1 — paired-falsification structure (CRITICAL)

T4.1 was **falsified** at AL+AZ (5 seeds, 0 / 5 positive). The natural
follow-up question is: "did the failure trace to the *masking* idea, or to
*masking at the check-in level* specifically?". T5.2b answers that by
operating at a **different hierarchy** via a **different mechanism**:

| Axis | T4.1 (falsified) | T5.2b (this finding) |
|------|---|---|
| **What is masked** | check-in INPUT features `data.x` (11-d cat one-hot + 4-d temporal) | post-pool POI embedding (D=64), zeroed out |
| **Graph used for context** | the check-in graph (`data.edge_index`: user-sequence edges) — passes through `checkin_encoder` again with masked input | POI-POI Delaunay graph (geographic neighbours) — never re-runs encoder |
| **Decoder input** | `masked_emb` from a SECOND encoder forward pass | **mean-aggregated Delaunay POI neighbour embeddings** (single-step gather) |
| **Decoder target** | the original 15-dim `data.x` row | per-POI mean category one-hot (or log visit count, or both) |
| **Loss** | SCE (γ=3) on the masked check-in features | SCE (γ=3) on the masked POI's feature aggregate |
| **Compute cost** | +1 encoder forward per step (≈+35 % wall-time) | +1 gather + 2 MLP layers (≈+2 % wall-time) |
| **Mechanism tested** | node-level feature reconstruction lifts | **neighbourhood-aggregate prediction** lifts |
| **Hypothesis origin** | GraphMAE (Hou et al. 2022) | HGI 2018 ablation: the POI-masked-recon objective carried the recipe lift, NOT the Node2Vec walks (per spec) |

**The T4.1 falsification does NOT transfer to T5.2b.** Different graph, different
mechanism, different compute envelope. Both could be true (only T5.2b lifts),
both could be false (paired-falsification value = "HGI's POI-recon recipe
does not transfer to a check-in substrate"), or T5.2b could lift while T4.1
did not (the lift was about *neighbourhood-aggregate prediction*, not
node-level reconstruction).

If T5.2b also falsifies at AL+AZ × 5 seeds, the paired finding is
**publishable**: it closes a 2-cell ablation that was previously open.

---

## Architecture

```
Check-in graph ──► CheckinEncoder ──► Checkin2POI pool ──► pos_poi_emb (P, D)
                                                                 │
                                                                 ├──► [canonical] POI2Region → contrastive p2r/r2c
                                                                 │
                                                                 └──► [T5.2b auxiliary]
                                                                       │
                                                                       ▼
                                                          1. Sample mask: rand(P) < 0.15
                                                          2. poi_emb_masked = pos_poi_emb.clone()
                                                             poi_emb_masked[mask] = 0
                                                          3. Aggregate Delaunay neighbours
                                                             (mean or GCN-norm)
                                                          4. Decode (MLP D → 2D → target_dim)
                                                          5. SCE / MSE on masked POIs vs target
                                                          + λ_mae_poi × L_recon → total loss
```

Decoder MLP: `Linear(D=64, 128) → PReLU → Linear(128, target_dim)`.
For `target_kind="category_aggregate"` and AL the target_dim = num_categories
≈ 11, so the decoder adds ~10 k params (well under any guardrail).

The decoder reads the **pre-augmentation** pooled POI embedding (`pos_poi_emb_pure`)
— mirroring the T4.3 isolation pattern so side-feature gradients never reach
the masked-POI decoder, and vice versa.

---

## Reconstruction target safety (leak audit)

Three target options, all derived from the encoder's INPUT:

1. **`category_aggregate`** (default) — per-POI mean of constituent check-ins'
   category one-hot. Information is already in `data.x` (each check-in's
   category one-hot, indexed by `checkin_to_poi`). The decoder adds STRUCTURE
   (it must predict from neighbours when the POI itself is masked) but NO
   new information.
2. **`visit_count_log`** — log1p of the number of check-ins at each POI.
   Derivable from `checkin_to_poi` alone (count = `bincount`).
3. **`both`** — concatenation of the above two.

**Critical distinction from fclass leak (CONCERNS.md C18 stack-watch):**
the check-in category one-hot is a per-VISIT category from the raw
foursquare data and is a *known input feature*. **fclass** is the
linear-probe target — a downstream label used to MEASURE leak. They share a
vocabulary but the encoder never sees fclass; the leak probe re-derives
fclass from the embeddings and trains a logreg. Reconstructing
`category_aggregate` does not import fclass into training. The
**user-held-out IJM probe** (`scripts/canonical_improvement/ijm_leak_probe.py`)
remains the empirical guardrail.

**Leak risk class for T5.2b**: same as T4.1 (POI-level masking + neighbour-based
reconstruction). Expected category-aggregate-target leak delta: small
(< +3 pp). Track on every multi-seed run; the +5 pp red flag from T1.1 / F50
applies.

---

## Composability with T5.1 / T5.2a / T5.3

- **Delaunay preprocess (shared with T5.2a)**: both T5.2a (Node2Vec random
  walks on POI-POI Delaunay) and T5.2b (neighbour aggregation for the
  decoder) consume the same `poi_delaunay_edge_index` artefact. The
  preprocess function exposes a single `build_poi_delaunay=True` flag.
  When agents merge, **deduplicate `_build_poi_delaunay_edges`**: both
  this branch (T5.2b) and the parallel T5.2a branch added a near-identical
  method. Pick one canonical implementation.
- **POI aggregates (T5.2b only)**: `poi_category_aggregate` and
  `poi_visit_count_log` are gated behind a SEPARATE flag
  `build_poi_aggregates=True`, so T5.2a does not pay for them.
- **T5.1 / T5.3**: T5.2b touches only the POI level. As long as T5.1 / T5.3
  are layered at check-in or region level, they compose additively.
  Multi-test enrolment: `--use-mae-poi --mae-poi-lambda 0.3` can stack
  with T3.2 ResLN / T1.5 hygiene / T2.x DropEdge.

---

## Phase 1 lessons applied

- **No FL-only ships**: T5.2b is spec-targeted at AL+AZ (small states) per
  the canonical_improvement INDEX. The decision rule: 5 seeds at AL AND 5
  seeds at AZ ≥ +30 pp fclass-probe and ≥ +1 σ in IJM-held-out probe to
  ship. FL is a sanity-check at a 6th seed, not a launch signal.
- **Multi-seed mandatory**: seeds = `42, 7, 1, 100, 0` (canonical_improvement
  pattern). Each seed gets a freshly-rebuilt graph (force_preprocess=True)
  so the mask-sample randomness is also seed-controlled (the
  `MaskedPOIDecoder` consumes `torch.rand` from the global PRNG when no
  explicit generator is passed; per-fold splits are NOT relevant here
  because the SSL stage is graph-level, not fold-level).
- **Explicit reg-axis kill criterion**: T5.2b must not regress next-POI
  Acc@10 by more than 1 pp vs canonical at AL or AZ. Tracked in the
  multi-seed comparison sheet.

---

## Gotchas

1. **POIs with zero Delaunay degree** (sparse-state outskirts): the
   neighbour aggregate falls back to the POI's own (zeroed) embedding,
   producing a degenerate all-zero decoder input. SCE clamp(min=0) +
   EPS guard in the cosine call prevent NaN. For affected POIs the gradient
   contribution is small but defined. Document in F62 — DO NOT add a
   special-case "skip" path because that breaks autograd shape.
2. **`force_preprocess` toggling**: enabling `--use-mae-poi` forces a
   preprocess rebuild (cached canonical graphs lack the new artefacts).
   `create_embedding` also auto-detects via cache-peek, but the regen
   script sets it explicitly to short-circuit.
3. **Visit-count-log + SCE**: cosine similarity on a scalar collapses to
   the sign indicator (≈ always +1 for log values ≥ 0). The CLI
   auto-switches to MSE when `target_kind="visit_count_log"`; document in
   the help text.
4. **Mask token convention**: T4.1 uses a *learned* `[MASK]` parameter at
   the check-in input. T5.2b deliberately uses a **zero** mask token so
   the decoder cannot read a learned-token bias at the masked slot —
   forcing it to derive the prediction from neighbours. This is a
   conscious design departure from T4.1.

---

## Files modified

| File | Change |
|---|---|
| `research/embeddings/check2hgi/preprocess.py` | Added `_build_poi_delaunay_edges()`, `_compute_poi_feature_aggregates()`, two new `build_*` flags, `get_data` emits optional artefacts. |
| `research/embeddings/check2hgi/model/variants.py` | Added `MaskedPOIDecoder` class (decoder MLP + mean/gcn aggregator + SCE/MSE loss). |
| `research/embeddings/check2hgi/model/Check2HGIModule.py` | Added `mae_poi_*` init args, decoder construction (gated), forward-pass loss accumulator, `loss()` adds `λ_mae_poi · L_mae_poi`. |
| `research/embeddings/check2hgi/check2hgi.py` | Plumbed args; loads/attaches `poi_delaunay_edge_index` + `poi_recon_target` to `data`; auto-rebuilds cache if artefacts missing. |
| `scripts/canonical_improvement/regen_emb_t3.py` | Added 7 CLI flags (`--use-mae-poi`, `--mae-poi-lambda`, `--mae-poi-mask-rate`, `--mae-poi-target`, `--mae-poi-gamma`, `--mae-poi-aggr`, `--mae-poi-loss-kind`); flips `force_preprocess` when `--use-mae-poi`. |
| `tests/canonical_improvement/test_encoders.py` | Added `test_masked_poi_decoder()` covering finite SCE, backward, mask_rate=0 no-op, seed-reproducible mask, MSE branch, empty-edges fallback. |

---

## Sample CLI for multi-seed run (AL+AZ, λ ∈ {0.1, 0.3})

```bash
for STATE in alabama arizona; do
  for SEED in 42 7 1 100 0; do
    for LAMBDA in 0.1 0.3; do
      SEED=$SEED python scripts/canonical_improvement/regen_emb_t3.py \
        --state $STATE --epoch 500 \
        --use-mae-poi --mae-poi-lambda $LAMBDA \
        --mae-poi-mask-rate 0.15 --mae-poi-target category_aggregate \
        --mae-poi-aggr mean --mae-poi-loss-kind sce \
        --weight-decay 0.05 --scheduler cosine --warmup-pct 0.1
      # … then run fclass probe + IJM probe + downstream MTL
    done
  done
done
```

Combine with T3.2 ResLN encoder (`--encoder resln`) for the stacked variant
once the standalone T5.2b lands its result.

---

## Open questions for the audit advisor

1. Should the decoder also see `poi_to_region` (one-hot or learned region
   embedding) as side-info? Currently it does not — pure neighbour aggregate.
   Adding region would lift compute by ~5 % and might shortcut the
   neighbourhood signal.
2. The Delaunay graph is built ONCE at preprocess time and shared with
   T5.2a. Should T5.2b use a different POI graph (e.g., k-NN k=8) to
   avoid coupling? The decision tree in INDEX.html says no — shared
   substrate is a feature, not a bug.
3. Symmetric GCN normalisation in `_aggregate_neighbours` currently uses
   the right-normalised dst form for "gcn" mode but only approximates
   left normalisation via per-src scaling. For dense Delaunay graphs
   (~6 neighbours / POI) the difference vs textbook D⁻¹ᐟ² A D⁻¹ᐟ² is small.
   Track if "gcn" is needed at all — the default "mean" may be enough.
