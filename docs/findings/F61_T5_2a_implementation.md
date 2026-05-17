# F61 — T5.2a Joint Node2Vec POI-POI skip-gram: Implementation Notes

**Status**: Scaffolding complete, pending execution.
**Date**: 2026-05-17
**Spec**: `docs/studies/canonical_improvement/INDEX.html#T5-2a`
**Sister F-trail entries**: F60 (T5.1 ID embedding, sister agent), F62 (T5.2b masked POI), F63 (T5.3 multi-view).

## What was built

Joint Node2Vec POI-POI skip-gram as a 4th auxiliary loss on top of c2hgi's
3 boundaries (c2p, p2r, r2c). No HGI artifact imported; the Delaunay graph
is built from scratch under the c2hgi training loop. No fclass L2
regularizer (spec gate against tautological leak).

### Loss formulation

```
L_total = α_c2p · L_c2p + α_p2r · L_p2r + α_r2c · L_r2c   (canonical)
        + λ_n2v · L_skipgram                              (T5.2a addition)

L_skipgram = - mean_{(u, v) ∈ walks}  [ log σ(z_u · z_v)
                                      + Σ_{u' ∈ NEG(v)} log σ(-z_u · z_{u'}) ]
```

where `z_*` is read from a **separate `nn.Embedding(N_poi, D)`** (default) or
the T5.1 ID table (opt-in via `--n2v-share-table-with-poi-id`).

## Files touched

| File | Purpose |
|------|---------|
| `research/embeddings/check2hgi/preprocess.py` | New `_build_poi_delaunay_edges()` + `build_poi_delaunay` flag → caches `poi_delaunay_edge_index` in graph pickle. |
| `research/embeddings/check2hgi/model/variants.py` | New `Node2VecPOIHead(nn.Module)`: holds POI table, generates Node2Vec walks once per epoch, computes skip-gram contrastive loss. |
| `research/embeddings/check2hgi/model/Check2HGIModule.py` | New `n2v_lambda` ctor arg + `attach_node2vec_head()` + `set_n2v_epoch()` + folded-in `L_total += n2v_lambda * head.compute_loss()`. |
| `research/embeddings/check2hgi/check2hgi.py` | Construct head with cached POI Delaunay edges, register as submodule (params flow into optimizer), bump epoch id at top of each epoch, gated preprocess re-run when cache lacks the new key. |
| `scripts/canonical_improvement/regen_emb_t3.py` | 8 new CLI flags (defaults preserve canonical); plumbs effective `n2v_lambda` into `cfg`. |
| `tests/canonical_improvement/test_encoders.py` | New `test_node2vec_poi_head()` with 4 checks (a-d). |

## Design decisions

1. **Separate POI table by default.** T5.1 (sister agent) is exposing a
   per-POI ID embedding under `model.poi_id_embedding`. T5.2a defaults to
   its own `nn.Embedding(N_poi, D)` so the skip-gram signal is **isolated**
   from T5.1's identity-slot signal. `--n2v-share-table-with-poi-id` lets
   the audit compare both modes. If shared, the table is registered only
   on the T5.1 module (no double-counting in optimizer).

2. **Skip-gram operates on a separate table, NOT on the post-pool c2hgi POI
   embedding.** Two reasons:
   * The c2hgi POI embedding is the output of `Checkin2POI(checkin_emb, ...)`
     which depends on the live encoder state — feeding skip-gram off it
     creates a recursive gradient path (encoder ← skip-gram ← encoder)
     that empirically destabilises GraphMAE-style auxiliaries (T4.1 dev
     log, 2026-05-16).
   * Spec methodology step 2 explicitly mentions "Add a learnable POI
     table", suggesting decoupling from the live POI embedding.
   * The auxiliary still couples to c2hgi *indirectly* via shared
     optimizer steps and (optionally) the shared T5.1 table.

3. **Walks regenerated once per epoch** (spec: "one batch of walks per
   epoch"). The cached walks tensor lives on the head as
   `_walks_cache`; `set_n2v_epoch(epoch_id)` invalidates it. We use
   `torch_geometric.nn.Node2Vec` for the walker (verified installed in
   this env); no manual walker fallback was needed.

4. **Delaunay graph at POI level only**, NOT check-in level. This is the
   explicit constraint from the spec, and is informed by T4.4's closure
   note: check-in-level Delaunay over-smoothed. POI-level edges keep the
   skip-gram signal at the right granularity (POI ↔ POI).

5. **No fclass L2 regularizer.** Spec gate: would be tautological leak
   into the fclass probe. Pure structural skip-gram only.

## Composability with other Tier-5 candidates

| Combination | Behavior |
|-------------|----------|
| T5.2a alone | New POI table; encoder untouched; +1 loss term. |
| T5.2a + T5.1 (`--n2v-share-table-with-poi-id=False`, default) | Two disjoint POI tables; signals stay separable for ablation. |
| T5.2a + T5.1 (`--n2v-share-table-with-poi-id=True`) | Shared table; skip-gram gradients flow into the T5.1 identity embeddings. Warning printed if T5.1 not enabled (auto-fallback to separate). |
| T5.2a + T5.2b | Independent (T5.2b is encoder-side masked recon; T5.2a is POI-side skip-gram). No coupling expected. |
| T5.2a + T5.3 (multi-view) | T5.3 wrapper builds two Check2HGI instances; T5.2a's POI Delaunay graph applies to View 1 only (View 2 has same_poi edges; spatial Delaunay over View 2's POIs is the same — could be re-used in a future iteration). Per T5.3 author note, no error expected. |
| T5.2a + T4.1 (GraphMAE) | Both add λ-weighted aux losses; defaults stack. Recommended ablation: T5.2a alone vs T5.2a + T4.1 to attribute lift. |

## Leak class analysis

* **POI graph structure** (Delaunay over lat/lon) is positionally
  informative but NOT fclass-labelled. Two POIs of different fclass that
  happen to be spatial neighbours will be pulled together in skip-gram
  space — this is the intended mechanism (vs HGI's POI2Vec which used
  fclass-level walks → hard leak).
* **Random negatives** are uniform over the POI pool, not stratified by
  fclass — no class supervision.
* **No fclass L2** (vs HGI poi2vec.py's hierarchical L2) — the only
  category-label path that made POI2Vec's 98% fclass leak is OUT.
* **User-held-out IJM probe** (the standard c2hgi leak probe) operates
  at the user-sequence level. T5.2a's skip-gram does not introduce a new
  user-level leak path because walks are over POI-POI structural edges
  (no user identity in the graph). Probe should remain valid.
* **fclass-probe lift attribution**: any +fclass lift from T5.2a is
  attributable to spatial-co-location semantics (POIs near each other
  tend to share fclass when zoning is consistent) — a legitimate mechanism
  per spec, distinct from the leak path.

## Phase 11 S3-b V2-c watch-out

The prior failure mode (AL pool collapse) was for per-POI mechanisms
where the c2p discriminator collapsed when the POI signal became too
deterministic. T5.2a is structurally different — it operates on a
SEPARATE POI table that does not enter the c2p discriminator path. So
the collapse failure mode should NOT trigger. However:

* The OPTIMIZER still couples both tables (via shared AdamW state on the
  total loss). At very high λ_n2v the skip-gram signal could dominate
  early training and starve the c2hgi boundaries of gradient signal.
* **Watch**: training-loss curve per boundary at AL. If `loss_c2p`
  plateaus early while `loss_skipgram` continues dropping, λ_n2v is too
  high. Spec sweep range {0.1, 0.3, 1.0} should bracket this.

## Phase 1 lessons applied

* **No FL-only ships**: scaffolding supports any state; spec mandates
  AL+AZ for primary, FL/CA/TX for paper-grade follow-up.
* **Multi-seed mandatory**: `regen_emb_t3.py` already honours `$SEED`
  env var for SSL encoder reproducibility (Tier-3 audit 2026-05-16).
  T5.2a inherits this — every seed reseeds both the c2hgi encoder AND
  the Node2Vec POI table init.
* **Explicit reg-axis kill criterion**: spec success bar is "≥ +40 pp
  fclass probe at AL+AZ AND no AL catastrophic collapse". Bonus on
  reg Acc@10. If reg drops by > 3 pp at AL → kill the recipe.

## Validation evidence

| Check | Result |
|-------|--------|
| Unit test `test_node2vec_poi_head()` | PASS (loss finite & >0; backward grads OK; empty-graph no-op = 0; init probe F1 = 0.150 vs chance 0.25) |
| All prior `test_encoders.py` tests | PASS (untouched) |
| Canonical CLI default (`regen_emb_t3.py --state alabama --epoch 5`) | `n2v_lambda=0.0, force_preprocess=False` — bit-identical to pre-T5.2a behavior |
| T5.2a CLI (`--use-node2vec-poi --n2v-lambda 0.3`) | `n2v_lambda=0.3, force_preprocess=True, walk_length=10, p=q=1.0, share_with_poi_id=False` |

## Sample multi-seed run invocation

```bash
for SEED in 42 7 1 100 0; do
  SEED=$SEED python scripts/canonical_improvement/regen_emb_t3.py \
      --state alabama \
      --epoch 500 \
      --use-node2vec-poi \
      --n2v-lambda 0.3
done
# Repeat for --state arizona; primary AL+AZ batch.
# λ-sweep ablation: re-run with --n2v-lambda 0.1 and --n2v-lambda 1.0.
```

## Open questions for executor

1. Confirm that downstream MTL training (`scripts/train.py`) sees no
   change — T5.2a affects only the SSL embedding generation step.
2. The skip-gram loss is computed full-batch over all cached walks per
   epoch. At Alabama scale (~80K POIs, walks_per_node=5, length=10 →
   400K walks × 4 context positions ≈ 1.6M pairs × 5 negs ≈ 8M scores)
   memory is ~512MB for D=64 — fine on A40. At Texas/Florida scale
   (~500K POIs) consider chunking the loss like Check2HGI_InfoNCE does
   (`chunk_size` kwarg pattern).
3. T5.1 hook contract: my code reads `model.poi_id_embedding` for the
   `--n2v-share-table-with-poi-id` path. Verify with T5.1 author that
   this attribute name matches their implementation; otherwise the
   share path warns and falls back to separate (safe default).
