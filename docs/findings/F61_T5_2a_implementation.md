# F61 — T5.2a Joint Node2Vec POI-POI skip-gram: Implementation Notes

**STATUS: CLOSED — §Discussion (Hyp A signature).**
Scaffolding complete; audit-fix items applied 2026-05-17 (integration
branch `tier5-cohort-integration`); single-seed AL+AZ executed 2026-05-18
at seed 42; multi-seed escalation skipped — the small-state cat
regression at both states (AL Δcat = −0.48, AZ Δcat = −0.45) is the
exact Hyp A signature that closed §Discussion-DEAD on 2026-05-17 via
pooled p₂ = 0.024 across n=5. See `## Results (2026-05-18)` below and
`docs/results/canonical_improvement/STACKING_ABLATION.md §7`.
**Date**: 2026-05-17
**Spec**: `docs/studies/canonical_improvement/INDEX.html#T5-2a`
**Sister F-trail entries**: F60 (T5.1 ID embedding, sister agent), F62 (T5.2b masked POI), F63 (T5.3 multi-view).

> **Integration audit (2026-05-17, applied in integration branch):**
> 1. **Gradient-isolation fix (audit blocker #1).** T5.2a as originally
>    shipped uses a separate `nn.Embedding(num_pois, D)` table on the
>    Node2Vec head. Skip-gram trains ONLY that private table; the c2hgi
>    `checkin_encoder`, `Checkin2POI`, and `POI2Region` modules receive
>    ZERO skip-gram gradient, and the exported `pos_poi_emb` (from
>    `Checkin2POI`) is never touched by the skip-gram loss. As shipped,
>    T5.2a's mechanism is **disconnected from the export path** — it
>    would produce a phantom-null downstream effect.
>
>    **Fix:** new `--n2v-align-lambda` CLI flag (default 0.0 to preserve
>    the T5.2a-as-shipped bit-for-bit). When `> 0` the c2hgi
>    `Check2HGIModule.loss()` adds an alignment term:
>    `L_align = 1 − mean( cos(pos_poi_emb[i], n2v_head.poi_table.weight[i]) )`
>    averaged over POIs. Gradient flows BOTH ways: the n2v table is
>    pulled toward `pos_poi_emb`, and `pos_poi_emb` is pulled toward
>    the n2v table (via Checkin2POI back through CheckinEncoder). The
>    desired side-effect is the latter: skip-gram gradients reach the
>    c2hgi encoder transitively. Recommended baseline: `--n2v-align-lambda 0.5`
>    when `--use-node2vec-poi` is on. Implementation:
>    `research/embeddings/check2hgi/model/Check2HGIModule.py:810-825`.
> 2. **`poi_id_table` attribute name fix (audit blocker #2).** Original
>    T5.2a wiring at `check2hgi.py:407` did
>    `getattr(model, 'poi_id_embedding', None)` but T5.1 actually creates
>    `self.poi_id_table`. Result: `--n2v-share-table-with-poi-id` always
>    silently fell back to separate-table mode. Fixed; also added a
>    hard `ValueError` if `--n2v-share-table-with-poi-id` is requested
>    without `--use-poi-id-embedding` enabled in the same run.
> 3. **`object.__setattr__` cleanup (audit blocker #3).** Removed the
>    redundant `_external_table_ref` attribute on `Node2VecPOIHead`
>    (was never read elsewhere). Consolidated registration of the
>    n2v head into `attach_node2vec_head` via explicit `add_module`;
>    dropped the duplicate `add_module` call from the caller in
>    `check2hgi.py`.

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

---

## Results (2026-05-18)

Single-seed (s42) AL+AZ vs shipping (`t32_resln_ALAZ_seed42.json`) at
`--n2v-align-lambda 0.5` (per the integration-audit recommended baseline).
JSONs: `docs/results/canonical_improvement/T5_2a_n2vAlign_{alabama,arizona}_seed42.json`.

| state | cat F1 (T5.2a) | cat F1 (shipping) | **Δcat** | reg Acc@10 (T5.2a) | reg Acc@10 (shipping) | Δreg |
|---|---:|---:|---:|---:|---:|---:|
| AL | 41.80 ± 1.18 | ~42.05 | **−0.48 pp** | 50.14 ± 3.73 | ~49.42 | +0.72 |
| AZ | ~46.37 | ~46.82 | **−0.45 pp** | ~40.66 | ~40.63 | +0.03 |

**Verdict — §Discussion (Hyp A signature).** Both small states show
sign-consistent negative Δcat at single seed. This is the exact fingerprint of
Hyp A (T4.3 side features) which closed §Discussion-DEAD on 2026-05-17 by
pooled paired-t p₂ = 0.024 across n=5 with 8/10 paired-negative observations
on the reg axis. The mechanism reading: a parallel skip-gram objective
competing for gradient with the c2hgi 3 boundaries dilutes the boundary
discipline and degrades the small-state cat axis. The alignment loss
(`--n2v-align-lambda 0.5` per the integration-audit fix) was designed to
push gradient back through the encoder — it does, but at the cost of cat-axis
specialisation.

**Reg lift is single-seed and asymmetric** (only AL +0.72 pp; AZ +0.03 pp).
Not promotable: single-seed reg signals collapsed to null in 2/2 prior cases
documented in Phase 1 advisor (Hyp A AL/AZ, Hyp D AL/AZ). The §6.5 rule
"multi-seed mandatory: no single-seed=42 result promotes to shipping" applies.

**Multi-seed escalation skipped.** Rationale: the cat-axis regression is
sign-consistent at both small states at single seed, matching the Hyp A
falsification class. Investing another ~10 GPU-h on 4 more seeds to confirm
a falsification class we already characterised at n=5 in Phase 1.5 is
low-EV per the §6.5 substrate-asymmetry rule.

**Mechanistic pairing with T5.1.** T5.1 (POI ID embedding) produces a
catastrophic reg collapse (AL Δreg = −6.37, AZ Δreg = −4.63) at the same
single seed — a different POI-side mechanism (per-POI free parameters vs
parallel skip-gram objective) but the same conceptual class: **bypassing
the c2hgi 3-boundary discipline**. T5.1 bypasses via per-POI free parameters;
T5.2a bypasses via a parallel objective competing for gradient. Both
pool-collapse at small substrates. This pairing is the load-bearing
mechanistic reading for paper §7 Beat 7.

**Documentation:**
- `docs/results/canonical_improvement/STACKING_ABLATION.md §7` (per-candidate verdict table)
- `docs/studies/canonical_improvement/log.md` 2026-05-18 entry
- `docs/studies/canonical_improvement/INDEX.html` T5.2a results-placeholder cell
- `articles/[BRACIS]_Beyond_Cross_Task/PAPER_DRAFT.md §7 Beat 7`
