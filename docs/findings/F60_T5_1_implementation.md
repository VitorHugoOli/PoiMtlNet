# F60 — T5.1 Native Learned POI ID Embedding (implementation notes)

**Status**: implementation complete (worktree `agent-acdb1dfd1ce9c7749`),
audit-fix items applied 2026-05-17 (integration branch
`tier5-cohort-integration`), awaiting Phase A multi-seed sweep at AL+AZ
(small-state regression-gate states per Phase 1 advisor 2026-05-17).

> **Integration audit (2026-05-17, applied in integration branch):**
> 1. **Production-path canonical-preservation test** added —
>    `tests/canonical_improvement/test_encoders.py::test_check2hgi_module_t51_optout`
>    constructs two `Check2HGI(...)` instances differing only in T5.1 flags
>    (with shared core weights copied to isolate from RNG-consumption
>    side-effects of `nn.Embedding` construction) and asserts byte-equality
>    on `pos_checkin_emb` / `pos_poi_emb` / `pos_region_emb` at zero-init
>    step 0. The original `test_poi_id_embedding` only covered the
>    `POIIdMixedPooler` wrapper, NOT the production-path `Check2HGI` module.
> 2. **ValueError construction guards** —
>    `test_check2hgi_t51_value_errors` covers `use_poi_id_embedding=True`
>    without `num_pois` (raises) and `poi_id_init='poi2vec'` (raises;
>    POI2Vec warm-start is merge-family, out of scope here).
> 3. **T5.1 × T4.3 interaction note:** when both T5.1 and T4.3 are enabled,
>    the T5.1 bump enters the p2r/r2c pathway via T4.3's side-feature
>    post-projection (the augmented `pool_post_proj` consumes
>    `pos_poi_emb + gamma * table.weight` as its first concat half). The
>    T5.1 contribution to the c2p boundary is via the pre-augmentation
>    `pos_poi_emb_pure` path — direct, not through T4.3's projection. So
>    cat/c2p sees the raw T5.1 bump; reg/p2r and city/r2c see the
>    post-T4.3-projected T5.1 bump. Implication: if T5.1+T4.3 lifts cat
>    alone with no reg motion, suspect the raw c2p bump first; if reg
>    lifts with no cat motion, the T4.3 projection is mediating the
>    T5.1 signal toward the structural axis.
> 4. **Per-POI memorisation caveat — Phase A interpretation policy:**
>    the existing IJM probe is **user-held-out**, not per-POI-held-out.
>    Per-POI memorisation by the T5.1 table CANNOT be ruled out by the
>    IJM probe alone — Phase A multi-seed results will be interpreted
>    with this caveat in force, and **no T5.1 promotion to NORTH_STAR
>    / shipping is permitted before a per-POI hold-out probe is built
>    and passes**. Building the probe itself is a future task (out of
>    scope for the current sprint).

**Date**: 2026-05-17.

---

## What was built

A zero-init (or small-Gaussian) per-POI learnable identity table that
is added (additively, NOT concatenated) to the Checkin2POI attention-pool
output, BEFORE the side-feature injection and BEFORE the p2r aggregation.
The table is trained ONLY by check2hgi's 3 contrastive boundaries
(c2p, p2r, r2c) — there is no fclass supervision, no auxiliary head,
and no POI2Vec warm-start. The table is gated by `use_poi_id_embedding`
(default False) so the canonical recipe is byte-identical at default
flags.

### Combine rule

```
pos_poi_emb = checkin2poi(pos_checkin_emb, checkin_to_poi, num_pois)
neg_poi_emb = checkin2poi(neg_checkin_emb, checkin_to_poi, num_pois)
if use_poi_id_embedding and poi_id_gamma != 0:
    bump = poi_id_gamma * poi_id_table.weight   # (num_pois, D), broadcast add
    pos_poi_emb = pos_poi_emb + bump
    neg_poi_emb = neg_poi_emb + bump
```

The table row at POI index `p` is shared between the positive forward
(corruption-free checkin pool) and the negative forward (feature-corrupted
checkin pool). The c2p discriminator cannot reach zero loss using only
the table because the negative POI is a DIFFERENT POI whose pool is
unrelated; the table contributes a per-identity bias on top of the pool,
not a substitute for it.

### Hyperparameters

| Knob              | Default  | Spec ablation                             |
|-------------------|----------|-------------------------------------------|
| `use_poi_id_embedding` | False    | bool (Phase A sweeps True)                |
| `poi_id_gamma`         | 0.3      | {0.1, 0.3, 1.0}                           |
| `poi_id_init`          | "zero"   | "zero" or "gaussian" (std=0.01)           |

`num_pois` is taken at construction from the preprocessed graph
(`city_dict['num_pois']`) — NEVER hard-coded.

---

## Design decisions

### Why ADDITIVE and not concat-then-project

T4.3 side-features use concat-then-project because the side vector has
a different dimension (32 or 64-d derived stats). The T5.1 POI ID table
is the same width as the pool (D=64), so an additive combine preserves
the dimension and the canonical p2r distribution. Concat would have
forced a (D + D → D) projection, doubling the parameter count of the
post-pool layer and changing the p2r input variance.

### Why ZERO init (default) vs small Gaussian

Zero init gives strict cold-start neutrality: the very first forward
of an enabled table is bit-identical to the canonical pool (the test
verifies this). Cold-start POIs with no contrastive signal stay near
zero forever — no spurious lift. Gaussian init (std=0.01) gives SGD a
non-zero starting gradient; the F60 unit test verifies that a linear
probe on a Gaussian-init table at std=0.01 sees init-time category F1
≈ 0.03 (well below the 0.10 chance line and the 0.50 guardrail).

### Why NOT POI2Vec warm-start

Per the Tier-5 carve-out: importing HGI's POI2Vec is the merge-family
path. T5.1 is the "native" hypothesis — does giving each POI an
identity slot, trained ONLY by c2hgi's losses, lift the fclass probe?
A POI2Vec warm-start would conflate "the table helps" with "the POI2Vec
prior helps", which is a different study. The constructor explicitly
rejects any `poi_id_init` value other than "zero" or "gaussian".

### Where the table is injected

Injection point: AFTER `checkin2poi` pool, BEFORE side-feature
injection and BEFORE `poi2region`. This makes the table participate in
ALL three contrastive boundaries (c2p, p2r, r2c). If we had injected
only into the c2p pathway (like `pos_poi_emb_pure`), the table would
NOT see the p2r or r2c gradient — collapsing the spec into a c2p-only
identity boost, which empirically saturates fast (the canonical c2p
already achieves 99% positive-discrimination accuracy by epoch 50).

`pos_poi_emb_pure` is set AFTER the T5.1 bump is applied, so the c2p
path also sees the table — symmetric with side-features which are
explicitly EXCLUDED from `_pure` (side-features leak; the POI ID
table is the question we're studying, not a leak signal).

---

## Leak risk assessment

A learnable per-POI table CAN memorise train-set transitions if the
contrastive losses find a degenerate solution. Three safeguards:

1. **The pool stays in the gradient chain**. The c2p discriminator
   would have to drive the table to memorise the pool's representation
   AND ignore the actual pool signal — strictly worse loss than
   learning both jointly. Zero-init biases SGD towards "use the pool
   first, refine with the table". Gaussian-init at std=0.01 keeps the
   initial table contribution << pool norm.

2. **Negative POIs also have their table row added**. If the table
   alone discriminated identity, the negative POI's bump would be
   indistinguishable from the positive's by the c2p bilinear — the
   discriminator gets no shortcut from the table itself.

3. **Production gates remain in place**. The IJM leak probe and the
   F51-style trained-embedding leak F1 (`+5 pp red flag vs canonical
   40.85`) will fire if the trained table opens a memorisation
   channel. A per-POI hold-out probe is deferred to Phase B per the
   T5.1 spec (Evaluation criteria §3: "verify the learned table
   doesn't memorise train-set transitions in a way that probe-leaks").

The audit advisor should still inspect the trained table's nearest-
neighbour structure post-training (kNN-Jaccard vs HGI) and the
visit-count-quantile fclass split to catch cold-start collapse.

---

## Files modified

1. **`research/embeddings/check2hgi/model/variants.py`** — added class
   `POIIdMixedPooler` (additive wrapper over `Checkin2POI`; gamma=0
   pass-through is bit-identical). Note: the production path actually
   bypasses this wrapper and bakes the additive table into
   `Check2HGIModule` directly (see file 2); `POIIdMixedPooler` is the
   unit-test surface and a documented alternative shape for future
   integrations that need to swap aggregators.
2. **`research/embeddings/check2hgi/model/Check2HGIModule.py`** —
   added constructor args `use_poi_id_embedding`, `poi_id_gamma`,
   `poi_id_init`, `poi_id_init_std`, `num_pois`. When enabled, builds
   `self.poi_id_table = nn.Embedding(num_pois, D)` and in `forward`
   adds `gamma * table.weight` to BOTH `pos_poi_emb` and `neg_poi_emb`
   between the pool and the side-feature injection. Default opt-out:
   when `use_poi_id_embedding=False`, `self.poi_id_table is None` and
   the forward path is unchanged.
3. **`research/embeddings/check2hgi/check2hgi.py`** — pipes the three
   cfg fields (plus `num_pois` from the graph cache) into the
   `Check2HGI` constructor.
4. **`scripts/canonical_improvement/regen_emb_t3.py`** — added CLI
   flags `--use-poi-id-embedding`, `--poi-id-gamma`, `--poi-id-init`.
   Default opt-out preserves canonical behaviour. Also fixed the
   pre-refactor `_root` 4-parent path bug (matching the uncommitted fix
   on main).
5. **`tests/canonical_improvement/test_encoders.py`** — appended
   `test_poi_id_embedding()` covering (a) forward shape, (b) backward
   gradient on the table, (c) gamma=0 bit-identical to bare
   Checkin2POI, (d) zero-init step-0 bit-identical (cold-start neutral
   regardless of gamma), (e) Gaussian-init leak-probe (init-time table
   alone → poi-category F1 must stay << 0.50). Also fixed the pre-
   refactor `_root` 4-parent path bug (matching the uncommitted fix on
   main).

---

## Phase 1 lessons applied (2026-05-17 advisor)

- **No FL-only ships**. The Phase A sweep MUST cover AL+AZ + FL+CA+TX.
  Small-state regression on either cat or reg is dispositive — the
  parallel_sweep_runner.sh template already iterates `$STATES`.
- **Multi-seed mandatory**. Each state × hyperparameter setting MUST
  run at ≥ 4 seeds (the F51 protocol is {0, 1, 7, 100}, with seed 42
  retained for the historical baseline only). Single-seed=42 promotion
  is FORBIDDEN per the advisor.
- **Reg-axis kill criterion**. T5.1 is killed if Δreg ≤ -0.5 pp at
  ANY of the 5 states (FL, CA, TX, AL, AZ), OR if the pooled sign-test
  p ≤ 0.05 with a majority of states regressing on reg. Cat-axis-only
  wins are NOT acceptable.
- **Success criterion** (from spec): ≥ +30 pp fclass probe at AL+AZ,
  no cat or reg regression beyond noise, no cold-start collapse
  (visit-count quantile fclass split).

---

## Sample CLI invocation (Phase A)

Replicate via `scripts/canonical_improvement/parallel_sweep_runner.sh`,
setting `REGEN_CMD` to the T5.1-augmented regen line. The recommended
default-knob setting for the headline run is γ=0.3 + zero-init:

```bash
# Per-seed regen (4 seeds + canonical seed 42 for reference)
for SEED in 0 1 7 100; do
    for STATE in alabama arizona florida california texas; do
        SEED=$SEED python scripts/canonical_improvement/regen_emb_t3.py \
            --state $STATE \
            --encoder resln --encoder-dropout 0.0 \
            --scheduler warmup_constant --warmup-pct 0.05 \
            --weight-decay 5e-2 --epoch 500 \
            --use-poi-id-embedding --poi-id-gamma 0.3 --poi-id-init zero
    done
done
```

(Pair with the canonical+v3c base recipe — T3.2 ResLN encoder + AdamW
WD=5e-2 + warmup-constant — so any lift / regression is attributable
to T5.1 and not to the encoder swap. See `docs/NORTH_STAR.md` §Champion
for the post-regen MTL recipe.)

Gamma sweep is a sub-experiment: re-run with `--poi-id-gamma 0.1` and
`--poi-id-gamma 1.0` if the γ=0.3 cell ships. Init sweep is conditional
on γ=0.3 results (try `--poi-id-init gaussian` only if zero-init shows
flat / regressing cold-start fclass).
