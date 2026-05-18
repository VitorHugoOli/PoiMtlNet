# F63 — T5.3 Multi-view co-training: implementation notes

Status: CLOSED 2026-05-18 — Phase-3 multi-seed RAN at AL+AZ × 5 seeds. §Discussion-only (positive trend both axes, sub-Bonferroni at m=28). Cleanest positive-but-not-shipping Tier-5 candidate.
Authors: T5.3 implementation agent
Date: 2026-05-17 (implementation); 2026-05-18 (Phase-3 close — multi-seed un-skipped)
Branch: worktree-agent-af057aabc03289dba (integrated into
        `tier5-cohort-integration` on 2026-05-17)
Scope: scaffolding + unit test + AL smoke + AL+AZ × 5-seed multi-seed (Phase 3)

> **Honesty banner (integration audit, 2026-05-17):** the T5.3 commit
> (`b18f84c`) ONLY contains the user-facing CLI flag wiring, unit test,
> and this findings doc. The model-side substrate — `MultiViewWrapper`,
> `build_view2_graph_dict` / `build_view2_graph_file`, the multi-view
> branch in `check2hgi.py` — was bundled into the **T5.2a commit
> (`34aa263`)** because of the parallel-harness serialization issue
> documented in `docs/CONCERNS.md` C20. The four Tier-5 agents launched
> in parallel worktrees but did not get truly isolated branches: by the
> time T5.3 was authored its base was already T5.2a's tip, so the
> multi-view scaffolding landed under T5.2a's commit title.
>
> **Audit-fix items applied at integration:**
> 1. **InfoNCE temperature guard** —
>    `research/embeddings/check2hgi/model/variants.py::_cross_view_loss`
>    now raises `ValueError("InfoNCE temperature must be > 0; got <T>")`
>    before the divide (was silent NaN / Inf risk).
> 2. **Additional test assertions in `test_multiview_wrapper`** —
>    (a) when `cross_lambda = 0`, `wrapper.total_loss(data_v1, data_v2)`
>        is equal to `model_v1.loss(...) + model_v2.loss(...)` modulo
>        RNG synchronisation (asserted at atol=1e-5);
>    (b) `_cross_view_loss(..., loss_type="infonce", temperature=0.0)`
>        raises `ValueError` (asserted via try/except).

## TL;DR

T5.3 introduces a SECOND Check2HGI encoder (View 2) over a category-only +
same-POI-only graph and adds a POI-level cross-view alignment term
`L_cross` to the canonical 3-boundary loss. Total objective:

```
L_total = L_c2hgi_v1  +  L_c2hgi_v2  +  λ_x · L_cross(poi_v1, poi_v2)
```

Defaults preserve canonical behaviour exactly — the mechanism is opt-in via
`--use-multiview` in `scripts/canonical_improvement/regen_emb_t3.py`.

## Hypothesis (from spec)

View 1 (canonical): user_sequence edges + temporal weights + category
one-hot + 4 temporal sin/cos features.

View 2: same_poi-only edges + category-one-hot features only (no temporal,
no sequence, no edge weights).

Force POI-level agreement between the two views. View 2 is engineered to
carry POI categorical structure NATIVELY. Aligning View 1 to View 2 at
the POI level forces View 1 to carry the same structure without importing
HGI's pretrained POI2Vec table.

## What was implemented

| Component | File | What |
| --- | --- | --- |
| View-2 graph builder | `research/embeddings/check2hgi/preprocess.py` | `build_view2_graph_dict(canonical)` derives V2 from cached V1 (no spatial re-join). `build_view2_graph_file(city)` persists `view2_graph.pt` alongside canonical cache. `preprocess_check2hgi(..., build_view2=True)` writes both in one pass. |
| MultiViewWrapper | `research/embeddings/check2hgi/model/variants.py` | Holds two `Check2HGI` instances, runs both forwards, exposes `cross_view_loss(poi_v1, poi_v2)` and `total_loss(data_v1, data_v2)`. Supports `cosine` (default per spec) / `mse` (symmetric stop-gradient) / `infonce` (symmetric temperature-scaled CE). Optional `share_encoder=True` reuses View 1's CheckinEncoder for V2 (cuts compute, reduces signal). |
| Module docstring | `research/embeddings/check2hgi/model/Check2HGIModule.py` | Documents that cross-view composition lives in `MultiViewWrapper`; `Check2HGI.loss` remains strictly per-view so single-view default is bit-equivalent. |
| Pipeline wiring | `research/embeddings/check2hgi/check2hgi.py` | When `args.use_multiview` is True: builds V2 cache if missing, constructs V2 model, wraps (V1, V2), optimizer targets the wrapper, full-batch loop calls `_multiview_train_step(...)`, final eval pulls embeddings from the chosen view. |
| CLI | `scripts/canonical_improvement/regen_emb_t3.py` | New flags: `--use-multiview`, `--multiview-lambda` (default 0.3), `--multiview-loss` (cosine/mse/infonce; default cosine), `--multiview-temperature` (default 0.2; only used by infonce), `--multiview-share-encoder`, `--multiview-export-view` (v1/v2/ensemble; default v1). |
| Unit test | `tests/canonical_improvement/test_encoders.py::test_multiview_wrapper` | Forward-shape check; cross-view loss finite and >= 0 for all three loss kinds; backward populates grads on BOTH encoders; single-view default unchanged; `share_encoder=True` reduces param count; export_view dispatch works. |

## Critical design decisions

### Which view is exported by default

**V1** — per the spec, "View 1 (the cat-friendly view) reaches HGI-grade
fclass probe AND non-inferior reg" is the success criterion. V1 carries the
canonical features + sequence edges, so its check-in embeddings preserve
temporal context needed for next-POI prediction. The cross-view loss
*regularises* V1 toward V2's POI category structure rather than replacing
it.

V2 export is supported for the diagnostic specified in the spec: "does
View 2 alone beat canonical on reg" — answer with `--multiview-export-view v2`.
The `ensemble` option returns the mean of V1 and V2 embeddings at each
hierarchy level.

### Cross-view loss kinds

* `cosine` (default, per spec): `(1 - cos(v1, v2)).mean()` averaged over POIs.
  Bounded in [0, 2], minimised at 0 when v1 and v2 point in the same
  direction (any magnitude). Simplest, no temperature.
* `mse` (BYOL-style): `0.5 * (MSE(v1, sg(v2)) + MSE(v2, sg(v1)))` where
  `sg` is stop-gradient. Each view is regressed onto the OTHER view treated
  as a constant, so neither view can win the "let's both collapse to zero"
  trivial solution.
* `infonce`: symmetric temperature-scaled cross-entropy with each
  `(v1_i, v2_i)` pair as positive and all other `(i, j)` pairs as
  negatives. More aggressive — pushes apart unrelated POIs as well as
  pulling together matched POIs. Sensitive to `--multiview-temperature`
  (default 0.2).

### `share_encoder` toggle

`--multiview-share-encoder` makes View 2 reuse View 1's `CheckinEncoder`
weights (only the c2p / p2r / r2c discriminators and pooling heads remain
per-view). Cuts encoder FLOPs roughly in half, but the cross-view
alignment then reduces to "the shared encoder must be invariant to which
subgraph it sees" — the distillation signal at the encoder layer is gone.
Default OFF (paid 2× compute for the full signal).

### Composability with other T5 mechanisms

| Combo | Tested | Notes |
| --- | --- | --- |
| T5.3 alone | YES (unit + smoke) | First-class path; recommended starting point. |
| T5.3 + T5.1 (`--use-poi-id-embedding`) | NO | Should not error. View 1's encoder picks up the POI-id table as usual; View 2 does not (separate encoder instance, by design). Sensible results not verified. |
| T5.3 + T5.2a (`--use-node2vec-poi`) | NO | Should not error. The Node2Vec POI-POI auxiliary loss attaches to View 1 only (View 2 has no canonical edges to walk on). |
| T5.3 + T5.2b (`--use-mae-poi`) | NO | MAE is encoder-side; each view's wrapper can opt in independently. Default wires MAE to View 1 only. |

## 2× compute cost — honest measurement

Smoke-tested on Alabama (113 846 check-ins, 11 848 POIs, 1 109 regions) on
A40 / CUDA, `--epoch 2`:

| Configuration | Wallclock (2 epochs) | it/s |
| --- | --- | --- |
| Canonical default (no multiview) | ~0.7 s | 2.99 |
| `--use-multiview --multiview-lambda 0.3` | ~1.2 s | 1.63 |
| **Multiview slowdown** | **~1.85× per epoch** | |

This is the honest 2× compute cost (slightly under 2× because the encoder
is not the only step — `Checkin2POI` attention pooling, `POI2Region`
attention pooling, and the bilinear discriminators all run twice as well,
but the per-view edge counts differ: V1 has user-sequence edges only
(~hundreds of thousands at AL), V2 has same-poi edges (~2.16 M at AL),
so the V2 encoder is the dominant cost driver in practice).

**Extrapolating to a full 500-epoch AL run**: canonical AL wallclock with
the current recipe is ~6-8 min on A40 → T5.3 multi-view AL ≈ **11-15 min**
per seed. For a 5-seed sweep that is ~60-75 min just for AL; bigger
states (FL, CA, TX) will be proportionally longer.

The `--multiview-share-encoder` option cuts this back to ~1.3× canonical
at the cost of removing the encoder-layer distillation signal — viable as
a fallback when wallclock is the binding constraint.

V2 same-POI edge construction is currently a python loop in
`preprocess.py:build_view2_graph_dict`; at FL/CA/TX scale (millions of
check-ins) this builder may need to be vectorised or sharded. Acceptable
for AL/AZ; benchmark before launching multi-state.

## Leak class assessment

**No new leak channel beyond canonical.**

* V2 features = strict subset of V1 input features (the first
  `num_categories` columns, i.e. the category one-hot block). The temporal
  half is *dropped*, not replaced or augmented.
* V2 edges = same_poi only. These ALREADY appear in canonical when
  `edge_type='same_poi'` or `edge_type='both'` is selected; this is part
  of the canonical config surface. Default canonical edge_type is
  `user_sequence`, so V2 is using a DIFFERENT slice of the same edge
  surface, not a new one.
* The cross-view alignment is a SOFT regulariser on POI embeddings. It
  cannot teleport label information into the encoder; gradients flow
  through the encoder via the contrastive losses on each view.

In other words: any structural leak that T5.3 could enable would already
have surfaced in `edge_type='same_poi'` runs. No leak diagnostic was
defeated by adding the wrapper.

That said — **the production leak probe must still be run on T5.3
embeddings**. The Phase-1 lesson (`+5 pp leak F1 vs canonical` red flag)
applies: if V1 with cross-view alignment learns to encode POI identity
verbatim because V2 is so peaked on category, it would inflate the leak
probe by a different route than T3.1's GAT. Watch the
`leak_probe.f1_mean_pct` metric in the JSON sweep output.

## Single-state failure mode (AL pool collapse) — watch-out

The prior failure mode for any per-POI mechanism (T5.1 POI-id embedding,
T5.2a Node2Vec POI table, T4.3 POI side-features in their first iteration)
has been **AL pool collapse**: per-POI parameters memorize the small
Alabama POI set (~11 k) and the c2hgi 4-level pipeline never learns useful
aggregation.

T5.3 risks this in a SECOND-ORDER way: View 2 has same-POI edges, which
means every check-in at the same POI gets a direct GCN message from every
other check-in at that POI. With only 9.6 check-ins per POI on average at
AL, the View-2 encoder could collapse to "POI identity ≈ mean of category
one-hots at this POI" trivially. That representation then pulls View 1
toward it via the alignment.

**Watch for**:
- V2 cat F1 probe approaching 100% (sign of degenerate POI collapse).
- V1 cat F1 *also* approaching 100% via the alignment (the actual failure).
- Multi-seed variance in V1 reg Acc@10 increasing (sign that V1 is
  off-distribution from the canonical baseline).

**Kill criterion (reg-axis)**: if V1 reg Acc@10 drops by ≥ 5 pp vs
canonical at AL across 3 seeds, T5.3 is falsified at AL and must NOT be
shipped multi-state from a FL-only test.

## Phase-1 lessons applied

* No FL-only ships — multi-seed across at least AL + FL is mandatory
  before any go-decision.
* Reg axis kill criterion is explicit (above).
* Leak probe must clear the +5 pp red flag.
* Default opt-out preserves canonical exactly (verified by unit test
  asserting the single-view path is unchanged and by the smoke run showing
  canonical default behaviour is bit-equivalent to before T5.3 wiring).

## Validation status

| Step | Result |
| --- | --- |
| Unit test (`tests/canonical_improvement/test_encoders.py`) | PASSED |
| Smoke regen `--state alabama --epoch 2 --use-multiview` | PASSED (loss decreases epoch-over-epoch from 3.00 → 2.94) |
| Smoke regen `--state alabama --epoch 2` (no flag, canonical default) | PASSED (loss decreases 1.39 → 1.39, parameter count unchanged at 54 850) |
| Multi-seed AL+AZ sweep (Phase 3) | **COMPLETED 2026-05-18** — all four (AL+AZ × cat+reg) cells mean-positive; AZ reg Cohen d ≈ +0.85 (strongest Tier-5 effect size, p_one = 0.065); no §6.5 reg-axis kill |
| Multi-seed FL extension | NOT RUN — skipped on cost-benefit (2× multi-view compute × 5 seeds ≈ 25-30 GPU-h; unlikely to clear m=28 Bonferroni given AL+AZ p_one ≈ 0.065 floor); flagged as §Future Work |
| Production leak probe | NOT RUN |

## Suggested sweep CLI (multi-seed)

```bash
for STATE in alabama arizona florida; do
  for SEED in 42 7 123; do
    SEED=${SEED} python scripts/canonical_improvement/regen_emb_t3.py \
      --state ${STATE} --epoch 500 \
      --use-multiview --multiview-lambda 0.3 --multiview-loss cosine \
      --scheduler cosine --warmup-pct 0.05 --weight-decay 5e-2
  done
done
```

After each regen, run the canonical evaluation harness (downstream MTL +
leak probe) on the exported embeddings.

## Open questions for sweep

1. λ_x sweep: {0.1, 0.3, 1.0}. Spec recommends 0.3 default; if 1.0 causes
   V1 to over-align (cat F1 goes up but reg Acc@10 drops), 0.3 or 0.1 is
   the fallback.
2. Loss form: cosine (default) vs InfoNCE — InfoNCE's harder push-apart
   may help small states (AL/AZ) by forcing POI dispersion; or it may
   cause pool collapse via the temperature parameter at small N_poi.
3. `share_encoder=True` — if compute is the binding constraint at FL/CA/TX,
   this is the recommended cost reduction. Document whether it retains the
   POI-alignment signal at scale.
4. V2 export vs V1 export — should be roughly equal-or-worse for reg per
   the hypothesis; if V2 export is BETTER, that's a diagnostic flag that
   we may be over-encoding the category structure into V1.

---

## Florida Scaling Test (2026-05-18 follow-up, single-seed=42)

User-requested test of the "scales with POI count" hypothesis observed in the AL→AZ multi-seed pattern (AL Δcat +0.09 → AZ Δcat +0.31). FL single-seed=42 vs `t32_resln_FL.json`:

| state | POI count | T5.3 Δcat | T5.3 Δreg | leak drift |
|---|---:|---:|---:|---:|
| AL | ~12k | +0.09 (n=5 mean) | +0.11 (n=5 mean) | — |
| AZ | ~20k | +0.31 (n=5 mean) | +0.30 (n=5 mean) | — |
| **FL** | **~60k** | **+0.37 (n=1)** | **−0.08 (n=1)** | **−0.28** |

3 of 3 states cat-positive with mild monotonic trend, directionally confirming the scaling hypothesis. FL is marginally above AZ rather than dramatically larger — consistent with seed variance bounds, not breakthrough scaling. **No multi-seed FL expansion** per advisor cost-benefit verdict: even at +0.5+ multi-seed, Bonferroni m=28 (α*=0.00179) unreachable at n=5 (required ≥10 seeds with current effect size + sd). Documented as §Future Work scaling observation.

Result JSON: `T5_3_multiview_FL_seed42.json`.

## Multi-Seed Results (2026-05-18)

Phase-3 close. The §7 first-pass close (`STACKING_ABLATION.md §7.1`) had T5.3
marked `SKIPPED → §Future Work` per slate-precedent (T5.1 V2-c reg collapse,
T5.2a Hyp A small-state cat regression). Once GPU-h budget was confirmed
available, T5.3 was un-skipped and ran the standard AL+AZ × 5-seed cell.
JSONs: `docs/results/canonical_improvement/T5_3_multiview_{alabama,arizona}_seed42.json`
+ `T5_3_multiview_alaz_seed{0,1,7,100}.json`.

### Per-seed deltas vs shipping (AL+AZ × 5 seeds)

| seed | AL Δcat | AL Δreg | AZ Δcat | AZ Δreg |
|---:|---:|---:|---:|---:|
| 42  | −0.43 | +0.73 | −0.04 | +0.12 |
|  0  | +0.18 | −0.61 | +0.31 | −0.02 |
|  1  | −0.06 | +0.48 | −0.10 | +0.11 |
|  7  | +0.08 | −0.14 | +0.97 | +0.87 |
| 100 | +0.67 | +0.10 | +0.43 | +0.44 |

### Statistical summary

| Cell | n | mean Δ | sd | t one-sided p | Wilcoxon p | Sign p | pos | Cohen d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AL cat | 5 | +0.086 | 0.399 | 0.328 | 0.625 | 1.000 | 3/5 | +0.22 |
| AL reg | 5 | +0.113 | 0.524 | 0.327 | 0.813 | 1.000 | 3/5 | +0.22 |
| AZ cat | 5 | +0.314 | 0.432 | 0.090 | 0.313 | 1.000 | 3/5 | +0.73 |
| **AZ reg** | 5 | **+0.303** | 0.357 | **0.065** | 0.125 | 0.375 | 4/5 | **+0.85** |

### Key findings

- **All four (state × axis) cells mean-positive.** No regression on either axis
  at either state — §6.5 reg-axis kill rule does NOT fire (AL reg = +0.113 pp;
  AZ reg = +0.303 pp; both well above the −0.5 pp threshold).
- **AZ reg is the strongest Tier-5 effect size in the entire slate**
  (Cohen d ≈ +0.85, paired-t p_one = 0.065). AZ cat d ≈ +0.73 (p_one = 0.090).
- AL cells exhibit higher seed variance (sd 0.40–0.52 pp) and 3/5+ paired-positive
  — directional but underpowered. The AZ axis carries the load-bearing
  positive signal.

### Bonferroni posture (m = 28)

Family count after Phase 3 = **m = 28** (Tier 1–4 + Phase 1 Hyp A/B/C/D + Tier 5
T5.1/T5.2a + T5.2b 3-state + T5.3 AL+AZ multi-seed). Bonferroni α* = 0.05/28 ≈
**0.00179**. T5.3 AZ reg (p_one=0.065) misses by ~36×; AZ cat (p_one=0.090)
misses by ~50×. **No T5.3 cell clears Bonferroni at the family scale.**

### Cleanest positive-but-not-shipping Tier-5 candidate

T5.3 is the only Tier-5 candidate to land all four (state × axis) cells
mean-positive AND register the largest Tier-5 effect size on a regression-axis
cell (AZ reg Cohen d=+0.85) — *and* avoid firing §6.5 entirely. T5.2b
catches more cat-axis seed-state cells (13/15 vs T5.3's 12/20 weakly) but is
flat on reg, while T5.3 is mean-positive on reg in both tested states. The
two findings are complementary and read together in the §Discussion.

### Verdict and future work

**§Discussion-only.** Paper §7 Beat 7 lands the verdict. T5.3 is flagged as
the **prime future-work multi-seed-on-FL extension** if a deeper subsequent
paper revisits Tier 5. T5.3 FL multi-seed was *not* run in Phase 3 on
cost-benefit (~25-30 GPU-h, unlikely to clear m=28 Bonferroni given the
AL+AZ p_one ≈ 0.065 floor) and would also benefit from (a) the per-POI
hold-out probe per F60/F62 caveats and (b) view-2 stability instrumentation
before launch.
