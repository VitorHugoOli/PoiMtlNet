# Phase 11 — substrate audit: experiment plan

Opened 2026-05-07 after the user reopened the study at the substrate
level (see `HISTORY.md` Phase 11). Big-picture rationale and untested
hypotheses live in `HISTORY.md`; this document is the **execution
plan** — what to run, in what order, decision rules, files to update.

For per-design / per-merge results see `DESIGN_*.md`. For Phase 1-10
narrative see `HISTORY.md`. For the closed-study verdict on the
merge-design *surface* see `STATE.md`.

## Context (one paragraph)

Phase 10 declared the merge-design architectural surface saturated
(six convergent variants ±0.1 pp at FL reg). That verdict holds for
the *merge mechanism*. Phase 11 audits the **c2hgi substrate** that
every merge inherits — three pieces never re-examined since the merge
family was launched: (a) negative-sampling strategy at the new c2p
boundary, (b) the consequence of computing a corrupted forward pass
whose outputs only reach r2c, (c) the POI2Region methodology
(designed for POI2Vec-stable inputs, not for c2hgi's pooled-check-in
mixture).

## Important code-reading correction

A first-pass critique claimed "c2hgi switched to embedding-level
corruption for a 2× speedup, weakening the negatives." Re-reading
`Check2HGIModule.py:128-140` shows this is **wrong**:

- The CLAUDE.md docs in `research/embeddings/check2hgi/` describe
  embedding-level corruption — but the actual code does
  feature-level corruption (`cor_x = corruption(data.x)` followed by
  a full encoder pass).
- What is actually weak is *downstream*: `neg_checkin_emb` is never
  used in `loss()`; `neg_poi_emb` is computed and only feeds
  `neg_region_emb`; only the r2c boundary actually consumes the
  corrupted forward pass.
- HGI behaves the same way at p2r (uses positive-pool POIs paired
  with mismatched regions, not corrupted-feature POIs), so the
  *pattern* is inherited, not buggy. But c2hgi's c2p — its novel
  boundary — inherits the same pattern with **no compensating hard-
  negative mining**.

The S1 experiment below targets the **real** weakness: weak negatives
at c2p, not the (already-correct) feature-level corruption.

The original "S4 — DGI-style same-identity neg at c2p" still has merit
as an additional substrate test because it would consume the otherwise-
wasted `neg_poi_emb`. Kept in the ladder.

## Experiment ladder (cost-ordered)

| ID | Test | Target | Cost | Status |
|---|---|---|---:|---|
| S1 | Hard-neg c2p (same-region different-POI) on canonical c2hgi at AL+AZ | substrate | 2-3 h | ✗ FALSIFIED 2026-05-07 (AL Δ=−1.21 pp, AZ Δ=−0.81 pp) |
| S2 | S1 mechanism stacked on top of J at AL+AZ | substrate × merge | 2-3 h | SKIPPED (S1 null) |
| S4 | Use `neg_poi_emb` at c2p (DGI-style same-identity corrupted-feature negative) at AL+AZ canonical | substrate | 1-2 h | ✗ FALSIFIED 2026-05-07 (AL Δ=−0.07 pp, AZ Δ=−1.16 pp; same fold-failure pattern as S1) |
| S3-a | Direct `Checkin2Region` pooler as additive regulariser on J; foreign-region negs after v1 same-identity-neg kill | methodology | 1-2 d | ✗ FALSIFIED 2026-05-07 (v1 + v2 both killed at 50-ep PMA-entropy probe; uniform attention under both negative formulations; c2r supervision redundant with 3-boundary path) |
| S3-b V1 | Checkin2Region replacement, foreign-region neg, no POI2Vec residual | methodology | 1-2 d | ✗ FALSIFIED 2026-05-07 (AL Δ vs J = −2.91 pp; AZ Δ vs J = −2.54 pp; S3-b V1 ≈ canonical c2hgi, not J — pool-source change buys ~0.1 pp; the J→canonical gap is the POI2Vec prior, not the pooler) |
| S3-b V2-a | Checkin2Region replacement + POI2Vec residual on check-ins | methodology | 30 min | **NOT RUN** per advisor — expected outcome ≈J based on Phase 8 convergence-saturation; would not extend contribution |
| S3-b V2-c | Checkin2Region replacement + per-check-in POI2Vec anchor (encoder-level fclass pull) | methodology | ~30 min | ✗ FALSIFIED 2026-05-08 — AL Δ vs canonical = −9.95 pp (catastrophic); AZ Δ vs canonical = −0.28 pp (mild). Anchor "fights c2hgi context → worse than J" predicted by advisor. New finding: state-asymmetry of anchor effect — small per-region pools (AL ≈ 10.7 POIs/region) collapse to mean(POI2Vec); larger pools (AZ) absorb the anchor. |
| S3-b V2-d | c2r hard-neg sampling (similarity-mined foreign regions) | methodology | ~1 day | OUT OF SCOPE per one-redo rule (V2-c was THE redo). |
| S3-b V2-e | Alternative pooler architecture (Set Transformer / GATv2 / time-aware) | methodology | ~1 day each | OUT OF SCOPE per one-redo rule. |

**Run order rationale:**

S1 first — the cheapest principled substrate test, mirroring HGI's
p2r 25 % hard-neg strategy at the c2p boundary that c2hgi added but
never gave hard negatives to.

S2 only matters if S1 shows lift — confirms whether the substrate fix
compounds with the pool-boundary merge mechanism (J).

S4 is the second cheapest substrate fix — feeds the otherwise-wasted
`neg_poi_emb` into the c2p loss as a same-identity corrupted-feature
negative (DGI form). Cheaper than S3 because it's a one-line wiring
change; runs before S3 to keep priors-low fixes before methodology
changes.

S3 is the user's "Check-in↔Region methodology" reframe in its
strongest form — drops the POI bottleneck on the region pathway
entirely. Higher cost (new module, retraining strategy, possible aux-
loss tuning), so it runs only if S1/S2 do not close the substrate
question.

## S1 — hard-neg c2p (canonical c2hgi, AL+AZ)

**Status**: code landed at `research/embeddings/check2hgi/model/Check2HGIModule.py`
(new `_sample_hard_negative_indices_c2p` + `c2p_hard_neg_prob` param)
and `research/embeddings/check2hgi/check2hgi.py` (`--c2p_hard_neg_prob`
CLI). Default 0.0 reproduces canonical c2hgi exactly. Existing
check2hgi tests (n=6) pass.

**Hypothesis.** The c2p boundary defines c2hgi's POI vectors. Random-
global negatives are too easy (a coffee-shop check-in vs POI #4521 is
almost always a different category in a different region — trivial
discrimination). Sampling from same-region different-POI forces the
discriminator to pick up finer-grained signal. If the c2p signal was
under-trained, sharpening it should propagate up to the POI level
that POI2Region consumes.

**Build plan** (one shell per state):

```bash
# AL canonical baseline already exists; this is the c2p-hardneg arm.
python research/embeddings/check2hgi/check2hgi.py \
  --city Alabama \
  --c2p_hard_neg_prob 0.25

python research/embeddings/check2hgi/check2hgi.py \
  --city Arizona \
  --c2p_hard_neg_prob 0.25
```

(0.25 mirrors HGI's existing p2r hard-neg rate. Pick 0.25 for the
first arm; if null, sweep ∈ {0.1, 0.5} only if S1 was directionally
positive but underpowered.)

**Eval plan**:

```bash
python scripts/p1_region_head_ablation.py \
  --state alabama --heads next_getnext_hard \
  --folds 5 --epochs 50 --seed 42 --input-type region \
  --region-emb-source check2hgi_canonical_c2p_hardneg_0_25 \
  --override-hparams d_model=256 num_heads=8 \
    transition_path=output/check2hgi/alabama/region_transition_log_seed42_fold1.pt \
  --per-fold-transition-dir output/check2hgi/alabama \
  --tag STL_ALABAMA_canonical_c2p_hardneg_0_25_reg_gethard_pf_5f50ep_leakfree
```

Repeat for AZ. Cat F1 evaluation via the standard `next_gru` 5f×50ep
recipe.

**Decision rule** (pre-registered):

- **Reg lift ≥ 0.5 pp on either AL or AZ** vs canonical baseline,
  Wilcoxon one-sided greater p ≤ 0.0625 → S1 succeeds. Substrate
  hypothesis is correct; proceed to S2 to confirm compounding with
  J.
- **Reg lift < 0.3 pp on both states OR cat F1 regresses ≥ 0.5 pp** →
  S1 null. Move to S4 (corruption rewiring) without sweeping
  hard_neg_prob further.
- **Reg between 0.3-0.5 pp on one state, null on other** → run a
  hard_neg_prob sweep ∈ {0.1, 0.5} on the directionally-positive
  state before deciding.

Also log: fclass POI probe (must stay 4 % on canonical — if it shifts
we have a generality side-effect worth noting).

**Outputs land in**:
- `output/check2hgi/{state}/embeddings_c2p_hardneg_0_25.parquet`
  (the build script writes the canonical filename — rename or version
  via output dir before the run)
- `docs/results/P1/region_head_{state}_..._c2p_hardneg_0_25_leakfree.json`
- `docs/results/paired_tests/substrate_audit_S1.json`
  (created by the finalize step; mirror `design_audit_al_az.json`'s
  schema)

## S2 — S1 stacked on J (AL+AZ)

Only if S1 succeeds. Build a J variant with `c2p_hard_neg_prob=0.25`
during the c2hgi pretraining that J's POI table is anchored against.
Same eval as J. Decision rule: reg lift ≥ 0.3 pp **on top of S1's
lift** in the same direction → substrate fix is composable; the
merge family gains free lift.

## S4 — `neg_poi_emb` at c2p (canonical AL)

Wire `neg_poi_emb[checkin_to_poi]` (same-identity, corrupted-feature
encoding) into `neg_poi_expanded` at the c2p boundary, as a third
option (canonical / hard-neg / corrupted-neg). DGI's original
formulation. ~1-2 h to implement (a few lines + a flag).

Decision rule: reg lift ≥ 0.3 pp without cat regression ≥ 0.5 pp.
Otherwise close.

## S3-b pre-registration (2026-05-07 — REOPEN)

**Hypothesis (user's original)**: replace `POI2Region` with a structural
analog `Checkin2Region` that pools per-check-in embeddings directly to
regions. The c2hgi POI level is preserved as an auxiliary path serving
the cat boundary, but the region pathway is `pos_checkin_emb →
Checkin2Region(zone=poi_to_region[checkin_to_poi]) → region_emb`. Loss
becomes `L_c2p + L_c2r + L_r2c` (where `L_c2r` replaces `L_p2r` —
contrastive between check-in vs region, foreign-region negs HGI-style).

**Variant ladder (tournament, one-redo limit)**:

- **V1 (start here)**: `Checkin2Region = POI2Region(D, num_heads)`
  with `zone=checkin_to_region`. Zero new design. Tests "does removing
  the redundancy and giving the PMA full p2r gradient overcome the
  S3-a uniform-attention collapse?"
- **V2 (only if V1 directionally positive but null)**: brought to
  advisor as a specific recommendation request, *not* pre-enumerated.
  Don't pre-commit to e.g. Set Transformer / time-aware pooling /
  hierarchical attention — the advisor's value is in narrowing.

**Pre-registered gates** (apply to V1 first):

1. **Cat preservation** — AL cat F1 (full 5f×50ep `next_gru`) within
   2 pp of *canonical c2hgi* (40.76); i.e., ≥38.76. S3-b changes the
   gradient flow through Checkin2POI (no parallel canonical path to
   `.detach()`), so cat-side risk is real and bounded by 2 pp.
2. **Reg lift gate** — AL mean Acc@10 ≥ 0.5 pp over J's 61.95;
   AZ folds {0,1,2,4} mean ≥0 over J (allow ~−1-2 pp drag on f3 from
   the architectural-sensitivity floor).
3. **Smoke gates (50-ep AL, before committing 500-ep)**:
   - *Loss decreases monotonically* across epochs 1, 25, 50
     (sanity — model trains).
   - *Reg Acc@10 fold-0 quick eval at ep50 ≥ 45 %*. Canonical
     full-trained 1-fold AL is ~58 %; ep50 partial training would
     be lower; 45 % is a "not broken" floor, not a "lifting" gate.
     If S3-b V1 at ep50 is at 30 %, architecture is broken — kill.
   - *Cat F1 fold-0 quick eval at ep50 ≥ 35 %*. Canonical full-
     trained AL cat is 40.76 %; "not catastrophic" floor.
   - **PR_norm gate dropped** (per canonical calibration: PR_norm=1.0
     at ep50 is the architecture's normal state, not a failure mode).
4. **One-redo rule** — V1 fails any pre-registered gate → V2 with a
   specific advisor-recommended alternative, *one* iteration. V2
   falsified → S3-b closed. No third architecture.

**Failure modes to watch**:

- *PMA collapse re-occurs*: same kill as S3-a. PR_norm > 2× canonical
  at epoch 50.
- *Cat regression*: gradient through Checkin2POI now sees region-
  level loss back-prop differently than under canonical. If cat
  drops > 2 pp at 50-ep, kill.
- *AZ-f3 floor*: applies to any architectural change (Phase 11
  finding). Expected ~−1-2 pp at AZ f3; not a kill, just expected drag.
- *Sparse regions*: log per-region check-in count distribution
  before launch — regions with 0 check-ins have zero L_c2r gradient
  (`nan_to_num` is the safety net but the boundary is unsupervised
  for those regions).

**Implementation watch** (per advisor):

Build `Check2HGI_S3b` as a fresh subclass of `Check2HGI_DesignJ`, not
copy-paste S3-a structure. S3-a's mental model is auxiliary-boundary;
S3-b's is **HGI's `HGIModule.forward` + `HGIModule.loss` pattern with
check-ins instead of POIs at the bottom level**. Reference
`research/embeddings/hgi/model/HGIModule.py` for the canonical
foreign-region neg sampling (gather all check-ins from a sampled
foreign region, pair them against the positive region's embedding) —
that's what we want as the *primary* p2r-equivalent boundary.

## S3 — direct `Checkin2Region` pooler (AL)

The user's methodology reframe. Two implementation sketches:

**S3-a (cheapest)**: add a parallel `Checkin2Region` PMA pooler that
pools `pos_checkin_emb` directly into regions using `checkin_to_poi
∘ poi_to_region`. Compute a fourth contrastive boundary
`L_c2r = -log σ(checkin · W_c2r · region_via_checkin)`. Keep the
existing 3 boundaries; add `alpha_c2r` (start at 0.2; sweep ∈ {0.1,
0.3} if directional). The POI level is preserved as a regulariser via
its existing c2p boundary, so cat does not lose the POI-aware path.

**S3-b (full reframe)**: replace `pos_region_emb = poi2region(...)`
with `pos_region_emb = checkin2region(pos_checkin_emb, ...)`. Drop
the POI bottleneck from the region pathway entirely. POI level only
exists for the c2p loss (auxiliary). Highest variance; only run if
S3-a is directionally positive.

Decision rule for S3-a: reg lift ≥ 0.5 pp at AL with cat non-
inferior at TOST 2 pp → methodology hypothesis confirmed; expand to
AZ + FL.

## Pre-registration: what we will and will not claim

- **Will claim** if any of S1/S2/S4 shows ≥ 0.5 pp Wilcoxon-strict
  lift on reg without cat regression: "the c2hgi substrate had a
  measurable substrate-level slack inherited by the whole merge
  family; closing it lifts X by Y pp."
- **Will claim** if S3-a shows ≥ 0.5 pp lift: "the POI bottleneck on
  the region pathway is methodologically suboptimal for c2hgi's
  pooled-check-in inputs; a direct check-in→region path closes Z pp
  of the HGI gap."
- **Will not claim** anything from a single-state ≥ 0.3 pp lift if
  the other state is null. With n=5 folds the noise floor is real;
  cross-state replication is the gate.
- **Will not claim** the merge-family Pareto verdict from Phase 10 is
  invalidated. Substrate fixes are *composable* with the merge
  family; if successful they shift the absolute level of every
  family member roughly equally.

## Files to update on completion

- `HISTORY.md`: append per-step result blocks (Phase 11.S1, 11.S2, …)
- `STATE.md`: update the headline table only if a substrate fix
  produces a new dominance regime
- `AUDIT_HGI_GAP.md`: add a "Lever 7 — substrate" entry under
  §4 with the empirical result (mirrors the L1, L3, L6 entries)
- This file: mark each test ✓/✗ inline next to its block

## What's running / pending

- S1: ✗ falsified (AL Δ=−1.21 pp, AZ Δ=−0.81 pp). Code at
  `Check2HGIModule.py` (`c2p_hard_neg_prob`) + `build_substrate_s1.py`.
- S2: skipped (depended on S1 success).
- S4: ✗ falsified (AL Δ=−0.07 pp, AZ Δ=−1.16 pp). Code at
  `Check2HGIModule.py` (`c2p_corrupted_neg`) + `build_substrate_s4.py`.
- S3-a: ✗ falsified (both v1 same-identity and v2 foreign-region
  formulations killed at 50-ep PMA-entropy probe; PR_norm=1.0
  uniform attention; supervision redundant with existing 3-boundary
  path). Code at `Check2HGI_S3a` in
  `scripts/probe/build_substrate_s3a.py`.
- S3-b: not run (was conditional on S3-a directional-positive).
- Pipeline: idle. Phase 11 final-closed.
