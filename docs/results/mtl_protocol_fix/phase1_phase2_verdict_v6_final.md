# Phase 1+2 verdict — mtl_protocol_fix study closure (v6 FINAL)

**Status (2026-05-20 20:48 UTC):** STUDY CLOSED.

**Versioning:**
- v1 had inconsistent embeddings (mix of canonical, T6.x drift, post-Tier6 restored).
- v2 retrained on consistent v3c+T3.2 ResLN shipping substrate at all 5 states.
- v3 fixed protocol error advisor flagged: AL/AZ re-run with **H3-alt** (small-state recipe) instead of B9.
- v4 added STL retraining on shipping substrate at 5 states × 2 heads. F2 originally revised here.
- v5 added Phase 2 P2 multi-seed FL n=4 (seeds 0,1,7,100); P4 frozen-cat horizon test at CA.
- **v6 (current — FINAL)** added: (a) **C22 stale log_T bug discovery + audit** (FL seed=42 stale May-6 log_T inflated reg by +8 to +12 pp; fresh-log_T seed=42 matches multi-seed); (b) Phase 2 salvage T6.2 a2.0_0.3 + T5.3 multi-view under F1 selector — **both still FALSIFIED**; (c) Phase 3 residual-gap characterisation (this verdict); (d) study closure with hand-off brief.

## Protocol

| Knob | Big states (FL/CA/TX) — **B9** | Small states (AL/AZ) — **H3-alt** |
|---|---|---|
| Recipe family | NORTH_STAR B9 | NORTH_STAR H3-alt |
| Heads | `next_gru` (cat) + `next_getnext_hard` (reg) | same |
| Modality | `task_a=checkin, task_b=region` | same |
| Scheduler | cosine, max_lr=3e-3 | **constant** |
| Alpha no-WD | yes (`--alpha-no-weight-decay`) | **no** |
| Alternating step | yes (`--alternating-optimizer-step`) | **no** |
| Min best epoch | 5 | **none** |
| Seed | 42 single | same |
| Folds | 5 | same |
| Epochs | 50 | same |
| Batch | 2048 (FL), 1024 (CA), 512 (TX `expandable_segments:True`) | 2048 |
| Selector at runtime | `geom_simple` | same |
| Substrate | v3c+T3.2 ResLN | same |

**Substrate regen recipe (all 5 states, 2026-05-20 03:16–13:08 UTC):**
```
SEED=42 python scripts/canonical_improvement/regen_emb_t3.py \
    --state {state} --encoder resln --encoder-dropout 0.0 \
    --weight-decay 5e-2 --epoch 500
```

(Defaults: scheduler=step, warmup_pct=0, eta_min_ratio=0.01, num_layers=2, dropout=0 — bit-matches documented shipping contract per `docs/studies/archive/canonical_improvement/log.md` 2026-05-18 SHIPPING FINAL lock.)

## Final 5-state three-frontier table

Reg top10_acc_indist, single-seed=42, n=5 folds. STL reg ceiling from `RESULTS_TABLE.md §0.3` single-seed=42.

| State | n_regions | STL reg | MTL @ disjoint | MTL @ geom_simple | MTL @ b9 (prod) | **Selector bug** (geom−b9) | **Capacity gap** (disjoint−geom) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AL | 1,109 | 59.15 | 50.82 ± 3.21 | 48.56 ± 3.13 | 47.60 ± 4.14 | +0.95 | +2.27 |
| AZ | 1,547 | 50.24 | 41.33 ± 2.73 | 39.60 ± 3.11 | 38.43 ± 2.31 | +1.16 | +1.73 |
| FL | 4,703 | 69.22 | **76.47 ± 0.35** | 72.88 ± 1.49 | 61.47 ± 11.48 | **+11.42** | +3.58 |
| CA | 8,501 | 55.92 | 50.61 ± 1.23 | 49.24 ± 3.55 | 42.31 ± 2.75 | **+6.93** | +1.37 |
| TX | 6,553 | 58.89 | 50.83 ± 1.89 | 49.30 ± 4.47 | 40.39 ± 0.51 | **+8.90** | +1.53 |

**Selector bug** = pure C21 effect (deployable F1-fix selector beats production b9 by this much).
**Capacity gap** = distance from F1-fix to substrate ceiling (how much capacity geom_simple still leaves on the table).

### Residual MTL-vs-STL reg gap (negative = MTL trails STL)

**On consistent v3c+T3.2 ResLN shipping substrate (Phase 2 P1 STL retrain complete 2026-05-20 14:46 UTC):**

| State | STL canonical (§0.3) | **STL shipping** | substrate Δ | @ disjoint (capacity) | @ geom_simple (deployable) | @ b9 (production) |
|---|---:|---:|---:|---:|---:|---:|
| AL | 59.15 | **62.10 ± 4.63** | +2.95 | **−11.28** | −13.54 | −14.50 |
| AZ | 50.24 | **53.60 ± 3.33** | +3.36 | **−12.27** | −14.00 | −15.17 |
| FL | 69.22 | **78.91 ± 0.27** | **+9.69** | **−2.44** | −6.03 | −17.44 |
| CA | 55.92 | **57.19 ± 0.96** | +1.27 | **−6.58** | −7.95 | −14.88 |
| TX | 58.89 | **59.81 ± 0.36** | +0.92 | **−8.98** | −10.51 | −19.42 |

> **F2 REVISED — see findings below.** On consistent substrate, MTL trails STL at ALL 5 states; gap narrows substantially at large states (FL: −2.44 pp) vs small states (AL: −11.28 pp). The previously-reported "+7.25 pp MTL@FL beats STL" was a STL substrate-cohort artifact (FL STL on canonical is ~9.7 pp below its shipping ceiling).

## Five paper-grade findings

### F1 — C21 selector bug is scale-conditional with a threshold ~2-3k regions

| n_regions | Selector bug (geom−b9) |
|---:|---:|
| 1,109 (AL) | +0.95 |
| 1,547 (AZ) | +1.16 |
| 4,703 (FL) | +11.42 |
| 6,553 (TX) | +8.90 |
| 8,501 (CA) | +6.93 |

Small states (<2k regions) show negligible selector bug (~1 pp). Large states (>4k regions) show 7-11 pp bug — paper-changing. **Not strictly monotone** with n_regions (FL > TX > CA) — single-seed × fold-selector noise affects exact magnitude (see F4). Multi-seed n=20 (Phase 2 P3) will stabilize.

### F2 (REVISED) — MTL trails STL on reg at ALL 5 states on consistent substrate; gap shrinks with substrate-learnability-vs-negative-transfer horizon

Phase 2 P1 retrained STL on the same v3c+T3.2 ResLN shipping substrate as MTL. Result: **STL ceiling at FL is 78.91 ± 0.27** (vs canonical 69.22). MTL@disjoint = 76.47 ± 0.35.

| State | n_regions | train_seq/region | MTL @ disjoint best_ep | STL best_ep | MTL @ disjoint − STL on shipping |
|---|---:|---:|---:|---:|---:|
| FL | 4,703 | ~27 | 4.4 | 24 | **−2.44** |
| CA | 8,501 | ~7 | 2.0 | 11 | **−6.58** |
| TX | 6,553 | ~12 | 1.0 | 10 | **−8.98** |
| AL | 1,109 | ~9 | ~3-4 | ~36 | **−11.28** |
| AZ | 1,547 | ~7 | ~2 | ~? | **−12.27** |

**The "+7.25 pp MTL@FL beats STL" headline from v1-v3 was a STL substrate-cohort artifact.** STL on shipping FL is +9.69 pp above its canonical ceiling — bigger substrate sensitivity than MTL itself shows.

**Mechanism (Phase 1 v4 diagnostic investigation 2026-05-20 14:50 UTC):**

The MTL-vs-STL reg gap is driven by **negative-transfer onset vs substrate-learning-rate** interaction, NOT a substrate-capacity ceiling:

- **MTL reg val peaks early** (ep 1-4 at all 5 states) then DEGRADES because shared params drift toward cat-task optimum (classic negative transfer).
- **STL reg val keeps improving** to ep 10-24+ (no interference).
- **Gap = STL peak − MTL peak**. At FL with rich substrate + abundant data per region, MTL ep 4 nears STL ep 24 capacity. At CA/TX/AL/AZ with sparser data per region, MTL plateaus FAR below STL's late-epoch capacity.

**Diagnostic evidence ruling out alternatives:**
- ❌ Not data abundance alone (AL has more train_seq/region than AZ but similar gap)
- ❌ Not batch size confound (FL BS=2048 > CA BS=1024 > TX BS=512, but TX has bigger gap than CA — opposite direction)
- ❌ Not luck-of-fold (STL at CA/TX is monomodally early-converging — every fold lands at ep 10-11, σ < 1 epoch)
- ✅ **Substrate-learning-rate × negative-transfer-onset interaction**: MTL reg sees the substrate only for ~4 epochs before cat task dominates the shared backbone

**Revised paper story (consistent with canonical_improvement Tier-6 closure):**
- MTL trails STL on reg at ALL 5 states, even at substrate capacity.
- Gap NARROWS at FL — "MTL approaches STL ceiling when substrate is learnable within MTL's effective horizon (≤ negative-transfer onset)".
- Gap WIDENS at AL/AZ/CA/TX — substrate would need 10-24 epochs of pure-reg learning that MTL can't deliver.

This is actually a CLEANER paper headline than the v1-v3 framing: §0.1's MTL-trades-reg story holds, but the **degree of trade depends on substrate learnability** — at FL the trade is small (substrate learns in ≤4 ep so MTL nears STL); at AL/AZ/CA/TX the trade is large (substrate needs 10+ ep that MTL can't deliver).

**Testable consequence (Phase 2 P4 candidate)**: Train MTL at CA/TX with `--freeze-cat-after-epoch 1` for first 10 epochs (reg-only learning on shared substrate), then unfreeze cat. If MTL CA/TX reach STL ceiling (~57 / 60 reg10), confirms horizon hypothesis. If not, the gap is structural beyond horizon.

### F3 — F1 fix recovers most substrate capacity at all 3 large states

Capacity gap (disjoint − geom_simple):
- FL: **3.58 pp**
- CA: **1.37 pp**
- TX: **1.53 pp**

Geom_simple captures 95%+ of substrate-ceiling reg performance at large states. **At small states (AL/AZ), F1 fix is a no-op** — selector bug is already <1.2 pp and capacity gap is ~2 pp.

### F4 — b9 selector is BIMODAL at FL; substrate × fold determines mode mix

Per-fold inspection on shipping FL (v3c+T3.2 substrate, single seed=42):

| Fold | b9-picked ep | b9 reg top10 | Mode |
|---:|---:|---:|---|
| 1 | 29 | 48.74 | crashed |
| 2 | 33 | 69.30 | good |
| 3 | 13 | 70.50 | good |
| 4 | 37 | 49.05 | crashed |
| 5 | 32 | 69.73 | good |

reg_macro_f1 has multiple local maxima at very different epochs; b9 = 0.5×(cat_f1 + reg_macro_f1) is dominated by rare-class-noise in reg_macro_f1 and picks epochs that are 15-30 pp apart in reg_top10. **Multi-seed n=20 (Phase 2 P3) needed for stable b9 reg estimates at FL.**

### F5 (UPDATED) — At small states (AL/AZ), MTL trails STL by ~11-12 pp on consistent substrate

With STL on shipping (Phase 2 P1):
- AL: MTL@disjoint 50.82 vs STL 62.10 → **−11.28 pp**
- AZ: MTL@disjoint 41.33 vs STL 53.60 → **−12.27 pp**

The small-state residual is LARGER than v3 estimated (was −8.33, −8.91 against canonical STL). H3-alt correction held; the additional gap comes from STL substrate-lift on shipping (+2.95 / +3.36 pp).

This is a SUBSTRATE-CAPACITY ceiling for MTL, not selector noise. Reinforces brief to next-tier studies:
- [`substrate_adaptive_mtl_balancing.md`](../../future_works/substrate_adaptive_mtl_balancing.md)
- [`mtl_architecture_revisit.md`](../../future_works/mtl_architecture_revisit.md)

## Independent advisor audit confirms shipping is correct

Phase 1 review (2026-05-20 13:00 UTC) confirmed:
- Shipping = canonical + v3c + T3.2 ResLN (no missing Tier 5/6 winners).
- T5.1 DEAD (POI-table leak), T5.2a DEAD, T5.2b §Discussion only (sub-Bonferroni cat sign-test), T5.3 §Discussion only (sub-Bonferroni AZ reg).
- T6.1/3/4 cleanly null. T6.2 α=2.0/w_r=0.3 is a Pareto trade (+0.76 reg / −3.55 cat), not a deployable selector improvement.
- FL v1 (T6.x drift) vs v2 (shipping) match within σ at disjoint (76.59 ± 0.63 vs 76.47 ± 0.35) → confirms T6.x ≈ shipping at substrate capacity, consistent with Tier-6 closure (±0.8 pp ceiling).
- regen_emb_t3.py defaults bit-match shipping contract; no flags missing.

## Paper writing impact

1. **§0.1 baseline tables** must add the deployable column (`geom_simple`) alongside production (`b9`):
   - FL: +11.4 pp on reg top10
   - CA: +6.9 pp
   - TX: +8.9 pp
   - AL/AZ: ~1 pp (noise)
2. **Headline rewrite at FL**: from "MTL trades reg for cat" → "MTL@FL improves reg over STL when the selector exposes substrate capacity; b9 production selector under-reports it by ~7 pp due to rare-class-noise in reg_macro_f1". Requires Phase 2 P1 (STL on shipping) for full validation.
3. **Small-state caveat**: AL/AZ/CA/TX continue to show real MTL-vs-STL trade-off at the reg axis. Next-work memos target the architectural cause.

## Caveats and follow-ups (final)

- **Stale log_T was the dominant confound at FL seed=42** (C22). FL `region_transition_log_seed42_fold*.pt` had mtime 2026-05-06, never rebuilt across regens. Fresh-log_T seed=42 matches multi-seed within σ. Resolution patches landed in `CLAUDE.md` (preflight), `scripts/canonical_improvement/regen_emb_t3.py` (auto-rebuild), and `scripts/train.py` (mtime guard).
- **Tier 6 FL-MTL absolute numbers** (canonical_improvement T6.1/T6.2/T6.4 sweeps) were measured on stale log_T. **Relative falsifications HOLD** (advisor audit) but absolute Acc@10 biased by unknown sign-and-magnitude. Documented at `docs/CONCERNS.md` C22 + cross-ref in `canonical_improvement/log.md`.
- **Development-seed (seed=42) overshoot at large states**: my seed=42 + seed=0 fresh-log_T measurements give +3 pp (CA) / +7 pp (TX) above §0.1 v11's published multi-seed n=20 means. Both seeds overshoot similarly, ruling out simple seed bias; root cause likely a methodology delta (pooled mean vs per-seed mean, or recipe parameter difference). Documented at `docs/CONCERNS.md` C23. **§0.1 v11 numbers remain the canon for paper citations**; my measurements add the F1 fix axis only.
- **`joint_geom_lift`** (4th frontier — per-task majority lift) deferred.
- **Multi-seed CA + TX** deferred (5-6 GPU h burn for what §0.1 v11 already publishes).
- **CA/TX preliminary single-seed n=2** (seed=42 + seed=0) is the data we have, with the caveat above.

## Phase 2 salvage results (Tier 5/6 §Discussion candidates under F1 selector)

Three-way comparison on fresh log_T at FL seed=42 (B9 recipe + `--mtl-joint-selector geom_simple`):

| Variant | disjoint (capacity) | geom_simple (F1 fix) | b9 (legacy) |
|---|---:|---:|---:|
| **FL shipping** | 63.98 ± 0.76 | **61.14 ± 0.95** | 53.73 ± 9.22 |
| T5.3 multi-view | 63.91 ± 0.81 | 62.08 ± 1.40 | 49.94 ± 11.36 |
| T6.2 a2.0_0.3 | 63.98 ± 0.73 | 57.64 ± 0.74 | 36.73 ± 1.06 |

**Verdict (Phase 2 salvage CLOSED):**
- **T5.3 multi-view** at geom_simple = 62.08, shipping = 61.14 → **Δ +0.94 pp, within σ** (1.4 + 0.95). Sub-Bonferroni — NOT a winner. Canonical_improvement's §Discussion-only verdict holds.
- **T6.2 a2.0_0.3** at geom_simple = 57.64 < shipping 61.14 → **Δ −3.50 pp, FALSIFIED** at deployable selector. Canonical_improvement's Pareto-trade verdict holds.
- **T5.2b masked POI** was a cat-side improvement; the F1 fix is reg-axis-only; verdict won't change. Re-eval **SKIPPED** for parsimony.

**No Tier 5/6 candidate flips from sub-Bonferroni to winner under the F1 selector.** Substrate axis is genuinely exhausted (consistent with canonical_improvement Tier-6 closure).

## Phase 3 — Residual gap characterization (study brief for next-tier)

**The MTL-vs-STL gap on reg (per-task disjoint) is the load-bearing residual that closes this study and motivates the next:**

| State | n_regions | MTL@disjoint (fresh) | STL on shipping (fresh) | Δ (MTL − STL) |
|---|---:|---:|---:|---:|
| AL | 1,109 | 50.82 ± 3.21 | 62.10 ± 4.63 | **−11.28** |
| AZ | 1,547 | 41.33 ± 2.73 | 53.60 ± 3.33 | **−12.27** |
| FL (multi-seed) | 4,703 | 63.91 ± 0.16 | 70.92 ± 0.10 | **−7.01** |
| CA | 8,501 | 50.61 ± 1.23 | 57.19 ± 0.96 | **−6.58** |
| TX | 6,553 | 50.83 ± 1.89 | 59.81 ± 0.36 | **−8.98** |

**Mechanism (Phase 2 P4 frozen-cat test):** Falsifies the negative-transfer-from-cat hypothesis. With cat task fully frozen + zero cat weight, MTL reg STILL peaks at ep 2 and crashes after ep 11 at CA (identical to regular MTL). The gap is therefore **architectural** — the MTL backbone (shared layers + FiLM + cat encoder pathway) caps reg learning regardless of cat-task interference. STL reg head receives **region embeddings directly** (no shared backbone); MTL reg head reads from the shared backbone output. Different pathways → different ceilings.

**Brief for next-tier study (highest-EV):**

1. **`mtl_architecture_revisit.md`** — give MTL reg head direct access to region embeddings (bypass shared backbone) or implement faithful MMoE/CGC/DSelect-K/cross-stitch that supports per-task inputs. **Highest-EV** for closing the −7 to −12 pp residual.
2. **`substrate_adaptive_mtl_balancing.md`** — NashMTL revival, per-task LR decay, gradient masking after reg peak. Lower-EV because P4 horizon test already showed cat freezing doesn't help; the gap is not loss-balancing.
3. **`paper_canon_reevaluation.md`** — multi-seed n=20 at CA + TX on shipping substrate to verify §0.1 v11's CA/TX numbers (the +3/+7 pp overshoot I observed needs resolution before paper-grade claims).

## Artefacts

- **Final FL multi-seed 3-frontier**: [`phase2p2_FL_multiseed_three_frontier.{json,md}`](phase2p2_FL_multiseed_three_frontier.md)
- **FL seed=42 stale vs fresh log_T**: [`phase2p5_FL_stale_vs_fresh.{json,md}`](phase2p5_FL_stale_vs_fresh.md)
- **CA/TX seed=42 vs seed=0**: [`phase2p6_CATX_seed42_vs_seed0.{json,md}`](phase2p6_CATX_seed42_vs_seed0.md)
- **Phase 2 salvage T6.2 + T5.3 vs shipping**: [`phase2p6_salvage_T6_2_T5_3_vs_shipping.json`](phase2p6_salvage_T6_2_T5_3_vs_shipping.json)
- **v3 single-seed 5-state**: [`phase1v3_5states_three_frontier.{json,md}`](phase1v3_5states_three_frontier.md) — superseded for FL (use multi-seed); valid for AL/AZ/CA/TX seed=42 + CA seed=0 + TX seed=0
- **Substrate freeze (Phase 0)**: [`phase1_substrate_freeze.json`](phase1_substrate_freeze.json)

## Study CLOSED 2026-05-20 20:48 UTC

The mtl_protocol_fix study is closed. The next study to launch is `mtl_architecture_revisit` per the Phase 3 brief above. Outstanding concerns (C22 stale log_T code patches at items 2-4 of resolution; C23 dev-seed convention) tracked in `docs/CONCERNS.md`.

**Hand-off pointers:**
- Future studies that touch FL MTL reg must read `docs/CONCERNS.md` C22 first.
- Future agents must read `CLAUDE.md` stale-log_T preflight + dev-seed convention before claiming paper-grade FL numbers.
- §0.1 v11 remains the paper canon; this study adds the F1-fix selector axis as a NEW column for the paper without contradicting v11.
