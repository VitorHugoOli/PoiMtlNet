# North-Star MTL Configuration

> 🎉 **F51 MULTI-SEED (2026-04-30):** B9 vs H3-alt validated across 5 seeds {42, 0, 1, 7, 100}. **Δreg = +3.48 ± 0.12 pp across seeds; pooled paired Wilcoxon (5 × 5 = 25 fold-pairs): p_reg = 2.98×10⁻⁸ (25/25 positive); p_cat = 1.33×10⁻⁵ (19/25 positive).** Cat reaches paper-grade once seeds pool. Absolute B9 reg σ_across_seeds = 0.11 pp — recipe is essentially deterministic in the partition-difficulty axis. The seed=42 +3.34 pp number was the worst-case seed; cross-seed mean is slightly larger. Full doc: `research/F51_MULTI_SEED_FINDINGS.md`.
>
> 📊 **F51 TIER 2 CAPACITY SWEEP (2026-04-30):** 21 capacity smokes across 7 dimensions (5f×30ep) confirm **B9 is locally optimal in 5/7 capacity dimensions**. No paper-grade lift available via capacity scaling. Two NEW negative findings: (a) **cat width-stability cliff** — wider shared backbone (`shared_layer_size` 384/512, `num_crossattn_blocks=4`, `crossattn_ffn_dim=1024`) breaks cat without affecting reg; (b) **F52's "mixing is dead at FL" is depth-conditional** — alive at `num_crossattn_blocks=3` (Pareto-trade: +0.75 reg / -2.62 cat), breaks cat at depth=4. Full doc: `research/F51_TIER2_CAPACITY_FINDINGS.md`.
>
> ⚠ **PER-SEED log_T LEAK (caught + fixed 2026-04-30 mid-F51-sweep):** the original C4 fix wrote per-fold log_T as `region_transition_log_fold{N}.pt` with no seed in the filename, but the trainer loaded that file regardless of its own `--seed`. At any seed != 42, ~80% of val users live in seed=42's fold-N TRAIN set → ~80% of val transitions leaked back into the prior, inflating absolute reg by ~9 pp. Fix: filename is now `region_transition_log_seed{S}_fold{N}.pt`; trainer hard-fails if missing or if a legacy unseeded file is present. Paired Δs from earlier runs survive (uniform-leak property — both arms read the same wrong prior on the same val set), but absolute numbers from the v1 multi-seed sweep are wrong; v2 (clean) is in `F51_MULTI_SEED_FINDINGS.md`.
>
> ⚠ **C4 LEAKAGE CAVEAT (added 2026-04-29 19:50, F50 T4):** All absolute `next_region` numbers below were measured under the legacy full-data `region_transition_log.pt` graph prior, which leaked val transitions into training. Direct measurement: ~13-17 pp inflation at convergence, propagating through 5 heads (`next_getnext_hard*`, `next_getnext`, `next_tgstan`, `next_stahyper`, `next_getnext_hard_hsm`). **Use `--per-fold-transition-dir` for any future run.** Under leak-free conditions, the committed champion is **B9 (P4 + Cosine + alpha-no-WD)**, headline numbers in the F51 banner above. See `research/F50_T4_C4_LEAK_DIAGNOSIS.md` and `research/F50_T4_BROADER_LEAKAGE_AUDIT.md`. The numbers below are KEPT for historical comparison and method derivation; the absolute targets (e.g. "73.61", "76.07") need a "−15 pp footnote" mental model when read.

**Status (2026-04-29 17:30 UTC, Pareto-corrected):** Champion is **P4 alternating-SGD + Cosine (max_lr=3e-3) + delayed-min selector (`min_epoch=10`)**. Earlier today P4+OneCycle was promoted as champion based on reg-only metrics; closer inspection of the cat-side data shows OneCycle DEGRADES cat F1 by −1.84 pp with one fold collapsing to 62.68 (vs 67-68 in others). **P4+Cosine is the Pareto-dominant variant**: reg +4.63 pp paper-grade (paired Wilcoxon p=0.0312, 5/5 positive), cat tied/slightly improved (+0.15 pp, no fold collapse). P4-alone is also Pareto-dominant (+4.04 reg, cat tied) but P4+Cosine is +0.59 pp stronger on reg.

P4+OneCycle (+6.08 reg / −1.84 cat) is documented as the **reg-only-optimal alternative** for ablation studies that don't constrain cat preservation. See `research/F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md §6.3-§6.4`.

**Status (2026-04-27):** Two complementary tracks now confirm the MTL story from different angles. **The previous recipe (F48-H3-alt per-head LR) is the committed champion**; substrate validation and architecture attribution both back it.

**Architecture-side (F48-H3-alt + F49, 2026-04-26 → 04-27):** Per-head LR recipe validated 5-fold on AL/AZ/FL — cat preserved within ~2 pp of B3, reg Acc@10 lifts by 6.7-15 pp over B3. AL **exceeds** STL F21c ceiling by +6.25 pp; AZ closes 75%; FL is most stable (σ=0.68). Three orthogonal negative controls (F40, F48-H1, F48-H2) bracket H3-alt as the unique design. **F49 attribution (2026-04-27):** the H3-alt reg lift on AL is *purely architectural* (+6.48 pp from architecture alone, F49c 5f × 50ep); cat-supervision transfer is null on all 3 states (≤|0.75| pp), refuting the legacy "+14.2 pp transfer" claim by ≥9σ on FL n=5. CH18 Tier A; CH19 Tier A. See `research/F48_H3_PER_HEAD_LR_FINDINGS.md` + `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`.

**Substrate-side (Phase 1 substrate validation, 2026-04-27):** Five-leg study on AL+AZ confirms the substrate side of the MTL claim:

1. **CH16 confirmed under matched-head, head-invariant** at AL+AZ (8/8 head-state probes positive, p=0.0312 each; ranges +11.58 to +15.50 pp).
2. **CH15 reframed** — under the matched MTL reg head (`next_getnext_hard`), C2HGI ≥ HGI (was "HGI > C2HGI" under STAN). The previous CH15 was head-coupled.
3. **CH18 — MTL B3 is substrate-specific.** Substituting HGI breaks the joint signal (cat −17 pp, reg −30 pp Acc@10_indist at both states; MTL+HGI is *worse than STL+HGI* on reg by ~37 pp).

These findings **do not** change the committed config — they explain *why* it works. See `research/SUBSTRATE_COMPARISON_FINDINGS.md` for the full Phase 1 verdict + `PHASE2_TRACKER.md` for FL/CA/TX replication queue.

**Status (2026-04-24):** Cat head refined via F27 from `NextHeadMTL` (Transformer) → `next_gru` (GRU). Paper-reshaping F21c finding noted in §§Caveats. See §Committed config below.

## Champion — F50 B9 (P4 + Cosine + α-no-WD) — multi-seed validated (2026-04-30 F51)

```
architecture         : mtlnet_crossattn
mtl_loss             : static_weight(category_weight = 0.75)
task_a head (cat)    : next_gru
task_b head (reg)    : next_getnext_hard                # STAN + α · log_T[last_region_idx]
task_a input         : check-in embeddings (9-step window)
task_b input         : region embeddings (9-step window)
hparams              : d_model=256, 8 heads, batch=2048, 50 epochs
                       seeds {42, 0, 1, 7, 100} all paper-grade ✅
LR scheduler         : Cosine(max_lr=3e-3)              # decay from peak
LR per param group   : cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3
optimizer step       : ALTERNATING per-batch (P4) — cat batch then reg batch, separate optimizer.step()
selector             : per-fold-best top10_acc_indist with --min-best-epoch 5
α-no-WD              : alpha scalar peeled out of AdamW weight_decay group (B9 refinement)
per-fold log_T       : MUST be seed-tagged: region_transition_log_seed{S}_fold{N}.pt
                       built via: scripts/compute_region_transition.py --state STATE --per-fold --seed S
```

**Single-line additive recipe vs H3-alt:**
```bash
--alternating-optimizer-step \
--scheduler cosine --max-lr 3e-3 \
--alpha-no-weight-decay \
--min-best-epoch 5 \
--per-fold-transition-dir output/check2hgi/STATE
```

**Multi-seed headline (FL 5f×50ep, leak-free per-seed log_T, ≥ep5):**

| seed | B9 reg ± σ | H3-alt reg ± σ | Δreg | p_reg | n+/n |
|---:|---:|---:|---:|:---:|:---:|
| 42 | 63.47 ± 0.75 | 60.12 ± 1.15 | **+3.34** | 0.0312 | 5/5 |
| 0 | 63.24 ± 0.89 | 59.58 ± 0.95 | **+3.65** | 0.0312 | 5/5 |
| 1 | 63.41 ± 1.16 | 60.02 ± 1.03 | **+3.39** | 0.0312 | 5/5 |
| 7 | 63.21 ± 0.50 | 59.72 ± 0.54 | **+3.49** | 0.0312 | 5/5 |
| 100 | 63.38 ± 0.93 | 59.87 ± 1.17 | **+3.51** | 0.0312 | 5/5 |
| **mean** | **63.34 ± 0.11** (across seeds) | 59.86 ± 0.22 | **+3.48 ± 0.12** | — | — |

**Pooled paired Wilcoxon (25 fold-pairs):** Δreg = +3.48 pp, **p = 2.98×10⁻⁸**, 25/25 positive. Δcat = +0.42 pp, **p = 1.33×10⁻⁵**, 19/25 positive.

Full doc: `research/F51_MULTI_SEED_FINDINGS.md`.

**⚠ Historical numbers below — kept for method derivation. The Pareto picture below was measured under the LEAKY full-data log_T at seed=42 only.** Under leak-free per-fold log_T (the current C4 fix) the absolute reg drops by ~13 pp uniformly; under multi-seed averaging the +3.48 pp Δ is stronger evidence than the single-seed +4.63 pp shown below. See the F51 multi-seed table above for the current paper-grade numbers.

**Pareto picture — paired Wilcoxon vs H3-alt (FL 5f × 50ep, seed 42, LEAKY):**

| metric | H3-alt | P4 alone | **P4 + Cosine** ⭐ | P4 + OneCycle |
|---|---:|---:|---:|---:|
| reg top10 @ ≥ep10 | 71.44 ± 0.76 | 75.48 ± 0.75 | **76.07 ± 0.62** | 77.52 ± 0.53 |
| Δreg vs H3-alt | — | +4.04 | **+4.63** | +6.08 |
| p(reg > H3-alt) | — | **0.0312** ✅ | **0.0312** ✅ | **0.0312** ✅ |
| cat F1 @ best | 68.36 ± 0.74 | 68.20 ± 0.69 | **68.51 ± 0.88** | 66.52 ± 2.29 ⚠ |
| Δcat vs H3-alt | — | −0.16 | **+0.15** | **−1.84** |
| Pareto verdict | predecessor | dominant ✅ | **dominant 🏆** | **TRADE** ⚠ |

**Per-fold reg @ ≥ep10:**
- P4 + Cosine: `[75.88, 75.10, 76.18, 76.60, 76.59]` mean 76.07, best_eps `{10,10,10,10,11}`
- vs H3-alt deltas: `[+5.02, +4.43, +4.98, +4.17, +4.54]` (5/5 positive, σ_Δ = 0.36)

**Per-fold cat F1:** P4+Cosine `[69.58, 67.23, 68.11, 68.84, 68.80]` — stable, no outlier. Compare to P4+OneCycle `[68.60, 66.38, 67.35, 62.68, 67.60]` — fold 4 collapses ~5 pp below others, blowing σ from 0.74 → 2.29.

**Mechanism (Pareto-aware):** P4 alternating-SGD is the necessary substrate — it gives reg its own optimizer step on its own batch, preventing post-ep-5 cat-dominance collapse. The scheduler choice then trades reg strength against cat stability:
- **Cosine** (decay from peak) — α gets early boost (best_ep ~ep 4-6 in greedy view), P4's separation preserves it through ep 10+, then graceful decay protects cat throughout. Net: +4.63 reg, +0.15 cat → Pareto-dominant.
- **OneCycle** (warmup → peak at ep 19-20) — second growth window for reg, but the high LR at ep 19-20 destabilises cat in some folds. Net: +6.08 reg, −1.84 cat → Pareto-trade.

**Tier-A negative controls confirm P4 is necessary** — every non-P4 config either fails reg significance or trades cat (or both). OneCycle without P4 (A1) underperforms H3-alt by 9 pp; cosine without P4 (A2) collapses to 67.59 ± 8.99 at ≥ep10. See `research/F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md §6.4` for the full table.

**Reg-only-optimal alternative — P4 + OneCycle** (run `_1636`): when cat preservation isn't a constraint (e.g. ablation studies focused on reg), `--scheduler onecycle --max-lr 3e-3 --pct-start 0.4` lifts reg by an additional +1.45 pp (77.52 vs 76.07) at the cost of a single-fold cat collapse. Documented but **not the committed default for joint MTL**.

---

## Predecessor — F50 P4 alone + delayed-min (2026-04-29, superseded same day)

P4 alternating-SGD with constant scheduler — the first paper-grade fix for the FL gap. Headline: 75.48 @ ≥ep10 (vs H3-alt's 71.44; +4.04 pp, paired Wilcoxon p=0.0312, 5/5 positive). Superseded by P4+OneCycle which is +2.04 pp stronger by composing with OneCycle's late-LR peak. Run dir: `_0520`.

---

## Predecessor — F48-H3-alt (2026-04-26)

```
architecture         : mtlnet_crossattn
mtl_loss             : static_weight(category_weight = 0.75)
task_a head (cat)    : next_gru
task_b head (reg)    : next_getnext_hard                # STAN + α · log_T[last_region_idx]
task_a input         : check-in embeddings (9-step window)
task_b input         : region embeddings (9-step window)
hparams              : d_model=256, 8 heads, batch=2048, 50 epochs, seed 42
LR scheduler         : constant (no OneCycleLR / no annealing)
LR per param group   : cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3   # ← new
```

**Single-line additive recipe vs B3:**
```bash
--scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3
```

**Headline numbers (5-fold × 50ep, seed 42):**

| State | cat F1 (B3 → H3-alt) | reg Acc@10 (B3 → H3-alt) | vs STL F21c |
|---|---|---|---|
| **AL** | 42.71 → **42.22 ± 1.00** (-0.49) | 59.60 → **74.62 ± 3.11** (+15.02) | **+6.25 EXCEEDS** ✓ |
| **AZ** | 45.81 → **45.11 ± 0.32** (-0.70) | 53.82 → **63.45 ± 2.49** (+9.63) | -3.29 (closes 75% of gap) |
| **FL** | 65.72† → **67.92 ± 0.72** (+2.20) | 65.26† → **71.96 ± 0.68** (+6.70) | TBD (F37 4050-assigned) |

†FL B3 ref is F32 1-fold n=1.

**Mechanism (single sentence):** α (graph-prior weight in `next_getnext_hard.head`) needs sustained 3e-3 to grow → reg lift; `shared_lr=1e-3` keeps cross-attn gentle so the cat path stays stable; `cat_lr=1e-3` keeps the cat encoder/head from diverging. The earlier monolithic-LR family (F44-F48-H2) couldn't satisfy both simultaneously because it forced α and the cat path to share an LR.

**Attribution refinement (F49, 2026-04-27):** the H3-alt mechanism above is the *operational* story — it explains why the optimizer recipe works. F49's 3-way decomposition asked the *causal* question — "what does the resulting MTL model do that STL `next_getnext_hard` doesn't?" — and showed the answer is **architecture, not cat-supervision transfer**. On AL the architecture alone (encoder-frozen λ=0, frozen-random cat features) lifts reg by +6.48 pp over STL F21c; cat-supervision via L_cat adds ≈ 0; cross-attn-mediated cat-encoder co-adaptation also adds ≈ 0. AZ shows the classical "architectural overhead, multi-task wrap rescues" pattern. FL's frozen variant is unstable (separate caveat). Operationally H3-alt is unchanged; the *paper claim* is sharpened from "joint MTL transfers cat→reg signal" to "the cross-attention architecture under the per-head LR regime extracts more reg signal from the same input than STL can." See `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` and `CLAIMS_AND_HYPOTHESES.md §CH19`.

**Cross-checks landed:**
- F40 (loss-side cat_weight ramp 0.75→0.25) — cat OK, reg only +1 pp → loss balance is not the lever
- F48-H1 (constant 1e-3 everywhere) — cat OK, reg flat → α needs LR ≥ 2e-3 to grow
- F48-H2 (warmup_constant 50→3e-3 plateau, single LR) — cat OK, reg WORSE → cat-vs-reg compete for shared cross-attn capacity at plateau LR
- F48-H3 (per-head with `shared_lr=3e-3`) — cat collapsed → shared cross-attn at 3e-3 destabilises cat path

H3-alt is the unique configuration in this design space. See `research/F48_H3_PER_HEAD_LR_FINDINGS.md` for the full derivation, `research/F48_H2_WARMUP_CONSTANT_FINDINGS.md` and `research/F40_SCHEDULED_HANDOVER_FINDINGS.md` for the negative controls, and `MTL_ARCHITECTURE_JOURNEY.md` for the end-to-end narrative from initial design to the current recipe.

**Note on `experiments/check2hgi_up/run_mtl_b3.py` and `docs/COLAB_GUIDE.md`:** both use the **predecessor B3 recipe** (`--max-lr 0.003`, no per-head LR), not H3-alt. This is deliberate for the check2hgi-up embedding-variant study (B3 is the established fair-comparison harness for downstream MTL, so embedding-variant-vs-baseline deltas are interpretable). For new MTL claims against STL, use the H3-alt recipe above. To extend `experiments/check2hgi_up` to H3-alt, append `--scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3` to the `run_mtl_b3.py` command and rename the script accordingly. Tracked in `PAPER_PREP_TRACKER.md §2.3` as a camera-ready follow-up.

## Predecessor — B3 50ep (2026-04-24, kept for reference)

**B3 champion (the predecessor recipe):**

```
architecture         : mtlnet_crossattn
mtl_loss             : static_weight(category_weight = 0.75)   # reg weight = 0.25
task_a head (cat)    : next_gru                                 # ← updated 2026-04-24 (F27)
task_b head (reg)    : next_getnext_hard                        # STAN + α · log_T[last_region_idx]
task_a input         : check-in embeddings (9-step window)
task_b input         : region embeddings (9-step window)
hparams              : d_model=256, 8 heads, max_lr=0.003, batch=2048, 50 epochs, seed 42
LR scheduler         : OneCycleLR (PyTorch default)
```

**F27 cat-head refinement — 5-fold AZ paired Wilcoxon** (B3 default `next_mtl` cat-head vs B3 `next_gru` cat-head):
| Metric | Δ mean | p_greater | Verdict |
|---|---:|---:|---|
| cat F1 | +2.37 pp | **0.0312** ✅ | significant |
| cat Acc@1 | +4.69 pp | **0.0312** ✅ | significant |
| reg MRR_indist | +1.50 pp | **0.0312** ✅ | significant |
| reg Acc@10_indist | +1.98 pp | 0.0625 | marginal |
| reg Acc@5_indist | +1.69 pp | 0.0625 | marginal |

Per-fold, all 5 cat folds positive on both cat F1 and cat Acc@1. See `research/F27_CATHEAD_FINDINGS.md` and `scripts/analysis/az_b3_cathead_wilcoxon.py`.

## AZ headline numbers under the new B3 (2026-04-24)

| Metric | Value |
|---|---:|
| Cat F1 | **0.4581 ± 0.0130** |
| Cat Acc@1 | **0.4930 ± 0.0067** |
| Reg Acc@10_indist | 0.5382 ± 0.0311 |
| Reg Acc@5_indist | 0.4054 ± 0.0340 |
| Reg MRR_indist | 0.2766 ± 0.0241 |

**vs STL Check2HGI cat (matched-class):** cat F1 0.4208 ± 0.0089 → **Δ = +3.73 pp** (much stronger than the pre-F27 +1.65 pp).
**vs STL STAN (reg ceiling):** reg Acc@10 0.5224 ± 0.0238 → Δ = +1.58 pp (tied within σ).
**vs STL GETNext-hard (F21c matched-head reg baseline):** reg Acc@10 0.6674 ± 0.0211 → Δ = **−12.92 pp** (MTL still trails on reg — F21c finding persists).

## Caveats — Phase-1 substrate-specific addendum (2026-04-27)

**MTL B3 only works with Check2HGI substrate.** Phase-1 Leg III (MTL counterfactual with HGI substituted, 5f × 50ep, seed 42 each at AL+AZ):

| State | MTL+C2HGI cat F1 | MTL+HGI cat F1 | MTL+C2HGI reg Acc@10_indist | MTL+HGI reg Acc@10_indist |
|---|---:|---:|---:|---:|
| AL | **42.71** | 25.96 | **59.60** | 29.95 |
| AZ | **45.81** | 28.70 | **53.82** | 22.10 |

The MTL configuration was tuned around Check2HGI's per-visit context. Substituting POI-stable HGI embeddings into the same B3 setup actively **breaks the reg head** (MTL+HGI Acc@10 = 29.95 < STL+HGI gethard Acc@10 = 67.52 at AL — a 37 pp regression vs the standalone HGI baseline). Paper framing implication: the MTL win is **interactional** with the substrate; F49's architectural attribution further qualifies it as **architecture interacts with substrate** (not "transfer happens").

Source: `results/hgi/{alabama,arizona}/mtlnet_lr1.0e-04_bs2048_ep50_20260427_*`. Full table in `CLAIMS_AND_HYPOTHESES.md §CH18`, `OBJECTIVES_STATUS_TABLE.md §2.4`, and the Phase-1 verdict at `research/SUBSTRATE_COMPARISON_FINDINGS.md`.

## Caveats — the F21c finding (SCALE-CONDITIONAL, reframed 2026-04-28 after F37 FL)

**F21c (2026-04-24):** STL-with-the-graph-prior (`next_getnext_hard` single-task) outperformed MTL-B3 on region by 12–14 pp Acc@10 at AL + AZ. Full analysis: `research/F21C_FINDINGS.md`.

**First-pass resolution (F48-H3-alt, 2026-04-26):** the gap was NOT structural to MTL on AL+AZ — it was a single confound in the LR schedule. Per-head LR (cat=1e-3, reg=3e-3, shared=1e-3) closed/exceeded the STL ceiling on AL (+6.25 pp, paired Wilcoxon p=0.0312) and closed 75% of the gap on AZ. Full analysis: `research/F48_H3_PER_HEAD_LR_FINDINGS.md`.

**FL closure (F37, 2026-04-28) — scale-conditional reframing:** F37 STL `next_getnext_hard` FL 5f Acc@10 = **82.44 ± 0.38**, far above MTL-H3-alt FL 73.65 (per-task best, top10_acc_indist) / 71.96 (joint best). Paired Wilcoxon **−8.78 pp p=0.0312, 5/5 folds negative**. The matched-head STL ceiling exceeds MTL-H3-alt at FL. The H3-alt recipe **does not lift reg above STL at FL scale** — at 4,702 regions, the cross-attention architecture pays an architectural cost (F49 Layer 3: architectural Δ vs STL = **−16.16 pp**) that the per-head LR cannot recover.

**Per-state pattern (architectural Δ from F49 + MTL vs STL gap):**

| State | Regions | Architectural Δ (frozen − STL) | MTL H3-alt vs STL F21c | Verdict |
|-------|--------:|------------------------------:|----------------------:|---------|
| AL | 1,109 | **+6.48 pp** (architecture wins) | **+6.25 pp** | MTL exceeds STL ✓ |
| AZ | 1,547 | −6.02 pp | −3.29 pp (75% closed) | classical pattern |
| FL | 4,702 | **−16.16 pp** (heavy cost) | **−8.78 pp p=0.0312** | STL ceiling above MTL ✗ |

**Implication for paper framing.** CH18/CH21 are reframed as scale-conditional: AL is the architecture-dominant state where MTL H3-alt > STL on reg. FL's headline reg ceiling is STL `next_getnext_hard` (the matched-head single-task baseline). The H3-alt recipe is still the recommended joint-deployment config — and at FL the **substrate-side cat advantage** (CH16 + CH18-substrate) carries the contribution; the architecture-side reg lift is AL-only.

Full analysis: `research/F37_FL_RESULTS.md`. Concern tracker: `CONCERNS.md §C15` (re-opened 2026-04-28 with FL caveat).

The B3-vs-STL framing below is preserved for the predecessor recipe (still relevant when the per-head LR mode is not used):

- **Joint-task single-model deployment:** B3 gives both `next_category` and `next_region` predictions in one forward pass. Two STL models (one GETNext-hard for region + one matched STL cat head) would beat B3 on region by 12 pp but require running two separate models.
- **Cat F1 lift over STL:** MTL-B3 does lift STL cat F1 (AZ +3.73 pp, p=0.0312). This contribution survives F21c.
- **FL-scale PCGrad pathology:** F2's late-stage-handover finding is independent and paper-worthy.

## Validation status across states (post-F27)

| State | Protocol | cat F1 | reg Acc@10 | Status |
|---|---|---:|---:|---|
| AL | 5f × 50ep (pre-F27) | 0.3928 ± 0.0080 | 0.5633 ± 0.0816 | superseded by F31 |
| AL | **5f × 50ep (post-F27, next_gru)** | **0.4271 ± 0.0137** | **0.5960 ± 0.0409** | ✅ **F31 validated, +3.43 cat / +3.27 reg Acc@10** |
| AZ | 5f × 50ep (pre-F27) | 0.4362 ± 0.0074 | 0.5276 ± 0.0392 | superseded |
| AZ | **5f × 50ep (post-F27, next_gru)** ⭐ | **0.4581 ± 0.0130** | **0.5382 ± 0.0311** | ✅ **committed; Wilcoxon p=0.0312 on 3 metrics** |
| FL | 1f × 50ep (pre-F27, F2 + F17 fold 1 ×2) | 0.6623 / 0.6706 | 0.6582 / 0.6655 | prior n=1 |
| FL | **1f × 50ep (post-F27, next_gru)** | 0.6572 | 0.6526 | ⚠️ **F32 — cat F1 −0.93 vs pre-F27 mean**; within n=1 noise but direction flipped |
| FL | **5f × 50ep H3-alt MTL** (F48-H3-alt, 2026-04-26) | **0.6792 ± 0.0072** | 0.7196 ± 0.0068 (top10_acc_indist) | ✅ MTL champion FL run |
| FL | **5f × 50ep STL `next_gru` cat** (F37 P1, 2026-04-28) | **0.6698 ± 0.0061** | — | ✅ matched-head cat ceiling. MTL > STL by **+0.94 pp** at FL ✓ |
| FL | **5f × 50ep STL `next_getnext_hard` reg** (F37 P2, 2026-04-28) | — | **0.8244 ± 0.0038** | ⚠️ matched-head reg ceiling **exceeds MTL-H3-alt by −8.78 pp p=0.0312, 5/5 folds**. CH18 reframes scale-conditional. |

## ⚠ F27 scale-dependence flag (2026-04-24)

The cat-head swap `NextHeadMTL → next_gru` **helps AL (+3.43 pp cat F1) and AZ (+2.37 pp, p=0.0312) but slightly hurts FL at n=1 (−0.93 pp cat F1)**. Three paths documented in `research/F27_CATHEAD_FINDINGS.md §Decision`:

- **A:** Commit `next_gru` universally (accept small FL cost for simpler narrative).
- **B:** Scale-dependent — `next_gru` for AL/AZ, `next_mtl` for FL/CA/TX.
- **C:** Run FL 5f B3+gru (~6 h MPS) to resolve definitively.

The NORTH_STAR config above currently reflects **A** pending user decision. If the user picks **B**, the cat head for FL/CA/TX reverts to `next_mtl` (MTLnet's historical default).

## History

### Post-F2 update (2026-04-23 evening)

F2 (`research/B5_FL_TASKWEIGHT.md`) completed all four phases. The Phase B3 configuration **`mtlnet_crossattn + static_weight(category_weight=0.75) + next_getnext_hard d=256, 8h`** at n=1 fold on FL delivers:

| Metric | Soft B-M13 (prior north-star) | B3 | Δ |
|---|---:|---:|---:|
| cat F1 | 0.6601 | **0.6623** | **+0.22 pp** |
| reg Acc@10_indist | 0.6062 | **0.6582** | **+5.20 pp** |
| reg Acc@5_indist | 0.3601 | **0.3988** | **+3.87 pp** |
| reg MRR_indist | 0.2555 | **0.2794** | **+2.39 pp** |

B3 Pareto-dominates soft at n=1 on every joint-score metric. The mechanism: cat-heavy weighting triggers a **late-stage handover** — cat head converges fast in early epochs, then the shared backbone becomes available to the region head for the remaining epochs (cat training extends to epoch 42 vs ≤10 for soft/pcgrad/equal-weight).

**Interim policy:** soft remains the reported north-star until F2's follow-up validation lands (§Re-evaluation triggers). If both checks hold, B3 becomes the new universal north-star.

### Follow-up required before committing B3

| Check | Cost | Pass criterion |
|---|:-:|---|
| B3 on FL, 5-fold | ~5–6 h MPS | σ on cat F1 does not pull B3 below soft (soft cat = 66.01 n=1; B3 cat = 66.23 n=1 — σ could be decisive) |
| B3 on AL, 5-fold | ~1 h MPS | B3 does not break the AL cat head (current AL-hard+pcgrad is 38.50 cat F1) |
| B3 on AZ, 5-fold | ~1–1.5 h MPS | B3 preserves the AZ region lift from B-M9d (53.25 Acc@10 with pcgrad) |

Note: static_weight is a simpler optimizer than PCGrad, and AL/AZ already work under the harder PCGrad; low-risk that B3 breaks them.

## Interim choice (still current, pre-B3-validation)

**`mtlnet_crossattn + pcgrad + next_getnext (soft probe) d=256, 8 heads`** (B-M6b on AL, B-M9b on AZ, B-M13 on FL). All paper tables currently reference this config.

If F2 follow-up passes and B3 replaces soft, the migration is a single-string swap in every paper-facing table plus a 5-fold re-run on each state — same wall-clock cost as any scientific revision.

## Why (short version)

| State | soft joint Acc@10 / cat F1 | hard joint Acc@10 / cat F1 | Winner |
|:-:|:-:|:-:|:-:|
| AL 5f | 56.49 / 38.56 | 57.96 / 38.50 | tied within σ |
| AZ 5f | 46.66 / 42.82 | 53.25 / 42.22 | hard on reg (+6.59 outside σ), cat σ-tied |
| FL 1f | **60.62 / 66.01** | 58.88 / **55.43** | **soft** — hard's cat head fails to train |

- **FL is the headline state** (per `CONCERNS.md §C01` — the paper's primary table is FL + CA + TX).
- Hard has a **diagnosed training failure at FL scale** (see `research/B5_FL_SCALING.md` + the 2026-04-23 JSON comparison in `review/2026-04-23_critical_review.md`): cat head's best-val F1 over 50 epochs is 55.43 vs soft's 66.01 under the identical fold split. Not noise — gradient imbalance.
- Soft scales uniformly across AL / AZ / FL. Cat F1 is within σ of the cross-attn + GRU champion at every state.

## What this choice costs us

- The **AZ +1.01 pp MTL-over-STL-STAN** result (53.25 vs 52.24) that currently sits in hard. Soft on AZ lands at 46.66 Acc@10, which is +3.70 pp above Markov-1 but −5.58 below STL STAN. Under soft, AZ reg is framed as "MTL beats Markov" rather than "MTL beats STL".
- The "faithful Yang 2022 SIGIR" framing. Soft is an adaptation (learned probe) rather than the original hard-index formulation.

## What hard is still used for

Hard remains a **reported ablation row**, not retired. In the paper:

> We propose MTL-GETNext-soft as the joint-task model. We report a faithful hard-index variant as an ablation: at region-cardinality ≤ 1.5 K it matches (AL, within σ) or dominates (AZ, +6.59 pp Acc@10, +3.08 pp MRR) the soft adaptation. At 4.7 K-region scale (FL), hard over-dominates the MTL gradient through PCGrad and the category head fails to train (best-val cat F1 0.554 across 50 epochs vs soft's 0.660). We analyse the mechanism in §X and recommend soft as the scale-robust default.

## Re-evaluation triggers

This choice is revisited if **any** of the following lands:

1. **F2 (FL task-weight sweep).** If `task_b_weight < 1` restores FL-hard cat F1 to ≥ 60 while keeping reg Acc@5 lift, hard becomes scale-uniform and is re-promoted as north.
2. **F12 (FL 5-fold hard) with σ showing cat F1 within σ of soft.** Would argue the 10 pp cat gap was n=1 amplification, not training pathology — low likelihood given the `diagnostic_task_best` analysis but empirically checkable.
3. **A new MTL variant** (e.g., per-task weight clipping, prior-magnitude normalisation) that rescues FL-hard without a task-weight hack. Post-paper research direction.

Until one of those lands: **soft is the headline MTL config**.

## Pointers

- Joint-execution comparison: `OBJECTIVES_STATUS_TABLE.md §2`
- Cross-state deltas: `research/B5_MACRO_ANALYSIS.md`
- FL failure-mode diagnosis: `research/B5_FL_SCALING.md` + `review/2026-04-23_critical_review.md §FL-hard training pathology`
- Open follow-ups that can change this: `FOLLOWUPS_TRACKER.md` F2, F12
