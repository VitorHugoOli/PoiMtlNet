# North-Star MTL Configuration

**Status (2026-04-29 17:00 UTC):** Champion upgraded again. **P4 alternating-SGD + OneCycleLR (max_lr=3e-3, pct_start=0.4) + delayed-min selector (`min_epoch=10`)** is the new committed champion at FL. P4-alone becomes a predecessor (still beats H3-alt, but P4+OneCycle beats P4-alone by +2.04 pp at ≥ep10, paired Wilcoxon p=0.0312, 5/5 folds positive). The two interventions compose additively because they act on orthogonal mechanisms: P4 gives reg its own optimizer step on its own batch (preventing post-ep-5 cat-dominance collapse), OneCycle then provides the late-epoch peak-LR window that aligns with α growth (peak at ep ~20). See §"Champion — F50 P4 + OneCycle" below and `research/F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md §6.3-§6.4`.

**Status (2026-04-27):** Two complementary tracks now confirm the MTL story from different angles. **The previous recipe (F48-H3-alt per-head LR) is the committed champion**; substrate validation and architecture attribution both back it.

**Architecture-side (F48-H3-alt + F49, 2026-04-26 → 04-27):** Per-head LR recipe validated 5-fold on AL/AZ/FL — cat preserved within ~2 pp of B3, reg Acc@10 lifts by 6.7-15 pp over B3. AL **exceeds** STL F21c ceiling by +6.25 pp; AZ closes 75%; FL is most stable (σ=0.68). Three orthogonal negative controls (F40, F48-H1, F48-H2) bracket H3-alt as the unique design. **F49 attribution (2026-04-27):** the H3-alt reg lift on AL is *purely architectural* (+6.48 pp from architecture alone, F49c 5f × 50ep); cat-supervision transfer is null on all 3 states (≤|0.75| pp), refuting the legacy "+14.2 pp transfer" claim by ≥9σ on FL n=5. CH18 Tier A; CH19 Tier A. See `research/F48_H3_PER_HEAD_LR_FINDINGS.md` + `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`.

**Substrate-side (Phase 1 substrate validation, 2026-04-27):** Five-leg study on AL+AZ confirms the substrate side of the MTL claim:

1. **CH16 confirmed under matched-head, head-invariant** at AL+AZ (8/8 head-state probes positive, p=0.0312 each; ranges +11.58 to +15.50 pp).
2. **CH15 reframed** — under the matched MTL reg head (`next_getnext_hard`), C2HGI ≥ HGI (was "HGI > C2HGI" under STAN). The previous CH15 was head-coupled.
3. **CH18 — MTL B3 is substrate-specific.** Substituting HGI breaks the joint signal (cat −17 pp, reg −30 pp Acc@10_indist at both states; MTL+HGI is *worse than STL+HGI* on reg by ~37 pp).

These findings **do not** change the committed config — they explain *why* it works. See `research/SUBSTRATE_COMPARISON_FINDINGS.md` for the full Phase 1 verdict + `PHASE2_TRACKER.md` for FL/CA/TX replication queue.

**Status (2026-04-24):** Cat head refined via F27 from `NextHeadMTL` (Transformer) → `next_gru` (GRU). Paper-reshaping F21c finding noted in §§Caveats. See §Committed config below.

## Champion — F50 P4 + OneCycle + delayed-min (2026-04-29 17:00 UTC)

```
architecture         : mtlnet_crossattn
mtl_loss             : static_weight(category_weight = 0.75)
task_a head (cat)    : next_gru
task_b head (reg)    : next_getnext_hard                # STAN + α · log_T[last_region_idx]
task_a input         : check-in embeddings (9-step window)
task_b input         : region embeddings (9-step window)
hparams              : d_model=256, 8 heads, batch=2048, 50 epochs, seed 42
LR scheduler         : OneCycleLR(max_lr=3e-3, pct_start=0.4)   # ← new (peak at ep 20/50)
LR per param group   : cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3
optimizer step       : ALTERNATING per-batch (P4) — cat batch then reg batch, separate optimizer.step()
selector             : delayed-min top10_acc_indist with min_epoch=10
```

**Single-line additive recipe vs H3-alt:**
```bash
--alternating-optimizer-step \
--scheduler onecycle --max-lr 3e-3 --pct-start 0.4 \
--min-best-epoch 10
```

**Headline numbers (FL 5f × 50ep, seed 42):**

| selector | H3-alt | P4 alone | **P4 + OneCycle** | Δ vs P4-alone | folds | paired Wilcoxon |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 77.16 | 78.55 | 77.52 | −1.03 | 1/4 | n.s. |
| ≥ep5 | 74.72 | 78.55 | 77.52 | −1.03 | — | n.s. |
| **≥ep10** | 71.44 | 75.48 | **77.52** | **+2.04** | **5/5** | **p=0.0312** ✅ |

Per-fold @ ≥ep10:
- P4 alone: `[74.89, 74.57, 76.02, 76.36, 75.59]`
- P4+OneCycle: `[77.33, 76.71, 77.62, 77.92, 78.04]`
- Per-fold Δ: `[+2.44, +2.14, +1.60, +1.56, +2.45]` (5/5 positive, σ_Δ=0.44)

Best epochs across folds: **{20, 19, 20, 19, 19}** — P4+OneCycle hits its peak at ep 19-20, exactly the OneCycle peak-LR window.

**Mechanism (compositional):** P4 alternating-SGD prevents the post-ep-5 reg degradation by giving reg its own optimizer step on its own batch. OneCycle (max_lr=3e-3, pct_start=0.4) places peak LR at ep 20 — exactly where α growth needs the LR magnitude. The two interventions act on orthogonal mechanisms: P4 = optimizer separation, OneCycle = LR scheduling. They compose additively for a +5.87 pp lift over H3-alt (71.44 → 77.52) and +2.04 pp over P4-alone (75.48 → 77.52).

**Tier-A negative controls confirm OneCycle needs P4** — OneCycle without P4 underperforms H3-alt by 9 pp (A1, A6) regardless of α_init or cat_weight. P4 is the necessary substrate.

See `research/F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md §6.3-§6.4` and `research/F50_T3_HYPERPARAM_BRAINSTORM.md` for full Tier-A run log.

**Closed: P4 + Cosine variant** (run `_1653`, 2026-04-29 17:11) — beats P4-alone by +0.58 pp at ≥ep10 (5/5 positive, mean 76.07 ± 0.62) but **loses to P4+OneCycle by −1.45 pp uniformly** (0/5 positive). This pins the mechanism: OneCycle's warmup ramp is what places peak LR at ep ~20, where α growth needs it; cosine's decay-from-peak gives the early boost away too soon. The warmup is **mechanistically necessary**, not an arbitrary scheduler shape choice.

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
