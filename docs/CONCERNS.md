# Check2HGI Study — Open Concerns Log

Living document for acknowledged risks, tensions, and framing questions we are *not* resolving now but want to revisit before the paper freezes. Each entry records the concern, the current pragmatic resolution, and what would re-open it.

**Convention:** `status ∈ {open, deferred, resolved, monitored}`. Add `— resolved YYYY-MM-DD (branch/commit)` when an entry closes.

---

## C01 — Two-state development is under-powered for a final paper

**Concern raised:** 2026-04-16. Our P1 baselines were measured on AL (small, sparse) and FL (large, dense). A reviewer can fairly push back on `n=2` states.

**Resolution (adopted):** AL is treated as a **development / ablation** state — cheap to iterate on, useful for architecture and modality choices. The **headline paper table** will cover the three large US states Check2HGI is trained on: **Florida, California, Texas**.

**Implication for the study plan:**
- AL runs remain in P1/P2/P3/P4 as development data, because 5f×50ep on AL ~ 30 min vs FL ~ 6 h. Speed of iteration matters.
- The P3 headline (bidirectional MTL comparison) and P4 (per-task input modality) must be re-run on CA and TX before paper submission.
- P2 (arch × optim grid) runs on AL only — it is ablation, not headline.

**Status:** `monitored`. Re-opens if any CA/TX data quality issue surfaces (Check2HGI embeddings must be available and validated on both).

---

## C02 — FL is near-saturated by Markov; bidirectional MTL lift may be invisible

**Concern raised:** 2026-04-16. Markov-1-region on FL is already 65.05% Acc@10; single-task GRU is 68.33%. The 3.3 pp margin is narrow enough that MTL may match but not exceed single-task on the region side. That would falsify the strict bidirectional thesis on FL.

**Resolution (adopted):** The headline story is *across* the three final states (FL + CA + TX), not FL alone. FL alone is allowed to tie or narrow-lift; CA and TX are where the bidirectional margin is expected to show (both are dense-enough to have real MTL signal but less than FL's near-saturation). If **all three** show null MTL lift on the region side, the paper pivots to the backup narrative (see C05).

**Status:** `monitored`. Re-evaluate after CA + TX P3 runs complete.

---

## C03 — Compound joint-score metric for bidirectional MTL

**Concern raised:** 2026-04-16. The two tasks use different primary metrics (macro-F1 vs Acc@10/MRR). "Both heads improved" was adjudicated anecdotally. We need a single scalar per run for paired tests and a leaderboard.

**Resolution (adopted):** Δm metric from Maninis et al., CVPR 2019 (formalised as the MTL-survey standard in Vandenhende et al., TPAMI 2021; used by NashMTL/PCGrad/LDC-MTL/DB-MTL for reporting). Formula and decision rule defined in `CLAIMS_AND_HYPOTHESES.md §CH02`. Short version: Δm = ½(r_A + r_B) with relative per-task deltas; bidirectional Pareto gate on (r_A > 0, r_B > 0) carries the thesis; Wilcoxon signed-rank on fold-level Δm carries statistical power.

**Status:** `resolved — 2026-04-16`.

---

## C04 — `next_mtl` (Transformer head) replaced with GRU for the region slot

**Concern raised:** 2026-04-16. The legacy MTLnet framework used a Transformer head (`next_mtl`: 4 layers × 4 heads × dim=64 × dropout=0.35) for the single sequential ("next") slot. When plugged into the *region-head* position, it collapsed to 7.4% Acc@10 (AL 5f×50ep), worse than a random baseline for a 1109-class problem.

**Framing vulnerability:** a reviewer may read the paper and ask "if MTL on Check2HGI is your contribution, why did you swap out the MTL framework's own sequence head?"

**Resolution (adopted):** Clarify in Methods that the MTLnet *framework* is a **backbone + plug-in heads** decomposition. The framework owns:
- task-specific encoders (2-layer MLPs),
- FiLM / CGC / MMoE / DSelectK / PLE shared backbone,
- task embeddings + `shared_parameters()` / `task_specific_parameters()` contracts for MTL optimizers (NashMTL, PCGrad, GradNorm).

The specific *head module* that consumes the shared-backbone output is a swappable component from the head registry (`next_gru`, `next_lstm`, `next_tcn_residual`, `next_temporal_cnn`, `next_mtl`). The default head was tuned for a 7-class next-category task; scaling to a 1109-class region target with the same hyperparameters is a head-task capacity mismatch, *not* an MTL-framework property.

**Paper framing:** "Following P1 head-ablation, we use `next_gru` for the region head throughout the MTL experiments. This is a like-for-like substitution within the MTLnet framework's head registry; all shared-backbone and gradient-manipulation components are unchanged. See Appendix §X."

**Status:** `resolved` — framing committed to Methods outline. Re-opens only if reviewers still push back.

---

## C05 — P3 null-result contingency plan

**Concern raised:** 2026-04-16. If MTL does not beat single-task on one or both heads on the final CA/TX table, CH01 (the headline claim) fails.

**Resolution (adopted backup narrative, documented here for future reference):**

> **"Input modality, not task sharing, is the locus of improvement in hierarchical POI MTL."**

P4's per-task input modality ablation (per-task vs concat vs shared-checkin vs shared-region) is the natural co-headline. If MTL lift is marginal but the per-task modality consistently beats shared-modality by a clear margin, that becomes the paper's main empirical contribution, with MTL framing demoted from "lift claim" to "the framework in which the modality choice is made."

**What to do when the decision is needed:** after P3 on CA/TX completes, compare:
- `|MTL_joint_score − SingleTask_joint_score|` averaged across CA/TX, and
- `|per_task_joint_score − best_shared_modality_joint_score|` from P4 on CA/TX.

If the second delta is larger, pivot the paper to the modality-first framing. CLAIMS_AND_HYPOTHESES would then promote CH03 to Tier A and demote CH01/CH02 to reported comparisons.

**Status:** `deferred` — decision triggered only if P3 falls short.

---

## C06 — Region head: GRU is the champion, TCN-residual is used for compute-heavy grids

**Concern raised:** 2026-04-16. In P1 on AL 5f×50ep, `next_gru` (56.94 ± 4.01) and `next_tcn_residual` (56.11 ± 4.02) are statistically indistinguishable on Acc@10, but TCN is ~20× faster per fold.

**Resolution (adopted):**
- **Champion for headline reporting (P3, P4 on CA/TX):** `next_gru`. It is the literature-aligned choice (HMT-GRN uses GRU) and its variance envelope contains TCN's mean.
- **Compute-efficient substitute for grids (P2 arch × optim, P4 development on AL):** `next_tcn_residual`. Enables 20× more configurations per wall-clock hour with statistically equivalent region-task performance.
- **Sanity check:** at the end of P2, re-run the top-3 arch × optim configurations with `next_gru` on AL to confirm TCN vs GRU ordering is preserved.

**Paper framing:** report both in the ablation table; state GRU as the champion used in the headline.

**Status:** `resolved`.

---

## C07 — Embedding comparison outcome: Check2HGI and HGI tie at the region level

**Concern raised:** 2026-04-16. We committed to Check2HGI without comparing against POI-level HGI on the same task. A reviewer will want to see this was a considered choice.

**Resolution (adopted — result in):** P1.5 ran Check2HGI vs HGI region embeddings on AL next_region single-task, 5f × 50ep, TCN-residual head, seed 42. Result: **statistically tied** (Check2HGI 56.11 ± 4.02 vs HGI 57.02 ± 2.92 Acc@10).

**Implication (pivot):** the paper's substrate-advantage claim cannot be "Check2HGI beats HGI" at the region level. It becomes "**Check2HGI uniquely enables per-task MTL with distinct modalities (check-in-level and region-level), which HGI architecturally cannot provide because HGI has no per-visit granularity.**" The advantage must then be shown at the **task-A (next_category) input** side (check-in-level contextual vectors) and at the **per-task modality design choice** side (P4 / CH03). Region-level embeddings are merely the "floor input" shared by both engines.

This tightens the paper's story: the contribution is framed around what Check2HGI *uniquely enables*, not what it *beats HGI at*. Closes C07.

**Status:** `resolved — 2026-04-16`. Re-opens only if HGI also shows some form of check-in-level contextual variation (it doesn't, by its architecture).

---

## C08 — CH04 (2× Markov-region gate) retired; reframing as pp-delta

**Concern raised:** 2026-04-16. The original CH04 required the best neural head to beat Markov-1-region by ≥ 2×. AL achieves 1.21× (47.01 → 56.94), FL achieves 1.05× (65.05 → 68.33). Retiring a gate we set looks like goalpost-moving.

**Resolution (adopted):** Reframe from multiplicative ratio to **absolute improvement in percentage points over the Markov floor**. On AL: +9.9 pp. On FL: +3.3 pp. These are defensible absolute improvements, particularly given FL's high Markov baseline reflects a data regime where near-term transitions dominate. CH04 becomes a *reported* comparison, not a gate.

**Paper framing:** "We report Markov-1-region as the simple floor (§Baselines). Single-task GRU lifts Acc@10 by +9.9 pp on AL and +3.3 pp on FL. The region task is learnable beyond Markov order-1 in both regimes."

**Status:** `resolved`.

---

## C09 — SSD reliability risk during long runs

**Concern raised:** 2026-04-16. External Thunderbolt SSD exhibited transient SIGBUS crashes during long training runs (4 crashes in one day during the FL P1 confirm). Root cause not fully diagnosed (physical reseating restored stability).

**Resolution (adopted):** Long P2/P3/P4 runs on FL/CA/TX must redirect `OUTPUT_DIR` to the boot volume (`/tmp/check2hgi_data`) and keep data loading from there. Code + venv remain on SSD, but the training loop's hot path (data reads, checkpoint writes) is insulated. Launch commands documented in `HANDOFF.md` examples.

**Status:** `monitored`. If CA/TX data doesn't fit on boot volume (18 GiB free), need alternative plan — possibly cloud or Colab for headline runs.

---

## C10 — External published comparisons: POI-RGNN + HGI-next-category article (CH17)

**Concern raised:** 2026-04-16 evening (user clarification). The paper's CH17 claim is that Check2HGI surpasses not only HGI in our controlled pipeline (CH16) but also published external baselines: POI-RGNN (known numbers: FL 31.8%, CA 34.5% macro-F1 for next-category on Gowalla state-level) and a prior HGI-based next-category article (specific reference pending user confirmation).

**What's needed:**
1. **POI-RGNN:** already have the reference (prior mention in HANDOFF). Line up our FL/CA/TX single-task next-category numbers (from P1.5b extension or P3 baseline) vs POI-RGNN's published numbers. No re-run of POI-RGNN is required; citation-and-compare.
2. **HGI-based next-category article:** user mentioned "courb article" that applied HGI to next-category — specific reference TBD. Once located, note whether the pipeline matches (data preprocessing, folds, head) so the comparison is fair. If the article's pipeline differs, frame as "same-data external comparison" rather than controlled.

**Resolution plan:**
- CH16 (P1.5b) provides the controlled comparison on our pipeline.
- CH17 provides the external anchors.
- Paper Results table gets two columns: "Prior work" (POI-RGNN + HGI-article) and "Ours" (Check2HGI single-task + MTL).

**Status:** `open` — needs the exact HGI-next-category reference from the user. Paper framing otherwise ready.

---

## C11 — User-leakage in STL next-task folds (fold-protocol mismatch) — RESOLVED

**Concern raised:** 2026-04-17 (during P2 critical review). `FoldCreator._create_single_task_folds()` used plain `StratifiedKFold` for the NEXT task without user-grouping while MTL used `StratifiedGroupKFold(groups=userid)`. Consequence: STL numbers inflated by user-taste memorisation across leaky val folds.

**Fix:** `src/data/folds.py::_create_single_task_folds` now uses `StratifiedGroupKFold(groups=userids)` for NEXT. See `issues/FOLD_LEAKAGE_AUDIT.md` for full write-up.

**Empirical resolution (2026-04-17):**
- Check2HGI STL cat F1: leaky 39.16 → fair 38.58 (−0.57 pp drop; robust)
- HGI STL cat F1: leaky 23.48 → fair 20.29 (−3.20 pp drop; leaky-dependent)
- CH16 delta grew: +15.67 pp → **+18.30 pp** (primary substrate claim stronger with fair folds).

**Budget test follow-up (2026-04-17, bg `bc1rlz40f`):** MTL dselectk+pcgrad at 5f × 50ep on AL:
- MTL cat F1 = 36.67 ± 2.14 vs STL fair = 38.58 ± 1.23 → Δ = −1.91 pp, σ-overlap **YES**. **Statistically tied on category** at matched compute + fair folds.
- MTL region Acc@10 = 47.62 ± 5.62 (with the Transformer head capping; standalone transformer was 7.4%, so MTL is lifting it by +40 pp via category's contextual signal through the shared backbone).

**Interpretation:** the original "MTL underperforms STL by 3 pp on category" was 2/3 fold-leakage + 1/3 compute/noise. After the fix, MTL and STL are statistically tied on category at matched compute; the remaining nominal gap is within σ.

**Status:** `resolved — 2026-04-17`.

---

## C12 — Hyperparameter mismatch across STL vs MTL baselines

**Concern raised:** 2026-04-18 (user feedback). Our "fair" STL vs MTL comparison uses different hyperparameters across baselines:

| Baseline | batch_size | epochs | max_lr | source |
|----------|-----------:|-------:|-------:|--------|
| STL cat (Check2HGI, via `default_next`) | 1024 | 50 | **0.01** | `scripts/train.py --task next` default |
| STL region GRU (via `p1_region_head_ablation.py`) | 2048 | 50 | **0.003** | per-head max_lr dict |
| MTL (via `default_mtl`) | 2048 | 50 | **0.001** | `configs.experiment.default_mtl` |

**Impact:** MTL's max_lr is 3× lower than the STL region GRU baseline and 10× lower than the STL cat baseline. A single OneCycleLR max_lr value is shared across all MTL parameters; the GRU region head cannot get its standalone-optimal LR of 0.003 while simultaneously allowing the MTL category path to train stably.

**Earlier-study precedent (from user):** Simple-baseline Markov initially underperformed on STL region head before hyperparameter tuning fixed it; the LR-calibration lesson was learned once but not propagated to MTL configs.

**Consequence for the "capacity-ceiling" claim:** the 5.4 pp "architectural overhead" measured by the λ=0.0 isolation was at MTL's default max_lr=0.001. At max_lr=0.003 (matching STL region GRU), this overhead may shrink. The "capacity-ceiling" claim needs re-verification at matched hyperparameters before it is paper-ready.

**Fix (in progress):** Added `--max-lr` CLI flag to `scripts/train.py`. Launched sweep on AL 5f × 50ep dselectk+pcgrad at max_lr ∈ {0.003, 0.01} (step 7, bg `bauhto2o2`). Results will decide whether any / all of the ablation findings require re-measurement.

**Second confound identified (2026-04-27, F49).** Beyond the LR mismatch, the loss-side λ=0 protocol itself does not cleanly isolate architectural overhead under cross-attention. Tracing gradient flow in `MTLnetCrossAttn._CrossAttnBlock.forward`, when `category_weight=0.0` the reg loss still propagates back to `category_encoder` parameters through `cross_ba`'s K/V projections (reg's stream queries cat-encoder outputs as K/V). The cat **encoder** is therefore implicitly trained as a reg-helper even though the cat **head** is frozen-at-init. The original "architectural overhead" number is the joint contribution of (i) pure architectural cost and (ii) cross-attention-mediated co-adaptation of the cat encoder to serve reg — two effects that can have opposite signs and cannot be disentangled from a single loss-side λ=0 measurement. Resolution path: F49 introduces a `--freeze-cat-stream` flag for a 3-way decomposition (STL / frozen-cat λ=0 / loss-side λ=0 / full MTL) under H3-alt, replacing the original 2-way decomposition with one that distinguishes architecture, co-adaptation, and transfer. Full analysis: `research/F49_LAMBDA0_DECOMPOSITION_GAP.md`; tracker: `FOLLOWUPS_TRACKER.md §F49`.

**Partial resolution (2026-04-27, F49 AL+AZ landed).** The H3-alt-regime 3-way decomposition completed on AL and AZ (FL in flight, bg `baupbogv6`):
- **AL:** the original "5.4 pp architectural overhead" claim is REVERSED under H3-alt. Frozen-cat λ=0 = 74.85 ± 2.38 vs STL F21c 68.37 ± 2.66 → architecture is **+6.48 pp BENEFIT**, not overhead. Co-adaptation (+0.09) and transfer (−0.32) are both ≈ 0; H3-alt's reg lift on AL is purely architectural.
- **AZ:** frozen-cat λ=0 = 60.72 ± 1.64 vs STL 66.74 ± 2.11 → architecture **does cost reg by 6.02 pp** at this state, but co-adaptation (+1.98) and transfer (+0.75) provide modest rescue. Full MTL still trails STL by 3.29 pp (F21c gap persists, just like the legacy framing predicted *for AZ*).
- **The original "uniform architectural overhead, scale-dependent transfer" framing is refuted by the AL data** independent of the LR confound: AL has near-zero overhead AND near-zero transfer under any clean measurement.
- The gradient-flow second confound (loss-side λ=0 silently trains cat encoder via cross-attn K/V) was empirically confirmed: 4 regression tests pass, including the load-bearing assertion that `category_encoder.weight.grad.abs().sum() > 0` after backward on `static_weight(category_weight=0.0)`.

FL outcome (in flight) decides whether the headline state matches AL's pattern (architecture wins outright) or AZ's pattern (classical MTL-with-rescue) or a third regime; until FL lands, the paper's "decomposition" claim cannot be committed. C12 closure conditional on FL completion + analysis writeup. Full results: `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`.

**Final resolution (2026-04-27 14:50, F49 FL landed).** FL 2-fold re-launch completed cleanly on /tmp-resident data (post-SSD-blip mitigation per C09): FL frozen-cat λ=0 = 73.82 ± 0.94, FL loss-side λ=0 = 72.48 ± 0.46, FL Full MTL H3-alt = 71.96 ± 0.68. Three-state mechanism patterns are now characterized: AL (architecture-dominant, transfer null), AZ (classical-MTL with rescue), FL (architecture + cat-side-co-adaptation-HURTS — encoder-frozen MTL beats Full MTL by +1.86 pp Acc@10). The original "uniform 5.4/24.93 pp architectural overhead + scaling transfer" framing is empirically refuted on all three states:
- The "uniform overhead" claim collapses because AL's architectural term is +6.48 pp (benefit), AZ's is −6.02 pp, and FL's encoder-frozen variant beats Full MTL outright. Sign and magnitude both vary per state.
- The "+14.2 pp scaling transfer at FL" claim is dead — measured transfer is ≤ |0.75| pp on all three states (AL −0.32, AZ +0.75, FL −0.52).

The methodological gap (gradient flow under cross-attention silently violating loss-side λ=0's "isolation" claim) is also empirically resolved: 4 regression tests pass, including the load-bearing assertion that cat encoder receives non-zero gradient through `cross_ba`'s K/V under `static_weight(category_weight=0.0)`. The encoder-frozen variant is the only clean architectural isolation in this design space; F49 demonstrates the difference matters numerically (FL: −1.34 pp co-adaptation contribution that loss-side λ=0 would have hidden).

Layer 1 + Layer 2 paper claims (cat-supervision transfer is small on all 3 states; loss-side ablation is unsound under cross-attention MTL) are committable. Layer 3 (absolute architectural Δ on FL vs STL F21c) remains gated on F37 (4050-assigned). Full analysis: `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` §10-13.

**Advisor caveat (2026-04-27, post-FL-write-up):** Layer-1 and Layer-2 claims are paper-grade, but several FL claims need walking back:
- **FL co-adapt = −1.34 pp** is ~1.27σ from zero at n=2 (σ_diff ≈ 1.05) — within noise, not significant.
- **FL transfer = −0.52 pp** is ~0.63σ — clearly noise.
- **"Encoder-frozen MTL beats Full MTL on FL by +1.86 pp"** is ~1.6σ — borderline. Don't headline.
- **The reproduction gate was not actually run against the published 52.27 protocol** — it ran at max_lr=1e-3, the published value was at max_lr=3e-3 (per HYBRID_DECISION_2026-04-20.md "fair LR"). **F49b corrected this 2026-04-27 14:52:** AL `static_weight λ=0 + max_lr=3e-3 + OneCycleLR + next_gru` 5f×50ep gives `top10_acc_indist = 53.18 ± 4.56` vs legacy 52.27 ± 5.03 → Δ = +0.91 pp at ~0.13σ, **σ-tight match**. Gate now passes cleanly; F49 infra reproduces the legacy protocol number; the H3-alt-regime AL/AZ/FL numbers are validated.

**Final resolution (2026-04-27 21:44, F49c FL n=5 landed).** F49c completed the FL 5-fold re-run cleanly on /tmp data (loss-side 191.79 min + frozen 171.22 min). Numbers:
- FL loss-side n=5: Acc@10 = **72.48 ± 1.40** (mean unchanged from n=2; σ realistic).
- FL frozen-cat n=5: Acc@10 = **64.22 ± 12.03** (mean **−9.60 pp from n=2**; σ ×13). Per-fold spread is 24 pp; the n=2 estimate was an unrepresentative fold-pair.
- (loss − frozen) **co-adapt = +8.27 pp at 0.68σ** — point estimate **flipped sign** from n=2 (was −1.34); direction now matches AL +0.09 and AZ +1.98.
- (Full − loss) **transfer = −0.52 pp at 0.34σ** confirmed null.
- (Full − frozen) = +7.74 pp at 0.64σ.

**Tree C (FL "H1b negative co-adaptation, encoder-frozen wins") is REFUTED at n=5.** The picture across all 3 states is now consistently **Tree A (architecture-dominant, cat-supervision transfer null, cat-encoder co-adaptation small/positive)**.

The frozen-cat reg path is **unstable on FL** specifically: Acc@10 σ = 12.03 (loss-side σ = 1.40); per-fold reg-best epochs {2, 14, 9, 4, 2} indicate 3 of 5 folds picked very early epochs, symptom of α-growth not engaging when cat features are random. This frozen-side instability at large class cardinality (4702) is a publishable methodological caveat — the encoder-frozen variant works cleanly on AL/AZ but breaks down on FL.

Empirical sub-claims (all paper-grade now):
- AL architectural +6.48 pp ± 2.4 (~2.7σ): **solid** ✓
- AZ architectural −6.02 pp ± 1.6 (~3.7σ): **solid** ✓
- AL/AZ/FL co-adapt + transfer all small (≤ |0.75| pp on transfer; co-adapt direction-positive when measurable): **solid** ✓ (Layer 1, refutes legacy +14.2 pp at ≥9σ on FL alone)
- Loss-side ablation is unsound under cross-attn MTL (Layer 2 methodological): **solid** ✓
- FL absolute architectural Δ vs STL: **closed by F37 2026-04-28** — frozen-cat λ=0 vs STL F21c FL (5f paired) Wilcoxon **−16.16 pp p=0.0312, 5/5 folds negative**; full MTL H3-alt vs STL F21c FL **−8.78 pp p=0.0312, 5/5 folds negative**.

**Final closure (2026-04-28, F37 FL landed).** F37 STL `next_getnext_hard` FL 5f × 50ep delivers **Acc@10 = 82.44 ± 0.38** (per-fold {0.8197, 0.8247, 0.8294, 0.8204, 0.8277}). Closes Layer 3 of the F49 attribution. The 3-state architectural-Δ pattern is now fully characterised: **{AL +6.48, AZ −6.02, FL −16.16} pp**. AL is the architecture-dominant outlier; AZ and FL pay an architectural cost (heavily on FL). Per-fold architectural deltas at FL are {−32.22, −31.05, −4.01, −5.42, −8.08} — folds 0+1 collapse with the FL frozen-cat instability we flagged earlier (per-fold reg-best epochs {2,14,9,4,2}); folds 2-4 are mild. The architectural cost magnitude has σ ~12 pp at FL but its sign is unambiguous (5/5 paired folds negative). Full analysis: `research/F37_FL_RESULTS.md` + `results/paired_tests/FL_layer3_after_f37.json`.

**Status:** `resolved 2026-04-28 — Layer 3 closed with negative architectural Δ at FL`. C12 fully closed; the original "5.4 pp uniform overhead + 14.2 pp scaling transfer" claim is dead in both legs (transfer null at all 3 states; architecture sign and magnitude vary per state — and on FL the cost is heavier than the original framing predicted).

---

## C13 — Alabama is a 10K-row dev state; may over-extrapolate to FL/CA/TX

**Concern raised:** 2026-04-18 (user feedback). All our ablation runs are on Alabama (10 K training rows × 1109 regions). The headline paper states are Florida (127 K × 4702), California, and Texas (both similarly large). A 10× data gap between dev and headline is risky:

1. **Shared-backbone MTL might fail on AL specifically because of under-parameterisation**, not because of a fundamental capacity-ceiling. Evidence: on FL (127K), MTL *does* lift category by +1.61 pp; on AL it does not.
2. **Per-task MTL architectures (cross-attention, MTLoRA) may scale differently**. The 50.72 vs 56.94 pp region gap on AL could compress or invert on FL.

**Available states:** AL (10K), Arizona (26K), Florida (127K). Georgia not in our Check2HGI data. California and Texas are headline.

**Proposed resolution:**
- **Keep AL for fast screening** — cheapest, catches gross bugs quickly.
- **Add Arizona (26K) as mid-scale validation** before FL. Tests whether AL findings replicate at 2× data.
- **Run top AL configs (cross-attn, MTLoRA) on FL at least 1 fold** to confirm the asymmetric pattern holds.
- The paper's "characterisation" section becomes a scale-curve: AL → Arizona → FL for both tasks. Makes the paper's scope clearer.

**Status:** `open — 2026-04-18`. Decision deferred to after C12 (max_lr sweep) resolves.

---

## C14 — F27 cat-head scale-dependence flag (FL may need a different task_a head)

**Concern raised:** 2026-04-24. The F27 cat-head ablation swapped task_a `next_mtl (Transformer) → next_gru` and delivered **+3.43 pp cat F1 on AL 5f (F31)** and **+2.37 pp on AZ 5f with Wilcoxon p=0.0312, 5/5 folds positive (F27 validation)**. On FL at n=1 the sign **flipped** (−0.93 pp cat F1, F32). Within n=1 noise, but direction is opposite. Two interpretations:

1. **n=1 fold-selection noise.** FL 1-fold cat F1 has ~0.9 pp variance across the three n=1 runs; F32's 0.6572 lands within the envelope [0.6623, 0.6706] minus noise.
2. **Genuine scale-dependence.** At 127K rows × 4.7K regions, the Transformer head has enough capacity to use more of the signal than the GRU's last-timestep-summarisation bottleneck.

**Resolution path:** F33 (Colab FL 5f × 50ep B3 + `next_gru`) is the decisive test. Three paths documented in `research/F27_CATHEAD_FINDINGS.md §Decision`:

- **Path A** — Commit `next_gru` universally. Simpler narrative; accept small FL cost if at 5f the cat F1 drop is within σ.
- **Path B** — Scale-dependent cat head: `next_gru` for AL/AZ, `next_mtl` for FL/CA/TX. Maximizes per-state performance but fragments the paper story.
- **Path C** — Decisive FL 5f run (F33, in flight on Colab). Settles σ and lets the user commit confidently to A or B.

The NORTH_STAR currently reflects **Path A** (`next_gru` universally) pending F33 results. If F33 shows FL cat F1 below the pre-F27 σ-envelope, Path B applies and `CA/TX` inherit the FL cat head (not the AL/AZ one).

**Paper implication:** if Path B is chosen, the headline table has a footnote "task_a head is scale-dependent (next_gru for AL+AZ ablations, next_mtl for FL+CA+TX headline)". Methodologically honest; aesthetically fragmented.

**Status:** `resolved 2026-04-28 — Path A confirmed via F50 T1.1`. F33 verification (existing H3-alt FL 5f run, no new compute) gave per-fold cat F1 = [67.65, 68.55, 68.04, 68.14, 68.69], mean = **68.21 ± 0.42**, every fold above the pre-F27 envelope [65.72, 67.06] by ≥ 0.6 pp. F32's n=1 −0.93 pp flip was fold-1 noise. Universal `next_gru` cat head committed; CA+TX P3 inherits without footnote. See `research/F50_T1_1_CAT_HEAD_PATH_DECISION.md`. Re-opens only if a future state flips below envelope.

---

## C15 — MTL coupling vs matched-head STL on reg (RE-OPENED 2026-04-28 — scale-conditional, FL flips)

**Concern raised:** 2026-04-24. F21c found STL `next_getnext_hard` beat MTL-B3 by 12-14 pp on reg Acc@10 at AL+AZ.

**First-pass resolution (2026-04-26):** F48-H3-alt closed the gap on AL+AZ. Treated as resolved cross-state pending F37 FL.

**Re-open (2026-04-28, F37 FL landed):** the explicit re-open trigger fired. **F37 STL `next_getnext_hard` FL 5f Acc@10 = 82.44 ± 0.38**, far above MTL-H3-alt FL 73.65 ± 1.25 (per-task best, top10_acc_indist) / 71.96 ± 0.68 (joint best). Paired Wilcoxon **−8.78 pp p=0.0312, 5/5 folds negative**. The matched-head STL ceiling exceeds MTL at FL by a margin paired-test-significant at the n=5 ceiling.

| State | F21c gap (B3 vs STL) | H3-alt vs STL | Wilcoxon | Resolution |
|---|---:|---:|:-:|---|
| AL | -12.04 pp | **+6.25 pp** | p=0.0312 (5/5 +) | **MTL EXCEEDS STL** ✓ |
| AZ | -13.98 pp | -3.29 pp | n.s. | 75% of B3 gap closed |
| FL | (was TBD) | **−8.78 pp** | **p=0.0312 (5/5 −)** | **STL EXCEEDS MTL** ✗ |

**Per-state architectural Δ (F49 frozen-cat vs STL F21c):** AL +6.48 pp; AZ −6.02 pp; **FL −16.16 pp** (5/5 paired Wilcoxon p=0.0312). The cross-attention architecture is **not** a universal lever for reg; it lifts AL but costs AZ and heavily costs FL.

**Updated thesis impact (after F37):**

- **CH18** retained Tier A but reframed as **scale-conditional**: AL exceeds STL, AZ closes 75%, FL ceiling above MTL. The headline contribution at FL is **substrate-only** (cat-side MTL > STL +0.94 pp p=TBD; reg-side MTL < STL).
- **CH21** (top-line claim) revised: "the MTL win is interactional architecture × substrate" holds on AL; at FL the substrate carries the cat advantage while the architecture costs reg. Paper framing must contrast AL (architecture-dominant lift) vs FL (substrate-only) explicitly.
- **Paper recipe** for the H3-alt regime is unchanged (still the recommended joint config when MTL is desired); the *per-task ceiling* on FL reg is `next_getnext_hard` STL.

**Mitigation in paper:**
- Headline tables report MTL H3-alt **and** STL F21c per state; let the per-state pattern speak.
- Discussion section explicitly characterises the architectural-cost-grows-with-cardinality pattern (1.1K → 1.5K → 4.7K regions; architectural Δ +6.5 / −6.0 / −16.2 pp).
- Limitations §6.1 already had a 2-state-dev-regime caveat; updated to flag scale-conditional architectural lift specifically.

**What this does NOT change:**
- Cat-side MTL > STL holds at all 3 states (+3.64 AL / +3.03 AZ / +0.94 FL).
- Substrate findings (CH16 head-invariance, CH18-substrate counterfactual, CH19 per-visit mechanism) are state-replicating questions (CH16 confirmed AL+AZ; FL via F36 queued).
- F49 Layer 1+2 (transfer null, methodological soundness) hold throughout.

**Status:** `re-opened 2026-04-28 with FL caveat — scale-conditional`. Paper-grade resolution: report per-state results without papering over the FL flip; reframe CH21 accordingly. **Mitigation = honest characterisation, not retraction.**

**2026-04-29 update:** F50 Tier 1 closed all three drop-in alternatives (T1.2 HSM head; T1.3 FAMO; T1.4 Aligned-MTL) — **none reaches +3 pp acceptance against the substrate-matched CUDA H3-alt baseline** (paired Wilcoxon p_greater ∈ {0.2188, 0.3125, 0.8438}). CH22b sub-claim added. The FL architectural cost is **robust to head + balancer changes**, strengthening the scale-conditional reading. Tier 1.5 (cross-attn mechanism probes) and Tier 2 (PLE / Cross-Stitch) further test the structural-incompatibility reading. See `research/F50_T1_RESULTS_SYNTHESIS.md`.

**2026-05-29 update (`substrate-protocol-cleanup` Tier B + Tier C evidence — tightens toward "architectural"; does NOT close):** Two new elimination strands narrow the MTL-vs-STL reg coupling toward an architectural (shared-backbone) cause rather than a substrate- or curriculum-side one:
- **Tier B (substrate axis):** four mechanistically-distinct substrate variants (Designs B/J, Lever 4, Lever 5) that dominate canonical c2hgi on STL reg at AL/AZ do **NOT** transfer that advantage to MTL+F1 — disjoint reg flat (|Δ|≤0.38 pp, all p≥0.44) while cat regresses ~−2.4 pp. The shared backbone washes out the substrate reg advantage. Source: `results/substrate_protocol_cleanup/tier_b/phase_b1b2b4_verdict.md` + `phase_b3_verdict.md`.
- **Tier C3 (cat→reg cross-attention K/V):** zeroing the cat-stream K/V into the reg-side cross-attention does NOT recover MTL reg or delay its peak (AL Δ−0.28 ns / AZ Δ+0.01 ns). Combined with P4's frozen-cat-params, both the cat-parameter AND cat-activation pathways to backbone capacity are exonerated. Source: `results/substrate_protocol_cleanup/tier_c/phase_c_verdict.md` §C3.

Net: the residual reg gap is now **triply confirmed** non-cat-interference / non-substrate (P4 params + C3 activations + B substrate); by elimination it points at the shared-backbone architecture, which `mtl_improvement` T2 owns. **C15 is NOT closed here** — the scale-conditional FL-flip framing of the *coupling* itself is unchanged and its formal close awaits the `mtl_improvement` architectural verdict + the §0 paper-canon re-run. log_T-KD (Tier A1 PROMOTED) is an orthogonal supervisory lift that does not bear on the coupling mechanism.

**2026-05-30 update (`substrate-protocol-cleanup` CLOSURE — regime bottleneck now confirmed architectural-and-LOCALIZED; still NOT closed):** Two further strands sharpen C15 from "by-elimination architectural" to a **localized, characterised** regime bottleneck:
- **HGI ceiling (the missing control):** even **HGI** — the STL `next_region` ceiling (+2.12 pp STL reg over canonical at FL) — gives **NO MTL reg advantage** (FL disjoint 64.49 vs 63.98, Δ+0.51, **p=0.41 NS**); HGI's STL reg win VANISHES under B9 joint training. So the coupling is not "designs fail to carry a better substrate"; **there is no substrate (not even HGI) whose reg advantage survives the MTL regime.** Source: `results/substrate_protocol_cleanup/tier_b_fl/hgi_mtl_fl.md`.
- **STL↔MTL isolation cell (apples-to-apples):** identical head/config/state/embeddings, STL `next_stan_flow` α=0 LEARNS region at **~73 % Acc@10** while the SAME config under MTL FLOORS at **~0.03 %** — pinning the loss of reg learning to the **joint-training regime**, not the head or substrate (caveat: α=0 is OOD → the claim is regime-and-config-scoped, B9/50ep, not "the encoder can never learn region under MTL"). Source: `results/substrate_protocol_cleanup/tier_b_fl/phase_b_fl_3way.md`.

**Status:** `architectural-and-localized — NOT closed`. The coupling is now characterised (regime/shared-backbone-dominated; substrate-, prior-, curriculum-, and activation-pathways all eliminated incl. the HGI ceiling and the STL↔MTL isolation cell). It remains OPEN pending the `mtl_improvement` architectural fix + the §0 paper-canon re-run. The v12 default flip (log_T-KD on, ResLN encoder) does **not** bear on this coupling — log_T-KD is an orthogonal prior-pathway reg lift, and ResLN is STL-only with **no MTL benefit** (consistent with this concern). Cross-link: [`results/RESULTS_TABLE.md §0.9`](results/RESULTS_TABLE.md), [`findings/F_SUBSTRATE_PROTOCOL_CLEANUP_SYNTHESIS.md`](findings/F_SUBSTRATE_PROTOCOL_CLEANUP_SYNTHESIS.md), [`results/CANONICAL_VERSIONS.md`](results/CANONICAL_VERSIONS.md).

---

## C16 — CH15 reframed as head-coupled, not retracted (Phase-1, RESOLVED 2026-04-27)

**Concern raised:** 2026-04-27. The original CH15 ("HGI > Check2HGI on next_region under STL STAN at all 3 states") was a paper-record claim. Phase-1 Leg II.2 showed the gap is **head-coupled to STAN's preference for POI-stable smoothness**: under the matched MTL reg head (`next_getnext_hard` = STAN + α·log_T graph prior), Check2HGI ≥ HGI everywhere (AL TOST non-inferior at δ=2pp Acc@10; AZ +2.34 pp Acc@10 / +1.29 pp MRR, p=0.0312, 5/5 folds positive). This **reframes CH15** but does not retract the STAN-head data.

**Policy adopted:** STAN-head data is preserved as a **head-sensitivity probe row** alongside the matched-head row. The paper reports both: "under STAN, HGI > C2HGI; under the matched MTL reg head, C2HGI ≥ HGI. The previous CH15 verdict was head-coupled, not pure substrate."

**Why this is a concern (not a clean win):** A reviewer might ask "why did you change the matched-head policy after the data?" Answer: F27 (2026-04-24) swapped the MTL cat head from `next_mtl` → `next_gru`, and after that swap the STAN reg-head was no longer "matched" to anything in the MTL config — the matched-head policy revision (`research/SUBSTRATE_COMPARISON_PLAN §1.2`) post-hoc aligned the STL baseline to the actual MTL reg head (`next_getnext_hard`). This is documented at the *plan* level (pre-registration) rather than the *result* level.

**Status:** `resolved 2026-04-27`. Re-opens only if a reviewer specifically challenges the matched-head policy revision as data-driven; in that case point at `SUBSTRATE_COMPARISON_PLAN §1.2` (which precedes the data) and the C2 head-agnostic sweep (which closes the head-sensitivity critique with 8/8 probes positive).

---

## C17 — `next_single` cat evidence demoted to head-sensitivity row (Phase-1, RESOLVED 2026-04-27)

**Concern raised:** 2026-04-27. The pre-Phase-1 CH16 "AL +18.30 pp" evidence used `next_single` head (P1.5b). After F27 the matched-head MTL cat head is `next_gru`; the matched-head `next_gru` STL gives Δ=+15.50 pp at AL (smaller than the 18.30 pp from `next_single`). The paper now reports `next_gru` as the headline cat-substrate Δ and demotes `next_single` to a head-sensitivity probe row.

**Concern surface:** A reviewer who reads the legacy "+18.30 pp" number elsewhere and sees "+15.50 pp" in the headline might think we cherry-picked downward. Counter-evidence: the C2 head-agnostic sweep shows BOTH heads (and `next_lstm` and the head-free linear probe) at p=0.0312 positive, with Δs ranging +11.58 to +15.50 pp. **No head choice flips the sign.** The matched-head row is the headline because it's matched; the other rows are reported as the head-sensitivity probe.

**Status:** `resolved 2026-04-27` per `SUBSTRATE_COMPARISON_FINDINGS.md §5` (8/8 probes positive at max significance). The legacy P1.5b `next_single` row remains in `OBJECTIVES_STATUS_TABLE.md §2.1` as a head-sensitivity probe — see also gap #15 of the audit (the `baselines/next_region/comparison.md` "Pattern summary" caveat).

---

## Index

| ID | Concern | Status | Trigger to revisit |
|----|---------|--------|--------------------|
| C01 | n=2 states is thin | monitored | CA/TX data quality issues |
| C02 | FL saturation limits MTL lift | monitored | After CA+TX P3 runs |
| C03 | Joint-score metric (Δm) | resolved 2026-04-16 | — |
| C04 | `next_mtl` → GRU head swap | resolved | Reviewer pushback |
| C05 | P3 null-result fallback | deferred | P3 fails on CA/TX |
| C06 | GRU champion, TCN for grids | resolved | TCN proves inferior in sanity check |
| C07 | Embedding comparison: Check2HGI vs HGI | resolved 2026-04-16 (tied; paper framing pivot) | HGI develops check-in granularity |
| C08 | CH04 retirement reframing | resolved | — |
| C09 | SSD reliability | monitored | Large data on CA/TX |
| C10 | POI-RGNN + HGI-article external baselines (CH17) | open | User provides HGI-article reference |
| **C11** | **User-leakage in STL next-task folds** | **resolved 2026-04-17** | — |
| **C12** | **Hyperparameter mismatch across STL vs MTL baselines** | **under investigation 2026-04-18** | Ablation step 7 max_lr sweep |
| **C13** | **Alabama is a 10K-row dev state; may over-extrapolate to FL/CA/TX** | **open** | Arizona (26K) as intermediate |
| **C14** | **F27 cat-head scale-dependence flag** | **resolved 2026-04-28 — Path A** | Re-opens only if a future state flips below pre-F27 envelope |
| **C15** | **MTL coupling vs matched-head STL on reg** | **resolved 2026-04-26 — H3-alt closes/exceeds gap on AL+AZ+FL** | F37 STL FL ceiling lands above MTL-H3-alt; or seed sweep σ blowup |
| **C16** | **CH15 reframed as head-coupled, not retracted** | **resolved 2026-04-27** | Reviewer challenges matched-head policy revision as data-driven |
| **C17** | **`next_single` cat evidence demoted to head-sensitivity row** | **resolved 2026-04-27** | Reviewer cites legacy +18.30 vs new +15.50 as cherry-picking |
| **C18** | **Encoder-swap leak-probe directional drift (T3.2 ResLN: +2.13 pp leak F1 over canonical at FL)** | **monitored 2026-05-15** | Cumulative drift across T3.3/T3.4 approaches or exceeds the +5 pp red flag |
| **C19** | **F51 `--folds 1` × 5-fold log_T leak bug — audit clears canonical_improvement** | **resolved 2026-05-15** | Any new code path that invokes `scripts/train.py --folds <5` without rebuilding log_T at the same n-splits |
| **C25** | **⭐ MTL reg trained on CLASS-WEIGHTED CE vs UNWEIGHTED STL ceiling → MTL→STL reg gap is largely an objective-mismatch confound (`default_mtl use_class_weights=True`)** | **under re-validation 2026-06-05** | AL/GE/FL re-baseline under unweighted reg CE + per-task-weighting fix + regime-finding re-test |

---

## C19 — F51 `--folds 1` × 5-fold log_T leak bug audit (RESOLVED 2026-05-15)

**Concern raised:** 2026-05-15. A sibling-branch study surfaced the F51 finding (`docs/findings/F51_MULTI_SEED_FINDINGS.md §0`): `scripts/train.py --folds 1` triggers `n_splits = max(2, 1) = 2` for the trainer's StratifiedGroupKFold, but `region_transition_log_seed{S}_fold{N}.pt` files on disk are built with `--n-splits 5` (default). The val users under n_splits=2 (~50 %) are NOT disjoint from the log_T's train users under n_splits=5 (~80 %) → ~30 % val users have their transitions leak into the prior → inflates reg top10_acc_indist by 13–23 pp.

**Audit scope:** entire canonical_improvement study (T1.x → T3.x, v3c series, all multi-seed runs).

**Audit findings (all clean):**
1. `docs/infra/a40/parallel_sweep_runner.sh` passes the **same `$N_FOLDS=5`** to both `compute_region_transition.py --n-splits` (line 104) and `scripts/train.py --folds` (line 117). Same `$SEED` for both. Mismatch is structurally impossible.
2. Grep across `logs/PSWEEP_*.log`, `scripts/`, `docs/infra/`: zero stray `--folds <5` invocations from canonical_improvement code paths. The only `--folds 1` site (`scripts/run_f51_seed42_verify.sh:38`) is an F51-era smoke test, never referenced from canonical_improvement.
3. `src/training/runners/mtl_cv.py:866-884` enforces seed-tagged path `region_transition_log_seed{S}_fold{N}.pt` and **hard-fails** (`FileNotFoundError`) if missing or if only a legacy unseeded file is present. Any completed run proves the seed-correct log_T was found.
4. Every recorded `docs/results/canonical_improvement/*.json` (v3c_FL_seed{0,1,7,100}, t32_resln_FL, t31_gat_FL, t31b_gat_noedge_FL, T1.5 v3c_wd5e2, T2.1, T2.4) has `len(cat_f1_per_fold) == 5` AND `len(reg_top10_per_fold) == 5`. No `--folds < 5` artefacts.
5. F51 fix landed in commit `0bb8a06` (~2 weeks before canonical_improvement began).

**Verdict:** all canonical_improvement results stand. v3c paper-grade gate (5/5 seeds, p=0.03125 reg) is clean. T3.2 single-seed and currently-running T3.2 multi-seed are clean. T3.1 catastrophic cat leak (99 % F1) is GAT-structural, not log_T-driven — every other variant ran with the SAME seed-42 log_T and none showed similar inflation.

**Mitigation landed (defense-in-depth):** `docs/infra/a40/parallel_sweep_runner.sh` now hard-fails at startup if `N_FOLDS < 2`, with a message pointing here and to F51. Prevents the bug class from being re-introduced via a typo or future caller.

**Status:** `resolved 2026-05-15`. Re-opens only if a new code path introduces a `--folds <N>` invocation that does not rebuild log_T at the same `n_splits`.

---

## C18 — Encoder-swap leak-probe directional drift (T3.2 ResLN at FL, MONITORED 2026-05-15)

**Concern raised:** 2026-05-15. T3.2 ResidualLN at FL single seed: leak F1 probe = 42.98 ± 0.34 vs canonical 40.85 ± 0.39 → **Δ = +2.13 pp**. This is well below the established +5 pp red-flag gate (T1.1) and is dwarfed by T3.1 GAT's +11.34 pp catastrophic structural leak, but it is a **non-zero directional drift in the same direction**. Mechanism flagged by the T3.2 advisor (read-only audit, 2026-05-15 17:14): the ResidualLN encoder's layer-2 residual connection provides a near-identity pathway from pre-normalised GCN-1 features, which the leak probe can exploit modestly. Structural but bounded — not pathological — because reg-axis improvement is in lockstep with cat (a label-leak shortcut would lift cat alone, as T3.1 demonstrated).

**Concern surface:** Future encoder variants (T3.3 R-GCN, T3.4 Time2Vec, any T4.x architecture swap that introduces residual identity pathways or attention) may incrementally erode the leak floor. Individually each variant could pass the +5 pp gate yet cumulatively cross it when stacked.

**Mitigation policy adopted:**
- Every T3.x and T4.x encoder swap result row must record leak F1 alongside cat/reg.
- Stack-watch rule: if **Σ(leak Δ across stacked accepted variants) > +5 pp vs canonical**, halt stacking and triage which variant carries the leak signature.
- The leak Δ alone is **not** disqualifying for stacking — it must be paired with the **reg-axis lift** check (label-disjoint axis). T3.1 was disqualified because cat lifted alone (+30 / reg ≈ 0); T3.2 lifts both axes in lockstep (cat +0.96 / reg +0.77) and is therefore not leak-driven at single seed. Multi-seed gate (paired sign test on v3c→ResLN reg AND canonical→ResLN cat) decides whether ResLN stacks.

**Status:** `monitored 2026-05-15`. Revisit when (a) T3.3, T3.4, or any T4.x variant lands with leak Δ > +2 pp, (b) the cumulative accepted-variant leak budget approaches +5 pp, or (c) any future variant lifts cat alone with no reg motion.

---

## C20 — Tier-5 cohort parallel-harness serialization (RESOLVED at integration, 2026-05-17)

**Concern raised:** 2026-05-17 during integration of Tier-5 cohort (T5.1 / T5.2a / T5.2b / T5.3). The cohort was launched in four parallel agent worktrees off the same base commit `a4c757b`, with the intent that each candidate land as an isolated commit reviewable independently. In practice the four worktrees were **not truly isolated**: by the time the second agent committed, the first had already pushed to the shared `main` ref, so subsequent commits were stacked linearly on top of one another rather than parallel siblings. Specifically:

- **T5.2a (`34aa263`)** was authored on the base `a4c757b` and committed first; subsequently became the new `main` tip.
- **T5.3 (`b18f84c`)** was authored on `34aa263` (i.e., on top of T5.2a) and accordingly **bundles a substantial amount of T5.2a substrate** — the `MultiViewWrapper`, `build_view2_graph_*`, the multi-view branch in `check2hgi.py` — that the T5.3 commit message attributes to T5.2a.
- T5.3's own commit (`b18f84c`) contains ONLY the user-facing CLI flag wiring, unit test, and findings doc — the substrate is in `34aa263` as a hidden dependency ("Trojan" in the audit terminology).
- **T5.1 (`e6c56bb`)** and **T5.2b (`0252644`)** were authored on the same base `a4c757b` but landed via cherry-pick during integration; both encountered conflicts with the now-stacked T5.2a / T5.3 baseline.

**Concrete consequence at integration time:** cherry-picking T5.1 and T5.2b onto the T5.3-tip worktree produced merge conflicts in `check2hgi.py`, `Check2HGIModule.py`, `variants.py`, `preprocess.py`, `regen_emb_t3.py`, `test_encoders.py`. Conflicts were resolved by keeping both sides (T5.x candidates are designed to be additively composable). The shared preprocess helper `_build_poi_delaunay_edges` and the `build_poi_delaunay` flag were de-duplicated (T5.2a and T5.2b had two near-equivalent implementations of the same logic in their respective branches).

**Audit-derived risks already mitigated** (see audit-fix list in the integration commit):
- T5.2a's `--n2v-share-table-with-poi-id` flag was reading `getattr(model, 'poi_id_embedding', None)` but T5.1 actually creates `self.poi_id_table` — so share-mode silently fell back to separate-table mode in every co-enabled run. Fixed; also added a hard `ValueError` if share is requested without T5.1 enabled.
- T5.2a as shipped trains a separate `nn.Embedding(num_pois, D)` that NEVER reaches the export path (Checkin2POI / CheckinEncoder), so the downstream MTL effect would be identically zero. Added `--n2v-align-lambda` alignment term bridging skip-gram gradients into the c2hgi encoder via cosine-alignment between `pos_poi_emb` and the n2v table.
- T5.3 `_cross_view_loss` divided by `float(temperature)` with no positivity guard. Added `ValueError` for `temperature <= 0`.
- Audit also flagged T5.1 lacked production-path canonical-preservation tests (only the `POIIdMixedPooler` wrapper was unit-tested); added `test_check2hgi_module_t51_optout` and `test_check2hgi_t51_value_errors` in `tests/canonical_improvement/test_encoders.py`.
- Added a cohort `test_all_t5_canonical_optout` asserting that Check2HGI with ZERO T5 flags allocates zero T5 parameters, produces finite forward outputs of canonical shapes, and yields a finite scalar loss.

**Status:** `resolved 2026-05-17` at the level of the integration branch (`tier5-cohort-integration`). Forward mitigation policy: for any future parallel-agent cohort, agents must commit to dedicated branches off the base, with NO push to the shared ref before integration; integration consolidates via cherry-pick from those branches onto a clean integration branch. The four Tier-5 candidate branches plus this concern entry constitute the audit trail.

---

## C21 — `joint_canonical_b9` selector throws away ~+11 pp of reg capacity from the canonical Check2HGI substrate (RESOLVED 2026-05-24; selector PROMOTED TO CODE DEFAULT 2026-06-03)

> **⭐ 2026-06-03 — the fix is now the CODE DEFAULT (closes the last gap).**
> The 2026-05-24 resolution validated `joint_geom_simple` but in practice the live default
> checkpoint selector was **still the broken v11 `joint_score = 0.5*(cat_f1+reg_f1)`** (the
> geom_simple path was only reachable opt-in, and even the `--save-task-best-snapshots` tracker
> used the interim `joint_geom_lift` acc1-form — NOT geom_simple). As of 2026-06-03 the **default**
> primary-checkpoint selector is `joint_geom_simple = sqrt(cat_macroF1 · reg_Acc@10)` (cat key
> `f1`, reg key `top10_acc_indist`), with **no majority normalization** (F1 and Acc@10 are already
> comparable [0,1] scales; reusing the single-class `majority_fraction` as an Acc@10 baseline would
> be cardinality-wrong). Controlled by `ExperimentConfig.checkpoint_selector` / CLI
> `--checkpoint-selector {geom_simple,joint_f1_mean,geom_lift}`:
> - **geom_simple** (default) — correct, validated (+5.62 pp deployable reg).
> - **joint_f1_mean** — the v11 paper-canon LEGACY (broken) formula; pass this to reproduce v11.
> - **geom_lift** — the interim 2026-04-15 acc1/majority-lift geometric mean (back-compat).
>
> Code sites updated (all now point at the same selected scalar): selection gate, `model_task.log_val`,
> and the `MultiTaskBestTracker` joint slot in `src/training/runners/mtl_cv.py`; field default in
> `src/configs/experiment.py`; flag in `scripts/train.py`. §0.1 (per-task diagnostic-best) is
> unaffected; v11 reproduction = `--checkpoint-selector joint_f1_mean`. See
> [`CANONICAL_VERSIONS.md`](results/CANONICAL_VERSIONS.md) §selector + the 2026-06-03 CHANGELOG entry.

**Resolution summary (2026-05-24, `mtl-protocol-fix` v6 final):** The F1 selector fix (`joint_geom_simple`) landed and validated at FL multi-seed (n=4 seeds × 5 folds, fresh log_T): MTL @ `joint_geom_simple` = 61.54 ± 4.54 vs MTL @ `joint_canonical_b9` = 55.92 ± 3.40 → **+5.62 pp deployable lift at multi-seed**. Capacity gap (disjoint − geom_simple) = 2.37 pp, i.e. **F1 fix recovers ~95% of substrate capacity at FL**. Closure verdict at [`results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md`](results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md). Paper canon n=20 re-run is deferred to [`future_works/paper_canon_reevaluation.md`](future_works/paper_canon_reevaluation.md), sequenced after `mtl_improvement` lands a champion.

---

**Concern raised:** 2026-05-19 during canonical_improvement Tier-6 / T6.4 evaluation. Matched-protocol analysis (shipping FL ep=50 single-seed=42 n=5 + dual-selector re-evaluation) revealed that the production MTL selector (`joint_score = 0.5 * (cat_macro_f1 + reg_macro_f1)` at `src/training/runners/mtl_cv.py:679`) **destroys ~11 pp of reg-top10 capacity that the canonical Check2HGI substrate already produces**. This is a property of the **shipping recipe AS-IS**, not specific to any substrate variant.

**Root cause:** `reg_macro_f1` over ~4 700 sparse FL regions is dominated by rare-class noise and stays ~16-18 % across the entire ep=1-50 trajectory. The selector cannot see `reg_top10_acc_indist`'s peak at ep ~4-5 (~76 %) and collapse by ep ~30 (~65 %). The mean-of-F1s formula is **scale-incoherent** when one head has 7 well-supported classes (cat_macro_f1 ≈ 0.70) and the other has 4 700 sparse classes (reg_macro_f1 ≈ 0.17). The production selector picks late epochs (ep ~29 ± 11) after reg has destabilised, and the resulting reg-top10 std balloons (σ ≈ 9 across folds, vs σ < 0.4 at the reg-best epoch).

**Concrete fingerprint — canonical shipping FL ep=50 single-seed=42, n=5 folds (NO substrate changes):**

| Selector | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| Per-task disjoint best | cat ~35, reg ~4 | 70.49 ± 0.86 | **76.12 ± 0.33** ← what the substrate produces |
| joint_geom_simple (= `sqrt(cat_f1 * reg_top10_indist)`) | 14.0 ± 8.5 | 67.93 ± 1.74 | 72.38 ± 2.20 |
| joint_canonical_b9 (production) | 29.2 ± 10.8 | 69.99 ± 1.13 | **65.38 ± 9.10** ← what the production selector ships |

**Capacity gap on the shipping substrate: ~10.7 pp reg top10** (76.12 − 65.38) — invisible to anyone reading the §0.1 RESULTS_TABLE numbers, which report at joint_canonical_b9.

**Cross-check against published §0.1 numbers:** shipping FL §0.1 multi-seed n=20 reports reg top10 = 63.27 ± 0.10 (= joint_canonical_b9 mean averaged across seeds). The single-seed matched value here (65.38 ± 9.10) is consistent within the single-seed variance. §0.1 reports joint-best, **not** reg-best — this is consistent across the published canon.

**Cross-check against T6.4 substrate variants (canonical_improvement Tier 6):** Under matched protocol single-seed=42 n=5, T6.4 variants (`--two-pass-corruption`, `--p2r-use-infonce τ=0.5`) give Δ_reg = +0.08-0.17 pp at per-task disjoint best vs shipping — **no substrate improvement above noise**. The substrate variants do NOT add capacity above canonical+v3c+T3.2; they only redistribute the same Pareto frontier (with slightly different optimal epochs).

**Why this matters for every study, not just Tier 6.** The protocol bug exists in the production B9 recipe **independently of any substrate change**. Canonical Check2HGI ALREADY produces +11 pp of reg-top10 capacity at FL that the production selector cannot extract. Any study comparing substrate variants under the production selector is comparing them at a checkpoint where reg has destabilised — the substrate-axis ordering may not generalise once the selector is fixed.

**Resolution (provisional):** documented as the urgent next-study scope under [`docs/studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md`](studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md). Three workstreams:

- **F1** — substrate-aware joint_score (use `reg_top10_acc_indist` instead of `reg_macro_f1`, or wire in the already-coded `joint_geom_lift` at `src/training/runners/mtl_cv.py:710`). One-line code change. **Re-evaluation of all canonical_improvement Tier 1-6 candidates AND the shipping baseline under the new selector requires zero retraining** — only re-scan of existing per-epoch val CSVs via `scripts/canonical_improvement/analyze_t64_selectors.py`.
- **F2** — substrate-adaptive MTL balancing (NashMTL revival on FL where the cvxpy solver is well-conditioned; per-task LR decay after reg peak; gradient masking after reg plateau). Goal: prevent reg destabilisation past its early peak so a single checkpoint near ep ~10-15 captures both heads near peak.
- **F3** — substrate × protocol 2×2 ablation as the proper paper headline: (shipping, T6.4 substrate variants) × (B9 selector, F1-fix selector). Shows the protocol-axis effect is **larger** than the substrate-axis effect on reg.

**Status:** `RESOLVED 2026-05-24` via `mtl-protocol-fix` v6 final (see resolution summary at the top of this concern). F1 selector code change landed at `mtl_cv.py:679`; FL multi-seed validated +5.62 pp deployable. Full §0.1 n=20 paper-canon re-evaluation across all 5 states deferred to [`future_works/paper_canon_reevaluation.md`](future_works/paper_canon_reevaluation.md) (sequenced after `mtl_improvement` lands champion). Re-opens if any new substrate intervention claims a paper-grade reg lift WITHOUT the F1 fix applied to its baseline.

**Numbers source-of-truth:** `docs/results/canonical_improvement/T6_4_dual_selector_final.{json,md}` (all 3 arms: shipping, two_pass, infonce τ=0.5; single-seed=42, n=5 folds, ep=50).

---

## C22 — `regen_emb_t3.py` silently leaves stale per-fold `region_transition_log.pt`; `train.py` does not validate freshness (DISCOVERED 2026-05-20)

**Concern raised:** 2026-05-20 17:48 UTC during `mtl_protocol_fix` Phase 2 P2 multi-seed FL audit.

**Bug:** `scripts/canonical_improvement/regen_emb_t3.py` regenerates check-in embeddings + `next_region.parquet`, but **does NOT rebuild `region_transition_log_seed{S}_fold{N}.pt`**. `scripts/train.py --per-fold-transition-dir DIR` loads these files at fold init time but **does NOT validate that their content is current** (no mtime/hash check vs `next_region.parquet`).

**Discovery fingerprint:** FL seed=42 `region_transition_log_seed42_fold*.pt` mtime was **2026-05-06 13:29** — two weeks old, never rebuilt across the canonical_improvement Tier 6 FL-MTL sweeps (which ran 2026-05-19) or the early `mtl_protocol_fix` Phase 1 v1-v4 (which ran 2026-05-20 03:00-13:13). Hash differs from a freshly-rebuilt file at the same seed.

**Empirical impact (FL seed=42, fresh log_T vs stale May-6 log_T):**

| Selector | STALE log_T | FRESH log_T | Δ (stale inflation) |
|---|---:|---:|---:|
| STL `next_stan_flow` Acc@10 | 78.91 ± 0.27 | **70.89 ± 0.52** | **+8.02 pp** |
| MTL @ joint_geom_simple | 72.88 ± 1.49 | **61.14 ± 0.95** | **+11.74 pp** |
| MTL @ joint_canonical_b9 | 61.47 ± 11.48 | 53.73 ± 9.22 | **+7.74 pp** |

Fresh-log_T seed=42 values match multi-seed {0,1,7,100} mean within σ → development-seed bias is **zero at FL** once log_T is fresh (the apparent "seed=42 outlier" was 100% stale-log_T artifact).

**Why the staleness survived:** Region-idx layout is STABLE across regens (`poi_to_region` map is cached in `temp/checkin_graph.pt` via `force_preprocess=False`), so the log_T row/col indices remain semantically valid. **Only the train-fold-partition entries of the matrix differ between OLD and NEW log_T at the same seed.** The mismatch leaks across train/val splits in a way that inflates Acc@10.

**Blast radius across prior studies:**
- **Tier 5 (T5.1, T5.2a/b, T5.3) — CLEAN.** All sandboxed in `runs/T*/` with per-(seed,state) log_T rebuilt via `resume_stage3plus.sh` (which calls `compute_region_transition.py --per-fold --seed $SEED`). Host-tree May-6 file never read.
- **Tier 6 AL/AZ — CLEAN.** Same sandboxed pattern.
- **Tier 6 FL-MTL sweeps (`t61/t62/t64_fl_mtl_sweep.sh`, FL block of `t63_sweep.sh`) — STALE.** These scripts swap embeddings + rebuild `next_region.parquet` but do NOT call `compute_region_transition.py`. They consumed the May-6 host-tree log_T verbatim. **Internally consistent (both shipping baseline AND every variant used SAME stale log_T) so RELATIVE falsifications HOLD; ABSOLUTE Acc@10 values biased by unknown sign-and-magnitude (~+0 to +12 pp likely).** No Tier 5/6 winner was missed: closest "almost-winner" T6.2 a2.0_0.3 was within +0.18 pp on stale log_T — cannot flip an 8+ pp gap.
- **`mtl_protocol_fix` Phase 1 v1-v4 FL seed=42 — STALE.** Final Phase 1 v5 verdict supersedes v4 with FRESH-log_T multi-seed FL numbers that match §0.1 v11 within σ.

**Resolution (provisional):**
1. **`CLAUDE.md` updated 2026-05-20** with mandatory stale-log_T preflight check before any MTL/STL run that passes `--per-fold-transition-dir`.
2. **Patch `regen_emb_t3.py`** to optionally trigger `compute_region_transition.py --per-fold --seed $SEED` after `next_region.parquet` regeneration. *(Open work item.)*
3. **Patch `scripts/train.py`** to fail loudly if `--per-fold-transition-dir` per-fold file mtime predates `output/check2hgi/{state}/input/next_region.parquet` mtime. *(Open work item.)*
4. **Patch all `t6*_fl_mtl*.sh` scripts** to call `compute_region_transition.py` immediately after `regen_emb_t3.py`. *(Open work item.)*
5. **Cite the caveat** when reporting any Tier 6 FL-MTL absolute Acc@10. Cross-reference [`docs/results/mtl_protocol_fix/phase1_verdict.md`](results/mtl_protocol_fix/phase1_verdict.md) (the closure document for this concern).

**Status:** `partially resolved 2026-05-20`. Items 1-3 landed (CLAUDE.md preflight rule, `regen_emb_t3.py` auto-calls `compute_region_transition.py`, `scripts/train.py` preflight raises on stale log_T). Item 4 (patch all `t6*_fl_mtl*.sh` scripts) deferred — those scripts belong to the closed `canonical_improvement` Tier 6 sweep and are unlikely to re-run; documented as known limitation. Item 5 cite-the-caveat is now standard practice in `mtl-protocol-fix` and `substrate-protocol-cleanup` AGENT_PROMPTs (C22 stale-log_T preflight gate at every run).

**Numbers source-of-truth:** `docs/results/mtl_protocol_fix/phase2p5_FL_stale_vs_fresh.{json,md}` (FL seed=42 stale vs fresh log_T side-by-side comparison; matched protocol single-seed n=5 folds ep=50).

---

## C23 — Development-seed (seed=42) contamination at large states; paper-grade canon uses seeds {0,1,7,100} (DOCUMENTED 2026-05-20)

**Concern raised:** 2026-05-20 18:21 UTC during `mtl_protocol_fix` Phase 2 P6 CA/TX preliminary.

**Observation:** Across the project, `seed=42` has been the **development seed** — all recipe choices (B9 vs H3-alt, BS=2048/1024/512, LR schedules, cat-weight, alpha-no-WD, alternating-step, min-best-epoch) were tuned by observing seed=42 validation. `RESULTS_TABLE.md §0.1 v11` paper-canonical numbers use seeds {0, 1, 7, 100} **explicitly excluding seed=42** to avoid development-seed contamination — standard ML practice (dev set ≠ test set).

**Empirical fingerprint (seed=42 single-seed vs §0.1 v11 multi-seed n=20, fresh log_T):**

| State | n_regions | seed=42 disjoint | §0.1 v11 (n=20) | Δ (dev-seed bias) |
|---|---:|---:|---:|---:|
| AL | 1,109 | 50.82 ± 3.21 | 50.17 ± 0.24 | +0.65 (within σ) |
| AZ | 1,547 | 41.33 ± 2.73 | 40.78 ± 0.07 | +0.55 (within σ) |
| FL | 4,703 | 63.91 ± 0.16 (multi-seed fresh) | 63.27 ± 0.10 | +0.64 (within σ) |
| **CA** | **8,501** | **50.61 ± 1.23** | **47.35 ± 0.11** | **+3.26 (significant)** |
| **TX** | **6,553** | **50.83 ± 1.89** | **42.84 ± 0.14** | **+7.99 (large)** |

Small states (AL/AZ) and FL: seed=42 ≈ multi-seed → no development bias. Large states (CA/TX): seed=42 substantially overshoots multi-seed → real development-seed overfit, likely from recipe-tuning at FL+seed=42 not generalising to CA/TX class-counts.

**Why this matters:** Any paper-grade comparison MUST use multi-seed {0,1,7,100} at minimum, not seed=42 alone. Single-seed=42 measurements at CA/TX overstate true generalisation by **+3 to +8 pp** on reg Acc@10.

**Resolution:** This is a **convention**, not a bug. The convention is now documented in:
- [`CLAUDE.md`](../CLAUDE.md) — paper-grade recipe block includes seeds {0,1,7,100} guidance + dev-seed warning.

**Status:** `documented 2026-05-20`. Re-opens if any future study reports seed=42-only numbers as paper-grade without flagging the convention.

**Numbers source-of-truth:** `docs/results/RESULTS_TABLE.md §0.1 v11` (paper canonical) + `docs/results/mtl_protocol_fix/phase1_verdict.md` (dev-seed bias empirical fingerprint).

---

## C24 — STAN/`next_stan_flow` bidirectional safety depends on target staying outside the 9-window (WATCH-ITEM 2026-05-28)

**Concern raised:** 2026-05-28 from `substrate-protocol-cleanup` Tier D1 window/mask audit advisor pass.

**Observation:** The B9 reg head `next_stan_flow` (STAN) intentionally uses bidirectional self-attention across the 9-position input window — no causal mask. The Tier D1 audit confirmed this is currently safe because `generate_sequences` (`src/data/inputs/core.py:59-89`) keeps the target check-in strictly outside the 9-window via half-open slicing + tail-shift excision.

**Risk:** Any future next-head variant that injects the target into the window — e.g. teacher-forcing-style training, target-aware positional bias, or a sliding-window shift that includes the target index — would silently leak via STAN's bidirectional attention. The B9 substrate + STAN combination is leak-safe only as long as the input-construction invariant holds.

**Status:** `watch-item 2026-05-28`. Re-audit triggered if any of:
- New head registered under `src/models/next/next_stan_*` or any bidirectional `next_*` variant.
- Change to `generate_sequences` window construction.
- New `task_b_input_type` introduced that re-interprets the 9-window.

**Reference:** `docs/studies/substrate-protocol-cleanup/window_mask_audit.md` + log.md 2026-05-28 advisor entry.

---

## C25 — MTL reg head trained on CLASS-WEIGHTED CE while the STL reg ceiling is UNWEIGHTED → the MTL→STL reg gap is substantially an objective-mismatch confound (DISCOVERED 2026-06-05; UNDER RE-VALIDATION)

**Concern raised:** 2026-06-05, from the `mtl_improvement` Tier-2P root-cause hunt (T2P.0).

**The bug.** `ExperimentConfig.default_mtl` silently sets `use_class_weights=True` (`src/configs/experiment.py:364`; the dataclass default `:235` is also `True`), while `default_next` (the STL next/reg factory) sets it `False` (`:403`). This flows to `src/training/runners/mtl_cv.py:1283-1291`, where the **MTL reg criterion** becomes `CrossEntropyLoss(weight=alpha_next)` with `alpha_next = compute_class_weights(...)` (`:1276`). So the ~1109/4702-region MTL reg head trains on **class-BALANCED CE**, whereas the STL reg ceiling (p1 / `default_next`) and every clean reconstruction train on **UNWEIGHTED CE**.

**Why it depresses the headline metric.** The reg head is reported by **`top10_acc_indist` (Acc@10)** — a frequency-weighted, head-heavy metric. Class-balancing up-weights rare regions and down-weights the ~22%-majority region → it optimizes *macro* accuracy and *away from* top-K. So the MTL reg number is depressed ~10-14pp **purely by the loss objective**, from epoch 1, on every fold; the effect scales with class count / imbalance (FL −14 > AL −10), which exactly matches the observed state-scaling.

**Verified.** T2P.0 (AL, `mtlnet_crossattn_dualtower` private_only) with `--no-class-weights` → reg disjoint **64.81 ≥ the STL (c) ceiling 62.88** (agent fold1 68.69), vs the buggy default **52.90**. The deficit is an objective mismatch, not architecture/substrate/joint-loop.

**Blast radius.** EVERY `default_mtl` MTL run used `use_class_weights=True` — undocumented in `NORTH_STAR.md` / `CANONICAL_VERSIONS.md`. Affected (absolute MTL reg ~10-14pp low): **§0.1 deployable MTL reg, the MTL→composite gap, the `mtl_improvement` Tier-2/2P "MTL sacrifices reg / irreducibly architectural / ship-the-composite" line, and potentially the central REGIME FINDING** ("STL substrate gains wash out in MTL" compared depressed-MTL-reg to full-STL-reg → must be re-tested under unweighted CE: does the substrate gain re-appear at MTL?). *Relative* within-MTL Δs (v14 vs canonical, dual-tower vs base_a — all class-weighted) are common-mode and likely hold. Directly **reopens C12** (STL-vs-MTL HP mismatch) and **bears on C15** (MTL coupling vs matched-head STL on reg — the "resolution" never controlled for this loss-objective axis).

**Pragmatic resolution (in flight, user-approved 2026-06-05).** (1) per-task class-weighting code fix (reg OFF for Acc@10; cat decided by macro-F1 — the current `--no-class-weights` couples both heads, `mtl_cv.py:1284-1290`); (2) re-test the regime finding + a real joint run (`category-weight 0.75`) + §0.1/composite re-baseline under unweighted reg CE at **AL/GE/FL** (CA/TX deferred). **Do NOT cite any absolute MTL-reg number until re-baselined.** Frozen (c)/(d) STL ceilings are UNAFFECTED (unweighted).

**Second, smaller artifact (logged here, separate):** the MTL reg deployment SNAPSHOT is selected by `MultiTaskBestTracker.reg_best` monitor = `accuracy` (Acc@1; `src/tracking/best_tracker.py:116`, preset `primary_metric=ACCURACY` `src/tasks/presets.py:100`), whose best epoch ≠ the Acc@10-best epoch → understates the *deployable* reg ~2-3pp. Independent of the disjoint metric (`per_metric_best.top10_acc_indist`, oracle-Acc@10, selector-independent).

**Status:** `under re-validation 2026-06-05`. Closes when the AL/GE/FL re-baseline under unweighted reg CE lands + the per-task-weighting fix is committed + the regime finding is re-tested.

**Reference:** `docs/studies/mtl_improvement/log.md` 2026-06-05 ROOT-CAUSE entry; `HANDOFF.md` §top; `PAPER_UPDATE.md` superseding banner. Scripts: `scripts/mtl_improvement/t2p0_*`.
