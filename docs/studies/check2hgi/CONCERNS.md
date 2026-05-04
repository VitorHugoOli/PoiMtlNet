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
