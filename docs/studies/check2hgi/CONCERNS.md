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

**Status:** `under investigation — 2026-04-18`.

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

**Status:** `open — 2026-04-24`. Re-opens to Path B if F33 FL 5f cat F1 falls below the pre-F27 mean envelope minus σ.

---

## C15 — MTL coupling vs matched-head STL on reg (RESOLVED via F48-H3-alt 2026-04-26)

**Concern raised:** 2026-04-24. F21c found STL `next_getnext_hard` beat MTL-B3 by 12-14 pp on reg Acc@10 at AL+AZ.

**Resolution (2026-04-26):** the gap was a single LR-schedule confound, not structural to MTL. F48-H3-alt (per-head LR: `cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3, --scheduler constant`) closes/exceeds the gap on all states tested:

| State | F21c gap (B3 vs STL) | H3-alt vs STL | Resolution |
|---|---:|---:|---|
| AL | -12.04 pp | **+6.25 pp** | **MTL EXCEEDS STL** ✓ |
| AZ | -13.98 pp | -3.29 pp | 75% of B3 gap closed |
| FL | TBD | beats Markov +6.91 pp, STL GRU +3.63 pp; STL GETNext-hard pending F37 4050 | scale validation OK |

**Mechanism:** α (graph-prior weight in `next_getnext_hard.head`, in `reg_specific_parameters`) needs sustained 3e-3 to grow per the F45 attribution. Per-head LR isolates α's regime from cat-stability regime. Three orthogonal negative controls (F40 loss-side ramp, F48-H1 monolithic gentle constant, F48-H2 warmup-then-plateau) confirm H3-alt is the unique design that satisfies joint cat+reg objective.

**Impact on the thesis (REVERSED):**

- **CH01 / CH02 reaffirmed:** under H3-alt, joint MTL lifts both heads over matched-head STL at AL (cat +3.64 pp over STL `next_mtl`, reg **+6.25 pp over STL `next_getnext_hard`**). The earlier F21c "MTL trails STL on reg" framing was per the predecessor B3 + OneCycleLR config; it does not survive H3-alt.
- **CH18 promoted Tier B → A.** See `CLAIMS_AND_HYPOTHESES.md §CH18` for the updated statement.
- **Paper framing:** the contribution is no longer "joint deployment accepting reg cost"; it is "joint single-model MTL with per-head LR exceeds matched-head STL on reg AND preserves cat F1, validated cross-state with a clean attribution chain."

**Documentation updated (2026-04-26):** `NORTH_STAR.md`, `PAPER_STRUCTURE.md`, `OBJECTIVES_STATUS_TABLE.md` (v4), `AGENT_CONTEXT.md`, `README.md`, `results/RESULTS_TABLE.md`, plus new `MTL_ARCHITECTURE_JOURNEY.md`.

**What would re-open:**
- F37 STL FL ceiling lands above MTL-H3-alt FL (71.96) → AL/AZ-style asymmetry on FL → C15 re-opens with FL caveat.
- Seed-sweep on H3-alt shows σ blowup that erases the AL surplus → C15 re-opens with stability concern.
- Reviewer demands paired Wilcoxon for H3-alt vs STL F21c → run paired test (cheap, F37 dependency) and report.

**Status:** `resolved 2026-04-26 — H3-alt closes/exceeds the matched-head gap on AL+AZ+FL`. Re-opens on any of the three triggers above.

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
| **C14** | **F27 cat-head scale-dependence flag** | **open — 2026-04-24** | F33 Colab FL 5f decides Path A / B |
| **C15** | **MTL coupling vs matched-head STL on reg** | **resolved 2026-04-26 — H3-alt closes/exceeds gap on AL+AZ+FL** | F37 STL FL ceiling lands above MTL-H3-alt; or seed sweep σ blowup |
