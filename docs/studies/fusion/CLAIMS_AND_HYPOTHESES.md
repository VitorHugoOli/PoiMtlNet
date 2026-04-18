# Claims and Hypotheses Catalog

This is the **authoritative list** of every claim/hypothesis we intend to validate. Each has:
- **ID** (C01, C02, …) for stable referencing.
- **Statement** — what we want to show.
- **Source** — why we care (prior finding / critical review / new hypothesis).
- **Test** — which experiment(s) validate it.
- **Phase** — where the test lives in the study plan.
- **Status** — `pending` / `running` / `confirmed` / `refuted` / `partial` / `abandoned`.
- **Evidence** — pointer to `docs/studies/results/.../` once tested.
- **Notes** — caveats, surprises, follow-ups.

**Rule:** no claim is made in the paper without `status ∈ {confirmed, partial}` and evidence pointer.

---

## Tier A — Claims that directly support the paper thesis

### C01 — Task-specific multi-source fusion improves over single-source HGI

**Statement:** Concatenating complementary signals (Sphere2Vec+HGI for category, HGI+Time2Vec for next) outperforms the single-source HGI embedding on joint F1.

**Source:** Core paper thesis.
**Test:** P3.1 — run champion arch+optimizer on {DGI, HGI, Fusion} at matched 5f × 50ep on AL and AZ.
**Phase:** P3.
**Status:** `pending` (prior AL result from pre-bug data suggested yes, must be re-validated).
**Notes:** If fusion ≈ HGI at matched settings, thesis weakens to "fusion is robust, not superior."

---

### C02 — Gradient-surgery optimizers improve over equal weighting on fusion

**Statement:** On multi-source fusion with scale-imbalanced sources, CAGrad/Aligned-MTL produce higher joint F1 than equal_weight at matched effective batch size and matched training budget.

**Source:** Prior Stage 1 result (Stage 1 showed 25% gap, but at short training and unmatched batch — weakened by T0.2 at matched batch showing essentially zero gap).
**Test:** P1.2 — within the 5×20 grid, compare ca/al vs eq for DSelectK on fusion, matched batch (grad_accum=1), at both 1f×10ep screening and 5f×50ep confirmation.
**Phase:** P1.
**Status:** `refuted` — **upgraded from `partially_refuted` after F2 multi-seed (3 seeds per config) on 2026-04-17.** Under matched effective batch and multi-seed on both states, gradient-surgery shows **no statistically significant advantage** over equal_weight/uncertainty_weighting. The single-seed +0.0051 AL gap at joint-peak inverts to −0.0034 under multi-seed (eq wins); at joint@T Δ = −0.0005 (tied). AZ: uw vs dwa within ±0.0018. All Welch-like t-stats |t| < 0.7. **C02 finding is null across both states, both checkpoint policies, and 3 seeds.**
**Evidence (P1a screen, 1f × 10ep, 2026-04-16):** Best-gradient vs best-equal_weight: AL +0.0010, AZ +0.0061. Top-10 of both states mixes gradient, loss-dynamic, and static optimizers; no optim class dominates.
**Evidence (P1b promote 2f × 15ep, 2026-04-17):** AL best-grad − best-eq = **−0.0011** (equal_weight edges grad-surgery).
**Evidence (P1c confirm 5f × 50ep, 2026-04-17):**
- AL, joint-peak selection: mmoe4 × gradnorm = 0.4082 vs best equal_weight cgc22×eq = 0.4031 → **Δ = +0.0051 (Z = 0.39)**.
- AL, **per-task-best selection**: best gradient 0.4220 vs best equal_weight 0.4229 → **Δ = −0.0009**. equal_weight tied for first.
- AZ, joint-peak selection: best gradient = 0.4369 vs best static (uw) = 0.4374 → Δ = −0.0005.
- AZ, per-task-best selection: best gradient 0.4406 vs best static 0.4416 → Δ = −0.0010.
- Combined: across both states and both checkpoint policies, effect size is within noise. Equal-weighting is fully competitive on fusion at matched batch.
**Evidence (F2 multi-seed 3 seeds × 4 configs, 5f × 50ep, 2026-04-17):**
- AL mmoe4 × gradnorm (gradient): joint@J = **0.4080 ± 0.0008**, joint@T = 0.4232 ± 0.0022.
- AL cgc22 × equal_weight (static): joint@J = **0.4115 ± 0.0087**, joint@T = 0.4237 ± 0.0026.
- AL Δ(grad − eq): joint@J = **−0.0034**, joint@T = **−0.0005**. Welch-like t ≈ −0.68, −0.25. Static *mean* beats grad at joint@J, tied at joint@T.
- AZ cgc21 × uncertainty_weighting (static): joint@J = 0.4337 ± 0.0049, joint@T = 0.4394 ± 0.0033.
- AZ cgc21 × dwa (loss-dynamic): joint@J = 0.4330 ± 0.0029, joint@T = 0.4412 ± 0.0040.
- AZ Δ(dwa − uw): joint@J = −0.0006, joint@T = +0.0018. Welch-like t ≈ +0.18, −0.58.
- **All 4 Welch |t| < 0.7.** No config class provides measurable ceiling advantage.
**Secondary finding (C18 reproducibility):** mmoe4 × gradnorm has the *lowest* seed variance (0.0008 on joint@J) — remarkably stable. cgc22 × equal_weight has the *highest* (0.0087) — driven by seed-2024 outlier (0.4204). The stable mean isn't gradient-surgery; it's the interaction of `mmoe4` routing with `gradnorm`'s adaptive weighting.
**Notes:** Ratifies N02 and the original T0.2 observation. **Updated paper narrative:** "Under matched effective batch, optimizer choice on 128-dim fusion is a null effect on joint F1 ceiling; it affects *reliability* (seed variance). The first-order lever on fusion is neither architecture nor optimizer — both are second-order at 5f × 50ep." See `issues/P1_METHODOLOGY_FLAWS.md` F1, F2.

---

### C03 — Equal weighting suffices on single-source embeddings (replicates Xin et al.)

**Statement:** On single-source DGI and HGI (no scale imbalance), equal_weight matches or exceeds adaptive optimizers on 2-task MTL.

**Source:** Xin et al. (NeurIPS 2022); prior ablation Finding 1.
**Test:** P1.3 — within the 5×20 grid, HGI and DGI runs with equal_weight vs ca/al/db/uw for best architecture.
**Phase:** P1 (extended with DGI/HGI runs) or P3.
**Status:** `pending`.
**Notes:** Confirming this is important because it's the "other half" of the fusion-specific claim — if equal_weight fails on HGI too, our fusion story needs different framing.

---

### C04 — Architecture rankings depend on the embedding source

**Statement:** The winning MTL architecture (CGC vs DSelectK vs MMoE vs base) changes when the embedding engine changes.

**Source:** Prior Finding 8 — CGC won on HGI, DSelectK won on DGI.
**Test:** P3.2 — run the full 5-architecture sweep (with best optimizer per group) on {DGI, HGI, Fusion} at AL.
**Phase:** P3.
**Status:** `pending`.
**Notes:** Even if not a paper headline, it's a cautionary finding for the MTL community.

---

### C05 — Expert-gating architectures beat hard parameter sharing with FiLM

**Statement:** CGC, MMoE, DSelectK all outperform the base MTLnet (FiLM + shared layers) on the joint score, regardless of optimizer.

**Source:** Prior Finding 7, 12.
**Test:** P1.4 — within the 5×20 grid, compare mean joint of base arch vs each expert-gating arch.
**Phase:** P1.
**Status:** `partial` — **downgraded 2026-04-17 after F4 base×eq confirm run.** Screen evidence (1f × 10ep, 75 cells per state) strongly supports the claim: every expert arch beats base with +0.03–0.045 on AL, +0.005–0.017 on AZ. **However, at confirm protocol (5f × 50ep), base × equal_weight matches or beats most expert cells** at joint@J and is within 0.0014 (pooled fold-noise 0.012) of *every* expert cell at joint@T. The direction of C05 inverts between the two protocols. Claim must be stated carefully: expert-gating helps at short training but not at matched 5f × 50ep.
**Evidence (P1a screen, 2026-04-16):** AL base mean joint = 0.352; cgc22 = 0.394, cgc21 = 0.384, mmoe4 = 0.396, dsk42 = 0.382 (+0.03–0.04 over base). AZ base mean = 0.414; cgc22 = 0.419, cgc21 = 0.431, mmoe4 = 0.426, dsk42 = 0.423 (+0.005–0.017 over base). Every expert-gating arch > base in both states.
**Evidence (F4 base × equal_weight AL 5f × 50ep, 2026-04-17):**
- Reported: `P1_AL_confirm_base_equal_weight_seed42` — joint@J = **0.4070**, joint@T = **0.4217**, cat = 0.8282, next = 0.2698.
- **AL confirm ranking at joint@J (with F4)**: mmoe4×gradnorm 0.4082 > **base×eq 0.4070** > cgc22×excess 0.4043 > cgc22×nash 0.4034 > cgc21×bayes 0.4034 > cgc22×eq 0.4031. Base lands 2nd, ahead of 4 expert cells.
- **AL confirm ranking at joint@T (with F4)**: cgc22×eq 0.4229 > mmoe4×gradnorm 0.4220 > cgc22×nash 0.4217 ≈ **base×eq 0.4217** ≈ cgc22×excess 0.4215 ≈ cgc21×bayes 0.4215. All 6 configs within 0.0014. Effective tie; base is indistinguishable from expert gating.
- Matched-optimizer comparison (same optimizer, vary arch): `base × eq` vs `cgc22 × eq` on AL: joint@J Δ = **+0.0040** (base wins); joint@T Δ = **−0.0011** (cgc22 wins by 1 part in 400). Flips sign across checkpoint policies — well inside noise.
**Notes:**
- The screen → confirm pattern: expert-gating matters most when training budget is tight (1f × 10ep). At 5 folds × 50 epochs, the extra training data/epochs let base arch catch up.
- Winning archs differ by state (AL favors mmoe4/cgc22; AZ favors cgc21) — partial signal for C04 (embedding → state dependence) within same fusion engine.
- **Paper implication:** reframe C05 as "expert-gating accelerates convergence; the long-training ceiling is the same." Weaker claim but still publishable; aligns with the C02/N02 framing.
- **Follow-up needed:** one `base × nash_mtl` (or `base × gradnorm`) confirm cell would strengthen the negative finding. Multi-seed of base × equal_weight would formalize the tie under F2-style variance estimation.

---

## Tier B — Claims about MTL itself (the core reviewer concern)

### C06 — Multi-task learning improves over single-task (when configured correctly)

**Statement:** With fusion + DSelectK + best optimizer, MTL-joint-training outperforms training each task independently with the same encoder.

**Source:** **Critical review flagged this as the #1 missing control.** Prior CBIC 2025 finding was MTL ≈ single-task; we need to show the new configuration overturns this.
**Test:** P2.1 — single-task-category vs single-task-next vs MTL-joint, all on fusion+DSelectK+best-optim.
**Phase:** P2.
**Status:** `pending` — **this is the #1 critical test of the whole paper.**
**Notes:** If MTL ≈ single-task, the "MTL works when configured right" thesis collapses. Reframe to "fusion improves POI prediction" (MTL orthogonal).

---

### C07 — MTL benefit is asymmetric per task

**Statement:** Category and next-POI tasks receive different magnitudes of benefit from MTL joint training.

**Source:** Prior observation that category F1 seemed to benefit more than next F1.
**Test:** P2.2 — per-task delta of MTL vs single-task on the same embedding and encoder.
**Phase:** P2.
**Status:** `pending`.

---

### C08 — Standalone head rankings do not transfer to MTL

**Statement:** A head that wins in isolated single-task training underperforms when plugged into the MTL pipeline where heads co-adapt with the backbone.

**Source:** Prior Finding 3 — Phase 4 showed DCN/TCN swaps hurt joint by 20% on HGI.
**Test:** P2.3 — run full 9×10 head grid on MTL (with best arch+optim); run best single-task heads on single-task setup; compare rankings.
**Phase:** P2.
**Status:** `pending`.

---

### C09 — Heads co-adapt with backbone (mechanism for C08)

**Statement:** A head trained within MTL, then evaluated with a different backbone, performs worse than when its evaluated backbone matches its training backbone.

**Source:** Hypothesis explaining C08.
**Test:** P2.4 (optional, advanced) — train MTL; freeze head; swap backbone; measure drop.
**Phase:** P2 (supplementary).
**Status:** `pending`.
**Notes:** Nice-to-have for mechanistic explanation but not blocking.

---

### C10 — DCN head specifically exploits Sphere2Vec×HGI cross-features on fusion

**Statement:** The Deep & Cross Network category head provides a unique benefit on fusion (vs on HGI-only) because it can learn explicit cross-features between the two embedding halves.

**Source:** Prior Stage 2 result (DCN +1.7% at short training on fusion) — disappeared at 50 epochs.
**Test:** P2.5 — DCN vs default head on fusion at matched 5f×50ep; also run DCN vs default on HGI-only.
**Phase:** P2.
**Status:** `partially_refuted` (DCN advantage was short-training only; re-test on new data).

---

## Tier C — Claims about embedding quality

### C11 — Embedding quality is the dominant factor in POI-MTL performance

**Statement:** The gap between embedding choices (DGI vs HGI vs Fusion) is larger than the gap between any other design choice (architecture, optimizer, heads) given a fixed embedding.

**Source:** Prior Finding 6 — HGI > DGI by 45% joint.
**Test:** P3.3 — compute the range of joint scores across embeddings (holding arch+optim+heads fixed) vs the range within an embedding (holding embedding fixed, varying other factors).
**Phase:** P3.
**Status:** `pending`.
**Notes:** If true, supports the claim "focus research on embedding, not MTL machinery."

---

### C12 — CBIC's config fails on ALL embeddings (not just DGI)

**Statement:** FiLM hard-sharing + NashMTL underperforms modern configurations regardless of which embedding is used.

**Source:** Needed to defend "the CBIC paper's 'MTL doesn't help' was configuration-specific, not data-specific."
**Test:** P3.4 — run CBIC config (base MTLnet + NashMTL) on {DGI, HGI, Fusion}; compare to new champion (DSelectK + best-optim) on same three.
**Phase:** P3.
**Status:** `pending`.
**Notes:** Key for the paper narrative correcting CBIC 2025.

---

### C13 — Fusion specifically (not just richer embedding dim) drives improvement

**Statement:** A 128-dim HGI (e.g., trained at doubled dim) would not achieve what fusion achieves — the gain is from complementary signals, not from more dimensions.

**Source:** Critical review noted the 64D→128D confound.
**Test:** P3.5 (optional, depends on embedding availability) — HGI at 128D vs Fusion at 128D on AL.
**Phase:** P3.
**Status:** `pending` — requires HGI-128D embedding generation.
**Notes:** Hard to test without retraining HGI at different dim. May become journal material.

---

### C14 — Time2Vec (temporal) specifically helps next-POI prediction

**Statement:** Adding Time2Vec to the next-POI input provides signal that HGI alone cannot replace.

**Source:** Prior Stage 0 result — fusion's +26% next F1 vs HGI-only.
**Test:** P3.6 — ablate Time2Vec from the fusion input for next task; compare.
**Phase:** P3.
**Status:** `pending`.

---

## Tier D — Hyperparameter / robustness claims

### C15 — DSelectK is insensitive to its hyperparameters near defaults

**Statement:** DSelectK(e=4, k=2, temp=0.5) is within ±2% joint of neighboring values in a reasonable range.

**Source:** Needed to defend arbitrary hparam choices.
**Test:** P4.1 — sweep e ∈ {2,4,6,8}, k ∈ {1,2,3,4}, temp ∈ {0.1,0.3,0.5,0.7,1.0}.
**Phase:** P4.
**Status:** `pending`.

---

### C16 — Growing the shared backbone improves MTL transfer

**Statement:** Increasing shared_layer_size or num_shared_layers above the current 256/4 improves joint F1.

**Source:** Prior Finding 4 — shared backbone is only 10% of model params.
**Test:** P4.2 — sweep shared_layer_size ∈ {128, 256, 384, 512}, num_shared_layers ∈ {2, 4, 6, 8}.
**Phase:** P4.
**Status:** `pending`.
**Notes:** Potentially a nice secondary finding for the paper.

---

### C17 — Batch size and learning rate choices are robust

**Statement:** Champion config performs within ±1% joint across batch ∈ {2048, 4096, 8192} and LR ∈ {5e-5, 1e-4, 2e-4}.

**Source:** Critical review — batch-size confound needed direct testing.
**Test:** P4.3 — grid over batch × LR.
**Phase:** P4.
**Status:** `pending`.
**Notes:** Directly resolves the T0.2 concern.

---

### C18 — Results are reproducible across seeds

**Statement:** Std across seeds {42, 123, 2024} is < 0.01 on joint F1.

**Source:** Prior Finding 10 (small std across seeds on DGI).
**Test:** P5.1 — champion at 3 seeds on AL (+ AZ). **Early answer delivered in F2 (2026-04-17)** via the P1 champion-candidate multi-seed drain.
**Phase:** P5 (early-resolved).
**Status:** `confirmed` — passes on joint@J for all 4 candidates tested; cgc22×eq borderline.
**Evidence (F2, 2026-04-17, 3 seeds per cell):**

| State | Config | joint@J std | joint@T std | Passes (<0.01)? |
|-------|--------|------------:|------------:|:---------------:|
| AL | mmoe4 × gradnorm | **0.0008** | 0.0022 | ✅ easily |
| AL | cgc22 × equal_weight | **0.0087** | 0.0026 | ✅ borderline on joint@J |
| AZ | cgc21 × uncertainty_weighting | 0.0049 | 0.0033 | ✅ |
| AZ | cgc21 × dwa | 0.0029 | 0.0040 | ✅ |

**Notes:** joint@T std is uniformly lower than joint@J std — per-task-best selection is more reproducible. Big seed variance for cgc22×eq joint@J is driven by seed-2024 outlier (0.4204). This suggests the joint-peak checkpoint landing for cgc22×eq is more sensitive to random init than for other configs. **P2 champion recommendation weighted to stability, not maximum mean:** pick mmoe4×gradnorm on AL.

---

## Tier E — Mechanistic / diagnostic claims (for paper figures)

### C19 — Scale imbalance causes source-level gradient conflict

**Statement:** On fusion, gradients w.r.t. the Sphere2Vec half of the input vector are systematically smaller than those w.r.t. the HGI half, and the cosine similarity between the two is lower than between task-level gradients.

**Source:** Mechanistic claim supporting C02.
**Test:** P5.2 — instrument training to log per-source gradient norms and cosines; run 1 fold; extract curves.
**Phase:** P5.
**Status:** `pending`.
**Notes:** Supporting figure, not headline result.

---

### C20 — Fusion has lower between-task gradient cosine than single-source

**Statement:** Gradient cosine between the two task losses is lower (more conflict) on fusion than on HGI-only.

**Source:** Hypothesis for why equal_weight fails on fusion specifically.
**Test:** P5.3 — log gradient cosine during training on fusion vs HGI-only.
**Phase:** P5.
**Status:** `pending`.

---

### C21 — MTL does not impose a wall-clock penalty over combined single-task training (with modern config)

**Statement:** CBIC reported MTL = 4× wall-clock of cumulative single-task; our champion should show this ratio is smaller.

**Source:** Critical review — close the loop on CBIC's compute concern.
**Test:** P2.6 — log wall-clock for single-task-cat, single-task-next, MTL; compare ratios.
**Phase:** P2 (instrumentation); rollup analysis in P6.
**Status:** `pending`.
**Notes:** Sharpened and analyzed alongside C22/C23 in P6. C21 captures the *raw* single-run wall-clock; C23 is the aggregate ratio claim.

---

## Tier G — Canonical MTL benefit claims revisited (Phase 6)

These test the classic Caruana (1997) / Ruder (2017) / Crawshaw (2020) MTL mechanisms against our modern config (DSelectK + fusion + gradient surgery). The CBIC 2025 paper tested several of these with a worse config and reported MTL ≤ single-task — P6 re-tests whether the new configuration flips those findings.

### C22 — MTL reaches target F1 in fewer epochs than cumulative single-task

**Statement:** For both tasks, MTL crosses a predefined target F1 threshold in ≤ the total epochs needed by the two single-task models combined.

**Source:** Caruana 1997 §2.1 (statistical data amplification → faster convergence). CBIC 2025 Table 3 reported ~3.2–3.8 epochs for all three models, so the per-epoch count was similar, but MTL's wall-clock ballooned. We re-test epoch efficiency with modern config.
**Test:** P6.1 — from per-epoch val F1 logs (already stored in `MetricStore` under each run dir), compute epochs-to-target for MTL vs each single-task. Target = 90% of each model's own best epoch score (intrinsic target, not fixed across models).
**Phase:** P6.
**Status:** `pending`.

---

### C23 — Modern MTL total wall-clock ≤ 2× cumulative single-task

**Statement:** `wall_MTL / (wall_single_cat + wall_single_next)` is less than 2 on the champion config.

**Source:** Sharpening C21. CBIC reported this ratio at ~4; modern gradient-surgery / expert-gating might cut it.
**Test:** P6.2 — post-hoc analysis of P2 single-task runs + champion MTL run on AL. Report the ratio at 5f × 50ep.
**Phase:** P6.
**Status:** `pending`.
**Notes:** If ratio > 2 we should still report honestly — negative findings are fine here.

---

### C24 — MTL shows smaller train-val generalization gap than single-task

**Statement:** Final-epoch `F1_train − F1_val` is lower for MTL's category head than for single-task-cat, and likewise for MTL's next head vs single-task-next.

**Source:** Caruana 1997 §2.5 (implicit bias / regularization); Ruder 2017 §3.5.
**Test:** P6.3 — pull train and val F1 trajectories from the existing `MetricStore` logs for P2 runs (no new training needed). Mean train-val gap over 5 folds per model.
**Phase:** P6.
**Status:** `pending`.
**Notes:** Free from existing logs; reviewer-expected result.

---

### C25 — MTL degrades more gracefully than single-task under input noise

**Statement:** When Gaussian noise σ ∈ {0.05, 0.1, 0.2} is added to the fused embedding at inference time, MTL's F1 drop is smaller than single-task's.

**Source:** Caruana 1997 §2.3 (eavesdropping / attention focusing); Ruder 2017 §3.2.
**Test:** P6.4 — re-evaluate (inference-only) best MTL and best single-task checkpoints on AL val folds with noise injection on the fusion input. Plot F1 vs σ.
**Phase:** P6.
**Status:** `pending`.
**Notes:** Inference-only — no retraining. ~1 h compute.

---

### C26 — MTL's advantage grows with less training data (sample efficiency)

**Statement:** The joint F1 gap between MTL and single-task is larger at 25% training data than at 100%.

**Source:** Caruana 1997 §2.1 — *the* empirical claim Caruana makes in his road-following and pneumonia experiments. The defining MTL mechanism.
**Test:** P6.5 — subsample AL training folds to {25%, 50%, 75%, 100%} (stratified, fixed seed). Train MTL + 2 single-task baselines at each fraction, 5 folds each. ~60 new runs.
**Phase:** P6.
**Status:** `deferred_to_journal` — ~60 runs is borderline for BRACIS timeline. Include as future work unless P6 finishes ahead of schedule. If attempted: restrict to AL + 3 fractions {25%, 50%, 100%} to cut compute to ~45 runs.

---

### C27 — MTL backbone transfers better than single-task backbone

**Statement:** Freezing the trained shared backbone and training a fresh linear head on a held-out-task or held-out-state gives higher F1 when the backbone came from MTL than from single-task training.

**Source:** Caruana 1997 §2.4 (representation bias); Ruder 2017 §3.4 (eavesdropping / shared-feature transfer).
**Test:** P6.6 — two variants:
  a) **Cross-task probe:** freeze MTL backbone from AL; train linear head for next-task only. Compare to single-task-next backbone with linear head.
  b) **Cross-state probe:** backbone trained on AL, linear head trained on AZ (and vice versa).
**Phase:** P6.
**Status:** `pending`.
**Notes:** Clean narrative for the paper's "MTL learns transferable representations" paragraph. ~2 h to script.

---

### C28 — No negative transfer: per-task MTL F1 ≥ best single-task F1

**Statement:** For *each* task individually (category and next), MTL's per-task F1 is not worse than the best single-task baseline at the 95% confidence level (Wilcoxon signed-rank paired across 5 folds).

**Source:** Crawshaw 2020 §1 (negative-transfer critique); Zhang & Yang 2022 "Survey on Negative Transfer" (IEEE/CAA JAS). **Reviewers 2024–2026 explicitly expect this test for any MTL paper.**
**Test:** P6.7 — paired-fold test on P2 single-task data vs champion MTL per-task F1. One row per fold, Wilcoxon signed-rank + Cohen's d.
**Phase:** P6.
**Status:** `pending`.
**Notes:** **Highest reviewer-priority gap in the current plan.** C06 tests MTL ≥ single-task on *joint* score; C28 tests the stronger per-task claim (neither head is sacrificed). Free from existing P2 logs — mandatory.

---

## Tier H — Audit findings from P0 leakage review (2026-04-15)

> Appended via append-only discipline — existing C01–C28 untouched.
> These claims arose from the in-study HGI leakage audit; they refine
> evaluation methodology rather than propose new model mechanisms.
> Companion docs: `docs/studies/fusion/issues/HGI_LEAKAGE_AUDIT.md` (technical),
> `docs/studies/fusion/issues/HGI_LEAKAGE_EXPLAINED.md` (glossary),
> `docs/studies/results/P0/leakage_ablation/`.

### C29 — Category F1 on OSM-Gowalla data primarily measures fclass-identity preservation, not learned representation quality

**Statement:** In every available Gowalla state (Alabama, Arizona,
California, Florida, Georgia, Texas), each OSM `fclass` maps to a unique
coarse `category` (`fclass → category` purity = 1.0, macro and
size-weighted, across ≥ 11k POIs per state). POI2Vec embeds at fclass
level, so every POI's HGI input feature is a deterministic function of
its fclass. Consequently, Category F1 on these datasets primarily measures
how faithfully a 64-dim embedding preserves fclass identity, and
spatial-structure-only signal contributes near-zero category-discriminative
capacity.

**Source:** In-study discovery, 2026-04-15 HGI leakage audit follow-up.
**Test:** `experiments/hgi_leakage_ablation.py` arm `C_fclass_shuffle` —
permute encoded fclass column across POIs (category intact, matched
`shuffle_fclass_seed` in Phase 3a and Phase 4), retrain POI2Vec + HGI +
MTL (1 fold, seed 42, DSelectK + aligned_mtl, HGI-only).
**Phase:** P0 (follow-up to leakage audit).
**Status:** `confirmed` cross-state on Alabama and Florida with paired
baselines (1 fold, seed 42):
- Alabama: Category macro F1 0.7855 → 0.1437 (Δ = −64.19 p.p.), Next-POI
  0.2383 → 0.1988 (Δ = −3.95 p.p.).
- Florida: Category macro F1 0.7649 → 0.1506 (Δ = −61.43 p.p.), Next-POI
  0.3627 → 0.2982 (Δ = −6.46 p.p.).

On both states Category lands at the 1/7 ≈ 0.143 random-chance floor
while Next-POI drops an order of magnitude less. Cross-state fclass→category
purity = 1.0 already confirmed on all six available Gowalla states
(`docs/studies/results/P0/leakage_ablation/fclass_purity.json`).

**Evidence:** `docs/studies/results/P0/leakage_ablation/alabama/C_fclass_shuffle/`
(embeddings, fclass .pt, MTL log, full results); `HGI_LEAKAGE_AUDIT.md` §7b, §8, §9.

**Implications for the paper:**
1. **Next-POI F1 becomes the primary representation-quality metric.**
   Arm C dropped Next-POI by only −3.95 p.p., confirming it was not
   riding on the fclass shortcut.
2. **Category F1 is a sanity check** on embedding fidelity. Cross-engine
   Category F1 comparisons (HGI vs DGI vs Fusion) should be framed as
   *"fclass-identity preservation in 64-dim"*.
3. **Joint F1 inherits the shortcut partially** (it weights Category
   alongside Next-POI). Options: (a) report the two metrics separately
   and avoid Joint F1 as primary evidence, (b) down-weight Category,
   (c) substitute Joint with a metric that doesn't inherit the shortcut.
4. **Evaluation section must include a caveat paragraph** stating (i)
   fclass→category determinism, (ii) arm C result, (iii) metric
   re-framing. Pre-empts the first objection any OSM-literate reviewer
   will raise.

### C31 — Fclass shortcut on fusion (does Sphere2Vec break the C29 shortcut?)

**Statement:** Category F1 on OSM-Gowalla fusion (HGI ⊕ Sphere2Vec, 128-dim) is
primarily dominated by the HGI half's fclass-identity preservation (per C29),
and Sphere2Vec does *not* substantially reduce the fclass-shortcut contribution
to category F1. Specifically: permuting fclass across POIs in the HGI half of
the fusion input should still produce a category F1 drop ≥ 30 p.p., similar
in magnitude to the HGI-only result (−64.2 p.p.), though possibly partially
mitigated by Sphere2Vec's spatial signal.

**Source:** Added 2026-04-17 (`issues/P1_METHODOLOGY_FLAWS.md` F3). Follow-up
to C29/N03.
**Test (primary, pending):** Arm-C `fclass_shuffle` variant of
`experiments/hgi_leakage_ablation.py` extended to fusion (shuffle HGI half's
fclass column; keep Sphere2Vec intact); retrain 1 fold on AL fusion at seed 42
with the P1c champion config.
**Test (cheap proxy, completed 2026-04-17):** Linear-probe (5-fold
StratifiedKFold, LogisticRegression C=1.0) on the **raw fusion category input**
(11,848 POIs × 128-dim, preset `space_hgi_time`: [Sphere2Vec(64), HGI(64)]).
Script: `/tmp/linear_probe_fusion.py`. Output:
`docs/studies/fusion/results/P1/linear_probe_fusion_AL.json`.
**Phase:** Blocking-issue for P1→P2 transition. Primary test still scheduled;
proxy is strong evidence.
**Status:** `partial` — proxy strongly supports the claim; full arm-C retrain
pending to quantify at the MTL level.

**Evidence (linear probe, AL fusion, 2026-04-17):**
| Probe input | # features | Category macro-F1 | Notes |
|-------------|-----------|-------------------|-------|
| Sphere2Vec half (cols 0-63) | 64 | **0.1108 ± 0.0008** | Below chance (1/7 = 0.143); carries essentially zero category signal. |
| HGI half (cols 64-127) | 64 | **0.6883 ± 0.0093** | 88% of the full-MTL ceiling (0.786 from C29 baseline). |
| Full fusion | 128 | **0.6815 ± 0.0106** | *Lower* than HGI alone — Sphere2Vec degrades a linear probe. |
| C29 reference (HGI-only, arm-C fclass shuffle) | 64 | 0.1437 | At shuffle, even MTL falls to chance. |

**Interpretation:**
- **Sphere2Vec carries ~zero category signal** at the embedding level. The
  representation was trained for spatial proximity, not categorization.
- **The HGI half of fusion carries 88% of the MTL's category capacity** via a
  single linear layer — strong evidence the fclass shortcut is fully inherited
  by fusion's HGI component.
- **Adding Sphere2Vec does not mitigate** the shortcut. A linear probe is
  slightly *degraded* by including the Sphere2Vec half. Any category-F1 gain
  from fusion vs HGI-only in downstream MTL must come from backbone
  nonlinearity, not from intrinsic fusion signal.
- **The stronger arm-C retrain (deferred 2026-04-18):** would regenerate HGI
  with shuffled fclass, rebuild fusion from (shuffled HGI + baseline
  Sphere2Vec), retrain MTL, and compare cat F1 to baseline. Expected result:
  MTL cat F1 drops to ~0.15 (chance), matching C29's HGI-only result within
  fold noise. **Deferred because:** the linear-probe evidence is decisive at
  the representation level (HGI-half alone retrieves 88% of MTL ceiling; a
  linear layer on Sphere2Vec alone retrieves nothing). Full retrain is a
  quantitative confirmation, not qualitative. Promoted from "blocking for P2"
  to "journal extension / robustness check". Status kept at `partial` until
  the full retrain lands; no decision in P2 relies on upgrading C31 to
  `confirmed`.

**Paper implications (preliminary — confirm with full arm-C):**
1. **Category F1 on fusion is not a representation-quality metric.** It is
   "fclass-identity preservation + slight backbone reshaping." N03 applies
   with equal force on fusion as on HGI-only.
2. **Next F1 remains the primary representation-quality metric** on fusion.
   The "fusion > HGI" story (C01) must lead with next F1 improvements.
3. **Joint F1 inherits the shortcut proportionally.** Under harmonic mean,
   joint F1 is dominated by the smaller value (next), which limits how much
   the shortcut can inflate joint. But any AL P1c joint comparison where
   cat F1 is ~0.82 is ~50% a measurement of fclass-preservation fidelity.

**Implications:**
- If confirmed: any cross-engine (HGI vs DGI vs Fusion) category F1 comparison
  in P3 must be framed as "fclass-preservation fidelity in X-dim," not
  "representation quality." **Next F1 remains the primary representation-quality
  metric** on fusion, as it already is on HGI (C29).
- If refuted (Sphere2Vec *does* break the shortcut partly): report the mitigation
  as a novel finding. Category F1 on fusion becomes a valid representation
  metric in a way it is not on HGI-only.
- Either outcome tightens N03 (which claims the shortcut *inherits* partially
  into joint) with quantified evidence.

---

### C32 — Joint-peak checkpoint selection biases config rankings on fusion MTL

**Statement:** On fusion MTL with OneCycleLR over 50 epochs, category peaks in
the second half of training (epochs 17–45 across AL P1c cells) while next peaks
in the first half (epochs 10–22). Reporting a single "joint best" checkpoint
collapses this asymmetry into an epoch that is past next's peak and before
category's peak, producing a harmonic-mean joint F1 that is a **tradeoff
artifact** rather than a property of the configuration. Consequently, config
rankings computed at joint-peak selection differ from those computed at
per-task-best selection.

**Statement, concrete form:** For AL P1c top-5, the ranking under joint-peak vs.
per-task-best selection is not rank-correlated (ρ < 0.5). The "AL winner"
`mmoe4 × gradnorm` (joint-peak) is mid-pack under per-task-best; the "AL loser"
`cgc22 × equal_weight` (joint-peak 5th) is top under per-task-best.

**Source:** In-study discovery, 2026-04-17, analyzing
`diagnostic_task_best` logs from P1c.
**Test:** Completed already. See `docs/studies/fusion/results/P1/SUMMARY.md`
§Per-task-best reanalysis and `issues/P1_METHODOLOGY_FLAWS.md` F1 for the table.
**Phase:** P1 (discovered), but applies to every MTL phase going forward.
**Status:** `confirmed` on AL P1c data; also demonstrated on AZ P1c (smaller
but same-direction effect).

**Evidence (AL P1c, 5f × 50ep, seed 42):**

| cell | joint@J | joint@T | Δ |
|------|---------|---------|----|
| cgc21 × bayesagg_mtl | 0.4034 | 0.4215 | +0.0181 |
| cgc22 × equal_weight | 0.4031 | **0.4229** | +0.0198 |
| cgc22 × excess_mtl | 0.4043 | 0.4215 | +0.0173 |
| cgc22 × nash_mtl | 0.4034 | 0.4217 | +0.0183 |
| mmoe4 × gradnorm | **0.4082** | 0.4220 | +0.0138 |

Joint@J winner: mmoe4 × gradnorm. Joint@T winner: cgc22 × equal_weight.

**Implications:**
1. P2 C06 (MTL vs single-task) comparisons must report **both** joint@J and
   joint@T, or MTL will be artificially disadvantaged on next F1 relative to
   a single-task-next baseline that naturally uses next-peak selection.
2. Ratifies C02 as `partially_refuted`: per-task-best selection shows
   equal_weight matches or beats grad-surgery on AL.
3. Supports C07 (asymmetric MTL benefit per task) mechanistically — the
   asymmetry is temporal (next peaks first, category peaks later).

**Resolution / paper methodology:**
- Report both checkpoint policies throughout P2–P6.
- Add `joint_f1_taskbest` to `_extract_observed` in `archive_result.py` so
  state.json carries both values going forward (P1 can be back-filled; data
  is already in each test's `full_summary.json`).
- Separate "scientific comparison" (per-task-best) from "deployment champion"
  (joint-peak) in paper text.

---

### C30 — No classical label leakage in HGI / POI2Vec training

**Statement:** No validation-set `category` labels flow into HGI or
POI2Vec training through any code path. The concerns originally raised
in `docs/issues/DATA_LEAKAGE_ANALYSIS.md` (written pre-refactor) about
transductive label leakage in HGI are not supported by the current code
+ ablation evidence.

**Source:** In-study discovery, 2026-04-15 — line-by-line code audit
+ arms A / B / A+B of `hgi_leakage_ablation.py`.
**Test:** `HGI_LEAKAGE_AUDIT.md` §2–§5 (cleared code paths inventory);
arms A and B null/asymmetric results.
**Phase:** P0.
**Status:** `confirmed`.

**Evidence:**
- HGI loss (`HGIModule.py:259-300`) is purely contrastive
  POI↔Region + Region↔City — no classification head, no CE against
  category, no read of `data.y` during training (grep-verified across
  `research/embeddings/hgi/**/*.py`).
- The only explicit `category` → embedding path (POI2Vec hierarchy L2
  loss, `poi2vec.py:162-174`) is cosmetic at `le_lambda = 1e-8`:
  arm A null result (Δ Category F1 = +1.36 p.p., direction wrong for
  leakage).
- Hard-negative sampling (`HGIModule.py:125-200`) uses per-region
  **fclass** distributions (public OSM), not category labels. Arm B
  shows asymmetric per-task trade-off (−1.3 Cat / +2.4 Next), not
  one-sided leakage signature.
- Best-epoch selection is training-loss only (`hgi.py:168-177`), no
  validation metric.
- `data.y` is never read during training; only used post-hoc to decorate
  the output parquet (`hgi.py:140, 186-188`).

**Caveat:** Transductive graph training *is* present — validation POIs
are nodes in the Delaunay graph seen during HGI's self-supervised
training. Standard GNN-benchmark practice, mild effect, should be
declared in the paper's methodology section as a caveat, not as a bug.

**Does NOT apply to Check2HGI:**
`research/embeddings/check2hgi/preprocess.py:217-219` concatenates
category one-hot directly into check-in node features. That *is*
classical label leakage and must be removed before Check2HGI is ever
activated. Tracked separately.

---

## Tier F — Refutations / things we don't claim

### N01 — We do NOT claim our framework is universally state-of-the-art

**Statement:** We claim it's SOTA on Gowalla state-split POI prediction with the 7-class taxonomy (Alabama/Arizona/Florida), not universally.

**Source:** Critical review. HAVANA (Santos et al.) uses the **same Gowalla** dataset with FL/CA/TX state splits — the non-1:1-ness of comparisons stems from **preprocessing and task formulation** (HAVANA does semantic venue annotation over a spatial/spectral graph; we do trajectory-based MTL with a 7-category taxonomy), not from different datasets. Numbers are therefore directionally comparable but not strictly head-to-head.

**Future-work note:** other POI benchmarks worth adding to strengthen external validity (journal extension, not BRACIS scope): **Foursquare NYC / TKY** (standard alternative trajectory benchmark, different user base, different taxonomy granularity), **Brightkite** (smaller Gowalla-era LBSN). Both would require new embedding pipelines and category harmonization — explicitly out of scope for the current study.

---

### N02 — We do NOT claim gradient surgery is required (as of T0.2 evidence)

**Statement:** The current best evidence (T0.2, pre-bug) suggests gradient surgery accelerates convergence but doesn't raise the ceiling at matched training budget. Will be re-tested in C02 on new data.

---

### N03 — We do NOT claim Category F1 measures learned spatial/semantic representation quality on OSM-Gowalla data

**Statement:** Any paper claim of the form "HGI/DGI/Fusion learns better
*representations* because Category F1 improves by X p.p." is not supported
on this dataset. Category F1 primarily reports how well a 64-dim embedding
preserves OSM fclass identity, because `fclass → category` is deterministic
in all Gowalla states we evaluated (purity = 1.0; see C29).

**Source:** Added 2026-04-15 in response to C29 / arm C result.
**Scope:** Applies to any engine that embeds at fclass level or
concatenates fclass-derived features. DGI, Check2HGI, and any fusion that
includes them all inherit the property.

**What we *do* claim instead:** Next-POI F1 is our representation-quality
metric. Category F1 is a retained sanity check on embedding fidelity
("does the 64-dim vector preserve the sub-type category enough to recover
it?"), reported separately with a caveat paragraph.

---

### N04 — Our next-F1 numbers are not directly comparable to CBIC 2025 / HAVANA

**Status:** `provisional` — the 1–3pp protocol-delta claim is rationalized
from a single CBIC-sanity run (AL/DGI, next_F1 = 0.243 vs CBIC's 0.26–0.28),
not from a matched-protocol experiment. Pending verification via
`P5_protocol_delta`.

**Statement:** CBIC 2025 and HAVANA report POI-prediction F1 under
`StratifiedKFold` (record-level splits — a user's check-ins can appear in
both train and val folds). Our pipeline uses `StratifiedGroupKFold(groups=userid)`
(user-isolated splits), which is strictly harder: validation users are fully
held out, so the model cannot exploit memorized trajectories. Absolute-number
comparisons are therefore not head-to-head; we conjecture 1–3pp lower on
next-F1 under otherwise-identical configs, pending direct measurement.

**Source:** In-study discovery, 2026-04-15, from P0.4 CBIC sanity
(next_F1 = 0.243 vs CBIC's reported 0.26–0.28) on matched
(AL, DGI, nash_mtl, 5f × 50ep, seed 42) configuration.

**Scope:** Applies to every AL/DGI, AL/HGI, AL/fusion, AZ/HGI, AZ/fusion,
FL/HGI, FL/fusion comparison in P1–P6 (all `placeid`-carrying parquets use
user-isolation). AZ/DGI and FL/DGI currently fall back to `StratifiedKFold`
(pre-bugfix parquets missing `placeid`) — inconsistent with the rest of the
study; tracked under open issue `az_fl_dgi_stale`.

**What we *do* claim:** Relative rankings (arch-A vs arch-B on the same
state × engine × fold-set) are fully valid. Absolute comparisons to
CBIC / HAVANA require an explicit caveat paragraph stating the stricter
split protocol.

**Paper implication:** Reviewers will ask why our numbers differ from CBIC
2025 on "the same" dataset. One-sentence answer: user-isolated splits.
Do **not** frame results as "outperforming CBIC" without qualifying on the
split protocol.

---

## How to use this file

- Each phase doc (`phases/Pk_*.md`) lists the **C-IDs** it tests.
- The coordinator reads this file to know which tests must run.
- After a test, update the `status` and `evidence` fields here.
- New hypotheses arising from surprising results get appended (C22, C23, …) with source "in-study discovery".

**Version:** this file is versioned in git. Don't delete entries; mark them `abandoned` with a reason.
