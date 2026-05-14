# MTL Ablation Protocol — 4 candidate fixes × paper-grade insights

**This is NOT a contingency plan — it's a paper contribution.** Running this ablation is worth doing regardless of whether CH01 succeeds or fails on FL, because every outcome produces a publishable finding:

- **If CH01 succeeds on FL with vanilla MTL**: ablation becomes a "what further improves?" section. Shows technique X adds +Y pp on top of vanilla MTL. Strengthens the paper's architectural contribution.
- **If CH01 fails on FL**: ablation is the paper's headline. Identifies the mechanism + the fix. Backbone-dilution framing with a solution.
- **If some techniques improve and others don't**: ablation characterises the pathology via which interventions work vs. don't. The ablation table *is* the finding.

**Goal:** determine which of 4 architectural candidates + 1 optimizer sanity check produces the best Δm on AL; promote the winner to FL/CA/TX at 5f × 50ep × 3 seeds for headline numbers.

**Budget:** ~10 h AL compute if we run all 5 techniques. Cheapest-first early-exit saves 4–6 h if a winner emerges early.

**Paper framing this enables (regardless of CH01 outcome):**

> *"Beyond vanilla shared-backbone MTL, we ablate four architectural interventions — asymmetric gradient transfer, curriculum warmup, per-task low-rank adapters (MTLoRA), and learned sparse routing (AdaShare) — to characterise when multi-task training improves over single-task on small-data POI prediction. We find that [winner] closes the dilution gap by [X pp], while [loser] does not, suggesting [mechanism]."*

Every technique is tied to a specific hypothesis about the MTL-on-POI pathology; every outcome — positive, negative, partial — maps to a paper-grade claim.

---

## 0. Fixed experimental setup (every technique below holds these constant)

| Variable | Value | Rationale |
|----------|-------|-----------|
| Dataset | Alabama (small, fast) | Stress test; FL/CA/TX replicate winners |
| Folds | 5 | Noise characterization |
| Epochs | 50 | Matched compute with STL fair baselines |
| Seed | 42 | Reproducible; can add 2 more seeds at end for champion only |
| Input modality | per-task (check-in→cat, region→reg) | CH03-winning design |
| Arch | `mtlnet_dselectk` | P2-screen champion |
| Base optimizer | `pcgrad` | Best for dselectk per P2-screen |
| Task-B head | `next_gru` | P1 champion |
| Folds protocol | Fair (`StratifiedGroupKFold`) | C11 closed |

**Common baseline (to beat):**
- P2-validate result: cat F1 36.08 ± 1.96, reg Acc@10 48.88 ± 6.26, **Δm = −14.12%**

**Target:** Δm ≥ −2% AND r_A, r_B both ≥ −0.5% (Pareto-near-tie = success).

---

## 1. RLW — sanity check (cheapest, run first)

**Hypothesis (H-RLW):** if Random Loss Weighting matches PCGrad's Δm, the gradient-manipulation family is saturated on our setup; no further optimizer sweeps justify compute.

**Protocol:**
- Single config: mtlnet_dselectk + `random_weight` + GRU head, AL 5f × 50ep, per-task modality.

**Success / learn criteria:**
- |RLW Δm − PCGrad Δm| ≤ 2% → optimizer family saturated. Move to architectural fixes (sections 2–5). **Paper insight:** "5 diverse optimizer families converge within noise on this task — the bottleneck is not gradient balancing."
- RLW Δm > PCGrad Δm by ≥ 2% → we've been underestimating the optimizer axis. Re-screen with RLW-adjacent variants.
- RLW Δm < PCGrad Δm by ≥ 2% → PCGrad's projection is doing useful work; further architectural fixes should use PCGrad as base.

**Compute:** ~30 min.

**Primary metric:** Δm delta vs PCGrad baseline. **Secondary:** per-task loss trajectories (plot; does RLW's random weighting cause fold-to-fold variance to explode?).

---

## 2. Gradient-scaling auxiliary — asymmetric transfer (cheapest architectural)

**Hypothesis (H-ASYM):** the strong task (region) should not be forced to share capacity symmetrically with the weak task (category). Scaling category's gradient through the shared backbone by factor λ < 1 should:
- (a) preserve region's capacity in the backbone (less pollution by category gradients)
- (b) still provide category regularisation benefit (the category head trains normally; only its gradient into the shared stack is attenuated)
- (c) allow us to find the λ* that maximises Δm by sweeping

**Protocol:**
- 5 configs: λ ∈ {0.0, 0.2, 0.5, 1.0, 2.0}
- λ=0.0 → category trains only its head (backbone sees only region gradient) — essentially STL region + dead cat head
- λ=1.0 → vanilla MTL (reproduces P2-validate)
- λ=2.0 → reverse asymmetry (category-dominant); should be worse if region is the stronger task
- Implementation: hook on the shared-backbone's backward pass — multiply `shared_next.grad_from_cat_loss * λ`. ~20 LOC as a gradient-filter wrapper.

**Success / learn criteria:**
- **Monotone in λ** (higher λ → worse Δm): confirms capacity dilution is uniform; category always dilutes when sharing. Paper insight: "Shared-backbone MTL has an implicit capacity tax; our data shows attenuating the weak task's backbone gradient strictly improves bidirectional Δm."
- **Bump at interior λ\*** (e.g., λ=0.2 best): the asymmetric optimum exists. Paper insight: "There is a sweet-spot of auxiliary influence; pure multi-task or pure single-task both underperform asymmetric MTL."
- **λ=1.0 best** (no improvement from scaling): asymmetric transfer hypothesis is wrong; look elsewhere. Confirms the pathology is purely representational, not gradient-flow-magnitude.

**Compute:** 5 × 30 min = 2.5 h.

**Analyses beyond Δm:**
- **Per-fold loss ratios** — at each λ, what fraction of shared-backbone parameter updates come from category vs region? Plot vs λ, look for the point where region's gradient dominates.
- **Head-level learning curves** — does category head's own loss curve change under different λ? (If scaling the backbone gradient doesn't affect category head training, ΔF1 on category should be near-zero across λ values.)
- **Representation similarity** — CKA or cosine similarity between shared_next(region) and shared_cat(category) at each λ. Lower similarity under lower λ = backbone is specialising, which is the point.

---

## 3. Asymmetric warmup — curriculum (orthogonal to #2)

**Hypothesis (H-WARM):** if we train region alone for the first N epochs, the shared backbone specialises on region's task. Then adding category as auxiliary (with λ=1 or lower) lets the backbone transfer useful region-specific features to category without pre-committing to a symmetric compromise.

**Protocol:**
- 4 configs: warmup ∈ {0, 10, 20, 40} epochs (of 50 total).
- warmup=0 → vanilla MTL (reproduces P2-validate).
- warmup=50 → pure region single-task (no MTL effect).
- During warmup: category task weight = 0; region weight = 1. Post-warmup: equal weight (or whatever pcgrad dictates).
- Implementation: conditional loss masking in `_train_epoch`. ~10 LOC.

**Success / learn criteria:**
- **warmup=20 or 40 gives best Δm**: curriculum works; mid-training addition of category is the right asymmetry. Paper insight: "On small-data 2-task MTL, curriculum-style auxiliary introduction improves Δm by X pp over simultaneous training."
- **warmup=0 best**: curriculum doesn't help; rules out a training-schedule fix. Informs: "joint training from epoch 1 is already near-optimal; shared-backbone pathology is not a warm-up artefact."
- **Monotone improvement with warmup**: the more region-only epochs, the better the final MTL; asymptotically MTL collapses to STL. Not a publishable win but a clear characterization.

**Compute:** 4 × 30 min = 2 h.

**Analyses beyond Δm:**
- **Where does category recover?** If category F1 at epoch 40 with warmup=20 is near STL, the backbone *can* host both tasks — just needs sequential specialization. Plot category F1 vs epoch under each warmup setting.
- **Region catastrophic forgetting** — does region Acc@10 drop after category is added? If yes, we see the classical MTL interference pattern.

---

## 4. MTLoRA — per-task low-rank capacity (highest-expected-lift, ~1 day to implement)

**Hypothesis (H-LORA):** adding small task-specific LoRA adapters around each shared residual block gives each task dedicated low-rank capacity. This should fully close the dilution gap because region gets its standalone ceiling via the combination (shared backbone + region adapter) while category gets (shared backbone + category adapter).

**Protocol:**
- 4 configs: rank ∈ {0, 4, 8, 16}.
- rank=0 → vanilla MTL (reproduces P2-validate).
- rank=4 → minimal per-task capacity (for 256-dim backbone: 2×256×4 = 2048 params per task per block).
- rank=8, 16 → intermediate to larger dedicated capacity.
- Implementation: wrap each `ResidualBlock` in a class that adds two LoRA branches (one per task); task id selects which branch applies to that task's forward pass. Following MTLoRA (CVPR 2024) reference implementation. ~300 LOC including tests.

**Success / learn criteria:**
- **rank=4 closes gap**: minimal dedicated capacity is sufficient. Paper insight: "2K params/task recovers the standalone ceiling; the shared backbone was only 2-3% under-parameterised per task."
- **only rank=16 works**: dilution is severe; requires significant per-task capacity. Tradeoff analysis: at what rank does total param count equal 2× STL (making MTL roughly equivalent to training two models)?
- **no rank helps**: LoRA's low-rank assumption is wrong for our case; the required per-task capacity is higher-rank. Try full rank adapters (= per-task towers) next.

**Compute:** 4 × 30 min = 2 h training + 1 day implementation.

**Analyses beyond Δm:**
- **LoRA weight magnitudes per task per block** — which blocks benefit most from per-task capacity? Possibly early blocks (feature extraction) need less, late blocks (task decoding) need more.
- **Parameter efficiency** — plot Δm vs rank. If Δm saturates at low rank, LoRA is an efficient fix.
- **Freeze experiments** — train LoRA-free MTL first, then add+fine-tune only LoRA. If this 2-stage approach matches joint training, we've found a cheap deployment path.

---

## 5. AdaShare — learned per-task sparse routing (bigger refactor)

**Hypothesis (H-ADASHARE):** each task learns a binary skip-gate over the 4 shared residual blocks. Region learns to skip blocks that over-encode category features; category skips blocks that over-encode region features. The shared backbone becomes a menu of blocks each task selectively uses.

**Protocol:**
- 3 configs: baseline (no gates) + AdaShare with 2 gate-regularisation weights (e.g., budget λ_gate ∈ {0.1, 0.5}).
- Gates are Gumbel-Softmax samples during training, argmax during eval.
- Implementation: add per-task 4-dim gate logits to MTLnet; apply during forward per block. ~200 LOC.

**Success / learn criteria:**
- **AdaShare beats baseline + MTLoRA**: per-task routing is the right mechanism for our pathology. Paper insight: "2-task capacity dilution is best resolved by sparse routing, not shared low-rank adaptation."
- **AdaShare ≈ MTLoRA**: both solutions address dilution; either works. Pick the cheaper one for the paper's headline.
- **AdaShare < baseline**: routing overhead (extra params, Gumbel sampling noise) dominates the benefit. Informs: "On small data, even a mild architectural complication hurts."

**Compute:** 3 × 30 min = 1.5 h training + 0.5-1 day implementation.

**Analyses beyond Δm:**
- **Learned routing patterns** — which blocks does each task actually use? Plot 2×4 heatmap of gate values post-training.
- **Consistency across folds** — does each fold converge to the same routing? High fold-to-fold variance = gates are under-constrained (raise λ_gate); low variance = stable structural fix.

---

## 6. Execution order and early-exit policy

```
Step 1: wait for FL 1f×50ep result.
  If FL SUCCEEDS (Δm ≥ 0 AND both r_A, r_B > 0):
    CH01 holds on headline states. Run FL + CA + TX 5f×50ep 3-seed. STOP contingency plan.
  Else continue:

Step 2: RLW sanity check (30 min).
  Goal: confirm optimizer axis is saturated. If RLW ≈ PCGrad, proceed.

Step 3: Gradient-scaling auxiliary (2.5 h, 5 configs).
  Goal: cheapest architectural intervention; learn whether asymmetric transfer is the axis.
  Early exit: if λ=0.2 achieves Δm ≥ -2%, we have a winner. Run FL + CA + TX with λ=0.2.

Step 4: Asymmetric warmup (2 h, 4 configs). Orthogonal to step 3.
  Goal: learn whether curriculum is the axis.
  Combine best of step 3 + step 4 if both help independently.

Step 5: MTLoRA (1 day implementation + 2 h compute, 4 configs).
  Only runs if steps 2-4 don't close the gap.

Step 6: AdaShare (1 day implementation + 1.5 h compute, 3 configs).
  Only runs if MTLoRA doesn't fully close the gap.

If all 5 steps fail to improve Δm: the paper's thesis is the backbone-dilution finding itself. Write up as:
  "On small-data 2-task POI MTL, a 4-technique ablation shows neither
   asymmetric transfer (gradient scaling, curriculum) nor per-task
   capacity (LoRA, sparse routing) closes the dilution gap. We posit
   this is a fundamental capacity-ceiling property on data regimes where
   a strong standalone head already saturates the extractable signal
   from the task's own input." [arch-ablation table as backbone]
```

---

## 7. What makes this protocol produce insights (not just results)

Each technique is tied to **one specific hypothesis** about the pathology's mechanism:

| Technique | Tests hypothesis |
|-----------|------------------|
| RLW | "Optimizer family is saturated; more optimizers won't help" |
| Gradient-scaling | "Asymmetric transfer can close the gap" |
| Warmup | "Curriculum (schedule, not arch) is the axis" |
| MTLoRA | "Per-task low-rank capacity is the answer" |
| AdaShare | "Per-task sparse routing is the answer" |

Every outcome — positive, negative, or partial — **teaches us something about the pathology**. Even if all 5 techniques fail, we have a **defensible, testable claim** about the limits of MTL on this task pair. That's the difference between "we ran experiments and got a disappointing number" vs "we empirically characterised the capacity-dilution pathology via a 5-technique ablation."

---

## 8. What we commit to measure for every run

Beyond the primary Δm:

1. **r_A, r_B** separately (Pareto components).
2. **Per-task F1 / Acc@10** in absolute terms (readers want to see both).
3. **Per-fold metric** not just mean (noise characterization).
4. **Training wall-time** (practitioners care).
5. **Training loss trajectories** per task (convergence patterns).

If an experiment produces a surprising result (positive or negative), follow up with a **diagnostic ablation** that isolates the cause. Don't report black-box numbers.
