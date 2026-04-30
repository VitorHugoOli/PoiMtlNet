# F50 follow-ups — B2 + F52 + F65 + F53 findings (in progress)

**Status (live, updated 2026-04-30 01:30 UTC):**
- ✅ B2/F64 — done (REJECTED)
- ✅ F52 — done (TIED with B9 = paper-grade architectural finding)
- 🟡 F65 — running on GPU (started 01:29:43)
- 🟡 CA P3 — queued behind F65 + watchdog
- ✅ F53 cw sweep — done (analysis pending below)

Reference: B9 clean champion (`mtlnet_lr1.0e-04_bs2048_ep50_20260429_1813`) — alt-SGD + cosine + alpha-no-WD + per-fold log_T at FL 5f×50ep.

Selection rule: per-fold-best @≥ep5, paired Wilcoxon signed-rank (5 folds, p_min=0.0625 for 5/5 directional).

---

## 1 · B2/F64 — warmup-decay LambdaLR on reg_head: REJECTED ❌

| metric | B2 | B9 ref | Δ | paired Wilcoxon | verdict |
|---|---:|---:|---:|---|---|
| reg top10_acc_indist | 58.28 ± 1.57 | 63.47 ± 0.75 | **−5.19** | p=0.0625, **0/5+ 5/5−** | **rejected** |
| reg mrr_indist | 45.08 ± 7.09 | 53.08 ± 0.54 | **−8.00** | p=0.0625, 0/5+ | rejected |
| reg f1 | 10.55 ± 1.13 | 12.21 ± 0.76 | −1.65 | p=0.0625, 0/5+ | rejected |
| cat f1 | 68.01 ± 0.90 | 68.59 ± 0.79 | −0.58 | p=0.0625, 0/5+ | rejected |
| cat accuracy | 70.88 ± 0.72 | 71.43 ± 0.73 | −0.55 | p=0.0625, 0/5+ | rejected |
| cat f1_weighted | 71.23 ± 0.67 | 71.87 ± 0.72 | −0.65 | p=0.0625, 0/5+ | rejected |

**Interpretation:** the warmup-decay reg_head LR shape (warmup ep 0-5 to peak 10× base, plateau 5-15, linear decay 15-50) **destabilizes** the head rather than unlocking late-window α growth. Strictly Pareto-dominated by B9 — losses on every metric, every fold, both tasks. The F50 T3 §6 hypothesis that "let α grow during the right window without sustained instability" can be achieved via this LR shape is **refuted**.

**Mechanism (consistent with F50 D5):** the reg encoder saturates at ~ep 5-6 under B9 anyway. Pumping reg_head LR 10× during ep 5-15 (when the encoder is already saturated) likely sends the head into instability rather than productive growth. The α growth window in B9 is structurally constrained by the encoder, not by the head's LR.

**Paper implication:** narrows the design space. P4-alone / B9 are the productive recipes; LR-shape interventions on reg_head don't stack.

---

## 2 · F52 — identity-crossattn (P5): TIED with B9 ✅ (paper-grade)

| metric | F52 P5 | B9 ref | Δ | paired Wilcoxon | verdict |
|---|---:|---:|---:|---|---|
| reg top10_acc_indist | 63.77 ± 1.12 | 63.47 ± 0.75 | +0.30 | p=0.8125, 3/5+ | **tied** |
| reg mrr_indist | 53.13 ± 0.53 | 53.08 ± 0.54 | +0.05 | p=0.6250, 3/5+ | tied |
| reg f1 | 12.21 ± 0.79 | 12.21 ± 0.76 | −0.00 | p=0.8125, 2/5+ | tied |
| cat f1 | 68.64 ± 0.91 | 68.59 ± 0.79 | +0.05 | p=0.4375, 4/5+ | tied (slight cat trend) |
| cat accuracy | 71.61 ± 0.87 | 71.43 ± 0.73 | +0.18 | p=0.1250, 4/5+ | trending tied+ |
| cat f1_weighted | 72.04 ± 0.90 | 71.87 ± 0.72 | +0.16 | p=0.1875, 4/5+ | trending tied+ |

**Interpretation:** zeroing the cross-attention mixing output (`a_upd = 0`, `b_upd = 0`) while keeping per-task FFN+LN structure produces results indistinguishable from full cross-attn within noise. **Cross-attn mixing is dead at FL.** The productive component of the shared backbone is the per-task FFN+LN depth, NOT the K/V cross-attention.

**Paper-grade architectural decomposition** (cleanest single-experiment result in F50 T1.5):
- P1 (no_crossattn, removes whole block) — measured earlier; ≈ tied to H3-alt
- **F52 P5 (zeros mixing, keeps FFN+LN) — tied to B9, both within p>0.4**
- → cross-attn mixing contributes nothing distinguishable from baseline within paired noise

Confirms `F50_T1_5_CROSSATTN_ABSORPTION.md` §5.3's prediction: cross-task mixing is absorbed by the per-task encoders before it reaches the shared backbone, leaving only the FFN+LN structure as the productive shared component.

**Paper implication:** the paper can drop the "cross-attn mixing is the productive shared component" framing entirely. The MTL backbone reduces to "two parallel per-task FFN+LN stacks" + the cross-attn mixing is dead weight. Simpler architecture, same numbers.

---

## 3 · F65 — joint-loader min_size_truncate: PENDING

Run started 01:29:43, ETA ~17 min. Tests whether the F50 D5 reg encoder saturation observation is partly driven by the joint-loader cycling pattern. Expected result if cycling is the cause: D5 reg saturation epoch should shift later under min_size_truncate.

---

## 4 · F53 — category_weight sensitivity sweep: PENDING analysis

6 runs done (`f53_h3alt_cw{0.25,0.50,0.75}_fl` + `f53_p1_cw{0.25,0.50,0.75}_fl`). Will run analyzer once F65 + CA P3 land to keep all paired comparisons in one pass.

Predicted patterns (from F50 T1.5):
- **If H3-alt > P1 grows as cw drops** → cross-attn UNLOCKS at low cat-loss weight → mixing is hyperparameter-dormant, not structurally dead.
- **If H3-alt ≈ P1 at all cw** → mixing is structurally dead (consistent with F52 P5 finding above).

Given F52 P5 ≈ B9, prediction strongly biases toward the second pattern.

---

## 5 · CA P3 unblock attempt: QUEUED

Watchdog armed; will fire after F65 finishes. CA = 8501 regions vs FL 4702. Tests:
- (a) whether the Lightning A100 15-GB RAM blocker is resolved on this 503 GB env
- (b) whether the 24 GB GPU 4090 can hold CA's 8501-region head + cross-attn

If OOM, we'll know the real blocker is GPU not RAM. If success, the CH18 cross-state portability claim extends to a 5th state (FL+AL+AZ+GA+CA).

---

## 6 · Cross-references

- `F50_T4_FINAL_SYNTHESIS.md` §1 — B9 champion headline (63.47 ± 0.75 reg / 68.59 ± 0.79 cat)
- `F50_T1_5_CROSSATTN_ABSORPTION.md` §5.2-5.3 — F52 P5 + F53 cw-sweep predictions
- `F50_T3_HYPERPARAM_BRAINSTORM.md` — B2/F64 design source (warmup-decay shape)
- `F50_D5_ENCODER_TRAJECTORY.md` — encoder saturation mechanism
- `C05_P3_NULL_RESULT_FALLBACK.md` — branching plan if CA P3 fails
- Code: `src/training/helpers.py` (`_build_reg_head_warmup_decay_lambda`); `src/models/mtl/mtlnet_crossattn/model.py` (`identity_attn`); `src/utils/progress.py` (`zip_longest_cycle` strategy)
- Run dirs: B2 `_0057`, F52 `_0115`, F65 `_0130` (in progress)
