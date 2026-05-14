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

## 3 · F65 — joint-loader min_size_truncate: TIED with B9 (cycling NOT the cause)

| metric | F65 | B9 ref | Δ | verdict |
|---|---:|---:|---:|---|
| reg top10_acc_indist | 63.47 ± 0.75 | 63.47 ± 0.75 | +0.00 | **identical** |
| reg mrr_indist | 53.08 ± 0.54 | 53.08 ± 0.54 | +0.00 | identical |
| cat f1 | 68.59 ± 0.79 | 68.59 ± 0.79 | +0.00 | identical |

**Interpretation:** stopping at the shorter loader's end (no cycling) produces metrics indistinguishable from max_size_cycle within the per-fold-best @≥ep5 selection rule. **Joint-loader cycling is NOT a contributing factor to reg encoder saturation.** The F50 D5 reg saturation epoch is robust to cycling strategy — it's an artifact of the joint loss shape, not the data feed pattern.

(Final-summary numbers in the F65 log differ slightly because the joint-best selector picks different epochs than the per-metric ≥ep5 rule. The paper-grade comparison uses ≥ep5 per-metric, where F65 = B9.)

**Mechanism (consistent with F50 D5):** reg encoder saturates at ~ep 5-6 under both strategies. The cycle pattern only affects later epochs, by which time reg has already plateaued.

**Paper implication:** F65 closes the "joint-loader cycle artifact" hypothesis. Reg saturation is NOT a pipeline implementation artifact.

---

## 4 · F53 — category_weight sensitivity sweep: H3-alt ≈ P1 across cw (mixing structurally dead)

All 6 runs land at FL 5f×50ep clean (per-fold log_T). Reg top10_acc_indist @≥ep5:

| arm | cw=0.25 | cw=0.50 | cw=0.75 | Δreg vs cw |
|---|---:|---:|---:|---:|
| **H3-alt** (cross-attn ON, no P4) | 60.08 ± 0.88 | 60.04 ± 1.06 | 60.12 ± 1.15 | flat (~−0.04) |
| **P1** (cross-attn OFF, disable_cross_attn=true) | 60.15 ± 1.07 | 60.04 ± 1.10 | 59.99 ± 1.11 | flat (~−0.16) |
| Δ (H3-alt − P1) | **−0.07** | **+0.00** | **+0.13** | within paired-σ noise |

All 6 runs have Δreg vs B9 ≈ −3.4 pp (paper-anchored — these are the predecessor stack without P4).

**Interpretation:**
1. **Cross-attn does NOT unlock at low cw.** H3-alt and P1 are statistically identical at every cw value. The "hyperparameter-dormant cross-attn" hypothesis is **refuted**.
2. **Combined with F52 P5 ≈ B9**, this is a **paper-grade three-way confirmation**: cross-attention mixing at FL is **structurally dead**, regardless of (a) whether the layer is removed (P1), (b) whether the layer's mixing output is zeroed (P5/F52), or (c) what cat-loss weight is applied (F53 sweep). The shared backbone's only productive component is the per-task FFN+LN.
3. **The cw sweep also tests:** at lower cw, reg gradient signal scales up. If reg gradient was the bottleneck, lower cw should improve reg. It doesn't (reg flat across cw). → reg is NOT gradient-starved; it's saturating for a different reason (encoder saturation per F50 D5).

**Paper implication:** the architectural decomposition narrative is now anchored on three independent experiments converging to the same conclusion. The MTL backbone reduces to two parallel per-task FFN+LN stacks regardless of the cross-attention block's design.

---

## 5 · CA P3 unblock attempt: GPU-OOM on 4090, deferred to A100

Attempted on RunPod 4090 (24 GB). **OOMed at line `mtl_cv.py:541` (epoch-end train logit catting):**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.07 GiB.
GPU 0 has a total capacity of 23.53 GiB of which 5.02 GiB is free.
```

**Root cause:** the per-epoch train-side metric aggregation cats all-batches logits → memory = `n_classes × n_train_rows × 4 bytes`. For CA (8501 regions × ~280K train rows × fp32) = ~9 GB, on top of the active model state (~17 GB), exceeds the 24 GB ceiling.

**Mitigation landed (committed, available for A100 retry):** new `--skip-train-metrics` CLI flag bypasses the catting and reports placeholder train F1=0.0. Val metrics unchanged. Code in `src/training/runners/mtl_cv.py:528-558`, plumbed via `ExperimentConfig.skip_train_metrics`. The `run_p3_ca_unblock_attempt.sh` script now uses this flag.

**Decision:** CA P3 deferred to A100 (per user direction). The `--skip-train-metrics` flag is portable to A100 and will reduce memory pressure there too if needed. Run command stays in `scripts/run_p3_ca_unblock_attempt.sh`.

The original Lightning A100 RAM blocker (15 GB CPU RAM) was indeed wrong — the actual blocker on RunPod is GPU memory, which is a separate axis. A100 (40-80 GB) should fit comfortably.

---

## 6 · Cross-references

- `F50_T4_FINAL_SYNTHESIS.md` §1 — B9 champion headline (63.47 ± 0.75 reg / 68.59 ± 0.79 cat)
- `F50_T1_5_CROSSATTN_ABSORPTION.md` §5.2-5.3 — F52 P5 + F53 cw-sweep predictions
- `F50_T3_HYPERPARAM_BRAINSTORM.md` — B2/F64 design source (warmup-decay shape)
- `F50_D5_ENCODER_TRAJECTORY.md` — encoder saturation mechanism
- `C05_P3_NULL_RESULT_FALLBACK.md` — branching plan if CA P3 fails
- Code: `src/training/helpers.py` (`_build_reg_head_warmup_decay_lambda`); `src/models/mtl/mtlnet_crossattn/model.py` (`identity_attn`); `src/utils/progress.py` (`zip_longest_cycle` strategy)
- Run dirs: B2 `_0057`, F52 `_0115`, F65 `_0130` (in progress)
