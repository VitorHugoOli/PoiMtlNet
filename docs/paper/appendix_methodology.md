# Methodological appendix — Loss-side `task_weight=0` ablation under cross-attention MTL

**Date:** 2026-04-28
**Target:** Submission paper appendix A (or supplementary §S.1). ~3 pages.
**Status:** Stand-alone draft; should also live as `research/F49_METHODOLOGICAL_NOTE.md` in the study tree.

---

## A.1 Statement

Setting `task_weight = 0` for one task in a cross-attention multi-task model **does not** isolate that task's architectural contribution from its co-adaptation contribution. The silenced encoder continues to receive gradient through the cross-attention K/V projections of the surviving task, producing a "partially trained" encoder that confounds any decomposition built on the loss-side ablation.

This applies to any MTL architecture where the silenced task's encoder feeds keys/values into the surviving task's attention computation: MulT, InvPT, HMT-GRN-style cross-modal attention, and our `MTLnetCrossAttn`. The clean isolation requires explicit `requires_grad=False` on the silenced encoder + head — what we call **encoder-frozen ablation**.

## A.2 The gradient-flow analysis

In a bidirectional cross-attention block, let `A` and `B` be the cat and reg streams. The block computes:

```
A' = A + cross_ab(Q_a, K_b, V_b)     # cat queries reg keys/values
B' = B + cross_ba(Q_b, K_a, V_a)     # reg queries cat keys/values
```

Suppose we set `category_weight = 0` so `L = L_reg`. Compute `∂L/∂θ_cat` for some weight `θ_cat` in the cat encoder:

- **Direct path (via L_cat):** `∂L_reg/∂L_cat = 0` because `category_weight = 0`. **Direct path is zero.**
- **Indirect path (via cross_ba's K/V):** `∂L_reg/∂B' → ∂B'/∂cross_ba(Q_b, K_a, V_a) → ∂cross_ba/∂(K_a, V_a) → ∂(K_a, V_a)/∂A → ∂A/∂θ_cat`. **Indirect path is nonzero.**

The cat encoder receives a *reg-shaped* gradient: it is being trained to produce K/V that reduce L_reg, not to produce features useful for cat. Loss-side `task_weight=0` does not freeze the cat encoder; it **redirects** its training signal. This is mechanistically not "no cat training"; it is "cat training as reg-helper."

## A.3 Empirical confirmation

We verify this with four passing regression tests in `tests/test_regression/test_mtlnet_crossattn_lambda0_gradflow.py`:

1. **Loss-side λ=0 cat encoder receives gradient through cross_ba K/V.** Sets `category_weight=0`, runs one backward pass, asserts `θ_cat.grad.norm() > 0` for cross-attention K/V projection layers in the cat stream.
2. **Cat head receives no gradient under loss-side λ=0.** Asserts `θ_cat_head.grad.norm() == 0` (cat head only flows from L_cat, which is zero).
3. **Frozen-cat encoder weights unchanged after `optimizer.step()` within fp32 epsilon.** Catches a subtle AdamW bug where `weight_decay=0.05` decays "frozen" weights (with `requires_grad=False`) silently — `setup_per_head_optimizer` filters `requires_grad=False` from every param group before AdamW construction.
4. **Optimiser's cat group has 0 trainable params under freeze.** Verifies the AdamW filter actually removed all frozen params.

## A.4 Quantitative impact (F49 decomposition)

We run the 3-way decomposition (encoder-frozen / loss-side / Full MTL) at AL+AZ+FL n=5. Each cell uses identical fold splits (`--no-folds-cache`, seed=42) so the comparison is paired.

```
Full MTL − STL  =  (frozen_λ0 − STL)        ← architecture alone
                +  (loss_λ0 − frozen_λ0)    ← cat-encoder co-adaptation via K/V
                +  (Full MTL − loss_λ0)     ← cat-supervision transfer
```

The **loss-side ablation** (`category_weight=0` only) reports:
- AL: `loss_λ0 − STL = +6.57 pp` — looks like "architecture + cat-encoder helping"
- AZ: `loss_λ0 − STL = −4.04 pp`

The **encoder-frozen ablation** (`requires_grad=False` + `category_weight=0`) reports:
- AL: `frozen_λ0 − STL = +6.48 pp` — pure architectural lift from cross-attention
- AZ: `frozen_λ0 − STL = −6.02 pp` — architecture costs

**The two ablations diverge by**:
- AL: 0.09 pp (cat encoder's K/V co-adaptation contributes ≈ 0)
- AZ: 1.98 pp (cat encoder's K/V co-adaptation rescues +1.98 pp)
- FL: 8.27 pp (cat encoder's K/V co-adaptation contributes large effect at scale, but high σ)

These differences ARE the load-bearing methodological signal. A reader using only loss-side λ=0 to "ablate" the cat task would *attribute the K/V co-adaptation to the architecture itself*, inflating the architectural credit. The encoder-frozen variant separates them.

## A.5 Implication for prior MTL literature

The loss-side `task_weight=0` ablation is widely used in MTL papers as the "remove this task" control. Where the underlying architecture has cross-task gradient flow (cross-attention, gating, mixture-of-experts with shared params), the ablation is **methodologically unsound**. Ablations from MulT, InvPT, HMT-GRN, and any architecture where one task's encoder feeds another's attention should be re-examined under encoder-frozen isolation. Our cross-attention case shows the gap can be ~2 pp on a balanced state and ~8 pp at scale — large enough to flip qualitative claims.

## A.6 Practical recipe for MTL ablations

When ablating "task A's contribution" in cross-attention MTL:

1. Set `task_a_weight = 0` in the loss (`L = w_b · L_b`).
2. **Also** set `requires_grad = False` on `task_a`'s encoder + head + any task-A-private parameters.
3. **Filter `requires_grad=False` params from optimiser groups** before AdamW construction. Without this filter, `weight_decay > 0` decays frozen weights silently. (See `src/training/helpers.py::setup_per_head_optimizer` for our implementation.)
4. Verify with regression tests that backward passes leave `task_a` params unchanged within fp32 epsilon.

The test at step 4 caught the silent AdamW decay bug in our F49 attribution work; it is mandatory.

## A.7 What encoder-frozen does NOT isolate

`ffn_a` and `ln_a*` — the block-internal cat-side FFN/LayerNorm — live in `shared_parameters()` and continue to train under L_reg in *both* F49 variants (frozen and loss-side). They process whatever K/V the cat-side stream feeds (random-init in frozen, slow co-adapting in loss-side); their *training* is identical across variants. The "totally-frozen-cat-side-block" variant (freeze `ffn_a + ln_a*` too) breaks the reg pipeline — `cross_ba` reads `a` outputs as K/V; if `a`'s in-block layers can't update, the shared cross-attn loses degrees of freedom needed for reg.

A redesigned isolation would `detach()` `a`'s outputs from autograd before `cross_ba` reads them — separating "block-internal cat processing trains" from "cat encoder trains via K/V". This is listed as a deferred follow-up (P9 in PAPER_PREP_TRACKER, ~1-2h dev + ~1h compute).

## A.8 Summary

- Loss-side `task_weight=0` ablation under cross-attention is **unsound** — it redirects gradient rather than removing it.
- The clean architectural decomposition requires **encoder-frozen** ablation (`requires_grad=False` + AdamW filter).
- Empirically, the gap between the two ablations is 0.09–8.27 pp across our 3 states — large enough to materially affect attribution.
- The methodological recipe in §A.6 should be adopted for any MTL paper that reports loss-side weight ablations on cross-attention or gated architectures.
