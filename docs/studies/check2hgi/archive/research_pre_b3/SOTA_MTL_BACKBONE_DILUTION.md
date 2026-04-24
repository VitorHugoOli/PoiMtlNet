# Research — SOTA MTL fixes for backbone dilution (2024–2026)

**Date:** 2026-04-17
**Triggered by:** `issues/BACKBONE_DILUTION.md` — MTL regresses on AL against the strong standalone GRU region head; shared-backbone capacity is the bottleneck, not gradient conflict.

## Framing our failure correctly

**Our pathology is capacity / routing, not gradient conflict.** We tested 5 optimizers (PCGrad, NashMTL, GradNorm, CAGrad, equal_weight) and 5 backbones (FiLM, CGC, MMoE, DSelectK, PLE). All failed the Δm bidirectional gate on AL. The 2024–2026 gradient-manipulation newcomers (FAMO, IMGrad, NTKMTL, UW-SO, FairGrad, STCH) are variations on reweight/reproject over the same shared backbone — conceptually in the bucket we already ran.

**The architectural-asymmetry bucket** is where a 14-pp Δm gap is plausibly closable.

---

## Top 4 candidates ranked by expected-lift / effort on 10 K-sample 2-task setup

### 1. Asymmetric / self-auxiliary framing — highest-leverage, directly addresses the pathology

- **Paper:** *Enabling Asymmetric Knowledge Transfer in Multi-Task Learning with Self-Auxiliaries* — [arXiv:2410.15875 (2024)](https://arxiv.org/html/2410.15875v1)
- **Claim:** directed, asymmetric knowledge transfer via self-auxiliaries. Under-parameterised auxiliaries regularise the shared representation without stealing capacity from the dominant task.
- **Why it matches:** our strong GRU-on-region becomes the dominant task; category becomes an auxiliary regulariser. No forced symmetric sharing. Implementable as a loss-scale + gradient-scale on the category branch — stays within our runner.

### 2. Knowledge-Graph Tokenisation for Behavior-Aware Next-POI — direct topic anchor

- **Paper:** [arXiv:2509.12350 (Sept 2025)](https://arxiv.org/html/2509.12350)
- **Claim:** main = next-POI; auxiliaries = next-category AND next-region. **Exactly our task setup.** Uses generative tokenisation; the task-orchestration contribution is reusable even without the LLM.
- **Applicability:** closest literature analogue to our track. Cite for the BRACIS framing and lift the auxiliary-weighting schedule.

### 3. AdaShare / per-task sparse routing through the shared stack

- **Anchors:** [thegradient.pub review of AdaShare](https://thegradient.pub/how-to-do-multi-task-learning-intelligently/); *Dual-Mask progressive sparse MTL*, [Pattern Recognition 2024](https://www.sciencedirect.com/science/article/abs/pii/S0031320324007015)
- **Claim:** each task learns a binary routing policy over the shared residual blocks. Region can **skip blocks that category needs**, avoiding the dilution bottleneck.
- **Applicability:** directly addresses "shared backbone has finite capacity split between 2 tasks." Invasive (per-task masks on the residual stack) but solves the pathology at its root.

### 4. MTLoRA — per-task LoRA adapters around shared residual blocks

- **Paper:** *MTLoRA: Low-Rank Adaptation for Efficient Multi-Task Learning*, [CVPR 2024 (arXiv:2403.20320)](https://arxiv.org/abs/2403.20320)
- **Refinements 2025:** LoRI — [arXiv:2504.07448](https://arxiv.org/abs/2504.07448) (freezes A, sparsifies B with task masks); MTL-LoRA — [AAAI 2025 (arXiv:2410.09437)](https://arxiv.org/abs/2410.09437).
- **Claim:** task-agnostic + task-specific LoRA gives each task dedicated low-rank capacity while sharing the bulk. Non-LLM, dense-prediction setting — maps to our sequential heads.
- **Applicability:** drops into FiLM-modulated residual blocks with minimal surgery; gives strong GRU-on-region dedicated capacity without removing sharing.

---

## Deprioritised — gradient-manipulation newcomers

Not a zero — useful as sanity-floor, but unlikely to close 14 pp given our 5 prior optimizer tries:

- **IMGrad** *Injecting Imbalance Sensitivity for MTL* — [IJCAI 2025 (arXiv:2503.08006)](https://arxiv.org/abs/2503.08006). Explicitly targets imbalance/dominance (which Nash/CAGrad under-address). Constrains projected gradient norms.
- **NTKMTL** *Mitigating Task Imbalance from NTK Perspective* — [NeurIPS 2025 (arXiv:2510.18258)](https://arxiv.org/abs/2510.18258). Balances per-task NTK eigenvalue convergence speeds.
- **FAMO** — [NeurIPS 2023 (arXiv:2306.03792)](https://arxiv.org/abs/2306.03792).
- **STCH** — [ICML 2024 (arXiv:2402.19078)](https://arxiv.org/html/2402.19078v3). Pareto-set scalarisation; applicability to heterogeneous-CE unclear.
- **UW-SO / Analytical UW** — [arXiv:2408.07985 (2024–25)](https://arxiv.org/html/2408.07985v1).

## Skip

- **FairGrad, CAMRL** — fairness/RL-oriented; limited transfer to our supervised setup.

---

## Cheapest-first action list

| # | Action | Effort | Rationale |
|---|--------|--------|-----------|
| 1 | **Random Loss Weighting (RLW)** | 1 line | [Lin et al., TMLR 2022](https://openreview.net/forum?id=jjtFD8A1Wx). Strong baseline; if RLW matches Nash/CAGrad, our optimizer family is saturated — architectural fix needed. |
| 2 | **Asymmetric loss schedule / weight-warmup** | ~20 LOC in `mtl_cv.py` | Train region alone N epochs, then add category with low weight (0.1 → 0.3). Tests self-auxiliary hypothesis. |
| 3 | **Scale-down auxiliary gradient into shared backbone** | pure hook | Multiply category's gradient through shared stack by 0.2, leave its own head at 1.0. No arch change. |
| 4 | **MTLoRA adapters on each residual block** | ~1 day | Dedicated capacity for region. Most likely single change to move Δm. |
| 5 | **AdaShare-style learned per-task skip gates** | bigger refactor | Definitively tests the capacity conjecture. |
| 6 | **IMGrad or NTKMTL** | moderate | Optimizer sanity-floor only. |

---

## Paper-framing recommendation

Pivot the thesis from "MTL helps both heads" (gradient-conflict framing) to:

> **"Shared-backbone MTL is the wrong inductive bias when one task has a strong standalone head. On small POI data, the dominant-task dilution pathology is capacity-bound, not gradient-bound; per-task routing closes the Δm gap where 5 gradient-balancing optimizers cannot."**

This is a **sharper** claim than the gradient-conflict framing, directly supported by AdaShare/MTLoRA/self-auxiliary lines of 2024–2025 work. It also positions our negative AL result (CH01 fails) as a **motivated discovery** rather than a disappointment.

## Primary sources

- FAMO — https://arxiv.org/abs/2306.03792
- STCH — https://arxiv.org/html/2402.19078v3
- IMGrad — https://arxiv.org/abs/2503.08006
- NTKMTL — https://arxiv.org/abs/2510.18258
- Self-auxiliaries — https://arxiv.org/html/2410.15875v1
- KG-Tokenisation next-POI — https://arxiv.org/html/2509.12350
- MTLoRA — https://arxiv.org/abs/2403.20320
- LoRI — https://arxiv.org/abs/2504.07448
- MTL-LoRA — https://arxiv.org/abs/2410.09437
- ImNext — https://www.sciencedirect.com/science/article/abs/pii/S0950705124003095
- UW-SO — https://arxiv.org/html/2408.07985v1
- RLW — https://openreview.net/forum?id=jjtFD8A1Wx
- AdaShare — https://thegradient.pub/how-to-do-multi-task-learning-intelligently/
- HMT-GRN — https://bhooi.github.io/papers/hmt_sigir22.pdf
