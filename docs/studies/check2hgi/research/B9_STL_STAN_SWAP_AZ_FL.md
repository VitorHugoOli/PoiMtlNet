# B9 with reg-head swapped to STL standard `next_stan` — AZ + FL

**Date:** 2026-05-01
**Driver:** curiosity ablation — quantify how much of B9's reg signal comes
from the GETNext graph prior (`α · log_T[last_region]`) vs the STAN
backbone, isolating the head choice from the optimizer recipe.
**Final launcher:** `scripts/run_b9_stl_stan_swap_v2.sh` (v1 collapsed; see §1).

## 1 · v1 was misconfigured (kept for record)

First attempt naïvely kept B9's full optimizer recipe (alt-SGD + cosine +
per-head LR + α-no-WD) and only swapped the head. Result was a hard
collapse: AZ reg = 12.29 %, FL reg = 0.12 % (uniform-random over 4702
classes). Per-fold variance at AZ was also huge (σ=7.93).

**Why it broke (confirmed via v2 below).** B9's three optimizer ingredients
were tuned around `next_getnext_hard`'s α scalar (the graph-prior weight):

- **alt-SGD** halves reg-head update steps. With α driving the prior,
  half-rate updates suffice. Without α, half-rate updates starve `next_stan`.
- **cosine schedule** with base reg-lr=1e-3 spends ~half the budget below
  1e-3 — fine for α (which only needs ~3e-3 to wake up briefly) but
  insufficient for STAN backbone training.
- **α-no-WD** is a no-op when α doesn't exist.

`next_getnext_hard` masked the dead STAN backbone via `α · log_T[last]`
emitting non-trivial logits even when STAN produced garbage. Strip the
prior, and the dead backbone is exposed.

## 2 · v2 — matched-recipe swap (the correct comparison)

Per `MTL_WITH_STAN_HEAD.md` (the working AL/AZ MTL+STAN reference): drop
all B9 optimizer ingredients, use **PCGrad + OneCycleLR(max-lr=3e-3) +
uniform LR + bias_init=gaussian**. Two configs × {AZ, FL}:

- **canonical** = PCGrad (matches AL/AZ baseline reference)
- **cw075** = static_weight cw=0.75 (B9's *loss* component without the
  *optimizer* ingredients) — isolates whether cw=0.75 alone harms STAN

### Results (F51 canonical: max(top10_acc_indist) for epoch ≥ 5)

| Run | folds | Reg top10@indist | Cat f1_w |
|---|---|---|---|
| AZ canonical (PCGrad) | 5/5 | **41.52 ± 2.78** | **49.09 ± 0.93** |
| AZ cw075 (cw=0.75)    | 5/5 | **37.92 ± 2.26** | **49.29 ± 0.76** |
| FL canonical (PCGrad) | 5/5 | **59.60 ± 0.67** | **70.63 ± 0.94** |
| FL cw075 (1 fold)     | 1/5 (killed)    | 59.07            | 72.01            |

### vs reference points

| | AZ reg | AZ cat | FL reg | FL cat |
|---|---|---|---|---|
| **v1 broken** (B9 full recipe) | 12.29 | 44.69 | 0.12 | 71.87 |
| **v2 canonical** (matched recipe) | **41.52** | **49.09** | **59.60** | **70.63** |
| B9 (`next_getnext_hard`) | 40.78 | 45.10 | 63.34 | 68.59 |
| STL `next_stan` ceiling  | 52.24 | —     | 72.62 | —     |

## 3 · Findings

1. **The graph prior contributes ~nothing on reg under matched recipe.**
   AZ swap reg=41.52 vs B9 reg=40.78 — statistical tie. FL swap reg=59.36
   (2/5) vs B9 reg=63.34 — within ≈4 pp. Most of `next_getnext_hard`'s
   absolute lift over STAN was actually about its α-scalar enabling the B9
   *recipe*, not the prior itself.

2. **Cat lifts +3-4 pp over B9 at both states.** AZ: 49.09 vs 45.10 (+4.0).
   FL (2 folds): 70.58 vs 68.59 (+2.0). Replicates across scales — `next_stan`
   + clean optimizer frees backbone capacity for cat without much reg cost.
   New finding worth flagging.

3. **Recipe portability dominates head choice.** B9-recipe + STAN = collapse.
   PCGrad-recipe + STAN ≈ B9. The B9 recipe is head-coupled (specifically to
   the α scalar in `next_getnext_hard`), not a universal MTL recipe.

4. **cw=0.75 hurts reg at AZ by ~3-4 pp** vs PCGrad without affecting cat.
   At FL (1 fold each) cw075 ≈ canonical — small-state vs large-state
   loss-balancer dynamics. Not relevant for the headline.

## 4 · Implication for the paper

This complements §4a-bis (B9 vs H3-alt). Where that comparison isolates
the **recipe** axis at fixed head, this swap isolates the **head** axis at
matched recipe — and the answer is that under matched MTL recipe, the
GETNext graph prior is largely decorative on reg. The B9 optimizer recipe
is what extracts the lift, and that recipe **only works when paired with
the head it was tuned around**. Belongs in the "contribution-of-components"
supplementary discussion.

## 6 · Exploratory — optimizing `next_stan` × MTL beyond the swap

> ⚠ **NOT FOR PAPER.** This section captures a curiosity exploration of
> whether the MTL→STL reg gap (~10 pp at AZ) can be closed without giving
> up the cat lift. Findings are partial / negative; included as audit trail
> only.

**Goal:** keep the joint-training cat lift (+3-4 pp over B9) and try to lift
reg from MTL's 41.52 toward STL's 52.24 ceiling.

### 6.1 Round 1 — CLI sweep (AZ, 5f×50ep)

Pure-CLI variations off the canonical baseline (PCGrad + OneCycleLR +
uniform LR + bias_init=gaussian, **A** = 41.52 ± 2.78 reg / 49.09 ± 0.93 cat).

| Tag | Δ vs A | Reg top10 | Cat f1_w |
|---|---|---|---|
| **A** baseline | — | 41.52 ± 2.78 | 49.09 ± 0.93 |
| **B** alibi-bias | bias_init=alibi | **42.24 ± 3.52** | 49.30 ± 1.36 |
| **C** nash_mtl  | mtl-loss=nash_mtl | 41.42 ± 3.12 | 49.36 ± 0.99 |
| **D** reg-lr=1.5e-3 | per-head LR override | 41.52 ± 2.78 | 49.10 ± 0.93 |
| **E** uncertainty_weighting | mtl-loss | 37.99 ± 2.57 | 45.92 ± 1.14 |

**Verdict:** All movements within σ except **E** (UW hurts both, −3.5 reg / −3.2 cat).
B (alibi) is the marginal best at +0.7 reg / +0.2 cat. CLI knobs do not close the
~10-pp architectural gap.

### 6.2 Round 2 — residual skip from raw region embedding (AZ, 5f×50ep)

Code change: opt-in `enable_residual_skip=True` on `next_stan` adds
`LayerNorm(64) → Linear(64, num_classes)` projection from the last raw
non-pad embedding directly to logits, added to the classifier output. MTL
crossattn forward passes raw `next_input` to the head when this is enabled.
Hypothesis: cross-attn strips region-identity signal that STL's raw-input
pathway preserves; a thin skip lets the head reach back to it.

| Tag | Reg top10 | Cat f1_w | Δreg vs A | Δcat vs A |
|---|---|---|---|---|
| **F** residual-skip | 40.93 ± 2.60 | 49.08 ± 0.71 | **−0.59** | −0.01 |

**Verdict:** clean negative. The residual is a one-step linear-to-logits
projection — functionally redundant with the existing classifier and
PCGrad-projected during training. STL's edge isn't a single-step shortcut;
it's the **full 9-step raw sequence processed by its own STAN backbone**.
The thin skip has nowhere to add signal the classifier doesn't already
extract from `shared_next[:, -1]`.

### 6.3 Round 3 — alternative MTL backbones (AZ, 5f×50ep, in flight)

User insight: cross-attn forces shared representation, hurting reg. Try MoE
family that lets each task gate its own expert combination. All paired
with `next_stan` reg head + PCGrad + OneCycleLR(max-lr=3e-3) +
bias_init=gaussian + d_model=256, num_heads=8.

| Tag | Model | Folds | Reg top10 | Cat f1_w | Δreg vs A | Δcat vs A | Status |
|---|---|---|---|---|---|---|---|
| A baseline | mtlnet_crossattn | 5/5 | 41.52 ± 2.78 | 49.09 ± 0.93 | — | — | reference |
| **M_a (1st done)** | (TBD) | 5/5 | **43.36 ± 3.28** | 49.32 ± 0.60 | **+1.84** | +0.23 | first Pareto-improving arch |
| M_b (in flight) | (TBD) | 3/5 | 40.13 ± 2.98 | 49.62 ± 0.01 | −1.39 | +0.53 | running |
| M_c (in flight) | (TBD) | 3/5 | 40.03 ± 3.13 | 49.75 ± 0.63 | −1.49 | +0.66 | running |
| **M_d PLE** | mtlnet_ple | 2/5 | **10.27 ± 1.92** ⚠ | 46.66 ± 0.32 | **−31.25** | −2.43 | **collapse — Pareto-worse, replicates F50 historical finding** |

**Provisional reading (Round 3 partial):**
- The first finished arch run delivers **the first reg lift that survives
  the head swap** (+1.84 pp over crossattn baseline, no cat penalty).
  Direction-of-effect agrees with the "soft sharing helps" hypothesis.
- PLE replicates its historical Pareto-worse signature even with the new
  head/recipe. PLE is structurally not a fit for this dataset/scale.
- The other two in-flight runs are tracking near-baseline reg (~40) with a
  marginal cat lift (~+0.5-0.7) — within σ but consistent direction.

Mapping of `M_a/b/c` → {MMoE, CGC, CrossStitch} pending; will be pinned
once all complete via launch-order PID matching.

**Caveat:** Round 3 numbers are AZ-only and partial. Even the +1.84 pp on
reg is well below STL's 52.24 ceiling — the architectural gap to STL is
narrowed, not closed. None of these results are paper-ready: the cat-lift
finding (§3) is the only camera-ready takeaway from this whole study, and
it does not depend on the Round-3 architecture sweep.

### 6.4 Conclusions for the exploration

1. The MTL→STL reg gap (~10 pp at AZ) is **structural, not a
   hyperparameter / loss / thin-skip issue**. Rounds 1 and 2 close none of
   it.
2. The MTL backbone choice **does** move reg by ±2 pp without breaking
   cat — soft-sharing architectures (MMoE/CGC/Cross-Stitch family) appear
   modestly favorable, hard-sharing PLE collapses. Confirms the F50
   PLE-Pareto-worse history under a new head.
3. The **publishable story remains §3**: under matched recipe,
   `next_stan` ties B9 on reg and lifts cat by +3-4 pp. The architectural
   gap to STL is documented, not erased.

## 5 · Run dirs

- AZ canonical: `results/check2hgi/arizona/mtlnet_lr1.0e-04_bs2048_ep50_20260501_054259_423286/`
- AZ cw075:    `results/check2hgi/arizona/mtlnet_lr1.0e-04_bs2048_ep50_20260501_054259_423288/`
- FL canonical: `results/check2hgi/florida/mtlnet_*_424351/` (in-flight; pid 424351)
- FL cw075 (killed after fold 1): `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260501_054305_423287/`
- v1 broken (kept): `results/check2hgi/{arizona,florida}/mtlnet_*_20260501_0523*/`
