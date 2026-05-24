# Experiment — B9 with HGI substrate

**Date:** 2026-05-16
**Question (from [`considerations.md`](considerations.md)):** *How behaves the B9 model with HGI instead of check2hgi?*
**Scope:** brief — single seed (42), 2 states (AL, AZ), 5-fold × 25 epochs, leak-free per-fold log_T.

## TL;DR

Under the canonical B9 recipe with **`--engine hgi`** substituted for `--engine check2hgi`:

| State | substrate | reg `top10_acc_indist` | cat F1 | Δ vs C2HGI baseline |
|---|---|---:|---:|---|
| AL | C2HGI (5f × 25ep, multi-seed) | 47.66 ± 3.07 | 35.91 ± 1.24 | — |
| AL | **HGI** (5f × 25ep, seed=42) | **61.48 ± 5.08** | **25.23 ± 0.64** | **reg +13.81, cat −10.68** |
| AZ | C2HGI (5f × 25ep, multi-seed) | 40.89 ± 1.95 | 42.69 ± 0.69 | — |
| AZ | **HGI** (5f × 25ep, seed=42) | **53.88 ± 2.68** | **29.07 ± 1.09** | **reg +12.99, cat −13.62** |

**The substrate × MTL interaction is bidirectional and large:**
- **HGI substrate buys ~+13 pp on reg** (in line with the substrate axis on reg from v11 §0.3).
- **HGI substrate costs ~−11–14 pp on cat** (matching CH16 / v11 §0.3 substrate axis on cat).

**The cross-attn MTL recipe is essentially transparent under HGI:** my MTL+HGI numbers match the v11 STL+HGI numbers to within ~0.5 pp on both heads. **Under C2HGI, MTL pays its known reg cost (~−9 pp vs STL ceiling) and earns a smaller cat lift.** Under HGI, neither happens — MTL≈STL.

## Setup

- Recipe: B9 verbatim — `mtlnet_crossattn`, `static_weight(cat=0.75)`, `next_gru` cat head, `next_getnext_hard` reg head, per-head LR (cat 1e-3, reg 3e-3, shared 1e-3), cosine schedule (max_lr=3e-3), alternating SGD, α-no-WD.
- Only change vs canonical: `--engine hgi` instead of `--engine check2hgi`. The `--task-a-input-type checkin --task-b-input-type region` flags still work but now load HGI's POI-level embeddings — same POI gets the same vector across all 9 sequence steps.
- Protocol: `--folds 5 --epochs 25 --seed 42`. Leak-free per-fold log_T (engine-agnostic; reused from `output/check2hgi/<state>/`).
- Reproducer: [`run_hgi_substrate.sh`](../../../scripts/mtl-exploration/run_hgi_substrate.sh).

## Results

### AL — 5-fold × 25 epochs × seed=42

| arm | reg top10_indist | cat F1 |
|---|---:|---:|
| C2HGI baseline (multi-seed n=20 from Step 3) | 47.66 ± 3.07 | 35.91 ± 1.24 |
| **HGI** (n=5 single seed) | **61.48 ± 5.08** | **25.23 ± 0.64** |
| Δ (HGI − C2HGI) | **+13.81 pp** | **−10.68 pp** |

### AZ — 5-fold × 25 epochs × seed=42

| arm | reg top10_indist | cat F1 |
|---|---:|---:|
| C2HGI baseline (multi-seed n=20 from Step 3) | 40.89 ± 1.95 | 42.69 ± 0.69 |
| **HGI** (n=5 single seed) | **53.88 ± 2.68** | **29.07 ± 1.09** |
| Δ (HGI − C2HGI) | **+12.99 pp** | **−13.62 pp** |

### Cross-check against v11 paper canon (RESULTS_TABLE §0.3, STL substrate axis, 5f × 50ep)

| State | v11 STL+HGI cat F1 | my MTL+HGI cat F1 (25ep) | Δ |
|---|---:|---:|---:|
| AL | 25.26 ± 1.18 | **25.23 ± 0.64** | **−0.03 pp** ≈ identical |
| AZ | 28.69 ± 0.79 | **29.07 ± 1.09** | +0.38 pp ≈ identical |

| State | v11 STL+HGI reg top10_indist | my MTL+HGI reg top10 (25ep) | Δ |
|---|---:|---:|---:|
| AL | 61.86 ± 3.29 | **61.48 ± 5.08** | −0.38 pp ≈ identical |
| AZ | 53.37 ± 2.55 | **53.88 ± 2.68** | +0.51 pp ≈ identical |

**MTL+HGI ≡ STL+HGI on both heads, both states.** Within ~0.5 pp on every cell.

## Interpretation — the substrate × MTL interaction

Putting this together with the v11 STL+C2HGI numbers and our MTL+C2HGI Step-3 numbers:

| State | STL+C2HGI reg | MTL+C2HGI reg | MTL vs STL Δ (C2HGI) | STL+HGI reg | **MTL+HGI reg** | MTL vs STL Δ (HGI) |
|---|---:|---:|---:|---:|---:|---:|
| AL | 59.15 (v11) | 47.66 (ours) | **−11.49** | 61.86 (v11) | **61.48 (ours)** | **−0.38** ≈ 0 |
| AZ | 50.24 (v11) | 40.89 (ours) | **−9.35** | 53.37 (v11) | **53.88 (ours)** | **+0.51** ≈ 0 |

| State | STL+C2HGI cat | MTL+C2HGI cat | MTL vs STL Δ (C2HGI) | STL+HGI cat | **MTL+HGI cat** | MTL vs STL Δ (HGI) |
|---|---:|---:|---:|---:|---:|---:|
| AL | 41.35 (v11) | 35.91 (ours) | −5.44 (25ep budget) | 25.26 (v11) | **25.23 (ours)** | **−0.03** ≈ 0 |
| AZ | 43.90 (v11) | 42.69 (ours) | −1.21 (25ep budget) | 28.69 (v11) | **29.07 (ours)** | **+0.38** ≈ 0 |

> Caveats: v11 STL numbers are 50-epoch; my MTL numbers are 25-epoch. The cat MTL−STL Δ under C2HGI is partly the 25ep budget cap (cat keeps climbing past ep 25 — see existing trajectory data). The reg numbers under both substrates are robust because reg peaks earlier (often by ep 8–15).

**Two clean takeaways:**

1. **Under HGI, the cross-attn MTL coupling is null.** MTL+HGI matches STL+HGI on both heads (within ~0.5 pp). The MTL machinery does nothing useful AND does nothing harmful with HGI substrate.

2. **Under C2HGI, the cross-attn MTL pays a known reg cost** (−9 to −11 pp vs STL+C2HGI ceiling). This is the classic v11 §0.1 tradeoff finding. The "−9 pp on reg" is *paid in exchange for the substrate's cat lift surviving joint training* — but only with C2HGI does that joint training happen at all in a non-trivial way.

## Why this matters

The CH18-substrate reframing in v11 (`docs/NORTH_STAR.md` §"Caveats — Phase-1 substrate-specific addendum") said:

> **MTL B3 only works with Check2HGI substrate.** Substituting POI-stable HGI embeddings into the same B3 setup actively breaks the reg head (MTL+HGI Acc@10 = 29.95 < STL+HGI Acc@10 = 67.52 at AL — a 37 pp regression).

That was measured on 2026-04-27, which is:
1. **Pre-C4 fix** (leak fix landed 2026-04-29) — so the runs used the legacy full-data `region_transition_log.pt` which leaked val transitions into the `α · log_T` prior. Inflation at convergence is ~13–17 pp on reg (`MTL_FLAWS_AND_FIXES.md §2.12`).
2. **Pre-B9 recipe** — the runs used the B3 recipe (single LR, OneCycleLR, no per-head LR, no cosine, no alt-SGD, no α-no-WD). B9 didn't exist yet.

So the "37 pp regression" between then and now is the combined effect of **two changes**, and I cannot strictly attribute it to the leak alone. Under leak-free *and B9*:

- MTL+HGI reg = 61.48 (AL) ≈ STL+HGI reg = 61.86 → **the catastrophic regression vanishes** under leak-free B9 protocol.
- MTL+HGI cat = 25.23 (AL) ≈ STL+HGI cat = 25.26 → **NO MTL transfer lift on cat either**.

**Honest framing of what's been established:**

- ✅ Under the current leak-free B9 protocol, MTL+HGI does NOT break either head — both match STL+HGI within ~0.5 pp.
- ✅ **B3+HGI under leak-free conditions has now been measured** (see §B3+HGI discriminator below). The catastrophic break was ~80% leak-driven and ~15% recipe-driven.

### B3+HGI discriminator (added 2026-05-16, FL still pending)

To isolate whether the CH18 catastrophe was the C4 leak or the B3 recipe instability, ran B3+HGI under leak-free per-fold log_T (5f × 50ep × seed=42, OneCycleLR max=0.003, single LR — no per-head, no cosine, no alt-SGD, no α-no-WD). Reproducer: [`run_b3_hgi.sh`](../../../scripts/mtl-exploration/run_b3_hgi.sh).

| State | reg top10_indist | cat F1 |
|---|---:|---:|
| AL | **57.38 ± 4.64** | 25.95 ± 0.98 |
| AZ | **46.95 ± 2.60** | 29.83 ± 0.83 |
| FL | pending (~5 h MPS in flight) | pending |

**Decomposition of the original AL catastrophe (29.95 → 61.86):**

| Step | reg | Δ from prior step | Cumulative recovery |
|---|---:|---:|---|
| Pre-leak-fix B3+HGI (CH18, 2026-04-27) | 29.95 | — | 0% (catastrophic baseline) |
| **+ leak fix** (C4 → per-fold log_T) at B3 | **57.38** | **+27.43 pp** | **~84% of catastrophe** ⭐ |
| + recipe upgrade B3 → B9 | 61.48 | +4.10 pp | ~13% additional |
| STL+HGI ceiling | 61.86 | +0.38 pp | fold noise |

**AZ decomposition (22.10 → 53.37):** leak fix +24.85 pp (~78%); recipe +6.93 pp (~22%).

**The CH18-substrate "37 pp catastrophic break" was DOMINANTLY a leak artefact.** The B3 recipe itself, under leak-free conditions, trails the STL+HGI ceiling by only ~5 pp on reg — within normal MTL-cost range, not catastrophic.

**Cat F1 under HGI substrate** is essentially constant across all four recipes (B3 pre-leak: 25.96 / B3 leak-free: 25.95 / B9 leak-free: 25.23 / STL: 25.26 at AL — all within 0.7 pp). HGI substrate's cat capacity is structurally capped at ~25 (AL) and ~29 (AZ) by the absence of per-visit context — recipe doesn't move it.

### Refined story (replaces the earlier honest-uncertainty framing)

Under leak-free B3+HGI conditions:

- **MTL does NOT break under HGI substrate.** The previously-reported catastrophic reg break (29.95 at AL) was a leak artefact. Leak-free B3+HGI reg = 57.38 (only 4.5 pp below the STL+HGI ceiling of 61.86).
- **Recipe contribution to the recovery is small (~12-22%).** Switching to B9 (per-head LR, cosine, alt-SGD) brings reg from 57.4 to 61.5 at AL — a tightening, not a rescue.
- **The cross-attention MTL is essentially transparent under HGI** for both B3 and B9 recipes (within ~5 pp of STL+HGI). It does not extract joint-training signal from a POI-stable substrate.

**Implication for `NORTH_STAR.md`:** the §Caveats Phase-1 paragraph stating "MTL B3 only works with Check2HGI substrate. ... MTL+HGI actively breaks the reg head (Acc@10 = 29.95 < STL+HGI Acc@10 = 67.52 at AL — a 37 pp regression)" is **largely refuted** by the leak-free measurement. The actual leak-free regression is 4.48 pp at AL and 6.42 pp at AZ — within normal MTL-cost range. Worth updating the doc to reflect the leak-free retraction once FL completes.

The refined story: **the cross-attention MTL only does measurable joint-training work under per-visit-contextual substrates (C2HGI). Under POI-stable substrates (HGI), the two task streams effectively decouple — cross-attn passes through.** Mechanism speculation: with per-POI-stable embeddings, the same POI gets the same K/V across all 9 sequence steps, making cross-stream attention an averaging operation that contributes no signal.

## Caveats

1. **Single seed (n=5 fold-pairs).** Multi-seed would tighten the σ but the effect sizes (~13 pp on both axes) are far above any plausible fold/seed noise.
2. **25-epoch budget.** Cat at AL/AZ keeps climbing past ep 25 under C2HGI, but the HGI cat numbers are already at the STL ceiling at ep 25 — extending to ep 50 won't change much for HGI. C2HGI MTL+cat would gain ~1-3 pp at 50ep, narrowing the −10/−13 pp Δ slightly but not changing the direction.
3. **AZ HGI reg σ = 2.68** is larger than C2HGI's (1.95). With n=5 the +13 pp Δ is still many σ above noise. Worth confirming at n=20 if this becomes a paper claim.
4. **Generalisation to FL/CA/TX not tested.** The substrate axis on reg is documented at all 5 states in v11 §0.3; the MTL × HGI null result here would likely hold at larger scale but we haven't measured.

## Open questions (for the main study)

1. **Why does MTL≡STL under HGI?** The "POI-stable K/V makes cross-attn into averaging" hypothesis is testable: directly inspect the attention weights of `_CrossAttnBlock.cross_ab` and `cross_ba` under both substrates. If HGI gives near-uniform attention, the mechanism is confirmed.
2. **Is the substrate × MTL interaction asymmetric on reg?** I see MTL+HGI reg ≈ STL+HGI reg, but MTL+C2HGI reg < STL+C2HGI reg by −9 to −11 pp. Why does the reg head only pay the MTL cost under C2HGI? Probably because C2HGI's per-visit context gives the cat path *something to learn from*, which competes for shared-backbone capacity with reg. Under HGI both paths see static features → no competition.
3. **Does the multi-seed AZ HGI result hold at n=20?** Quick to verify (~50 min MPS); promotes the null-MTL-under-HGI claim to paper-grade.
4. **Cross-state replication.** FL+CA+TX at 5f×25ep×seed=42 with HGI would round out the picture for the main study.

## Files

- Run script: [`run_hgi_substrate.sh`](../../../scripts/mtl-exploration/run_hgi_substrate.sh)
- Run dirs: `results/hgi/{alabama,arizona}/mtlnet_lr1.0e-04_bs2048_ep25_20260516_*`
- Logs: `logs/hgi_b9_{alabama,arizona}_seed42_5f25ep.log`
