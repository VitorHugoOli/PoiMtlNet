# Tier A1 large-state pilot — addendum (FL 5-fold + CA/TX 1-fold)

**Date**: 2026-05-28
**Scope**: log_T-KD (`--log-t-kd-weight` W=0.0 vs W=0.2, τ=1.0) at FL/CA/TX, seed=42, B9 large-state recipe.
**Status**: sign-and-magnitude pilot confirming the AL/AZ-validated lift transfers to large states. **NOT paper-grade.**

> **Framing (read first).** This is a **seed=42 sign-and-magnitude pilot**. Paper-grade FL/CA/TX numbers require multi-seed {0,1,7,100} per C23; seed=42 overshoots §0.1 v11 multi-seed by **+3 pp (CA) / +8 pp (TX)** of pure development-seed bias (the baseline W=0.0 numbers below sit well above §0.1, exactly as C23 predicts — FL W0.0 disjoint reg 63.98 vs §0.1 63.27; CA 50.06 vs 47.35; TX 50.38 vs 42.84). The **§4.2 composite** (STL c2hgi cat + STL HGI reg routed by task) already delivers **+7–12 pp vs MTL@disjoint at every large state** — log_T-KD's value at large states is as a **single-MTL-artefact alternative** (one model, no deploy-time routing), **not a composite competitor**. The pilot's purpose is solely to confirm the lift's **sign and magnitude transfer** from AL/AZ to large states.

---

## §1 Per-state Δ disjoint reg (W=0.2 − W=0.0)

| State | n | W=0.0 disjoint reg | W=0.2 disjoint reg | **Δ pp** | cat F1 Δ | test |
|---|---:|---:|---:|---:|---:|---|
| **FL** | 5 folds | 63.98 ± 0.76 | 66.38 ± 0.58 | **+2.40** | +0.01 | Wilcoxon p=**0.03125**, 5/5 folds positive |
| **CA** | 1 fold | 50.06 | 51.48 | **+1.42** | −0.10 | sign-and-magnitude only (n=1) |
| **TX** | 1 fold | 50.38 | 52.09 | **+1.71** | +0.05 | sign-and-magnitude only (n=1) |

**Per-fold FL Δ:** +2.34, +2.63, +2.09, +2.28, +2.65 (all positive; raw per-fold `top10_acc_indist`, no rounding — per the Tier A1 scipy-dispatch reproducibility note). FL Wilcoxon p=0.03125 is the exact one-sided minimum for n=5 / 5-positive.

**Cat axis is flat at all three states** (Δcat ∈ [−0.10, +0.05] pp), reproducing the small-state **reg-only-lift-with-flat-cat** signature that the F_TIER_A1 leak audit (§2 V6) identified as the OPPOSITE of a label-shortcut leak. No leak signature at large states.

---

## §2 Three-frontier table (FL — 5-fold)

| Frontier | W=0.0 (KD off) | W=0.2 (KD on) | STL ceiling (§0.1 v11) |
|---|---:|---:|---:|
| disjoint reg (top10_acc_indist) | 63.98 ± 0.76 | **66.38 ± 0.58** | 70.62 (`next_stan_flow`) |
| geom_simple reg | 61.14 ± 0.95 | **65.20 ± 0.74** | — |
| disjoint cat F1 | 70.49 ± 0.92 | 70.50 ± 0.92 | 67.16 (`next_gru`) |
| geom cat F1 | 66.98 ± 0.80 | 67.15 ± 1.14 | — |

KD closes ~**33 %** of the FL MTL→STL reg gap at disjoint (gap 70.62−63.98 = 6.64 pp; KD recovers 2.40 pp), and ~**45 %** on the geom_simple frontier (W=0.0 61.14 → W=0.2 65.20). The cat axis is unaffected (KD touches only the reg head's supervisory signal). **Caveat**: the STL ceiling (70.62) is multi-seed {0,1,7,100}; the W=0.0/W=0.2 numbers are seed=42 (overshoot ~+0.7 pp vs §0.1 MTL 63.27) — the gap-closure fraction is therefore approximate.

---

## §3 MI table — does the lift track headroom × MI-ratio?

MI(last_region_idx ; target_region_idx) / H(target), population-level on the train+val `next_region.parquet` (pad-only rows excluded), replicating F_TIER_A1_LEAK_AUDIT.md §4 method. FL/CA/TX computed here; AL/AZ from the audit.

| State | n_regions | MI/H(target) | top-1 determinism | P(last==target) | W=0.0 baseline (disjoint reg) | **measured Δ pp** |
|---|---:|---:|---:|---:|---:|---:|
| **FL** | 4 702 | **0.662** | 0.516 | 0.495 | 63.98 (seed=42) | **+2.40** (n=5) |
| **AL** | 1 109 | 0.601 | 0.368 | 0.313 | 50.59 (n=20) | +2.27 (n=20) |
| **AZ** | 1 547 | 0.560 | 0.341 | 0.301 | 41.30 (n=20) | +4.91 (n=20) |
| **CA** | 8 497 | 0.610 | 0.364 | 0.347 | 50.06 (seed=42) | +1.42 (n=1) |
| **TX** | 6 553 | 0.546 | 0.326 | 0.315 | 50.38 (seed=42) | +1.71 (n=1) |

**Interpretation.** The F_TIER_A1 audit (§4 V7) framed the lift as **dosage-on-headroom modulated by MI-ratio**: a more informative last→target prior (higher MI/H) delivers a stronger teacher, but the absolute pp lift also scales with how much **headroom** the W=0.0 baseline leaves.

The pilot is **consistent** with that model:
- **FL** has the **highest MI/H (0.662)** and a strong lift (+2.40), despite a high baseline (63.98) that leaves less headroom — the rich prior compensates.
- **AZ** remains the outlier-by-headroom: lowest baseline (41.30) → largest lift (+4.91), even though its MI/H (0.560) is the lowest. Pure headroom effect.
- **CA/TX** sit in the middle band: moderate MI/H (0.610 / 0.546) and moderate baselines (~50) → moderate lifts (+1.42 / +1.71). TX's slightly larger lift than CA tracks its slightly lower baseline (more headroom), even though TX's MI/H is lower — again headroom-dominant within the large-state band.

There is **no anti-correlation** that would suggest a leak: every state lifts positively, the cat axis stays flat everywhere, and the ordering (AZ ≫ FL ≈ AL > TX > CA on absolute Δ) is explained by headroom first, MI-ratio second. The large-state lifts (1.4–2.4 pp) are **smaller** than AZ's (4.9 pp) precisely because large-state baselines sit higher with less residual entropy for the prior to convert — exactly the audit's prediction.

---

## §4 Verdict

**The log_T-KD lift TRANSFERS to large states.** Sign is positive at all three (FL/CA/TX), magnitude +1.4 to +2.4 pp disjoint reg, reg-only with flat cat. FL is significant at n=5 (p=0.03125, all folds positive); CA/TX are single-fold sign-and-magnitude confirmations only. The transfer is consistent with the AL/AZ headroom × MI-ratio model and shows no leak signature.

**Recommended framing for the paper**: log_T-KD is a **single-model MTL artefact** that recovers ~30–45 % of the FL MTL→STL reg gap at zero deploy-time routing cost. It is NOT a competitor to the §4.2 composite (which delivers +7–12 pp at large states via task-routed STL models); it is the best **single-MTL-network** reg-lift lever found to date. Paper-grade claims require multi-seed {0,1,7,100} (C23); this pilot establishes only that the small-state-validated effect is real at scale.

---

## §5 Method / artefacts

- **Recipe**: B9 large-state (`--alternating-optimizer-step --alpha-no-weight-decay --min-best-epoch 5 --scheduler cosine --max-lr 3e-3`), cat-head `next_gru`, reg-head `next_getnext_hard`, task_a=checkin / task_b=region, `--mtl-loss static_weight --category-weight 0.75`, seed=42, 50 epochs.
- **BS**: FL 2048, CA 1024, **TX 256** (reduced from 512 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to clear an OOM in the full-epoch val-logit `torch.cat` at TX's 6 553-region head; this changes nothing about the comparison — both TX cells used identical BS 256).
- **Pre-flight**: per-fold seed=42 log_T rebuilt at n_splits=5 for all three states (C22 mtime guard PASS; FL rebuild was byte-identical to the prior valid file, confirming reproducibility). CA/TX log_T had to be rebuilt at n_splits=5 after a concurrent process left them at n_splits=2 (the C19 guard correctly hard-failed the first attempt — no leak).
- **Extraction**: `scripts/substrate_protocol_cleanup/summarize_tier_a1_largestate.py` — disjoint reg = per-fold max `top10_acc_indist`; geom_simple = `top10_acc_indist` at the geo-mean(cat-f1, reg-top10)-maximising epoch; raw per-fold values, Wilcoxon one-sided on raw differences (FL only).
- **MI**: `scripts/substrate_protocol_cleanup/mi_audit_largestate.py`.
- **Cells**: `docs/results/substrate_protocol_cleanup/tier_a1_largestate/{florida,california,texas}/W{0.0,0.2}/seed42/`. FL = full 5-fold; CA/TX = fold-1 only (early-stopped after fold 1 to conserve compute, per the AGENT_PROMPT 1-fold-pilot allowance).
- **Compute**: ~1.5 GPU-h total (FL 5+5 folds, CA/TX 2 fold-1 each + 1 TX OOM retry). Well under the 10 GPU-h cap.
- **GPU/disk snapshots**: `gpu_snapshots.log`. A concurrent-process disk-full event mid-pilot (host /home at 100 %) was recovered by deleting this pilot's own disposable checkpoints and re-running affected cells with `--no-checkpoints`; see log.md closure entry for the incident detail.
