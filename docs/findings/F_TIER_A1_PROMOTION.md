# F_TIER_A1_PROMOTION — log_T-KD multi-seed promotion (substrate-protocol-cleanup Tier A1)

**Date:** 2026-05-29. **Study:** [`studies/substrate-protocol-cleanup/`](../studies/substrate-protocol-cleanup/) Tier A1. **Claim:** CH26. **Cross-ref:** [`F_TIER_A1_LEAK_AUDIT.md`](F_TIER_A1_LEAK_AUDIT.md) (independent 7-vector leak audit — NO LEAK). **Cost:** ~2.5 GPU-h (1 h small-state sweep + 1.5 h large-state pilot).

## Question

Phase-3 Rank-1 (`results/mtl_protocol_fix/phase3_rank1_findings.md`) promoted a log_T knowledge-distillation term at the reg head at single-seed=42 (+2.40/+5.06/+2.32 pp disjoint reg at AL/AZ/FL @ W=0.2). Does the effect survive multi-seed {0,1,7,100} at the small states (paper-grade n=20), and does it transfer to large states?

## Mechanism

A KD term is added to the reg-task loss: `task_b_loss = CE + W · τ² · KL(softmax(reg_logits/τ) ‖ softmax(log_T[last_region_idx]/τ))`, W=0.2, τ=1.0. The teacher is the train-only per-fold first-order region-transition log-prior `log_T`, indexed by the per-sample observed last region (`poi_0..poi_8` only, never `target_poi`). This is a *second* pressure on top of the head's existing additive `α·log_T` prior. Pad rows excluded. W=0.0 short-circuits the entire block (RNG-neutral baseline).

**Provenance:** the `--log-t-kd-weight` / `--log-t-kd-tau` flags and the KL term in `src/training/runners/mtl_cv.py` were implemented in this study (the Phase-3 numbers came from uncommitted local code). The mechanism is preserved verbatim from the findings doc. Implementation provenance is documented in `log.md` (2026-05-28) and is NOT a CONCERN per user direction.

## Method

- **Small-state sweep:** AL+AZ × seeds {0,1,7,100} × W ∈ {0.0, 0.2} × 5 folds = 16 cells (n=20 per state per W). H3-alt recipe. Paired Wilcoxon (one-sided, paired by seed×fold, W=0.2 > W=0.0) on disjoint reg `top10_acc_indist`. Honest paired baseline (W=0.0 re-run at the same 4 seeds).
- **Large-state pilot:** FL 5-fold + CA/TX 1-fold, seed=42 only (sign-and-magnitude pilot).
- Per-fold seed-tagged log_T verified fresh (C22/C4 mtime guard) before every cell.

## Results

**Small states (paper-grade, n=20):**

| State | n | mean Δ reg pp | folds + | Wilcoxon p (1-sided) | cat Δ (disjoint) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| AL | 20 | **+2.27** | 20/20 | **9.54e-07** | −0.20 | PROMOTE |
| AZ | 20 | **+4.91** | 20/20 | **9.54e-07** | +0.08 | PROMOTE |

Both clear the α=0.05 gate by ~5 orders of magnitude. Multi-seed reproduces single-seed=42 (+2.40/+5.06) within 0.15 pp — no dev-seed (C23) bias at small states. AZ's larger lift is dosage-on-headroom (AZ W=0.0 baseline 41.30 % vs AL 50.59 %), not a stronger structural shortcut (MI/H ratios within 4 pp; see leak audit §4).

> **Reproducibility note (scipy ≥ 1.16):** on raw per-fold values (no ties) `wilcoxon` selects `method='exact'` → p=9.537e-07 (2⁻²⁰). Re-running on 2-dp rounded Δs forces ties → `method='approx'` → p≈4.42e-05. Always use raw CSV values. (Verified in the Tier A1 verdict advisor pass.)

**Large states (seed=42 PILOT — NOT paper-grade):**

| State | n | W=0.0 disjoint reg | W=0.2 | Δ pp | cat Δ | test |
|---|---:|---:|---:|---:|---:|---|
| FL | 5 | 63.98 | 66.38 | **+2.40** | +0.01 | Wilcoxon p=0.031, 5/5 + |
| CA | 1 | 50.06 | 51.48 | **+1.42** | −0.10 | sign-and-magnitude |
| TX | 1 | 50.38 | 52.09 | **+1.71** | +0.05 | sign-and-magnitude |

Sign positive at all three; reg-only with flat cat (no leak signature). **W=0.0 baselines overshoot §0.1 multi-seed exactly as C23 predicts** (FL +0.7, CA +2.7, TX +7.5 pp of dev-seed bias) — large-state paper-grade requires {0,1,7,100}.

## Verdict

**PROMOTED at small states (paper-grade, n=20). TRANSFERS at large states (seed=42 pilot only).** Leak-audited clean across 7 vectors (`F_TIER_A1_LEAK_AUDIT.md`): reg-only lift + flat cat + fresh train-fold-only log_T + non-near-deterministic prior (MI/H ≈ 0.58) — the opposite of every historical leak signature.

## Framing (mandatory in paper prose)

- This is the **isolated** effect of the log_T-KD supervisory signal alone. It is **smaller than and distinct from** the §4.2 composite headline (+7–12 pp disjoint reg via STL c2hgi-cat + STL HGI-reg). Frame as "supervisory distillation of a train-only first-order region-Markov prior", not a novel architecture, and not the project's strongest reg lift.
- Value proposition: a **single-MTL-artefact** reg lift with **no deploy-time routing cost** (unlike the composite's two-model deploy), orthogonal to the B9 recipe — a free upgrade that stacks onto whatever architectural champion `mtl_improvement` lands.

## Closure table (Tier A1 scope, disjoint reg Acc@10)

| Metric | Canonical c2hgi MTL (W=0.0) | This study (W=0.2) | Δ pp | Source |
|---|---:|---:|---:|---|
| AL reg Acc@10 @ disjoint (n=20) | 50.59 | 52.85 | **+2.27** | `tier_a1/phase_a1_verdict.md` |
| AZ reg Acc@10 @ disjoint (n=20) | 41.30 | 46.22 | **+4.91** | `tier_a1/phase_a1_verdict.md` |
| FL reg Acc@10 @ disjoint (5-fold pilot) | 63.98 | 66.38 | +2.40 | `tier_a1_largestate/...addendum.md` |
| AL/AZ cat F1 @ disjoint | (baseline) | ≈ unchanged (−0.20 / +0.08) | — | `tier_a1/phase_a1_verdict.md` |
| reg Acc@10 @ STL ceiling | (unchanged) | — | — | RESULTS_TABLE §0.1 |

## Artefacts

- Small-state verdict: `results/substrate_protocol_cleanup/tier_a1/phase_a1_verdict.md`
- Large-state addendum: `results/substrate_protocol_cleanup/tier_a1_largestate/phase_a1_largestate_addendum.md`
- Leak audit: `findings/F_TIER_A1_LEAK_AUDIT.md`
- Implementation: `scripts/train.py`, `src/configs/experiment.py`, `src/training/runners/mtl_cv.py`, `tests/test_substrate_protocol_cleanup_flags.py` (TestLogTKD)
- Summarizer: `scripts/substrate_protocol_cleanup/summarize_tier_a1.py`
