# Tier C FL verdict — protocol coherence at FL scale (C1 routing / C2 reg-freeze / C3 zero-cat-kv)

**Date**: 2026-05-29
**Scope**: Florida, seed=42, 5 folds, **B9 large-state recipe** (cosine max-lr 3e-3, alternating-optimizer-step, alpha-no-weight-decay, min-best-epoch 5), `--engine check2hgi`, `--no-checkpoints`.
**Baseline**: the FL canonical Tier B MTL cell (`tier_b_fl/mtl_canonical`, disjoint reg 63.98, disjoint cat 70.49 — matches `phase_b_fl_3way.md` §2).
**Method**: RAW per-fold values, paired Wilcoxon (scipy exact). reg tested one-sided `cell < baseline` (does the intervention HURT reg?); cat one-sided `cell > baseline` (does it IMPROVE cat?). Two fronts (disjoint + joint/geom_simple). Analyser: `scripts/substrate_protocol_cleanup/analyze_tier_c_fl.py`.

**Verdict in one line:** **C2 §4.4-closed and C3 P4-closed both HOLD at FL scale on both fronts** — no N gives a significant cat gain without a reg regression; zeroing the cat K/V path does not move reg. (C1 — see §C1 below; routing cell run this session.)

---

## C2 — `--reg-freeze-at-epoch N ∈ {2,4,6}` (§4.4 freeze-reg-after-peak)

| Cell | reg DISJOINT (Δ, p_hurt) | cat DISJOINT (Δ, p_gt) | reg JOINT (Δ) | cat JOINT (Δ) |
|---|---:|---:|---:|---:|
| canonical baseline | 63.98 | 70.49 | 61.14 | 66.98 |
| **C2 N=2** | 56.29 (**−7.69, p=0.0312**) | 70.47 (−0.02, p=0.69) | 56.24 (−4.91) | 70.47 (+3.49) |
| **C2 N=4** | 63.86 (−0.12, p=0.31) | 70.44 (−0.05, p=0.78) | 63.63 (+2.48) | 70.27 (+3.29) |
| **C2 N=6** | 63.93 (−0.05, p=0.31) | 70.50 (+0.01, p=0.50) | 62.43 (+1.28) | 70.31 (+3.33) |

reg disjoint Δ per-fold: N=2 `[−9.79,−11.83,−5.87,−5.28,−5.67]`; N=4 `[0.0,−0.14,−0.46,−0.30,+0.31]`; N=6 `[0.0,−0.02,−0.22,−0.30,+0.30]`.

### Decision gate: does Δcat improve at any N WITHOUT Δreg regression > σ_fold?

**NO at every N — identical to AL/AZ.**
- N=2 freezes BEFORE the FL reg peak (FL peaks later than small states) → reg collapses −7.69 pp (p=0.0312) for a null cat Δ (−0.02).
- N=4 / N=6 preserve disjoint reg (Δ within −0.12 pp, ns) but cat Δ is null (≤+0.01 pp, p≥0.5).
- The joint-front "cat +3.3 pp" is the **geom-selection artefact** (freezing reg early flattens the reg curve, so the geom-max epoch lands at a higher-cat / lower-reg point) — NOT a real cat gain; disjoint cat is flat (≤−0.05 pp).

**VERDICT: C2 ARCHIVE confirmed at FL. §4.4 closed at large-state scale.** The asymmetric freeze-reg-after-peak curriculum buys no cat gain at any N without a reg cost, exactly as at AL/AZ.

---

## C3 — `--zero-cat-kv` (P4 residual K/V capacity-stealing test)

| Cell | reg DISJOINT (Δ, p_hurt) | cat DISJOINT (Δ) | reg JOINT (Δ) | cat JOINT (Δ) |
|---|---:|---:|---:|---:|
| canonical baseline | 63.98 | 70.49 | 61.14 | 66.98 |
| **C3 zero-cat-kv** | 64.01 (**+0.03, p=0.78**) | 70.38 (−0.12) | 60.72 (−0.43, p=0.31) | 67.21 (+0.23) |

reg disjoint Δ per-fold: `[−0.14,+0.01,+0.16,−0.08,+0.22]`.

### Decision gate: does reg peak LATER or improve in magnitude under K/V-zeroed cat path?

**NO — identical to AL/AZ.** Zeroing the cat-stream K/V feeding `cross_ba` shifts disjoint reg by +0.03 pp (ns, p=0.78) and joint reg by −0.43 pp (ns); no later peak, no magnitude gain.

**VERDICT: C3 P4 FULLY CLOSED confirmed at FL.** The residual MTL-vs-STL reg gap is NOT cross-attention K/V capacity-stealing at FL scale — consistent with the isolation cell (`phase_b_fl_3way.md` §3), which localises the MTL-reg limitation to the joint-training regime / α·log_T anchor dominance, not the cat→reg attention path.

---

## C1 — 3-snapshot routing (variant A)

Fresh FL B9 run with `--save-task-best-snapshots`, then `route_task_best.py` per fold with the landed modality fix (`--task-b-input-type region` passed explicitly and confirmed persisted in config: task_a=checkin, task_b=region), snapshots deleted after scoring. Decision-gate metric = reg routing gain (reg_best vs joint_best Acc@10).

| Metric | routed mean | joint_best mean | Δ (routed−joint) | per-fold Δ | folds+ | Wilcoxon p_gt |
|---|---:|---:|---:|---|---:|---:|
| **reg Acc@10** | 61.29 | 58.49 | **+2.80** | [1.57,2.88,2.68,2.98,3.87] | **5/5** | **0.0312 ✓** |
| cat F1 | 70.15 | 68.17 | +1.98 | [2.17,1.94,1.91,1.91,1.96] | 5/5 | 0.0312 ✓ |

**FL C1 CLEARS the +2 pp reg routing gate** (+2.80 pp, 5/5 folds, p=0.0312) — and cleanly, with NO degenerate snapshot (unlike AL, which failed on a single fold-3 Acc@1-selected degenerate reg-best). At FL the reg-best snapshot routes +2.80 pp over the deployable joint-best on the correct region modality. This is the THIRD state to pass (AZ +2.54 passed, AL failed, FL +2.80 passes); per variant-A framing this strengthens the §Discussion case for per-task 3-snapshot routing — though the cross-state pattern (AL's degenerate fold) still argues for an Acc@10-aligned reg-best selector + degenerate-snapshot guard before any promotion. Cross-ref `tier_c/phase_c_verdict.md` §C1.

Note the regime: routed reg ~61 % here is the joint-best-deployable regime; the disjoint-oracle reg is ~64 % (§2 of `phase_b_fl_3way.md`). Routing recovers ~2.8 pp of the disjoint↔joint deployable gap by reading the reg head at its own best epoch.

---

## Artefacts

- C2/C3 cells: `docs/results/substrate_protocol_cleanup/tier_c_fl/{c2_n2,c2_n4,c2_n6,c3_zerokv}/florida/seed42/.../metrics/`
- Analysis JSON: `tier_c_fl/tier_c_fl_analysis.json`
- Analyser: `scripts/substrate_protocol_cleanup/analyze_tier_c_fl.py`
- C1: `tier_c_fl/c1_route/florida/seed42/` (this session)
