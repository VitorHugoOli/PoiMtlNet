# Backlog — Pending Follow-ups

**Date:** 2026-04-21 (updated 2026-04-22 end of B5 session). Consolidates all follow-ups mentioned but not executed, with priority ranking and paper-impact estimate.

## Status snapshot — end of 2026-04-22 session

- ✅ B1, B5, B6, B7 COMPLETED. B5 retraining (hard index) delivered
  +6.59 pp Acc@10 on AZ and revealed a scale-dependent trade-off on FL.
  See `B5_RESULTS.md`, `B5_MACRO_ANALYSIS.md`, `B5_FL_SCALING.md`.
- ❌ B2 superseded: FL 5-fold MTL-GETNext-soft not yet run; FL 1-fold
  already done as B-M13 (soft champion, 60.62 Acc@10). Priority lowered
  now that FL-hard 1f is also in hand.
- ❌ B3 (multi-seed n=3) still open — **now paper-blocking** after the
  B5 scale-dependent finding. Seed 42 n=1 across AL/AZ/FL isn't enough
  to defend the "scale-dependent champion" claim to a reviewer.
- ❌ B4 (per-fold transition matrix) still open — low-priority hygiene
  fix.
- ❌ NEW B12 (FL 5-fold hard) — high priority given the n=1 finding.
- ❌ NEW B13 (FL task-weight rebalancing) — tests whether the cat
  task can be rescued via `task_b_weight < 1`.

## High priority (paper-blocking or paper-strengthening)

### B1. FL 1-fold MTL GETNext sanity check
- **Status:** ✅ Completed 2026-04-21 (B-M13 soft, 1f × 50ep FL, Acc@10=60.62).
- See `GETNEXT_FINDINGS.md` + `RESULTS_TABLE.md` B-M13.

### B2. FL 5-fold MTL GETNext-soft (if 1-fold confirms)
- **Status:** Superseded / not-run. 1-fold soft delivered 60.62 Acc@10
  which closed the MTL-STL gap meaningfully. 5-fold would tighten σ
  but is now lower-priority than the hard variant (B12).

### B3. Multi-seed (n=3) headline for FL/CA/TX
- **Status:** Documented in `phases/P7_headline_states.md §11` on 2026-04-21; not executed.
- **Why:** Reviewer stability check; σ over seeds is more conservative than σ over folds alone.
- **Effort:** 2× §2.4 per state = 12–20 h extra per state.
- **Seeds:** 42 (existing), 123, 2024.
- **Priority:** must-do for submission; can run across CA + TX machines in parallel.

## Medium priority (paper-nice-to-have)

### B4. Per-fold transition matrix (leakage-safe GETNext)
- **Status:** Not implemented. Current GETNext uses a single transition matrix built from ALL training data.
- **Why:** Val users' training segments contribute to transition counts — mild leakage. A per-fold build removes this.
- **Effort:** Modify `scripts/compute_region_transition.py` to accept fold train indices; call it at fold-setup time in training loop; head reloads per fold. ~4 h + re-runs of AL/AZ MTL GETNext (~2 h).
- **Expected outcome:** Similar mean, possibly slightly lower (more honest). Paper can claim "per-fold leakage-safe".

### B5. Hard-indexed `last_region_idx` (faithful GETNext)
- **Status:** ✅ Completed 2026-04-22. Implemented (commits `6a2f808`
  + `ea65fb3`), retrained on AL + AZ 5f × 50ep + FL 1f × 50ep.
- **Results:** AL +1.47 pp Acc@10 (within σ), AZ **+6.59 pp** (outside
  σ, paper headline), FL 1f mixed (region wins 4/5 metrics but cat
  F1 drops −10.58 pp due to scale-induced gradient imbalance).
  Full write-ups: `B5_RESULTS.md` (AL+AZ), `B5_FL_SCALING.md` (FL),
  `B5_MACRO_ANALYSIS.md` (cross-method comparison), `B5_HARD_VS_SOFT_INFERENCE.md`
  (inference-time ablation that motivated the retraining).
- **Knock-on:** AZ MTL-hard Acc@10 (53.25) now BEATS AZ STL STAN (52.24) —
  first time MTL surpasses STL on region Acc@10 in this study.

### B6. Learned α inspection (GETNext / TGSTAN / STA-Hyper)
- **Status:** ✅ Completed 2026-04-21. See `GETNEXT_FINDINGS.md §α inspection`.
- Original gating note (kept for reference):
- **Why:** Quantify how much weight the graph prior receives (α) across states. If α→0 on AZ and α→1 on AL, tells us when the prior matters.
- **Effort:** Re-run one fold with checkpoints enabled per head family per state, load state dict, print α. ~30 min of compute + 30 min analysis.
- **Expected outcome:** α > 0.3 on AL for GETNext; TBD on AZ. Paper gets a one-line figure or claim.
- **Risk:** may require adding a CLI flag or modifying `_default_checkpoint_callbacks` to write minimal checkpoints (just final epoch) for this purpose.

### B7. GETNext + ALiBi init combination
- **Status:** ✅ Completed 2026-04-22 on AL. See `B7_ALIBI_GETNEXT_FINDINGS.md`.
- **Result:** +1.08 pp Acc@10 (56.38 → 57.46) and −11% σ (4.11 → 3.66). Lift within σ; documented as optional paper stabilizer, not a default.
- **AZ still open.** Low value given AZ σ on GETNext is already 2.93 (ALiBi stabilizer effect diminishes).

## Low priority (research / future work)

### B8. TGSTAN and STA-Hyper full reproductions
- **Status:** Pragmatic adaptations exist; not faithful reproductions.
- **Why:** Our versions lack raw Δt/Δd (TGSTAN's bilinear ST bias) and full hypergraph construction (STA-Hyper's hyperedge convolution).
- **Effort:** Major — would require extending data pipeline to include raw timestamps + coords per step, and writing hypergraph code. ~2-3 weeks of work.
- **Expected outcome:** Unknown; could be similar to our adapted versions or meaningfully different.
- **Recommendation:** Defer to a follow-up paper.

### B9. GETNext with true flow-map + Φ = (Φ₁1ᵀ + 1Φ₂ᵀ) ⊙ (L̃ + J) 
- **Status:** Our adaptation skips the left-hand Φ₁/Φ₂ projections. We just use `log T` directly.
- **Why:** The original GETNext's Φ combines learned POI/user projections with the Laplacian. We could add this with minimal cost.
- **Effort:** ~50 LOC + re-run. ~30 min.
- **Expected outcome:** Unknown; could lift a bit more.

### B10. PIF-style user-specific region frequency prior
- **Status:** Not implemented. Mentioned in STAN paper.
- **Why:** Tests whether personalized frequency (how often a user visits each region) adds signal beyond the global transition prior.
- **Effort:** Build per-user region frequency at training time, pass to head (needs user ID column in batches). ~3-4 h.
- **Expected outcome:** Could help on AZ/FL where user-level habits are repeated. Medium-priority research question.

### B11. DataAug: trajectory augmentation for small-data AL
- **Status:** Not considered.
- **Why:** AL σ on d=256 STAN was 10.09 pp — per-fold variance from 2.5K val/fold. Augmenting by bootstrap or user-permutation could help.
- **Effort:** 2-4 h depending on strategy.
- **Expected outcome:** Lower σ, same mean.

### B12. FL 5-fold MTL-GETNext-hard ⭐ NEW (from B5 FL scaling finding)
- **Status:** Not yet run. Motivated by B5 1f × 50ep showing mixed
  signal (region Acc@5 +13.53 pp but cat F1 −10.58 pp).
- **Why:** Need 5-fold σ to decide whether the cat regression is real
  or n=1 fold-selection noise. If real, the paper must NOT report FL
  hard as headline; if noise, FL-hard might align with AZ's clean lift.
- **Effort:** ~5-6h on MPS (FL 5f × 50ep, similar batch count to B-M13).
- **Blocks:** paper's scale-dependent-champion claim.

### B13. FL task-weight rebalancing ⭐ NEW (from B5 FL scaling finding)
- **Status:** Not yet run.
- **Why:** The hypothesis behind FL cat-task regression is that `α·log_T`
  over 4703 regions saturates task_b gradient, starving task_a. A lower
  `task_b_weight` in PCGrad could rebalance. Tests whether FL-hard can
  deliver region lift AND preserve cat.
- **Plan:** sweep `task_b_weight ∈ {0.25, 0.50, 0.75}` on FL 1f × 50ep.
- **Effort:** 3 × ~45 min = ~2.25h.
- **Paper value:** if one weight restores cat to ≥62 F1 while keeping
  Acc@5 lift, FL-hard becomes the headline. Otherwise document the
  asymmetry as a scale-dependent finding.

### B14. Paired significance test on AZ B5 Δ ⭐ NEW (paper-strengthening)
- **Status:** Not yet run.
- **Why:** Claim is "MTL-hard (B-M9d) beats MTL-soft (B-M9b) on AZ by
  +6.59 pp Acc@10". A paired Wilcoxon signed-rank test across folds
  would give a p-value. At n=5 with all 5 fold-deltas positive, p is
  almost certainly < 0.05.
- **Effort:** 30 min (CPU script reading both run's fold json files).
- **Paper value:** defendable significance claim in the paper narrative.

## Already completed in session (for reference)

| ID | What | Status |
|---|---|---|
| — | STAN head implementation | ✅ |
| — | Critical review + SOTA survey | ✅ |
| — | AZ Markov baselines (fills B-B7 TBD) | ✅ |
| — | ALiBi init scale-dependent finding | ✅ |
| — | GETNext head implementation | ✅ |
| — | MTL GETNext AL + AZ: +5 to +11 pp lift | ✅ |
| — | TGSTAN head implementation | ✅ |
| — | STA-Hyper head implementation | ✅ |
| — | STL TGSTAN + STA-Hyper on AL + AZ | ✅ |
| — | MTL TGSTAN + STA-Hyper on AL + AZ | 🔄 running |

## Prioritization rationale (updated 2026-04-22)

1. **B3 + B12 + B14** are now paper-blocking for BRACIS: multi-seed
   headline + FL 5f hard + significance test tighten the post-B5
   champion claims on AZ and close the FL scaling ambiguity.
2. **B13** is paper-nice-to-have: if cat-task rebalancing works, FL
   hard could become the headline on all three states.
3. **B4** is a theoretical-soundness fix that can wait for the
   camera-ready.
4. **B8-B11** are follow-up-paper material.

If next session has ~6h budget, recommended order:
  1. **B14** (paired Wilcoxon on AZ B-M9b vs B-M9d, ~30 min CPU) —
     cheapest way to strengthen the AZ claim for the paper.
  2. **B13** (FL task-weight sweep, ~2.25h) — tells us whether FL
     hard can be rescued; avoids the ambiguity in B12.
  3. **B12** (FL 5-fold hard, ~5-6h) — only if B13 doesn't clearly
     rescue OR the paper needs 5-fold σ on FL-hard regardless.
  4. **B3** (multi-seed n=3 on AL+AZ+FL for champion configs) —
     parallelisable across machines; plan launches on 4050 + M4 Pro.
