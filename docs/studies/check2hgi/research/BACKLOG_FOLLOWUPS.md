# Backlog — Pending Follow-ups

**Date:** 2026-04-21. Consolidates all follow-ups mentioned but not executed, with priority ranking and paper-impact estimate.

## High priority (paper-blocking or paper-strengthening)

### B1. FL 1-fold MTL GETNext sanity check
- **Status:** Not yet run.
- **Why:** MTL GETNext lifted AL by +11 pp, AZ by +5.6 pp. On FL (127K rows, 4.7K regions), prior MTL STAN d=256 1f tied GRU on Acc@10 but lost MRR/Acc@5 — a graph prior should recover those. FL is the paper's headline state.
- **Command:** same as AZ MTL GETNext with `--state florida --folds 1`. Need `/tmp/check2hgi_data/check2hgi/florida/region_transition_log.pt` (not yet built).
- **Effort:** ~5 sec to build transition matrix + ~30 min training = ~30 min wall-clock.
- **Expected outcome:** +5-10 pp lift over MTL GRU (57.60 → 62-67 Acc@10_indist); meaningful if within σ of STL GRU (68.33).

### B2. FL 5-fold MTL GETNext (if 1-fold confirms)
- **Status:** Blocked by B1.
- **Why:** 5-fold gives σ for the paper headline. Only run if B1 shows a clear lift.
- **Effort:** ~6-10 h on MPS.

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
- **Status:** Not implemented. Current GETNext uses a learned soft probe on the last-step embedding.
- **Why:** Allows ablation "hard vs soft index" to show the probe isn't hiding leakage or learning the wrong thing.
- **Effort:** Extend `next_region.parquet` to include `last_region_idx` column (extend `build_next_region_frame` to compute from poi_8). Modify dataset/dataloader to pass the aux column. ~4-6 h including re-runs.
- **Expected outcome:** Either hard == soft (probe faithful; claim "our soft probe IS GETNext"), OR hard > soft (we should commit to hard).

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

## Prioritization rationale

1. **B1-B3** are paper-blocking for BRACIS: multi-seed headline + FL GETNext confirm the new best-MTL number.
2. **B4-B6** are paper-strengthening: close theoretical-soundness gaps (leakage-safe, faithful index, α interpretation).
3. **B7** is a quick variance-reduction experiment.
4. **B8-B11** are follow-up-paper material.

If next session has ~6h budget, recommended order: B1 → B6 (learned α) → B7 (ALiBi × GETNext) → B3 start.
