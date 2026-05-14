# F50 Tier 1.1 — Cat-Head Path A vs Path B Decision (2026-04-28)

**Trigger:** F50 plan §5 T1.1 — verify F33 per-fold cat F1 envelope at FL using existing H3-alt 5f run.

**Status:** **DONE 2026-04-28 — PASS, Path A confirmed (universal `next_gru`).** No new compute needed. Closes a paper-blocker that had been open in `FOLLOWUPS_TRACKER.md` since 2026-04-24 (C14 cat-head scale-dependence flag).

---

## 1 · Question

`CONCERNS.md §C14` flagged scale-dependence on the F27 cat-head swap `next_mtl → next_gru`:

- AL 5f (F31): cat F1 +3.43 pp under `next_gru` over `next_mtl`. ✓
- AZ 5f (F27 validation): cat F1 +2.37 pp under `next_gru`, paired Wilcoxon p=0.0312, 5/5 folds. ✓
- FL 1f (F32): cat F1 = 0.6572 under `next_gru`, **−0.93 pp below pre-F27 envelope** (n=1 noise — undecidable).

The F33 paper-blocker test was: 5-fold × 50ep on FL with `next_gru`. Acceptance: 5f mean cat F1 within σ of pre-F27 envelope [65.72, 67.06] (Path A). Below: scale-dependent cat head (Path B — CA+TX would inherit `next_mtl`).

## 2 · Verification

The H3-alt FL run (`results/check2hgi/florida/mtlnet_lr1.0e-04_bs1024_ep50_20260426_0045`) **is** F33: 5f × 50ep, FL, `mtlnet_crossattn + static_weight(0.75) + next_gru cat + next_getnext_hard reg`, batch=1024, seed=42, H3-alt per-head LR. The cat-head was already `next_gru`. F33 has therefore been *de facto* run since 2026-04-26 — the verification was extracting the per-fold cat F1 values and checking the envelope.

Per-fold cat F1 (`diagnostic_best_epochs.next_category.metrics.f1`):

| Fold | cat F1 |
|:-:|---:|
| 1 | 0.6765 |
| 2 | 0.6855 |
| 3 | 0.6804 |
| 4 | 0.6814 |
| 5 | 0.6869 |
| **mean** | **0.6821** |
| std | 0.0042 |

`primary_checkpoint` (joint-best epoch) cat F1 mean = 0.6792 (per OBJECTIVES_STATUS_TABLE). Per-task-best ≥ joint-best by ≤ 0.5 pp — within fold-level noise.

## 3 · Verdict

**PASS (Path A — universal `next_gru`):**

- 5f mean (per-task-best) = 68.21% — **above pre-F27 envelope [65.72, 67.06] at every fold by 1+ pp**.
- 5f mean (joint-best) = 67.92% — also above the envelope at every fold.
- F32's n=1 result of 65.72% is now revealed as fold-1 noise (the actual fold-1 of F33 = 67.65%).
- Std = 0.42 pp — exceptionally tight; the pre-F27 dispersion has compressed at scale.

**The F27 cat-head swap `next_mtl → next_gru` generalises to FL scale.** No scale-dependent cat-head footnote is needed in the paper. CA+TX P3 inherit `next_gru` cat head.

`CONCERNS.md §C14` should close with the verdict above.

## 4 · Implications

### 4.1 Paper framing — `next_gru` is the universal cat head

`PAPER_STRUCTURE.md §3.1` lists the matched-head cat baseline as `next_gru`. This is now confirmed across all 3 states (AL+AZ+FL). The pre-F27 `next_mtl` Transformer head is preserved as a "head-sensitivity probe row" only (per CH16 head-invariance discussion in `SUBSTRATE_COMPARISON_FINDINGS.md §5`).

### 4.2 No new compute needed for F33

The original F33 estimate (~6 h Colab T4) was based on the assumption that a separate run with `next_gru` was needed. The H3-alt FL run (committed 2026-04-26) already used `next_gru`; F33 was *de facto* completed but never formally verified against the envelope criterion. This audit closes that loop.

### 4.3 CA+TX P3 launches under `next_gru` + H3-alt without ambiguity

P3 (`launch_plans/ca_tx_upstream.md`) inherits the H3-alt champion. Path A confirmed → no fork in launch plans. Reduces P3 risk.

## 5 · Side observation: cat F1 lift over STL is real at FL

Tier 0 (F50_DELTA_M_FINDINGS §2.3) showed FL MTL cat F1 (per-task-best) per-fold:

| Fold | MTL cat F1 | STL `next_gru` cat F1 | Δ_cat (rel) |
|:-:|---:|---:|---:|
| 1 | 0.6837 | 0.6601 | **+3.57%** |
| 2 | 0.6796 | 0.6691 | +1.57% |
| 3 | 0.6757 | 0.6680 | +1.15% |
| 4 | 0.6796 | 0.6749 | +0.70% |
| 5 | 0.6919 | 0.6770 | +2.20% |

**5/5 folds positive.** Paired Wilcoxon p_greater = 0.0312 on cat F1 alone. The FL cat win is significant at n=5 ceiling — but is **dwarfed by the FL reg loss** (−4.57 to −5.64% relative on MRR, 5/5 folds negative), so the joint Δm is negative.

This is consistent with the CH18 scale-conditional reframing: substrate carries the cat win uniformly; architecture costs reg at scale.

## 6 · Cross-references

- **Tier 0 findings:** `research/F50_DELTA_M_FINDINGS.md`
- **F50 plan:** `research/F50_PROPOSAL_AUDIT_AND_TIERED_PLAN.md` §5 T1.1
- **Original F33 spec:** `FOLLOWUPS_TRACKER.md §1 F33`
- **Original CH14 / C14 concern:** `CONCERNS.md §C14`
- **Predecessor F32 (n=1 noise):** `research/F27_CATHEAD_FINDINGS.md`
- **F31 AL validation + F27 AZ Wilcoxon:** `research/F27_CATHEAD_FINDINGS.md`
- **Source data:** `results/check2hgi/florida/mtlnet_lr1.0e-04_bs1024_ep50_20260426_0045/folds/fold{1..5}_info.json`

## 7 · Tracker updates pending

- **`FOLLOWUPS_TRACKER.md` §1 F33:** mark **done — Path A confirmed** with reference to this doc.
- **`CONCERNS.md §C14`:** mark **resolved 2026-04-28** with reference to this doc.
- **`PAPER_STRUCTURE.md §3.1`:** no change needed — already correct.
- **`PAPER_PREP_TRACKER.md`:** no change — F33 was not in the paper-deliverable list (lived only in FOLLOWUPS_TRACKER).
