# Future Work — Head, sequence-window, and batch-balance audit

**Date drafted:** 2026-05-20
**Updated:** 2026-05-28 — partial status update:
- **§A head re-design:** absorbed into [`docs/studies/mtl_improvement/`](../studies/mtl_improvement/) T7 (head re-ablation under the new backbone). Co-designed with arch.
- **§B window/mask audit:** moved to [`docs/studies/substrate-protocol-cleanup/`](../studies/substrate-protocol-cleanup/) Tier D as a cheap no-GPU pilot. ~1 day calendar.
- **§C batch class-balance:** **PARTIALLY FALSIFIED** by `mtl-protocol-fix` Phase 3 Rank 2. The `WeightedRandomSampler` variant regresses disjoint reg by **−18 to −30 pp at AL/AZ** (Wilcoxon p=1.0000 in both states; FL skipped). See [`docs/results/mtl_protocol_fix/phase3_rank2_findings.md`](../results/mtl_protocol_fix/phase3_rank2_findings.md). The focal-loss-only variant of §C remains untested, but the sampler form is closed.

**Source:** [`docs/studies/mtl-exploration/considerations.md`](../studies/mtl-exploration/considerations.md) (user bullets on heads, window/mask, batch balance)
**Sequencing:** §A in `mtl_improvement`; §B in `substrate-protocol-cleanup`; §C focal-loss variant only — sequence with whichever study reopens reg-head supervision next.

## What's deferred

Three small but cross-cutting audits the user has flagged but never executed under the current leak-free protocol:

### A. Head re-design and per-task best-fit

User's framing:
> "the previous head experiments for the check2hgi for the next-reg and next-cat was very superficial, and the results weren't concrete since the fusion and HGI each had a better fit with a STL head. I believe that with the new check2hgi we should be more rigorous and re-try the STL heads for these both heads at the same time we should evaluate on how these STL heads are implemented; maybe for each task we should have different variations to best fit they need."

Concrete:
- `next_gru` (cat head) and `next_stan_flow`/`next_getnext_hard` (reg head) were grandfathered from the P1 ablation under the pre-2026-05-15-leak protocol.
- Head-to-head under per-fold log_T (leak-free) is missing for: `next_lstm`, `next_transformer_pf`, `next_getnext` (no hard-neg), `next_stan_baseline`, `next_single`.
- **Per-task fit**: which head is best on cat vs reg may diverge; current shipping treats them as a single grid.
- **"A melhor cabeça muda com o MTL"** (from `considerations.md`): the best STL head may not be the best MTL head — needs explicit STL ⊕ MTL coupling.

### B. Sequence-window construction + mask correctness audit

User's framing:
> "Are the windows sequence of the next-reg been created correctly and the mask in the code been applied correctly?"

Concrete:
- `src/data/inputs/core.py` — `generate_sequences` builds non-overlapping sliding windows of size 9 + 1 target, short sequences padded with -1. Verify there is no leak (target check-in in input window).
- Causal masking in `NextHeadMTL` (`src/models/heads/next.py`) — verify causal-mask construction over window of size 9; assert no future-leak from position i to position j > i.
- `task_a_input_type=checkin` vs `task_b_input_type=region` — confirm both modalities use the same masking discipline.
- **For next-region specifically**: confirm `region_idx` target construction (`scripts/regenerate_next_region.py`) is per-check-in and not per-POI, and that the per-fold log_T comes from train-only transitions for the matching seed/fold.

### C. Batch class-balance experiment

User's framing:
> "the dataset during the batch has different distribution of the tasks, maybe we should try to balance it more and see if it has an effect on the performance of the model... oversampling or undersampling the data for each task or class."

Concrete:
- Current shipping uses **weighted CE** (class weights computed in `src/data/folds.py`). Never compared head-to-head with a class-balanced batch sampler or focal-loss-only weighting at FL.
- **Long-tail hypothesis**: FL has ~4 700 regions; head/torso/tail split is highly skewed. Both reg_macro_f1 collapse (CONCERNS C21) and reg head plateau may be downstream of long-tail under-fitting.
- **Per-task vs per-class balance**: balancing samples per task (cat vs reg loss weight) is already exposed via `--category-weight`; balancing samples per class within each task is the unexplored axis.
- Existing scaffolding: `src/losses/focal.py` (focal loss with γ=2.0) is implemented and registered but **never run cleanly under per-fold log_T** for the reg head.

## Why deferred

1. All three are diagnostic / variance-source audits, not headline lifts. They are best executed in series WITH the MTL-architecture revisit (each architecture may interact differently with head choice, window discipline, batch sampling).
2. EV depends on what `mtl-protocol-fix` reveals — if F1 closes the gap, the audits become polish; if F1 leaves a 5+ pp residual, they become candidates for the load-bearing fix.
3. The window/mask audit is the cheapest (~1 day, no GPU) and could be promoted as a pre-flight gate for the next study if the user wants confidence sooner.

## Acceptance criterion

When picked up:

1. **A — Head sweep**: per-task best head identified at FL/AL/AZ under per-fold log_T; paper-side table (cat-best STL head, reg-best STL head, cat-best MTL head, reg-best MTL head) populated.
2. **B — Window/mask audit**: written audit report at `docs/studies/...AUDIT.md` confirming no leak in window or causal-mask construction; OR identifying a leak and producing a fix + before/after numbers.
3. **C — Batch balance**: A/B test of weighted-CE vs class-balanced sampler vs focal-loss at FL × 3 seeds × 5 folds; verdict (significant lift / null / regression) documented.

## Cost (estimated)

- A — Head sweep: ~10-15 GPU-h (5 heads × 5 seeds × 5 folds × MTL+STL at FL).
- B — Audit: ~1 day no GPU.
- C — Batch balance: ~5-8 GPU-h at FL alone.
- **Total: ~15-25 GPU-h + ~1 week calendar.**

## Live docs the work would touch

- `src/data/inputs/core.py` — possible window/mask fix
- `src/data/folds.py` — possible batch sampler change
- `src/losses/focal.py` — possible default change
- `docs/findings/MTL_FLAWS_AND_FIXES.md` — entry for any leak found in audit B
- `docs/results/P1/` — head-sweep results
- `docs/CHANGELOG.md` — timeline entry

## Pointers

- User's window-mask concern: [`docs/studies/mtl-exploration/considerations.md`](../studies/mtl-exploration/considerations.md)
- User's batch-balance concern: same file
- Head registry: `src/models/heads/`
- Leak-fix point: `src/training/runners/mtl_cv.py` n_splits guard (2026-05-15)
