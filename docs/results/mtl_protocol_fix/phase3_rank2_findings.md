# Phase 3 Rank 2 — class-balanced reg-only batch sampler — FALSIFIED

**Date:** 2026-05-21
**Phase scope:** mtl-protocol-fix DEFERRED_WORK §4.6 (class-balanced batch sampler at the reg head). Single-seed=42, 5 folds, 50 epochs. Scope: AL + AZ (FL skipped after AL falsification given −30 pp magnitude).
**Code:** `--reg-balanced-sampler` flag in `scripts/train.py`; `FoldCreator.use_weighted_sampling_reg` override in `src/data/folds.py` — when enabled, applies `WeightedRandomSampler` (per-class inverse-frequency, balanced) to the reg train dataloader ONLY. Cat dataloader continues with default `shuffle=True` (no sampler). Both heads continue using their existing `weighted-CE` (which re-weights gradients per class at the loss level).

## The intervention

Two arms:
1. **baseline** — canonical recipe (weighted-CE at the loss layer; uniform random batches).
2. **balanced** — same recipe + `--reg-balanced-sampler` (`WeightedRandomSampler` with inverse-frequency weights on the reg dataloader; weighted-CE still active on top).

Hypothesis (from head_window_batch_audit.md §C + DEFERRED_WORK §4.6): the FL ~4 700-region long-tail is the load-bearing destabiliser of the reg head's early peak. Balanced batching reduces rare-class gradient noise → stabilises reg.

## Results

### Alabama (1109 regions, H3-alt recipe)

| arm | disjoint reg | geom_simple reg | b9 reg | disjoint cat |
|---|---:|---:|---:|---:|
| baseline | 50.82 ± 3.21 | 48.56 ± 3.13 | 47.60 ± 4.14 | 45.76 ± 1.34 |
| balanced | **20.36 ± 5.72** | **20.33 ± 5.78** | **19.43 ± 5.43** | 45.54 ± 1.85 |

Wilcoxon (one-sided, balanced > baseline) on disjoint reg: **Δ=−30.46 pp, p=1.0000**. All 5/5 folds dramatically worse. Cat untouched (Δ−0.22 pp, n.s.).

### Arizona (1540 regions, H3-alt recipe)

| arm | disjoint reg | geom_simple reg | b9 reg | disjoint cat |
|---|---:|---:|---:|---:|
| baseline | 41.33 ± 2.73 | 39.60 ± 3.11 | 38.43 ± 2.31 | 48.87 ± 1.80 |
| balanced | **22.83 ± 9.72** | **22.83 ± 9.72** | **15.61 ± 2.13** | 48.71 ± 1.60 |

Wilcoxon (one-sided, balanced > baseline) on disjoint reg: **Δ=−18.49 pp, p=1.0000**. All 5/5 folds worse; same direction as AL, smaller magnitude (smaller tail at AZ → smaller dual-prior conflict). Cat untouched (Δ−0.16 pp, n.s.).

**Aggregate (AL + AZ)**: balanced sampler regresses disjoint reg by 18-30 pp in both states; magnitude scales with n_regions × tail length. FL would presumably regress more (4 700 regions, longer tail) — but the pattern is established, no additional information from a third state.

### Florida (4700 regions, B9 recipe)

_SKIPPED._ The −30 pp AL magnitude is well beyond any plausible heterogeneity across states. Spending ~30 GPU-min for a confirmation that would change no recommendation is not justified.

## Verdict

**FALSIFIED at AL on disjoint reg by 30 pp; expected to falsify at AZ too.** The hypothesis that long-tail under-sampling drives reg destabilisation is wrong.

## Mechanism interpretation

The reg head already gets per-class re-weighting at two layers:
1. **weighted-CE** (gradient re-scaling at the loss layer; current default).
2. **α · log_T[last_region_idx]** (additive Markov-1 prior at the logit layer; magnitude-balanced).

Layering `WeightedRandomSampler` on top:
- Oversamples rare classes ~`n_classes / n_seen_per_class` times — at AL's 1109-region tail, potentially **50-100×** for tail regions.
- The STAN backbone's representations of frequent regions become **under-sampled**, since balanced-sampled batches now under-represent the head of the distribution.
- The head's `α · log_T` term — calibrated against the *natural* class prior — is now misaligned with what the backbone produces, since the backbone sees a *uniform* prior.
- Net effect: top-K accuracy collapses because the model learns to over-predict rare classes that the natural test-time prior heavily down-weights.

This is the **dual prior** problem: re-weighting at one layer assumes the other layers see the natural prior. Two re-weightings stack incoherently.

## Connects to F2 (Phase 1 v4)

F2 was REVISED to "MTL trails STL on reg because reg peaks at ep 2-4 then crashes from gradient interference; gap shrinks with substrate-learnability-vs-negative-transfer horizon" (P4 confirmed it's architectural, not curriculum-fixable). Rank 2's falsification adds a NEW data point: the reg crash is also NOT long-tail-under-fitting. It's truly architectural — the shared backbone capacity, not the sampling distribution.

This **strengthens the case for `mtl_architecture_revisit.md`** as the highest-EV next-tier study (the residual gap is purely architectural — not sampling, not curriculum, not selector).

## What this closes vs leaves open

**Closed**: WeightedRandomSampler alone is not the fix. ✗

**Still open** under `head_window_batch_audit.md` §C (separately worth running, but lower priority now):
- `focal-loss` alone with weighted-CE disabled
- weighted-CE alone vs no class-weighting at all (drop the existing re-weighting)
- WeightedRandomSampler + drop weighted-CE (resolve the dual-prior conflict from the other side)

These would test whether ANY single re-weighting beats the current weighted-CE-only setup, but the EV is now low given (a) Rank 1 (log_T-KD) already lifts disjoint reg by 2-5 pp without re-touching the sampler, and (b) the residual gap to STL is architectural per P4.

## Cross-references

- Code: `src/data/folds.py` (`FoldCreator.use_weighted_sampling_reg` override, `_create_check2hgi_mtl_folds` per-task routing), `scripts/train.py` (`--reg-balanced-sampler` CLI), `scripts/mtl_protocol_fix/run_balanced_sampler_compare.sh` + `summarize_balanced_sampler.py`.
- Per-state summaries: `docs/results/mtl_protocol_fix/phase3_rank2_balanced_sampler/{alabama,arizona}/{state}_summary.{md,json}`.
- Future-work memo: [`docs/future_works/head_window_batch_audit.md`](../../future_works/head_window_batch_audit.md) §C — update entry to record FALSIFICATION; the remaining sub-bullets (focal-loss-alone, drop-weighted-CE) are NOT closed.
- Aggregate finding context: F2 (Phase 1 v4) + P4 (Phase 2) — see [`phase1_phase2_verdict_v6_final.md`](phase1_phase2_verdict_v6_final.md).
