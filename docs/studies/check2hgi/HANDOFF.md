# Check2HGI Study — Session Handoff (2026-04-16, updated end-of-day)

## Status at a glance

| Phase | Status |
|---|---|
| **P0** | ✅ complete — integrity, simple baselines (region-level Markov added), CH14 audit |
| **P1** | ✅ complete — 5 heads × checkin/region/concat baselined on AL, GRU confirmed on AL 5f×50ep + FL 5f×50ep, `next_mtl` scrapped as region-head candidate |
| **P2-prep** | ✅ CGC/MMoE/DSelectK/PLE ported to TaskSet-aware |
| **P4-prep** | ✅ per-task input pipeline shipped (`FoldCreator.task_a_input_type` / `task_b_input_type`, CLI flags, `region_sequence.py`) |
| P2 | ready to launch — arch × optim grid with per-task input modality |
| P3 | blocked on P2 |
| P4 | unblocked (pipeline ready); gated on P3 champion |
| P5 | gated on P4 |
| P6 | research track — no blocker |

## Thesis (clarified 2026-04-16)

**Bidirectional.** MTL `{next_category, next_region}` must improve *both* heads over their single-task baselines. A one-sided lift does not satisfy the thesis. See CH01/CH02 in CLAIMS_AND_HYPOTHESES.md.

Architectural plan for P3/P4: **per-task input modality** — feed check-in embedding sequence to `category_encoder`, region embedding sequence to `next_encoder`, let the shared backbone bridge. Supersedes the earlier "dual-stream concat" framing of CH03.

## Task pair

**{next_category (7 classes), next_region (~1K classes)}** on Check2HGI check-in-level embeddings.

## Key numbers (2026-04-16)

| Metric | AL | FL |
|---|---|---|
| Check2HGI single-task next-cat macro-F1 | **38.67%** (5f×50ep) | TBD |
| Check2HGI single-task next-region Acc@10 (region-emb, `next_gru`, **5f×50ep**) | **56.94% ± 4.01** | **68.33% ± 0.58** |
| Check2HGI single-task next-region Acc@10 (region-emb, `next_tcn_residual`, 5f×50ep) | 56.11% ± 4.02 | — (ties GRU on AL, 20× faster) |
| Check2HGI single-task next-region Acc@10 (region-emb, `next_gru` scaled hd=384, 1f×50ep) | 54.68% (single-fold) | — |
| Check2HGI single-task next-region Acc@10 (check-in-emb, `next_gru` default, 1f×30ep) | 20.11% (below Markov!) | — |
| Check2HGI single-task next-region Acc@10 (concat input, `next_gru`, 1f×30ep) | 49.57% (worse than region-only) | — |
| **Single-task factor vs Markov-1-region** | **1.21× (AL)** | **1.050× (FL)** |
| POI-RGNN published next-cat (FL/CA/TX) | 31.8–34.5% | 34.5% |
| Simple floor: majority next-cat | 34.2% | 24.7% |
| Simple floor: Markov next-cat | 31.7% | 37.2% |
| **Simple floor: Markov-1-region next-region Acc@10** | **47.01%** | **65.05%** |
| Simple floor: Markov-2-region / Markov-3-region Acc@10 | 37.9% / 35.2% | 59.2% / 56.7% |

**Note:** the old `markov_1step` (POI-level) baseline (21.3% AL / 45.9% FL) had a ~50% fallback rate to top-k-popular — it was a degenerate baseline. The corrected `markov_1step_region` is the paper-reported floor. The POI-level entry is kept for continuity but not used as a floor.

## P1 findings (frozen)

1. **Head champions:** `next_gru` (AL 56.94 ± 4.01, FL 68.33 ± 0.58) and `next_tcn_residual` (AL 56.11 ± 4.02) tie statistically. TCN is ~20× faster per fold — preferred when compute is binding in P2.
2. **`next_mtl` (Transformer) scrapped as region head:** catastrophic failure on region task (AL 7.40 ± 1.72 vs GRU 56.94). Architectural mismatch — head was designed for 7-class next-category; its dropout=0.35 × 4 transformer layers × 9-position sequence × 1109/4702-class softmax don't converge in 50ep. Remains a valid candidate for the category (Task-A) slot where it's the legacy champion.
3. **Input modality dominates head choice.** Check-in-emb input caps all heads at ~20% Acc@10 (AL, at Markov floor). Region-emb input lifts `next_gru` to 53–57% Acc@10 on AL, 68% on FL.
4. **Concat input is worse than region-only** (49.57% vs 53.33% on AL 1f×30ep). Check-in context = noise for region target. → motivates per-task input modality (P4).
5. **CH04 retired as a gate.** Best neural head on AL is 1.21× Markov-1-region; on FL 1.050×. Does not meet the original 2× target. Demoted to a reported comparison, not a go/no-go.
6. **FL is a dense-data regime.** FL Markov-1-region already sits at 65.05% Acc@10 because (r_t, r_{t+1}) transitions have 85%+ coverage on 159K training rows. Neural generalisation over 9-step context adds ~3.3 pp. Implication for P3: the bidirectional MTL lift on FL is the binding constraint — AL has more headroom (~10 pp).
7. **Scaling `next_gru` (hidden_dim 256→384, num_layers 2→3, label_smoothing=0.1, 30→50ep)** adds ~+1.4 pp. Returns diminishing.

## Ready for P2

All P2 pre-reqs are met:

1. ✅ **TaskSet-aware MTL variants** (CGC/MMoE/DSelectK/PLE + FiLM) — verified legacy-bit-exact.
2. ✅ **Per-task input pipeline** — `FoldCreator.task_a_input_type` / `task_b_input_type`, `scripts/train.py --task-a-input-type --task-b-input-type`, `src/data/inputs/region_sequence.py`. All 121 prior tests pass.
3. ✅ **P1 baselines frozen** — region-task single-task benchmark locked for paired comparison against MTL.

## Open decisions for P2 launch

1. **Region head champion.** GRU (safe, well-characterised) vs TCN-residual (statistically equivalent, 20× faster). Recommend TCN for P2 grid, GRU for P3 headline.
2. **P2 grid scope.** arch ∈ {FiLM, CGC, MMoE, DSelectK, PLE} × optim ∈ {NashMTL, PCGrad, GradNorm, naive} = 20 configs × 2 states × 5f×50ep. AL alone ~10 h on M2 Pro (per fold ~6 min with TCN region head). Full matrix = ~20 h.
3. **P3 multi-seed count.** 3 seeds × 5 folds = 15 paired samples per comparison.
4. **P4 config matrix.** 4 input variants × multi-seed × 2 states = 12 runs at 5f×50ep. Budget ~8 h. Confirm all 4 or drop `shared_region` (would be bad for category).
