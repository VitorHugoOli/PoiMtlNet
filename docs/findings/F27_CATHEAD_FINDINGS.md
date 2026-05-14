# F27 — Cat-Head Ablation on MTL-B3

**Date:** 2026-04-24. **Tracker:** `FOLLOWUPS_TRACKER.md §F27`. **Sources:** `results/F27_cathead_sweep/az_1f50ep_cathead_*.json`. **Script:** `scripts/run_f27_cathead_sweep.sh`.

## Headline

**`next_gru` beats B3's current `next_mtl` (Transformer) as the task_a head on both cat and reg at AZ 1-fold.** +2.69 pp cat F1, +5.24 pp cat Acc@1, +3.83 pp reg Acc@10. `next_stan` is a close 2nd with a slight edge on reg MRR. LSTM is strictly worst (−5 pp cat F1).

## Protocol

- **Fixed config:** MTL-B3 (`mtlnet_crossattn + static_weight(category_weight=0.75) + next_getnext_hard d=256, 8h`).
- **Varied:** task_a head (cat head) across `{default=next_mtl, next_mtl, next_gru, next_lstm, next_stan, next_tcn_residual, next_temporal_cnn}`.
- **State / folds / epochs:** AZ 1-fold × 50 epochs, seed 42.
- **Purpose:** Does the choice of task_a head materially change B3's joint-task numbers? 1-fold AZ is the cheapest screen; if a head wins by > 2 pp at 1-fold we'd validate at 5-fold.

## Results (AZ 1-fold × 50 epochs, joint_score epoch)

| Rank | Cat head | Cat F1 | Cat Acc@1 | Reg Acc@10 | Reg Acc@5 | Reg MRR | Reg F1 |
|:-:|---|---:|---:|---:|---:|---:|---:|
| **1** | **`next_gru`** ⭐ | **0.4360** | **0.4711** | 0.4793 | 0.3652 | 0.2409 | 0.0640 |
| 2 | `default` / `next_mtl` (B3 current) | 0.4091 | 0.4187 | 0.4410 | 0.3266 | 0.2223 | 0.0616 |
| 3 | `next_stan` | 0.4036 | 0.4121 | **0.4754** | 0.3600 | **0.2515** | **0.0669** |
| 4 | `next_tcn_residual` | 0.3871 | 0.4260 | 0.4518 | 0.3368 | 0.2333 | 0.0606 |
| 5 | `next_temporal_cnn` | 0.3814 | 0.4023 | 0.4506 | 0.3382 | 0.2302 | 0.0589 |
| 6 | `next_lstm` | 0.3589 | 0.4160 | 0.3386 | 0.2439 | 0.1710 | 0.0445 |

**Sanity check:** `default` and `next_mtl` rows are bit-identical (as expected — MTLnet's `_build_next_head(name=None, ...)` falls back to `NextHeadMTL` with `dropout=0.1`; explicit `--cat-head next_mtl` does the same via `create_model("next_mtl", ...)` with `_filter_kwargs`).

## Δ vs B3 default (next_mtl)

| Cat head | Δ Cat F1 | Δ Cat Acc@1 | Δ Reg Acc@10 | Δ Reg MRR | Net verdict |
|---|---:|---:|---:|---:|---|
| `next_gru` | **+0.0269** | **+0.0524** | **+0.0383** | +0.0186 | **wins on both tasks** |
| `next_stan` | −0.0055 | −0.0066 | +0.0344 | +0.0292 | trades small cat loss for reg gain |
| `next_tcn_residual` | −0.0220 | +0.0073 | +0.0108 | +0.0110 | small cat loss, modest reg |
| `next_temporal_cnn` | −0.0277 | −0.0164 | +0.0096 | +0.0079 | worse on cat |
| `next_lstm` | −0.0502 | −0.0027 | **−0.1024** | **−0.0513** | much worse on both |

## Interpretation

- **GRU is the right architectural bias for the 7-class cat task on a 9-step sequence.** Transformer (next_mtl) over-parameterises at this low output cardinality; the RNN's inductive bias (sequential last-step summarisation) fits better.
- **STAN as task_a head** is competitive on reg (best reg_MRR) but slightly worse on cat. This is surprising — STAN's design target is spatial-temporal attention with pairwise biases, so for 7-class cat it adds parameters without commensurate structure. Interesting nonetheless.
- **LSTM's failure is stark** — −10 pp reg Acc@10 vs default. Something about the LSTM's forget-gate dynamics collides with the shared-backbone output shape. Not investigated further; LSTM is ruled out.
- **TCN heads are mid-pack on reg but weaker on cat.** They don't help.

## Caveats

- **n=1 fold only.** Fold-selection noise on AZ typically ±1-2 pp at best-epoch selection. The +2.69 pp cat F1 lift for GRU is at the boundary of fold noise; +5.24 pp cat Acc@1 is more robust. Needs 5-fold confirmation.
- **Comparing 1f numbers to 5f headline:** the default's AZ 1f cat F1 is 0.4091, while B3's AZ 5f cat F1 is 0.4362 ± 0.0074. The 5f mean is +2.7 pp higher than the 1f — roughly in line with fold variance. So a 1f +2.69 pp gain for GRU at 5f could land anywhere in [−0.5, +4.0] after folding.
- **AZ only.** AL and FL untested for cat-head choice. The winner on AZ might not generalise.

## Implications for B3

If the AZ 5-fold validation confirms the 1-fold ranking, **B3's cat head should be swapped from `NextHeadMTL` (Transformer) to `next_gru`.** This is a micro-change to the task_set preset (set `task_a.head_factory = "next_gru"`) and would immediately reframe:

- AZ cat F1 likely rises from 0.4362 ± 0.0074 → ~0.45-0.46 range (estimated).
- AZ reg Acc@10 likely rises from 0.5276 ± 0.0392 → higher (though reg was also lifted under GRU-cat-head).
- The cat-F1 MTL-over-STL claim (currently +1.65 pp p=0.0312 on AZ) would become stronger (+3-5 pp estimated).

## Combined with F21C finding

**The paper now has two mechanism claims, not one:**
1. **F21c** — STL with the graph-prior head (GETNext-hard) outperforms MTL on reg by 12-14 pp at AL/AZ. MTL is NOT adding value on region beyond the head choice.
2. **F27** — Within MTL, the cat head choice matters: GRU cat head beats the Transformer default by 2-3 pp cat F1. B3's cat head should be swapped.

Combined, the paper's re-framed contribution:
- **Check2HGI > HGI on cat F1** (CH16, unchanged).
- **Graph-prior head (GETNext-hard) is the new SOTA for region at ~10³-scale region cardinalities** (F21c).
- **For joint-task prediction, MTL-B3 with `next_gru` cat head is the single-model option** (F27 refinement).
- **Matched-head STL baselines matter** — unmatched baselines overstate MTL's gains.
- **FL-scale has a PCGrad × hard-prior gradient-starvation failure mode** (F2 mechanism, separately paper-worthy).

## Next step

**5-fold validation of `next_gru` as B3 cat head on AZ.** Cost ~25-30 min MPS. If the 1-fold delta holds, update NORTH_STAR.md to swap B3's task_a head.

After AZ validation:
- AL 5-fold validation (~15 min MPS).
- FL 1-fold validation (~90 min MPS).

## Validation outcomes (2026-04-24)

### AZ 5-fold (F27 validation, pid 9999 → `results/F27_validation/az_5f50ep_b3_cathead_gru.json`)

Paired Wilcoxon (B3+default vs B3+next_gru, 5 folds):

| Metric | Δ mean (gru − default, pp) | p_greater | Folds positive | Verdict |
|---|---:|---:|:-:|---|
| **cat F1** | **+2.37** | **0.0312** | **5/5** | significant |
| **cat Acc@1** | **+4.69** | **0.0312** | **5/5** | significant |
| **reg MRR_indist** | **+1.50** | **0.0312** | **5/5** | significant |
| reg Acc@10_indist | +1.98 | 0.0625 | 4/5 | marginal |
| reg Acc@5_indist | +1.69 | 0.0625 | 4/5 | marginal |
| reg macro-F1 | +0.35 | 0.0625 | 4/5 | marginal |

Three metrics hit minimum-achievable p-value (all 5 folds positive). Paper-strength evidence on AZ.

### AL 5-fold (F31, pid 10542 → `results/F27_validation/al_5f50ep_b3_cathead_gru.json`)

Direct comparison to pre-F27 AL B3 (5f × 50ep):

| Metric | F31 AL B3+gru | Pre-F27 AL B3 | Δ |
|---|---:|---:|---:|
| cat F1 | **0.4271 ± 0.0137** | 0.3928 ± 0.0080 | **+3.43 pp** |
| cat Acc@1 | **0.4582 ± 0.0151** | ≈0.4110 | **+4.72 pp** |
| reg Acc@10_indist | **0.5960 ± 0.0409** | 0.5633 ± 0.0816 | **+3.27 pp** (σ tighter) |
| reg Acc@5_indist | 0.4601 ± 0.0445 | 0.4281 ± 0.0789 | +3.20 pp |
| reg MRR_indist | 0.3074 ± 0.0287 | 0.2855 ± 0.0533 | +2.19 pp |

**Even bigger than AZ.** Plus notable: **AL MTL reg Acc@10 = 59.60 ≥ STL STAN 59.20 — first time MTL matches/surpasses STL STAN on AL reg Acc@10.** The F21c gap to STL-GETNext-hard (68.37) narrows from −12.04 to −7.77 pp.

### FL 1-fold (F32, pid 10703 → `results/F27_validation/fl_1f50ep_b3_cathead_gru.json`)

Compare to the two prior n=1 FL B3 runs (with default next_mtl cat head):

| Metric | F32 FL B3+gru | F2 Phase B3 (pre-F27) | F17 fold 1 (pre-F27) | Δ (F32 vs pre-F27 mean) |
|---|---:|---:|---:|---:|
| cat F1 | 0.6572 | 0.6623 | 0.6706 | **−0.93 pp** |
| cat Acc@1 | 0.6860 | 0.6870 | 0.6935 | −0.43 pp |
| reg Acc@10_indist | 0.6526 | 0.6582 | 0.6655 | −0.93 pp |
| reg Acc@5_indist | 0.4342 | 0.3988 | 0.5360 | (n=1 noise; F17 had a spike) |
| reg MRR_indist | 0.2956 | 0.2794 | 0.3129 | (n=1 noise) |

**FL flips sign.** The +3-4 pp cat F1 gain on AL/AZ becomes a −1 pp cat F1 *loss* on FL. All within n=1 noise (~1 pp typical variance), but the direction is opposite. Two interpretations:

1. **n=1 fold-selection noise.** FL 1-fold cat F1 has ~0.9 pp variance across the three n=1 runs we have. The F32 value (0.6572) is within the envelope [0.6623, 0.6706] of the pre-F27 runs minus noise. Could land on either side at 5f.
2. **Genuine scale-dependence.** At 127 K rows × 4.7 K regions, the Transformer head (next_mtl) has enough capacity to use more of the signal than the GRU's last-timestep-summarisation bottleneck. Smaller states (AL 10K, AZ 26K) favour GRU's inductive bias; larger states favour the Transformer.

**Decision:** held pending user input. Three paths:

- **Path A — Commit `next_gru` universally.** Simpler narrative. FL cat F1 might land ~0.5–1 pp below the default at 5f. Worth it for 3+ pp gains on AL/AZ and the paper's clean "one model" story.
- **Path B — Scale-dependent cat head.** `next_gru` for AL/AZ, `next_mtl` for FL/CA/TX. Maximizes each state's performance but fragments the story.
- **Path C — Run FL 5f B3+gru (~6 h MPS).** Decisive; settles the σ question and lets us commit confidently to A or B.

## Updated NORTH_STAR policy (pending user)

If user picks **A**: NORTH_STAR stays as committed 2026-04-24 (task_a head = next_gru universally).
If user picks **B**: NORTH_STAR updated to note per-state cat head; CA/TX follow the FL pattern (next_mtl).
If user picks **C**: NORTH_STAR stays at next_gru with a "pending FL 5f" caveat.

## Files

- `results/F27_cathead_sweep/az_1f50ep_cathead_*.json` (7 configs, 1-fold each).
- `scripts/run_f27_cathead_sweep.sh` (launcher).
- CLI flag changes: `scripts/train.py` (added `--cat-head`, `--cat-head-param`) and `src/tasks/presets.py` (`resolve_task_set` gained `task_a_head_factory` parameter).
