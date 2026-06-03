# Training-pipeline deep-dive audit (2026-06-03)

Two parallel auditor agents read the next-cat + next-reg pipelines end-to-end (input formation,
batching, loss, optimizer/scheduler, heads, metrics) to find a systematic issue capping BOTH STL
ceilings regardless of the head. Triggered by the user's instinct: "maybe we are losing a piece in
the big picture." **One HIGH finding (the answer is YES), plus secondary levers.**

## ⭐ HIGH — Non-overlapping windows discard ~7.5–8.4× of the training signal
`src/data/inputs/core.py::generate_sequences` defaults `stride = window_size` (=9) → NON-OVERLAPPING
windows (core.py:40, :56, :59; hard-default call at :388). It is the single windowing chokepoint for
BOTH heads (`builders.py` → `convert_user_checkins_to_sequences` → `generate_sequences`; `next_region.py`
/ `region_sequence.py` only re-read `sequences_next.parquet`, they do not re-window).

**Measured (Alabama):** 113,846 check-ins, 3,858 users → **only 12,709 sequences** on disk. The
"~12,709 samples" everywhere in the study is WINDOWS, not check-ins — a 9.0× reduction. Overlapping
(stride=1) → ~96,326 full-9 windows (7.5×) or ~108,073 (8.4×) if every check-in with ≥1 history is a
target. Non-overlapping windowing is NOT standard for next-POI/next-cat (STAN/GETNext/DeepMove/Flashback
use stride 1). With only ~10k train sequences over ~821 region classes, the reg head is data-starved;
cat (7 classes) less so but still loses 9× the examples. **This caps both ceilings identically,
independent of head — the systematic piece.**

**Leak-safety of the fix (VERIFIED):** the CV split is StratifiedGroupKFold(groups=userid); fold 0 has
0 users shared train↔val. All of a user's sequences land in one fold → overlapping windows cannot cross
the train/val boundary → **overlap is leak-free under this split.** (The 22.4% "target reappears in its
own 9-history" rate is legitimate user revisits, unchanged by overlap.)

**Recommendation:** the single highest-EV experiment in the audit. Thread a `stride` param from the
pipeline (`generate_next_input_from_checkins`/`convert_user_checkins_to_sequences`/`generate_sequences`),
rebuild inputs at stride=1 (or 2–3 as a compute compromise), re-measure both ceilings. Control for
correlated-val (overlapping val rows from one user are near-duplicates → wider σ, not bias; the
user-grouped split keeps it honest). **FOUNDATIONAL: this would change the frozen (c)/(d) ceilings, the
MTL board, the per-fold log_T, AND the v11 paper canon — all built on non-overlapping windows. Validate
in isolation first (separate input dir, frozen substrate untouched); if it lifts the ceilings, it is a
strategic/paper-level decision, not a silent swap.**

## MED — 58% of users dropped by MIN_SEQUENCE_LENGTH=5
`core.py:52` drops users with <5 check-ins → AL drops 2,236/3,858 users (58%) but only 4,151/113,846
check-ins (3.65%). Biases the ceiling toward heavy users; removes the cold-start slice. Lowering to 2
(a user with ≥2 check-ins yields ≥1 history→next pair), combined with stride=1, recovers most of them.
Changes which population the ceiling describes.

## MED — 13% "leftover-branch" is a different prediction task
`core.py:71-79`: when the last block has no real check-in after the window, it takes the last real
visit as the target and DELETES it from history (non-contiguous). Fires for 1,622/12,709 (12.8%) AL rows.
These predict "the user's final check-in from everything before" rather than the consistent
"next-after-a-fixed-9-window." Stride=1 nearly eliminates this branch for free.

## MED (training) — cat runs fp16 autocast WITHOUT GradScaler; reg runs fp32
`_single_task_train.py:65-69,87,154`: cat wraps BOTH train and val in `torch.autocast(fp16)` with no
GradScaler → gradient-underflow risk + autocasts the val forward (changes the logits the macro-F1 argmax
sees) vs reg's fp32. Small effect on a 7-class argmax metric, but a genuine cat/reg inconsistency + a
latent correctness smell. Fix: add a GradScaler OR run cat in fp32/bf16. Cheap to confirm (re-pin AL cat
with autocast off, compare to 49.97).

## MED (training) — 50-epoch OneCycle late-epoch overfit
train_acc→0.9 while val_loss rises after ~ep14. Both loops select the diagnostic-best val epoch (so the
REPORTED number is not corrupted), but OneCycle's annealing tail means the val-best model is caught
mid-schedule. A shorter OneCycle (~20–25 ep) or early-stopping could let it converge AT its peak. The one
schedule lever with plausible upside on both ceilings; cheap single-state test.

## LOW / comparability (no ceiling impact)
- Per-fold reseeding differs: reg `seed_everything(seed+fold)`, cat seeds once upstream → σ bands not
  strictly comparable (matters only for T2 go/no-go gates). Unify.
- Determinism flags fire in p1's path, maybe not cat's → GRU/cuDNN nondeterminism noise on cat only.
- Reg `top10_acc` is over ALL classes (not in-dist-masked despite the name) — CORRECT + consistent with
  the CE loss axis; val-only best-epoch selection (standard mild optimistic bias, same as cat's f1-best).
  No loss/metric mismatch, no leak.
- logit-adjust offset is applied at EVAL too (changes argmax) — defensible (optimizes macro-F1, prior is
  train-only) but the cat ceiling reflects "logit-adjusted inference," not adjusted-train/raw-infer. Note it.
- Verify the frozen reg recipe passes `freeze_alpha=True` (not just `alpha_init=0`) so "prior OFF" is exact
  (it does — R1 used `freeze_alpha=True alpha_init=0.0`).
- Stale `next_region.parquet` freshness guard checks only row-count + userid (not mtime/hash) → a stale
  embedding could pass silently (benign today: the reg STL path reads X from the fresh `next.parquet`).

## VERIFIED CLEAN
Padding (no real all-zero embedding collides with the pad sentinel; min |emb|₁ = 54.99), masking (all
padding trailing, last-valid pooling correct for next_gru + next_stan_flow), label alignment (target
strictly after the window, no target-embedding leak), region-label alignment (100% match to stored
next_region.parquet), fold leakage (user-disjoint), class-weight leakage (train-only), loss
double-application (none — class-weights ignored when loss_calibration is set), train/eval mode, the
α=0 prior-off path. Heads, pooling, masking, loss, metrics are correct and leak-free.

## Bottom line
The head-independent cap the user suspected is REAL = **non-overlapping stride=9 windowing** (7.5–8.4×
training-data loss, leak-safe to fix). Everything else is correct; secondary levers are the short-user
drop, the leftover-branch, cat-fp16-autocast, and the 50-epoch schedule. The windowing fix is foundational
(touches the frozen ceilings + paper canon) → validate in isolation, then a strategic decision.
