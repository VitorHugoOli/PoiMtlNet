# HANDOFF — MTL Improvement track (read this FIRST, then `log.md` + `INDEX.html`)

**As of 2026-06-03.** Branch `mtl-improve` (≈51 commits ahead of `main`, all pushed). Working tree clean.
This is a single "you are here" snapshot. Full chronology: `log.md` (27 dated entries). Design + per-tier
results: `INDEX.html`. Canonical numbers: `docs/results/mtl_improvement/TIER01_RESULTS.md`.

---

## 1. Where we are
- **Tier 0 + Tier 1 are COMPLETE and FROZEN.** The (c)/(d) STL ceilings are the immutable track yardstick.
- **Tier S (STL head search) is COMPLETE — a reviewer-proof NEGATIVE**: no head, encoder, aux loss, OR
  per-task-tuned challenger beats the tuned per-task incumbent. The head is NOT the lever (regime finding).
- **Tier 2 (T2.1 dual-tower) is the next step and has NOT started.** It is the centerpiece — the architectural
  gap in the joint MTL reg pathway. Now *better motivated* (see the overlap finding §4).
- A major out-of-band finding (overlapping windows) was validated + documented as future-work; **the
  non-overlapping canon is deliberately KEPT** for whole-study consistency.

## 2. The FROZEN ceilings (immutable — T2-T5 Δ are measured against these)
Recipe: **reg** = `next_stan_flow` α=0 (log_T prior OFF); **cat** = `next_gru` logit-adjust τ=0.5.
| | AL | AZ | GE | FL |
|---|---|---|---|---|
| (c) cat macro-F1 | 49.97 | 51.01 | 58.12 | 69.97 |
| (c) reg Acc@10 | 62.88 | 55.11 | 58.45 | 73.31 |
| (d) composite reg (max v14/HGI α0) | 63.58 | 55.11 | 58.76 | 73.62 |
Guard: run `scripts/mtl_improvement/t14_freeze_sanity.py` after ANY ceiling change (asserts cat-ceiling
arch == NextHeadGRU + every ceiling ≥ the MTL it bounds). Currently GREEN. **Caveat:** these are
seed=42 single-seed; §0.1 paper numbers use seeds {0,1,7,100}.

## 3. What got done this session (newest first) + pointers
1. **Overlap-window study** (`future_works/overlapping_windows.md`, `results/.../overlap_window_probe.md`):
   non-overlapping windows cap ceilings ~+5–9.8pp at small-state STL (head-independent). Validated harness
   + real STL + real MTL. **Documented as future-work; canon kept.** See §4.
2. **Pipeline deep-dive audit** (`PIPELINE_AUDIT_2026-06-03.md`): the windowing cap (HIGH) + secondary levers
   (58% short-user drop, cat fp16-autocast-no-scaler, 50ep overfit). Everything else CLEAN.
3. **Critical advisor on heads + per-task tuning + STAN** → STAN-attention for cat FALSIFIED even tuned
   (−7.4pp); next_lstm tuned ties but never wins. Mechanistic: recurrence wins cat, attention wins reg —
   the frozen ceilings already use the right head per task.
4. **Tier S Prong A + B**: all coded heads + new SSM (`next_mamba`) + SimGCL aux (`next_gru_simgcl`) — none
   beats the incumbent. (INDEX S.1/S.2/S.3/S.4.)
5. **CAT CEILING BUG (caught + fixed)**: `train.py --cat-head` is silently ignored on `--task next` (MTL-track
   only) → the cat ceiling had run `next_single` not `next_gru`. Re-pinned with `--model next_gru` (the +8pp
   AL correction in §2). Two advisor passes confirmed the re-pin sound + added the freeze-sanity guard.
6. **T1.4** (the tuning that set the frozen ceilings): new leak-free loss code `src/losses/calibrated.py`
   (logit-adjust/focal/tail-loss; 19 tests). Full per-task tune → reg α=0 + cat logit-adjust τ=0.5 win.

## 4. The overlap finding — what the next agent MUST understand
Non-overlapping windows (stride=9) train on ~8.5× less data than overlap (stride=1). Validated at AL (real
pipeline): **cat rising-tide** (STL +9.77 / MTL +8.92), **reg gap WIDENS** (STL +5.13 / MTL +0.50; the
STL→MTL reg gap 8.34→12.96). The MTL reg shared-backbone bottleneck *can't absorb the extra data* → this
**STRENGTHENS the regime/dual-tower thesis**. HGI behaves like v14 (+4.89), stays tied. **Decision: keep
non-overlap canon; it's future-work.** All probe code is isolated in engine `check2hgi_dk_ovl` — the
canonical substrate is UNTOUCHED. If ever adopted, see the rebuild checklist in the future-work memo.

## 5. Traps / gotchas (don't repeat these)
- **`--cat-head`/`--reg-head` are MTL-track-only.** For STL `--task next`, use `--model <head>`. Always
  verify the arch that actually ran (`results/.../model/arch.txt`).
- **Sanity-check any STL ceiling ≥ the MTL it bounds** (the cat bug was visible as ceiling < MTL). Use
  `t14_freeze_sanity.py`.
- **`--per-fold-transition-dir` must be the seeded log_T** (`region_transition_log_seed{S}_fold{N}.pt`);
  default log_T leaks ~+3pp. Stale-log_T guard: mtime(log_T) > mtime(next_region.parquet).
- **The registry silently drops unknown kwargs** — a head can ignore a param you think you set.
- **Frozen folds + the moving-baseline guard**: Tier-S/T5 winners feed candidates, NEVER re-open (c)/(d).
- Repo pre-stages unrelated `articles/*` — always `git add` with explicit pathspec + check `git show --stat`.

## 6. New reusable assets (this session)
- `src/losses/calibrated.py` (+ test) — leak-free logit-adjust/focal/balanced/CB/LDAM; wired into
  `next_cv.py` via `ExperimentConfig.loss_calibration` + `p1` + `train.py` flags.
- Heads: `next_mamba` (selective-SSM), `next_gru_simgcl` (SimGCL aux + `model.aux_loss` trainer hook in
  `_single_task_train.py`). Both sound but lose/tie — keep as tested assets, not in use.
- Overlap infra: backward-compatible `stride` param (`core.py`/`builders.py`), engine-aware region-seq
  (`region_sequence.py` + `folds.py` + `p1` `seq_engine`), isolated engine `check2hgi_dk_ovl`,
  `build_overlap_probe_engine.py`, `overlap_probe.py`.
- Scripts: `scripts/mtl_improvement/` — `t14_*` (T1.4 sweep/validate/repin/agg/sanity), `tierS_*`
  (screen/confirm/unit), `stan_for_cat.sh`, `overlap_*`.

## 7. THE NEXT STEP (open decision — surface to the user, do NOT autopilot)
Per the AGENT_PROMPT tier-boundary cadence, the user decides. The clear next is **Tier 2 — T2.1 dual-tower**
(INDEX `#tier2`): the reg-private full-STAN backbone vs the frozen (c)/(d), regime×substrate 2×2, frozen-fold
paired, with the mandatory **unit-test gate + per-arch LR mini-sweep** (hard rule 7/10) BEFORE the multi-fold
launch. The overlap finding makes T2 better-motivated (more data widens the reg gap → the bottleneck is real).
Other open/optional items: T4.0 loss-scale/RLW litmus (cheap, ungated); the cheap training levers (cat fp32 vs
fp16-autocast, shorter 25ep schedule — audit MED); overlap-pattern confirm at AZ/GE/FL + multi-seed.

## 8. How to resume
1. Read this → `log.md` (bottom 12 entries) → `INDEX.html` §"The regime finding" + the Tier 2 cards →
   `TIER01_RESULTS.md`.
2. `git pull`; confirm `t14_freeze_sanity.py` is GREEN.
3. For Tier 2: build the dual-tower per `B9_STL_STAN_SWAP §6.4` + `future_works/mtl_architecture_revisit`,
   clear the unit-test gate, run the per-arch LR mini-sweep at AL+AZ, then full protocol. Score vs frozen (c)/(d).
4. STOP + surface at the tier boundary (advisor pass → summary → user decision).
