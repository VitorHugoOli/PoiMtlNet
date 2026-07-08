# A40-4 — Faithful STAN CA/TX — STATUS (2026-07-07)

> **Verdict so far: STAN clears the Markov/simple floor at BOTH large states → it IS citable → the "STAN infeasible"
> footnote will drop.** Run is **paused at a partial** (user request) — TX 4/5, CA 2/5 — all folds banked as durable
> JSONs; resumable. The earlier "infeasible / days-per-state" footnote was over-conservative: STAN runs on the A40 at
> large C (bf16 stable, no NaN wall) — it's just **slow (~2.6 h/fold)**, ~10× the handoff's "1.5-2 h/state" guess.

## Partial results (v6 recipe, patience-10, seed 0)
| state | folds done | per-fold Acc@10 | **mean** | floor (best-simple) | verdict |
|---|---|---|---:|---:|---|
| **TX** (R=6553) | **4/5** (0,1,2,3) | 61.48, 61.69, 61.50, 62.00 | **61.67** | 54.94 | ✅ **clears** |
| **CA** (R=8501) | **2/5** (0,1) | 58.43, 58.60 | **58.52** | 52.09 | ✅ **clears** |

Cross-fold variance tiny (TX ±0.2, CA ±0.1). Both means are **~7 pp above floor** and land **below our MTL reg**
(CA 65.66 / TX 67.02) — i.e. STAN citable but does not threaten the "we lead" story, exactly as expected. (Markov-1
floors are lower still: TX 35.6 / CA 31.5; A40.md's "~55/~52" bar is the *best-simple* baseline.)

## Recipe (matched to the citable FL/AL/AZ v6 numbers)
`research/baselines/stan/train.py --state {ca,tx} --seed 0 --epochs 200 --folds 5 --only-fold {k} --batch-size 2048
--amp bf16 --compile --d-model 128 --patience 10 --tag v6_p10`, `PYTHONPATH=repo:src`.
- **patience 10 (not 20) is quality-neutral**: STAN converges at best_epoch ~1-2, so the best-epoch checkpoint (the
  reported metric) is identical to patience-20 at ~half the wall-time. Verified across all TX/CA folds (best_ep ≤ 3).
- **v6 = the corrected/audited STAN** (matching layer + scalar `[K]` tables + `F.embedding` backward — commits
  `1b83c1c1/1eeb43fd/abcd7a06/507a5f22`). NOT the stale v4 in `docs/findings/FAITHFUL_STAN_FINDINGS.md` (which
  reported STAN *below* Markov, pre-`F.embedding`-fix).

## Prerequisite (done): memory-bounded ETL
CA/TX STAN inputs were regenerated with the new `--streaming` ETL (commit `a92d8b16`; the earlier-cited `06c24757` was
a pre-rebase hash unreachable from main — same content): TX 4.04M / CA 3.12M windows,
content-identical to the in-memory path (parity-verified on AL). **Do NOT** run the ETL without `--streaming` at CA/TX
(the in-memory build OOMs at ~112 GB). Inputs live at `output/baselines/stan/{california,texas}/inputs.parquet`.

## To RESUME (finish the remaining 4 folds ≈ ~10 h)
```bash
bash docs/studies/closing_data/v17_completion/stan_catx/run_stan_interleaved.sh   # detached; see below
```
Resumable: it skips the 6 done folds and runs `CA2, TX4, CA3, CA4`. Order interleaves TX/CA so both means fill early.
Launch detached (survives session teardown), then poll — it is **self-contained + robust** (see design below).

## On completion (5/5 both states) — acceptance actions
Aggregate the 5 `faithful_stan_{state}_5f_200ep_v6_p10_fold{k}.json` → 5f-mean. If both still clear (they will):
1. Fill Table 3 CA/TX STAN cells + **drop the `--`$^{\dagger}$ "STAN infeasible" clause for CA/TX** (ReHDM stays footnoted).
2. Update `docs/baselines/next_region/stan.md` + `RESULTS_BOARD §4`. Device-label **A40-fp32? no — bf16+compile**; note it.
3. STAN stays **below MTL reg** → changes no verdict (coverage).

## ⚠ The RAM incident + root cause (netdata-confirmed) — the design lessons
The **first** CA/TX STAN run (all-TX-first, single 5-fold process) was killed at 21:49 by **our own RAM watchdog** when
`MemAvailable` crashed from ~107 GB to **944 MB**. Root cause = **a shared-box NEIGHBOR**, not STAN:
- netdata `system.ram` shows RAM spiking to **80-100 GB every few minutes, INCLUDING after STAN was already dead**
  (21:59 hit 82 GB used with zero STAN running). STAN itself was steady ~18 GB for 4.5 h (no leak).
- The kill hit **mid-fold-1 (a normal training step)**, NOT a fold boundary — fold-0's end (with its full-metrics
  recompute) had completed cleanly. So it was **not** the fold-end recompute or a STAN pathology.

**Lessons baked into the current design (`run_stan_interleaved.sh` / `run_stan_foldwise.sh`):**
1. **Fold-by-fold, per-fold JSON** (`--only-fold`) — a kill costs ≤1 fold, not the whole run. Resumable (skip-if-JSON).
2. **NO self-watchdog** — STAN's ~18 GB is a good citizen; a preemptive watchdog *mis-targets* it (kills the innocent
   small job for the neighbor's 80 GB spike). If RAM is truly exhausted, let the **kernel OOM-killer** pick — it scores
   by memory use → it targets the real hog. (The old `run_stan_catx.sh` watchdog is retained only for reference.)
3. **Warm/shared `TORCHINDUCTOR_CACHE_DIR`** across folds — the eval's free, zero-numerical-effect win (folds skip cold
   recompiles). See `TRAINING_SPEEDUP_EVAL.md`.
4. **Detached (`setsid`)** so a session teardown doesn't kill the run; poll actively (disowned jobs don't notify).

> ⚠ **bash `set -u` + combined-`local` gotcha (hit 3× this study):** `local ST=$1 F=$2 J="...${ST}..."` throws
> `ST: unbound variable` because `${ST}` is expanded within the same `local` before `ST` is established. Split it:
> `local ST=$1 F=$2; local J="...${ST}..."`. (It was *masked* in `run_stan_foldwise.sh` by a global loop var `ST`,
> but fired in `run_stan_interleaved.sh` whose loop var is `job` — and earlier in `finalize_reg.sh`.)
