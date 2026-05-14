# Session Handoff — 2026-05-01 (Paper Closure on H100)

**Branch:** `worktree-check2hgi-mtl`
**Hardware:** H100 80 GB (Lightning Studio)
**Duration:** ~6 hours wall (started ~00:14 UTC, finished ~05:08 UTC)
**Total runs landed:** 51 paper-grade runs (15 Tier-3 smokes + 28 paper-closure + 8 H3-alt gap-fill)

This is the operational handoff for the agent picking up after the paper-closure
session. It complements the paper-grade results doc
[`PAPER_CLOSURE_RESULTS_2026-05-01.md`](PAPER_CLOSURE_RESULTS_2026-05-01.md) and
the phase plan [`PAPER_CLOSURE_PHASES.md`](PAPER_CLOSURE_PHASES.md). Read both
those first; this doc covers what's *new* and *what to do next*.

---

## TL;DR — what landed

1. **F51 Tier 3 (Phase 0)** — 15 smokes 5f×30ep on B9 base × {weight_decay,
   max_grad_norm, eta_min, OneCycle pct_start, AdamW eps}. **Clean negative.**
   B9 is locally optimal in the optimizer/scheduler axis too. Reviewer-rebuttal
   insurance banked. CLI flags `--weight-decay`, `--adam-eps`, `--max-grad-norm`,
   `--eta-min` patched into `scripts/train.py` for future ablations.
2. **Cross-state P3 + STL ceilings (Phase 1)** — CA + TX MTL B9 + H3-alt anchor;
   CA + TX STL ceilings (cat + reg); FL STL reg multi-seed extension (4 extra
   seeds). 12 runs.
3. **AL + AZ multi-seed (Phase 2)** — 8 STL reg multi-seed + 8 MTL B9 multi-seed
   × 4 seeds. Symmetrizes the architectural-Δ scale curve error bars.
4. **Late-session gap-fill — AL + AZ H3-alt multi-seed** (8 runs after audit
   prompt "have we made executions on H3-alt?"). Closed the recipe-comparison
   axis. **Surfaced a major finding** (see §3 below).
5. **Wilcoxon JSONs** for both axes:
   - `research/PAPER_CLOSURE_WILCOXON.json` — MTL B9 vs STL ceiling, both tasks.
   - `research/PAPER_CLOSURE_RECIPE_WILCOXON.json` — B9 vs H3-alt, both tasks.
6. **Canonical analysis script** — `scripts/analysis/paper_closure_wilcoxon.py`
   (mirrors F51 extraction methodology). Reproducible end-to-end.

---

## 1 · The two reframes — paper-narrative impact

This session produced **two paper-grade reframes** that supersede prior claims:

### Reframe 1: Architectural-Δ is sign-consistent across all states (was: scale-conditional, AL-favors-MTL)

F49 Layer 3 had reported AL-MTL exceeding STL F21c by +6.48 pp on reg (a leaky
measurement under the legacy full-data `region_transition_log.pt`). Under
leak-free symmetric comparison with seeded per-fold log_T:

| State | n_pairs | Δ_reg pp | p_reg | Δ_cat pp | p_cat |
|---|---:|---:|---:|---:|---:|
| AL | 20 | **−11.04** | **1.9e-06** | −0.19 | 0.76 (≈tied) |
| AZ | 20 | **−12.27** | **1.9e-06** | **+1.90** | **1.9e-06** |
| FL | 5  | −7.99 | 0.0625 | (n/a — F37 lacks per-fold) | — |
| CA | 5  | −8.92 | 0.0625 | +1.94 | 0.0625 |
| TX | 5  | −16.69 | 0.0625 | +2.02 | 0.0625 |

**Headline:** classic MTL tradeoff — MTL B9 < STL `next_getnext_hard` on reg by
7-17 pp at every state; MTL B9 ≥ STL `next_gru` on cat at every state by 0 to
+2 pp. AL is the only state where the cat gain is ≈0.

The "scale-conditional architecture-dominant at AL" framing in F49 was a leak
artifact (pre-F50 AL inflation ≈ 27 pp; the published "13-17 pp" was a FL-
calibrated estimate).

### Reframe 2: B9 champion claim is FL-scale-specific (was: universal champion)

F51 multi-seed established B9 > H3-alt at FL (Δ_reg = +3.48 pp, p=2.98e-8).
Under cross-state validation:

| State | n_pairs | Δ_reg pp | p_reg | Δ_cat pp | p_cat | Verdict |
|---|---:|---:|---:|---:|---:|---|
| AL | 20 | **−0.35** | **1.9e-03** | **−2.22** | **1.9e-06** | **H3-alt > B9 on cat; reg tied** |
| AZ | 20 | −0.09 | 0.23 (n.s.) | **−0.96** | **7.1e-04** | **H3-alt > B9 on cat; reg tied** |
| FL | 25 | **+3.48** | **3.0e-08** | +0.42 | 1.3e-05 | B9 > H3-alt (F51) |
| CA | 5  | +4.74 | 0.062 (5/5) | +0.72 | 0.125 (4/5) | B9 directional |
| TX | 5  | +1.76 | 0.125 (4/5) | +0.64 | 0.125 (4/5) | B9 directional |

**Headline:** B9 is FL-tuned. The three additions over H3-alt (alt-SGD,
cosine, α-no-WD) help on FL but **hurt cat training significantly at small
states** (AL p=1.9e-6, AZ p=7.1e-4). At large states (CA/TX) B9 directionally
wins both tasks but n=5 single-seed limits inference. **The optimal MTL recipe
is scale-conditional.**

Mechanism hypothesis: B9's additions target FL's reg-saturation problem (D5);
at smaller transition graphs the saturation is less severe AND alt-SGD's
per-step temporal gradient separation costs cat-side signal that small states
can't afford to lose. The α-no-WD ingredient targets `next_getnext_hard`'s α
growth specifically — also less load-bearing at small scale.

---

## 2 · What survived from prior work

Several prior findings were validated by leak-free re-measurement:

- **F49 Layer 1+2 mechanism findings** — cat-supervision transfer is null
  (≤|0.75| pp on AL/AZ/FL); loss-side `task_weight=0` is methodologically
  unsound under cross-attn MTL. Both **survive** as paper-grade methodological
  contributions independent of the leak.
- **F37 P1 cat ceiling at FL (0.6698 ± 0.0061)** — uses `next_gru` cat head, no
  log_T involvement → was always leak-free. Survives.
- **F51 multi-seed at FL** — already used seeded per-fold log_T. Survives
  (B9 > H3-alt at FL is the reframed-but-true-at-FL champion claim).
- **F51 Tier 2 capacity sweep** — B9 locally optimal in 5/7 capacity dims at
  FL. Holds at FL; the cross-state recipe-portability finding (§1 Reframe 2)
  doesn't contradict the FL-local capacity result.
- **D5 encoder saturation receipt** — survives; mechanism is structural to
  `next_getnext_hard`'s α growth dynamics, leak-independent.
- **F63 α-trajectory figure** — survives (mechanism not affected by leak).

---

## 3 · Patches landed — durable across sessions

| File | Change | Reason |
|---|---|---|
| `scripts/train.py` | +4 CLI flags: `--weight-decay`, `--adam-eps`, `--max-grad-norm`, `--eta-min` | Tier 3 sweep needed exposed config knobs |
| `src/configs/experiment.py` | Added `eta_min: float = 0.0` field | Cosine-tail floor LR |
| `src/training/helpers.py` | `setup_scheduler` accepts `eta_min` → `CosineAnnealingLR` | Wire-up |
| `src/training/runners/mtl_cv.py` | Pass `eta_min=getattr(config, "eta_min", 0.0)` to scheduler builder | Wire-up |
| `src/tracking/experiment.py` | `start_date` now `%Y%m%d_%H%M%S_<pid>` | Parallel run-dir collision fix (was minute-granular → 3-way parallel runs collided) |
| `scripts/p1_region_head_ablation.py` | Seeded log_T preferred (`region_transition_log_seed{S}_fold{N}.pt`) with legacy fallback | The F51 seeded-naming migration only updated `mtl_cv.py`; the STL ablation script was left expecting the old naming and broke under the post-F51 deletion of legacy log_T files |

All 152 training + 127 tracking unit tests pass post-patch. Validated end-to-end
on FL B9 reference run `_20260430_0110`: extraction returns 63.14 ± 1.15
(F51 published 63.47 ± 0.75 — matches within 0.3 pp).

---

## 4 · Failures encountered + how they were diagnosed

The Phase 1+2 first-attempt run (28 runs in parallel) hit 17/28 failures.
Diagnosed and fixed:

1. **OOM at CA/TX MTL parallel.** Two simultaneous MTL jobs at CA (8501 regions)
   or TX (6553) exceeded H100 80 GB during the train-side logit cat
   (`mtl_cv.py:541`: `torch.cat(all_task_b_logits)` allocates ~9 GB on top of
   each job's ~36-40 GB). **Fix:** retry serialized big-state MTL pairs;
   exported `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for fragmentation.
2. **STL reg `FileNotFoundError` on legacy log_T name.** Pre-F51 leak-fix the
   per-fold log_T was named `region_transition_log_fold{N}.pt`; F51 migrated to
   `region_transition_log_seed{S}_fold{N}.pt` and the trainer hard-fails on
   the legacy name. The migration **only updated `mtl_cv.py`**; the STL ablation
   script `scripts/p1_region_head_ablation.py` was left expecting the legacy
   filename. Pre-flight had deleted the legacy files (because the trainer
   hard-fails on them) → STL reg crashed across all states. **Fix:** patched
   `p1_region_head_ablation.py` to prefer seeded names with a loud-warning
   legacy fallback.

The retry pass (`scripts/run_paper_closure_retry.sh`) cleared all 17 failures.

---

## 5 · Run-dir map for the paper-closure runs

All under `results/check2hgi/<state>/mtlnet_lr1.0e-04_bs2048_ep50_20260501_*`
(MTL) or `results/check2hgi/<state>/next_lr1.0e-04_bs2048_ep50_20260501_*`
(STL cat). PIDs are seconds+PID-suffixed for collision-free parallel runs.

**MTL — paper closure:**

| Tag | State | Recipe | Seed | Run dir suffix |
|---|---|---|---:|---|
| ca_h3alt | CA | H3-alt | 42 | `_011216_406571` |
| ca_b9_retry | CA | B9 | 42 | `_015857_412969` |
| tx_b9_retry | TX | B9 | 42 | `_023224_413998` |
| tx_h3alt_retry | TX | H3-alt | 42 | `_031509_414897` |
| al_b9_seed{0,1,7,100} | AL | B9 | (4) | `_014904_{411673,411676,411691,411706}` |
| al_h3alt_seed{0,1,7,100} | AL | H3-alt | (4) | `_045921_{419781,419787,419793,419795}` |
| az_b9_seed{0,1,7,100} | AZ | B9 | (4) | `_015{206,208,209,209}_{412194,412291,412306,412356}` |
| az_h3alt_seed{0,1,7,100} | AZ | H3-alt | (4) | `_05024{2,3,4,5}_{420315,420355,420423,420546}` |

**STL reg — paper closure:** under `docs/studies/check2hgi/results/P1/region_head_*_paper_close_*.json`.
**STL cat — paper closure:** CA `_011357_409058`; TX `_012451_409450`.

PID-to-seed mapping for AL/AZ B9 and H3-alt: launch order in the script was
seeds 0 → 1 → 7 → 100, so PIDs ascend in seed order.

---

## 6 · What's next (camera-ready P1 — not paper-blocking)

| Item | Priority | ETA on H100 | Why |
|---|---|---|---|
| CA + TX MTL B9 + H3-alt multi-seed (4 extra seeds × 2 states × 2 arms) | P1 | ~3-4 h | Symmetrize CA/TX error bars; currently single-seed n=5 |
| AL + AZ + FL **STL cat** multi-seed (extends F37's n=1) | P1 | ~30 min | Symmetric multi-seed extension closes cat-side error bars |
| FL B9 paired Wilcoxon at multi-seed scale | P2 | needs F51 dirs from Drive | Per-fold data not locally archived |
| `scripts/analysis/f50_delta_m.py` rerun on leak-free pool | P2 zero-compute | ~5 min | CH22 Δm scoreboard refresh |
| `RESULTS_TABLE.md` v6 (full body rewrite, not just header) | P1 | ~1 h author | Replace the leaky-flagged rows |
| `PAPER_DRAFT.md` rewrite for the two reframes | **P0 paper-side** | ~2-4 h author | The science is settled; this is authorship work for the human author |

---

## 7 · Pointers to docs updated this session

All have a 2026-05-01 reframe banner at the top:

- [`PAPER_CLOSURE_RESULTS_2026-05-01.md`](PAPER_CLOSURE_RESULTS_2026-05-01.md) — comprehensive (both axes, methodology validation, run-dir map)
- [`PAPER_CLOSURE_PHASES.md`](PAPER_CLOSURE_PHASES.md) — phase plan + cancellation log
- [`NORTH_STAR.md`](NORTH_STAR.md) — both reframe banners + cross-state Δ tables at top
- [`HANDOVER.md`](HANDOVER.md) — TL;DR pillars updated; "Recipe-selection reframe" + "External resources" sections; "Done 2026-05-01" + "Deferred / camera-ready" lists
- [`OBJECTIVES_STATUS_TABLE.md`](OBJECTIVES_STATUS_TABLE.md) — F49 Layer 3 absolute-claim refutation banner; mechanism intact
- [`PAPER_PREP_TRACKER.md`](PAPER_PREP_TRACKER.md) — both reframe banners
- [`FOLLOWUPS_TRACKER.md`](FOLLOWUPS_TRACKER.md) — top-level paper-closure pointer with both reframes summarized
- [`results/RESULTS_TABLE.md`](results/RESULTS_TABLE.md) — leak-free Δ table + recipe-selection table; full v6 body rewrite deferred

---

## 8 · External resources (Drive backup)

The user maintains a Drive folder aggregating result snapshots from other
machines (Colab T4, RunPod 4090, A100):
**https://drive.google.com/drive/folders/1cka4py5MElM-mDbBW5JC8JLS36qnjmq-**

Consult when a referenced run dir (e.g. F51 multi-seed `_20260430_05XX`) is no
longer local. Drive needs human auth; download the relevant subfolder and place
under `results/check2hgi/<state>/` preserving the original run-dir name so
analysis scripts find it.

---

## 9 · Reproducibility — how to redo the paper-closure analysis

```bash
# 1. Regenerate Wilcoxon JSONs from existing run dirs (no re-training needed).
PYTHONPATH=src python scripts/analysis/paper_closure_wilcoxon.py
# Writes docs/studies/check2hgi/research/PAPER_CLOSURE_WILCOXON.json

# 2. (B9 vs H3-alt across 5 states) — currently inline in the session log;
#    see SESSION_HANDOFF if pulling out as a script.

# 3. Verify methodology against F51 reference (sanity check):
PYTHONPATH=src python -c "
import pandas as pd, statistics
from pathlib import Path
d = Path('results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260430_0110')
folds = []
for f in (1,2,3,4,5):
    df = pd.read_csv(d/'metrics'/f'fold{f}_next_region_val.csv')
    folds.append(float(df[df.epoch>=5]['top10_acc_indist'].max()))
print(f'reg = {statistics.mean(folds)*100:.2f} ± {statistics.stdev(folds)*100:.2f}')
# Expected: 63.14 ± 1.15 (F51 published 63.47 ± 0.75 — within 0.3 pp)
"

# 4. (Optional) Rerun the run matrix from scratch: scripts/run_*.sh
#    All paper-closure launchers preserved at scripts/run_paper_closure_*.sh
#    and scripts/run_f51_tier3_sweep.sh.
```
