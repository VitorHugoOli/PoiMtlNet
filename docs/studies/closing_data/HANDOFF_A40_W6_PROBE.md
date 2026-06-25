# HANDOFF — A40 · W6 category-side encoder-isolation probe · self-contained · 2026-06-25

> **You are the A40. Read ONLY this file, then execute.** ONE job: run the **category-side encoder-isolation
> probe** that closes the last open regular-track mechanism blocker (W6). The flag (`--freeze-reg-stream`) + runner
> already shipped (commit `ae898042`). Decisions: `../../../articles/[mobiwac]/BASELINE_HANDOFF.md`; results →
> [`RESULTS_BOARD.md`](RESULTS_BOARD.md).
>
> ✅ Your prior jobs are DONE/stopped: CSLSL cascade (AL/AZ/FL tie, merged §1b); the FL same-device champ-G
> comparand was STOPPED (it only re-confirmed the cross-device ±0.01 FL tie — no gain). The card is free for this.

## 1 · Why this run (the claim it defends)
The paper's load-bearing **mechanism** sentence: the joint **category** win is "**a stronger shared encoder, NOT
region→category transfer**" (PAPER_PLAN §6.2; answers R3's structural-bottleneck question, W6). The only probe on
file (F49) freezes the *cat* stream — it measures the **region** pathway, the OPPOSITE direction. A reviewer who
reads F49 catches the mismatch and W6 stands. **This run tests the claim directly**, in the right direction.

**Mechanism of the test:** freeze the **region** stream (`next_encoder` + `next_poi`, requires_grad=False) so it
is stuck at init and **cannot co-adapt as a cat-helper via cross-attention K/V**, and zero the reg loss
(`--category-weight 1.0`). Train champion-G, read the **category** head:
- **probe cat F1 ≈ full-MTL cat, ≫ STL cat ceiling** → the cat win is the shared **TRUNK** (architecture/capacity) → **W6 CLOSED**.
- **probe cat F1 ≈ STL cat ceiling** → the cat win **was** region→category transfer → the claim must be rewritten.

## 2 · Setup (once)
```bash
cd <A40 repo>; git checkout main && git pull        # MUST include ae898042 (--freeze-reg-stream) + the --help fix
export PYTHONPATH=src
export DISABLE_AMP=1 MTL_DISABLE_AMP=1               # PR #43 fp16 gate → true fp32, no autocast (board protocol)
export MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_board
# sanity: the flag is registered
python scripts/train.py --help | grep -q -- --freeze-reg-stream && echo "flag present ✓"
```

## 3 · Prereqs per state (build if absent — same as the board)
For each of {alabama, arizona, florida}: the dk_ovl base + the seeded per-fold log_T must exist (the log_T is
unused here — reg loss is off + alpha frozen — but the engine/preflight wants it present and fresh):
```bash
python scripts/mtl_improvement/build_overlap_probe_engine.py <state> 1          # if dk_ovl base absent
python scripts/compute_region_transition.py --state <state> --engine check2hgi_dk_ovl --per-fold --seed 0 --n-splits 5
```
v14 substrate + check2hgi graph maps must be on disk (FL ~3.3 G; from SSD or rebuild). Keep ≥30 G free for FL.

## 4 · Run (seed 0 × 5f, AL/AZ/FL) — smoke first
```bash
MODE=smoke bash scripts/run_freeze_reg_probe.sh                                  # AL, 1 fold × 2 ep — sanity only
STATES="alabama arizona florida" bash scripts/run_freeze_reg_probe.sh           # the real run
```
The runner pins champion-G on `check2hgi_dk_ovl` + `--category-weight 1.0 --freeze-reg-stream` (minus the cascade
pins), fp32. **Cheap** — cat is the 7-class head; no wide-region accumulation. (FL is the long pole; AL/AZ fast.)

## 5 · Sanity gates (STOP if any fails)
- **The freeze must take.** The first-fold `[per-head-LR] optimizer groups (name, lr, trainable_params)` line must
  show the reg group(s) (`reg` or `reg_encoder`+`reg_head`) at **0 trainable params**. If not, the run aborts with
  a `RuntimeError` ("freeze_reg_stream=True but the optimizer's reg group(s) still contain trainable params") — do
  not paper over it.
- **Cat head trains normally** (best-epochs late, no NaN/ep12 collapse — the #43 fp16 gate is on via DISABLE_AMP).
- The reg metric is meaningless here (reg frozen + loss off) — **ignore reg; read only the cat head.**

## 6 · Score + record
Extract **cat macro-F1** (5f mean) per state from the rundir, compare against the on-disk numbers:

| State | STL cat ceiling | **full-MTL cat** (RESULTS_BOARD §1) | probe cat (freeze-reg) | verdict |
|---|---:|---:|---:|---|
| AL | 55.87 | 63.56 | ⟨fill⟩ | trunk if ≫55.87 |
| AZ | 57.13 | 63.39 | ⟨fill⟩ | trunk if ≫57.13 |
| FL | 75.15 | 79.82 | ⟨fill⟩ | trunk if ≫75.15 |

Ceilings: `docs/results/closing_data/h100/<state>_s0_stl_cat_ceiling.json` (device-robust, 7 classes → A40 fine).
Write the three probe numbers + the verdict into a new **`docs/studies/closing_data/W6_ENCODER_ISOLATION.md`** and
a one-line entry in `RESULTS_BOARD.md` (§1 reading or a new §1c). Commit a small result JSON per state. **n=5
provisional** (seed 0; {1,7,100}→n=20 post-deadline).

## 7 · Interpreting the result (for the paper)
- **Expected (and what closes W6):** probe cat lands **near the full-MTL cat and well above the STL ceiling** →
  the shared trunk is a stronger category encoder *on its own*; the cat win is architecture, not transfer. Cite
  this as the encoder-isolation evidence in §6.2 (replacing/augmenting the F49 citation, which is region-side).
- **If probe cat drops toward the ceiling** → the cat win depended on the region task → the mechanism sentence must
  change to "joint training (incl. region) lifts category"; flag immediately (it reshapes contribution-2's prose).
- Either way it is a real, defensible result — do **not** spin it. Report the numbers + the honest reading.

## 8 · Do NOT
Run the FL representation block (H100), the CTLE diagnosis (M4/done), or any dropped baseline. Do not commit to
main / merge. Work on your branch + open a PR (as before); the orchestrator audits + records the verdict.
