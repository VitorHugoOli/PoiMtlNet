# BASELINE handoff — **A40** (CUDA, Ampere) · self-contained · 2026-06-25

> **You are the A40. Read ONLY this file, then execute.** Phase = baselines, final. Your NEW job: the **W6
> category-side encoder-isolation probe** (closes the one open regular-track mechanism blocker). Decisions:
> `../../../articles/[mobiwac]/BASELINE_HANDOFF.md`; results → [`RESULTS_BOARD.md`](RESULTS_BOARD.md).
>
> ✅ **CSLSL cascade (the old A40 job) is DONE** — AL/AZ/FL all tie our parallel champion-G (Δjoint ≤ 0.02 pp;
> §1b). ⏹ **STOP the FL same-device champ-G comparand** (user, 2026-06-25): its first folds already reproduce the
> board champ-G, so it only confirms the cross-device ±0.01 FL tie we already have — no marginal gain. The §1b FL
> cascade Δ stays cross-device (a tie is a tie; the same-device comparand is a post-deadline nicety). Free the card.
>
> **Protocol:** seed 0 × 5 folds (n=5), gated stride-1 overlap `check2hgi_dk_ovl`, **fp32**, leak-free, user-disjoint.

## 0 · Setup (once)
```bash
cd <A40 repo>; git checkout main && git pull       # MUST include the --freeze-reg-stream commit (ae898042)
export PYTHONPATH=src
export DISABLE_AMP=1 MTL_DISABLE_AMP=1              # PR #43 fp16 gate (fp32, no autocast)
export MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_board
```

## 1 · What the W6 probe is (why it matters)
The paper's load-bearing mechanism claim: the joint **category** win is "**a stronger shared encoder, NOT
region→category transfer**" (PAPER_PLAN §6.2, answers R3's structural-bottleneck question). The only probe on file
(F49) measures the OPPOSITE direction. This run tests the claim directly: freeze the **region** stream
(`next_encoder` + `next_poi`, requires_grad=False) so it can't co-adapt as a cat-helper via cross-attn K/V, and
zero the reg loss (`--category-weight 1.0`). Then train champion-G and read the **cat** head:
- **probe cat F1 ≈ full-MTL cat ≫ STL cat ceiling** → the win is the shared TRUNK (architecture) → **W6 closed**.
- **probe cat F1 ≈ STL cat ceiling** → the win WAS region→category transfer (the claim must change).

> 📋 **Full step-by-step run-spec: [`HANDOFF_A40_W6_PROBE.md`](HANDOFF_A40_W6_PROBE.md)** (prereqs, smoke, sanity
> gates, scoring table, verdict interpretation). The summary below is the quick version.

## 2 · Task — run the probe (seed 0 × 5f, AL/AZ/FL)
```bash
STATES="alabama arizona florida" bash scripts/run_freeze_reg_probe.sh
#  (build dk_ovl + per-fold log_T first per state if absent — the runner/preflight tells you;
#   smoke first: MODE=smoke bash scripts/run_freeze_reg_probe.sh)
```
The flag + runner landed in commit `ae898042` (`--freeze-reg-stream`; a runtime guard RuntimeErrors at fold-0 if
the reg group still has trainable params, so a non-propagating freeze can't pass silently). The cat-ceiling
comparand is on disk (`docs/results/closing_data/h100/<state>_s0_stl_cat_ceiling.json`) and is device-robust
(7 classes) → the A40 is fine for this.

## 3 · Validation + outputs
- **Sanity:** the `[per-head-LR]` line must show the reg group(s) at **0 trainable params** (else the freeze
  didn't take — the run aborts with a RuntimeError). Cat head trains normally.
- **Read:** probe cat macro-F1 per state vs (a) the STL cat ceiling and (b) the full-MTL cat (RESULTS_BOARD §1).
  Commit a small result JSON per state; record the verdict (trunk vs transfer) + the three numbers in a new
  `docs/studies/closing_data/W6_ENCODER_ISOLATION.md` + RESULTS_BOARD. **n=5 provisional.**
- Do **NOT** run the FL representation block (H100), the CTLE diagnosis (M4/done), or any dropped baseline.
