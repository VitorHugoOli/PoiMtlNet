# W6 — category-side encoder-isolation probe (A40, 2026-06-25) — **CLOSED**

> **Verdict: W6 CLOSED. The joint CATEGORY win is the shared TRUNK (architecture/capacity), NOT
> region→category transfer.** With the region stream frozen-at-init (no learned region signal, cannot
> co-adapt as a cat-helper via cross-attention K/V) and the reg loss off, the category head keeps the **entire**
> joint lift: probe cat ≈ full-MTL cat (±0.3 pp) and **≫ the STL cat ceiling (+4.6 … +7.6 pp)** at all three
> states. This is the direct, correct-direction evidence for the paper's §6.2 mechanism sentence (the only
> prior probe, F49, freezes the *cat* stream → measures the region pathway, the opposite direction).

## 1 · The test
Freeze the **region** stream (`next_encoder` + `next_poi`, `requires_grad=False` → stuck at init) via
`--freeze-reg-stream`, and zero the reg loss (`--category-weight 1.0`). Train champion-G, read the **category**
head. Comparand = the **STL cat ceiling** (dedicated single-task category, on disk) and **full-MTL cat**
(RESULTS_BOARD §1). Mechanism: a frozen-at-init region stream provides no learned region signal and cannot
co-adapt through the cross-attention K/V, so any cat lift over the STL ceiling must come from the shared trunk.
- probe cat ≈ full-MTL cat ≫ STL ceiling  → **shared trunk** (architecture), not transfer → **W6 closed**.
- probe cat ≈ STL ceiling                 → the cat win **was** region→category transfer → rewrite §6.2.

## 2 · Result (seed 0 × 5f, A40, true fp32, region stream frozen)
| State | STL cat ceiling | full-MTL cat (§1) | **probe cat (freeze-reg)** | Δ vs ceiling | Δ vs full-MTL | verdict |
|---|---:|---:|---:|---:|---:|---|
| AL | 55.87 | 63.56 | **63.50 ±1.74** | **+7.63** | −0.06 | **trunk** ✅ |
| AZ | 57.13 | 63.39 | **63.67 ±1.28** | **+6.54** | +0.28 | **trunk** ✅ |
| FL | 75.15 | 79.82 | **79.79 ±0.46** | **+4.64** | −0.03 | **trunk** ✅ |

cat = macro-F1 at the f1-best epoch, fold-mean ±pstd (matched scorer `a40_score_matched.py`). Per-fold probe cat:
AL [63.59,64.73,64.82,64.20,60.14] · AZ [65.38,62.67,64.88,63.38,62.03] · FL [79.47,79.90,79.66,79.30,80.62].
Best-epochs late (FL 47–50; AL/AZ 14–23) — normal champion-G cat trajectory, no NaN/early-collapse.

**Reading:** at every state the probe cat is statistically indistinguishable from full-MTL cat (Δ ≤ 0.3 pp ≪
fold-std) and clears the STL cat ceiling by **+4.6 … +7.6 pp**. The category lift survives **in full** with the
region stream frozen → it is the stronger **shared encoder on its own** (a trunk/capacity effect), not
region→category transfer. The paper's §6.2 mechanism sentence stands as written; cite this (category-side,
direct) probe in place of / alongside the F49 (region-side) citation.

## 3 · Sanity gates (all passed)
- **Freeze took** — every state's first-fold `[per-head-LR] optimizer groups` shows the **reg group at 0
  trainable params** (`('reg', …, 0)`); cat 1,731,079 + shared 1,584,128 train. 0 freeze RuntimeErrors.
- **Cat head trains normally** — late best-epochs, **0 non-finite skips** (true fp32 via `DISABLE_AMP=1`),
  no ep12 collapse.
- reg metric ignored (reg frozen + loss off → meaningless), per design.

## 4 · Provenance (reproduce)
- Runner `scripts/run_freeze_reg_probe.sh` (champion-G v16 on `check2hgi_dk_ovl` + `--category-weight 1.0
  --freeze-reg-stream`, seed 0 × 5f, `--compile --tf32`, true fp32 `DISABLE_AMP=1 MTL_DISABLE_AMP=1`).
  **Runner fix this session** (commit in this PR): added `--canon none` + 3 omitted v16 flags
  (`--checkpoint-selector geom_simple --no-{reg,cat}-class-weights`) + dk_ovl log_T dir — without `--canon
  none` the runner's own `MTL_STRICT=1` hard-failed the v16-vs-dk_ovl wrong-substrate guard before training.
- Substrate: `check2hgi_dk_ovl` (v14 design_k re-windowed gated stride-1, MIN_SEQ=10); AL/AZ rebuilt + FL log_T
  refreshed this session (the log_T is **unused** here — reg loss off + α frozen — but the preflight wants it
  fresh). Flag + runner shipped in `ae898042`.
- JSONs: `docs/results/closing_data/a40/{al,az,fl}_w6_freezereg_s0.json`. STL cat ceilings:
  `docs/results/closing_data/h100/<state>_s0_stl_cat_ceiling.json` (7-class, device-robust → A40 fine).
- **n=5 provisional** (seed 0; {1,7,100}→n=20 post-deadline).

## 5 · For the paper (do NOT spin)
Expected result obtained: the cat win is architecture, not transfer. Use as the encoder-isolation evidence in
§6.2 (R3 structural-bottleneck / REVIEW_PANEL W6). Honest framing: "with the region stream frozen at
initialization, the shared encoder alone recovers the full multi-task category gain (+4.6…+7.6 pp over the
dedicated single-task ceiling) — the category benefit is a stronger shared representation, not cross-task
(region→category) transfer."
