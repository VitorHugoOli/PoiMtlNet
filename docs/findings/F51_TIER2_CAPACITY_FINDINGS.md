# F51 Tier 2 — encoder/backbone capacity smokes (DONE 2026-04-30 16:33 UTC)

**Status:** ✅ done. 21/21 smokes complete in ~4 h on 4090. **One PROMOTE candidate (Pareto trade) + two regressions + three cat-collapse modes + 15 ties.**

**Mission (from `F50_NORTH_STAR_DEEP_EXPLORATION_PROMPT.md` §3 Tier 2):** "Reg-side encoder saturation (D5 finding) is partly a capacity bottleneck. Larger reg encoder might delay saturation."

**Setup:** 5f×30ep at FL with seed=42, B9 recipe, one capacity knob altered per smoke. Reference: B9 seed=42 capped @≤ep30 (matches smoke window): reg 63.47 ± 0.75 / cat 68.06 ± 0.94. Decision rule: promote iff Δreg ≥ +0.5 pp OR Δ(reg-best epoch) ≥ 1.

> ⚠ **Why 5f×30ep, not 1f×50ep?** The original prompt suggested 1-fold smokes. But `--folds 1` triggers `n_splits = max(2, 1) = 2` in the trainer, producing a 2-fold split that doesn't match the 5-fold-keyed per-fold log_T (the F51 §0 fix would prevent silent leak now, but the smokes simply couldn't run with the 5-fold log_T). Using 5f×30ep keeps the smoke internally consistent with the seed-correct log_T, at the cost of ~10 min/smoke instead of ~3 min. The 30-ep window captures B9's reg-best window (B9 reg-best ep is {6,6,5,5,5} on seed=42 — well below 30).

---

## 1 · The PROMOTE candidate (Pareto trade)

| knob | reg ± σ | Δreg | reg-best ep | cat ± σ | Δcat | verdict |
|---|---:|---:|---:|---:|---:|---|
| `num_crossattn_blocks=3` | 64.22 ± 1.16 | **+0.75** | 5.6 | 65.44 ± 1.47 | **-2.62** | **Pareto-trade** |

**Per-fold breakdown** (B9 ref seed=42 vs `num_crossattn_blocks=3`):

| fold | Δreg | Δcat |
|---|---:|---:|
| 1 | +1.45 | -5.69 |
| 2 | +0.10 | -2.54 |
| 3 | +0.72 | -0.92 |
| 4 | +1.54 | -2.97 |
| 5 | -0.05 | -0.98 |

4/5 folds improve reg (mean +0.75 pp, p=0.0625), 5/5 folds degrade cat (mean -2.62 pp). Same structural pattern as F50's P4+OneCycle ⚠ trade.

**Mechanism (NEW finding, refines F52's "mixing is dead at FL"):**
- F52 P5 ran 2 cross-attn blocks with **identity mixing** (zeroed K/V output) and tied B9 → mixing is dead at depth=2
- This Tier 2 result: 3 blocks with mixing on gives Δreg +0.75 pp, with cat trade
- 4 blocks with mixing on: cat collapses entirely

→ Cross-attention mixing is **suppressed by depth**, not absent. F52's "mixing is dead" claim should be qualified as "mixing is dead at B9's 2-block depth"; at 3 blocks mixing contributes a small reg lift but breaks cat. **B9's 2-block default is the Pareto-stable point.**

**Paper implication:** F52's three-way confirmation (P1 / P5 / F53 cw sweep) of "cross-attn mixing structurally dead" should be qualified as depth-conditional. The mechanism narrative gains a third supporting receipt: cross-attn capacity has a sharp Pareto cliff at B9's 2-block / 256-dim setting.

---

## 2 · Cat collapse modes — wider shared backbone breaks cat training

Three knobs catastrophically destabilized cat without affecting reg:

| knob | reg ± σ | cat mean | cat σ | failure mode |
|---|---:|---:|---:|---|
| `shared_layer_size=384` | 63.26 ± 0.73 | 58.52 | **19.54** | fold 2 collapses to F1=23.6 (others ~67); 4/5 OK |
| `shared_layer_size=512` | 63.12 ± 0.65 | **8.46** | 0.18 | **5/5 folds** fail to learn cat; loss frozen at ~1.94 from ep 1 |
| `num_crossattn_blocks=4` | 63.22 ± 0.81 | 29.89 | **29.35** | multi-fold collapse (some at ~7-15%, others recover to ~57%) |
| `crossattn_ffn_dim=1024` | 62.85 ± 0.66 | 54.43 | **25.87** | multi-fold collapse (1 fold ~7%, others ~63-67%) |

**Reg unaffected in all four cases** (within ±0.6 pp of B9 ref's 63.47).

**Mechanism interpretation:**
- All four knobs widen/deepen the shared backbone past B9's defaults (256-dim, 2 blocks, 256 FFN-dim)
- Reg path is shielded: per-head reg LR (3e-3) + P4 alternating-SGD updates reg-side params on dedicated batches with no cat gradient interference
- Cat path is not shielded: per-head cat LR (1e-3) is conservative for a wider shared layer + cat FFN. The FFN dynamics depend on the shared LayerNorm input distribution; widening shifts the distribution and cat's lower LR cannot adapt within the cosine schedule
- This is consistent with F50 D5's "cat encoder keeps drifting through ep 38" — cat needs a stable backbone width to adapt slowly

**Paper implication:** **B9's 256-dim shared backbone is locally optimal on the capacity axis** — not just historical default. Adds a third Pareto-worse direction (alongside PLE expert routing and F62 two-phase scheduling). The temporal-dynamics narrative gains a stability dimension: B9 sits at a width-stability margin where reg is shielded and cat fits.

---

## 3 · Regressions (Δreg < -0.5 pp)

| knob | Δreg | Δcat | interpretation |
|---|---:|---:|---|
| `num_encoder_layers=4` | -0.60 | -2.07 | Adding 4-layer per-task encoder on top of B9's 2-layer hurts both — overfit/over-regularization regime |
| `crossattn_ffn_dim=1024` | -0.62 | -13.63 (collapse) | Already noted in §2; reg also drops -0.62 pp |

These are dominated knobs — do NOT combine with B9.

---

## 4 · Ties (Δreg within ±0.5 pp, no cat collapse)

15/21 smokes were tied with B9 — most of B9's defaults are insensitive to ±50% perturbation:

| knob class | levels tested | result |
|---|---|---|
| `encoder_layer_size` | 128, 256(B9), 384, 512 | tied across all (range -0.41 to +0.0) |
| `num_encoder_layers` | 1, 2(B9), 3 | tied; 4 regresses |
| `encoder_dropout` | 0.05, 0.1(B9), 0.2, 0.3 | tied on reg; cat degrades monotonically with dropout |
| `shared_layer_size=128` | 128 (vs 256 B9) | tied; 384/512 collapse cat |
| `num_crossattn_blocks=1` | 1 (vs 2 B9) | tied; 3 = Pareto-trade; 4 collapses cat |
| `num_crossattn_heads` | 2, 4(B9), 8, 16 | tied across all |
| `crossattn_ffn_dim` | 128, 256(B9), 512 | tied; 1024 regresses + collapses cat |

→ B9's defaults are an **insensitive plateau** in 5 of 7 dimensions. The architecture has very little headroom to improve via capacity scaling.

---

## 5 · Mechanism summary — what Tier 2 tells us

1. **Reg encoder is NOT capacity-bound.** Halving / 1.5×/2×-ing encoder layer size or depth changes Δreg by < 0.5 pp. Confirms F50 D5: reg encoder saturation is **temporal**, not capacity (encoder physically stops drifting at ep 5–6 regardless of width).
2. **Cross-attn mixing is dead at depth=2 (B9), but not at depth=3.** Adding a 3rd block recovers a +0.75 pp reg lift, suggesting mixing has a real but small contribution that B9 deliberately suppresses (in exchange for cat stability).
3. **Cat training has a sharp width-stability cliff.** Three widening knobs (shared_layer_size 384/512, num_crossattn_blocks=4, crossattn_ffn_dim=1024) all catastrophically break cat without affecting reg. P4 alternating-SGD + higher per-head reg LR shields reg; cat has no such shield.
4. **B9 is locally optimal on 5 of 7 capacity dimensions.** The architecture has no obvious capacity-scaling lift available.

**Net implication:** Tier 2 closes the architecture-via-capacity-scaling track. The +0.75 pp `num_crossattn_blocks=3` lift is **NOT promoted to paper-grade** — the cat trade is too large to be Pareto-positive (same disposition as P4+OneCycle in F50).

---

## 6 · No promotion to multi-seed paper-grade

The single PROMOTE candidate (`num_crossattn_blocks=3`) is **Pareto-trade**, not Pareto-positive. To be paper-grade-eligible, a Tier 2 candidate would need:
- Δreg ≥ +0.5 pp paper-grade Wilcoxon
- Δcat ≥ -0.3 pp (within 1 σ of B9 cat noise)

`num_crossattn_blocks=3` fails the cat criterion (-2.62 pp, outside σ). **No knob promoted to multi-seed validation.**

---

## 7 · Cross-references

- `F51_MULTI_SEED_FINDINGS.md` — Tier 1 multi-seed validation (paper-grade headline)
- `F50_T4_FINAL_SYNTHESIS.md` §1 — B9 champion baseline
- `F50_D5_ENCODER_TRAJECTORY.md` — reg encoder saturation receipt (Tier 2 confirms it's temporal, not capacity)
- `F50_B2_F52_F65_F53_FINDINGS.md` §2 — F52 P5 "mixing is dead at depth=2" (Tier 2 refines this to depth-conditional)
- `F51_NORTH_STAR_DEEP_EXPLORATION_PROMPT.md` §3 Tier 2 — exploration spec
- Code:
  - Sweep runner: `scripts/run_f51_tier2_capacity_smoke.sh` (5f×30ep, B9 base)
  - Analyzer: `scripts/analysis/f51_tier2_analysis.py`
- Structured output: `F51_tier2_results.json`
- Logs: `logs/f51_tier2_master.log`, `logs/f51_t2_<knob>.log` (21 files)
- Run dirs: `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep30_2026*` (21 dirs)
