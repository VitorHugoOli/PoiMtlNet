# F50 T4 — Re-Run Decision Matrix (2026-04-29 20:00 UTC)

**Question:** Given that the C4 leak is *approximately* uniform (drops vary ~4 pp across recipes from −12.84 to −17.04 measured), which F50 ablations are worth re-running for new evaluation? Which can keep their pre-C4 numbers with a footnote?

**Decision rule:** Re-run if `|leaky Δ| < 5 pp` (small Δs are fragile to a 4 pp re-shuffling). Skip if `|leaky Δ| > 5 pp` (sign and magnitude robustly preserved). Always re-run anything load-bearing for the paper's headline.

---

## 1 · The uniform-leak observation

| recipe | pre-C4 reg | post-C4 reg (clean) | drop |
|---|---:|---:|---:|
| H3-alt CUDA REF (≥ep10) | 71.44 | 49.42 | −22.02* |
| H3-alt CUDA REF (≥ep5) | 74.72 | 60.12 | **−14.60** |
| P4 + Cosine (≥ep5) | 78.55 | 63.23 | **−15.32** |
| P4 + Cosine (≥ep10) | 75.48 | 60.42 | **−15.06** |
| P4 alone (≥ep10) | 75.48 | 60.42 | **−15.06** |
| Cross-state GA, AL, AZ | n/a leaky | per-fold from start | — |

(* ≥ep10 H3-alt drop is anomalous because H3-alt clean has σ=10.67 at ≥ep10 — late-fold collapse. ≥ep5 is the right comparison window for clean conditions.)

**Variance ~0.7 pp at ≥ep5** across measured runs (very tight). The "uniform leak" hypothesis holds **strongly** for the H3-alt-family of recipes (`next_getnext_hard` head + full log_T). Recipes within ±0.7 pp leaky → likely within ±0.7 pp clean → **smaller Δs are NOT robust**.

---

## 2 · Re-run priority (sharp, sourced from `F50_CORRECTED_SCOREBOARD.md`)

### ✅ DO NOT re-run (verdict robust to ±1 pp leak shift)

| recipe | leaky Δ | leaky verdict | reason |
|---|---:|---|---|
| T1.2 HSM | −5.16 | tied/negative | |Δ| > 4× the leak variance; preserved |
| A1 onecycle50 | −9.00 (≥ep10) / −3.35 (≥ep5) | OneCycle alone hurts | |Δ| huge |
| A2 cosine50 (≥ep10) | catastrophic σ=8.99 | collapse | preserved (collapse is recipe-property) |
| A6 cw0.25+onecycle | −9.22 / −3.14 | OneCycle alone hurts | preserved |
| D6 reg_head_lr=3e-2 | −59.87 / NaN | DIVERGED | preserved (diverged is unambiguous) |
| F50 D1 STL α=0 (frozen) | n/a | encoder-only ceiling 72.61 | **NO LEAK at all** (α=0 ⇒ log_T contributes 0); 72.61 is the real number |
| **All cross-state** (AL, AZ, GA) | already clean (per-fold log_T from start) | — | already in leak-free regime |

### 🎯 RE-RUN — load-bearing for paper headline

| # | recipe | leaky Δ | clean status | priority | est min |
|---|---|---:|---|---|---|
| 1 | **STL F37 ceiling** | n/a | 🟡 queued (`tmux stl_clean`) | **TIER 0** — paper headline number | 19 |
| 2 | **B9 (P4+Cosine+α-no-WD)** | n/a (new) | ✅ done (63.47) | TIER 0 | — |
| 3 | **P4-alone** | +3.83 | ✅ done (63.41, +3.28 clean) | TIER 0 | — |
| 4 | **P4+Cosine champion** | +4.63 | ✅ done (63.23, +3.11 clean) | TIER 0 | — |
| 5 | **P4+OneCycle "max-reg" alt** | +6.08 (Pareto-trade) | 🟡 running now | TIER 0 | ~17 |
| 6 | **F62 two-phase (mode=step)** | NEVER ran | 🟡 queued (`tmux f62_clean`) | TIER 0 | 19 |

### ⚠ RE-RUN — small leaky Δ that could flip under clean

These are within the ~0.7 pp leak-variance band and could re-rank under clean conditions. Re-run **one representative each** to confirm, not all of them.

| recipe | leaky Δ | re-run? | rationale |
|---|---:|---|---|
| T1.4 Aligned-MTL | +0.45 | ⚠ **yes — single representative** | Closest-to-paper-grade non-P4 architectural alt |
| **PLE-lite** | +0.25 | ⚠ **YES** (P0-E #73) | CH18 substrate claim depends on this |
| Cross-Stitch (default) | +0.26 | skip | tied with PLE; one rep is enough |
| Cross-Stitch detach | +0.01 | skip | tied |
| T1.3 FAMO | −0.66 | skip | tied; finding ("FAMO ≈ H3-alt") preserved by sign |
| P1 no_crossattn | −0.29 | skip | tied; cross-attn-removal claim doesn't depend on absolute |
| P2 detach K/V | −0.06 | skip | tied |
| P3 cat_freeze@10 | −0.14 | skip | tied |
| MTL cw=0.50 | +0.29 | skip | cw sweep finding preserved by trajectory |
| MTL cw=0.25 | +0.38 | skip | tied |
| MTL cw=0.0 (D8) | +0.30 | skip-ish | finding (ep-5 plateau) doesn't depend on absolute |
| D3 reg_enc_lr=3e-2 | −0.11 | skip | tied |
| D3 reg_enc_lr=1e-2 | −0.10 | skip | tied |
| A3 alpha=2.0 | −0.43 | skip | the init-artifact mechanism is preserved |
| A4 epochs100 | −0.05 | skip | tied |
| A5 stacked | −0.06 | skip | tied |

### 🔬 RE-RUN — different leak structure (audit findings)

The audit found `next_tgstan` has TWO trainable amplifiers (α + per-sample gate). Its leak could be NON-uniform — possibly bigger than the 13-17 pp we measured on `next_getnext_hard`. **This is a different epistemic question** ("does the leak bigger for some heads?") not "are the relative numbers preserved".

| # | recipe | leaky Δ | priority | est min |
|---|---|---:|---|---|
| 7 | **TGSTAN clean smoke** (1f×10ep) | not measured | **TIER 0 — second-most-likely C4-class leak** | 6 |
| 8 | **HSM head clean smoke** (1f×10ep) | n/a | TIER 1 — confirms uniform leak holds for parent/child classifier | 6 |

---

## 3 · Final re-run roster

After this analysis, **8 runs** total need to land for the paper:

| # | recipe | status | est min (cumulative) |
|---|---|---|---:|
| 1 | H3-alt clean (anchor) | ✅ done | — |
| 2 | P4-alone clean | ✅ done | — |
| 3 | P4+Cosine clean (P0-A) | ✅ done | — |
| 4 | P4+OneCycle clean | 🟡 running ~17 min | 17 |
| 5 | B9 alpha-no-WD clean | ✅ done | — |
| 6 | STL F37 clean | 🟡 queued | 36 |
| 7 | F62 two-phase clean | 🟡 queued | 55 |
| 8 | PLE-lite clean smoke (representative architectural alt) | ⏸ NOT QUEUED | 60 (+5 min smoke) |
| 9 | TGSTAN clean smoke | ⏸ NOT QUEUED | 65 |

**Skip 16 other F50 runs** — their relative Δs hold under uniform leak; they're either too far from paper-grade (negative) or already tied within noise.

**End of pipeline ~21:05 UTC** for current queue (P4+OneCycle, STL, F62). PLE + TGSTAN smokes can run after — total ~75 min more.

---

## 4 · What this means for the paper

**Headline numbers needed (all clean):**

| recipe | reg @≥ep5 | cat F1 | source |
|---|---:|---:|---|
| **STL F37 ceiling** | TBD (~66 estimated) | n/a | clean run queued |
| **MTL champion (P4-alone or B9)** | 63.41-63.47 | 67.82-68.59 | ✅ measured |
| **MTL H3-alt baseline** | 60.12 | 68.34 | ✅ measured |
| **Δ champion vs H3-alt** | +3.28 to +3.34 pp | within σ | paired Wilcoxon p=0.0312, 5/5 ✅ |

**The story doesn't change** — the paper still claims a paper-grade Δ. But:
- The "8.83 pp STL-MTL gap" framing is wrong; gap is ~3-6 pp clean.
- The "champion = 76.07" headline becomes "champion = 63.4".
- The "10 architectural alternatives all fail" claim is preserved by uniform leak (one representative re-run will confirm).

The most COMPLEX result (B9 Pareto-dominant on BOTH tasks) needed clean baselines; we have them. The recipe story SIMPLIFIES under clean conditions because cosine and alpha-no-WD don't matter much — **P4 alone is the intervention.**

---

## 5 · Action items right now

1. ✅ Documented validity matrix in `F50_T4_PRIOR_RUNS_VALIDITY.md`
2. ✅ Documented leak diagnosis in `F50_T4_C4_LEAK_DIAGNOSIS.md`
3. ✅ Documented broader leakage audit in `F50_T4_BROADER_LEAKAGE_AUDIT.md`
4. **THIS DOC** ✅: re-run decision matrix
5. ⏸ Queue PLE-lite clean (TIER 1 representative architectural verification)
6. ⏸ Queue TGSTAN smoke (TIER 0 hidden risk)
7. After all queued runs land: write the consolidated F50 T4 SYNTHESIS doc
