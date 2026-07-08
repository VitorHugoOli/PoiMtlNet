# M1-PARTIAL — v17 pre-registered stats vs the n=20 best-vs-best ceilings (2026-07-08, rev 2)

> ⚠ **This is M1-PARTIAL.** It covers the 4 datasets that are fully n=20 on both sides today
> (**AL, AZ, FL, Istanbul** — ceilings per [`../CEILINGS_N20_FINAL.md`](../CEILINGS_N20_FINAL.md),
> **AZ ceiling = 56.43, the corrected bs8192@0.005 arm**). **CA/TX are pending A1** (v17 MTL n=20
> on the A40). The paper's headline family is 6 datasets (protocol §5.2) — **the full 6-dataset
> family Holm re-runs after A1**; everything here is the pre-registered analysis of the committed
> subset, not the final family verdict.
>
> **rev 2 (2026-07-08):** A0 (`c51b1183`) committed the AL/AZ/FL v17 MTL per-seed values
> (`train_perf_multifold/n20_perhead_runs/summary.tsv`) → the AL/AZ/FL cells flip from PENDING to
> tested at the **seed-level (n=4)** footing, same as Istanbul. All four datasets now carry verdicts.
>
> Pre-registration: [`../../STATISTICAL_PROTOCOL.md`](../../STATISTICAL_PROTOCOL.md) (§2 cat
> superiority → paired one-sided Wilcoxon; §3 reg → TOST non-inferiority at **δ_reg = 2 pp**;
> §4 pairing discipline; §5.2 Holm on the cat family only; §8 deviation log). Test conventions
> mirror `scripts/closing_data/superiority_wilcoxon.py` + `region_match_tost.py`.
>
> Reproduce: `.venv/bin/python docs/studies/closing_data/v17_completion/stats_n20/m1_stats_n20.py`
> (reads ONLY committed artifacts; aborts if any recomputed aggregate stops matching the board).

## 0 · Artifact → board reproduction gate (16/16 PASS)

Every board number used below was recomputed from the committed artifacts before testing:
STL cat ceilings AL 56.815 / AZ 56.435 / FL 74.509 (`../cat_ceiling_sweep/sweep_results/` per-fold
JSONs, 4 seeds × 5 folds); STL reg ceilings AL 70.111 / AZ 59.459 / FL 76.697 / Istanbul 75.158
(`docs/results/P1/region_head_*_ovl_stl_reg_{s0,topup_s{1,7,100}}.json` per-fold arrays); **v17 MTL
AL 64.540/69.801, AZ 65.835/59.563, FL 79.848/77.421 (the A0 `summary.tsv` per-seed rows, `recipe=new`
= the bs8192 + per-head cat-lr 1e-3 v17 arm — the FL `recipe=base` bs2048 anchor rows are excluded)**;
Istanbul MTL 63.329/75.440 + cat ceiling 54.738 (`../h3_istanbul/step3_runs/*.txt`). All match
`CEILINGS_N20_FINAL.md` / `perhead_lr_n20.md` to rounding.

## 1 · Per-dataset results

Δ orientation = **MTL − STL ceiling** (positive = MTL better). δ_reg = 2 pp (pre-registered, §3.2).
**Pairing level for every cell: SEED-LEVEL paired, n=4** — each observation is a per-seed 5-fold
mean, paired by seed {0,1,7,100} across arms sharing the frozen fold construction (§4). Per-fold
(n=20) pairing is not yet possible from the committed tree (LIMITS #2).

### Category — superiority (MTL > dedicated STL cat ceiling)

| Dataset | Δcat (pp) | pairs+ | Wilcoxon p (exact, 1-sided) | paired t(3) p (1-sided) | Holm-adj p (m=4, on t) | Verdict |
|---|---:|---|---|---|---|---|
| **AL** | **+7.73 ± 0.12** | 4/4 | 0.0625 (= n=4 floor) | 4.5e-07 (≈67 σ_d) | **5.3e-07** ✓ | **outperforms** |
| **AZ** | **+9.40 ± 0.11** | 4/4 | 0.0625 (= n=4 floor) | 2.1e-07 (≈87 σ_d) | **5.3e-07** ✓ | **outperforms** |
| **FL** | **+5.34 ± 0.02** | 4/4 | 0.0625 (= n=4 floor) | 4.2e-09 (≈320 σ_d) | **1.7e-08** ✓ | **outperforms** |
| **Istanbul** | **+8.59 ± 0.09** | 4/4 | 0.0625 (= n=4 floor) | 1.8e-07 (≈92 σ_d) | **5.3e-07** ✓ | **outperforms** |

All four survive Holm at α = 0.05 on the paired-t footing (see deviation log — at n=4 the exact
Wilcoxon is floor-limited at 0.0625 for **any** effect size, so the pre-registered Wilcoxon itself
cannot clear α until the per-fold n=20 vectors land).

### Region — TOST non-inferiority (δ_reg = 2 pp; equivalence cells, NOT in the cat Holm family per §5.2)

| Dataset | Δreg (pp) | TOST p | 90 % CI (pp) | Verdict |
|---|---:|---|---|---|
| **AL** | **−0.31 ± 0.15** | 8.8e-05 | (−0.482, −0.139) ⊂ ±2 | **matches (TOST)** |
| **AZ** | **+0.10 ± 0.09** | 1.3e-05 | (+0.001, +0.206) ⊂ ±2 | **matches (TOST)** (CI ≥ 0 but grazes it — do NOT upgrade to "beats"; supplementary superiority t p = 0.049, marginal) |
| **FL** | **+0.72 ± 0.04** | 5.8e-06 | (+0.671, +0.776) ⊂ ±2 | **matches (TOST)** — CI entirely > 0, descriptively **beats** (suppl. t p = 3.2e-05) |
| **Istanbul** | **+0.28 ± 0.04** | 1.2e-06 | (+0.240, +0.323) ⊂ ±2 | **matches (TOST)** — CI entirely > 0, descriptively **beats** (suppl. t p = 2.7e-04) |

**Per-seed Δs (seeds 0/1/7/100)** — cat: AL [+7.73, +7.88, +7.60, +7.69], AZ [+9.40, +9.41, +9.26,
+9.53], FL [+5.32, +5.34, +5.33, +5.36], Istanbul [+8.61, +8.47, +8.59, +8.70]; reg: AL [−0.13,
−0.46, −0.39, −0.26], AZ [+0.23, +0.03, +0.05, +0.11], FL [+0.68, +0.71, +0.72, +0.78], Istanbul
[+0.25, +0.31, +0.25, +0.32].

### Deviation log (protocol §8)

1. **Seed-level pairing (n=4) instead of the pre-registered per-fold n=20.** The committed MTL sides
   are per-seed 5-fold means (A0 `summary.tsv`; Istanbul step3 txts) — the per-fold matched-score
   sidecars remain A40-only (LIMITS #2). Seed-level pairing is valid (§4: same frozen folds, same
   seed set); no per-fold values were fabricated, nothing pooled unpaired.
2. **Paired t reported alongside the pre-registered Wilcoxon.** At n=4 the exact one-sided Wilcoxon's
   minimum attainable p is 1/2⁴ = **0.0625 > α** — a power ceiling (the n=4 analogue of §2's n=5
   note): all four cat cells sit exactly at the floor with 4/4 positive. The paired t (df=3) is the
   powered seed-level test; with effects of 67–320 σ_d the verdicts do not hinge on distributional
   fine print. The per-fold n=20 Wilcoxon (the pre-registered test proper) runs when the sidecars
   are pulled.
3. **Holm applied to the paired-t family** (m=4: AL/AZ/FL/Istanbul cat superiority) — on the Wilcoxon
   footing the whole family is floor-locked (best possible Holm-adj = 4 × 0.0625 = 0.25), so the
   correction is reported on the t p-values; all four reject at FWER 0.05 (smallest margin: adj
   p = 5.3e-07).

## 2 · LIMITS (honest gaps — read before citing)

1. **CA/TX are absent entirely** — v17 MTL is seed-0 (n=5) there; the n=20 top-up is **A1** on the
   A40 (`../A40.md`). The 6-dataset family Holm (protocol §5.2) re-runs after A1. Until then every
   verdict here carries the M1-PARTIAL banner.
2. **All cells are seed-level (n=4), not per-fold (n=20).** Still missing from the committed tree
   (A40 gitignored rundirs):
   - AL/AZ/FL: the per-PID `a40_score_matched.py` sidecar JSONs (tags `n20ph_{state}_{recipe}_s{seed}`,
     `cat_per_fold`/`reg_per_fold` arrays) inside
     `results/check2hgi_dk_ovl/{state}/mtlnet_*bs8192_ep50_*_{pid}/`;
   - Istanbul: the `h3ist_mtl_s{S}` sidecars + the cat-ceiling `stl_cat_ceiling_score.json` per rundir.
   Pulling those upgrades every cell to the pre-registered per-fold n=20 Wilcoxon/TOST (which breaks
   the Wilcoxon floor, §2). ~~Prior gap: AL/AZ/FL per-seed values not committed~~ — **closed by A0
   (`c51b1183`,** `n20_perhead_runs/summary.tsv`**)**.
3. **The STL sides are fully committed at n=20 per-fold** (cat: sweep JSONs at AL/AZ/FL; reg: P1
   JSONs at all four) — but they are consumed here as per-seed means to match the MTL side's
   granularity (pairing must be at a common footing).
4. **Wilcoxon at n=4 is floor-limited** (min p = 0.0625) — reported per cell, with the paired t as
   the powered sensitivity test (deviation log #2). Resolved automatically once per-fold vectors land.
5. **AZ reg**: the 90 % CI lower bound is +0.001 — non-inferiority is comfortable, but any "beats"
   phrasing at AZ is unsupported (supplementary superiority p = 0.049, at the boundary). The board's
   "≈matches" wording is exactly right.

## 3 · Bottom line (M1-partial)

- **At all 4 fully-n=20 datasets, the v17 champion beats the dedicated category ceiling (Δcat
  +5.3…+9.4 pp; seed-level paired t, every cell Holm-corrected p < 1e-6 at FWER 0.05) and is TOST
  non-inferior on region at δ_reg = 2 pp (every 90 % CI within ±2; AL the only negative point
  estimate at −0.31, bounded above −0.5).** FL and Istanbul additionally beat the region ceiling
  descriptively (CIs entirely positive); AZ grazes zero — keep it "matches."
- **The pre-registered per-fold n=20 tests remain the citation-grade target** — pending the A40
  sidecar pull (LIMITS #2) and A1 for CA/TX; then the full 6-dataset Holm family replaces this
  partial.
