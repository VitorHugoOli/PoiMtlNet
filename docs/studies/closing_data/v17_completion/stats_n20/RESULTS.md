# M1-PARTIAL — v17 pre-registered stats vs the n=20 best-vs-best ceilings (2026-07-08)

> ⚠ **This is M1-PARTIAL.** It covers the 4 datasets that are fully n=20 on both sides today
> (**AL, AZ, FL, Istanbul** — ceilings per [`../CEILINGS_N20_FINAL.md`](../CEILINGS_N20_FINAL.md),
> **AZ ceiling = 56.43, the corrected bs8192@0.005 arm**). **CA/TX are pending A1** (v17 MTL n=20
> on the A40). The paper's headline family is 6 datasets (protocol §5.2) — **the full 6-dataset
> family Holm re-runs after A1**; everything here is the pre-registered analysis of the committed
> subset, not the final family verdict.
>
> Pre-registration: [`../../STATISTICAL_PROTOCOL.md`](../../STATISTICAL_PROTOCOL.md) (§2 cat
> superiority → paired one-sided Wilcoxon; §3 reg → TOST non-inferiority at **δ_reg = 2 pp**;
> §4 pairing discipline; §5.2 Holm on the cat family only; §8 deviation log). Test conventions
> mirror `scripts/closing_data/superiority_wilcoxon.py` + `region_match_tost.py`.
>
> Reproduce: `.venv/bin/python docs/studies/closing_data/v17_completion/stats_n20/m1_stats_n20.py`
> (reads ONLY committed artifacts; aborts if any recomputed aggregate stops matching the board).

## 0 · Artifact → board reproduction gate (all PASS)

Every board number used below was recomputed from the committed per-fold/per-seed artifacts before
testing: STL cat ceilings AL 56.815 / AZ 56.435 / FL 74.509 (from `../cat_ceiling_sweep/sweep_results/`
per-fold JSONs, 4 seeds × 5 folds); STL reg ceilings AL 70.111 / AZ 59.459 / FL 76.697 / Istanbul 75.158
(from `docs/results/P1/region_head_*_ovl_stl_reg_{s0,topup_s{1,7,100}}.json` per-fold arrays); Istanbul
MTL 63.329 / 75.440 and cat ceiling 54.738 (from `../h3_istanbul/step3_runs/*.txt` per-seed means).
All match `CEILINGS_N20_FINAL.md` to rounding.

## 1 · Per-dataset results

Δ orientation = **MTL − STL ceiling** (positive = MTL better). δ_reg = 2 pp (pre-registered, §3.2).

| Dataset | Cell | Δ (pp) | Pairing level used | Test | statistic / p | Holm-adj p (cat family) | Verdict |
|---|---|---:|---|---|---|---|---|
| **AL** | cat | +7.73 | — (MTL side not committed) | pre-registered paired Wilcoxon **PENDING** | — | — | **PENDING (artifact gap)** |
| **AL** | reg | −0.31 | — | pre-registered paired TOST **PENDING** | — | n/a (TOST not in Holm family) | **PENDING (artifact gap)** |
| **AZ** | cat | +9.40 | — | paired Wilcoxon **PENDING** | — | — | **PENDING (artifact gap)** |
| **AZ** | reg | +0.10 | — | paired TOST **PENDING** | — | n/a | **PENDING (artifact gap)** |
| **FL** | cat | +5.34 | — | paired Wilcoxon **PENDING** | — | — | **PENDING (artifact gap)** |
| **FL** | reg | +0.72 | — | paired TOST **PENDING** | — | n/a | **PENDING (artifact gap)** |
| **Istanbul** | cat | **+8.59 ± 0.09** | **seed-level paired, n=4** (per-seed 5-fold means) | one-sided Wilcoxon (exact) + paired t (sensitivity) | Wilcoxon p = 0.0625 (**= the n=4 floor**, 4/4 positive); paired t(3) p = **1.8e-07** (Δ ≈ 92 σ_d) | ≤ 7.1e-07 (Bonferroni m=4 bound; < 0.05) | **outperforms** (seed-level) |
| **Istanbul** | reg | **+0.28 ± 0.04** | **seed-level paired, n=4** | TOST non-inferiority, δ = 2 pp | TOST p = **1.2e-06**; 90 % CI = **(+0.240, +0.323)** pp ⊂ (−2, +2) | n/a (equivalence cell, §5.2) | **matches (TOST)** — CI also entirely > 0, so descriptively beats |

**Istanbul per-seed Δs (seeds 0/1/7/100):** cat [+8.612, +8.469, +8.588, +8.696]; reg [+0.250,
+0.308, +0.251, +0.316].

### Pairing level actually used, per cell (spell-out)

- **Istanbul (both cells): SEED-level paired, n=4** — the committed MTL side is the per-seed 5-fold
  means (`../h3_istanbul/step3_runs/mtl_{cat,reg}_s{0,1,7,100}.txt`; identical to the per-seed arrays
  in `../h3_istanbul/RESULTS.md`). Both arms share the frozen fold construction and seed set (§4), so
  seed-level pairing is valid; per-fold pairing is not possible from the committed tree (see LIMITS).
  **Stated per the job's pairing discipline: this is the n=4 seed-level test, not the n=20 per-fold
  test.** No per-fold values were fabricated; nothing was pooled unpaired.
- **AL/AZ/FL (all cells): NO test run** — the v17 MTL side has **no committed per-seed or per-fold
  values at all** (see LIMITS). Only the n=20 aggregate mean ± cross-seed σ exists in
  [`../../perhead_lr_n20.md`](../../perhead_lr_n20.md). Per the M1 discipline ("document exactly what
  is missing rather than approximating"), these cells are **PENDING**, with descriptive Δs only.

### Deviation log (protocol §8)

1. **Istanbul cat: paired t reported alongside the pre-registered Wilcoxon.** At n=4 the exact
   one-sided Wilcoxon's minimum attainable p is 1/2⁴ = **0.0625 > α** — it cannot clear 0.05 at this
   footing *regardless of effect size* (a power ceiling, the n=4 analogue of §2's n=5 note). The
   paired t (df=3) is the powered seed-level test; with Δ ≈ 92 σ_d its p = 1.8e-07 and the verdict
   does not hinge on distributional fine print. The final per-fold n=20 Wilcoxon (the pre-registered
   test proper) runs when the per-fold artifacts are pulled (LIMITS #2).
2. **Holm on an incomplete family.** Only 1 of 4 cat cells is testable today, so the Holm adjustment
   cannot be computed as pre-registered. We report the **conservative Bonferroni bound** (m=4):
   Istanbul p_adj ≤ 4 × 1.8e-07 = 7.1e-07 < 0.05. Since Holm ≤ Bonferroni, Istanbul's "outperforms"
   survives **any** completion of the family; AL/AZ/FL enter the Holm computation when their tests run.

## 2 · What the descriptive Δs already say (no p attached)

For AL/AZ/FL the effect sizes vs the committed noise scales are large on cat and small-vs-margin on
reg — stated descriptively only (no verdict without the pre-registered test):

| State | Δcat (pp) | ceiling cross-seed σ_cat | MTL cross-seed σ_cat (`perhead_lr_n20.md`) | Δreg (pp) | vs δ_reg = 2 |
|---|---:|---:|---:|---:|---|
| AL | +7.73 | 0.026 | 0.098 | −0.31 | well inside margin |
| AZ | +9.40 | 0.096 | 0.019 | +0.10 | inside margin (positive) |
| FL | +5.34 | 0.033 | 0.028 | +0.72 | positive (candidate "beats") |

## 3 · LIMITS (honest gaps — read before citing)

1. **CA/TX are absent entirely** — v17 MTL is seed-0 (n=5) there; the n=20 top-up is **A1** on the
   A40 (`../A40.md`). The 6-dataset family Holm (protocol §5.2) re-runs after A1. Until then every
   verdict here carries the M1-PARTIAL banner.
2. **AL/AZ/FL v17 MTL per-seed×per-fold values are NOT in the committed tree.** The n=20 runs exist
   (2026-06-29/30, A40) but their artifacts are gitignored on the run machine
   (`docs/studies/train_perf_multifold/.gitignore` pattern `*_runs/`). Missing, specifically:
   - `docs/studies/train_perf_multifold/n20_perhead_runs/summary.tsv` (per-seed cat/reg 5-fold means,
     8+4+4 rows) — enough for the **seed-level n=4** tests;
   - the per-PID rundirs `results/check2hgi_dk_ovl/{alabama,arizona,florida}/mtlnet_*bs8192_ep50_*_{pid}/`
     with the `a40_score_matched.py` sidecar JSONs (tags `n20ph_{state}_{recipe}_s{seed}`,
     `cat_per_fold`/`reg_per_fold` arrays) — enough for the full **per-fold n=20** tests.
   Committing either from the A40 unblocks the pre-registered tests; only the aggregates in
   `perhead_lr_n20.md` were committed. **Nothing was approximated in their place.**
3. **Istanbul is seed-level (n=4), not per-fold (n=20).** The H3 driver
   (`../h3_istanbul/run_step3_n20.sh`) wrote the matched-score sidecars (tags `h3ist_mtl_s{S}`) and
   `stl_cat_ceiling_score.json` into the A40 rundirs — not committed. The committed per-seed means
   are sufficient for the seed-level tests above; the per-fold Wilcoxon/TOST upgrade needs those
   sidecars pulled. (The Istanbul **reg ceiling** side IS committed per-fold — the four P1 JSONs.)
4. **The STL sides are fully committed at n=20** (cat: sweep per-fold JSONs at AL/AZ/FL; reg: P1
   per-fold JSONs at all four datasets) — the artifact gap is exclusively on the MTL side (and the
   Istanbul cat-ceiling per-fold detail).
5. **Wilcoxon at n=4 is floor-limited** (min p = 0.0625) — reported, with the paired t as the powered
   sensitivity test (deviation log #1). Not an issue once the per-fold vectors land (n=20 breaks the
   ceiling, §2).

## 4 · Bottom line (M1-partial)

- **Istanbul (the one fully-committed pairing): the v17 champion beats the dedicated category ceiling
  (+8.59 pp, seed-level paired t p = 1.8e-07, Bonferroni-bounded p_adj ≤ 7.1e-07) and is TOST
  non-inferior on region at δ_reg = 2 pp (p = 1.2e-06, 90 % CI +0.24…+0.32 pp — entirely positive,
  i.e. it in fact beats).** The champion-G signature is now statistically backed on the non-US corpus.
- **AL/AZ/FL: descriptively unchanged (cat +5.3…+9.4, reg −0.31…+0.72) but the pre-registered tests
  await one small artifact pull from the A40** (LIMITS #2) — no test was run rather than run wrong.
- **CA/TX: await A1**; then the full 6-dataset Holm family replaces this partial.
