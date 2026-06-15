# mtl_frontier — FINDINGS

Mechanism narratives per R-lever. Outcomes-table + gate decisions live in `STATE.md`; the
cross-study one-liner lives in `docs/studies/log.md`.

---

## R1 — log_C co-location prior + ESMM probability-chain coupling — **NULL** (2026-06-15)

**Verdict.** The ESMM-style co-location prior `prior(reg)=Σ_c P(reg|c)·P̂(c)`, distilled into the
reg head as a **second KD term on top of log_T-KD**, gives a **real but sub-promote-threshold** reg
lift and **does not clear the ≥0.3 pp multi-seed gate**. Champion G + log_T-KD is unchanged. Not a
v17 candidate; not a `closing_data` gate. **Proceed to R2.**

**Numbers (matched bar = top10_acc_indist·(1−ood) @ indist-best epoch; comparand = G WITH log_T-KD 0.2):**

| state | seed-0 Δreg | multi-seed {0,1,7,100} Δreg | Δcat | Wilcoxon (n=20 fold-seed) | gate ≥0.3 |
|---|---|---|---|---|---|
| AL (small) | +0.331 | **+0.207 ± 0.196** | +0.20 | p=0.0079, 15/20 pairs + | **FAIL** |
| FL (large) | +0.171 | (seed-0 null — not multi-seeded) | −0.27 | — | **FAIL** |

Per-seed AL Δreg: `[+0.33, +0.40, −0.11, +0.21]`. Weight sweep (AL seed 0): Δreg is **non-monotonic in
W_C, peaking at W=0.2** — W=0.4 → +0.118, W=0.6 → **−0.999** (over-weighting the prior craters reg).
So no weight clears the bar: W=0.2 is the optimum and it lands at +0.207 multi-seed.

**Why it's a (small-but-real) positive, not a hard zero.** The direction is reliably positive at AL
(Wilcoxon p=0.008, 15/20 fold-seed pairs, no cat regression — Δcat +0.20). So log_C is **not fully
redundant** with the Markov-1 log_T prior — the cat-marginalized co-location carries a *little*
independent signal. The R1 falsifier ("Δreg ≤ log_T-KD alone") is therefore not literally met; the
lever just doesn't clear the *promote* bar.

**Why the increment is small (mechanism).** Three compounding reasons, all consistent with the
established regime:
1. **Heavy overlap with log_T.** Categories cluster spatially, so `P(region|cat)` and
   `P(region|last-region)` are strongly correlated — the co-location prior largely re-expresses
   structure log_T already injects. The *incremental* information over log_T-KD is what R1 measures,
   and it is small by construction.
2. **Weak-auxiliary regime.** The conditioning variable is the **7-class (~2.8-bit)** category
   posterior — far below the 180–300-class vocabularies behind every positive category-aux result in
   the next-POI literature (`FINAL_SYNTHESIS.md §4`). A 7-way `P̂(c)` is a low-resolution selector over
   the region simplex, so the mixture `Σ_c P(reg|c)·P̂(c)` is a blunt prior.
3. **Scale-conditional, like log_T-KD itself.** The lift concentrates at the small state (AL +0.21)
   and washes out at scale (FL +0.17, and a mild cat trade −0.27) — the same small-state-binding
   signature log_T-KD shows (strong AL/AZ, weak FL/CA/TX). At FL the reg head is already well-fit
   from dense data + log_T, leaving no room for the blunt co-location prior.

**Construction / leak hygiene (for reuse).** `scripts/compute_region_colocation.py` mirrors
`compute_region_transition.py`: train-userids-only, per-fold, per-seed `region_colocation_log_seed{S}_fold{N}.pt`
(column-normalized `log P(region|cat)`, `[n_regions, n_cats]`), with the same StratifiedGroupKFold
split (verified engine-invariant) and the same seed/fold/n_splits + stale-mtime guards in the trainer.
The KD branch (`mtl_cv.py`) uses the **detached** cat posterior as the teacher factor and a one-shot
C28 mechanism-fires assertion (teacher confirmed ~37× peaked over uniform — the prior is non-trivial;
the null is a real measured effect, not a dead codepath). Code shipped behind `--log-c-kd-weight`
(default 0.0 = off); champion G defaults unchanged.

**Artifacts.** `docs/results/mtl_frontier/{r1_screen_results.json, r1_al_multiseed_results.json,
r1_weight_sweep_al.json}`; drivers `scripts/mtl_frontier/{r1_screen.sh, r1_al_multiseed.sh}`;
aggregators `{r1_agg.py, r1_multiseed_agg.py}`.
