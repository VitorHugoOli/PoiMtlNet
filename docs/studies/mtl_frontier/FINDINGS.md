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

---

## R2 — STEM-AFTB per-layer/direction gating sweep — **NULL** (scale-conditional cat lift; not v17) (2026-06-15)

**Verdict.** Parameterizing champion G's 2-block cross-attn sharing as per-layer, per-direction
All-Forward/Task-specific-Backward stop-grad gates (STEM, AAAI'24): **reg is a clean multi-seed null
everywhere**, and the cat axis shows a **real but AL-only, scale-decaying lift that does not
generalize** — so it is **not a v17 lever**. Champion G unchanged. Mechanism is citable (a STEM-AFTB
dose-response in the cos≈0 regime). **Proceed to R3.**

**Directional gates (new code, `_CrossAttnBlock`).** `detach_ab` = cat reads reg forward-only (detach
reg K/V in `cross_ab`) → reg pathway gets no gradient from L_cat; `detach_ba` = reg reads cat
forward-only → cat pathway gets no gradient from L_reg. Per-block spec via `--model-param aftb_spec=`
(comma per block, '+'-join of {ab,ba}/none). Unit-verified: G propagates the cross-task gradient
(tiny, ~1e-6 — the cos≈0 signature); each detach removes it cleanly (autograd `None`). Champion G =
`aftb_spec=None`.

**Configs (2 blocks)** — base(G) · aftb_all(ab+ba,ab+ba) · aftb_late(none,ab+ba) ·
aftb_early(ab+ba,none) · reg_protect(ab,ab) · cat_protect(ba,ba). Comparand = pure champion G (v16,
KD-off).

**AL seed-0 screen → all 5 cross the gate** (cat +0.67…+1.31; reg best reg_protect +0.42). **AL
multi-seed {0,1,7,100}** collapses the reg signal (best aftb_late +0.173, p=0.009 — significant but
sub-threshold; reg_protect +0.42→+0.05, i.e. seed-0 noise) and leaves a **high-variance AL cat lift**
(aftb_late +0.636±0.45, cat_protect +0.372, aftb_all +0.348, aftb_early +0.316 — 4 configs cross +0.3
on the 4-seed mean, 3–4/4 seeds positive each).

**Multi-state confirm (`aftb_late`, the best config) — the cat lift DECAYS with scale:**

| state | scale (regions) | Δreg | Δcat | ≥0.3 |
|---|---|---|---|---|
| AL | ~1.1k | +0.173 | **+0.636** | ✓ |
| AZ | ~1.5k | −0.087 | +0.173 | — |
| GE | ~3–4k | +0.060 | +0.158 | — |
| FL | ~4.7k | +0.019 | **−0.026** | — |

**Only AL clears 0.3; the cat effect fades monotonically to zero at FL.** Reg is null at every state.

**Mechanism (citable).** Any AFTB stop-grad — in either direction, at either layer — recovers
small-state cat, and the magnitude **decays with state scale**: at small data the (tiny, cos≈0)
cross-task gradient is mildly *harmful noise* to the cat head, and removing it helps; at scale the cat
head is well-fit and the gradient is immaterial. This is the **data-starved-regime** prediction
(Bingel & Søgaard'17) measured as a clean STEM-AFTB dose-response, and it independently corroborates
the study's regime findings: (a) the reg gap is **not** a sharing-topology problem (no AFTB config
moves reg — the dual-tower already isolates reg + cos≈0 leaves nothing to gate); (b) asymmetric
read-only sharing helps only where the main task is data-starved. It is the **inverse of the G′
FL-only dead-end** — an AL-only effect that vanishes at the paper-headline state — so, like G′, it is
**not a multi-state champion lever**. Citable against STEM (AAAI'24): the AFTB pattern that helps
industrial RecSys gives, in this LBSN regime, a scale-decaying small-state-only cat effect and no reg
benefit.

**Artifacts.** `docs/results/mtl_frontier/{r2_aftb_results.json, r2_al_multiseed_results.json,
r2_multistate_results.json}`; drivers `scripts/mtl_frontier/{r2_aftb_screen.sh, r2_al_multiseed.sh,
r2_multistate_confirm.sh}`; aggregators `{r2_agg.py, r2_multiseed_agg.py, r2_multistate_agg.py}`. Code:
directional `detach_ab`/`detach_ba` + `aftb_spec` in `src/models/mtl/mtlnet_crossattn/model.py`
(champion G defaults unchanged).

---

## R3 — CrossDistil (warm-up + error-correction + reverse reg→cat) — **NULL** (2026-06-15)

**Verdict.** CrossDistil (AAAI'22) — the live-teacher generalization of log_T-KD — does **not** beat
G + log_T-KD. The error-correction + warm-up refinements **do not rescue** R1's forward cat→reg null,
the genuinely-new **reverse reg→cat** arm's AL gain is **seed-0 noise** (washes out multi-seed), and
FL is null throughout. Champion G + log_T-KD unchanged. **Proceed to R10.**

**Configs (vs G + log_T-KD), warmup=15 ep, ec λ=0.3.** Forward arm = R1's co-location KD + CrossDistil
error-correction `teacher*=(1−λ)·teacher+λ·onehot(y)` + warm-up. Reverse arm = distill the reg-implied
category prior `Σ_r P(cat|r)·P̂_reg(r)` (detached reg posterior) into the cat head.

**AL+FL seed-0 screen:**

| config | AL Δreg | AL Δcat | FL Δreg | FL Δcat |
|---|---|---|---|---|
| r3_fwd_ec (CrossDistil-refined forward) | +0.063 | **−0.182** | +0.108 | −0.195 |
| r3_rev (reverse reg→cat) | +0.126 | **+0.452** | −0.008 | +0.087 |
| r3_both | +0.118 | +0.263 | +0.133 | −0.143 |

- **The forward CrossDistil refinements do NOT rescue R1** — r3_fwd_ec is null on reg (+0.06) and
  *hurts* cat (−0.18). Error-correction + warm-up do not turn the sub-threshold R1 co-location prior
  into a lever. R1's forward verdict stands at a higher standard.
- **r3_rev** was the only seed-0 promote (AL cat +0.452) — the same AL-only/FL-null shape as R2.

**r3_rev AL multi-seed {0,1,7,100}: the seed-0 cat lift is NOISE.** Δcat **+0.100 ± 0.282** (Wilcoxon
n=4 p=0.31; per-seed [+0.452, **−0.335**, +0.168, +0.114] — one seed negative); Δreg +0.081 ± 0.098
(p=0.025, sub-threshold). **Gate FAILS.** Unlike R2's AFTB cat lift (which held multi-seed at AL before
failing the *multi-state* test), r3_rev's seed-0 AL effect does not even survive AL multi-seeding.

**Mechanism / takeaway.** CrossDistil is the named, refined version of this repo's one proven lever
(log_T-KD = a static-teacher instance). Making the teacher *live* (cat posterior), *calibrated*
(τ), *error-corrected* (GT blend), *warm-up-gated*, and *bidirectional* adds **nothing** over the
static log_T teacher already in G. This is the strongest form of the "output-level prior family is
saturated by log_T-KD" conclusion: the cat↔reg label-space coupling carries no further multi-seed,
multi-state signal. Together with R1 (static co-location) and R2 (sharing topology), **all three
first-wave levers are nulls** that reproduce the same regime: small-state-only, high-variance,
FL-null, reg-immovable.

**Artifacts.** `docs/results/mtl_frontier/{r3_screen_results.json, r3_rev_al_multiseed_results.json}`;
drivers `scripts/mtl_frontier/r3_screen.sh` (+ `/tmp/r3_alms/run.sh`); aggregators
`{r3_agg.py, r3_rev_multiseed_agg.py}`. Code: `log_C_rev` buffer + warm-up/ec/reverse arms behind
`--log-c-kd-warmup-epochs`/`--log-c-kd-ec-lambda`/`--cat-kd-weight` (champion G defaults unchanged).
