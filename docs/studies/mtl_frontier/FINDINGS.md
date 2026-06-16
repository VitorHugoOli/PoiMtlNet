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

---

## R10 — GRM-gated cross-attn read (Memory-Caching primitive, "on the layers") — **NULL** (2026-06-15)

**Verdict.** The input-dependent **Gated Residual Memory** read (arXiv:2602.24281 primitive) — a
learned, continuous generalization of R2's binary AFTB masks — **does not beat champion G**. The seed-0
FL cat lift (+0.324) is **seed noise** (multi-seed +0.085 ± 0.203, two seeds negative). This is the
**exact falsifier the R10 spec named: GRM ≡ G within noise** (the learned gate reproduces the
hand-built dual-tower asymmetry — a citable null). Champion G unchanged.

**Mechanism (new code, `_CrossAttnBlock`).** Per-sample, per-dim gate `γ=σ(W·masked-mean_seq(query))`
on each stream's cross-attn read: `a ← LN(a + γ_a⊙CrossAttn(a,b))`. R2 gated the *gradient* (binary
detach); GRM gates the *forward read magnitude*, input-conditioned, gradient flowing through γ. Bias
init +2 → γ≈0.88 (untrained ≈ G). Unit-verified input-dependent (γ∈[0.815,0.924] across samples,
C28 non-trivial). Enabled via `--model-param crossattn_grm=True`; G default unchanged.

**Screen (G+GRM vs champion G, seed0):** AL Δreg +0.141 / Δcat +0.178 (null); **FL Δreg −0.026 / Δcat
+0.324** (crossed the gate — the FIRST FL effect in the study, where FL cat is tight, R0 σ=0.04).

**FL multi-seed {0,1,7,100}: WASHED OUT.** Δcat **+0.085 ± 0.203** (Wilcoxon n=4 p=0.31; per-seed
[+0.324, −0.053, +0.238, −0.170] — two seeds negative); Δreg −0.027 ± 0.028 (p=0.82). **Gate FAILS.**

**Takeaway.** Making R2's sharing gate **learned and input-conditioned** adds nothing over champion G's
fixed asymmetry — GRM ≡ the hand-built dual-tower. The cos≈0 regime leaves the cross-task read with no
input-conditioned structure worth gating: where the binary mask was a (small-state) effect, the learned
gate is a wash at the matched standard. The Memory-Caching headline (long-context recall) was never
applicable (fixed length-9 windows); its transplantable gated-read primitive, applied "on the layers",
is null here. **SSC (top-k router) was NOT pursued** — the GRM primary showed no signal, so the
falsifier "GRM ≡ G" already closes the R10 family per the spec (run R2 first → GRM is "what if AFTB were
learned" → null like AFTB).

**Trained-γ confirmation (2026-06-15, post-audit — the null is NOT a frozen-gate artifact).** An
independent audit asked whether "GRM≡G" could be a dead mechanism (γ trapped near 1). A diagnostic run
(`--model-param crossattn_grm=True`, per-epoch mean γ logged; AL seed0) shows the **gate is the opposite
of trapped — it trains a LARGE, consistent, input-conditioned swing and still lands on G's accuracy:**

| | init | ep10 | ep25 | final |
|---|---|---|---|---|
| γ_a (cat reads reg) | 0.874 | 0.845 | ~0.49 | **~0.31** |
| γ_b (reg reads cat) | 0.874 | 0.854 | ~0.62 | **~0.55** |

(means over 2 blocks, 3 folds shown; metrics AL cat 52.56 / reg 64.49 ≈ champion G within noise.) The
learned gate **discovered it should roughly halve the cat→reg read** (γ_a 0.87→0.31) and modestly cut
reg→cat — a decisive, trained modulation — yet accuracy is identical to the fixed full-read G. So the
cross-attn read can be gated however the data dictates and the metrics do not move: there is no
exploitable input-conditioned cross-task signal (cos≈0). This rules out the frozen-gate artifact and
makes "GRM ≡ G" a genuine regime result. Diagnostic code: `grm_gamma_a/grm_gamma_b` per-epoch
(`mtl_cv.py`, `_CrossAttnBlock.last_gamma_{a,b}`).

**Efficiency (champion G strictly dominates G+GRM).** G+GRM **matches** G's accuracy (the null) but
costs **+263,168 params (+5.6%; 4.66M→4.92M, all in the shared cross-attn)** and is **not faster**
(matched-seed FL wall-clock 1.02×/1.05×/0.98× = equal within GPU-sharing noise). Same accuracy, more
parameters, equal speed → no efficiency case for R10; champion G dominates.

**Artifacts.** `docs/results/mtl_frontier/{r10_screen_results.json, r10_fl_multiseed_results.json}`;
drivers `/tmp/r10_screen/run.sh`, `/tmp/r10_flms/run.sh`; aggregators `{r10_agg.py,
r10_fl_multiseed_agg.py}`. Code: `crossattn_grm` + `_masked_mean_seq` in
`src/models/mtl/mtlnet_crossattn/model.py` (champion G defaults unchanged).

---

## SYNTHESIS — first wave + R10: four nulls, one regime (2026-06-15)

| lever | family | mechanism | best seed-0 | multi-seed verdict |
|---|---|---|---|---|
| **R1** | output-level prior | log_C co-location KD `Σ_c P(reg\|c)·P̂(c)` | AL reg +0.33 | **null** (+0.207, sub-threshold) |
| **R2** | asymmetric sharing | binary STEM-AFTB stop-grad masks | AL cat +1.31 | **null** (AL-only; decays to FL −0.03) |
| **R3** | output-level coupling | CrossDistil (live teacher, ec, warm-up, reverse) | AL cat +0.45 | **null** (washes out; log_T-KD saturates the family) |
| **R10** | asymmetric sharing | learned input-conditioned GRM gate | FL cat +0.32 | **null** (washes out; GRM ≡ G) |

### AUDIT — "seed-0 is always best" (user-flagged) → determinism + real seed variance, results stand (2026-06-16)

The user noticed every lever's delta was largest at seed-0. Audited:
- **Champion G is fully DETERMINISTIC** on the current code: two distinct runs (different rundirs/PIDs)
  at the same seed are bit-identical (Δcat = 0.0000 at all 4 FL seeds; `gmatched_manifest.tsv`,
  `cc_rematched_results.json`). No run-to-run non-determinism.
- **The earlier "run-to-run noise (AL 0.38 / FL 0.11)" was CODE DRIFT**, not noise — it compared
  current-code (mtl_frontier, June 15) baselines to the OLD mtl_improvement R0 numbers (June 6, on a
  different code state). Champion-G cat shifted ~0.1–0.4 between the two code versions. **Within
  mtl_frontier every comparison uses the same code → valid; do NOT mix mtl_frontier absolute numbers
  with the older R0 bar.**
- **Reused baselines were CORRECT** (deterministic, matched per seed). No "low-draw" inflation.
- **The seed-0 pattern is REAL seed variance:** champion-G FL cat per seed is deterministically
  `[73.012, 73.212, 73.181, 73.143]` — seed-0 is genuinely the weakest seed (same at AL). Levers lift
  the weak seed-0 baseline most → "best at seed-0." Not a bug; the multi-seed gate averages it out.
- **Implication:** the nulls are unchanged (deterministic, matched); the conditional-coupling positive
  is CONFIRMED real (matched re-eval = exactly +0.235 FL cat, noise floor 0.000), not a baseline artifact.

### Conditional coupling (cat output → reg input feature, iMTL/GETNext) — the FIRST genuine positive (sub-threshold) (2026-06-16)

> **AUDIT-CONFIRMED (2026-06-16):** the matched, replicated, deterministic re-eval reproduces this
> exactly (FL Δcat +0.235, 4/4 seeds positive [+0.45,+0.07,+0.21,+0.21], G noise floor 0.000; +0.16
> even excluding the weak seed-0). It is a genuine reproducible effect, not a seed-0/baseline artifact.
> The richer 256-dim `features` variant HURTS FL (−0.31/−0.36) → the sub-threshold cap is the regime
> (the weak 7-class semantic prior is the useful signal), not a fixable weak-conditioning HP.

**Verdict.** The advisor's recommended out-of-the-cos≈0-box direction. Feed the cat head's posterior
`softmax(cat_logits)` [B,7] as an **input feature** into the reg head (fused additively before the
classifier, train+inference; zero-init → untrained ≡ G; `cond_coupling=posterior`). Unlike the 7
prior nulls, this is a **real, consistent, multi-seed Pareto positive at FL** — the **first non-null
direction** — but it lands **just below the 0.3 promote gate**.

**Seed-0 screen (both variants, vs champion G):** AL cat +0.962 (e2e) / +0.383 (detach); **FL cat
+0.450 (e2e) / +0.413 (detach)**; reg flat-to-slightly-neg. The coupling fires hard (cond_norm FL
0.13→4.57 — the reg head leans on the cat prediction). First lever with a solid positive FL cat across
2 variants × 2 states.

**cc_e2e multi-seed {0,1,7,100}:**

| state | Δreg | Δcat | per-seed Δcat |
|---|---|---|---|
| AL | +0.089 ± 0.15 | −0.056 ± 0.71 (washed out) | [+0.96, −0.31, −0.99, +0.12] |
| **FL** | **+0.070** (Wilcoxon n=20 p=0.035) | **+0.235 ± 0.14** (n=4 p=0.0625) | **[+0.45, +0.07, +0.21, +0.21] — 4/4 positive** |

AL washed out (the seed-0 +0.96 was a small-state flare). **FL is genuine:** cat +0.235 with **every
seed positive** (the only lever in the study with no negative seed) AND reg significantly positive
(+0.070, p=0.035) — a real both-heads Pareto gain. **It fails the ≥0.3 gate (cat +0.235 < 0.3), so no
v17 promotion** — but it is qualitatively different from the nulls.

**Why this is the right mechanism, and why it's capped.** Conditional coupling changes the reg head's
**input distribution** (it gets the predicted category as a feature) rather than re-gating the cos≈0
cross-task gradient — exactly the one channel FINAL_SYNTHESIS §4 + the literature (iMTL/GETNext) name
as able to beat parity here. That it produces the study's only genuine multi-seed positive **validates
both** (a) the direction (input-side conditioning works where output-side/sharing levers are null), and
(b) the **weak-auxiliary cap**: the conditioning signal is the **7-class (~2.8-bit) posterior**, the
weakest possible — so the gain is real but small (+0.235), below the bar. This is the cleanest
in-study confirmation of the "weak-7-class-auxiliary regime" framing. **Open test (the obvious next
step):** richer conditioning — feed the cat head's **penultimate features / a category embedding**
(GETNext's actual form, 256-dim not 7-dim) — does the FL gain clear 0.3, or is the cap a hard wall?
This also directly answers the "is the sub-threshold a weak-signal (fixable) limitation vs the regime"
question. Artifacts: `cc_screen_results.json`, `cc_e2e_multiseed_results.json`; code
`cond_coupling`/`cond_detach`/`cond_proj` (champion G unchanged).

---

### Follow-up screens (user ideas, advisor-structured) — three more nulls (2026-06-15)

After the 4-null first wave, three user-proposed follow-ups were screened (advisor evaluation in
the session record). All null; one is informative.

**Idea 1 — R10 GRM at the other states (close the "FL-only multi-seed" gap).** GRM screened at AZ/GE
seed0 + AL multi-seed {0,1,7,100} vs champion G (baselines reused). **Null everywhere:** AZ Δreg
−0.239 / Δcat +0.140; GE +0.083 / +0.269; AL 4-seed +0.141 / +0.178. Confirms the trained-γ
prediction — GRM's "nothing to gate" is state-agnostic; R10 is null at all 4 built states.

**Idea 2 — `aux_gated` fusion: input-dependent β in the reg head (the best-shot idea).** New fusion
mode `feat = priv + γ(·)·aux_proj(shared)`, γ=σ(MLP([priv;shared])) per-dim (init γ≈0.12≈champion β).
The gate is **fully alive and moves**: γ trains 0.12→**0.30 (AL)** / 0.12→**0.47 (FL)** — it learns to
*open* the shared pathway. Result: AL cat **+0.48** (seed-0 flare) but **FL cat −0.85 (crater)**, reg
flat. **Not a lever — actively harmful at scale, and it re-confirms the champion's design:** champion
`aux` drives β→**0** (closes the shared→reg pathway, X3); an input-conditioned gate *opens* it, which
dilutes the large-state cat exactly as the falsified `gated` mode did (gated 73.06 < aux 73.57). The
additive-input-dependent point between `aux`(scalar) and `gated`(convex) inherits `gated`'s FL
weakness. Input-conditioning does not rescue the shared-pathway fusion. (Code: `fusion_mode=aux_gated`
+ `aux_gamma` diagnostic; champion `aux` unchanged.)

**Idea 3 — stack the best-direction variants** (G + log_T-KD + log_C-KD + reverse cat-KD + aftb_late +
GRM) vs G+log_T-KD, AL+FL seed0, pre-registered to promote only on **super-additivity** (>0.3 over the
best single component). **Sub-additive null:** AL cat +0.456 < the best single component (aftb_late
+0.636); FL +0.091 reg / +0.109 cat. The levers are redundant views of one mechanism and **interfere**
(as `r3_both` already showed) — stacking is worse than the best component, not better. Fails the
pre-registered gate.

**Net:** the three follow-ups add a 5th/6th/7th null and one positively-useful datum — Idea 2 shows
the champion's β→0 / `aux`-over-`gated` choice is *robust to input-conditioning*, strengthening the
architecture claim. The recommended next direction (out of the cos≈0 box that all of R1/R2/R3/R10 +
these three have now nulled) is **conditional coupling**: feed the cat head's *output* as an *input
feature* to the reg head (GETNext/iMTL pattern) — the one mechanism with a literature prior for
beating parity in this regime (FINAL_SYNTHESIS §4; advisor recommendation). Artifacts:
`docs/results/mtl_frontier/followup_results.json`; driver `scripts/mtl_frontier/followup_screens.sh`;
agg `followup_agg.py`.

---

**One conclusion, four independent confirmations.** Every output-level coupling (R1 static prior, R3
live distillation) and every sharing-topology lever (R2 binary gate, R10 learned gate) is a **null over
champion G**. The recurring pattern — a promising single-seed effect (always at the state with the
loosest cat variance) that **washes out under multi-seeding** — is the signature of the **cos≈0,
data-rich, weak-7-class-auxiliary regime** documented in `archive/mtl_improvement/FINAL_SYNTHESIS.md`.
The study's two proven wins (the dual-tower architecture + the static log_T-KD prior) are **not
extended** by their literature-frontier generalizations: CrossDistil = G's already-present static
teacher; STEM-AFTB / GRM = G's already-present hand-built asymmetry. **No lever promotes to v17; champion
G stands.** This is a strong, citable negative for the paper: the post-2022 MTL frontier (asymmetric
modularity + output-level coupling), brought to this LBSN regime, reproduces — but does not beat — the
two mechanisms the `mtl_improvement` study already found. Remaining program (R4-R9) deferred; the
optimizer aisle stays closed (Kurin/Xin/Mueller).
