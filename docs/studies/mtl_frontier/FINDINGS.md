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

## ★ CAT↑/REG↓ — WHY, and is reg being shortchanged? **No — reg is at its STL ceiling, not lost** (2026-06-17 audit)

**The question (user, 2026-06-17):** across the levers cat seems to improve while reg degrades — "it seems
we are losing something." A 4-lens investigation (data · measurement-artifact · mechanism · adversarial
bug-hunt), each **adversarially verified** (all 4 verifications: *holds=True, refuted=False*), with every
load-bearing number reproduced from rundirs, gives a decisive answer: **nothing is being lost on reg; reg
is genuinely immovable because it is already at its single-task ceiling, and the cat↑/reg↓ impression is
selective perception + a diagnostic-best epoch-decoupling artifact, not a real systematic trade and not a
bug.** Four converging findings:

**1. The pattern is mostly selective perception (DATA).** Across the full 41-pair (cat_delta, reg_delta)
table (every results JSON, multi-seed + seed0, all states), the **largest quadrant is cat+/reg+ (19)**, not
cat+/reg− (only 7); the cat–reg delta correlation is **positive, not negative** (Pearson +0.026 all /
+0.39 multi-seed) — a genuine "reg funds cat" trade would require a strong *negative* correlation. **reg is
positive in 28/41** (mean +0.047, median +0.063; max +0.331 at R1 AL). The cat↑/reg↓ impression comes from
reading the two most-recent levers against the wrong frame: **R5** (cat+0.47/reg−0.10) only vs the
*handicapped global-W-KD-ON* base — vs the true KD-off champion it is Pareto-dominated on *both* heads; and
**R10** (cat+0.085/reg−0.027 multi-seed) is pure seed noise (2/4 cat seeds negative, reg Wilcoxon p=0.82).

**2. R5's reg-down is partly a measurement artifact (MEASUREMENT).** cat and reg peak **~26 epochs apart**
(cat-best ~ep18-20, reg-best ~ep44-46). The study's "diagnostic-best" metric reads each task at its *own*
best epoch, so it harvests reg at its late peak where R5's small reg cost is maximally visible. At the
**single deployable joint (geom_simple) checkpoint** — which sits ~+1.6 ep from cat-best, ~24 ep before
reg-best — R5 covmax's reg-down **collapses to a null** (diag −0.100 → joint −0.025, n=20 Wilcoxon p=0.73)
while the cat-up *survives* (+0.54). So at one deployable model R5 is cat-up/reg-**flat**, not reg-down.
(coventr's reg-down partly survives, −0.094, p=0.038.) The (1−ood) factor is verified **bit-exact and
non-differential** (cancels in matched deltas). By contrast the cc family's both-heads-up is **real and
grows** at the joint checkpoint (reg +0.070 diag → +0.166 joint, p=0.021) — genuine transfer.

**3. The mechanism — a β-gated regime split (MECHANISM).** The reg head's *only* shared-trunk channel is
`feat = priv + β·aux_proj(shared)` (head.py:403, β init 0.1). The **trained β is gated by state scale**:
at **FL, β converges to ≈0 in every arm** (champion-G all 4 seeds ≈ −0.00005; every R4 weight; every
lever) → there is **no shared reg contribution for cat gains to "rob,"** so the cat↑ cannot be coming at
reg's shared-trunk expense. At small states the channel is **live** (β = +0.078 GE, +0.088 AZ, +0.11…0.12
AL) and **levers move BOTH heads together** (R1 AL cat +0.20 / reg +0.21; R2 AL aftb cat +0.64 / reg +0.17).
cat is movable everywhere because it harvests the shared cross-attn trunk (the easy 7-class task); reg at FL
is driven by its **private STAN tower, which is saturated** (R4: as cw drops 0.85→0.40 the private-tower
drift *triples* 8.4→28.8 with reg accuracy *flat* 72.4→73.0 — moves hard for no payoff). cos(∇cat,∇reg)≈0
throughout. cc fits cleanly: it moves reg up via a **new input channel** (cond_norm 0.13→4.56) while β stays
≈0 — reg gains via the category-posterior *input*, never the shared trunk.

**4. No bug shortchanges reg, and reg is NOT under-trained (BUG-HUNT — the decisive check).** Adversarially
testing every reg-shortchange channel: **MTL reg MATCHES the same-substrate STL reg ceiling within −0.09
(AL) … −0.31 (FL) pp** (reproduced from `R0_matched_metric_bar.json`) — reg is **already at its single-task
ceiling**, so there is no hidden gap for a bug to hide behind and no recoverable reg gain to lose. The eval
path is the joint forward with **no cat-contamination for champion G** (cond_coupling='none' → the cc branch
is skipped, β≈0 insulates reg); the matched-bar `top10_acc_indist·(1−ood)` is **mathematically exact**
(native full top-K == indist·(1−ood) to float epsilon — OOD never lands in top-K), ood is non-differential
between arms, the KD/α prior is a **verified no-op at eval** (head_alpha=0), fp16 eval is immaterial and
*tie-optimistic toward MTL*, and the analyses use the **reg-favourable** diagnostic-best field (not the
~0.9pp-lower joint field). reg sends ~zero gradient to the shared trunk (grad-norm 1.7e-5) so the
w_reg=0.25 loss weight does not starve it.

**The answer, in one line.** **We are not losing anything on reg.** reg is immovable because it is *already
maxed* (at its STL ceiling) and, at the headline state, *architecturally insulated* (β→0, private tower
saturated, cos≈0) — so the levers can only move the easy shared-trunk-harvesting cat task, and where the
shared→reg channel is alive (small states) they move **both** heads together. The cat↑/reg↓ reading was an
artifact of one lever's wrong comparand (R5) + one lever's seed noise (R10) + the diagnostic-best epoch
decoupling. **Paper framing:** this is the cleanest possible statement of the regime — *MTL achieves
reg-parity-at-the-STL-ceiling with a free cat lift; it cannot exceed reg because reg has no headroom and no
transfer channel at scale* — exactly the cos≈0 / dual-tower / weak-2.8-bit-auxiliary regime, and precisely
the parity-not-gain outcome the post-2022 MTL literature predicts.

**Artifacts.** Audit workflow `mtl-frontier-audit` (18 agents: 10 per-experiment reviews + 4 cat/reg lenses
+ 4 adversarial verifications); evidence reproduced from `docs/results/mtl_frontier/*.json`,
`docs/results/mtl_improvement/R0_matched_metric_bar.json`, and the rundir diagnostics CSVs (head_beta,
head_alpha, grad_norm_*_shared, reg_encoder_drift). See `## AUDIT (2026-06-17)` below for per-experiment
verdicts + the two follow-up bugs it surfaced.

---

## AUDIT (2026-06-17) — per-experiment impl+outcome review (18-agent workflow) + 2 follow-up bugs found & fixed

An adversarial advisor reviewed **each experiment individually** (implementation correctness + independent
reproduction of every headline number from rundirs + outcome soundness). **9 of 10 review units came back
impl=correct / outcome=sound with numbers reproduced to 3 decimals and no reg-shortchange** — specifically
**R1, R2, R3, R10, cc, R-CC+, R4, R5, R9** are all clean (the core R-program is implementation-correct and
its verdicts hold). The audit surfaced **two real bugs, both in the inherited `followup` experiments**
(prior agent), plus one cross-table reporting concern in R4:

**BUG 1 (FIXED) — FU2 `aux_gated` ran with the shared pathway SEVERED.** `next_stan_flow_dualtower/head.py:208`
built `shared_stan` only for `("gated","aux")` and **omitted `"aux_gated"`** → for `aux_gated`,
`shared_stan=None`, so `_fuse` returned `priv + γ·aux_proj(0) = priv + γ·bias` — the input-dependent β
*never saw the shared pathway it was meant to gate*. The FU2 promote-decision (don't adopt aux_gated) is
still safe, but its **mechanistic narrative** ("γ learns to *open* the shared pathway"; "re-confirms champion
β→0 over an *open* pathway") was **invalid**. **Fix:** added `"aux_gated"` to the build set (champion G uses
`"aux"`, already listed → G bit-identical; 5/5 identity tests pass). **Re-ran** FU2 (fixed) AL+FL seed0 — see
the corrected FU2 result in the Follow-up section.

**BUG 2 (FIXED) — FU1 "AL 4-seed" was silently SEED0-ONLY.** `followup_agg.py` read the seed from `p[2]`
(blank) instead of `p[1]` in `r2_al_multiseed_manifest.tsv` → captured 0 multi-seed baselines → the FU1
Idea-1 AL number reported as "4-seed +0.141 reg / +0.178 cat" was actually **seed-0 only**. The true 4-seed
mean is **reg +0.045 / cat −0.066** (cat goes slightly *negative*) — *more* null, so the verdict (R10-GRM
null everywhere) is unchanged and strengthened. **Fix:** read seed from `p[1]`; re-aggregated.

**CONCERN (documented) — R4 uses a different cat "diagnostic-best" convention than the cc/R5 tables.** R4's
aggregators use **convention A** (max over the *mean-fold* epoch curve); cc/R-CC+/R5 use **convention B**
(`full_summary.diagnostic_task_best` = per-fold-best then mean). On identical bit-deterministic champion-G
rundirs this is a fixed **+0.259 pp cat-side offset** (reg is convention-insensitive: ~+0.03pp). It does
**not** change any verdict (R4 applies convention A uniformly, so its internal near-corner conclusion is
intact; cc deltas are convB-vs-convB). The only hazard is **cross-table**: do NOT read R4's cw=0.75 champion
cat (72.878, convA) against the cc/R5 KD-off champion cat (73.137, convB) — the 0.26pp gap is purely the
convention, not a real cat regression. Recommend standardizing cross-table comparands to convB
(full_summary) in any future synthesis.

**Net.** The audit **validates the core R-program** (R-CC+/R4/R5/R9 this session + R1/R2/R3/R10/cc prior
are implementation-correct and outcome-sound), **resolves the cat↑/reg↓ concern** (no loss, no bug — see
the `## CAT↑/REG↓` section above), and **corrects two inherited follow-up bugs** (neither flips any promote
decision). Champion G stands.

---

## SYNTHESIS — 10 lever-families: 9 nulls + 1 sub-threshold positive + R4 (paper-front), one regime (2026-06-17)

| lever | family | mechanism | best seed-0 | multi-seed verdict |
|---|---|---|---|---|
| **R1** | output-level prior | log_C co-location KD `Σ_c P(reg\|c)·P̂(c)` | AL reg +0.33 | **null** (+0.207, sub-threshold) |
| **R2** | asymmetric sharing | binary STEM-AFTB stop-grad masks | AL cat +1.31 | **null** (AL-only; decays to FL −0.03) |
| **R3** | output-level coupling | CrossDistil (live teacher, ec, warm-up, reverse) | AL cat +0.45 | **null** (washes out; log_T-KD saturates the family) |
| **R10** | asymmetric sharing | learned input-conditioned GRM gate | FL cat +0.32 | **null** (washes out; GRM ≡ G) |
| FU-1 | asymmetric sharing | R10 GRM at AZ/GE/AL | — | **null** everywhere (AL 4-seed corrected +0.045 reg / −0.066 cat) |
| FU-2 | input-dependent fusion | `aux_gated` (input-dep β in reg head) | AL cat +0.73 | **null + harmful** (audit-fixed: FL craters BOTH −0.91 cat / −0.31 reg; aux>gated validly re-confirmed) |
| FU-3 | stacking | best-stack of R1+R2+R3+R10 | AL cat +0.46 | **null** (sub-additive; < best component) |
| **★ CC** | **input-side conditioning** | **cat posterior → reg input feature (iMTL/GETNext)** | **FL cat +0.45** | **REAL but sub-threshold** (FL cat **+0.235**, reg **+0.070**, 4/4 seeds positive, audit-confirmed deterministic; richer 256-dim features HURT) |
| **R-CC+** | **input-side conditioning (family map)** | signal {calib,argmax,topk} × inject {film,concat_seq} × output-side logit prior | FL cat +0.45 (calib, ties cc) | **NULL** — every axis ties or **underperforms** additive cc; family caps **+0.235** (calib +0.237, argmax +0.214, concat +0.033 washes out, logitp +0.016 null). The cc cap is the **regime**, not the injection knob |
| **R4** | Pareto-front profiling | scalarization weight-sweep + epoch-front (frozen champion) | — (paper-narrative) | **DONE** — weight-front a **near-corner** (champion cw=0.75 dominant); real trade is the **epoch axis** (stable, geom_simple ep18–20) → **resolves C21**; PaLoRA declined (dual-tower reg-privacy) |
| **R5** | per-instance KD gating | redistribute log_T-KD weight by Markov-coverage (covmax/coventr) | FL cat +0.47 vs global-W | **NULL** — fired the gate but a **comparand artifact**: only recovers log_T-KD's own FL cat-cost; vs the true (KD-off) champion R5 is **−0.22 cat / −0.06 reg** (worse on both); reg ≤ global-W (falsifier met) |

**The shape of the answer.** Everything that **re-gates the cos≈0 cross-task gradient or the output-prior
channel** (R1/R2/R3/R10/FU) is **null** — those channels are empty/saturated. The ONE family that
produces *real* transfer is **input-side conditional coupling** (give the reg head the predicted category
as new input information) — and even that is **capped below the 0.3 gate by the weak 7-class (~2.8-bit)
auxiliary** (richer raw conditioning overfits and hurts). So: champion G's two wins (dual-tower +
log_T-KD) are **replicated but not exceeded** by the post-2022 MTL frontier in this regime — a strong,
citable negative — with conditional coupling marking exactly where (and how little) genuine transfer is
available. **R-CC+ (2026-06-17) has now fully mapped that family** (cleaner semantic conditioning,
FiLM/input-side injection, cat-conditioned logit prior) and found **no variant exceeds the original
additive-posterior cc** — it caps at +0.235 < 0.3, so the sub-threshold bound is the **regime** (weak
2.8-bit auxiliary), not an unexplored knob. The conditional-coupling direction is closed at sub-threshold.
See `## R-CC+` above. *(Historical pointer — R4–R9 are all executed/closed; see `FINAL_SYNTHESIS.md` §3/§5/§7.)*

### (historical, 2026-06-15) first wave + R10 framing

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

## R-CC+ — extend the conditional-coupling family — **NULL** (the cc cap is the regime, not the injection knob) (2026-06-17)

**Verdict.** R-CC+ pushes the one direction that produced real transfer (conditional coupling) along
**three orthogonal axes** — (1) the cat *signal* (softmax / calibrated-τ / discrete-argmax / top-k),
(2) the *injection point* (additive-feature / FiLM / input-side concat-into-sequence / output-side
logit prior), and (3) richer conditioning (already-tested `features`). **Every variant either ties or
underperforms the original additive-posterior `cc_e2e`, and the whole family caps at FL cat ≈ +0.21…+0.24
multi-seed — none clears the 0.3 gate.** The sub-threshold bound is therefore the **regime** (data-rich
main task + weak **7-class ~2.8-bit** auxiliary), **not** a fixable conditioning hyperparameter. Champion
G unchanged; no v17 promotion. This closes the conditional-coupling family.

**New code (all default-off, champion G bit-identical — independently audited, see below).** Three reg-head
axes on `next_stan_flow_dualtower` + the cat-signal transform on `mtlnet_crossattn_dualtower`:
`cond_signal` ∈ {softmax,calibrated(`cond_temp`),argmax,topk(`cond_topk`)} · `cond_inject` ∈
{add,film,concat_seq,none} · `cond_logit_prior` (learned cat→region logit map). Every injection module is
**zero-initialized** → the untrained head ≡ G; FiLM is `(1+γ)·feat+β` (γ=β=0 → identity); concat_seq adds
only on **non-pad** steps so the private-STAN pad mask is preserved. The signal `cat_cond` derives ONLY
from the model's **own predicted** `out_cat` (softmax/argmax) — never the GT label (leak-checked).

**Seed-0 screen (AL+FL, vs fresh matched champion G; gate ≥0.3 either head):**

| variant (FL) | mechanism | Δcat seed0 | note |
|---|---|---|---|
| cc_e2e | posterior, additive (the prior cc) | **+0.450** | reproduces the known +0.45 → no code drift |
| cc_calib | calibrated τ=2, additive | +0.450 | ties posterior |
| cc_argmax | discrete one-hot (GETNext form), additive | +0.418 | ties posterior |
| cc_topk | top-2 mask, additive | +0.438 | ties posterior |
| cc_film | FiLM γ,β on fused feature | +0.243 | **worse** than additive |
| cc_concat | input-side, concat into private-STAN seq (GETNext-faithful) | +0.259 | **worse** than additive |
| cc_logitp | output-side learned cat→region prior | +0.016 | **null** (output channel saturated, cf. R1/R3) |

(reg ~flat +0.02…+0.05 for all. AL = the noisy small state: all variants +0.45…+0.96 cat, the known
seed-0-weakest pattern that washes out multi-seed.)

**FL multi-seed {0,1,7,100} (matched same-batch G, the 4 promote-eligible additive-family + the input-side
control):**

| config | Δreg (n=20, Wilcoxon) | Δcat (4-seed) | per-seed Δcat | gate ≥0.3 |
|---|---|---|---|---|
| cc_e2e | +0.070 (p=0.035) | **+0.235 ± 0.136** | [+0.45,+0.07,+0.21,+0.21] (4/4 +) | **FAIL** |
| cc_calib | +0.067 (p=0.020) | +0.237 ± 0.138 | [+0.45,+0.11,+0.27,+0.12] (4/4 +) | **FAIL** |
| cc_argmax | +0.066 (p=0.008) | +0.214 ± 0.127 | [+0.42,+0.08,+0.22,+0.14] (4/4 +) | **FAIL** |
| cc_concat | +0.020 (p=0.23) | +0.033 ± 0.137 | [+0.26,+0.02,**−0.05**,**−0.10**] (washes out) | **FAIL** |

**Mechanism / what each axis taught us.**
1. **Signal shaping does nothing.** Calibrating (τ=2), discretizing (argmax — GETNext's literal form), or
   sparsifying (top-k) the 7-class posterior all land within ~0.02 of the plain posterior, multi-seed and
   seed-0. The bottleneck is the auxiliary's **information content (~2.8 bits)**, not the signal's form —
   and `cond_proj(softmax(·))` is already a soft category-embedding lookup, so the "cleaner embedding"
   variants are re-parameterizations of the same map. The HANDOFF hypothesis ("a cleaner semantic signal
   beats the raw 7-dim posterior") is **falsified**.
2. **Injection point matters, and additive-late is best.** FiLM and the input-side concat are both
   *worse* than the additive feature injection. The input-side `concat_seq` — multi-seeded specifically
   to rule out an identity-init slow-start confound (advisor request) — **washes out** (Δcat +0.033, two
   seeds negative), i.e. it is genuinely worse, not merely slow-starting. So `cc_e2e`'s additive form is
   the **family optimum**, not an arbitrary choice.
3. **Output-side and richer-signal both fail.** The learned cat→region logit prior is null (+0.016 —
   the output-prior channel is saturated, the same wall R1/R3 hit), and the richer 256-dim `features`
   conditioning was already shown to **hurt** FL (−0.31/−0.36, the prior cc work) — overfitting the
   noisy penultimate. Raising raw capacity makes it worse, not better.

**The synthesis.** Across signal × injection × output-side × capacity, **nothing exceeds the original
additive-posterior cc, and that optimum caps at +0.235 < 0.3.** This is the cleanest possible in-study
proof that the conditional-coupling sub-threshold is a **hard regime cap set by the 2.8-bit category
auxiliary**, not an unexplored-hyperparameter artifact. It is the honest paper framing: input-side
conditioning is the one channel that yields *real* transfer here (both heads, every seed positive on the
additive family), and the LBSN next-POI regime's weak category vocabulary bounds it just below promotion.

**Independent advisor audit (2026-06-17).** A separate reviewer read the code and verified, with
file:line evidence, all five correctness claims **PASS**: (1) G bit-identity when off, (2) zero-init ≡ G
for every inject mode (incl. FiLM identity + concat_seq pad-mask preservation), (3) **no GT leakage**
(`cat_cond` ← predicted `out_cat` only), (4) train/eval consistency (eval runs the joint `forward`, so
conditioning fires identically; the disjoint `next_forward` path is inert by design), (5) valid signal
transforms. No bugs. Its methodological catch — don't drop the input-side `concat_seq` on confounded
seed-0 evidence — was adopted (the concat multi-seed addendum above), and confirmed the cap rather than
breaking it.

**The one untested lever (future, not pursued — out of "pause-after-R-CC+" scope).** The advisor's
remaining suggestion is a **cross-attention coupling** (cat penultimate as K/V, queried by the reg pooled
feature) so the reg head can *select* relevant cat dims per-sample, rather than additive/scalar injection.
Given (a) `features` already hurts and (b) the bottleneck is diagnosed as auxiliary *information*, the
expectation is another sub-threshold result — but it is the one structurally-distinct mechanism not yet
run, and is logged here for completeness.

**Artifacts.** `docs/results/mtl_frontier/{ccplus_screen_results.json, ccplus_fl_multiseed_results.json}`;
drivers `scripts/mtl_frontier/{ccplus_screen.sh, ccplus_fl_multiseed.sh, ccplus_fl_concat_addendum.sh}`;
aggregators `{ccplus_agg.py, ccplus_fl_multiseed_agg.py}`; manifests `{ccplus_manifest.tsv,
ccplus_fl_multiseed_manifest.tsv}`. Code: `cond_signal`/`cond_temp`/`cond_topk`/`cond_inject`/
`cond_logit_prior` in `next_stan_flow_dualtower/head.py` + the signal transform in
`mtlnet_crossattn_dualtower/model.py` (champion G defaults unchanged).

---

## R4 — Pareto-front profiling (PaLoRA-style) — **DONE (paper-narrative; resolves C21)** (2026-06-17)

**Verdict.** R4 publishes the cat↔reg trade-off **front** instead of defending one geom_simple point,
permanently resolving the C21/geom_simple selector saga. Two axes were profiled on the **frozen champion
G** (FL, multi-seed {0,1,7,100}; R4 is paper-narrative — no promote gate): **(1) the scalarization /
mixture-weight axis is a near-corner** — champion cw=0.75 is Pareto-dominant and the tasks are weakly
coupled there (the spec's falsifier, "front collapses → publishable regime datum"); **(2) the
deployment-epoch axis carries the real, stable cat↔reg trade** — 12–16 Pareto-optimal epochs per run,
geom_simple deploying a consistent ep18–20 point across all seeds. So C21 is an **epoch-deployment** choice
on a real epoch-front, **not** a weight/architecture problem. PaLoRA-proper was **not built** (justified
below).

**Method.** Sweep `--category-weight` (static_weight: `loss=(1−cw)·L_reg + cw·L_cat`) on the frozen
champion, reg/cat heads + recipe identical to G, log_T-KD off. Each weight = a deployable model; the
(cat-F1 ceiling, reg-Acc@10 ceiling) set is the achievable front. Per run we also extract the per-epoch
(mean-fold cat-F1, mean-fold reg-matched) trajectory → the within-run epoch Pareto front, and the
geom_simple-selected epoch (argmax_e √(cat_e·reg_e)). No new model code — only drivers + aggregators.

**(1) Scalarization weight-front, 4-seed diagnostic-best ceilings (FL):**

| cw | cat-F1 (mean±std) | reg-Acc@10 (mean±std) | note |
|---|---|---|---|
| 0.55 (reg-favored) | 72.005 ± 0.013 | 72.988 ± 0.079 | reg-corner |
| **0.75 (champion)** | **72.878 ± 0.119** | **72.938 ± 0.057** | Pareto-dominant |
| 0.85 (cat-favored) | 72.387 ± 0.061 | 72.427 ± 0.123 | **dominated** (both ↓) |

(seed-0 also swept {0.40, 0.70, 0.92}: the reg ceiling is flat ~72.9–73.05 for cw ≤ 0.75, then cw=0.92
**craters reg to 64.26** — pure reg-loss starvation, a degenerate under-training point, not a Pareto
trade.) **Lowering cw 0.75→0.55 buys only +0.050 pp reg for −0.873 pp cat; raising to 0.85 loses on
BOTH heads.** The achievable weight-front is therefore a **tiny near-vertical segment** with champion
cw=0.75 at the knee — the cat↔reg tasks barely trade off on the loss-weight (and, by the argument below,
the mixture) axis. **Near-corner / weak-coupling confirmed multi-seed** — exactly the cos≈0, dual-tower,
champion-near-joint-optimum regime.

**(2) Champion epoch-trajectory front (the real C21 locus), stable across seeds:**

| seed | #Pareto epochs | geom_simple epoch | deployed (cat, reg) |
|---|---|---|---|
| 0 | 14 | 19 | (72.66, 71.88) |
| 1 | 14 | 18 | (72.91, 71.92) |
| 7 | 16 | 19 | (73.03, 72.13) |
| 100 | 12 | 20 | (72.88, 72.05) |

Within a single champion run there are 12–16 Pareto-optimal epochs (the late-epoch cat↔reg tension is
real: per-task ceilings sit at different epochs), and **geom_simple deploys a consistent ep18–20 point at
every seed**. This is the genuine selector locus: publishing this epoch-front (with geom_simple's pick
marked) is R4's "publish the front, not a point" deliverable, and it shows the C21 choice is stable and
well-localised — not the knife-edge the selector saga implied.

**Why PaLoRA-proper was not built (mechanistic, not a shortcut).** PaLoRA traces a front by mixing
**shared-trunk** LoRA adapters. But in the dual-tower champion, **reg's signal flows through its private
STAN tower**, not the shared trunk (the cat head harvests the shared trunk; reg is isolated — the X3
dual-tower finding). A shared-trunk adapter mixture therefore moves **cat** and barely moves **reg** — it
would reproduce the same near-collapsed weight-front measured above, at substantial implementation cost.
The weight-sweep already *is* the convex-emphasis front, and it collapses to a near-corner; a LoRA-mixture
front would too. **The dual-tower's reg-privacy means a shared-trunk Pareto-profiler cannot trace a reg
trade-off in this architecture** — itself a clean, citable paper point (and the honest reading of the
R4 falsifier).

**Takeaway for the paper.** The cat↔reg front in this LBSN-MTL regime is (a) near-degenerate on the
loss-weight / shared-trunk-mixture axis — champion G sits at the joint corner, tasks weakly coupled
(cos≈0) — and (b) a real but **stable, well-localised** trade on the deployment-epoch axis, where the
geom_simple selector lands consistently. Reporting the epoch-front + the geom_simple point (rather than
arguing one scalar) is the principled resolution of the C21 selector class. Champion G unchanged; nothing
to promote (paper-narrative lever).

**⚠ Convention note (2026-06-17 audit).** R4's aggregators report cat diagnostic-best via **convention A**
(max over the *mean-fold* epoch curve); the cc/R-CC+/R5 tables use **convention B** (`full_summary`
per-fold-best then mean). On identical champion-G rundirs these differ by a fixed **+0.259 pp** on the cat
axis (reg is convention-insensitive). R4's internal conclusions are intact (convention A applied uniformly),
but do NOT cross-compare R4's cw=0.75 champion cat (72.878, convA) against the cc/R5 KD-off champion cat
(73.137, convB) — the gap is the convention, not a real effect. Standardize to convB for any cross-table
comparand.

**Artifacts.** `docs/results/mtl_frontier/{r4_scalar_front_results.json, r4_front_multiseed_results.json}`;
drivers `scripts/mtl_frontier/{r4_scalar_front.sh, r4_front_multiseed.sh}`; aggregators
`{r4_front_agg.py, r4_front_multiseed_agg.py}`; manifests `{r4_scalar_front_manifest.tsv,
r4_front_multiseed_manifest.tsv}` (+ champion cw=0.75 reuses the deterministic `ccplus_fl_multiseed`
base rows). No code changes (champion G untouched).

---

## R5 — per-instance KD gating — **NULL** (a fired gate that is a comparand artifact) (2026-06-17)

**Verdict.** Per-instance gating of the log_T-KD weight (redistribute the batch-mean-fixed KD weight by
Markov-coverage of the sample's last-region transition row — peaked row ⇒ upweight) produces a robust FL
multi-seed **cat lift that clears the ≥0.3 gate against its designed global-W comparand** — but this is a
**comparand artifact**: the lift merely recovers part of log_T-KD's own FL cat-cost, and against the
study's true FL champion (KD-OFF G) R5 is **worse on BOTH heads**. On the KD's own target axis (reg) gated
≤ global-W (the R5 falsifier is met). **No v17 promotion; champion G stands.** Independently advisor-audited
(verdict: null, do not escalate).

**The decisive table (FL, 4-seed {0,1,7,100}, diagnostic-best; cat=macro-F1, reg=matched Acc@10):**

| config | cat | reg | vs KD-OFF champion |
|---|---|---|---|
| **KD-OFF champion G** (the true FL champion; §0.1 paper canon) | **73.137 ± 0.076** | **72.958 ± 0.06** | — |
| R5 base = G + GLOBAL log_T-KD W=0.2 (gate=none) | 72.441 ± 0.059 | 72.995 ± 0.055 | cat −0.696 / reg +0.037 |
| R5 covmax (gate=coverage_max) | 72.913 ± 0.081 | 72.895 ± 0.068 | **cat −0.224 / reg −0.063** |
| R5 coventr (gate=coverage_entropy) | 72.863 ± 0.079 | 72.912 ± 0.073 | cat −0.274 / reg −0.046 |

**vs the designed global-W comparand:** covmax Δcat **+0.472 ± 0.042** (4/4 seeds + [0.514,0.404,0.471,0.498],
p=0.0625 = the n=4 sign-consistency floor), Δreg **−0.100** (Wilcoxon p≈1.0 — significantly *worse*);
coventr Δcat +0.422, Δreg −0.082. **vs the true champion (KD-OFF G):** both variants are negative on cat
AND reg. AL seed0: covmax +0.236 reg/+0.181 cat, coventr +0.189 reg/−0.014 cat — **gate fails at AL** (the
state where log_T-KD actually lives).

**Why the gate fired, and why it is null (mechanism).** log_T-KD is a **reg-side** KD that is "small-state
only" — at the large headline state FL it **costs 0.70 pp cat for +0.04 pp reg** (the global-W base sits at
72.441 cat vs the KD-off champion's 73.137). Per-instance gating (concentrating the fixed KD budget on
high-Markov-coverage samples) **recovers ~2/3 of that self-inflicted cat-cost** (+0.472 of the 0.696 lost)
via the shared cross-attn trunk — a genuine, clean, mechanistically-interpretable spillover (`r5_gate_std`
≈ 0.4–0.6 confirms the gate genuinely redistributes, C28-live) — **but it never fully recovers it, and it
gives back reg on the KD's own axis (−0.10).** So:
- Against the **designed control** (global-W KD-on), R5 measurably shifts the cat↔reg trade — a citable
  mechanism datum (instance-gated KD redistribution works as a knob).
- Against the **v17 comparand** (deployable champion G = KD-OFF), R5 is Pareto-**dominated** (−0.22 cat,
  −0.06 reg), and is itself **dominated by the trivial known alternative** "turn log_T-KD off at FL"
  (which recovers the full 0.70 cat at ~flat reg). The falsifier "gated ≤ global-W" holds on reg.

**Takeaway.** R5 is the same regime as R1/R2/R3/R10/cc: an output-prior/KD-family lever that is
small-state-flavoured, FL-null-or-harmful, and reg-immovable. Its one positive (the +0.472-vs-global-W cat
recovery) is a **comparand artifact** of the handicapped log_T-KD-ON FL base — exactly the kind of fired
gate the matched-baseline protocol (re-baseline on the true champion; don't compare across recipes) exists
to catch. **Methodological note for the paper/next agent:** the promote-gate must be read against the
*deployable champion*, not a lever's internal control; R5 is the case study. **Incidental finding** (worth
citing): this cleanly re-quantifies that **log_T-KD(0.2) is cat-harmful at FL** (−0.70 cat), and that
per-instance Markov-coverage gating recovers ~2/3 of that cost but never beats KD-off.

**Audit refinement (2026-06-17).** The cat/reg audit (`## CAT↑/REG↓` section) shows R5's reg-down is also
*partly a diagnostic-best artifact*: cat and reg peak ~26 epochs apart, and at the single deployable
geom_simple checkpoint R5 covmax is cat-up/reg-**flat** (Δreg −0.025, n=20 p=0.73 = null) rather than
reg-down. So R5 is even more clearly "not a reg loss" — it neither beats the KD-off champion (Pareto-
dominated) nor genuinely sacrifices reg at the deployable point; it just partially un-does log_T-KD's FL
cat-cost. Verdict unchanged (NULL, no v17).

**Code (champion G bit-identical when off; default `none`).** `--log-t-kd-gate {none,coverage_max,
coverage_entropy}` → `ExperimentConfig.log_t_kd_gate` → the KD branch in `mtl_cv.py` applies a per-sample,
batch-mean-1-normalized weight from the teacher-row peakedness (`coverage_max`=max-prob, `coverage_entropy`
=normalized 1−H); C28 diagnostic `r5_gate_std`. (A one-line `_math` scope bug cost the first `coventr`
pass — `coverage_max` was unaffected — fixed + smoke-verified before the multi-seed.)

**Artifacts.** `docs/results/mtl_frontier/{r5_screen_results.json, r5_fl_multiseed_results.json}`; drivers
`scripts/mtl_frontier/{r5_screen.sh, r5_coventr_fix.sh, r5_fl_multiseed.sh}`; aggregators `{r5_agg.py,
r5_fl_multiseed_agg.py}`; manifests `{r5_screen_manifest.tsv, r5_fl_multiseed_manifest.tsv}`. Code:
`log_t_kd_gate` in `experiment.py` / `train.py` / `mtl_cv.py` (champion G defaults unchanged).

---

## R9 — residual optimizer sanity (BayesAgg-MTL) — **DONE (optimizer aisle closed citably)** (2026-06-17)

**Verdict.** The R9 spec named **BayesAgg-MTL** (ICML'24) "the one non-vacuous optimizer at cos≈0"
(weights by gradient *uncertainty*, not conflict). Verification + a champion-recipe re-run close it:
**(1)** the repo's `bayesagg_mtl` is **not faithful** to the paper — it weights by gradient **magnitude**
variance (Kendall-style task difficulty, inverse-variance EMA), **not** the ICML'24 posterior
gradient-**direction** uncertainty; **(2)** re-run at the **champion recipe** (not the registry defaults
that produced the undiagnosed 19-arm crater) it **still craters cat** — AL **37.754** (−14.6 vs G 52.378;
reproduces the documented 19-arm 37.75 to 3 decimals), FL **63.689** (−9.3 vs 73.012), reg flat. So the
crater is an **impl pathology** (the magnitude-variance weighting catastrophically down-weights the
7-class cat head against the 1109-class reg head), **not** a defaults artifact → the 19-arm bayesagg null
is now **diagnosed**. The faithful ICML'24 method is **unbuilt** and **expected-null by the regime**;
**R9(b) Smooth-Tchebycheff is unmotivated** (R4's front is near-degenerate, not the non-convex front that
would justify it). **Optimizer aisle stays closed; champion G stands.**

**Champion-recipe screen (seed0, `--mtl-loss bayesagg_mtl` vs G `static_weight 0.75`):**

| state | champion cat / reg | bayesagg cat / reg | Δcat |
|---|---|---|---|
| AL | 52.378 / 62.381 | **37.754** / 62.554 | **−14.624** (crater) |
| FL | 73.012 / 72.929 | **63.689** / 72.786 | **−9.323** (crater) |

**Why this closes the aisle.** The optimizer aisle was already closed by the 19-arm registry null
(`mtl_improvement` Tier 4) + Kurin/Xin (NeurIPS'22, "scalarization suffices") + Mueller (TMLR'25, "gradient
conflict/geometry doesn't predict transfer"). R9 removes the one asterisk: the 19-arm bayesagg cell was
flagged "misconfigured-at-defaults, undiagnosed." The champion-recipe re-run shows the crater is **recipe-
invariant** (AL 37.75 reproduced exactly) → the repo impl is simply broken for this 7-vs-1109-class task,
not a valid test of uncertainty-weighting. The **faithful** ICML'24 BayesAgg (Bayesian gradient-direction
posterior, inverse-variance aggregation) is the single optimizer arm never run here; a faithful port is a
substantial Laplace/last-layer reimplementation, and at cos≈0 with a representational (not optimization)
gap, the regime predicts null — so it is **logged as the one remaining optimizer probe and deferred**, not
built (consistent with R9's "trivial close-out, expected null" scoping). No promotion (optimizer
close-out).

**Artifacts.** `docs/results/mtl_frontier/r9_screen_results.json`; driver
`scripts/mtl_frontier/r9_bayesagg_screen.sh`; aggregator `r9_agg.py`; manifest `r9_screen_manifest.tsv`.
No code changes (existing `bayesagg_mtl` re-run as-is; champion G untouched).

---

## R6 / R7 / R8 — the medium-priority tail — **R7 MEASURED (Merge<G); R6/R8 predicted-negative, deferred** (2026-06-17)

> **Status note.** **R7 was RUN (user-requested 2026-06-17) → MEASURED negative (Merge < joint G; see
> below).** R6 and R8 remain substantial *from-scratch* implementations (training-loop fork/merge surgery;
> a 2→3-task architecture refactor) whose outcomes are **mechanistically determined** by the regime + the
> specific prior results below; per the study's *What NOT to pursue* discipline they stay **rigorous
> predicted-negatives, deferred as future work** (the user chose to run R7 only). Each carries its
> mechanism, determining evidence, and reopening falsifier; **a measured negative for R6/R8 is one driver
> away if the paper wants it.**

### R6 — ForkMerge-style weight forking — predicted ≤ champion G
**Mechanism.** Periodic forks into branches with different (w_cat, w_reg); select/merge on *validation*
error (NeurIPS'23). At K=2 = "fork a cat-leaning and a reg-leaning branch, keep the better on val, repeat."
**Why determined.** Two existing results bracket it: (a) **`scheduled_static`** (a time-varying task-weight
schedule) is **already in the 19-arm null** — fixed weight-scheduling does not beat static; (b) **R4**
showed the *static* weight-front is a **near-corner** (champion cw=0.75 Pareto-dominant; lowering→0.55 buys
+0.05pp reg for −0.87pp cat, raising→0.85 loses on both). ForkMerge's only delta over (a)+(b) is
*validation-driven adaptive* weight selection — and there is no slack for it to exploit: the achievable
(cat,reg) set is a corner with the champion weight already on it. **Falsifier (to reopen):** a val-driven
adaptive weight beats static cw=0.75 by ≥0.3 either head, multi-seed — which would contradict both the
scheduled_static null and the R4 near-corner. **Cost if run:** fork/merge surgery in `mtl_cv` (no
checkpoint-resume exists) + a multi-period driver.

### R7 — Merge-vs-joint (ZipIt!/SIMO) — **DONE (MEASURED) 2026-06-17: Merge < joint G** (citable LBSN negative)
**Mechanism.** Two independent single-task specialists in the champion-G dual-tower arch (cat-only cw=1.0,
reg-only cw=0.0, FL seed0). The **ensemble** (each task served by its own specialist) is the *best-case*
merge and a rigorous **upper bound** on any weight-merge (sharing weights can only constrain per-task
performance). If even the ensemble loses to joint G, no merge can match joint training.

**Result (FL seed0, matched bar):**

| | cat | reg |
|---|---|---|
| **joint champion G** | **73.012** | **72.929** |
| cat-specialist (cw=1.0) | 72.235 | (reg head untrained) |
| reg-specialist (cw=0.0) | (cat head untrained) | 72.952 |
| **ensemble = best-case merge** | **72.235 (−0.777 vs G)** | **72.952 (+0.023 vs G)** |

**Verdict: Merge < joint G.** Even the best-case ensemble **loses −0.78 pp cat to gain +0.02 pp reg** — a
net loss; any real weight-soup is ≤ this bound. **The joint trunk's cat benefit is unrecoverable by merging
independently-trained specialists** — exactly the tangent-space prediction (Ortiz-Jimenez NeurIPS'23:
from-scratch experts don't share a basin) realized in LBSN-MTL. reg is **identical** joint-vs-specialist
(+0.02), consistent with the β→0 reg-insulation finding (reg doesn't use the shared trunk at FL, so neither
joint co-training nor merging moves it). **Conservative bound:** the cat-specialist is a `cw=1.0` (λ=0 reg)
ablation, which — per the paper's own C3 "cross-attention λ=0 pitfall" — *still co-adapts* the reg encoder
via cross-attn K/V, so cat-specialist cat (72.235) is *higher* than a true from-scratch STL cat (~70); the
real joint advantage is ~−3 pp, so **−0.78 understates Merge<G.** Falsifier (Merge ≥ G) NOT met. Champion G
stands. **Artifacts:** `docs/results/mtl_frontier/r7_merge_results.json`; driver
`scripts/mtl_frontier/r7_specialists.sh`; agg `r7_agg.py`; manifest `r7_specialists_manifest.tsv`.

### R8 — Auxiliary third task (next-visit-time) — predicted rising-tide null
**Mechanism.** Add a next-visit-time loss as a 3rd auxiliary task (GETNext/Where-and-When pattern).
**Why determined.** The **rising-tide rule** (this repo's repeated finding: substrate/auxiliary gains lift
STL as much as MTL — `substrate-protocol-cleanup` CLOSURE) predicts a time-aux lifts the STL reg ceiling as
much as the MTL reg, i.e. **no net MTL-specific transfer** — exactly the spec's own falsifier ("lifts STL
and MTL equally"). The cos≈0 regime means a 3rd task adds no first-order cross-task transfer through the
trunk either. **Architecture cost is also disproportionate:** this codebase hardcodes **2-task** MTL
(`StaticWeightLoss` asserts `n_tasks==2`; the dual-tower has exactly cat+reg heads), so R8 is a multi-file
2→3-task refactor (loss, model, runner, input builders) for an expected rising-tide null — the data
*supports* it (check-ins carry `datetime`/`local_datetime`), but the cost/expected-value is poor.
**Falsifier:** time-aux lifts MTL reg strictly more than a paired STL-reg+time control, ≥0.3 multi-seed.
**MUST run with the paired STL control** if executed.

**Net (R6/R7/R8).** All three are bounded to null/negative by the completed regime picture (cos≈0,
dual-tower, champion-near-joint-optimum) plus their specific determinants (scheduled_static-null + R4
near-corner for R6; tangent-space theory for R7; rising-tide + 2-task architecture for R8). **R7 was
measured (Merge < joint G, −0.78 cat / +0.02 reg) and confirms the prediction** — the joint trunk's cat
co-training benefit is unrecoverable by merging. R6 and R8 are closed by reasoned prediction (measured
runs available on request), completing the R1–R10 + R-CC+ + R4/R5/R7/R9 program without spending
disproportionate implementation effort on the two remaining mechanistically-determined nulls.

---

### Follow-up screens (user ideas, advisor-structured) — three more nulls (2026-06-15)

After the 4-null first wave, three user-proposed follow-ups were screened (advisor evaluation in
the session record). All null; one is informative.

**Idea 1 — R10 GRM at the other states (close the "FL-only multi-seed" gap).** GRM screened at AZ/GE
seed0 + AL multi-seed {0,1,7,100} vs champion G (baselines reused). **Null everywhere:** AZ Δreg
−0.239 / Δcat +0.140; GE +0.083 / +0.269; AL 4-seed **Δreg +0.045 / Δcat −0.066** *(2026-06-17 audit
correction — `followup_agg.py` read the seed from the wrong column and reported the AL number as seed-0-
only "+0.141/+0.178"; the true 4-seed mean is +0.045 reg / −0.066 cat — cat slightly NEGATIVE, even more
null)*. Confirms the trained-γ prediction — GRM's "nothing to gate" is state-agnostic; R10 is null at all
4 built states.

**Idea 2 — `aux_gated` fusion: input-dependent β in the reg head (the best-shot idea).** New fusion
mode `feat = priv + γ(·)·aux_proj(shared)`, γ=σ(MLP([priv;shared])) per-dim (init γ≈0.12≈champion β).
**⚠ 2026-06-17 AUDIT: the original FU2 run was BUGGY** — `head.py:208` omitted `"aux_gated"` from the
shared-tower build list, so `shared_stan=None` and the lever silently degenerated to `priv + γ·aux_proj(0)
= priv + γ·bias` (the shared pathway it was meant to gate **was never present**). The original numbers
(AL cat +0.48 / FL cat −0.85, reg flat) and the "γ *opens* the shared pathway" narrative were therefore
**invalid**. **Fixed** (`"aux_gated"` added to the build set; champion `aux` untouched → G bit-identical)
and **re-ran** AL+FL seed0. **Corrected result (shared pathway genuinely live):** AL Δreg **−0.221** /
Δcat **+0.728** (seed-0 cat flare, now with a reg cost); **FL Δreg −0.308 / Δcat −0.910 — craters BOTH
heads.** This **strengthens** the verdict and now supports it *validly*: with the shared pathway actually
open, the input-dependent γ dilutes large-state performance and is **harmful on both axes at FL** —
re-confirming the champion's `aux`/β→0 design (close the shared→reg pathway) over a genuinely-open
input-conditioned gate, exactly as the falsified convex `gated` mode predicted. Not a lever. (Code:
`fusion_mode=aux_gated` + `aux_gamma` diagnostic; champion `aux` unchanged.)

**Idea 3 — stack the best-direction variants** (G + log_T-KD + log_C-KD + reverse cat-KD + aftb_late +
GRM) vs G+log_T-KD, AL+FL seed0, pre-registered to promote only on **super-additivity** (>0.3 over the
best single component). **Sub-additive null:** AL cat +0.456 < the best single component (aftb_late
+0.636); FL +0.091 reg / +0.109 cat. The levers are redundant views of one mechanism and **interfere**
(as `r3_both` already showed) — stacking is worse than the best component, not better. Fails the
pre-registered gate.

**Net:** the three follow-ups add a 5th/6th/7th null and one positively-useful datum — Idea 2 (now on the
*audit-corrected, bug-fixed* run) shows the champion's β→0 / `aux`-over-`gated` choice is **harmful to
overturn even with an input-conditioned, genuinely-open shared pathway** (FL craters both heads −0.91/
−0.31), *validly* strengthening the architecture claim. The recommended next direction (out of the cos≈0
box that all of R1/R2/R3/R10 + these three have now nulled) is **conditional coupling**: feed the cat
head's *output* as an *input feature* to the reg head (GETNext/iMTL pattern) — the one mechanism with a
literature prior for beating parity in this regime (FINAL_SYNTHESIS §4; advisor recommendation). Artifacts:
`docs/results/mtl_frontier/followup_results.json`; drivers `scripts/mtl_frontier/{followup_screens.sh,
fu2_rerun.sh}`; agg `followup_agg.py` (seed-column fix). *Audit note: the original FU2 numbers (severed
shared pathway) and the FU1 "AL 4-seed" (seed-0-only) are corrected above; see `## AUDIT (2026-06-17)`.*

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
two mechanisms the `mtl_improvement` study already found. *(Historical note: "remaining program R4–R9" was
later all executed/closed — R4/R7/R9 done, R5 null, R6/R8 reasoned-negatives; see `FINAL_SYNTHESIS.md`.)*
The optimizer aisle stays closed (Kurin/Xin/Mueller + R9).
