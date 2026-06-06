# MTL Improvement — Tier 0 + Tier 1 results of record (2026-06-03)

Consolidated numbers for the Tier-0/Tier-1 execution. The narrative + verdicts live in
[`docs/studies/mtl_improvement/INDEX.html`](../../studies/mtl_improvement/INDEX.html) (Results
blocks) and [`log.md`](../../studies/mtl_improvement/log.md). This file is the durable, greppable
record — especially for the MTL board cells, whose per-fold run dirs sit under the gitignored
`results/` tree.

**Protocol** (unless noted): leak-free, seeded per-fold `region_transition_log_seed{S}_fold{N}.pt`,
frozen-fold partitions (`frozen_folds/{state}_seed{S}.json`). reg = `top10_acc_indist` (Acc@10 %),
cat = macro-F1 (%). Selector default = `geom_simple = sqrt(cat_F1·reg_Acc@10)` (C21).

---

## States on the scale axis

| band | state | n_regions | n_samples | n_users | recipe |
|---|---|---|---|---|---|
| small | AL | 1,109 | 12,709 | 1,622 | H3-alt |
| small | AZ | 1,547 | 26,396 | 3,331 | H3-alt |
| **middle** | **GE** | **2,283** | **44,978** | **5,038** | H3-alt |
| large | FL | 4,703 | 159,175 | 13,935 | B9 |

GE onboarded from scratch 2026-06-03 (GA TIGER shapefile downloaded; canonical check2hgi + HGI +
POI2Vec + v14 design_k + seeded log_T). GE HGI trained in 118 s.

---

## T0.2 / T0.3 — MTL board: v14 vs matched canonical (Δ = v14 − canon; Δ>0 ⇒ v14 better)

KD-off, seeds {0,1,7,100}, 5-fold. FL/AL/AZ imported from
[`../v14_mtl_vs_canonical.md`](../v14_mtl_vs_canonical.md); GE produced here (`ge_board.sh` +
`ge_board_agg.py`). **Caveat:** FL/AL/AZ "canonical" = FROZEN v11 on-disk substrate (privileged
draw ~+0.5pp reg); GE "canonical" = freshly built today. FL=B9, AL/AZ/GE=H3-alt.

### Deployable (geom_simple selector) — the honest single-model number
| state | v14 reg | canon reg | Δreg | v14 cat | canon cat | Δcat |
|---|---|---|---|---|---|---|
| AL | 50.14 | 48.00 | **+2.14** | 46.50 | 45.29 | +1.21 |
| AZ | 37.78 | 38.79 | **−1.01** | 48.52 | 47.81 | +0.72 |
| GE | 42.64 | 43.79 | **−1.15** | 56.13 | 55.42 | +0.70 |
| FL | 61.21 | 61.54 | **−0.33** | 66.73 | 66.77 | −0.04 |

### Diagnostic-best (per-task own-best epoch)
| state | v14 reg | canon reg | Δreg | v14 cat | canon cat | Δcat |
|---|---|---|---|---|---|---|
| AL | 47.23 | 49.46 | −2.23 | 46.78 | 45.96 | +0.82 |
| AZ | 38.27 | 40.29 | −2.02 | 48.75 | 48.86 | −0.11 |
| GE | 44.61 | 46.08 | −1.47 | 57.07 | 57.26 | −0.18 |
| FL | 61.28 | 61.49 | −0.21 | 70.26 | 70.34 | −0.09 |

**Verdict:** regime CONFIRMED at every band incl. GE (middle) — v14 ≈ matched canonical in MTL; the
STL dual-axis gains do not survive. Deployable-Δreg sign pattern **AL +2.14 / AZ −1.01 / GE −1.15 /
FL −0.33** ⇒ v14 reg-survival is **AL-specific, not a scale gradient** (magnitudes confounded by
fresh/frozen + recipe; only the sign is interpretable; each state's Δ is a valid paired comparison).
GE board run dirs: `results/{check2hgi_design_k_resln_mae_l0_1,check2hgi}/georgia/mtlnet_*ep50*`
(see `/tmp/ge_board/manifest.tsv` provenance / `scripts/_v14_run/{,canon_}manifest.tsv` for FL/AL/AZ).

---

## T1.1 — (c) STL-on-v14 ceiling (seed42, 5-fold, in-harness)

| state | cat v14 / canon / Δ | reg v14 / canon / HGI | v14 closes of canon→HGI reg |
|---|---|---|---|
| AL | 39.13 / 38.35 / +0.78 | 62.32 / (61.21§) / 63.05 | — |
| AZ | 43.16 / 42.92 / +0.25 | 52.87 / (53.06§) / 53.50 | — |
| GE | 54.02 / 53.42 / +0.60 | 55.81 / 54.36 / 56.50 | **~68%** (HGI keeps +0.69) |
| FL | 65.88 / 64.61 / +1.27 | 70.28 / 69.43 / 70.62 | ~68% (HGI keeps +0.34) |

§ = §0.1 frozen ref (AL/AZ canon STL-reg not re-run in-harness). v14 STL cat ≥ canon at every band;
v14 STL reg closes ~68% of canon→HGI at GE exactly as at FL (Delaunay lever reproduces at the
middle band). reg JSONs: `docs/results/P1/region_head_{state}_region_5f_50ep_{t13_cfg1_raw_v14,
ge_stlreg_v14,ge_stlreg_canon,*_stlreg_hgi}_s42.json`; cat:
`results/{eng}/{state}/next_*ep50*/summary/full_summary.json`.

---

## T1.2 — (d) composite deploy ceiling (cat=STL-v14, reg=STL-HGI) vs MTL deployable reg

| state | composite cat | composite reg | MTL deployable reg (v14) | composite gap over MTL |
|---|---|---|---|---|
| AL | 39.13 | 63.05 | 50.14 | **+12.91** |
| AZ | 43.16 | 53.50 | 37.78 | **+15.72** |
| GE | 54.02 | 56.50 | 42.64 | **+13.86** |
| FL | 65.88 | 70.62 | 61.21 | **+9.41** |

The two-model composite beats the single MTL model by +9.4 to +15.7pp reg at zero cat cost; gap is
larger at small/middle states. **This is the in-harness upper bound the T2.1 dual-tower must
approach inside one model.**

---

## T1.3 — encoder-isolation probe (gates Tier 2)

STL `next_stan_flow` reg on v14 region-emb, 3 configs, AL/AZ/FL, 5f×50ep seed42, frozen-fold paired.
`t13_encoder_probe.sh` + `t13_agg.py`.

| state | cfg1 raw (ceiling) | cfg2 +MTL encoder | cfg3 +input LN | gap cfg1−cfg2 |
|---|---|---|---|---|
| AL | 62.32±4.00 | 63.25±3.94 | 62.11±3.98 | −0.93 |
| AZ | 52.87±2.70 | 53.44±2.91 | 52.94±2.74 | −0.57 |
| FL | 70.28±0.54 | 70.40±0.53 | 70.24±0.54 | −0.13 |

### Prior-OFF re-run (advisor P1 — embeddings-only, freeze_alpha=True alpha_init=0.0)
| state | cfg1 raw | cfg2 +MTL encoder | gap cfg1−cfg2 |
|---|---|---|---|
| AL | 62.88±4.05 | 63.77±3.60 | −0.88 |
| AZ | 55.11±2.96 | 55.41±2.79 | −0.30 |
| FL | 73.31±0.41 | 73.97±0.57 | −0.66 |

**Verdict:** the MTL encoder costs ~0pp on BOTH the with-prior and embeddings-only metrics (cfg2 ≥
cfg1 everywhere) → the encoder architecture is NOT the residual; the locus is the joint-training
dynamics (cross-attn / PCGrad / shared-backbone handoff) → **T2.1 dual-tower is the lever; the
encoder-bypass would not help.** Honest scope: tests a standalone encoder, not the jointly-trained
one in situ. Side-finding: at FL embeddings-only (73.31) > with-prior (70.28) → the log_T prior is a
net drag on FL STL reg with v14's strong embeddings (flag for Tier 3). JSONs:
`docs/results/P1/region_head_{state}_region_5f_50ep_{t13_cfg*,t13po_cfg*}_v14_s42.json`. Graph regen
determinism verified (region_idx exact-match; cfg1 FL 70.28 ≈ landed 70.24).

---

## T1.4 — Tuned-incumbent ceiling (FROZEN (c)/(d) · closes Tier 1)

Full per-task HP tune of the incumbent heads; STL-alone on v14, frozen folds, seeded log_T,
5f×50ep seed=42. New leak-free loss code `src/losses/calibrated.py` (logit-adjust + focal +
label-smoothing + balanced/CB/LDAM; class stats from TRAIN split only; 19 unit tests). Two
harnesses: **reg** via `p1_region_head_ablation.py` (loads v14 region-emb); **cat** via
`train.py`/`next_cv.py` (the tool that produced the T1.1 cat ceiling — p1's trainer lands cat
macro-F1 ~16pp low). Phase-1 search AL+FL → winners validated AZ+GE. Drivers `t14_sweep.sh`,
`t14_validate_azge.sh`, `t14_hgi_hardening.sh`; agg `t14_agg.py`.

### Winners (scale-robust, single config per task — no per-state branching)
- **reg = `next_stan_flow`, α=0 (log_T prior OFF), default HP, no tail-loss.**
- **cat = `next_gru`, logit-adjustment τ=0.5 (Menon ICLR'21), no class-weighting / focal / ls.**

**Caveats (advisor 2026-06-03):** (1) (c)-reg is the α=0 (prior-OFF) STL best, but the deployed v12 MTL
reg runs log_T-KD ON — so (c)-reg and the MTL reg it is compared against differ in prior treatment. This
is correct for a ceiling ("best achievable STL reg"), but the "fraction of (d) recovered" gate must NOT be
misread as prior-matched. (2) The cat balanced baseline reproduces the T1.1 ceiling **within 0.26pp**
(38.87 vs 39.13, within-σ non-determinism) — not bit-exact; the winner (41.86) is +2.7pp clear regardless.

### ⚠ CAT CEILING BUG + RE-PIN (2026-06-03) — read before trusting any cat number
`train.py --cat-head X` is **silently ignored on `--task next`** (it only takes effect on the MTL
`is_check2hgi_track` path). For plain `--task next` the model is `config.model_name`, default
`next_single`. So the ENTIRE cat ceiling (T1.1 + the first T1.4 freeze) ran **next_single, not
next_gru** despite the `--cat-head next_gru` flag. Caught by the Tier-S S.2 screen (all 8 encoders
returned an identical 41.86 → arch dump showed `NextHeadSingle`). **Tell-tale that was visible all
along:** the mis-pinned AL cat "ceiling" 41.86 was BELOW the MTL deployable cat 46.50 — impossible
for a real STL ceiling. Fix: `--cat-head`→`--model` in the drivers; `t14_cat_repin.sh` re-ran the cat
ceiling with the actual next_gru. Loss winner (logit-adjust τ=0.5) holds on the corrected model.
Correction is scale-dependent: **AL +8.11 (49.97 vs 41.86), AZ +0.57, GE −1.45, FL −0.02.** Reg is
UNAFFECTED (p1 honours `--heads`; per-config reg numbers differ). `next_single` beats next_gru at GE
(59.57 vs 58.12) → an S.2 candidate (Tier S), not the (c) head.

### (c) STL-on-v14 ceiling — FROZEN (cat = real next_gru; Δ_cat = logit-adjust lift over next_gru balanced)
| state | (c) cat macro-F1 | (c) reg Acc@10 |
|---|---|---|
| AL | 49.97 (+1.88) | 62.88 (+0.56) |
| AZ | 51.01 (+2.79) | 55.11 (+2.24) |
| GE | 58.12 (+4.01) | 58.45 (+2.64) |
| FL | 69.97 (+2.69) | 73.31 (+3.03) |
Sanity restored: (c)-cat > MTL deployable cat at every state (AL 49.97>46.50, AZ 51.01>48.52,
GE 58.12>56.13, FL 69.97>66.73). Verified by `t14_freeze_sanity.py` (arch=NextHeadGRU all states +
ceiling≥bounded-MTL asserts).
**Footnote (re-advisor 2026-06-03):** the ceiling bounds the *deployable* MTL cat (the claim above).
Against the MTL **diagnostic-best** cat (oracle per-task epoch, multi-seed {0,1,7,100}), (c)-cat sits
+3.19/+2.26/+1.05 above at AL/AZ/GE but **−0.29 BELOW at FL** (69.97 vs 70.26). This is NOT a bug
recurrence (arch confirmed NextHeadGRU); it is a seed/metric confound — (c) is seed42 single-seed +
the deployable basis, MTL-diag is multi-seed + an oracle epoch — and the gap is ~0.34σ (within fold
σ=0.86), a tie. Valid as a ceiling vs the *deployable* number; optionally re-run FL (c)-cat multi-seed
to confirm the inversion is a single-seed artifact if FL cat headroom matters at later tiers.

### (d) composite deploy ceiling — FROZEN (both arms α=0-hardened; cat = real next_gru)
reg arm = max(v14-α0, HGI-α0) per state; cat arm = (c)-cat-v14.
| state | (d) cat | (d) reg (source) | MTL deploy reg | gap over MTL |
|---|---|---|---|---|
| AL | 49.97 | 63.58 (HGI-α0) | 50.14 | +13.44 |
| AZ | 51.01 | 55.11 (v14-α0) | 37.78 | +17.33 |
| GE | 58.12 | 58.76 (HGI-α0) | 42.64 | +16.12 |
| FL | 69.97 | 73.62 (HGI-α0) | 61.21 | +12.41 |

### Key reg sweep evidence (AL / FL, Acc@10)
default-prior 62.32 / 70.28 · **α=0 62.88 / 73.31** · α0+LDAM(0.5) 61.11 / 72.92 · α0+CB 52.02 / — ·
α0+d_model256 62.55 / 73.02 · α0+dropout0.1 61.76 / 72.01 · α0+lr1e-3 60.24 / — · prior+LDAM 53.96 / —.
→ α=0 plain best at every state; tail-loss/dropout/LR lose decisively (CB cratered to 52). **d_model=256
is a statistical TIE, not a loss** (top AL arm 62.55±4.43 ≈ base 62.88, ~0.07σ; ~0.5σ behind at FL) —
base retained on parsimony (advisor 2026-06-03). The frozen ceiling is unaffected.

### Key cat sweep evidence (AL / FL, macro-F1)
balanced(=T1.1 ceiling) 38.87 / 65.79 · **logit-adjust τ0.5 41.86 / 69.99** · τ1.0 40.04 / 67.48 ·
τ1.0+ls05 39.74 / 67.49 · τ1.0+balanced 30.15 / — · focal+τ+ls+bal(combo) 24.34 / — · bal+ls05 38.94 / —.
→ logit-adjust τ=0.5 ALONE best at every state (AL 41.86 / AZ 50.44 / GE 59.57 / FL 69.99); stacking
with class-weighting or focal over-corrects and craters.

### HGI hardening (the (d)-arm consistency footnote) — STL-HGI-reg α=0, Acc@10
| state | v14-α0 | HGI-α0 | winner (within σ) |
|---|---|---|---|
| AL | 62.88 | 63.58 | HGI +0.70 |
| AZ | 55.11 | 54.87 | v14 +0.24 |
| GE | 58.45 | 58.76 | HGI +0.31 |
| FL | 73.31 | 73.62 | HGI +0.31 |
**Finding (advisor-sharpened 2026-06-03):** once both substrates are α=0-hardened, v14 and HGI STL-reg
are **statistically indistinguishable** at all four states — every Δ ≤ 0.70pp and < 0.5σ pooled
(AL/AZ/GE ~0.1σ; FL ~0.5σ); v14 even edges AZ. NOT a "marginal surviving edge" — a tie. HGI's apparent
reg-substrate advantage was an **artifact of the (now-dropped) log_T prior**. (d) keeps the per-state max
as a conservative ceiling, but the two-substrate composite is no longer reg-motivated — this *strengthens*
the regime thesis (substrate axis exhausted; the gap is architectural). Paper-relevant.

**(c)/(d) are FROZEN** as the immutable track-internal yardstick: T2-T5 Δ are measured against them;
Tier S / T5 may not re-open them (the §0.1 paper baseline stays refreshable at T6.2). Loss code +
all class stats are train-only (leak-guarded); cat ceiling moved a lot from one lever (logit-adjust) —
a tuning win on the incumbent head, NOT a new architecture (that is Tier S).

JSON pointers — reg α=0: `region_head_{AL,AZ,FL}_..._t13po_cfg1_raw_v14_s42.json` + `..._ge_r_a0.json`;
HGI α=0: `..._{al,az,ge,fl}_stlreg_hgi_a0.json`; reg HP probes `..._r_a0_*.json`; cat la05 rundirs in
`/tmp/t14/manifest_*` → `results/check2hgi_design_k_resln_mae_l0_1/{state}/next_*ep50*/summary/full_summary.json`.

---

## Tier S Prong-A — coded-head unit screen (concurrent; promotion HELD until floor frozen)
`tierS_unit_screen.py` (hard rule 16c "coded ≠ working"): **15/17** registered `next_*` heads pass
build+forward+backward on a synthetic batch. The 2 flagged (`next_getnext_hard_hsm`,
`next_stan_flow_hsm`) need a prebuilt `hierarchy_path` artifact (`build_region_hierarchy.py`) —
deferrable, NOT bit-rot. No bit-rot in the core candidate set → GPU screens may proceed; promotion
scored against the now-frozen (c)/(d).

---

## Advisor pass (2026-06-03) — 4 findings, all applied
P1 (HIGH) T1.3 over-read → prior-off re-run + scope downgrade. P2 (MED) gradient mixes fresh/frozen
+ recipe → sign-pattern-only. P3 (LOW) "conclusive" → "directionally clear" (single-seed fold-SD).
P4 (LOW) driver manifest path says `embedding_eval/` but p1 writes `docs/results/P1/` (numbers fine).

## Audit close-out (O1–O5) — 2026-06-04

Closing the 5 open items from [`../../studies/mtl_improvement/AUDIT_TIER1_TIERS_2026-06-03.md §6`](../../studies/mtl_improvement/archaive/AUDIT_TIER1_TIERS_2026-06-03.md).

### O1 — the α=0 "prior is a drag" finding (SETTLED — both audit hypotheses falsified)
Re-ran the prior-ON STL-reg config (`next_stan_flow`, **learnable** α init 0.1, prior ON, v14 region-emb,
frozen folds, seeded log_T, 5f×50ep seed42) and read the **converged learned α** per fold (the model was
never persisted, so this is a faithful re-run via `o1_alpha_probe.py`, which monkeypatches p1's `_build_head`
to capture the trained head). The replication is exact — prior-ON reproduces the documented Acc@10 at every
state — so the α reading is trustworthy.

| state | learned α (mean±sd) | prior-ON Acc@10 (replication) | α=0 ceiling (prior-OFF) | Δ (α0 − priorON) | standalone log_T Acc@10 | Markov-1-region ref |
|---|---|---|---|---|---|---|
| AL | **+0.454** ± 0.001 | 62.32 ✓ | 62.88 | +0.56 | 50.86 | 47.01 |
| AZ | **+0.789** ± 0.002 | 52.87 ✓ | 55.11 | +2.24 | 44.11 | — |
| GE | **+0.944** ± 0.005 | 55.81 ✓ | 58.45 | +2.64 | 49.29 | — |
| FL | **+1.095** ± 0.003 | 70.28 ✓ | 73.31 | +3.03 | 66.15 | 65.05 |

**Both audit hypotheses are FALSIFIED.** α did **not** converge ≈0 (the "unexplained optimization artifact"
hypothesis) and did **not** stay ≈0.1 (the "model didn't drop the prior" hypothesis). Instead **α converges to a
large positive value (0.45 → 1.09)** — the model actively *leans into* the prior. Yet the prior-ON ceiling is
**0.56–3.03 pp WORSE than α=0 at every state**. (α is **larger at the larger / higher-coverage states**: AL 0.45
< AZ 0.79 < GE 0.94 < FL 1.09, and the Δ tracks the same way — but this is n=4 noisy points, *suggestive, not a
fitted trend*, and is NOT load-bearing for the finding. Note also: the captured α is the **final-epoch (50)**
value while prior-ON Acc@10 is the best-epoch snapshot — immaterial given σ_α ≈ 0.001–0.005, but they are not
read at the same epoch.)

The standalone log_T prior (prior alone, no encoder) scores 44–66 % Acc@10 and **validates against the
authoritative Markov-1-region floors** (AL 50.86 ≈ 47.01; FL 66.15 ≈ 65.05 — `docs/AGENT_CONTEXT.md` §184-185;
the legacy "~21.3 %" is the *degenerate POI-level* markov, not comparable). So the prior carries real signal —
**"α=0 wins" is NOT "transition priors are worthless."**

**Phenomenology — the drag is the finding; the mechanism is NOT isolated (advisor 2026-06-04).** The probe
establishes two facts cleanly: a learnable α converges large (and the head leans hard into the prior), yet
prior-ON generalizes 0.56–3.03 pp *worse* than the prior-free (α=0) encoder at every state. **The fixed additive
log_T prior is therefore a net drag on the STL-reg ceiling.** The *most likely* cause is that the **train-only**
per-fold prior generalizes worse than transitions the encoder is forced to internalize (a train/val gap) — but
the probe does **not** discriminate this from two observationally-equivalent alternatives: **additive
scale-mismatch** (a large α mis-scales already-well-scaled encoder logits) or **double-counting** (STAN already
models transitions; an additive Markov bias is redundant). State the drag as *observed*, the train/val-gap as
*most-likely-but-not-proven*. This is coherent with the MTL-vs-STL split (audit §2d): in MTL log_T enters as a
**KD loss on the shared representation** (helps a starved backbone); in STL as an **additive logit bias** (hurts
a head that already fits transitions). **Reframed claim: "the fixed additive log_T prior is a net drag on the STL
reg ceiling."** Do NOT write "embeddings subsume transitions" (the prior is a strong standalone floor), nor "α
stuck at its optimum" (it converged large), nor assert the co-adaptation mechanism as proven (it is one of three
candidate causes). The §2c corollary stands and is strengthened: HGI's apparent reg edge was this same
additive-prior artifact — with both substrates α=0-hardened, v14 ≈ HGI (every Δ < 0.5σ). JSON:
`o1_alpha_probe.json`; driver `scripts/mtl_improvement/o1_alpha_probe.py`. **Leak audit (advisor): NONE** —
`last_region_idx` is the last *input* region (not the target), per-fold log_T is train-only, and the probe's
SGKF(seed42) split matches the log_T builder's split bit-for-bit.

### O4 — Tier-S reporting gaps closed
- **`next_hybrid` accounted** (it ran, was a reporting omission not a dropped arm): AL cat macro-F1 **49.34** with
  logit-adjust τ=0.5 — below the re-pinned next_gru floor (49.97), in the recurrent-family ~tie cluster with
  next_lstm (49.76). All 8 S.2 cat encoders lose-to-or-tie the floor → the negative is unchanged. (INDEX §S.2.)
- **`*_hsm` deferral noted**: the 2 hierarchical-softmax reg heads (`next_getnext_hard_hsm`, `next_stan_flow_hsm`)
  were never GPU-screened — they need a prebuilt `hierarchy_path` (`scripts/build_region_hierarchy.py`); a
  deliberate prerequisite deferral, NOT bit-rot. (INDEX §S.1.)

### O5 — paper limitations note carried in
Added limitation **(vi)** to `articles/[BRACIS]_Beyond_Cross_Task/PAPER_DRAFT.md` §7 Beat 3: non-overlapping
length-9 windows under-supervise (≈8× fewer history→next pairs); internal Δs are apples-to-apples on both arms;
the AL dense-supervision probe (MTL→STL reg gap *widens* 8.34→12.96, not closes) pre-empts the "your reg gap is
just under-supervision" reviewer attack; the dense rebuild is deferred to the `docs/future_works/overlapping_windows.md`
follow-up study (user decision, audit §6). NOT shipped silently.

### O2 — multi-seed cat (next_lstm / next_single) at AZ+GE — CLOSED 2026-06-04
Multi-seed {0,1,7,100}, 5f×50ep, `train.py --task next --model X --logit-adjust-tau 0.5` (the cat-floor tool),
vs the frozen (c)-cat next_gru floor. AZ+GE are the states where each head nominally won single-seed.

| head | state | seeds (macro-F1) | mean±sd | (c) floor | Δ | verdict |
|---|---|---|---|---|---|---|
| next_lstm | AZ | 51.3/51.4/51.0/51.4 | 51.26 ± 0.19 | 51.01 | +0.25 | tie |
| next_lstm | GE | 58.1/58.0/58.3/58.7 | 58.30 ± 0.31 | 58.12 | +0.18 | tie |
| next_single | AZ | 50.8/50.7/51.0/51.4 | 50.98 ± 0.29 | 51.01 | −0.03 | tie |
| next_single | GE | 59.8/59.5/59.6/59.8 | **59.66 ± 0.17** | 58.12 | **+1.54** | **clears ≥0.5pp (GE-specific)** |

**next_lstm: the single-seed nominal wins do NOT survive multi-seed.** AZ +0.48→+0.25, GE +0.51→+0.18 — both
collapse to ties. With AL −0.21 / FL +0.14 (single-seed, `tierS_confirm.sh`), next_lstm is a **tie at all four
states** → no win anywhere. The Tier-S crack ("failed to demonstrate a win") is closed → **"shown no win"**.

**next_single: a robust but GE-SPECIFIC win.** Its GE win is real and tightens multi-seed (+1.45 single →
**+1.54 ± 0.17** multi-seed) — NOT a single-seed artifact. But it is state-conditional (AL −8.11, AZ −0.03), so
it **fails the ≥2-band promotion gate** (clears only middle) → it does **NOT** change the scale-robust frozen (c)
head (next_gru). Per the audit's narrow framing ("clears ≥0.5pp multi-seed → a real T5.2 candidate") it **enters
the T5.2 candidate set as a state-conditional option**, re-judged under MTL at Tier 5. **Does NOT re-open frozen
(c)** (moving-baseline guard, hard-rule 15): (c) GE-cat stays next_gru 58.12; next_single 59.66 is a logged
candidate, not a re-pin. **Stated plainly (advisor 2026-06-04, no "hidden ceiling"):** next_single's 59.66 *is* a
higher STL GE-cat number than (c)'s 58.12 — so the **per-state** STL GE-cat ceiling is 59.66; (c) is the
**scale-robust incumbent** (single config, no per-state branching), not the per-state STL maximum, and next_single
is retained as a GE-only candidate precisely because its edge collapses/inverts elsewhere (−8.11 at AL). **Net:
the multi-band Tier-S cat negative HOLDS** — no scale-robust head beats the incumbent; the one validated
state-specific candidate is logged, not shipped.

### O3 — multi-seed FL (c)-cat — CLOSED 2026-06-04
FL next_gru, logit-adjust τ=0.5, seeds {0,1,7,100}: **69.96 ± 0.08** (70.0/69.9/69.9/70.0). This **validates the
seed-42 frozen (c)-cat ceiling 69.97** — they agree to 0.01pp, σ=0.08 (so the single-seed yardstick is
representative at FL). **The inversion vs the MTL diagnostic-best (70.26) PERSISTS multi-seed** (−0.30pp) — so it
is **NOT** the single-seed artifact the audit hypothesised. It is tiny (~0.35σ at fold-σ≈0.86) and explained:
(c) is the STL ceiling on the *deployable* basis and correctly bounds the *deployable* MTL cat (69.96 ≫ 66.73);
the MTL *diagnostic-best* is an oracle per-task epoch plus a small positive cat transfer at FL (the board shows
FL Δcat≈0, i.e. MTL cat ≈ STL cat at FL). Not a bug recurrence — arch confirmed NextHeadGRU, freeze-sanity hard
checks pass. Honest caveat retained in the (c) footnote; the (c) ceiling is valid vs the deployable number.

JSONs: O2 rundirs in `/tmp/o2o3/manifest_o2.tsv` → `results/check2hgi_design_k_resln_mae_l0_1/{arizona,georgia}/next_*ep50*`;
O3 in `manifest_o3.tsv` → `.../florida/next_*ep50*`. Aggregator `o2o3_agg.py`. Compute: O2 16 runs + O3 4 runs ≈ 6 GPU-h.

## Scripts (all on branch `mtl-improve`)
`scripts/mtl_improvement/`: `freeze_folds.py` (T0.0 + drift-guard), `ge_board.sh` + `ge_board_agg.py`
(T0.2/T0.3), `t13_encoder_probe.sh` + `t13_prioroff.sh` + `t13_agg.py` (T1.3),
`build_ge_hgi_train.sh` + `t1_ceilings.sh` (T1.1/T1.2), `t14_sweep.sh` + `t14_validate_azge.sh` +
`t14_hgi_hardening.sh` + `t14_agg.py` (T1.4), `tierS_unit_screen.py` (Tier-S Prong-A),
`o1_alpha_probe.py` (O1 learned-α + standalone-prior probe), `o2o3_multiseed_cat.sh` + `o2o3_agg.py`
(O2/O3 multi-seed cat). Loss code:
`src/losses/calibrated.py` + `tests/test_losses/test_calibrated.py`; wiring in `next_cv.py` +
`ExperimentConfig.loss_calibration` + `train.py`/`p1_region_head_ablation.py` flags. GE onboarding:
`scripts/_v14_run/build_ge.sh`.
