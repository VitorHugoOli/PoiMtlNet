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

### (c) STL-on-v14 ceiling — FROZEN (Δ vs pre-T1.4 T1.1 in parens)
| state | (c) cat macro-F1 | (c) reg Acc@10 |
|---|---|---|
| AL | 41.86 (+2.73) | 62.88 (+0.56) |
| AZ | 50.44 (+7.28) | 55.11 (+2.24) |
| GE | 59.57 (+5.55) | 58.45 (+2.64) |
| FL | 69.99 (+4.11) | 73.31 (+3.03) |

### (d) composite deploy ceiling — FROZEN (both arms α=0-hardened)
reg arm = max(v14-α0, HGI-α0) per state; cat arm = (c)-cat-v14.
| state | (d) cat | (d) reg (source) | MTL deploy reg | gap over MTL |
|---|---|---|---|---|
| AL | 41.86 | 63.58 (HGI-α0) | 50.14 | +13.44 |
| AZ | 50.44 | 55.11 (v14-α0) | 37.78 | +17.33 |
| GE | 59.57 | 58.76 (HGI-α0) | 42.64 | +16.12 |
| FL | 69.99 | 73.62 (HGI-α0) | 61.21 | +12.41 |

### Key reg sweep evidence (AL / FL, Acc@10)
default-prior 62.32 / 70.28 · **α=0 62.88 / 73.31** · α0+LDAM(0.5) 61.11 / 72.92 · α0+CB 52.02 / — ·
α0+d_model256 62.55 / 73.02 · α0+dropout0.1 61.76 / 72.01 · α0+lr1e-3 60.24 / — · prior+LDAM 53.96 / —.
→ α=0 plain best at every state; all tail-loss/HP variants lose.

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
**Finding:** once both substrates are α=0-hardened, HGI's STL-reg edge collapses to ≤0.70pp (within
fold-σ); v14 ties at AZ. The two-substrate composite premise survives but is now marginal — HGI's
reg-substrate advantage was largely an artifact of the (now-dropped) log_T prior. Paper-relevant.

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

## Scripts (all on branch `mtl-improve`)
`scripts/mtl_improvement/`: `freeze_folds.py` (T0.0 + drift-guard), `ge_board.sh` + `ge_board_agg.py`
(T0.2/T0.3), `t13_encoder_probe.sh` + `t13_prioroff.sh` + `t13_agg.py` (T1.3),
`build_ge_hgi_train.sh` + `t1_ceilings.sh` (T1.1/T1.2), `t14_sweep.sh` + `t14_validate_azge.sh` +
`t14_hgi_hardening.sh` + `t14_agg.py` (T1.4), `tierS_unit_screen.py` (Tier-S Prong-A). Loss code:
`src/losses/calibrated.py` + `tests/test_losses/test_calibrated.py`; wiring in `next_cv.py` +
`ExperimentConfig.loss_calibration` + `train.py`/`p1_region_head_ablation.py` flags. GE onboarding:
`scripts/_v14_run/build_ge.sh`.
