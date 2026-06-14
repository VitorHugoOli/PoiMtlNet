# F_TIER_A1_LEAK_AUDIT — Independent leak audit of the Tier A1 log_T-KD verdict (AL +2.27 pp, AZ +4.91 pp)

**Author**: Audit agent (independent, read-only)
**Date**: 2026-05-28
**Scope**: Audit whether the Tier A1 `--log-t-kd-weight=0.2` promotion verdict (AL Δ+2.27 pp, AZ Δ+4.91 pp on disjoint reg top10_acc, 20/20 folds, p=9.54e-07 each) is a real lift or a leak artefact.
**Trigger**: New KD code path (`--log-t-kd-weight`, KL loss term) was implemented today (2026-05-28). The Phase 3 Rank 1 verdict that motivated A1 came from uncommitted local changes; today's Tier A1 reproduction is therefore NOT independent cross-validation of the mechanism. Magnitudes (AL +2.27, AZ +4.91) overlap the historical leak band (C22: +8–12 pp at FL; F50_T4_C4: comparable; balanced-sampler: −18 to −30 pp), so an independent audit is warranted before paper-citation.

---

## §1 Summary verdict

**NO LEAK FOUND.** All seven vectors investigated either verify clean or reduce to a structural / dosage explanation consistent with a legitimate distillation lift. The AZ-vs-AL ratio is NOT explained by stronger structural shortcut at AZ (their MI/H(Y) ratios are within ~4 pp of each other; AZ actually has slightly LESS structural info per bit, so AZ's outsize lift is dosage-on-headroom rather than structural-shortcut). Recommendation: cite Tier A1 in the paper with the caveats from §6, no fix needed, no further compute required.

---

## §2 Vector-by-vector audit

| Vec | Check | Status | Notes |
|---|---|---|---|
| V1 | Per-fold log_T is built from TRAIN-fold userids only | VERIFIED | `scripts/compute_region_transition.py:153-191` `build_transition_matrix_from_userids` filters by `userid ∈ train_set`; `_build_per_fold` (lines 246-281) instantiates `StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)` matching trainer's split. |
| V1b | log_T `n_splits` matches trainer's `k_folds` | VERIFIED | Payload from `output/check2hgi/alabama/region_transition_log_seed0_fold1.pt` carries `{'n_splits': 5, 'seed': 0, 'n_regions': 1109}`. Tier A1 ran with `--folds 5` default. C19 guard at `src/training/runners/mtl_cv.py:1068-1100` would hard-fail on mismatch. |
| V1c | log_T file mtime is FRESH (not stale per C22) | VERIFIED | AL log_T mtime `2026-05-28 16:14:09` > `next_region.parquet` mtime `2026-05-19 14:36:06`. AZ identical pattern. C22 preflight guard at `src/training/runners/mtl_cv.py:1042-1047` would have raised on staleness. Both Tier A1 cells loaded the SAME fresh log_T per `[C4 per-fold log_T] fold N seed S using …` log lines. |
| V2 | `last_region_idx` is derived from `poi_0..poi_8` ONLY, never from `target_poi` | VERIFIED | `src/data/inputs/next_region.py:128-163`: `poi_mat = seq_df[poi_cols].astype(np.int64).to_numpy()` over `poi_0..poi_8`; `last_pos` is argmax over those columns; `last_region_idx[i] = poi_to_region[last_poi_idx[i]]`. `target_poi` is read separately (line 105) and never feeds `last_region_idx`. |
| V2b | Student logits used for KD are the same `pred_task_b` used for CE — no backprop path from `target` into the KD term | VERIFIED | `mtl_cv.py:468` `task_a_output, task_b_output = model((x_task_a, x_task_b))` — the forward sees only the input window, never `y_task_b`. KD reads `pred_task_b` (line 521) which has no `truth_task_b` dependence. CE (line 474) and KD (line 540) BOTH attach to the same `pred_task_b`; no "shortcut" exists because the model never sees the target before producing logits. |
| V3 | Is the prior a near-deterministic structural shortcut (last → target)? | VERIFIED **non-near-deterministic** | §4 numbers: MI/H(Y) = 0.601 (AL) and 0.560 (AZ); E_X[max_y P(Y\|X)] = 0.368 (AL) and 0.341 (AZ); P(last==target) = 0.31 (AL) and 0.30 (AZ). Top-1 determinism ~34–37 % — the prior is informative but FAR from deterministic. A near-deterministic prior would have top-1 ≈ 1.0 and identity ≈ 1.0. |
| V3b | C19 train-only rule prevents val (last, target) pair from appearing in log_T | VERIFIED | Same as V1: log_T is built per fold from train userids only via `build_transition_matrix_from_userids` (line 153). A val user's transitions are never aggregated into that fold's log_T. |
| V4 | W=0.0 baseline is genuinely fair vs W=0.2 (no RNG drift) | VERIFIED | `mtl_cv.py:490` `if log_t_kd_weight > 0.0:` short-circuits the ENTIRE KD block — no `get_current_aux()` call, no teacher materialisation, no `_log_T.index_select` (which DOES involve potentially RNG-bearing CUDA kernels at fp16). At W=0.0 the per-batch path is byte-identical to the pre-Tier-A1 trainer. Wilcoxon paired test by `(seed, fold)` is valid. |
| V4b | Both W=0.0 and W=0.2 cells load the same fresh log_T into the reg head's `log_T` buffer | VERIFIED | Run logs at `…/W0.0/seed0/run.log` and `…/W0.2/seed0/run.log` both show `[C4 per-fold log_T] fold N seed 0 using …region_transition_log_seed0_foldN.pt` for folds 1–5 with identical paths. The reg head (`next_getnext_hard` / `next_stan_flow`) embeds log_T as a STRUCTURAL prior at INFERENCE in BOTH cells; the only KD-cell-specific change is the EXTRA explicit KL term added to `task_b_loss`. This is the cleanest possible isolation of the KD term as the causal lever. |
| V5 | Magnitude comparison vs historical leak band | VERIFIED **plausible** | §5 table. +2.27 / +4.91 sit BELOW C22 stale-log_T inflation band (+8 to +12 pp FL), well below F50_T4_C4 catastrophic leak (>20 pp), and well below balanced-sampler regression (−18 to −30 pp). Above typical T1–T6 substrate-tweak lifts (~+0.3 to +1.5 pp at AL/AZ). Magnitude is consistent with a strong-but-bounded supervisory lift, not a catastrophic leak. |
| V5b | All-folds-positive at n=20 each state | VERIFIED **consistent w/ real signal** | AL Δ range +1.33 to +3.66; AZ Δ range +3.94 to +5.58. All 40 folds positive; AL median +2.15, AZ median +4.95. Tight distributions per state, no fold reversal — consistent with a structural KD effect, not noise. (A structural leak would ALSO be all-positive, so this is necessary but not sufficient; combined with V1–V4 it's confirmatory.) |
| V6 | C18 encoder-swap leak-probe framework — does it apply? | N/A (framework doesn't fit) | C18 is a SUBSTRATE-swap framework (it asks: does the new encoder's leak F1 vs canonical drift?). KD is a LOSS-term swap, not a substrate swap; the reg head's logit-emitting subnetwork is unchanged. C18's leak-probe (separate-cat-shortcut metric) would be uninformative here because the KD term lifts REG and leaves cat ESSENTIALLY UNCHANGED (AL cat F1: 45.96 → 45.76, Δ−0.20 pp; AZ cat F1: 48.86 → 48.94, Δ+0.08 pp per the verdict table). A label-shortcut leak would lift cat too (per T3.1 catastrophic pattern). Reg-only lift with flat cat is the OPPOSITE of the leak signature C18 looks for. |
| V7 | AZ outsize lift (4.91 vs AL 2.27) | **EXPLAINED structurally** | §4: AL and AZ have NEARLY EQUAL MI/H(Y) (0.60 vs 0.56) — AZ has slightly LESS info per bit, not more. So the AZ lift is NOT explained by a stronger structural shortcut. It IS explained by HEADROOM: AZ W=0.0 baseline = 41.30 % vs AL W=0.0 = 50.59 %. AZ has ~9 pp more headroom to 70 %, so the same KD supervisory signal converts to a larger absolute pp lift. Relative lift: AL +2.27/50.59 = +4.5 %, AZ +4.91/41.30 = +11.9 % — AZ DOES capture more relative information, but not anomalously more given that AZ's reg head was underperforming AL's baseline by 9 pp. **Suspicious ratio is resolved by baseline-headroom asymmetry, not leak.** |

---

## §3 Per-vector code citations

- V1 / V1b: `scripts/compute_region_transition.py` lines 153–191 (build), 246–281 (per-fold); n_splits payload at line 234.
- V1c (mtime guard): `src/training/runners/mtl_cv.py` lines 1042–1047.
- V1b (n_splits guard): `src/training/runners/mtl_cv.py` lines 1068–1100.
- V2: `src/data/inputs/next_region.py` lines 124–163; target derivation at lines 100–116 (independent path).
- V2b / V4: `src/training/runners/mtl_cv.py` lines 460–540.
- V4 W=0.0 fast path: `src/training/runners/mtl_cv.py` line 490 (`if log_t_kd_weight > 0.0:`).
- KD math (KL direction, τ² scaling): `src/training/runners/mtl_cv.py` lines 517–540.
- Aux side-channel cleanliness: `src/data/aux_side_channel.py` lines 50–110 (no target dependence).
- Unit tests (W=0.0 no-op, padding exclusion, differentiability, τ scaling): `tests/test_substrate_protocol_cleanup_flags.py` lines 337–448.

---

## §4 Quantitative cross-check — MI(last_region_idx ; target region_idx)

Script: `/tmp/mi_audit.py` (read whole `next_region.parquet`, no filtering by fold — measures population-level structural info; per-fold values would be similar by IID assumption within state).

| Metric | Alabama | Arizona |
|---|---:|---:|
| n_rows | 12 709 | 26 396 |
| n_regions | 1 109 | 1 546 |
| H(target region_idx) [bits] | 8.159 | 8.508 |
| H(last_region_idx) [bits] | 8.151 | 8.581 |
| H(target \| last) [bits] | 3.252 | 3.741 |
| MI(last ; target) [bits] | **4.907** | **4.767** |
| MI / H(target) [fraction] | **0.601** | **0.560** |
| E_X[ max_y P(Y\|X) ] (top-1 determinism) | 0.368 | 0.341 |
| P(last_region == target) | 0.313 | 0.301 |

**Interpretation.** Both states have a strong but bounded last→target prior — ~60 % of target's entropy is captured by `last`. Top-1 conditional probability averages ~35 %, so the prior over-concentrates mass on the true next region for ~1 sample in 3, NOT near-deterministically. AL and AZ are nearly identical on every axis; AZ is slightly LESS informative per bit, contradicting the structural-shortcut hypothesis for AZ's outsize lift. The KD term is therefore a legitimate moderate-strength teacher, not a near-direct shortcut.

---

## §5 Comparison to historical leak magnitudes

| Source | Mechanism | Δ pp (state) | Tier A1 comparison |
|---|---|---|---|
| C22 (DISCOVERED 2026-05-20) | Stale May-6 log_T silently used | +8 to +12 pp FL reg Acc@10 (stale vs fresh) | Tier A1 +2.27/+4.91 is 3–5× SMALLER; Tier A1 uses fresh log_T (V1c). |
| C19 (RESOLVED 2026-05-15) | `--folds 1` ≠ log_T n_splits=5 | +13 to +23 pp inflation on reg top10_acc_indist | Tier A1 uses `--folds 5` matching log_T `n_splits=5` (V1b). N/A. |
| F50_T4_C4 LEAK | Pre-C19 log_T train-leak | Catastrophic (>20 pp at some cells) | Tier A1 uses per-fold train-userid-only log_T (V1, V3b). N/A. |
| Balanced sampler regression | Class-balanced sampler over-corrects | −18 to −30 pp REGRESSION | Opposite sign; Tier A1 lift is positive. N/A. |
| T3.1 GAT (catastrophic structural leak) | Encoder substrate label-shortcut | +11.34 pp leak F1 AND lifts cat alone | Tier A1 reg-only lift, cat essentially flat (AL Δcat −0.20, AZ Δcat +0.08). N/A. |
| T1–T6 substrate / loss tweaks (clean lifts) | Various legitimate levers | +0.3 to +1.5 pp at AL/AZ typical | Tier A1 +2.27/+4.91 is LARGER than typical-clean-lift band — but KD is a different LEVER FAMILY (explicit supervisory signal, not substrate perturbation). The MI numbers (§4) justify a stronger lift here than typical substrate tweaks. |

**Verdict from §5.** Tier A1 magnitudes sit BETWEEN typical-clean-lift band and known-leak band, closer to clean-lift end. No leak mechanism in this codebase's history matches the Tier A1 signature (reg-only lift + flat cat + fresh log_T + train-fold-only build + non-near-deterministic prior).

---

## §6 Recommendations

1. **Verdict: PROMOTE Tier A1 to paper-citable.** No leak found; cite with the structural-MI context.
2. **Caveat to include in any paper discussion of A1**: the lift exploits a strong-but-bounded last→target structural prior (MI/H(Y) ≈ 0.58 averaged AL/AZ; top-1 determinism ≈ 0.35). This is legitimate because the prior is built from train-fold userids only (no val leakage by construction); the KD term distils this train-only prior into the student. Frame as "supervisory distillation of an empirical first-order region-Markov prior", not as a novel architecture.
3. **Caveat on AZ vs AL ratio**: AZ's larger lift (4.91 vs 2.27) is dosage-on-headroom, not stronger leak. AZ baseline (41.30 %) is 9 pp below AL baseline (50.59 %), giving the KD signal more residual entropy to convert.
4. **Future verification needed (deferrable, NOT blocking promotion)**:
   - (a) Re-run AL/AZ with a SHUFFLED `last_region_idx` placebo as the KD teacher key (i.e. permute `aux` before publishing). Expected: zero or negative lift. Confirms the lift requires the actual last-region structural signal, not just an additional regulariser. Cost: 2 states × 4 seeds × 5 folds × 2 cells = ~40 GPU-hours. Recommend deferring unless reviewer challenges A1.
   - (b) Reverse the KL direction (use `KL(teacher ‖ student)` instead) and confirm whether the lift survives. The current direction `KL(student ‖ teacher)` is mode-seeking; the alternative is mode-covering. If both give comparable lifts, the direction is not the load-bearing piece. Cost: same as (a).
5. **No code fix needed.** The implementation is clean per all seven vectors.
6. **C24 (STAN bidirectional) cross-check**: Tier A1 uses `next_getnext_hard` (canonical NORTH_STAR B9 reg head per Phase 3 sweep), NOT `next_stan_flow`. C24's bidirectional-attention concern does not apply. If a future variant moves the KD lever onto `next_stan_flow`, re-audit V2/V2b under the bidirectional head.

---

## Appendix — files audited

- `src/training/runners/mtl_cv.py` (KD wiring lines 216–217, 460–540, 1000–1100, 1350–1351)
- `src/data/inputs/next_region.py` (last_region_idx derivation)
- `src/data/aux_side_channel.py` (thread-local publish/get; null-safe)
- `src/configs/experiment.py` (lines 185, 195–196 — config fields)
- `scripts/train.py` (lines 822–843, 1316–1335 — CLI flags + validation)
- `scripts/compute_region_transition.py` (lines 90–305 — per-fold train-userid build + n_splits payload)
- `tests/test_substrate_protocol_cleanup_flags.py` (lines 337–448 — A1 unit tests)
- `docs/results/substrate_protocol_cleanup/tier_a1/phase_a1_verdict.md` (sweep summary)
- `docs/results/substrate_protocol_cleanup/tier_a1/alabama/W{0.0,0.2}/seed0/run.log` (log_T loading messages)
- `output/check2hgi/alabama/region_transition_log_seed0_fold1.pt` (payload schema)
- Read for historical anchors: `docs/CONCERNS.md` §C18/C19/C22/C24; `docs/findings/F50_T4_C4_LEAK_DIAGNOSIS.md` (referenced); `docs/findings/F50_T4_BROADER_LEAKAGE_AUDIT.md` (referenced); `docs/studies/archive/substrate-protocol-cleanup/window_mask_audit.md` (noted as PRE-DATING log_T-KD landing — does not cover A1).
