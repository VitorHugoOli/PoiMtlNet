# CHANGELOG — Check2HGI Study

> **Why this file exists.** During paper preparation we hit the multiple-sources-of-truth confusion trap several times: numbers diverging across PAPER_CLOSURE_RESULTS, FINAL_SURVEY, NORTH_STAR, RESULTS_TABLE, and intermediate handoffs. Two external Codex audits (commits `7a60e1c` → `ed90e8a` → `6de13ca` → `8a95e92` → `8444b31`) caught load-bearing errors that traced back to "which version of the table am I citing?". This CHANGELOG is the single chronological source of truth for *what changed when, why, and which numbers to trust now*.
>
> **Rules of use.**
> - When in doubt about a number, the canonical source is the **latest dated row** below pointing to `results/RESULTS_TABLE.md §0`.
> - Any document elsewhere in the repo that contradicts the latest row is **stale** and should either be archived (under `archive/`) or fixed in the same commit that introduces the contradiction.
> - Workflow / prompt artefacts go to `archive/` once the work they coordinated has landed; only canonical / current docs live at top level.
> - **Do not edit historic rows.** Add new dated rows; the historic record stays.

---

## Canonical sources of truth (current)

| What | Where | Last updated |
|---|---|---|
| Five-state architectural-Δ + cat-Δ Wilcoxon | `results/RESULTS_TABLE.md §0.1` | **v11, 2026-05-02** (FL upgraded to n=20; all five states paper-grade on §0.1) |
| Δm joint score (CH22 leak-free) | `results/RESULTS_TABLE.md §0.2` | v6 (leak-free) — unchanged in v7/v8/v9/v10 |
| Substrate axis (CH16 cat + CH15 reg reframing) | `results/RESULTS_TABLE.md §0.3` + `FINAL_SURVEY.md §2-§4` | v6 — unchanged in v7/v8/v9/v10 |
| Recipe selection (B9 vs H3-alt) | `results/RESULTS_TABLE.md §0.4` | **v9, 2026-05-02** (TX upgraded to n=20 multi-seed) |
| External baselines | `results/RESULTS_TABLE.md §0.5–§0.6` | unchanged |
| Champion config + recipe | `NORTH_STAR.md` | v10-aligned entry banner; historical derivation preserved below |
| Claim catalogue | `CLAIMS_AND_HYPOTHESES.md` (whitelist banner) | CH16 / CH18-cat / CH15 reframing / CH19 / CH22 are paper-facing safe |
| Article-side paper docs | `articles/[BRACIS]_Beyond_Cross_Task/` | v10-aligned 2026-05-02 (BRACIS submission) |

**Rule of single-source.** Paper tables are sourced from `RESULTS_TABLE.md §0` and only that. `PAPER_CLOSURE_RESULTS_2026-05-01.md` is background provenance — it has been moved to `archive/post_paper_closure_2026-05-01/` because numerous numbers there were superseded by later `RESULTS_TABLE` updates (e.g., FL Δ_reg simple mean-diff −7.28 vs paired Δ −7.99; AL/AZ STL cat means refreshed from single-seed to multi-seed; CA/TX §0.1 upgraded to n=20 in v10).

---

## Timeline of findings (most recent first)

### 2026-05-19 (final) — canonical_improvement Tier 6 CLOSED — full hypothesis falsified across all four pre-registered mechanism families

After the earlier 2026-05-19 entry below (T6.4 falsified + selector bug surfaced), the remaining Tier-6 candidates were run with locked pre-registered criteria. **Tier 6 closes operationally falsified across all four mechanism families.**

| ID | Mechanism | Per-task disjoint Δ_reg | Verdict |
|---|---|---:|---|
| T6.4 ×2 | Loss-shape reform (InfoNCE @ p2r, two-pass corruption) | +0.08 to +0.17 | FALSIFIED |
| T6.1 ×5 (4 orig + 1 robust) | POI↔POI co-visit InfoNCE 4th boundary | +0.05 to +0.20 | FALSIFIED |
| T6.2 ×4 | HGI-inspired composite edge weights (Delaunay + cross-region penalty) | +0.23 to +0.76 | §Discussion (cat -1.3 to -3.6 pp Pareto trade) |
| T6.3 ×2 (AL/AZ stage-1 only) | Low-rank per-POI bias at Checkin2POI attention-logit | halted | FALSIFIED at G3 hi/lo ratio gate (AZ r=8: 3.58× → 3.09×) |

**Per-task-disjoint reg-top10 ceiling at FL clusters at ~76.1-76.9 pp across all 11 cells with σ ~0.3 pp.** No Tier-6 intervention delivers a deployable single-checkpoint improvement under `joint_geom_simple`. The cleanest paper-claim phrasing: *"Tier 6 falsifies the POI-internal-supervision hypothesis under matched protocol at FL across all four pre-registered mechanism families. No cell delivers a deployable single-checkpoint improvement; per-task-disjoint reg-capacity stays bounded at ~76.1 pp under canonical static_weight w_cat=0.75 MTL balancing."*

**The load-bearing Tier-6 finding is NOT a substrate result.** It is `CONCERNS.md` C21 / `CLAIMS_AND_HYPOTHESES.md` CH23-B: the production `joint_canonical_b9` selector throws away ~10.7 pp of reg-top10 capacity from the canonical Check2HGI substrate itself. Substrate-axis effect: ±0.8 pp. Protocol-axis effect: +10.7 pp. The natural next study is the mtl-exploration F1/F2/F3 workstream.

**Pre-registration discipline held.** Two advisor consults (2026-05-19, both general-purpose independent agents) locked criteria before the T6.1/T6.2/T6.3 sweeps; in all three cases the locked criteria fired correctly and the closure narrative did not drift post-hoc. Specifically: (1) the T6.1 "+11-13 pp reg lift" claim from the morning of 2026-05-19 was a cross-selector comparison artefact corrected after matched-protocol analysis; (2) the T6.2 +0.76 pp per-task-disjoint reg lift was kept as §Discussion-only per advisor warning ("don't let it grow into a substrate claim"); (3) the T6.3 AL/AZ-first kill-check fired automatically on AZ r=8 hi/lo ratio compression, halting before FL stage 2.

**Artefacts**:
- T6.1: `docs/results/canonical_improvement/T6_1_lambda{0_05,0_1,0_2,0_3}/florida_mtl/` + `T6_1_dual_selector.{json,md}` + `T6_1_robustness_lambda0_2/florida_mtl/` + `T6_1_robustness_dual_selector.{json,md}`
- T6.2: `docs/results/canonical_improvement/T6_2_a{1_5_0_3,1_5_0_5,2_0_0_3,2_0_0_5}/florida_mtl/` + `T6_2_dual_selector.{json,md}`
- T6.3: `docs/results/canonical_improvement/T6_3_r{4,8}/{alabama,arizona}/` + `G3_{alabama,arizona}_T6_3_r{4,8}.json` (no FL — gate aborted)
- Code: `research/embeddings/check2hgi/model/Checkin2POI.py` (T6.3 attention-logit bias), `research/embeddings/check2hgi/model/Check2HGIModule.py` (T6.1/T6.4), `research/embeddings/check2hgi/preprocess.py` (T6.1 covisit_pairs, T6.2 composite C3), `scripts/canonical_improvement/t6{1,2,3}_sweep.sh` (per-mechanism sweep scripts)
- Closure log: `docs/studies/canonical_improvement/log.md` 2026-05-19 final entry (the authoritative one — supersedes the earlier same-day entries in framing while keeping their numerical results)

**Doc-correction sweep** (all updated 2026-05-19):
- `docs/studies/canonical_improvement/INDEX.html` Tier 6 — closure callout box + T6.2/T6.3 results blocks (rewrite, not append)
- `docs/CONCERNS.md` C21 — unchanged from earlier 2026-05-19 update (still load-bearing, scope confirmed)
- `docs/CLAIMS_AND_HYPOTHESES.md` CH23-A/B — extended to include T6.1, T6.2, T6.3 falsifications

**Closure path: mtl-exploration F1/F2/F3.** All canonical_improvement substrate work formally closed; future substrate interventions pre-route through C21 reading.

---

### 2026-05-19 — canonical_improvement Tier-6 / T6.4 FALSIFIED at matched protocol; `joint_canonical_b9` selector bug surfaces as the real finding

**Tier 6 was reopened 2026-05-18** to re-attempt the POI-level supervision hypothesis the user felt was under-explored in Tier 5. Built G3 (per-POI hold-out leak probe, calibrated against T5.1's known leak Δ_low = +3.82 pp), implemented **T6.4** (InfoNCE @ p2r + two-pass corruption) as opt-in default-off code paths in `Check2HGIModule.py`, swept the variants × {AL, AZ, FL} at ep=500, ran FL MTL under canonical B9 ep=50.

An initial ep=15 protocol-cap attempt was attacked by advisor consult #1 as post-hoc val-leak; advisor consult #2 supported a full-ep=50 dual-selector framing instead. A shipping FL ep=50 single-seed=42 n=5 baseline was added for matched-protocol comparison. The matched-protocol comparison then **falsified the Tier-6 substrate hypothesis** and surfaced a separate, more important finding about the production B9 selector itself.

**Matched-protocol dual-selector results (FL, single-seed=42, n=5 folds, ep=50):**

| Selector | shipping | T6.4 two_pass | T6.4 infonce τ=0.5 | Δ T6.4 vs shipping |
|---|---:|---:|---:|---|
| Per-task disjoint best: cat F1 | 70.49 ± 0.86 | 70.55 ± 0.85 | 70.49 ± 0.95 | **+0.00 to +0.06** |
| Per-task disjoint best: reg top10 | **76.12 ± 0.33** | 76.20 ± 0.27 | 76.29 ± 0.29 | **+0.08 to +0.17** |
| `joint_geom_simple`: cat F1 | 67.93 ± 1.74 | 67.33 ± 2.06 | 67.12 ± 2.45 | −0.60 to −0.81 |
| `joint_geom_simple`: reg top10 | 72.38 ± 2.20 | 73.33 ± 2.28 | 73.48 ± 2.48 | +0.95 to +1.10 |
| `joint_canonical_b9` (production): cat F1 | 69.99 ± 1.13 | 70.13 ± 1.06 | 70.28 ± 0.82 | +0.14 to +0.29 |
| `joint_canonical_b9` (production): reg top10 | 65.38 ± **9.10** | 61.19 ± **11.86** | 56.78 ± **11.79** | **−4.19 to −8.60** |

Reference: shipping FL §0.1 multi-seed n=20 reports reg top10 = 63.27 ± 0.10 — matches the matched-protocol `joint_canonical_b9` single-seed value (65.38 ± 9.10) within single-seed variance. §0.1 reports joint-best, not reg-best.

**Finding 1 — Tier-6 T6.4 substrate hypothesis FALSIFIED at matched protocol.** T6.4 variants add Δ_reg = +0.08-0.17 pp over shipping at per-task disjoint best — well within fold σ (~0.3) and not statistically meaningful at n=5. The InfoNCE-and-two-pass code paths land as opt-in infrastructure (default-off, byte-identical, useful for future studies that pair them with other interventions), but the variants alone are §Discussion-only and the paper claim for T6.4 is "falsified at matched protocol." The original "+11 pp reg lift" claim from 2026-05-19 was a cross-selector comparison artefact (T6.4 reg-best ep vs shipping §0.1 joint-best ep) — not a substrate effect. See `CLAIMS_AND_HYPOTHESES.md` CH23-A.

**Finding 2 — `joint_canonical_b9` selector throws away ~+11 pp of reg-top10 capacity from the canonical Check2HGI substrate itself.** Per-task disjoint best on shipping reaches reg top10 = 76.12; production selector reaches 65.38. Gap = ~10.7 pp on the shipping substrate, with no substrate change. The bug is **not Tier-6-specific** — it exists in the production B9 recipe AS-IS. Root cause: `reg_macro_f1` over ~4 700 sparse FL regions is dominated by rare-class noise (stays ~16-18 % across full ep=1-50 trajectory) and is blind to reg_top10's collapse from ~76 % at ep ~5 to ~65 % at ep ~30. The mean-of-F1s formula is scale-incoherent when one head has 7 well-supported classes (cat_macro_f1 ≈ 0.70) and the other has 4 700 sparse classes (reg_macro_f1 ≈ 0.17). See `CLAIMS_AND_HYPOTHESES.md` CH23-B and `CONCERNS.md` C21.

**Locked decisions (2026-05-19):**
- T6.4 does not promote to multi-seed or to AL/AZ MTL evaluation. Falsified.
- AL G3 gate violation (T6.4 low-visit Δ +1.05-1.41 pp vs +1 pp budget) is moot since T6.4 has no path to shipping under any selector.
- §0.1 reg numbers are reported under a known-broken selector. Until the F1 fix (below) is applied to the shipping baseline, **reg-side conclusions drawn from §0.1 multi-seed numbers under-report the substrate's reg capacity by ~10 pp**. The current paper canon stands as-is (it's internally consistent) but any future MTL paper should pair the §0.1-style numbers with the F1-fix numbers.
- All canonical_improvement Tier 1-6 candidate runs on disk can be re-analysed under any new selector **without retraining** via `scripts/canonical_improvement/analyze_t64_selectors.py` (reads per-epoch val CSVs).

**mtl-exploration F1 fix is URGENT — for shipping itself, not just for substrate variants.** Workstreams (`docs/studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md`):
- **F1** — substrate-aware joint_score (`reg_top10_acc_indist` instead of `reg_macro_f1`, or wire in the already-coded `joint_geom_lift` at `mtl_cv.py:710`). One-line code change. Re-evaluate **shipping AND all Tier 1-6 candidates** under the new selector without retraining; expose the ~11 pp reg-top10 capacity that the production selector currently hides.
- **F2** — substrate-adaptive MTL balancing (NashMTL revival on FL where the cvxpy solver is well-conditioned; per-task LR decay after reg peak; gradient masking after reg plateau). Goal: prevent reg destabilisation past its early peak so a single checkpoint near ep ~10-15 captures both heads near peak with low σ.
- **F3** — substrate × protocol 2×2 ablation as paper headline: (shipping, T6.4 substrate) × (B9 selector, F1-fix selector). Likely outcome based on this study: the protocol-axis effect dominates the substrate-axis effect on reg.

**Cross-references updated.** `CONCERNS.md` C21 (rewritten — not T6.4-specific; the bug is in shipping); `CLAIMS_AND_HYPOTHESES.md` CH23-A/CH23-B (locked claims with falsified-and-corrected framing); `AGENT_CONTEXT.md` blocker callout (rewritten); `NORTH_STAR.md` (B9 selector limitation warning rewritten to flag the bug as applying to the shipping recipe itself); `docs/studies/canonical_improvement/log.md` 2026-05-19 entry; `docs/studies/canonical_improvement/INDEX.html` T6.4 Results; `docs/studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md` (rewritten with matched-protocol numbers); `docs/studies/mtl-exploration/README.md` URGENT banner (rewritten).

**Artefacts.** `scripts/canonical_improvement/analyze_t64_selectors.py` (dual-selector tool, reads per-fold val CSVs); `docs/results/canonical_improvement/T6_4_dual_selector_final.{json,md}` (all 3 arms at matched single-seed=42 n=5 ep=50 — replaces the earlier `T6_4_dual_selector_preliminary.{json,md}` whose numbers were §0.1-vs-single-seed cross-selector comparisons).

---

### 2026-05-18 follow-up — canonical_improvement Tier-5 Phase-3 closed (no shipping change)

**Tier-5 Phase-3 closed; canonical+v3c+T3.2 remains the shipping stack.** After the 2026-05-18 first-pass Tier-5 close (`docs/results/canonical_improvement/STACKING_ABLATION.md §7.1-§7.5`), two further multi-seed cells landed in Phase 3:
- **T5.2b multi-seed extended to FL** (5 seeds × FL; `T5_2b_maePoi_FL_seed{42,0,1,7,100}.json`). 4/5 paired-positive on FL cat (mean +0.234 pp); FL reg flat at −0.069 pp. Closes 3-state coverage. **3-state cross-state cat sign-test 13/15 paired-positive, p = 0.0074** — strongest single piece of Tier-5 evidence.
- **T5.3 multi-seed ran** (AL+AZ × 5 seeds; `T5_3_multiview_{alabama,arizona}_seed42.json` + `T5_3_multiview_alaz_seed{0,1,7,100}.json`). §7.1 had T5.3 marked SKIPPED → §Future Work; Phase 3 un-skipped it. All four (AL+AZ × cat+reg) cells mean-positive; AZ reg Cohen d ≈ +0.85 (strongest Tier-5 effect size), p_one = 0.065 — sub-Bonferroni at m=28.

**Multiple-testing posture (Phase-3 update):** family count tightens from m = 26 (§7.3) to **m = 28** (Tier 1–4 + Phase 1 Hyp A/B/C/D + Tier 5 T5.1/T5.2a + T5.2b 3-state + T5.3 AL+AZ multi-seed). Bonferroni α* = 0.05/28 ≈ 0.00179. **No Tier-5 cell clears it.** T5.2b pooled cat sign-test (p=0.0074) misses by ~4× — closest to threshold.

**Shipping stack unchanged:** `canonical Check2HGI + v3c (AdamW WD=5e-2) + T3.2 ResLN encoder`. §5 paper headlines stand. Tier 5 closes as §Discussion-only in the BRACIS draft (Beats 5/6/7/8 in `PAPER_DRAFT.md §7`).

**Artefacts.** `docs/results/canonical_improvement/STACKING_ABLATION.md §7.6` (Phase-3 closeout); `docs/studies/canonical_improvement/log.md` (2026-05-18 follow-up entry); `docs/studies/canonical_improvement/INDEX.html` (T5.x pills updated; Phase-3 callout); `docs/findings/F62_T5_2b_implementation.md` (FL multi-seed section); `docs/findings/F63_T5_3_implementation.md` (multi-seed results replace SKIPPED placeholder); `articles/[BRACIS]_Beyond_Cross_Task/PAPER_DRAFT.md §7` (Beats 5/6/7/8); `articles/[BRACIS]_Beyond_Cross_Task/AUDIT_LOG.md §7` (record of Beats 5/6/7 → 5/6/7/8 replacement).

---

### 2026-05-02 — RESULTS_TABLE v11 (FL §0.1 arch-Δ upgraded to n=20 — all five states paper-grade)

**FL §0.1 architectural-Δ row upgraded from n=5 (single seed=42) to n=20 (seeds {0,1,7,100} × 5 folds).**
- Δ_reg = −7.34 pp, p = 1.9e-06, 0/20 fold-pairs positive (sign-consistent negative).
- Δ_cat = +1.40 pp, p = 2e-06, 20/20 fold-pairs positive.
- MTL B9 cat F1 = 68.56 ± 0.79 % (matches seed=42 reference 68.51 %); reg Acc@10 = 63.27 ± 0.10 %.
- **The last remaining headline asymmetry is closed.** All five states (AL/AZ/CA/TX/FL) are now n=20 multi-seed on §0.1 with paper-grade significance on the cat axis (AL small-significantly negative; AZ/CA/TX/FL paper-grade positive) and all five paper-grade significant on reg.
- Recipe used the canonical B9 invocation (matches `scripts/run_f51_multiseed_fl.sh` and CA/TX `run_h100_camera_ready_gaps.sh`): `--cat-head next_gru --reg-head next_getnext_hard --task-a-input-type checkin --task-b-input-type region --category-weight 0.75 --alternating-optimizer-step --scheduler cosine --max-lr 3e-3 --alpha-no-weight-decay`.

**Artefacts.** `research/FL_CAT_DELTA_WILCOXON.json` (new); `scripts/analysis/fl_cat_delta_wilcoxon.py`; `scripts/run_h100_fl_mtl_b9_multiseed.sh` (4-way H100 launcher with canonical recipe).

---

### 2026-05-02 — RESULTS_TABLE v10 (CA+TX §0.1 arch-Δ upgraded to n=20)

**CA and TX §0.1 architectural-Δ rows upgraded from n=5 (single seed=0) to n=20 (seeds {0,1,7,100} × 5 folds).**
- CA: Δ_reg = −9.50 pp p=2e-06; Δ_cat = +1.68 pp p=2e-06. Both axes paper-grade significant.
- TX: Δ_reg = −16.59 pp p=2e-06; Δ_cat = +1.89 pp p=2e-06. Both axes paper-grade significant.
- **All five states now have at minimum n=20 arch-Δ evidence** (AL/AZ n=20, FL n=5 ceiling, CA/TX n=20).
- Classic MTL tradeoff confirmed paper-grade at all large-scale states: reg trails STL by 7–17 pp; cat leads STL by 1.2–1.9 pp.

**Artefacts.** `research/ARCH_DELTA_WILCOXON.json` (new); `scripts/analysis/arch_delta_wilcoxon.py` (new); `scripts/run_h100_arch_delta_stl_ca_tx.sh` (16-run launcher).

---

### 2026-05-02 — RESULTS_TABLE v9 (TX recipe multi-seed landed; commit `928bdad`)

**TX B9 vs H3-alt upgraded from n=5 (single-seed=42) to n=20 (seeds {0,1,7,100} × 5 folds).**
- Δ_reg = +1.87 pp, p = 7e-04. Δ_cat = +0.52 pp, p = 2e-04. Both axes paper-grade significant.
- TX joins FL and CA as a large-scale state where B9 is paper-grade superior to H3-alt.
- **Recipe-selection narrative strengthened:** B9 is paper-grade at FL/CA/TX (all three large-scale states, n=20); H3-alt remains better at small scale (AL/AZ). Scale-conditional claim is now symmetric across all five states.

**Camera-ready audit item now fully closed.** The last remaining single-seed n=5 entry in §0.4 is resolved.

**Artefacts.** `research/GAP_FILL_WILCOXON.json` (TX section added); `scripts/analysis/gap_fill_wilcoxon.py` (Analysis C block).

---

### 2026-05-01 — RESULTS_TABLE v8 (Gap 1 + Gap 2 Wilcoxon landed; commit `bd707e8`)

**A) Cat-Δ Wilcoxon at AL/AZ/FL against multi-seed STL ceiling** (was "pending re-run" for 2 weeks).
- AL: Δ_cat = −0.78 pp, paired Wilcoxon p = 0.036 (n = 20 multi-seed; 14/20 fold-pairs negative). Statistically small-significantly negative. Magnitude small (~1.9 % relative on 41 % F1 scale).
- AZ: Δ_cat = +1.20 pp, p < 1e-04 (n = 20; 18/20 positive). Paper-grade.
- FL: Δ_cat = +1.52 pp (refined from +1.43 mean-diff to paired Δ), p = 0.0625 (n = 5 ceiling; 5/5 folds positive at seed = 42). Sign-consistent positive at single-seed ceiling.

**B) FL MTL B9 cat F1 refined 68.59 → 68.51 ± 0.51** (multi-seed pooled).

**C) CA recipe-selection (B9 vs H3-alt) upgraded to n = 20 multi-seed.**
- Δ_reg = +4.18 pp, p < 1e-04. Δ_cat = +0.51 pp, p < 1e-04. Paper-grade significant on both tasks.
- TX still pending multi-seed (single-seed n = 5 ceiling at submission).

**Lessons.**
- The "pending re-run" framing in the paper draft was a real BRACIS-rigour gap; running `gap_fill_wilcoxon.py` against the existing per-fold JSONs took ~5 minutes and resolved it.
- AL's "≈ tied" framing was generous given the formal stat (p = 0.036 in the negative direction). Honest framing now reports both significance and magnitude.

**Artefacts.** `research/GAP_FILL_WILCOXON.json`; `scripts/analysis/gap_fill_wilcoxon.py`.

---

### 2026-05-01 — Paper closure (Phases 1-3; commit `03af55c`)

**Five-state cross-state P3 + multi-seed at AL/AZ/FL + recipe ablation.**
- 28 paper-grade runs (5f × 50ep, leak-free per-fold log_T).
- AL/AZ B9 multi-seed at {0, 1, 7, 100}; FL B9 multi-seed at {42, 0, 1, 7, 100}; CA/TX seed = 42 single-seed.
- STL ceilings landed at all 5 states with multi-seed at AL/AZ/FL.

**Architectural-Δ picture (the classic MTL tradeoff, sign-consistent across 5 states):**
- Reg: MTL B9 < STL `next_stan_flow` at every state by 7–17 pp.
- Cat: MTL B9 ≥ STL `next_gru` at every state by 0 to +2 pp (refined to four-of-five-states-positive in v8 once cat-Δ Wilcoxon landed at AL).

**Reframe vs F49 (the leak that misled us).** F49's "AL +6.48 pp MTL > STL on reg" headline was a leak artefact of pre-F50 measurements (full-data `region_transition_log.pt` leaks ~13–27 pp). Under leak-free symmetric comparison, AL's reg pattern matches every other state. The headline "scale-conditional architecture-dominant state" framing from F49 is **superseded**.

**Lessons.**
- Leak detection (F44, F50 T4) caused a paper-reshaping reframe twice. Always run leak-free comparisons before declaring a champion.
- The B9 vs H3-alt recipe split is **scale-conditional**: B9 is FL-tuned; H3-alt is small-state universal. No single recipe wins on both axes across all states.

**Artefacts.** `PAPER_CLOSURE_RESULTS_2026-05-01.md` (now in `archive/post_paper_closure_2026-05-01/`); `research/PAPER_CLOSURE_WILCOXON.json`; `research/PAPER_CLOSURE_RECIPE_WILCOXON.json`.

---

### 2026-04-30 — F51 multi-seed validation (commit `f87321f`)

**B9 vs H3-alt across 5 seeds.** Pooled paired Wilcoxon (5 × 5 = 25 fold-pairs): Δ_reg = +3.48 ± 0.12 pp; p_reg = 2.98 × 10⁻⁸ (25/25 positive); p_cat = 1.33 × 10⁻⁵ (19/25 positive). Recipe is essentially deterministic across seeds (σ_across_seeds = 0.11 pp).

**F51 Tier 2 capacity sweep.** 21 capacity smokes confirm B9 is locally optimal in 5/7 capacity dimensions. Cat width-stability cliff at `shared_layer_size 384/512`. F52's "mixing is dead at FL" is depth-conditional.

**Per-seed log_T leak (caught + fixed mid-sweep 2026-04-30).** The original C4 fix wrote per-fold log_T as `region_transition_log_fold{N}.pt` with no seed in the filename, but the trainer loaded that file regardless of `--seed`. At any seed ≠ 42, ~80 % of val users live in seed = 42's fold-N TRAIN set → ~80 % val transition leak → reg inflated ~9 pp. **Fix:** filename is now `region_transition_log_seed{S}_fold{N}.pt`; trainer hard-fails if missing.

**Lesson.** Per-seed leakage is subtle. Always seed-tag prior files; hard-fail on missing or unseeded ones.

---

### 2026-04-29 to 2026-04-30 — Phase 3 leakage closure (F50 T4; commit `473dd41`)

**C4 leakage diagnosed and fixed.** Legacy full-data `region_transition_log.pt` leaked val transitions into training; ~13–17 pp inflation propagated through 5 heads. **Fix:** `--per-fold-transition-dir` builds log_T from train-fold-only edges per fold.

**Substrate-asymmetric leak.** C2HGI exploited the leaky log_T more than HGI (α grew more aggressively in C2HGI runs). This **inverted CH18-reg sign at every state** under leak-free measurement: HGI ≥ Check2HGI on reg by 1.6–3.1 pp under matched-head STL (vs the pre-leak-free framing where Check2HGI > HGI on reg).

**Lessons.**
- Leak symmetry is not guaranteed. Check whether different substrates exploit the leak differently before declaring a finding.
- The CH18-reg "MTL substrate-specific" claim was a leak artefact at the reg-side; only the cat-side substrate finding (CH16, CH18-cat) survives leak-free.

**Artefacts.** `research/F50_T4_C4_LEAK_DIAGNOSIS.md`; `research/F50_T4_BROADER_LEAKAGE_AUDIT.md`; `research/F50_T4_PRIOR_RUNS_VALIDITY.md`.

---

### 2026-04-29 — sklearn version reproducibility caveat (commit `4f2a982`)

`StratifiedGroupKFold(shuffle=True)` produces **different fold splits** across sklearn 1.3.2 → 1.8.0 (PR #32540). Within-phase paired tests are unaffected (both arms in each comparison ran in the same env on the same folds), but absolute leak-magnitude attribution across phases mixes leak removal with fold-shift. Disclosure: `FINAL_SURVEY.md §8`. **Fix-forward:** freeze fold indices via `scripts/study/freeze_folds.py` for any future runs.

---

### 2026-04-28 — F37 FL closing + F50 audit (commit `76f2443`)

**F37 STL `next_gru` cat 5f on FL = 66.98 ± 0.61** (pre-multi-seed; seed = 42). Δ_cat = +0.94 pp at FL — cat-side claim survives at scale. **STL `next_getnext_hard` reg 5f on FL = 82.44 ± 0.38** (legacy leaky). At the time, FL Δ_reg was reported as −8.78 pp paired Wilcoxon p = 0.0312, 5/5 folds negative. *(Post-leak-free: this leaky number was inflated; the leak-free FL Δ_reg is −7.99 in v8.)*

**F50 audit.** External-critic-driven audit of the MTL proposal — tiered plan T0/T1/T2/T3.
- T0 Δm joint score (Maninis 2019): backed CH22 — Pareto-positive at AL/AZ on MRR; Pareto-negative at FL on Acc@10. (Sign-flipped after leak-free reframe; final v8 has Δm-MRR positive at FL only.)
- T1 drop-in fixes: FAMO, Aligned-MTL, HSM-reg-head — none reach paired-Wilcoxon significance at FL against H3-alt.

**Artefacts.** `research/F37_FL_RESULTS.md`; `research/F50_DELTA_M_FINDINGS.md`; `research/F50_T1_RESULTS_SYNTHESIS.md`.

---

### 2026-04-27 — F49 attribution analysis (3-way decomposition; commit `a1996e9`)

**3-way decomposition (encoder-frozen λ = 0 / loss-side λ = 0 / Full MTL).**
- Cat-supervision transfer through `L_cat` is null on AL/AZ/FL n = 5 (≤ |0.75| pp); refuted the legacy "+14.2 pp transfer at FL" claim by ≥ 9σ on FL alone.
- AL: H3-alt reg lift = +6.48 pp from architecture alone (frozen-random cat features). *Post-leak-free, this turned out to be a leak artefact (asymmetric leak inflated MTL more than STL); see 2026-05-01 paper-closure entry.*

**Layer 2 methodological contribution (survives leak-free).** Loss-side `task_weight = 0` ablation is **unsound under cross-attention MTL**: the silenced encoder co-adapts via attention K/V projections; encoder-frozen isolation is the only clean architectural decomposition. Generalises to MulT, InvPT, and any cross-task interaction MTL. Regression tests in `tests/test_regression/test_mtlnet_crossattn_lambda0_gradflow.py`.

**Lesson.** The cross-attn ablation pitfall is the most general methodological survival of this study. The substantive AL "architecture-dominant" finding did not survive leak-free re-measurement.

**Artefacts.** `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`; `research/F49_LAMBDA0_DECOMPOSITION_GAP.md`.

---

### 2026-04-27 — Phase 1 substrate validation (commit `f0b2c95`)

**Five-leg substrate study at AL+AZ.**
- CH16 head-invariant: 8/8 head-state probes positive at p = 0.0312 (linear / `next_gru` / `next_single` / `next_lstm`); cat-side substrate Δ +11.58 to +15.50 pp.
- CH15 reframing (initial): under matched MTL reg head `next_getnext_hard`, C2HGI ≥ HGI on reg. *Later leak-free re-run sign-flipped this — see 2026-04-29 entry.*
- CH18 (initial): MTL B3 substrate-specific (HGI substitution breaks reg by 30 pp). *The reg-side of this claim was a leak artefact; only cat-side (CH18-cat) survives.*
- CH19: per-visit context = ~72 % of cat substrate gap at AL (POI-pooled counterfactual). Single-state mechanism evidence; survives all subsequent re-measurements.

**Lesson.** Phase 1 substrate findings on the *cat side* (CH16, CH18-cat, CH19) were robust to leak-free re-measurement. Reg-side findings (CH15 reframing, CH18-reg) were leak-dependent and sign-flipped.

---

### 2026-04-26 — F48-H3-alt champion (commit before `a1996e9`)

**Per-head LR recipe.** `cat_lr = 1e-3, reg_lr = 3e-3, shared_lr = 1e-3` (constant). At the time, claimed +6.25 pp MTL > STL on AL reg. Three orthogonal negative controls (F40, F48-H1, F48-H2) bracketed H3-alt as unique in its design space.

*Post-leak-free reframe.* The "+6.25 pp MTL > STL on AL reg" was a leak artefact. The H3-alt recipe survives as the small-state recipe (paper-grade better than B9 on cat at AL/AZ in `RESULTS_TABLE §0.4`); the original "AL architecture-dominant" lift narrative is superseded.

**Artefacts.** `research/F48_H3_PER_HEAD_LR_FINDINGS.md`; `MTL_ARCHITECTURE_JOURNEY.md` (preserved as supplementary material narrative).

---

### 2026-04-24 — F21c gap discovery; F27 cat-head refinement

**F21c.** Matched-head STL `next_getnext_hard` beat MTL B3 on reg by 12–14 pp at AL/AZ. Filed CH18 Tier B (methodological limitation). *Triggered the F38–F48 attribution chain and eventually the H3-alt recipe.*

**F27.** Cat head `NextHeadMTL` → `next_gru` (+3.43 pp AL 5f, +2.37 pp AZ 5f at p = 0.0312). FL flipped sign at n = 1 (later resolved by H3-alt FL 5f).

**Lessons.**
- Always run matched-head STL ceilings before declaring an MTL win. Comparing MTL against unmatched STL heads is over-claiming.
- Single-fold sign-flips are noise; resolve via 5-fold or multi-seed before reframing.

**Artefacts.** `research/F21C_FINDINGS.md`; `research/F27_CATHEAD_FINDINGS.md`.

---

### 2026-04-22 to 2026-04-23 — B3 champion identified

**B3 = `mtlnet_crossattn + static_weight(cat=0.75) + next_mtl (later next_gru) + next_getnext_hard`.** Validated 5-fold on AL + AZ + FL-1f. Beats baselines on cat F1; beats Markov-1-region on AL/AZ (FL Markov-saturated).

**Mechanism.** Late-stage handover under unbalanced static weighting: cat head converges fast in early epochs under high cat weight, then the shared backbone becomes available to reg in the remaining epochs (cat training extends to ep ~42 vs ≤ 10 for soft/equal-weight).

---

### 2026-04-16 to 2026-04-22 — Phase B-M architecture search (Tier B → B3)

Iterated through MTL backbone variants (`mtlnet`, `mtlnet_cgc`, `mtlnet_ple`, `mtlnet_mmoe`, `mtlnet_dselectk`, `mtlnet_crossattn`) and MTL loss variants (NashMTL, PCGrad, EqualWeight, StaticWeight, UncertaintyWeighting). Convergent choices: cross-attn backbone + static_weight(0.75) + GETNext-hard reg head.

---

### 2026-04-13 to 2026-04-16 — Initial study setup

Branch and study scope established. Phase 0 simple baselines (Markov, Majority). P1 head ablation. P1.5b user-disjoint fair-folds reset (after discovering CH16 measurement was leaky under non-grouped StratifiedKFold).

**Lesson.** User-disjoint cross-validation matters: under non-grouped folds, HGI memorises user-POI co-visit structure and over-performs; under user-disjoint folds, the substrate-asymmetry on cat (CH16) sharpens.

---

## Lessons learned (paper-prep meta)

These are the rules that came out of this study. Apply them to the **next** study.

### Single-source-of-truth discipline

1. **One canonical numerical source per paper.** For check2hgi this is `results/RESULTS_TABLE.md §0`. Every other doc references it; numbers in other docs that diverge are either stale (mark, fix, or archive) or audit-historical (clearly framed as such).
2. **Date-stamp the canonical source.** RESULTS_TABLE now has v6 → v7 → v8 → v9 → v10 stamps. Article-side cites the version explicitly. When the canonical updates, downstream docs must update in the same commit.
3. **Background provenance vs. canon.** Files like PAPER_CLOSURE_RESULTS that record the lab-trail of how a number was computed are valuable as audit but **not** as paper canon. Move them to `archive/` with a deprecation banner once the number lands in the canonical source.
4. **Use a CHANGELOG (this file).** Timeline-organised, dated, with what changed and why. Future readers (including yourself in 6 months) will not remember the F-trail; they will read this CHANGELOG.

### Leak detection discipline

1. **Run leak-free comparisons before declaring a champion.** Two of our biggest narrative reversals (F49 architecture-dominant; CH18-reg substrate-specific) were leak artefacts.
2. **Check leak symmetry.** Different substrates / methods can exploit a leak differently — symmetric removal is what reveals the true ordering.
3. **Seed-tag prior files.** A subtle per-seed log_T leak almost slipped through F51 multi-seed; only a hard-fail on missing seeded files saved us.

### Paired-test ceiling discipline

1. **n = 5 paired Wilcoxon has a ceiling at p = 0.0312 one-sided / 0.0625 two-sided.** State this once in §Experimental Setup; do not let reviewers think p = 0.0312 is a coincidence.
2. **Multi-seed pooling breaks the n = 5 ceiling.** Always pool fold-pairs across seeds where computable. n = 20 (4 seeds × 5 folds) reaches sub-1e-4 p-values. Without it, claims sit at the ceiling regardless of effect size.
3. **Honest small-significance framing.** AL's Δ_cat = −0.78 pp at p = 0.036 (n = 20) is *small-significantly negative*. Don't call it "tied" (that hides the significance) or "MTL trails STL on cat" (that overstates the magnitude); state both axes.

### Story-spine discipline

1. **Reviewer-facing ≠ workflow-facing.** "In flight on H100", "ETA ~1 h", "must check before T3 commits" belong in working notes, never in paper-prep docs that sub-agents will inherit.
2. **External critics are worth their weight.** Two Codex audit passes caught story-level overclaims (scale-sensitive title with TX outlier; substrate +33 pp conflating STL with MTL counterfactual; AL "≈ tied" understating significance). The cost of paying for a critical review is far less than the cost of a desk-rejection.
3. **Honest framing wins at BRACIS.** The 2023 best paper (*Embracing Data Irregularities*) led with "low computational cost", not peak F1. Our paper leads with "the substrate carries; the architecture pays" — also honest.

### Code-as-source-of-truth discipline

1. **The Wilcoxon should be a script, not a manual computation.** `scripts/analysis/gap_fill_wilcoxon.py` is reproducible and re-runnable. JSON artefacts are versioned and citable.
2. **Always emit a JSON.** `GAP_FILL_WILCOXON.json` (n = 20 fold-vectors per state per axis) is what lets a reviewer verify our numbers without re-running 28 paper-grade jobs.

---

## Pointers (where things live now)

> ⚠ **Updated 2026-05-14:** check2hgi study promoted from `docs/studies/check2hgi/` to `docs/` root. The tree below is the **historical (pre-2026-05-14)** snapshot — kept as the v10/v11 reference. For the current layout see [`docs/README.md`](README.md). Mapping:
> - `docs/studies/check2hgi/<file>` → `docs/<file>` (top-level docs)
> - `docs/studies/check2hgi/results/` → `docs/results/`
> - `docs/studies/check2hgi/research/` (F-trail) → `docs/findings/`
> - `docs/studies/check2hgi/research/{canonical_improvement,merge_design,hgi_category_injection}/` → `docs/studies/<name>/` (now active follow-up studies)
> - `docs/studies/check2hgi/baselines/` → `docs/baselines/` (merged with existing BASELINE.md)
> - `docs/studies/check2hgi/{paper,scope,review,launch_plans}/` → `docs/<name>/`
> - `docs/studies/check2hgi/issues/` → `docs/issues/check2hgi/`
> - `docs/studies/check2hgi/archive/<subdir>/` → `docs/archive/check2hgi-<subdir>/`

Historical (pre-2026-05-14) tree:

```
docs/studies/check2hgi/
├── README.md                              ← navigation hub (canonical-source-aware)
├── CHANGELOG.md                           ← THIS FILE (timeline + lessons)
├── AGENT_CONTEXT.md                       ← study briefing (post-v10)
├── NORTH_STAR.md                          ← champion config (post-v10)
├── CLAIMS_AND_HYPOTHESES.md               ← claim catalogue with whitelist banner
├── FINAL_SURVEY.md                        ← substrate panel (canonical)
├── CONCERNS.md                            ← acknowledged risks audit log
├── MTL_ARCHITECTURE_JOURNEY.md            ← supplementary material narrative (F-trail)
├── PAPER_BASELINES_STRATEGY.md            ← which baselines in which paper table
├── results/
│   ├── RESULTS_TABLE.md §0                ← THE canonical numerical source (v10)
│   ├── paired_tests/, P0/, P1/, ...       ← raw JSON artefacts
├── research/
│   ├── GAP_FILL_WILCOXON.json             ← v9 Wilcoxon JSON (cat-Δ + TX recipe landed)
│   ├── PAPER_CLOSURE_WILCOXON.json
│   ├── PAPER_CLOSURE_RECIPE_WILCOXON.json
│   ├── F49_LAMBDA0_DECOMPOSITION_GAP.md   ← cross-attn methodology contribution
│   ├── F50_DELTA_M_FINDINGS_LEAKFREE.md
│   ├── F51_MULTI_SEED_FINDINGS.md
│   ├── SUBSTRATE_COMPARISON_FINDINGS.md
│   └── ...                                ← per-experiment findings
├── baselines/                             ← faithful baseline ports + audits
├── paper/                                 ← paper-prep artefacts (methodology, results, limitations)
├── review/                                ← dated critical reviews
├── issues/, scope/, launch_plans/         ← audit / planning sub-dirs
└── archive/
    ├── post_paper_closure_2026-05-01/     ← stale paper-closure docs (this cleanup)
    ├── 2026-04-20_status_reports/
    ├── pre_b3_framing/
    ├── research_pre_b3/
    ├── research_pre_b5/
    ├── phases_original/
    └── v1_wip_mixed_scope/
```

**Article-side (BRACIS submission):** `articles/[BRACIS]_Beyond_Cross_Task/` — the working paper folder. Sub-agent fan-out plan in `PAPER_STRUCTURE.md` there; per-paragraph beats in `PAPER_DRAFT.md` there; numerical / statistical contracts in `STATISTICAL_AUDIT.md` and `TABLES_FIGURES.md` there.

---

## Maintenance

This file is the chronological record. Append new dated rows at the top (most recent first); never edit historic rows. When a doc is moved to `archive/`, log the move here under that day's row.
