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
| Five-state architectural-Δ + cat-Δ Wilcoxon | `results/RESULTS_TABLE.md §0.1` | **v10, 2026-05-02** (CA+TX upgraded to n=20; all four axes p=2e-06) |
| Δm joint score (CH22 leak-free) | `results/RESULTS_TABLE.md §0.2` | v6 (leak-free) — unchanged in v7/v8/v9/v10 |
| Substrate axis (CH16 cat + CH15 reg reframing) | `results/RESULTS_TABLE.md §0.3` + `FINAL_SURVEY.md §2-§4` | v6 — unchanged in v7/v8/v9/v10 |
| Recipe selection (B9 vs H3-alt) | `results/RESULTS_TABLE.md §0.4` | **v9, 2026-05-02** (TX upgraded to n=20 multi-seed) |
| External baselines | `results/RESULTS_TABLE.md §0.5–§0.6` | unchanged |
| Champion config + recipe | `NORTH_STAR.md` | v8 (mirrors RESULTS_TABLE) |
| Claim catalogue | `CLAIMS_AND_HYPOTHESES.md` (whitelist banner) | CH16 / CH18-cat / CH15 reframing / CH19 / CH22 are paper-facing safe |
| Article-side paper docs | `articles/[BRACIS]_Beyond_Cross_Task/` | v8-aligned 2026-05-01 (BRACIS submission) |

**Rule of single-source.** Paper tables are sourced from `RESULTS_TABLE.md §0` and only that. `PAPER_CLOSURE_RESULTS_2026-05-01.md` is background provenance — it has been moved to `archive/post_paper_closure_2026-05-01/` because numerous numbers there were superseded by RESULTS_TABLE v7/v8 (e.g., FL Δ_reg simple mean-diff −7.28 vs paired Δ −7.99; AL/AZ STL cat means refreshed from single-seed to multi-seed).

---

## Timeline of findings (most recent first)

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
2. **Date-stamp the canonical source.** RESULTS_TABLE has v6 → v7 → v8 stamps. Article-side cites the version explicitly. When the canonical updates, downstream docs must update in the same commit.
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

## Pointers (where things live now, post-cleanup)

```
docs/studies/check2hgi/
├── README.md                              ← navigation hub (canonical-source-aware)
├── CHANGELOG.md                           ← THIS FILE (timeline + lessons)
├── AGENT_CONTEXT.md                       ← study briefing (post-v8)
├── NORTH_STAR.md                          ← champion config (post-v8)
├── CLAIMS_AND_HYPOTHESES.md               ← claim catalogue with whitelist banner
├── FINAL_SURVEY.md                        ← substrate panel (canonical)
├── CONCERNS.md                            ← acknowledged risks audit log
├── MTL_ARCHITECTURE_JOURNEY.md            ← supplementary material narrative (F-trail)
├── PAPER_BASELINES_STRATEGY.md            ← which baselines in which paper table
├── results/
│   ├── RESULTS_TABLE.md §0                ← THE canonical numerical source (v8)
│   ├── paired_tests/, P0/, P1/, ...       ← raw JSON artefacts
├── research/
│   ├── GAP_FILL_WILCOXON.json             ← v8 Wilcoxon JSON (cat-Δ landed)
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
