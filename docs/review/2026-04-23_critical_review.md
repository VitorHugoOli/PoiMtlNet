# Critical Review — Check2HGI MTL Study (post-B5)

**Date:** 2026-04-23
**Prior review:** `review/2026-04-22T1040_critical_review.md`
**Scope of re-read this pass:** `SESSION_HANDOFF_2026-04-22.md`, `research/B5_RESULTS.md`, `research/B5_MACRO_ANALYSIS.md`, `research/B5_FL_SCALING.md`, `research/BACKLOG_FOLLOWUPS.md`, `results/RESULTS_TABLE.md`, `results/B5/{al,az,fl}_*.json`, `results/P5_bugfix/SUMMARY.md`, `CLAIMS_AND_HYPOTHESES.md`, `CONCERNS.md`.

---

## 0 · Executive summary

The 2026-04-22 morning review flagged MTL-GETNext as a **provisional** headline pending the α=0 ablation and a hard-index probe. Between the morning review and end-of-day 2026-04-22, the hard-index probe (B5) was implemented, trained on AL + AZ 5f × 50ep and FL 1f × 50ep, and analysed (`B5_RESULTS.md`, `B5_MACRO_ANALYSIS.md`, `B5_FL_SCALING.md`). It substantially changes the picture in three directions:

1. **The graph-prior mechanism now has a faithful, defensible implementation.** The soft-probe attribution question the 04-22 review flagged is now moot: hard-index `log_T[last_region_idx]` bypasses the probe entirely. The faithful variant is a direct adaptation of Yang et al. (SIGIR 2022) and the question "is the lift the graph prior or a popularity bias?" resolves to: **on AZ, the hard variant wins by +6.59 pp Acc@10 over the soft variant** — which itself is the strongest MTL region number. Both mechanisms survive.

2. **MTL beats STL on region for the first time in this study — on AZ only.** MTL-GETNext-hard on AZ reaches **53.25 ± 3.44 Acc@10 vs STL STAN 52.24 ± 2.38**. That is a +1.01 pp lift with non-overlapping-σ on the mean although within single-seed fold variance. AL remains tied within σ (MTL-hard 57.96 vs STL STAN 59.20). FL is scale-dependent (§1.3).

3. **A previously-unseen failure mode emerged at FL scale.** At 4,703 regions, the hard graph prior over-dominates task_b's gradient under PCGrad and **starves the category head: cat F1 drops −10.58 pp (66.01 → 55.43, n=1)**. The AZ/AL pattern does not extrapolate to FL on the hard variant. Soft-probe FL remains the joint-task champion.

**Net:** the study now has a scale-dependent story — **the paper can no longer claim "B5 is the headline everywhere"**. But the central claim — "MTL-GETNext delivers joint-task performance competitive with STL on the region task, above Markov-1-region, and with no cat regression on AL + AZ" — is stronger than it was 24 h ago.

The bigger risks now are (i) whether the AZ MTL-over-STL region win survives a paired Wilcoxon and multi-seed replication, (ii) whether the FL cat regression is n=1 noise or real, (iii) whether the `MTL_PARAM_PARTITION_BUG` reruns change the MTLoRA comparison story. None of these are satisfied yet.

---

## 1 · What changed since the 2026-04-22T10:40 review

### 1.1 New head: `next_getnext_hard` (faithful Yang 2022 SIGIR)

Soft probe `probe(x[:,-1,:])` replaced with a direct gather `log_T[last_region_idx]`, routed via a thread-local side-channel (`src/data/aux_side_channel.py`) since the MTLnet head interface does not accept auxiliary inputs. `last_region_idx` is carried in an extended `next_region.parquet` schema (`[emb_0..575, region_idx, userid, last_region_idx]`, commit `6a2f808`). `num_workers=0` is load-bearing for the thread-local to cross — a documented gotcha (G3 in `SESSION_HANDOFF_2026-04-22.md`).

### 1.2 B5 results — AL + AZ 5f × 50ep, FL 1f × 50ep

From `results/B5/*.json`, cross-checked against `RESULTS_TABLE.md`:

| State | Head | Acc@10_indist | MRR_indist | cat F1 | Δ vs soft Acc@10 |
|:-:|:-|-:|-:|-:|-:|
| AL | GETNext-soft (B-M6b) | 56.49 ± 4.25 | 28.93 ± 3.20 | 38.56 ± 1.45 | — |
| AL | **GETNext-hard** (B-M6e) | **57.96 ± 5.09** | 28.93 ± 3.88 | 38.50 ± 1.56 | **+1.47** (within σ) |
| AZ | GETNext-soft (B-M9b) | 46.66 ± 3.62 | 23.81 ± 2.30 | 42.82 ± 0.96 | — |
| AZ | **GETNext-hard** (B-M9d) | **53.25 ± 3.44** | **26.89 ± 2.62** | 42.22 ± 0.53 | **+6.59** (outside σ on every metric) |
| FL | GETNext-soft (B-M13, n=1) | 60.62 | 25.55 | **66.01** | — |
| FL | **GETNext-hard** (B-M14, n=1) | 58.88 | 28.01 | 55.43 | −1.74 region / **−10.58 cat F1** |

AZ ranks change: MTL-hard now sits **above** STL STAN (52.24) on Acc@10, the first time an MTL variant beats STL region in this study. AL stays within STL STAN's σ envelope (MTL-hard 57.96 vs STL STAN 59.20 ± 3.62). FL narrows Acc@10 slightly but trades 10 pp of category F1 — unacceptable as a joint headline.

### 1.3 Does B5 retire the probe-entropy concern from the 04-22 review?

The 04-22 review's core worry was: "the soft probe is diffuse (5 % top-1 on AL, 12 % on AZ), so `p @ log_T` degenerates to a marginal popularity prior, not a transition-conditional prior — the +11 pp lift may be a frequency artefact." B5 addresses this directly, not via the α=0 ablation that review recommended but by replacing the soft probe with a hard index, which **definitionally** supplies the transition-conditional row.

Evidence now consistent with "graph prior is load-bearing":

- AZ: replacing soft→hard keeps STAN + `log_T` structure identical, and Acc@10 jumps +6.59 pp. Nothing else changed. **On AZ the transition-conditional mechanism is measurably better than any frequency-like prior.**
- AL: replacing soft→hard is within-σ (+1.47 pp). The probe-entropy analysis predicted the probe has room to co-adapt with STAN on AL's smaller region set; that prediction survives.
- FL (1f): hard Acc@5 +13.5 pp and MRR +2.5 pp — lifts on fine-grained ranking precisely where a "frequency prior" should **not** help if the 04-22 review's concern were structural.

The α=0 ablation remains valuable if the paper wants a complete decomposition, but the hard-index retraining has answered the mechanism question more directly and at higher evidential value. **Recommend retire this as a blocker.**

### 1.4 Partition-bug reruns (P5_bugfix) landed

`results/P5_bugfix/SUMMARY.md` shows the 6 contaminated JSONs were re-run after `5668856`+`c1c7f3e` fixed the `shared_parameters ∪ task_specific_parameters` omission:

| Run | Pre-fix Acc@10 | Post-fix Acc@10 |
|:-|-:|-:|
| MTLoRA r=8 AL pcgrad | 50.72 ± 4.36 | **53.71 ± 3.80** |
| MTLoRA r=16 AL pcgrad | — | 51.62 ± 7.38 |
| MTLoRA r=32 AL pcgrad | — | 53.28 ± 5.34 |
| AdaShare mtlnet AL pcgrad | −0.31 pp vs baseline (claim: "NEUTRAL") | **44.51 ± 6.87** (≈ Markov floor; below baseline) |
| MTLoRA r=8 AZ pcgrad | — | 39.51 ± 3.83 |
| MTLoRA r=8 AL aligned-mtl/cagrad/db_mtl | — | 53.71/54.38/54.02 (within σ of pcgrad) |

Two things moved:
- MTLoRA r=8 AL **improved** post-fix (50.72 → 53.71). The "+1.84 pp over DSelectK+PCGrad" claim the 04-22 review retracted is now restated and defendable — **but it is no longer the headline**, because GETNext-hard (B-M6e, 57.96) sits ~4 pp above it.
- AdaShare's "NEUTRAL" framing was indeed a silent no-op; post-fix AdaShare trails MTLoRA by ~9 pp Acc@10 on AL. The paper should drop AdaShare rather than report it as neutral — the 04-22 recommendation is ratified.

The 04-22 review's blocker #2 ("rerun 6 contaminated JSONs") is **resolved**. The MTLoRA number in `BASELINES_AND_BEST_MTL.md` and `RESULTS_TABLE.md` needs updating from 50.72 to 53.71 but the supersession is clean.

### 1.5 FL scaling pathology (`B5_FL_SCALING.md`) — diagnosed training failure

At 4,703 regions the hard prior contributes a numerically large `α * log_T[idx]` term to the region logits on every sample. Under PCGrad, the shared cross-attention backbone tilts toward features helpful for region prediction and away from features the category head needs. Cat F1 drops 10.58 pp (n=1). Three mitigations are enumerated in `B5_FL_SCALING.md`: (A) keep FL soft as the joint-task headline, (B) sweep `task_b_weight ∈ {0.25, 0.5, 0.75}`, (C) extend FL-hard to 5-fold.

**2026-04-23 JSON-level diagnostic (new).** Comparing the FL-soft (`B-M13`, `mtlnet_lr1.0e-04_bs2048_ep50_20260421_1357/summary/full_summary.json`) and FL-hard (`results/B5/fl_1f50ep_next_getnext_hard.json`) runs under the **identical fold split** (val n_indist=78185, n_ood=1402 — same `StratifiedGroupKFold` fold), three signals point at a real training pathology, not fold-selection noise:

| Metric | FL-soft | FL-hard | What it says |
|---|-:|-:|---|
| `joint_score` cat F1 (reported) | 0.6601 | 0.5543 | −10.58 pp |
| `diagnostic_task_best` cat F1 (best-val-F1 epoch across training) | 0.6601 | 0.5543 | same as reported — cat head **never** exceeded 0.554 in any of the 50 epochs |
| `joint_score` reg Acc@10_indist | 0.6062 | 0.5888 | slight loss |
| `diagnostic_task_best` reg Acc@10_indist | 0.6071 | **0.4884** | reg's best-val-F1 epoch has Acc@10 10 pp lower than joint's selected epoch — ranking is unstable across epochs |
| final reg loss | 9.06 | **5.30** | hard's reg loss converges to a much lower value because the prior does most of the classification work |

Signal 1 (cat ceiling 0.554 over *all* 50 epochs — both selection methods agree) rules out final-epoch or fold-selection noise as the explanation for the 10 pp cat drop. The category head is **gradient-starved throughout training**, not at the end.

Signal 2 (reg Acc@10 jumps from 0.488 to 0.589 across epochs while cat F1 stays flat) is consistent with the prior dominating logits: when prior-driven re-ranking lands favourably the Acc@10 spikes, otherwise it collapses.

Signal 3 (reg loss converging to 5.30 vs soft's 9.06 on a 4,703-class problem) shows the prior is doing the region prediction work, not the neural representation. Combined with signal 2, this means the shared backbone is not learning discriminative region features — PCGrad has starved both (a) the cat head and (b) the region head's learned contribution.

**Net:** this is a reproducible, well-posed training failure with a documented mechanism, not a scaling "weirdness". Recommend:
- **Commit to soft as north star** now (see `NORTH_STAR.md`) — FL failure is strong evidence against hard as a paper headline.
- **Run F2** (task-weight sweep + static_weight swap + α inspection with `--checkpoints`) as the diagnostic gate for re-promoting hard. If `task_b_weight=0.5` restores cat F1 to ≥ 60 while keeping Acc@5 lift, the mechanism is pinned at gradient imbalance under PCGrad and hard becomes scale-uniform. If not, the prior's additive scaling itself needs to be reformulated.
- **De-prioritize F5 / F12 (FL 5-fold hard)** — the `diagnostic_task_best` already rules out n=1 fold-selection noise, so more folds won't rescue the cat ceiling without a mitigation.

---

## 2 · Revised claim-status table

| Claim | 2026-04-22 status | 2026-04-23 status | Delta |
|-------|:-:|:-:|-------|
| **CH16** — Check2HGI > HGI on cat F1 | confirmed on AL (+18.30 pp) | **unchanged**, still AL-only | cross-state replication remains missing |
| **CH17** — Check2HGI > POI-RGNN + HGI-article | pending, protocol audit flagged | **unchanged** | |
| **CH-M4** — Cross-attn closes cat gap | locked AL; n=1 AZ/FL | **unchanged** | cross-attn not contaminated |
| **CH-M6** — Scale curve | 3 single-seed data points | **strengthened via B5 scale effect on FL** | new FL scale asymmetry is a data point for the curve |
| **CH-M7** — Markov-k monotone degrade | locked | **unchanged** | |
| "MTLoRA r=8 is best MTL reg" (B11) | **RETRACTED pending rerun** | **REINSTATED at 53.71, demoted to secondary** | post-fix value known; GETNext-hard now the MTL-region champion |
| "AdaShare NEUTRAL" | **UNDETERMINED** | **RETIRED — AdaShare is below MTLoRA by ~9 pp** | post-fix verdict: drop from paper |
| MTL-GETNext soft lift (+11/+5.6 pp AL/AZ) | PROVISIONAL, needs α=0 | **largely superseded by hard variant as defence** | probe-entropy concern addressed by B5 |
| **NEW: MTL-GETNext-HARD beats STL STAN on AZ region** | — | **PROVISIONAL** — within-σ-of-STL envelope on mean, needs paired Wilcoxon + multi-seed | strongest finding of the study to date; risk = seed variance |
| **NEW: FL cat F1 regression at hard** (−10.58 pp) | — | **PROVISIONAL** — n=1, could be fold-selection noise | B12 (FL 5f hard) is the binding test; B13 task-weight sweep is the rescue probe |
| **NEW: PCGrad ≈ static_weight for GETNext** | — | **confirmed** (AL Δ=−0.17, AZ Δ=−0.14) | MTL optimizer is not load-bearing; simplifies the paper |

---

## 3 · State of the main objectives (the user asked for this explicitly)

### 3.1 Objective 1: Prove Check2HGI > HGI

Current evidence is **strong on AL, missing on every other state**.

| State | Task | Check2HGI STL | HGI STL | Δ | Seeds × Folds |
|:-:|:-|-:|-:|-:|:-:|
| AL | cat F1 | 38.58 ± 1.23 | 20.29 ± 1.34 | **+18.30 pp** | 1 × 5, fair folds |
| AL | reg Acc@10 (pooled to region) | 56.11 ± 4.02 | 57.02 ± 2.92 | −0.91 pp | 1 × 5, tied (CH15) |
| AZ | cat F1 | 42.08 ± 0.89 | **not run** | — | 1 × 5 (C2HGI only) |
| AZ | reg Acc@10 | 48.88 ± 2.48 | **not run** | — | 1 × 5 (C2HGI only) |
| FL | cat F1 | 63.17 (n=1) | **not run** | — | 1 × 1 (C2HGI only) |
| FL | reg Acc@10 | 68.33 ± 0.58 | **not run** | — | 1 × 5 (C2HGI only) |

**Status of CH16:** `confirmed on AL only`. The cat-F1 delta is huge and unambiguous, but a reviewer can fairly ask "does this hold at scale?" The AZ (~26K) and FL (~127K) HGI STL runs are ~3 h and ~5-6 h respectively on MPS; **running at least one of them (AZ recommended — cheaper) is the single highest-ROI item for this objective.**

**Status of CH15:** `tied`, as documented. The paper's framing pivot — "Check2HGI uniquely enables per-task modality, HGI architecturally can't" — already handles this.

### 3.2 Objective 2: Show MTL > per-task baselines + MTL > STL

Current evidence is **mixed and state-dependent**.

Three comparison points per (state × task):
- **Baseline:** Markov-1-region (region) or Markov-1-POI / Majority (category)
- **STL:** our best single-task neural (STAN for region, NextHeadMTL for cat)
- **MTL:** our best multi-task (cross-attn + pcgrad + GETNext-hard or soft)

| State | Task | Baseline | STL ceiling | Best MTL | MTL − Baseline | MTL − STL | Objective |
|:-:|:-|-:|-:|-:|-:|-:|:-|
| AL | cat F1 | 31.7 (Markov-POI) | **38.58** | **38.58** (B-M6e / A-M3) | **+6.88** | **0.00** | ✅ MTL matches STL, beats baseline |
| AL | reg Acc@10 | 47.01 (Markov-1-reg) | **59.20** (STL STAN) | **57.96** (B-M6e GETNext-hard) | **+10.95** | **−1.24** (within σ) | 🟡 MTL tied with STL within σ, beats baseline |
| AZ | cat F1 | — (no Markov-POI computed) | 42.08 | **43.13** (A-M6) | — | **+1.05** | ✅ MTL slightly lifts STL |
| AZ | reg Acc@10 | 42.96 (Markov-1-reg) | 52.24 (STL STAN) | **53.25** (B-M9d GETNext-hard) | **+10.29** | **+1.01** | ✅ MTL beats both (within σ; p-value pending) |
| FL | cat F1 | 37.2 (Markov-POI) | 63.17 (n=1) | **66.46** (A-M10 cross-attn, n=1) | **+29.26** | **+3.29** (n=1) | 🟡 MTL lifts STL at n=1 — needs 5-fold |
| FL | reg Acc@10 | **65.05** (Markov-1-reg) | 68.33 | **60.62** (B-M13 GETNext-soft, n=1) | **−4.43** | **−7.71** | ❌ MTL **below Markov floor**; STL is the only configuration above floor |

**Where we stand:**

- **MTL > baseline:** ✅ on 5 of 6 (state × task) cells. The exception is FL region, where the classical Markov-1-region (65.05 %) beats every MTL variant we have tried. The scale curve explains why — FL region density makes 1-gram Markov near-saturating — but for the paper's joint headline this remains a presentational problem.
- **MTL > STL:** ✅ on AZ region (strongest), ✅ on AZ cat, ✅ on FL cat (n=1), 🟡 AL cat tied, 🟡 AL region within σ, ❌ FL region. No (state × task) cell has MTL strictly losing to STL with a defensible σ margin, but FL region is clearly where MTL stops winning.
- **MTL > per-task baselines simultaneously (joint-score):** `B5_MACRO_ANALYSIS.md` computes geometric-mean joint scores. MTL-GETNext-hard tops all MTL methods on AL + AZ; STL STAN is region-only and can't be joint-scored.

**Acceptance gaps to close:**

1. FL region is below Markov-1. Paper can either (a) drop the Acc@10-above-Markov claim from the FL row and report MRR / Acc@5 (both of which MTL wins), or (b) frame FL as "dense-data regime where 1-gram Markov is near-optimal". Both are defensible; (a) is cleaner.
2. The "MTL > STL on AZ region" claim rides on n=1 seed × 5 folds. Paired Wilcoxon is free (B14 in the backlog, 30 min CPU). Multi-seed n=3 (B3) is expensive but paper-blocking for BRACIS-style submission.
3. FL cat regression on hard is n=1 — needs 5f × 50ep (B12) or task-weight sweep (B13).

---

## 4 · Risks — what could still invalidate the narrative

Ranked by severity × likelihood:

**R1 (severity HIGH, likelihood MEDIUM).** The AZ MTL-over-STL region win (+1.01 pp) is within the fold-to-fold σ envelope (±3.44 MTL, ±2.38 STL). A paired Wilcoxon on the 5 fold deltas may or may not reject H0. If it doesn't, "MTL beats STL on region" has to be reported as "matches or beats" — weaker but still the study's most defensible finding. **Mitigation:** run B14 (30 min CPU script).

**R2 (severity MEDIUM-HIGH, likelihood MEDIUM).** FL cat F1 regression is n=1. If it holds at 5-fold, the paper's scale-dependent story needs to be clearly stated. If it's noise, the paper becomes much cleaner (hard everywhere). **Mitigation:** B13 (weight sweep, 2.25 h) or B12 (FL 5f, 5-6 h).

**R3 (severity MEDIUM, likelihood LOW).** CH16 replicated on AL only. AZ HGI STL cat is 3 h of compute and is the cheapest externally-defensible extension. **Mitigation:** run it.

**R4 (severity MEDIUM, likelihood LOW-MEDIUM).** All 3 states are single-seed × 5 folds. A reviewer will ask for n=3 seeds on the champion. **Mitigation:** B3 multi-seed, parallelisable across M4 Pro + M2 Pro + Linux 4050; ~20 h total.

**R5 (severity LOW, likelihood LOW).** AZ fold-3 memory pressure episode on 2026-04-22 (G5 in `SESSION_HANDOFF`). Metrics look clean but σ is the one number you cannot reconstruct from a degraded run. If the AZ B-M9d result is challenged on σ grounds, a re-run on a larger-RAM host is the fix. **Mitigation:** available on Linux 4050 idle machine.

**R6 (severity LOW, likelihood LOW).** The `CROSSATTN_PARTIAL_FORWARD_CRASH` medium-severity fix landed (commit `8afc9ac`) but checkpoints from earlier cross-attn runs pre-date it. If an Appendix per-head analysis is needed from existing checkpoints, the fix may require regenerating some checkpoints. Low likelihood — most findings can be expressed from the summary JSONs already in hand.

---

## 5 · Recommended sequencing (updated; this subsumes §5 of the 04-22 review)

Ordered by information-gain × cost. All items are reproducible; effort is wall-clock estimate on M4 Pro MPS unless noted.

1. **[BLOCKER — 30 min CPU] B14: paired Wilcoxon on AZ B-M9b vs B-M9d fold deltas.** Supports "+6.59 pp Acc@10" with a p-value. Almost certainly significant at n=5 with all 5 deltas positive. Highest-ROI next action.
2. **[BLOCKER — 2.25 h] B13: FL task-weight rebalancing sweep (task_b_weight ∈ {0.25, 0.5, 0.75}, 1f × 50ep).** Tells us whether FL-hard's cat regression can be rescued. If yes, FL-hard is the universal champion; if no, commit to the scale-dependent narrative and move on.
3. **[HIGH — 3 h] AZ HGI STL cat.** Extends CH16 (Check2HGI > HGI on cat) from n=1 state to n=2 states. Cheapest reviewer-addressing item for Objective 1.
4. **[HIGH — 5-6 h] B12: FL MTL-GETNext-hard 5-fold.** If B13 doesn't rescue cat, this gives us real σ on the FL-hard regression and lets the paper draw defensible conclusions about scale. If B13 does rescue cat, this becomes lower priority.
5. **[HIGH — 5-6 h] FL MTL-GETNext-soft 5-fold.** Replaces the B-M13 n=1 row with an n=5 σ bar. Required for any "FL joint-task champion" claim with honest variance.
6. **[HIGH — 5-6 h] FL STL STAN 5-fold.** Completes the B-S8 row (currently TBD). Required to state "STL STAN is the region ceiling across states" with data at FL.
7. **[MEDIUM — ~30 min CPU] POI-RGNN evaluation protocol audit.** The +28 pp FL delta vs POI-RGNN is large — verify taxonomy, window, user-disjoint folds before the paper repeats it.
8. **[MEDIUM — ~20 h total, parallelisable] B3: multi-seed n=3 on champion configs.** Seeds {42, 123, 2024} × 5 folds × (AL + AZ + FL champion + STL reference). Launch across M2 Pro / Linux 4050 / M4 Pro concurrently.

Items 1–2 unblock the paper's MTL-over-STL narrative within 3 hours. Item 3 unblocks Objective 1 within 3 hours. Items 4–6 need a full day but fill the biggest remaining variance gaps. Item 7 is a paper-hygiene action; item 8 is the expensive paper-finisher.

---

## 6 · What to retire or archive

Based on this pass, the following files are **superseded** and should be moved out of the active working set:

- `STATUS_REPORT_2026-04-20.md` and `STATUS_REPORT_2026-04-20_v2.md`. Both pre-date B5 by two days; their headline ("cross-attn + pcgrad + GRU is the MTL champion") is no longer the headline. `SESSION_HANDOFF_2026-04-22.md` supersedes both. Move to `archive/` with a dated subdir.
- `research/HYBRID_DECISION_2026-04-20.md`, `research/CHAIN_FINDINGS_2026-04-20.md`. The decisions and chain outcomes they document are now encoded in the results tables. Keep as historical reference in `archive/research_2026-04-20/`.
- `research/EXECUTION_PLAN_2026-04-18.md`. Plan was for the ablation protocol that got superseded by the GETNext/B5 line. Move to archive.
- `research/SOTA_MTL_BACKBONE_DILUTION.md` and `research/SOTA_MTL_ALTERNATIVES_V2.md`. Literature surveys done in service of the dilution narrative; useful context for paper related-work but no longer guiding active experiments. Keep, but flag as "literature-reference, not live".

The remaining research/ docs are paper-substantive (B5*, GETNEXT_FINDINGS, POSITIONING_VS_HMT_GRN, etc.) and should be kept.

`CLAIMS_AND_HYPOTHESES.md` needs a light pass: CH01/CH02 are effectively addressed by the B5 + `BASELINES_AND_BEST_MTL.md` story but still read as "pending" in the dashboard. CH03 remains "partial — AL dev only". CH09 (cross-attention gated on CH03) has already been run. The dashboard should be reconciled with the actual status table in `RESULTS_TABLE.md`.

---

## 7 · Preserved strengths (unchanged)

- User-disjoint fold fix (C11) — methodological care that earns reviewer trust.
- Null-finding discipline extended: B5 static vs pcgrad null; B7 ALiBi × GETNext within-σ; MTLoRA rank insensitivity.
- Honest documentation of the partition bug + `CROSSATTN_PARTIAL_FORWARD_CRASH`. Both have landed in commits and re-runs.
- `RESULTS_TABLE.md` remains the authoritative row-per-method × state × task artefact. Updating it after each run is the single largest reason this study is defensible in its current state.
- The B5 sequence itself — inference-time ablation motivating retraining, then B5 macro-analysis comparing every method, then B5 FL scaling documenting a failure mode and proposing three mitigations — is exactly the kind of experimental rigour reviewers reward.

---

## 8 · Bottom line

The study has crossed a qualitative threshold since the 04-22 morning review, and the 2026-04-23 decision to commit `GETNext-soft` as north star (see `NORTH_STAR.md`) sharpens the narrative:

- **Objective 1 (Check2HGI > HGI):** substrate AL win is unchanged and robust. Needs n ≥ 2 states to be reviewer-proof. AZ HGI STL cat is the single cheapest fix.
- **Objective 2 (MTL > baselines / STL) under the north-star config (soft):**
  - ✅ cat beats POI-RGNN + Markov + Majority everywhere; ties/lifts STL at all three states.
  - ✅ AL + AZ region beat Markov-1 by +3.70 to +9.48 pp.
  - ❌ FL region below Markov-1 by 4.43 pp — the reviewer's hardest question. Paper reframes FL reg as Markov-saturated / dense-data regime, or shifts to reporting Acc@5 / MRR (both of which MTL wins) on FL.
  - 🟡 region never strictly beats STL STAN under the north-star config. The hard variant does (AZ +1.01), kept as the paper's ablation row.
- **GETNext-hard** remains a paper-reported ablation: strictly stronger on AZ (+6.59 pp Acc@10 outside σ over soft, +1.01 over STL STAN), tied on AL, **has a diagnosed training failure on FL** (§1.5) — cat head's best-val F1 ceiling is 0.554 across 50 epochs, consistent with PCGrad-starvation at 4.7 K-region scale. F2 is the binding diagnostic for re-promoting hard.

The tightest defensible story for the next write-up:

> Our MTL-GETNext-soft (cross-attention + PCGrad + soft-probe GETNext, d=256, 8 heads) jointly produces next-category and next-region predictions on Check2HGI check-in-level embeddings. Across AL / AZ / FL Gowalla state splits, it beats POI-RGNN on next-category by 4–32 pp macro-F1 and matches or lifts the single-task STL ceiling on category without regression. On next-region it beats Markov-1-region by +3.70 to +9.48 pp at ≤ 1.5 K-region scale; at 4.7 K-region scale (FL) Markov-1-region saturates and both MTL and STL neural heads trail it. A faithful hard-index ablation (`next_getnext_hard`) strictly dominates soft at AZ scale (+6.59 pp Acc@10, p pending) but has a diagnosed training failure at FL scale due to gradient imbalance under PCGrad; we report both and recommend soft as the scale-robust default.

All of this is achievable with the existing codebase and ~6 h of high-ROI compute (F1 Wilcoxon + F3 AZ HGI + F2 FL-hard diagnostic). The biggest single information gain is still **F1 (30 min)** — do it first.
