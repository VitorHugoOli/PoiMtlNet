# Paper Structure — Single Source of Truth

**Created:** 2026-04-23. **Updated 2026-04-27** with two complementary tracks: Phase-1 substrate-validation findings (substrate-side: CH16 head-invariant + CH15 reframed + CH18 MTL-substrate-specific) AND F49 architectural attribution (architecture-side: CH19 transfer-null + Layer 2 methodological). **Owner:** this file defines the paper's table layout, baseline set, STL matching policy, and scope decisions. All other docs reference this.

> **2026-04-27 STL-baseline matching policy revision.** Matched-head STL = `next_gru` (cat) / `next_getnext_hard` (reg) — these are the post-F27 MTL B3 task heads. The pre-Phase-1 matched-head policy (which used `next_single` for cat, STAN for reg) is retained as a **head-sensitivity probe row** (still valid as evidence; closes C2 critique). See `research/SUBSTRATE_COMPARISON_PLAN.md` §1.2 + `research/SUBSTRATE_COMPARISON_FINDINGS.md` §5.

> **F49 paper-claim layering (2026-04-27).** The H3-alt champion is unchanged. F49 sharpens *why* it works:
> - **Layer 1 (Tier A):** cat-supervision transfer is small (≤|0.75| pp) on AL/AZ/FL n=5 — refutes legacy +14.2 pp claim by ≥9σ on FL alone.
> - **Layer 2 (Tier A, methodological):** loss-side `task_weight=0` ablation is unsound under cross-attention MTL (silenced encoder co-adapts via attention K/V); encoder-frozen isolation is the only clean architectural decomposition.
> - **Layer 3 (paper-grade for AL/AZ; FL absolute Δ pending F37):** AL architectural = +6.48 ± 2.4 pp (~2.7σ); AZ architectural = −6.02 ± 1.6 pp (~3.7σ) with cat-side rescue.
>
> See `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` and `CLAIMS_AND_HYPOTHESES.md §CH19`.

---

## 1 · Champion model (`NORTH_STAR.md`, `MTL_ARCHITECTURE_JOURNEY.md`)

**F48-H3-alt** (champion candidate, 2026-04-26) — B3 architecture + per-head LR:

```
architecture   : mtlnet_crossattn
mtl_loss       : static_weight(category_weight = 0.75)
task_a head    : next_gru                                 # F27 (was next_mtl)
task_b head    : next_getnext_hard                        # STAN + α · log_T[last_region_idx]
task_a input   : check-in embeddings (9-step window)
task_b input   : region embeddings (9-step window)
hparams        : d_model=256, 8 heads, batch=2048 (1024 on FL), 50 epochs, seed 42
LR scheduler   : constant (no OneCycleLR / no annealing)
LR per group   : cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3   # ← H3-alt recipe
```

CLI delta vs predecessor B3: `--scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3`.

**Predecessor (B3, 2026-04-24, kept as comparand):** B3 = same architecture but with OneCycleLR (max_lr=0.003) and single LR. The H3-alt contribution is measured against this baseline.

### 1.1 · Paper-reshaping findings since 2026-04-23

- **F27 (2026-04-24)** — cat-head swap `next_mtl → next_gru` lifts cat F1 by +3.43 pp (AL 5f) and +2.37 pp (AZ 5f, Wilcoxon p=0.0312 on 3 metrics). **FL 1f flipped sign (−0.93 pp)** at n=1 noise; H3-alt FL 5f (+2.20 pp over predecessor B3) resolves the flag. See `research/F27_CATHEAD_FINDINGS.md`.
- **F21c (2026-04-24)** — matched-head STL `next_getnext_hard` outperformed B3 on reg Acc@10 by 12–14 pp on AL + AZ. Triggered the CH18 attribution chain (F38–F48). **RESOLVED by F48-H3-alt (2026-04-26):** per-head LR closes/exceeds the gap (AL +6.25 pp over STL F21c, AZ closes 75% of B3 gap). CH18 promoted Tier B → A. See `research/F21C_FINDINGS.md` and `research/F48_H3_PER_HEAD_LR_FINDINGS.md`.
- **F48-H3-alt (2026-04-26)** — per-head LR is the new champion candidate. Paper-strength MTL-over-STL contribution validated 5-fold on AL+AZ+FL. Three orthogonal negative controls (F40 loss-side, F48-H1 gentle constant, F48-H2 warmup-then-plateau) bracket H3-alt as the unique design satisfying joint cat+reg. See `MTL_ARCHITECTURE_JOURNEY.md` for end-to-end derivation.
- **F49 (2026-04-27)** — 3-way decomposition (encoder-frozen λ=0 / loss-side λ=0 / Full MTL) under H3-alt regime. **AL: H3-alt reg lift is +6.48 pp from architecture alone**; cat-supervision transfer ≈ 0. **AZ: classical-MTL pattern** (architectural overhead, multi-task wrap rescues part). **FL n=5: cat-supervision transfer null on all 3 states** (≤|0.75| pp); refutes legacy "+14.2 pp transfer" claim at ≥9σ on FL alone. **Methodological contribution (Layer 2):** loss-side `task_weight=0` ablation is unsound under cross-attention MTL (silenced encoder co-adapts via attention K/V). CH19 added (Tier A). See `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`. F49b reproduction gate at `max_lr=3e-3` matches legacy 52.27 cleanly (53.18 ± 4.56 vs 52.27 ± 5.03, ~0.13σ).

Validation status (as of 2026-04-26):

| State | Status | H3-alt source |
|---|---|---|
| AL 5f | ✅ validated | `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260425_1843/summary/full_summary.json` |
| AZ 5f | ✅ validated | `results/check2hgi/arizona/mtlnet_lr1.0e-04_bs2048_ep50_20260425_1853/summary/full_summary.json` |
| FL 5f | ✅ validated (batch=1024) | `results/check2hgi/florida/mtlnet_lr1.0e-04_bs1024_ep50_20260426_0045/summary/full_summary.json` |
| FL F49 5f loss-side λ=0 | ✅ (F49c) | `results/check2hgi/florida/f49c_lossside_5f_2026042715/summary/full_summary.json` |
| FL F49 5f frozen-cat λ=0 | ✅ (F49c, frozen-cat unstable) | `results/check2hgi/florida/f49c_frozen_5f_2026042718/summary/full_summary.json` |
| CA 5f | 🔴 pending — upstream pipeline missing | F22/F24 |
| TX 5f | 🔴 pending — upstream pipeline missing | F23/F25 |

---

## 2 · Paper scope

### 2.1 Scope split

- **Headline paper table:** **FL + CA + TX**, all 5-fold × 50-epoch, user-disjoint `StratifiedGroupKFold`, seed 42.
- **Ablations / mechanism studies:** **AL + AZ 5-fold** and **FL 1-fold** (already complete). The mechanism findings (FL-hard gradient starvation, late-stage handover, α trajectory, PCGrad-vs-static isolation) live in this band.
- **Out of scope:** multi-seed n=3 on champions (deferred, `F8` in tracker — will be re-added only after headline is locked).

### 2.2 Why this split works

AL is dev-scale (10 K rows, 1.1 K regions) and AZ is mid-scale (26 K, 1.5 K); both are too small to serve as reviewer-facing headlines but are excellent ablation beds (cheap to iterate, clean σ, already 5-fold). FL is the smallest *headline* state (127 K, 4.7 K regions); CA and TX are in the same scale class. The paper's "headline states are the three largest US states in our Check2HGI training set" framing is defensible per `CONCERNS.md §C01`.

---

## 3 · Baselines — per task

### 3.1 next_category (7 classes, macro-F1 primary)

> **Audit hub:** `baselines/next_category/` ([README](../../studies/check2hgi/baselines/README.md), [comparison.md](../../studies/check2hgi/baselines/next_category/comparison.md), per-baseline files [poi_rgnn.md](../../studies/check2hgi/baselines/next_category/poi_rgnn.md), [mha_pe.md](../../studies/check2hgi/baselines/next_category/mha_pe.md), per-state results at `baselines/next_category/results/{state}.json`). Faithful baseline ports landed 2026-04-27 (commit `6e3cd49`).

| Baseline | Type | Source | Use |
|---|:-:|---|---|
| **Majority-class** | simple floor | P0 per-state stats | table floor row |
| **Markov-1-POI** | simple floor | P0 per-state stats | table floor row |
| **POI-RGNN** (Capanema et al. 2019, reproduced) | external published | `docs/baselines/BASELINE.md` + `docs/baselines/POI_RGNN_AUDIT.md` + `baselines/next_category/poi_rgnn.md` + `baselines/next_category/results/<state>.json` | primary external comparison; same Gowalla state-level partition, same 7-category taxonomy. Our +28–32 pp delta on FL is a conservative lower bound (POI-RGNN reproduction used non-user-disjoint folds — see audit). |
| **MHA+PE** (Zeng et al. 2019) | external published | `baselines/next_category/mha_pe.md` + per-state JSONs | secondary external comparison. |
| **STL Check2HGI cat** (matched-substrate STL) | internal ceiling | Phase-1 `next_gru` matched-head: `results/phase1_perfold/{AL,AZ}_check2hgi_cat_gru_5f50ep.json` (post-2026-04-27); legacy `next_single` at `P1_5b/*` (head-sensitivity row, post-CONCERNS C17) | the "MTL lifts STL" comparison; **headline cat head: `next_gru`** (matched-head MTL B3) |
| **STL HGI cat** | substrate ablation (CH16) | Phase-1 matched-head + 4-head sweep at AL+AZ landed 2026-04-27 (`results/phase1_perfold/{AL,AZ}_hgi_cat_*_5f50ep.json`); FL/CA/TX queued in `PHASE2_TRACKER.md` | proves substrate claim Check2HGI > HGI on cat. **Phase-1 confirms head-invariance: 8/8 head-state probes positive at p=0.0312, Δ +11.58 to +15.50 pp.** |

### 3.2 next_region (~1 K (AL) / ~1.5 K (AZ) / ~4.7 K (FL/CA/TX) classes, Acc@10 primary, MRR + Acc@5 secondary)

> **Audit hub:** `baselines/next_region/` ([README](../../studies/check2hgi/baselines/README.md), [comparison.md](../../studies/check2hgi/baselines/next_region/comparison.md), per-baseline files [stan.md](../../studies/check2hgi/baselines/next_region/stan.md), [rehdm.md](../../studies/check2hgi/baselines/next_region/rehdm.md), per-state results at `baselines/next_region/results/{state}.json`). Faithful baseline ports + Phase-1 matched-head + MTL counterfactual rows landed 2026-04-27.

| Baseline | Type | Source | Use |
|---|:-:|---|---|
| **Random** | theoretical floor | table anchor only | |
| **Majority** | simple floor | P0 | table floor row |
| **Top-K popular** | simple floor | P0 | table floor row |
| **Markov-1-region** | classical prior | P0 (AL/FL/AZ done; CA/TX pending) | **primary simple floor.** Binds hard on FL (4.7 K region scale — Markov-saturated; see §6). |
| **Markov-k-region, k=2,…,9** | classical priors | P0 (AL/FL done for k=2..9) | demonstrates monotone degrade with k; used to isolate "neural over equal-context Markov" (CH-M7). |
| **STL GRU** | neural ceiling (secondary) | P1 (AL/FL done, AZ implicit) + `baselines/next_region/comparison.md` | literature-aligned with HMT-GRN; 2nd-best STL after STAN. |
| **STL STAN** (Luo WWW'21 adapt) | head-sensitivity probe (was: neural ceiling primary) | P1 AL + AZ done; **FL still pending (F6)**. `baselines/next_region/stan.md` + per-state JSONs. **CH15 reframed as head-coupled (CONCERNS §C16) — preserved as head-sensitivity row, not headline.** | preserved as the "head-coupled CH15" probe row; not the headline (matched-head reg head is `next_getnext_hard`). |
| **STL `next_getnext_hard`** (matched-head MTL reg head) | **headline matched-head STL** | Phase-1 Leg II.2: `results/phase1_perfold/{AL,AZ}_{check2hgi,hgi}_reg_gethard_5f50ep.json` (post-2026-04-27); FL pending **F37** (4050-assigned). | **headline reg STL**: matched to MTL B3 reg head. Used by both Phase-1 (substrate Δ) and F49 (vs MTL architectural). |
| **REHDM** (faithful port) | external concept-aligned | `baselines/next_region/rehdm.md` + per-state JSONs | external comparison; Phase-1 finding: ReHDM STL underperforms STAN STL by 20–37 pp at our protocol. |
| **HMT-GRN, MGCL** | concept-aligned, different dataset | cited, not direct-comparable (CH10) | framing anchor, not paper table row. |

### 3.3 Important: GETNext is NOT a baseline

GETNext-hard (`next_getnext_hard` = STAN + `α · log_T[last_region_idx]`) is part of B3's region head, not a comparison method. The relevant comparison decomposition is:

| Comparison | What it measures |
|---|---|
| MTL-B3 vs STL STAN | Does MTL + graph prior together beat the STL neural ceiling? (Currently: AZ reg Acc@10 tied, AL tied within σ, FL pending.) |
| MTL-B3 vs STL GETNext-hard (F21) | Does MTL **coupling** help on top of the GETNext head alone? (Currently unmeasured. F21 fills this.) |
| MTL-B3 vs Markov-1-region | Does the neural model beat the classical prior? (AL/AZ yes; FL approaches saturation.) |
| STL GETNext-hard vs STL STAN | Does adding the graph prior to STAN help single-task? (Unmeasured. F21 by-product.) |

F21 (STL GETNext-hard per state) is the new critical baseline item. Cheapest: reuse the existing MTL training infrastructure but with `--task next` single-task flow.

---

## 4 · STL-baseline matching policy

Compare MTL-B3 vs STL baselines along two axes:
1. **Matched-head STL** — uses the same head class as B3's corresponding task head. Honest "MTL vs STL" comparison.
2. **Literature-aligned STL** — uses the published strong single-task architecture. Reference ceiling.

| Task | Matched-head STL (post-F27, post-Phase-1) | Head-sensitivity probe (former matched-head) | Literature-aligned STL |
|---|---|---|---|
| next-category | **`next_gru`** — matches MTL B3's task_a head exactly. AL+AZ Phase-1: C2HGI 40.76/43.21, HGI 25.26/28.69, Δ=+15.50/+14.52 pp p=0.0312 each. | `next_single` (existing P1_5b — Δ=+18.30 pp at AL) and `next_lstm` — preserve as C2 head-sensitivity rows. | **POI-RGNN** (external). |
| next-region | **`next_getnext_hard`** (F21c + Phase-1 HGI side). AL+AZ: C2HGI 68.37/66.74, HGI 67.52/64.40 (AL TOST non-inf, AZ +2.34 pp p=0.0312). | STAN (existing CH15-style — preserved as head-sensitivity row showing the head-coupled flip). | **STL STAN** (AL/AZ/FL all done in `baselines/next_region/comparison.md`). |

**Paper caveat to state explicitly:** "Our STL next-category baseline uses the `next_mtl` Transformer head rather than the MTLnet framework's internal `CategoryHeadMTL` multi-path ensemble. These are both small-head, last-token-softmax classifiers over a sequence encoder; the architectural distance is small. The MTL figure in the paper reports the version of the comparison where both heads are matched to the MTLnet architecture."

---

## 5 · Table layout (see `results/RESULTS_TABLE.md` for the live populated version)

Per-state headline (FL / CA / TX), one block per state:

```
=== Florida ===

Baselines
| Method          | cat F1 | reg Acc@10 | reg Acc@5 | reg MRR |
| Majority        |    –   |  22.25     |   –       |   –     |
| Markov-1-region |    –   |  65.05     |   –       |   –     |
| POI-RGNN        |  34.49 |    –       |   –       |   –     |

Single-Task (STL)
| Method                    | cat F1 | reg Acc@10 | ... |
| STL Check2HGI cat (matched)       |  X.XX  |    –       |     |
| STL STAN (reg ceiling)             |   –    |   X.XX     |     |
| STL GETNext-hard (matched-head)    |   –    |   X.XX     |     |
| STL HGI cat (substrate ablation)   |  X.XX  |    –       |     |

Multi-Task (MTL)
| Method                     | cat F1 | reg Acc@10 | ... |
| MTL-B3 (cross-attn + static + GETNext-hard) | X.XX | X.XX | ... |

Ablation rows (optional, can live in appendix)
| MTL-soft (same arch, soft probe)          | X.XX | X.XX | |
| MTL-hard + pcgrad                          | X.XX | X.XX | |
```

Then a cross-state "best of each" summary:

```
=== Cross-state (paper headline) ===

| State | cat F1 champion | reg Acc@10 champion | MTL-B3 joint row |
| FL    | B3 = X.XX       | STL STAN = X.XX      | X.XX / X.XX       |
| CA    | B3 = X.XX       | ...                   | ...               |
| TX    | B3 = X.XX       | ...                   | ...               |
```

Ablation tables (AL + AZ + FL-1f) live under a separate heading and are not required for the main claim, just for method verification / mechanism.

---

## 6 · FL next-region Markov caveat — approach (a)

**Decision:** FL next-region Acc@10 is a "Markov-saturated regime" and we acknowledge it as a limitation.

- Markov-1-region on FL = 65.05.
- STL GRU = 68.33 (+3.28 pp).
- STL STAN at FL = pending, expected in 66–70 range by the AL/AZ pattern.
- MTL-B3 (n=1, two replicates): 58.88 / 66.55 Acc@10.

**Paper framing (approach (a)):** "On dense-data state splits where Markov-1-region transitions cover ≥85 % of validation rows (e.g., Florida Gowalla, 127 K check-ins), the classical 1-gram prior is near-optimal for Acc@10 on short horizons. Our neural models achieve +3 to +5 pp improvements on secondary metrics (Acc@5, MRR) that reflect finer-grained ranking, but do not clear Markov-1 on Acc@10 at this scale."

This is revisited after CA + TX 5-fold data lands. If CA and TX also show Markov-saturation on Acc@10, the caveat is a genuine scale-dependent property and is stated as such. If CA and TX show the neural models clearing Markov (as AL / AZ do), the caveat stays FL-specific.

---

## 7 · Objective → evidence map

| Objective | Evidence state | Binding work to close |
|---|:-:|---|
| **CH16** — Check2HGI > HGI on cat F1 | 🟢 **AL+AZ matched-head + head-invariant (8/8 probes p=0.0312, Phase-1 closed 2026-04-27)** · 🔴 FL/CA/TX | F36 (FL Phase-2 grid) + F38 (CA) + F40 (TX) — see `PHASE2_TRACKER.md`. |
| **CH15** — Check2HGI ≥ HGI on reg under matched MTL head | 🟢 **AZ +2.34 pp p=0.0312, AL TOST non-inf at δ=2 pp (Phase-1)** · 🔴 FL/CA/TX | F36c (FL reg STL). Reframed from prior CH15 (head-coupled). |
| **CH18** — MTL B3 substrate-specific | 🟢 **AL+AZ (cat −17 pp / reg −30 pp under HGI substitution, Phase-1)** · 🔴 FL/CA/TX | F36d (FL MTL counterfactual). |
| **CH19** — Per-visit mechanism (~72%) | 🟢 **AL (Phase-1)** · 🟡 FL/CA/TX optional | F41 (FL extension only if reviewer asks). |
| **MTL > baselines — cat** | 🟢 clean everywhere we have data (AL/AZ/FL-1f) | CA/TX headline runs close this. |
| **MTL > baselines — reg** | 🟢 AL/AZ (beat Markov by 10+ pp) · 🟡 FL (Markov saturation — approach (a)) | Acknowledge approach (a). |
| **MTL > STL — cat** | 🟢 AZ (+3.73 pp post-F27, Wilcoxon p=0.0312 on cat F1/Acc@1/reg MRR, 5/5 folds) · 🟢 AL (+4.13 pp post-F27/F31) · 🟢 FL-1f (+2–3 pp) | FL 5f σ (F33 Colab). |
| **MTL > STL — reg (vs STL STAN ceiling)** | 🟢 AL post-F27 (+0.40 pp over STAN, first cross) · 🟡 AZ (tied σ; +3.75 pp reg macro-F1 at p=0.0312) · 🔴 FL (trails) | F6 (FL STL STAN 5f) + FL 5f MTL. |
| **MTL coupling vs matched-head STL (F21c → H3-alt)** | 🟢 **RESOLVED by H3-alt 2026-04-26 + F49 attribution 2026-04-27.** AL: MTL-H3-alt EXCEEDS STL F21c by **+6.25 pp** ✓. AZ: closes 75% of B3 gap. FL: beats Markov+STL GRU; absolute Δ vs STL F21c pending F37. F49 attributes the lift as architecture-dominant (AL +6.48 pp from architecture alone, ~2.7σ). CH18 → Tier A; CH20 added Tier A. | Status now green. F37 closes Layer 3. |
| **MTL B3 substrate-specific (CH18 Phase-1)** | 🟢 AL+AZ confirmed — MTL+HGI breaks reg by 30 pp (cat −17 pp). MTL+HGI is *worse than STL+HGI* on reg by ~37 pp at AL. See `research/SUBSTRATE_COMPARISON_FINDINGS.md §3`. | Phase-2 FL/CA/TX in `PHASE2_TRACKER.md`. |
| **CH16 head-invariant (Phase-1 + C2)** | 🟢 AL+AZ — 8/8 head-state probes positive at p=0.0312 (linear/next_gru/next_single/next_lstm); Δ +11.58 to +15.50 pp. CH15 reframed as head-coupled. | Phase-2 cross-state replication. |
| **Per-visit-context mechanism (CH19 Phase-1 C4)** | 🟢 AL — POI-pooled C2HGI counterfactual: per-visit context = ~72% of cat substrate gap; training signal residual = ~28%. | Phase-2 C4 extension to FL is optional. |
| **Joint claim — architecture × substrate (CH21)** | 🟢 **TOP-LINE PAPER CLAIM (Tier A) 2026-04-27.** Synthesis of CH18 (substrate-specific) + CH19 (per-visit mechanism) + CH20 (architecture-dominant). The MTL win is interactional, not transfer. See `CLAIMS_AND_HYPOTHESES.md §CH21` + `SESSION_HANDOFF_2026-04-27.md §0.3`. | Paper section drafting (already on `PAPER_PREP_TRACKER.md §5`). |

---

## 8 · What this doc is NOT

- Not the live tracker. See `FOLLOWUPS_TRACKER.md`.
- Not the results table. See `results/RESULTS_TABLE.md`.
- Not the mechanism write-up. See `research/B5_FL_TASKWEIGHT.md` + `research/B5_MACRO_ANALYSIS.md` + `research/GETNEXT_FINDINGS.md`.
- Not the critical review. See `review/2026-04-23_critical_review.md`.

This file defines scope and structure; the cited files hold the evidence.
