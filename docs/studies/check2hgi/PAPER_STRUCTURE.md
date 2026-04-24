# Paper Structure — Single Source of Truth

**Created:** 2026-04-23. **Last updated:** 2026-04-24 (post-F27 cat-head swap + F21c matched-head finding). **Owner:** this file defines the paper's table layout, baseline set, STL matching policy, and scope decisions. All other docs reference this.

---

## 1 · Champion model (`NORTH_STAR.md`)

**B3** — a single MTL configuration that produces both heads jointly:

```
architecture   : mtlnet_crossattn
mtl_loss       : static_weight(category_weight = 0.75)   # reg weight = 0.25
task_a head    : next_gru                                 # ← updated 2026-04-24 (F27), was next_mtl
task_b head    : next_getnext_hard                        # STAN + α · log_T[last_region_idx]
task_a input   : check-in embeddings (9-step window)
task_b input   : region embeddings (9-step window)
hparams        : d_model=256, 8 heads, max_lr=0.003, batch=2048, 50 epochs, seed 42
```

### 1.1 · Paper-reshaping findings since 2026-04-23

- **F27 (2026-04-24)** — cat-head swap `next_mtl → next_gru` lifts cat F1 by +3.43 pp (AL 5f) and +2.37 pp (AZ 5f, Wilcoxon p=0.0312 on 3 metrics). **FL 1f flipped sign (−0.93 pp)** → scale-dependence flag open; resolved by F33 (Colab FL 5f). See `research/F27_CATHEAD_FINDINGS.md`.
- **F21c (2026-04-24)** — matched-head STL `next_getnext_hard` (graph-prior alone, single-task) **outperforms MTL-B3 on reg Acc@10 by 12–14 pp on AL + AZ**. The MTL coupling does not add value beyond the head choice at ablation-state scale. See `research/F21C_FINDINGS.md`. Paper framing shifts to Option B: matched-head STL baselines are a methodological finding; MTL-B3 is positioned as the joint-single-model deployment option, not the reg SOTA. CH18 in the claim catalog captures this.

Validation status (as of 2026-04-23):

| State | Status | Source |
|---|---|---|
| AL 5f | ✅ validated | `results/B3_validation/al_5f50ep_b3.json` |
| AZ 5f | ✅ validated | `results/B3_validation/az_5f50ep_b3.json` |
| FL 1f | ✅ two replicates | `results/F2_fl_diagnostic/fl_1f50ep_hard_static_cat0.75.json` + `results/check2hgi/florida/mtlnet_*_20260423_0630/folds/fold1_info.json` |
| FL 5f | 🔴 pending | (F17 attempt killed at fold 2; needs re-run) |
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

| Baseline | Type | Source | Use |
|---|:-:|---|---|
| **Majority-class** | simple floor | P0 per-state stats | table floor row |
| **Markov-1-POI** | simple floor | P0 per-state stats | table floor row |
| **POI-RGNN** (Capanema et al. 2019, reproduced) | external published | `docs/baselines/BASELINE.md` + `docs/baselines/POI_RGNN_AUDIT.md` | primary external comparison; same Gowalla state-level partition, same 7-category taxonomy. Our +28–32 pp delta on FL is a conservative lower bound (POI-RGNN reproduction used non-user-disjoint folds — see audit). |
| **STL Check2HGI cat** (matched-substrate STL) | internal ceiling | `P1_5b/*` for AL/AZ; pending for FL/CA/TX | the "MTL lifts STL" comparison; uses `next_mtl` Transformer head (matched-class to `CategoryHeadMTL`) |
| **STL HGI cat** | substrate ablation (CH16) | AL only so far; F3 adds AZ, F9 adds FL, CA/TX nice-to-have | proves substrate claim Check2HGI > HGI on cat |

### 3.2 next_region (~1 K (AL) / ~1.5 K (AZ) / ~4.7 K (FL/CA/TX) classes, Acc@10 primary, MRR + Acc@5 secondary)

| Baseline | Type | Source | Use |
|---|:-:|---|---|
| **Random** | theoretical floor | table anchor only | |
| **Majority** | simple floor | P0 | table floor row |
| **Top-K popular** | simple floor | P0 | table floor row |
| **Markov-1-region** | classical prior | P0 (AL/FL/AZ done; CA/TX pending) | **primary simple floor.** Binds hard on FL (4.7 K region scale — Markov-saturated; see §6). |
| **Markov-k-region, k=2,…,9** | classical priors | P0 (AL/FL done for k=2..9) | demonstrates monotone degrade with k; used to isolate "neural over equal-context Markov" (CH-M7). |
| **STL GRU** | neural ceiling (secondary) | P1 (AL/FL done, AZ implicit) | literature-aligned with HMT-GRN; 2nd-best STL after STAN. |
| **STL STAN** (Luo WWW'21 adapt) | **neural ceiling (primary)** | P1 AL + AZ done; **FL pending (F6)** | "STL SOTA" ceiling; primary comparand for MTL-beats-STL claim. |
| **STL GETNext-hard** (matched-head STL) — **NEW in tracker F21** | **head-choice ablation** | pending — per state | isolates the MTL-coupling contribution. MTL-B3 vs STL-GETNext-hard measures what joint training adds beyond the shared head. |
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

| Task | Matched-head STL | Literature-aligned STL |
|---|---|---|
| next-category | `next_mtl` (Transformer) — proxy for `CategoryHeadMTL`. Both are "small 7-class classifiers over a sequence encoder"; the architectural distance is small. Current STL cat numbers (AL 38.58, AZ 42.08, FL 63.17 n=1) use `next_mtl`. | **POI-RGNN** (external). |
| next-region | **STL GETNext-hard** (F21, pending). | **STL STAN** (AL/AZ done; FL pending F6). |

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
| **CH16** — Check2HGI > HGI on cat F1 | 🟢 AL (+18.30 pp σ-clean) · 🔴 AZ/FL missing | F3 (AZ HGI cat 5f) + F9 (FL HGI cat 5f). CA/TX nice-to-have. |
| **MTL > baselines — cat** | 🟢 clean everywhere we have data (AL/AZ/FL-1f) | CA/TX headline runs close this. |
| **MTL > baselines — reg** | 🟢 AL/AZ (beat Markov by 10+ pp) · 🟡 FL (Markov saturation — approach (a)) | Acknowledge approach (a). |
| **MTL > STL — cat** | 🟢 AZ (+3.73 pp post-F27, Wilcoxon p=0.0312 on cat F1/Acc@1/reg MRR, 5/5 folds) · 🟢 AL (+4.13 pp post-F27/F31) · 🟢 FL-1f (+2–3 pp) | FL 5f σ (F33 Colab). |
| **MTL > STL — reg (vs STL STAN ceiling)** | 🟢 AL post-F27 (+0.40 pp over STAN, first cross) · 🟡 AZ (tied σ; +3.75 pp reg macro-F1 at p=0.0312) · 🔴 FL (trails) | F6 (FL STL STAN 5f) + FL 5f MTL. |
| **MTL coupling vs matched-head STL (F21c)** | 🔴 AL/AZ: **STL GETNext-hard beats MTL-B3 by 12–14 pp Acc@10** (see CH18) · 🔴 FL unmeasured | Drives Option B reframing; FL F21c pending. |

---

## 8 · What this doc is NOT

- Not the live tracker. See `FOLLOWUPS_TRACKER.md`.
- Not the results table. See `results/RESULTS_TABLE.md`.
- Not the mechanism write-up. See `research/B5_FL_TASKWEIGHT.md` + `research/B5_MACRO_ANALYSIS.md` + `research/GETNEXT_FINDINGS.md`.
- Not the critical review. See `review/2026-04-23_critical_review.md`.

This file defines scope and structure; the cited files hold the evidence.
