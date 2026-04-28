# Check2HGI Follow-ups Tracker

**Created:** 2026-04-23. **Updated 2026-04-27** after Phase-1 substrate validation. Phase-2 (FL+CA+TX) follow-ups live in [`baselines/PHASE2_TRACKER.md`](baselines/PHASE2_TRACKER.md) — this tracker now points there for substrate-comparison work.

> **Phase 1 closed 2026-04-27.** F3, F9, F26, F21c-FL deferred — superseded by Phase-2 tracker which replicates the full 3-leg grid (substrate probe + matched-head STL + MTL counterfactual) at FL/CA/TX. See `baselines/PHASE1_VERDICT.md` for outcome.

**Champion:** `mtlnet_crossattn + static_weight(category_weight=0.75) + next_getnext_hard d=256, 8h`. See [`NORTH_STAR.md`](NORTH_STAR.md) and [`PAPER_STRUCTURE.md`](PAPER_STRUCTURE.md).

**Convention.** One row per open work item. Status ∈ `{pending, in_progress, blocked, done, gated, retired}`. Priority P1 (blocker) → P4 (low). Owner defaults to `m4_pro` (this machine); `m2_pro`, `4050`, `any` also valid. Cost is wall-clock estimate at the stated hardware.

**Read before working:**
1. [`NORTH_STAR.md`](NORTH_STAR.md) — champion config + re-evaluation triggers.
2. [`PAPER_STRUCTURE.md`](PAPER_STRUCTURE.md) — paper scope + baselines + tables.
3. [`review/2026-04-23_critical_review.md`](review/2026-04-23_critical_review.md) — analytical state.

---

## 1 · Ready-now (P1/P2 — paper-blocking)

| # | Pri | Item | Obj | Owner | Cost | Acceptance criterion | Why |
|---|:-:|---|:-:|:-:|:-:|---|---|
| **F21c** | **P1** | STL GETNext-hard 5f per state (strict matched-head reg baseline) ⭐ | 2b | m4_pro | done 2026-04-24 | **AL: 68.37 ± 2.66 Acc@10 · AZ: 66.74 ± 2.11 Acc@10.** STL-with-graph-prior **dominates MTL-B3 by 12-14 pp on reg** at both ablation states. Paper-reshaping finding in `research/F21C_FINDINGS.md`. FL 5f not launched — awaiting headline-path decision. |
| **F27** | **P1** | Cat-head ablation + 5-fold validation on AZ | — | m4_pro | **done 2026-04-24** | 7-config 1f screen (research/F27_CATHEAD_FINDINGS.md) → `next_gru` winner → AZ 5f × 50ep paired Wilcoxon p=0.0312 on cat F1 / cat Acc@1 / reg MRR (5/5 folds +). B3's cat head **swapped `next_mtl` → `next_gru`**. AZ cat F1 0.4362 → 0.4581. |
| **F31** | **P1** | **B3 (post-F27) AL 5f × 50ep validation** | — | m4_pro | ~15 min MPS | Re-run AL with `--cat-head next_gru`. Pass: cat F1 ≥ 0.3928 (old B3) AND reg Acc@10 ≥ 0.5633 (old B3), both within σ. | Verify the F27 cat-head choice generalises off AZ before adopting it universally. |
| **F32** | **P2** | **B3 (post-F27) FL 1f × 50ep re-check** | — | m4_pro | **done 2026-04-24 04:05** | Flipped sign vs AL/AZ: cat F1 = 0.6572 (−0.93 pp vs pre-F27 mean), cat Acc@1 = 0.6860 (−0.43 pp), reg Acc@10 = 0.6526 (−0.93 pp). All within n=1 noise but direction is opposite. **Scale-dependence flag raised** — see `research/F27_CATHEAD_FINDINGS.md §Validation outcomes`. |
| **F33** | **P1** | **FL 5f × 50ep B3+next_gru (decisive F27 FL resolution)** ⭐ | 2b | **colab** | ~6 h Colab T4 | Run MTL-B3 on FL with `--cat-head next_gru` at 5 folds. Compare to the two pre-F27 FL n=1 points (cat F1 0.6623 / 0.6706). Pass: cat F1 5f-mean within σ of 0.6623-to-0.6706 envelope → Path A (universal next_gru). Fail: cat F1 below envelope → Path B (scale-dependent cat head). | Scale-dependence was flagged by F32. FL is the paper's headline state — 5-fold σ is the binding criterion for committing the B3 champion universally. Colab run (not M4 Pro) to avoid the OOM/kill history at FL scale. |
| **F34** | **P1** | **CA upstream pipeline + CA 1f × 50ep B3+next_gru** | headline | **colab** | ~6–12 h Colab T4 (pipeline + 1f train) | Steps: (a) Check2HGI embedding training on CA, (b) `compute_region_transition.py --state california`, (c) `create_inputs_check2hgi.pipe.py --state california`, (d) B3+gru 1f×50ep training. Land `results/headline/california/fl_1f50ep_b3_gru.json`. | First CA data point. Also tests whether F27's cat-head choice is scale-stable beyond FL (CA ~500K check-ins). |
| **F35** | **P1** | **TX upstream pipeline + TX 1f × 50ep B3+next_gru** | headline | **colab** | ~6–12 h Colab T4 (pipeline + 1f train) | Same as F34 but TX. | First TX data point. |
| **F3** | — | (✅ closed 2026-04-27 by Phase-1) AZ HGI STL cat | 1 | — | done | Phase-1 superseded F3 with matched-head `next_gru` evidence: AZ Δ=+14.52 pp p=0.0312. See `baselines/PHASE1_VERDICT.md` §2.1. |
| **F22** | — | (retired — merged into F34) CA upstream pipeline | — | — | — | CA upstream moved to F34 (Colab). | |
| **F23** | — | (retired — merged into F35) TX upstream pipeline | — | — | — | TX upstream moved to F35 (Colab). | |
| **F24** | **P2** | CA 5f headline (after F34 1f confirms config) | headline | colab/any | ~20–25 h | Full baselines + STL + MTL at 5f × 50ep seed 42. Gated on F34. | Part of the paper's primary table; launch only after F34 1f shows the config works on CA data. |
| **F25** | **P2** | TX 5f headline (after F35 1f confirms config) | headline | colab/any | ~20–25 h | Same as F24 but TX. Gated on F35. | |
| **F4** | **P1** | FL MTL-B3 5-fold (clean re-run) | headline | m4_pro / any | ~6–8 h MPS | Replace F17 partial result with a clean n=5 FL run of B3. With F20 per-fold persistence, partial progress survives crashes. | FL headline needs real σ. F17 attempt was user-killed; retry. |
| **F9** | — | (✅ moved into PHASE2_TRACKER F36b) FL HGI STL cat 5f | 1 | m4_pro | — | Phase-2 tracker covers FL HGI STL cat as part of the matched-head Phase-2 grid. See `baselines/PHASE2_TRACKER.md §F36b`. |
| **F36–F40** | **P1** | Phase 2 (FL + CA + TX) substrate-comparison grid | headline | m4_pro | ~30 h × 3 states + upstream pipelines | See [`baselines/PHASE2_TRACKER.md`](baselines/PHASE2_TRACKER.md) for full per-state grid: substrate probe + cat STL × 2 substrates + reg STL × 2 substrates + MTL counterfactual. | Phase 2 of substrate-comparison plan. Authorised by `PHASE1_VERDICT.md §6` (strong claim confirmed at AL+AZ). |

---

## 2 · Gated

| # | Pri | Item | Gate |
|---|:-:|---|---|
| **F5** | P4 | FL MTL-GETNext-soft 5-fold | Only if the paper explicitly needs to compare soft vs B3 at FL 5-fold. Currently covered by the F2 B3-vs-soft comparison at n=1 (B3 Pareto-dominates); 5f replication is nice-to-have, not blocking. |

---

## 3 · Deferred (follow-up paper or post-champion-freeze)

| # | Item | Cost | When to revisit |
|---|---|:-:|---|
| **F8** | **Multi-seed n=3 on champion configs** | **~20 h MPS total, parallelisable** | **Deferred by user 2026-04-23.** Held until headline (FL/CA/TX) completes + B3 is frozen. Seeds {42, 123, 2024} × 5 folds × headline configs. |
| F12 | Per-fold transition matrix (leakage-safe GETNext) | ~4 h impl + 2 h reruns | Camera-ready, if reviewer asks for per-fold protocol. |
| F13 | GETNext with true flow-map Φ = (Φ₁1ᵀ + 1Φ₂ᵀ) ⊙ (L̃ + J) | ~30 min impl + rerun | Follow-up paper. |
| F14 | PIF-style user-specific region frequency prior | ~3–4 h | Follow-up paper. |
| F15 | TGSTAN + STA-Hyper full reproductions | ~2–3 weeks | Follow-up paper. |
| F16 | Encoder enrichment (P6: temporal / spatial / graph features) | multi-week track | Post-paper research direction. |
| F26 | AZ HGI STL cat for CH16 | ~3 h | Nice-to-have at headline states (CA/TX). Deferred unless reviewer asks. |
| **F21a** | **DROPPED** FL STL STAN 5f | — | Dropped 2026-04-24 by user decision — STAN is our reg baseline reference, not an ablation data point we need to publish. |
| **F21b** | Archived — STL GETNext (soft) 5f per state | ~12 h total | User-deprioritised 2026-04-24. Would give a "graph prior alone" reference vs hard, but not blocking any headline claim. Revisit if reviewer asks specifically about soft-vs-hard at the STL level. |
| **F27** | Cat-head ablation (try `next_gru`, `next_stan`, ensemble, etc. as task_a heads on MTL-B3) | ~1 h 1-fold AZ sweep + 5f runs for any winner | Follow-up / camera-ready if a reviewer asks "did you try other cat heads". Current `CategoryHeadTransformer` has not been head-ablated. Cost-benefit today: cat isn't the bottleneck, reopening the cat-head choice could break the committed north-star. |

---

## 4 · Done (audit trail, this push 2026-04-21 → 2026-04-23)

| Item | Finished | Evidence |
|---|:-:|---|
| `MTL_PARAM_PARTITION_BUG` fix + 6 contaminated reruns | 2026-04-22 | commits `5668856` `c1c7f3e`; `results/P5_bugfix/SUMMARY.md` |
| `CROSSATTN_PARTIAL_FORWARD_CRASH` fix | 2026-04-22 | commit `8afc9ac`; `tests/test_regression/test_mtlnet_crossattn_partial_forward.py` |
| B5 hard-index GETNext implementation + AL + AZ 5f retraining + FL 1f | 2026-04-22 | commits `6a2f808` `ea65fb3`; `results/B5/*.json`; `research/B5_RESULTS.md` |
| ALiBi × GETNext on AL (B7) | 2026-04-22 | `research/B7_ALIBI_GETNEXT_FINDINGS.md` |
| α-inspection (GETNext / TGSTAN / STA-Hyper, AL+AZ) | 2026-04-21 | `research/GETNEXT_FINDINGS.md §α inspection` |
| PCGrad-vs-static attribution on GETNext | 2026-04-22 | `research/ATTRIBUTION_PCGRAD_VS_STATIC.md` |
| North-star committed initially to `GETNext-soft` | 2026-04-23 | `NORTH_STAR.md` (pre-F2) |
| Critical review post-B5 | 2026-04-23 | `review/2026-04-23_critical_review.md` |
| **F2** FL-hard training-pathology diagnosis (4 phases, ~2h37m) | 2026-04-23 03:18 | `research/B5_FL_TASKWEIGHT.md` · B3 identified as new champion candidate · Pareto-dominates soft at FL n=1 on all 4 joint metrics |
| **F1** AZ Wilcoxon B-M9b soft vs B-M9d hard | 2026-04-23 | `research/B5_AZ_WILCOXON.md` · Every region metric p=0.0312 (n=5 minimum) · `scripts/analysis/az_wilcoxon.py` |
| **F11** MTLoRA post-fix tables update | 2026-04-23 | `results/BASELINES_AND_BEST_MTL.md` + `results/RESULTS_TABLE.md` |
| **F7** POI-RGNN protocol audit | 2026-04-23 | `docs/baselines/POI_RGNN_AUDIT.md` |
| **F10** CLAIMS_AND_HYPOTHESES dashboard reconciled | 2026-04-23 | `CLAIMS_AND_HYPOTHESES.md §Summary dashboard` |
| **F18** B3 5f AL | 2026-04-23 03:39 | `results/B3_validation/al_5f50ep_b3.json` · cat F1 +0.78 pp over B-M6e · no regression |
| **F19** B3 5f AZ | 2026-04-23 04:05 | `results/B3_validation/az_5f50ep_b3.json` · cat F1 +1.40 pp vs B-M9d, +1.54 pp vs STL STAN · Pareto-dominates soft (+0.80 cat, +6.10 reg) |
| **F19-followup** B3 Wilcoxon vs STL at AZ | 2026-04-23 | `research/B3_AZ_WILCOXON_VS_STL.md` · cat F1 +1.65 pp p=0.0312 (strict MTL-over-STL) · reg macro-F1 +3.75 pp p=0.0312 · reg Acc@10 tied · reg MRR −7.94 pp (STL wins) |
| **F20** per-fold persistence in `MLHistory` | 2026-04-23 06:35 | `src/tracking/storage.py::save_fold_partial` + `src/tracking/experiment.py::step` hook · 4 regression tests + 186 full-suite green · validated in vivo (F17 fold 1 persisted despite user-kill) |
| Doc cleanup — archive pre-B3 framing + research, create `PAPER_STRUCTURE.md` | 2026-04-23 | `archive/pre_b3_framing/`, `archive/research_pre_b3/`, `archive/phases_original/`, new `PAPER_STRUCTURE.md` |

---

## 5 · Progress snapshot (2026-04-23 evening)

| Objective | AL | AZ | FL | CA | TX | What closes |
|---|:-:|:-:|:-:|:-:|:-:|---|
| **Obj 1** CH16 (C2HGI > HGI cat, matched-head head-invariant) | 🟢 **+15.50 pp p=0.0312 (Phase-1)** | 🟢 **+14.52 pp p=0.0312 (Phase-1)** | 🔴 | 🔴 | 🔴 | F36 (FL Phase-2) + F38 (CA) + F40 (TX) |
| **CH15 reframed** (C2HGI ≥ HGI reg, matched MTL head) | 🟢 **AL TOST non-inf (Phase-1)** | 🟢 **AZ +2.34 pp p=0.0312 (Phase-1)** | 🔴 | 🔴 | 🔴 | F36c (FL reg STL) |
| **CH18** MTL B3 substrate-specific | 🟢 **AL Δ_cat +16.75 / Δ_reg +29.65 (Phase-1)** | 🟢 **AZ Δ_cat +17.11 / Δ_reg +31.72 (Phase-1)** | 🔴 | 🔴 | 🔴 | F36d (FL MTL CF) |
| **CH19** Per-visit mechanism (~72%) | 🟢 **AL (Phase-1)** | 🟡 not extended | 🟡 optional | 🟡 | 🟡 | F41 (FL extension only if asked) |
| **Obj 2a** MTL > baselines (cat) | 🟢 | 🟢 | 🟢 n=1 | 🔴 | 🔴 | F24 + F25 |
| **Obj 2a** MTL > baselines (reg) | 🟢 +10.95 pp | 🟢 +10.29 pp | 🟡 Markov saturation (approach a) | 🔴 | 🔴 | F24 + F25 |
| **Obj 2b** MTL > STL (cat) | 🟡 tied | 🟢 +1.65 pp p=0.0312 | 🟢 n=1 +0.22 | 🔴 | 🔴 | F24 + F25 + F4 |
| **Obj 2b** MTL > STL (reg) | 🟡 σ-tied | 🟢 reg F1 +3.75 pp p=0.0312, Acc@10 tied | 🟡 n=1 +5.20 pp | 🔴 | 🔴 | F4 + F6 + F24 + F25 |
| **NEW** MTL coupling > head alone | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | **F21 per state** |

---

## 6 · Recommended execution order (P1 first)

**On m4_pro (this machine):**
1. **F21 AL + AZ + FL** — matched-head STL baselines (~3+3+6 = ~12 h sequential). Highest information gain per hour.
2. **F3 AZ HGI STL cat** (~3 h) — cheap CH16 extension.
3. **F4 FL MTL-B3 5f clean** (~6–8 h) — headline FL σ.
4. **F6 FL STL STAN 5f** (~5–6 h) — region ceiling row at FL.

**On m2_pro (in parallel, from kickoff):**
1. **F22 CA pipeline** (~4–8 h) → **F24 CA headline** (~20–25 h).
2. **F23 TX pipeline** (~4–8 h) → **F25 TX headline** (~20–25 h).

**On 4050 (if available):** overflow from m4_pro queue.

The m2_pro path is the critical-path wall-clock: CA and TX upstream + headline = ~50–70 h each. m4_pro fills in FL-specific runs + matched-head STLs + Obj 1 replication in parallel.
