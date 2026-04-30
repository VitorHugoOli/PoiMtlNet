# Check2HGI Study — Comprehensive Overview & Forgotten-items Audit

**Date:** 2026-04-27. **Author:** session compilation. **Purpose:** synthesize where the study stands, the live work queues (`PAPER_PREP_TRACKER.md` + `FOLLOWUPS_TRACKER.md`), and a faithful cross-reference against the archived original P0–P7 phase plans (`archive/phases_original/`) to surface any task or action that may have been forgotten or lost in the 2026-04-23 reorganisation.

---

## 1 · Where we stand right now

**Champion config:** `F48-H3-alt` per-head LR (2026-04-26). Single-line recipe:

```bash
--scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3
```

on top of B3 architecture (`mtlnet_crossattn + static_weight(cat=0.75) + next_gru cat + next_getnext_hard reg, d=256, 8h, batch=2048, 50ep`). Predecessor B3 (50ep + OneCycleLR) preserved as comparand.

**Most recent finding (F49, 2026-04-27):** 3-way decomposition (encoder-frozen λ=0 / loss-side λ=0 / Full MTL) under H3-alt. 5-fold on all 3 states (AL/AZ/FL). The H3-alt reg lift on AL is **purely architectural (+6.48 pp from architecture alone)**; cat-supervision transfer is null on all 3 states (≤|0.75| pp). **Refutes** the legacy "+14.2 pp transfer" claim by ≥9σ on FL alone. Adds methodological contribution Layer 2: loss-side `task_weight=0` ablation is unsound under cross-attention MTL.

**Headline numbers (5-fold × 50ep, H3-alt):**

| State | cat F1 | reg Acc@10 | vs STL F21c (matched-head) |
|---|---:|---:|---|
| AL | 42.22 ± 1.00 | **74.62 ± 3.11** | **+6.25 EXCEEDS** ✓ |
| AZ | 45.11 ± 0.32 | 63.45 ± 2.49 | −3.29 (closes 75% of B3 gap) |
| FL | **67.92 ± 0.72** | **71.96 ± 0.68** | TBD (F37 4050) |

---

## 2 · Doc inventory (live + archived)

**Active study root** (`docs/studies/check2hgi/`, all updated 2026-04-27):
- `README.md` — entry point + nav
- `AGENT_CONTEXT.md` — long-form briefing
- `NORTH_STAR.md` — current champion + predecessor + scale-dependence flag
- `PAPER_STRUCTURE.md` — paper scope, baselines, STL-matching policy
- `PAPER_PREP_TRACKER.md` — paper-deliverable tracker (NEW 2026-04-27)
- `FOLLOWUPS_TRACKER.md` — per-experiment work queue
- `OBJECTIVES_STATUS_TABLE.md` — v5 scorecard
- `CLAIMS_AND_HYPOTHESES.md` — CH01–CH19 catalog
- `CONCERNS.md` — C01–C15 risk register
- `MTL_ARCHITECTURE_JOURNEY.md` — end-to-end derivation story
- `SESSION_HANDOFF_2026-04-{22,24,24_PM,26,27}.md` — operational gotchas + chronology
- `research/F*_FINDINGS.md` — paper-substantive notes (B3, B5, B7, F21C, F27, F38, F40, F41, F44, F48-H1/H2/H3, F49, GETNEXT, NASH_MTL, POSITIONING_VS_HMT_GRN, SOTA_STAN, STAN_FOLLOWUPS, etc.)
- `results/RESULTS_TABLE.md` — canonical per-state × per-method canonical table
- `issues/` — bug audits (MTL_PARAM_PARTITION_BUG, FOLD_LEAKAGE_AUDIT, CROSSATTN_PARTIAL_FORWARD_CRASH, BACKBONE_DILUTION, GRADNORM_EXPERT_GATING, MODEL_DESIGN_REVIEW_2026-04-22, REGION_HEAD_MISMATCH)

**Archived:**
- `archive/phases_original/` — P0–P7 (P6 missing — originally encoder-enrichment)
- `archive/pre_b3_framing/` — MASTER_PLAN, KNOWLEDGE_SNAPSHOT, etc.
- `archive/research_pre_b3/`, `archive/research_pre_b5/` — pre-B-era notes
- `archive/2026-04-20_status_reports/`, `archive/v1_wip_mixed_scope/` — status snapshots

---

## 3 · PAPER_PREP_TRACKER.md — what's needed for submission

### 3.1 — Paper-grade claims **committable now** (write-ready)

| Claim | Status | Source |
|---|---|---|
| **CH16** Check2HGI > HGI on next-cat (AL +18.30 pp) | ✅ AL only | `results/P1_5b/` |
| **CH17** Check2HGI > POI-RGNN external | ✅ audit done | `docs/baselines/POI_RGNN_AUDIT.md` |
| **CH18** H3-alt closes/exceeds matched-head STL gap on reg | ✅ AL+AZ+FL | `research/F48_H3_PER_HEAD_LR_FINDINGS.md` |
| **CH19 Layer 1** Cat-supervision transfer ≤ \|0.75\| pp on AL/AZ/FL n=5 | ✅ | `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` |
| **CH19 Layer 2** Loss-side `task_weight=0` ablation unsound under cross-attn MTL | ✅ + 4 tests | `research/F49_LAMBDA0_DECOMPOSITION_GAP.md` |
| AL architectural = +6.48 ± 2.4 pp ~2.7σ | ✅ | F49 5f |
| AZ architectural = −6.02 ± 1.6 pp ~3.7σ | ✅ | F49 5f |
| F49b reproduction gate PASSED | ✅ | F49b clean repro |

### 3.2 — Headline-blocking (must land before submission)

| # | Item | Cost | Owner |
|---|---|:-:|:-:|
| **P1** | F37 STL `next_gru` cat 5f per state | ~3h | 4050 |
| **P2** | F37 STL `next_getnext_hard` reg 5f on FL (closes Layer 3) | ~2h | 4050 |
| **P3** | CA + TX upstream pipelines + 5f H3-alt | F34/F35 ~12h Colab + F24/F25 ~25h | Colab |

### 3.3 — Sharpens claims (P2 nice-to-have)

| # | Item | Cost |
|---|---|:-:|
| P4 | Paired Wilcoxon p-values on F49 cells per fold | ~30 min |
| P5 | Paired Wilcoxon: H3-alt vs predecessor B3 across folds (AL+AZ) | ~30 min |
| P6 | Seed sweep H3-alt on AL+AZ {0, 7, 100} | ~3h MPS |
| P7 | `MTL_ARCHITECTURE_JOURNEY.md` post-F49 narrative pass | ~30 min |

### 3.4 — Camera-ready / post-submission

| # | Item |
|---|---|
| P8 | `POSITIONING_VS_HMT_GRN.md` rewrite for F49 (currently has deprecation note) |
| P9 | Block-internal `ffn_a` ablation |
| P10 | α instrumentation per epoch per fold per F49 variant |
| P11 | FL frozen-cat reg-path instability investigation |
| P12 | Extend `experiments/check2hgi_up` to H3-alt (user-decision) |

### 3.5 — Pending doc rewrites

- `results/RESULTS_TABLE.md` needs F49 cells (encoder-frozen λ=0 / loss-side λ=0) per state
- `research/POSITIONING_VS_HMT_GRN.md` full rewrite for camera-ready
- A new `research/F49_METHODOLOGICAL_NOTE.md` — Layer 2 standalone short note

### 3.6 — Submission-readiness checklist (from PAPER_PREP_TRACKER §5)

- [x] CH16 substrate — AL evidence
- [ ] CH16 cross-state (F3 AZ HGI STL cat)
- [x] CH17 external baselines (POI-RGNN audit)
- [x] CH18 reg gap closure (AL+AZ+FL)
- [x] CH19 Layer 1+2 (transfer null + methodological)
- [ ] CH19 Layer 3 (FL absolute architectural Δ — blocked on F37)
- [ ] CA 5f headline (F24, blocked on F34)
- [ ] TX 5f headline (F25, blocked on F35)
- [ ] Paired Wilcoxon (P4 + P5)
- [ ] Methods/Results/Discussion sections drafted with F27+B3+H3-alt+F49 attribution
- [ ] Limitations: AL is dev state; FL frozen-cat instability; CA/TX gated
- [ ] Reviewer-rebuttal ammo (per-fold JSONs + repro gate + 4 regression tests)

---

## 4 · FOLLOWUPS_TRACKER.md — work queue snapshot

### 4.1 — Ready-now (P1/P2 — paper-blocking)

| # | Item | Status |
|---|---|---|
| **F37** | STL `next_gru` cat 5f per state | **pending — 4050-assigned** |
| **F33** | FL 5f×50ep B3+next_gru (decisive F27 FL resolution) | **pending — Colab T4** |
| **F34** | CA upstream pipeline + CA 1f×50ep B3+next_gru | **pending — Colab T4** |
| **F35** | TX upstream pipeline + TX 1f×50ep B3+next_gru | **pending — Colab T4** |
| **F24** | CA 5f headline (gated on F34) | **gated** |
| **F25** | TX 5f headline (gated on F35) | **gated** |
| **F4** | FL MTL-B3 5-fold (clean re-run) | **pending** — replaces F17 partial |
| **F3** | AZ HGI STL cat (5f×50ep, fair folds) | **pending — m4_pro** |
| **F9** | FL HGI STL cat 5f | **pending — extends CH16** |

### 4.2 — Done (recent, for audit)

- F49 / F49b / F49c (all done 2026-04-27)
- F48-H3-alt + F48-H1 + F48-H2 + F48-H3 (done 2026-04-25/26)
- F40 / F41 / F42 / F43 / F44 / F45 / F46 / F47 (done)
- F38 / F39 (refuted)
- F31 / F32 / F27 / F21c (done)
- F18 / F19 / F19-followup / F20 / F2 / F1 / F11 (done)

### 4.3 — Deferred (follow-up paper / post-champion-freeze)

| # | Item |
|---|---|
| **F8** | **Multi-seed n=3 on champion configs** — deferred 2026-04-23 by user |
| F12 | Per-fold transition matrix (leakage-safe GETNext) |
| F13 | GETNext with true flow-map Φ |
| F14 | PIF-style user-specific frequency prior |
| F15 | TGSTAN + STA-Hyper full reproductions |
| F16 | Encoder enrichment (P6: temporal/spatial/graph features) |
| F26 | AZ HGI STL cat for CH16 (duplicate of F3) |
| F36 | CH15 under graph-prior head |
| F21a | **DROPPED** by user |
| F21b | Archived — STL GETNext (soft) 5f per state |
| F27-followup | Cat-head ablation (next_stan, ensemble, etc.) |

### 4.4 — Gated

- **F5** FL MTL-GETNext-soft 5-fold — covered at n=1 already

---

## 5 · Cross-reference vs `archive/phases_original/` — original P0–P7 plan

The original plan (P0_preparation → P1_single_task → P1.5_embedding → P2_mtl_headline → P3_dual_stream → P4_cross_attention → P5_ablations → P7_headline_states) was retired 2026-04-23. Below is a faithful cross-check of what each phase planned vs what actually got done.

### P0 Preparation — mostly DONE; **2 forgotten/deferred items**

| Test | Status | Notes |
|---|---|---|
| P0.1 Integrity checks (PI.1–PI.7) | ✅ | implicit through study moving on |
| P0.2 Label round-trip (20 hand-verified samples) | ⚠️ | not formally documented; embedded in pipeline tests |
| **P0.3 Code deltas for `next_poi` task** | 🟡 RETIRED | Original task pair `{next_poi, next_region}` was retired in favor of `{next_category, next_region}` 2026-04-16. Deliberate scope decision. `CHECK2HGI_NEXT_POI_REGION` preset, `next_poi.py`, etc. were never built. ✓ no action needed. |
| P0.4 Smoke test | ✅ | done |
| P0.5 Simple baselines | ✅ | done for AL/AZ/FL via `scripts/compute_simple_baselines.py` (JSONs in `results/P0/simple_baselines/{alabama,arizona,florida}/`); Markov k=1..9 extension done 2026-04-20. **CA + TX still missing** (gated on F34/F35). |
| **P0.6 fclass-shuffle ablation (CH14)** | ⚠️ **DEFERRED** | `ch14_audit.md` documents code-inspection done; unconditional shuffle ablation **NEVER executed**. Stated as "not a blocker for P1" — but may be **paper-relevant rigor** if reviewers push back on shortcut audit. ⚠️ **Forgotten/deferred — worth flagging**. |
| **P0.7 Transductive-leakage audit (original CH15)** | 🔴 **NEVER EXECUTED** | `ch15_audit.md` has only the procedure — never implemented. Original CH15 = "held-out-user-fold Check2HGI retrain" upper-bound estimate. **Note:** CH15 number was **REUSED** in `CLAIMS_AND_HYPOTHESES.md` for a different claim ("Check2HGI ≈ HGI on region task") — **this creates an audit-trail problem**. The transductive audit is a Tier-E paper limitation. ⚠️ **Forgotten — worth flagging**. |
| P0.8 Final preflight gate | ✅ | implicit |

### P1 Single-task baselines — REPURPOSED & DONE

- ✅ STL Check2HGI cat (AL +AZ); STL HGI cat (AL only — F3/F9 pending)
- ✅ STL region heads (AL, FL); AZ implicit
- ✅ STL STAN (AL, AZ); 🔴 **FL STL STAN never run (F6)**
- ✅ STL GETNext-hard matched-head (AL, AZ via F21c); 🔴 **FL pending (F37)**
- 🟡 CH06 OOD-restricted Acc@K reporting → **implemented as `*_indist` metrics in `mtl_eval.py::_ood_restricted_topk`** ✓ (confusingly renamed but preserved)

### P1.5 Embedding comparison — DONE & PIVOTED

- ✅ Check2HGI vs HGI on cat F1 (AL: +18.30 pp clean)
- 🔴 AZ HGI STL cat (F3) and FL HGI STL cat (F9) **NOT run**
- Pivoted from "Check2HGI > HGI on region" (refuted, tied) to "Check2HGI > HGI on cat" (confirmed AL)

### P2 MTL Headline (`P2_mtl_headline.md`) — STALE, REFRAMED

| Original goal | Now |
|---|---|
| `{next_poi, next_region}` MTL on AL+FL × 3 seeds × 5 folds | **Retired**. New task pair `{next_category, next_region}`. |
| n=15 paired samples for Wilcoxon | Done at n=5 only; **F8 multi-seed n=3 deferred** |
| Class-weighted FL run | Not the lever — H3-alt + per-head LR is |
| **CH11 seed variance** | 🔴 **NEVER measured** — F8 deferred 2026-04-23 |

### P3 Dual-stream input (`P3_dual_stream.md`) — STALE

- The "concat both modalities" design was empirically inferior; replaced with **per-task input modality** (`task_a=checkin, task_b=region`)
- Embedded in champion config; never run as standalone P3.

### P4 Cross-attention (`P4_cross_attention.md`) — DONE & SUPERSEDED

- ✅ `MTLnetCrossAttn` implemented and is the architecture in the H3-alt champion
- ✅ AL ablation done; FL replication done at scale (NS-1 effectively)
- 🔴 NS-2 Hybrid (cross-attn cat + dselectk reg) — **NEVER executed**
- 🔴 NS-3 Multi-seed cross-attention on AL — **NEVER done** (subset of F8)
- 🔴 NS-4 Block-count sweep {1,2,3,4} — **NEVER done** (low-priority, defer)

### P5 Ablations (`P5_ablations.md`) — DISTRIBUTED, partially DONE

| Original sub-test | Now |
|---|---|
| CH09 head architecture sweep (5 heads × AL) | ⚠️ **PARTIAL** — F27 covered cat heads (`next_mtl` vs `next_gru`); region head explored via B5/F21c. **Full 5-head sweep on cat side never run; the 5-head sweep on region was only done at STL** |
| CH10 MTL optimiser sweep on **AL + FL** | ⚠️ **PARTIAL** — covered by `ATTRIBUTION_PCGRAD_VS_STATIC.md` + F2; but only at n=1 fold; **no FL multi-optimizer 5f comparison** |
| **CH11 Seed variance (3 seeds)** | 🔴 **NEVER DONE** — F8 deferred |

### P6 Encoder enrichment — NOT IN SCOPE (post-paper)

- Originally a research-gated phase (CH12 temporal Time2Vec, CH13 spatial Sphere2Vec). Never started. Deferred to follow-up paper per `CLAIMS_AND_HYPOTHESES.md`.

### P7 Headline states (`P7_headline_states.md`) — IN PROGRESS, **major gap**

| Plan | Now |
|---|---|
| FL: 5 runs × 5 folds | ✅ FL H3-alt 5f done; 🔴 FL STL STAN (F6); 🔴 FL STL GETNext-hard (F37); 🔴 FL λ=0 decomposition done as **F49c** (loss-side + frozen-cat) but **STL F21c absent** |
| **CA**: embeddings + 5 runs × 5 folds | 🔴 **NOT STARTED** — F34/F24 |
| **TX**: embeddings + 5 runs × 5 folds | 🔴 **NOT STARTED** — F35/F25 |
| Multi-seed n=3 on headline (P7 §11) | 🔴 deferred (F8) |
| §11.5 GETNext-hard headline | ✅ used as the reg head universally |

The original P7 launcher (`scripts/p7_launcher.sh`) exists but was designed for the **pre-H3-alt** recipe (PCGrad + cross-attn + soft probe). It's the wrong protocol for the H3-alt champion and would need updating before CA/TX runs.

---

## 6 · Forgotten / overlooked items (the ones worth your attention)

These are tasks from the original plan that have **slipped through** without explicit deferral or completion in the trackers:

### 🔴 High priority (paper-relevant)

1. **P0.6 fclass-shuffle ablation (CH14)** — only code inspection done; the unconditional shuffle ablation that critical-review §2.3 recommended was deferred but never resumed. Cheap (~30 min) and would harden CH14's "no shortcut" claim against reviewer pushback.

2. **P0.7 / original CH15 transductive-leakage audit** — never executed. The CH15 number was reused for a different (region-task tie) claim. The transductive-leakage upper-bound is a paper-relevant Tier-E limitation that has **no current evidence**. Either run it (~20 min Check2HGI retrain + 10 min eval on fold 0) or formally declare it as a known limitation in CONCERNS.md (it isn't — there's no C-entry for the original CH15).

3. **CH11 / F8 Seed variance** — the entire "paper has seed σ" requirement is currently absent. AL/AZ/FL all run at seed=42 only. Reviewers will ask. P6 in PAPER_PREP_TRACKER mentions "AL+AZ {0, 7, 100}" as nice-to-have but it's **not on the paper-blocking list**.

### 🟡 Medium priority (sharpens claims)

4. **P5 head-architecture full sweep (CH09)** — only F27 (cat head: `next_mtl` vs `next_gru`) was done. The region side has B5/F21c but no formal 5-head MTL sweep. Tracker doesn't list this.

5. **P5 optimiser sweep on FL (CH10)** — only AL has the PCGrad-vs-static attribution; FL multi-optimizer 5-fold comparison absent.

6. **C10 — locating the HGI-based next-category external reference** — CH17 is "pending" because the user owes a specific HGI-next-category article reference. **Open in CONCERNS.md** but not in PAPER_PREP_TRACKER.

7. **NS-2 Hybrid (cross-attn cat + dselectk reg)** — listed as future-work in archived P4. Probably permanently deferred but not formally retired.

### 🟢 Low priority (already paper-ready or properly deferred)

- F8 multi-seed (deferred by user)
- P6 encoder enrichment (CH11/CH12/CH13 deferred)
- F21a (DROPPED), F21b, F12-F16 (follow-up paper)

---

## 7 · Operational state (from SESSION_HANDOFF_2026-04-27)

**In-flight at session end:** F49c FL 5-fold re-run (`bb6evts6r`) — landed 21:44, refuted Tree C, confirmed Tree A across all 3 states.

**Background queue:** none currently in flight.

**Known operational gotchas (G1–G8):** see SESSION_HANDOFF_2026-04-22 + 04-27. Notable:

- C09 SSD reliability: long FL runs require `/tmp`-resident I/O after observed SIGBUS (Thunderbolt SSD)
- FL needs `--batch-size 1024` (2048 silently OOM-killed)
- MPS: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` + `PYTORCH_ENABLE_MPS_FALLBACK=1`

---

## 8 · TL;DR — what to do next

**If you want the paper submittable in the shortest path:**

1. **Run F37 on 4050** (STL `next_gru` cat 5f + STL `next_getnext_hard` reg 5f on FL) — closes Layer 3 + matched-head story
2. **Quick wins on m4_pro** (~1h total): P4 + P5 paired-Wilcoxon scripts on existing JSONs
3. **Decide on F8 multi-seed scope** before reviewer pushback forces it
4. **Resolve forgotten items 1+2** (CH14 shuffle + CH15-original transductive) — either run them (cheap) or formally retire them in CONCERNS.md

**Then on Colab (the real critical path):**

5. F33 → if Path A holds, commit `next_gru` universally
6. F34 → F24 (CA upstream + 5f headline)
7. F35 → F25 (TX upstream + 5f headline)

This unblocks the **headline FL+CA+TX table** which is the paper's main claim. Without CA/TX you have a 3-state ablation paper (AL+AZ+FL); with CA/TX you have the headline paper as originally scoped.

---

## 9 · Cross-references

- `PAPER_PREP_TRACKER.md` — paper-deliverable tracker
- `FOLLOWUPS_TRACKER.md` — per-experiment work queue
- `archive/phases_original/{P0_preparation,P1_single_task_baselines,P1.5_embedding_comparison,P2_mtl_headline,P3_dual_stream,P4_cross_attention,P5_ablations,P7_headline_states}.md` — the original plans cross-referenced above
- `results/P0/{ch14_audit.md,ch15_audit.md,simple_baselines/}` — P0 preparation artefacts
- `SESSION_HANDOFF_2026-04-27.md` — most-recent operational state
- `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` — most-recent science
