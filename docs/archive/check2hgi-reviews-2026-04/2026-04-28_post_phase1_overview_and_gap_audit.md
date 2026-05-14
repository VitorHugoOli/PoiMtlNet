# Check2HGI Study — Overview & Gap Audit

**Date:** 2026-04-28
**Trigger:** Audit conducted post-merge `a1996e9` (Phase-1 substrate validation + F49 architecture attribution + audit-fix doc sweep)
**Scope:** Compile current state, missing tasks across `PAPER_PREP_TRACKER.md` and `FOLLOWUPS_TRACKER.md`, and verify nothing was forgotten from `archive/phases_original/` (P0–P7).
**Predecessor:** `2026-04-27_study_overview_and_forgotten_items.md` (written ~23:09 on 2026-04-27, before the 5 commits at 23:22–23:30; superseded by this doc).

---

## 1. Where we are

**Phase 1 substrate validation (AL+AZ): CLOSED 2026-04-27** with 4 paper-grade findings:
- **CH16** Check2HGI > HGI on cat F1 — head-invariant, 8/8 probes p=0.0312, +11.58 to +15.50 pp
- **CH15** reframed — head-coupled (HGI > C2HGI under STAN; flips to C2HGI ≥ HGI under matched MTL reg head)
- **CH18** MTL+HGI breaks the joint signal — cat −17 pp, reg −30 pp at AL+AZ (Tier B → **A**)
- **CH19** mechanism — per-visit context = ~72% of cat substrate gap (POI-pooled counterfactual at AL)

**F49 architecture attribution (AL+AZ+FL n=5): COMPLETE 2026-04-27**:
- Cat-supervision transfer ≤ |0.75| pp on all 3 states — **refutes legacy "+14.2 pp transfer" claim by ≥9σ on FL alone**
- AL architecture alone +6.48 pp (~2.7σ); AZ −6.02 pp; FL absolute Δ pending F37
- **Layer 2 methodological contribution**: loss-side `task_weight=0` ablation is unsound under cross-attention MTL — encoder-frozen isolation is the only clean variant. Applies retroactively to MulT/InvPT/HMT-GRN literature.

**CH21 top-line synthesis (Tier A):** "MTL win is interactional architecture × substrate, not transfer."

**Champion config (NORTH_STAR):** F48-H3-alt — `mtlnet_crossattn + static_weight(cat=0.75) + next_gru + next_getnext_hard, d=256, 8h`, `--scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3`. Validated AL+AZ+FL 5-fold; AL exceeds matched-head STL by **+6.25 pp Acc@10**.

**Phase 2 (FL+CA+TX replication): authorized 2026-04-27, NOT STARTED.** FL data on disk; CA+TX gated on upstream pipelines.

**Paper drafting status: zero prose written.** All claims locked, all tables have either filled cells or templates, but Methods / Results / Limitations / Appendix are skeleton-only.

---

## 2. PAPER_PREP_TRACKER — what's still owed

### Headline-blocking (must land before submission)

| ID | Item | Cost | Owner | Status |
|----|------|------|-------|--------|
| **P1** | F37 STL `next_gru` cat 5f per state | ~3h | 4050 | NOT STARTED |
| **P2** | F37 STL `next_getnext_hard` reg 5f on FL | ~2h | 4050 | NOT STARTED |
| **P3** | CA + TX upstream pipelines + 5f H3-alt | ~37h | Colab | NOT STARTED |

P1+P2 are short (~5h GPU) and unblock the FL absolute architectural Δ + cat-side STL ceiling claims. P3 is the heavy headline expander; risk-register fallback is "AL+AZ+FL only."

### Nice-to-have (sharpens claims, ~7h total)

| ID | Item | Cost |
|----|------|------|
| **P4** | Paired Wilcoxon p-values on F49 cells per fold | ~30 min |
| **P5** | Paired Wilcoxon H3-alt vs B3 across folds | ~30 min |
| **P6** | Seed sweep H3-alt on AL+AZ {0, 7, 100} | ~3h MPS |
| **P7** | MTL_ARCHITECTURE_JOURNEY post-F49 narrative pass | ~30 min |

### Camera-ready / post-submission (P8–P12)

- **P8** POSITIONING_VS_HMT_GRN.md rewrite (~2h) — legacy +14.2 pp / 25 pp overhead claims now refuted, deprecation note added
- **P9** Block-internal `ffn_a` ablation (~2-3h) — F49 plan flags `_CrossAttnBlock.ffn_a/ln_a*` are NOT frozen in either F49 variant
- **P10** α instrumentation per epoch per fold per F49 variant
- **P11** FL frozen-cat reg-path instability investigation (~30 min)
- **P12** Extend `experiments/check2hgi_up` to H3-alt (decision needed)

### Drafting (NOT COSTED in tracker, ~10–15h estimated)

- Methods · Results · Methodological appendix · Limitations
- New docs: `CH21_JOINT_CLAIM.md`, `F49_METHODOLOGICAL_NOTE.md` (~3pp Layer 2)

---

## 3. FOLLOWUPS_TRACKER — what's still owed

### Open / paper-blocking (5 items, all NOT STARTED)

| ID | Item | Cost | Notes |
|----|------|------|-------|
| **F33** | FL 5f B3+`next_gru` decisive test | Colab ~6h | Gates Path A (universal `next_gru`) vs Path B (scale-dependent cat head) |
| **F34** | CA upstream pipeline + 1f B3+gru | Colab ~6-12h | First CA data point |
| **F35** | TX upstream pipeline + 1f B3+gru | Colab ~6-12h | First TX data point |
| **F37** | STL `next_gru` cat 5f per state | 4050 ~3h | Same as P1 in PAPER_PREP_TRACKER |
| **F4**  | FL MTL-B3 5f clean re-run | ~6-8h MPS | **Status unclear — verify whether already done.** |

### Ambiguous statuses (need verification)

- **F31** B3 post-F27 AL 5f validation — marked "pending" but F27/F32 done 2026-04-24; unclear if executed
- **F39** appears twice with conflicting status — 1f screen done; 5f sweep status unclear

### CONCERNS still open

| ID | Sev | Issue |
|----|-----|-------|
| **C10** | High | POI-RGNN + HGI-next-category external refs — **user owes specific HGI-next-category article reference** |
| **C14** | Med | F27 cat-head scale-dependence flag — F33 is the decisive test |
| **C13** | Med | AL 10K vs FL/CA/TX 127K extrapolation risk |
| **C01** | Med | n=2 states thin for final paper |
| **C02** | Med | FL Markov saturation (65.05) narrows MTL margin |
| **C12** | High | Marked "resolved 2026-04-27" but **conditionally** — Layer 3 explicitly pending F37 |

---

## 4. Original plan gap audit (`archive/phases_original/`)

The single most important finding from this audit: **the original P0–P7 plan and the current docs use three coexisting label schemes** (P0–P7, "Phase 1 / Phase 2", F##-iterative, plus B-letter architecture labels) with **no reconciliation document**. The archive README is the only sentence acknowledging the shift.

### Forgotten / silent drops (no explicit retirement memo)

| # | Item | Severity | Recommendation |
|---|------|----------|----------------|
| 1 | **`next_poi` task pivot** — original P0–P2 targeted `{next_poi, next_region}`; current uses `{next_category, next_region}`. Silent task-pair pivot. | **HIGH — narrative-changing** | Add a one-paragraph memo (PAPER_STRUCTURE.md or NORTH_STAR.md) explaining the pivot. |
| 2 | **CH15 ID reused** for a different claim — original = transductive-leakage audit (P0.7); current = head-coupled finding. Same ID, different content. | **HIGH — audit-trail risk** | Rename current finding (CH15-revised or CH15b) or add explicit redefinition note in CLAIMS_AND_HYPOTHESES. |
| 3 | **P0.6 CH14 fclass-shuffle ablation** — code inspection done, experiment never run | Med | ~30 min run OR formally retire in CONCERNS.md |
| 4 | **P1.5 POI2HGI vs Check2HGI** — entire phase abandoned; substrate comparator switched to HGI without memo | Med | Add a sentence in PAPER_STRUCTURE explaining the comparator choice. |
| 5 | **CH01/CH02 original** literal "MTL > STL on next_poi" — silently replaced by CH18/CH20 | Low (subsumed) | Add a "deprecated claims" footnote in CLAIMS_AND_HYPOTHESES. |
| 6 | **P3 CH03/CH08** dual-stream concat — never tested. P4 ran anyway despite P3-gate. | Med | Add memo: "P3 dual-stream design retired in favor of per-task input modality before execution." |
| 7 | **P5.1 CH09** full 5-cell head ablation — replaced by single binary F27 | Low | Mention in Methods that we did a binary swap on AZ as a pragmatic substitute. |
| 8 | **P5.2 CH10** optimiser ablation (Nash vs equal_weight vs CAGrad) — not done, no verdict | Med | Either run a small sweep on AL OR retire formally. |
| 9 | **P0.2** label round-trip spot check (20-sample placeid verification) — no mention | Low | Either run (~10 min) or note as routine pre-flight done. |
| 10 | **P0.8** final preflight gate (8 boolean conditions) — no documented verdict | Low | Likely done implicitly; note for completeness. |

### Properly retired (no action needed — already tracked)

- F8 multi-seed n=3 — explicit P6 deferral · Multi-seed n=5 — explicit P7 §11 · NS-2 hybrid cross-attn — explicit future work · POI-granularity next-POI — explicit out-of-scope · CA/TX C2 head-agnostic probes — explicit PHASE2_TRACKER §6

---

## 5. Narrative-changing observations

1. **The `next_poi` → `next_category` task pivot is the biggest undocumented foundational decision.** Anyone reading the current paper-facing docs cannot reconstruct *why* without diving into archive. Recommend adding a one-paragraph memo.

2. **CH15 ID reuse is an audit-trail problem.** Two different scientific claims share one ID across the doc history. Recommend renaming.

3. **F4 (FL MTL-B3 5f clean re-run) status is genuinely unclear.** Listed as ready-now P1 but no completion evidence. Recommend verifying before scheduling new work.

4. **C12 "resolved 2026-04-27" is conditional on F37.** If F37 STL FL ceiling lands above MTL H3-alt FL (71.96 Acc@10), CH15+CH18 reverse and the headline reframes. Track this as a real risk, not a closed item.

5. **F49 Layer 2 (loss-side ablation unsound under cross-attn MTL) is novel and paper-grade**, but currently lives only in F49 docs, not in a stand-alone `F49_METHODOLOGICAL_NOTE.md` or paper appendix outline.

6. **The recent review** (`review/2026-04-27_*_overview*.md`) is mostly still-relevant despite being pre-commit. The 5 post-review commits touched FOLLOWUPS_TRACKER, PAPER_PREP_TRACKER, OBJECTIVES_STATUS_TABLE, PHASE2_TRACKER, baselines/, NORTH_STAR, CLAIMS_AND_HYPOTHESES — but the review's flagged forgotten items live in `results/P0/` audits and `archive/phases_original/` which were not touched by those commits.

---

## 6. Critical path to submission

```
Compute (~60h, mostly Colab)
├── F33 (Colab ~6h)        → Path A/B for cat-head — gates F34/F35 logic
├── F37 (4050 ~3h)         → P1+P2: cat-side STL + FL absolute architectural Δ
├── F36 FL Phase-2 grid    → ~3h Colab — substrate replication FL
├── F34/F35 CA/TX upstream → ~24h Colab
└── F24/F25 CA/TX 5f H3-alt → ~25h Colab

Drafting (~15h)
├── Methods, Results, Limitations sections
├── CH21_JOINT_CLAIM.md (with figure suggestions)
├── F49_METHODOLOGICAL_NOTE.md (Layer 2 paper note)
└── POSITIONING_VS_HMT_GRN.md rewrite (camera-ready)

Cheap stat-strengthening (~2h)
└── P4-P7: paired Wilcoxon + seed sweep + journey narrative pass
```

---

## 7. Proposed next moves (ranked)

1. **Verify F4 status** — is FL MTL-B3 5f already done or genuinely not started?
2. **Author the two missing memos** — task pivot rationale + CH15 rename. Both ~30 min, both narrative-critical.
3. **Run F37 (P1+P2)** — ~5h on the 4050; unblocks cat-side STL ceiling + FL Layer 3.
4. **Run paired Wilcoxon p-values on F49 cells** (P4) and **H3-alt vs B3** (P5) — ~1h on existing JSONs.
5. **Decide on CH14 / CH10 / P0.2** — run-or-retire each to clean the ledger before camera-ready.
6. **Launch F33 + F36 in parallel on Colab** — ~9h total.
7. **Start drafting Methods + Results** — claims are locked, no point waiting on CA/TX.
8. **Schedule CA/TX upstream pipelines** when ready.
