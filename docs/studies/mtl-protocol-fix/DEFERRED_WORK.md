# Deferred work — mtl-protocol-fix study (post-closure inventory)

> **Purpose.** The original `mtl-protocol-fix` planning memo (2026-05-20 deep-dive) enumerated §3 "Open gaps ranked by EV/cost" + §4 "Other approaches the user invited" — 11 distinct items. This document inventories which ran (and what they found) vs which remain deferred, and maps each deferred item to its landing in `docs/future_works/` or `docs/studies/`.
>
> **For future agents:** this is the **only authoritative deferred-work map** for the mtl-protocol-fix research line. If an item is listed below as "DEFERRED", it has NOT been touched by this study — open the linked memo and pick it up.

**Drafted:** 2026-05-21 (study closure post-mortem).
**Source:** Planning memo §3 + §4 as captured in [`considerations.md`](considerations.md).

---

## Done / falsified / partially done (no further action in this study)

| Original ID | Item | Status | Where the result landed |
|---|---|---|---|
| §3 Rank 1 | F1 selector fix (`joint_geom_simple`) | **DONE 2026-05-20** | Phase 1 v5/v6 — confirmed +5.6 pp at FL multi-seed deployable axis; scale-conditional bug confirmed. See [`INDEX.html` Phase 1 results](INDEX.html#) + [`phase1_phase2_verdict_v6_final.md`](../../results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md). |
| §3 Rank 3 (T6.2) | T6.2 a2.0_0.3 re-eval under F1 | **DONE 2026-05-20 (FALSIFIED)** | geom_simple = 57.64 vs shipping 61.14 (−3.50 pp). See [`phase2p6_salvage_T6_2_T5_3_vs_shipping.json`](../../results/mtl_protocol_fix/phase2p6_salvage_T6_2_T5_3_vs_shipping.json). |
| §3 Rank 3 (T5.3) | T5.3 multi-view re-eval under F1 | **DONE 2026-05-20 (sub-Bonferroni, no flip)** | geom_simple = 62.08 vs shipping 61.14 (+0.94 within σ). Same artefact as above. |
| §3 Rank 3 (T5.2b) | T5.2b masked POI feature recon re-eval under F1 | **SKIPPED (intentional)** | T5.2b is a cat-side improvement; the F1 fix is reg-axis only — verdict cannot change under selector. Documented in Phase 2 closure entry. Re-eval would consume ~25 GPU-min without informational return. |

**Substrate axis is genuinely exhausted** (no Tier 5/6 candidate flips under F1 selector). Consistent with `canonical_improvement` Tier-6 closure.

---

## §3 Rank 5 — Merge-design Lever 4/5/6 (cross-study integration)

| ID | Item | Status | Landing |
|---|---|---|---|
| §3 Rank 5 Lever 4 | POI2Vec at the p2r boundary | **DEFERRED** | Lives in [`docs/studies/merge_design/`](../merge_design/) Lever 4. Substrate-axis cross-study work; never falsified at the merge-design level. Could close 0.3-0.6 pp of the residual STL substrate-Δ at FL/AZ. |
| §3 Rank 5 Lever 5 | Distribution-level distill on Design M (KL on top-k softmax over neighbours) | **DEFERRED** | Lives in [`docs/studies/merge_design/`](../merge_design/) Lever 5. ~6 GPU-h. |
| §3 Rank 5 Lever 6 | Two-output engine (cat-grade + reg-grade tables) | **DEFERRED** | Lives in [`docs/studies/merge_design/`](../merge_design/) Lever 6. Principled candidate; ~16-24 GPU-h. **Strongest prior for closing the residual STL substrate-Δ without breaking cat.** |

Cross-reference: the residual MTL-vs-STL gap (−7 to −12 pp on consistent shipping substrate) is mechanism-identified as **architectural** (P4 frozen-cat test), not substrate. Lever 6's "two-output engine" approach would let MTL reg head consume a reg-tuned substrate table directly — the same intervention the P4 finding motivates from the MTL side.

---

## §4 — Other approaches the user invited (full inventory)

Each row maps to an existing future-work memo or flags a new memo to draft when next-tier work picks the item up.

| # | Item | Status | Landing |
|---:|---|---|---|
| §4.1 | **Per-task best-epoch shipping** (architectural variant of F1 — track per-task best states independently; ship cat head from cat-best, reg head from reg-best, shared backbone from joint-best). ~2-3 days impl. | **DEFERRED** | Fits inside [`mtl_architecture_revisit.md`](../../future_works/mtl_architecture_revisit.md) — add a §"Per-task best-epoch deployable" sub-track when that study opens. **Closes the disjoint-vs-joint capacity gap at deploy time** (currently ~2-3 pp at FL/CA/TX, ~12 pp at AL/AZ). |
| §4.2 | **HGI reg-head ensemble** — train Check2HGI cat head + HGI reg head, ship as composite. Side-steps substrate-vs-substrate fight. | **DEFERRED — partial home** | Overlaps with merge_design Lever 6 (two-output engine). For the *deployable composite* (not the integrated engine), draft a new memo under `docs/future_works/composite_two_substrate_engine.md` when this becomes the highest-EV path. |
| §4.4 | **Curriculum / task scheduling** — train cat first to convergence, then unfreeze reg; OR alternating-task with explicit ratio control. | **PARTIALLY FALSIFIED** | Phase 2 P4 frozen-cat horizon test (cat fully frozen + cat_weight=0 from epoch 0) showed MTL reg STILL peaks at ep 2 and crashes by ep 11. The "give reg a head start" form of curriculum is unlikely to help. The "freeze reg after its peak, then train cat" form is untested — could be a small experiment under [`substrate_adaptive_mtl_balancing.md`](../../future_works/substrate_adaptive_mtl_balancing.md). |
| §4.5 | **Anchor reg-head supervision to Markov-1-region transitions** — per-fold log_T already loaded; expose it as additional supervisory signal (not just as a head input). Cheap experiment. | **DEFERRED** | Fits inside [`reg_head_architecture_sweep.md`](../../future_works/reg_head_architecture_sweep.md) — add a §"log_T as supervisory signal (not just head input)" sub-track. Cheap (~2 GPU-h); could clarify whether the next_stan_flow head's α·log_T blend is using log_T optimally. |
| §4.6 | **Class-balanced batch sampler at the reg head** — undersample dominant regions per batch. Cheap; the FL 4 700-class long-tail is a likely culprit. | **DEFERRED** | Already enumerated in [`head_window_batch_audit.md`](../../future_works/head_window_batch_audit.md) §"batch class-balance" sub-section. No new memo needed. |
| §4.7 | **Merge-design "Design J" or "Design B" engine wholesale under F1** — different POI engine entirely; already beat canonical on reg with full TOST at AL/AZ. | **DEFERRED** | Cross-study with [`docs/studies/merge_design/`](../merge_design/). Re-evaluate Designs J/B under the F1 selector (`joint_geom_simple`) via `scripts/canonical_improvement/analyze_t64_selectors.py`. Add to merge_design's next-pass agenda. |
| §4.8 | **POI feature reconstruction with HGI as decoder target** — instead of decoding raw POI features, decode HGI's POI embedding (distillation-style, no concat). Same hard-rule-compatible scope as Tier 4. | **DEFERRED — NEW MEMO NEEDED** | No existing future_works memo covers this distillation framing. When the next study opens, draft `docs/future_works/poi_decoder_hgi_distill.md` with this scope. Could re-open the substrate axis if HGI's spatial inductive bias (Delaunay POI-POI) can be transferred without the concat-style leak that T4 risked. |

(Item §4.3 was not in the original list as enumerated.)

---

## Recommended next-tier study priority (carried from v6 verdict)

1. **HIGHEST-EV — [`mtl_architecture_revisit.md`](../../future_works/mtl_architecture_revisit.md)** absorbs §4.1 + §4.2 (architecturally). The P4 finding (mechanism is architectural, not interference) makes this the load-bearing track to close the −7 to −12 pp residual.
2. **SECONDARY — [`reg_head_architecture_sweep.md`](../../future_works/reg_head_architecture_sweep.md)** absorbs §4.5; small, fast, may inform §4.1.
3. **CROSS-STUDY — [`docs/studies/merge_design/`](../merge_design/)** for §3 Rank 5 Levers 4/5/6 + §4.7 (Design J/B under F1).
4. **WHEN PAPER REVISION TIME — [`paper_canon_reevaluation.md`](../../future_works/paper_canon_reevaluation.md)** re-runs §0.1 n=20 multi-seed at the new selector + arch. Includes the CA/TX +3/+7 pp overshoot resolution (C23).
5. **NEW MEMOS TO DRAFT WHEN PICKED UP:**
   - `composite_two_substrate_engine.md` for §4.2 (deployable HGI reg-head + c2hgi cat-head composite, distinct from merge_design Lever 6's integrated engine).
   - `poi_decoder_hgi_distill.md` for §4.8 (HGI embedding as decoder target).
6. **LOWER-EV — [`substrate_adaptive_mtl_balancing.md`](../../future_works/substrate_adaptive_mtl_balancing.md)** absorbs §4.4 partially; the P4 falsification reduces its priority but the "freeze reg after peak" variant is still untested.

---

## Closure decisions captured here (not lost to time)

- **§3 Rank 1 F1 fix worked** but didn't close the gap on its own — at FL it recovers most substrate capacity (deployable-vs-disjoint = 2.4 pp gap) but the disjoint-vs-STL gap (−7 pp at FL multi-seed, larger elsewhere) is the architectural residual that hands work off to mtl_architecture_revisit.md.
- **§3 Rank 3 substrate-axis re-eval under F1 found no winners** — the substrate axis is genuinely exhausted (consistent with canonical_improvement Tier-6 closure). Any future substrate work must justify a NEW mechanism not in the Tier 1-6 + T5/T6 §Discussion families.
- **§4.4 curriculum/scheduling is mostly falsified** — P4 frozen-cat shows the cat task is NOT the bottleneck. Future loss-balancing work should target the shared-backbone capacity, not cat-reg gradient balancing.
- **§4.2 HGI reg-head ensemble + §3 Rank 5 Lever 6 share a common mechanism** — both expose the reg head to a reg-tuned substrate signal directly, bypassing the shared-backbone bottleneck. They are alternatives, not complements; whichever lands first should resolve whether the other is needed.

---

## How to use this document

- **If you arrive on the mtl-protocol-fix branch wanting to do more work**: read this file first, then `log.md`, then `INDEX.html`. Anything listed as DONE / FALSIFIED / SKIPPED here is closed — don't re-run.
- **If you arrive on a successor study (architecture revisit, etc.)**: this file lists which §3/§4 items your study should pick up. Cross-reference into your own AGENT_PROMPT.
- **If you discover a NEW gap not in §3 / §4**: add a row to the appropriate table above and (if new) drop a future_works memo.

**Closure provenance:** [`phase1_phase2_verdict_v6_final.md`](../../results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md) §Caveats and follow-ups.
