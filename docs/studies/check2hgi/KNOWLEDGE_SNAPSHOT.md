# Knowledge Snapshot — Check2HGI Study (2026-04-15)

**Purpose:** a single entry-point doc that any future agent (or human) can read to understand *exactly* where the Check2HGI study is and why. Sibling to `docs/studies/fusion/KNOWLEDGE_SNAPSHOT.md`.

If you read nothing else, read this. Then drop into the rest of `docs/studies/check2hgi/` for the plan.

---

## Study in one paragraph

This study is a **standalone** investigation: does adding a next-region auxiliary task under MTL improve next-POI prediction on Check2HGI check-in-level contextual embeddings, at matched compute, without per-head negative transfer? **No cross-engine comparison** (no HGI) and **no prior-work replication** (no CBIC, no HAVANA/PGC/POI-RGNN — those are fusion-study territory). Baselines are internal: single-task Check2HGI is the reference for every MTL / dual-stream / cross-attention claim, anchored by simple-baselines floor (majority/random/Markov/top-K computed on our data in P0.5). External literature numbers (HMT-GRN, MGCL, etc.) appear only in an appendix with scope caveat. Runs on Gowalla-AL + Gowalla-FL with AZ as triangulation. Ranking metrics (Acc@K, MRR, NDCG) are primary; macro-F1 reported for completeness only.

---

## Why this study exists as a separate study

Three orthogonal reasons:

1. **Scope.** Next-POI prediction is a ranking task over 10³–10⁵ classes; POI-category classification is a balanced 7-class classification. Mixing their claim catalogs confuses the paper contribution.
2. **Embeddings.** Check2HGI produces per-check-in vectors (one per event); HGI produces per-POI vectors. The check-in granularity is wasted on per-POI tasks and uniquely useful for sequence tasks.
3. **Baselines.** Next-POI literature uses HMT-GRN, MGCL, LSTPM, STAN, GETNext, ImNext, Graph-Flashback — ranking-metric papers on FSQ-NYC/TKY or Gowalla-global. POI-category literature uses HAVANA, PGC, POI-RGNN on Gowalla state-level. The two citation trees barely overlap.

---

## What we (currently) believe (tier: hypothesis, not yet confirmed on clean data)

These hypotheses drive the study. None confirmed; all map to specific CH-claims.

1. **A hierarchical auxiliary task (next-region) provides useful inductive bias for next-POI prediction under MTL** — region-level supervision structures the shared backbone so the fine-grained POI head learns transferable features. **Maps to CH01 (HEADLINE).** Expected: 2-task MTL > single-task next-POI on Check2HGI.

2. **No per-head negative transfer** — NashMTL (or equal_weight alternative) retains both heads' single-task performance. **Maps to CH02.**

3. **Region embeddings as an explicit input stream help independently of the auxiliary task** — giving the encoder direct region information (via dual-stream concat) improves next-POI beyond what the shared backbone alone picks up from the next-region head. **Maps to CH03 (Tier-A).** Tested in P3.

4. **Region-input gain is state-dependent** — preliminary linear-probe experiment showed check-in → region recovery degrades from 3.4× (AL) to 1.7× (AZ) to 1.04× (FL) majority lift, so FL should benefit most from dual-stream input. **Maps to CH08.** Caveat: FL probe was a non-converged LR fit; rerun with class-weighted MLP before P3.

5. **Bidirectional cross-attention between check-in and region streams extracts structure concat hides** (gated on #3) — closer to HMT-GRN's hierarchical-attention design. **Maps to CH07.**

6. **Check-in-level embeddings are doing real work** — simple baselines (majority / random / 1-step Markov / top-K popular / user-history top-K) are substantially beaten by any learned Check2HGI model. **Maps to CH04 (floor / known-good reference).** This is the lower-bound check that catches pipeline bugs before any headline claim.

**None confirmed yet.** They drive P1–P4.

---

## The critical open question (#1 reviewer risk)

**Does MTL with next-region actually help next-POI prediction on Check2HGI?** (CH01)

Prior MTL-research has found that adding auxiliary tasks helps in some settings but hurts in others (Xin et al. 2022 showed equal-weight can match adaptive methods on single-source embeddings). On Gowalla state-level Check2HGI data, with a 2-task pair that has ×10 or more label-space cardinality asymmetry (next_poi ~10–80K classes vs next_region ~1–5K classes), the behaviour is unstudied. **This is the single most important control in this study — if CH01 refutes, the paper thesis reframes around CH03 (dual-stream input helps even without MTL).**

Multi-seed (n=15) is the default for this test per review-agent finding — n=5 Wilcoxon has near-zero statistical power at the 2pp effect sizes we'd call decisive.

## The #2 reviewer risk

**Does Check2HGI's preprocessing contain a task shortcut?**

Fusion study's C29 finding (`docs/studies/fusion/issues/HGI_LEAKAGE_AUDIT.md`) showed the Gowalla HGI/POI2Vec preprocessing has an fclass → category 1:1 determinism that makes category classification trivially solvable. Check2HGI uses a different preprocessing chain — it may or may not have an analogous shortcut for next-POI / next-region. **Maps to CH14.** P0.6 does both (a) code inspection and (b) unconditional fclass-shuffle ablation on AL single-task next-POI. If the ablation shows a large Acc@10 drop, the representation relies on a shortcut and the paper reframes around the finding.

---

## Where this study is now (2026-04-15)

| Artefact | Status |
|---|---|
| Check2HGI embeddings (AL + FL + AZ) | ✅ generated (AZ for triangulation only) |
| Next-region label parquet (AL: 1109 regions, FL: 4703, AZ: 1547) | ✅ generated |
| TaskConfig / TaskSet infrastructure in code | ✅ |
| `check2hgi_next_poi_region` preset | ⚠️ **to be added** (current preset is `check2hgi_next_region` which targets next_category as slot A — wrong for the new scope) |
| Next-POI label derivation (placeid → poi_idx) | ⚠️ **to be added** (similar to next_region, but index into poi_to_region's input rather than the region column) |
| `FoldCreator.MTL_CHECK2HGI` path | ✅ but loads next_category as slot A — **needs a new variant for next_poi as slot A** |
| `scripts/train.py --task-set check2hgi_next_poi_region` | ⚠️ **to be added** |
| MTLnet sequential-A forward path | ✅ |
| Per-head num_classes + hand-rolled high-cardinality metrics | ✅ |
| `val_joint_lift` monitor | ✅ |
| End-to-end CLI smoke on AL (with the *old* preset) | ✅ exit 0, 7.88s — but wrong task pair |

**Code deltas needed before P1 can run:**
1. Add `next_poi.py` loader (mirrors `src/data/inputs/next_region.py` but uses `placeid_to_idx` on `target_poi`).
2. Add `IoPaths.get_next_poi(state, engine)` helper.
3. Add `CHECK2HGI_NEXT_POI_REGION` preset in `src/tasks/presets.py`.
4. Add `pipelines/create_inputs_check2hgi_next_poi.pipe.py` to materialise next_poi labels (row-aligned with `next.parquet`).
5. Extend `FoldCreator._create_check2hgi_mtl_folds` to route next_poi labels into slot A when the preset is `check2hgi_next_poi_region`.

All of this is estimated at ~2–3h. The scaffolding from the prior (mixed-scope) work is largely reusable — we're swapping the task_a labels from `next_category` to `next_poi`.

---

## What's been deferred or explicitly cut

Captured under `docs/studies/check2hgi/archive/v1_wip_mixed_scope/` (the prior scope-mixed work):

- Full 5-architecture × 20-optimiser grid (fusion P1 port).
- Expert-gating architectures (CGC / MMoE / DSelect-K).
- Frozen-backbone head-swap co-adaptation probe.
- next_time_gap third auxiliary task.
- POI-category as the primary head on Check2HGI.

---

## Related studies

- **`docs/studies/fusion/`** — POI-category classification on fused POI-level embeddings. Read its `KNOWLEDGE_SNAPSHOT.md` first for the overall project narrative; this snapshot layers on the Check2HGI-specific context.
- Fusion's C29 (fclass shortcut) result directly motivates CH14 in this study.
- Fusion's C06 (MTL vs single-task) is the analogue of CH02 here, adapted to next-POI + next-region.
