# Knowledge Snapshot — Check2HGI Study (2026-04-15)

**Purpose:** a single entry-point doc that any future agent (or human) can read to understand *exactly* where the Check2HGI study is and why. Sibling to `docs/studies/fusion/KNOWLEDGE_SNAPSHOT.md`.

If you read nothing else, read this. Then drop into the rest of `docs/studies/check2hgi/` for the plan.

---

## Study in one paragraph

This study tests whether **check-in-level contextual embeddings** (Check2HGI) plus a **next-region auxiliary task** improve **next-POI prediction** over POI-level embeddings (HGI) with single-task next-POI training. It runs on Gowalla state-level data (AL + FL, with AZ as triangulation). The natural evaluation is ranking-metric (Acc@K, MRR, NDCG), not macro-F1. It is a sibling study to the fusion track (`docs/studies/fusion/`), which handles POI-category classification on POI-level fused embeddings and uses HAVANA/PGC/POI-RGNN as baselines — a different task and different baseline lineage.

---

## Why this study exists as a separate study

Three orthogonal reasons:

1. **Scope.** Next-POI prediction is a ranking task over 10³–10⁵ classes; POI-category classification is a balanced 7-class classification. Mixing their claim catalogs confuses the paper contribution.
2. **Embeddings.** Check2HGI produces per-check-in vectors (one per event); HGI produces per-POI vectors. The check-in granularity is wasted on per-POI tasks and uniquely useful for sequence tasks.
3. **Baselines.** Next-POI literature uses HMT-GRN, MGCL, LSTPM, STAN, GETNext, ImNext, Graph-Flashback — ranking-metric papers on FSQ-NYC/TKY or Gowalla-global. POI-category literature uses HAVANA, PGC, POI-RGNN on Gowalla state-level. The two citation trees barely overlap.

---

## What we (currently) believe (tier: hypothesis, not yet confirmed on clean data)

These are the hypotheses this study exists to validate or refute. None are confirmed; all map to specific CH-claims.

1. **Check-in-level embeddings carry more trajectory-phase signal than POI-level embeddings** — contextual information (morning vs evening, weekday vs weekend) that a static POI-level embedding discards. **Maps to CH01.** Expected: Check2HGI single-task next-POI > HGI single-task next-POI on Acc@10.

2. **A hierarchical auxiliary task (next-region) provides useful inductive bias for next-POI prediction under MTL** — the region-level supervision structures the shared backbone so the fine-grained POI head learns transferable features. **Maps to CH02.** Expected: 2-task MTL > single-task on next-POI Acc@10.

3. **No per-head negative transfer** — under NashMTL (or a suitable equal-weight alternative), both heads retain their single-task performance. **Maps to CH03.**

4. **Region embeddings as an explicit input stream help most on states where check-in embeddings lose region signal** — preliminary linear-probe experiment (2026-04-15) showed check-in → region recovery degrades from 3.4× majority lift on AL to 1.04× on FL, so FL should benefit most from dual-stream input. **Maps to CH06 and CH11.**

5. **Bidirectional cross-attention between check-in and region streams extracts structure concat hides** (gated on #4) — closer to HMT-GRN's hierarchical-attention design. **Maps to CH07.**

**None of these are confirmed yet.** They are the hypotheses that drive P1–P4.

---

## The critical open question (#1 reviewer risk)

**Does Check2HGI actually beat HGI at matched training budget?**

The fusion study flagged (advisor + critical-review agent) that Check2HGI was trained to 500 epochs with the loss still decreasing — while HGI's reference checkpoints may have been trained for longer or with different optimiser settings. Without budget matching, any CH01 result could be attributed to training time rather than embedding quality. **This is the single most important control in this study.**

Mitigation plan: match FLOPs (or epoch-count + parameter count) between HGI and Check2HGI before P1 runs; see `docs/studies/check2hgi/archive/v1_wip_mixed_scope/TRAINING_BUDGET_DECISION.md` for the three options considered.

---

## The #2 reviewer risk

**Does Check2HGI inherit the HGI fclass → category shortcut?**

Fusion study's C29 finding (`docs/studies/fusion/issues/HGI_LEAKAGE_AUDIT.md`) showed the Gowalla HGI/POI2Vec pipeline has an fclass → category 1:1 determinism that makes category classification trivially solvable. Check2HGI's preprocessing may or may not inherit this. If it does, and it also propagates to next-POI predictions, our headline comparison is confounded. **Maps to CH14.** To be audited in P0.

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
