# CH15 — Transductive embedding-leakage audit

**Date:** 2026-04-15
**Verdict:** `planned` — implementation ready, awaiting compute allocation.

---

## What this audit measures

Check2HGI is trained on 100% of Gowalla check-ins (all users) before the downstream MTL task holds users out via StratifiedGroupKFold. Validation-fold users' entire trajectories shaped the embeddings the model is then evaluated on. This is the "upstream transductive leakage" flagged by fusion C30.

For next-POI prediction (our task), this is potentially more consequential than for category classification: the label IS a POI index whose embedding was directly shaped by the sequences being predicted.

## Quantification (Alabama, fold 0)

- Fold 0 val users: **321** (out of 1,622 total)
- Fold 0 val user check-ins: **21,971** (19.3% of 113,846 total)
- Remaining train check-ins: **91,875** (80.7%)

Training on 81% of the data should still produce usable embeddings (the preprocessing + GCN are designed for large-scale unsupervised pretraining where 80% coverage is normal).

## Procedure (when executed)

1. **Modify** `research/embeddings/check2hgi/preprocess.py` to accept an optional `exclude_userids: set[int]` parameter. When provided, filter `self.checkins` to exclude those users' rows before building the graph + node features + edges.

2. **Regenerate** Check2HGI embeddings for Alabama with fold-0 val users excluded:
   ```bash
   EXCLUDE_USERS_FOLD=0 python pipelines/embedding/check2hgi.pipe.py --state alabama
   ```
   (~20 min on MPS)

3. **Generate** next_poi and next_region labels from the held-out embeddings (same pipeline, pointing at the new artefacts).

4. **Run** single-task next-POI on fold 0 only, using the held-out embeddings:
   ```bash
   python scripts/train.py --state alabama --engine check2hgi --task next_poi \
     --folds 1 --epochs 50 --seed 42 \
     --folds-path <path_to_held_out_folds>
   ```

5. **Compare** val Acc@10 on fold 0:
   - Standard (transductive): from P1.1.AL (when it runs)
   - Held-out (inductive): from this audit

6. **Delta:**
   - `|Δ Acc@10| < 1 pp` → **CH15 bounded**: transductive leakage is small on this data.
   - `|Δ Acc@10| > 2 pp` → **CH15 significant**: paper limitations section must prominently mention this, and the community should retrain embeddings per-fold for rigorous evaluation (expensive but doable).

## Current status

- Code changes spec'd but not implemented.
- Estimated effort: ~2h (code change + retrain + eval + analysis).
- **Does not gate P1.** CH15 is Tier-E (limitation audit); its result doesn't change the CH01/CH02/CH03 headline claims' experimental protocol — it just characterises how much the transductive-training assumption inflates reported numbers.

## Recommendation

Run this AFTER P2 (when a P1.1.AL Acc@10 reference exists to compare against). That way the delta has a concrete anchor. Running it now would produce a number with nothing to compare to.
