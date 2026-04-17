# Phase P2 — HEADLINE: MTL lift + no negative transfer

> **⚠️ STALE (2026-04-16).** Predates the task-pair correction (next_poi → next_category), the bidirectional-thesis clarification (both heads must improve), and the per-task-input-modality design. Phase numbering also shifted: the headline is now master-plan **P3**; this file's P2 number is obsolete. Authoritative plan: `docs/studies/check2hgi/MASTER_PLAN.md §P3`. Claims: `CLAIMS_AND_HYPOTHESES.md` (CH01 bidirectional; CH02 paired statistical test). Read for historical framing only.

**Goal:** resolve the paper's two headline claims — that the 2-task MTL `{next_poi, next_region}` on Check2HGI beats the single-task next-POI baseline (CH01), and that neither head regresses vs its single-task counterpart (CH02). Multi-seed by design for statistical power.

**Duration:** ~12h (2 states × 3 seeds × 5 folds × 50 epochs).

**Embedded claims tested:**
- **CH01 (headline)** — MTL next-POI Acc@10 > P1 single-task next-POI Acc@10.
- **CH02** — MTL per-head Acc@10 ≥ single-task per-head baseline (both heads).
- **CH11** — seed variance computed as a by-product.

**Gates:** P1 complete; CH04 floor passed on all states × tasks; frozen folds available.

---

## Why multi-seed is the default here

Review-agent finding §2.2: with n=5 folds, Wilcoxon signed-rank has **min possible p-value = 0.03125** (and only when all 5 folds agree on direction). A single fold flipping direction takes p above 0.05. For the "≥ 2 pp" effect size we need to detect, n=5 has near-zero power. n=15 (3 seeds × 5 folds) gives:
- min p = 2⁻¹⁵ ≈ 3 × 10⁻⁵ (plenty of headroom)
- realistic MDE at 95% power: ~1.2 pp
- matches how next-POI literature papers typically report variance

Pre-P1 code-delta list includes extending `scripts/train.py` to loop over a list of seeds inside a single invocation, archiving per-seed results.

---

## Experiments

| # | State | Seeds | Folds/seed | Purpose |
|---|---|---|---|---|
| P2.1.AL | AL | {42, 123, 2024} | 5 | CH01 + CH02 AL headline |
| P2.1.FL | FL | {42, 123, 2024} | 5 | CH01 + CH02 FL headline |

Task-set: `check2hgi_next_poi_region`. Baseline config = P2 champion from `docs/studies/check2hgi/state.json` (initially defaults: MTLnet + FiLM + NashMTL). FL runs with `--use-class-weights` to mitigate the 22.5% majority next_region class.

Per-run outputs: 3 × 5 = 15 paired samples per state.

---

## Runbook

```bash
# P2.1 AL
STUDY_DIR=docs/studies/check2hgi python scripts/train.py \
  --state alabama --engine check2hgi --task mtl \
  --task-set check2hgi_next_poi_region \
  --folds 5 --epochs 50 \
  --seed 42 --seed 123 --seed 2024 \
  --gradient-accumulation-steps 1 --batch-size 4096

# P2.1 FL (class-weighted)
STUDY_DIR=docs/studies/check2hgi python scripts/train.py \
  --state florida --engine check2hgi --task mtl \
  --task-set check2hgi_next_poi_region \
  --folds 5 --epochs 50 \
  --seed 42 --seed 123 --seed 2024 \
  --gradient-accumulation-steps 1 --batch-size 4096 \
  --use-class-weights
```

The `--seed` multi-arg ability is a pre-P1 code-delta (add to `scripts/train.py::_parse_args`). Alternative: loop externally.

---

## Analysis

### CH01 — MTL > single-task next-POI

For each state, compare per-paired-sample (seed × fold):
- MTL next_poi Acc@10 (from P2.1)
- single-task next_poi Acc@10 (from P1.1, same frozen folds)

Wilcoxon signed-rank on the 15 paired deltas, α=0.05.

**Verdicts:**
- `confirm` — MTL > single by ≥ 2pp AND paired-test p < 0.05.
- `partial` — positive trend, 1–2pp, OR p < 0.10.
- `refute` — MTL ≤ single. Paper pivots per MASTER_PLAN §"Minimum viable paper".

**Report in `ANALYSIS.md`:** paired delta distribution (histogram), mean ± std, Wilcoxon statistic, p-value.

### CH02 — No per-head negative transfer

For each head (next_poi, next_region):
- `val_acc10_<head>` under MTL ≥ single-task baseline

Paired Wilcoxon per head, same 15 samples. Verdict:
- Both heads ≥ single-task with p > 0.05 (non-significant drop) → `confirm CH02`.
- Either head drops significantly (p < 0.05 negative) → `refute CH02 for that head`; document as FL-specific or AL-specific depending on pattern. Investigate mitigations (class weights already enabled on FL).

### CH11 — Seed variance bound

From the 15 paired samples: compute std(next_poi_acc10) across the 15 values.

**Verdict:**
- std < 2pp → seed variance is below the "decisive" threshold. CH01 and subsequent claims can use 2pp as the confidence threshold.
- std ≥ 2pp → downgrade all claims' confidence thresholds to 1.5σ of observed variance.

---

## Output artefacts

```
docs/studies/check2hgi/results/P2/
├── P2.1.AL/
│   ├── seed_42/
│   │   ├── summary.json
│   │   ├── ood_summary.json
│   │   └── per_fold/
│   ├── seed_123/
│   ├── seed_2024/
│   └── aggregate.json           # 15 paired samples summary
├── P2.1.FL/
└── ANALYSIS.md                  # CH01 + CH02 + CH11 verdicts
```

---

## Decision gate → P3 and P5

Proceed to P3 and P5 when:
1. Both P2.1 runs archived with per-seed + aggregate summaries.
2. CH01 resolved (confirmed / partial / refuted with evidence).
3. CH02 resolved.
4. CH11 seed variance computed.

P3 and P5 are independent after P2 — run in parallel on separate machines if available.

**If CH01 refutes on both states:** pause before P3. The rest of the plan assumes MTL has value; without it, consider whether P3 (dual-stream, independent of MTL) is still worth running.
