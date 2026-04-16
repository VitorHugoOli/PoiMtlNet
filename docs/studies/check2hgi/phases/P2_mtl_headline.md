# Phase P2 — MTL headline

**Goal:** validate the headline paper claim — that the 2-task MTL `{next_poi, next_region}` on Check2HGI improves next-POI prediction over the P1 single-task baseline, without per-head negative transfer.

**Duration:** ~5h (2 states × 1 MTL config × 5f × 50ep).

**Embedded claims tested:**
- CH02 — MTL lift (headline).
- CH03 — No per-head negative transfer.

**Gates:** P1 complete with CH01 at least `partial`; frozen folds available.

---

## Experiments

| # | State | Task set | Optimiser | Seed | Purpose |
|---|---|---|---|---|---|
| P2.1.AL | AL | check2hgi_next_poi_region | NashMTL | 42 | CH02 + CH03 headline |
| P2.1.FL | FL | check2hgi_next_poi_region | NashMTL | 42 | CH02 + CH03 replication |

Default architecture: baseline MTLnet with FiLM + shared residual backbone. NextHeadMTL on both slots. `val_joint_lift` as checkpoint monitor.

**FL note:** enable `--use-class-weights` on FL to mitigate the 22.5% next_region majority class (see HANDOFF §Advisor-2 and critical-review Rev-2).

---

## Runbook

```bash
# P2.1.AL
STUDY_DIR=docs/studies/check2hgi python scripts/train.py \
  --state alabama --engine check2hgi --task mtl \
  --task-set check2hgi_next_poi_region \
  --folds 5 --epochs 50 --seed 42 \
  --gradient-accumulation-steps 1 --batch-size 4096

# P2.1.FL (with class weights)
STUDY_DIR=docs/studies/check2hgi python scripts/train.py \
  --state florida --engine check2hgi --task mtl \
  --task-set check2hgi_next_poi_region \
  --folds 5 --epochs 50 --seed 42 \
  --gradient-accumulation-steps 1 --batch-size 4096 \
  --use-class-weights
```

---

## Analysis

### CH02 — MTL > single-task on next_POI

Paired comparison across 5 folds:
- MTL next_poi Acc@10 (P2.1.AL) vs single-task next_poi Acc@10 (P1.1.AL.C2HGI)
- Same on FL

**Wilcoxon signed-rank, α=0.05.**

Verdicts:
- **confirm** — MTL > single by > 2pp AND paired-test p < 0.05.
- **partial** — positive trend, < 2pp or p < 0.10.
- **refute** — MTL ≤ single. Paper pivots to "check-in-level embeddings alone suffice, auxiliary task doesn't help on this data" — still publishable.

### CH03 — No per-head negative transfer

For each head:
- `val_next_poi_acc10` under MTL ≥ single-task baseline (from P1.1)
- `val_next_region_acc10` under MTL ≥ single-task baseline (from P1.2)

If either drops by > 2pp: **CH03 partial or refuted**. Document as "asymmetric transfer — one task gains, the other regresses." This is the #1 reviewer concern for MTL papers; plan honestly.

**FL-specific risk:** without class weights, NashMTL tends to over-weight the region head (large majority class gives low loss, so alpha concentrates on it), starving next_poi's gradient. We pre-emptively ran FL with `--use-class-weights`; if CH03 is still negative on FL, the diagnosis is likely still class-imbalance — report it.

---

## Output artefacts

```
docs/studies/check2hgi/results/P2/
├── P2.1.AL/
├── P2.1.FL/
└── ANALYSIS.md
```

---

## Decision gate → P3 and P5

Proceed to P3 + P5 only when:

1. Both P2.1 runs archived.
2. CH02 status ∈ {confirmed, partial, refuted} with `ANALYSIS.md`.
3. CH03 status documented. If refuted on both states, escalate before P3 — the rest of the plan depends on MTL being useful.

P3 and P5 are independent of each other after P2 completes.
