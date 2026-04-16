# Phase P5 — Ablations

**Goal:** characterise the sensitivity of the P2 headline result to three design axes — head architecture, MTL optimiser, and random seed — enough to satisfy a reviewer's ask without blowing the compute budget.

**Duration:** ~4h (Alabama only; FL reserved for headline phases).

**Embedded claims tested:**
- CH08 — `next_mtl` transformer beats simpler sequence heads on next_POI.
- CH09 — NashMTL vs equal_weight vs CAGrad on the new task pair.
- CH10 — Seed variance is below the "decisive" threshold (≤ 2pp).

**Gates:** P2 complete.

---

## Experimental design

P5 is deliberately **lighter** than fusion's P1 arch × optim grid. We do not sweep 5 × 20 = 100 cells. Instead, each axis is ablated along a small, claim-justified menu.

### CH08 — Head architecture

Fix everything at P2 champion (dual-stream off; single-stream check-in emb only; MTLnet + FiLM; NashMTL). Vary the `task_a` head factory (and symmetrically `task_b`, since both are sequential).

| # | task_a head | task_b head | Purpose |
|---|---|---|---|
| P5.1.mtl (baseline) | next_mtl | next_mtl | P2 reference |
| P5.1.lstm | next_lstm | next_lstm | Classic recurrent |
| P5.1.gru | next_gru | next_gru | Classic recurrent |
| P5.1.tcn | next_tcn_residual | next_tcn_residual | TCN |
| P5.1.cnn | next_temporal_cnn | next_temporal_cnn | Shallow CNN |

5 cells. Alabama only. 5-fold × 50 epochs each. ~2h total.

### CH09 — MTL optimiser

Fix at P2 champion heads. Vary the MTL criterion.

| # | MTL criterion | Purpose |
|---|---|---|
| P5.2.nash (baseline) | nash_mtl | P2 reference |
| P5.2.eq | equal_weight | Static baseline; Xin-et-al. test |
| P5.2.cagrad | cagrad | Gradient-surgery alt |

3 cells. Alabama. ~1h.

### CH10 — Seed variance

Fix at P2 champion. Run 2 additional seeds.

| # | Seed | Purpose |
|---|---|---|
| P5.3.seed42 (baseline) | 42 | already done (P2.1.AL) |
| P5.3.seed123 | 123 | Variance sample |
| P5.3.seed2024 | 2024 | Variance sample |

2 new runs. Alabama. ~1h.

---

## Runbook

```bash
# P5.1 — head sweep
for head in next_mtl next_lstm next_gru next_tcn_residual next_temporal_cnn; do
  STUDY_DIR=docs/studies/check2hgi python scripts/train.py \
    --state alabama --engine check2hgi --task mtl \
    --task-set check2hgi_next_poi_region \
    --model-param head_a_factory=$head --model-param head_b_factory=$head \
    --folds 5 --epochs 50 --seed 42 \
    --gradient-accumulation-steps 1 --batch-size 4096
done

# P5.2 — optimiser sweep
for loss in nash_mtl equal_weight cagrad; do
  STUDY_DIR=docs/studies/check2hgi python scripts/train.py \
    --state alabama --engine check2hgi --task mtl \
    --task-set check2hgi_next_poi_region \
    --mtl-loss $loss \
    --folds 5 --epochs 50 --seed 42 \
    --gradient-accumulation-steps 1 --batch-size 4096
done

# P5.3 — seed sweep
for seed in 123 2024; do
  STUDY_DIR=docs/studies/check2hgi python scripts/train.py \
    --state alabama --engine check2hgi --task mtl \
    --task-set check2hgi_next_poi_region \
    --folds 5 --epochs 50 --seed $seed \
    --gradient-accumulation-steps 1 --batch-size 4096
done
```

**Note:** `--model-param head_a_factory=...` requires wiring in the CLI → task_set path (`resolve_task_set(preset, task_a_head_params=...)` or similar). Add to the pre-P5 code-delta list if not already present.

---

## Analysis

### CH08

Rank the 5 head variants by Acc@10 on next_poi. Confirm `next_mtl` wins (or partial: within noise of the winner).

### CH09

- If `nash_mtl` > `equal_weight` by ≥ 1pp on joint_lift → **confirm CH09 positive** (gradient-surgery helps).
- If within 1pp → **confirm CH09 negative** (equal_weight suffices; drop NashMTL's cvxpy dependency for simplicity).
- If `equal_weight` > `nash_mtl` → **refute CH09 positive** (strong finding — replicates Xin et al. beyond single-source embeddings).

### CH10

Compute fold-variance on next_poi Acc@10 across the 3 seeds. If std > 2pp → the "≥ 2pp lift" claims in CH01/CH02/CH06 need downgrading. If std ≤ 1pp → headline decisions are solid.

---

## Output

```
docs/studies/check2hgi/results/P5/
├── P5.1.heads/
│   ├── next_mtl/
│   ├── next_lstm/
│   ├── next_gru/
│   ├── next_tcn_residual/
│   └── next_temporal_cnn/
├── P5.2.optim/
│   ├── nash_mtl/
│   ├── equal_weight/
│   └── cagrad/
├── P5.3.seeds/
│   ├── seed123/
│   └── seed2024/
└── ANALYSIS.md
```

---

## Decision gate → branch merge

Branch merges when:

1. CH01, CH02, CH03 resolved (P1 + P2).
2. ≥ 2 of {CH06, CH07, CH11} resolved (P3 + P4).
3. CH08, CH09, CH10 resolved (this phase).
4. Legacy + fusion tests green.
5. `docs/PAPER_FINDINGS.md` has a check2HGI section drafted (sibling to fusion's).

If Tier A CH02 refutes, the branch still merges — infrastructure + honest empirical results stand.
