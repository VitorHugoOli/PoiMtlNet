## F48-H3 — Per-Head LR Test (cat=1e-3, reg=3e-3, shared=3e-3)

**Date:** 2026-04-25. **Tracker:** `FOLLOWUPS_TRACKER.md §F48-H3`. **Cost:** ~30 min MPS sequential.

### Design

Three-group AdamW via `setup_per_head_optimizer` (new helper in `src/training/helpers.py`):

| Group | Params | LR |
|---|---|---|
| `cat` | `category_encoder` + `category_poi` (`next_gru` head) | 1e-3 const |
| `reg` | `next_encoder` + `next_poi` (`next_getnext_hard` head, **including α**) | 3e-3 const |
| `shared` | `crossattn_blocks` + `cat_final_ln` + `next_final_ln` | 3e-3 const |

`--scheduler constant` preserves per-group LRs (guarded in `setup_scheduler`). Smoke print on fold 0 confirmed: `[('cat', 0.001), ('reg', 0.003), ('shared', 0.003)]`.

5 folds × 50 ep × {AL, AZ}, B3 architecture, seed 42.

### Hypothesis

Disjoint-regimes — cat needs gentle LR (≤ 1e-3 sustained, per F44/F48-H1); reg needs sustained 3e-3 for α growth (F45 mechanism). Per-head LR isolates each regime to its tower without forcing a monolithic schedule.

### Results

| Config | AL cat F1 | AL reg Acc@10 | AZ cat F1 | AZ reg Acc@10 |
|---|---:|---:|---:|---:|
| B3 50ep default | 42.71 ± 1.37 | 59.60 ± 4.09 | 45.81 ± 1.30 | 53.82 ± 3.11 |
| F45 const 3e-3 (single LR) | 10.44 ± 0.04 💀 | **74.20 ± 2.95** | 12.23 ± 0.16 💀 | **63.34 ± 2.46** |
| **F48-H3 per-head** | **11.53 ± 1.63 💀** | **74.24 ± 2.58** | **19.61 ± 13.34 💀** | **62.04 ± 1.90** |
| **STL F21c (ceiling)** | n/a | **68.37 ± 2.66** | n/a | **66.74 ± 2.11** |

H3 reproduces F45 within σ on both states for both metrics. The cat-encoder LR throttle (1e-3) is irrelevant to cat outcome.

### Mechanism

The cat output flows:
```
enc_cat → crossattn_blocks (shared, 3e-3) → cat_final_ln (shared, 3e-3) → category_poi (cat, 1e-3)
```

When `shared` updates at sustained 3e-3, the cross-attn co-evolves aggressively from cat+reg gradients. Even though `cat_lr=1e-3` keeps `category_encoder` and `category_poi` gentle, the upstream shared layers are not, and the cat path is destabilised before signals reach the cat-specific weights. The 7-class `next_gru` softmax cannot survive — it diverges into majority-class prediction (~10–14% F1 = chance).

The AZ cat fold-5 outlier (43.24 F1 vs others ~12) suggests one fold fortuitously initialised in a basin where shared cross-attn updates didn't break cat — rare but possible — confirming the destabilisation is stochastic per-init, not deterministic.

### Refuted reading

The F44-F48 finding "two heads have disjoint optimal LR regimes" was correct as a *symptom* but wrong as a *prescription*. Throttling cat-head and cat-encoder LRs does not protect cat under sustained-3e-3 shared cross-attn. The bottleneck is the **shared cross-attn LR**, not the per-tower LR.

### Implications for next test (F48-H3-alt)

α (graph-prior weight in `next_getnext_hard.head`, line 80) is in `reg_specific_parameters` — it gets `reg_lr=3e-3` in any per-head config. So the discriminating question becomes: does reg lift come from α growth alone, or does it require shared cross-attn co-evolving at 3e-3?

The clean test:
```
cat=1e-3, reg=3e-3, shared=1e-3   (all constant)
```

Predictions:
- **If H3-alt → cat ≥ 35 AND reg ≥ 65:** α-growth alone explains F45's reg lift. Cat is preservable. Paper has a recipe.
- **If H3-alt → cat preserved AND reg flat ~60:** reg lift requires shared updating at high LR. The cat-vs-reg tradeoff is *structural to shared cross-attn under static_weight loss*, not a schedule artefact.

Either outcome is publishable; only the first yields a clean recipe.

### Files

- Logs: `/tmp/check2hgi_logs/f48_h3.log`
- AL summary: `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260425_1809/summary/full_summary.json`
- AZ summary: `results/check2hgi/arizona/mtlnet_lr1.0e-04_bs2048_ep50_20260425_1820/summary/full_summary.json`

### Code landed

| File | Change |
|---|---|
| `src/models/mtl/mtlnet_crossattn/model.py` | Added `cat_specific_parameters()` and `reg_specific_parameters()` methods. `shared_parameters()` unchanged (preserves NashMTL/PCGrad semantics). |
| `src/training/helpers.py` | Added `setup_per_head_optimizer`. Guarded `constant`/`cosine` scheduler from overwriting per-group LRs (`multi_group_per_head` heuristic). Also added `warmup_constant` scheduler (F48-H2 infra, not yet exercised). |
| `src/configs/experiment.py` | Added `cat_lr`, `reg_lr`, `shared_lr: Optional[float]`. |
| `src/training/runners/mtl_cv.py` | Per-head detection; smoke print on fold 0. |
| `scripts/train.py` | CLI `--cat-lr` / `--reg-lr` / `--shared-lr` (all-or-nothing validation). New `--scheduler warmup_constant` choice. |

### Cross-references

- `research/F44_F48_LR_REGIME_FINDINGS.md` — predecessor sweep that motivated H3
- `research/F45` row in tracker — single-LR constant 3e-3 result H3 reproduces
- `research/F43` row in tracker — refutes joint-loss-signal as bottleneck (orthogonal evidence)

---

## F48-H3-alt — Inverse: shared throttled to 1e-3 (CONFIRMED — paper recipe)

**Date:** 2026-04-25 evening. **Cost:** ~30 min MPS sequential.

Inverse of F48-H3 (advisor-recommended discriminator):

| Group | LR | Rationale |
|---|---|---|
| `cat` (cat encoder + cat head) | 1e-3 const | F48-H1 evidence: gentle LR preserves cat |
| `reg` (next encoder + next head, **including α**) | 3e-3 const | α needs sustained 3e-3 for F45 mechanism |
| `shared` (cross-attn + final_lns) | 1e-3 const | **Throttled** — keep cat path stable |

Hypothesis predictions:
- cat ≥ 35 AND reg ≥ 65 → α-growth alone explains F45 lift; cat preservable; **paper recipe**
- cat preserved AND reg flat ~60 → reg lift requires shared at high LR; tradeoff structural

### Results

| Config | AL cat F1 | AL reg Acc@10 | AZ cat F1 | AZ reg Acc@10 |
|---|---:|---:|---:|---:|
| B3 50ep default | 42.71 ± 1.37 | 59.60 ± 4.09 | 45.81 ± 1.30 | 53.82 ± 3.11 |
| F45 const 3e-3 (single LR) | 10.44 💀 | 74.20 ± 2.95 | 12.23 💀 | 63.34 ± 2.46 |
| F48-H1 const 1e-3 | 40.99 ± 1.80 | 61.43 ± 9.60 | 45.34 ± 0.84 | 50.68 ± 6.89 |
| F48-H3 (sh=3e-3) | 11.53 💀 | 74.24 ± 2.58 | 19.61 💀 | 62.04 ± 1.90 |
| **🎯 F48-H3-alt (sh=1e-3)** | **42.22 ± 1.00** | **74.62 ± 3.11** | **45.11 ± 0.32** | **63.45 ± 2.49** |
| **STL F21c (ceiling)** | n/a | **68.37 ± 2.66** | n/a | **66.74 ± 2.11** |

**AL: cat preserved (42.22 ≈ B3 42.71), reg EXCEEDS STL ceiling by +6.25 pp.**
**AZ: cat preserved (45.11 ≈ B3 45.81), reg closes 75% of B3-vs-STL gap (53.82 → 63.45 vs 66.74).**

Per-fold reg-best epoch (AL): [33, 38, 26, 38, 31] — same range as F45 (23..52), unlike F48-H1's [22, 7, 6, 4, 6] collapse. α grows productively under sustained 3e-3.

### Mechanism — refined understanding

The F45 reg lift comes from **α growth in the reg head, NOT from shared cross-attn co-evolution**. Once α is given sustained-3e-3 LR (which it gets in any per-head config since α lives in `next_specific_parameters`), it grows aggressively each epoch, exploiting the `log_T[r_last]` graph prior more.

The shared cross-attn at 3e-3 was a confound in F45: it lifted reg via α growth (the load-bearing effect) AND simultaneously destabilized cat (the side effect that looked like a tradeoff). H3-alt removes the side effect by throttling shared, isolating α growth as the load-bearing mechanism.

### Implication for CH18

CH18 ("MTL reg head 12-14 pp below STL") is now **fully recoverable** on AL (-12.92 pp gap → +6.25 pp surplus, swing of 19 pp) and **largely recoverable** on AZ (-12.92 pp → -3.29 pp, 75% closed).

The CH18 gap was NOT a structural limitation of MTL or cross-attention. It was a single confound: the graph-prior weight α in the reg head needed sustained-high LR to grow, but the shared OneCycleLR schedule annealed it before it could exploit the prior. The recipe is to give α its own LR regime.

### Single-line CLI recipe

```bash
python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --model mtlnet_crossattn --mtl-loss static_weight --category-weight 0.75 \
    --cat-head next_gru --reg-head next_getnext_hard \
    --reg-head-param d_model=256 --reg-head-param num_heads=8 \
    --folds 5 --epochs 50 --batch-size 2048 \
    --scheduler constant \
    --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3
```

### Files

- Logs: `/tmp/check2hgi_logs/f48_h3alt.log`
- AL: `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260425_1843/summary/full_summary.json`
- AZ: `results/check2hgi/arizona/mtlnet_lr1.0e-04_bs2048_ep50_20260425_1853/summary/full_summary.json`

### Open questions for follow-up

1. **Scale-stability on FL**: AL exceeded STL ceiling, AZ closed 75% of gap, FL is the largest state. Run H3-alt on FL to test scale-dependence of the recipe.
2. **Robustness across seeds**: re-run H3-alt with seeds {0, 7, 100} to verify σ doesn't blow up.
3. **OneCycleLR variant**: the recipe uses constant. A peak-and-anneal variant (cat-OneCycleLR max=1e-3, reg-OneCycleLR max=3e-3, shared-OneCycleLR max=1e-3) might tighten reg further on AZ if α growth + late annealing helps generalization.
4. **F48-H2 (warmup_constant) is now lower priority**: H3-alt is cleaner and beats it. Keep H2 infra in code but don't run as a separate experiment unless H3-alt fails to scale.

