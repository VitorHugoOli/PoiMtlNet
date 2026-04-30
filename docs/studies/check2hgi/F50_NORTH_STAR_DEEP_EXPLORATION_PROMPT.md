# F50 — North-Star deep-exploration prompt (for the next agent)

**Mission:** explore the hyperparameter space, the input pipeline, and the data flow of the *current* north-star MTL model deeply enough to (a) deliver new paper-grade lifts on top of B9, (b) build a complete mental model of why B9 works, and (c) document the mapping from each knob → which part of the training dynamics it actually affects. This is the FINAL exploration round before the paper freezes.

**You are NOT** doing another broad ablation sweep. The previous F50 study (Apr 28–30, 2026) screened 70+ items across architecture, optimizer, and hyperparameters. Read the synthesis docs first, then pick targeted experiments based on what's still un-probed.

---

## 0 · Read these first (in order, ~90 min total)

1. `docs/studies/check2hgi/HANDOVER.md` — bootstrap + headline numbers + reproduction recipe.
2. `docs/studies/check2hgi/research/F50_T4_FINAL_SYNTHESIS.md` — canonical state of F50.
3. `docs/studies/check2hgi/research/F50_HISTORY.md` — chronological narrative; understand WHY each decision was made.
4. `docs/studies/check2hgi/research/F50_RESULTS_TABLE.md` — all paper-grade numbers.
5. `docs/studies/check2hgi/research/F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md` — temporal-dynamics narrative + α-growth mechanism.
6. `docs/studies/check2hgi/research/F50_D5_ENCODER_TRAJECTORY.md` — encoder-saturation receipt + Frobenius drift logging.
7. `docs/studies/check2hgi/research/F50_T4_C4_LEAK_DIAGNOSIS.md` — the C4 graph-prior leak; **all your runs MUST use `--per-fold-transition-dir`**.
8. `docs/studies/check2hgi/research/F50_T4_PRIOR_RUNS_VALIDITY.md` — which pre-2026-04-29 runs survive C4.
9. `docs/studies/check2hgi/research/F50_B2_F52_F65_F53_FINDINGS.md` — most recent follow-ups (B2 rejected, F52 P5 paper-grade tied, F65 cycling not the cause, F53 cw sweep flat).
10. `docs/studies/check2hgi/NORTH_STAR.md` — current champion config.

After reading those, you should know:
- The B9 champion = `mtlnet_crossattn + static_weight(cw=0.75) + next_gru cat + next_getnext_hard reg + per-head LR (1e-3/3e-3/1e-3) + cosine(max_lr=3e-3) + alternating-SGD (P4) + α-no-WD + min_best_epoch=5 + per-fold log_T`.
- Δreg = +3.34 pp vs H3-alt, paired Wilcoxon p=0.0312, 5/5 positive on BOTH tasks at FL.
- The bottleneck is *temporal*, not architectural — reg encoder physically saturates at ep 5–6 while cat keeps drifting through ep 38; α grows late (ep 30+) but reg-best is at ep 5.
- Cross-attention mixing is structurally dead at FL (3-way confirmed: P1 ≈ F52 P5 ≈ F53 cw sweep).
- C4 leak inflated all pre-fix `next_getnext_hard*` numbers by 13–17 pp uniformly.

---

## 1 · Mandatory gates (your runs MUST satisfy these, no exceptions)

| gate | reason | flag |
|---|---|---|
| Per-fold transition matrix | C4 leak inflates by 13–17 pp without it | `--per-fold-transition-dir output/check2hgi/<state>` |
| Min best-epoch = 5 | GETNext α init artifact captures ep 0–2 as "best" | `--min-best-epoch 5` |
| Per-head LR triplet | B9 requires it (α-no-WD group construction) | `--cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3` |
| Static weight loss | Required by `--alternating-optimizer-step` | `--mtl-loss static_weight --category-weight 0.75` |
| Determinism | Paired Wilcoxon needs aligned folds | `--seed 42 --no-folds-cache` |
| 5 folds × 50 epochs | Paper-grade significance threshold | `--folds 5 --epochs 50` |

**Default per-fold log_T setup:**
```bash
python scripts/compute_region_transition.py --state florida --per-fold
```
(Already done for FL/AL/AZ/GA/CA on the current env — verify with `ls output/check2hgi/florida/region_transition_log_fold*.pt`.)

---

## 2 · Current champion (your starting point)

```bash
python scripts/train.py \
  --task mtl --task-set check2hgi_next_region \
  --state florida --engine check2hgi \
  --model mtlnet_crossattn \
  --cat-head next_gru --reg-head next_getnext_hard \
  --reg-head-param d_model=256 --reg-head-param num_heads=8 \
  --reg-head-param transition_path=output/check2hgi/florida/region_transition_log.pt \
  --task-a-input-type checkin --task-b-input-type region \
  --folds 5 --epochs 50 --seed 42 --batch-size 2048 \
  --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
  --gradient-accumulation-steps 1 \
  --per-fold-transition-dir output/check2hgi/florida \
  --no-checkpoints --no-folds-cache \
  --min-best-epoch 5 \
  --mtl-loss static_weight --category-weight 0.75 \
  --alternating-optimizer-step \
  --scheduler cosine --max-lr 3e-3 \
  --alpha-no-weight-decay
```

**Reference numbers** (FL clean, ≥ep5): reg top10 = 63.47 ± 0.75; cat F1 = 68.59 ± 0.79.

Reference run dir: `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260429_1813`.

---

## 3 · Recommended exploration order (pick one tier at a time)

Each tier has a hypothesis. Don't run the next tier until you've answered the current one. **Rule:** every paper-grade claim needs paired Wilcoxon p ≤ 0.0625 with 5/5 directional folds.

### Tier 1 — Multi-seed variance (validate the champion is not a seed artifact)

**Hypothesis:** the +3.34 pp Δreg is robust to seed noise.

- Run B9 at seed ∈ {0, 1, 7, 42, 100} on FL (keep folds=5 each).
- Run H3-alt anchor at the same 5 seeds.
- Per-seed paired Wilcoxon B9 vs H3-alt; aggregate to a meta-Wilcoxon across seeds.
- **Decision rule:** if 4/5 seeds show Δ ≥ +2.5 pp → robust. If only 2-3/5 → fragile, paper claim weakens to "FL-strong with seed-conditioning."

ETA: ~3.5 hours on a 4090 (10 × 19 min).

### Tier 2 — Encoder/backbone capacity

**Hypothesis:** reg-side encoder saturation (D5 finding) is partly a capacity bottleneck. Larger reg encoder might delay saturation.

Knobs to sweep (1 fold × 50 epochs first as smoke; 5 folds only if smoke promising):
| knob | current | sweep | hypothesis |
|---|---:|---|---|
| `encoder_layer_size` | 256 | {128, 384, 512} | larger encoder = later saturation |
| `num_encoder_layers` | 2 | {1, 3, 4} | more depth = richer representations |
| `encoder_dropout` | 0.1 | {0.05, 0.2, 0.3} | regularize against early saturation |
| `shared_layer_size` | 256 | {128, 384, 512} | (must match encoder out) |
| `num_crossattn_blocks` | 2 | {1, 3, 4} | depth of (dead) mixing — likely no-op given F52 |
| `num_crossattn_heads` | 4 | {2, 8, 16} | head-count vs head-dim trade-off |
| `crossattn_ffn_dim` | 256 | {128, 512, 1024} | F52 says FFN does the work — does width help? |

CLI: pass via `--task-a-input-type` etc. Some need code patches; check `MTLnet.__init__` and `MTLnetCrossAttn.__init__`.

**Smoke before paper-grade:** 1 fold × 50 ep × bs=2048 → check if reg val plateau shifts past ep 5–6. If yes, escalate to 5 folds.

### Tier 3 — Optimizer/scheduler "second-order" hyperparameters

**Hypothesis:** the scheduler shape matters more than peak LR; under-explored corners exist.

| knob | current | sweep | hypothesis |
|---|---:|---|---|
| `weight_decay` | 0.05 | {0.0, 0.01, 0.1, 0.2} | α is exempt; rest of model under-regularized? |
| AdamW `eps` | 1e-8 | {1e-7, 1e-6} | numerical stability in cosine tail |
| `gradient_accumulation_steps` | 1 | {2, 4, 8} | effective batch size 4096–16384 (P4 needs grad-acc=1 strictly!) |
| `max_grad_norm` | 1.0 | {0.5, 2.0, 5.0, ∞} | tighter clipping → smoother α trajectory? |
| `--scheduler cosine`'s decay rate | full | restart every 25/30 ep | cosine restarts after first reg-best |
| `--pct-start` for OneCycle | 0.3 | {0.1, 0.4, 0.5} | (only relevant if revisiting OneCycle) |
| Cosine `eta_min` | 0 | {1e-5, 1e-4} | non-zero floor in late epochs |
| Optimizer family | AdamW | Lion, AdaFactor, RAdam, Sophia | second-order info or sign-momentum |

**Caveat:** P4 alternating-SGD requires `--gradient-accumulation-steps 1`. If you sweep grad-acc, also disable P4.

**Smoke pattern:** 1 fold × 30 ep, look for divergence/plateau-shift; promote to 5f×50ep only if smoke shifts reg-best epoch or plateau height.

### Tier 4 — Input pipeline (data-flow exploration)

**Hypothesis:** input construction is locked at SLIDE_WINDOW=9 + non-overlapping; this is unexplored territory.

Read these files first:
- `src/configs/model.py` — `InputsConfig.SLIDE_WINDOW = 9`, `PAD_VALUE = 0`
- `src/data/inputs/core.py` — `generate_sequences()` builds non-overlapping sliding windows
- `src/data/folds.py` — `FoldCreator` → `StratifiedGroupKFold(seed=42)`
- `src/data/dataset.py` — `POIDataset` wrapper

Knobs:
| knob | current | sweep | hypothesis |
|---|---|---|---|
| `SLIDE_WINDOW` | 9 | {5, 7, 11, 15, 20} | longer context → more α growth signal? |
| Stride (non-overlapping vs overlapping) | non-overlapping | overlap=1, 2, 3 | overlapping → more samples per user → reg saturates later |
| `PAD_VALUE` | 0 | learned pad token (requires model patch) | informative padding |
| Batch size | 2048 | {512, 1024, 4096} | smaller batch + more grad-acc steps = different α steps/epoch |
| Train/val split | 80/20 within fold | tested via fold structure | (no sweep — fixed by StratifiedGroupKFold) |
| `--task-a-input-type` | checkin | poi (POI-level embeddings) | per-visit context vs POI summary |
| Embedding precision | fp32 | fp16 (autocast already on CUDA) | check fp16 numerical stability of α growth |

**Mandatory smoke:** before changing `SLIDE_WINDOW` or stride, run `python pipelines/create_inputs.pipe.py` to regenerate input parquets — the cached parquets at `output/check2hgi/florida/input/` use the current config.

### Tier 5 — Reg head internals

**Hypothesis:** `next_getnext_hard`'s α scalar is the bottleneck; head-internal architectural choices matter.

Read first:
- `src/models/next/next_getnext_hard/head.py` — α scalar + log_T integration
- `src/models/next/next_getnext_hard_hsm/head.py` — hierarchical softmax variant

Knobs (some require patching):
| knob | current | sweep | hypothesis |
|---|---|---|---|
| α init | 0.1 | {0.0, 0.5, 1.0, 2.0} | larger init → α already grown at ep 0 → reg-best shifts? |
| α activation | identity | {sigmoid, softplus, exp} | bound α growth to a saturation curve |
| α layer-norm scale | none | trainable γ around α | normalize α growth per fold |
| `transition_path` smoothing | per-fold log_T | + Laplace ε ∈ {1e-3, 1e-2, 1e-1} | denser prior matrix |
| log_T bidirectional | one-way | symmetric (T + Tᵀ)/2 | symmetric prior |
| `next_getnext_hard_hsm` cat tree | flat | hierarchical | already tested; rejected. Skip. |

**Code change required for most of these.** Add unit tests in `tests/test_models/`.

### Tier 6 — Loss landscape

**Hypothesis:** static_weight cw=0.75 is locked, but the actual loss components / weights inside each task are unexplored.

| knob | current | sweep | hypothesis |
|---|---|---|---|
| Cat task loss | weighted CE (class-balanced) | focal (γ=2), label-smoothing (ε=0.1), seesaw | imbalanced cat handling |
| Reg task loss | CE on logits | + auxiliary KL to log_T prior | regularize α growth |
| Joint loss | sum | geometric mean (NashMTL-style) | balanced gradient magnitude |
| `--use-class-weights` | True | False, computed-per-fold | sampler vs weighted CE |

**Caveat:** D8 cw=0 (cat-loss removed) gave reg = 74.06 → reg-side improvements via cat-loss reweighting probably won't unlock further.

### Tier 7 — Cross-state CA + TX (P3 portability) — **A100 ONLY**

This is paper-blocking but locked on infrastructure. Run `scripts/run_p3_ca_unblock_attempt.sh` once you have an A100 (40-80 GB GPU). The script's recipe is B9 verbatim. Per-fold log_T already exists at `output/check2hgi/california/region_transition_log_fold{1..5}.pt`.

Decision tree for outcomes: `docs/studies/check2hgi/research/C05_P3_NULL_RESULT_FALLBACK.md`.

---

## 4 · Methodology guidelines (don't skip)

### Smoke → 5-fold promotion

For each new knob:
1. **Smoke (1 fold × 30 ep, bs=2048, ~3 min):** check if reg val curve shifts at all relative to B9 reference. If reg-best epoch shifts < 1 epoch AND peak shifts < 0.5 pp → don't promote.
2. **Half-grade (2 folds × 50 ep, bs=2048, ~7 min):** if smoke promising, run 2 folds for σ estimate.
3. **Paper-grade (5 folds × 50 ep, bs=2048, ~17 min on FL):** if 2-fold result still positive, run full 5 folds.

### Acceptance criteria (paper-grade ⇔ headline-eligible)

A new champion candidate must satisfy ALL:
- Δreg ≥ +0.5 pp vs B9 (above measurement noise σ_pool ≈ 0.7 pp)
- Δcat ≥ −0.3 pp (don't degrade cat by more than 1 σ)
- paired Wilcoxon p ≤ 0.0625 on reg
- ≥ 4/5 folds positive on reg (5/5 is best)
- No fold collapses (< 95% of mean reg)

### Diagnostic logging (already in mtl_cv.py)

Use the existing per-epoch diagnostics for mechanism analysis:
- `head_alpha` — α scalar trajectory (F63)
- `reg_encoder_drift_from_init`, `reg_encoder_step_drift` — encoder Frobenius drift (D5)
- `cat_encoder_drift_from_init`, `cat_encoder_step_drift` — cat-side comparison
- `reg_encoder_l2norm`, `cat_encoder_l2norm` — current norms
- `grad_cosine_shared` — shared-backbone gradient cosine
- `grad_norm_next_region_shared`, `grad_norm_next_category_shared` — per-task grad norms
- `loss_weight_*`, `loss_ratio_*` — for loss-weight diagnostics
- `gate_entropy_*` — for PLE / gating models

Plot any new finding's α + encoder-drift trajectories. Compare against B9's reference at `figs/f63_alpha_trajectory.png` and `figs/f50_d5_encoder_trajectory.png`.

### Paired analysis

Use `scripts/analysis/f50_b2_f52_f65_f53_analysis.py` — it ingests per-fold val CSVs and emits the standard paired-Wilcoxon markdown table. Extend it (don't rewrite) for new arms.

---

## 5 · What NOT to explore (settled or out-of-scope)

- ❌ NashMTL / GradNorm / FAMO / Aligned-MTL on top of B9 — tested and rejected (Tier 1.x failures).
- ❌ Cross-Stitch / MMoE / DSelectK at FL — architectural alternatives all tied or worse.
- ❌ PLE — Pareto-WORSE under leak-free (NEW finding, paper-grade negative).
- ❌ B2/F64 warmup-decay reg_head LR — REJECTED, Pareto-dominated.
- ❌ F62 two-phase (cw=0→0.75 step) — REJECTED.
- ❌ F65 min-size-truncate joint loader — TIED with B9, cycling not the cause.
- ❌ `next_getnext_hard_hsm` (HSM head) — STL +0.21 vs flat, MTL −3.01. Done.
- ❌ category_weight ∈ {0.0, 0.25, 0.50} — F53 sweep done, all flat.
- ❌ `next_lstm` / `next_tcn` cat heads — F27 confirmed `next_gru` is the universal cat head (Path A).
- ❌ B10 batch=1024 — tested, loses by −2.54 pp Δreg.
- ❌ Identity-crossattn (F52 P5) and disable-crossattn (P1) — both tied with B9; cross-attn mixing is structurally dead.

If you find yourself proposing one of the above, re-read the relevant doc first.

---

## 6 · Output expectations

For each new finding, produce:

1. **A short findings doc** at `docs/studies/check2hgi/research/F50_<name>_FINDINGS.md` with:
   - Hypothesis (1 sentence)
   - Setup (recipe + run dirs)
   - Results table (paired Wilcoxon)
   - Interpretation (what mechanism does this affect — encoder saturation? α growth? loss-weight?)
   - Verdict (paper-grade / tied / rejected)
   - Cross-references

2. **Update `F50_T4_PRIORITIZATION.md`** with a new log line. Keep the existing format:
   ```
   | YYYY-MM-DD HH:MM | **<finding name>**: <one-line outcome>. Full doc: `<filename>`. |
   ```

3. **If a new champion emerges:** update `NORTH_STAR.md` Champion section + `F50_T4_FINAL_SYNTHESIS.md` headline + `F50_RESULTS_TABLE.md` table.

4. **Commit at every paper-grade landing.** Push to `worktree-check2hgi-mtl` branch.

5. **DO NOT modify** the C4 fix, the per-head optimizer construction, the F61 selector, or the per-fold log_T builder unless you find a bug in them.

---

## 7 · Compute budget guidance

On a 24 GB 4090 at FL:
- 1 fold × 50 ep × bs=2048 ≈ 3 min
- 5 folds × 50 ep × bs=2048 ≈ 17 min
- 5 folds × 100 ep × bs=2048 ≈ 35 min

Realistic single-day budget: 30–50 5-fold runs. Plan accordingly:
- Tier 1 multi-seed: 10 runs × 17 min = ~3 hours
- Tier 2 capacity sweep: ~7 knobs × 1f smoke + 3 paper-grade = ~1 hour
- Tier 3 optimizer sweep: ~6 knobs × 1f smoke + 2 paper-grade = ~45 min

**Allocate >50% of compute to Tier 1 (multi-seed)** before touching Tiers 2-6. Without seed-robustness the +3.34 pp claim may not survive review.

---

## 8 · Useful scripts (already in the repo)

```bash
scripts/run_p1_h3alt_f62_catchup.sh        # Template for B9 anchor + F62
scripts/run_f50_d5_encoder_traj.sh         # D5 paired runs (1-fold mechanism diagnostic)
scripts/run_f50_b2_f52_f65_fl.sh           # B2 + F52 + F65 follow-up template
scripts/run_f53_cw_sweep_fl.sh             # cw sensitivity sweep template
scripts/run_p3_ca_unblock_attempt.sh       # CA P3 (A100-targeted)
scripts/runpod_fetch_data.sh               # Fetch state data from Drive
scripts/compute_region_transition.py       # Build per-fold log_T (mandatory pre-flight)
scripts/analysis/f50_b2_f52_f65_f53_analysis.py  # Paired-Wilcoxon analyzer
scripts/analysis/f50_d5_encoder_traj_plot.py     # D5 plot generator
```

Tests:
```bash
PYTHONPATH=src pytest tests/test_training/ tests/test_models/ tests/test_tracking/ -q
# Should report ~210 passed.
```

---

## 9 · Closing — what success looks like

A successful exploration round produces:
- **Multi-seed validation** of the +3.34 pp claim (gives or takes paper-grade confidence).
- **At least 2 new mechanism receipts** beyond F63 α + D5 encoder (e.g., gradient cosine over training, loss-landscape curvature, batch-level α step size).
- **Either a new champion** with Δ ≥ +0.5 pp vs B9 paper-grade **OR** a documented "we explored everything and B9 is locked" verdict.
- **A complete data-flow doc** explaining how a single check-in row at the parquet level becomes a logit on next_region — input pipeline → fold split → DataLoader → encoder → cross-attn → head → loss → backprop. Useful for future agents to debug or extend.

When done, write `docs/studies/check2hgi/research/F51_NORTH_STAR_DEEP_DIVE_FINDINGS.md` summarizing everything, commit, and update HANDOVER.md to point to it.

Good luck.
