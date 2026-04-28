# Data sources index

For every JSON in this `data/` tree, this file records:
- **Canonical original location** in the study (the run's full result dir, including checkpoints + classification reports + train/val curves).
- **Reproduction tag** passed to the trainer.
- **Date** the run was finalised.

The JSONs in this folder are extracted summaries (per-fold metrics or paired-test outputs). The canonical locations have the full artefacts.

---

## Linear probe (`linear_probe/`)

Generator: `scripts/probe/substrate_linear_probe.py`. No training (logistic regression on raw embeddings, ~2–4 sec per cell). All on `output/<engine>/<state>/input/next.parquet` last-window-position slice (cols 512..575), 5-fold StratifiedGroupKFold(seed=42).

| File | Canonical original | Run date |
|---|---|---|
| `alabama_check2hgi_last.json` | `docs/studies/check2hgi/results/probe/alabama_check2hgi_last.json` | 2026-04-27 |
| `alabama_hgi_last.json` | `docs/studies/check2hgi/results/probe/alabama_hgi_last.json` | 2026-04-27 |
| `arizona_check2hgi_last.json` | `docs/studies/check2hgi/results/probe/arizona_check2hgi_last.json` | 2026-04-27 |
| `arizona_hgi_last.json` | `docs/studies/check2hgi/results/probe/arizona_hgi_last.json` | 2026-04-27 |
| `alabama_check2hgi_pooled_last.json` | `docs/studies/check2hgi/results/probe/alabama_check2hgi_pooled_last.json` | 2026-04-27 |

---

## Cat STL per-fold (`cat_stl_per_fold/`)

Trainer: `scripts/train.py --task next --state $STATE --engine $ENGINE --model $HEAD --folds 5 --epochs 50 --seed 42 --no-checkpoints`. AdamW(1e-4, wd=0.01) + OneCycleLR(max=1e-2) + batch 1024 (the `default_next` config in `src/configs/experiment.py`).

For each row, the **per-fold JSON** contains `{fold_0..fold_4: {f1, accuracy}}` extracted from `<canonical_dir>/folds/foldN_info.json::diagnostic_best_epochs.next.metrics`. The canonical dir also has classification reports + train/val curves.

| File | Canonical original training dir | Tag |
|---|---|---|
| `AL_check2hgi_cat_gru_5f50ep.json` | `results/check2hgi/alabama/next_lr1.0e-04_bs1024_ep50_20260427_1713/` | (no tag — direct invocation) |
| `AL_hgi_cat_gru_5f50ep.json` | `results/hgi/alabama/next_lr1.0e-04_bs1024_ep50_20260427_1716/` | (no tag) |
| `AZ_check2hgi_cat_gru_5f50ep.json` | `results/check2hgi/arizona/next_lr1.0e-04_bs1024_ep50_20260427_1718/` | (no tag) |
| `AZ_hgi_cat_gru_5f50ep.json` | `results/hgi/arizona/next_lr1.0e-04_bs1024_ep50_20260427_1724/` | (no tag) |
| `AL_check2hgi_cat_single_5f50ep.json` | `results/check2hgi/alabama/next_lr1.0e-04_bs1024_ep50_20260427_1829/` | (C2 head sweep) |
| `AL_hgi_cat_single_5f50ep.json` | `results/hgi/alabama/next_lr1.0e-04_bs1024_ep50_20260427_1831/` | (C2) |
| `AZ_check2hgi_cat_single_5f50ep.json` | `results/check2hgi/arizona/next_lr1.0e-04_bs1024_ep50_20260427_1908/` | (C2) |
| `AZ_hgi_cat_single_5f50ep.json` | `results/hgi/arizona/next_lr1.0e-04_bs1024_ep50_20260427_1912/` | (C2) |
| `AL_check2hgi_cat_lstm_5f50ep.json` | `results/check2hgi/alabama/next_lr1.0e-04_bs1024_ep50_20260427_1833/` | (C2) |
| `AL_hgi_cat_lstm_5f50ep.json` | `results/hgi/alabama/next_lr1.0e-04_bs1024_ep50_20260427_1850/` | (C2) |
| `AZ_check2hgi_cat_lstm_5f50ep.json` | `results/check2hgi/arizona/next_lr1.0e-04_bs1024_ep50_20260427_1915/` | (C2) |
| `AZ_hgi_cat_lstm_5f50ep.json` | `results/hgi/arizona/next_lr1.0e-04_bs1024_ep50_20260427_1953/` | (C2) |

Orchestrators: `scripts/run_phase1_cat_stl.sh` (matched-head AL+AZ), `scripts/run_phase1_c2_head_sweep.sh` (C2 head sweep).

---

## Reg STL per-fold (`reg_stl_per_fold/`)

Trainer: `scripts/p1_region_head_ablation.py --heads next_getnext_hard --folds 5 --epochs 50 --seed 42 --input-type region --region-emb-source $ENGINE --override-hparams d_model=256 num_heads=8 transition_path=...`. AdamW + OneCycleLR + batch 2048.

Each per-fold JSON contains `{fold_0..fold_4: {acc1, acc5, acc10, mrr, f1}}` extracted from `<canonical_dir>::heads.next_getnext_hard.per_fold[i]`.

| File | Canonical original | Tag |
|---|---|---|
| `AL_check2hgi_reg_gethard_5f50ep.json` | `docs/studies/check2hgi/results/B3_baselines/stl_getnext_hard_al_5f50ep.json` | F21c (`stl_gethard`) |
| `AZ_check2hgi_reg_gethard_5f50ep.json` | `docs/studies/check2hgi/results/B3_baselines/stl_getnext_hard_az_5f50ep.json` | F21c (`stl_gethard`) |
| `AL_hgi_reg_gethard_5f50ep.json` | `docs/studies/check2hgi/results/P1/region_head_alabama_region_5f_50ep_STL_ALABAMA_hgi_reg_gethard_5f50ep.json` | `STL_ALABAMA_hgi_reg_gethard_5f50ep` |
| `AZ_hgi_reg_gethard_5f50ep.json` | `docs/studies/check2hgi/results/P1/region_head_arizona_region_5f_50ep_STL_ARIZONA_hgi_reg_gethard_5f50ep.json` | `STL_ARIZONA_hgi_reg_gethard_5f50ep` |

Orchestrator: `scripts/run_phase1_reg_stl.sh`.

> **Transition matrix is substrate-independent.** Both Check2HGI and HGI runs read the same `output/check2hgi/<state>/region_transition_log.pt` (regions are POI-derived, not embedding-derived).

---

## MTL counterfactual per-fold (`mtl_counterfactual_per_fold/`)

Trainer: `scripts/train.py --task mtl --state $STATE --engine hgi --task-set check2hgi_next_region --model mtlnet_crossattn --mtl-loss static_weight --category-weight 0.75 --reg-head next_getnext_hard --cat-head next_gru --folds 5 --epochs 50 --seed 42 --no-checkpoints`.

Each per-fold JSON contains:
- `_cat.json`: `{fold_0..fold_4: {f1, accuracy}}` from `<canonical>/folds/foldN_info.json::diagnostic_best_epochs.next_category.metrics`.
- `_reg.json`: `{fold_0..fold_4: {f1, acc1, acc5, acc10, acc10_indist, mrr}}` from `next_region.metrics` with `top10_acc_indist` aliased to `acc10`.

| File | Canonical original training dir |
|---|---|
| `AL_hgi_mtl_cat.json` + `AL_hgi_mtl_reg.json` | `results/hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260427_1746/` |
| `AZ_hgi_mtl_cat.json` + `AZ_hgi_mtl_reg.json` | `results/hgi/arizona/mtlnet_lr1.0e-04_bs2048_ep50_20260427_1759/` |

Orchestrator: `scripts/run_phase1_mtl_counterfactual.sh`.

**Comparator**: existing MTL B3 with Check2HGI substrate, lives at:
- `docs/studies/check2hgi/results/F27_validation/al_5f50ep_b3_cathead_gru.json` (cat F1 0.4271; reg only-aggregate stored, no per-fold)
- `docs/studies/check2hgi/results/F27_validation/az_5f50ep_b3_cathead_gru.json` (cat F1 0.4581)

> **Note:** the C2HGI MTL B3 runs only retain aggregate metrics in their result JSONs (no per-fold breakdown stored at run-time). For paired tests against the Phase-1 MTL+HGI, the recommended approach is to re-aggregate from the source `results/check2hgi/<state>/mtlnet_*` run dirs (they have the per-fold info JSONs). This is queued as a follow-up, not blocking — the Δ_cat ≈ +17 pp and Δ_reg ≈ +30 pp are far outside σ at both states.

---

## C4 POI-pooled per-fold (`c4_poi_pooled/`)

Generator chain:
1. `scripts/probe/build_check2hgi_pooled.py --state alabama` — writes `output/check2hgi_pooled/alabama/{embeddings.parquet, input/next.parquet}`.
2. `scripts/train.py --task next --state alabama --engine check2hgi_pooled --model next_gru --folds 5 --epochs 50 --seed 42 --no-checkpoints` — produces the matched-head STL run.

| File | Canonical original training dir |
|---|---|
| `AL_check2hgi_pooled_cat_gru_5f50ep.json` | `results/check2hgi_pooled/alabama/next_lr1.0e-04_bs1024_ep50_20260427_1826/` |

The linear-probe variant of C4 (head-free) is at `linear_probe/alabama_check2hgi_pooled_last.json`.

---

## Paired tests (`paired_tests/`)

Generator: `scripts/analysis/substrate_paired_test.py`. Inputs: per-fold JSONs from this folder. Outputs: paired-t + Wilcoxon + (where applicable) TOST non-inferiority at δ=2 pp.

Each output JSON contains:
- `check2hgi_per_fold` + `hgi_per_fold` + `deltas`
- `superiority`: `{n, mean, median, std, n_positive/negative/zero, paired_t_p_greater, wilcoxon_p_greater, shapiro_w, normality_assumed}`
- `non_inferiority_tost` (when `--tost-margin` provided): `{margin, p_lower_one_sided, non_inferior_at_alpha_0.05, shifted_mean}`

| File | Test |
|---|---|
| `alabama_cat_f1.json` | Matched-head cat (next_gru) Δ F1 — AL |
| `alabama_single_cat_f1.json` | Head-sensitivity probe (next_single) Δ F1 — AL |
| `alabama_lstm_cat_f1.json` | Head-sensitivity probe (next_lstm) Δ F1 — AL |
| `arizona_cat_f1.json` | Matched-head cat — AZ |
| `arizona_single_cat_f1.json` | Head-sensitivity probe — AZ |
| `arizona_lstm_cat_f1.json` | Head-sensitivity probe — AZ |
| `alabama_acc10_reg_acc10.json` | Matched-head reg (next_getnext_hard) Δ Acc@10 + TOST δ=0.02 — AL |
| `alabama_mrr_reg_mrr.json` | Matched-head reg Δ MRR + TOST — AL |
| `arizona_acc10_reg_acc10.json` | Matched-head reg Δ Acc@10 + TOST — AZ |
| `arizona_mrr_reg_mrr.json` | Matched-head reg Δ MRR + TOST — AZ |

Original location of the same JSONs: `docs/studies/check2hgi/results/paired_tests/`.

---

## Reverse map — from a paper-table number to the underlying JSON

Suppose a reviewer asks: "Where does the +14.52 pp AZ matched-head cat F1 lift come from?"

1. **Aggregate**: in `phase1_verdict.md §5.1` row "AZ next_gru".
2. **Per-fold deltas**: `data/paired_tests/arizona_cat_f1.json::deltas` → `[+13.46, +11.67, +15.46, +13.81, +14.61]`. Mean = +13.80? Wait — the paired-test JSON computes `c2hgi_per_fold − hgi_per_fold` per fold, so deltas should match. (Actual values: open the JSON.)
3. **Source per-fold metrics**: `data/cat_stl_per_fold/AZ_check2hgi_cat_gru_5f50ep.json` and `data/cat_stl_per_fold/AZ_hgi_cat_gru_5f50ep.json` (5 fold-dicts each).
4. **Training curves + classification reports**: per `cat_stl_per_fold` table above — `results/check2hgi/arizona/next_lr1.0e-04_bs1024_ep50_20260427_1718/` for C2HGI, `results/hgi/arizona/next_lr1.0e-04_bs1024_ep50_20260427_1724/` for HGI.

This three-level chain (aggregate → per-fold → training artefact) is what makes this folder audit-self-contained.
