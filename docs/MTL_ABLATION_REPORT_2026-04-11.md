# MTLnet Ablation Report (2026-04-11)

## Scope

This report consolidates the main findings and implementation outcomes from the MTLnet upgrade and ablation work done in this thread:

- Plan review and correction (validity-first sequencing).
- Implementation of additional MTL losses and architecture variants.
- Canonical experiment runner and candidate matrix for repeatable ablations.
- Full staged ablation runs on:
  - `engine=dgi`, `state=alabama`
  - `engine=hgi`, `state=alabama` (using copied local baseline artifacts)

---

## High-Impact Findings

1. **Original “sequence collapse before shared backbone” diagnosis was stale.**  
   Current model path already preserves sequence structure into shared components.

2. **Validity issues were more important than architecture changes.**  
   Early priorities were correct experiment mechanics, joint-model selection, and consistent command-path execution.

3. **Canonical CLI exposure mattered for reproducibility.**  
   Running losses/architectures only via ad-hoc config shims was too fragile; candidates must be callable through standard tooling.

4. **Loss/architecture interactions were engine-dependent.**
   - On `dgi`, `DSelect-k + db_mtl` won promoted runs.
   - On `hgi`, `CGC s2t2 + equal_weight` won promoted joint score.

5. **Optimization objective tradeoff remains real.**  
   Joint score winners are not always next-task winners; select checkpoint by target objective, not single global intuition.

---

## What Was Implemented

### 1) New/expanded optimization candidates

Implemented and integrated into the candidate workflow:

- `equal_weight`
- `static_weight`
- `uncertainty_weighting`
- `famo`
- `fairgrad`
- `bayesagg_mtl`
- `go4align`
- `excess_mtl`
- `stch`
- `db_mtl`

Related docs were added under `docs/mtl_optimizers/`.

### 2) Architecture candidates

Ablation matrix includes:

- `mtlnet` (base)
- `mtlnet_mmoe` (`num_experts=4`)
- `mtlnet_cgc` variants:
  - `(shared=2, task=1)`
  - `(shared=1, task=1)`
  - `(shared=2, task=2)`
  - `(shared=4, task=1)`
- `mtlnet_dselectk` (`num_experts=4`, `num_selectors=2`, `temperature=0.5`)

### 3) DSelect-k variant

Implemented `mtlnet_dselectk` as a sequence-aware MoE-style variant with selector/gate diagnostics and integration into model registry and candidate configs.

### 4) Repeatable ablation workflow

Added/used:

- `experiments/mtl_candidates.py`
- `experiments/run_mtl_ablation.py`

to run staged matrices and promoted reruns in a reproducible way.

---

## Ablation Protocol Used

For both DGI and HGI architecture sweeps:

1. Candidate pool: 21 architecture+loss combinations (`equal_weight`, `db_mtl`, `fairgrad_a20` across all architecture variants above).
2. Stage A: `1 fold x 10 epochs` on all candidates.
3. Ranking metric: `joint_score` from run summary (`0.5 * next_macro_f1 + 0.5 * category_macro_f1`).
4. Stage B (promotion): top 3 candidates rerun with `2 folds x 15 epochs`.
5. Seed: `42`.
6. State: `alabama`.

---

## DGI Results

### Stage A (1 fold x 10 epochs) top candidates

1. `arch_cgc_s2t1_db_mtl` — joint `0.3288`
2. `arch_dselectk_e4k2_db_mtl` — joint `0.3262`
3. `arch_dselectk_e4k2_fairgrad_a20` — joint `0.3246`

### Stage B (2 folds x 15 epochs) promoted results

1. `arch_dselectk_e4k2_db_mtl` — joint `0.3337`, next_f1 `0.2469`, category_f1 `0.4204`
2. `arch_dselectk_e4k2_fairgrad_a20` — joint `0.3334`, next_f1 `0.2449`, category_f1 `0.4219`
3. `arch_cgc_s2t1_db_mtl` — joint `0.3299`, next_f1 `0.2349`, category_f1 `0.4249`

**Best DGI model (joint):** `arch_dselectk_e4k2_db_mtl`

---

## HGI Results (Local Copied Baseline)

Baseline artifacts copied from:

- `/Users/vitor/Desktop/mestrado/ingred/output/hgi/alabama`

to:

- `/Users/vitor/Desktop/mestrado/ingred/.claude/worktrees/mtlnet-improve/output/hgi/alabama`

Note: first sandbox attempt failed due `torch_shm_manager` sandbox restrictions; rerun executed outside sandbox successfully.

### Stage A (1 fold x 10 epochs) top candidates

1. `arch_cgc_s2t2_equal` — joint `0.4300`
2. `arch_cgc_s2t2_db_mtl` — joint `0.4291`
3. `arch_dselectk_e4k2_db_mtl` — joint `0.4223`

### Stage B (2 folds x 15 epochs) promoted results

1. `arch_cgc_s2t2_equal` — joint `0.4855`, next_f1 `0.2591`, category_f1 `0.7119`
2. `arch_cgc_s2t2_db_mtl` — joint `0.4775`, next_f1 `0.2576`, category_f1 `0.6974`
3. `arch_dselectk_e4k2_db_mtl` — joint `0.4748`, next_f1 `0.2638`, category_f1 `0.6858`

**Best HGI model (joint):** `arch_cgc_s2t2_equal`  
**Best HGI model (next_f1 among promoted):** `arch_dselectk_e4k2_db_mtl`

---

## Best Models (Current Recommendation)

If selecting one model per engine using promoted joint score:

- **DGI:** `arch_dselectk_e4k2_db_mtl`
- **HGI:** `arch_cgc_s2t2_equal`

If next-task performance is primary on HGI:

- Prefer `arch_dselectk_e4k2_db_mtl` (higher promoted next_f1).

---

## Important Knowledge / Interpretation

1. **`db_mtl` is consistently strong** across engines and architectures.
2. **DSelect-k is very competitive** and best on DGI promoted runs.
3. **CGC s2t2 is strongest on HGI joint/category behavior.**
4. **Engine choice changes the winner.**  
   Architecture/optimizer conclusions do not fully transfer between DGI and HGI.
5. **Joint metric can hide task-priority differences.**  
   For deployment, decide early whether joint balance or next-task specialization is the primary objective.

---

## Reproducibility Artifacts

### DGI

- `results/ablations/arch_variants_db_eq_fairgrad/all_1fold_10ep_seed42/summary.csv`
- `results/ablations/arch_variants_db_eq_fairgrad/all_promoted_2fold_15ep_seed42/summary.csv`

### HGI (successful rerun)

- `results/ablations/arch_variants_hgi_db_eq_fairgrad_escalated/all_1fold_10ep_seed42/summary.csv`
- `results/ablations/arch_variants_hgi_db_eq_fairgrad_escalated/all_promoted_2fold_15ep_seed42/summary.csv`

---

## Recommended Next Steps

1. Run `5-fold` confirmation for:
   - DGI: `dselectk_e4k2_db_mtl` vs `dselectk_e4k2_fairgrad_a20`
   - HGI: `cgc_s2t2_equal` vs `dselectk_e4k2_db_mtl`
2. Report confidence intervals / paired tests on fold-level metrics.
3. Add explicit objective profiles:
   - `joint_balanced`
   - `next_priority`
   - `category_priority`
4. Keep the staged protocol as default; avoid full-grid expansion before short-run evidence.
