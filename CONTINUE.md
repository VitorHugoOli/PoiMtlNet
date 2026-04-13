# Continue: Full Fusion Ablation Study

## What Was Done

The MTLnet improvement work (merged from `copilot/create-improvement-plan-for-mtlnet`)
added new MTL optimizers (CAGrad, Aligned-MTL, DWA), a new architecture
(PLE), and new task heads (TCN Residual, Conv-Attention, Transformer
RelPos, Linear Probe). A Phase 4 head-swap ablation was run on HGI
alabama. All findings are consolidated in `docs/KNOWLEDGE_BASE_2026-04-13.md`
— **read that first** before doing anything.

## What Needs to Be Done

Run the full fusion ablation study — the paper's main experimental
contribution.

### Pre-Flight Checks

1. **Verify fusion data exists:**
   ```bash
   ls output/fusion/alabama/input/category.parquet
   ls output/fusion/alabama/input/next.parquet
   ```
   If missing, symlink from main repo:
   ```bash
   ln -s /Users/vitor/Desktop/mestrado/ingred/output/fusion output/fusion
   ```

2. **Verify checkins data exists:**
   ```bash
   ls data/checkins/Alabama.parquet
   ```
   If missing:
   ```bash
   ln -s /Users/vitor/Desktop/mestrado/ingred/data/checkins data/checkins
   ln -s /Users/vitor/Desktop/mestrado/ingred/data/miscellaneous data/miscellaneous
   ```

3. **Run tests to verify codebase:**
   ```bash
   PYTHONPATH=src python -m pytest tests/ -x -q
   ```
   Expected: 692 passed, 0 failed.

### Execution

The study is fully automated via `experiments/full_fusion_ablation.py`.
Run stages sequentially — each reads results from the previous stage.

```bash
# Stage 0: Baseline comparison (~15 min)
# Runs 3 fusion candidates + 1 HGI reference to verify fusion works
# and establish a comparison point.
python experiments/full_fusion_ablation.py --stage 0

# Stage 1: Architecture x Optimizer sweep (~2 hours)
# 25 candidates (5 archs x 5 optimizers) screened at 1f/10ep.
# Top 5 auto-promoted to 2f/15ep.
python experiments/full_fusion_ablation.py --stage 1

# Stage 2: Head variants on top-3 (~90 min)
# Tests DCN category head and TCN-residual next head on Stage 1 winners.
python experiments/full_fusion_ablation.py --stage 2

# Stage 3: Full 5-fold confirmation (~4-6 hours)
# Top 3 overall at 5f/50ep with full statistical reporting.
python experiments/full_fusion_ablation.py --stage 3

# Stage 4: Cross-state validation (~2 hours)
# Top 1 from Stage 3 run on florida to test generalization.
# REQUIRES: fusion inputs for florida.
python experiments/full_fusion_ablation.py --stage 4
```

Use `--dry-run` to preview commands without executing.

### After Each Stage

1. Check the summary CSV:
   ```
   results/ablations/full_fusion_study/{stage_label}/summary.csv
   ```

2. Copy `docs/full_ablation_study/ANALYSIS_TEMPLATE.md` and fill in
   results for that stage.

3. **Critical finding — fusion scale imbalance:**
   HGI has 15x larger L2 norm than Sphere2Vec, 8.7x larger than Time2Vec.
   The model almost entirely ignores the smaller source (0.7% dependence
   on Sphere2Vec, 2.4% on Time2Vec). Per-source normalization was tested
   and **hurts** (accuracy dropped 0.606 -> 0.504). Do NOT normalize.
   Stage 0's fusion-vs-HGI comparison will reveal if the auxiliary
   embeddings contribute anything at all. See
   `docs/full_ablation_study/FUSION_RATIONALE.md` for full analysis.

4. **Decision points:**
   - **After Stage 0:** If fusion joint score < HGI by >10%, stop and
     investigate fusion quality before proceeding.
   - **After Stage 1:** Check if equal_weight wins again (confirms prior
     HGI finding generalizes to fusion). Note any architecture changes.
   - **After Stage 2:** Head swaps likely won't help (Phase 4 showed
     this on HGI). If DCN category head does help, it's because of
     cross-features between Sphere2Vec and HGI dimensions — worth
     discussing in the paper.
   - **After Stage 3:** Report mean +/- std across 5 folds. Run paired
     t-test between top-1 and runner-up.

### Stage 4: Florida Prerequisite

Stage 4 requires fusion inputs for florida. Check:
```bash
ls output/fusion/florida/input/category.parquet
```
If missing, generate them:
```bash
PYTHONPATH=src python pipelines/fusion.pipe.py  # edit state="florida" at top
```

## Key Design Decisions to Understand

1. **Why 5 optimizers, not all 19?**
   Equal_weight, db_mtl, cagrad, aligned_mtl, uncertainty_weighting span
   the full spectrum: static, gradient-EMA, conflict-averse, eigendecomp,
   learned. Prior ablation showed the rest don't reach top-3. Reviewers
   expect uncertainty_weighting (Kendall 2018).

2. **Why default heads in Stage 1?**
   Phase 4 proved that standalone head rankings don't transfer to MTL.
   Stage 2 tests heads only on winning backbones to isolate the effect.

3. **Why no PLE?**
   PLE scored worst in Phase 4 (joint=0.235). Excluded from the main
   study but can be added as supplementary if CGC wins again.

4. **Fusion is task-specific:**
   Category gets Sphere2Vec(64)+HGI(64), Next gets HGI(64)+Time2Vec(64).
   Both are 128-dim but from different embedding sources. See
   `docs/full_ablation_study/FUSION_RATIONALE.md` for justification.

## Reference Documents

| Document | Contents |
|----------|----------|
| `docs/KNOWLEDGE_BASE_2026-04-13.md` | **Start here** — all empirical findings, architectural insights, what was implemented and rejected |
| `docs/full_ablation_study/STUDY_DESIGN.md` | Full experimental protocol with rationale |
| `docs/full_ablation_study/CANDIDATE_MATRIX.md` | Every candidate by stage |
| `docs/full_ablation_study/FUSION_RATIONALE.md` | Why this fusion combination |
| `docs/full_ablation_study/ANALYSIS_TEMPLATE.md` | Template for reporting stage results |
| `docs/MTL_ABLATION_REPORT_2026-04-13.md` | Phase 3-4 results (head swaps, new implementations) |
| `docs/MTL_ABLATION_REPORT_2026-04-11.md` | Phase 1-2 results (arch + optimizer on HGI/DGI) |
| `plan/NEW_CANDIDATES_ANALYSIS.md` | Literature research on optimizers/architectures |
| `plan/HEAD_ARCHITECTURE_ANALYSIS.md` | Literature research on task heads |

## File Locations

| What | Where |
|------|-------|
| Study runner | `experiments/full_fusion_ablation.py` |
| Ablation runner | `src/ablation/runner.py` |
| Candidates | `src/ablation/candidates.py` |
| New losses | `src/losses/{cagrad,aligned_mtl,dwa}/loss.py` |
| New architecture | `src/models/mtl/mtlnet_ple/model.py` |
| New heads | `src/models/next/{next_tcn_residual,next_conv_attn,next_transformer_relpos}/head.py` |
| New head | `src/models/category/category_linear/head.py` |
| Results | `results/ablations/full_fusion_study/` |
