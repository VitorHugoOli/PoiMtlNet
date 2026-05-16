# mtl-exploration — support study

> **Status (2026-05-15):** support / scaffold study. Not the main MTL research track. The main study will be conducted separately.
>
> This folder gathers the artefacts from a one-day deep audit of the existing MTL state of the art + a follow-up ablation that surfaced (and helped fix) a real codebase bug. Its purpose is to give the next agent (and the user) a clean baseline before committing to the *real* MTL study.

## What lives here

| File | Purpose | When to read |
|---|---|---|
| **[`INDEX.html`](INDEX.html)** ⭐ | Comprehensive state-of-the-art audit: every MTL backbone, loss, optimizer, head, ablation we have run, plus paper-canonical v11 numbers and the "is the input the right fit" question. | Read first when joining the MTL track. |
| **[`EXPERIMENT_NO_ENCODERS.md`](EXPERIMENT_NO_ENCODERS.md)** | Running diary of the *"what if we pass the 64-dim embeddings direct to cross-attn"* ablation. Includes the 2×2 factorial (cells A–D), advisor critique, cell-E thin hybrid, and the AZ multi-seed paper-grade confirmation. | Read second — the experimental detail. |
| **[`EXPERIMENT_HGI_SUBSTRATE.md`](EXPERIMENT_HGI_SUBSTRATE.md)** | Brief experiment: B9 with `--engine hgi` instead of `--engine check2hgi`. Result: MTL+HGI ≡ STL+HGI on both heads (cross-attn MTL is null under HGI substrate) — refutes the pre-leak-free CH18-substrate "−37 pp catastrophic break" claim. | Read fourth — the substrate × MTL interaction finding. |
| **[`LEAK_BLAST_RADIUS_AUDIT.md`](LEAK_BLAST_RADIUS_AUDIT.md)** | Technical audit of the `--folds < 5` × per-fold log_T `n_splits=5` mismatch bug — Part 1: codebase blast radius + trainer-level fix that landed 2026-05-15; Part 2: self-audit of this study's findings under the leak (no retractions). | Read third — the codebase impact. |
| `run_az_multiseed.sh`, `run_ms_chain_v2.sh` | Re-launchable invocation scripts for the canonical multi-seed reproducers. | Reference for reproducing. |
| `logs/` | Training stdout (per-run logs + master/queue logs). | Reference for raw trajectories. |

## What this study contributed (≈ 1 day of work)

1. **A comprehensive MTL state-of-the-art audit** ([`INDEX.html`](INDEX.html)) summarising every backbone / loss / head / schedule / ablation tested in the project, plus the paper-canonical v11 numbers and a sourced answer to *"is the input data the right fit for the model?"*. Used as the onboarding artefact for the upcoming main MTL study.

2. **A clean factorial discrimination** (`EXPERIMENT_NO_ENCODERS.md`): the B9 encoder MLP is over-engineered for `next_region` at AZ. A single `Linear(64→256)` per task matches the 2-layer MLP within statistical noise (Δ_reg = −0.01 pp, paired Wilcoxon p = 0.66 at n=20). The encoder's job is *dim-lift*, not non-linear feature transformation — the cross-attention block's own per-stream FFN provides the non-linearity.

3. **A loud-failure fix for the `--folds < 5` × per-fold log_T leak** (`LEAK_BLAST_RADIUS_AUDIT.md` + `docs/findings/MTL_FLAWS_AND_FIXES.md §2.13`). The bug was previously documented as script-level lore (warning headers in `run_f51_tier2_*.sh`); now hard-fails at the trainer with rebuild instructions. v11 paper numbers verified unaffected.

## What does NOT live here (and where to look)

- **Paper-canonical numbers:** `docs/results/RESULTS_TABLE.md §0` (v11). Never cite this folder for paper-facing numbers.
- **The MTL state-of-the-art narrative:** [`docs/MTL_ARCHITECTURE_JOURNEY.md`](../../MTL_ARCHITECTURE_JOURNEY.md) and [`docs/findings/`](../../findings/) (the F-trail).
- **The "real" follow-up studies:** [`docs/studies/canonical_improvement/`](../canonical_improvement/) and [`docs/studies/merge_design/`](../merge_design/).
- **The codebase fix details:** `src/training/runners/mtl_cv.py` (n_splits guard) and `scripts/compute_region_transition.py` (payload n_splits field), both 2026-05-15.

## Reproducing the experiments

The early single-fold smokes (`run_no_encoders.sh`, `run_chain.sh`,
`run_linear_encoders.sh`) were deleted on 2026-05-16 — they produced
LEAKY `--folds 1` runs that the n_splits guard now hard-fails on anyway,
and the bug-discovery story is documented in §Part 1 of
`LEAK_BLAST_RADIUS_AUDIT.md`. The two remaining scripts reproduce the
clean leak-free multi-seed result:

```bash
# AZ × {baseline, linear} × 4 seeds × 5 folds × 25ep (paper-grade Step 2)
bash docs/studies/mtl-exploration/run_az_multiseed.sh

# AL × {baseline, linear, linear_ln} × 4 seeds + AZ × linear_ln × 4 seeds
# (cross-state Step 3 + cell E)
bash docs/studies/mtl-exploration/run_ms_chain_v2.sh
```

Set `DATA_ROOT` and `OUTPUT_DIR` env vars if running outside the canonical repo root.

## Headline result (n=20 paired Wilcoxon, AL + AZ × 4 seeds × 5 folds × 25-ep)

The encoder ablation is **scale-conditional**:

| Axis | At AZ (1.5k regions) | At AL (1.1k regions) |
|---|---|---|
| reg `top10_acc_indist` | Cell E (`Linear + LayerNorm`) ≡ Cell D (baseline 2-MLP) within Δ = −0.12 pp (p=0.40, n=20) | Cell E ≡ Cell D within Δ = +0.14 pp (p=0.62, n=20) |
| cat F1 | Cell E ≡ Cell D within Δ = −0.025 pp (p=0.81, n=20) | **Cell D dominates Cell E by −2.57 pp (p=0.0001, n=20)** |

**Reading:** the 2-MLP encoder is over-engineered at AZ scale (Linear + LayerNorm matches it on both axes), but load-bearing for cat at AL scale. Simplification is scale-conditional, not universal.

**v11 paper-canon validation:** my multi-seed AZ baseline reg = **40.89 ± 1.95** ≈ v11's **40.78 ± 0.07** ✓ — the leak-free protocol matches paper canon. (AL baseline reg = 47.66 is 2.5 pp below v11's 50.17 due to the 25-epoch budget cap, which is symmetric across arms.)

Full result tables in [`EXPERIMENT_NO_ENCODERS.md §Step 3`](EXPERIMENT_NO_ENCODERS.md).

## Open question for the user

After the chain completes, the codebase implication is:

> If cell E (Linear + LayerNorm) closes the small cat lift at AZ (Δ_cat → 0) **and** generalises to AL, then the canonical B9 encoder simplifies to:
>
> ```python
> self.category_encoder = nn.Sequential(nn.Linear(feature_size, shared_layer_size), nn.LayerNorm(shared_layer_size))
> self.next_encoder     = nn.Sequential(nn.Linear(feature_size, shared_layer_size), nn.LayerNorm(shared_layer_size))
> ```
>
> saving ~150K params per encoder, ~4 LOC, at zero measured cost. This is a *simplification*, not an *improvement* — the canonical doesn't get beaten.

Whether to land this simplification in the canonical model code (vs leave it as a documented observation) is a decision for the main study.

## Pointers (most-relevant cited sources)

- `docs/results/RESULTS_TABLE.md` — paper-canonical numbers (v11)
- `docs/NORTH_STAR.md` — champion config (B9 + H3-alt scale-conditional)
- `docs/MTL_ARCHITECTURE_JOURNEY.md` — F-trail narrative
- `docs/CHANGELOG.md` — chronological findings
- `docs/findings/MTL_FLAWS_AND_FIXES.md §2.12, §2.13` — leak catalog
- `docs/findings/F50_D5_ENCODER_TRAJECTORY.md` — caveated (absolute numbers leak-inflated)
- `src/models/mtl/mtlnet_crossattn/model.py` — backbone; new flags: `no_task_encoders`, `linear_encoders`, `linear_ln_encoders`
- `src/training/runners/mtl_cv.py` — n_splits guard (2026-05-15 hard-fail)
- `scripts/compute_region_transition.py` — writer of the per-fold log_T, now stashes `n_splits` in payload

## Gaps surfaced during this study (also in §13 of INDEX.html)

These are observations from the deep audit; not actionable in this study but worth flagging for the main one:

- `docs/context/MTL_ARCHITECTURES.md` still quotes fusion-study legacy CGC/MMoE/DSelectK numbers (1-fold × 10-epoch on the legacy 7+7 task pair). Should be migrated or caveated.
- `docs/context/TASKS.md` describes the *legacy* task pair (`category` + `next-category`, both 7-class), not the current Check2HGI pair (`next_category` + `next_region`). The current pair is described in `AGENT_CONTEXT.md` but not in the canonical `context/` folder.
- `docs/COLAB_GUIDE.md` and `experiments/check2hgi_up/run_mtl_b3.py` still bake in the predecessor B3 recipe (`--max-lr 0.003`, no per-head LR).
- Flat single-check-in cat ablation has not been run — would isolate whether the +10 to +23 pp lift over the linear probe is sequence-driven or capacity-driven (open in §14 of INDEX.html).
- FAMO and Aligned-MTL launchers exist (F50 T1.3 / T1.4) but were never run under per-fold seed-tagged log_T.
- `mtlnet_crossstitch` backbone is scaffolded; never head-to-head with `mtlnet_crossattn`.
