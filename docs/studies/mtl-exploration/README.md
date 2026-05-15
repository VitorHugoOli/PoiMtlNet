# mtl-exploration — support study

> **Status (2026-05-15):** support / scaffold study. Not the main MTL research track. The main study will be conducted separately.
>
> This folder gathers the artefacts from a one-day deep audit of the existing MTL state of the art + a follow-up ablation that surfaced (and helped fix) a real codebase bug. Its purpose is to give the next agent (and the user) a clean baseline before committing to the *real* MTL study.

## What lives here

| File | Purpose | When to read |
|---|---|---|
| **[`INDEX.html`](INDEX.html)** ⭐ | Comprehensive state-of-the-art audit: every MTL backbone, loss, optimizer, head, ablation we have run, plus paper-canonical v11 numbers and the "is the input the right fit" question. | Read first when joining the MTL track. |
| **[`EXPERIMENT_NO_ENCODERS.md`](EXPERIMENT_NO_ENCODERS.md)** | Running diary of the *"what if we pass the 64-dim embeddings direct to cross-attn"* ablation. Includes the 2×2 factorial (cells A–D), advisor critique, cell-E thin hybrid, and the AZ multi-seed paper-grade confirmation. | Read second — the experimental detail. |
| **[`LEAK_BLAST_RADIUS_AUDIT.md`](LEAK_BLAST_RADIUS_AUDIT.md)** | Technical audit of the `--folds < 5` × per-fold log_T `n_splits=5` mismatch bug — what triggers it, blast radius across all v11 / F-trail / active-study claims, the trainer-level fix that landed 2026-05-15. | Read third — the codebase impact. |
| **[`STUDY_FINDINGS_LEAK_AUDIT.md`](STUDY_FINDINGS_LEAK_AUDIT.md)** | Self-audit of this study's own findings under the leak. Per-conclusion impact analysis + re-run decision matrix. | Skim — confirms why no re-runs needed. |
| `run_chain.sh`, `run_*.sh` | Re-launchable invocation scripts for every experiment phase. | Reference for reproducing. |
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

Every phase has a re-launchable script:

```bash
# Phase 1 — the original no_encoders ablation (--folds 1 LEAKY, kept for audit)
bash docs/studies/mtl-exploration/run_chain.sh no_encoders alabama arizona florida

# Phase 2 — the linear-encoders factorial (cells B + C, --folds 1 LEAKY, kept for audit)
bash docs/studies/mtl-exploration/run_linear_encoders.sh arizona

# Phase 3 — AZ multi-seed (--folds 5 leak-free; the paper-grade confirmation)
bash docs/studies/mtl-exploration/run_az_multiseed.sh

# Phase 4 — extension: AL multi-seed + cell E thin-hybrid (linear+LN)
bash docs/studies/mtl-exploration/run_ms_chain_v2.sh
```

Set `DATA_ROOT` and `OUTPUT_DIR` env vars if running outside the canonical repo root.

## What's still in-flight (will be committed when done)

`run_ms_chain_v2.sh` is currently executing 16 runs (AL × {D, C, E} × 4 seeds + AZ × E × 4 seeds, 5-fold × 25-ep each). Per-fold log_T at AL is being built for seeds {0,1,7,100} as part of the chain. Expected wall-clock: ~2 h on MPS. Completion flag: `logs/_ms_v2_done.flag`.

When the chain completes:
- Update `EXPERIMENT_NO_ENCODERS.md` with the n=20 AL paired Wilcoxon for cell C vs D.
- Update the same doc with the cell E vs D paired Wilcoxon at AZ (and AL).
- Decide whether the "encoder MLP is over-engineered" claim has cross-state support (AL+AZ) or remains AZ-scoped.

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
