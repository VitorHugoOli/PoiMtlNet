# Lane 1 — G0.1-advisory + loss-scale advisory: turn-key prep (GPU-queued)

> Created 2026-06-18 (`study/pre-freeze-a40`). The two recipe-touching advisories of
> `HANDOFF_A40_PREFREEZE §1`, fully specced against the current code so launch is one validated pass
> the moment the A40 GPU frees (currently ~40/46 GB held by another user `lucas.lana`). Both levers are
> **windowing-independent** (no sequence/log_T rebuild). **STOP for the user on ≥0.3 pp either head.**

## Why these are PREP, not done

Both touch the **canonical** path (`mtl_cv.py` train loop / `static_weight` loss) and the gate is
recipe-changing, so correctness matters more than speed and an end-to-end GPU validation is mandatory
before trusting a result. The GPU is blocked, so this doc pins the exact design + commands; on GPU-free:
**implement → CPU-smoke the data/loss path → launch → score**.

## The gate's correct comparand — a SAME-CODE A/B (not the pinned R0 json)

`mtl_frontier/FINDINGS.md` is explicit: **do NOT compare current-code numbers to the old
`R0_matched_metric_bar.json`** (June-6 code; champion-G cat drifted ~0.1–0.4 pp between code states).
So the advisory runs **both** arms on TODAY's code at the same seed and compares them head-to-head:

- **Baseline arm** = champion G as-is (random pairing — the two train loaders shuffle independently).
- **Treatment arm** = the lever flag ON.
- **Δ = treatment − baseline**, per head, at **AL + FL seed 0**. ≥0.3 pp either head → multi-seed
  {0,1,7,100} → if it holds, **STOP for user** (recipe → v17 candidate). Null → record EXPLICITLY
  EXCLUDED in the freeze notes (mirror the composite/routing exclusion wording).

Champion-G **deterministic** (FINDINGS audit: bit-identical re-runs), so a single baseline run per state
suffices; seed-0 is the *weakest* G seed, so a real lever shows here first.

Reference magnitudes (R0 bar, OLD code — orientation only, NOT the gate): AL cat 52.91 / reg-full 62.57;
FL cat 73.16 / reg-full 72.97; GE cat 61.43 / reg-full 58.35. Champion-G FL cat per seed (current code,
deterministic) = [73.012, 73.212, 73.181, 73.143].

## Champion-G baseline invocation (the as-is arm)

`--canon v16` bundles the full G recipe (v14 engine, dualtower, onecycle, unweighted CE, KD off; omits
`--checkpoint-selector` → `geom_simple` default). Equivalent explicit form (CHAMPION.md §3):

```bash
python scripts/train.py --task mtl --task-set check2hgi_next_region \
  --engine check2hgi_design_k_resln_mae_l0_1 --state florida --seed 0 \
  --epochs 50 --folds 5 --batch-size 2048 \
  --model mtlnet_crossattn_dualtower \
  --mtl-loss static_weight --category-weight 0.75 \
  --cat-head next_gru --reg-head next_stan_flow_dualtower \
  --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
  --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
  --task-a-input-type checkin --task-b-input-type region \
  --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
  --log-t-kd-weight 0.0 \
  --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/florida \
  --no-checkpoints
# AL: swap --state alabama + --per-fold-transition-dir .../alabama. Both have fresh v14 + log_T (verified).
```
Run the C28 hygiene: PID-suffixed rundir, per-run seed echo, freshness preflight (now centralized —
`src/data/log_t_freshness.py`; both states verified fresh this session).

## Lever A — G0.1 aligned-pairing (the lone recipe-CHANGING gate)

**What today does (random pairing):** the MTL cat train loader (`x_task_a`) and reg train loader
(`x_task_b`) are two `_create_dataloader(..., shuffle=True, seed)` instances (`folds.py:1054-1090`).
Two DataLoaders with the same seed still shuffle **independently** → in each mixed step cat-window *k*
is paired with reg-window *m≠k*. **Val is already aligned** (shuffle=False). The X1 roll probe (Δ−0.004)
proved the *deployed* model is pairing-invariant but is **circular** vs "mixing is learnable under
aligned training" — hence this gate.

**Precondition CONFIRMED this session:** `x_task_a[i]` and `x_task_b[i]` are the **same window i** —
both `_resolve_x(...)` off the same row-aligned `next.parquet` (`folds.py:899-1016`). So aligned pairing
= iterate both train loaders under ONE shared permutation; no re-indexing of labels needed.

**Design (flag `--aligned-pairing`, default off; MTL check2hgi path only):**
1. In `_create_check2hgi_mtl_folds`, when the flag is set, build the **cat + reg train tensors into one
   `TensorDataset`** (`x_task_a[train], y_cat[train], x_task_b[train], y_region[train]` [+ `aux[train]`])
   and a single `DataLoader(shuffle=True, generator=g(seed))`. Emit it on `FoldResult` as a new
   `joint_train_loader` (leave the existing per-task loaders for the val path + non-flag default).
2. In `mtl_cv.py`'s mixed-batch loop (the `zip_longest_cycle` over the two train loaders,
   ~`mtl_cv.py:463`), when a `joint_train_loader` is present, iterate it instead — each batch already
   carries the aligned (cat, reg[, aux]) tuple, so cat-row *k* trains paired with reg-row *k*.
3. **CPU smoke (no GPU):** assert that for a tiny synthetic fold the joint batch's cat and reg rows map
   to the same underlying sequence index (compare against `x_task_a`/`x_task_b` gather by the sampler's
   permutation). This is the highest-risk correctness point — verify before the real run.

**Alternative (lighter, if the joint-dataset refactor proves invasive):** drive both existing train
loaders with a shared `torch.Generator` *and* a shared `RandomSampler` index list regenerated identically
per epoch. Riskier (two `iter()` calls must consume the same permutation in lockstep across workers) —
prefer the joint-dataset path; keep this only as fallback.

**Caveat to carry:** aligned pairing interacts with stride-1's denser supervision, so this **current-base**
advisory may not transfer to the frozen (possibly-overlap) base — the **binding** G0.1 re-runs on the
frozen base, full {0,1,7,100}, and is the only run that can re-pin v16→v17 (EXECUTION_PLAN §3).

## Lever B — loss-scale normalization (the "left on the table" advisory, user-approved RUN)

**Mechanism:** the two raw CEs enter `StaticWeightLoss.get_weighted_loss` as `losses=[reg_ce, cat_ce]`
and are combined `0.25·reg_ce + 0.75·cat_ce` (`src/losses/static_weight/loss.py:34-40`). But cat is
7-class (CE scale ~ln 7 ≈ 1.95) and reg is ~1.1k–4.7k-class (CE scale ~ln n_reg ≈ 7.0 [AL] … 8.5 [FL]) —
a ~3.6–4.3× built-in magnitude gap that `cw=0.75` may be partly/accidentally undoing.

**Design (flag `--loss-scale-norm`, default off):** divide each task CE by `log(num_classes)` **before**
the static weight: `reg_ce/ln(n_reg)`, `cat_ce/ln(7)`, then `0.25·· + 0.75··`. Pass the per-task class
counts into the loss (the runner already knows `n_regions` and the 7-cat constant). Mechanistically distinct
from every closed R-gate (a *magnitude* fix → survives the P4 gradient-conflict null), so it is genuinely
untested-and-unexcluded.
- **CPU unit test (no GPU):** feed dummy logits/targets at 7 and ~1000 classes; assert the normalized
  combined loss equals `0.25·reg_ce/ln(n_reg) + 0.75·cat_ce/ln(7)` and that flag-off is bit-identical to
  the current loss.

**Gate:** ≥0.3 pp either head (AL+FL seed0, same-code A/B) → v17 candidate → STOP for user; null → record
EXPLICITLY EXCLUDED in the freeze notes.

## Run plan when GPU frees (≥30 GB; the background waiter fires)

1. Implement Lever B (loss path) + its CPU unit test; implement Lever A (joint loader) + its CPU smoke.
2. Baseline G: AL seed0, FL seed0 (deterministic → 1 run each). ~FL/AL fit the A40 at bs2048 (memory
   note: FL/AL/AZ/GE fit; CA/TX do not — not needed here).
3. Lever A arm: AL seed0, FL seed0. Lever B arm: AL seed0, FL seed0. (4 treatment runs + 2 baselines.)
4. Score per-task **diagnostic-best** (`fold_info.json`, NOT joint `full_summary.json` — ref_mtl_metric_field),
   matched metric/forward. Compute Δ = treatment − baseline per head.
5. Any Δ ≥ 0.3 pp → multi-seed {1,7,100} that arm → if it holds, **STOP, hand to user** (v17 candidate).
   All null → write the EXCLUDED records into the freeze notes + the gate ledger rows (G0.1, loss-scale).

## Ledger rows this lane closes (PRE_FREEZE_PROGRAM gate ledger)

- **G0.1 aligned-pairing** — advisory now; binding run on the frozen base re-pins recipe if it fires.
- **Loss-scale normalization** (EXECUTION_PLAN §8 #5 / Decisions #5 RUN) — advisory; null → on-the-record exclude.
