# H100 Camera-Ready Gaps Prompt

Paste this to the next H100 agent. This prompt is only for the **remaining
camera-ready compute gaps** after the 2026-05-01 paper-closure run matrix.

---

You are picking up the **Check2HGI MTL study** on branch
`worktree-check2hgi-mtl` in an H100 80 GB environment.

## 1 · Scope

Do **only** these two deferred compute items:

1. **CA + TX MTL multi-seed, B9 vs H3-alt**
2. **AL + AZ + FL STL cat multi-seed**

Do **not** re-run Phase 3, do **not** re-run FL F51 B9 multi-seed, and do
**not** touch already-closed seed=42 anchor runs unless a run is missing or
corrupt.

## 2 · What is already done

These are already closed and should be treated as read-only baselines:

- CA + TX seed=42 MTL anchors for **B9** and **H3-alt**
- CA + TX seed=42 STL ceilings (cat + reg)
- AL + AZ multi-seed **B9** MTL
- AL + AZ multi-seed STL **reg**
- FL multi-seed STL **reg**
- FL multi-seed B9 MTL from F51
- All leak-free paper-closure docs updated through 2026-05-01

The missing pieces are:

- **CA + TX:** add seeds `{0, 1, 7, 100}` for both **B9** and **H3-alt**
- **AL + AZ + FL:** add seeds `{0, 1, 7, 100}` for STL **cat** (`next_gru`)

Seed `42` already exists for these axes and is the baseline anchor. Do not
duplicate it unless the existing run is unusable.

## 3 · Read first

Read these before launching anything:

1. `docs/studies/check2hgi/HANDOVER.md`
2. `docs/studies/check2hgi/SESSION_HANDOFF_2026-05-01.md`
3. `docs/studies/check2hgi/PAPER_CLOSURE_RESULTS_2026-05-01.md`
4. `docs/studies/check2hgi/results/RESULTS_TABLE.md`
5. `docs/studies/check2hgi/research/F50_DELTA_M_FINDINGS_LEAKFREE.md`

For command patterns, inspect:

1. `scripts/run_paper_closure_h100.sh`
2. `scripts/run_paper_closure_h3alt_al_az.sh`

## 4 · Objective

Close the last two camera-ready evidence gaps:

- **Recipe-selection symmetry at large states:** CA/TX currently have only
  seed=42 for B9 vs H3-alt. Add 4 extra seeds so CA/TX no longer rely on
  single-seed directional claims.
- **Cat-side ceiling symmetry:** AL/AZ/FL STL cat currently rely on a single
  seed. Add 4 extra seeds so cat-side error bars are symmetric with the
  multi-seed treatment elsewhere.

## 5 · Canonical constraints

Do not change these:

- model: `mtlnet_crossattn`
- MTL cat head: `next_gru`
- MTL reg head: `next_getnext_hard` / `next_stan_flow`
- STL cat model: `next_gru`
- folds: `5`
- epochs: `50`
- batch size: `2048`
- MTL per-head LR: `cat=1e-3, reg=3e-3, shared=1e-3`
- `--min-best-epoch 5`
- leak-free per-fold transitions for MTL runs
- no changes to study hparams, loss, or data protocol

Use seeds:

```text
0, 1, 7, 100
```

## 6 · Exact tasks

### 6.1 · CA + TX MTL multi-seed, B9 vs H3-alt

Run:

- `state in {california, texas}`
- `seed in {0, 1, 7, 100}`
- `recipe in {B9, H3-alt}`

Total:

- `2 states × 4 seeds × 2 recipes = 16 runs`

Recipes:

- **B9** = H3-alt + `--alternating-optimizer-step` + `--scheduler cosine --max-lr 3e-3` + `--alpha-no-weight-decay`
- **H3-alt** = `--scheduler constant --max-lr 3e-3`

Use leak-free per-seed per-fold transitions:

- `output/check2hgi/<state>/region_transition_log_seed{S}_fold{N}.pt`

If any seeded per-fold transition files are missing for CA or TX, build them
before launching the affected runs.

### 6.2 · AL + AZ + FL STL cat multi-seed

Run:

- `state in {alabama, arizona, florida}`
- `seed in {0, 1, 7, 100}`
- task = STL cat `next_gru`

Total:

- `3 states × 4 seeds = 12 runs`

This axis is cheaper because it does not depend on `log_T`.

## 7 · Command templates

### 7.1 · MTL B9 / H3-alt

Use the existing paper-closure launcher pattern from
`scripts/run_paper_closure_h100.sh`. The per-run form should look like:

```bash
python -u scripts/train.py \
  --task mtl --task-set check2hgi_next_region \
  --state <STATE> --engine check2hgi \
  --model mtlnet_crossattn \
  --cat-head next_gru --reg-head next_getnext_hard \
  --reg-head-param d_model=256 --reg-head-param num_heads=8 \
  --reg-head-param transition_path=$OUTPUT_DIR/check2hgi/<STATE>/region_transition_log.pt \
  --task-a-input-type checkin --task-b-input-type region \
  --folds 5 --epochs 50 --seed <SEED> \
  --batch-size 2048 \
  --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
  --gradient-accumulation-steps 1 \
  --per-fold-transition-dir $OUTPUT_DIR/check2hgi/<STATE> \
  --no-checkpoints --no-folds-cache \
  --min-best-epoch 5 \
  --mtl-loss static_weight --category-weight 0.75 \
  <RECIPE_FLAGS>
```

Where `<RECIPE_FLAGS>` is:

```bash
# B9
--alternating-optimizer-step --scheduler cosine --max-lr 3e-3 --alpha-no-weight-decay

# H3-alt
--scheduler constant --max-lr 3e-3
```

### 7.2 · STL cat `next_gru`

Use:

```bash
python -u scripts/train.py \
  --task next --state <STATE> --engine check2hgi \
  --model next_gru \
  --folds 5 --epochs 50 --seed <SEED> \
  --batch-size 2048 \
  --max-lr 3e-3 \
  --gradient-accumulation-steps 1 \
  --no-checkpoints
```

If there is an existing script for this axis in the workspace, reuse it.
Otherwise create a small shell launcher locally in the pod; do not commit a new
launcher unless it is clean and obviously reusable.

## 8 · Parallelism guidance

H100 80 GB is the target.

Recommended:

- Run **CA/TX MTL** at `MAX_JOBS=2`
- Run **AL/AZ/FL STL cat** at `MAX_JOBS=4`

Expected ETA:

- CA/TX MTL multi-seed: **~3-4 h**
- AL/AZ/FL STL cat multi-seed: **~30 min**
- Total with overhead: **~4-5 h**

## 9 · Output expectations

For the MTL runs, preserve the current run-dir naming conventions so the
analysis scripts can discover them naturally.

For the STL cat runs, the end state should allow refreshed seed-aggregated cat
ceiling summaries for:

- AL
- AZ
- FL

Do not overwrite historical JSONs manually. Generate new artifacts or update
the canonical aggregate files through the existing analysis path.

## 10 · After compute finishes

Do the minimum analysis needed to make the new runs usable:

1. verify all 28 new runs completed
2. extract summary numbers for:
   - CA/TX B9 vs H3-alt, both tasks, pooled across seeds
   - AL/AZ/FL STL cat means and seed σ
3. update the relevant docs if and only if the numbers are clean:
   - `docs/studies/check2hgi/HANDOVER.md`
   - `docs/studies/check2hgi/SESSION_HANDOFF_2026-05-01.md` or a new dated handoff
   - `docs/studies/check2hgi/results/RESULTS_TABLE.md`
4. stage only the new analysis/doc artifacts, never large run dirs

If the analysis refresh becomes larger than expected, stop after producing a
short summary and leave the doc rewrite to a follow-up commit.

## 11 · Hard stops

Stop and report before changing anything if:

- CA/TX seeded per-fold transitions are missing or inconsistent
- any reused baseline run dir is absent or corrupted
- the new CA/TX multi-seed result flips the current directional B9 > H3-alt story
- the FL STL cat multi-seed extension materially changes the cat-side narrative

## 12 · Final deliverable

Return with:

- whether all 28 runs landed
- pooled CA/TX B9 vs H3-alt deltas and p-values if computed
- AL/AZ/FL STL cat multi-seed means and σ
- any doc files updated
- any missing/corrupt prerequisites that blocked completion

Be concise. The goal is to close the last camera-ready compute gaps, not to
re-open the scientific story.
