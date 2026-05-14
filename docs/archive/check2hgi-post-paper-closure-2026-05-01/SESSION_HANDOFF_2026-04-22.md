# Session Handoff — 2026-04-22 EOD

**For the next agent that picks up this study.** Start here.

## 30-second summary

- **B5 (faithful hard GETNext) is implemented, trained, and analysed on AL + AZ + FL.**
- **AZ** gets a clean +6.59 pp Acc@10 lift (46.66 → 53.25) outside σ on every region metric — MTL now beats STL STAN on AZ Acc@10 (first time).
- **AL** is a tie within σ (+1.47 pp).
- **FL** is mixed: region mostly wins (Acc@5 +13.5 pp) but cat F1 drops 10.58 pp at 4703-region scale. Likely gradient imbalance.
- **Paper narrative now: scale-dependent champion.** AZ hard headline; AL soft≈hard tie; FL soft retains joint-task quality.
- **Champion rankings post-B5:**
  - AZ: **MTL-GETNext-HARD (B-M9d, 53.25 ± 3.44 Acc@10)** > STL STAN (52.24) > MTL-soft (46.66) > MTLoRA (39.51) > AdaShare (not run AZ)
  - AL: STL STAN (59.20) > **MTL-hard 57.96 ≈ MTL-soft 56.49** (tied within σ) > MTLoRA (53.71) > AdaShare (44.51)
  - FL: soft **MTL B-M13 (60.62 Acc@10, cat F1 66.01)** holds joint-task crown; hard trades −10 pp cat for region lift

## Read this before touching anything

These are the operational gotchas that bit this session and will bite you too:

### G1. `--reg-head` CLI override must resolve before FoldCreator

The check2HGI preset's `task_b.head_factory` starts as `"next_gru"`.
The `resolve_task_set(..., task_b_head_factory=args.reg_head)` call in
`scripts/train.py` applies the CLI override. **We had a latent bug
where this call happened AFTER the FoldCreator was constructed** —
FoldCreator always saw `next_gru` and never activated the B5
aux-dataloader path. Fix landed in commit `ea65fb3` (apply override
once early in addition to the post-fold call).

If you add a new head that needs a different dataloader class:
- Check the early-override block in `scripts/train.py` (labelled
  "Apply --reg-head / --reg-head-param overrides BEFORE fold creation").
- Gate the FoldCreator branch on the exact `head_factory` string.

### G2. `PCGrad` requires every `task_specific_parameters()` to be in the graph

`torch.autograd.grad(losses.sum(), task_specific_parameters)` has
`allow_unused=False`. If any parameter in `task_specific_parameters()`
isn't reached by *any* task's loss, PCGrad crashes with:

> RuntimeError: One of the differentiated Tensors appears to not have
> been used in the graph.

This bit us twice:
- Once on `mtlnet_dselectk` MTLoRA params (legacy task_set code path;
  fixed in `c1c7f3e`).
- Once on `NextHeadGETNextHard.alpha` when its forward fell back to
  `return stan_logits` without using alpha (fixed in `ea65fb3` —
  fallback now adds `alpha * 0.0` to keep it in the graph).

Pattern for new heads: make sure every `nn.Parameter` on the head
is touched on every forward pass. If a branch sometimes skips a
parameter, multiply it by 0 explicitly so autograd still records
the edge.

Alternative-pattern MTL losses (NashMTL, static_weight, db_mtl, etc.)
use scalar `.backward()` and are NOT affected. If smoke-testing a
new head, test with BOTH `--mtl-loss pcgrad` AND `--mtl-loss static_weight`.

### G3. `num_workers=0` is load-bearing for the B5 aux side-channel

`src/data/aux_side_channel.py` uses `threading.local` to pass
`last_region_idx` from the dataloader to the head. This works when
the main thread iterates both the loader and runs forward (the MPS
path, which sets `num_workers=0` in `_get_num_workers()`).

If you change `_get_num_workers()` to return `>0` on MPS, aux will
stop propagating: child workers load batches in separate processes
and the thread-local state doesn't cross process boundaries.

If you need multi-worker dataloading, rewrite the aux pathway to
pass through the batch tuple instead of a thread-local.

### G4. MPS sleep-induced SIGBUS on long runs

Our first partition-bugfix rerun crashed with `exit 138` (SIGBUS)
when the Mac went to sleep mid-training (2h into a 4-6h run). Both
resumes were launched under `caffeinate -s` which prevents sleep.

**Rule:** any training run > 45 min on M4 Pro MPS must be wrapped
in `caffeinate -s`. Use the `scripts/run_b5_hard_mtl.sh` or
`scripts/rerun_partition_bugfix_resume.sh` as templates.

### G5. AZ fold-3 intermittent slowdown (memory pressure)

During the AZ B5 run (2026-04-22 18:43), fold 3 took 27 min vs fold
1's 9 min — a 3× slowdown. `top` showed the training process at
9.5 GB resident with 96 MB free and active swap. Root cause: some
other macOS process (Spotlight, window server) grew temporarily
and pushed the MPS tensors into swap.

**Mitigation if you see similar:** kill Spotlight indexing and window
compositor work (quit non-essential apps) for long AZ / FL runs.
The fold's metrics finish correctly, just slowly.

### G6. `next_region.parquet` schema version

As of commit `6a2f808`, the file now has 4 columns:
`[emb_0..emb_575, region_idx, userid, last_region_idx]`.

Older readers that select by column name (every existing reader in
`src/data/folds.py`) work unchanged. If you write a new reader:
- **DO:** select by column name (`df["region_idx"]`).
- **DO NOT:** iterate `df.columns` expecting a fixed count.

Regenerate with `scripts/regenerate_next_region.py --state X` if
the parquet lacks `last_region_idx` (the B5 head will raise with
a clear error message pointing at that script).

### G7. Old α-inspection checkpoints are NOT the 5-fold champion

The α-inspection runs (`mtl__check2hgi_next_region_20260421_19****`)
are **2-fold × 50 epoch** with `--checkpoints` enabled. The
champion runs (B-M6b, B-M9b, B-M13) were **5-fold × 50 epoch**
with `--no-checkpoints`. If you need to do inference-time analysis
on the champion config, the α-inspection weights are the best
available proxy but are NOT bit-identical.

The epoch-46/47 α-inspection weights *are* trained for the full
schedule (soft-probe has matured) so inference ablations on them
are reasonable; the earlier B5 inference run (`eval_hard_vs_soft_region_idx.py`)
initially used epoch 9 by mistake and reported inflated +20 pp deltas.
Use epoch 46 AL / epoch 47 AZ for correct analysis.

## Current-state reference

### Commits to know

| Commit | What |
|---|---|
| `0a3fecc` | FL B5 hard scaling analysis (latest) |
| `0e35734` | B5 macro analysis vs STL + other MTL families |
| `1873de4` | B5 AZ hard result + paper reframing |
| `bf20807` | B5 AL hard result + launcher |
| `ea65fb3` | B5 bugfix: `--reg-head` ordering + PCGrad-safe fallback |
| `6a2f808` | B5 feat: next_getnext_hard head + aux side-channel |
| `c832eb6` | Post-fix summary + MTL-GETNext reframed as MTL champion |
| `5668856` + `c1c7f3e` | Partition-bug fix (MTL param-partition) |
| `8afc9ac` | Cross-attn partial-forward fix |

### Key docs (read in this order)

1. `SESSION_HANDOFF_2026-04-22.md` (this file).
2. `research/B5_RESULTS.md` — AL + AZ B5 retraining results.
3. `research/B5_MACRO_ANALYSIS.md` — full method-vs-method comparison.
4. `research/B5_FL_SCALING.md` — FL scaling finding + cat-task regression.
5. `research/ATTRIBUTION_PCGRAD_VS_STATIC.md` — PCGrad is not load-bearing.
6. `results/P5_bugfix/SUMMARY.md` — partition-bug fix + champion reframing.
7. `results/RESULTS_TABLE.md` — canonical row-per-method table.
8. `research/BACKLOG_FOLLOWUPS.md` — ordered open items (B12/B13/B14 added).

### Command templates (copy-paste safe)

Launch B5 hard retraining (AL example):

```bash
cd /path/to/check2hgi-mtl
export PY=/path/to/.venv/bin/python
export OUTPUT_DIR=/tmp/check2hgi_data DATA_ROOT=/tmp/check2hgi_data PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1
nohup caffeinate -s "$PY" -u scripts/train.py \
    --task mtl --task-set check2hgi_next_region \
    --state alabama --engine check2hgi \
    --folds 5 --epochs 50 --seed 42 \
    --task-a-input-type checkin --task-b-input-type region \
    --model mtlnet_crossattn --mtl-loss pcgrad \
    --reg-head next_getnext_hard \
    --reg-head-param d_model=256 --reg-head-param num_heads=8 \
    --reg-head-param transition_path=$OUTPUT_DIR/check2hgi/alabama/region_transition_log.pt \
    --max-lr 0.003 --gradient-accumulation-steps 1 --no-checkpoints \
    > /tmp/stan_logs/b5_al.log 2>&1 &
```

Swap `alabama` → `arizona` / `florida` for other states. Regenerate
the parquet first: `"$PY" scripts/regenerate_next_region.py --state <state>`.

### Open paper-blocking follow-ups (priority order)

| # | Task | Effort | Why |
|---|---|---|---|
| B14 | Paired Wilcoxon on AZ B-M9b vs B-M9d | 30 min CPU | Adds `p < 0.05` to the AZ lift claim |
| B13 | FL task-weight rebalancing sweep (1f) | ~2.25h MPS | Can FL-hard be rescued to keep cat F1? |
| B12 | FL 5-fold MTL-GETNext-hard | ~5-6h MPS | Tighten σ on FL; confirm cat regression is real |
| B3 | Multi-seed n=3 on champion configs | ~20h MPS total | Paper requires σ over seeds too |

B14 is the cheapest and highest-ROI next action. Do it first.

### Open research follow-ups (post-paper)

B4 (per-fold transition matrix), B8–B11. See `BACKLOG_FOLLOWUPS.md`.

## Cross-machine state

- **M4 Pro (this machine):** `/tmp/check2hgi_data/` has all three
  states ready, transition matrices built, parquets regenerated with
  `last_region_idx`. All partition-bugfix + B5 runs archived under
  `docs/studies/check2hgi/results/{P5_bugfix,B5}/*.json`.
- **M2 Pro:** last known state was running `probe_a7_optimizers.sh`
  (now complete, archived). Idle.
- **Linux 4050:** idle. Available for B3 multi-seed splits if
  launched manually; B5 handoff doc at `research/B5_HANDOFF.md`
  describes the recipe.

## What a new agent should NOT do

- **Don't re-run partition-bugfix without the fix applied.** The fix
  is already landed in main; verify commit `5668856`+`c1c7f3e` are
  present before launching `scripts/rerun_partition_bugfix*.sh` or
  `scripts/probe_a7_optimizers.sh`.
- **Don't regenerate `next_region.parquet` during a live training
  run.** Sequential re-runs will pick up the new file and may crash
  if the schema changes unexpectedly. Safer: finish all in-flight
  runs first, regenerate, resume.
- **Don't delete the α-inspection checkpoint dirs under
  `results/check2hgi/*/checkpoints/`.** They're the only state dicts
  we have for GETNext/TGSTAN/STA-Hyper and are needed for any
  inference-time ablation.
- **Don't launch MPS training without `caffeinate -s`** if the run
  will exceed ~45 min. See G4.
- **Don't push to `main`.** All work is on `worktree-check2hgi-mtl`.
