# Phase 3 Handoff Prompt — paste this to your next Lightning.ai agent

You are picking up the **Check2HGI substrate-comparison study** in a fresh Lightning.ai pod with A100 (40 GB). The user has Drive but **the pod cannot mount it**; data comes via gdown links the user provides.

You inherit a clean branch state on `worktree-check2hgi-mtl`. **Do not push to `main`.** Stay on the worktree branch throughout.

## 1 · What's done (DO NOT re-run)

Phase 2 closed STL at all 5 states (AL+AZ+FL+CA+TX):
- **CH16 confirmed** — cat substrate gap (matched-head `next_gru`) significant at max-n=5 paired Wilcoxon p=0.0312, 5/5 folds positive at every state.
- **CH15 reframed** — substrate-equivalence on reg under matched MTL head, TOST δ=2pp non-inferior at all 5 states.
- All per-fold + paired-test JSONs live in git under `docs/studies/check2hgi/results/`.

Cat STL data (used in CH16) is leakage-free already (no transition matrix). Substrate linear probe is leakage-free (head-free).

## 2 · What you need to do — Phase 3 Scope D

Re-run **all reg STL + MTL B3 cells with leakage-free per-fold transition matrices** so the entire reg/MTL evidence chain becomes leakage-free, in one consistent protocol across all 5 states. Resolves AUDIT-C4 / F44.

| Cells to run | Count | A100 40 GB ETA (4× parallel) |
|---|---|---|
| Per-fold transition build (CPU) | 25 matrices (5 states × 5 folds) | ~25 min |
| Reg STL `next_getnext_hard` × 5 states × 2 engines | 10 cells | ~50 min |
| MTL B3 cross-attn × 5 states × 2 engines | 10 cells | ~110 min |
| Finalize (extract + paired tests, CPU) | — | ~10 min |

**Total ~2.7 h on 4× A100 (40 GB)**, ~$13. T4 will OOM, do not use.

Existing Phase 2 leaky data is preserved as a historical reference; Phase 3 outputs use a **`_pf` suffix** to coexist (don't overwrite).

## 3 · First read (in this order)

1. `docs/studies/check2hgi/PHASE3_TRACKER.md` — work plan, status board, GPU strategy, acceptance criteria
2. `docs/studies/check2hgi/PHASE3_LIGHTNING_HANDOFF.md` — concrete pod-setup walkthrough
3. `docs/studies/check2hgi/PHASE2_TRACKER.md` §0–§2 — context on what's already closed
4. `CLAUDE.md` (repo root) — project structure
5. **Skim** `docs/studies/check2hgi/research/SUBSTRATE_COMPARISON_FINDINGS.md` — current paper-grade verdicts

Don't read the F50/F49 architectural-mechanism docs unless something forces you there. They're a parallel track.

## 4 · Concrete steps

### 4.1 · Bootstrap the pod

Ask the user for the AL/AZ/FL gdrive folder IDs if you don't already have them (CA + TX are hardcoded as defaults). Then:

```bash
cd /teamspace/studios/this_studio
git clone https://github.com/VitorHugoOli/PoiMtlNet.git
cd PoiMtlNet
git checkout worktree-check2hgi-mtl

STATES="alabama arizona florida california texas" \
  AL_C2HGI_GDID=<id> AL_HGI_GDID=<id> \
  AZ_C2HGI_GDID=<id> AZ_HGI_GDID=<id> \
  FL_C2HGI_GDID=<id> FL_HGI_GDID=<id> \
  bash scripts/setup_lightning_pod.sh
```

Defaults if user doesn't have AL/AZ/FL ready: drop those env vars, fall back to `STATES="california texas"`. Document which states you ran.

Known gdrive IDs already wired in:

| folder | ID |
|---|---|
| `output/check2hgi/california` | `1ZLL8FHPeO7I-3DEfVBogW1C1eFE76ttv` |
| `output/check2hgi/texas` | `1bLfFDEOM1BJ2ELoQUnd_qMXFpxGsZ7UF` |
| `output/hgi/california` | `1nMNaFgEEc1RwoH_o8_wasL9ENOkOCdKJ` |
| `output/hgi/texas` | `1g43xNSlJZBXStt3YGruOvCZTI-_WW4OQ` |

### 4.2 · Smoke test before launching the grid

```bash
PYTHONPATH=src OUTPUT_DIR=$(pwd)/output \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u scripts/train.py \
  --task mtl --state california --engine check2hgi \
  --task-set check2hgi_next_region --model mtlnet_crossattn \
  --mtl-loss static_weight --category-weight 0.75 \
  --reg-head next_getnext_hard --cat-head next_gru \
  --reg-head-param d_model=256 --reg-head-param num_heads=8 \
  --reg-head-param transition_path=$(pwd)/output/check2hgi/california/region_transition_log.pt \
  --batch-size 2048 --folds 1 --epochs 2 --seed 42 --no-checkpoints
```

If it runs to fold 1 epoch 2 cleanly → pod is ready. If OOM despite A100 → check `nvidia-smi --query-gpu=memory.total` (need ≥40 GB).

### 4.3 · Run the grid

```bash
# Step 1 — per-fold transitions (CPU, ~25 min total)
bash scripts/build_phase3_per_fold_transitions.sh

# Step 2 — parallel grid (~2.7 h on 4× A100)
nohup bash scripts/run_phase3_parallel.sh > logs/phase3/orchestrator.log 2>&1 &
echo $! > logs/phase3/orchestrator.pid

# Monitor
tail -f logs/phase3/orchestrator.log
```

The orchestrator auto-detects GPU count: ≥4 GPUs → 4-way parallel; 2-3 GPUs → 2-way parallel waves; 1 GPU → falls back to sequential `scripts/run_phase3_grid.sh`.

If anything fails mid-grid, fix root cause (don't bypass safeguards) and re-launch — the orchestrator skips already-built transition matrices and the per-cell scripts overwrite their tagged run dirs cleanly.

### 4.4 · Finalize

```bash
python3 scripts/finalize_phase3.py
```

Produces:
- `docs/studies/check2hgi/results/phase1_perfold/<STATE>_<engine>_{reg_gethard_pf_5f50ep, mtl_{cat,reg}_pf}.json`
- `docs/studies/check2hgi/results/paired_tests/<state>_{reg_acc10, reg_mrr, mtl_cat_f1, mtl_reg_acc10, mtl_reg_mrr}_pf.json`
- Cross-state CH15 + CH18 status board (printed)

### 4.5 · Document + commit + push

After the grid finishes, update:

1. **`docs/studies/check2hgi/PHASE3_TRACKER.md`** — flip 🔴 → 🟢 in the §Status board for each cell that landed.
2. **`docs/studies/check2hgi/research/SUBSTRATE_COMPARISON_FINDINGS.md`** — append a new section "Phase 3 — Scope D leakage-free CH15+CH18 closure" with cross-state tables (mirror the structure of the Phase 2 sections you'll find at the top of that doc).
3. **`docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md`** — bump CH15 + CH18 status to `confirmed leakage-free at AL+AZ+FL+CA+TX` (or the subset you ran).
4. **`docs/studies/check2hgi/FOLLOWUPS_TRACKER.md`** — close F44.

Then:

```bash
git add docs/studies/check2hgi/results/{phase1_perfold,paired_tests,P1}/*_pf*.json \
        docs/studies/check2hgi/results/paired_tests/*_pf.json \
        docs/studies/check2hgi/{PHASE3_TRACKER,research/SUBSTRATE_COMPARISON_FINDINGS,CLAIMS_AND_HYPOTHESES,FOLLOWUPS_TRACKER}.md
git commit -m "study(check2hgi): Phase 3 Scope D — leakage-free reg STL + MTL CH18 closed at <N>/5 states"
git pull --rebase origin worktree-check2hgi-mtl   # remote may have moved
git push origin worktree-check2hgi-mtl
```

### 4.6 · Drive bundle for gitignored artefacts

Run dirs (`results/{c2hgi,hgi}/<state>/mtlnet_*_pf*`) and Lightning logs are gitignored. Bundle them and have the user upload to Drive:

```bash
DATE=$(date +%Y-%m-%d)
BUNDLE=/teamspace/studios/this_studio/phase3_drive_bundle_$DATE
mkdir -p $BUNDLE/{results,logs,output_per_fold_transitions}
for state in alabama arizona florida california texas; do
  for engine in check2hgi hgi; do
    cp -r results/$engine/$state/mtlnet_* $BUNDLE/results/ 2>/dev/null || true
  done
  if [ -f "output/check2hgi/$state/region_transition_log_fold1.pt" ]; then
    mkdir -p $BUNDLE/output_per_fold_transitions/$state
    cp output/check2hgi/$state/region_transition_log_fold*.pt $BUNDLE/output_per_fold_transitions/$state/
  fi
done
cp -r logs/phase3 $BUNDLE/logs/
cd /teamspace/studios/this_studio
tar czf phase3_drive_bundle_$DATE.tar.gz $(basename $BUNDLE)/
ls -lh phase3_drive_bundle_$DATE.tar.gz
```

Tell the user the path; they'll download via Lightning Files panel and upload to Drive `mestrado_data/PoiMtlNet/phase3_archives/`.

## 5 · Hard constraints — do not violate

- **Don't change canonical hparams.** Pinned: `lr=1e-4, bs=2048, NashMTL static_weight α_cat=0.75, model=mtlnet_crossattn, cat_head=next_gru, reg_head=next_getnext_hard, d_model=256, num_heads=8, 5-fold StratifiedGroupKFold(userids, seed=42), 50 epochs`. Phase 2 used the same — only the transition matrix changes.
- **Don't push to `main`.**
- **Don't run on T4** — proven OOM at every batch size for CA-scale MTL on 15 GB VRAM.
- **Don't overwrite Phase 2 leaky data** — Phase 3 uses `_pf` suffix; both must coexist in repo.
- **Don't skip the smoke test** before launching the full grid — the up-front 5-min validation saves hours of misallocated GPU time.
- **Don't `git add -A`** — stage explicitly to avoid committing temp files / large run dirs (those are gitignored anyway, but the habit prevents accidents).
- **Don't bypass pre-commit hooks** with `--no-verify`. If a hook fails, fix it.

## 6 · Useful preferences from prior sessions

- Per-fold extraction key: `diagnostic_best_epochs.next_category.metrics.{f1,accuracy}` and `diagnostic_best_epochs.next_region.metrics.{top10_acc_indist,mrr_indist,f1}` (the storage layer writes 1-indexed `fold{i}_info.json` files; map to 0-indexed `fold_<i>` keys in extracted JSONs to match the rest of the repo).
- Cross-validate Lightning vs Drive when possible: TX cells matched within <0.15 pp. Same expectation here — if Phase 3 numbers diverge by >1 pp from Phase 2 leaky baseline, investigate (probably a fold-split or transition-matrix mismatch).
- For tight memory: `MTL_BATCH_SIZE=1024 bash scripts/run_phase3_mtl_cell.sh ...`. Wall-clock unchanged (samples/sec ~constant). Only use if A100 OOMs unexpectedly.
- The orchestrators set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically — recovers ~300 MB of fragmented cache.

## 7 · Acceptance criteria for closure

Per state, all of:
- Both reg STL cells (`_pf` tag) finish all 5 folds.
- Both MTL cells (`_pf` tag) finish all 5 folds.
- `<state>_reg_acc10_pf.json` shows TOST non-inferiority at α=0.05.
- `<state>_mtl_cat_f1_pf.json` shows Wilcoxon p<0.05 (C2HGI > HGI on cat F1).
- `<state>_mtl_reg_acc10_pf.json` shows Wilcoxon p<0.05 (C2HGI > HGI on reg Acc@10).

Phase 3 closes when ≥4 of 5 states meet all acceptance bullets and CH18 is confirmed leakage-free at ≥4 of 5 states (matches Phase 2's "≥2 of 5" bar but raised because we now have full-grid data).

## 8 · If something breaks

- **Pod runs out of disk** during gdown — `df -h /teamspace/studios/this_studio/`; need ≥40 GB free.
- **GPU disappears** (Lightning preempts) — relaunch is safe; the orchestrator's per-cell scripts are independent. Already-completed cells leave their tagged run dirs untouched; failed cells overwrite cleanly on retry.
- **Smoke test OOMs on A100** — confirm you're on 40 GB+ A100, not 16 GB V100 or T4. `nvidia-smi --query-gpu=memory.total --format=csv`.
- **Per-fold transition build fails** — check `output/check2hgi/<state>/temp/sequences_next.parquet` and `output/check2hgi/<state>/temp/checkin_graph.pt` exist (gdown should have downloaded these as part of the upstream c2hgi folder).
- **Paired test "no per-fold X samples extracted"** — the JSON layout mismatched the analyzer's expected keys; my finalize script writes `fold_<i>` 0-indexed; if a hand-edited JSON drifted, regenerate via `python3 scripts/finalize_phase3.py`.

## 9 · Communication style preferences

- Don't narrate ToolSearch results or environment exploration verbosely. Brief status updates only at meaningful milestones (cell completion, OOM, paired-test results).
- After the grid finishes, post a single concise summary table of all 5 states × all paired tests instead of dumping JSONs.
- If you encounter an unexpected scientific result (e.g., CH18 fails at one state), flag it loudly and stop — the user wants to discuss before doc updates.

## 10 · The user

Vitor (vho2009@hotmail.com). Working from a Lightning.ai Studio. Brazilian time zone, replies primarily in English with occasional Portuguese phrases. Auto mode is enabled — proceed without asking for routine decisions, but pause on anything destructive or scientifically surprising.

Good luck. Branch `worktree-check2hgi-mtl`. PHASE3 docs are your map. Don't reinvent.
