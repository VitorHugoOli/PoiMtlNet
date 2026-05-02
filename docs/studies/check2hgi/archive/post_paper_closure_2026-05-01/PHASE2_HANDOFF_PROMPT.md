# Phase 2 hand-off prompt — finish CA + TX

Paste this into a fresh agent session on this branch (`worktree-check2hgi-mtl`).

---

You are continuing Phase 2 of the check2hgi substrate-comparison study. Read first, then act:

1. `docs/studies/check2hgi/PHASE2_TRACKER.md` — full state board, including §TX block (per-fold reg-hgi metrics captured from log) and §7 (handoff notes).
2. `docs/studies/check2hgi/research/SUBSTRATE_COMPARISON_FINDINGS.md` — Phase 1 verdict + cross-state synthesis so far.
3. `docs/studies/check2hgi/research/SUBSTRATE_COMPARISON_PLAN.md` — 3-leg framework + acceptance criteria.

## What's done (don't re-run)

- AL/AZ Phase 1 closed.
- FL Phase 2 closed (T1).
- CA STL closed: probe ✅, cat STL ×2 ✅ (Wilcoxon p=0.0312), reg STL ×2 ✅ (TOST non-inf).
- TX probes ✅, cat STL ×2 ✅ (Δ=+28.26 pp).
- TX reg STL c2hgi ✅ (`results/P1/region_head_texas_*_check2hgi_reg_gethard_5f50ep.json` on Drive at `/content/drive/MyDrive/mestrado_data/PoiMtlNet/results/P1/`).
- TX reg STL hgi: 4/5 folds completed, fold 5 + aggregate JSON lost when the Colab session died from credit exhaustion at ~14:14 UTC on 2026-04-29. Per-fold metrics for folds 0–3 are recorded verbatim in PHASE2_TRACKER §TX.

## What you must finish — in this order

### Block 1 — TX reg STL hgi fold 5 (~15 min on T4 if checkpoint survived; ~75 min from scratch)

Reconnect to a GPU Colab session with credits. Follow the daemon-launch pattern in `notebooks/colab_check2hgi_mtl.ipynb` (see existing cells `5SrowOATEX0r`, `bhehLuXUC-_-`, `ThM0GK5vET15`):

1. Mount Drive, clone `worktree-check2hgi-mtl`, install deps + PyG wheels matching torch version.
2. **Drive paths in this Colab account use `mestrado_data/PoiMtlNet/`**, not `mestrado/PoiMtlNet/`. Both directories exist; only the first has data.
3. Copy TX parquets (~7.4 GB) from `mestrado_data/PoiMtlNet/output/{check2hgi,hgi}/texas/` to `/content/output/`.
4. Pull the checkpoint file `results/P1/region_head_texas_region_5f_50ep_STL_TEXAS_hgi_reg_gethard_5f50ep.checkpoint.json` from Drive into the repo's `docs/studies/check2hgi/results/P1/` directory if it survived rsync. If it did, the run will auto-resume from fold 4. If not, just re-run from fold 0.
5. Detached daemon (`nohup setsid bash …`):
   ```bash
   python3 -u scripts/p1_region_head_ablation.py --state texas \
     --heads next_getnext_hard --folds 5 --epochs 50 --seed 42 \
     --input-type region --region-emb-source hgi \
     --override-hparams d_model=256 num_heads=8 \
       transition_path=/content/output/check2hgi/texas/region_transition_log.pt \
     --tag STL_TEXAS_hgi_reg_gethard_5f50ep
   ```
6. After exit rc=0, rsync `results/P1/` and the run dirs back to Drive.

### Block 2 — Harvest TX results to repo + paired tests (~5 min, CPU)

From a Colab cell, base64-stream these from Drive:

- `results/probe/texas_check2hgi_last.json` and `texas_hgi_last.json`
- `results/check2hgi/texas/next_lr1.0e-04_bs1024_ep50_20260429_0458/folds/fold{1..5}_info.json`
- `results/hgi/texas/next_lr1.0e-04_bs1024_ep50_20260429_0528/folds/fold{1..5}_info.json`
- `results/P1/region_head_texas_*_STL_TEXAS_check2hgi_reg_gethard_5f50ep.json`
- `results/P1/region_head_texas_*_STL_TEXAS_hgi_reg_gethard_5f50ep.json` (after Block 1)

Decode locally and write to:

- `docs/studies/check2hgi/results/probe/texas_{check2hgi,hgi}.json`
- `docs/studies/check2hgi/results/phase1_perfold/TX_{check2hgi,hgi}_cat_gru_5f50ep.json` — extract per-fold from `diagnostic_best_epochs.next.metrics` (keep `f1`, `accuracy`)
- `docs/studies/check2hgi/results/P1/region_head_texas_*.json` — verbatim
- `docs/studies/check2hgi/results/phase1_perfold/TX_{check2hgi,hgi}_reg_gethard_5f50ep.json` — extract per-fold; rename `top10_acc` → `acc10`, keep `f1`, `acc1`, `acc5`, `acc10`, `mrr`

Run paired tests via `scripts/analysis/substrate_paired_test.py`:

- cat F1: `--check2hgi TX_check2hgi_cat_gru_5f50ep.json --hgi TX_hgi_cat_gru_5f50ep.json --metric f1 --task cat --state texas`
- reg Acc@10: `--check2hgi TX_check2hgi_reg_gethard_5f50ep.json --hgi TX_hgi_reg_gethard_5f50ep.json --metric acc10 --task reg --state texas`
- reg MRR: same with `--metric mrr`
- Output JSONs go to `docs/studies/check2hgi/results/paired_tests/`

Then update `PHASE2_TRACKER.md` TX row STL portion 🟡→🟢 with paired Δ + p-values, and append a TX section to `research/SUBSTRATE_COMPARISON_FINDINGS.md` (STL only — CH18 MTL still pending).

Commit + push to `worktree-check2hgi-mtl`.

### Block 3 — CA + TX MTL CH18 (BLOCKED — needs user decision)

Both `MTL+HGI` and `MTL+C2HGI` 5-fold runs SIGKILL at fold 1 epoch 1 on every device tried so far:

- Colab T4: cgroup OOM rc=137 at ~12 GB.
- M4 Pro 64 GB: macOS jetsam silent SIGKILL even with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`.

Root cause (advisor diagnosis): `src/data/folds.py::_create_check2hgi_mtl_folds` materialises **all 5 folds × full-N tensors × 2 task copies** in memory simultaneously. ~28 GiB of "other allocations" observed in the MPS error. Same blocker for CA and TX.

**Do NOT relaunch MTL on M4 or Colab T4 without first asking the user to pick one of:**

1. **Loosen CH18 acceptance to ≥ 2 of {AL, AZ, FL, CA, TX}** — already met via AL ✓ AZ ✓ FL ✓. Document CA/TX MTL as paper-methodology footnote ("MTL CH18 confirmed at smaller-scale states; large-state MTL deferred due to memory constraints in fold-store implementation"). 0 min effort.
2. **Patch `src/data/folds.py` for lazy fold loading** — refactor `_create_check2hgi_mtl_folds` to stream from disk per fold instead of holding all 5 in memory. Expected RSS drop from ~28 GiB to ~6 GiB. Then re-run MTL on M4 for both substrates × 2 states. ~2 days work.
3. **Colab Pro High-RAM** — 50 GB cgroup; smoke peak was 47 GiB so this fits. Run MTL+C2HGI + MTL+HGI for CA + TX (4 runs × ~50 min on T4 with High-RAM). $10/mo + ~3 h compute.

If user picks **C2** or **C3**, the MTL B3 north-star CLI (per `NORTH_STAR.md`):

```bash
python3 -u scripts/train.py --task mtl --state {california,texas} --engine {check2hgi,hgi} \
  --task-set check2hgi_next_region --model mtlnet_crossattn \
  --mtl-loss static_weight --category-weight 0.75 \
  --reg-head next_getnext_hard --cat-head next_gru \
  --reg-head-param d_model=256 --reg-head-param num_heads=8 \
  --reg-head-param transition_path=$OUTPUT_DIR/check2hgi/{state}/region_transition_log.pt \
  --folds 5 --epochs 50 --seed 42 --no-checkpoints
```

After both engines × both states land:

- Extract per-fold from `results/<engine>/<state>/mtlnet_*/folds/foldN_info.json::diagnostic_best_epochs.next_{category,region}.metrics`
- Write `docs/studies/check2hgi/results/phase1_perfold/{CA,TX}_{check2hgi,hgi}_mtl_{cat,reg}.json`
- Cat: keep `f1`, `accuracy`. Reg: keep `f1`, `acc1=top1_acc_indist`, `acc5=top5_acc_indist`, `acc10=top10_acc_indist`, `mrr=mrr_indist`
- Run paired tests (cat F1, reg Acc@10) per state via `scripts/analysis/substrate_paired_test.py`
- Update PHASE2_TRACKER MTL rows to ✅ with paired Δ + p-values
- Append confirmed CH18 to `research/SUBSTRATE_COMPARISON_FINDINGS.md`
- Mark Phase 2 closed; study moves to write-up

## Failure modes already mapped (don't re-discover)

- **macOS jetsam** silent SIGKILL on RSS pressure; no traceback, just `resource_tracker: leaked semaphore`. Not solvable with `PYTORCH_MPS_HIGH_WATERMARK_RATIO`.
- **Colab T4 cgroup OOM** rc=137 at ~12 GB.
- **MPSGraph INT_MAX dim limit** at CA's 286K × 8497 ≈ 2.4B in `_rank_of_target` — already patched in `src/tracking/metrics.py` (commit `3769203`) by falling back to CPU on MPS for the chunked comparison.
- **Colab runtime resets** wipe `/content/*` (incl. PID file, parquets, local results not yet rsync'd). Detached daemons survive MCP cell timeouts but NOT runtime resets. Always rsync to Drive between experiments.
- **Colab credit exhaustion** kills the session unannounced; the daemon you launched is gone with the runtime. Plan around this by running the most expensive remaining experiment first.
- **Local Drive sync** (`~/Library/CloudStorage/GoogleDrive-…/My Drive`) does NOT contain the Colab path. The `/content/drive/MyDrive/mestrado_data/PoiMtlNet` tree is only reachable from a live Colab session. Mirror via base64 streaming.
- **This Colab account uses `mestrado_data/PoiMtlNet`**, not `mestrado/PoiMtlNet`. Earlier conversation logs reference the wrong path; the tracker is correct.

## Constraints

- Don't push to `main`. Push to `worktree-check2hgi-mtl`.
- Don't rebase published commits.
- For UI/CLI changes, type-check + run targeted unit tests; not needed for tracker doc edits.
- Capture per-fold metrics from logs as you go, into the tracker §TX block — so they survive even if the next Colab dies.
