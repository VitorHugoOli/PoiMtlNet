# docs/infra/h100/ — H100 SSH bare-metal

Dedicated H100 box (80 GB, SSH access). Used for paper-closure final runs and any compute that exceeds Lightning's per-pod limits or where you want a long-lived dedicated environment.

## When to use H100 over Lightning

- **Use H100** when: you need >40 GB GPU memory in a single device (e.g., FL multi-state batch sweeps), the workload is days-long and you want one stable env, you've been handed a specific H100 box for a specific deliverable.
- **Use Lightning** when: ad-hoc bursts, multi-pod parallelism, you want the start-and-stop billing model.
- **Use RunPod** when: you want CUDA + SSH workflow but A100/H100 isn't required (RTX 4090 is faster than T4, cheaper than A100).

## Quick start

Assumes someone gave you SSH credentials to an H100 box already provisioned with CUDA + Python.

```bash
# 1. Connect + clone
ssh user@h100-box
git clone <repo-url> PoiMtlNet
cd PoiMtlNet
git checkout main                              # or worktree-check2hgi-mtl pre-merge

# 2. Bootstrap env (use the runpod_setup.sh — it's CUDA-pinned and works on bare H100 too)
bash scripts/runpod_setup.sh
source .venv/bin/activate

# 3. Fetch data via gdown (no Drive mount on bare-metal SSH)
python scripts/phase3_download_drive.py --state florida

# 4. Launch in tmux (mandatory — SSH disconnects kill foreground processes)
tmux new -s mtl 'bash scripts/run_h100_camera_ready_gaps.sh 2>&1 | tee logs/h100_$(date +%s).log'
# Detach: Ctrl-b d
```

## H100-specific scripts

| Script | Purpose |
|---|---|
| `scripts/run_h100_camera_ready_gaps.sh` | Camera-ready gaps closure (the multi-state finals). |
| `scripts/run_h100_arch_delta_stl_ca_tx.sh` | Architecture-delta STL on CA + TX. |
| `scripts/run_h100_tx_2way.sh` | TX-only 2-way comparison. |
| `scripts/run_h100_fl_mtl_b9_multiseed.sh` | FL MTL B9 multi-seed (the main paper-table cell at FL). |
| `scripts/run_paper_closure_h100.sh` | Generic paper-closure launcher. |

## Conventions

- Always run inside `tmux` (or `screen`). SSH-detach loses foreground work.
- Pipe outputs to `logs/h100_<timestamp>.log` so a reconnect can `tail -f` to check progress.
- Sync results back via `scp` or `rsync` after each run; don't accumulate untouched run dirs on the H100 disk.
- Use `nvidia-smi --gpu-reset -i 0` between large runs if you see between-fold CUDA fragmentation (rare on H100 but possible).

## Wall-clock reference

The B9 MTL FL 5f×50ep multi-seed (n=20 seeds) on a single H100:
- ~10 min per seed × 20 seeds = ~3.5 h wall-clock
- ~$10 at $3/h (typical bare-metal rate)

Compare to ~10 h on a Lightning A100 single-GPU and infeasible on M4 Pro MPS.

## Historical context

The `h100/pervisit-fl-ca-tx-results` branch (commit `a858177`, 2026-05-04) carried the FL+CA+TX per-visit counterfactual results. Those results were integrated into `worktree-check2hgi-mtl` via commit `4b20085 D2 closed`. Per the 2026-05-14 reorg verification, all 9 per-cell JSONs are content-identical (sha256-verified) on the integrated branch — the h100 branch can be archived (tagged + deleted).

The original integration prompt for that work is preserved as an appendix below for reference; the work itself is **closed**.

---

## Appendix: Historical H100 FL+CA+TX per-visit integration prompt (CLOSED 2026-05-04)

This is the one-time integration prompt that brought the per-visit results into the main branch. Kept here as the canonical example of an H100 → main result-integration handoff. Do NOT execute again — the work is done.

### Original prompt content

The compute is **already done**; this prompt covered the integration step: pulling the FL+CA+TX result artifacts from the compute branch into `worktree-check2hgi-mtl` and closing the D2 deferred item.

#### What was done

The FL+CA+TX per-visit counterfactual (CH19) ran successfully on branch `h100/pervisit-fl-ca-tx-results` (commit `a858177`, 2026-05-04). Nine cells (3 states × {canonical, POI-pooled, HGI}), matched-head `next_gru` STL, seed=42, 5f × 50ep, bs=1024, H100 80 GB. Per-fold JSONs, a 5-state summary doc, and an updated `per-visit.png` were committed to that branch.

**Results (5-state confirmed CH19):**

| State | C2HGI | Pooled | HGI | total gap | per-visit pp | training-sig pp | per-visit % |
|---|---:|---:|---:|---:|---:|---:|---:|
| AL | 40.76 ± 1.50 | 29.57 | 25.26 ± 1.06 | +15.50 | +11.19 | +4.31 | **72%** |
| AZ | 43.17 ± 0.28 | 34.09 ± 0.63 | 28.99 ± 0.51 | +14.18 | +9.08 | +5.10 | **64%** |
| FL | 63.48 ± 1.04 | 37.42 ± 0.76 | 34.46 ± 0.97 | +29.02 | +26.06 | +2.96 | **90%** |
| CA | 60.55 ± 0.81 | 34.47 ± 0.44 | 31.14 ± 1.00 | +29.41 | +26.08 | +3.33 | **89%** |
| TX | 60.35 ± 0.30 | 34.93 ± 0.71 | 32.19 ± 0.61 | +28.16 | +25.42 | +2.74 | **90%** |

Per-visit share ~89–90% at large states (FL/CA/TX) vs 64–72% at small states (AL/AZ): **two-band pattern**.

#### What integration did (now closed)

1. Cherry-picked the 9 per-fold JSONs from `h100/pervisit-fl-ca-tx-results` into `docs/results/phase1_perfold/` (now under `docs/results/phase1_perfold/` post-2026-05-14 reorg).
2. Re-rendered `articles/[BRACIS]_Beyond_Cross_Task/src/figs/per-visit.png` as a 5-panel figure via `articles/[BRACIS]_Beyond_Cross_Task/src/figs/render_per_visit_5state.py`.
3. Updated `articles/[BRACIS]_Beyond_Cross_Task/src/sections/mechanism.tex §6.1` to cover all five states with the two-band pattern.
4. Closed D2 in `articles/[BRACIS]_Beyond_Cross_Task/REVIEW.MD`.

The full prompt detail (cherry-pick command lines, sed-edits, hard-stops, deliverable spec) lives in the git history of this file at commit `a858177`. If you ever need to repeat a similar integration handoff, use it as a template.
