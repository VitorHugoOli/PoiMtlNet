# Phase 2 — CA T2 handoff (mid-flight, 2026-04-29 ~01:00 UTC)

T2 CA grid was running on Colab T4 daemon (PID 31094, launched 23:59 UTC 2026-04-28). Colab session ran out of compute credits / runtime reset before completing all 7 experiments. User is migrating to a fresh Colab session to resume.

## State at handoff (2 of 7 captured)

- **F38a probe c2hgi** ✅ captured locally — `docs/studies/check2hgi/results/probe/california_check2hgi_last.json`
  - F1 = 37.45 ± 0.26 (5-fold)
- **F38a probe hgi** ✅ captured locally — `docs/studies/check2hgi/results/probe/california_hgi_last.json`
  - F1 = 21.32 ± 0.14 (5-fold)
- **Δ probe (c2hgi − hgi) = +16.13 pp** F1, all 5 folds positive (range +15.74 to +16.64)

> The probe JSONs were reconstructed from Colab monitor cell output captured 00:47 UTC. They preserve `f1_per_fold` + `acc_per_fold` + aggregate stats; the `cat_to_id` vocabulary mapping was truncated in the captured output and is not included. This does NOT affect paired-test usage (only `f1_per_fold` is needed). For full provenance, the canonical files were on the prior Colab session's Drive at `/content/drive/MyDrive/mestrado/PoiMtlNet/results/probe/california_*_last.json` before runtime reset.

## What's missing (5 of 7)

| # | Experiment | Status at reset |
|---|---|---|
| F38b | cat STL `next_gru` × c2hgi | Was in fold 1 / epoch ~39 (96 KB log on Drive, no run dir / summary sync'd) |
| F38b | cat STL `next_gru` × hgi | Not started |
| F38c | reg STL `next_getnext_hard` × c2hgi | Not started |
| F38c | reg STL `next_getnext_hard` × hgi | Not started |
| F38d | MTL B3 counterfactual × hgi | Not started |

## CH16 update at CA — scale-amplifying pattern continues

| State | n_rows | Linear probe Δ F1 |
|---|---:|---:|
| AL | 12 K | +12.14 |
| AZ | 26 K | +11.58 |
| FL | 159 K | +15.03 |
| **CA** | **358 K** | **+16.13** |
| TX | TBD | TBD |

CA has 2.3× FL's data and the substrate-only Δ is +1.1 pp larger than FL — consistent with the "substrate gap grows with scale on cat" finding from FL closure. Once the matched-head STL lands, expect the cat-F1 Δ to exceed FL's +29 pp.

## Resume plan in next Colab session

1. **Re-do setup:** mount Drive (`/content/drive/MyDrive/mestrado/PoiMtlNet`), clone repo at `worktree-check2hgi-mtl`, install deps (`requirements_colab.txt`).
2. **Copy CA parquets to /content/output (local NVMe)** — the daemon launcher in commit `b549add` (cell that wrote `/content/run_T2_CA_daemon.sh`) already does this. ~6.7 GB total.
3. **Probes already done** — orchestrator's `run F38a_*` commands are no-ops once their probe JSONs exist on Drive (skip via `already_done` if you re-add that gate, otherwise just delete those two `run` lines from the orchestrator).
4. **Launch the daemon for the remaining 5:**
   ```bash
   nohup setsid bash /content/run_T2_CA_daemon.sh < /dev/null > /content/drive/.../phase2_logs/california_T2_orchestrator.log 2>&1 &
   ```
5. **Monitor cell** polls `/content/T2_pid` and Drive for new `full_summary.json` files. ETA per remaining experiment: ~30 min for cat STL c2hgi (large CA), similar for cat STL hgi, ~30 min × 2 for reg STL pairs, ~50 min for MTL CF. Total ETA ~3 h on T4.

## When T2 closes

Same as T1 closure:
1. Stream remaining JSONs via base64 if Drive doesn't sync to local M4 in time.
2. Extract per-fold to `docs/studies/check2hgi/results/phase1_perfold/CA_*.json` (matching FL/AL/AZ schema).
3. Run paired tests (`scripts/analysis/substrate_paired_test.py`) for cat F1 + reg Acc@10/MRR (TOST δ=2pp).
4. Update PHASE2_TRACKER CA row 🔴→🟢, append CA section to `research/SUBSTRATE_COMPARISON_FINDINGS.md` after FL.
5. Mark TaskUpdate #2 completed, then start T3 (TX) with the same daemon pattern.

## Don't lose

- Orchestrator script content is identical to FL T1's `/content/run_F36d_daemon.sh` but parametrised for `STATE=california` and runs 7 experiments instead of 1. The notebook cell that builds it (cellId `cSmTB7VThniC` in the previous notebook) has the full source.
- Daemon launcher pattern (`nohup setsid bash <script> < /dev/null`) is the only validated way to survive Colab cell-timeouts and kernel idle disconnects. Do NOT use blocking-poll launchers.
- Drive root on the active Colab account is `/content/drive/MyDrive/mestrado/PoiMtlNet/` (NOT `mestrado_data/`). On the M4 local mount it appears as `mestrado (1)/PoiMtlNet/` due to Drive aliasing.
