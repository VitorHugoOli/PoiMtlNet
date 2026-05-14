# Assignment — Phase P1a (screen) — 2026-04-16

**Scope:** 210 planned tests = 5 archs × 21 optimizers × 2 states × 1 stage (screen).
**Tier:** `screen` — 1 fold × 10 epochs × batch 4096 × seed 42 × fusion engine.
**State pinning:** Alabama → M2 Pro, Arizona → Linux 4050 (hardware-consistency contract, see `docs/studies/fusion/machines.yaml`).
**Coordinator:** Opus on M2 Pro (this machine). Owns state.json writes. Sub-worker machines use `--no-sync` and never touch state.json.

---

## Pre-flight (all machines)

1. **Git up to date:** `git pull --ff-only` (must include commit `2f3e301` — P1 screen enrollment).
2. **Python venv:** `.venv` with Python 3.12, `sklearn==1.8.0`, `torch==2.11.0`. Other versions may change `StratifiedGroupKFold` fold assignment and invalidate the paired-comparison contract.
3. **Fold hash check:** worker does this automatically before each test via `launch_test.py` preflight. If it aborts with `hash mismatch`, refreeze: `.venv/bin/python scripts/study/freeze_folds.py --state <STATE> --engine fusion --task mtl --force`.
4. **Data on machine:**
   - M2 Pro: already has `output/fusion/alabama/` (confirmed, input + folds).
   - Linux 4050: must rsync `output/fusion/arizona/{input/,folds/}` from M2 Pro (≈233 MB).

---

## mac-m2  (primary, owns state.json)

**Tests:** 105 (all P1_AL_screen_*).
**Test-ID list:** `docs/studies/fusion/assignments/P1_2026-04-16_screen/mac-m2_test_ids.txt`.
**Estimated wall-clock:** 105 × ~1 min ≈ 1h 45m sequential on MPS.

### Prompt to paste in the local Sonnet session

<!-- paste-block-start -->
```
# P1a screen executor — mac-m2 (Alabama)

You are a sub-worker for the fusion study's P1 screen stage. Execute the
Alabama test queue below sequentially. Do NOT touch state.json directly —
use --no-sync so state.json stays coordinator-owned.

## Hard rules

- Run ONLY the test-IDs in the list at
  `docs/studies/fusion/assignments/P1_2026-04-16_screen/mac-m2_test_ids.txt`.
- Use --no-sync on every `study.py next` call.
- Do NOT touch src/, state.json, or any other study file.
- Do NOT modify experiment.py, model configs, or loss registry.
- Stop immediately if any single test fails or hash-check aborts.

## Environment

```bash
cd /Users/vitor/Desktop/mestrado/ingred
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export WORKER_ID=mac-m2
```

## Loop

```bash
while read TID; do
  [ -z "$TID" ] && continue
  echo "=== $TID ==="
  .venv/bin/python scripts/study/study.py next \
      --phase P1 --test-id "$TID" --no-sync \
      --worker-id mac-m2 2>&1 | tail -20
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAILED: $TID (rc=$rc) — stopping loop"
    exit $rc
  fi
done < docs/studies/fusion/assignments/P1_2026-04-16_screen/mac-m2_test_ids.txt
```

## At the end

Print a summary listing:
  - how many tests ran successfully (count of `results/<phase>/<test_id>/summary.json` files under docs/studies/fusion/results/P1/ belonging to AL)
  - any stale .heartbeat files (indicates a silent death)
  - first failing test-ID if the loop stopped early

Do NOT run `study.py import` or `study.py analyze`. The coordinator
(Opus) does that on reconciliation.
```
<!-- paste-block-end -->

---

## linux-4050  (CUDA sub-worker)

**Tests:** 105 (all P1_AZ_screen_*).
**Test-ID list:** `docs/studies/fusion/assignments/P1_2026-04-16_screen/linux-4050_test_ids.txt`.
**Estimated wall-clock:** 105 × ~30s on 4050 ≈ 55 min.

### One-time setup on linux-4050

```bash
# 1. Repo + venv
git clone <repo-url> ingred && cd ingred  # or git pull if already cloned
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Rsync Arizona data from M2 Pro
rsync -avz --progress \
  vitor@<m2-pro-host>:/Users/vitor/Desktop/mestrado/ingred/output/fusion/arizona/ \
  output/fusion/arizona/

# Or if git-lfs / alternative transfer: ensure the following paths exist
# and match SHA-256 of the M2 Pro originals:
#   output/fusion/arizona/input/category.parquet
#   output/fusion/arizona/input/next.parquet
#   output/fusion/arizona/folds/fold_indices_mtl.pt
#   output/fusion/arizona/folds/fold_indices_mtl.meta.json
```

### Prompt to paste in the linux-4050 Sonnet session

<!-- paste-block-start -->
```
# P1a screen executor — linux-4050 (Arizona)

You are a sub-worker for the fusion study's P1 screen stage on a CUDA
machine. Execute the Arizona test queue below sequentially. Do NOT touch
state.json directly — use --no-sync.

## Hard rules

- Run ONLY the test-IDs in the list at
  `docs/studies/fusion/assignments/P1_2026-04-16_screen/linux-4050_test_ids.txt`.
- Use --no-sync on every `study.py next` call.
- Do NOT touch src/, state.json, or any other study file.
- Do NOT modify experiment.py, model configs, or loss registry.
- Stop immediately if any single test fails or hash-check aborts.

## Pre-flight

Confirm fold meta hash matches the current parquet SHA-256 (launch_test.py
does this automatically; if it aborts, refreeze locally rather than
overriding):

```bash
.venv/bin/python scripts/study/freeze_folds.py --state arizona --engine fusion --task mtl --force
```

If you had to refreeze, commit the new meta back to git and push; the
coordinator will need to pull to reconcile.

## Environment

```bash
cd /path/to/ingred
export WORKER_ID=linux-4050
# CUDA should auto-detect via src/configs/globals.py
```

## Loop

```bash
while read TID; do
  [ -z "$TID" ] && continue
  echo "=== $TID ==="
  .venv/bin/python scripts/study/study.py next \
      --phase P1 --test-id "$TID" --no-sync \
      --worker-id linux-4050 2>&1 | tail -20
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAILED: $TID (rc=$rc) — stopping loop"
    exit $rc
  fi
done < docs/studies/fusion/assignments/P1_2026-04-16_screen/linux-4050_test_ids.txt
```

## After all tests run

Rsync results back to the coordinator (M2 Pro):

```bash
rsync -avz --progress \
  docs/studies/fusion/results/P1/ \
  vitor@<m2-pro-host>:/Users/vitor/Desktop/mestrado/ingred/docs/studies/fusion/results/P1/
# Also sync the per-test heartbeat/summary artifacts
```

## Final report

Print a summary with:
  - successful test count (count summary.json files in docs/studies/fusion/results/P1/P1_AZ_*/)
  - stale .heartbeat files (if any)
  - first failing test-ID if the loop stopped early

Do NOT run `study.py import` or `study.py analyze`.
```
<!-- paste-block-end -->

---

## After both machines finish (coordinator workflow)

User pings this Opus session with `/coordinator P1`. Coordinator will:

1. Verify `results/<phase>/<test_id>/` exists for all 210 test-IDs.
2. Check for stale `.worker-*.heartbeat` files (silent-death signal).
3. Loop `study.py import` + `study.py analyze` over every new result.
4. Rollup verdicts by claim (C02, C03, C04, C05) and by arch × optim.
5. Pick top-10 arch/optim combos (by joint F1) for P1b promote stage.
6. Generate `P1_2026-04-XX_promote/ASSIGNMENT.md` with 20 test-IDs
   (top-10 × 2 states).
7. Report gate-check status: whether C05 (expert > FiLM) can be called
   from screen alone, or needs promote/confirm evidence.

---

## Known risks / mitigations

| Risk | Mitigation |
|---|---|
| CUDA vs MPS numerical drift across states | State-pinning: AL only on M2 Pro, AZ only on Linux-4050. No cross-machine paired comparisons. |
| Sonnet silently mutates src/ or state.json | Hard rules in prompt. Audit by diff before accepting (see `memory/feedback_sonnet_worker_audit.md`). |
| One machine finishes early, sits idle | Acceptable — screen budget is ~1h total. If concerning, user can manually redirect by stopping one loop and starting it on the other machine. |
| Fold hash mismatch after rsync | launch_test.py aborts the affected test with a clear error; refreeze locally per pre-flight. |
| Results rsync back to M2 Pro missing files | coordinator `/coordinator P1` scans on-disk results and lists any missing test-IDs; user re-runs those subset. |
