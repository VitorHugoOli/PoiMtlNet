# Phase 2 — FL status (2026-04-28, mid-flight)

5/7 experiments harvested from Drive after Colab T4 sessions. 2 still pending.

## Headline

**FL substrate gap on cat is much larger than AL+AZ.** Linear probe and matched-head STL both confirm CH16 with stronger effect:

| Leg | C2HGI | HGI | **Δ** | AL ref | AZ ref |
|---|---|---|---|---|---|
| I — linear probe (head-free) | 40.77 ± 1.11 | 25.74 ± 0.26 | **+15.03 pp** | +12.14 | +11.58 |
| II.1 — cat STL `next_gru` matched | **63.43 ± 0.88** | 34.41 ± 0.94 | **+29.02 pp** | +15.50 | +14.52 |
| II.2 — reg STL `next_getnext_hard` Acc@10 | 82.54 ± 0.42 (5f) | 82.22 ± 0.50 (4f) | +0.27 pp (4f paired) | +0.85 | +2.34 |

**Update for SUBSTRATE_COMPARISON_FINDINGS:**

- **CH16 (cat substrate, head-invariant)**: confirmed at FL with **larger** magnitude (+29 vs +15 AL/AZ). Substrate gap grows monotonically with data scale on cat — consistent with the linear probe (+15 FL vs +12 AL / +12 AZ) and the matched-head STL (+29 vs +15.5 / +14.5). This **strengthens** CH16 rather than just replicating it.
- **CH15 reframing (reg ≥ HGI under matched MTL head)**: tiny gap (+0.27 pp Acc@10) at FL, well within TOST δ=2pp non-inferiority. Consistent with AL pattern (AL had +0.85 within TOST). FL reg substrate effect is essentially neutralised at scale.

## What's harvested in the host repo

| File | Notes |
|---|---|
| `results/probe/florida_check2hgi_last.json` | Leg I probe c2hgi |
| `results/probe/florida_hgi_last.json` | Leg I probe hgi |
| `results/phase1_perfold/FL_check2hgi_cat_gru_5f50ep.json` | Leg II.1 c2hgi (5 folds) |
| `results/phase1_perfold/FL_hgi_cat_gru_5f50ep.json` | Leg II.1 hgi (5 folds) |
| `results/P1/region_head_florida_region_5f_50ep_STL_FLORIDA_check2hgi_reg_gethard_5f50ep.json` | Leg II.2 c2hgi (5 folds, native format) |
| `results/phase1_perfold/FL_hgi_reg_gethard_PARTIAL_4f.json` | Leg II.2 hgi (**4 of 5 folds** — extracted from log) |
| `results/phase2_logs/florida_F36*.log` | All 5 launcher logs |

## What's missing

1. **F36c reg STL HGI fold 4** — Colab session SIGKILL'd at start of fold 3 (per the 2KB partial log) and another run got 4 folds before the cell timeout disconnected. Either:
   - Re-run all 5 folds on M4 MPS or fresh Colab session (~30 min T4 / ~5 h MPS)
   - Run only fold 4 by adapting the script (requires manual checkpoint construction)
   - Accept 4-fold paired test (n=4 reduces statistical power but preserves direction)
2. **F36d MTL counterfactual (HGI substrate)** — never ran. Needed for CH18 confirmation at FL. ~30 min on T4 / ~5 h MPS.

## Why Colab kept failing

The `run_code_cell` tool has ~38 min timeout. The blocking-poll launcher (`while proc.poll() is None: time.sleep(60)`) keeps the cell alive past that, causing MCP to disconnect — and Colab eventually treats the kernel as idle and resets the runtime, killing the detached subprocess too.

**Fix for next attempt:** launch the orchestrator as a true daemon (`nohup setsid bash run.sh < /dev/null > log 2>&1 &`), have the cell return immediately, monitor via separate short cells. The bash process is then decoupled from the kernel and survives MCP/cell timeouts.

## Next session plan

Either:
- **(a) Retry on Colab** with the daemon-launcher pattern. Run F36c reg HGI (5f) + F36d MTL CF.
- **(b) Run on M4 MPS** locally via `caffeinate -s python3 scripts/...` — slower (~5h) but no disconnect risk. Existing `scripts/run_phase1_*.sh` orchestrators need a tiny patch to add `florida` to the state list.

Once those 2 land, run paired tests:
```bash
PFD=docs/studies/check2hgi/results/phase1_perfold
python3 scripts/analysis/substrate_paired_test.py \
  --check2hgi $PFD/FL_check2hgi_cat_gru_5f50ep.json \
  --hgi       $PFD/FL_hgi_cat_gru_5f50ep.json \
  --metric f1 --task cat --state florida \
  --output docs/studies/check2hgi/results/paired_tests/florida_cat_f1.json
```

Then update PHASE2_TRACKER FL row 🔴 → 🟢, update SUBSTRATE_COMPARISON_FINDINGS with FL numbers, commit.
