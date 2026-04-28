# Check2HGI vs HGI — substrate comparison audit hub

**Created 2026-04-27** as a self-contained audit unit for the substrate-comparison work. Everything needed to verify the substrate claims (CH16 + CH15-reframed + CH18 + CH19) lives here: the plan, the verdict, the Phase-2 tracker, the per-fold metrics, the paired-test outputs, and the linear-probe outputs. Original training-result dirs (which contain checkpoints + train/val curves + classification reports) remain at their canonical study locations and are pointed to from `data/SOURCES.md`.

> **Why this folder exists.** The substrate claim is the paper's central scientific finding. It will be the most-audited section by reviewers. Centralising the docs + data here means an auditor can verify every number from a single tree without having to dig through the broader `results/` hierarchy.

---

## Folder layout

```
check2hgi_v_hgi/
├── README.md                       ← you are here
├── plan.md                         ← SUBSTRATE_COMPARISON_PLAN — phase-gated 3-leg framework
├── phase1_verdict.md               ← AL+AZ outcome matrix + paper-ready findings (Phase 1 closed)
├── phase2_tracker.md               ← FL+CA+TX work queue (Phase 2)
└── data/
    ├── SOURCES.md                  ← canonical-path index for every data file in this folder
    ├── linear_probe/               ← Leg I — head-free substrate diagnostic
    │   ├── alabama_check2hgi_last.json
    │   ├── alabama_hgi_last.json
    │   ├── arizona_check2hgi_last.json
    │   ├── arizona_hgi_last.json
    │   └── alabama_check2hgi_pooled_last.json    ← C4 mechanism counterfactual
    ├── cat_stl_per_fold/           ← Leg II.1 (matched-head + head-sensitivity probes)
    │   ├── {AL,AZ}_{check2hgi,hgi}_cat_gru_5f50ep.json     ← matched-head (next_gru = MTL B3 cat)
    │   ├── {AL,AZ}_{check2hgi,hgi}_cat_single_5f50ep.json  ← C2 head-sensitivity probe (next_single)
    │   └── {AL,AZ}_{check2hgi,hgi}_cat_lstm_5f50ep.json    ← C2 head-sensitivity probe (next_lstm)
    ├── reg_stl_per_fold/           ← Leg II.2 — matched-head reg STL
    │   └── {AL,AZ}_{check2hgi,hgi}_reg_gethard_5f50ep.json ← matched-head (next_getnext_hard = MTL B3 reg)
    ├── mtl_counterfactual_per_fold/  ← Leg III — MTL B3 with HGI substituted
    │   └── {AL,AZ}_hgi_mtl_{cat,reg}.json                  ← per-fold cat F1 + reg Acc@10_indist + MRR
    ├── c4_poi_pooled/              ← C4 — POI-pooled C2HGI mechanism counterfactual
    │   └── AL_check2hgi_pooled_cat_gru_5f50ep.json
    └── paired_tests/               ← C3 — Wilcoxon + paired-t + TOST analyser outputs
        ├── alabama_cat_f1.json     ← matched-head cat (next_gru) Wilcoxon
        ├── alabama_single_cat_f1.json     ← head-sensitivity probe Wilcoxon
        ├── alabama_lstm_cat_f1.json
        ├── arizona_cat_f1.json
        ├── arizona_single_cat_f1.json
        ├── arizona_lstm_cat_f1.json
        ├── alabama_acc10_reg_acc10.json   ← matched-head reg (next_getnext_hard) Wilcoxon + TOST δ=2pp
        ├── alabama_mrr_reg_mrr.json
        ├── arizona_acc10_reg_acc10.json
        └── arizona_mrr_reg_mrr.json
```

---

## How to read in 60 seconds (auditor's entry point)

1. **Open [`phase1_verdict.md`](phase1_verdict.md) §6** — the resolved outcome-interpretation matrix. Strong claim confirmed at AL+AZ.
2. Pick any cell from the §1–§5 result tables in `phase1_verdict.md`.
3. Trace it to `data/`:
   - For matched-head cat F1: `data/cat_stl_per_fold/<state>_<engine>_cat_gru_5f50ep.json` (5 fold-dicts).
   - For matched-head reg Acc@10: `data/reg_stl_per_fold/<state>_<engine>_reg_gethard_5f50ep.json`.
   - For MTL+HGI counterfactual: `data/mtl_counterfactual_per_fold/<state>_hgi_mtl_<task>.json`.
   - For statistical claim: `data/paired_tests/<state>_<task>_<metric>.json` carries the per-fold deltas, paired-t, Wilcoxon, and TOST results.
4. Need the *training* curves + classification reports + checkpoints behind a per-fold number? See `data/SOURCES.md` for the canonical-path map.

## Headline findings (from `phase1_verdict.md`)

### CH16 — Check2HGI > HGI on next_category, head-invariant

| State | Probe | C2HGI F1 | HGI F1 | Δ | Wilcoxon p_greater |
|---|---|---:|---:|---:|---:|
| AL | Linear (head-free) | 30.84 ± 2.02 | 18.70 ± 1.38 | **+12.14** | n/a |
| AL | next_gru (matched-head MTL) | **40.76 ± 1.50** | 25.26 ± 1.06 | **+15.50** | **0.0312** |
| AL | next_single | 38.71 ± 1.32 | 26.76 ± 0.36 | **+11.96** | **0.0312** |
| AL | next_lstm | 38.38 ± 1.08 | 23.94 ± 0.84 | **+14.44** | **0.0312** |
| AZ | Linear (head-free) | 34.12 ± 1.22 | 22.54 ± 0.45 | **+11.58** | n/a |
| AZ | next_gru (matched-head MTL) | **43.21 ± 0.78** | 28.69 ± 0.71 | **+14.52** | **0.0312** |
| AZ | next_single | 42.20 ± 0.72 | 29.69 ± 0.97 | **+12.50** | **0.0312** |
| AZ | next_lstm | 41.86 ± 0.84 | 26.50 ± 0.29 | **+15.36** | **0.0312** |

**8/8 head-state probes positive at maximum-significance n=5 paired Wilcoxon (5/5 folds positive each).**

### CH15 reframing — Check2HGI ≥ HGI on next_region under matched MTL head

| State | Probe | C2HGI Acc@10 | HGI Acc@10 | Δ | Wilcoxon p_greater (Acc@10) | TOST δ=2pp |
|---|---|---:|---:|---:|---:|---|
| AL | STAN (existing CH15) | 59.20 ± 3.62 | **62.88 ± 3.90** | −3.68 | — | — |
| AL | next_getnext_hard (matched MTL) | **68.37 ± 2.66** | 67.52 ± 2.80 | +0.85 | 0.0625 marginal | non-inferior ✅ |
| AZ | STAN | 52.24 ± 2.38 | **54.86 ± 2.84** | −2.62 | — | — |
| AZ | next_getnext_hard (matched MTL) | **66.74 ± 2.11** | 64.40 ± 2.42 | **+2.34** | **0.0312** ✅ | non-inferior ✅ |

The previous CH15 verdict was **head-coupled** to STAN's preference for POI-stable smoothness; under the matched MTL reg head (graph prior), the substrate gap closes (AL) or reverses sign in C2HGI's favor (AZ).

### CH18 — MTL B3 is substrate-specific

MTL B3 (`mtlnet_crossattn + static cat=0.75 + next_gru cat + next_getnext_hard reg`), 5f×50ep seed 42:

| State | Substrate | cat F1 | reg Acc@10_indist | Δ_cat (C2HGI−HGI) | Δ_reg (C2HGI−HGI) |
|---|---|---:|---:|---:|---:|
| AL | C2HGI (existing) | **42.71 ± 1.37** | **59.60 ± 4.09** | — | — |
| AL | HGI (counterfactual) | 25.96 ± 1.61 | 29.95 ± 1.89 | **+16.75** | **+29.65** |
| AZ | C2HGI (existing) | **45.81 ± 1.30** | **53.82 ± 3.11** | — | — |
| AZ | HGI (counterfactual) | 28.70 ± 0.51 | 22.10 ± 1.63 | **+17.11** | **+31.72** |

MTL+HGI is **worse than STL+HGI** on reg by ~37 pp at AL (STL HGI gethard = 67.52 → MTL HGI = 29.95). The B3 configuration was tuned around Check2HGI's per-visit context and does not generalise to HGI substrate.

### CH19 — Per-visit context = ~72% of cat substrate gap

POI-pooled Check2HGI (mean per `placeid` across check-ins) under matched-head `next_gru` STL at AL:

| Substrate | Linear probe F1 | Matched-head STL F1 |
|---|---:|---:|
| Check2HGI canonical | 30.84 ± 2.02 | 40.76 ± 1.50 |
| **Check2HGI POI-pooled** | **23.20 ± 1.08** | **29.57** |
| HGI | 18.70 ± 1.38 | 25.26 ± 1.06 |

Decomposition under matched-head STL (substrate gap +15.50 pp):
- Per-visit context = canonical − pooled = **+11.19 pp (~72%)**
- Training signal = pooled − HGI = **+4.31 pp (~28%)**

Per-visit variation is the dominant mechanism; training signal is a real but secondary contribution.

---

## Reproduction commands

All runs on M4 Pro under `caffeinate -s`. See `plan.md` §1 for the matched-head + grid definitions.

```bash
export PYTHONPATH=src
export DATA_ROOT=/Users/vitor/Desktop/mestrado/ingred
export OUTPUT_DIR=/Users/vitor/Desktop/mestrado/ingred/output
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Leg I — substrate-only linear probe (head-free)
python3 scripts/probe/substrate_linear_probe.py --state $STATE --engine $ENGINE

# Leg II.1 — cat STL with matched-head next_gru
caffeinate -s python3 scripts/train.py \
  --task next --state $STATE --engine $ENGINE --model next_gru \
  --folds 5 --epochs 50 --seed 42 --no-checkpoints

# Leg II.2 — reg STL with matched-head next_getnext_hard
caffeinate -s python3 scripts/p1_region_head_ablation.py \
  --state $STATE --heads next_getnext_hard \
  --folds 5 --epochs 50 --seed 42 --input-type region \
  --region-emb-source $ENGINE \
  --override-hparams d_model=256 num_heads=8 \
      "transition_path=$OUTPUT_DIR/check2hgi/$STATE/region_transition_log.pt" \
  --tag STL_${STATE}_${ENGINE}_reg_gethard_5f50ep

# Leg III — MTL counterfactual (HGI substituted into B3)
# Pre-flight: build HGI's input/next_region.parquet (substrate-free labels)
python3 scripts/probe/build_hgi_next_region.py --state $STATE
caffeinate -s python3 scripts/train.py \
  --task mtl --state $STATE --engine hgi \
  --task-set check2hgi_next_region --model mtlnet_crossattn \
  --mtl-loss static_weight --category-weight 0.75 \
  --reg-head next_getnext_hard --cat-head next_gru \
  --folds 5 --epochs 50 --seed 42 --no-checkpoints

# C4 mechanism — POI-pooled C2HGI
python3 scripts/probe/build_check2hgi_pooled.py --state alabama
python3 scripts/probe/substrate_linear_probe.py --state alabama --engine check2hgi_pooled
caffeinate -s python3 scripts/train.py \
  --task next --state alabama --engine check2hgi_pooled --model next_gru \
  --folds 5 --epochs 50 --seed 42 --no-checkpoints

# Paired tests + TOST (post-hoc on per-fold JSONs in this folder)
python3 scripts/analysis/substrate_paired_test.py \
  --check2hgi data/cat_stl_per_fold/AL_check2hgi_cat_gru_5f50ep.json \
  --hgi       data/cat_stl_per_fold/AL_hgi_cat_gru_5f50ep.json \
  --metric f1 --task cat --state alabama --tost-margin 0.02
```

---

## Cross-references (study-wide)

- `../../README.md` — study entry point.
- `../../SESSION_HANDOFF_2026-04-27.md` — Phase-1 session handoff.
- `../../CLAIMS_AND_HYPOTHESES.md §CH16, CH15, CH18, CH19` — the claim catalog entries this folder underwrites.
- `../../OBJECTIVES_STATUS_TABLE.md` — paper-objective scorecard.
- `../../NORTH_STAR.md §Caveats — Phase-1 substrate-specific addendum` — MTL B3 substrate-specific note.
- `../../PAPER_STRUCTURE.md §4` — STL-baseline matching policy revised to use this folder's matched-head data.
- `../README.md` — broader baselines tracker (this folder is a subset of it focused on substrate-comparison only).
