# Baselines — index & protocol

**Purpose.** Single source of truth for the external-baseline numbers we report in the paper. Pulls together the *what is each baseline*, *why is it a baseline (not our model)*, *how do we run it*, and *what numbers did we get* — across both tasks and all states.

This folder is the read-from for the paper's baseline tables. The raw training JSONs continue to live under `docs/studies/check2hgi/results/` (per the existing study layout); this folder *aggregates* them by baseline + state + task.

---

## Layout

```
baselines/
├── README.md                              ← you are here (protocol + status board)
├── next_category/                         ← task: predict next check-in's category (7 classes, macro-F1 primary)
│   ├── <baseline_name>.md                 ← one per baseline (paper, why-different, adaptations, repro-cmd)
│   ├── results/<state>.json               ← per-state aggregated metrics across all baselines
│   └── comparison.md                      ← cross-baseline comparison table (faithful vs check2hgi vs hgi vs Markov)
└── next_region/                           ← task: predict next check-in's region (~1.1–4.7K classes, Acc@10 primary)
    ├── <baseline_name>.md
    ├── results/<state>.json
    └── comparison.md
```

> **Open question for owner:** the original spec said "3 folders" but only listed `next_category` and `next_region`. Created with 2. If a third task subfolder is needed (e.g. `next_poi`, or a `_floors/` shared-floor folder), add it here and update this README.

---

## Naming conventions

- **Tasks** use snake_case (`next_category`, `next_region`) — matches `task_a` / `task_b` everywhere else in the study.
- **Baseline filenames** use lowercase + underscores (e.g. `stan.md`, `getnext.md`, `hmt_grn.md`).
- **State filenames** use the lowercase state name (e.g. `alabama.json`, `arizona.json`, `florida.json`).
- **Variant keys** inside the baseline subdict of a state JSON:
    - `faithful` — the literature-faithful reproduction (raw inputs, paper architecture).
    - `stl_check2hgi` — single-task with our Check2HGI substrate as input.
    - `stl_hgi` — single-task with our HGI substrate as input.
    - (Other variants — `mtl_check2hgi`, `alibi_init`, etc. — added as needed.)

---

## `<baseline_name>.md` template

Every baseline file under `next_*/` follows this structure:

```markdown
# <Baseline Name>

## Source
- **Paper:** Author, Title. Venue Year. <link>.
- **Reference impl:** <github URL or "none publicly available">.

## Why this is a baseline (not our model)
- 1–2 sentences on what the baseline is testing in our paper (e.g. "external SOTA ceiling on the region task" / "from-scratch reproduction quantifying substrate contribution").

## What's faithful, what's adapted
- **Faithful to paper:** <list>.
- **Adapted because our task / data differ:** <list with one-sentence justification per item>.

## Variants we run
- `faithful` — raw inputs from `data/checkins/`, paper architecture.
- `stl_check2hgi` — same architecture but consuming our Check2HGI substrate as input.
- `stl_hgi` — same architecture but consuming our HGI substrate as input.

## Reproduction commands
```bash
# Per-variant CLI invocations
```

## Source JSONs
| Variant | State | JSON |
|---|---|---|
| `faithful` | AL | `docs/studies/check2hgi/results/baselines/...` |
| `stl_check2hgi` | AL | `docs/studies/check2hgi/results/P1/...` |
| ...      | ... | ... |
```

---

## `<state>.json` schema (v1)

```jsonc
{
  "schema_version": 1,
  "state": "alabama",
  "task": "next_region",
  "n_rows": 12709,
  "n_classes": 1109,
  "protocol": {
    "folds": 5,
    "epochs": 50,
    "seed": 42,
    "stratification": "target_category",
    "groups": "userid",
    "splitter": "StratifiedGroupKFold(shuffle=True)"
  },
  "floors": {
    "majority_class":    { "acc10_mean": 0.91, "acc10_std": null, "source_json": "..." },
    "markov_1_region":   { "acc10_mean": 47.01, "acc10_std": 3.55, "source_json": "..." },
    "top_k_popular":     { "acc10_mean": ..., "acc10_std": ..., "source_json": "..." }
  },
  "baselines": {
    "stan": {
      "faithful": {
        "acc1_mean": 7.86,  "acc1_std": 1.11,
        "acc5_mean": 23.27, "acc5_std": 2.09,
        "acc10_mean": 34.46, "acc10_std": 3.88,
        "mrr_mean": 16.45,  "mrr_std": 1.47,
        "macro_f1_mean": 0.92, "macro_f1_std": 0.32,
        "tag": "FAITHFUL_STAN_al_5f50ep_v4",
        "date": "2026-04-26",
        "source_json": "docs/studies/check2hgi/results/baselines/faithful_stan_alabama_5f_50ep_FAITHFUL_STAN_al_5f50ep_v4.json"
      },
      "stl_check2hgi": { "...": "..." },
      "stl_hgi":       { "...": "..." }
    },
    "<other_baseline>": { "...": "..." }
  }
}
```

**Rules:**
- All percentages stored as percentages (47.01 = 47.01 %, not 0.4701) — matches what the JSONs print.
- `*_std` is sample std across folds (numpy default `ddof=0` to match the result-JSON `aggregate.*_std` fields).
- `source_json` is repo-relative path so it survives moves of `OUTPUT_DIR`.
- `tag` is the run-tag passed to the trainer's `--tag` flag (lets us locate the originating command).
- `date` is YYYY-MM-DD of when the run was finalised.

---

## `comparison.md` template (per task)

Top-level summary table:

```markdown
# next_region — baseline comparison

## Cross-baseline summary (Acc@10 primary, mean ± σ)

| Baseline | Variant | AL | AZ | FL | CA | TX |
|---|---|---:|---:|---:|---:|---:|
| Markov-1-region (floor) | — | 47.01 ± 3.55 | 42.96 ± 2.05 | 65.05 | — | — |
| STAN | faithful | 34.46 ± 3.88 | 38.96 ± 3.41 | 65.36 ± 0.69 | 🔴 | 🔴 |
| STAN | stl_check2hgi | 59.20 ± 3.62 | 52.24 ± 2.38 | 72.62 ± 0.52 | 🔴 | 🔴 |
| STAN | stl_hgi | 62.88 ± 3.90 | 54.86 ± 2.84 | 73.58 ± 0.43 | 🔴 | 🔴 |
| <other_baseline> | ... | ... | ... | ... | ... | ... |

(🔴 = pending, 🟡 = partial / 1-fold, ✅ = 5-fold complete.)

## Per-baseline detail
[per-baseline section pulling all metrics from <state>.json files]
```

---

## Status board (live)

> Legend: ✅ 5-fold/seed complete · 🟡 partial · 🔴 pending · ⚪ intentionally out of scope (see `GAP_A_CLOSURE_20260430.md`).

### Coverage matrix — all states × all baselines

**External baselines (leak-free by construction; reported in headline tables).**

| Baseline / variant | task | AL | AZ | FL | CA | TX | GA |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Majority + Markov-1-POI floors | next_category | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Markov-K-cat floors (k=1..9) | next_category | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MHA+PE — `faithful` (5f×11ep) | next_category | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| POI-RGNN — `faithful` (5f×35ep) | next_category | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Markov-1-region floor | next_region | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| STAN — `faithful` (5f×50ep) | next_region | ✅ | ✅ | ✅ | ⚪ | ⚪ | ✅ |
| STAN — `stl_check2hgi` (5f×50ep) | next_region | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| STAN — `stl_hgi` (5f×50ep) | next_region | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ReHDM — `faithful` (5 seeds × 50ep, paper proto) | next_region | ✅ | ✅ | ✅ § | ⚪ | ⚪ | ✅ § |
| ReHDM — `stl_check2hgi` (5f×50ep, study proto) | next_region | ✅ | ✅ | ✅ | ⚪ | ⚪ | ✅ |
| ReHDM — `stl_hgi` (5f×50ep, study proto) | next_region | ✅ | ✅ | ✅ | ⚪ | ⚪ | ✅ |

§ ReHDM faithful at FL + GA uses `batch_size=128 + lr/max_lr scaled 4×` (linear scaling rule from paper b=32). Validated on AL/AZ within 1σ of paper-batch references. Inter-seed σ (5 seeds), not inter-fold — see [`next_region/rehdm.md §"Protocol & architecture choices per variant"`](next_region/rehdm.md).

CA/TX faithful axis (STAN-faithful, ReHDM-faithful) was scoped out per [`../GAP_A_CLOSURE_20260430.md`](../GAP_A_CLOSURE_20260430.md): substrate axis already 5-state via STAN-STL, faithful-axis ETA at CA/TX scale (~75-120 h/state ReHDM, ~5-7 h/fold STAN) was not justified given coverage. ReHDM at CA/TX therefore ⚪.

**Substrate-comparison tracks (Phase 1+2+3 — the paper's primary contribution; reported in [`../FINAL_SURVEY.md`](../FINAL_SURVEY.md)).**

| Track | Head | AL | AZ | FL | CA | TX | GA |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **Substrate linear probe** (Leg I, head-free, leak-free by design) | LR on emb | ✅ | ✅ | ✅ | ✅ | ✅ | ⚪ |
| **Cat STL CH16** (Leg II.1) | `next_gru` | ✅ | ✅ | ✅ | ✅ | ✅ | ⚪ |
| **Reg STL CH15 — leak-free Phase 3** (`_pf` per-fold transitions) | `next_getnext_hard_pf` | ✅ | ✅ | ✅ | ✅ | ✅ | ⚪ |
| **MTL B9 cat-side CH18-cat** (leak-free) | `next_gru` co-trained | ✅ | ✅ | ✅ | ✅ | ✅ | ⚪ |
| **MTL B9 reg-side CH18-reg** (leak-free) | `next_getnext_hard_pf` co-trained | ✅ | ✅ | ✅ | ✅ | ✅ | ⚪ |

⚪ GA is scoped to external-baseline coverage only; the Phase 1-3 substrate-comparison axis was closed at AL/AZ/FL/CA/TX (5 states with paper-grade Wilcoxon p=0.0312 = max-n=5). Adding GA to the substrate axis is not necessary for paper claims.

**Phase 1 leaky panel — preserved for the F44 leak-shift analysis in [`../FINAL_SURVEY.md §6`](../FINAL_SURVEY.md), do NOT cite as headline substrate finding.**

| Track (leaky) | AL Acc@10 | AZ Acc@10 | FL Acc@10 | CA Acc@10 | TX Acc@10 |
|---|---:|---:|---:|---:|---:|
| Check2HGI `next_getnext_hard` (leaky) | 68.37 | 66.74 | 82.54 | 70.63 | 69.31 |
| HGI `next_getnext_hard` (leaky)       | 67.52 | 64.40 | 82.25 | 71.29 | 69.90 |

Sign-flips at all 5 states once the `α·log_T` leak is removed (substrate-asymmetric). See [`next_region/comparison.md §"Substrate-head matched STL — leak-free"`](next_region/comparison.md) for the corrected Phase 3 numbers.

### `next_region/` — Acc@10 (mean ± σ); ★ = best per state

| Baseline | AL | AZ | FL | CA | TX | GA |
|---|---:|---:|---:|---:|---:|---:|
| Markov-1-region (floor)            | 47.01 ± 3.55 | 42.96 ± 2.05 | 65.05 ± 0.93 | 52.09 ± 0.80 | 54.94 ± 0.46 | 48.19 ± 2.18 |
| STAN — `faithful`                  | 34.46 ± 3.88 | 38.96 ± 3.41 | 65.36 ± 0.69 | ⚪ skip | ⚪ skip | 40.68 ± 1.10 |
| STAN — `stl_check2hgi`             | 59.20 ± 3.62 | 52.24 ± 2.38 | 72.62 ± 0.52 | 58.82 ± 1.04 | 61.35 ± 0.36 | 56.35 ± 2.40 |
| **STAN — `stl_hgi`**               | 62.88 ± 3.90 | 54.86 ± 2.84 | **★ 73.58 ± 0.43** | **★ 60.45 ± 0.97** | **★ 62.70 ± 0.37** | **★ 58.58 ± 1.86** |
| **ReHDM — `faithful` (paper-proto)** § | **★ 66.06 ± 0.98** | **★ 54.65 ± 0.77** | 65.68 ± 0.26 | ⚪ skip | ⚪ skip | 55.82 ± 0.76 |
| ReHDM — `stl_check2hgi` ‡          | 26.22 ± 1.58 | 23.24 ± 1.27 | 38.74 ± 0.49 | ⚪ skip | ⚪ skip | 22.31 ± 1.31 |
| ReHDM — `stl_hgi` ‡                | 42.78 ± 2.82 | 34.00 ± 3.02 | 54.49 ± 0.32 | ⚪ skip | ⚪ skip | 35.07 ± 1.98 |
| GETNext-hard `_pf` — `stl_check2hgi` (leak-free) | 59.15 ± 3.48 | 50.24 ± 2.51 | 69.22 ± 0.52 | 55.92 ± 1.20 | 58.89 ± 1.28 | 🔴 pending |
| GETNext-hard `_pf` — `stl_hgi` (leak-free)       | 61.86 ± 3.29 | 53.37 ± 2.55 | 71.34 ± 0.64 | 57.77 ± 1.12 | 60.47 ± 1.26 | 🔴 pending |

> § ReHDM `faithful` uses the paper's protocol (chronological 80/10/10 + 24h sessions, 5 seeds). σ is inter-seed; not cell-for-cell σ-comparable to StratifiedGroupKFold rows. AL/AZ at paper b=64; FL+GA at b=128 + 4× lr scaling (validated within 1σ on AL/AZ — see `rehdm.md`).
> ‡ ReHDM `stl_*` may have a known mask-handling bug (target_mask all-ones over padded positions). Patch in flight 2026-05-01; numbers above are pre-patch. Re-run will land separately.

### `next_category/` — macro-F1 (mean ± σ); ★ = best per state

| Baseline | AL | AZ | FL | CA | TX | GA |
|---|---:|---:|---:|---:|---:|---:|
| Majority class (floor)             |  7.28 ± 0.00 |  7.25 ± 0.00 |  5.66 ± 0.00 |  7.04 ± 0.00 |  6.76 ± 0.00 |  6.69 ± 0.00 |
| Markov-1-POI (floor)               | 16.81 ± 1.06 | 19.48 ± 0.63 | 27.60 ± 0.32 | 24.95 ± 1.18 | 25.85 ± 0.55 | 21.36 ± 0.36 |
| best Markov-K-cat (floor)          | 20.50 ± 0.67 (k=5) | 23.92 ± 2.26 (k=5) | 29.74 ± 1.19 (k=3) | 27.59 ± 0.61 (k=5) | 28.67 ± 0.66 (k=5) | 27.01 ± 1.10 (k=3) |
| MHA+PE — `faithful`                | 18.95 ± 0.71 | 24.99 ± 0.85 | 32.06 ± 0.23 | 29.13 ± 0.71 | 29.91 ± 0.43 | 27.62 ± 0.97 |
| **POI-RGNN — `faithful`**          | **★ 23.80 ± 1.12** | **★ 27.64 ± 2.34** | **★ 33.35 ± 1.14** | **★ 30.71 ± 0.82** | **★ 32.08 ± 0.70** | **★ 30.24 ± 0.87** |
| C2HGI cat — matched-head `next_gru` (substrate axis) | 40.76 ± 1.68 | 43.21 ± 0.87 | 63.43 ± 0.98 | 59.94 ± 0.59 | 60.24 ± 1.84 | 🔴 pending |
| HGI cat — matched-head `next_gru` (substrate axis)   | 25.26 ± 1.18 | 28.69 ± 0.79 | 34.41 ± 1.05 | 31.13 ± 1.04 | 31.89 ± 0.55 | 🔴 pending |
| Δ matched-head (Wilcoxon p_greater)                  | +15.50 (p=0.0312) | +14.52 (p=0.0312) | +29.02 (p=0.0312) | +28.81 (p=0.0312) | +28.34 (p=0.0312) | 🔴 pending |
| Substrate linear probe — C2HGI F1                    | 30.84 ± 2.26 | 34.12 ± 1.36 | 40.77 ± 1.24 | 37.45 ± 0.29 | 38.38 ± 0.28 | 🔴 pending |
| Substrate linear probe — HGI F1                      | 18.70 ± 1.54 | 22.54 ± 0.50 | 25.74 ± 0.29 | 21.32 ± 0.16 | 22.33 ± 0.25 | 🔴 pending |
| Substrate linear probe — Δ (C2HGI − HGI)             | +12.14 | +11.58 | +15.03 | +16.13 | +16.06 | 🔴 pending |

(★ marks the best non-floor numeric value per state. Substrate-axis and Markov-K-cat-floor rows below the dividing line are **secondary** to the headline external-baseline rows above; ★ is restricted to the headline rows so the table reflects the published-architecture ranking. Substrate-axis cells for GA are pending — substrate-comparison runs were originally scoped out for GA and are being launched 2026-05-01 to close the matrix.)

---

## How to add a new baseline result

1. Run the experiment via the relevant CLI (Faithful STAN: `python -m research.baselines.stan.train ...` ; STL substrate: `python scripts/p1_region_head_ablation.py ...`).
2. Update or create `next_<task>/results/<state>.json` with the new `baselines.<name>.<variant>` block (see schema above). Mark `tag`, `date`, and `source_json`.
3. Update the relevant `next_<task>/<baseline_name>.md` to mention the new variant + repro command if not already there.
4. Refresh `next_<task>/comparison.md` summary table.
5. Refresh the status board in this README.
6. (Optional) if the result changes a paper claim, update `PAPER_STRUCTURE.md §7` and `CLAIMS_AND_HYPOTHESES.md`.

---

## Cross-references

- **Substrate comparison (Check2HGI vs HGI)** — plan: [`../research/SUBSTRATE_COMPARISON_PLAN.md`](../research/SUBSTRATE_COMPARISON_PLAN.md); Phase-1 findings + sources appendix: [`../research/SUBSTRATE_COMPARISON_FINDINGS.md`](../research/SUBSTRATE_COMPARISON_FINDINGS.md); Phase-2 work queue: [`../PHASE2_TRACKER.md`](../PHASE2_TRACKER.md). Per-fold data lives in `../results/{phase1_perfold,probe,paired_tests}/`. Phase-1 matched-head summary rows are in `next_category/comparison.md` and `next_region/comparison.md`.
- Per-method findings docs: `../research/FAITHFUL_STAN_FINDINGS.md`, `../research/STAN_HGI_FINDINGS.md`, `../research/STAN_THREE_WAY_COMPARISON.md`, `../research/F21C_FINDINGS.md`.
- Paper-level table layout: `../PAPER_STRUCTURE.md §3` and `../results/RESULTS_TABLE.md`.
- Headline objective tracking: `../OBJECTIVES_STATUS_TABLE.md`.
