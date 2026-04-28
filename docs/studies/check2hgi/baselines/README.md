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

### `next_region/`

| Baseline | AL | AZ | FL | CA | TX |
|---|---|---|---|---|---|
| Markov-1-region (floor)  | ✅ | ✅ | ✅ | 🔴 | 🔴 |
| **STAN — faithful**      | ✅ | ✅ | ✅ | 🔴 | 🔴 |
| **STAN — stl_check2hgi** | ✅ | ✅ | ✅ | 🔴 | 🔴 |
| **STAN — stl_hgi**       | ✅ | ✅ | ✅ | 🔴 | 🔴 |
| **ReHDM — faithful**     | ✅ | ✅ | 🔴 | 🔴 | 🔴 |
| **ReHDM — stl_check2hgi** | ✅ | 🔴 | 🔴 | 🔴 | 🔴 |
| **ReHDM — stl_hgi**      | ✅ | ✅ | 🔴 | 🔴 | 🔴 |
| GETNext-hard — stl_check2hgi (matched-head, F21c) | ✅ **68.37 ± 2.66** | ✅ **66.74 ± 2.11** | 🔴 (F36c) | 🔴 | 🔴 |
| **GETNext-hard — stl_hgi (matched-head, Phase-1)** | ✅ **67.52 ± 2.80** | ✅ **64.40 ± 2.42** | 🔴 (F36c) | 🔴 | 🔴 |
| **Δ matched-head (paired Wilcoxon, Acc@10)** | +0.85 (p=0.0625 marg, TOST non-inf) | **+2.34 (p=0.0312)** | 🔴 | 🔴 | 🔴 |
| **MTL B3 — substrate-specific (CH18)** | C2HGI 59.60 vs HGI **29.95** | C2HGI 53.82 vs HGI **22.10** | 🔴 (F36d) | 🔴 | 🔴 |
| GRU — stl_check2hgi      | ✅ | ✅ | ✅ | 🔴 | 🔴 |
| GRU — stl_hgi            | ✅ | ✅ | 🔴 | 🔴 | 🔴 |

> ⚠️ **ReHDM uses the paper's protocol** (chronological 80/10/10 + 24h sessions, 5 seeds). The std column for ReHDM rows is inter-seed, not inter-fold; absolute σ values are not directly comparable to the StratifiedGroupKFold-based baselines above. See `next_region/rehdm.md` §"Protocol divergence".

### `next_category/`

| Baseline | AL | AZ | FL | CA | TX |
|---|---|---|---|---|---|
| Majority class (floor) | ✅ | ✅ | ✅ | 🔴 | 🔴 |
| Markov-1-POI (floor) | ✅ | ✅ | ✅ | 🔴 | 🔴 |
| Markov-K-cat (k=1..9) (floor) | ✅ | ✅ | ✅ | 🔴 | 🔴 |
| POI-RGNN — faithful | ✅ | ✅ | ✅ | 🔴 | 🔴 |
| MHA+PE — faithful | ✅ | ✅ | ✅ | 🔴 | 🔴 |
| **C2HGI cat — matched-head `next_gru` (Phase-1)** | ✅ **40.76 ± 1.50** | ✅ **43.21 ± 0.78** | 🔴 (F36b) | 🔴 | 🔴 |
| **HGI cat — matched-head `next_gru` (Phase-1)** | ✅ **25.26 ± 1.06** | ✅ **28.69 ± 0.71** | 🔴 (F36b) | 🔴 | 🔴 |
| **Δ matched-head (paired Wilcoxon p_greater)** | **+15.50 (p=0.0312)** | **+14.52 (p=0.0312)** | 🔴 | 🔴 | 🔴 |
| Substrate linear probe (head-free, Leg I) — C2HGI / HGI / Δ | 30.84 / 18.70 / **+12.14** | 34.12 / 22.54 / **+11.58** | 🔴 (F36a) | 🔴 | 🔴 |
| C2 head-sensitivity probe (next_single, next_lstm) | ✅ all positive | ✅ all positive | 🔴 (not Phase-2 priority) | 🔴 | 🔴 |
| C2HGI cat — `next_single` (legacy P1_5b) | ✅ 38.58 ± 1.23 | ✅ 42.08 ± 0.89 | 🟡 (1f) | 🔴 | 🔴 |
| HGI cat — `next_single` (legacy P1_5b) | ✅ 20.29 ± 1.34 | ✅ 29.69 ± 0.97 (Phase-1) | 🔴 | 🔴 | 🔴 |

(Updated when a new run lands. Mirror in `PAPER_STRUCTURE.md §7` if the cell affects the headline objective.)

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

- **Substrate comparison audit hub** (Check2HGI vs HGI — plan, Phase-1 verdict, Phase-2 tracker, per-fold data, paired tests, linear probes): [`check2hgi_v_hgi/`](check2hgi_v_hgi/README.md). Self-contained for paper-review audits.
- Per-method findings docs: `../research/FAITHFUL_STAN_FINDINGS.md`, `../research/STAN_HGI_FINDINGS.md`, `../research/STAN_THREE_WAY_COMPARISON.md`, `../research/F21C_FINDINGS.md`.
- Paper-level table layout: `../PAPER_STRUCTURE.md §3` and `../results/RESULTS_TABLE.md`.
- Headline objective tracking: `../OBJECTIVES_STATUS_TABLE.md`.
