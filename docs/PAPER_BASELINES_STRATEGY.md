# Paper Baselines Strategy (2026-05-01)

Single source of truth for *which baselines appear in which paper table*, the role each row plays, and what is **deliberately scoped out**. Supersedes earlier per-method scope notes scattered across `PAPER_STRUCTURE.md` and `next_region/comparison.md`.

## Headline next_region table — what readers see

| Row | Role | Protocol | Source |
|---|---|---|---|
| Markov-1-region | floor | study (5-fold StratifiedGroupKFold) | `P0/simple_baselines/<state>/next_region.json` |
| **STAN faithful** (Luo et al., WWW 2021) | external literature, 2021 | study (5f×50ep) | `baselines/faithful_stan_<state>_*.json` |
| **ReHDM faithful** (Li et al., IJCAI 2025) | external literature, 2025 (region-aware) | paper protocol (chronological 80/10/10 + 24h sessions, 5 seeds) | `baselines/REHDM_BS128_<state>_5seeds_50ep_*.json` |
| **STAN-STL × Check2HGI** | substrate-axis probe, our input | study | `P1/region_head_<state>_*_STL_<STATE>_check2hgi_stan_5f50ep.json` |
| **STAN-STL × HGI** | substrate-axis probe, our input | study | `P1/region_head_<state>_*_STL_<STATE>_hgi_stan_5f50ep.json` |
| **STAN-Flow STL × Check2HGI** (`_pf` leak-free) | head-sensitivity probe (graph-prior reg head) | study, per-fold transitions | `phase1_perfold/<S>_check2hgi_reg_gethard_pf_5f50ep.json` |
| **STAN-Flow STL × HGI** (`_pf` leak-free) | head-sensitivity probe | study, per-fold transitions | `phase1_perfold/<S>_hgi_reg_gethard_pf_5f50ep.json` |
| **MTL × Check2HGI / HGI** (our model) | headline | study | `phase1_perfold/<S>_<engine>_mtl_reg_pf.json` |

**Drop from headline (move to supplementary footnote in `next_region/rehdm.md`):**

- ❌ **ReHDM `stl_check2hgi` / `stl_hgi`** — architecture-bound underperformance under cold-user StratifiedGroupKFold + frozen substrate input (1-layer POI encoder + theta/last-pos pooling vs STAN's 4-layer + posenc + last-pos attention). Surgical fixes attempted 2026-05-01 (`use_positional`, `pool_last_pos` flags in `model_stl.py`) — **negative result**. See [`research/REHDM_STL_DIAGNOSIS_20260501.md`](research/REHDM_STL_DIAGNOSIS_20260501.md). Cells exist in `comparison.md` for completeness; not in the headline cross-baseline summary.

## Headline next_category table

| Row | Role |
|---|---|
| Majority class / Markov-1-POI / Markov-K-cat | floors |
| **MHA+PE faithful** (Zeng et al., 2019) | external literature |
| **POI-RGNN faithful** | external literature, GNN baseline |
| Substrate linear probe × C2HGI / HGI | substrate-axis probe (head-free, leak-free by design) |
| **`next_gru` STL × C2HGI / HGI** | substrate-axis probe (matched-head, CH16) |
| **MTL × C2HGI / HGI** (our model) | headline |

## Why not GETNext (Yang et al. SIGIR 2022) as a literature baseline?

We considered adding GETNext-faithful and decided against it:

1. **Task mismatch.** GETNext is a *next-POI* method (~10K classes). Adapting it to *next-region* requires (a) replacing the POI head with a region head, (b) replacing the friendship graph with a region adjacency, (c) re-tuning hyperparams. The result is "GETNext, but adapted" — same sausage-making argument we already have for ReHDM-STL. Adds defensive surface without strengthening the recency claim.
2. **Recency story already covered.** ReHDM (IJCAI 2025) is *more recent* than GETNext (SIGIR 2022) AND is natively region-aware. Reviewers asking *"why no recent next-POI baseline?"* are answered by ReHDM. Adding GETNext duplicates that claim with an older method.
3. **Naming-conflict risk.** Our matched-head probe (`next_getnext_hard`) is **inspired by** the α·log_T graph-prior pattern from GETNext but is **not** a faithful reproduction (different backbone — STAN attention; different task — region not POI). Putting "GETNext" in the literature baseline column risks readers conflating it with our internal probe.

We cite GETNext as the inspiration for the graph-prior pattern in our matched-head probe (see methods §3.4 and the renamed STAN-Flow head below).

## STAN-Flow naming (renamed 2026-05-01)

Previously called `next_getnext_hard` in the registry and "GETNext-hard matched-head" in tables. Renamed to **STAN-Flow** for the paper:

| Component | What it is |
|---|---|
| **Backbone** | STAN-style 4-layer transformer encoder + last-position attention pooling (Luo et al. 2021). |
| **Prior** | GETNext-style `α · log_T` next-region transition prior added to logits (Yang et al. 2022, but adapted from POIs to TIGER tract regions). |
| **`_pf` suffix** | Phase 3 leak-free per-fold transition matrix (`region_transition_log_fold[1..5].pt`), built from train-only edges per fold. |

**Naming rationale:**
- "STAN-" tells readers what the backbone is (familiar 2021 reference).
- "-Flow" refers to the trajectory-flow prior — the `log_T` matrix encodes region transition flow. Avoids the "GETNext" name to prevent conflation with the published next-POI method.

**Registry status (`src/models/registry.py`):**
- `next_stan_flow` — paper-facing alias.
- `next_getnext_hard` — legacy alias preserved for back-compat with existing scripts, JSON filenames (e.g. `*_reg_gethard_pf_5f50ep.json`), and prior commits. Both aliases resolve to the same `NextHeadGETNextHard` class.
- Result-file paths and shell scripts continue to use the legacy `gethard` segment to avoid breaking ~60 historical scripts and ~35 Python files. Paper text always uses **STAN-Flow**.

**Where STAN-Flow appears in the paper:**
- `paper/methods.md §3.3 Task-specific heads`
- `paper/results.md` Table 1 reg STL row
- `paper/limitations.md §6.1 + §6.4`
- `next_region/comparison.md "Substrate-head matched STL — leak-free"` panel
- `FINAL_SURVEY.md §4` (Reg STL CH15 reframing)

## What this means in practice

- **Result-file filenames are NOT renamed.** `region_head_<state>_*_reg_gethard_pf_5f50ep.json`, `region_transition_log_fold[1..5].pt`, etc. retain the legacy `gethard` / `getnext` segments. Renaming would break every script, finalize_phase3.py path, and historical handoff doc.
- **Internal research docs are NOT updated.** `research/F21C_FINDINGS.md`, `research/F49_*.md`, `archive/`, `scope/`, `review/`, etc. retain the original name as audit trail. Search-and-replace there would corrupt the historical record.
- **Shell scripts are NOT renamed.** Filenames like `run_f21c_stl_getnext_hard.sh` are historical run records.
- **Smoke-tested 2026-05-01:** `--heads next_stan_flow` runs end-to-end on GA c2hgi (1f×2ep, Acc@10=0.349 in 4s, aux channel + per-fold log_T paths all wired correctly via the alias).
