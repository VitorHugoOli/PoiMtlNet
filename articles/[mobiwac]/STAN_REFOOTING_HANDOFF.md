# MobiWac 2026: Region-baseline re-footing handoff (STAN board-match + HGI-Istanbul)

> **Why this exists (audit finding, 2026-06-26).** Table 3's region externals are on the **wrong footing** vs our
> headline numbers. The board (our STL/MTL region) is **seed 0, gated stride-1 overlap (`check2hgi_dk_ovl`,
> MIN_SEQ=10), fp32**. But:
> - **STAN-`stl_hgi`** (AL 62.88 / AZ 54.86 / FL 73.58 / CA 60.45 / TX 62.70) was measured at **seed 42** and on the
>   **non-overlap** windowing (source JSONs `region_head_<state>_region_5f_50ep_STAN_HGI_<state>_5f50ep.json`, dated
>   2026-04-25, pre-overlap-board). Wrong seed + wrong windowing.
> - **Istanbul STAN** (PR #51) was run on **our Check2HGI** (`stl_check2hgi`) and on **set-a** (non-overlap): its
>   70.39 lands on the set-a ceiling 70.37, not the board stride-1 ceiling 74.80. Wrong substrate + wrong windowing.
> - **ReHDM-faithful** (AL 66.06 / AZ 54.65 / FL 65.68) is on its **own** protocol (chronological 80/10/10 + 24h
>   sessions + 5 seeds), not our 5-fold user-disjoint CV. This is acceptable AS a published-method reference, but it
>   is NOT a paired/matched comparison.
> - **HMT-GRN** is already board-matched (seed 0, stride-1 overlap) — the one clean region external today.
>
> **Decision (user, 2026-06-26 — incremental, two phases):**
> 1. **PHASE 1 (do first): re-run STAN-`stl_hgi` at the board footing** (seed 0, stride-1 overlap, HGI substrate)
>    for **all Gowalla states: AL/AZ/FL/CA/TX**. **Istanbul STAN is already running on the M4 (Mac) — EXCLUDED from
>    this handoff; do not run it here, and no HGI-Istanbul build is needed here.**
> 2. **PHASE 2 (after Phase 1): run ReHDM-faithful** — **AL/AZ/Istanbul in parallel first, then FL/CA/TX as
>    possible** (faithful is heavy at FL/CA/TX scale; footnote-infeasible is acceptable, as for the Gowalla CA/TX
>    today). ReHDM-faithful is reported as a **published-method reference under its own protocol** (chronological
>    split, 5 seeds), gap-to-ceiling, never a paired/matched cell.
> 3. **Comparability hierarchy (decision 2026-06-26):** **HMT-GRN** (faithful, region-native, board-matched) is
>    the **PRIMARY** region-native comparison. **STAN** (architecture on a pretrained HGI place embedding) and
>    **ReHDM** (own protocol) are **SECONDARY references, each with its regime labeled** — STAN is the one
>    non-faithful region baseline (faithful-STAN from raw falls below the Markov floor and is CA/TX-infeasible, so we
>    footnote that and do not headline it; STAN gets the standard HGI embedding, NOT our Check2HGI). Until the STAN
>    re-runs land, the STAN cells in Table 3 are **PENDING — do not cite the seed-42 / non-overlap numbers as final.**

---

## House rules (same board as everyone)
- **seed 0 × 5 folds (n=5)**; gated **stride-1 overlap**, the SAME windowing as the board STL reg ceiling
  (`region_head_<state>_region_5f_50ep_<state>_ovl_stl_reg_s0.json`); **fp32** (set `DISABLE_AMP=1`); leak-clean
  per-fold **train-only** log_T (rebuild if stale — see CLAUDE.md stale-log_T trap); user-disjoint folds.
- **HGI substrate input** (NOT Check2HGI): `--region-emb-source hgi`. STAN on our Check2HGI is NOT the baseline.
- Verify healthy late best-epochs per fold; never cite a NaN-collapsed fold.
- Commit one result JSON per state + a one-line finding; branch + PR, do not merge to main (orchestrator audits).

## The template run (what produced the board STL reg ceiling)
The board STL reg ceiling at each state is the SAME region-head ablation, on the overlap windowing, seed 0. The STAN
baseline is that run with two swaps: head `next_stan` (the published STAN attention) instead of `next_stan_flow`,
and substrate `hgi` instead of `check2hgi`. Mirror the ceiling command exactly, swap those two flags:

```bash
export DISABLE_AMP=1 PYTHONPATH=src
python scripts/p1_region_head_ablation.py \
    --state <state> --heads next_stan --folds 5 --epochs 50 --seed 0 \
    --input-type region --region-emb-source hgi \
    --engine <the-same-stride1-overlap-engine-the-board-ceiling-used> \
    --per-fold-transition-dir <per-fold seed-0 log_T dir> \
    --tag STAN_HGI_OVL_<state>_5f50ep_s0
#  -> docs/results/P1/region_head_<state>_region_5f_50ep_STAN_HGI_OVL_<state>_5f50ep_s0.json
```
> ⚠ **Confirm the exact overlap-windowing flag against the board ceiling run before launching** — open the board
> ceiling JSON's run config (or the script that produced `*_ovl_stl_reg_s0.json`) and match its windowing/engine/
> input wiring exactly. The ONLY differences from the ceiling are `--heads next_stan` and `--region-emb-source hgi`.
> Row-count gate: the STAN-HGI-overlap region inputs MUST have the same windowed row counts as the board ceiling.

## Phase 1 — STAN at board footing (do FIRST; all Gowalla states)
Run the template above for **AL, AZ, FL, CA, TX**. HGI embeddings already exist (AL/AZ/FL) or were built for the
Tbl-2 HGI cells (CA done PR #52; TX building under Blocker 3 — STAN-TX waits on the TX HGI build). Run them
incrementally (one state at a time is fine; disk is tight, build→train→delete the per-state HGI inputs if needed).
**Istanbul STAN is on the M4 — do NOT run it here, and do NOT build HGI-Istanbul here.**

**Acceptance (Phase 1):** STAN-HGI-overlap Acc@10 below our MTL reg (AL 69.81 / AZ 59.34 / FL 77.28 / CA 65.66 /
TX 67.02) at every Gowalla state, on the matched seed-0 / stride-1 footing. Commit one JSON per state +
`docs/baselines/next_region/stan.md` row; mark the old seed-42 `STAN_HGI` numbers superseded-for-the-paper.

## Phase 2 — ReHDM faithful (do AFTER Phase 1)
Run ReHDM in its **faithful** form (its own architecture + raw inputs + own protocol: chronological 80/10/10 + 24h
sessions + 5 seeds). **Order: AL/AZ/Istanbul in parallel first, then FL/CA/TX as possible.**
- **AL/AZ** faithful already exist (66.06 / 54.65) — re-confirm or reuse; cheap, run in parallel with Istanbul.
- **Istanbul** faithful is NEW: it needs an **FSQ→mahalle region-assignment adapter** (the ReHDM ETL assigns
  regions via US-only geometry; map the Istanbul target to the mahalle taxonomy the board uses). Run it alongside
  AL/AZ. If the adapter proves infeasible by the deadline, footnote Istanbul ReHDM as not-available (do NOT fall
  back to `stl_check2hgi` on our substrate — that is dropped).
- **FL** faithful already exists (65.68); **CA/TX** faithful is heavy (~75–120 h/state) — run **as possible**, else
  footnote "faithful infeasible at scale" (as today).

ReHDM is reported as a **published-method reference under its own protocol**, gap-to-ceiling, NOT a paired/matched
cell. Never report it as if it were on our seed-0 / stride-1 folds. The Istanbul ReHDM `stl_check2hgi` (running in
PR #51, on our substrate + set-a) is **dropped** — superseded by the faithful run above (or the not-available
footnote).

## After the runs: paper-doc updates
- **Table 3** (`src/tables/tbl3_results.tex`): replace the seed-42 STAN cells with the seed-0/stride-1 values; the
  STAN footnote states "STAN architecture on the standard HGI place embedding, seed 0, stride-1 overlap (our
  footing)"; the ReHDM footnote states "ReHDM under its own published protocol (reference)".
- **`docs/baselines/next_region/comparison.md`** + `stan.md`: add the board-footing STAN row; mark the old
  seed-42 `STAN_HGI` row as superseded-for-the-paper (kept for the April substrate study only).
- **`RESULTS_BOARD.md §4`** + **`PAPER_PLAN.md §5.4 / §7`**: STAN board-footing done; ReHDM = footnoted reference.
- **Istanbul row of Table 3**: STAN comes from the **M4** run (not this handoff); POI-RGNN/Markov category cells
  from PR #51 are windowing-robust and can go in now (they sit far below our 53.20/59.89 regardless).

## Acceptance checklist
**Phase 1 (STAN, Gowalla):**
- [ ] STAN-`stl_hgi` re-run at **seed 0 / stride-1 overlap / HGI substrate** committed for **AL/AZ/FL/CA/TX**
  (Istanbul STAN is on the M4, not here).
- [ ] Row-count gate passed (STAN-HGI-overlap inputs == board ceiling row counts) per state.
- [ ] fp32 verified; healthy late best-epochs; no NaN-collapsed fold cited.
- [ ] STAN < our MTL reg at every Gowalla state, on the matched footing.

**Phase 2 (ReHDM, after Phase 1):**
- [ ] ReHDM-faithful AL/AZ/Istanbul run in parallel (Istanbul via the FSQ→mahalle adapter, or footnoted not-available).
- [ ] ReHDM-faithful FL/CA/TX as possible, else footnoted infeasible-at-scale.
- [ ] ReHDM labeled own-protocol reference (never a paired cell); the Istanbul `stl_check2hgi` stop-gap dropped.
- [ ] seed-42 STAN numbers struck from the paper artifacts.
