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
> **Decision (user, 2026-06-26):** (1) **Re-run STAN-`stl_hgi` at the board footing** (seed 0, stride-1 overlap,
> HGI substrate) for AL/AZ/FL/CA/TX **and Istanbul** (Istanbul needs an HGI build first). (2) **ReHDM-faithful stays
> as a footnoted published reference** (its own protocol), never as a paired cell. (3) HMT-GRN stays as the matched
> region-native external. Until the STAN re-runs land, the STAN cells in Table 3 are **PENDING — do not cite the
> seed-42 / non-overlap numbers as final.**

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

## Per state
1. **AL, AZ, FL, CA, TX** — HGI embeddings already exist (AL/AZ/FL) or were built for the Tbl-2 HGI cells
   (CA done PR #52; TX building). Run the template above. **Acceptance:** STAN-HGI-overlap Acc@10 below our MTL reg
   (AL 69.81 / AZ 59.34 / FL 77.28 / CA 65.66 / TX 67.02) at every state, on the matched seed-0/stride-1 footing.
2. **Istanbul** — HGI-Istanbul does NOT exist yet. **Build it first** (HGI trains on CPU; mirror the CA/TX HGI build
   `scripts/closing_data/build_catx_hgi.sh`, pointed at Istanbul / the mahalle region set), verify
   `region_embeddings` nan=False std>0, then run the template with `--state istanbul --region-emb-source hgi` on the
   **stride-1** Istanbul windowing (the board Istanbul is stride-1 GCN; match it). **Acceptance:** STAN-HGI-overlap
   Istanbul Acc@10 below our MTL reg 74.28, on the stride-1 footing (NOT the set-a 70.39 that PR #51 produced).

## ReHDM (no re-run; framing only)
Keep ReHDM-`faithful` (AL 66.06 / AZ 54.65 / FL 65.68; CA/TX footnoted infeasible at scale). **Label it explicitly
as a published-method reference under ReHDM's own protocol** (chronological 80/10/10 + 24h sessions + 5 seeds),
gap-to-ceiling, NOT a paired/matched cell. Do NOT report it as if it were on our seed-0/stride-1 folds. The Istanbul
ReHDM (`stl_check2hgi`, running in PR #51) is on our substrate + set-a — drop it or, if wanted, run ReHDM-faithful
for Istanbul with an FSQ→mahalle region adapter (heavy; footnote-infeasible is acceptable, matching CA/TX).

## After the runs: paper-doc updates
- **Table 3** (`src/tables/tbl3_results.tex`): replace the seed-42 STAN cells with the seed-0/stride-1 values; the
  STAN footnote states "STAN architecture on the standard HGI place embedding, seed 0, stride-1 overlap (our
  footing)"; the ReHDM footnote states "ReHDM under its own published protocol (reference)".
- **`docs/baselines/next_region/comparison.md`** + `stan.md`: add the board-footing STAN row; mark the old
  seed-42 `STAN_HGI` row as superseded-for-the-paper (kept for the April substrate study only).
- **`RESULTS_BOARD.md §4`** + **`PAPER_PLAN.md §5.4 / §7`**: STAN board-footing done; ReHDM = footnoted reference.
- **Istanbul row of Table 3**: fill STAN (board-footing) once it lands; POI-RGNN/Markov category cells from PR #51
  are windowing-robust and can go in now (they sit far below our 53.20/59.89 regardless).

## Acceptance checklist
- [ ] STAN-`stl_hgi` re-run at **seed 0 / stride-1 overlap / HGI substrate** committed for AL/AZ/FL/CA/TX.
- [ ] HGI-Istanbul built (nan=False, std>0); STAN-`stl_hgi` Istanbul at stride-1 committed.
- [ ] Row-count gate passed (STAN-HGI-overlap inputs == board ceiling row counts) per state.
- [ ] fp32 verified; healthy late best-epochs; no NaN-collapsed fold cited.
- [ ] STAN < our MTL reg at every state, on the matched footing.
- [ ] ReHDM labeled own-protocol reference; seed-42 STAN numbers struck from the paper artifacts.
