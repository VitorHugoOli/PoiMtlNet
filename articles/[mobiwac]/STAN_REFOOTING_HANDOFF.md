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
> **Decision (user, 2026-06-26 — UPDATED after the faithful-STAN literature + implementation audit):**
> 1. **PHASE 1 (do first): run FAITHFUL STAN** — STAN's OWN embeddings learned end-to-end **from raw**
>    (`research/baselines/stan/`), NOT fed our HGI/Check2HGI embedding (the literature norm; feeding STAN a
>    pretrained embedding is non-standard). At the board footing (seed 0, **stride-1 overlap**, user-disjoint folds)
>    for **AL/AZ/FL** (CA/TX faithful-STAN is infeasible at scale → footnote infeasible, like ReHDM). **Istanbul STAN
>    is on the M4 — EXCLUDED here.**
>    ⚠ **The current faithful-STAN v4 numbers (AL 34.46 / AZ 38.96, below the Markov floor) are UNDER-TRAINED
>    ARTIFACTS — DO NOT CITE THEM.** A two-agent audit (2026-06-26) found they are confounded by under-training
>    (best-epochs at 49/50, still climbing), stride-9 data starvation (~9x too few windows), and an under-powered
>    STAN-DERIVED head. The Phase-1 re-run MUST apply the audit fixes (§"Phase 1") before any STAN number is reported.
> 2. **PHASE 2 (after Phase 1): run ReHDM-faithful** — **AL/AZ/Istanbul in parallel first, then FL/CA/TX as
>    possible** (faithful is heavy at FL/CA/TX scale; footnote-infeasible is acceptable). ReHDM-faithful is reported
>    as a **published-method reference under its own protocol** (chronological split, 5 seeds), never a paired cell.
> 3. **Comparability hierarchy:** **HMT-GRN** (faithful, region-native, board-matched, multi-task) is the **PRIMARY**
>    region-native comparison. **STAN (faithful, from raw — after the fixed re-run)** and **ReHDM (own protocol)** are
>    **SECONDARY references, each labeled.** The substrate-bound **STAN-`stl_hgi`** (STAN on our HGI embedding) is now
>    ONLY an OPTIONAL, explicitly-labeled **ablation** (isolates substrate vs architecture), NEVER the headline STAN
>    cell. Until the fixed faithful re-run lands, the STAN cells in Table 3 are **PENDING — cite no STAN number.**

---

## House rules (same board as everyone)
- **seed 0 × 5 folds (n=5)**; gated **stride-1 overlap**, the SAME windowing as the board STL reg ceiling
  (`region_head_<state>_region_5f_50ep_<state>_ovl_stl_reg_s0.json`); **fp32** (set `DISABLE_AMP=1`); leak-clean
  per-fold **train-only** log_T (rebuild if stale — see CLAUDE.md stale-log_T trap); user-disjoint folds.
- **Faithful STAN = STAN's own embeddings learned end-to-end from raw** (`research/baselines/stan/`), NOT fed our
  HGI/Check2HGI embedding. The substrate-bound `stl_hgi` variant (STAN on HGI) is an OPTIONAL labeled ablation only.
- **Train to convergence:** the per-fold best-epoch must land BEFORE the epoch cap (the current run clips at 49/50 —
  under-trained). Raise epochs (≈150-200) or early-stop on a real plateau; verify macro-F1 is not ~0 and Acc@1 is sane.
- Verify healthy best-epochs per fold (not at the cap, not at epoch ≤5); never cite a NaN-collapsed or degenerate fold.
- Commit one result JSON per state + a one-line finding; branch + PR, do not merge to main (orchestrator audits).

## The faithful STAN run (`research/baselines/stan/`)
Faithful STAN is its OWN code — STAN learns its own POI embedding end-to-end from raw and applies the
spatio-temporal interval attention. It is **NOT** the `next_stan` region-head ablation (that is the substrate-bound
`stl_hgi` variant). Runner: `research/baselines/stan/train.py` + `etl.py`.

**Three audit-mandated fixes before any number is reported** (2026-06-26 two-agent audit — `FAITHFUL_STAN_FINDINGS.md`):
1. **Stride-1 overlap windowing.** `etl.py` currently builds NON-overlap stride-9 windows (MIN_HISTORY=5) → ~9x too
   few training windows (data starvation; same class as the CTLE stride defect). Switch the ETL to **stride-1
   overlap, MIN_SEQ=10** to match the board footing. Row-count gate: STAN's windowed rows must match the board
   region-ceiling row counts per state.
2. **Train to convergence.** `--epochs 50` clips the run (best-epochs at 49/50, still climbing). Raise to **≈150-200
   with early-stop on a real Acc@10 plateau**; the chosen best-epoch must land BEFORE the cap. **Seed 0** (the old
   runs are seed 42).
3. **Verify the head is not degenerate.** macro-F1 must not be ~0 and Acc@1 must be sane (the v4 head collapsed to a
   proximity/popularity prior → high Acc@10, ~0 macro-F1). If it is still degenerate after fixes 1-2, apply the
   conditional head fix below.

```bash
export PYTHONPATH=. DATA_ROOT=... OUTPUT_DIR=...
# 1) rebuild ETL at stride-1 / MIN_SEQ=10 (edit the window stride in research/baselines/stan/etl.py)
python -m research.baselines.stan.etl --state <state>            # confirm flags vs the script
# 2) train faithful STAN to convergence, seed 0
python -m research.baselines.stan.train \
    --state <state> --folds 5 --epochs 200 --early-stop --seed 0 \
    --tag FAITHFUL_STAN_OVL_<state>_5f_s0
#  -> docs/results/baselines/faithful_stan_<state>_*_FAITHFUL_STAN_OVL_<state>_5f_s0.json
```
> ⚠ Confirm flags against `research/baselines/stan/train.py --help` (the stride / early-stop flags may need adding).
> The OLD v4 numbers (AL 34.46 / AZ 38.96) are under-trained artifacts — SUPERSEDE them, do not cite.

**Conditional head fix (only if fixes 1-2 leave the head degenerate):** the v4 head is STAN-DERIVED, not faithful —
it replaced STAN's learned `Linear(max_len, 1)` matching collapse with a softmax-weighted mixture and stripped the
residual/LayerNorm. Tighten `model.py` toward the reference STAN (`Linear(M,1)` collapse + restore residual/LN;
reference repo `yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation`, `layers.py`) so we report
STAN, not a crippled variant. **Flag this to the orchestrator first — it is an architecture change, not config.**

## Phase 1 — faithful STAN at board footing (do FIRST)
Run the faithful-STAN spec above for **AL, AZ, FL** (faithful STAN is **infeasible at CA/TX scale** → footnote
"faithful infeasible at scale", like ReHDM; HMT-GRN + Markov carry the CA/TX region-external cells). **Istanbul STAN
is on the M4 — not here.**

**Acceptance (Phase 1):** faithful STAN at seed 0 / stride-1 / **converged** (best-epoch < cap, macro-F1 above floor)
at AL/AZ/FL; CA/TX footnoted infeasible. Commit one JSON per state + a `docs/baselines/next_region/stan.md` row;
mark the old seed-42 / v4 numbers superseded. **If converged faithful STAN STILL lands below the Markov floor at
AL/AZ, THAT is the honest reportable result** (with the next-POI-vs-coarse-region note); if it clears Markov, report
that. Either way the number is trustworthy only AFTER the three fixes.

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
- **Table 3** (`src/tables/tbl3_results.tex`): fill the STAN cells with the **faithful, converged, seed-0 / stride-1**
  values (AL/AZ/FL; CA/TX footnoted infeasible); the STAN footnote states "STAN run faithfully (its own embeddings,
  from raw) on our common protocol (seed 0, stride-1 overlap)"; the ReHDM footnote states "ReHDM under its own
  published protocol (reference)". The substrate-bound `stl_hgi` STAN, if kept, is a clearly-labeled ablation row.
- **`docs/baselines/next_region/comparison.md`** + `stan.md`: add the faithful board-footing STAN row; mark the old
  seed-42 / v4 `FAITHFUL_STAN` numbers superseded-for-the-paper (under-trained, kept only for the v1→v4 audit trail).
- **`RESULTS_BOARD.md §4`** + **`PAPER_PLAN.md §5.4 / §7`**: faithful STAN board-footing done; ReHDM = footnoted reference.
- **Istanbul row of Table 3**: STAN comes from the **M4** run (not this handoff); POI-RGNN/Markov category cells
  from PR #51 are windowing-robust and can go in now (they sit far below our 53.20/59.89 regardless).

## Acceptance checklist
**Phase 1 (faithful STAN, Gowalla):**
- [ ] Faithful STAN (own embeddings, from raw) re-run at **seed 0 / stride-1 overlap / converged (best-epoch < cap)**
  committed for **AL/AZ/FL** (CA/TX footnoted infeasible-at-scale; Istanbul STAN is on the M4, not here).
- [ ] macro-F1 above floor and Acc@1 sane per fold (NOT the v4 collapse); old v4/seed-42 numbers struck.
- [ ] Row-count gate passed (faithful-STAN stride-1 windowed rows == board ceiling row counts) per state.
- [ ] Converged: best-epoch lands BEFORE the cap (the v4 run clipped at 49/50); no NaN/degenerate fold cited.
- [ ] STAN reported on the matched footing; if it still lands below Markov after convergence, that is the honest result.

**Phase 2 (ReHDM, after Phase 1):**
- [ ] ReHDM-faithful AL/AZ/Istanbul run in parallel (Istanbul via the FSQ→mahalle adapter, or footnoted not-available).
- [ ] ReHDM-faithful FL/CA/TX as possible, else footnoted infeasible-at-scale.
- [ ] ReHDM labeled own-protocol reference (never a paired cell); the Istanbul `stl_check2hgi` stop-gap dropped.
- [ ] seed-42 STAN numbers struck from the paper artifacts.
