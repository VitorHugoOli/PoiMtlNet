# HANDOFF — A40 pre-freeze lanes (turn-key) · G0.1 · overlap-validation · canonical-v14

> Machine: **A40 (unmetered CUDA workhorse)**, free after `mtl_frontier` R4–R9. These lanes gate the
> `closing_data` **P2 freeze** under the user's full-scope decision (ADOPT overlapping windows + complete all
> open experiments before writing). **Master plan: [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md)** (read §1–§4b + §8).
> Branch `study/pre-freeze-a40`; do **not** commit to `main`. Honor `CLAUDE.md` (canonical recipe + stale-log_T
> preflight) and C28 (PID-suffixed rundirs, per-run seed echo, no `ls -dt|head`). Each lane closes with a
> verdict + a gate-ledger row in `PRE_FREEZE_PROGRAM.md`, and **STOP for the user** on any recipe-/base-changing
> result. The 4 base-forks are RESOLVED (EXECUTION_PLAN §8); 3 new audit items are flagged there.

## 0 · Do FIRST (blockers + hygiene — before any board cell)

1. **GE (Georgia) v14 — verify-or-build (M0 BLOCKER).** `ls output/check2hgi/georgia/` — **absent on the user's
   local box** (only `data/checkins/Georgia.parquet` exists). On the A40: if a GE v14 exists, **hash-compare** it
   to the canonical anchor; if absent or non-identical, **build it on the SAME fixed machine+seed as the CA/TX
   builds** (Lane 3) — substrate + TIGER tracts + region cardinality (record it next to CA ~8.5k / TX ~6.5k) +
   `poi_to_region` + frozen folds + seeded log_T. **STOP CONDITION: no n=20 board cell launches until all 6
   states have a hash-manifested v14 from the same anchor.**
2. **Rebuild the stale AL `design_k` log_T** (it is older than `next_region.parquet` right now):
   `python scripts/compute_region_transition.py --state alabama --per-fold --seed {0,1,7,100}`.
3. **Centralize the freshness preflight.** Wire the mtime assert (`log_T mtime > next_region.parquet mtime`) into
   every `--per-fold-transition-dir` consumer (`a4_eval.py`, `p1_region_head_ablation.py` lack it). Turn-key check
   before EVERY such run (A40 is Linux → `stat -c`):
   ```bash
   stat -c '%Y %n' output/check2hgi_design_k_resln_mae_l0_1/{state}/region_transition_log_seed{S}_fold*.pt
   stat -c '%Y %n' output/check2hgi_design_k_resln_mae_l0_1/{state}/input/next_region.parquet
   # if any log_T mtime < next_region.parquet mtime → rebuild that seed's log_T first
   ```

## 1 · Lane 1 — G0.1-advisory + the loss-scale advisory (run together at FL+AL, FIRST)

- **G0.1-advisory** (the lone recipe-changing gate): ONE shared permutation for BOTH MTL train loaders — a single
  sampler / joint dataset (same seed on two `DataLoader(shuffle=True)` is **NOT** enough; `src/data/folds.py:1054-1080`).
  Champion **G** recipe EXACTLY (the `NORTH_STAR.md` invocation), v14 substrate, geom_simple. **AL+FL seed 0** vs
  the R0 matched-metric bar. This is **advisory** (current base) — the **binding** G0.1 reruns on the frozen base,
  full {0,1,7,100}, 0.3 pp gate pre-registered (only the binding run re-pins v16→v17). ⚠ aligned-pairing may
  interact with stride-1's denser supervision → the current-base result may not transfer.
- **Loss-scale normalization advisory** (EXECUTION_PLAN §8 #5 — the one "left on the table" lever): divide each CE
  by `log(num_classes)` before the static `cw=0.75` (cat `ln7≈1.95` vs reg `ln~9000≈9.1` ⇒ ~4.7× magnitude gap).
  A few lines in the loss path, no new infra; run it as one extra FL pass alongside G0.1-advisory. **≥0.3 pp either
  head → v17 candidate (STOP for user); null → record EXPLICITLY EXCLUDED in the freeze notes** (mirror the
  composite/routing exclusion wording).

## 2 · Lane 2 — overlap VALIDATION (gate the base change) + the 4-path leak re-audit

ADOPT is the user's decision, but the disciplined way is to **validate at scale FIRST** (it's AL/single-seed/KD-off
only today, with an FL-saturation warning: AL +9.8 but FL +1.3). Reproduce the overlap effect at **FL + one
small/mid state, multi-seed**; estimate the full-base rebuild cost. **If weak at FL-scale, STOP for the user** —
that is itself a finding.

**Leak re-audit checklist — the stride-9 CLEAN verdict does NOT cover stride-1. Confirm all FOUR fold paths
individually (STOP: do not freeze windowing until all four pass), anchored to Luca et al. (ML 2023):**
- [ ] (a) MTL `StratifiedGroupKFold(userid)` — user-disjoint holds under stride-1.
- [ ] (b) STL-NEXT `StratifiedGroupKFold(userid)` — user-disjoint holds under stride-1.
- [ ] (c) **STL-CATEGORY plain `StratifiedKFold`** — the one NON-user-grouped carve-out (`FOLD_LEAKAGE_AUDIT`
  line 108); under stride-1 per-(POI,window) rows can straddle the cat fold boundary — the **most likely to leak**.
- [ ] (d) second-dataset **E2 chronological per-user** stride-1 split — boundary-straddling windows near the
  80/10/10 cut (EXECUTION_PLAN §1a).

## 3 · Lane 3 — canonical-v14 builds + hash manifest (windowing-INDEPENDENT → start now)

- **Build CA + TX v14** (and GE per §0): `scripts/canonical_improvement/regen_emb_t3.py --state {S} --encoder resln`
  + the design_k/mae flags per `CANONICAL_VERSIONS.md §v14`, then
  `scripts/substrate_protocol_cleanup/postbuild_design_substrate.sh check2hgi_design_k_resln_mae_l0_1 {state}`.
  **Pin `--seed` explicitly on every build and use the SAME machine+seed for CA/TX/GE** so the build IS the
  canonical hash anchor (else a second non-identical artifact). Per-state deps first: POI2Vec teacher + Delaunay
  POI-POI graph + region artifacts. **Measure the CA build's first epochs before promising the rest** — TX is ~3×
  FL; if a build won't fit the metered H100 6 h, run it here on the A40 (unmetered).
- **Emit the hash manifest:** sha256 of each `output/check2hgi_design_k_resln_mae_l0_1/{state}/…` artifact, so P3
  + the C1/A2/A4 studies are checked for identity against ONE anchor.
- ⚠ **Do NOT build seeded log_T / sequences / folds yet** beyond the AL fix in §0 — those are windowing-DEPENDENT
  and rebuild after the overlap decision (Lane 2). Building them now = guaranteed throwaway.

## 4 · STOP conditions (hand back to the user)

- G0.1-advisory or the loss-scale advisory fires ≥0.3 pp either head → potential v17.
- Overlap reproduction is weak at FL-scale, or any of the 4 leak-paths fails under stride-1.
- A GE build can't be made identical to the anchor, or any state's v14 hash doesn't match.
- Before **M2/M4**: confirm `scripts/evaluate.py` either has the dual-tower partial-forward override
  (CROSSATTN_PARTIAL_FORWARD_CRASH) or is NOT on the board's eval path (board uses `route_task_best.py` /
  `mtl_cv` internal eval). The C1 reg-metric caveat (plain `top10_acc`/partial forward ≠ board
  `top10_acc_indist`/joint) travels with every C1 panel number.

## 5 · NOT in scope here
P3 (the full board: all states × 4 seeds × 5 folds) is **post-freeze** and not started until P2 commits. The
reading lane (B1–B5 triage + RUN_MATRIX) is driven separately. Baseline *code* implementation is windowing-
independent (it can proceed), but baseline *runs* + seeded log_T + sequences + folds wait for the freeze.
