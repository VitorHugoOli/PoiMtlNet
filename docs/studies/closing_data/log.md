# closing-data — study log (append-only, newest at the bottom)

## 2026-06-12 — Study SCAFFOLDED (not launched)

**Phase**: pre-launch scaffold, created at the close of `mtl_improvement` (design agent + user).

**What exists**
- `AGENT_PROMPT.md` (mission, hard rules C25–C28, read-first) + `PLAN.md` (DRAFT v0 phases:
  P0 pre-freeze gates → P1 cross-study re-eval sweep → P2 recipe FREEZE barrier → P3 CA/TX majors →
  P4 final tables/doc sync).
- Inherited from `mtl_improvement` (CLOSED 2026-06-12, see its `FINAL_SYNTHESIS.md`):
  the **G0.1 aligned-pairing pre-freeze gate** (spec: `docs/results/mtl_improvement/X_SERIES_FINDINGS.md §X1`),
  the **CA/TX majors spec** (mtl_improvement INDEX `#T6-1`), and the scale checks to fold in
  (HSM at 8.5k/6.5k; `next_conv_attn` FL-only cat lever).

**Decision**
- DRAFT only — launch requires user sign-off on `PLAN.md` (incl. the 3 open questions at its foot).

**Next**
- User: review/refine `PLAN.md`; decide launch timing (after remaining studies settle).

---

## 2026-06-12 — RE-SCOPED (pre-launch, user): full experimental base for the NEW paper, not just CA/TX completeness

**Phase**: pre-launch draft refinement (design agent + user).

**What changed (PLAN.md v0 → v1)**
- The study is the **experimental engine for the NEW paper** (story defined in a follow-up effort;
  may not reuse every BRACIS experiment — but the base is regenerated here, story-agnostic).
- Scope expanded from "CA/TX majors" to: **STL baselines RE-RUN + MTL champion + every relevant
  BRACIS-suite experiment, at ALL states × 4 seeds {0,1,7,100} × 5 folds (n=20/cell)** under the
  frozen recipe.
- New **P1b — BRACIS suite inventory → `RUN_MATRIX.md`** (walks `TABLES_FIGURES.md` T1–T5 +
  `RESULTS_TABLE.md §0.1–0.6` + the baselines strategy; per cell: RE-RUN / REUSE / STORY-DEPENDENT
  with exact run specs). The matrix is what Phase 3 executes and what the user signs off with the
  freeze.
- P2 freeze now ALSO pins the **substrate identity** (v14 vs canonical — user decision, drives
  M0/M1 scope) and the n=20 protocol for every cell.
- P3 restructured: M0 missing artifacts (CA/TX builds + 4-seed log_T everywhere + baseline-engine
  artifacts) → M1 STL baselines re-run (ceilings, composite, external engines) → M2 champion all
  states → M3 remaining suite cells → M4 one full-board matched-metric re-score with per-cell
  provenance (the new paper's source of truth).
- Open questions updated (substrate identity; whether the external-baseline T5 set is kept at the
  full n=20 protocol).

**Next**
- User: sign off PLAN.md v1 (4 open questions at its foot); follow-up effort defines the new
  paper's story; launch when the remaining studies are settled.

---

## 2026-06-12 — Pre-launch questions RESOLVED (user) → PLAN v2; machine allocation defined

**Phase**: pre-launch draft refinement (design agent + user).

**Decisions (user)**
1. Studies settled — working assumption YES; P1a verifies (`merge_design` first).
2. **Substrate = v14** or whichever newer blessed base exists at launch (re-check
   `CANONICAL_VERSIONS.md` then). Single-substrate board.
3. **External baselines: RE-RUN under the NEW regime** at the full n=20 protocol — no reuse of
   BRACIS-era lighter-protocol numbers. P1b dispositions per-engine existence/relevance only.
4. No timeline pressure. **Execution split across 3 machines**: H100 (**6 h TOTAL** — the metered
   burst, default = CA/TX v14 builds, job list must be timed on the A40 before spending), A40
   (unmetered workhorse — the run board), M4 Pro 32GB (prep/scoring/small-state lane, MPS caveats
   per `docs/infra/`). Coordination: per-machine manifests merged at M4; C28 rundir rules.

**Remaining launch gate**: user sign-off on `RUN_MATRIX.md` (P1b) together with the P2 freeze.

---

## 2026-06-12 — HANDOFF.md created (the returning-agent entry point)

**Phase**: pre-launch documentation.

**What happened**
- `HANDOFF.md` written: state of play, arrival checklist (incl. the drift check against
  `CANONICAL_VERSIONS.md` at launch), the inherited specs (G/v16, matched-metric scoring, the G0.1
  gate, the P1b suite list), the 5 already-made decisions (do NOT re-ask), the machine allocation
  with the H100 metering rules, the C25–C28 trap list, and the pointer map. Read order is now
  HANDOFF → AGENT_PROMPT → PLAN → log.

---

## 2026-06-17 — C1 (3-snapshot per-task routing) confirm-on-G → PROMOTE (deploy panel)

**Phase**: G0.2 gate (escalated from P1a STORY-DEPENDENT by user 2026-06-16). Completed. Branch
`study/c1-confirm-g`, Mac M2 Pro (MPS, fp32). **STOP for user before closing the gate.**

**What ran**
- Built v14 substrate (`check2hgi_design_k_resln_mae_l0_1`) **locally on MPS** at FL + AL
  (`build_design_k_delaunay.py`, 500 ep); postbuild inputs + seed-tagged per-fold log_T
  (`scripts/closing_data/c1_prep_substrate.sh`).
- Champion **G** (`--canon v16`) with `--save-task-best-snapshots --no-checkpoints`, FL + AL ×
  seeds {0,1,7,100} × 5 folds. Independent re-score of the 3 per-fold snapshots vs the single
  `geom_simple` checkpoint (`route_task_best.py`); paired Wilcoxon (`c1_aggregate.py`).

**Result** (per-task-best routing − single geom_simple checkpoint; reg = Acc@10):
- AL n=20: Δreg **+1.554 pp** (Wilcoxon p=0.0001); Δcat +0.109 pp (n.s.). PROMOTE.
- FL n=20: Δreg **+0.625 pp** (p<0.0001); Δcat +0.187 pp (sig. positive). PROMOTE.
- POOLED n=40: Δreg +1.089 pp, Δcat +0.148 pp. PROMOTE.

**Decision** — **PROMOTE, gate CLOSED → SUPPORTIVE diagnostic panel ONLY, NOT primary deploy**
(user, 2026-06-17). Deploy mode; G weights unchanged; NOT v17. NOT subsumed on G; FL's +0.63 ≪
pre-C25 +2.80 ⇒ most of the old 2/3-state signal was recovered by C25-fix + dual-tower + geom_simple,
a real residual remains. AL (which FAILED pre-C25 on a degenerate Acc@1 snapshot) now passes 5/5
every seed — the v15 Acc@10 reg-monitor fixed that mode.
**User methodological scope:** per-task routing loads 2 snapshots / runs 2 forwards ≈ task-specialised
models → forfeits the "single model for N tasks" property the MTL thesis rests on. So the single
`geom_simple` checkpoint stays the **headline/primary**; C1 is reported as deploy-time per-task
*selection* headroom (single-ckpt ≤ C1 ≤ STL ceiling), NOT the task ceiling, NOT the proposal.
Verdict doc: `C1_VERDICT.md`. Gate ledger + PLAN G0.2 updated.

**Notes / traps hit**
- Under `--no-checkpoints`, snapshots + config land at the **top-level** `results/<engine>/<state>/`
  (not the timestamped rundir) and are **shared across seeds** → driver isolates per seed to
  `c1_snap_s<N>/` + copies the seed-stamped `config.json`.
- `route_task_best.py` must **NOT** be passed `--task-set`: that forces the default-preset heads and
  the dual-tower state_dict fails to load. Omit it → it reconstructs champion-G heads from config.json.

**Next** — C1 CLOSED. **G0.1 aligned-pairing is now the lone open recipe-changing P0 gate** before
the freeze.

---

## How to add an entry

```markdown
## YYYY-MM-DD — Short title

**Phase**: P?.? in flight / completed / paused.
**What ran** / **Result** (numbers, per-state) / **Decision** / **Blocker** / **Next**
```
Rules: append at the bottom; never edit historic entries; if a gate promotes, STOP and record the
user decision before proceeding.

---

## 2026-06-19 — Pre-freeze gates RUN on the A40 (`study/pre-freeze-a40`) — freeze prerequisites MET

**Phase**: P0 pre-freeze gates — completed (the freeze barrier P2 prerequisites are now met).

**What ran / Result**
- **Lane 1** G0.1 aligned-pairing = **ADVISORY NULL** (FL null; AL aligned hurts cat) + loss-scale lever **EXCLUDED** → recipe stays **v16** (no v17).
- **Lane 2** overlapping-windows **ADOPT supported** (AL cat +8.12 / FL cat +3.64); stride-1 leak re-audit **CLOSED CLEAN** (all 4 paths incl. (d) E2-chrono).
- **Lane 3** CA + TX v14 substrates built → **all 6 states hash-manifested** (`V14_HASH_MANIFEST.json`) → **§0 STOP-condition LIFTED**.
- **New production code BYTE-IDENTICAL** (S1 train-metric streaming, S2 chunked val-metric [scored path], dataset-on-GPU auto-fit, `<U32` builder fix): FL non-overlap MTL = **73.0116 / 73.5414** exact.
- **Baselines** — 7 INCLUDE externals implemented + adversarially audited + cheap-fixed + enum-registered; **POI2Vec resolved** (GeoTreeSkipGram honest rename + faithful AAAI'17 build); `--only-fold k` added for P3 per-fold scoring.
- **Speed levers** — `--compile --tf32` **ADOPTED + pinned** for the P3 board (result-neutral +0.05 pp, ~15% faster); torch stays 2.11; workers skipped.

**Decision**: freeze prerequisites met — recipe v16 + byte-identical code; the P3 board re-baselines absolutes by ~0.05 pp (compile pin) only.
**Next (P3, post-freeze)**: n=20 board build (compiled); B1/B2b/faithful-POI2Vec per-fold scoring via `--only-fold`; B2a/GeoTreeSkipGram include/exclude decision; reg-ceiling Acc@10-under-overlap via the T3 matched scorer. Full per-gate docs under `docs/studies/pre_freeze_gates/` + `closing_data/{BASELINES_IMPL_AUDIT,FREEZE_READINESS}.md`.

## 2026-06-20/21 — gated-overlap board prep (Q1/Q2/Q3) + perf
Consolidated findings: `docs/studies/pre_freeze_gates/OVERLAP_BOARD_FINDINGS.md`.
- Board windowing = stride-1 GATED (M1 emit_tail=False) + min_seq=10, enforced as default + train-time guard.
- Host-RAM OOM fixed (lazy per-fold folds + dropped dead FoldData.x; byte-identical). CA/TX overlap MTL fit
  the A40 via auto-fit; NEVER MTL_DATASET_GPU=1 for CA/TX (OOM). TX ~160s/epoch (~11h full).
- Q2 FL board-grade (gated, seed-42): champion-G cat +3.12 (beats ceiling) / reg −1.12 (FLAG — was −0.35
  non-overlap; needs multi-seed; seed-0 compiled run in progress).
- Q3 compile/tf32: quality-neutral; the "32min warmup" was an eager-fallback (cache_size_limit=8). Fixed via
  MTL_COMPILE_DYNAMIC=1 + shared TORCHINDUCTOR_CACHE_DIR + cache_size_limit→64 → ~13-15% faster, 0 warmup on
  cache reuse. Board compiled path: --compile --tf32 MTL_COMPILE_DYNAMIC=1 + shared cache, applied uniformly.
- All code on PR #29 (byte-identical + perf).

## 2026-06-22 — M2 Pro lane: light SC baseline EMBEDDINGS built (PR #30, `study/board-m2pro`)

**Phase**: P3 board prep — the M2 Pro (run on an M4 Pro, 24 GB / MPS) lane of `HANDOFF_BOARD_M2PRO.md`.
Built the light substrate-column baseline embeddings (device-tolerant inputs; matched-head COMPARISON
stays CUDA-only). Full lane log: `M2PRO_BUILD_LOG.md`; built-artifact record: `M2PRO_MANIFEST.md` (226 cells).

**Built on the Mac (✅):** B2c one-hot64 6/6 states · b2b skip-gram 80/80 (AL/AZ/GA/FL) ·
CTLE 60 (AL/AZ/GA) · POI2Vec 80/80 (AL/AZ/GA/FL). All stride-1 gated-overlap, train-only per fold.
**Routed to A40** (>1 h/cell on the Mac, compute-bound): CTLE FL/CA/TX · POI2Vec CA/TX · b2b CA/TX
(A40 lane live, `M2PRO_BUILD_LOG.md §9`).

**3 builder defects/gaps found + fixed** (§2/§7): CTLE had **no `--stride`** → silently built stride-9
non-overlap (12,709 vs 96,326 rows) — the §3d "inherits windowing" assumption was wrong; both CTLE+POI2Vec
did per-batch `.item()` (MPS sync anti-pattern vs canonical `mtl_cv.py:818`) → on-device accumulation;
POI2Vec needs per-cell scratch-staging (the §3c command was incomplete).

**Stride-1 gate = NO-OP confirmed** (§1): `min_seq` 5 vs 10 identical at stride-1 (emit_tail=False geometry
needs ≥10 check-ins anyway; AL both = 96,326). Caveat (§9d): row-*set* identity vs the design-k `check2hgi_dk_ovl`
base is inferred from equal counts, **not** verified — the CUDA comparison hard-tests it.

**3 infra incidents on the 24 GB box** (§7/§10): external SSD dropped off the bus **3×** under write load →
fixed by `--work-dir` (heavy transient I/O to internal); CPU oversubscription at 6 workers (load 22) →
`--threads` cap; **RAM-exhaustion crash** → `--ram-floor-gb` gate + 3 workers. FL POI2Vec finished via
**all-internal execution** (substrate staged internal, build+handoff internal, embeddings moved back to SSD →
zero per-cell SSD I/O). **Finding**: internal storage is a RELIABILITY win, NOT speed, for POI2Vec
(~60 min/cell internal ≈ ~51 min SSD — it's `phi`/CBOW compute-bound) → confirms CTLE/POI2Vec-CA/TX belong on the A40.

**Decision/Next**: Mac lane CONTENT-complete; flagged for audit (handoff §4.4) on PR #30. The multi-GB
embeddings stay gitignored (manifest = the committed record, §4.3). **Orchestrator**: transfer the Mac's
`output/board_baselines/` + `output/baseline_b2c_onehot64/` to the CUDA board for the comparison; the Mac does
not merge/transfer. Open: train.py end-to-end *consumption* of a baseline engine dir is shape-verified only
(plumbing smoke deferred to the orchestrator). Parked (opt-in, user-approved-conditionally): AL-ownership
fold-1 MPS fit check.

---

## 2026-06-24 — A40: CSLSL cascade (role-3 baseline) AL+AZ — cascade ≈ parallel (dead tie)

**Phase**: baselines (`BASELINE_A40.md` role-3). **What ran**: the CSLSL/CatDM **cascade** (`b4_cascade.py`,
directed cat→region, symmetric cross-attn disabled) vs **champion-G**, both on `check2hgi_dk_ovl`, seed 0 × 5f,
gated stride-1 overlap MIN_SEQ=10, **true fp32** (`MTL_DISABLE_AMP=1`, 0 skips), all on the **same A40** (clean
same-device Δ; champion-G re-run alongside each cascade, GPU kept at 2 concurrent runs).

**Result** (cat macro-F1 / reg FULL top10 / joint √(cat·reg)):
- AL cascade 63.45/69.48/66.39 vs champ-G 63.25/69.65/66.37 → **Δjoint +0.02**.
- AZ cascade 63.63/59.18/61.37 vs champ-G 63.44/59.36/61.36 → **Δjoint +0.00**.

→ **Cascade ≈ parallel champion-G — a dead tie** (Δjoint ≤ 0.02 pp ≪ fold-std ~1.3–3.3). Cascade trades a hair
of cat (+0.20) for a hair of reg (−0.17/−0.18). Our parallel bidirectional cross-attention **matches the
dominant published multi-task alternative at equal cost**. Cascade did NOT beat champion-G (wiring sanity OK).
A40 champ-G reproduces the board H100 champ-G (AZ ±0.05; AL within fold-std). **n=5 provisional** (seed 0).

**Prep**: rebuilt AL dk_ovl at MIN_SEQ=10 (was MIN_SEQ=5 + stale log_T), built AZ dk_ovl fresh; engine-aware
seeded per-fold log_T for both. **Env note**: `MTL_STRICT` deliberately OFF (auto-canon v16 → the
v16-pins-v14 substrate guard hard-fails under STRICT on dk_ovl; WARN-only without it, numerically inert).

**Outputs**: `RESULTS_BOARD.md §1b`, `../../results/closing_data/MACS_BOARD_RESULTS.md`, cell
`CSLSL_CASCADE.md`; JSONs `../../results/closing_data/a40/{al,az}_{cascade,champG_a40}_s0.json`.
**Deferred** (deadline; "only if cheap" per handoff): CA/TX cascade.
