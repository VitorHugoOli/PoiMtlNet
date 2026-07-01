# M0 / P3 Execution Plan — the ready-the-moment-it-freezes regeneration plan

> Created 2026-06-16. Operationalizes `PLAN.md` Phase 3 (M0–M4) into a concrete artifact-inventory +
> machine-routed sequence. **P3 is post-freeze and is NOT running now** (user, 2026-06-16) — this doc is
> the staged plan so the heavy spend executes cleanly the moment P2 freezes.
>
> **Reads:** `PLAN.md` (phase defs), `PHASE1_VERDICT.md §3` (the RUN_MATRIX RE-RUN/REUSE/STORY-DEPENDENT
> dispositions P3 executes), `PRE_FREEZE_PROGRAM.md` (gate ledger + machine allocation),
> `docs/results/CANONICAL_VERSIONS.md §v14` (the substrate build recipe).

## 0 · Gate before P3 can start

P3 cannot begin until **P2 freezes**, which cannot commit with any G0.2 ledger row open: G0.1
(aligned-pairing), **C1** (3-snapshot routing, escalated), A2/A4, the overlapping-windows decision,
the baseline triage, and the `mtl_frontier` R-levers. **Exception:** M0a (substrate builds) is
substrate-identity work and can **pre-stage now** (§2) — no open gate changes the v14 substrate identity.

## 1 · The board

6 states × seeds {0,1,7,100} × 5 folds = **n=20 / cell**, under the frozen recipe (v16/champion-G +
v14 substrate + `geom_simple` selector + unweighted CE). States: **AL, AZ, FL, CA, TX, GE**. Cells
RE-RUN / REUSE / STORY-DEPENDENT per `PHASE1_VERDICT.md §3` (T3/§0.1 = the inverted central table =
highest-priority RE-RUN; T1 = REUSE; substrate panels = STORY-DEPENDENT on the single-v14 board).

## 2 · M0 — prerequisite artifacts (current inventory + actions)

On-disk audit (2026-06-16, this M2 box — **verify per machine; artifacts are scattered across A40/Mac**):

| State | v14 substrate | seeded log_T {0,1,7,100} | M0 action | Machine |
|---|---|---|---|---|
| **AL** / **AZ** | ✅ | ✅ (×5 folds, incl. seed 42) | none — complete | — |
| **FL** | ✅ | ❌ only seed-42 on this box | build {0,1,7,100}×5 seeded log_T (the multi-seed runs lived on the A40 — **find + consolidate** before rebuilding) | M2/M4 MPS or A40 |
| **CA** / **TX** | ❌ (only the old GCN substrate present) | ❌ | **build v14 substrate** + seeded log_T | **H100** (build) → priors anywhere |
| **GE** | ❌ (absent here) | ❌ | mtl_improvement validated champion G at GE → its v14 **likely exists on the A40**: **verify + sync first**, build only if genuinely absent | A40 (verify/sync) or MPS (build if small) |

**Substrate build recipe (pin exactly — `CANONICAL_VERSIONS.md §v14`):** v14 =
`check2hgi_design_k_resln_mae_l0_1` = ResLN encoder + HGI Delaunay POI-POI GCN reg lever (design_k, λ=0.1)
+ masked-POI mae cat lever. Build via `scripts/canonical_improvement/regen_emb_t3.py --state {State}
--encoder resln` + the design_k/mae flags, then `scripts/substrate_protocol_cleanup/postbuild_design_substrate.sh
check2hgi_design_k_resln_mae_l0_1 {state}`. **Per-state deps:** the POI2Vec teacher + Delaunay POI-POI
graph + region artifacts (`poi_to_region`, region adjacency) must exist for CA/TX/GE first.

**Sequencing — the one subtlety that decides what pre-stages now:**
- **M0a (substrate builds) — windowing-INDEPENDENT** (the check2hgi embedding is per-check-in; windowing
  is applied downstream when forming sequences). ⇒ **safe to pre-stage NOW**, in parallel with the
  pre-freeze gates. CA/TX → H100; GE → verify/sync from A40.
- **M0b (seeded per-fold log_T) — windowing-DEPENDENT** (built from train-split *sequences*). ⇒ **wait
  for the overlapping-windows gate** to settle; if overlap is adopted, the sequences (and thus every
  log_T) rebuild. Building log_T before that decision risks throwing it away.

**Validation (every built/synced artifact):** pin the build command + `--canon`; spot-check downstream
STL metrics are sane vs the AL/AZ/FL reference; **stale-log_T freshness preflight** (`CLAUDE.md`: log_T
mtime > `next_region.parquet` mtime) before any `--per-fold-transition-dir` run. A substrate rebuilt on a
different machine must be confirmed to match the canonical recipe (it is the *frozen input* — non-identity
breaks the whole board's comparability).

## 3 · P3 sequence (post-freeze; executes `RUN_MATRIX.md`)

| Step | What | Scope | Machine |
|---|---|---|---|
| **M0** | finish prerequisites (§2): CA/TX builds + GE sync + all seeded log_T | 6 states | H100 (CA/TX build) + MPS/A40 (priors) |
| **M1** | STL baselines (the comparand ceilings), per-task | 6 × 20 | A40 (AL/AZ overflow → MPS) |
| **M2** | MTL champion G | 6 × 20 | A40 (CA/TX too — A40 unmetered handles training; H100 was only the metered *build*) |
| **M3** | RE-RUN suite cells (T3/§0.1 first) + chosen external baselines (B1–B5 + STAN/ReHDM/…), on the frozen base, **mirroring the adopted windowing** | per RUN_MATRIX | A40 (+ MPS small-state overflow) |
| **M4** | full-board re-score / aggregate → final tables (geom_simple selector, gap-to-ceiling framing, bridging metrics); redraw F-arch for the dual-tower | all cells | **MPS** (scoring/aggregation — no training) |
| **L4** | external-validity: `second_dataset` Phase V — champion G + STL ceilings + Markov-1 on NYC/Istanbul (within-user CV) **and on the E2 chrono split = the A5 bridge** (train on chrono-train, select on val, report on test; gap-to-ceiling per F2) | 2 cities, 4 seeds | CUDA box (A40 / H100 spare) — may overlap the P3 tail |

## 4 · Machine allocation (P3)

| Machine | Metered | P3 role |
|---|---|---|
| **H100** | 6 h metered | **CA/TX v14 substrate builds (M0a) ONLY.** Get in, build, validate, get out. Do NOT also run CA/TX champion/STL training here — that is the A40's job. ⚠ 6 h is workable for CA+TX builds but leaves little margin; if a build needs a rerun, request more time. |
| **A40** | unmetered | the training board — M1 + M2 + M3 at all 6 states × n=20 (the bulk, ~days). Frees up after `mtl_frontier` R4–R9. GE v14 verify/sync here. |
| **M2 / M4 Pro** | local MPS | M0b seeded-log_T builds (post-windowing-decision), M4 scoring/aggregation, doc settling, and small-state (AL/AZ) STL/baseline overflow only if the A40 saturates (MPS caveats: fp32, no AMP, slower). |

## 5 · Provenance / leak discipline (C28 — non-negotiable on every RE-RUN cell)

PID-suffixed rundirs + per-run seed echo; stale-log_T freshness preflight before every
`--per-fold-transition-dir` run; matched metric / seeds / folds / precision on both sides of every
comparison; pin `--canon` in every driver; the v14 substrate must be byte-/recipe-identical across
machines. End-to-end baselines (M3) must use the **adopted windowing** (the comparability rule).

## 6 · Critical path

1. **NOW (pre-freeze, parallel with gates):** M0a — CA/TX v14 builds (H100) + GE verify/sync (A40) +
   CA/TX/GE POI2Vec/Delaunay/region deps. This removes the longest pole from the post-freeze path.
2. **At freeze:** RUN_MATRIX signed off (P1b) + all G0.2 gates closed.
3. **Post-freeze:** M0b seeded log_T (once the windowing decision is final) → M1 → M2 → M3 on the A40
   (the multi-day bulk) → M4 scoring (MPS) → L4 external validation.

**Bottom line:** the only genuinely-missing heavy artifacts are the **CA/TX v14 substrate builds** (H100,
pre-stageable now) and the **multi-seed seeded log_T** (FL definitely; built once the windowing decision
lands). GE is most likely a sync, not a build. Everything else is A40 training time the freeze gates.
