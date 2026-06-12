# closing-data — phase plan (DRAFT v1, 2026-06-12 — re-scoped per user)

> **DRAFT — launch requires user sign-off.** Phases strictly ordered: P0/P1 are cheap and can
> interleave; **P2 (FREEZE) is a hard barrier** — nothing in P3/P4 starts before the freeze commit.
>
> **⭐ What this study IS (re-scoped 2026-06-12):** the **experimental engine for the NEW paper**.
> The story/narrative of that paper is defined in a follow-up effort and may not reuse every BRACIS
> experiment — but the *results base* is regenerated HERE, once, under the final frozen recipe:
> **STL baselines re-run + MTL champion + every relevant BRACIS-suite experiment, at ALL states ×
> 4 seeds {0,1,7,100} × 5 folds (n=20/state/cell)**. Closing-data is story-agnostic: it errs on the
> side of regenerating the full base (a cell can be dropped from the paper later; it cannot be
> un-run later without re-opening the heavy spend). It is still NOT an improvement study — any
> promotable lever found en route becomes a user go/no-go gate, never ad-hoc tuning.

---

## Phase 0 — Pre-freeze gates (cheap tests that could still change the recipe)

### G0.1 — Aligned-pairing training test ★ (inherited from mtl_improvement X1 — MANDATORY)
- **Why**: the MTL cross-attn trained on randomly-paired windows for the entire improvement study;
  the roll probe proved the published numbers pairing-safe but is **circular** against "mixing is
  learnable under aligned pairing." A positive changes the recipe → must precede the freeze.
- **Spec**: one shared permutation for both MTL train loaders (single sampler/joint dataset — same
  seed on two `DataLoader(shuffle=True)` is NOT enough; `src/data/folds.py:1054-1080`). G at AL+FL
  seed0 vs the R0 bar. Full spec: `docs/results/mtl_improvement/X_SERIES_FINDINGS.md §X1` banner.
- **Gate**: ≥0.3pp either head → multi-seed → STOP for user (recipe → v17). Null → v16 freezes;
  the "wins without per-sample mixing" wording is fully earned.

### G0.2 — (placeholder) gates added by the Phase-1 harvest
Each needs spec, cost, promote-gate, and user sign-off BEFORE running. Default: empty.

## Phase 1 — Re-evaluation + inventory (reading-heavy, ~no GPU)

### P1a — Cross-study re-evaluation sweep ("did we leave anything on the table?")
Walk every row of `docs/studies/log.md`. Per study: read the closure doc, confirm the closure is
sound (the mtl_improvement audits showed closures can hide dead-codepath nulls and manifest races),
list anything promotable that was parked. Known candidates to adjudicate (pre-freeze gate /
fold into P3 / future-work / drop):
- mtl_improvement INDEX `#T7-FW` mechanisms → default future-work (the new paper's follow-up), NOT here.
- `next_conv_attn` FL-only cat lever and T5.3-HSM-at-8.5k/6.5k → default fold into P3 (cheap).
- `merge_design` / `substrate-protocol-cleanup` / `embedding_eval` parked items — **apply the
  regime-dependence test**: any MTL-null measured under pre-C25 class-weighted CE is
  regime-dependent and may need a re-read (not necessarily a re-run).
- **Output**: `PHASE1_VERDICT.md` — per-study one-liner + the (possibly empty) gate/fold-in lists.

### P1b — BRACIS experiment-suite inventory → the RUN MATRIX ★ (the load-bearing artifact)
The new paper starts from the BRACIS experimental base; the story (follow-up effort) will choose
from it. Walk **`articles/[BRACIS]_Beyond_Cross_Task/TABLES_FIGURES.md` (T1–T5 + figures)** and
**`docs/results/RESULTS_TABLE.md §0.1–§0.6`**, plus `docs/PAPER_BASELINES_STRATEGY.md`. Per
table/figure/cell, record in **`RUN_MATRIX.md`**:
- what runs back it today (engine, states, seeds, folds, recipe/version, harness);
- disposition under the final recipe: **RE-RUN** (numbers change with recipe/substrate — the
  default for anything model-derived) / **REUSE** (recipe-independent, e.g. T1 dataset statistics)
  / **STORY-DEPENDENT** (park, decide with the new paper's story — e.g. cut-to-prose items);
- exact run spec for every RE-RUN cell (command, engine, states × seeds × folds, scorer) and its
  prerequisite artifacts (substrate builds, seeded log_T, frozen folds).
Known suite members to inventory: **T2** substrate ablation (two-panel), **T3/§0.1** MTL-vs-STL
both tasks (already n=20 protocol — re-runs under the final recipe), **T4/§0.2** Δm joint score,
**T5/§0.5–0.6** external baselines per state (DGI/HGI/HMRM/Time2Vec/… — decide per-engine whether
the new paper keeps them and at what protocol), **§0.3** substrate axis, **§0.4** recipe selection.
**Output**: `RUN_MATRIX.md` = the definitive run ledger P3 executes. User signs it off with the freeze.

## Phase 2 — RECIPE + PROTOCOL FREEZE (hard barrier — one commit)

- **Recipe**: v16 (champion G) or v17 if a P0 gate promoted — pinned in
  `docs/results/CANONICAL_VERSIONS.md` + `NORTH_STAR.md`.
- **Substrate identity** (user decision — paper-level): the new paper's numbers live on ONE declared
  substrate (v14 `check2hgi_design_k_resln_mae_l0_1` vs canonical `check2hgi`); mixed tables only if
  the story explicitly compares substrates (T2-style).
- **Protocol**: ALL states × seeds **{0,1,7,100}** × **5 folds** (n=20) for every cell; user-disjoint
  frozen folds; matched-metric scoring (`r0_matched_rescore.py` method — FULL `top10_acc`, fp32-eval
  parity); checkpoint selector `geom_simple`; per-state recipe variants decision (ONE recipe, or a
  documented small/large-state split — the B9/H3-alt history says this can be scale-conditional).
- Everything after this commit cites it.

## Phase 3 — FULL EXPERIMENTAL-BASE REGENERATION (the single heavy spend; executes `RUN_MATRIX.md`)

### M0 — Missing artifacts
v14 builds at **CA (8.5k) + TX (6.5k)** (`scripts/canonical_improvement/regen_emb_t3.py`, design_k
per CANONICAL_VERSIONS §v14) + **seeded per-fold log_T for ALL FOUR reporting seeds** at every state
(`compute_region_transition.py --per-fold --seed S`) + frozen folds; freshness preflight per
CLAUDE.md (stale log_T silently inflates reg). Any baseline-engine artifact the RUN_MATRIX needs at
states that lack it (e.g. HGI/DGI embeddings at CA/TX). **Measure the first build's wall-time before
promising the rest** — builds dominate the budget.

### M1 — STL baselines re-run (the comparand side)
At ALL states × {0,1,7,100} × 5f under the frozen protocol: per-task **STL ceilings** (cat +
reg, p1 harness on the frozen substrate); the **(d) composite** (incl. its HGI-reg component if the
matrix keeps it); the **external baseline engines** (T5 set) per the RUN_MATRIX disposition.

### M2 — MTL champion
Frozen-recipe **G at ALL states × {0,1,7,100} × 5f**. Recorded prediction to test at CA/TX: the C25
margins are LARGEST there (the confound scaled with class count).

### M3 — Remaining BRACIS-suite cells
T2 substrate-ablation panels, T4/§0.2 Δm cells, §0.4 recipe cells, and the folded-in scale checks
(HSM-vs-flat at 8.5k/6.5k STL-level; `next_conv_attn` if P1a promoted it) — exactly as dispositioned
in `RUN_MATRIX.md`. STORY-DEPENDENT cells stay parked unless the user pulls them in.

### M4 — Full-board re-score
ONE scoring pass over everything (matched metric, single script, committed JSON + markdown):
champion vs STL ceilings vs composite vs baselines, all states, n=20 — **the new paper's source of
truth**. Per-cell provenance (rundir, seed, fold, commit) embedded; C28 manifest rules enforced
(PID-suffixed rundirs, per-run seed echo, no `ls -dt|head`).

## Phase 4 — Results base hand-off + doc sync

- Publish the base: a dedicated results section/doc (e.g. `docs/results/closing_data/FINAL_BOARD.md`
  + JSONs) — versioned, story-agnostic, citable per-cell.
- Sync `CANONICAL_VERSIONS.md` (freeze pin), `docs/studies/log.md` (closure row), `CHANGELOG.md`.
- **Hand-off to the new-paper story effort** (the follow-up prompt/study): the board + the
  RUN_MATRIX dispositions + the STORY-DEPENDENT parked list. §0.1/BRACIS-doc restatements remain
  author-decision items (don't silently rewrite paper-canon tables — PAPER_UPDATE rule).
- Closure: `FINAL_SYNTHESIS.md` for this study; archive per `docs/studies/README.md` policy.

---

## Budget sketch (refine after P1b + the first CA build)
P0–P1: <1 GPU-h + reading. P3: dominated by (a) the CA/TX v14 builds (multi-day class — measure
first) and (b) the full board ≈ [#engines-kept × 6 states + G + ceilings + composite] × 4 seeds × 5f
— at A40 timing (~14 min/seed-FL-MTL, small states ~2-3 min) the champion+ceiling board is hours,
not days; the baseline-engine re-runs scale with how much of T5 the matrix keeps. P4: ~0.

## Open questions for the user before launch
1. Are all improvement studies the user wants closed actually closed (is `merge_design`
   ACTIVE-CLOSING settled)?
2. **Substrate identity for the new paper** — v14 everywhere (the better base, breaks continuity
   with BRACIS-v11) vs canonical (continuity, weaker)? Drives M0/M1 scope.
3. **External baselines (T5 set)** — keep for the new paper? If yes, at the full n=20 protocol
   (BRACIS ran them lighter) → sizeable M1 addition; per-engine keep/drop happens in P1b.
4. Timeline pressure (camera-ready / thesis dates) that constrains P3 ordering?
