# closing-data — phase plan (DRAFT v0, 2026-06-12)

> **DRAFT.** Scaffolded at `mtl_improvement` close; refine + get user sign-off before launching.
> Phases are strictly ordered: 0 and 1 are cheap and can interleave; 2 (FREEZE) is a hard barrier —
> nothing in 3/4 starts before the freeze is committed.

---

## Phase 0 — Pre-freeze gates (cheap tests that could still change the recipe)

### G0.1 — Aligned-pairing training test ★ (inherited from mtl_improvement X1 — MANDATORY)
- **Why**: the MTL cross-attn trained on randomly-paired windows for the entire improvement study
  (two independent shuffled loaders); the roll probe proved the published numbers pairing-safe but
  is **circular** against the counterfactual "mixing is learnable under aligned pairing." If aligned
  training lifts either head, the final recipe changes — so this MUST run before the freeze.
- **Spec**: make both MTL train loaders consume ONE shared permutation (single sampler/joint
  dataset — passing the same seed to two `DataLoader(shuffle=True)` is NOT enough, they draw
  sequentially from the global RNG; `src/data/folds.py:1054-1080`). Run G at AL+FL seed0 vs the R0
  bar. Full spec + the circularity analysis: `docs/results/mtl_improvement/X_SERIES_FINDINGS.md §X1`
  correction banner.
- **Gate**: ≥0.3pp on either head → multi-seed {0,1,7,100} → STOP for user (recipe change → v17).
  Null → v16 freezes as-is; the "wins without per-sample mixing" paper wording is then fully earned.

### G0.2 — (placeholder) gates added by the Phase-1 harvest
- Phase 1 may surface parked levers worth a cheap pre-freeze check. Each needs: spec, cost,
  promote-gate, and user sign-off BEFORE running. Default: empty.

## Phase 1 — Cross-study re-evaluation sweep ("did we leave anything on the table?")

Walk every row of `docs/studies/log.md` (the cross-study outcomes registry). Per study: read its
closure doc, confirm the closure is sound (the mtl_improvement audits showed closures can hide
dead-codepath nulls and manifest races), and list anything promotable that was parked. Known
harvest candidates to adjudicate (decide: pre-freeze gate / fold into Phase 3 / future-work / drop):
- `mtl_improvement` INDEX `#T7-FW` mechanisms (conditional coupling, cat-conditioned prior,
  semantic-ID region factorization, cat-transition feature, consistency loss) — default: future-work
  (next paper), NOT this study.
- `next_conv_attn` FL-only cat lever (+1.06 FL, craters small states) — scale-conditional; decide
  whether Phase 3 tests it at CA/TX.
- T5.3 HSM null (tested at FL 4.7k only) — fold the 8.5k/6.5k check into Phase 3 (cheap, STL-level).
- `merge_design` / `substrate-protocol-cleanup` / `embedding_eval` parked items — verify their
  closure docs list nothing promotable under the post-C25 regime (their MTL verdicts predate the
  C25 fix — check whether any MTL-null was measured under class-weighted CE and is therefore
  regime-dependent; the regime-dependence re-open test from mtl_improvement applies).
- **Output**: `PHASE1_VERDICT.md` — per-study one-liner + the (possibly empty) gate/fold-in lists.

## Phase 2 — RECIPE FREEZE (hard barrier)

- Pin the final recipe as the frozen canon (v16, or v17 if G0.x promoted) in
  `docs/results/CANONICAL_VERSIONS.md` + `NORTH_STAR.md`.
- Freeze the protocol: seeds {0,1,7,100}, 5-fold user-disjoint splits, matched-metric scoring
  (`r0_matched_rescore.py` method), per-state recipe variants if any survive (B9/H3-alt history
  says recipes can be scale-conditional — decide ONE recipe or a documented split).
- Commit the freeze as its own commit; everything after cites it.

## Phase 3 — The majors (the single heavy spend)

### M1 — CA/TX substrate builds
v14 `check2hgi_design_k_resln_mae_l0_1` at CA (8.5k regions) + TX (6.5k) via
`scripts/canonical_improvement/regen_emb_t3.py` (design_k recipe per CANONICAL_VERSIONS §v14) +
seeded per-fold log_T (`compute_region_transition.py --per-fold --seed S` for every reporting seed)
+ frozen folds. **Measure build wall-time from the first build before promising the second.**

### M2 — Champion + ceilings at CA/TX
Frozen-recipe G + the (c) STL ceilings (p1 harness, cat + reg) at {0,1,7,100}, 5-fold if budget
allows (1-fold = directional only, flag it). Score on the matched-metric bar. The recorded
prediction to test: **C25 margins are LARGEST at CA/TX** (the confound scaled with class count).
Folded-in scale checks (from Phase 1 defaults): HSM-vs-flat at 8.5k/6.5k (STL-level, cheap);
`next_conv_attn` cat head IF Phase 1 promoted it.

### M3 — Full-board re-score
One final 6-state matched-metric table (AL/AZ/GE/FL/CA/TX): G vs (c) ceilings vs (d) composite,
multi-seed, single scoring script, committed JSON + markdown.

## Phase 4 — Final tables + canonical-doc sync

- `docs/results/RESULTS_TABLE.md` final-state section (coordinate with the author's T6.2 paper-canon
  restatement — do NOT silently rewrite §0.1, it needs author sign-off per PAPER_UPDATE).
- Sync `NORTH_STAR.md`, `CLAIMS_AND_HYPOTHESES.md`, `CHANGELOG.md`, `docs/studies/log.md` (closure
  row), and the BRACIS folder pointers.
- Closure: `FINAL_SYNTHESIS.md` for this study; archive per `docs/studies/README.md` policy.

---

## Budget sketch (refine after the first CA build)
Phase 0–1: < 1 GPU-h + reading. Phase 3: dominated by the two v14 builds (multi-day class — measure
first) + ~16–40 champion/ceiling runs at large-state cost (~15–30 min each on A40). Phase 4: ~0.

## Open questions for the user before launch
1. Are all improvement studies the user wants closed actually closed (is `merge_design`
   ACTIVE-CLOSING settled)?
2. 5-fold vs 1-fold at CA/TX (compute vs paper-grade claims)?
3. Does the BRACIS camera-ready timeline need Phase 3 before a specific date?
