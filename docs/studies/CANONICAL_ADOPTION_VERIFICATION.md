# Canonical Adoption Verification — `STUDIES_IMPROVEMENTS_SUMMARY.md` audit

**Date:** 2026-05-29
**Auditor:** verification/audit agent (read-only against docs + `src/` + `scripts/`; numbers cross-checked, not invented).
**Input under audit:** [`docs/studies/STUDIES_IMPROVEMENTS_SUMMARY.md`](STUDIES_IMPROVEMENTS_SUMMARY.md)
**Question:** are the promoted study improvements actually incorporated as canonical practice (docs + code), have they been combined and tested against the previous canonical baseline, and can the updated implementation become the new canonical baseline yet?

**Bottom line up front:** the **two v12 defaults (log_T-KD, ResLN) are fully and consistently adopted in BOTH docs and code, with passing tests** — the single most important verification objective passes cleanly. BUT the updated implementation is **NOT yet a valid replacement for the v11 paper baseline**, because (a) log_T-KD is paper-grade only at AL/AZ — FL/CA/TX are seed=42 pilots; (b) the improvements have **never been run stacked**, neither with each other nor with the §4.2 composite, nor re-tabled into §0; (c) ResLN gives **no MTL benefit at all** (so it does not move the headline reg/cat MTL numbers); and (d) the dominant MTL reg gap (−7 to −17 pp) is untouched and explicitly owned by the still-empty `mtl_improvement` study. The correct framing — which the docs themselves adopt — is **v11 stays paper-canon; v12 = v11 + a validated small-state reg lift + an STL-only encoder**, not a wholesale new baseline.

---

## Verdict table

| # | Improvement | Promoted? | Docs adoption (file:line) | Code adoption (file:line) | Combined-tested? | Regression risk | Verdict as NEW CANONICAL |
|---|---|---|---|---|---|---|---|
| 1 | **log_T-KD W=0.2/τ=1.0** (reg-loss KL term, v12 default, MTL `check2hgi_next_region`) | YES — paper-grade AL/AZ | ADOPTED: `CANONICAL_VERSIONS.md:98-167`, `RESULTS_TABLE.md:201-228`, `NORTH_STAR.md:1`, `CLAIMS_AND_HYPOTHESES.md:6` (CH26), `CHANGELOG.md:33-52`, `CLAUDE.md:164-177` | ADOPTED (default wired): `scripts/train.py:1334-1372` (`_V12_LOG_T_KD_DEFAULT_W=0.2`, scoped ON); loss term `src/training/runners/mtl_cv.py:478-540` | NO — only paired W=0.0 vs W=0.2 in isolation; never stacked w/ ResLN/composite/champion | Low (cat flat AL −0.20 / AZ +0.08; leak-audited) | **READY at AL/AZ; NOT READY at FL/CA/TX** (pilot only) |
| 2 | **ResLN encoder** (`ResidualLNEncoder`, v12 future-build default) | PROMOTED (STL cat only) | ADOPTED: `CANONICAL_VERSIONS.md:104-116,176`, `RESULTS_TABLE.md:241`, `NORTH_STAR.md:1`, `CLAIMS_AND_HYPOTHESES.md:9` (CH29), `CLAUDE.md:177` | ADOPTED (default wired): `research/embeddings/check2hgi/check2hgi.py:277` (`getattr(args,'encoder','resln')`), `scripts/canonical_improvement/regen_emb_t3.py:59` (`default="resln"`) | NO — STL × design only; never combined into a shipped MTL run; frozen GCN substrate NOT rebuilt | None to MTL (explicitly NO MTL benefit) | **NOT a baseline change** — STL/generality only; correctly opt-in for MTL |
| 3 | log_T-KD additive-on-B9 + substrate-robust confirm | CONFIRMED (combined KD×substrate, FL pilot) | ADOPTED narrative: `CLOSURE.md` (KD additive +2.40 p=0.031; design_b+KD≈canonical Δ−0.03) | n/a (uses same default as #1) | YES (KD × substrate) but FL seed=42 only | Low | READY-as-pilot only (seed=42) |
| 4 | v3c AdamW WD=5e-2 (embedding trainer) | PROMOTED-provisional (absorbed by ResLN) | ADOPTED note: `canonical_improvement_coverage_audit.md`; CH-trail | Opt-in `--weight-decay 0.05`; statistically equivalent to omitting in stack | Combined-in-stack (with ResLN) | None | NOT a standalone baseline change (absorbed) |
| 5 | C4 stale-log_T mtime preflight guard | LANDED (defensive) | `CLOSURE.md:19,38` | ADOPTED: `src/training/runners/mtl_cv.py:1034-1045` (C22 guard) | n/a (code-only) | None | READY (defensive only) |
| 6 | C1 per-task 3-snapshot routing (variant A) | NOT promoted (§discussion footnote) | `CLOSURE.md:66`, `phase_c_verdict.md`, `STUDIES_…:29` | Prototype only: `--save-task-best-snapshots` `scripts/train.py:1387-1391`; `scripts/route_task_best.py` (+modality persist `:270-279`) | NO; one-state-pass (AZ/FL ✓, AL fail on degenerate fold) | High (Acc@1-selector pathology; AL −7.89) | **NOT READY** — needs Acc@10 selector + degenerate guard + multi-seed |
| 7 | Tier B substrate swaps (Design B/J/M/L, Lever 4/5) under MTL | FALSIFIED/NULL | `RESULTS_TABLE.md:234-243` (§0.9 regime), `CLOSURE.md`, `CLAIMS:8` (CH28) | n/a (no promotion) | Combined w/ MTL+F1 fronts | n/a | Correctly NOT adopted |
| 8 | C2 freeze-reg curriculum / C3 zero-cat-KV | FALSIFIED (close §4.4/P4) | `CLOSURE.md:65`, `phase_c_verdict.md` | `--reg-freeze-at-epoch`, `--zero-cat-kv` opt-in (`scripts/train.py:1300-1317`) | Isolated | n/a | Correctly NOT adopted |
| 9 | §4.2 composite (two-substrate deploy routing) — project reg headline +7–12 pp | CONFIRMED (predecessor study) | `NORTH_STAR.md:1`, `RESULTS_TABLE.md` §0 trail, `future_works/composite_two_substrate_engine.md` | NOT a single-model default; deploy-time two-model routing, not in `train.py` | NOT combined with log_T-KD | n/a (deploy footprint) | Separate deploy artefact; NOT folded into the canonical single-model recipe |
| 10 | merge_design STL designs (B/H/I/J/M) | CONFIRMED STL Pareto, ≤HGI | `STATE.md`; `RESULTS_TABLE.md:241` (opt-in research variants) | Registered/opt-in only | Isolated per design | n/a | Correctly NOT in ship recipe |

**Legend:** "ADOPTED" = present and self-consistent across the named canonical doc(s)/code with file:line; "Combined-tested?" = was it executed *stacked* with the other promoted levers and re-evaluated against the previous canonical baseline.

---

## Objective 1 — Canonical adoption in docs

**Verdict: ADOPTED and internally consistent across all five required docs for the two real promotions (log_T-KD, ResLN). No contradictions or stale text found that would mislead a future agent.** The docs go further than the summary implies: a dedicated version registry now exists.

- **`docs/results/CANONICAL_VERSIONS.md`** (10 KB, dated 2026-05-30) pins **v11 = paper canon (FROZEN, GCN substrate, log_T-KD OFF)** and **v12 = v11 + log_T-KD W=0.2 + ResLN**, with an explicit reproduction map (`CANONICAL_VERSIONS.md:86-96, 158-167, 170-180`). This is the single most important adoption artefact and it is excellent: it states the on-disk substrate is still the frozen v11 GCN (`:118-132`), so a v12-default run is really "v11 substrate + log_T-KD."
- **`docs/NORTH_STAR.md:1`** — top banner documents the v11→v12 flip, the regime finding, and that ResLN has **NO MTL benefit**. Consistent with CANONICAL_VERSIONS.
- **`docs/results/RESULTS_TABLE.md`** — §0.1 explicitly **labelled v11** (`:68`, "log_T-KD OFF, GCN substrate"); §0.8 (`:201-228`) isolates the log_T-KD lift and is labelled **v12 DEFAULT** with the paper-grade-AL/AZ / pilot-FL/CA/TX grade; §0.9 holds the regime/ResLN rows (`:234-243`). Cross-references to CANONICAL_VERSIONS are present.
- **`docs/CLAIMS_AND_HYPOTHESES.md:6,8,9`** — CH26 (log_T-KD, "PROMOTED multi-seed n=20 small states; large-state pilot; v12 default; **still a study-section claim, NOT on the paper whitelist — promotion waits on the §0 re-run**"), CH28 (regime bottleneck), CH29 (ResLN STL-only). The whitelist gating is correctly conservative.
- **`docs/CHANGELOG.md:33-52`** — dated 2026-05-30 v12-flip row with mechanism, grade, tests, and "Docs settled" list; 2026-05-29 study-closure row.
- **Root `CLAUDE.md:164-177`** — adds a "CANONICAL VERSIONS" banner and two new "v12 defaults flipped" gotchas (#4 log_T-KD, #5 ResLN), with the v11-reproduction flags.

**One citation-shorthand caveat (not an error):** the summary cites verdict files as e.g. `tier_a1/phase_a1_verdict.md`. These do **not** live under `docs/studies/substrate-protocol-cleanup/` (that dir holds only `CLOSURE.md`, `INDEX.md`, `log.md`, `window_mask_audit.md`, `canonical_improvement_coverage_audit.md`, `considerations.md`, `AGENT_PROMPT.md`). They live under **`docs/results/substrate_protocol_cleanup/<tier>/`** (verified present: `tier_a1/phase_a1_verdict.md`, `tier_a1_largestate/phase_a1_largestate_addendum.md`, `tier_b/phase_b_two_front.md`, `tier_b_fl/phase_b_fl_3way.md`, `tier_b_fl/hgi_mtl_fl.md`, `tier_c/phase_c_verdict.md`, `tier_resln/phase_resln_verdict.md`, …). The summary's paths are valid relative to the results dir; a future agent should resolve them there, not under `studies/`.

---

## Objective 2 — Canonical adoption in code

**Verdict: ADOPTED. Both v12 defaults are actually wired as defaults in code, scoped correctly, and covered by passing tests.**

- **log_T-KD default (the new CLI default):** `scripts/train.py:1334-1372`. `_V12_LOG_T_KD_DEFAULT_W = 0.2`, `_V12_LOG_T_KD_DEFAULT_TAU = 1.0`. The default is applied at the **CLI layer only**, gated on `config.task_type == "mtl" and task_set == CHECK2HGI_NEXT_REGION.name` (`:1337-1340`), and only when the user did **not** pass `--log-t-kd-weight` (`:1363-1372`). An explicit value (including 0.0 for v11) is honoured verbatim (`:1341-1362`). The `ExperimentConfig` dataclass field default **stays 0.0** (task-agnostic) — confirmed by `tests/...:666-681 test_config_field_default_zero`. This matches the docs exactly.
- **log_T-KD loss term (the mechanism):** `src/training/runners/mtl_cv.py:478-540` — KL of `softmax(reg_logits/τ)` against the per-fold train-only `log_T[last_region_idx]`, τ²-scaled, added to `task_b_loss` with weight `log_t_kd_weight`; threaded from config at `:1350-1351`. Padding rows excluded (verified by `test_padding_rows_excluded`).
- **ResLN encoder default:** `research/embeddings/check2hgi/check2hgi.py:277` (`_encoder_name = getattr(args,'encoder','resln')`) and `scripts/canonical_improvement/regen_emb_t3.py:59` (`ap.add_argument("--encoder", default="resln")`). Future builds default to ResLN; frozen on-disk substrate untouched. Consistent with docs.
- **C4 stale-log_T mtime guard:** `src/training/runners/mtl_cv.py:1034-1045` — refuses to start if log_T mtime predates the next_region parquet. Landed as claimed.
- **C1 routing prototype + the scoring-bug fix:** `scripts/train.py:1387-1391` (`--save-task-best-snapshots`); `scripts/route_task_best.py` persists/reads `task_{a,b}_input_type` (`:270-279`), which is the modality-persistence fix the summary describes. Correctly opt-in, NOT a default.
- **C2/C3 levers:** `--reg-freeze-at-epoch` and `--zero-cat-kv` are opt-in flags (`scripts/train.py:1300-1317`), matching their FALSIFIED status (not defaulted).

**Tests:** `tests/test_substrate_protocol_cleanup_flags.py` — `TestLogTKDCLIDefault` has the claimed **5 cases** (`:710,718,...` — `test_v12_default_on_for_check2hgi_region_mtl`, `test_explicit_zero_recovers_v11`, `test_explicit_weight_wins`, `test_no_default_for_legacy_task_set`, `test_no_default_for_category_task`). **Ran the file: 23 passed** (`.venv/bin/python -m pytest`). The KD-term math tests (`TestLogTKD`, `:585-680`) also pass.

---

## Objective 3 — Combined (stacked) execution check

**Verdict: the improvements have NOT been run together as a stacked baseline, and the docs are honest about this.** Each promotion was validated in isolation (or against its own one-lever paired baseline).

Explicitly untested combinations (each verified absent from `docs/results/`):

1. **log_T-KD × ResLN** — never stacked. log_T-KD was validated on the **GCN (v11) substrate**; ResLN is STL-only and gives no MTL benefit, so the stack was deemed pointless for MTL reg — but it was therefore **never executed or measured**. A "full v12" (ResLN substrate + log_T-KD MTL run) does not exist on disk.
2. **log_T-KD × §4.2 composite deploy** — never combined (`STUDIES_…:254`; `CLOSURE.md:107` frames log_T-KD as "stacks onto whatever lands," i.e. future).
3. **log_T-KD × the architectural champion** — impossible to test: `mtl_improvement` has **no champion and no results on file** (`docs/results/mtl_improvement` does not exist). The "free upgrade that stacks" claim (`CLOSURE.md:107`) is aspirational.
4. **C1 routing × log_T-KD / multi-seed** — C1 is one-state-pass, never combined or multi-seeded.
5. **Tier-B substrate variants × log_T-KD at large states** — only FL design_b+KD pilot exists (seed=42).

There is **no single executed run** that turns on every promoted lever and is compared head-to-head against the v11 baseline. The strongest "combined" evidence is the FL KD-additive-on-B9 pilot (KD × substrate, seed=42 only, item #3 above).

---

## Objective 4 — Next-Reg STL gap vs HGI

**Verdict: PARTIALLY CLOSED at FL; EQUALISED at AL (via ResLN); NOT closed (still ~0.4–2 pp short) at AZ. No variant significantly beats HGI on STL reg at any state. This is an STL story only — it does NOT transfer to MTL.**

STL canonical→HGI gap and closure (sources: `STATE.md:23`, `phase_b_fl_3way.md`, `phase_resln_verdict.md`, all under `docs/results/substrate_protocol_cleanup/`):

- **FL:** canonical 69.22 (gethard) → **Design J 70.34 (+1.12, p=0.0312 ✓) closes ~53%** of the 2.12-pp gap to HGI 71.34. Still ~1.0 pp below HGI. **PARTIALLY closed.**
- **AL:** **ResLN+design_b 61.99 ≈ HGI 61.86 (beat-HGI p=0.31, tie)**; ResLN+design_j 62.10 nominally exceeds HGI (p=0.50). **EQUALISED (not significantly beaten).**
- **AZ:** ResLN+design_b 52.98 vs HGI 53.37 — closes ~80% of canonical's −2.04 gap but HGI keeps a nominal lead. **NOT fully closed.**

The no-prior cross-check **widens** HGI−J to +1.64 (`phase_b_fl_3way.md`), confirming the residual is real embedding quality, not a prior artefact.

**Crucial scope limit:** this gap-closure is **STL-only**. Under MTL the gap question is moot because **there is no MTL substrate gap to close** (Objective 5 / the regime finding): HGI's +2.12-pp STL reg win **vanishes** under B9 joint training (FL MTL reg HGI 64.49 vs canonical 63.98, Δ+0.51, p=0.41 NS; `phase_b_fl_3way.md`, `hgi_mtl_fl.md`).

---

## Objective 5 — MTL non-regression

**Verdict: NO regression. The v12 default MAINTAINS MTL Next-Cat (flat) and IMPROVES MTL Next-Reg via log_T-KD at AL/AZ (paper-grade) and FL (pilot). ResLN adds nothing to MTL (neither helps nor hurts). The deep −7 to −17 pp MTL reg architectural gap is untouched — but that is pre-existing v11, not a v12 regression.**

Before/after (W=0.0 → W=0.2, the v12 default delta; `phase_a1_verdict.md:12-20`, `RESULTS_TABLE.md:201-228`):

- **MTL Next-Reg (disjoint Acc@10):** AL 50.59 → 52.85 (**+2.27**, n=20 seeds {0,1,7,100}, p=9.54e-07, 20/20 folds); AZ 41.30 → 46.22 (**+4.91**, same n/p). FL 63.98 → 66.38 (**+2.40**, seed=42, 5/5, p=0.031 — **pilot**); CA +1.42 / TX +1.71 (1-fold).
- **MTL Next-Cat (disjoint F1):** AL 45.96 → 45.76 (**−0.20**, flat); AZ 48.86 → 48.94 (**+0.08**, flat). FL/CA/TX cat Δ ∈ [−0.10, +0.05]. No meaningful cat regression.
- **ResLN under MTL:** all 9 MTL cells NS on both axes (`phase_resln_verdict.md`) — no regression, no gain.

The −0.20 AL cat dip is within noise (σ_fold ~2 pp). **No improvement causes a real MTL regression.** The standing v11 architectural gap (§0.1: AL −11.04, AZ −12.27, FL −7.34, CA −9.50, TX −16.59 reg; cat MTL≥STL at 4/5 states; `RESULTS_TABLE.md:70-78`) is **not made worse** by v12 — log_T-KD recovers ~2–5 pp of it at small states and ~2.4 pp (pilot) at FL.

---

## Objective 6 — Statistical reliability & reproducibility

**Verdict: reliable and multi-seed ONLY at AL/AZ. FL/CA/TX are seed=42 pilots and explicitly NOT paper-grade. No stale-log_T contamination in the A1 promotion. ResLN cat is 5-seed paper-grade (STL).**

- **AL/AZ log_T-KD — PAPER-GRADE:** seeds {0,1,7,100}, n=20, paired Wilcoxon **p=9.537e-07**, 20/20 folds positive (`phase_a1_verdict.md:24-27`). Multi-seed reproduces seed=42 within 0.15 pp (no dev-seed/C23 bias at small states). A documented scipy-dispatch caveat (exact p=9.54e-07 on raw CSV vs approx 4.42e-05 on 2-dp-rounded values) is correctly flagged (`phase_a1_verdict.md:29`).
- **FL/CA/TX log_T-KD — PILOT (seed=42), NOT paper-grade:** FL n=5 (p=0.031), CA/TX n=1 (sign-and-magnitude). The W=0.0 baselines **overshoot §0.1 multi-seed by +0.7/+2.7/+7.5 pp of pure dev-seed bias** (C23); the paired Δ survives but absolute values are not §0.1-comparable. `RESULTS_TABLE.md:224` and `CANONICAL_VERSIONS.md:115` both flag this. **Paper-grade FL/CA/TX requires re-running W=0.0/W=0.2 at {0,1,7,100}.**
- **Stale-log_T:** the A1 promotion is NOT contaminated (fresh per-fold seed-tagged log_T; the C4 mtime guard now enforces freshness). Note the separate Tier-6 FL-MTL sweep stale-May-6-log_T caveat (CLAUDE.md) is a *different* artefact and does not touch A1.
- **ResLN cat:** STL cat F1 +0.86 FL / +1.48 AL / +1.70 AZ, 5/5 seeds, p=0.03125 (`phase_resln_verdict.md`, `canonical_improvement` T3.2) — paper-grade STL, but explicitly STL-only.
- **§0 paper tables:** §0.1 remains v11 (KD off). The v12 lift lives in §0.8/§0.9 as an additive isolation, **not re-tabled into §0**. CH26 is explicitly "NOT on the paper whitelist — pending §0 re-run."

---

## Blocking gaps before this can become the new canonical baseline

Prioritized, actionable:

1. **[BLOCKER] log_T-KD not multi-seeded at FL/CA/TX.** FL is n=5 seed=42; CA/TX are n=1. Re-run W=0.0/W=0.2 at seeds {0,1,7,100}, 5 folds, before any large-state paper-grade claim or §0 re-table. (`RESULTS_TABLE.md:224`, `CANONICAL_VERSIONS.md:115`.)
2. **[BLOCKER] The stacked run was never executed.** No run on disk turns on log_T-KD + ResLN substrate + (optionally) composite and compares to v11. At minimum, run "full v12" = ResLN substrate rebuild + log_T-KD MTL at one state to confirm the composition does not interact badly. Currently the v12 "default" run trains on the **frozen v11 GCN substrate** (ResLN default only affects future builds) — so the shipped default is really "v11 substrate + log_T-KD," not full v12. Document or close this gap.
3. **[BLOCKER] §0 paper tables not re-run for v12.** §0.1 stays v11; the lift is isolated in §0.8/§0.9. CH26 promotion to the paper whitelist is explicitly gated on a §0 re-run that has not happened. Until then, the paper baseline is unchanged.
4. **[NEEDS WORK] C1 per-task routing is one-state-pass and selector-brittle.** AL fails on a genuine degenerate Acc@1-selected snapshot. The fix (swap reg-best selector to Acc@10 + add a degenerate-snapshot guard in `MultiTaskBestTracker`) is NOT implemented; multi-seed re-run required before any §0.x promotion. (`phase_c_verdict.md:120-125`.)
5. **[NEEDS WORK] ResLN MTL substrate never rebuilt/evaluated end-to-end.** ResLN is defaulted for future builds but the only evidence is STL; a full ResLN-substrate MTL run was never done (correctly, per the regime finding — but it means "v12 substrate" numbers do not exist).
6. **[OPEN] The dominant MTL reg gap (−7 to −17 pp) is unfixed and `mtl_improvement` has zero results.** Every non-architectural cause is eliminated; the actual architectural fix is unbuilt (`docs/results/mtl_improvement` does not exist). The "log_T-KD stacks onto whatever lands" claim is untestable until a champion exists.
7. **[MINOR] Tier-B "cat −2.4 pp is a build-scope confound" was never cleanly re-tested** with a byte-identical-embeddings region-only swap (`phase_b_fl_3way.md:104`). Does not affect the v12 promotions (which are not Tier-B), but leaves the substrate-cat verdict slightly hedged.
8. **[MINOR] HGI MTL ceiling is FL-only (seed=42, 5-fold).** The "MTL flattens everyone" regime claim is FL-anchored plus AL/AZ α=0 isolation; AL/AZ/CA/TX HGI-MTL not run.

---

## Bottom line — can the updated implementation become the new canonical baseline yet?

**CONDITIONAL — and the docs already encode the correct conditional, so no change of framing is needed.**

- **As a wholesale replacement for the v11 paper baseline: NO.** §0 has not been re-run; FL/CA/TX log_T-KD is pilot-only; the stack was never executed; ResLN gives no MTL benefit and its substrate was never rebuilt+evaluated; the headline MTL reg gap is untouched. v11 must remain the FROZEN paper canon (as `CANONICAL_VERSIONS.md` mandates).
- **As the new code *default* (v12) layered on a frozen v11: YES, and it is already correctly done.** log_T-KD and ResLN are wired as scoped defaults, reproduction flags (`--log-t-kd-weight 0.0`, `--encoder gcn`) are documented and tested, and every canonical doc points to `CANONICAL_VERSIONS.md`. This is the right design: **ship canonical + log_T-KD; treat ResLN/substrate as STL/generality only.**

**Conditions to upgrade v12 from "validated small-state default" to "new paper baseline":** (1) multi-seed log_T-KD at FL/CA/TX {0,1,7,100}; (2) execute at least one true stacked/full-v12 run and compare to v11; (3) re-run §0 with v12 defaults and promote CH26 off the study-section into the paper whitelist; (4) resolve C1 routing if it is to be part of the recipe. Until then, the honest status is exactly what the repo states: **v11 = paper canon (frozen); v12 = v11 + a paper-grade-small-state reg lift + an STL-only encoder default.**
