# MTL Improvement Track — Progress Log

> ⭐ **New agent? Read [`HANDOFF.md`](HANDOFF.md) FIRST** — a single current-state snapshot (frozen
> ceilings, what's done, traps, the next step) that saves reconstructing from the entries below.

Append-only chronological log. Every agent working on this track adds entries here.
Dates are absolute (e.g. `2026-05-16`), never relative ("today", "yesterday").

Sections at the bottom of each entry:
- **Decision** if you changed direction.
- **Blocker** if you got stuck (and what unblocked you, in a later entry).
- **Chain status**: which tier the chain is in; whether the chain is preserved or broken.
- **Next** what the next agent should pick up.

---

## 2026-06-02 — MAJOR REFRAME: v2 plan, rebased on v14 + the regime finding

**Phase**: Design re-centered; still no experiments run. **The v1 plan below (2026-05-16) is SUPERSEDED by INDEX.html v2** — read INDEX.html §"What changed (v1→v2)" first.

**Why the reframe.** Four studies since the v1 design collapsed most of the v1 surface and produced a sharper target. User direction: rebase on the new **v14** substrate, compare vs canonical paper (v11), fold in fitting `future_works`, and sharpen per-phase metrics for next-reg/next-cat.

**What I read (grounding).** `CANONICAL_VERSIONS.md §v11–v14`; `embedding_eval/FINAL_SYNTHESIS.md` (v14 verdict + the regime finding); all 9 `future_works/` memos + README routing; `mtl-protocol-fix/PRIORITY_IMPACT.md` + `phase1_phase2_verdict_v6_final.md` (P4 + three-frontier + selector); `B9_STL_STAN_SWAP_AZ_FL.md` (the residual-skip falsification + §6.4 gap decomposition); `composite_two_substrate_engine.md`; recent git log (embedding_eval → v14).

**The five load-bearing facts that reshaped the plan:**
1. **v14** (`check2hgi_design_k_resln_mae_l0_1`) is the STL dual-axis champion (cat 67.36 ≫ HGI; reg 0.7024 closes ~69% of canon→HGI gap). Built at FL only; AL/AZ/CA/TX pending. **STL-only.**
2. **The regime finding**: substrate gains wash out under cross-attn MTL — v14 ≈ canonical in MTL (2-fold seed42 pilot); HGI-routing also washed out. The MTL regime, not the substrate, is the wall.
3. **P4 frozen-cat**: reg peaks at ep 2 even with cat frozen → the gap is **architectural** (shared-backbone reg pathway), NOT cat-vs-reg interference → loss-balancing DEMOTED.
4. **Residual-skip FALSIFIED** (B9_STL_STAN_SWAP §6.2, −0.59pp); encoder-MLP-depth flat; ~75% of the gap is the **missing full private backbone** (§6.4). → centerpiece sharpened to a **reg-private full backbone / dual-tower**, NOT a generic backbone bake-off and NOT a thin skip.
5. **log_T-KD** is the one confirmed MTL-reg lever (+2.4/+5.06/+2.32pp, now v12 default); **composite** (STL c2hgi cat + STL HGI reg) is the deploy ceiling (+7-12pp reg).

**Structural changes v1→v2:**
- 8 tiers → 6; 32 experiments → ~16.
- New base = v14 (was v11). New comparand = 4 anchors: (a) v14-base-MTL, (b) frozen v11 §0.1 [paper continuity, privileged-draw label], (c) STL-on-v14 ceiling, (d) composite ceiling. **Fresh-vs-frozen discipline** mandatory (compare v14-fresh vs canonical-fresh gcn_ctrl for regime claims).
- **Three-frontier reporting** (STL / MTL@disjoint / MTL@geom_simple deployable) is now mandatory — the v1 plan missed the selector axis entirely.
- Tier 2 (architecture) re-centered on the dual-tower; loss demoted to gated Tier 4; batch-sampling dropped (falsified).
- The load-bearing T2.1 runs on BOTH v14 AND canonical-fresh (regime×substrate 2×2) — advisor-mandated: can't show "fixing the regime makes the substrate matter" on v14 alone.
- Folded in: part2_routing → T2.1; mtl_architecture_revisit → T2.2-2.4; substrate_adaptive_balancing → T4.1 (gated); composite_two_substrate → T1.2 + T6 fallback; reg_head_sweep/head_window_batch → T3.2/T5; paper_canon_reevaluation → T6.2; poi_decoder_hgi_distill → standby (out of chain).

**Advisor pre-write pass (model-level) — 4 items, all applied:**
1. Verify the reg-head bypass wasn't already tried → it WAS (residual-skip falsified); centerpiece changed from "bypass" to "full private backbone (dual-tower)."
2. 4-anchor comparand + fresh-vs-frozen as the metric frame (the user's "sharpen metrics" ask).
3. Headline arch experiment must run on both substrates (regime×substrate interaction) — built into T2.1.
4. Don't over-read the 2-fold pilot → T0.3 regime entry gate confirms v14≈canonical at multi-seed before committing; don't resurrect two-checkpoint disjoint-deploy (user-dropped) — disjoint is reporting-only.

**Memory written:** `mtl_regime_finding.md` (the architectural-gap + falsified-levers + confirmed-lever facts).

**Chain status**: v2 design complete; Tier 0 not launched.

**Next**: a final advisor sub-agent pass on the v2 plan (the user asked for it), then implementing agent starts Tier 0 (build v14 per state → fresh-vs-frozen → regime entry gate → three-frontier). T0.1 (v14 build) + T0.3 (regime gate) are the prerequisites for everything.

---

## 2026-06-02 — Final advisor pass on v2; 5 fixes applied

**Phase**: Design hardening (advisor sub-agent review of the v2 plan, per the user's ask). All 5 substantive items applied as edits.

1. **Encoder-isolation diagnostic added as T1.3 (gates T2.1).** Advisor: §6.4 names FOUR co-equal suspects for the ~75% residual (upstream `next_encoder`, PCGrad surgery, joint-loss interference, sequence-vs-pooled handoff) — the plan over-attributed it all to "missing private backbone." §6.6's own recommended FIRST test (run STL `next_stan_flow` WITH the MTL `next_encoder` prepended, ≈5 GPU-h) is the cheapest probe. T1.3 runs it before the ~320 GPU-h Tier 2: if the encoder owns the residual, the cheap fix is encoder-bypass, not the dual-tower.
2. **T2.1 re-prioritized: gated-fusion (b) is now the PRIMARY variant.** Advisor: variant (a) private-only is, by construction, the composite (anchor d) trained jointly — not a novel result if it wins. Only gated-fusion (b) tests the real thesis (shared backbone ADDS to the private tower). Added a **PCGrad-off / static_weight arm** inside T2.1 (PCGrad is suspect iii, baked into the recipe not gated).
3. **Regime×substrate interaction demoted from headline to secondary.** Advisor: v14's STL reg edge over canonical-fresh is only ~+0.8pp at FL — below the n=5 ≥1pp floor — so the interaction may null even if the architecture works. The defensible headline is now "dual-tower recovers X% of the composite reg gap inside one model" (substrate-agnostic); the v14&gt;canonical interaction is a bonus secondary readout.
4. **Metrics consistency fixed.** The per-task table named joint primary = Δm(cat F1 + reg MRR) but every gate used Acc@10. Now **reg Acc@10_indist is THE reg metric for every gate, table, and the Δm primary**; MRR is secondary-only. Consistency rule added to §Metrics.
5. **v14 framing qualified as FL-conditional.** Subtitle no longer says "on the v14 substrate" universally — substrate-survival claims are FL-conditional (Delaunay reg lever is state-dependent); the regime/architecture work is substrate-agnostic and survives the T0.1 fallback.

**Advisor's honesty verdict:** "excellent — v14 consistently framed STL-only/MTL-unproven; all T2.1 outcomes called paper-grade; composite fallback explicit; no over-claim." Metrics contract called "ship-ready." Centerpiece called "defensible after the two fixes (encoder probe + gated-fusion/PCGrad-off)."

**Nothing high-EV was dropped** (advisor cross-checked all 9 future_works memos; `task_pivot_memo` is paper-prose, correctly unmapped). The only real gap was the missing §6.6 encoder probe — now T1.3.

**Chain status**: v2 design hardened; Tier 0 not launched. **Ready for implementing-agent handoff.**

**Next**: implementing agent — read AGENT_PROMPT.md (v2) → INDEX.html → build worktree on `mtl-improve` → Tier 0. The cheap T1.3 encoder probe should run early (it can re-scope the expensive Tier 2).

---

## 2026-06-02 — User points folded in (frozen folds, per-task disjoint+joint, hard-share arm)

**Phase**: Design refinement (3 user points). All applied.

1. **Frozen-fold paired design** (user: "freeze the folds so we don't face data interference and only see model infrastructure"). Added a §Metrics protocol: for infra-isolation experiments (Tier 2 architecture, T1.3 encoder probe, T5 head swaps), all arms use the identical `StratifiedGroupKFold(seed=S)` partition + matching seeded log_T, compared fold-by-fold so fold-composition variance cancels in the Δ. Documented what it controls (data-difficulty variance, the dominant small-state noise) vs not (init/batch-order — vary seed but keep the split aligned per seed). NOT applied to the T6 ship run / T0.3 regime gate (those vary folds for generalization). Tier 2 intro + AGENT_PROMPT rule 2b updated.

2. **Per-task disjoint + joint for BOTH tasks** (user: "for MTL runs compare the disjoint results for each task AND the joint result"). The v2 metrics had three-frontier reg only + cat as a single number. Now every MTL run reports a **2×2: {cat, reg} × {disjoint, joint}** — each head at its own best epoch (disjoint capacity) AND both heads at the single deployable `geom_simple` checkpoint (joint). Added the mandatory subsection + example block + consistency footnote; AGENT_PROMPT rule 2 updated. The cat disjoint−joint and reg disjoint−joint gaps quantify the per-task cost of one shared checkpoint (the mtl-protocol-fix capacity-gap + selector-bug, now applied to cat too).

3. **Hard-share backbone arm** (user: "we're only using soft-share, right? eval with a literature agent whether to add a hard-share backbone"). Spawned a literature+codebase agent. **Verdict: user is right** — every live Tier-2 candidate is soft-share (cross-stitch, MoE, cross-attn) or private (dual-tower); no clean hard-parameter-share arm. Added **T2.0 hard-share anchor** (FiLM-off `mtlnet` = textbook Caruana shared trunk + task heads), completing the sharing spectrum hard←→soft←→private. Expected to LOSE (P4 = task-starvation-by-sharing, so more sharing is wrong; "Rethinking Hard-Parameter Sharing" arXiv 2107.11359 recommends less). Value = the completed spectrum that frames the dual-tower win + pre-empts the reviewer "did you try plain hard sharing?". Agent verdict: do NOT add AdaShare/Learning-to-Branch — at 2 tasks they collapse to branch-depth, which the dual-tower already spans (cite as related work for T2.1). Sources: Vandenhende TPAMI'21, Crawshaw'20, Rethinking-HPS arXiv 2107.11359, AdaShare NeurIPS'20, Learning-to-Branch ICML'20, BMTAS BMVC'20. Tier 2 compute 320→340 GPU-h.

**Chain status**: v2 design + 3 user points; ready for handoff. Tier 0 not launched.

**Next**: unchanged — implementing agent starts Tier 0; T1.3 encoder probe early; Tier 2 spans the full spectrum under frozen folds with per-task disjoint+joint reporting.

---

## 2026-06-02 — Tier-0 frozen-fold reference board + advisor gap-fixes

**Phase**: Tier-0 restructure (user point) + advisor eval. All applied.

**User point:** "freeze folds should be prepared in Tier 0, and execute STL + MTL v11 & v14 on it, so we can eval if current results are equivalent to the paper and prior studies and future comparisons are clear."

**Restructure applied:**
- **T0.0 (NEW)** — freeze the fold partition + seeded log_T as an immutable, hash-checked artifact (manifest per state×seed); gates all of Tier 0+.
- **T0.2 (REBUILT) — the frozen-fold reference board + equivalence gate.** Runs all anchors on the ONE frozen partition: MTL v11 (frozen substrate, paper-canon), MTL canonical-fresh (gcn_ctrl), MTL v14, STL-on-v14, STL canonical-fresh + STL HGI, composite. **Two equivalence checks: MTL v11 frozen-fold ≈ paper §0.1, and three-frontier ≈ mtl-protocol-fix + v14 STL ≈ embedding_eval.** This is the calibration gate — prove the harness reproduces the paper + prior studies BEFORE any improvement, so every later Δ is trustworthy. Divergence beyond tolerance → STOP at Tier 0, not Tier 6.
- **T0.3** reframed to read the board's v14-MTL vs canonical-fresh-MTL cells as the n=20 regime test (no separate run).
- **Tier 1 slimmed:** T1.1/T1.2 are now confirm-and-pin from the T0.2 board (no fresh runs); T1.3 (encoder probe) is the real Tier-1 work, marked PARALLEL-ELIGIBLE (runs as soon as T0.0 lands).

**Advisor eval (model-level, full context) — affirmed the direction ("calibrate-before-improve is exactly right"), gave 5 fixes, all applied:**
1. **Partition identity (BLOCKER).** The frozen partition must be bit-identical to the §0.1-generating `FoldCreator` path (same sklearn 1.8.0, group key, seeds) + hash-check vs surviving §0.1 artifact — else a board "divergence" can't isolate the harness. Added as a T0.0 precondition.
2. **Pre-register tolerances (BLOCKER).** Equivalence cells use {0,1,7,100} NEVER seed42 (C23: seed42 overshoots §0.1 by +3/+7pp CA/TX by design); write accept-tolerances citing C22/C23 so the gate doesn't false-STOP on documented offsets; C23 is a CA/TX issue (board clean at AL/AZ/FL) → flag for T6.
3. **Compute ~3× low (BLOCKER, planning).** Full {0,1,7,100} × 3 MTL substrates × 3 states ≈ 400-600 GPU-h. Scoped: full multi-seed equivalence at FL only; AL/AZ board cells at seed42, multi-seed deferred to T6. Tier 0 compute updated 150 → ~300-350 GPU-h.
4. **Dual-tower success bar (STRATEGIC).** The composite ALREADY delivers +7-12pp reg at deploy (two models). So the integrated dual-tower wins over it only if reg-recovery is high enough that single-model economy + cat-survival justify not shipping two models — a ⅓-gap recovery that stays 5-8pp below composite is a SCIENTIFIC result, not a shippable win. Reframed T2.1's gate: the bar is the deployment trade vs the composite, not a raw gap-fraction.
5. **v14 build-input check (BLOCKER).** Verify POI2Vec teacher + HGI Delaunay edges + HGI POI emb exist at AL/AZ before T0.1; fold build cost in if absent. Added to T0.0.
- Process: T1.3 marked parallel-eligible (off the serial T0.0→T0.1→T0.2 critical path). Advisor's explicit note: "this is the 4th revision, the plan is in good shape — stop expanding scope after this." Agreed — no new machinery beyond these fixes.

**Chain status**: v2 + frozen-fold reference board + 5 advisor fixes. Ready for handoff. Tier 0 not launched.

**Next**: implementing agent — T0.0 (freeze partition, provenance-matched to §0.1) → T0.1 (build v14, check inputs first) + T1.3 (encoder probe, parallel) → T0.2 (reference board + equivalence gate — the calibration checkpoint). Do not proceed past T0.2 if v11 frozen-fold ≠ §0.1 beyond tolerance.

---

## 2026-06-03 — Git pull: execution started; state strategy + hardware/parallelization folded in

**Phase**: Plan sync with the execution environment (user: git pull happened, eval impact; add GE middle state + 5f/1f strategy; AGENT_PROMPT update; parallelize on i9-14900K + A40 45GB).

**Git pull (9e08b49f → 7a27e688) — what landed + impact:**
- **`docs/results/v14_mtl_vs_canonical.md` — our T0.2/T0.3 already executed at FL+AL+AZ.** v14-MTL vs matched-canonical-MTL, 5f × {0,1,7,100}, leak-free. **Regime gate CONFIRMED**: v14 ≈ matched canonical in MTL (FL clean tie; AL +2.1 reg/+1.2 cat deployable; AZ within noise). STL ceilings replicate exactly (v11 69.43 / v14 70.24 / HGI 70.62). → marked T0.3 DONE-at-FL+AL+AZ; the architectural-lever framing is validated, not at-risk.
- **Selector default flipped (C21, 2026-06-03):** `geom_simple = sqrt(cat_F1·reg_Acc@10)` is now the code default (was broken `0.5·(cat_f1+reg_f1)`). §0.1 is diagnostic-best (selector-independent). → updated T0.4 (selector is default, not a manual wire) + the metrics framing.
- **Matched-in-harness baseline rule (empirically pinned):** same canonical yields 53.7-64.5 reg@10 in-harness vs §0.1's 63.27 — a harness offset. The valid Δ is vs matched in-harness baseline; §0.1 is reference-only. → refined the T0.2 equivalence gate (don't expect frozen-fold v11 = §0.1 to the decimal; expect the offset).
- **New memo `joint_selection_and_loss_combination.md`:** geom_simple is literature-sound (GM, collapse-averse); the real loss lever is **scale-normalization** (~4.7× CE magnitude gap, 7-class ln7≈1.95 vs 9k-class ln9000≈9.1) — normalize CE by log(num_classes); **FAMO** is the O(1) balancer (PCGrad/Nash O(k)-infeasible at 9k). → re-ranked Tier 4: added **T4.0 (loss-scale norm + RLW litmus, ungated, highest-EV)**; T4.1 re-scoped to FAMO-led, O(k) methods deprioritized.
- **`scripts/_v14_run/` drivers + manifest:** runs are SERIAL; A40 timing FL ≈14min/seed, AZ ≈3min, AL ≈1.6min (5f×50ep); builds (design_k 500ep) are the heavy spine. → the parallelization headroom.
- CANONICAL_VERSIONS/CONCERNS/CHANGELOG/TASKS updated (C21 selector, v14 MTL eval) — consistent with the above; no contradiction to the plan.

**Changes applied to the plan:**
1. **§What already landed (new):** records the regime confirmation, selector default, matched-in-harness rule, new levers — so Tier 0 isn't re-run.
2. **§State strategy (new):** AL/AZ (small, 5f) · **GE (middle, 5f — NEW, the small↔huge bridge, user-essential, not yet built)** · FL (large, 5f) · CA/TX (huge, 1f optional). Decision rule: 5f bands are main evidence; CA/TX 1f directional. Added **T0.1b — onboard GE** (raw data → substrates → frozen folds → log_T). GE threaded into T0.2 board + Tier-4 states.
3. **§Execution & parallelization (new):** i9-14900K (32t) + A40 45GB. **CUDA MPS** collocation for small-state runs (they underutilize the A40 — 20%→60% util, up to 4× per MPS literature); VRAM-budget concurrency cap; FL ≤2 concurrent; ONE build at a time; CPU-parallel prep overlapped; don't-hurt rules. Searched + cited NVIDIA MPS docs + arXiv 2209.06018 + Databricks.
4. **AGENT_PROMPT.md:** added the 2026-06-03 landed-results banner + hard rules 13 (states) + 14 (parallelization).

**Chain status**: plan synced with execution; T0.2/T0.3 DONE at FL+AL+AZ (regime confirmed); remaining Tier-0 = GE onboarding + GE board cells + (cheap) T1.3 encoder probe. Ready to continue execution.

**Next**: T0.1b (onboard GE — heaviest new build, schedule early) → GE board cells → T1.3 encoder probe (parallel, gates T2) → Tier 2. Use MPS collocation for the small/middle-state seed sweeps.

**Advisor pass (2026-06-03) — 5 reconciliation fixes, all applied (advisor said: apply + hand off, no more revisions):**
- **A (behavior):** the matched-in-harness rule wasn't propagated. Bannered the §Pinned-Baselines residual table as mtl-protocol-fix-harness/reference-only; made T2.1's "fraction of composite gap" bar reference the T0.2 IN-HARNESS composite (cross-harness subtraction = ~2.6pp meaningless error).
- **B (check):** GE raw data CONFIRMED present (`data/checkins/Georgia.parquet`) → T0.1b is a build task, not data acquisition. Updated T0.1b + §State strategy.
- **C (behavior):** corrected the inflated CPU-era compute estimates with measured A40 timing (FL 5f×50ep ≈14min/seed). Run-bound tiers are tens of GPU-h, not hundreds; builds are the only heavy item (measure from build logs on the box). Added a corrected-envelope callout to §Execution; flagged the compute-fear scope-downs (T0.2 FL-only, CA/TX 1-fold, Tier-4 gated) as possibly-unnecessary — relax once build time is known. (Note: build logs are on the remote A40 box, gitignored — not measurable from here.)
- **D:** promoted T4.0 (loss-scale norm + RLW litmus) to RUN-EARLY alongside T1.3 (orthogonal to the architecture per P4; its RLW signal informs the expensive tiers).
- **E (honesty):** AL is a selector-dependent CROSSOVER, not a clean tie (deployable v14 wins +2.14 reg; diagnostic-best v14 worse −2.23). Updated T0.3 to report both bases; reframed GE as mapping the **substrate-survival-vs-scale gradient** (AL partial-survival → GE middle → FL clean-tie), a sharper job than a generality check.

**Design status: DONE per advisor — next action is execution, not revision.** T0.2/T0.3 regime gate confirmed at FL+AL+AZ; remaining Tier-0 = GE onboarding (T0.1b) + GE board cells + T1.3 encoder probe + T4.0 (run early).

### Review-cadence cautions added to AGENT_PROMPT (user-requested, 2026-06-03)
Strengthened workflow §4–§6 + execution §8 with three disciplines: (1) **per-tier review cadence** — advisor pass on the tier results → write a tier summary → **STOP and surface it to the user to decide how to proceed** (no autopilot into the next tier); (2) **end-of-track implementation-correctness review** — a dedicated advisor/code-review sub-agent that verifies the code is right for THIS case (param partitions / log_T / selector / frozen-fold guards; not just that numbers look good — guards against the F49-leak / stale-log_T / wrong-selector bug class) + a final whole-track advisor pass; (3) **continuous `log.md` + `INDEX.html` updates** — fill Results blocks + log decisions as they happen, a task isn't `completed` until both are written. **(4, added next) commit-cadence discipline** — workflow §7: small frequent commits (per experiment/decision/build), explicit pathspec + `git status` check (the repo pre-stages unrelated `articles/*`), dedicated branch + deliberate PR merge, push regularly so remote-A40 results are durable.

---

## 2026-06-03 — EXECUTION START (implementing agent): T0.0 frozen folds + T0.1b GE onboarding launched

**Phase**: Tier 0 in flight. New implementing-agent session. User scope this session: **finish Tier 0 + Tier 1, then stop at the boundary for review.** User decisions: (a) work on branch **`mtl-improve` in this checkout** (no worktree — single box, drivers hardcode the main repo path); (b) **start the GE build in background early**.

**Ground-truth audit before any action** (verified on the box, not from docs):
- A40 idle; venv `.venv/bin/python` (3.12, sklearn 1.8.0); MPS control present; no jobs running.
- v14 substrate (`check2hgi_design_k_resln_mae_l0_1`) built at **FL, AL, AZ only**. Canonical + HGI at AL/AZ/CA/FL/TX. **GE built nowhere.**
- T0.2/T0.3 regime gate confirmed at FL+AL+AZ landed in `docs/results/v14_mtl_vs_canonical.md` (imported, not re-run). Drivers in `scripts/_v14_run/`.
- Seeded per-fold log_T present for AL/AZ/FL at seeds {0,1,7,100,42}.

**T0.0 — frozen-fold manifest DONE for AL/AZ/FL** (`scripts/mtl_improvement/freeze_folds.py`; manifests at `docs/results/mtl_improvement/frozen_folds/{state}_seed{S}.json`, 15 files for seeds {0,1,7,100,42}). Provenance-matched (advisor #1): reuses the EXACT split from `compute_region_transition.py::_build_per_fold` == `FoldCreator._create_mtl_folds_with_isolation` (`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(load_next_data(state, CHECK2HGI), y=next_category, groups=userid)`) — the same split the on-disk seeded log_T was built from, so it IS the operative partition. Records per-fold train/val userid lists + per-fold + partition hashes + sklearn version. `--check` mode is the preflight drift-guard (verified idempotent: live == frozen, all OK). GE manifests pending its canonical build.

**T0.1b — GE onboarding launched (background, detached).** `scripts/_v14_run/build_ge.sh` (modeled on the tested `build_v13_catx.sh`; disk-guarded, fail-fast, idempotent skips, stage markers in `/tmp/ge_build/`). Stages: A canonical check2hgi (preprocess+500ep GCN train + sequences — the canonical-fresh comparand AND the graph/sequences v14 consumes) → B HGI **preprocess** (Delaunay edges.csv, v14 prereq) → C POI2Vec teacher (100ep) → D v14 design_k (500ep) → E seeded log_T {0,1,7,100,42} + postbuild + copy to v14. HGI **training** (region embeddings, for STL-HGI/composite ceilings) is the **deferrable tail** (separate script) — v14 only needs HGI preprocess.

**BLOCKER found + resolved: GE shapefile was MISSING.** The plan/AGENT_PROMPT asserted GE was "build, not data-acquisition" based on `data/checkins/Georgia.parquet` alone — but check2hgi + HGI preprocess BOTH require the TIGER census-tract shapefile (`Resources.TL_GA` → `data/miscellaneous/tl_2022_13_tract_GA/`), which was **absent**. Downloaded the public US Census TIGER 2022 GA tract file (`tl_2022_13_tract.zip`, FIPS 13) → unzipped to the expected dir (matches the FL layout: .shp/.dbf/.prj/.shx/.cpg). Keeps GE methodologically identical to the other 5 states (census-tract boundaries, not a synthetic grid). `Georgia.parquet` = 402,581 check-ins. **Lesson: a state's onboarding needs BOTH the checkins parquet AND its TIGER shapefile — verify the shapefile, not just the checkins.**

**Decision**: HGI training for GE deferred (composite/STL-HGI is "held until a winner emerges" per plan); GE essentials (canonical + v14 + folds + log_T) cover the T0.2 board MTL cells + T0.3 regime cell + STL v14/canonical ceilings, which is the in-scope Tier-0/1 need.

**Chain status**: Tier 0 in flight (T0.0 done AL/AZ/FL; T0.1b GE build running). Chain preserved.

**Next**: monitor GE build → write GE frozen-fold manifest → import landed FL/AL/AZ board cells + add GE board/regime cells (T0.2/T0.3) → cheap GPU diagnostics T1.3 (encoder probe, gates T2) + T4.0 (loss-scale/RLW) once GPU frees → T1.1/T1.2 ceilings → Tier-0/1 boundary advisor pass + STOP for user.

---

## 2026-06-03 — TIER 0 + TIER 1 SUMMARY (boundary review — STOP for user)

**Phase**: Tier 0 done; Tier 1 substantially done (T1.3 the gating experiment closed; ceilings partially pinned). Per workflow §4: advisor pass run, summary written, **now surfacing to the user — no autopilot into Tier 2.**

### What ran (all 2026-06-03, A40, branch `mtl-improve` in-checkout)
- **T0.0 frozen-fold manifests** — AL/AZ/FL/GE × seeds {0,1,7,100,42} (20 files), provenance-matched to the trainer/log_T `StratifiedGroupKFold` split; `--check` drift-guard idempotent.
- **T0.1b GE onboarding** — built from scratch (~16 min): downloaded the missing GA TIGER shapefile, then canonical check2hgi + HGI-preprocess + POI2Vec + v14 design_k + 25 seeded log_T. **n_regions = 2283 (middle band, between AZ 1547 & FL 4703).** HGI-train (composite/STL-HGI) deferred. Canonical GE `next_region.parquet` was missing for the MTL canon arm → built via the engine-agnostic `build()`.
- **T0.2/T0.3 GE board** — MTL v14 vs matched canonical, H3-alt, KD-off, seeds {0,1,7,100}.
- **T0.4** — selector/guard verification (geom_simple default, n_splits guard, 3-frontier trackers) — DONE by code inspection.
- **T1.3 encoder-isolation probe** + **prior-OFF re-run** — the Tier-2 gate.
- **Graph regen** — AL/AZ/FL `checkin_graph.pt` had been cleaned post-build; regenerated (user-authorized) and **determinism directly verified** (recomputed region_idx == stored, n_regions exact-match §0.1; cfg1 FL 70.28 ≈ landed 70.24).

### Headline results
**GE board (middle band):** deployable Δreg **−1.15** (v14 42.64 / canon 43.79), Δcat **+0.70**; diag Δreg −1.47. **Regime CONFIRMED at the middle band** — v14's STL gains don't survive MTL; small cat edge persists, reg slightly worse. Deployable-Δreg sign pattern **AL +2.14 / AZ −1.01 / GE −1.15 / FL −0.33** → v14 reg-survival is **AL-specific, NOT a scale gradient** (advisor P2: magnitudes confounded — GE canon is FRESH while AL/AZ/FL canon is FROZEN; FL=B9 vs GE=H3-alt; only the sign pattern is interpretable; each state's Δ is a valid paired comparison).

**T1.3 (the gate):** prepending the MTL `next_encoder` to the STL reg backbone costs ~0pp at every state — with prior (AL −0.93/AZ −0.57/FL −0.13) AND prior-OFF/embeddings-only (AL −0.88/AZ −0.30/FL −0.66, cfg2 ≥ cfg1 everywhere, FL tight σ). **The encoder architecture is NOT the residual** (advisor P1 settled by the prior-off re-run — the prior was not masking encoder damage). The locus is the **joint-training dynamics** (cross-attn co-adaptation / PCGrad / shared-backbone handoff) → **T2.1 dual-tower is the right lever; the cheap encoder-bypass would not help.** Honest scope: tests a standalone encoder, not the jointly-trained one in situ — but since the architecture is neutral, joint dynamics are the locus. **Side-finding:** at FL embeddings-only (73.31) > with-prior (70.28) → the log_T prior is a net DRAG on FL STL reg with v14 embeddings (scale/substrate-dependent — flag for Tier 3).

**Ceilings:** T1.1 v14 STL reg ceilings measured as a by-product of T1.3 cfg1 (with-prior): AL 62.32 / AZ 52.87 / FL 70.28 (≈ landed). T1.2 FL in-harness composite deploy ceiling +9.4pp reg pinned. GE STL-reg + AL/AZ/GE HGI-STL/composite still pending (GE HGI-train is the deferred tail).

### Advisor pass (general-purpose sub-agent, adversarial) — 4 findings, all applied
- **P1 (HIGH, FIXED):** T1.3 was over-read (α·log_T prior could mask the embedding path; `--mtl-preencoder` is a standalone re-impl). → ran the prior-OFF re-run (rescues the verdict on the embeddings-only metric) + downgraded scope to "standalone encoder not harmful → joint dynamics are the locus → dual-tower." 
- **P2 (MED, FIXED):** cross-state gradient mixes GE-fresh vs AL/AZ/FL-frozen canon + B9 vs H3-alt → annotated as sign-pattern-only, not a scale curve; per-state Δ valid (paired).
- **P3 (LOW, FIXED):** "conclusive at σ±0.5" was single-seed fold-SD → downgraded to "directionally clear."
- **P4 (LOW, noted):** t13 driver/manifest record `embedding_eval/` but p1 saves to `docs/results/P1/` (numbers fine; aggregator + INDEX read the right path).
- Cleared by advisor: log_T freshness (both arms fresh), composite arithmetic, freeze_folds provenance, STL replication.

### Falsified / promoted this tier
- **Falsified:** the encoder-bypass shortcut (T1.3 — encoder not the residual). The clean "substrate-survival scale gradient" (P2 — it's AL-specific sign, not a curve).
- **Promoted/confirmed:** the regime finding at the GE middle band; the dual-tower (T2.1) as the Tier-2 lever (now positively motivated, not just by elimination).

### Out-of-scope-but-flagged
T4.0 (loss-scale norm + RLW) NOT run (it is Tier 4; the user scoped this session to Tier 0+1). T0.5 bespoke instrumentation (grad-cosine/α-trajectory) NOT built — existing trackers already emit per-task disjoint/joint + compute panel; defer until Tier-2 diagnostics need it. GE STL-reg + HGI-STL/composite at AL/AZ/GE pending (GE HGI-train tail).

**Chain status**: Tier 0 complete; Tier 1 gate (T1.3) closed → dual-tower confirmed as the Tier-2 centerpiece. Chain preserved. **STOP — awaiting user decision on how to proceed into Tier 2.**

**Next (proposed, for user to confirm)**: (a) optional: finish T1.1/T1.2 cleanly (GE STL-reg + AL/AZ/GE HGI-STL/composite — needs GE HGI-train, ~hours) and run T4.0 early; OR (b) proceed to Tier 2 — implement the T2.1 dual-tower (reg-private full STAN backbone, gated-fusion PRIMARY + PCGrad-off arm), unit-test the param-partition, per-arch LR mini-sweep, frozen-fold paired on AL/AZ/GE/FL. The advisor's standing caution: the dual-tower's private backbone is a NEW param group — wire it into `shared/cat_specific/reg_specific_parameters()` bijectively.

---

## 2026-06-03 — T1.1/T1.2 ceilings COMPLETE (user chose finish-ceilings-first)

**Phase**: Tier 1 ceilings pinned in-harness at all 4 main states (AL/AZ/GE/FL). User picked "finish Tier 1 ceilings first" at the boundary.

**GE HGI trained** (`build_ge_hgi_train.sh`, phase4 pickle + phase5 train, CPU, **only 118s** — GE is small) → `output/hgi/georgia/region_embeddings.parquet` (2283, 65), clean. The deferred T0.1b HGI tail is now done; GE is fully onboarded (canonical + v14 + HGI).

**Ceiling sweep** (`t1_ceilings.sh`, seed42 5f, GPU, ran in parallel with the CPU HGI train):
- **(c) STL-on-v14 — pinned at all 4 states.** cat F1: v14 ≥ canon everywhere (AL +0.78 / AZ +0.25 / GE +0.60 / FL +1.27). reg Acc@10 (v14 / canon / HGI): AL 62.32 / – / 63.05; AZ 52.87 / – / 53.50; GE 55.81 / 54.36 / 56.50; FL 70.28 / 69.43 / 70.62. **v14 STL reg closes ~68% of the canon→HGI gap at GE — exactly as at FL** (the Delaunay lever reproduces at the middle band), HGI keeping the edge. v14's STL dual-axis gain is real at every band; it only washes out in MTL (regime finding).
- **(d) composite deploy ceiling — pinned in-harness at all 4 states** (cat=STL-v14, reg=STL-HGI): composite reg vs MTL deployable reg = **AL +12.91 / AZ +15.72 / GE +13.86 / FL +9.41**. The two-model composite beats the single MTL model by +9.4–15.7pp reg at zero cat cost; the gap is LARGER at small/middle states. This is the in-harness upper bound the T2.1 dual-tower must approach (supersedes the cross-harness mtl-protocol-fix AL/AZ figures).

**Tier-1 anchors (c)+(d) now fully pinned in-harness** → every later Tier-2 Δ (esp. "fraction of the composite gap recovered") is measured against a clean, same-harness composite at every state. INDEX T1.1/T1.2 Results blocks filled with the full matrices.

**Caveat:** AL/AZ canonical STL-reg not re-run in-harness (used §0.1 frozen ref 61.21/53.06); the (d) composite reg arm uses HGI-STL (run in-harness), so this doesn't affect the composite bar.

**Chain status**: Tier 0 + Tier 1 COMPLETE (ceilings pinned). Chain preserved. **Ready for Tier 2 (T2.1 dual-tower) — awaiting user go.**

**Next**: Tier 2 T2.1 dual-tower (reg-private full STAN backbone, gated-fusion PRIMARY + PCGrad-off arm; param-partition unit-test; per-arch LR mini-sweep; frozen-fold paired AL/AZ/GE/FL; measure fraction of the in-harness composite gap recovered). Optionally T4.0 (loss-scale/RLW, ungated) first.

---

## 2026-06-03 — Plan addition: STL ceiling-hardening (T1.4) + STL Frontier deep-dive (Tier S)

**Phase**: Design addition (no new experiments run). Inserted into the v2 plan after T1.1/T1.2; precedes Tier 2.

**Why.** User flagged a remembered-but-missing thread: the L0-L2 substrate-protocol-cleanup work shows STL heads leave potential behind, and (separately) the public baselines hint at STL headroom — so we should sweep + literature-search + try new STL archs and HP-tune for both tasks, before Tier 5, since Tier 5 is STL-*within*-MTL while this is STL-*alone* (and best-STL ≠ best-STL-in-MTL).

**Grounding (2 parallel Explore agents + 1 web lit-scan agent).**
1. **The genuine gap is narrower than "STL is under-built."** The L0-L2 "potential left behind" is mostly *encoder/substrate* (ResLN +9-10pp cat AL/AZ; design_k +0.9-1.1pp reg) — already captured by v14. The truly-untouched axis is the **STL head**: no dedicated head sweep or HP-tune has ever run, and the current reg head (`next_stan_flow`) was chosen in the *pre-leak* P1 ablation, never re-validated leak-free (confirmed: `reg_head_architecture_sweep.md`).
2. **Lit scan (skeptical):** do NOT import next-POI *models* (GETNext/STHGCN/Graph-Flashback/MobGT/Diff-POI) — their headline is the transition prior we already own via α·log_T, and they need raw GPS/time/category/POI-level inputs our 9×D=256 check-in embeddings lack. Only encoder *cores* transplant (self-attention/SASRec, Mamba block, attention pooling). The 9k-class "blocker" is a phantom for an STL head. The honest lever = encoder-backbone swap + imbalance losses (logit-adj/focal for cat, tail-loss for reg) + re-tuned α.

**Advisor pass (pre-write).** Confirmed the framing and caught the one must-resolve issue: the **moving-baseline trap** — if a late STL win lifts anchors (c)/(d) after T2 was already scored against the old ceiling, the whole three-frontier scoreboard shifts mid-track. Also: frame as **ceiling/baseline integrity, NOT "STL→MTL transfer"** (the regime finding predicts STL gains wash out jointly; selling it as an MTL lever argues against our own headline); cat = **loss-only** (cat isn't the MTL bottleneck); carry the de-scoping verdict verbatim.

**Decision (user-aligned via AskUserQuestion).**
- Split into two: **T1.4 (bounded ceiling-hardening)** runs in Tier 1, BLOCKS T2, and then **FREEZES (c)/(d)** for the rest of the track; the **deep search runs as a full in-track tier (Tier S)** [user choice] but feeds T5 + future-work **only** — it never re-opens the frozen ceiling (the moving-baseline resolution).
- **T1.4 scope** (tuning-first): α re-tune per-state leak-free (highest-impact scalar; heed the T1.3 FL side-finding that the prior is a net drag at large states) + imbalance losses (reg tail-loss; cat loss-only logit-adj+focal+label-smoothing on `next_gru`) + ONE SASRec encoder probe (the gated entry to Tier S).
- **Tier S scope**: S.1 reg encoder/arch deep search (SASRec, Mamba block, SimGCL aux, + the leak-free reg-head registry candidates); S.2 cat loss-first + one attention-pooling encoder. STL-alone, **FL+AL+AZ 5f + GE confirm** [user-chosen coverage]. Runs **parallel with Tier 2-4 via MPS** (STL-only → cheap → near-zero added wall-clock).

**Files touched (this session).** `INDEX.html`: new T1.4 card + new Tier S section (S.1/S.2) + de-scoping callout; TOC + CSS pill; chain diagram (T1.4 freeze + Tier S branch rejoining at T5); metric map (T1.4 + Tier S rows); why-order + moving-baseline callout; audit C1, future-works pointers (reg_head_sweep/head_window_batch → T1.4/Tier S), Tier-5 candidate-set note, parallelization Tier-S filler, 2 new hard rules (frozen ceiling; no model imports). `reg_head_architecture_sweep.md`: corrected the stale "absorbed into T7" pointer.

**Chain status**: Tier 0 + Tier 1 (T1.1/T1.2) COMPLETE; **T1.4 inserted as the remaining Tier-1 gate before Tier 2**; Tier S added as a parallel branch. Chain preserved (T1.4 is a Tier-1 prerequisite, not an out-of-order run).

**Next**: implementing agent runs **T1.4** (bounded ceiling-hardening; freeze (c)/(d) on completion; update T1.1/T1.2 + §Pinned-baselines + log.md) BEFORE T2.1. Launch **Tier S** (S.1/S.2) concurrently under MPS alongside Tier 2. Per-tier advisor + summary + STOP-for-user at the T1.4/Tier-S boundary, per workflow §4.

---

## 2026-06-03 — Refinement: T1.4 expanded to the FULL tuned-incumbent ceiling (the "floor"); Tier S = new-archs only

**Phase**: Design refinement of the same-day addition above (no experiments run).

**Why.** User asked: where is the *sweep + hypertuning of the CURRENT heads* (per task), as a **floor** to compare Tier S against? The first cut had T1.4 as a *bounded* hardening (α + top losses + one probe) and the deep work in Tier S — which left the *fully-HP-tuned current head* as nobody's first-class job.

**Advisor verdict (decisive — overruled my first proposal).** I floated "add a walled-off S.0 inside Tier S." The advisor pushed back: **expand T1.4 into the full tuned-incumbent ceiling instead.** The discriminating argument: T2's promotion gate is literally "fraction of the composite (d) recovered." If T1.4 freezes a bounded/under-tuned ceiling and the truly-tuned incumbent (found later) is higher, the gate is **sandbagged** — the dual-tower clears a soft target and T3/T4/T5 build on a go/no-go that evaporates once the real ceiling is known. Under-tuning the baseline you measure everything against is a *measurement error*, not "discipline." ("Bounded" was only ever about not letting the *new-arch search* balloon — never about under-tuning the ceiling.) I switched to the advisor's design; it fulfils the user's "floor" intent better.

**Resulting clean split (de-dups a thing the first cut had wrong):**
- **T1.4 = full per-task HP tune of the CURRENT heads = the floor.** Reg (`next_stan_flow`): α re-tune per-state (incl. α→0 at large, per T1.3 FL-drag) + tail-loss + ranked HP axes (depth/dim/dropout/LR/warmup). Cat (`next_gru`): loss calibration (logit-adj+focal+label-smoothing, train-only prior) + ranked HP axes (pooling/dropout/LR/warmup). **Cat loss work lives HERE** (was duplicated in T1.4 *and* S.2). Two-phase: search at FL+AL, validate at FL+AL+AZ 5f + GE confirm. The tuned best **SETS + FREEZES (c)/(d)**.
- **Tier S = NEW archs only**, scored vs the frozen tuned floor. S.1 reg (SASRec/Mamba/contrastive/registry — the SASRec probe moved OUT of T1.4 to here, it's a new arch). S.2 cat NEW encoder only (attention-pooling/TCN). **S.3 (new) = no-run synthesis** = the user's "huge picture": per-task ladder default / tuned-current / best-new-arch, explicitly labelled a **promotion test (unequal HP budget), not an equal-budget tournament**, with a **lose-within-noise → second-round-tune safeguard** so the budget asymmetry can't bury a real winner.
- **(d)-arm consistency** raised one level: if the reg tune raises STL-HGI reg, apply it to (d)'s HGI-reg arm too (or declare it un-tuned by design) — under-tuning it keeps T2's bar artificially low.

**Also folded in the earlier final-advisor fixes** (same session): freeze-rule now distinguishes the immutable track-internal yardstick from the §0.1 paper baseline (refreshable at T6.2); T1.1/T1.2 marked provisional-until-T1.4.

**Files touched.** `INDEX.html` (T1.4 rewritten; S.2 → encoder-only; new S.3 card; TOC/metric-map/chain/why-order/audit/future-works/Tier-1-callout all re-propagated). `AGENT_PROMPT.md` (exec order T1.4 + Tier S). `reg_head_architecture_sweep.md` (pointer). 2 commits on branch `mtl-improvement-stl-frontier`.

**Chain status**: unchanged from the entry above — T1.4 is the remaining Tier-1 gate before T2; Tier S parallel. Chain preserved.

**Next**: implementing agent runs the **full T1.4 tune** (sets+freezes (c)/(d)) before T2.1; launches Tier S (S.1/S.2 → S.3) under MPS alongside Tier 2. Advisor + summary + STOP-for-user at the boundary.

---

## 2026-06-03 — Callback: Tier S grounded in the REAL coded registry (test what we have) + new-arch prong kept

**Phase**: Design correction of Tier S (no experiments run).

**Why.** User: "we have heads already coded in src/* and we are not testing them — eval them, and eval if your changes are valid or you'd callback something." Correct instinct — a real callback.

**Code inventory (Explore + grep, authoritative).** `src/models/registry.py` decorator registry holds **~17 `next_*` heads** and **9 `category_*` heads**. Canonical: only `next_gru` (cat) + `next_stan_flow`/`next_getnext_hard` (reg). **Coded but never run:** `next_lstm`, `next_hybrid`, `next_temporal_cnn`, `next_tcn_residual`, `next_conv_attn`, `next_single`, `next_transformer_optimized`, `next_transformer_relpos`, `next_stan` (no-prior), `next_getnext` (soft-probe), `next_tgstan`, `next_stahyper`, `next_stan_flow_hsm`. **The plan named FICTIONAL heads:** `next_stan_baseline` (→ real `next_stan`), `next_transformer_pf` (→ `next_transformer_relpos`), `next_sasrec` (→ already exists as `next_transformer_relpos`/`_optimized`/`next_single`), `next_mixture_softmax` (→ `next_stan_flow_hsm`), `next_mamba` (genuinely absent).

**Callback applied.** Tier S was framed around "BUILD SASRec/Mamba" + a partly-fictional candidate list. Corrected to **Prong A = sweep the already-coded `next_*` heads** (real registry names, near-zero implementation cost — the user's point). Fixed the same fictional names in the pre-existing T5.1/T5.2 lists.

**Advisor pass (this round).** Three corrections: (1) **BLOCKING — exclude the `category_*` family.** Verified in code: `category_*` take flat `[B,D]` (`category_ensemble(input_dim,…)`), but the cat task feeds a 9-step sequence (`next_gru` masks over `[B,9,D]`), AND `generate_category_input` **rejects check-in-level engines like Check2HGI** (`src/data/inputs/builders.py:60-75`). Wrong modality + substrate → out (would need an adapter = new code for a different task). (2) "coded ≠ free" — bit-rot + the registry silently drops unknown kwargs (a log_T-aware head can train while ignoring the prior) → **unit-test gate on every head** + a **screen-then-promote funnel** (AL×1-seed screen → top-2-3 to full FL+AL+AZ 5f + GE). (3) advisor wanted Mamba demoted to future-work.

**User override (follow-up message).** "About the new archs you proposed, besides we have similar head, let's keep exploring them, don't callback." → **Prong B (faithful SASRec / Mamba / SimGCL) KEPT in Tier S** (overrides advisor #3), even where a similar coded head exists, so we can compare faithful-new vs coded-similar. Full-pipeline next-POI models (GETNext/STHGCN/…) stay OUT (need raw side-info). `category_*` stays OUT (hard constraint, not overridden).

**Net Tier S = two prongs:** A (coded registry sweep) + B (new-arch builds), both per task, both through the unit-test gate + funnel; S.3 "huge picture" now a 4-rung ladder (default / tuned-current / best-coded / best-new). **My T1.4/Tier-S STRUCTURE is valid (advisor confirmed); the callback was the candidate LIST.**

**Files touched.** `INDEX.html` (Tier S intro → two prongs; scope-boundaries callout rewritten; S.1/S.2/S.3 real names + funnel + category_* exclusion; T5.1/T5.2 real names; hard rule 16; TOC; chain; metric map). `AGENT_PROMPT.md` (Tier S exec order + hard rule 16). 1 commit on branch `mtl-improvement-stl-frontier`.

**Chain status**: unchanged — T1.4 remaining Tier-1 gate; Tier S parallel. Chain preserved.

**Next**: implementing agent — Prong A first (unit-test + screen the coded `next_*` heads), Prong B builds after; both feed T5. Advisor + summary + STOP-for-user at the boundary.

---

## 2026-06-03 — Tier S: add S.3 (analyze + compose) ; renumber synthesis → S.4

**Phase**: Design addition (no experiments run).

**Why.** User: worth adding a step in S.1/S.2 to (a) eval results via advising + agents and **compose/merge approaches** (does one head's mechanism help another?), and (b) have an agent **study the best approach per task**. Asked me to eval carefully.

**My eval (with pushback).**
- **Compose step — ACCEPT, but it's a forking-paths trap without guardrails.** Composition is how orthogonal gains stack (best encoder ⊕ best prior ⊕ SimGCL aux — the lit scan already had SimGCL as "bolt onto the winning encoder"). But "merge approaches" can balloon into an unbounded grid, and composing many compounds on the same val folds + reporting the max is selection bias (Cawley-Talbot, already cited in the selector memo); plus heavy non-additivity history here (PLE/MMoE). → add it gated.
- **Per-task deep-study agent — ACCEPT NARROWED.** A broad upfront SOTA survey mostly re-confirms the lit scan's deflationary verdict, and the regime (not the STL head) is the wall. → fold it in as a *results-triggered analyst* (interpret WHY winners won on our substrate + propose orthogonal compositions), NOT a separate broad survey.

**Advisor pass.** Affirmed both calls (esp. folding #2 into #1 as a results-triggered analyst). Two corrections, applied:
1. **Keep S.3 LIGHT — don't over-build the guardrail.** My 4 mitigations partly duplicate the existing Tier-S gate (≥0.5pp multi-seed Wilcoxon + lose-within-noise→second-round). Reuse it; the ONLY new rule is "name 2-3 orthogonal compositions (encoder ⊕ aux ⊕ prior, never two encoders) BEFORE seeing the max." No parallel pre-registration/held-out ceremony.
2. **Frame S.3 output as an STL candidate-generator for T5**, not a destination — the composition that matters for deploy is the MTL one; the STL compose is a cheap pre-filter, re-judged under MTL (consistent with the moving-baseline guard).

**Cumulative-scope flag (advisor — taking to the user).** Tier S has grown bounded→full across this session (S.1/S.2 → +S.3 compose +S.4 synthesis → Prong A/B → analyst agent). Each addition is individually guarded, but Tier S is now the most intricate branch — and it's the off-critical-path STL branch the regime finding predicts washes out in MTL. Not a compute flag (it's cheap + MPS-parallel); an attention + new-code-risk flag (Prong-B builds + compose = fresh code on the least-headline branch). The "huge picture per task" is reviewer-useful + scientifically real + user-requested, so this can be the right call — but it should be ONE explicit user decision (keep growing vs cap here), not item-by-item accretion. **Surfacing it to the user now.**

**Structure now:** S.1 reg search (coded+new) · S.2 cat search (coded+new) · **S.3 analyze + compose (gated, results-triggered analyst)** · **S.4 synthesis "huge picture" (5-rung ladder)**. Existing synthesis card renumbered S.3→S.4. The analyst agent is execution-time (results-triggered) — NOT spawned now (no data yet).

**Files touched.** `INDEX.html` (new S.3 compose card; synthesis → S.4; intro, TOC, metric map, rungs). `AGENT_PROMPT.md` (Tier S card list). 1 commit on branch.

**Chain status**: unchanged — T1.4 remaining Tier-1 gate; Tier S parallel. Chain preserved.

**Decision (user 2026-06-03):** cumulative-scope question answered → **"keep it open to grow."** Tier S is an explicit open exploration sandbox — executor may add S.5/S.6 cards as ideas surface, no re-scoping checkpoint, PROVIDED each clears the unit-test + promotion gates, obeys the moving-baseline guard + scope boundaries, and is logged. Recorded as a callout in INDEX Tier-S intro + AGENT_PROMPT. Caveat carried: Tier S runs parallel to Tier 2-4 and must not starve the regime work (the headline).

**Next**: implementing agent: T1.4 (full tune → freeze (c)/(d)) → Tier S (Prong A screen → Prong B builds → S.3 analyze+compose → S.4 picture), parallel with Tier 2-4 under MPS. Advisor + summary + STOP-for-user at the boundary.

---

## 2026-06-03 — T1.4 harness built + sweeps launched; Tier-S Prong-A unit screen

**Phase**: T1.4 in flight (closing Tier 1). User decisions this session: **(1) loss scope = full design as written** (implement logit-adjust + focal + tail-loss, leak-guarded + unit-tested); **(2) Tier S = concurrent** Prong-A screening under MPS, hold promotion until the floor freezes.

**New code (committed, unit-tested).**
- `src/losses/calibrated.py` (`CalibratedLoss` + `build_calibrated_loss`): Menon **logit-adjustment** (`tau·log P_train(y)`), **focal**, **label-smoothing**, and the imbalance axis `tail_mode ∈ {balanced, cb, ldam}` (sklearn-balanced / Class-Balanced / LDAM margins). **All class statistics come from the TRAIN split only** (factory takes `y_train`) — F49-class leak guard. At all-default knobs it is bit-identical to `CrossEntropyLoss`. 19 unit tests (default==CE, train-only prior, `balanced==compute_class_weights`, finite grad, 1.5k-class region scale).

**Two integration points (one per arm) — discovered by a substrate sanity check.**
- **reg → `p1_region_head_ablation.py`** (the tool T1.3 used; loads v14 region-emb via `--region-emb-source`). Calibrated loss wired into the per-fold criterion; new `--focal-gamma/--logit-adjust-tau/--tail-loss/...` flags. v14 added to `--engine-override` choices.
- **cat → `train.py --task next` / `next_cv.py`** (the tool that produced the T1.1 cat ceiling). **Why not p1 for cat:** p1's trainer lands **macro-F1 ~16pp low** (14.88 plain / 23.29 balanced vs the 39.13 ceiling) — its AdamW LR/scheduler/wd diverge from `next_cv.py` (default_next `max_lr=1e-2`). So the cat tune MUST run in `next_cv.py` for an apples-to-apples baseline. Wired via a new `ExperimentConfig.loss_calibration` dict (empty → legacy path) built from TRAIN-fold targets. **Validated:** calibrated `balanced` reproduces the ceiling exactly — AL cat 5f/50ep **macro-F1 38.87±1.34 == T1.1 39.13**.

**Sweeps launched** (`scripts/mtl_improvement/t14_sweep.sh <reg|cat> <state>`, seed=42, 5f×50ep, frozen folds, seeded log_T). Phase-1 search at **AL+FL** (4 background arms concurrent on the A40): reg = alpha=0 branch × {ldam0.5/0.3, cb, dropout0.1/0.5, d_model256, lr1e-3, combo} + one tail-with-prior sanity arm; cat = {balanced baseline, logit-adjust τ0.5/1.0, focal, label-smoothing, combos, LR/dropout points}. Winner validated at **AZ+GE** next. Agg: `t14_agg.py`.

**Early AL signal (partial).** reg: R1 α=0 (62.88) ≥ R0 default-prior (62.32) > LDAM (61.11) — **tail-loss HURTS reg at AL so far**; consistent with the T1.3 side-finding that the strong v14 embeddings don't want extra imbalance machinery. cat: balanced baseline 38.87 pinned; calibration arms pending.

**Tier-S Prong-A unit screen** (`tierS_unit_screen.py`, hard rule 16c "coded ≠ working"): **15/17** coded `next_*` heads build+forward+backward clean; the 2 `*_hsm` heads need a prebuilt `hierarchy_path` artifact (deferrable, not bit-rot). No bit-rot in the core candidate set → Prong-A GPU screens can proceed once the T1.4 AL arms free GPU capacity. Promotion/scoring HELD until (c)/(d) freeze.

**Chain status**: Tier 1 closing (T1.4 in flight); chain preserved. Tier S branched off, screening concurrently, promotion gated on the frozen floor.

**Next**: monitor the 4 sweep arms → `t14_agg.py` to pick per-task winners at AL+FL → validate winners at AZ+GE → T1.4d freeze (c)/(d) + update INDEX/TIER01_RESULTS → T1.4e advisor + Tier-1-close summary + STOP for user. Tier-S GPU screens after AL arms clear.

---

## 2026-06-03 — T1.4 CLOSED → (c)/(d) FROZEN → TIER 1 COMPLETE (advisor pass + STOP for user)

**Phase**: **Tier 1 COMPLETE.** T1.4 closed; the track-internal yardstick (c)/(d) is frozen.

**Winners (scale-robust — a single config per task wins at all 4 states, no per-state branching):**
- **reg = `next_stan_flow`, α=0 (log_T prior OFF)**, default HP, no tail-loss.
- **cat = `next_gru`, logit-adjustment τ=0.5** (Menon), no class-weighting / focal / label-smoothing.

**Frozen (c) STL-on-v14** (Δ vs pre-T1.4): cat AL 41.86(+2.73)/AZ 50.44(+7.28)/GE 59.57(+5.55)/FL 69.99(+4.11); reg AL 62.88(+0.56)/AZ 55.11(+2.24)/GE 58.45(+2.64)/FL 73.31(+3.03).
**Frozen (d) composite** (reg arm = max(v14-α0,HGI-α0), both hardened): reg AL 63.58/AZ 55.11/GE 58.76/FL 73.62; gap over MTL-deploy **+12.4 to +17.3pp** (grew vs pre-T1.4 → T2's bar correctly RAISED, not sandbagged).

**Findings.** (1) Dropping the log_T prior wins reg everywhere (the T1.3 FL side-finding generalises, grows with scale). Tail-loss (LDAM/CB), dropout, lower-LR all lose decisively (CB→52); d_model=256 a tie. (2) logit-adjust τ=0.5 lifts cat macro-F1 +2.7 to +7.3pp (biggest at the most-imbalanced AZ); logit-adjust ALONE wins — stacking with class-weighting (AL 30.15) or focal/combos (24.34) over-corrects and craters. (3) With BOTH substrates α=0-hardened, v14 and HGI STL-reg are statistically indistinguishable (every Δ < 0.5σ) → HGI's reg edge was a log_T-prior artifact → the two-substrate composite is no longer reg-motivated → **strengthens the regime thesis** (substrate axis exhausted; the gap is architectural).

**New code (all committed, leak-guarded, unit-tested).** `src/losses/calibrated.py` (19 tests, train-only class stats) wired into TWO harnesses — reg via `p1` (loads v14 region-emb), cat via `next_cv.py` (`ExperimentConfig.loss_calibration`; the tool that made the T1.1 cat ceiling — p1's trainer lands cat ~16pp low). Drivers `t14_sweep.sh` / `t14_validate_azge.sh` / `t14_hgi_hardening.sh`; agg `t14_agg.py`. 19 reg JSONs committed; cat config→rundir manifests persisted to `docs/results/mtl_improvement/t14_manifests/`.

**Advisor pass (sub-agent, 2026-06-03) — verdict: freeze fundamentally SOUND; 3 fixes applied.**
- **A LEAK: SOUND** — logit-adjust offset applied train-time-only (`calibrated.py:121`), eval path has no offset (`shared_evaluate`); counts from train fold only (`next_cv.py:127` / p1 `y_train`). The +7.28 AZ jump is a legit Menon effect, not a leak.
- **B ceiling legitimacy: SOUND** (α=0 is a buffer, can't drift; ceiling = STL-best is right) — caveat documented: (c)-reg is prior-OFF vs MTL's prior-ON, don't misread the gate as prior-matched.
- **C cross-tool: SOUND** — balanced reproduced **within 0.26pp** (38.87 vs 39.13), relabeled (not "exact").
- **D pruning: NEEDS-FIX (applied)** — d_model=256 was the TOP AL arm (62.55±4.43, ~0.07σ) and a tie at FL, not "lost"; relabeled "tied within σ; base retained on parsimony." Ceiling value unaffected.
- **E substrate: OVER-READ (applied)** — "marginal surviving edge" oversold noise; reframed to "statistically indistinguishable (all Δ<0.5σ); HGI edge = log_T-prior artifact" everywhere (INDEX T1.2/T1.4 + TIER01). This is more correct AND better for the regime story.
- **F config sanity: SOUND** — end-to-end wiring verified; craters are real over-corrections (not NaN/bug).
Optional items applied: durable manifests; prior-mismatch + 0.26pp caveats in TIER01. (Deferred: dumping loss_calibration into each run dir — manifests cover provenance.)

**Tier-S Prong-A** (concurrent, per user decision): 15/17 coded `next_*` heads pass the build+fwd+bwd unit screen (`tierS_unit_screen.py`) — no bit-rot; 2 `*_hsm` heads need a prebuilt hierarchy artifact (deferrable). Promotion HELD until now (floor frozen) → GPU screens can proceed against the frozen (c)/(d).

**Chain status**: **Tier 1 COMPLETE; (c)/(d) FROZEN (immutable yardstick).** Chain preserved. T2 is now correctly gated against a tuned, advisor-audited ceiling.

**STOP for user (mandatory tier-boundary cadence).** Surfacing the Tier-1 close for a go/no-go decision on what's next: (a) start **Tier 2** (T2.1 dual-tower, the centerpiece) against the frozen ceiling; (b) run the parallel **Tier S** GPU screens (coded-head sweep) first/concurrently; (c) run the ungated **T4.0** loss-scale/RLW litmus (cheap, flagged run-early). Do NOT auto-roll into Tier 2.

**Next**: await user direction on (a)/(b)/(c). Whichever is chosen, scored against the FROZEN (c)/(d).

---

## 2026-06-03 — Tier S started (user choice) → caught + fixed a CAT CEILING BUG; (c)/(d) re-pinned

**Phase**: Tier S Prong-A screens (user chose "Tier S GPU screens first" at the Tier-1 boundary). The screen surfaced a bug in the just-frozen cat ceiling; fixed and re-pinned before any Tier-2 work.

**THE BUG.** `train.py --cat-head X` is **silently ignored on `--task next`** — it only takes effect on the MTL `is_check2hgi_track` path (`scripts/train.py:1700`). For plain `--task next` the model is `config.model_name` (default `next_single`). So the **entire cat ceiling (T1.1 + the first T1.4 freeze) ran `next_single`, not `next_gru`**, despite `--cat-head next_gru`. **How it surfaced:** the S.2 cat-encoder screen ran 8 encoders via `--cat-head` and they ALL returned an identical 41.86 — an impossible tie; the arch dump showed `NextHeadSingle`. **Tell-tale that was sitting in the data the whole time (incl. through the advisor pass):** the mis-pinned AL cat "ceiling" 41.86 was BELOW the MTL deployable cat 46.50 — a real STL ceiling can't be under the MTL it bounds.

**THE FIX.** `--cat-head`→`--model` in `t14_sweep.sh` + `tierS_screen.sh` (the flag honoured on `--task next`); new `t14_cat_repin.sh` re-ran the cat ceiling with the actual `next_gru` at all 4 states. **Loss winner (logit-adjust τ=0.5) holds on the corrected model** (la05 > la10 > balanced everywhere; combo craters). Reg is UNAFFECTED — p1 honours `--heads` (per-config reg numbers differ; verified).

**RE-PINNED (c)-cat (real next_gru la05):** AL 49.97 / AZ 51.01 / GE 58.12 / FL 69.97 (was 41.86/50.44/59.57/69.99). Correction is **scale-dependent**: AL +8.11, AZ +0.57, GE −1.45, FL −0.02 — large at small states, ~0 at FL. Sanity restored: (c)-cat > MTL-cat at every state. (d)-cat arm updated to match; (d)-reg arm + (c)-reg unchanged. INDEX T1.1/T1.2/T1.4 + pinned callout + TIER01 all updated.

**S.1 reg screen (Prong A) — clean NEGATIVE (reviewer-proof).** No coded reg head beats the frozen α=0 floor (AL 62.88): `next_stan` 62.88 (== floor) + `next_tgstan` 62.84 tie; everything else below. `next_stan` == floor confirms `next_stan_flow α=0 ≈ next_stan` (α=0 zeros the prior path). The tuned incumbent reg head IS the STL reg ceiling.

**S.2 cat screen — INVALID first run (the bug), re-running with `--model`.** First run was all `next_single` (the bug). Re-launched corrected. Known points so far: `next_gru` 49.97 (floor), `next_single` 41.86 (AL) but BEATS next_gru at GE (59.57 vs 58.12) — a state-dependent S.2 candidate. Full corrected screen in flight.

**Lesson (for the next agent).** Two silent-flag traps now confirmed in this codebase: (1) loss-calibration kwargs were fine (wired through config), but (2) `--cat-head`/`--reg-head` are MTL-track-only — for STL `--task next` use `--model`. Always verify the arch actually ran (`results/.../model/arch.txt`), and **sanity-check that any STL ceiling sits ABOVE the MTL it bounds.**

**Chain status**: Tier 1 re-frozen (cat corrected); chain preserved. The frozen (c)/(d) now sit on the correct heads + pass the ceiling>MTL sanity check.

**Next**: finish the corrected S.2 cat screen → rank vs the re-pinned floor → record S.2 verdict. Then await user direction on Tier 2 vs more Tier S vs T4.0. A re-advisor on the cat re-pin is warranted at the next boundary (the first advisor pass missed the ceiling<MTL tell-tale).

---

## 2026-06-03 — Re-advisor on the cat re-pin (user-requested) — re-pin SOUND; 1 LOW anomaly + standing guard added

**Phase**: integrity re-audit of the cat re-pin (the first advisor missed the bug, so user chose a focused re-advisor before Tier 2).

**Verdict: re-pin is SOUND (items 1-8).** Sub-advisor verified against the actual files: (1) all 4 `g_la05` rundirs have `model/arch.txt` = `NextHeadGRU` (not NextHeadSingle); (2) numbers reconcile to the JSONs (AL 49.97 / AZ 51.01 / GE 58.12 / FL 69.966); (3) (c)-cat > MTL-deploy-cat at every state; (4) la05 ≥ la10 ≥ balanced + combo craters (loss winner valid on next_gru); (5) the logit-adjust knob is live (train.py:1156→next_cv.py:126→calibrated.py; balanced≠la05 proves it); (6) leak guard intact (train-only prior); (7) reg unaffected (p1 honours --heads — distinct per-head numbers); (8) freezing (c)-cat on next_gru (not next_single, which only wins at GE) is the correct incumbent-ceiling choice → next_single relegated to S.2/T5.

**One LOW anomaly found (sibling sanity hunt).** (c)-cat vs MTL **diagnostic-best** cat: +3.19/+2.26/+1.05 at AL/AZ/GE but **−0.29 at FL** (69.97 < 70.26). NOT a bug recurrence (arch=GRU); a seed/metric confound (seed42 single-seed + deployable basis vs multi-seed + oracle epoch), ~0.34σ — a tie. The written claim ("(c)-cat > MTL **deployable** cat") holds. → footnoted in TIER01 §(c); optional FL multi-seed (c)-cat re-run deferred.

**Standing guard added (the re-advisor's MED fix — the cheap check that would have caught the original bug).** `scripts/mtl_improvement/t14_freeze_sanity.py`: (1) ARCH CHECK — reads `model/arch.txt` for each cat-ceiling rundir, asserts head == NextHeadGRU (catches the silent-flag class of bug instantly); (2) ORDERING CHECK — asserts (c)-cat ≥ MTL-deploy-cat, (c)-reg ≥ MTL-deploy-reg, (d)-reg ≥ (c)-reg at all states; (3) INFO — (c)-cat vs MTL-diag-best (flags the FL note). **Runs GREEN: all hard checks pass.** Adopt as a post-freeze gate.

**Process lesson (meta).** The bug reached freeze + passed the first advisor because the cat ceiling and the MTL board come from DIFFERENT harnesses with different flag semantics (`--cat-head` is a no-op on `--task next`), and no one asserted the cross-harness invariant "STL ceiling ≥ the MTL it bounds" or verified the head that actually ran. Both halves are now encoded in `t14_freeze_sanity.py`.

**Chain status**: Tier 1 FROZEN + re-audited + guarded; chain preserved. (c)/(d) trustworthy; T2 correctly gated.

**Next**: await user direction — Tier 2 (T2.1 dual-tower) vs S.2 multi-state confirm (next_lstm/next_single) vs Prong B new-arch builds vs T4.0. Run `t14_freeze_sanity.py` after any future ceiling change.

---

## 2026-06-03 — Tier-S Prong-A COMPLETE: multi-state confirm → NO head change (reviewer-proof negative)

**Phase**: Tier-S Prong-A closed. User asked whether the Prong-A screen warrants a head change; ran the multi-state confirm to decide before Prong B.

**cat — next_lstm confirmed (5f/50ep, logit-adjust τ=0.5) vs the next_gru floor:** AL 49.76 (Δ−0.21) / AZ 51.49 (Δ+0.48) / GE 58.63 (Δ+0.51) / FL 70.11 (Δ+0.14). Ties next_gru within fold-σ at every state; no ≥0.5pp multi-seed win, and ~5× slower. **next_single** (all-state from the re-pin, since the old mis-pinned ceiling WAS next_single): wins ONLY at GE (+1.45), loses −8 at AL → fails the multi-band gate.

**reg — cross-check:** S.1 AL screen was a clean negative (no head beat the α=0 floor; next_stan==floor ≡ next_stan_flow α=0). Scale-check: next_tgstan tied at AL (62.84) but loses FL (72.20 vs 73.31, Δ−1.11).

**Decision: NO head change for either task.** The tuned incumbents (`next_gru` cat, `next_stan_flow α=0` reg) are the STL ceiling across ALL coded heads, multi-state — a reviewer-proof Prong-A negative, consistent with the regime finding (the head is not the lever). next_lstm logged as a co-equal fallback only; no re-run needed (no candidate warranted a change). INDEX S.2 block + manifests updated.

**Chain status**: Tier 1 frozen+guarded; Tier-S Prong-A complete (negative). Chain preserved.

**Next** (user directive): proceed to **Prong B** — genuinely-new arch builds. SASRec is redundant (≈ next_transformer_relpos, already screened → loses for cat). Focus the build on the non-redundant candidates: a Mamba/SSM encoder + SimGCL auxiliary (bolt-on). Low-EV per Prong A + the regime finding, but in scope (user: keep exploring). Each clears the unit-test gate; scored vs the frozen floor; feeds T5 only.

---

## 2026-06-03 — Tier-S Prong-B: built Mamba-lite (next_mamba) → loses both tasks → Tier-S NEGATIVE complete

**Phase**: Prong-B new-arch build. Tier S (Prong A + B) now concludes.

**Built `next_mamba`** — a dependency-free pure-torch **selective state-space (Mamba-lite)** head (`src/models/next/next_mamba/head.py`): input-dependent Δ/B/C selectivity + SiLU gate + diagonal A + D skip; sequential scan (L=9, no kernel needed). Passed the unit-test gate (build+fwd+bwd finite, [B,nc] out, 325k/608k params). mamba-ssm not installed; SASRec skipped (redundant with next_transformer_relpos, already screened).

**AL screen — Mamba LOSES both:** cat 41.44 vs next_gru 49.97 (**−8.53**, lands in the transformer cluster ~41-42); reg 61.66 vs next_stan_flow α=0 62.88 (**−1.22**, like the other generic encoders). The SSM behaves like the transformers, not the recurrent winners → fails the AL screen → no multi-state promotion.

**Tier-S synthesis (the "huge picture", S.4) — reviewer-proof NEGATIVE.** Per task, the tuned T1.4 incumbent wins:
- **cat:** recurrent (GRU≈LSTM ~50) ≫ transformer/SSM (~41-42) ≫ CNN (~33-37). No encoder beats tuned next_gru.
- **reg:** STAN-family (next_stan_flow α=0 ≡ next_stan) is the ceiling; every generic encoder (LSTM/SSM) ~61-62 below.
S.3 (compose) NOT triggered (nothing promoted). **Conclusion: the STL head is NOT the lever** — the gap is architectural, in the JOINT MTL dynamics (the regime finding), not the per-task head. The frozen (c)/(d) stand; T5 inherits next_lstm/next_single as co-equal fallbacks only.

**Deferred (low-EV):** a SimGCL auxiliary-loss bolt-on — the one orthogonal mechanism not probed. The encoder axis is exhausted + the regime finding both argue against EV; not run autonomously — flagged for the user.

**Chain status**: Tier 1 frozen+guarded; **Tier-S (Prong A + B) COMPLETE — reviewer-proof negative**. Chain preserved. The headline (the architecture gap) is Tier 2.

**Next**: STOP for user — the natural next is **Tier 2 (T2.1 dual-tower)**, where the gap actually lives. Optional remaining Tier-S item: SimGCL aux (low-EV). T4.0 loss-scale litmus also still open.

---

## 2026-06-03 — Tier-S Prong-B SimGCL aux (user-requested) → ties the floor → Tier S DEFINITIVELY complete

**Phase**: built + screened the one orthogonal non-encoder mechanism (SimGCL aux). Tier S now fully closed.

**Built `next_gru_simgcl`** — the incumbent GRU cat encoder + a **SimGCL auxiliary contrastive loss** (Yu SIGIR'22): norm-bounded sign-preserving noise → two views → InfoNCE consistency term, exposed via `model.aux_loss` and added by a guarded one-line trainer hook (`_single_task_train.py`; no-op for every other head). Unit-test gate PASS (aux in train, None in eval). Arch confirmed `NextHeadGRUSimGCL` with the contrastive projection (genuinely ran, not a silent fallback).

**AL screen (with logit-adjust τ=0.5) — TIES the floor:** cl_weight=0.1 → 50.03 (Δ+0.06 vs next_gru 49.97); cl_weight=0.05 → 49.89 (Δ−0.08). The contrastive regularizer does NOT move the cat ceiling.

**Tier S DEFINITIVELY complete — comprehensive reviewer-proof negative.** Prong A (all coded cat encoders + reg heads) + Prong B (new SSM `next_mamba` loses both; SimGCL aux ties) → **no head, encoder, OR auxiliary regularizer beats the tuned incumbent.** The per-task STL axis is exhausted across every mechanism class we have. The lever is the joint MTL architecture → Tier 2.

**Chain status**: Tier 1 frozen+guarded; **Tier S COMPLETE (Prong A + B, all mechanisms) — reviewer-proof negative.** Chain preserved.

**Next**: STOP for user — **Tier 2 (T2.1 dual-tower)** is the clear next (the architectural gap). T4.0 loss-scale litmus also open (cheap, ungated). New reusable assets from Tier S: `next_mamba`, `next_gru_simgcl`, the `model.aux_loss` trainer hook, `t14_freeze_sanity.py`.

---

## 2026-06-03 — Critical advisor on the new heads + per-task tuning + the STAN concern (user-requested)

**Phase**: meticulous re-evaluation (user: "the STAN baseline beats our head — we're missing something; eval the new heads, check we tune per-task, search lit / compose"). Advisor sub-agent + targeted re-runs.

**Advisor verdict (grounded, read the code).**
1. **New heads SOUND.** `next_mamba` selective-SSM math is faithful (correct ZOH discretisation, diagonal recurrence, selectivity, SiLU gate, D skip) — its −8.5pp cat loss is a FAIR result, not a bug. `next_gru_simgcl` SimGCL is faithful + the aux-loss trainer hook is leak-free (val path clean) — the +0.06 tie is real. **→ drop both** (sound but lose/tie).
2. **Per-task tuning gap CONFIRMED (the user was right).** The Tier-S screens ran every challenger at REGISTRY DEFAULTS with one LR, never the per-arch LR mini-sweep that hard-rule 7 mandates. "No head beats the incumbent" was under-evidenced for the near-ties.
3. **STAN concern — category confusion + a real gap.** Our reg ceiling (`next_stan_flow α=0`) *IS* the STAN head; "faithful STAN" on raw features scores ~28pp LOWER (not comparable). So no STAN beats our reg head on a matched embedding. BUT the advisor surfaced a genuine zero-cost untested opportunity: **the cat screen excluded the STAN family — our cat head is a plain attention-free GRU, while STAN-attention WINS reg.** STAN-for-cat was never tried.

**Decisive re-run — STAN-for-cat, FAIR per-arch tune (the user's core hypothesis tested).** `next_stan` as cat head (the attention backbone), logit-adjust τ=0.5, AL 5f/50ep, LR×d_model sweep (the hard-rule-7 axes the screen skipped): best (d256, lr1e-2) = **42.60**, all 6 configs **38.6–42.6 — i.e. −7.4 to −11.3pp BELOW the GRU floor 49.97.** The per-task tune did NOT rescue it. **The user's hypothesis (STAN-attention helps cat) is FALSIFIED, decisively, even tuned.**

**Mechanistic insight (the real answer).** For next-**category** (7-class, short window): **recurrence (GRU/LSTM ~50) ≫ attention (STAN/transformer/SSM ~41) ≫ CNN (~34).** For next-**region** (many-class): STAN-attention wins. The best architecture is **genuinely task-dependent** — GRU for cat, STAN for reg — which is exactly what the frozen ceilings already use. The intuition "STAN should help cat" is reasonable but empirically wrong here.

**Composition #2 (GRU⊕STAN-attention) SKIPPED** — the advisor gated it on "STAN-attention being competitive for cat"; at −7.4pp it is not, so a fusion with the weak attention branch won't help. Reg GCN-term #3 deferred (low-EV, gated). `next_lstm` per-task tune (the one default-tie) run to close the hard-rule-7 gap on the only near-tie [result in next entry].

**Chain status**: Tier 1 frozen+guarded; Tier S negative now HARDENED by a fair per-task tune of the top candidate. Chain preserved.

**next_lstm per-task tune (the one default-tie) — CLOSED, no win.** AL LR×dropout sweep (logit-adjust τ=0.5): lr1e-2 (default) = 49.76 at both dropout 0.3 and 0.1; lr3e-3 = 49.27. **Best tuned LSTM = 49.76 (the known −0.21 tie); no config beats GRU 49.97, dropout is neutral, lower LR is worse.** Combined with the STAN-for-cat falsification, the per-task-tuning gap is now fully closed for BOTH near-candidates — neither beats the GRU cat ceiling even when fairly tuned. JSON: `t14_manifests/lstm_cat_tune_alabama.tsv`.

**Investigation CLOSED.** All advisor HIGH/MED actions addressed: STAN-for-cat (falsified, −7.4pp even tuned), next_lstm tune (no win), composition #2 (skipped — gated), new heads (sound, dropped), STAN caveat (documented). The cat ceiling (next_gru τ=0.5) and reg ceiling (next_stan_flow α=0) are defended against the per-task-tuning + STAN critiques. The architecture is task-dependent (recurrence for cat, attention for reg) and the frozen ceilings already use the right head per task.

**Next**: back to the Tier-2 decision (T2.1 dual-tower) — the joint MTL architecture is the only remaining lever. Ceilings fully audited + fairly tuned.

---

## 2026-06-03 — PIPELINE DEEP-DIVE AUDIT (user-requested) → found a HIGH systematic cap: non-overlapping windows

**Phase**: two parallel auditor agents on the next-cat + next-reg pipelines (input formation, batching, loss, optimizer/scheduler, heads, metrics). User instinct: "maybe we are losing a piece in the big picture." Full report: [`PIPELINE_AUDIT_2026-06-03.md`](PIPELINE_AUDIT_2026-06-03.md).

**⭐ HIGH FINDING — the user was RIGHT.** `generate_sequences` (`src/data/inputs/core.py:40,56,59`) defaults `stride = window_size = 9` → **NON-OVERLAPPING windows**, the single chokepoint for both heads. Measured AL: 113,846 check-ins → **only 12,709 sequences** (the "~12,709 samples" everywhere is WINDOWS, not check-ins). Overlapping (stride=1) → **~96k–108k, a 7.5–8.4× increase** in (history→next) supervision. Non-overlapping is NOT standard for next-POI/next-cat. This caps BOTH ceilings identically, **independent of head** — exactly the systematic piece. **Leak-safe to fix** (VERIFIED: StratifiedGroupKFold(userid) keeps a user's windows in one fold → overlap can't cross train/val).

**Secondary (all cheap):** 58% of users dropped by MIN_SEQUENCE_LENGTH=5 (3.65% of check-ins; biases to heavy users); 13% "leftover-branch" rows are a different task (stride=1 fixes both for free); cat trainer runs fp16-autocast WITHOUT GradScaler + autocasts val (vs reg fp32); 50-ep OneCycle late-overfit (shorter ~25ep may lift both). **Everything else VERIFIED CLEAN** (padding, masking, label + region-label alignment, fold/class-weight leakage, loss double-application, pooling, α=0 path).

**FOUNDATIONAL caveat.** The windowing fix would change the frozen (c)/(d) ceilings, the MTL board, the per-fold log_T, AND the v11 PAPER CANON — all built on non-overlapping windows. NOT a silent swap. Plan: validate in ISOLATION first (separate input dir at AL, frozen substrate untouched, re-measure both ceilings with stride=1), then a strategic/paper-level decision if it lifts.

**Chain status**: Tier 1 frozen ceilings now have a known FOUNDATIONAL caveat (windowing); surfacing to user before any rebuild. Chain preserved (nothing changed on disk).

**Next**: STOP for user decision — run the isolated AL overlapping-window probe (validate the lift, ~code + rebuild + 2 retrains) vs note-and-defer vs proceed to Tier 2 on the current canon. This is potentially the most impactful finding of the track (paper-wide).

---

## 2026-06-03 — Overlap-window probe VALIDATED the audit finding: data-scarcity-dependent ceiling lift

**Phase**: isolated validation (user chose AL+FL + both tasks + cheap levers). Threaded a backward-compatible `stride` param; built a self-contained probe (`scripts/mtl_improvement/overlap_probe.py`) that runs overlap vs non-overlap through ONE harness matching next_cv/p1. Results doc: [`docs/results/mtl_improvement/overlap_window_probe.md`](../../results/mtl_improvement/overlap_window_probe.md).

**Result (control reproduces the frozen ceiling within ~1pp → faithful):**
| task | state | control (s9) | overlap | LIFT |
|---|---|---|---|---|
| cat | AL | 50.73 | 60.23 (s1) | **+9.50** |
| cat | FL | 71.10 | 72.40 (s2) | **+1.30** |
| reg | AL | 63.99 | 68.96 (s1) | **+4.97** |

**The lift is DATA-SCARCITY-DEPENDENT** — large at small states (AL cat +9.5 / reg +5.0), modest at large (FL cat +1.3). At FL (159k seqs) the model is near data-saturation; at AL (12.7k) 8.5× more (history→next) supervision helps a lot. Lifts are well beyond σ; leak-safe (user-grouped split keeps a user's overlapping windows in one fold). Overlapping windows are genuinely different prediction pairs, not duplicates.

**The user's instinct + the audit were RIGHT**: a head-independent systematic cap (non-overlapping windowing) was hiding ~5-9.5pp of small-state ceiling — invisible to all the Tier-S head-swaps (which is exactly why none of them moved the needle).

**FOUNDATIONAL / paper-touching.** The frozen (c)/(d), the MTL board, the per-fold log_T, and the v11 PAPER CANON are all non-overlapping. Open questions before any canon change: (1) real-pipeline re-validation (harness is ~+1pp optimistic); (2) **does overlap help MTL too, or only STL?** — if it lifts STL more than MTL at small states, the STL→MTL gap WIDENS (bears on the regime finding + the dual-tower story); (3) AZ/GE + FL-reg overlap not yet run.

**Chain status**: Tier-1 frozen ceilings now have a VALIDATED foundational caveat (windowing); nothing changed on disk (probe is isolated). Chain preserved.

**Next**: STRATEGIC DECISION for the user — adopt overlap as canon (major rebuild, changes paper, lifts small-state ceilings) vs document-as-headroom/future-work (keep consistent non-overlap canon) vs investigate-further (real-pipeline + MTL-with-overlap) before deciding. The cheap training levers (cat fp32, shorter schedule) remain un-run (minor vs this).

---

## 2026-06-03 — Overlap-window study COMPLETE: real-pipeline STL + MTL (user: real-pipeline then MTL then act)

**Phase**: full validation done. Built an isolated probe engine `check2hgi_dk_ovl` (v14 re-windowed stride=1; embeddings symlinked; frozen substrate untouched) + engine-aware region-seq plumbing (p1/train.py STL + MTL fold creator). Results: [`overlap_window_probe.md`](../../results/mtl_improvement/overlap_window_probe.md).

**REAL-PIPELINE STL (caveat resolved):** cat AL 49.97→59.74 (**+9.77**), reg AL 62.88→68.01 (**+5.13**) — matches the isolated harness (60.23/68.96).

**MTL-with-overlap (the decisive question) — cat RISING-TIDE, reg GAP WIDENS:** real cross-attn MTL, KD-off, same recipe, only windowing differs. cat 46.30→55.21 (**+8.92** ≈ STL +9.77); reg 54.54→55.05 (**+0.50** vs STL +5.13). The STL→MTL reg gap goes **8.34 → 12.96** with overlap. **The shared-backbone reg pathway cannot exploit the extra data; STL reg fully does.** → overlap STRENGTHENS the regime/dual-tower finding (more data makes the architectural bottleneck MORE visible), while cat is a clean rising tide across STL+MTL.

**Net.** The user's "are we losing a piece?" instinct found a real, validated, head-independent data-formation cap — AND running it through MTL converted it from "a threat to the ceilings" into EVIDENCE FOR the central thesis (the reg architectural bottleneck). Caveats: AL/single-seed/KD-off/prior-free-reg recipe (control uses the same recipe → windowing delta valid).

**Chain status**: Tier-1 frozen ceilings have a validated foundational caveat (windowing); the regime finding is strengthened, not threatened. Nothing changed on disk in the canonical substrate. Chain preserved.

**Next**: STRATEGIC DECISION — adopt-overlap-as-canon (rebuild) vs document-as-key-finding+future-work vs confirm-the-pattern-at-AZ/GE/FL first. Then back to Tier 2 (now even better motivated).

---

## 2026-06-03 — Overlap study CLOSED (user decision): document as key finding + future-work; KEEP non-overlap canon

**Phase**: closure. User: show MTL disjoint+joint; run overlap on HGI STL; document + keep consistency.

**MTL disjoint + joint** (AL, KD off, same recipe): cat joint 46.30→55.21 (+8.92) / disjoint 46.52→55.90 (+9.39); reg joint 54.54→55.05 (+0.50) / disjoint 53.47→54.46 (+1.00). **The reg-gap-widens result holds on BOTH selectors** — even at its per-task-best epoch the MTL reg can't absorb the overlap data.

**HGI-overlap STL reg** (overlapping seqs + HGI region emb): 63.58→**68.47 (+4.89)** ≈ v14's +5.13. With overlap, v14 (68.01) ≈ HGI (68.47) for STL reg — they stay tied (consistent with T1.4 v14≈HGI). The composite (d) reg arm rises uniformly too.

**DECISION (user):** document as a **key finding + future-work**, **KEEP the non-overlapping canon** for whole-study consistency (frozen (c)/(d), MTL board, log_T, v11 paper canon all non-overlap; every within-study comparison stays valid). NOT adopted (would need a multi-state rebuild + re-paper). Memo: [`docs/future_works/overlapping_windows.md`](../../future_works/overlapping_windows.md). All probe code is isolated (engine `check2hgi_dk_ovl`); the canonical substrate is untouched.

**Net result of the user's "are we losing a piece?" instinct:** found a real, validated, head-independent data-formation cap (~+5 to +9.8pp at small-state STL) that ALSO sharpens the central thesis (the MTL reg bottleneck can't use the extra data → gap widens). Recorded; canon unchanged.

**Chain status**: Tier-1 frozen ceilings + canon UNCHANGED (consistency kept); overlap is a documented future-work that strengthens the regime story. Chain preserved.

**Next**: back to the Tier-2 decision (T2.1 dual-tower) — now even better motivated (the reg-gap-widens-with-more-data result is fresh evidence for the architectural bottleneck). Cheap training levers (cat fp32, shorter schedule) + the overlap-at-AZ/GE/FL multi-seed confirm remain optional future items.

---

## 2026-06-04 — Audit close-out (O1–O5 from AUDIT_TIER1_TIERS_2026-06-03)

**Phase**: closing the 5 ranked audit items. Tier-1/Tier-S unchanged (these are confirmations + reporting, NOT new ceilings). Result files: `docs/results/mtl_improvement/{o1_alpha_probe.json,TIER01_RESULTS.md §Audit close-out}`.

**O1 — α=0 "prior is a drag": SETTLED, both audit hypotheses FALSIFIED.** Re-ran the prior-ON STL-reg config (`next_stan_flow`, learnable α init 0.1, prior ON, v14, frozen folds, seeded log_T, 5f×50ep s42) via a faithful probe (`o1_alpha_probe.py`, monkeypatches p1's `_build_head` to read the trained α — the model was never persisted). Replication exact (prior-ON reproduces 62.32/70.28/52.87/55.81).
- **Learned α converges LARGE and grows with scale**: AL +0.454 / AZ +0.789 / GE +0.944 / FL +1.095 (σ≈0.001–0.005). NOT ≈0 (the "optimization artifact" hypothesis) and NOT ≈0.1 (the "didn't drop the prior" hypothesis) — the model actively LEANS INTO the prior. Yet prior-ON is still 0.56/2.24/2.64/3.03 pp BELOW α=0 (AL/AZ/GE/FL), the gap also growing with scale.
- **The prior carries real signal**: standalone log_T Acc@10 = 50.86 (AL) / 66.15 (FL), validating against the authoritative Markov-1-region floors 47.01/65.05 (`AGENT_CONTEXT.md §184-185`; the legacy "~21.3%" is the degenerate POI-level markov, not comparable). So "α=0 wins" is NOT "transition priors are worthless."
- **Mechanism (sharper than the audit hoped): a train/val co-adaptation drag.** With the prior present the STAN encoder over-leans on the train-only log_T (α↑), and the crutch-dependent solution generalizes worse than the α=0 encoder forced to internalize transitions. Coherent with MTL-vs-STL (§2d): MTL log_T = KD loss on the shared rep (helps a starved backbone); STL log_T = additive logit bias (hurts a co-adapting head).

**Decision (O1)**: reframe the written claim to **"the fixed additive log_T prior is a net drag on the STL reg ceiling — learnable α leans into it yet generalizes worse than α=0."** NOT "embeddings subsume transitions," NOT a stuck-α bug. §2c HGI-prior-artifact corollary stands + strengthened. Applied to `TIER01_RESULTS.md §O1`, INDEX T1.4 verdict, AUDIT §6.

**O4 — Tier-S reporting closed.** `next_hybrid` recovered (AL cat 49.34 < 49.97 floor; it RAN — a reporting omission, not a dropped arm; all 8 S.2 encoders lose/tie → negative unchanged). `*_hsm` deferral (needs prebuilt `hierarchy_path`; not bit-rot) noted in INDEX §S.1.

**O5 — paper limitations carried in.** Added limitation (vi) to `PAPER_DRAFT.md §7 Beat 3` — non-overlapping windows under-supervise (≈8× fewer pairs); internal Δs apples-to-apples; AL rebuttal (MTL→STL reg gap widens 8.34→12.96) pre-empts the under-supervision reviewer attack; dense rebuild deferred to `future_works/overlapping_windows.md`. NOT shipped silently.

**O2/O3 — multi-seed cat: RUNNING.** `o2o3_multiseed_cat.sh` {0,1,7,100}: O2 = next_lstm + next_single cat at AZ+GE (vs floor AZ 51.01 / GE 58.12; ≥0.5pp at ≥2 bands → real T5.2 candidate, does NOT re-open (c)); O3 = FL next_gru cat (resolves the (c)-cat 69.97 vs MTL-diag 70.26 inversion). next_lstm ~5× slower (the long pole). Aggregator `o2o3_agg.py`; verdicts pending.

**Findings**: O1 numeric above. **Frozen (c)/(d) UNCHANGED** — O1/O2/O3 are confirmations/reporting; the immutable yardstick is untouched (O2 winners, if any, become T5.2 candidates only).

**Chain status**: Tier-1 + Tier-S frozen ceilings UNCHANGED; audit close-out is confirmatory. Chain preserved. T2.1 dual-tower remains the next decision.

**Next**: (1) when O2/O3 land, run `o2o3_agg.py`, fill the `<!-- O2O3_RESULTS_PLACEHOLDER -->` in `TIER01_RESULTS.md` + INDEX §S.2 multi-state-confirm verdict + AUDIT §6 O2/O3 rows, run `t14_freeze_sanity.py` (must stay GREEN), commit. (2) Then surface the closed audit to the user at the Tier-1→Tier-2 boundary (per the review cadence) for the T2.1 go decision.

---

## 2026-06-04 — O2/O3 multi-seed cat LANDED → audit close-out COMPLETE (O1–O5 all closed)

**Phase**: the O2/O3 multi-seed sweeps ({0,1,7,100}, ≈6 GPU-h) finished; audit AUDIT_TIER1_TIERS_2026-06-03 is now fully closed. Frozen (c)/(d) UNCHANGED; freeze-sanity GREEN.

**O2 — Tier-S cat crack closed (multi-band negative HOLDS, with one honest nuance).**
- **next_lstm**: the single-seed nominal wins EVAPORATE multi-seed — AZ +0.48→+0.25 (51.26±0.19), GE +0.51→+0.18 (58.30±0.31); with AL −0.21 / FL +0.14 it is a **tie at all 4 states**. "Failed to show a win" → **"shown no win."**
- **next_single**: GE win is REAL and robust — +1.45 single → **+1.54±0.17** multi-seed (59.66 vs floor 58.12). But GE-SPECIFIC (AL −8.11, AZ −0.03) → fails the ≥2-band gate. Per the audit's narrow rule it ENTERS the **T5.2 candidate set as a state-conditional option** (re-judged under MTL); **does NOT re-open frozen (c)** (moving-baseline guard — (c) GE-cat stays next_gru 58.12, the scale-robust incumbent). The absolute STL GE-cat best-over-heads is next_single 59.66, but (c) is the tuned-INCUMBENT ceiling by design.
- Net: no scale-robust head beats the incumbent → the reviewer-proof Tier-S cat negative stands at the multi-band level; one validated GE-specific candidate logged for T5.2.

**O3 — FL (c)-cat inversion resolved (not a single-seed artifact, not a bug).** Multi-seed FL (c)-cat = **69.96±0.08** validates the seed42 frozen 69.97 (agree 0.01pp). The inversion vs MTL diag-best (70.26) **PERSISTS** multi-seed (−0.30pp) — so it is NOT the seed artifact the audit guessed; it is tiny (~0.35σ) + explained (oracle epoch + small FL cat transfer; board Δcat≈0). (c) validly bounds the DEPLOYABLE MTL cat (69.96 ≫ 66.73). Freeze-sanity hard checks pass; the (c) footnote caveat is retained.

**Decision**: all 5 audit items CLOSED. No change to the frozen yardstick. The reframed O1 claim ("the fixed additive log_T prior is a net drag on the STL reg ceiling") + the O2 nuance (next_single GE-specific T5.2 candidate) + the O5 paper limitation are the substantive deltas.

**Advisor pass (2026-06-04, user-requested at the tier boundary)**: independent adversarial review (leak-checked the 4 surfaces — NONE found; verified the O1 probe is a bit-faithful re-run reproducing 62.32/70.28/52.87/55.81). Verdict: O1 SOUND-WITH-REVISION, O2/O3 SOUND. Corrections applied: (1) **O1 mechanism softened** — "co-adaptation drag" was 1 of 3 observationally-equivalent stories (train/val-gap vs additive scale-mismatch vs transition double-counting); reframed to phenomenology ("the fixed additive prior is a net drag on the STL-reg ceiling"; mechanism most-likely-but-not-isolated). (2) **"grows with scale" → "larger at higher-coverage states (n=4, suggestive)"** (not a fitted law; not load-bearing). (3) noted the α(epoch-50)/Acc@10(best-epoch) read mismatch (immaterial, σ_α tiny). (4) **O2 footnote** now states plainly that next_single 59.66 > (c) 58.12 at GE (per-state STL GE-cat ceiling is 59.66; (c) is the scale-robust incumbent, not the per-state max) — closes the "hidden ceiling" read. None reopen (c)/(d) or block Tier 2.

**Chain status**: Tier-1 + Tier-S frozen ceilings UNCHANGED; audit close-out confirmatory. Chain preserved. T2.1 dual-tower is the next decision.

**Next**: surface the fully-closed audit to the user at the Tier-1→Tier-2 boundary for the **T2.1 dual-tower go decision** (per the review cadence — STOP, don't autopilot). Tier-2 build: dual-tower per `B9_STL_STAN_SWAP §6.4` + `future_works/mtl_architecture_revisit`, unit-test gate + per-arch LR mini-sweep (hard rules 7/10) BEFORE multi-fold, scored vs frozen (c)/(d).

---

## 2026-06-04 — Independent review of the close-out (user-requested) → ENDORSED; no doc change; proceed to Tier 2

**Phase**: review of the O1–O5 close-out (no experiments). Verdict: the close-out is **sound — proceed to Tier 2 unchanged.**

**Evaluation.** O1 is a genuine, honestly-scoped finding (it falsified BOTH prior audit hypotheses: α converges *large* 0.45→1.09, the head leans into the prior, yet the prior drags the ceiling 0.56–3.03pp; standalone prior validates vs Markov-1 floors → not "embeddings subsume transitions," not a stuck-α bug; mechanism correctly stated as 1-of-3). O2 (multi-band cat negative holds; next_single GE-specific T5.2 candidate), O3 (FL inversion benign, bounds deployable), O4/O5 all clean. Leak audit NONE, freeze-sanity GREEN, frozen (c)/(d) unchanged. No substantive objection.

**Considered + REJECTED a steer (recorded so it is not re-litigated): pulling the multi-state (AZ/GE/FL) dense-supervision regime re-confirm FORWARD as a Tier-2 parallel probe.** Rejected after advisor pushback, for three reasons: (1) **it changes no Tier-2 action** — Tier 2 runs and is interpreted under the frozen non-overlap regime regardless; the dense re-confirm strengthens a *paper sentence*, not Tier 2's premise or go/no-go. (2) **The architectural reading is already triply-supported**, incl. the **windowing-INDEPENDENT P4 frozen-cat test** (reg peaks ep 2 with cat frozen — orthogonal to stride) + the 4-state non-overlap regime confirm + the AL dense probe (gap *widens*). For the re-confirm to overturn anything, AZ/GE/FL would have to behave opposite to AL in exactly the direction P4 says is structural — low-probability. (3) **It re-opens the windowing follow-up the user just deferred** (O5 / `overlapping_windows.md` explicitly scopes the AZ/GE/FL multi-seed re-confirm there) — and would be the 3rd scope-expansion after two trims. **Decision: leave the dense re-confirm parked in the follow-up study.** If a future agent can name a concrete way a dense result would change the Tier-2 *build* or go/no-go (not a paper sentence), revisit — none identified.

**O1 mechanism** (why a leaned-into prior hurts): leave as the existing **Tier-3 carry-over** (log_T = KD-on-representation in MTL vs additive-bias in STL; T3.1 re-sweeps log_T-KD on the new stack and naturally probes the distinction). Do NOT build a mechanism-isolation card now.

**Where attention belongs instead (for the next agent): the Tier-2 gates, not probes around them.** The two places T2.1 fails if it fails: (i) the **param-partition check** — the dual-tower's private backbone is a NEW param group; a silent omission in `shared/cat/reg_specific_parameters()` is the F49 class of bug (wire it into the partition; assert bijective+exhaustive in the unit-test gate); (ii) the **per-arch LR mini-sweep** — the B9_STL_STAN_SWAP collapse was the B9 recipe applied blind to a non-α head.

**Chain status**: unchanged — frozen (c)/(d) hold; close-out endorsed; **GO for T2.1**. Chain preserved.

**Next**: start Tier 2 (T2.1 dual-tower) per HANDOFF §9 — gates first (param-partition + per-arch LR), then multi-fold, scored vs frozen (c)/(d).

---

## 2026-05-16 — Track designed, awaiting execution (v1 — SUPERSEDED by the 2026-06-02 reframe above)

**Phase**: Design complete; no experiments run yet.

**What happened**

- Folder `docs/studies/mtl_improvement/` created. INDEX.html + this log + AGENT_PROMPT.md landed.
- Design session: read every preliminary file in `docs/studies/mtl-exploration/` (INDEX.html, considerations.md, EXPERIMENT_NO_ENCODERS.md, EXPERIMENT_HGI_SUBSTRATE.md, LEAK_BLAST_RADIUS_AUDIT.md, README.md). Re-read `canonical_improvement/` as structural template (INDEX.html, log.md head, AGENT_PROMPT.md, considerations.md).
- Three-dimensional audit (conceptual / technical-feasibility / metrics+baseline-robustness) of the 11 considerations in `mtl-exploration/considerations.md`. Verdict table in INDEX.html §Audit. All 10 numbered considerations accepted; 2 were right-sized (C2 from "experiment" to "pre-flight gate"; C4 with row-pairing-constraint flag).
- Ran 7 parallel breadth sub-agents (heads / backbone / loss / optimization / data-sampling / input-modality / instrumentation). Total: ~288 candidate directions across all angles. Strongest pulls summarized into the experiment slate in INDEX.html §Breadth.
- Critical correction from sub-agent 6: I had the encoder-asymmetry hypothesis inverted relative to AL Step-3 evidence in `EXPERIMENT_NO_ENCODERS.md`. Cat needs the thick MLP at AL scale (−2.57 pp paper-grade loss if simplified); reg fine with Linear+LN. Carried into design.
- Built dependency map. Designed 8-tier chain (T0 hygiene + diagnostics → T1 STL ceilings → T2a cheap backbones → T2b heavy backbones → T3 loss → T4 batch → T5 LR/optim → T6 α formula → T7 final head re-ablation → T8 multi-seed ship). Within each tier, experiments are parallel; tier-to-tier is sequential with explicit decision gates.
- User alignment captured via `AskUserQuestion`:
  - Tier 2 scope: aggressive — 8 archs faithful (4 cheap T2a + 4 heavy T2b).
  - C3 audit: full leak-free 3-way decomposition at 5 states × 1 seed + cross-arch check on T2 winner.
  - Folder/branch: new folder `docs/studies/mtl_improvement/` + new branch `mtl-improve`.
  - HGI substrate re-check: yes — one cheap HGI run per T2 winner.
- Pre-write advisor pass (model-level `advisor()`). Five substantive items addressed before writing:
  1. **Tier 2 LR confound mitigation**: per-arch light LR mini-sweep (constant 1e-3 / B9 per-head / arch-suggested-default; 5f × 30ep × seed=42 at AL+AZ) before judging the winner. Each arch wins or loses under its own best LR regime.
  2. **Per-tier decision gates**: every tier card now has explicit win-condition, no-winner fallback, and early-stop-if-headline-closes language.
  3. **C2 right-sized** to pre-flight gate (not a tier-defining experiment).
  4. **C4 row-pairing constraint flagged** in the audit row.
  5. **T0.7 same-machine sanity check** clarified: NOT v11-redundant; it is the canonical_improvement-style pre-flight pattern. Document the rationale in the experiment card so the implementing agent doesn't skip it.
  - Smaller flags also addressed: per-tier compute estimates, scale-conditional-vs-universal recipe commitment, Tier 2a/2b phased split with stop gate.

**Decisions**

1. **Branch `mtl-improve`** is the dedicated worktree. Do NOT contaminate `canonical_improvement` or `check2hgi-up` work.
2. **Scale-conditional framing preserved**: the ship recipe in T8 may be B9-prime at FL/CA/TX and H3-alt-prime at AL/AZ. Hunting a single universal recipe is permitted as a stretch outcome but is NOT a win condition.
3. **HGI substrate** is locked out as the substrate (per `EXPERIMENT_HGI_SUBSTRATE.md`); each T2 winner gets one cheap HGI sanity probe (single seed, AL+AZ, 5f × 25ep). If MTL+HGI is non-null under any new arch, escalate to user.
4. **Tier 2 phased split**: T2a (4 cheap archs ≤60 LOC each) gates T2b (4 heavy archs 80–250 LOC each). T2b becomes optional if a T2a winner already clears paper-grade significance.
5. **Per-arch light LR mini-sweep is mandatory** for every backbone candidate in T2a + T2b. Cost: 2–3× single-LR-comparison budget, but the alternative is silent confound.
6. **F49 audit scope**: 5 states × seed=42 × leak-free + one cross-arch validation on T2 winner. ~30 GPU-h H100.

**Chain status**: Tier 0 not yet launched.

**Next**

1. **Implementing agent** must read in order: this log, `AGENT_PROMPT.md`, `INDEX.html` (top-down, including §Execution guidelines), then NORTH_STAR.md + RESULTS_TABLE.md §0 + `mtl-exploration/EXPERIMENT_*.md` for grounding.
2. Create the dedicated worktree on branch `mtl-improve` before launching anything.
3. Start with Tier 0 in parallel:
   - **T0.2 must complete BEFORE T1** (mask/pad audit gates Tier 1 launch).
   - **T0.7 must complete BEFORE T1** (same-machine re-baseline; pin `B9_v11_repro`).
   - T0.1 / T0.3 / T0.4 / T0.5 / T0.6 can run in parallel.
4. Use `TaskCreate` to break down each Tier-0 experiment into unit-test → validate → launch → import → analyze sub-tasks.
5. After Tier 0 completes, call advisor with the Tier-0 results before launching Tier 1.

---

## 2026-05-16 — Post-write advisor pass — design tightened

**Phase**: Design hardening; still no experiments run.

**What happened**

Spawned a mandatory final-advisor sub-agent (per the user brief) to stress-test the design before handoff. Advisor surfaced five substantive items, all applied as text edits to INDEX.html:

1. **T2 LR mini-sweep was structurally biased toward the incumbent.** Original sweep had 3 regimes (constant 1e-3 / B9 per-head / arch-suggested-default). Advisor noted: 33% of the sweep is the incumbent recipe; new arch-specific params (cross-stitch per-channel α, AdaShare Gumbel gates, TaskExpert prompts) get no dedicated LR group; 30 epochs is short for slow-α-growth archs; AL+AZ-only sweep excludes states where heavy archs may differentially win. **Fix:** widened to 5 regimes (added R4 per-arch-group LR for new param groups, R5 B9+warmup-5%), extended to 40 epochs, added FL mini-sweep cell for T2b.2 (AdaShare) and T2b.3 (TaskExpert).
2. **Decision gates were loose** — many tier cards had `<div class="gate">Standard.</div>`. Advisor warned: "Pareto-dominates" without a magnitude floor invites cherry-picking. **Fix:** added explicit minimum-effect floor to all gates — `≥ 1 pp on targeted axis (n=5 single-seed) OR ≥ 0.5 pp (multi-seed n=15+) AND other axis non-inferior at TOST δ=2 pp`.
3. **HGI sanity probe was cargo-culted at single-seed n=5** — cannot detect a substrate × arch interaction at fold-σ scale. **Fix:** bumped to 2 seeds {42, 0} × AL+AZ × 5f × 30ep with explicit ≥ 2 pp escalation threshold; below threshold = informational only, no escalation.
4. **T4 was over-scoped.** Advisor: T4.2 (cat oversampling) is approximately subsumed by existing weighted-CE path; T4.4 (geo hard negatives) is too speculative for the cost. **Fix:** marked T4.2 and T4.4 as DEFERRED with explicit deferral-rationale blocks; kept T4.1 and T4.3 only. Tier-4 budget revised 80 → 40 GPU-h.
5. **Two chain-break risks were missing.** **Fix:**
   - Added "T1 STL_v2 winner ≠ Table A head" row: chain compares to BOTH (Table A for v11-paper continuity, STL_v2 for upper bound) in every Tier-2–7 results block.
   - Added "T0.5 instrumentation overhead pushes fold wall > 10%" row: T0.5 acceptance now includes a perf gate (instrumented run must complete within 10% of uninstrumented baseline; if over, demote per-step gradient cosine to per-N-step sampling).

Compute budget revisions per advisor: T2b 300 → 450 GPU-h; T8 200 → 300 GPU-h. Total envelope is now **~1700 GPU-h** (full chain incl. T2b) or **~1250 GPU-h** (if T2a wins the gate and T2b is skipped). Added top-level total-compute summary to TL;DR.

**Items NOT changed** (advisor flagged but design judgment preferred current spec):
- C3 (F49 audit) kept at the full L-cost scope as user-aligned; advisor agreed.
- T2a/T2b structure preserved; only the mini-sweep was widened.
- No tier was added or removed beyond the T4 deferrals.

**Chain status**: design hardened; still pre-launch.

**Next**: same as previous entry. Implementing agent should treat the updated INDEX.html as the source of truth; the changes above are reflected inline (T2a/T2b mini-sweep callout, T4.2/T4.4 deferred blocks, decision-gate floors throughout, chain-break-risk additions, T0.5 perf gate, compute revisions).

---

## 2026-05-28 — T0.2 mask-audit shared artefact landed (do NOT re-audit)

**What happened**

- Per the D1 ↔ T0.2 handoff protocol in [`docs/studies/substrate-protocol-cleanup/INDEX.md`](../substrate-protocol-cleanup/INDEX.md#d1--t02-mask-audit-handoff-mandatory-before-launching-d1), the sister study `substrate-protocol-cleanup` executed the window / causal-mask audit on 2026-05-28 as Tier D1.
- Shared artefact at [`docs/studies/substrate-protocol-cleanup/window_mask_audit.md`](../substrate-protocol-cleanup/window_mask_audit.md).
- **Verdict: overall CLEAN — no leak found across all 5 scope items** (`generate_sequences`, `NextHeadMTL` causal mask, `last_region_idx`, per-fold log_T builder, modality discipline). CONCERNS C19 guard confirmed still holding at `src/training/runners/mtl_cv.py:858-954`.

**Implication for this study**

- **T0.2 is satisfied. Do NOT re-audit.** Cite the shared artefact path in any T0.2-dependent finding doc.
- T1 launch is no longer gated on a leak verdict (it would have been if D1 found a leak).
- Three informational caveats from the audit (not blockers): CLAUDE.md path drift, STAN bidirectionality dependency on §1/§3 guarantees, C22 stale-log_T being runbook-enforced not code-enforced.

**Chain status**: T0.2 closed-by-handoff (audit performed in sister branch); chain preserved.

**Next**: implementing agent on this branch may proceed with T0 perf gates and T1 launch when ready. No mask-related blocker remains.

---

## 2026-06-04 — T2.1 DIAGNOSTIC LADDER → dual-tower FALSIFIED (clean negative) + a recipe finding

**Phase**: Tier 2.1 — the centerpiece experiment RESOLVED (negative). Mechanism probe running.

**Result** (AL+AZ, R3 onecycle, seed42, 5f×50ep, KD-OFF, **frozen-fold paired**, Δ vs the matched
`base_a` = `next_getnext_hard`@onecycle zero-point). Verified arms ran the intended models
(base_a=MTLnetCrossAttn+NextHeadStanFlow; dt_*=MTLnetCrossAttnDualTower+private_stan); per-fold log_T
loaded; base_a AL disjoint cross-checked from fold JSONs = [60.87,56.58,58.82,56.53,49.45]→56.45.

| arm | AL reg@10 disj | Δ vs base | AZ reg@10 disj | Δ vs base | cat (AL/AZ) |
|---|---|---|---|---|---|
| **base_a** (matched (a)@onecycle) | **56.45** | 0 | **44.26** | 0 | 48.51 / 49.43 |
| dt_gated_on | 53.95 | **−2.50** | 42.13 | **−2.13** | 48.44 / 49.38 |
| dt_priv_on | 52.41 | −4.04 | 40.50 | −3.76 | 47.64 / 49.54 |
| dt_priv_off (killer cell) | 52.32 | −4.13 | 40.55 | −3.71 | 48.07 / 49.80 |
| (c) STL reg ceiling | 62.88 | — | 55.11 | — | — |

**Verdict: T2.1 dual-tower FALSIFIED at AL+AZ.** <span>FALSIFIED</span>
- The dual-tower **LOSES to the same-recipe baseline by −2.1 to −4.1pp reg** at both states (cat ≈ tie).
- **Monotonic: more shared backbone = better reg** (base_a 56.45 > gated 53.95 > private_only 52.41).
  This is the **OPPOSITE of the §6.4 "missing private backbone" hypothesis** — the shared cross-attn
  backbone HELPS reg; isolating the reg head into a private tower HURTS it.
- **Killer cell** `dt_priv_off` (= "(c)-replica trained jointly", prior-OFF, no gate) = 52.32/40.55 —
  still **−10.6/−14.6 below (c)** AND below base_a. The private tower recovers NOTHING toward the STL
  ceiling. The MTL→STL reg gap is **not** the shared-vs-private pathway.
- The earlier "+2.6/+3.1 vs landed (a)" (LR sweep) was **entirely recipe drift** (advisor's catch
  confirmed): the matched onecycle baseline (56.45/44.26) ≫ landed H3-alt (a) (~47/38).

**Orthogonal finding (valuable byproduct) — the ONECYCLE recipe is a real reg lever.** The EXISTING
reg head (`next_getnext_hard`, no architecture change) at onecycle reaches disjoint reg AL 56.45 /
AZ 44.26 vs the landed H3-alt (a) diagnostic-best 47.23 / 38.27 → **~+6-9pp**. This is an
**optimizer/recipe** improvement (Tier-4 territory), NOT architecture, and it means the (a) MTL reg
baseline is itself improvable by a scheduler swap. CAVEAT: seed42 (dev seed; AL/AZ seed42≈multiseed
per CLAUDE.md, but needs multi-seed + FL validation before any ceiling/baseline change). Does NOT touch
the frozen (c)/(d) (those are STL ceilings; this is an MTL-baseline recipe finding). Flag for Tier 4 /
follow-up — do NOT re-pin anything off seed42.

**Decision (pending mechanism probe + user checkpoint)**
- Dual-tower did NOT survive the cheap ladder → **do NOT spend FL/3-seed on it** (advisor decision logic).
- Mechanism probe running (`t21_mech.sh`): `{base_a, dt_priv_off} × cat-weight=0 × AL+AZ` = reg-only
  training (P4 frozen-cat applied here). Tells us if the residual is multi-task competition (cat0 ≈ (c))
  or the joint harness/optimizer itself (cat0 << (c)).
- This is a **Tier-2-boundary-class decision** → surface to user: ship-composite negative + the recipe
  lever + whether to (a) confirm the negative at FL (1 matched pair), (b) pursue onecycle as a Tier-4
  baseline improvement, (c) proceed to the remaining Tier-2 cards (T2.0 hard-share anchor, T2.2
  CrossStitch) or close Tier 2 as "architecture doesn't move the reg gap; composite is the answer."

**Mechanism probe (`t21_mech.sh`, cat-weight=0 = reg-only, AL+AZ, R3 onecycle, seed42, 5f×50ep)**
| arm (reg-only) | AL reg@10 disj | vs (c) | vs cat0.75 | AZ reg@10 disj | vs (c) | vs cat0.75 |
|---|---|---|---|---|---|---|
| base_cat0 (shared head) | 57.42 | −5.46 | +0.97 | 44.83 | −10.28 | +0.57 |
| dtpriv_cat0 (private tower) | 52.98 | −9.90 | +0.66 | 40.83 | −14.28 | +0.28 |
- **Turning cat OFF barely moves reg (+0.3..+1.0pp)** → the MTL→STL reg gap is **NOT multi-task
  interference** (sharper P4 — cat-weight-0 on both heads, not just frozen-cat).
- **Even reg-only, the shared head stays −5.5/−10.3 below (c)** → the residual is the **harness/optimizer
  wrapper itself**, not cat competition. The private tower stays worse than the shared head reg-only too.
- One confound remains in base_cat0 vs (c): base uses prior-ON + wd0.05; (c) is prior-OFF + wd0.01.
  → `base_clean` (cross-attn + next_getnext_hard, **prior-OFF + wd0.01 + cat0 + onecycle**) isolates the
  cross-attn wrapper, matched to (c) on prior/wd/scheduler. **RESULT: AL 57.07 / AZ 44.62 disjoint —
  vs base_cat0 (prior-ON wd0.05) −0.35/−0.21 (≈ NO CHANGE), still −5.81/−10.49 below (c).** So prior+wd
  are NOT the gap; the residual is the **cross-attn MTL harness/architecture path itself** (reg read
  through next_encoder 64→256 + cross-attn + shared_next vs STL's bare STAN-on-raw-64). Claim #2 STANDS
  and strengthens: the MTL→STL reg gap is NOT interference, NOT the prior, NOT wd → the joint cross-attn
  wrapper. Residual confound (small): base_clean runs train.py-MTL-harness vs (c)'s p1-STL-harness.

**Advisor (Tier-2 boundary pass) — applied**: negative SOUND (paired + monotonic dose-response +
mechanism); label scoped "FALSIFIED at AL+AZ seed42 paired"; harden with ONE FL gated-vs-base pair +
T2.0 hard-share (4th dose-response point). cat0 zeroes the cat *objective* not the cross-attn *pathway*
(phrasing fixed). The frozen (a) anchor is now known-improvable by onecycle (don't re-pin off seed42;
future "fraction-recovered" framing must use the onecycle (a)). Skip T2.3/T2.4; keep T2.2 CrossStitch
as optional rebuttal card.

---

## 2026-06-04 — ONECYCLE VALIDATION (multi-seed) → small-state recipe WIN; TIER 2 final verdict

**Phase**: Tier 2 closing. onecycle {0,1,7,100} × AL/AZ/FL vs landed (a) (H3-alt small / B9 FL).

**Result (`T21_onecycle_validation_multiseed.txt`):**
| state (vs landed recipe) | reg_disj | reg_dep | cat_disj | cat_dep |
|---|---|---|---|---|
| AL (vs H3-alt) | 56.46±0.36 (**+9.23**) | 56.14 (**+6.00**) | 48.78 (+2.00) | 48.62 (+2.12) |
| AZ (vs H3-alt) | 44.22±0.18 (**+5.95**) | 44.06 (**+6.28**) | 49.52 (+0.77) | 49.44 (+0.92) |
| FL (vs B9) | 62.38±0.32 (+1.10) | 61.22 (+0.01) | 69.10 (−1.16) | 66.01 (−0.72) |

- **STATE-CONDITIONAL recipe win**: at small states (AL/AZ) the blessed recipe is H3-alt (weak constant
  LR); onecycle (aggressive schedule, **no alt-opt**) wins **+6-9pp reg AND +1-2pp cat**, multi-seed,
  tight σ (0.18-0.43) → not seed42 luck. At FL the recipe is already B9 (strong) → onecycle ≈ B9 (reg
  neutral, cat −1pp). Mechanism: onecycle = B9's aggressive schedule WITHOUT B9's alt-opt (which the LR
  sweep showed is the WORST knob for the reg head) → the small-state sweet spot H3-alt + B9 both missed.
- **Improves the (a) MTL baseline at small states** → shrinks the composite advantage: AL MTL→composite
  gap 13.44→7.44pp (deploy 56.14 vs composite 63.58); AZ 17.33→11.05pp. The "composite +12-17pp" claim
  (HANDOFF §9) was partly an artifact of the weak H3-alt small-state recipe.
- Does NOT change the T2.1 architecture verdict (dual-tower loses to base_a, both at onecycle).

**=== TIER 2 FINAL VERDICT (two results) ===**
1. **ARCHITECTURE: NEGATIVE (paper-grade, hardened AL/AZ/FL).** No MTL architecture closes the MTL→STL
   reg gap. Clean 5-point sharing dose-response (CrossStitch ≥ base_a ≈ hard-share ≫ dual-tower) +
   3-cell mechanism (gap is the cross-attn joint harness, not interference/prior/wd). **Composite
   (two-model deploy) is the deployable answer.** CrossStitch = weak partial (within σ, not a closer).
2. **RECIPE: a real Tier-4 WIN.** onecycle replaces H3-alt at small states for **+6-9pp reg / +1-2pp
   cat** (multi-seed). Actionable, paper-relevant — re-states the (a) baseline + composite-advantage.

**Open for user (recipe-change is consequential):** adopt onecycle as the small-state recipe (replacing
H3-alt in NORTH_STAR), validate at CA/TX, and re-state §0.1 (a)/composite small-state numbers? Cross-check
vs canonical_improvement recipe history first (was onecycle-without-alt-opt previously tested/rejected?).

**Chain status**: Tier 2 complete (architecture negative + recipe win); chain preserved.

**Soundness sanity-check (DONE):**
- **Leak-free**: onecyc_val runs loaded 5 seeded per-fold log_T each, KD OFF (verified in logs).
- **GENUINELY NEW, not a re-discovery** (sub-agent recipe-history audit): H3-alt was blessed over B9 at
  small states because B9's **alt-SGD** hurts cat (AL Δcat −2.22 p=1.9e-6, AZ −0.96 p=7.1e-4;
  `RESULTS_TABLE.md §0.4`, `NORTH_STAR.md:18-24`) — the rejection was pinned on **alt-opt, not the
  scheduler**. But H3-alt dropped BOTH the aggressive scheduler AND alt-opt. The "aggressive schedule +
  no alt-opt" cell was only ever tested at **FL** (F50_T3 A1/A2, where it loses to B9 because FL already
  has B9); **never at AL/AZ**. H3-alt was a binary over-correction; onecycle keeps the helpful lever
  (schedule), drops only the harmful one (alt-opt) → the small-state sweet spot both H3-alt and B9 missed.
  **Genuinely new, paper-relevant.**

**Next**: surface complete verdict + recipe decision to user (adopt onecycle small-state recipe? Tier 3?).

---

## 2026-06-04 — RECIPE × STATE MATRIX (user-requested) → scale-conditional recipe law decomposed

**Phase**: Tier 2 — recipe-matrix completion (user asked: B9@small, H3-alt@FL, CA/TX B9-vs-onecycle).
Baseline head, seed42 5f×50ep. Full write-up: `docs/results/mtl_improvement/T21_recipe_matrix.md`.

**Matrix (reg@10 disj / cat-F1; (L)=landed multi-seed):**
| recipe | AL | AZ | FL |
|---|---|---|---|
| H3-alt | 47.23/46.78 (L) | 38.27/48.75 (L) | **62.42/67.38** (NEW) |
| onecycle | **56.45/48.51** | **44.26/49.43** | 61.87/65.82 |
| B9 | **50.96/42.80** (NEW) | **38.32/46.82** (NEW) | 61.28/70.26 (L) |

**Decomposition (clean):**
- **Small states: onecycle dominates both axes.** reg: onecycle ≫ B9 ≥ H3-alt (AL 56.45/50.96/47.23)
  — aggressive schedule helps reg (B9>H3-alt), no-alt-opt helps MORE (onecycle>B9 +5.5; alt-opt costs
  reg too). cat: onecycle>H3-alt≫B9 (AL 48.51/46.78/**42.80** — B9's alt-SGD crushes small-state cat).
  → onecycle = aggressive schedule + no alt-opt = the sweet spot H3-alt and B9 both missed.
- **Large state (FL): reg recipe-insensitive (~61-62); B9 wins cat (70.26).** alt-opt FLIPS SIGN by
  scale — hurts small-state cat, helps large-state cat. So B9 stays right at FL (cat).

**Recommendation:** AL/AZ → onecycle; FL/CA/TX → keep B9. Re-states the small-state (a) baseline +
shrinks the composite advantage. NORTH_STAR small-state recipe-change candidate (user decision).

**CA/TX:** the canonical temp/ graph maps were cleaned → `build_region_sequence_tensor` FileNotFound.
**Safely regenerated** `checkin_graph.pt` (`t21_regen_catx_graph.py`: backup + isolated preprocess —
writes ONLY the graph .pt, deterministic spatial join — + verified 99.99% alignment vs next_region
last_region_idx; canonical embeddings/sequences UNTOUCHED). CA/TX runs are heavy (8.5k/6.5k regions,
~31GB/run) → must run **CONC=1 serial** (CONC=2 OOMs). B9+onecycle @ CA/TX running serially.

**Chain status**: Tier 2 — recipe matrix conclusive at AL/AZ/FL; CA/TX confirmation in flight.

**Next**: CA/TX complete → final big-picture advisor + surface recipe-change decision to user.

---

## 2026-06-04 — CAPSTONE ADVISOR (whole-Tier-2 implementation + decisions + big picture)

**Phase**: Tier 2 final review (AGENT_PROMPT item 5 — implementation-correctness + whole-track pass).

**Verdict: implementation CORRECT for this case; decisions SOUND; both headlines hold; Tier 2 CLOSEABLE.**

- **Implementation (verified end-to-end):** dual-tower fidelity chain exact (shared_stan = (a) head
  8heads/0.1do/d128; private_stan = (c) STL 4heads/0.3do/d128/raw64; distinct `priv_*` names genuinely
  dodge the inject-filter clobber — the load-bearing P0 fix is real). Matched baseline = `next_getnext_hard`
  = the shipped reg head. Param partition auto-covered + unit-gate-asserted (real F49-class guard). Mechanism
  cells isolate what's claimed; the ONE residual confound (train.py-MTL vs p1-STL harness for base_clean vs
  (c)) is honestly disclosed → sharpen paper wording to "the joint MTL reg pathway as instantiated," not
  provably "the cross-attn block" (deploy conclusion unchanged). CA/TX regen safety proof
  (parquet-untouched + 99.99% align) is necessary+sufficient for "canonical not corrupted"; the 0.01%
  (~hundreds of rows, 8.5k-region tract-boundary/tie) is benign — note it. Rundir-race fix sound.
- **Decisions:** the advisor matched-baseline REORDER was the save (converted a false "+2.6/+3.1 win"
  recipe-drift into the true negative). Killer-cell promotion, PCGrad/aux gating, hardening-after-negative,
  onecycle multiseed-before-claim + the recipe-history audit = clean hygiene. The ~2h CA/TX serial is mild
  over-confirmation (FL already anchors large states) — lower-EV but not wrong; non-load-bearing.
- **Scope:** R1 correctly "FALSIFIED (seed42 paired AL/AZ + FL pair)" — carry the seed caveat with
  "paper-grade." R2 "+6-9pp" honest (onecycle also beats B9 +5.5 reg at small states → not just beating the
  weak H3-alt). "Shrinks composite advantage" arithmetic correct: (c)/(d) frozen STL, recipe-independent;
  only (a)-deploy moves up. CrossStitch under-claimed (σ-bounded, not promoted) = honest.
- **Big picture (for paper):** Tier 2 = two paper-grade results pointing the same way. (R1) No single-model
  MTL architecture closes the MTL→STL reg gap; the deployable answer is the **two-model composite**; the gap
  is **irreducibly the joint cross-attn harness** (not interference/prior/wd/head). (R2) onecycle re-states
  the small-state MTL (a) baseline UP +6-9pp → part of the "composite wins +12-17pp" was an under-tuned
  small-state recipe artifact. **Single most important next action: the onecycle adoption decision** (it
  re-states §0.1(a) + the composite-advantage comparison the paper rests on).
- **Open before close:** (1) onecycle adoption re-states §0.1(a) + composite-advantage everywhere (HANDOFF
  §9, NORTH_STAR, PAPER_DRAFT) — multiseed AL/AZ done, CA/TX in flight; do NOT re-pin off seed42; one more
  grep of canonical_improvement recipe history before NORTH_STAR change. (2) CrossStitch 3-seed = optional
  hardening. (3) FL dual-tower 3-seed = optional reviewer insurance. (4) Re-run `t14_freeze_sanity.py`
  after any onecycle re-pin (first baseline-number change the track makes). Frozen (c)/(d) untouched ✓.

**Chain status**: Tier 2 closeable (R1+R2); CA/TX confirmation in flight (non-load-bearing).

**Next**: CA/TX lands → fold into recipe matrix → surface onecycle-adoption decision + Tier-2 close to user.

---

## 2026-06-04 — HARDENING: FL pair + T2.0 + T2.2 → negative CONFIRMED across AL/AZ/FL; dose-response

**Phase**: Tier 2 hardening (user-approved). All onecycle per-head, seed42, 5f×50ep, prior-ON, KD-OFF.
Added `cat_specific/reg_specific_parameters` to base MTLnet so T2.0 hard-share runs the matched recipe
(partition verified bijective+exhaustive; 17 mtlnet tests pass). FL seed42 v14 log_T staged+fresh.

**Sharing dose-response (reg@10 disjoint; `T21_dose_response_50ep_seed42.txt`):**
| arm | AL | AZ | FL | Δreg vs base_a |
|---|---|---|---|---|
| T2.2 CrossStitch | 58.12 | 45.10 | 63.43 | **+0.8..+1.7** |
| base_a (cross-attn, most-shared) | 56.45 | 44.26 | 61.87 | 0 |
| T2.0 hard-share (mtlnet trunk) | 56.28 | 43.72 | 61.35 | −0.2..−0.5 |
| T2.1 dual gated | 53.95 | 42.13 | **58.98** | −2.1..−2.9 |
| T2.1 dual private-only | 52.41 | 40.50 | — | −3.8..−4.0 |
| (c) STL ceiling | 62.88 | 55.11 | 73.31 | — |

**Findings:**
1. **Dual-tower negative CONFIRMED at FL** (gated 58.98 vs base_a 61.87 = −2.89) → the negative holds
   across the full coverage range AL/AZ/FL. The advisor's "harden at the large state" satisfied.
2. **Clean monotonic sharing→reg dose-response** at all 3 states: CrossStitch ≥ base_a ≈ hard-share ≫
   gated > private-only. More cross-task sharing helps reg; isolation (the dual-tower) hurts. The §6.4
   "missing private backbone" hypothesis is decisively refuted with a 5-point curve.
3. **T2.0 hard-share ≈ base_a** (within 0.5pp) — hard (shared trunk) and soft (cross-attn) sharing are
   equivalent for reg; both ≫ the dual-tower. The cross-attn champion is not the bottleneck.
4. **T2.2 CrossStitch is the only architecture that doesn't lose reg** — slight +0.8..+1.7 vs base_a,
   CONSISTENT across 3 states (cf §6.3 +1.84 "within σ"), but cat MIXED (AL −1.71, AZ +0.17, FL +1.41)
   and still **−4.8 to −9.9pp below the (c) ceiling** → a weak partial within fold-σ, NOT a gap-closer.
   Worth a multi-seed look as a minor reg-lever; does not change the verdict.
5. **No architecture closes the MTL→STL reg gap** (best = CrossStitch, still −4.8..−9.9 below (c)) →
   **composite (two-model deploy) is the answer.** Verdict hardened.
6. **The onecycle "+6-9pp reg lever" is SMALL-STATE-specific:** at FL, base_a@onecycle reg 61.87 ≈
   landed B9 61.28 (+0.6) but cat 65.82 < B9 70.26 (**−4.4pp cat!**). So onecycle helps reg at AL/AZ,
   is reg-neutral + cat-HURTING at FL. The multi-seed validate stage (running, CONC=2) quantifies this —
   the lever is NOT a universal Tier-4 win.

**Chain status**: T2.1 NEGATIVE hardened across AL/AZ/FL; chain preserved.

**Next**: read validate (onecycle multi-seed AL/AZ/FL) → final Tier-2 verdict + advisor + write-up.

---

## 2026-06-04 — TIER 2 SUMMARY (T2.1 centerpiece resolved: NEGATIVE) — surface to user

**Tier-2 status: the load-bearing card (T2.1 dual-tower) is a clean NEGATIVE. Composite is the answer.**

**What ran** (all v14, R3 onecycle, seed42, 5f×50ep, KD-OFF, frozen-fold paired, AL+AZ):
- Unit gate GREEN; end-to-end validated; LR mini-sweep (R3 onecycle winner; B9 worst).
- Diagnostic ladder (matched baseline + gated + private_only + killer cell) + 2 mechanism probes
  (cat-weight=0; prior-OFF+wd0.01).

**The verdict (4 findings):**
1. **T2.1 dual-tower FALSIFIED** (AL+AZ, paired): a private reg STAN tower LOSES to the same-recipe
   shared-backbone baseline by −2.1..−4.1pp reg (cat ≈ tie). Monotonic **more-shared = better reg**
   (base_a 56.45 > gated 53.95 > private_only 52.41 @AL). **Refutes the §6.4 "missing private backbone"
   hypothesis** — the shared cross-attn backbone HELPS reg; isolating the reg tower hurts.
2. **The MTL→STL reg gap is the cross-attn HARNESS, not interference/prior/wd** (3 mechanism cells):
   cat-weight=0 barely moves reg (not multi-task competition); prior-OFF+wd0.01 barely moves reg (not the
   α-prior, not weight decay). Even reg-only, recipe-matched to (c), the harness reg head sits −5.8/−10.5
   below the STL ceiling. **No single-model change inside the cross-attn MTL harness closes the gap.**
3. **→ The composite (STL-cat ⊕ STL-HGI-reg, two models) is the deployable answer** — the HANDOFF §9
   "paper-grade negative" branch is now the live result.
4. **Byproduct (Tier-4 lever): onecycle lifts the EXISTING reg head ~+6-9pp** vs landed H3-alt (a)
   (base_a disjoint 56.45/44.26 vs landed 47.23/38.27; deploy ≈ disjoint, so not oracle-only). The
   frozen (a) MTL anchor is **known-improvable**; seed42 — needs multi-seed+FL validation before re-pin.

**Open decisions for the user (the tier-boundary STOP):**
- (H) Harden the negative: run ONE FL `base_a` vs `dt_gated_on` pair (onecycle; needs FL seed42 v14
  log_T built) + T2.0 hard-share anchor (AL+AZ; recipe caveat: base `mtlnet` lacks per-head LR) to
  complete the 4-point dose-response curve. ~1h A40.
- (R) Pursue the onecycle recipe lever as a Tier-4 baseline improvement (multi-seed + FL).
- (C) Close Tier 2 on "architecture doesn't move the reg gap; ship composite" + write it up; defer
  T2.2 CrossStitch as an optional rebuttal card; skip T2.3/T2.4 (prior-falsified).

**Chain status**: T2.1 resolved (NEGATIVE); chain preserved. Composite-fallback branch is live.

**Next**: user decides H/R/C (no autopilot — tier-boundary cadence).

---

**Chain status**: T2.1 resolved (negative); chain preserved. The "composite is the deploy fallback"
branch is now the live hypothesis (HANDOFF §9 "paper-grade negative").

**Next**: read `base_clean` → Tier-2 advisor pass on the negative → write Tier-2 summary → STOP + surface
to user (tier-boundary cadence).

---

## 2026-06-04 — Advisor on LR result → REORDER to a diagnostic ladder (recipe-drift catch)

**Phase**: Tier 2.1 — advisor pass on the LR sweep; full protocol reordered before launch.

**Advisor verdict (applied)**
- **R3_onecycle winner ENDORSED** — but it's a **cat-driven + STL-schedule-fidelity** pick: on reg alone
  R3 *ties* R1 (constant 1e-3) within fold-σ (AL 53.05 vs 52.96; AZ 41.03 vs 40.77). R3's real edge is
  cat (+1.2–2.3pp, outside noise) + it's the schedule the private tower was trained under at STL. R3
  dominates-or-ties everywhere, never loses → still the right pick. Don't oversell a reg win over R1.
- **CRITICAL recipe-drift catch — my "+2.6/+3.1 vs (a)" is partly NOT architecture.** The landed (a)
  AL/AZ numbers (50.14/37.78) were **H3-alt** (≈constant), not onecycle. The dual-tower at *constant*
  (R1) deploy reg is already 51.94/40.39 = **+1.8/+2.6 above landed (a)** — that gap is recipe/epoch/
  harness drift, NOT the architecture. **The clean architectural Δ REQUIRES a matched `next_getnext_hard`
  @onecycle baseline** (frozen-fold paired, same seed/folds) as the zero-point, at every state. Folded
  into the ladder as arm #1.
- **Promote the killer cell `private_only @ prior-OFF` to stage 1** = "the (c) STL ceiling trained
  jointly" (private tower IS the raw-64→STAN (c)-replica; prior-OFF makes it the faithful α=0 (c) recipe;
  private_only removes the shared pathway). If it ≈ (c) → the gate/shared-mix is the drag; if << (c) →
  the residual is irreducibly joint-optimization (→ composite is the deploy answer, paper-grade negative).
- **Reorder (D3):** run a cheap AL+AZ **diagnostic ladder** FIRST (answers the headline in <1h A40), spend
  FL/multi-seed only on survivors. Defer fusion beauty-contest, substrate-2×2, HGI probe.
- **Driver traps flagged** (`t21_full_protocol.sh`, NOT launched as written): FL fusion_pick CRASHES (no
  FL v14 seed42 log_T — verified 0 files); HGI stage CRASHES (no HGI log_T at AL/AZ); FL pcgrad+alt-opt
  arg-invalid; FL→B9 hardcoding re-confounds (FL must be onecycle to match). The ladder (AL+AZ only)
  sidesteps all of these.

**Decision**: build a purpose-built `t21_ladder.sh` (AL+AZ, R3 onecycle, seed42, 5f×50ep, static_weight
cat0.75, KD-OFF, seeded per-fold log_T). Core 4 arms × 2 states = 8 runs:
  1. `base_a`      = mtlnet_crossattn + next_getnext_hard, prior-ON → **matched (a)@onecycle zero-point**
  2. `dt_gated_on` = dualtower gated, prior-ON → primary thesis (Δ vs #1)
  3. `dt_priv_on`  = dualtower private_only, prior-ON → gate-vs-no-gate isolation
  4. `dt_priv_off` = dualtower private_only, prior-OFF → **killer cell** ((c) trained jointly)
Conditional isolators (only if #4 << (c)): reg-head-wd=0.01 (needs a new flag) + cat-weight=0 (P4 frozen-cat
probe). Then surface the decomposition to the user (the real ship/partial/negative decision point).

**Chain status**: T2.1 in flight; chain preserved.

**Next**: launch `t21_ladder.sh`; aggregate vs (a)-matched + (c)/(d); surface decomposition to user.

---

## 2026-06-04 — T2.1 implementation + unit gate + LR mini-sweep (R3 onecycle wins)

**Phase**: Tier 2.1 — implemented, unit-gated, LR mini-sweep DONE; full protocol next.

**What happened**
- Implemented `NextHeadStanFlowDualTower` + `MTLnetCrossAttnDualTower`; unit-test gate (`t21_unit_gate.py`)
  GREEN (partition bijective+exhaustive, α Parameter/buffer per prior, capacity confined to next_poi,
  fusion semantics, prior fires, next_forward carries the private tower). 187 model tests pass; freeze
  sanity GREEN. End-to-end AL smoke validated the on-disk structure + per-fold log_T load + KD-OFF.
- **Driver race caught + fixed**: concurrent `train.py` runs mis-mapped rundirs via `ls -dt|head -1`
  (3 regimes → 1 dir; bit-identical agg numbers were the tell). Fixed with PID-suffix capture
  (`...{ts}_{os.getpid()}`), discarded the bad run, re-ran clean. (Memory: ref-concurrent-rundir-race.)
- **LR mini-sweep** (hard rule 7): variant (b) gated, prior-ON, v14, KD-OFF, AL+AZ, 5f×40ep×seed42,
  5 regimes. PID-safe, MPS-collocated CONC=4 (~11GB VRAM). All 10 runs distinct.

**Findings** (40ep seed42 — DIRECTIONAL, not final; full protocol is 50ep + multi-seed)
- **WINNER = R3_onecycle** (`--scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3`),
  consistent across AL+AZ on disjoint reg, deploy reg, AND cat:
  | | reg@10 disj | reg@10 deploy | Δreg vs (a) | cat-F1 deploy | Δcat vs (a) |
  |---|---|---|---|---|---|
  | AL R3 | 53.05 | 52.75 | **+2.61** | 48.52 | **+2.02** |
  | AZ R3 | 41.03 | 40.86 | **+3.08** | 49.64 | **+1.12** |
  - Ranking AL: R3 53.05 > R1 52.96 > R4 52.16 > R5 51.87 > **R2_b9 49.98 (WORST)**.
    AZ: R3 41.03 > R1 40.77 > R5 39.44 > R4 38.55 > **R2_b9 37.96 (WORST)**.
  - **R2_b9 worst** — the B9_STL_STAN_SWAP pattern: `--alternating-optimizer-step` trains reg on half the
    batches → under-trains the heavier dual-tower. The mini-sweep did exactly its job (would have
    sandbagged the arch under B9). R3 = the STL-reg-ceiling scheduler (onecycle max-lr 3e-3) → the
    private STAN tower prefers the schedule it was trained under at STL. Mechanistically clean.
- **The encouraging signal**: gated dual-tower @ R3 BEATS the (a) v14-MTL deployable baseline on BOTH
  axes (+2.6/+3.1pp reg, +1.1/+2.0pp cat) — no collapse, modest dual-axis gain.
- **The crux question**: disjoint reg (53.05/41.03) is still ~9.8/14.1pp BELOW the (c) STL ceiling
  (62.88/55.11), recovering only ~18-19% of the MTL→composite gap. The private tower IS structurally
  the STL backbone (raw 64→STAN) yet underperforms it jointly. Candidate causes to decompose in the
  full protocol: (i) the gate mixes in the cat-shaped shared pathway (→ private_only control), (ii) the
  in-head α-prior is ON here but the (c) ceiling is prior-OFF — O1 says prior is a STL drag (→ prior-OFF
  control), (iii) joint optimizer/wd (0.05 vs STL 0.01) under-trains it.

**Decision** (pending advisor + user checkpoint)
- Full-protocol recipe: AL/AZ = R3 onecycle. **Open: FL recipe** — B9 (FL production, but WORST for the
  dual-tower) vs onecycle (the arch's preference, but then confounds vs the B9-trained (a) at FL). Leaning
  to run a **matched-recipe (a)-head baseline at onecycle** (frozen-fold paired, hard rule 2b) so the FL
  architectural Δ is clean. Advisor to confirm.

**Chain status**: T2.1 in flight; chain preserved.

**Next**: advisor pass on the LR result + recipe/baseline decision → user checkpoint → full protocol
(fusion-mode pick {gated,private_only,aux} + matched baseline, then prior-OFF + PCGrad refine, 2×2, 3-seed).

---

## 2026-06-04 — Tier 2 STARTED: T2.1 dual-tower design + advisor review (pre-code)

**Phase**: Tier 2.1 in flight (design + review complete; implementation next).

**What happened**
- Onboarded Tier 2 per HANDOFF §9. Confirmed Tier 0/1/S + audit closed & frozen; `t14_freeze_sanity.py` GREEN.
- Read the load-bearing code: `mtlnet_crossattn/model.py`, `next_stan_flow/head.py`, `next_stan/head.py`
  (the STAN backbone + `forward_features` pool), `mtlnet/model.py` (`_build_next_head` inject+filter,
  param partitions), `helpers.py` (`setup_per_head_optimizer` + α-no-wd + reg-encoder/head LR split).
- Traced CLI→model→trainer plumbing (sub-agent): how `--reg-head`/`--per-fold-transition-dir`/
  `--mtl-loss`/`--cat-lr/--reg-lr/--shared-lr`/`--alternating-optimizer-step` wire through; how the
  per-fold head rebuild injects the seeded `transition_path`; how PCGrad enumerates
  `shared_parameters()`/`task_specific_parameters()`; how `geom_simple` (deployable) vs
  `diagnostic_best_epochs` (disjoint) are reported.
- Pinned the **(a) baseline**: HANDOFF §9 "MTL deployable reg" = the JOINT-GEOM-SIMPLE rows of
  `v14_mtl_vs_canonical.md` (AL 50.14 / AZ 37.78 / GE 42.64 / FL 61.21), run **KD-OFF, in-head α-prior
  ON**. KD-on-the-new-stack is **Tier 3 / T3.1** → Tier 2 holds **KD OFF** for a clean architectural Δ.
- Confirmed the **frozen-fold paired** design = deterministic `StratifiedGroupKFold(random_state=seed)`
  (all arms at a seed share folds automatically); `freeze_folds.py` is a drift-guard, run `--check`
  preflight, not a partition loader.
- Wrote the design proposal `T2.1_DUALTOWER_DESIGN.md` (8 open questions) → ran a **rigorous advisor
  sub-agent** critique BEFORE any code.

**Design (post-advisor)**
- New reg head `NextHeadStanFlowDualTower` (registry `next_stan_flow_dualtower`): a **private full-STAN
  backbone on the raw [B,9,64] region sequence** (faithful (c)-STL replica) + the existing **shared**
  STAN on the cross-attn output [B,9,256] (faithful (a) replica), fused at the pooled [B,128] feature
  by per-dim sigmoid **gate** (variant b PRIMARY) / **private_only** (a) / **aux** (c), then a single
  classifier + the α·log_T prior. Private tower lives **inside `next_poi`** → automatically in
  `reg_specific_parameters()`+`task_specific_parameters()` (advisor confirmed partition stays
  bijective+exhaustive). Model subclass `MTLnetCrossAttnDualTower` overrides `forward` **and
  `next_forward`** to pass the post-mask raw `next_input` as `raw_region_seq`.

**Decision** (advisor-driven, applied to the design doc §6)
- **Fidelity fix (P0):** use **distinct param names** `priv_num_heads=4`/`priv_dropout=0.3` (STL
  defaults, NOT injected) for the private tower; let the injected `num_heads=8`/`dropout=0.1` drive the
  shared tower (matches (a)). `d_model=128`, `bias="alibi"` both. Frozen (c) reg ceiling recipe verified:
  `NextHeadSTAN` defaults + AdamW lr=1e-4 **wd=0.01** OneCycleLR max_lr=3e-3, α=0 prior-OFF.
- **Prior:** primary arm prior-ON (match (a)); **prior-OFF control** on the winning mode (true
  (c)-backbone replica + MTL-regime O1 test).
- **`next_forward` MUST pass raw** (disjoint diagnostic-best is the headline metric) — unit-tested.
- None-fallback keeps private tower+β in-graph; **PCGrad gated to winning mode** (9→pick→3, not 18);
  gate-bias init +1.0 toward private + log mean-gate/epoch; **stale-log_T mtime preflight** each stage;
  wd=0.05 global mismatch vs STL 0.01 accepted+documented (single-model recipe).

**Chain status**: T2.1 in flight; chain preserved (Tier 0/1/S frozen, untouched).

**Next**
- Implement `NextHeadStanFlowDualTower` + `MTLnetCrossAttnDualTower`, register, then the **unit-test
  gate** (hard rule 10) before any multi-fold launch. Then LR mini-sweep (b gated, v14, AL+AZ,
  5f×40ep×seed42, 5 regimes).

---

## How to add an entry to this log

Use this template for every working session:

```markdown
## YYYY-MM-DD — Short title

**Phase**: Tier X.Y in flight / completed / paused.

**What happened**
- Bullet point.
- Bullet point.

**Findings** (if any)
- Numeric, per-state where applicable.

**Decision** (if any)
- What changed and why.

**Blocker** (if any)
- What's blocking; what input is needed to unblock.

**Chain status**: T?-? in flight / chain preserved / chain broken (reason).

**Next**
- What the next agent should pick up. Be specific (experiment ID, state, seed, expected wall).
```

**Rules**:
- Append at the bottom; never edit historic entries.
- Date is the YYYY-MM-DD of the work session (UTC if cross-zone).
- If you finish a tier, flag it explicitly (`Tier X COMPLETE — pinned recipe: ...`).
- If you break the chain (run an out-of-order experiment), document `**Chain status**: broken because <reason>` AND the re-execution plan to restore it.
- If you fork the design (add a new experiment not in INDEX.html), add it to the HTML in the same session AND document here.
