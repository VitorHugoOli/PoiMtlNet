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

## 2026-06-04 — Tier-2 review → REDIRECT: new Tier 2P (joint-training protocol); "irreducibly architectural" overturned

**Phase**: independent review of the pulled Tier-2 results (T2.0/2.1/2.2 + dose-response + mechanism cells + onecycle recipe). User concern: "we are not closing the gap with STL." Verdict: **Tier 2 sound as executed, but its headline conclusion is wrong — redirect to a protocol tier.**

**Evaluation.** Implementation faithful (advisor-verified param-partition, distinct priv/shared backbones, prior controls); the topology negative is multi-seed-hardened; the dose-response (`CrossStitch ≥ base_a ≈ hard-share ≫ dual-tower`) and mechanism cells are clean; onecycle is a genuine +5–9pp small-state recipe win. No objection to *what was done*.

**The contradiction (the find).** The close-out says "the gap is irreducibly architectural; ship the composite." But the `private_only` arm **IS the STL reg topology by construction** (design doc §2.1/§6), it **ran under the good onecycle recipe** (verified: `t21_harden.sh:29` `--scheduler onecycle`; base_a AL 56.45 matches the onecycle cell), and it **still lost ~10pp to STL-standalone** (AL 52.41 vs 62.88; prior-OFF 52.32). An identical topology failing *only when trained jointly* ⟹ the residual is **the joint training PROTOCOL, not the topology.** Tier 2 varied *how to share*, never *how to train* — and onecycle (a pure protocol change) already moved the gap. Advisor independently confirmed this is a logical contradiction in the close-out's framing, headline-flipping, not scope-creep.

**Two confounds tightened (advisor), now encoded as the T2P.0 linchpin.** The private-tower ran at the joint **wd=0.05** vs STL's **wd=0.01**, and used the joint recipe — so "joint training poisons reg" is confounded with "wrong wd/recipe." The cleanest isolation: re-run `private_only prior-OFF` at **wd=0.01** (STL-matched), all else identical to (c). Recovers→recipe mismatch (lever = asymmetric per-task recipe); still-collapsed→the joint loop itself (lever = staged/sequential). Also flagged: verify `--category-weight 0.0` zeroes cat's gradient (not just the loss scalar) in `mtl_cv.py`.

**Decision (user, AskUserQuestion 2026-06-04): "Pursue the protocol axis."** Added **Tier 2P — joint-training protocol** (INDEX `#tier2p`): **T2P.0** linchpin (gates) → **T2P.1** staged (reg→freeze→cat) / **T2P.2** asymmetric per-task recipe / **T2P.3** composite-distillation. **Goal = composite-quality reg in ONE deployable model; the 2-model composite is the null.** Framing trap recorded (advisor): staged gives reg≈STL *by construction* (arithmetic, one flag from the dropped two-checkpoint deploy) — the real question is **does cat survive a frozen-reg trunk?** Distinguish a deployment win (½ params) from a scientific gap-closure (beats the composite). Closest prior art: the P4 `--freeze-cat-after-epoch` staged idea (`phase1_phase2_verdict_v6_final.md:118`) was hypothesised but never run as a gap-closer.

**Docs changed.** INDEX: revised the Tier-2 final-decision callout (topology-negative but NOT "irreducibly architectural"), new Tier 2P section (T2P.0–T2P.3 + final-decision), TOC. HANDOFF: §0 redirect, §0b reframed (T2.3/T2.4 now confirmatory-of-topology), new §0c (T2P.0 onboarding + the exact CLI). T2.3/T2.4 remain as the cheap confirmatory close of the topology card-set.

**Separate open item (NOT bundled):** the onecycle small-state recipe reshapes a §0.1 claim → author sign-off pending (HANDOFF §0(2)).

**Chain status**: Tier-2 topology axis CLOSED (negative); **Tier 2P OPEN, T2P.0 is the next step.** Chain preserved (Tier 2P is the evidence-pointed continuation of the regime attack, not a detour). Frozen (c)/(d) unchanged.

**Next**: A40 runs **T2P.0** (the linchpin) → STOP + surface the verdict → launch the chosen lever (T2P.1 or T2P.2). Finish T2.3/T2.4 confirmatory alongside.

---

## 2026-06-04 — T2P.0 LINCHPIN ran + a load-bearing advisor catch (fp16-no-scaler harness confound) → fp32 CONTROL in flight

**Phase**: Tier 2P — T2P.0 executed; new agent onboarded (read AGENT_PROMPT/HANDOFF/PAPER_UPDATE + full dual-tower code audit). Before any further lever, an fp32 control is running. **NOT yet surfaced to user — verdict pending the control.**

**Code audit before launch (user asked for meticulous eval) — all CLEAN:**
- Dual-tower (`next_stan_flow_dualtower` + `mtlnet_crossattn_dualtower`) read end-to-end vs design §6: fidelity chain exact (`private_stan`=NextHeadSTAN.forward_features on raw [B,9,64], priv 4heads/0.3do/d128; distinct `priv_*` names dodge the inject-filter); partition bijective+exhaustive (private tower ∈ next_poi ∈ reg+task, excluded from shared); `t21_unit_gate.py` re-run **GREEN**.
- **T2P.0 precheck (advisor-flagged) CONFIRMED**: `static_weight` with `--category-weight 0.0` → weights `[1-cw,cw]=[1,0]` → `0.0*cat_loss` = exactly zero cat gradient into shared/cat params (`static_weight/loss.py:36`); in `private_only` reg ignores `shared_next` so reg gradient never touches cross-attn/encoder either. Empirically confirmed: cat-F1 collapsed to 6-9% (cat untrained). Only residual diff from STL = the joint loop.
- Sister audits (background agents): **T2.3 MoE/CGC partition GREEN** (independently re-verified `t23_audit_partition.py`; -lite caveats accurate — per-task-input adaptation, DSelect-K misnamed dense combo; gate OPEN for -lite confirmatory, no rebuild). **T2.4 hybrids scoped** (3 archs, partition-wiring per arm, F49 substring-trap flagged; crossstitch→crossattn HIGH leak risk, SwiGLU LOW).

**T2P.0 result** (`mtlnet_crossattn_dualtower` private_only prior-OFF, **cat-weight 0 + wd 0.01**, onecycle, KD-OFF, v14, 5f×50ep seed42, seeded per-fold log_T; `t2p0_manifest.tsv`, agg via `t21_agg.py`):
| state | T2P.0 reg@10 disj | prior wd0.05 cell | (c) STL reg | Δ from wd fix | Δ vs (c) |
|---|---|---|---|---|---|
| AL | 52.90 ±4.20 | dtpriv_cat0 52.98 / ladder 52.32 | 62.88 | **−0.08** | −9.98 |
| AZ | 40.80 ±2.87 | dtpriv_cat0 40.83 | 55.11 | **−0.03** | −14.31 |
| FL | 59.53 ±0.40 | (none exact) | 73.31 | — | −13.78 |

**Two findings (both survived an adversarial advisor pass):**
1. **wd is RULED OUT** — 0.05→0.01 moved reg ~0pp (AL −0.08, AZ −0.03). wd-match to (c) is now exact (p1 (c) uses AdamW wd=0.01; T2P.0 wd=0.01).
2. **`max_size_cycle` is a NO-OP here** — and the reason is stronger than equal parquet rows: the check2hgi MTL preset feeds **one shared X (next.parquet) through a single shared `sgkf.split`** to both task slots (`folds.py:_create_check2hgi_mtl_folds`), so cat/reg loaders are equal-length **by construction every fold** (verified counts: AL 12709/12709, AZ 26396/26396, FL 159175/159175). `batches_per_epoch=max(len(reg),len(cat))` cycles nothing; reg gets its natural step count + an onecycle LR trajectory identical to p1's (grad_accum=1; `OneCycleLR(max_lr=3e-3)` scalar broadcasts to all groups → the per-head `--cat/reg/shared-lr` are FLATTENED to 3e-3 under onecycle, which for reg coincidentally = p1's reg max_lr). **Corollary: T2P.2's per-task-LR + no-cycle-starve levers are INERT under onecycle** → T2P.2 is even more moot than first argued.

**⚠ ADVISOR CATCH (load-bearing, headline-shaping) — the gate is NOT clean: an fp16-no-GradScaler HARNESS confound.** The CUDA MTL trainer runs `torch.autocast(float16)` with **NO GradScaler anywhere** (`mtl_cv.py:286`, grep: zero `scaler`); the (c) STL ceiling (p1 harness, `p1_region_head_ablation.py`) runs **fp32**. fp16-no-scaler on a ~1k-class STAN ranking head degrades exactly Acc@10, worse at larger class spaces — matching the gap ordering (FL ~14pp > AL ~10pp). Because T2P.0 already stripped joint *dynamics* to near-zero (cat-grad 0 + private_only + matched LR/wd/folds), the residual is **mostly trainer/precision harness, not loop poison** — the OPPOSITE of "the joint loop caps reg." So "joint loop poisons reg → T2P.1" is **confounded** and must not be gated directly. (Advisor also ruled out: fold-partition mismatch — both use the same seeded StratifiedGroupKFold; residual-skip path — off by default. The fused-classifier dropout 0.1 vs STL 0.3 is a real but second-order fidelity gap.)

**Action: cheapest decisive control = re-run the EXACT T2P.0 cell in fp32.** Added a diagnostic env-var gate `MTL_DISABLE_AMP=1` in `mtl_cv.py` (forces the fp32 path; default unset = canonical fp16 untouched; verified import + present). Driver `t2p0_fp32_control.sh` (FL decisive σ=0.40 + AL), running now.
- **fp32 jumps toward (c)** → gap was PRECISION/harness, NOT joint-loop poison → T2P.1's premise ("reg trained outside the joint harness reaches (c)") undermined; the real fix is the trainer precision path. Also re-frames the WHOLE MTL→STL reg gap (every Tier-2 MTL number ran fp16 vs fp32 STL ceilings — though RELATIVE within-MTL comparisons like the dose-response are unaffected since all fp16).
- **fp32 stays ~T2P.0 (~52/59)** → harness exonerated → T2P.1 (staged) gates cleanly.

**Corrections applied to interpretation (advisor): lead the gate on FL (±0.40, ~34σ), not AL (±4.20, ~2.4σ noisy); demote the "joint dynamics" claim; flag the fp16 confound as the leading unexplained residual; record the onecycle-flattens-per-head-LR quirk.**

**Chain status**: Tier 2P — T2P.0 ran but the gate is confounded; fp32 control decides. Frozen (c)/(d) untouched.

**Next**: fp32 control lands → combined verdict (precision-artifact vs joint-poison) → **STOP + surface to user** (tier-boundary cadence; do NOT auto-launch T2P.1/T2P.2). If precision-artifact: the bigger story is the fp16-no-scaler trainer path (affects the MTL-vs-STL-ceiling gap magnitude study-wide) — propose a GradScaler/fp32 fix + re-baseline scope to the user.

---

## 2026-06-04 — ⛔ RETRACTED (same session): the "input-artifact confound" below was a FALSE ALARM — MTL reg input is BYTE-IDENTICAL to the (c) ceiling input

**RETRACTION (the honest correction, kept for the lab-notebook trail).** The hypothesis below — "the MTL reg head eats `next_region.parquet` (check-in-level) while (c) eats pooled `region_embeddings.parquet`, a different space (cosine 0.125)" — is **WRONG**. I compared the WRONG tensor. `next_region.parquet`'s emb columns are an **unused red herring**: `FoldCreator._create_check2hgi_mtl_folds` (`folds.py:902,985`) uses `next_region.parquet` ONLY for the **labels** (`region_idx`) + `last_region_idx` (aux), and builds the actual reg INPUT fresh via `data.inputs.region_sequence.build_region_sequence_tensor` (a pooled `region_embeddings.parquet` lookup). **Direct test: `build_region_sequence_tensor` (MTL reg input) vs p1's `_build_region_sequence_tensor` (the (c) input) are BYTE-IDENTICAL — max abs diff 0.0, shape (159175,9,64).** So the MTL reg head AND the dual-tower private tower (raw_region_seq=next_input=task_b X) eat the **SAME** input as (c). **The dual-tower design premise ("private STAN on next_input IS exactly the STL path") is TRUE. There is NO input confound.** My cosine-0.125 measurement was against `next_region.parquet`'s stale emb columns, which the MTL pipeline never feeds the reg head.

**What the confirmatory `t2p0_input_artifact.sh` actually showed** (p1, next_stan_flow α=0, 5f50ep): `region` (pooled lookup) AL 66.79 / FL 73.87 (FL ✓≈ frozen (c) 73.31); `checkin` (next.parquet check-in emb — p1's "checkin" loads next.parquet, NOT next_region) AL 21.87 / FL 37.79. The huge region≫checkin delta just confirms a STAN needs region embeddings (not raw check-in emb) — it does NOT bear on the MTL-vs-(c) gap, because the MTL reg already uses the `region` input (verified identical).

**NET — the original Tier-2P reading STANDS, now SHARPENED.** With the input ruled out (identical), wd ruled out, cycle a no-op, and precision ruled out (fp32 control AL 52.92/FL 59.66 ≈ fp16), the isolated reg STAN (T2P.0, dual-tower private, dynamics-neutralized) reaches **52.90/59.53 in the train.py-MTL harness vs 62.88–66.8 in the p1 harness on the IDENTICAL input** → the residual ~10-14pp is a genuine **train.py-MTL / joint-harness** effect, not input/precision/wd/cycle. **Open sub-question (decides T2P.1's value):** is it the JOINT mixed-loop specifically (→ T2P.1 staged removes reg from it) or a train.py-vs-p1 single-task harness-implementation detail (e.g. the dual-tower fused-classifier dropout 0.1 vs p1 STAN 0.3; the next_forward eval path; data ordering)? The cleanest isolation = run the isolated reg STAN as a train.py `--task next` single-task (NOT p1, NOT joint) on the (c) input — if ≈66 the joint mixed-loop is the poison (T2P.1 helps); if ≈52 it's a train.py-harness detail (staging won't help; fix the harness/recipe).

**Lesson logged (feedback-class):** the study's recurring "probe the task's REAL input artifact" gotcha bit me — but in REVERSE: `next_region.parquet`'s emb columns LOOK like the reg input but are unused; the real input is `build_region_sequence_tensor`. Always verify the tensor the dataloader actually yields (`_resolve_x`), not the parquet that shares the task's name.

---

## 2026-06-05 — TIER-2 FINAL: dual-tower CLOSES the FL gap → the MTL→STL reg gap is CLOSED at all 3 states (re-validation COMPLETE)
**Combo** (dual-tower gated + prior-OFF, FL, multi-seed, unweighted) = **73.01±0.01** (−0.30 vs (c) ceiling 73.31). ≈ dual_gated-alone (73.06, prior-ON) → prior-OFF does NOT stack with the dual-tower (the private tower already captures the prior benefit). **The dual-tower (73.06) is the FL champion, −0.25 vs ceiling (tight σ).**
**⟹ THE MTL→STL REG GAP IS CLOSED at all 3 states by a single jointly-trained MTL model:** AL base_a 64.52 (**+1.6** vs ceiling), GE base_a 57.84 (−0.6), FL dual-tower 73.06 (−0.25). What was a "−7 to −17pp irreducible architectural gap" is now ≤0.6pp (≈0 within multi-seed noise) once the loss matches the metric + (at the large state) the dual-tower is used. **Tier-2 architecture verdict: POSITIVE — the dual-tower closes the gap; it was the centerpiece NEGATIVE under the class-weighting confound.** RE-VALIDATION COMPLETE.

**Final close-out doc state (organized + pushed):** HANDOFF top = clean "you are here" (finding/fix/re-validation table/narrative-flip/provenance); PAPER_UPDATE = paper-facing reframe; CONCERNS C25 = re-validated; CLAIMS/NORTH_STAR/INDEX banners updated. **Remaining (smaller):** CANONICAL_VERSIONS new pinned version (unweighted recipe), Acc@1→Acc@10 reg-monitor code fix, FL-B9 §0.1-table recipe-continuity follow-up, optional HGI 3-way regime arm. T2P.1/2/3 UNNEEDED; T2.3/T2.4 optional stretch (push reg ABOVE ceiling, not close a gap).

---

## 2026-06-05 — ⭐⭐ TIER-2 RE-RUN under the fix → ORDERINGS FLIPPED (dual-tower WORST→BEST); FL gap closeable (prior + dual-tower)
**Phase**: Tier-2 architecture re-run under the C25 fix (FL, unweighted, multi-seed {0,1,7,100}; `c25_tier2_refix.sh`). **The user's question — "could T2.1/T2.2 be different now?" — answered: YES, decisively.**
| arm (FL, unweighted) | reg@10 (mean±sd) | Δreg vs base_a | gap vs (c) 73.31 |
|---|---|---|---|
| **dual_gated (T2.1)** | **73.06±0.07** | **+1.51** | **−0.25** |
| prior_off (base_a α=0) | 72.94±0.07 | +1.39 | −0.37 |
| crossstitch (T2.2) | 71.98±0.12 | +0.43 | −1.33 |
| base_a (ref, prior-ON) | 71.55±0.08 | 0 | −1.76 |
| hardshare (T2.0) | 71.48±0.08 | −0.07 | −1.83 |
**Findings:**
1. **THE ORDERINGS FLIPPED — the common-mode assumption was FALSE (user vindicated).** Under the OLD class-weighted recipe: `CrossStitch ≥ base_a ≈ hard-share ≫ dual-tower` (dual-tower WORST, −2.89 at FL). Under the unweighted fix: **`dual_gated > prior_off > crossstitch > base_a ≈ hardshare`** — the **dual-tower went from WORST to BEST (+1.51 vs base_a)**, nearly closing the FL gap (−0.25). The dual-tower's "loss" WAS the class-weighting interacting non-uniformly with its private reg tower. **The Tier-2 "irreducibly architectural negative" is fully OVERTURNED.**
2. **The residual FL gap is the α·log_T PRIOR, not architecture per se.** prior_off (base_a with α=0, matching the (c) ceiling's prior-OFF) = 72.94 (−0.37 vs ceiling) — closes ~80% of base_a's −1.76 gap. O1's "prior is a drag on the STL ceiling" extends to MTL.
3. **Both gap-closers stack** → dual-tower + prior-OFF should fully close / beat the ceiling (combo running). hard-share ≈ base_a (hard≈soft, unchanged); CrossStitch a small +0.43 (was the only non-loser, consistent).
**⟹ A single MTL model (dual-tower, or base_a prior-OFF) reaches the STL ceiling at FL** — the last residual is closed by architecture/prior, NOT a fundamental MTL limit. Combined with the regime + composite re-validation: the entire "MTL sacrifices reg / ship composite / architecture-negative" edifice is overturned.

**Next**: dual-tower prior-OFF combo (does it fully close/beat the FL ceiling?) → then surface the complete Tier-2 + re-validation picture + the paper-reframe decision. T2P.1/2/3 likely UNNEEDED (the gap is ~closed); T2.3/T2.4 optional (the axis is live but the gap is already closed).

---

## 2026-06-05 — wd THEORY REFUTED → the residual FL gap is NOT weight-decay; proceeding to Tier 2 (user sequence)
`c25_wd_theory.sh` (wd=0.01 vs the recipe's wd=0.05, v14, unweighted real-joint, AL/GE/FL × {0,1,7,100}):
| state | wd0.05 reg | wd0.01 reg | Δ | (c) ceiling | gap@wd0.01 |
|---|---|---|---|---|---|
| AL | 64.52 | 64.56 | +0.04 | 62.88 | +1.68 |
| GE | 57.84 | 57.80 | −0.04 | 58.45 | −0.65 |
| FL | 71.55 | 71.50 | −0.05 | 73.31 | **−1.81** |
- **wd makes ~0 difference** (±0.04, within noise) — wd=0.01 (matching the (c) ceiling) does NOT close the FL gap. **The residual FL −1.8pp is GENUINE (not class-weighting, not wd).** Candidates now: (a) joint cat-competition at the large state (architectural — Tier 2), (b) the prior axis — MTL base_a uses `next_getnext_hard` (prior-ON, α learnable) but the (c) ceiling is `next_stan_flow` α=0 (prior-OFF); O1 found prior-ON is a drag on the STL ceiling, so the FL gap may be the prior. → the Tier-2 re-run includes a prior-OFF arm to decompose this.

**Next**: Tier 2 re-run under the fix at FL multi-seed (base_a, dual-tower, CrossStitch, hard-share, + prior-OFF arm) → does any architecture close the −1.8pp FL gap? do the orderings change vs the class-weighted Tier-2 (the user's question)?

---

## 2026-06-05 — ⭐⭐⭐ MULTI-SEED RE-VALIDATION LANDED → REGIME FINDING OVERTURNED + COMPOSITE ADVANTAGE DISSOLVED (paper-redefining)
**Phase**: the paper-grade re-validation (`c25_revalidate.sh`, 24 runs: v14 vs canon × AL/GE/FL × {0,1,7,100}, unweighted real-joint, onecycle, KD-OFF). **Surface to user — this flips the central narrative.**

**Results (multi-seed {0,1,7,100}, tight σ ~0.1):**
| state | v14 reg / cat | canon reg / cat | (c) STL ceiling | (d) composite reg | Δreg(v14−canon) | v14 reg vs (d) |
|---|---|---|---|---|---|---|
| AL | 64.52 / 53.38 | 62.60 / 53.43 | 62.88 / 49.97 | 63.58 | **+1.92** | **+0.94 (BEATS composite)** |
| GE | 57.84 / 61.37 | 56.34 / 61.61 | 58.45 / 58.12 | 58.76 | **+1.49** | −0.92 |
| FL | 71.55 / 71.89 | 70.74 / 72.07 | 73.31 / 69.97 | 73.62 | **+0.81** | −2.07 |

**Four paper-redefining findings:**
1. **REGIME FINDING OVERTURNED.** Δreg(v14−canon) is POSITIVE at all 3 states (+1.92/+1.49/+0.81, σ~0.1 → significant). The old (class-weighted) regime finding said "v14 ≈ canon in MTL → the STL substrate gain washes out under the cross-attn MTL regime." With unweighted reg CE, **v14 > canon in MTL — the substrate gain TRANSFERS.** The "washes out in MTL" headline (`v14_mtl_vs_canonical.md`, CH28) was a class-weighting artifact. (The substrate Δ in MTL is smaller than the STL Δ — partial transfer — but it is POSITIVE and significant, not a wash-out.)
2. **MTL reg ≈ the STL ceiling.** v14: AL 64.52 (+1.6 vs (c)), GE 57.84 (−0.6), FL 71.55 (−1.8). The "MTL sacrifices reg −7 to −17pp" gap is GONE; what remains is a small −1.8pp residual at FL only (→ wd theory).
3. **COMPOSITE ADVANTAGE (CH25) DISSOLVED.** vs (d) composite reg: a SINGLE MTL model is **+0.94 at AL (BEATS the composite)**, −0.92 GE, −2.07 FL. The "composite = +7-12pp over MTL@disjoint" headline collapses to ~−2 to +1pp. **The 2-model composite is no longer the deployable reg answer — a single MTL model matches/beats it.**
4. **CAT also re-baselined up ~+3-5pp** — MTL cat EXCEEDS the STL cat ceiling at all 3 states (AL 53.4>49.97, GE 61.4>58.12, FL 71.9>69.97). (cat was also class-weighted; unweighting + the onecycle recipe stack — see C25 cat-axis.)

**§0.1 re-baseline (the canon arm = v11 GCN substrate = the §0.1 substrate):** MTL reg AL 62.60 / GE 56.34 / FL 70.74 (multi-seed) — vs the OLD class-weighted §0.1 MTL reg (~50/42/61) = **+10-13pp**. The §0.1 architectural-Δ table (MTL reg ≪ STL) is substantially re-stated: MTL reg now ≈ STL ceiling.

**THE NARRATIVE FLIP:** OLD = "MTL sacrifices reg (−7..−17pp); the gap is irreducibly architectural; ship the 2-model composite." NEW = **"MTL reg ≈/> the STL ceiling AND the composite once the loss matches the metric (unweighted CE vs frequency-weighted Acc@10); the substrate gain transfers to MTL; cat improves too."** This dissolves the Tier-2/2P architecture-negative AND the composite headline (CH25) AND overturns the regime finding (CH28). Caveats: residual FL reg gap −1.8pp (wd theory next); recipe = onecycle (not the §0.1 B9 — FL B9 follow-up for exact table continuity); frozen (c)/(d) untouched (unweighted STL, valid comparands).

**Next**: (per user sequence) wd theory (running, `c25_wd_theory.sh`) → does wd=0.01 close the FL −1.8pp residual? → then Tier 2 re-run under the fix (test the orderings — the user's question — at FL multi-seed). Then the doc promotions (regime/CH25/CH28 re-statement, NORTH_STAR per-task-weighting, PAPER_UPDATE reframe).

---

## 2026-06-05 — RE-VALIDATION: single-seed re-baseline CONFIRMS the reframe at AL/GE/FL; multi-seed sweep LAUNCHED (user: "both re-runs now")
**Single-seed re-baseline** (real-joint, v14, onecycle, seed42; `c25_rebaseline.sh`, best=both-unweighted vs old=both-weighted):
| state | best reg/cat | old reg/cat | (c) ceiling reg/cat | Δreg | Δcat |
|---|---|---|---|---|---|
| AL | 64.82 / 53.51 | 56.45 / 48.51 | 62.88 / 49.97 | +8.37 | +5.00 |
| GE | 57.91 / 61.31 | 48.00 / 56.94 | 58.45 / 58.12 | +9.91 | +4.37 |
| FL | 71.41 / 72.08 | 61.87 / 69.08 | 73.31 / 69.97 | +9.54 | +3.00 |
- **With the fix, MTL reg ≈ the STL ceiling at ALL 3 states** (AL +1.9, GE −0.5, FL −1.9 vs (c)) and **cat EXCEEDS the ceiling everywhere** (+2.1 to +3.5). The pre-C25 (both-weighted) recipe was −8 to −10pp reg. **The "MTL sacrifices reg" gap is gone** once the loss matches the metric. (seed42 dev-seed; multi-seed below for paper-grade.)
**Multi-seed re-validation sweep LAUNCHED** (`c25_revalidate.sh`, tracked `buunjh5f3`): **v14 vs canon (check2hgi v11 GCN) × AL/GE/FL × seeds {0,1,7,100}**, MTL real-joint, unweighted, onecycle, KD-OFF. Merges re-run #1 (regime: Δ=v14−canon — does the STL substrate gain transfer to MTL now?) + #2 (§0.1 re-baseline = the canon arm, multi-seed) + CH25 re-derivation (composite vs canon MTL reg). Seeded log_T for all seeds verified on disk. HGI 3-way arm deferred (needs log_T + AL/GE next_region). ~1.5h.
**Reframe close-out (doc-only) DONE/PENDING:** INDEX banner (Tier 2/2P RETIRED, T2.3/T2.4 CANCELLED) ✓; PENDING (after multi-seed confirms): NORTH_STAR/CANONICAL promotion of per-task weighting (a real recipe WIN: cat-unweighted +3-5pp), Acc@1→Acc@10 monitor code fix, PAPER_UPDATE full reframe.

**Next**: multi-seed sweep lands → aggregate v14-vs-canon multi-seed table (does substrate transfer? regime finding verdict) + canon §0.1 re-baseline vs (c) ceilings + CH25 re-derivation → surface the re-validated regime + §0.1 picture + the paper-narrative decision.

---

## 2026-06-05 — STRATEGIC RE-RUN ASSESSMENT (user-requested: "is it worth re-running Tier 2 / any part of the study?") → large REFRAME, narrow re-validation
**Phase**: strategic agent assessment of what the C25 fix invalidates / salvages / requires re-running, before spending re-baseline compute. **Surface to user — this is a paper-narrative-level decision.**

**THE REFRAME (the agent's bottom line):** the paper's central reg story flips from **"MTL sacrifices reg (−7 to −17pp); the gap is irreducibly architectural; ship the 2-model composite"** → **"MTL reg matches/approaches the STL ceiling once the loss matches the reported metric — the gap was an objective mismatch (class-balanced CE vs frequency-weighted Acc@10)."** This DISSOLVES the Tier-2/2P architecture-negative, SHRINKS/KILLS the composite advantage (CH25), and puts the REGIME FINDING (the track's headline) genuinely back in play. **But the re-validation cost is contained: essentially TWO multi-seed MTL sweeps.**

**Verdicts (INVALIDATED / SALVAGEABLE-common-mode / NEEDS-RERUN / KEEP):**
- **Tier 2 architecture search:** premise INVALIDATED (the "gap" was the confound; unweighted MTL reg ≥ ceiling with NO arch change). The *relative* orderings (dual-tower < base_a, dose-response, mechanism cells) are common-mode → SALVAGEABLE as relative claims, but they no longer answer a live question. **RETIRE** "irreducibly architectural" + "ship composite as *the* reg answer"; dual-tower/crossstitch code stays as tested assets, off the critical path.
- **Tier 2P (joint-loop):** MOOT — the "joint loop caps reg" was the class-weighting (already overturned). **RETIRE T2P.1/2/3.**
- **T2.3 (MoE) + T2.4 (hybrids):** **CANCEL** — they complete an architecture-negative card set that no longer exists. *Single biggest compute saving.*
- **REGIME FINDING:** **NEEDS-RERUN (highest EV)** — the v14-vs-canonical MTL tie is common-mode (likely holds), but "does the STL substrate gain TRANSFER to MTL under unweighted reg CE?" was NEVER tested (it crosses the weighted/unweighted boundary). This is the track headline. NB the α=0 0.03% floor (>70pp collapse, `phase_b_fl_3way.md:70-73`) is a DISTINCT OOD phenomenon NOT dissolved by the confound (but already hedged as regime-scoped).
- **§0.1 absolute MTL reg + CH25 composite:** **NEEDS-RERUN (very high EV)** — §0.1 MTL reg ~10-14pp understated; CH25's +7-12pp composite advantage was vs depressed MTL reg → shrinks/dissolves. Re-derive CH25 from the same numbers (no extra run).
- **CAT side:** also ~5pp understated (cat was weighted too); §0.1 cat row NEEDS-RERUN (same job); cat-relative arch claims common-mode. **Careful: cat recipe-consistency (B9 vs deployable) + class-weight corrections STACK — don't double-count.**
- **onecycle recipe (R2):** KEEP-AS-IS (recipe-level, common-mode).
- **CH26 log_T-KD (+2-5pp):** SALVAGEABLE common-mode (both arms weighted), but FLAG — if unweighted MTL reg is already at ceiling, KD headroom may shrink → fold a W=0.2 arm into the regime re-test.
- **canonical_improvement Tier 6 (selector), embedding_eval (STL-only), frozen (c)/(d):** KEEP-AS-IS (insulated).

**RANKED re-runs (EV/cost):** (1) Regime re-test: v14/design_k vs canonical vs HGI, MTL, **unweighted reg**, AL/GE/FL, seeds {0,1,7,100}, leak-free — decides the thesis; mid-tens GPU-h. (2) §0.1 re-baseline (MTL reg+cat) on the **v11 GCN paper substrate**, B9, FL multi-seed — re-establishes the central table; tens GPU-h. (3) +W=0.2 KD arm in #1. (4) Acc@1→Acc@10 reg-monitor fix (code). (5) Promote per-task-weighting (cat-unweighted +5pp = real recipe WIN) to NORTH_STAR/CANONICAL (doc-only). **DON'T re-run:** T2.3/T2.4, Tier 2P, dose-response/mechanism, onecycle matrix, canonical_improvement Tier 6, embedding_eval, (c)/(d).

**⚠ CRITICAL caution (agent):** the in-flight `c25_rebaseline.sh` is NECESSARY but NOT SUFFICIENT for any paper claim — it is **v14-substrate, single-seed-42, no canonical/HGI arm**. It settles "magnitude of the fix on one substrate at the dev seed," nothing else. Do NOT let it stand in for re-run #1 (regime) or #2 (§0.1 on v11 GCN). Per C23, seed=42 overshoots multi-seed at large states.

**Next**: re-baseline lands → aggregate AL/GE/FL best-vs-old → STOP + surface the reframe + the ranked re-run plan to the user to scope (which re-runs to launch; confirm the retire/cancel list; the paper-narrative decision).

---

## 2026-06-05 — CAT-AXIS TEST (user-requested: "did we test cat unweighted?") → BOTH heads unweighted is best; my cat=weighted default was WRONG
**Phase**: per-task class-weighting validation BEFORE the full re-baseline (user: "eval the best combination before run the executions"). I had set the cat default to WEIGHTED by REASONING ("balancing helps macro-F1") — the user (correctly) insisted on testing it.
**Real joint recipe** (mtlnet_crossattn + next_getnext_hard reg + next_gru cat, category-weight 0.75, onecycle, v14, AL, 5f×50ep seed42):
| arm (reg-CE / cat-CE) | reg@10 disj | cat-F1 disj | reg@10 dep | cat-F1 dep |
|---|---|---|---|---|
| old (weighted / weighted) — pre-C25 | 56.45 | 48.51 | 56.06 | 48.51 |
| fix (UNweighted / weighted) | 64.51 | 48.37 | 61.98 | 48.14 |
| **catu (UNweighted / UNweighted)** | **64.82** | **53.51** | **64.20** | **53.47** |
| (c) STL ceiling | 62.88 | 49.97 | — | — |
- **Reg axis (old→fix): +8.06pp** (56.45→64.51 ≥ ceiling 62.88) — the reg fix holds on the REAL joint recipe (not just the cat-0 isolation).
- **Cat axis (fix→catu, i.e. cat-weighted→cat-unweighted at reg-unweighted): +5.14pp cat-F1** (48.37→53.51). **My "class-balancing helps macro-F1" assumption was FALSE** — unweighted cat is BETTER for macro-F1 (the balanced averaging is over per-class F1, but the WEIGHTED loss destabilises the 7-class cat head under joint training). cat-unweighted also slightly helps reg (64.82 vs 64.51, within noise).
- **⟹ BOTH heads UNWEIGHTED is the validated best default** (reg 64.82 ≥ ceiling AND cat 53.51 > ceiling 49.97). **Code default flipped to both-unweighted** (`default_mtl use_class_weights_{reg,cat}=False`). The cat axis is being re-confirmed at GE/FL in the full re-baseline. **Lesson: don't reason about loss-vs-metric interactions — TEST them (the user's instinct, twice now: use_class_weights itself + the cat default).**

**Next**: full re-baseline (best=both-unweighted vs old=both-weighted × AL/GE/FL) running → settles the reg+cat axes at scale + the MTL-vs-ceiling claim → then the regime-finding re-test.

---

## 2026-06-05 — RE-AUDIT (user-requested: "are we missing another use_class_weights?") → class-weighting is the SINGLE dominant confound; 2 smaller real secondaries flagged
**Phase**: post-root-cause skeptical re-audit (independent agent + my config-factory diff). **Verdict: class-weighting is THE dominant confound; NO second bug of comparable magnitude.** The user's worry is addressed.
- **My catch (config diff):** `default_next` is `use_class_weights=True` (`experiment.py:444`), NOT False — corrected the propagated error in HANDOFF/CONCERNS/log. The (c) ceiling is **p1/unweighted** (`build_calibrated_loss` no-calibration = plain CE; synthetic-verified bit-identical to `F.cross_entropy`), not `default_next`.
- **Why the probes hit 66 (the mechanism, now explained):** `t2p0_mechanism_probe.py` + `t2p0_singletask_isolation.py` both use **unweighted `F.cross_entropy`** → they silently dropped the class weighting → that's why "mixed loader" / "full-model forward" couldn't reproduce the gap. The dismissals were CORRECT; the weighting was the poison.
- **Secondary #1 (real, MEDIUM): wd=0.05 (MTL global default) vs 0.01 (STL ceiling + `default_next`).** `default_mtl weight_decay=0.05` (`experiment.py:347`); T2P.0 patched it to 0.01, but the **real §0.1 canonical recipe runs wd=0.05 UN-overridden** → the (a)-vs-(c) comparison carries a wd mismatch ON TOP of class-weights. Same silent-asymmetry class. **Re-baseline must test wd=0.01 vs 0.05 (both unweighted) — if Δ>1pp, wd-match the (a)-vs-(c).**
- **Secondary #2 (real, smaller): the Acc@1 reg checkpoint monitor** (`best_tracker.py:116 reg_monitor='accuracy'`, preset `primary_metric=ACCURACY`) — ACTIVE post-C2-fix (`train.py:206-208` wires it). Costs ~3.5-4pp (`MTL_FLAWS_AND_FIXES §2.10`) BUT only on the **deployable / `diagnostic_best_epochs[...]['metrics']`** number — NOT `per_metric_best.top10_acc_indist.best_value` (top10's own argmax, mismatch-free), which the study reads (memory `ref_mtl_metric_field`). Fix: set reg monitor to a top10 metric, OR enforce the per_metric_best read. Cheap, worth doing.
- **Cleared (re-confirmed):** weighted sampling (OFF both — `FoldCreator` default False), fp16 (non-causal), dataloader kwargs, grad-accum/clip, optimizer betas/eps, calibrated-loss defaults, per-fold log_T rebuild (MTL stricter).
- **Part D — fix GENERALIZES** to the real joint recipe (class-weight code is recipe-agnostic). ONE interaction: `--no-class-weights` unweights BOTH cat+reg at category_weight=0.75 → verify cat F1 holds (cat is 7-class balanced → small; → per-task weighting if it regresses). Exact real-joint re-baseline command captured in the agent report + task #11.

**Next**: implement per-task class-weighting (reg OFF, cat tbd) + the Acc@1 monitor fix (task #10), then the AL/GE/FL re-baseline incl. the wd=0.01-vs-0.05 control (task #11).

---

## 2026-06-05 — ⚠⚠ MECHANISM PROBE (user-requested) → the reg gap is `mtl_cv`-IMPLEMENTATION-SPECIFIC, NOT structural; a faithful joint reconstruction reaches the CEILING

**Phase**: Tier 2P — mechanism probe of WHY the joint loop costs 10-14pp. **Surfaced to user — this is bigger than T2P.1 and may reframe the regime finding.** FL mixed-loop confirmation in flight.

**The result — four independent reconstructions reach the (c) ceiling; ONLY `mtl_cv` is the outlier:**
| harness | AL reg@10 | FL reg@10 | notes |
|---|---|---|---|
| p1 (blessed STL ceiling) | 62.88 | 73.31 | the frozen (c) |
| my standalone head, single-task loop | 62.88 | 73.12 | `t2p0_singletask_isolation.py` |
| my FULL model + per-head opt, single-task | 63.01 | — | `t2p0_mechanism_probe.py` |
| my FULL model + per-head opt, **real-cat MIXED loop** (cat-weight 0) | **63.20** | (pending) | `--mixed` |
| **`mtl_cv` joint loop (T2P.0)** | **52.90** | **59.53** | the outlier |

**Controls ruled out as the cause** (verified identical between my reconstruction and T2P.0): input (byte-identical), head, recipe (LR/wd/onecycle/epochs/bs), folds, precision (fp32 control), **grad accumulation** (default_mtl=1, not the dataclass-default 2), **loss weighting** (use_class_weights=False → unweighted CE both), **grad-clip** (1.0 both), **metric selection** (`per_metric_best.top10_acc_indist` = oracle best epoch), **early stopping** (T2P.0 ran all 50 epochs — verified from per-epoch CSV), **mixed-batch structure** (my real-cat mixed loop reaches 63.20), **full-model forward + per-head optimizer** (my full-model probe reaches 63.01).

**T2P.0 reg trajectory (FL fold1, `metrics/fold1_next_region_val.csv`):** rises 0→58 by ep6, then SLOWLY climbs to a **plateau at 59.30 (peak ep44)**, ran all 50 ep. A genuine **low training plateau**, NOT an early-peak-crash, NOT a measurement/selection/early-stop artifact. `mtl_cv` trains reg to a ~14pp-worse optimum than the identical head/input/recipe in any other harness.

**⟹ THE FINDING (strong + surprising + high-value):** the MTL→STL reg gap is **specific to the `mtl_cv` training-loop implementation**, NOT to MTL/joint-training per se, NOT the architecture/substrate/input/recipe. A faithful re-implementation of the SAME joint training (full model + per-head optimizer + real-cat mixed loop + cat-weight 0) reaches the **(c) ceiling**. Load-bearing because:
- The gap may be a **recoverable `mtl_cv` issue** — fixing it could lift MTL reg ~10-14pp **without staging** (T2P.1 may be unnecessary).
- ALL the study's MTL reg numbers (the **regime finding**, the dose-response, §0.1 deployable reg) ran through `mtl_cv` → may be **systematically ~10-14pp depressed**. The Tier-2 "joint loop caps reg" negative is REAL (mtl_cv does cap reg) but its CAUSE is an implementation detail, not a fundamental MTL limitation.

**⚠ HONEST CAVEAT (do NOT over-claim yet):** my reconstructions independently match the **blessed p1 ceiling** (strong evidence they're correct, not leaky — 4 harnesses incl. p1 sharing a leak while mtl_cv alone is correct is low-probability). BUT the EXACT `mtl_cv` behavior causing the deficit is **NOT yet pinpointed**. Before any paper claim or "free reg recovery," the next step MUST pinpoint it — bisect mtl_cv toward the reconstruction, or run mtl_cv with checkpoints + re-eval reg with a direct forward, or diff the dataloader/criterion/step path. Residual chance my reconstruction omits something mtl_cv does correctly (making 63 optimistic) — p1 agreement argues against it.

**Chain status**: Tier 2P — T2P.0 resolved AND now localized to the `mtl_cv` implementation. SUPERSEDES the simple "→ T2P.1 staged" plan. Frozen (c)/(d) untouched.

**Next**: **STOP + surface to user.** Decide: (a) pinpoint the exact mtl_cv cause (decisive next probe — bisect/checkpoint-reeval), (b) re-assess T2P.1 vs an mtl_cv fix, (c) gauge blast radius on the regime finding + §0.1. Do NOT autopilot into T2P.1.

---

## 2026-06-05 — PINPOINT (axis nailed) + ASSESS (blast radius) — user chose "pinpoint and assess"

**PINPOINT — axis = TRAINING, not measurement; exact line needs a runtime bisect.** Continued ruling-out:
- **Metric RULED OUT (empirical):** added mtl_cv's exact `top10_acc_indist` (mtl_eval.py:28-51) to my reconstruction → FL indist **73.7** (≈ plain 73.1); mtl_cv indist 59.53. Same metric, ~14pp apart. indist ≥ plain (excludes cold-start), so the metric difference can't explain it.
- **Dataloader RULED OUT:** `POIDataset.__getitem__` returns raw `(features[idx], labels[idx])` — no normalization/transform; loader is shuffle + no weighted sampling (use_weighted_sampling=False). Same tensor my reconstruction feeds (build_region_sequence_tensor, byte-identical to (c)).
- **Training-forward code RULED OUT:** mtl_cv:478-485 is `model((x_a,x_b))` → `CrossEntropyLoss(out_next)` — identical to my reconstruction.
- **Axis = TRAINING:** my indist eval (= mtl_eval's) gives 73.7 on good weights; mtl_cv's same-metric eval gives 59 on its own weights → same eval, worse weights → mtl_cv TRAINS reg to a worse optimum. (Gold-standard confirmation = checkpoint-reeval of mtl_cv weights with a direct forward; deferred.)
- **Cumulative ruled-out (none is the cause):** input-tensor, head, recipe, optimizer (per-head), precision, grad-accum, loss-weighting, grad-clip, metric (plain+indist), early-stop, mixed-batch-structure, full-model-forward, training-forward-code, dataloader-transform. The exact `mtl_cv` line is UNPINNED — it requires a **runtime bisect** (progressively morph the passing reconstruction `t2p0_mechanism_probe.py --mixed` into the mtl_cv loop until reg drops to 59; or instrument mtl_cv). That is the clean next-session task. **Surprising honest state: a 14pp deficit with every obvious cause excluded ⟹ a subtle mtl_cv interaction, OR (low-prob, p1-agreement argues against) a shared optimism in 4 reconstructions.**

**Final reconstruction table (both states):**
| harness (identical head/input/recipe/folds/fp16) | AL reg@10 | FL reg@10 |
|---|---|---|
| p1 (blessed (c) ceiling) | 62.88 | 73.31 |
| standalone single-task | 62.88 | 73.12 (indist 73.7) |
| full-model single-task | 63.01 | — |
| full-model real-cat MIXED loop (cat-wt 0) | 63.20 | 73.16 |
| **mtl_cv joint (T2P.0)** | **52.90** | **59.53** |

**ASSESS — blast radius (structural reasoning; common-mode cancellation):**
- **RELATIVE findings are UNAFFECTED** (both arms run through mtl_cv, so a ~14pp common-mode reg depression cancels in the Δ): the **regime finding** (v14-MTL ≈ canonical-MTL → substrate washes out jointly), the **Tier-2 architecture negative** (dual-tower < base_a; the dose-response), and all within-MTL comparisons HOLD. The paper's central negatives don't move.
- **ABSOLUTE MTL-reg numbers are potentially ~14pp understated** (all ran through mtl_cv): §0.1 deployable reg, the **MTL→composite reg gap**, and the "MTL sacrifices reg" magnitude. **If the mtl_cv deficit is real + fixable, MTL reg could approach the STL ceiling → the composite advantage shrinks dramatically or dissolves, and "MTL sacrifices reg" largely goes away.** This is the high-EV upside — but GATED on the runtime-bisect confirmation (don't re-state §0.1 or the composite gap until the exact cause is found + a fix validated leak-free).
- **STL ceilings (c)/(d) UNAFFECTED** (p1 harness, not mtl_cv) — they remain valid; they're the target MTL reg should be able to reach if the mtl_cv deficit is fixed.

**Chain status**: Tier 2P — T2P.0 resolved; the reg deficit is mtl_cv-training-specific (axis nailed, exact line pending a runtime bisect). Relative findings safe; absolute MTL-reg numbers are the open upside. Frozen (c)/(d) untouched.

**Next**: surface pinpoint+assess to user. Recommended decisive next step = the **runtime bisect** (morph the passing `--mixed` reconstruction into mtl_cv) to find the exact line, THEN decide fix-mtl_cv (free reg recovery, re-baseline) vs T2P.1 (staged). Do NOT re-state any absolute reg number until the cause is found + fix validated.

### Advisor pass (2026-06-05, user-requested before the bisect) — verdict: SOUND, but CHECKPOINT-REEVAL FIRST; 2 corrections applied
Adversarial review of the finding + the bisect plan. **Verdict: finding SOUND (reconstructions NOT leaky — fold split byte-identical + user-disjoint, `n_regions` matches mtl_cv's cross-fold max, the p1-agreement is independent not a shared flaw); load-bearing exclusions HOLD** (re-derived: cat-weight-0 zeroes cat grad; OneCycleLR scalar-max_lr flattens per-head LRs identically in BOTH — a real no-op under onecycle, NOT under constant/cosine; gradient-cosine `autograd.grad(retain_graph=True)` is benign; cat/reg equal-N so no cycling; eval path at `private_only` is logit-identical across full-forward/next_forward/standalone → measurement CANNOT differ for identical weights). **Two changes:**
1. **CHECKPOINT-REEVAL FIRST (before any bisect).** "Axis = training" is ARGUED not PROVEN — the probes ran `--no-checkpoints` (no weights saved). The cheap gold-standard: re-run T2P.0 (AL+FL) with `--checkpoints`, load the saved per-fold reg weights, re-eval Acc@10 with a direct `next_forward`. ≈59 → training-axis confirmed → bisect; ≈73 → a measurement/summary artifact inside mtl_cv (framing collapses to a reporting bug, MUCH cheaper). Do NOT bisect or re-state any absolute reg number until this returns.
2. **Blast-radius TIGHTENED — the deficit is NON-UNIFORM (state-dependent: AL −9.98 vs FL −13.6, scales with class-count/N).** So common-mode cancellation is guaranteed ONLY for **same-state, same-arch, same-N** Δs (v14-MTL vs canonical-MTL *at a fixed state*; dual-tower vs base_a *at a fixed state* — those HOLD). **Cross-state aggregates + any arch comparison that changes the reg head's class space/difficulty are NOT guaranteed common-mode → must be re-checked after the fix.** (Supersedes the "absolutes shift, relatives all safe" wording above.)

### CHECKPOINT-REEVAL DONE (2026-06-05) → TRAINING-AXIS CONFIRMED (gold-standard)
Re-ran T2P.0 AL with `--save-task-best-snapshots` (reproduced reg disjoint **52.90** exactly), then loaded each fold's `task_best_snapshots/fold{N}_reg_best.pt` (mtl_cv's ACTUAL reg-best full-model weights) and re-evaluated reg Acc@10 with a direct `next_forward` + my verified top-k, INDEPENDENT of mtl_cv's eval/summary code (`scripts/mtl_improvement/t2p0_ckpt_reeval.py`). Result: **plain 50.51 / indist 52.13** (5 folds) — matches mtl_cv's reported 52.90, NOT the 62.88 reconstruction. **⟹ mtl_cv's reg WEIGHTS are genuinely at ~52; the deficit is TRAINING (mtl_cv converges reg to a worse optimum), NOT a measurement/reporting artifact.** (next_poi keys all loaded — reproducing ~52 proves it; the 55 missing/20 unexpected keys are the ignored cross-attn path, harmless under private_only.) The advisor's required gate is GREEN → the bisect is justified.

**Bisect plan (checkpoint-reeval CONFIRMED training-axis ✓) — ranked suspects:** (1) the **DataLoader construction** `folds._create_dataloader` (num_workers / persistent_workers / `worker_init_fn` / drop_last / worker-seeded shuffle) — the probes use plain `TensorDataset` + `num_workers=0 drop_last=False`; (1b) the aux-loader path IF it wraps reg (NB: `next_stan_flow_dualtower` is NOT in `_HEADS_REQUIRING_AUX_MTL` → use_aux=False, so likely moot — verify); (2) `evaluate_model` val coverage via `zip_longest_cycle` (mtl_eval.py:116) — count distinct val samples scored vs `len(va)`; (3) de-prioritised: static_weight path, gradient-cosine, per-head opt, OneCycleLR (all ruled out from code).

### BISECT-BY-READING EXHAUSTED (2026-06-05) → cause is an EMERGENT runtime interaction; every static component ruled out
Worked the ranked suspects + everything else. **All RULED OUT** (each verified identical between mtl_cv and the passing reconstruction, OR shown not to matter): DataLoader kwargs (`_create_dataloader`: shuffle=True, num_workers=**0** on this box, drop_last=False, dataset pre-moved to DEVICE — functionally identical to the probe); aux-loader (use_aux=False for `next_stan_flow_dualtower` — confirmed, no AuxPublishingLoader); **fold split** (no frozen cache → mtl_cv generates fresh `sgkf(42)` = the probe's split — run log "Generating folds on the fly"; reeval matching 52.90 confirms same val); **model fresh per fold** (mtl_cv.py:1155, inside the loop — no reuse); **torch.compile OFF** (config.use_torch_compile default false); **freeze_cat_stream OFF**; **scheduler identical** (`setup_scheduler` pct_start=None → OneCycleLR default 0.3 = the probe's explicit 0.3; div_factor/anneal defaults match); **reg head IDENTICAL** (checkpoint key-diff: the 55-missing/20-unexpected are ALL `category_poi` = next_mtl-vs-next_gru cat head — ZERO `next_poi` diff); **cat head ELIMINATED empirically** (probe with `task_a_head_factory=next_gru` to match mtl_cv = AL **62.85** ≈ ceiling, NOT 52).

**⟹ The deficit lives in NO single static component.** The passing reconstruction (`t2p0_mechanism_probe.py --mixed`: full model + next_gru cat + per-head opt + onecycle pct0.3 + real-cat mixed loop + cat-weight 0 + fp16 + identical fold split + identical reg head + identical data) reaches **62.85 (AL) / 73.16 (FL)**; mtl_cv with all the same reaches **52.90 / 59.53**. The cause is therefore an **EMERGENT runtime interaction in the mtl_cv loop** (per-step optimization-trajectory divergence, the diagnostic-callback/criterion machinery, or a runtime coupling none of the static reads expose) — it cannot be pinned by reading. The next step to find the EXACT line is a **step-level runtime trace**: instrument mtl_cv (env-gated) to dump, for fold 1, the reg head's per-step grad-norm + per-epoch reg val Acc@10, run BOTH mtl_cv and the probe at an identical seed/init/batch-order, and localize WHERE they diverge (batch-1 → per-batch op; later → accumulation/RNG). This is a focused debugging session (1-3 instrumented runs).

### ⚠⚠⚠ INDEPENDENT AGENT BREAKTHROUGH (2026-06-05, user-requested eval before the trace) → the deficit is a `train_with_cross_validation` WRAPPER bug, NOT emergent; reg CAN reach the ceiling in the production loop
A fresh expert agent (212k tokens) **FALSIFIED my "emergent / no static component" conclusion** with a hard bit-level test:
- **Production `train_model` (`mtl_cv.py:187`) called DIRECTLY with production `FoldCreator` folds → reg PEAK Acc@10 = 68.33 (AL fold1, ≥ ceiling).**
- **The SAME `train_model` called via the `train_with_cross_validation` WRAPPER (`mtl_cv.py:1005-1392`) → 55.27** (bit-identical to the on-disk T2P.0 trajectory).
- Trajectories diverge from **epoch 1** (direct 3.2 vs wrapper 0.9). Every declared knob verified identical (wd, per-head LRs, max_lr, onecycle, grad_accum=1, static_weight cat_w=0 → reg_w=1, optimizer group membership 24/24 private-tower params in reg group, scheduler total_steps=250, model bit-identical 3,882,084 params). The reg subgraph is fully isolated in private_only (`grad_norm_next_region_shared=0.0` on disk). Per-step reg grad + post-1-step reg weights **bit-identical** (max|Δ|=0.0) between the probe path and the static_weight+grad-cosine path → those are PROVEN non-causes.
- **⟹ The fault is in the wrapper PREAMBLE (`mtl_cv.py:1021-1392`)** — some per-fold setup op perturbs the SAME train_model from epoch 1, despite matching configs. Agent could not pin the single line; recommends a **deletion-bisect** of the preamble (strip blocks until 55→68; 3-5 cheap single-fold runs) over my grad-trace plan.
- **SECOND independent artifact:** the reg SNAPSHOT is selected by `MultiTaskBestTracker.reg_best` monitor = **Acc@1** (`best_tracker.py:116` default `reg_monitor='accuracy'`; preset `primary_metric=ACCURACY` `presets.py:100`), whose best epoch ≠ the Acc@10-best epoch → the deployable reg number is further understated ~2-3pp (agent: monitor-selected 65.70 vs Acc@10 peak 68.21 on the direct run). NB this is SEPARATE from the disjoint metric the study cites (`per_metric_best.top10_acc_indist` = oracle Acc@10, selector-independent) — so the ~13pp DISJOINT deficit (52.90 vs 68) is the WRAPPER-TRAINING bug, the Acc@1 issue is an additional deployable-number understatement.
- **Reconstruction soundness CONFIRMED:** production `train_model` (not just my hand-written probe) reaches 68 → the reconstructions + p1 are RIGHT; the wrapper is the anomaly. The agent's blast-radius: the "MTL sacrifices reg / irreducible joint-loop wall" framing is an artifact of (a) the wrapper bug + (b) the Acc@1 selection — NOT a real architectural ceiling. Fixing it lifts MTL reg toward ~68 study-wide → collapses the "MTL sacrifices reg" narrative + shrinks the composite advantage. **Do NOT re-state §0.1 absolute MTL-reg until the wrapper line is pinned + a leak-free fix validated.**

### Disambiguation (mine, 2026-06-05): the wrapper bug is PER-FOLD, NOT the fold-0 FLOPs profiling
The FLOPs block (`mtl_cv.py:1317-1327`) does `next(iter(reg_loader))` (consumes shuffle RNG) + a forward pass (dropout RNG) but only `if history.flops is None` = **fold 0 only**. Checked the on-disk T2P.0 per-fold ORACLE peak Acc@10: **fold1 55.27 / fold2 55.28 / fold3 55.65 / fold4 52.58 / fold5 45.71 — ALL ~55**, not just fold0. ⟹ FLOPs (fold-0-only) is RULED OUT; the perturbation runs EVERY fold. Remaining per-fold preamble suspects to deletion-bisect: the optimizer/scheduler build order, `history.set_model_parms`→`optimizer.state_dict()`/`scheduler.state_dict()` (1295-1315), the per-fold task_set/head rebuild (1038-1152; agent tested log_T-load = 68, so likely not it), the criterion/`compute_class_weights` build (1187-1291), `MultiTaskBestTracker` (1336-1354). **NOTE the puzzle: my hand-written probe (setup_per_head_optimizer + OneCycleLR + a hand loop) = 63, the agent's direct REAL train_model = 68, the wrapper = 55 — so the wrapper's per-fold SETUP (not train_model, not the optimizer/scheduler functions which the probe also uses) is the locus.**

### ✅✅✅ ROOT CAUSE FOUND + VERIFIED (2026-06-05) — the reg deficit is the CLASS-WEIGHTED reg CE loss (an objective mismatch), NOT the wrapper, NOT architecture
Deletion-bisect agent (218k tokens) + my independent verification PINNED it. **The prior "wrapper preamble" localization was a red herring** (the agent's "direct train_model = 68" simply used UNWEIGHTED CE). The actual cause:
- **`src/configs/experiment.py:364` — `default_mtl` sets `use_class_weights=True`** (dataclass default :235 also True). The STL reg **ceiling (c) was set by `p1_region_head_ablation.py`** = **unweighted** CE (`build_calibrated_loss`, no calibration). **[re-audit correction 2026-06-05: `default_next` :444 is ALSO True; only `default_category` :403 is False — my earlier "default_next=False" was wrong. The (c) ceiling is p1/unweighted, not default_next.]** This flows to **`mtl_cv.py:1283-1291`**: `next_criterion = CrossEntropyLoss(weight=alpha_next)` where `alpha_next = compute_class_weights(...)` (:1276). **The MTL reg head trains on class-BALANCED CE.** On the ~1109-region imbalanced head (~22% majority), class-balancing up-weights rare regions / down-weights frequent ones → optimizes MACRO accuracy and AWAY from top-K → depresses `top10_acc_indist` from epoch 1, plateau ~55 vs ~68. Per-fold (every fold weighted) → all-folds-~55 ✓. State-scaling EXPLAINED: more regions → more imbalance → bigger distortion (FL −14 > AL −10) ✓.
- **The STL (c) ceiling (p1) + my reconstructions + the agent's direct train_model all used UNWEIGHTED CE → 68.** So the MTL-vs-STL-ceiling gap was substantially an **objective mismatch** (class-weighted MTL reg vs unweighted STL reg), NOT a fundamental MTL/architecture limitation.
- **VERIFICATION (independent, mine):** T2P.0 AL + `--no-class-weights` → reg disjoint **64.81** (5-fold mean) ≥ STL (c) ceiling 62.88 (agent's fold1 = 68.69); buggy default = 52.90. Deficit GONE. The flag `--no-class-weights` (train.py:486 → use_class_weights=False) is the fix; no code change needed for the per-run fix.
- **My earlier "loss-weighting ruled out" was WRONG** — I checked `getattr(config,"use_class_weights",False)` (the fallback) but `default_mtl` sets it True; the actual config value is True. Lesson: check the factory default, not the getattr fallback.

**⟹ IMPLICATIONS (study-redefining — needs author sign-off; do NOT auto-act):**
1. **The "MTL sacrifices reg / irreducible architectural wall / ship-the-composite" narrative is substantially an artifact of class-weighting the MTL reg loss** (vs unweighted STL ceiling). With matched unweighted CE, MTL reg ≈/≥ the STL ceiling → the composite advantage shrinks/dissolves; Tier-2/2P's architecture search was attacking a confound.
2. **Blast radius — ALL `default_mtl` MTL runs used `use_class_weights=True`** (undocumented in NORTH_STAR/CANONICAL_VERSIONS). §0.1 absolute MTL reg, the MTL→composite gap, the dose-response absolutes are ~10-14pp depressed. RELATIVE within-MTL Δs (v14 vs canonical, dual-tower vs base_a — all class-weighted) are COMMON-MODE and likely hold — **BUT the REGIME FINDING needs re-examination**: it concluded "STL substrate gains wash out in MTL" by comparing depressed-MTL-reg to full-STL-reg; whether the substrate gain RE-APPEARS at MTL under unweighted CE must be re-tested (the central headline of the whole track is in question).
3. **Per-task nuance:** `--no-class-weights` is a SINGLE flag coupling BOTH cat + reg criteria (`mtl_cv.py:1284-1290`). Reg (Acc@10) wants UNWEIGHTED; cat (macro-F1) may BENEFIT from weighting. A clean fix likely needs **per-task class-weighting** (reg off, cat decided separately) — a code change. (In T2P.0 cat-weight=0 so cat is moot here.)
4. **Second artifact (separate, smaller):** reg deployment snapshot selected by Acc@1 (`best_tracker.py:116` reg_monitor='accuracy') understates the deployable reg ~2-3pp — independent of the disjoint metric.

**Chain status**: the T2P.0 "joint loop caps reg" finding is OVERTURNED — the cap was the class-weighted reg loss. Frozen (c)/(d) untouched (they're unweighted STL, valid). The fix is verified; the study-wide re-baseline + the regime-finding re-test + the per-task-weighting code decision need the user.

**Next**: SURFACE to user (major). Decisions: (a) adopt unweighted reg CE (the fix) — per-run flag now, default flip + per-task weighting later; (b) re-test the REGIME FINDING under unweighted CE (does the substrate gain transfer to MTL?); (c) scope the §0.1 / composite-advantage re-baseline; (d) the Acc@1→Acc@10 reg_monitor fix. Do NOT re-state absolute MTL-reg numbers until (a)-(c) done leak-free.

---

**STRATEGIC FORK (SUPERSEDED by the root-cause above — kept for trail):** the reg recovery is ALREADY PROVEN achievable by a CLEAN loop (the reconstruction = ceiling) AND by the production train_model called directly (68). So two paths: **(A) find+fix the exact mtl_cv line** → free ~14pp reg recovery across ALL MTL runs + re-baseline §0.1/composite (best outcome, but needs the runtime trace — open-ended); **(B) PIVOT to T2P.1 (staged / detached-reg training)** → achieves the SAME reg recovery in one deployable model by training reg OUTSIDE the buggy joint loop (proven to hit the ceiling), WITHOUT needing the exact line. (A) fixes the science study-wide; (B) ships the outcome now. They're complementary — B can proceed while A is investigated.

---

## 2026-06-05 — ✅ JOINT-LOOP ISOLATION (user-requested decider) → the JOINT TRAINING LOOP is the poison; T2P.1 (staged) WILL recover reg to ceiling

**Phase**: Tier 2P — the decisive isolation that resolves T2P.0. **Surfaced to user next.**

**The experiment** (`scripts/mtl_improvement/t2p0_singletask_isolation.py`): train the EXACT T2P.0 reg head (`NextHeadStanFlowDualTower`, `fusion_mode=private_only`, prior-OFF — self-contained: forward ignores shared `x`, reads only raw_region_seq) on the IDENTICAL input (`build_region_sequence_tensor`, byte-identical to (c)) + IDENTICAL folds (`StratifiedGroupKFold(5, shuffle, random_state=42).split(X, y_cat, groups=userids)` — same as `folds.py`) + IDENTICAL recipe (AdamW wd=0.01, OneCycleLR max_lr=3e-3 pct_start=0.3, 50ep, bs2048, **fp16**) — but in a **SINGLE-TASK loop**: no cat loader, no `max_size_cycle` mixed iteration, no per-head optimizer. So vs T2P.0 the only removed variable is the joint `mtl_cv` trainer machinery.

| | single-task loop (this) | T2P.0 joint loop | (c) STL ceiling | joint-loop cost |
|---|---|---|---|---|
| AL reg@10 | **62.88 ± 3.79** | 52.90 | 62.88 | **−9.98** |
| FL reg@10 | **73.12 ± 0.41** | 59.53 | 73.31 | **−13.6** |

**VERDICT — the JOINT mtl_cv training loop is the poison.** The IDENTICAL head/input/recipe/folds/precision reaches the **(c) ceiling** in a plain single-task loop (AL 62.88 = ceiling; FL 73.12 ≈ 73.31) but loses **10–14pp** in the joint loop. This is decisive: the MTL→STL reg gap is NOT the head, NOT the input (byte-identical), NOT precision (fp32 control), NOT wd, NOT cycle, NOT the topology, NOT cat interference (cat-weight 0 = zero cat gradient), and NOT a generic train.py-vs-p1 harness detail (this single-task loop IS train.py-side code + fp16 and it hits the ceiling). **It is specifically the joint `mtl_cv` training loop.**

**Consequences:**
1. **Tier-2P premise VINDICATED + localized.** "The joint loop caps reg" is now empirically pinned to the joint trainer itself, the cleanest possible statement of the P4 finding.
2. **T2P.1 (staged) is the clear lever AND its reg arm is PROVEN.** Staged trains reg single-task → reg = ceiling **by construction** (this experiment IS that: single-task reg = 62.88/73.12 ≈ ceiling). So a single deployable model CAN carry ceiling-quality reg — the headline the track has chased. **The ONLY open T2P.1 question is whether CAT survives a frozen-reg trunk** (the framing trap, now sharply isolated). The 2-model composite remains the null to beat.
3. **Open SCIENTIFIC puzzle (mechanism, not blocking):** WHY does the joint loop cost 10-14pp when cat-weight=0 (cat contributes ZERO gradient and reg never reads the cross-attn in private_only)? Mechanistically the reg params should get identical gradients. Candidates: (a) mixed-batch iteration perturbs dropout RNG / batch-order for the reg stream; (b) a subtlety in mtl_cv's reg eval (`next_forward` zero-A partial) vs a direct forward — though the deployable geom_simple metric corroborates ~52-59, arguing against a pure measurement artifact; (c) some mtl_cv plumbing (per-head optimizer / full-model forward) interaction. Worth a targeted probe before the paper claims the mechanism; does NOT change the lever (T2P.1).

**Caveat (honest):** the single-task loop is my minimal trainer (standard AdamW/onecycle/CE/fp16), not literally mtl_cv-with-cat-removed. It is faithful to the head/input/recipe/folds/eval, and it reaches the ceiling — so it bounds the result cleanly (a plain loop suffices; the joint loop does not). A future tightening could add a `--single-task` path INSIDE mtl_cv to remove the last plumbing differences, but the 10-14pp recovery is far outside that residual.

**Chain status**: Tier 2P — T2P.0 RESOLVED (joint loop = poison); T2P.1 strongly motivated + reg-recovery proven. Frozen (c)/(d) untouched.

**Next**: **STOP + surface to user** → on approval, run **T2P.1 (staged: reg→freeze→cat)** focusing on the real open question (does cat survive the frozen-reg trunk? does the one model beat the composite?). Optionally a short mechanism probe (why the joint loop costs 10-14pp at cat-weight 0). Then finish T2.3/T2.4 confirmatory.

---

## 2026-06-04 — [SUPERSEDED BY THE RETRACTION ABOVE] hypothesis: T2P.0 gap = input-artifact confound

**Phase**: Tier 2P — the T2P.0 fp32 control + a root-cause dig SEEMED to overturn the linchpin's premise. **This hypothesis was REFUTED in-session by the byte-identical tensor test above — kept for the trail.**

**fp32 control (`t2p0_fp32_control.sh`, MTL_DISABLE_AMP=1, FL+AL):** reg@10 disj **AL 52.92 (vs fp16 52.90, +0.02) / FL 59.66 (vs fp16 59.53, +0.13)**. **fp16-no-GradScaler precision is EXONERATED** (~0pp). So the ~10-14pp gap survives EVERY dynamics control: topology, interference (cat0), prior, wd, cycle-starvation, AND precision.

**Root-cause dig (the find).** With all dynamics ruled out, the residual must be a train.py-MTL vs p1-STL **pipeline** difference. Traced the (c)-ceiling provenance: `t14_sweep.sh`/`t14_validate_azge.sh` built (c) via `p1_region_head_ablation.py --heads next_stan_flow --input-type region --region-emb-source <v14> --override-hparams freeze_alpha=True alpha_init=0.0`. **`--input-type region`** routes through p1's `_build_region_sequence_tensor` = a **pooled `region_embeddings.parquet` lookup** ([n_regions,64], 4703 rows at FL). But the MTL reg head + the dual-tower private tower are fed **`next_region.parquet`** ([N,9,64] check-in-level contextual emb). p1's OWN `--input-type` enum proves these are two different artifacts: `checkin`→`next_region.parquet` (= the MTL input), `region`→pooled lookup (= the (c) ceiling).

**MEASURED — they are different embedding spaces** (`next_region` per-step vs `pooled[last_region_idx]`, 2000 rows/state): **0/2000 exact matches; cosine mean AL 0.197 / AZ 0.111 / FL 0.125; nearest-pooled dist 3.83**. NOT the same input.

**⟹ The dual-tower design's central premise is FALSE.** `T2.1_DUALTOWER_DESIGN.md §1` claims "the dual-tower adds a private STAN on the raw 64-dim sequence (**exactly the STL path**)" and feeds it `next_input`. But `next_input` = `next_region.parquet` (check-in-level), while the (c) STL ceiling ate the pooled `region_embeddings.parquet`. **The private tower was NEVER a true (c) replica — it ate a different (apparently worse-for-reg) input.** So:
- **T2.1 (dual-tower) and T2P.0 (linchpin) are both CONFOUNDED by input representation.** "The private STL-topology tower trained jointly loses ~10pp to STL-standalone" — the basis of the entire Tier-2P redirect (HANDOFF §0) — conflates joint-training with check-in-level-vs-pooled-region input.
- The Tier-2 "irreducibly architectural / joint cross-attn harness caps reg" headline may be **substantially an input-artifact difference**, not joint-training dynamics. (The within-MTL dose-response, all on `next_region.parquet`, is UNAFFECTED — only the MTL-vs-(c)-ceiling ABSOLUTE gap is confounded.)

**Decisive confirmatory experiment IN FLIGHT (`t2p0_input_artifact.sh`):** the SAME `next_stan_flow α=0` head, SAME p1 harness/recipe, on BOTH input-types at AL+FL. `region` arm validates the harness (should reproduce (c) ~62.88/73.31); `checkin` arm = the matched comparand (same `next_region.parquet` the MTL eats). **Within-session paired delta = pure input effect.**
- **checkin << region (toward MTL reg ~53/60)** → the MTL-reg gap is the INPUT REPRESENTATION → reframes Tier 2/2P; the real lever = give MTL reg the pooled-region pathway (a NEW, un-falsified dual-tower variant: private tower on `region_embeddings.parquet`, not `next_region.parquet`).
- **checkin ≈ region (~62/73)** → check-in input is fine for STL reg → the MTL gap is genuinely architectural/joint after all (T2P.1 gates cleanly).

**This opens a concrete NEW lever the track never tested:** a dual-tower whose PRIVATE tower consumes the pooled `region_embeddings.parquet` sequence (the (c) input) fused with the shared check-in-level pathway — i.e. give the one MTL model access to BOTH representations. Potentially the gap-closer the track has been chasing.

**Chain status**: Tier 2P — the linchpin's "joint loop" reading is overturned pending the confirmatory paired run; the gap is (very likely) an input-artifact confound. Frozen (c)/(d) untouched (they remain valid STL ceilings on THEIR input; the issue is they were compared to MTL reg on a DIFFERENT input).

**Next**: input-artifact paired run lands → **STOP + surface the full reframing to the user** (decide: confirm + pursue the pooled-region private-tower lever / re-state the Tier-2 headline / etc.). Do NOT auto-launch. This supersedes the simple T2P.1-vs-T2P.2 gate.

---

## 2026-06-07 — Tiers 3–6 RE-SCOPED under G (user re-opened "Tier 3 is moot"; advisor-calibrated)

**Phase**: re-planning (no runs). User: "explore Tier 3/4/5/6 (CA/TX in 6) — those had promising points; add necessary points/tiers from future_works; use advising." I had called Tiers 3–6 blanket-MOOT last turn — that was too hasty.

**The correction (the organizing principle).** A breadth agent classified every Tier 3–6 card + all ~13 future_works memos {DONE / MOOT / LIVE-PROMISING / LIVE-COMPLETENESS}. Key insight (advisor-affirmed): the C25/G reversal did NOT moot the chain uniformly — it **RE-OPENED the cards whose prior negative/moot verdict depended on the now-removed regime** (class-weighted, shared-backbone) and left the regime-independent verdicts closed. Re-open test = "was the prior verdict regime-dependent?"

**LIVE set (post-G), advisor-calibrated to ~4 probes + 3 completeness (not 12 runs):**
- **R0 (FREE, FIRST)** multi-state matched-metric re-score — the "matches" verb is FL-only (B-A2); pin G−ceiling at AL/AZ/GE. Defines the bar every probe must move.
- **R1 ★** overlap-under-G — RE-OPENED: the dense-supervision MTL-negative was the *shared backbone* failing to absorb data; G's private STAN tower may now absorb it = a mechanism test of the dual-tower thesis. Engine+ceiling built at AL.
- **R2 ★** dual-substrate routing HGI→reg (FL pilot) — REFRAMED: C25 falsified its "substrate washes out" premise; HGI's +0.36pp STL reg edge ≈ G's matched shortfall; hook built, HGI region-emb on disk at FL.
- **R3** T4.0 loss-scale norm + RLW — untested, DISTINCT from the balancers (T2V.6) and the category-weight sweep; the ~4.7× CE-magnitude gap never ÷log(K). Ungated, near-free on G.
- **R4** HSM high-cardinality reg head (`next_stan_flow_hsm`) — never GPU-run (hierarchy build script exists); large-state-specific reg lever where the matches→beats headroom lives.
- **PARKED contingencies** (only if R1–R4 stall): T3.2 richer priors (recast as a private-tower feature — its additive-α/KD pathways are falsified), T4.2 Lion/Prodigy.
- **DONE/MOOT (stand):** T3.1 (KD identical to G), T4.1 (balancers swept), T5.1/5.2 (heads settled), composite/MoE/sampler/architecture-revisit (regime-independent verdicts).

**Two advisor corrections encoded (the C25-trap, again):** (1) **Magnitude** — a lever that lifts STL reg lifts the *ceiling* too, so every probe measures **G−ceiling on the MATCHED metric** (the matches→beats delta), NOT "a +5pp lever for a 0.35pp gap" (the agent's arithmetic over-promised R1/R2). (2) **Mechanistic framing** — the valuable output may be a mechanism finding (e.g. "the private tower absorbs data the shared regime wasted" = causal dual-tower evidence), not a new number; report the mechanism, pass or fail.

**Sequencing:** R0 (free, pins the bar) → R1+R2+R3 (cheap, gated on R0's matched bar) → R4 → contingencies only if stall. **Completeness is a PARALLEL track** (doesn't block probes): CA/TX (heavy v14 build; scale-conditional reviewer challenge) + paper-canon restatement (author).

**Docs:** INDEX new section "Tiers 3–6 RE-SCOPED" (`#tier36`) — the principle + the full DONE/MOOT/LIVE table + cards R0–R4 + the sequencing/parked callout; old Tier 3–6 cards banner-marked "pre-G" (kept for the trail). HANDOFF current-state block updated (⛔→♻). TOC updated.

**Chain status**: study re-opened post-G with the regime-dependence-filtered live set. G remains the champion baseline; all probes score Δ vs the R0 matched-metric bar.

**Next**: A40 — R0 (free re-score) first to pin the bar, then the R1/R2/R3 cheap probes (gated), then R4; CA/TX + paper-canon in parallel. Advisor + STOP-for-user at the boundary.

---

## 2026-06-07 — Re-scope folded IN PLACE into Tiers 3/4/5/6 (user: no parallel umbrella; Tier 4 = the optimization tier, expanded with src/losses)

**Phase**: structural re-org of the prior-turn re-scope (no runs). User: "keep Tier 4 a separate tier, apply my changes directly into it + EXPAND with the other optimizations in src/*; same for Tier 5/6 — alter the scope in place, not a new re-scope umbrella."

**What changed.** Removed the parallel "Tiers 3–6 RE-SCOPED" umbrella (with R0–R4 cards) and folded the live work into the REAL tier sections:
- **Tier 3 → "Reg-input pathway (prior · supervision · substrate)"**: houses R1 (overlap-under-G) + R2 (dual-substrate routing); T3.1 KD = done/moot; richer-priors = parked. (Old pre-G T3.1/T3.2 cards deleted; the principle + DONE/MOOT/LIVE table kept here as cross-tier context.)
- **Tier 4 → "Optimization (loss + optimizer)" — user-prioritised, EXPANDED with `src/losses`**: T4.0 loss-scale-norm + RLW (=R3); **T4.1 = the FULL registry sweep** — inventoried `src/losses/registry.py` (~20 losses); T2V.6 swept only 4 (famo/cagrad/nash/uw); the rest (db_mtl, dwa, gradnorm, aligned_mtl, uw_so, stch, scheduled_static, fairgrad, bayesagg_mtl, excess_mtl, equal_weight, random_weight=RLW) untested under G → the expansion; T4.2 optimizer (per-head LR + AdamW knobs; Lion/Prodigy noted as new-dep contingency since `helpers.py` is AdamW-only).
- **Tier 5 → heads (post-G)**: T5.1/T5.2 marked DONE (STAN load-bearing, next_gru+plain-CE — regime-independent, stand); added **T5.3 HSM** high-cardinality reg head (=R4) as the one live residual.
- **Tier 6 → ship+completeness**: added **T6.0/R0** (FREE matched-metric re-score, do-FIRST, gates the probes); T6.1 re-framed to CA/TX + multi-seed ship of G; T6.2 paper-canon.
- TOC, the DONE/MOOT/LIVE table cross-refs, and the HANDOFF block all updated to the in-place structure. INDEX tags balanced; no duplicate Tier 3.

**Carried framing (unchanged):** the regime-dependence re-open test; the C25 magnitude rule (score G−ceiling on the MATCHED metric); mechanistic framing; sequencing T6.0/R0 → R1+R2+T4.0/T4.1 → T5.3, CA/TX+paper-canon parallel.

**Chain status**: Tiers 3–6 re-scoped in place under G; G remains the champion baseline. No runs yet.

**Next**: A40 — T6.0/R0 (free, pins the bar) first; then the Tier 3 (R1/R2) + Tier 4 (T4.0 + the registry sweep) cheap probes; T5.3 HSM; CA/TX + paper-canon parallel.

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

## 2026-06-04 — CA 1-fold directional (user: cancel TX/CA 5-fold) → large-state recipe = no dominance

**Phase**: Tier 2 — large-state recipe confirmation (fast). 5-fold CA/TX was ~5h/run (impractical) →
user cancelled → 1-fold CA only (`--folds 1` → n_splits=2 single fold; leak-free 2-fold seeded log_T).

**CA (8501 regions, 1-fold directional — NOT comparable to 5-fold absolutes; read B9-vs-onecycle Δ only):**
- onecycle: reg 47.05 / cat 61.28 · B9: reg 49.60 / cat 59.32 → **B9 and onecycle TRADE** (B9 +2.5 reg,
  onecycle +2.0 cat); **neither dominates**.

**Large-state recipe verdict (FL 5-fold multi-seed + CA 1-fold):** onecycle does NOT dominate at scale
(FL: reg-tie, B9 wins cat +4.4; CA: B9 wins reg, onecycle wins cat) → **keep B9 at large states**. The
small-state onecycle dominance (AL/AZ multi-seed, +6-9 reg / +1-2 cat) is the clear, actionable win.

**Hygiene:** the 2-fold log_T build OVERWROTE CA fold1/2 (filenames don't encode n_splits) → **restored
CA 5-fold seed42 log_T** (rebuilt --n-splits 5; 5 files verified). CA back to clean state. TX left as-is
(graph regenerated, not run). Driver `t21_ca_1fold.sh`.

**Chain status**: Tier 2 — recipe finding complete (small dominate / large no-dominance); CA/TX 5-fold
not pursued (impractical wall-clock; directional 1-fold sufficed).

**Next**: surface complete Tier-2 close + onecycle-adoption decision to user.

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

## 2026-06-04 — TIER 2 FINAL: hardening (multi-seed) + v11 onecycle confirmation + adoption

**Phase**: Tier 2 close (user: "adopt + harden first" → "write close-out + HANDOFF"). All onecycle,
KD-OFF, seeded per-fold log_T, 5f×50ep.

**Hardening (harden2, multi-seed) — both verdicts hold:**
- **FL dual-tower negative HARDENED multi-seed** ({0,1,7}): dt_gated reg 59.03±0.15 vs base_a 62.38 =
  **Δ −3.35** (tighter + bigger than seed42's −2.89). The architecture negative is multi-seed-solid at FL.
- **CrossStitch partial is REAL but small** ({0,1,7}): Δreg **+1.07/+1.02** at AL/AZ (σ 0.08–0.33, OUTSIDE
  noise; seed42 "within σ" was too conservative), +0.37 FL; cat MIXED (AL −1.40 / AZ −0.11 / FL +0.59).
  → CrossStitch genuinely improves reg ~1pp (the ONLY arch that does) at a small AL-cat cost, still
  −5 to −10pp below the (c) ceiling → a real weak-partial, NOT a gap-closer. Verdict unchanged.

**v11 onecycle confirmation (paper substrate, AL/AZ {0,1,7,100}, diagnostic-best to match §0.1):**
| | onecyc reg | vs §0.1 B9 | arch-Δ reg | onecyc cat | vs §0.1 B9 | arch-Δ cat |
|---|---|---|---|---|---|---|
| AL | 53.15±0.44 | +2.98 | −8.06 (was −11.04) | 47.93±0.16 | +7.36 | +6.58 (was −0.78) |
| AZ | 41.54±0.23 | +0.76 | −11.52 (was −12.28) | 49.79±0.17 | +4.69 | +5.89 (was +1.20) |

- **The recipe finding TRANSFERS to v11** (onecycle > B9 on the paper substrate), but with NUANCE:
  - **reg gain over B9 is MODEST on v11** (AL +2.98, AZ +0.76 — smaller than v14's +5.5) → the reg
    arch-deficit shrinks only modestly (AL −11→−8, AZ −12.3→−11.5).
  - **cat gain over B9 is LARGE** (AL +7.36, AZ +4.69) — but this is mostly because **§0.1's table uses B9
    (alt-SGD tanks small-state cat), not the SHIPPED H3-alt.** vs H3-alt (cat ~46.78), onecycle cat (~47.9)
    is only +1pp. So the §0.1 "AL cat −0.78" was a **B9-recipe artifact**; the deployable cat is positive.

**ADOPTION DECISION (user-approved "adopt + harden first"):**
- **Adopt onecycle as the recommended small-state (AL/AZ) MTL recipe** (NORTH_STAR) — dominates H3-alt
  (v14 +6-9 reg / +1-2 cat, multi-seed) and beats B9 (v11 paper substrate, modest reg + large cat).
- **Keep B9 at large states** (FL/CA: onecycle does not dominate; B9 wins cat at scale).
- **§0.1 re-statement (DELICATE — touches the BRACIS architectural-Δ headline):** documented as an
  ANNOTATION (provenance-preserving, not overwriting B9 numbers) — the implied onecycle small-state
  arch-Δ + the nuance (modest reg shrink; cat-flip entangled with §0.1's B9-vs-H3-alt choice). Flagged
  for user review before it enters the submission proper. Do NOT silently rewrite §0.1.

**Chain status**: Tier 2 COMPLETE — architecture NEGATIVE (multi-seed hardened) + onecycle recipe WIN
(adopted small-state). Frozen (c)/(d) untouched; freeze-sanity GREEN.

**Next**: write PAPER_UPDATE.md (Tier-2 close-out) + rewrite HANDOFF.md (Tier 2 complete); commit + push.

---

## 2026-06-05 — STRETCH CLOSED (T2.3 MoE + T2.4 SwiGLU, both NULL on reg) + #19 §0.1 continuity A/B → fix lifts §0.1 FL +3.15reg/+3.52cat

**Phase**: Tier-2 optional stretch (user: "close the three remaining gaps + execute the optional stretch") — COMPLETE. All FL, multi-seed {0,1,7,100}, UNWEIGHTED real-joint (C25 default), onecycle KD-OFF, seeded per-fold log_T (freshness verified vs next_region.parquet). Disjoint reg = `per_metric_best.next_region.top10_acc_indist`; cat = `diagnostic_task_best.next_category.f1`. Aggregator: `scripts/mtl_improvement/c25_stretch_agg.py`.

**What happened**
- **Gaps #17/#18 closed** (committed earlier): CANONICAL_VERSIONS v15 pin (C25 unweighted recipe + reproduction map `--reg/cat-class-weights`); Acc@1→Acc@10 reg checkpoint monitor (`PrimaryMetric.TOP10` + reg `primary_metric` → TOP10; selector-independent diagnostic/disjoint reads unaffected).
- **T2.3 MoE** built+ran (`mtlnet_mmoe`, `mtlnet_cgc` added to `c25_tier2_refix.sh`).
- **T2.4 SwiGLU** built (`mtlnet_crossattn_swiglu`: subclass of mtlnet_crossattn; pre-norm + SwiGLU FFN, parallel bidirectional attn, hidden=round(2/3·ffn_dim) for param parity; only `_build_shared_backbone` overridden). Unit gate `t24_swiglu_unit_gate.py` GREEN — partition bijective+exhaustive, shared-backbone capacity Δ=0.086% vs baseline, structure (3-proj SwiGLU hidden=171) confirmed.
- **#19 FL-B9 §0.1 continuity** ran (canon GCN, B9 recipe = cosine + alt-opt + α-no-wd + min-best-5, UNWEIGHTED) + a **same-harness WEIGHTED A/B** (`c25_fl_b9_weighted.sh`, `--reg/cat-class-weights`) to control for harness drift vs the published §0.1.

**Findings**
- **Full FL stretch board (reg@10 disj / cat-F1 diag, n=4 seeds):**
  | arm | reg | cat | Δreg vs base_a | Δreg vs ceiling(73.31) | intervention |
  |---|---|---|---|---|---|
  | dual_gated (T2.1) | **73.06**±0.08 | 72.03 | +1.51 | **−0.25** | private reg pathway |
  | prior_off | 72.94±0.07 | 72.02 | +1.39 | −0.37 | α·log_T prior OFF |
  | crossstitch | 71.97±0.12 | 72.08 | +0.42 | −1.34 | soft feature-share |
  | cgc (T2.3) | 71.77±0.06 | 71.58 | +0.22 | −1.55 | MoE expert capacity |
  | swiglu (T2.4) | 71.71±0.14 | **72.13**±0.03 | +0.16 | −1.60 | modern backbone |
  | mmoe (T2.3) | 71.68±0.12 | 71.57 | +0.13 | −1.63 | MoE expert capacity |
  | hardshare | 71.48±0.08 | 71.62 | −0.07 | −1.83 | hard-share anchor |
- **#19 §0.1 continuity — SAME-HARNESS A/B (B9-canon, only the class-weight flag differs):**
  | | reg@10 | cat-F1 |
  |---|---|---|
  | WEIGHTED (pre-C25 repro) | 63.91±0.16 | 70.34±0.06 |
  | UNWEIGHTED (C25 fix) | 67.06±0.08 | 73.86±0.06 |
  | **Δ (fix)** | **+3.15** | **+3.52** |
  - Harness drift quantified: same-harness weighted (63.91/70.34) vs published §0.1 (63.27/68.56) = +0.64 reg / +1.78 cat → the cross-harness delta (+3.79/+5.30) was inflated by drift; the **clean fix effect is +3.15/+3.52**. (Cross-check: weighted cat 70.34 == `v14_mtl_vs_canonical.md` canon FL cat 70.34 → harness consistent.)
  - **§0.1 architectural Δ_reg** (vs §0.1 STL ceiling 70.62): weighted −6.71 → unweighted **−3.56** (reg gap ~halved). **Δ_cat**: +3.18 → **+6.70** (cat advantage ~doubled).

**Decision**
- **Stretch verdict LOCKED — architecture capacity is NOT the bottleneck.** TWO independent "more/better architecture" interventions (MoE expert capacity; SwiGLU modern backbone) are NULL on reg (+0.13..+0.22, within σ). Only the TWO mechanism interventions move it: prior-OFF (+1.39) and the private un-mixed reg pathway / dual-tower (+1.51). The residual MTL→STL reg gap is the **α·log_T prior + shared-pathway dilution, not architecture or backbone quality.** dual_gated (73.06, −0.25 from ceiling) remains FL best.
- **SwiGLU's only real effect is cat:** 72.13 = best cat of any arm (+0.24 vs base_a) — a small modern-backbone cat bump, no reg movement.
- **MulT-faithful + crossstitch→crossattn hybrids = NOT BUILT (dropped):** they'd be a 3rd architecture-capacity test of a twice-falsified hypothesis. Model class + gate for SwiGLU are committed and reusable if ever needed.
- **#19 result is paper-relevant:** the C25 fix lifts the EXACT §0.1 paper-canon FL MTL row by +3.15 reg / +3.52 cat and roughly halves the central architectural reg gap. → RESULTS_TABLE §0.1 gets a continuity ANNOTATION (NOT a table rewrite — author sign-off, same convention as the onecycle annotation).

**Chain status**: Tier-2 stretch COMPLETE. All three gaps (#17/#18/#19) closed. Frozen (c)/(d) untouched.

**Next**: consolidate docs (HANDOFF stretch-done, RESULTS_TABLE §0.1 continuity annotation, CONCERNS C25 cross-ref); commit + push. Study is at a clean stopping point — no open GPU work. CA/TX deferred per user (large-state compute).

---

## 2026-06-06 — ⭐⭐⭐ CEILING BROKEN: a single MTL model (dual aux + prior-OFF) BEATS both STL ceilings + matches the composite (FL 4-seed)

**Phase**: Tier-2 combo screen (user: "execute the speculative hybrids + other combinations; screen at 1 seed, promote promising to 4 seeds; the dual_gated gates are promising, try the shared component as crossstitch/swiglu/etc.; use an advisor"). COMPLETE — and it produced the study's strongest result. All FL, unweighted onecycle KD-OFF, disjoint reg `per_metric_best.top10_acc_indist` / cat `diagnostic_task_best.f1`.

**Advisor first (user-requested).** Code-grounded architecture advisor (general-purpose subagent) reframed the user's hypothesis decisively: **the only reg-moving levers are the private reg tower + the α·log_T prior — both FLAG-controllable on the existing dual-tower with ZERO new code.** Swapping the shared backbone (crossstitch/swiglu/MulT) underneath a dual-tower is a CAT play, not a reg play — the dual-tower gate (inits ≈0.73 toward private) suppresses the shared pathway, so SwiGLU/MoE-style upgrades can't move reg. Advisor's standout call was a combo the user did NOT name: **`aux` fusion** (`priv + β·shared`, β init 0.1) ADDS the shared pathway WITHOUT diluting the private tower (unlike `gated`, which makes them compete) → the best mechanistic fit to the "dilution" finding. Advisor DROPPED dual+crossstitch (mechanism fights the dual-tower premise, HIGH effort, null) and MulT/crossstitch→crossattn (shared-capacity lever, twice-falsified).

**What happened.** Built combo (F) `mtlnet_crossattn_dualtower_swiglu` (diamond-avoiding subclass of dualtower, overrides only `_build_shared_backbone`; gate `t24_dualtower_swiglu_gate.py` GREEN). 1-seed screen (seed 0) of 4 arms, then promoted the 2 ceiling-breakers to 4 seeds {0,1,7,100}. Drivers: `c25_combos_{screen,promote}.sh`.

**Findings**
- **1-seed screen (FL, seed 0):** (G) dual aux+prior-OFF **73.56/73.12 ★**; (H) dual private_only+prior-OFF **73.43/72.19 ★**; (A) dual gated+prior-OFF 73.00/72.09 (≈ dual_gated; prior-OFF redundant with the gated prior-ON); (F) dual+SwiGLU+prior-OFF 72.87/72.09 (SwiGLU NULL on reg — advisor's prediction confirmed: the shared-backbone swap is not a reg lever).
- **4-seed confirmation {0,1,7,100} (the headline):**
  | arm | reg@10 | cat-F1 | vs (c) STL ceiling | vs (d) composite | vs dual_gated |
  |---|---|---|---|---|---|
  | **(G) dual aux + prior-OFF** | **73.57±0.06** | **73.16±0.04** | reg **+0.26** / cat **+3.19** | reg −0.05 (TIE) | reg +0.51 / cat +1.13 |
  | (H) dual private_only + prior-OFF | 73.42±0.03 | 72.17±0.07 | reg +0.11 / cat +2.20 | reg −0.20 | reg +0.36 / cat +0.14 |
  - (G) per-seed reg = [73.56, 73.55, 73.51, 73.67] — EVERY seed clears the (c) ceiling 73.31; σ=0.06.
  - Anchors: (c) STL reg ceiling 73.31 / cat 69.97; (d) composite reg 73.62 (max v14/HGI α0); dual_gated(prior-ON) 73.06/72.03; base_a 71.55/71.89.

**Decision**
- **(G) `dual aux + prior-OFF` is the new FL MTL champion.** A SINGLE joint MTL model that (1) **beats the (c) STL reg ceiling** (+0.26), (2) **beats the (c) STL cat ceiling** (+3.19), (3) **matches the (d) two-model composite reg** (−0.05, within σ) while also winning cat. Config: `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` (`raw_embed_dim=64 fusion_mode=aux freeze_alpha=True alpha_init=0.0`), v14 substrate, unweighted onecycle KD-OFF.
- **The MTL tradeoff is not just dissolved — it is INVERTED.** OLD: "MTL sacrifices reg; ship the 2-model composite." NEW (post-C25 + this combo): **a single joint model is Pareto-OPTIMAL — it beats both single-task ceilings AND matches the composite.** The composite is now strictly dominated (same reg, worse cat, 2× the params/inference).
- **Mechanism, fully resolved:** the residual gap was (a) the α·log_T prior (biased logit term — remove it) + (b) shared-pathway DILUTION of the private reg representation. `aux` fusion fixes (b) by adding shared as a non-attenuating residual; prior-OFF fixes (a). Neither is architecture capacity (MoE, SwiGLU, gated-competition all null/worse).
- **(H) private_only confirms** the private tower alone (no shared pathway) already clears the ceiling — i.e. the reg signal is carried entirely by the un-diluted private STAN; the shared pathway's only positive contribution is via `aux` (additive, small) and to cat.

**Caveat (recorded):** the (c)/(d) ceilings are seed=42 single-seed; (G) is 4-seed. The margin is robust (all (G) seeds ≥73.51 > 73.31) but a fully seed-matched claim would re-run (c)/(d) multi-seed. FL only — AL/AZ/GE/CA/TX not yet run for (G) (per user, large states deferred).

**Chain status**: Combo screen COMPLETE; (G) promoted + confirmed. Frozen (c)/(d) untouched.

**Next**: promote (G) into PAPER_UPDATE.md (headline table → Pareto-positive), HANDOFF top block, NORTH_STAR.md (FL-validated champion, flagged 1-state pending AL/AZ/GE multi-state confirm). Optional future: (G) at AL/AZ/GE (small states, cheap) to make it multi-state; re-run (c)/(d) multi-seed to seed-match the ceiling claim.

---

## 2026-06-06 — G GENERALIZES: beats both ceilings at AL/AZ/GE too (1-seed); speculative hybrids NULL; G sweep finds no Pareto gain

**Phase**: G follow-up screens (user: "run the dropped speculative hybrids; run G at the other states 1 seed; advisor on G's sweep space"). COMPLETE. Driver `c25_gv2.sh` (13 arms, seed 0, FL+AL/AZ/GE, unweighted onecycle KD-OFF). 2nd advisor (G sweep space) ran first — see its analysis in the previous session; key call: reg headroom is THIN (G ties the composite), highest-EV is multi-state validation not micro-sweeps, fp32 is the one reg lever with a documented sign, aux→gated/sequence-level fusion would REGRESS the ceiling-break.

**Findings (seed-0 screen, reg disjoint / cat diag):**
- **(I) Speculative hybrids @ FL — NULL/NEGATIVE on reg (advisor confirmed):** MulT-faithful (`mtlnet_crossattn_mult`) reg 71.28 / cat 71.92; crossstitch→crossattn (`mtlnet_crossattn_xstitch`) reg 71.13 / cat 72.02. **Both BELOW base_a (71.55)** and far below dual_gated/ceiling. Adding intra-stream self-attention (MulT) or composing cross-stitch+cross-attn = the shared-pathway capacity lever, now falsified a 4th/5th independent way (after MoE, SwiGLU). Clean negatives; not pursued further.
- **(II) ⭐ G MULTI-STATE — beats BOTH (c) ceilings at every other available state (1-seed):**
  | state | g reg | (c) reg | Δreg | g cat | (c) cat | Δcat |
  |---|---|---|---|---|---|---|
  | AL | 64.61 | 62.88 | **+1.73** | 52.75 | 49.97 | **+2.78** |
  | AZ | 55.69 | 55.11 | **+0.58** | 54.77 | 51.01 | **+3.76** |
  | GE | 59.34 | 58.45 | **+0.89** | 61.63 | 58.12 | **+3.51** |
  G's "single MTL model beats both STL ceilings" now holds at **4/4 available states** (FL 4-seed + AL/AZ/GE 1-seed). CA/TX have no v14 substrate (can't run without building it). **This is the high-value result — the Pareto-positive headline generalizes.**
- **(III) G sweep @ FL — NO Pareto improvement over G (73.57/73.16); G is well-tuned:**
  - `g_fp32` (MTL_DISABLE_AMP=1): reg **73.70 (+0.13)** but cat **72.05 (−1.11)** — the advisor-predicted fp16→fp32 reg lift IS real (documented sign at 4703-class) but it's a **reg/cat TRADE, not a win** (geom 72.87 < G's 73.36 → G stays champion). Worth noting as a precision-sensitivity finding.
  - `g_catw0.50` reg 73.59 (+0.02, noise) / cat 72.19 (−0.97); `g_catw0.65` 73.46/72.48; `g_catw0.85` 73.07/72.90 — category-weight trades reg↔cat monotonically; 0.75 (G) is a good joint point.
  - `g_pdrop0.1` reg 72.52 (−1.05, worse) ; `g_pdrop0.2` 73.52 (−0.05 ≈ G) — priv_dropout 0.3 (G default) confirmed near-optimal; lowering hurts.
  - `g_kd0.1`/`g_kd0.2`: reg 73.56 / cat 73.12 — IDENTICAL to G. Soft log_T-KD adds nothing on the dual-tower (KD was tuned on the old single-pathway arch; doesn't transfer). Confirms prior-OFF is the right call; no KD re-entry needed.

**Decision**
- **G's config is locked — no sweep beat it.** category-weight 0.75, priv_dropout 0.3, aux fusion, prior-OFF (KD off), AMP-on are all at/near their joint optimum. The sweep validated G rather than improving it (the advisor's thin-headroom prediction held exactly).
- **Speculative hybrids = closed NEGATIVE.** MulT + crossstitch→crossattn join MoE + SwiGLU as falsified shared-pathway-capacity interventions. The architecture-capacity hypothesis is now falsified 5 independent ways.
- **G multi-state is the promotion candidate** (beats ceiling at AL/AZ/GE 1-seed) → propose 4-seed {1,7,100} confirmation at AL/AZ/GE to make the Pareto-positive claim multi-state paper-grade (ASKED user before launching).

**Chain status**: G follow-ups COMPLETE. Frozen (c)/(d) untouched.

**Next**: (pending user OK) 4-seed AL/AZ/GE confirmation of G; optional fp32 + (c)/(d) multi-seed seed-match. Then paper-doc restatement (CH25/CH28/§0.1 → Pareto-positive, multi-state).

---

## 2026-06-06 — ✅ G CONFIRMED MULTI-STATE @ 4 SEEDS: beats BOTH STL ceilings at ALL 4 available states (paper-grade)

**Phase**: G multi-state confirmation (user: "execute" the 4-seed AL/AZ/GE run). COMPLETE. Driver `c25_g_multistate.sh` — G (dualtower + aux + prior-OFF) at AL/AZ/GE × seeds {1,7,100} (seed 0 reused from `c25_gv2`), v14, unweighted onecycle KD-OFF. Aggregated with the FL 4-seed.

**Findings — G 4-seed {0,1,7,100} mean reg / cat vs the (c) STL ceilings (reg disjoint / cat diag):**
| state | G reg | (c) reg | Δreg | G cat | (c) cat | Δcat | verdict |
|---|---|---|---|---|---|---|---|
| AL | 64.47±0.11 | 62.88 | **+1.59** | 52.91±0.27 | 49.97 | **+2.94** | ★ beats BOTH |
| AZ | 55.75±0.21 | 55.11 | **+0.64** | 54.48±0.74 | 51.01 | **+3.47** | ★ beats BOTH |
| GE | 59.37±0.04 | 58.45 | **+0.92** | 61.43±0.26 | 58.12 | **+3.31** | ★ beats BOTH |
| FL | 73.57±0.06 | 73.31 | **+0.26** | 73.16±0.04 | 69.97 | **+3.19** | ★ beats BOTH |

**Decision**
- **The Pareto-positive headline is now MULTI-STATE, MULTI-SEED paper-grade.** A single jointly-trained MTL model (G = `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` aux+prior-OFF, v14, unweighted onecycle) **beats BOTH single-task STL ceilings (reg AND cat) at all 4 available states**, 4 seeds each, tight σ (reg σ 0.04–0.21). The 1-seed AL/AZ/GE screen held under multi-seed (deltas essentially unchanged: AL +1.73→+1.59, AZ +0.58→+0.64, GE +0.89→+0.92).
- **The MTL tradeoff is INVERTED at every state**, not just FL. OLD framing ("MTL sacrifices reg; ship the 2-model composite; the −7..−17pp arch-Δ is the paper's central tension") is fully dissolved AND replaced by a strictly stronger positive: joint training Pareto-dominates single-task at all measured states.
- **CA/TX** have no v14 substrate built → would need ~the v14 embedding build first; deferred (large-state compute, per user).

**Chain status**: G multi-state CONFIRMED. Frozen (c)/(d) untouched. No open GPU work.

**Next**: ONLY the BRACIS paper-doc restatement remains (CH25/CH28/§0.1 → Pareto-positive multi-state) — an author decision. Optional (not blocking): re-run (c)/(d) ceilings multi-seed to seed-match (currently seed=42; G margins are robust to it); G at CA/TX after a v14 build.

---

## 2026-06-06 — TIER 2V OPENED: critique close-out begun (Tier 2/2P closed; T2V.3 param; T2V.1 launched)

**Phase**: executing `CRITIQUE_TIER2_C25_2026-06-06.md` §7 (user: "close the points raised + update Tier 2/2P in INDEX + everything P0–P3"). This entry = the first batch.

**What happened**
- **Tier 2 + Tier 2P CLOSED in INDEX** (doc, no runs): Tier-2 banner UNDER RE-VALIDATION→RESOLVED post-C25; filled T2.3 (-lite MoE NULL) + T2.4 (SwiGLU/MulT/xstitch NULL/neg) Results blocks with the landed stretch data; Tier-2 final-decision → POSITIVE (topology axis: private reg tower + aux is the win); **Tier 2P → MOOT** (the "joint loop poisons reg" hypothesis WAS the C25 confound; T2P.0 wd refuted; T2P.1/.2/.3 superseded). Flagged the pre-C25 alt-arch dose-response as UNDER-POWERED (confounded + un-swept loss-weight) pending the T2V.4 fair re-rank.
- **T2V.3 param count DONE** (the critique's "½ params" correction): G = 6,241,536 params = base_a (5,951,224) **+4.9%** (+290,312, of which the reg-private STAN tower is 273,800). G adds ONE small private tower, does NOT duplicate the backbone → "strictly dominates 2 models / ½ params" RE-FRAMED to "one model at ≈1.05× base_a." Checkpoint re-eval + T2V.2 full-metric/tail still pending (G ran `--no-checkpoints` → need one checkpointed G re-run; G's summary already has `accuracy_macro` + `_indist`).
- **T2V.1 LAUNCHED** (the #1 gate — multi-seed (c)/(d) ceilings {0,1,7,100} × AL/AZ/GE/FL; `t2v1_ceilings_multiseed.sh`): (c) reg = next_stan_flow α=0 (p1, v14); (c) cat = next_gru + logit-adjust τ=0.5; (d) reg = HGI-α0 at FL only (HGI region-emb on disk only at FL — AL/AZ/GE HGI margin ≤0.7pp, substrate absent → (d) stays seed-42 there, flagged). ~36 STL runs, CONC=3. Caught + fixed a bash single-line-`local` `set -u` gotcha (`local a=$1 b=$2 key="${b}…"` expands `${b}` from the OUTER scope; split the declaration).

**Decision**: G is a validated positive — Tier 2V VALIDATES/EXTENDS, does not re-litigate (per critique §7). Everything (P0+P1–P3) is in scope (user). Dependency: T2V.4–9 score against the T2V.1 multi-seed ceilings → T2V.1 lands first.

**Chain status**: Tier 2V in flight. Frozen seed-42 (c)/(d) untouched (T2V.1 ADDS multi-seed columns, does not overwrite — the immutable-yardstick rule).

**Next**: aggregate T2V.1 → re-state G's Δ vs the multi-seed ceilings (fill INDEX T2V.1) → checkpointed-G re-run for T2V.2/T2V.3-reeval → then P1 (T2V.4 alt-arch fair re-rank, the one dominant test) → P2/P3.

---

## 2026-06-06 — ✅ T2V.1 LANDED: the reg-"beats-ceiling" headline HOLDS seed-matched (the critique's #1 gate)

**Phase**: Tier 2V P0, the #1 paper-blocking gate (CRITIQUE §3 G1). Multi-seed (c)/(d) STL ceilings {0,1,7,100} × AL/AZ/GE/FL, 36 STL runs, 0 FAIL. Driver `t2v1_ceilings_multiseed.sh`.

**Findings — multi-seed (c) ceilings vs G (seed-matched):**
| state | (c)reg ms | (c)reg s42 | G reg | Δ | (c)cat ms | G cat | Δ | verdict |
|---|---|---|---|---|---|---|---|---|
| AL | 62.67±0.13 | 62.88 | 64.47 | **+1.80** | 50.35±0.69 | 52.91 | **+2.56** | ★ both |
| AZ | 54.80±0.22 | 55.11 | 55.75 | **+0.95** | 50.39±0.13 | 54.48 | **+4.09** | ★ both |
| GE | 58.44±0.06 | 58.45 | 59.37 | **+0.93** | 57.50±0.12 | 61.43 | **+3.93** | ★ both |
| FL | 73.27±0.06 | 73.31 | 73.57 | **+0.30** | 69.96±0.07 | 73.16 | **+3.20** | ★ both |
(d) composite reg FL multi-seed = 73.49±0.03 (HGI-α0); G FL 73.57 → **+0.08 ahead** (was a seed-42 tie).

**Decision**
- **The reg "beats both ceilings" headline HOLDS, seed-matched, at 4/4 states — paper-grade.** The critique's central worry (the seed-42 ceiling was a lucky draw; a multi-seed ceiling could swing several pp at small states and flip the margins) is **resolved**: the STL ceilings are remarkably stable across seeds (σ ≤ 0.7 incl. small states — the "3–4pp fold-σ" is within-run fold noise that the per-seed mean averages out). Seed-matched (c) ≈ seed-42 (c) everywhere; G's margins barely move.
- **Conservative read intact:** the p1 ceiling harness is fp32, G runs fp16 (precision offset DISADVANTAGES G's reg per T2V.2's fp32-G +0.13) → G beating the fp32 ceiling on fp16 is a lower bound.
- **(d)-HGI** recomputed at FL only (substrate on disk only at FL); AL/AZ/GE (d) stays seed-42 (HGI margin ≤0.7pp, not load-bearing). The "G ties/beats the composite" framing is now seed-matched at FL.
- **The critique's recommended downgrade ("matches" not "beats") is NOT triggered** — G ≥ (c) reg at 4/4 (not just ≥3/4). The cat gain (+2.5..+4.1pp) was never in doubt and is conservative (G's cat is plain unweighted vs the ceiling's logit-adjust τ=0.5).

**Chain status**: T2V.1 CLOSED (the #1 gate passes). Frozen seed-42 (c)/(d) untouched — T2V.1 ADDED multi-seed columns (immutable-yardstick rule).

**Next**: (1) one **checkpointed-G re-run at FL** (G ran `--no-checkpoints`) → unblocks **T2V.2** (full top10_acc + popularity-bins + macro vs prior-ON, the tail-regression check) + the **T2V.3 independent checkpoint re-eval**; (2) then **P1 T2V.4** — the alt-arch fair re-rank (per-arch category-weight, post-C25) — the critique's one dominant test.

---

## 2026-06-06 — T2V.2/3 CLOSED: prior-OFF has NO tail cost; G reproduces; artifact-foreclosed

**Phase**: Tier 2V P0 (CRITIQUE §3 G2 + §3). FL seed-0; G (prior-OFF) vs prior-ON (aux fusion held constant). Driver `t2v23_priorcheck.sh`.

**Findings (G prior-OFF vs prior-ON, FL seed0):** reg top10_indist −0.01, **accuracy_macro (tail) −0.04**, top1_indist −0.01, mrr_indist +0.01, cat −0.09 — ALL within 1-seed noise. **T2V.2 verdict: NO tail regression** — prior-OFF is a free choice. The critique's worry (G stacks unweighted-CE + α=0, both head-favouring, could trade against the tail) is dissolved: under the dual-tower the α·log_T prior is VESTIGIAL (learnable α→≈0 post-C25), so prior-OFF ≈ prior-ON on head AND macro/tail. (The +1.39 prior-OFF gain in the stretch was on the SHARED reg head base_a→prior_off; under the dual-tower the prior is already ≈dead.)

**T2V.3:** g_ckpt reg 73.56 == G seed-0 (73.56) → REPRODUCES exactly. Param count DONE (G=base_a+4.9%). Artifact-foreclosure: T2V.1's INDEPENDENT p1 harness produces the ceilings G beats (cross-harness) + disjoint metric selector-independent → not a single-harness artifact. (No .pt saved to the rundir → literal load-and-forward skipped; cross-harness evidence supersedes.)

**Decision**: P0 gates (T2V.1/2/3) all PASS — the reg headline is seed-matched (beats 4/4), tail-clean, reproduced, artifact-foreclosed, param-honest. The critique's "fragile reg" concerns are CLOSED; the cat gain was always robust.

**Chain status**: T2V.2/3 CLOSED. T2V.4 (alt-arch fair re-rank) running.

**Next**: aggregate T2V.4 (the one dominant test) when it lands → then P1 remainder (T2V.5 reg-head sweep, T2V.6 optimizer/FAMO) + P2 T2V.7 (logit-adjust cat).

---

## 2026-06-06 — T2V.4 CLOSED: alt-arch fair re-rank → G HOLDS (the falsification is now un-confounded, paper-safe)

**Phase**: Tier 2V P1, the critique's ONE dominant test (§6.4–6.6). Standalone alt-archs re-ranked POST-C25, each at its own category-weight {0.5,0.65,0.75}, FL seed0. Driver `t2v4_altarch_rerank.sh` (12 runs; a CONC=3→2 OOM-recovery on the heavy MoE arms, 0 net FAIL).

**Findings — best reg@10 per arch (its own cat-weight optimum) vs G 73.57 / (c) ceiling 73.31:**
| arch (standalone) | best reg @cw | cat | Δ vs G | Δ vs (c) |
|---|---|---|---|---|
| crossstitch | 71.94 @0.65 | 72.14 | −1.63 | −1.37 |
| mmoe (-lite) | 71.69 @0.65 | 71.7 | −1.88 | −1.62 |
| cgc (-lite) | 71.69 @0.50 | 71.6 | −1.88 | −1.62 |
| hardshare | 71.45 @0.50 | 71.6 | −2.12 | −1.86 |

**Decision**
- **G HOLDS — no fairly-tuned standalone alt-arch ties/beats it.** All land at ~base_a level (71.4–71.9), none reaches even the (c) ceiling, category-weight barely moves reg (reg is architecture-bound for these archs). This RESOLVES the critique's §6.4 worry that the "architecture-capacity falsified" claim was under-powered/confounded: the alt-archs were run STANDALONE (not gate-suppressed shared-pathway swaps), un-confounded (post-C25), and per-arch-tuned — and still lose by 1.6–2.1pp. **"The private reg tower, not architecture capacity, is the reg lever" is now paper-safe.**
- No -lite surprised → the faithful learned-gate CGC build is NOT triggered.
- crossstitch is the best alt (its prior weak-partial status holds) but −1.63 vs G.
- Sub-sweep (a) backbone-UNDER-the-dual-tower (a cat/aux lever) deferred to T2V.5/T2V.8 — the standalone result already settles the architecture claim. CRITIQUE §6.4 annotated RESOLVED.

**Chain status**: T2V.4 CLOSED (P1 dominant test). All P0 + the dominant P1 test now pass.

**Next**: T2V.5 (reg-private HEAD sweep + d_model/aux-β + cat-private-tower ablation), T2V.6 (optimizer/FAMO + per-task LR/precision); then P2 T2V.7 (logit-adjust on G's cat — highest-EV cat lever) + T2V.8 (combine) + P3 T2V.9 (CA/TX).

---

## 2026-06-07 — T2V.5/6/7 CLOSED: NO hypertuning lever beats G → T2V.8 moot, G is robustly optimal

**Phase**: Tier 2V P1/P2 winner-hypertuning (CRITIQUE §4.1/§6.2/§6.5). FL seed0, 5 flag-arms. Driver `t2v567_hypertune.sh`. (New code: MTL cat criterion now honours `--logit-adjust-tau` via leak-free `build_calibrated_loss`, commit 72c77f54.)

**Findings (vs G reg 73.57 / cat 73.16):**
- **T2V.7 logit-adjust on cat — NEGATIVE.** la τ=0.5 cat 72.29 (−0.87); τ=1.0 cat 70.45 (−2.71); reg unaffected. Logit-adjust LIFTED the STL cat ceiling (+2.7) but HURTS the MTL cat — opposite regime. **Resolves the "unweighted cat wins is unexplained" puzzle (§4.4):** in joint training the cat head is already above its STL ceiling (rising tide) and sits in a different regime; logit-adjust over-corrects. Plain unweighted CE is genuinely the MTL cat optimum.
- **T2V.5 reg-tower internals — G's STAN is RIGHT-SIZED.** priv d_model=256 reg 72.70 (−0.87, overfit/hurts); priv heads=8 reg 73.55 (−0.02, tie). No headroom either way → the §6.2 over-provisioning question answers "no"; the STAN→other-head swap + cat-private-tower code-builds are NOT motivated (S.1 STAN-is-reg-winner stands).
- **T2V.6 FAMO — ≈ G.** reg 73.44 (−0.13) / cat 73.40 (+0.24), both noise. static_weight confirmed fine; the optimizer/balancer is not a confound on G (§6.5 resolved). FAMO ran clean (no per-head-LR conflict).

**Decision**
- **NO hypertuning lever beats G on any axis (cat-loss, reg-tower-size, optimizer).** G is robustly optimal. **→ T2V.8 (combine the winning levers) is MOOT** — there are no winners to stack. The T2V.5 head-swap + cat-private-tower builds are unmotivated (no tower headroom). Document them as tested-negative / not-motivated rather than running them.
- The only remaining substantive card is **T2V.9 (CA/TX)** — the scale-conditional completeness gap (needs a v14 design_k substrate build).

**Chain status**: T2V.4/5/6/7 CLOSED; T2V.8 MOOT. P0 + P1 + P2 all done. Only P3 (T2V.9 CA/TX) remains.

**Next**: T2V.9 — build v14 (`check2hgi_design_k_resln_mae_l0_1`) at CA/TX, then champion G + the (c)/(d) ceilings (the heavy serial-spine build). Then the Tier-2V close-out summary + CRITIQUE §7 checklist.

---

## 2026-06-07 — ✅ TIER 2V CLOSED + STUDY CLOSE-OUT: critique fully addressed; T2V.9 deferred (user)

**Phase**: Tier 2V close-out. Every `CRITIQUE_TIER2_C25_2026-06-06` concern resolved; T2V.9 (CA/TX) deferred to future-work by user decision; study at a clean close.

**Tier 2V scorecard (all cards):**
- **P0** — T2V.1 ✅ reg headline HOLDS seed-matched (4/4; multi-seed ceilings stable σ≤0.7; the "downgrade to matches" was NOT triggered) · T2V.2 ✅ no tail regression (prior-OFF ≈ prior-ON on macro) · T2V.3 ✅ reproduced + artifact-foreclosed (independent p1 harness) + "½ params" → +4.9%.
- **P1** — T2V.4 ✅ alt-arch FAIR re-rank (standalone, post-C25, per-arch cat-weight) → G holds, all alts −1.6..−2.1pp → falsification un-confounded/paper-safe · T2V.5 ✅ private STAN right-sized (d_model=256 hurts, heads=8 ties) → head-swap/cat-private builds NOT motivated · T2V.6 ✅ FAMO ≈ G → static_weight fine, optimizer not a confound.
- **P2** — T2V.7 ✅ logit-adjust HURTS MTL cat (plain CE is the optimum; resolves the §4.4 "unweighted cat wins" puzzle; cat-calibration code committed + works) · T2V.8 MOOT (no winning lever to combine).
- **P3** — T2V.9 (CA/TX) DEFERRED to documented future-work (`docs/future_works/mtl_improvement_catx_scale_conditional.md`) — the 4-state result is already paper-grade; CA/TX is the lone scale-conditional completeness extension + the single most expensive step (design_k build + champion + ceilings on CA 8501 / TX 6553 regions). Prediction recorded: the C25 benefit scales with class count → CA/TX should show the LARGEST margins.

**Decision / final state**
- **G is the validated, robust, paper-safe champion.** `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` (aux, prior-OFF), v14, unweighted onecycle KD-OFF. Beats both single-task STL ceilings (reg AND cat) at all 4 available states, 4-seed; the MTL tradeoff is INVERTED. The critique's "fragile reg" worries are all closed; the cat gain was always robust + conservative (G's plain CE vs the ceiling's logit-adjust).
- **The ONLY remaining work is the BRACIS paper-doc restatement** (CH25/CH28/§0.1 → Pareto-positive multi-state) — an author decision. No open experimental work.

**Chain status**: Tier 2V CLOSED. Study at a clean stopping point. Frozen seed-42 (c)/(d) untouched (T2V.1 added multi-seed columns).

**Next**: author-side BRACIS restatement. Optional future: T2V.9 CA/TX (future-work memo).

---

## 2026-06-07 — ⚠⭐ CRITIQUE §8 RESIDUALS CLOSED: B-A2 tempers the reg headline to "MATCHES" (metric-mismatch caught); B-A1/B-A4/optimizers all confirm G

**Phase**: Tier 2V §8 residual queue (the items the close-out substituted/deferred). FL seed0. Drivers `t2v_dropped_flagarms.sh` (focal/cb + uncert/cagrad/nash), `t2v5_headswap.sh` (priv gru/lstm/tcn + reg-head-lr), `t2v_ba2_snapshot.sh` (checkpointed G + route_task_best).

**⚠ B-A2 (independent checkpoint re-eval) — THE consequential one.** route_task_best re-scored G's saved reg_best snapshots (a DIFFERENT code path from mtl_cv): reg Acc@10 (FULL top10_acc) folds [73.77,71.97,73.38,72.54,73.00], **mean 72.93**. This (a) FORECLOSES harness inflation — G's reg reproduces ~73 via an independent path; the −0.63 vs G's reported 73.56 is the **mechanical OOD penalty** (ood_fraction 0.83%; 73.56×(1−0.0083)≈72.95), not inflation — AND (b) **caught a metric-mismatch in the "beats" headline**: p1 computes ONLY full `top10_acc` (no `_indist`), so the (c) ceiling 73.31 is FULL, while G's reported 73.56 is `top10_acc_indist`. **On a MATCHED metric G is ~0.35pp BELOW the ceiling** (G-full 72.93 vs ceiling-full 73.31 = −0.38; or G-indist 73.56 vs ceiling-indist ≈73.9 = −0.34). **→ the "+0.26 beats reg ceiling" was G-indist-vs-ceiling-full; on matched metrics G MATCHES (Pareto-non-inferior, within ~0.4pp), NOT beats.** Exactly the §8 A-2 recommendation — the critique's insistence on the re-eval was vindicated. The cat +3pp beat is UNAFFECTED (cat F1 is a single metric, no indist/full split). **CLAIM CORRECTION: reg "matches the STL ceiling"; cat "beats by +3pp". The Pareto-positive / inverted-tradeoff story STANDS (matches reg + beats cat dissolves the −7..−17pp tension) — only the reg verb tempers from "beats" to "matches".**

**⭐ B-A1 (lighter private tower — USER PRIORITY) — STAN is LOAD-BEARING (tested, not assumed).** vs G(stan, 273,800 priv params, reg 73.57): gru (478,592, heavier) 71.17 (−2.40); lstm (627,072) 70.15 (−3.42); **tcn (90,752, 3× lighter) 71.77 (−1.80)**. NONE matches STAN — heavier RNNs AND the 3× lighter TCN all lose 1.8–3.4pp. So the §6.2 "is the full STAN over-provisioned?" question is answered NO: the STAN's pairwise spatiotemporal-attention inductive bias is genuinely the right private-tower architecture (not just capacity). "The full STAN is load-bearing, not over-provisioned" is now a TESTED claim. §6.2 CLOSED.

**B-A4-cat (cat loss family) — plain CE is the MTL cat optimum.** focal γ=2 cat 71.44 (−1.72); class-balanced 73.04 (−0.12); (logit-adjust −0.87/−2.71 from T2V.7). The ENTIRE cat-loss family (logit-adjust/focal/CB) loses or ties → plain unweighted CE is genuinely the MTL cat optimum (the §4.4 "unweighted cat wins" puzzle fully resolved across the family). reg unaffected (cat-only lever).

**T2V.6 optimizers (completing FAMO) — static_weight confirmed across the family.** uncertainty-weighting reg 73.55/cat 72.14; CAGrad 73.67/71.84; Nash 73.65/71.61 (no cvxpy error); reg-head-lr 1e-4 73.56/73.12. All ≈ G on reg or trade reg(+0.1, noise)↔cat(−1.3..−1.5); NONE Pareto-beats G. With FAMO (≈G), the whole optimizer/balancer axis confirms static_weight — the user's §6.5 confound concern fully resolved for G.

**Decision**
- **Reg claim TEMPERED to "matches" (B-A2 metric-mismatch).** Propagate to the claim docs (CLAIMS CH30, NORTH_STAR, CANONICAL_VERSIONS §v16, CHAMPION.md, PAPER_UPDATE.md, HANDOFF): reg = "MATCHES the STL ceiling (Pareto-non-inferior, within ~0.4pp matched-metric)"; cat = "BEATS by +3pp (conservative loss)". The inverted-tradeoff headline stands.
- **B-A1/B-A4/optimizers all confirm G is robust + well-tuned** — no lever beats it on any axis; the STAN tower is load-bearing.
- **Lone remaining residual: B-A3 (cat-private-tower ablation, narrative-only, predicted null) — needs a cat dual-tower build.** B-A4-reg popularity-binning is moot (the full metric 72.93 + macro proxy already show no differential tail effect).

**Chain status**: §8 residuals CLOSED except B-A3 (narrative). The reg headline correction is the key output. Driver bug fixed (t2v_ba2 `pid=$!` foreground).

**Next**: propagate the "matches not beats" reg tempering to all claim docs; then B-A3 (cat-private) if wanted.

---

## 2026-06-07 — B-A3 SURPRISE: cat-private tower HELPS cat +1.58 (a potential champion improvement) → all §8 residuals closed

**Phase**: last CRITIQUE §8 residual (B-A3 cat-private-tower ablation, narrative). Built `mtlnet_crossattn_dualtower_catpriv` (gives the CAT head a private tower on raw checkin, aux fusion — symmetric to reg). FL seed0.

**Finding (⚠ surprise):** cat-private cat **74.74 (+1.58 vs G 73.16)**, reg 73.52 (−0.05 ≈ G). **The predicted NULL is WRONG** — the cat head ALSO benefits from a private raw-checkin tower, at no reg cost. Because `aux` fusion is additive (priv + β·shared), cat keeps the shared rising tide AND gains private signal the shared cross-attn (which mixes cat+reg) doesn't fully exploit.

**Decision**
- **Asymmetry narrative REFINED:** not "reg needs private / cat wants shared" but "**BOTH tasks benefit from private+shared**; reg's private tower is load-bearing (carries reg), cat's is additive (lifts cat +1.58 at no reg cost)." The clean-asymmetry story is too strong; state the nuanced version.
- **Potential NEW champion (promote-on-surprise):** a both-private dual-tower (G + cat-private) Pareto-improves G — cat +1.58, reg flat — and would widen the cat-ceiling beat to +4.77 (74.74 vs 69.97). **1-seed; +1.58 is well outside the ~0.1–0.3 cat σ → promote to multi-seed {0,1,7,100} to confirm** (proposed to user; NOT auto-launched). The critique's test-don't-assume discipline caught a real improvement where a null was predicted — exactly why the residuals were worth finishing.
- **ALL CRITIQUE §8 residuals now closed:** B-A1 (STAN load-bearing), B-A2 (reg "beats"→"matches", harness-clean), B-A3 (cat-private SURPRISE +1.58), B-A4 (plain CE optimal) + T2V.6 optimizers (static_weight confirmed). Only the BRACIS paper-doc restatement remains (author) — PLUS the optional both-private multi-seed promotion.

**Chain status**: §8 residuals CLOSED. B-A3 opened a new (optional) champion-improvement thread.

**Next**: (pending user) multi-seed confirm of the both-private dual-tower (cat-private champion). Then paper-doc restatement.

---

## 2026-06-07 — ⭐ G&prime; CONFIRMED: both-private dual-tower Pareto-improves G (cat +1.61, reg flat) — a champion upgrade from the B-A3 surprise

**Phase**: B-A3 multi-seed promotion. The seed-0 cat-private surprise (+1.58) was tested at {0,1,7,100} FL. Driver `ba3_catpriv_multiseed.sh`.

**Finding — the lift HOLDS rock-solid:** G' (both-private dual-tower) reg **73.59±0.07** / cat **74.77±0.04** (per-seed cat 74.71/74.81/74.80/74.74) vs G reg 73.57 / cat 73.16. **Δcat +1.61 (σ 0.04), Δreg +0.03 (flat).** The cat-ceiling beat WIDENS from +3.19 to **+4.80** (74.77 vs 69.97).

**Decision**
- **G' = `mtlnet_crossattn_dualtower_catpriv` (cat-head + reg-head both `next_stan_flow_dualtower`, aux + prior-OFF) is the new FL champion.** Adding a private tower to the CAT head (private STAN on raw checkin, additive aux) lifts cat +1.61 at ZERO reg cost → a clean Pareto improvement over G. The B-A3 "predicted null" was a genuine improvement — vindicating the finish-the-residuals discipline.
- **reg framing UNCHANGED — "matches" (B-A2).** G''s reg path is identical to G; reg 73.59 ≈ G 73.57; the +0.28 indist vs the (c) ceiling is the same indist-vs-full artifact → reg matches (Pareto-non-inferior). The UPGRADE is on cat.
- **Scope: FL-only 4-seed.** G stays the multi-state-confirmed result (AL/AZ/GE/FL); G' is the FL-confirmed cat upgrade. Multi-state confirm of G' (AL/AZ/GE) is the natural next step (proposed to user, NOT auto-launched).
- Propagate G' to the champion docs (CHAMPION/NORTH_STAR/CANONICAL_VERSIONS/CLAIMS/HANDOFF) as the cat-improved variant; keep reg "matches".

**Chain status**: B-A3 promotion CLOSED — G' confirmed. All critique residuals done.

**Next**: (pending user) multi-state confirm of G' at AL/AZ/GE. Then the BRACIS paper-doc restatement (now: matches reg + beats cat by ~+4.8pp with G').

---

## 2026-06-07 — ⚠ G prime DEMOTED: cat-private tower is FL-ONLY, CRATERS small-state cat → G remains the multi-state champion

**Phase**: G' multi-state confirmation (user-requested). G' (both-private dual-tower) at AL/AZ/GE × {0,1,7,100}. Driver `gprime_multistate.sh`. 12 runs, 0 FAIL.

**Finding (⚠ the FL lift does NOT generalize):** G' cat vs G cat — AL **37.66 (−15.25!)**, AZ **42.02 (−12.45)**, GE 57.84 (−3.59), FL 74.77 (+1.61). reg flat at every state (+0.02..+0.11). **The cat-private STAN tower OVERFITS at small-state data scale and CRATERS cat 3.6–15.3pp at AL/AZ/GE** (AL/AZ fall BELOW the STL cat ceiling). It only helps at FL's large data scale.

**Decision**
- **G' is DEMOTED — it is NOT a multi-state champion.** The cat-private tower is a scale-conditional FL-only effect that is *catastrophic* at small states. **G (cat-shared) remains THE champion** — cat beats the ceiling at all 4 states, reg matches.
- **MECHANISM CORRECTED 2026-06-07 (advisor root-cause — see CHAMPION.md §G′):** the small-state collapse is **UNDERFIT, not overfit** (the earlier "overfits smaller AL/AZ data" was WRONG). Evidence = the train-F1 trajectory: AL G′ cat **train**-F1 caps at **0.45** (vs the `next_gru` head's **0.98**), tiny train–val gap → textbook underfit. The off-label `next_stan_flow_dualtower` cat head (built for ~1000s of region classes) is over-regularized (`priv_dropout=0.3`) on a GRU-tuned LR schedule → never converges at small data; FL is the only state with enough data to train the heavy private tower (the +1.61 is robust, σ 0.04pp, but scale-gated). No wiring bug (α·log_T prior correctly OFF; input plumbing correct). → A **rescue screen** (`gprime_rescue_screen.sh`, 1-seed AL+FL: lower `priv_dropout`, softer cat-lr, smaller tower) is testing recoverability — if it lifts AL to ≥ G while keeping FL's gain, G′ is re-promotable; else the head is fundamentally FL-only.
- **I over-promoted G' on the FL-only 4-seed result** (propagated "new champion" to CHAMPION/NORTH_STAR/CANONICAL_VERSIONS/CLAIMS/HANDOFF). **REVERT that** — restate G' as an FL-only experimental variant (cat +1.61 at FL only; craters small states; not shippable). The B-A3 finding stands as a *mechanism* note (cat CAN benefit from a private tower at large data scale) but is NOT a champion change.
- **The multi-state test (user's call) caught the FL-only trap** — exactly the don't-ship-an-FL-only-result discipline (AGENT_PROMPT rule 13 / the moving-baseline guard). A 1-state win that reverses at scale.
- reg robustness reconfirmed: the reg path (G's dual-tower) is flat across all states under the cat-tower change → reg is stable.

**Chain status**: G' demoted to FL-only variant; G is the champion. All critique residuals remain closed.

**Next**: revert the G' champion propagation in the 5 claim docs → G' = FL-only experimental note; G stays champion. Then the BRACIS restatement (G: matches reg + beats cat at 4 states).

---

## 2026-06-07 — ⛔ G′ RESCUE SCREEN CLOSED: no lever recovers small-state cat; the FL gain needs the very config that craters small states → G′ is irreducibly large-state-only

**What.** `gprime_rescue_screen.sh` — 1-seed (s0) AL+FL, 6 arms testing the advisor's ranked hypertuning levers on the catpriv cat head: L0 base (priv_dropout=0.3, cat-lr=1e-3, d=128) · L1 priv_dropout=0.1 · L2 cat-lr=3e-4 · L3 d_model=64+heads=2 · L12 drop+lr · L13 drop+small. Question: does any lever lift **AL cat to ≥ G (52.91) WITHOUT losing FL's +1.58**? Metric: `per_metric_best.next_category.f1.mean` (diagnostic-best, per-task) + reg `top10_acc_indist`.

| arm | AL cat (Δ vs G) | FL cat (Δ vs G) | note |
|---|---|---|---|
| L0_base (drop 0.3) | 37.49 (−15.42) | **74.74 (+1.58)** | FL gain present; AL crater |
| L1 drop 0.1 | 37.38 (−15.53) | 73.17 (+0.01) | dropout↓ **kills FL gain**, AL flat |
| L2 cat-lr 3e-4 | 37.49 (−15.42) | 74.74 (+1.58) | LR inert both states |
| L3 small tower | **38.44 (−14.47)** | 74.40 (+1.24) | best AL (+0.95 vs L0) but still −14.5; FL gain partly eroded |
| L12 drop+lr | 37.38 (−15.53) | 73.17 (+0.01) | = L1 |
| L13 drop+small | 38.01 (−14.90) | 73.75 (+0.59) | both eroded |

reg flat everywhere (AL ~64.5 / FL ~73.5) — the reg path is unchanged → robust.

**Verdict — NO RESCUE. G′ is irreducibly large-state-only.**
- **AL:** every lever stays **−14.5 to −15.5 pp below G**; the best (smaller tower, L3) closes <1 pp of a 15 pp hole. Lowering dropout (L1) and softening LR (L2) do **nothing** at AL.
- **FL:** the +1.58 gain survives **only at the original `priv_dropout=0.3`**; dropping it to 0.1 **erases the gain** (74.74→73.17). Smaller tower (L3) partly erodes it too (74.40).
- **The tension is irreducible:** the heavy-regularized large private tower that *produces* the FL gain is exactly what *cannot* learn small-state 7-class cat; and the only lever that nudges AL up (shrink the tower) simultaneously eats the FL gain. There is no setting that wins both.

**Refined mechanism (sharper than the "underfit" note above):** it is NOT merely over-regularization — L1 (drop 0.1) didn't help AL at all, so reducing the regularizer doesn't unlock fit. The `next_stan_flow_dualtower` flow/attention head is **architecturally mismatched** for a 7-class target at small-state data scale; the smaller-tower marginal gain (+0.95) is the only directional signal and it's nowhere near sufficient. "Underfit" is correct as a symptom; the *cause* is head↔task-cardinality mismatch, not a tunable hyperparameter.

**Decision.** G′ stays **DEMOTED — an FL-only experimental variant, NOT a champion.** **G (cat-SHARED `next_gru`) is THE multi-state champion.** No further G′ work is motivated (the lever space is exhausted at 1-seed; a multi-seed confirm would only sharpen an already-decisive −14.5pp gap). The B-A3/G′ line is **CLOSED**.

**Chain status**: closed cleanly — G is champion; G′ catpriv is a documented dead-end (FL-only, head↔7-class mismatch). All critique residuals + the rescue follow-up are now closed.

**Next**: nothing on G′. Open items unchanged: (author) BRACIS paper-doc restatement (G: matches reg + beats cat, 4 states); (optional, deferred) build the CA v14 substrate to test G at a second large state.

---

## 2026-06-08 — ✅ R0 / T6.0 LANDED: matched-metric G−ceiling bar PINNED multi-state (FREE, gates Tier 3)

**Phase**: Tier 6 R0 — the FREE prerequisite that gates every reopened probe (R1/R2/T4/T5.3). New session; user scope: **work on Tier 3.** Per the documented global order, R0 (the matched-metric bar) runs FIRST because R1/R2's gate is "does it move G−ceiling on the matched metric" and that bar was FL-only (B-A2).

**What happened**
- The reg "matches the ceiling" verb was matched-metric-verified at **FL only** (B-A2): G's reported reg was `top10_acc_indist` while the (c) p1 ceiling is the FULL `top10_acc`. R0 re-scores the EXISTING G runs onto the FULL metric at AL/AZ/GE/FL — **zero retraining**.
- **Method** (`scripts/mtl_improvement/r0_matched_rescore.py`): `full = indist·(1−ood_fraction)` per fold at the indist-best epoch (= G's full-best epoch, since `ood_fraction` is epoch-invariant per fold → argmax full = argmax indist). OOD targets are always wrong in the full metric, so the relation is exact. **Validated** against B-A2's independent `route_task_best` code path at FL: my CSV conversion gives 72.95 vs B-A2's 72.93 (per-fold [73.79,71.97,73.41,72.55,73.03] vs [73.77,71.97,73.38,72.54,73.00]).
- Ceilings re-extracted from the same T2V.1 p1 JSONs / ccat runs → **reconcile EXACTLY** with the T2V.1-reported multi-seed ceilings (cat 50.35/50.39/57.50/69.96; reg 62.67/54.80/58.44/73.27; (d) 73.49). G-indist cross-check reproduces the reported headline exactly (64.47/55.75/59.37/73.57) → the harness + rundir map are sound.

**Findings — the matched-metric bar (mean ± std over seeds {0,1,7,100}):**
| state | G-full reg | (c) reg ceil (full) | Δreg matched | G cat F1 | (c) cat | Δcat |
|---|---|---|---|---|---|---|
| AL | 62.57±0.10 | 62.67±0.13 | **−0.09** | 52.91±0.27 | 50.35 | **+2.56** |
| AZ | 54.68±0.24 | 54.80±0.22 | **−0.12** | 54.48±0.74 | 50.39 | **+4.08** |
| GE | 58.35±0.04 | 58.44±0.06 | **−0.09** | 61.43±0.26 | 57.50 | **+3.93** |
| FL | 72.97±0.06 | 73.27±0.06 | **−0.31** | 73.16±0.04 | 69.96 | **+3.20** |

FL (d) composite (HGI-α0) full = 73.49 → G-full − composite = **−0.53** (the seed-42 "+0.08 ahead the composite" was the indist-vs-full artifact).

**Decision**
- **The reg "matches" verb (B-A2) GENERALIZES multi-state.** On the matched FULL metric, G is Pareto-non-inferior on reg at all 4 states (Δreg −0.09 to −0.31 pp, within fold σ ~0.04–0.24); cat beats +2.6 to +4.1 pp everywhere (exact, unaffected by the mismatch). The inverted-tradeoff / Pareto-positive headline STANDS verbatim (matches reg + beats cat) — now multi-state matched-metric-verified, not FL-only.
- **The reg matched-gap any Tier-3/4/5 probe must move is TINY and consistent (~0.1–0.3 pp), largest at FL (−0.31) and AZ (−0.12), ~nil at AL/GE (−0.09).** This is the magnitude-rule reality: there is no +5pp opportunity; a probe that lifts STL reg lifts the ceiling too. FL/AZ are where probes have the most (still small) room.

**Chain status**: R0 CLOSED. The bar is pinned; R1/R2 (Tier 3) + T4/T5.3 are now measurable. Frozen (c)/(d) untouched (re-score only). Committed.

**Next**: R1 (overlap-under-G at AL — does G's private tower absorb the dense-supervision lift the shared backbone wasted?) → R2 (HGI→reg routing at FL). Both score Δ vs this R0 bar + report mechanism.

---

## 2026-06-08 — ✅ R1 LANDED: overlap-under-G is a MECHANISM WIN — the private tower absorbs dense supervision (dual-tower thesis confirmed)

**Phase**: Tier 3 R1 (the top-candidate reg-pathway probe). AL seed=42 clean paired 2×2 in the current harness. Driver `scripts/mtl_improvement/r1_overlap_under_g.sh`; results `docs/results/mtl_improvement/R1_overlap_under_g.md`.

**What happened**
- The overlap study (2026-06-03) found dense (stride-1) supervision lifts **STL reg +5.13** but, under the **OLD regime** (class-weighted CE + the SHARED `next_stan` reg backbone), **MTL reg lifted only +0.50 → the gap WIDENED ~+4.6pp** (the shared backbone couldn't absorb the data). R1 re-runs the SAME windowing contrast under **champion G** (dual-tower private STAN reg pathway, post-C25 unweighted, aux + prior-OFF) to test the dual-tower thesis.
- Ran 4 cells at AL seed42: {non-overlap v14, overlap dk_ovl} × {STL reg ceiling (p1 α=0), champion G}. Reg scored matched-metric (R0 method: G-full = indist·(1−ood); ceiling = p1 full top10_acc). log_T is inert (G prior-OFF + KD-off); the v14 seed-42 log_T satisfies the trainer's existence/freshness guards (point `--per-fold-transition-dir` at the v14 dir while `--engine check2hgi_dk_ovl` drives the overlap data).

**Findings (AL seed42):**
| windowing | STL reg ceil | G reg-full | ΔG−ceil reg | G cat F1 |
|---|---|---|---|---|
| non-overlap | 62.88 ± 4.05 | 62.77 ± 4.0 | **−0.11** | 52.79 |
| overlap | 68.01 ± 4.22 | 67.67 ± 3.6 | **−0.34** | 61.18 |

- STL reg ceiling lift = **+5.12** (reproduces prior +5.13 exactly). **G reg-full lift = +4.89 ≈ the STL lift.**
- ΔG−ceil reg shift = **−0.23pp** (−0.11 → −0.34), within fold σ (~4). cat rising-tide +8.39 (STL +9.77).
- Cross-checks: non-overlap reg ceiling 62.88 = frozen §2 seed42; non-overlap ΔG−ceil −0.11 = R0 multi-seed −0.09 → seed-42 cell sound.

**Decision**
- **MECHANISM WIN — the dual-tower thesis is CONFIRMED.** The identical windowing intervention that the OLD shared backbone could not use (MTL reg +0.50 vs STL +5.13, gap widened) is now **fully absorbed by G's private STAN reg tower** (MTL reg +4.89 ≈ STL +5.12, gap unchanged). G's MTL reg pathway behaves like an STL pathway under data scaling — exactly the dual-tower design intent, and **direct causal evidence for the central architectural claim, independent of the C25 loss confound.**
- **G−ceiling is NULL (rising tide):** overlap lifts G and the ceiling ~equally, so the matched bar barely moves (−0.23pp, within σ). Per the magnitude rule this is NOT a champion-improving number — the value is the **mechanism**, exactly as the R1 card anticipated ("paper-grade even if small").
- Non-overlap canon unchanged (the 2026-06-03 user decision stands); overlap stays documented future-work, now ALSO carrying this dual-tower mechanism evidence.

**Chain status**: R1 CLOSED (mechanism win; matched bar null). Frozen (c)/(d) untouched. Committed. R2 next.

**Next**: R2 — dual-substrate routing HGI→reg at FL (does HGI's +0.36pp STL reg edge survive the joint dynamics under G?). Then the Tier-3 boundary review (advisor + summary + STOP for user).

---

## 2026-06-08 — ✅ R2 LANDED: HGI→reg routing — substrate edge TRANSFERS under G (premise falsified) but rising-tide → NULL on matched bar

**Phase**: Tier 3 R2 (dual-substrate routing pilot). FL multi-seed {0,1,7,100}. Driver `scripts/mtl_improvement/r2_dual_substrate_routing.sh`; results `docs/results/mtl_improvement/R2_dual_substrate_routing.md`.

**What happened**
- Routed HGI region-emb into champion G's PRIVATE reg tower (cat task_a + shared-aux stay v14) via the `REGION_EMB_ENGINE=hgi` env-var hook (`src/data/folds.py:980`) — inference-time swap, no rebuild (HGI region_embeddings.parquet on disk at FL, 64-dim = dual-tower raw_embed_dim). Verified the routing is real (HGI vs v14 region-emb mean |Δ|=0.30; routed seq tensor differs). log_T inert (G prior-OFF + KD-off; v14 seeded log_T satisfies guards). The memo was shelved on the "substrate washes out in MTL" premise that C25 falsified — R2 tests whether HGI's +0.36pp STL reg edge survives the joint dynamics under G.

**Findings (FL, 4 seeds, matched-metric):**
| arm | reg-full | cat F1 | Δ vs G-v14 | Δ vs (d) HGI ceil |
|---|---|---|---|---|
| G-HGI-routed | 73.22 ± 0.08 | 73.05 ± 0.15 | **+0.25** reg / −0.11 cat | **−0.27** |
| G-v14 (R0 control) | 72.97 ± 0.06 | 73.16 ± 0.04 | — | (vs (c) v14: −0.30) |

Per-seed G-HGI reg-full [73.14, 73.20, 73.36, 73.18]. (d) HGI STL ceiling 73.49; (c) v14 ceiling 73.27.

**Decision**
- **POSITIVE mechanism — the substrate edge TRANSFERS under G.** G-HGI beats G-v14 by **+0.25pp reg** (outside σ 0.08), capturing ~⅔ of HGI's +0.36pp STL edge. This **refutes the "substrate washes out in MTL" premise** that shelved the part-2 routing memo — exactly the C25 reframe (substrate transfers to MTL post-C25). Clean confirmation.
- **NULL on the deployable bar — rising tide.** HGI lifts BOTH G (+0.25) and its own STL ceiling (+0.22) equally → the matched G−ceiling gap is unchanged: −0.27 (HGI) ≈ −0.30 (v14, R0). Per the gate ("promote iff routing moves G−ceiling >0") this is NULL → v14 is sufficient under G; the +0.25 absolute reg gain costs a 2nd inference substrate (HGI alongside v14) — not justified for +0.25pp at flat cat.
- **Same shape as R1**: post-C25, both the data-scale lever (R1 overlap) AND the substrate lever (R2 HGI) transfer into G's private reg tower (they did NOT under the old shared backbone), but neither beats the achievable ceiling because the ceiling moves with them. The mechanism (transfer) is the value; the matched bar (deployability vs ceiling) holds.

**Chain status**: R2 CLOSED (substrate transfers / matched bar null). Frozen (c)/(d) untouched. Committed. Tier-3 live reg-pathway probes (R1 + R2) both done.

**Next**: Tier-3 boundary review — advisor pass on R0/R1/R2 → tier summary → STOP for user (decide Tier 4 optimization probes vs other). No autopilot.

---

## 2026-06-08 — Tier-3 BOUNDARY REVIEW: advisor pass + R1b de-confound (R1 corrected) + tier summary — STOP for user

**Phase**: Tier-3 boundary cadence (workflow §4): advisor pass → R1b correction → summary → STOP. No autopilot into Tier 4.

### Advisor pass (adversarial sub-agent on R0/R1/R2)
Verdicts: **R0 SOUND** (one wording fix: the indist→full conversion is validated-at-FL/by-analogy at AL/AZ/GE, not "exact"; error is in G's favour — applied). **R1 OVER-READ** (the "dual-tower absorbs dense supervision" attribution confounds C25-unweighting with the tower; the +0.50→+4.89 swing changed BOTH the loss and the backbone — needs the unweighted+shared cell). **R2 SOUND** (null on matched bar correct; comparing to the (d) HGI ceiling is the right framing; substrate-transfer is a standalone falsification worth keeping). Leak/confound check: **clean** (log_T truly inert under prior-OFF+KD-off; R2 routing verified to swap only the reg-tower region-emb). Top recommendation: run the de-confound cell or soften R1.

### R1b — de-confound (ran the cell; `r1b_shared_overlap_deconfound.sh`)
SHARED backbone (mtlnet_crossattn + next_stan_flow prior-OFF = G minus the private tower), C25-unweighted onecycle, AL seed42, overlap vs non-overlap:
| overlap reg lift | regime | lift |
|---|---|---|
| OLD overlap study | class-weighted + shared | +0.50 |
| **R1b** | **C25-unweighted + shared** | **+4.32** |
| R1 (G) | C25-unweighted + private tower | +4.89 |

**The shared backbone, once unweighted, ALSO absorbs the dense lift (+4.32 ≈ tower +4.89, within σ).** → **The absorber is C25 unweighting, NOT the dual-tower.** R1's mechanism claim CORRECTED: the old "overlap gap-widens" was the class-weighting confound; R1 is further evidence for C25 as the dominant reg lever, not dual-tower-specific evidence. The matched-bar null (rising tide) is unaffected. (This is the advisor pass doing its job — caught an over-read before it reached the paper.)

### ⭐ TIER 3 SUMMARY (R0 + R1 + R2) — reg-pathway levers TRANSFER post-C25 but the matched bar holds

**What ran:** R0 (free matched-metric re-score, AL/AZ/GE/FL), R1 (overlap-under-G, AL) + R1b (de-confound), R2 (HGI→reg routing, FL multi-seed). ~5 GPU-h total.

**The three results, one story:**
1. **R0 — the bar is pinned.** G **matches** the reg ceiling (Δ matched −0.09/−0.12/−0.09/−0.31 at AL/AZ/GE/FL, Pareto-non-inferior) and **beats** cat (+2.6..+4.1) multi-state. The reg matched-headroom any probe must move is **tiny + consistent (~0.1–0.3pp)**, largest at FL/AZ. Reconciles exactly with the T2V.1 ceilings.
2. **R1 — overlap (data-scale lever): rising tide, null on the matched bar.** Dense supervision lifts G (+4.89) and the ceiling (+5.12) equally → matched bar unchanged. R1b: the absorption is driven by **C25**, not the tower (shared +4.32 ≈ private +4.89).
3. **R2 — HGI (substrate lever): substrate edge TRANSFERS (premise falsified), rising tide, null on the matched bar.** HGI's +0.36pp STL edge survives under G (G-HGI +0.25 over G-v14, refuting "substrate washes out in MTL") but lifts G and its own ceiling equally → matched gap unchanged (−0.27 ≈ −0.30); +0.25 not worth a 2nd inference substrate → not promoted.

**Unified conclusion:** post-C25, BOTH reg-pathway levers (more data, better substrate) **transfer into the MTL reg pathway** (they did NOT under the old class-weighted regime), confirming the C25 reframe — but **neither beats the achievable ceiling**, because any lever that lifts MTL reg lifts its STL ceiling too (the magnitude rule, now empirically demonstrated twice). **G remains the champion; no Tier-3 probe changes it.** The valuable outputs are (a) the pinned multi-state matched bar (R0), (b) two clean falsifications of the "substrate/data washes out in MTL" family (R2 + R1's C25 attribution), (c) the demonstrated mechanism that the matched gap is structurally hard to move from the reg-INPUT side.

**Implication for Tier 4 (the user's call):** the reg-input side is exhausted (R1/R2). The remaining lever class is the **loss/optimization axis** (Tier 4: loss-scale-norm T4.0, the full balancer registry T4.1). Per the advisor: balancers are the ONE place a lever could genuinely move G−ceiling, because most have **no STL analogue** → the ceiling is FIXED for them (unlike R1/R2). Tier 4 must (i) score Δ vs the R0 bar, (ii) run the param-partition unit-test on G's dual-tower per gradient-surgery balancer (private tower ∈ `reg_specific`, not `shared` — a mis-partition silently corrupts gradient-surgery results), (iii) watch the cvxpy/ECOS NashMTL collapse. T4.1 driver scaffolding already committed (0a8f1753).

**Chain status**: Tier 3 CLOSED (R0/R1/R1b/R2 done; R1 corrected). Frozen (c)/(d) untouched. Champion G unchanged. Committed + pushed.

**STOP — surfacing to the user (tier-boundary cadence).** Decision needed: proceed to Tier 4 (T4.0 loss-scale + T4.1 balancer registry under G), or T5.3 (HSM high-card reg head), or CA/TX completeness, or close the study. No autopilot.

**Next** (pending user): if Tier 4 → T4.0 first (cheap, highest-EV, ungated) then T4.1 (per-method-tuned registry, arch-wired to G).

---

## 2026-06-08 — TIER 3 CLOSED (advisor close/continue decision) → proceeding to Tier 4

**Phase**: Tier-3 close/continue decision (user-requested 2nd advisor: "eval our finds + decide close vs other promising Tier-3 parts; if close, finalize docs then follow to Tier 4").

**Advisor verdict: CLOSE Tier 3.** Reasoning (full text in the session): (1) R1/R2 did NOT stall — they *resolved* (decisive rising-tide nulls + mechanism), so the T3-richer "run only if R1/R2 stall" trigger does not fire. (2) T3-richer-as-private-tower-input-feature is predictably a THIRD rising tide: a richer prior fed as an input feature still has an STL analogue (feed the STL ceiling the same feature → ceiling moves with it), same magnitude-rule trap; its other pathways (additive-α, KD) are already falsified under G. (3) No remaining reg-pathway lever is both non-rising-tide and unfalsified. (4) The Tier-3 summary's forward claim is CORRECT: the loss/optimization axis (Tier 4) is the one lever class with NO STL analogue → the STL ceiling is FIXED for balancers/scale-norm (unlike R1/R2/T5.3, which move the ceiling with them) → Tier 4 is the only place a lever can structurally move G−ceiling, strictly higher-EV than T5.3 (rising-tide reg-head lever) or CA/TX (deferrable completeness).

**Decision (applied):** Tier 3 CLOSED — reg-input axis exhausted; champion G unchanged. T3-richer marked NOT-RUN (trigger didn't fire). Docs finalized: INDEX `#tier36` (CLOSED banner + T3-richer not-run block), HANDOFF top (Tier-3-closed status line), this entry. Value captured for the paper: the pinned multi-state matched bar (R0, hardens the "matches" reg verb beyond FL-only) + two clean falsifications of the "washes out in MTL" family (R1's C25 attribution, R2's substrate transfer).

**Chain status**: Tier 3 CLOSED. Champion G unchanged. → Tier 4.

**Next**: Tier 4 — **T4.0 first** (loss-scale normalization: divide each CE by log(num_classes), re-tune w; + RLW litmus — cheap, ungated, highest-EV; RLW tells us early if the inter-task weight even matters). Then T4.1 (full src/losses registry, per-method-tuned, arch-wired to G's dual-tower partition; param-partition unit-test per gradient-surgery balancer; cvxpy/ECOS Nash-collapse watch). Score Δ vs the R0 matched bar.

---

## 2026-06-08 — Tier 4 START: T4.0b RLW litmus → inter-task weight is NOT the bottleneck (T4.1 low-EV)

**Phase**: Tier 4 (loss/optimization), first move. T4.0b = the cheap ungated RLW pre-check (the gate that decides whether the L-cost T4.1 balancer sweep is worth running).

**What happened**
- Ran champion G with `--mtl-loss random_weight` (RLW, Lin TMLR'22 — Dirichlet per-step random task weights) vs G (static_weight cw0.75), AL+FL seed0, matched-metric. Driver `scripts/mtl_improvement/t40_rlw_litmus.sh` (needed `--canon none` + explicit G flags + `--no-reg/cat-class-weights`, since the v16 canon auto-injects `--category-weight` which conflicts with random_weight).

**Findings (seed0, matched-metric):**
| state | RLW reg-full | G reg-full | Δreg | RLW cat | G cat | Δcat |
|---|---|---|---|---|---|---|
| AL | 62.31 | 62.64 | −0.33 | 54.25 | 52.75 | +1.49 |
| FL | 73.02 | 72.95 | +0.07 | 71.94 | 73.12 | −1.18 |

**Decision**
- **RLW ≈ G → the inter-task weight is NOT the bottleneck.** Random per-step weighting (spanning the whole simplex over training) matches tuned static_weight on reg (±0.33pp) and merely trades on cat (zero-sum). Canonical RLW litmus signal. → **The full T4.1 balancer registry (~16 methods, all manipulating inter-task weighting / gradient combination) is predictably LOW-EV** under G — it will not beat static_weight. Consistent with P4 ("balancing low-EV"), T2V.6 (famo/cagrad/nash/uw ≈ G at defaults), and now the weight axis directly shown flat. Neither RLW nor the matched bar moves; G sits at the reg ceiling regardless of inter-task weighting.
- **T4.0a (loss-scale normalization)** — the distinct intra-task-scale mechanism (~4.3× CE-magnitude gap, ln4703 vs ln7) — remains genuinely UNTESTED (needs a small code change: divide each CE by log(num_classes) before the combiner). It's the highest-remaining Tier-4 EV, but RLW indirectly exercises extreme scale imbalances and still matches G → scale-norm is also predicted low-EV.

**Chain status**: Tier 4 opened; T4.0b (RLW litmus) DONE. The litmus down-weights the entire balancer axis. Champion G unchanged. Committed.

**Next / decision point for the user**: the RLW litmus (the T4.0 gate) says the inter-task weight axis is flat → T4.1 is low-EV. Options: (a) implement + run T4.0a scale-norm (distinct mechanism, small code change, but predicted low-EV); (b) run a minimal confirmatory T4.1 subset (e.g. db_mtl ⊕ scale-norm, the one with a scale rationale) anyway for completeness; (c) treat the loss/optimization axis as low-EV-confirmed (RLW + P4 + T2V.6) and move to T5.3 (HSM reg head) / CA-TX completeness / close. Surfaced to user.

---

## 2026-06-08 — ✅ TIER 4 CLOSED: audit found+fixed a balancer wiring bug; gradient cosine≈0 = the mechanism; static_weight is Pareto-optimal

**Phase**: Tier 4 close-out. The first T4.1 screen ("no balancer beats static_weight; all cluster at equal_weight") was flagged by the user as suspicious → meticulous audit (2 sub-agents + direct probes + corrected re-runs + 3 figures). Full write-up: `docs/results/mtl_improvement/T4_audit_and_verdict.md`.

**The audit (user-requested "we must be losing something"):**
- **Real bug found:** gradient-surgery (cagrad/pcgrad/aligned_mtl) applies its combined gradient ONLY to `shared_parameters()`; under G's dual-tower the private reg tower (∈ `reg_specific`, >80% of the reg pathway) trains at UNIT weight always → they collapse to ≈equal-weighting (live probe: cagrad logs w=[1,0] but its private-tower grad == equal_weight's). → those 3 cells DON'T COUNT.
- **Misconfigured at defaults:** gradnorm (lr=1e-3 + L1-renorm-to-sum=2 → range 0.016, can't reach 0.25/0.75), dwa (per-batch loss history → pinned ≈1.0), fairgrad (step_size too small).
- **Latent preflight bug FIXED:** `_BACKWARD_ONLY_LOSSES` omitted cagrad+aligned_mtl (TypeError under grad_accum>1; safe here as default_mtl pins grad_accum=1).
- **Correctly wired (valid):** nash, uw, uw_so, db_mtl, bayesagg, excess, stch, go4align, famo, scheduled_static, equal_weight, static_weight, random_weight.

**The decisive mechanism — task gradients are ORTHOGONAL.** cos(∇cat,∇reg) on shared ≈ 0 at BOTH states (FL +0.0007 / AL +0.0026, band [−0.08,+0.19]). No conflict → no balancer CAN help (even correctly wired). Figure `figs/grad_cosine_tasks.png`. Literature (Kurin/Xin NeurIPS'22, Royer'23, 2025) confirms: tuned scalarization beating advanced MTO at k=2-with-tuned-baseline is the EXPECTED result; balancers only help under strongly-negative cosine.

**Corrected re-runs (bug-aware retune + Xin'22 fairness), FL+AL seed0:**
- gradnorm@lr0.05: FL +0.12 reg / −1.29 cat; nash@max_norm2.2: FL +0.09 reg / −1.51 cat (AL cat +1.50 REVERSES at FL → state-dependent, not a win). Both just trade cat for reg (cos≈0 → no free lunch).
- static cw-sweep {0.5,0.6,0.66,0.8}: a monotone reg↔cat trade; NO cw Pareto-beats 0.75 → G's weight is on the Pareto front (baseline tuned as hard as challengers).
- **T4.0a scale-norm FALSIFIED:** divides reg CE by log(4703)=8.46 → starves the high-card reg head (FL reg 35.47 at default cw; 70.25/68.52/63.51 across reg-favorable cw, never reaching G 72.95). The large reg CE is a genuinely harder task needing the gradient, not unfair dominance.
- **T4.0b RLW** (earlier): inter-task weight not a sensitive lever.

**Decision**
- **Tier 4 CLOSED — clean, bug-free, fairness-checked NEGATIVE.** Six convergent lines (RLW, full screen, tuned+arch-wired re-run, static sweep, scale-norm falsified, cosine≈0). G's `static_weight cw=0.75` is Pareto-optimal; the loss/optimization axis yields no gain. Paper-grade ("no balancer beats tuned scalarization at k=2"). No multi-seed promotion warranted.
- **Unifying mechanistic payoff:** the orthogonal task gradients explain the WHOLE study — why balancers can't help (Tier 4), why forcing more sharing failed (Tier 2: it would induce the conflict that isn't there), and why the dual-tower wins (protect reg, let cat harvest the shared representation). The "matches reg / beats cat +3pp" Pareto-positive result = the signature of an orthogonal task pair handled by the right architecture.
- **3 figures** produced (for the BRACIS talk): grad_cosine_tasks, t4_balancer_scatter_FL, t4_loss_weight_trajectories_FL.

**Open follow-up flagged to user:** the +3pp cat transfer is representation-level (not gradient-mediated, per cosine≈0); a clean ablation (category-only with the same cross-attn trunk) would verify it's region-driven vs architecture-driven — proposed, not yet run.

**Chain status**: Tier 4 CLOSED. Champion G unchanged. Tiers 3+4 both closed. Committed + pushed.

**Next** (pending user): (a) optional cat-transfer ablation; (b) T5.3 HSM reg head (the last live reg-pathway lever); (c) CA/TX completeness; (d) study close → paper restatement.

---

## 2026-06-08 — (a) cat-transfer ablation + (b) T5.3 HSM — both decisive

**Phase**: two user-requested follow-ups after the Tier-3/4 closes. seed0. Docs: `docs/results/mtl_improvement/cat_transfer_and_T53.md`.

**(a) Cat-transfer ablation — the +3pp MTL-cat gain is ARCHITECTURE-dominated, not region-transfer.**
Ran G's recipe with `--category-weight 1.0` (reg loss ×0 → reg gradient OFF; reg cratered to 0.12%, confirming the shared trunk is cat-only-trained). Decomposition (STL cat → cat+trunk-reg-OFF → G):
| state | STL (no trunk) | cat+trunk reg-OFF | G (reg ON) | architecture | region-transfer |
|---|---|---|---|---|---|
| AL | 50.35 | 53.46 | 52.75 | **+3.11** | **−0.71** |
| FL | 69.96 | 72.23 | 73.12 | **+2.27** | **+0.89** |
→ The cat win over STL is **mostly the cross-attn architecture** (+2.3..+3.1pp); genuine region co-training adds only **+0.89 at FL** and slightly HURTS at AL (−0.71). Exactly consistent with gradient-orthogonality (little direct transfer; the gain is a better encoder + a small scale-dependent representation transfer). Caveat: the STL ceiling used logit-adjust (which inflates STL cat / hurts MTL cat) → the architecture component is conservative. **Paper implication:** re-state the cat result as "the joint cross-attn architecture is a better category encoder (+ small region→cat transfer at scale)," NOT "the region task teaches category."

**(b) T5.3 HSM reg head — FALSIFIED.** HSM (`next_stan_flow_hsm`) is single-pathway (no dual-tower) → tested the mechanism at the STL/ceiling level first (built FL hierarchy, 69 clusters; p1 FL seed0 prior-OFF): flat `next_stan_flow` 73.22±0.77 vs `next_stan_flow_hsm` 73.21±0.80 — identical (−0.01, within σ). Hierarchical softmax gives NO accuracy gain at 4,700 classes (speed/memory technique, not accuracy). No dual-tower-HSM build motivated. Tier-5's last residual closed; flat softmax sufficient. (CA/TX untested; gain unlikely.)

**Decision**
- Both clean: cat gain = architecture-dominated (refines the paper claim, consistent with orthogonality); HSM = flat (Tier-5 closed). **Champion G unchanged.**
- New code: `--loss-scale-norm` (T4, gated) earlier; FL region hierarchy artifact built. No champion change from either probe.

**Chain status**: (a)+(b) done. Tiers 2V/3/4/5 all closed; champion G robust. Remaining open work is completeness only: CA/TX (C-A) + paper-canon restatement (C-B), both author/heavy.

**Next** (pending user): CA/TX completeness (heavy v14 build) and/or BRACIS paper-doc restatement; otherwise the experimental study is at a clean, comprehensive close.

---

## 2026-06-09 — T5.2 cat-head sweep RE-RUN UNDER champion G (user: "why is Tier 5 closed? it's important") → next_gru confirmed, on real evidence

**Phase**: Tier 5 re-opened + executed properly. User challenged the Tier-5 closure — correctly: T5.1/T5.2 had carried "DONE" verdicts INHERITED from Tier-S (STL head search) + B-A1 (reg private-tower swap) + B-A4/T2V (cat LOSS family). Only T5.3 (HSM) had been run this arc. The genuine open gap was a clean cat-ENCODER swap under champion G — especially important since the cat-transfer ablation showed the cat gain is architecture-driven. Ran it.

**What ran:** 10 cat-capable encoders (next_gru control + lstm/transformer_relpos/transformer_optimized/single/temporal_cnn/tcn_residual/conv_attn/hybrid/mamba) × {AL,FL} seed0, matched-metric, swap `--cat-head` only under G. Driver `t52_cathead_sweep.sh`; data `T52_cathead_sweep.json`; doc `T52_cathead_sweep.md`.

**Findings (Δcat vs next_gru):**
| cat-head | AL | FL | |
|---|---|---|---|
| next_gru (G) | 52.75 | 73.12 | CHAMPION — only head strong at both |
| next_conv_attn | −21.50 | **+1.06** | FL-only (craters AL) |
| next_temporal_cnn | −23.60 | +0.59 | FL-only |
| next_lstm | −1.28 | +0.34 | FL-only |
| next_hybrid | −2.55 | +0.22 | ≈/worse |
| transformers/single/mamba/tcn | −9…−16 | −0.9…−9.5 | worse |

reg flat across all heads (AL range 0.31 / FL 1.16) — the dual-tower isolates reg from the cat head.

**Decision**
- **next_gru CONFIRMED as the multi-state cat champion UNDER G** (real under-G evidence, not inherited). **No head wins at both states** — every FL-beater (conv_attn +1.06, temporal_cnn +0.59, lstm +0.34) craters at AL (−1.3…−23.6). Same FL-only trap as G′ / next_single (CONCERNS §C26) → fails the multi-state band gate.
- **Bonus (future-work, NOT adopted):** `next_conv_attn` is a genuine +1.06 FL-only cat lever (scale-conditional, in the overlap/design_k family) — logged for a possible scale-conditional paper cat head, not a champion change (the study ships one multi-state recipe).
- **Honesty correction:** "Tier 5 closed" was previously closed-by-inheritance; it is now **closed-by-direct-under-G-evidence** for the cat axis. Reg axis (T5.1): coded private-tower types (stan/gru/lstm/tcn) swept in B-A1 (STAN load-bearing); other reg archs as private towers need new code (low-EV, STAN-family also won at STL). T5.3 HSM falsified. → Tier 5 genuinely closed.

**Chain status**: Tier 5 re-validated under G; champion G unchanged. Committed.

**Next** (pending user): only Tier 6 completeness remains — CA/TX + BRACIS paper restatement (see HANDOFF_TIER5.md).

---

## 2026-06-10 — Final critical-advisor pass → P0 (orthogonality confound) RESOLVED + P1 decomposition multi-seeded → clear to close

**Phase**: user-requested final "are we missing anything?" advisor pass + acting on it. The advisor marked Tier 4 / G′ / HSM / R1 / the headline G claim all genuinely CLOSED, and flagged exactly ONE P0 + paper-safety polish. Both run; both strengthen the conclusions.

**P0 (the one load-bearing gap) — is cos≈0 INTRINSIC or induced by the dual-tower's β=0.1 aux gating?** RESOLVED — intrinsic. Evidence (`results/mtl_improvement/orthogonality_intrinsic_test.md`):
- P0-a (free, from logged grad-norms): in the dual-tower, reg's shared gradient IS attenuated (reg:cat ratio 0.26 AL / 0.47 FL) — the confound was real to check.
- P0-b (decisive): in a **fully-shared** model (`mtlnet_crossattn` + `next_stan_flow`, no private tower) where reg's shared gradient is **larger** than cat's (ratio **1.26 AL / 1.78 FL**), the cosine is **still ≈0** (+0.0024 AL / +0.0017 FL). The orthogonality persists exactly where the confound predicted it should vanish → **intrinsic to the task pair, not architecture-induced.** This STRENGTHENS the paper: the dual-tower *exploits* a pre-existing orthogonality, it doesn't manufacture it.

**P1 — multi-seed the cat-transfer decomposition (4-seed {0,1,7,100}):** architecture-dominated holds; region→cat transfer **+1.08 FL** (σ~0.08, firmed up from seed0 +0.89) / **−0.67 AL** (from −0.71). Both signs robust. + honesty caveat added: the reg-OFF ablation isn't perfectly clean (cat still attends to reg's K/V in the bidirectional cross-attn at reg-weight 0).

**Decision**
- **We are CLEAR to close.** The advisor's single P0 is resolved in favor of the existing narrative (now tested, not asserted); the P1 numbers are multi-seed paper-safe. No conclusion changed; several were strengthened.
- **GPU note:** a non-study `python3 main.py` job (26GB, not ours) was running on the shared A40 — left untouched; our runs fit in the headroom.
- Docs settled: `orthogonality_intrinsic_test.md` (new), WHY doc (Q2 multi-seed + co-adaptation caveat; Q3 intrinsic-tested), `cat_transfer_and_T53.md` (multi-seed), CLAIMS CH31 (tested-intrinsic) + CH30 (multi-seed cat numbers), CHAMPION.md.

**Chain status**: study closed + hardened; champion G unchanged. Only Tier-6 completeness (CA/TX) + paper restatement remain (HANDOFF_TIER5.md).

**Next** (pending user): Tier 6 completeness / paper restatement; the experimental + mechanistic track is comprehensively closed.

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
