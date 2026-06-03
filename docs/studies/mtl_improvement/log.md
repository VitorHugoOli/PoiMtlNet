# MTL Improvement Track — Progress Log

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

**Next**: await the user's cumulative-scope decision (cap Tier S at S.1-S.4, or keep growing). Then implementing agent: T1.4 → Tier S (Prong A screen → Prong B → S.3 compose → S.4 picture).

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
