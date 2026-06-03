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
