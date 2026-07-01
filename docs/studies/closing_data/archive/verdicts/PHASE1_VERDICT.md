# PHASE 1a VERDICT — cross-study re-evaluation under the frozen-candidate recipe

> **Study:** `closing_data` · **Phase:** P1a (cross-study re-eval sweep) · **Date:** 2026-06-16 ·
> **Branch:** `study/closing-data-p1a` · **Type:** reading-only analysis, NO code/data changes.
>
> **What this answers (PLAN §P1a + AGENT_PROMPT mission #1):** for each closed study, does its
> headline verdict still HOLD under the frozen-candidate recipe? Which parked levers should become a
> `closing_data` G0.2 pre-freeze gate vs be confirmed dead? And what are the RE-RUN / REUSE /
> STORY-DEPENDENT dispositions of the BRACIS suite that P1b turns into `RUN_MATRIX.md`?
>
> **Companion already on disk:** `docs/research/community_insights.md` is a thorough cross-study
> re-read (2026-06-12). This verdict does not re-derive it; it adds the **recipe-dependence
> adjudication** that re-read did not do.

---

## 0 · The frozen-candidate recipe (the yardstick every closure is tested against)

Pinned from `docs/results/CANONICAL_VERSIONS.md` §v14–v16 + `mtl_improvement/FINAL_SYNTHESIS.md` §2,
re-checked at launch (no drift since scaffold — v14 still the blessed base, v16 still champion):

| Axis | Frozen candidate | Was (BRACIS / v11 paper canon) |
|---|---|---|
| **Recipe / champion** | **v16 = champion G**: `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` (aux fusion, `freeze_alpha=True alpha_init=0.0` → **additive prior OFF**), cat head `next_gru`, `static_weight cw=0.75`, **onecycle** max-lr 3e-3 | B9 (FL/CA/TX) / H3-alt (AL/AZ), single cross-attn, additive prior ON |
| **Loss** | **UNWEIGHTED CE both heads** (v15, the C25 fix) | class-WEIGHTED CE (the C25 confound) |
| **Substrate** | **v14 = `check2hgi_design_k_resln_mae_l0_1`** (ResLN + Delaunay POI-GCN reg lever + mae cat lever) | canonical `check2hgi` GCN substrate (frozen `output/check2hgi/<state>/`) |
| **Selector** | **`geom_simple`** = √(cat_F1 · reg_Acc@10) | broken `0.5·(cat_f1+reg_f1)` (`joint_f1_mean`) |
| **log_T-KD** | **OFF** (W=0.0) — and **null on G**, see §2 | v12 default W=0.2 (single-pathway) |
| **Protocol** | all states × seeds {0,1,7,100} × 5 folds (n=20), matched-metric FULL `top10_acc` fp32 | mixed (some §0.x cells single-seed-42 pilots) |

**The one fact that reshapes everything below:** the frozen recipe **inverts the BRACIS headline
thesis.** §0.1/T3 reported "MTL sacrifices region prediction by −7…−17 pp"; under v16/G that gap
**dissolves** — a single MTL model *matches* the STL reg ceiling (matched Δ −0.09…−0.31) and *beats*
the STL cat ceiling +2.6…+4.1 (`mtl_improvement/FINAL_SYNTHESIS.md` §2, R0 bar, 4 states × 4 seeds).
That single inversion is why most of the model-derived suite RE-RUNS (§3) and the paper story changes.

---

## 1 · Per-study HOLD / FLAG verdict

Legend: **HOLDS** = portable to the frozen recipe as-is · **HOLDS (re-read)** = conclusion survives
but was measured under a superseded recipe/confound so re-state, no re-run needed · **RE-RUN** =
numbers change with the frozen recipe, must regenerate · **SUPERSEDED** = the question itself is moot
under the frozen recipe.

| Study | Closed | Recipe it closed under | Headline verdict | Status under frozen recipe |
|---|---|---|---|---|
| **mtl_improvement** | 2026-06-12 | **IS the frozen recipe** (v15/v16, unweighted, geom_simple, v14, {0,1,7,100}) | C25 confound WAS the "MTL sacrifices reg" gap; champion G matches STL reg + beats STL cat; mechanism = gradient orthogonality; no balancer/optimizer/substrate/prior lever helps on G | **HOLDS** — this study *defines* the frozen recipe. The R0 matched-metric bar is the citable base. One counterfactual open by construction → **G0.1 gate** (§2). |
| **embedding_eval** | 2026-06-02 | STL leak-free multi-seed FL (build of v14); MTL check only a seed-42 2-fold pilot | v14 is the STL dual-axis champion (cat ≈ frozen-canon ≫ HGI; reg closes ~69% of HGI gap). **NO MTL benefit** from v14 or routing — regime is the wall | **HOLDS** for the STL/substrate identity (v14 IS the frozen substrate). The "no MTL benefit" verdict is **re-confirmed** by mtl_improvement's full multi-seed v14-vs-canonical MTL run (§0.1 annotation + `v14_mtl_vs_canonical.md`): v14 ≈ canonical in MTL. Adjacency-aware region-head lead = STL/future-work (§2). |
| **substrate-protocol-cleanup** | 2026-05-28 | **pre-C25** (class-weighted CE), B9/H3-alt, **two-front scoring incl. the deployable `geom_simple` front** (Tier B `phase_b_two_front.md`), mostly seed-42 small-state | Substrate axis null in MTL (regime finding); log_T-KD W=0.2 promoted (+2–5 pp AL/AZ, n=20); HGI's STL reg edge vanishes in MTL (p=0.41); curriculum/freeze forms dead | **MIXED.** Regime finding (substrate null in MTL) **HOLDS (re-read)** — re-confirmed post-C25 on G (every substrate/optimizer/prior lever null on the dual-tower). Selector is NOT the gap: Tier B already scored on the `geom_simple` front (the now-frozen selector), so its null verdict is selector-portable; the residual caveat is pre-C25 + single-cross-attn arch, not the selector. **log_T-KD promotion is SUPERSEDED on G** — X2 finds KD null on the dual-tower (§2); §0.8 is a B9/v12-single-pathway result, NOT a champion-G result. 3-snapshot routing C1 → adjudicated as the one live harvest item (§2). |
| **mtl-protocol-fix** | 2026-05-24 | **pre-C25**, FL multi-seed n=4 + small-state seed-42; introduced `geom_simple` | `geom_simple` selector recovers ~95% deploy capacity (+5.6 pp FL) vs the broken F1-mean; residual MTL-reg gap is architectural (not cat-vs-reg, not long-tail, not substrate); class-balanced sampler catastrophic (−18…−30 pp); composite STL c2hgi+HGI = then-headline | **HOLDS (re-read).** The `geom_simple` selector is **ADOPTED** into the frozen recipe — its qualitative win is recipe-independent (geom-mean of per-task primary metrics); the absolute "+5.6 pp" was measured pre-C25 so don't quote the number, cite the mechanism. "Residual gap is architectural" was the correct pointer that mtl_improvement then closed via the dual-tower. Class-balanced-sampler null = recipe-stable (it's a metric-geometry fact, insight #8). Composite headline is **SUPERSEDED** (G dominates it — `FINAL_SYNTHESIS.md` §2). |
| **canonical_improvement** | 2026-05-19 | **pre-C25**, class-weighted CE, single seed-42, canonical/v3c substrate (v14 didn't exist) | Substrate axis exhausted at ±0.8 pp; ResLN +1.2–1.7 pp cat (STL-only); GATv2 label-copy leak; DropEdge = budget-rescue artifact; boundary-weight inert | **HOLDS (re-read), substrate-scoped.** Hygiene findings (ResLN STL-cat, GATv2 leak, DropEdge inversion, α/boundary inert) are recipe-stable *as STL/representation findings*. The "±0.8 pp substrate ceiling" was measured on the **pre-v14 substrate** — v14 (design_k) itself overturned one discard (design_k was wrongly killed at AL/AZ here, reopened at FL by embedding_eval). So the ceiling claim is **substrate-relative**, not a universal — but no action: v14 is already the frozen base. |
| **merge_design** | ACTIVE-CLOSING (2026-05-06) | **pre-C25**, STL-focused AL/AZ leak-free + Phase-11 audit | Designs A–M / Levers 1–6 saturated or falsified; STL merge family (B/H/J) Pareto-dominates canonical; design_k ≈ design_j on reg | **HOLDS, no open thread.** The one orphan (Lever 5, distribution-level POI distillation) was **rescued and FALSIFIED** by substrate-protocol-cleanup Tier B4 (null in MTL under geom_simple, anchor-dominated). design_k graduated into the v14 substrate. **No surviving open item** despite the "ACTIVE-CLOSING" label → safe to mark CLOSED. (PLAN flagged this study to verify first; verified.) |
| **hgi_category_injection** | 2026-05-04 | **pre-C25**, AZ-only, seed-42, 30 ep | 6 injection variants null-to-noise; HGI's POI2Vec ≈ an fclass lookup so injecting category into it is tautological | **HOLDS (mechanism), stays CLOSED.** Mechanism (category-redundant injection is inert) is substrate-intrinsic, not recipe-sensitive. Re-open criteria (FL/CA/TX multi-state) are **out of `closing_data` scope** unless the story pulls it in. No gate. |
| **fusion** (archived engine) | archived 2026-05-14 | **pre-C25 + NashMTL ECOS bug + HGI leak** (triple-confounded) | Optimizer choice is embedding-dependent: gradient-surgery essential under multi-source fusion, equal-weight suffices single-source | **NOT PORTABLE (do not cite MTL numbers).** Triple-confounded; the frozen recipe is single-source `static_weight`, so the multi-source optimizer claim doesn't transfer. The "equal-weight suffices single-source" direction is *consistent with* (not evidence for) the static-weight choice. Engine stays first-class in code; study stays archived. No gate. |

### Q1 closures reached under a superseded recipe/selector/confound (the explicit flags)

1. **§0.8 log_T-KD promotion (PAPER-GRADE at AL/AZ, n=20) is SUPERSEDED on champion G.** It was
   measured on the **B9/H3-alt single-pathway** recipe with the additive prior live. On G (dual-tower,
   prior OFF) the **X2 test is NULL** (FL reg +0.05 / AL reg −0.13, both ≪ the 0.3 pp gate; FL cat
   −0.57) — and the *old* "KD adds nothing on the dual-tower" was itself a dead-codepath bug (aux gate,
   C28), so X2 is the **first real test** and it stands. **Implication:** log_T-KD is a pre-G lever, not
   a champion-G lever. The §0.8 table is reusable only as a *standalone B9-recipe panel* (STORY-DEPENDENT,
   §3), never as part of the G board. Source: `mtl_improvement/X_SERIES_FINDINGS.md §X2`.
2. **§0.1 / §0.2 / §0.4 (the BRACIS MTL headline + Δm + recipe selection) were measured pre-C25 under
   class-weighted CE on the GCN substrate with B9/H3-alt.** All three are SUPERSEDED — see §3. §0.4
   specifically: "B9-vs-H3-alt is scale-conditional" is **moot** — G uses onecycle, which the §0.1
   annotation shows dominates H3-alt/B9 at small states. The recipe-selection question changes shape.
3. **substrate-protocol-cleanup + canonical_improvement + merge_design + mtl-protocol-fix all closed
   pre-C25** (C25 was discovered 2026-06-05; these closed in May). Their **MTL** conclusions are
   regime-dependent and were correctly re-read by `community_insights.md` insight #1 and re-confirmed
   on G by mtl_improvement. Their **STL** conclusions are recipe-stable. None needs a re-run *as a
   closure*; their numbers feed the suite re-run (§3), not a re-litigation.

---

## 2 · Parked / promotable levers → G0.2 gate dispositions

**The Q2 cross-check, corrected.** The prompt's anchor — *"output-priors are the one live MTL-reg
channel; substrate/optimizer levers are null"* — is a **B9/v12-era** statement (§0.8). Under champion
**G it no longer holds**: X2 shows log_T-KD null on the dual-tower, and X3 shows the model
**gradient-gates the shared→reg pathway off** (β: 0.108→≈0 by epoch ~25, WD-independent) — reg lives
entirely in the private tower. **So on G there is no demonstrated live MTL-reg channel: substrate,
optimizer, AND output-prior levers are all null.** This *strengthens* the freeze rationale and reframes
the prior-based exploration gates: `mtl_frontier`'s R1/R3 start from "the prior channel was null on G,"
not "the channel is live." (Beyond-parity mechanisms — conditional coupling, cat-conditioned prior,
semantic-ID region factorization — remain untested on G and are the genuine frontier; `FINAL_SYNTHESIS.md` §4.)

**The G0.2 gate ledger already exists** in `docs/studies/PRE_FREEZE_PROGRAM.md` (G0.1 + mtl_frontier
R1/R2/R3/R10 + pre_freeze_gates A2/A4/overlapping-windows + baseline_gap B1–B5). P1a's job is **not to
re-derive it** but to disposition the *parked items from the closed studies + `docs/future_works/`*
against it. Each below gets exactly one disposition:

| Parked lever | Source | Disposition under frozen recipe |
|---|---|---|
| **Aligned-pairing training test** | mtl_improvement X1 (open by construction) | **ALREADY A GATE (G0.1)** — the only mandatory P0 gate. The roll probe is circular against "mixing learnable under aligned pairing"; needs ONE shared permutation for both loaders. ≥0.3 pp either head → v17. Keep as-is. |
| **log_T co-location prior / cross-task distillation** | mtl_frontier R1/R3 + `future_works/joint_selection*` | **ALREADY GATES (R1/R3).** Re-anchor: these now start from "prior channel null on G" (X2/X3), so the bar is "does a *richer* output prior beat the null." Disposition unchanged; rationale corrected. |
| **STEM-AFTB gating / GRM-SSC layer read** | mtl_frontier R2/R10 | **ALREADY GATES (R2/R10).** Asymmetric-sharing exploration — the live frontier per §2. Keep. |
| **3-snapshot per-task routing (C1)** ★ | substrate-protocol-cleanup | **THE ONE LIVE HARVEST ITEM — needs a user call (decision below the table).** It is the only parked lever with a positive signal that the regime finding does NOT auto-null. |
| **Adjacency-aware region head (GCN², log_T-orthogonal)** | embedding_eval | **NOT a gate — STL/future-work.** Only positive STL lead left in embedding_eval (+0.56 pp NS at L2), **never tested in MTL.** Under the regime finding (substrate/encoder washes out in MTL) the prior is MTL-null; it is a substrate/representation lever for a future MTL-frontier study, not a closing_data gate. |
| **Overlapping windows** | `future_works/overlapping_windows.md` (validated AL) | **ALREADY A GATE (pre_freeze_gates).** Base change → if adopted forces full-base rebuild + leak re-audit + baseline re-match. Must resolve before freeze. Keep. |
| **A2 feature-concat control / A4 transductivity bound** | pre_freeze_gates | **ALREADY GATES (interpretation/disclosure).** Don't change numbers, change claims; must resolve pre-freeze so RUN_MATRIX records caveats. Keep. |
| **External baselines B1–B5 (CTLE, POI2Vec, HMT-GRN-MTL, cascade, Flashback)** | baseline_gap | **ALREADY A GATE (P1b inventory decision).** Which engines the board carries is a RUN_MATRIX row decision; runs fold into P3 at the frozen protocol. Keep. |
| **Composite two-substrate engine** | mtl-protocol-fix §4.2 + `future_works/composite_two_substrate_engine.md` | **STORY-DEPENDENT (deploy-pattern panel), CONFIRMED NOT a champion.** G dominates the composite on the joint reading (`FINAL_SYNTHESIS.md` §2 + corrections registry). Park for the story; don't gate. |
| **Lever 5 (distribution POI distillation), Designs B/J, HGI-as-MTL-substrate** | merge_design / substrate-protocol-cleanup Tier B | **CONFIRMED DEAD in MTL.** All null under the regime finding (anchor-dominated reg head); STL gains real but don't transfer. No gate. |
| **POI-decoder HGI distill, dual-substrate routing, substrate-adaptive balancing** | `future_works/*` | **FUTURE-WORK / regime-null.** Per PRE_FREEZE_PROGRAM dispositions; substrate-adaptive balancing + dual-substrate routing are regime-null under C25 (re-read, not re-run). No gate. |

### ★ The 3-snapshot per-task routing decision (the one harvest item that warrants a user call)

This is the only parked lever with a positive signal that the regime finding does **not** dispose of,
so it gets a real adjudication rather than a hand-wave:

- **What it is:** at deploy time, route each task to its *own* best-epoch checkpoint (two snapshots
  from one training run) instead of picking a single joint checkpoint. Prototype shipped
  (`--save-task-best-snapshots`, `route_task_best.py`).
- **Evidence:** passed the pre-registered +2 pp reg gate at **2 of 3 states** — AZ +2.54 (p=0.031, 5/5),
  FL +2.80 (p=0.0312, 5/5, clean) — and **failed at AL** on one genuine degenerate Acc@1-selected
  snapshot. Verdict in `substrate-protocol-cleanup/CLOSURE.md` is "§Discussion footnote, not promoted;
  an Acc@10-aligned reg-best selector + a degenerate-snapshot guard are warranted before promotion."
- **Why it is NOT auto-dead:** it is a **deploy-pattern / output-side lever**, not a substrate or
  encoder lever — so the regime finding (substrate/encoder washes out in MTL) does not null it the way
  it nulls Lever 5 / Designs B/J. I verified it is **not covered by any mtl_frontier gate**: R2 is
  per-layer AFTB stop-gradient gating *between the towers*; R10 is GRM/SSC read between towers — neither
  is per-task temporal-snapshot routing. "Fold into R2" would be a mis-assignment.
- **Why it is NOT clean either:** measured **pre-C25, single cross-attn arch, against the depressed
  old baseline, on a checkpoint-selection axis the frozen `geom_simple` selector now partly addresses**.
  Its mechanism (route to per-task-best) is a two-checkpoint deploy pattern — the same family as the
  composite (two models), which G already dominates on the joint reading. So part of its 2.8 pp may be
  recovering exactly what C25-fix + dual-tower + geom_simple already recovered. **Untested on G.**
- **Recommendation (user call):** treat as **STORY-DEPENDENT (alternative deploy panel)** + a **cheap
  P3 confirm-on-G** (one FL+AL G run with per-task-best routing vs single geom_simple checkpoint) — it
  changes a *deploy mode*, not the single-model champion recipe, so it does not need to be a recipe
  (G0.2) gate that blocks the freeze. **Escalate to a G0.2 gate only if the user wants it settled
  before the freeze** (it is cheap: ~the C1 train×2 + re-score×2 budget, ≪1 GPU-h). Either way it is
  NOT confirmed dead — flag it open for the user, do not bury it.

**Net Q2:** **no NEW *recipe* (G0.2) gate is mandatory from the closed-study harvest** — the existing
PRE_FREEZE_PROGRAM ledger already homes every recipe-changing candidate, and every closed-study lever
except one is either already-gated, regime-dead, or STL/future-work. **The one exception is 3-snapshot
routing C1** (above): not regime-killed, untested on G, recommended as a STORY-DEPENDENT deploy panel +
cheap P3 confirm — a user go/no-go, not a silent drop. The adjacency-aware region-head lead
(embedding_eval, STL +0.56 NS, MTL-untested) stays STL/future-work. The one correction P1a contributes
to the freeze rationale: **on champion G *all three* lever classes (substrate, optimizer, output-prior)
are null**, not just substrate/optimizer — X2/X3 close the prior channel §0.8 had left open. ⚠ X2/X3 are
**single-seed (seed0), FL+AL only** — the direction is decisive (both ≪ the 0.3 pp gate) but the
*multi-seed* confirmation that the prior channel is dead on G is exactly what `mtl_frontier` R1/R3 will
establish; treat "no live channel on G" as pilot-confirmed, multi-seed pending.

---

## 3 · RUN_MATRIX inputs for P1b (BRACIS suite disposition)

Rule (PLAN §P1b): numbers change with recipe/substrate → **RE-RUN** (default for anything
model-derived); recipe-independent → **REUSE**; cut-to-prose / substrate-comparison / narrative-gated
→ **STORY-DEPENDENT**. The single-substrate board (v14) makes every Check2HGI-vs-HGI panel
story-dependent by construction.

| Cell | What backs it today | Disposition | One-line reason |
|---|---|---|---|
| **T1** dataset stats (§4.1) | `data/<state>` + `output/check2hgi/<state>/regions` | **REUSE** (recompute placeholder cells) | Corpus facts — recipe-independent. Only caveat: **if overlapping-windows is adopted**, sequence/traj-len counts change → then RE-RUN. Add CA/TX/GE rows. |
| **T2** substrate ablation, C2HGI vs HGI both tasks (§5.1 / §0.3) | STL `next_gru` + `next_stan_flow`, GCN substrate, seed-42 | **STORY-DEPENDENT → RE-RUN if kept** | Single-substrate board drops the comparison by default; substrate identity changed (GCN→v14) so numbers change if the story keeps the panel. |
| **T3** MTL vs STL both tasks (§5.2 / **§0.1**) | B9/H3-alt, class-weighted CE, GCN, n=20 | **RE-RUN** ★ | The central table. Frozen recipe **inverts** it (−7…−17 pp gap → matches/beats). Highest-priority regeneration; the story flips with it. |
| **T4** Δm joint score (§5.2 / §0.2) | leak-free CH22, mostly seed-42 (FL n=25) | **RE-RUN** | Derived from T3 + selector-dependent (geom_simple); recipe + selector both changed. |
| **T5** external baselines (§5.3 / §0.5–0.6) | STAN/ReHDM/POI-RGNN/MHA+PE, BRACIS-era lighter protocol | **RE-RUN** (per-engine inclusion STORY-DEPENDENT) | **User-decided: re-run at full n=20 under the new regime.** baseline_gap B1–B5 add net-new rows; P1b dispositions *which* engines, not *whether* the protocol changes. End-to-end baselines must mirror the adopted windowing. |
| **F1** per-visit mechanism, AL (§6.1, REQUIRED) | STL linear-probe + `next_gru`, canonical C2HGI vs pooled vs HGI (§0.7) | **STORY-DEPENDENT → RE-RUN if kept** | Substrate-asymmetry mechanism for the *canonical* substrate; under a single-v14 board it depends on whether the story keeps the per-visit narrative; STL numbers but substrate-identity-dependent. |
| **F2** scale-progression scatter (optional) | §0.1 Δ_reg column | **STORY-DEPENDENT (likely obsolete)** | Plots the OLD −7…−17 pp gap that *dissolves* under G; the "scale-conditional cost" story it visualizes no longer exists. Re-derive only if a residual-gap story survives. |
| **F-arch** architecture schematic (optional) | — | **REUSE (redraw, no compute)** | Diagram, not numbers — but must be redrawn for the **dualtower** (G), not single cross-attn. |
| **§0.4** recipe selection B9 vs H3-alt | n=20 multi-seed | **SUPERSEDED → RE-RUN/reframe** | G uses onecycle; B9-vs-H3-alt is moot. New question: onecycle vs per-state recipe variant (PLAN P2 "one recipe or documented small/large split"). |
| **§0.8** log_T-KD reg lift | B9/v12 single-pathway, AL/AZ n=20 | **STORY-DEPENDENT standalone panel only** | Null on G (X2). Reusable only as a B9-recipe lever panel if the story wants the "prior channel pre-G" result; NOT a G-board cell. |
| **§0.9** substrate-null-in-MTL (regime finding) | pre-C25, B9, seed-42 | **RE-RUN (lightweight) / re-read** | The mechanism HOLDS on G (re-confirmed), but the supporting cells are pre-C25; regenerate the key isolation cell under the frozen recipe so the regime claim cites a G-era number, or cite the mtl_improvement R0/X-series instead. |

**Provenance discipline for every RE-RUN cell** (C28 hard rules, AGENT_PROMPT §Hard rules):
PID-suffixed rundirs + per-run seed echo; seed-tagged per-fold log_T freshness preflight before any
`--per-fold-transition-dir` run; matched metric/seeds/folds/precision on both sides; pin `--canon` in
every driver. v14 substrate must be built at **CA/TX/GE** first (M0 — only FL/AL/AZ exist today per
CANONICAL_VERSIONS §v14: "AL/AZ/CA/TX pending" — verify at P3 launch).

---

## 4 · Flags that should block or reshape the freeze

1. **The story changes, not just the numbers (reshape, not block).** The frozen recipe inverts the
   BRACIS central thesis (§0). T3/T4/F2 and the framing of T2/F1 were built on "MTL sacrifices reg."
   `closing_data` is story-agnostic and regenerates the base regardless, but the freeze notes + the
   hand-off to the new-paper story effort must carry this prominently: **the new base will say MTL
   *matches* reg and *beats* cat, the opposite of the submission.** (Already owned author-side by
   `mtl_improvement/PAPER_UPDATE.md`; do not silently rewrite §0.1 — PAPER_UPDATE rule.)
2. **No live MTL-reg channel on G (reshape Q2 anchor).** §2 — substrate, optimizer, AND output-prior
   levers are all null on champion G (substrate null = regime finding; optimizer null = T4 convergent
   negative; prior null = X2/X3). The freeze rationale and the mtl_frontier R1/R3 gate framing must
   start from this, not from the §0.8 "prior is the live channel" statement (which is pre-G). ⚠ The
   prior-null evidence (X2/X3) is **single-seed (seed0), FL+AL** — decisive in direction (≪0.3 pp gate)
   but pilot-grade; the multi-seed confirmation is `mtl_frontier` R1/R3's job. State it as
   pilot-confirmed, not settled.
3. **G0.1 (aligned-pairing) is the only open recipe-changing gate inherited here** and remains
   mandatory before P2 — it is the one untested counterfactual to "G wins without per-sample mixing."
   No closed-study lever adds a new gate (§2). The freeze cannot commit with G0.1 (or any
   PRE_FREEZE_PROGRAM ledger row) open.
4. **M0 prerequisite, not a P1a blocker:** v14 substrate exists only at FL/AL/AZ today; CA/TX/GE builds
   + seeded per-fold log_T for all four reporting seeds are the P3 critical path (H100 6 h budget).
   Flagged so P1b's RE-RUN specs carry the build prerequisite.

**Bottom line:** every closed study's *qualitative* verdict survives or is cleanly superseded — none
needs re-opening as a closure. The heavy consequence is in §3: the model-derived BRACIS suite
(T3/T4/F2 + the §0.x MTL cells) RE-RUNS because the frozen recipe inverts the headline; the
substrate/mechanism panels (T2/F1/§0.3/§0.7) are STORY-DEPENDENT on the single-v14 board; only corpus
facts (T1) and the schematic (F-arch) REUSE. No new G0.2 gate from the harvest; G0.1 stays the lone
recipe-changing P0 gate.
