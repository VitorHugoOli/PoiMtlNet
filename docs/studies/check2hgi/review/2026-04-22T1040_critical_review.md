# Critical Review — Check2HGI MTL Study (re-evaluation)

**Author:** external critical review (BRACIS reviewer lens)
**Date:** 2026-04-22T10:40 (America/Sao_Paulo)
**Prior review:** produced 2026-04-21 on the same branch (conversation artefact).
**Corpus delta read this pass:** `GETNEXT_FINDINGS.md`, `B5_PROBE_ENTROPY_FINDINGS.md`, `B7_ALIBI_GETNEXT_FINDINGS.md`, `issues/MODEL_DESIGN_REVIEW_2026-04-22.md`, `issues/MTL_PARAM_PARTITION_BUG.md`, `issues/CROSSATTN_PARTIAL_FORWARD_CRASH.md`, attribution memory note, last 10 commits on `worktree-check2hgi-mtl`.

---

## 0 · TL;DR on the new state

Two things happened in the 48 h since the previous review:

1. **A new candidate headline emerged** — `MTL-GETNext` (trajectory-flow graph prior over STAN) lifts MTL region Acc@10 by **+11.4 pp on AL, +5.6 pp on AZ, +3 pp on FL (1f)**, closes the MTL→STL gap by ~50 %, and — for the first time in the study — puts MTL region **above the Markov-1-region floor on AZ** (46.66 vs 42.96). Category F1 is unchanged.

2. **A pre-P5 self-audit on 2026-04-22 surfaced 2 blockers + 10 design smells.** The blockers are real and named (`MTL_PARAM_PARTITION_BUG.md`, `CROSSATTN_PARTIAL_FORWARD_CRASH.md`). The design smells include one that directly undermines the new GETNext story: the GETNext probe has no auxiliary supervision and — from B5's probe-entropy analysis — behaves as a **near-uniform frequency prior**, not a transition-conditional prior. The +11 pp lift may be a popularity bias, not the graph prior.

**Net direction:** the ceiling visible in the results table is higher than it was on 2026-04-21, but the **attribution of that ceiling is less clear**. Before MTL-GETNext can enter a paper as a methodological claim, the α=0 ablation and a hard-index probe comparison are both required.

**BRACIS deadline context:** 2026-04-20 AoE is two days past. This review assumes (a) the submission went out with the evidence that existed on 2026-04-20 evening (cross-attn + pcgrad as headline), OR (b) the deadline was missed. Either way, MTL-GETNext is not in the submitted paper — its findings are post-deadline. It matters for camera-ready or a follow-up venue.

---

## 1 · What changed since 2026-04-21

### 1.1 New head: `next_getnext` (trajectory-flow graph prior)

Adapted from Yang et al., SIGIR 2022. Final logits = `STAN(x) + α · softmax(probe(x[:,-1,:])) @ log_T`, where `log_T` is a pre-computed region-transition log-probability matrix (Laplace-smoothed, per state) and `probe` is a learnable linear `[embed_dim → n_regions]` classifier. α is a learnable scalar initialised to 0.1.

| State | MTL GRU (prior baseline) | MTL GETNext d=256 | Δ Acc@10 | Markov-1 floor | Above floor? |
|-------|-------------------------:|-------------------:|---------:|---------------:|:------------:|
| AL 5f | 45.09 ± 5.37 | **56.49 ± 4.25** | **+11.40 pp** | 47.01 | ✅ +9.48 |
| AZ 5f | 41.07 ± 3.46 | **46.66 ± 3.62** | **+5.59 pp** | 42.96 | ✅ +3.70 |
| FL 1f | 57.60 | **60.62** (n=1) | +3.02 pp | 65.05 | ❌ −4.43 |

Cat F1 unchanged within σ on both states (no trade-off). σ on AL halves (10.09 → 4.25 vs STAN d=256). **On its face, this is the largest single-intervention MTL region lift in the study.**

### 1.2 Attribution test (2026-04-22, memory-archived)

`mtl-loss = static_weight` vs `pcgrad` on AL/AZ with GETNext head, everything else fixed:

| State | Static | PCGrad | Δ | Verdict |
|-------|-------:|-------:|---:|---------|
| AL 5f | 56.21 ± 3.91 | 56.38 ± 4.11 | −0.17 pp | within σ |
| AZ 5f | 47.20 ± 2.55 | 47.34 ± 2.93 | −0.14 pp | within σ |

**Good result for the story's internal coherence:** PCGrad is not doing the work; the lift is head-driven. **But this test only narrows the attribution — it does not confirm it is the graph prior** rather than some other consequence of the STAN+probe head class (§1.3 below).

### 1.3 B5 probe-entropy analysis (2026-04-22) — red flag on the mechanism

With MTL-GETNext checkpoints (AL 1f, AZ 1f, epoch 8–9 of 50):

| State | Top-1 mean | Entropy | % of uniform H | Unique argmax used |
|-------|-----------:|--------:|---------------:|--------------------:|
| AL (1109 regions) | **5.3 %** | 5.76 | 82 % | 88 / 1109 (8 %) |
| AZ (1540 regions) | **12.3 %** | 4.93 | 67 % | 103 / 1540 (7 %) |

- The soft probe is **diffuse**, not one-hot. On AL, five regions take 60 % of the argmax mass; on AZ, five regions take 59 %.
- When `probe` is diffuse, `p @ log_T` **degenerates toward `mean(log_T, axis=0)` — a marginalised frequency prior, NOT a transition-conditional prior**. The functional behaviour is close to adding a "popular-next-region" bias to the logits.
- This is directly consistent with the FL metric-divergence pattern (GETNext lifts Acc@10 by +3 pp but **regresses Acc@1 by −3.3 pp, Acc@5 by −13.5 pp, MRR by −3 pp**): a frequency prior improves top-k coverage but does not sharpen the top-1/5 ranking, and at 4703 regions it spreads probability across many candidates.

**The critical implication for the paper:** the `+11 pp AL / +5.6 pp AZ` lift is likely *not* attributable to GETNext's graph-prior mechanism. The currently-written headline "The graph prior sidesteps shared-backbone dilution by re-injecting closed-form transition signal directly to the output logits" (`GETNEXT_FINDINGS.md §Paper-level positioning`) is **not yet supported by the evidence** — the transition prior is effectively a marginal popularity prior in practice.

### 1.4 Self-audit items from `MODEL_DESIGN_REVIEW_2026-04-22.md` (authored 2026-04-22)

Most salient items for this review:

- **#1 — GETNext probe has no auxiliary supervision.** The author's own review predicts: "the GETNext lift you observed comes primarily from the STAN backbone, not from the transition-flow prior." Proposed test: load the trained GETNext checkpoint and force `α = 0`. **If Acc@10 drops ≤ 1 pp, the prior is not doing material work; retract the GETNext story and rename the head `next_stan_v2`.** This ablation has not yet been run.
- **#3 — DSelectK is not sparse.** The paper claim "DSelectK's sparse routing dominates" is mechanistically wrong; the implementation is a multi-softmax convex combination over all experts (no top-k, no sparsity). The *ranking* holds ("DSelectK leaderboard position is genuine") but the *mechanism* is not sparsity.
- **#4 — STAN `pair_bias` unregularised.** ALiBi init should become the default (already confirmed to reduce σ on AZ; see `STAN_FOLLOWUPS_FINDINGS.md` and B7 below).
- **#5 — AdaShare has no temperature annealing / sparsity pressure.** Even after the partition bug is fixed, AdaShare will most likely still behave like always-on and land neutral.
- **#6 — DSelectK + MTLoRA + α-skip summed without single-term isolation.** The "MTLoRA lift" cannot be attributed to LoRA vs α-skip vs both.
- **#7 — `shared_parameters()` / `task_specific_parameters()` use substring matching.** Exactly the class of design that produced the partition bug below.

### 1.5 B7 — ALiBi × GETNext on AL (2026-04-22)

`Acc@10 = 56.38 ± 4.11` (baseline GETNext) → `57.46 ± 3.66` (+ ALiBi). Δ = +1.08 pp mean, σ −11 %. Within σ. Author's own decision: "Keep as optional paper artefact, not default" — correct call.

### 1.6 Bug 1 — `MTL_PARAM_PARTITION_BUG.md` (HIGH severity, 2026-04-22)

`PCGrad`, `CAGrad`, and `Aligned-MTL` all set `p.grad = g` only for params inside `shared_parameters ∪ task_specific_parameters`. Anything outside that union has `.grad = None` after `optimizer.zero_grad(set_to_none=True)` and AdamW leaves it untouched.

Two parameter groups sit outside the union:

| Class | Param | Starting value | Behaviour under PCGrad/CAGrad/Aligned-MTL |
|-------|-------|---------------:|------------------------------------------|
| `MTLnet` | `adashare_logits` | 2.0 (→ sigmoid = 0.88) | Never trains → gates stay ≈ 0.88 → behaves like always-on FiLM baseline |
| `MTLnetDSelectK` | `lora_A_cat`, `lora_B_cat`, `lora_A_next`, `lora_B_next`, `skip_alpha_cat`, `skip_alpha_next` | 0 (LoRA B + α init) | Never trains → LoRA branch contributes 0, α-skip contributes 0 |

Contaminated runs (per `MTL_PARAM_PARTITION_BUG.md §Re-run triage`):

| # | File | Config | Claim affected |
|---|------|--------|----------------|
| 1 | `ablation_04_mtlora_r8_al_5f50ep.json` | dselectk + pcgrad + MTLoRA r=8 AL | **B11 / A7** — "best MTL reg Acc@10 = 50.72" |
| 2 | `ablation_04_mtlora_r16_al_5f50ep.json` | MTLoRA r=16 AL | rank-sweep |
| 3 | `ablation_04_mtlora_r32_al_5f50ep.json` | MTLoRA r=32 AL | rank-sweep |
| 4 | `ablation_05_adashare_mtlnet_al_5f50ep.json` | adashare + pcgrad AL | "AdaShare NEUTRAL" |
| 5 | `az2_mtlora_r8_fairlr_5f50ep.json` | MTLoRA r=8 AZ | AZ replication |
| 6 | `rerun_R4_mtlora_r8_fairlr_al_5f50ep.json` | MTLoRA r=8 AL fair-LR | leaderboard |

The author's own assessment: "MTLoRA r=8 gives +1.84 pp over DSelectK+PCGrad" will **likely evaporate into noise**; "AdaShare NEUTRAL" was a silent no-op, not a real result.

The champion (cross-attn + pcgrad, A-M3 / B-M4) is **not** in the contaminated list — cross-attn has no LoRA / AdaShare and its partition covers every submodule. All P8 MTL-STAN / MTL-GETNext runs are also safe. The headline is preserved; the secondary MTLoRA narrative is not.

### 1.7 Bug 2 — `CROSSATTN_PARTIAL_FORWARD_CRASH.md` (MEDIUM severity, 2026-04-22)

`MTLnetCrossAttn` overrides `.forward` but not `cat_forward` / `next_forward`. The inherited implementations reference `self.film` and `self.shared_layers`, neither of which exist on the cross-attn subclass. `scripts/evaluate.py` + the partial-forward tests crash with `AttributeError` the moment they try a single-head eval on a cross-attn checkpoint.

Training metrics (and reported headline numbers) are **not** contaminated — training goes through `.forward` which is correctly overridden. But any per-head breakdown the paper may want for the Appendix cannot currently be produced from the checkpoints.

### 1.8 FL still below Markov-1 on the headline task

Even MTL-GETNext FL (1f) = 60.62 vs Markov-1-region 65.05 = **−4.43 pp**. Every MTL config tested on FL remains below the classical 1-gram closed-form baseline on the region task. The scale-curve narrative (region gap widens with class count) continues to hold; the hoped-for mechanism break ("graph prior rescues region") does not reach FL scale in the evidence currently on disk.

---

## 2 · Revised status of the claims catalogue

| Claim | Status 2026-04-21 | Status 2026-04-22 | Delta |
|-------|:----------------:|:-----------------:|-------|
| **CH16** — Check2HGI > HGI on cat F1 | confirmed (+18.30 pp, AL only) | **unchanged** | substrate claim intact |
| **CH17** — Check2HGI > POI-RGNN + prior HGI article | pending + protocol audit flag | **unchanged** | audit still required |
| **CH-M4** — Cross-attn closes cat gap | locked AL, n=1 AZ/FL | **unchanged** | cross-attn is not contaminated |
| **CH-M5** — Fair LR dominates architecture | locked | **unchanged** | |
| **CH-M6** — Scale curve | 3 single-seed data points | **unchanged** | |
| **CH-M7** — Markov-k monotone degrade | locked | **unchanged** | |
| **CH-M8** — Cat→reg transfer scale-dependent | locked at 1f FL | **unchanged** | |
| "MTLoRA r=8 is best MTL reg config" (B11) | confirmed 50.72 ± 4.36 | **RETRACTED** pending re-run; +1.84 pp vs DSelectK likely seed noise | partition bug (§1.6) |
| "AdaShare NEUTRAL" | confirmed (−0.31 pp) | **UNDETERMINED** — previous test was a silent no-op | partition bug (§1.6) |
| **(new) MTL-GETNext lift** — +11/+5.6 pp AL/AZ | — | **PROVISIONAL** pending α=0 ablation | probe-entropy (§1.3) + design review #1 |
| **(new) TGSTAN ≈ GETNext on AZ** | — | recorded; same attribution concern | same probe issue in both heads |
| **(new) PCGrad ≈ static with GETNext** | — | confirmed (Δ within σ) | narrows attribution, does not confirm graph prior |
| **(new) B7 ALiBi × GETNext** | — | +1 pp mean / −11 % σ within σ | correctly kept as optional |

---

## 3 · BRACIS lens — revised verdict

### 3.1 Submitted-on-2026-04-20 scenario (most likely)

If the paper went out on the deadline with cross-attn + pcgrad as the MTL headline:

- **Preserved in the submitted evidence:** CH16 (substrate AL), CH-M4 (cross-attn cat gap-closing), CH-M7 (Markov monotone-degrade), CH-M5 (fair LR dominates), CH-M6 (scale curve — weak on n=1 FL), λ=0 decomposition methodology.
- **Likely invalidated by the 2026-04-22 audit but possibly already in the submitted paper:**
  - The MTLoRA narrative (B11 = 50.72 was a headline number in `BASELINES_AND_BEST_MTL.md §Task B AL`). If the paper published this number, the **partition bug forces a correction** in camera-ready or via an erratum. Mitigating detail: the *real* best MTL reg on AL was cross-attn λ=0.5 at 50.26 (B12), within σ of the contaminated 50.72. Substantively the claim "MTL reg tops out around 50 % on AL" survives; the attribution to MTLoRA does not.
  - The "AdaShare NEUTRAL" sentence, if it made it into the paper, is unsupported.
- **Not in the paper (post-deadline):** MTL-GETNext, MTL-TGSTAN, MTL-STA-Hyper, all attribution tests, all α inspection, B5 / B7 findings.

### 3.2 Missed-deadline scenario

If the paper slipped, the study is **better positioned for a follow-up venue** (IJCNN, SBC-CSBC, SIGSPATIAL-short) but **worse positioned than it thinks** because:

- The new MTL-GETNext headline is under active audit and may retract.
- Without the α=0 ablation and hard-index comparison, MTL-GETNext cannot enter a paper as a *method* claim. It can enter only as an empirical observation with the probe-entropy caveat.
- The partition bug's 6 re-runs are ~3–4 h of compute, achievable, but change the shape of the ablation table.

### 3.3 What a reviewer will ask (in decreasing order of likelihood)

1. "You claim a +11 pp lift from a trajectory-flow graph prior. Did you run the α=0 ablation? What does it show?" → this review's strongest recommendation is to **run this before the MTL-GETNext story is written up**.
2. "You reported an MTL-region number of 50.72 using MTLoRA r=8. Your audit found that LoRA parameters received no gradients under PCGrad. What is the corrected number?"
3. "Your MTL region is below the Markov-1 floor on FL. Why should practitioners use this pipeline over the closed-form Markov baseline?" → requires the graph-prior mechanism to hold; currently it does not hold cleanly.
4. "Your scale curve uses FL at n=1 fold. Please report FL 5-fold." → same as 2026-04-21 critique, still unresolved.
5. "POI-RGNN reproduced at 34.49 % FL cat F1, yours is 63.17 %. Please describe the evaluation protocol differences." → **protocol audit flag from the prior review remains open**.

---

## 4 · What is the tightest defensible BRACIS narrative as of 2026-04-22

Updated minimal-story list (delta from 2026-04-21 version in **bold**):

1. **CH16** — Check2HGI > HGI on next-cat F1 (AL +18.30 pp). Still requires ≥ 1 additional state for reviewer comfort. FL HGI STL is 3–5 h of compute.
2. **CH-M4** — Cross-attn matches STL cat on AL, lifts it on AZ (+1.05 pp marginal), on FL 1f (+3.29 pp, **no σ**). Cross-attn is *not* contaminated by the partition bug; this claim survives the audit.
3. **CH-M7** — Markov-k monotonically degrades with k; neural over-Markov-9 gap isolates representation-learning. Small, clean, unaffected by any audit finding.
4. **λ=0 decomposition methodology.** Cross-attn λ=0 runs are safe; the tool-level contribution is intact.
5. **(new, provisional)** MTL region above the Markov floor on AZ *via a head-level graph-prior bypass* — **conditional on the α=0 ablation showing ≥ 2 pp drop from GETNext → pure STAN**. If the ablation shows ≤ 1 pp drop, rename to `next_stan_v2` and drop the graph-prior framing.

**Demoted since 2026-04-21:**
- Any MTLoRA claim. **Retracted pending re-run.**
- "AdaShare NEUTRAL" as a characterised finding. **Retracted as undetermined.**
- "DSelectK's sparsity wins." **Mechanism claim wrong**; ranking may hold but sparsity is not what drives it.

**Unchanged demotions from 2026-04-21:**
- CH01 / CH02 (bidirectional MTL lift) — retired.
- CH03 per-task modality as tier-A — still only 1f × 20ep AL exploratory; move to appendix.
- CH17 as headline — still needs POI-RGNN protocol audit.

---

## 5 · Recommended sequencing before any camera-ready or resubmission

Ordered by attribution impact × cost:

1. **[BLOCKER — 1 h] α=0 ablation on MTL-GETNext checkpoints (AL + AZ).** Directly adjudicates whether the graph prior is doing work or is a frequency-bias artefact. If ≤ 1 pp drop, the head is effectively STAN; the paper narrative reverts to STAN at matched compute.
2. **[BLOCKER — 3–4 h] Apply `MTL_PARAM_PARTITION_BUG` fixes + re-run the 6 contaminated JSONs.** Until this is done, any claim citing `B11`, `A7`, `B-M2`, or `AdaShare NEUTRAL` is unpublishable.
3. **[BLOCKER — 5 min] Apply `CROSSATTN_PARTIAL_FORWARD_CRASH` fix.** Required before per-head Appendix tables can be produced from cross-attn checkpoints.
4. **[HIGH — 4–6 h] B5 hard-index probe.** If the soft probe is diffuse (it is — 5–12 % top-1), the hard-index variant is a fair scientific test of whether transition-conditional (not frequency) priors help. This is the paper-worthy version of the GETNext claim.
5. **[HIGH — 3–5 h each] FL 5-fold MTL cross-attn + pcgrad + FL 5-fold STL cat.** Still the single largest missing evidence item for the BRACIS-style headline.
6. **[MEDIUM — 3–5 h] AZ HGI STL cat.** Extends CH16 from n=1 state to n=2 states. Cheap.
7. **[MEDIUM — ~30 min] POI-RGNN evaluation protocol audit.** Verify taxonomy / window / folds match before the +28 pp claim is repeated.
8. **[LOW — multi-seed on the FL champion, ~15 h]** If time allows, addresses the single-seed critique.

Items 1–3 should land before any public statement is made about MTL-GETNext.

---

## 6 · Preserved strengths (unchanged from prior review)

- **User-disjoint fold bugfix (C11)** — exactly the methodological care that earns reviewer trust.
- **Null-finding discipline.** Nash-MTL ≠ PCGrad, loss-balancing null, hybrid null, ALiBi-AL null, B5 attribution test (static ≈ pcgrad). Above the typical BRACIS bar.
- **λ=0 decomposition methodology.** Independent of whether the numbers favour MTL.
- **Fair-LR investigation (C12).** Identifying the LR confound and rerunning the grid was the correct engineering call.
- **Honest bookkeeping.** CONCERNS.md + issues/ + research/ all maintained with dates and status; this review would not be possible without it.
- **The 2026-04-22 self-audit itself.** Catching `MTL_PARAM_PARTITION_BUG` 2 days past the deadline, on the same day as the attribution probe-entropy analysis, is the kind of rigour most graduate studies don't apply. The findings hurt the headline; surfacing them does not.

---

## 7 · Bottom line

As of 2026-04-22, the study has simultaneously produced (a) its strongest candidate MTL-region headline to date (MTL-GETNext) and (b) credible internal evidence that the mechanism the headline ascribes to that lift is **not the actual driver**. The correct next action is not "write up MTL-GETNext"; it is "run the α=0 ablation and the hard-index probe, and let the ablation decide the framing". This is a 4–6 h compute investment for a qualitative change in what can be claimed.

The cross-attention headline (CH-M4, +3.29 pp FL cat, n=1) remains intact, uncontaminated by the partition bug, and is the safest paper foundation. CH16 (Check2HGI > HGI) remains the best-supported single-number finding and deserves the simple cross-state replication it has not yet received.

The MTLoRA-based numbers in `BASELINES_AND_BEST_MTL.md` and `RESULTS_TABLE.md` **must** be regenerated or annotated `(SUPERSEDED: MTL_PARAM_PARTITION_BUG)` before the tables appear in a paper. If the submitted BRACIS paper contains the `B11 = 50.72` number, an erratum or camera-ready correction is appropriate.

---

## References within the repo

- Prior review conversation (2026-04-21) — parent artefact of this document.
- `docs/studies/check2hgi/research/GETNEXT_FINDINGS.md`
- `docs/studies/check2hgi/research/B5_PROBE_ENTROPY_FINDINGS.md`
- `docs/studies/check2hgi/research/B7_ALIBI_GETNEXT_FINDINGS.md`
- `docs/studies/check2hgi/issues/MODEL_DESIGN_REVIEW_2026-04-22.md`
- `docs/studies/check2hgi/issues/MTL_PARAM_PARTITION_BUG.md`
- `docs/studies/check2hgi/issues/CROSSATTN_PARTIAL_FORWARD_CRASH.md`
- `docs/studies/check2hgi/results/BASELINES_AND_BEST_MTL.md`
- `docs/studies/check2hgi/results/RESULTS_TABLE.md`
- `docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md`
- `docs/studies/check2hgi/CONCERNS.md`
- `docs/BRACIS_GUIDE.md`
- Auto-memory note: `attribution_pcgrad_vs_static_2026-04-22.md`
