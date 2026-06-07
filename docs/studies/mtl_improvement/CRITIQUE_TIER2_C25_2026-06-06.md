# Critical analysis — the C25 unweighting finding + champion G (2026-06-06)

**Scope.** Cold, skeptical review of the Tier-2 reversal: the C25 class-weighting confound, the multi-seed
re-validation, and champion **G** (`mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower`, `aux` fusion,
prior-OFF, v14, unweighted onecycle). Method: read CONCERNS §C25, log 2026-06-05/06, the re-validation +
G multi-state tables, code (`mtl_cv.py`, `METRICS.md`, the c25 drivers), + advisor. **Written ahead of the
planned hypertuning-the-winner phase** — the forward levers (§4) are scoped for it.

**One-line verdict.** The finding is **sound and well-verified** — do not re-doubt the core. The headline
splits cleanly: the **cat gain is robust and conservative** (near-bulletproof); the **reg "beats the STL
ceiling" claim is fragile** and currently gated on three cheap checks. Nothing here is fatal; all of it
fits the hypertuning phase.

---

## 1. What is SOUND (affirm — don't manufacture doubt)
- **The C25 confound is real and dominant.** MTL reg trained on class-WEIGHTED CE while the metric
  (`top10_acc_indist`, frequency-weighted) + the STL ceiling are unweighted → class-balancing optimizes
  *macro*, away from top-K → ~10-14pp reg depression, scaling with class count. Verified the right way:
  deletion-bisect, a re-audit that found no second equal confound, the wd secondary *controlled* (not the
  cause), and — crucially — the **cat-axis was TESTED not assumed** (unweighted cat empirically +5pp, the
  "balancing helps macro-F1" prior falsified).
- **The narrative flip is multi-seed.** Regime finding overturned (Δreg v14−canon +1.92/+1.49/+0.81, σ~0.1),
  composite advantage dissolved, §0.1 MTL reg +10-13pp — all {0,1,7,100}.
- **The architecture-capacity hypothesis is falsified 5 independent ways** (MoE, SwiGLU, MulT,
  crossstitch→crossattn, more-blocks). Thorough; the dual-tower-as-the-arch conclusion is well-isolated.

## 2. The split that should frame the paper: robust cat, fragile reg
- **ROBUST — the cat gain.** G cat beats the STL cat ceiling by **+2.9 to +3.5pp at all 4 states, 4 seeds,
  tight σ**. And it is **conservative**: G's cat head is **plain unweighted CE** (verified — no
  `--logit-adjust` in any G driver), while the (c) cat ceiling used **logit-adjust τ=0.5**. G wins using a
  *weaker* macro-F1 loss than its bar → the cat result is the strong, hard-to-attack headline. Lead with it.
- **FRAGILE — the reg "beat."** Three reasons it is softer than the banner ("beats both STL ceilings"):

## 3. Gates on the reg-"beats-ceiling" headline (cheap; do before the paper claim)

**G1 — the (c)/(d) ceilings are SINGLE-SEED (seed 42); G is seeds {0,1,7,100} — DISJOINT seeds. THE #1
must-fix.** This is not "4-seed vs 1-of-4," it is two different seed sets. Variance cuts both ways: FL
margin is **+0.26** (tiny, though σ_G=0.06); the small-state margins look bigger (+1.59 AL) **but
small-state reg fold-σ is 3-4pp**, so a seed-42 ceiling could swing several pp. So *neither* end of "beats
both reg ceilings" is currently defensible. **Re-run (c)/(d) at {0,1,7,100}** (cheap, belongs in the
hypertuning phase) — it is the bar the whole headline is measured against, **not optional**. Until then the
honest, still-strong reg claim is: **"MTL no longer sacrifices reg — a single model MATCHES its STL ceiling
and the composite,"** which pairs with the next point.

**G3 — frame the reg result as RECOVERY, not a magical beat.** G's reg path *is* the STL reg architecture
(a private STAN on raw region) as its private tower; the +0.26 over the ceiling is the small `aux` boost.
So "joint training beats single-task on reg" over-reads it — the precise claim is "the private tower
*recovers* the STL ceiling inside the joint model (the C25 confound was what suppressed it), and cat rides
a genuine rising tide." That is a Pareto-non-inferior reg + a real cat gain — strong without over-claiming.

**G2 — report G on a TAIL/MACRO reg slice + the FULL (non-indist) metric vs prior-ON (the within-dist tail
check, NOT cold-start).** G stacks two head-favouring choices — **unweighted CE + prior-OFF (α=0)** — both
of which optimize the frequency-weighted, head-dominated `Acc@10`, the *exact* axis C25 just showed can
trade against the tail. So the real question is symmetric to the confound we just fixed: **is the champion
winning the head at the tail's expense?** Check = G's reg on a macro/tail-region slice + the full
(`top10_acc`, not `_indist`) metric, vs the prior-ON config. (NB: the cold-start/OOD framing is the *weak*
version — `_indist` excludes unseen regions for both arms, so nothing is differentially hidden there, a
train-Markov prior can't help unseen regions, and the "α=0→0.03% collapse" was measured under the OLD
class-weighted regime so it is likely itself a C25 casualty, not evidence the prior protects OOD. Use the
within-distribution tail framing, which ties directly to the C25 mechanism and can't be waved off as
"indist is our protocol.")

**Plus two the banner glosses:**
- **G is NOT "free vs 2 models."** "Composite strictly dominated, 2× params" is wrong on cost: G *adds a
  full private STAN tower* on top of base_a, so it sits *between* base_a and the composite on params/FLOPs.
  The single-model-efficiency claim needs **G's actual param/latency count**, not "1 vs 2."
- **The gold-standard checkpoint re-eval was DEFERRED** (log:883). The entire flip rests on the training
  harness's own eval. For a result that overturns the paper's narrative, **load the G checkpoint and
  re-score with an independent forward once** — it forecloses "it's another harness artifact" (the study
  has already had several harness/measurement artifacts: C25 itself, the Acc@1 monitor, 2 retracted
  hypotheses). Cheap insurance, high value.
- Minor: `base_a` / `dual_gated` in the dose-response are single-seed; the "G > base_a +2pp" arch-contribution
  claim should use a multi-seed `base_a`.

## 4. What was missing / unexplored + arch improvements (for the hypertuning phase)
1. **⭐ logit-adjust on G's cat head — highest-EV cat lever, currently OPEN.** The cat ceiling shows
   logit-adjust τ=0.5 helps macro-F1; G's cat doesn't use it (plain unweighted). Adding it to G's cat is
   untested and likely pushes cat further (and explains the unexplained "plain unweighted cat wins" puzzle —
   plain unweighted may not even be the cat optimum, just better than *weighted*). The c25_gv2 sweep never
   tried it.
2. **CA/TX — the scale-conditional gap.** The confound scaled with class count → CA/TX (8.5k/6.5k regions)
   are where the fix's benefit is predicted *largest*, and they're missing (need a v14 build). For the
   "scales with class count" / scale-conditional story specifically, this is the real completeness gap (not
   a blocker for the 4-state result).
3. **G's sweep was 1-seed and narrow** (c25_gv2, seed-0) — "G is locked" is over-stated. Untested headroom
   for hypertuning: aux-β init/learnability + fusion granularity (currently β init 0.1); private-tower
   `d_model`/depth (currently the STL default 128/4-head); per-task LR for the private tower; and a
   **per-task precision split** — the global fp32 toggle gave +0.13 reg / −1.11 cat (a trade); fp32-reg +
   fp16-cat might capture the reg gain without the cat loss.
4. **The "unweighted cat wins" mechanism is unexplained** (why does plain CE beat weighted CE on macro-F1?).
   Tied to #1 — resolve it by testing the loss family (unweighted vs weighted vs logit-adjust vs focal) on
   the cat head under joint training; pick the cat optimum rather than "unweighted because weighted was bad."

## 5. Honest paper framing (what to claim now vs after the gates)
- **Claim now (robust):** "Once the reg loss matches the reported metric (the C25 fix), joint MTL is
  Pareto-non-inferior to single-task — a single model **matches** the STL reg ceiling/composite AND
  **substantially beats** the STL cat ceiling (+3pp, all states, conservative loss). The previously-reported
  −7..−17pp 'MTL sacrifices reg' tension was an objective-mismatch artifact."
- **Upgrade to "beats both ceilings" ONLY after G1** (multi-seed (c)/(d)) + **G2** (tail/full-metric shows no
  tail regression) + **the checkpoint re-eval**. With those, the inverted-tradeoff headline is paper-grade.
- Account G's params honestly (§3); add CA/TX for the scale-conditional claim (§4.2).

**Bottom line:** excellent, sound work that genuinely flips the study to a positive result. The cat gain is
already paper-grade and conservative. The reg "beat" needs three cheap checks (multi-seed ceilings = the
big one; a tail/full-metric report; one checkpoint re-eval) before it is more than "matches." The
hypertuning phase has real headroom on the cat side (logit-adjust) and a genuine completeness gap at CA/TX.

---

## 6. Final thoughts (2026-06-06) — the G *architecture*: heaviness, alternatives, optimizer confound (user considerations; advisor + lit-scanned)

### 6.0 What G's architecture actually is (from code — corrects a common misread)
The **cat** head reads ONLY the shared cross-attn output (`category_poi` = `next_gru` on `shared_cat`) — **cat
does NOT pass through any STAN.** Only **reg** has the dual STAN: a **private** STAN on the raw `[B,9,64]`
region sequence **in parallel** to the shared cross-attn pathway (NOT "before" it), fused at the reg head by
`aux`: `feat = priv + β·aux_proj(shared)`, β learnable. So G's overhead vs `base_a` is **one extra STAN tower,
reg-only** — moderate, not 2×; the cat path is untouched.

### 6.1 Is a per-task private backbone a good MTL pattern? (point 1 — lit-grounded, for the docs)
Yes as a *family*: selective/soft parameter sharing — Cross-Stitch (Misra CVPR'16), Sluice (Ruder'17), MMoE
(Ma KDD'18), **PLE/CGC (Tang RecSys'20)**, branched-MTL (Vandenhende arXiv:1904.02920), task grouping
(Standley ICML'20). G ≈ a **degenerate CGC (1 shared + 1 reg-private "expert") with a *static* `priv + β·shared`
fusion** instead of CGC's *learned input-dependent gate*. The asymmetry (only the negative-transfer task gets a
private branch) is principled but unnamed. **Lit caveat:** a full-duplicate backbone is the *expensive* end of
the family; the field's thrust (CGC, branched-MTL, adapters) is task-specific capacity bought *more cheaply*.
- **Is the static fusion a weakness? Tested here — NO.** The head's `gated` mode (a learned per-dim gate) was
  run and **LOST to `aux`** (FL seed-0: gated 73.00 < aux 73.57) — gating makes priv/shared *compete*, `aux`
  adds shared as a non-attenuating residual. So G's fusion is locally validated; the lit "static < learned-gate"
  worry doesn't bite. (A *faithful* CGC — shared+task experts, softmax gate over experts — is a different,
  untested object; see 6.6.)

### 6.2 Alternatives to the STAN private tower (point 1) — UNTESTED, an *efficiency* question
"Must the private tower be a full STAN?" is **open in the literature** (STAN > RNN is established; "nothing
lighter matches STAN as a private MTL tower" is not). The private tower carries the reg signal (`private_only`
already clears the ceiling) → it is load-bearing, but a **GRU/TCN or shallow-attention private tower of matched
budget** is un-run. Frame it as **"is the full-STAN duplicate over-provisioned?"** (the §3 param-cost critique),
NOT as accuracy headroom.

### 6.3 Why not a cat private tower? (point 1) — principled; a confirm-only ablation
Cat is **positive-transfer**: MTL cat *exceeds* its STL ceiling (+3pp) *because* it shares (rising tide). A
private cat tower isolates cat from that → **predicted: helps nothing, costs params.** One cheap ablation to
*confirm the asymmetry is principled* (reg needs private, cat wants shared) — not to find a win. This asymmetry
is the cleanest mechanistic story in the paper; confirming it is high-value for the *narrative*, low for the metric.

> ✅ **RESOLVED 2026-06-06 (T2V.4).** The standalone alt-archs were re-ranked POST-C25, each at its OWN best `category-weight` {0.5,0.65,0.75}, FL 1-seed: hardshare 71.45 / crossstitch 71.94 / mmoe 71.69 / cgc 71.69 — **all lose to G (73.57) by 1.6–2.1pp and none reaches the (c) ceiling (73.31)**, with category-weight barely moving reg. Run STANDALONE (not under the dual-tower) → the §6.4 "the 5 nulls are gate-suppressed shared-pathway swaps" objection no longer applies. The falsification is now FAIR + un-confounded → paper-safe. No -lite surprised → the faithful CGC build is NOT triggered. See INDEX `#T2V-4`.

### 6.4 Other backbones (point 2) — the falsifications are doubly-confounded
Post-C25, only the **cross-attn family** ran (base, dual-tower, SwiGLU/MulT/xstitch *under* it). Standalone
**hard-share / CrossStitch / MMoE / CGC** were ranked **PRE-C25** — under the class-weighting confound that
*inverted* the dual-tower's own ranking (worst→best) — and at a **single un-swept loss weight**; never re-ranked
post-fix. So "architecture-capacity falsified 5 ways / not-architecture" is **about the alternatives and is
under-powered** (the 5 nulls are shared-pathway swaps the dual-tower gate suppresses by design → expected nulls,
saying little about standalone alternatives).

### 6.5 The optimizer (point 3) — sharpest; scoped precisely
**It is a confound on the FALSIFICATIONS, not on G.** Every Tier-2 run used `--mtl-loss static_weight` (no
gradient-balancer), and `category-weight` was swept **only for G** (→0.75), never for the alternatives. Lit is
direct — **Xin et al. NeurIPS'22**: per-arch loss-weight tuning dominates, the HP effect dwarfs the
MTL-optimizer effect, and comparing architectures under one fixed scalarization can manufacture illusory
rankings. Therefore:
- **G stands** — validated at its own loss-weight optimum; a balancer/more tuning only helps it. Its win is NOT
  suspect.
- **The alternative-arch negatives ARE suspect** (pre-C25 + un-swept weight). **Do not cite hard-share/
  CrossStitch/MMoE as "closed/falsified."**
- **Timing (answers "now or before"):** per-arch **loss-weight** = a *ranking-validity prerequisite* → tune it
  for any arch you want to falsify (with the re-rank). The gradient **balancer** (FAMO, O(1), the feasible one
  at k=2/9k) = *winner-polish* → on G, in hypertuning. (FAMO/Aligned-MTL were tried pre-C25 on the base arch and
  did NOT unpin reg-best, `MTL_FLAWS §304` → low expected headroom on G, but a confirmatory pass is cheap.)

### 6.6 The ONE dominant follow-up (bounded — not a six-experiment program)
**Re-rank the strongest alternative architecture POST-C25, each with its own `category-weight` sweep, vs G.**
This addresses points 1-generalization, 2, and most of 3 at once. The principled comparand is a **faithful
learned-gate CGC (1 shared + 1 reg-private expert)** — G's named generalization — but the registry MMoE/CGC/PLE
are **"-lite"** (non-canonical, `MTL_FLAWS §4.1`), so a faithful CGC is a *small build*, not a flag. Bounded
path: (1) re-run the existing **-lite MMoE/CGC + standalone hard-share + CrossStitch**, 1-seed FL, each with a
3-point `category-weight` sweep, unweighted/post-C25; (2) promote any that ties/beats G to multi-seed; (3) build
the faithful CGC **only if a -lite surprises**. Everything else (6.2 lighter tower, 6.3 cat-tower, 6.5 FAMO-on-G)
are cheap ablations / winner-polish, **not** a parallel program. Resist breadth.

### 6.7 Net
G is a **validated positive — leave it standing, hypertune it (don't re-validate it).** The genuinely open work
the user's instincts surface: (a) **un-confound the alternative-arch falsifications** (per-arch weight, post-C25
— so the paper can honestly say "we compared architectures fairly," not "we falsified them under the bug + one
weight"); (b) **winner hypertuning on G** — logit-adjust on the cat head (§4.1, highest-EV), per-arch weight + a
FAMO confirmatory pass, the lighter-tower efficiency ablation, per-task precision; (c) the cat-private-tower
confirm-ablation for the *narrative*. Deepest paper risk stays §3 (multi-seed ceilings + tail/full-metric);
deepest *architecture* risk is 6.4/6.5 — citing under-tuned negatives as closed.

**Lit refs (for the docs):** Cross-Stitch (Misra CVPR'16); Sluice (Ruder arXiv:1706.05098); MMoE (Ma KDD'18);
PLE/CGC (Tang RecSys'20, DOI 10.1145/3383313.3412236); branched-MTL (arXiv:1904.02920); task grouping (Standley
ICML'20, arXiv:1905.07553); **Xin "Do MTO methods even help?" NeurIPS'22 arXiv:2209.11379**; Kurin "Unitary
Scalarization" NeurIPS'22 arXiv:2201.04122; **FAMO (Liu NeurIPS'23 arXiv:2306.03792, O(1))**; STAN (Luo WWW'21
arXiv:2102.04095).

---

## 7. Execution TODO (for the A40) — closes this critique → see `INDEX.html #tier2v`
These cards translate §3–§6 into runs. Priority: **P0 = paper-blocking · P1 = the one dominant arch test ·
P2 = winner hypertuning · P3 = completeness.** Recipe baseline = G (`mtlnet_crossattn_dualtower` +
`next_stan_flow_dualtower` aux+prior-OFF, v14, unweighted onecycle KD-OFF). Score Δ vs frozen (c)/(d) + the
composite null; seeded per-fold log_T; frozen-fold paired. **G is a validated positive — VALIDATE/EXTEND it,
do not re-litigate it.** Mark each `[ ]→[x]` + fill the INDEX Tier-2V Results block + a `log.md` entry on land.

> **STATUS 2026-06-07 — CLOSE-OUT (headline-gating concerns) + §8 RESIDUALS (2026-06-06).** P0 (T2V.1/2/3)
> ✅, P1 (T2V.4/5/6) ✅, P2 (T2V.7) ✅ + T2V.8 MOOT, doc close-out ✅. **Verdict: every concern that gates
> headline *correctness* is resolved — G is the validated, paper-safe champion.** The remaining work is
> robustness / efficiency / completeness, enumerated in **§8**: the **lighter GRU/TCN private tower** (§6.2
> efficiency, un-run — **user priority**), the literal **checkpoint re-eval** (licenses the "beats" verb vs
> the safe "matches"), and the cat-private-tower narrative ablation. None blocks the 4-state Pareto-positive
> result. **CA/TX is out of the A40 queue** — pure future-work
> ([`docs/future_works/mtl_improvement_catx_scale_conditional.md`](../../future_works/mtl_improvement_catx_scale_conditional.md)),
> resumed on the final model. Per-card results: INDEX `#tier2v`; chronology: `log.md` 2026-06-06/07;
> **the A40 queue: §8**.

**P0 — validate the headline (do these before any "beats both ceilings" paper claim):**
- [x] **T2V.1 ⭐ multi-seed (c)/(d) ceilings** {0,1,7,100} × AL/AZ/GE/FL (reg next_stan_flow α=0; cat next_gru
      logit-adjust τ=0.5; recompute (d)). Seed-match every G−ceiling Δ. *The #1 gate — the FL margin is +0.26
      vs a single-seed bar.* (§3 G1)
- [x] **T2V.2 G full-metric + tail slice** — re-score G vs prior-ON on `top10_acc` (full, not `_indist`) +
      popularity-binned Acc@10 + macro, FL+AL. Is G winning the head at the tail's expense? (§3 G2)
- [x] **T2V.3 checkpoint re-eval** — independent forward on G's saved checkpoint (forecloses harness artifact,
      `log:883`) + G param/FLOP count vs base_a & composite (fix the "½ params" claim). (§3)

**P1 — the multi-axis exploration (the axes never fairly searched). DISCIPLINE: OFAT around G, each config
gets a FAIR optimizer (per-arch category-weight min), then COMBINE (T2V.8). No axis "falsified" until tuned.**
- [x] **T2V.4 — Axis 1: shared backbones.** (a) backbone *under the dual-tower* {cross-attn[G], MMoE-lite,
      CGC-lite, CrossStitch, hard-share, simple/none} (mainly a cat + aux-reg lever); (b) the same *standalone*
      (is the dual-tower uniquely the best reg mechanism?). Each × category-weight {0.5,0.65,0.75}, post-C25,
      1-seed FL → promote ties to multi-seed. Faithful learned-gate CGC ONLY if a -lite surprises. (§6.4–§6.6)
- [x] **T2V.5 — Axis 2: reg-private HEAD in place of STAN** {next_gru/lstm/tcn_residual/transformer_relpos/
      conv_attn…} matched-budget, each per-arch-tuned — accuracy (beats STAN here?) AND efficiency (is full
      STAN over-provisioned?). + private-tower d_model/depth/aux-β/granularity. + cat-private-tower ablation
      (predicted null → confirms the asymmetry). (§6.2/§6.3)
- [x] **T2V.6 — Axis 3: optimizer / loss-balancer (the user's confound concern).** On G + the top T2V.4/.5
      configs: {static_weight[G] · per-arch category-weight · FAMO (O(1)) · uncertainty-weighting · CAGrad/Nash
      if k=2 tractable} + per-task LR + per-task precision split (fp32-reg/fp16-cat). Does a balancer lift G OR
      re-rank the alternatives (an arch that lost under static_weight ties G under FAMO → falsifications were
      optimization-confounded)? The whole study ran static_weight only. (§6.5)

**P2 — cat loss + synthesis:**
- [x] **T2V.7 — Axis 4: ⭐ logit-adjust on G's cat head** (τ∈{0.5,1.0}; + focal/CB) — highest-EV cat lever; G's
      cat is plain unweighted, the ceiling used logit-adjust. (§4.1)
- [~] **T2V.8 — COMBINE the winning levers** — **MOOT (2026-06-07): no lever won** (T2V.5/6/7 all ≈/<G), nothing to combine. (best backbone × private head × optimizer × cat loss) onto G →
      multi-seed confirm vs G + the T2V.1 ceilings. Watch non-additivity (PLE/MMoE history). (§6.7)

**P3 — completeness:**
- [~] **T2V.9 CA/TX** — DEFERRED 2026-06-07 (user): documented future-work; 4-state result is paper-grade.  — build v14 substrate → champion + ceilings at the two largest (highest-imbalance) states;
      the scale-conditional headline's real gap. (§4.2)

**A40 doc close-out (housekeeping — fill the data + close the stale tiers, user-requested):**
- [x] **Close Tier 2 (topology):** fill INDEX T2.3 (-lite MoE) + T2.4 (SwiGLU/MulT/xstitch) Results blocks with
      the landed C25-stretch data (all NULL on reg); flip the Tier-2 banner + final-decision from "UNDER
      RE-VALIDATION" → RESOLVED post-C25 (orderings flipped; dual-tower G is the winner).
- [x] **Close Tier 2P (joint-loop):** mark MOOT — the "joint loop poisons reg" hypothesis WAS the C25
      class-weighting confound; T2P.0 wd sub-question answered (wd not the cause) + T2P.1/.2/.3 superseded.
      Banner the cards superseded → C25 / Tier 2V; keep for the trail.
- [x] Keep `HANDOFF.md` + this `§7` + INDEX Tier-2V Results blocks in sync as cards land.

---

## 8. Remaining work after the Tier 2V close-out (2026-06-06) — what's missing + the A40 queue

**Context.** Tier 2V cleared every critique concern that gates **headline correctness** (the P0 reg gates,
the P1 dominant arch re-rank, the P2 cat-loss lever). The items below are the **residuals** — robustness,
efficiency, completeness and narrative work that the close-out either *substituted*, judged
*"not motivated"*, or *deferred*. They are split into **(A) closeable by analysis (done in this doc pass,
no GPU)** and **(B) the A40 execution queue (needs GPU)**. **None blocks the 4-state Pareto-positive
headline.** The two worth doing before the paper hardens are **B-A1 (lighter tower — user priority)** and
**B-A2 (checkpoint re-eval — only if the paper says "beats")**.

### (A) Closed in this doc pass — no execution required

- **A-1. HANDOFF navigation de-staled.** `HANDOFF.md` §0b/§0c still read *"NEXT AGENT STARTS HERE —
  T2P.0"* for a tier the top block declares MOOT → a fresh agent reading top-down could be sent to a dead
  experiment. The §0b/§0c headers are now marked **SUPERSEDED** (→ top block / Tier 2V close-out). The
  housekeeping item "keep HANDOFF in sync" is now fully done. **[closed 2026-06-06]**
- **A-2. T2V.3 framing decision (the claim *verb*).** Until the literal checkpoint re-eval lands (B-A2),
  the **paper-safe claim is "matches / Pareto-non-inferior"** — bulletproof: seed-matched G ≥ (c) reg at
  **4/4** states, cat **+2.5…+4.1 pp** on a *conservative* loss (plain CE vs the ceiling's logit-adjust).
  The stronger **"beats both ceilings"** verb rests on **thin FL margins (+0.30 reg / +0.08 vs composite)**
  measured inside the `train.py`/`mtl_cv` harness, and is licensed **only after B-A2**. **Recommendation:
  write "matches" now; upgrade to "beats" iff B-A2 confirms.** **[decision recorded — author's to apply in
  the paper]**

### (B) A40 execution queue (GPU) — priority order

> Common protocol for every card: score Δ vs the **frozen (c)/(d) ceilings + the composite null**; seeded
> per-fold `log_T`; **frozen-fold paired**; sweep **per-arch `category-weight` {0.5,0.65,0.75}** before
> calling any arm "lost" (the C25 lesson); 1-seed FL to screen → promote any tie/beat to multi-seed
> {0,1,7,100}. Baseline = **G** (`mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower`, aux + prior-OFF,
> v14, unweighted onecycle, KD-OFF). FL reference: G reg **73.57** / cat **73.16**; (c) ceiling **73.31**.

**⭐ B-A1 — LIGHTER private reg tower (GRU / TCN / shallow-attention), matched budget. [USER PRIORITY — DO NOT DROP.]**
The §6.2 *efficiency* question is genuinely **un-run**. T2V.5 answered only *"is a **bigger** STAN better?"*
(no — `d_model=256` hurts, `heads=8` ties) — it did **not** answer *"could something **cheaper** match
STAN?"* The private STAN tower is **273,800 params** (G = base_a + 4.9%); the open question is whether a
GRU/TCN/shallow-attention private tower at **matched-or-smaller budget** holds G's reg.
- **Build:** add `next_gru_dualtower` / `next_tcn_dualtower` analogous to `next_stan_flow_dualtower`
  (private-tower head-type as a param; same `aux` fusion, prior-OFF, raw `[B,9,64]` region sequence; shared
  cross-attn backbone unchanged). Size each to **±10 %** of the STAN tower (or run a small budget ladder
  64/96/128). Unit-test gate first (param-partition bijective+exhaustive — the private tower is a new param
  group; same hard-rule-10 check as `next_stan_flow_dualtower`).
- **Run:** FL seed-0 + per-arch `category-weight`, then promote ties to multi-seed.
- **Both outcomes are useful:** (i) a lighter tower **matches** STAN → an efficiency win + a stronger
  "right-sized" story; (ii) it **loses** → *"the full STAN is load-bearing, not over-provisioned"* becomes a
  **tested** claim, not an assumption. Either way the §6.2 critique is genuinely closed.
- Cost: ~small build + ≈ a 9–12-run sweep (a few A40-h).

**B-A2 — T2V.3 literal checkpoint re-eval (licenses the "beats" verb).**
The reproduction already done (73.56 == 73.56) is `train.py`-vs-`train.py` and **cannot catch a
`train.py`-harness inflation**; the cross-harness argument (ceilings come from the independent `p1`
harness) *mitigates* but does not *foreclose* the artifact the critique named (track record: C25, the Acc@1
monitor, 2 retractions). Mitigant on record: fp16 runs **against** G (fp32-G was +0.13 reg), so the beat
survives a known offset — but that is not the literal check.
- **Run:** one **checkpointed** G at FL (drop `--no-checkpoints`) → load the `.pt` → independent forward in a
  clean / `p1`-style eval path → confirm reg@10 within noise of 73.57.
- Cheap insurance; **only needed if the paper uses "beats" rather than "matches"** (see A-2).

**B-A3 — cat-private-tower ablation (narrative-only). [low metric priority, high narrative value]**
Add a private tower to the **cat** head (predicted **null** — cat is positive-transfer; it *exceeds* its STL
ceiling because it shares the rising tide). Confirms the asymmetry **"reg needs private / cat wants shared"**
— the cleanest mechanistic story in the paper. FL seed-0; score cat vs G. ~1–2 runs.

**B-A4 — minor loss / metric completeness (optional).**
- *Cat tail:* focal / class-balanced on G's cat head (only logit-adjust τ was tested in T2V.7 — it lost;
  these likely lose too, but are untested).
- *Reg tail granularity:* popularity-binned Acc@10 + the full (non-`_indist`) `top10_acc` on G vs prior-ON
  (the `accuracy_macro` proxy already showed no tail cost; the bins are finer-grained for §3 G2).

> **Out of scope for this queue: CA/TX (the old T2V.9).** Removed by user — it is **not** A40 work now.
> It lives as pure future-work in [`docs/future_works/mtl_improvement_catx_scale_conditional.md`](../../future_works/mtl_improvement_catx_scale_conditional.md)
> (gates only the "scales with class count" claim; the 4-state result stands without it; resume on the final model).

**A40 ordering:** **B-A1** (user priority) → B-A2 (iff "beats" wanted) → B-A3 → B-A4.
