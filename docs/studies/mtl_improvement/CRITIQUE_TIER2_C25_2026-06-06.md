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

**P0 — validate the headline (do these before any "beats both ceilings" paper claim):**
- [ ] **T2V.1 ⭐ multi-seed (c)/(d) ceilings** {0,1,7,100} × AL/AZ/GE/FL (reg next_stan_flow α=0; cat next_gru
      logit-adjust τ=0.5; recompute (d)). Seed-match every G−ceiling Δ. *The #1 gate — the FL margin is +0.26
      vs a single-seed bar.* (§3 G1)
- [ ] **T2V.2 G full-metric + tail slice** — re-score G vs prior-ON on `top10_acc` (full, not `_indist`) +
      popularity-binned Acc@10 + macro, FL+AL. Is G winning the head at the tail's expense? (§3 G2)
- [ ] **T2V.3 checkpoint re-eval** — independent forward on G's saved checkpoint (forecloses harness artifact,
      `log:883`) + G param/FLOP count vs base_a & composite (fix the "½ params" claim). (§3)

**P1 — un-confound the alternative-arch falsifications (one dominant test):**
- [ ] **T2V.4 fair re-rank** — hard-share / CrossStitch / -lite MMoE+CGC, EACH × category-weight {0.5,0.65,0.75},
      unweighted post-C25, 1-seed FL, vs G → promote ties to multi-seed; build a faithful learned-gate CGC ONLY
      if a -lite surprises. Retires "5-ways-falsified" or re-opens the axis. (§6.4–§6.6)

**P2 — hypertune the champion G:**
- [ ] **T2V.5 ⭐ logit-adjust on G's cat head** (τ∈{0.5,1.0}) — highest-EV cat lever; G's cat is plain
      unweighted, the ceiling used logit-adjust. (§4.1)
- [ ] **T2V.6 G micro-levers** — aux-β init/learnability + granularity; private-tower d_model/depth; per-task
      LR; per-task precision split (fp32-reg/fp16-cat). (§6)
- [ ] **T2V.7 FAMO on G** (confirmatory) + **lighter private tower** (GRU/TCN — efficiency) + **cat-private-tower**
      (predicted null — confirms the asymmetry). (§6.2/§6.3/§6.5)

**P3 — completeness:**
- [ ] **T2V.8 CA/TX** — build v14 substrate → G + ceilings at the two largest (highest-imbalance) states; the
      scale-conditional headline's real gap. (§4.2)

**A40 doc close-out (housekeeping — fill the data + close the stale tiers, user-requested):**
- [ ] **Close Tier 2 (topology):** fill INDEX T2.3 (-lite MoE) + T2.4 (SwiGLU/MulT/xstitch) Results blocks with
      the landed C25-stretch data (all NULL on reg); flip the Tier-2 banner + final-decision from "UNDER
      RE-VALIDATION" → RESOLVED post-C25 (orderings flipped; dual-tower G is the winner).
- [ ] **Close Tier 2P (joint-loop):** mark MOOT — the "joint loop poisons reg" hypothesis WAS the C25
      class-weighting confound; T2P.0 wd sub-question answered (wd not the cause) + T2P.1/.2/.3 superseded.
      Banner the cards superseded → C25 / Tier 2V; keep for the trail.
- [ ] Keep `HANDOFF.md` + this `§7` + INDEX Tier-2V Results blocks in sync as cards land.
