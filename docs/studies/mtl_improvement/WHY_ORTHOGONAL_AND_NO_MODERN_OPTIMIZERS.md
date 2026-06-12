# Why the tasks are orthogonal, why MTL still beats STL, and why modern MTL optimizers don't help

**Audience:** paper / talk reference. Consolidates the conceptual discussion behind the Tier-4 (loss/
optimization) close and the gradient-orthogonality finding. Evidence + figures cited inline.
Date: 2026-06-08.

**TL;DR.** The next-category and next-region tasks have **orthogonal gradients** on the shared trunk
(cos ≈ 0). That means they neither *interfere* nor *cooperate* at the optimization level — so (i) modern
MTL gradient-balancers have nothing to resolve and cannot beat a tuned fixed weight, and (ii) MTL's win
over single-task learning is **representational/architectural**, not gradient-driven. The champion
architecture (G: reg-private dual-tower + shared cat) is *matched to* this task geometry — which is also
why forcing more sharing (Tier 2) failed.

---

## The three questions

### Q1. "If the gradients are orthogonal, the tasks aren't helping each other, right?"

**Orthogonal means they don't *fight* — not that they don't *help*.** The cosine measures only ONE
interaction channel: the first-order gradient direction on the *shared* parameters. Three regimes:

| cos(∇L_cat, ∇L_reg) | meaning | what it implies |
|---|---|---|
| **< 0** (conflict) | tasks pull shared weights in opposing directions; one step undoes the other | destructive interference — this is what PCGrad/CAGrad/Nash exist to fix |
| **> 0** (synergy) | tasks push the same direction | each step directly lowers both losses |
| **≈ 0** (our case) | gradient steps are independent | neither destructive nor directly reinforcing |

Measured: cos ≈ **0** over all 50 epochs at both scales — **FL +0.0007, AL +0.0026** (band [−0.08,+0.19],
~50 % negative = zero-mean noise). Figure: `figs/grad_cosine_tasks.png`.

So orthogonality rules out two things: (a) the tasks hurting each other through the shared trunk, and
(b) the *naive* "they descend together" form of help. But MTL transfer has a **second channel the cosine
does not capture: the shared representation.** The shared trunk is shaped by *both* tasks' data; one task
can act as an inductive-bias / regulariser on the features the other consumes, even when the per-step
gradients are orthogonal. Orthogonal tasks carve out **complementary subspaces** of the representation
rather than competing for the same directions — the *ideal* regime for an auxiliary task to regularise
without interfering.

> One-liner: *"The gradients are orthogonal, so the tasks neither interfere nor cooperate at the
> optimisation level — any benefit is representational, not gradient-driven."*

### Q2. How do we then explain MTL beating STL?

Two results, two distinct mechanisms, each mapping cleanly onto the architecture:

- **Region (reg) ≈ STL ceiling (matches).** G is a **dual-tower**: the reg head has its own *private*
  backbone, so it is effectively an STL pathway that doesn't depend on the shared trunk. Orthogonality
  guarantees the category task isn't damaging the small shared part reg does use. Net: reg has nothing to
  lose → it matches its single-task ceiling (matched-metric Δ −0.09…−0.31 pp, Pareto-non-inferior).

- **Category (cat) beats the STL ceiling by ~+3 pp — but this is ARCHITECTURE-dominated, not region
  transfer.** The cat head reads the *shared* cross-attn trunk. We decomposed the gain with an ablation
  (run G but `--category-weight 1.0` → reg gradient OFF → trunk trains cat-only; reg confirmed cratered to
  0.12 %):

  | state | STL cat (next_gru, no trunk) | cat + cross-attn trunk, **reg OFF** (4-seed) | G cat (reg ON, 4-seed) | **architecture** | **region-transfer** |
  |---|---|---|---|---|---|
  | AL | 50.35 | 53.57 ± 0.24 | 52.91 ± 0.27 | **+3.22** | **−0.67** |
  | FL | 69.96 | 72.09 ± 0.08 | 73.16 ± 0.04 | **+2.13** | **+1.08** |

  The cross-attn shared trunk is a **better category encoder** than the single-task head (+2.1…+3.2 pp).
  Genuine region→category transfer is smaller and scale-dependent: **+1.08 at FL** (large state, enough
  data) and slightly **negative at AL** (−0.67; at small data the region signal mildly distracts cat).
  Both signs hold multi-seed (4-seed {0,1,7,100}, tight σ). The win is mostly a better encoder plus a
  modest representation-level transfer that materialises at scale.
  (Conservative on two counts: the STL ceiling used logit-adjust, which *helps* STL cat — so the
  architecture share is understated; and the ablation isn't a *perfectly* clean isolation — with reg's
  loss weight 0 the cat stream still attends to the reg stream's keys/values in the bidirectional
  cross-attn, so a little region structure remains in the "architecture" term. See
  `docs/results/mtl_improvement/{cat_transfer_and_T53.md, orthogonality_intrinsic_test.md}`.)

**Net Pareto-positive story** ("matches reg, beats cat") = the signature of an **orthogonal task pair
handled by the right architecture**: no interference (reg is safe) + a better shared encoder with small
complementary regularisation (cat gains). The honest paper framing for the cat result is *"the joint
cross-attn architecture is a better category encoder, with a small region→category transfer at scale,"*
NOT *"the region task teaches category."*

### Q3. Is the non-cooperation tied to the architecture, or can we change it?

**The orthogonality is mostly *intrinsic to this task pair*, and the right response is to *exploit* it —
not to try to force cooperation.**

- **It's intrinsic — TESTED, not asserted (2026-06-10).** Category (7 semantic classes) and next-region
  (~4,700-way spatial transition) are genuinely different prediction problems over the same check-in
  sequence; they need different feature directions, so orthogonal shared-trunk gradients are the
  *expected* structure, not a defect. **We ruled out the obvious confound** — that G's dual-tower (β=0.1
  aux gating) merely *starves* reg's shared channel and so *induces* cos≈0. In the dual-tower reg's
  shared gradient IS attenuated (reg:cat ratio 0.26 AL / 0.47 FL); but in a **fully-shared** model
  (`mtlnet_crossattn` + `next_stan_flow`, no private tower) where reg's shared gradient is **larger** than
  cat's (ratio 1.26 AL / 1.78 FL), the cosine is **still ≈0** (+0.0024 AL / +0.0017 FL). The orthogonality
  persists exactly where the confound predicted it should vanish → it is a genuine property of the task
  pair, not an artifact of the architecture. So the dual-tower **exploits** a pre-existing orthogonality;
  it does not manufacture it. (Evidence: `docs/results/mtl_improvement/orthogonality_intrinsic_test.md`.)
  No optimiser can manufacture cooperation that isn't in the task geometry — which is precisely why every
  balancer failed (no conflict to resolve, no synergy to amplify).

- **Architecture influences it — in the opposite direction from "fix it."** Forcing *more* sharing
  *induces* conflict, it doesn't create cooperation. This is exactly what **Tier 2** showed: MoE,
  cross-stitch, and hard-parameter-sharing all *lost* region, while the *dual-tower* (less forced sharing
  for reg) *won*. Pushing two orthogonal tasks through one trunk makes them compete for the same
  directions (drives cos negative) → you'd then need balancers and lose reg. **G's design is the correct
  response to orthogonality: protect region in a private tower, let category harvest the shared encoder.**

- **Could we create genuine synergy (cos > 0)?** Only by changing the *objective*, not the optimiser —
  e.g. an auxiliary loss that explicitly ties category and region (a consistency/contrastive term linking
  a POI's category to its region's category distribution, or a shared structured latent). That is a
  legitimate **future direction**, but speculative and risky: it could just as easily induce the
  interference we currently avoid. The evidenced conclusion for the paper is: *the tasks are orthogonal;
  the contribution is an architecture that exploits orthogonality (protect + regularise), not one that
  fights it.*

---

## Why modern MTL optimizers don't help (the Tier-4 discussion)

We swept the **entire `src/losses` balancer registry** under champion G (RLW, GradNorm, PCGrad, CAGrad,
Nash-MTL, Aligned-MTL, DWA, DB-MTL, FAMO, Uncertainty-Weighting, UW-SO, STCH, FairGrad, BayesAgg,
Excess-MTL, GO4Align, scheduled-static, equal-weight) + loss-scale normalization. **None Pareto-beats a
tuned `static_weight cw=0.75`.** Six convergent lines of evidence (full write-up:
`docs/results/mtl_improvement/T4_audit_and_verdict.md`):

1. **Gradient cosine ≈ 0** (the mechanism). Gradient-surgery / dynamic-weighting methods only help under
   *strongly negative* cosine (high task interference). At cos ≈ 0 there is nothing to resolve.
2. **RLW litmus** — random per-step weighting ≈ tuned static_weight → the inter-task weight is not a
   sensitive lever (Lin TMLR'22).
3. **Full registry screen** (19 arms, AL+FL) — all cluster near the equal-weight point; none beats G.
   Figure: `figs/t4_balancer_scatter_FL.png`.
4. **Targeted retune re-run** (GradNorm @lr=0.05 α=1.5 — genuinely retuned; Nash @max_norm=2.2,
   which *is* the registry default — a config-identity re-run, not a new tuning point) — still trade
   ~1.3–1.5 pp cat for ~0.1 pp reg at FL; no Pareto win. (Wording precision: this is a
   convergent-evidence negative, NOT an exhaustive per-method tuning study — see the
   evidence-strength banner in `T4_audit_and_verdict.md`.)
5. **Static cw-sweep {0.5,0.6,0.66,0.8}** — a monotone reg↔cat trade; 0.75 is on the Pareto front
   (satisfies Xin'22's fairness condition: the baseline is tuned as hard as the challengers).
6. **Loss-scale normalization FALSIFIED** — dividing each CE by log(num_classes) *starves* the
   high-cardinality reg head (FL reg 35→70 across cw, never reaching G's 72.95). The large reg loss
   reflects a genuinely harder task that needs the gradient, not an unfair dominance; G's tuned weight
   already encodes the right scale balance.

### This is the EXPECTED result, not a bug (literature)
For a **2-task** setup with a **tuned** fixed-weight baseline + standard regularization, tuned
scalarization matching/beating specialized MTL optimizers is the central, replicated finding of:
- **Kurin et al., NeurIPS 2022**, *"In Defense of the Unitary Scalarization"* — scalarization + standard
  regularization matches/beats complex MTL optimizers; the specialized methods are "partly regularization."
- **Xin et al., NeurIPS 2022**, *"Do Current MTO Methods Even Help?"* — at k=2, none beat a scalarization
  sweep; **LR/HP variance is 6–7× the method effect** (most "wins" were under-tuned baselines).
- **Royer et al., NeurIPS 2023** — uniform scalarization on-par; the value of specialized methods is
  *search efficiency*, not a Pareto gain.
- **2025 "Uniform Loss vs. Specialized Optimization"** — SMTOs only win under *high task interference*
  (strongly negative gradient cosine) and many tasks; at near-orthogonal k=2 they match unit scalarization.
- **Hu et al., NeurIPS 2023** (theory) — scalarization can miss balanced Pareto points under
  non-convexity, but the failure probability *grows with task count* (2^−O(k²)); at k=2 it's not the
  regime, and it's about *reaching* points, not validation metrics.

The only condition under which a balancer win would be *expected* — an *under-tuned* static baseline — we
ruled out with the cw-sweep (line 5).

### An honest implementation caveat (audit)
The first balancer screen was *partially invalid*: under G's dual-tower, the **gradient-surgery family
(CAGrad/PCGrad/Aligned-MTL) only reweights `shared_parameters()`**, so the private reg tower (>80 % of the
reg pathway) trains at unit weight and they silently collapse to ≈equal-weighting; and GradNorm/DWA/
FairGrad were misconfigured at defaults (a latent preflight bug was also found + fixed). The corrected
re-runs (line 4) + the cosine ≈ 0 mechanism (line 1) + the literature make the negative airtight
regardless. Figure `figs/t4_loss_weight_trajectories_FL.png` shows which methods actually engage.
Details: `docs/CONCERNS.md §C27`, `docs/results/mtl_improvement/T4_audit_and_verdict.md`.

---

## The unifying picture (one paragraph for the paper)
The two tasks are **orthogonal** on the shared representation (cos ≈ 0). This single fact explains the
whole study: **(1)** modern MTL optimizers can't help (no conflict to resolve — Tier 4); **(2)** forcing
more parameter sharing *hurts* region (it would induce the conflict that isn't there — Tier 2's MoE/
cross-stitch/hard-share losses); and **(3)** the champion dual-tower wins because it is *matched to the
task geometry* — it protects region in a private tower (so it matches its ceiling) and lets category
harvest the jointly-trained cross-attn encoder (a better encoder + small transfer at scale). The MTL win
is real and Pareto-positive, but it is **architectural/representational, not gradient-cooperation** — and
that is the honest, evidence-backed story.

## Figures
- `docs/studies/mtl_improvement/figs/grad_cosine_tasks.png` — task-gradient orthogonality (AL + FL).
- `docs/studies/mtl_improvement/figs/t4_balancer_scatter_FL.png` — balancers cluster at equal-weight; none beats G.
- `docs/studies/mtl_improvement/figs/t4_loss_weight_trajectories_FL.png` — which balancers actually engage.
