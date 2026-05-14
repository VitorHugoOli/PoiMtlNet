# Critical Analysis — Check2HGI MTL Proposal (2026-04-28)

**Trigger:** user asked for a deep critical evaluation of the MTL proposal:
is the current approach correct, are the head + shared-layer compatible,
are there better heads, should we try a different approach, or many at
once?

**Source basis:** the same dataset of trackers + reviews + archived plans
read for the 2026-04-28 compiled overview, plus 7 targeted web searches on
recent (2023-2026) MTL literature.

**My role here:** lay out the genuine ambiguities and the unexplored
levers, not pre-resolve framing decisions the user has already made
deliberately. The empirical work is clean and rigorous. There are real
open questions about (a) whether the current MTL framing is the strongest
paper centroid, (b) whether the head + shared-layer combination is
structurally adequate, and (c) which untested architectural alternatives
have the best information value-per-hour. I lay these out neutrally.

---

## 0. Executive summary

The empirical work is rigorous and the per-state pattern is clean: at
1.1K regions (AL) MTL exceeds matched-head STL by +6.25 pp; at 1.5K (AZ)
it ties; at 4.7K (FL) it loses by 8.78 pp p=0.0312. The architectural
cost grows monotonically with region cardinality. The cat-side MTL > STL
relation holds at all three states (+0.94 to +3.64 pp), driven by the
substrate.

The trackers show a deliberate framing decision in `PAPER_DRAFT.md §1`:
the user already considered and rejected a "MTL beats STL" defensive
framing in favour of *Beyond Cross-Task Transfer*, which already
foregrounds the transfer-null finding. So the framing isn't naïvely "MTL
wins"; it is "we refute conventional MTL transfer and propose a
characterised joint deployment". That framing reads cleanly. A reviewer
can accept it.

What the analysis below adds is:

1. **Tier 0 (~1h, no compute):** the joint Δm computation across
   AL+AZ+FL using existing JSONs is the single highest-leverage missing
   analysis. It disambiguates whether MTL Pareto-loses at FL or sits on
   the frontier. The cat F1 lift, even at 0.94 pp, may carry the joint
   metric depending on Δm normalisation. This is bookkeeping, not
   re-litigation, and it sits *underneath* whichever paper framing wins.

2. **Tier 1 (~14h compute):** four cheap empirical tests (F33 cat-head
   decision; FAMO/Aligned-MTL drop-in; hierarchical softmax;
   2-model-STL ensemble cost) that either reinforce or break the "FL
   architectural cost is fundamental" reading. **Worth running before P3
   launches**, because a Tier-1 result that changes the FL champion
   would force CA+TX re-runs under the new champion — and CA+TX is the
   ~37h Colab pole on the critical path.

3. **Head + shared-layer compatibility analysis:** three independent
   structural symptoms (disjoint optimal LRs, divergent inductive biases,
   head-size mismatch) all support an "incompatibility grows with
   scale" reading. The architectural-Δ pattern is consistent with this.

4. **Methodological elevation:** the F49 Layer 2 finding (loss-side
   `task_weight=0` ablation is unsound under cross-attention MTL) is
   broadly applicable to MulT/InvPT/HMT-GRN and deserves more prominent
   placement in the paper than its current §3.6.

5. **Framing tension surfaced (not pre-resolved):** the substrate is the
   most generalisable finding; the architectural lift is AL-only. The
   user can defend either Path A (current framing, with Tier-1 insurance)
   or Path B (substrate-first reframe). I describe both honestly.

---

## 1. The 3-state pattern (what the data actually say)

| Region cardinality | State | MTL vs matched-head STL on reg | Architectural Δ (F49 frozen-cat vs STL F21c) |
|---:|---|---:|---:|
| 1.1K | AL | **+6.25 pp** ✓ (Wilcoxon p=0.0312) | +6.48 pp ~2.7σ |
| 1.5K | AZ | −3.29 pp (n.s.) | −6.02 pp ~3.7σ |
| 4.7K | FL | **−8.78 pp** ✗ (p=0.0312, 5/5 folds neg) | **−16.16 pp** p=0.0312 |

The cat-side MTL > STL relation holds at all three states (+3.64 AL /
+3.03 AZ / **+0.94** FL) — and the FL +0.94 pp is post-F37 P1 (matched
STL `next_gru` 66.98 ± 0.61 vs MTL H3-alt 67.92 ± 0.72). This is small
but consistent.

So the picture is two-pronged:

- **Cat side: MTL > STL at every state**, magnitude shrinks with scale.
- **Reg side: MTL > STL only at AL**, MTL ≪ STL at FL, with a monotone
  architectural cost in between.

The current paper framing acknowledges this as "scale-conditional"
(CH18 Tier A reframed; CH21 reframed per-state). That is honest.

The genuine open question is **whether 0.94 pp cat F1 + (−8.78 pp reg)
sums to a Pareto-positive joint metric** in the MTL-survey-standard Δm
formulation. **This is computable from existing JSONs in ~1 hour.** See
Tier 0 below.

## 2. Head + shared-layer compatibility — the user's specific question

I read the data as showing **strong incompatibility at scale**. Three
independent symptoms:

### 2.1 Disjoint optimal LR regimes (3× ratio)

H3-alt's recipe: `cat_lr=1e-3`, `reg_lr=3e-3`, `shared_lr=1e-3`,
constant schedule. The cat path saturates at 3e-3 (collapses; F45
confirmed at AL/AZ); the reg head's α-prior needs 3e-3 sustained for α
to grow. The shared layer sits at 1e-3 to keep cat alive. F48-H1 / F48-H2
/ F40 confirmed empirically that no single shared LR or schedule makes
both heads thrive.

This is a **structural symptom of incompatibility**: two heads with
different optima on the same shared backbone means the backbone is being
asked to serve objectives that pull in different directions. H3-alt is
a workaround that decouples the optimisation, not a principled mechanism.

### 2.2 Different inductive biases

- Cat head (`next_gru`): recurrent, last-token softmax, good at
  sequential intent shifts.
- Reg head (`next_getnext_hard`): STAN attention + `α·log_T[last_region]`
  graph prior, good at positional/structural region transitions.

These are fundamentally different prediction problems consuming a shared
representation. AL's small cardinality lets this work; FL's larger
cardinality exposes the friction.

### 2.3 Head-size mismatch (3 orders of magnitude)

7-class softmax vs 4,702-class softmax. The shared backbone's finite
capacity is being divided across two regimes that need radically
different feature distributions. Shared features useful for a 7-way
decision have no a priori reason to be useful for a 4.7K-way decision.

### 2.4 Verdict

The empirical data exhibit *all three* canonical symptoms of MTL head
incompatibility. The fact that H3-alt rescues AL, partially AZ, and not
FL is consistent with "incompatibility grows with scale" — exactly what
the architectural-Δ pattern shows.

That said, calling the heads "incompatible" doesn't mean MTL is
hopeless. It means **the current shared-layer mechanism (fixed
cross-attention with loss-side weighting) is inadequate for incompatible
heads at scale**. Mechanisms that *learn* the share-vs-task split (PLE,
Cross-Stitch, MTI-Net) explicitly target this and have not been tested.

## 3. F49 Layer 2 is the paper's strongest methodological contribution and is underplayed

The finding "loss-side `task_weight=0` ablation is unsound under
cross-attention MTL because the silenced task's encoder co-adapts via
attention K/V projections" is **genuinely novel and broadly applicable**.

It applies retroactively to:

- **MulT** (Tsai et al., ACL 2019) — multimodal cross-attention
- **InvPT** (Ye et al., ECCV 2022) — inverted pyramid cross-task
- **HMT-GRN** (Lim et al., SIGIR 2022) — closest competitor in POI
- Any cross-task interaction MTL with `task_weight=0` ablation

The current `PAPER_DRAFT.md §3.1` puts this at **C4 of 4 contributions**
and Methods §3.6 (Methodological Appendix or Limitations). I'd argue it
deserves promotion to **C2 or C3**, between the substrate finding and
the architectural finding. Its applicability extends beyond POI and is a
reusable contribution any cross-attention MTL paper should cite.

## 4. The framing question (not pre-resolved)

Two viable paper centroids exist in the same data:

### Path A — Current framing (in PAPER_DRAFT.md as committed)

> *Beyond Cross-Task Transfer: Per-Head Learning Rates and Check-In-Level
> Embeddings for Multi-Task POI Prediction*

Strengths:
- Foregrounds the transfer-null finding (CH20 Layer 1).
- Names both technical contributions in the subtitle.
- Already considered alternatives in `PAPER_DRAFT.md §1`; this title
  was preferred over four alternates.
- The cat-side MTL > STL relation holds at all three states.

Risks:
- Headline-state reg result shows MTL ≪ STL by 8.78 pp. Reviewer asks:
  "why deploy MTL at FL when STL beats it on the primary reg metric?"
- The "single-model deployment" rebuttal is a deployment argument, not
  a performance argument.
- Tier-0 Δm result could undercut or reinforce this; it is unrun.

### Path B — Substrate-first reframe (alternative)

> *Substrate Carries the Joint Task: Check-In-Level Embeddings for POI
> Category and Region Prediction*

Strengths:
- Substrate gain (+12-15 pp cat F1) generalises across all 3 states.
- POI-RGNN external comparison is +28-32 pp at FL — that's a
  publishable result on its own.
- F49 Layer 2 stays as a clean methodological contribution.
- No "MTL works" claim that the FL data can refute.

Risks:
- Loses the "novel MTL recipe" claim.
- The H3-alt recipe is genuinely an interesting AL-only finding that
  this framing demotes.

### My honest assessment

**I do not have enough evidence to recommend a pivot.** The Tier-0 Δm
computation is the load-bearing missing analysis. If MTL Pareto-dominates
on Δm at FL (cat lift carrying the small reg cost), Path A reads cleanly
and Path B is unnecessary. If MTL Pareto-loses on Δm at FL, Path A becomes
fragile and Path B is worth considering.

**Run Tier 0 first. Then decide.**

## 5. What hasn't been tried — ranked by information value

### Tier 0 — analysis-only, ~1h, no compute

| # | Item | Cost | Information value |
|---|---|:--:|---|
| **T0.1** | **Joint Δm computation across AL+AZ+FL using existing JSONs** | ~1h | Decides whether MTL Pareto-loses or sits on the frontier at FL. Loadbearing for Path A vs B framing decision. Reuses `scripts/analysis/p4_p5_wilcoxon_offline.py`. |

### Tier 1 — small targeted tests, ~14h compute, run *before* P3 launches

| # | Item | Cost | Information value |
|---|---|:--:|---|
| **T1.1** | **F33 — FL 5f×50ep B3+`next_gru` cat-head decision (Path A vs Path B)** | ~6h Colab T4 | Already in FOLLOWUPS_TRACKER as P1 paper-blocking. Decides whether `next_gru` cat head generalises beyond AL+AZ or needs scale-dependent head. Must land before CA+TX P3 because P3 inherits the cat-head choice. |
| **T1.2** | **Hierarchical softmax on the reg head at FL** | ~3h Colab T4 | Tests whether reg-side head-capacity mismatch (4.7K-class flat softmax) drives the architectural cost. Cheapest test of the head-side incompatibility hypothesis. |
| **T1.3** | **FAMO drop-in replacement of `static_weight` at FL** | ~3h Colab T4 | Newer (NeurIPS 2023) gradient balancer. **Caveat:** FAMO's reported wins are on NYUv2/CityScapes/CelebA/QM9, which look nothing like POI's 4.7K-class long-tail. Domain transfer is *speculative*. Worth running because the cost is low, but don't expect this to fix FL. |
| **T1.4** | **Aligned-MTL drop-in replacement** | ~2h Colab T4 | Same caveat as T1.3 — designed for high-dim task vectors, validated on dense vision benchmarks. Speculative for POI. |

### Tier 2 — architecture changes, run *only if* Tier 1 motivates (~30-50h)

| # | Item | Cost | Information value |
|---|---|:--:|---|
| **T2.1** | **PLE backbone** (Tang et al., RecSys 2020) with same heads at AL+AZ+FL | ~25h | Explicit task-specific + shared expert separation. The most theoretically-motivated alternative to fixed cross-attention for incompatible heads at scale. |
| **T2.2** | **Cross-Stitch Network** with same heads at AL+AZ | ~15h | Learned per-layer share/task-specific weights. Directly tests whether forced sharing (vs learned sharing) is the FL bottleneck. |
| **T2.3** | **ROTAN** (KDD 2024) reg head as STL ceiling reference at AL+AZ+FL | ~20h | Tests whether GETNext-hard is the right reg ceiling, or whether rotation-based temporal attention is better. Recent (2024) competitor. |

### Tier 3 — bigger pivots, run *only if* Tier 1+2 confirm MTL fails (~50h+)

| # | Item | Cost |
|---|---|:--:|
| **T3.1** | **Bi-Level GSL prototype head** for FL reg (long-tail mitigation via prototypes) | ~30h |
| **T3.2** | **Distillation: STL teacher → MTL student** (single-model deployment without sharing penalty) | ~30h |
| **T3.3** | **Substrate-first paper rewrite** (Path B) — no compute | ~10h |

## 6. Critical-path recommendation

**Order of operations:**

1. **Tier 0 (~1h, no compute, today).** Compute joint Δm across AL+AZ+FL
   from existing JSONs. This decides whether MTL Pareto-loses at FL.
2. **Tier 1 in parallel (~14h compute, 1-2 days end-to-end), before P3 launches.**
   F33 is already paper-blocking and unblocks the cat-head decision for
   CA+TX. Hierarchical softmax + FAMO + Aligned-MTL test specific FL
   hypotheses cheaply; treat the FAMO/Aligned-MTL results as exploratory
   given the domain gap.
3. **Reconvene with the user after Tier 0 + Tier 1** before committing
   to Tier 2/3 or a paper-pivot. The information from these constrains
   which Tier-2 directions are worth ~25h+ each.
4. **P3 (CA+TX upstream + 5f H3-alt) launches *after* Tier 1**, not in
   parallel. P3 is ~37h Colab and inherits whichever champion config
   Tier 1 settles on. If Tier 1 changes the FL champion, P3 needs to
   re-run under the new champion — running concurrently risks wasted
   Colab compute.

The "monotone architectural cost grows with region cardinality" pattern
(1.1K → 1.5K → 4.7K → +6.5 / −6.0 / −16.2 pp) is the cleanest per-state
characterisation in the data. CA + TX P3 extends it to a 5-point scale
curve and turns "FL outlier" into a "scale curve" story — which is
strictly better for the paper.

**Tier 1 is insurance for P3.** If Tier 1 reveals a working FL configuration,
the paper has a much stronger story and CA+TX runs under that
configuration. If Tier 1 confirms current findings, P3 proceeds under
H3-alt and the paper retains its current framing with one additional
layer of "we tried these alternatives and they failed" rebuttal ammunition.

## 7. Direct answers to the user's questions

> **"Are we doing it correctly?"**

The empirical work is correct and rigorous. The current paper framing is
defensible (it foregrounds transfer-null, not "MTL wins"). The
load-bearing missing analysis is the joint Δm computation across all 3
states from existing JSONs (~1h).

> **"Could we try a different approach?"**

Yes — three tiers above. The cheapest with highest information value is
**Tier 0** (Δm from existing data). Tier 1 (~14h) tests four specific
hypotheses about why FL fails. **Run Tier 0 + Tier 1 before P3 launches.**

> **"Maybe the head and shared layer is not so compatible or not well defined?"**

This is the right intuition and the data support it. Three independent
symptoms (disjoint optimal LRs, divergent inductive biases, head-size
mismatch) all support a "heads structurally incompatible at scale"
reading. H3-alt is a workaround that decouples optimisation but doesn't
fix the underlying incompatibility. Alternative shared-layer mechanisms
(PLE, Cross-Stitch) explicitly target this and haven't been tried.

> **"Have we discovered other heads that produce better results worth the try?"**

For reg: hierarchical softmax (cheapest test, ~3h on FL); ROTAN (KDD
2024, ~20h); Bi-Level GSL prototypes for long-tail (~30h). For cat:
F33 is the unresolved decisive test — `next_gru` works at AL+AZ but
flipped sign at FL n=1 (F32). The C14 scale-dependence flag is open.

> **"Better to try a totally new approach? Or many at once?"**

Don't do many at once now. Sequential strategy is better:

- Tier 0 (~1h) — analysis-only, no risk
- Tier 1 (~14h, parallel) — cheap targeted tests including F33
  paper-blocker
- *Reconvene with user, then* Tier 2 if needed
- *Paper-pivot decision is informed by* Tier 0 + Tier 1, not preemptive

## 8. Caveats and explicit speculation flags

- I did not run any experiments. Recommendations are based on tracker
  data + literature search summaries.
- I have not read the full FAMO / Aligned-MTL / PLE / ROTAN papers; my
  recommendations are based on abstracts + secondary summaries. **The
  FAMO and Aligned-MTL claims of "comparable or superior" performance
  are validated on NYUv2/CityScapes/CelebA/QM9 — none of which are
  long-tail multi-class benchmarks remotely like POI's 4.7K-class reg
  head. Treat T1.3 + T1.4 as exploratory only.**
- The literature search was 7 queries; a more thorough survey would
  extend to ~30 papers across MTL benchmarks (NYUv2 / CityScapes /
  CelebA / QM9 / WikiText) plus POI domain (Gowalla / FSQ / commercial).
- I have a personal bias toward substrate-first framing because it's the
  cleaner scientific story. The current MTL framing in PAPER_DRAFT.md is
  *not* "MTL wins" naïvely — it's "we refute conventional MTL transfer
  and offer a characterised joint deployment." That's a smarter framing
  than my §4 originally credited. The user's deliberate choice in
  PAPER_DRAFT.md §1 alternates is defensible. I retract the
  "must pivot" register from earlier drafts of this analysis.

## 9. Sources

- FAMO (NeurIPS 2023): https://arxiv.org/abs/2306.03792
- Aligned-MTL (CVPR 2023): "Independent Component Alignment for Multi-Task Learning", Senushkin et al.
- PLE (RecSys 2020): https://dl.acm.org/doi/fullHtml/10.1145/3383313.3412236
- Bi-Level GSL (arXiv 2024): https://arxiv.org/html/2411.01169v1
- ROTAN (KDD 2024): https://dl.acm.org/doi/10.1145/3637528.3671809
- ForkMerge (arXiv 2023): https://arxiv.org/abs/2301.12618
- DST (OpenReview 2024): https://openreview.net/forum?id=myjAVQrRxS
- Cross-task Attention (WACV 2023): https://arxiv.org/html/2206.08927
- Hierarchical softmax (large-scale): https://arxiv.org/pdf/1812.05737
- Balanced Meta-Softmax (NeurIPS 2020): https://proceedings.neurips.cc/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf
- Awesome MTL paper list: https://github.com/thuml/awesome-multi-task-learning
- Long-tailed survey (arXiv 2024): https://arxiv.org/html/2408.00483v1
- MGCL (Frontiers 2024): https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1428785/full
- Selective Task Group Updates (arXiv 2025): https://arxiv.org/html/2502.11986
- Fantastic Multi-Task Gradient Updates (arXiv 2025): https://arxiv.org/html/2502.00217
