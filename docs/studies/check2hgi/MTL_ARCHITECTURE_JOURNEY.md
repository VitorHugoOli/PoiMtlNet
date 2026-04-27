# MTL Architecture Journey — From Initial Design to F48-H3-alt

**Status:** living document, last updated 2026-04-26 after F48-H3-alt + F40 + F48-H2 closure. **Audience:** future agents picking up the branch, paper readers needing the back-story, and the author when revisiting why a particular design decision was made.

This document tells the story of how the Check2HGI MTL configuration arrived at the current per-head LR recipe. It complements the per-experiment research notes (`research/F*_FINDINGS.md`) by stitching them into a single narrative — what we tried, what failed, what survived, and why each step led to the next.

The intent is not exhaustive history (`FOLLOWUPS_TRACKER.md` and the per-F notes serve that purpose). The intent is **causal**: each section leads to the next because of an empirical signal, not because of a calendar.

---

## 0 · Problem framing

The Check2HGI study trains a single MTL model that predicts:
- `next_category` (7 classes) — macro-F1 primary
- `next_region` (~1K AL / ~1.5K AZ / ~4.7K FL classes) — Acc@10 + MRR primary

The two heads share an MTLnet-style backbone with task-specific encoders, a shared mixing layer, and per-task prediction heads. The motivation is **deployment** (one forward pass, both predictions) **and** the hope of joint-training transfer (each head's gradient improves the other's representation). The latter is the paper's central testable claim.

---

## 1 · Phase B-M: finding the architecture

The first months iterated through MTL backbone variants (`mtlnet`, `mtlnet_cgc`, `mtlnet_ple`, `mtlnet_mmoe`, `mtlnet_dselectk`, `mtlnet_crossattn`) and MTL loss variants (NashMTL, PCGrad, EqualWeight, StaticWeight, UncertaintyWeighting, FAMO, etc.). The convergent choices:

- **Backbone:** `mtlnet_crossattn` — bidirectional cross-attention between cat and reg streams. Outperformed mlp-shared-backbone variants and gating variants (CGC, PLE, MMoE, DSelectK) on the joint Acc@10 + cat-F1 metric.
- **Loss:** explored PCGrad → NashMTL → static_weight. PCGrad (gradient-projection) and NashMTL (Nash equilibrium) both helped on AL/AZ but PCGrad failed at FL scale (cat head's best-val F1 collapsed to ~55 vs ~66 — see `research/B5_FL_SCALING.md`). NashMTL had memory issues. **`static_weight(category_weight=0.75)`** emerged as the simpler, scale-stable winner.
- **Cat head:** `NextHeadMTL` (Transformer with attention pooling) was the early default.
- **Reg head:** explored GRU → STAN (Yang WWW'21) → GETNext-soft (learned probe) → GETNext-hard (faithful Yang SIGIR'22 reproduction with `α · log_T[last_region_idx]` graph prior). **`next_getnext_hard`** (the hard graph-prior head) won by enabling α to grow over training.

This phase produced **B3 (the predecessor champion):**

```
mtlnet_crossattn + static_weight(cat=0.75) + NextHeadMTL (cat) + next_getnext_hard (reg)
d=256, 8 heads, max_lr=3e-3, OneCycleLR (PyTorch default), 50 epochs, batch=2048
```

5-fold validated on AL/AZ. The mechanism behind static_weight(cat=0.75) was **late-stage handover** — cat head converges fast in early epochs under high cat weight, then the shared backbone becomes available to reg in the remaining epochs (cat training extends to ep ~42 vs ≤10 for soft/equal-weight). See `research/B5_RESULTS.md` and `research/B5_FL_TASKWEIGHT.md`.

---

## 2 · F27: cat-head refinement

The Transformer-based `NextHeadMTL` cat head was inherited from MTLnet's defaults. F27 swapped it for `next_gru` (a 2-layer GRU with attention pooling) and re-validated:

| State | Cat F1: NextHeadMTL → next_gru | Wilcoxon p_greater |
|---|---|---|
| AL 5f | +3.43 pp | sig |
| AZ 5f | +2.37 pp | **0.0312** |
| FL 1f | −0.93 pp | n=1 noise |

`next_gru` won decisively at AL/AZ. FL flipped sign at n=1 — flagged as scale-dependence pending F33 (FL 5f). The committed B3 (post-F27) used `next_gru` for AL/AZ; FL kept the open flag.

**Post-F27 B3 numbers (the 2026-04-24 snapshot):**

| State | cat F1 | reg Acc@10 |
|---|---:|---:|
| AL | 42.71 ± 1.37 | 59.60 ± 4.09 |
| AZ | 45.81 ± 1.30 | 53.82 ± 3.11 |
| FL | 65.72 (1f) | 65.26 (1f) |

These numbers were the "current truth" for ~24 hours before F21c reshaped everything.

---

## 3 · F21c: the gap

F21c introduced a **matched-head STL baseline**: `next_getnext_hard` trained as a single-task model (no cat head, no shared backbone, no MTL coupling), with the same input pipeline as MTL-B3's reg path.

Result was uncomfortable:

| State | STL `next_getnext_hard` Acc@10 | MTL-B3 Acc@10 | Δ (MTL − STL) |
|---|---:|---:|---:|
| AL | **68.37 ± 2.66** | 56.33 ± 8.16 | **−12.04** |
| AZ | **66.74 ± 2.11** | 52.76 ± 3.92 | **−13.98** |

**STL with the same head class beat MTL on region by 12–14 pp.** The MTL coupling was not adding value on the region head — it was actively diluting.

Initial response: filed as **CH18 (Tier B — methodological limitation)** in the claims catalog. The paper would be reframed: MTL-B3 still adds cat F1 lift (+4.13 AL / +3.73 AZ vs STL cat) and produces both heads in one forward pass, but it doesn't beat matched-head STL on reg. The escalation criterion to Tier A was: "a future MTL variant closes ≥75% of the 12-14 pp gap without regressing cat F1."

The `research/F21C_FINDINGS.md §Next steps` section listed candidate recovery mechanisms (per-task weight clipping, prior-magnitude normalisation, per-fold transition matrix, PIF-style frequency prior, etc.). None had been tested.

---

## 4 · CH18 attribution chain — systematic refutation of factors (F38–F48)

Rather than guessing which recovery mechanism to try, the next phase **systematically refuted plausible causes** of the gap. Each F-numbered experiment isolates one factor.

### Factor 1: checkpoint selection (F38, refuted)

Hypothesis: maybe MTL is selecting a worse checkpoint per fold than STL. F38 re-analysed existing JSONs, comparing `diagnostic_task_best` epoch selection. Δ ≤ 0.4 pp at Acc@10. **Refuted with zero compute.** `research/F38_CHECKPOINT_SELECTION.md`.

### Factor 2: loss weight (F39, refuted)

Hypothesis: cat=0.75 starves reg. F39 swept cat_weight ∈ {0.25, 0.50, 0.75} on AL 5f. Reg Acc@10 window: 0.64 pp across all three weights. **Refuted.**

### Factor 3a: upstream MLP pre-encoder (F41, refuted)

Hypothesis: the task-specific encoder (Linear+ReLU+LayerNorm+Dropout) before the shared backbone is the contamination source — it's MTL-specific and STL doesn't have it. F41 ran STL `next_getnext_hard` with the MTL pre-encoder added. Result: AL 67.95 ± 2.67 vs STL puro 68.37 (Δ=−0.42, σ-tied); AZ 66.30 vs 66.74. **Refuted.** `research/F41_PREENCODER_FINDINGS.md`.

### Factor 4: epoch budget (F42, refuted inversely)

Hypothesis: 50 epochs is too few; the reg head needs more. F42 ran B3 AL 5f × 150ep with default OneCycleLR. Result: WORSE — reg went from 59.60 → 56.14 and cat from 42.71 → 40.68. **Refuted in the opposite direction**: more epochs hurt because OneCycleLR's annealing tail extended into territory the schedule wasn't designed for.

### Factor 5: OneCycleLR magnitude (F44, refuted)

Hypothesis: max_lr=3e-3 was too high. F44 ran B3 AL+AZ 150ep with max_lr=1e-3 (gentler peak). Reg Acc@10 stayed flat (58.82 AL, 47.91 AZ) — actually worse on AZ. **Refuted.**

### Factor 6 (the breakthrough): no LR annealing at all (F45)

Hypothesis: maybe annealing IS the problem — α (the graph-prior weight in `next_getnext_hard.head`) needs sustained high LR to grow, and OneCycleLR cuts it off. F45 ran 150ep with **constant LR=3e-3** (no warmup, no annealing).

Result was dramatic AND informative:

| State | F45 reg Acc@10 | F45 cat F1 | vs STL F21c |
|---|---:|---:|---|
| AL | **74.20 ± 2.95** | **10.44 ± 0.04 💀** | **+5.83 EXCEEDS** |
| AZ | **63.34 ± 2.46** | **12.23 ± 0.16 💀** | -3.40 |

**The reg arch IS capable of beating STL.** AL exceeded the STL ceiling. But cat collapsed to majority-class baseline (~10–12% F1 on 7 classes).

This was the load-bearing insight. Three more experiments (F46 short-warmup, F47 75ep intermediate budget, F48-H1 constant 1e-3) confirmed:
- Any OneCycleLR variant: reg flat at ~60, cat preserved
- Constant 3e-3: reg lifts to 74, cat dies
- Constant 1e-3: cat preserved, reg flat (the head's α never gets the LR it needs to grow)

Conclusion of the F44-F48 sweep: **two heads have disjoint optimal LR regimes**. Cat needs LR ≤ 1e-3 sustained or warmup-then-anneal. Reg needs LR ≥ 2e-3 sustained 50+ epochs. No monolithic schedule serves both. See `research/F44_F48_LR_REGIME_FINDINGS.md`.

---

## 5 · F48-H3: the per-head LR experiment (and its surprise)

The disjoint-regimes finding suggested the obvious next step: **build an optimizer with separate LRs per head**. The hypothesis was clean — give cat its gentle regime AND reg its sustained regime, in the same training run.

Implementation (~80 LOC, commit `565c478`):
- Added `cat_specific_parameters` / `reg_specific_parameters` methods to `MTLnetCrossAttn`
- New `setup_per_head_optimizer` builds AdamW with three param groups (cat, reg, shared)
- Guarded the `constant` scheduler from overwriting per-group LRs
- CLI flags `--cat-lr / --reg-lr / --shared-lr`

The first config (F48-H3) used the advisor-recommended default `shared_lr = reg_lr = 3e-3`. The reasoning was: cross-attn blocks are in the reg gradient path, so throttling them would reproduce F44 (gentle peak hurts reg).

**F48-H3 result was unexpected:** essentially F45 with extra knobs that didn't matter.

| State | cat F1 | reg Acc@10 |
|---|---:|---:|
| AL | 11.53 ± 1.63 💀 | 74.24 ± 2.58 |
| AZ | 19.61 ± 13.34 💀 | 62.04 ± 1.90 |

Cat collapsed despite `cat_lr=1e-3`. The cat encoder LR was throttled, but the cross-attn shared layers updating at 3e-3 destabilised the cat path **upstream** — by the time the cat-head got the signal, the shared cross-attn had already corrupted the cat-relevant features. Per-head cat-encoder LR was an irrelevant lever when shared was at 3e-3.

This was a **rich negative result**. It revealed that the F45 reg-lift mechanism was actually two coupled effects:
1. α growth in the reg head (drives reg lift)
2. Cat collapse (uncontested reg gradient through shared cross-attn)

The advisor reversed: the *real* discriminating test was the inverse — `shared_lr = 1e-3` (gentle), keeping shared cross-attn stable while letting α grow in the reg head where reg-only gradient flows.

---

## 6 · F48-H3-alt: the recipe (the win)

```
cat_lr    = 1e-3 constant   # cat encoder + cat head
reg_lr    = 3e-3 constant   # reg encoder + reg head (including α)
shared_lr = 1e-3 constant   # cross-attn blocks + final_lns
```

α (graph-prior weight in `next_getnext_hard.head`, line 80) is in `reg_specific_parameters` — it gets the sustained 3e-3 it needs. Shared cross-attn at 1e-3 stays stable enough for cat. Cat encoder at 1e-3 doesn't diverge.

**Results:**

| State | cat F1 (B3 → H3-alt, Δ) | reg Acc@10 (B3 → H3-alt, Δ) | vs STL F21c |
|---|---|---|---|
| **AL** | 42.71 → **42.22 ± 1.00** (-0.49) | 59.60 → **74.62 ± 3.11** (+15.02) | **+6.25 EXCEEDS** ✓ |
| **AZ** | 45.81 → **45.11 ± 0.32** (-0.70) | 53.82 → **63.45 ± 2.49** (+9.63) | -3.29 (closes 75% of B3 gap) |
| **FL** | 65.72† → **67.92 ± 0.72** (+2.20) | 65.26† → **71.96 ± 0.68** (+6.70) | TBD (F37 4050) |

†FL B3 from F32 1-fold n=1.

**CH18 promoted from Tier B to Tier A** the same evening — the recipe satisfies the original "≥75% gap closure without regressing cat F1" criterion AND exceeds STL on AL. See `research/F48_H3_PER_HEAD_LR_FINDINGS.md`.

---

## 7 · Bracketing controls — F40 and F48-H2

Two more experiments tested whether the per-head LR mechanism was unique or whether simpler levers could substitute.

### F40 — loss-side scheduling (orthogonal lever)

`ScheduledStaticWeightLoss` interpolates `cat_weight` linearly from 0.75 (ep 0) to 0.25 (ep 49). Same B3 architecture, same OneCycleLR. Tests whether shifting the gradient budget temporally (cat first, reg later) substitutes for the per-head LR.

| State | cat F1 | reg Acc@10 (B3, Δ) | Pareto +3pp |
|---|---|---|:-:|
| AL | 42.63 | 60.81 (+1.21) | ✗ |
| AZ | 44.98 | 54.39 (+0.57) | ✗ |

Cat preserved, reg only marginally lifted. **Loss-side scheduling does not substitute** — under OneCycleLR both heads still anneal together; α can't grow regardless of gradient share. `research/F40_SCHEDULED_HANDOVER_FINDINGS.md`.

### F48-H2 — warmup-then-plateau (single LR with cat-protection)

`warmup_constant` scheduler ramps LR from ~1e-4 to 3e-3 over 50 epochs, then holds 3e-3 for 100 epochs (total 150ep, single LR). Cat survives the gentle ramp; reg gets a 100-epoch plateau at 3e-3 — what the F45 mechanism needs.

| State | cat F1 (B3, Δ) | reg Acc@10 (B3, Δ) |
|---|---|---|
| AL | 41.35 (-1.36) | **57.84 (-1.76)** |
| AZ | 44.45 (-1.36) | **48.91 (-4.91)** |

Cat preserved BUT reg WORSE than B3. The 100-epoch plateau didn't lift reg.

Why? **Because cat survived.** When cat is alive at the plateau, both heads compete for shared cross-attn capacity. α tries to grow in the reg head but its effective gradient through shared is contested by cat's gradient. The F45 lift was not "sustained LR enables α" alone — it was "cat collapses → uncontested reg gradient through shared → α grows." H2 has the LR ingredient but lacks the cat-collapse ingredient. `research/F48_H2_WARMUP_CONSTANT_FINDINGS.md`.

### Why H3-alt is unique

The three negative controls map a 3-axis design space:

| Lever | Cat | Reg | Why |
|---|:-:|:-:|---|
| F45 — single LR everywhere @ 3e-3 | 💀 | ↑↑ (74) | uncontested reg gradient → α grows |
| F40 — loss-side cat_weight ramp | OK (43) | ≈ (61) | OneCycleLR confound — α can't grow |
| F48-H2 — warmup+plateau, single LR | OK (41) | ↓ (58) | cat-vs-reg compete for shared at plateau LR |
| **F48-H3-alt — per-head LR** | **OK (42)** | **↑↑ (75)** | **shared gentle for cat + α=3e-3 in head where reg-only gradient flows** |

Only H3-alt satisfies the joint cat+reg objective. The other configurations each fix one objective at the cost of the other. This is the "killer chart" of the attribution chain.

---

## 8 · What the paper now claims

CH18 reframes from "MTL trails STL by 12–14 pp on reg" to:

> **MTL with per-head LR (cat=1e-3, reg=3e-3, shared=1e-3, all constant) preserves cat performance and lifts the GETNext-hard reg head by 6.7-15 pp over single-LR MTL across 3 states. On the smallest state (AL), MTL exceeds the STL ceiling by 6.25 pp. The mechanism: the graph-prior weight α in `next_getnext_hard` requires sustained high LR to grow; per-head LR isolates this requirement from the cat-stability requirement, which needs gentle LR upstream.**

This is a paper-strength MTL-over-STL claim AND a clean attribution mechanism, supported by three negative controls that bracket the H3-alt design as unique in its space.

---

## 9 · What we can still evolve from here

The H3-alt recipe is paper-ready, but several open directions could strengthen or extend it:

### Near-term (in scope of current paper)

1. **F37 STL FL ceiling** — currently 4050-assigned; the cleanest "MTL exceeds STL on FL" comparison needs the matched-head STL run. Once we have it, we'll know whether AL's "+6.25 pp surplus" pattern repeats at FL scale, or whether FL is structurally tighter.
2. **Seed sweep on H3-alt** — 5-fold σ is solid but the recipe was tested at seed 42 only. A 3-seed sweep ({0, 7, 100} on AL+AZ) would harden the σ confidence intervals.
3. **Wilcoxon paired test** — paired test for H3-alt vs B3 across folds would give the formal statistical strength claim (similar to the F27 cat-head Wilcoxon at p=0.0312).

### Mid-term (paper extensions)

4. **OneCycleLR per-head variant** — current recipe uses constant LR. A per-head OneCycleLR (cat max=1e-3, reg max=3e-3, shared max=1e-3) might yield even tighter reg σ on AZ via late annealing, without giving up the α growth window.
5. **Three-state validation at 5-fold** — FL ran successfully with batch=1024 (~4.3h). Re-run AZ at the same batch for σ-tightening, then compare cat F1 and reg Acc@10 with paired tests.
6. **Combined loss + LR recipe** — F40 alone failed; H3-alt alone wins. A combined recipe (per-head LR + scheduled cat_weight) might yield marginal additional reg lift if cat over-converges in the early epochs and freeing its weight late helps reg further. Low expected value but cheap to test.

### Long-term (follow-up paper or larger headline)

7. **CA + TX upstream + 5-fold H3-alt** — the headline paper covers FL+CA+TX. CA/TX upstream pipelines are not yet built. Once they land, H3-alt becomes the universal recipe for the headline table.
8. **Mechanism deepening** — instrument α directly (log α value per epoch per fold) to confirm the "α growth → reg lift" mechanism quantitatively, not just by inference. The F45/F48-H1 per-fold reg-best-epoch data already supports this story; explicit α traces would close it.
9. **Transfer to other MTL backbones** — the per-head LR mechanism should generalize beyond `MTLnetCrossAttn`. CGC/PLE/MMoE/DSelectK MTL variants could be retrofitted with `cat_specific_parameters` / `reg_specific_parameters` and re-validated. If H3-alt's pattern holds across architectures, it's a more general MTL contribution than just "a recipe for our specific model."
10. **Beyond GETNext-hard** — α-style learnable graph-prior weights appear in many recent POI models (HMT-GRN, MGCL, etc.). The per-head LR mechanism may be a general tool for any MTL model with a parameter that needs sustained high LR to grow. Worth a survey paper or a follow-up methods note.

---

## 10 · The shape of the journey

Looking back, the trajectory has three phases:

**Architecture search (B-M → B3):** which backbone, which loss, which heads. Roughly Phase B5 plus B3 emergence. Outcome: `mtlnet_crossattn + static_weight(0.75) + next_gru + next_getnext_hard`. Static, well-trodden ML choices.

**Reframing crisis (F21c):** matched-head STL beat MTL by 12-14 pp on reg. Forced honest reckoning — the MTL coupling wasn't adding value on the region head with the canonical training recipe.

**Mechanism + recovery (F38–F48):** systematic refutation of plausible factors (checkpoint, weights, encoder, epochs, OneCycleLR magnitude) → identification of α growth as the load-bearing reg lift mechanism (F45) → identification of cat-vs-shared-cross-attn destabilisation as the cat collapse mechanism (F48-H3) → per-head LR recipe (F48-H3-alt) → bracketing controls (F40, F48-H2) confirming uniqueness.

The lesson is **negative controls are paper material**. The H3-alt recipe alone is "we found a thing that works." H3-alt + F40 + F48-H2 + F48-H3 is "we found the unique thing in this design space and the others are publishable refutations of the obvious alternatives." That is a stronger paper.

The work is not done — the open directions above continue the line — but the central CH18 claim has flipped from a methodological limitation to a paper-grade MTL-over-STL contribution with a clean mechanism. That's the inflection point this document records.

---

## Cross-references

- `NORTH_STAR.md` — current committed config (predecessor B3 + champion candidate H3-alt)
- `CLAIMS_AND_HYPOTHESES.md §CH18` — formal claim, now Tier A
- `FOLLOWUPS_TRACKER.md §F37–F48` — per-experiment status and acceptance criteria
- `research/F44_F48_LR_REGIME_FINDINGS.md` — the LR regime sweep
- `research/F48_H3_PER_HEAD_LR_FINDINGS.md` — H3 + H3-alt + FL scale
- `research/F40_SCHEDULED_HANDOVER_FINDINGS.md` — loss-side negative control
- `research/F48_H2_WARMUP_CONSTANT_FINDINGS.md` — warmup-plateau negative control
- `research/F21C_FINDINGS.md` — the original gap finding that triggered the chain
- `research/F27_CATHEAD_FINDINGS.md` — the cat-head refinement that locked B3
- `research/B5_RESULTS.md` — the B-M → B3 emergence
