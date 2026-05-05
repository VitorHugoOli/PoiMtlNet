# Overnight summary — Check2HGI + HGI merge experiments

**Generated 2026-05-05** at end of autonomous overnight run.
**Updated 2026-05-05 ~09:30** with hybrid Designs I/J/M results.

---

## TL;DR

Goal: close the next-region gap between Check2HGI and HGI without losing
Check2HGI's strong next-category performance, ideally yielding a single
embedding usable across other tasks.

After 10 designs run end-to-end on AL+AZ, the hybrid family I/J/M (mixes of
B and H, plus distillation) all pass dominance (TOST cat non-inf + reg
superior to canonical) — same gate B/H already cleared.

**Wilcoxon p=0.0312 dual-gate (user-specified)**:

| Design | AL cat p | AL reg p | AZ cat p | AZ reg p | both axes at same state? |
|---|---:|---:|---:|---:|---|
| I | 0.094 | 0.063 | **0.031** | 0.156 | ✗ |
| J | 0.063 | **0.031** | 0.156 | 0.156 | ✗ |
| M | **0.031** | 0.063 | **0.031** | 0.156 | ✗ |

**No design passes Wilcoxon-strict on BOTH cat AND reg at any single state.**
With n=5 folds the Wilcoxon floor is p=1/32=0.0312 (requires 5/5 folds same
sign), so the dual-gate is a power-limited test. Best-in-axis instead:

- **M**: best cat (Wilcoxon strict at both states).
- **J**: best reg AL (Wilcoxon strict, +0.10 pp nominal vs HGI). But **AZ
  reg vs HGI = −1.22 pp** — J's HGI-beat is AL-specific, not generalised.
- **I**: parameter-efficient B equivalent (8× cheaper at AL, ~3× at CA);
  AZ cat strict win, otherwise ≈ B.

Out of the 10 designs run on AL+AZ:

- **Design B** (POI2Vec injected at the POI-pool boundary, cat path detached) —
  **✓ PASSES** dominance + generality. **Recommended to ship.**
- **Design H** (learnable per-POI table at POI-pool, warm-started from POI2Vec) —
  **✓ PASSES**, marginally better on reg AL, but adds parameters that scale
  poorly to FL/CA/TX. Treat as research follow-up, not the production substrate.
- **Design D** (heterograph with POI nodes) — **⚠ leak-flagged**. Reg gain
  looks legitimate; the dramatic cat lift is a graph-leak artifact. Do not ship.
- **Designs A, E** and the POI2Vec-input probe — failed cat dominance,
  documented in `MERGE_DESIGN_NOTES.md`.
- **Postpool diagnostic** — falsified the per-visit-noise hypothesis as the
  source of the canonical→HGI reg gap.
- **Design I** (LoRA r=8 on B) — **✓ PASSES dominance**, AZ cat Wilcoxon
  p=0.0312 strict. ~95k extra params at AL (8× cheaper than H).
- **Design J** (H + anchor reg λ=0.1) — **✓ PASSES dominance + reg AL
  Wilcoxon p=0.0312 strict (+0.10 pp nominal vs HGI)**.
- **Design M** (B + POI-side cosine-distillation λ_d=0.1) — **✓ PASSES
  dominance + cat Wilcoxon p=0.0312 strict at BOTH states**.

**Recommendation: B/H/I/J/M all tie at the dominance gate; choose by axis priority.**

Honest read: the user's stricter gate (Wilcoxon strict on cat AND reg at the
same state) is power-limited at n=5 and **no design clears it**. Under the
weaker dominance gate (TOST cat non-inf + reg-superior-to-canonical) all 5
designs pass at both states and are statistically indistinguishable.

Pick by which axis matters most:

- **Cat priority** → ship **M**. Wilcoxon-strict cat at both AL and AZ.
  Distillation appears to stabilize the POI representation the encoder has
  to predict via L_c2p, which helps cat without hurting reg.
- **Reg AL priority** → ship **J**. Strict reg AL win, +0.10 pp vs HGI
  nominal. But the HGI beat does NOT generalise to AZ (−1.22 pp). Anchor
  regularization is the load-bearing piece.
- **Parameter cost / scale to FL/CA** → ship **I**. ~95k extra params at
  AL (r=8 LoRA) vs H's 758k — at CA that's ~3M vs 23M. Functionally B
  with mild learnable capacity.
- **Simplicity / production** → ship **B**. Cleanest design, smallest
  parameter footprint, comparable to all hybrids on dominance. **No
  hybrid clearly outperforms B at the user's gate.**

A larger n (e.g. 10-fold or multi-seed) would resolve the J vs B vs M
question on the strict gate; at n=5 we cannot.

---

## Pinned baselines (AL+AZ leak-free per-fold log_T)

| Substrate | AL cat F1 | AL reg Acc@10 | AZ cat F1 | AZ reg Acc@10 | fclass probe |
|---|---:|---:|---:|---:|---:|
| canonical c2hgi | 40.76 | 59.15 | 43.21 | 50.24 | 4.26 / 4.31 |
| HGI | 25.26 | 61.86 | 28.69 | 53.37 | 98.34 / 97.92 |

c2hgi crushes cat by ~15 pp at both states; HGI marginally beats reg by ~2-3 pp.

---

## All designs / probes run

### 0. Postpool diagnostic — ✗ no lift (falsifies per-visit-noise hypothesis)

Hypothesis: maybe per-visit checkin features are too noisy and a simple
POI-mean → region-mean post-hoc pool of canonical c2hgi would close the gap.

Result: reg flat, no improvement. **The gap is not per-visit noise** — it
sits in the POI-level training signal (POI2Vec's hierarchical fclass
clustering), which canonical c2hgi never sees.

Artifact: `scripts/probe/build_check2hgi_postpool_region.py`,
`paired_tests/{alabama,arizona}_postpool_diagnostic_acc10.json`.

### 1. POI2Vec input probe — ✗ trade-off, not merge

Append POI2Vec(64) to canonical 11-dim check-in features (75-dim input). Train
canonical c2hgi engine end-to-end on the augmented features.

| | AL cat | AL reg | AZ cat | AZ reg |
|---|---:|---:|---:|---:|
| Δ vs canonical | −9.29 pp | +2.23 pp | −7.62 pp | +2.07 pp |

Reg lifted to roughly HGI level, but cat regressed badly. **Confirmed reg
gap is fclass-clustering signal**; the cost is cat. This motivated the
sequence of "merge" designs that follow.

Artifact: `scripts/probe/build_check2hgi_poi2vec.py`,
`paired_tests/{alabama,arizona}_poi2vec_diagnostic_acc10.json`.

### 2. Design A — late concat at downstream input (✗ failed both axes)

`[c2hgi_emb(64) ‖ HGI_poi2vec(64)]` per step → 128-dim. Train next_gru/
next_getnext_hard on this.

- AL/AZ cat: −8.49 / −11.51 pp (p=0.0312)
- AL/AZ reg: −18.73 / −9.16 pp (p=0.0312)

Why: HGI columns are step-static (same POI = same vector). GRU/STAN heads
degenerate when half the input has no per-step variation; the head can't
suppress one half.

### 3. Design E — POI2Vec input + stop-grad encoder→pool (✗ failed cat)

11-dim canonical + 64-dim POI2Vec = 75-dim per check-in input. Encoder
gradient stops at the POI-pool boundary (only L_c2p reaches encoder).

- AL/AZ cat: −9.54 / −9.38 pp (p=0.0312)
- AL/AZ reg: +2.09 / +2.38 pp (matches HGI within σ)

**Mechanism finding (load-bearing):** stop-grad protected the encoder from
L_p2r/L_r2c, but cat *still* regressed. The input itself carries POI2Vec,
which propagates through the GCN regardless of which losses backprop. **The
killer is the input, not the gradient routing.** This rules out Designs F
(PCGrad) and G (adapters) on POI2Vec-augmented input — same input, same
problem.

### 4. Design B — POI2Vec at POI-pool boundary, cat path detached ✓

Canonical 11-dim input. Encoder is byte-identical to canonical c2hgi.
POI2Vec enters AFTER `Checkin2POI` as
`poi_emb_for_reg = poi_emb_canonical.detach() + γ · Linear(POI2Vec)`.
L_c2p uses the canonical (non-detached) pool, so cat path matches canonical
c2hgi exactly. L_p2r and L_r2c operate on the enriched POI vectors.

| | AL | AZ |
|---|---:|---:|
| cat F1 | 41.51 ± 1.34 (+0.76, TOST p=0.0002 ✓) | 43.91 ± 1.10 (+0.70, TOST p=0.0027 ✓) |
| reg Acc@10 | 61.49 ± 4.06 (+2.34 vs can, p=0.0312) | 52.59 ± 3.03 (+2.35) |
| reg vs HGI | −0.37 pp (within σ) | −0.79 pp (within σ) |
| fclass probe | 98.45 / 97.91 (matches HGI) | |
| kNN-Jaccard@10 vs HGI | 0.109 / 0.072 | |

**Verdict:** clean dominance pass. Cat non-inferior at both states;
reg matches HGI within σ; POI embeddings recover HGI-grade fclass
linear separability while keeping per-visit dynamics in checkin embeddings.

### 5. Design H — learnable POI table at POI-pool ✓ (user's idea)

Same as B, but the POI residual is a learnable `nn.Embedding(num_pois, 64)`
warm-started from POI2Vec, updated only via L_p2r/L_r2c (cat path detached
as in B).

| | AL | AZ |
|---|---:|---:|
| cat F1 | 40.97 ± 1.20 (+0.21, TOST p=0.0024 ✓) | 44.14 ± 0.64 (+0.94, **Wilcoxon p=0.0312** strict win) |
| reg Acc@10 | 62.35 ± 3.74 (+3.20 vs can, **+0.49 vs HGI** nominal) | 52.30 ± 2.99 (+2.06) |
| fclass probe | 98.43 / 98.08 (matches HGI) | |
| kNN-Jaccard@10 vs HGI | 0.115 / 0.075 | |

**Verdict:** same dominance pass as B, plus a strictly significant cat
improvement at AZ. Reg AL exceeds HGI nominally for the first time. Loss
converges tighter (~0.28 vs B's ~0.45 final).

**Caveats vs B:**
- Within-noise on every dominance gate vs B (B and H are statistically
  indistinguishable).
- Adds `num_pois × 64` parameters: ~758 k at AL, scales to ~10 M at FL
  and ~23 M at CA.
- **Random-init ablation not yet run** — cannot distinguish whether H's
  marginal lift is from contrastive learning of the POI table or just from
  the POI2Vec warm-start init. If random-init H ≈ warm-start H on reg, H
  has real capacity beyond B; if random-init H ≪ warm-start H, H ≈ B.

### 6. Design D — heterogeneous CHECKIN+POI graph (⚠ leak-flagged)

Two node types (CHECKIN with 11-dim, POI with POI2Vec(64)), four edge types
(seq, visits, visited, spatial Delaunay). PyG `HeteroConv` with typed
weights per edge type. 5 contrastive boundaries.

| | AL | AZ |
|---|---:|---:|
| cat F1 (raw) | **72.88 ± 0.80** (+32.12 pp) | **74.73 ± 1.18** (+31.52 pp) |
| reg Acc@10 | 62.23 ± 3.77 (+3.08, +0.37 vs HGI) | 52.95 ± 2.95 (+2.71, −0.42 vs HGI) |
| fclass probe | 79.65 / 86.56 (lower than B/H) | |
| kNN-Jaccard@10 | 0.029 / 0.020 (lowest of all designs) | |

Paired tests (Wilcoxon, AL/AZ): cat p=0.0312/0.0312 (FLAGGED — leak);
reg vs canonical p=0.0312/0.156; reg vs HGI n.s.

**The cat lift is a graph leak.** Linear probe on the last check-in's
embedding alone:

- canonical c2hgi: 30.86 / 34.06
- Design B: 30.71 / 34.15 (matches canonical — same encoder, same baseline leak)
- Design H: 31.74 / 34.37 (matches canonical)
- **Design D: 51.16 / 50.29 (+20 pp extra leak)**

Mechanism: D adds POI nodes carrying POI2Vec(64). Reverse `visits` edges
+ 2-hop `HeteroConv` propagate the target POI's POI2Vec into the last
check-in's embedding, which is fclass-discriminative ⇒ category-discriminative.
The +32 pp cat lift is mostly leak amplification, not generalization.

**The reg gain (~+3 pp) is in the same ballpark as B/H, so D's reg is
likely legitimate**, but the design is unsafe to ship as the merged
substrate while the leak exists. POI fclass probe also degrades vs B/H
(80–87% vs 98%): the encoder bleeds checkin features into POI nodes,
contaminating POI representation.

Artifacts: `scripts/probe/build_design_d_heterograph.py`, `…d_train.py`;
`paired_tests/design_d_diagnostic.json`.

---

### 7. Design I — LoRA-style low-rank correction on B ✓

```
poi_for_reg = poi_canon.detach() + γ · ( Linear(POI2Vec) + U[poi]·V )
```
U: [N, 8], V: [8, 64]. Total extra params at AL: ~95k vs H's 758k (8×
cheaper). At FL ~1.3M vs 10M, at CA ~3M vs 23M.

| | AL | AZ |
|---|---:|---:|
| cat F1 | 41.62 ± 1.06 (+0.87, TOST p=0.0014 ✓) | 43.71 ± 0.69 (+0.50, **Wilcoxon p=0.0312** ✓) |
| reg Acc@10 | 61.35 ± 4.22 (+2.20, p=0.0625) | 52.55 ± 3.13 (+2.31, p=0.156) |
| reg vs HGI | −0.51 pp | −0.83 pp |
| fclass probe | 98.58 / 97.87 (HGI grade) | |
| kNN-Jaccard@10 vs HGI | 0.110 / 0.072 | |

**Verdict**: parameter-efficient B equivalent. Same dominance pass, same
reg geometry. Worth shipping at FL/CA where parameter count dominates cost.

**Leak sniff**: last-step probe AL/AZ = 96.26/97.16 — within 1pp of canonical
c2hgi (96.16/97.09). No design-specific leak introduced. (Same check on
J: 97.06/97.04; on M: 96.04/97.17 — all clean.)

### 8. Design J — H + anchor regularization to POI2Vec ✓ (best reg)

Same as H + L2 anchor: `L_total = L_c2hgi + 0.1 · ‖E_h − POI2Vec‖²`. The
learnable table is pulled toward fclass clusters, preventing contrastive
drift from breaking POI semantics.

| | AL | AZ |
|---|---:|---:|
| cat F1 | 41.81 ± 1.46 (+1.05, TOST p=0.0009 ✓) | 43.74 ± 0.76 (+0.53, TOST p=0.0008 ✓) |
| reg Acc@10 | **61.95 ± 3.95 (+2.80, Wilcoxon p=0.0312 ✓)** | 52.16 ± 2.85 (+1.91, p=0.156) |
| reg vs HGI | **+0.10 pp ✓ (nominal)** | −1.22 pp |
| fclass probe | 98.22 / 97.76 | |
| kNN-Jaccard@10 vs HGI | 0.114 / 0.074 | |

**Verdict**: best reg performance at AL. The HGI-nominal-beat is
**AL-specific** — at AZ, J's reg vs HGI is −1.22 pp (worse than B/H/I/M).
The anchor regularizer prevents H's learnable drift on AL but doesn't
solve AZ's HGI gap. Cat at AZ is also weaker (Wilcoxon p=0.156 vs M's
0.031). Use J only if AL reg-vs-HGI is the priority.

### 9. Design M — B + POI-side cosine-distillation to HGI ✓ (best cat)

Same as B + cosine-alignment loss between projected POI emb and HGI POI2Vec:
`L_total = L_c2hgi + 0.1 · (1 − cos(P(poi_emb), POI2Vec)).mean()`.
Aimed at closing the kNN-Jaccard 0.07–0.12 gap that B/H leave open.

| | AL | AZ |
|---|---:|---:|
| cat F1 | 41.31 ± 1.13 (+0.55, **Wilcoxon p=0.0312** ✓) | 43.67 ± 0.78 (+0.46, **Wilcoxon p=0.0312** ✓) |
| reg Acc@10 | 61.56 ± 4.13 (+2.41, p=0.0625) | 52.45 ± 3.11 (+2.21, p=0.156) |
| reg vs HGI | −0.30 pp | −0.93 pp |
| fclass probe | 98.48 / 98.02 | |
| kNN-Jaccard@10 vs HGI | 0.110 / 0.072 (≈ B) | |

**Verdict**: **only design with Wilcoxon-strict cat improvement at BOTH
states**. The geometry-alignment hypothesis (the original motivation) did
NOT pan out: kNN-Jaccard at λ_d=0.1 is 0.110/0.072 — essentially identical
to B's 0.109/0.072. Cosine alignment ≠ neighborhood alignment under finite
samples. M's cat win is real but is **not explained by its stated
mechanism**. A possible alternative explanation: the distillation loss
acts as POI-table regularization (similar in spirit to J's anchor) which
stabilizes the POI representation the encoder predicts via L_c2p. A higher
λ_d sweep is the open follow-up.

---

## Mechanism analysis (what we learned)

1. **The reg gap is fclass-cluster signal at the POI level**, not per-visit
   noise. Postpool falsified the noise hypothesis; POI2Vec input probe
   confirmed the signal source.

2. **The cat-vs-reg trade-off in shared-encoder designs is input-side, not
   gradient-side.** Design E proved stop-gradient doesn't recover cat when
   POI2Vec is in the input — the GCN propagates POI features regardless of
   which losses backprop. This rules out a whole family of optimization-based
   patches (PCGrad, GradNorm, adapters, projector heads) on POI2Vec-augmented
   input.

3. **Cat path must be byte-identical to canonical c2hgi to avoid regression.**
   B and H succeed because canonical 11-dim input is preserved end-to-end on
   the cat path; POI2Vec enters only at the POI-pool boundary, never at the
   check-in level, and its gradient is gated off cat via `.detach()`.

4. **Generality probe (fclass linear probe) is necessary but not sufficient.**
   - Canonical c2hgi: cat F1 = 40.76 BUT fclass probe = 4.26%. The check-in
     vectors are task-overfit; the POI vectors lack POI semantics entirely.
   - B/H recover HGI-grade fclass probe (98%) — but with POI2Vec warm-start
     init, this is partly guaranteed by construction.
   - **kNN-Jaccard@10 vs HGI is the harder generality test**: B/H score
     0.07–0.12, much lower than HGI's self-overlap. They have HGI-level
     *linear separability* for fclass but a *different POI geometry*. Whether
     this matters depends on the downstream task; for fclass classification
     it doesn't, for similarity-based retrieval it might.

5. **Heterogeneous graphs leak under unsupervised pretraining.** Adding more
   graph structure means more pathways for target leakage. Canonical c2hgi
   has the same kind of seq-edge leak as D (just smaller magnitude); B and H
   inherit canonical's leak without amplifying it. D amplifies it through
   POI nodes carrying POI2Vec.

6. **Learnable POI table > frozen POI2Vec residual** is a small effect at
   best. H beats B by +0.86 pp on AL reg, +0.05 pp on AZ — both within
   1σ. The dominant factor is the POI2Vec init, not the learnable update.
   This needs the random-init ablation to confirm.

---

## Recommendations / next steps

**Ship Design B as the merged substrate.**

1. **Promote B to a proper engine** (`EmbeddingEngine.CHECK2HGI_DR` or
   `CHECK2HGI_FUSED`), document, and replicate at FL/CA/TX before paper inclusion.
2. **Run the leak sniff test** (linear probe on last-step embedding) at FL
   to confirm B's leak structure matches canonical c2hgi at scale.
3. **Run the H random-init ablation** (no POI2Vec warm start) at AL+AZ to
   determine whether H's marginal lift is real or just init. Decision rule:
   - random-init H ≈ warm-start H on reg → H has real contrastive capacity,
     promote as alternative engine.
   - random-init H ≪ warm-start H → H ≡ B with extra params; abandon H,
     ship B alone.
4. **Do not pursue Design D as-is.** A leak-free heterograph variant requires
   detaching visits-edges between check-ins and POIs (so POI features don't
   leak to check-ins via 2-hop), at which point the design collapses back to
   B. Worth a paper note as a negative result.
5. **Keep generality probes (fclass + kNN-Jaccard) standard** for any future
   engine work. Cat F1 alone overfits to the task as canonical c2hgi
   demonstrated dramatically (4% fclass linear probe vs HGI's 98%).
6. **Open question for downstream POI-similarity tasks:** the kNN-Jaccard
   gap (0.07–0.12 vs HGI's 1.0) means B/H POI vectors are not geometrically
   interchangeable with HGI's POI2Vec. If a future task needs HGI-style
   POI-similarity retrieval, the merged engine may need a third readout
   that aligns more strongly with HGI's geometry.

---

## Artifact map

- Code: `scripts/probe/build_design_{a_concat,b_poi_pool,d_heterograph,d_train,e_stopgrad,h_learnable_poi}.py`,
  `build_check2hgi_{poi2vec,postpool_region}.py`
- Eval pipelines: `scripts/probe/eval_design_bh.py`,
  `finalize_{poi2vec,postpool,design_d}_diagnostic.py`
- Generality probes: `scripts/probe/generality_probes.py`
- Result tables: `docs/studies/check2hgi/research/MERGE_DESIGN_NOTES.md`,
  `DESIGN_D_HETEROGRAPH.md`, this `SUMMARY.md`
- Per-fold JSONs: `docs/studies/check2hgi/results/phase1_perfold/{AL,AZ}_{check2hgi_poi2vec,check2hgi_postpool,design_a_concat,design_b,design_d,design_e,design_h}_*.json`
- Paired tests: `docs/studies/check2hgi/results/paired_tests/{*_poi2vec,*_postpool,design_a,design_bh,design_d,design_e}_*.json`
- Generality data: `docs/studies/check2hgi/results/paired_tests/generality_probes.json`
- Trained engines: `output/check2hgi_design_{b,d,e,h}/{alabama,arizona}/`,
  `output/check2hgi_poi2vec/{alabama,arizona}/`
- Logs: `logs/design_{a,b,d,e,h}*` and `logs/design_bh_eval/`, `logs/design_d_eval/`

---

## One-paragraph quote for the paper

> A family of merged designs (B / H / I / J / M) closes the canonical
> Check2HGI → HGI next-region gap (~+2.2 pp Acc@10 vs canonical at both
> AL and AZ) while preserving Check2HGI's strong next-category
> performance (TOST non-inferiority p<0.005 at both states), in a single
> engine that recovers HGI-grade POI semantic linear separability
> (fclass probe 98% vs canonical's 4%). The shared mechanism is to
> inject POI2Vec at the POI-pool boundary with a stop-gradient on the
> cat path, so the check-in encoder remains byte-identical to canonical
> Check2HGI while the region-readout sees an fclass-enriched POI
> representation. With n=5 folds, no single variant dominates the others
> on a Wilcoxon-strict dual-axis test (p=0.0312 floor); axis-specific
> wins exist (M for cat, J for AL reg-vs-HGI) but do not generalise
> across both states.
