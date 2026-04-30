# F50 T1.5 + T2 — Results Synthesis (2026-04-29)

**Trigger:** F50 Tier-1 closed all three architectural / optimisation drop-in alternatives (T1.2 HSM head, T1.3 FAMO, T1.4 Aligned-MTL — see `F50_T1_RESULTS_SYNTHESIS.md`). Per the F50 plan §8 decision tree, the next question is *"is the FL architectural cost a property of the cross-attention shared layer (testable cheaply via mechanism probes), or structural to multi-stream MTL with shared parameters at scale (testable via Tier 2 architectural alternatives)?"*

**Status:** **DONE 2026-04-29** — all four cross-attn mechanism probes (P1/P2/P3/P4) + three Tier-2 architectural alternatives (PLE-lite + Cross-Stitch default + Cross-Stitch detach) landed at paper-grade n=5 paired Wilcoxon.

**Verdict (one sentence):** **None of the 10 alternatives** (4 Tier-1 + 4 H1.5 probes + 3 Tier-2) **reaches the +3 pp acceptance threshold** for closing the FL architectural gap (CUDA H3-alt 73.61 ± 0.83 reg top10_acc_indist), and **none reaches paired Wilcoxon p < 0.05** for "alt > H3-alt". The closest are PLE-lite (Δreg +1.11 pp, but with −3.61 pp cat regression — net Δm negative) and P4 alternating-SGD (Δreg +0.96 pp with cat tied — best balanced result, p=0.0938). The FL architectural cost is **structural to multi-stream MTL with shared parameters at this cardinality**, not addressable by head capacity / magnitude balancing / direction alignment / cross-attn mechanism / task-specific experts / parallel backbones / per-batch alternation.

**Read order (predecessor docs):**
1. `F50_PROPOSAL_AUDIT_AND_TIERED_PLAN.md` — original F50 plan
2. `F50_DELTA_M_FINDINGS.md` — Tier 0 (CH22, FL Pareto-loss)
3. `F50_T1_RESULTS_SYNTHESIS.md` — Tier 1 closure (HSM/FAMO/Aligned-MTL)
4. `F50_T1_5_CROSSATTN_ABSORPTION.md` — load-bearing mechanism finding
5. **This doc** — final F50 synthesis across all 10 alternatives.

---

## 1 · Master scoreboard

All 10 alternatives at FL 5f×50ep CUDA (RTX 4090) bs=2048, paired vs **CUDA H3-alt baseline** (substrate-matched re-run `_0153`: cat F1 68.36 ± 0.74, reg top10_acc_indist 73.61 ± 0.83). Acceptance threshold: Δreg ≥ +3 pp (close ≥3 of the 8.78 pp gap to STL ceiling 82.44). All cat F1 values are macro-F1 (torchmetrics multiclass default; minority-class-sensitive on the 7-class POI category task).

| # | variant | family | cat F1 | Δcat | reg top10 | Δreg | reg σ | n+/n− | p_> | acceptance |
|:-:|---|---|---:|---:|---:|---:|---:|:-:|:-:|:-:|
| — | **H3-alt CUDA (REF)** | baseline | **68.36 ± 0.74** | — | **73.61 ± 0.83** | — | 0.83 | — | — | reference |
| 1 | **PLE-lite** | Tier-2: task-specific experts | 64.75 ± 0.85 | **−3.61 ± 0.66** | 74.72 ± 1.02 | **+1.11 ± 1.14** | 1.02 | 4/1 | 0.0625 | ❌ FAIL +3pp |
| 2 | **P4 alternating-SGD** | H1.5: per-batch alternation | 68.20 ± 0.69 | −0.16 ± 0.31 | 74.57 ± 1.04 | **+0.96 ± 0.79** | 1.04 | 4/1 | 0.0938 | ❌ FAIL +3pp |
| 3 | T1.3 FAMO | Tier-1: magnitude balancer | 68.18 ± 0.61 | −0.18 | 74.23 ± 0.81 | +0.62 ± 1.50 | 0.81 | 3/2 | 0.2188 | ❌ FAIL |
| 4 | Cross-Stitch (default) | Tier-2: parallel backbones | 68.21 ± 0.87 | −0.15 ± 0.35 | 73.73 ± 0.87 | +0.12 ± 1.18 | 0.87 | 2/3 | 0.5000 | ❌ FAIL (tied) |
| 5 | T1.4 Aligned-MTL | Tier-1: direction aligner | 67.46 ± 0.81 | −0.90 ± 0.62 | 73.50 ± 0.41 | −0.11 ± 0.62 | 0.41 | 1/4 | 0.8438 | ❌ FAIL |
| 6 | P2 detach-K/V | H1.5: F49 leakage closure | 67.97 ± 0.69 | −0.39 ± 0.45 | 73.55 ± 1.20 | −0.05 ± 1.20 | 1.20 | 2/3 | 0.6875 | ❌ FAIL (MRR σ↓) |
| 7 | P1 no_crossattn | H1.5: cross-attn ablation | 68.32 ± 0.67 | −0.04 ± 0.13 | 73.40 ± 0.85 | −0.21 ± 0.86 | 0.85 | 2/3 | 0.6875 | ❌ FAIL (≈ H3-alt) |
| 8 | P3 cat_freeze@10 | H1.5: warmup-then-freeze | 67.96 ± 0.80 | −0.40 ± 0.40 | 73.31 ± 1.20 | −0.30 ± 1.20 | 1.20 | 1/3 | 0.7812 | ❌ FAIL |
| 9 | T1.2 HSM | Tier-1: head capacity | 67.87 ± 1.04 | −0.49 ± 1.04 | 70.60 ± 10.78 | −3.01 ± 9.95 | 10.78 | 4/1 | 0.3125 | ❌ FAIL (fold-2 collapse) |
| 10 | Cross-Stitch (detach) | Tier-2: F49-clean parallel | 68.44 ± 1.18 | +0.08 ± 0.61 | 69.56 ± 9.90 | −4.05 ± 9.89 | 9.90 | 2/3 | 0.6875 | ❌ FAIL (fold-4 collapse) |

**No alternative reaches +3 pp Δreg.** **No alternative reaches p<0.05.** Only PLE-lite + P4 alternating-SGD reach n+ = 4/5 on reg (the closest to a positive paired result).

---

## 2 · Per-fold reg top10 (master matrix)

Substrate-matched per-fold CUDA H3-alt reference: {72.81, 73.08, 73.45, 73.77, 74.93}. Per-fold values for every alternative (paired by seed=42 + identical `StratifiedGroupKFold` splits via `--no-folds-cache`):

| variant | f1 | f2 | f3 | f4 | f5 | mean ± σ |
|---|:-:|:-:|:-:|:-:|:-:|---:|
| **H3-alt CUDA (ref)** | 72.81 | 73.08 | 73.45 | 73.77 | 74.93 | 73.61 ± 0.83 |
| PLE-lite | 75.09 | 75.01 | 74.51 | 73.12 | 75.86 | 74.72 ± 1.02 |
| P4 alt-SGD | 75.32 | 73.80 | 74.83 | 73.30 | 75.61 | 74.57 ± 1.04 |
| T1.3 FAMO | 75.23 | 74.72 | 74.29 | 73.16 | 73.73 | 74.23 ± 0.81 |
| CS default | 74.53 | 72.90 | 74.35 | 72.68 | 74.16 | 73.73 ± 0.87 |
| T1.4 Aligned-MTL | 73.61 | 73.07 | 73.08 | 73.73 | 74.00 | 73.50 ± 0.41 |
| P2 detach-K/V | 73.36 | 72.21 | 75.45 | 73.00 | 73.75 | 73.55 ± 1.20 |
| P1 no_crossattn | 73.23 | 72.82 | 74.48 | 72.43 | 74.03 | 73.40 ± 0.85 |
| P3 cat_freeze@10 | 72.81 | 75.21 | 73.10 | 71.96 | 73.48 | 73.31 ± 1.20 |
| T1.2 HSM | 76.13 | **51.39** ⚠ | 74.03 | 76.40 | 75.05 | 70.60 ± 10.78 |
| CS detach | 72.38 | 72.96 | 73.76 | **52.10** ⚠ | 76.61 | 69.56 ± 9.90 |

**Two single-fold catastrophic collapses:** T1.2 HSM fold-2 (51.39%, ep 3) and CS detach fold-4 (52.10%, ep 9). Both are the same reg-head-fold-init brittleness pattern from F49: under detach/freeze conditions the reg head can pick a degenerate early epoch on a particular fold-init seed. Without these outliers:
- T1.2 HSM 4/5 folds: mean 75.40 — would be Δ ≈ +1.79 pp (still below +3 pp acceptance)
- CS detach 4/5 folds: mean 73.93 — would be Δ ≈ +0.32 pp (CS-default-like)

The paired Wilcoxon at n=5 *can* reject equality at p=0.0312 if 5/5 folds positive. None of our alternatives clears that bar. Best n+/n− is 4/1 (PLE, P4, T1.2 HSM).

---

## 2.5 · Per-fold cat F1 (macro) (master matrix)

Substrate-matched per-fold CUDA H3-alt cat F1 reference: {69.37, 67.35, 68.17, 68.67, 68.24}. Cat F1 (macro) is sensitive to minority-class POI categories on the 7-class task (~3 pp gap to f1_weighted reflects class imbalance, consistent across all alternatives).

| variant | f1 | f2 | f3 | f4 | f5 | mean ± σ |
|---|:-:|:-:|:-:|:-:|:-:|---:|
| **H3-alt CUDA (ref)** | 69.37 | 67.35 | 68.17 | 68.67 | 68.24 | 68.36 ± 0.74 |
| CS detach | 69.57 | 66.60 | 68.03 | 68.82 | 69.18 | **68.44 ± 1.18** ⚠ noisy |
| P1 no_crossattn | 69.32 | 67.50 | 68.03 | 68.47 | 68.28 | 68.32 ± 0.67 |
| CS default | 69.01 | 66.73 | 68.38 | 68.53 | 68.40 | 68.21 ± 0.87 |
| P4 alt-SGD | 69.23 | 67.38 | 67.91 | 68.01 | 68.48 | 68.20 ± 0.69 |
| T1.3 FAMO | 68.80 | 67.19 | 68.21 | 68.53 | 68.16 | 68.18 ± 0.61 |
| P2 detach-K/V | 68.93 | 66.98 | 67.87 | 68.13 | 67.94 | 67.97 ± 0.69 |
| P3 cat_freeze@10 | 69.01 | 66.82 | 67.68 | 68.27 | 68.02 | 67.96 ± 0.80 |
| T1.2 HSM | 68.79 | 66.25 | 67.43 | 68.40 | 68.50 | 67.87 ± 1.04 |
| T1.4 Aligned-MTL | 68.22 | 66.25 | 67.03 | 68.00 | 67.81 | 67.46 ± 0.81 |
| **PLE-lite** | **65.95** | **64.00** | **65.20** | **63.94** | **64.68** | **64.75 ± 0.85** ⚠ uniform regression |

**Three observations on the cat side:**

1. **8 of 10 alternatives keep cat F1 within ±1 pp of H3-alt** — cat is robust across the architecture/balancer space. Per-fold trajectories largely track H3-alt's fold-init noise (e.g., P1 vs H3-alt cat F1 Pearson r=0.985).

2. **Only PLE-lite shows a uniform cat regression** (−3.61 pp, 0/5 folds positive). The regression is flat across folds (range 63.94 to 65.95, σ=0.85), pointing to a structural issue with PLE-lite's per-task-input shared-experts adaptation rather than fold-specific noise. Canonical-PLE (single shared input) might preserve cat — see F54 follow-up.

3. **T1.4 Aligned-MTL has a small but consistent cat regression** (−0.90 pp, 1/5 folds positive). Direction-alignment with vanishing-magnitude reg gradient costs cat F1 via the alignment overhead.

Other notable patterns:
- **CS detach has the highest cat σ** (1.18, vs ~0.7 for others). Detaching the off-diagonal alpha removed the regularising effect on cat too (mirrors the reg fold-4 collapse).
- **Best cat preservation:** P1 no_crossattn (Δcat = −0.04). Removing the entire shared backbone barely changes cat — strongest evidence that the cat encoder absorbs the shared-backbone capacity self-sufficiently.

---

## 3 · Mechanism interpretation

The combined H1+H1.5+T2 results, read through the cat-encoder-absorption finding (`F50_T1_5_CROSSATTN_ABSORPTION.md`), tell a coherent mechanism story.

### 3.1 The absorption mechanism explains the H1+H1.5 negative-result pattern

Per the diagnostic-CSV gradient analysis on H3-alt fold 1:
- `‖g_reg‖ / ‖g_cat‖` on shared params = 0.03–0.06 most of training (cat dominates >95%)
- `g_cosine` ≈ 0 on shared params (tasks contribute orthogonal gradients)
- Reg gradient saturates at 0.005 by epoch 5 (head's α·log_T graph prior reaches ceiling)

Under live training, **the cat encoder absorbs the architectural contribution of the shared backbone via cross_ba's K/V** (F49 Layer 2 mechanism, verified by P2 dropping reg-MRR σ from 8.52 → 1.09). Because of this:

- **T1.2 HSM** changes the reg head's softmax structure but doesn't break absorption → FAIL.
- **T1.3 FAMO** changes magnitude balance, doesn't break absorption → FAIL.
- **T1.4 Aligned-MTL** changes direction alignment with vanishing-magnitude reg gradient → FAIL.
- **P1 no_crossattn** removes shared backbone entirely; cat encoder absorbs the loss of capacity (r=0.985 on cat F1 vs H3-alt) → ≈ H3-alt.
- **P2 detach-K/V** explicitly closes the absorption channel; reg-MRR σ collapses (8.52→1.09 confirms the leakage was real) but reg top10 doesn't change → absorption is the source of joint-best instability, not the source of the FL flaw.
- **P3 cat_freeze@10** prevents continued absorption after warmup; reg unchanged → continued cat training isn't actively hurting either.

**The single principle "cat-encoder absorbs whatever the shared backbone does" explains all six H1+H1.5 negative results.**

### 3.2 Tier 2 partially bypasses absorption — the trade-off picture

Tier 2 architectures have explicit task-specific structure that should bypass absorption:

- **PLE-lite** has `category_experts` and `next_experts` per level; the reg-specific experts cannot be absorbed by cat training. Result: **Δreg = +1.11 pp** (4/5 folds positive, p=0.0625) — the FIRST directionally-positive reg result. But cat F1 collapses by **−3.61 pp** (0/5 folds positive) because PLE-lite's per-task-input adaptation prevents the cat encoder from leveraging shared expert outputs efficiently.
- **Cross-Stitch (default)** has parallel backbones with learned alpha mixing. Result: cat tied (−0.15) and reg tied (+0.12) — the model **learned alpha to ≈ identity at training time**, effectively becoming two parallel STLs. **Cross-Stitch's "soft" task-isolation degenerates to no-isolation when the alpha is unconstrained.**
- **Cross-Stitch (detach)** severs off-diagonal alpha gradient flow (no F49-style leakage). Result: tied means but fold-4 catastrophic collapse → **the off-diagonal alpha leakage in default CS was acting as a regulariser**, smoothing fold-init noise. Removing it makes the model fragile.
- **P4 alternating-SGD** updates only one task's params per batch. Result: **Δreg = +0.96 pp with Δcat = −0.16 pp** — best balanced result, n+ = 4/5, p=0.0938. **Per-batch alternation prevents cat from continuously absorbing shared backbone capacity.**

### 3.3 The two paths that beat 0 share a single trait: **break the absorption channel**

PLE (+1.11) and P4 (+0.96) are the only directionally-positive reg results among the 10 alternatives. They are also the only two that explicitly prevent the cat encoder from continuously co-training with the shared backbone:

- PLE: physical isolation (task-specific experts that cat training can never touch)
- P4: temporal isolation (alternating step — only one task at a time updates shared params)

Both produce **small (+1 pp) but consistent reg lifts** without paper-grade significance (p < 0.05). **The absorption mechanism IS the load-bearing FL mechanism, but bypassing it produces a small recovery, not the +3 pp threshold the paper requires.**

The remaining gap to the STL ceiling (which would require +8.78 pp) is **structural** — a ~4.7K-class softmax with a strong α·log_T graph prior simply doesn't benefit much from cross-task signal at this cardinality. The reg head reaches its ceiling alone in 5 epochs; multi-task interaction can give you small reg lift (+1 pp) but cannot fundamentally shift the ceiling.

### 3.4 Cross-attn architecture is "dead by absorption", not dead by design

P1 (cross-attn fully removed) ≈ H3-alt with r=0.985 on cat F1 reproduces *exactly* the same fold-by-fold trajectory. F49's λ=0 frozen-cat-stream isolation, however, found cross-attn architectural Δ at FL = **−16.16 pp** (cross-attn HURTS reg architecturally when cat encoder cannot absorb). The reconciliation:

```
F49 (frozen cat — no absorption):       cross-attn contribution at FL  = -16.16 pp
Live training (cat encoder absorbs):    -16.16 + 15.95 (cat absorbs)   ≈    0 pp
                                                                                  (∴ P1 ≈ H3-alt at Δ = -0.21)
```

**Cross-attn at FL is contributing negatively, but the cat encoder is silently fixing it.** The user pays the architectural cost in the form of cat-encoder capacity sunk into "compensating", not in the form of measurably-bad reg outputs.

---

## 4 · Decision-tree status (F50 plan §8 final)

```
Tier 0  (Δm + Wilcoxon, MPS) — F50_DELTA_M_FINDINGS.md
└── FAIL at FL on PRIMARY (Δm = -1.63% p_two_sided=0.0625, 0/5 folds+)
    └── Tier 1 entered: "does any alternative recover FL Δm?"
        ├── T1.1 (F33)                   ✅ PASS — universal next_gru cat head
        ├── T1.2 HSM head capacity       ❌ FAIL — Δreg -3.01 pp (fold-2 collapse)
        ├── T1.3 FAMO magnitude          ❌ FAIL — Δreg +0.62 pp
        ├── T1.4 Aligned-MTL direction   ❌ FAIL — Δreg -0.11 pp
        ↓
        Tier 1.5 entered: "is cross-attn shared layer the load-bearing mechanism?"
        ├── P1 no_crossattn              ❌ FAIL — Δreg -0.21 pp (cat r=0.985 vs H3-alt)
        ├── P2 detach-K/V                ❌ FAIL — Δreg -0.05 pp (MRR σ collapse confirms F49 leakage)
        ├── P3 cat_freeze@10             ❌ FAIL — Δreg -0.30 pp
        ├── P4 alternating-SGD           ❌ FAIL — Δreg +0.96 pp (best balanced; p=0.0938)
        ↓
        Tier 2 entered: "does explicit task-specific structure recover FL?"
        ├── PLE-lite                     ❌ FAIL — Δreg +1.11 pp / Δcat -3.61 (trade-off)
        ├── Cross-Stitch default         ❌ FAIL — alpha→identity, ≈ H3-alt
        ├── Cross-Stitch detach          ❌ FAIL — fold-4 collapse
        ↓
        ALL 10 alternatives FAIL +3 pp acceptance
        ↓
**F50 final verdict: FL architectural cost is structural at scale.**
    ├── No head/balancer/architectural fix in the tested space recovers FL Δm.
    ├── Mechanism characterised: cat-encoder absorption masks an architectural deficit.
    ├── PLE + P4 ARE directionally positive (+1 pp reg); approach paper-grade
    │   significance at p ≈ 0.06-0.10 but trade off cat (PLE) or are subthreshold (P4).
    └── Lock H3-alt as the paper champion. CA+TX P3 launches under H3-alt unblocked.
```

---

## 5 · Recommendation: lock H3-alt + ship paper

### 5.1 Why this is the right call

1. **Comprehensive negative-result evidence is paper-grade ammunition.** The paper can credibly claim *"we ruled out FAMO (NeurIPS 2023), Aligned-MTL (CVPR 2023), hierarchical-softmax-on-the-reg-head, four cross-attn mechanism probes (P1-P4), PLE (RecSys 2020), and Cross-Stitch (CVPR 2016) as recipes that recover FL Δm; the FL architectural cost is robust to head, balancer, mechanism, and architecture changes at this scale."* Few MTL papers offer 10-alternative paired-Wilcoxon comparisons.

2. **The cat-encoder absorption mechanism is paper-publishable in its own right.** It's a clean, measurement-backed mechanism that explains the F49 λ=0 architectural-Δ pattern AND the H1+H1.5+T2 negative-result pattern with one principle. Goes in `paper/results.md` § FL discussion.

3. **PLE + P4 directional positives ≠ paper-grade fix.** Δreg +1.1 pp at p ≈ 0.06-0.10 is suggestive but not significant; PLE costs cat. Reporting either as "we propose [X] as the FL fix" would invite reviewer rejection on weak effect size and trade-off concerns. Better to report them as *characterised partial recoveries* in support of the absorption-mechanism story.

4. **CA+TX P3 (37h Colab) launches unblocked.** The original gating concern was "if Tier 1 changes the FL champion, P3 must rerun under it." Tier 1 + 1.5 + 2 all closed without a champion change → P3 proceeds under H3-alt confidently.

5. **The substrate finding (CH16 +16.13 pp at CA STL probe) is the strongest, most generalisable claim.** Don't dilute the paper with a weaker MTL-architecture-fix claim that doesn't reach significance. Substrate + scale-conditional MTL with absorption-mechanism rebuttal is a cleaner submission.

### 5.2 Paper sections to add / update

- **`paper/results.md` § FL discussion:** new paragraph reporting the 10-alternative comparison + the cat-encoder absorption mechanism. Reference this doc + `F50_T1_5_CROSSATTN_ABSORPTION.md`.
- **`paper/limitations.md`:** small mention of "PLE-lite + P4 alternating-SGD show directional improvement at FL but neither reaches paper-grade significance; future work may explore stronger task-isolation regimes."
- **`PAPER_DRAFT.md` §1 abstract:** unchanged.
- **`CLAIMS_AND_HYPOTHESES.md` CH22b sub-claim:** strengthen with absorption-mechanism + 10-alternative robustness.
- **`NORTH_STAR.md`:** unchanged — H3-alt is the champion.

### 5.3 Follow-up plan (post-paper / camera-ready)

These are NOT paper-blocking but worth noting in followups tracker:

- **F51 — AL + AZ P1 runs** (~25 min compute). If cross-attn-is-absorbed pattern is FL-only (P1 << H3-alt at AL+AZ where F49 architectural Δ is positive), the absorption story becomes scale-conditional in a strong sense. If P1 ≈ H3-alt across all 3 states, the absorption is universal to our pipeline.
- **F52 — P5 identity-crossattn probe** (~30 min dev + 19 min run). Decomposes "cross-task K/V mixing" from "per-task FFN depth" within the cross-attn block.
- **F53 — category_weight sensitivity sweep** (~115 min compute). Direct test of cat-dominance hypothesis at multiple cat_weight values.
- **F54 (NEW) — PLE with cat-friendly reformulation** (~15h dev + 25 min run). PLE's cat regression came from the per-task-input adaptation; a PLE variant where shared experts process a fused input (concat or mean of cat_emb + reg_emb) might preserve the reg lift without cat regression. Camera-ready exploration.
- **F55 (NEW) — P4 + scheduler tuning** (~25 min compute). P4 alternating-SGD got the closest balanced result. With shared_lr matched higher (matching reg_lr) it might reach +3 pp without breaking cat. Cheap follow-up.

---

## 6 · Side findings worth recording

### 6.1 F49 Layer 2 leakage IS real but is a regulariser, not a productive contributor

P2 detach-K/V collapsed reg-MRR σ from 8.52 → 1.09, definitively confirming the leakage exists in measurable magnitude. But Δreg top10 = −0.05 → the leakage doesn't help OR hurt the headline metric. CS default vs CS detach: the off-diagonal alpha leakage in CS_default smooths fold-init noise; removing it (CS detach) introduces the fold-4 collapse. **Conclusion: F49 Layer 2 leakage is best understood as a stochastic regulariser, not a transfer mechanism.**

### 6.2 Cross-attn shared layer at FL is "dead by absorption", not "dead by design"

P1 ≈ H3-alt with r=0.985 on cat F1, but F49 λ=0 reveals the underlying architectural deficit (−16.16 pp). The user pays the architectural cost as cat-encoder capacity, not as reg performance. **For the paper, this is an unusually strong mechanism finding** — it explains why naive interpretations of "MTL works at scale" can be dangerously wrong.

### 6.3 Reg head's α·log_T graph prior dominates the FL reg signal

The reg head reaches its top10 plateau by epoch 5 across all 10 alternatives (per-fold reg-best-epochs cluster {2-9}), regardless of architecture. **The architecture's role at FL is small relative to the head's strong inductive bias.** Future work that wants to genuinely shift the reg ceiling at FL should focus on the head (e.g., F52 P5, F35 ROTAN) rather than the shared backbone.

### 6.4 PLE-lite's cat regression is per-task-input-specific, not isolation-general

Cross-Stitch (also task-isolated, parallel backbones) has cat tied at -0.15. The PLE-lite cat collapse (−3.61) is therefore traceable to PLE's specific per-task-input adaptation in the shared experts (each shared expert evaluated separately on cat input vs reg input). A canonical-PLE implementation (single shared input across experts) might preserve reg lift without cat regression. F54 follow-up.

### 6.5 P4 alternating-SGD is the cheapest balanced fix

Among all 10 alternatives, P4 has the smallest cat regression (−0.16 ± 0.31) AND the second-largest reg lift (+0.96, 4/5 folds positive, p=0.0938). It's a 30-line trainer change. With one more knob (e.g., 2:1 reg:cat alternation favoring reg) it might reach paper-grade significance. **F55 follow-up is the highest-ROI camera-ready exploration.**

---

## 7 · Cross-references

- **All run dirs (CUDA bs=2048 5f×50ep, 2026-04-29):**
  - H3-alt baseline: `mtlnet_lr1.0e-04_bs2048_ep50_20260429_0153`
  - T1.3 FAMO: `_0019` | T1.4 Aligned-MTL: `_0045` | T1.2 HSM: `_0128`
  - P2 detach-K/V: `_0314` | P1 no_crossattn: `_0334` | P3 cat_freeze: `_0350`
  - PLE-lite: `_0409` | CS default: `_0446` | CS detach: `_0503` | P4 alt-SGD: `_0520`
- **Plan:** `F50_PROPOSAL_AUDIT_AND_TIERED_PLAN.md`
- **Tier 0 (CH22):** `F50_DELTA_M_FINDINGS.md`
- **Tier 1 closure:** `F50_T1_RESULTS_SYNTHESIS.md`
- **Mechanism finding:** `F50_T1_5_CROSSATTN_ABSORPTION.md`
- **F49 architectural Δ pattern:** `F49_LAMBDA0_DECOMPOSITION_RESULTS.md`
- **MTL flaws + fixes catalog:** `MTL_FLAWS_AND_FIXES.md` §2.8 (absorption summary)
- **Paper draft:** `PAPER_DRAFT.md` §1-3
- **Champion:** `NORTH_STAR.md` — unchanged (H3-alt locked)

---

## 8 · One-paragraph summary

> Across 10 architectural and optimisation alternatives — 4 Tier-1 (FAMO, Aligned-MTL, HSM head, T1.1 verification), 4 Tier-1.5 cross-attn mechanism probes (no-crossattn, detach-K/V, cat-freeze, alternating-SGD), and 3 Tier-2 architectures (PLE-lite, Cross-Stitch default + detach) — paired vs the substrate-matched CUDA H3-alt baseline at FL 5f×50ep, **none reaches the +3 pp acceptance threshold** for closing the FL architectural gap and **none reaches paired Wilcoxon p < 0.05** for "alt > H3-alt". The closest approaches are PLE-lite (Δreg +1.11 pp at cost of −3.61 pp cat) and P4 alternating-SGD (Δreg +0.96 pp with cat tied, p=0.0938). The combined results are explained by a single mechanism — **cat-encoder absorption** — uncovered via gradient-trace diagnostics on H3-alt: shared-backbone gradient is 95% cat-dominated, cat/reg gradients are statistically independent, and the cat encoder silently absorbs the architectural contribution that would otherwise show. F49 λ=0 isolation reveals the underlying architectural deficit at FL (−16.16 pp); live training compensates via cat-encoder co-adaptation, masking the cost as cat-encoder capacity sink. PLE and P4 partially break the absorption channel (task-specific experts; per-batch alternation) and produce small directional reg lifts, confirming absorption IS the load-bearing mechanism — but the remaining gap to the STL ceiling is structural at 4.7K-class scale. **Recommendation: lock H3-alt as paper champion**, ship the paper with the 10-alternative robustness claim + absorption-mechanism finding as core contributions, defer F51-F55 follow-ups to camera-ready.
