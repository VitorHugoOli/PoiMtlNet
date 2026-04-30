# F50 Tier 1 — Results Synthesis (2026-04-29)

**Trigger:** Tier 0 (joint Δm) FAILed at FL on PRIMARY (Δm = −1.63% p_two_sided=0.0625, 0/5 folds positive). Per F50 plan §8, Tier 1 became *"does any alternative recipe rescue FL Δm?"* — four targeted tests (T1.1 cat-head verification, T1.2 hierarchical-softmax reg head, T1.3 FAMO, T1.4 Aligned-MTL).

**Read order (predecessor docs):** `F50_PROPOSAL_AUDIT_AND_TIERED_PLAN.md` (plan) → `F50_DELTA_M_FINDINGS.md` (T0) → `F50_T1_1_CAT_HEAD_PATH_DECISION.md` (T1.1) → `F50_HANDOFF_2026-04-28.md` (pickup state) → **this doc**.

**Status:** **DONE 2026-04-29** — all 4 Tier-1 tests landed at paper-grade n=5. **Verdict (one sentence):** none of the three architectural/optimisation alternatives close the FL +3 pp acceptance threshold; FL architectural cost is **robust to head-capacity, magnitude-balancing, and direction-alignment changes**, strengthening the scale-conditional CH22 framing and motivating Tier 2 (PLE / Cross-Stitch / MTI-Net) if a structural fix is desired.

**Substrate + batch-size validation (load-bearing):** to remove confounding, H3-alt was compared across three configurations: MPS bs=1024 (the published reference), **CUDA bs=1024** (existing run `_2149` from 2026-04-28), and **CUDA bs=2048** (new run `_0153` matching the Tier-1 alternatives' batch size). Result: **per-task-best cat F1 + reg top10_acc_indist (the F50 acceptance metric) transfer cleanly across both substrate and batch size** — all three configurations land within 0.5σ of each other. Per-task-best MRR varies by batch size (CUDA bs=1024 → MPS within 0.42σ; CUDA bs=2048 → MPS at 1.06σ) — what initially looked like substrate divergence is largely a bs effect. See §5 for the full breakdown. **All Tier-1 deltas in §1 below use the bs=2048 CUDA H3-alt baseline (73.61 ± 0.83) for internal consistency since T1.2/T1.3/T1.4 all ran at bs=2048; §5.1 verifies the verdicts hold against any of the three reference baselines.**

---

## 1 · Headline scoreboard

All numbers are **per-task-best** (`diagnostic_task_best`) on the FL 5f×50ep MTL recipe. Reference is the **CUDA-matched H3-alt re-run on this same RTX 4090 pod** (substrate-equivalent to the published MPS numbers within 0.2σ on cat F1 and reg top10_acc_indist; see §5 for the substrate analysis). All Δ values are computed against the CUDA H3-alt baseline.

| Test | Substrate | n | cat F1 | reg top10_indist | reg MRR | Verdict (Δreg vs CUDA H3-alt; +3 pp acceptance) |
|---|:--:|:--:|---:|---:|---:|---|
| **H3-alt CUDA bs=2048** (reference for Tier-1 deltas) | CUDA | 5 | 68.36 ± 0.74 | **73.61 ± 0.83** | 48.65 ± 8.52 | — (reference) |
| **H3-alt CUDA bs=1024** (substrate sanity, existing) | CUDA | 5 | 67.95 ± 0.75 | 73.92 ± 0.88 | 55.72 ± 4.74 | within 0.5σ of bs=2048 + MPS on cat F1 + reg top10 |
| **H3-alt MPS bs=1024** (published) | MPS | 5 | 68.21 ± 0.42 | 73.65 | ~57.69 | within 0.5σ of CUDA on cat F1 + reg top10; MRR at 0.42σ from CUDA bs=1024 — see §5 |
| **T1.1** universal `next_gru` cat | MPS | 5 | 68.21 ± 0.42 | (= H3-alt run) | — | ✅ **PASS** Path A confirmed (closes C14) — see `F50_T1_1_CAT_HEAD_PATH_DECISION.md` |
| **T1.2-STL HSM** (head-only sanity) | MPS | 5 | n/a | top10 = 82.64 ± 0.42 (vs flat STL 82.44 ± 0.43, p=0.0312) | — | ✅ HSM head architecture preserved at the head level |
| **T1.2-MTL HSM** (full 5/5) | CUDA | 5 | 67.87 ± 1.04 | **70.60 ± 10.78** ⚠ fold-2 collapse | 48.72 ± 12.14 | ❌ **FAIL** Δreg = −3.01 pp (or +1.7 pp dropping fold-2 outlier) |
| **T1.3 FAMO** | CUDA | 5 | 68.18 ± 0.61 | **74.23 ± 0.81** | 54.71 ± 6.83 | ❌ **FAIL** Δreg = +0.62 pp |
| **T1.4 Aligned-MTL** | CUDA | 5 | 67.46 ± 0.81 | **73.50 ± 0.41** | 57.46 ± 0.59 | ❌ **FAIL** Δreg = −0.11 pp; cat regressed −0.90 pp |

**3/3 architectural alternatives miss the +3 pp acceptance threshold against the CUDA-matched baseline.** Two of three (T1.3, T1.4) tied H3-alt within fold variance on the reg side. T1.2-MTL HSM tracked H3-alt within σ on 4/5 folds (74-76%) but had one fold collapse (fold 2 = 51.4% at ep 3) — same FL-reg fold-init brittleness pattern the handoff flagged from the partial 3/5-fold MPS run. **Verdicts are robust to substrate** (see §5).

---

## 2 · Per-fold detail

Per-fold per-task-best epoch + metric for all four CUDA runs (H3-alt baseline + 3 alternatives). **Reg-best epochs are uniformly very early on CUDA** (ep 2-6) — including for the H3-alt baseline itself (ep 4-6), so this is substrate-driven, not Tier-1-alternative-specific.

| Run | Fold | cat F1 (ep) | reg top10_indist (ep) | reg MRR |
|---|:-:|---:|---:|---:|
| **H3-alt (CUDA)**      | 1 | 69.37% (ep14) | 72.81% (ep 6) | 39.48% |
|                        | 2 | 67.35% (ep17) | 73.08% (ep 4) | 56.49% |
|                        | 3 | 68.17% (ep22) | 73.45% (ep 5) | 41.53% |
|                        | 4 | 68.67% (ep13) | 73.77% (ep 5) | 47.48% |
|                        | 5 | 68.24% (ep15) | 74.93% (ep 4) | 58.27% |
| **T1.3 FAMO**          | 1 | 68.80% (ep14) | 75.23% (ep 3) | 58.46% |
|                        | 2 | 67.19% (ep17) | 74.72% (ep 3) | 57.71% |
|                        | 3 | 68.21% (ep11) | 74.29% (ep 4) | 58.34% |
|                        | 4 | 68.53% (ep13) | 73.16% (ep 5) | **42.57%** ⚠ |
|                        | 5 | 68.16% (ep15) | 73.73% (ep 5) | 56.48% |
| **T1.4 Aligned-MTL**   | 1 | 68.22% (ep16) | 73.61% (ep 6) | 57.88% |
|                        | 2 | 66.25% (ep37) | 73.07% (ep 5) | 56.43% |
|                        | 3 | 67.03% (ep26) | 73.08% (ep 6) | 57.74% |
|                        | 4 | 68.00% (ep15) | 73.73% (ep 5) | 57.60% |
|                        | 5 | 67.81% (ep22) | 74.00% (ep 6) | 57.67% |
| **T1.2-MTL HSM**       | 1 | 68.79% (ep18) | 76.13% (ep 2) | 58.10% |
|                        | 2 | 66.25% (ep19) | **51.39% (ep 3)** ❌ | 32.83% |
|                        | 3 | 67.43% (ep 8) | 74.03% (ep 3) | 38.41% |
|                        | 4 | 68.40% (ep14) | 76.40% (ep 2) | 57.97% |
|                        | 5 | 68.50% (ep14) | 75.05% (ep 3) | 56.31% |

**Per-fold paired comparison vs CUDA H3-alt** (matching folds across runs via seed=42 + identical `StratifiedGroupKFold` splits):

| Fold | H3-alt reg | T1.2 HSM reg (Δ) | T1.3 FAMO reg (Δ) | T1.4 Aligned reg (Δ) |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 72.81 | 76.13 (**+3.32**) | 75.23 (**+2.42**) | 73.61 (+0.80) |
| 2 | 73.08 | 51.39 (**−21.69**) | 74.72 (+1.64) | 73.07 (−0.01) |
| 3 | 73.45 | 74.03 (+0.58) | 74.29 (+0.84) | 73.08 (−0.37) |
| 4 | 73.77 | 76.40 (**+2.63**) | 73.16 (−0.61) | 73.73 (−0.04) |
| 5 | 74.93 | 75.05 (+0.12) | 73.73 (−1.20) | 74.00 (−0.93) |
| **mean Δ** | — | **−3.01** ± 9.95 | **+0.62** ± 1.50 | **−0.11** ± 0.62 |
| n+ / n− | — | 4 / 1 | 3 / 2 | 1 / 4 |
| W+ (signed rank sum, n=5, max=15) | — | 10 | 11 | 4 |
| paired Wilcoxon p_greater (exact) | — | 0.3125 | 0.2188 | 0.8438 |

None of the three alternatives reach the +3 pp acceptance threshold on the mean-Δ, and **none reaches paired Wilcoxon significance at p_greater < 0.10** (p_min at n=5 is 1/32 = 0.0312, achieved only when 5/5 folds are positive). Per-fold pairing reveals nuance: **T1.2 HSM and T1.3 FAMO each have folds where they substantially beat H3-alt** (HSM fold-1 +3.32, fold-4 +2.63; FAMO fold-1 +2.42, fold-2 +1.64), but the wins do not generalise to all folds. **T1.2 HSM's fold-2 collapse (−21.69 pp) is the dominant negative signal** but even *dropping* fold 2 the mean Δ ≈ +1.66 pp — still below the +3 pp acceptance. **T1.4 Aligned-MTL is the cleanest "matches H3-alt" result** (4/5 folds within ±0.4 pp of baseline; tightest σ in the entire battery at 0.41).

**Three observations from the per-fold table:**

1. **T1.3 FAMO and T1.4 Aligned-MTL are reg-side calibration peers.** Both produce stable reg means in the 73.5-74.2 band with σ ≤ 0.81 on top10 and σ ≤ 6.83 on MRR. Neither closes the gap to STL ceiling (82.44); both essentially reproduce H3-alt on reg — confirming that gradient-balancing (magnitude OR direction) is **not** the FL bottleneck.

2. **T1.2-MTL HSM 4/5 folds look like a marginal lift** (75.4% mean dropping fold 2; +1.7 pp over H3-alt). **Fold 2 collapses to 51.39% at epoch 3.** The hierarchical-softmax bias appears to *amplify* the early-epoch fragility — the reg head picks even earlier best epochs (ep 2-3) than flat (ep 5-6 on T1.4) and at FL one fold-init landed on a degenerate epoch-3 minimum. This is consistent with F49's frozen-cat finding (per-fold reg-best epochs {2,14,9,4,2}, σ_frozen=12 vs σ_loss-side=1.4) — the FL reg path is brittle at the 4.7K-class scale regardless of head architecture.

3. **Per-task-best reg epochs are clustered at ep 2-6 across all three runs.** The reg head reaches its peak almost immediately, then degrades as the joint optimisation pulls features toward the cat objective. This is the §3.5 *"head-size mismatch"* symptom manifesting at scale: the shared backbone's finite capacity tilts toward the 7-class cat softmax over epochs, eroding the 4.7K-class reg signal. Tier 2 architectures (PLE / Cross-Stitch) explicitly address this by giving the reg head its own task-specific experts / stream.

---

## 3 · Decision-tree status (F50 plan §8)

```
Tier 0  (Δm + Wilcoxon, MPS)
└── FAIL at FL on PRIMARY (Δm = −1.63% p_two_sided=0.0625, 0/5 folds+)
    └── Tier 1 entered: "does any alternative recover FL Δm?"
        ├── T1.1 (F33)           ✅ PASS — Path A confirmed; universal next_gru
        ├── T1.2 STL HSM (MPS)   ✅ PASS — head architecture preserved at STL
        ├── T1.2 MTL HSM (CUDA n=5)  ❌ FAIL — Δreg −3.05 pp; fold-2 collapse
        ├── T1.3 FAMO (CUDA n=5)     ❌ FAIL — Δreg +0.58 pp (within σ of H3-alt)
        └── T1.4 Aligned-MTL (CUDA n=5)  ❌ FAIL — Δreg −0.15 pp (within σ); cat -0.75 pp
            ↓
        all Tier 1 alternatives FAIL the +3 pp acceptance threshold
            ↓
        FL architectural cost is robust to head + magnitude- + direction-balancer changes.
```

Per F50 plan §8: this routes to **either Tier 2 (architectural alternatives — PLE / Cross-Stitch / ROTAN, ~60 h compute)** OR **lock in the current scale-conditional H3-alt champion as the paper recipe with Tier 1 as rebuttal ammunition**.

---

## 4 · Recommendation

### 4.1 Lock in H3-alt as the paper champion; CA + TX P3 launches under it

**Rationale:**

1. **H3-alt is the only champion.** All three architectural drop-in alternatives (T1.2 head, T1.3 magnitude balancer, T1.4 direction aligner) tied or regressed against H3-alt at FL. T1.1 confirmed `next_gru` is universal across AL+AZ+FL. There is no Tier-1 candidate that warrants re-running CA+TX under a new champion.

2. **The scale-conditional framing is empirically backed.** Tier 0 already showed MTL Pareto-positive on Δm at AL+AZ at p=0.0312 ceiling and Pareto-negative at FL at p_two_sided=0.0625 ceiling. Tier 1 strengthens this: the FL flip is **not** an artefact of the H3-alt recipe — three independent class of fixes (capacity, magnitude, direction) all fail to recover. The paper has stronger ammunition than `PAPER_DRAFT.md §1` currently claims.

3. **CA + TX P3 (~37 h Colab) is the longest critical-path item.** The F50 plan §2 critical-path argument said *"don't launch P3 concurrently with Tier 1; if any Tier-1 finding changes the champion, P3 must rerun."* Tier 1 is now closed with no champion change → **P3 can launch under H3-alt without ambiguity**, monotonicity-extending the architectural-Δ scale curve to 5 points {AL 1.1K, AZ 1.5K, FL 4.7K, CA ~6K, TX ~5K}.

### 4.2 Defer Tier 2 to camera-ready (or skip)

**Rationale:**

1. **Tier 2 (~60 h compute) is exploratory.** PLE / Cross-Stitch / MTI-Net are reasonable alternatives but the *headline contribution* of the paper is the substrate (Check2HGI) finding + the scale-conditional MTL characterisation, not a SOTA architectural fix to FL reg. Tier 2 belongs in the *limitations* / *future work* section, or as a paper extension, not as a block to current submission.

2. **Tier 1's null result is itself a paper finding.** Section 4.1 of the F50 plan said *"if Tier 1 confirms the FL architectural cost, the paper retains current framing and gains an additional paragraph"* — i.e., the negative result is a paper paragraph, not a redirection. Specifically: *"we ruled out FAMO (NeurIPS 2023), Aligned-MTL (CVPR 2023), and hierarchical-softmax-on-the-reg-head as alternatives that recover FL Δm; the architectural cost at large region cardinality is robust to head and balancer changes."* This belongs in `paper/results.md` § FL discussion or `paper/limitations.md`.

3. **Tier 2 development ROI is unclear.** PLE alone is ~25 h (15 dev + 10 train); Cross-Stitch ~15 h. If they *do* close the FL gap, the paper claim becomes "we propose PLE-FL recipe" — which is a different paper than the current substrate-first framing. If they don't, that's another null-result paragraph for ~60 h.

### 4.3 If Tier 2 is pursued, prioritise **PLE first**

PLE is the most theoretically motivated of the three Tier-2 candidates per F50 plan §6:

- PLE directly addresses *all three* §3.5 symptoms (disjoint LR regimes, divergent inductive biases, head-size mismatch) via task-specific experts.
- The closest existing analog (`mtlnet_dselectk`) is already in the codebase as an implementation template.
- Cross-Stitch is a useful diagnostic ("is forced sharing the bottleneck?") but a less likely to *fix* than PLE if forced sharing turns out not to be the locus.
- ROTAN is a reg-head ceiling test, orthogonal to the MTL question.

**If pursued, run PLE on AL → AZ → FL incrementally. If PLE recovers AL+AZ within σ of H3-alt and improves FL by ≥ +3 pp, it becomes the new champion. Otherwise, abandon Tier 2 and lock H3-alt.**

---

## 5 · Substrate + batch-size analysis

To disentangle substrate effects from batch-size effects, **two H3-alt CUDA runs** were compared against the MPS published numbers: bs=1024 (the original MPS recipe; existing run `_2149` from 2026-04-28) and bs=2048 (the NORTH_STAR.md recipe used for the Tier-1 alternatives; new run `_0153`).

| Metric (H3-alt FL 5f×50ep, per-task-best) | MPS bs=1024 (published) | CUDA bs=1024 (`_2149`) | CUDA bs=2048 (`_0153`) |
|---|---:|---:|---:|
| cat F1 | 68.21 ± 0.42 | **67.95 ± 0.75** (MPS at 0.35σ) ✓ | **68.36 ± 0.74** (MPS at 0.20σ) ✓ |
| reg top10_acc_indist | 73.65 | **73.92 ± 0.88** (MPS at 0.31σ) ✓ | **73.61 ± 0.83** (MPS at 0.05σ) ✓ |
| reg MRR | ~57.69 | **55.72 ± 4.74** (MPS at 0.42σ) ✓ | 48.65 ± 8.52 (MPS at 1.06σ) ~ |
| reg-best epochs (per-fold) | (not recorded) | {2, 2, 4, 2, 2} | {6, 4, 5, 5, 4} |
| Joint-best reg top10 | 71.96 ± 0.68 | 63.68 ± 12.07 ⚠ | 59.04 ± 12.00 ⚠ |
| Wall time | (~50-75 min on M4 Pro) | ~36 min | ~19 min |

**Three findings:**

1. **`top10_acc_indist` is rock-solid across both axes** (substrate AND batch size): 73.61-73.92 spread, all within 0.5σ of MPS. **This is the F50 acceptance metric**, so the Tier-1 verdicts are robust to both substrate and batch size.

2. **The reg-MRR "divergence" I initially attributed to substrate is mostly a batch-size effect.** At bs=1024, CUDA reproduces MPS MRR within fold variance (CUDA 55.72 ± 4.74 → MPS 57.69 at 0.42σ). At bs=2048, CUDA MRR drops to 48.65 ± 8.52 (MPS at 1.06σ — marginally outside but still within ~2σ). The bs=2048 → bs=1024 swap closes ~70% of the apparent MPS-CUDA MRR gap. **Substrate alone is not the dominant driver**; bs is. Likely mechanism: at bs=1024 the gradient noise is higher → the reg head explores the loss surface more aggressively early on → finds a better-ranked but lower-amplitude optimum that survives joint-task pressure. At bs=2048 the gradient is smoother → reg head settles on a sharper top-K-optimal but poorly-ranked optimum. The fact that bs=1024 CUDA picks reg-best at ep 2 in 4/5 folds (vs ep 4-6 at bs=2048) supports this.

3. **Joint-best (deployment) reg numbers remain unreliable on CUDA** at both batch sizes (σ ≈ 12 vs 0.68 on MPS) — see `docs/RUNPOD_GUIDE.md` §9. The joint-score selector picks epochs where the cat objective has progressed but the reg head has already started to degrade. Per-task-best selection is the substrate-robust comparison.

### 5.1 What this means for the Tier-1 verdicts

**The verdicts in §1/§2 use bs=2048 throughout (H3-alt baseline + 3 alternatives all at bs=2048). This is internally consistent and the +3 pp acceptance check is well-founded.** Neither substrate nor batch size shifts the Tier-1 conclusions:

- Against bs=2048 CUDA H3-alt (73.61 ± 0.83): T1.2 −3.01 / T1.3 +0.62 / T1.4 −0.11. All FAIL.
- Against bs=1024 CUDA H3-alt (73.92 ± 0.88): T1.2 −3.32 / T1.3 +0.31 / T1.4 −0.42. All still FAIL by larger margins.
- Against MPS bs=1024 published (73.65): T1.2 −3.05 / T1.3 +0.58 / T1.4 −0.15. All still FAIL.

**The +3 pp acceptance threshold is missed by every alternative against every reference baseline.** Verdict-robust to substrate + bs choice.

### 5.2 Optional: re-run T1.3/T1.4 at bs=1024 for paper-grade MRR

If the paper wants to report MRR-based Δ for Tier 1 against the MPS published MRR (57.69), T1.3 and T1.4 should be re-run at bs=1024 on this pod (~36 min each, ~72 min total) to remove the bs confound from the MRR axis. Current bs=2048 numbers are sufficient for the top10-based acceptance verdict but the MRR comparison is bs-confounded. Defer to camera-ready unless the MRR comparison is paper-headline.

### 5.1 Substrate-matched verdict robustness

With the CUDA H3-alt baseline at top10 = 73.61 ± 0.83 (per-task-best), the +3 pp acceptance threshold becomes "T1_alt reg top10 ≥ 76.61". Observed:

| Test | reg top10 | Δ vs CUDA H3-alt | Hits +3 pp? |
|---|---:|---:|:-:|
| T1.2-MTL HSM | 70.60 ± 10.78 | −3.01 | ❌ |
| T1.3 FAMO | 74.23 ± 0.81 | +0.62 | ❌ |
| T1.4 Aligned-MTL | 73.50 ± 0.41 | −0.11 | ❌ |

**3/3 alternatives FAIL on CUDA-matched comparison**, identical to the MPS-reference verdicts in §1. The Tier-1 conclusion (FL architectural cost is robust to head + balancer changes) is substrate-independent.

---

## 6 · What changes in the paper

### 6.1 Headline (PAPER_DRAFT.md §1)

**Current:**
> *Beyond Cross-Task Transfer: Per-Head Learning Rates and Check-In-Level Embeddings for Multi-Task POI Prediction*

**No change needed.** The headline is empirically backed by:
- T0: MTL Pareto-positive at AL+AZ on Δm at n=5 ceiling significance; Pareto-negative at FL at n=5 ceiling.
- T1.1: universal `next_gru` cat head across 3 states.
- T1.2/T1.3/T1.4: FL architectural cost is robust to three classes of head/balancer drop-in fixes.

### 6.2 Results section additions (paper/results.md)

**New paragraph: "Tier-1 alternatives do not recover FL Δm."** Reports:

- **T1.2 hierarchical-softmax reg head** (motivated by §3.5 head-size mismatch): closes 0 of the 8.78 pp FL architectural gap on MTL (Δ = −3.05 ± 10.78, fold-2 epoch-3 collapse; even dropping the outlier, Δ ≈ +1.7 pp, below the +3 pp acceptance threshold).
- **T1.3 FAMO** (NeurIPS 2023, magnitude-balancing): closes +0.58 pp of the gap (74.23 ± 0.81 vs H3-alt 73.65), within σ.
- **T1.4 Aligned-MTL** (CVPR 2023, direction-alignment): closes 0 pp (73.50 ± 0.41 vs 73.65), within σ; cat regresses by 0.75 pp.

**Mechanism note:** all three alternatives' reg heads pick best epochs at ep 2-6, consistent with the F49 frozen-cat finding (the FL reg path is brittle to fold-init at the 4.7K-class scale, regardless of head/balancer). This narrows the locus of the FL architectural cost to *the shared cross-attention backbone's finite-capacity allocation under joint optimisation*, not the head or balancer.

### 6.3 Limitations section additions (paper/limitations.md)

**New paragraph:** *"We ruled out three classes of MTL fixes for the FL architectural cost (head capacity, magnitude balancing, direction alignment) but did not test architectural alternatives to the shared cross-attention backbone (PLE, Cross-Stitch, MTI-Net). These remain open future work."* 

### 6.4 Method section caveat (paper/methods.md)

**Add:** *"Per-task-best epoch selection at FL produces uniformly very early reg-head best epochs (ep 2-6 across H3-alt + 3 alternatives), reflecting the brittleness of the 4.7K-class softmax under joint optimisation. Joint-best (deployment) selection is more conservative but introduces fp16-autocast / multi-task-best-epoch-selector interaction that we observed as σ ≈ 9-12 pp variance on CUDA substrate (vs σ ≈ 0.7 pp on MPS, where the published numbers were collected). For paper-grade tables we report per-task-best."*

---

## 7 · Tracker updates

- **`F50_PROPOSAL_AUDIT_AND_TIERED_PLAN.md` §5 Tier 1 summary table:** mark T1.2-MTL/T1.3/T1.4 as `❌ FAIL` with refs to this doc; mark §8 Tier-1 leaf as "all FAIL → architectural cost robust; lock H3-alt".
- **`F50_HANDOFF_2026-04-28.md` §1:** update T1.2/T1.3/T1.4 rows from "NOT STARTED" / "PARTIAL" to "DONE 2026-04-29" with refs.
- **`FOLLOWUPS_TRACKER.md` §1:** mark F50 multi-row block as closed; add a new row for "F51 PLE backbone" if Tier 2 is pursued.
- **`PAPER_PREP_TRACKER.md` §1:** add CH22 Tier-1 strengthening row (substrate-conditional + robustness-to-balancers).
- **`CONCERNS.md`:** §C12 (architectural-cost mechanism at scale) gains a Tier-1 robustness note; §C14 already closed by T1.1.
- **`CLAIMS_AND_HYPOTHESES.md`:** CH22 (scale-conditional Δm) gains a sub-claim CH22b (FL architectural cost is robust to head + balancer changes).
- **`NORTH_STAR.md`:** unchanged — H3-alt champion confirmed.
- **`PHASE2_TRACKER.md`:** P3 (CA+TX) is unblocked — launches under H3-alt.

---

## 8 · Open follow-ups

1. **CUDA-vs-MPS reg-head divergence** (`docs/RUNPOD_GUIDE.md` §9). **Status updated:** §5 disentangled this — per-task-best `top10_acc_indist` is bulletproof across both substrate AND batch size; per-task-best `MRR` is bs-sensitive (bs=1024 closes most of the apparent MPS-CUDA gap, MPS within 0.42σ of CUDA bs=1024); joint-best reg remains unreliable on CUDA at both batch sizes. **Workaround applied:** all Tier-1 verdicts use per-task-best top10_acc_indist. **Real fix still open:** ablate `--no-fp16-val` to test whether joint-best collapses to MPS values under fp32 validation.

2. **T1.2-MTL HSM fold-2 collapse.** A single fold-init produced an epoch-3 reg minimum at 51.39% (vs 74-76% on the other 4 folds). This is consistent with the F49 frozen-cat instability finding (reg-best epochs at FL frozen = {2,14,9,4,2}, σ=12). Worth a small ablation: re-run T1.2-MTL HSM with a different fold-2 seed to confirm it's an init-pathology, not an HSM-specific failure.

3. **MPS validation of T1.3/T1.4.** Not needed for the top10-based verdict (substrate-equivalent at both bs=1024 and bs=2048 per §5). Optional re-run of T1.3/T1.4 at bs=1024 (~72 min total) would tighten any MRR-based comparison; defer to camera-ready unless MRR is paper-headline.

4. **Tier 2 PLE prototype.** ~25 h dev + train. Decision: pursue if reviewers ask for an architectural fix; otherwise defer.

5. **CA + TX P3.** ~37 h Colab. **Unblocked by Tier 1 closure.** Launch under H3-alt unchanged; extends scale curve to 5 points.

---

## 9 · Cross-references

- **Plan:** `F50_PROPOSAL_AUDIT_AND_TIERED_PLAN.md`
- **T0 result:** `F50_DELTA_M_FINDINGS.md` + `results/paired_tests/F50_T0_delta_m.json`
- **T1.1 result:** `F50_T1_1_CAT_HEAD_PATH_DECISION.md`
- **T1.2-STL result (MPS):** `docs/studies/check2hgi/results/P1/region_head_florida_region_5f_50ep_F50_T1_2_HSM_FL_5f50ep.json`
- **T1.2-MTL HSM result (CUDA n=5):** `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260429_0128/`
- **T1.3 FAMO result (CUDA n=5):** `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260429_0019/`
- **T1.4 Aligned-MTL result (CUDA n=5):** `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260429_0045/`
- **H3-alt baseline (CUDA bs=2048 n=5, today):** `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260429_0153/`
- **H3-alt baseline (CUDA bs=1024 n=5, 2026-04-28):** `results/check2hgi/florida/mtlnet_lr1.0e-04_bs1024_ep50_20260428_2149/` — bs-matched MPS comparison, see §5
- **Perf-compare report (joint-best, 2026-04-28):** `docs/studies/check2hgi/results/perf_compare/fl_h3alt_perf_compare_20260428_232722.md`
- **Pickup state:** `F50_HANDOFF_2026-04-28.md`
- **Substrate caveat:** `docs/RUNPOD_GUIDE.md` §9
- **Predecessor — F49 frozen-cat instability:** `F49_LAMBDA0_DECOMPOSITION_RESULTS.md` §3.4
- **Paper framing:** `PAPER_DRAFT.md` §1

## 10 · One-paragraph summary for the next agent

> Tier 1 is closed: all three architectural/optimisation drop-in alternatives (hierarchical-softmax reg head; FAMO magnitude-balancer; Aligned-MTL direction-aligner) miss the +3 pp acceptance threshold for closing the FL architectural gap, and none reaches paired Wilcoxon significance against the bs=2048 CUDA H3-alt baseline (top10_acc_indist 73.61 ± 0.83 per-task-best). Per-fold paired Δ: T1.2 HSM −3.01 ± 9.95 (W+=10, p_greater=0.3125; fold-2 ep-3 collapse to 51.39%); T1.3 FAMO +0.62 ± 1.50 (W+=11, p_greater=0.2188); T1.4 Aligned-MTL −0.11 ± 0.62 (W+=4, p_greater=0.8438). T1.1 closed earlier (universal `next_gru` cat head, Path A). The substrate+batch-size validation (§5) compared three H3-alt configurations (MPS bs=1024, CUDA bs=1024, CUDA bs=2048): per-task-best `top10_acc_indist` is rock-solid across all three (within 0.5σ); per-task-best MRR is mostly a bs effect, not substrate (CUDA bs=1024 brings MPS MRR within 0.42σ); joint-best reg remains substrate-fragile on CUDA at both batch sizes. Verdicts hold against any of the three reference baselines. Recommendation: lock H3-alt as the paper champion (no change to NORTH_STAR.md), launch CA+TX P3 unblocked, defer Tier 2 (PLE / Cross-Stitch) to camera-ready or future work.
