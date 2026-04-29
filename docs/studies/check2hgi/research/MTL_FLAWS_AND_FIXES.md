# MTL — Codebase Implementation, Empirical Flaws, and Candidate Fixes

**Created 2026-04-28 (post-F50 audit).** Living catalog. Consolidates what we know about how multi-task learning is wired in this codebase and what's empirically broken about it at FL scale.

This doc is the **single entry point** for any agent investigating MTL improvements on the check2HGI study. Companion docs:

- `MTL_ARCHITECTURE_JOURNEY.md` — narrative history (Phase B-M → F48-H3-alt → F49 → F37 → F50)
- `research/F50_PROPOSAL_AUDIT_AND_TIERED_PLAN.md` — tiered experimental plan currently in flight
- `research/F50_HANDOFF_2026-04-28.md` — pickup state per task
- `review/2026-04-28_critical_analysis.md` — framing tension + recommendation chain

---

## 1 · How MTL is implemented in this codebase

### 1.1 Architecture (`src/models/mtl/mtlnet_crossattn/model.py`)

**`MTLnetCrossAttn`** — bidirectional cross-attention between two task streams. Forward pass:

```
category_input  ──→ category_encoder (Linear+ReLU+LN+Dropout × n_layers) ──→ a
next_input      ──→ next_encoder (same)                                   ──→ b

for block in crossattn_blocks (default n=2):
    a_upd = cross_ab(Q=a, K=b, V=b)
    a = ln_a1(a + a_upd); a = ln_a2(a + ffn_a(a))
    b_upd = cross_ba(Q=b, K=a, V=a)             ← REG QUERIES (UPDATED) CAT AS K/V
    b = ln_b1(b + b_upd); b = ln_b2(b + ffn_b(b))

shared_cat = cat_final_ln(a)
shared_next = next_final_ln(b)

out_cat  = category_poi(shared_cat)        # cat head, e.g. next_gru
out_next = next_poi(shared_next)           # reg head, e.g. next_getnext_hard
```

The reg stream queries the cat stream's K/V every block. **F49's key finding is that this introduces silent gradient flow:** under loss-side `category_weight=0`, `L_reg` still propagates back to `category_encoder` through `cross_ba`'s K/V projections. The cat encoder is implicitly trained as a **reg-helper**, not a cat predictor. See `research/F49_LAMBDA0_DECOMPOSITION_GAP.md` for the gradient-flow proof.

### 1.2 Per-head LR optimizer (`src/training/helpers.py::setup_per_head_optimizer`)

Three param groups:

| Group | Members | Default LR (H3-alt) |
|---|---|---|
| `cat`    | `category_encoder` + `category_poi` (cat head) | 1e-3 |
| `reg`    | `next_encoder` + `next_poi` (reg head)        | 3e-3 |
| `shared` | crossattn blocks + cat_final_ln + next_final_ln | 1e-3 |

Partition exposed by `cat_specific_parameters() / reg_specific_parameters() / shared_parameters()` on `MTLnetCrossAttn`. **Disjoint optimal LR (3× ratio) is itself a symptom of head incompatibility** — see §2.2 below.

### 1.3 Aux side channel (`src/data/aux_side_channel.py`)

Thread-local `_publish_aux(...)` / `get_current_aux()` for passing per-batch tensors to the head WITHOUT touching the trainer's forward signature. Used by `next_getnext_hard` to deliver `last_region_idx` for the `α · log_T[last_region]` graph prior.

**Critical gating** in `src/data/folds.py:~894`:

```python
_HEADS_REQUIRING_AUX_MTL = {"next_getnext_hard", "next_getnext_hard_hsm"}
use_aux = task_b_head in _HEADS_REQUIRING_AUX_MTL
```

**Always extend this set** when adding a new head that consumes `last_region_idx`. The previous hardcoded equality check silently broke F50's HSM head (0% reg F1 across 38 epochs before discovery — see `F50_HANDOFF_2026-04-28.md §3`).

### 1.4 Loss balancing (`src/losses/registry.py`)

Available losses, all registered:

```
static_weight   (champion: category_weight=0.75)
nash_mtl        (NashMTL, ICML 2022)
pcgrad          (PCGrad, NeurIPS 2020)
gradnorm        (Chen et al. 2018)
cagrad          (Liu et al. 2021)
equal_weight    (Xin et al. baseline)
uncertainty_weighting   (Kendall & Gal 2018)
random_weight (rlw)     (Lin et al. 2022)
dwa             (Liu et al. 2019)
famo            (Liu et al. NeurIPS 2023) ← T1.3 candidate
aligned_mtl     (Senushkin et al. CVPR 2023) ← T1.4 candidate
db_mtl          (Lin et al. 2024)
go4align        (2024)
fairgrad        (2023)
excess_mtl      (2024)
bayesagg_mtl    (2024)
stch
uw_so
scheduled_static (loss-side ramping; F40)
```

The H3-alt champion uses `static_weight(category_weight=0.75)`. F50 T1.3/T1.4 swap this for FAMO / Aligned-MTL respectively.

### 1.5 Training loop (`src/training/runners/mtl_cv.py`)

- 5-fold StratifiedGroupKFold (user-disjoint), seed 42
- Joint optimization: `loss = mtl_criterion(loss_cat, loss_reg)`
- Per-task best epoch tracked (`diagnostic_best_epochs`) for paper reporting
- Joint best epoch tracked (`primary_checkpoint`) for "deployment" metrics
- Aux loader wrapping triggered by §1.3 set membership

### 1.6 Heads (`src/models/next/`)

Reg-side options registered (subset relevant to F50):

- `next_gru` — GRU recurrent + last-token softmax (cat-side champion post-F27)
- `next_lstm` / `next_tcn_residual` / `next_temporal_cnn` / `next_mtl` (Transformer)
- `next_stan` — Yang WWW'21 spatio-temporal attention
- `next_getnext` — soft graph prior (learned probe over `log_T`)
- `next_getnext_hard` ⭐ — hard graph prior (Yang SIGIR'22 faithful, `α · log_T[last_region]`)
- `next_getnext_hard_hsm` (NEW F50) — hierarchical-additive softmax variant
- `next_stahyper` / `next_tgstan` / `next_hybrid` — explored, none beat `next_getnext_hard`

The `α · log_T` mechanism is **load-bearing** at FL scale — it provides the bulk of the reg accuracy. STAN-only (without graph prior) under-trains. See `research/B5_HARD_VS_SOFT_INFERENCE.md`.

---

## 2 · Empirical flaws (chronological discovery)

### 2.1 The headline pattern: **architectural cost grows monotonically with region cardinality**

| State | n_regions | MTL H3-alt vs matched STL on reg | Architectural Δ (F49 frozen-cat λ=0) |
|---|---:|---:|---:|
| AL | 1,109 | **+6.25 pp** ✓ p=0.0312 | +6.48 pp ~2.7σ |
| AZ | 1,547 | −3.29 pp (n.s.) | −6.02 pp ~3.7σ |
| FL | 4,702 | **−8.78 pp** ✗ p=0.0312 | **−16.16 pp** p=0.0312 |

The architectural cost grows ~steeply with cardinality (extrapolation suggests CA ~6K regions ≈ −20 pp, TX ~5K ≈ −18 pp). **This is the central paper finding** — the MTL contribution is interactional + scale-conditional.

### 2.2 The three structural-incompatibility symptoms

1. **Disjoint optimal LR regimes (3× ratio)** — F40, F45, F48-H1, F48-H2 all confirmed no single LR/schedule works for both heads. H3-alt's `cat=1e-3, reg=3e-3, shared=1e-3` is a workaround that decouples optimization but doesn't align objectives.
2. **Divergent inductive biases** — cat head (GRU recurrent) vs reg head (STAN attention + graph prior). Forced to share one cross-attn representation.
3. **Head-size mismatch (3 orders of magnitude)** — 7-class softmax vs 4,702-class softmax. Shared backbone capacity is split across regimes that need different feature distributions.

### 2.3 F49 Layer 2 — the methodological bombshell

**Loss-side `task_weight=0` ablation is unsound under cross-attention MTL.** When `category_weight=0`, the cat encoder still receives gradient through `cross_ba`'s K/V projections — implicitly co-adapting as a reg-helper. The legacy "+14.2 pp transfer" claim measured cat_encoder co-adaptation, not transfer.

The clean architectural decomposition requires **encoder-frozen** isolation (set `category_encoder.requires_grad = False` AND `category_poi.requires_grad = False`). F49 implements this via a `--freeze-cat-stream` flag.

**This finding generalises** to MulT (Tsai et al. ACL 2019), InvPT (Ye et al. ECCV 2022), HMT-GRN (Lim et al. SIGIR 2022), and any cross-task interaction MTL with task-weight ablations. Paper-grade methodological contribution.

### 2.4 Cat-supervision transfer is null at our scale

F49 3-way decomposition (encoder-frozen λ=0 / loss-side λ=0 / Full MTL) gives transfer = `Full − loss-side` ≈ 0 at all 3 states (≤|0.75| pp). The conventional MTL framing — "joint training transfers signal between tasks" — is **empirically refuted** on this problem. The reg-side benefit at AL (+6.48 pp from architecture alone) is purely architectural; cat training adds ≈ 0.

### 2.5 Joint Δm Pareto-loses at FL (F50 T0)

Under the MTL-survey-standard Δm metric (Maninis CVPR 2019, Vandenhende TPAMI 2021):

| State | Δm primary (cat F1 + reg MRR) | n+/n- | p_greater | Verdict |
|:-:|:-:|:-:|:-:|:-:|
| AL | **+8.70%** | 5/0 | **0.0312** | MTL Pareto-wins (n=5 ceiling) |
| AZ | **+3.19%** | 5/0 | **0.0312** | MTL wins on MRR (marginal on top-K) |
| FL | **−1.63%** | 0/5 | 1.0 (p_two_sided=**0.0625**) | **MTL Pareto-loses (n=5 ceiling)** |

Cat-side advantage is uniform across states (Δ_cat F1 in [+0.7%, +7.0%] across 15 folds). Reg-side flip is monotone in cardinality. CH22 in `CLAIMS_AND_HYPOTHESES.md`.

### 2.6 Bonus: AZ MRR-Δm > top5-Δm asymmetry

At AZ, PRIMARY (MRR-based) Δm is significantly positive (+3.19%, p=0.0312); SECONDARY (top5-based) is null (−0.38%, p=0.500). MTL produces **better-ranked predictions** than STL even when raw top-K is similar. Mechanism distinction worth a paragraph in `paper/results.md`.

### 2.9 The 8.83 pp FL gap is **TEMPORAL**, not architectural (F50 T3 — 2026-04-29)

⭐ **Load-bearing finding** — supersedes the cat-encoder-absorption interpretation as the *proximate* mechanism for the FL reg gap (absorption still holds for cat F1 robustness, but is not the FL reg bottleneck).

**Observation:** STL reg-best epoch is at **{16, 17, 18, 20, 20}** (top10 = 82.44). Across every MTL configuration we have tested — H3-alt, T1.2 HSM, T1.3 FAMO, T1.4 Aligned-MTL, P1/P2/P3/P4, PLE-lite, Cross-Stitch (default + detach), cat_weight ∈ {0, 0.25, 0.50, 0.75}, reg_encoder_lr ∈ {3e-2, 1e-2}, reg_head_lr ∈ {3e-2, 1e-1} — the MTL **reg-best epoch is structurally pinned at ep 4-6** and **reg top10 = 73-75 pp**.

**Decisive test (D8):** with `category_weight = 0.0` (no cat loss whatsoever), MTL reg = 74.06 ± 0.71 with reg-best = {4, 5, 5, 5, 5}. **Cat dominance is REFUTED as the cause.**

**Confirmation (D6):** reg_head_lr=3e-2 fold 1 reaches reg-best = ep 0, top10 = 77.93 (close to STL ceiling). α growth IS mechanistically achievable in MTL — but the joint training pipeline destabilises it within a few batches.

**The 8.83 pp gap = the value of α growth that STL gets via 17 epochs of training but MTL is structurally prevented from reaching.**

**Likely proximate mechanisms (untested):**
- Constant scheduler vs OneCycleLR — H3-alt uses `--scheduler constant`; STL F37 used OneCycleLR (where peak-LR phase coincides with α-growth window in STL ep 15-20).
- Per-task-best epoch selection greedy bias — picks first local minimum at ep 5.
- Joint dataloader cycling artifacts at the 4.7K-region scale.

Full details: `research/F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md`.

---

### 2.8 Cross-attn shared backbone is "absorbed" by cat encoder at FL (F50 H1.5 P1 — 2026-04-29)

P1 (`--disable-cross-attn`) at FL 5f×50ep produces a model statistically indistinguishable from H3-alt:
- Per-fold paired Pearson r on **cat F1 = 0.985** (98.5% correlation, fold-init noise dominates)
- Per-fold paired Pearson r on **reg top10 = 0.336** (cross-attn changes reg outputs unsystematically — noise, not signal)
- Paired Wilcoxon two-sided cannot reject equality on either task (cat p=0.6250, reg p=0.8125)
- Mean Δreg = −0.21 ± 0.86 pp (within fold variance of CUDA H3-alt baseline 73.61 ± 0.83)

**The mechanism, from `diagnostics/fold1_diagnostics.csv` gradient traces:**

| epoch | g_cosine_shared | ‖g_reg‖_shared | ‖g_cat‖_shared | reg/cat |
|---:|---:|---:|---:|---:|
| 5 | +0.002 | 0.005 | 0.157 | **0.029** |
| 10 | −0.018 | 0.005 | 0.148 | **0.036** |
| 40 | −0.040 | 0.005 | 0.120 | **0.042** |
| 50 | +0.152 | 0.012 | 0.213 | 0.057 |

- Cat dominates shared-backbone gradient by **10–30×** most of training
- g_cosine ≈ 0 → cat/reg gradients on shared params are statistically independent (no task agreement to share)
- Reg gradient saturates at 0.005 by epoch 5 — reg head's α·log_T graph prior reaches its ceiling without help from shared backbone

**Connection to F49 λ=0 isolation:** F49 measured FL architectural Δ = **−16.16 pp** under `--freeze-cat-stream` (cat encoder frozen, no absorption). Live training's cat encoder absorbs the architectural cost by silently co-adapting as a reg-helper via cross_ba K/V (F49 Layer 2 leakage; verified by P2 detach-K/V collapsing reg-MRR σ from 8.52 → 1.09). **Net contribution to reg ≈ 0** in live training because cat-encoder compensation cancels the architectural cost.

**Implication:** the entire H1 + H1.5 negative-result pattern (FAMO/Aligned-MTL/HSM/P1/P2/P3 all FAIL +3 pp acceptance) is explained by a single mechanism — cat-encoder absorption masks any change to the shared backbone. Tier 2 (PLE/Cross-Stitch with task-specific isolation) is the only remaining test that can bypass the absorption channel.

Full analysis: `research/F50_T1_5_CROSSATTN_ABSORPTION.md`.

---

### 2.7 Hierarchical inductive bias on the reg head doesn't help (F50 T1.2 — full n=5)

T1.2 implemented an additive hierarchical bias (`parent_logit + child_logit + α·log_T`) on the reg head. STL HSM matched flat STL at FL (+0.21 pp p=0.0312) — architecture preserved at the head level. **MTL HSM at full n=5 (CUDA 2026-04-29):** reg top10_acc_indist = 70.60 ± 10.78 vs CUDA H3-alt 73.61 ± 0.83 → **Δ = −3.01 pp** (paired W+=10, p_greater=0.3125; even dropping the fold-2 collapse outlier the mean ≈ +1.66 pp, still below +3 pp acceptance). **Confirms n=3 directional refutation at paper grade.**

This **rules out an entire class of head-side fixes** — the FL architectural cost is structural to the cross-attention shared layer, not the head's softmax dimensionality. See `research/F50_T1_RESULTS_SYNTHESIS.md` §1-§2.

---

## 3 · Hypothesis space for "what fixes FL"

The remaining hypothesis space, after T1.2 + H1 closure (2026-04-29):

| # | Hypothesis | Test | Cost | Status |
|---|---|---|---|---|
| H1 | **Loss-side balancing** — newer gradient surgery (FAMO 2023, Aligned-MTL 2023) handles negative-transfer better than `static_weight(0.75)`. | T1.3 FAMO + T1.4 Aligned-MTL | ~19 min each on RTX 4090 | ✅ **CLOSED — FAIL** (2026-04-29). FAMO Δreg = +0.62 pp (W+=11, p=0.2188); Aligned-MTL Δreg = −0.11 pp (W+=4, p=0.8438). Neither reaches +3 pp acceptance or paired Wilcoxon significance. See `F50_T1_RESULTS_SYNTHESIS.md`. |
| **H1.5** | **Cross-attn mechanism probes** — direct ablations of the `MTLnetCrossAttn` mechanism that F49 Layer 2 attributed: P1 `--no-cross-attn` (bypass cross-attn blocks), P2 `--detach-crossattn-kv` (no cat↔reg gradient leakage), P3 `--freeze-cat-encoder-after-epoch N` (prevent continued co-adaptation), P4 `--separate-optimizers` (per-task AdamW, no shared group). | 4 × FL 5f×50ep | ~80 min train + ~6h dev total | 🔵 **IN FLIGHT** (Stage 1.5, 2026-04-29). Cheap minimal-edit probes BEFORE committing dev time to PLE / Cross-Stitch. |
| H2 | **Backbone-side decoupling** — explicit task-specific experts (PLE/CGC) prevent cat encoder from being conscripted as reg-helper. | T2.1 PLE (`mtlnet_ple`, `_components.py::PLELiteLayer`) — already implemented; per-head LR partition added 2026-04-29 | ~30 min train (4090) — DEV ALREADY DONE | 🔵 **READY TO RUN** (smoke ✓ on Georgia). Gated on H1.5 verdict for FL launch. **Caveat:** the codebase variant is a PLE-*lite* (per-task-input shared-experts adaptation), not canonical Tang RecSys 2020 — see §4.1 "Audit findings" below. |
| H3 | **Forced-vs-learned sharing** — Cross-Stitch / MTI-Net let the model learn how much to share per layer. | T2.2 Cross-Stitch | ~10-12h dev + ~30 min train (4090) | 🔵 **IN PROGRESS** (Stage 2 dev, parallel to H1.5 runs). |
| H4 | **Reg ceiling itself** — maybe `next_getnext_hard` isn't the right reg head and a stronger STL (e.g. ROTAN KDD 2024) shifts the ceiling. | T2.3 ROTAN STL | ~20h dev + ~5h train | DEFERRED — orthogonal to MTL question; out of current scope. |
| H5 | **Long-tail prototypes** — Bi-Level GSL-style cluster prototypes for the 4.7K-class softmax. | T3.1 | ~30h | last-resort |
| H6 | **Distillation** — STL teachers → MTL student. Captures STL ceiling but provides single-model deployment. | T3.2 | ~30h | last-resort |

**Note on H1.5 (new tier between H1 and H2):** F49 Layer 2 *attributed* the silent gradient flow through `cross_ba`'s K/V projections; nothing has been *done* about it under full-MTL conditions. The four probes are minimal-edit ablations that test the F49 mechanism directly. Each probe gives a falsifiable verdict in ~19 min train. If P1 (no-cross-attn) ≈ H3-alt at FL, the cross-attn shared layer is null/hurting and PLE is overkill. If P2 (detach-K/V) recovers FL Δm, the leakage is the FL flaw and we have a paper-headline minimal-edit fix. If P3 (cat-freeze post-warmup) recovers FL, "warm-then-freeze" is the recipe. If P4 differs from H3-alt's per-head LR, optimizer-level decoupling matters.

If H1+H1.5+H2+H3 all fail, the conclusion is: **"the FL architectural cost is robust to head + balancer + cross-attn-mechanism + backbone changes; cross-attention MTL is fundamentally cardinality-limited at this scale."** This is paper-grade ammunition for the scale-conditional CH21 framing.

---

## 3.1 · Audit findings on the existing MTL variants (2026-04-29)

The codebase already has `mtlnet_ple`, `mtlnet_cgc`, `mtlnet_mmoe`, `mtlnet_dselectk` registered (see `src/models/mtl/_components.py`). A critical review of each before staking F50 Tier-2 runs on them:

| Variant | Status vs published reference | Caveat for paper |
|---|---|---|
| **PLE-lite** (Tang RecSys 2020) | "Lite" — each shared expert is evaluated *per task input separately*. Shared experts have shared **parameters** but not shared **outputs**. | Defensible for our heterogeneous-input (cat=checkin emb, reg=region emb) setting, but diverges from canonical PLE. If PLE-lite recovers FL Δm, paper says "we adapt PLE for per-task-input MTL"; if not, we cannot conclude canonical PLE would also fail. |
| **CGC-lite** (Tang RecSys 2020) | Same per-task-input adaptation as PLE-lite (PLE = stack of CGC). | Same. |
| **MMoE-lite** (Ma KDD 2018) | Same per-task-input adaptation; all-shared experts; per-task gates. | Same. |
| **"DSelect-K"** (intended: Hazimeh NeurIPS 2021) | **Misnamed** — implementation is a dense convex combination via multi-softmax (`num_selectors` parallel softmaxes mixed by another softmax), NOT the Gumbel top-k sparse routing of the original. | **Do not claim "we tested DSelect-K".** Closer to multi-head MMoE with extra mixture step. Honestly disclosed in the docstring; needs paper-disclosure if reported. |

**F49 Layer 2 leakage check:** PLE/CGC/MMoE-style architectures do **NOT** exhibit the cross-attn K/V leakage F49 found. Per-batch gradient flow is task-specific (cat_gate sees cat_input only, etc.). Parameter-level coupling via `shared_experts` is the intended sharing mechanism, not a silent bug. **This is the exact reason PLE is the strongest H2 candidate** — the structural fix to F49's mechanism.

**Per-head LR support:** added to MTLnetPLE 2026-04-29 (`cat_specific_parameters` / `reg_specific_parameters`); CGC/MMoE/DSelectK do NOT have per-head LR partition yet (would error under `--cat-lr/--reg-lr/--shared-lr`; not on F50 critical path).

**Cross-Stitch (Misra CVPR 2016):** genuinely not implemented. New `mtlnet_crossstitch` to be added in F50 Stage 2.

**`MTLnetPLE.next_forward` shape consistency:** verified harmless 2026-04-29. The 2D `dummy_cat = zeros(B, D)` vs 3D `enc_next = [B, T, D]` is silently broadcasted; cat-side output is computed-then-discarded; only the next-side output (correctly shaped) is returned.

---

## 4 · External literature mapped to our flaws

| Paper | Year | Flaw it targets | Worth trying here? |
|---|:-:|---|---|
| **FAMO** (Liu et al., NeurIPS 2023) | 2023 | Negative-transfer / conflicting gradients via O(1)-cost adaptive task weighting | **T1.3 — speculative.** Reported on dense-vision (NYUv2/CelebA/QM9), not long-tail multi-class. |
| **Aligned-MTL** (Senushkin et al., CVPR 2023) | 2023 | Direction-side conflicting-gradient via condition-number minimization | **T1.4 — speculative.** Same domain caveat. |
| **PLE / CGC** (Tang et al., RecSys 2020 best paper) | 2020 | Forced-sharing inducing negative transfer; uses task-specific + shared experts with progressive routing | **T2.1.** Industrial-scale validated. Codebase has `mtlnet_dselectk` (related family). |
| **Cross-Stitch** (Misra et al., CVPR 2016) | 2016 | Forced sharing — learns per-layer share/task-specific weights | **T2.2.** Pure form of "learned share". |
| **MTI-Net** (Vandenhende et al., ECCV 2020) | 2020 | Task-affinity-aware sharing per layer | possible T2 alternative |
| **AdaShare / RotoGrad** | 2020/2022 | Auto-share decisions / dynamic gradient homogenisation | possible alternatives |
| **ForkMerge** (Jiang et al., NeurIPS 2023) | 2023 | Explicit negative-transfer via periodic branch forking | possible Tier 2 |
| **DST (Dropped Scheduled Task)** (OpenReview 2024) | 2024 | Task-dropping during joint optimization | possible Tier 2 |
| **CoBa** (EMNLP 2024) | 2024 | Convergence balancer for MTL fine-tuning | LLM-leaning |
| **MulT** (Tsai et al., ACL 2019) | 2019 | Cross-modal cross-attention — affected by F49 Layer 2 critique | reference |
| **InvPT** (Ye et al., ECCV 2022) | 2022 | Inverted pyramid cross-task — affected by F49 Layer 2 | reference |
| **HMT-GRN** (Lim et al., SIGIR 2022) | 2022 | Hierarchical region MTL — closest POI competitor | benchmark anchor |
| **MGCL** (Frontiers 2024) | 2024 | Multi-granularity contrastive (location + region + category) | possible substrate-side fix |
| **Bi-Level GSL** (arXiv 2024) | 2024 | POI + prototype-level graph for long-tail mitigation | T3.1 |
| **ROTAN** (KDD 2024) | 2024 | Rotation-based temporal attention next-POI head | T2.3 reg ceiling test |
| **Diff-POI** | 2023 | Diffusion-based POI generation | post-paper |
| **Hierarchical softmax** (Mikolov 2013, Mnih NIPS 2008) | classic | Large-vocab classification efficiency | T1.2 used additive variant; not yet true HSM (decomposed loss) |
| **Balanced Meta-Softmax** (Ren et al., NeurIPS 2020) | 2020 | Long-tail softmax with frequency-aware correction | T3 candidate |

---

## 5 · Operational rules learned the hard way

### 5.1 Always smoke-test before a long FL run

Before launching any new head/loss config at FL 5f×50ep (~3.5h MPS), run a 1f×2ep AL MTL smoke (~30s):

```bash
PYTHONPATH=src DATA_ROOT=... OUTPUT_DIR=... \
  python scripts/train.py \
    --task mtl --task-set check2hgi_next_region \
    --state alabama --engine check2hgi \
    --model mtlnet_crossattn --mtl-loss <loss> \
    --cat-head next_gru --reg-head <new_head> \
    --reg-head-param ... \
    --task-a-input-type checkin --task-b-input-type region \
    --folds 1 --epochs 2 --seed 42 --batch-size 1024 \
    --scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --no-checkpoints --no-folds-cache
```

Expect: reg F1 ≥ 0.5%, reg accuracy ≥ 5% by epoch 2. If both stay 0, suspect:
1. Aux side-channel gate (`src/data/folds.py` — `_HEADS_REQUIRING_AUX_MTL` set membership)
2. Param partition (`MTLnetCrossAttn.cat_specific_parameters / reg_specific_parameters`)
3. Head output shape mismatch

### 5.2 Use FL `--batch-size 1024` (2048 silently OOM-kills)

Per AGENT_CONTEXT and observed in the H3-alt champion run. Above that, MPS sometimes silently crashes during fold 2 epoch 23.

### 5.3 Don't trust MTL "transfer" claims based on loss-side λ=0 alone

F49 Layer 2: under cross-attention, the silenced encoder co-adapts via attention K/V. Use encoder-frozen isolation (`requires_grad=False` on `category_encoder` and `category_poi`) for the architectural decomposition.

### 5.4 Per-task-best vs joint-best metrics — pick consistently

`diagnostic_best_epochs` = per-task-best (each task at its own best epoch). `primary_checkpoint` = joint-best. Δm and paired Wilcoxon should pair on **same selection rule on both sides** (STL is per-task-best by definition). Tier 0 used per-task-best — see `F50_DELTA_M_FINDINGS.md §3.4`.

### 5.5 Tier 1 alternatives must run BEFORE P3 (CA+TX, ~37h Colab)

If a Tier 1 alternative changes the FL champion, P3 must re-run under the new config. Currently P3 is gated on Tier 1 reconvene per `PHASE2_TRACKER.md`.

---

## 6 · Open questions (the F50 search frontier)

These are the genuinely unanswered questions the F50 audit surfaced:

1. ~~**Does FAMO recover FL?**~~ ✅ **NO** — T1.3 closed 2026-04-29 (Δreg = +0.62 pp, fails +3 pp acceptance).
2. ~~**Does Aligned-MTL recover FL?**~~ ✅ **NO** — T1.4 closed 2026-04-29 (Δreg = −0.11 pp).
3. **If gradient-balancing fails, does PLE recover FL?** Backbone-side test — Stage 2 dev in flight.
4. **Is the architectural cost monotonicity 4-state or just 3-state?** CA+TX P3 extends to 5 points (handled on Colab; out of current pod scope).
5. **What's the joint-deployment "deployment Δm"** (joint-best epoch for MTL) vs the per-task-best "potential Δm" reported in T0? Could differ by ~0.3-0.5 pp on means. Note: joint-best on CUDA shows σ ≈ 12 pp (substrate-fragile under fp16-autocast — `RUNPOD_GUIDE` §9).
6. **Does ROTAN beat GETNext-hard at single-task?** Tests whether reg ceiling itself is higher than 82.44 at FL. **DEFERRED** — orthogonal to MTL question.
7. ~~**What is the cross-attention K/V mechanism doing differently at FL vs AL?**~~ 🔵 **IN FLIGHT** — H1.5 probes (P1-P4) directly test this 2026-04-29.
8. **Does the AZ MRR-vs-top-K asymmetry replicate at AL?** Bonus mechanism finding worth investigating.
9. **(NEW) Does removing F49 Layer 2 leakage recover FL Δm?** P2 `--detach-crossattn-kv` test — see H1.5.
10. **(NEW) Is the cross-attn shared layer load-bearing at FL or null?** P1 `--no-cross-attn` test — see H1.5.

---

## 7 · How this doc should evolve

Update each section as new evidence lands:

- §2 (Empirical flaws) — append new F-experiment findings as one paragraph + table row
- §3 (Hypothesis space) — flip `Status` cell when an H gets tested
- §4 (Literature) — add new papers as discovered
- §5 (Operational rules) — append new lessons as found
- §6 (Open questions) — close as evidence lands; add new questions

Treat this as a **catalog**, not a narrative. Narrative lives in `MTL_ARCHITECTURE_JOURNEY.md`.

---

## 8 · Cross-references

- `MTL_ARCHITECTURE_JOURNEY.md` — narrative history (B-M → F48-H3-alt)
- `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` — F49 attribution + Layer 2
- `research/F37_FL_RESULTS.md` — F37 closing; FL STL ceiling = 82.44 ± 0.38
- `research/F50_PROPOSAL_AUDIT_AND_TIERED_PLAN.md` — F50 tiered plan
- `research/F50_DELTA_M_FINDINGS.md` — F50 T0 result (CH22)
- `research/F50_T1_1_CAT_HEAD_PATH_DECISION.md` — F50 T1.1 (F33 PASS)
- `research/F50_HANDOFF_2026-04-28.md` — F50 pickup state
- `review/2026-04-28_critical_analysis.md` — opinion piece + framing recommendation
- `CLAIMS_AND_HYPOTHESES.md §CH18, §CH20, §CH21, §CH22` — all MTL-flaw claims
- `CONCERNS.md §C12, §C15` — open MTL-related concerns
- `NORTH_STAR.md` — current champion (F48-H3-alt, unchanged by F50 so far)
