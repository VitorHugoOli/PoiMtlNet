# F49 — λ=0.0 Isolation Decomposition Gap (planning note)

**Date:** 2026-04-27. **Tracker:** `FOLLOWUPS_TRACKER.md §F49`. **Status:** planning — no compute landed yet. **Cost (planned):** AL+AZ both variants ~3 h MPS; FL both variants ~9 h MPS (batch=1024).

**Review pass (2026-04-27):** independent technical review verdict YELLOW — proceed with named modifications. Five issues found and incorporated:
- B1: in-block `ffn_a / ln_a1 / ln_a2` continue to train under L_reg in both variants → §"B-side processing inside the shared block" added; "frozen-cat" renamed to "encoder-frozen" throughout to reflect the actual scope.
- B2: outcome interpretation pre-committed the sign of the co-adaptation term → §"Outcome interpretation" rewritten sign-agnostic; new H1b acknowledges `loss_λ0 < frozen_λ0` as a publishable possibility.
- B3: acceptance criterion #3 incorrectly equated `static_weight(0)` with `PCGrad(0)` for the original-protocol reproduction → criterion now specifies PCGrad+OneCycleLR for the reproduction check, with a note on why `static_weight` is fine for the H3-alt-regime measurements but not for matching the original 52.27 number.
- C1: `setup_per_head_optimizer` does not currently filter `requires_grad=False`, so AdamW's weight_decay would silently decay the frozen cat encoder → implementation sketch §2 promoted from "single-line guard" to required edit; regression test (4b) added with norm-snapshot assertion to catch the bug.
- C4: no regression test covered the loss-side gradient-flow claim itself → test (4a) added, asserting `category_encoder.weight.grad.abs().sum() > 0` under loss-side λ=0.

## Question

The pre-B3 ablation series (`archive/research_pre_b5/CHAIN_FINDINGS_2026-04-20.md`, `HYBRID_DECISION_2026-04-20.md`) used **loss-side λ=0** (`static_weight(category_weight=0.0)`) to decompose the MTL-vs-STL gap on `next_region` into "architectural overhead" plus "cat→reg transfer":

| State | STL reg | λ=0 reg | Full MTL reg | Overhead | Transfer |
|---|---:|---:|---:|---:|---:|
| AL | 56.94 | 52.27 | 52.41 | **−4.67 pp** | +0.14 |
| FL | 68.33 | 43.40 | 57.60 | **−24.93 pp** | +14.20 |

This decomposition is referenced in `research/POSITIONING_VS_HMT_GRN.md §72-77, §86` as a paper-distinctive contribution: "we are first to decompose architectural overhead vs transfer in semantic-auxiliary MTL on check-in-level embeddings."

`CONCERNS.md §C12` (2026-04-18, status `under investigation`) already flagged that the overhead numbers were measured at MTL's old default `max_lr=0.001` while STL was measured at `max_lr=0.003` — an HP confound.

**F49 raises a second, more fundamental confound: gradient flow through cross-attention makes loss-side λ=0 not a clean architectural isolation.**

## Gradient-flow analysis

Under `category_weight=0.0`, the total loss is `L = 1·L_reg + 0·L_cat = L_reg`. Tracing the cross-attention forward pass in `src/models/mtl/mtlnet_crossattn/model.py::_CrossAttnBlock.forward`:

```python
a_upd = cross_ab(Q=a, K=b, V=b)        # a queries b
a = ln_a1(a + a_upd); a = ln_a2(a + ffn_a(a))
b_upd = cross_ba(Q=b, K=a, V=a)        # b queries (updated) a   ← load-bearing
b = ln_b1(b + b_upd); b = ln_b2(b + ffn_b(b))
```

The reg stream's `b` goes through `cross_ba(Q=b, K=a, V=a)` where `a` is derived from `category_encoder(cat_input)`. So `∂L_reg/∂a ≠ 0` (linear via `cross_ba`'s K/V projections), and that gradient propagates back to `category_encoder` through the `cross_ab` block.

**Who actually receives gradient under loss-side λ=0:**

| Component | Gradient under λ=0? | Notes |
|---|:-:|---|
| `category_encoder` | ✅ **yes** | Through `cross_ba` K/V path (load-bearing) |
| `next_encoder` | ✅ yes | Direct reg path |
| All `_CrossAttnBlock` weights (cross_ab, cross_ba, ffn_a, ffn_b, ln_*) | ✅ yes | All in reg's gradient path |
| `next_final_ln`, `next_poi` (reg head) | ✅ yes | Direct |
| `cat_final_ln` | ❌ no | Only consumer is cat head (zero-weighted in loss) |
| `category_poi` (cat head) | ❌ no | Same reason |

**Implication:** under loss-side λ=0, the cat **encoder** is **not idle**. It is supervised — by `L_reg` flowing back through `cross_ba`'s K/V projections — to produce features that are useful for the reg stream's queries. The cat encoder ends up trained as a **reg-helper**, not as a cat predictor. Only the cat head + `cat_final_ln` are effectively frozen at init.

This means `(loss-side λ=0 reg − STL reg)` is **not** a clean measure of "architectural overhead." It is a measure of "architecture **plus** reg-helper co-adaptation of the cat encoder, jointly." The two effects can have opposite signs and cannot be disentangled from a single λ=0 measurement.

## Why this matters now

H3-alt (2026-04-26) gives MTL reg = +6.25 pp over STL on AL. The original decomposition would imply:

```
overhead + transfer = +6.25 pp on AL
```

But "overhead" here mixes pure architectural cost with the (sign-agnostic) cross-attn co-adaptation contribution. Under H3-alt's `reg_lr=3e-3 constant`, cross-attn params train far more aggressively than they did in the original measurement (where shared LR was 1e-3 with OneCycleLR annealing). The magnitude *and sign* of the cat-encoder-as-reg-helper effect under H3-alt is unknown — we have no direct measurement of either component.

A reviewer reading both `POSITIONING_VS_HMT_GRN.md` (claiming a 25 pp architectural overhead) and `MTL_ARCHITECTURE_JOURNEY.md` (claiming H3-alt exceeds STL by +6.25 pp) will see the inconsistency and pull on the thread. The decomposition needs to be either (a) re-measured cleanly under H3-alt, (b) redefined in a way that is methodologically sound under cross-attention, or (c) dropped.

## Proposed 3-way decomposition

A clean decomposition requires **two** isolation runs per state, not one:

| Variant | Implementation | What it isolates |
|---|---|---|
| **STL_reg** | `embedding → next_getnext_hard → softmax`, no MTL pipeline. | Reference. Already have F21c numbers (AL 68.37, AZ 66.74; FL pending F37). |
| **frozen-cat λ=0** *(new)* | `category_weight=0.0` AND `category_encoder.requires_grad_(False)` AND `category_poi.requires_grad_(False)` AND `category_encoder.eval()`. Cat stream contributes only fixed random-init features to cross_ba's K/V; cross-attn + reg path train normally. | "Pure architectural cost (encoder-frozen)": what does the architecture cost reg when the cat **encoder** is fixed random-init and never adapts? |
| **loss-side λ=0** *(new)* | `category_weight=0.0` only. Cat encoder receives gradient through `cross_ba` K/V; cat head + `cat_final_ln` get zero gradient. | "Architecture + cat-encoder-as-reg-helper": what loss-side ablation actually measures, with the role of cross-attn-mediated cat-encoder co-adaptation made explicit. |
| **Full MTL_reg** | H3-alt champion (already have AL 74.62, AZ 63.45, FL 71.96). | Reference for the upper end of the decomposition. |

The 3-way decomposition becomes:

```
full MTL − STL  =  (frozen_λ0 − STL)        ← architectural overhead/benefit (encoder-frozen)
                + (loss_λ0 − frozen_λ0)     ← cat-encoder co-adaptation (sign-agnostic; see Outcome interpretation)
                + (full − loss_λ0)          ← cat-supervision transfer (the contribution of L_cat itself)
```

Each term has the **predominant** mechanism interpretation listed; the decomposition is a telescoping identity in *measurement* terms (trivially true), but the *mechanism* labels rest on the assumption that each variant changes one thing. The next subsection notes one in-block contamination that softens "pure architectural" to "predominantly architectural."

### B-side processing inside the shared block (acknowledged contamination)

Each `_CrossAttnBlock` contains per-stream `ffn_a`, `ln_a1`, `ln_a2` — "cat-side processing inside the shared block." These live in `shared_parameters()` (yielded by `for block in crossattn_blocks: yield from block.parameters()`) and **continue to train under L_reg in both the loss-side AND the frozen-cat variants**, because the reg stream consumes their outputs as K/V via the in-block residual chain (each block's `cross_ba` reads `a` after `ffn_a` has already run in earlier blocks). Freezing them would corrupt the reg pipeline rather than isolate architecture.

The frozen-cat variant therefore measures: **pure architectural overhead + in-block cat-side FFN co-adaptation**, and the `(loss_λ0 − frozen_λ0)` term measures co-adaptation of the cat **encoder** specifically (the 64→256 pre-encoder MLP). The cleaner phrasing is "encoder-frozen" rather than "cat-stream-frozen," and the doc adopts this terminology throughout.

The reg pre-encoder (`next_encoder`, also a 64→256 MLP) is **not** frozen in either variant. Its supervision is L_reg directly, which is proper STL-like training; F41 already established that this MTL-style pre-encoder, when bolted onto STL, does not move reg Acc@10 outside σ on AL or AZ (`research/F41_PREENCODER_FINDINGS.md`). The asymmetry between cat (frozen in the encoder-frozen variant) and reg (always trainable) is therefore principled, not arbitrary.

## Acceptance criteria for the experiment

The experiment runs cleanly if and only if:

1. **`frozen-cat λ=0` does not crash.** The frozen cat encoder produces well-defined K/V into `cross_ba` (no NaN, no shape mismatch). A regression test (`tests/test_models/test_mtlnet_crossattn_frozen_cat.py`) asserts `category_encoder.weight.grad is None` after a backward pass and that `next_logits` are finite over a smoke batch.
2. **AdamW weight-decay handling for frozen params.** AdamW applies `wd · θ` even when gradient is zero. To prevent silent decay of the frozen cat encoder weights, the optimizer must either (a) exclude frozen params from any param group, or (b) set `weight_decay=0` for the cat-specific group when the freeze flag is set. Default to (a): `setup_per_head_optimizer` filters `requires_grad=False` params out of every group.
3. **`loss-side λ=0` reproduces, within σ, the architectural-pipeline number** measured before C12 — that is, when re-run with the *original* training regime, AL should land near 52.27 ± 5.03 and FL near 43.40 (1f). This validates the new infra rather than the new claim. Original protocol (per `archive/research_pre_b5/EXECUTION_PLAN_2026-04-18.md` line 30): `--mtl-loss static_weight --category-weight 0.0 --max-lr 1e-3` with the default OneCycleLR scheduler, `mtlnet_crossattn` model, `next_gru` reg head (the F-series legacy reg head — F21c's `next_getnext_hard` came later). The H3-alt-regime measurement is the same loss but per-head LR + constant scheduler + `next_getnext_hard`, so the reproduction is at a different point in the design space and can only validate that the F49 infra emits sensible numbers under the *legacy* recipe.

Outcome interpretation (sign-agnostic — pre-committing the sign would be the same kind of mistake C12 was):

- **H1a:** Under H3-alt, `frozen_λ0 ≈ STL` (within σ). "Architectural overhead" is ~0; the apparent cost in the pre-B5 measurement was the joint LR+co-adaptation confound. `loss_λ0 > STL` further shows the cat encoder serving as reg-helper is itself a positive contribution. Strengthens H3-alt: full MTL reg lift = small architectural benefit + positive co-adaptation + cat transfer, all positive.
- **H1b:** `loss_λ0 < frozen_λ0`. The cat-encoder-as-reg-helper specializes K/V projections in a way that overfits or reduces attention entropy, while randomized K/V in the frozen-cat variant acts as an implicit regularizer. This is a publishable finding (cross-attn co-adaptation can hurt), would re-frame H3-alt's lift as "pure architectural + transfer, despite negative co-adaptation," and is consistent with noise-robustness findings in some MulT/InvPT settings. Magnitude and sign of `(loss_λ0 − frozen_λ0)` is an empirical question we are not allowed to pre-commit.
- **H2:** `frozen_λ0 < STL` (architectural overhead survives even with the LR fix). H3-alt's lift then = transfer overcomes overhead. Different paper narrative; still publishable, requires updating `MTL_ARCHITECTURE_JOURNEY.md`.
- **H3:** `frozen_λ0 ≈ loss_λ0 ≈ full MTL` (each addition contributes ~0). Would falsify the decomposition story and demand we drop or replace it. Unlikely given F45/H3-alt's α-growth mechanism, but a possible outcome.
- **H4 (worst case):** `frozen_λ0` is unstable / NaN-prone because `cross_ba` does not expect frozen-random K/V. Unlikely (cross-attention works on arbitrary inputs by design) but worth catching with the smoke test.

## Implementation sketch

**Code changes (~60 LOC total + 2 regression tests):**

1. `scripts/train.py` — new flag `--freeze-cat-stream` (default False). When True, after model construction:
   ```python
   for p in model.category_encoder.parameters(): p.requires_grad_(False)
   for p in model.category_poi.parameters():     p.requires_grad_(False)
   model.category_encoder.eval()  # disables dropout in the cat encoder
   ```
   Validation: requires `--mtl-loss static_weight --category-weight 0.0` (frozen cat without zero loss is incoherent — would silently drop the cat-loss gradient).
2. `src/training/helpers.py::setup_per_head_optimizer` — **must filter `requires_grad=False` from every param group before constructing AdamW**. The current implementation passes the param iterators verbatim; AdamW's `weight_decay=0.05` (project default) applies `wd · θ` to params with `grad=None`, **silently shrinking the frozen cat encoder weights toward zero across 50 epochs**, which would invalidate the entire frozen-cat measurement. Required edit:
   ```python
   cat_params    = [p for p in model.cat_specific_parameters()    if p.requires_grad]
   reg_params    = [p for p in model.reg_specific_parameters()    if p.requires_grad]
   shared_params = [p for p in model.shared_parameters()           if p.requires_grad]
   ```
   This is non-optional. The regression test in (4b) below catches the silent-decay bug if anyone refactors this away.
3. `src/training/runners/mtl_cv.py` — emit a smoke print on fold 0 listing trainable-param counts per group; assert frozen-cat group reports 0 trainable params when `--freeze-cat-stream` is set.
4. **Two regression tests** (`tests/test_models/test_mtlnet_crossattn_lambda0_gradflow.py`):

   **(4a) Loss-side λ=0 — confirms the load-bearing claim of this whole project.** Build the model with `static_weight(category_weight=0.0)`, do NOT freeze, run forward on a batch, run backward on the joint loss. Assert:
   - `category_encoder.weight.grad is not None`
   - `category_encoder.weight.grad.abs().sum() > 0`
   - `category_poi.weight.grad is None` (or zero)
   - `cat_final_ln.weight.grad is None` (or zero)

   This is the smoke alarm for any future refactor that breaks the K/V gradient path.

   **(4b) Frozen-cat λ=0 — confirms freeze + optimizer-filter actually freezes weights.** Build the model with `--freeze-cat-stream` semantics applied, build optimizer via `setup_per_head_optimizer`, snapshot `category_encoder.weight.norm()`, run forward + backward + `optimizer.step()`, assert the norm is unchanged to within fp32 epsilon. This is the test that catches the AdamW-weight-decay-on-frozen-params bug (the silent decay would shift the norm even with grad=None).

**Launcher:** `scripts/run_f49_lambda0_decomposition.sh`. Two functions: `run_loss_side` (just adds `--category-weight 0.0`) and `run_frozen_cat` (adds `--category-weight 0.0 --freeze-cat-stream`). Loops over states ∈ {al, az, fl} with H3-alt LR per-head + constant scheduler.

## Methodological angle (paper-relevant)

This is more than a fix to our internal decomposition. The MTL literature on loss-side `task_weight=0` ablations (`static_weight`, NashMTL surveys, MTL-loss-balancing papers) implicitly assumes shared-backbone architectures where setting a task's loss to zero genuinely silences that task's parameters. Cross-task interaction architectures (cross-attention MTL: MulT, InvPT, our `mtlnet_crossattn`) violate that assumption: the silenced task's encoder still co-adapts via attention K/V to serve the active task.

Worth a paragraph in the paper's methods or limitations: **"loss-side ablation underestimates architectural overhead in cross-attention MTL because the silenced task's encoder is implicitly supervised by the active task's gradient through attention K/V projections. We use frozen-stream isolation to measure pure architectural overhead, distinct from the cross-attention co-adaptation effect."**

## Cross-references

- `archive/research_pre_b5/CHAIN_FINDINGS_2026-04-20.md` — original 2-way decomposition numbers.
- `archive/research_pre_b5/HYBRID_DECISION_2026-04-20.md` — original AL cross-attn λ=0 measurement (52.27 ± 5.03) under PCGrad+OneCycleLR.
- `CONCERNS.md §C12` — pre-existing LR confound flag (still open). F49 supersedes / extends C12.
- `research/POSITIONING_VS_HMT_GRN.md §72-77, §86` — paper-relevant decomposition framing that this measurement supports or refutes.
- `research/F48_H3_PER_HEAD_LR_FINDINGS.md` — H3-alt champion; F49 is the natural follow-up.
- `MTL_ARCHITECTURE_JOURNEY.md §9` — open directions; this note formalises the "α instrumentation / decomposition" item.
- `research/F41_PREENCODER_FINDINGS.md` — establishes that the MTL-style 64→256 pre-encoder, when added to STL, does not move reg Acc@10 outside σ on AL/AZ. Justifies leaving `next_encoder` (reg pre-encoder) trainable in both F49 variants while freezing only `category_encoder` in the encoder-frozen variant.
- `src/models/mtl/mtlnet_crossattn/model.py:44-106` — `_CrossAttnBlock.forward` (gradient-flow source). Specifically: `cross_ba(Q=b, K=a, V=a)` at line 99-100 is the load-bearing gradient path that makes loss-side λ=0 not a clean isolation.
- `src/models/mtl/mtlnet_crossattn/model.py:323-352` — `shared_parameters` / `cat_specific_parameters` / `reg_specific_parameters` (param-group definitions). Note: `_CrossAttnBlock`'s per-stream `ffn_a / ln_a1 / ln_a2` are yielded by `shared_parameters` (line 327-328 `for block in crossattn_blocks: yield from block.parameters()`) and continue to train under L_reg in both F49 variants.
- `src/training/helpers.py::setup_per_head_optimizer` — required edit site for the `requires_grad` filter (must land before any frozen-cat compute).

## Status

`planning — 2026-04-27`. No compute landed. Not yet referenced in `NORTH_STAR.md` or `OBJECTIVES_STATUS_TABLE.md` until results land. Promote to active when AL+AZ both variants run and the regression test lands.
