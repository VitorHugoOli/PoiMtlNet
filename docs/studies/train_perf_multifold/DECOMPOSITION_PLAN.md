# Decomposition plan (workflow wf_266c2a32-457)

## Modern-torch digest

# Modern torch (2.4–2.11) best-practices digest — byte-identical axis flagged for each

The load-bearing distinction: **API renames that delegate to the same kernels are byte-identical (SAFE for a frozen repo); anything that reorders FP ops or changes which params get stepped is NUMERICS-CHANGING (must be excluded).** Verdicts below are grounded in source I actually fetched.

---

## 1. AMP API — `torch.cuda.amp` → `torch.amp` / `torch.autocast`

**Modern recommendation (official):** Replace `torch.cuda.amp.autocast(...)` with `torch.amp.autocast("cuda", ...)` and `torch.cuda.amp.GradScaler(...)` with `torch.amp.GradScaler("cuda", ...)`. The 2.12 `amp.html` states verbatim that the old forms are *"deprecated"* and to *"Please use `torch.amp.autocast("cuda", args...)`"* / *"`torch.amp.GradScaler("cuda", args...)` instead."* `torch.autocast(device_type, ...)` is just the alias of `torch.amp.autocast` (same callable), so both `torch.amp.autocast("cuda", ...)` and `torch.autocast("cuda", ...)` are equivalent and recommended.

**Byte-identical?** **YES — pure rename, SAFE.** I read the v2.11.0 source. `torch.cuda.amp.autocast` is `@deprecated` (emits `FutureWarning`: *"`torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead."*) and its `__init__` does nothing but:
```python
super().__init__("cuda", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)
```
i.e. it **already is** `torch.amp.autocast` with `"cuda"` hardcoded. `GradScaler` is identical — `@deprecated`, then `super().__init__("cuda", init_scale=..., growth_factor=..., backoff_factor=..., growth_interval=..., enabled=...)` with all args passed unchanged. Same class, same kernels, same defaults. The swap only removes the warning; it cannot change any scored number.
- Sources: [v2.11.0 cuda/amp/autocast_mode.py](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/cuda/amp/autocast_mode.py), [v2.11.0 cuda/amp/grad_scaler.py](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/cuda/amp/grad_scaler.py), [amp.html (2.12)](https://docs.pytorch.org/docs/2.12/amp.html)

---

## 2. `torch.is_autocast_enabled(device_type)` vs no-arg

**Modern recommendation:** Pass the device: `torch.is_autocast_enabled("cuda")`. The `device_type: str` parameter was **added in 2.4.0**; before that the function took no args and was implicitly CUDA. On 2.11 the no-arg form still works (defaults to CUDA-equivalent behavior) but the explicit form is the device-agnostic, forward-compatible one. (Note a separate function, `torch.amp.is_autocast_available(device_type)`, answers a *different* question — availability, not enabled-state.)

**Byte-identical?** **YES — it is a read-only query, SAFE.** Adding/removing the `"cuda"` arg returns the same boolean and does not touch any tensor math. Caveat is purely portability: code calling `is_autocast_enabled("cuda")` is incompatible with torch < 2.4 — irrelevant here (repo is pinned 2.11.0).
- Sources: [huggingface/transformers#43508](https://github.com/huggingface/transformers/issues/43508) (documents the 2.4.0 signature addition), [amp/autocast_mode.py @ main](https://github.com/pytorch/pytorch/blob/main/torch/amp/autocast_mode.py)

---

## 3. `optimizer.zero_grad(set_to_none=True)`

**Modern recommendation:** It is **already the default** in modern torch. The 2.12 doc signature is `Optimizer.zero_grad(set_to_none=True)`, described as *"Instead of setting to zero, set the grads to None. Default: True"* — *"lower memory footprint, and can modestly improve performance."* So you do not need to pass it; the recommended path is to rely on the default (or pass `True` explicitly for clarity).

**Byte-identical?** **CONDITIONAL — this is the one rename-looking change that can move numbers.** The doc explicitly warns: *"torch.optim optimizers have a different behavior if the gradient is 0 or None (in one case it does the step with a gradient of 0 and in the other it skips the step altogether)."* Concretely, with `None` grads the optimizer **skips** the update for that param, whereas with zeros it performs a step that — under `weight_decay` (AdamW uses `weight_decay=0.05` here) and momentum — still moves the weight. Therefore:
- If the current code **already uses the default / `True`**, or every param receives a gradient every step, switching is a **no-op → byte-identical, SAFE.**
- If the current code **explicitly passes `set_to_none=False`** (or relied on an old version whose default was `False`), flipping it is **NUMERICS-CHANGING → must be EXCLUDED.** Verify what the repo currently does before touching this.
- Source: [Optimizer.zero_grad (2.12)](https://docs.pytorch.org/docs/2.12/generated/torch.optim.Optimizer.zero_grad.html)

---

## 4. `torch.compile` hygiene

**Modern usage:** (a) Compile **once** and reuse — the compiled callable holds a guarded cache keyed on shapes/dtypes; calling it repeatedly with matching guards reuses kernels and does **not** recompile. (b) For varying input shapes, use `dynamic=True` or `torch._dynamo.mark_dynamic(tensor, dim)` so a single dynamic-shape graph is compiled instead of one-per-shape, avoiding `recompile_limit`/`cache_size_limit` (`torch._dynamo.config.recompile_limit`, default 8; beyond it the fn silently falls back to eager). (c) Diagnose with `TORCH_LOGS="recompiles,graph_breaks"` and validate with `fullgraph=True`. (d) `torch._dynamo.reset()` (a.k.a. `torch.compiler.reset()`) wipes the **Dynamo** cache to a clean state — use it in benchmarks/between unrelated compiles — but note it does **not** clear the on-disk Inductor cache.

**On "compile once + `load_state_dict` across folds":** This is **sound** — `load_state_dict` mutates parameter *values* in place; it does not change tensor shapes/dtypes/graph structure, so guards still pass and **no recompile is triggered**. You can build one compiled module and reload weights per fold. (If you instead construct a *fresh* `nn.Module` per fold and re-`compile`, call `torch._dynamo.reset()` between them to avoid stale guards / cache-limit collisions; the newer `isolate_recompiles=True` option also helps.)

**Byte-identical?** **NO — `torch.compile` is inherently NUMERICS-CHANGING; treat the compiled path as frozen, do not "tidy" it.** PyTorch states plainly: *"Applying torch.compile … will cause numerical changes because of (1) reordering of floating-point ops during … fusion and (2) use of lower precision data types … 100% bitwise compatibility … is not expected."* Implications for this repo:
- `--compile`/`--tf32` are board-recipe values (per CLAUDE.md), so the **frozen numbers were produced *with* a specific compile + Inductor autotune config**. Changing compile hygiene (different `mode`, toggling `dynamic`, different autotune, reset timing) can change **kernel selection → different FP results.** Exclude from byte-identical edits.
- **Safe sub-changes:** the API-rename class (modernizing the *surrounding* AMP/zero_grad calls) and pure structural refactors are byte-identical as long as the compiled region's inputs, dtype, autocast scope, and compile arguments are reproduced exactly. Verify peak/kernel-determining flags (`tf32`, autotune) are unchanged.
- Sources: [PT2 production-models blog (bitwise quote)](https://pytorch.org/blog/training-production-ai-models/), [compiler FAQ](https://docs.pytorch.org/docs/stable/torch.compiler_faq.html), [Dealing with Recompilations](https://docs.pytorch.org/docs/stable/compile/programming_model.recompilation.html), [torch.compiler.reset](https://docs.pytorch.org/docs/stable/generated/torch.compiler.reset.html), [troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html)

---

## 5. `@torch.inference_mode()` vs `@torch.no_grad()` for eval

**Modern recommendation:** `inference_mode` is the faster default for pure inference/eval — *"analogous to no_grad … better performance by disabling view tracking and version counter bumps."* Prefer `no_grad` when eval outputs (or tensors derived from them) **must re-enter an autograd-tracked computation**, because *"tensors created in this mode cannot be used in computations recorded by autograd"* and will raise at the boundary. Neither sets eval mode — you still need `model.eval()` for dropout/BN.

**Byte-identical?** **YES for the scored numbers — SAFE, with a functional (not numeric) caveat.** `inference_mode` runs the **same forward kernels** and produces **identical values** to `no_grad`; the differences (no view tracking, no version-counter bumps, frozen `requires_grad`) are bookkeeping, not math. So metrics computed under either are the same. The only risk is **runtime errors** (not number changes) if an inference-mode tensor is later fed into a grad-tracked graph — a guardrail, not a silent corruption. For an eval/metric-only block (typical here, where val metrics are computed and never backpropped), swapping `no_grad → inference_mode` is byte-identical and a free speedup. If any eval tensor is reused in a training graph, keep `no_grad`.
- Source: [inference_mode (2.12)](https://docs.pytorch.org/docs/2.12/generated/torch.autograd.grad_mode.inference_mode.html)

---

## 6. Refactoring a ~1100-line training-step function (clean-code patterns)

All of this is **byte-identical by construction — pure structural refactor, SAFE — provided you do not reorder math, change accumulation dtypes, alter RNG-consuming call order, or move ops in/out of the autocast/compile scope.** Mechanical-equivalence is the contract: extracting code into a helper must not change *when* anything executes relative to RNG, optimizer steps, or autocast entry/exit.

- **Extract Method on seams, not arbitrarily.** Pull out cohesive blocks that already have a clear single purpose and a narrow interface: per-fold setup (model/optim/scheduler/criteria construction), the inner train-step (forward → loss → backward → step), the eval/metric pass, checkpoint-selection scoring, logging. Rule of thumb from the literature: extract when a block exceeds ~40 lines or is duplicated. Keep the hot inner loop body inline if splitting it would move ops across the autocast/compile boundary (that boundary is numerics-load-bearing here).
- **Guard clauses / early return over nesting.** Replace nested `if` pyramids (early-stop, timeout, target-F1 cutoffs, padding skips) with top-of-function guard clauses that `return`/`continue` immediately, keeping the happy path flat and linear.
- **Group config into dataclasses.** The repo already has `ExperimentConfig`/`InputsConfig`; pass one config object instead of long positional arg lists (reduces 64-field threading and accidental arg-swaps). Pure plumbing — no numeric effect.
- **Structured logging over `print`.** Use the `logging` module (or the existing `MLHistory`/tracking layer) with key=value fields instead of `print`, so logs are filterable and don't interleave with progress bars. No effect on scored numbers.
- **What to keep inline:** the exact sequence of `optimizer.zero_grad → (autocast) forward → loss → scaler.scale(loss).backward → scaler.step → scaler.update → scheduler.step`, and anything inside the `torch.compile`d region. Extracting *around* it is fine; reordering *within* it is not.
- Sources: [Extract Method (refactoring.guru)](https://refactoring.guru/extract-method), [Replace Nested Conditional with Guard Clauses](https://refactoring.guru/replace-nested-conditional-with-guard-clauses)

---

### One-line verdict table
| Change | Verdict |
|---|---|
| `torch.cuda.amp.autocast/GradScaler` → `torch.amp.*("cuda", …)` / `torch.autocast("cuda", …)` | **Byte-identical — SAFE** (delegates to same class) |
| `is_autocast_enabled()` → `is_autocast_enabled("cuda")` | **Byte-identical — SAFE** (read-only) |
| Rely on `zero_grad(set_to_none=True)` default | **SAFE only if repo isn't currently `False`/all-params-grad**; flipping `False→True` is **NUMERICS-CHANGING** |
| `torch.compile` re-tidy (mode/dynamic/autotune/reset) | **NUMERICS-CHANGING — EXCLUDE**; reproduce frozen compile config exactly; compile-once + `load_state_dict` reuse is sound and doesn't recompile |
| eval `no_grad` → `inference_mode` | **Byte-identical — SAFE** for metrics; only risk is a runtime error if eval tensors re-enter autograd |
| Extract-method / guard-clauses / dataclass / structured logging | **Byte-identical — SAFE** if math/RNG/autocast-scope ordering is preserved |

Key fetched sources: [v2.11.0 autocast source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/cuda/amp/autocast_mode.py), [v2.11.0 GradScaler source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/cuda/amp/grad_scaler.py), [amp.html](https://docs.pytorch.org/docs/2.12/amp.html), [inference_mode](https://docs.pytorch.org/docs/2.12/generated/torch.autograd.grad_mode.inference_mode.html), [zero_grad](https://docs.pytorch.org/docs/2.12/generated/torch.optim.Optimizer.zero_grad.html), [compiler FAQ](https://docs.pytorch.org/docs/stable/torch.compiler_faq.html), [PT2 bitwise-compat blog](https://pytorch.org/blog/training-production-ai-models/), [recompilation guide](https://docs.pytorch.org/docs/stable/compile/programming_model.recompilation.html).

## Code map

Now I'll map the refactoring targets systematically, analyzing the code for self-contained, extractable blocks.

## REFACTORING MAP: Byte-Identical Decomposition Targets

### FILE: `/home/vitor.oliveira/PoiMtlNet/src/training/runners/mtl_cv.py`

#### `train_model()` – Lines 308–1275 (~967 lines actual)

**TIER 1: Pure-Diagnostic Blocks (ZERO RNG RISK)**

**Block 1.1: `_log_epoch_diagnostics` (lines 612–1076, ~465 lines)**
- **Exact range:** Lines 612–1076 (init accumulators, streaming-metric setup, then INSIDE epoch loop after per-batch: grad cosine, encoder trajectory, gate stats, alpha/beta/gamma, GRM gate)
- **Inputs:** `model`, `fold_history`, `next_enc_init`, `cat_enc_init`, `next_enc_prev`, `cat_enc_prev`, `dataloader_next.train.y`, `dataloader_category.train.y`, epoch_idx, task_b_name, task_a_name
- **Outputs:** `diagnostic_payload` dict (logged to `fold_history.log_diagnostic()`); mutates `next_enc_prev`, `cat_enc_prev` (closure state)
- **Side effects:** None on model/optimizer/gradients; purely informational
- **Path classification:** PURE-DIAGNOSTIC (no impact on checkpoint selection, early-stop, or metric ranking)
- **Risk: LOWEST** — Extractable immediately; byte-identical if you preserve `_flatten_encoder()` helper and fold_history reference.

**Block 1.2: `_apply_loss_scale_normalization` (lines 653–664, ~12 lines)**
- **Exact range:** Lines 657–664
- **Inputs:** `task_b_loss`, `task_a_loss`, `pred_task_b.shape[-1]`, `pred_task_a.shape[-1]`, `loss_scale_norm` flag
- **Outputs:** Mutated `task_b_loss`, `task_a_loss`
- **Side effects:** None (pure arithmetic)
- **Path classification:** SCORED (feeds into MTL combiner)
- **Risk: LOW** — Deterministic; no RNG. Extract if you group with loss computation; byte-identical if you test per-batch output.

---

**TIER 2: Scored-Path Blocks (Require Multi-Seed Parity Testing)**

**Block 2.1: `_compute_loss_with_kd` (lines 649–751, ~103 lines)**
- **Exact range:** Lines 649–774 (CE loss, loss-scale norm, THREE KD terms: log_T-KD at 666–674, log_C-KD at 691–729, cat-KD at 734–761, then reg-freeze zero at 770–771)
- **Inputs:** 
  - `pred_task_b`, `truth_task_b`, `pred_task_a`, `truth_task_a` (forward-pass outputs)
  - `next_criterion`, `category_criterion` (CE losses)
  - `model` (for log_T/log_C buffers and gate tracking)
  - `loss_scale_norm`, `log_t_kd_weight`, `log_t_kd_tau`, `log_t_kd_gate`, `log_c_kd_weight`, `log_c_kd_tau`, `log_c_kd_warmup_epochs`, `log_c_kd_ec_lambda`, `cat_kd_weight`, `cat_kd_tau`
  - `epoch_idx`, `_reg_frozen_post_peak` flag
- **Outputs:** `task_b_loss`, `task_a_loss` (scalars, may include KD addends)
- **Side effects:** 
  - Mutates `globals()["_LOGC_FIRED"]`, `globals()["_CATKD_FIRED"]` (diagnostic flags, idempotent one-shot)
  - Writes to model: `model._r5_gate_std` (diagnostic)
- **Path classification:** SCORED (loss directly affects backward pass and gradient flow; selects checkpoints via _val metrics)
- **Constraint:** MUST NOT MOVE: RNG unaffected, but byte-identical requires tests on multiple seeds/states that KD addends compute identically per batch.
- **Risk: MEDIUM** — Extract only after comprehensive per-batch forward-pass parity testing across seeds/configs (log_T-KD gate modes, log_C/cat-KD warmup epochs, ec_lambda variants). Write fixture to compare (epoch, batch) → (loss values) across pre/post-refactor.

---

**TIER 3: Control-Flow & State Transitions (Low RNG Risk if Boundary Unchanged)**

**Block 3.1: `_handle_alpha_unfreeze` (lines 513–526, ~14 lines)**
- **Exact range:** Lines 513–526
- **Inputs:** `epoch_idx`, `alpha_frozen_until_epoch`, `_alpha_unfrozen` flag, `model.next_poi.alpha`
- **Outputs:** Mutates `_alpha_unfrozen`, `model.next_poi.alpha.requires_grad`
- **Side effects:** Parameter-level state change; stdout print
- **Path classification:** SCORED (affects which params receive gradients)
- **Constraint:** Boundary must fire at exact epoch; no reordering of loop iterations.
- **Risk: LOW** — Extract as pure guard block; byte-identical as long as epoch_idx comparison is unchanged.

**Block 3.2: `_handle_category_freeze` (lines 531–543, ~13 lines)**
- **Exact range:** Lines 531–543
- **Inputs:** `epoch_idx`, `freeze_cat_after_epoch`, `_cat_frozen_post_warmup`, `model.category_encoder`, `model.category_poi`
- **Outputs:** Mutates `_cat_frozen_post_warmup`, param requires_grad, model.eval()
- **Side effects:** Parameter-level state, model mode
- **Path classification:** SCORED
- **Risk: LOW** — Same as 3.1.

**Block 3.3: `_handle_region_freeze` (lines 545–573, ~29 lines)**
- **Exact range:** Lines 545–573
- **Inputs:** `epoch_idx`, `reg_freeze_at_epoch`, `_reg_frozen_post_peak`, `model.next_encoder`, `model.next_poi`
- **Outputs:** Mutates `_reg_frozen_post_peak`, param requires_grad, model.eval()
- **Side effects:** Param state, model mode; note: this flag is READ later (line 770) to zero task_b_loss
- **Path classification:** SCORED
- **Risk: LOW** — But note side effect: `_reg_frozen_post_peak` gates loss zeroing on line 770. Extract only with clear state-passing (return bool or mutate closure var in original scope).

---

**TIER 4: Per-Batch Gradient Accumulation & Clipping**

**Block 4.1: `_handle_gradient_accumulation_step` (lines 826–878, ~53 lines)**
- **Exact range:** Lines 826–878 (the `should_step` block: partial-group rescale, alternating-SGD inactive param zero, grad clipping, finiteness guard, step/scheduler)
- **Inputs:** `batch_idx`, `batches_per_epoch`, `gradient_accumulation_steps`, `accumulated_in_group`, `model.parameters()`, `loss`, `max_grad_norm`, `mtl_criterion`, `_alt_inactive_params`, `_mtl_strict`, `epoch_idx`
- **Outputs:** Updated model state (optimizer.step), scheduler step, accumulated_in_group reset to 0
- **Side effects:** Optimizer/scheduler state; conditional gradient clipping + finiteness guard (no-op if step skipped)
- **Path classification:** SCORED (gradient updates)
- **Constraint:** MUST preserve `should_step` condition order & grad-clipping guards (finiteness check at lines 863–873). Reordering breaks the nonfinite-guard's protection.
- **Risk: MEDIUM-HIGH** — Extract as helper but test that step counts, grad norms, and model param updates match per-batch. The alternating_optimizer_step (lines 849–852) path adds complexity.

---

**TIER 5: Validation & Checkpoint Gating (Complex, Scored Path)**

**Block 5.1: `_run_validation_and_update_checkpoints` (lines 1078–1220, ~143 lines)**
- **Exact range:** Lines 1078–1220 (val metrics, OOD label set caching, joint selector computation, state_dict gating, fold_history logging, task_best_tracker update)
- **Inputs:** `model`, `dataloader_next`, `dataloader_category`, evaluation criteria & losses, `DEVICE`, `fold_history`, `task_best_tracker`, `checkpoint_selector`, `joint_min_epoch`, task names & num_classes
- **Outputs:** Updated `fold_history` (val metrics, best tracking), updated `task_best_tracker`, conditional `state_dict`
- **Side effects:** Checkpoint disk writes (inside task_best_tracker), fold_history mutations
- **Path classification:** SCORED (selects which model state is saved & finalized)
- **Constraint:** `evaluate_model()` call (line 1098) must run; checkpoint_selector dispatch (C21: `geom_simple`, `geom_lift`, `joint_f1_mean`) must match the tracker's monitor setting. OOD-restricted metrics (CH06) require train-label cache.
- **Risk: HIGH** — Extract only with full integration tests: same seed/fold → identical state_dict epochs, same checkpoint rank. The interact ion between fold_history.best tracking, task_best_tracker, and the joint_selector is complex.

---

### FILE: `/home/vitor.oliveira/PoiMtlNet/src/data/folds.py`

#### `_create_check2hgi_mtl_folds()` – Lines 1207–1501 (~295 lines)

**TIER 1: Pure I/O & Validation**

**Block 1.1: `_load_and_validate_check2hgi_data` (lines 1228–1306, ~79 lines)**
- **Exact range:** Lines 1228–1306 (engine guard, load X + y_cat + y_region, alignment guards, aux extraction)
- **Inputs:** `state`, `embedding_engine`, `self.task_set` (optional, to detect aux-requiring heads)
- **Outputs:** `X`, `y_cat`, `y_region`, `y_last_region` (or None), `aux_tensor` (or None), `use_aux` flag
- **Side effects:** File I/O (load parquets), validation logging
- **Path classification:** PURE-DIAGNOSTIC (input validation only; no fold-membership RNG)
- **Constraint:** Must preserve row alignment check (userid parity between next.parquet and next_region.parquet).
- **Risk: LOWEST** — Extract immediately; byte-identical as long as parquet paths are unchanged.

**Block 1.2: `_resolve_per_task_input_modality` (lines 1338–1366, ~29 lines)**
- **Exact range:** Lines 1338–1366 (dispatch on `task_a_input_type`/`task_b_input_type` to build x_task_a, x_task_b)
- **Inputs:** `task_a_input_type`, `task_b_input_type`, `state`, `embedding_engine`, `x_checkin`
- **Outputs:** `x_task_a`, `x_task_b` (tensors)
- **Side effects:** Lazy import of region_sequence builders
- **Path classification:** DETERMINISTIC I/O (no RNG)
- **Risk: LOW** — Extract as pure logic dispatch; byte-identical if input types are unchanged.

---

**TIER 2: Fold Split & Index Recording (On RNG Consumption Path)**

**Block 2.1: `_record_fold_indices_and_manifests` (lines 1398–1417, ~20 lines)**
- **Exact range:** Lines 1398–1417 (loop over splits from StratifiedGroupKFold, record fold indices & manifests)
- **Inputs:** `splits` (list of (train_idx, val_idx) from SGKFold.split), `self.n_splits`, `self.seed`
- **Outputs:** Updates `self._fold_indices[TaskType.NEXT]`, `self._fold_indices[TaskType.CATEGORY]`, `self._fold_manifests`
- **Side effects:** None (pure data structure updates)
- **Path classification:** SCORED/FOLD-MEMBERSHIP (the indices ARE the fold split; they gate which rows train/val)
- **Constraint:** CANNOT extract into separate function that re-iterates splits — you'd lose the exact (fold_idx, train_idx, val_idx) pairing. Must stay inside the SGKFold loop.
- **Risk: MEDIUM** — If you refactor, ensure loop order over `enumerate(splits)` is preserved exactly. The fold_idx numbering (0-indexed) is baked into downstream log_T filenames (1-indexed fold{i_fold+1}).

---

**TIER 3: Lazy Fold Builder (Encapsulates Loaders & Sampling)**

**Block 3.1: `_build_fold` inner function (lines 1419–1495, ~77 lines)**
- **Exact range:** Lines 1419–1495 (the nested _build_fold(fold_idx) that builds train/val loaders per fold)
- **Inputs:** `fold_idx` (integer), captured: `splits`, `x_task_a`, `x_task_b`, `y_cat_tensor`, `y_region_tensor`, `aux_tensor` (or None), `use_aux`, `self.*` (batch_size, seed, use_weighted_sampling, aligned_pairing, task_a_input_type)
- **Outputs:** `FoldResult(next=..., category=..., joint_train_loader=...)`
- **Side effects:** DataLoader creation (RNG-seeded via sampler generators), model file I/O via gc.collect()
- **Path classification:** SCORED/SELECTION (sampler RNG from seed determines shuffle order within fold; wrong seed changes train distribution)
- **Constraint:** RNG draws inside WeightedRandomSampler(generator=...) and aligned_pairing's _create_aligned_joint_loader(...seed + fold_idx). The seed MUST be fold-offset for distinct-per-fold but reproducible shuffle.
- **Risk: MEDIUM** — Extract only after parity testing: same seed/fold_idx → byte-identical DataLoader shuffle order (or verify via _get_sampler_indices or by comparing first-batch order across runs).

---

#### `_create_mtl_folds()` – Lines 1032–1205 (~173 lines)

**TIER 1: POI Classification (Deterministic, Already Extracted)**

**Block 1.1: `_classify_pois` (lines 866–885, ~20 lines) — ALREADY EXTRACTED**
- Used by _create_mtl_folds at line 1101 to partition POIs into train_excl, val_excl, ambiguous.
- Pure function, zero RNG.

**TIER 2: Category Index Derivation (Core User-Isolation Protocol)**

**Block 2.1: `_derive_category_fold_from_poi_classification` (lines 1099–1131, ~33 lines)**
- **Exact range:** Lines 1099–1131 (classify POIs, then derive category train/val indices via np.isin)
- **Inputs:** `poi_users` (dict built from raw checkins), `train_users`, `val_users` (sets from next-task fold split), `cat_placeids` (np array), `use_poi_protocol` flag, `_cat_fold_splits` (fallback for pre-Phase-2)
- **Outputs:** `train_cat_idx`, `val_cat_idx` (np arrays of category fold indices)
- **Side effects:** None (pure array operations)
- **Path classification:** SCORED/FOLD-MEMBERSHIP (the category indices ARE the fold split; wrong indices leak train data into val)
- **Constraint:** MUST preserve user→POI→placeid→cat_placeid chain. The str-coercion (line 1127) is critical for dtype matching. Cannot reorder the np.isin calls.
- **Risk: MEDIUM** — Extract only after parity testing: same user split → byte-identical category indices. Test on multiple seeds to verify POI classification is deterministic.

**Block 2.2: `_build_mtl_fold_result` (lines 1150–1203, ~54 lines)**
- **Exact range:** Lines 1150–1203 (build fold_result with category+next dataloaders, append fold manifest)
- **Inputs:** `fold_idx`, `train_next_idx`, `val_next_idx`, `train_cat_idx`, `val_cat_idx`, tensors (x_next, y_next, x_cat, y_cat), self.batch_size, self.seed, self.use_weighted_sampling, fold_manifests data
- **Outputs:** `FoldResult` (added to fold_results dict), appended fold_manifest
- **Side effects:** DataLoader creation (RNG-seeded), gc.collect() calls
- **Path classification:** SCORED (sampler RNG)
- **Constraint:** Same as _build_fold in check2hgi path — seed must be stable per fold.
- **Risk: MEDIUM** — Extract but test sampler order parity.

---

### `train_with_cross_validation()` – Lines 1279–1936 (~657 lines)

**TIER 1: Setup & Utilities**

**Block 1.1: Task naming & profiler init (lines 1286–1308, ~23 lines)**
- Pure setup, extractable but not high-value (already concise).

**TIER 2: Scored-Path, Complex Gating (Per-Fold Log_T Swapping)**

**Block 2.1: `_apply_per_fold_transition_priors` (lines 1322–1526, ~205 lines)**
- **Exact range:** Lines 1331–1526 (the entire per_fold_dir conditional block with log_T/log_C guard chaining)
- **Inputs:** `config.per_fold_transition_dir`, `config.embedding_engine`, `config.seed`, `i_fold`, `config.k_folds`, `config.log_t_kd_weight`, `config.log_c_kd_weight`, `config.cat_kd_weight`, `ts` (task_set)
- **Outputs:** Modified `per_fold_model_params` dict (updated task_b.head_params with transition_path, colocation_path)
- **Side effects:** File system checks (stat, exists), torch.load, logging & validation errors
- **Path classification:** SCORED/SELECTION (log_T leaks val transitions if mismatched; silently inflates reg Acc@10 by +8–40 pp)
- **Constraint:** RNG-INDEPENDENT but DATA-DEPENDENT. Must preserve:
  - Stale mtime check (lines 1365–1379)
  - n_splits parity check (lines 1380–1425)
  - Engine parity check (lines 1426–1465)
  - Per-fold log_C conditionals (lines 1474–1517)
  - The 1-indexed fold numbering (line 1341: `fold{i_fold + 1}`)
- **Risk: HIGH** — This block has MANY guards; each can silently fail. Extract only if you:
  1. Write integration tests: same seed/fold/engine → identical model_state/metrics
  2. Verify guards fire correctly (mtime, n_splits, engine, seed comparisons)
  3. Document the 1-indexed filename convention
  4. Add a fixture that compares per-fold runs with/without per_fold_dir set to ensure metrics are byte-identical when using the correct log_T

---

**TIER 3: Loss & Optimizer Setup (Moderate, Control-Flow Heavy)**

**Block 3.1: `_setup_loss_criteria` (lines 1598–1755, ~158 lines)**
- **Exact range:** Lines 1598–1755 (create mtl_criterion, per-task class weights, optional calibrated loss for cat)
- **Inputs:** `config.mtl_loss`, `config.mtl_loss_params`, `config.epochs`, `config.use_class_weights`, `config.use_class_weights_reg/cat` overrides, `config.loss_calibration`, `dataloader_next.train.y`, `dataloader_category.train.y`, task num_classes
- **Outputs:** `mtl_criterion`, `alpha_next`, `alpha_cat`, `next_criterion`, `category_criterion`
- **Side effects:** File I/O if building_calibrated_loss (reads train stats)
- **Path classification:** SCORED (loss directly affects training)
- **Constraint:** class_weight computation is deterministic but order-dependent on the input y tensors.
- **Risk: MEDIUM** — Extract but test that class weights compute identically (bincount order).

**Block 3.2: `_setup_optimizer_and_scheduler` (lines 1606–1656, ~51 lines)**
- **Exact range:** Lines 1606–1656 (per-head vs single-LR optimizer, scheduler setup)
- **Inputs:** `model`, `config.cat_lr`, `config.reg_lr`, `config.shared_lr`, `config.weight_decay`, `config.optimizer_eps`, `config.max_lr`, `config.epochs`, `batches_per_epoch`, `config.gradient_accumulation_steps`, config scheduler params
- **Outputs:** `optimizer`, `scheduler`
- **Side effects:** None (pure object construction)
- **Path classification:** SCORED (optimizer state)
- **Risk: LOW** — Extract as pure factory; byte-identical if config params are unchanged.

---

**TIER 4: Stream Freezing & Smoke Print**

**Block 4.1: `_apply_stream_freeze` (lines 1561–1587, ~27 lines)**
- **Exact range:** Lines 1561–1587 (freeze_cat_stream, freeze_reg_stream setup + smoke print)
- **Inputs:** `config.freeze_cat_stream`, `config.freeze_reg_stream`, `model`, `optimizer`, first fold flag
- **Outputs:** Model param requires_grad mutations, stdout print
- **Side effects:** Parameter-level state
- **Path classification:** SCORED (affects which params are trained)
- **Risk: LOW** — Extract but preserve param group filtering in optimizer (the freeze MUST be reflected in requires_grad so AdamW skips weight decay).

---

**TIER 5: Task-Best Snapshots (High-Value but Complex)**

**Block 5.1: `_initialize_task_best_tracker` (lines 1800–1818, ~19 lines)**
- **Exact range:** Lines 1800–1818
- **Inputs:** `config.save_task_best_snapshots`, `history`, `task_a_name`, `task_b_name`, `results_path`
- **Outputs:** `task_best_tracker`, `task_best_save_dir`
- **Side effects:** Directory creation
- **Path classification:** PURE-CONTROL (checkpoint archival, no gradient impact)
- **Risk: LOWEST** — Extract as pure factory.

**Block 5.2: `_save_task_best_snapshots` (lines 1867–1887, ~21 lines)**
- **Exact range:** Lines 1867–1887
- **Inputs:** `task_best_tracker`, `task_best_save_dir`, `fold_idx`
- **Outputs:** None (side-effect: writes .pt files)
- **Side effects:** Disk I/O
- **Path classification:** PURE-CONTROL
- **Risk: LOWEST** — Extract.

---

## RANKED EXTRACTION PLAN (Highest Value / Lowest Risk First)

| Rank | Function | Location | Lines | Category | Value | Risk | Notes |
|------|----------|----------|-------|----------|-------|------|-------|
| 1 | `_log_epoch_diagnostics` | mtl_cv.py 612–1076 | 465 | Diagnostic | ⭐⭐⭐ | NONE | Pure assembly; no impact on metrics. **EXTRACT NOW** |
| 2 | `_load_and_validate_check2hgi_data` | folds.py 1228–1306 | 79 | I/O + Validation | ⭐⭐ | NONE | Input guards only. **EXTRACT NOW** |
| 3 | `_resolve_per_task_input_modality` | folds.py 1338–1366 | 29 | Logic | ⭐ | NONE | Dispatch only. **EXTRACT NOW** |
| 4 | `_save_task_best_snapshots` | mtl_cv.py 1867–1887 | 21 | Control | ⭐ | NONE | Disk I/O only. **EXTRACT NOW** |
| 5 | `_initialize_task_best_tracker` | mtl_cv.py 1800–1818 | 19 | Control | ⭐ | NONE | Factory only. **EXTRACT NOW** |
| 6 | `_apply_stream_freeze` | mtl_cv.py 1561–1587 | 27 | Control | ⭐ | LOW | Preserve requires_grad state. **EXTRACT W/ TEST** |
| 7 | `_setup_optimizer_and_scheduler` | mtl_cv.py 1606–1656 | 51 | Setup | ⭐⭐ | LOW | Factory; test param group counts. **EXTRACT W/ TEST** |
| 8 | `_setup_loss_criteria` | mtl_cv.py 1598–1755 | 158 | Setup | ⭐⭐ | LOW | Class-weight order; test determinism. **EXTRACT W/ TEST** |
| 9 | `_apply_loss_scale_normalization` | mtl_cv.py 657–664 | 12 | Scored | ⭐ | LOW | Group with loss computation. **EXTRACT W/ BYTE-TEST** |
| 10 | `_compute_loss_with_kd` | mtl_cv.py 649–774 | 126 | Scored | ⭐⭐⭐ | MEDIUM | Core loss logic; multiple KD branches. **EXTRACT W/ MULTI-SEED PARITY TEST** |
| 11 | Freeze boundaries (3 blocks) | mtl_cv.py 513–573 | 57 | Control | ⭐ | LOW | Preserve epoch-boundary guards. **EXTRACT W/ TEST** |
| 12 | `_handle_gradient_accumulation_step` | mtl_cv.py 826–878 | 53 | Scored | ⭐⭐ | MEDIUM | Complex step logic; test grad norms. **EXTRACT W/ BYTE-TEST** |
| 13 | `_derive_category_fold_from_poi_classification` | folds.py 1099–1131 | 33 | Fold-Membership | ⭐⭐⭐ | MEDIUM | Core MTL split; test parity on user splits. **EXTRACT W/ PARITY GATE** |
| 14 | `_build_fold` (check2hgi) | folds.py 1419–1495 | 77 | Fold-Membership | ⭐⭐ | MEDIUM | Sampler RNG; test shuffle parity. **EXTRACT W/ RNG PARITY TEST** |
| 15 | `_apply_per_fold_transition_priors` | mtl_cv.py 1331–1526 | 195 | Scored/Selection | ⭐⭐⭐ | HIGH | Log_T leak risk (8–40 pp error). **EXTRACT ONLY W/ INTEGRATION + MULTI-SEED PARITY TEST** |
| 16 | `_run_validation_and_update_checkpoints` | mtl_cv.py 1078–1220 | 143 | Scored/Selection | ⭐⭐⭐ | HIGH | Checkpoint gating complexity. **EXTRACT ONLY W/ FULL REGRESSION TEST** |

---

## MUST NOT MOVE (RNG/State Invariants)

1. **seed_everything() call** (line 1317 in train_with_cross_validation): Must fire BEFORE fold creation; do NOT extract into fold builder.
2. **StratifiedGroupKFold.split() call** (lines 1083, 1385): Draws from global numpy RNG; must stay in fold-creation sequence to preserve downstream RNG state.
3. **Per-fold log_T filename keying** (e.g., line 1341: `fold{i_fold + 1}`): Fold numbering is 1-indexed on disk; `i_fold` (0-indexed dict key) MUST use `i_fold + 1` in filenames.
4. **Per-batch DataLoader shuffle order**: seed-dependent; extract sampler logic only with RNG parity testing.
5. **fold_history task names** (lines 1287–1288, 373–374): Slot A/B names depend on task_set; must stay synchronized across train_model, train_with_cross_validation, and logging code.

---

## Multi-Seed Parity Test Fixtures (Required Before Extraction #10, #13, #15)

**Test pattern for scored-path extractions:**

```python
# Fixture: run_with_seed_fold_pair(seed, fold_idx, config, extract_before=True)
# – create folds with seed S, train fold F under extract_before=True (original code)
# – store metrics + model_state + checkpoint epochs
# – run again with extract_before=False (refactored code)
# – assert metrics, checkpoint epochs, model_state are byte-identical
```

Apply to: `_compute_loss_with_kd` (per-batch loss outputs), `_derive_category_fold_from_poi_classification` (fold indices), `_apply_per_fold_transition_priors` (model metrics under per-fold log_T).

---

## Summary

**Immediate wins (Extract today, zero tests needed):**
- Blocks 1–5 above: ~595 lines extracted, diagnostic/I/O/factory code only

**High-value but requires parity gates (Extract next sprint with tests):**
- Block 10 (_compute_loss_with_kd): 126 lines, SCORED, multi-seed parity mandatory
- Block 13 (_derive_category_fold...): 33 lines, FOLD-MEMBERSHIP, parity on user splits
- Block 15 (_apply_per_fold_transition_priors): 195 lines, SCORED/SELECTION, integration + multi-seed + guard validation tests

**Deferrable (complex, high integration cost):**
- Block 16 (_run_validation_and_update_checkpoints): 143 lines, SCORED, full regression suite needed

Total potential extraction: ~1100 lines (50% of train_model + train_with_cross_validation combined), with ~595 lines requiring zero tests and ~500 lines requiring comprehensive byte-parity testing before merge.

SendUserFile({ files: ["refactoring_map.txt"], status: "normal" })

## Final ordered plan

I have verified the code against the map. Key finding: the AMP/`set_to_none` "modernization" wins are already implemented in the target files, so this plan is almost entirely pure structural extraction. Here is the final ordered work plan.

---

# FINAL ORDERED WORK PLAN — slimming `train_model` / `train_with_cross_validation` / fold creators (FROZEN-NUMERICS)

## Grounding facts I verified before planning (these change the plan vs. the map)

1. **AMP-API modernization is ALREADY DONE in scope.** `mtl_cv.py:437-438`, `mtl_eval.py:103`, `shared_evaluate.py:27`, `mtl_validation.py`, `_single_task_train.py:75` all already use the modern `torch.autocast(DEVICE.type, dtype=…)` form. There is **no `torch.cuda.amp.*` and no `GradScaler` anywhere in the three target functions or the fold creators** (mtl_cv.py runs fp16 autocast with *no* scaler by design). The only legacy `torch.cuda.amp.*` left in the repo is `research/embeddings/space2vec/space2vec.py:51,76,244` — **OUT OF SCOPE**. → No AMP edits in this plan.
2. **`zero_grad(set_to_none=True)` is ALREADY explicit** at `mtl_cv.py:640` and `:877`; there is **no `set_to_none=False` anywhere in the repo**. → No `set_to_none` edit; per digest §3 the only numerics-changing variant (flip `False→True`) does not apply. Leave as-is.
3. **`is_autocast_enabled`** appears only at `src/models/next/next_stan/head.py:130` and already passes `x.device.type`. Out of scope, already modern.
4. **`inference_mode` is used nowhere.** The two `no_grad` sites are `mtl_cv.py:897` (train-metric, diagnostic-only, *no forward inside the block* → zero speedup) and `mtl_eval.py:136` `@torch.no_grad() evaluate_model` (the **canonical scored VAL path**). → See EXCLUDE list; not a free win here.
5. **Map correction (important).** The map's "Block 1.1 `_log_epoch_diagnostics`, lines 612–1076, one 465-line helper" is **not contiguous** — that range interleaves the *scored* inner loop (KD loss 649–774, train-metric `no_grad` 897, grad-accumulation step 826–878). You **cannot** lift 612–1076 as a single function. Extract the *leaf* diagnostic computations individually instead (below).
6. **Already-extracted helpers** (don't re-do): `_flatten_task_grads` (69), `_compute_gradient_cosine` (84), `_compute_joint_selectors` (180), `_log_t_kd_loss` (216). Only **one** nested closure remains in `train_model`: `_flatten_encoder` at `mtl_cv.py:473`. `_build_fold` (folds.py:1419) is a closure inside `_create_check2hgi_mtl_folds`.

---

## PHASE 0 — Build the parity oracle FIRST (no source edits yet)

**Step 0 — Stand up the byte-parity harness.** All scored gates run in **eager mode (no `--compile`) + fp32 (`MTL_DISABLE_AMP=1`)** so any diff is attributable to the refactor, not to Inductor/fp16 nondeterminism. Capture, on a small state (AL, H3-alt recipe, seeds you'll reuse):
- **G-FOLD digest:** sha256 of `self._fold_indices[NEXT]` + `[CATEGORY]` (all folds) and the first-3-batch sample-index order of every train loader, fixed seed.
- **G-LOSS trace:** per-`(epoch,batch)` tuple `(task_a_loss, task_b_loss, combined)` for fold 0, ≥2 seeds.
- **G-STEP trace:** G-LOSS + per-optimizer-step `total_norm` + sha256 of `model.state_dict()` after each step.
- **G-MULTISEED:** full eager-fp32 run at seeds {0,1,7,100} on AL — per-task **`diagnostic_best_epochs`** metrics (from `fold_info.json`, NOT `full_summary.json` — see memory `ref_mtl_metric_field`) + selected-checkpoint epoch.
- **G-COMPILE-FINAL (sanity, run once at the very end):** one champion-G run with the *frozen board recipe* (`--compile --tf32`, real precision) at AL, confirm it still reproduces the §0.1 cell within the recipe's own run-to-run tolerance.
Risk: **SAFE** (read-only). Commit the captured baselines as fixtures.

---

## PHASE 1 — SAFE pure-block extractions (zero scored-path impact)

> Gate for every Phase-1 step: **G-DIAG** = one seed/fold eager-fp32 smoke; assert `fold_info.json` diagnostic_best epochs + selected-checkpoint epoch + per-task val metrics are byte-identical, **and** the diagnostic payload is identical. These blocks have no RNG and don't touch model/optim/grad, so G-DIAG is sufficient.

**Step 1 — Promote `_flatten_encoder` closure to module fn.**
`mtl_cv.py:473` → `def _flatten_encoder(encoder) -> torch.Tensor`. Pure (concatenates encoder params). Prereq for Step 2. Inputs: encoder module. Output: 1-D tensor. Side-effects: none. **SAFE**. Gate: G-DIAG.

**Step 2 — Extract the leaf epoch-diagnostic computations** (the map's mis-scoped "Block 1.1"), as 2–3 small helpers, NOT one function:
- `def _log_encoder_trajectory(fold_history, model, next_enc_init, cat_enc_init, next_enc_prev, cat_enc_prev, epoch_idx) -> tuple[Tensor, Tensor]` (returns new `prev` vectors; uses `_flatten_encoder`).
- `def _log_gate_film_stats(fold_history, model, epoch_idx) -> None` (gate stats, alpha/beta/gamma, GRM gate).
Inputs/outputs as above; **side-effects:** only `fold_history.log_diagnostic(...)` + closure-state `*_enc_prev` returned (not mutated in place across the boundary). Path: PURE-DIAGNOSTIC. **SAFE**. Gate: G-DIAG. *Do not* pull any scored line (KD 649–774, train-metric 897, step 826–878) into these.

**Step 3 — `_load_and_validate_check2hgi_data`.**
`folds.py:1228–1306` → `def _load_and_validate_check2hgi_data(self, state, embedding_engine) -> tuple[X, y_cat, y_region, y_last_region|None, aux_tensor|None, use_aux: bool]`. Inputs: state, engine, `self.task_set`. Side-effects: parquet reads + validation logging (preserve the userid row-alignment guard). No RNG. **SAFE**. Gate: **G-FOLD** (digest must be byte-identical) + G-UNIT (existing `tests/test_data`).

**Step 4 — `_resolve_per_task_input_modality`.**
`folds.py:1338–1366` → `def _resolve_per_task_input_modality(self, task_a_input_type, task_b_input_type, state, embedding_engine, x_checkin) -> tuple[x_task_a, x_task_b]`. Deterministic dispatch (lazy region-sequence import). **SAFE**. Gate: **G-FOLD**.

**Step 5 — Task-best snapshot factory + writer.**
`mtl_cv.py:1800–1818` → `def _initialize_task_best_tracker(config, history, task_a_name, task_b_name, results_path) -> tuple[tracker|None, save_dir|None]`.
`mtl_cv.py:1867–1887` → `def _save_task_best_snapshots(task_best_tracker, task_best_save_dir, fold_idx) -> None`.
Side-effects: dir creation / `.pt` writes only; no gradient path. **SAFE**. Gate: G-DIAG (assert same `.pt` filenames/epochs emitted).

---

## PHASE 2 — Setup/factory extractions (low risk; deterministic but scored-adjacent)

> Gate: **G-STEP** (these define optimizer/loss state, so verify per-step grad-norm + param checksums match for fold 0 × 2 seeds) + G-UNIT.

**Step 6 — `_setup_optimizer_and_scheduler`.**
`mtl_cv.py:1606–1656` → `def _setup_optimizer_and_scheduler(model, config, batches_per_epoch) -> tuple[Optimizer, Scheduler]`. Pure construction (per-head vs single-LR groups, OneCycle/cosine/constant). **SAFE→MED.** Constraint: preserve **param-group ordering and membership exactly** (AdamW weight-decay application is order/group-sensitive). Gate: **G-STEP** + assert identical `optimizer.param_groups` count/lrs.

**Step 7 — `_setup_loss_criteria`.**
`mtl_cv.py:1598–1755` → `def _setup_loss_criteria(config, dataloader_next, dataloader_category, n_classes_next, n_classes_cat) -> tuple[mtl_criterion, alpha_next, alpha_cat, next_criterion, category_criterion]`. Constraint: class-weight `bincount` order is determinism-load-bearing — do not reorder the `.train.y` reductions. Side-effects: may read train-stats file for calibrated loss. **MED.** Gate: **G-STEP** (loss values at batch 0 must match bit-for-bit) + G-UNIT (`tests/test_losses`).

**Step 8 — Stream/param freeze handlers** (group the four small boundary blocks):
- `_apply_stream_freeze(config, model, optimizer, is_first_fold)` ← `mtl_cv.py:1561–1587`.
- `_handle_alpha_unfreeze(model, epoch_idx, alpha_frozen_until_epoch, already_unfrozen) -> bool` ← `513–526`.
- `_handle_category_freeze(model, epoch_idx, freeze_cat_after_epoch, already_frozen) -> bool` ← `531–543`.
- `_handle_region_freeze(model, epoch_idx, reg_freeze_at_epoch, already_frozen) -> bool` ← `545–573`.
Constraints: each MUST fire at the exact `epoch_idx` boundary; return the updated flag so the caller keeps the state in `train_model` scope (do **not** hide the flag in the helper). **Critically** `_reg_frozen_post_peak` is read downstream at `:770` to zero `task_b_loss` — thread it back explicitly. Side-effects: `requires_grad` mutations + `model.eval()` + stdout. **MED.** Gate: **G-STEP** + G-MULTISEED (freeze epochs must land identically across seeds).

---

## PHASE 3 — Scored-path extractions (multi-seed eager parity mandatory)

> Gate for every Phase-3 step: **G-LOSS + G-STEP + G-MULTISEED**, all eager-fp32. Extract one at a time; commit only when all three are byte-identical.

**Step 9 — `_compute_loss_with_kd` (fold loss-scale-norm into it).**
`mtl_cv.py:649–774` (incl. loss-scale-norm 657–664, log_T-KD via existing `_log_t_kd_loss` 666–674, log_C-KD 689–729, cat-KD 734–761, reg-freeze zero 770–771) →
`def _compute_loss_with_kd(pred_task_b, truth_task_b, pred_task_a, truth_task_a, next_criterion, category_criterion, model, epoch_idx, kd_cfg, reg_frozen_post_peak) -> tuple[task_b_loss, task_a_loss]` where `kd_cfg` bundles the 9 KD scalars (`loss_scale_norm, log_t_kd_*, log_c_kd_*, cat_kd_*`).
Constraints: **must stay inside the `with _autocast_ctx:` scope** (caller keeps the `with`; helper body runs within it) — moving CE/KD out of autocast is numerics-changing. Preserve the idempotent one-shot global flags `_LOGC_FIRED`/`_CATKD_FIRED` and `model._r5_gate_std` writes. **MED.** Gate: G-LOSS per-(epoch,batch) across KD modes (gate on/off, warmup, `ec_lambda` variants) + G-MULTISEED.

**Step 10 — `_handle_gradient_accumulation_step`.**
`mtl_cv.py:826–878` → `def _handle_gradient_accumulation_step(model, optimizer, scheduler, mtl_criterion, batch_idx, batches_per_epoch, grad_accum_steps, accumulated_in_group, max_grad_norm, alt_inactive_params, epoch_idx, strict) -> int` (returns reset `accumulated_in_group`).
Constraints: preserve the EXACT `should_step` condition, partial-group rescale, alternating-SGD inactive-param zeroing (849–852), grad-clip → **finiteness guard (`guard_finite_step`, 863–873)** → `optimizer.step` → `scheduler.step` → `zero_grad(set_to_none=True)` order. Reordering breaks the non-finite guard. **MED-HIGH.** Gate: G-STEP (per-step grad-norm + param checksum) + G-MULTISEED.

**Step 11 — MTL fold-membership + builder extractions in folds.py.**
- `_derive_category_fold_from_poi_classification(poi_users, train_users, val_users, cat_placeids, use_poi_protocol, cat_fold_splits) -> (train_cat_idx, val_cat_idx)` ← `1099–1131`. Preserve the user→POI→placeid→cat chain and the `str`-coercion (`:1127`); do not reorder the `np.isin` calls.
- `_build_mtl_fold_result(...)` ← `1150–1203`; `_build_fold(...)` ← `1419–1495`. These create DataLoaders whose **sampler RNG is seeded `seed + fold_idx`** — the helper must receive `fold_idx` and reproduce the offset exactly.
**MED.** Gate: **G-FOLD** (fold-index digest + first-batch shuffle order byte-identical) + G-MULTISEED.

---

## PHASE 4 — RISKY scored/selection blocks (hardest gates; extract last, or stop here)

> Gate: full regression suite + **G-MULTISEED at {0,1,7,100} on AL *and* one large state (FL)** + explicit guard-firing assertions. If any gate diverges, **revert and leave inline.**

**Step 12 — `_apply_per_fold_transition_priors`.**
`mtl_cv.py:1331–1526` → `def _apply_per_fold_transition_priors(config, per_fold_model_params, i_fold, task_set) -> dict`.
Constraints (each a silent-failure surface): the **1-indexed `fold{i_fold+1}`** filename convention (`:1341`); the **stale-mtime guard** (1365–1379); **n_splits parity** (1380–1425); **engine parity** (1426–1465); per-fold log_C conditionals (1474–1517). A mismatched log_T silently inflates reg Acc@10 by **+8…+40 pp** (see memory `feedback_p1_default_logT_leaks`). **RISKY.** Gate: G-MULTISEED + a fixture asserting each guard still fires (feed a stale/short/wrong-engine log_T and confirm the same error/skip).

**Step 13 — `_run_validation_and_update_checkpoints`.**
`mtl_cv.py:1078–1220` → `def _run_validation_and_update_checkpoints(model, dataloader_next, dataloader_category, eval_criteria, fold_history, task_best_tracker, checkpoint_selector, joint_min_epoch, names, num_classes) -> state_dict|None`.
Constraints: the `evaluate_model()` call (`:1098`) is the **canonical scored VAL** pass; the `checkpoint_selector` dispatch (`geom_simple`/`geom_lift`/`joint_f1_mean`, C21) must match the tracker's monitor; OOD-restricted metrics (CH06) need the train-label cache. **RISKY.** Gate: full regression + G-MULTISEED asserting **identical selected-checkpoint epoch and saved `state_dict` hash** per seed/fold.

---

## EXCLUDE — numerics-changing or no-value; do NOT touch

- **`torch.compile` "hygiene"** — any change to `mode`/`dynamic`/autotune/`reset` timing or moving ops in/out of the compiled region. Frozen numbers were produced with a specific Inductor config; per digest §4 this reorders FP ops. **EXCLUDE.** Reproduce the compiled region's inputs/dtype/autocast scope **exactly**; only refactor *around* it.
- **`zero_grad` set_to_none flip** — already `True`; per digest §3 flipping is the one rename-looking numerics-changer. Never flip; don't "tidy" it to rely-on-default either (keep explicit `True`).
- **`no_grad → inference_mode` on `evaluate_model` (`mtl_eval.py:136`)** — the canonical scored VAL path, out of slimming scope; conservatively **EXCLUDE** (values are equal per digest §5, but it's the frozen-metric path and a runtime-error risk if any val tensor re-enters a graph; not worth it). The `no_grad` at `mtl_cv.py:897` has **no forward inside** → zero speedup; leave as `no_grad`.
- **The streaming train-metric block (mtl_cv.py ~885–960)** — already a perf-exact CPU/GPU-reduction path with an explicit "do NOT apply to VAL / do not change reduction" warning. Do not "simplify" reductions, accumulation dtype, or CPU-vs-GPU placement.
- **RNG anchors — MUST NOT MOVE:** `seed_everything(config.seed + i_fold)` (`mtl_cv.py:1317`); `torch.manual_seed/np.random.seed(seed)` (`folds.py:953-954`); every `StratifiedGroupKFold(...).split(...)` (`folds.py:996,1089,1397`); sampler `torch.Generator()` (`folds.py:415,560`); `np.random.seed(worker_seed)` (`:335`). Extracting around these is fine; changing call order is numerics-changing.
- **Inner-loop math order** — the `autocast forward → loss → backward → (accum) → clip → finiteness-guard → step → scheduler.step → zero_grad` sequence stays inline and in order (digest §6 "keep inline").

---

## Sequencing recommendation

Run **13 steps in 5 phases, strictly gated, one commit per step.** First build the Phase-0 eager-fp32 parity oracle (G-FOLD digest, G-LOSS/G-STEP traces, G-MULTISEED on AL) — **do not edit source until the baselines are captured.** Land all of Phase 1 (Steps 1–5, the genuinely free pure-diagnostic/I-O/factory lifts, ~600 lines) behind the cheap **G-DIAG/G-FOLD** gate; this is where the bulk of the slimming value sits at near-zero risk, and it's a clean stopping point if time is short. Phase 2 (setup/factories) gates on **G-STEP**; Phase 3 (loss/grad-step/fold-membership) is the first place numbers can move, so gate every step on **G-LOSS + G-STEP + G-MULTISEED** and never batch two scored extractions into one commit. Defer Phase 4 (`per_fold_transition_priors`, validation/checkpoint) to last, behind the full regression suite plus G-MULTISEED on **both** a small (AL) and large (FL) state with explicit guard-firing fixtures. **Hard rule: if any multi-seed eager-fp32 parity ever diverges by a single ULP, revert that step immediately and leave the block inline — a slimmer file is never worth perturbing a frozen number.** Note that there is **no AMP/`set_to_none`/`is_autocast_enabled` modernization work** to do (already modern in scope), so this plan is ~100% structural extraction, which is the safest class — and the final **G-COMPILE-FINAL** champion-G run (real `--compile --tf32` recipe at AL) is the closing sanity check that the frozen board path still reproduces §0.1.