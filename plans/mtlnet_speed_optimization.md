# MTLnet Training Speed Optimization Plan

**Author:** Claude Opus 4.6 (analysis from 2026-04-11 session)
**Audience:** Future Claude Code session that will implement the optimizations
**Scope:** Performance-only changes. **No change to model math, loss computation, RNG-dependent behavior, or metric values.**
**Repo root:** `/Users/vitor/Desktop/mestrado/ingred`
**Target venv:** `.venv_new` (Python 3.12, PyTorch 2.9.1, MPS on Apple Silicon)

---

## 1. Context you need before touching anything

### 1.1 Canonical baselines — DO NOT REGRESS

There are TWO reference points — the authoritative one is the Nov 5 baseline, but a performance-only change should also reproduce the current post-fix numbers. Any optimization in this plan should match the "Current post-fix" column within ±1pp (the row-to-row noise seen across same-code re-runs on MPS).

**Nov 5, 2025 baseline** (read directly from `results/baselines/dgi/{state}/mtlnet_*/summary/full_summary.json`):

| State | Cat F1 | Cat Acc | Next F1 | Next Acc |
|---|---|---|---|---|
| Alabama | **0.4625** | **54.19%** | **0.2659** | **31.29%** |
| Arizona | **0.4759** | **54.73%** | **0.2847** | **31.66%** |

**Current post-fix metrics** (2026-04-10/11, after the three fixes below + the NashMTL ECOS solver fix in PR #7):

| State | Cat F1 | Cat Acc | Next F1 | Next Acc | Total time |
|---|---|---|---|---|---|
| Alabama | ~0.452 | ~54.1% | ~0.265 | ~36.8% | ~22 min |
| Arizona | ~0.453 | ~53.3% | ~0.273 | ~37.9% | ~22-29 min (varies w/ system load) |

**Read this carefully — the current code does NOT fully reproduce the Nov 5 baseline:**
- Cat F1 is ~0.01 below baseline on both states.
- Cat Acc matches baseline on Alabama, ~1.5pp below on Arizona.
- Next F1 matches baseline on Alabama, slightly below on Arizona.
- **Next Acc is ~5-6pp ABOVE baseline** on both states — this is a bonus from dropping `weight=alpha_next` (combined with the fixed NashMTL solver doing real gradient balancing instead of the silent `[1,1]` collapse it was doing under the Nov 5 baseline).

The ~0.01 Cat F1 shortfall is likely stochastic (different DGI embedding seed + different PyTorch kernel numerics vs the Nov 5 run). It is within run-to-run noise and is NOT a target for optimization work.

**The speed optimization rule**: any change in this plan should reproduce the Current post-fix metrics above within ±1pp per metric, measured on Arizona MTL DGI. If a change moves a metric by more than 1pp, revert and investigate — that's a numerics change masquerading as a performance change.

**Verification command after any optimization:**
```bash
.venv_new/bin/python scripts/train.py --task mtl --state arizona --engine dgi
# Then compare the resulting summary/full_summary.json against the "Current post-fix" row above
```

### 1.2 Fixes that are already on `main` — DO NOT UNDO

Three fixes from commits `92e8bc6` and `2639596` are load-bearing. If you accidentally revert any of these while optimizing, the metrics will tank:

1. **`src/models/mtlnet.py:115-123`** — `category_poi` must be `CategoryHeadTransformer(num_tokens=2, token_dim=shared_layer_size//2, num_layers=num_layers, num_heads=num_heads, dropout=0.1, num_classes=num_classes)`. It was changed to `CategoryHeadEnsemble` at commit `4e31f3f` (Nov 11, 2025) and that caused a ~4pp regression — do not re-introduce.
2. **`src/models/mtlnet.py:206`** — the category head receives `shared_cat.squeeze(1)` because `CategoryHeadTransformer.forward` expects `[B, D]` not `[B, 1, D]`.
3. **`src/training/runners/mtl_cv.py:295-296`** — **both** criteria must be unweighted: `CrossEntropyLoss(reduction='mean')`. Do NOT pass `weight=alpha_cat` or `weight=alpha_next`. This was verified empirically to give +5-7pp accuracy vs weighted (under NashMTL gradient balancing, the class weights cause rapid minority-class overfitting). The `alpha_next` / `alpha_cat` variables can stay computed for logging, but must not be wired into the criterion.

See `/Users/vitor/.claude/projects/-Users-vitor-Desktop-mestrado-ingred/memory/mtl_category_loss_unweighted.md` for the full rationale.

### 1.3 Related note — NashMTL solver fix

A separate fix landed around the same time (see `memory/nash_mtl_solver_bug.md` — NashMTL had been silently collapsing to `[1,1]` weights because cvxpy couldn't find the ECOS solver, hidden by a bare `except` clause). `requirements.txt` now pins `ecos==2.0.14`. Do not remove that pin.

### 1.4 Hardware / environment

- Apple M2 Pro, 12 cores, 32 GB unified memory
- macOS 25.4.0
- PyTorch **2.9.1** (2.11.0 is latest; upgrade is a separate optional task in this plan)
- MPS backend, no CUDA
- Dataset sizes are tiny (Arizona: cat=20440×64≈5 MB, next=25083×9×64≈58 MB). Entire dataset fits in unified memory trivially.
- Current per-fold time: ~250-260s (500 optimizer steps @ ~0.5s/step). Most of that is per-step overhead, not compute.

### 1.5 Ground rules

- **No dependency upgrades** in the "safe batch" (items #1-#7 below). Torch upgrade and torch.compile are optional separate tasks.
- **No numerics changes**: no AMP/float16, no different reduction modes, no architecture tweaks.
- **Always run the full test suite** after each change: `.venv_new/bin/pytest tests/test_models/ tests/test_training/ tests/test_data/ -q`. Should be 354+ passing, 79 skipped, 0 failures.
- **Benchmark before and after** each change: time a single-fold Arizona run to see per-step and per-fold improvement. Use `time .venv_new/bin/python scripts/train.py --task mtl --state arizona --engine dgi` or measure fold duration in `folds/fold1_info.json`.
- **Commit each item as its own commit** with a message describing the expected speedup and what was measured.

---

## 2. Optimizations to implement (safe batch — items #1–#7)

Target combined speedup when all seven land: **~20-40% per fold** (Arizona: 22 min → ~14-17 min), with zero metric change.

### Item #1 — `num_workers=0` on MPS (5-10% speedup, 1 line)

**File:** `src/data/folds.py`
**Function:** `_get_num_workers` (around line 126-131)

**Current code:**
```python
def _get_num_workers() -> int:
    # On MPS (macOS), multiprocessing IPC overhead exceeds benefit for
    # in-memory tensor datasets. Keep workers low to avoid fork overhead.
    if DEVICE.type == 'mps':
        return min(2, os.cpu_count() or 1)
    return min(8, os.cpu_count() or 1)
```

**Change to:**
```python
def _get_num_workers() -> int:
    # MPS + in-memory tensor datasets: num_workers=0 is fastest.
    # Each worker is a forked Python process that pickles the tensor over
    # IPC per epoch — pure overhead when the dataset is already a torch
    # tensor in RAM. See PyTorch Lightning MPS docs.
    if DEVICE.type == 'mps':
        return 0
    return min(8, os.cpu_count() or 1)
```

**Why:** `POIDataset` holds preloaded torch tensors. Workers add no value (no image decoding, no augmentation). Each worker is ~50 MB of forked Python process. The existing `num_workers=2` creates 2 subprocesses per dataloader × (train + val) × (next + category) × 5 folds = 20 subprocess lifetimes per run, each doing nothing but ferrying already-in-memory slices.

**Downstream side effect:** `_create_dataloader` in the same file uses `num_workers=0` implicitly when this returns 0. Check that `persistent_workers=num_workers > 0` (line 250) and `prefetch_factor=2 if num_workers > 0 else None` (line 251) still work correctly — they're already written to handle the 0 case.

**Verification:**
```bash
.venv_new/bin/pytest tests/test_data/ -q
.venv_new/bin/python scripts/train.py --task mtl --state arizona --engine dgi
# Compare fold1_info.json duration vs baseline
```

### Item #2 — Cache `shared_parameters()` / `task_specific_parameters()` lists per fold (2-5% speedup)

**Files:**
- `src/training/runners/mtl_cv.py` (pass cached lists into `train_model`)
- No change needed in `src/models/mtlnet.py` — the generator methods stay.

**Current code (`mtl_cv.py:114-118` inside the per-batch loop):**
```python
loss, _ = mtl_criterion.backward(
    torch.stack([next_loss, category_loss]),
    shared_parameters=list(model.shared_parameters()),
    task_specific_parameters=list(model.task_specific_parameters()),
)
```

**Problem:** These generators walk `named_parameters()` and run string-matching filters (`"shared_layers" in name or "task_embedding" in name or "film" in name`) on every batch. 500 batches × 5 folds = 2500 redundant filterings per run.

**Fix:**

1. Cache the lists once per fold before calling `train_model`. In `train_with_cross_validation` (around line 269 where the model is created):

```python
model = create_model(config.model_name, **config.model_params).to(DEVICE)

# Cache parameter group lists — they don't change during training
cached_shared_params = list(model.shared_parameters())
cached_task_params = list(model.task_specific_parameters())
```

2. Add two new kwargs to `train_model` signature (around line 30-46):

```python
def train_model(model: torch.nn.Module,
                optimizer,
                scheduler,
                dataloader_next: TaskFoldData,
                dataloader_category: TaskFoldData,
                next_criterion,
                category_criterion,
                mtl_criterion,
                num_epochs,
                num_classes,
                shared_parameters: list = None,        # NEW
                task_specific_parameters: list = None,  # NEW
                fold_history=FoldHistory.standalone({'next', 'category'}),
                max_grad_norm: float = 1.0,
                timeout: Optional[int] = None,
                next_target_cutoff: Optional[float] = None,
                category_target_cutoff: Optional[float] = None,
                callbacks: Optional[list] = None,
                ):
```

3. Inside `train_model`, fall back to generator if not passed (keeps function standalone-usable):

```python
if shared_parameters is None:
    shared_parameters = list(model.shared_parameters())
if task_specific_parameters is None:
    task_specific_parameters = list(model.task_specific_parameters())
```

4. Update the `mtl_criterion.backward` call to use the cached lists:

```python
loss, _ = mtl_criterion.backward(
    torch.stack([next_loss, category_loss]),
    shared_parameters=shared_parameters,
    task_specific_parameters=task_specific_parameters,
)
```

5. In `train_with_cross_validation`, pass them through:

```python
train_model(
    model, optimizer, scheduler,
    dataloader_next, dataloader_category,
    next_criterion, category_criterion, mtl_criterion,
    config.epochs, num_classes,
    shared_parameters=cached_shared_params,
    task_specific_parameters=cached_task_params,
    fold_history=history.get_curr_fold(),
    ...
)
```

**Verification:** run the same tests as item #1 plus `tests/test_losses/test_nash.py` if it exists.

### Item #3 — Pre-transfer fold tensors to device once, index on-device (10-25% speedup — biggest single win)

**Files:**
- `src/data/folds.py` — add a new dataloader-like abstraction or per-fold device prefetch
- `src/training/runners/mtl_cv.py` — drop the `.to(DEVICE)` lines from the hot loop
- `src/training/runners/mtl_eval.py` — same for validation

**Approach A (simpler, recommended):** keep the DataLoader API but pre-move the underlying tensors.

In `src/data/folds.py`, `POIDataset.__init__`:

```python
class POIDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor, device: Optional[torch.device] = None):
        # Pre-move to target device if specified. This is safe for MPS
        # because the datasets are small (<100MB) and fit in unified memory.
        if device is not None and features.device != device:
            self.features = features.to(device)
            self.targets = targets.to(device)
        else:
            self.features = features if features.device.type == 'cpu' else features.cpu()
            self.targets = targets if targets.device.type == 'cpu' else targets.cpu()
```

Then in `_create_dataloader`, thread `device=DEVICE` through when `DEVICE.type == 'mps'`:

```python
def _create_dataloader(
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        use_weighted_sampling: bool,
        seed: int,
) -> DataLoader:
    num_workers = _get_num_workers()

    # Only pre-move to device when num_workers=0 (MPS path). With workers,
    # tensors must remain on CPU because forked processes cannot share GPU
    # memory safely.
    dataset_device = DEVICE if num_workers == 0 else None

    sampler = None
    if use_weighted_sampling:
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        sampler = _create_weighted_sampler(y_np, seed)
        shuffle = False

    return DataLoader(
        POIDataset(x, y, device=dataset_device),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        pin_memory_device=str(DEVICE) if hasattr(DEVICE, 'index') else None,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
```

Then in `src/training/runners/mtl_cv.py` (around line 97-102) — after item #1 is applied, `num_workers=0`, and after this item's changes, the tensors come out of the DataLoader already on device. Make the `.to(DEVICE)` calls cheap no-ops (or remove them, but keep them for safety when a CPU path is ever used):

```python
for data_next, data_category in progress.iter_epoch():
    x_next, y_next = data_next
    x_category, y_category = data_category
    # These become no-ops when item #3 is applied (tensors already on device),
    # but are kept for safety in case a CPU dataloader is ever used.
    if x_next.device != DEVICE:
        x_next = x_next.to(DEVICE, non_blocking=True)
        y_next = y_next.to(DEVICE, non_blocking=True)
        x_category = x_category.to(DEVICE, non_blocking=True)
        y_category = y_category.to(DEVICE, non_blocking=True)
    ...
```

Do the exact same pattern in `src/training/runners/mtl_eval.py` (around line 33-36).

**Why approach A not B:** A more aggressive approach (throw away DataLoader entirely, use `torch.randperm` on-device + tensor indexing) would give another 2-5% on top but breaks the `WeightedRandomSampler` branch and requires rewriting the batching loop. Not worth the risk for a performance-only PR.

**Caveat for test suite:** some tests construct `POIDataset` directly with CPU tensors. The `device=None` default preserves old behavior. Verify: `.venv_new/bin/pytest tests/test_data/test_folds.py -q` (or whatever the folds test file is).

### Item #4 — `PYTORCH_ENABLE_MPS_FALLBACK=1` (safety, no perf)

**File:** `src/configs/globals.py`

**Current content (around line 1-5):**
```python
import torch

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
```

**Add above the DEVICE line:**
```python
import os
import torch

# Enable CPU fallback for operations not yet implemented on MPS.
# Required for some PyTorch 2.9+ operators during training on Apple Silicon.
# Safe: a no-op on CUDA/CPU, silently delegates unsupported ops to CPU on MPS.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
```

**Why:** currently the code works fine on MPS but has no safety net if a torch upgrade (item #8) or new model variant introduces an op that MPS doesn't implement. Setting the env var at import time ensures it's always in effect, regardless of how the script is invoked. `setdefault` respects any user-set value.

### Item #5 — Remove dead `zero_grad` at `mtl_cv.py:92` (0 perf, cleanup)

**File:** `src/training/runners/mtl_cv.py`

**Current code (lines ~91-104):**
```python
    # Reset gradients at the beginning
    optimizer.zero_grad(set_to_none=True)

    # Iterate over batches with automatic progress tracking
    for data_next, data_category in progress.iter_epoch():
        # Move data to device
        x_next, y_next = data_next
        ...
        optimizer.zero_grad(set_to_none=True)
```

The first `zero_grad` (line 92) runs once per epoch, then line 104 immediately re-zeros on the first batch. Line 92 is dead.

**Change:** delete the line 92 `optimizer.zero_grad(set_to_none=True)` and its comment. Nothing else.

### Item #6 — NashMTL on-device norm (1-3% speedup during weight updates)

**File:** `src/losses/nash_mtl.py`

**Current code (around lines 224-228 inside `get_weighted_loss`):**
```python
self.normalization_factor = (
    torch.norm(GTG).detach().cpu().numpy().reshape((1,))
)
GTG = GTG / self.normalization_factor.item()
alpha = self.solve_optimization(GTG.cpu().detach().numpy())
```

This does 3 MPS→CPU transfers on every weight-update step. The solver (`cvxpy` with ECOS) runs on CPU and needs the numpy copy, so that sync is unavoidable. But the `torch.norm` computation and division can stay on-device.

**Change:**
```python
# Compute norm on-device, divide on-device, only sync once just before solver
norm = torch.norm(GTG).detach()
GTG_norm = GTG / norm

# Keep normalization_factor for logging (cpu scalar)
self.normalization_factor = norm.cpu().numpy().reshape((1,))

# Single sync for the solver input (unavoidable — cvxpy runs on CPU)
alpha = self.solve_optimization(GTG_norm.detach().cpu().numpy())
```

**Verification:** this is the most numerics-sensitive change in the safe batch. Run:
```bash
.venv_new/bin/pytest tests/test_losses/ -q
.venv_new/bin/python scripts/train.py --task mtl --state arizona --engine dgi
# Verify summary/full_summary.json matches baseline within ±0.5pp
```

The weighted-loss output should be bitwise identical (same math, just different device residency). If the metrics drift, something else is wrong — revert and investigate.

### Item #7 — Dedupe padding mask (1-2% speedup)

**Files:**
- `src/models/mtlnet.py` (compute mask once)
- `src/models/heads/next.py` (accept mask from outside)

**Current duplication:**
- `mtlnet.py:180`: `mask = (next_input.abs().sum(dim=-1) == pad_value)` → zero-fills pad tokens
- `next.py:112` (inside `NextHeadMTL.forward`): `padding_mask = (x.abs().sum(dim=-1) == 0)` → recomputes the same mask

Both are an `abs().sum(-1) == const` pass over a `[B, 9, 256]` tensor. Small but duplicated.

**Fix approach A (minimal change):** don't change the `NextHeadMTL.forward` signature; instead, pass the mask through `torch.Tensor._metadata`-style attribute. **Not recommended** — fragile.

**Fix approach B (cleaner):** add an optional `padding_mask` kwarg to `NextHeadMTL.forward`:

In `src/models/heads/next.py` `NextHeadMTL.forward` (around line 108):
```python
def forward(self, x, padding_mask=None):
    batch_size, seq_length, _ = x.size()
    device = x.device

    if padding_mask is None:
        padding_mask = (x.abs().sum(dim=-1) == 0)
    x = self.pe(x, padding_mask)
    ...
```

In `src/models/mtlnet.py` `MTLnet.forward` (around lines 180, 208):
```python
pad_value = InputsConfig.PAD_VALUE
next_mask = (next_input.abs().sum(dim=-1) == pad_value)  # (batch_size, seq_len)
next_input = next_input.masked_fill(next_mask.unsqueeze(-1), 0)
...
# After shared_layers processing, shared_next has the SAME padding pattern
# because zero inputs stay zero through linear+layernorm+leakyrelu. Reuse
# the mask directly:
out_next = self.next_poi(shared_next, padding_mask=next_mask)
```

**Careful caveat:** verify that `shared_next` (post shared_layers) still has `abs().sum(-1) == 0` on the same positions as `next_input`. It should, because:
- LayerNorm of zero vector is zero (when elementwise_affine=True with bias=0) — **actually it's the bias value, not zero!** LayerNorm's `bias` parameter is initialized to 0 but becomes learnable, so this assumption may break.
- Linear of zero vector → bias — again, not zero in general.

**Therefore approach B may be unsafe** unless you carefully audit the shared_layers path. If the mask assumption breaks, you'd silently corrupt the padding mask inside the transformer attention → different metric numbers.

**Recommendation:** either skip item #7 in the safe batch, OR take the narrower fix: cache the mask inside `MTLnet` during the same forward() call and pass *only* to `NextHeadMTL`, but reconstruct it *inside* `NextHeadMTL` if the input has been transformed. Actually the simplest safe fix is to compute the mask ONCE in `MTLnet.forward` using `next_input` (the original), then pass that same mask through because the padding positions are known ahead of time and don't depend on intermediate activations:

```python
# mtlnet.py forward
pad_value = InputsConfig.PAD_VALUE
next_mask = (next_input.abs().sum(dim=-1) == pad_value)
next_input = next_input.masked_fill(next_mask.unsqueeze(-1), 0)
...
# Pass the ORIGINAL padding mask derived from raw input, not recomputed
out_next = self.next_poi(shared_next, padding_mask=next_mask)
```

In `next.py` `NextHeadMTL.forward`:
```python
def forward(self, x, padding_mask=None):
    batch_size, seq_length, _ = x.size()
    device = x.device
    # Use provided mask if available, otherwise compute from input
    if padding_mask is None:
        padding_mask = (x.abs().sum(dim=-1) == 0)
    x = self.pe(x, padding_mask)
    ...
```

This is safe because the padding mask is a property of the original sequence positions, not of activations. `shared_next` at position `i` being non-zero after processing doesn't change whether position `i` was padding in the raw input.

**Verification is critical** — run the full regression test suite and compare `fold1_info.json` duration + metrics before/after. If metrics drift by more than floating-point noise, revert and investigate.

If you're at all unsure, **skip item #7**. The combined savings from #1-#6 alone should get the 20-30% speedup target.

---

## 3. Optional optimizations (separate commits, higher risk)

### Item #8 — Upgrade PyTorch 2.9.1 → 2.11.0 (5-10% speedup, medium risk)

**Risk:** RNG streams and MPS kernel numerics may shift slightly between minor versions, which could invalidate reproducibility against the Nov 5 baseline.

**Procedure:**
1. Create a worktree: `git worktree add .claude/worktrees/torch-upgrade -b perf/torch-upgrade`
2. In the worktree: `.venv_new/bin/pip install -U torch torch-geometric`
3. Run the full test suite. Fix any breakages that come from API changes.
4. Run Arizona MTL DGI and compare against baseline. The F1/accuracy should still be within ±1pp (noise from different kernel numerics is expected).
5. If metrics are stable, update `requirements.txt` with the new pin and commit.
6. If metrics drift significantly, **do not merge** — keep investigating which op is the culprit (likely a transformer attention kernel change).

### Item #9 — `torch.compile(model, mode='reduce-overhead')` (15-40% if it works, medium risk)

**Risk:** compile may fail or fall back silently on some ops; adds one-time warmup cost; may not give expected speedup on MPS (mature on CUDA, newer on MPS).

**Procedure:**

In `src/training/runners/mtl_cv.py` around line 269 where the model is created:
```python
model = create_model(config.model_name, **config.model_params).to(DEVICE)

# Optional: compile for per-step overhead reduction
if os.environ.get('TORCH_COMPILE', '0') == '1':
    try:
        model = torch.compile(model, mode='reduce-overhead', dynamic=False)
        logger.info("Model compiled with torch.compile(mode='reduce-overhead')")
    except Exception as e:
        logger.warning(f"torch.compile failed, falling back: {e}")
```

**Validation:**
```bash
TORCH_COMPILE=1 .venv_new/bin/python scripts/train.py --task mtl --state arizona --engine dgi
```

Watch the first epoch — it will be slow due to compile warmup (~30s). Subsequent epochs should be noticeably faster.

**Caveat:** `torch.compile` may not be compatible with the `mtl_criterion.backward()` manual-grad-manipulation pattern used by NashMTL. If you see errors about graph breaks during `autograd.grad`, you may need `fullgraph=False` or disable compile for this model.

### Item #10 — Replace `sklearn.classification_report` with `torchmetrics` (2-5% speedup)

**Risk:** different numerical behavior on edge cases (empty classes, NaN handling). Metric values should be the same but tested, not assumed.

**Procedure:** install `torchmetrics` (probably already installed via torch-geometric), replace the `classification_report` calls in `mtl_cv.py` and `mtl_eval.py` with on-device `MulticlassF1Score(num_classes=7, average='macro')` and `MulticlassAccuracy(num_classes=7, average='micro')`.

Keep the existing sklearn call for the final per-class JSON report (`validation_best_model` path) — that's only called once per fold and produces the per-class breakdown.

---

## 4. Execution plan for the implementing session

### Phase A — Setup (5 min)

1. `cd /Users/vitor/Desktop/mestrado/ingred`
2. `git status` — verify clean working tree. If there are uncommitted changes, ask the user what to do.
3. `git log --oneline -5` — confirm you see `2639596 fix(mtl): drop weight=alpha_next from next-POI criterion too` and `92e8bc6 fix(mtl): restore Nov 5 baseline...` at the top. If not, you're on the wrong commit/branch — stop and ask.
4. Read `/Users/vitor/.claude/projects/-Users-vitor-Desktop-mestrado-ingred/memory/MEMORY.md` and the linked memory files, especially `mtl_category_loss_unweighted.md` and `nash_mtl_solver_bug.md`. These explain the three load-bearing fixes you must not regress.

### Phase B — Baseline measurement (10 min)

1. Run `.venv_new/bin/pytest tests/test_models/ tests/test_training/ tests/test_data/ tests/test_losses/ -q`. Record the pass count.
2. Run `time .venv_new/bin/python scripts/train.py --task mtl --state arizona --engine dgi > /tmp/baseline_arizona.log 2>&1`. Record total wall time.
3. Read `results/dgi/arizona/mtlnet_lr1.0e-04_bs2048_ep50_<timestamp>/summary/full_summary.json`. Record Cat F1, Cat Acc, Next F1, Next Acc.
4. Read each fold's `fold*_info.json` `duration` field. Record per-fold durations.

### Phase C — Implement items #1-#6 in separate commits

Implement each item in order, one commit per item. After each:

1. Run `.venv_new/bin/pytest tests/test_models/ tests/test_training/ tests/test_data/ tests/test_losses/ -q` — must still pass.
2. For items #3 and #6 specifically (potential numerics sensitivity), run the full Arizona MTL and verify metrics match baseline within ±0.5pp.
3. Commit with message format:
   ```
   perf(mtl): <item description>

   <what was changed and why>
   Expected speedup: X-Y% on MPS.
   Verified: tests pass, Arizona metrics unchanged (Cat F1=<value> vs baseline <value>).

   Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
   ```

### Phase D — Evaluate item #7 (padding mask dedup)

This is the highest-risk item in the safe batch. Read the caveat in item #7 carefully. If you're not confident the mask-reuse is safe, **skip it** and document why in the commit message of the batch summary.

### Phase E — Full validation (15 min)

1. Run the **full** test suite: `.venv_new/bin/pytest -q`. Verify 354+ passed, 0 failed. (After the PR #7 NashMTL additions, expect ≥ 363 passed.)
2. Run both Alabama and Arizona MTL DGI end-to-end. Compare each state's summary against the "Current post-fix" row in Section 1.1 (NOT the Nov 5 baselines — the current code intentionally performs differently on Next Acc because of the unweighted-CE fix). Minimum acceptable thresholds (±1pp band around the post-fix numbers):
   - Alabama: Cat F1 ≥ 0.442, Cat Acc ≥ 53.1%, Next F1 ≥ 0.255, Next Acc ≥ 35.8%
   - Arizona: Cat F1 ≥ 0.443, Cat Acc ≥ 52.3%, Next F1 ≥ 0.263, Next Acc ≥ 36.9%
3. Record per-fold durations for both. Total should be noticeably below the 22-29 min post-fix timing.

### Phase F — Report to user

Summarize in a single message:
- What was implemented (list of commits)
- Measured speedup (before → after wall time, per-fold mean)
- Metric deltas (should be within noise for every metric)
- Any items skipped and why
- Recommendation on whether to proceed with items #8, #9, #10

---

## 5. Reference — file paths cheat sheet

| File | Purpose | Key lines |
|---|---|---|
| `src/training/runners/mtl_cv.py` | MTL training runner | Items #2, #3, #5 |
| `src/training/runners/mtl_eval.py` | Validation loop | Item #3 (same pattern) |
| `src/data/folds.py` | Fold creation & DataLoader | Items #1, #3 |
| `src/models/mtlnet.py` | MTLnet model forward | Item #7 |
| `src/models/heads/next.py` | NextHeadMTL forward | Item #7 |
| `src/losses/nash_mtl.py` | NashMTL loss | Item #6 |
| `src/configs/globals.py` | Device + env setup | Item #4 |
| `tests/test_models/` | Model-level tests | Verification |
| `tests/test_training/` | Training runner tests | Verification |
| `tests/test_data/` | Fold creation tests | Verification |
| `tests/test_losses/` | Loss tests | Verification (item #6) |
| `scripts/train.py` | CLI entrypoint | `--task mtl --state <state> --engine dgi` |
| `results/dgi/<state>/mtlnet_*/summary/full_summary.json` | Run metrics | Baseline comparison |
| `results/dgi/<state>/mtlnet_*/folds/foldN_info.json` | Per-fold duration + best_epochs | Speed measurement |
| `results/baselines/dgi/<state>/` | Historical Nov 5 baseline | Don't modify |

## 6. Sources consulted (for context)

These informed the analysis; you don't need to re-read them unless you hit an unexpected MPS issue:

- [Apple Developer — Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
- [PyTorch `torch.mps` documentation](https://docs.pytorch.org/docs/stable/mps.html)
- [HuggingFace — PyTorch training on Apple silicon](https://huggingface.co/docs/transformers/en/perf_train_special)
- [HuggingFace Accelerate — MPS](https://huggingface.co/docs/accelerate/en/usage_guides/mps)
- [PyTorch Lightning — MPS training basics](https://lightning.ai/docs/pytorch/stable/accelerators/mps_basic.html)
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)
- [Apple Silicon PyTorch MPS: Setup and Speed Expectations (tillcode)](https://tillcode.com/apple-silicon-pytorch-mps-setup-and-speed-expectations/)

---

## 7. Final note to the implementing session

If anything in this plan conflicts with the current state of the code — for instance, the line numbers drift because other commits landed in the meantime, or a test fails in an unexpected way — **STOP and tell the user** rather than trying to force the change through. The metric baselines in Section 1.1 are the source of truth; anything that violates them is a regression even if the tests pass.

The whole point of this plan is speed-only improvements with zero quality change. If at any point during implementation you find that a "safe" change actually moves a metric, revert it, and either document why it's actually a quality change (and defer it) or find a different approach.

Good luck.
