# MTLnetCrossAttn Missing `cat_forward` / `next_forward` Overrides — Audit

**Severity:** MEDIUM — blocks the evaluation / ablation path (`scripts/evaluate.py` + partial-forward tests) for any checkpoint trained with `mtlnet_crossattn`. Does **not** affect training or reported training metrics; training-loop forward uses `MTLnetCrossAttn.forward`, which is correctly overridden.

**Detected:** 2026-04-22 during the pre-P5 model/optimizer critical review.

**Status:** OPEN — 5-minute fix, no re-runs needed.

---

## TL;DR

`MTLnetCrossAttn` overrides `_build_shared_backbone` (to install `crossattn_blocks` + final LNs instead of FiLM + shared residual stack) and `forward` (to route through cross-attention) — but it does **not** override `cat_forward` / `next_forward`. The inherited base-class implementations (src/models/mtl/mtlnet/model.py:479-514) reference `self.film` and `self.shared_layers`, neither of which exist on the cross-attn subclass. Calling either method raises `AttributeError: 'MTLnetCrossAttn' object has no attribute 'film'`.

Every other MTL variant (`MTLnetCGC`, `MTLnetMMoE`, `MTLnetDSelectK`, `MTLnetPLE`) overrides both. Cross-attn is the only omission.

---

## Evidence

### 1. Partial-forward methods are actually called

| Caller | Purpose |
|---|---|
| `scripts/evaluate.py:170,174` | Single-head evaluation of a checkpoint via `forward_fn=lambda m, x: m.cat_forward(x)` / `m.next_forward(x)` |
| `tests/test_models/test_mtlnet.py:88,96,112,126,275,297,298` | Bit-exact parity + isolated-head tests |

If anyone runs `scripts/evaluate.py --checkpoint <path_to_crossattn_model>` today, it crashes the moment the evaluator tries the single-head path.

### 2. Missing attributes

The base-class `cat_forward` (src/models/mtl/mtlnet/model.py:489-501):
```python
def cat_forward(self, category_input: torch.Tensor) -> torch.Tensor:
    ...
    enc = self.category_encoder(category_input)
    task_id = torch.zeros(enc.size(0), dtype=torch.long, device=enc.device)
    task_emb = self.task_embedding(task_id)      # <-- does not exist on crossattn
    modulated = self.film(enc, task_emb)          # <-- does not exist on crossattn
    shared = self.shared_layers(modulated)        # <-- does not exist on crossattn
    ...
```
`MTLnetCrossAttn._build_shared_backbone` (src/models/mtl/mtlnet_crossattn/model.py:165-187) installs only `crossattn_blocks`, `cat_final_ln`, `next_final_ln`. The `task_embedding` / `film` / `shared_layers` attributes never get created because the subclass overrides the exact method that would have registered them.

---

## Impact

- **Training / reported checkpoints: not affected.** `.forward(inputs)` is correctly overridden → all training runs land their gradients on the right params.
- **Evaluation of trained cross-attn checkpoints via `scripts/evaluate.py`: broken.** If the paper's final evaluation walks through this script (it should, for the per-head reports), it will crash on cross-attn rows.
- **Isolated-head ablations: broken.** Any ablation that disables one head's input by using `cat_forward` / `next_forward` (instead of feeding a zeros tensor to the other stream in `.forward`) cannot run on cross-attn.

No prior evaluation result is contaminated, because the broken methods raise `AttributeError` loudly rather than silently returning junk.

---

## Fix

Add both overrides on `MTLnetCrossAttn`. The cross-attn block needs a B-stream input — the natural design is to run the other stream with a zeros placeholder and an all-True pad mask, so the cross-attention gives the unused stream zero weight:

```python
def cat_forward(self, category_input: torch.Tensor) -> torch.Tensor:
    pad_value = InputsConfig.PAD_VALUE
    mask_a = None
    if self._task_a_is_sequential:
        mask_a = (category_input.abs().sum(dim=-1) == pad_value)
        category_input = category_input.masked_fill(mask_a.unsqueeze(-1), 0)

    enc_cat = self.category_encoder(category_input)
    # Minimal-viable B stream: zeros, fully padded. Attention returns
    # 0 contribution from B, so A is effectively self-attended via the
    # block's FFN path only. This is semantically a "partial forward"
    # approximation; for bit-exact parity with forward(), call
    # forward((cat, real_b)) and index [0].
    enc_next = torch.zeros(
        enc_cat.size(0), enc_cat.size(-2) if enc_cat.dim() == 3 else 1,
        enc_cat.size(-1), device=enc_cat.device, dtype=enc_cat.dtype,
    )
    b_pad_mask = torch.ones(
        enc_next.size(0), enc_next.size(1),
        device=enc_next.device, dtype=torch.bool,
    )

    a, b = enc_cat, enc_next
    for block in self.crossattn_blocks:
        a, b = block(a, b, a_pad_mask=mask_a, b_pad_mask=b_pad_mask)
    shared_cat = self.cat_final_ln(a)

    if self._task_set is not LEGACY_CATEGORY_NEXT and mask_a is not None:
        shared_cat = shared_cat.masked_fill(mask_a.unsqueeze(-1), 0)

    if self._task_a_is_sequential:
        return self.category_poi(shared_cat)
    return self.category_poi(shared_cat.squeeze(1)).view(
        -1, self.num_classes_task_a
    )
```
Mirror implementation for `next_forward`.

**Caveat on partial-forward semantics:** because cross-attention lets the two streams exchange content, a "cat-only forward" on a cross-attn model is not equivalent to running just the cat stream — the B stream's contribution is part of A's representation. The zeros-placeholder trick gets you a stable, deterministic, interpretable approximation (≡ "A with no info from B"); it is **not** bit-exact with `forward((cat, real_b))[0]`. Document this in the docstring of the overrides, same as the base class's `cat_forward` already documents its eval-only bit-exactness caveat.

Alternative (stricter): make the methods `raise NotImplementedError("CrossAttn has no partial-forward semantics; call forward((cat, next))[0] instead.")` and update `scripts/evaluate.py` to detect the model class and route through `.forward(...)[0]` / `.forward(...)[1]` for cross-attn. This is more defensible scientifically (no zeros-placeholder approximation) but requires touching the evaluator.

---

## Verification

Add to `tests/test_models/test_mtlnet.py` (or a new `test_mtlnet_crossattn.py`):
- Instantiate `MTLnetCrossAttn`, call `cat_forward(...)` and `next_forward(...)`. Assert output shapes.
- Assert that `forward((cat, next))[0]` equals `cat_forward(cat)` up to the documented partial-forward caveat — or, if the NotImplementedError route is taken, assert the error is raised.

---

## References

- Detected in the same review that produced `MTL_PARAM_PARTITION_BUG.md` (2026-04-22).
- `src/models/mtl/mtlnet_crossattn/model.py` — file needing the two new methods.
- `src/models/mtl/mtlnet/model.py:479-514` — the broken inherited implementation.
- `scripts/evaluate.py:170,174` — main caller that will currently crash on cross-attn checkpoints.
