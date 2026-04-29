"""Regression tests for src/training/helpers.py.

Coverage focus: ``compute_class_weights`` must accept tensors that live on
ANY device (CPU, MPS, CUDA), not just CPU numpy arrays. The original
signature only accepted numpy and crashed with ``TypeError: can't convert
mps:0 device type tensor to numpy`` when callers passed device-resident
target tensors fed by PR #8's on-device-tensors optimization. Centralizing
the .cpu() boundary inside this helper means future call sites cannot
silently re-introduce the bug.
"""
import numpy as np
import pytest
import torch

from training.helpers import compute_class_weights


CPU = torch.device("cpu")


class TestComputeClassWeightsTensorInputs:
    """Pin the on-device tensor input contract."""

    def test_numpy_input_still_works(self):
        # Backward-compat: the original numpy.ndarray path must still work
        # so existing call sites that pre-extract a numpy array don't break.
        targets = np.array([0, 1, 2, 0, 1, 0], dtype=np.int64)
        w = compute_class_weights(targets, num_classes=3, device=CPU)
        assert isinstance(w, torch.Tensor)
        assert w.dtype == torch.float32
        assert w.shape == (3,)
        assert w.device == CPU

    def test_cpu_tensor_input(self):
        # CPU tensors must work directly — no manual .numpy() at call sites.
        targets = torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.long)
        w = compute_class_weights(targets, num_classes=3, device=CPU)
        assert w.shape == (3,)
        assert w.device == CPU

    def test_cpu_tensor_and_numpy_produce_same_weights(self):
        # The tensor path must match the numpy path bit-for-bit (modulo
        # dtype) so the helper is a true generalization, not a divergence.
        targets_list = [0, 1, 2, 0, 1, 0, 2, 2, 1, 0]
        w_np = compute_class_weights(np.array(targets_list, dtype=np.int64), 3, CPU)
        w_t = compute_class_weights(torch.tensor(targets_list, dtype=torch.long), 3, CPU)
        torch.testing.assert_close(w_np, w_t)

    def test_mps_tensor_input(self):
        # The headline regression case: targets pre-loaded onto MPS by
        # the on-device-tensors optimization must NOT raise
        # "can't convert mps:0 device type tensor to numpy".
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this host")
        targets_cpu = torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.long)
        targets_mps = targets_cpu.to("mps")
        try:
            w = compute_class_weights(targets_mps, num_classes=3, device=CPU)
        except TypeError as exc:
            pytest.fail(
                f"compute_class_weights must accept MPS-resident tensors, "
                f"got TypeError: {exc}"
            )
        # Output device is what the caller asked for, not the input device.
        assert w.device == CPU
        # And it must match the CPU-tensor path numerically.
        w_cpu = compute_class_weights(targets_cpu, num_classes=3, device=CPU)
        torch.testing.assert_close(w, w_cpu)

    def test_mps_tensor_with_grad_does_not_raise(self):
        # Defensive: even if a future call site passes a tensor with
        # requires_grad=True, the helper must not raise (it must detach
        # internally to call .numpy()).
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this host")
        # We use float here only because requires_grad needs a non-int dtype;
        # then we round to long via .long() before passing. The point is
        # the helper detaches whatever it gets.
        targets = torch.tensor(
            [0.0, 1.0, 2.0, 0.0], requires_grad=True
        ).to("mps").long()
        # Note: .long() drops requires_grad anyway, so this is mostly a
        # smoke check that the .detach() in the helper is harmless.
        compute_class_weights(targets, num_classes=3, device=CPU)


# ---------------------------------------------------------------------------
# B9 — alpha exempt from weight decay
# ---------------------------------------------------------------------------

class _MTLnetWithAlpha(torch.nn.Module):
    """Minimal MTL stub exposing the API setup_per_head_optimizer needs.

    cat_specific / reg_specific / shared_parameters() + a `next_poi.alpha`
    Parameter — mirrors the real mtlnet_crossattn surface that B9 reads.
    """

    def __init__(self):
        super().__init__()
        self.cat_part = torch.nn.Linear(4, 4)
        self.next_encoder = torch.nn.Linear(4, 4)
        self.shared_part = torch.nn.Linear(4, 4)

        # next_poi has an alpha Parameter (mirrors next_getnext_hard)
        self.next_poi = torch.nn.Linear(4, 4)
        self.next_poi.alpha = torch.nn.Parameter(torch.tensor(0.5))

    def cat_specific_parameters(self):
        return list(self.cat_part.parameters())

    def reg_specific_parameters(self):
        return list(self.next_encoder.parameters()) + list(self.next_poi.parameters())

    def shared_parameters(self):
        return list(self.shared_part.parameters())


class TestB9AlphaNoWeightDecay:
    """B9 hypothesis test: AdamW WD applied to single-scalar alpha shrinks
    it toward 0 every step and fights gradient-driven growth. Putting alpha
    in its own zero-WD group should preserve growth."""

    def test_alpha_kept_in_reg_group_when_flag_off(self):
        """Default (legacy): alpha is in the reg group with WD applied."""
        from training.helpers import setup_per_head_optimizer

        model = _MTLnetWithAlpha()
        opt = setup_per_head_optimizer(
            model, cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3,
            weight_decay=0.05, alpha_no_weight_decay=False,
        )
        names = [g.get("name") for g in opt.param_groups]
        assert "alpha_no_wd" not in names
        # alpha should be in the reg group (no split, no _head/_encoder split)
        reg_group = next(g for g in opt.param_groups if g.get("name") == "reg")
        alpha_id = id(model.next_poi.alpha)
        assert alpha_id in {id(p) for p in reg_group["params"]}

    def test_alpha_isolated_with_zero_wd_when_flag_on(self):
        from training.helpers import setup_per_head_optimizer

        model = _MTLnetWithAlpha()
        opt = setup_per_head_optimizer(
            model, cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3,
            weight_decay=0.05, alpha_no_weight_decay=True,
        )
        names = [g.get("name") for g in opt.param_groups]
        assert "alpha_no_wd" in names

        alpha_group = next(g for g in opt.param_groups if g.get("name") == "alpha_no_wd")
        assert alpha_group["weight_decay"] == 0.0
        # alpha is exactly the only thing in this group
        assert len(alpha_group["params"]) == 1
        assert id(alpha_group["params"][0]) == id(model.next_poi.alpha)

        # And it is NOT in the reg group anymore
        reg_group = next(g for g in opt.param_groups if g.get("name") == "reg")
        assert id(model.next_poi.alpha) not in {id(p) for p in reg_group["params"]}

    def test_alpha_isolated_under_per_encoder_lr_split(self):
        """B9 should compose with D6 (reg_head_lr split) — alpha still
        peels out, even when reg is split into reg_encoder + reg_head."""
        from training.helpers import setup_per_head_optimizer

        model = _MTLnetWithAlpha()
        opt = setup_per_head_optimizer(
            model, cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3,
            weight_decay=0.05,
            reg_head_lr=3e-2,  # triggers the split path
            alpha_no_weight_decay=True,
        )
        names = [g.get("name") for g in opt.param_groups]
        assert names == ["cat", "reg_encoder", "reg_head", "shared", "alpha_no_wd"]

        alpha_group = next(g for g in opt.param_groups if g.get("name") == "alpha_no_wd")
        assert alpha_group["weight_decay"] == 0.0
        # Alpha gets reg_head_lr (3e-2) since it lives in next_poi
        assert alpha_group["lr"] == pytest.approx(3e-2)
        assert id(alpha_group["params"][0]) == id(model.next_poi.alpha)

        reg_head_group = next(g for g in opt.param_groups if g.get("name") == "reg_head")
        assert id(model.next_poi.alpha) not in {id(p) for p in reg_head_group["params"]}
