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
