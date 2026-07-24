"""Tests for autotune.lsf.worker_entry GPU-count guard.

The guard runs before ``ray start`` and fails fast when a host can't back the
requested ``--num-gpus``. Without it, Ray over-declares the GPU count and the
mismatch only surfaces much later as an opaque
``device >= 0 && device < num_gpus INTERNAL ASSERT FAILED`` in
``torch.cuda.set_device(local_rank)`` during NCCL backend setup.
"""

import pytest

from autotune.lsf import worker_entry


class TestVisibleGpuCount:
    def test_counts_cuda_visible_devices(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6")
        assert worker_entry._visible_gpu_count() == 7

    def test_empty_cuda_visible_devices_is_zero(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert worker_entry._visible_gpu_count() == 0

    def test_ignores_trailing_and_blank_entries(self, monkeypatch):
        # A trailing comma / stray spaces must not inflate the count.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0, 1 ,2,")
        assert worker_entry._visible_gpu_count() == 3

    def test_falls_back_to_torch_when_no_mask(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        import sys
        import types

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 8,
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert worker_entry._visible_gpu_count() == 8


class TestAssertGpuCount:
    def test_raises_when_fewer_visible_than_requested(self, monkeypatch):
        monkeypatch.setattr(worker_entry, "_visible_gpu_count", lambda: 7)
        with pytest.raises(RuntimeError, match="exposes 7 usable GPU"):
            worker_entry._assert_gpu_count(8)

    def test_passes_when_equal(self, monkeypatch):
        monkeypatch.setattr(worker_entry, "_visible_gpu_count", lambda: 8)
        worker_entry._assert_gpu_count(8)  # no raise

    def test_passes_when_more_visible_than_requested(self, monkeypatch):
        # Over-provisioning is harmless — only under-provisioning crashes later.
        monkeypatch.setattr(worker_entry, "_visible_gpu_count", lambda: 8)
        worker_entry._assert_gpu_count(4)  # no raise

    def test_proceeds_when_probe_unknown(self, monkeypatch):
        # An unknown count (None) must warn-and-proceed, not block the launch.
        monkeypatch.setattr(worker_entry, "_visible_gpu_count", lambda: None)
        worker_entry._assert_gpu_count(8)  # no raise
