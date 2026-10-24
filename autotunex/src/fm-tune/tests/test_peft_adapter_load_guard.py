"""The PEFT adapter-load CPU guard must wrap all of ``train()``, not just resume.

Under ``-gpu mode=exclusive_process``, PEFT resolves its load device via
``infer_device()``, which returns a bare ``"cuda"`` (cuda:0) on every rank. Two
distinct calls inside ``Trainer.train()`` hit that path:

1. ``_load_from_checkpoint()`` — resume, at the start.
2. ``_load_best_model()`` — at the end, whenever ``load_best_model_at_end=True``.

The guard used to be gated on a truthy ``resume_arg``, which covered (1) only. Once
final training started setting ``load_best_model_at_end=True``, (2) crashed ranks
1..N-1 with "CUDA-capable device(s) is/are busy or unavailable" — at the very end
of an otherwise successful run. These tests pin the unconditional scoping so the
gating cannot come back.
"""

import re

import pytest

from autotune.trainers._resume import peft_adapter_load_on_cpu

# Every multi-GPU driver that calls trainer.train() on a possibly-PEFT model.
# Single-GPU drivers are exempt: one process, one GPU, no cross-rank contention.
MULTI_GPU_DRIVERS = [
    "autotune/trainers/driver_multi_hf_fsdp.py",
    "autotune/trainers/driver_multi_hf_ds.py",
    "autotune/trainers/driver_multi_trl_fsdp.py",
    "autotune/trainers/driver_multi_trl_ds.py",
]


class TestGuardPatchesInferDevice:
    def test_patches_every_bound_reference(self):
        # infer_device is imported BY NAME into several PEFT modules, so each holds
        # its own binding and all must be patched.
        peft_model = pytest.importorskip("peft.peft_model")
        save_and_load = pytest.importorskip("peft.utils.save_and_load")
        mods = [m for m in (peft_model, save_and_load) if hasattr(m, "infer_device")]
        assert mods, "expected at least one PEFT module exposing infer_device"
        with peft_adapter_load_on_cpu():
            for mod in mods:
                assert mod.infer_device() == "cpu"

    def test_restores_originals(self):
        peft_model = pytest.importorskip("peft.peft_model")
        original = peft_model.infer_device
        with peft_adapter_load_on_cpu():
            assert peft_model.infer_device is not original
        assert peft_model.infer_device is original

    def test_restores_on_exception(self):
        peft_model = pytest.importorskip("peft.peft_model")
        original = peft_model.infer_device
        with pytest.raises(RuntimeError):
            with peft_adapter_load_on_cpu():
                raise RuntimeError("boom")
        assert peft_model.infer_device is original

    def test_is_reentrant_safe(self):
        peft_model = pytest.importorskip("peft.peft_model")
        original = peft_model.infer_device
        with peft_adapter_load_on_cpu():
            with peft_adapter_load_on_cpu():
                assert peft_model.infer_device() == "cpu"
            assert peft_model.infer_device() == "cpu"
        assert peft_model.infer_device is original


class TestDriversScopeGuardUnconditionally:
    """Regression: the guard must not be gated on resume_arg."""

    @pytest.mark.parametrize("path", MULTI_GPU_DRIVERS)
    def test_no_resume_gated_guard(self, path):
        src = open(path).read()
        # The old shape: `if resume_arg:` immediately followed by the guard.
        assert not re.search(
            r"if\s+resume_arg\s*:\s*\n\s*with\s+peft_adapter_load_on_cpu\(\)",
            src,
        ), f"{path}: guard is gated on resume_arg; _load_best_model() would be unprotected"

    @pytest.mark.parametrize("path", MULTI_GPU_DRIVERS)
    def test_every_train_call_is_guarded(self, path):
        src = open(path).read()
        train_calls = re.findall(r"^\s*(?:\w+\s*=\s*)?trainer\.train\(", src, re.M)
        assert len(train_calls) == 1, (
            f"{path}: expected exactly one trainer.train() call, found {len(train_calls)}; "
            "a second unguarded call would reintroduce the crash"
        )
        # That call must sit inside the guard's with-block.
        assert re.search(
            r"with\s+peft_adapter_load_on_cpu\(\)\s*:\s*\n\s*(?:\w+\s*=\s*)?trainer\.train\(",
            src,
        ), f"{path}: trainer.train() is not wrapped in peft_adapter_load_on_cpu()"

    @pytest.mark.parametrize("path", MULTI_GPU_DRIVERS)
    def test_guard_is_imported(self, path):
        assert "from autotune.trainers._resume import peft_adapter_load_on_cpu" in open(path).read()
