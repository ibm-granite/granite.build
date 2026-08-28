"""Unit tests for autotune.device — the platform-decision module.

All tests monkeypatch torch so they pass on CI with no GPU and on a Mac.
device.py imports torch lazily inside functions, so patching torch.cuda /
torch.backends.mps attributes here takes effect.
"""

import os
from copy import deepcopy

import pytest
import torch

from autotune.device import (
    Accelerator,
    apply_platform_guards,
    configure_runtime_env,
    detect_accelerator,
    model_load_kwargs,
    object_store_bytes,
    ray_num_gpus,
    resolve_attn_implementation,
    resolve_precision,
)


@pytest.fixture(autouse=True)
def _clear_device_env(monkeypatch):
    monkeypatch.delenv("FMTUNE_DEVICE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot and restore the full environment around every test.

    This prevents environment variables set by configure_runtime_env from
    leaking into other tests when monkeypatch doesn't track them (happens when
    delenv is a no-op for a key that didn't exist pre-test).
    """
    orig = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(orig)


class TestDetectAccelerator:
    def test_cuda_detected_with_device_count(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        accel = detect_accelerator()
        assert isinstance(accel, Accelerator)
        assert accel.kind == "cuda"
        assert accel.count == 4
        assert accel.supports_distributed is True
        assert accel.supports_4bit is True
        assert accel.supports_flash_attn is True

    def test_cuda_count_honours_visible_devices(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        assert detect_accelerator().count == 2

    def test_falls_back_to_cpu_when_nothing_available(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        accel = detect_accelerator()
        assert accel.kind == "cpu"
        assert accel.count == 0
        assert accel.supports_distributed is False
        assert accel.supports_bf16 is False

    def test_override_mps(self, monkeypatch):
        import torch

        monkeypatch.setenv("FMTUNE_DEVICE", "mps")
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", lambda *a: True)
        accel = detect_accelerator()
        assert accel.kind == "mps"
        assert accel.count == 1
        assert accel.supports_distributed is False
        assert accel.supports_4bit is False
        assert accel.supports_flash_attn is False
        assert accel.supports_bf16 is True

    def test_override_cpu_wins_over_cuda(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setenv("FMTUNE_DEVICE", "cpu")
        assert detect_accelerator().kind == "cpu"

    def test_invalid_override_raises(self, monkeypatch):
        monkeypatch.setenv("FMTUNE_DEVICE", "tpu")
        with pytest.raises(ValueError, match="FMTUNE_DEVICE"):
            detect_accelerator()


def _accel(kind, bf16):
    return Accelerator(kind, 1, kind == "cuda", kind == "cuda", kind == "cuda", bf16)


class TestResolvePrecision:
    def test_cuda_bf16_stays_bf16(self):
        assert resolve_precision("bf16", _accel("cuda", True)) == "bf16"

    def test_cuda_fp32_stays_fp32(self):
        assert resolve_precision("fp32", _accel("cuda", True)) == "fp32"

    def test_mps_bf16_when_supported_and_probe_ok(self, monkeypatch):
        monkeypatch.setattr("autotune.device.supports_autocast_bf16", lambda: True)
        assert resolve_precision("bf16", _accel("mps", True)) == "bf16"

    def test_mps_bf16_downgrades_when_macos_too_old(self):
        # supports_bf16 False (macOS < 14) → fp32 regardless of probe
        assert resolve_precision("bf16", _accel("mps", False)) == "fp32"

    def test_mps_bf16_downgrades_when_probe_fails(self, monkeypatch):
        monkeypatch.setattr("autotune.device.supports_autocast_bf16", lambda: False)
        assert resolve_precision("bf16", _accel("mps", True)) == "fp32"

    def test_mps_probe_skipped_when_disabled(self, monkeypatch):
        # probe_autocast=False must NOT call the probe (guard path in main.py)
        def _boom():
            raise AssertionError("probe must not run when probe_autocast=False")

        monkeypatch.setattr("autotune.device.supports_autocast_bf16", _boom)
        assert resolve_precision("bf16", _accel("mps", True), probe_autocast=False) == "bf16"

    def test_mps_explicit_fp32_honoured(self):
        assert resolve_precision("fp32", _accel("mps", True)) == "fp32"

    def test_cpu_always_fp32(self):
        assert resolve_precision("bf16", _accel("cpu", False)) == "fp32"

    def test_invalid_precision_raises(self):
        with pytest.raises(ValueError, match="precision"):
            resolve_precision("int4", _accel("cuda", True))


class TestResolveAttnImplementation:
    def test_cuda_preserves_flash_attention_2(self):
        assert resolve_attn_implementation("flash_attention_2", _accel("cuda", True)) == "flash_attention_2"

    def test_mps_downgrades_flash_attention_2_to_eager(self):
        assert resolve_attn_implementation("flash_attention_2", _accel("mps", True)) == "eager"

    def test_mps_preserves_eager(self):
        assert resolve_attn_implementation("eager", _accel("mps", True)) == "eager"

    def test_mps_preserves_sdpa(self):
        assert resolve_attn_implementation("sdpa", _accel("mps", True)) == "sdpa"

    def test_cpu_downgrades_flash(self):
        assert resolve_attn_implementation("flash_attention_2", _accel("cpu", False)) == "eager"


class TestModelLoadKwargs:
    def test_cuda_matches_legacy_literal(self):
        # Regression lock: must equal the dict driver_single.py used pre-MPS.
        assert model_load_kwargs(_accel("cuda", True), "bf16") == {
            "device_map": "auto",
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }

    def test_mps_device_map_is_none(self):
        kw = model_load_kwargs(_accel("mps", True), "bf16")
        assert kw["device_map"] is None
        assert kw["dtype"] == torch.bfloat16
        assert kw["low_cpu_mem_usage"] is True

    def test_fp32_dtype(self):
        assert model_load_kwargs(_accel("mps", True), "fp32")["dtype"] == torch.float32


class TestRayNumGpus:
    def test_cuda_passthrough(self):
        assert ray_num_gpus(_accel("cuda", True), 1) == 1
        assert ray_num_gpus(_accel("cuda", True), 4) == 4

    def test_mps_zero(self):
        assert ray_num_gpus(_accel("mps", True), 1) == 0

    def test_cpu_zero(self):
        assert ray_num_gpus(_accel("cpu", False), 1) == 0


class TestApplyPlatformGuards:
    def _mps(self):
        return _accel("mps", True)

    def test_cuda_is_strict_noop(self):
        tc = {
            "use_flash_attention": "flash_attention_2",
            "num_gpus_per_trial": 4,
            "train_implementation": "FSDP",
            "precision": "bf16",
        }
        un = {"max_concurrent_trials": 8}
        tc_before, un_before = deepcopy(tc), deepcopy(un)
        apply_platform_guards(tc, un, "lora", "none", _accel("cuda", True))
        assert tc == tc_before and un == un_before

    def test_qlora_raises(self):
        with pytest.raises(ValueError, match="[Qq]LoRA"):
            apply_platform_guards({}, {}, "qlora", "none", self._mps())

    def test_rl_algo_raises(self):
        with pytest.raises(ValueError, match="RL"):
            apply_platform_guards({}, {}, "none", "grpo", self._mps())

    def test_multi_gpu_raises(self):
        with pytest.raises(ValueError, match="num_gpus_per_trial"):
            apply_platform_guards({"num_gpus_per_trial": 2}, {}, "lora", "none", self._mps())

    def test_flash_attn_downgraded(self, caplog):
        tc = {"use_flash_attention": "flash_attention_2", "num_gpus_per_trial": 1}
        apply_platform_guards(tc, {}, "lora", "none", self._mps())
        assert tc["use_flash_attention"] == "eager"

    def test_max_concurrent_trials_clamped(self):
        un = {"max_concurrent_trials": 4}
        apply_platform_guards({"num_gpus_per_trial": 1}, un, "lora", "none", self._mps())
        assert un["max_concurrent_trials"] == 1

    def test_precision_downgraded_when_no_bf16(self):
        tc = {"num_gpus_per_trial": 1, "precision": "bf16"}
        apply_platform_guards(tc, {}, "lora", "none", _accel("mps", False))
        assert tc["precision"] == "fp32"

    def test_benign_mps_config_unchanged(self):
        tc = {"num_gpus_per_trial": 1, "use_flash_attention": "eager", "precision": "fp32"}
        tc_before = deepcopy(tc)
        apply_platform_guards(tc, {"max_concurrent_trials": 1}, "sft", "none", self._mps())
        assert tc == tc_before

    # --- MLX backend ---------------------------------------------------------

    def test_qlora_allowed_on_mlx_backend(self):
        tc, tune = {"num_gpus_per_trial": 1}, {"max_concurrent_trials": 1}
        # Must not raise when backend is mlx.
        apply_platform_guards(tc, tune, "qlora", "none", self._mps(), backend="mlx")

    def test_qlora_still_rejected_on_torch_backend(self):
        tc, tune = {"num_gpus_per_trial": 1}, {"max_concurrent_trials": 1}
        with pytest.raises(ValueError, match="QLoRA"):
            apply_platform_guards(tc, tune, "qlora", "none", self._mps(), backend="torch")

    def test_unsupported_tuner_rejected_on_mlx(self):
        tc, tune = {"num_gpus_per_trial": 1}, {"max_concurrent_trials": 1}
        with pytest.raises(ValueError, match="MLX backend supports"):
            apply_platform_guards(tc, tune, "vera", "none", self._mps(), backend="mlx")

    def test_mlx_backend_rejected_off_apple_silicon(self):
        tc, tune = {"num_gpus_per_trial": 1}, {"max_concurrent_trials": 1}
        with pytest.raises(ValueError, match="Apple Silicon"):
            apply_platform_guards(tc, tune, "lora", "none", _accel("cuda", True), backend="mlx")

    def test_cuda_torch_backend_still_noop(self):
        # Default backend + CUDA => strict no-op (the load-bearing invariant).
        tc = {
            "num_gpus_per_trial": 4,
            "use_flash_attention": "flash_attention_2",
            "precision": "bf16",
            "train_implementation": "FSDP",
        }
        tune = {"max_concurrent_trials": 8}
        before_tc, before_tune = deepcopy(tc), deepcopy(tune)
        apply_platform_guards(tc, tune, "qlora", "dpo", _accel("cuda", True), backend="torch")
        assert tc == before_tc and tune == before_tune

    def test_mlx_rl_algo_raises(self):
        with pytest.raises(ValueError, match="RL"):
            apply_platform_guards({"num_gpus_per_trial": 1}, {}, "lora", "grpo", self._mps(), backend="mlx")

    def test_mlx_multi_gpu_raises(self):
        with pytest.raises(ValueError, match="num_gpus_per_trial"):
            apply_platform_guards({"num_gpus_per_trial": 2}, {}, "lora", "none", self._mps(), backend="mlx")

    def test_mlx_max_concurrent_trials_clamped(self):
        tune = {"max_concurrent_trials": 4}
        apply_platform_guards({"num_gpus_per_trial": 1}, tune, "lora", "none", self._mps(), backend="mlx")
        assert tune["max_concurrent_trials"] == 1


class TestConfigureRuntimeEnv:
    def _clear(self, monkeypatch):
        for k in [
            "NCCL_ALGO",
            "VLLM_USE_V1",
            "PYTORCH_ENABLE_MPS_FALLBACK",
            "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION",
            "TOKENIZERS_PARALLELISM",
        ]:
            monkeypatch.delenv(k, raising=False)

    def test_cuda_sets_nccl_and_vllm(self, monkeypatch):
        self._clear(monkeypatch)
        configure_runtime_env(_accel("cuda", True))
        assert os.environ["NCCL_ALGO"] == "Ring"
        assert os.environ["VLLM_USE_V1"] == "1"
        assert os.environ["RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"] == "0.5"
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"

    def test_mps_omits_nccl_sets_fallback(self, monkeypatch):
        self._clear(monkeypatch)
        configure_runtime_env(_accel("mps", True))
        assert "NCCL_ALGO" not in os.environ
        assert "VLLM_USE_V1" not in os.environ
        assert "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION" not in os.environ
        assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


class TestObjectStoreBytes:
    def test_default_is_2gib(self, monkeypatch):
        monkeypatch.delenv("FMTUNE_OBJECT_STORE_BYTES", raising=False)
        assert object_store_bytes() == 2 * 1024 * 1024 * 1024

    def test_override(self, monkeypatch):
        monkeypatch.setenv("FMTUNE_OBJECT_STORE_BYTES", "1073741824")
        assert object_store_bytes() == 1073741824

    def test_bad_override_falls_back(self, monkeypatch):
        monkeypatch.setenv("FMTUNE_OBJECT_STORE_BYTES", "lots")
        assert object_store_bytes() == 2 * 1024 * 1024 * 1024
