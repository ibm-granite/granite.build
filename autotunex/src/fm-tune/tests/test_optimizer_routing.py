"""Tests for the AutotuneOptimizer driver-selection decision table.

The routing logic in AutotuneOptimizer.fit() / fit_best_config() is an
inline series of `if rl_algo in ... import driver_*` branches. Without
running real trials, we can't directly invoke that code. Instead we:

  1. Import AutotuneOptimizer to verify the module loads cleanly.
  2. Codify the decision table that fit() implements, asserting the
     expected (multi_gpu, rl_algo, train_implementation) → driver mapping.
     If anyone refactors fit(), they MUST update this table to match.
  3. Verify the constants the optimizer routes on (AUTOTUNE_OFFLINE_RL,
     AUTOTUNE_ONLINE_RL) include the expected algorithm names.
"""

import pytest

from autotune.constants import AUTOTUNE_OFFLINE_RL, AUTOTUNE_ONLINE_RL


def test_optimizer_imports_cleanly():
    """Smoke: AutotuneOptimizer module can be imported without side effects."""
    from autotune.optimizer import AutotuneOptimizer  # noqa: F401


# ---------------------------------------------------------------------------
# Decision table the optimizer implements (see optimizer.py:298–344).
# Format: (multi_gpu, rl_algo, train_implementation) → driver module path
# ---------------------------------------------------------------------------
ROUTING_TABLE = [
    # Single-GPU (multi_gpu=False)
    (False, "none", "DeepSpeed", "autotune.trainers.driver_single"),
    (False, "none", "FSDP", "autotune.trainers.driver_single"),
    (False, "dpo", "DeepSpeed", "autotune.trainers.driver_single_trl"),
    (False, "kto", "DeepSpeed", "autotune.trainers.driver_single_trl"),
    # Multi-GPU online RL → verl regardless of train_implementation
    (True, "ppo", "DeepSpeed", "autotune.trainers.driver_multi_verl"),
    (True, "grpo", "FSDP", "autotune.trainers.driver_multi_verl"),
    (True, "dapo", "DeepSpeed", "autotune.trainers.driver_multi_verl"),
    # Multi-GPU offline RL — TRL with DS or FSDP
    (True, "dpo", "DeepSpeed", "autotune.trainers.driver_multi_trl_ds"),
    (True, "dpo", "FSDP", "autotune.trainers.driver_multi_trl_fsdp"),
    (True, "kto", "DeepSpeed", "autotune.trainers.driver_multi_trl_ds"),
    (True, "kto", "FSDP", "autotune.trainers.driver_multi_trl_fsdp"),
    # Multi-GPU SFT/PEFT
    (True, "none", "DeepSpeed", "autotune.trainers.driver_multi_hf_ds"),
    (True, "none", "FSDP", "autotune.trainers.driver_multi_hf_fsdp"),
]


def _select_driver(multi_gpu: bool, rl_algo: str, train_implementation: str, backend: str = "torch") -> str:
    """Replica of the optimizer's selection logic — kept in lock-step
    with optimizer.py. If you change one, change the other."""
    if not multi_gpu:
        if rl_algo in AUTOTUNE_OFFLINE_RL:
            return "autotune.trainers.driver_single_trl"
        elif rl_algo in AUTOTUNE_ONLINE_RL:
            raise ValueError(f"Online RL {rl_algo} not supported on single GPU")
        else:
            return "autotune.trainers.driver_single_mlx" if backend == "mlx" else "autotune.trainers.driver_single"
    # multi-gpu
    if rl_algo in AUTOTUNE_ONLINE_RL:
        return "autotune.trainers.driver_multi_verl"
    if rl_algo in AUTOTUNE_OFFLINE_RL:
        return (
            "autotune.trainers.driver_multi_trl_fsdp"
            if train_implementation == "FSDP"
            else "autotune.trainers.driver_multi_trl_ds"
        )
    return (
        "autotune.trainers.driver_multi_hf_fsdp"
        if train_implementation == "FSDP"
        else "autotune.trainers.driver_multi_hf_ds"
    )


@pytest.mark.parametrize("multi_gpu,rl_algo,train_impl,expected", ROUTING_TABLE)
def test_decision_table(multi_gpu, rl_algo, train_impl, expected):
    """Each row in ROUTING_TABLE codifies a branch of optimizer.fit()."""
    assert _select_driver(multi_gpu, rl_algo, train_impl) == expected


def test_single_gpu_online_rl_raises():
    """Online RL on single GPU is unsupported and must raise."""
    for rl_algo in AUTOTUNE_ONLINE_RL:
        with pytest.raises(ValueError):
            _select_driver(multi_gpu=False, rl_algo=rl_algo, train_implementation="DeepSpeed")


def test_single_device_backend_routing():
    # Default torch backend -> the HF/PyTorch single-device driver.
    assert _select_driver(False, "none", "FSDP") == "autotune.trainers.driver_single"
    assert _select_driver(False, "none", "FSDP", backend="torch") == "autotune.trainers.driver_single"
    # MLX backend -> the MLX single-device driver.
    assert _select_driver(False, "none", "FSDP", backend="mlx") == "autotune.trainers.driver_single_mlx"
    # backend does not affect multi-GPU or RL routing.
    assert _select_driver(True, "none", "FSDP", backend="mlx") == "autotune.trainers.driver_multi_hf_fsdp"
    assert _select_driver(False, "dpo", "DeepSpeed", backend="mlx") == "autotune.trainers.driver_single_trl"


def test_mlx_driver_module_importable():
    # The module the mlx branch imports must exist and expose the driver fn.
    from autotune.trainers.driver_single_mlx import train_driver_single_gpu  # noqa: F401


class TestRoutingConstantsAreCoherent:
    """If a new RL algo is added, ensure it's classified offline or online."""

    def test_no_orphan_rl_algos(self):
        from autotune.constants import AUTOTUNE_RL_ALGO

        for algo in AUTOTUNE_RL_ALGO:
            if algo == "none":
                continue
            assert algo in AUTOTUNE_OFFLINE_RL or algo in AUTOTUNE_ONLINE_RL, (
                f"RL algo {algo!r} is not classified as offline or online — "
                "the optimizer's driver selection won't know how to route it."
            )


# ---------------------------------------------------------------------------
# Accelerator → final-training routing (fit_best_config).
# multi_gpu is derived from accel.supports_distributed, so on MPS/CPU the
# single-device driver (driver_single) is used regardless of train_implementation.
# ray_num_gpus is the per-trial Ray GPU reservation.
# ---------------------------------------------------------------------------
ACCELERATOR_ROUTING = [
    # (kind, supports_distributed, expected_multi_gpu, ray_gpus_for_1_per_trial)
    ("cuda", True, True, 1),
    ("mps", False, False, 0),
    ("cpu", False, False, 0),
]


@pytest.mark.parametrize("kind,dist,expected_multi_gpu,ray_gpus", ACCELERATOR_ROUTING)
def test_accelerator_routing(kind, dist, expected_multi_gpu, ray_gpus):
    from autotune.device import Accelerator, ray_num_gpus

    accel = Accelerator(kind, 1 if kind != "cpu" else 0, dist, dist, dist, dist)
    # fit_best_config derives: multi_gpu = accel.supports_distributed
    assert accel.supports_distributed is expected_multi_gpu
    # Ray bundle for a 1-GPU-per-trial request
    assert ray_num_gpus(accel, 1) == ray_gpus
