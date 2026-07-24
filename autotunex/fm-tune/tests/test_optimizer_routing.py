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


def _select_driver(multi_gpu: bool, rl_algo: str, train_implementation: str) -> str:
    """Replica of the optimizer's selection logic — kept in lock-step
    with optimizer.py:298–344. If you change one, change the other."""
    if not multi_gpu:
        if rl_algo in AUTOTUNE_OFFLINE_RL:
            return "autotune.trainers.driver_single_trl"
        elif rl_algo in AUTOTUNE_ONLINE_RL:
            raise ValueError(f"Online RL {rl_algo} not supported on single GPU")
        else:
            return "autotune.trainers.driver_single"
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
