"""Assembly helpers shared by the remote granite.build spec builders."""

from __future__ import annotations

from uuid import UUID

from autotunex.services.launch._shared import (
    REWARD_FUNCTION_PATH,
    inject_reward_function,
    num_gpus_per_node,
    start_command,
)
from autotunex.services.launch.protocols import LaunchContext

JOB_ID = UUID("11111111-1111-1111-1111-111111111111")


def _ctx(**overrides: object) -> LaunchContext:
    base: dict[str, object] = {
        "job_id": JOB_ID,
        "model": "ibm/granite",
        "model_source": "huggingface",
        "experiment_name": "exp1",
        "tuning_type": "lora",
        "rl_tuner_type": None,
        "config_name": "my-config",
        "config_data": {},
        "dataset_name": "alpaca",
        "dataset_uri": "hf://huggingface.co/datasets/ibm/alpaca",
        "data_format": "jsonl",
        "autotune": True,
        "seed": 42,
        "reward_function_code": None,
        "reward_function_name": None,
    }
    base.update(overrides)
    return LaunchContext(**base)  # type: ignore[arg-type]


def test_num_gpus_multiplies_concurrent_trials_by_gpus_per_trial() -> None:
    config = {
        "tune_config": {"max_concurrent_trials": {"default": 2}},
        "training_config": {"num_gpus_per_trial": {"default": 4}},
    }

    assert num_gpus_per_node(config) == 8


def test_num_gpus_defaults_each_factor_to_one() -> None:
    assert num_gpus_per_node({}) == 1


def test_start_command_carries_the_cuda_home_and_core_flags() -> None:
    command = start_command(
        _ctx(),
        config_file="/tmp/my-config.yaml",
        callback_url=None,
        cuda_home="/opt/share/cuda-12.9",
    )

    assert command.startswith(
        "export CUDA_HOME=/opt/share/cuda-12.9 && export LOG_PATH=$OUTPUT_PATH && python main.py "
    )
    assert "--config_file /tmp/my-config.yaml" in command
    assert "--train_file {{ bindings.dataset_files.binding.path }}/alpaca_train.jsonl" in command
    assert "--model_name_or_path ibm/granite" in command
    assert f"--job_id {JOB_ID}" in command


def test_start_command_conditional_flags() -> None:
    on = start_command(_ctx(autotune=True), config_file="/c.yaml", callback_url=None, cuda_home="x")
    off = start_command(
        _ctx(autotune=False, rl_tuner_type="ppo"),
        config_file="/c.yaml",
        callback_url="https://cb.example",
        cuda_home="x",
    )

    assert "--no_autotune" not in on
    assert "--no_autotune" in off
    assert "--rl_algo ppo" in off
    assert "--autotunex_server_url https://cb.example" in off


def test_inject_reward_function_returns_empty_without_code() -> None:
    config: dict[str, object] = {
        "training_rl_config": {"reward_function_path": {"default": "orig.py"}}
    }

    files = inject_reward_function(config, _ctx())

    assert files == {}
    assert config["training_rl_config"]["reward_function_path"]["default"] == "orig.py"  # type: ignore[index]


def test_inject_reward_function_rewrites_paths_and_returns_file() -> None:
    config: dict[str, object] = {
        "training_rl_config": {
            "reward_function_path": {"default": "orig.py"},
            "reward_function_name": {"default": "compute_score"},
        }
    }
    ctx = _ctx(
        reward_function_code="def score():\n    return 1.0\n", reward_function_name="my_reward"
    )

    files = inject_reward_function(config, ctx)

    assert files[REWARD_FUNCTION_PATH].startswith("def score")
    assert config["training_rl_config"]["reward_function_path"]["default"] == REWARD_FUNCTION_PATH  # type: ignore[index]
    assert config["training_rl_config"]["reward_function_name"]["default"] == "my_reward"  # type: ignore[index]
