"""The pure LSF/SkyPilot build-spec builder. Assertions parse the YAML."""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import yaml

from autotunex.services.launch._shared import REWARD_FUNCTION_PATH
from autotunex.services.launch.lsf_spec import build_lsf_spec
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
        "config_data": {
            "tune_config": {"max_concurrent_trials": {"default": 2}},
            "training_config": {"num_gpus_per_trial": {"default": 1}},
        },
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


def _spec(ctx: LaunchContext, **overrides: object) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "environment_uri": "space://environments/skypilot/lsf/example-cluster",
        "image": "registry.example.com/tuner:1",
        "venv_path": "/step_venv",
        "cuda_home": "/opt/share/cuda-12.9",
        "trainer_repo": "https://example.com/trainer.git",
        "trainer_ref": "main",
        "cluster": "example-cluster",
        "queue": "normal",
        "accelerators": "H100:2",
        "memory": "256+",
        "num_cpus_per_node": 32,
        "total_memory_per_node": "256Gi",
        "poll_interval_seconds": 30,
        "callback_url": None,
    }
    kwargs.update(overrides)
    text = build_lsf_spec(ctx, **kwargs)
    return cast("dict[str, Any]", yaml.safe_load(text))


def _target(doc: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", doc["granite.build"]["targets"]["autotunex-tune"])


def _step_config(doc: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", _target(doc)["steps"][0]["config"])


def test_spec_is_wrapped_and_uses_the_autotunex_tune_step() -> None:
    doc = _spec(_ctx())

    assert doc["granite.build"]["name"] == "autotunex-exp1"
    assert "autotunex-tune" in doc["granite.build"]["targets"]
    assert _target(doc)["environment_uri"] == "space://environments/skypilot/lsf/example-cluster"
    assert _target(doc)["steps"][0]["step_uri"] == "space://steps/autotunex-tune"


def test_dataset_input_and_env_output() -> None:
    doc = _spec(_ctx())

    assert (
        _target(doc)["inputs"]["dataset_files"]["uri"] == "hf://huggingface.co/datasets/ibm/alpaca"
    )
    assert _target(doc)["outputs"]["checkpoint"]["uri"] == "env://{{ binding.path }}"
    assert _target(doc)["outputs"]["checkpoint"]["type"] == "model"


def test_poll_intervals_and_custom_code_config() -> None:
    config = _step_config(_spec(_ctx()))

    assert config["poll_interval_seconds"] == 30
    assert config["log_retrieval_interval_seconds"] == 30
    ccc = config["custom_code_config"]
    assert ccc["github_url"] == "https://example.com/trainer.git"
    assert ccc["github_ref"] == "main"
    assert ccc["setup_command"] == (
        'git checkout main && pip install -e ".[full]" && pip list && nvidia-smi'
    )
    assert ccc["dir_to_save"] == "$OUTPUT_PATH"


def test_start_command_uses_lsf_cuda_home_and_dataset_binding() -> None:
    command = _step_config(_spec(_ctx()))["custom_code_config"]["start_command"]

    assert command.startswith(
        "export CUDA_HOME=/opt/share/cuda-12.9 && export LOG_PATH=$OUTPUT_PATH && python main.py "
    )
    assert "--config_file /tmp/my-config.yaml" in command
    assert "--train_file {{ bindings.dataset_files.binding.path }}/alpaca_train.jsonl" in command
    assert f"--job_id {JOB_ID}" in command


def test_skypilot_block_carries_image_venv_and_embedded_config() -> None:
    config = _step_config(_spec(_ctx()))

    sky = config["skypilot"]
    assert sky["image"] == "registry.example.com/tuner:1"
    assert sky["venv_path"] == "/step_venv"
    embedded = yaml.safe_load(sky["additional_files"]["/tmp/my-config.yaml"])
    assert embedded["tune_config"]["max_concurrent_trials"]["default"] == 2


def test_compute_config_derives_gpus_and_fixes_node_count() -> None:
    compute = _step_config(_spec(_ctx()))["compute_config"]

    assert compute["num_gpus_per_node"] == 2  # 2 concurrent * 1 per trial
    assert compute["num_cpus_per_node"] == 32
    assert compute["num_nodes"] == 1
    assert compute["total_memory_per_node"] == "256Gi"


def test_launcher_resources_include_set_keys() -> None:
    resources = _step_config(_spec(_ctx()))["launcher_config"]["resources"]

    assert resources["cluster"] == "example-cluster"
    assert resources["accelerators"] == "H100:2"
    assert resources["zone"] == "normal"
    assert resources["memory"] == "256+"


def test_launcher_resources_omit_unset_keys_for_zero_gpu() -> None:
    resources = _step_config(_spec(_ctx(), accelerators=None, queue=None, memory=None))[
        "launcher_config"
    ]["resources"]

    assert resources == {"cluster": "example-cluster"}


def test_config_file_name_sanitizes_spaces() -> None:
    doc = _spec(_ctx(config_name="finance detect v2"))

    files = _step_config(doc)["skypilot"]["additional_files"]
    assert "/tmp/finance_detect_v2.yaml" in files
    assert (
        "--config_file /tmp/finance_detect_v2.yaml"
        in (_step_config(doc)["custom_code_config"]["start_command"])
    )


def test_online_rl_injects_reward_file_and_rewrites_paths() -> None:
    ctx = _ctx(
        rl_tuner_type="ppo",
        config_data={
            "training_rl_config": {
                "reward_function_path": {"default": "orig.py"},
                "reward_function_name": {"default": "compute_score"},
            }
        },
        reward_function_code="def score():\n    return 1.0\n",
        reward_function_name="my_reward",
    )

    config = _step_config(_spec(ctx))

    files = config["skypilot"]["additional_files"]
    embedded = yaml.safe_load(files["/tmp/my-config.yaml"])
    assert embedded["training_rl_config"]["reward_function_path"]["default"] == REWARD_FUNCTION_PATH
    assert embedded["training_rl_config"]["reward_function_name"]["default"] == "my_reward"
    assert files[REWARD_FUNCTION_PATH].startswith("def score")
    assert "--rl_algo ppo" in config["custom_code_config"]["start_command"]


def test_config_data_is_not_mutated() -> None:
    ctx = _ctx(
        rl_tuner_type="ppo",
        config_data={"training_rl_config": {"reward_function_path": {"default": "orig.py"}}},
        reward_function_code="def score():\n    return 1.0\n",
    )

    _spec(ctx)

    assert ctx.config_data["training_rl_config"]["reward_function_path"]["default"] == "orig.py"
