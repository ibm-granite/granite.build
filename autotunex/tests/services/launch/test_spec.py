"""The pure build-spec builder. Assertions parse the YAML, never match strings.

The generated document is a granite.build ``build.yaml`` — a top-level
``granite.build`` key wrapping ``targets.custom`` — so the helpers below navigate
that tree rather than the flat shape an earlier draft used.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import yaml

from autotunex.services.launch.protocols import LaunchContext
from autotunex.services.launch.spec import REWARD_FUNCTION_PATH, build_spec

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
            "training_config": {"num_gpus_per_trial": {"default": 4}},
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


def _spec(ctx: LaunchContext, *, callback_url: str | None = None) -> dict[str, Any]:
    text = build_spec(
        ctx,
        runtime_image="registry/tuner:1",
        trainer_repo="github.example/trainer.git",
        trainer_ref="stage",
        output_uri_root="hf://huggingface.co/ibm-research",
        callback_url=callback_url,
    )
    return cast("dict[str, Any]", yaml.safe_load(text))


def _custom(doc: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", doc["granite.build"]["targets"]["custom"])


def _step_config(doc: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", _custom(doc)["steps"][0]["config"])


def _command(doc: dict[str, Any]) -> str:
    return cast(str, _step_config(doc)["custom_code_config"]["start_command"])


def test_spec_is_wrapped_and_named_for_the_experiment() -> None:
    doc = _spec(_ctx())

    assert "granite.build" in doc
    assert doc["granite.build"]["name"] == "autotunex-exp1"
    assert _custom(doc)["steps"][0]["step_uri"] == "space://steps/custom_code"


def test_dataset_uri_becomes_the_dataset_files_input() -> None:
    doc = _spec(_ctx())

    assert (
        _custom(doc)["inputs"]["dataset_files"]["uri"] == "hf://huggingface.co/datasets/ibm/alpaca"
    )


def test_output_uri_is_root_plus_autotunex_and_short_job_id() -> None:
    doc = _spec(_ctx())

    assert _custom(doc)["outputs"]["custom"]["uri"] == (
        "hf://huggingface.co/ibm-research/autotunex_11111111/"
    )


def test_custom_code_carries_repo_and_checks_out_the_trainer_ref() -> None:
    ccc = _step_config(_spec(_ctx()))["custom_code_config"]

    assert ccc["github_url"] == "github.example/trainer.git"
    assert ccc["setup_command"] == (
        'git checkout stage && pip install -e ".[full]" && pip list && nvidia-smi'
    )
    assert ccc["dir_to_save"] == "."


def test_start_command_is_the_main_py_invocation() -> None:
    command = _command(_spec(_ctx()))

    assert command.startswith(
        "export CUDA_HOME=/usr/local/cuda-13.0 && export LOG_PATH=$OUTPUT_PATH && python main.py "
    )
    assert "--config_file /tmp/my-config.yaml" in command
    assert "--train_file {{ bindings.dataset_files.binding.path }}/alpaca_train.jsonl" in command
    assert (
        "--validation_file {{ bindings.dataset_files.binding.path }}/alpaca_validation.jsonl"
        in command
    )
    assert "--model_name_or_path ibm/granite" in command
    assert "--run_name exp1" in command
    assert "--output_model_name exp1" in command
    assert "--cleanup --save_history" in command
    assert f"--job_id {JOB_ID}" in command


def test_config_file_name_sanitizes_spaces_consistently() -> None:
    doc = _spec(_ctx(config_name="finance detect v2"))

    files = _step_config(doc)["k8s"]["additional_files"]
    assert "/tmp/finance_detect_v2.yaml" in files
    assert "--config_file /tmp/finance_detect_v2.yaml" in _command(doc)


def test_data_format_drives_the_train_and_validation_extension() -> None:
    command = _command(_spec(_ctx(data_format="csv")))

    assert "/alpaca_train.csv" in command
    assert "/alpaca_validation.csv" in command


def test_image_and_embedded_config_live_under_k8s() -> None:
    config = _step_config(_spec(_ctx()))

    assert config["k8s"]["image"] == "registry/tuner:1"
    embedded = yaml.safe_load(config["k8s"]["additional_files"]["/tmp/my-config.yaml"])
    assert embedded["tune_config"]["max_concurrent_trials"]["default"] == 2


def test_compute_config_derives_gpus_and_fixes_the_rest() -> None:
    compute = _step_config(_spec(_ctx()))["compute_config"]

    assert compute["num_gpus_per_node"] == 8  # 2 concurrent * 4 per trial
    assert compute["num_cpus_per_node"] == 32
    assert compute["num_nodes"] == 1
    assert compute["total_memory_per_node"] == "256Gi"


def test_no_autotune_flag_present_only_when_autotune_false() -> None:
    on = _command(_spec(_ctx(autotune=True)))
    off = _command(_spec(_ctx(autotune=False)))

    assert "--no_autotune" not in on
    assert "--no_autotune" in off


def test_callback_url_emitted_only_when_set() -> None:
    without = _command(_spec(_ctx()))
    with_cb = _command(_spec(_ctx(), callback_url="https://api.example/callback"))

    assert "--autotunex_server_url" not in without
    assert "--autotunex_server_url https://api.example/callback" in with_cb


def test_rl_algo_flag_present_only_for_rl() -> None:
    sft = _command(_spec(_ctx()))
    rl = _command(_spec(_ctx(rl_tuner_type="ppo")))

    assert "--rl_algo" not in sft
    assert "--rl_algo ppo" in rl


def test_tuning_algo_flag_present_only_when_set() -> None:
    with_type = _command(_spec(_ctx(tuning_type="lora")))
    without_type = _command(_spec(_ctx(tuning_type=None)))

    assert "--tuning_algo lora" in with_type
    assert "--tuning_algo" not in without_type


def test_online_rl_rewrites_reward_fields_and_injects_the_file() -> None:
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

    embedded = yaml.safe_load(config["k8s"]["additional_files"]["/tmp/my-config.yaml"])
    assert embedded["training_rl_config"]["reward_function_path"]["default"] == REWARD_FUNCTION_PATH
    assert embedded["training_rl_config"]["reward_function_name"]["default"] == "my_reward"
    assert config["k8s"]["additional_files"][REWARD_FUNCTION_PATH].startswith("def score")


def test_config_data_is_deep_copied_not_mutated() -> None:
    ctx = _ctx(
        rl_tuner_type="ppo",
        config_data={"training_rl_config": {"reward_function_path": {"default": "orig.py"}}},
        reward_function_code="def score():\n    return 1.0\n",
    )

    _spec(ctx)

    # The caller's object is untouched — the rewrite happened on a copy.
    assert ctx.config_data["training_rl_config"]["reward_function_path"]["default"] == "orig.py"
