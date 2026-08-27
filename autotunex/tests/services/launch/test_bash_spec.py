"""The pure local-bash build-spec builder. Assertions parse the YAML, never match strings.

The generated document is a granite.build ``build.yaml`` in the local-bash shape —
a top-level ``granite.build`` key wrapping ``targets.custom`` bound to
``space://environments/bash`` and a single ``space://steps/autotune`` step — so the
helpers below navigate that tree rather than the flat shape an earlier draft used.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import yaml

from autotunex.services.launch.bash_spec import build_bash_spec
from autotunex.services.launch.protocols import LaunchContext

JOB_ID = UUID("87b98242-7ebd-476e-90f4-147a727ca4ad")


def _ctx(**overrides: object) -> LaunchContext:
    base: dict[str, object] = {
        "job_id": JOB_ID,
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_source": "huggingface",
        "experiment_name": "SmolLM2-135M-Instruct_smollm2",
        "tuning_type": "lora",
        "rl_tuner_type": None,
        "config_name": "config",
        "config_data": {"tune_config": {"num_samples": {"default": 4}}},
        "dataset_name": "policy_val",
        "dataset_uri": "hf://huggingface.co/datasets/ibm-research/policy_val_193a11ef",
        "data_format": "jsonl",
        "autotune": True,
        "seed": 42,
        "reward_function_code": None,
        "reward_function_name": None,
    }
    base.update(overrides)
    return LaunchContext(**base)  # type: ignore[arg-type]


def _custom(text: str) -> dict[str, Any]:
    doc = cast("dict[str, Any]", yaml.safe_load(text))
    return cast("dict[str, Any]", doc["granite.build"]["targets"]["custom"])


def test_build_bash_spec_matches_the_bash_shape() -> None:
    text = build_bash_spec(
        _ctx(),
        fm_tune_root="https://github.example.com/org/monorepo",
        fm_tune_ref=None,
        fm_tune_extra="full,mlx",
        backend="mlx",
        callback_url="http://localhost:8001",
        output_uri_root="file:///data/artifacts",
    )

    doc = cast("dict[str, Any]", yaml.safe_load(text))["granite.build"]
    target = doc["targets"]["custom"]
    step = target["steps"][0]

    assert doc["name"] == "custom-model-SmolLM2-135M-Instruct_smollm2"
    assert doc["version"] == "0.0.1"
    assert target["environment_uri"] == "space://environments/bash"
    assert target["inputs"]["model"]["uri"] == "hf:///HuggingFaceTB/SmolLM2-135M-Instruct"
    assert (
        target["inputs"]["dataset_files"]["uri"]
        == "hf://huggingface.co/datasets/ibm-research/policy_val_193a11ef"
    )
    assert (
        target["outputs"]["custom"]["uri"]
        == "file:///data/artifacts/autotune_87b98242-7ebd-476e-90f4-147a727ca4ad/"
    )
    assert step["step_uri"] == "space://steps/autotune"
    assert step["config"]["compute_config"]["num_nodes"] == 1
    assert step["config"]["autotune-config"] == {"tune_config": {"num_samples": {"default": 4}}}


def test_build_bash_spec_env_block() -> None:
    text = build_bash_spec(
        _ctx(autotune=False),
        fm_tune_root="R",
        fm_tune_ref="oss-main",
        fm_tune_extra="full,mlx",
        backend="mlx",
        callback_url="http://localhost:8001",
        output_uri_root="file:///data/artifacts",
    )

    env = _custom(text)["steps"][0]["config"]["bash"]["env"]

    assert env["FM_TUNE_ROOT"] == "R"
    assert env["FM_TUNE_REF"] == "oss-main"
    assert env["FM_TUNE_EXTRA"] == "full,mlx"
    assert env["BACKEND"] == "mlx"
    assert env["NO_AUTOTUNE"] == "true"
    assert env["RUN_NAME"] == "SmolLM2-135M-Instruct_smollm2"
    assert env["JOB_ID"] == "87b98242-7ebd-476e-90f4-147a727ca4ad"
    assert env["AUTOTUNEX_SERVER_URL"] == "http://localhost:8001"


def test_build_bash_spec_fm_tune_ref_is_null_when_unset() -> None:
    text = build_bash_spec(
        _ctx(),
        fm_tune_root="R",
        fm_tune_ref=None,
        fm_tune_extra="full,mlx",
        backend="mlx",
        callback_url=None,
        output_uri_root="file:///data/artifacts",
    )

    env = _custom(text)["steps"][0]["config"]["bash"]["env"]

    assert env["FM_TUNE_REF"] is None


def test_build_bash_spec_defaults_server_url_when_callback_absent() -> None:
    text = build_bash_spec(
        _ctx(),
        fm_tune_root="R",
        fm_tune_ref=None,
        fm_tune_extra="full,mlx",
        backend="mlx",
        callback_url=None,
        output_uri_root="file:///data/artifacts",
    )

    env = _custom(text)["steps"][0]["config"]["bash"]["env"]

    assert env["AUTOTUNEX_SERVER_URL"] == "http://localhost:8001"


def test_build_bash_spec_no_autotune_is_false_when_autotune_true() -> None:
    text = build_bash_spec(
        _ctx(autotune=True),
        fm_tune_root="R",
        fm_tune_ref=None,
        fm_tune_extra="full,mlx",
        backend="mlx",
        callback_url=None,
        output_uri_root="file:///data/artifacts",
    )

    env = _custom(text)["steps"][0]["config"]["bash"]["env"]

    assert env["NO_AUTOTUNE"] == "false"


def test_build_bash_spec_anchors_output_under_root() -> None:
    # A trailing slash on the root must not double up in the emitted URI, and the
    # absolute root must be preserved verbatim (gbserver honors an absolute file:
    # URI; a relative one would resolve against gbserver's own CWD instead).
    text = build_bash_spec(
        _ctx(),
        fm_tune_root="R",
        fm_tune_ref=None,
        fm_tune_extra="full,mlx",
        backend="mlx",
        callback_url=None,
        output_uri_root="file:///data/artifacts/",
    )

    assert (
        _custom(text)["outputs"]["custom"]["uri"]
        == "file:///data/artifacts/autotune_87b98242-7ebd-476e-90f4-147a727ca4ad/"
    )


def test_build_bash_spec_does_not_mutate_config_data() -> None:
    original = {"tune_config": {"num_samples": {"default": 4}}}
    ctx = _ctx(config_data=original)

    build_bash_spec(
        ctx,
        fm_tune_root="R",
        fm_tune_ref=None,
        fm_tune_extra="full,mlx",
        backend="mlx",
        callback_url=None,
        output_uri_root="file:///data/artifacts",
    )

    assert original == {"tune_config": {"num_samples": {"default": 4}}}


def test_build_bash_spec_passes_file_uri_dataset_through() -> None:
    # A standalone local dataset is referenced by an absolute file:// directory
    # URI; the bash spec is scheme-agnostic and must carry it verbatim.
    file_uri = "file:///data/artifacts/datasets/193a11ef-0000-0000-0000-000000000000"
    text = build_bash_spec(
        _ctx(dataset_uri=file_uri),
        fm_tune_root="R",
        fm_tune_ref=None,
        fm_tune_extra="full,mlx",
        backend="mlx",
        callback_url=None,
        output_uri_root="file:///data/artifacts",
    )

    assert _custom(text)["inputs"]["dataset_files"]["uri"] == file_uri
