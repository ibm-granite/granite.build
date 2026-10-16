"""Tests for the local runner's protocols, context, and reward injection."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from autotunex.services.local.protocols import (
    LocalRunContext,
    LogRecord,
    inject_reward_function,
)


def test_inject_reward_function_writes_file_and_rewrites_config(tmp_path: Path) -> None:
    config = {
        "training_rl_config": {
            "reward_function_path": {"default": "old"},
            "reward_function_name": {"default": "old"},
        }
    }

    inject_reward_function(
        config,
        output_dir=tmp_path,
        code="def compute_score(): ...",
        name="compute_score",
    )

    written = tmp_path / "reward_function.py"
    assert written.read_text().startswith("def compute_score")
    assert config["training_rl_config"]["reward_function_path"]["default"] == str(written)
    assert config["training_rl_config"]["reward_function_name"]["default"] == "compute_score"


def test_inject_reward_function_creates_missing_output_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "run"

    inject_reward_function(
        {"training_rl_config": {"reward_function_path": {"default": "old"}}},
        output_dir=target,
        code="reward = 1",
        name=None,
    )

    assert (target / "reward_function.py").read_text() == "reward = 1"


def test_inject_reward_function_leaves_config_untouched_when_keys_missing(
    tmp_path: Path,
) -> None:
    config: dict[str, Any] = {"tune_config": {"num_samples": {"default": 4}}}

    inject_reward_function(config, output_dir=tmp_path, code="x = 1", name="score")

    assert config == {"tune_config": {"num_samples": {"default": 4}}}
    assert (tmp_path / "reward_function.py").exists()


def test_inject_reward_function_skips_name_when_none(tmp_path: Path) -> None:
    config = {
        "training_rl_config": {
            "reward_function_path": {"default": "old"},
            "reward_function_name": {"default": "keep"},
        }
    }

    inject_reward_function(config, output_dir=tmp_path, code="x = 1", name=None)

    assert config["training_rl_config"]["reward_function_name"]["default"] == "keep"


def test_local_run_context_is_constructible() -> None:
    ctx = LocalRunContext(
        job_id=UUID(int=1),
        model="m",
        model_source="huggingface",
        experiment_name="e",
        tuning_algo="lora",
        rl_algo="none",
        config_name="c",
        config_data={},
        train_file=Path("/t"),
        validation_file=Path("/v"),
        output_dir=Path("/o"),
        seed=1,
        autotune=True,
        cleanup=True,
        save_history=True,
        reward_function_code=None,
        reward_function_name=None,
    )

    assert ctx.tuning_algo == "lora"


def test_log_record_is_constructible() -> None:
    record = LogRecord(
        trial_id="t01",
        level="INFO",
        filename="x.py",
        message="hello",
        iteration=1,
        epoch=0.5,
    )

    assert record.message == "hello"
