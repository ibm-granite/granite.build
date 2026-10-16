"""The TuningLauncher seam: value objects and Protocol conformance."""

from __future__ import annotations

from uuid import uuid4

from autotunex.services.launch.protocols import LaunchContext, LaunchHandle, TuningLauncher


class _FakeLauncher:
    async def launch(self, ctx: LaunchContext) -> LaunchHandle:
        return LaunchHandle(build_id=None, pr_url=None)


def test_fake_launcher_satisfies_the_protocol() -> None:
    """A type-level assertion: mypy fails here if the Protocol drifts."""
    launcher: TuningLauncher = _FakeLauncher()

    assert launcher is not None


def test_launch_context_carries_the_launch_inputs() -> None:
    ctx = LaunchContext(
        job_id=uuid4(),
        model="ibm/granite",
        model_source="huggingface",
        experiment_name="exp",
        tuning_type="lora",
        rl_tuner_type=None,
        config_name="my-config",
        config_data={"a": 1},
        dataset_name="alpaca",
        dataset_uri="s3://x",
        data_format="jsonl",
        autotune=True,
        seed=42,
        reward_function_code=None,
        reward_function_name=None,
    )

    assert ctx.model == "ibm/granite"
    assert ctx.config_data == {"a": 1}
    assert ctx.config_name == "my-config"
    assert ctx.data_format == "jsonl"
