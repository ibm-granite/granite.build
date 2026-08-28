"""get_job_runner selects the runner from settings."""

from __future__ import annotations

from autotunex.api.deps import get_job_runner
from autotunex.services.local.runner import LocalJobRunner
from autotunex.services.runner import InProcessJobRunner, NoOpJobRunner
from tests.conftest import make_settings


def test_none_backend_yields_the_noop_runner() -> None:
    runner = get_job_runner(make_settings())

    assert isinstance(runner, NoOpJobRunner)


def test_llmb_backend_yields_the_in_process_runner() -> None:
    settings = make_settings().model_copy(
        update={
            "job_backend": "llmb",
            "job_runtime_image": "registry/tuner:1",
            "job_trainer_repo": "https://example/trainer.git",
            "job_output_uri_root": "s3://bucket/runs",
        }
    )

    runner = get_job_runner(settings)

    assert isinstance(runner, InProcessJobRunner)


def test_local_backend_yields_the_local_job_runner() -> None:
    settings = make_settings().model_copy(update={"job_backend": "local"})

    runner = get_job_runner(settings)

    assert isinstance(runner, LocalJobRunner)


def test_llmb_standalone_yields_the_in_process_runner_for_the_bash_spec() -> None:
    settings = make_settings().model_copy(
        update={
            "job_backend": "llmb",
            "gb_environment": "standalone",
            "gb_server_url": "http://localhost:9000",
        }
    )

    runner = get_job_runner(settings)

    assert isinstance(runner, InProcessJobRunner)
