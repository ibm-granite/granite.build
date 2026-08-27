"""Launcher selection from settings."""

from __future__ import annotations

import pytest

from autotunex.core.config import Settings
from autotunex.services.launch import bash_spec, lsf_spec
from autotunex.services.launch import spec as custom_spec
from autotunex.services.launch.llmb import LlmbBuildCanceller, LlmbTuningLauncher
from autotunex.services.launch.registry import get_build_canceller, get_tuning_launcher


def _llmb_settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        job_backend="llmb",
        job_runtime_image="registry/tuner:1",
        job_trainer_repo="https://example/trainer.git",
        job_output_uri_root="s3://bucket/runs",
        gb_server_url="https://gbserver.example",
    )


def test_llmb_backend_builds_the_llmb_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GB_TOKEN", "gb_xxx")

    launcher = get_tuning_launcher(_llmb_settings())

    assert isinstance(launcher, LlmbTuningLauncher)


def test_llmb_launcher_receives_the_configured_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GB_TOKEN", "gb_xxx")
    settings = _llmb_settings()

    launcher = get_tuning_launcher(settings)

    assert isinstance(launcher, LlmbTuningLauncher)
    assert launcher._tags == settings.gb_tags


def test_none_backend_is_rejected() -> None:
    settings = Settings(_env_file=None, environment="test", job_backend="none")

    with pytest.raises(ValueError):
        get_tuning_launcher(settings)


def test_registry_selects_bash_builder_when_standalone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GB_TOKEN", "t")
    settings = Settings(
        _env_file=None,
        environment="test",
        job_backend="llmb",
        gb_environment="standalone",
        gb_server_url="https://gbserver.example",
    )

    launcher = get_tuning_launcher(settings)

    # The launcher's builder is the bash one (inspect the partial's target func).
    assert isinstance(launcher, LlmbTuningLauncher)
    assert getattr(launcher._spec_builder, "func", None) is bash_spec.build_bash_spec


def test_registry_bash_builder_anchors_output_under_artifact_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GB_TOKEN", "t")
    settings = Settings(
        _env_file=None,
        environment="test",
        job_backend="llmb",
        gb_environment="standalone",
        gb_server_url="https://gbserver.example",
        artifact_dir="/data/artifacts",
    )

    launcher = get_tuning_launcher(settings)

    # The bound output_uri_root must be the artifact_dir as an ABSOLUTE file:// URI
    # so gbserver writes the run's artifacts onto the persistent volume rather than
    # resolving a relative file: URI against its own (ephemeral) CWD.
    bound = launcher._spec_builder.keywords["output_uri_root"]  # type: ignore[attr-defined]
    assert bound == "file:///data/artifacts"


def test_registry_selects_custom_code_builder_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GB_TOKEN", "t")

    launcher = get_tuning_launcher(_llmb_settings())

    assert isinstance(launcher, LlmbTuningLauncher)
    assert getattr(launcher._spec_builder, "func", None) is custom_spec.build_spec


def test_registry_selects_lsf_builder_when_lsf_cluster_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GB_TOKEN", "t")
    settings = Settings(
        _env_file=None,
        environment="test",
        job_backend="llmb",
        gb_environment="standalone",
        gb_server_url="http://localhost:9000",
        lsf_cluster="example-cluster",
        lsf_environment_uri="space://environments/skypilot/lsf/example-cluster",
        lsf_image="registry.example.com/tuner:1",
        job_trainer_repo="https://example.com/trainer.git",
    )

    launcher = get_tuning_launcher(settings)

    assert isinstance(launcher, LlmbTuningLauncher)
    assert getattr(launcher._spec_builder, "func", None) is lsf_spec.build_lsf_spec


def test_get_build_canceller_returns_llmb_canceller_for_llmb_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GB_TOKEN", "gb_xxx")

    canceller = get_build_canceller(_llmb_settings())

    assert isinstance(canceller, LlmbBuildCanceller)


def test_get_build_canceller_rejects_none_backend() -> None:
    settings = Settings(_env_file=None, environment="test", job_backend="none")

    with pytest.raises(ValueError):
        get_build_canceller(settings)


def test_registry_still_selects_bash_when_standalone_without_lsf_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GB_TOKEN", "t")
    settings = Settings(
        _env_file=None,
        environment="test",
        job_backend="llmb",
        gb_environment="standalone",
        gb_server_url="http://localhost:9000",
    )

    launcher = get_tuning_launcher(settings)

    assert isinstance(launcher, LlmbTuningLauncher)
    assert getattr(launcher._spec_builder, "func", None) is bash_spec.build_bash_spec


def test_registry_selects_lsf_builder_with_uppercase_standalone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """granite.build's own uppercase ``GB_ENVIRONMENT=STANDALONE`` still selects LSF.

    The value is normalized case-insensitively before the launcher registry compares it.
    """
    monkeypatch.setenv("GB_TOKEN", "t")
    settings = Settings(
        _env_file=None,
        environment="test",
        job_backend="llmb",
        gb_environment="STANDALONE",
        gb_server_url="http://localhost:9000",
        lsf_cluster="example-cluster",
        lsf_environment_uri="space://environments/skypilot/lsf/example-cluster",
        lsf_image="registry.example.com/tuner:1",
        job_trainer_repo="https://example.com/trainer.git",
    )

    launcher = get_tuning_launcher(settings)

    assert isinstance(launcher, LlmbTuningLauncher)
    assert getattr(launcher._spec_builder, "func", None) is lsf_spec.build_lsf_spec
