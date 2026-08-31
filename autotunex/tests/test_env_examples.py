"""Each ``envs/*.env.example`` file loads into a valid ``Settings`` for its runner.

These files exist to give someone setting up one job runner a focused,
copy-and-run configuration. That only holds if each file still matches
``core/config.py``: a variable renamed in the settings, or a required value
left unset in the example, would turn a documented setup into a startup crash
for whoever copied it. The OSS audit flagged exactly this drift for the root
``.env.example``; these tests keep the per-runner files honest.

Each file is loaded through the same pydantic-settings dotenv path production
uses (``Settings(_env_file=<path>)``), so a stale key surfaces as a
``ValidationError`` here rather than in a deployment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from autotunex.core.config import Settings

_ENVS_DIR = Path(__file__).resolve().parents[1] / "envs"


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ambient ``AUTOTUNEX_``/``GB_``/``HF_`` variables before each test.

    ``_env_file=<path>`` overrides the default ``.env``, but an exported
    ``AUTOTUNEX_*`` variable (or the ``GB_TOKEN`` a developer keeps set for the
    llmb tests) still outranks the file, so the assertion would depend on the
    developer's shell rather than the file under test. Stripping them first makes
    each load hermetic. Function-scoped, so a case is free to set the tokens it
    needs afterwards — that ``setenv`` runs after this fixture and is undone at
    teardown.
    """
    for name in [key for key in os.environ if key.startswith("AUTOTUNEX_")]:
        monkeypatch.delenv(name)
    for name in ("GB_ENVIRONMENT", "GB_TOKEN", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_exactly_the_four_runner_files_are_present() -> None:
    found = sorted(path.name for path in _ENVS_DIR.glob("*.env.example"))

    assert found == [
        "bash.env.example",
        "local.env.example",
        "lsf.env.example",
        "remote.env.example",
    ]


def test_local_example_loads_as_the_local_runner() -> None:
    settings = Settings(_env_file=str(_ENVS_DIR / "local.env.example"))

    assert settings.job_backend == "local"
    assert settings.gb_environment is None
    assert settings.dataset_storage_backend == "local"


def test_bash_example_loads_as_the_standalone_bash_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GB_TOKEN", "test-gb-token")
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")

    settings = Settings(_env_file=str(_ENVS_DIR / "bash.env.example"))

    assert settings.job_backend == "llmb"
    assert settings.gb_environment == "standalone"
    assert settings.gb_server_url  # required — the reconcile loop polls it


def test_lsf_example_loads_as_the_lsf_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GB_TOKEN", "test-gb-token")
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")

    settings = Settings(_env_file=str(_ENVS_DIR / "lsf.env.example"))

    assert settings.job_backend == "llmb"
    assert settings.gb_environment == "standalone"
    assert settings.lsf_cluster  # the discriminator is set → LSF variant
    assert settings.lsf_environment_uri
    assert settings.lsf_image
    assert settings.job_trainer_repo  # required in LSF mode


def test_remote_example_loads_as_the_custom_code_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GB_TOKEN", "test-gb-token")
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")

    settings = Settings(_env_file=str(_ENVS_DIR / "remote.env.example"))

    assert settings.job_backend == "llmb"
    assert settings.gb_environment is None  # unset selects the remote custom_code spec
    assert settings.job_runtime_image  # the custom_code cluster inputs are all set
    assert settings.job_trainer_repo
    assert settings.job_output_uri_root
