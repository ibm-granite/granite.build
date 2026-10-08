# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Settings.resolved_dataset_storage: the one dataset-storage decision.

``shutil.which`` and the two token env vars are patched per case, so no case
depends on a real ``llmb`` install or on a developer's exported tokens.
"""

from __future__ import annotations

import pytest

from autotunex.core.config import DatasetStorage
from tests.conftest import make_settings


def _env(monkeypatch: pytest.MonkeyPatch, *, llmb: bool, hf: bool, gb: bool) -> None:
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}" if llmb else None)
    for name, present in (("HF_TOKEN", hf), ("GB_TOKEN", gb)):
        if present:
            monkeypatch.setenv(name, "tok")
        else:
            monkeypatch.delenv(name, raising=False)


# (dataset_storage_backend, gb_environment, lsf_cluster, llmb, hf, gb) -> expected
CASES: list[tuple[str, str | None, str | None, bool, bool, bool, DatasetStorage]] = [
    # forced local: a file:// locator only on the same-host bash standalone build
    ("local", "standalone", None, True, True, True, "local_file_uri"),
    ("local", "standalone", "lsf-a", True, True, True, "local"),
    ("local", "prod", None, True, True, True, "local"),
    # forced huggingface is always huggingface (bash standalone is refused at validation)
    ("huggingface", "prod", None, False, True, True, "huggingface"),
    ("huggingface", "standalone", "lsf-a", True, True, True, "huggingface"),
    # auto in standalone never pushes, whatever the tokens
    ("auto", "standalone", None, True, True, True, "local_file_uri"),
    ("auto", "standalone", "lsf-a", True, True, True, "local"),
    # auto outside standalone pushes only with llmb and both tokens
    ("auto", "prod", None, True, True, True, "huggingface"),
    ("auto", "prod", None, False, True, True, "local"),
    ("auto", "prod", None, True, False, True, "local"),
    ("auto", "prod", None, True, True, False, "local"),
    ("auto", None, None, True, True, True, "huggingface"),
]


@pytest.mark.parametrize(
    ("backend", "gb_environment", "lsf_cluster", "llmb", "hf", "gb", "expected"), CASES
)
def test_resolved_dataset_storage(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    gb_environment: str | None,
    lsf_cluster: str | None,
    llmb: bool,
    hf: bool,
    gb: bool,
    expected: DatasetStorage,
) -> None:
    _env(monkeypatch, llmb=llmb, hf=hf, gb=gb)
    settings = make_settings(
        dataset_storage_backend=backend,  # type: ignore[arg-type]
        gb_environment=gb_environment,
        lsf_cluster=lsf_cluster,
    )

    assert settings.resolved_dataset_storage == expected


def test_resolution_follows_the_environment_after_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read per call, like get_storage_backend: a token added later is seen."""
    _env(monkeypatch, llmb=True, hf=True, gb=False)
    settings = make_settings(gb_environment="prod")
    assert settings.resolved_dataset_storage == "local"

    monkeypatch.setenv("GB_TOKEN", "tok")

    assert settings.resolved_dataset_storage == "huggingface"  # type: ignore[comparison-overlap]
