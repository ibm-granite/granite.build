# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""HF-import settings: the availability gate and its app-config exposure."""

from __future__ import annotations

import pytest

from autotunex.api.routers.app_config import get_app_config
from autotunex.core.config import Settings
from tests.conftest import make_settings


def _gate_settings(**overrides: object) -> Settings:
    """Settings on the standalone-bash happy path, with one field disqualified."""
    base: dict[str, object] = {"gb_environment": "standalone", "lsf_cluster": None}
    base.update(overrides)
    return make_settings(**base)  # type: ignore[arg-type]


def test_defaults_are_permissive_on_standalone_bash() -> None:
    assert make_settings(gb_environment="standalone").hf_import_available is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"hf_import_enabled": False},
        {"gb_environment": "kubernetes"},
        {"lsf_cluster": "some-cluster"},
    ],
)
def test_gate_closes_on_any_disqualifier(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
) -> None:
    # A non-standalone deployment is open only when it can push; with no tokens it
    # stores locally with no locator, so the gate stays closed.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GB_TOKEN", raising=False)
    assert _gate_settings(**overrides).hf_import_available is False


def test_caps_have_sane_defaults() -> None:
    settings = make_settings(gb_environment="standalone")
    assert settings.hf_import_max_rows == 50_000
    assert settings.hf_import_max_bytes == 5 * 1024**3
    assert settings.hf_import_namespaces == []


async def test_app_config_exposes_the_derived_gate_not_the_raw_flag() -> None:
    response = await get_app_config(make_settings(gb_environment="standalone", lsf_cluster="c"))
    assert response.hf_import.available is False
    assert response.hf_import.max_rows == 50_000


def _push_env(monkeypatch: pytest.MonkeyPatch, *, llmb: bool = True) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("GB_TOKEN", "gb_xxx")
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}" if llmb else None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"gb_environment": "prod"},
        {"gb_environment": "prod", "dataset_storage_backend": "huggingface"},
    ],
)
def test_gate_opens_on_a_cluster_that_pushes_to_huggingface(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
) -> None:
    _push_env(monkeypatch)
    assert make_settings(**overrides).hf_import_available is True  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "llmb"),
    [
        ({"gb_environment": "prod", "hf_import_enabled": False}, True),
        ({"gb_environment": "prod"}, False),
        ({"gb_environment": "prod", "dataset_storage_backend": "local"}, True),
        ({"gb_environment": "standalone", "lsf_cluster": "c"}, True),
        (
            {
                "gb_environment": "standalone",
                "lsf_cluster": "c",
                "dataset_storage_backend": "huggingface",
            },
            True,
        ),
    ],
)
def test_gate_stays_closed_where_the_dataset_gets_no_usable_locator(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object], llmb: bool
) -> None:
    _push_env(monkeypatch, llmb=llmb)
    assert make_settings(**overrides).hf_import_available is False  # type: ignore[arg-type]


@pytest.mark.parametrize("missing", ["HF_TOKEN", "GB_TOKEN"])
def test_gate_stays_closed_on_a_cluster_missing_a_token(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    _push_env(monkeypatch)
    monkeypatch.delenv(missing)
    assert make_settings(gb_environment="prod").hf_import_available is False


async def test_app_config_publishes_the_open_gate_on_a_pushing_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _push_env(monkeypatch)
    response = await get_app_config(make_settings(gb_environment="prod"))
    assert response.hf_import.available is True
