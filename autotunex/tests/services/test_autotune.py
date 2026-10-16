"""Unit tests for the AutotuneCore seam.

The adapter's import path is exercised by injecting a fake ``autotune.catalog``
module into ``sys.modules`` (auto-undone by ``monkeypatch``); the missing-package
path is forced by injecting a ``None`` entry, which makes ``from autotune.catalog
import ...`` raise ImportError even when ``autotune`` happens to be installed (it
is an optional extra — absent in CI, present in a granite-build install). The
memoized loaders are cache-cleared around every test so a payload from one test
never leaks into the next.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from autotunex.core.exceptions import AutotuneCoreUnavailableError
from autotunex.services.autotune import (
    AutotuneCore,
    AutotuneCoreAdapter,
    _load_config_template,
    _load_dataset_types,
    _type_to_str,
)


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    _load_config_template.cache_clear()
    _load_dataset_types.cache_clear()
    yield
    _load_config_template.cache_clear()
    _load_dataset_types.cache_clear()


def _install_fake_autotune(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: dict[str, Any] | None = None,
    dataset_types: dict[str, Any] | None = None,
    on_types: list[int] | None = None,
) -> None:
    """Install a fake ``autotune.catalog`` into ``sys.modules`` for one test."""
    catalog = types.ModuleType("autotune.catalog")

    def get_autotune_config() -> dict[str, Any]:
        return config if config is not None else {}

    def get_autotune_dataset_types() -> dict[str, Any]:
        if on_types is not None:
            on_types.append(1)
        return dataset_types if dataset_types is not None else {}

    catalog.get_autotune_config = get_autotune_config  # type: ignore[attr-defined]
    catalog.get_autotune_dataset_types = get_autotune_dataset_types  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autotune", types.ModuleType("autotune"))
    monkeypatch.setitem(sys.modules, "autotune.catalog", catalog)


def test_the_adapter_satisfies_the_protocol() -> None:
    core: AutotuneCore = AutotuneCoreAdapter()

    assert core is not None


def test_type_to_str_unwraps_type_objects() -> None:
    assert _type_to_str(str) == "str"
    assert _type_to_str("already-a-string") == "already-a-string"


async def test_get_config_template_returns_autotune_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_autotune(monkeypatch, config={"tune_config": {"x": {"default": 1}}})

    template = await AutotuneCoreAdapter().get_config_template()

    assert template == {"tune_config": {"x": {"default": 1}}}


async def test_get_dataset_types_stringifies_column_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_autotune(
        monkeypatch,
        dataset_types={"sft": {"columns": {"prompt": {"type": str}, "count": {"type": int}}}},
    )

    dataset_types = await AutotuneCoreAdapter().get_dataset_types()

    assert dataset_types["sft"]["columns"]["prompt"]["type"] == "str"
    assert dataset_types["sft"]["columns"]["count"]["type"] == "int"


async def test_missing_autotune_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ``None`` entry makes ``from autotune.catalog import ...`` raise ImportError, so
    # the missing-package path is exercised whether or not ``autotune`` is installed.
    monkeypatch.setitem(sys.modules, "autotune", None)
    monkeypatch.setitem(sys.modules, "autotune.catalog", None)

    with pytest.raises(AutotuneCoreUnavailableError):
        await AutotuneCoreAdapter().get_config_template()


async def test_dataset_types_loader_is_memoized(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    _install_fake_autotune(monkeypatch, dataset_types={"sft": {"columns": {}}}, on_types=calls)
    adapter = AutotuneCoreAdapter()

    await adapter.get_dataset_types()
    await adapter.get_dataset_types()

    assert len(calls) == 1


async def test_real_adapter_serves_the_catalog_when_autotune_is_installed() -> None:
    # No monkeypatching: exercises the real, installed torch-free autotune.catalog.
    pytest.importorskip("autotune.catalog")

    adapter = AutotuneCoreAdapter()
    template = await adapter.get_config_template()
    dataset_types = await adapter.get_dataset_types()

    assert isinstance(template, dict) and template
    assert {"dataset_type_a", "dataset_type_b", "dataset_type_c", "dataset_type_d"} <= set(
        dataset_types
    )
    # dataset-type column `type` objects must be stringified for JSON-serializability
    for dtype in dataset_types.values():
        for col in dtype.get("columns", {}).values():
            assert not isinstance(col.get("type"), type)
