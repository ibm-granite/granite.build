# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import os

import paths


def _clear(monkeypatch):
    for var in (
        "AUTOTUNEX_DATA_DIR",
        "AUTOTUNE_DATASETS_PATH",
        "AUTOTUNE_RESULTS_PATH",
        "DMF_CACHE",
        "LOG_PATH",
        "AUTOTUNEX_MODELS_DIR",
    ):
        monkeypatch.delenv(var, raising=False)


def test_defaults_nest_under_cwd_autotunex_data(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.chdir(tmp_path)
    base = os.path.join(str(tmp_path), "AUTOTUNEX_DATA")

    assert paths.data_dir() == base
    assert paths.datasets_path() == os.path.join(base, "datasets")
    assert paths.results_path() == os.path.join(base, "results")
    assert paths.dmf_cache() == os.path.join(base, "cache")
    assert paths.log_path() == os.path.join(base, "logs")
    assert paths.models_dir() == os.path.join(base, "results", "registry")


def test_data_dir_override_relocates_all(monkeypatch, tmp_path):
    _clear(monkeypatch)
    root = str(tmp_path / "custom-root")
    monkeypatch.setenv("AUTOTUNEX_DATA_DIR", root)

    assert paths.data_dir() == root
    assert paths.datasets_path() == os.path.join(root, "datasets")
    assert paths.results_path() == os.path.join(root, "results")
    assert paths.dmf_cache() == os.path.join(root, "cache")
    assert paths.log_path() == os.path.join(root, "logs")
    assert paths.models_dir() == os.path.join(root, "results", "registry")


def test_specific_var_overrides_take_precedence(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("AUTOTUNEX_DATA_DIR", str(tmp_path / "root"))
    monkeypatch.setenv("AUTOTUNE_DATASETS_PATH", "/explicit/datasets")
    monkeypatch.setenv("AUTOTUNE_RESULTS_PATH", "/explicit/results")
    monkeypatch.setenv("DMF_CACHE", "/explicit/cache")
    monkeypatch.setenv("LOG_PATH", "/explicit/logs")

    assert paths.datasets_path() == "/explicit/datasets"
    assert paths.results_path() == "/explicit/results"
    assert paths.dmf_cache() == "/explicit/cache"
    assert paths.log_path() == "/explicit/logs"
    # models_dir nests under the (overridden) results_path when its own var is unset
    assert paths.models_dir() == os.path.join("/explicit/results", "registry")


def test_models_dir_own_override_wins(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("AUTOTUNE_RESULTS_PATH", "/explicit/results")
    monkeypatch.setenv("AUTOTUNEX_MODELS_DIR", "/explicit/models")
    assert paths.models_dir() == "/explicit/models"


def test_functions_reread_env_each_call(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("LOG_PATH", "/first")
    assert paths.log_path() == "/first"
    monkeypatch.setenv("LOG_PATH", "/second")
    assert paths.log_path() == "/second"
