# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Resolved data-directory paths for the API server.

Every data dir defaults to a subfolder under AUTOTUNEX_DATA_DIR
(default <cwd>/AUTOTUNEX_DATA). Each specific env var still overrides its own
subfolder when set. Functions re-read os.getenv on every call so tests that
monkeypatch env vars — and any runtime env change — are honored (module-level
constants captured at import would not be).
"""

import os


def data_dir() -> str:
    return os.getenv("AUTOTUNEX_DATA_DIR", os.path.join(os.getcwd(), "AUTOTUNEX_DATA"))


def datasets_path() -> str:
    return os.getenv("AUTOTUNE_DATASETS_PATH", os.path.join(data_dir(), "datasets"))


def results_path() -> str:
    return os.getenv("AUTOTUNE_RESULTS_PATH", os.path.join(data_dir(), "results"))


def dmf_cache() -> str:
    return os.getenv("DMF_CACHE", os.path.join(data_dir(), "cache"))


def log_path() -> str:
    return os.getenv("LOG_PATH", os.path.join(data_dir(), "logs"))


def models_dir() -> str:
    return os.getenv("AUTOTUNEX_MODELS_DIR", os.path.join(results_path(), "registry"))
