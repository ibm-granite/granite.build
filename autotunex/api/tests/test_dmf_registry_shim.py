# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import inspect

from services.registry.base import ModelRegistry


def test_model_registry_abc_surface():
    # The ABC must declare exactly these 10 public methods (9 + dmf-era set).
    expected = {
        "get_checkpoints",
        "pull_all_checkpoint_files",
        "pull_checkpoint_file",
        "get_models",
        "get_all_models",
        "publish_model",
        "delete_model",
        "get_model_detail",
        "get_model_card",
        "search_models",
    }
    declared = {
        name
        for name, _ in inspect.getmembers(ModelRegistry, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert expected <= declared, f"missing: {expected - declared}"


def test_model_registry_async_methods_are_coroutines():
    assert inspect.iscoroutinefunction(ModelRegistry.get_models)
    assert inspect.iscoroutinefunction(ModelRegistry.get_all_models)
    assert inspect.iscoroutinefunction(ModelRegistry.publish_model)
    assert inspect.iscoroutinefunction(ModelRegistry.delete_model)
    # These are SYNC in the real Dmf — must stay sync:
    assert not inspect.iscoroutinefunction(ModelRegistry.get_checkpoints)
    assert not inspect.iscoroutinefunction(ModelRegistry.get_model_detail)


import subprocess
import sys


def test_dmf_service_shim_points_at_dmf_registry():
    import services.dmf_service as shim
    from services.registry.dmf_backend import DmfRegistry

    assert shim.Dmf is DmfRegistry


def test_importing_dmf_backend_does_not_import_lakehouse():
    # Lazy-import guard: importing the module must NOT pull lakehouse.
    # Run in a FRESH interpreter so prior imports in this session don't mask it.
    code = (
        "import sys; "
        "import services.registry.dmf_backend as m; "
        "assert 'lakehouse' not in sys.modules, 'lakehouse imported eagerly'; "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=".",  # tests run from api/
        env={**_env_with_pythonpath()},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def _env_with_pythonpath():
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    return env
