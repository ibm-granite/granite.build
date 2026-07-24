# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Import-health checks for the dataset-storage refactor.

Verifies the refactored packages + back-compat shims import and expose the
symbols their consumers rely on. The full FastAPI/MCP entrypoints import the
Granite Build client (gbcli, via dependencies->gb_service) and FastMCP (fastmcp,
via mcp_server), neither installed in this sandbox, so importing them is covered
by a separate check that skips when those deps are absent (run it in a
deps-equipped env).
"""

import importlib
import importlib.util
import pytest


def test_shims_reexport_dataset_and_file_symbols():
    from services import dataset_service, file_service

    # dataset_service shim
    assert dataset_service.Dataset is not None
    assert dataset_service.DatasetIntelligence is not None
    # file_service shim still exposes the runner-critical helpers
    assert callable(file_service.zip_folder)
    assert callable(file_service.get_training_file_path)
    assert callable(file_service.get_dataset_data)
    assert "UPLOAD_DIR" in file_service.CONFIG


def test_refactor_packages_import():
    from services.datasets import Dataset, DatasetIntelligence
    from services.datasets.service import Dataset as SvcDataset
    from services.datasets.intelligence import DatasetIntelligence as Intel
    from services.storage import (
        get_storage_backend,
        StorageBackend,
        DatasetRef,
        DatasetFiles,
        StorageLocator,
    )
    from services.storage.local_backend import LocalStorageBackend

    assert Dataset is SvcDataset
    assert DatasetIntelligence is Intel
    # the public storage surface (ABC + locator + dataclasses) must resolve
    assert all(
        sym is not None
        for sym in (StorageBackend, DatasetRef, DatasetFiles, StorageLocator)
    )
    # local backend is the default when GB disabled
    assert isinstance(get_storage_backend(gb_enabled=False), LocalStorageBackend)


def test_file_package_modules_import():
    from services.file import validation, parsing, streaming, reads

    assert hasattr(validation, "FileValidator")
    assert hasattr(parsing, "FileParser")
    assert hasattr(streaming, "stream_to_disk")
    assert hasattr(reads, "get_dataset_data")


def test_tus_modules_import():
    for name in (
        "services.datasets.tus_metadata",
        "services.datasets.tus_rendezvous",
        "services.datasets.tus_finalize",
        "services.datasets.tus_app",  # relies on the conftest tuspyserver stub
    ):
        mod = importlib.import_module(name)
        assert mod is not None
    from services.datasets.tus_app import create_dataset_tus_router

    assert callable(create_dataset_tus_router)


@pytest.mark.skipif(
    importlib.util.find_spec("gbcli") is None
    or importlib.util.find_spec("fastmcp") is None,
    reason="gbcli/fastmcp not installed; run in a deps-equipped env to verify server/mcp/dependencies import",
)
def test_server_entrypoints_import():
    import dependencies  # noqa: F401
    import mcp_server  # noqa: F401
    from services.storage.gb_backend import GBStorageBackend  # noqa: F401

    assert callable(dependencies.get_dataset_service)
