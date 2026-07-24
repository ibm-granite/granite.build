# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
from services import file_service
from services.storage.base import DatasetFiles, DatasetRef, StorageNotFound
from services.storage.local_backend import LocalStorageBackend


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    monkeypatch.setitem(file_service.CONFIG, "UPLOAD_DIR", str(tmp_path))
    return str(tmp_path)


def _write_dataset(upload_dir, dataset_id="d1", name="ds", rows=20):
    d = os.path.join(upload_dir, dataset_id, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "ds_train.jsonl"), "w") as f:
        for i in range(rows):
            f.write(json.dumps({"i": i}) + "\n")
    return d


async def test_persist_returns_empty_locator_and_keeps_files(upload_dir):
    d = _write_dataset(upload_dir)
    backend = LocalStorageBackend()
    loc = await backend.persist(
        DatasetFiles(
            dataset_id="d1",
            name="ds",
            data_format="jsonl",
            local_dir=d,
            train_file="ds_train.jsonl",
            validation_file="ds_validation.jsonl",
        )
    )
    assert loc.artifact_id is None and loc.artifact_url is None
    assert os.path.exists(os.path.join(d, "ds_train.jsonl"))  # untouched


async def test_preview_is_row_bounded(upload_dir):
    _write_dataset(upload_dir, rows=1000)
    backend = LocalStorageBackend()
    rows = await backend.preview(
        DatasetRef("d1", "ds", "jsonl"), "ds_train.jsonl", limit=10
    )
    assert len(rows) == 10 and rows[0] == {"i": 0}


async def test_delete_removes_dataset_folder(upload_dir):
    _write_dataset(upload_dir)
    backend = LocalStorageBackend()
    await backend.delete(DatasetRef("d1", "ds", "jsonl"))
    assert not os.path.exists(os.path.join(upload_dir, "d1"))


async def test_preview_missing_file_raises_storage_not_found(upload_dir):
    backend = LocalStorageBackend()
    with pytest.raises(StorageNotFound):
        await backend.preview(
            DatasetRef("nope", "ds", "jsonl"), "ds_train.jsonl", limit=5
        )


async def test_delete_is_idempotent(upload_dir):
    _write_dataset(upload_dir)
    backend = LocalStorageBackend()
    await backend.delete(DatasetRef("d1", "ds", "jsonl"))
    # second delete on an already-removed dataset must not raise
    await backend.delete(DatasetRef("d1", "ds", "jsonl"))
