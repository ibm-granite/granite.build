# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from services.storage.base import DatasetFiles


@pytest.fixture
def files(tmp_path):
    d = tmp_path / "d1" / "ds"
    d.mkdir(parents=True)
    (d / "ds_train.jsonl").write_text('{"i":0}\n')
    return DatasetFiles(
        dataset_id="d1-uuid-aaaa-bbbb",
        name="ds",
        data_format="jsonl",
        local_dir=str(d),
        train_file="ds_train.jsonl",
        validation_file="ds_validation.jsonl",
    )


async def test_persist_pushes_and_returns_locator(files):
    from services.storage import gb_backend

    backend = gb_backend.GBStorageBackend()
    backend.gb = MagicMock()
    backend.gb.command_executor = AsyncMock(return_value="pushed uuid=ABC uri=gb://x")
    with patch.object(gb_backend, "extract_uuid_uri", return_value=("ABC", "gb://x")):
        loc = await backend.persist(files)

    assert loc.artifact_id == "ABC"
    assert loc.artifact_url == "gb://x"
    sent = backend.gb.command_executor.call_args.args[0]
    assert "artifact" in sent and "push" in sent
    assert f"{files.name}_{files.dataset_id[:8]}" in sent


async def test_persist_wraps_failure_as_storage_error(files):
    from services.storage import gb_backend
    from services.storage.base import StorageError

    backend = gb_backend.GBStorageBackend()
    backend.gb = MagicMock()
    backend.gb.command_executor = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(StorageError):
        await backend.persist(files)
