# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from services.datasets.service import Dataset
from services.storage.base import StorageLocator
import models as api


def _db_with_dataset():
    db = MagicMock()
    db.get_dataset = AsyncMock(
        return_value={
            "id": "d1",
            "name": "ds",
            "train_file": "ds_train",
            "validation_file": "ds_validation",
            "data_format": "jsonl",
        }
    )
    db.update_dataset_metadata = AsyncMock(return_value={"id": "d1"})
    db.delete_dataset = AsyncMock(return_value=True)
    return db


async def test_get_dataset_uses_backend_preview():
    db = _db_with_dataset()
    ds = Dataset(db)
    fake_backend = MagicMock()
    fake_backend.preview = AsyncMock(side_effect=[[{"r": 1}], [{"r": 2}]])
    with patch(
        "services.datasets.service.get_storage_backend", return_value=fake_backend
    ):
        result = await ds.get_dataset(id="d1", user_id="u1")
    assert result["train_data"] == [{"r": 1}]
    assert result["validation_data"] == [{"r": 2}]
    # both previews bounded to limit=10
    for call in fake_backend.preview.call_args_list:
        assert call.kwargs.get("limit") == 10


async def test_delete_routes_through_backend():
    db = _db_with_dataset()
    ds = Dataset(db)
    fake_backend = MagicMock()
    fake_backend.delete = AsyncMock()
    with patch(
        "services.datasets.service.get_storage_backend", return_value=fake_backend
    ):
        ok = await ds.delete_dataset(id="d1", user_id="u1")
    assert ok is True
    fake_backend.delete.assert_awaited_once()
    db.delete_dataset.assert_awaited_once()


async def test_finalize_records_locator_from_backend():
    db = _db_with_dataset()
    ds = Dataset(db)
    fake_backend = MagicMock()
    fake_backend.persist = AsyncMock(return_value=StorageLocator("AID", "gb://u"))
    data = {
        "id": "d1",
        "name": "ds",
        "train_file": "ds_train",
        "validation_file": "ds_validation",
    }
    with patch(
        "services.datasets.service.get_storage_backend", return_value=fake_backend
    ):
        await ds._finalize_upload("d1", "u1", data, metadata={"data_format": "jsonl"})
    saved = db.update_dataset_metadata.call_args.kwargs["metadata"]
    assert saved["artifact_id"] == "AID" and saved["artifact_url"] == "gb://u"


async def test_ai_methods_delegate_to_intelligence():
    db = _db_with_dataset()
    ds = Dataset(db)
    ds._intelligence.generate_parsing_strategy = AsyncMock(
        return_value={"type": "direct_mapping"}
    )
    out = await ds.generate_parsing_strategy([{"input": "a", "output": "b"}], "jsonl")
    assert out == {"type": "direct_mapping"}


async def test_finalize_rolls_back_on_persist_failure():
    db = _db_with_dataset()
    ds = Dataset(db)
    ds.delete_dataset = AsyncMock(return_value=True)
    fake_backend = MagicMock()
    fake_backend.persist = AsyncMock(side_effect=RuntimeError("backend down"))
    data = {
        "id": "d1",
        "name": "ds",
        "train_file": "ds_train",
        "validation_file": "ds_validation",
    }
    with patch(
        "services.datasets.service.get_storage_backend", return_value=fake_backend
    ):
        with pytest.raises(HTTPException) as exc_info:
            await ds._finalize_upload(
                "d1", "u1", data, metadata={"data_format": "jsonl"}
            )
    assert exc_info.value.status_code == 400
    ds.delete_dataset.assert_awaited_once_with(id="d1", user_id="u1")


# ---------------------------------------------------------------------------
# push_dataset — reload-409 fix (unfinalized placeholder reuse)
# ---------------------------------------------------------------------------


def _make_dataset_info(name="my-dataset", description="desc", user_id="u1"):
    """Return a minimal DatasetInfo as the create endpoint would receive."""
    return api.DatasetInfo(name=name, description=description, user_id=user_id)


def _db_for_push(
    existing_by_name=None,
    system_existing=None,
    check_exists=False,
    inserted=None,
):
    """Build a mock db whose push_dataset-relevant methods are wired up."""
    db = MagicMock()
    db.check_dataset_exists = AsyncMock(return_value=check_exists)
    # get_dataset_by_name_and_user is called twice: once for user_id, once for SYSTEM_USER.
    # Use side_effect list to return different values on successive calls.
    db.get_dataset_by_name_and_user = AsyncMock(
        side_effect=[existing_by_name, system_existing]
    )
    if inserted is not None:
        db.insert_dataset = AsyncMock(return_value=inserted)
    else:
        db.insert_dataset = AsyncMock()
    return db


async def test_push_dataset_placeholder_reuse_returns_existing_id():
    """
    Same-name row exists but train_file_size and train_records are NULL
    (unfinalized placeholder, e.g. after a page reload).
    push_dataset must return the existing id without raising and without
    calling insert_dataset.
    """
    _PLACEHOLDER_ID = "11111111-2222-3333-4444-555555555555"
    placeholder_row = {
        "id": _PLACEHOLDER_ID,
        "user_id": "u1",
        "name": "my-dataset",
        "description": "desc",
        "train_file_size": None,
        "train_records": None,
        "train_file": "my-dataset_train",
        "validation_file": "my-dataset_validation",
    }
    db = _db_for_push(existing_by_name=placeholder_row)
    ds = Dataset(db)
    result = await ds.push_dataset(_make_dataset_info())

    assert str(result.id) == _PLACEHOLDER_ID
    db.insert_dataset.assert_not_awaited()


async def test_push_dataset_finalized_duplicate_raises_409():
    """
    Same-name row exists AND train_file_size is populated (finalized).
    push_dataset must raise HTTP 409 — genuine duplicate name.
    """
    finalized_row = {
        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "user_id": "u1",
        "name": "my-dataset",
        "description": "desc",
        "train_file_size": 12345,
        "train_records": 100,
    }
    db = _db_for_push(existing_by_name=finalized_row)
    ds = Dataset(db)
    with pytest.raises(HTTPException) as exc_info:
        await ds.push_dataset(_make_dataset_info())
    assert exc_info.value.status_code == 409
    db.insert_dataset.assert_not_awaited()


async def test_push_dataset_no_existing_calls_insert():
    """
    No same-name row exists — normal create path calls insert_dataset.
    """
    new_info = _make_dataset_info()
    new_info.id = None  # will be set by insert_dataset

    # Simulate insert_dataset setting .id on the passed object and returning it.
    async def _insert(dataset):
        import uuid

        dataset.id = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        return dataset

    db = MagicMock()
    db.check_dataset_exists = AsyncMock(return_value=False)
    # First call (user lookup) -> None, second call (system lookup) -> None.
    db.get_dataset_by_name_and_user = AsyncMock(side_effect=[None, None])
    db.insert_dataset = AsyncMock(side_effect=_insert)

    ds = Dataset(db)
    result = await ds.push_dataset(new_info)

    db.insert_dataset.assert_awaited_once()
    assert result.id is not None


async def test_push_dataset_system_reserved_name_raises_409():
    """
    Name collides with a SYSTEM_USER dataset — must raise HTTP 409 regardless of
    whether a user-owned row also exists (no user-owned row in this test).
    """
    system_row = {
        "id": "sys-uuid",
        "user_id": "00000000-0000-0000-0000-000000000001",
        "name": "reserved-dataset",
        "description": "system",
        "train_file_size": None,
        "train_records": None,
    }
    db = MagicMock()
    db.check_dataset_exists = AsyncMock(return_value=False)
    # First call (user lookup) -> None (no user-owned row), second (system) -> system_row.
    db.get_dataset_by_name_and_user = AsyncMock(side_effect=[None, system_row])
    db.insert_dataset = AsyncMock()

    ds = Dataset(db)
    with pytest.raises(HTTPException) as exc_info:
        await ds.push_dataset(_make_dataset_info(name="reserved-dataset"))
    assert exc_info.value.status_code == 409
    assert "reserved" in exc_info.value.detail.lower()
    db.insert_dataset.assert_not_awaited()
