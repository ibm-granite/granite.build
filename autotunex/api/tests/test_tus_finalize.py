# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

from services.datasets.tus_finalize import handle_completed_file
from services.datasets.tus_metadata import UploadIntent
from services.datasets.tus_rendezvous import UploadRendezvous


def _dataset():
    dataset = MagicMock()
    dataset.upload_and_split_dataset = AsyncMock(
        return_value={"id": "d1", "ok": "split"}
    )
    dataset.upload_datasets = AsyncMock(return_value={"id": "d1", "ok": "custom"})
    return dataset


def _touch(tmp_path, name):
    p = tmp_path / name
    p.write_text('{"input": "a", "output": "b"}\n')
    return str(p)


async def test_autosplit_finalizes_via_split(tmp_path):
    dataset = _dataset()
    rv = UploadRendezvous()
    src = _touch(tmp_path, "source.jsonl")
    intent = UploadIntent(
        dataset_id="d1",
        filename="source.jsonl",
        role="source",
        expects=["source"],
        train_set_percentage=80,
    )
    result = await handle_completed_file(src, intent, dataset, rv, user_id="u1")
    assert result == {"id": "d1", "ok": "split"}
    dataset.upload_and_split_dataset.assert_awaited_once()
    kw = dataset.upload_and_split_dataset.call_args.kwargs
    assert kw["dataset_id"] == "d1" and kw["user_id"] == "u1"
    assert kw["train_set_percentage"] == 80


async def test_custom_validation_last_file_finalizes(tmp_path):
    dataset = _dataset()
    rv = UploadRendezvous()
    train = _touch(tmp_path, "train.jsonl")
    val = _touch(tmp_path, "val.jsonl")
    expects = ["train", "validation"]
    train_intent = UploadIntent(
        dataset_id="d1",
        filename="train.jsonl",
        role="train",
        expects=expects,
    )
    val_intent = UploadIntent(
        dataset_id="d1",
        filename="val.jsonl",
        role="validation",
        expects=expects,
    )
    # First file: does NOT finalize.
    first = await handle_completed_file(train, train_intent, dataset, rv, user_id="u1")
    assert first is None
    dataset.upload_datasets.assert_not_awaited()
    # Second file: finalizes via upload_datasets.
    second = await handle_completed_file(val, val_intent, dataset, rv, user_id="u1")
    assert second == {"id": "d1", "ok": "custom"}
    dataset.upload_datasets.assert_awaited_once()
    assert dataset.upload_datasets.call_args.kwargs["user_id"] == "u1"


async def test_finalize_fires_exactly_once(tmp_path):
    dataset = _dataset()
    rv = UploadRendezvous()
    src = _touch(tmp_path, "source.jsonl")
    intent = UploadIntent(
        dataset_id="d1",
        filename="source.jsonl",
        role="source",
        expects=["source"],
        train_set_percentage=80,
    )
    await handle_completed_file(src, intent, dataset, rv, user_id="u1")
    # A duplicate completion (e.g. retried PATCH) must not finalize twice.
    await handle_completed_file(src, intent, dataset, rv, user_id="u1")
    assert dataset.upload_and_split_dataset.await_count == 1


async def test_missing_sibling_does_not_finalize(tmp_path):
    dataset = _dataset()
    rv = UploadRendezvous()
    train = _touch(tmp_path, "train.jsonl")
    intent = UploadIntent(
        dataset_id="d1",
        filename="train.jsonl",
        role="train",
        expects=["train", "validation"],
    )
    result = await handle_completed_file(train, intent, dataset, rv, user_id="u1")
    assert result is None
    dataset.upload_datasets.assert_not_awaited()
    dataset.upload_and_split_dataset.assert_not_awaited()


async def test_failed_finalize_is_retryable_then_succeeds(tmp_path):
    """A transient finalize failure releases the once-only claim so a tus-retried
    completion can re-attempt; a subsequent success finalizes."""
    dataset = _dataset()
    # First dispatch raises (transient error), then a retry succeeds.
    dataset.upload_and_split_dataset = AsyncMock(
        side_effect=[RuntimeError("boom"), {"id": "d1", "ok": "split"}]
    )
    rv = UploadRendezvous()
    src = _touch(tmp_path, "source.jsonl")
    intent = UploadIntent(
        dataset_id="d1",
        filename="source.jsonl",
        role="source",
        expects=["source"],
        train_set_percentage=80,
    )
    # First completion: dispatch raises and the claim is released.
    try:
        await handle_completed_file(src, intent, dataset, rv, user_id="u1")
        raise AssertionError("expected dispatch to raise")
    except RuntimeError:
        pass
    # Retry: group re-claimable, finalize re-attempts and succeeds.
    result = await handle_completed_file(src, intent, dataset, rv, user_id="u1")
    assert result == {"id": "d1", "ok": "split"}
    assert dataset.upload_and_split_dataset.await_count == 2


async def test_succeeding_finalize_is_once_only_not_retried(tmp_path):
    """After a SUCCESSFUL finalize, a duplicate completion no-ops (once-only)."""
    dataset = _dataset()
    rv = UploadRendezvous()
    src = _touch(tmp_path, "source.jsonl")
    intent = UploadIntent(
        dataset_id="d1",
        filename="source.jsonl",
        role="source",
        expects=["source"],
        train_set_percentage=80,
    )
    first = await handle_completed_file(src, intent, dataset, rv, user_id="u1")
    assert first == {"id": "d1", "ok": "split"}
    second = await handle_completed_file(src, intent, dataset, rv, user_id="u1")
    assert second is None
    assert dataset.upload_and_split_dataset.await_count == 1
