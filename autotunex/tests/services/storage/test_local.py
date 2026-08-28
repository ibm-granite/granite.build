"""LocalStorageBackend against a real temp directory.

persist moves staged files into ``<root>/<id>/``; preview reads bounded rows;
delete idempotently removes the id directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from autotunex.services.storage.local import LocalStorageBackend


def _staged_jsonl(dir_: Path, name: str, rows: list[dict[str, int]]) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


async def test_persist_moves_files_and_returns_no_artifact_refs(tmp_path: Path) -> None:
    backend = LocalStorageBackend(root=tmp_path / "store")
    dataset_id = uuid4()
    staging = tmp_path / "staging"
    train = _staged_jsonl(staging, "train.jsonl", [{"i": 1}])

    artifact_id, artifact_url = await backend.persist(
        dataset_id=dataset_id, name="ds", data_format="jsonl", train=train, validation=None
    )

    assert artifact_id is None and artifact_url is None
    assert (tmp_path / "store" / str(dataset_id) / "ds_train.jsonl").exists()
    assert not train.exists()  # moved, not copied


async def test_persist_emits_dataset_dir_file_uri_when_enabled(tmp_path: Path) -> None:
    backend = LocalStorageBackend(root=tmp_path / "store", emit_file_uri=True)
    dataset_id = uuid4()
    staging = tmp_path / "staging"
    train = _staged_jsonl(staging, "train.jsonl", [{"i": 1}])

    artifact_id, artifact_url = await backend.persist(
        dataset_id=dataset_id, name="ds", data_format="jsonl", train=train, validation=None
    )

    expected = (tmp_path / "store" / str(dataset_id)).resolve().as_uri()
    assert artifact_id is None
    assert artifact_url == expected
    assert artifact_url.startswith("file:///")  # absolute URI, mountable by same-host gbserver
    assert (tmp_path / "store" / str(dataset_id) / "ds_train.jsonl").exists()


async def test_preview_returns_bounded_rows(tmp_path: Path) -> None:
    backend = LocalStorageBackend(root=tmp_path / "store")
    dataset_id = uuid4()
    staging = tmp_path / "staging"
    train = _staged_jsonl(staging, "train.jsonl", [{"i": n} for n in range(5)])
    await backend.persist(
        dataset_id=dataset_id, name="ds", data_format="jsonl", train=train, validation=None
    )

    preview = await backend.preview(
        dataset_id=dataset_id, name="ds", data_format="jsonl", artifact_url=None, rows=2
    )

    assert preview.train == [{"i": 0}, {"i": 1}]
    assert preview.validation == []


async def test_delete_is_idempotent(tmp_path: Path) -> None:
    backend = LocalStorageBackend(root=tmp_path / "store")
    dataset_id = uuid4()

    await backend.delete(dataset_id=dataset_id, name="ds", artifact_url=None)  # nothing there
    await backend.delete(dataset_id=dataset_id, name="ds", artifact_url=None)  # still fine


def test_path_rejects_escaping_name(tmp_path: Path) -> None:
    """Defense-in-depth: a traversal name must not resolve outside the dataset dir.

    Independent of the ``DatasetCreate`` Pydantic validator (Task 8) — this guard
    holds even if a name reaches ``_path`` by some other route.
    """
    backend = LocalStorageBackend(root=tmp_path)
    dataset_id = uuid4()

    with pytest.raises(ValueError):
        backend._path(dataset_id, "../../etc/passwd", "jsonl", split="train")


def test_path_stays_within_dataset_dir(tmp_path: Path) -> None:
    backend = LocalStorageBackend(root=tmp_path)
    dataset_id = uuid4()

    path = backend._path(dataset_id, "clean", "jsonl", split="train")

    assert path.resolve().is_relative_to(backend._dir(dataset_id).resolve())
