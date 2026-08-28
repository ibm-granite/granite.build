"""Dataset request/response schemas.

Pins the two contract properties most likely to regress: ``DatasetCreate``
forbids server-owned fields (a client cannot set ``artifact_url`` or ``status``),
and ``DatasetRead`` round-trips from ORM attributes including the nested
``associated_jobs`` and optional ``preview``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from autotunex.models.dataset import (
    DatasetCreate,
    DatasetJobRef,
    DatasetPreview,
    DatasetRead,
)
from autotunex.models.status import DatasetStatus, RunStatus


def test_create_defaults_data_format_to_jsonl() -> None:
    body = DatasetCreate(name="ds", description="desc")

    assert body.data_format == "jsonl"


def test_create_forbids_unknown_and_server_owned_fields() -> None:
    with pytest.raises(ValidationError):
        DatasetCreate(name="ds", description="d", artifact_url="http://evil")  # type: ignore[call-arg]


def test_create_requires_a_non_empty_name() -> None:
    with pytest.raises(ValidationError):
        DatasetCreate(name="", description="d")


def test_create_treats_description_as_optional() -> None:
    # description maps to a nullable column: omitting it yields None (stored NULL).
    assert DatasetCreate(name="ds").description is None


def test_read_carries_status_jobs_and_optional_preview() -> None:
    now = datetime.now(UTC)
    job_ref = DatasetJobRef(id=uuid4(), experiment_name="exp", status=RunStatus.PENDING)

    read = DatasetRead(
        id=uuid4(),
        user_id="u",
        name="ds",
        description="d",
        data_format="jsonl",
        status=DatasetStatus.READY,
        status_detail=None,
        train_file="ds_train",
        train_records=10,
        train_file_size=1234,
        validation_file="ds_validation",
        validation_records=2,
        validation_file_size=99,
        artifact_id=None,
        artifact_url=None,
        associated_jobs=[job_ref],
        created_at=now,
        updated_at=now,
        preview=DatasetPreview(train=[{"a": 1}], validation=[]),
    )

    assert read.status is DatasetStatus.READY
    assert read.associated_jobs[0].experiment_name == "exp"
    assert read.preview is not None and read.preview.train == [{"a": 1}]


def test_read_preview_defaults_to_none() -> None:
    now = datetime.now(UTC)

    read = DatasetRead(
        id=uuid4(),
        user_id="u",
        name="ds",
        description="d",
        data_format="jsonl",
        status=DatasetStatus.EMPTY,
        train_file="ds_train",
        validation_file="ds_validation",
        created_at=now,
        updated_at=now,
    )

    assert read.preview is None
    assert read.associated_jobs == []


@pytest.mark.parametrize("bad", ["../evil", "a/b", "a\\b", "..", "foo/../bar"])
def test_dataset_create_rejects_path_separators_in_name(bad: str) -> None:
    with pytest.raises(ValidationError):
        DatasetCreate(name=bad)


def test_dataset_create_accepts_a_plain_name() -> None:
    assert DatasetCreate(name="my-dataset_v1").name == "my-dataset_v1"
