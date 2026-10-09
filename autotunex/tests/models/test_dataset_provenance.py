# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Dataset HF provenance fields and the new importing status."""

from __future__ import annotations

from autotunex.models.status import DatasetStatus


def test_importing_is_a_dataset_status() -> None:
    assert DatasetStatus.IMPORTING.value == "importing"


def test_dataset_to_read_carries_hf_provenance() -> None:
    """``dataset_to_read`` is the only construction site for ``DatasetRead``.

    A field missing from its call omits it from every read (it would still
    default to ``None`` on the schema, so this must check actual values, not
    just field presence).
    """
    from datetime import UTC, datetime
    from uuid import uuid4

    from autotunex.db.tables.datasets import DatasetTable
    from autotunex.services.mappers import dataset_to_read

    now = datetime.now(UTC)
    dataset = DatasetTable(
        id=uuid4(),
        user_id=str(uuid4()),
        name="ds",
        train_file="ds_train",
        validation_file="ds_validation",
        data_format="parquet",
        status=DatasetStatus.READY,
        hf_repo_id="o/r",
        hf_revision="abc123",
        hf_config="default",
        hf_split="train",
        hf_provenance={"train_retained_rows": 5},
        created_at=now,
        updated_at=now,
    )

    read = dataset_to_read(dataset, [])

    assert read.hf_repo_id == "o/r"
    assert read.hf_revision == "abc123"
    assert read.hf_config == "default"
    assert read.hf_split == "train"
    assert read.hf_provenance == {"train_retained_rows": 5}


def test_dataset_table_has_provenance_columns() -> None:
    from autotunex.db.tables.datasets import DatasetTable

    columns = set(DatasetTable.__table__.columns.keys())
    assert {"hf_repo_id", "hf_revision", "hf_config", "hf_split", "hf_provenance"} <= columns
