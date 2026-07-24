# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import pytest
from services.storage.base import (
    StorageBackend,
    DatasetRef,
    DatasetFiles,
    StorageLocator,
    StorageError,
    StorageNotFound,
    StorageValidationError,
)


def test_dataclasses_carry_expected_fields():
    ref = DatasetRef(dataset_id="d0", name="base", data_format="jsonl")
    assert (ref.dataset_id, ref.name, ref.data_format) == ("d0", "base", "jsonl")

    files = DatasetFiles(
        dataset_id="d1",
        name="ds",
        data_format="jsonl",
        local_dir="/tmp/d1/ds",
        train_file="ds_train.jsonl",
        validation_file="ds_validation.jsonl",
    )
    # inherited fields + own fields all present and constructable together
    assert files.dataset_id == "d1"
    assert files.name == "ds"
    assert files.data_format == "jsonl"
    assert files.local_dir == "/tmp/d1/ds"
    assert files.train_file == "ds_train.jsonl"
    assert files.validation_file == "ds_validation.jsonl"

    loc = StorageLocator(artifact_id="a1", artifact_url="s3://x")
    assert (loc.artifact_id, loc.artifact_url) == ("a1", "s3://x")


def test_error_hierarchy():
    assert issubclass(StorageNotFound, StorageError)
    assert issubclass(StorageValidationError, StorageError)


def test_cannot_instantiate_abstract_backend():
    with pytest.raises(TypeError):
        StorageBackend()  # abstract methods unimplemented
