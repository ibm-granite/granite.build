# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

# api/tests/test_tus_metadata.py
import pytest
from services.datasets.tus_metadata import parse_upload_metadata, UploadIntent


def test_autosplit_intent():
    md = {
        "dataset_id": "d1",
        "filename": "train.jsonl",
        "role": "source",
        "expects": "source",
        "train_set_percentage": "80",
    }
    intent = parse_upload_metadata(md)
    assert isinstance(intent, UploadIntent)
    assert intent.dataset_id == "d1"
    assert intent.role == "source"
    assert intent.expects == ["source"]
    assert intent.train_set_percentage == 80
    assert intent.column_mapping is None


def test_custom_validation_intent_with_mapping():
    md = {
        "dataset_id": "d1",
        "filename": "train.jsonl",
        "role": "train",
        "expects": "train,validation",
        "column_mapping": '{"in": "input"}',
    }
    intent = parse_upload_metadata(md)
    assert intent.role == "train"
    assert intent.expects == ["train", "validation"]
    assert intent.train_set_percentage is None
    assert intent.column_mapping == {"in": "input"}


def test_missing_required_field_raises():
    with pytest.raises(ValueError):
        parse_upload_metadata(
            {"filename": "t.jsonl", "role": "source", "expects": "source"}
        )


def test_bad_role_raises():
    with pytest.raises(ValueError):
        parse_upload_metadata(
            {
                "dataset_id": "d1",
                "filename": "t.jsonl",
                "role": "bogus",
                "expects": "source",
            }
        )


def test_malformed_column_mapping_raises():
    with pytest.raises(ValueError):
        parse_upload_metadata(
            {
                "dataset_id": "d1",
                "filename": "t.jsonl",
                "role": "train",
                "expects": "train,validation",
                "column_mapping": "{not json",
            }
        )


def test_column_mapping_non_object_raises():
    with pytest.raises(ValueError):
        parse_upload_metadata(
            {
                "dataset_id": "d1",
                "filename": "t.jsonl",
                "role": "train",
                "expects": "train,validation",
                "column_mapping": "[1, 2, 3]",
            }
        )


def test_role_not_in_expects_raises():
    with pytest.raises(ValueError):
        parse_upload_metadata(
            {
                "dataset_id": "d1",
                "filename": "t.jsonl",
                "role": "train",
                "expects": "validation",
            }
        )
