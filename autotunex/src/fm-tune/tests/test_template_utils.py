"""Tests for autotune.template_utils.hf_cache_path_to_uri."""

import pytest

from autotune.template_utils import hf_cache_path_to_uri


def test_hf_cache_path_resolves_to_hf_uri():
    path = (
        "/gb-read-write/hfcache/example-org/example-dataset/"
        "d4efcbfc84d255b5e9e36393461db93bb6b5894debd71868b807f1d21e10c987/example_train.jsonl"
    )
    uri, name = hf_cache_path_to_uri(path)
    assert uri == "hf:///datasets/example-org/example-dataset"
    assert name == "example-dataset"


def test_local_gbserver_hf_cache_path_resolves_to_hf_uri():
    path = "/home/user/.cache/gbserver/hf/example-org/example-dataset_9bb5cc58/main/example_validation.jsonl"
    uri, name = hf_cache_path_to_uri(path)
    assert uri == "hf:///datasets/example-org/example-dataset"
    assert name == "example-dataset"


def test_unmatched_path_raises_value_error():
    with pytest.raises(ValueError, match="does not match expected"):
        hf_cache_path_to_uri("/some/random/path/train.jsonl")


def test_resolve_dataset_uri_local_path_falls_back():
    from autotune.template_utils import resolve_dataset_uri

    uri, name = resolve_dataset_uri("data/train.jsonl")
    assert uri is None
    assert name == "train"


def test_resolve_dataset_uri_hf_cache_still_resolves():
    from autotune.template_utils import resolve_dataset_uri

    path = "/gb-read-write/hfcache/example-org/example-dataset/abc123/example_train.jsonl"
    uri, name = resolve_dataset_uri(path)
    assert uri == "hf:///datasets/example-org/example-dataset"
    assert name == "example-dataset"
