"""Tests for autotune.template_utils.lakehouse_path_to_uri."""

import pytest

from autotune.template_utils import lakehouse_path_to_uri


def test_lakehouse_path_resolves_to_lh_uri():
    path = (
        "/gb-lakehouse-prod-read-only/filesets/granite_dot_build/public/shared/"
        "climate/20250906T064534/climate_train.jsonl"
    )
    uri, name = lakehouse_path_to_uri(path)
    assert uri == "lh://prod/granite_dot_build.public/filesets/fileset_shared/climate/20250906T064534"
    assert name == "climate"


def test_hf_cache_path_resolves_to_hf_uri():
    path = (
        "/gb-read-write/hfcache/ibm-research/finance-test/"
        "d4efcbfc84d255b5e9e36393461db93bb6b5894debd71868b807f1d21e10c987/finance_train.jsonl"
    )
    uri, name = lakehouse_path_to_uri(path)
    assert uri == "hf:///datasets/ibm-research/finance-test"
    assert name == "finance-test"


def test_unmatched_path_raises_value_error():
    with pytest.raises(ValueError, match="does not match expected"):
        lakehouse_path_to_uri("/some/random/path/train.jsonl")


def test_resolve_dataset_uri_local_path_falls_back():
    from autotune.template_utils import resolve_dataset_uri

    uri, name = resolve_dataset_uri("datasets/finance_train.jsonl")
    assert uri is None
    assert name == "finance_train"


def test_resolve_dataset_uri_hf_cache_still_resolves():
    from autotune.template_utils import resolve_dataset_uri

    path = "/gb-read-write/hfcache/ibm-research/finance-test/abc123/finance_train.jsonl"
    uri, name = resolve_dataset_uri(path)
    assert uri == "hf:///datasets/ibm-research/finance-test"
    assert name == "finance-test"
