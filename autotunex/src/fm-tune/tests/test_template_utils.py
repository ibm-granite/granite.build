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


@pytest.mark.parametrize(
    "path, uri, name",
    [
        # an app-pushed <name>_<id8> repo, file at the revision root: the suffix is
        # part of the repo name, so it stays in the URI
        (
            "/Users/u/.cache/gbserver/hf/ibm-research/finance-classification_9bb5cc58/main/"
            "finance-classification_validation.jsonl",
            "hf:///datasets/ibm-research/finance-classification_9bb5cc58",
            "finance-classification_9bb5cc58",
        ),
        # a repo name ending in hex-like digits is not truncated
        (
            "/Users/u/.cache/gbserver/hf/acme/news_20240101/main/data/train.parquet",
            "hf:///datasets/acme/news_20240101",
            "news_20240101",
        ),
        # gbserver's own <owner>/<repo>/<revision> layout, file in a subdir
        (
            "/Users/u/.cache/gbserver/hf/axolotl-ai-co/alpaca_100_test/main/data/train-00000-of-00001.parquet",
            "hf:///datasets/axolotl-ai-co/alpaca_100_test",
            "alpaca_100_test",
        ),
        # SkyPilot/Slurm: the default <shared_workdir>/hf_cache, same layout
        (
            "/shared/hf_cache/axolotl-ai-co/alpaca_100_test/main/data/train.parquet",
            "hf:///datasets/axolotl-ai-co/alpaca_100_test",
            "alpaca_100_test",
        ),
        ("/shared/hf_cache/org/ds/main/train.jsonl", "hf:///datasets/org/ds", "ds"),
        # the cluster cache, file in a subdirectory of the hash dir
        ("/gb-read-write/hfcache/org/ds/abc123/data/train.parquet", "hf:///datasets/org/ds", "ds"),
    ],
)
def test_local_gbserver_hf_cache_path_resolves_to_hf_uri(path, uri, name):
    assert lakehouse_path_to_uri(path) == (uri, name)


@pytest.mark.parametrize(
    "path",
    [
        # hf_cache only as a whole path segment, and only with a revision dir
        "/shared/my_hf_cache/org/ds/main/train.jsonl",
        "/shared/hf_cache/org/train.jsonl",
    ],
)
def test_non_cache_paths_do_not_resolve(path):
    with pytest.raises(ValueError, match="does not match expected"):
        lakehouse_path_to_uri(path)


def _snapshot(root, rel):
    """A dir laid out like an HF download: it holds a .cache/huggingface folder."""
    path = root / rel
    (path / ".cache" / "huggingface").mkdir(parents=True)
    return str(path)


@pytest.mark.parametrize(
    "rel, name",
    [
        # bash: gbserver's local cache, <owner>/<repo>/<revision>
        (
            ".cache/gbserver/hf/hf-internal-testing/tiny-random-LlamaForCausalLM/main",
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
        ),
        # SkyPilot/Slurm: the default <shared_workdir>/hf_cache
        ("shared/hf_cache/ibm-granite/granite-4.0-h-350m/main", "ibm-granite/granite-4.0-h-350m"),
        # a pinned revision is dropped: owner/repo@rev is no valid HF id, and a job has
        # no revision field to carry it
        ("shared/hf_cache/ibm-granite/granite-4.0-h-350m/v2", "ibm-granite/granite-4.0-h-350m"),
    ],
)
def test_model_display_name_of_a_snapshot(tmp_path, rel, name):
    from autotune.template_utils import model_display_name

    path = _snapshot(tmp_path, rel)
    assert model_display_name(path) == name
    assert model_display_name(path + "/") == name


def test_model_display_name_keeps_a_checkpoint_under_hf_cache(tmp_path):
    # Same shape as a snapshot, but not an HF download: it must stay a path.
    from autotune.template_utils import model_display_name

    path = tmp_path / "data" / "hf_cache" / "runs" / "ckpt" / "final"
    path.mkdir(parents=True)
    assert model_display_name(str(path)) == str(path)


@pytest.mark.parametrize(
    "path",
    [
        # a plain HF id, a non-cache dir, and a dataset-style path deeper than the revision
        "ibm-granite/granite-4.0-h-350m",
        "/models/my-model",
        "/shared/hf_cache/o/r/main/sub",
        # hf_cache only as a whole path segment
        "/data/my_hf_cache/runs/ckpt/final",
    ],
)
def test_model_display_name_passes_other_values_through(path):
    from autotune.template_utils import model_display_name

    assert model_display_name(path) == path


@pytest.mark.parametrize(
    "path, fmt",
    [
        ("/c/data/train-00000-of-00001.parquet", "parquet"),
        ("finance_train.JSONL", "jsonl"),
        ("/c/train.csv", "csv"),
        # aliases mapped the way the main service's upload path maps them
        ("/c/train.pq", "parquet"),
        ("/c/train.json", "jsonl"),
        ("/c/train.txt", None),
        ("/c/train", None),
    ],
)
def test_data_format_from_path(path, fmt):
    from autotune.template_utils import data_format_from_path

    assert data_format_from_path(path) == fmt
