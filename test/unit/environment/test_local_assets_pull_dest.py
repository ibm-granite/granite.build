# SPDX-License-Identifier: Apache-2.0

"""Where bash, docker and skypilot pull an hf:// input under the cache dir.

A model on huggingface.co keeps the ``<owner>/<repo>/<revision>`` layout, so
existing caches stay valid. Other types, other hosts and paths in a bucket each
get a directory of their own, so two inputs share one only when they name the
same content (#715).
"""

from pathlib import Path

import pytest

from gbcommon.uri.hf import HfURI
from gbserver.environment.local_assets import hf_pull_dest

_CACHE = Path("/cache")


@pytest.mark.parametrize(
    ("uri", "dest"),
    [
        ("hf:///org/r", "org/r/main"),
        ("hf://huggingface.co/models/org/r/v1", "org/r/v1"),
        ("hf:///org/r/main/config.json", "org/r/main"),
        ("hf:///datasets/org/r/main", "datasets/org/r/main"),
        ("hf:///spaces/org/r/main/app.py", "spaces/org/r/main"),
        ("hf:///buckets/org/b", "buckets/org/b"),
        ("hf:///buckets/org/b/sub/x", "buckets/org/b/sub/x"),
        ("hf:///buckets/org/b//sub", "buckets/org/b/sub"),
        ("hf://hf.example.com/org/r", "@hf.example.com/org/r/main"),
        (
            "hf://hf.example.com:8443/datasets/org/r/v1",
            "@hf.example.com_8443/datasets/org/r/v1",
        ),
    ],
    ids=[
        "model",
        "model-explicit-host-and-type",
        "model-path-in-repo",
        "dataset",
        "space",
        "bucket",
        "bucket-path",
        "bucket-leading-slash",
        "other-host",
        "other-host-port",
    ],
)
def test_hf_pull_dest(uri, dest):
    assert hf_pull_dest(_CACHE, HfURI.parse(uri)) == _CACHE / dest


def test_hf_pull_dest_keeps_types_hosts_and_bucket_paths_apart():
    """One owner/repo as a model, dataset, space, on another host and as two
    paths of a bucket: no two share a pull dir or nest in another's."""
    uris = [
        "hf:///org/r/main",
        "hf:///datasets/org/r/main",
        "hf:///spaces/org/r/main",
        "hf://hf.example.com/org/r/main",
        "hf:///buckets/org/r/a",
        "hf:///buckets/org/r/b",
    ]
    dests = [hf_pull_dest(_CACHE, HfURI.parse(uri)) for uri in uris]
    assert len(set(dests)) == len(dests)
    for a in dests:
        for b in dests:
            assert a == b or not b.is_relative_to(a), (a, b)
