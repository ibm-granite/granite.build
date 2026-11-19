# SPDX-License-Identifier: Apache-2.0

"""Where bash, docker and skypilot pull an hf:// input under the cache dir.

A model on huggingface.co keeps the ``<owner>/<repo>/<revision>`` layout, so
existing caches stay valid. Other types and other hosts get a directory of
their own, so a model never shares one with a dataset, space or bucket of the
same name. Each bucket input gets its own directory, named by its path, so no
two bucket pull dirs nest (#715).
"""

import hashlib
from pathlib import Path

import pytest

from gbcommon.uri.hf import HfURI

_CACHE = Path("/cache")


def _leaf(path_in_bucket: str) -> str:
    """The directory a bucket input is pulled into, under its bucket."""
    return hashlib.sha256(path_in_bucket.encode()).hexdigest()[:16]


@pytest.mark.parametrize(
    ("uri", "dest"),
    [
        ("hf:///org/r", "org/r/main"),
        ("hf://huggingface.co/models/org/r/v1", "org/r/v1"),
        ("hf:///org/r/main/config.json", "org/r/main"),
        ("hf:///datasets/org/r/main", "datasets/org/r/main"),
        ("hf:///spaces/org/r/main/app.py", "spaces/org/r/main"),
        ("hf:///buckets/org/b", f"buckets/org/b/{_leaf('')}"),
        ("hf:///buckets/org/b/sub/x", f"buckets/org/b/{_leaf('sub/x')}"),
        ("hf:///buckets/org/b//sub/x/", f"buckets/org/b/{_leaf('sub/x')}"),
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
        "bucket-path-slashes",
        "other-host",
        "other-host-port",
    ],
)
def test_pull_dest(uri, dest):
    assert HfURI.parse(uri).pull_dest(_CACHE) == _CACHE / dest


def test_pull_dest_keeps_types_hosts_and_bucket_inputs_apart():
    """One owner/repo as a model, dataset, space, on another host, and as a
    whole bucket and two nested paths of it: no two share a pull dir or nest in
    another's, so each pull has its own lock and its own files."""
    uris = [
        "hf:///org/r/main",
        "hf:///datasets/org/r/main",
        "hf:///spaces/org/r/main",
        "hf://hf.example.com/org/r/main",
        "hf:///buckets/org/r",
        "hf:///buckets/org/r/a",
        "hf:///buckets/org/r/a/b",
    ]
    dests = [HfURI.parse(uri).pull_dest(_CACHE) for uri in uris]
    assert len(set(dests)) == len(dests)
    for a in dests:
        for b in dests:
            assert a == b or not b.is_relative_to(a), (a, b)


def test_paths_of_one_repo_share_its_pull_dir():
    """Every path of one repo revision pulls into its dir, behind one lock, and
    is bound at its own path there."""
    hfuris = [
        HfURI.parse(uri)
        for uri in ("hf:///datasets/org/r/main/a", "hf:///datasets/org/r/main/b")
    ]
    dests = {hfuri.pull_dest(_CACHE) for hfuri in hfuris}
    assert dests == {_CACHE / "datasets" / "org" / "r" / "main"}
    (dest,) = dests
    assert [hfuri.local_path(dest) for hfuri in hfuris] == [dest / "a", dest / "b"]
