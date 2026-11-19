# SPDX-License-Identifier: Apache-2.0

"""Where bash, docker and skypilot pull an hf:// input under the cache dir.

A model on huggingface.co keeps the ``<owner>/<repo>/<revision>`` layout, so
existing caches stay valid. Other types and other hosts get a directory of
their own, so a model never shares one with a dataset, space or bucket of the
same name. Paths of one repo or bucket share its directory, where each lands at
its own path (#715).
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
        ("hf:///buckets/org/b/sub/x", "buckets/org/b"),
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
        "other-host",
        "other-host-port",
    ],
)
def test_hf_pull_dest(uri, dest):
    assert hf_pull_dest(_CACHE, HfURI.parse(uri)) == _CACHE / dest


def test_hf_pull_dest_keeps_types_and_hosts_apart():
    """One owner/repo as a model, dataset, space, bucket and on another host: no
    two share a pull dir or nest in another's."""
    uris = [
        "hf:///org/r/main",
        "hf:///datasets/org/r/main",
        "hf:///spaces/org/r/main",
        "hf://hf.example.com/org/r/main",
        "hf:///buckets/org/r",
    ]
    dests = [hf_pull_dest(_CACHE, HfURI.parse(uri)) for uri in uris]
    assert len(set(dests)) == len(dests)
    for a in dests:
        for b in dests:
            assert a == b or not b.is_relative_to(a), (a, b)


@pytest.mark.parametrize(
    "uris",
    [
        ("hf:///buckets/org/b", "hf:///buckets/org/b/a", "hf:///buckets/org/b/a/x"),
        ("hf:///datasets/org/r/main/a", "hf:///datasets/org/r/main/b"),
    ],
    ids=["bucket", "repo"],
)
def test_paths_of_one_repo_share_its_pull_dir(uris):
    """Every path of a bucket or repo pulls into one dir, behind one lock, and
    is bound at its own path there, so nested paths never nest pull dirs."""
    hfuris = [HfURI.parse(uri) for uri in uris]
    dests = {hf_pull_dest(_CACHE, hfuri) for hfuri in hfuris}
    assert len(dests) == 1
    (dest,) = dests
    bound = [hfuri.local_path(dest) for hfuri in hfuris]
    assert len(set(bound)) == len(bound)
    assert all(path.is_relative_to(dest) for path in bound)
