# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

_HF_CACHE_RE = re.compile(r"^/gb-read-write/hfcache/([^/]+)/([^/]+)/[^/]+/[^/]+$")
# Local gb server cache, e.g. ~/.cache/gbserver/hf/{org}/{dataset}_{hash}/{revision}/{filename} —
# the dataset segment bakes a hex hash suffix into the name.
_LOCAL_GBSERVER_HF_CACHE_RE = re.compile(r"^.*/\.cache/gbserver/hf/([^/]+)/([^/]+)_[0-9a-f]{6,}/[^/]+/[^/]+$")


def hf_cache_path_to_uri(path: str) -> tuple[str, str]:
    """Convert an HF datasets cache path to an hf:// URI.

    Args:
        path: An HF datasets cache path like
            /gb-read-write/hfcache/example-org/example-dataset/<hash>/example_train.jsonl
            or a local gb server HF cache path (e.g. when running gbserver on a laptop) like
            ~/.cache/gbserver/hf/example-org/example-dataset_9bb5cc58/main/example_validation.jsonl

    Returns:
        Tuple of (uri, name), e.g.
            ("hf:///datasets/example-org/example-dataset", "example-dataset")
    """
    path = path.strip().rstrip("/")

    hf_match = _HF_CACHE_RE.match(path)
    if hf_match:
        org, dataset = hf_match.groups()
        return f"hf:///datasets/{org}/{dataset}", dataset

    local_hf_match = _LOCAL_GBSERVER_HF_CACHE_RE.match(path)
    if local_hf_match:
        org, dataset = local_hf_match.groups()
        return f"hf:///datasets/{org}/{dataset}", dataset

    raise ValueError(f"Path does not match expected HF cache format: {path}")


def stem_from_path(path: str) -> str:
    """Return the filename without extension from a path."""
    return os.path.splitext(os.path.basename(path))[0]


def resolve_dataset_uri(path: str) -> tuple[str | None, str]:
    """Best-effort ``(uri, name)`` for a train/eval file path.

    HF-cache paths convert to ``hf://`` URIs via :func:`hf_cache_path_to_uri`.
    Any other path (a plain local file such as ``data/train.jsonl``, common
    for local/MPS runs) has no such URI, so return ``(None, <filename stem>)``
    instead of raising. These values feed only the optional AutoTuneX bridge
    (off by default) and info logs, so a plain local path must never crash the
    run.
    """
    try:
        return hf_cache_path_to_uri(path)
    except ValueError:
        return None, stem_from_path(path)
