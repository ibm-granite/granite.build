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


def lakehouse_path_to_uri(path: str, strip_last_n: int = 1) -> tuple[str, str]:
    """Convert a local lakehouse mount path or HF cache path to an lh:// / hf:// URI.

    Args:
        path: Absolute path like
            /gb-lakehouse-prod-read-only/filesets/granite_dot_build/public/shared/climate/20250906T064534/climate_train.jsonl
            or an HF datasets cache path like
            /gb-read-write/hfcache/ibm-research/finance-test/<hash>/finance_train.jsonl
            or a local gb server HF cache path (e.g. when running gbserver on a laptop) like
            ~/.cache/gbserver/hf/ibm-research/finance-classification_9bb5cc58/main/finance-classification_validation.jsonl
        strip_last_n: Number of trailing path segments to remove (default 1 removes the filename).
            Only applies to the lakehouse path format; ignored for HF cache paths (prod or local
            gbserver), which have a fixed depth.

    Returns:
        Tuple of (uri, name).
        Lakehouse example:
            ("lh://prod/granite_dot_build.public/filesets/fileset_shared/climate/20250906T064534", "climate")
        HF cache example:
            ("hf:///datasets/ibm-research/finance-test", "finance-test")
    """
    path = path.strip().rstrip("/")

    match = re.match(
        r"^/gb-lakehouse-(\w+)-read-only/filesets/([^/]+)/([^/]+)/shared/(.+)$",
        path,
    )
    if match:
        env, namespace, scope, rest = match.groups()

        parts = rest.split("/")
        if strip_last_n > 0:
            parts = parts[:-strip_last_n] if strip_last_n < len(parts) else []
        if not parts:
            raise ValueError(f"Nothing left after stripping {strip_last_n} segment(s) from: {rest}")

        name = parts[0]
        remaining = "/".join(parts)
        return f"lh://{env}/{namespace}.{scope}/filesets/fileset_shared/{remaining}", name

    hf_match = _HF_CACHE_RE.match(path)
    if hf_match:
        org, dataset = hf_match.groups()
        return f"hf:///datasets/{org}/{dataset}", dataset

    local_hf_match = _LOCAL_GBSERVER_HF_CACHE_RE.match(path)
    if local_hf_match:
        org, dataset = local_hf_match.groups()
        return f"hf:///datasets/{org}/{dataset}", dataset

    raise ValueError(f"Path does not match expected lakehouse or HF cache format: {path}")


def stem_from_path(path: str) -> str:
    """Return the filename without extension from a path."""
    return os.path.splitext(os.path.basename(path))[0]


def resolve_dataset_uri(path: str) -> tuple[str | None, str]:
    """Best-effort ``(uri, name)`` for a train/eval file path.

    Lakehouse mount paths and HF-cache paths convert to ``lh://`` / ``hf://``
    URIs via :func:`lakehouse_path_to_uri`. Any other path (a plain local file
    such as ``datasets/finance_train.jsonl``, common for local/MPS runs) has no
    such URI, so return ``(None, <filename stem>)`` instead of raising. These
    values feed only the optional AutoTuneX bridge (off by default) and info
    logs, so a plain local path must never crash the run.
    """
    try:
        return lakehouse_path_to_uri(path)
    except ValueError:
        return None, stem_from_path(path)
