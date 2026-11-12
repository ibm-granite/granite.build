#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""Scheme-independent identity of a space/assets repo URI.

The same repo is written as ``https://host/o/r``, ``git+ssh://[git@]host/o/r.git@ref``,
``ssh://git@host:22/o/r.git`` or ``git@host:o/r.git`` depending on how it is
accessed. Identity and the stored form ignore that access detail.
"""

import os
import urllib.parse
from typing import Optional, Tuple

from gbcommon.uri.git import split_repo_path


def _git_identity(host: str, path: str) -> Tuple[str, str, str, str]:
    owner, repo = split_repo_path(path)
    return ("git", host.lower(), owner.lower(), repo.lower())


def repo_identity(uri: str) -> Tuple[str, ...]:
    """Return the identity tuple of a repo/space URI.

    * **Remote git** (``https`` / ``git+ssh`` / ``ssh`` / ``git``, and the scp
      form ``[user@]host:owner/repo``): ``("git", host, owner, repo)``,
      case-insensitive, ignoring scheme, userinfo, port, ``.git``, a pip-style
      ``@<ref>`` and a ``#subdirectory=`` fragment.
    * **Local** (``file://`` or a bare path): ``("file", <normalized path>)``,
      case-sensitive.

    Parsing is total: malformed input yields empty components, not an error.
    """
    # scp-like git URL ([user@]host:owner/repo(.git)) has no "scheme://", so
    # urlparse would read it as a path: a ':' before any '/' with a relative tail.
    if "://" not in uri:
        head, sep, tail = uri.partition(":")
        if sep and "/" not in head and tail and not tail.startswith("/"):
            return _git_identity(head.rsplit("@", 1)[-1], tail)
    parsed = urllib.parse.urlparse(uri, scheme="file")
    if parsed.scheme in ("file", ""):
        # netloc + path mirrors gbcommon.uri.space._file_uri_to_path.
        path = (parsed.netloc or "") + (parsed.path or "")
        return ("file", os.path.normpath(path) if path else "")
    return _git_identity(parsed.hostname or "", parsed.path)


def same_repo(uri_a: str, uri_b: str) -> bool:
    """Return whether two URIs identify the same repo/location."""
    return repo_identity(uri_a) == repo_identity(uri_b)


def normalize_repo_uri(uri: str) -> str:
    """Return the canonical stored form of a repo URI.

    Remote git repos become ``https://host/owner/repo`` (the form gb_spaces
    ``git_repo_uri`` holds); local paths become ``file://<abs path>``.

    Raises:
        ValueError: The URI does not name a host/owner/repo or a local path.
    """
    ident = repo_identity(uri)
    if ident[0] == "file":
        if not ident[1]:
            raise ValueError(f"not a repo URI: {uri!r}")
        return f"file://{os.path.abspath(ident[1])}"
    _, host, _, _ = ident
    # Keep the original case of owner/repo; identity lower-cases them.
    if "://" not in uri:
        path = uri.partition(":")[2]
    else:
        path = urllib.parse.urlparse(uri).path
    owner, repo = split_repo_path(path)
    if not (host and owner and repo):
        raise ValueError(f"not a host/owner/repo URI: {uri!r}")
    return f"https://{host}/{owner}/{repo}"


def to_config_uri(source_uri: str, ref: Optional[str]) -> str:
    """The git+ssh config URI a git-backed space resolves to, without GitHub.

    ``https://host/o/r`` + ``gbspace-config`` -> ``git+ssh://host/o/r.git@gbspace-config``,
    matching what GitURI.get_gb_space_config_uri returns after its branch probe.
    Local sources are returned as ``file://``.
    """
    normalized = normalize_repo_uri(source_uri)
    if normalized.startswith("file://"):
        return normalized
    parsed = urllib.parse.urlparse(normalized)
    path = f"{parsed.path}.git" + (f"@{ref}" if ref else "")
    return f"git+ssh://{parsed.netloc}{path}"
