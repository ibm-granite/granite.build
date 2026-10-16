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

"""Shared helper for materializing SSH private keys to disk.

Used by both the native LSF environment (per-launch temp key file) and the
inline SkyPilot config materializer (stable, content-addressed key file). The
path policy differs per caller; the secure write (atomic, ``0600``, normalized
trailing newline) is shared here so key handling stays consistent.
"""

import os
from pathlib import Path


def write_private_key_file(key_contents: str, dest: Path) -> Path:
    """Write SSH private-key contents to ``dest`` with secure permissions.

    The parent directory is created if missing (best-effort ``0700``), and the
    key is written atomically (``*.gbtmp`` + ``os.replace``) with ``0600``
    permissions and a single normalized trailing newline. Key contents are never
    logged.

    :param key_contents: The private key material (e.g. PEM text).
    :param dest: Destination path for the key file.
    :returns: ``dest``.
    :raises ValueError: If ``key_contents`` is empty or whitespace-only.
    """
    if not key_contents or not key_contents.strip():
        raise ValueError("write_private_key_file: empty key_contents")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(dest.parent, 0o700)
    except OSError:
        # Best-effort: a pre-existing dir we don't own should not abort the write.
        pass
    data = key_contents.rstrip("\n") + "\n"
    tmp = dest.with_name(dest.name + ".gbtmp")
    tmp.write_text(data, encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, dest)
    return dest
