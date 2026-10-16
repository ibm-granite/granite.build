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

"""File store (``file:``) transfer capability per environment class.

The ``file:`` store is **not** auto-registered (unlike ``env://`` / ``mem://``):
an environment declares ``space://assetstores/file`` in its ``environment.yaml``
with only the modes it implements. These tests pin the per-class dispatch
capability that determines which modes are legal to declare — a backend must not
advertise a transfer direction it can't service:

  * ``Bash`` implements BOTH ``pullasset_filestore`` and ``pushasset_filestore``
    (may declare load + push).
  * ``Docker`` implements ``pushasset_filestore`` only, NO ``pullasset_filestore``
    (may declare push only — declaring load would fail at pull dispatch).
  * ``Runpod`` implements neither (must not declare the file store at all).

The shipped ``environment.yaml`` declarations that rely on this are asserted in
``test/unit/space/test_space_config.py``.
"""

import asyncio

import pytest

from gbserver.environment.bash import Bash
from gbserver.environment.docker import Docker
from gbserver.environment.runpod import Runpod


@pytest.fixture
def bash_env():
    """A Bash environment with a dummy event queue and no declared stores."""
    return Bash(event_q=asyncio.Queue())


@pytest.fixture
def docker_env():
    """A Docker environment with a dummy event queue and no declared stores."""
    return Docker(event_q=asyncio.Queue())


@pytest.fixture
def runpod_env():
    """A Runpod environment — a backend with no file transfer support."""
    return Runpod(event_q=asyncio.Queue())


def test_bash_supports_file_load_and_push(bash_env):
    """Bash implements both directions, so it may declare load + push."""
    assert "filestore" in bash_env.pullasset_types
    assert "filestore" in bash_env.pushasset_types


def test_docker_supports_file_push_only(docker_env):
    """Docker implements push but NOT pull — it may declare push only.

    Declaring ``load`` for docker would let a ``file:`` input resolve to the
    Filestore and then fail at pull dispatch ("assetstore type filestore is not
    supported"), which is exactly what the push-only declaration avoids.
    """
    assert "filestore" in docker_env.pushasset_types
    assert "filestore" not in docker_env.pullasset_types


def test_runpod_supports_no_file_transfer(runpod_env):
    """A backend with neither method must not declare the file store."""
    assert "filestore" not in runpod_env.pullasset_types
    assert "filestore" not in runpod_env.pushasset_types


@pytest.mark.asyncio
async def test_pushasset_pullasset_filestore_roundtrip(bash_env, tmp_path):
    """A file artifact round-trips through the bash file store methods directly.

    Exercises ``pushasset_filestore`` (copy source -> file: destination) and
    ``pullasset_filestore`` (bind the destination path) without relying on store
    registration.
    """
    src = tmp_path / "src.txt"
    src.write_text("hello file store")
    dest = tmp_path / "dest.txt"
    dest_uri = f"file://{dest}"

    await bash_env.pushasset_filestore(binding={"path": str(src)}, uri=dest_uri)
    assert dest.read_text() == "hello file store"

    binding_config, extra = await bash_env.pullasset_filestore(uri=dest_uri)
    assert binding_config["binding"]["path"] == str(dest)
    assert extra is None
