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

"""Verify file: is auto-registered on environments that can transfer files.

``Environment._register_default_filestore`` registers the bundled ``file:``
(Filestore) asset store when the ``environment.yaml`` does not declare one, so
file: input/output work without a per-environment entry — but only on the
environment classes that implement file transfer (``pullasset_filestore`` /
``pushasset_filestore``), i.e. ``Bash`` and ``Docker``. Classes without file
transfer (e.g. ``Runpod``) must **not** register it, so they don't falsely
advertise file: support. These tests construct bare environments (no config,
hence no declared stores) and assert the guard behaves accordingly.
"""

import asyncio

import pytest

from gbcommon.uri.uri import URI
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


def _file_stores(env):
    """Return the registered asset stores whose base_uri is ``file:``."""
    return [
        s
        for s in env.supported_assetstores
        if s.config.base_uri and s.config.base_uri.startswith("file:")
    ]


def test_default_filestore_registered_on_bash(bash_env):
    """The bundled file: store is auto-registered even with no declaration."""
    stores = _file_stores(bash_env)
    assert len(stores) == 1
    assert stores[0].type == "Filestore"


def test_default_filestore_registered_on_docker(docker_env):
    """Docker implements file push (bind-mounts inputs), so it registers too."""
    stores = _file_stores(docker_env)
    assert len(stores) == 1
    assert stores[0].type == "Filestore"


def test_default_filestore_not_registered_without_transfer(runpod_env):
    """A backend without pullasset_filestore must not advertise file: support."""
    assert "filestore" not in runpod_env.pullasset_types
    assert _file_stores(runpod_env) == []


def test_filestore_dispatch_available(bash_env):
    """The pull/push dispatch tables expose the filestore handlers."""
    assert "filestore" in bash_env.pullasset_types
    assert "filestore" in bash_env.pushasset_types


def test_file_uri_resolves_to_filestore(bash_env):
    """A file: URI resolves to the registered Filestore via _get_storeconfig."""
    uri = URI.get_uri("file:///tmp/artifact")
    assetstore, config = bash_env._get_storeconfig(uri)
    assert assetstore is not None
    assert assetstore.type == "Filestore"
    assert config.store_uri == "file:"


@pytest.mark.asyncio
async def test_pushasset_pullasset_filestore_roundtrip(bash_env, tmp_path):
    """A file artifact round-trips through the auto-registered bash file store.

    This is the end-to-end proof that removing the declared ``local`` store and
    relying on the implicit builtin file store still moves bytes: push copies a
    source file to a file: destination, and a subsequent pull binds that path.
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
