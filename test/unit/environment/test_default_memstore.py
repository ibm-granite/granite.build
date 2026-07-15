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

"""Verify mem:// is auto-registered on every environment, mirroring env://.

``Environment._register_default_memstore`` registers a default ``mem://``
(Memstore) asset store when the ``environment.yaml`` does not declare one, so
mem:// input/output work on all backends without a per-environment entry. These
tests construct bare environments (no config, hence no declared stores) and
assert the default store is present, dispatch resolves it, and a value
round-trips through the shared in-memory store.
"""

import asyncio

import pytest

from gbcommon.uri.uri import URI
from gbserver.environment.bash import Bash
from gbserver.environment.docker import Docker


@pytest.fixture
def bash_env():
    """A Bash environment with a dummy event queue and no declared stores."""
    return Bash(event_q=asyncio.Queue())


@pytest.fixture
def docker_env():
    """A Docker environment with a dummy event queue and no declared stores."""
    return Docker(event_q=asyncio.Queue())


def _mem_stores(env):
    """Return the registered asset stores whose base_uri is ``mem://``."""
    return [s for s in env.supported_assetstores if s.config.base_uri == "mem://"]


def test_default_memstore_registered(bash_env):
    """The bundled mem:// store is auto-registered even with no declaration."""
    stores = _mem_stores(bash_env)
    assert len(stores) == 1
    assert stores[0].type == "Memstore"


def test_default_memstore_registered_on_docker(docker_env):
    """Auto-registration lives on the base class, so all subclasses get it."""
    assert len(_mem_stores(docker_env)) == 1


def test_memstore_dispatch_available(bash_env):
    """The pull/push dispatch tables expose the inherited memstore handlers."""
    assert "memstore" in bash_env.pullasset_types
    assert "memstore" in bash_env.pushasset_types


def test_mem_uri_resolves_to_memstore(bash_env):
    """A mem:// URI resolves to the registered Memstore via _get_storeconfig."""
    uri = URI.get_uri("mem://rm-server")
    assetstore, config = bash_env._get_storeconfig(uri)
    assert assetstore is not None
    assert assetstore.type == "Memstore"
    assert config.store_uri == "mem://"


@pytest.mark.asyncio
async def test_pushasset_pullasset_memstore_roundtrip(bash_env):
    """A producer's verbatim state value round-trips through shared_mem_store.

    Values like a service URL must survive intact (no path normalisation), which
    is the whole reason mem:// exists alongside env://.
    """
    service_url = "http://host:8000"
    await bash_env.pushasset_memstore(
        binding={"state": service_url},
        binding_id="rm-server",
        uri="mem://rm-server",
    )
    assert bash_env.shared_mem_store["mem://rm-server"] == service_url

    binding_config, extra = await bash_env.pullasset_memstore(uri="mem://rm-server")
    assert binding_config["binding"]["state"] == service_url
    assert extra is None
