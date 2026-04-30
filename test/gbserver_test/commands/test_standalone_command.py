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

"""Tests for the gbserver standalone command."""

import os
import socket
import threading
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner

SAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "samples"
STANDALONE_SPACE_DIR = SAMPLES_DIR / "standalone" / "standalone-quickstart"


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestStandaloneCommand:
    """Tests for the gbserver standalone CLI command."""

    def test_command_is_discoverable(self):
        """Verify the standalone command file is auto-discovered by the CLI."""
        from gbserver.cli import GraniteBuildServerCLI

        cli_group = GraniteBuildServerCLI(name="gbserver")
        commands = cli_group.list_commands(ctx=None)
        assert (
            "standalone" in commands
        ), f"'standalone' not found in discovered commands: {commands}"

    def test_standalone_starts_and_serves_api(self):
        """Start the standalone server in a background thread and verify the REST API responds."""
        # Environment vars MUST be set before importing _run_standalone
        # because root_api imports trigger storage initialization at module level.
        env = {
            "GBSERVER_METADATA_STORAGE": "sqlite",
            "GBSERVER_DEFAULT_BUILDRUNNER_TYPE": "thread",
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "",
        }

        port = _find_free_port()
        started_event = threading.Event()

        def on_started():
            started_event.set()

        with patch.dict(os.environ, env):
            # Reset singleton storage so it picks up the sqlite backend from env.
            from gbserver.storage import singleton_storage
            from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

            singleton_storage.set_storage_factory(SqliteStorageFactory())

            from gbserver.commands.command_standalone import _run_standalone

            thread = threading.Thread(
                target=_run_standalone,
                kwargs={
                    "port": port,
                    "space_dir": str(STANDALONE_SPACE_DIR),
                    "on_started": on_started,
                },
                daemon=True,
                name="test-standalone-server",
            )
            thread.start()

            # Wait for server startup (up to 30 seconds)
            assert started_event.wait(
                timeout=30
            ), "Standalone server did not start within 30 seconds"

            # Retry until uvicorn is fully accepting connections.
            last_err = None
            for _ in range(20):
                try:
                    response = httpx.get(f"http://127.0.0.1:{port}/api/v1", timeout=2)
                    assert (
                        response.status_code == 200
                    ), f"Expected 200, got {response.status_code}: {response.text}"
                    data = response.json()
                    assert "message" in data, f"Response missing 'message' key: {data}"
                    last_err = None
                    break
                except httpx.ConnectError as e:
                    last_err = e
                    time.sleep(0.25)
            if last_err is not None:
                pytest.fail(
                    f"Could not connect to standalone server on port {port}: {last_err}"
                )
