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
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner
from libgbtest.utils import AbstractSingletonStorageUsingTest

from gbserver.storage.singleton_storage import reset_admin_storage
from gbserver.storage.sql.engine_cache import get_singleton_engine_cache
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.storage_factory import StorageFactory

TEST_DATA_DIR = Path(__file__).parent.parent.parent.parent / "test-data"
STANDALONE_SPACE_DIR = TEST_DATA_DIR / "e2e" / "standalone" / "standalone-quickstart"


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestStandaloneCommand(AbstractSingletonStorageUsingTest):
    """Tests for the gbserver standalone CLI command.

    Based on AbstractSingletonStorageUsingTest so the standalone server registers its
    space into this test's own prefixed SQLite tables, never into shared ``gb_spaces``
    left cached by an earlier test. Follows test_build_runner_cli.py: SQLite forced,
    GB_HOME_DIR isolated to a temp dir, and the whole temp db discarded in teardown.
    """

    @classmethod
    def _get_storage_factory(cls) -> StorageFactory:
        """Force SQLite, whatever the session's default factory is."""
        return SqliteStorageFactory()

    def setup_method(self, method):
        """Isolate GB_HOME_DIR before the base class builds the prefixed storage."""
        self._prev_gb_home = os.environ.get("GB_HOME_DIR")
        self._gb_home = tempfile.mkdtemp(prefix="gb_standalone_cmd_")
        os.environ["GB_HOME_DIR"] = self._gb_home
        super().setup_method(method)

    def teardown_method(self, method):
        """Discard the isolated db and restore GB_HOME_DIR (not the base per-table drop,
        which can hang on a SQLite db the server thread wrote to)."""
        get_singleton_engine_cache().dispose_all()
        self.storage = None
        reset_admin_storage()
        if self._prev_gb_home is None:
            os.environ.pop("GB_HOME_DIR", None)
        else:
            os.environ["GB_HOME_DIR"] = self._prev_gb_home
        shutil.rmtree(self._gb_home, ignore_errors=True)

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
            from gbserver.commands.command_standalone import _run_standalone

            server_holder: dict = {}

            thread = threading.Thread(
                target=_run_standalone,
                kwargs={
                    "port": port,
                    "space_dir": str(STANDALONE_SPACE_DIR),
                    "on_started": on_started,
                    "on_server_created": lambda s: server_holder.__setitem__(
                        "server", s
                    ),
                },
                daemon=True,
                name="test-standalone-server",
            )
            thread.start()

            try:
                # Wait for server startup (up to 30 seconds)
                assert started_event.wait(
                    timeout=30
                ), "Standalone server did not start within 30 seconds"

                # Retry until uvicorn is fully accepting connections.
                last_err = None
                for _ in range(20):
                    try:
                        response = httpx.get(
                            f"http://127.0.0.1:{port}/api/v1", timeout=2
                        )
                        assert (
                            response.status_code == 200
                        ), f"Expected 200, got {response.status_code}: {response.text}"
                        data = response.json()
                        assert (
                            "message" in data
                        ), f"Response missing 'message' key: {data}"
                        last_err = None
                        break
                    except httpx.ConnectError as e:
                        last_err = e
                        time.sleep(0.25)
                if last_err is not None:
                    pytest.fail(
                        f"Could not connect to standalone server on port {port}: {last_err}"
                    )
            finally:
                # Stop the server so its BuildWatcher thread is stopped rather than
                # left polling global storage for the rest of the test session
                # (the daemon flag prevents a hang, but a live watcher could still
                # pick up builds submitted by later tests on this worker).
                server = server_holder.get("server")
                if server is not None:
                    server.should_exit = True
                thread.join(timeout=15)
                # Fail loudly if it didn't stop: teardown_method then resets the
                # singleton and restores GB_HOME_DIR, so a still-running BuildWatcher
                # would rebuild unprefixed storage in the real ~/.granite.build.
                assert (
                    not thread.is_alive()
                ), "standalone server thread did not stop within 15s"
