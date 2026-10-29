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

"""End-to-end test: standalone REST API server submit and query.

This test verifies that the standalone REST API server can:
1. Start in a background thread with SQLite storage and API key auth.
2. Accept a build submission via POST /api/v1/builds/.
3. Return the build status via GET /api/v1/builds/{id}.
"""

import base64
import io
import os
import random
import shutil
import socket
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from libgbtest.utils import AbstractSingletonStorageUsingTest

from gbserver.storage.singleton_storage import reset_admin_storage
from gbserver.storage.sql.engine_cache import get_singleton_engine_cache
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.storage_factory import StorageFactory

TEST_DATA_DIR = Path(__file__).parent.parent.parent.parent / "test-data"
STANDALONE_BUILD_DIR = TEST_DATA_DIR / "e2e" / "standalone" / "standalone-quickstart"


def _get_free_port() -> int:
    """Find a free TCP port on localhost."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_build_archive(build_dir: Path) -> str:
    """Create a base64-encoded zip archive from a build directory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in build_dir.rglob("*"):
            if path.is_file():
                zf.write(path, str(path.relative_to(build_dir)))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _shutdown_standalone_server(
    server_holder: dict, server_thread: threading.Thread, timeout: float = 15.0
) -> None:
    """Gracefully stop the in-thread standalone server and join its thread.

    Setting ``should_exit`` lets uvicorn's ``server.run()`` return so
    ``_run_standalone`` reaches its ``finally`` and calls ``build_watcher.stop()``.
    Without this the daemon BuildWatcher keeps polling for the rest of the test
    session — harmless on its own, but it floods the logs and can interfere with
    later tests on the same worker.

    Args:
        server_holder: dict populated by the ``on_server_created`` callback under
            the key ``"server"`` with the ``uvicorn.Server`` instance.
        server_thread: the thread running ``_run_standalone``.
        timeout: seconds to wait for the server thread to exit.

    Raises:
        AssertionError: if the thread is still alive after ``timeout``. Teardown then
            resets the singleton and restores GB_HOME_DIR, so a still-running
            BuildWatcher would rebuild unprefixed storage in the real
            ~/.granite.build; the test must fail rather than leave that behind.
    """
    server = server_holder.get("server")
    if server is not None:
        server.should_exit = True
    server_thread.join(timeout=timeout)
    assert (
        not server_thread.is_alive()
    ), f"standalone server thread did not stop within {timeout}s"


class TestStandaloneRestApiE2E(AbstractSingletonStorageUsingTest):
    """End-to-end tests for the standalone REST API server.

    Based on AbstractSingletonStorageUsingTest so each test gets its own prefixed
    SQLite admin tables, set up before the server starts and the singleton reset
    after. Follows test_build_runner_cli.py: SQLite forced, GB_HOME_DIR isolated to
    a temp dir, and the whole temp db discarded in teardown.
    """

    @classmethod
    def _get_storage_factory(cls) -> StorageFactory:
        """Force SQLite, whatever the session's default factory is."""
        return SqliteStorageFactory()

    def setup_method(self, method):
        """Isolate GB_HOME_DIR before the base class builds the prefixed storage."""
        self._prev_gb_home = os.environ.get("GB_HOME_DIR")
        self._gb_home = tempfile.mkdtemp(prefix="gb_standalone_rest_e2e_")
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

    def test_submit_and_query_build(self):
        """Start the standalone server, submit a build, and query its status."""
        from gbserver.commands.command_standalone import _run_standalone

        port = _get_free_port()
        api_key = f"test-key-{random.randint(1, 100000)}"

        env = {
            "GBSERVER_METADATA_STORAGE": "sqlite",
            "GBSERVER_DEFAULT_BUILDRUNNER_TYPE": "thread",
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": api_key,
        }

        started_event = threading.Event()

        def on_started():
            started_event.set()

        server_holder: dict = {}

        def run_server():
            _run_standalone(
                port=port,
                space_dir=str(STANDALONE_BUILD_DIR),
                on_started=on_started,
                on_server_created=lambda s: server_holder.__setitem__("server", s),
            )

        server_thread = threading.Thread(
            target=run_server,
            name="standalone-rest-api-e2e",
            daemon=True,
        )

        try:
            with patch.dict(os.environ, env):
                server_thread.start()

                # Wait for the server to finish starting up.
                assert started_event.wait(
                    timeout=30
                ), "Standalone server did not start within 30 seconds"
                # Retry until uvicorn is fully accepting connections.
                base_url = f"http://127.0.0.1:{port}/api/v1"
                headers = {"Authorization": f"Bearer {api_key}"}
                for _ in range(20):
                    try:
                        requests.get(f"{base_url}", headers=headers, timeout=2)
                        break
                    except requests.ConnectionError:
                        time.sleep(0.25)

                # -- Step 1: Submit a build via POST /api/v1/builds/ --
                archive = _make_build_archive(STANDALONE_BUILD_DIR)
                submit_resp = requests.post(
                    f"{base_url}/builds/",
                    json={
                        "name": "standalone-quickstart-e2e",
                        "build_archive": archive,
                        "space_name": "standalone",
                        "username": "test-user",
                    },
                    headers=headers,
                    timeout=10,
                )
                assert (
                    submit_resp.status_code == 200
                ), f"Build submission failed: {submit_resp.status_code} {submit_resp.text}"
                build_id = submit_resp.json()["build_id"]
                assert build_id, "build_id should be a non-empty string"

                # -- Step 2: Query the build via GET /api/v1/builds/{id} --
                status_resp = requests.get(
                    f"{base_url}/builds/{build_id}",
                    headers=headers,
                    timeout=10,
                )
                assert (
                    status_resp.status_code == 200
                ), f"Build query failed: {status_resp.status_code} {status_resp.text}"
                body = status_resp.json()
                assert (
                    body["build"]["uuid"] == build_id
                ), f"Returned build uuid {body['build']['uuid']} does not match {build_id}"
                assert (
                    body["build"]["name"] == "standalone-quickstart-e2e"
                ), f"Returned build name {body['build']['name']} does not match 'standalone-quickstart-e2e'"
                assert (
                    body["build"]["space_name"] == "standalone"
                ), f"Returned build space_name {body['build']['space_name']} does not match 'standalone'"

                # -- Step 3: Verify spaces_for_user returns the standalone space --
                spaces_resp = requests.get(
                    f"{base_url}/spaces/spaces_for_user",
                    headers=headers,
                    timeout=10,
                )
                assert (
                    spaces_resp.status_code == 200
                ), f"spaces_for_user failed: {spaces_resp.status_code} {spaces_resp.text}"
                spaces = spaces_resp.json()["spaces"]
                space_names = [s["name"] for s in spaces]
                assert (
                    "standalone" in space_names
                ), f"'standalone' space not in spaces_for_user response: {space_names}"

        finally:
            # Stop the server first so its BuildWatcher is stopped (not leaked)
            # before teardown_method discards the db it polls.
            _shutdown_standalone_server(server_holder, server_thread)

    def test_no_apikey_spaces_for_user(self):
        """Verify spaces_for_user works WITHOUT GBSERVER_API_KEY (real standalone scenario).

        This mimics the exact scenario when a user runs:
            gbserver standalone --space-dir test-data/e2e/standalone/standalone-quickstart
        without setting GBSERVER_API_KEY, and gbcli sends an empty Bearer token.
        """
        from gbserver.commands.command_standalone import _run_standalone

        port = _get_free_port()

        # NO GBSERVER_API_KEY — localhost-only auth (the real default scenario).
        env = {
            "GBSERVER_METADATA_STORAGE": "sqlite",
            "GBSERVER_DEFAULT_BUILDRUNNER_TYPE": "thread",
            "GBSERVER_AUTH_MODE": "apikey",
        }
        # Make sure GBSERVER_API_KEY is NOT set.
        env_unset = {k: "" for k in ["GBSERVER_API_KEY"]}

        started_event = threading.Event()

        def on_started():
            started_event.set()

        server_holder: dict = {}

        def run_server():
            _run_standalone(
                port=port,
                space_dir=str(STANDALONE_BUILD_DIR),
                on_started=on_started,
                on_server_created=lambda s: server_holder.__setitem__("server", s),
            )

        server_thread = threading.Thread(
            target=run_server,
            name="standalone-nokey-e2e",
            daemon=True,
        )

        try:
            with patch.dict(os.environ, {**env, **env_unset}):
                server_thread.start()

                assert started_event.wait(
                    timeout=30
                ), "Standalone server did not start within 30 seconds"
                base_url = f"http://127.0.0.1:{port}/api/v1"
                for _ in range(20):
                    try:
                        requests.get(base_url, timeout=2)
                        break
                    except requests.ConnectionError:
                        time.sleep(0.25)

                # 1. Verify space is in SQLite storage directly.
                from gbserver.storage.singleton_storage import get_admin_storage

                admin_storage = get_admin_storage()
                standalone_space = admin_storage.space_storage.get_by_name("standalone")
                assert (
                    standalone_space is not None
                ), "Space 'standalone' not found in SQLite storage!"
                assert standalone_space.name == "standalone"
                assert standalone_space.git_repo_uri.startswith("file://")

                # 2. Verify GET /api/v1/spaces/ returns the space (simple list).
                list_resp = requests.get(f"{base_url}/spaces/", timeout=10)
                assert (
                    list_resp.status_code == 200
                ), f"GET /spaces/ failed: {list_resp.status_code} {list_resp.text}"
                spaces_list = list_resp.json()["spaces"]
                assert any(
                    s["name"] == "standalone" for s in spaces_list
                ), f"'standalone' not in GET /spaces/ response: {spaces_list}"

                # 3. Verify spaces_for_user with EMPTY Bearer token (like gbcli sends).
                spaces_resp = requests.get(
                    f"{base_url}/spaces/spaces_for_user",
                    headers={"Authorization": "Bearer "},
                    timeout=10,
                )
                assert (
                    spaces_resp.status_code == 200
                ), f"spaces_for_user (empty Bearer) failed: {spaces_resp.status_code} {spaces_resp.text}"
                user_spaces = spaces_resp.json()["spaces"]
                user_space_names = [s["name"] for s in user_spaces]
                assert (
                    "standalone" in user_space_names
                ), f"'standalone' not in spaces_for_user response: {user_spaces}"

                # 4. Verify build submission works without API key.
                archive = _make_build_archive(STANDALONE_BUILD_DIR)
                submit_resp = requests.post(
                    f"{base_url}/builds/",
                    json={
                        "name": "hello-nokey",
                        "build_archive": archive,
                        "space_name": "standalone",
                        "username": "standalone",
                    },
                    headers={"Authorization": "Bearer "},
                    timeout=10,
                )
                assert (
                    submit_resp.status_code == 200
                ), f"Build submission (no key) failed: {submit_resp.status_code} {submit_resp.text}"
                build_id = submit_resp.json()["build_id"]
                assert build_id, "build_id should be non-empty"

        finally:
            # Stop the server first so its BuildWatcher is stopped (not leaked)
            # before teardown_method discards the db it polls.
            _shutdown_standalone_server(server_holder, server_thread)
