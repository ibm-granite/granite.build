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
import socket
import threading
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

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


class TestStandaloneRestApiE2E:
    """End-to-end tests for the standalone REST API server."""

    def test_submit_and_query_build(self):
        """Start the standalone server, submit a build, and query its status."""
        from gbserver.commands.command_standalone import _run_standalone
        from gbserver.storage import singleton_storage
        from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

        # Use a unique table prefix to avoid collisions with other tests.
        table_prefix = f"rest_e2e_{random.randint(1, 100000)}_"

        # Reset singleton storage to SQLite with a unique prefix.
        singleton_storage.set_storage_factory(SqliteStorageFactory())
        storage = singleton_storage.set_storage_prefix(table_prefix)

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

        def run_server():
            _run_standalone(
                port=port,
                space_dir=str(STANDALONE_BUILD_DIR),
                on_started=on_started,
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
            # Clean up: delete test tables.
            for store in [
                storage.build_storage,
                storage.target_storage,
                storage.step_storage,
                storage.space_storage,
                storage.artifact_registry,
                storage.event_storage,
            ]:
                try:
                    store.delete_table()
                except Exception:
                    pass

    def test_no_apikey_spaces_for_user(self):
        """Verify spaces_for_user works WITHOUT GBSERVER_API_KEY (real standalone scenario).

        This mimics the exact scenario when a user runs:
            gbserver standalone --space-dir test-data/e2e/standalone/standalone-quickstart
        without setting GBSERVER_API_KEY, and gbcli sends an empty Bearer token.
        """
        from gbserver.commands.command_standalone import _run_standalone
        from gbserver.storage import singleton_storage
        from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

        table_prefix = f"nokey_e2e_{random.randint(1, 100000)}_"

        singleton_storage.set_storage_factory(SqliteStorageFactory())
        storage = singleton_storage.set_storage_prefix(table_prefix)

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

        def run_server():
            _run_standalone(
                port=port,
                space_dir=str(STANDALONE_BUILD_DIR),
                on_started=on_started,
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
            # Clean up: delete test tables.
            for store in [
                storage.build_storage,
                storage.target_storage,
                storage.step_storage,
                storage.space_storage,
                storage.artifact_registry,
                storage.event_storage,
            ]:
                try:
                    store.delete_table()
                except Exception:
                    pass
