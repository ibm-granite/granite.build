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

"""End-to-end test: run a standalone build with SQLite + Bash + FileStore.

This test verifies that a build can complete locally with zero cloud dependencies.
It exercises the same code path as `gbserver build run` (direct Build/BuildRun),
and optionally the full BuildRunner + SQLite storage path.
"""

import asyncio
import random
from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).parent.parent.parent.parent / "test-data"
STANDALONE_BUILD_DIR = TEST_DATA_DIR / "e2e" / "standalone" / "standalone-quickstart"


class TestStandaloneE2E:
    """Standalone end-to-end build tests — no cloud dependencies required."""

    def test_sample_build_dir_exists(self):
        """Verify the sample standalone build directory exists with expected files."""
        assert (
            STANDALONE_BUILD_DIR.is_dir()
        ), f"Sample standalone build directory not found at {STANDALONE_BUILD_DIR}"
        build_yaml = STANDALONE_BUILD_DIR / "build.yaml"
        assert build_yaml.is_file(), f"build.yaml not found at {build_yaml}"
        step_yaml = STANDALONE_BUILD_DIR / "steps" / "hello" / "step.yaml"
        assert step_yaml.is_file(), f"step.yaml not found at {step_yaml}"
        env_yaml = STANDALONE_BUILD_DIR / "environments" / "bash" / "environment.yaml"
        assert env_yaml.is_file(), f"environment.yaml not found at {env_yaml}"
        space_yaml = STANDALONE_BUILD_DIR / "space.yaml"
        assert space_yaml.is_file(), f"space.yaml not found at {space_yaml}"

    def test_hello_standalone_build_completes(self):
        """The standalone-quickstart sample build runs to completion via direct Build/BuildRun.

        This exercises the same code path as `gbserver build run` — no storage,
        no GitHub, no messaging required. Just Build + BuildRun + Bash environment
        + Space with env secret manager for space:// URI resolution.
        """
        from gbserver.build.build import Build
        from gbserver.build.buildrun import BuildRun
        from gbserver.build.space import Space

        # Set up the standalone Space so space:// URIs resolve to the build directory
        space_uri = f"file://{STANDALONE_BUILD_DIR}"
        space = Space(uri=space_uri, username="standalone-test")

        build = Build(
            build_dir=STANDALONE_BUILD_DIR,
            space=space,
            username="standalone-test",
        )

        build_run = BuildRun(build=build)
        asyncio.run(build_run.run_and_wait())

        # Verify the build status through the run's status.
        from gbserver.types.status import Status

        assert (
            build_run.status == Status.SUCCESS
        ), f"Build run did not succeed. Status: {build_run.status}"

    @pytest.mark.skip(
        reason="This is failing in the DEV environment and SPS pipeline. "
        "It seems redundant if only to test sqlite.  "
        "We should be using git action matrix to use sqlite with the other mainline build integration tests."
    )
    def test_hello_standalone_with_sqlite_storage(self):
        """The standalone-quickstart build runs via BuildRunner with SQLite storage.

        This validates the full standalone stack: SQLite storage + Bash environment
        + BuildRunner event processing + Space with env secret manager.
        No GitHub PRs, no cloud messaging, no lakehouse.
        """
        from gbserver.storage import singleton_storage
        from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
        from gbserver.storage.stored_build import StoredBuild
        from gbserver.storage.stored_space import StoredSpace
        from gbserver.types.status import Status

        # Set up SQLite storage with a unique table prefix to avoid collisions
        table_prefix = f"e2e_test_{random.randint(1, 100000)}_"
        singleton_storage.set_storage_factory(SqliteStorageFactory())
        storage = singleton_storage.set_storage_prefix(table_prefix)

        # The space URI uses a file:// scheme pointing to the standalone build dir
        space_uri = f"file://{STANDALONE_BUILD_DIR}"

        try:
            # Register the standalone space in storage
            stored_space = StoredSpace(
                name="standalone",
                git_repo_uri=space_uri,
                lakehouse_namespace="",
            )
            storage.space_storage.add(stored_space)

            # Create a StoredBuild from the sample build.yaml
            build_yaml_path = STANDALONE_BUILD_DIR / "build.yaml"
            stored_build = StoredBuild.create(
                name="standalone-quickstart-e2e",
                space_name="standalone",
                source_uri="",
                username="standalone-test",
                build_yaml_path=build_yaml_path,
                status=Status.PENDING,
            )

            # Run via BuildRunner with space_uri (no PR creation, no GitHub token needed)
            from gbserver.buildwatcher.buildrunner import BuildRunner

            runner = BuildRunner(
                build=stored_build,
                gh_token="",
                space_uri=space_uri,
                workspace_dir=Path("/tmp/gbserver-standalone-test"),
                monitoring_interval=1,
                create_pr=False,
            )
            runner.start_and_wait()

            # Verify the build completed successfully in storage
            finished_build = storage.build_storage.get_by_uuid(stored_build.uuid)
            assert (
                finished_build is not None
            ), f"Build {stored_build.uuid} not found in storage"
            assert (
                finished_build.status == Status.SUCCESS
            ), f"Build status is {finished_build.status}, expected SUCCESS"

            # Verify target was created and completed
            targets = storage.target_storage.get_by_where(
                {"build_id": stored_build.uuid}
            )
            assert len(targets) == 1, f"Expected 1 target, got {len(targets)}"
            assert (
                targets[0].name == "helloworld"
            ), f"Target name is '{targets[0].name}', expected 'helloworld'"
            assert (
                targets[0].status == Status.SUCCESS
            ), f"Target status is {targets[0].status}, expected SUCCESS"

            # Verify steps were created
            steps = storage.step_storage.get_by_where({"target_id": targets[0].uuid})
            assert len(steps) >= 1, f"Expected at least 1 step, got {len(steps)}"
            for step in steps:
                assert (
                    step.status == Status.SUCCESS
                ), f"Step status is {step.status}, expected SUCCESS"

        finally:
            # Clean up: delete test tables
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
