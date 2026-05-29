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

# from .artifact import ArtifactStoreType, ArtifactType
# from .resources import ResourceSpec, ResourceTypeimport os
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Self, Union

from gbserver.buildrunner.abstractbuildrunner import AbstractBuildRunner
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.constants import (
    DEFAULT_GH_API_ENDPOINT,
    DEFAULT_ROOT_WORKSPACE_DIR,
    GBSERVER_GITHUB_TOKEN,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class QueuedOutput:
    def __init__(self, std):
        self.q = queue.Queue()
        self.std = std
        self.stop_requested = False
        self.thread = None

    def _enqueue_line(self):
        for line in iter(self.std.readline, b""):
            self.q.put(line)
            if self.stop_requested:
                break
        self.std.close()

    def start(self):
        self.thread = threading.Thread(target=self._enqueue_line, args=())
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        if self.thread is None:
            return
        self.stop_requested = True
        self.thread.join()
        self.thread = None

    def show_lines(self, file=sys.stdout) -> None:
        lines = self.get_lines()
        for line in lines:
            # print(line, file=file, flush=True)
            logger.log(level=0, msg=line)

    def get_lines(self) -> list[str]:
        lines = []
        while True:
            try:
                line = self.q.get_nowait()
                lines.append(line.strip())
            except queue.Empty:
                break
        return lines


class BuildRunnerProcess(AbstractBuildRunner):
    """
    This implementation of AbstractBuildRunner starts an inmemory BuildRunner in a new process using the gbserver CLI.

    NOTE: this class is really only intended as a proof-of-concept for the AbstractBuildRunner design and will likley
    not be used by the BuildWatcher in the long run.  Instead, the BuildRunnerJob will be used once implemented.
    """

    def __init__(
        self: Self,
        build: StoredBuild,
        gh_token: str = GBSERVER_GITHUB_TOKEN,
        workspace_dir: Union[str, Path] = DEFAULT_ROOT_WORKSPACE_DIR,
        monitoring_interval: int = 5,
        gh_api_endpoint: str = DEFAULT_GH_API_ENDPOINT,
    ) -> None:
        super().__init__(
            build=build,
            gh_token=gh_token,
            workspace_dir=workspace_dir,
            monitoring_interval=monitoring_interval,
            gh_api_endpoint=gh_api_endpoint,
        )
        self.is_running = False
        self.is_stop_requested = True

    def stop(self: Self) -> None:
        """
        Stop the build that was started using start_and_wait().
        Upon returning, this must also cause the call to start_and_wait() to return.
        """
        self.is_stop_requested = True
        while self.is_running:
            time.sleep(self.monitoring_interval)

    def start_and_wait(self: Self) -> None:
        """
        Start job/pod running the BuildRunner using the gbserver build-runner CLI.
        The following should be passed to the CLI either as command line options or env vars :
            1) build id (stored in build storage.) (cli option)
            2) gh_token (cli option or GBSERVER_GITHUB_TOKEN env var)
            3) workspace_dir (cli option)
            4) monitoring interval (cli option)
            5) gh_api_endpoint (cli option)
        Returns after the build has completed/failed/cancelled or stop() has been called from another thread.
        """
        item = self.storage.build_storage.get_by_uuid(self.stored_build.uuid)
        if item is None:
            self.storage.build_storage.add(self.stored_build)
        cli = [
            "gbserver",
            "--gb-admin-table-prefix",
            f"{self.storage.table_name_prefix}",  # Needed to support testing which uses prefixes
            "build-runner",
            "--build-id",
            f"{self.stored_build.uuid}",
            "--gh-token",
            f"{self.gh_token}",
            "--workspace-dir",
            f"{str(self.workspace_dir)}",
            "--monitoring-interval",
            f"{self.monitoring_interval}",
            "--gh-api-endpoint",
            f"{self.gh_api_endpoint}",
        ]
        logger.info(f"Starting build-runner process: {cli}")
        self.is_stop_requested = False
        env = os.environ
        process = subprocess.Popen(
            args=cli, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        assert isinstance(process, subprocess.Popen)
        self.is_running = True
        stdout = QueuedOutput(process.stdout)
        stderr = QueuedOutput(process.stderr)
        try:
            stdout.start()
            stderr.start()
            while not self.is_stop_requested:
                stdout.show_lines(file=sys.stdout)
                stderr.show_lines(
                    file=sys.stderr
                )  # TODO: All output from the subprocess seems to be coming to stderr, none to stdout.
                try:
                    # process.wait(self.monitoring_interval)  # TODO this causes the i/o to come out in chunks.
                    process.wait(0.1)
                except subprocess.TimeoutExpired:
                    pass
        except Exception as e:
            logger.error(f"Got exception {e}")
            raise e
        finally:
            stdout.stop()
            stderr.stop()
            self.is_running = False
