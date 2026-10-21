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

"""Shell access to the local Docker SLURM cluster, for tests that inspect what a build
left on its shared filesystem or act on its SLURM jobs."""

import os
import subprocess


def on_cluster(script: str) -> str:
    """Run a shell script on the Docker SLURM login node; return its stdout.

    Same SSH endpoint and key as the skip gate
    (``integration.environment.test_skypilot_slurm_e2e._slurm_cluster_reachable``).
    """
    proc = subprocess.run(
        [
            "ssh",
            "-i",
            os.path.expanduser("~/.ssh/slurm_docker_key"),
            "-p",
            os.environ.get("SLURM_SSH_PORT", "2222"),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            f"root@{os.environ.get('SLURM_SSH_HOST', '127.0.0.1')}",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"cluster command failed: {proc.stderr}"
    return proc.stdout
