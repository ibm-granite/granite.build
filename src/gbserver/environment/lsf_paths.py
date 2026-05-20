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

"""Pure path helpers for LSF remote asset directories.

Kept separate from gbserver.environment.lsf.Lsf so both the build runtime
and the REST API can compute the same paths without instantiating the
(stateful, SSH-aware) Lsf environment.
"""

from pathlib import Path, PurePosixPath
from typing import Optional


def build_launch_sub_dir(launch_id: str) -> str:
    """Directory name under a step-run dir that holds a single launch's outputs."""
    return "launch-" + launch_id


def build_remote_root_dir(
    workspace_remote_dir: Optional[str], build_id: str
) -> PurePosixPath:
    """Absolute POSIX path of a build's remote root directory."""
    rel = Path(f"llm-build-{build_id}")
    if workspace_remote_dir:
        return PurePosixPath(workspace_remote_dir) / rel.as_posix()
    return PurePosixPath(rel.as_posix())


def build_workspace_sub_dir(
    build_id: str,
    target_name: str,
    targetrun_id: str,
    step_name: str,
    targetsteprun_id: str,
    launch_id: str,
) -> Path:
    """Path of the step-launch sub-directory relative to the LSF workspace root."""
    return (
        Path(f"llm-build-{build_id}")
        / f"target-{target_name}"
        / f"target-run-{targetrun_id}"
        / f"step-{step_name}"
        / f"step-run-{targetsteprun_id}"
        / build_launch_sub_dir(launch_id)
    )


def build_remote_asset_dir(
    workspace_remote_dir: Optional[str],
    build_id: str,
    target_name: str,
    targetrun_id: str,
    step_name: str,
    targetsteprun_id: str,
    launch_id: str,
) -> PurePosixPath:
    """Absolute POSIX path of a step-launch's remote asset directory.

    Mirrors Lsf._get_final_asset_dir for the ssh/remote case — i.e. when
    `workspace_remote_dir` is set. For local-only Lsf runs use the Lsf
    method directly.
    """
    sub = build_workspace_sub_dir(
        build_id=build_id,
        target_name=target_name,
        targetrun_id=targetrun_id,
        step_name=step_name,
        targetsteprun_id=targetsteprun_id,
        launch_id=launch_id,
    )
    if workspace_remote_dir:
        return PurePosixPath(workspace_remote_dir) / sub.as_posix()
    return PurePosixPath(sub.as_posix())
