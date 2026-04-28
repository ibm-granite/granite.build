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

"""
Types related to sections in the step.yaml.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from gbserver.types.config import Config


class StepConfigLsfBsubSection(BaseModel):
    """The config.lsf.bsub section in the step.yaml (also build.yaml)"""

    # 1. very much not managed, user does the launching inside LSF
    jobid: str = ""
    log_path: str = ""  # to be used with jobid, an existing log file to monitor
    # 2. not managed, we do the launching from RIS3
    args: str = ""
    # 3. managed, we also construct the bsub command
    additional_args: str = ""
    queue: str = ""
    jobs_group: str = ""
    job_name: str = ""


class StepConfigLsfSection(BaseModel):
    """The config.lsf section in the step.yaml (also build.yaml)"""

    bsub: StepConfigLsfBsubSection = StepConfigLsfBsubSection()


class StepConfigWorkloadPythonEnvSection(BaseModel):
    """The config.workload.python_env section in the step.yaml (also build.yaml)"""

    env_dirs: List[str] = Field(default_factory=list)
    venv: str = ""
    conda: str = ""


class StepConfigWorkloadSection(BaseModel):
    """The config.workload section in the step.yaml (also build.yaml)"""

    path: str = ""
    args: str = ""
    workspace_dir: str = ""
    output_dir: str = ""
    python_env: StepConfigWorkloadPythonEnvSection = (
        StepConfigWorkloadPythonEnvSection()
    )


class StepConfigSection(BaseModel):
    """The config section in the step.yaml (also build.yaml)"""

    workload: StepConfigWorkloadSection = StepConfigWorkloadSection()
    lsf: StepConfigLsfSection = StepConfigLsfSection()
    retry_enabled_default: Optional[bool] = None
    retry_transparently_default: bool = True


class StepEnvConfig(Config):
    """Base class for config specific to the environment from the step.yaml"""
