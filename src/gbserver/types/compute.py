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
Types used for configuring compute
"""

from typing import Any, Self

from pydantic import ConfigDict, Field

from gbserver.types.config import Config


class ComputeConfig(Config):
    """The compute resources required to run the job."""

    # https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.populate_by_name
    # https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.validate_by_name
    # https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.validate_by_alias
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    numNodes: int = Field(default=1, alias="num_nodes")
    numCpusPerNode: int = Field(default=0, alias="num_cpus_per_node")
    numGpusPerNode: int = Field(default=8, alias="num_gpus_per_node")
    numRoceGdrPerNode: int = Field(default=0, alias="num_roce_gdr_per_node")
    gpuModel: str = Field(default="", alias="gpu_model")
    totalMemoryPerNode: str = Field(default="", alias="total_memory_per_node")
    totalEphemeralStoragePerNode: str = Field(default="", alias="total_ephemeral_storage_per_node")

    def model_post_init(self: Self, context: Any, /) -> None:
        assert self.numNodes > 0
        assert self.numGpusPerNode >= 0
        if self.numCpusPerNode <= 0:
            if self.numGpusPerNode == 0:
                self.numCpusPerNode = 1
            else:
                self.numCpusPerNode = self.numGpusPerNode * 8
        if self.totalMemoryPerNode == "":
            self.totalMemoryPerNode = f"{self.numGpusPerNode * 64}G"
