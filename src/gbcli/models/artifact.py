# Copyright Granite.Build Authors
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

"""Artifact metadata models."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ArtifactMetadata:
    """Metadata about an uploaded or registered artifact.

    Attributes:
        uri: Canonical URI of the artifact (e.g., hf://model/organization/repo-name).
        artifact_type: Type of artifact: model, dataset, space, or bucket.
        registry: Registry system where artifact is stored.
        local_path: Original local path (if uploaded from local).
        repo_id: Repository ID in the registry format (organization/repo-name).
        artifact_hash: Optional hash/checksum of the artifact for verification.
        is_private: Whether the artifact is in a private repository.
    """

    uri: str
    artifact_type: Literal["model", "dataset", "space", "bucket"]
    registry: str
    repo_id: str
    local_path: str | None = None
    artifact_hash: str | None = None
    is_private: bool = False

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for logging/storage.

        Returns:
            Dictionary representation of metadata.
        """
        return {
            "uri": self.uri,
            "artifact_type": self.artifact_type,
            "registry": self.registry,
            "repo_id": self.repo_id,
            "local_path": self.local_path,
            "artifact_hash": self.artifact_hash,
            "is_private": self.is_private,
        }
