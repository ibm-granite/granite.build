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
