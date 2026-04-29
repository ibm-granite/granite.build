"""Lineage metadata models."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class LineageNode:
    """Represents a node in the lineage graph (input or output artifact).

    Attributes:
        uri: Canonical URI of the artifact.
        metadata: Additional metadata about the artifact.
    """

    uri: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of lineage node.
        """
        return asdict(self)


@dataclass
class LineageRecord:
    """Structured lineage record persisted.

    Attributes:
        run_name: Optional MLflow run name.
        parent_run_name: Optional parent MLflow run name for nested runs.
        parameters: Key-value pairs logged as MLflow parameters.
        tags: Key-value pairs logged as MLflow tags.
        input_artifacts: List of input artifacts consumed by the run.
        output_artifacts: List of output artifacts produced by the run.
        timestamp: ISO 8601 timestamp (UTC) when the lineage record was created.
    """

    run_name: str | None = None
    parent_run_name: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    input_artifacts: list[LineageNode] = field(default_factory=list)
    output_artifacts: list[LineageNode] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of lineage record.
        """
        return {
            "run_name": self.run_name,
            "parent_run_name": self.parent_run_name,
            "parameters": self.parameters,
            "tags": self.tags,
            "input_artifacts": [node.to_dict() for node in self.input_artifacts],
            "output_artifacts": [node.to_dict() for node in self.output_artifacts],
            "timestamp": self.timestamp,
        }
