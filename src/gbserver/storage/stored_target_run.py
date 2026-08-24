from datetime import datetime
from typing import Optional, Self

from pydantic import Field

from gbserver.storage.storage import BaseStoredItem
from gbserver.types.status import Status


class StoredTargetRun(BaseStoredItem):
    # Required initializations
    build_id: str
    environment_uri: str

    # Defaulting initializations
    name: str = ""
    status: Status = Status.PENDING
    status_msg: str = ""
    input_artifacts: dict[str, str] = Field(default_factory=dict)
    """The name of the input targets mapped to a single artifact uuid"""

    output_artifacts: dict[str, list[str]] = Field(default_factory=dict)
    """The name of the output targets mapped to a list of artifact uuids (multiple for checkpoints)"""

    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    target_hash: str = ""
    """SHA-256 hex of the target definition. Set only on successful runs."""

    retry_of_target_id: str = ""
    """UUID of the prior FAILED StoredTargetRun in the same build that this run
    retried; empty if this run is not a retry."""

    def __init__(self: Self, **kwargs):
        super().__init__(**kwargs)
