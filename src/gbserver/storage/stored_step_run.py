from datetime import datetime
from typing import Optional, Self

from gbserver.storage.storage import BaseStoredItem
from gbserver.types.status import Status


class StoredStepRun(BaseStoredItem):
    # Required initializations
    build_id: str
    target_id: str
    definition_uri: str

    # Defaulting initializations
    status: Status = Status.PENDING
    status_msg: str = ""
    config: dict = {}
    config_dir: str = ""
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def __init__(self: Self, **kwargs):
        super().__init__(**kwargs)
