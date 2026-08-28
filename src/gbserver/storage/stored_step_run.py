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
    # Runtime key/values produced by the step itself (e.g. a resolved git commit
    # SHA), pushed via the GB_STEP_METADATA_KEY/VALUE stdout hook and merged by
    # the buildrunner. Kept separate from `config` (the rendered build.yaml input)
    # so step-generated data never mutates the declared configuration. Serialized
    # into the row's JSON blob like `config`, so it needs no column/migration and
    # old rows deserialize to {}.
    metadata: dict = {}
    config_dir: str = ""
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def __init__(self: Self, **kwargs):
        super().__init__(**kwargs)
