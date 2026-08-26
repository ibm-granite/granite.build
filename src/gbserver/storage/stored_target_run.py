from datetime import datetime, timezone
from typing import Iterable, Optional, Self

from pydantic import Field

from gbserver.storage.storage import BaseStoredItem
from gbserver.types.status import Status

# Timezone-aware epoch used to order runs by a finished_at that may be unset or
# offset-naive. get_time() stamps finished_at as tz-aware, but SQLite's
# DateTime(timezone=True) column can read back offset-naive, so the key must
# normalize both or max()-style comparison raises "can't compare offset-naive
# and offset-aware datetimes".
_SORT_EPOCH = datetime.fromtimestamp(0, tz=timezone.utc)


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


def _finished_key(target: "StoredTargetRun") -> datetime:
    """Sort key ordering targets by when they finished, newest greatest.

    Args:
        target: the target run to key.

    Returns:
        The target's ``finished_at`` coerced to timezone-aware; ``_SORT_EPOCH``
        when it is unset, so targets with no finished timestamp sort oldest and
        never raise on comparison against tz-aware timestamps.
    """
    ts = target.finished_at
    if ts is None:
        return _SORT_EPOCH
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def latest_finished_target(
    targets: Iterable["StoredTargetRun"],
) -> Optional["StoredTargetRun"]:
    """Return the target that finished most recently, or None if there are none.

    ``get_by_where`` returns rows in an undefined order, so callers narrowing to a
    set that can hold more than one run of the same target (e.g. the SUCCESS runs
    sharing a ``target_hash`` within one in-place-retried build) must not trust
    the first row — the stale/older run would win. Ordering by ``finished_at``
    keeps the newest, whose output artifacts are the ones to reuse.

    Args:
        targets: ``StoredTargetRun``s, in any order.

    Returns:
        The target with the greatest ``finished_at`` (unset sorts oldest), or
        ``None`` when ``targets`` is empty.
    """
    targets = list(targets)
    if not targets:
        return None
    return max(targets, key=_finished_key)


def latest_success_per_target(
    targets: Iterable["StoredTargetRun"],
) -> list["StoredTargetRun"]:
    """Keep only the latest-finished SUCCESS run per target name.

    In-place build retry reuses one build id, so a target can accrue more than
    one SUCCESS run within a build: a prior success whose output artifacts never
    fully registered is re-run, and a reuse-disabled build re-runs every target.
    Only the latest such run is authoritative — recording or listing the earlier
    ones double-counts lineage and shows the target twice on the status page.

    Callers must pass **only SUCCESS targets**; FAILED history (which is never
    deduplicated) is the caller's responsibility to preserve separately.

    Args:
        targets: SUCCESS ``StoredTargetRun``s, in any order.

    Returns:
        One target per name — the one with the greatest ``finished_at`` — in
        first-appearance order of the kept names (so a newest-first input stays
        newest-first).
    """
    best: dict[str, "StoredTargetRun"] = {}
    for target in targets:
        current = best.get(target.name)
        if current is None or _finished_key(target) > _finished_key(current):
            best[target.name] = target
    return list(best.values())
