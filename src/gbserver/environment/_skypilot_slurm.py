"""SLURM-specific helpers for SkyPilot environments.

Read from what SkyPilot already returns; no extra login-node round trip.
"""

import re
from typing import Any, Optional

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# SkyPilot's SLURM instance ids are ``job<SLURM job id>-<node>``.
_SLURM_INSTANCE_ID_RE = re.compile(r"^job(\d+)-")


def slurm_job_id_from_handle(handle: Any) -> Optional[str]:
    """The SLURM job id behind a launched SkyPilot cluster, or None.

    Reads the head instance's ``job_id`` tag from ``cached_cluster_info``,
    falling back to its ``job<id>-<node>`` instance id. Only all-digit ids are
    accepted. Never raises.

    Args:
        handle: The handle returned by ``sky.launch``'s ``stream_and_get``.

    Returns:
        The SLURM job id, or None if the handle doesn't expose one.
    """
    try:
        info = getattr(handle, "cached_cluster_info", None)
        if info is None:
            return None
        tags = getattr(info.get_head_instance(), "tags", None)
        job_id = str(tags.get("job_id", "")) if isinstance(tags, dict) else ""
        if job_id.isdigit():
            return job_id
        head_id = getattr(info, "head_instance_id", None)
        match = (
            _SLURM_INSTANCE_ID_RE.match(head_id) if isinstance(head_id, str) else None
        )
        return match.group(1) if match else None
    except Exception as e:  # pylint: disable=broad-except
        logger.debug("Could not read the SLURM job id from the handle: %s", e)
        return None
