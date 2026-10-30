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

"""One job's full content, read from this server's builds when it is a granite.build job.

The index rows are slim: they carry enough to draw the graph, and a ``retrieve``
group naming the build and target run behind a granite.build job. Whatever store
indexed the job (the lineage job table, W&B, an import), its detail is read from
the gb DB through that build, never from an external system, so the UI depends on
nothing but this server. A job with no ``build_id`` that the lineage job table
holds (a lineage event, an import) gets that table's payloads instead.

A job with neither, or whose build or target this server does not have, reports
``detail_available=False`` with a reason, and the caller still gets everything the
index itself recorded.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from gbserver.lineage.attributes import (
    PAYLOAD_EXECUTION_STATS,
    PAYLOAD_INPUT_PARAMS,
    PAYLOAD_OUTPUT_STATS,
    PAYLOAD_SOURCE_CODE,
    origin_id,
    payload_detail,
)
from gbserver.lineage.db_responses import global_job_entry
from gbserver.storage.stored_lineage_row import JobStore
from gbserver.utils.redaction import redact_sensitive

logger = logging.getLogger(__name__)

PAYLOAD_KEYS = (
    PAYLOAD_INPUT_PARAMS,
    PAYLOAD_EXECUTION_STATS,
    PAYLOAD_OUTPUT_STATS,
    PAYLOAD_SOURCE_CODE,
)

# Called with a StoredBuild before its targets and steps are returned; raises when
# the caller may not read it. The API passes ``authorize_build_read_access``: step
# configs are exposed only to those who could read the build's own status.
AuthorizeBuild = Callable[[Any], None]

# Called with a job's listing entry; returns whether the caller may see the job's
# space-gated fields. The API checks space membership off ``job_namespace``, as the
# graph's access filter does.
AuthorizeEntry = Callable[[Dict[str, Any]], bool]


class JobDetailUnavailable(Exception):
    """The store has no detail for this job; the message says why."""


class JobDetailDenied(JobDetailUnavailable):
    """The caller may not read this job's detail; reported as ``detail_denied``."""


def _payload(source: Dict[str, Any]) -> Dict[str, Any]:
    """The four payloads from ``source``, with step configs redacted.

    The same masking as ``api.lineage.get_redacted_job_input_params``: every read
    path masks ``job_input_params`` unconditionally, whatever the writer did.
    """
    payload = {key: source[key] for key in PAYLOAD_KEYS if source.get(key)}
    if payload.get(PAYLOAD_INPUT_PARAMS):
        payload[PAYLOAD_INPUT_PARAMS] = redact_sensitive(payload[PAYLOAD_INPUT_PARAMS])
    return payload


def _fetch_lineage_job(job_id: str, storage: Any) -> Dict[str, Any]:
    """The payloads the lineage job table holds for a job with no build."""
    job = storage.lineage_job_storage.get_job(job_id)
    if job is None:
        raise JobDetailUnavailable("Only the index is recorded")
    return {"detail": _payload(payload_detail(job.attributes))}


def _fetch_target(
    build_id: str,
    target_run_uuid: str,
    storage: Any,
    authorize_build: AuthorizeBuild,
) -> Dict[str, Any]:
    """The target run and its steps, in the shape ``GET /builds/{id}/status`` uses."""
    build = storage.build_storage.get_by_uuid(build_id) if build_id else None
    target = (
        storage.target_storage.get_by_uuid(target_run_uuid) if target_run_uuid else None
    )
    if build is None:
        # No id in the text: the caller has not been authorized for this build.
        raise JobDetailUnavailable("The build is not on this server")
    try:
        authorize_build(build)
    except Exception as e:
        raise JobDetailDenied("Not authorized to read this build") from e
    # The index names both ids; a target of another build is not this job's, so
    # it is not shown under this build's authorization.
    if target is None or target.build_id != build.uuid:
        raise JobDetailUnavailable("The target run is not on this server")

    steps = storage.step_storage.get_by_where({"target_id": target.uuid})
    build_dict = build.model_dump(mode="json")
    build_dict["build_archive"] = ""
    return {
        "build_id": build.uuid,
        "build": build_dict,
        "target": {
            **target.model_dump(mode="json"),
            "steps": [step.model_dump(mode="json") for step in steps],
        },
    }


def fetch_job_detail(
    entry: Dict[str, Any],
    rows: List[Any],
    admin_storage: Any,
    authorize_build: AuthorizeBuild,
    authorize_entry: Optional[AuthorizeEntry] = None,
) -> Dict[str, Any]:
    """Extend a listing entry with the job's content from this server's builds.

    Args:
        entry: the job's listing entry, as ``db_service._job_entry`` builds it.
        rows: the job's index rows; the first one's ``job_store`` and origin ids
            are used (every row of one job is written by one writer).
        admin_storage: the admin storage holding the builds, targets and steps.
        authorize_build: see :data:`AuthorizeBuild`.
        authorize_entry: see :data:`AuthorizeEntry`; ``None`` skips the check.
    """
    first = rows[0]
    try:
        job_store = JobStore(first.job_store)
    except ValueError:
        job_store = JobStore.OTHER
    # ``origin_id`` reads ``retrieve``, falling back to ``origin.ids`` for older
    # rows -- as the graph does, so the node and the drawer agree.
    build_id = origin_id(first.attributes, "build_id")
    target_run_uuid = origin_id(first.attributes, "target_run_uuid")

    # Fail closed: the global fields only, unless the caller passes the space
    # check. The full entry (owner, namespace, tags, origin ids) describes a job
    # beyond a lineage trace; a caller who may read it gets it whether or not the
    # build is still here (an import, a pruned build). The build check releases
    # the build and target only, never these.
    entry_readable = authorize_entry is None or authorize_entry(entry)
    detail: Dict[str, Any] = {
        **global_job_entry(entry),
        "job_store": job_store.value,
        "detail_available": True,
        "detail_error": None,
        "detail_denied": False,
        "target_run_uuid": None,
    }
    fetched: Dict[str, Any] = {}
    build_missing = False
    if not build_id:
        # Not a granite.build job: only the lineage job table may hold more. Its
        # payloads have no build check of their own, so the space check gates them.
        if job_store != JobStore.LINEAGE_JOB:
            detail.update(
                detail_available=False, detail_error="Only the index is recorded"
            )
        elif entry_readable:
            try:
                detail.update(_fetch_lineage_job(entry["job_id"], admin_storage))
            except JobDetailUnavailable as e:
                detail.update(detail_available=False, detail_error=str(e))
            except Exception:  # pylint: disable=broad-except
                logger.exception("Reading lineage job %s failed", entry["job_id"])
                detail.update(
                    detail_available=False, detail_error="Could not read the job"
                )
        else:
            detail.update(detail_available=False)
    else:
        try:
            fetched = _fetch_target(
                build_id, target_run_uuid, admin_storage, authorize_build
            )
        except JobDetailDenied as e:
            # The build's own check is the stricter one: nothing beyond the trace.
            detail.update(
                detail_available=False, detail_error=str(e), detail_denied=True
            )
            return detail
        except JobDetailUnavailable as e:
            detail.update(detail_available=False, detail_error=str(e))
            build_missing = True
        except Exception:  # pylint: disable=broad-except
            logger.exception("Reading the build of job %s failed", entry["job_id"])
            # The exception text can carry SQL internals; it stays in the log.
            detail.update(
                detail_available=False, detail_error="Could not read the build"
            )
    if entry_readable:
        detail.update(entry)
        if not fetched:
            # The target these rows name, as the index recorded it.
            detail.update(
                build_id=build_id or None, target_run_uuid=target_run_uuid or None
            )
    elif build_missing:
        # A caller outside the job's space gets the same answer whether or not the
        # build is still here, so the reply does not tell them it exists.
        detail.update(
            detail_error="Not authorized to read this build", detail_denied=True
        )
    else:
        # Withheld, not missing: say so, or the client shows a bare N/A.
        detail["detail_denied"] = True
    detail.update(fetched)
    return detail
